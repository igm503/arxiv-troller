"""Authenticated JSON API used by Heedless Backbones.

Every request goes to /api/ingestion/ with an "action". Reads are GET with query parameters; writes are
POST with a JSON body. The API uses the same account, tags and search functions as the website.

GET  tags                                   the user's tags
GET  tag      tag, [cursor]                 papers in a tag, by arXiv ID
GET  papers   since, [cursor]               papers created or updated since a time
GET  search   type, q | paper | tag,        the site's search: type is keyword, title, paper (similar to one
              [since | date_filter],        paper, by arXiv ID) or tag (similar to a tag's papers). since is an
              [category], [cursor]          exact time; date_filter is a site preset (default 1week)
GET  similar  tag, since, [cursor]          each tagged paper's 20 nearest papers created since a time,
                                            five tagged papers per page
POST bulk_add     tag, arxiv_ids            add up to 200 papers to a tag, creating it if needed
POST bulk_remove  tag, arxiv_ids            remove up to 200 papers from a tag
POST copy_tag     source, target            add every paper in one tag to another

Responses are {"ok": true, ...}; lists include "next_cursor", which is null on the last page.
Errors are {"ok": false, "error": ...} with status 400 (bad input), 401 (not logged in) or 405.
"""
import base64
import json
from datetime import datetime

from django.contrib.postgres.search import SearchQuery
from django.db import transaction
from django.db.models import Q
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from .models import Paper, Tag, TaggedPaper
from .views import (
    MAX_RESULTS,
    get_date_cutoff,
    get_similar_embeddings,
    keyword_search,
    paper_search,
    parse_search_query,
    tag_search,
    title_search,
)

PAGE_SIZE = 100
MAX_IDS = 200
SEARCHES = {"keyword": keyword_search, "title": title_search, "paper": paper_search, "tag": tag_search}
READS = {"tags", "tag", "papers", "search", "similar"}
WRITES = {"bulk_add", "bulk_remove", "copy_tag"}


def serialize(paper):
    return {"arxiv_id": paper.arxiv_id, "title": paper.title, "abstract": paper.abstract,
            "created": paper.created.isoformat(), "updated": (paper.updated or paper.created).isoformat(),
            "categories": paper.categories}


def response(papers, next_cursor=None, **extra):
    return JsonResponse({"ok": True, "papers": [serialize(paper) for paper in papers],
                         "next_cursor": next_cursor, **extra})


def error(message, status=400):
    return JsonResponse({"ok": False, "error": message}, status=status)


def parse_since(value):
    since = datetime.fromisoformat(value)
    if timezone.is_naive(since):
        raise ValueError("since must include a timezone")
    return since


def offset_page(papers, cursor):
    cursor = int(cursor or 0)
    if cursor < 0:
        raise ValueError("Invalid cursor")
    page = list(papers[cursor:cursor + PAGE_SIZE + 1])
    return response(page[:PAGE_SIZE], cursor + PAGE_SIZE if len(page) > PAGE_SIZE else None)


@require_http_methods(["GET", "POST"])
def api(request):
    if not request.user.is_authenticated:
        return error("Authentication required", 401)
    try:
        params = json.loads(request.body) if request.method == "POST" else request.GET
        if not hasattr(params, "get"):
            raise ValueError("Expected an object")
        action = params.get("action")
        if action in WRITES:
            if request.method != "POST":
                return error("POST required", 405)
            return update_tag(request.user, action, params)
        if action not in READS:
            raise ValueError("Unknown action")
        if request.method != "GET":
            return error("GET required", 405)
        if action == "tags":
            return JsonResponse({"ok": True, "tags": list(Tag.objects.filter(user=request.user).values("id", "name"))})
        if action == "tag":
            tag = Tag.objects.get(user=request.user, name=params["tag"])
            return offset_page(Paper.objects.filter(taggedpaper__tag=tag).order_by("arxiv_id"), params.get("cursor"))
        if action == "papers":
            since = parse_since(params["since"])
            papers = Paper.objects.filter(Q(created__gte=since) | Q(updated__gte=since)).order_by("-created", "-pk")
            return offset_page(papers, params.get("cursor"))
        if action == "search":
            return search(request.user, params)
        return similar(request.user, params)
    except (ValueError, TypeError, KeyError, Tag.DoesNotExist) as exc:
        return error(str(exc))


def encode_cursor(ids):
    return base64.urlsafe_b64encode(",".join(map(str, sorted(ids))).encode()).decode()


def decode_cursor(cursor):
    if not cursor:
        return set()
    return {int(pid) for pid in base64.b64decode(cursor, altchars=b"-_", validate=True).decode().split(",") if pid}


def search(user, params):
    """Run one of the site's searches; pages like the site's "load more", by excluding papers already returned."""
    kind = params.get("type")
    if kind not in SEARCHES:
        raise ValueError("type must be keyword, title, paper or tag")
    if params.get("since") and params.get("date_filter"):
        raise ValueError("Pass since or date_filter, not both")
    if params.get("since"):
        since = parse_since(params["since"])
    else:
        date_filter = params.get("date_filter") or "1week"
        since = get_date_cutoff(date_filter)
        if since is None and date_filter != "all":
            raise ValueError("Unknown date_filter")
    seen = decode_cursor(params.get("cursor"))
    context = {
        "query": "",
        "title_query": None,
        "single_paper_id": None,
        "parsed_tag_for_search": None,
        "current_tag": None,
        "since": since,
        "category_filter": params.get("category", ""),
        "exclude_ids": set(seen),
    }
    if kind in ("keyword", "title"):
        query = params.get("q", "").strip()
        if not query or (kind == "keyword" and not parse_search_query(query)):
            raise ValueError("Search query required")
        context["query" if kind == "keyword" else "title_query"] = query
    elif kind == "paper":
        try:
            context["single_paper_id"] = str(Paper.objects.get(arxiv_id=params["paper"]).id)
        except Paper.DoesNotExist:
            raise ValueError(f"Unknown paper {params['paper']}")
    else:
        context["parsed_tag_for_search"] = Tag.objects.get(user=user, name=params["tag"])
    papers = list(SEARCHES[kind](context)[0])
    seen |= {paper.id for paper in papers}
    more = papers and len(seen) < MAX_RESULTS
    return response(
        papers,
        encode_cursor(seen) if more else None,
        search={"type": kind, "since": since and since.isoformat(), "category": context["category_filter"]},
    )


def similar(user, params):
    """For five tagged papers per page, their 20 nearest papers created since a time (tagged papers excluded)."""
    tag = Tag.objects.get(user=user, name=params["tag"])
    since = parse_since(params["since"])
    cursor = int(params.get("cursor", 0))
    if cursor < 0:
        raise ValueError("Invalid cursor")
    seeds = list(Paper.objects.filter(taggedpaper__tag=tag).order_by("arxiv_id")[cursor:cursor + 6])
    tagged = set(TaggedPaper.objects.filter(tag=tag).values_list("paper_id", flat=True))
    candidates = Paper.objects.filter(created__gte=since).exclude(id__in=tagged)
    filters = {"cutoff": since, "category": "", "excluded": tagged}
    found = {}
    for seed in seeds[:5]:
        for paper in get_similar_embeddings(seed, candidates, 20, filters):
            found[paper.pk] = paper
    return response(found.values(), cursor + 5 if len(seeds) > 5 else None)


def arxiv_ids(params):
    ids = params["arxiv_ids"]
    if not isinstance(ids, list) or len(ids) > MAX_IDS or not all(isinstance(value, str) for value in ids):
        raise ValueError(f"Expected at most {MAX_IDS} arXiv IDs")
    papers = Paper.objects.filter(arxiv_id__in=ids)
    return papers, sorted(set(ids) - set(papers.values_list("arxiv_id", flat=True)))


@transaction.atomic
def update_tag(user, action, params):
    missing = []
    if action == "bulk_remove":
        tag = Tag.objects.get(user=user, name=params["tag"])
        papers, missing = arxiv_ids(params)
        TaggedPaper.objects.filter(tag=tag, paper__in=papers).delete()
    else:
        if action == "copy_tag":
            source = Tag.objects.get(user=user, name=params["source"])
            target_name = params["target"]
            if source.name == target_name:
                raise ValueError("Source and target tags must differ")
            papers = Paper.objects.filter(taggedpaper__tag=source)
        else:
            target_name = params["tag"]
            papers, missing = arxiv_ids(params)
        if not isinstance(target_name, str) or not 1 <= len(target_name.strip()) <= 100:
            raise ValueError("Invalid tag name")
        tag, _ = Tag.objects.get_or_create(user=user, name=target_name)
        TaggedPaper.objects.bulk_create([TaggedPaper(tag=tag, paper=paper) for paper in papers], ignore_conflicts=True)
    return JsonResponse({"ok": True, "tag": tag.name, "missing": missing,
                         "count": TaggedPaper.objects.filter(tag=tag).count()})
