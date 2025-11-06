import json
import re
from datetime import timedelta
import random
import time

from django.contrib.postgres.search import SearchQuery, SearchRank
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.template.loader import render_to_string
from django.utils import timezone
from pgvector.django import L2Distance, HammingDistance
from django.db import connection

from .models import (
    Paper,
    Tag,
    TaggedPaper,
    EmbeddingGeminiHalf3072,
    EmbeddingGeminiHalf512,
    EmbeddingVoyageHalf2048,
    EmbeddingVoyageBit2048,
    EmbeddingVoyageHalf256,
)

EMBEDDING_MODEL = EmbeddingVoyageBit2048
if "Bit" in EMBEDDING_MODEL.__name__:
    DISTANCE_FUNCTION = HammingDistance
else:
    DISTANCE_FUNCTION = L2Distance
RESULTS_PER_PAGE = 20
MAX_RESULTS = 400


def process_latex_commands(text):
    # Text formatting commands
    text = re.sub(r"\\textbf\{([^}]+)\}", r"<strong>\1</strong>", text)
    text = re.sub(r"\\textit\{([^}]+)\}", r"<em>\1</em>", text)
    text = re.sub(r"\\emph\{([^}]+)\}", r"<em>\1</em>", text)
    text = re.sub(r"\\texttt\{([^}]+)\}", r"<code>\1</code>", text)
    text = re.sub(r"\\underline\{([^}]+)\}", r"<u>\1</u>", text)

    # URLs and hrefs
    text = re.sub(r"\\url\{([^}]+)\}", r'<a href="\1" target="_blank">\1</a>', text)
    text = re.sub(r"\\href\{([^}]+)\}\{([^}]+)\}", r'<a href="\1" target="_blank">\2</a>', text)
    text = re.sub(
        r'(?<!href=")(?<!">)(https?://[^\s<>"]+?)(\.)?(?=\s|$)',
        lambda m: f'<a href="{m.group(1)}" target="_blank">{m.group(1)}</a>{m.group(2) if m.group(2) else ""}',
        text,
    )

    # Special characters and escapes
    text = re.sub(r"\\%", "%", text)
    text = re.sub(r"\\&", "&", text)
    text = re.sub(r"\\\$", "$", text)
    text = re.sub(r"\\#", "#", text)
    text = re.sub(r"\\_", "_", text)
    text = re.sub(r"\\\{", "{", text)
    text = re.sub(r"\\\}", "}", text)
    text = re.sub(r"\\textbackslash(?:\{\})?", r"\\", text)
    text = re.sub(r"\\~", "~", text)
    text = re.sub(r"\\\^", "^", text)

    # LaTeX quotes
    text = re.sub(r"``", '"', text)
    text = re.sub(r"''", '"', text)
    text = re.sub(
        r"`",
        """, text)
    text = re.sub(r"'", """,
        text,
    )

    # Common spacing commands
    text = re.sub(r"\\,", " ", text)
    text = re.sub(r"~", "&nbsp;", text)
    text = re.sub(r"\\\\", "<br>", text)

    return text


def get_date_cutoff(date_filter):
    """Convert date filter string to datetime cutoff"""
    if date_filter == "all":
        return None

    now = timezone.now()
    date_cutoffs = {
        "1day": now - timedelta(days=1),
        "3day": now - timedelta(days=3),
        "1week": now - timedelta(days=7),
        "1month": now - timedelta(days=30),
        "3months": now - timedelta(days=90),
        "6months": now - timedelta(days=180),
        "1year": now - timedelta(days=365),
        "2years": now - timedelta(days=730),
    }
    return date_cutoffs.get(date_filter)


def search(request):
    """Unified view for all search types: keyword, tag similarity, single paper similarity"""

    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

    if is_ajax and request.method == "POST":
        data = json.loads(request.body)
        exclude_ids = set(data.get("exclude_ids", []))
        query_params = data.get("query_params", {})

    else:
        query_params = request.GET
        exclude_ids = set()

    # Parse query for special syntax: "tag: X" or "paper: X"
    raw_query = query_params.get("q", "").strip()
    parsed_single_paper_id = None
    parsed_search_all_tag = None
    parsed_tag_for_search = None  # Tag parsed from query - for SEARCH ONLY, not drawer
    title_query = None
    actual_query = raw_query
    query_error = None

    if raw_query:
        # Check for "tag: X" pattern (exactly one space after colon)
        tag_match = re.match(r"^tag:\s(.+)$", raw_query)
        # Check for "paper: X" pattern (exactly one space after colon)
        paper_match = re.match(r"^paper:\s(.+)$", raw_query)
        # Check for "title: X" pattern (exactly one space after colon)
        title_match = re.match(r"^title:\s(.+)$", raw_query)
        if tag_match:
            tag_name = tag_match.group(1)
            if request.user.is_authenticated:
                try:
                    parsed_tag_for_search = Tag.objects.get(user=request.user, name=tag_name)
                    parsed_search_all_tag = "1"
                    actual_query = ""
                except Tag.DoesNotExist:
                    query_error = f"Tag '{tag_name}' does not exist"
            else:
                query_error = "You must be logged in to search by tag"
        elif paper_match:
            arxiv_id = paper_match.group(1).strip()
            try:
                paper = Paper.objects.get(arxiv_id=arxiv_id)
                parsed_single_paper_id = str(paper.id)
                actual_query = ""
            except Paper.DoesNotExist:
                query_error = f"Paper with arXiv ID '{arxiv_id}' does not exist"
        elif title_match:
            title_query = title_match.group(1).strip()

    context = {
        "user_tags": [],
        "current_tag": None,  # Will be set from URL only, not from query
        "query": raw_query,
        "date_filter": query_params.get("date_filter", ""),
        "category_filter": query_params.get("category", ""),
        "single_paper_id": parsed_single_paper_id or query_params.get("single_paper"),
        "search_all_tag": parsed_search_all_tag or query_params.get("search_all"),
        "title_query": title_query,
        "exclude_ids": exclude_ids,
        "query_error": query_error,
        "parsed_tag_for_search": parsed_tag_for_search,  # Pass to context for search functions
    }

    with connection.cursor() as cursor:
        cursor.execute("SET hnsw.ef_search = 256")
        cursor.execute("SET hnsw.iterative_scan = 'relaxed_order'")
        cursor.execute("SET hnsw.max_scan_tuples = 1000")

    if request.user.is_authenticated:
        context["user_tags"] = Tag.objects.filter(user=request.user).prefetch_related(
            "tagged_papers"
        )

    # Set current_tag from URL parameter ONLY (for drawer state)
    current_tag_id = query_params.get("tag")
    if current_tag_id is not None and request.user.is_authenticated:
        context["current_tag"] = get_object_or_404(Tag, id=current_tag_id, user=request.user)

        # Load drawer content
        if not is_ajax:
            sort = request.GET.get("sort", "added")
            tagged_papers = TaggedPaper.objects.filter(tag=context["current_tag"]).select_related(
                "paper"
            )
            if sort == "alpha":
                tagged_papers = tagged_papers.order_by("paper__title")
            elif sort == "submitted":
                tagged_papers = tagged_papers.order_by("-paper__created")
            elif sort == "updated":
                tagged_papers = tagged_papers.order_by("-paper__updated")
            else:  # added (default)
                tagged_papers = tagged_papers.order_by("-added_at")

            # Process LaTeX in titles for drawer
            tagged_papers_processed = []
            for tagged in tagged_papers:
                tagged_papers_processed.append(
                    {
                        "paper": tagged.paper,
                        "processed_title": process_latex_commands(tagged.paper.title),
                        "added_at": tagged.added_at,
                    }
                )
            context["tagged_papers"] = tagged_papers_processed

    if not context["date_filter"]:
        context["date_filter"] = "1week"

    # If there's a query error, don't execute search
    if query_error:
        papers, search_context = [], None
    elif context["single_paper_id"]:
        papers, search_context = paper_search(context)
    elif context["search_all_tag"] and context["parsed_tag_for_search"]:
        papers, search_context = tag_search(context)
    elif context["title_query"]:
        papers, search_context = title_search(context)
    elif actual_query:
        papers, search_context = keyword_search(context)
    else:
        papers, search_context = [], None

    has_more = bool(len(papers)) and (len(context["exclude_ids"]) + RESULTS_PER_PAGE < MAX_RESULTS)

    # Get all categories
    all_categories = set()
    for paper in papers:
        if paper.categories:
            all_categories.update(paper.categories)
    all_categories = sorted(list(all_categories))

    # Get user's tags
    paper_tags = {}
    if request.user.is_authenticated:
        paper_ids = [p.id for p in papers]
        tagged = TaggedPaper.objects.filter(
            tag__user=request.user, paper_id__in=paper_ids
        ).select_related("tag")
        for tp in tagged:
            if tp.paper_id not in paper_tags:
                paper_tags[tp.paper_id] = []
            paper_tags[tp.paper_id].append(tp.tag.name)

    results = [
        {
            "paper": paper,
            "tags": paper_tags.get(paper.id, []),
            "processed_title": process_latex_commands(paper.title),
            "processed_abstract": process_latex_commands(paper.abstract),
        }
        for paper in papers
    ]

    if is_ajax:
        html = render_to_string(
            "papers/paper_card.html",
            {
                "results": results,
                "user_tags": context["user_tags"],
                "current_tag": context["current_tag"],
            },
            request=request,
        )

        return JsonResponse(
            {
                "success": True,
                "html": html,
                "has_more": has_more,
            }
        )

    context["results"] = results
    context["has_more"] = has_more
    context["search_context"] = search_context
    context["all_categories"] = all_categories
    context["show_filters"] = search_context is not None

    return render(request, "papers/search.html", context)


def title_search(context):
    """Execute title substring search - returns (results, has_more, search_context, all_categories)"""
    query = context["title_query"]
    search_context = {
        "type": "title",
        "query": query,
    }

    valid_paper_query = get_valid_papers(context)
    papers = valid_paper_query.filter(title__icontains=query)
    papers = papers.prefetch_related("authors").order_by("-created")

    papers = papers[:RESULTS_PER_PAGE]

    return papers, search_context


def keyword_search(context):
    """Execute keyword search - returns (results, has_more, search_context, all_categories)"""
    query = context["query"]
    search_context = {
        "type": "keyword",
        "query": query,
    }
    valid_paper_query = get_valid_papers(context)

    search_query = SearchQuery(query, config="english")

    papers = valid_paper_query.filter(search_vector=search_query)
    papers = papers.annotate(rank=SearchRank("search_vector", search_query))
    papers = papers.order_by("-rank", "-created")
    papers = papers.prefetch_related("authors")[:RESULTS_PER_PAGE]

    return papers, search_context


def paper_search(context):
    """Execute similarity search for a single paper - returns (results, has_more, search_context, all_categories)"""
    paper_id = context["single_paper_id"]

    # Try to get paper - handle both internal ID and arXiv ID
    try:
        # First try as internal ID (integer)
        paper = Paper.objects.get(id=int(paper_id))
    except (ValueError, Paper.DoesNotExist):
        # If that fails, try as arXiv ID
        try:
            paper = Paper.objects.get(arxiv_id=paper_id)
        except Paper.DoesNotExist:
            # If both fail, raise 404
            paper = get_object_or_404(Paper, id=paper_id)

    search_context = {
        "type": "single_paper",
        "paper": paper,
    }

    valid_paper_query = get_valid_papers(context, paper)

    papers = get_similar_embeddings(paper, valid_paper_query, RESULTS_PER_PAGE)

    return papers, search_context


def tag_search(context):
    """Execute similarity search for all papers in a tag - returns (results, has_more, search_context, all_categories)"""
    tag = context["parsed_tag_for_search"]  # Use the tag from query parsing, not drawer state

    search_context = {
        "type": "tag_all",
        "tag": tag,  # Include tag in search context so template can show which tag was searched
    }

    tagged_papers = TaggedPaper.objects.filter(tag=tag).select_related("paper")
    tagged_papers = [tagged.paper for tagged in tagged_papers]

    if not tagged_papers:
        return [], search_context

    random.shuffle(tagged_papers)

    valid_paper_query = get_valid_papers(context)

    # Calculate papers per source - need enough to cover offset + page + 1
    total_needed = RESULTS_PER_PAGE
    res_per_source = max(1, total_needed // max(1, len(tagged_papers))) + 1
    # we won't use more than 10 results per source

    results = []
    seen_papers = set()
    count = 0
    start_time = time.time()
    for paper in tagged_papers:
        if time.time() - start_time > 2:  # need to finish before timeout
            res_per_source = total_needed
        similars = get_similar_embeddings(paper, valid_paper_query, res_per_source)
        new_similars = []

        for similar in similars:
            if similar.id not in seen_papers:
                new_similars.append(similar)
                seen_papers.add(similar.id)
                count += 1

        if new_similars:
            results.append(new_similars)

        if count >= total_needed:
            print("breaking")
            break

    papers = []
    for i in range(res_per_source):
        for result_group in results:
            if len(result_group) > i:
                papers.append(result_group[i])

    return papers, search_context


def paper_detail(request, paper_id):
    """Display paper details with option to search similar from here"""
    paper = get_object_or_404(Paper, id=paper_id)

    authors = (
        paper.authors.through.objects.filter(paper=paper).select_related("author").order_by("order")
    )
    has_embedding = EMBEDDING_MODEL.objects.filter(paper=paper).exists()

    abstract = process_latex_commands(paper.abstract)

    # Get tag context if present
    tag_id = request.GET.get("tag")
    current_tag = None
    user_tags = []
    tagged_papers = []
    paper_tags = []

    if request.user.is_authenticated:
        user_tags = Tag.objects.filter(user=request.user).prefetch_related("tagged_papers")
        if tag_id:
            current_tag = Tag.objects.filter(id=tag_id, user=request.user).first()
            if current_tag:
                # Get tagged papers for drawer
                sort = request.GET.get("sort", "added")
                tagged_papers_qs = TaggedPaper.objects.filter(tag=current_tag).select_related(
                    "paper"
                )
                if sort == "alpha":
                    tagged_papers_qs = tagged_papers_qs.order_by("paper__title")
                elif sort == "submitted":
                    tagged_papers_qs = tagged_papers_qs.order_by("-paper__created")
                elif sort == "updated":
                    tagged_papers_qs = tagged_papers_qs.order_by("-paper__updated")
                else:  # added (default)
                    tagged_papers_qs = tagged_papers_qs.order_by("-added_at")

                # Process LaTeX in titles for drawer
                for tagged in tagged_papers_qs:
                    tagged_papers.append(
                        {
                            "paper": tagged.paper,
                            "processed_title": process_latex_commands(tagged.paper.title),
                            "added_at": tagged.added_at,
                        }
                    )

        # Get tags for this specific paper
        paper_tag_objs = TaggedPaper.objects.filter(
            tag__user=request.user, paper=paper
        ).select_related("tag")
        paper_tags = [tp.tag.name for tp in paper_tag_objs]

    return render(
        request,
        "papers/detail.html",
        {
            "paper": paper,
            "authors": authors,
            "has_embedding": has_embedding,
            "abstract": abstract,
            "current_tag": current_tag,
            "user_tags": user_tags,
            "tagged_papers": tagged_papers,
            "paper_tags": paper_tags,
        },
    )


def get_valid_papers(context, current_paper=None):
    tag = context.get("current_tag")

    excluded_ids = context.get("exclude_ids", set())

    if tag is not None:
        tagged_papers = TaggedPaper.objects.filter(tag=tag).select_related("paper")
        tagged_paper_ids = set(tagged_papers.values_list("paper_id", flat=True))
        excluded_ids.update(tagged_paper_ids)

    if current_paper is not None:
        excluded_ids.add(current_paper.id)

    paper_query = Paper.objects.exclude(id__in=excluded_ids)
    date_cutoff = get_date_cutoff(context["date_filter"])
    if date_cutoff:
        paper_query = paper_query.filter(created__gte=date_cutoff)
    if context["category_filter"]:
        paper_query = paper_query.filter(categories__contains=[context["category_filter"]])

    return paper_query


def get_similar_embeddings(paper, valid_paper_query, num_results):
    embedding = EMBEDDING_MODEL.objects.filter(paper=paper).first()
    if not embedding:
        return []
    similar_embeddings = list(
        EMBEDDING_MODEL.objects.filter(paper__in=valid_paper_query)
        .annotate(distance=DISTANCE_FUNCTION("vector", embedding.vector))
        .select_related("paper")
        .prefetch_related("paper__authors")
        .order_by("distance")[:num_results]
    )
    return [emb.paper for emb in similar_embeddings]
