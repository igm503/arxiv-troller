import json
import re
from datetime import timedelta
import random

from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.template.loader import render_to_string
from django.utils import timezone
from pgvector.django import L2Distance
from django.db import connection

from .models import Paper, Embedding, Tag, TaggedPaper

RESULTS_PER_PAGE = 40
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
    text = re.sub(
        r"\\href\{([^}]+)\}\{([^}]+)\}", r'<a href="\1" target="_blank">\2</a>', text
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

    context = {
        "user_tags": [],
        "current_tag": None,
        "query": query_params.get("q", ""),
        "date_filter": query_params.get("date_filter", ""),
        "category_filter": query_params.get("category", ""),
        "single_paper_id": query_params.get("single_paper"),
        "search_all_tag": query_params.get("search_all"),
        "exclude_ids": exclude_ids,
    }

    with connection.cursor() as cursor:
        cursor.execute("SET hnsw.ef_search = 200")
        cursor.execute("SET hnsw.iterative_scan = 'strict_order'")

    if request.user.is_authenticated:
        context["user_tags"] = Tag.objects.filter(user=request.user).prefetch_related(
            "tagged_papers"
        )

    current_tag_id = query_params.get("tag")
    if current_tag_id is not None and request.user.is_authenticated:
        context["current_tag"] = get_object_or_404(
            Tag, id=current_tag_id, user=request.user
        )

        if not is_ajax:  # Only need drawer for full page load
            sort = request.GET.get("sort", "time")
            tagged_papers = TaggedPaper.objects.filter(
                tag=context["current_tag"]
            ).select_related("paper")
            if sort == "alpha":
                tagged_papers = tagged_papers.order_by("paper__title")
            else:
                tagged_papers = tagged_papers.order_by("-added_at")
            context["tagged_papers"] = tagged_papers

    if context["single_paper_id"]:
        papers, search_context = paper_search(context)
    elif context["search_all_tag"] and context["current_tag"]:
        papers, search_context = tag_search(context)
    elif context["query"]:
        papers, search_context = keyword_search(context)
    else:
        papers, search_context = [], None

    has_more = bool(len(papers)) and (
        len(context["exclude_ids"]) + RESULTS_PER_PAGE < MAX_RESULTS
    )

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
        {"paper": paper, "tags": paper_tags.get(paper.id, [])} for paper in papers
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


def keyword_search(context):
    """Execute keyword search - returns (results, has_more, search_context, all_categories)"""
    query = context["query"]

    if not context["date_filter"]:
        context["date_filter"] = "all"

    search_context = {
        "type": "keyword",
        "query": query,
    }

    valid_paper_query = get_valid_papers(context)
    papers = valid_paper_query.filter(title__icontains=query)
    papers = papers.prefetch_related("authors").order_by("-created")

    papers = papers[:RESULTS_PER_PAGE]

    return papers, search_context


def paper_search(context):
    """Execute similarity search for a single paper - returns (results, has_more, search_context, all_categories)"""
    paper_id = context["single_paper_id"]
    paper = get_object_or_404(Paper, id=paper_id)

    if not context["date_filter"]:
        context["date_filter"] = "1week"

    search_context = {
        "type": "single_paper",
        "paper": paper,
    }

    valid_paper_query = get_valid_papers(context, paper)

    papers = get_similar_embeddings(paper, valid_paper_query, RESULTS_PER_PAGE)

    return papers, search_context


def tag_search(context):
    """Execute similarity search for all papers in a tag - returns (results, has_more, search_context, all_categories)"""
    tag = context["current_tag"]

    if not context["date_filter"]:
        context["date_filter"] = "1week"

    search_context = {
        "type": "tag_all",
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

    results = []
    seen_papers = set()
    count = 0
    for paper in tagged_papers:
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

    # Get authors in order
    authors = (
        paper.authors.through.objects.filter(paper=paper)
        .select_related("author")
        .order_by("order")
    )

    # Check if embedding exists
    has_embedding = Embedding.objects.filter(
        paper=paper, model_name="gemini-embedding-001", embedding_type="abstract"
    ).exists()

    abstract = process_latex_commands(paper.abstract)

    # Get tag context if present
    tag_id = request.GET.get("tag")
    current_tag = None
    user_tags = []
    tagged_papers = []
    paper_tags = []

    if request.user.is_authenticated:
        user_tags = Tag.objects.filter(user=request.user).prefetch_related(
            "tagged_papers"
        )
        if tag_id:
            current_tag = Tag.objects.filter(id=tag_id, user=request.user).first()
            if current_tag:
                # Get tagged papers for drawer
                sort = request.GET.get("sort", "time")
                tagged_papers = TaggedPaper.objects.filter(
                    tag=current_tag
                ).select_related("paper")
                if sort == "alpha":
                    tagged_papers = tagged_papers.order_by("paper__title")
                else:
                    tagged_papers = tagged_papers.order_by("-added_at")

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
        paper_query = paper_query.filter(
            categories__contains=[context["category_filter"]]
        )

    return paper_query


def get_similar_embeddings(paper, valid_paper_query, num_results):
    valid_paper_ids = valid_paper_query.values_list("id", flat=True)

    embedding = Embedding.objects.filter(
        paper=paper,
        model_name="gemini-embedding-001",
        embedding_type="abstract",
    ).first()
    if not embedding:
        return []

    similar_embeddings = list(
        Embedding.objects.filter(
            model_name="gemini-embedding-001",
            embedding_type="abstract",
            paper_id__in=valid_paper_ids,
        )
        .annotate(distance=L2Distance("vector", embedding.vector))
        .select_related("paper")
        .prefetch_related("paper__authors")
        .order_by("distance")[:num_results]
    )

    return [emb.paper for emb in similar_embeddings]
