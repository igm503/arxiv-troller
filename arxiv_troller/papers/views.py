import re
from datetime import datetime, timedelta

from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.template.loader import render_to_string
from django.utils import timezone
from pgvector.django import L2Distance
from django.db import connection

from .models import Paper, Embedding, Tag, TaggedPaper, RemovedPaper

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
        "1week": now - timedelta(days=7),
        "1month": now - timedelta(days=30),
        "3months": now - timedelta(days=90),
        "6months": now - timedelta(days=180),
        "1year": now - timedelta(days=365),
        "2years": now - timedelta(days=730),
    }
    return date_cutoffs.get(date_filter)


def unified_search_view(request):
    """Unified view for all search types: keyword, tag similarity, single paper similarity"""
    # Check if this is an AJAX request for more results
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
    offset = int(request.GET.get("offset", 0))

    with connection.cursor() as cursor:
        cursor.execute("SET hnsw.ef_search = 200")
        cursor.execute("SET hnsw.iterative_scan = 'strict_order'")

    # Initialize context
    context = {
        "user_tags": [],
        "current_tag": None,
        "query": request.GET.get("q", ""),
        "date_filter": request.GET.get("date_filter", ""),
        "category_filter": request.GET.get("category", ""),
        "single_paper_id": request.GET.get("single_paper"),
        "search_all_tag": request.GET.get("search_all"),
        "offset": offset,
    }

    if request.user.is_authenticated:
        context["user_tags"] = Tag.objects.filter(user=request.user).prefetch_related(
            "tagged_papers"
        )

    tag_id = request.GET.get("tag")
    if tag_id and request.user.is_authenticated:
        context["current_tag"] = get_object_or_404(Tag, id=tag_id, user=request.user)

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

    # Execute search and get results + metadata
    if context["single_paper_id"]:
        papers, search_context = _execute_single_paper_search(context)
    elif context["search_all_tag"] and context["current_tag"]:
        papers, search_context = _execute_tag_similarity_search(context)
    elif context["query"]:
        papers, search_context = _execute_keyword_search(context)
    else:
        papers, search_context = [], None

    has_more = len(papers) > RESULTS_PER_PAGE
    if has_more:
        papers = papers[:RESULTS_PER_PAGE]
    has_more = has_more and (offset + RESULTS_PER_PAGE < MAX_RESULTS)

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

    # Handle AJAX request - return JSON
    if is_ajax:
        html = render_to_string(
            "papers/_paper_card.html",
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
                "next_offset": offset + len(results),
            }
        )

    # Handle full page load
    context["results"] = results
    context["has_more"] = has_more
    context["search_context"] = search_context
    context["all_categories"] = all_categories
    context["show_filters"] = search_context is not None
    context["next_offset"] = offset + len(results)

    return render(request, "papers/unified.html", context)


def _execute_keyword_search(context):
    """Execute keyword search - returns (results, has_more, search_context, all_categories)"""
    query = context["query"]
    offset = context["offset"]

    if not context["date_filter"]:
        context["date_filter"] = "all"

    search_context = {
        "type": "keyword",
        "query": query,
    }

    # Apply filters
    papers = Paper.objects.filter(title__icontains=query)
    date_cutoff = get_date_cutoff(context["date_filter"])
    if date_cutoff:
        papers = papers.filter(created__gte=date_cutoff)
    if context["category_filter"]:
        papers = papers.filter(categories__contains=[context["category_filter"]])

    # Fetch one more than needed to check if there are more results
    papers = papers.prefetch_related("authors").order_by("-created")[
        offset : offset + RESULTS_PER_PAGE + 1
    ]
    papers = list(papers)

    return papers, search_context


def _execute_single_paper_search(context):
    """Execute similarity search for a single paper - returns (results, has_more, search_context, all_categories)"""
    paper_id = context["single_paper_id"]
    paper = get_object_or_404(Paper, id=paper_id)
    offset = context["offset"]

    if not context["date_filter"]:
        context["date_filter"] = "1week"

    search_context = {
        "type": "single_paper",
        "paper": paper,
    }

    embedding = Embedding.objects.filter(
        paper=paper,
        model_name="gemini-embedding-001",
        embedding_type="abstract",
    ).first()

    if not embedding:
        return [], search_context

    # Build excluded IDs
    excluded_ids = {paper.id}
    if context["current_tag"]:
        removed_ids = set(
            RemovedPaper.objects.filter(tag=context["current_tag"]).values_list(
                "paper_id", flat=True
            )
        )
        excluded_ids.update(removed_ids)

    # Build base query
    paper_query = Paper.objects.exclude(id__in=excluded_ids)
    date_cutoff = get_date_cutoff(context["date_filter"])
    if date_cutoff:
        paper_query = paper_query.filter(created__gte=date_cutoff)
    if context["category_filter"]:
        paper_query = paper_query.filter(
            categories__contains=[context["category_filter"]]
        )

    valid_paper_ids = paper_query.values_list("id", flat=True)

    # Get similar papers
    similar_embeddings = list(
        Embedding.objects.filter(
            model_name="gemini-embedding-001",
            embedding_type="abstract",
            paper_id__in=valid_paper_ids,
        )
        .annotate(distance=L2Distance("vector", embedding.vector))
        .select_related("paper")
        .prefetch_related("paper__authors")
        .order_by("distance")[offset : offset + RESULTS_PER_PAGE + 1]
    )

    papers = [emb.paper for emb in similar_embeddings]

    return papers, search_context


def _execute_tag_similarity_search(context):
    """Execute similarity search for all papers in a tag - returns (results, has_more, search_context, all_categories)"""
    tag = context["current_tag"]
    offset = context["offset"]

    if not context["date_filter"]:
        context["date_filter"] = "1week"

    search_context = {
        "type": "tag_all",
    }

    tagged_papers = TaggedPaper.objects.filter(tag=tag).select_related("paper")
    source_papers = list(tagged_papers)

    if not source_papers:
        return [], search_context

    # Get removed and tagged paper IDs
    removed_paper_ids = set(
        RemovedPaper.objects.filter(tag=tag).values_list("paper_id", flat=True)
    )
    tagged_paper_ids = set(tagged_papers.values_list("paper_id", flat=True))
    excluded_ids = removed_paper_ids | tagged_paper_ids

    # Build base query
    paper_query = Paper.objects.exclude(id__in=excluded_ids)
    date_cutoff = get_date_cutoff(context["date_filter"])
    if date_cutoff:
        paper_query = paper_query.filter(created__gte=date_cutoff)
    if context["category_filter"]:
        paper_query = paper_query.filter(
            categories__contains=[context["category_filter"]]
        )

    valid_paper_ids = paper_query.values_list("id", flat=True)

    # Calculate papers per source - need enough to cover offset + page + 1
    total_needed = offset + RESULTS_PER_PAGE + 1
    papers_per_source = max(1, total_needed // max(1, len(source_papers))) + 1

    results = []
    for tagged in source_papers:
        embedding = Embedding.objects.filter(
            paper=tagged.paper,
            model_name="gemini-embedding-001",
            embedding_type="abstract",
        ).first()

        if not embedding:
            continue

        similars = list(
            Embedding.objects.filter(
                model_name="gemini-embedding-001",
                embedding_type="abstract",
                paper_id__in=valid_paper_ids,
            )
            .annotate(distance=L2Distance("vector", embedding.vector))
            .select_related("paper")
            .prefetch_related("paper__authors")
            .order_by("distance")[:papers_per_source]
        )

        if similars:
            results.append([similar.paper for similar in similars])

    # Interleave results
    interleaved_results = []
    for i in range(papers_per_source):
        for result_group in results:
            if len(result_group) > i:
                interleaved_results.append(result_group[i])

    # Remove duplicates
    seen_papers = set()
    unique_results = []
    for paper in interleaved_results:
        if paper.id not in seen_papers:
            seen_papers.add(paper.id)
            unique_results.append(paper)
    papers = unique_results

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
