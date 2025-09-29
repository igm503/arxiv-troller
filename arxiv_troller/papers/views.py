import re
from datetime import datetime, timedelta

from django.shortcuts import render, get_object_or_404
from django.core.paginator import Paginator
from pgvector.django import L2Distance
from django.db import connection
from django.db.models import Q

from .models import Paper, Embedding, Tag, TaggedPaper, RemovedPaper


def search_papers(request):
    """Search for papers by title."""
    query = request.GET.get("q", "")
    papers = None

    if query:
        papers = (
            Paper.objects.filter(title__icontains=query)
            .prefetch_related("authors", "paperauthor_set__author")
            .order_by("-created")[:100]
        )

    return render(
        request,
        "papers/search.html",
        {
            "query": query,
            "papers": papers,
        },
    )


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

    # Special characters and escapes
    text = re.sub(r"\\%", "%", text)
    text = re.sub(r"\\&", "&", text)
    text = re.sub(r"\\\$", "$", text)
    text = re.sub(r"\\#", "#", text)
    text = re.sub(r"\\_", "_", text)
    text = re.sub(r"\\\{", "{", text)
    text = re.sub(r"\\\}", "}", text)
    text = re.sub(r"\\textbackslash(?:\{\})?", r"\\", text)  # Fixed: use raw string
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
    text = re.sub(r"\\,", " ", text)  # thin space
    text = re.sub(r"~", "&nbsp;", text)  # non-breaking space
    text = re.sub(r"\\\\", "<br>", text)  # line break

    return text


def paper_detail(request, paper_id):
    """Display paper details."""
    paper = get_object_or_404(Paper, id=paper_id)

    # Get authors in order
    authors = (
        paper.authors.through.objects.filter(paper=paper).select_related("author").order_by("order")
    )

    # Check if embedding exists
    has_embedding = Embedding.objects.filter(
        paper=paper, model_name="gemini-embedding-001", embedding_type="abstract"
    ).exists()

    abstract = process_latex_commands(paper.abstract)

    return render(
        request,
        "papers/detail.html",
        {
            "paper": paper,
            "authors": authors,
            "has_embedding": has_embedding,
            "abstract": abstract,
        },
    )


def similar_papers(request, paper_id):
    """Find similar papers using vector similarity."""
    with connection.cursor() as cursor:
        cursor.execute("SET hnsw.ef_search = 1000")

    paper = get_object_or_404(Paper, id=paper_id)

    # Get the embedding for this paper
    embedding = get_object_or_404(
        Embedding,
        paper=paper,
        model_name="gemini-embedding-001",
        embedding_type="abstract",
    )

    # Get filter parameters from request
    date_filter = request.GET.get("date_filter", "all")
    category_filter = request.GET.get("category", "")
    page_number = request.GET.get("page", 1)

    # First, get the paper IDs that match our date filter
    paper_query = Paper.objects.all()

    # Apply date filter to papers first
    if date_filter != "all":
        today = datetime.now().date()
        if date_filter == "1month":
            date_cutoff = today - timedelta(days=30)
        elif date_filter == "3months":
            date_cutoff = today - timedelta(days=90)
        elif date_filter == "6months":
            date_cutoff = today - timedelta(days=180)
        elif date_filter == "1year":
            date_cutoff = today - timedelta(days=365)
        elif date_filter == "2years":
            date_cutoff = today - timedelta(days=730)
        else:
            date_cutoff = None

        if date_cutoff:
            paper_query = paper_query.filter(created__gte=date_cutoff)

    # Apply category filter to papers
    if category_filter:
        paper_query = paper_query.filter(categories__contains=[category_filter])

    # Get valid paper IDs
    valid_paper_ids = paper_query.values_list("id", flat=True)

    # Now filter embeddings by these paper IDs
    similar_embeddings = (
        Embedding.objects.filter(
            model_name="gemini-embedding-001",
            embedding_type="abstract",
            paper_id__in=valid_paper_ids,  # Only search within filtered papers
        )
        .exclude(paper=paper)
        .annotate(distance=L2Distance("vector", embedding.vector))
        .select_related("paper")
        .order_by("distance")[:1000]
    )

    # IMPORTANT: We need to limit the queryset BEFORE pagination to avoid counting all embeddings
    # This is a limitation when using vector similarity - we can't efficiently paginate millions of results
    # So we limit to top 1000 most similar papers for filtering/pagination
    similar_embeddings = similar_embeddings[:1000]

    # Convert to list to avoid re-evaluation
    similar_list = list(similar_embeddings)

    # Get all unique categories from these results for filter dropdown
    all_categories = set()
    for emb in similar_list:
        if emb.paper.categories:
            all_categories.update(emb.paper.categories)
    all_categories = sorted(list(all_categories))

    # Paginate the limited results
    paginator = Paginator(similar_list, 20)
    page_obj = paginator.get_page(page_number)

    # Build results with paper info
    similar_papers = []
    for emb in page_obj:
        similar_papers.append(
            {
                "paper": emb.paper,
                "distance": emb.distance,
                "similarity_score": 1 / (1 + emb.distance),  # Convert L2 distance to a score
            }
        )

    return render(
        request,
        "papers/similar.html",
        {
            "original_paper": paper,
            "similar_papers": similar_papers,
            "page_obj": page_obj,
            "date_filter": date_filter,
            "category_filter": category_filter,
            "all_categories": all_categories,
        },
    )


def unified_search_view(request, tag_id=None):
    """Unified view for search, similarity, and tag pages"""
    with connection.cursor() as cursor:
        cursor.execute("SET hnsw.ef_search = 1000")

    context = {
        "user_tags": [],
        "current_tag": None,
        "mode": "search",
        "query": request.GET.get("q", ""),
        "single_paper_id": request.GET.get("single_paper"),  # For single paper similarity
    }

    # Get user tags if logged in
    if request.user.is_authenticated:
        context["user_tags"] = Tag.objects.filter(user=request.user).prefetch_related(
            "tagged_papers"
        )

    # Handle tag-based similarity search
    if tag_id and request.user.is_authenticated:
        tag = get_object_or_404(Tag, id=tag_id, user=request.user)
        context["current_tag"] = tag
        context["mode"] = "tag"

        # Get sorting preference
        sort = request.GET.get("sort", "time")  # 'time' or 'alpha'

        # Get tagged papers
        tagged_papers = TaggedPaper.objects.filter(tag=tag).select_related("paper")
        if sort == "alpha":
            tagged_papers = tagged_papers.order_by("paper__title")
        else:
            tagged_papers = tagged_papers.order_by("-added_at")

        context["tagged_papers"] = tagged_papers
        context["sort"] = sort

        # Get removed paper IDs for this tag
        removed_paper_ids = set(
            RemovedPaper.objects.filter(tag=tag).values_list("paper_id", flat=True)
        )

        # Get already tagged paper IDs
        tagged_paper_ids = set(tagged_papers.values_list("paper_id", flat=True))

        # Get date filter
        date_filter = request.GET.get("date_filter", "1month")
        context["date_filter"] = date_filter

        # Check if searching for single paper similarity
        single_paper_id = request.GET.get("single_paper")

        if single_paper_id:
            # Single paper similarity
            source_papers = [get_object_or_404(TaggedPaper, tag=tag, paper_id=single_paper_id)]
            papers_per_source = 40
        else:
            # All tagged papers similarity
            source_papers = list(tagged_papers[:25])  # Limit to 25 papers
            papers_per_source = max(1, min(40, 1000 // max(1, len(source_papers))))

        # Collect similarity results with proper interleaving
        interleaved_results = []

        for round_idx in range(papers_per_source):
            for source_idx, tagged in enumerate(source_papers):
                embedding = Embedding.objects.filter(
                    paper=tagged.paper,
                    model_name="gemini-embedding-001",
                    embedding_type="abstract",
                ).first()

                if not embedding:
                    continue

                # Build date-filtered query
                paper_query = Paper.objects.exclude(
                    Q(id__in=removed_paper_ids) | Q(id__in=tagged_paper_ids)
                )

                if date_filter != "all":
                    today = datetime.now().date()
                    date_cutoffs = {
                        "1month": today - timedelta(days=30),
                        "3months": today - timedelta(days=90),
                        "6months": today - timedelta(days=180),
                        "1year": today - timedelta(days=365),
                        "2years": today - timedelta(days=730),
                    }
                    if date_filter in date_cutoffs:
                        paper_query = paper_query.filter(created__gte=date_cutoffs[date_filter])

                valid_paper_ids = paper_query.values_list("id", flat=True)

                # Get the nth most similar paper for this source
                similar = (
                    Embedding.objects.filter(
                        model_name="gemini-embedding-001",
                        embedding_type="abstract",
                        paper_id__in=valid_paper_ids,
                    )
                    .annotate(distance=L2Distance("vector", embedding.vector))
                    .select_related("paper")
                    .order_by("distance")[round_idx : round_idx + 1]  # Get just the nth result
                ).first()

                if similar:
                    interleaved_results.append(
                        {
                            "paper": similar.paper,
                            "distance": similar.distance,
                            "source_paper": tagged.paper,
                            "similarity_score": 1 / (1 + similar.distance),
                        }
                    )

        # Remove duplicates while preserving order
        seen_papers = set()
        unique_results = []
        for result in interleaved_results:
            if result["paper"].id not in seen_papers:
                seen_papers.add(result["paper"].id)
                unique_results.append(result)

        # Limit to 1000 results
        unique_results = unique_results[:1000]

        # Paginate
        paginator = Paginator(unique_results, 20)
        page_obj = paginator.get_page(request.GET.get("page", 1))
        context["similar_papers"] = page_obj
        context["page_obj"] = page_obj

    # Handle regular search
    elif context["query"]:
        context["mode"] = "search"

        papers = (
            Paper.objects.filter(title__icontains=context["query"])
            .prefetch_related("authors", "paperauthor_set__author")
            .order_by("-created")[:100]
        )
        context["papers"] = papers

    return render(request, "papers/unified.html", context)
