import re
from datetime import datetime, timedelta

from django.shortcuts import render, get_object_or_404
from django.core.paginator import Paginator
from pgvector.django import L2Distance
from django.db import connection
from pgvector.django import L2Distance

from .models import Paper, Embedding


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
    text = re.sub(r'\\url\{([^}]+)\}', r'<a href="\1" target="_blank">\1</a>', text)
    text = re.sub(r'\\href\{([^}]+)\}\{([^}]+)\}', r'<a href="\1" target="_blank">\2</a>', text)

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

    # Start with base query
    query = Embedding.objects.filter(
        model_name="gemini-embedding-001", embedding_type="abstract"
    ).exclude(paper=paper)

    # Apply date filter
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
            query = query.filter(paper__created__gte=date_cutoff)

    # Apply category filter
    if category_filter:
        query = query.filter(paper__categories__contains=[category_filter])

    # Calculate L2 distance and get results
    similar_embeddings = (
        query.annotate(distance=L2Distance("vector", embedding.vector))
        .select_related("paper")
        .order_by("distance")
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
