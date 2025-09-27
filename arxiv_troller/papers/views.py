from django.shortcuts import render, get_object_or_404
from django.db.models import Q
from django.core.paginator import Paginator
from django.db import connection
from pgvector.django import L2Distance
from .models import Paper, Embedding, Author


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

    return render(
        request,
        "papers/detail.html",
        {
            "paper": paper,
            "authors": authors,
            "has_embedding": has_embedding,
        },
    )


def similar_papers(request, paper_id):
    """Find similar papers using vector similarity."""
    paper = get_object_or_404(Paper, id=paper_id)

    # Get the embedding for this paper
    embedding = get_object_or_404(
        Embedding,
        paper=paper,
        model_name="gemini-embedding-001",
        embedding_type="abstract",
    )

    # Find similar papers using pgvector
    similar_embeddings = (
        Embedding.objects.filter(model_name="gemini-embedding-001", embedding_type="abstract")
        .exclude(paper=paper)
        .annotate(distance=L2Distance("vector", embedding.vector))
        .select_related("paper")
        .order_by("distance")[:60]
    )

    # Build results with paper info
    similar_papers = []
    for emb in similar_embeddings:
        similar_papers.append(
            {
                "paper": emb.paper,
                "distance": emb.distance,
                "similarity": 1 / (1 + emb.distance),  # Convert distance to similarity
            }
        )

    return render(
        request,
        "papers/similar.html",
        {
            "original_paper": paper,
            "similar_papers": similar_papers,
        },
    )
