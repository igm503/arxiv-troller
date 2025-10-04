import re
from datetime import datetime, timedelta

from django.shortcuts import render, get_object_or_404, redirect
from django.core.paginator import Paginator
from pgvector.django import L2Distance
from django.db import connection
from django.db.models import Q

from .models import Paper, Embedding, Tag, TaggedPaper, RemovedPaper


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
    text = re.sub(r"\\textbackslash(?:\{\})?", r"\\", text)
    text = re.sub(r"\\~", "~", text)
    text = re.sub(r"\\\^", "^", text)

    # LaTeX quotes
    text = re.sub(r"``", '"', text)
    text = re.sub(r"''", '"', text)
    text = re.sub(r"`", """, text)
    text = re.sub(r"'", """, text)

    # Common spacing commands
    text = re.sub(r"\\,", " ", text)
    text = re.sub(r"~", "&nbsp;", text)
    text = re.sub(r"\\\\", "<br>", text)

    return text


def get_date_cutoff(date_filter):
    """Convert date filter string to datetime cutoff"""
    if date_filter == "all":
        return None
    
    today = datetime.now().date()
    date_cutoffs = {
        "1week": today - timedelta(days=7),
        "1month": today - timedelta(days=30),
        "3months": today - timedelta(days=90),
        "6months": today - timedelta(days=180),
        "1year": today - timedelta(days=365),
        "2years": today - timedelta(days=730),
    }
    return date_cutoffs.get(date_filter)


def unified_search_view(request):
    """Unified view for all search types: keyword, tag similarity, single paper similarity"""
    with connection.cursor() as cursor:
        cursor.execute("SET hnsw.ef_search = 1000")
    
    # Initialize context
    context = {
        "user_tags": [],
        "current_tag": None,
        "query": request.GET.get("q", ""),
        "date_filter": request.GET.get("date_filter", ""),
        "category_filter": request.GET.get("category", ""),
        "single_paper_id": request.GET.get("single_paper"),
        "search_all_tag": request.GET.get("search_all"),
        "results": [],
        "page_obj": None,
        "search_context": None,
        "show_filters": False,
        "tagged_papers": [],
        "all_categories": [],
    }
    
    # Get user tags if logged in
    if request.user.is_authenticated:
        context["user_tags"] = Tag.objects.filter(user=request.user).prefetch_related(
            "tagged_papers"
        )
    
    # Check for tag context
    tag_id = request.GET.get("tag")
    if tag_id and request.user.is_authenticated:
        context["current_tag"] = get_object_or_404(Tag, id=tag_id, user=request.user)
        
        # Get tagged papers for drawer
        sort = request.GET.get("sort", "time")
        tagged_papers = TaggedPaper.objects.filter(tag=context["current_tag"]).select_related("paper")
        if sort == "alpha":
            tagged_papers = tagged_papers.order_by("paper__title")
        else:
            tagged_papers = tagged_papers.order_by("-added_at")
        context["tagged_papers"] = tagged_papers
    
    # Determine search type and execute
    if context["single_paper_id"]:
        # Single paper similarity search
        _execute_single_paper_search(request, context)
    elif context["search_all_tag"] and context["current_tag"]:
        # All papers in tag similarity search
        _execute_tag_similarity_search(request, context)
    elif context["query"]:
        # Keyword search
        _execute_keyword_search(request, context)
    
    return render(request, "papers/unified.html", context)


def _execute_keyword_search(request, context):
    """Execute keyword search"""
    query = context["query"]
    
    # Set default filters for keyword search
    if not context["date_filter"]:
        context["date_filter"] = "all"
    
    context["search_context"] = {
        "type": "keyword",
        "query": query,
    }
    context["show_filters"] = True
    
    # Apply filters
    papers = Paper.objects.filter(title__icontains=query)
    
    date_cutoff = get_date_cutoff(context["date_filter"])
    if date_cutoff:
        papers = papers.filter(created__gte=date_cutoff)
    
    if context["category_filter"]:
        papers = papers.filter(categories__contains=[context["category_filter"]])
    
    papers = papers.prefetch_related("authors").order_by("-created")[:1000]
    
    # Get all categories for filter
    all_categories = set()
    for paper in papers:
        if paper.categories:
            all_categories.update(paper.categories)
    context["all_categories"] = sorted(list(all_categories))
    
    # Get user's tags for these papers if logged in
    paper_tags = {}
    if request.user.is_authenticated:
        paper_ids = [p.id for p in papers]
        tagged = TaggedPaper.objects.filter(
            tag__user=request.user,
            paper_id__in=paper_ids
        ).select_related('tag')
        
        for tp in tagged:
            if tp.paper_id not in paper_tags:
                paper_tags[tp.paper_id] = []
            paper_tags[tp.paper_id].append(tp.tag.name)
    
    # Format results
    results = [{"paper": paper, "similarity_score": None, "tags": paper_tags.get(paper.id, [])} for paper in papers]
    
    # Paginate
    paginator = Paginator(results, 20)
    page_obj = paginator.get_page(request.GET.get("page", 1))
    
    context["results"] = page_obj
    context["page_obj"] = page_obj


def _execute_single_paper_search(request, context):
    """Execute similarity search for a single paper"""
    paper_id = context["single_paper_id"]
    paper = get_object_or_404(Paper, id=paper_id)
    
    # Set default filters for similarity search
    if not context["date_filter"]:
        context["date_filter"] = "1week"
    
    context["search_context"] = {
        "type": "single_paper",
        "paper": paper,
    }
    context["show_filters"] = True
    
    # Get embedding
    embedding = Embedding.objects.filter(
        paper=paper,
        model_name="gemini-embedding-001",
        embedding_type="abstract",
    ).first()
    
    if not embedding:
        context["results"] = []
        return
    
    # Build excluded paper IDs
    excluded_ids = {paper.id}
    if context["current_tag"]:
        removed_ids = set(
            RemovedPaper.objects.filter(tag=context["current_tag"]).values_list("paper_id", flat=True)
        )
        excluded_ids.update(removed_ids)
    
    # Build base query
    paper_query = Paper.objects.exclude(id__in=excluded_ids)
    
    date_cutoff = get_date_cutoff(context["date_filter"])
    if date_cutoff:
        paper_query = paper_query.filter(created__gte=date_cutoff)
    
    if context["category_filter"]:
        paper_query = paper_query.filter(categories__contains=[context["category_filter"]])
    
    valid_paper_ids = paper_query.values_list("id", flat=True)
    
    # Get similar papers
    similar_embeddings = (
        Embedding.objects.filter(
            model_name="gemini-embedding-001",
            embedding_type="abstract",
            paper_id__in=valid_paper_ids,
        )
        .annotate(distance=L2Distance("vector", embedding.vector))
        .select_related("paper")
        .prefetch_related("paper__authors")
        .order_by("distance")[:1000]
    )
    
    # Get all categories
    all_categories = set()
    for emb in similar_embeddings:
        if emb.paper.categories:
            all_categories.update(emb.paper.categories)
    context["all_categories"] = sorted(list(all_categories))
    
    # Get user's tags for these papers if logged in
    paper_tags = {}
    if request.user.is_authenticated:
        paper_ids = [emb.paper.id for emb in similar_embeddings]
        tagged = TaggedPaper.objects.filter(
            tag__user=request.user,
            paper_id__in=paper_ids
        ).select_related('tag')
        
        for tp in tagged:
            if tp.paper_id not in paper_tags:
                paper_tags[tp.paper_id] = []
            paper_tags[tp.paper_id].append(tp.tag.name)
    
    # Format results
    results = [
        {
            "paper": emb.paper,
            "similarity_score": 1 / (1 + emb.distance),
            "tags": paper_tags.get(emb.paper.id, [])
        }
        for emb in similar_embeddings
    ]
    
    # Paginate
    paginator = Paginator(results, 20)
    page_obj = paginator.get_page(request.GET.get("page", 1))
    
    context["results"] = page_obj
    context["page_obj"] = page_obj


def _execute_tag_similarity_search(request, context):
    """Execute similarity search for all papers in a tag"""
    tag = context["current_tag"]
    
    # Set default filters for similarity search
    if not context["date_filter"]:
        context["date_filter"] = "1week"
    
    context["search_context"] = {
        "type": "tag_all",
    }
    context["show_filters"] = True
    
    # Get tagged papers
    tagged_papers = TaggedPaper.objects.filter(tag=tag).select_related("paper")
    source_papers = list(tagged_papers[:25])  # Limit to 25 papers
    
    if not source_papers:
        context["results"] = []
        return
    
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
        paper_query = paper_query.filter(categories__contains=[context["category_filter"]])
    
    valid_paper_ids = paper_query.values_list("id", flat=True)
    
    # Calculate papers per source for interleaving
    papers_per_source = max(1, min(40, 1000 // max(1, len(source_papers))))
    
    # Collect interleaved similarity results
    interleaved_results = []
    
    for round_idx in range(papers_per_source):
        for tagged in source_papers:
            embedding = Embedding.objects.filter(
                paper=tagged.paper,
                model_name="gemini-embedding-001",
                embedding_type="abstract",
            ).first()
            
            if not embedding:
                continue
            
            # Get the nth most similar paper for this source
            similar = (
                Embedding.objects.filter(
                    model_name="gemini-embedding-001",
                    embedding_type="abstract",
                    paper_id__in=valid_paper_ids,
                )
                .annotate(distance=L2Distance("vector", embedding.vector))
                .select_related("paper")
                .prefetch_related("paper__authors")
                .order_by("distance")[round_idx : round_idx + 1]
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
    
    # Get all categories
    all_categories = set()
    for result in unique_results:
        if result["paper"].categories:
            all_categories.update(result["paper"].categories)
    context["all_categories"] = sorted(list(all_categories))
    
    # Get user's tags for these papers
    paper_tags = {}
    if request.user.is_authenticated:
        paper_ids = [r["paper"].id for r in unique_results]
        tagged = TaggedPaper.objects.filter(
            tag__user=request.user,
            paper_id__in=paper_ids
        ).select_related('tag')
        
        for tp in tagged:
            if tp.paper_id not in paper_tags:
                paper_tags[tp.paper_id] = []
            paper_tags[tp.paper_id].append(tp.tag.name)
    
    # Add tags to results
    for result in unique_results:
        result["tags"] = paper_tags.get(result["paper"].id, [])
    
    # Paginate
    paginator = Paginator(unique_results, 20)
    page_obj = paginator.get_page(request.GET.get("page", 1))
    
    context["results"] = page_obj
    context["page_obj"] = page_obj


def paper_detail(request, paper_id):
    """Display paper details with option to search similar from here"""
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
                sort = request.GET.get("sort", "time")
                tagged_papers = TaggedPaper.objects.filter(tag=current_tag).select_related("paper")
                if sort == "alpha":
                    tagged_papers = tagged_papers.order_by("paper__title")
                else:
                    tagged_papers = tagged_papers.order_by("-added_at")
        
        # Get tags for this specific paper
        paper_tag_objs = TaggedPaper.objects.filter(
            tag__user=request.user,
            paper=paper
        ).select_related('tag')
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


def similar_papers(request, paper_id):
    """Redirect to unified search with single paper context"""
    
    params = f"?single_paper={paper_id}"
    
    # Preserve tag context if present
    tag_id = request.GET.get("tag")
    if tag_id:
        params += f"&tag={tag_id}"
    
    # Set default filter
    if not request.GET.get("date_filter"):
        params += "&date_filter=1week"
    
    return redirect(f"/papers/{params}")
