from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login as auth_login, logout as auth_logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from .models import Tag, TaggedPaper, Paper


def login_view(request):
    if request.method == "POST":
        email = request.POST.get("email", "").lower().strip()
        if email:
            user, created = User.objects.get_or_create(
                username=email, defaults={"email": email}
            )
            user.backend = "django.contrib.auth.backends.ModelBackend"
            auth_login(request, user)
            return redirect("papers:search")
    return render(request, "papers/login.html")


def logout_view(request):
    auth_logout(request)
    return redirect("papers:search")


@login_required
@require_POST
def add_to_tag(request):
    """AJAX endpoint to add paper to tag"""
    paper_id = request.POST.get("paper_id")
    tag_name = request.POST.get("tag_name", "").strip()
    tag_id = request.POST.get("tag_id")

    if not paper_id:
        return JsonResponse({"error": "Missing paper_id"}, status=400)

    paper = get_object_or_404(Paper, id=paper_id)

    if tag_id:
        tag = get_object_or_404(Tag, id=tag_id, user=request.user)
    elif tag_name:
        tag, created = Tag.objects.get_or_create(user=request.user, name=tag_name)
    else:
        return JsonResponse({"error": "Missing tag_name or tag_id"}, status=400)

    TaggedPaper.objects.get_or_create(tag=tag, paper=paper)

    return JsonResponse({"success": True, "tag_id": tag.id, "tag_name": tag.name})


@login_required
@require_POST
def remove_from_tag(request):
    """Remove paper from tag"""
    paper_id = request.POST.get("paper_id")
    tag_id = request.POST.get("tag_id")

    tag = get_object_or_404(Tag, id=tag_id, user=request.user)
    TaggedPaper.objects.filter(tag=tag, paper_id=paper_id).delete()

    return JsonResponse({"success": True, "tag_name": tag.name})


@login_required
@require_POST
def rename_tag(request):
    """Rename a tag"""
    tag_id = request.POST.get("tag_id")
    new_name = request.POST.get("new_name", "").strip()

    if not new_name:
        return JsonResponse({"error": "Name cannot be empty"}, status=400)

    tag = get_object_or_404(Tag, id=tag_id, user=request.user)
    tag.name = new_name
    tag.save()

    return JsonResponse({"success": True, "new_name": new_name})


@login_required
def get_tag_drawer(request):
    """AJAX endpoint to get tag drawer HTML for switching tags"""
    tag_id = request.GET.get("tag_id")
    if not tag_id:
        return JsonResponse({"error": "Missing tag_id"}, status=400)
    tag = get_object_or_404(Tag, id=tag_id, user=request.user)

    # Get tagged papers
    sort = request.GET.get("sort", "time")
    tagged_papers = TaggedPaper.objects.filter(tag=tag).select_related("paper")
    if sort == "alpha":
        tagged_papers = tagged_papers.order_by("paper__title")
    else:
        tagged_papers = tagged_papers.order_by("-added_at")

    # Build HTML for papers list
    papers_html = ""
    if tagged_papers.exists():
        for tagged in tagged_papers:
            title_truncated = tagged.paper.title[:60]
            if len(tagged.paper.title) > 60:
                title_truncated += "..."

            papers_html += f"""
            <div style="padding: 5px; border-bottom: 1px solid #cccc; font-size: 0.9rem; cursor: pointer; position: relative;" 
                 onclick="searchSinglePaper({tagged.paper.id})" 
                 class="drawer-paper-card">
                <div style="font-weight: 600; color: #2c3e50; margin-bottom: 4px; font-size: 0.95rem;">
                    <a href="/papers/paper/{tagged.paper.id}/?tag={tag.id}" 
                       onclick="event.stopPropagation()"
                       style="color: inherit; text-decoration: none;">
                        {title_truncated}
                    </a>
                </div>
                <div style="color: #999; font-size: 0.85rem; font-family: monospace; margin-bottom: 8px;">
                    <a href="https://arxiv.org/abs/{tagged.paper.arxiv_id}" 
                       target="_blank" 
                       onclick="event.stopPropagation()"
                       style="color: inherit; text-decoration: none;">
                        {tagged.paper.arxiv_id}
                    </a>
                    <button class="remove-btn btn-small drawer-remove-btn"
                            onclick="event.stopPropagation(); removeFromTag({tagged.paper.id})"
                            style="display: none; position: absolute; bottom: 20px; right: 8px; padding: 2px 8px; line-height: 1;">
                        −
                    </button>
                </div>
            </div>
            """
    else:
        papers_html = '<p style="padding: 15px; color: #999; text-align: center;">No papers tagged yet</p>'

    return JsonResponse(
        {
            "success": True,
            "tag_name": tag.name,
            "tag_id": tag.id,
            "papers_html": papers_html,
        }
    )
