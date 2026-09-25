from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login as auth_login, logout as auth_logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.template.loader import render_to_string
from django.views.decorators.http import require_POST
from .models import Tag, TaggedPaper, Paper
from .views import tag_drawer_papers


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

    # Check if another tag with this name already exists for this user
    if Tag.objects.filter(user=request.user, name=new_name).exclude(id=tag_id).exists():
        return JsonResponse(
            {"error": f"You already have a tag named '{new_name}'"}, status=400
        )

    tag.name = new_name
    tag.save()

    return JsonResponse({"success": True, "new_name": new_name})


@login_required
@require_POST
def delete_tag(request):
    """Delete a tag and all its associations"""
    tag_id = request.POST.get("tag_id")

    if not tag_id:
        return JsonResponse({"error": "Missing tag_id"}, status=400)

    tag = get_object_or_404(Tag, id=tag_id, user=request.user)
    tag_name = tag.name

    # Django will automatically delete associated TaggedPaper entries due to CASCADE
    tag.delete()

    return JsonResponse({"success": True, "deleted_tag_name": tag_name})


@login_required
def get_tag_drawer(request):
    """AJAX endpoint to get tag drawer HTML for switching tags"""
    tag_id = request.GET.get("tag_id")
    if not tag_id:
        return JsonResponse({"error": "Missing tag_id"}, status=400)
    tag = get_object_or_404(Tag, id=tag_id, user=request.user)

    papers_html = render_to_string(
        "papers/tag_drawer_papers.html",
        {"tagged_papers": tag_drawer_papers(tag, request.GET.get("sort", "added")), "current_tag": tag},
        request=request,
    )

    return JsonResponse(
        {
            "success": True,
            "tag_name": tag.name,
            "tag_id": tag.id,
            "papers_html": papers_html,
        }
    )
