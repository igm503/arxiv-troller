from django.urls import path

from . import views, views_auth

app_name = "papers"
urlpatterns = [
    path("", views.unified_search_view, name="search"),
    path("login/", views_auth.login_view, name="login"),
    path("logout/", views_auth.logout_view, name="logout"),
    path("tag/<int:tag_id>/", views.unified_search_view, name="tag"),
    path("tag/<int:tag_id>/settings/", views_auth.tag_settings, name="tag_settings"),
    path("ajax/add-to-tag/", views_auth.add_to_tag, name="add_to_tag"),
    path("ajax/remove-from-tag/", views_auth.remove_from_tag, name="remove_from_tag"),
    path(
        "ajax/remove-from-search/",
        views_auth.remove_from_search,
        name="remove_from_search",
    ),
    path(
        "ajax/unremove-from-search/",
        views_auth.unremove_from_search,
        name="unremove_from_search",
    ),
    path("ajax/rename-tag/", views_auth.rename_tag, name="rename_tag"),
    path("paper/<int:paper_id>/", views.paper_detail, name="detail"),
    path("paper/<int:paper_id>/similar/", views.similar_papers, name="similar"),
]
