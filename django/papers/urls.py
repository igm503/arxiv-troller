from django.urls import path
from django.views.generic import TemplateView
from . import views, views_auth, ingestion_api

app_name = "papers"

urlpatterns = [
    path("api/", ingestion_api.api, name="api"),
    path("api/ingestion/", ingestion_api.api, name="ingestion_api"),
    path("llms.txt", TemplateView.as_view(template_name="papers/llms.txt", content_type="text/plain; charset=utf-8"),
         name="llms_txt"),
    path("", views.search, name="search"),
    path("login/", views_auth.login_view, name="login"),
    path("logout/", views_auth.logout_view, name="logout"),
    path("ajax/add-to-tag/", views_auth.add_to_tag, name="add_to_tag"),
    path("ajax/remove-from-tag/", views_auth.remove_from_tag, name="remove_from_tag"),
    path("ajax/rename-tag/", views_auth.rename_tag, name="rename_tag"),
    path("ajax/delete-tag/", views_auth.delete_tag, name="delete_tag"),
    path("ajax/get-tag-drawer/", views_auth.get_tag_drawer, name="get_tag_drawer"),
    path("paper/<int:paper_id>/", views.paper_detail, name="detail"),
]
