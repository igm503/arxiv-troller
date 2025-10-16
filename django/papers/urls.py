from django.urls import path
from . import views, views_auth

app_name = "papers"

urlpatterns = [
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
