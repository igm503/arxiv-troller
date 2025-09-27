from django.urls import path
from . import views

app_name = 'papers'

urlpatterns = [
    path('', views.search_papers, name='search'),
    path('paper/<int:paper_id>/', views.paper_detail, name='detail'),
    path('paper/<int:paper_id>/similar/', views.similar_papers, name='similar'),
]
