from django.db import models
from django.contrib.postgres.fields import ArrayField
from pgvector.django import HalfVectorField

class Author(models.Model):
    keyname = models.CharField(max_length=200)
    forenames = models.CharField(max_length=200, null=True, blank=True)
    
    def __str__(self):
        if self.forenames:
            return f"{self.forenames} {self.keyname}"
        return self.keyname

class Paper(models.Model):
    arxiv_id = models.CharField(max_length=50, unique=True, db_index=True)
    created = models.DateTimeField()
    title = models.TextField()
    abstract = models.TextField()
    categories = ArrayField(models.CharField(max_length=50), default=list, blank=True)
    updated = models.DateTimeField(null=True, blank=True)
    authors = models.ManyToManyField(Author, through="PaperAuthor")
    
    def __str__(self):
        return f"{self.arxiv_id}: {self.title[:100]}"
    
    class Meta:
        ordering = ["-created"]

class PaperAuthor(models.Model):
    paper = models.ForeignKey(Paper, on_delete=models.CASCADE)
    author = models.ForeignKey(Author, on_delete=models.CASCADE)
    order = models.PositiveIntegerField()
    
    class Meta:
        ordering = ["order"]
        unique_together = ("paper", "author", "order")

class Citation(models.Model):
    citing_paper = models.ForeignKey(Paper, on_delete=models.CASCADE, related_name="references")
    cited_paper = models.ForeignKey(Paper, on_delete=models.CASCADE, related_name="citations")
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ("citing_paper", "cited_paper")

class Embedding(models.Model):
    paper = models.ForeignKey(Paper, on_delete=models.CASCADE)
    model_name = models.CharField(max_length=100)
    embedding_type = models.CharField(max_length=50)
    vector = HalfVectorField(dimensions=3072)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ("paper", "model_name", "embedding_type")
        indexes = [
            models.Index(fields=["model_name", "embedding_type"]),
        ]
