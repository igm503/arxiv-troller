from django.db import models
from django.contrib.postgres.fields import ArrayField
from django.contrib.auth.models import User
from pgvector.django import HalfVectorField, BitField


class Author(models.Model):
    keyname = models.CharField(max_length=200)
    forenames = models.CharField(max_length=200, null=True, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["keyname", "forenames"],
                name="papers_author_keyname_forenames_unique",
                nulls_distinct=False,
            )
        ]

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


class EmbeddingGeminiHalf3072(models.Model):
    paper = models.OneToOneField(Paper, on_delete=models.CASCADE, primary_key=True)
    vector = HalfVectorField(dimensions=3072)
    created_at = models.DateTimeField(auto_now_add=True)


class EmbeddingGeminiHalf512(models.Model):
    paper = models.OneToOneField(Paper, on_delete=models.CASCADE, primary_key=True)
    vector = HalfVectorField(dimensions=512)
    created_at = models.DateTimeField(auto_now_add=True)


class EmbeddingVoyageHalf2048(models.Model):
    paper = models.OneToOneField(Paper, on_delete=models.CASCADE, primary_key=True)
    vector = HalfVectorField(dimensions=2048)
    created_at = models.DateTimeField(auto_now_add=True)


class EmbeddingVoyageHalf256(models.Model):
    paper = models.OneToOneField(Paper, on_delete=models.CASCADE, primary_key=True)
    vector = HalfVectorField(dimensions=256)
    created_at = models.DateTimeField(auto_now_add=True)


class EmbeddingVoyageBit2048(models.Model):
    paper = models.OneToOneField(Paper, on_delete=models.CASCADE, primary_key=True)
    vector = BitField(length=2048)
    created_at = models.DateTimeField(auto_now_add=True)


class Tag(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="tags")
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("user", "name")
        ordering = ["name"]

    def __str__(self):
        return f"{self.user.username} - {self.name}"


class TaggedPaper(models.Model):
    tag = models.ForeignKey(Tag, on_delete=models.CASCADE, related_name="tagged_papers")
    paper = models.ForeignKey(
        Paper, on_delete=models.CASCADE
    )  # No quotes needed since Paper is in same file
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("tag", "paper")
        ordering = ["-added_at"]
