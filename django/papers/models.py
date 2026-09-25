from django.db import models
from django.contrib.postgres.fields import ArrayField
from django.contrib.postgres.search import SearchVectorField
from django.contrib.auth.models import User
from django.contrib.postgres.indexes import GinIndex
from pgvector.django import HalfVectorField, BitField, HnswIndex


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
    search_vector = SearchVectorField(null=True)
    categories = ArrayField(models.CharField(max_length=50), default=list, blank=True)
    updated = models.DateTimeField(null=True, blank=True)
    authors = models.ManyToManyField(Author, through="PaperAuthor")

    def __str__(self):
        return f"{self.arxiv_id}: {self.title[:100]}"

    def delete_embeddings(self):
        """Drop every embedding of this paper so it is re-embedded from the current abstract."""
        for model in EMBEDDING_MODELS:
            model.objects.filter(paper=self).delete()

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


class EmbeddingVoyage3Half2048(models.Model):
    paper = models.OneToOneField(Paper, on_delete=models.CASCADE, primary_key=True)
    vector = HalfVectorField(dimensions=2048)
    created_at = models.DateTimeField(auto_now_add=True)


class EmbeddingVoyage3Half256(models.Model):
    paper = models.OneToOneField(Paper, on_delete=models.CASCADE, primary_key=True)
    vector = HalfVectorField(dimensions=256)
    created_at = models.DateTimeField(auto_now_add=True)


class EmbeddingVoyage3Bit2048(models.Model):
    paper = models.OneToOneField(Paper, on_delete=models.CASCADE, primary_key=True)
    vector = BitField(length=2048)
    created_at = models.DateTimeField(auto_now_add=True)


class EmbeddingVoyage4(models.Model):
    paper = models.OneToOneField(Paper, on_delete=models.CASCADE, primary_key=True)
    vector = HalfVectorField(dimensions=2048)
    bits = BitField(length=2048)
    # Copied from the paper so filtered vector search never joins papers_paper
    created = models.DateTimeField()
    categories = ArrayField(models.CharField(max_length=50))

    class Meta:
        indexes = [
            HnswIndex(
                name="v4_bits_hnsw",
                fields=["bits"],
                m=32,
                ef_construction=256,
                opclasses=["bit_hamming_ops"],
            ),
            models.Index(fields=["created"], name="v4_created_idx"),
            GinIndex(fields=["categories"], name="v4_categories_idx"),
        ]


class EmbeddingVoyage4Recent(models.Model):
    """Papers from the last RECENT_DAYS, with their own HNSW graph for fast short-window search."""

    RECENT_DAYS = 31

    paper = models.OneToOneField(Paper, on_delete=models.CASCADE, primary_key=True)
    vector = HalfVectorField(dimensions=2048)
    created = models.DateTimeField()
    categories = ArrayField(models.CharField(max_length=50))

    class Meta:
        indexes = [
            HnswIndex(
                name="v4_recent_hnsw",
                fields=["vector"],
                m=16,
                ef_construction=64,
                opclasses=["halfvec_l2_ops"],
            ),
            models.Index(fields=["created"], name="v4_recent_created_idx"),
            GinIndex(fields=["categories"], name="v4_recent_categories_idx"),
        ]


EMBEDDING_MODELS = [
    EmbeddingGeminiHalf3072,
    EmbeddingGeminiHalf512,
    EmbeddingVoyage3Half2048,
    EmbeddingVoyage3Half256,
    EmbeddingVoyage3Bit2048,
    EmbeddingVoyage4,
    EmbeddingVoyage4Recent,
]


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
