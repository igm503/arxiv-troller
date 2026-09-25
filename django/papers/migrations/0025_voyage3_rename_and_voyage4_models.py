"""Rename the original Voyage models to Voyage 3 and adopt the Voyage 4 tables as regular models.

The existing voyage4.embeddings and voyage4.rolling30 tables (and their HNSW indexes) are moved into
public in place, so nothing is re-embedded or re-indexed. Archive bookkeeping columns, the float32
rescoring column, the ingestion queue triggers and the voyage4 schema are dropped.
"""
import django.contrib.postgres.fields
import django.contrib.postgres.indexes
import django.db.models.deletion
import pgvector.django
from django.db import migrations, models

SQL = """
DROP TRIGGER IF EXISTS voyage4_queue_insert ON public.papers_paper;
DROP TRIGGER IF EXISTS voyage4_queue_update ON public.papers_paper;
DROP FUNCTION IF EXISTS voyage4.queue_paper();
DROP TABLE IF EXISTS voyage4.pending_papers;
DROP TABLE IF EXISTS voyage4.control;

ALTER TABLE voyage4.embeddings SET SCHEMA public;
ALTER TABLE public.embeddings RENAME TO papers_embeddingvoyage4;
ALTER INDEX IF EXISTS public.embeddings_pkey RENAME TO papers_embeddingvoyage4_pkey;
ALTER TABLE papers_embeddingvoyage4 OWNER TO CURRENT_USER;
ALTER TABLE papers_embeddingvoyage4
    DROP COLUMN abstract_sha, DROP COLUMN archive_path, DROP COLUMN archive_row,
    DROP COLUMN vector_sha, DROP COLUMN saved_at;
ALTER TABLE papers_embeddingvoyage4 ADD CONSTRAINT papers_embeddingvoyage4_paper_id_fk
    FOREIGN KEY (paper_id) REFERENCES papers_paper(id) DEFERRABLE INITIALLY DEFERRED;
CREATE INDEX IF NOT EXISTS v4_bits_hnsw ON papers_embeddingvoyage4
    USING hnsw (bits bit_hamming_ops) WITH (m=32, ef_construction=256);

ALTER TABLE voyage4.rolling30 SET SCHEMA public;
ALTER TABLE public.rolling30 RENAME TO papers_embeddingvoyage4recent;
ALTER INDEX IF EXISTS public.rolling30_pkey RENAME TO papers_embeddingvoyage4recent_pkey;
ALTER INDEX IF EXISTS public.v4_rolling30_hnsw RENAME TO v4_recent_hnsw;
ALTER INDEX IF EXISTS public.v4_rolling30_created_idx RENAME TO v4_recent_created_idx;
ALTER INDEX IF EXISTS public.v4_rolling30_categories_idx RENAME TO v4_recent_categories_idx;
ALTER TABLE papers_embeddingvoyage4recent OWNER TO CURRENT_USER;
ALTER TABLE papers_embeddingvoyage4recent DROP COLUMN abstract_sha, DROP COLUMN full_vector;
ALTER TABLE papers_embeddingvoyage4recent ADD CONSTRAINT papers_embeddingvoyage4recent_paper_id_fk
    FOREIGN KEY (paper_id) REFERENCES papers_paper(id) DEFERRABLE INITIALLY DEFERRED;
CREATE INDEX IF NOT EXISTS v4_recent_hnsw ON papers_embeddingvoyage4recent
    USING hnsw (vector halfvec_l2_ops) WITH (m=16, ef_construction=64);

-- RESTRICT: fails if anything unexpected is still in the schema
DROP SCHEMA voyage4;
"""


def embedding_fields():
    return [
        (
            "paper",
            models.OneToOneField(
                on_delete=django.db.models.deletion.CASCADE,
                primary_key=True,
                serialize=False,
                to="papers.paper",
            ),
        ),
        ("vector", pgvector.django.HalfVectorField(dimensions=2048)),
    ]


def filter_fields():
    return [
        ("created", models.DateTimeField()),
        (
            "categories",
            django.contrib.postgres.fields.ArrayField(
                base_field=models.CharField(max_length=50), size=None
            ),
        ),
    ]


class Migration(migrations.Migration):
    dependencies = [
        ("papers", "0024_voyage4_ingestion_queue"),
    ]

    operations = [
        migrations.RenameModel("EmbeddingVoyageHalf2048", "EmbeddingVoyage3Half2048"),
        migrations.RenameModel("EmbeddingVoyageHalf256", "EmbeddingVoyage3Half256"),
        migrations.RenameModel("EmbeddingVoyageBit2048", "EmbeddingVoyage3Bit2048"),
        migrations.SeparateDatabaseAndState(
            database_operations=[migrations.RunSQL(SQL)],
            state_operations=[
                migrations.CreateModel(
                    name="EmbeddingVoyage4",
                    fields=embedding_fields()
                    + [("bits", pgvector.django.BitField(length=2048))]
                    + filter_fields(),
                    options={
                        "indexes": [
                            pgvector.django.HnswIndex(
                                ef_construction=256,
                                fields=["bits"],
                                m=32,
                                name="v4_bits_hnsw",
                                opclasses=["bit_hamming_ops"],
                            ),
                            models.Index(fields=["created"], name="v4_created_idx"),
                            django.contrib.postgres.indexes.GinIndex(
                                fields=["categories"], name="v4_categories_idx"
                            ),
                        ],
                    },
                ),
                migrations.CreateModel(
                    name="EmbeddingVoyage4Recent",
                    fields=embedding_fields() + filter_fields(),
                    options={
                        "indexes": [
                            pgvector.django.HnswIndex(
                                ef_construction=64,
                                fields=["vector"],
                                m=16,
                                name="v4_recent_hnsw",
                                opclasses=["halfvec_l2_ops"],
                            ),
                            models.Index(fields=["created"], name="v4_recent_created_idx"),
                            django.contrib.postgres.indexes.GinIndex(
                                fields=["categories"], name="v4_recent_categories_idx"
                            ),
                        ],
                    },
                ),
            ],
        ),
    ]
