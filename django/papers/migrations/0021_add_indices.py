from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("papers", "0020_alter_embeddinggeminihalf512_vector_and_more"),
    ]

    operations = [
        migrations.RunSQL(
            sql="""
                CREATE INDEX papers_embeddinggemini512_vector_hnsw_idx 
                ON papers_embeddinggeminihalf512 
                USING hnsw (vector halfvec_l2_ops) 
                WITH (m='32', ef_construction='256');
            """,
            reverse_sql="DROP INDEX IF EXISTS papers_embeddinggemini512_vector_hnsw_idx;",
        ),
        migrations.RunSQL(
            sql="""
                CREATE INDEX papers_embeddingvoyagehalf2048_vector_hnsw_idx 
                ON papers_embeddingvoyagehalf2048 
                USING hnsw (vector halfvec_l2_ops) 
                WITH (m='32', ef_construction='256');
            """,
            reverse_sql="DROP INDEX IF EXISTS papers_embeddingvoyagehalf2048_vector_hnsw_idx;",
        ),
        migrations.RunSQL(
            sql="""
                CREATE INDEX papers_embeddingvoyagehalf256_vector_hnsw_idx 
                ON papers_embeddingvoyagehalf256 
                USING hnsw (vector halfvec_l2_ops) 
                WITH (m='32', ef_construction='256');
            """,
            reverse_sql="DROP INDEX IF EXISTS papers_embeddingvoyagehalf256_vector_hnsw_idx;",
        ),
        migrations.RunSQL(
            sql="""
                CREATE INDEX papers_embeddingvoyagebit2048_vector_hnsw_idx 
                ON papers_embeddingvoyagebit2048 
                USING hnsw (vector bit_hamming_ops) 
                WITH (m='32', ef_construction='256');
            """,
            reverse_sql="DROP INDEX IF EXISTS papers_embeddingvoyagebit2048_vector_hnsw_idx;",
        ),
    ]
