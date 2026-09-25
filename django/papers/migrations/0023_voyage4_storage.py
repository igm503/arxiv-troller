"""Additive Voyage 4 storage. Reversal intentionally preserves all saved data."""
from django.db import migrations

SQL = '''
CREATE SCHEMA IF NOT EXISTS voyage4;
CREATE TABLE IF NOT EXISTS voyage4.embeddings (
    paper_id bigint PRIMARY KEY,
    abstract_sha text NOT NULL,
    created timestamptz NOT NULL,
    categories varchar[] NOT NULL,
    vector halfvec(2048) NOT NULL,
    bits bit(2048) NOT NULL,
    archive_path text NOT NULL,
    archive_row integer NOT NULL,
    vector_sha text NOT NULL,
    saved_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS v4_created_idx ON voyage4.embeddings(created);
CREATE INDEX IF NOT EXISTS v4_categories_idx ON voyage4.embeddings USING gin(categories);
CREATE TABLE IF NOT EXISTS voyage4.rolling30 (
    paper_id bigint PRIMARY KEY,
    abstract_sha text NOT NULL,
    created timestamptz NOT NULL,
    categories varchar[] NOT NULL,
    vector halfvec(2048) NOT NULL,
    full_vector vector(2048) NOT NULL
);
CREATE INDEX IF NOT EXISTS v4_rolling30_created_idx ON voyage4.rolling30(created);
CREATE INDEX IF NOT EXISTS v4_rolling30_categories_idx ON voyage4.rolling30 USING gin(categories);
CREATE TABLE IF NOT EXISTS voyage4.control (
    key text PRIMARY KEY,
    value jsonb NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);
INSERT INTO voyage4.control(key,value) VALUES('ready','false') ON CONFLICT DO NOTHING;
'''

class Migration(migrations.Migration):
    dependencies=[('papers','0022_add_tsvector')]
    operations=[migrations.RunSQL(SQL,reverse_sql=migrations.RunSQL.noop)]
