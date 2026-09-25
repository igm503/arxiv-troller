"""Queue metadata/abstract changes without altering existing ingestion or embeddings."""
from django.db import migrations

class Migration(migrations.Migration):
    dependencies=[('papers','0023_voyage4_storage')]
    operations=[migrations.RunSQL('''
      CREATE TABLE IF NOT EXISTS voyage4.pending_papers (
        paper_id bigint PRIMARY KEY, queued_at timestamptz NOT NULL DEFAULT clock_timestamp()
      );
      CREATE OR REPLACE FUNCTION voyage4.queue_paper() RETURNS trigger
      LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
      BEGIN
        INSERT INTO voyage4.pending_papers(paper_id,queued_at) VALUES(NEW.id,clock_timestamp())
        ON CONFLICT(paper_id) DO UPDATE SET queued_at=excluded.queued_at;
        RETURN NEW;
      END $$;
      CREATE TRIGGER voyage4_queue_insert AFTER INSERT ON public.papers_paper
        FOR EACH ROW EXECUTE FUNCTION voyage4.queue_paper();
      CREATE TRIGGER voyage4_queue_update AFTER UPDATE OF abstract,created,categories ON public.papers_paper
        FOR EACH ROW WHEN (OLD.abstract IS DISTINCT FROM NEW.abstract OR OLD.created IS DISTINCT FROM NEW.created
          OR OLD.categories IS DISTINCT FROM NEW.categories) EXECUTE FUNCTION voyage4.queue_paper();
    ''',reverse_sql=migrations.RunSQL.noop)]
