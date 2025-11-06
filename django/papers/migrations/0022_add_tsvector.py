from django.db import migrations
from django.contrib.postgres.indexes import GinIndex
from django.contrib.postgres.search import SearchVectorField


def populate_search_vector(apps, schema_editor):
    """Populate the search_vector field using raw SQL"""
    schema_editor.execute(
        """
        UPDATE papers_paper
        SET search_vector = 
            setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
            setweight(to_tsvector('english', coalesce(abstract, '')), 'B')
    """
    )


def create_search_trigger(apps, schema_editor):
    """Create custom trigger function and trigger to automatically update search_vector with weights"""
    schema_editor.execute(
        """
        CREATE FUNCTION papers_paper_search_vector_trigger() RETURNS trigger AS $$
        BEGIN
            NEW.search_vector := 
                setweight(to_tsvector('english', coalesce(NEW.title, '')), 'A') ||
                setweight(to_tsvector('english', coalesce(NEW.abstract, '')), 'B');
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql;
    """
    )

    schema_editor.execute(
        """
        CREATE TRIGGER papers_paper_search_vector_update
        BEFORE INSERT OR UPDATE OF title, abstract
        ON papers_paper
        FOR EACH ROW
        EXECUTE FUNCTION papers_paper_search_vector_trigger();
    """
    )


def drop_search_trigger(apps, schema_editor):
    """Drop the search_vector trigger and function"""
    schema_editor.execute(
        """
        DROP TRIGGER IF EXISTS papers_paper_search_vector_update ON papers_paper;
    """
    )
    schema_editor.execute(
        """
        DROP FUNCTION IF EXISTS papers_paper_search_vector_trigger();
    """
    )


class Migration(migrations.Migration):
    dependencies = [
        ("papers", "0021_add_indices"),
    ]

    operations = [
        migrations.AddField(
            model_name="paper",
            name="search_vector",
            field=SearchVectorField(null=True),
        ),
        migrations.RunPython(
            populate_search_vector,
            reverse_code=migrations.RunPython.noop,
        ),
        migrations.AddIndex(
            model_name="paper",
            index=GinIndex(
                fields=["search_vector"],
                name="papers_search_vector_idx",
            ),
        ),
        migrations.RunPython(
            create_search_trigger,
            reverse_code=drop_search_trigger,
        ),
    ]
