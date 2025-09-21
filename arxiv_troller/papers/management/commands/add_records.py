from datetime import datetime, timezone
import warnings
import os

import msgpack
from tqdm import tqdm
from django.core.management.base import BaseCommand
from django.db import transaction

from papers.models import Paper, Author, PaperAuthor


class Command(BaseCommand):
    help = "Load existing msgpack records into Django database"

    def add_arguments(self, parser):
        parser.add_argument(
            "--records-dir",
            default="records",
            help="Directory containing msgpack files",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=10000,
            help="Batch size for database inserts",
        )

    def handle(self, *args, **options):
        # Suppress timezone warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="django.db.models.fields")

        records_dir = options["records_dir"]
        batch_size = options["batch_size"]

        files = [f for f in os.listdir(records_dir) if f.endswith(".msgpack")]
        files.sort()
        files = files[44:]

        # Count total records
        total_records = 0
        self.stdout.write("Counting records...")
        for file in files:
            with open(os.path.join(records_dir, file), "rb") as f:
                data = msgpack.unpack(f)
                total_records += len(data)

        self.stdout.write(f"Found {total_records} total records")

        processed_records = 0
        skipped_records = 0

        with tqdm(total=total_records, desc="Loading records") as pbar:
            for file in files:
                with open(os.path.join(records_dir, file), "rb") as f:
                    data = msgpack.unpack(f)

                # Process in large batches
                for i in range(0, len(data), batch_size):
                    batch = data[i : i + batch_size]
                    created, skipped = self.process_batch(batch)
                    processed_records += created
                    skipped_records += skipped

                    pbar.update(len(batch))
                    pbar.set_postfix(
                        {
                            "created": processed_records,
                            "skipped": skipped_records,
                            "rate": (
                                f"{processed_records / (pbar.n / total_records * 100 + 0.01):.0f}/s"
                                if pbar.n > 0
                                else "0/s"
                            ),
                        }
                    )

        self.stdout.write(
            f"Loaded {processed_records} records, skipped {skipped_records} duplicates"
        )

    def process_batch(self, batch):
        with transaction.atomic():
            # Step 1: Filter out papers that already exist
            arxiv_ids = [item["id"] for item in batch]
            existing_paper_ids = set(
                Paper.objects.filter(arxiv_id__in=arxiv_ids).values_list("arxiv_id", flat=True)
            )

            new_items = [item for item in batch if item["id"] not in existing_paper_ids]
            skipped_count = len(batch) - len(new_items)

            if not new_items:
                return 0, skipped_count

            # Step 2: Collect all unique authors from new papers
            all_author_data = {}  # (keyname, forenames) -> author_data
            for item in new_items:
                for author_data in item["authors"]:
                    key = (author_data["keyname"], author_data.get("forenames"))
                    all_author_data[key] = author_data

            # Step 3: Create any missing authors in bulk
            existing_authors = Author.objects.filter(
                keyname__in=[key[0] for key in all_author_data.keys()]
            )
            existing_author_keys = {(a.keyname, a.forenames) for a in existing_authors}

            new_authors = []
            for key, author_data in all_author_data.items():
                if key not in existing_author_keys:
                    new_authors.append(
                        Author(
                            keyname=author_data["keyname"],
                            forenames=author_data.get("forenames"),
                        )
                    )

            if new_authors:
                Author.objects.bulk_create(new_authors, ignore_conflicts=True)

            # Step 4: Build author lookup dictionary
            all_authors = Author.objects.filter(
                keyname__in=[key[0] for key in all_author_data.keys()]
            )
            author_lookup = {(a.keyname, a.forenames): a for a in all_authors}

            # Step 5: Create all papers in bulk
            papers_to_create = []
            for item in new_items:
                created_date = datetime.strptime(item["created"], "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
                updated_date = None
                if item.get("updated"):
                    updated_date = datetime.strptime(item["updated"], "%Y-%m-%d").replace(
                        tzinfo=timezone.utc
                    )

                categories = item["categories"]
                if not isinstance(categories, list):
                    categories = [categories]

                papers_to_create.append(
                    Paper(
                        arxiv_id=item["id"],
                        title=item["title"],
                        abstract=item["abstract"],
                        created=created_date,
                        updated=updated_date,
                        categories=categories,
                    )
                )

            Paper.objects.bulk_create(papers_to_create, ignore_conflicts=True)

            # Step 6: Get the newly created papers and create author relationships
            new_papers = {
                p.arxiv_id: p
                for p in Paper.objects.filter(arxiv_id__in=[item["id"] for item in new_items])
            }

            paper_author_relationships = []
            for item in new_items:
                paper = new_papers[item["id"]]
                for order, author_data in enumerate(item["authors"]):
                    author_keyname = author_data["keyname"]
                    if isinstance(author_keyname, str):
                        author_keyname = author_keyname[:200]
                    author_forenames = author_data.get("forenames")
                    if isinstance(author_forenames, str):
                        author_forenames = author_forenames[:200]
                    author_key = (author_keyname, author_forenames)
                    author = author_lookup[author_key]
                    paper_author_relationships.append(
                        PaperAuthor(paper=paper, author=author, order=order)
                    )

            PaperAuthor.objects.bulk_create(paper_author_relationships, ignore_conflicts=True)

            return len(new_items), skipped_count
