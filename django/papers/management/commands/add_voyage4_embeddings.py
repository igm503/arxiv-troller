from datetime import timedelta
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from django.core.management.base import BaseCommand
from django.db import connection
from django.utils import timezone
from dotenv import load_dotenv
from voyageai import Client

from papers.models import Paper, EmbeddingVoyage4, EmbeddingVoyage4Recent
from .limiter import RateLimiter


class Command(BaseCommand):
    help = "Generate Voyage 4 embeddings for new papers and refresh the recent-paper index"

    def __init__(self):
        super().__init__()
        self._local = threading.local()
        self.rate_limiter = None  # Will be initialized in handle()

    def add_arguments(self, parser):
        parser.add_argument("--model", default="voyage-4-large", help="Embedding model to use")
        parser.add_argument("--batch-size", type=int, default=128, help="Batch size for processing")
        parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
        parser.add_argument("--rate-limit", type=float, default=1.0, help="API calls per second")

    def get_client(self):
        """Get or create a client for the current thread"""
        if not hasattr(self._local, "client"):
            self._local.client = Client()
        return self._local.client

    def handle(self, *args, **options):
        load_dotenv()
        model_name = options["model"]
        batch_size = options["batch_size"]
        num_workers = options["workers"]

        self.rate_limiter = RateLimiter(options["rate_limit"])

        # The API rejects empty inputs, so blank abstracts are never embedded
        papers_queryset = (
            Paper.objects.filter(embeddingvoyage4__isnull=True)
            .exclude(abstract__regex=r"^\s*$")
            .order_by("id")
        )

        all_ids = list(papers_queryset.values_list("id", flat=True))
        self.stdout.write(f"Processing {len(all_ids)} papers with {num_workers} workers")

        id_chunks = [all_ids[i : i + batch_size] for i in range(0, len(all_ids), batch_size)]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            with tqdm(total=len(all_ids), desc="Processing papers") as pbar:
                futures = [
                    executor.submit(self.process_batch_by_ids, chunk, model_name, pbar)
                    for chunk in id_chunks
                ]

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        self.stdout.write(f"Batch failed: {e}")

        self.refresh_recent()

    def process_batch_by_ids(self, id_chunk, model_name, pbar):
        """Process a batch of papers by their IDs"""

        batch = list(
            Paper.objects.filter(id__in=id_chunk).only("id", "abstract", "created", "categories")
        )

        if not batch:
            return

        client = self.get_client()

        texts = [paper.abstract for paper in batch]

        try:
            self.rate_limiter.acquire()

            embeddings = client.embed(
                texts, model=model_name, input_type=None, output_dimension=2048
            ).embeddings

            EmbeddingVoyage4.objects.bulk_create(
                [
                    EmbeddingVoyage4(
                        paper=paper,
                        vector=embedding,
                        bits="".join("1" if x > 0 else "0" for x in embedding),
                        created=paper.created,
                        categories=paper.categories,
                    )
                    for paper, embedding in zip(batch, embeddings)
                ]
            )
            pbar.update(len(batch))

        except Exception as e:
            self.stdout.write(f"Batch failed: {e}")
            pbar.update(len(batch))

    def refresh_recent(self):
        """Drop papers that aged out of the recent index and copy in newly embedded ones."""
        cutoff = timezone.now() - timedelta(days=EmbeddingVoyage4Recent.RECENT_DAYS)
        expired, _ = EmbeddingVoyage4Recent.objects.filter(created__lt=cutoff).delete()
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {EmbeddingVoyage4Recent._meta.db_table} (paper_id, vector, created, categories)
                SELECT paper_id, vector, created, categories FROM {EmbeddingVoyage4._meta.db_table}
                WHERE created >= %s
                ON CONFLICT (paper_id) DO NOTHING
                """,
                [cutoff],
            )
            added = cursor.rowcount
        self.stdout.write(f"Recent index: {added} added, {expired} expired")
