from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from django.core.management.base import BaseCommand
import numpy as np
from dotenv import load_dotenv
from voyageai import Client

from papers.models import (
    Paper,
    EmbeddingVoyageHalf2048,
    EmbeddingVoyageHalf256,
    EmbeddingVoyageBit2048,
)
from .limiter import RateLimiter


class Command(BaseCommand):
    help = "Generate embeddings for papers"

    def __init__(self):
        super().__init__()
        self._local = threading.local()
        self.rate_limiter = None  # Will be initialized in handle()

    def add_arguments(self, parser):
        parser.add_argument("--model", default="voyage-3-large", help="Embedding model to use")
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

        # Initialize rate limiter
        self.rate_limiter = RateLimiter(options["rate_limit"])

        papers_queryset = Paper.objects.filter(embeddingvoyagehalf2048__isnull=True).order_by("id")

        total = papers_queryset.count()
        self.stdout.write(f"Processing {total} papers with {num_workers} workers")
        self.stdout.write(f"Rate limit: {options['rate_limit']} calls/second")

        all_ids = list(papers_queryset.values_list("id", flat=True))

        id_chunks = [all_ids[i : i + batch_size] for i in range(0, len(all_ids), batch_size)]

        self.stdout.write(f"Created {len(id_chunks)} batches")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            with tqdm(total=total, desc="Processing papers") as pbar:
                futures = [
                    executor.submit(self.process_batch_by_ids, chunk, model_name, pbar)
                    for chunk in id_chunks
                ]

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        self.stdout.write(f"Batch failed: {e}")

    def process_batch_by_ids(self, id_chunk, model_name, pbar):
        """Process a batch of papers by their IDs"""

        batch = list(Paper.objects.filter(id__in=id_chunk).only("id", "abstract"))

        if not batch:
            return

        client = self.get_client()

        texts = [paper.abstract for paper in batch]

        try:
            self.rate_limiter.acquire()

            embeddings = client.embed(
                texts, model=model_name, input_type=None, output_dimension=2048
            ).embeddings

            embedding_2048_objects = []
            embedding_256_objects = []
            embedding_bit2048_objects = []
            for paper, embedding in zip(batch, embeddings):
                embedding_2048_objects.append(
                    EmbeddingVoyageHalf2048(paper=paper, vector=embedding)
                )
                embedding_256 = embedding[:256]
                norm_256 = np.linalg.norm(embedding_256)
                embedding_256 = embedding_256 / norm_256
                embedding_256_objects.append(
                    EmbeddingVoyageHalf256(paper=paper, vector=embedding_256)
                )
                embedding_bit = "".join("1" if x > 0 else "0" for x in embedding)
                embedding_bit2048_objects.append(
                    EmbeddingVoyageBit2048(paper=paper, vector=embedding_bit)
                )

            EmbeddingVoyageHalf2048.objects.bulk_create(embedding_2048_objects)
            EmbeddingVoyageHalf256.objects.bulk_create(embedding_256_objects)
            EmbeddingVoyageBit2048.objects.bulk_create(embedding_bit2048_objects)
            pbar.update(len(batch))

        except Exception as e:
            self.stdout.write(f"Batch failed: {e}")
            pbar.update(len(batch))
