from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from google import genai
from google.genai import types
from django.core.management.base import BaseCommand
import numpy as np
from dotenv import load_dotenv

from papers.models import Paper, EmbeddingGeminiHalf3072, EmbeddingGeminiHalf512
from .limiter import RateLimiter


class Command(BaseCommand):
    help = "Generate embeddings for papers"

    def __init__(self):
        super().__init__()
        self._local = threading.local()
        self.rate_limiter = None

    def add_arguments(self, parser):
        parser.add_argument(
            "--model", default="gemini-embedding-001", help="Embedding model to use"
        )
        parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
        parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
        parser.add_argument("--rate-limit", type=float, default=0.5, help="API calls per second")

    def get_client(self):
        """Get or create a client for the current thread"""
        if not hasattr(self._local, "client"):
            self._local.client = genai.Client()
        return self._local.client

    def handle(self, *args, **options):
        load_dotenv()
        model_name = options["model"]
        batch_size = options["batch_size"]
        num_workers = options["workers"]

        # Initialize rate limiter
        self.rate_limiter = RateLimiter(options["rate_limit"])

        papers_queryset = Paper.objects.filter(embeddinggeminihalf3072__isnull=True).order_by("id")

        total = papers_queryset.count()
        self.stdout.write(f"Processing {total} papers with {num_workers} workers")
        self.stdout.write(f"Rate limit: {options['rate_limit']} calls/second")

        # Pre-fetch all IDs (this is fast)
        all_ids = list(papers_queryset.values_list("id", flat=True))

        # Create ID chunks
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

            embeddings = [
                np.array(e.values)
                for e in client.models.embed_content(
                    model=model_name,
                    contents=texts,
                    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
                ).embeddings
            ]

            embedding_objects = []
            embedding_reduced_objects = []
            for paper, embedding in zip(batch, embeddings):
                embedding_objects.append(
                    EmbeddingGeminiHalf3072(paper=paper, vector=embedding.tolist())
                )
                embedding_512 = embedding[:512]
                norm_512 = np.linalg.norm(embedding_512)
                if norm_512 > 0:
                    embedding_512 = embedding_512 / norm_512
                embedding_reduced_objects.append(
                    EmbeddingGeminiHalf512(paper=paper, vector=embedding_512.tolist())
                )

            EmbeddingGeminiHalf3072.objects.bulk_create(embedding_objects)
            EmbeddingGeminiHalf512.objects.bulk_create(embedding_reduced_objects)
            pbar.update(len(batch))

        except Exception as e:
            self.stdout.write(f"Batch failed: {e}")
            pbar.update(len(batch))
