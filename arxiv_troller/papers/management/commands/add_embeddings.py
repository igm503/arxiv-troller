from tqdm import tqdm
from google import genai
from google.genai import types
from django.core.management.base import BaseCommand
import numpy as np
from dotenv import load_dotenv

from papers.models import Paper, Embedding


class Command(BaseCommand):
    help = "Generate embeddings for papers"

    def add_arguments(self, parser):
        parser.add_argument(
            "--model", default="gemini-embedding-001", help="Embedding model to use"
        )
        parser.add_argument("--batch-size", type=int, default=64, help="Batch size for processing")

    def handle(self, *args, **options):
        load_dotenv()
        model_name = options["model"]
        batch_size = options["batch_size"]

        client = genai.Client()

        # Get papers without embeddings for this model
        papers_without_embeddings = Paper.objects.exclude(
            embedding__model_name=model_name, embedding__embedding_type="abstract"
        )

        total = papers_without_embeddings.count()
        self.stdout.write(f"Processing {total} papers")

        for i in tqdm(range(0, total, batch_size)):
            batch = list(papers_without_embeddings[i : i + batch_size])
            texts = [paper.abstract for paper in batch]

            embeddings = [
                np.array(e.values)
                for e in client.models.embed_content(
                    model=model_name,
                    contents=texts,
                    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
                ).embeddings
            ]

            # Save to database
            embedding_objects = []
            for paper, embedding in zip(batch, embeddings):
                embedding_objects.append(
                    Embedding(
                        paper=paper,
                        model_name=model_name,
                        embedding_type="abstract",
                        vector=embedding.tolist(),
                    )
                )

            Embedding.objects.bulk_create(embedding_objects, ignore_conflicts=True)
