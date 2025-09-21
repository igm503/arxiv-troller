from django.core.management.base import BaseCommand
from papers.models import Paper, Citation
import requests
import time
from tqdm import tqdm


class Command(BaseCommand):
    help = "Harvest citations from Semantic Scholar"

    def handle(self, *args, **options):
        papers = Paper.objects.filter(citations__isnull=True)  # Papers without citation data

        for paper in tqdm(papers):
            try:
                self.get_citations_for_paper(paper)
            except Exception as e:
                self.stdout.write(f"Error processing {paper.arxiv_id}: {e}")

    def get_citations_for_paper(self, paper):
        url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{paper.arxiv_id}/citations"
        params = {"fields": "citingPaper.paperId,citingPaper.externalIds", "limit": 500}

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()

            for citation_data in data.get("data", []):
                citing_paper_info = citation_data["citingPaper"]
                external_ids = citing_paper_info.get("externalIds", {})

                if external_ids.get("ArXiv"):
                    citing_arxiv_id = external_ids["ArXiv"]
                    try:
                        citing_paper = Paper.objects.get(arxiv_id=citing_arxiv_id)
                        Citation.objects.get_or_create(citing_paper=citing_paper, cited_paper=paper)
                    except Paper.DoesNotExist:
                        continue  # Citing paper not in our database yet
