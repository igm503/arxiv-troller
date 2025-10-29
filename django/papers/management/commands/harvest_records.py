import traceback
from datetime import datetime, timedelta, timezone

from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import xmltodict
from django.core.management.base import BaseCommand
from django.db.models import Max

from papers.models import Paper, Author, PaperAuthor


class Command(BaseCommand):
    help = "Harvest arXiv metadata and save to database"

    def add_arguments(self, parser):
        parser.add_argument(
            "--everything",
            action="store_false",
            dest="recent_only",
            default=True,
            help="Harvest everything, not just recent updates",
        )

    def handle(self, *args, **options):
        total = 0
        base_url = "https://oaipmh.arxiv.org/oai?verb=ListRecords"
        url = f"{base_url}&metadataprefix=arxiv"
        last_date = None

        recent = options["recent_only"]
        if recent:
            most_recent = Paper.objects.aggregate(Max("created"))["created__max"]
            last_date = most_recent.replace(tzinfo=None)
            last_date_str = last_date.strftime("%Y-%m-%d")
            url = f"{base_url}&metadataPrefix=arXiv&from={last_date_str}"

        with tqdm() as pbar:
            while True:
                try:
                    response = requests.get(url)
                    xml_response = response.text.strip()
                    metadata = xmltodict.parse(xml_response)

                    for item in metadata["OAI-PMH"]["ListRecords"]["record"]:
                        arxiv_data = item["metadata"]["arXiv"]
                        created = self.save_paper(arxiv_data)

                        last_updated_str = arxiv_data.get("updated")
                        if last_updated_str is None:
                            last_updated_date = datetime.strptime(arxiv_data["created"], "%Y-%m-%d")
                        else:
                            last_updated_date = datetime.strptime(last_updated_str, "%Y-%m-%d")

                        if last_date is None or last_updated_date > last_date:
                            last_date = last_updated_date

                        pbar.update(1)
                        if created:
                            total += 1

                    pbar.set_description(f"Total: {total}, last date: {last_date}")

                    token = self.get_resumption_token_bs(xml_response)
                    if token is None:
                        if last_date is not None and last_date < datetime.now() - timedelta(days=7):
                            self.stdout.write(f"Restarting from {last_date}")
                            date_string = last_date.strftime("%Y-%m-%d")
                            url = f"{base_url}&metadataPrefix=arXiv&from={date_string}"
                        else:
                            break
                    else:
                        url = f"{base_url}&resumptionToken={token}"

                except Exception as e:
                    self.stdout.write(f"Error: {e}")
                    self.stdout.write(traceback.format_exc())  # Prints full traceback to stdout
                    if last_date is not None and last_date < datetime.now() - timedelta(days=7):
                        self.stdout.write(f"Restarting from {last_date}")
                        date_string = last_date.strftime("%Y-%m-%d")
                        url = f"{base_url}&metadataPrefix=arXiv&from={date_string}"
                    else:
                        break

        self.stdout.write(f"Harvested {total} records")

    def get_resumption_token_bs(self, xml_response):
        soup = BeautifulSoup(xml_response, "xml")
        token_elem = soup.find("resumptionToken")
        return token_elem.get_text() if token_elem and token_elem.get_text().strip() else None

    def save_paper(self, data):
        categories = (
            data["categories"].strip().split(" ")
            if isinstance(data["categories"], str)
            else data["categories"]
        )

        paper, created = Paper.objects.get_or_create(
            arxiv_id=data["id"],
            defaults={
                "title": data["title"],
                "abstract": data["abstract"],
                "created": datetime.fromisoformat(data["created"]).replace(tzinfo=timezone.utc),
                "updated": (
                    datetime.fromisoformat(data["updated"]).replace(tzinfo=timezone.utc)
                    if data.get("updated")
                    else None
                ),
                "categories": categories,
            },
        )

        if created:
            if isinstance(data["authors"], list):
                authors = data["authors"]
            elif isinstance(data["authors"], dict):
                if "author" in data["authors"] and isinstance(data["authors"]["author"], list):
                    authors = data["authors"]["author"]
                else:
                    authors = [data["authors"]["author"]]
            for i, author_data in enumerate(authors):
                author, _ = Author.objects.get_or_create(
                    keyname=author_data["keyname"],
                    forenames=author_data.get("forenames"),
                )
                PaperAuthor.objects.create(paper=paper, author=author, order=i)
            return True
