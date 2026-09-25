from datetime import timedelta
from unittest.mock import Mock, patch

import numpy as np
from django.test import SimpleTestCase, TestCase
from django.utils import timezone
from requests import Response
from requests.utils import get_encoding_from_headers

from papers import voyage4_search
from papers.management.commands.add_voyage4_embeddings import Command as Voyage4Command
from papers.management.commands.harvest_records import Command
from papers.models import (
    EMBEDDING_MODELS,
    EmbeddingVoyage3Bit2048,
    EmbeddingVoyage4,
    EmbeddingVoyage4Recent,
    Paper,
)


class HarvestEncodingTests(SimpleTestCase):
    def test_harvest_preserves_utf8_math_in_xml_without_charset(self):
        response = Response()
        response.status_code = 200
        response.headers["Content-Type"] = "text/xml"
        response.encoding = get_encoding_from_headers(response.headers)
        response._content = b"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH><ListRecords><record><metadata><arXiv>
<id>2609.26532</id><created>2099-01-01</created>
<abstract>External BFCL and $\xcf\x84$-style evaluations</abstract>
</arXiv></metadata></record><record><metadata><arXiv>
<id>other</id><created>2099-01-01</created><abstract>Plain text</abstract>
</arXiv></metadata></record></ListRecords></OAI-PMH>"""
        self.assertIn("$Ï\x84$", response.text)

        with patch("papers.management.commands.harvest_records.requests.get", return_value=response) as get, patch.object(
            Command, "save_paper", return_value=True
        ) as save_paper:
            Command().handle(recent_only=False)

        get.assert_called_once_with("https://oaipmh.arxiv.org/oai?verb=ListRecords&metadataPrefix=arXiv")
        self.assertEqual(
            save_paper.call_args_list[0].args[0]["abstract"],
            "External BFCL and $τ$-style evaluations",
        )

    def test_reharvest_repairs_existing_mojibake_when_source_matches(self):
        paper = Mock()
        paper.title = "REFLEX"
        paper.abstract = "External BFCL and $Ï\x84$-style evaluations"
        data = {
            "id": "2609.26532",
            "title": "REFLEX",
            "abstract": "External BFCL and $τ$-style evaluations",
            "created": "2026-09-22",
            "categories": "cs.AI",
        }

        with patch("papers.management.commands.harvest_records.Paper.objects.get_or_create", return_value=(paper, False)):
            Command().save_paper(data)

        self.assertEqual(paper.abstract, data["abstract"])
        paper.save.assert_called_once_with(update_fields=["abstract"])
        paper.delete_embeddings.assert_called_once_with()


class Voyage4Tests(TestCase):
    def setUp(self):
        # Paper i is the ith nearest neighbour of paper 0; the last one is old and outside the recent window
        now = timezone.now()
        self.papers = []
        for i, (age, category) in enumerate(
            [(1, "cs.LG"), (2, "cs.LG"), (3, "math.AG"), (4, "cs.LG"), (5, "cs.LG"), (60, "cs.LG")]
        ):
            paper = Paper.objects.create(
                arxiv_id=f"test{i}",
                created=now - timedelta(days=age),
                title=f"Paper {i}",
                abstract=f"Abstract {i}",
                categories=[category],
            )
            vector = np.zeros(2048)
            vector[0], vector[1] = 1, i * 0.1
            vector /= np.linalg.norm(vector)
            EmbeddingVoyage4.objects.create(
                paper=paper,
                vector=vector.tolist(),
                bits="".join("1" if x > 0 else "0" for x in vector),
                created=paper.created,
                categories=paper.categories,
            )
            self.papers.append(paper)
        self.ids = [p.id for p in self.papers]

    def search(self, cutoff, category="", excluded=(), limit=20):
        return voyage4_search.similar_ids(
            self.ids[0], cutoff=cutoff, category=category, excluded=set(excluded), limit=limit
        )

    def test_refresh_recent_adds_new_and_expires_old(self):
        stale = self.papers[-1]
        EmbeddingVoyage4Recent.objects.create(
            paper=stale, vector=[1] + [0] * 2047, created=stale.created, categories=stale.categories
        )
        Voyage4Command().refresh_recent()
        self.assertEqual(
            sorted(EmbeddingVoyage4Recent.objects.values_list("paper_id", flat=True)), self.ids[:-1]
        )

    def test_recent_and_general_search_filter_and_rank(self):
        Voyage4Command().refresh_recent()
        now = timezone.now()
        month, year = now - timedelta(days=30), now - timedelta(days=365)
        # Recent window, general window, and the "all time" (no cutoff) path
        self.assertEqual(self.search(month), self.ids[1:5])
        self.assertEqual(self.search(year), self.ids[1:])
        self.assertEqual(self.search(None), self.ids[1:])
        # Category filter, exclusions, and a limit smaller than the number of matches
        self.assertEqual(self.search(month, category="cs.LG"), [self.ids[1], self.ids[3], self.ids[4]])
        self.assertEqual(
            self.search(year, category="cs.LG", excluded=[self.ids[1]]), [self.ids[3], self.ids[4], self.ids[5]]
        )
        self.assertEqual(self.search(month, limit=2), self.ids[1:3])

    def test_recent_window_covers_one_month_filter(self):
        with patch.object(voyage4_search, "query_sql", wraps=voyage4_search.query_sql) as query_sql:
            self.search(timezone.now() - timedelta(days=30))
            self.assertTrue(query_sql.call_args.kwargs["recent"])
            self.search(timezone.now() - timedelta(days=90))
            self.assertFalse(query_sql.call_args.kwargs["recent"])

    def test_paper_without_embedding_has_no_similar_papers(self):
        EmbeddingVoyage4.objects.filter(paper=self.papers[0]).delete()
        self.assertEqual(self.search(None), [])

    def test_delete_embeddings_removes_every_model(self):
        paper = self.papers[0]
        EmbeddingVoyage3Bit2048.objects.create(paper=paper, vector="1" * 2048)
        Voyage4Command().refresh_recent()
        paper.delete_embeddings()
        self.assertFalse(any(model.objects.filter(paper=paper).exists() for model in EMBEDDING_MODELS))
        self.assertTrue(EmbeddingVoyage4.objects.filter(paper=self.papers[1]).exists())
