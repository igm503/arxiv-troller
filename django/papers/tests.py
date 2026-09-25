from datetime import timedelta
from unittest.mock import Mock, patch

import numpy as np
from django.contrib.auth.models import User
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
    Tag,
    TaggedPaper,
)
from papers.views import process_latex_commands


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


class LatexTests(SimpleTestCase):
    def test_commands_escapes_and_links(self):
        cases = {
            r"\textbf{a} \emph{b} \texttt{c}": "<strong>a</strong> <em>b</em> <code>c</code>",
            r"\url{http://y.org}": '<a href="http://y.org" target="_blank">http://y.org</a>',
            r"\href{http://y.org}{here}": '<a href="http://y.org" target="_blank">here</a>',
            "see https://x.org/a.": 'see <a href="https://x.org/a" target="_blank">https://x.org/a</a>.',
            r"50\% \& \$x\$ \#1 a\_b \{c\}": "50% & $x$ #1 a_b {c}",
            r"a\~b a~b x\,y": "a&nbsp;b a&nbsp;b x y",
            r"line\\next": "line<br>next",
            "``quoted''": '"quoted"',
        }
        for text, expected in cases.items():
            self.assertEqual(process_latex_commands(text), expected, text)

    def test_single_backtick_becomes_opening_quote(self):
        self.assertEqual(process_latex_commands("a `quoted' word"), "a \u2018quoted' word")


class PageTests(TestCase):
    def setUp(self):
        Voyage4Tests.setUp(self)
        Voyage4Command().refresh_recent()

    def login_with_tag(self, papers):
        user = User.objects.create(username="reader")
        tag = Tag.objects.create(user=user, name="reading")
        for paper in papers:
            TaggedPaper.objects.create(tag=tag, paper=paper)
        self.client.force_login(user)
        return tag

    def test_tag_counts_and_drawer_on_search_and_detail_pages(self):
        tag = self.login_with_tag(self.papers[:2])
        Tag.objects.create(user=tag.user, name="zzz")
        Tag.objects.create(user=tag.user, name="aaa")
        response = self.client.get("/", {"tag": tag.id})
        html = response.content.decode()
        self.assertLess(html.index("aaa (0)"), html.index("reading (2)"))
        self.assertLess(html.index("reading (2)"), html.index("zzz (0)"))
        self.assertEqual([t["paper"].id for t in response.context["tagged_papers"]], self.ids[1::-1])
        response = self.client.get(f"/paper/{self.ids[2]}/", {"tag": tag.id, "sort": "alpha"})
        self.assertEqual([t["paper"].id for t in response.context["tagged_papers"]], self.ids[:2])
        self.assertContains(response, f"/paper/{self.ids[0]}/?tag={tag.id}&sort=alpha")

    def test_ajax_drawer_matches_page_drawer(self):
        self.papers[0].title = r"Zeta \textbf{Bold} <script>"
        self.papers[0].save()
        tag = self.login_with_tag(self.papers[:2])
        html = self.client.get("/ajax/get-tag-drawer/", {"tag_id": tag.id, "sort": "alpha"}).json()["papers_html"]
        self.assertIn("Zeta &lt;strong&gt;Bold&lt;/strong&gt; &lt;script&gt;", html)
        self.assertLess(html.index(f"removeFromTag({self.ids[1]})"), html.index(f"removeFromTag({self.ids[0]})"))
        self.assertIn(f"/paper/{self.ids[0]}/?tag={tag.id}&sort=alpha", html)
        empty = Tag.objects.create(user=tag.user, name="empty")
        html = self.client.get("/ajax/get-tag-drawer/", {"tag_id": empty.id}).json()["papers_html"]
        self.assertIn("No papers tagged yet", html)

    def test_tag_search_returns_neighbours_of_tagged_papers(self):
        self.login_with_tag(self.papers[:2])
        response = self.client.get("/", {"q": "tag: reading", "date_filter": "1month"})
        ids = [r["paper"].id for r in response.context["results"]]
        self.assertEqual(sorted(ids), self.ids[:5])

    def test_paper_search_and_load_more(self):
        response = self.client.get("/", {"single_paper": self.ids[0], "date_filter": "1month"})
        self.assertEqual([r["paper"].id for r in response.context["results"]], self.ids[1:5])
        response = self.client.post(
            "/",
            {"exclude_ids": self.ids[1:3], "query_params": {"single_paper": str(self.ids[0]), "date_filter": "1month"}},
            content_type="application/json",
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )
        html = response.json()["html"]
        self.assertEqual([pid for pid in self.ids if f'data-paper-id="{pid}"' in html], self.ids[3:5])
