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

    def test_tag_search_returns_neighbours_but_never_tagged_papers(self):
        self.login_with_tag(self.papers[:2])
        # With the drawer closed and open, the tag's own papers are excluded
        for params in [{}, {"tag": Tag.objects.get(name="reading").id}]:
            response = self.client.get("/", {"q": "tag: reading", "date_filter": "1month", **params})
            ids = [r["paper"].id for r in response.context["results"]]
            self.assertEqual(sorted(ids), self.ids[2:5])

    def test_tag_search_skips_similarity_when_window_is_empty(self):
        self.login_with_tag(self.papers[:2])
        with patch.object(voyage4_search, "similar_ids") as similar_ids:
            response = self.client.get("/", {"q": "tag: reading", "date_filter": "1day"})
        self.assertEqual(response.context["results"], [])
        similar_ids.assert_not_called()

    def test_tag_search_forces_hnsw_only_for_unfiltered_recent_windows(self):
        args = dict(cutoff=timezone.now(), excluded=set(), limit=2, vector="[1]", bits="1")
        hidden = "e.created + interval '0' >= %s"
        sql = lambda **kw: voyage4_search.query_sql(**{**args, **kw})[0]
        self.assertIn(hidden, sql(recent=True, category="", prefer_hnsw=True))
        self.assertNotIn(hidden, sql(recent=True, category="", prefer_hnsw=False))
        self.assertNotIn(hidden, sql(recent=True, category="cs.LG", prefer_hnsw=True))
        self.assertNotIn(hidden, sql(recent=False, category="", prefer_hnsw=True))
        self.assertNotIn(hidden, sql(recent=True, category="", prefer_hnsw=True, exact=True))
        self.login_with_tag(self.papers[:2])
        with patch.object(voyage4_search, "query_sql", wraps=voyage4_search.query_sql) as query_sql:
            self.client.get("/", {"q": "tag: reading", "date_filter": "1week"})
        self.assertTrue(query_sql.call_args_list[0].kwargs["prefer_hnsw"])

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


class IngestionApiTests(TestCase):
    def setUp(self):
        PageTests.setUp(self)
        self.tag = PageTests.login_with_tag(self, self.papers[:2])

    def get(self, **params):
        return self.client.get("/api/ingestion/", params)

    def post(self, **body):
        return self.client.post("/api/ingestion/", body, content_type="application/json")

    def arxiv_ids(self, response):
        return [p["arxiv_id"] for p in response.json()["papers"]]

    def test_requires_login_and_valid_actions(self):
        self.assertEqual(self.get(action="unknown").status_code, 400)
        self.assertEqual(self.get(action="bulk_add").status_code, 405)
        self.assertEqual(self.post(action="tags").status_code, 405)
        self.client.logout()
        self.assertEqual(self.get(action="tags").status_code, 401)

    def test_tags_tag_and_papers(self):
        self.assertEqual(self.get(action="tags").json()["tags"], [{"id": self.tag.id, "name": "reading"}])
        self.assertEqual(self.arxiv_ids(self.get(action="tag", tag="reading")), ["test0", "test1"])
        since = (timezone.now() - timedelta(days=3, hours=12)).isoformat()
        self.assertEqual(self.arxiv_ids(self.get(action="papers", since=since)), ["test0", "test1", "test2"])
        self.assertEqual(self.get(action="papers", since="2026-01-01T00:00:00").status_code, 400)

    def test_search_types_mirror_the_site(self):
        month = (timezone.now() - timedelta(days=30)).isoformat()
        cases = [
            (dict(type="keyword", q="abstract", date_filter="all"), 6),
            (dict(type="title", q="Paper", since=month), ["test0", "test1", "test2", "test3", "test4"]),
            (dict(type="paper", paper="test0", date_filter="1month"), ["test1", "test2", "test3", "test4"]),
            (dict(type="paper", paper="test0", date_filter="1month", category="cs.LG"), ["test1", "test3", "test4"]),
        ]
        for params, expected in cases:
            response = self.get(action="search", **params)
            ids = self.arxiv_ids(response)
            if isinstance(expected, int):
                self.assertEqual(len(ids), expected, params)
            else:
                self.assertEqual(ids, expected, params)
        response = self.get(action="search", type="tag", tag="reading", since=month).json()
        self.assertEqual(sorted(p["arxiv_id"] for p in response["papers"]), ["test2", "test3", "test4"])
        self.assertEqual(response["search"], {"type": "tag", "since": month, "category": ""})

    def test_search_pages_without_repeating_papers(self):
        first = self.get(action="search", type="paper", paper="test0", date_filter="all").json()
        self.assertEqual(len(first["papers"]), 5)
        second = self.get(action="search", type="paper", paper="test0", date_filter="all", cursor=first["next_cursor"]).json()
        self.assertEqual((second["papers"], second["next_cursor"]), ([], None))

    def test_search_rejects_bad_input(self):
        for params in [dict(type="nope", q="x"), dict(type="keyword", q=" "), dict(type="paper", paper="missing"),
                       dict(type="tag", tag="missing"), dict(type="title", q="x", since="2026-01-01T00:00:00+00:00", date_filter="all"),
                       dict(type="title", q="x", date_filter="5years"), dict(type="title", q="x", cursor="%%%")]:
            self.assertEqual(self.get(action="search", **params).status_code, 400, params)

    def test_similar_excludes_tagged_papers_and_uses_created(self):
        since = timezone.now() - timedelta(days=4, hours=12)
        Paper.objects.filter(id=self.ids[5]).update(updated=timezone.now())
        response = self.get(action="similar", tag="reading", since=since.isoformat()).json()
        self.assertEqual(sorted(p["arxiv_id"] for p in response["papers"]), ["test2", "test3"])
        self.assertIsNone(response["next_cursor"])

    def test_bulk_add_bulk_remove_and_copy_tag(self):
        response = self.post(action="bulk_add", tag="new", arxiv_ids=["test2", "test3", "nope"]).json()
        self.assertEqual((response["count"], response["missing"]), (2, ["nope"]))
        response = self.post(action="bulk_remove", tag="new", arxiv_ids=["test2", "nope"]).json()
        self.assertEqual((response["count"], response["missing"]), (1, ["nope"]))
        self.assertEqual(self.arxiv_ids(self.get(action="tag", tag="new")), ["test3"])
        self.assertEqual(self.post(action="copy_tag", source="reading", target="new").json()["count"], 3)
        self.assertEqual(self.post(action="bulk_remove", tag="missing", arxiv_ids=[]).status_code, 400)
        self.assertEqual(self.post(action="bulk_add", tag="new", arxiv_ids=["x"] * 201).status_code, 400)
