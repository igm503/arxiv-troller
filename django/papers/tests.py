from unittest.mock import Mock, patch

from django.test import SimpleTestCase
from requests import Response
from requests.utils import get_encoding_from_headers

from papers.management.commands.harvest_records import Command


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
