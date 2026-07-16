#!/usr/bin/env python3
"""Deterministic integrity and scope checks for the NPS textile corpus."""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "NPS-TEXTILES-K-2002"
SOURCE_URL = (
    "https://www.nps.gov/subjects/museums/upload/"
    "MHI_AppK_TextilesObjects.pdf"
)
SOURCE_SHA256 = "08b5aafe64a97c579b1e3415cff936b25871463914dd4382e5be9918f0ea1217"
SOURCE_SHA512 = (
    "0a047d44a05b1ec0c47a39b029c4e841c3377e6b214a09452f651eaa0a7d3cbd"
    "c4696937d2e5dc22786cebec01ff305d658262b2f434a0b0fd36af592da2c326"
)
ARTIFACT_PATHS = {
    "CORPUS.md",
    "README.md",
    "REPORT.md",
    "components.jsonl",
    "dispositions.jsonl",
    "source_snapshot/provenance.json",
    "sources.jsonl",
    "tests/test_corpus.py",
}


def load_json(path: str) -> object:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def load_jsonl(path: str) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (ROOT / path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        cls.report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
        cls.manifest = load_json("manifest.json")
        cls.sources = load_jsonl("sources.jsonl")
        cls.dispositions = load_jsonl("dispositions.jsonl")
        cls.components = load_jsonl("components.jsonl")
        cls.provenance = load_json("source_snapshot/provenance.json")

    def test_manifest_schema_readiness_and_artifact_hashes(self) -> None:
        self.assertEqual(self.manifest["schema"], "site-corpus-manifest")
        self.assertEqual(self.manifest["schema_version"], "1.0")
        self.assertEqual(
            self.manifest["corpus_schema"],
            "museum-textile-mechanism-markdown",
        )
        self.assertEqual(self.manifest["corpus_schema_version"], "1.0")
        self.assertIs(self.manifest["direct_training_ready"], True)
        self.assertEqual(self.manifest["source_ids"], [SOURCE_ID])

        artifacts = self.manifest["artifacts"]
        self.assertEqual(set(artifacts), ARTIFACT_PATHS)
        for relative_path, metadata in artifacts.items():
            path = ROOT / relative_path
            self.assertTrue(path.is_file(), relative_path)
            self.assertEqual(metadata["sha256"], sha256(path), relative_path)
            self.assertEqual(metadata["bytes"], path.stat().st_size, relative_path)

    def test_exact_first_party_source_and_rights_gate(self) -> None:
        self.assertEqual(len(self.sources), 1)
        source = self.sources[0]
        self.assertEqual(source["schema"], "site-corpus-source")
        self.assertEqual(source["schema_version"], "1.0")
        self.assertEqual(source["source_id"], SOURCE_ID)
        self.assertEqual(source["source_url"], SOURCE_URL)
        self.assertEqual(source["publication_year"], 2002)
        self.assertEqual(source["selected_printed_pages"], "K:2-K:13")
        self.assertEqual(source["http_status"], 200)
        self.assertEqual(source["media_type"], "application/pdf")
        self.assertEqual(source["body_bytes"], 2_190_643)
        self.assertEqual(source["body_sha256"], SOURCE_SHA256)
        self.assertEqual(source["body_sha512"], SOURCE_SHA512)
        self.assertEqual(source["page_count"], 50)
        self.assertEqual(source["rights_status"], "public_domain_us_federal_work")
        self.assertIn("17 U.S.C. 105", source["rights_basis"])
        self.assertIs(source["manual_page_audit"], True)
        self.assertIs(source["manual_component_audit"], True)
        self.assertIs(source["direct_training_ready"], True)

        self.assertEqual(self.provenance["schema"], "site-corpus-provenance")
        self.assertEqual(self.provenance["schema_version"], "1.0")
        self.assertEqual(self.provenance["source_id"], SOURCE_ID)
        self.assertEqual(self.provenance["retrieval"]["source_url"], SOURCE_URL)
        self.assertEqual(self.provenance["retrieval"]["body_sha256"], SOURCE_SHA256)
        self.assertEqual(self.provenance["retrieval"]["body_sha512"], SOURCE_SHA512)
        self.assertEqual(self.provenance["pdf_metadata"]["pages"], 50)
        self.assertEqual(
            self.provenance["bibliographic_resolution"]["publication_year"],
            2002,
        )
        self.assertEqual(
            self.provenance["rights"]["rights_status"],
            "public_domain_us_federal_work",
        )
        self.assertIs(self.provenance["audit"]["direct_training_ready"], True)

    def test_complete_page_disposition_audit(self) -> None:
        self.assertEqual(len(self.dispositions), 50)
        self.assertEqual(
            [record["pdf_page"] for record in self.dispositions],
            list(range(1, 51)),
        )
        for record in self.dispositions:
            page = record["pdf_page"]
            expected_printed = None if page <= 4 else f"K:{page - 4}"
            self.assertEqual(record["source_id"], SOURCE_ID)
            self.assertEqual(record["printed_page"], expected_printed)
            self.assertEqual(record["audit"], "manual text and visual review")
            self.assertIn(record["decision"], {"partial", "exclude"})

        partial_pages = [
            record["pdf_page"]
            for record in self.dispositions
            if record["decision"] == "partial"
        ]
        self.assertEqual(partial_pages, list(range(6, 17)))
        self.assertEqual(
            sum(record["decision"] == "exclude" for record in self.dispositions),
            39,
        )

    def test_complete_component_exclusion_audit(self) -> None:
        self.assertEqual(len(self.components), 19)
        self.assertEqual(
            {record["component_id"] for record in self.components},
            {f"figure-k{number:02d}" for number in range(1, 17)}
            | {"controlled-01", "controlled-02", "controlled-03"},
        )
        self.assertEqual(
            sum(record["component_type"] == "figure" for record in self.components),
            16,
        )
        self.assertEqual(
            sum(
                record["component_type"] == "controlled_source"
                for record in self.components
            ),
            3,
        )
        self.assertTrue(all(record["source_id"] == SOURCE_ID for record in self.components))
        self.assertTrue(all(record["decision"] == "exclude" for record in self.components))

        figure_pages = {
            record["component_id"]: record["pdf_page"]
            for record in self.components
            if record["component_type"] == "figure"
        }
        self.assertEqual(
            figure_pages,
            {
                "figure-k01": 10,
                "figure-k02": 18,
                "figure-k03": 21,
                "figure-k04": 23,
                "figure-k05": 24,
                "figure-k06": 26,
                "figure-k07": 27,
                "figure-k08": 27,
                "figure-k09": 29,
                "figure-k10": 30,
                "figure-k11": 31,
                "figure-k12": 35,
                "figure-k13": 37,
                "figure-k14": 39,
                "figure-k15": 41,
                "figure-k16": 44,
            },
        )

    def test_dense_claim_cited_and_explicitly_labeled_corpus(self) -> None:
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", self.corpus)
            if paragraph.strip() and not paragraph.lstrip().startswith("#")
        ]
        citations = re.findall(r"\[NPS-TEXTILES-K-2002,[^\]]+\]", self.corpus)
        self.assertGreaterEqual(len(self.corpus.split()), 2_000)
        self.assertEqual(len(paragraphs), 43)
        self.assertEqual(len(citations), 43)
        for paragraph in paragraphs:
            self.assertRegex(
                paragraph,
                r"^Museum-textile evidence(?: boundary)?:",
            )
            self.assertRegex(paragraph, r"\[NPS-TEXTILES-K-2002,[^\]]+\]$")

        self.assertEqual(self.manifest["statistics"]["corpus_words"], len(self.corpus.split()))
        self.assertEqual(self.manifest["statistics"]["claim_paragraphs"], len(paragraphs))
        self.assertEqual(self.manifest["statistics"]["claim_citations"], len(citations))

    def test_required_museum_textile_mechanisms_are_present(self) -> None:
        required_phrases = (
            "animal, plant, and synthetic families",
            "Animal fibers are protein-based",
            "Plant fibers are primarily cellulose",
            "Stem fibers are also called bast fibers",
            "regenerated fibers derived from natural polymers",
            "Fiber identity alone does not determine a textile's behavior",
            "gradual breaking of long-chain fiber molecules into shorter chains",
            "Textile fibers are hygroscopic",
            "Dimensional movement creates mechanical stress",
            "Visible and ultraviolet light affect both colorants and fibers",
            "Dirt and dust can contain sharp mineral particles",
            "Microbes and insects can use textile fibers",
            "Mold and mildew favor warm, damp museum environments",
            "museum preventive-conservation handbook rather than a controlled aging experiment",
        )
        for phrase in required_phrases:
            self.assertIn(phrase, self.corpus, phrase)

    def test_no_prohibited_transfer_prescriptions_or_embedded_media(self) -> None:
        prohibited_terms = (
            "jute",
            "hemp",
            "bondage",
            "body",
            "upline",
            "anchor",
            "hardpoint",
            "human suspension",
            "hygiene",
            "conditioning",
            "drying",
            "inspection",
            "retirement",
            "fybrene",
            "velcro",
            "mylar",
        )
        for term in prohibited_terms:
            self.assertIsNone(
                re.search(rf"\b{re.escape(term)}\b", self.corpus, re.IGNORECASE),
                term,
            )

        numeric_prescription_patterns = (
            r"\b\d+(?:\.\d+)?\s*%",
            r"\b\d+(?:\.\d+)?\s*(?:degrees?|°)\s*[CF]\b",
            r"\b\d+(?:\.\d+)?\s*(?:lux|lumens?|foot[ -]?candles?)\b",
            r"\b(?:relative humidity|RH)\s*(?:of|at|=|:)?\s*\d+",
        )
        for pattern in numeric_prescription_patterns:
            self.assertIsNone(re.search(pattern, self.corpus, re.IGNORECASE), pattern)

        self.assertNotIn("http://", self.corpus)
        self.assertNotIn("https://", self.corpus)
        self.assertNotIn("![", self.corpus)
        self.assertNotIn("<img", self.corpus.lower())
        self.assertIn(
            "None establishes performance, maintenance, or safety rules for functional rope",
            self.corpus,
        )

    def test_report_records_manual_scope_and_exclusions(self) -> None:
        required_report_text = (
            "Physical PDF pages audited: 50",
            "All fifty physical page images were manually reviewed",
            "physical pages six through seventeen",
            "sixteen numbered figures",
            "nineteen categorical exclusions",
            "Numerical temperature, relative-humidity, and illumination prescriptions",
            "Direct material-performance transfer to functional rope",
            "The original PDF is not redistributed",
        )
        for phrase in required_report_text:
            self.assertIn(phrase, self.report, phrase)


if __name__ == "__main__":
    unittest.main(verbosity=2)
