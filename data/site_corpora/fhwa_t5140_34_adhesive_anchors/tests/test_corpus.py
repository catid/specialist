#!/usr/bin/env python3
"""Deterministic integrity and scope checks for the FHWA advisory corpus."""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "FHWA-T5140.34-2018"
SOURCE_URL = "https://www.fhwa.dot.gov/Bridge/t514034.pdf"
SOURCE_SHA256 = "716c49feea8867c18c32230405716edd2282886987494abccb052743f2571e9d"
SOURCE_SHA512 = (
    "c95dbaed4c10ec9f78724de9235365f2f888dbc9602cd112c2c3822be2ab0c6a"
    "042622239772d0744e9b56dbdc252e3f3031b460041d0a49cf9f62d360543eb4"
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

    def test_manifest_schema_readiness_and_hashes(self) -> None:
        self.assertEqual(self.manifest["schema"], "site-corpus-manifest")
        self.assertEqual(self.manifest["schema_version"], "1.0")
        self.assertEqual(
            self.manifest["corpus_schema"],
            "non-bondage-civil-infrastructure-failure-governance-markdown",
        )
        self.assertEqual(self.manifest["corpus_schema_version"], "1.0")
        self.assertEqual(self.manifest["source_ids"], [SOURCE_ID])
        self.assertIs(self.manifest["direct_training_ready"], True)

        artifacts = self.manifest["artifacts"]
        self.assertEqual(set(artifacts), ARTIFACT_PATHS)
        for relative_path, metadata in artifacts.items():
            path = ROOT / relative_path
            self.assertTrue(path.is_file(), relative_path)
            self.assertEqual(metadata["bytes"], path.stat().st_size, relative_path)
            self.assertEqual(metadata["sha256"], sha256(path), relative_path)

        package_files = {
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        self.assertEqual(package_files, ARTIFACT_PATHS | {"manifest.json"})

    def test_exact_official_source_and_public_domain_gate(self) -> None:
        self.assertEqual(len(self.sources), 1)
        source = self.sources[0]
        self.assertEqual(source["schema"], "site-corpus-source")
        self.assertEqual(source["schema_version"], "1.0")
        self.assertEqual(source["source_id"], SOURCE_ID)
        self.assertEqual(source["corporate_author"], "Federal Highway Administration")
        self.assertEqual(source["publication_date"], "2018-01-16")
        self.assertEqual(source["source_url"], SOURCE_URL)
        self.assertEqual(source["http_status"], 200)
        self.assertEqual(source["media_type"], "application/pdf")
        self.assertEqual(source["body_bytes"], 215_665)
        self.assertEqual(source["body_sha256"], SOURCE_SHA256)
        self.assertEqual(source["body_sha512"], SOURCE_SHA512)
        self.assertEqual(source["page_count"], 4)
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
        self.assertEqual(self.provenance["pdf_metadata"]["pages"], 4)
        self.assertEqual(
            self.provenance["bibliographic_resolution"]["publication_date"],
            "2018-01-16",
        )
        self.assertEqual(
            self.provenance["rights"]["rights_status"],
            "public_domain_us_federal_work",
        )
        self.assertIs(self.provenance["audit"]["direct_training_ready"], True)

    def test_complete_four_page_audit(self) -> None:
        self.assertEqual(len(self.dispositions), 4)
        self.assertEqual(
            [record["pdf_page"] for record in self.dispositions],
            [1, 2, 3, 4],
        )
        self.assertEqual(
            [record["decision"] for record in self.dispositions],
            ["partial", "partial", "partial", "exclude"],
        )
        self.assertTrue(all(record["source_id"] == SOURCE_ID for record in self.dispositions))
        self.assertTrue(
            all(
                record["audit"] == "manual text and visual review"
                for record in self.dispositions
            )
        )
        self.assertEqual(self.dispositions[-1]["retained"], "none")
        self.assertIn("third-party load-classification table", self.dispositions[-1]["excluded"])

    def test_complete_component_exclusion_audit(self) -> None:
        self.assertEqual(len(self.components), 8)
        self.assertEqual(
            {record["component_id"] for record in self.components},
            {
                "controlled-01-fhwa-page-branding",
                "controlled-02-incorporated-standards",
                "controlled-03-certification-programs",
                "controlled-04-product-and-installation-detail",
                "controlled-05-project-prescriptions",
                "figure-01-structural-schematic",
                "controlled-06-load-material-timeline-examples",
                "table-01-third-party-load-classification",
            },
        )
        self.assertTrue(all(record["source_id"] == SOURCE_ID for record in self.components))
        self.assertTrue(all(record["decision"] == "exclude" for record in self.components))
        self.assertEqual(
            sum(record["component_type"] == "figure" for record in self.components),
            1,
        )
        self.assertEqual(
            sum(record["component_type"] == "table" for record in self.components),
            1,
        )
        table = next(
            record
            for record in self.components
            if record["component_id"] == "table-01-third-party-load-classification"
        )
        self.assertEqual(table["pdf_page"], [3, 4])

    def test_dense_claim_cited_and_explicitly_labeled_corpus(self) -> None:
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", self.corpus)
            if paragraph.strip() and not paragraph.lstrip().startswith("#")
        ]
        citations = re.findall(r"\[FHWA-T5140\.34-2018,[^\]]+\]", self.corpus)
        self.assertGreaterEqual(len(self.corpus.split()), 1_200)
        self.assertEqual(len(paragraphs), 26)
        self.assertEqual(len(citations), 26)
        for paragraph in paragraphs:
            self.assertRegex(
                paragraph,
                r"^Non-bondage civil-infrastructure evidence(?: boundary)?:",
            )
            self.assertRegex(paragraph, r"\[FHWA-T5140\.34-2018,[^\]]+\]$")

        self.assertEqual(self.manifest["statistics"]["corpus_words"], len(self.corpus.split()))
        self.assertEqual(self.manifest["statistics"]["claim_paragraphs"], len(paragraphs))
        self.assertEqual(self.manifest["statistics"]["claim_citations"], len(citations))

    def test_required_failure_and_governance_lessons(self) -> None:
        required_phrases = (
            "fatal suspended-ceiling failure",
            "inadequate creep resistance in the adhesive system as the primary cause",
            "irregular overhead installation as a contributing factor",
            "when tension persists, time-dependent creep must be considered",
            "orientation by itself is not enough to classify the condition",
            "how the whole structure transfers a persistent force into the anchorage",
            "system qualification and installer qualification as necessary controls",
            "trained personnel should carry out adhesive-anchor installation",
            "continuous inspection as a control",
            "owner remains responsible for defining inspector qualifications",
            "qualified inspection, engineering evaluation or testing, and remediation or replacement",
            "Owners also need guidance for identifying persistent versus transient conditions",
        )
        for phrase in required_phrases:
            self.assertIn(phrase, self.corpus, phrase)

        self.assertIn(
            "The advisory itself identifies the investigated event as a ceiling-panel collapse",
            self.corpus,
        )
        self.assertIn(
            "supplies no casualty narrative",
            self.corpus,
        )

    def test_no_standards_products_recipes_examples_or_embedded_media(self) -> None:
        prohibited_terms = (
            "ACI",
            "AASHTO",
            "CRSI",
            "NCHRP",
            "fast set",
            "fast-set",
            "epoxy",
            "steel rod",
            "concrete",
            "plastic",
            "hole drilling",
            "hole cleaning",
            "adhesive storage",
            "adhesive mixing",
            "manufacturer",
            "MPII",
            "bond stress",
            "preload",
            "dead load",
            "live load",
            "truck",
            "wind",
            "ice load",
            "months",
            "years",
        )
        for term in prohibited_terms:
            self.assertIsNone(
                re.search(rf"\b{re.escape(term)}\b", self.corpus, re.IGNORECASE),
                term,
            )

        numeric_patterns = (
            r"\b\d+(?:\.\d+)?\s*(?:lb|lbs|pounds?|kips?|kn|newtons?|mpa|psi)\b",
            r"\b\d+(?:\.\d+)?\s*%",
            r"\b\d+(?:\.\d+)?\s*(?:hours?|days?|weeks?|months?|years?)\b",
        )
        for pattern in numeric_patterns:
            self.assertIsNone(re.search(pattern, self.corpus, re.IGNORECASE), pattern)

        self.assertNotIn("http://", self.corpus)
        self.assertNotIn("https://", self.corpus)
        self.assertNotIn("![", self.corpus)
        self.assertNotIn("<img", self.corpus.lower())
        self.assertNotIn("|---", self.corpus)

    def test_non_bondage_no_capacity_boundary_is_explicit(self) -> None:
        self.assertIn(
            "does not certify, size, select, rate, or establish capacity",
            self.corpus,
        )
        self.assertIn("DIY hardpoint", self.corpus)
        self.assertIn("human-suspension system", self.corpus)
        self.assertIn("provides no human-suspension capacity inference", self.corpus)
        for line in self.corpus.splitlines():
            lowered = line.lower()
            if "diy hardpoint" in lowered or "human-suspension" in lowered:
                self.assertTrue(
                    any(
                        boundary in lowered
                        for boundary in (
                            "does not",
                            "nothing in",
                            "provides no",
                        )
                    ),
                    line,
                )

    def test_report_documents_scope_and_transparency(self) -> None:
        required_report_text = (
            "Physical PDF pages: 4",
            "All four physical pages were extracted and read manually",
            "eight categorical exclusions",
            "one structural schematic",
            "one third-party table",
            "PDF itself says only “ceiling panel collapse”",
            "does not state a casualty count",
            "Any certification, sizing, selection, rating, or capacity statement",
            "The original PDF is not redistributed",
        )
        for phrase in required_report_text:
            self.assertIn(phrase, self.report, phrase)


if __name__ == "__main__":
    unittest.main(verbosity=2)
