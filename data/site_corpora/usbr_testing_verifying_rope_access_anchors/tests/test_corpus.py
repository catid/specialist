#!/usr/bin/env python3
"""Deterministic integrity and scope checks for the USBR anchor-research corpus."""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ID = "USBR-PROJECT-6390"
BULLETIN_ID = "USBR-BULLETIN-2014-09"
PROJECT_URL = "https://usbr.gov/research/projects/detail.cfm?id=6390"
BULLETIN_URL = "https://usbr.gov/research/projects/download_product.cfm?id=2259"
PROJECT_SHA256 = "451093515a67b4043988cc9122f3f567e7df17eb9fdcffd26f096f2f8f8bab0f"
PROJECT_SHA512 = (
    "8f1ae6889878a8abfbbaf178569fef2de69bed89b2438480bffb0851699ab7a86"
    "ad2b9919f139bbe6d36b10f3245d17f000787849ae1281945442dcc321d5e96"
)
BULLETIN_SHA256 = "95758fafca146ded6a672a9660f46e233d497cae3012a641dcc3ef17f13ff4e5"
BULLETIN_SHA512 = (
    "a050e31194b4c2cc5c8c12fefa83819b5f022f6e142f35b9702dd01038c74fa9"
    "d58ffbf30f5056a4904b4d4d23ae6353592dbc13a7012fde7b06e963c01e8874"
)
ARTIFACT_PATHS = {
    "CORPUS.md",
    "README.md",
    "REPORT.md",
    "components.jsonl",
    "dispositions.jsonl",
    "source_snapshot/provenance.json",
    "sources.jsonl",
    "surfaces.jsonl",
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
        cls.surfaces = load_jsonl("surfaces.jsonl")
        cls.components = load_jsonl("components.jsonl")
        cls.provenance = load_json("source_snapshot/provenance.json")

    def test_manifest_schema_readiness_hashes_and_file_set(self) -> None:
        self.assertEqual(self.manifest["schema"], "site-corpus-manifest")
        self.assertEqual(self.manifest["schema_version"], "1.0")
        self.assertEqual(
            self.manifest["corpus_schema"],
            "non-bondage-rope-access-qualitative-evidence-markdown",
        )
        self.assertEqual(self.manifest["corpus_schema_version"], "1.0")
        self.assertEqual(self.manifest["source_ids"], [PROJECT_ID, BULLETIN_ID])
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
        self.assertFalse(any(path.suffix.lower() in {".pdf", ".html", ".png", ".jpg"} for path in ROOT.rglob("*")))

    def test_exact_two_allowed_first_party_bodies(self) -> None:
        self.assertEqual(len(self.sources), 2)
        by_id = {source["source_id"]: source for source in self.sources}
        self.assertEqual(set(by_id), {PROJECT_ID, BULLETIN_ID})

        project = by_id[PROJECT_ID]
        self.assertEqual(project["schema"], "site-corpus-source")
        self.assertEqual(project["schema_version"], "1.0")
        self.assertEqual(project["source_url"], PROJECT_URL)
        self.assertEqual(project["http_status"], 200)
        self.assertEqual(project["body_bytes"], 23_040)
        self.assertEqual(project["body_sha256"], PROJECT_SHA256)
        self.assertEqual(project["body_sha512"], PROJECT_SHA512)
        self.assertEqual(project["rights_status"], "public_domain_us_federal_work")
        self.assertIn("17 U.S.C. 105", project["rights_basis"])
        self.assertIs(project["manual_surface_audit"], True)
        self.assertIs(project["direct_training_ready"], True)

        bulletin = by_id[BULLETIN_ID]
        self.assertEqual(bulletin["schema"], "site-corpus-source")
        self.assertEqual(bulletin["schema_version"], "1.0")
        self.assertEqual(bulletin["source_url"], BULLETIN_URL)
        self.assertEqual(bulletin["http_status"], 200)
        self.assertEqual(bulletin["media_type"], "application/pdf")
        self.assertEqual(bulletin["body_bytes"], 1_258_428)
        self.assertEqual(bulletin["body_sha256"], BULLETIN_SHA256)
        self.assertEqual(bulletin["body_sha512"], BULLETIN_SHA512)
        self.assertEqual(bulletin["page_count"], 2)
        self.assertEqual(bulletin["rights_status"], "public_domain_us_federal_work")
        self.assertIn("17 U.S.C. 105", bulletin["rights_basis"])
        self.assertIs(bulletin["manual_page_audit"], True)
        self.assertIs(bulletin["direct_training_ready"], True)

    def test_provenance_enforces_protected_report_zero_access(self) -> None:
        self.assertEqual(self.provenance["schema"], "site-corpus-provenance")
        self.assertEqual(self.provenance["schema_version"], "1.0")
        policy = self.provenance["retrieval_policy"]
        self.assertEqual(policy["allowed_urls"], [PROJECT_URL, BULLETIN_URL])
        self.assertEqual(policy["allowed_bodies_retrieved"], 2)
        self.assertEqual(policy["protected_document_id"], 1018)
        self.assertEqual(policy["protected_document_status"], "hard_exclude_no_access")
        self.assertEqual(policy["protected_document_requests"], 0)
        self.assertIs(policy["protected_document_accessed"], False)
        self.assertIs(policy["protected_document_inferred_or_reconstructed"], False)

        bodies = {body["source_id"]: body for body in self.provenance["bodies"]}
        self.assertEqual(set(bodies), {PROJECT_ID, BULLETIN_ID})
        self.assertEqual(bodies[PROJECT_ID]["body_sha256"], PROJECT_SHA256)
        self.assertEqual(bodies[BULLETIN_ID]["body_sha256"], BULLETIN_SHA256)
        self.assertEqual(self.provenance["pdf_metadata"]["pages"], 2)
        self.assertEqual(
            self.provenance["rights"]["rights_status"],
            "public_domain_us_federal_work",
        )
        self.assertIs(self.provenance["audit"]["direct_training_ready"], True)
        self.assertNotIn("1018", self.corpus)

    def test_complete_pdf_page_audit(self) -> None:
        self.assertEqual(len(self.dispositions), 2)
        self.assertEqual(
            [record["pdf_page"] for record in self.dispositions],
            [1, 2],
        )
        self.assertTrue(all(record["source_id"] == BULLETIN_ID for record in self.dispositions))
        self.assertTrue(all(record["decision"] == "partial" for record in self.dispositions))
        self.assertTrue(
            all(
                record["audit"] == "manual text and visual review"
                for record in self.dispositions
            )
        )

    def test_complete_html_surface_audit(self) -> None:
        self.assertEqual(len(self.surfaces), 10)
        self.assertEqual(
            [record["surface_id"] for record in self.surfaces],
            [f"html-{number:02d}-{suffix}" for number, suffix in (
                (1, "document-shell"),
                (2, "global-navigation"),
                (3, "project-identity"),
                (4, "research-question"),
                (5, "need-and-benefit"),
                (6, "contributing-partners"),
                (7, "review-claim"),
                (8, "protected-document-1018"),
                (9, "public-bulletin-listing"),
                (10, "trailing-page-chrome"),
            )],
        )
        self.assertTrue(all(record["source_id"] == PROJECT_ID for record in self.surfaces))
        self.assertEqual(
            sum(record["decision"] == "partial" for record in self.surfaces),
            4,
        )
        self.assertEqual(
            sum(record["decision"] == "exclude" for record in self.surfaces),
            5,
        )
        protected = next(record for record in self.surfaces if "1018" in record["surface_id"])
        self.assertEqual(protected["decision"], "hard_exclude_no_access")
        self.assertEqual(protected["retained"], "none")
        self.assertIn("not requested", protected["audit"])

    def test_complete_component_exclusion_audit(self) -> None:
        self.assertEqual(len(self.components), 16)
        self.assertTrue(
            all(
                record["decision"] in {"exclude", "hard_exclude_no_access"}
                for record in self.components
            )
        )
        self.assertEqual(
            sum(record["component_type"] == "photograph" for record in self.components),
            3,
        )
        protected = next(
            record
            for record in self.components
            if record["component_id"] == "html-02-protected-final-report-1018"
        )
        self.assertEqual(protected["decision"], "hard_exclude_no_access")
        self.assertIn("not requested", protected["reason"])
        self.assertEqual(
            self.provenance["audit"]["component_records"],
            len(self.components),
        )

    def test_dense_claim_cited_and_explicitly_labeled_corpus(self) -> None:
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", self.corpus)
            if paragraph.strip() and not paragraph.lstrip().startswith("#")
        ]
        citations = re.findall(
            r"\[(?:USBR-PROJECT-6390|USBR-BULLETIN-2014-09),[^\]]+\]",
            self.corpus,
        )
        self.assertGreaterEqual(len(self.corpus.split()), 1_200)
        self.assertEqual(len(paragraphs), 27)
        self.assertEqual(len(citations), 27)
        for paragraph in paragraphs:
            self.assertRegex(
                paragraph,
                r"^Non-bondage rope-access evidence(?: boundary)?:",
            )
            self.assertRegex(
                paragraph,
                r"\[(?:USBR-PROJECT-6390|USBR-BULLETIN-2014-09),[^\]]+\]$",
            )

        self.assertEqual(self.manifest["statistics"]["corpus_words"], len(self.corpus.split()))
        self.assertEqual(self.manifest["statistics"]["claim_paragraphs"], len(paragraphs))
        self.assertEqual(self.manifest["statistics"]["claim_citations"], len(citations))

    def test_required_qualitative_findings_and_limits(self) -> None:
        required_phrases = (
            "intentionally simulated installation defects",
            "comparatively weak and stronger concrete as a test variable",
            "condition and strength of concrete in Reclamation's aging structures as unknown",
            "Installation uncertainty and substrate uncertainty are distinct",
            "failure modes differed qualitatively",
            "a more progressive outcome and an abrupt component failure",
            "multi-anchor load-sharing behavior is angle-sensitive",
            "multi-anchor load sharing is non-additive",
            "cannot be obtained safely by simply adding individual-anchor results",
            "not intended to provide one fixed standard for every anchoring situation",
            "Aging” is therefore context for uncertainty, not evidence that every older substrate is weak",
        )
        for phrase in required_phrases:
            self.assertIn(phrase, self.corpus, phrase)

    def test_no_capacities_products_standards_procedures_or_exact_examples(self) -> None:
        prohibited_terms = (
            "OSHA",
            "SPRAT",
            "ANSI",
            "CFR",
            "Powers",
            "Hilti",
            "Redhead",
            "kilonewton",
            "pound",
            "inch",
            "hydraulic",
            "clevis",
            "threaded rod",
            "vehicle anchor",
            "I-beam",
            "Lock-Out",
            "Tag-Out",
            "drop test",
            "static load",
            "pull testing",
            "pull test",
            "hole",
            "shear cone",
            "bolt fracture",
            "epoxy",
        )
        for term in prohibited_terms:
            self.assertIsNone(
                re.search(rf"\b{re.escape(term)}\b", self.corpus, re.IGNORECASE),
                term,
            )

        citation_free = re.sub(r"\[[^\]]+\]", "", self.corpus)
        self.assertIsNone(re.search(r"\d", citation_free))
        self.assertNotIn("http://", self.corpus)
        self.assertNotIn("https://", self.corpus)
        self.assertNotIn("![", self.corpus)
        self.assertNotIn("<img", self.corpus.lower())
        self.assertNotIn("|---", self.corpus)

    def test_no_diy_or_human_suspension_certification(self) -> None:
        self.assertIn(
            "does not certify, size, select, rate, install, test, or approve",
            self.corpus,
        )
        self.assertIn("DIY hardpoint", self.corpus)
        self.assertIn("human-suspension system", self.corpus)
        self.assertIn("do not authorize DIY proof testing", self.corpus)
        self.assertIn("do not certify any ceiling, hardpoint", self.corpus)

        for line in self.corpus.splitlines():
            lowered = line.lower()
            if any(term in lowered for term in ("diy", "hardpoint", "human-suspension")):
                self.assertTrue(
                    any(
                        boundary in lowered
                        for boundary in (
                            "does not certify",
                            "do not authorize",
                            "do not certify",
                        )
                    ),
                    line,
                )

    def test_report_records_manual_scope_and_hard_exclusion(self) -> None:
        required_report_text = (
            "two explicitly allowed public Bureau of Reclamation bodies",
            "Document 1018 is a hard exclusion",
            "It was not requested or accessed",
            "ten material surfaces",
            "The two bulletin pages",
            "sixteen records",
            "three photographs",
            "Multi-anchor load sharing was angle-sensitive",
            "not intended to provide one fixed standard",
            "Neither source body is redistributed",
        )
        for phrase in required_report_text:
            self.assertIn(phrase, self.report, phrase)


if __name__ == "__main__":
    unittest.main(verbosity=2)
