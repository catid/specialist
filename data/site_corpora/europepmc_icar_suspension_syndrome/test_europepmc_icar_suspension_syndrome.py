import hashlib
import json
import re
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build
import capture_source


class EuropePmcIcarSuspensionSyndromeCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build.build()
        cls.markdown = build.MARKDOWN.read_text(encoding="utf-8")
        cls.manifest = json.loads(build.MANIFEST.read_text(encoding="utf-8"))
        cls.report = json.loads(build.REPORT.read_text(encoding="utf-8"))
        cls.provenance = json.loads(build.PROVENANCE.read_text(encoding="utf-8"))
        cls.xml_bytes = build.XML.read_bytes()
        cls.root = ET.fromstring(cls.xml_bytes)

    def test_offline_build_is_byte_deterministic(self):
        first = build.render()
        self.assertEqual(first, build.render())
        self.assertEqual(first[0], self.markdown)
        self.assertEqual(first[1], build.MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(first[2], build.REPORT.read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256(self.markdown.encode()).hexdigest(), self.report["hashes"]["markdown_sha256"])
        self.assertEqual(hashlib.sha256(build.MANIFEST.read_bytes()).hexdigest(), self.report["hashes"]["manifest_sha256"])

    def test_capture_is_one_exact_authorized_route(self):
        self.assertEqual(capture_source.SOURCE_URL, build.SOURCE_URL)
        capture_source.validate_url(build.SOURCE_URL)
        for forbidden in (
            "https://europepmc.org/article/MED/38081341",
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10710713/",
            "https://link.springer.com/article/10.1186/s13049-023-01164-z",
            "http://www.ebi.ac.uk/europepmc/webservices/rest/PMC10710713/fullTextXML",
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10710713/fullTextXML?format=json",
        ):
            with self.assertRaises(ValueError):
                capture_source.validate_url(forbidden)
        self.assertEqual(
            self.report["capture_boundary"],
            {
                "authorized_get_routes": [build.SOURCE_URL],
                "excluded_routes": ["Springer publisher endpoint", "United States PMC endpoint"],
                "request_count": 1,
            },
        )

    def test_snapshot_identity_license_and_bibliography_are_exact(self):
        self.assertEqual(len(self.xml_bytes), build.SOURCE_BYTES)
        self.assertEqual(hashlib.sha256(self.xml_bytes).hexdigest(), build.SOURCE_SHA256)
        ids = {
            node.attrib.get("pub-id-type"): build.normalize("".join(node.itertext()))
            for node in self.root.findall("./front/article-meta/article-id")
        }
        self.assertEqual((ids["doi"], ids["pmcid"], ids["pmid"]), (build.DOI, build.PMCID, build.PMID))
        license_text = build.normalize(" ".join(self.root.find("./front/article-meta/permissions/license").itertext()))
        self.assertIn("creativecommons.org/licenses/by/4.0", license_text)
        self.assertIn("Creative Commons Attribution 4.0", license_text)
        self.assertEqual(self.manifest["license"], {"name": "CC BY 4.0", "url": build.LICENSE_URL})
        self.assertEqual(self.manifest["article"]["authors"], build.AUTHORS)

    def test_queue_identifier_error_is_non_training_provenance_only(self):
        mismatch = self.provenance["discovery_metadata_discrepancy"]
        self.assertEqual(mismatch["queued_pmid"], "38081341")
        self.assertEqual(mismatch["jats_pmid"], "38071341")
        self.assertEqual(self.report["repository_provenance_discrepancy"], mismatch)
        self.assertNotIn("discovery_metadata_discrepancy", self.manifest)
        self.assertNotIn("38081341", self.markdown)
        self.assertNotIn("38081341", build.MANIFEST.read_text())
        self.assertIn("PMID 38071341", self.markdown)

    def test_methods_evidence_quality_and_limitations_are_retained(self):
        evidence = self.report["evidence_level"]
        self.assertEqual(
            (
                evidence["epidemiology_studies"],
                evidence["pathophysiology_studies"],
                evidence["quality_good"],
                evidence["quality_fair"],
                evidence["quality_poor"],
                evidence["randomized_intervention_studies"],
                evidence["experimental_prevention_diagnosis_treatment_studies"],
            ),
            (2, 21, 9, 8, 6, 0, 0),
        )
        self.assertFalse(evidence["exact_incidence_known"])
        self.assertFalse(evidence["bondage_transfer_validated"])
        for required in (
            "5.8 million on-rope hours",
            "23 articles were included",
            "9 studies were rated good, 8 fair, and 6 poor",
            "no experimental studies directly testing prevention, diagnosis, or treatment",
            "121 records were returned",
            "abstract says the online search yielded 210 articles",
            "initial-search-count discrepancy",
            "no inter-rater reliability statistic",
        ):
            self.assertIn(required, self.markdown)

    def test_mechanism_myth_correction_and_recommendation_grades_are_bounded(self):
        for required in (
            "venous pooling in the legs",
            "no proof that pooling alone",
            "neurocardiogenic response",
            "finds no evidence for older advice to keep a rescued casualty seated",
            "position the casualty supine",
            "standard advanced life-support algorithms",
            "not a safe exposure limit",
            "not a bondage-suspension threshold",
            "not validated for bondage suspension",
        ):
            self.assertIn(required, self.markdown)
        self.assertEqual(self.report["recommendation_grades_preserved"], ["1C", "1B", "2B", "2B", "2C", "1A", "2C"])
        for grade in set(self.report["recommendation_grades_preserved"]):
            self.assertIn(f"({grade})", self.markdown)

    def test_non_qa_urls_taxonomy_and_split_boundary(self):
        self.assertTrue(self.manifest["direct_training_ready"] and self.manifest["non_qa"])
        self.assertIn("not validated as a bondage-suspension protocol", self.manifest["domain_transfer"])
        urls = set(re.findall(r"https://[^\s)]+", self.markdown))
        self.assertEqual(urls, {build.SOURCE_URL, build.LICENSE_URL})
        lowered = self.markdown.casefold()
        for forbidden in (
            "question:",
            "answer:",
            "how to tie",
            "place the rope",
            "wrap the rope",
            "guarantees safety",
            "proves that",
            "<|im_start|>",
            "<|im_end|>",
            "</think>",
            "heldout",
            "benchmark id",
        ):
            self.assertNotIn(forbidden, lowered)
        labels = re.findall(r"^Categories: (.+)$", self.markdown, flags=re.MULTILINE)
        self.assertEqual(len(labels), self.report["counts"]["sections"])
        observed = {label for line in labels for label in line.split(", ")}
        self.assertEqual(observed, build.TAXONOMY)
        self.assertIn("one split", self.manifest["document_disjoint_requirement"])
        self.assertIn("validation, OOD, shadow, and sealed-holdout", self.manifest["protected_split_requirement"])


if __name__ == "__main__":
    unittest.main()
