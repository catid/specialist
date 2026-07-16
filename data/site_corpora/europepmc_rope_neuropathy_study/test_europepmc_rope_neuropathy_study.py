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


class EuropePmcRopeNeuropathyCorpus(unittest.TestCase):
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
        second = build.render()
        self.assertEqual(first, second)
        self.assertEqual(first[0], self.markdown)
        self.assertEqual(first[1], build.MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(first[2], build.REPORT.read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256(self.markdown.encode()).hexdigest(),
            self.report["hashes"]["markdown_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(build.MANIFEST.read_bytes()).hexdigest(),
            self.report["hashes"]["manifest_sha256"],
        )

    def test_capture_is_one_exact_authorized_route(self):
        self.assertEqual(capture_source.SOURCE_URL, build.SOURCE_URL)
        capture_source.validate_url(build.SOURCE_URL)
        for forbidden in (
            "https://europepmc.org/article/MED/37324199",
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10294117/",
            "https://www.cureus.com/articles/155296",
            "http://www.ebi.ac.uk/europepmc/webservices/rest/PMC10294117/fullTextXML",
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10294117/fullTextXML?format=json",
        ):
            with self.assertRaises(ValueError):
                capture_source.validate_url(forbidden)
        self.assertEqual(
            (
                self.provenance["capture_scope"],
                self.provenance["request_count"],
                self.provenance["requested_url"],
                self.provenance["final_url"],
                self.provenance["http_status"],
            ),
            (
                "one_authorized_embl_ebi_europepmc_fullTextXML_get",
                1,
                build.SOURCE_URL,
                build.SOURCE_URL,
                200,
            ),
        )
        self.assertEqual(
            self.report["capture_boundary"],
            {
                "authorized_get_routes": [build.SOURCE_URL],
                "excluded_routes": ["Cureus publisher endpoint", "United States PMC endpoint"],
                "request_count": 1,
            },
        )

    def test_snapshot_identity_license_and_bibliography_are_exact(self):
        self.assertEqual(len(self.xml_bytes), build.SOURCE_BYTES)
        self.assertEqual(hashlib.sha256(self.xml_bytes).hexdigest(), build.SOURCE_SHA256)
        self.assertEqual(self.root.tag, "article")
        ids = {
            node.attrib.get("pub-id-type"): build.normalize("".join(node.itertext()))
            for node in self.root.findall("./front/article-meta/article-id")
        }
        self.assertEqual((ids["doi"], ids["pmcid"], ids["pmid"]), (build.DOI, build.PMCID, build.PMID))
        license_text = build.normalize(" ".join(self.root.find("./front/article-meta/permissions/license").itertext()))
        self.assertIn(build.LICENSE_URL, license_text)
        self.assertIn("Creative Commons Attribution License", license_text)
        self.assertEqual(self.manifest["license"], {"name": "CC BY 3.0", "url": build.LICENSE_URL})
        self.assertEqual(self.manifest["article"]["authors"], build.AUTHORS)

    def test_discovery_identifier_discrepancy_is_preserved_not_resolved_by_extra_fetch(self):
        mismatch = self.provenance["discovery_metadata_discrepancy"]
        self.assertEqual(mismatch["queued_pmid"], "37324199")
        self.assertEqual(mismatch["jats_pmid"], "37384078")
        self.assertIn("no MED or publisher route was queried", mismatch["resolution"])
        self.assertNotIn("discovery_metadata_discrepancy", self.manifest)
        self.assertEqual(self.report["repository_provenance_discrepancy"], mismatch)
        self.assertNotIn("37324199", self.markdown)
        self.assertIn("PMID 37384078", self.markdown)

    def test_methods_results_and_limitations_retain_exact_denominators(self):
        evidence = self.report["evidence_level"]
        self.assertEqual(
            (
                evidence["main_cohort_people"],
                evidence["counted_injury_instances"],
                evidence["medical_attention_people"],
                evidence["nerve_conduction_study_people"],
            ),
            (10, 16, 5, 2),
        )
        self.assertFalse(evidence["population_incidence_estimable"])
        self.assertFalse(evidence["population_prevalence_estimable"])
        self.assertFalse(evidence["causal_protocol_validated"])
        for required in (
            "9 of 10 individuals",
            "13 of 16 counted injury instances",
            "77.3% conduction block",
            "two minutes to five months",
            "small sample",
            "selection bias",
            "retrospective recall",
            "no uninjured comparison group",
            "cannot estimate the incidence or prevalence",
        ):
            self.assertIn(required, self.markdown)

    def test_markdown_is_non_qa_attributed_and_not_a_tying_protocol(self):
        self.assertTrue(self.manifest["direct_training_ready"] and self.manifest["non_qa"])
        self.assertEqual(self.report["role"], "canonical_markdown_direct_training_source")
        self.assertIn(build.TITLE, self.markdown)
        urls = set(re.findall(r"https://[^\s)]+", self.markdown))
        self.assertEqual(urls, {build.SOURCE_URL, build.LICENSE_URL})
        lowered = self.markdown.casefold()
        for forbidden in (
            "question:",
            "answer:",
            "step 1",
            "place the rope",
            "wrap the rope",
            "tie the person",
            "guarantees safety",
            "proves that",
            "<|im_start|>",
            "<|im_end|>",
            "</think>",
            "heldout",
            "benchmark id",
        ):
            self.assertNotIn(forbidden, lowered)
        self.assertIn("does not reproduce a tying method", lowered)
        self.assertIn("cannot define a safe time", lowered)
        self.assertIn("does not validate a universally safe configuration", lowered)

    def test_taxonomy_counts_and_split_boundary(self):
        labels = re.findall(r"^Categories: (.+)$", self.markdown, flags=re.MULTILINE)
        self.assertEqual(len(labels), self.report["counts"]["sections"])
        observed = {
            label
            for line in labels
            for label in line.split(", ")
        }
        self.assertEqual(observed, build.TAXONOMY)
        self.assertEqual(set(self.report["supported_categories"]), build.TAXONOMY)
        self.assertIn("one split", self.manifest["document_disjoint_requirement"])
        self.assertIn("validation, OOD, shadow, and sealed-holdout", self.manifest["protected_split_requirement"])


if __name__ == "__main__":
    unittest.main()
