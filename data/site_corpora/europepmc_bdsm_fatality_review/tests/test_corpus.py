from __future__ import annotations

import hashlib
import json
import re
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text(element) -> str:
    return " ".join("".join(element.itertext()).split())


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = read_json(ROOT / "manifest.json")
        cls.provenance = read_json(ROOT / "source_snapshot/provenance.json")
        cls.dispositions = read_jsonl(ROOT / "dispositions.jsonl")
        cls.corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        cls.report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
        cls.xml_path = ROOT / "source_snapshot/fullTextXML.xml"
        cls.article = ET.parse(cls.xml_path).getroot()

    def test_manifest_declares_one_direct_non_qa_document(self) -> None:
        self.assertEqual(self.manifest["schema"], "site-corpus-manifest-v1")
        self.assertEqual(
            self.manifest["resource_id"], "europepmc_bdsm_fatality_review"
        )
        self.assertEqual(
            self.manifest["artifact_role"],
            "canonical_markdown_direct_training_source",
        )
        self.assertTrue(self.manifest["direct_training_ready"])
        self.assertTrue(self.manifest["non_qa"])
        self.assertEqual(self.manifest["training_document_count"], 1)
        self.assertGreaterEqual(self.manifest["direct_training_word_count"], 2900)

    def test_jats_identity_is_bound_to_all_three_identifiers(self) -> None:
        expected = {
            "pmcid": "PMC8813685",
            "pmid": "34383118",
            "doi": "10.1007/s00414-021-02674-0",
        }
        for kind, value in expected.items():
            node = self.article.find(f".//article-id[@pub-id-type='{kind}']")
            self.assertIsNotNone(node)
            self.assertEqual(text(node), value)
            self.assertEqual(self.provenance["article"][kind], value)
            self.assertEqual(self.manifest["article"][kind], value)
        self.assertEqual(
            text(self.article.find(".//article-title")),
            "How safe is BDSM? A literature review on fatal outcome in BDSM play",
        )
        self.assertEqual(
            self.provenance["article"]["authors"],
            ["Anouk Schori", "Christian Jackowski", "Corinna A. Schön"],
        )

    def test_only_authorized_emb_l_ebi_route_was_used(self) -> None:
        route = (
            "https://www.ebi.ac.uk/europepmc/webservices/rest/"
            "PMC8813685/fullTextXML"
        )
        boundary = self.provenance["network_boundary"]
        self.assertEqual(boundary["get_routes_used"], [route])
        self.assertEqual(boundary["head_routes_used"], [route])
        self.assertFalse(boundary["publisher_route_accessed"])
        self.assertFalse(boundary["united_states_pmc_route_accessed"])
        self.assertFalse(boundary["other_article_body_route_accessed"])
        self.assertEqual(self.manifest["entries"][0]["url"], route)

    def test_remote_and_normalized_snapshot_integrity(self) -> None:
        source = self.provenance["authorized_source"]
        self.assertEqual(source["remote_raw_byte_length"], 84_800)
        self.assertEqual(
            source["remote_raw_sha256"],
            "70bbf0a86d13f9b2b34ef99f3b78bcc7cf3ee87b34dfd0eba78f7941fe7bd0bc",
        )
        self.assertEqual(source["snapshot_byte_length"], 84_801)
        self.assertEqual(
            source["snapshot_sha256"],
            "5c1e66790925eed475b554fe673c04ba6a42363ba1691aa3e2d20934f6434bcf",
        )
        self.assertEqual(source["snapshot_byte_length"], self.xml_path.stat().st_size)
        self.assertEqual(source["snapshot_sha256"], sha256(self.xml_path))
        self.assertTrue(self.xml_path.read_bytes().endswith(b">\n"))
        self.assertIn("terminal LF", source["normalization"])

    def test_cc_by_4_license_and_change_notice(self) -> None:
        license_text = text(self.article.find(".//license"))
        self.assertIn("Creative Commons Attribution 4.0 International", license_text)
        self.assertIn("creativecommons.org/licenses/by/4.0", license_text)
        self.assertEqual(
            self.provenance["license"]["name"],
            "Creative Commons Attribution 4.0 International",
        )
        self.assertEqual(self.manifest["license"]["name"], "CC BY 4.0")
        self.assertTrue(
            self.provenance["license"]["attribution_and_change_notice_present"]
        )
        self.assertIn("attributed, safety-focused adaptation", self.corpus)
        self.assertIn("corrects", self.corpus)

    def test_output_hashes_and_lengths(self) -> None:
        for item in self.manifest["outputs"].values():
            path = ROOT / item["path"]
            self.assertEqual(item["byte_length"], path.stat().st_size)
            self.assertEqual(item["sha256"], sha256(path))

    def test_case_count_and_cause_arithmetic_match_jats(self) -> None:
        table2 = self.article.find(".//table-wrap[@id='Tab2']")
        self.assertEqual(len(table2.findall(".//tbody/tr")), 17)
        table3 = self.article.find(".//table-wrap[@id='Tab3']")
        rows = [
            [text(cell) for cell in list(row)]
            for row in table3.findall(".//tbody/tr")
        ]
        self.assertIn(["Strangulation", "15"], rows)
        self.assertIn(["Suffocation/oronasal occlusion", "1"], rows)
        self.assertIn(["Hemorrhage combined with alcohol intoxication", "1"], rows)
        self.assertEqual(15 + 1 + 1, 17)
        audit = self.provenance["evidence_audit"]
        self.assertEqual(audit["review_case_count"], 17)
        self.assertEqual(audit["cause_categories"]["strangulation"], 15)
        self.assertIn("15 of 17", self.corpus)
        self.assertIn("88.2 percent of its case set", self.corpus)

    def test_three_denominators_remain_distinct(self) -> None:
        audit = self.provenance["evidence_audit"]["german_autopsy_series"]
        self.assertEqual(
            audit,
            {
                "years": "1993-2017",
                "all_autopsies": 16437,
                "non_natural_sexual_activity_deaths": 74,
                "bdsm_associated_cases": 3,
            },
        )
        for row in (
            "| Cases in the literature review | 17 |",
            "| BDSM-associated cases within the German sexual-death subset | 3 | 74 |",
            "| BDSM-associated cases within all autopsies in that series | 3 | 16,437 |",
        ):
            self.assertIn(row, self.corpus)
        self.assertIn("Neither is the risk per BDSM practitioner or per scene", self.corpus)
        self.assertIn("supply no denominator at all", self.corpus)

    def test_toxicology_missingness_and_correlation_gate(self) -> None:
        tox = self.provenance["evidence_audit"]["decedent_toxicology"]
        self.assertEqual(tox, {"mentioned": 13, "positive": 8, "negative": 5})
        for phrase in (
            "13 of 17 cases",
            "Eight of those 13",
            "four cases",
            "denominator-ambiguous",
            "association is not causation",
            "no outcome-negative comparison group",
            "Detection is not automatically proof of impairment",
        ):
            self.assertIn(phrase, self.corpus)
        lower = self.corpus.lower()
        self.assertNotIn("substances caused 61.5 percent", lower)
        self.assertNotIn("drugs caused the deaths", lower)
        self.assertNotIn("64.3 percent of all", lower)

    def test_experience_safeword_and_cpr_missingness_gates(self) -> None:
        audit = self.provenance["evidence_audit"]
        self.assertEqual(audit["both_participants_described_as_not_novices"], 9)
        self.assertEqual(
            audit["safeword_reporting"],
            {"explicitly_absent": 1, "not_reported": 16},
        )
        for phrase in (
            "all nine cases",
            "missing or partial in eight cases",
            "one case explicitly noted no safeword",
            "other 16 did not report",
            "did not guarantee prevention",
        ):
            self.assertIn(phrase, self.corpus + self.report)
        self.assertIn("results section describes two cases", self.corpus)
        self.assertIn("discussion refers to three cases", self.corpus)
        self.assertIn("Knowledge was not measured consistently", self.corpus)
        self.assertIn("cannot establish an association", self.corpus)

    def test_source_internal_asphyxia_label_issue_is_visible(self) -> None:
        table1 = self.article.find(".//table-wrap[@id='Tab1']")
        rows = {
            text(row[0]): text(row[1])
            for row in table1.findall(".//tbody/tr")
            if len(list(row)) == 2
        }
        self.assertIn("position of an individual", rows["Traumatic asphyxia"])
        self.assertIn("external chest compression", rows["Positional asphyxia"])
        self.assertIn("likely internal label inversion", self.corpus)
        self.assertIn("does not silently teach either row", self.corpus)
        self.assertIn(
            "descriptions labeled traumatic and positional asphyxia appear inverted",
            " ".join(self.provenance["evidence_audit"]["source_internal_issues"]),
        )

    def test_sensitive_case_narratives_and_techniques_are_not_training_text(self) -> None:
        lower = self.corpus.lower()
        excluded_markers = (
            "double hanging",
            "pendulum",
            "metal tubes",
            "shoelaces",
            "wooden board",
            "bag over head",
            "towel in the mouth",
            "rope around the neck for",
            "case #",
        )
        for marker in excluded_markers:
            self.assertNotIn(marker, lower)
        self.assertNotIn("23 and 49", lower)
        self.assertNotIn("34.9", lower)
        self.assertNotIn("dominatrix", lower)
        self.assertIn(
            "No case narrative, apparatus layout, body position, sequence of failure, or reconstruction is retained",
            self.corpus,
        )
        self.assertIn("no method for neck compression", self.report)
        snapshot_disposition = next(
            row
            for row in self.dispositions
            if row["record_id"] == "bdsm-fatality-022"
        )
        self.assertFalse(snapshot_disposition["direct_training_included"])
        self.assertEqual(
            snapshot_disposition["disposition"],
            "retained_provenance_not_direct_training",
        )

    def test_no_prevalence_or_safety_guarantee_transfer(self) -> None:
        lower = self.corpus.lower()
        forbidden = (
            "bdsm fatality rate is",
            "17 cases prove",
            "88.2 percent of all bdsm",
            "breath control can be made safe",
            "a safeword makes",
            "experience makes",
            "all fatalities are preventable.",
        )
        for claim in forbidden:
            self.assertNotIn(claim, lower)
        for gate in (
            "not a cohort study",
            "a population survey",
            "nothing in the paper establishes the probability",
            "publication bias",
            "cannot establish that one activity is more or less common",
            "cannot be eliminated",
        ):
            self.assertIn(gate, lower)
        self.assertEqual(
            self.manifest["population_inference_status"],
            "forbidden_no_population_or_exposure_denominator",
        )
        self.assertEqual(
            self.manifest["technique_transfer_status"],
            "forbidden_no_instructional_reconstruction",
        )

    def test_non_qa_and_anti_identifier_trivia_surface(self) -> None:
        for marker in (
            "<|im_start|>",
            "<|im_end|>",
            "</think>",
            "Question:",
            "Answer:",
            "Which canonical",
            "What is the DOI",
            "What is the PMCID",
            "http://",
            "https://",
        ):
            self.assertNotIn(marker, self.corpus)
        self.assertIn("not title, author, DOI, PMCID, PMID, page, or URL recall", self.corpus)
        policy = self.manifest["qa_derivation_policy"]
        self.assertIn("identifier recall", policy["forbidden"])
        self.assertTrue(policy["inherit_document_split"])

    def test_dispositions_have_exact_counts_and_required_exclusions(self) -> None:
        self.assertEqual(len(self.dispositions), 22)
        self.assertEqual(
            len({row["record_id"] for row in self.dispositions}),
            len(self.dispositions),
        )
        included = [
            row for row in self.dispositions if row["direct_training_included"]
        ]
        excluded = [
            row for row in self.dispositions if not row["direct_training_included"]
        ]
        self.assertEqual((len(included), len(excluded)), (15, 7))
        by_id = {row["record_id"]: row for row in self.dispositions}
        self.assertEqual(
            by_id["bdsm-fatality-007"]["disposition"],
            "excluded_sensitive_case_narratives",
        )
        self.assertEqual(
            by_id["bdsm-fatality-014"]["disposition"],
            "excluded_as_comparable_rate_evidence",
        )
        self.assertEqual(
            by_id["bdsm-fatality-020"]["disposition"],
            "included_bounded_rejected_overclaims",
        )

    def test_split_hygiene_and_protected_data_boundary(self) -> None:
        split = self.provenance["split_hygiene"]
        self.assertTrue(split["non_qa"])
        self.assertFalse(split["protected_data_accessed_during_build"])
        self.assertEqual(
            set(split["protected_data_classes_not_accessed"]),
            {"validation", "OOD", "shadow", "sealed holdout", "protected QA"},
        )
        self.assertIn("PMC8813685", split["document_disjoint_requirement"])
        self.assertEqual(
            split["document_disjoint_requirement"],
            self.manifest["document_disjoint_requirement"],
        )
        self.assertIn(
            "validation, OOD, shadow, sealed-holdout, and protected-QA",
            self.manifest["protected_split_requirement"],
        )

    def test_manual_review_boundary_is_documented(self) -> None:
        review = self.provenance["review_method"]
        self.assertEqual(review["mode"], "manual_source_review_and_hand_cleanup")
        self.assertIn("manually selected", review["automation_boundary"])
        self.assertFalse(review["case_narratives_copied_to_training"])
        self.assertFalse(review["technique_reconstructions_copied_to_training"])
        self.assertFalse(review["source_tables_copied_to_training"])
        self.assertTrue(review["aggregate_counts_recomputed_against_jats"])
        self.assertIn("rewritten, and critically reviewed by hand", self.report)


if __name__ == "__main__":
    unittest.main()
