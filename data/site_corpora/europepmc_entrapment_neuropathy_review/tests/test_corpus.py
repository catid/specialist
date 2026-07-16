from __future__ import annotations

import hashlib
import json
import re
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"


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
            self.manifest["resource_id"],
            "europepmc_entrapment_neuropathy_review",
        )
        self.assertEqual(
            self.manifest["artifact_role"],
            "canonical_markdown_direct_training_source",
        )
        self.assertTrue(self.manifest["direct_training_ready"])
        self.assertTrue(self.manifest["non_qa"])
        self.assertEqual(self.manifest["training_document_count"], 1)
        self.assertEqual(self.manifest["direct_training_word_count"], 2977)
        self.assertEqual(self.manifest["whitespace_delimited_word_count"], 3002)

    def test_jats_identity_is_bound_to_all_three_identifiers(self) -> None:
        expected = {
            "pmcid": "PMC7382548",
            "pmid": "32766466",
            "doi": "10.1097/PR9.0000000000000829",
        }
        for kind, value in expected.items():
            node = self.article.find(f".//article-id[@pub-id-type='{kind}']")
            self.assertIsNotNone(node)
            self.assertEqual(text(node), value)
            self.assertEqual(self.provenance["article"][kind], value)
            self.assertEqual(self.manifest["article"][kind], value)
        self.assertEqual(
            text(self.article.find(".//article-title")),
            "Entrapment neuropathies: a contemporary approach to "
            "pathophysiology, clinical assessment, and management",
        )
        self.assertEqual(
            self.provenance["article"]["authors"],
            ["Annina B Schmid", "Joel Fundaun", "Brigitte Tampin"],
        )
        self.assertEqual(self.provenance["article"]["publication_date"], "2020-07-22")

    def test_only_authorized_emb_l_ebi_body_route_was_used(self) -> None:
        route = (
            "https://www.ebi.ac.uk/europepmc/webservices/rest/"
            "PMC7382548/fullTextXML"
        )
        boundary = self.provenance["network_boundary"]
        self.assertEqual(boundary["get_routes_used"], [route])
        self.assertEqual(boundary["head_routes_used"], [route])
        self.assertFalse(boundary["canonical_metadata_route_accessed"])
        self.assertFalse(boundary["publisher_route_accessed"])
        self.assertFalse(boundary["united_states_pmc_route_accessed"])
        self.assertFalse(boundary["other_article_body_route_accessed"])
        self.assertFalse(boundary["external_figure_or_table_route_accessed"])
        self.assertEqual(self.manifest["entries"][0]["url"], route)
        self.assertFalse(
            self.provenance["source_locator"]["accessed_during_build"]
        )

    def test_remote_and_normalized_snapshot_integrity(self) -> None:
        source = self.provenance["authorized_source"]
        self.assertEqual(source["remote_raw_byte_length"], 220_620)
        self.assertEqual(
            source["remote_raw_sha256"],
            "9d3244720e670f574c8b203635f54d6c00f7cdcfbcf89d1c8155bba486d086b5",
        )
        self.assertEqual(source["snapshot_byte_length"], 220_621)
        self.assertEqual(
            source["snapshot_sha256"],
            "da030b3e0a84759e96c4639399f8b41921c277df638213a3d1bd77848e38c0cc",
        )
        self.assertEqual(source["snapshot_byte_length"], self.xml_path.stat().st_size)
        self.assertEqual(source["snapshot_sha256"], sha256(self.xml_path))
        self.assertTrue(self.xml_path.read_bytes().endswith(b">\n"))
        self.assertIn("terminal LF", source["normalization"])

    def test_cc_by_4_license_attribution_and_change_notice(self) -> None:
        license_text = text(self.article.find(".//license"))
        self.assertIn("Creative Commons Attribution License 4.0", license_text)
        license_link = self.article.find(".//license//ext-link")
        self.assertIsNotNone(license_link)
        self.assertEqual(
            license_link.get(XLINK_HREF),
            "http://creativecommons.org/licenses/by/4.0/",
        )
        self.assertEqual(
            self.provenance["license"]["name"],
            "Creative Commons Attribution 4.0 International",
        )
        self.assertEqual(self.manifest["license"]["name"], "CC BY 4.0")
        self.assertTrue(
            self.provenance["license"]["attribution_and_change_notice_present"]
        )
        self.assertIn("License and adaptation notice", self.corpus)
        self.assertIn("manually curated adaptation", self.corpus)

    def test_output_hashes_and_lengths(self) -> None:
        for item in self.manifest["outputs"].values():
            path = ROOT / item["path"]
            self.assertEqual(item["byte_length"], path.stat().st_size)
            self.assertEqual(item["sha256"], sha256(path))

    def test_narrative_review_design_is_explicit(self) -> None:
        method = self.provenance["review_method"]
        self.assertEqual(method["source_design"], "contemporary narrative review")
        self.assertEqual(len(method["systematic_review_elements_not_reported"]), 5)
        for phrase in (
            "contemporary narrative review",
            "does not report a systematic search strategy",
            "prespecified eligibility criteria",
            "uniform risk-of-bias appraisal",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertIn("narrative_review", self.manifest["source_type"])

    def test_human_and_preclinical_evidence_are_not_flattened(self) -> None:
        for phrase in (
            "acute, severe nerve injury",
            "milder, chronic compromise",
            "a closer model remains a model",
            "Preclinical observation",
            "Human association",
            "Human diagnostic-performance finding",
            "Clinical inference",
            "Those transitions should remain hypotheses",
        ):
            self.assertIn(phrase, self.corpus)
        boundary = self.provenance["evidence_audit"]["human_preclinical_boundary"]
        self.assertIn("cannot define a safe rope exposure", boundary)
        self.assertEqual(
            self.manifest["experimental_transfer_status"],
            "forbidden_no_human_rope_threshold_or_operational_transfer",
        )

    def test_experimental_pressure_values_are_fully_excluded(self) -> None:
        lower = self.corpus.lower()
        for marker in (
            "20 to 30",
            "20–30",
            "20-30",
            "mm hg",
            "millimetres of mercury",
            "millimeters of mercury",
        ):
            self.assertNotIn(marker, lower)
        self.assertNotRegex(lower, r"\b\d+(?:\.\d+)?\s*(?:mmhg|kpa|psi)\b")
        self.assertIn("The source includes experimental pressure values", self.corpus)
        self.assertIn("They are deliberately omitted here", self.corpus)
        self.assertEqual(
            self.manifest["pressure_transfer_status"],
            "forbidden_values_omitted_no_safe_threshold",
        )

    def test_rope_transfer_boundary_covers_every_prohibited_derivation(self) -> None:
        for phrase in (
            "It supplies no safe rope pressure",
            "no safe body location",
            "no safe exposure time",
            "no schedule for checking a tied person",
            "no waiting period after symptoms",
            "cannot be used to calculate when a nerve will be injured",
            "do not determine recovery, prognosis, treatment, self-care, release timing, or return to rope",
            "does not support",
            "specifying wrap width, tension, direction, duration, or check interval",
            "teaching a release maneuver",
            "deciding when someone can return to rope",
        ):
            self.assertIn(phrase, self.corpus)
        forbidden = self.provenance["clinical_boundary"]["forbidden_derivations"]
        self.assertGreaterEqual(len(forbidden), 11)
        self.assertTrue(self.provenance["clinical_boundary"]["not_a_triage_protocol"])
        self.assertEqual(
            self.manifest["direct_bondage_claim_status"],
            "forbidden_source_did_not_study_rope",
        )

    def test_rope_neurological_symptoms_require_qualified_evaluation(self) -> None:
        expected = (
            "Numbness, tingling, altered sensation, electric or burning pain, "
            "weakness, loss of coordination, or other neurological symptoms "
            "after rope require evaluation by a qualified medical professional."
        )
        self.assertIn(expected, self.corpus)
        self.assertEqual(
            self.provenance["clinical_boundary"]["qualified_evaluation_required"],
            expected,
        )
        self.assertIn("A symptom report is not a diagnosis", self.corpus)
        self.assertIn("the article does not define a safe delay", self.corpus)

    def test_symptoms_signs_diagnosis_and_damage_remain_distinct(self) -> None:
        for heading in (
            "## Symptoms, signs, and diagnosis",
            "A **symptom**",
            "A **sign**",
            "A **diagnosis**",
        ):
            self.assertIn(heading, self.corpus)
        for phrase in (
            "One symptom does not identify a lesion",
            "One normal sign does not exclude it",
            "One positive maneuver does not prove it",
            "pain, structural nerve damage, and abnormal conduction do not always travel together",
            "absence of pain does not rule out loss of function",
            "presence of pain does not prove axonal damage",
        ):
            self.assertIn(phrase, self.corpus)
        boundaries = self.provenance["evidence_audit"]["presentation_boundaries"]
        self.assertIn("symptom is not examination sign", boundaries)
        self.assertIn("sign is not diagnosis", boundaries)

    def test_radicular_pain_and_radiculopathy_are_not_synonyms(self) -> None:
        self.assertIn("**Radicular pain**", self.corpus)
        self.assertIn("without demonstrable loss of nerve-root function", self.corpus)
        self.assertIn("**Radiculopathy**", self.corpus)
        self.assertIn("manifests with loss of function", self.corpus)
        self.assertIn(
            "Preserve the distinction between radicular pain and radiculopathy",
            self.corpus,
        )

    def test_mechanosensitivity_is_not_a_diagnostic_binary(self) -> None:
        for phrase in (
            "neither necessary nor sufficient for diagnosis",
            "can occur without a nerve lesion",
            "a confirmed lesion may lack mechanosensitivity",
            "Self-performing a named maneuver after rope exposure cannot clear the nerve",
            "omits maneuver instructions",
        ):
            self.assertIn(phrase, self.corpus)
        lower = self.corpus.lower()
        for source_maneuver in (
            "phalen test",
            "tinel sign",
            "spurling test",
            "straight leg raise test",
            "upper-limb neurodynamic test",
        ):
            self.assertNotIn(source_maneuver, lower)

    def test_test_limitations_remain_attached(self) -> None:
        for phrase in (
            "early carpal tunnel involvement may escape detection",
            "value is less established for some proximal conditions",
            "false positives and false negatives occur",
            "level suggested by imaging may not match the clinical examination",
            "size is one finding within a diagnostic workup",
            "They do not replace history and examination",
            "condition, stage, reference standard, population, protocol, and interpreter",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(
            len(self.provenance["evidence_audit"]["test_limitations"]),
            6,
        )
        self.assertEqual(
            self.manifest["diagnostic_test_status"],
            "supportive_not_standalone_not_clearance",
        )

    def test_management_prognosis_and_recovery_details_are_excluded(self) -> None:
        lower = self.corpus.lower()
        for marker in (
            "23%",
            "29%",
            "48%",
            "odds ratio 5.62",
            "recommended at night only",
            "local steroid injections",
            "oral corticosteroids",
            "tumour necrosis factor blockers",
            "lumbar microdiscectomy",
            "reoperation rates",
            "surgery is indicated",
        ):
            self.assertNotIn(marker, lower)
        excluded = next(
            row
            for row in self.dispositions
            if row["record_id"] == "entrapment-neuropathy-022"
        )
        self.assertFalse(excluded["direct_training_included"])
        self.assertEqual(
            excluded["disposition"],
            "excluded_entire_section_from_substantive_training",
        )
        self.assertEqual(
            self.manifest["management_transfer_status"],
            "forbidden_section_excluded_no_treatment_self_care_or_return_to_rope",
        )

    def test_all_external_visuals_were_audited_and_not_fetched(self) -> None:
        figures = self.article.findall(".//fig")
        tables = self.article.findall(".//table-wrap")
        self.assertEqual([node.get("id") for node in figures], ["F1", "F2", "F3", "F4"])
        self.assertEqual([node.get("id") for node in tables], ["T1"])
        f2 = next(node for node in figures if node.get("id") == "F2")
        self.assertIn("adapted from 55", text(f2.find("caption")))
        t1 = tables[0]
        self.assertEqual(len(t1.findall(".//table")), 0)
        self.assertEqual(
            [g.get(XLINK_HREF) for g in t1.findall(".//graphic")],
            ["painreports-5-e829-g003.jpg"],
        )
        audit = self.provenance["visual_component_audit"]
        self.assertEqual(len(audit["figures"]), 4)
        self.assertEqual(len(audit["tables"]), 1)
        self.assertEqual(audit["external_binary_count_fetched"], 0)
        self.assertTrue(all(not item["binary_fetched"] for item in audit["figures"]))
        self.assertFalse(audit["tables"][0]["binary_fetched"])
        snapshot_files = [p.name for p in (ROOT / "source_snapshot").iterdir()]
        self.assertEqual(sorted(snapshot_files), ["fullTextXML.xml", "provenance.json"])

    def test_dispositions_cover_source_and_training_boundaries(self) -> None:
        self.assertEqual(len(self.dispositions), 25)
        self.assertEqual(sum(row["direct_training_included"] for row in self.dispositions), 19)
        self.assertEqual(sum(not row["direct_training_included"] for row in self.dispositions), 6)
        self.assertEqual(
            [row["record_id"] for row in self.dispositions],
            [f"entrapment-neuropathy-{n:03d}" for n in range(1, 26)],
        )
        self.assertEqual(self.manifest["disposition_record_count"], 25)
        self.assertEqual(self.manifest["included_or_narrowed_disposition_count"], 19)
        self.assertEqual(self.manifest["excluded_disposition_count"], 6)
        snapshot = self.dispositions[-1]
        self.assertFalse(snapshot["direct_training_included"])
        self.assertEqual(
            snapshot["disposition"],
            "retained_provenance_not_direct_training",
        )

    def test_corpus_is_prose_not_qa_chat_or_identifier_drill(self) -> None:
        lower = self.corpus.lower()
        for marker in (
            "<|im_start|>",
            "<|im_end|>",
            "</think>",
            '"question":',
            '"answer":',
            "http://",
            "https://",
        ):
            self.assertNotIn(marker, lower)
        self.assertNotRegex(self.corpus, r"(?m)^Q:\s|^A:\s")
        policy = self.manifest["qa_derivation_policy"]
        for forbidden in (
            "title",
            "author",
            "DOI",
            "PMCID",
            "PMID",
            "URL",
            "pressure",
            "duration",
            "placement",
            "treatment",
            "prognosis",
        ):
            self.assertIn(forbidden, policy["forbidden"])
        self.assertTrue(policy["inherit_document_split"])

    def test_split_hygiene_and_protected_data_boundary(self) -> None:
        split = self.provenance["split_hygiene"]
        self.assertTrue(split["non_qa"])
        self.assertEqual(split["source_document_unit"], "PMC7382548")
        self.assertFalse(split["protected_data_accessed_during_build"])
        self.assertGreaterEqual(len(split["protected_data_classes_not_accessed"]), 9)
        self.assertIn("one split before chunking", split["document_disjoint_requirement"])
        self.assertEqual(
            self.manifest["document_disjoint_requirement"],
            split["document_disjoint_requirement"],
        )
        self.assertIn("protected-QA", self.manifest["protected_split_requirement"])

    def test_report_and_manifest_counts_match_corpus(self) -> None:
        regex_count = len(re.findall(r"[\w\u2019'-]+", self.corpus, flags=re.UNICODE))
        whitespace_count = len(self.corpus.split())
        self.assertEqual(regex_count, 2977)
        self.assertEqual(whitespace_count, 3002)
        self.assertEqual(self.manifest["direct_training_word_count"], regex_count)
        self.assertEqual(
            self.manifest["whitespace_delimited_word_count"], whitespace_count
        )
        self.assertIn("3,002-whitespace-word", self.report)
        self.assertIn("2,977 Unicode-regex words", self.report)
        self.assertIn("25 manual inclusion", self.report)


if __name__ == "__main__":
    unittest.main()
