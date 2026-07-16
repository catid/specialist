from __future__ import annotations

import hashlib
import json
import re
import unittest
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


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = read_json(ROOT / "manifest.json")
        cls.provenance = read_json(ROOT / "source_snapshot/provenance.json")
        cls.dispositions = read_jsonl(ROOT / "dispositions.jsonl")
        cls.corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")
        cls.report = (ROOT / "REPORT.md").read_text(encoding="utf-8")

    def test_manifest_declares_one_direct_non_qa_document(self) -> None:
        self.assertEqual(self.manifest["schema"], "site-corpus-manifest-v1")
        self.assertEqual(
            self.manifest["resource_id"], "usfs_region5_hazard_tree_2022"
        )
        self.assertEqual(
            self.manifest["artifact_role"],
            "canonical_markdown_direct_training_source",
        )
        self.assertTrue(self.manifest["direct_training_ready"])
        self.assertTrue(self.manifest["non_qa"])
        self.assertEqual(self.manifest["training_document_count"], 1)
        self.assertEqual(self.manifest["direct_training_word_count"], 4201)
        self.assertEqual(self.manifest["whitespace_delimited_word_count"], 4226)

    def test_publication_identity_and_region_scope(self) -> None:
        publication = self.provenance["publication"]
        self.assertEqual(publication["title"], "Hazard Tree Identification and Mitigation")
        self.assertEqual(publication["report_number"], "FHP Report RO-22-01")
        self.assertEqual(publication["revision_as_displayed"], "March 2022")
        self.assertEqual(publication["region"], "Pacific Southwest Region (Region 5)")
        self.assertEqual(len(publication["authors"]), 8)
        self.assertEqual(publication["authors"][0], "Peter A. Angwin")
        self.assertEqual(publication["authors"][-1], "Sherry Hazelhurst")
        self.assertEqual(self.manifest["publication"], publication)
        for phrase in (
            "Pacific Southwest Region",
            "Region 5",
            "California forests",
            "It is not a universal arboriculture standard",
        ):
            self.assertIn(phrase, self.corpus)

    def test_exact_official_pdf_integrity_is_recorded(self) -> None:
        expected_url = (
            "https://www.fs.usda.gov/sites/nfs/files/legacy-media/r05/"
            "Hazard%20Tree%20Identification%20and%20Mitigation%20-%202022.pdf"
        )
        remote = self.provenance["remote_object"]
        self.assertEqual(remote["url"], expected_url)
        self.assertEqual(remote["http_status"], 200)
        self.assertEqual(remote["content_type"], "application/pdf")
        self.assertEqual(remote["content_length"], 1_597_581)
        self.assertEqual(
            remote["sha256"],
            "a743018dd717a8c101c2a714dd6d37ed06bb981795a816d8cc43939adaa28612",
        )
        self.assertEqual(remote["last_modified"], "Tue, 18 Feb 2025 11:52:44 GMT")
        self.assertFalse(remote["retained_in_repository"])
        self.assertEqual(self.manifest["entries"][0]["url"], expected_url)
        self.assertEqual(self.manifest["entries"][0]["sha256"], remote["sha256"])

    def test_network_boundary_has_only_the_official_pdf(self) -> None:
        expected_url = self.provenance["canonical_pdf_url"]
        network = self.provenance["network_boundary"]
        self.assertEqual(network["get_routes_used"], [expected_url])
        self.assertEqual(network["head_routes_used"], [expected_url])
        for key in (
            "publisher_html_route_accessed",
            "alternate_pdf_route_accessed",
            "external_reference_route_accessed",
            "figure_or_image_route_accessed",
            "other_body_route_accessed",
        ):
            self.assertFalse(network[key])

    def test_pdf_structure_and_visible_authorship_rule(self) -> None:
        structure = self.provenance["pdf_structure"]
        self.assertEqual(structure["format"], "PDF 1.7")
        self.assertEqual(structure["digital_page_count"], 40)
        self.assertFalse(structure["encrypted"])
        self.assertEqual(
            structure["embedded_metadata"]["title"],
            "Hazard Tree Identification and Mitigation, Region 5, USFS",
        )
        self.assertEqual(structure["embedded_metadata"]["author"], "fsdefaultUser")
        self.assertIn("visible report cover", structure["identity_rule"])
        self.assertEqual(self.provenance["review_method"]["digital_pages_reviewed"], 40)

    def test_pdf_is_referenced_not_retained_or_direct_training(self) -> None:
        self.assertFalse(any(ROOT.rglob("*.pdf")))
        self.assertEqual(
            sorted(p.name for p in (ROOT / "source_snapshot").iterdir()),
            ["provenance.json"],
        )
        self.assertFalse(self.provenance["training_boundary"]["source_pdf_direct_training"])
        entry = self.manifest["entries"][0]
        self.assertEqual(entry["disposition"], "referenced_not_retained")
        self.assertFalse(entry["direct_training"])

    def test_rights_review_is_federal_text_only_not_blanket(self) -> None:
        rights = self.provenance["rights_review"]
        self.assertEqual(
            rights["decision"],
            "include_manually_transformed_eligible_federal_text_with_component_exclusions",
        )
        self.assertIn("17 U.S.C. 105", rights["basis"])
        self.assertFalse(rights["source_rights_statement_found"])
        self.assertFalse(rights["component_credit_statement_found"])
        self.assertIn("not evidence", rights["interpretation_of_absence"])
        self.assertFalse(rights["source_visuals_reused"])
        self.assertFalse(rights["source_table_layouts_reused"])
        self.assertFalse(rights["source_forms_reused"])
        self.assertFalse(rights["legal_opinion"])
        self.assertIn("blanket rights conclusion", self.report)

    def test_every_visual_component_group_is_audited_and_excluded(self) -> None:
        audit = self.provenance["component_audit"]
        self.assertEqual(audit["numbered_figure_count"], 6)
        self.assertEqual(audit["numbered_table_count"], 1)
        self.assertEqual(audit["raster_xobject_occurrence_count"], 18)
        self.assertEqual(
            audit["pages_with_raster_xobjects"],
            [1, 3, 7, 9, 20, 27, 28, 31, 32],
        )
        components = audit["components"]
        self.assertEqual(len(components), 9)
        self.assertEqual(
            sum(item["raster_xobject_occurrences"] for item in components), 18
        )
        self.assertTrue(all(not item["reused"] for item in components))
        self.assertTrue(all(item["disposition"].startswith("excluded") for item in components))
        self.assertEqual(audit["photo_caption_count_found"], 0)
        self.assertFalse(audit["photograph_reused"])
        self.assertEqual(audit["separately_fetched_component_count"], 0)
        self.assertIn("Figures 1–6", self.corpus)

    def test_prediction_limit_and_residual_risk_are_prominent(self) -> None:
        for phrase in (
            "not every tree failure can be predicted",
            "reduce risk, but cannot eliminate it",
            "Tree failures will occur regardless of inspection intensity",
            "absence of a visible defect is not proof",
            "cannot promise that failure will wait for the next scheduled inspection",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertIn("all tree failures cannot be predicted", self.report.lower())
        self.assertEqual(
            self.provenance["evidence_audit"]["central_uncertainty"],
            "It is impossible to predict all tree failures; a professional program can reduce but not eliminate risk.",
        )
        self.assertEqual(
            self.manifest["prediction_status"],
            "impossible_to_predict_all_failures_risk_reduced_not_eliminated",
        )

    def test_failure_defect_target_hazard_and_loss_are_distinct(self) -> None:
        for heading in (
            "**A defect**",
            "**Failure**",
            "**A target**",
            "**Loss**",
            "**Hazard**",
        ):
            self.assertIn(heading, self.corpus)
        for phrase in (
            "A defect without an exposed target is not the same management problem",
            "A visible feature is not automatically a defect",
            "mechanical failure produces a loss only when it affects a target",
            "administrative prioritization tool",
            "not a physical measurement of breaking strength",
        ):
            self.assertIn(phrase, self.corpus)

    def test_inspection_frequency_stays_contextual(self) -> None:
        for phrase in (
            "Inspection interval is a decision, not a universal calendar",
            "public use, tree-failure history, local insect, disease, weather, and fire impacts",
            "checks before reopening closed roads",
            "additional checks after major storms or fires",
            "rationale for choosing or changing an interval should be documented",
            "A calendar date should not override evidence of change",
        ):
            self.assertIn(phrase, self.corpus)
        lower = self.corpus.lower()
        for forbidden in (
            "every tree must be inspected annually",
            "an annual inspection guarantees",
            "one inspection per year is sufficient",
            "safe until the next inspection",
        ):
            self.assertNotIn(forbidden, lower)

    def test_target_exposure_is_kept_separate_from_defect(self) -> None:
        for phrase in (
            "Exposure duration and frequency matter",
            "stationary targets",
            "mobile target",
            "occupancy, use pattern, and whether people stop",
            "size and height of the tree part, distance and direction, lean, slope, and obstacles",
            "does not prove that a nearby tree is defective",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(
            self.manifest["target_exposure_status"],
            "retained_as_consequence_context_not_failure_proof",
        )

    def test_all_required_defect_families_and_interactions_are_present(self) -> None:
        for heading in (
            "## Cracks: evidence of change, not one diagnosis",
            "## Branch unions, old injury, and poor architecture",
            "## Decay, cavities, fruiting bodies, and cankers",
            "## Dead trees, tops, branches, and biological attack",
            "## Roots and the site around the tree",
            "## Recent lean and long-compensated lean are different observations",
        ):
            self.assertIn(heading, self.corpus)
        for phrase in (
            "Connected defects often matter more than isolated labels",
            "Crown health and vigor are not reliable measures of internal wood condition",
            "Reliably predicting when a dead tree will fail is nearly impossible",
            "green, healthy-looking tree can still have root disease",
            "Corrected growth is evidence",
            "does not certify strong roots or future stability",
        ):
            self.assertIn(phrase, self.corpus)

    def test_root_and_site_disturbance_context_is_complete(self) -> None:
        for phrase in (
            "Soil compaction, erosion, fire, flooding or prolonged saturation",
            "excavation and construction",
            "prolonged heavy-equipment use",
            "repeated foot or vehicle traffic",
            "new soil cracks, mounding, or movement near the base",
            "root-plate lifting, broken roots, or partial windthrow",
            "exposed or undermined roots after erosion",
            "Construction, excavation, vehicle traffic, heavy equipment, and concentrated foot traffic",
        ):
            self.assertIn(phrase, self.corpus)

    def test_monitoring_weather_and_records_do_not_certify(self) -> None:
        for phrase in (
            "Monitoring is not the same as deciding that a defect is harmless",
            "Repeated observations can show change in cracks, lean, crown, roots, decay indicators",
            "may need review after significant weather",
            "what is being watched",
            "who is qualified to interpret it",
            "which changes or events prompt earlier reassessment",
            "cannot promise that failure will wait",
        ):
            self.assertIn(phrase, self.corpus + self.report)
        self.assertIn("weather does not operate as a simple trigger", self.corpus.lower())

    def test_certified_arborist_and_two_domain_referral_gate(self) -> None:
        for phrase in (
            "certified arborist when local expertise is absent for work on large hardwoods",
            "qualified arboricultural professional",
            "qualified rigging or structural professional",
            "One professional role does not automatically supply the other",
            "not a sling or human-suspension design",
        ):
            self.assertIn(phrase, self.corpus)
        boundary = self.provenance["anchor_and_human_suspension_boundary"]
        self.assertEqual(len(boundary["required_professional_domains"]), 2)
        self.assertFalse(boundary["source_studied_load_bearing_anchors"])
        self.assertFalse(boundary["source_studied_human_suspension"])
        self.assertFalse(boundary["tree_health_equals_anchor_capacity"])

    def test_no_tree_anchor_or_human_suspension_certification(self) -> None:
        for phrase in (
            "Tree health is not anchor capacity",
            "provides no tree, branch, sling, anchor, hardpoint, or load certification",
            "Human suspension requires domain-specific expertise; this report cannot approve it",
            "Its hazard ratings are not working-load limits",
            "no construction or attachment instructions",
            "This corpus provides no load-bearing approval",
        ):
            self.assertIn(phrase, self.corpus + self.report)
        boundary = self.provenance["anchor_and_human_suspension_boundary"]
        self.assertEqual(len(boundary["forbidden_certifications"]), 8)
        self.assertEqual(len(boundary["forbidden_derivations"]), 7)
        self.assertEqual(
            self.manifest["anchor_transfer_status"],
            "forbidden_tree_health_not_anchor_capacity_no_load_approval",
        )

    def test_nonportable_numbers_species_and_diseases_are_absent(self) -> None:
        lower = self.corpus.lower()
        for marker in (
            "one-third rule",
            "1/3",
            "one and a half tree",
            "1.5-times",
            "two tree lengths",
            "150 feet",
            "300 feet",
            "50% mortality",
            "3 points",
            "score 4 points",
            "heterobasidion",
            "phytophthora ramorum",
            "echinodontium",
            "porodaedalia",
            "laetiporus",
            "survey123 mobile",
        ):
            self.assertNotIn(marker, lower)
        self.assertNotRegex(lower, r"\b\d+(?:\.\d+)?\s*(?:feet|tree lengths|points)\b")
        self.assertEqual(
            self.manifest["universalization_status"],
            "forbidden_region5_species_disease_score_and_distance_rules_not_portable",
        )

    def test_mitigation_cutting_forms_and_construction_are_excluded(self) -> None:
        lower = self.corpus.lower()
        for source_instruction in (
            "moving picnic tables",
            "cut to resemble a natural break",
            "filling decay cavities with concrete",
            "cables, braces, and support poles are sometimes installed",
            "the tree should be felled",
            "use of a paint gun",
            "choose reference points that are permanent structures",
            "request accounts at",
        ):
            self.assertNotIn(source_instruction, lower)
        for record_id in (
            "hazard-tree-r5-008",
            "hazard-tree-r5-010",
            "hazard-tree-r5-021",
            "hazard-tree-r5-022",
            "hazard-tree-r5-027",
            "hazard-tree-r5-028",
        ):
            row = next(r for r in self.dispositions if r["record_id"] == record_id)
            self.assertFalse(row["direct_training_included"])
        self.assertEqual(
            self.manifest["mitigation_status"],
            "excluded_no_cutting_felling_pruning_topping_cabling_bracing_or_construction",
        )

    def test_dispositions_are_complete_and_sequential(self) -> None:
        self.assertEqual(len(self.dispositions), 30)
        self.assertEqual(sum(r["direct_training_included"] for r in self.dispositions), 21)
        self.assertEqual(sum(not r["direct_training_included"] for r in self.dispositions), 9)
        self.assertEqual(
            [r["record_id"] for r in self.dispositions],
            [f"hazard-tree-r5-{n:03d}" for n in range(1, 31)],
        )
        self.assertEqual(self.manifest["disposition_record_count"], 30)
        self.assertEqual(self.manifest["included_or_narrowed_disposition_count"], 21)
        self.assertEqual(self.manifest["excluded_disposition_count"], 9)

    def test_corpus_is_prose_not_chat_qa_or_identifier_drill(self) -> None:
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
            "report number",
            "URL",
            "page",
            "species",
            "disease",
            "score",
            "distance",
            "cutting",
            "anchor",
            "load",
        ):
            self.assertIn(forbidden, policy["forbidden"])
        self.assertTrue(policy["inherit_document_split"])

    def test_split_hygiene_and_protected_data_boundary(self) -> None:
        split = self.provenance["split_hygiene"]
        self.assertTrue(split["non_qa"])
        self.assertEqual(split["source_document_unit"], "FHP-RO-22-01-March-2022")
        self.assertFalse(split["protected_data_accessed_during_build"])
        self.assertEqual(len(split["protected_data_classes_not_accessed"]), 10)
        self.assertIn("one split before chunking", split["document_disjoint_requirement"])
        self.assertEqual(
            self.manifest["document_disjoint_requirement"],
            split["document_disjoint_requirement"],
        )
        self.assertIn("protected-QA", self.manifest["protected_split_requirement"])

    def test_output_hashes_and_lengths(self) -> None:
        for item in self.manifest["outputs"].values():
            path = ROOT / item["path"]
            self.assertEqual(item["byte_length"], path.stat().st_size)
            self.assertEqual(item["sha256"], sha256(path))

    def test_report_manifest_and_corpus_counts_match(self) -> None:
        regex_count = len(re.findall(r"[\w\u2019'-]+", self.corpus, flags=re.UNICODE))
        whitespace_count = len(self.corpus.split())
        citation_count = len(
            re.findall(r"\((?:USDA Forest Service R5, 2022, )?pp?\.", self.corpus)
        )
        self.assertEqual(regex_count, 4201)
        self.assertEqual(whitespace_count, 4226)
        self.assertEqual(citation_count, 43)
        self.assertEqual(self.manifest["direct_training_word_count"], regex_count)
        self.assertEqual(
            self.manifest["whitespace_delimited_word_count"], whitespace_count
        )
        self.assertEqual(self.manifest["manual_page_citation_count"], citation_count)
        self.assertIn("4,226-whitespace-word", self.report)
        self.assertIn("4,201 Unicode-regex words", self.report)
        self.assertIn("30 disposition records", self.report)


if __name__ == "__main__":
    unittest.main()
