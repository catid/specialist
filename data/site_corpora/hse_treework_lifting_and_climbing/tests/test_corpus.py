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
            self.manifest["resource_id"], "hse_treework_lifting_and_climbing"
        )
        self.assertEqual(
            self.manifest["artifact_role"],
            "canonical_markdown_direct_training_source",
        )
        self.assertTrue(self.manifest["direct_training_ready"])
        self.assertTrue(self.manifest["non_qa"])
        self.assertEqual(self.manifest["training_document_count"], 1)
        self.assertEqual(self.manifest["direct_training_word_count"], 3592)
        self.assertEqual(self.manifest["whitespace_delimited_word_count"], 3605)

    def test_all_eight_page_identities_and_raw_digests_are_exact(self) -> None:
        expected = {
            "hse-arboriculture": (
                "https://www.hse.gov.uk/treework/safety-topics/arboriculture.htm",
                46310,
                "0cf03b91b2d1aff23902a3bb4879c0ffc5c4023c76d36fb39760748d821fb0f8",
                "Wed, 24 Jun 2026 19:00:57 GMT",
            ),
            "hse-aerial-work": (
                "https://www.hse.gov.uk/treework/safety-topics/aerial-work.htm",
                39764,
                "1130f6592362623506ccb03e5713ff1697509f7ac429820607ca8f1b4354a06c",
                "Thu, 02 Jul 2026 13:24:28 GMT",
            ),
            "hse-tree-climbing": (
                "https://www.hse.gov.uk/treework/safety-topics/climbing-operations.htm",
                41485,
                "5d090257d25ad947c242ce2033a1efa20f8d43613c125fa36f1f53800cb8cfd1",
                "Wed, 24 Jun 2026 19:00:58 GMT",
            ),
            "hse-loler-hub": (
                "https://www.hse.gov.uk/work-equipment-machinery/loler.htm",
                34572,
                "b0291c079851a9c0509db36afcde82e0639d6570972c067b9d6c74d56662e633",
                "Wed, 24 Jun 2026 19:01:53 GMT",
            ),
            "hse-thorough-examinations": (
                "https://www.hse.gov.uk/work-equipment-machinery/thorough-examinations-lifting-equipment.htm",
                47486,
                "ac6b66cdc1f6ee505703f9fff7a2b081254c94977c7cab8f972c1af46eb9382e",
                "Wed, 24 Jun 2026 19:01:57 GMT",
            ),
            "hse-treework-height": (
                "https://www.hse.gov.uk/treework/safety-topics/height.htm",
                40112,
                "1bd19f428053cac56338d1c89e0035d9af7558ec4cc2f654c8eca6ff889e4543",
                "Wed, 24 Jun 2026 19:00:59 GMT",
            ),
            "hse-indg367-landing": (
                "https://www.hse.gov.uk/pubns/indg367.htm",
                36865,
                "2074fd759c0fc828883ffdcc3e54ceac17fb6030ab8573b359f85848979dd5b2",
                "Wed, 24 Jun 2026 18:56:56 GMT",
            ),
            "hse-copyright": (
                "https://www.hse.gov.uk/help/copyright.htm",
                39349,
                "8cdc470fc361541b29a3fac77c27a93f80eda59cd480eed45bc975d76b522849",
                "Wed, 24 Jun 2026 18:43:43 GMT",
            ),
        }
        self.assertEqual(len(self.provenance["pages"]), 8)
        for page in self.provenance["pages"]:
            url, size, digest, modified = expected[page["page_id"]]
            self.assertEqual(page["url"], url)
            self.assertEqual(page["http_status"], 200)
            self.assertEqual(page["content_type"], "text/html")
            self.assertEqual(page["raw_byte_length"], size)
            self.assertEqual(page["raw_sha256"], digest)
            self.assertEqual(page["last_modified"], modified)
            self.assertFalse(page["raw_html_retained"])

    def test_page_level_normalized_audit_metrics_are_exact(self) -> None:
        expected = {
            "hse-arboriculture": (6426, 984, "0e937bf2a6d37ae3eb397300bec526a984f321413bb23f53b6cb39beddae9e22"),
            "hse-aerial-work": (2303, 373, "f83137d8ee4c5759da1f303ba91e4c98fba687d3c4d5a7f415b4bf01e2039be7"),
            "hse-tree-climbing": (3258, 540, "0bb232c1566cb8b853a050dd1dd45784df92690e6b0af664518cb88308671f24"),
            "hse-loler-hub": (513, 66, "90766ea705ec288d012027a107f1a2f64dee2dba3627a417664cd573af6a1f48"),
            "hse-thorough-examinations": (10176, 1601, "6d970c6f422bdfa62f5d16b90f329fac65f2b98ec239905ae0184ce4a069b6a9"),
            "hse-treework-height": (2273, 379, "c7f3797e0afeb6cb307752fd54286ca3d52591a4cf1a6fa938320cf1c1b2cc42"),
            "hse-indg367-landing": (564, 89, "205afee3d779341d3cc7263b0f0c1c36698a886d2365ae2084b0c7575b000e73"),
            "hse-copyright": (2649, 422, "f10a3779e7e9bf1b94dda8488f95319f9b37e8955c251dc3874cba5e3a5c4986"),
        }
        for page in self.provenance["pages"]:
            byte_count, word_count, digest = expected[page["page_id"]]
            self.assertEqual(page["normalized_main_text_byte_length"], byte_count)
            self.assertEqual(page["normalized_main_text_word_count"], word_count)
            self.assertEqual(page["normalized_main_text_sha256"], digest)
        summary = self.provenance["retrieval_summary"]
        self.assertEqual(summary["page_count"], 8)
        self.assertEqual(summary["raw_byte_length_total"], 325943)
        self.assertEqual(summary["normalized_main_text_byte_length_total"], 28162)
        self.assertEqual(summary["normalized_main_text_word_count_total"], 4454)
        self.assertTrue(summary["page_level_raw_hashes_recorded"])

    def test_only_allowlisted_hse_pages_were_accessed(self) -> None:
        pages = self.provenance["pages"]
        expected_urls = [page["url"] for page in pages]
        network = self.provenance["network_boundary"]
        self.assertEqual(network["head_and_get_routes_used"], expected_urls)
        for key in (
            "external_legislation_routes_accessed",
            "external_industry_routes_accessed",
            "video_routes_accessed",
            "linked_pdf_routes_accessed",
            "archived_routes_accessed",
            "image_or_multimedia_routes_accessed",
            "other_hse_body_routes_accessed",
        ):
            self.assertFalse(network[key])
        self.assertEqual(len(self.manifest["entries"]), 8)
        self.assertEqual([e["url"] for e in self.manifest["entries"]], expected_urls)

    def test_raw_html_and_normalized_extraction_are_not_retained(self) -> None:
        files = sorted(
            str(path.relative_to(ROOT)) for path in ROOT.rglob("*") if path.is_file()
        )
        self.assertFalse(any(name.endswith((".html", ".htm")) for name in files))
        self.assertFalse(any("normalized" in name for name in files))
        self.assertEqual(
            sorted(p.name for p in (ROOT / "source_snapshot").iterdir()),
            ["provenance.json"],
        )
        method = self.provenance["normalization_and_review_method"]
        self.assertFalse(method["raw_html_retained"])
        self.assertFalse(method["raw_html_direct_training"])
        self.assertFalse(method["normalized_extraction_retained"])
        self.assertFalse(method["normalized_extraction_direct_training"])

    def test_ogl_v3_rights_and_attribution_are_present(self) -> None:
        rights = self.provenance["rights_review"]
        self.assertEqual(rights["license_name"], "Open Government Licence v3.0")
        self.assertEqual(
            rights["license_url"],
            "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/",
        )
        preferred = (
            "Contains public sector information published by the Health and Safety "
            "Executive and licensed under the Open Government Licence."
        )
        self.assertEqual(rights["preferred_acknowledgement"], preferred)
        self.assertIn(preferred, self.corpus)
        self.assertIn("eligible HSE Crown text", self.corpus)
        self.assertFalse(rights["hse_logo_reused"])
        self.assertFalse(rights["image_or_illustration_reused"])
        self.assertFalse(rights["video_or_multimedia_reused"])
        self.assertFalse(rights["linked_publication_reused"])
        self.assertFalse(rights["legal_opinion"])

    def test_logo_media_svg_video_and_linked_works_are_excluded(self) -> None:
        audit = self.provenance["component_and_link_audit"]
        self.assertEqual(audit["substantive_main_content_img_tags"], 0)
        self.assertEqual(audit["substantive_main_content_video_tags"], 0)
        self.assertEqual(audit["substantive_main_content_iframe_tags"], 0)
        self.assertEqual(audit["shared_back_to_top_svg_occurrences_in_main_container"], 7)
        self.assertEqual(audit["external_video_links_found"], 1)
        self.assertEqual(audit["external_video_links_accessed"], 0)
        self.assertEqual(audit["linked_pdf_routes_accessed"], 0)
        self.assertEqual(audit["third_party_routes_accessed"], 0)
        self.assertFalse(audit["raw_html_direct_training"])
        self.assertIn("HSE logos", self.corpus)
        self.assertIn("videos", self.corpus)
        self.assertIn("linked publications", self.corpus)

    def test_uk_occupational_arboriculture_scope_is_prominent(self) -> None:
        for phrase in (
            "paid or managed tree work in the United Kingdom",
            "occupational arboriculture system",
            "not automatically a standard of care outside UK occupational arboriculture",
            "Keep the United Kingdom occupational arboriculture context attached",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(
            self.provenance["jurisdiction_and_domain"],
            "United Kingdom occupational health and safety guidance for arboriculture and work at height",
        )
        self.assertEqual(
            self.manifest["jurisdiction_transfer_status"],
            "uk_occupational_treework_context_required_no_universal_legal_transfer",
        )

    def test_planning_supervision_and_task_specific_competence_are_retained(self) -> None:
        for phrase in (
            "properly planned, appropriately supervised, and carried out safely",
            "risk assessment, suitable equipment, competent managers and workers",
            "planning for emergencies and rescue",
            "training specific to the task",
            "Different responsibilities need different competence",
            "practical skill, theoretical understanding, experience, awareness of limits",
        ):
            self.assertIn(phrase, self.corpus)

    def test_person_is_load_and_anchor_is_inside_system_boundary(self) -> None:
        for phrase in (
            "a load includes a person",
            "attachments used for anchoring, fixing, or supporting it",
            "anchor is a lifting-system component",
            "person cannot be omitted from the load case",
            "The anchor is a lifting-system component, not scenery outside the system boundary",
            "Equipment meant for material handling is not automatically equipment for lifting or supporting a person",
        ):
            self.assertIn(phrase, self.corpus)
        boundary = self.provenance["evidence_and_scope_audit"]["retained_system_boundary"]
        self.assertEqual(len(boundary), 5)
        self.assertEqual(
            self.manifest["system_boundary_status"],
            "person_is_load_supporting_attachments_and_tree_anchor_are_system_components",
        )

    def test_tree_context_keeps_species_age_condition_and_disease(self) -> None:
        for phrase in (
            "differences among species and the effects of age, condition, and disease",
            "Visible health alone does not prove capacity",
            "naming a species does not establish",
            "diseased or decayed trees",
            "does not provide a repeatable field test that certifies a branch",
        ):
            self.assertIn(phrase, self.corpus)

    def test_anchor_line_and_supplementary_roles_keep_common_mode_caveats(self) -> None:
        for phrase in (
            "two load-bearing anchor points",
            "working line and a safety line that are separately anchored",
            "Supplementary anchors",
            "any two branches create redundancy",
            "share a common failure",
            "not automatically a full-strength backup",
            "does not operationalize the exception",
            "not convenience or habit",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(
            self.manifest["redundancy_status"],
            "roles_retained_quantity_does_not_prove_independence_or_capacity",
        )

    def test_swing_drift_and_geometry_are_conceptual_not_construction(self) -> None:
        for phrase in (
            "uncontrolled swings can lead to impact",
            "drift, free fall, unintended release, and pendulum swing",
            "change the direction and magnitude of force on an anchor",
            "That does not make “add another line” a complete instruction",
            "This digest gives no placement or construction method",
        ):
            self.assertIn(phrase, self.corpus)

    def test_ground_roles_communication_and_line_management_are_complete(self) -> None:
        for phrase in (
            "ground team as passive observers",
            "plan the job with the climber",
            "maintain communication",
            "watch the climber",
            "share workload",
            "knots, kinks, tangles, loose branch material, machinery, obstructions, vehicles",
            "not to wrap a working rope around any part of the body",
            "unintended anchor or entanglement point",
        ):
            self.assertIn(phrase, self.corpus)

    def test_stop_and_reassess_is_an_active_control(self) -> None:
        for phrase in (
            "If unsure, stop and reassess",
            "continue assessing the operation",
            "modify the plan and risk assessment",
            "Stopping is a control, not an admission of incompetence",
            "The need to continue does not make an unresolved condition acceptable",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(
            self.manifest["reassessment_status"],
            "continual_assessment_stop_and_reassess_when_unsure",
        )

    def test_rescue_readiness_is_preplanned_but_not_taught(self) -> None:
        for phrase in (
            "at least two people present",
            "available, competent, and equipped to perform aerial rescue without delay",
            "reliable means of rescue",
            "rescue capability must exist before the emergency",
            "does not teach aerial rescue",
            "does not define how a casualty should be reached, transferred, lowered",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(
            self.manifest["rescue_status"],
            "readiness_staffing_competence_and_availability_retained_no_rescue_procedure",
        )

    def test_daily_weekly_interim_maintenance_and_thorough_layers_are_distinct(self) -> None:
        for phrase in (
            "inspected each day by a competent person",
            "daily pre-use check",
            "written weekly inspection record",
            "a **pre-use check**",
            "an **interim inspection**",
            "**maintenance**",
            "a **thorough examination**",
            "systematic, detailed examination",
            "practical and theoretical knowledge and experience",
            "should not simply assess their own maintenance work",
        ):
            self.assertIn(phrase, self.corpus)
        self.assertEqual(
            self.manifest["inspection_status"],
            "layered_daily_recorded_interim_maintenance_and_thorough_examination",
        )

    def test_defect_withdrawal_and_traceable_records_are_retained(self) -> None:
        for phrase in (
            "protected from unauthorized alteration",
            "competent observer, date, inspection type",
            "withdrawn or quarantined",
            "not be used until the risk is effectively addressed",
            "documentation cannot add physical strength",
        ):
            self.assertIn(phrase, self.corpus)

    def test_exact_ratings_intervals_devices_and_operational_steps_are_absent(self) -> None:
        lower = self.corpus.lower()
        for marker in (
            "3-way action gate",
            "prussic",
            "mechanical descender",
            "footlocking",
            "single rope technique",
            "mobile elevating work platform",
            "safe working load (swl)",
            "every 6 months",
            "every 12 months",
            "schedule 1 of loler",
            "regulation 9",
            "11 items listed",
            "lantra awards",
            "city and guilds",
            "youtube",
        ):
            self.assertNotIn(marker, lower)
        self.assertNotRegex(lower, r"\b\d+\s*(?:kg|kn|months?)\b")
        self.assertEqual(
            self.manifest["operational_detail_status"],
            "excluded_no_exact_ratings_intervals_devices_climbing_construction_or_rescue_steps",
        )

    def test_no_tree_hardpoint_bondage_or_human_suspension_certification(self) -> None:
        for phrase in (
            "No statement in this digest certifies a tree, branch, tree union, anchor, hardpoint",
            "cannot be copied into bondage and presented as approved",
            "They do not study recreational rope bondage",
            "Never convert this guidance into tree, hardpoint, bondage, or human-suspension certification",
            "not load-bearing approval",
        ):
            self.assertIn(phrase, self.corpus + self.report)
        boundary = self.provenance["cross_domain_boundary"]
        self.assertEqual(len(boundary["forbidden_certifications"]), 9)
        self.assertEqual(len(boundary["forbidden_transfers"]), 6)
        self.assertEqual(
            self.manifest["cross_domain_status"],
            "forbidden_no_tree_hardpoint_bondage_or_human_suspension_certification",
        )

    def test_dispositions_are_complete_and_sequential(self) -> None:
        self.assertEqual(len(self.dispositions), 33)
        self.assertEqual(sum(r["direct_training_included"] for r in self.dispositions), 20)
        self.assertEqual(sum(not r["direct_training_included"] for r in self.dispositions), 13)
        self.assertEqual(
            [r["record_id"] for r in self.dispositions],
            [f"hse-treework-{n:03d}" for n in range(1, 34)],
        )
        self.assertEqual(self.manifest["disposition_record_count"], 33)
        self.assertEqual(self.manifest["included_or_narrowed_disposition_count"], 20)
        self.assertEqual(self.manifest["excluded_disposition_count"], 13)

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
            "URL",
            "title",
            "retrieval date",
            "regulation number",
            "equipment rating",
            "statutory interval",
            "device",
            "brand",
            "knot",
            "technique",
            "legal citation",
        ):
            self.assertIn(forbidden, policy["forbidden"])
        self.assertTrue(policy["inherit_document_split"])

    def test_split_hygiene_and_protected_data_boundary(self) -> None:
        split = self.provenance["split_hygiene"]
        self.assertTrue(split["non_qa"])
        self.assertEqual(
            split["source_document_unit"], "HSE-treework-lifting-climbing-2026-07-16"
        )
        self.assertFalse(split["protected_data_accessed_during_build"])
        self.assertEqual(len(split["protected_data_classes_not_accessed"]), 10)
        self.assertIn("one source-family split", split["document_disjoint_requirement"])
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
        citation_count = len(re.findall(r"\[HSE [^\]]+\]", self.corpus))
        self.assertEqual(regex_count, 3592)
        self.assertEqual(whitespace_count, 3605)
        self.assertEqual(citation_count, 28)
        self.assertEqual(self.manifest["direct_training_word_count"], regex_count)
        self.assertEqual(
            self.manifest["whitespace_delimited_word_count"], whitespace_count
        )
        self.assertEqual(self.manifest["page_label_citation_count"], citation_count)
        self.assertIn("3,605-whitespace-word", self.report)
        self.assertIn("3,592 Unicode-regex words", self.report)
        self.assertIn("33 disposition records", self.report)


if __name__ == "__main__":
    unittest.main()
