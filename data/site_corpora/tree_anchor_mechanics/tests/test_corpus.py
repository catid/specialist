from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REASON = "treeconsult_copyright_notice_requires_written_permission_for_text_reuse"
INTERNAL_REASON = "treeconsult_written_permission_required_for_text_reuse"
OUTSIDE_REASON = "outside_requested_tree_anchor_mechanics_and_written_permission_required"
EXTERNAL_REASON = "external_landing_404_and_separate_rights_review_required"
OUTPUTS = ("CORPUS.md", "REPORT.md", "content_records.jsonl", "inventory.jsonl", "manifest.json", "url_dispositions.jsonl")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = read_json(ROOT / "manifest.json")
        cls.policy = read_json(ROOT / "policy_decision.json")
        cls.provenance = read_json(ROOT / "source_snapshot/provenance.json")
        cls.snapshot_inventory = read_json(ROOT / "source_snapshot/document_inventory.json")
        cls.inventory = read_jsonl(ROOT / "inventory.jsonl")
        cls.dispositions = read_jsonl(ROOT / "url_dispositions.jsonl")
        cls.corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")

    def test_rights_deferral_and_zero_training_surface(self) -> None:
        self.assertEqual(self.policy["decision_reason"], REASON)
        self.assertEqual(self.manifest["policy_reason"], REASON)
        self.assertFalse(self.policy["direct_training_ready"])
        self.assertFalse(self.manifest["direct_training_ready"])
        self.assertTrue(self.manifest["non_qa"])
        self.assertEqual(self.manifest["artifact_role"], "rights_deferred_metadata_inventory")
        self.assertEqual((ROOT / "content_records.jsonl").read_bytes(), b"")
        self.assertEqual(self.manifest["content_record_count"], 0)
        self.assertEqual(self.manifest["training_document_count"], 0)
        self.assertEqual(self.manifest["training_section_count"], 0)
        self.assertEqual(self.manifest["direct_training_word_count"], 0)

    def test_exact_short_rights_evidence_and_policy_body_not_retained(self) -> None:
        fragment = "no texts, excerpts of texts, images or parts of images may otherwise be used without written permission"
        self.assertEqual(self.policy["rights_evidence_fragment"], fragment)
        self.assertEqual(self.policy["rights_evidence_fragment_word_count"], 17)
        self.assertLess(self.policy["rights_evidence_fragment_word_count"], 25)
        self.assertFalse(self.policy["legal_notice_body_snapshot_retained"])
        self.assertFalse(self.provenance["legal_notice"]["body_snapshot_retained"])
        self.assertFalse(self.provenance["downloads_index"]["body_snapshot_retained"])
        self.assertEqual(self.policy["audited_at"], self.provenance["completed_at"])
        self.assertEqual(self.policy["legal_notice_body_sha256"], self.provenance["legal_notice"]["body_sha256"])
        self.assertEqual(self.policy["legal_notice_body_byte_length"], self.provenance["legal_notice"]["body_byte_length"])
        self.assertNotIn("legal-notice.htm", {path.name for path in (ROOT / "source_snapshot").iterdir()})
        self.assertNotIn("downloads.htm", {path.name for path in (ROOT / "source_snapshot").iterdir()})

    def test_research_bodies_were_never_requested_or_retained(self) -> None:
        self.assertEqual(self.provenance["research_document_get_requests"], 0)
        self.assertEqual(self.provenance["research_resource_head_requests"], 19)
        self.assertEqual(self.provenance["canonical_instructional_body_snapshots_retained"], 0)
        self.assertEqual(self.provenance["inspected_html_body_snapshots_retained"], 0)
        for item in self.snapshot_inventory:
            self.assertEqual(item["access_audit"]["method"], "HEAD")
            self.assertFalse(item["body_retrieved"])
            self.assertFalse(item["body_snapshot_retained"])
        self.assertFalse(list(ROOT.rglob("*.pdf")))

    def test_exact_inventory_provenance_dates_versions_and_dispositions(self) -> None:
        self.assertEqual(len(self.snapshot_inventory), 19)
        self.assertEqual(len(self.inventory), 19)
        self.assertEqual([item["source_order"] for item in self.snapshot_inventory], list(range(1, 20)))
        self.assertEqual({item["inventory_id"] for item in self.snapshot_inventory}, {item["inventory_id"] for item in self.inventory})
        self.assertEqual({item["url"] for item in self.snapshot_inventory}, {item["url"] for item in self.inventory})
        self.assertEqual(len({item["url"] for item in self.inventory}), 19)
        for item in self.inventory:
            self.assertTrue(item["title_as_displayed"])
            self.assertTrue(item["citation_as_displayed"])
            self.assertIn("authors_as_displayed", item)
            self.assertTrue(item["publication_metadata"]["date_as_displayed"])
            self.assertTrue(item["publication_metadata"]["year_claims"])
            self.assertIn("version_as_displayed", item["publication_metadata"])
            self.assertEqual(item["disposition"], "excluded_rights_deferred")
            self.assertFalse(item["content_retrieved_for_corpus"])
            self.assertFalse(item["direct_training_included"])

    def test_access_and_scope_counts_are_exact(self) -> None:
        internal = [item for item in self.inventory if item["hosted_by_treeconsult"]]
        external = [item for item in self.inventory if not item["hosted_by_treeconsult"]]
        candidate = [item for item in self.inventory if item["scope_class"].startswith("candidate_")]
        outside = [item for item in self.inventory if item["scope_class"] == "outside_requested_tree_anchor_mechanics"]
        self.assertEqual((len(internal), len(external), len(candidate), len(outside)), (18, 1, 17, 1))
        for item in internal:
            self.assertEqual(item["access_audit"]["http_status"], 200)
            self.assertEqual(item["access_audit"]["content_type"], "application/pdf")
            expected = OUTSIDE_REASON if item in outside else INTERNAL_REASON
            self.assertEqual(item["reason"], expected)
        self.assertEqual(external[0]["url"], "https://www.hse.gov.uk/research/rrhtm/rr668.htm")
        self.assertEqual(external[0]["access_audit"]["http_status"], 404)
        self.assertEqual(external[0]["reason"], EXTERNAL_REASON)
        self.assertEqual(self.manifest["inventory_record_count"], 19)
        self.assertEqual(self.manifest["same_domain_pdf_count"], 18)
        self.assertEqual(self.manifest["same_domain_pdf_head_200_count"], 18)
        self.assertEqual(self.manifest["candidate_scope_record_count"], 17)

    def test_conflicting_metadata_is_preserved_not_silently_resolved(self) -> None:
        by_id = {item["inventory_id"]: item for item in self.inventory}
        first_conflict = by_id["treeconsult-rigging-002"]["publication_metadata"]
        second_conflict = by_id["treeconsult-rigging-006"]["publication_metadata"]
        self.assertEqual(first_conflict["year_claims"], [2019, 2018])
        self.assertEqual(first_conflict["status"], "conflicting_years_preserved")
        self.assertEqual(second_conflict["year_claims"], [2012, 2011])
        self.assertEqual(second_conflict["status"], "conflicting_years_preserved")

    def test_snapshot_hashes_lengths_and_robots_access_signal(self) -> None:
        snapshot = self.manifest["source_snapshot"]
        for item in snapshot.values():
            path = ROOT / item["path"]
            self.assertEqual(item["sha256"], sha(path))
            self.assertEqual(item["byte_length"], path.stat().st_size)
        robots = (ROOT / "source_snapshot/robots.txt").read_text(encoding="utf-8")
        self.assertIn("User-agent: *", robots)
        self.assertIn("Disallow: /cms/", robots)
        self.assertIn("Disallow: /includes/", robots)
        for key in ("robots", "legal_notice", "downloads_index"):
            self.assertIsNone(self.provenance[key]["content_signal"])
            self.assertIsNone(self.provenance[key]["x_robots_tag"])

    def test_capture_utility_enforces_narrow_network_boundary(self) -> None:
        spec = importlib.util.spec_from_file_location("capture_metadata", ROOT / "capture_metadata.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader
        spec.loader.exec_module(module)
        for url in (
            "https://www.tree-consult.org/robots.txt",
            "https://www.tree-consult.org/legal-notice.htm",
            "https://www.tree-consult.org/downloads.htm",
        ):
            module.validate_get_url(url)
        for url in (
            "https://www.tree-consult.org/",
            "https://www.tree-consult.org/research-projects.htm",
            "https://www.tree-consult.org/upload/mediapool/pdf/rigging_und_seilklettertechnik/fangstoss.pdf",
            "https://example.com/robots.txt",
        ):
            with self.assertRaises(ValueError):
                module.validate_get_url(url)
        module.validate_head_url("https://www.tree-consult.org/upload/mediapool/pdf/rigging_und_seilklettertechnik/fangstoss.pdf")
        module.validate_head_url("https://www.hse.gov.uk/research/rrhtm/rr668.htm")
        for url in (
            "https://www.tree-consult.org/downloads.htm",
            "https://www.tree-consult.org/upload/mediapool/pdf/baumkontrolle/example.pdf",
            "https://example.com/report.pdf",
        ):
            with self.assertRaises(ValueError):
                module.validate_head_url(url)

    def test_taxonomy_gaps_no_human_suspension_claim_and_split_rule(self) -> None:
        self.assertEqual(self.manifest["taxonomy_mappings"], [])
        self.assertEqual(
            self.manifest["taxonomy_mapping_status"],
            "no_training_sections_due_to_rights_deferral",
        )
        self.assertEqual(
            self.manifest["candidate_taxonomy_scope_for_permission_planning_only"],
            ["rigging_mechanics", "uplines_suspension_hardpoints"],
        )
        self.assertGreaterEqual(len(self.manifest["genuine_gaps"]), 8)
        self.assertIn("neither certifies a tree or hardpoint", self.corpus)
        self.assertIn("before Markdown chunking or QA derivation", self.manifest["document_disjoint_requirement"])
        self.assertIn("sealed-holdout", self.manifest["protected_split_requirement"])

    def test_compliance_notice_has_no_document_title_or_url_training_layer(self) -> None:
        for item in self.snapshot_inventory:
            self.assertNotIn(item["title_as_displayed"], self.corpus)
            self.assertNotIn(item["url"], self.corpus)
        self.assertNotIn("/upload/mediapool/pdf/", self.corpus)
        for marker in ("<|im_start|>", "<|im_end|>", "</think>", "Question:", "Answer:", "Which canonical"):
            self.assertNotIn(marker, self.corpus)

    def test_manifest_output_hashes_and_offline_build_determinism(self) -> None:
        self.assertEqual(self.manifest["policy_decision"]["sha256"], sha(ROOT / "policy_decision.json"))
        for name, item in self.manifest["outputs"].items():
            self.assertEqual(item["sha256"], sha(ROOT / name))
            self.assertEqual(item["byte_length"], (ROOT / name).stat().st_size)
        before = {name: sha(ROOT / name) for name in OUTPUTS}
        subprocess.run([sys.executable, str(ROOT / "build.py")], cwd=ROOT, check=True)
        after = {name: sha(ROOT / name) for name in OUTPUTS}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
