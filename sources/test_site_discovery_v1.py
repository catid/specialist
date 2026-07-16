#!/usr/bin/env python3
"""Contract tests for source discovery and canonical Markdown corpus queues."""

import json
import unittest
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
CORPUS_QUEUE = ROOT / "site_corpus_queue_v1.json"
DISCOVERY_QUEUE = ROOT / "site_discovery_queue_v1.json"
CANDIDATE_LEDGER = ROOT / "site_discovery_candidates_v1.jsonl"
DISCOVERY_REPORT = ROOT / "site_discovery_report_v1.md"


SCORE_FIELDS = {
    "information_density": "information_density_score_0_to_5",
    "technical_specificity": "technical_specificity_score_0_to_5",
    "provenance_quality": "provenance_quality_score_0_to_5",
    "durability": "durability_score_0_to_5",
    "novel_coverage": "novel_coverage_score_0_to_5",
    "safety_context": "safety_context_score_0_to_5",
    "duplication_penalty": "duplication_penalty_0_to_5",
    "commerce_noise_penalty": "commerce_noise_penalty_0_to_5",
}


class SourceCorpusContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = json.loads(CORPUS_QUEUE.read_text(encoding="utf-8"))
        cls.discovery = json.loads(DISCOVERY_QUEUE.read_text(encoding="utf-8"))
        cls.candidates = [
            json.loads(line)
            for line in CANDIDATE_LEDGER.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cls.report = DISCOVERY_REPORT.read_text(encoding="utf-8")

    def test_markdown_and_qa_are_distinct_required_layers(self) -> None:
        layers = self.corpus["dataset_layers"]
        markdown = layers["canonical_markdown_source_corpus"]
        qa = layers["derived_manual_qa"]
        self.assertTrue(markdown["required"])
        self.assertTrue(markdown["direct_training_ready"])
        self.assertTrue(markdown["non_qa"])
        self.assertTrue(qa["must_link_to_markdown_source_record"])
        self.assertTrue(qa["automatic_generation_without_manual_review_forbidden"])

    def test_approved_taxonomy_is_complete_and_unique(self) -> None:
        expected = {
            "lineage_history_people",
            "ties_knots_frictions",
            "uplines_suspension_hardpoints",
            "anatomy_injury_prevention",
            "rope_materials_maintenance",
            "rigging_mechanics",
            "consent_scene_management",
            "bottoming_skills",
            "pattern_architecture",
            "emergency_procedures",
            "rope_handling",
            "teaching_evaluation",
            "accessibility_adaptation",
            "aesthetics_performance",
            "terminology_cultural_context",
        }
        ids = [row["category_id"] for row in self.corpus["knowledge_taxonomy"]["categories"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), expected)
        for row in self.corpus["knowledge_taxonomy"]["categories"]:
            self.assertGreaterEqual(len(row["coverage"].split()), 8)

    def test_every_requested_resource_has_one_unique_queue_entry(self) -> None:
        queue = self.corpus["queue"]
        ids = [row["resource_id"] for row in queue]
        urls = [row["url"] for row in queue]
        self.assertGreaterEqual(len(queue), 25)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(urls), len(set(urls)))
        self.assertIn("crash_restraint", ids)
        crash = next(row for row in queue if row["resource_id"] == "crash_restraint")
        self.assertIn("entire accessible site inventory", crash["scope"])

    def test_rope365_completion_record_is_preserved(self) -> None:
        rope365 = next(
            row for row in self.corpus["queue"] if row["resource_id"] == "rope365"
        )
        self.assertEqual(rope365["status"], "complete")
        self.assertEqual(rope365["worker"], "rope365_corpus_clean")
        self.assertEqual(
            rope365["commit"], "30222126ffaf67404417f8dd892c18c2f956e1e0"
        )
        self.assertEqual(
            set(rope365["artifacts"]), {"manifest", "markdown", "report", "tests"}
        )

    def test_discovery_is_evidence_first_and_separate_from_extraction(self) -> None:
        contract = self.discovery["discovery_contract"]
        self.assertTrue(contract["evaluate_the_actual_source_not_search_snippets"])
        self.assertTrue(contract["deduplicate_against_existing_corpus_queue"])
        self.assertTrue(contract["separate_source_discovery_from_corpus_extraction"])
        self.assertTrue(contract["sealed_eval_ood_shadow_holdout_and_existing_qa_access_forbidden"])
        self.assertIn("novel_coverage_score_0_to_5", self.discovery["candidate_fields"])
        self.assertIn("accept_targeted_scope", self.discovery["decisions"])

    def test_candidate_ledger_has_complete_manual_review_batches(self) -> None:
        required = set(self.discovery["candidate_fields"]) | {"review_batch"}
        batches = Counter()
        for row in self.candidates:
            self.assertTrue(required.issubset(row), row["candidate_id"])
            batches[row["review_batch"]] += 1
        self.assertTrue(batches)
        for batch_id, count in batches.items():
            self.assertRegex(batch_id, r"^discovery_batch_\d{3}$")
            self.assertGreaterEqual(count, 10)
            self.assertLessEqual(count, 20)

    def test_candidate_identity_access_and_evidence_fields_are_valid(self) -> None:
        ids = [row["candidate_id"] for row in self.candidates]
        urls = [row["canonical_url"] for row in self.candidates]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(urls), len(set(urls)))
        for row in self.candidates:
            parsed = urlparse(row["canonical_url"])
            self.assertIn(parsed.scheme, {"http", "https"})
            self.assertTrue(parsed.netloc)
            self.assertIs(type(row["accessible"]), bool)
            self.assertTrue(row["access_notes"].strip())
            self.assertTrue(row["discovered_from"])
            self.assertTrue(row["representative_pages"])
            datetime.fromisoformat(row["checked_at_utc"].replace("Z", "+00:00"))
            for page in row["representative_pages"]:
                self.assertEqual(set(page), {"url", "note"})
                self.assertIn(urlparse(page["url"]).scheme, {"http", "https"})
                self.assertTrue(page["note"].strip())

    def test_candidate_scores_follow_the_approved_formula_exactly(self) -> None:
        allowed_decisions = set(self.discovery["decisions"])
        for row in self.candidates:
            scores = {name: row[field] for name, field in SCORE_FIELDS.items()}
            for score in scores.values():
                self.assertIs(type(score), int)
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 5)
            expected = (
                2 * scores["information_density"]
                + 2 * scores["technical_specificity"]
                + scores["provenance_quality"]
                + scores["durability"]
                + 2 * scores["novel_coverage"]
                + scores["safety_context"]
                - 2 * scores["duplication_penalty"]
                - 2 * scores["commerce_noise_penalty"]
            )
            self.assertEqual(row["priority_score"], expected, row["candidate_id"])
            self.assertIn(row["decision"], allowed_decisions)

    def test_candidate_taxonomy_is_approved(self) -> None:
        approved = {
            row["category_id"]
            for row in self.corpus["knowledge_taxonomy"]["categories"]
        }
        for row in self.candidates:
            categories = row["supported_taxonomy_categories"]
            self.assertTrue(categories)
            self.assertEqual(len(categories), len(set(categories)))
            self.assertTrue(set(categories).issubset(approved), row["candidate_id"])

    def test_accepted_candidates_are_pending_in_extraction_queue(self) -> None:
        queue_by_id = {row["resource_id"]: row for row in self.corpus["queue"]}
        legacy_urls = {
            row["url"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" not in row
        }
        accepted = [
            row for row in self.candidates if row["decision"].startswith("accept_")
        ]
        self.assertTrue(accepted)
        for candidate in accepted:
            self.assertNotIn(candidate["canonical_url"], legacy_urls)
            queued = queue_by_id[candidate["candidate_id"]]
            self.assertEqual(queued["status"], "pending")
            self.assertEqual(queued["url"], candidate["canonical_url"])
            self.assertEqual(queued["scope"], candidate["recommended_crawl_scope"])
            self.assertEqual(
                queued["discovery_candidate_id"], candidate["candidate_id"]
            )
            self.assertEqual(
                queued["discovery_priority_score"], candidate["priority_score"]
            )
            self.assertEqual(
                queued["supported_taxonomy_categories"],
                candidate["supported_taxonomy_categories"],
            )

    def test_deferred_and_rejected_candidates_are_not_queued(self) -> None:
        queued_candidate_ids = {
            row["discovery_candidate_id"]
            for row in self.corpus["queue"]
            if "discovery_candidate_id" in row
        }
        expected_accepted = {
            row["candidate_id"]
            for row in self.candidates
            if row["decision"].startswith("accept_")
        }
        self.assertEqual(queued_candidate_ids, expected_accepted)

    def test_report_covers_each_review_batch_and_decision(self) -> None:
        for batch_id in {row["review_batch"] for row in self.candidates}:
            self.assertIn(batch_id, self.report)
        for decision in set(self.discovery["decisions"]):
            self.assertIn(decision, self.report)


if __name__ == "__main__":
    unittest.main()
