#!/usr/bin/env python3
"""Contract tests for source discovery and canonical Markdown corpus queues."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CORPUS_QUEUE = ROOT / "site_corpus_queue_v1.json"
DISCOVERY_QUEUE = ROOT / "site_discovery_queue_v1.json"


class SourceCorpusContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = json.loads(CORPUS_QUEUE.read_text(encoding="utf-8"))
        cls.discovery = json.loads(DISCOVERY_QUEUE.read_text(encoding="utf-8"))

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
        self.assertEqual(len(queue), 25)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(urls), len(set(urls)))
        self.assertIn("crash_restraint", ids)
        crash = next(row for row in queue if row["resource_id"] == "crash_restraint")
        self.assertIn("entire accessible site inventory", crash["scope"])

    def test_discovery_is_evidence_first_and_separate_from_extraction(self) -> None:
        contract = self.discovery["discovery_contract"]
        self.assertTrue(contract["evaluate_the_actual_source_not_search_snippets"])
        self.assertTrue(contract["deduplicate_against_existing_corpus_queue"])
        self.assertTrue(contract["separate_source_discovery_from_corpus_extraction"])
        self.assertTrue(contract["sealed_eval_ood_shadow_holdout_and_existing_qa_access_forbidden"])
        self.assertIn("novel_coverage_score_0_to_5", self.discovery["candidate_fields"])
        self.assertIn("accept_targeted_scope", self.discovery["decisions"])


if __name__ == "__main__":
    unittest.main()
