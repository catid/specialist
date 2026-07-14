#!/usr/bin/env python3
"""Focused regression tests for the deterministic context-merit audit."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_context_merit_audit as builder
from qa_quality import normalize_text


class ContextMeritAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        builder.main()
        self.audit = builder.read_jsonl(builder.AUDIT)
        self.curation = builder.read_jsonl(builder.CURATION)

    def test_selection_is_exact_and_excludes_prior_reviews(self) -> None:
        active = builder.read_jsonl(builder.ACTIVE_DATASET)
        ranked, _ = builder.ranked_unreviewed(active)
        self.assertEqual(
            tuple(item["row"]["fact_id"] for item in ranked[:25]),
            builder.EXPECTED_SELECTION,
        )
        reviewed = builder.reviewed_fact_ids()
        for item in ranked[:25]:
            row = item["row"]
            self.assertNotIn(row["fact_id"], reviewed)
            self.assertNotIn("review", row)
            self.assertNotIn("curation", row)

    def test_ranking_features_and_order_are_reproducible(self) -> None:
        active = builder.read_jsonl(builder.ACTIVE_DATASET)
        first, _ = builder.ranked_unreviewed(active)
        second, _ = builder.ranked_unreviewed(active)
        first_keys = [
            (item["row"]["fact_id"], item["features"])
            for item in first[:25]
        ]
        second_keys = [
            (item["row"]["fact_id"], item["features"])
            for item in second[:25]
        ]
        self.assertEqual(first_keys, second_keys)
        self.assertTrue(all(
            item["features"]["risk_score"] >= 9 for item in first[:25]))

    def test_sources_and_evidence_are_hash_pinned(self) -> None:
        by_id = {row["fact_id"]: row for row in self.audit}
        for specification in builder.SPECS:
            row = by_id[specification["fact_id"]]
            self.assertEqual(
                builder.file_sha256(specification["source_path"]),
                row["source_document_file_sha256"],
            )
            self.assertEqual(
                builder.text_sha256(row["support_evidence"]),
                row["support_evidence_sha256"],
            )
            supported_answer = specification.get(
                "answer", row["active_answer"])
            self.assertIn(
                normalize_text(supported_answer),
                normalize_text(row["support_evidence"]),
            )

    def test_manual_decisions_cover_all_25(self) -> None:
        self.assertEqual(len(self.audit), 25)
        counts = {
            decision: sum(row["decision"] == decision for row in self.audit)
            for decision in {"keep", "drop", "edit"}
        }
        self.assertEqual(counts, {"keep": 17, "drop": 5, "edit": 3})
        self.assertEqual(
            tuple(row["fact_id"] for row in self.audit),
            builder.EXPECTED_SELECTION,
        )

    def test_pending_curation_is_append_safe_and_extractive(self) -> None:
        self.assertEqual(len(self.curation), 8)
        self.assertEqual(
            {action: sum(row["action"] == action for row in self.curation)
             for action in {"drop", "edit"}},
            {"drop": 5, "edit": 3},
        )
        prior = {
            row["fact_id"]
            for row in builder.read_jsonl(builder.QUALITY_MERIT_CURATION)
        }
        self.assertFalse(prior & {row["fact_id"] for row in self.curation})
        for row in self.curation:
            if row["action"] == "edit":
                self.assertEqual(row["support_type"], "extractive")
                self.assertIn(normalize_text(row["answer"]),
                              normalize_text(row["evidence"]))

    def test_report_is_pending_and_projection_is_identity_checked(self) -> None:
        report = json.loads(builder.REPORT.read_text())
        self.assertEqual(report["status"], "segregated_pending_not_promoted")
        self.assertEqual(report["audit"]["by_decision"],
                         {"drop": 5, "edit": 3, "keep": 17})
        projection = report["isolated_build_projection"]
        self.assertEqual(projection["output_rows"], 795)
        self.assertEqual(projection["reviewed_keep_fact_ids_preserved"], 17)
        self.assertEqual(projection["prior_pending_addition_fact_ids_preserved"],
                         38)
        self.assertEqual(projection["new_drops_applied"], 5)
        self.assertEqual(projection["new_edits_applied"], 3)
        self.assertEqual(projection["unexpected_fact_ids"], 0)
        self.assertTrue(projection["repeat_dataset_byte_identical"])
        policy = report["sealed_evaluation_policy"]
        self.assertFalse(policy["manual_review_opened_eval_or_heldout_content"])
        self.assertFalse(policy["generator_opens_eval_or_heldout_content"])

    def test_generator_is_byte_deterministic(self) -> None:
        first = tuple(builder.file_sha256(path) for path in
                      (builder.AUDIT, builder.CURATION, builder.REPORT))
        builder.main()
        second = tuple(builder.file_sha256(path) for path in
                       (builder.AUDIT, builder.CURATION, builder.REPORT))
        self.assertEqual(first, second)

    def test_active_s5_is_frozen(self) -> None:
        self.assertEqual(
            builder.file_sha256(builder.ACTIVE_DATASET),
            "62e7ae28c86a458d4d33bf3f73f1b91b873c86e3f70ce87706a7394d1f391507",
        )
        self.assertEqual(
            builder.file_sha256(builder.ACTIVE_REPORT),
            "3ca1cbf68dcc281176b25ca14a894edd6db7a3c705849d97f4efcdc6240b974a",
        )
        self.assertEqual(
            tuple(builder.file_sha256(path)
                  for path in builder.ACTIVE_CURATIONS),
            (
                "ee28db85bea4a74b0114b4312f52129bc8699a2b4b74c02b755f363e8198edeb",
                "8b68caf2a6a4411f9ae26a05759c6ceb6852eaf05c28171b4a11dd93033ebadd",
            ),
        )


if __name__ == "__main__":
    unittest.main()
