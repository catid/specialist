#!/usr/bin/env python3
"""Focused regression tests for the active bare-answer merit audit."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_bare_answer_merit_audit as builder
from qa_quality import normalize_text


class BareAnswerMeritAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        builder.main()
        self.audit = builder.read_jsonl(builder.AUDIT)
        self.curation = builder.read_jsonl(builder.CURATION)

    def test_sources_and_evidence_are_hash_pinned(self) -> None:
        for specification in builder.SPECS:
            document = json.loads(specification["raw_path"].read_text())
            self.assertEqual(document["url"], specification["url"])
            self.assertEqual(
                builder.text_sha256(document["text"]),
                specification["document_sha256"],
            )
            self.assertIn(specification["evidence"], document["text"])
            self.assertIn(
                normalize_text(specification["answer"]),
                normalize_text(specification["evidence"]),
            )

    def test_decisions_are_four_keeps_and_one_existing_drop(self) -> None:
        self.assertEqual(len(self.audit), 5)
        self.assertEqual(
            [row["decision"] for row in self.audit],
            ["keep", "keep", "keep", "drop_already_pending", "keep"],
        )
        self.assertEqual(
            [row["answer"] for row in self.audit],
            ["no", "TK", "Ma", "Ryū", "Lay"],
        )

    def test_keeps_are_context_complete_and_source_supported(self) -> None:
        keeps = [row for row in self.audit if row["decision"] == "keep"]
        self.assertEqual(len(keeps), 4)
        for row in keeps:
            self.assertIn(
                row["reason_code"],
                {"concise_answer_context_complete",
                 "technical_term_context_complete"},
            )
            self.assertIn(
                normalize_text(row["answer"]),
                normalize_text(row["evidence"]),
            )

    def test_ryu_drop_is_prior_pending_and_not_duplicated(self) -> None:
        row = next(row for row in self.audit
                   if row["fact_id"] == "fact-6c642360467617e05b13")
        self.assertEqual(row["decision"], "drop_already_pending")
        prior = {
            item["fact_id"]: item
            for item in builder.read_jsonl(builder.QUALITY_MERIT_CURATION)
        }
        self.assertEqual(prior[row["fact_id"]]["action"], "drop")
        self.assertEqual(prior[row["fact_id"]]["reason_code"],
                         "contextless_or_low_value")
        self.assertEqual(self.curation, [])

    def test_report_is_pending_and_projection_is_complete(self) -> None:
        report = json.loads(builder.REPORT.read_text())
        self.assertEqual(report["status"], "segregated_pending_not_promoted")
        self.assertEqual(report["audit"]["by_decision"],
                         {"drop_already_pending": 1, "keep": 4})
        projection = report["isolated_build_projection"]
        self.assertEqual(projection["output_rows"], 800)
        self.assertEqual(projection["reviewed_keep_fact_ids_preserved"], 4)
        self.assertTrue(projection["reviewed_already_pending_drop_applied"])
        self.assertEqual(projection["prior_pending_addition_fact_ids_preserved"],
                         38)
        self.assertEqual(projection["unexpected_fact_ids"], 0)
        self.assertTrue(projection["repeat_byte_identical"])
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
