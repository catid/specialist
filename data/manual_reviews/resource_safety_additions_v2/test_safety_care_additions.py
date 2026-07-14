#!/usr/bin/env python3
"""Regression tests for pending safety/care additions tranche 2."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_safety_care_additions as builder
from qa_quality import leakage_reason, qa_pair_from_record


class SafetyCareAdditionsTest(unittest.TestCase):
    def setUp(self) -> None:
        builder.main()
        self.rows = builder.read_jsonl(builder.OUTPUT)

    def test_source_and_evidence_are_hash_pinned(self) -> None:
        for fact in builder.MANUAL_FACTS:
            specification = builder.SOURCE_DOCUMENTS[fact["document"]]
            document = json.loads(specification["path"].read_text())
            self.assertEqual(
                builder.text_sha256(document["text"]),
                specification["document_sha256"],
            )
            self.assertIn(fact["evidence"], document["text"])
            self.assertIn(fact["answer"], fact["evidence"])

    def test_generated_rows_are_canonical_and_unique(self) -> None:
        self.assertEqual(len(self.rows), 10)
        self.assertEqual(len({row["fact_id"] for row in self.rows}), 10)
        self.assertEqual(len({row["question"] for row in self.rows}), 10)
        for row in self.rows:
            self.assertEqual(
                qa_pair_from_record(row),
                (row["question"], row["answer"]),
            )
            self.assertIn(row["answer"], row["evidence"])

    def test_no_active_or_prior_pending_collision(self) -> None:
        comparison = builder.qa_facts(
            (builder.ACTIVE_DATASET, *builder.PRIOR_PENDING),
            "active_or_prior_pending",
        )
        for row in self.rows:
            self.assertIsNone(leakage_reason(
                row["question"], row["answer"], comparison))

    def test_report_is_pending_and_respects_sealed_eval_policy(self) -> None:
        report = json.loads(builder.REPORT.read_text())
        self.assertEqual(report["status"], "segregated_pending_not_promoted")
        self.assertEqual(report["output"]["rows"], 10)
        policy = report["sealed_evaluation_policy"]
        self.assertFalse(policy["manual_review_opened_eval_or_heldout_content"])
        self.assertFalse(policy["generator_opens_eval_or_heldout_content"])
        self.assertEqual(
            report["isolated_build_projection"]["output_rows"], 804)

    def test_generator_is_byte_deterministic(self) -> None:
        first = (builder.file_sha256(builder.OUTPUT),
                 builder.file_sha256(builder.REPORT))
        builder.main()
        second = (builder.file_sha256(builder.OUTPUT),
                  builder.file_sha256(builder.REPORT))
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
        expected_curations = (
            "ee28db85bea4a74b0114b4312f52129bc8699a2b4b74c02b755f363e8198edeb",
            "8b68caf2a6a4411f9ae26a05759c6ceb6852eaf05c28171b4a11dd93033ebadd",
        )
        self.assertEqual(
            tuple(builder.file_sha256(path)
                  for path in builder.ACTIVE_CURATIONS),
            expected_curations,
        )


if __name__ == "__main__":
    unittest.main()
