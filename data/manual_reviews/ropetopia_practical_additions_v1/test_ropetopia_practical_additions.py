#!/usr/bin/env python3
"""Focused regression tests for the pending RopeTopia-lineage tranche."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_ropetopia_practical_additions as builder
from qa_quality import leakage_reason, qa_pair_from_record, stable_fact_id


class RopeTopiaPracticalAdditionsTest(unittest.TestCase):
    def setUp(self) -> None:
        builder.main()
        self.rows = builder.read_jsonl(builder.ADDITIONS)

    def test_successor_sources_map_to_ropetopia_manifest(self) -> None:
        manifest = json.loads(builder.ROPETOPIA_MANIFEST.read_text())
        resources = {item["id"]: item for item in manifest["resources"]}
        for specification in builder.SOURCE_DOCUMENTS.values():
            document = json.loads(specification["path"].read_text())
            self.assertEqual(document["source"], "wykd")
            self.assertEqual(
                builder.text_sha256(document["text"]),
                specification["document_sha256"],
            )
            resource = resources[specification["manifest_resource_id"]]
            self.assertEqual(
                resource["canonical_url"],
                specification["original_ropetopia_url"],
            )

    def test_additions_are_extractive_canonical_and_unique(self) -> None:
        self.assertEqual(len(self.rows), 8)
        self.assertEqual(len({row["fact_id"] for row in self.rows}), 8)
        self.assertEqual(len({row["question"] for row in self.rows}), 8)
        for row in self.rows:
            source = json.loads(
                (builder.ROOT / row["source_lineage"][
                    "raw_successor_document"]).read_text()
            )
            self.assertIn(row["evidence"], source["text"])
            self.assertIn(row["answer"], row["evidence"])
            self.assertEqual(
                qa_pair_from_record(row),
                (row["question"], row["answer"]),
            )

    def test_no_active_or_prior_pending_collision(self) -> None:
        comparison = builder.qa_facts(
            (builder.ACTIVE_DATASET, *builder.PRIOR_PENDING),
            "active_or_prior_pending",
        )
        for row in self.rows:
            self.assertIsNone(leakage_reason(
                row["question"], row["answer"], comparison))

    def test_tasuki_merit_edit_is_supported_and_append_safe(self) -> None:
        decisions = builder.read_jsonl(builder.CURATION)
        self.assertEqual(decisions, [builder.CURATION_DECISION])
        decision = decisions[0]
        document = json.loads(builder.EDIT_SOURCE["path"].read_text())
        self.assertIn(decision["evidence"], document["text"])
        self.assertNotIn(decision["answer"], decision["evidence"])
        self.assertEqual(decision["support_type"], "manual_paraphrase")
        self.assertTrue(decision["paraphrase_rationale"])
        self.assertEqual(decision["answer"], "an X on the back")
        self.assertEqual(
            stable_fact_id(decision["question"], decision["answer"]),
            "fact-b97b252cc6fc305c9196",
        )
        active_curation_ids = {
            row["fact_id"]
            for path in builder.ACTIVE_CURATIONS
            for row in builder.read_jsonl(path)
        }
        self.assertNotIn(decision["fact_id"], active_curation_ids)

    def test_report_is_pending_and_respects_sealed_eval_policy(self) -> None:
        report = json.loads(builder.REPORT.read_text())
        self.assertEqual(report["status"], "segregated_pending_not_promoted")
        self.assertFalse(
            report["source_scope"]["recovered_ropetopia_article_bodies_used"])
        policy = report["sealed_evaluation_policy"]
        self.assertFalse(policy["manual_review_opened_eval_or_heldout_content"])
        self.assertFalse(policy["generator_opens_eval_or_heldout_content"])
        self.assertEqual(
            report["isolated_build_projection"]["output_rows"], 812)

    def test_generator_is_byte_deterministic(self) -> None:
        first = tuple(builder.file_sha256(path) for path in (
            builder.ADDITIONS, builder.CURATION, builder.REPORT))
        builder.main()
        second = tuple(builder.file_sha256(path) for path in (
            builder.ADDITIONS, builder.CURATION, builder.REPORT))
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
