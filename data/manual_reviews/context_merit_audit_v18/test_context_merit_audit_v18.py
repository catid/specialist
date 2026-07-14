#!/usr/bin/env python3
"""Focused, non-mutating regression tests for context audit v18."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_context_merit_audit_v18 as builder
from qa_quality import normalize_text


class ContextMeritAuditV18Test(unittest.TestCase):
    def setUp(self) -> None:
        frozen = {path: path.read_bytes()
                  for path in (builder.AUDIT, builder.CURATION, builder.REPORT)}
        self.addCleanup(self._assert_frozen, frozen)
        temporary = tempfile.TemporaryDirectory(
            prefix=".test-context-merit-v18-", dir=HERE)
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name)
        for attribute, filename in (
            ("AUDIT", "context_merit_audit_v18.jsonl"),
            ("CURATION", "pending_curation_context_merit_v18.jsonl"),
            ("REPORT", "report_context_merit_v18.json"),
        ):
            patcher = mock.patch.object(builder, attribute, output / filename)
            patcher.start()
            self.addCleanup(patcher.stop)
        builder.main()
        self.audit = builder.read_jsonl(builder.AUDIT)
        self.curation = builder.read_jsonl(builder.CURATION)

    def _assert_frozen(self, frozen: dict[Path, bytes]) -> None:
        for path, expected in frozen.items():
            self.assertEqual(path.read_bytes(), expected,
                             f"test mutated frozen artifact: {path}")

    def test_selection_is_exact_and_nonoverlapping(self) -> None:
        ranked, _, _ = builder.ranked_unreviewed(
            builder.read_jsonl(builder.ACTIVE_DATASET))
        self.assertEqual(tuple(item["row"]["fact_id"] for item in ranked[:25]),
                         builder.EXPECTED_SELECTION)
        prior = set()
        for version in range(1, 18):
            path = (builder.DATA / "manual_reviews" /
                    f"context_merit_audit_v{version}" /
                    f"context_merit_audit_v{version}.jsonl")
            prior.update(row["fact_id"] for row in builder.read_jsonl(path))
        self.assertFalse(prior & set(builder.EXPECTED_SELECTION))

    def test_sources_and_evidence_are_hash_pinned(self) -> None:
        by_id = {row["fact_id"]: row for row in self.audit}
        for spec in builder.SPECS:
            row = by_id[spec["fact_id"]]
            self.assertEqual(builder.file_sha256(spec["source_path"]),
                             row["source_document_file_sha256"])
            self.assertEqual(builder.text_sha256(row["support_evidence"]),
                             row["support_evidence_sha256"])
            answer = spec.get("answer", row["active_answer"])
            self.assertIn(normalize_text(answer),
                          normalize_text(row["support_evidence"]))

    def test_decisions_and_curation(self) -> None:
        self.assertEqual(len(self.audit), 25)
        self.assertEqual(
            {decision: sum(row["decision"] == decision for row in self.audit)
             for decision in ("keep", "drop", "edit")},
            {"keep": 12, "drop": 7, "edit": 6})
        self.assertEqual(len(self.curation), 13)
        self.assertEqual(
            {action: sum(row["action"] == action for row in self.curation)
             for action in ("drop", "edit")}, {"drop": 7, "edit": 6})
        for row in self.curation:
            if row["action"] == "edit":
                self.assertEqual(row["support_type"], "extractive")
                self.assertIn(normalize_text(row["answer"]),
                              normalize_text(row["evidence"]))

    def test_pending_curation_does_not_overlap_prior(self) -> None:
        prior = set()
        for path in (builder.QUALITY_MERIT_CURATION,
                     builder.TASUKI_CURATION, *builder.CONTEXT_CURATIONS):
            prior.update(row["fact_id"] for row in builder.read_jsonl(path))
        self.assertFalse(prior & {row["fact_id"] for row in self.curation})

    def test_report_projection_and_sealed_policy(self) -> None:
        report = json.loads(builder.REPORT.read_text())
        self.assertEqual(report["schema"], "context-merit-audit-report-v18")
        self.assertEqual(report["audit"]["by_decision"],
                         {"drop": 7, "edit": 6, "keep": 12})
        projection = report["isolated_build_projection"]
        self.assertEqual(projection["output_rows"], 668)
        self.assertEqual(
            projection["output_sha256"],
            "02d7dbbb860f875d9ee4a3f8d94e2dcc43eb09e97548fd7029da2e0ebaf3026a")
        self.assertEqual(projection["reviewed_keep_fact_ids_preserved"], 12)
        self.assertEqual(projection["unexpected_fact_ids"], 0)
        self.assertTrue(projection["repeat_dataset_byte_identical"])
        self.assertFalse(report["sealed_evaluation_policy"]
                               ["manual_review_opened_eval_or_heldout_content"])

    def test_generator_is_deterministic_and_prior_is_frozen(self) -> None:
        outputs = (builder.AUDIT, builder.CURATION, builder.REPORT)
        first = tuple(builder.file_sha256(path) for path in outputs)
        prior_files = tuple(
            path for version in range(1, 18)
            for path in sorted((builder.DATA / "manual_reviews" /
                                f"context_merit_audit_v{version}").glob("*"))
            if path.is_file())
        before = tuple(builder.file_sha256(path) for path in prior_files)
        builder.main()
        self.assertEqual(first,
                         tuple(builder.file_sha256(path) for path in outputs))
        self.assertEqual(before,
                         tuple(builder.file_sha256(path) for path in prior_files))

    def test_active_s5_is_frozen(self) -> None:
        self.assertEqual(
            builder.file_sha256(builder.ACTIVE_DATASET),
            "62e7ae28c86a458d4d33bf3f73f1b91b873c86e3f70ce87706a7394d1f391507")
        self.assertEqual(
            builder.file_sha256(builder.ACTIVE_REPORT),
            "3ca1cbf68dcc281176b25ca14a894edd6db7a3c705849d97f4efcdc6240b974a")


if __name__ == "__main__":
    unittest.main()
