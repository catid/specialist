#!/usr/bin/env python3
"""Focused regression tests for context-merit audit tranche v4."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_context_merit_audit_v4 as builder
from qa_quality import normalize_text


class ContextMeritAuditV4Test(unittest.TestCase):
    def setUp(self) -> None:
        frozen_paths = (builder.AUDIT, builder.CURATION, builder.REPORT)
        frozen_bytes = {path: path.read_bytes() for path in frozen_paths}
        self.addCleanup(self._assert_frozen_outputs_unchanged, frozen_bytes)

        temporary = tempfile.TemporaryDirectory(
            prefix=".test-context-merit-v4-", dir=HERE)
        self.addCleanup(temporary.cleanup)
        output_dir = Path(temporary.name)
        for attribute, filename in (
            ("AUDIT", "context_merit_audit_v4.jsonl"),
            ("CURATION", "pending_curation_context_merit_v4.jsonl"),
            ("REPORT", "report_context_merit_v4.json"),
        ):
            patcher = mock.patch.object(
                builder, attribute, output_dir / filename)
            patcher.start()
            self.addCleanup(patcher.stop)

        builder.main()
        self.audit = builder.read_jsonl(builder.AUDIT)
        self.curation = builder.read_jsonl(builder.CURATION)

    def _assert_frozen_outputs_unchanged(
            self, frozen_bytes: dict[Path, bytes]) -> None:
        for path, expected in frozen_bytes.items():
            self.assertEqual(path.read_bytes(), expected,
                             f"test mutated frozen artifact: {path}")

    def test_future_context_tranches_are_not_historical_inputs(self) -> None:
        manual_root = builder.DATA / "manual_reviews"
        prior = manual_root / "context_merit_audit_v3" / "synthetic.jsonl"
        future = manual_root / "context_merit_audit_v999" / "synthetic.jsonl"
        reader = mock.Mock(return_value=[{"fact_id": "fact-synthetic-prior"}])
        with mock.patch.object(Path, "rglob", return_value=[prior, future]), \
                mock.patch.object(builder, "read_jsonl", reader):
            reviewed = builder.reviewed_fact_ids()
        self.assertEqual(
            builder.PRIOR_CONTEXT_MERIT_DIRS,
            frozenset(f"context_merit_audit_v{i}" for i in range(1, 4)),
        )
        self.assertEqual(reviewed, {"fact-synthetic-prior"})
        self.assertEqual(reader.call_args_list, [mock.call(prior)])

    def test_selection_is_next_nonoverlapping_ranked_25(self) -> None:
        active = builder.read_jsonl(builder.ACTIVE_DATASET)
        ranked, _, _ = builder.ranked_unreviewed(active)
        self.assertEqual(
            tuple(item["row"]["fact_id"] for item in ranked[:25]),
            builder.EXPECTED_SELECTION,
        )
        prior_ids = set()
        for version in (1, 2, 3):
            path = (builder.DATA / "manual_reviews" /
                    f"context_merit_audit_v{version}" /
                    f"context_merit_audit_v{version}.jsonl")
            prior_ids.update(row["fact_id"]
                             for row in builder.read_jsonl(path))
        self.assertFalse(prior_ids & set(builder.EXPECTED_SELECTION))

    def test_prior_review_and_active_provenance_are_excluded(self) -> None:
        reviewed = builder.reviewed_fact_ids()
        active = builder.read_jsonl(builder.ACTIVE_DATASET)
        ranked, _, _ = builder.ranked_unreviewed(active)
        for item in ranked[:25]:
            row = item["row"]
            self.assertNotIn(row["fact_id"], reviewed)
            self.assertFalse(any(
                field in row for field in builder.common.ACTIVE_REVIEW_FIELDS))

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
            answer = specification.get("answer", row["active_answer"])
            self.assertIn(normalize_text(answer),
                          normalize_text(row["support_evidence"]))

    def test_manual_decisions_cover_all_25(self) -> None:
        self.assertEqual(len(self.audit), 25)
        self.assertEqual(
            {decision: sum(row["decision"] == decision for row in self.audit)
             for decision in {"keep", "drop", "edit"}},
            {"keep": 15, "drop": 8, "edit": 2},
        )
        self.assertEqual(tuple(row["fact_id"] for row in self.audit),
                         builder.EXPECTED_SELECTION)

    def test_pending_curation_support_and_append_safety(self) -> None:
        self.assertEqual(len(self.curation), 10)
        self.assertEqual(
            {action: sum(row["action"] == action for row in self.curation)
             for action in {"drop", "edit"}},
            {"drop": 8, "edit": 2},
        )
        prior = set()
        for path in (builder.QUALITY_MERIT_CURATION,
                     builder.TASUKI_CURATION, *builder.CONTEXT_CURATIONS):
            prior.update(row["fact_id"] for row in builder.read_jsonl(path))
        self.assertFalse(prior & {row["fact_id"] for row in self.curation})
        edits = [row for row in self.curation if row["action"] == "edit"]
        self.assertEqual([row["support_type"] for row in edits],
                         ["extractive", "extractive"])
        for row in edits:
            self.assertIn(normalize_text(row["answer"]),
                          normalize_text(row["evidence"]))

    def test_report_is_pending_and_projection_is_identity_checked(self) -> None:
        report = json.loads(builder.REPORT.read_text())
        self.assertEqual(report["status"], "segregated_pending_not_promoted")
        self.assertEqual(report["audit"]["by_decision"],
                         {"drop": 8, "edit": 2, "keep": 15})
        projection = report["isolated_build_projection"]
        self.assertEqual(projection["output_rows"], 778)
        self.assertEqual(projection["reviewed_keep_fact_ids_preserved"], 15)
        self.assertEqual(projection["prior_pending_addition_fact_ids_preserved"],
                         38)
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
