#!/usr/bin/env python3
"""Focused, non-mutating regression tests for context audit v25."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_context_merit_audit_v25 as builder
from qa_quality import normalize_text


class ContextMeritAuditV25Test(unittest.TestCase):
    def setUp(self) -> None:
        frozen = {path: path.read_bytes()
                  for path in (builder.AUDIT, builder.CURATION, builder.REPORT)}
        self.addCleanup(self._assert_frozen, frozen)
        temporary = tempfile.TemporaryDirectory(
            prefix=".test-context-merit-v25-", dir=HERE)
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name)
        for attribute, filename in (
            ("AUDIT", "context_merit_audit_v25.jsonl"),
            ("CURATION", "pending_curation_context_merit_v25.jsonl"),
            ("REPORT", "report_context_merit_v25.json"),
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

    def test_secondary_selection_is_exact_and_ranked(self) -> None:
        active = builder.read_jsonl(builder.ACTIVE_DATASET)
        ranked = builder.secondary_ranked(active)
        self.assertEqual(len(ranked), 155)
        self.assertEqual(
            tuple(item["row"]["fact_id"] for item in ranked[:25]),
            builder.EXPECTED_SELECTION)
        selected, excluded, provenance = builder.ranked_unreviewed(active)
        self.assertEqual(tuple(item["row"]["fact_id"]
                               for item in selected[:25]),
                         builder.EXPECTED_SELECTION)
        self.assertEqual((excluded, provenance), (0, 0))

    def test_every_selection_is_one_prior_keep_not_rereviewed(self) -> None:
        prior = {}
        for version, path in enumerate(builder.CONTEXT_AUDITS[:20], 1):
            for row in builder.read_jsonl(path):
                prior.setdefault(row["fact_id"], []).append((version, row))
        rereviewed = {
            row["fact_id"] for path in builder.CONTEXT_AUDITS[20:]
            for row in builder.read_jsonl(path)
        }
        self.assertFalse(rereviewed & set(builder.EXPECTED_SELECTION))
        for fact_id in builder.EXPECTED_SELECTION:
            self.assertEqual(len(prior[fact_id]), 1)
            version, row = prior[fact_id][0]
            self.assertEqual(row["decision"], "keep")
            self.assertEqual(version,
                             builder.SECONDARY_PRIOR_VERSIONS[fact_id])

    def test_projected_baseline_and_lineage_are_pinned(self) -> None:
        report = json.loads(builder.REPORT.read_text())
        baseline = report["selection"]["projected_baseline"]
        self.assertEqual(baseline, builder.PROJECTED_SELECTION_BASELINE)
        self.assertEqual(baseline["rows"], 619)
        self.assertEqual(
            baseline["sha256"],
            "267fa20ea6c41521147d83e1086a19a48d04478ddc52f9b442cb1c8cd231c72d")
        by_id = {row["fact_id"]: row for row in self.audit}
        self.assertEqual(
            {fact_id: by_id[fact_id]["active_index"]
             for fact_id in builder.EXPECTED_SELECTION},
            builder.PROJECTED_ACTIVE_INDICES)
        self.assertTrue(all(row["review_pass"] ==
                            "secondary_prior_keep_reaudit"
                            for row in self.audit))
        for fact_id in builder.EXPECTED_SELECTION:
            prior = by_id[fact_id]["prior_review"]
            self.assertEqual(prior["decision"], "keep")
            self.assertEqual(prior["version"],
                             builder.SECONDARY_PRIOR_VERSIONS[fact_id])
            self.assertEqual(builder.file_sha256(builder.ROOT / prior["path"]),
                             prior["sha256"])

    def test_sources_and_evidence_are_hash_pinned(self) -> None:
        by_id = {row["fact_id"]: row for row in self.audit}
        paraphrase = "fact-d7e6a05a62554a6dbe9d"
        for spec in builder.SPECS:
            row = by_id[spec["fact_id"]]
            self.assertEqual(builder.file_sha256(spec["source_path"]),
                             row["source_document_file_sha256"])
            self.assertEqual(builder.text_sha256(row["support_evidence"]),
                             row["support_evidence_sha256"])
            answer = spec.get("answer", row["active_answer"])
            if spec["fact_id"] == paraphrase:
                self.assertEqual(row["source_support"], "source_composite")
            else:
                self.assertIn(normalize_text(answer),
                              normalize_text(row["support_evidence"]))

    def test_decisions_and_curation(self) -> None:
        self.assertEqual(len(self.audit), 25)
        self.assertEqual(
            {decision: sum(row["decision"] == decision for row in self.audit)
             for decision in ("keep", "drop", "edit")},
            {"keep": 6, "drop": 5, "edit": 14})
        self.assertEqual(len(self.curation), 19)
        self.assertEqual(
            {action: sum(row["action"] == action for row in self.curation)
             for action in ("drop", "edit")}, {"drop": 5, "edit": 14})
        edits = [row for row in self.curation if row["action"] == "edit"]
        self.assertEqual(
            sum(row["support_type"] == "extractive" for row in edits), 13)
        self.assertEqual(
            sum(row["support_type"] == "manual_paraphrase" for row in edits),
            1)
        for row in edits:
            if row["support_type"] == "extractive":
                self.assertIn(normalize_text(row["answer"]),
                              normalize_text(row["evidence"]))
            else:
                self.assertTrue(row["paraphrase_rationale"])

    def test_pending_curation_does_not_overlap_prior(self) -> None:
        prior = set()
        for path in (builder.QUALITY_MERIT_CURATION,
                     builder.TASUKI_CURATION, *builder.CONTEXT_CURATIONS):
            prior.update(row["fact_id"] for row in builder.read_jsonl(path))
        self.assertFalse(prior & {row["fact_id"] for row in self.curation})

    def test_report_projection_and_sealed_policy(self) -> None:
        report = json.loads(builder.REPORT.read_text())
        self.assertEqual(report["schema"], "context-merit-audit-report-v25")
        self.assertEqual(report["audit"]["by_decision"],
                         {"drop": 5, "edit": 14, "keep": 6})
        projection = report["isolated_build_projection"]
        self.assertEqual(projection["output_rows"], 614)
        self.assertEqual(
            projection["output_sha256"],
            "fc704352e4bd193c10b81117c54459bb96cb111274536d6d58e6ab3fe4220d8a")
        self.assertEqual(projection["reviewed_keep_fact_ids_preserved"], 6)
        self.assertEqual(projection["prior_pending_addition_fact_ids_preserved"],
                         38)
        self.assertEqual(projection["unexpected_fact_ids"], 0)
        self.assertTrue(projection["repeat_dataset_byte_identical"])
        self.assertFalse(report["sealed_evaluation_policy"]
                               ["manual_review_opened_eval_or_heldout_content"])

    def test_all_prior_decision_artifacts_are_byte_pinned(self) -> None:
        report = json.loads(builder.REPORT.read_text())
        pins = report["frozen_prior_decision_artifacts"]
        self.assertEqual(len(pins), 72)
        self.assertEqual([row["path"] for row in pins], [
            str(path.relative_to(builder.ROOT))
            for path in builder.prior_decision_artifacts()
        ])
        for row in pins:
            self.assertEqual(builder.file_sha256(builder.ROOT / row["path"]),
                             row["sha256"])

    def test_generator_is_deterministic_and_prior_is_frozen(self) -> None:
        outputs = (builder.AUDIT, builder.CURATION, builder.REPORT)
        first = tuple(builder.file_sha256(path) for path in outputs)
        prior_files = builder.prior_decision_artifacts()
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
