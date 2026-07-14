#!/usr/bin/env python3
"""Focused, non-mutating regression tests for context audit v49."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_context_merit_audit_v49 as builder


class ContextMeritAuditV49Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix=".test-v49-", dir=HERE)
        output = Path(cls.temp.name)
        cls.baseline_path = output / "baseline.jsonl"
        cls.baseline_report = output / "baseline.report.json"
        builder.build_projection(cls.baseline_path, cls.baseline_report,
                                 builder.PRIOR_PROJECTION_CURATIONS)
        cls.baseline = builder.read_jsonl(cls.baseline_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def setUp(self) -> None:
        frozen = {path: path.read_bytes()
                  for path in (builder.AUDIT, builder.CURATION, builder.REPORT)}
        self.addCleanup(lambda: [self.assertEqual(path.read_bytes(), data)
                                 for path, data in frozen.items()])
        temporary = tempfile.TemporaryDirectory(prefix=".run-v49-", dir=HERE)
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name)
        for name, filename in (("AUDIT", "audit.jsonl"),
                               ("CURATION", "curation.jsonl"),
                               ("REPORT", "report.json")):
            patcher = mock.patch.object(builder, name, output / filename)
            patcher.start()
            self.addCleanup(patcher.stop)
        builder.main()
        self.audit = builder.read_jsonl(builder.AUDIT)
        self.curation = builder.read_jsonl(builder.CURATION)

    def test_baseline_is_exact_v48_projection(self) -> None:
        self.assertEqual(len(self.baseline), 548)
        self.assertEqual(builder.file_sha256(self.baseline_path),
                         builder.PROJECTED_SELECTION_BASELINE["sha256"])
        self.assertEqual(json.loads(self.baseline_report.read_text())
                         ["eval_fact_count"], 612)

    def test_selection_is_exact_deterministic_top_ten(self) -> None:
        ranked = builder.ranked_unreviewed_direct(self.baseline)
        self.assertEqual(len(ranked), 76)
        self.assertEqual(tuple(item["row"]["fact_id"] for item in ranked[:10]),
                         builder.EXPECTED_SELECTION)
        self.assertEqual({item["row"]["fact_id"]: item["active_index"]
                          for item in ranked[:10]},
                         builder.PROJECTED_ACTIVE_INDICES)

    def test_every_selection_is_direct_and_unreviewed(self) -> None:
        by_id = {row["fact_id"]: row for row in self.baseline}
        reviewed = builder.prior_reviewed_fact_ids()
        for fact_id in builder.EXPECTED_SELECTION:
            self.assertFalse(by_id[fact_id].get("curation"))
            self.assertNotIn(fact_id, reviewed)

    def test_projected_baseline_counts_are_pinned(self) -> None:
        direct = [row for row in self.baseline if not row.get("curation")]
        reviewed = builder.prior_reviewed_fact_ids()
        self.assertEqual((len(direct),
                          sum(row["fact_id"] in reviewed for row in direct),
                          sum(row["fact_id"] not in reviewed for row in direct)),
                         (254, 178, 76))

    def test_sources_and_evidence_are_hash_pinned(self) -> None:
        audit = {row["fact_id"]: row for row in self.audit}
        baseline = {row["fact_id"]: row for row in self.baseline}
        for spec in builder.SPECS:
            row = audit[spec["fact_id"]]
            self.assertEqual(builder.file_sha256(spec["source_path"]),
                             row["source_document_file_sha256"])
            self.assertEqual(builder.text_sha256(row["support_evidence"]),
                             row["support_evidence_sha256"])
            evidence, support = builder.previous.source_evidence(
                spec, baseline[spec["fact_id"]])
            self.assertEqual((evidence, support),
                             (row["support_evidence"], row["source_support"]))

    def test_decisions_and_curation_are_exact(self) -> None:
        self.assertEqual(len(self.audit), 10)
        self.assertEqual({d: sum(row["decision"] == d for row in self.audit)
                          for d in ("keep", "drop", "edit")},
                         {"keep": 8, "drop": 1, "edit": 1})
        self.assertEqual(len(self.curation), 2)
        self.assertEqual({row["action"] for row in self.curation},
                         {"drop", "edit"})

    def test_book_title_edit_is_exact(self) -> None:
        row = next(row for row in self.audit
                   if row["fact_id"] == "fact-da12e05feb5af6c4533b")
        self.assertEqual(row["edited_question"],
                         "Which book by Midori is mentioned as an early English reference?")
        self.assertEqual(row["edited_answer"],
                         "The Seductive Art of Japanese Bondage")
        self.assertEqual(row["source_support"], "normalized_extractive")

    def test_projection_lineage_is_complete(self) -> None:
        for row in self.audit:
            lineage = row["projection_lineage"]
            self.assertEqual((lineage["active_index"], lineage["baseline_rows"],
                              lineage["baseline_sha256"]),
                             (row["active_index"], 548,
                              builder.PROJECTED_SELECTION_BASELINE["sha256"]))
            self.assertFalse(lineage["prior_context_merit_review"])

    def test_pending_curation_does_not_overlap_prior(self) -> None:
        prior = {row["fact_id"] for path in (
            builder.QUALITY_MERIT_CURATION, builder.TASUKI_CURATION,
            *builder.CONTEXT_CURATIONS) for row in builder.read_jsonl(path)}
        self.assertFalse(prior & {row["fact_id"] for row in self.curation})

    def test_report_counts_projection_and_sealed_policy(self) -> None:
        report = json.loads(builder.REPORT.read_text())
        self.assertEqual(report["schema"], "context-merit-audit-report-v49")
        self.assertEqual(report["audit"]["by_decision"],
                         {"drop": 1, "edit": 1, "keep": 8})
        self.assertEqual(report["new_pending_curation"]["edit_support_types"],
                         {"extractive": 1, "manual_paraphrase": 0})
        projection = report["isolated_build_projection"]
        self.assertEqual((projection["output_rows"], projection["output_sha256"]),
                         (547, "e92dc20eec64faf1c49d2660520ec972261411a62eb4053de86cf4d67f31da2c"))
        self.assertEqual(projection["prior_pending_addition_fact_ids_preserved"], 37)
        self.assertEqual(projection["sealed_eval_fact_count_reported_by_tooling"], 612)
        self.assertFalse(report["sealed_evaluation_policy"]
                               ["manual_review_opened_eval_or_heldout_content"])

    def test_all_prior_decision_artifacts_are_byte_pinned(self) -> None:
        pins = json.loads(builder.REPORT.read_text())[
            "frozen_prior_decision_artifacts"]
        self.assertEqual(len(pins), 144)
        self.assertEqual([row["path"] for row in pins], [
            str(path.relative_to(builder.ROOT))
            for path in builder.prior_decision_artifacts()])
        for row in pins:
            self.assertEqual(builder.file_sha256(builder.ROOT / row["path"]),
                             row["sha256"])

    def test_generator_is_deterministic_and_prior_is_frozen(self) -> None:
        outputs = (builder.AUDIT, builder.CURATION, builder.REPORT)
        before = tuple(builder.file_sha256(path) for path in outputs)
        prior = tuple(builder.file_sha256(path)
                      for path in builder.prior_decision_artifacts())
        builder.main()
        self.assertEqual(before, tuple(builder.file_sha256(path)
                                       for path in outputs))
        self.assertEqual(prior, tuple(builder.file_sha256(path)
                                      for path in builder.prior_decision_artifacts()))

    def test_active_s5_is_frozen(self) -> None:
        expected = {
            builder.ACTIVE_DATASET: "62e7ae28c86a458d4d33bf3f73f1b91b873c86e3f70ce87706a7394d1f391507",
            builder.ACTIVE_REPORT: "3ca1cbf68dcc281176b25ca14a894edd6db7a3c705849d97f4efcdc6240b974a",
            builder.ROOT / "data/train_qa_curated_v1.curation.jsonl": "ee28db85bea4a74b0114b4312f52129bc8699a2b4b74c02b755f363e8198edeb",
            builder.ROOT / "data/train_qa_kinbakutoday.curation.jsonl": "8b68caf2a6a4411f9ae26a05759c6ceb6852eaf05c28171b4a11dd93033ebadd",
        }
        for path, digest in expected.items():
            self.assertEqual(builder.file_sha256(path), digest)

    def test_future_versions_cannot_enter_v49(self) -> None:
        self.assertEqual((len(builder.CONTEXT_CURATIONS),
                          len(builder.CONTEXT_AUDITS)), (48, 48))
        self.assertIn("context_merit_audit_v48",
                      str(builder.CONTEXT_CURATIONS[-1]))
        self.assertFalse(any("context_merit_audit_v49" in str(path)
                             for path in (*builder.CONTEXT_CURATIONS,
                                          *builder.CONTEXT_AUDITS)))


class IsolatedProjectionV49Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix=".projection-v49-", dir=HERE)
        output = Path(cls.temp.name)
        curations = (*builder.PRIOR_PROJECTION_CURATIONS, builder.CURATION)
        cls.datasets = []
        for run in (1, 2):
            dataset = output / f"projection-{run}.jsonl"
            builder.build_projection(dataset, output / f"report-{run}.json",
                                     curations)
            cls.datasets.append(dataset)
        cls.rows = builder.read_jsonl(cls.datasets[0])
        cls.report = json.loads((output / "report-1.json").read_text())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_projection_is_byte_identical_and_hash_pinned(self) -> None:
        self.assertEqual([hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in self.datasets],
                         ["e92dc20eec64faf1c49d2660520ec972261411a62eb4053de86cf4d67f31da2c"] * 2)
        self.assertEqual((len(self.rows), self.report["eval_fact_count"]),
                         (547, 612))

    def test_projection_applies_exact_decisions_and_upstream_repairs(self) -> None:
        by_id = {row["fact_id"]: row for row in self.rows}
        by_original = {row.get("curation", {}).get("original_fact_id"): row
                       for row in self.rows if row.get("curation")}
        for spec in builder.SPECS:
            if spec["decision"] == "keep":
                self.assertIn(spec["fact_id"], by_id)
            elif spec["decision"] == "drop":
                self.assertNotIn(spec["fact_id"], by_id)
                self.assertNotIn(spec["fact_id"], by_original)
            else:
                self.assertEqual(by_original[spec["fact_id"]]["answer"],
                                 spec["answer"])
                self.assertEqual(by_id[spec["fact_id"]]["curation"]
                                 ["original_fact_id"], spec["fact_id"])
        self.assertNotIn("fact-93c032484cf3a72fcc5c", by_id)
        self.assertIn("distinct facets", by_original[
            "fact-05a050c66a8ee25a8fee"]["answer"])

    def test_37_useful_pending_additions_are_represented(self) -> None:
        expected = {row["fact_id"] for path in builder.PRIOR_PENDING_ADDITIONS
                    for row in builder.read_jsonl(path)}
        represented = {row["fact_id"] for row in self.rows}
        represented.update(row.get("curation", {}).get("original_fact_id")
                           for row in self.rows if row.get("curation"))
        self.assertEqual(len(expected), 38)
        expected.remove("fact-93c032484cf3a72fcc5c")
        self.assertTrue(expected <= represented)

    def test_all_24_owner_urls_are_preserved(self) -> None:
        manifest = json.loads(builder.RESOURCE_MANIFEST.read_text())
        expected = {url for resource in manifest["resources"]
                    for url in (resource["canonical_url"],
                                resource.get("recommendation_url")) if url}
        blob = "\n".join(json.dumps(row, ensure_ascii=False)
                         for row in self.rows)
        self.assertEqual(len(expected), 24)
        self.assertFalse({url for url in expected if url not in blob})

    def test_projection_has_unique_ids_and_questions(self) -> None:
        self.assertEqual(len({row["fact_id"] for row in self.rows}), 547)
        self.assertEqual(len({row["question"] for row in self.rows}), 547)


if __name__ == "__main__":
    unittest.main()
