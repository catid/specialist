#!/usr/bin/env python3
"""Focused, non-mutating regression tests for context audit v52."""

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
import build_context_merit_audit_v52 as builder
from qa_quality import normalize_text


class ContextMeritAuditV52Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix=".test-v52-", dir=HERE)
        out = Path(cls.temp.name)
        cls.baseline_path = out / "baseline.jsonl"
        cls.baseline_report = out / "baseline.report.json"
        builder.build_projection(cls.baseline_path, cls.baseline_report,
                                 builder.PRIOR_PROJECTION_CURATIONS)
        cls.baseline = builder.read_jsonl(cls.baseline_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def setUp(self) -> None:
        frozen = {p: p.read_bytes() for p in
                  (builder.AUDIT, builder.CURATION, builder.REPORT)}
        self.addCleanup(lambda: [self.assertEqual(p.read_bytes(), d)
                                 for p, d in frozen.items()])
        temp = tempfile.TemporaryDirectory(prefix=".run-v52-", dir=HERE)
        self.addCleanup(temp.cleanup)
        out = Path(temp.name)
        for name, filename in (("AUDIT", "audit.jsonl"),
                               ("CURATION", "curation.jsonl"),
                               ("REPORT", "report.json")):
            patcher = mock.patch.object(builder, name, out / filename)
            patcher.start()
            self.addCleanup(patcher.stop)
        builder.main()
        self.audit = builder.read_jsonl(builder.AUDIT)
        self.curation = builder.read_jsonl(builder.CURATION)

    def test_baseline_is_exact_v51_projection(self) -> None:
        self.assertEqual(len(self.baseline), 546)
        self.assertEqual(builder.file_sha256(self.baseline_path),
                         builder.PROJECTED_SELECTION_BASELINE["sha256"])
        self.assertEqual(json.loads(self.baseline_report.read_text())
                         ["eval_fact_count"], 612)

    def test_selection_is_exact_deterministic_top_ten(self) -> None:
        ranked = builder.ranked_unreviewed_direct(self.baseline)
        self.assertEqual(len(ranked), 46)
        self.assertEqual(tuple(x["row"]["fact_id"] for x in ranked[:10]),
                         builder.EXPECTED_SELECTION)
        self.assertEqual({x["row"]["fact_id"]: x["active_index"]
                          for x in ranked[:10]}, builder.PROJECTED_ACTIVE_INDICES)

    def test_every_selection_is_direct_and_unreviewed(self) -> None:
        by_id = {r["fact_id"]: r for r in self.baseline}
        reviewed = builder.prior_reviewed_fact_ids()
        for fact_id in builder.EXPECTED_SELECTION:
            self.assertFalse(by_id[fact_id].get("curation"))
            self.assertNotIn(fact_id, reviewed)

    def test_projected_baseline_counts_are_pinned(self) -> None:
        direct = [r for r in self.baseline if not r.get("curation")]
        reviewed = builder.prior_reviewed_fact_ids()
        self.assertEqual((len(direct), sum(r["fact_id"] in reviewed for r in direct),
                          sum(r["fact_id"] not in reviewed for r in direct)),
                         (248, 202, 46))

    def test_sources_and_evidence_are_hash_pinned(self) -> None:
        audit = {r["fact_id"]: r for r in self.audit}
        baseline = {r["fact_id"]: r for r in self.baseline}
        validator = builder.previous.previous.previous.previous.source_evidence
        for spec in builder.SPECS:
            row = audit[spec["fact_id"]]
            self.assertEqual(builder.file_sha256(spec["source_path"]),
                             row["source_document_file_sha256"])
            self.assertEqual(builder.text_sha256(row["support_evidence"]),
                             row["support_evidence_sha256"])
            self.assertEqual(validator(spec, baseline[spec["fact_id"]]),
                             (row["support_evidence"], row["source_support"]))
            for fragment in spec.get("paraphrase_support_fragments", ()):
                self.assertIn(normalize_text(fragment),
                              normalize_text(row["support_evidence"]))

    def test_decisions_and_curation_are_exact(self) -> None:
        self.assertEqual(len(self.audit), 10)
        self.assertEqual({d: sum(r["decision"] == d for r in self.audit)
                          for d in ("keep", "drop", "edit")},
                         {"keep": 7, "drop": 3, "edit": 0})
        self.assertEqual(len(self.curation), 3)
        self.assertTrue(all(r["action"] == "drop" for r in self.curation))

    def test_nylon_care_paraphrase_is_documented(self) -> None:
        row = next(r for r in self.audit
                   if r["fact_id"] == "fact-bb9e3bb1450b30eb26fb")
        self.assertEqual(row["source_support"], "manual_paraphrase")
        self.assertTrue(row["paraphrase_rationale"])
        self.assertEqual(row["decision"], "keep")

    def test_projection_lineage_is_complete(self) -> None:
        for row in self.audit:
            lineage = row["projection_lineage"]
            self.assertEqual((lineage["active_index"], lineage["baseline_rows"],
                              lineage["baseline_sha256"]),
                             (row["active_index"], 546,
                              builder.PROJECTED_SELECTION_BASELINE["sha256"]))
            self.assertFalse(lineage["prior_context_merit_review"])

    def test_pending_curation_does_not_overlap_prior(self) -> None:
        prior = {r["fact_id"] for p in (builder.QUALITY_MERIT_CURATION,
                 builder.TASUKI_CURATION, *builder.CONTEXT_CURATIONS)
                 for r in builder.read_jsonl(p)}
        self.assertFalse(prior & {r["fact_id"] for r in self.curation})

    def test_report_counts_projection_and_sealed_policy(self) -> None:
        report = json.loads(builder.REPORT.read_text())
        self.assertEqual(report["schema"], "context-merit-audit-report-v52")
        self.assertEqual(report["audit"]["by_decision"], {"drop": 3, "keep": 7})
        self.assertEqual(report["new_pending_curation"]["edit_support_types"],
                         {"extractive": 0, "manual_paraphrase": 0})
        projection = report["isolated_build_projection"]
        self.assertEqual((projection["output_rows"], projection["output_sha256"]),
                         (543, "21ca3f4e5be03d7dad647ec5a175f67227df064343c6f6a1ba336325ac96637e"))
        self.assertEqual(projection["prior_pending_addition_fact_ids_preserved"], 37)
        self.assertEqual(projection["sealed_eval_fact_count_reported_by_tooling"], 612)
        self.assertFalse(report["sealed_evaluation_policy"]
                               ["manual_review_opened_eval_or_heldout_content"])

    def test_all_prior_decision_artifacts_are_byte_pinned(self) -> None:
        pins = json.loads(builder.REPORT.read_text())[
            "frozen_prior_decision_artifacts"]
        self.assertEqual(len(pins), 153)
        self.assertEqual([r["path"] for r in pins],
                         [str(p.relative_to(builder.ROOT))
                          for p in builder.prior_decision_artifacts()])
        for row in pins:
            self.assertEqual(builder.file_sha256(builder.ROOT / row["path"]),
                             row["sha256"])

    def test_generator_is_deterministic_and_prior_is_frozen(self) -> None:
        outputs = (builder.AUDIT, builder.CURATION, builder.REPORT)
        before = tuple(builder.file_sha256(p) for p in outputs)
        prior = tuple(builder.file_sha256(p)
                      for p in builder.prior_decision_artifacts())
        builder.main()
        self.assertEqual(before, tuple(builder.file_sha256(p) for p in outputs))
        self.assertEqual(prior, tuple(builder.file_sha256(p)
                                      for p in builder.prior_decision_artifacts()))

    def test_active_s5_is_frozen(self) -> None:
        expected = {
            builder.ACTIVE_DATASET: "62e7ae28c86a458d4d33bf3f73f1b91b873c86e3f70ce87706a7394d1f391507",
            builder.ACTIVE_REPORT: "3ca1cbf68dcc281176b25ca14a894edd6db7a3c705849d97f4efcdc6240b974a",
            builder.ROOT / "data/train_qa_curated_v1.curation.jsonl": "ee28db85bea4a74b0114b4312f52129bc8699a2b4b74c02b755f363e8198edeb",
            builder.ROOT / "data/train_qa_kinbakutoday.curation.jsonl": "8b68caf2a6a4411f9ae26a05759c6ceb6852eaf05c28171b4a11dd93033ebadd",
        }
        for path, digest in expected.items():
            self.assertEqual(builder.file_sha256(path), digest)

    def test_future_versions_cannot_enter_v52(self) -> None:
        self.assertEqual((len(builder.CONTEXT_CURATIONS),
                          len(builder.CONTEXT_AUDITS)), (51, 51))
        self.assertIn("context_merit_audit_v51", str(builder.CONTEXT_CURATIONS[-1]))
        self.assertFalse(any("context_merit_audit_v52" in str(p)
                             for p in (*builder.CONTEXT_CURATIONS,
                                       *builder.CONTEXT_AUDITS)))


class IsolatedProjectionV52Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix=".projection-v52-", dir=HERE)
        out = Path(cls.temp.name)
        curations = (*builder.PRIOR_PROJECTION_CURATIONS, builder.CURATION)
        cls.datasets = []
        for run in (1, 2):
            dataset = out / f"projection-{run}.jsonl"
            builder.build_projection(dataset, out / f"report-{run}.json", curations)
            cls.datasets.append(dataset)
        cls.rows = builder.read_jsonl(cls.datasets[0])
        cls.report = json.loads((out / "report-1.json").read_text())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_projection_is_byte_identical_and_hash_pinned(self) -> None:
        self.assertEqual([hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in self.datasets],
                         ["21ca3f4e5be03d7dad647ec5a175f67227df064343c6f6a1ba336325ac96637e"] * 2)
        self.assertEqual((len(self.rows), self.report["eval_fact_count"]), (543, 612))

    def test_projection_applies_exact_decisions_and_upstream_repairs(self) -> None:
        by_id = {r["fact_id"]: r for r in self.rows}
        by_original = {r.get("curation", {}).get("original_fact_id"): r
                       for r in self.rows if r.get("curation")}
        for spec in builder.SPECS:
            if spec["decision"] == "keep":
                self.assertIn(spec["fact_id"], by_id)
            else:
                self.assertNotIn(spec["fact_id"], by_id)
                self.assertNotIn(spec["fact_id"], by_original)
        self.assertNotIn("fact-93c032484cf3a72fcc5c", by_id)
        self.assertIn("distinct facets", by_original[
            "fact-05a050c66a8ee25a8fee"]["answer"])

    def test_37_useful_pending_additions_are_represented(self) -> None:
        expected = {r["fact_id"] for p in builder.PRIOR_PENDING_ADDITIONS
                    for r in builder.read_jsonl(p)}
        represented = {r["fact_id"] for r in self.rows}
        represented.update(r.get("curation", {}).get("original_fact_id")
                           for r in self.rows if r.get("curation"))
        self.assertEqual(len(expected), 38)
        expected.remove("fact-93c032484cf3a72fcc5c")
        self.assertEqual(len(expected), 37)
        self.assertTrue(expected <= represented)

    def test_all_24_owner_urls_are_preserved(self) -> None:
        manifest = json.loads(builder.RESOURCE_MANIFEST.read_text())
        expected = {u for resource in manifest["resources"]
                    for u in (resource["canonical_url"],
                              resource.get("recommendation_url")) if u}
        blob = "\n".join(json.dumps(r, ensure_ascii=False) for r in self.rows)
        self.assertEqual(len(expected), 24)
        self.assertFalse({u for u in expected if u not in blob})

    def test_projection_has_unique_ids_and_questions(self) -> None:
        self.assertEqual(len({r["fact_id"] for r in self.rows}), 543)
        self.assertEqual(len({r["question"] for r in self.rows}), 543)


if __name__ == "__main__":
    unittest.main()
