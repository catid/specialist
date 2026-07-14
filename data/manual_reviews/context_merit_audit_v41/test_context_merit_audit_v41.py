#!/usr/bin/env python3
"""Focused, non-mutating regression tests for context audit v41."""

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
import build_context_merit_audit_v41 as builder
from qa_quality import normalize_text


class ContextMeritAuditV41Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.baseline_temp = tempfile.TemporaryDirectory(
            prefix=".test-baseline-v41-", dir=HERE)
        output = Path(cls.baseline_temp.name)
        cls.baseline_path = output / "projection-v40.jsonl"
        cls.baseline_report_path = output / "projection-v40.report.json"
        builder.build_projection(cls.baseline_path, cls.baseline_report_path,
                                 builder.PRIOR_PROJECTION_CURATIONS)
        cls.baseline = builder.read_jsonl(cls.baseline_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.baseline_temp.cleanup()

    def setUp(self) -> None:
        frozen = {path: path.read_bytes()
                  for path in (builder.AUDIT, builder.CURATION, builder.REPORT)}
        self.addCleanup(self._assert_frozen, frozen)
        temporary = tempfile.TemporaryDirectory(
            prefix=".test-context-merit-v41-", dir=HERE)
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name)
        for attribute, filename in (
            ("AUDIT", "context_merit_audit_v41.jsonl"),
            ("CURATION", "pending_curation_context_merit_v41.jsonl"),
            ("REPORT", "report_context_merit_v41.json"),
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

    def test_baseline_is_exact_v40_projection(self) -> None:
        self.assertEqual(len(self.baseline), 566)
        self.assertEqual(builder.file_sha256(self.baseline_path),
                         builder.PROJECTED_SELECTION_BASELINE["sha256"])
        report = json.loads(self.baseline_report_path.read_text())
        self.assertEqual(report["eval_fact_count"], 612)

    def test_selection_is_exact_deterministic_top_ten(self) -> None:
        ranked = builder.ranked_unreviewed_direct(self.baseline)
        self.assertEqual(len(ranked), 156)
        self.assertEqual(
            tuple(item["row"]["fact_id"] for item in ranked[:10]),
            builder.EXPECTED_SELECTION)
        self.assertEqual(
            {item["row"]["fact_id"]: item["active_index"]
             for item in ranked[:10]}, builder.PROJECTED_ACTIVE_INDICES)

    def test_every_selection_is_direct_and_previously_unreviewed(self) -> None:
        by_id = {row["fact_id"]: row for row in self.baseline}
        reviewed = builder.prior_reviewed_fact_ids()
        for fact_id in builder.EXPECTED_SELECTION:
            self.assertNotIn("curation", by_id[fact_id])
            self.assertNotIn(fact_id, reviewed)

    def test_projected_baseline_counts_are_pinned(self) -> None:
        direct = [row for row in self.baseline if not row.get("curation")]
        reviewed = builder.prior_reviewed_fact_ids()
        self.assertEqual(len(direct), 284)
        self.assertEqual(sum(row["fact_id"] in reviewed for row in direct), 128)
        self.assertEqual(sum(row["fact_id"] not in reviewed for row in direct),
                         156)
        report = json.loads(builder.REPORT.read_text())
        self.assertEqual(report["selection"]["projected_baseline"],
                         builder.PROJECTED_SELECTION_BASELINE)

    def test_sources_and_evidence_are_hash_pinned(self) -> None:
        by_id = {row["fact_id"]: row for row in self.audit}
        baseline = {row["fact_id"]: row for row in self.baseline}
        for spec in builder.SPECS:
            row = by_id[spec["fact_id"]]
            self.assertEqual(builder.file_sha256(spec["source_path"]),
                             row["source_document_file_sha256"])
            self.assertEqual(builder.text_sha256(row["support_evidence"]),
                             row["support_evidence_sha256"])
            answer = spec.get("answer", row["active_answer"])
            expected_support = spec.get(
                "support_type", "normalized_extractive")
            if expected_support == "manual_paraphrase":
                self.assertTrue(spec["paraphrase_rationale"])
                for fragment in spec["paraphrase_support_fragments"]:
                    self.assertIn(normalize_text(fragment),
                                  normalize_text(row["support_evidence"]))
            else:
                self.assertIn(normalize_text(answer),
                              normalize_text(row["support_evidence"]))
            evidence, support = builder.source_evidence(
                spec, baseline[spec["fact_id"]])
            self.assertEqual(evidence, row["support_evidence"])
            self.assertEqual(support, expected_support)

    def test_decisions_and_curation_are_exact(self) -> None:
        self.assertEqual(len(self.audit), 10)
        self.assertEqual(
            {decision: sum(row["decision"] == decision for row in self.audit)
             for decision in ("keep", "drop", "edit")},
            {"keep": 9, "drop": 0, "edit": 1})
        self.assertEqual(len(self.curation), 1)
        self.assertEqual(
            {action: sum(row["action"] == action for row in self.curation)
            for action in ("drop", "edit")}, {"drop": 0, "edit": 1})

    def test_edit_is_a_documented_manual_paraphrase(self) -> None:
        edits = [row for row in self.curation if row["action"] == "edit"]
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0]["support_type"], "manual_paraphrase")
        self.assertEqual(
            edits[0]["answer"],
            "single- and double-column ties, basic frictions, and rope handling")
        self.assertTrue(edits[0]["paraphrase_rationale"])

    def test_projection_lineage_is_complete(self) -> None:
        for row in self.audit:
            lineage = row["projection_lineage"]
            self.assertEqual(lineage["active_index"], row["active_index"])
            self.assertEqual(lineage["baseline_rows"], 566)
            self.assertEqual(lineage["baseline_sha256"],
                             builder.PROJECTED_SELECTION_BASELINE["sha256"])
            self.assertFalse(lineage["prior_context_merit_review"])
            self.assertEqual(row["review_pass"],
                             "first_context_merit_review_of_v40_projection_row")

    def test_pending_curation_does_not_overlap_prior(self) -> None:
        prior = set()
        for path in (builder.QUALITY_MERIT_CURATION,
                     builder.TASUKI_CURATION, *builder.CONTEXT_CURATIONS):
            prior.update(row["fact_id"] for row in builder.read_jsonl(path))
        self.assertFalse(prior & {row["fact_id"] for row in self.curation})

    def test_report_counts_projection_and_sealed_policy(self) -> None:
        report = json.loads(builder.REPORT.read_text())
        self.assertEqual(report["schema"], "context-merit-audit-report-v41")
        self.assertEqual(report["audit"]["rows"], 10)
        self.assertEqual(report["selection"]["rows_selected"], 10)
        self.assertEqual(report["selection"]["eligible_unreviewed_rows"], 156)
        self.assertEqual(report["audit"]["by_decision"],
                         {"edit": 1, "keep": 9})
        projection = report["isolated_build_projection"]
        self.assertEqual(projection["output_rows"], 566)
        self.assertEqual(
            projection["output_sha256"],
            "3508923a183b639b9a3546f3edb8d972e11189e522b7686e62cd4ec134e46bc2")
        self.assertEqual(projection["reviewed_keep_fact_ids_preserved"], 9)
        self.assertEqual(projection["prior_pending_addition_fact_ids_preserved"],
                         38)
        self.assertEqual(projection["sealed_eval_fact_count_reported_by_tooling"],
                         612)
        self.assertEqual(projection["unexpected_fact_ids"], 0)
        self.assertEqual(projection["validated_runs"], 2)
        self.assertTrue(projection["repeat_dataset_byte_identical"])
        self.assertFalse(report["sealed_evaluation_policy"]
                               ["manual_review_opened_eval_or_heldout_content"])

    def test_all_prior_decision_artifacts_are_byte_pinned(self) -> None:
        report = json.loads(builder.REPORT.read_text())
        pins = report["frozen_prior_decision_artifacts"]
        self.assertEqual(len(pins), 120)
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
        expected = {
            builder.ACTIVE_DATASET:
                "62e7ae28c86a458d4d33bf3f73f1b91b873c86e3f70ce87706a7394d1f391507",
            builder.ACTIVE_REPORT:
                "3ca1cbf68dcc281176b25ca14a894edd6db7a3c705849d97f4efcdc6240b974a",
            builder.ROOT / "data/train_qa_curated_v1.curation.jsonl":
                "ee28db85bea4a74b0114b4312f52129bc8699a2b4b74c02b755f363e8198edeb",
            builder.ROOT / "data/train_qa_kinbakutoday.curation.jsonl":
                "8b68caf2a6a4411f9ae26a05759c6ceb6852eaf05c28171b4a11dd93033ebadd",
        }
        for path, digest in expected.items():
            self.assertEqual(builder.file_sha256(path), digest)

    def test_future_versions_cannot_enter_v41_by_directory_discovery(self) -> None:
        self.assertEqual(len(builder.CONTEXT_CURATIONS), 40)
        self.assertEqual(len(builder.CONTEXT_AUDITS), 40)
        self.assertIn("context_merit_audit_v40",
                      str(builder.CONTEXT_CURATIONS[-1]))
        self.assertIn("context_merit_audit_v40",
                      str(builder.CONTEXT_AUDITS[-1]))
        self.assertFalse(any("context_merit_audit_v41" in str(path) or
                             "context_merit_audit_v41" in str(path)
                             for path in (*builder.CONTEXT_CURATIONS,
                                          *builder.CONTEXT_AUDITS)))


class IsolatedProjectionV41Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(
            prefix=".test-projection-v41-", dir=HERE)
        output = Path(cls.temporary.name)
        cls.datasets = []
        cls.reports = []
        curations = (*builder.PRIOR_PROJECTION_CURATIONS, builder.CURATION)
        for run in (1, 2):
            dataset = output / f"projection-{run}.jsonl"
            report = output / f"projection-{run}.report.json"
            builder.build_projection(dataset, report, curations)
            cls.datasets.append(dataset)
            cls.reports.append(report)
        cls.rows = builder.read_jsonl(cls.datasets[0])
        cls.report = json.loads(cls.reports[0].read_text())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_projection_is_byte_identical_and_hash_pinned(self) -> None:
        digests = [hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in self.datasets]
        self.assertEqual(digests, [
            "3508923a183b639b9a3546f3edb8d972e11189e522b7686e62cd4ec134e46bc2",
            "3508923a183b639b9a3546f3edb8d972e11189e522b7686e62cd4ec134e46bc2",
        ])
        self.assertEqual(len(self.rows), 566)
        self.assertEqual(self.report["eval_fact_count"], 612)

    def test_projection_applies_exact_v41_decisions(self) -> None:
        by_id = {row["fact_id"]: row for row in self.rows}
        by_original = {
            row.get("curation", {}).get("original_fact_id"): row
            for row in self.rows if row.get("curation", {}).get("original_fact_id")
        }
        for spec in builder.SPECS:
            fact_id = spec["fact_id"]
            if spec["decision"] == "keep":
                self.assertIn(fact_id, by_id)
            elif spec["decision"] == "drop":
                self.assertNotIn(fact_id, by_id)
                self.assertNotIn(fact_id, by_original)
            else:
                self.assertIn(fact_id, by_original)
                edited = by_original[fact_id]
                self.assertEqual(edited["question"], spec["question"])
                self.assertEqual(edited["answer"], spec["answer"])
        repaired_v37 = by_original["fact-920250f8d4d0e8f09be1"]
        self.assertEqual(
            repaired_v37["answer"],
            "Start with a very long rope, tie the same favorite tie, shorten "
            "and retie it repeatedly until it feels too short, and compare "
            "different diameters.")
        self.assertNotIn("does the rope feels", repaired_v37["answer"])
        repaired_culture = by_original["fact-b1976a3e15c9d8fae3fc"]
        self.assertTrue(repaired_culture["answer"].startswith("Because shibari"))
        self.assertNotIn("shibari, at the end of the day, reflecting",
                         repaired_culture["answer"])
        repaired_preference = by_original["fact-c1697f4f94aa32b0b9f6"]
        self.assertEqual(
            repaired_preference["answer"],
            "Try different ropes and compare their properties to discover "
            "your own preferences.")
        repaired_reproducibility = by_original["fact-ac7fbf74e38372bfbbbd"]
        self.assertNotIn("knit", repaired_reproducibility["answer"].casefold())
        prior_v38 = {
            row["fact_id"]: row
            for row in builder.read_jsonl(builder.CONTEXT_AUDITS[-3])
        }
        venue = prior_v38["fact-55da0d1915d37b1d077f"]
        self.assertIn("Barajūjikan (Rosencreutz)", venue["support_evidence"])
        self.assertIn("first SM bar in western Japan", venue["support_evidence"])
        repaired_knots = by_original["fact-c13d971e01ce85fc8b11"]
        self.assertEqual(
            repaired_knots["answer"],
            "Learn and practice a few knots, then consider which tying "
            "contexts make each knot useful.")
        self.assertNotIn("analyze in which context", repaired_knots["answer"])
        repaired_history = by_original["fact-ef2cab8c383642b09db1"]
        self.assertNotIn("rather than as", repaired_history["question"])
        self.assertIn("rather than a how-to guide", repaired_history["question"])
        self.assertNotIn("should not be reproduced", repaired_history["answer"])
        self.assertEqual(
            repaired_history["answer"],
            "The page treats the text as a historical artifact showing how "
            "rope, fantasy, and self-transformation were imagined in "
            "mid-century Japanese fetish writing, rather than as a how-to "
            "guide.")
        self.assertNotIn("guide to reproduce", repaired_history["answer"])
        repaired_slipped_hitch = by_original["fact-34c77c437152b75d6d55"]
        self.assertIn("unlocked entry and locked exit",
                      repaired_slipped_hitch["question"])
        self.assertIn("entry point will not be locked",
                      repaired_slipped_hitch["answer"])
        self.assertIn("lock the exit into place",
                      repaired_slipped_hitch["answer"])
        repaired_beginner_scope = by_original["fact-82258f3d54c1714346bd"]
        self.assertEqual(
            repaired_beginner_scope["answer"],
            "single- and double-column ties, basic frictions, and rope handling")
        self.assertNotIn("ties to basic", repaired_beginner_scope["answer"])

    def test_all_38_pending_additions_are_preserved(self) -> None:
        expected = set()
        for path in builder.PRIOR_PENDING_ADDITIONS:
            expected.update(row["fact_id"] for row in builder.read_jsonl(path))
        self.assertEqual(len(expected), 38)
        self.assertTrue(expected <= {row["fact_id"] for row in self.rows})

    def test_all_24_owner_urls_are_preserved(self) -> None:
        manifest = json.loads(builder.RESOURCE_MANIFEST.read_text())
        expected = set()
        for resource in manifest["resources"]:
            expected.add(resource["canonical_url"])
            if resource.get("recommendation_url"):
                expected.add(resource["recommendation_url"])
        self.assertEqual(len(expected), 24)
        blob = "\n".join(json.dumps(row, ensure_ascii=False)
                          for row in self.rows)
        self.assertFalse({url for url in expected if url not in blob})

    def test_projection_has_unique_ids_and_questions(self) -> None:
        self.assertEqual(len({row["fact_id"] for row in self.rows}), 566)
        self.assertEqual(len({row["question"] for row in self.rows}), 566)


if __name__ == "__main__":
    unittest.main()
