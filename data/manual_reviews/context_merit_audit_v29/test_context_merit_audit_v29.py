#!/usr/bin/env python3
"""Focused, non-mutating regression tests for context audit v29."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_context_merit_audit_v29 as builder
from qa_quality import normalize_text


class ContextMeritAuditV29Test(unittest.TestCase):
    def setUp(self) -> None:
        frozen = {path: path.read_bytes()
                  for path in (builder.AUDIT, builder.CURATION, builder.REPORT)}
        self.addCleanup(self._assert_frozen, frozen)
        temporary = tempfile.TemporaryDirectory(
            prefix=".test-context-merit-v29-", dir=HERE)
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name)
        for attribute, filename in (
            ("AUDIT", "context_merit_audit_v29.jsonl"),
            ("CURATION", "pending_curation_context_merit_v29.jsonl"),
            ("REPORT", "report_context_merit_v29.json"),
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
        self.assertEqual(len(ranked), 55)
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
        self.assertEqual(baseline["rows"], 591)
        self.assertEqual(
            baseline["sha256"],
            "790549d4a1a9f65c7538ea50e6eb6f329b5bd6ae429cd4cac12cf38bee8e2b6e")
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
        paraphrase = "fact-452f36d13d2af7da6837"
        for spec in builder.SPECS:
            row = by_id[spec["fact_id"]]
            self.assertEqual(builder.file_sha256(spec["source_path"]),
                             row["source_document_file_sha256"])
            self.assertEqual(builder.text_sha256(row["support_evidence"]),
                             row["support_evidence_sha256"])
            answer = spec.get("answer", row["active_answer"])
            if spec["fact_id"] == paraphrase:
                self.assertEqual(row["source_support"], "source_composite")
                self.assertIn("stop immediately", row["support_evidence"])
                self.assertIn("slow down", row["support_evidence"])
                self.assertIn("I'm all good", row["support_evidence"])
            else:
                self.assertIn(normalize_text(answer),
                              normalize_text(row["support_evidence"]))

    def test_decisions_and_curation(self) -> None:
        self.assertEqual(len(self.audit), 25)
        self.assertEqual(
            {decision: sum(row["decision"] == decision for row in self.audit)
             for decision in ("keep", "drop", "edit")},
            {"keep": 15, "drop": 4, "edit": 6})
        self.assertEqual(len(self.curation), 10)
        self.assertEqual(
            {action: sum(row["action"] == action for row in self.curation)
             for action in ("drop", "edit")}, {"drop": 4, "edit": 6})
        edits = [row for row in self.curation if row["action"] == "edit"]
        self.assertEqual(
            sum(row["support_type"] == "extractive" for row in edits), 5)
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
        self.assertEqual(report["schema"], "context-merit-audit-report-v29")
        self.assertEqual(report["audit"]["by_decision"],
                         {"drop": 4, "edit": 6, "keep": 15})
        projection = report["isolated_build_projection"]
        self.assertEqual(projection["output_rows"], 587)
        self.assertEqual(
            projection["output_sha256"],
            "e05be81d4c4e2cc9038c9225cbb4372b2bbc627cb4ba219c4dc93c14ad87ba13")
        self.assertEqual(projection["reviewed_keep_fact_ids_preserved"], 15)
        self.assertEqual(projection["prior_pending_addition_fact_ids_preserved"],
                         38)
        self.assertEqual(projection["sealed_eval_fact_count_reported_by_tooling"],
                         612)
        self.assertEqual(projection["unexpected_fact_ids"], 0)
        self.assertTrue(projection["repeat_dataset_byte_identical"])
        self.assertFalse(report["sealed_evaluation_policy"]
                               ["manual_review_opened_eval_or_heldout_content"])

    def test_all_prior_decision_artifacts_are_byte_pinned(self) -> None:
        report = json.loads(builder.REPORT.read_text())
        pins = report["frozen_prior_decision_artifacts"]
        self.assertEqual(len(pins), 84)
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
        expected = {
            "data/train_qa_curated_v1.curation.jsonl":
                "ee28db85bea4a74b0114b4312f52129bc8699a2b4b74c02b755f363e8198edeb",
            "data/train_qa_kinbakutoday.curation.jsonl":
                "8b68caf2a6a4411f9ae26a05759c6ceb6852eaf05c28171b4a11dd93033ebadd",
        }
        for path, digest in expected.items():
            self.assertEqual(builder.file_sha256(builder.ROOT / path), digest)


class IsolatedProjectionV29Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(
            prefix=".test-projection-v29-", dir=HERE)
        output = Path(cls.temporary.name)
        cls.datasets = []
        cls.reports = []
        inputs = [
            "data/train_qa_verified_leakfree_v2.jsonl",
            "data/train_qa_manual_v1.jsonl",
            "data/rope_resource_qa_v1.jsonl",
            "data/rope_resource_factual_qa_v1.jsonl",
            "data/rope_resource_manual_v1.jsonl",
            "data/rope_topia_manual_v1.jsonl",
            *(str(path.relative_to(builder.ROOT))
              for path in builder.PRIOR_PENDING_ADDITIONS),
        ]
        eval_paths = [
            "data/eval_qa.jsonl", "data/eval_qa_v2.jsonl",
            "data/eval_qa_v3.jsonl", "data/ood_qa.jsonl",
            "data/ood_qa_v3.jsonl",
        ]
        curations = [
            *(str(path.relative_to(builder.ROOT))
              for path in builder.ACTIVE_CURATIONS),
            str(builder.QUALITY_MERIT_CURATION.relative_to(builder.ROOT)),
            str(builder.TASUKI_CURATION.relative_to(builder.ROOT)),
            *(str(path.relative_to(builder.ROOT))
              for path in builder.CONTEXT_CURATIONS),
            str(builder.CURATION.relative_to(builder.ROOT)),
        ]
        for run in (1, 2):
            dataset = output / f"projection-{run}.jsonl"
            report = output / f"projection-{run}.report.json"
            command = [
                sys.executable, "build_curated_qa.py",
                "--inputs", *inputs,
                "--eval", *eval_paths,
                "--curation", *curations,
                "--output", str(dataset), "--report", str(report),
            ]
            subprocess.run(command, cwd=builder.ROOT, check=True,
                           stdout=subprocess.DEVNULL)
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
            "e05be81d4c4e2cc9038c9225cbb4372b2bbc627cb4ba219c4dc93c14ad87ba13",
            "e05be81d4c4e2cc9038c9225cbb4372b2bbc627cb4ba219c4dc93c14ad87ba13",
        ])
        self.assertEqual(len(self.rows), 587)
        self.assertEqual(self.report["eval_fact_count"], 612)

    def test_projection_applies_exact_v29_decisions(self) -> None:
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
        self.assertEqual(len({row["fact_id"] for row in self.rows}), 587)
        self.assertEqual(len({row["question"] for row in self.rows}), 587)


if __name__ == "__main__":
    unittest.main()
