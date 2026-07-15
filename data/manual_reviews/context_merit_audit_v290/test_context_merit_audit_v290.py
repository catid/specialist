#!/usr/bin/env python3
"""Regression tests for integrating three distinct technique/equipment documents."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_context_merit_audit_v290 as b


class ContextMeritV290(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix=".test-v290-", dir=HERE)
        directory = Path(cls.temp.name)
        cls.baseline = directory / "baseline.jsonl"
        cls.baseline_report = directory / "baseline.report.json"
        b.build_v289_baseline(cls.baseline, cls.baseline_report)
        cls.baseline_rows = b.read_jsonl(cls.baseline)
        cls.output = directory / "projection.jsonl"
        cls.output_report = directory / "projection.report.json"
        cls.datasets = []
        cls.reports = []
        for _ in (1, 2):
            b.build_projection(cls.output, cls.output_report)
            cls.datasets.append(cls.output.read_bytes())
            cls.reports.append(cls.output_report.read_bytes())
        cls.rows = b.read_jsonl(cls.output)
        cls.additions = b.read_jsonl(b.ADDITIONS)
        cls.audit = b.read_jsonl(b.AUDIT)
        cls.report = json.loads(b.REPORT.read_text())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_01_baseline(self) -> None:
        self.assertEqual((len(self.baseline_rows), b.file_sha256(self.baseline)), (492, b.PROJECTED_SELECTION_BASELINE["sha256"]))

    def test_02_additions(self) -> None:
        self.assertEqual((len(self.additions), b.file_sha256(b.ADDITIONS)), (3, b.EXPECTED_ADDITIONS_SHA256))
        self.assertEqual(len({row["document_sha256"] for row in self.additions}), 3)
        self.assertEqual(len({row["url"] for row in self.additions}), 3)

    def test_03_audit(self) -> None:
        self.assertEqual((len(self.audit), {row["decision"] for row in self.audit}), (3, {"add"}))
        self.assertEqual({row["fact_id"] for row in self.audit}, {row["fact_id"] for row in self.additions})

    def test_04_no_curation(self) -> None:
        self.assertEqual(b.read_jsonl(b.CURATION), [])
        self.assertEqual(b.OUTPUT_PROJECTION_CURATIONS, b.PRIOR_PROJECTION_CURATIONS)

    def test_05_projection(self) -> None:
        self.assertEqual((len(self.rows), b.file_sha256(self.output)), (495, b.EXPECTED_OUTPUT_SHA256))
        projected = {row["fact_id"]: row for row in self.rows}
        for row in self.additions:
            self.assertEqual((projected[row["fact_id"]]["question"], projected[row["fact_id"]]["answer"]), (row["question"], row["answer"]))

    def test_06_determinism(self) -> None:
        self.assertEqual(self.datasets[0], self.datasets[1])
        self.assertEqual(self.reports[0], self.reports[1])

    def test_07_aggregate_eval_only(self) -> None:
        self.assertEqual(json.loads(self.output_report.read_text())["eval_fact_count"], 612)
        policy = self.report["sealed_evaluation_policy"]
        self.assertEqual((policy["manual_worker_opened_eval_or_heldout_content"], policy["manual_worker_received_eval_or_heldout_content"]), (False, False))

    def test_08_prior_pins(self) -> None:
        pins = self.report["frozen_prior_decision_artifacts"]
        self.assertEqual(len(pins), 867)
        self.assertTrue(all(b.file_sha256(b.ROOT / row["path"]) == row["sha256"] for row in pins))

    def test_09_content_guards(self) -> None:
        norm = lambda value: " ".join(value.casefold().split())
        self.assertEqual({key: sum(count > 1 for count in Counter(norm(row[key]) for row in self.rows).values()) for key in ("fact_id", "question", "answer")}, {"fact_id": 0, "question": 0, "answer": 0})
        blob = self.output.read_text()
        self.assertEqual({token: blob.count(token) for token in ("<|im_start|>", "<|im_end|>", "</think>")}, {"<|im_start|>": 0, "<|im_end|>": 0, "</think>": 0})

    def test_10_resource_urls(self) -> None:
        manifest = json.loads(b.RESOURCE_MANIFEST.read_text())
        urls = {url for row in manifest["resources"] for url in (row["canonical_url"], row.get("recommendation_url")) if url}
        blob = self.output.read_text()
        self.assertEqual((len(urls), sum(url in blob for url in urls)), (24, 24))

    def test_11_capacity_delta(self) -> None:
        before = b.conservative_capacity(self.baseline_rows)
        after = b.conservative_capacity(self.rows)
        self.assertEqual(before, b.EXPECTED_CAPACITY["before"])
        self.assertEqual(after, b.EXPECTED_CAPACITY["after"])
        self.assertEqual(self.report["conservative_capacity"]["delta"], {"conflict_units": 3, "equipment_material": 2, "resources_general": 0, "safety_consent": 0, "technique": 1})

    def test_12_report(self) -> None:
        self.assertEqual((self.report["schema"], self.report["isolated_build_projection"]["output_sha256"]), ("context-merit-audit-report-v290", b.EXPECTED_OUTPUT_SHA256))


if __name__ == "__main__":
    unittest.main()
