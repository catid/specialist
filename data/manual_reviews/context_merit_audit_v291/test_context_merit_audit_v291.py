#!/usr/bin/env python3
"""Regression tests for integrating the second technique tranche."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_context_merit_audit_v291 as b


class ContextMeritV291(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix=".test-v291-", dir=HERE)
        directory = Path(cls.temp.name)
        cls.baseline = directory / "baseline.jsonl"
        b.build_baseline(cls.baseline, directory / "baseline.report.json")
        cls.output = directory / "projection.jsonl"
        cls.output_report = directory / "projection.report.json"
        cls.datasets, cls.reports = [], []
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

    def test_01_baseline_and_additions(self) -> None:
        self.assertEqual((len(b.read_jsonl(self.baseline)), b.file_sha256(self.baseline)), (495, b.BASELINE_SHA256))
        self.assertEqual((len(self.additions), b.file_sha256(b.ADDITIONS)), (3, b.EXPECTED_ADDITIONS_SHA256))
        self.assertEqual((len({r["url"] for r in self.additions}), len({r["document_sha256"] for r in self.additions})), (3, 3))

    def test_02_audit_and_empty_curation(self) -> None:
        self.assertEqual((len(self.audit), {r["decision"] for r in self.audit}), (3, {"add"}))
        self.assertEqual({r["fact_id"] for r in self.audit}, {r["fact_id"] for r in self.additions})
        self.assertEqual(b.read_jsonl(b.CURATION), [])

    def test_03_projection(self) -> None:
        self.assertEqual((len(self.rows), b.file_sha256(self.output)), (498, b.EXPECTED_OUTPUT_SHA256))
        projected = {row["fact_id"]: row for row in self.rows}
        self.assertTrue(all(row["fact_id"] in projected for row in self.additions))

    def test_04_determinism_and_eval_aggregate(self) -> None:
        self.assertEqual(self.datasets[0], self.datasets[1])
        self.assertEqual(self.reports[0], self.reports[1])
        self.assertEqual(json.loads(self.output_report.read_text())["eval_fact_count"], 612)
        self.assertFalse(self.report["sealed_evaluation_policy"]["manual_worker_opened_eval_or_heldout_content"])

    def test_05_prior_pins(self) -> None:
        pins = self.report["frozen_prior_decision_artifacts"]
        self.assertEqual(len(pins), 870)
        self.assertTrue(all(b.file_sha256(b.ROOT / row["path"]) == row["sha256"] for row in pins))

    def test_06_content_guards(self) -> None:
        norm = lambda value: " ".join(value.casefold().split())
        self.assertEqual({key: sum(n > 1 for n in Counter(norm(row[key]) for row in self.rows).values()) for key in ("fact_id", "question", "answer")}, {"fact_id": 0, "question": 0, "answer": 0})
        blob = self.output.read_text()
        self.assertEqual({token: blob.count(token) for token in ("<|im_start|>", "<|im_end|>", "</think>")}, {"<|im_start|>": 0, "<|im_end|>": 0, "</think>": 0})

    def test_07_resource_urls(self) -> None:
        manifest = json.loads(b.RESOURCE_MANIFEST.read_text())
        urls = {url for row in manifest["resources"] for url in (row["canonical_url"], row.get("recommendation_url")) if url}
        self.assertEqual((len(urls), sum(url in self.output.read_text() for url in urls)), (24, 24))

    def test_08_capacity(self) -> None:
        self.assertEqual(b.conservative_capacity(b.read_jsonl(self.baseline)), b.EXPECTED_CAPACITY["before"])
        self.assertEqual(b.conservative_capacity(self.rows), b.EXPECTED_CAPACITY["after"])
        self.assertEqual(self.report["conservative_capacity"]["delta"], {"conflict_units": 3, "equipment_material": 0, "resources_general": 0, "safety_consent": 0, "technique": 3})

    def test_09_report(self) -> None:
        self.assertEqual((self.report["schema"], self.report["isolated_build_projection"]["output_sha256"]), ("context-merit-audit-report-v291", b.EXPECTED_OUTPUT_SHA256))


if __name__ == "__main__":
    unittest.main()
