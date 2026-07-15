#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_context_merit_audit_v306 as b


class ContextMeritV306(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix=".test-v306-", dir=HERE)
        d = Path(cls.temp.name)
        cls.base = d / "base.jsonl"
        b.build_baseline(cls.base, d / "base.report.json")
        cls.out = d / "out.jsonl"
        cls.projection_report = d / "out.report.json"
        cls.datasets, cls.reports = [], []
        for _ in (1, 2):
            b.build_projection(cls.out, cls.projection_report)
            cls.datasets.append(cls.out.read_bytes())
            cls.reports.append(cls.projection_report.read_bytes())
        cls.rows = b.read_jsonl(cls.out)
        cls.report = json.loads(b.REPORT.read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_projection(self):
        self.assertEqual(
            (len(b.read_jsonl(self.base)), b.file_sha256(self.base), len(self.rows), b.file_sha256(self.out)),
            (539, b.BASELINE_SHA256, 542, b.EXPECTED_OUTPUT_SHA256),
        )
        self.assertEqual((self.datasets[0], self.reports[0]), (self.datasets[1], self.reports[1]))

    def test_eval_and_duplicate_guards(self):
        self.assertEqual(json.loads(self.projection_report.read_text())["eval_fact_count"], 612)
        normalize = lambda value: " ".join(value.casefold().split())
        self.assertEqual(
            {k: sum(count > 1 for count in Counter(normalize(r[k]) for r in self.rows).values()) for k in ("fact_id", "question", "answer")},
            {"fact_id": 0, "question": 0, "answer": 0},
        )
        blob = self.out.read_text()
        self.assertFalse(any(token in blob for token in ("<|im_start|>", "<|im_end|>", "</think>")))

    def test_urls_capacity_and_policy(self):
        manifest = json.loads(b.RESOURCE_MANIFEST.read_text())
        urls = {u for resource in manifest["resources"] for u in (resource["canonical_url"], resource.get("recommendation_url")) if u}
        blob = self.out.read_text()
        self.assertEqual((len(urls), sum(url in blob for url in urls)), (24, 24))
        self.assertEqual(b.conservative_capacity(self.rows), b.EXPECTED_CAPACITY["after"])
        self.assertEqual(
            self.report["conservative_capacity"]["delta"],
            {"conflict_units": 3, "equipment_material": 0, "resources_general": 1, "safety_consent": 1, "technique": 1},
        )
        self.assertFalse(self.report["sealed_evaluation_policy"]["manual_worker_opened_eval_or_heldout_content"])


if __name__ == "__main__":
    unittest.main()
