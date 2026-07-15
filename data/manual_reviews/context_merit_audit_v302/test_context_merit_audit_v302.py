#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_context_merit_audit_v302 as b


class ContextMeritV302(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix=".test-v302-", dir=HERE)
        directory = Path(cls.temp.name)
        cls.baseline = directory / "baseline.jsonl"
        b.build_baseline(cls.baseline, directory / "baseline.report.json")
        cls.output, cls.output_report = directory / "projection.jsonl", directory / "projection.report.json"
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
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_inputs(self):
        self.assertEqual((len(b.read_jsonl(self.baseline)), b.file_sha256(self.baseline)), (528, b.BASELINE_SHA256))
        self.assertEqual((len(self.additions), b.file_sha256(b.ADDITIONS)), (3, b.EXPECTED_ADDITIONS_SHA256))
        self.assertEqual((len({r["url"] for r in self.additions}), len({r["document_sha256"] for r in self.additions})), (3, 3))

    def test_audit_projection(self):
        self.assertEqual((len(self.audit), {r["decision"] for r in self.audit}, b.read_jsonl(b.CURATION)), (3, {"add"}, []))
        self.assertEqual((len(self.rows), b.file_sha256(self.output)), (531, b.EXPECTED_OUTPUT_SHA256))

    def test_determinism_eval_policy(self):
        self.assertEqual(self.datasets[0], self.datasets[1])
        self.assertEqual(self.reports[0], self.reports[1])
        self.assertEqual(json.loads(self.output_report.read_text())["eval_fact_count"], 612)
        self.assertFalse(self.report["sealed_evaluation_policy"]["manual_worker_opened_eval_or_heldout_content"])

    def test_checkpoint(self):
        self.assertEqual(self.report["prior_checkpoint"]["candidate"], {"rows": 528, "sha256": b.BASELINE_SHA256})
        self.assertTrue(all(b.file_sha256(b.ROOT / row["path"]) == row["sha256"] for row in self.report["prior_checkpoint"]["artifacts"]))

    def test_guards(self):
        norm = lambda value: " ".join(value.casefold().split())
        self.assertEqual({key: sum(n > 1 for n in Counter(norm(row[key]) for row in self.rows).values()) for key in ("fact_id", "question", "answer")}, {"fact_id": 0, "question": 0, "answer": 0})
        blob = self.output.read_text()
        self.assertEqual({token: blob.count(token) for token in ("<|im_start|>", "<|im_end|>", "</think>")}, {"<|im_start|>": 0, "<|im_end|>": 0, "</think>": 0})

    def test_urls_capacity(self):
        manifest = json.loads(b.RESOURCE_MANIFEST.read_text())
        blob = self.output.read_text()
        urls = {url for row in manifest["resources"] for url in (row["canonical_url"], row.get("recommendation_url")) if url}
        self.assertEqual((len(urls), sum(url in blob for url in urls)), (24, 24))
        self.assertEqual(b.conservative_capacity(self.rows), b.EXPECTED_CAPACITY["after"])
        self.assertEqual(self.report["conservative_capacity"]["delta"], {"conflict_units": 3, "equipment_material": 0, "resources_general": 0, "safety_consent": 0, "technique": 3})

    def test_report(self):
        self.assertEqual((self.report["schema"], self.report["isolated_build_projection"]["output_sha256"]), ("context-merit-audit-report-v302", b.EXPECTED_OUTPUT_SHA256))


if __name__ == "__main__":
    unittest.main()
