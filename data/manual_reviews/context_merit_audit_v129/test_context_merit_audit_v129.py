#!/usr/bin/env python3
"""Regression tests for the v129 hermetic second-stage projection repair."""
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_context_merit_audit_v129 as b


class V129(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix=".test-v129-", dir=HERE)
        out = Path(cls.temp.name)
        cls.base = out / "base.jsonl"
        cls.base_report = out / "base.report.json"
        b.build_projection(cls.base, cls.base_report, b.PRIOR_PROJECTION_CURATIONS)
        cls.baseline = b.read_jsonl(cls.base)
        cls.dataset = out / "projection.jsonl"
        cls.projection_report = out / "projection.report.json"
        cls.dataset_bytes = []
        cls.report_bytes = []
        for _ in (1, 2):
            b.build_projection(cls.dataset, cls.projection_report, b.OUTPUT_PROJECTION_CURATIONS)
            cls.dataset_bytes.append(cls.dataset.read_bytes())
            cls.report_bytes.append(cls.projection_report.read_bytes())
        cls.rows = b.read_jsonl(cls.dataset)
        cls.projected = json.loads(cls.projection_report.read_text())
        cls.audit = b.read_jsonl(b.AUDIT)
        cls.curation = b.read_jsonl(b.CURATION)
        cls.report = json.loads(b.REPORT.read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_01_baseline(self):
        self.assertEqual((len(self.baseline), b.file_sha256(self.base)), (515, b.PROJECTED_SELECTION_BASELINE["sha256"]))

    def test_02_eval(self):
        self.assertEqual(json.loads(self.base_report.read_text())["eval_fact_count"], 612)

    def test_03_selection(self):
        self.assertEqual((b.EXPECTED_SELECTION, b.PROJECTED_ACTIVE_INDICES), ((), {}))

    def test_04_direct(self):
        self.assertEqual(sum(not row.get("curation") for row in self.baseline), 92)

    def test_05_decisions(self):
        self.assertEqual(self.audit, [])

    def test_06_curation(self):
        self.assertEqual(self.curation, [])

    def test_07_bridge_provenance(self):
        self.assertEqual(self.projected["inputs"], [b.BRIDGE_INPUT_LABEL])
        self.assertEqual(list(self.projected["input_sha256"]), [b.BRIDGE_INPUT_LABEL])
        self.assertEqual(list(self.projected["counts"]["by_input"]), [b.BRIDGE_INPUT_LABEL])

    def test_08_projection(self):
        self.assertEqual((len(self.rows), b.file_sha256(self.dataset)), (515, b.EXPECTED_OUTPUT_SHA256))

    def test_09_determinism(self):
        self.assertEqual((self.dataset_bytes[0], self.report_bytes[0]), (self.dataset_bytes[1], self.report_bytes[1]))

    def test_10_policy(self):
        policy = self.report["sealed_evaluation_policy"]
        self.assertEqual(
            (
                policy["manual_worker_opened_eval_or_heldout_content"],
                policy["manual_worker_received_eval_or_heldout_content"],
                policy["automated_collision_tool_reads_sealed_content"],
                self.report["isolated_build_projection"]["repeat_projection_report_byte_identical"],
            ),
            (False, False, True, True),
        )

    def test_11_pins(self):
        pins = self.report["frozen_prior_decision_artifacts"]
        self.assertEqual(len(pins), 384)
        for row in pins:
            self.assertEqual(b.file_sha256(b.ROOT / row["path"]), row["sha256"])

    def test_12_no_content_change(self):
        self.assertEqual(self.dataset_bytes[0], self.base.read_bytes())

    def test_13_additions(self):
        expected = {row["fact_id"] for path in b.PRIOR_PENDING_ADDITIONS for row in b.read_jsonl(path)}
        expected -= {"fact-93c032484cf3a72fcc5c", "fact-64a4807147c057265799", "fact-63c654d8cdad2602da36"}
        represented = {row["fact_id"] for row in self.rows} | {row.get("curation", {}).get("original_fact_id") for row in self.rows}
        self.assertEqual(len(expected), 35)
        self.assertTrue(expected <= represented)

    def test_14_urls(self):
        manifest = json.loads(b.RESOURCE_MANIFEST.read_text())
        urls = {url for row in manifest["resources"] for url in (row["canonical_url"], row.get("recommendation_url")) if url}
        blob = "\n".join(json.dumps(row) for row in self.rows)
        self.assertEqual((len(urls), {url for url in urls if url not in blob}), (24, set()))

    def test_15_unique(self):
        self.assertEqual(
            (len({row["fact_id"] for row in self.rows}), len({row["question"] for row in self.rows}), self.projected["eval_fact_count"]),
            (515, 515, 612),
        )

    def test_16_report(self):
        self.assertEqual(
            (self.report["schema"], self.report["audit"]["by_decision"], self.report["isolated_build_projection"]["output_sha256"]),
            ("context-merit-audit-report-v129", {"drop": 0, "edit": 0, "keep": 0}, b.EXPECTED_OUTPUT_SHA256),
        )

    def test_17_no_fixed_scratch(self):
        self.assertFalse((HERE / ".v117-squashed-input.jsonl").exists())
        self.assertFalse((HERE / ".v117-squashed-input.report.json").exists())
        self.assertEqual(list(HERE.glob(".v129-bridge-*")), [])

    def test_18_concurrent_bridge_builds(self):
        out = Path(self.temp.name)

        def build(label):
            dataset = out / f"concurrent-{label}.jsonl"
            report = out / f"concurrent-{label}.report.json"
            b.build_projection(dataset, report, b.OUTPUT_PROJECTION_CURATIONS)
            parsed = json.loads(report.read_text())
            parsed["output"] = "<projection-output>"
            canonical_report = json.dumps(parsed, indent=2, sort_keys=True) + "\n"
            return dataset.read_bytes(), canonical_report.encode()

        with ThreadPoolExecutor(max_workers=2) as pool:
            first, second = pool.map(build, ("a", "b"))
        self.assertEqual(first, second)
        self.assertEqual(list(HERE.glob(".v129-bridge-*")), [])


if __name__ == "__main__":
    unittest.main()
