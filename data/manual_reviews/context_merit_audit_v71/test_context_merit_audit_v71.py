#!/usr/bin/env python3
"""Focused regression tests for Wikipedia history audit v71."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_context_merit_audit_v71 as b


class V71(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix=".test-v71-", dir=HERE)
        output = Path(cls.temp.name)
        cls.base = output / "base.jsonl"
        cls.base_report = output / "base.report.json"
        b.build_projection(cls.base, cls.base_report, b.PRIOR_PROJECTION_CURATIONS)
        cls.baseline = b.read_jsonl(cls.base)
        cls.dataset = output / "projection.jsonl"
        cls.projection_report = output / "projection.report.json"
        cls.dataset_bytes = []
        cls.report_bytes = []
        for _ in (1, 2):
            b.build_projection(cls.dataset, cls.projection_report, b.OUTPUT_PROJECTION_CURATIONS)
            cls.dataset_bytes.append(cls.dataset.read_bytes())
            cls.report_bytes.append(cls.projection_report.read_bytes())
        cls.rows = b.read_jsonl(cls.dataset)
        cls.projection = json.loads(cls.projection_report.read_text())
        cls.audit = b.read_jsonl(b.AUDIT)
        cls.curation = b.read_jsonl(b.CURATION)
        cls.report = json.loads(b.REPORT.read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_01_baseline(self):
        self.assertEqual(
            (len(self.baseline), b.file_sha256(self.base)),
            (537, b.PROJECTED_SELECTION_BASELINE["sha256"]),
        )

    def test_02_eval_tooling(self):
        self.assertEqual(json.loads(self.base_report.read_text())["eval_fact_count"], 612)

    def test_03_selection(self):
        self.assertEqual(
            tuple(item["row"]["fact_id"] for item in b.selected(self.baseline)),
            b.EXPECTED_SELECTION,
        )

    def test_04_indices(self):
        self.assertEqual(
            {
                item["row"]["fact_id"]: item["active_index"]
                for item in b.selected(self.baseline)
            },
            b.PROJECTED_ACTIVE_INDICES,
        )

    def test_05_direct_count(self):
        self.assertEqual(sum(not row.get("curation") for row in self.baseline), 150)

    def test_06_decisions(self):
        self.assertEqual(
            {
                decision: sum(row["decision"] == decision for row in self.audit)
                for decision in ("keep", "drop", "edit")
            },
            {"keep": 0, "drop": 1, "edit": 1},
        )

    def test_07_curation(self):
        self.assertEqual(
            (len(self.curation), {row["action"] for row in self.curation}),
            (2, {"edit", "drop"}),
        )

    def test_08_sources(self):
        audits = {row["fact_id"]: row for row in self.audit}
        for spec in b.SPECS:
            self.assertEqual(
                b.file_sha256(spec["source_path"]),
                audits[spec["fact_id"]]["source_document_file_sha256"],
            )

    def test_09_evidence(self):
        for row in self.audit:
            self.assertEqual(
                b.text_sha256(row["support_evidence"]),
                row["support_evidence_sha256"],
            )

    def test_10_lineage(self):
        for row in self.audit:
            self.assertEqual(
                (row["review_pass"], row["projection_lineage"]["baseline_sha256"]),
                (
                    "wikipedia_history_source_quality_reaudit",
                    b.PROJECTED_SELECTION_BASELINE["sha256"],
                ),
            )

    def test_11_projection(self):
        self.assertEqual(
            (len(self.rows), b.file_sha256(self.dataset)),
            (536, b.EXPECTED_OUTPUT_SHA256),
        )

    def test_12_determinism(self):
        self.assertEqual(
            (self.dataset_bytes[0], self.report_bytes[0]),
            (self.dataset_bytes[1], self.report_bytes[1]),
        )

    def test_13_policy(self):
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
        self.assertNotIn("generator_opens_eval_or_heldout_content", policy)

    def test_14_prior_pins(self):
        pins = self.report["frozen_prior_decision_artifacts"]
        self.assertEqual(len(pins), 210)
        for row in pins:
            self.assertEqual(b.file_sha256(b.ROOT / row["path"]), row["sha256"])

    def test_15_actions_applied(self):
        represented = {
            row.get("curation", {}).get("original_fact_id"): row
            for row in self.rows
            if row.get("curation")
        }
        self.assertEqual(
            represented["fact-7e567cc256ae27660f3f"]["question"],
            b.SPECS[0]["question"],
        )
        self.assertNotIn(
            "fact-e96b48731c29761bc94d",
            {row["fact_id"] for row in self.rows},
        )

    def test_16_additions(self):
        expected = {
            row["fact_id"]
            for path in b.PRIOR_PENDING_ADDITIONS
            for row in b.read_jsonl(path)
        }
        expected -= {"fact-93c032484cf3a72fcc5c", "fact-64a4807147c057265799"}
        represented = {row["fact_id"] for row in self.rows} | {
            row.get("curation", {}).get("original_fact_id")
            for row in self.rows
        }
        self.assertEqual(len(expected), 36)
        self.assertTrue(expected <= represented)

    def test_17_urls(self):
        manifest = json.loads(b.RESOURCE_MANIFEST.read_text())
        urls = {
            url
            for resource in manifest["resources"]
            for url in (resource["canonical_url"], resource.get("recommendation_url"))
            if url
        }
        blob = "\n".join(json.dumps(row) for row in self.rows)
        self.assertEqual((len(urls), {url for url in urls if url not in blob}), (24, set()))

    def test_18_unique(self):
        self.assertEqual(
            (
                len({row["fact_id"] for row in self.rows}),
                len({row["question"] for row in self.rows}),
                self.projection["eval_fact_count"],
            ),
            (536, 536, 612),
        )

    def test_19_report(self):
        self.assertEqual(
            (
                self.report["schema"],
                self.report["audit"]["by_decision"],
                self.report["isolated_build_projection"]["output_sha256"],
            ),
            (
                "context-merit-audit-report-v71",
                {"drop": 1, "edit": 1},
                b.EXPECTED_OUTPUT_SHA256,
            ),
        )


if __name__ == "__main__":
    unittest.main()
