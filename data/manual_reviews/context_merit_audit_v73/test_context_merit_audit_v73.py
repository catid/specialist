#!/usr/bin/env python3
"""Focused regression tests for Rope365 safety audit v73."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_context_merit_audit_v73 as b


class V73(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix=".test-v73-", dir=HERE)
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
            (536, b.PROJECTED_SELECTION_BASELINE["sha256"]),
        )

    def test_02_eval_tooling(self):
        self.assertEqual(json.loads(self.base_report.read_text())["eval_fact_count"], 612)

    def test_03_selection_and_indices(self):
        by_fact_id = {
            row["fact_id"]: index for index, row in enumerate(self.baseline, 1)
        }
        self.assertEqual(
            {fact_id: by_fact_id[fact_id] for fact_id in b.EXPECTED_SELECTION},
            b.PROJECTED_ACTIVE_INDICES,
        )

    def test_04_direct_count(self):
        self.assertEqual(sum(not row.get("curation") for row in self.baseline), 145)

    def test_05_decisions(self):
        self.assertEqual(
            (len(self.audit), {row["decision"] for row in self.audit}),
            (4, {"edit"}),
        )

    def test_06_curation(self):
        self.assertEqual(
            (len(self.curation), {row["action"] for row in self.curation}),
            (4, {"edit"}),
        )

    def test_07_support_types(self):
        self.assertEqual({row["support_type"] for row in self.curation}, {"manual_paraphrase"})
        self.assertTrue(all(row["paraphrase_rationale"] for row in self.curation))

    def test_08_source(self):
        self.assertTrue(
            all(
                row["source_document_file_sha256"] == b.file_sha256(b.SOURCE)
                for row in self.audit
            )
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
                    "rope365_safety_action_and_mechanism_reaudit",
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
        self.assertEqual(len(pins), 216)
        for row in pins:
            self.assertEqual(b.file_sha256(b.ROOT / row["path"]), row["sha256"])

    def test_15_actions_applied(self):
        represented = {
            row.get("curation", {}).get("original_fact_id"): row
            for row in self.rows
            if row.get("curation")
        }
        for spec in b.SPECS:
            self.assertEqual(
                (represented[spec["fact_id"]]["question"], represented[spec["fact_id"]]["answer"]),
                (spec["question"], spec["answer"]),
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
                "context-merit-audit-report-v73",
                {"edit": 4},
                b.EXPECTED_OUTPUT_SHA256,
            ),
        )


if __name__ == "__main__":
    unittest.main()
