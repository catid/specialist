#!/usr/bin/env python3
"""Focused regression tests for the natural wet-jute wording repair v70."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_context_merit_audit_v70 as b


class V70(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix=".test-v70-", dir=HERE)
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
        cls.audit = b.read_jsonl(b.AUDIT)
        cls.curation = b.read_jsonl(b.CURATION)
        cls.report = json.loads(b.REPORT.read_text())
        cls.projection = json.loads(cls.projection_report.read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_01_baseline(self):
        self.assertEqual(
            (len(self.baseline), b.file_sha256(self.base)),
            (537, b.PROJECTED_SELECTION_BASELINE["sha256"]),
        )

    def test_02_eval_tooling_count(self):
        self.assertEqual(json.loads(self.base_report.read_text())["eval_fact_count"], 612)

    def test_03_audit(self):
        self.assertEqual(
            (len(self.audit), self.audit[0]["fact_id"], self.audit[0]["decision"]),
            (1, b.ORIGINAL_FACT_ID, "edit"),
        )

    def test_04_exact_natural_question(self):
        rows = [
            row
            for row in self.rows
            if row.get("curation", {}).get("original_fact_id") == b.ORIGINAL_FACT_ID
        ]
        self.assertEqual(
            [(row["question"], row["answer"]) for row in rows],
            [("How can jute rope feel after getting wet and drying out?", "spongy and springy")],
        )

    def test_05_awkward_question_absent(self):
        self.assertNotIn(b.AWKWARD_QUESTION, {row["question"] for row in self.rows})

    def test_06_curation_supersession(self):
        self.assertEqual(
            ({row["fact_id"] for row in self.curation}, len(self.curation)),
            (
                {
                    b.ORIGINAL_FACT_ID,
                    "fact-069a861dbb2bea9e47ca",
                    "fact-f7e802bf0b2759290dc6",
                },
                3,
            ),
        )

    def test_07_carried_v69_semantics(self):
        old = {row["fact_id"]: row for row in b.read_jsonl(b.V69_CURATION)}
        new = {row["fact_id"]: row for row in self.curation}
        for fact_id in ("fact-069a861dbb2bea9e47ca", "fact-f7e802bf0b2759290dc6"):
            self.assertEqual(
                (new[fact_id]["question"], new[fact_id]["answer"]),
                (old[fact_id]["question"], old[fact_id]["answer"]),
            )

    def test_08_source(self):
        self.assertEqual(b.file_sha256(b.SOURCE), self.audit[0]["source_document_file_sha256"])

    def test_09_evidence(self):
        self.assertEqual(
            b.text_sha256(self.audit[0]["support_evidence"]),
            self.audit[0]["support_evidence_sha256"],
        )

    def test_10_output(self):
        self.assertEqual(
            (len(self.rows), b.file_sha256(self.dataset)),
            (537, b.EXPECTED_OUTPUT_SHA256),
        )

    def test_11_dataset_determinism(self):
        self.assertEqual(self.dataset_bytes[0], self.dataset_bytes[1])

    def test_12_report_determinism(self):
        self.assertEqual(self.report_bytes[0], self.report_bytes[1])

    def test_13_report_observations(self):
        normalized = dict(self.projection)
        normalized["output"] = "<projection-output>"
        digest = b.sha_bytes((json.dumps(normalized, indent=2, sort_keys=True) + "\n").encode())
        projection = self.report["isolated_build_projection"]
        self.assertEqual(
            (
                projection["repeat_dataset_byte_identical"],
                projection["repeat_projection_report_byte_identical"],
                projection["projection_report_normalized_sha256"],
            ),
            (True, True, digest),
        )

    def test_14_sealed_policy(self):
        policy = self.report["sealed_evaluation_policy"]
        self.assertEqual(
            (
                policy["manual_worker_opened_eval_or_heldout_content"],
                policy["manual_worker_received_eval_or_heldout_content"],
                policy["automated_collision_tool_reads_sealed_content"],
            ),
            (False, False, True),
        )
        self.assertNotIn("generator_opens_eval_or_heldout_content", policy)

    def test_15_prior_pins(self):
        pins = self.report["frozen_prior_decision_artifacts"]
        self.assertEqual(len(pins), 207)
        for row in pins:
            self.assertEqual(b.file_sha256(b.ROOT / row["path"]), row["sha256"])

    def test_16_direct_count(self):
        self.assertEqual(sum(not row.get("curation") for row in self.rows), 150)

    def test_17_additions(self):
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

    def test_18_urls(self):
        manifest = json.loads(b.RESOURCE_MANIFEST.read_text())
        urls = {
            url
            for resource in manifest["resources"]
            for url in (resource["canonical_url"], resource.get("recommendation_url"))
            if url
        }
        blob = "\n".join(json.dumps(row) for row in self.rows)
        self.assertEqual((len(urls), {url for url in urls if url not in blob}), (24, set()))

    def test_19_unique_and_report(self):
        self.assertEqual(
            (
                len({row["fact_id"] for row in self.rows}),
                len({row["question"] for row in self.rows}),
                self.projection["eval_fact_count"],
                self.report["schema"],
                self.report["isolated_build_projection"]["output_sha256"],
            ),
            (537, 537, 612, "context-merit-audit-report-v70", b.EXPECTED_OUTPUT_SHA256),
        )


if __name__ == "__main__":
    unittest.main()
