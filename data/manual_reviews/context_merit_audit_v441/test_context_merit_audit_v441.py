#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_context_merit_audit_v441 as b


class ContextMeritV441(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b.main()
        cls.tmp = tempfile.TemporaryDirectory(prefix=".test-v441-", dir=HERE)
        d = Path(cls.tmp.name)
        cls.base = d / "base.jsonl"
        b.build_baseline(cls.base, d / "base.report.json")
        cls.out = d / "out.jsonl"
        cls.projection_report = d / "out.report.json"
        cls.datasets = []
        cls.reports = []
        for _ in (1, 2):
            b.build_projection(cls.out, cls.projection_report)
            cls.datasets.append(cls.out.read_bytes())
            cls.reports.append(cls.projection_report.read_bytes())
        cls.base_rows = b.read_jsonl(cls.base)
        cls.rows = b.read_jsonl(cls.out)
        cls.report = json.loads(b.REPORT.read_text())
        cls.audits = b.read_jsonl(b.AUDIT)
        cls.curations = b.read_jsonl(b.CURATION)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_exactly_three_projection_edits(self):
        self.assertEqual(
            (len(self.base_rows), b.file_sha256(self.base), len(self.rows), b.file_sha256(self.out)),
            (531, b.BASELINE_SHA256, 531, b.EXPECTED_OUTPUT_SHA256),
        )
        self.assertEqual((self.datasets[0], self.reports[0]), (self.datasets[1], self.reports[1]))
        selected = {spec["fact_id"] for spec in b.SPECS}
        base_facts = {row["fact_id"] for row in self.base_rows}
        output_facts = {row["fact_id"] for row in self.rows}
        self.assertEqual(base_facts - output_facts, selected)
        self.assertEqual(len(output_facts - base_facts), 3)
        edited = {row.get("curation", {}).get("original_fact_id"): row for row in self.rows}
        self.assertEqual(
            {
                key: (value["question"], value["answer"])
                for key, value in edited.items()
                if key in selected
            },
            {spec["fact_id"]: (spec["question"], spec["answer"]) for spec in b.SPECS},
        )

    def test_full_snapshot_source_evidence_and_url_gate(self):
        self.assertEqual(
            (len(self.curations), Counter(row["action"] for row in self.curations)),
            (3, Counter({"edit": 3})),
        )
        self.assertEqual(self.report["audit"]["bounded_manual_selection_rows"], 3)
        by_fact = {row["fact_id"]: row for row in self.curations}
        audit_by_fact = {row["fact_id"]: row for row in self.audits}
        for spec in b.SPECS:
            curation = by_fact[spec["fact_id"]]
            audit = audit_by_fact[spec["fact_id"]]
            source_document = b.ROOT / spec["source_document"]
            source = json.loads(source_document.read_text())
            self.assertIn(curation["evidence"], source["text"])
            self.assertEqual(source["url"], curation["evidence_url"])
            self.assertEqual(audit["source_snapshot_chars_manually_reviewed"], len(source["text"]))
            self.assertEqual(
                (audit["support_evidence"], audit["support_evidence_sha256"]),
                (curation["evidence"], b.text_sha256(curation["evidence"])),
            )
            self.assertEqual(audit["source_document_file_sha256"], b.file_sha256(source_document))
            self.assertTrue(curation["source_lineage"])

    def test_eval_duplicate_and_protocol_invariants(self):
        self.assertEqual(json.loads(self.projection_report.read_text())["eval_fact_count"], 612)
        normalize = lambda text: " ".join(text.casefold().split())
        self.assertEqual(
            {
                key: sum(value > 1 for value in Counter(normalize(row[key]) for row in self.rows).values())
                for key in ("fact_id", "question", "answer")
            },
            {"fact_id": 0, "question": 0, "answer": 0},
        )
        self.assertFalse(
            any(token in self.out.read_text() for token in ("<|im_start|>", "<|im_end|>", "</think>"))
        )

    def test_urls_capacity_and_sealed_policy(self):
        manifest = json.loads(b.RESOURCE_MANIFEST.read_text())
        urls = {
            url
            for resource in manifest["resources"]
            for url in (resource["canonical_url"], resource.get("recommendation_url"))
            if url
        }
        blob = self.out.read_text()
        self.assertEqual((len(urls), sum(url in blob for url in urls)), (24, 24))
        self.assertEqual(b.conservative_capacity(self.rows), b.EXPECTED_CAPACITY_AFTER)
        policy = self.report["sealed_evaluation_policy"]
        self.assertFalse(policy["manual_worker_opened_eval_heldout_ood_shadow_or_benchmark_content"])
        self.assertFalse(policy["manual_worker_received_eval_heldout_ood_shadow_or_benchmark_content"])


if __name__ == "__main__":
    unittest.main()
