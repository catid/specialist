#!/usr/bin/env python3
import json
import re
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_context_merit_audit_v443 as b


class ContextMeritV443(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b.main()
        cls.tmp = tempfile.TemporaryDirectory(prefix=".test-v443-", dir=HERE)
        directory = Path(cls.tmp.name)
        cls.base = directory / "base.jsonl"
        b.build_baseline(cls.base, directory / "base.report.json")
        cls.out = directory / "out.jsonl"
        cls.projection_report = directory / "out.report.json"
        cls.datasets, cls.reports = [], []
        for _ in (1, 2):
            b.build_projection(cls.out, cls.projection_report)
            cls.datasets.append(cls.out.read_bytes())
            cls.reports.append(cls.projection_report.read_bytes())
        cls.base_rows = b.read_jsonl(cls.base)
        cls.rows = b.read_jsonl(cls.out)
        cls.audits = b.read_jsonl(b.AUDIT)
        cls.curations = b.read_jsonl(b.CURATION)
        cls.additions = b.read_jsonl(b.ADDITIONS)
        cls.report = json.loads(b.REPORT.read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_exact_15_row_repair_and_524_row_projection(self):
        self.assertEqual(
            (len(self.base_rows), b.file_sha256(self.base), len(self.rows), b.file_sha256(self.out)),
            (531, b.BASELINE_SHA256, 524, b.EXPECTED_OUTPUT_SHA256),
        )
        self.assertEqual((self.datasets[0], self.reports[0]), (self.datasets[1], self.reports[1]))
        direct_ids = {value[1] for value in b.DIRECT.values()}
        base_ids = {row["fact_id"] for row in self.base_rows}
        output_ids = {row["fact_id"] for row in self.rows}
        self.assertEqual(base_ids - output_ids, direct_ids)
        self.assertEqual(output_ids - base_ids, {row["fact_id"] for row in self.additions})
        self.assertEqual((len(self.curations), Counter(row["action"] for row in self.curations)), (15, Counter(drop=15)))
        self.assertEqual(len(self.additions), 8)

    def test_full_direct_and_successor_inventory(self):
        self.assertEqual(len(self.audits), 22)
        self.assertEqual(Counter(row["decision"] for row in self.audits), Counter(replace=8, drop=7, keep=7))
        direct_audits = [row for row in self.audits if row["decision"] != "keep"]
        self.assertEqual({row["fact_id"] for row in direct_audits}, {value[1] for value in b.DIRECT.values()})
        self.assertTrue(all(row["active_answer"].startswith("https://rope-topia.com/") for row in direct_audits))
        replacement_audits = [row for row in direct_audits if row["decision"] == "replace"]
        self.assertEqual({row["replacement_fact_id"] for row in replacement_audits}, {row["fact_id"] for row in self.additions})
        self.assertTrue(all(row["support_evidence"] for row in self.audits if row["decision"] == "keep"))

    def test_replacement_evidence_is_exact_and_hash_pinned(self):
        snapshots = b.archive_evidence()
        for row in self.additions:
            resource_id = row["source_lineage"]["ropetopia_manifest_resource_id"]
            self.assertEqual(row["evidence"], snapshots[resource_id]["exact_qa_evidence"])
            self.assertEqual(row["evidence_sha256"], b.text_sha256(row["evidence"]))
            if row["source"] == "wykd":
                source_path = b.ROOT / row["source_lineage"]["raw_successor_document"]
                self.assertIn(row["evidence"], json.loads(source_path.read_text())["text"])
            else:
                self.assertEqual(row["document_sha256"], snapshots[resource_id]["archive"]["extracted_body_sha256"])
                self.assertEqual(len(row["source_lineage"]["archive_html_sha256"]), 64)
                self.assertTrue(row["source_lineage"]["archive_capture_timestamp"].isdigit())

    def test_metadata_trivia_regressions_are_absent(self):
        for row in self.rows:
            self.assertFalse(any(pattern.search(row["question"]) for pattern in b.TRIVIA_PATTERNS), row["question"])
        self.assertFalse(any(row.get("source") == "rope_topia" for row in self.rows))
        self.assertFalse(any(row.get("source") == "rope_topia_archive" and row.get("kind") == "qa_resource_index" for row in self.rows))
        self.assertFalse(any(row.get("source") == "rope_topia_archive" and row["answer"].strip().startswith(("http://", "https://")) for row in self.rows))

    def test_eval_duplicate_protocol_url_and_capacity_invariants(self):
        self.assertEqual(json.loads(self.projection_report.read_text())["eval_fact_count"], 612)
        normalize = lambda text: " ".join(text.casefold().split())
        self.assertEqual({
            key: sum(value > 1 for value in Counter(normalize(row[key]) for row in self.rows).values())
            for key in ("fact_id", "question", "answer")
        }, {"fact_id": 0, "question": 0, "answer": 0})
        self.assertFalse(any(token in self.out.read_text() for token in ("<|im_start|>", "<|im_end|>", "</think>")))
        resources = json.loads(b.RESOURCE_MANIFEST.read_text())["resources"]
        urls = {url for resource in resources for url in (resource["canonical_url"], resource.get("recommendation_url")) if url}
        blob = self.out.read_text()
        self.assertEqual((len(urls), sum(url in blob for url in urls)), (24, 24))
        self.assertEqual(b.conservative_capacity(self.rows), b.EXPECTED_CAPACITY_AFTER)

    def test_lineage_count_corpus_and_sealed_policy(self):
        recovered = [
            row for row in self.rows
            if row.get("source") == "rope_topia_archive"
            or (row.get("source") == "wykd" and (row.get("source_lineage") or {}).get("ropetopia_manifest_resource_id"))
        ]
        self.assertEqual(len(recovered), 15)
        corpus_report = json.loads(b.corpus.REPORT.read_text())
        self.assertEqual(corpus_report["coverage"]["indexed_pages"], 15)
        self.assertEqual(corpus_report["coverage"]["substantive_pages_included"], 8)
        self.assertEqual(corpus_report["coverage"]["pages_excluded_with_reason"], 7)
        self.assertEqual(self.report["v52_isolation"], {
            "active_v52_inputs_modified": False,
            "candidate_status": "isolated_pending_not_promoted",
        })
        policy = self.report["sealed_evaluation_policy"]
        self.assertFalse(policy["manual_worker_opened_eval_heldout_ood_shadow_or_benchmark_content"])
        self.assertFalse(policy["manual_worker_received_eval_heldout_ood_shadow_or_benchmark_content"])


if __name__ == "__main__":
    unittest.main()
