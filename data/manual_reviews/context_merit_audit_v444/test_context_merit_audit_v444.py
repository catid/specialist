#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_context_merit_audit_v444 as b


class ContextMeritV444(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b.main()
        cls.tmp = tempfile.TemporaryDirectory(prefix=".test-v444-", dir=HERE)
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
        cls.report = json.loads(b.REPORT.read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_exact_six_drops_and_518_row_projection(self):
        self.assertEqual(
            (len(self.base_rows), b.file_sha256(self.base), len(self.rows), b.file_sha256(self.out)),
            (524, b.BASELINE_SHA256, 518, b.EXPECTED_OUTPUT_SHA256),
        )
        self.assertEqual((self.datasets[0], self.reports[0]), (self.datasets[1], self.reports[1]))
        base_ids = {row["fact_id"] for row in self.base_rows}
        output_ids = {row["fact_id"] for row in self.rows}
        self.assertEqual(base_ids - output_ids, set(b.DROP_SPECS))
        self.assertFalse(output_ids - base_ids)
        self.assertEqual((len(self.curations), Counter(row["action"] for row in self.curations)), (6, Counter(drop=6)))

    def test_complete_metadata_inventory_and_awkward_tranche(self):
        self.assertEqual(len(self.audits), 32)
        self.assertEqual(Counter(row["decision"] for row in self.audits), Counter(keep=26, drop=6))
        self.assertEqual(Counter(row["review_class"] for row in self.audits), Counter({
            "whole_candidate_metadata_scan": 24,
            "metadata_drop": 5,
            "awkward_tranche_keep": 2,
            "awkward_tranche_drop": 1,
        }))
        scan = self.report["whole_candidate_metadata_scan"]
        self.assertEqual(scan["rows_scanned"], 524)
        self.assertEqual(scan["legitimate_owner_resource_recommendations_kept"], 24)
        self.assertEqual(scan["nonowner_metadata_rows_dropped"], 5)

    def test_owner_resource_answers_and_all_requested_urls_survive(self):
        by_fact = {row["fact_id"]: row for row in self.rows}
        self.assertEqual(set(b.OWNER_FACT_IDS), {fact_id for fact_id in by_fact if by_fact[fact_id].get("source") == b.OWNER_SOURCE})
        resources = json.loads(b.RESOURCE_MANIFEST.read_text())["resources"]
        urls = {url for resource in resources for url in (resource["canonical_url"], resource.get("recommendation_url")) if url}
        blob = self.out.read_text()
        self.assertEqual((len(urls), sum(url in blob for url in urls)), (24, 24))

    def test_nonowner_metadata_recall_regressions_are_absent(self):
        for row in self.rows:
            if row.get("source") != b.OWNER_SOURCE:
                self.assertFalse(any(pattern.search(row["question"]) for pattern in b.REGRESSION_PATTERNS), row["question"])
        self.assertFalse(set(b.DROP_SPECS) & {row["fact_id"] for row in self.rows})

    def test_no_new_facts_from_pending_dense_corpora(self):
        self.assertEqual(self.report["isolated_build_projection"]["new_additions_applied"], 0)
        self.assertTrue(all(row["action"] == "drop" for row in self.curations))
        drops = {row["fact_id"]: row for row in self.audits if row["decision"] == "drop"}
        self.assertEqual({row["site_corpus_status_at_review"] for row in drops.values()}, {"pending", "in_progress"})

    def test_eval_duplicate_protocol_capacity_and_sealed_policy(self):
        self.assertEqual(json.loads(self.projection_report.read_text())["eval_fact_count"], 612)
        normalize = lambda text: " ".join(text.casefold().split())
        self.assertEqual({
            key: sum(value > 1 for value in Counter(normalize(row[key]) for row in self.rows).values())
            for key in ("fact_id", "question", "answer")
        }, {"fact_id": 0, "question": 0, "answer": 0})
        self.assertFalse(any(token in self.out.read_text() for token in ("<|im_start|>", "<|im_end|>", "</think>")))
        self.assertEqual(b.conservative_capacity(self.rows), b.EXPECTED_CAPACITY_AFTER)
        self.assertEqual(self.report["v52_isolation"], {
            "active_v52_inputs_modified": False,
            "candidate_status": "isolated_pending_not_promoted",
        })
        policy = self.report["sealed_evaluation_policy"]
        self.assertFalse(policy["manual_worker_opened_eval_heldout_ood_shadow_or_benchmark_content"])


if __name__ == "__main__":
    unittest.main()
