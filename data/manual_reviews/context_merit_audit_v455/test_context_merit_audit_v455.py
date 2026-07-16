#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_context_merit_audit_v455 as b


class ContextMeritV455(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b.main()
        cls.tmp = tempfile.TemporaryDirectory(prefix=".test-v455-", dir=HERE)
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

    def test_exact_decisions_and_497_row_projection(self):
        self.assertEqual(
            (len(self.base_rows), b.file_sha256(self.base), len(self.rows), b.file_sha256(self.out)),
            (499, b.BASELINE_SHA256, 497, b.EXPECTED_OUTPUT_SHA256),
        )
        self.assertEqual((self.datasets[0], self.reports[0]), (self.datasets[1], self.reports[1]))
        base_ids = {row["fact_id"] for row in self.base_rows}
        output_ids = {row["fact_id"] for row in self.rows}
        dropped = {fact_id for fact_id, spec in b.SPECS.items() if spec["decision"] == "drop"}
        kept = {fact_id for fact_id, spec in b.SPECS.items() if spec["decision"] == "keep"}
        self.assertEqual(base_ids - output_ids, dropped)
        self.assertFalse(output_ids - base_ids)
        self.assertTrue(kept <= output_ids)
        self.assertEqual(Counter(row["action"] for row in self.curations), Counter(drop=2))

    def test_audit_scope_decisions_and_classes_are_exact(self):
        self.assertEqual(len(self.audits), 6)
        self.assertEqual(Counter(row["decision"] for row in self.audits), Counter(keep=4, drop=2))
        self.assertEqual(
            set(row["review_class"] for row in self.audits),
            {
                "exploding_knot_behavior_keep",
                "bend_definition_tradeoff_keep",
                "generic_curriculum_goal_drop",
                "slip_knot_tradeoff_keep",
                "handcuff_knot_risk_keep",
                "minor_marking_recall_drop",
            },
        )
        self.assertTrue(all(row["knot_scope"] for row in self.audits))
        self.assertEqual({row["source"] for row in self.audits}, {"rope365"})

    def test_all_rows_match_full_pinned_source(self):
        source = json.loads((b.ROOT / b.SOURCE_DOCUMENT).read_text())
        self.assertEqual(b.file_sha256(b.ROOT / b.SOURCE_DOCUMENT), b.SOURCE_FILE_SHA256)
        self.assertEqual((source["url"], len(source["text"]), b.text_sha256(source["text"])),
                         (b.SOURCE_URL, b.SOURCE_CHARS, b.SOURCE_TEXT_SHA256))
        for audit in self.audits:
            self.assertIn(audit["support_evidence"], source["text"])
            self.assertEqual(audit["document_sha256"], b.SOURCE_TEXT_SHA256)
            self.assertEqual(audit["source_document_file_sha256"], b.SOURCE_FILE_SHA256)

    def test_only_generic_or_redundant_rows_are_dropped(self):
        output_ids = {row["fact_id"] for row in self.rows}
        dropped = {row["fact_id"]: row for row in self.audits if row["decision"] == "drop"}
        self.assertEqual(set(dropped), {
            "fact-f85dd797b667780b5fca",
            "fact-6a6f49157591a8d29ea9",
        })
        self.assertEqual(
            {row["review_class"] for row in dropped.values()},
            {"generic_curriculum_goal_drop", "minor_marking_recall_drop"},
        )
        for row in dropped.values():
            self.assertTrue(set(row["substantive_survivors"]) <= output_ids)

    def test_all_requested_resources_and_owner_facts_survive(self):
        urls = b.requested_urls()
        blob = self.out.read_text()
        self.assertEqual((len(urls), sum(url in blob for url in urls)), (24, 24))
        self.assertEqual(self.report["requested_resource_coverage"], {
            "manifest_urls": 24,
            "manifest_urls_present": 24,
            "owner_resource_facts": 24,
        })
        self.assertEqual(
            set(b.OWNER_FACT_IDS),
            {row["fact_id"] for row in self.rows if row.get("source") == b.OWNER_SOURCE},
        )

    def test_duplicates_protocol_capacity_and_source_boundary(self):
        self.assertEqual(json.loads(self.projection_report.read_text())["eval_fact_count"], 612)
        normalize = lambda text: " ".join(text.casefold().split())
        self.assertEqual({
            key: sum(value > 1 for value in Counter(normalize(row[key]) for row in self.rows).values())
            for key in ("fact_id", "question", "answer")
        }, {"fact_id": 0, "question": 0, "answer": 0})
        self.assertFalse(any(token in self.out.read_text() for token in ("<|im_start|>", "<|im_end|>", "</think>")))
        self.assertEqual(b.conservative_capacity(self.base_rows), b.EXPECTED_CAPACITY_BEFORE)
        self.assertEqual(b.conservative_capacity(self.rows), b.EXPECTED_CAPACITY_AFTER)
        self.assertEqual(self.report["conservative_capacity"]["delta"], {
            "conflict_units": 0,
            "equipment_material": 0,
            "resources_general": 0,
            "safety_consent": 1,
            "technique": -1,
        })
        self.assertEqual(self.report["source_boundary"], {
            "new_or_pending_corpus_inputs": 0,
            "protected_eval_heldout_holdout_ood_shadow_probe_rows_opened": 0,
            "training_or_eval_artifacts_modified": 0,
        })
        self.assertEqual(self.report["isolated_build_projection"]["new_additions_applied"], 0)
        self.assertFalse(self.report["sealed_evaluation_policy"]["manual_worker_opened_eval_heldout_ood_shadow_or_benchmark_content"])

    def test_two_layer_contract_source_inventory_and_v52(self):
        contract = self.report["training_layer_contract"]
        self.assertIn("Distinct first-class", contract["cleaned_markdown_site_corpora"])
        self.assertIn("Distinct first-class", contract["derived_qa"])
        self.assertIn("Do not collapse", contract["deduplication_policy"])
        self.assertEqual(self.report["source_snapshot_inventory"], {
            "documents": 1,
            "paths": [b.SOURCE_DOCUMENT],
            "reviewed_rows": 6,
            "total_unique_characters": b.SOURCE_CHARS,
        })
        self.assertEqual(self.report["v52_isolation"], {
            "active_v52_inputs_modified": False,
            "candidate_status": "isolated_pending_not_promoted",
        })


if __name__ == "__main__":
    unittest.main()
