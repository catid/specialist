#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_context_merit_audit_v447 as b


class ContextMeritV447(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b.main()
        cls.tmp = tempfile.TemporaryDirectory(prefix=".test-v447-", dir=HERE)
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

    def test_exact_decisions_and_513_row_projection(self):
        self.assertEqual(
            (len(self.base_rows), b.file_sha256(self.base), len(self.rows), b.file_sha256(self.out)),
            (515, b.BASELINE_SHA256, 513, b.EXPECTED_OUTPUT_SHA256),
        )
        self.assertEqual((self.datasets[0], self.reports[0]), (self.datasets[1], self.reports[1]))
        base_ids = {row["fact_id"] for row in self.base_rows}
        output_ids = {row["fact_id"] for row in self.rows}
        edited = {fact_id for fact_id, spec in b.SPECS.items() if spec["decision"] == "edit"}
        dropped = {fact_id for fact_id, spec in b.SPECS.items() if spec["decision"] == "drop"}
        kept = {fact_id for fact_id, spec in b.SPECS.items() if spec["decision"] == "keep"}
        expected_new = {
            b.stable_fact_id(spec["edited_question"], spec["edited_answer"])
            for spec in b.SPECS.values() if spec["decision"] == "edit"
        }
        self.assertEqual(base_ids - output_ids, edited | dropped)
        self.assertEqual(output_ids - base_ids, expected_new)
        self.assertTrue(kept <= output_ids)
        self.assertEqual(Counter(row["action"] for row in self.curations), Counter(edit=5, drop=2))

    def test_audit_scope_and_review_classes_are_exact(self):
        self.assertEqual(len(self.audits), 10)
        self.assertEqual(Counter(row["decision"] for row in self.audits), Counter(edit=5, keep=3, drop=2))
        self.assertEqual(Counter(row["review_class"] for row in self.audits), Counter({
            "complete_operational_keep": 3,
            "pattern_architecture_edit": 2,
            "knot_function_tradeoff_edit": 2,
            "pattern_handling_edit": 1,
            "unsupported_visual_step_drop": 1,
            "label_only_redundant_drop": 1,
        }))
        self.assertTrue(all(row["source"] == "rope365" for row in self.audits))

    def test_repairs_match_exact_qas_and_full_source_snapshots(self):
        output = {row["fact_id"]: row for row in self.rows}
        audits = {row["fact_id"]: row for row in self.audits}
        for old_fact_id, spec in b.SPECS.items():
            source_path = b.ROOT / spec["source_document"]
            source = json.loads(source_path.read_text())
            self.assertEqual(b.file_sha256(source_path), spec["source_document_file_sha256"])
            self.assertEqual(len(source["text"]), spec["source_document_chars"])
            self.assertIn(audits[old_fact_id]["support_evidence"], source["text"])
            self.assertEqual(source["url"], audits[old_fact_id]["url"])
            if spec["decision"] == "edit":
                new_fact_id = b.stable_fact_id(spec["edited_question"], spec["edited_answer"])
                self.assertEqual(
                    (output[new_fact_id]["question"], output[new_fact_id]["answer"]),
                    (spec["edited_question"], spec["edited_answer"]),
                )

    def test_low_merit_drops_have_pinned_operational_survivors(self):
        output = {row["fact_id"]: row for row in self.rows}
        self.assertEqual(self.report["redundancy_quarantine"]["dropped_to_surviving_fact"], {
            "fact-f6faf5703b157e22121d": "fact-4d9cba14a6eb00ad7e1d",
            "fact-42526efd7c67bf9799a3": "fact-b381baeee64c518a057b",
        })
        for fact_id, expected in b.REDUNDANT_SURVIVORS.items():
            self.assertEqual((output[fact_id]["question"], output[fact_id]["answer"]), (expected["question"], expected["answer"]))

    def test_owner_resources_and_requested_urls_survive(self):
        by_fact = {row["fact_id"]: row for row in self.rows}
        self.assertEqual(
            set(b.OWNER_FACT_IDS),
            {fact_id for fact_id in by_fact if by_fact[fact_id].get("source") == b.OWNER_SOURCE},
        )
        resources = json.loads(b.RESOURCE_MANIFEST.read_text())["resources"]
        urls = {
            url for resource in resources
            for url in (resource["canonical_url"], resource.get("recommendation_url")) if url
        }
        blob = self.out.read_text()
        self.assertEqual((len(urls), sum(url in blob for url in urls)), (24, 24))

    def test_duplicates_protocol_capacity_and_policy(self):
        self.assertEqual(json.loads(self.projection_report.read_text())["eval_fact_count"], 612)
        normalize = lambda text: " ".join(text.casefold().split())
        self.assertEqual({
            key: sum(value > 1 for value in Counter(normalize(row[key]) for row in self.rows).values())
            for key in ("fact_id", "question", "answer")
        }, {"fact_id": 0, "question": 0, "answer": 0})
        self.assertFalse(any(token in self.out.read_text() for token in ("<|im_start|>", "<|im_end|>", "</think>")))
        self.assertEqual(b.conservative_capacity(self.base_rows), b.EXPECTED_CAPACITY_BEFORE)
        self.assertEqual(b.conservative_capacity(self.rows), b.EXPECTED_CAPACITY_AFTER)
        self.assertEqual(self.report["isolated_build_projection"]["new_additions_applied"], 0)
        self.assertFalse(self.report["sealed_evaluation_policy"]["manual_worker_opened_eval_heldout_ood_shadow_or_benchmark_content"])

    def test_two_layer_contract_source_inventory_and_v52(self):
        contract = self.report["training_layer_contract"]
        self.assertIn("Distinct first-class", contract["cleaned_markdown_site_corpora"])
        self.assertIn("Distinct first-class", contract["derived_qa"])
        self.assertIn("Do not collapse", contract["deduplication_policy"])
        self.assertEqual(self.report["source_snapshot_inventory"], {
            "documents": 8,
            "paths": sorted({spec["source_document"] for spec in b.SPECS.values()}),
            "reviewed_rows": 10,
            "total_unique_characters": 24956,
        })
        self.assertEqual(self.report["v52_isolation"], {
            "active_v52_inputs_modified": False,
            "candidate_status": "isolated_pending_not_promoted",
        })


if __name__ == "__main__":
    unittest.main()
