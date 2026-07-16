#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_context_merit_audit_v452 as b


class ContextMeritV452(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b.main()
        cls.tmp = tempfile.TemporaryDirectory(prefix=".test-v452-", dir=HERE)
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

    def test_exact_decisions_and_501_row_projection(self):
        self.assertEqual(
            (len(self.base_rows), b.file_sha256(self.base), len(self.rows), b.file_sha256(self.out)),
            (504, b.BASELINE_SHA256, 501, b.EXPECTED_OUTPUT_SHA256),
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
        self.assertEqual(Counter(row["action"] for row in self.curations), Counter(edit=5, drop=3))

    def test_audit_scope_decisions_and_classes_are_exact(self):
        self.assertEqual(len(self.audits), 10)
        self.assertEqual(Counter(row["decision"] for row in self.audits), Counter(edit=5, drop=3, keep=2))
        self.assertEqual(
            set(row["review_class"] for row in self.audits),
            {
                "vendor_additive_advice_drop",
                "product_listing_trivia_drop",
                "manufacturer_specific_drying_edit",
                "redundant_vendor_care_drop",
                "fiber_washing_scope_edit",
                "maintenance_exercise_scope_edit",
                "wash_preparation_keep",
                "accidental_wetting_scope_edit",
                "jute_machine_wash_keep",
                "wash_frequency_tradeoff_edit",
            },
        )
        self.assertTrue(all(row["material_scope"] for row in self.audits))

    def test_repairs_match_exact_qas_and_pinned_source_evidence(self):
        output = {row["fact_id"]: row for row in self.rows}
        audits = {row["fact_id"]: row for row in self.audits}
        for old_fact_id, spec in b.SPECS.items():
            source_path = b.ROOT / spec["source_document"]
            self.assertEqual(b.file_sha256(source_path), spec["source_document_file_sha256"])
            b.validate_source(
                spec,
                {**output.get(old_fact_id, {}), "fact_id": old_fact_id, "url": audits[old_fact_id]["url"]},
                audits[old_fact_id]["support_evidence"],
            )
            if spec["decision"] == "edit":
                new_fact_id = b.stable_fact_id(spec["edited_question"], spec["edited_answer"])
                self.assertEqual(
                    (output[new_fact_id]["question"], output[new_fact_id]["answer"]),
                    (spec["edited_question"], spec["edited_answer"]),
                )

    def test_dropped_rows_are_vendor_duplicates_or_product_trivia(self):
        dropped = {row["fact_id"]: row for row in self.audits if row["decision"] == "drop"}
        self.assertEqual(set(dropped), {
            "fact-f2eaf9d6f97b647d504c",
            "fact-ead3a4a26fcd40415f3e",
            "fact-1505e15e61043ad0d5cd",
        })
        self.assertEqual(
            {row["review_class"] for row in dropped.values()},
            {"vendor_additive_advice_drop", "product_listing_trivia_drop", "redundant_vendor_care_drop"},
        )
        output_ids = {row["fact_id"] for row in self.rows}
        self.assertTrue(set(b.OWNER_SURVIVORS) <= output_ids)

    def test_owner_resource_survivors_and_all_requested_urls(self):
        output = {row["fact_id"]: row for row in self.rows}
        self.assertEqual(self.report["surviving_owner_resource_facts"], {
            "surviving_facts": sorted(b.OWNER_SURVIVORS),
            "urls": sorted(b.OWNER_SURVIVORS.values()),
        })
        for fact_id, expected_url in b.OWNER_SURVIVORS.items():
            self.assertEqual(output[fact_id]["url"], expected_url)
        self.assertEqual(
            set(b.OWNER_FACT_IDS),
            {fact_id for fact_id in output if output[fact_id].get("source") == b.OWNER_SOURCE},
        )
        resources = json.loads(b.RESOURCE_MANIFEST.read_text())["resources"]
        urls = {
            url for resource in resources
            for url in (resource["canonical_url"], resource.get("recommendation_url")) if url
        }
        blob = self.out.read_text()
        self.assertEqual((len(urls), sum(url in blob for url in urls)), (24, 24))

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
            "documents": 8,
            "paths": sorted({spec["source_document"] for spec in b.SPECS.values()}),
            "reviewed_rows": 10,
            "total_unique_characters": 35878,
        })
        self.assertEqual(self.report["v52_isolation"], {
            "active_v52_inputs_modified": False,
            "candidate_status": "isolated_pending_not_promoted",
        })


if __name__ == "__main__":
    unittest.main()
