#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_context_merit_audit_v453 as b


class ContextMeritV453(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b.main()
        cls.tmp = tempfile.TemporaryDirectory(prefix=".test-v453-", dir=HERE)
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

    def test_exact_decisions_and_499_row_projection(self):
        self.assertEqual(
            (len(self.base_rows), b.file_sha256(self.base), len(self.rows), b.file_sha256(self.out)),
            (501, b.BASELINE_SHA256, 499, b.EXPECTED_OUTPUT_SHA256),
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
        self.assertEqual(Counter(row["action"] for row in self.curations), Counter(edit=2, drop=2))

    def test_audit_scope_decisions_and_classes_are_exact(self):
        self.assertEqual(len(self.audits), 6)
        self.assertEqual(Counter(row["decision"] for row in self.audits), Counter(edit=2, drop=2, keep=2))
        self.assertEqual(
            set(row["review_class"] for row in self.audits),
            {
                "coined_label_recall_drop",
                "historical_interpretation_edit",
                "story_term_recall_drop",
                "pseudonym_function_attribution_edit",
                "archival_uncertainty_keep",
                "translation_limitation_keep",
            },
        )
        self.assertTrue(all(row["history_scope"] for row in self.audits))
        self.assertEqual({row["source"] for row in self.audits}, {"kinbakutoday"})

    def test_repairs_match_exact_qas_and_full_pinned_source(self):
        output = {row["fact_id"]: row for row in self.rows}
        source = json.loads((b.ROOT / b.SOURCE_DOCUMENT).read_text())
        self.assertEqual(b.file_sha256(b.ROOT / b.SOURCE_DOCUMENT), b.SOURCE_FILE_SHA256)
        self.assertEqual((source["url"], len(source["text"]), b.text_sha256(source["text"])),
                         (b.SOURCE_URL, b.SOURCE_CHARS, b.SOURCE_TEXT_SHA256))
        for audit in self.audits:
            self.assertIn(audit["support_evidence"], source["text"])
            self.assertEqual(audit["document_sha256"], b.SOURCE_TEXT_SHA256)
        for spec in b.SPECS.values():
            if spec["decision"] == "edit":
                new_fact_id = b.stable_fact_id(spec["edited_question"], spec["edited_answer"])
                self.assertEqual(
                    (output[new_fact_id]["question"], output[new_fact_id]["answer"]),
                    (spec["edited_question"], spec["edited_answer"]),
                )

    def test_label_only_drops_have_substantive_survivors(self):
        output_ids = {row["fact_id"] for row in self.rows}
        dropped = {row["fact_id"]: row for row in self.audits if row["decision"] == "drop"}
        self.assertEqual(set(dropped), {
            "fact-84fe512655ff109a657e",
            "fact-ed15766580bb03fdee1b",
        })
        self.assertEqual(
            {row["review_class"] for row in dropped.values()},
            {"coined_label_recall_drop", "story_term_recall_drop"},
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
