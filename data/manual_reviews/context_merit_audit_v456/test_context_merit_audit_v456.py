#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_context_merit_audit_v456 as b


class ContextMeritV456(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b.main()
        cls.tmp = tempfile.TemporaryDirectory(prefix=".test-v456-", dir=HERE)
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

    def test_exact_decisions_and_projection_identity(self):
        self.assertEqual(
            (
                len(self.base_rows),
                b.file_sha256(self.base),
                len(self.rows),
                b.file_sha256(self.out),
            ),
            (
                b.BASELINE_ROWS,
                b.BASELINE_SHA256,
                b.EXPECTED_OUTPUT_ROWS,
                b.EXPECTED_OUTPUT_SHA256,
            ),
        )
        self.assertEqual(
            (self.datasets[0], self.reports[0]),
            (self.datasets[1], self.reports[1]),
        )
        self.assertEqual(
            Counter(row["decision"] for row in self.audits),
            Counter(drop=4, edit=2, keep=2),
        )
        self.assertEqual(
            Counter(row["action"] for row in self.curations),
            Counter(drop=4, edit=2),
        )

    def test_reviewed_fact_replacement_and_drop_sets_are_exact(self):
        base_ids = {row["fact_id"] for row in self.base_rows}
        output_ids = {row["fact_id"] for row in self.rows}
        dropped = {
            fact_id
            for fact_id, spec in b.SPECS.items()
            if spec["decision"] in {"drop", "edit"}
        }
        expected_new = {
            b.stable_fact_id(spec["edited_question"], spec["edited_answer"])
            for spec in b.SPECS.values()
            if spec["decision"] == "edit"
        }
        kept = {
            fact_id
            for fact_id, spec in b.SPECS.items()
            if spec["decision"] == "keep"
        }
        self.assertEqual(base_ids - output_ids, dropped)
        self.assertEqual(output_ids - base_ids, expected_new)
        self.assertEqual(
            kept,
            {"fact-cc58bd014cc6acfaaf2b", "fact-cf160d03f19be3d053ab"},
        )
        self.assertTrue(kept <= output_ids)

    def test_full_pinned_sources_and_manual_support_are_exact(self):
        self.assertEqual({row["source"] for row in self.audits}, {"wykd"})
        self.assertEqual(len(self.audits), 8)
        for text_hash, source in b.SOURCES.items():
            path = b.ROOT / source["path"]
            document = json.loads(path.read_text())
            self.assertEqual(b.file_sha256(path), source["file_sha256"])
            self.assertEqual(document["url"], source["url"])
            self.assertEqual(len(document["text"]), source["chars"])
            self.assertEqual(b.text_sha256(document["text"]), text_hash)
        for audit in self.audits:
            source = b.SOURCES[audit["document_sha256"]]
            document = json.loads((b.ROOT / source["path"]).read_text())
            self.assertIn(audit["support_evidence"], document["text"])
            self.assertEqual(audit["source_document_file_sha256"], source["file_sha256"])
            self.assertEqual(audit["source_document_text_sha256"], audit["document_sha256"])
            self.assertEqual(audit["support_evidence_sha256"], b.text_sha256(audit["support_evidence"]))

    def test_edits_are_natural_complete_and_source_bounded(self):
        output = {row["fact_id"]: row for row in self.rows}
        for spec in b.SPECS.values():
            if spec["decision"] != "edit":
                continue
            fact_id = b.stable_fact_id(spec["edited_question"], spec["edited_answer"])
            row = output[fact_id]
            self.assertEqual(
                (row["question"], row["answer"]),
                (spec["edited_question"], spec["edited_answer"]),
            )
            self.assertNotIn("Does WykD claim", row["question"])
            self.assertGreaterEqual(len(row["answer"].split()), 15)
            self.assertEqual(row["source"], "wykd")
            self.assertIn(row["evidence"], json.loads(
                (b.ROOT / b.SOURCES[row["document_sha256"]]["path"]).read_text()
            )["text"])

    def test_all_requested_resources_and_owner_facts_survive(self):
        urls = b.requested_urls()
        blob = self.out.read_text()
        self.assertEqual((len(urls), sum(url in blob for url in urls)), (24, 24))
        self.assertEqual(
            self.report["requested_resource_coverage"],
            {
                "manifest_urls": 24,
                "manifest_urls_present": 24,
                "owner_resource_facts": 24,
            },
        )
        self.assertEqual(
            set(b.OWNER_FACT_IDS),
            {
                row["fact_id"]
                for row in self.rows
                if row.get("source") == b.OWNER_SOURCE
            },
        )

    def test_provenance_duplicates_protocol_and_capacity(self):
        self.assertEqual(json.loads(self.projection_report.read_text())["eval_fact_count"], 612)
        normalize = lambda text: " ".join(text.casefold().split())
        self.assertEqual(
            {
                key: sum(
                    value > 1
                    for value in Counter(normalize(row[key]) for row in self.rows).values()
                )
                for key in ("fact_id", "question", "answer")
            },
            {"fact_id": 0, "question": 0, "answer": 0},
        )
        self.assertFalse(
            any(
                token in self.out.read_text()
                for token in ("<|im_start|>", "<|im_end|>", "</think>")
            )
        )
        self.assertTrue(
            all(
                row.get("source")
                and row.get("url")
                and row.get("document_sha256")
                and row.get("evidence")
                for row in self.rows
            )
        )
        self.assertEqual(b.conservative_capacity(self.base_rows), b.EXPECTED_CAPACITY_BEFORE)
        self.assertEqual(b.conservative_capacity(self.rows), b.EXPECTED_CAPACITY_AFTER)
        self.assertEqual(
            self.report["conservative_capacity"]["delta"],
            {
                key: b.EXPECTED_CAPACITY_AFTER[key] - b.EXPECTED_CAPACITY_BEFORE[key]
                for key in b.EXPECTED_CAPACITY_BEFORE
            },
        )

    def test_report_slice_source_boundary_and_layer_contract(self):
        self.assertEqual(
            self.report["baseline_checkpoint"],
            {
                "commit": "edbc5805160ab607012b6d3a1b24f1443b3c999f",
                "rows": 497,
                "sha256": b.BASELINE_SHA256,
                "version": "v455",
            },
        )
        self.assertEqual(
            self.report["source_snapshot_inventory"],
            {
                "documents": 3,
                "paths": sorted(source["path"] for source in b.SOURCES.values()),
                "reviewed_rows": 8,
                "total_unique_characters": 9655,
            },
        )
        self.assertEqual(
            self.report["source_boundary"],
            {
                "new_or_pending_corpus_inputs": 0,
                "protected_eval_heldout_holdout_ood_shadow_probe_rows_opened": 0,
                "raw_markdown_or_snapshot_documents_modified": 0,
                "scout_files_opened_or_modified": 0,
                "training_or_eval_artifacts_modified": 0,
            },
        )
        contract = self.report["training_layer_contract"]
        self.assertIn("Distinct first-class", contract["cleaned_markdown_site_corpora"])
        self.assertIn("Distinct first-class", contract["derived_qa"])
        self.assertIn("Do not collapse", contract["deduplication_policy"])
        self.assertIn("source URL", contract["provenance_requirement"])
        self.assertFalse(
            self.report["sealed_evaluation_policy"][
                "manual_worker_opened_eval_heldout_ood_shadow_or_benchmark_content"
            ]
        )
        self.assertEqual(
            self.report["v52_isolation"],
            {
                "active_v52_inputs_modified": False,
                "candidate_status": "isolated_pending_not_promoted",
            },
        )


if __name__ == "__main__":
    unittest.main()
