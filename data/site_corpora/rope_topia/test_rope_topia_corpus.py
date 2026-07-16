#!/usr/bin/env python3
import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_rope_topia_corpus as b


class RopeTopiaCorpusTest(unittest.TestCase):
    def setUp(self):
        b.build()
        self.manifest = json.loads(b.MANIFEST.read_text())
        self.evidence = json.loads(b.EVIDENCE.read_text())
        self.report = json.loads(b.REPORT.read_text())
        self.markdown = b.MARKDOWN.read_text()

    def test_complete_inventory_and_explicit_dispositions(self):
        inventory = self.manifest["inventory"]
        self.assertEqual(len(inventory), 15)
        self.assertEqual({row["resource_id"] for row in inventory}, set(b.ARCHIVES))
        self.assertEqual(
            {status: sum(row["corpus_status"] == status for row in inventory) for status in ("included", "excluded")},
            {"included": 8, "excluded": 7},
        )
        self.assertTrue(all(row["reason"] for row in inventory))
        self.assertEqual(sum("archive" in row for row in inventory), 14)

    def test_archive_and_successor_provenance_is_hash_pinned(self):
        self.assertEqual(len(self.evidence["archive_pages"]), 8)
        self.assertEqual(len(self.evidence["successor_pages"]), 3)
        for row in self.evidence["archive_pages"]:
            self.assertEqual(b.text_sha256(row["exact_qa_evidence"]), row["exact_qa_evidence_sha256"])
            self.assertEqual(len(row["archive"]["html_sha256"]), 64)
            self.assertEqual(len(row["archive"]["extracted_body_sha256"]), 64)
            self.assertTrue(row["archive"]["capture_timestamp"].isdigit())
        for row in self.evidence["successor_pages"]:
            source = json.loads((b.ROOT / row["path"]).read_text())
            self.assertEqual(b.text_sha256(source["text"]), row["text_sha256"])
            self.assertEqual(b.file_sha256(b.ROOT / row["path"]), row["file_sha256"])

    def test_markdown_is_dense_paraphrase_with_source_per_section(self):
        self.assertIn("Retrieved and manually reviewed: 2026-07-16", self.markdown)
        for row in self.manifest["inventory"]:
            if row["corpus_status"] == "included":
                self.assertIn(row["canonical_url"], self.markdown)
        for row in self.evidence["archive_pages"]:
            self.assertNotIn(row["exact_qa_evidence"], self.markdown)
        required = {
            "personal space", "position, pressure, and duration", "several references",
            "purpose-built emergency cutting tool", "recurring pattern", "Ipponnawa",
            "mindfulness", "experienced community members",
        }
        self.assertTrue(all(term in self.markdown for term in required))

    def test_report_and_build_are_deterministic(self):
        first = tuple(b.file_sha256(path) for path in (b.MARKDOWN, b.EVIDENCE, b.MANIFEST, b.REPORT))
        b.build()
        second = tuple(b.file_sha256(path) for path in (b.MARKDOWN, b.EVIDENCE, b.MANIFEST, b.REPORT))
        self.assertEqual(first, second)
        self.assertEqual(self.report["coverage"], {
            "archive_captures_found": 14,
            "indexed_pages": 15,
            "pages_excluded_with_reason": 7,
            "substantive_pages_included": 8,
            "verified_successor_bodies": 3,
        })
        self.assertFalse(self.report["sealed_evaluation_policy"]["eval_or_heldout_opened"])


if __name__ == "__main__":
    unittest.main()
