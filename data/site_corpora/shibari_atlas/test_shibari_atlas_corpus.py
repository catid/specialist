#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path
from urllib.parse import urlsplit


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MARKDOWN = HERE / "shibari_atlas.md"
MANIFEST = HERE / "manifest.json"
REPORT = HERE / "report.json"
SNAPSHOT = HERE / "curated_records.json"
BUILDER = ROOT / "build_shibari_atlas_corpus.py"


class ShibariAtlasCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.markdown = MARKDOWN.read_text(encoding="utf-8")
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

    def test_canonical_inventory_is_complete_and_decided(self) -> None:
        inventory = self.manifest["canonical_inventory"]["urls"]
        self.assertEqual(len(inventory), 2078)
        self.assertEqual(len({item["url"] for item in inventory}), len(inventory))
        self.assertEqual(
            self.manifest["canonical_inventory"]["decision_counts"],
            {"exclude": 364, "include": 1714},
        )
        for item in inventory:
            self.assertEqual(urlsplit(item["url"]).netloc, "shibariatlas.org")
            self.assertIn(item["decision"], {"include", "exclude"})
            self.assertTrue(item["reason"])

    def test_every_included_page_has_explicit_markdown_attribution(self) -> None:
        included = [
            item["url"]
            for item in self.manifest["canonical_inventory"]["urls"]
            if item["decision"] == "include"
        ]
        for url in included:
            self.assertIn(f"Source URL: {url}", self.markdown)

    def test_access_and_category_counts(self) -> None:
        coverage = self.report["coverage"]
        self.assertEqual(coverage["gated_urls"], 0)
        self.assertEqual(coverage["media_only_urls"], 0)
        self.assertEqual(
            coverage["included_by_category"],
            {"core": 4, "event_occurrence": 222, "lineage_profile": 1421, "source_catalogue": 67},
        )
        self.assertEqual(self.report["errors"], [])

    def test_source_catalogue_is_machine_provenance_not_trainable_flood(self) -> None:
        rows = [row for record in self.snapshot["records"] for row in record.get("source_rows", [])]
        self.assertEqual(len(rows), 4794)
        self.assertTrue(all(row["title"] and row["url"] and row["kind"] for row in rows))
        counts = self.report["content"]["word_counts"]
        self.assertLess(counts["markdown_provenance_appendix"], counts["markdown_core_before_provenance_appendix"] // 100)
        self.assertIn("retained as machine-readable provenance", self.markdown)

    def test_research_and_fact_structures_are_preserved(self) -> None:
        required = (
            "##### What is bakushi?",
            "##### What is kinbaku?",
            "##### What is hojōjutsu?",
            "##### Strong",
            "##### Medium",
            "##### Weak",
            "##### Deshi",
            "##### Certification",
            "### Safety / Consent / Rope-Bottom Pedagogy",
            "#### Programme",
            "| Category | Count | Share |",
        )
        for marker in required:
            self.assertIn(marker, self.markdown)

    def test_navigation_and_search_boilerplate_is_absent(self) -> None:
        forbidden = (
            "Skip to main content",
            "Type at least two characters",
            "Enter opens the first result",
            "Search index did not load",
            "Retry interactive graph",
            "Search, filters and source details require JavaScript",
            "No sources match",
            "Clear search and filters",
            "Suggest correction on the relevant entry",
            "Collapse all",
        )
        for phrase in forbidden:
            self.assertNotIn(phrase.casefold(), self.markdown.casefold())

    def test_no_known_mechanical_compression_damage(self) -> None:
        malformed = (
            r"\bdata; interface\b",
            r"\bpeople; kinbaku\b",
            r",;|;;|\.\s*;",
            r"\bpresent-dayly\b",
            r"\bclinitial\b",
            r"\ba subsequently correction\b",
            r"\bframes the atlas as\b",
        )
        for pattern in malformed:
            self.assertIsNone(re.search(pattern, self.markdown, flags=re.IGNORECASE), pattern)

    def test_content_hashes_and_deterministic_offline_build(self) -> None:
        markdown_hash = hashlib.sha256(MARKDOWN.read_bytes()).hexdigest()
        snapshot_hash = hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()
        self.assertEqual(markdown_hash, self.manifest["corpus"]["sha256"])
        self.assertEqual(snapshot_hash, self.manifest["machine_readable_provenance"]["sha256"])
        result = subprocess.run(
            [sys.executable, str(BUILDER), "--check"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_scope_did_not_touch_eval_or_active_training_inputs(self) -> None:
        review = self.snapshot["review"]
        self.assertFalse(review["sealed_or_holdout_data_accessed"])
        self.assertFalse(review["external_pages_copied"])
        self.assertFalse(review["verbatim_site_mirror"])


if __name__ == "__main__":
    unittest.main()
