#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUILDER = HERE / "build_rope365_corpus.py"
GUIDE_SOURCE = HERE / "guide_source.md"
SNAPSHOT = HERE / "source_snapshot.json"
MARKDOWN = HERE / "rope365.md"
MANIFEST = HERE / "manifest.json"
REPORT = HERE / "report.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Rope365CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.markdown = MARKDOWN.read_text(encoding="utf-8")

    def test_canonical_inventory_is_complete_and_decided(self) -> None:
        pages = self.snapshot["pages"]
        self.assertEqual(232, len(pages))
        self.assertEqual(232, len({page["url"] for page in pages}))
        self.assertEqual({"included", "excluded"}, {page["decision"] for page in pages})
        self.assertTrue(all(page["reason"].strip() for page in pages))
        by_kind = {kind: sum(page["sitemap_kind"] == kind for page in pages) for kind in ("page", "post", "category")}
        self.assertEqual({"page": 197, "post": 33, "category": 2}, by_kind)

    def test_coverage_counts(self) -> None:
        pages = self.snapshot["pages"]
        self.assertEqual(111, sum(page["decision"] == "included" for page in pages))
        self.assertEqual(121, sum(page["decision"] == "excluded" for page in pages))
        self.assertEqual(346, self.snapshot["raw_record_count"])
        self.assertEqual(211, self.snapshot["unique_raw_url_count"])
        self.assertEqual(111, self.report["summary"]["included_pages"])

    def test_every_included_page_has_a_page_level_source_url(self) -> None:
        included = {page["url"] for page in self.snapshot["pages"] if page["decision"] == "included"}
        cited = set(re.findall(r"^Source: (https://rope365\.com/[^\s]+)$", self.markdown, flags=re.MULTILINE))
        self.assertEqual(included, cited)

    def test_direct_training_artifact_is_dense_and_clean(self) -> None:
        self.assertEqual("canonical_trainable_source_corpus", self.manifest["artifact_role"])
        self.assertEqual("canonical_trainable_source_corpus", self.report["artifact_role"])
        words = re.findall(r"\b[\w’'-]+\b", self.markdown, flags=re.UNICODE)
        self.assertGreaterEqual(len(words), 12_000)
        self.assertGreaterEqual(len(re.findall(r"^#{1,3} ", self.markdown, flags=re.MULTILINE)), 70)
        self.assertNotIn("Leave a Comment", self.markdown)
        self.assertNotIn("Subscribe to the mailing list", self.markdown)
        self.assertNotIn("wp-content", self.markdown)
        self.assertNotIn("$", self.markdown)

    def test_important_domains_and_limitations_are_explicit(self) -> None:
        required = (
            "Nerve injury", "Somerville", "crossing hitch", "box-tie", "Rope bondage can cause",
            "Hojōjutsu", "Itō Seiu", "Osada Eikichi", "Freestanding frames", "upline system design",
            "ceiling joists", "No complete suspension", "Media-only", "not medical advice",
        )
        for phrase in required:
            self.assertIn(phrase, self.markdown)

    def test_manifest_hashes_and_metrics(self) -> None:
        artifact = self.manifest["artifact"]
        self.assertEqual(sha256(MARKDOWN), artifact["sha256"])
        self.assertEqual(MARKDOWN.stat().st_size, artifact["bytes"])
        self.assertEqual(sha256(SNAPSHOT), self.manifest["build_inputs"]["source_snapshot_sha256"])
        self.assertEqual(artifact["sha256"], self.report["artifact_sha256"])

    def test_raw_snapshot_provenance_hashes(self) -> None:
        checked = 0
        for page in self.snapshot["pages"]:
            relative = page["raw_snapshot"]
            if relative is None:
                continue
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            record = json.loads(path.read_text(encoding="utf-8"))
            digest = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
            self.assertEqual(page["raw_text_sha256"], digest, page["url"])
            checked += 1
        self.assertEqual(211, checked)

    def test_clean_room_report(self) -> None:
        access = self.report["clean_room_access"]
        self.assertTrue(access["clean_zero_forbidden_path_access"])
        self.assertEqual(0, access["forbidden_path_access_count"])
        self.assertEqual([], access["forbidden_path_access"])
        self.assertFalse(access["broad_data_directory_search_performed"])
        self.assertFalse(access["existing_qa_training_or_manual_review_artifacts_read"])
        self.assertFalse(self.report["artifact_contract"]["qa_dataset_modified"])

    def test_default_build_is_deterministic(self) -> None:
        before = {path.name: sha256(path) for path in (MARKDOWN, MANIFEST, REPORT)}
        subprocess.run([sys.executable, str(BUILDER)], cwd=ROOT, check=True)
        after = {path.name: sha256(path) for path in (MARKDOWN, MANIFEST, REPORT)}
        self.assertEqual(before, after)
        self.assertEqual(GUIDE_SOURCE.read_bytes(), MARKDOWN.read_bytes())


if __name__ == "__main__":
    unittest.main()
