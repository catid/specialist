from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
NS = "http://www.sitemaps.org/schemas/sitemap/0.9"
REASON = "terms_of_service_prohibits_content_collection_and_storage"
OUTPUTS = ("CORPUS.md", "REPORT.md", "content_records.jsonl", "inventory.jsonl", "manifest.json", "url_dispositions.jsonl")


def json_file(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def jsonl(name: str):
    return [json.loads(line) for line in (ROOT / name).read_text(encoding="utf-8").splitlines() if line]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locs(path: Path) -> list[str]:
    root = ET.fromstring(path.read_bytes())
    return [node.text.strip() for node in root.findall(f".//{{{NS}}}loc") if node.text]


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json_file("manifest.json")
        cls.provenance = json_file("source_snapshot/provenance.json")
        cls.policy = json_file("policy_decision.json")
        cls.inventory = jsonl("inventory.jsonl")

    def test_policy_exclusion_and_zero_training_content(self) -> None:
        self.assertEqual((ROOT / "content_records.jsonl").read_bytes(), b"")
        self.assertEqual(self.manifest["policy_reason"], REASON)
        self.assertEqual(self.policy["decision_reason"], REASON)
        self.assertFalse(self.manifest["direct_training_ready"])
        self.assertEqual(self.manifest["content_record_count"], 0)
        self.assertEqual(self.manifest["canonical_instructional_body_snapshots_retained"], 0)
        self.assertFalse(self.policy["terms_audit"]["body_snapshot_retained"])

    def test_metadata_only_capture_and_snapshot_integrity(self) -> None:
        self.assertEqual(self.provenance["capture_scope"], "robots_and_sitemap_metadata_only")
        self.assertEqual(self.provenance["canonical_page_body_requests"], 0)
        for item in self.provenance["resources"]:
            path = ROOT / item["path"]
            self.assertEqual(path.stat().st_size, item["byte_length"])
            self.assertEqual(sha(path), item["sha256"])
            self.assertIn(item["role"], {"robots_policy", "sitemap_index", "child_sitemap"})

    def test_complete_sitemap_inventory_is_excluded(self) -> None:
        resources = {item["url"]: item for item in self.provenance["resources"]}
        children = set(locs(ROOT / "source_snapshot/sitemap-index.xml"))
        expected: dict[str, set[str]] = {}
        for url in children:
            for canonical in locs(ROOT / resources[url]["path"]):
                expected.setdefault(canonical, set()).add(url)
        actual = {item["url"]: set(item["discovered_in"]) for item in self.inventory}
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), self.manifest["inventory_record_count"])
        for item in self.inventory:
            self.assertEqual(item["disposition"], "excluded")
            self.assertEqual(item["reason"], REASON)
            self.assertFalse(item["content_retrieved_for_corpus"])
            self.assertFalse(item["direct_training_included"])

    def test_capture_utility_rejects_canonical_pages(self) -> None:
        spec = importlib.util.spec_from_file_location("capture_metadata", ROOT / "capture_metadata.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader
        spec.loader.exec_module(module)
        for url in ("https://shibaristudy.com/robots.txt", "https://shibaristudy.com/sitemap.xml"):
            module.validate_metadata_url(url)
        for url in ("https://shibaristudy.com/", "https://shibaristudy.com/programs/example", "https://shibaristudy.com/pages/terms-conditions", "https://example.com/sitemap.xml"):
            with self.assertRaises(ValueError):
                module.validate_metadata_url(url)

    def test_no_chat_ui_or_url_trivia_training_surface(self) -> None:
        surface = (ROOT / "CORPUS.md").read_text(encoding="utf-8") + (ROOT / "content_records.jsonl").read_text(encoding="utf-8")
        for marker in ("<|im_start|>", "<|im_end|>", "</think>", "Question:", "Answer:", "Sign in", "Add to cart", "<script"):
            self.assertNotIn(marker, surface)
        self.assertNotIn("Which canonical", surface)

    def test_manifest_hashes_and_document_split_rule(self) -> None:
        self.assertEqual(self.manifest["policy_decision"]["sha256"], sha(ROOT / "policy_decision.json"))
        for name, item in self.manifest["outputs"].items():
            self.assertEqual(item["sha256"], sha(ROOT / name))
            self.assertEqual(item["byte_length"], (ROOT / name).stat().st_size)
        self.assertIn("canonical source document", self.manifest["document_disjoint_requirement"])

    def test_offline_build_is_deterministic(self) -> None:
        before = {name: sha(ROOT / name) for name in OUTPUTS}
        subprocess.run([sys.executable, str(ROOT / "build.py")], cwd=ROOT, check=True)
        after = {name: sha(ROOT / name) for name in OUTPUTS}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
