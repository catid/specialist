from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
NS = "http://www.sitemaps.org/schemas/sitemap/0.9"
REASON = "robots_and_content_signal_ai_training_disallowed"
OUTPUTS = (
    "CORPUS.md",
    "REPORT.md",
    "content_records.jsonl",
    "inventory.jsonl",
    "manifest.json",
    "url_dispositions.jsonl",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def xml_locs(path: Path) -> list[str]:
    root = ET.fromstring(path.read_bytes())
    return [node.text.strip() for node in root.findall(f".//{{{NS}}}loc") if node.text]


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = read_json(ROOT / "manifest.json")
        cls.provenance = read_json(ROOT / "source_snapshot" / "provenance.json")
        cls.inventory = read_jsonl(ROOT / "inventory.jsonl")
        cls.dispositions = read_jsonl(ROOT / "url_dispositions.jsonl")
        cls.corpus = (ROOT / "CORPUS.md").read_text(encoding="utf-8")

    def test_zero_content_records_and_direct_training_disabled(self) -> None:
        self.assertEqual((ROOT / "content_records.jsonl").read_bytes(), b"")
        self.assertEqual(self.manifest["content_record_count"], 0)
        self.assertFalse(self.manifest["direct_training_ready"])
        self.assertTrue(self.manifest["non_qa"])
        self.assertEqual(self.manifest["artifact_role"], "policy_exclusion_notice")

    def test_policy_is_exact_and_visible(self) -> None:
        robots = (ROOT / "source_snapshot" / "robots.txt").read_text(encoding="utf-8")
        policy = "Content-Signal: search=yes,ai-train=no,use=reference"
        self.assertIn(policy, robots)
        self.assertRegex(robots, r"(?s)User-agent: GPTBot\s+Disallow: /")
        self.assertEqual(self.manifest["policy_line"], policy)
        self.assertIn(policy, self.corpus)
        self.assertIn(REASON, self.corpus)

    def test_full_sitemap_inventory_and_disposition_equality(self) -> None:
        resources = {item["url"]: item for item in self.provenance["resources"]}
        index_urls = set(xml_locs(ROOT / "source_snapshot" / "sitemap_index.xml"))
        child_urls = {item["url"] for item in self.provenance["resources"] if item["role"] == "child_sitemap"}
        self.assertEqual(index_urls, child_urls)

        expected_memberships: dict[str, set[str]] = {}
        for sitemap_url in index_urls:
            for canonical_url in xml_locs(ROOT / resources[sitemap_url]["path"]):
                expected_memberships.setdefault(canonical_url, set()).add(sitemap_url)
        actual_memberships = {item["url"]: set(item["discovered_in"]) for item in self.inventory}
        self.assertEqual(actual_memberships, expected_memberships)
        self.assertEqual(len(self.inventory), self.manifest["inventory_record_count"])
        self.assertTrue(self.inventory)
        for item in self.inventory:
            self.assertEqual(item["disposition"], "excluded")
            self.assertEqual(item["reason"], REASON)
            self.assertFalse(item["content_retrieved"])
            self.assertFalse(item["direct_training_included"])

        excluded = {item["url"] for item in self.dispositions if item["disposition"] == "excluded"}
        metadata = {item["url"] for item in self.dispositions if item["disposition"] == "retrieved_metadata"}
        self.assertEqual(excluded, set(expected_memberships))
        self.assertEqual(metadata, {item["url"] for item in self.provenance["resources"]})

    def test_zero_content_citation_equality(self) -> None:
        # With no content records, both the content-source set and training-citation
        # set are empty. Robots/sitemap links in CORPUS.md are provenance only.
        content_sources: set[str] = set()
        training_citations: set[str] = set()
        self.assertEqual(content_sources, training_citations)
        for item in self.inventory:
            path = re.sub(r"^https://(?:www\.)?theduchy\.com", "", item["url"])
            if path not in {"", "/"}:
                self.assertNotIn(path, self.corpus)

    def test_no_page_body_retrieval(self) -> None:
        self.assertEqual(self.provenance["capture_scope"], "robots_and_sitemap_metadata_only")
        self.assertEqual(self.provenance["page_body_requests"], 0)
        self.assertEqual(self.manifest["page_body_requests"], 0)
        for resource in self.provenance["resources"]:
            self.assertIn(resource["role"], {"robots_policy", "sitemap_index", "child_sitemap"})
            self.assertTrue(
                resource["url"].endswith("/robots.txt")
                or resource["url"].endswith("sitemap_index.xml")
                or resource["url"].endswith("-sitemap.xml")
            )

    def test_capture_utility_rejects_content_routes(self) -> None:
        spec = importlib.util.spec_from_file_location("capture_metadata", ROOT / "capture_metadata.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader
        spec.loader.exec_module(module)
        for approved in (
            "https://www.theduchy.com/robots.txt",
            "https://www.theduchy.com/sitemap_index.xml",
            "https://www.theduchy.com/tutorial-sitemap.xml",
        ):
            module.validate_metadata_url(approved)
        for rejected in (
            "https://www.theduchy.com/",
            "https://www.theduchy.com/tutorials/example/",
            "https://theduchy.com/login/",
            "https://example.com/sitemap.xml",
            "https://www.theduchy.com/sitemap.xml?route=/tutorial",
        ):
            with self.assertRaises(ValueError, msg=rejected):
                module.validate_metadata_url(rejected)

    def test_snapshot_hashes_lengths_and_timestamps(self) -> None:
        manifest_snapshot = {item["url"]: item for item in self.manifest["source_snapshot"]}
        self.assertEqual(set(manifest_snapshot), {item["url"] for item in self.provenance["resources"]})
        for resource in self.provenance["resources"]:
            path = ROOT / resource["path"]
            self.assertEqual(sha256(path), resource["sha256"])
            self.assertEqual(path.stat().st_size, resource["byte_length"])
            self.assertRegex(resource["retrieved_at"], r"^2026-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z$")
            self.assertEqual(manifest_snapshot[resource["url"]]["retrieved_at"], resource["retrieved_at"])

    def test_output_hashes(self) -> None:
        for name, record in self.manifest["outputs"].items():
            path = ROOT / name
            self.assertEqual(record["sha256"], sha256(path))
            self.assertEqual(record["byte_length"], path.stat().st_size)

    def test_offline_build_is_deterministic(self) -> None:
        before = {name: sha256(ROOT / name) for name in OUTPUTS}
        subprocess.run([sys.executable, str(ROOT / "build.py")], check=True, cwd=ROOT)
        after = {name: sha256(ROOT / name) for name in OUTPUTS}
        self.assertEqual(before, after)

    def test_no_ui_qa_or_title_trivia_leakage(self) -> None:
        direct_surface = self.corpus + (ROOT / "content_records.jsonl").read_text(encoding="utf-8")
        for marker in ("<html", "<script", "wp-admin", "Sign in", "Add to cart", "Question:", "Answer:"):
            self.assertNotIn(marker, direct_surface)
        self.assertNotIn("?", direct_surface)
        self.assertNotIn("/tutorials/", direct_surface)

    def test_taxonomy_media_and_gate_limitations_are_explicit(self) -> None:
        self.assertEqual(self.manifest["taxonomy_mappings"], [])
        self.assertEqual(
            self.manifest["media_and_gate_handling"],
            "not_assessed_without_canonical_page_access",
        )
        self.assertIn("Media and access gates were not assessed", self.corpus)


if __name__ == "__main__":
    unittest.main()
