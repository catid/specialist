import copy
import json
import tempfile
import unittest
from pathlib import Path

import build_resource_qa
from qa_quality import parse_qa


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "sources" / "rope_resources_v1.json"
PLAYLIST = ROOT / "sources" / "hip_harness_playlist_v1.json"


class ResourceQATests(unittest.TestCase):
    def write_manifest(self, directory, manifest):
        path = Path(directory) / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path

    def test_real_manifest_builds_and_represents_every_resource(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.jsonl"
            report_path = Path(directory) / "report.json"
            report = build_resource_qa.build(MANIFEST, output, report_path)
            manifest = json.loads(MANIFEST.read_text())
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(report["counts"]["resources"], 23)
            self.assertEqual(report["counts"]["output"], 29)
            represented = {url for row in rows for url in row["canonical_urls"]}
            self.assertEqual(
                represented,
                {resource["canonical_url"] for resource in manifest["resources"]},
            )
            supplied = {url for row in rows for url in row["supplied_urls"]}
            self.assertEqual(
                supplied,
                {resource["supplied_url"] for resource in manifest["resources"]},
            )
            answer_urls = {
                token.rstrip(";")
                for row in rows for token in row["answer"].split()
                if token.startswith("https://")
            }
            metadata_urls = {url for row in rows for url in row["urls"]}
            self.assertLessEqual(answer_urls, metadata_urls)
            for row in rows:
                self.assertEqual(parse_qa(row["text"]),
                                 (row["question"], row["answer"]))

    def test_duplicate_resource_id_is_rejected(self):
        manifest = json.loads(MANIFEST.read_text())
        manifest["resources"][1]["id"] = manifest["resources"][0]["id"]
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_manifest(directory, manifest)
            with self.assertRaisesRegex(ValueError, "duplicate resource id"):
                build_resource_qa.load_manifest(path)

    def test_non_https_url_is_rejected(self):
        manifest = json.loads(MANIFEST.read_text())
        manifest["resources"][0]["canonical_url"] = "http://example.com/"
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_manifest(directory, manifest)
            with self.assertRaisesRegex(ValueError, "public HTTPS URL"):
                build_resource_qa.load_manifest(path)

    def test_unknown_category_resource_is_rejected(self):
        manifest = json.loads(MANIFEST.read_text())
        manifest["category_questions"][0]["resource_ids"].append("missing")
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_manifest(directory, manifest)
            with self.assertRaisesRegex(ValueError, "unknown resource ids"):
                build_resource_qa.load_manifest(path)

    def test_build_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            first = directory / "first.jsonl"
            second = directory / "second.jsonl"
            build_resource_qa.build(MANIFEST, first, directory / "first.json")
            build_resource_qa.build(MANIFEST, second, directory / "second.json")
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_hip_harness_playlist_inventory_is_complete_and_discovery_only(self):
        manifest = json.loads(MANIFEST.read_text())
        playlist = json.loads(PLAYLIST.read_text())
        resource = next(item for item in manifest["resources"]
                        if item["id"] == "hip_harness_playlist")
        current = playlist["items"]
        hidden = playlist["hidden_unavailable_items"]
        current_ids = [item["video_id"] for item in current]
        hidden_ids = [item["video_id"] for item in hidden]
        self.assertEqual(playlist["schema"], "youtube-playlist-inventory-v1")
        self.assertEqual(playlist["supplied_url"], resource["supplied_url"])
        self.assertEqual(playlist["canonical_url"], resource["canonical_url"])
        self.assertEqual(playlist["reported_item_count"], 109)
        self.assertEqual(len(current), 105)
        self.assertEqual(len(hidden), 4)
        self.assertEqual([item["position"] for item in current],
                         list(range(1, 106)))
        self.assertEqual(len(set(current_ids)), 105)
        self.assertEqual(len(set(hidden_ids)), 4)
        self.assertFalse(set(current_ids) & set(hidden_ids))
        self.assertEqual(len(current) + len(hidden),
                         playlist["reported_item_count"])
        self.assertEqual(playlist["content_use"]["mode"], "discovery_only")
        self.assertFalse(playlist["content_use"]["is_endorsement"])
        self.assertFalse(playlist["content_use"]["training_qa_ready"])
        for item in current + hidden:
            self.assertEqual(
                item["watch_url"],
                f"https://www.youtube.com/watch?v={item['video_id']}",
            )


if __name__ == "__main__":
    unittest.main()
