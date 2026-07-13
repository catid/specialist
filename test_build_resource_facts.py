import json
import tempfile
import unittest
from pathlib import Path

from build_resource_facts import build


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class ResourceFactBuildTests(unittest.TestCase):
    def fixture(self, root):
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "schema": "rope-resource-manifest-v1",
            "resources": [{
                "id": "fixture", "name": "Fixture", "category": "learning",
                "purpose": "testing", "supplied_url": "https://example.test/",
                "canonical_url": "https://example.test/",
            }],
        }))
        packet = root / "packet.jsonl"
        row = {
            "answer": "It describes peer-to-peer rope education.",
            "claim_type": "durable_org",
            "evidence": "The group describes its focus as peer-to-peer rope education.",
            "evidence_url": "https://example.test/about",
            "question": "How does Fixture describe its educational focus?",
            "resource_id": "fixture", "reviewer": "reader",
            "source": "fixture", "verified_at": "2026-07-13",
        }
        write_jsonl(packet, [row])
        return manifest, packet, row

    def test_valid_packet_builds_provenance_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, packet, _ = self.fixture(root)
            output = root / "output.jsonl"
            report = build(manifest, [packet], output, root / "report.json", [])
            item = json.loads(output.read_text())
            self.assertEqual(report["counts"]["output"], 1)
            self.assertEqual(item["kind"], "qa_resource_manual_fact")
            self.assertEqual(item["review"]["reviewer"], "reader")

    def test_weakly_supported_answer_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, packet, row = self.fixture(root)
            row["answer"] = "The frame weighs seventy pounds and is made of steel."
            write_jsonl(packet, [row])
            with self.assertRaisesRegex(ValueError, "weakly supported"):
                build(manifest, [packet], root / "out.jsonl",
                      root / "report.json", [])

    def test_off_site_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, packet, row = self.fixture(root)
            row["evidence_url"] = "https://other.test/about"
            write_jsonl(packet, [row])
            with self.assertRaisesRegex(ValueError, "off the resource site"):
                build(manifest, [packet], root / "out.jsonl",
                      root / "report.json", [])

    def test_volatile_question_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, packet, row = self.fixture(root)
            row["question"] = "What is Fixture's current ticket price?"
            write_jsonl(packet, [row])
            with self.assertRaisesRegex(ValueError, "volatile or low-value"):
                build(manifest, [packet], root / "out.jsonl",
                      root / "report.json", [])


if __name__ == "__main__":
    unittest.main()
