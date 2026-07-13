import json
import tempfile
import unittest
from pathlib import Path

from build_manual_qa import build


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class ManualQABuildTests(unittest.TestCase):
    def fixture(self, root):
        raw = root / "raw"
        raw.mkdir()
        (raw / "doc.json").write_text(json.dumps({
            "source": "source-a",
            "url": "https://example.test/a",
            "text": "If jute rope gets wet, dry the rope under tension.",
        }))
        candidates = root / "candidates.jsonl"
        write_jsonl(candidates, [{
            "fact_id": "fact-old",
            "source": "source-a",
            "url": "https://example.test/a",
            "question": "How should wet jute rope be dried?",
            "answer": "under tension",
        }])
        return raw, candidates

    def test_valid_review_emits_provenance_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, candidates = self.fixture(root)
            review = root / "review.jsonl"
            output = root / "output.jsonl"
            subset = root / "subset.jsonl"
            write_jsonl(review, [{
                "action": "keep",
                "candidate_fact_ids": ["fact-old"],
                "question": "How should wet jute rope be dried?",
                "answer": "under tension",
                "evidence": "If jute rope gets wet, dry the rope under tension.",
                "reason": "clear and directly grounded",
                "source": "source-a",
                "url": "https://example.test/a",
                "reviewer": "reader-a",
                "batch": "batch-test",
            }])
            report = build(candidates, [review], raw, output, [], subset)
            item = json.loads(output.read_text())
            subset_item = json.loads(subset.read_text())
        self.assertEqual(report["counts"]["output"], 1)
        self.assertEqual(item["kind"], "qa_manual")
        self.assertEqual(item["review"]["candidate_fact_ids"], ["fact-old"])
        self.assertEqual(subset_item["kind"], "qa_manual_candidate")
        self.assertEqual(len(report["reviewed_candidates_sha256"]), 64)

    def test_incomplete_source_review_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, candidates = self.fixture(root)
            with candidates.open("a") as destination:
                destination.write(json.dumps({
                    "fact_id": "fact-unreviewed",
                    "source": "source-a",
                    "url": "https://example.test/a",
                    "question": "What material gets wet?",
                    "answer": "jute rope",
                }) + "\n")
            review = root / "review.jsonl"
            write_jsonl(review, [{
                "action": "drop", "candidate_fact_ids": ["fact-old"],
                "reason": "redundant", "source": "source-a",
                "url": "https://example.test/a", "reviewer": "reader-a",
                "batch": "batch-test",
            }])
            with self.assertRaisesRegex(ValueError, "does not cover 1"):
                build(candidates, [review], raw, root / "output.jsonl", [])

    def test_non_source_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, candidates = self.fixture(root)
            review = root / "review.jsonl"
            write_jsonl(review, [{
                "action": "edit", "candidate_fact_ids": ["fact-old"],
                "question": "How should wet jute rope be dried?",
                "answer": "in direct sunlight",
                "evidence": "Dry it in direct sunlight.",
                "reason": "rewrite", "source": "source-a",
                "url": "https://example.test/a", "reviewer": "reader-a",
                "batch": "batch-test",
            }])
            with self.assertRaisesRegex(ValueError, "not a source-text excerpt"):
                build(candidates, [review], raw, root / "output.jsonl", [])


if __name__ == "__main__":
    unittest.main()
