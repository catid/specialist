import json
import tempfile
import unittest
from pathlib import Path

from build_curated_qa import merge
from qa_quality import EvalFact, stable_fact_id


def record(question, answer, kind="qa_test"):
    return {
        "answer": answer,
        "fact_id": stable_fact_id(question, answer),
        "kind": kind,
        "question": question,
        "source": "fixture",
        "text": f"Question: {question}\nAnswer: {answer}",
        "url": "https://example.test/",
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class CuratedQABuildTests(unittest.TestCase):
    def test_merge_is_deterministic_and_counts_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            write_jsonl(first, [record("Where is resource B?", "https://b.test/")])
            write_jsonl(second, [record("Where is resource A?", "https://a.test/")])
            output = root / "output.jsonl"
            report = merge([first, second], output, root / "report.json", [])
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(report["counts"]["output"], 2)
            self.assertEqual(rows[0]["question"], "Where is resource A?")

    def test_duplicate_question_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            write_jsonl(first, [record("What is a bight?", "a loop")])
            write_jsonl(second, [record("What is a bight?", "a curve")])
            with self.assertRaisesRegex(ValueError, "duplicate question"):
                merge([first, second], root / "out.jsonl",
                      root / "report.json", [])

    def test_evaluation_leakage_is_excluded_and_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            write_jsonl(source, [record("What is a bight?", "a loop")])
            facts = [EvalFact("What is a bight?", "a loop", "eval-1", "heldout")]
            output = root / "out.jsonl"
            report = merge([source], output, root / "report.json", facts)
            self.assertEqual(output.read_text(), "")
            self.assertEqual(report["counts"]["excluded"], 1)
            self.assertEqual(
                report["counts"]["exclusion_reasons"],
                {"exact_question": 1},
            )


if __name__ == "__main__":
    unittest.main()
