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
            write_jsonl(
                first, [
                    record(
                        "Where is resource B?", "https://b.test/")])
            write_jsonl(
                second, [
                    record(
                        "Where is resource A?", "https://a.test/")])
            output = root / "output.jsonl"
            report = merge([first, second], output, root / "report.json", [])
            rows = [json.loads(line)
                    for line in output.read_text().splitlines()]
            self.assertEqual(report["counts"]["output"], 2)
            self.assertEqual(rows[0]["question"], "Where is resource A?")

    def test_merge_canonicalizes_legacy_prompt_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            item = record("What material is described?", "jute")
            item["text"] = (
                "Answer this question briefly and factually:\n\n"
                "What material is described?\n\njute"
            )
            write_jsonl(source, [item])
            output = root / "out.jsonl"
            merge([source], output, root / "report.json", [])
            row = json.loads(output.read_text())
            self.assertEqual(
                row["text"],
                "Question: What material is described?\nAnswer: jute",
            )

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
            facts = [
                EvalFact(
                    "What is a bight?",
                    "a loop",
                    "eval-1",
                    "heldout")]
            output = root / "out.jsonl"
            report = merge([source], output, root / "report.json", facts)
            self.assertEqual(output.read_text(), "")
            self.assertEqual(report["counts"]["excluded"], 1)
            self.assertEqual(
                report["counts"]["exclusion_reasons"],
                {"exact_question": 1},
            )

    def test_manual_curation_edits_and_drops_are_auditable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            edit = record("What material is described?", "jute")
            drop = record("What was the old sale price?", "$10")
            write_jsonl(source, [edit, drop])
            curation = root / "curation.jsonl"
            write_jsonl(curation, [
                {
                    "action": "edit",
                    "answer": "jute rope",
                    "evidence": "The material described is jute rope.",
                    "evidence_url": "https://example.test/",
                    "expected_answer": "jute",
                    "expected_question": "What material is described?",
                    "fact_id": edit["fact_id"],
                    "question": "What rope material is described?",
                    "reason": (
                        "Names the subject and uses the complete phrase."
                    ),
                    "reason_code": "contextless_question",
                    "reviewed_at": "2026-07-14",
                    "reviewer": "fixture-reviewer",
                },
                {
                    "action": "drop",
                    "expected_answer": "$10",
                    "expected_question": "What was the old sale price?",
                    "fact_id": drop["fact_id"],
                    "reason": "Historical sale trivia is volatile.",
                    "reason_code": "volatile_commerce",
                    "reviewed_at": "2026-07-14",
                    "reviewer": "fixture-reviewer",
                },
            ])
            output = root / "out.jsonl"
            report = merge(
                [source],
                output,
                root /
                "report.json",
                [],
                curation)
            rows = [json.loads(line)
                    for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["answer"], "jute rope")
            self.assertEqual(rows[0]["quality_schema"], "curated-qa-v1")
            self.assertEqual(
                rows[0]["curation"]["original_fact_id"], edit["fact_id"])
            self.assertEqual(report["curation"]["by_action"],
                             {"drop": 1, "edit": 1})
            self.assertEqual(
                report["counts"]["exclusion_reasons"],
                {"manual_curation:volatile_commerce": 1},
            )

    def test_stale_manual_curation_decision_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            item = record("What material is described?", "jute")
            write_jsonl(source, [item])
            curation = root / "curation.jsonl"
            write_jsonl(curation, [{
                "action": "drop",
                "expected_answer": "hemp",
                "expected_question": item["question"],
                "fact_id": item["fact_id"],
                "reason": "Fixture stale decision.",
                "reason_code": "fixture",
                "reviewed_at": "2026-07-14",
                "reviewer": "fixture-reviewer",
            }])
            with self.assertRaisesRegex(ValueError, "stale curation decision"):
                merge([source], root / "out.jsonl", root / "report.json", [],
                      curation)

    def test_multiple_curation_ledgers_are_applied_and_attributed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            first = record("Which material is named?", "jute")
            second = record("Which material is softer?", "hemp")
            write_jsonl(source, [first, second])
            ledgers = [root / "first.jsonl", root / "second.jsonl"]
            write_jsonl(ledgers[0], [{
                "action": "drop",
                "expected_answer": first["answer"],
                "expected_question": first["question"],
                "fact_id": first["fact_id"],
                "reason": "Fixture drop.",
                "reason_code": "fixture_drop",
                "reviewed_at": "2026-07-14",
                "reviewer": "fixture-reviewer",
            }])
            write_jsonl(ledgers[1], [{
                "action": "edit",
                "answer": "soft hemp",
                "evidence": "The material is soft hemp.",
                "evidence_url": second["url"],
                "expected_answer": second["answer"],
                "expected_question": second["question"],
                "fact_id": second["fact_id"],
                "question": "Which soft material is named?",
                "reason": "Makes the fixture self-contained.",
                "reason_code": "fixture_edit",
                "reviewed_at": "2026-07-14",
                "reviewer": "fixture-reviewer",
            }])
            output = root / "out.jsonl"
            report = merge(
                [source], output, root / "report.json", [], ledgers)
            row = json.loads(output.read_text())
            self.assertEqual(
                row["curation"]["decision_file"], str(ledgers[1].resolve()))
            self.assertEqual(report["curation"]["decisions"], 2)
            self.assertEqual(
                [item["path"] for item in report["curation"]["artifacts"]],
                [str(path.resolve()) for path in ledgers],
            )

    def test_curation_accepts_recorded_url_evidence_url(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            item = record("Which URL is listed?", "https://old.test/")
            item["url_evidence_url"] = "https://example.test/sitemap.xml"
            write_jsonl(source, [item])
            curation = root / "curation.jsonl"
            write_jsonl(curation, [{
                "action": "edit",
                "answer": "https://new.test/",
                "evidence": "<loc>https://new.test/</loc>",
                "evidence_url": item["url_evidence_url"],
                "expected_answer": item["answer"],
                "expected_question": item["question"],
                "fact_id": item["fact_id"],
                "question": "Which URL does the sitemap list?",
                "reason": (
                    "Uses the source record's explicit sitemap evidence."
                ),
                "reason_code": "broken_canonicalization",
                "reviewed_at": "2026-07-14",
                "reviewer": "fixture-reviewer",
            }])
            output = root / "out.jsonl"
            merge([source], output, root / "report.json", [], curation)
            row = json.loads(output.read_text())
            self.assertEqual(row["answer"], "https://new.test/")
            self.assertEqual(
                row["evidence"], "<loc>https://new.test/</loc>")
            self.assertEqual(
                row["evidence_url"],
                "https://example.test/sitemap.xml",
            )
            self.assertEqual(row["url"], "https://example.test/")


if __name__ == "__main__":
    unittest.main()
