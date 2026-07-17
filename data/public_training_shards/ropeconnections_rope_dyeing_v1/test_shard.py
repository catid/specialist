#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
URL = "https://www.ropeconnections.com/how-i-dyed-my-rope/"
CANDIDATES = ROOT / "data/train_qa_v3_clean_leakfree_v4_candidates.jsonl"
CHUNKS = ROOT / "data/train_chunks_v2.jsonl"
SOURCE = HERE / "source.md"
REVIEW = HERE / "candidate_review.jsonl"
QA = HERE / "qa.jsonl"
REPORT = HERE / "report.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PublicTrainingShardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.review = load_jsonl(REVIEW)
        cls.qa = load_jsonl(QA)
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.candidates = [row for row in load_jsonl(CANDIDATES) if row.get("url") == URL]
        cls.chunks = [row for row in load_jsonl(CHUNKS) if row.get("url") == URL]

    def test_exact_candidate_shard_is_fully_consumed_once(self) -> None:
        self.assertEqual(40, len(self.candidates))
        expected = {row["fact_id"] for row in self.candidates}
        consumed = [fact_id for row in self.review for fact_id in row["candidate_fact_ids"]]
        self.assertEqual(40, len(consumed))
        self.assertEqual(40, len(set(consumed)))
        self.assertEqual(expected, set(consumed))

    def test_decisions_and_output_counts_are_pinned(self) -> None:
        action_counts = {
            action: sum(row["action"] == action for row in self.review)
            for action in ("drop", "edit", "add")
        }
        self.assertEqual({"drop": 4, "edit": 5, "add": 4}, action_counts)
        self.assertEqual(9, len(self.qa))
        self.assertEqual(12, sum(len(row["candidate_fact_ids"]) for row in self.review if row["action"] == "drop"))
        self.assertEqual(28, sum(len(row["candidate_fact_ids"]) for row in self.review if row["action"] == "edit"))

    def test_output_exactly_matches_edit_and_add_decisions(self) -> None:
        decisions = {
            (row["question"], row["answer"], row["evidence"], row["action"])
            for row in self.review
            if row["action"] in {"edit", "add"}
        }
        output = {
            (row["question"], row["answer"], row["evidence"], row["review"]["action"])
            for row in self.qa
        }
        self.assertEqual(decisions, output)

    def test_every_answer_is_extractively_grounded(self) -> None:
        for row in self.qa:
            self.assertIn(row["evidence"], self.source)
            self.assertIn(row["answer"], row["evidence"])
            self.assertEqual(URL, row["url"])
            self.assertEqual("ropeconnections", row["source"])
            self.assertEqual(f"Question: {row['question']}\nAnswer: {row['answer']}", row["text"])

    def test_questions_are_clean_unique_and_not_trivia(self) -> None:
        protocol_tokens = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")
        questions = [row["question"] for row in self.qa]
        self.assertEqual(len(questions), len({question.casefold() for question in questions}))
        for row in self.qa:
            rendered = row["question"] + "\n" + row["answer"]
            self.assertFalse(any(token in rendered for token in protocol_tokens))
            self.assertIsNone(re.search(r"\b(url|web address|canonical link|which website)\b", row["question"], re.I))
            self.assertNotIn("provided text", row["question"].casefold())
            self.assertNotIn("the context", row["question"].casefold())
            self.assertNotIn("welcome to night vale", rendered.casefold())
            self.assertNotIn("lincraft", rendered.casefold())
            self.assertNotIn("paknsave", rendered.casefold())
            self.assertNotIn("boil", rendered.casefold())

    def test_source_is_separate_dense_markdown(self) -> None:
        self.assertTrue(self.source.startswith("# How I Dyed My Rope\n"))
        self.assertIn(f"Source: {URL}", self.source)
        self.assertGreaterEqual(len(re.findall(r"\b[\w’'-]+\b", self.source)), 1_250)
        self.assertNotIn("Question:", self.source)
        self.assertNotIn("Answer:", self.source)

    def test_source_copy_preserves_three_original_training_chunks(self) -> None:
        self.assertEqual(3, len(self.chunks))
        marker = f"Source: {URL}\n\n"
        source_body = self.source.split(marker, maxsplit=1)[1]
        original = "\n".join(row["text"] for row in self.chunks)
        original = original.replace("How I Did It (Reasoning Included)", "## How I Did It (Reasoning Included)")
        original = original.replace("Update:", "## Update")
        normalize = lambda text: re.sub(r"\s+", " ", text).strip()
        self.assertEqual(normalize(original), normalize(source_body))

    def test_report_counts_hashes_and_promotion_boundary(self) -> None:
        self.assertEqual(sha256(CANDIDATES), self.report["candidate_input"]["sha256"])
        self.assertEqual(40, self.report["candidate_input"]["matching_rows"])
        self.assertEqual(sha256(CHUNKS), self.report["source_input"]["sha256"])
        self.assertEqual(3, self.report["source_input"]["matching_chunks"])
        self.assertEqual(9, self.report["review"]["output_qa_rows"])
        quality = self.report["quality"]
        self.assertEqual(0, quality["brand_shopping_or_media_trivia"])
        self.assertEqual(0, quality["unqualified_boiling_claims_in_qa"])
        self.assertTrue(quality["raw_source_requires_downstream_material_content_safety_review"])
        self.assertTrue(quality["qa_requires_downstream_material_safety_review"])
        for key, path in (("source_markdown", SOURCE), ("candidate_review", REVIEW), ("qa", QA)):
            self.assertEqual(sha256(path), self.report["artifacts"][key]["sha256"])
        boundary = self.report["boundary"]
        self.assertFalse(boundary["evaluation_content_read"])
        self.assertFalse(boundary["protected_content_read"])
        self.assertFalse(boundary["web_crawl_performed"])
        self.assertFalse(boundary["active_training_snapshot_modified"])
        self.assertEqual("pending_semantic_leakage_audit", boundary["promotion_status"])


if __name__ == "__main__":
    unittest.main()
