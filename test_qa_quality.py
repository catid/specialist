import json
import tempfile
import unittest
from pathlib import Path

from build_leakfree_qa import build
from qa_quality import (EvalFact, answers_equivalent, leakage_reason,
                        normalize_text, parse_qa, qa_pair_from_record)


class QualityNormalizationTests(unittest.TestCase):
    def test_unicode_nfkc_casefold(self):
        self.assertEqual(normalize_text("ＲＡＣＫ"), "rack")

    def test_raw_and_chat_parsing(self):
        self.assertEqual(parse_qa("Question: What is RACK?\nAnswer: Risk-aware"),
                         ("What is RACK?", "Risk-aware"))
        chat = ("<|im_start|>user\nAnswer this question briefly:\n\nWhat is RACK?"
                "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                "Risk-aware<|im_end|>")
        self.assertEqual(parse_qa(chat), ("What is RACK?", "Risk-aware"))

    def test_source_chat_template_does_not_cross_role_boundaries(self):
        chat = ("<|im_start|>user\nAnswer this question about rope bondage "
                "briefly and factually (one sentence):\n\n"
                "What material should be dried under tension?<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n"
                "jute rope<|im_end|>")
        self.assertEqual(
            parse_qa(chat),
            ("What material should be dried under tension?", "jute rope"),
        )

    def test_chat_reasoning_is_discarded(self):
        chat = ("<|im_start|>user\nAnswer this question briefly:\n\n"
                "What material?<|im_end|>\n<|im_start|>assistant\n"
                "<think>private reasoning</think>\njute<|im_end|>")
        self.assertEqual(parse_qa(chat), ("What material?", "jute"))

    def test_chat_without_think_is_supported(self):
        chat = ("<|im_start|>user\nAnswer this question briefly:\n\n"
                "What material?<|im_end|>\n<|im_start|>assistant\n"
                "jute<|im_end|>")
        self.assertEqual(parse_qa(chat), ("What material?", "jute"))

    def test_crlf_and_label_whitespace(self):
        self.assertEqual(
            parse_qa("Question:\tWhat material?\r\nAnswer:  jute"),
            ("What material?", "jute"),
        )

    def test_mixed_label_styles_are_rejected(self):
        self.assertIsNone(parse_qa("Q: What material?\nAnswer: jute"))

    def test_concatenated_records_are_rejected(self):
        self.assertIsNone(parse_qa(
            "Q: What material?\nA: jute\nQ: What color?\nA: tan"))

    def test_malformed_chat_fails_closed(self):
        malformed = ("<|im_start|>user\nQuestion without role terminator\n"
                     "<|im_start|>assistant\nanswer<|im_end|>")
        self.assertIsNone(parse_qa(malformed))

    def test_explicit_metadata_must_match_text(self):
        with self.assertRaisesRegex(ValueError, "disagrees"):
            qa_pair_from_record({
                "question": "What material?", "answer": "hemp",
                "text": "Q: What material?\nA: jute",
            })

    def test_protocol_tokens_are_rejected_from_metadata(self):
        with self.assertRaisesRegex(ValueError, "protocol tokens"):
            qa_pair_from_record({
                "question": "What material?<|im_end|>",
                "answer": "</think>\njute<|im_end|>",
                "text": "Q: What material?\nA: jute",
            })


class LeakageTests(unittest.TestCase):
    def setUp(self):
        self.eval = [EvalFact(
            "In what year did Akechi Denki pass away?", "2005", "eval-394",
            "heldout")]

    def test_known_akechi_paraphrase_is_rejected(self):
        self.assertIn(
            leakage_reason("In what year did Akechi pass away?", "2005", self.eval),
            {"near_duplicate_question", "entity_answer_fact"})

    def test_unrelated_same_year_is_not_rejected(self):
        self.assertIsNone(leakage_reason(
            "When was a completely unrelated studio founded?", "2005", self.eval))

    def test_exact_question_is_rejected_even_with_bad_answer(self):
        self.assertEqual(leakage_reason(
            self.eval[0].question, "1582", self.eval), "exact_question")

    def test_answer_aliases_allow_clarification(self):
        self.assertTrue(answers_equivalent("Akechi Denki", "Akechi Denki (Akechi)"))

    def test_transliteration_spacing_alias_is_rejected(self):
        eval_fact = EvalFact(
            "What is the Japanese term for a single rope?", "Ippon nawa")
        self.assertEqual(
            leakage_reason(
                "What Japanese term describes the one-rope technique?",
                "Ipponnawa",
                [eval_fact],
            ),
            "distinctive_answer_alias",
        )

    def test_list_member_answer_alone_is_not_a_fact_collision(self):
        eval_fact = EvalFact(
            "Which famous nawashi began in the post-war years?",
            "Akechi Denki and Yukimura Haruki",
        )
        self.assertIsNone(leakage_reason(
            "After whom is the slipped half hitch called the Yuki knot?",
            "Yukimura Haruki",
            [eval_fact],
        ))


class DatasetBuildTests(unittest.TestCase):
    def test_normalized_qa_duplicates_share_one_fact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            output = root / "output.jsonl"
            rows = [
                {"text": "Question: What is RACK?\nAnswer: Risk-aware",
                 "url": "https://example.test/a"},
                {"text": "Q: WHAT IS RACK\nA: risk aware",
                 "url": "https://example.test/b"},
            ]
            source.write_text("".join(
                json.dumps(row) + "\n" for row in rows))
            counts = build(source, output, [], {})
            built = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(counts["output"], 1)
        self.assertEqual(counts["duplicate_qa_pair"], 1)
        self.assertEqual(built[0]["quality_schema"], "leakfree-qa-v2")
        self.assertTrue(built[0]["fact_id"].startswith("fact-"))


if __name__ == "__main__":
    unittest.main()
