import tempfile
import unittest
from pathlib import Path

from promote_curation_ledgers import (promote, read_jsonl, unique_rows,
                                      write_jsonl)


def decision(fact_id, answer="old", action="edit"):
    return {
        "action": action,
        "expected_answer": answer,
        "expected_question": f"Question {fact_id}?",
        "fact_id": fact_id,
    }


class PromoteCurationLedgersTests(unittest.TestCase):
    def test_replacement_keeps_position_and_append_follows_base(self):
        base = [decision("a"), decision("b")]
        replacement = decision("a", action="drop")
        appended = decision("c", action="drop")
        rows = promote(base, [appended], [replacement])
        self.assertEqual([row["fact_id"] for row in rows], ["a", "b", "c"])
        self.assertEqual(rows[0]["action"], "drop")

    def test_append_cannot_silently_replace_base(self):
        with self.assertRaisesRegex(ValueError, "use replacement"):
            promote([decision("a")], [decision("a", action="drop")], [])

    def test_replacement_must_match_expected_qa(self):
        replacement = decision("a", answer="different", action="drop")
        with self.assertRaisesRegex(ValueError, "changes expected_answer"):
            promote([decision("a")], [], [replacement])

    def test_jsonl_output_is_deterministic_and_sorted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            write_jsonl(path, [{"fact_id": "a", "action": "drop"}])
            first = path.read_bytes()
            self.assertEqual(read_jsonl(path)[0]["fact_id"], "a")
            write_jsonl(path, [{"action": "drop", "fact_id": "a"}])
            self.assertEqual(path.read_bytes(), first)

    def test_equivalent_cross_ledger_decisions_are_confirmations(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jsonl"
            second = Path(directory) / "second.jsonl"
            row = decision("a", action="drop")
            write_jsonl(first, [row])
            write_jsonl(second, [{**row, "reviewer": "second"}])
            rows, confirmations = unique_rows([first, second], "append")
            self.assertEqual(len(rows), 1)
            self.assertEqual(confirmations, ["a"])

    def test_conflicting_cross_ledger_decisions_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jsonl"
            second = Path(directory) / "second.jsonl"
            write_jsonl(first, [decision("a", action="drop")])
            write_jsonl(second, [decision("a", action="edit")])
            with self.assertRaisesRegex(ValueError, "conflicting append"):
                unique_rows([first, second], "append")


if __name__ == "__main__":
    unittest.main()
