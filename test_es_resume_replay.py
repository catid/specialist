import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import es_replay_to_servers
import es_train
import es_train_acc


class GenerationRngTests(unittest.TestCase):
    @staticmethod
    def draw_generation(module, global_seed, generation):
        rng = module.generation_rng(global_seed, generation)
        batch = rng.sample(range(100), 12)
        seeds = [rng.randrange(1, 2**60) for _ in range(8)]
        return batch, seeds

    def test_resume_draws_match_uninterrupted_run(self):
        for module in (es_train, es_train_acc):
            with self.subTest(module=module.__name__):
                uninterrupted = [self.draw_generation(module, 17, g)
                                 for g in range(7)]
                resumed = [self.draw_generation(module, 17, g)
                           for g in range(4, 7)]
                self.assertEqual(resumed, uninterrupted[4:])
                self.assertNotEqual(uninterrupted[0], uninterrupted[4])

    def test_trainers_use_the_same_rng_scheme(self):
        for generation in range(5):
            self.assertEqual(
                self.draw_generation(es_train, -3, generation),
                self.draw_generation(es_train_acc, -3, generation))

    def test_negative_generation_is_rejected(self):
        for module in (es_train, es_train_acc):
            with self.subTest(module=module.__name__):
                with self.assertRaises(ValueError):
                    module.generation_rng(1, -1)


class JournalLoadingTests(unittest.TestCase):
    def test_journal_must_be_contiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.jsonl"
            path.write_text('{"gen": 0}\n{"gen": 2}\n')
            for module in (es_train, es_train_acc):
                with self.subTest(module=module.__name__):
                    with self.assertRaisesRegex(ValueError, "expected 1"):
                        module.load_journal(path)

    def test_file_identity_uses_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text("one\ntwo\n")
            first = es_train.file_identity(path, 2)
            second = es_train_acc.file_identity(path, 2)
            self.assertEqual(first, second)
            self.assertEqual(first["items"], 2)
            self.assertEqual(first["bytes"], 8)
            path.write_text("one\nthree\n")
            self.assertNotEqual(first["sha256"],
                                es_train.file_identity(path, 2)["sha256"])


class ReplayTests(unittest.TestCase):
    def write_journal(self, rows):
        temporary = tempfile.NamedTemporaryFile(mode="w", delete=False)
        with temporary:
            for row in rows:
                temporary.write(json.dumps(row) + "\n")
        self.addCleanup(Path(temporary.name).unlink)
        return temporary.name

    @staticmethod
    def modern_row(generation, include_regex):
        return {
            "schema_version": 2,
            "gen": generation,
            "ops": [[100 + generation, 0.25]],
            "sigma": 0.01,
            "rank": 4,
            "include_regex": include_regex,
        }

    @patch.object(es_replay_to_servers, "perturb")
    def test_replay_honors_each_rows_target(self, perturb_mock):
        path = self.write_journal([
            self.modern_row(0, r"layers\.0\."),
            self.modern_row(1, None),
        ])
        count = es_replay_to_servers.replay_journal(path, [30001, 30002])
        self.assertEqual(count, 2)
        self.assertEqual(perturb_mock.call_args_list, [
            call(30001, [(100, 0.25)], 0.01, rank=4, mode="commit",
                 include_regex=r"layers\.0\."),
            call(30002, [(100, 0.25)], 0.01, rank=4, mode="commit",
                 include_regex=r"layers\.0\."),
            call(30001, [(101, 0.25)], 0.01, rank=4, mode="commit",
                 include_regex=None),
            call(30002, [(101, 0.25)], 0.01, rank=4, mode="commit",
                 include_regex=None),
        ])

    @patch.object(es_replay_to_servers, "perturb")
    def test_ambiguous_legacy_journal_fails_before_mutation(self, perturb_mock):
        path = self.write_journal([{
            "gen": 0, "ops": [[9, 0.5]], "sigma": 0.02, "rank": 8,
        }])
        with self.assertRaisesRegex(ValueError, "target is ambiguous"):
            es_replay_to_servers.replay_journal(path, [30001])
        perturb_mock.assert_not_called()

    @patch.object(es_replay_to_servers, "perturb")
    def test_legacy_target_override_is_explicit(self, perturb_mock):
        path = self.write_journal([{
            "gen": 0, "ops": [[9, 0.5]], "sigma": 0.02, "rank": 8,
        }])
        es_replay_to_servers.replay_journal(
            path, [30001], legacy_include_regex="front|back")
        perturb_mock.assert_called_once_with(
            30001, [(9, 0.5)], 0.02, rank=8, mode="commit",
            include_regex="front|back")

    @patch.object(es_replay_to_servers, "perturb")
    def test_legacy_full_model_requires_affirmation(self, perturb_mock):
        path = self.write_journal([{
            "gen": 0, "ops": [[9, 0.5]], "sigma": 0.02, "rank": 8,
        }])
        es_replay_to_servers.replay_journal(
            path, [30001], allow_legacy_full_model=True)
        perturb_mock.assert_called_once_with(
            30001, [(9, 0.5)], 0.02, rank=8, mode="commit",
            include_regex=None)

    @patch.object(es_replay_to_servers, "perturb")
    def test_malformed_modern_row_cannot_use_legacy_override(self, perturb_mock):
        row = self.modern_row(0, None)
        del row["include_regex"]
        path = self.write_journal([row])
        with self.assertRaisesRegex(ValueError, "schema_version >= 2"):
            es_replay_to_servers.replay_journal(
                path, [30001], legacy_include_regex="anything")
        perturb_mock.assert_not_called()

    @patch.object(es_replay_to_servers, "perturb")
    def test_empty_target_is_rejected_instead_of_becoming_full_model(
            self, perturb_mock):
        path = self.write_journal([self.modern_row(0, "")])
        with self.assertRaisesRegex(ValueError, "empty include_regex"):
            es_replay_to_servers.replay_journal(path, [30001])
        perturb_mock.assert_not_called()

    @patch.object(es_replay_to_servers, "perturb")
    def test_all_rows_are_validated_before_first_mutation(self, perturb_mock):
        malformed = self.modern_row(1, "back")
        del malformed["rank"]
        path = self.write_journal([
            self.modern_row(0, "front"), malformed,
        ])
        with self.assertRaisesRegex(ValueError, "invalid replay fields"):
            es_replay_to_servers.replay_journal(path, [30001])
        perturb_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
