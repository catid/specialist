import json
import tempfile
import unittest
from pathlib import Path

import es_train_acc


class AccuracyRewardTests(unittest.TestCase):
    def test_unicode_answers_are_scoreable(self):
        self.assertEqual(es_train_acc.toks("縛"), ["縛"])
        self.assertEqual(es_train_acc.answer_score("縛", "縛"), 1.0)
        self.assertEqual(es_train_acc.answer_score("緊", "縛"), 0.0)

    def test_nfkc_and_casefold_exact_match(self):
        self.assertEqual(es_train_acc.answer_score("ＲＡＣＫ", "rack"), 1.0)

    def test_partial_credit_is_bounded_below_exact_match(self):
        partial = es_train_acc.answer_score("risk aware consensual", "risk aware")
        self.assertGreater(partial, 0.0)
        self.assertLess(partial, 0.31)


class FitnessShapingTests(unittest.TestCase):
    def test_raw_shaping_preserves_sparse_reward_magnitude(self):
        rewards = [0.0, 0.00075, 0.0]
        self.assertEqual(es_train_acc.shape_fitness(rewards, "raw"), rewards)

    def test_unknown_shaping_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown shaping"):
            es_train_acc.shape_fitness([0.0], "mystery")


class ShuffledStreamTests(unittest.TestCase):
    def test_first_epoch_has_no_replacement(self):
        batches = [es_train_acc.batch_indices_without_replacement(12, 4, 9, g)
                   for g in range(3)]
        flat = [index for batch in batches for index in batch]
        self.assertEqual(len(flat), 12)
        self.assertEqual(set(flat), set(range(12)))

    def test_generation_is_resume_independent(self):
        direct = es_train_acc.batch_indices_without_replacement(17, 5, 9, 7)
        resumed = es_train_acc.batch_indices_without_replacement(17, 5, 9, 7)
        self.assertEqual(direct, resumed)


class ResumeManifestTests(unittest.TestCase):
    @staticmethod
    def expected():
        return {
            "trainer": "es_train_acc",
            "include_regex": "front",
            "shaping": "ranks",
            "global_seed": 11,
            "rng_scheme": es_train_acc.RNG_SCHEME,
            "hparams": {"pairs": 4},
            "data": {"sha256": "data"},
            "probe_data": {"sha256": "eval"},
        }

    def test_matching_manifest_is_accepted(self):
        row = dict(self.expected(), schema_version=2)
        es_train_acc.validate_resume([row], self.expected())

    def test_changed_target_is_rejected(self):
        row = dict(self.expected(), schema_version=2, include_regex="back")
        with self.assertRaisesRegex(ValueError, "include_regex"):
            es_train_acc.validate_resume([row], self.expected())

    def test_legacy_resume_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "predates"):
            es_train_acc.validate_resume([{"schema_version": 1}], self.expected())


if __name__ == "__main__":
    unittest.main()
