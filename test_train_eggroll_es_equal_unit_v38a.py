#!/usr/bin/env python3

import unittest

import train_eggroll_es_equal_unit_v38a as subject


class EqualUnitTrainerV38ATest(unittest.TestCase):
    def test_frozen_train_bundle_and_weights(self):
        bundle = subject.load_equal_unit_train_bundle(
            subject.shadow.OUTPUT_DIR / "fold_3_train.jsonl",
            "97fc920ac39f67536df26977de951e8c34bf8486eb8f42fbb0a67687f025a92a",
            subject.shadow.MANIFEST,
            "7d2a8f2b86f9007aa2bfe8ae043be15647451cc4bbea53a18d5915085879ee9d",
        )
        self.assertEqual(bundle["dataset"]["rows"], 448)
        self.assertEqual(bundle["conflict_units"], 208)
        self.assertAlmostEqual(sum(bundle["weights"]), 1.0)
        self.assertEqual(len(bundle["questions"]), 448)
        self.assertFalse(hasattr(subject.EqualUnitUpdateMixinV38A, "apply_seed_coefficients"))

    def test_fixed_update_recipe(self):
        self.assertEqual(subject.POPULATION_SIZE, 32)
        self.assertEqual(subject.SIGMA, 0.0003)
        self.assertEqual(subject.ALPHA, 0.00015)
        self.assertEqual(len(subject.SEEDS), 32)


if __name__ == "__main__":
    unittest.main()
