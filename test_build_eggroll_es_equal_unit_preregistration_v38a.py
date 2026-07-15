#!/usr/bin/env python3

import unittest

import build_eggroll_es_equal_unit_preregistration_v38a as subject


class PreregistrationV38ATest(unittest.TestCase):
    def test_exact_nonzero_train_only_recipe(self):
        value = subject.build()
        recipe = value["recipe"]
        self.assertEqual(recipe["population_size"], 32)
        self.assertEqual(recipe["signed_directions"], 64)
        self.assertEqual(recipe["sigma"], 0.0003)
        self.assertEqual(recipe["alpha"], 0.00015)
        self.assertFalse(recipe["alpha_search"])
        self.assertEqual(recipe["conflict_units"], 208)
        self.assertEqual(recipe["packed_runtime_tensors"], 23)
        self.assertFalse(value["shadow_dev_external_eval_ood_or_holdout_opened"])
        self.assertEqual(
            value["content_sha256_before_self_field"],
            subject.hashing.canonical_sha256({
                key: item for key, item in value.items()
                if key != "content_sha256_before_self_field"
            }),
        )


if __name__ == "__main__":
    unittest.main()
