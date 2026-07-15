#!/usr/bin/env python3

import unittest

import build_sft_equal_unit_preregistration_v37a as subject


class EqualUnitPreregistrationV37ATest(unittest.TestCase):
    def test_exact_confirmatory_recipe(self):
        value = subject.build()
        self.assertEqual(value["comparison_binding"]["confirmatory_fold"], 3)
        self.assertFalse(
            value["comparison_binding"]["shadow_dev_opened_during_preregistration"]
        )
        self.assertEqual(value["recipe"]["rows_per_epoch"], 448)
        self.assertEqual(value["recipe"]["expected_optimizer_steps"], 48)
        self.assertEqual(value["recipe"]["distributed_padding_rows_per_epoch"], 0)
        self.assertFalse(value["recipe"]["renormalize_weights_within_batch"])
        self.assertEqual(
            value["content_sha256_before_self_field"],
            subject.engine.canonical_sha256({
                key: item for key, item in value.items()
                if key != "content_sha256_before_self_field"
            }),
        )


if __name__ == "__main__":
    unittest.main()
