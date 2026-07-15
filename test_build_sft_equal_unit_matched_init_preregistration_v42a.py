#!/usr/bin/env python3

import unittest

import build_sft_equal_unit_matched_init_preregistration_v42a as subject


class MatchedInitializationPreregistrationV42ATest(unittest.TestCase):
    def test_exact_matched_confirmatory_recipe(self):
        value = subject.build()
        self.assertEqual(value["comparison_binding"]["confirmatory_fold"], 3)
        self.assertFalse(
            value["comparison_binding"][
                "shadow_dev_opened_during_preregistration"
            ]
        )
        self.assertEqual(value["recipe"]["rows_per_epoch"], 448)
        self.assertEqual(value["recipe"]["expected_optimizer_steps"], 48)
        self.assertEqual(value["recipe"]["effective_global_batch_size"], 28)
        self.assertEqual(value["recipe"]["initialization_seed"], 20_260_715_041)
        self.assertTrue(value["recipe"]["fp32_trainable_adapter"])
        self.assertEqual(
            value["initialization"]["weights_file_sha256"],
            subject.sft.INITIAL_WEIGHTS_SHA256_V42A,
        )
        self.assertEqual(
            value["content_sha256_before_self_field"],
            subject.engine.canonical_sha256({
                key: item for key, item in value.items()
                if key != "content_sha256_before_self_field"
            }),
        )


if __name__ == "__main__":
    unittest.main()
