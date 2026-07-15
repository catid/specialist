#!/usr/bin/env python3

import unittest

import build_sft_equal_unit_matched_init_preregistration_v42b as subject


class DirectCanonicalLoadPreregistrationV42BTest(unittest.TestCase):
    def test_retry_changes_only_loader(self):
        value = subject.build()
        self.assertEqual(value["comparison_binding"]["confirmatory_fold"], 3)
        self.assertFalse(
            value["comparison_binding"][
                "shadow_dev_opened_during_preregistration"
            ]
        )
        self.assertEqual(value["recipe"]["expected_optimizer_steps"], 48)
        self.assertEqual(value["recipe"]["initialization_seed"], 20_260_715_041)
        self.assertIn("only_change_from_v42a", value["recipe"])
        self.assertTrue(
            value["predecessor_failure"][
                "failed_before_optimizer_construction"
            ]
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
