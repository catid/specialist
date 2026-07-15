#!/usr/bin/env python3
import unittest

import build_sft_es_runtime_comparison_v36c as evidence
import run_sft_train_only_control_v36a as runtime


class SFTERSRuntimeComparisonV36CTest(unittest.TestCase):
    def test_recomputed_train_only_comparison(self):
        value = evidence.build()
        comparison = value["comparison"]
        self.assertFalse(value["contains_validation_ood_or_holdout_content"])
        self.assertEqual(comparison["es_sequence_presentations"], 17_920)
        self.assertEqual(comparison["es_token_presentations"], 990_144)
        self.assertEqual(comparison["sft_optimizer_steps"], 57)
        self.assertAlmostEqual(
            comparison["es_over_sft_padded_sequence_ratio"],
            17_920 / 1_596,
        )
        self.assertAlmostEqual(
            comparison["es_over_sft_nominal_token_ratio"],
            990_144 / 119_454,
        )
        self.assertGreater(comparison["es_scoring_critical_path_seconds"], 18)
        self.assertLess(comparison["es_scoring_critical_path_seconds"], 19)
        self.assertEqual(
            value["content_sha256_before_self_field"],
            runtime.canonical_sha256({
                key: item for key, item in value.items()
                if key != "content_sha256_before_self_field"
            }),
        )


if __name__ == "__main__":
    unittest.main()
