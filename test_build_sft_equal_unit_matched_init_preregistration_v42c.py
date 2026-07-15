#!/usr/bin/env python3

import unittest

import build_sft_equal_unit_matched_init_preregistration_v42c as subject


class LowerLearningRatePreregistrationV42CTest(unittest.TestCase):
    def test_only_hpo_change_is_bound(self):
        value = subject.build()
        args = subject.arguments()
        command = value["recipe"]["command"]
        self.assertEqual(value["recipe"]["learning_rate"], 3e-5)
        self.assertEqual(
            command[command.index("--learning-rate") + 1], "3e-05"
        )
        self.assertEqual(value["recipe"]["expected_optimizer_steps"], 48)
        self.assertEqual(value["dataset"]["rows"], 448)
        self.assertEqual(value["initialization"]["tensor_count"], 70)
        self.assertEqual(value["artifacts"]["output_dir"], args.output_dir)
        self.assertEqual(
            value["content_sha256_before_self_field"],
            subject.v42b.engine.canonical_sha256({
                key: item for key, item in value.items()
                if key != "content_sha256_before_self_field"
            }),
        )


if __name__ == "__main__":
    unittest.main()
