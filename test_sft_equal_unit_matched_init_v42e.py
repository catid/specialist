#!/usr/bin/env python3

import unittest

import build_sft_equal_unit_matched_init_preregistration_v42e as builder
import run_sft_equal_unit_matched_init_v42e as runner


class ThreeEminusFourHPOV42ETest(unittest.TestCase):
    def test_preregistration_and_command(self):
        value = builder.build()
        command = value["recipe"]["command"]
        self.assertEqual(value["recipe"]["learning_rate"], 3e-4)
        self.assertEqual(command[command.index("--learning-rate") + 1], "0.0003")
        self.assertEqual(command[3], str(runner.v42b.SFT_SCRIPT))
        self.assertEqual(value["recipe"]["expected_optimizer_steps"], 48)
        self.assertEqual(value["dataset"]["rows"], 448)
        self.assertEqual(
            value["content_sha256_before_self_field"],
            builder.v42b.engine.canonical_sha256({
                key: item for key, item in value.items()
                if key != "content_sha256_before_self_field"
            }),
        )


if __name__ == "__main__":
    unittest.main()
