#!/usr/bin/env python3

import unittest

import build_sft_equal_unit_matched_init_preregistration_v42c as builder
import run_sft_equal_unit_matched_init_v42c as subject


class LowerLearningRateRunnerV42CTest(unittest.TestCase):
    def test_command_changes_only_bound_learning_rate(self):
        args = builder.arguments()
        command = subject.build_train_command(args)
        self.assertEqual(
            command[command.index("--learning-rate") + 1], "3e-05"
        )
        self.assertEqual(command[3], str(subject.v42b.SFT_SCRIPT))
        self.assertEqual(command[1:3], ["--standalone", "--nproc-per-node=4"])


if __name__ == "__main__":
    unittest.main()
