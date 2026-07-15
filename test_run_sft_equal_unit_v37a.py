#!/usr/bin/env python3

import unittest

import run_sft_equal_unit_v37a as subject


class EqualUnitRunnerV37ATest(unittest.TestCase):
    def test_command_binds_equal_unit_inputs(self):
        args = subject.parser().parse_args([
            "--dataset", "/tmp/train.jsonl",
            "--dataset-sha256", "a" * 64,
            "--dataset-rows", "448",
            "--expected-conflict-units", "208",
            "--expected-weight-identity-sha256", "b" * 64,
            "--output-dir", "/tmp/out",
            "--stdout-log", "/tmp/stdout",
            "--gpu-log", "/tmp/gpu",
            "--report", "/tmp/report",
            "--attempt-report", "/tmp/attempt",
            "--preregistration", "/tmp/prereg",
            "--preregistration-sha256", "c" * 64,
            "--preregistration-content-sha256", "d" * 64,
            "--save-steps", "16",
        ])
        command = subject.build_train_command(args)
        self.assertIn(str(subject.SFT_SCRIPT), command)
        self.assertEqual(command[command.index("--data-rows") + 1], "448")
        self.assertEqual(command[command.index("--expected-conflict-units") + 1], "208")
        self.assertEqual(command[command.index("--save-steps") + 1], "16")


if __name__ == "__main__":
    unittest.main()
