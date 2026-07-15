#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

import build_sft_equal_unit_matched_init_preregistration_v42a as builder
import run_sft_equal_unit_matched_init_v42a as subject


class MatchedInitializationRunnerV42ATest(unittest.TestCase):
    def test_command_binds_four_rank_source_and_equal_unit_inputs(self):
        args = builder.arguments()
        command = subject.build_train_command(args)
        self.assertEqual(command[1:3], ["--standalone", "--nproc-per-node=4"])
        self.assertEqual(command[3], str(subject.SFT_SCRIPT))
        self.assertEqual(
            command[command.index("--initial-adapter") + 1],
            str(subject.sft.INITIAL_ADAPTER_V42A),
        )
        self.assertEqual(
            command[command.index("--initial-adapter-weights-sha256") + 1],
            subject.sft.INITIAL_WEIGHTS_SHA256_V42A,
        )
        self.assertEqual(
            command[command.index("--expected-conflict-units") + 1], "208"
        )
        self.assertEqual(command[command.index("--save-steps") + 1], "16")

    def test_preregistration_round_trip_binds_initialization(self):
        value = builder.build()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preregistration_v42a.json"
            builder.engine.atomic_write_json(path, value)
            args = builder.arguments()
            args.preregistration = str(path)
            args.preregistration_sha256 = builder.engine.file_sha256(path)
            args.preregistration_content_sha256 = value[
                "content_sha256_before_self_field"
            ]
            command = subject.build_train_command(args)
            audit = subject.validate_preregistration(args, command)
        self.assertEqual(
            audit["source_initialization"]["weights_file_sha256"],
            subject.sft.INITIAL_WEIGHTS_SHA256_V42A,
        )
        self.assertEqual(
            audit["implementation_bindings"]["objective"],
            builder.EXPECTED["equal_unit_objective_sha256"],
        )

    def test_runtime_initialization_audit_is_exact(self):
        audit = subject.expected_initialization_runtime_audit_v42a()
        self.assertTrue(audit["source"]["verified"])
        self.assertTrue(audit["loaded"]["matches_source_tensor_bytes"])
        self.assertEqual(
            audit["loaded"]["source_tensor_identity_sha256"],
            subject.sft.INITIAL_TENSOR_IDENTITY_SHA256_V42A,
        )


if __name__ == "__main__":
    unittest.main()
