#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import build_sft_equal_unit_matched_init_preregistration_v42b as builder
import run_sft_equal_unit_matched_init_v42b as subject


class DirectCanonicalLoadRunnerV42BTest(unittest.TestCase):
    def test_command_uses_retry_script_and_same_recipe(self):
        args = builder.arguments()
        command = subject.build_train_command(args)
        self.assertEqual(command[3], str(subject.SFT_SCRIPT))
        self.assertEqual(command[1:3], ["--standalone", "--nproc-per-node=4"])
        self.assertEqual(
            command[command.index("--initial-adapter-weights-sha256") + 1],
            subject.source_contract.INITIAL_WEIGHTS_SHA256_V42A,
        )
        self.assertEqual(command[command.index("--save-steps") + 1], "16")

    def test_failed_predecessor_is_content_addressed(self):
        value = subject.validate_failed_predecessor_v42b()
        self.assertTrue(value["failed_before_optimizer_construction"])
        self.assertTrue(value["retry_changes_only_adapter_load_mechanism"])
        self.assertIn("distributed_operation", value["failure_signature"])

    def test_preregistration_round_trip(self):
        value = builder.build()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preregistration_v42b.json"
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
            audit["adapter_loader"]["loader"],
            "get_peft_model_plus_inplace_canonical_state_copy",
        )
        self.assertEqual(
            audit["source_initialization"]["weights_file_sha256"],
            subject.source_contract.INITIAL_WEIGHTS_SHA256_V42A,
        )


if __name__ == "__main__":
    unittest.main()
