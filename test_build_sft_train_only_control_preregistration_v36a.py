#!/usr/bin/env python3
import json
import unittest

import build_sft_train_only_control_preregistration_v36a as prereg
import run_sft_train_only_control_v36a as runtime


class SFTControlPreregistrationV36ATest(unittest.TestCase):
    def test_frozen_train_only_contract(self):
        value = json.loads(prereg.PREREGISTRATION.read_text())
        self.assertFalse(value["contains_validation_ood_or_holdout_content"])
        self.assertEqual(value["dataset"]["rows"], 531)
        self.assertEqual(value["dataset"]["unique_documents"], 289)
        self.assertEqual(value["recipe"]["expected_optimizer_steps"], 57)
        self.assertTrue(value["recipe"]["equal_microbatch_sizes"])
        command = value["recipe"]["command"]
        self.assertEqual(command[command.index("--save-steps") + 1], "19")
        self.assertNotIn("--eval-data", value["recipe"]["command"])
        self.assertEqual(
            value["content_sha256_before_self_field"],
            runtime.canonical_sha256({
                key: item for key, item in value.items()
                if key != "content_sha256_before_self_field"
            }),
        )

    def test_historical_preregistration_rejects_changed_sft_source(self):
        value = json.loads(prereg.PREREGISTRATION.read_text())
        args = prereg._arguments()
        args.preregistration_sha256 = runtime.file_sha256(prereg.PREREGISTRATION)
        args.preregistration_content_sha256 = value[
            "content_sha256_before_self_field"
        ]
        with self.assertRaisesRegex(RuntimeError, "implementation binding"):
            runtime.validate_preregistration(args, runtime.build_train_command(args))


if __name__ == "__main__":
    unittest.main()
