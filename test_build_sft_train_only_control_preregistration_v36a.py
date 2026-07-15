#!/usr/bin/env python3
import unittest

import build_sft_train_only_control_preregistration_v36a as prereg
import run_sft_train_only_control_v36a as runtime


class SFTControlPreregistrationV36ATest(unittest.TestCase):
    def test_frozen_train_only_contract(self):
        value = prereg.build()
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

    def test_runtime_revalidates_the_persisted_preregistration(self):
        value = prereg.build()
        args = prereg._arguments()
        args.preregistration_sha256 = runtime.file_sha256(prereg.PREREGISTRATION)
        args.preregistration_content_sha256 = value[
            "content_sha256_before_self_field"
        ]
        observed = runtime.validate_preregistration(
            args, runtime.build_train_command(args)
        )
        self.assertEqual(
            observed["content_sha256"],
            value["content_sha256_before_self_field"],
        )


if __name__ == "__main__":
    unittest.main()
