#!/usr/bin/env python3
import unittest

import build_sft_train_only_control_preregistration_v36b as prereg
import run_sft_train_only_control_v36a as runtime


class SFTControlPreregistrationV36BTest(unittest.TestCase):
    def test_fresh_attempt_is_bound_to_zero_step_failure(self):
        value = prereg.build()
        self.assertEqual(value["retry_lineage"]["prior_optimizer_steps"], 0)
        self.assertFalse(value["retry_lineage"]["prior_model_candidate_written"])
        self.assertEqual(value["recipe"]["expected_optimizer_steps"], 57)
        command = value["recipe"]["command"]
        self.assertEqual(command[command.index("--save-steps") + 1], "19")
        self.assertEqual(
            value["content_sha256_before_self_field"],
            runtime.canonical_sha256({
                key: item for key, item in value.items()
                if key != "content_sha256_before_self_field"
            }),
        )

    def test_persisted_preregistration_revalidates(self):
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
            observed["content_sha256"], value["content_sha256_before_self_field"]
        )


if __name__ == "__main__":
    unittest.main()
