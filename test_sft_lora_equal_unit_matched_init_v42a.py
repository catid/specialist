#!/usr/bin/env python3

import copy
import unittest

import torch
from safetensors.torch import load_file

import sft_lora_equal_unit_matched_init_v42a as subject


class MatchedInitializationSFTV42ATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_audit = subject.validate_initialization_artifact_v42a(
            subject.INITIAL_ADAPTER_V42A
        )
        cls.source_state = load_file(
            str(subject.INITIAL_ADAPTER_V42A / "adapter_model.safetensors"),
            device="cpu",
        )

    def test_sealed_source_and_exact_loaded_state(self):
        self.assertEqual(
            self.source_audit["weights_file_sha256"],
            subject.INITIAL_WEIGHTS_SHA256_V42A,
        )
        self.assertEqual(self.source_audit["tensor_count"], 70)
        self.assertEqual(self.source_audit["elements"], 4_528_128)
        self.assertEqual(self.source_audit["layers"], [20, 21, 22, 23])
        loaded = subject.validate_loaded_adapter_state_v42a(
            self.source_state,
            subject.INITIAL_ADAPTER_V42A,
            self.source_audit,
        )
        self.assertTrue(loaded["matches_source_tensor_bytes"])
        self.assertTrue(loaded["verified_before_optimizer_construction"])
        self.assertEqual(loaded["loaded_dtype"], "torch.float32")
        self.assertEqual(loaded["loaded_tensor_count"], 70)

    def test_loaded_state_mismatch_fails_closed(self):
        changed = dict(self.source_state)
        key = next(
            key for key in sorted(changed)
            if torch.count_nonzero(changed[key]).item() > 0
        )
        changed[key] = changed[key].clone()
        changed[key].view(-1)[0] += 1.0
        with self.assertRaisesRegex(RuntimeError, "differs from source"):
            subject.validate_loaded_adapter_state_v42a(
                changed,
                subject.INITIAL_ADAPTER_V42A,
                self.source_audit,
            )

    def test_v37a_equal_unit_objective_is_reused_exactly(self):
        self.assertIs(subject.EqualUnitTrainer, subject.objective.EqualUnitTrainer)
        self.assertIs(
            subject.assign_equal_unit_weights,
            subject.objective.assign_equal_unit_weights,
        )

    def test_recipe_rejects_wrong_initialization_identity(self):
        args = subject.parser().parse_args([
            "--data", "/tmp/train.jsonl",
            "--data-sha256", "a" * 64,
            "--data-rows", "448",
            "--expected-conflict-units", "208",
            "--expected-weight-identity-sha256", "b" * 64,
            "--initial-adapter", str(subject.INITIAL_ADAPTER_V42A),
            "--initial-adapter-weights-sha256",
            subject.INITIAL_WEIGHTS_SHA256_V42A,
            "--initial-adapter-config-sha256",
            subject.INITIAL_CONFIG_SHA256_V42A,
            "--initial-adapter-manifest-sha256",
            subject.INITIAL_MANIFEST_SHA256_V42A,
            "--initial-adapter-manifest-content-sha256",
            subject.INITIAL_MANIFEST_CONTENT_SHA256_V42A,
            "--initial-adapter-tensor-identity-sha256",
            subject.INITIAL_TENSOR_IDENTITY_SHA256_V42A,
            "--out", "/tmp/out",
        ])
        self.assertEqual(
            subject.validate_recipe_arguments_v42a(args), [20, 21, 22, 23]
        )
        changed = copy.copy(args)
        changed.initial_adapter_weights_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "initialization argument"):
            subject.validate_recipe_arguments_v42a(changed)


if __name__ == "__main__":
    unittest.main()
