#!/usr/bin/env python3

import unittest
from unittest import mock

import torch
from safetensors.torch import load_file

import sft_lora_equal_unit_matched_init_v42b as subject


class DirectCanonicalLoadV42BTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = load_file(
            str(
                subject.v42a.INITIAL_ADAPTER_V42A
                / "adapter_model.safetensors"
            ),
            device="cpu",
        )

    def test_exact_inplace_copy_and_readback(self):
        runtime = {
            key: torch.zeros_like(value) for key, value in self.source.items()
        }
        with mock.patch.object(
            subject,
            "_canonical_adapter_state_v42b",
            return_value=runtime,
        ):
            audit = subject.copy_exact_canonical_state_v42b(
                object(), self.source
            )
        self.assertTrue(audit["verified"])
        self.assertFalse(audit["peft_pretrained_conversion_invoked"])
        self.assertFalse(audit["set_peft_model_state_dict_invoked"])
        self.assertEqual(audit["tensor_count"], 70)
        self.assertEqual(audit["elements"], 4_528_128)
        for key in self.source:
            self.assertTrue(torch.equal(runtime[key], self.source[key]))

    def test_metadata_mismatch_fails_before_copy(self):
        runtime = dict(self.source)
        key = sorted(runtime)[0]
        runtime[key] = runtime[key].to(torch.bfloat16)
        with mock.patch.object(
            subject,
            "_canonical_adapter_state_v42b",
            return_value=runtime,
        ):
            with self.assertRaisesRegex(RuntimeError, "metadata mismatch"):
                subject.copy_exact_canonical_state_v42b(object(), self.source)

    def test_expected_audit_binds_broken_path_bypass(self):
        audit = subject.expected_loader_audit_v42b()
        self.assertEqual(
            audit["loader"],
            "get_peft_model_plus_inplace_canonical_state_copy",
        )
        self.assertIn("distributed_operation", audit["compatibility_reason"])
        self.assertEqual(
            audit["source_weights_file_sha256"],
            subject.v42a.INITIAL_WEIGHTS_SHA256_V42A,
        )


if __name__ == "__main__":
    unittest.main()
