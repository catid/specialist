#!/usr/bin/env python3

import unittest

import build_sealed_candidate_eval_preregistration_v39a as subject


class PreregistrationV39ATest(unittest.TestCase):
    def test_frozen_four_arm_aggregate_only_protocol(self):
        value = subject.build()
        self.assertEqual(list(value["arms"]), ["base_a", "base_b", "sft_v37a", "es_v38a"])
        self.assertEqual(value["shadow_protocol"]["rows"], 83)
        self.assertEqual(value["shadow_protocol"]["conflict_units"], 51)
        self.assertFalse(value["heldout_or_holdout_access_authorized"])
        self.assertTrue(value["firewall"]["aggregate_only"])
        self.assertEqual(
            value["content_sha256_before_self_field"],
            subject.runtime.canonical_sha256({
                key: item for key, item in value.items()
                if key != "content_sha256_before_self_field"
            }),
        )


if __name__ == "__main__":
    unittest.main()
