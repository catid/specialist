#!/usr/bin/env python3

import unittest

import run_eggroll_es_equal_unit_v38a as subject


class RuntimeV38ATest(unittest.TestCase):
    def test_worker_identity_validation(self):
        rows = [{
            "schema": "eggroll-es-worker-runtime-identity-v38a",
            "pid": 100 + index,
            "inter_engine_rank": index,
            "cuda_visible_devices": str(index),
            "cuda_current_device": 0,
        } for index in range(4)]
        self.assertEqual(subject.validate_worker_identities(rows), {
            0: 100, 1: 101, 2: 102, 3: 103,
        })

    def test_fixed_artifact_paths(self):
        self.assertNotIn("eval", subject.EXPERIMENT)
        self.assertNotIn("holdout", subject.EXPERIMENT)
        self.assertEqual(subject.EXPECTED_GPU_IDS, (0, 1, 2, 3))

    def test_sealed_update_gate_rejects_unchanged_state(self):
        with self.assertRaises(RuntimeError):
            subject.validate_sealed_update({
                "status": "one_nonzero_update_sealed_train_only",
                "alpha": 0.00015,
                "sigma": 0.0003,
                "coefficients": [1.0] * 32,
                "standardization": {"zero_spread": False},
                "application": {
                    "target_alpha": 0.00015,
                    "update_sequence": 1,
                    "manifest": {"expected_base_sha256": "same"},
                    "final_identity": {
                        "sha256": "same", "unselected": {"sha256": "u"},
                    },
                    "commits": [{"committed": True}] * 4,
                    "post_commit_states": [{}] * 4,
                },
                "snapshot_reports": [{
                    "written": True, "rank": 0, "tensor_count": 23,
                    "tensor_elements": 142_999_552,
                }],
            })


if __name__ == "__main__":
    unittest.main()
