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


if __name__ == "__main__":
    unittest.main()
