import unittest
from unittest.mock import patch

import es_common
from es_common import centered_ranks, zscore


class RewardShapingTests(unittest.TestCase):
    def test_centered_ranks_all_ties_are_zero(self):
        self.assertEqual(centered_ranks([1.0] * 8), [0.0] * 8)

    def test_centered_ranks_average_ties(self):
        self.assertEqual(
            centered_ranks([1.0, 0.0, 1.0, 2.0]),
            [0.0, -0.5, 0.0, 0.5],
        )

    def test_centered_ranks_preserve_antisymmetry_without_ties(self):
        utilities = centered_ranks([0.0, 1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(utilities), 0.0)
        for actual, expected in zip(utilities, [-0.5, -1 / 6, 1 / 6, 0.5]):
            self.assertAlmostEqual(actual, expected)

    def test_zscore_all_ties_are_zero(self):
        self.assertEqual(zscore([3.0] * 4), [0.0] * 4)

    def test_zscore_is_centered_and_unit_variance(self):
        shaped = zscore([1.0, 2.0, 4.0, 8.0])
        self.assertAlmostEqual(sum(shaped), 0.0)
        self.assertAlmostEqual(sum(value * value for value in shaped) / 4, 1.0)


class ReplicaVerificationTests(unittest.TestCase):
    @staticmethod
    def state(master="master", serving="serving", manifest="manifest"):
        return {
            "manifest": {"target_manifest_hash": manifest, "num_units": 46,
                         "total_parameters": 123},
            "master_digest": {"digest": master},
            "serving_digest": {"digest": serving},
        }

    @patch.object(es_common, "perturb_info")
    def test_identical_replicas_are_accepted(self, info):
        info.side_effect = lambda *_args: self.state()
        result = es_common.verify_replica_state([1, 2, 3, 4], "front", True)
        self.assertEqual(result["num_replicas"], 4)
        self.assertEqual(result["num_units"], 46)

    @patch.object(es_common, "perturb_info")
    def test_master_divergence_is_rejected(self, info):
        states = [self.state(), self.state(master="different")]
        info.side_effect = lambda *_args: states.pop(0)
        with self.assertRaisesRegex(RuntimeError, "FP32 master"):
            es_common.verify_replica_state([1, 2], "front", True)


if __name__ == "__main__":
    unittest.main()
