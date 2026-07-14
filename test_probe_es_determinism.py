import unittest

from probe_es_determinism import prediction_sha256, score_splits, verify_server_modes


class DeterminismProbeTests(unittest.TestCase):
    def test_prediction_hash_is_order_sensitive_and_unicode_stable(self):
        first = prediction_sha256(["縄", "rope"])
        self.assertEqual(first, prediction_sha256(["縄", "rope"]))
        self.assertNotEqual(first, prediction_sha256(["rope", "縄"]))

    def test_server_mode_requires_deterministic_pytorch_sampling(self):
        valid = {
            "port": 30001,
            "model_path": "model",
            "version": "v1",
            "enable_deterministic_inference": True,
            "sampling_backend": "pytorch",
            "attention_backend": "flashinfer",
            "disable_radix_cache": True,
        }
        verify_server_modes([valid, {**valid, "port": 30002}])
        with self.assertRaisesRegex(RuntimeError, "not in deterministic"):
            verify_server_modes([{**valid, "enable_deterministic_inference": False}])

    def test_score_splits_uses_accuracy_reward(self):
        items = [
            {"split": "train", "answer": "rope"},
            {"split": "heldout", "answer": "jute rope"},
        ]
        scores = score_splits(items, ["rope", "jute"])
        self.assertEqual(scores["train"], 1.0)
        self.assertGreater(scores["heldout"], 0.0)
        self.assertLess(scores["heldout"], 1.0)


if __name__ == "__main__":
    unittest.main()
