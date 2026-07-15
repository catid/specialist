#!/usr/bin/env python3

import copy
import types
import unittest

import torch

import build_train_shadow_folds_v37a as shadow
import sft_lora_equal_unit_v37a as subject


class EqualUnitSFTV37ATest(unittest.TestCase):
    def test_fold_three_weights_are_exactly_equal_by_unit(self):
        rows = [
            subject.json.loads(line)
            for line in (
                shadow.OUTPUT_DIR / "fold_3_train.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line
        ]
        weights, audit = subject.assign_equal_unit_weights(copy.deepcopy(rows))
        self.assertEqual(audit["rows"], 448)
        self.assertEqual(audit["conflict_units"], 208)
        self.assertAlmostEqual(sum(weights), 448.0)
        units = shadow.build_conflict_units(rows)
        masses = [sum(weights[index] for index in unit["indices"]) for unit in units]
        self.assertTrue(all(abs(value - 448 / 208) < 1e-12 for value in masses))

    def test_weighted_answer_only_loss(self):
        labels = torch.tensor([[-100, -100, 1], [-100, -100, 0]])
        logits = torch.tensor([
            [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]],
            [[0.0, 0.0], [0.0, 4.0], [4.0, 0.0]],
        ])

        class Model:
            def __call__(self, **kwargs):
                del kwargs
                return types.SimpleNamespace(logits=logits)

        inputs = {
            "input_ids": torch.ones_like(labels),
            "labels": labels,
            "example_weight": torch.tensor([0.5, 1.5]),
        }
        observed = subject.EqualUnitTrainer.compute_loss(None, Model(), inputs)
        first = torch.nn.functional.cross_entropy(logits[0, 1:2], labels[0, 2:3])
        second = torch.nn.functional.cross_entropy(logits[1, 1:2], labels[1, 2:3])
        expected = (0.5 * first + 1.5 * second) / 2
        self.assertTrue(torch.allclose(observed, expected))


if __name__ == "__main__":
    unittest.main()
