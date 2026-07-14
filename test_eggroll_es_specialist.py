import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from train_eggroll_es_specialist import (
    build_train_loader,
    eval_metrics,
    exact_step_seed,
    extract_answer,
    specialist_collate,
    specialist_reward,
    specialist_template,
)


class EggrollSpecialistAdapterTests(unittest.TestCase):
    def test_exact_step_seed_preserves_zero(self):
        self.assertEqual(exact_step_seed(0, 3), 3)
        self.assertEqual(exact_step_seed(None, 3), 45)

    def test_collate_matches_upstream_prompt_target_contract(self):
        prompts, targets = specialist_collate([
            {"question": "What rope?", "answer": "hemp"},
            {"question": "What knot?", "answer": "bowline"},
        ])
        self.assertEqual(prompts, ["What rope?", "What knot?"])
        self.assertEqual(targets, ["hemp", "bowline"])

    def test_template_disables_thinking_and_opens_assistant_role(self):
        prompt = specialist_template("What is a bight?")
        self.assertIn("<|im_start|>user\nWhat is a bight?<|im_end|>", prompt)
        self.assertTrue(prompt.endswith("<think>\n\n</think>\n\n"))

    def test_reward_ignores_protocol_and_thinking_text(self):
        response = "<think>wrong draft</think>\nhemp<|im_end|>"
        self.assertEqual(extract_answer(response), "hemp")
        self.assertEqual(specialist_reward(response, "hemp"), ("exact", 1.0))

    def test_eval_metrics_averages_saved_rewards(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "eval-output"
            output.mkdir()
            (output / "model_eval_taskvalidation_iteration3.json").write_text(
                '[{"reward": 1.0}, {"reward": 0.25}]'
            )
            self.assertEqual(
                eval_metrics(directory, 2, ["validation"]),
                {"validation": 0.625},
            )

    def test_train_shuffle_is_independent_of_global_torch_rng(self):
        rows = [
            {"question": str(index), "answer": str(index)}
            for index in range(12)
        ]
        first = build_train_loader(rows, batch_size=4, seed=42)
        torch.manual_seed(999)
        torch.rand(100)
        second = build_train_loader(rows, batch_size=4, seed=42)
        first_order = [question for batch, _ in first for question in batch]
        torch.manual_seed(123)
        torch.rand(200)
        second_order = [question for batch, _ in second for question in batch]
        self.assertEqual(first_order, second_order)


if __name__ == "__main__":
    unittest.main()
