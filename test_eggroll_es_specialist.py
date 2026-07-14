import unittest
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from train_eggroll_es_specialist import (
    build_train_loader,
    compare_ood_prose,
    dispatch_ood_prose,
    eval_metrics,
    exact_step_seed,
    extract_answer,
    load_ood_prose,
    prepare_ood_prose_items,
    prompt_token_logprobs,
    run_exact_steps,
    specialist_collate,
    specialist_reward,
    specialist_template,
    summarize_ood_prose,
)


class FakeRemoteGenerate:
    def __init__(self, engine_index, calls):
        self.engine_index = engine_index
        self.calls = calls

    def remote(self, prompts, sampling_params, use_tqdm):
        self.calls.append((
            self.engine_index, prompts, sampling_params, use_tqdm,
        ))
        return [
            f"engine{self.engine_index}: {prompt['prompt_token_ids'][0]}"
            for prompt in prompts
        ]


class FakeEngine:
    def __init__(self, engine_index, calls):
        self.generate = FakeRemoteGenerate(engine_index, calls)


class FakeTokenizer:
    def encode(self, text, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("the prose gate must not add tokens")
        return [ord(character) for character in text]


class FakePool:
    def close(self):
        pass

    def join(self):
        pass


class FakeExactTrainer:
    def __init__(self, logging_dir):
        self.logging_dir = str(logging_dir)
        self.model_name = "fake/model"
        self.sigma = 0.001
        self.alpha = 0.0005
        self.population_size = 8
        self.batch_size = 4
        self.mini_batch_size = 4
        self.max_tokens = 16
        self.global_seed = 42
        self.eval_dataloader_dict = {"validation": []}
        self.train_dataloader = []
        self.logging = "none"
        self.mp_pool = FakePool()
        self.cleaned_up = False

    def eval_step(self, iteration):
        output = Path(self.logging_dir) / "eval-output"
        output.mkdir(parents=True, exist_ok=True)
        filename = f"model_eval_taskvalidation_iteration{iteration + 1}.json"
        path = output / filename
        path.write_text('[{"reward": 0.5}]')

    def cleanup(self):
        self.cleaned_up = True


def fake_output(token_ids, logprobs):
    positions = [None]
    positions.extend({
        token_id: SimpleNamespace(logprob=logprob)
    }
        for token_id, logprob in zip(token_ids[1:], logprobs)
    )
    return SimpleNamespace(
        prompt_token_ids=token_ids,
        prompt_logprobs=positions,
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

    def test_load_ood_prose_validates_and_hashes_exact_artifact(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prose.jsonl"
            raw = (
                '{"item_id":"one","split":"ood_prose","text":"abc"}\n'
                '{"item_id":"two","split":"ood_prose","text":"def"}\n'
            ).encode("utf-8")
            path.write_bytes(raw)
            dataset = load_ood_prose(path)
            self.assertEqual(
                dataset["sha256"], hashlib.sha256(raw).hexdigest(),
            )
            self.assertEqual(
                [row["item_id"] for row in dataset["rows"]],
                ["one", "two"],
            )

    def test_load_ood_prose_rejects_duplicate_ids(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prose.jsonl"
            row = {"item_id": "same", "split": "ood_prose", "text": "ab"}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_ood_prose(path)

    def test_prepare_ood_prose_is_exact_and_never_truncates(self):
        rows = [{
            "item_id": "one", "text": "abc", "split": "ood_prose",
            "source": "test",
        }]
        prepared = prepare_ood_prose_items(rows, FakeTokenizer(), 3)
        self.assertEqual(prepared[0]["prompt_token_ids"], [97, 98, 99])
        with self.assertRaisesRegex(ValueError, "above the explicit cap"):
            prepare_ood_prose_items(rows, FakeTokenizer(), 2)

    def test_dispatch_uses_all_engines_and_restores_item_order(self):
        calls = []
        engines = [FakeEngine(index, calls) for index in range(3)]
        items = [
            {"prompt_token_ids": [index, index + 10]}
            for index in range(6)
        ]
        outputs = dispatch_ood_prose(
            engines, items, "params", lambda handles: handles,
        )
        self.assertEqual(
            outputs,
            ["engine0: 0", "engine1: 1", "engine2: 2",
             "engine0: 3", "engine1: 4", "engine2: 5"],
        )
        self.assertEqual([call[0] for call in calls], [0, 1, 2])
        self.assertTrue(all(call[3] is False for call in calls))

    def test_prompt_logprobs_skip_only_contextless_first_token(self):
        output = fake_output([11, 12, 13], [-0.25, -1.5])
        self.assertEqual(
            prompt_token_logprobs(output, [11, 12, 13]),
            [-0.25, -1.5],
        )

    def test_prompt_logprobs_reject_misaligned_engine_output(self):
        output = fake_output([11, 99], [-0.25])
        with self.assertRaisesRegex(ValueError, "different"):
            prompt_token_logprobs(output, [11, 12])

    def test_summary_is_token_weighted_not_mean_of_item_means(self):
        items = [
            {
                "item_id": "short", "source": "x", "title": None,
                "url": None, "normalized_source_url": None,
                "text_sha256": "text-a", "token_ids_sha256": "tokens-a",
                "prompt_token_ids": [1, 2],
            },
            {
                "item_id": "long", "source": "x", "title": None,
                "url": None, "normalized_source_url": None,
                "text_sha256": "text-b", "token_ids_sha256": "tokens-b",
                "prompt_token_ids": [3, 4, 5, 6],
            },
        ]
        outputs = [
            fake_output([1, 2], [-1.0]),
            fake_output([3, 4, 5, 6], [-3.0, -3.0, -3.0]),
        ]
        summary = summarize_ood_prose(items, outputs)
        self.assertEqual(summary["scored_token_count"], 4)
        self.assertEqual(summary["mean_token_logprob"], -2.5)
        self.assertNotEqual(summary["mean_token_logprob"], -2.0)

    def test_gate_requires_alignment_and_applies_tolerance(self):
        baseline = {
            "mean_token_logprob": -2.0,
            "items": [{
                "item_id": "one", "text_sha256": "text",
                "token_ids_sha256": "tokens", "prompt_token_count": 3,
                "scored_token_count": 2,
            }],
        }
        final = {
            "mean_token_logprob": -2.01,
            "items": [dict(baseline["items"][0])],
        }
        self.assertFalse(compare_ood_prose(baseline, final, 0.0)["passed"])
        self.assertTrue(compare_ood_prose(baseline, final, 0.02)["passed"])
        final["items"][0]["item_id"] = "other"
        with self.assertRaisesRegex(ValueError, "not aligned"):
            compare_ood_prose(baseline, final, 0.02)

    def test_exact_run_writes_aligned_ood_gate_to_summary(self):
        aligned_item = {
            "item_id": "one", "text_sha256": "text",
            "token_ids_sha256": "tokens", "prompt_token_count": 3,
            "scored_token_count": 2,
        }
        baseline = {
            "item_count": 1, "scored_token_count": 2,
            "sum_token_logprob": -4.0, "mean_token_logprob": -2.0,
            "items": [dict(aligned_item)],
            "results_path": "baseline.json",
        }
        final = {
            "item_count": 1, "scored_token_count": 2,
            "sum_token_logprob": -4.02, "mean_token_logprob": -2.01,
            "items": [dict(aligned_item)],
            "results_path": "final.json",
        }
        dataset = {
            "path": "/frozen/ood.jsonl", "sha256": "dataset-sha",
            "rows": [{"item_id": "one"}],
        }
        with TemporaryDirectory() as directory:
            trainer = FakeExactTrainer(directory)
            with patch(
                "train_eggroll_es_specialist.score_ood_prose",
                side_effect=[baseline, final],
            ):
                summary = run_exact_steps(
                    trainer, 0, ood_prose=dataset,
                    max_ood_prose_degradation=0.02,
                )
            saved = json.loads(
                (Path(directory) / "run_summary.json").read_text()
            )
        self.assertTrue(trainer.cleaned_up)
        self.assertEqual(summary, saved)
        self.assertEqual(
            saved["ood_prose"]["evaluations"]["baseline"]["items"],
            saved["ood_prose"]["evaluations"]["final"]["items"],
        )
        self.assertAlmostEqual(saved["ood_prose"]["gate"]["delta"], -0.01)
        self.assertTrue(saved["ood_prose"]["gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
