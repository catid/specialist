import unittest
import hashlib
import json
import random
from multiprocessing import TimeoutError as PoolTimeoutError
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from train_eggroll_es_specialist import (
    atomic_json_write,
    bounded_ray_get,
    build_train_loader,
    close_trainer_preserving_primary,
    compare_ood_prose,
    dispatch_qa_eval,
    dispatch_ood_prose,
    eval_metrics,
    exact_step_seed,
    extract_answer,
    fixed_ray_get_timeout,
    load_ood_prose,
    maybe_save_best_checkpoint_atomic,
    prepare_ood_prose_items,
    prompt_token_logprobs,
    run_standard_fit_exact,
    save_checkpoint_atomic,
    run_exact_steps,
    safe_postprocess_outputs,
    specialist_collate,
    specialist_reward,
    specialist_template,
    summarize_ood_prose,
    unique_population_seeds,
    validate_population_seeds,
    OODProseGateFailure,
    _TruthyZero,
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


class FakeQARemoteGenerate:
    def __init__(self, engine_index, calls):
        self.engine_index = engine_index
        self.calls = calls

    def remote(self, prompts, sampling_params, use_tqdm):
        self.calls.append((
            self.engine_index, list(prompts), sampling_params, use_tqdm,
        ))
        return [
            f"engine{self.engine_index}: {prompt}"
            for prompt in prompts
        ]


class FakeQAEngine:
    def __init__(self, engine_index, calls):
        self.generate = FakeQARemoteGenerate(engine_index, calls)


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
        self.assertEqual((_TruthyZero() or 42) + 3, 3)

    def test_population_seeds_are_unique_without_reordering_first_draws(self):
        class CollisionRng:
            def __init__(self):
                self.calls = 0

            def integers(self, _low, _high, size, dtype):
                del dtype
                self.calls += 1
                values = [7, 7, 9] if self.calls == 1 else [11]
                self.assert_size = size
                return np.asarray(values, dtype=np.int64)

        rng = CollisionRng()
        seeds = unique_population_seeds(rng, 3)
        self.assertEqual(seeds, [7, 9, 11])
        self.assertEqual(validate_population_seeds(seeds, 3), seeds)
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_population_seeds([7, 7, 9], 3)

    def test_bounded_ray_get_translates_provider_timeout(self):
        class GetTimeoutError(Exception):
            pass

        observed = []

        def get(_handle, timeout=None):
            observed.append(timeout)
            raise GetTimeoutError("slow actor")

        fake_ray = SimpleNamespace(
            get=get,
            exceptions=SimpleNamespace(GetTimeoutError=GetTimeoutError),
        )
        with self.assertRaisesRegex(TimeoutError, "test actor"):
            bounded_ray_get(fake_ray, "ref", "test actor", timeout=7.5)
        self.assertEqual(observed, [7.5])

    def test_fixed_ray_get_context_caps_inherited_waits_and_restores(self):
        observed = []

        def original(handle, timeout=None):
            observed.append((handle, timeout))
            return handle

        fake_ray = SimpleNamespace(get=original)
        with fixed_ray_get_timeout(
            fake_ray, timeout=7.5, context="inherited actor wait",
        ):
            self.assertEqual(fake_ray.get("a"), "a")
            self.assertEqual(fake_ray.get("b", timeout=20.0), "b")
        self.assertIs(fake_ray.get, original)
        self.assertEqual(observed, [("a", 7.5), ("b", 7.5)])

    def test_training_scoring_skips_unused_decode(self):
        class Result:
            @staticmethod
            def get(timeout):
                del timeout
                return "exact", 1.0

        trainer = SimpleNamespace(
            tokenizer=SimpleNamespace(
                decode=lambda *_args, **_kwargs: self.fail(
                    "training path decoded an unused response"
                )
            ),
            mp_pool=SimpleNamespace(
                apply_async=lambda *_args, **_kwargs: Result(),
            ),
            task=lambda *_args: None, reward_function_timeout=1,
            rollout_reduce="mean", n_samples=1,
        )
        generation = SimpleNamespace(
            prompt="p", outputs=[SimpleNamespace(text="r", token_ids=[1])],
        )
        result = safe_postprocess_outputs(trainer, [generation], ["a"])
        self.assertEqual(result["rewards"], [1.0])

    def test_reward_timeout_rows_are_zero_not_unbound_or_stale(self):
        class AsyncResult:
            def __init__(self, value):
                self.value = value

            def get(self, timeout):
                self.timeout = timeout
                if isinstance(self.value, BaseException):
                    raise self.value
                return self.value

        class Pool:
            def __init__(self):
                self.values = iter([
                    PoolTimeoutError("slow-first"),
                    ("exact", 1.0), PoolTimeoutError("slow-after-success"),
                ])

            def apply_async(self, _task, _args):
                return AsyncResult(next(self.values))

        trainer = SimpleNamespace(
            tokenizer=SimpleNamespace(
                decode=lambda ids, skip_special_tokens: str(ids)
            ),
            mp_pool=Pool(), task=lambda *_args: None,
            reward_function_timeout=1, rollout_reduce="mean", n_samples=1,
        )
        generations = [
            SimpleNamespace(
                prompt=f"p{index}", outputs=[SimpleNamespace(
                    text=f"r{index}", token_ids=[index],
                )],
            ) for index in range(3)
        ]
        result = safe_postprocess_outputs(
            trainer, generations, ["a", "b", "c"], eval=True,
        )
        self.assertEqual(result["rewards"], [0.0, 1.0, 0.0])
        self.assertEqual(result["results"][0]["format"], "timeout")
        self.assertEqual(result["results"][2]["reward"], 0.0)
        self.assertEqual(result["results"][2]["format"], "timeout")
        with self.assertRaisesRegex(ValueError, "batch sizes"):
            safe_postprocess_outputs(trainer, generations, ["a"], eval=True)
        generations[0].outputs = []
        with self.assertRaisesRegex(RuntimeError, "rollout cardinality"):
            safe_postprocess_outputs(trainer, generations, ["a", "b", "c"])

    def test_default_fit_executes_exactly_requested_updates(self):
        calls = []

        class Remote:
            def remote(self, method, args):
                calls.append((method, args))
                Path(args[0]).write_bytes(b"checkpoint")
                return "saved"

        with TemporaryDirectory() as directory:
            trainer = SimpleNamespace(
                num_iterations=3,
                train_dataloader=[(["q"], ["a"])],
                template=lambda text: f"prompt:{text}",
                global_seed=0, population_size=4, eval_freq=2,
                logging_dir=directory,
                engines=[SimpleNamespace(collective_rpc=Remote())],
            )
            trainer.eval_calls = []
            trainer.train_calls = []
            trainer.eval_step = lambda iteration: trainer.eval_calls.append(iteration)

            def train_step(**kwargs):
                validate_population_seeds(kwargs["seeds"], 4)
                trainer.train_calls.append(kwargs)

            trainer.train_step = train_step
            fake_ray = SimpleNamespace(get=lambda handle, timeout=None: handle)
            with patch.dict("sys.modules", {"ray": fake_ray}):
                checkpoint = run_standard_fit_exact(trainer)
            self.assertTrue(checkpoint.is_file())
        self.assertEqual(
            [row["iteration"] for row in trainer.train_calls], [0, 1, 2],
        )
        self.assertEqual(trainer.eval_calls, [0, 2, 3])
        self.assertTrue(str(checkpoint).endswith(
            "checkpoint-es_fine_tuned_iteration_3/pytorch_model.pth"
        ))
        self.assertEqual(calls[0][0], "save_self_weights_to_disk")

    def test_default_fit_does_not_overwrite_baseline_when_eval_freq_is_one(self):
        class Remote:
            def remote(self, _method, args):
                Path(args[0]).write_bytes(b"checkpoint")
                return "saved"

        with TemporaryDirectory() as directory:
            trainer = SimpleNamespace(
                num_iterations=1,
                train_dataloader=[(["q"], ["a"])],
                template=lambda text: text,
                global_seed=0, population_size=2, eval_freq=1,
                logging_dir=directory,
                engines=[SimpleNamespace(collective_rpc=Remote())],
                eval_calls=[], train_calls=[],
            )
            trainer.eval_step = lambda iteration: trainer.eval_calls.append(iteration)
            trainer.train_step = lambda **kwargs: trainer.train_calls.append(kwargs)
            fake_ray = SimpleNamespace(get=lambda handle, timeout=None: handle)
            with patch.dict("sys.modules", {"ray": fake_ray}):
                run_standard_fit_exact(trainer)
        self.assertEqual(trainer.eval_calls, [0, 1])
        self.assertEqual(len(trainer.train_calls), 1)

    def test_default_fit_skips_duplicate_periodic_final_evaluation(self):
        class Remote:
            def remote(self, _method, args):
                Path(args[0]).write_bytes(b"checkpoint")
                return "saved"

        with TemporaryDirectory() as directory:
            trainer = SimpleNamespace(
                num_iterations=2,
                train_dataloader=[(["q"], ["a"])],
                template=lambda text: text,
                global_seed=0, population_size=2, eval_freq=2,
                logging_dir=directory,
                engines=[SimpleNamespace(collective_rpc=Remote())],
                eval_calls=[], train_calls=[],
            )
            trainer.eval_step = lambda iteration: trainer.eval_calls.append(iteration)
            trainer.train_step = lambda **kwargs: trainer.train_calls.append(kwargs)
            fake_ray = SimpleNamespace(
                get=lambda handle, timeout=None: handle,
            )
            with patch.dict("sys.modules", {"ray": fake_ray}):
                run_standard_fit_exact(trainer)
        self.assertEqual(trainer.eval_calls, [0, 2])
        self.assertEqual(len(trainer.train_calls), 2)

    def test_default_fit_eval_freq_one_covers_every_completed_update(self):
        class Remote:
            def remote(self, _method, args):
                Path(args[0]).write_bytes(b"checkpoint")
                return "saved"

        with TemporaryDirectory() as directory:
            trainer = SimpleNamespace(
                num_iterations=3,
                train_dataloader=[(["q"], ["a"])],
                template=lambda text: text,
                global_seed=0, population_size=2, eval_freq=1,
                logging_dir=directory,
                engines=[SimpleNamespace(collective_rpc=Remote())],
                eval_calls=[], train_calls=[],
            )
            trainer.eval_step = lambda iteration: trainer.eval_calls.append(
                iteration
            )
            trainer.train_step = lambda **kwargs: trainer.train_calls.append(
                kwargs
            )
            fake_ray = SimpleNamespace(
                get=lambda handle, timeout=None: handle,
            )
            with patch.dict("sys.modules", {"ray": fake_ray}):
                run_standard_fit_exact(trainer)
        self.assertEqual(trainer.eval_calls, [0, 1, 2, 3])
        self.assertEqual(len(trainer.train_calls), 3)

    def test_best_checkpoint_uses_atomic_checkpoint_publisher(self):
        calls = []

        class Remote:
            def remote(self, method, args):
                calls.append((method, args))
                Path(args[0]).write_bytes(b"best")
                return "saved"

        with TemporaryDirectory() as directory:
            output = Path(directory) / "eval-output"
            output.mkdir()
            (output / "model_eval_taskvalidation_iteration1.json").write_text(
                '[{"reward": 0.75}]\n', encoding="utf-8",
            )
            trainer = SimpleNamespace(
                logging_dir=directory,
                eval_dataloader_dict={"validation": []},
                best_avg=float("-inf"), experiment_name="trial",
                engines=[SimpleNamespace(collective_rpc=Remote())],
            )
            fake_ray = SimpleNamespace(
                get=lambda handle, timeout=None: handle,
            )
            with patch.dict("sys.modules", {"ray": fake_ray}):
                checkpoint = maybe_save_best_checkpoint_atomic(trainer, 0)
                duplicate = maybe_save_best_checkpoint_atomic(trainer, 0)
            self.assertTrue(checkpoint.is_file())
            self.assertIsNone(duplicate)
            self.assertEqual(trainer.best_avg, 0.75)
            self.assertEqual(len(calls), 1)

    def test_poisoned_trainer_cannot_publish_checkpoint(self):
        calls = []

        class Remote:
            def remote(self, method, args):
                calls.append((method, args))
                return "saved"

        trainer = SimpleNamespace(
            _specialist_state_poisoned=True,
            engines=[SimpleNamespace(collective_rpc=Remote())],
        )
        with TemporaryDirectory() as directory, self.assertRaisesRegex(
            RuntimeError, "state is poisoned",
        ):
            save_checkpoint_atomic(trainer, Path(directory) / "checkpoint")
        self.assertEqual(calls, [])

    def test_cleanup_failure_does_not_replace_primary_error(self):
        trainer = SimpleNamespace()
        primary = ValueError("training failed")
        with patch(
            "train_eggroll_es_specialist.close_trainer",
            side_effect=RuntimeError("cleanup failed"),
        ):
            with self.assertRaises(ValueError) as raised:
                try:
                    raise primary
                finally:
                    close_trainer_preserving_primary(trainer)
        self.assertIs(raised.exception, primary)
        self.assertEqual(
            primary.__notes__,
            ["trainer cleanup also failed: RuntimeError"],
        )

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
                '{"item_id":"one","split":"ood_prose","text":"abc",'
                '"normalized_source_url":"doc://one"}\n'
                '{"item_id":"two","split":"ood_prose","text":"def",'
                '"normalized_source_url":"doc://two"}\n'
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
            row = {
                "item_id": "same", "split": "ood_prose", "text": "ab",
                "normalized_source_url": "doc://same",
            }
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_ood_prose(path)

    def test_load_ood_prose_rejects_missing_source_identity(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prose.jsonl"
            path.write_text(
                json.dumps({
                    "item_id": "one", "split": "ood_prose", "text": "ab",
                }) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "normalized_source_url"):
                load_ood_prose(path)

    def test_frozen_ood_prose_items_are_unique_source_documents(self):
        path = Path(__file__).resolve().parent / "data/ood_prose_v3.jsonl"
        rows = load_ood_prose(path)["rows"]
        document_ids = [row.get("normalized_source_url") for row in rows]
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(
            isinstance(document_id, str) and document_id
            for document_id in document_ids
        ))
        self.assertEqual(len(set(document_ids)), len(document_ids))

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

    def test_qa_eval_dispatches_uneven_batch_across_all_engines_in_order(self):
        calls = []
        engines = [FakeQAEngine(index, calls) for index in range(4)]
        prompts = [f"prompt-{index}" for index in range(10)]
        outputs = dispatch_qa_eval(
            engines, prompts, "params", lambda handles: handles,
        )
        self.assertEqual(outputs, [
            f"engine{index % 4}: prompt-{index}"
            for index in range(10)
        ])
        self.assertEqual([call[0] for call in calls], [0, 1, 2, 3])
        self.assertEqual(
            [call[1] for call in calls],
            [
                ["prompt-0", "prompt-4", "prompt-8"],
                ["prompt-1", "prompt-5", "prompt-9"],
                ["prompt-2", "prompt-6"],
                ["prompt-3", "prompt-7"],
            ],
        )
        self.assertTrue(all(call[3] is False for call in calls))

    def test_qa_eval_rejects_changed_engine_cardinality(self):
        calls = []
        engines = [FakeQAEngine(index, calls) for index in range(2)]

        def drop_one_result(handles):
            handles[0].pop()
            return handles

        with self.assertRaisesRegex(RuntimeError, "request count"):
            dispatch_qa_eval(
                engines, ["a", "b", "c"], "params", drop_one_result,
            )

    def test_atomic_json_failure_preserves_previous_eval_artifact(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "eval-output" / "result.json"
            path.parent.mkdir()
            path.write_text('[{"reward": 0.25}]\n', encoding="utf-8")
            with patch(
                "train_eggroll_es_specialist.os.replace",
                side_effect=OSError("rename failed"),
            ), self.assertRaisesRegex(OSError, "rename failed"):
                atomic_json_write(path, [{"reward": 1.0}])
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                '[{"reward": 0.25}]\n',
            )
            self.assertEqual(list(path.parent.iterdir()), [path])

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
                "item_id": "one", "normalized_source_url": "doc://one",
                "text_sha256": "text",
                "token_ids_sha256": "tokens", "prompt_token_count": 3,
                "scored_token_count": 2, "sum_token_logprob": -4.0,
            }],
        }
        final = {
            "mean_token_logprob": -2.01,
            "items": [dict(baseline["items"][0])],
        }
        final["items"][0]["sum_token_logprob"] = -4.02
        self.assertFalse(compare_ood_prose(baseline, final, 0.0)["passed"])
        self.assertTrue(compare_ood_prose(baseline, final, 0.02)["passed"])
        final["items"][0]["item_id"] = "other"
        with self.assertRaisesRegex(ValueError, "not aligned"):
            compare_ood_prose(baseline, final, 0.02)

    def test_document_bootstrap_is_token_weighted_and_reproducible(self):
        baseline_items = [
            {
                "item_id": "a", "normalized_source_url": "doc://a",
                "text_sha256": "a", "token_ids_sha256": "a-tokens",
                "prompt_token_count": 2, "scored_token_count": 1,
                "sum_token_logprob": -1.0,
            },
            {
                "item_id": "b", "normalized_source_url": "doc://b",
                "text_sha256": "b", "token_ids_sha256": "b-tokens",
                "prompt_token_count": 4, "scored_token_count": 3,
                "sum_token_logprob": -3.0,
            },
        ]
        final_items = [dict(item) for item in baseline_items]
        final_items[0]["sum_token_logprob"] = -0.5
        final_items[1]["sum_token_logprob"] = -3.6
        baseline = {"mean_token_logprob": -1.0, "items": baseline_items}
        final = {"mean_token_logprob": -1.025, "items": final_items}

        random_state = random.getstate()
        first = compare_ood_prose(baseline, final)
        second = compare_ood_prose(baseline, final)

        self.assertEqual(random.getstate(), random_state)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["delta"], -0.025)
        self.assertNotAlmostEqual(first["delta"], 0.15)
        self.assertAlmostEqual(
            first["paired_document_bootstrap_95_ci"][0], -0.2,
        )
        self.assertAlmostEqual(
            first["paired_document_bootstrap_95_ci"][1], 0.5,
        )
        self.assertEqual(first["bootstrap"]["samples"], 20000)
        self.assertEqual(first["bootstrap"]["seed"], 20260714)

    def test_bootstrap_groups_multiple_items_from_one_document(self):
        baseline_items = []
        for item_id, document_id, token_count, logprob_sum in (
            ("a1", "doc://a", 1, -1.0),
            ("a2", "doc://a", 2, -2.0),
            ("b1", "doc://b", 3, -3.0),
        ):
            baseline_items.append({
                "item_id": item_id,
                "normalized_source_url": document_id,
                "text_sha256": f"text-{item_id}",
                "token_ids_sha256": f"tokens-{item_id}",
                "prompt_token_count": token_count + 1,
                "scored_token_count": token_count,
                "sum_token_logprob": logprob_sum,
            })
        final_items = [dict(item) for item in baseline_items]
        report = compare_ood_prose(
            {"mean_token_logprob": -1.0, "items": baseline_items},
            {"mean_token_logprob": -1.0, "items": final_items},
            bootstrap_samples=100,
        )
        self.assertEqual(report["bootstrap"]["document_count"], 2)
        self.assertEqual(report["paired_document_bootstrap_95_ci"], [0.0, 0.0])

    def test_bootstrap_gate_uses_ci_lower_bound_and_inclusive_margin(self):
        baseline_item = {
            "item_id": "one", "normalized_source_url": "doc://one",
            "text_sha256": "text", "token_ids_sha256": "tokens",
            "prompt_token_count": 2, "scored_token_count": 1,
            "sum_token_logprob": 0.0,
        }
        final_item = dict(baseline_item)
        final_item["sum_token_logprob"] = -0.02
        baseline = {"mean_token_logprob": 0.0, "items": [baseline_item]}
        final = {"mean_token_logprob": -0.02, "items": [final_item]}
        self.assertTrue(compare_ood_prose(
            baseline, final, 0.02, bootstrap_samples=10,
        )["passed"])
        self.assertFalse(compare_ood_prose(
            baseline, final, 0.019, bootstrap_samples=10,
        )["passed"])

    def test_bootstrap_fails_closed_on_invalid_inputs(self):
        item = {
            "item_id": "one", "normalized_source_url": "doc://one",
            "text_sha256": "text", "token_ids_sha256": "tokens",
            "prompt_token_count": 2, "scored_token_count": 1,
            "sum_token_logprob": -1.0,
        }
        evaluation = {"mean_token_logprob": -1.0, "items": [item]}
        with self.assertRaisesRegex(ValueError, "samples"):
            compare_ood_prose(
                evaluation, evaluation, bootstrap_samples=0,
            )
        missing_document = dict(item)
        missing_document["normalized_source_url"] = None
        with self.assertRaisesRegex(ValueError, "document identity"):
            compare_ood_prose(
                {"mean_token_logprob": -1.0, "items": [missing_document]},
                {"mean_token_logprob": -1.0, "items": [missing_document]},
                bootstrap_samples=10,
            )

    def test_exact_run_writes_aligned_ood_gate_to_summary(self):
        aligned_item = {
            "item_id": "one", "normalized_source_url": "doc://one",
            "text_sha256": "text",
            "token_ids_sha256": "tokens", "prompt_token_count": 3,
            "scored_token_count": 2, "sum_token_logprob": -4.0,
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
        final["items"][0]["sum_token_logprob"] = -4.02
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
        evaluations = saved["ood_prose"]["evaluations"]
        baseline_item = evaluations["baseline"]["items"][0]
        final_item = evaluations["final"]["items"][0]
        for field in (
            "item_id", "normalized_source_url", "text_sha256",
            "token_ids_sha256", "prompt_token_count", "scored_token_count",
        ):
            self.assertEqual(baseline_item[field], final_item[field])
        self.assertAlmostEqual(saved["ood_prose"]["gate"]["delta"], -0.01)
        self.assertEqual(
            saved["ood_prose"]["schema"],
            "eggroll-es-ood-prose-logprob-v2",
        )
        self.assertEqual(
            saved["ood_prose"]["gate"]["bootstrap"]["samples"], 20000,
        )
        self.assertTrue(saved["ood_prose"]["gate"]["passed"])

    def test_failed_ood_gate_is_nonzero_and_prevents_checkpoint_creation(self):
        item = {
            "item_id": "one", "normalized_source_url": "doc://one",
            "text_sha256": "text", "token_ids_sha256": "tokens",
            "prompt_token_count": 2, "scored_token_count": 1,
            "sum_token_logprob": 0.0,
        }
        baseline = {
            "mean_token_logprob": 0.0, "items": [dict(item)],
            "item_count": 1, "scored_token_count": 1,
            "sum_token_logprob": 0.0, "results_path": "baseline.json",
        }
        final = json.loads(json.dumps(baseline))
        final["mean_token_logprob"] = -1.0
        final["sum_token_logprob"] = -1.0
        final["items"][0]["sum_token_logprob"] = -1.0
        final["results_path"] = "final.json"
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
                with self.assertRaises(OODProseGateFailure):
                    run_exact_steps(
                        trainer, 0, save_final_checkpoint=True,
                        ood_prose=dataset, max_ood_prose_degradation=0.02,
                    )
            summary = json.loads(
                (Path(directory) / "run_summary.json").read_text()
            )
            checkpoint_dir = (
                Path(directory) / "checkpoint-es_exact_steps_0"
            )
        self.assertTrue(trainer.cleaned_up)
        self.assertEqual(summary["status"], "failed_ood_prose_gate")
        self.assertIsNone(summary["checkpoint"])
        self.assertFalse(checkpoint_dir.exists())

    def test_passing_ood_gate_precedes_atomic_checkpoint_publish(self):
        events = []
        item = {
            "item_id": "one", "normalized_source_url": "doc://one",
            "text_sha256": "text", "token_ids_sha256": "tokens",
            "prompt_token_count": 2, "scored_token_count": 1,
            "sum_token_logprob": -1.0,
        }
        evaluation = {
            "mean_token_logprob": -1.0, "items": [dict(item)],
            "item_count": 1, "scored_token_count": 1,
            "sum_token_logprob": -1.0, "results_path": "eval.json",
        }
        dataset = {
            "path": "/frozen/ood.jsonl", "sha256": "dataset-sha",
            "rows": [{"item_id": "one"}],
        }

        class Remote:
            def remote(self, method, args):
                self.method = method
                events.append("save")
                Path(args[0]).write_bytes(b"checkpoint")
                return "saved"

        original_compare = compare_ood_prose

        def ordered_compare(*args, **kwargs):
            events.append("compare")
            return original_compare(*args, **kwargs)

        with TemporaryDirectory() as directory:
            trainer = FakeExactTrainer(directory)
            trainer.engines = [SimpleNamespace(collective_rpc=Remote())]
            fake_ray = SimpleNamespace(
                get=lambda handle, timeout=None: handle,
                shutdown=lambda: None,
            )
            with patch.dict("sys.modules", {"ray": fake_ray}), patch(
                "train_eggroll_es_specialist.score_ood_prose",
                side_effect=[evaluation, evaluation],
            ), patch(
                "train_eggroll_es_specialist.compare_ood_prose",
                side_effect=ordered_compare,
            ):
                summary = run_exact_steps(
                    trainer, 0, save_final_checkpoint=True,
                    ood_prose=dataset, max_ood_prose_degradation=0.0,
                )
            checkpoint = Path(summary["checkpoint"])
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(
                summary["checkpoint_sha256"],
                hashlib.sha256(b"checkpoint").hexdigest(),
            )
        self.assertEqual(events, ["compare", "save"])

    def test_checkpoint_rpc_failure_never_publishes_final_directory(self):
        class Remote:
            def remote(self, method, args):
                del method
                Path(args[0]).write_bytes(b"partial")
                raise RuntimeError("save failed")

        trainer = SimpleNamespace(
            engines=[SimpleNamespace(collective_rpc=Remote())],
        )
        fake_ray = SimpleNamespace(get=lambda handle, timeout=None: handle)
        with TemporaryDirectory() as directory, patch.dict(
            "sys.modules", {"ray": fake_ray},
        ):
            final = Path(directory) / "checkpoint-final"
            with self.assertRaisesRegex(RuntimeError, "save failed"):
                save_checkpoint_atomic(trainer, final)
            self.assertFalse(final.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
