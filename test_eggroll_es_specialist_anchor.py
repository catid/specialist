import hashlib
import json
import random
from types import SimpleNamespace

import pytest

from train_eggroll_es_specialist_anchor import (
    AnchoredStepMixin,
    coefficient_sha256,
    dispatch_eval_batch,
    load_anchor_prose,
    prepare_anchor_items,
    run_exact_steps,
    score_anchor_outputs,
    select_anchor_items,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return [ord(character) for character in text]


def prompt_output(token_ids, value):
    logprobs = [None]
    logprobs.extend({
        token_id: SimpleNamespace(logprob=value)
    } for token_id in token_ids[1:])
    return SimpleNamespace(
        prompt_token_ids=token_ids,
        prompt_logprobs=logprobs,
    )


def anchor_row(index, text="abc"):
    return {
        "document_id": f"local-doc-{index}",
        "item_id": f"item-{index}",
        "split": "anchor_prose",
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def test_anchor_loader_requires_matching_report_pin(tmp_path):
    path = tmp_path / "anchor.jsonl"
    raw = (json.dumps(anchor_row(1)) + "\n").encode()
    path.write_bytes(raw)
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "schema": "general-prose-anchor-build-v1",
        "output_rows": 1,
        "output_sha256": hashlib.sha256(raw).hexdigest(),
        "protected_artifacts": [{"sha256": "protected"}],
    }))
    loaded = load_anchor_prose(path, report)
    assert loaded["sha256"] == hashlib.sha256(raw).hexdigest()
    assert loaded["report"]["protected_artifact_count"] == 1
    bad = json.loads(report.read_text())
    bad["output_sha256"] = "wrong"
    report.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="does not pin"):
        load_anchor_prose(path, report)


def test_prepare_and_selection_are_exact_rotating_and_rng_isolated():
    items = prepare_anchor_items(
        [anchor_row(index, f"text-{index}") for index in range(5)],
        FakeTokenizer(), 20,
    )
    state = random.getstate()
    first = select_anchor_items(items, 0, 2, 42)
    second = select_anchor_items(items, 1, 2, 42)
    repeated = select_anchor_items(items, 0, 2, 42)
    assert [item["item_id"] for item in first] == [
        item["item_id"] for item in repeated
    ]
    assert not ({item["item_id"] for item in first} & {
        item["item_id"] for item in second
    })
    assert random.getstate() == state


def test_anchor_score_is_selected_token_and_token_weighted():
    items = [
        {"prompt_token_ids": [1, 2]},
        {"prompt_token_ids": [3, 4, 5, 6]},
    ]
    outputs = [
        prompt_output([1, 2], -1.0),
        prompt_output([3, 4, 5, 6], -3.0),
    ]
    assert score_anchor_outputs(items, outputs) == pytest.approx(-2.5)


class FakeCollective:
    def __init__(self, engine):
        self.engine = engine

    def remote(self, method, args):
        self.engine.events.append((method, args))
        if method == "perturb_self_weights":
            self.engine.seed = args[0]
        return (method, args)


class FakeGenerate:
    def __init__(self, engine):
        self.engine = engine

    def remote(self, prompts, params, use_tqdm):
        self.engine.events.append((
            "generate", self.engine.seed, prompts, params, use_tqdm,
        ))
        outputs = [
            SimpleNamespace(reward=float(self.engine.seed))
            for prompt in prompts if isinstance(prompt, str)
        ]
        outputs.extend(
            prompt_output(prompt["prompt_token_ids"], -self.engine.seed / 10)
            for prompt in prompts if isinstance(prompt, dict)
        )
        return outputs


class FakeEngine:
    def __init__(self):
        self.events = []
        self.seed = None
        self.collective_rpc = FakeCollective(self)
        self.generate = FakeGenerate(self)


class FakePopulationTrainer(AnchoredStepMixin):
    def __init__(self):
        self.engines = [FakeEngine(), FakeEngine()]
        self.n_vllm_engines = 2
        self.n_samples = 1
        self.global_seed = 42
        self.sigma = 0.001

    def _resolve(self, handles):
        return handles

    def _sampling_params(self, **kwargs):
        return kwargs

    def _postprocess_outputs(self, outputs, targets):
        rewards = [output.reward for output in outputs]
        return {"avg_reward": sum(rewards) / len(rewards)}


def test_domain_and_anchor_use_same_seed_before_restore_on_every_engine():
    trainer = FakePopulationTrainer()
    anchor_items = [
        {"item_id": "anchor", "prompt_token_ids": [1, 2, 3]},
    ]
    domain = object()
    metrics, anchor_scores, _ = trainer._evaluate_population_with_anchor(
        [11, 22], ["domain prompt"], ["answer"], domain,
        anchor_items, iteration=0,
    )
    assert metrics[11]["avg_reward"] == 11.0
    assert metrics[22]["avg_reward"] == 22.0
    assert anchor_scores == pytest.approx({11: -1.1, 22: -2.2})
    for engine, seed in zip(trainer.engines, [11, 22]):
        assert [event[0] for event in engine.events] == [
            "perturb_self_weights", "generate", "restore_self_weights",
        ]
        generate = engine.events[1]
        assert generate[1] == seed
        assert generate[2][0] == "domain prompt"
        assert generate[2][1]["prompt_token_ids"] == [1, 2, 3]
        assert generate[3][0] is domain
        assert generate[3][1]["prompt_logprobs"] == 1


class EvalGenerate:
    def __init__(self, calls):
        self.calls = calls

    def remote(self, prompts, params, use_tqdm):
        self.calls.append(list(prompts))
        return [f"output:{prompt}" for prompt in prompts]


def test_multi_engine_eval_matches_single_engine_and_restores_order():
    prompts = [f"prompt-{index}" for index in range(9)]
    one_calls = []
    one = [SimpleNamespace(generate=EvalGenerate(one_calls))]
    one_result = dispatch_eval_batch(
        one, prompts, "params", lambda value: value,
    )
    four_calls = []
    four = [
        SimpleNamespace(generate=EvalGenerate(four_calls)) for _ in range(4)
    ]
    four_result = dispatch_eval_batch(
        four, prompts, "params", lambda value: value,
    )
    assert four_result == one_result
    assert four_result == [f"output:{prompt}" for prompt in prompts]
    assert len(four_calls) == 4
    assert sorted(prompt for call in four_calls for prompt in call) == sorted(
        prompts
    )


class ApplyRemote:
    def __init__(self, calls):
        self.calls = calls

    def remote(self, method, args):
        self.calls.append((method, args))
        return (method, args)


class ApplyEngine:
    def __init__(self, calls):
        self.collective_rpc = ApplyRemote(calls)


class FakeApplyTrainer(AnchoredStepMixin):
    def __init__(self, tmp_path):
        self.calls = []
        self.engines = [ApplyEngine(self.calls), ApplyEngine(self.calls)]
        self.population_size = 2
        self.logging_dir = str(tmp_path)

    def _resolve(self, handles):
        return handles


def test_fixed_coefficients_support_monotonic_resident_alpha_increments(
    tmp_path, monkeypatch,
):
    trainer = FakeApplyTrainer(tmp_path)
    coefficients = [0.5, -0.5]
    plan = {
        "iteration": 0,
        "seeds": [11, 22],
        "coefficients": coefficients,
        "coefficient_sha256": coefficient_sha256([11, 22], coefficients),
        "applied_alpha": 0.0,
        "applications": [],
    }
    trainer._latest_anchor_plan = plan
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    trainer.apply_seed_coefficients(plan, 0.00005)
    trainer.apply_seed_coefficients(plan, 0.00020)
    updates = [call for call in trainer.calls if call[0] == (
        "update_weights_from_seeds"
    )]
    assert updates[0][1] == ([11, 22], coefficients, 0.00005, 2)
    assert updates[1][1][0:2] == ([11, 22], coefficients)
    assert updates[1][1][2] == pytest.approx(0.00015)
    assert plan["applied_alpha"] == pytest.approx(0.00020)
    with pytest.raises(ValueError, match="monotonic"):
        trainer.apply_seed_coefficients(plan, 0.00010)


def test_applier_rejects_changed_fixed_coefficients(tmp_path):
    trainer = FakeApplyTrainer(tmp_path)
    plan = {
        "iteration": 0,
        "seeds": [11, 22],
        "coefficients": [0.5, -0.5],
        "coefficient_sha256": coefficient_sha256(
            [11, 22], [0.5, -0.5],
        ),
        "applied_alpha": 0.0,
        "applications": [],
    }
    trainer._latest_anchor_plan = plan
    plan["coefficients"][0] = 0.6
    with pytest.raises(ValueError, match="changed"):
        trainer.apply_seed_coefficients(plan, 0.0001)


def test_run_summary_pins_new_base_projection_and_upstream_code(
    tmp_path, monkeypatch,
):
    trainer = SimpleNamespace(
        anchor_dataset={
            "path": "anchor.jsonl", "sha256": "anchor-sha",
            "rows": [anchor_row(1)], "report": {"sha256": "report-sha"},
        },
        anchor_items_per_step=2,
        anchor_max_input_tokens=512,
        min_anchor_cosine=0.1,
        anchor_step_plans=[],
        logging_dir=str(tmp_path),
    )
    monkeypatch.setattr(
        "train_eggroll_es_specialist_anchor.base.run_exact_steps",
        lambda trainer, *args, **kwargs: {"schema": "base"},
    )
    summary = run_exact_steps(trainer, 0)
    assert summary["schema"] == "eggroll-es-anchored-exact-run-v1"
    assert summary["anchor"]["dataset"]["sha256"] == "anchor-sha"
    assert set(summary["implementation"]) == {
        "anchor_trainer", "base_trainer", "coefficient_projection",
        "upstream_trainer", "upstream_worker_extension",
    }
    assert all(
        len(identity["sha256"]) == 64
        for identity in summary["implementation"].values()
    )
