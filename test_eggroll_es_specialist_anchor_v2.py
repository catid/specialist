import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from eggroll_es_worker_v2 import ExactAuditWorkerExtension
import run_eggroll_es_anchor_line_search_v2 as line_search_v2
import train_eggroll_es_specialist_anchor_v2 as anchor_v2
from train_eggroll_es_specialist_anchor_v2 import (
    ExactRestoredAnchoredStepMixin,
)


def prompt_output(token_ids, value):
    logprobs = [None]
    logprobs.extend({
        token_id: SimpleNamespace(logprob=value)
    } for token_id in token_ids[1:])
    return SimpleNamespace(
        prompt_token_ids=token_ids,
        prompt_logprobs=logprobs,
    )


def test_bf16_add_then_subtract_is_not_an_exact_restore_sentinel():
    original = torch.tensor([0.00166], dtype=torch.bfloat16)
    approximate = original.clone()
    noise = torch.ones_like(approximate)
    approximate.add_(0.0003 * noise)
    approximate.add_(-0.0003 * noise)
    assert not torch.equal(approximate, original)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(
            [0.00166, -0.25, 1.0], dtype=torch.bfloat16,
        ))


def test_worker_exact_reference_restores_both_perturbation_signs():
    worker = object.__new__(ExactAuditWorkerExtension)
    worker.model_runner = SimpleNamespace(model=TinyModel())
    initial = worker.model_runner.model.weight.detach().clone()
    saved = worker.save_self_exact_reference(chunk_bytes=2)
    assert saved["parameter_count"] == 1
    for sign in (1.0, -1.0):
        with torch.no_grad():
            worker.model_runner.model.weight.add_(sign * 0.0003)
        assert not torch.equal(worker.model_runner.model.weight, initial)
        assert worker.restore_self_weights_exact() is True
        assert torch.equal(worker.model_runner.model.weight, initial)
        assert worker.verify_self_exact_reference(chunk_bytes=3)["passed"]
    with pytest.raises(RuntimeError, match="subtractive"):
        worker.restore_self_weights(seed=1, sigma=0.0003)


class FakeCollective:
    def __init__(self, engine):
        self.engine = engine

    def remote(self, method, args):
        self.engine.events.append((method, args))
        if method == "perturb_self_weights":
            self.engine.seed = int(args[0])
            return True
        if method == "restore_self_weights_exact":
            self.engine.seed = None
            return True
        raise AssertionError(f"unexpected collective method {method}")


class FakeGenerate:
    def __init__(self, engine):
        self.engine = engine

    def remote(self, prompts, params, use_tqdm):
        del use_tqdm
        kind = "anchor" if prompts and isinstance(prompts[0], dict) else "domain"
        self.engine.events.append((
            "generate", kind, self.engine.seed, list(prompts), params,
        ))
        if kind == "domain":
            return [
                SimpleNamespace(reward=float(self.engine.seed))
                for _ in prompts
            ]
        return [
            prompt_output(prompt["prompt_token_ids"], -self.engine.seed / 10)
            for prompt in prompts
        ]


class FakeEngine:
    def __init__(self):
        self.events = []
        self.seed = None
        self.collective_rpc = FakeCollective(self)
        self.generate = FakeGenerate(self)


class FakePopulationTrainer(ExactRestoredAnchoredStepMixin):
    def __init__(self):
        self.engines = [FakeEngine() for _ in range(4)]
        self.n_vllm_engines = 4
        self.n_samples = 1
        self.global_seed = 42
        self.sigma = 0.001

    def _resolve(self, handles):
        return handles

    def _sampling_params(self, **kwargs):
        return kwargs

    def _postprocess_outputs(self, outputs, targets):
        assert len(outputs) == len(targets)
        rewards = [output.reward for output in outputs]
        return {"avg_reward": sum(rewards) / len(rewards)}


def evaluate_with_anchor_count(count):
    trainer = FakePopulationTrainer()
    anchor_items = [
        {
            "item_id": f"anchor-{index}",
            "prompt_token_ids": [1, 2, 3 + index],
        }
        for index in range(count)
    ]
    domain_params = object()
    metrics, anchors, _ = trainer._evaluate_population_with_anchor(
        [11, 22, 33, 44],
        ["domain-a", "domain-b"],
        ["answer-a", "answer-b"],
        domain_params,
        anchor_items,
        iteration=0,
    )
    return trainer, metrics, anchors, domain_params


def test_domain_scores_and_batches_are_invariant_across_anchor_counts():
    one, one_metrics, _, one_domain_params = evaluate_with_anchor_count(1)
    three, three_metrics, _, three_domain_params = evaluate_with_anchor_count(3)
    assert one_metrics == three_metrics
    for trainer, domain_params in (
        (one, one_domain_params), (three, three_domain_params),
    ):
        for engine in trainer.engines:
            assert [event[0] for event in engine.events] == [
                "perturb_self_weights", "generate", "generate",
                "restore_self_weights_exact",
            ]
            domain_call, anchor_call = engine.events[1:3]
            assert domain_call[1] == "domain"
            assert domain_call[3] == ["domain-a", "domain-b"]
            assert domain_call[4] is domain_params
            assert anchor_call[1] == "anchor"
            assert all(isinstance(prompt, dict) for prompt in anchor_call[3])


class ApplyParent:
    def apply_seed_coefficients(self, plan, target_alpha):
        self.parent_applications.append(target_alpha)
        plan["applied_alpha"] = target_alpha
        plan["applications"].append({"target_alpha": target_alpha})
        return plan

    def _persist_anchor_plan(self, plan):
        self.persisted.append(plan)


class GateCollective:
    def remote(self, method, args):
        assert args == ()
        if method == "save_self_exact_reference":
            return {"sha256": "same"}
        raise AssertionError(method)


class GateTrainer(ExactRestoredAnchoredStepMixin, ApplyParent):
    def __init__(self):
        self.parent_applications = []
        self.persisted = []
        self.engines = [
            SimpleNamespace(collective_rpc=GateCollective())
            for _ in range(4)
        ]

    def _resolve(self, handles):
        return handles


def test_nonzero_update_requires_passed_identity_audit_and_refreshes_reference():
    trainer = GateTrainer()
    plan = {"applied_alpha": 0.0, "applications": []}
    with pytest.raises(RuntimeError, match="passed alpha-zero"):
        trainer.apply_seed_coefficients(plan, 0.00005)
    assert trainer.parent_applications == []
    plan["identity_audit"] = {"passed": True}
    trainer.apply_seed_coefficients(plan, 0.00005)
    assert trainer.parent_applications == [0.00005]
    assert len(plan["applications"][-1]["exact_reference_states"]) == 4


def test_failed_identity_audit_journal_is_machine_readable(tmp_path):
    audit = {
        "schema": "eggroll-es-alpha-zero-identity-audit-v2",
        "status": "failed",
        "passed": False,
        "failure": {"update_applied": False},
    }
    trainer = object.__new__(GateTrainer)
    trainer.logging_dir = str(tmp_path)
    trainer._persist_identity_audit(0, audit)
    persisted = json.loads((
        tmp_path / "alpha-zero-identity-audit-iteration-1.json"
    ).read_text())
    assert persisted["status"] == "failed"
    assert persisted["failure"]["update_applied"] is False


class EstimateParent:
    def estimate_step_coefficients(
        self, iteration, seeds, input_text, target_text,
    ):
        del seeds, input_text, target_text
        plan = {
            "iteration": iteration,
            "applied_alpha": 0.0,
            "applications": [],
        }
        self._persist_anchor_plan(plan)
        return plan

    def _persist_anchor_plan(self, plan):
        self.persisted_plans.append(plan)


class VerifyCollective:
    def remote(self, method, args):
        assert method == "verify_self_exact_reference"
        assert args == ()
        # Real vLLM collective_rpc returns one result per TP worker.
        return [{"passed": True}]


class DriftAuditTrainer(ExactRestoredAnchoredStepMixin, EstimateParent):
    def __init__(self, tmp_path):
        self.logging_dir = str(tmp_path)
        self.engines = [
            SimpleNamespace(collective_rpc=VerifyCollective())
            for _ in range(4)
        ]
        self.anchor_items = [{"item_id": "train-anchor"}]
        self.anchor_items_per_step = 1
        self.global_seed = 42
        self.n_samples = 1
        self.train_temperature = 0.0
        self.train_top_p = 1.0
        self.max_tokens = 2
        self.mini_batch_size = 2
        self._exact_reference_states = [[{"sha256": "base"}]] * 4
        self.persisted_plans = []
        self.probes = iter([
            {"domain_output_sha256": "before"},
            {"domain_output_sha256": "after"},
        ])

    def _resolve(self, handles):
        return handles

    def _sampling_params(self, **kwargs):
        return kwargs

    def _iter_minibatches(self, inputs, targets, size):
        del size
        yield inputs, targets

    def _identity_probe(self, *args, **kwargs):
        del args, kwargs
        return next(self.probes)


def test_post_population_eval_drift_fails_before_any_update(tmp_path):
    trainer = DriftAuditTrainer(tmp_path)
    with pytest.raises(RuntimeError, match="evaluation drifted"):
        trainer.estimate_step_coefficients(
            0, [11, 22, 33, 44], ["q1", "q2"], ["a1", "a2"],
        )
    journal = json.loads((
        tmp_path / "alpha-zero-identity-audit-iteration-1.json"
    ).read_text())
    assert journal["status"] == "failed"
    assert journal["passed"] is False
    assert journal["failure"]["update_applied"] is False
    assert trainer.persisted_plans[-1]["identity_audit"]["passed"] is False


def test_resident_v2_driver_effective_anchor_api(tmp_path):
    assert line_search_v2.validate_effective_anchor_api() == (
        "coefficient_sha256", "load_anchor_prose", "load_trainer",
    )
    coefficients = [0.5, -0.5]
    digest = anchor_v2.coefficient_sha256([11, 22], coefficients)
    plan = {
        "seeds": [11, 22],
        "coefficients": coefficients,
        "coefficient_sha256": digest,
    }
    previous = line_search_v2.driver_v1.anchor
    line_search_v2.driver_v1.anchor = anchor_v2
    try:
        line_search_v2.driver_v1._verify_plan(plan, digest)
    finally:
        line_search_v2.driver_v1.anchor = previous

    text = "A short train-only prose anchor."
    row = {
        "document_id": "local-document",
        "item_id": "local-item",
        "split": "anchor_prose",
        "text": text,
        "text_sha256": anchor_v2.anchor_v1.normalized_text_sha256(text),
    }
    raw = (json.dumps(row) + "\n").encode("utf-8")
    anchor_path = tmp_path / "anchor.jsonl"
    anchor_path.write_bytes(raw)
    report_path = tmp_path / "anchor.report.json"
    report_path.write_text(json.dumps({
        "schema": "general-prose-anchor-build-v1",
        "output_rows": 1,
        "output_sha256": hashlib.sha256(raw).hexdigest(),
        "protected_artifacts": [],
    }))
    loaded = anchor_v2.load_anchor_prose(anchor_path, report_path)
    assert loaded["sha256"] == hashlib.sha256(raw).hexdigest()
    assert loaded["rows"][0]["item_id"] == "local-item"
