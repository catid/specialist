#!/usr/bin/env python3
"""Focused fail-closed tests for V57 train-only HPO."""

from __future__ import annotations

import copy

import lora_es_conservative_actor_robust_v57 as design57
import lora_es_fragile_maximin_projection_v55b as v55b
import lora_es_nested_population_v52 as v52


def _gate(deltas_by_metric: dict[str, list[float]]) -> dict:
    checks = {name: True for name in v52.TRAIN_GATE_NAMES_V52}
    return {
        "schema": "synthetic-inherited-gate",
        "passed": True,
        "checks": checks,
        "metrics": {
            name: {"paired_actor_deltas": values}
            for name, values in deltas_by_metric.items()
        },
        "content_sha256": "old",
    }


def _all_metrics(value: float = 0.0) -> dict[str, list[float]]:
    return {
        metric: [value, value, value, value]
        for metric, _strict in design57.ACTOR_ROBUST_METRICS_V57.values()
    }


def test_v57_scale_plans_are_exact_conservative_prefix() -> None:
    projection = v55b.maximin_projection_v55b()
    plans = design57.scale_plans_v57(projection)
    assert [item["target_norm_ratio"] for item in plans] == list(
        design57.SCALE_ORDER_V57
    )
    assert all(item["actual_norm_ratio"] < 0.25 for item in plans)
    assert len({item["coefficient_sha256"] for item in plans}) == len(plans)


def test_v57_actor_gate_passes_all_four_valid_deltas() -> None:
    metrics = _all_metrics(0.0)
    metrics["domain"] = [1e-6, 2e-6, 3e-6, 4e-6]
    source = _gate(metrics)
    result = design57.tighten_gate_v57(lambda: copy.deepcopy(source))
    assert result["passed"] is True
    assert result["actor_robustness_v57"]["passed"] is True
    assert result["content_sha256"] != "old"


def test_v57_actor_gate_rejects_one_negative_qa_actor() -> None:
    metrics = _all_metrics(0.0)
    metrics["domain"] = [1e-6, 2e-6, 3e-6, 4e-6]
    metrics["qa_answer_logprob"] = [1e-4, -1e-9, 2e-4, 3e-4]
    result = design57.tighten_gate_v57(lambda: _gate(metrics))
    assert result["passed"] is False
    assert result["checks"]["qa_logprob_noninferiority"] is False
    assert result["actor_robustness_v57"]["receipts"][
        "qa_logprob_noninferiority"
    ]["all_four_actor_deltas_passed"] is False


def test_v57_domain_requires_strict_positive_actor_deltas() -> None:
    metrics = _all_metrics(0.0)
    result = design57.tighten_gate_v57(lambda: _gate(metrics))
    assert result["checks"]["domain_point_improvement"] is False


def test_v57_does_not_mutate_inherited_gate() -> None:
    metrics = _all_metrics(0.0)
    metrics["domain"] = [1e-6] * 4
    source = _gate(metrics)
    frozen = copy.deepcopy(source)
    design57.tighten_gate_v57(lambda: source)
    assert source == frozen
