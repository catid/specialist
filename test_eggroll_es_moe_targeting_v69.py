#!/usr/bin/env python3

from __future__ import annotations

import copy
import json

import pytest

pytest.skip("historical V1-bound V69 suite is nonpromotable", allow_module_level=True)

import eggroll_es_moe_targeting_v69 as subject


def _role_evidence() -> dict:
    return {
        "weighted_unit_mean_delta": 0.01,
        "weighted_unit_mean_delta_95_lcb": 0.0,
        "component_delta_95_lcb": {
            name: 0.0 for name in subject.trust_v67.COMPONENT_ORDER
        },
        "candidate_hard_event_counts": {
            name: 0 for name in subject.trust_v67.HARD_EVENT_ORDER
        },
    }


def _trust_receipt(*, ood_qa_lcb: float = -0.01) -> dict:
    return subject.trust_v67.evaluate_trust_region({
        "schema": "specialist-train-dev-ood-evidence-v67",
        "contract_content_sha256": (
            subject.trust_v67.EXPECTED_EVALUATION_CONTRACT_CONTENT_SHA256
        ),
        "protected_access_count": 0,
        "protected_source_opened": False,
        "roles_evaluated": list(subject.trust_v67.SAFE_EVALUATION_ROLES),
        "train": _role_evidence(),
        "dev": _role_evidence(),
        "ood": {
            "qa_mean_reward_delta_95_lcb": ood_qa_lcb,
            "qa_exact_count_delta": -1,
            "prose_mean_token_logprob_delta_95_lcb": -0.01,
        },
    })


def _layer_map(value: object) -> dict[str, object]:
    return {str(layer): value for layer in subject.TARGET_LAYERS}


def _attempt(arm_id: str, start: float, end: float) -> dict:
    return {
        "arm": arm_id,
        "gpu_residency_intervals": [
            {
                "physical_gpu_id": gpu,
                "start_s": start,
                "end_s": end,
            }
            for gpu in range(4)
        ],
        "optimization_generated_rollouts": 2048,
        "evaluation_generated_rollouts": 83,
        "generated_tokens": 1000,
        "teacher_forced_tokens": 2000,
        "sft_nonpadding_tokens": 0,
    }


def _evidence_for_arm(
    spec: dict,
    *,
    start: float,
    reward: float,
) -> dict:
    end = start + 100.0
    tuned = spec["router_policy"] == "router_tuned"
    receipt = _trust_receipt()
    return {
        "schema": "qwen36-moe-targeting-arm-evidence-v69",
        "arm_id": spec["arm_id"],
        "surface_sha256": spec["surface"]["surface_sha256"],
        "launch_sequence_index": spec["launch_sequence_index"],
        "phase": spec["phase"],
        "started_monotonic_s": start,
        "ended_monotonic_s": end,
        "aggregate_quality_reward": reward,
        "target_activity": {
            "registered_surface_sha256": spec["surface"]["surface_sha256"],
            "registered_module_count": spec["surface"]["module_count"],
            "updated_parameter_count": spec["surface"]["parameter_count"],
            "nonzero_update_elements": spec["surface"]["parameter_count"],
            "update_l2_by_family": {
                family: 1.0 for family in spec["surface"]["families"]
            },
            "unsupported_target_count": 0,
        },
        "routing_metrics": {
            "router_weight_sha256_before": "a" * 64,
            "router_weight_sha256_after": ("b" if tuned else "a") * 64,
            "router_update_l2": 1.0 if tuned else 0.0,
            "observation_tokens_per_layer": _layer_map(4096),
            "active_experts_per_layer": _layer_map(240),
            "normalized_entropy_per_layer": _layer_map(0.90),
            "expert_load_cv_per_layer": _layer_map(1.0),
            "coverage_delta_from_base_per_layer": _layer_map(0),
            "topk_disagreement_rate_from_base": 0.01,
            "probability_total_variation_mean_from_base": 0.01,
            "routing_jsd_mean_from_base": 0.01,
            "route_inventory_sha256": "c" * 64,
        },
        "throughput": {
            "generated_tokens": 1000,
            "wall_seconds": 100.0,
            "generated_tokens_per_second": 10.0,
        },
        "es_update_budget": copy.deepcopy(spec["es_update_budget"]),
        "compute_attempts": [_attempt(spec["arm_id"], start, end)],
        "quality_trust_bindings": [
            subject.bind_quality_receipt(spec["arm_id"], seed, receipt)
            for seed in (1701, 1702, 1703)
        ],
        "raw_output_access_count": 0,
        "protected_access_count": 0,
        "protected_source_opened": False,
    }


def _passing_evidence() -> list[dict]:
    specs = subject.build_arm_specs()
    rewards = (0.10, 0.12, 0.20, 0.30)
    return [
        _evidence_for_arm(
            specs[arm_id], start=index * 100.0, reward=rewards[index]
        )
        for index, arm_id in enumerate(subject.ARM_ORDER)
    ]


def _gate(result: dict, arm_id: str, name: str) -> bool:
    return result["hard_gate_checks"][f"{arm_id}:{name}"]


def test_checkpoint_and_reference_adapter_geometry_is_exact():
    geometry = subject.build_geometry_manifest()
    assert len(geometry["base_records"]) == 47
    assert len(geometry["reference_adapter_records"]) == 70
    assert geometry["reference_adapter_parameter_count"] == 4_528_128
    families = geometry["families"]
    assert families[subject.FAMILY_SHARED_SEQUENCE]["base_module_count"] == 19
    assert families[subject.FAMILY_SHARED_EXPERT]["base_module_count"] == 12
    assert families[subject.FAMILY_ROUTER]["base_module_count"] == 4
    routed = families[subject.FAMILY_ROUTED_EXPERT]
    assert routed["base_ndim_values"] == [3]
    assert routed["hypothetical_rank32_lora_parameters"] == 184_549_376
    assert routed["launch_supported"] is False
    scalar = families[subject.FAMILY_SHARED_SCALAR_GATE]
    assert scalar["hypothetical_rank32_lora_parameters"] == 262_272
    assert scalar["launch_supported"] is False


def test_alias_key_or_shape_drift_fails_closed():
    geometry = subject.build_geometry_manifest()
    geometry["base_records"][0]["base_key"] += ".alias"
    with pytest.raises(RuntimeError, match="alias, shape, or coverage drift"):
        subject.validate_geometry_manifest(geometry)


def test_arm_order_and_equal_parameter_update_budget_are_sealed():
    specs = subject.build_arm_specs()
    assert tuple(specs) == subject.ARM_ORDER
    frozen = specs[subject.PROMOTABLE_CONTRAST[0]]
    tuned = specs[subject.PROMOTABLE_CONTRAST[1]]
    assert frozen["surface"]["parameter_count"] == 4_233_216
    assert tuned["surface"]["parameter_count"] == 4_233_216
    assert tuned["surface"]["rank_histogram"] == {"24": 6, "32": 29}
    assert frozen["es_update_budget"] == tuned["es_update_budget"]
    assert frozen["es_update_budget"]["perturbation_scalar_draws"] == 67_731_456


@pytest.mark.parametrize("family", (
    subject.FAMILY_ROUTED_EXPERT,
    subject.FAMILY_SHARED_SCALAR_GATE,
))
def test_unsupported_lora_surfaces_are_nonlaunchable(family):
    with pytest.raises(RuntimeError, match="unsupported LoRA target"):
        subject.build_custom_surface(iter((family,)))


def test_passing_analysis_reports_family_quality_ood_routing_and_throughput():
    result = subject.analyze_targeting(_passing_evidence())
    assert result["all_hard_gates_passed"] is True
    assert result["selected_promotable_arm"] == subject.PROMOTABLE_CONTRAST[1]
    selected = subject.require_selection(result)
    assert selected == subject.PROMOTABLE_CONTRAST[1]
    assert result["compute_match"]["arms"] == sorted(subject.ARM_ORDER)
    assert all(
        totals["charged_gpu_seconds"] == 400.0
        for totals in result["compute_totals"].values()
    )
    for observation in result["arm_observations"].values():
        assert observation["target_families"]
        assert observation["throughput"]["generated_tokens_per_second"] == 10.0
        assert observation["routing"]["active_experts_per_layer"] == _layer_map(240)
        assert all(
            seed["passed"] and seed["ood"]["qa_exact_count_delta"] == -1
            for seed in observation["quality_by_seed"].values()
        )


@pytest.mark.parametrize(("mutation", "gate"), (
    (
        lambda arm: arm["target_activity"].__setitem__(
            "registered_module_count",
            arm["target_activity"]["registered_module_count"] - 1,
        ),
        "all_target_modules_registered",
    ),
    (
        lambda arm: arm["target_activity"].__setitem__(
            "nonzero_update_elements", 0
        ),
        "target_update_is_active",
    ),
    (
        lambda arm: arm["target_activity"].__setitem__(
            "updated_parameter_count",
            arm["target_activity"]["updated_parameter_count"] - 1,
        ),
        "updated_parameter_count_exact",
    ),
    (
        lambda arm: arm["es_update_budget"].__setitem__(
            "direction_count", subject.DIRECTION_COUNT - 1
        ),
        "es_update_budget_exact",
    ),
))
def test_absent_inactive_or_unequal_update_surface_cannot_promote(mutation, gate):
    evidence = _passing_evidence()
    arm_id = subject.PROMOTABLE_CONTRAST[1]
    mutation(evidence[-1])
    result = subject.analyze_targeting(evidence)
    assert _gate(result, arm_id, gate) is False
    assert result["promotion_eligible"] is False
    with pytest.raises(RuntimeError, match="failed a hard gate"):
        subject.require_selection(result)


def test_router_change_cannot_hide_inside_high_frozen_aggregate_quality():
    evidence = _passing_evidence()
    frozen = evidence[2]
    frozen["aggregate_quality_reward"] = 1.0
    frozen["routing_metrics"]["router_weight_sha256_after"] = "d" * 64
    frozen["routing_metrics"]["router_update_l2"] = 0.01
    result = subject.analyze_targeting(evidence)
    assert _gate(
        result, subject.PROMOTABLE_CONTRAST[0],
        "router_parameter_change_matches_policy",
    ) is False
    assert result["selected_promotable_arm"] is None


@pytest.mark.parametrize(("field", "value", "gate"), (
    ("topk_disagreement_rate_from_base", 0.201, "topk_routing_drift_bounded"),
    ("routing_jsd_mean_from_base", 0.051, "routing_jsd_bounded"),
))
def test_router_drift_cannot_hide_inside_high_tuned_quality(field, value, gate):
    evidence = _passing_evidence()
    evidence[-1]["aggregate_quality_reward"] = 1.0
    evidence[-1]["routing_metrics"][field] = value
    result = subject.analyze_targeting(evidence)
    assert _gate(result, subject.PROMOTABLE_CONTRAST[1], gate) is False
    assert result["promotion_eligible"] is False


def test_frozen_router_arms_must_finish_before_router_tuned_arm_starts():
    evidence = _passing_evidence()
    evidence[-1]["started_monotonic_s"] = 250.0
    evidence[-1]["ended_monotonic_s"] = 350.0
    result = subject.analyze_targeting(evidence)
    assert _gate(result, evidence[-1]["arm_id"], "launch_order") is False
    assert result["promotion_eligible"] is False


def test_ood_failure_on_one_seed_blocks_an_otherwise_strong_arm():
    evidence = _passing_evidence()
    arm = evidence[-1]
    failed = _trust_receipt(ood_qa_lcb=-0.020001)
    arm["quality_trust_bindings"][0] = subject.bind_quality_receipt(
        arm["arm_id"], 1701, failed
    )
    result = subject.analyze_targeting(evidence)
    assert _gate(
        result, arm["arm_id"], "train_dev_ood_trust_passes_all_seeds"
    ) is False
    assert result["promotion_eligible"] is False


def test_unequal_four_gpu_compute_is_rejected():
    evidence = _passing_evidence()
    for interval in evidence[-1]["compute_attempts"][0][
        "gpu_residency_intervals"
    ]:
        interval["end_s"] += 5.0
    with pytest.raises(RuntimeError, match="GPU-second matched"):
        subject.analyze_targeting(evidence)


def test_throughput_receipt_cannot_disagree_with_charged_work():
    evidence = _passing_evidence()
    evidence[-1]["throughput"]["generated_tokens_per_second"] = 999.0
    result = subject.analyze_targeting(evidence)
    assert _gate(
        result, evidence[-1]["arm_id"], "throughput_receipt_consistent"
    ) is False


def test_unknown_or_protected_evidence_fails_closed():
    evidence = _passing_evidence()
    evidence[0]["unknown"] = True
    with pytest.raises(ValueError, match="keys changed"):
        subject.analyze_targeting(evidence)
    evidence = _passing_evidence()
    evidence[0]["protected_access_count"] = 1
    with pytest.raises(RuntimeError, match="prohibited"):
        subject.analyze_targeting(evidence)


def test_preregistration_is_deterministic_content_addressed_and_content_free():
    first = subject.build_preregistration()
    second = subject.build_preregistration()
    assert first == second
    assert first["content_sha256_before_self_field"] == subject.canonical_sha256({
        key: value for key, value in first.items()
        if key != "content_sha256_before_self_field"
    })
    serialized = json.dumps(first, sort_keys=True)
    for forbidden in (
        "protected_question", "protected_answer", "protected_excerpt",
        "selected_opaque_item_identities", "per_item",
    ):
        assert forbidden not in serialized


def test_persisted_preregistration_matches_builder():
    subject.validate_preregistration(json.loads(
        subject.PREREGISTRATION.read_text(encoding="utf-8")
    ))


def test_mutated_analysis_receipt_is_rejected():
    result = subject.analyze_targeting(_passing_evidence())
    result["selected_promotable_arm"] = subject.PROMOTABLE_CONTRAST[0]
    with pytest.raises(RuntimeError, match="invalid or mutated"):
        subject.require_selection(result)
