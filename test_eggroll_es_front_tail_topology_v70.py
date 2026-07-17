#!/usr/bin/env python3

from __future__ import annotations

import copy
import json

import pytest

import eggroll_es_front_tail_topology_v70 as subject


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


def _compute_attempt(arm_id: str, start: float, end: float) -> dict:
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
        "optimization_generated_rollouts": (
            subject.OPTIMIZATION_ROLLOUTS_PER_RUN
        ),
        "evaluation_generated_rollouts": (
            subject.TOTAL_EVALUATION_ROLLOUTS_PER_RUN
        ),
        "generated_tokens": 1000,
        "teacher_forced_tokens": 0,
        "sft_nonpadding_tokens": 0,
    }


def _run_evidence(
    spec: dict,
    *,
    seed: int,
    sequence_index: int,
    reward: float,
) -> dict:
    surface = spec["surface"]
    start = sequence_index * 100.0
    end = start + 100.0
    selected_layers = surface["selected_layers"]
    sensitivity_rollouts = (
        subject.LAYER_SENSITIVITY_ROLLOUTS_PER_RUN // len(selected_layers)
    )
    device_total = 100 * 1024**3
    peak_reserved = 80 * 1024**3
    traffic_bytes = 10 * 1024**3
    receipt = _trust_receipt()
    return {
        "schema": "qwen36-front-tail-topology-run-evidence-v70",
        "arm_id": spec["arm_id"],
        "training_seed": seed,
        "launch_sequence_index": sequence_index,
        "surface_sha256": surface["surface_sha256"],
        "started_monotonic_s": start,
        "ended_monotonic_s": end,
        "aggregate_quality_reward": reward,
        "target_activity": {
            "registered_surface_sha256": surface["surface_sha256"],
            "registered_runtime_mapping_sha256": surface[
                "runtime_mapping_sha256"
            ],
            "registered_module_count": surface["module_count"],
            "registered_tensor_count": surface["tensor_count"],
            "canonical_master_elements": surface["parameter_count"],
            "updated_parameter_count": surface["parameter_count"],
            "nonzero_update_elements": surface["parameter_count"],
            "update_l2_by_family": {
                family: 1.0 for family in subject.TARGET_FAMILIES
            },
            "runtime_adapter_allocated_elements": surface[
                "runtime_allocated_elements"
            ],
            "runtime_adapter_dtype_bytes": 2,
            "rank_padding_elements": 0,
            "runtime_max_lora_rank": surface["rank"],
            "inactive_target_count": 0,
            "unsupported_target_count": 0,
            "effective_lora_scaling": 2.0,
        },
        "layerwise_sensitivity": {
            str(layer): {
                "module_count": surface["layer_module_counts"][str(layer)],
                "parameter_count": surface[
                    "layer_parameter_counts"
                ][str(layer)],
                "nonzero_update_elements": surface[
                    "layer_parameter_counts"
                ][str(layer)],
                "update_l2": 1.0,
                "router_update_l2": 0.5,
                "masked_reward_delta": 0.01,
                "sensitivity_rollouts": sensitivity_rollouts,
            }
            for layer in selected_layers
        },
        "routing_by_layer": {
            str(layer): {
                "router_weight_sha256_before": "a" * 64,
                "router_weight_sha256_after": "b" * 64,
                "router_update_l2": 0.5,
                "observation_tokens": 4096,
                "active_experts": 240,
                "normalized_entropy": 0.90,
                "expert_load_cv": 1.0,
                "topk_disagreement_rate_from_base": 0.01,
                "routing_jsd_mean_from_base": 0.01,
                "route_inventory_sha256": "c" * 64,
            }
            for layer in selected_layers
        },
        "system_metrics": {
            "device_total_bytes": device_total,
            "peak_allocated_bytes": 79 * 1024**3,
            "peak_reserved_bytes": peak_reserved,
            "safe_headroom_bytes": device_total - peak_reserved,
            "adapter_resident_bytes": surface["runtime_allocated_elements"] * 2,
            "generated_tokens": 1000,
            "wall_seconds": 100.0,
            "generated_tokens_per_second": 10.0,
            "memory_traffic_bytes": traffic_bytes,
            "memory_traffic_seconds": 100.0,
            "memory_bandwidth_gib_per_second": 0.1,
        },
        "es_update_budget": copy.deepcopy(spec["es_update_budget"]),
        "compute_attempt": _compute_attempt(spec["arm_id"], start, end),
        "quality_trust_binding": subject.bind_quality_receipt(
            spec["arm_id"], seed, receipt
        ),
        "raw_output_access_count": 0,
        "protected_access_count": 0,
        "protected_source_opened": False,
    }


def _passing_evidence() -> list[dict]:
    specs = subject.build_arm_specs()
    rewards = {
        subject.CONTROL_ARM: 0.20,
        "early_contiguous_motif_r32": 0.30,
        "late_contiguous_motif_r32": 0.25,
        "symmetric_early_late_motifs_r16": 0.24,
        "matched_distributed_motifs_r8": 0.22,
    }
    return [
        _run_evidence(
            specs[arm_id], seed=seed, sequence_index=index,
            reward=rewards[arm_id],
        )
        for index, (seed, arm_id) in enumerate(subject._schedule())
    ]


def _find(evidence: list[dict], arm_id: str, seed: int = 1701) -> dict:
    return next(
        item for item in evidence
        if item["arm_id"] == arm_id and item["training_seed"] == seed
    )


def test_actual_40_layer_hybrid_and_current_adapter_geometry_are_sealed():
    geometry = subject.build_architecture_manifest()
    assert geometry["num_layers"] == 40
    assert geometry["layer_types"] == list(subject.MOTIF_TYPES) * 10
    assert geometry["all_target_record_count"] == 350
    assert geometry["current_active_module_count"] == 35
    assert geometry["current_active_tensor_count"] == 70
    assert geometry["reference_adapter_parameter_count"] == 4_528_128
    assert all(
        unit["semantic_mergeability_proven"] is False
        for unit in geometry["shape_compatible_motif_ranges"]
    )


def test_alias_shape_or_motif_drift_fails_closed():
    geometry = subject.build_architecture_manifest()
    geometry["all_target_records"][0]["runtime_target"] += ".alias"
    with pytest.raises(RuntimeError, match="key, layer, shape, or runtime alias"):
        subject.validate_architecture_manifest(geometry)


def test_all_five_layouts_are_exactly_parameter_family_and_compute_matched():
    specs = subject.build_arm_specs()
    assert tuple(specs) == subject.ARM_ORDER
    assert {
        arm_id: spec["surface"]["rank"] for arm_id, spec in specs.items()
    } == {
        "early_contiguous_motif_r32": 32,
        "late_contiguous_motif_r32": 32,
        "symmetric_early_late_motifs_r16": 16,
        "current_middle_late_motif_r32": 32,
        "matched_distributed_motifs_r8": 8,
    }
    assert {spec["surface"]["parameter_count"] for spec in specs.values()} == {
        4_528_128
    }
    assert {
        subject.canonical_sha256(spec["surface"]["family_parameter_counts"])
        for spec in specs.values()
    }.__len__() == 1
    assert {spec["surface"]["effective_lora_scaling"]
            for spec in specs.values()} == {2.0}
    assert {spec["surface"]["runtime_allocated_elements"]
            for spec in specs.values()} == {4_921_344}
    assert {spec["surface"]["runtime_packing_duplicate_elements"]
            for spec in specs.values()} == {393_216}
    assert {spec["es_update_budget"]["perturbation_scalar_draws"]
            for spec in specs.values()} == {72_450_048}
    assert specs["symmetric_early_late_motifs_r16"]["surface"][
        "module_count"
    ] == 70
    assert specs["matched_distributed_motifs_r8"]["surface"][
        "module_count"
    ] == 140


@pytest.mark.parametrize(("starts", "rank"), (
    ((1,), 32),
    ((0,), 16),
    ((0, 36), 32),
    ((0, 0), 16),
))
def test_nonaligned_or_budget_cheating_custom_surfaces_are_rejected(starts, rank):
    with pytest.raises((ValueError, RuntimeError)):
        subject.build_custom_surface(starts, rank)


def test_old_exact_insertion_mappings_remain_closed_negative_not_reopened():
    registry = subject.build_insertion_continuation_registry()
    assert set(registry["candidates"]) == {
        "insert_front_e005", "insert_middle_e005", "insert_back_e005"
    }
    assert registry["all_existing_epsilon_005_hypotheses_remain_closed"] is True
    for candidate in registry["candidates"].values():
        assert candidate["checkpoint_shapes_and_vllm_mapping_exact"] is True
        assert candidate["prior_train_only_result"] == "closed_negative_v23a_r3"
        assert candidate["continuation_launch_authorized"] is False
        assert candidate["semantic_mergeability_proven"] is False


def test_passing_analysis_selects_early_and_reports_sensitivity_quality_systems():
    result = subject.analyze_topology(_passing_evidence())
    assert result["integrity_passed"] is True
    assert result["selected_hpo_topology"] == "early_contiguous_motif_r32"
    assert subject.require_hpo_selection(result) == "early_contiguous_motif_r32"
    assert result["protected_terminal_evaluation_performed"] is False
    assert result["protected_terminal_promotion_eligible"] is False
    assert result["insertion_or_duplication_continuation_authorized"] is False
    assert all(
        totals["charged_gpu_seconds"] == 1200.0
        for totals in result["compute_totals"].values()
    )
    early = result["run_observations_by_arm_and_seed"][
        "early_contiguous_motif_r32"
    ]["1701"]
    assert set(early["layerwise_sensitivity"]) == {"0", "1", "2", "3"}
    assert early["quality"]["ood"]["qa_exact_count_delta"] == -1
    assert early["system_metrics"]["generated_tokens_per_second"] == 10.0


@pytest.mark.parametrize(("field", "value", "gate"), (
    ("registered_module_count", 34, "module_and_tensor_coverage_exact"),
    ("inactive_target_count", 1, "no_inactive_or_unsupported_targets"),
    ("unsupported_target_count", 1, "no_inactive_or_unsupported_targets"),
    ("rank_padding_elements", 1, "runtime_allocation_has_no_rank_padding"),
    ("runtime_adapter_allocated_elements", 4_528_129,
     "runtime_allocation_has_no_rank_padding"),
    ("effective_lora_scaling", 4.0, "lora_scaling_equal_across_ranks"),
))
def test_inactive_unsupported_rank_padding_or_scaling_cheats_fail_integrity(
    field, value, gate
):
    evidence = _passing_evidence()
    arm = _find(evidence, "early_contiguous_motif_r32")
    arm["target_activity"][field] = value
    result = subject.analyze_topology(evidence)
    name = f"early_contiguous_motif_r32:seed1701:{gate}"
    assert result["integrity_checks"][name] is False
    assert result["selected_hpo_topology"] is None
    with pytest.raises(RuntimeError, match="failed integrity"):
        subject.require_hpo_selection(result)


def test_router_change_and_drift_cannot_hide_in_high_aggregate_reward():
    evidence = _passing_evidence()
    arm = _find(evidence, "early_contiguous_motif_r32")
    arm["aggregate_quality_reward"] = 1.0
    arm["routing_by_layer"]["0"]["router_weight_sha256_after"] = "a" * 64
    arm["routing_by_layer"]["0"]["topk_disagreement_rate_from_base"] = 0.99
    result = subject.analyze_topology(evidence)
    gate = (
        "early_contiguous_motif_r32:seed1701:"
        "layer_0_router_policy_and_drift_bounded"
    )
    assert result["integrity_checks"][gate] is False
    assert result["selected_hpo_topology"] is None


def test_layerwise_activity_or_sensitivity_coverage_cannot_be_omitted():
    evidence = _passing_evidence()
    arm = _find(evidence, "symmetric_early_late_motifs_r16")
    del arm["layerwise_sensitivity"]["39"]
    with pytest.raises(ValueError, match="keys changed"):
        subject.analyze_topology(evidence)


def test_unequal_rollouts_or_compute_fail_closed():
    evidence = _passing_evidence()
    arm = _find(evidence, "early_contiguous_motif_r32")
    arm["compute_attempt"]["optimization_generated_rollouts"] -= 1
    with pytest.raises(RuntimeError, match="rollout budget changed"):
        subject.analyze_topology(evidence)
    evidence = _passing_evidence()
    arm = _find(evidence, "early_contiguous_motif_r32")
    for interval in arm["compute_attempt"]["gpu_residency_intervals"]:
        interval["end_s"] += 5.0
    with pytest.raises(RuntimeError, match="residency does not match"):
        subject.analyze_topology(evidence)


def test_posthoc_arm_or_schedule_overlap_cannot_be_selected():
    evidence = _passing_evidence()
    evidence[0]["arm_id"] = "posthoc_front_six"
    with pytest.raises(ValueError, match="foreign"):
        subject.analyze_topology(evidence)
    evidence = _passing_evidence()
    evidence[1]["started_monotonic_s"] = 50.0
    evidence[1]["ended_monotonic_s"] = 150.0
    for interval in evidence[1]["compute_attempt"]["gpu_residency_intervals"]:
        interval["start_s"] = 50.0
        interval["end_s"] = 150.0
    result = subject.analyze_topology(evidence)
    assert result["integrity_checks"]["schedule_index_1:nonoverlap"] is False
    assert result["selected_hpo_topology"] is None


def test_candidate_ood_failure_retains_control_instead_of_hiding_degradation():
    evidence = _passing_evidence()
    arm = _find(evidence, "early_contiguous_motif_r32")
    failed = _trust_receipt(ood_qa_lcb=-0.020001)
    arm["quality_trust_binding"] = subject.bind_quality_receipt(
        arm["arm_id"], arm["training_seed"], failed
    )
    result = subject.analyze_topology(evidence)
    gate = result["candidate_promotion_gates"][arm["arm_id"]]
    assert gate["checks"]["train_dev_ood_trust_all_seeds"] is False
    assert result["selected_hpo_topology"] == "late_contiguous_motif_r32"


def test_slow_or_bandwidth_heavy_candidate_is_not_promoted():
    evidence = _passing_evidence()
    for seed in subject.TRAINING_SEEDS:
        arm = _find(evidence, "early_contiguous_motif_r32", seed)
        arm["system_metrics"]["generated_tokens"] = 900
        arm["system_metrics"]["generated_tokens_per_second"] = 9.0
        arm["compute_attempt"]["generated_tokens"] = 900
        arm["system_metrics"]["memory_traffic_bytes"] = 11 * 1024**3
        arm["system_metrics"]["memory_bandwidth_gib_per_second"] = 0.11
    result = subject.analyze_topology(evidence)
    gate = result["candidate_promotion_gates"][
        "early_contiguous_motif_r32"
    ]
    assert gate["checks"]["throughput_noninferior_each_seed"] is False
    assert gate["checks"]["memory_traffic_noninferior_each_seed"] is False
    assert result["selected_hpo_topology"] == "late_contiguous_motif_r32"


def test_candidate_vram_delta_gate_and_absolute_headroom_are_separate():
    evidence = _passing_evidence()
    for seed in subject.TRAINING_SEEDS:
        arm = _find(evidence, "early_contiguous_motif_r32", seed)
        reserved = 83 * 1024**3
        arm["system_metrics"]["peak_allocated_bytes"] = 82 * 1024**3
        arm["system_metrics"]["peak_reserved_bytes"] = reserved
        arm["system_metrics"]["safe_headroom_bytes"] = 100 * 1024**3 - reserved
    result = subject.analyze_topology(evidence)
    gate = result["candidate_promotion_gates"][
        "early_contiguous_motif_r32"
    ]
    assert gate["checks"]["peak_vram_delta_bounded_each_seed"] is False
    assert result["integrity_passed"] is True
    evidence = _passing_evidence()
    arm = _find(evidence, "early_contiguous_motif_r32")
    reserved = 98 * 1024**3
    arm["system_metrics"]["peak_allocated_bytes"] = 97 * 1024**3
    arm["system_metrics"]["peak_reserved_bytes"] = reserved
    arm["system_metrics"]["safe_headroom_bytes"] = 100 * 1024**3 - reserved
    result = subject.analyze_topology(evidence)
    name = (
        "early_contiguous_motif_r32:seed1701:"
        "vram_receipt_consistent_and_safe"
    )
    assert result["integrity_checks"][name] is False
    assert result["selected_hpo_topology"] is None


def test_protected_or_unknown_evidence_fails_closed():
    evidence = _passing_evidence()
    evidence[0]["protected_access_count"] = 1
    with pytest.raises(RuntimeError, match="prohibited"):
        subject.analyze_topology(evidence)
    evidence = _passing_evidence()
    evidence[0]["posthoc"] = True
    with pytest.raises(ValueError, match="keys changed"):
        subject.analyze_topology(evidence)


def test_preregistration_is_deterministic_content_addressed_and_content_free():
    first = subject.build_preregistration()
    second = subject.build_preregistration()
    assert first == second
    assert first["content_sha256_before_self_field"] == subject.canonical_sha256({
        key: value for key, value in first.items()
        if key != "content_sha256_before_self_field"
    })
    assert first["outstanding_before_task_acceptance"] == {
        "dependency_issue": "specialist-0j5.14",
        "phase_separated_vram_and_bandwidth_profile_complete": False,
        "multi_seed_gpu_runs_complete": False,
        "protected_terminal_evaluation_complete": False,
        "task_must_remain_in_progress": True,
    }
    serialized = json.dumps(first, sort_keys=True)
    for forbidden in (
        "protected_question", "protected_answer", "protected_excerpt",
        "selected_opaque_item_identities", "per_item_score",
    ):
        assert forbidden not in serialized


def test_persisted_preregistration_matches_builder():
    subject.validate_preregistration(json.loads(
        subject.PREREGISTRATION.read_text(encoding="utf-8")
    ))


def test_mutated_analysis_is_rejected():
    result = subject.analyze_topology(_passing_evidence())
    result["selected_hpo_topology"] = subject.CONTROL_ARM
    with pytest.raises(RuntimeError, match="invalid or mutated"):
        subject.require_hpo_selection(result)
