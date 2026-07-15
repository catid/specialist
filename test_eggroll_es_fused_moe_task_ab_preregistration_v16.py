#!/usr/bin/env python3

import copy
import hashlib
import json

import pytest

import eggroll_es_fused_moe_task_ab_preregistration_v16 as prereg


def _candidate(default_seconds=None, tuned_seconds=None):
    frozen = prereg.build_preregistration_v16()
    default_seconds = default_seconds or [1.0] * 16
    tuned_seconds = tuned_seconds or [0.9] * 16
    estimator = {
        "stability": copy.deepcopy(frozen["evidence"]["v13"]["stability"]),
        "robust_aggregate_sha256": "c" * 64,
    }
    arm = {
        "diagnostic_content_sha256": "a" * 64,
        "dense_result_manifest_sha256": "b" * 64,
        "task_output_sha256": "d" * 64,
        "compact_estimator": estimator,
        "generation_timing": {},
        "all_integrity_audits_passed": True,
        "persisted_raw_content": False,
    }
    value = {
        "schema": "eggroll-es-fused-moe-task-ab-summary-v16",
        "experiment_name": prereg.EXPERIMENT_NAME_V16,
        "alpha": 0.0,
        "model_update_applied": False,
        "validation_ood_heldout_or_benchmark_used": False,
        "arm_order": list(prereg.ARM_ORDER_V16),
        "arms": {
            "default_triton": copy.deepcopy(arm),
            "tuned_triton": copy.deepcopy(arm),
        },
        "panel_bundle_content_sha256": (
            prereg.V13_PANEL_BUNDLE_CONTENT_SHA256_V16
        ),
        "panel_identities": copy.deepcopy(frozen["task"]["panel_identities"]),
        "perturbation_basis_sha256": (
            prereg.V13_PERTURBATION_BASIS_SHA256_V16
        ),
        "all_integrity_audits_passed": True,
        "persisted_response_vectors_or_row_content": False,
    }
    for name, times in (
        ("default_triton", default_seconds),
        ("tuned_triton", tuned_seconds),
    ):
        value["arms"][name]["generation_timing"] = {
            "wave_seconds": list(times),
            "total_seconds": sum(times),
        }
    value["content_sha256_before_self_field"] = prereg.canonical_sha256(value)
    return value


def _reseal(value):
    value["content_sha256_before_self_field"] = prereg.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def test_v16_binds_only_sealed_config_speed_and_v13_evidence():
    benchmark, config, v13 = prereg.load_bound_evidence_v16()
    assert benchmark["end_to_end_triton_generation"]["aggregate"] == {
        "median_default_seconds": 2.464496353502909,
        "median_tuned_seconds": 2.105515083494538,
        "median_default_tokens_per_second": 3324.012973673293,
        "median_tuned_tokens_per_second": 3890.736134101703,
        "median_time_speedup": 1.1704957009439074,
        "worst_per_gpu_speedup": 1.1646831453362367,
    }
    assert len(config) == 12
    assert v13["content_sha256_before_self_field"] == (
        prereg.V13_EVIDENCE_CONTENT_SHA256_V16
    )


def test_v16_is_separate_train_only_exact_v13_systems_ab():
    value = prereg.build_preregistration_v16()
    assert value["separation"] == {
        "failed_v15b_architecture_result_used": False,
        "retained_model_recipe": "V13 middle-late layers 20-23",
        "systems_ab_only": True,
        "no_model_or_data_hpo": True,
    }
    assert value["contains_validation_ood_heldout_or_benchmark_content"] is False
    assert value["task"]["alpha"] == 0.0
    assert value["task"]["model_update_allowed"] is False
    assert value["hardware"]["gpu_ids"] == [0, 1, 2, 3]
    assert value["hardware"]["all_four_gpus_required_every_signed_wave"]
    assert value["arms"]["only_intended_difference"] == (
        "exact_tuned_config_folder_activation"
    )


def test_v16_timing_boundary_and_thresholds_are_fixed_before_runtime():
    gate = prereg.build_preregistration_v16()["promotion_gate"]["timing"]
    assert gate["minimum_total_generation_time_speedup"] == 1.05
    assert gate["minimum_median_paired_wave_speedup"] == 1.05
    assert gate["minimum_nonregressive_wave_count"] == 14
    assert gate["required_wave_count"] == 16
    assert gate["post_hoc_shapes_repetitions_or_endpoints_allowed"] is False
    boundary = prereg.build_preregistration_v16()["timing_protocol"]["boundary"]
    for excluded in (
        "init", "model load", "JIT warmup", "perturb", "restore", "scoring",
    ):
        assert excluded in boundary


def test_v16_gate_passes_exact_outputs_and_material_task_speedup_only():
    gate = prereg.evaluate_candidate_v16(_candidate())
    assert gate["eligible_for_later_opt_in_training_preregistration"]
    assert gate["eligible_for_model_update"] is False
    assert gate["eligible_to_open_evaluation"] is False
    assert all(gate["exact_equivalence"].values())
    assert gate["timing"]["total_speedup_passed"]
    assert gate["timing"]["median_speedup_passed"]
    assert gate["timing"]["nonregressive_wave_count_passed"]


@pytest.mark.parametrize(
    "field",
    [
        "diagnostic_content_sha256", "dense_result_manifest_sha256",
        "task_output_sha256", "compact_estimator",
    ],
)
def test_v16_every_exact_equivalence_field_is_conjunctive(field):
    candidate = _candidate()
    candidate["arms"]["tuned_triton"][field] = (
        {"changed": True} if field == "compact_estimator" else "e" * 64
    )
    _reseal(candidate)
    gate = prereg.evaluate_candidate_v16(candidate)
    assert gate["exact_equivalence"][f"{field}_equal"] is False
    assert not gate["eligible_for_later_opt_in_training_preregistration"]


def test_v16_total_median_and_nonregressive_timing_rules_are_conjunctive():
    total_fail = prereg.evaluate_candidate_v16(
        _candidate([1.0] * 16, [0.96] * 16)
    )
    assert not total_fail["timing"]["total_speedup_passed"]
    assert not total_fail["timing"]["median_speedup_passed"]

    mixed = [0.9] * 13 + [1.01] * 3
    count_fail = prereg.evaluate_candidate_v16(_candidate([1.0] * 16, mixed))
    assert count_fail["timing"]["total_speedup_passed"]
    assert count_fail["timing"]["median_speedup_passed"]
    assert not count_fail["timing"]["nonregressive_wave_count_passed"]


def test_v16_sorted_json_candidate_replays_and_update_tampering_fails():
    persisted = json.loads(json.dumps(_candidate(), sort_keys=True))
    assert prereg.evaluate_candidate_v16(persisted)[
        "eligible_for_later_opt_in_training_preregistration"
    ]
    persisted["model_update_applied"] = True
    _reseal(persisted)
    with pytest.raises(RuntimeError, match="candidate summary contract"):
        prereg.evaluate_candidate_v16(persisted)


def test_v16_frozen_preregistration_is_exact_rebuild():
    frozen = json.loads(prereg.PREREGISTRATION_PATH_V16.read_text())
    assert frozen == prereg.build_preregistration_v16()
    assert frozen["content_sha256_before_self_field"] == (
        "82569802ad89a0c3c92e4bb0a28a2db867a0bcb01ab7268ff9ab6048558a115c"
    )
    assert hashlib.sha256(
        prereg.PREREGISTRATION_PATH_V16.read_bytes()
    ).hexdigest() == (
        "53d796cee9c0ef67fdcb549f9bda55a978dd2840812d6a0ae0a8a4e363d24853"
    )
