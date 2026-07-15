#!/usr/bin/env python3
"""CPU-only tests for the V28C BF16 train-step runtime."""

from __future__ import annotations

import copy
import json
import sys
import time
from types import SimpleNamespace

import pytest

import eggroll_es_v27c_train_step_ab_preregistration_v28c as prereg
import run_eggroll_es_v27c_train_step_ab_v28c as runtime


def synthetic_diagnostic():
    panel_analysis = {
        name: {"coefficients": [float(index) for index in range(32)]}
        for name in prereg.PANEL_NAMES_V28C
    }
    value = {
        "responses": {name: {"fixed": True} for name in prereg.PANEL_NAMES_V28C},
        "analysis": {
            "panel_analysis": panel_analysis,
            "robust_optimization_aggregate": {
                "coefficients": [float(index) for index in range(32)],
            },
        },
        "identity_audit": {"passed": True},
        "population_boundary_audit_v4": {"passed": True},
        "perturbation_basis": {"seed_sha256": "a" * 64},
        "panel_contract": {name: {"rows": 56} for name in prereg.PANEL_NAMES_V28C},
        "common_random_numbers": {"same": True},
        "hardware_coverage": {"engines": 4},
    }
    value["content_sha256_before_self_field"] = runtime.canonical_sha256(value)
    return value


def synthetic_pair_results(*, speed_ratio=1.02, memory_ratio=1.0):
    results = {}
    for pair in prereg.PAIR_ORDER_V28C:
        default = {
            "full_elapsed_ns": int(round(1_000_000 * speed_ratio)),
            "configure_elapsed_ns": int(round(100_000 * speed_ratio)),
            "estimate_elapsed_ns": int(round(900_000 * speed_ratio)),
            "peak_allocated_bytes": 1_000_000,
            "peak_reserved_bytes": 2_000_000,
            "activity": {"peak_nvml_fraction": 0.72},
        }
        tuned = {
            "full_elapsed_ns": 1_000_000,
            "configure_elapsed_ns": 100_000,
            "estimate_elapsed_ns": 900_000,
            "peak_allocated_bytes": int(round(1_000_000 * memory_ratio)),
            "peak_reserved_bytes": int(round(2_000_000 * memory_ratio)),
            "activity": {"peak_nvml_fraction": 0.73},
        }
        results[pair] = {"arms": {
            "default_empty": default, "v27c_tuned": tuned,
        }}
    return results


def idle_certificate():
    return {
        "gpus": [
            {
                "physical_gpu_id": index,
                "nvml_uuid": f"uuid-{index}",
                "pci_bus_id": f"pci-{index}",
                "total_bytes": 100,
            }
            for index in range(4)
        ],
        "all_four_idle": True,
        "content_sha256_before_self_field": "a" * 64,
    }


def test_v28c_loads_exact_frozen_preregistration_and_layer_plan():
    value = runtime.load_preregistration_v28c()
    assert value["content_sha256_before_self_field"] == (
        "8158f1b8bdd04fde43b48369434484473b0b7686e31b15ed9c2e501108d5c1fd"
    )
    layer = runtime.load_layer_bundle_v28c()
    assert layer["model_config_sha256"] == prereg.MODEL_CONFIG_SHA256_V28C
    assert layer["plan_sha256"] == runtime.driver_v10.MIDDLE_LATE_PLAN_SHA256_V10


def test_v28c_implementation_identity_is_deterministic_and_complete():
    first = runtime.implementation_identity_v28c()
    second = runtime.implementation_identity_v28c()
    assert first == second
    assert first["bundle_sha256"] == runtime.canonical_sha256(first["files"])
    assert set(first["files"]) == set(runtime.IMPLEMENTATION_PATHS_V28C)
    assert first["files"]["v28c_runtime_tests"]["path"].endswith(
        "test_run_eggroll_es_v27c_train_step_ab_v28c.py"
    )


def test_v28c_worktree_allowlist_is_exact_and_version_bounded():
    status = (
        "?? data/manual_reviews/context_merit_audit_v390/a.json\n"
        "?? data/manual_reviews/context_merit_audit_v402/b.json\n"
        "?? experiments/dataset_probes/a.json\n"
        "?? experiments/gpu_utilization_v31_base_qwen36_train_reward_probe.jsonl\n"
    )
    result = runtime.validate_worktree_status_v28c(status)
    assert result["all_tracked_files_clean"] is True
    assert result["allowed_untracked_entry_count"] == 4
    with pytest.raises(RuntimeError, match="committed-clean"):
        runtime.validate_worktree_status_v28c(
            "?? data/manual_reviews/context_merit_audit_v389/a.json\n"
        )
    with pytest.raises(RuntimeError, match="committed-clean"):
        runtime.validate_worktree_status_v28c(" M tracked.py\n")
    with pytest.raises(RuntimeError, match="committed-clean"):
        runtime.validate_worktree_status_v28c("?? arbitrary.txt\n")


def test_v28c_argv_rejects_nontrain_update_eval_and_fp8_surfaces():
    for token in (
        "--checkpoint=x", "--update", "--validation=x", "--eval",
        "--promotion=x", "--fp8-model=x",
    ):
        with pytest.raises(ValueError, match="forbidden runtime surface"):
            runtime._assert_train_only_argv_v28c([token])
    runtime._assert_train_only_argv_v28c(["--v28c-dry-run"])


def test_v28c_compact_contract_rejects_raw_diagnostic_timing_memory_and_pid():
    for value in (
        {"diagnostic": {}}, {"elapsed_ns": 10}, {"memory_samples": []},
        {"pid": 4}, {"coefficients": [1.0]}, {"prompt_token_ids": [1]},
    ):
        with pytest.raises(RuntimeError, match="forbidden keys"):
            runtime._assert_compact_v28c(value)
    runtime._assert_compact_v28c({
        "median_default_over_tuned_ratio": 1.02,
        "commitment_sha256": "a" * 64,
        "all_four_active": True,
    })


def test_v28c_exact_equivalence_covers_coefficients_aggregate_and_guards():
    default = synthetic_diagnostic()
    tuned = copy.deepcopy(default)
    result = runtime.exact_equivalence_v28c(default, tuned)
    assert result["pass"] is True
    assert result["all_five_panel_coefficient_arrays_exact"] is True
    assert result["robust_aggregate_coefficients_exact"] is True
    assert result["identity_guard_exact"] is True
    assert result["population_boundary_guard_exact"] is True
    tuned["analysis"]["panel_analysis"]["optimization_0"]["coefficients"][0] = 99.0
    assert runtime.exact_equivalence_v28c(default, tuned)["pass"] is False


def test_v28c_three_endpoint_bootstrap_passes_fixed_meaningful_thresholds():
    result = runtime.performance_summary_v28c(synthetic_pair_results())
    assert result["pair_count"] == 12
    assert result["all_three_inferential_endpoints_passed"] is True
    assert result["absolute_peak_nvml_gate_passed"] is True
    speed = result["endpoints"]["complete_train_step_speed"]
    assert speed["median_default_over_tuned_ratio"] == pytest.approx(1.02)
    assert speed["familywise_lower_confidence_bound"] == pytest.approx(1.02)
    assert speed["point_threshold"] == 1.01
    assert speed["lower_bound_strict_threshold"] == 1.0
    assert result["bootstrap_draw_plan_sha256"] == (
        prereg.bootstrap_draw_plan_v28c()[1]
    )


def test_v28c_speed_or_memory_regression_fails_closed():
    slow = runtime.performance_summary_v28c(
        synthetic_pair_results(speed_ratio=1.005)
    )
    assert slow["endpoints"]["complete_train_step_speed"]["pass"] is False
    memory = runtime.performance_summary_v28c(
        synthetic_pair_results(memory_ratio=1.03)
    )
    assert memory["endpoints"]["peak_torch_allocated"]["pass"] is False
    assert memory["endpoints"]["peak_torch_reserved"]["pass"] is False


def test_v28c_gate_authority_remains_narrow_on_pass_and_failure():
    equivalence = {"all_twelve_pairs_exact": True}
    performance = {
        "all_three_inferential_endpoints_passed": True,
        "absolute_peak_nvml_gate_passed": True,
    }
    integrity = {"all_integrity_gates_passed": True}
    gate = runtime.gate_v28c(equivalence, performance, integrity)
    assert gate["pass"] is True
    assert gate["decision"] == (
        "authorize_only_exact_v27c_table_in_a_separately_frozen_"
        "bf16_train_only_training_recipe"
    )
    for key in (
        "direct_recipe_adoption_authorized", "model_update_authorized",
        "checkpoint_write_authorized", "evaluation_authorized",
        "dataset_promotion_authorized", "nontrain_reuse_authorized",
        "fp8_reuse_authorized",
    ):
        assert gate[key] is False
    performance["all_three_inferential_endpoints_passed"] = False
    assert runtime.gate_v28c(equivalence, performance, integrity)["decision"] == (
        "retain_empty_default_bf16_training_recipe"
    )


def test_v28c_actor_identity_validation_checks_exact_config_and_strips_pids():
    baseline = runtime._physical_identity_map_v28c(idle_certificate())
    identities = []
    for gpu_id in range(4):
        identities.append({
            "schema": "eggroll-es-v28c-actor-runtime-identity",
            "arm": "default_empty",
            "model_path": str(prereg.MODEL_PATH_V28C),
            "physical_gpu_id": gpu_id,
            "ray_gpu_id_raw": str(gpu_id),
            "cuda_visible_devices": str(gpu_id),
            "nvml_uuid": f"uuid-{gpu_id}",
            "pci_bus_id": f"pci-{gpu_id}",
            "total_bytes": 100,
            "pid": 1000 + gpu_id,
            "config_folder": "/tmp/empty",
            "config_source": "generic_fallback_none",
            "config_content_sha256": None,
            "vllm_fused_moe_file_sha256": runtime.VLLM_FUSED_MOE_SHA256_V28C,
            "vllm_envs_file_sha256": runtime.VLLM_ENVS_SHA256_V28C,
        })
    pids, commitment = runtime.validate_actor_identities_v28c(
        identities, "default_empty", "/tmp/empty", baseline,
    )
    assert pids == {index: 1000 + index for index in range(4)}
    assert len(commitment) == 64
    identities[2]["config_source"] = "exact_committed_v27c_table"
    with pytest.raises(RuntimeError, match="mapping changed"):
        runtime.validate_actor_identities_v28c(
            identities, "default_empty", "/tmp/empty", baseline,
        )


def test_v28c_activity_monitor_requires_pid_bound_simultaneous_four_gpu_use(monkeypatch):
    fake = SimpleNamespace()
    fake.NVMLError_NotSupported = type("NVMLError_NotSupported", (Exception,), {})
    fake.nvmlInit = lambda: None
    fake.nvmlShutdown = lambda: None
    fake.nvmlDeviceGetHandleByIndex = lambda index: index
    fake.nvmlDeviceGetComputeRunningProcesses = lambda handle: [
        SimpleNamespace(pid=1000 + handle)
    ]
    fake.nvmlDeviceGetGraphicsRunningProcesses = lambda _handle: []
    fake.nvmlDeviceGetUtilizationRates = lambda _handle: SimpleNamespace(gpu=50)
    fake.nvmlDeviceGetMemoryInfo = lambda _handle: SimpleNamespace(
        used=50, total=100,
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)

    def work():
        time.sleep(0.04)
        return "done"

    result, audit = runtime.monitor_estimator_activity_v28c(
        {index: 1000 + index for index in range(4)}, work,
    )
    assert result == "done"
    assert audit[
        "all_four_expected_processes_and_positive_utilization_simultaneously"
    ] is True
    assert audit["qualifying_sample_count"] > 0
    assert audit["peak_nvml_fraction"] == 0.5
    assert len(audit["commitment_sha256"]) == 64


def test_v28c_group_loop_aborts_before_later_groups_on_first_failure(monkeypatch):
    calls = []

    def fail_fourth(pair, arm, group_index, *_args):
        calls.append((pair, arm, group_index))
        if group_index == 3:
            raise RuntimeError("synthetic group failure")
        diagnostic = synthetic_diagnostic()
        return ({
            "diagnostic": diagnostic,
            "diagnostic_commitment_sha256": diagnostic[
                "content_sha256_before_self_field"
            ],
            "actor_identity_commitment_sha256": "b" * 64,
            "cleanup_certificate_sha256": "c" * 64,
            "activity": {
                "commitment_sha256": "d" * 64,
                "peak_nvml_fraction": 0.7,
                "sample_count": 10,
                "qualifying_sample_count": 1,
                "all_four_expected_processes_and_positive_utilization_simultaneously": True,
            },
            "full_elapsed_ns": 100,
            "configure_elapsed_ns": 10,
            "estimate_elapsed_ns": 90,
            "peak_allocated_bytes": 100,
            "peak_reserved_bytes": 100,
        }, idle_certificate())

    monkeypatch.setattr(runtime, "_run_one_arm_v28c", fail_fourth)
    with pytest.raises(RuntimeError, match="synthetic group failure"):
        runtime.run_counterbalanced_probe_v28c(
            prereg.build_preregistration_v28c(), {}, {}, idle_certificate(),
        )
    assert len(calls) == 4
    assert calls[-1][2] == 3


def test_v28c_fresh_exclusive_attempt_and_run_paths_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "OUTPUT_DIRECTORY_V28C", tmp_path)
    monkeypatch.setattr(runtime.runtime_r2, "certify_runtime_environment_r2", lambda: {
        "content_sha256_before_self_field": "e" * 64,
    })
    monkeypatch.setattr(runtime, "live_cpu_disk_audit_v28c", lambda: {
        "content_sha256_before_self_field": "f" * 64,
    })
    (tmp_path / runtime.ATTEMPT_NAME_V28C).write_text("claimed")
    with pytest.raises(RuntimeError, match="fresh exclusive"):
        runtime.run_exact_v28c(
            {}, {}, {}, {}, {}, {
                "content_sha256_before_self_field": "a" * 64,
            },
        )


def test_v28c_dry_run_is_cpu_only_and_emits_handoff_hashes(capsys):
    payload = runtime.main(["--v28c-dry-run"])
    emitted = json.loads(capsys.readouterr().out)
    assert emitted == payload
    assert payload["gpu_launched"] is False
    assert payload["train_only_runtime_launched"] is False
    assert payload["fresh_four_tp1_engine_group_count"] == 24
    assert payload["total_generation_request_budget"] == 443_520
    assert payload["required_python"].endswith("es-at-scale/.venv/bin/python")
    assert len(payload["implementation_bundle_sha256"]) == 64
    assert len(payload["recipe_sha256"]) == 64
