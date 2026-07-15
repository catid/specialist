#!/usr/bin/env python3
"""CPU-only tests for the V29H serialized-FP8 full-model runtime A/B."""

from __future__ import annotations

import copy
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

import build_vllm_moe_fp8_full_model_runtime_ab_preregistration_v29h as prereg
import run_vllm_moe_fp8_full_model_runtime_ab_v29h as runtime


def synthetic_pair_results(*, latency_ratio=1.01, memory_ratio=1.0):
    values = {}
    for pair in prereg.PAIR_ORDER_V29H:
        default_elapsed = np.full((7, 4), int(round(1_000_000 * latency_ratio)))
        tuned_elapsed = np.full((7, 4), 1_000_000)
        default_allocated = np.full((7, 4), 10_000_000)
        default_reserved = np.full((7, 4), 20_000_000)
        tuned_allocated = np.full(
            (7, 4), int(round(10_000_000 * memory_ratio))
        )
        tuned_reserved = np.full(
            (7, 4), int(round(20_000_000 * memory_ratio))
        )
        common = {
            "activity": {"peak_nvml_fraction": 0.72},
        }
        values[pair] = {"arms": {
            "default_empty": {
                **common,
                "elapsed_ns": default_elapsed,
                "peak_allocated_bytes": default_allocated,
                "peak_reserved_bytes": default_reserved,
                "peak_nvml_fraction": np.full((7, 4), 0.72),
            },
            "v29_selected_tuned": {
                **common,
                "elapsed_ns": tuned_elapsed,
                "peak_allocated_bytes": tuned_allocated,
                "peak_reserved_bytes": tuned_reserved,
                "peak_nvml_fraction": np.full((7, 4), 0.73),
            },
        }}
    return values


def synthetic_output(token_id=42, logprob=-0.25):
    selected = SimpleNamespace(logprob=logprob)
    candidate = SimpleNamespace(
        token_ids=[token_id],
        logprobs=[{token_id: selected}],
        cumulative_logprob=logprob,
    )
    return SimpleNamespace(outputs=[candidate])


def test_v29h_loads_exact_preregistration_and_implementation_identity():
    value = runtime.load_preregistration_v29h()
    assert value["content_sha256_before_self_field"] == (
        "17e0cf1b7ea560e8e446d50bffcd97f6f110cda4ff0624abd5b37e6ce83908d8"
    )
    implementation = runtime.implementation_identity_v29h()
    assert implementation["bundle_sha256"] == runtime.canonical_sha256(
        implementation["files"]
    )
    assert set(implementation["files"]) == set(runtime.IMPLEMENTATION_PATHS_V29H)


def test_v29h_synthetic_requests_have_exact_profiles_and_no_content_surface():
    first, first_audit = runtime.synthetic_requests_v29h(0)
    second, second_audit = runtime.synthetic_requests_v29h(0)
    assert first == second
    assert first_audit == second_audit
    assert {gpu: len(item["prompt_token_ids"]) for gpu, item in first.items()} == {
        0: 256, 1: 512, 2: 1024, 3: 2048,
    }
    assert all(
        200 <= token < 1200
        for item in first.values() for token in item["prompt_token_ids"]
    )
    assert first_audit["combined_prompt_tokens"] == 3840
    assert first_audit[
        "dataset_tokenizer_decoding_or_semantic_content_opened"
    ] is False
    assert first_audit["raw_token_ids_persisted"] is False
    different, _audit = runtime.synthetic_requests_v29h(1)
    assert different != first
    with pytest.raises(ValueError, match="pair index changed"):
        runtime.synthetic_requests_v29h(8)


def test_v29h_output_contract_is_numeric_exact_and_never_reads_text():
    contract = runtime.output_contract_v29h(synthetic_output())
    assert contract == {
        "generated_token_id": 42,
        "selected_logprob": -0.25,
        "cumulative_logprob": -0.25,
        "generated_token_count": 1,
        "integer_output_shape": [1],
    }
    broken = synthetic_output()
    broken.outputs[0].token_ids = [1, 2]
    with pytest.raises(RuntimeError, match="geometry changed"):
        runtime.output_contract_v29h(broken)


def test_v29h_exact_pair_equivalence_covers_reference_and_all_timed_calls():
    base = {
        "reference_contracts": [{"generated_token_id": index} for index in range(4)],
        "reference_commitment_sha256": "a" * 64,
        "timed_commitments_sha256": "b" * 64,
        "all_four_engines_generated_every_call": True,
    }
    assert runtime.exact_pair_equivalence_v29h(base, copy.deepcopy(base))["pass"] is True
    tuned = copy.deepcopy(base)
    tuned["timed_commitments_sha256"] = "c" * 64
    assert runtime.exact_pair_equivalence_v29h(base, tuned)["pass"] is False


def test_v29h_hierarchical_bootstrap_shape_and_constant_statistic():
    pair_draws, call_draws, _sha = prereg.bootstrap_draw_plan_v29h()
    values = np.full((8, 7), 1.01)
    replicates = runtime._hierarchical_replicates_v29h(
        values, pair_draws, call_draws,
    )
    assert replicates.shape == (50_000,)
    assert np.all(replicates == 1.01)
    with pytest.raises(RuntimeError, match="matrix changed"):
        runtime._hierarchical_replicates_v29h(
            np.ones((8, 6)), pair_draws, call_draws,
        )


def test_v29h_performance_passes_all_ten_preregistered_endpoints():
    summary = runtime.performance_summary_v29h(synthetic_pair_results())
    assert summary["pair_count"] == 8
    assert summary["matched_timing_calls_per_pair"] == 7
    assert summary["all_ten_performance_endpoints_passed"] is True
    assert summary["absolute_nvml_gate_passed"] is True
    assert len(summary["latency_by_physical_gpu"]) == 4
    assert len(summary["peak_vram_by_physical_gpu"]) == 4
    for endpoint in summary["latency_by_physical_gpu"].values():
        assert endpoint["median_default_over_tuned_ratio"] == pytest.approx(1.01)
        assert endpoint["familywise_lower_confidence_bound"] == pytest.approx(1.01)
        assert endpoint["pass"] is True
    assert summary["global"]["full_model_latency"][
        "geometric_mean_speedup"
    ] == pytest.approx(1.01)
    assert summary["global"]["full_model_latency"]["pass"] is True
    assert summary["bootstrap_draw_plan_sha256"] == (
        prereg.bootstrap_draw_plan_v29h()[2]
    )


def test_v29h_meaningful_global_speed_and_memory_gates_fail_closed():
    slow = runtime.performance_summary_v29h(
        synthetic_pair_results(latency_ratio=1.001)
    )
    assert slow["global"]["full_model_latency"]["pass"] is False
    assert slow["all_ten_performance_endpoints_passed"] is False
    memory = runtime.performance_summary_v29h(
        synthetic_pair_results(memory_ratio=1.03)
    )
    assert all(
        endpoint["pass"] is False
        for endpoint in memory["peak_vram_by_physical_gpu"].values()
    )
    assert memory["global"]["peak_vram"]["pass"] is False


def test_v29h_gate_authority_is_narrow_on_pass_and_failure():
    equivalence = {"all_eight_pairs_exact": True}
    performance = {
        "all_ten_performance_endpoints_passed": True,
        "absolute_nvml_gate_passed": True,
    }
    integrity = {"all_runtime_integrity_gates_passed": True}
    gate = runtime.gate_v29h(equivalence, performance, integrity)
    assert gate["pass"] is True
    assert gate["decision"] == (
        "authorize_only_exact_v29_table_in_a_separately_frozen_"
        "serialized_fp8_train_only_recipe_ab"
    )
    for key in (
        "direct_table_or_recipe_adoption_authorized",
        "model_update_or_training_authorized", "checkpoint_write_authorized",
        "dataset_promotion_authorized",
        "evaluation_validation_heldout_ood_or_benchmark_access_authorized",
        "nontrain_runtime_reuse_authorized",
    ):
        assert gate[key] is False
    integrity["all_runtime_integrity_gates_passed"] = False
    assert runtime.gate_v29h(equivalence, performance, integrity)["decision"] == (
        "retain_empty_default_serialized_fp8_runtime"
    )


def test_v29h_compact_contract_rejects_raw_runtime_or_semantic_values():
    for value in (
        {"prompt_token_ids": [1]}, {"token_ids": [2]}, {"text": "x"},
        {"outputs": []}, {"logprobs": []}, {"elapsed_ns": 3},
        {"memory_samples": []}, {"pid": 4}, {"bootstrap_draws": []},
    ):
        with pytest.raises(RuntimeError, match="forbidden keys"):
            runtime._assert_compact_v29h(value)
    runtime._assert_compact_v29h({
        "median_default_over_tuned_ratio": 1.01,
        "commitment_sha256": "a" * 64,
        "dataset_or_semantic_content_opened": False,
    })


def test_v29h_worktree_allowlist_is_exact_and_tracked_changes_fail():
    accepted = runtime.validate_worktree_status_v29h(
        "?? data/manual_reviews/context_merit_audit_v390/a.json\n"
        "?? experiments/dataset_probes/a.json\n"
        "?? experiments/gpu_utilization_v31_base_qwen36_train_reward_probe.jsonl\n"
    )
    assert accepted["allowed_untracked_entry_count"] == 3
    for status in (
        "?? data/manual_reviews/context_merit_audit_v389/a.json\n",
        "?? unrelated.txt\n", " M tracked.py\n",
    ):
        with pytest.raises(RuntimeError, match="committed-clean"):
            runtime.validate_worktree_status_v29h(status)


def test_v29h_argv_rejects_dataset_eval_update_checkpoint_and_bf16_paths():
    runtime._assert_closed_argv_v29h(["--v29h-dry-run"])
    for token in (
        "--dataset=x", "--eval", "--checkpoint=x", "--update",
        "--heldout=x", "--bf16-model=x",
    ):
        with pytest.raises(ValueError, match="forbidden surface"):
            runtime._assert_closed_argv_v29h([token])


def test_v29h_monitor_requires_all_four_exact_active_pids(monkeypatch):
    class FakeRay:
        def wait(self, pending, num_returns, timeout):
            del num_returns, timeout
            time.sleep(0.01)
            return [pending[0]], pending[1:]

        @staticmethod
        def get(_ready):
            return None

    fake_nvml = SimpleNamespace()
    fake_nvml.NVMLError_NotSupported = type("NVMLError_NotSupported", (Exception,), {})
    fake_nvml.nvmlInit = lambda: None
    fake_nvml.nvmlShutdown = lambda: None
    fake_nvml.nvmlDeviceGetHandleByIndex = lambda index: index
    fake_nvml.nvmlDeviceGetComputeRunningProcesses = lambda gpu: [
        SimpleNamespace(pid=1000 + gpu)
    ]
    fake_nvml.nvmlDeviceGetGraphicsRunningProcesses = lambda _gpu: []
    fake_nvml.nvmlDeviceGetUtilizationRates = lambda _gpu: SimpleNamespace(gpu=80)
    fake_nvml.nvmlDeviceGetMemoryInfo = lambda _gpu: SimpleNamespace(
        used=50, total=100,
    )
    monkeypatch.setitem(__import__("sys").modules, "pynvml", fake_nvml)
    result = runtime.monitor_futures_v29h(
        FakeRay(), [object(), object(), object(), object()],
        {gpu: 1000 + gpu for gpu in range(4)},
    )
    assert result[
        "all_four_assigned_pids_and_positive_utilization_simultaneously"
    ] is True
    assert result["simultaneous_positive_sample_count"] == 4
    assert result["peak_nvml_fraction"] == 0.5


def test_v29h_group_loop_aborts_before_later_groups(monkeypatch):
    calls = []

    def fail_third(arm, pair_index, group_index, _requests):
        calls.append((arm, pair_index, group_index))
        if group_index == 2:
            raise RuntimeError("synthetic group failure")
        contract = [{"generated_token_id": gpu} for gpu in range(4)]
        result = {
            "reference_contracts": contract,
            "reference_commitment_sha256": "a" * 64,
            "timed_commitments_sha256": "b" * 64,
            "all_four_engines_generated_every_call": True,
            "actor_identity_commitment_sha256": "c" * 64,
            "cleanup_certificate_sha256": "d" * 64,
            "activity": {
                "commitment_sha256": "e" * 64,
                "all_four_assigned_pids_and_positive_utilization_simultaneously": True,
                "sample_count": 10,
                "simultaneous_positive_sample_count": 2,
                "peak_nvml_fraction": 0.7,
            },
            "elapsed_ns": np.ones((7, 4), dtype=np.int64),
            "peak_allocated_bytes": np.ones((7, 4), dtype=np.int64),
            "peak_reserved_bytes": np.ones((7, 4), dtype=np.int64),
            "peak_nvml_fraction": np.full((7, 4), 0.7),
        }
        idle = {
            "all_four_idle": True,
            "content_sha256_before_self_field": "f" * 64,
        }
        return result, idle

    monkeypatch.setattr(runtime, "run_one_group_v29h", fail_third)
    with pytest.raises(RuntimeError, match="synthetic group failure"):
        runtime.run_counterbalanced_v29h(
            prereg.build_preregistration_v29h(),
            {"all_four_idle": True},
        )
    assert len(calls) == 3
    assert calls[-1][2] == 2


def test_v29h_dry_run_is_cpu_only_and_exposes_exact_handoff(capsys):
    payload = runtime.main(["--v29h-dry-run"])
    assert json.loads(capsys.readouterr().out) == payload
    assert payload["gpu_launched"] is False
    assert payload["dataset_or_semantic_content_opened"] is False
    assert payload["pair_count"] == 8
    assert payload["fresh_four_tp1_engine_group_count"] == 16
    assert payload["serialized_fp8_tp1_model_load_count"] == 64
    assert payload["total_generation_request_budget"] == 640
    assert payload["total_prompt_token_budget"] == 614_400
    assert payload["synchronized_activity_witness_count"] == 64
    assert payload["required_python"].endswith("es-at-scale/.venv/bin/python")
    assert len(payload["implementation_bundle_sha256"]) == 64
    assert len(payload["recipe_sha256"]) == 64
