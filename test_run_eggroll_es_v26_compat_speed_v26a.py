#!/usr/bin/env python3
"""CPU-only tests for the counterbalanced V26A runtime launcher."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import eggroll_es_v26_compat_speed_preregistration_v26a as prereg
import run_eggroll_es_v26_compat_speed_v26a as runtime


def _equivalence(pass_value=True):
    reference = np.linspace(-8.0, -0.1, 280, dtype=np.float64).reshape(5, 56)
    candidate = reference + (0.001 if pass_value else 1.0)
    tokens = np.arange(280, dtype=np.int64)
    return runtime.equivalence_metrics_v26a(
        reference, candidate, tokens, tokens.copy()
    )


def _performance(*, hybrid_elapsed_ns=950_000_000):
    elapsed = np.full((2, 7, 4), 1_000_000_000, dtype=np.int64)
    for gpu_id in prereg.PHYSICAL_GPU_IDS_V26A:
        hybrid_wave = 1 if gpu_id in (0, 2) else 0
        elapsed[hybrid_wave, :, gpu_id] = hybrid_elapsed_ns
    peak = np.full_like(elapsed, 720)
    resident = {
        f"{wave}_gpu_{gpu_id}": {"used_bytes": 700, "total_bytes": 1_000}
        for wave in prereg.WAVE_ORDER_V26A
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V26A
    }
    return runtime.speed_memory_summary_v26a(elapsed, 10_000, resident, peak)


def test_load_and_recipe_preserve_exact_two_wave_crossover_and_narrow_authority():
    value = runtime.load_preregistration_v26a()
    implementation = {"bundle_sha256": "implementation-test"}
    recipe = runtime.recipe_v26a(value, implementation)
    assert tuple(recipe["waves"]) == ("wave_a", "wave_b")
    assert recipe["runtime_contract"]["wave_backend_mapping"] == {
        "wave_a": [
            "full_fp8", "fp8_routed_bf16_backbone_v26",
            "full_fp8", "fp8_routed_bf16_backbone_v26",
        ],
        "wave_b": [
            "fp8_routed_bf16_backbone_v26", "full_fp8",
            "fp8_routed_bf16_backbone_v26", "full_fp8",
        ],
    }
    assert recipe["runtime_contract"][
        "total_generation_requests_all_engines_all_waves"
    ] == 22_400
    assert recipe["gate"]["pass_authority"] == (
        "authorize_only_a_separate_fresh_preregistered_train_only_training_A_B"
    )
    assert recipe["direct_model_adoption_allowed"] is False
    assert recipe["evaluation_allowed"] is False


def test_equivalence_metrics_apply_all_preregistered_thresholds():
    passed = _equivalence(True)
    assert passed["pass"] is True
    assert passed["metrics"]["mean_absolute_logprob_delta"] == pytest.approx(0.001)
    failed = _equivalence(False)
    assert failed["pass"] is False
    assert failed["metrics"]["maximum_absolute_logprob_delta"] == pytest.approx(1.0)


def test_speed_memory_is_paired_on_each_physical_gpu_and_reports_order_strata():
    performance = _performance()
    assert tuple(performance["pairs"]) == prereg.PAIR_ORDER_V26A
    assert len(performance["cells"]) == 8
    assert len(performance["memory"]) == 8
    assert all(item["pass"] is True for item in performance["pairs"].values())
    assert performance["pairs"]["physical_gpu_0"]["load_order"] == "fp8_first"
    assert performance["pairs"]["physical_gpu_1"]["load_order"] == "hybrid_first"
    assert performance["load_order_strata"]["fp8_first"][
        "physical_gpu_ids"
    ] == [0, 2]
    assert performance["load_order_strata"]["hybrid_first"][
        "physical_gpu_ids"
    ] == [1, 3]
    assert performance["bootstrap"]["draw_plan_sha256"] == (
        prereg.timing_bootstrap_draw_plan_v26a()[1]
    )
    assert performance["bootstrap"][
        "raw_draws_replicates_and_timing_vectors_persisted"
    ] is False


def test_speed_gate_fails_one_slow_physical_gpu_pair_and_is_conjunctive():
    performance = _performance()
    elapsed = np.full((2, 7, 4), 1_000_000_000, dtype=np.int64)
    elapsed[1, :, 0] = 1_200_000_000
    for gpu_id in (1, 2, 3):
        hybrid_wave = 1 if gpu_id in (0, 2) else 0
        elapsed[hybrid_wave, :, gpu_id] = 950_000_000
    resident = {
        f"{wave}_gpu_{gpu_id}": {"used_bytes": 700, "total_bytes": 1_000}
        for wave in prereg.WAVE_ORDER_V26A
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V26A
    }
    failed_performance = runtime.speed_memory_summary_v26a(
        elapsed, 10_000, resident, np.full_like(elapsed, 720)
    )
    assert failed_performance["pairs"]["physical_gpu_0"]["pass"] is False
    equivalence = {pair: _equivalence(True) for pair in prereg.PAIR_ORDER_V26A}
    integrity = {"all_integrity_audits_passed": True}
    passed_gate = runtime.evaluate_gate_v26a(equivalence, performance, integrity)
    assert passed_gate["pass"] is True
    assert passed_gate["direct_model_adoption_authorized"] is False
    failed_gate = runtime.evaluate_gate_v26a(
        equivalence, failed_performance, integrity
    )
    assert failed_gate["pass"] is False
    assert failed_gate["decision"] == (
        "retain_existing_FP8_model_for_any_later_training_A_B"
    )


def test_launcher_source_requires_fresh_wave_boundary_and_one_actor_per_gpu():
    launch_source = inspect.getsource(runtime.V26ProbeController.launch_engines)
    orchestration_source = inspect.getsource(runtime.run_counterbalanced_probe_v26a)
    assert "num_gpus=1" in launch_source
    assert "tensor_parallel_size=1" in launch_source
    assert "for rank in prereg.PHYSICAL_GPU_IDS_V26A" in launch_source
    assert "PlacementProbeV26A" in launch_source
    assert "strategies_by_gpu[rank]" in launch_source
    assert 'lifetime="detached"' not in launch_source
    assert "set(strategies_by_gpu) != set(prereg.PHYSICAL_GPU_IDS_V26A)" in (
        launch_source
    )
    assert "nvml_uuid" in launch_source
    assert "pci_bus_id" in launch_source
    assert "for wave_index, wave in enumerate(prereg.WAVE_ORDER_V26A)" in (
        orchestration_source
    )
    assert orchestration_source.index("controller.close()") < (
        orchestration_source.index("wait_for_interwave_gpu_idle_v26a(")
    )
    assert "preparation_controller._prepare_train_requests()" in orchestration_source
    assert "baseline_physical" in orchestration_source
    assert "interwave physical-GPU identity changed" in orchestration_source


def test_v26a_ray_gpu_id_canonicalization_is_strict_and_representation_only():
    for gpu_id in prereg.PHYSICAL_GPU_IDS_V26A:
        assert runtime.normalize_ray_gpu_id_v26a(gpu_id) == gpu_id
        assert runtime.normalize_ray_gpu_id_v26a(str(gpu_id)) == gpu_id
    for invalid in (True, False, -1, 4, "00", "4", "GPU-0", 0.0, None, [0]):
        with pytest.raises(RuntimeError, match="Ray GPU ID"):
            runtime.normalize_ray_gpu_id_v26a(invalid)


def test_v26a_interwave_cleanup_poll_is_bounded_and_preserves_identity(monkeypatch):
    gpus = [
        {
            "physical_gpu_id": gpu_id,
            "nvml_uuid": f"GPU-{gpu_id}",
            "pci_bus_id": f"0000:{gpu_id:02x}:00.0",
            "total_bytes": 1_000,
            "running_process_count": 0,
        }
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V26A
    ]
    baseline = {"all_four_idle": True, "gpus": gpus}
    observations = []
    for process_count in (1, 1, 0):
        rows = [dict(item) for item in gpus]
        rows[0]["running_process_count"] = process_count
        observations.append({
            "all_four_idle": process_count == 0,
            "gpus": rows,
        })
    monkeypatch.setattr(
        runtime, "_observe_all_four_gpus_v26a", lambda: observations.pop(0)
    )
    clock = [0.0]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        runtime.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    certificate = runtime.wait_for_interwave_gpu_idle_v26a(baseline)
    assert certificate["all_four_idle"] is True
    assert certificate["poll_count"] == 3
    assert certificate["elapsed_milliseconds"] == 1_000
    assert certificate["bounded_async_cleanup_wait"] is True


def test_v26a_interwave_cleanup_poll_times_out_fail_closed(monkeypatch):
    gpus = [
        {
            "physical_gpu_id": gpu_id,
            "nvml_uuid": f"GPU-{gpu_id}",
            "pci_bus_id": f"0000:{gpu_id:02x}:00.0",
            "total_bytes": 1_000,
            "running_process_count": 1,
        }
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V26A
    ]
    baseline_rows = [dict(item, running_process_count=0) for item in gpus]
    monkeypatch.setattr(
        runtime, "_observe_all_four_gpus_v26a",
        lambda: {"all_four_idle": False, "gpus": gpus},
    )
    times = iter((0.0, 31.0))
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(times))
    with pytest.raises(RuntimeError, match="within 30 seconds"):
        runtime.wait_for_interwave_gpu_idle_v26a({
            "all_four_idle": True, "gpus": baseline_rows,
        })


def test_real_launch_checks_environment_idle_models_before_attempt_claim():
    source = inspect.getsource(runtime.run_exact_v26a)
    ordered = [
        "certify_runtime_environment_r2()",
        "_source_provenance_v23a(implementation)",
        "live_model_audit_v26a(preregistration)",
        "assert_all_four_gpus_idle_v26a()",
        "_exclusive_write_json_v23a(attempt_path, attempt)",
        "run_counterbalanced_probe_v26a(preregistration, gpu_idle)",
    ]
    positions = [source.index(item) for item in ordered]
    assert positions == sorted(positions)


def test_dry_run_never_checks_or_launches_gpus(monkeypatch, capsys):
    monkeypatch.setattr(
        runtime, "assert_all_four_gpus_idle_v26a",
        lambda: (_ for _ in ()).throw(AssertionError("GPU check called")),
    )
    monkeypatch.setattr(
        runtime, "live_model_audit_v26a",
        lambda _value: (_ for _ in ()).throw(AssertionError("model audit called")),
    )
    payload = runtime.main(["--v26a-dry-run"])
    capsys.readouterr()
    assert payload["gpu_launched"] is False
    assert payload["gpu_idle_check_executed"] is False
    assert payload[
        "direct_adoption_update_checkpoint_evaluation_authorized"
    ] is False


def test_real_runtime_requires_exact_hashes_and_compact_outputs(monkeypatch):
    value = runtime.load_preregistration_v26a()
    implementation = {"bundle_sha256": "implementation-test"}
    recipe = runtime.recipe_v26a(value, implementation)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    for key in runtime.runtime_v23a.MOE_OVERRIDE_ENVIRONMENT_V23A:
        monkeypatch.delenv(key, raising=False)
    args = SimpleNamespace(
        v26a_dry_run=False,
        expected_implementation_bundle_sha256=None,
        expected_recipe_sha256=None,
    )
    with pytest.raises(ValueError, match="expected implementation hash"):
        runtime.validate_runtime_v26a(args, value, implementation, recipe)
    with pytest.raises(RuntimeError, match="forbidden keys"):
        runtime._assert_compact({"scores": []})
