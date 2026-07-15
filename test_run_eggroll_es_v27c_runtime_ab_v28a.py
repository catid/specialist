#!/usr/bin/env python3
"""CPU-only tests for the fail-closed V28A V27C task-runtime launcher."""

from __future__ import annotations

import copy
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import eggroll_es_v27c_runtime_ab_preregistration_v28a as prereg
import run_eggroll_es_v27c_runtime_ab_v28a as runtime


def _output(offset=0.0, *, token=7, text="x"):
    gold = np.linspace(-8.0, -0.1, 280, dtype=np.float64).reshape(5, 56)
    gold = gold + offset
    return {
        "gold_mean_logprobs": gold,
        "gold_dense_commitments": [f"dense-{index}-{offset}" for index in range(5)],
        "generated_token_ids": [[token] for _ in range(280)],
        "generated_text": [text for _ in range(280)],
        "generated_selected_logprobs": np.full(280, -0.25 + offset),
        "generated_cumulative_logprobs": np.full(280, -0.25 + offset),
        "generated_decoded_tokens": [text for _ in range(280)],
    }


def _performance(*, tuned_elapsed=950_000_000, tuned_peak=790):
    elapsed = np.full((2, 9, 4), 1_000_000_000, dtype=np.int64)
    peaks = np.full((2, 9, 4), 800, dtype=np.int64)
    resident = {}
    for wave_index, wave in enumerate(prereg.WAVE_ORDER_V28A):
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V28A:
            arm = prereg.WAVE_ARM_ORDER_V28A[wave][gpu_id]
            if arm == "v27c_tuned":
                elapsed[wave_index, :, gpu_id] = tuned_elapsed
                peaks[wave_index, :, gpu_id] = tuned_peak
            resident[f"{wave}_gpu_{gpu_id}"] = {
                "used_bytes": 700, "total_bytes": 1_000,
            }
    return runtime.performance_summary_v28a(elapsed, 10_000, resident, peaks)


def _equivalence(value=True):
    return {
        pair: {"exact_output_equivalence": value}
        for pair in prereg.PAIR_ORDER_V28A
    }


def test_v28a_load_and_recipe_bind_bf16_model_exact_table_and_narrow_authority():
    value = runtime.load_preregistration_v28a()
    recipe = runtime.recipe_v28a(value, {"bundle_sha256": "implementation-test"})
    assert recipe["positive_evidence_commit"] == (
        "76e3b10730e5a7be9c5f2c298bc75095abf5a9c8"
    )
    assert recipe["model_contract"]["path"] == (
        "/home/catid/specialist/models/Qwen3.6-35B-A3B"
    )
    assert recipe["tuned_table_contract"]["file_sha256"] == (
        prereg.TUNED_TABLE_FILE_SHA256_V28A
    )
    assert recipe["only_intended_arm_difference"] == "VLLM_TUNED_CONFIG_FOLDER"
    assert recipe["runtime_environment_contract_r2"]["versions"] == {
        "ray": "2.56.0", "torch": "2.11.0+cu130", "vllm": "0.25.0",
    }
    assert recipe["gate"]["pass_authority"] == (
        "authorize_only_a_separately_frozen_train_only_training_recipe_A_B"
    )
    assert recipe["direct_recipe_adoption_allowed"] is False
    assert recipe["evaluation_allowed"] is False


def test_v28a_exact_output_components_have_zero_numeric_tolerance():
    left = _output()
    right = _output()
    components = runtime.output_equivalence_components_v28a(left, right)
    assert all(components.values())
    assert runtime.exact_output_equal_v28a(left, right) is True
    changed = _output()
    changed["generated_selected_logprobs"][10] = np.nextafter(
        changed["generated_selected_logprobs"][10], 0.0
    )
    changed_components = runtime.output_equivalence_components_v28a(left, changed)
    assert changed_components["generated_selected_logprobs_exact"] is False
    assert runtime.exact_output_equal_v28a(left, changed) is False
    assert runtime.output_commitment_v28a(left) != runtime.output_commitment_v28a(changed)


def test_v28a_every_exact_output_component_is_exercised_independently():
    original = _output()
    mutations = {
        "gold_mean_logprobs_exact": lambda value: value["gold_mean_logprobs"].__setitem__(
            (0, 0), np.nextafter(value["gold_mean_logprobs"][0, 0], 0.0)
        ),
        "gold_dense_commitments_exact": lambda value: value[
            "gold_dense_commitments"
        ].__setitem__(0, "changed"),
        "generated_token_id_lists_exact": lambda value: value[
            "generated_token_ids"
        ][0].__setitem__(0, 8),
        "generated_text_exact": lambda value: value["generated_text"].__setitem__(
            0, "changed"
        ),
        "generated_selected_logprobs_exact": lambda value: value[
            "generated_selected_logprobs"
        ].__setitem__(0, np.nextafter(value["generated_selected_logprobs"][0], 0.0)),
        "generated_cumulative_logprobs_exact": lambda value: value[
            "generated_cumulative_logprobs"
        ].__setitem__(0, np.nextafter(value["generated_cumulative_logprobs"][0], 0.0)),
        "generated_decoded_tokens_exact": lambda value: value[
            "generated_decoded_tokens"
        ].__setitem__(0, "changed"),
    }
    for expected_false, mutate in mutations.items():
        changed = copy.deepcopy(original)
        mutate(changed)
        components = runtime.output_equivalence_components_v28a(original, changed)
        assert components[expected_false] is False
        assert runtime.exact_output_equal_v28a(original, changed) is False


def test_v28a_performance_pairs_every_gpu_and_passes_improvement_fixture():
    performance = _performance()
    assert tuple(performance["pairs"]) == prereg.PAIR_ORDER_V28A
    assert len(performance["cells"]) == 8
    assert len(performance["memory"]) == 8
    assert all(item["throughput_pass"] for item in performance["pairs"].values())
    assert all(item["vram_pass"] for item in performance["pairs"].values())
    assert performance["global_task_throughput"]["pass"] is True
    assert performance["pairs"]["physical_gpu_0"]["load_order"] == "default_first"
    assert performance["pairs"]["physical_gpu_1"]["load_order"] == "tuned_first"
    assert performance["bootstrap"]["draw_plan_sha256"] == (
        prereg.bootstrap_draw_plan_v28a()[2]
    )
    assert performance["bootstrap"][
        "raw_draws_replicates_timing_and_memory_vectors_persisted"
    ] is False


def test_v28a_per_gpu_nonregression_and_global_improvement_gates_are_conjunctive():
    passing = _performance()
    slow = _performance(tuned_elapsed=1_100_000_000)
    assert all(item["throughput_pass"] is False for item in slow["pairs"].values())
    assert slow["global_task_throughput"]["pass"] is False
    integrity = {"all_integrity_audits_passed": True}
    passed_gate = runtime.evaluate_gate_v28a(_equivalence(True), passing, integrity)
    assert passed_gate["pass"] is True
    assert passed_gate["direct_recipe_adoption_authorized"] is False
    failed_speed = runtime.evaluate_gate_v28a(_equivalence(True), slow, integrity)
    assert failed_speed["pass"] is False
    failed_exact = runtime.evaluate_gate_v28a(_equivalence(False), passing, integrity)
    assert failed_exact["pass"] is False
    assert failed_exact["decision"] == "retain_empty_default_config_training_recipe"


def test_v28a_peak_vram_nonregression_is_per_gpu_and_absolute():
    high = _performance(tuned_peak=990)
    assert all(item["vram_pass"] is False for item in high["pairs"].values())
    assert any(
        item["within_absolute_peak_fraction"] is False
        for item in high["memory"].values()
    )


def test_v28a_ray_gpu_id_canonicalization_is_strict():
    for gpu_id in prereg.PHYSICAL_GPU_IDS_V28A:
        assert runtime.normalize_ray_gpu_id_v28a(gpu_id) == gpu_id
        assert runtime.normalize_ray_gpu_id_v28a(str(gpu_id)) == gpu_id
    for invalid in (True, False, -1, 4, "00", "4", "GPU-0", 0.0, None, [0]):
        with pytest.raises(RuntimeError, match="Ray GPU ID"):
            runtime.normalize_ray_gpu_id_v28a(invalid)


def test_v28a_cleanup_polls_wait_and_preserve_identity_for_both_phases(monkeypatch):
    gpus = [
        {
            "physical_gpu_id": gpu_id,
            "nvml_uuid": f"GPU-{gpu_id}",
            "pci_bus_id": f"0000:{gpu_id:02x}:00.0",
            "total_bytes": 1_000,
            "running_process_count": 0,
        }
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V28A
    ]
    baseline = {"all_four_idle": True, "gpus": gpus}
    observations = []
    for process_count in (1, 0):
        rows = [dict(item) for item in gpus]
        rows[0]["running_process_count"] = process_count
        observations.append({"all_four_idle": process_count == 0, "gpus": rows})
    monkeypatch.setattr(
        runtime, "_observe_all_four_gpus_v28a", lambda: observations.pop(0)
    )
    clock = [0.0]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        runtime.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    certificate = runtime.wait_for_interwave_gpu_idle_v28a(baseline)
    assert certificate["cleanup_phase"] == "interwave"
    assert certificate["poll_count"] == 2
    assert certificate["elapsed_milliseconds"] == 500
    assert certificate["bounded_async_cleanup_wait"] is True
    monkeypatch.setattr(
        runtime, "_observe_all_four_gpus_v28a",
        lambda: {"all_four_idle": True, "gpus": [dict(item) for item in gpus]},
    )
    final_certificate = runtime.wait_for_final_gpu_idle_v28a(baseline)
    assert final_certificate["cleanup_phase"] == "final"
    assert final_certificate["poll_count"] == 1
    assert final_certificate["all_four_idle"] is True
    assert final_certificate["bounded_async_cleanup_wait"] is True


def test_v28a_launcher_discovers_placement_and_isolates_actor_config_environment():
    launch = inspect.getsource(runtime.V28ProbeController.launch_engines)
    orchestration = inspect.getsource(runtime.run_counterbalanced_probe_v28a)
    assert "PlacementProbeV28A" in launch
    assert "strategies_by_gpu[gpu_id]" in launch
    assert 'lifetime="detached"' not in launch
    assert "normalize_ray_gpu_id_v28a(ids[0])" in launch
    assert '"env_vars": {"VLLM_TUNED_CONFIG_FOLDER": str(folder)}' in launch
    assert "get_moe_configs(256, 512, None)" in launch
    assert "generic_fallback_none" in launch
    assert "exact_committed_v27c_table" in launch
    assert 'model=spec["model_path"]' in launch
    assert "tensor_parallel_size=1" in launch
    assert orchestration.index("controller.close()") < orchestration.index(
        "wait_for_interwave_gpu_idle_v28a("
    )
    assert orchestration.index("controller.close()") < orchestration.index(
        "wait_for_final_gpu_idle_v28a("
    )
    assert '"final_idle_certificate": final_idle_certificate' in orchestration
    assert '"all_four_gpus_idle_after_wave_b_cleanup"' in orchestration


def test_v28a_real_order_is_cpu_disk_audits_then_immediate_idle_then_claim():
    source = inspect.getsource(runtime.run_exact_v28a)
    ordered = [
        "certify_runtime_environment_r2()",
        "_source_provenance_v23a(implementation)",
        "live_cpu_disk_audit_v28a()",
        "assert_all_four_gpus_idle_v28a()",
        "_exclusive_write_json_v23a(attempt_path, attempt)",
        "run_counterbalanced_probe_v28a(",
    ]
    positions = [source.index(item) for item in ordered]
    assert positions == sorted(positions)
    assert source.index("assert_all_four_gpus_idle_v28a()") > source.index(
        'if attempt_path.exists() or run_directory.exists()'
    )
    audit_source = inspect.getsource(runtime.live_cpu_disk_audit_v28a)
    assert "model_seal_v23a._file_fingerprints" in audit_source
    assert "exact Qwen3.6 all-files fingerprint changed" in audit_source
    assert "exact Qwen3.6 shard content identity changed" in audit_source


def test_v28a_dry_run_opens_no_gpu_model_or_dataset_surface(monkeypatch, capsys):
    monkeypatch.setattr(
        runtime, "assert_all_four_gpus_idle_v28a",
        lambda: (_ for _ in ()).throw(AssertionError("GPU check called")),
    )
    monkeypatch.setattr(
        runtime, "live_cpu_disk_audit_v28a",
        lambda: (_ for _ in ()).throw(AssertionError("model audit called")),
    )
    monkeypatch.setattr(
        runtime.V28ProbeController, "prepare_train_requests_v28a",
        lambda: (_ for _ in ()).throw(AssertionError("dataset called")),
    )
    payload = runtime.main(["--v28a-dry-run"])
    capsys.readouterr()
    assert payload["gpu_launched"] is False
    assert payload["gpu_idle_check_executed"] is False
    assert payload["dataset_rows_or_tokens_opened"] is False
    assert payload[
        "adoption_update_checkpoint_evaluation_or_promotion_authorized"
    ] is False


def test_v28a_real_runtime_requires_exact_hashes_and_compact_outputs(monkeypatch):
    value = runtime.load_preregistration_v28a()
    implementation = {"bundle_sha256": "implementation-test"}
    recipe = runtime.recipe_v28a(value, implementation)
    monkeypatch.delenv("VLLM_TUNED_CONFIG_FOLDER", raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    for key in runtime.MOE_CONFOUNDING_ENV_V28A:
        monkeypatch.delenv(key, raising=False)
    args = SimpleNamespace(
        v28a_dry_run=False,
        expected_implementation_bundle_sha256=None,
        expected_recipe_sha256=None,
    )
    with pytest.raises(ValueError, match="expected implementation hash"):
        runtime.validate_runtime_v28a(args, value, implementation, recipe)
    with pytest.raises(RuntimeError, match="forbidden keys"):
        runtime._assert_compact_v28a({"texts": []})


def test_v28a_source_never_names_or_opens_nontrain_data():
    source = inspect.getsource(runtime)
    for forbidden in (
        "load_frozen_heldout(", "heldout.arrow", "ood_qa.arrow",
        "validation.arrow", "eval_dataloader",
    ):
        assert forbidden not in source
    assert '"model_update_applied": False' in source
    assert '"checkpoint_written": False' in source
    assert '"evaluation_opened": False' in source
    assert '"dataset_promotion_applied": False' in source
