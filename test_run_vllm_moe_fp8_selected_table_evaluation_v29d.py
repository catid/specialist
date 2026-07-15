#!/usr/bin/env python3

import copy
import inspect
import json
from types import SimpleNamespace

import pytest

import build_vllm_moe_fp8_selected_table_evaluation_preregistration_v29d as prereg
import run_vllm_moe_fp8_selected_table_evaluation_v29d as runtime


def _records():
    records = []
    for repetition, seed in enumerate(prereg.SEEDS_V29D):
        arms = {"default": [], "tuned": []}
        for gpu, batch in prereg.BATCH_BY_GPU_V29D.items():
            base = {
                "physical_gpu_id": gpu,
                "batch_size": batch,
                "seed": seed,
                "input_recipe_sha256": f"input-{repetition}-{gpu}",
                "output_sha256": f"output-{repetition}-{gpu}",
                "output_shape": [batch, 2048],
                "output_dtype": "torch.bfloat16",
                "peak_memory_allocated_bytes": 1000,
                "peak_memory_reserved_bytes": 1200,
            }
            arms["default"].append({
                **base, "arm": "default", "kernel_time_microseconds": 101.0,
            })
            arms["tuned"].append({
                **base, "arm": "tuned", "kernel_time_microseconds": 100.0,
            })
        records.append({"repetition": repetition, "seed": seed, "arms": arms})
    return records


def test_v29d_runner_binds_prereg_and_exact_closed_recipe():
    value = runtime.load_preregistration_v29d()
    implementation = runtime.implementation_identity_v29d()
    recipe = runtime.recipe_v29d(value, implementation)
    assert recipe["preregistration"]["file_sha256"] == runtime.PREREG_FILE_SHA256_V29D
    assert recipe["preregistration"]["content_sha256"] == runtime.PREREG_CONTENT_SHA256_V29D
    assert recipe["selected_table"]["file_sha256"] == prereg.TABLE_FILE_SHA256_V29D
    assert recipe["kernel_contract"]["dtype"] == "fp8_w8a8"
    assert recipe["strict_synthetic_kernel_only"] is True
    assert recipe["authority"]["direct_table_adoption_authorized"] is False


def test_v29d_dry_run_is_gpu_and_live_audit_free(monkeypatch, capsys):
    monkeypatch.setattr(
        runtime, "assert_all_four_gpus_idle_v29d",
        lambda: (_ for _ in ()).throw(AssertionError("GPU called")),
    )
    monkeypatch.setattr(
        runtime, "live_cpu_disk_audit_v29d",
        lambda: (_ for _ in ()).throw(AssertionError("live audit called")),
    )
    value = runtime.main(["--v29d-dry-run"])
    captured = json.loads(capsys.readouterr().out)
    assert captured == value
    assert value["gpu_launched"] is False
    assert value["evaluation_launched"] is False


def test_v29d_real_launch_requires_explicit_flag_all_hashes_and_commit(monkeypatch):
    preregistration = runtime.load_preregistration_v29d()
    implementation = runtime.implementation_identity_v29d()
    recipe = runtime.recipe_v29d(preregistration, implementation)
    args = SimpleNamespace(
        v29d_dry_run=False, launch_v29d=False,
        expected_preregistration_commit=None,
        expected_implementation_bundle_sha256=None,
        expected_recipe_sha256=None,
    )
    with pytest.raises(ValueError, match="explicit --launch-v29d"):
        runtime.validate_launch_arguments_v29d(args, implementation, recipe)
    args.launch_v29d = True
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    with pytest.raises(ValueError, match="implementation hash"):
        runtime.validate_launch_arguments_v29d(args, implementation, recipe)


def test_v29d_ray_gpu_id_canonicalization_is_strict():
    for gpu in prereg.PHYSICAL_GPU_IDS_V29D:
        assert runtime.normalize_ray_gpu_id_v29d(gpu) == gpu
        assert runtime.normalize_ray_gpu_id_v29d(str(gpu)) == gpu
    for bad in (True, False, -1, 4, "00", "4", "GPU-0", 0.0, None, [0]):
        with pytest.raises(RuntimeError, match="Ray GPU ID"):
            runtime.normalize_ray_gpu_id_v29d(bad)


def test_v29d_paired_bootstrap_passes_only_all_conjunctive_gates():
    records = _records()
    summary = runtime.summarize_v29d(records, bootstrap_resamples=1000)
    assert summary["pass"] is True
    assert summary["exact_output_equivalence"]["matched_pairs"] == 32
    assert summary["global"]["familywise_latency_lower_bound"] == pytest.approx(1.01)
    assert summary["global"]["familywise_peak_vram_upper_bound"] == 1.0
    assert all(item["latency_gate_pass"] for item in summary["per_gpu"].values())
    assert all(item["peak_vram_gate_pass"] for item in summary["per_gpu"].values())

    output_failure = copy.deepcopy(records)
    output_failure[0]["arms"]["tuned"][0]["output_sha256"] = "changed"
    assert runtime.summarize_v29d(output_failure, bootstrap_resamples=100)["pass"] is False

    latency_failure = copy.deepcopy(records)
    for record in latency_failure:
        record["arms"]["tuned"][1]["kernel_time_microseconds"] = 102.0
    assert runtime.summarize_v29d(latency_failure, bootstrap_resamples=100)["pass"] is False

    vram_failure = copy.deepcopy(records)
    for record in vram_failure:
        record["arms"]["tuned"][2]["peak_memory_reserved_bytes"] = 1201
    assert runtime.summarize_v29d(vram_failure, bootstrap_resamples=100)["pass"] is False


def test_v29d_changed_paired_input_identity_fails_closed():
    records = _records()
    records[0]["arms"]["tuned"][0]["input_recipe_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="deterministic input identity"):
        runtime.summarize_v29d(records, bootstrap_resamples=10)


def test_v29d_inflight_monitor_requires_simultaneous_exact_pid_uuid_activity(monkeypatch):
    rows = [
        {
            "physical_gpu_id": gpu,
            "nvml_uuid": prereg.v29b.GPU_IDENTITIES_V29B[gpu]["uuid"],
            "running_process_pids": [1000 + gpu],
            "gpu_utilization_percent": 20 + gpu,
        }
        for gpu in prereg.PHYSICAL_GPU_IDS_V29D
    ]
    monkeypatch.setattr(
        runtime, "_observe_all_four_gpus_v29d",
        lambda: {"all_four_idle": False, "gpus": rows},
    )

    class FakeRay:
        @staticmethod
        def wait(pending, num_returns, timeout):
            return [pending[0]], pending[1:]

        @staticmethod
        def get(_values):
            return None

    result = runtime._monitor_four_futures_v29d(
        FakeRay, [0, 1, 2, 3], {gpu: 1000 + gpu for gpu in range(4)},
    )
    assert result["pass"] is True
    rows[2]["gpu_utilization_percent"] = 0
    with pytest.raises(RuntimeError, match="all four assigned PIDs"):
        runtime._monitor_four_futures_v29d(
            FakeRay, [0], {gpu: 1000 + gpu for gpu in range(4)},
        )


def test_v29d_runner_uses_exact_official_fp8_kernel_and_closed_surfaces():
    worker = inspect.getsource(runtime._worker_evaluate_v29d)
    source = inspect.getsource(runtime)
    assert "module.benchmark_config(" in worker
    assert "selected, batch_size, 256, 1024, 2048, 8, dtype" in worker
    assert "True, False, False" in worker
    assert "block_quant_shape=[128, 128]" in worker
    assert "module.fused_experts = capture_fused_experts" in worker
    assert "torch.cuda.max_memory_allocated()" in worker
    assert "torch.cuda.max_memory_reserved()" in worker
    assert "ray.init(num_gpus=4, include_dashboard=False)" in source
    assert "workers[gpu].__ray_call__.remote(lambda self: os.getpid())" in source
    for forbidden in ("load_frozen_train(", "heldout.arrow", "ood_qa.arrow"):
        assert forbidden not in source


def test_v29d_compact_output_rejects_raw_payloads():
    for key in ("raw_pids", "timing_vectors", "output_tensor", "heldout_rows"):
        with pytest.raises(RuntimeError, match="forbidden keys"):
            runtime._assert_compact_v29d({key: []})
