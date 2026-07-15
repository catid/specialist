#!/usr/bin/env python3

import ast
import copy
import inspect
import json
import sys
import textwrap
from types import SimpleNamespace

import pytest

import build_vllm_moe_fp8_selected_table_evaluation_retry_preregistration_v29e as prereg
import run_vllm_moe_fp8_selected_table_evaluation_retry_v29e as runtime


def _records():
    records = []
    for repetition, seed in enumerate(prereg.SEEDS_V29E):
        arms = {"default": [], "tuned": []}
        for gpu, batch in prereg.BATCH_BY_GPU_V29E.items():
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


def test_v29e_runner_binds_retry_failure_selection_and_closed_recipe():
    value = runtime.load_preregistration_v29e()
    implementation = runtime.implementation_identity_v29e()
    recipe = runtime.recipe_v29e(value, implementation)
    assert recipe["preregistration"]["file_sha256"] == (
        runtime.PREREG_FILE_SHA256_V29E
    )
    assert recipe["retry_of"]["v29d_failure_evidence_commit"] == (
        prereg.FAILURE_EVIDENCE_COMMIT_V29E
    )
    assert recipe["selection_evidence"]["commit"] == prereg.EVIDENCE_COMMIT_V29E
    assert recipe["kernel_contract"]["official_num_iters"] == 1000
    assert len(recipe["schedule"]["fixed_seeds"]) == 8
    assert len(recipe["statistical_contract"]["endpoints"]) == 10
    assert recipe["authority"]["direct_table_adoption_authorized"] is False
    assert recipe["strict_synthetic_kernel_only"] is True


def test_v29e_dry_run_is_gpu_and_live_audit_free(monkeypatch, capsys):
    monkeypatch.setattr(
        runtime, "assert_all_four_gpus_idle_v29e",
        lambda: (_ for _ in ()).throw(AssertionError("GPU called")),
    )
    monkeypatch.setattr(
        runtime, "live_cpu_disk_audit_v29e",
        lambda: (_ for _ in ()).throw(AssertionError("live audit called")),
    )
    value = runtime.main(["--v29e-dry-run"])
    captured = json.loads(capsys.readouterr().out)
    assert captured == value
    assert value["gpu_launched"] is False
    assert value["evaluation_launched"] is False


def test_v29e_real_launch_requires_explicit_flag_all_hashes_and_commit(monkeypatch):
    value = runtime.load_preregistration_v29e()
    implementation = runtime.implementation_identity_v29e()
    recipe = runtime.recipe_v29e(value, implementation)
    args = SimpleNamespace(
        v29e_dry_run=False, launch_v29e=False,
        expected_preregistration_commit=None,
        expected_implementation_bundle_sha256=None,
        expected_recipe_sha256=None,
    )
    with pytest.raises(ValueError, match="explicit --launch-v29e"):
        runtime.validate_launch_arguments_v29e(args, implementation, recipe)
    args.launch_v29e = True
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    with pytest.raises(ValueError, match="implementation hash"):
        runtime.validate_launch_arguments_v29e(args, implementation, recipe)


def test_v29e_ray_gpu_id_canonicalization_is_strict():
    for gpu in prereg.PHYSICAL_GPU_IDS_V29E:
        assert runtime.normalize_ray_gpu_id_v29e(gpu) == gpu
        assert runtime.normalize_ray_gpu_id_v29e(str(gpu)) == gpu
    for bad in (True, False, -1, 4, "00", "4", "GPU-0", 0.0, None, [0]):
        with pytest.raises(RuntimeError, match="Ray GPU ID"):
            runtime.normalize_ray_gpu_id_v29e(bad)


def test_v29e_activity_witness_synchronizes_each_measured_operation(monkeypatch):
    calls = []

    class FakeCuda:
        @staticmethod
        def synchronize():
            calls.append("sync")

        @staticmethod
        def empty_cache():
            calls.append("empty_cache")

    fake_torch = SimpleNamespace(
        bfloat16="bfloat16",
        cuda=FakeCuda,
        ones=lambda *args, **kwargs: object(),
        empty=lambda *args, **kwargs: object(),
        mm=lambda *args, **kwargs: calls.append("mm"),
    )
    monotonic_values = iter((0.0, 0.0, 0.8, 0.8))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(runtime.time, "time_ns", lambda: 123)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(monotonic_values))
    result = runtime._run_activity_witness_v29e(123)
    assert calls == ["mm", "sync", "mm", "sync", "empty_cache", "sync"]
    assert result["iteration_count"] == 1
    assert result["elapsed_at_least_preregistered_duration"] is True


def test_v29e_witness_source_has_sync_immediately_after_measured_mm():
    tree = ast.parse(textwrap.dedent(inspect.getsource(runtime._run_activity_witness_v29e)))
    measured = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.While)
        and any(
            isinstance(child, ast.Attribute) and child.attr == "monotonic"
            for child in ast.walk(node.test)
        )
    )
    assert isinstance(measured.body[0], ast.Expr)
    assert ast.unparse(measured.body[0].value).startswith("torch.mm(")
    assert isinstance(measured.body[1], ast.Expr)
    assert ast.unparse(measured.body[1].value) == "torch.cuda.synchronize()"


def test_v29e_worker_excludes_witness_then_resets_peak_before_1000_iter_benchmark():
    worker = inspect.getsource(runtime._worker_evaluate_v29e)
    witness_index = worker.index("_run_activity_witness_v29e(")
    reset_index = worker.index("torch.cuda.reset_peak_memory_stats()")
    benchmark_index = worker.index("module.benchmark_config(")
    assert witness_index < reset_index < benchmark_index
    assert "num_iters=prereg.OFFICIAL_NUM_ITERS_V29E" in worker
    assert "torch.cuda.max_memory_allocated()" in worker
    assert "torch.cuda.max_memory_reserved()" in worker


def test_v29e_monitor_uses_50ms_and_requires_simultaneous_exact_pid_activity(monkeypatch):
    rows = [
        {
            "physical_gpu_id": gpu,
            "nvml_uuid": runtime.v29b.GPU_IDENTITIES_V29B[gpu]["uuid"],
            "running_process_pids": [1000 + gpu],
            "gpu_utilization_percent": 20 + gpu,
        }
        for gpu in prereg.PHYSICAL_GPU_IDS_V29E
    ]
    monkeypatch.setattr(
        runtime, "_observe_all_four_gpus_v29e",
        lambda: {"all_four_idle": False, "gpus": rows},
    )
    timeouts = []

    class FakeRay:
        @staticmethod
        def wait(pending, num_returns, timeout):
            timeouts.append(timeout)
            return [pending[0]], pending[1:]

        @staticmethod
        def get(_values):
            return None

    result = runtime._monitor_four_futures_v29e(
        FakeRay, [0, 1, 2, 3], {gpu: 1000 + gpu for gpu in range(4)},
    )
    assert result["pass"] is True
    assert result[
        "simultaneous_all_four_assigned_pids_and_positive_utilization_count"
    ] == 4
    assert timeouts == [0.05] * 4
    rows[2]["gpu_utilization_percent"] = 0
    with pytest.raises(RuntimeError, match="all four assigned PIDs"):
        runtime._monitor_four_futures_v29e(
            FakeRay, [0], {gpu: 1000 + gpu for gpu in range(4)},
        )


def test_v29e_runner_uses_fresh_four_worker_waves_exact_pids_and_common_target():
    arm = inspect.getsource(runtime.run_arm_v29e)
    source = inspect.getsource(runtime)
    assert "ray.init(num_gpus=4, include_dashboard=False)" in arm
    assert "workers[gpu].__ray_call__.remote(lambda self: os.getpid())" in source
    target_index = arm.index("common_activity_target_unix_ns = time.time_ns()")
    futures_index = arm.index("futures = [")
    monitor_index = arm.index("_monitor_four_futures_v29e(")
    assert target_index < futures_index < monitor_index
    assert "ray.shutdown()" in arm


def test_v29e_paired_bootstrap_preserves_all_conjunctive_gates():
    records = _records()
    summary = runtime.summarize_v29e(records, bootstrap_resamples=1000)
    assert summary["pass"] is True
    assert summary["repetitions"] == 8
    assert summary["endpoint_count"] == 10
    assert summary["exact_output_equivalence"]["matched_pairs"] == 32
    assert summary["global"]["familywise_latency_lower_bound"] == pytest.approx(1.01)
    assert summary["global"]["familywise_peak_vram_upper_bound"] == 1.0
    changed = copy.deepcopy(records)
    changed[0]["arms"]["tuned"][0]["output_sha256"] = "changed"
    assert runtime.summarize_v29e(changed, bootstrap_resamples=100)["pass"] is False


def test_v29e_changed_paired_input_identity_fails_closed():
    records = _records()
    records[0]["arms"]["tuned"][0]["input_recipe_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="deterministic input identity"):
        runtime.summarize_v29e(records, bootstrap_resamples=10)


def test_v29e_compact_output_rejects_raw_payloads_and_source_has_no_data_access():
    for key in ("raw_pids", "timing_vectors", "output_tensor", "heldout_rows"):
        with pytest.raises(RuntimeError, match="forbidden keys"):
            runtime._assert_compact_v29e({key: []})
    source = inspect.getsource(runtime)
    for forbidden in ("load_frozen_train(", "heldout.arrow", "ood_qa.arrow"):
        assert forbidden not in source
