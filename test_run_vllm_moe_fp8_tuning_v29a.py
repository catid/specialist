#!/usr/bin/env python3

import inspect
import json
from types import SimpleNamespace

import pytest

import build_vllm_moe_fp8_tuning_preregistration_v29a as prereg
import run_vllm_moe_fp8_tuning_v29a as runtime


def _config():
    return {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
    }


def test_v29a_runner_binds_exact_preregistration_and_fp8_only_recipe():
    value = runtime.load_preregistration_v29a()
    implementation = runtime.implementation_identity_v29a()
    recipe = runtime.recipe_v29a(value, implementation)
    assert recipe["preregistration"]["file_sha256"] == (
        runtime.PREREG_FILE_SHA256_V29A
    )
    assert recipe["preregistration"]["content_sha256"] == (
        runtime.PREREG_CONTENT_SHA256_V29A
    )
    assert recipe["model_contract"]["path"].endswith("Qwen3.6-35B-A3B-FP8")
    assert recipe["tuning_contract"]["dtype_cli"] == "fp8_w8a8"
    assert recipe["tuning_contract"][
        "block_shape_argument_to_every_official_worker"
    ] == [128, 128]
    assert recipe["no_bf16_v27c_table_path_or_content_is_an_input"] is True
    assert "v27c_tuned_table" not in implementation["files"]


def test_v29a_dry_run_opens_no_gpu_or_live_model_audit(monkeypatch, capsys):
    monkeypatch.setattr(
        runtime, "assert_all_four_gpus_idle_v29a",
        lambda: (_ for _ in ()).throw(AssertionError("GPU check called")),
    )
    monkeypatch.setattr(
        runtime, "live_cpu_disk_audit_v29a",
        lambda: (_ for _ in ()).throw(AssertionError("model audit called")),
    )
    value = runtime.main(["--v29a-dry-run"])
    captured = json.loads(capsys.readouterr().out)
    assert captured == value
    assert value["gpu_launched"] is False
    assert value["tuner_launched"] is False
    assert value["dataset_or_evaluation_surface_opened"] is False


def test_v29a_real_launch_requires_explicit_flag_and_frozen_hashes(monkeypatch):
    value = runtime.load_preregistration_v29a()
    implementation = runtime.implementation_identity_v29a()
    recipe = runtime.recipe_v29a(value, implementation)
    empty = SimpleNamespace(
        v29a_dry_run=False,
        launch_v29a=False,
        expected_implementation_bundle_sha256=None,
        expected_recipe_sha256=None,
    )
    with pytest.raises(ValueError, match="explicit --launch-v29a"):
        runtime.validate_launch_arguments_v29a(empty, implementation, recipe)
    empty.launch_v29a = True
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    with pytest.raises(ValueError, match="expected implementation hash"):
        runtime.validate_launch_arguments_v29a(empty, implementation, recipe)


def test_v29a_ray_gpu_id_canonicalization_is_strict():
    for gpu_id in prereg.PHYSICAL_GPU_IDS_V29A:
        assert runtime.normalize_ray_gpu_id_v29a(gpu_id) == gpu_id
        assert runtime.normalize_ray_gpu_id_v29a(str(gpu_id)) == gpu_id
    for invalid in (True, False, -1, 4, "00", "4", "GPU-0", 0.0, None, [0]):
        with pytest.raises(RuntimeError, match="Ray GPU ID"):
            runtime.normalize_ray_gpu_id_v29a(invalid)


def test_v29a_selected_table_requires_exact_no_space_fp8_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(prereg, "OUTPUT_DIRECTORY_V29A", tmp_path)
    path = tmp_path / prereg.EXPECTED_OUTPUT_FILENAME_V29A
    value = {"triton_version": "3.6.0"}
    value.update({str(batch): _config() for batch in prereg.BATCH_SIZES_V29A})
    path.write_text(json.dumps(value), encoding="utf-8")
    selected = runtime.validate_selected_table_v29a()
    assert selected["configs"]["256"] == _config()
    assert "[128,128]" in selected["path"]
    path.rename(tmp_path / path.name.replace("[128,128]", "[128, 128]"))
    with pytest.raises(RuntimeError, match="filename or file count changed"):
        runtime.validate_selected_table_v29a()


def test_v29a_real_cpu_audit_rehashes_every_model_file_and_weight_shard():
    source = inspect.getsource(runtime.live_cpu_disk_audit_v29a)
    assert "model_seal_v23a._file_fingerprints" in source
    assert "MODEL_ALL_FILES_FINGERPRINT_SHA256_V29A" in source
    assert "MODEL_WEIGHT_SHARDS_V29A" in source
    assert "exact FP8 model all-files identity changed" in source
    assert "exact FP8 model all-shard identity changed" in source
    assert "dataset_train_evaluation_or_nontrain_surface_opened" in source


def test_v29a_official_workers_are_mapped_by_probe_not_creation_order_and_concurrent():
    source = inspect.getsource(runtime.run_official_tuner_v29a)
    assert "strategies_by_gpu[gpu_id]" in source
    assert "normalize_ray_gpu_id_v29a(ids[0])" in source
    assert "set(strategies_by_gpu)" in source
    assert "module.BenchmarkWorker.options" in source
    assert '[{"CPU": 1, "GPU": 1}]' in source
    assert '"CPU": 0.1' not in source
    assert "module.get_configs_compute_bound(False, [128, 128])" in source
    assert source.index("futures = [") < source.index(
        "_monitor_four_tuning_futures_v29a("
    ) < source.index("selected = ray.get(futures)")
    assert "_official_actor_pids_v29a(ray, workers)" in source
    assert "module.save_configs(" in source
    assert 'lifetime="detached"' not in source


def test_v29a_inflight_monitor_requires_process_and_positive_utilization_on_each_uuid(
    monkeypatch,
):
    rows = [
        {
            "physical_gpu_id": gpu_id,
            "nvml_uuid": prereg.GPU_IDENTITIES_V29A[gpu_id]["uuid"],
            "running_process_count": 1,
            "running_process_pids": [1000 + gpu_id],
            "gpu_utilization_percent": gpu_id + 1,
        }
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V29A
    ]
    monkeypatch.setattr(
        runtime, "_observe_all_four_gpus_v29a",
        lambda: {"all_four_idle": False, "gpus": rows},
    )

    class FakeRay:
        @staticmethod
        def wait(pending, num_returns, timeout):
            return [pending[0]], pending[1:]

        @staticmethod
        def get(_ready):
            return None

    result = runtime._monitor_four_tuning_futures_v29a(
        FakeRay,
        ["gpu0", "gpu1", "gpu2", "gpu3"],
        {gpu_id: 1000 + gpu_id for gpu_id in prereg.PHYSICAL_GPU_IDS_V29A},
    )
    per_gpu = result["per_gpu"]
    assert set(per_gpu) == set(prereg.PHYSICAL_GPU_IDS_V29A)
    assert all(item["running_process_observed"] for item in per_gpu.values())
    assert all(
        item["assigned_official_actor_pid_observed"] for item in per_gpu.values()
    )
    assert all(
        item["positive_gpu_utilization_observed"] for item in per_gpu.values()
    )
    assert len({item["nvml_uuid"] for item in per_gpu.values()}) == 4
    assert result[
        "simultaneous_all_four_assigned_actor_pids_and_positive_utilization_observation_count"
    ] >= 1
    assert result["simultaneous_all_four_requirement_passed"] is True
    rows[2]["gpu_utilization_percent"] = 0
    with pytest.raises(RuntimeError, match="positive utilization"):
        runtime._monitor_four_tuning_futures_v29a(
            FakeRay,
            ["only"],
            {gpu_id: 1000 + gpu_id for gpu_id in prereg.PHYSICAL_GPU_IDS_V29A},
        )


def test_v29a_sequential_only_gpu_activity_fails_concurrency_certificate(monkeypatch):
    observations = []
    for active_gpu in prereg.PHYSICAL_GPU_IDS_V29A:
        observations.append({
            "all_four_idle": False,
            "gpus": [
                {
                    "physical_gpu_id": gpu_id,
                    "nvml_uuid": prereg.GPU_IDENTITIES_V29A[gpu_id]["uuid"],
                    "running_process_count": int(gpu_id == active_gpu),
                    "running_process_pids": (
                        [1000 + gpu_id] if gpu_id == active_gpu else []
                    ),
                    "gpu_utilization_percent": int(gpu_id == active_gpu) * 50,
                }
                for gpu_id in prereg.PHYSICAL_GPU_IDS_V29A
            ],
        })
    monkeypatch.setattr(
        runtime, "_observe_all_four_gpus_v29a", lambda: observations.pop(0)
    )

    class FakeRay:
        @staticmethod
        def wait(pending, num_returns, timeout):
            return [pending[0]], pending[1:]

        @staticmethod
        def get(_ready):
            return None

    with pytest.raises(RuntimeError, match="positive utilization"):
        runtime._monitor_four_tuning_futures_v29a(
            FakeRay,
            ["gpu0", "gpu1", "gpu2", "gpu3"],
            {gpu_id: 1000 + gpu_id for gpu_id in prereg.PHYSICAL_GPU_IDS_V29A},
        )


def test_v29a_compiler_policy_is_exact_official_out_of_resources_only():
    source = prereg.OFFICIAL_TUNER_PATH_V29A.read_text(encoding="utf-8")
    assert "except triton.runtime.autotuner.OutOfResources:" in source
    assert "CompilationError" not in source
    assert "PassManager::run failed" not in source
    runtime_source = inspect.getsource(runtime.run_official_tuner_v29a)
    assert "CompilationError" not in runtime_source
    assert "PassManager::run failed" not in runtime_source
    policy = runtime.load_preregistration_v29a()["compiler_error_policy"]
    assert policy["additional_CompilationError_skip_authorized"] is False
    assert policy["additional_RuntimeError_skip_authorized"] is False


def test_v29a_real_order_is_environment_model_audit_idle_then_exclusive_claim():
    source = inspect.getsource(runtime.run_exact_v29a)
    ordered = [
        "certify_runtime_environment_r2()",
        "live_cpu_disk_audit_v29a()",
        "assert_all_four_gpus_idle_v29a()",
        "_exclusive_write_json_v29a(prereg.ATTEMPT_PATH_V29A, attempt)",
        "run_official_tuner_v29a(prereg.OUTPUT_DIRECTORY_V29A)",
        "wait_for_final_gpu_idle_v29a()",
    ]
    positions = [source.index(item) for item in ordered]
    assert positions == sorted(positions)
    assert source.index("assert_all_four_gpus_idle_v29a()") > source.index(
        "OUTPUT_DIRECTORY_V29A.exists()"
    )


@pytest.mark.parametrize("failure_surface", ("selected_table", "report_write"))
def test_v29a_post_tuner_finalization_failure_rewrites_durable_attempt_failed(
    monkeypatch, tmp_path, failure_surface,
):
    attempt_path = tmp_path / ".attempt.json"
    report_path = tmp_path / "report" / "report.json"
    output_directory = tmp_path / "selected"
    monkeypatch.setattr(prereg, "ATTEMPT_PATH_V29A", attempt_path)
    monkeypatch.setattr(prereg, "REPORT_PATH_V29A", report_path)
    monkeypatch.setattr(prereg, "OUTPUT_DIRECTORY_V29A", output_directory)
    environment = runtime._seal({"schema": "environment"})
    live_audit = runtime._seal({"schema": "live-audit"})
    prelaunch = runtime._seal({"schema": "prelaunch", "all_four_idle": True})
    final_idle = runtime._seal({"schema": "final", "all_four_idle": True})
    monkeypatch.setattr(
        runtime.runtime_r2, "certify_runtime_environment_r2", lambda: environment
    )
    monkeypatch.setattr(runtime, "live_cpu_disk_audit_v29a", lambda: live_audit)
    monkeypatch.setattr(
        runtime, "assert_all_four_gpus_idle_v29a", lambda: prelaunch
    )
    monkeypatch.setattr(runtime, "wait_for_final_gpu_idle_v29a", lambda: final_idle)
    configs = {batch: _config() for batch in prereg.BATCH_SIZES_V29A}
    monkeypatch.setattr(
        runtime,
        "run_official_tuner_v29a",
        lambda _directory: {"selected_configs": configs},
    )
    selected = {
        "path": str(output_directory / prereg.EXPECTED_OUTPUT_FILENAME_V29A),
        "file_sha256": "1" * 64,
        "content_sha256": "2" * 64,
        "triton_version": "3.6.0",
        "configs": {str(key): value for key, value in configs.items()},
    }
    if failure_surface == "selected_table":
        monkeypatch.setattr(
            runtime,
            "validate_selected_table_v29a",
            lambda: (_ for _ in ()).throw(RuntimeError("synthetic selected failure")),
        )
    else:
        monkeypatch.setattr(runtime, "validate_selected_table_v29a", lambda: selected)
        original_write = runtime._exclusive_write_json_v29a

        def fail_report_write(path, value):
            if path == report_path:
                raise RuntimeError("synthetic report failure")
            return original_write(path, value)

        monkeypatch.setattr(runtime, "_exclusive_write_json_v29a", fail_report_write)
    preregistration = runtime.load_preregistration_v29a()
    implementation = {"bundle_sha256": "implementation"}
    recipe = runtime._seal({
        "schema": "recipe",
        "preregistration": {
            "file_sha256": runtime.PREREG_FILE_SHA256_V29A,
            "content_sha256": runtime.PREREG_CONTENT_SHA256_V29A,
        },
    })
    with pytest.raises(RuntimeError, match="synthetic"):
        runtime.run_exact_v29a(preregistration, implementation, recipe)
    durable = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert durable["status"] == "failed"
    assert durable["phase"] == "after_worker_cleanup_or_finalization_failure"
    assert durable["final_idle_certificate_sha256"] == (
        final_idle["content_sha256_before_self_field"]
    )
    assert durable["failure"]["raw_message_or_traceback_persisted"] is False
    assert not report_path.exists()


def test_v29a_compact_contract_rejects_data_or_raw_search_surfaces():
    with pytest.raises(RuntimeError, match="forbidden keys"):
        runtime._assert_compact_v29a({"validation_rows": []})
    source = inspect.getsource(runtime)
    for forbidden in (
        "load_frozen_train(", "heldout.arrow", "ood_qa.arrow",
        "S6_V27C_MOE_TUNING_SELECTION_EVIDENCE.json",
    ):
        assert forbidden not in source
