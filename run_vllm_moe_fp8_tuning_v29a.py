#!/usr/bin/env python3
"""Fail-closed runner for the preregistered four-GPU FP8 vLLM MoE tuner."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import build_eggroll_es_insertion_model_seal_v23a as model_seal_v23a
import build_vllm_moe_fp8_tuning_preregistration_v29a as prereg
import run_eggroll_es_insertion_stability_v23a_retry_r2 as runtime_r2


ROOT = Path(__file__).resolve().parent
PREREG_PATH_V29A = prereg.OUTPUT_PATH_V29A
PREREG_FILE_SHA256_V29A = (
    "3fabaef03dde9944005937d4d975b4d546f330d66a426580720d0e3da5dc768d"
)
PREREG_CONTENT_SHA256_V29A = (
    "bf605c113bec2730599160618bd8e1b483a4317b024f68a309b80c7ed436f8cb"
)
TEST_PATH_V29A = (ROOT / "test_run_vllm_moe_fp8_tuning_v29a.py").resolve()
PREREG_TEST_PATH_V29A = (
    ROOT / "test_build_vllm_moe_fp8_tuning_preregistration_v29a.py"
).resolve()
MOE_ENVIRONMENT_V29A = (
    "VLLM_TUNED_CONFIG_FOLDER",
    "VLLM_BATCH_INVARIANT",
    "VLLM_MOE_TUNE_CACHE_CLEAR_INTERVAL",
    "VLLM_ROCM_USE_AITER",
    "VLLM_ROCM_USE_AITER_MOE",
    "VLLM_USE_DEEP_GEMM",
    "VLLM_MOE_USE_DEEP_GEMM",
)
ALLOWED_CONFIG_KEYS_V29A = {
    "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
    "num_warps", "num_stages",
}
IMPLEMENTATION_PATHS_V29A = {
    "v29a_preregistration_builder": Path(prereg.__file__).resolve(),
    "v29a_preregistration_tests": PREREG_TEST_PATH_V29A,
    "v29a_preregistration": PREREG_PATH_V29A,
    "v29a_runtime": Path(__file__).resolve(),
    "v29a_runtime_tests": TEST_PATH_V29A,
    "official_vllm_025_tuner": prereg.OFFICIAL_TUNER_PATH_V29A,
    "installed_vllm_fused_moe": prereg.FUSED_MOE_PATH_V29A,
    "installed_vllm_fused_moe_config": prereg.FUSED_MOE_CONFIG_PATH_V29A,
    "installed_vllm_envs": prereg.VLLM_ENVS_PATH_V29A,
    "fp8_model_identity_builder": (
        ROOT / "build_qwen36_fp8_routed_experts_bf16_backbone_v26.py"
    ),
    "fp8_model_identity_auditor": (
        ROOT / "audit_qwen36_fp8_routed_experts_bf16_backbone_v26.py"
    ),
    "v28b_positive_evidence_builder": (
        ROOT / "build_eggroll_es_v28a_runtime_positive_evidence_v28b.py"
    ),
    "v28b_positive_evidence_tests": (
        ROOT / "test_build_eggroll_es_v28a_runtime_positive_evidence_v28b.py"
    ),
    "v28b_positive_evidence": prereg.V28B_EVIDENCE_PATH_V29A,
}
FORBIDDEN_COMPACT_KEYS_V29A = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "prompt_token_ids", "token_ids", "text", "texts", "responses",
    "training_rows", "evaluation_rows", "validation_rows", "heldout_rows",
    "ood_rows", "compiler_log", "progress_log", "timing_vectors",
    "search_results",
}


canonical_sha256 = prereg.canonical_sha256
file_sha256 = prereg.file_sha256


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _seal(value):
    value = copy.deepcopy(value)
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def _recursive_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).lower()
            yield from _recursive_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _recursive_keys(item)


def _assert_compact_v29a(value):
    overlap = FORBIDDEN_COMPACT_KEYS_V29A & set(_recursive_keys(value))
    if overlap:
        raise RuntimeError(f"V29A compact output contains forbidden keys: {sorted(overlap)}")


def _exclusive_write_json_v29a(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def _rewrite_json_v29a(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_preregistration_v29a():
    value = json.loads(PREREG_PATH_V29A.read_text(encoding="utf-8"))
    if (
        file_sha256(PREREG_PATH_V29A) != PREREG_FILE_SHA256_V29A
        or value.get("content_sha256_before_self_field")
        != PREREG_CONTENT_SHA256_V29A
        or canonical_sha256(_without_self(value)) != PREREG_CONTENT_SHA256_V29A
    ):
        raise RuntimeError("V29A preregistration identity changed")
    return prereg.validate_preregistration_v29a(value)


def implementation_identity_v29a():
    files = {
        key: {
            "path": str(Path(path).resolve()),
            "file_sha256": file_sha256(path),
        }
        for key, path in IMPLEMENTATION_PATHS_V29A.items()
    }
    return {
        "files": files,
        "bundle_sha256": canonical_sha256(files),
    }


def recipe_v29a(preregistration, implementation):
    return _seal({
        "schema": "vllm-moe-fp8-tuning-recipe-v29a",
        "experiment_name": prereg.EXPERIMENT_NAME_V29A,
        "preregistration": {
            "path": str(PREREG_PATH_V29A),
            "file_sha256": PREREG_FILE_SHA256_V29A,
            "content_sha256": PREREG_CONTENT_SHA256_V29A,
        },
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "official_source_contract": copy.deepcopy(
            preregistration["official_source_contract"]
        ),
        "model_contract": copy.deepcopy(preregistration["model_contract"]),
        "hardware_contract": copy.deepcopy(preregistration["hardware_contract"]),
        "runtime_environment_contract": copy.deepcopy(
            preregistration["runtime_environment_contract"]
        ),
        "tuning_contract": copy.deepcopy(preregistration["tuning_contract"]),
        "compiler_error_policy": copy.deepcopy(
            preregistration["compiler_error_policy"]
        ),
        "authority": copy.deepcopy(preregistration["authority"]),
        "fresh_exclusive_paths": {
            "attempt": str(prereg.ATTEMPT_PATH_V29A),
            "report": str(prereg.REPORT_PATH_V29A),
            "selected_table_directory": str(prereg.OUTPUT_DIRECTORY_V29A),
        },
        "no_bf16_v27c_table_path_or_content_is_an_input": True,
        "train_evaluation_validation_heldout_ood_or_benchmark_data_access_allowed": False,
    })


def normalize_ray_gpu_id_v29a(value):
    if isinstance(value, bool):
        raise RuntimeError("V29A Ray GPU ID representation changed")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value in {"0", "1", "2", "3"}:
        result = int(value)
    else:
        raise RuntimeError("V29A Ray GPU ID representation changed")
    if result not in prereg.PHYSICAL_GPU_IDS_V29A:
        raise RuntimeError("V29A Ray GPU ID physical range changed")
    return result


def live_cpu_disk_audit_v29a():
    prereg.validate_static_inputs_v29a()
    all_files = model_seal_v23a._file_fingerprints(prereg.MODEL_PATH_V29A)
    if (
        len(all_files) != prereg.MODEL_ALL_FILES_SIZE_MANIFEST_V29A["file_count"]
        or sum(item["bytes"] for item in all_files.values())
        != prereg.MODEL_ALL_FILES_SIZE_MANIFEST_V29A["total_bytes"]
        or canonical_sha256(all_files)
        != prereg.MODEL_ALL_FILES_FINGERPRINT_SHA256_V29A
    ):
        raise RuntimeError("V29A exact FP8 model all-files identity changed")
    index = json.loads(
        (prereg.MODEL_PATH_V29A / "model.safetensors.index.json").read_text(
            encoding="utf-8"
        )
    )
    shard_names = sorted(set(index["weight_map"].values()))
    shard_records = [
        {
            "file": name,
            "bytes": all_files[name]["bytes"],
            "sha256": all_files[name]["sha256"],
        }
        for name in shard_names
    ]
    shard_manifest = {
        "file_count": len(shard_records),
        "total_bytes": sum(item["bytes"] for item in shard_records),
        "manifest_sha256": canonical_sha256(shard_records),
    }
    if shard_manifest != prereg.MODEL_WEIGHT_SHARDS_V29A:
        raise RuntimeError("V29A exact FP8 model all-shard identity changed")
    source = prereg.OFFICIAL_TUNER_PATH_V29A.read_text(encoding="utf-8")
    if (
        "except triton.runtime.autotuner.OutOfResources:" not in source
        or "CompilationError" in source
        or "PassManager::run failed" in source
        or "configs = _distribute(" not in source
        or "save_configs(" not in source
    ):
        raise RuntimeError("V29A official compiler policy or tuner source changed")
    return _seal({
        "schema": "vllm-moe-fp8-tuning-live-cpu-disk-audit-v29a",
        "model_all_files_fingerprint_sha256": canonical_sha256(all_files),
        "model_all_56_files_rehashed": True,
        "model_all_42_weight_shards_rehashed": True,
        "model_weight_shard_manifest": shard_manifest,
        "model_config_sha256": file_sha256(prereg.MODEL_PATH_V29A / "config.json"),
        "model_index_sha256": file_sha256(
            prereg.MODEL_PATH_V29A / "model.safetensors.index.json"
        ),
        "official_tuner_sha256": file_sha256(prereg.OFFICIAL_TUNER_PATH_V29A),
        "installed_fused_moe_sha256": file_sha256(prereg.FUSED_MOE_PATH_V29A),
        "installed_fused_moe_config_sha256": file_sha256(
            prereg.FUSED_MOE_CONFIG_PATH_V29A
        ),
        "installed_vllm_envs_sha256": file_sha256(prereg.VLLM_ENVS_PATH_V29A),
        "official_only_OutOfResources_skip_verified": True,
        "additional_compilation_or_runtime_exception_filter_present": False,
        "dataset_train_evaluation_or_nontrain_surface_opened": False,
    })


def _observe_all_four_gpus_v29a():
    import pynvml

    pynvml.nvmlInit()
    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode("ascii")
        rows = []
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V29A:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            name = pynvml.nvmlDeviceGetName(handle)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            pci = pynvml.nvmlDeviceGetPciInfo(handle).busId
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            if isinstance(uuid, bytes):
                uuid = uuid.decode("ascii")
            if isinstance(pci, bytes):
                pci = pci.decode("ascii")
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            processes = []
            for function_name in (
                "nvmlDeviceGetComputeRunningProcesses",
                "nvmlDeviceGetGraphicsRunningProcesses",
            ):
                function = getattr(pynvml, function_name, None)
                if function is not None:
                    try:
                        processes.extend(function(handle))
                    except pynvml.NVMLError_NotSupported:
                        pass
            process_pids = sorted({int(item.pid) for item in processes})
            rows.append({
                "physical_gpu_id": gpu_id,
                "name": str(name),
                "driver_version": str(driver),
                "nvml_uuid": str(uuid),
                "pci_bus_id": str(pci),
                "total_bytes": int(memory.total),
                "running_process_count": len(process_pids),
                "running_process_pids": process_pids,
                "gpu_utilization_percent": int(utilization.gpu),
            })
    finally:
        pynvml.nvmlShutdown()
    for row in rows:
        expected = prereg.GPU_IDENTITIES_V29A[row["physical_gpu_id"]]
        if (
            row["name"]
            != "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
            or row["driver_version"] != "610.43.02"
            or row["nvml_uuid"] != expected["uuid"]
            or row["pci_bus_id"].upper() != expected["pci_bus_id"].upper()
            or row["total_bytes"] != expected["total_bytes"]
        ):
            raise RuntimeError("V29A exact physical-GPU identity changed")
    return {
        "gpus": rows,
        "all_four_idle": all(row["running_process_count"] == 0 for row in rows),
    }


def assert_all_four_gpus_idle_v29a():
    observation = _observe_all_four_gpus_v29a()
    if observation.get("all_four_idle") is not True:
        raise RuntimeError("V29A requires all four GPUs idle before claim")
    return _seal({
        "schema": "vllm-moe-fp8-tuning-prelaunch-idle-certificate-v29a",
        **observation,
    })


def wait_for_final_gpu_idle_v29a(*, timeout_seconds=30.0, interval_seconds=0.5):
    if timeout_seconds != 30.0 or interval_seconds != 0.5:
        raise RuntimeError("V29A final GPU cleanup polling contract changed")
    started = time.monotonic()
    polls = 0
    while True:
        observation = _observe_all_four_gpus_v29a()
        polls += 1
        elapsed = time.monotonic() - started
        if observation.get("all_four_idle") is True:
            return _seal({
                "schema": "vllm-moe-fp8-tuning-final-idle-certificate-v29a",
                **observation,
                "poll_count": polls,
                "elapsed_milliseconds": int(round(elapsed * 1000.0)),
                "bounded_async_cleanup_wait": True,
            })
        remaining = timeout_seconds - elapsed
        if remaining <= 0:
            raise RuntimeError("V29A final GPU cleanup exceeded 30 seconds")
        time.sleep(min(interval_seconds, remaining))


def _load_official_tuner_v29a():
    if file_sha256(prereg.OFFICIAL_TUNER_PATH_V29A) != prereg.OFFICIAL_TUNER_SHA256_V29A:
        raise RuntimeError("V29A official vLLM tuner identity changed")
    spec = importlib.util.spec_from_file_location(
        "vllm_benchmark_moe_fp8_v29a", prereg.OFFICIAL_TUNER_PATH_V29A
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.triton.__version__ != "3.6.0":
        raise RuntimeError("V29A Triton runtime version changed")
    return module


def _validate_config_v29a(config):
    if (
        not isinstance(config, dict)
        or set(config) != ALLOWED_CONFIG_KEYS_V29A
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in config.values()
        )
    ):
        raise RuntimeError("V29A selected official config geometry changed")
    return config


def validate_selected_table_v29a():
    directory = prereg.OUTPUT_DIRECTORY_V29A
    expected = directory / prereg.EXPECTED_OUTPUT_FILENAME_V29A
    entries = sorted(directory.iterdir())
    if (
        entries != [expected]
        or not expected.is_file()
        or expected.is_symlink()
        or "[128, 128]" in expected.name
    ):
        raise RuntimeError("V29A official FP8 output filename or file count changed")
    value = json.loads(expected.read_text(encoding="utf-8"))
    if (
        value.get("triton_version") != "3.6.0"
        or {key for key in value if key != "triton_version"}
        != {str(batch) for batch in prereg.BATCH_SIZES_V29A}
    ):
        raise RuntimeError("V29A selected table batch coverage changed")
    configs = {
        str(batch): _validate_config_v29a(value[str(batch)])
        for batch in prereg.BATCH_SIZES_V29A
    }
    return {
        "path": str(expected),
        "file_sha256": file_sha256(expected),
        "content_sha256": canonical_sha256(value),
        "triton_version": value["triton_version"],
        "configs": configs,
    }


def _probe_identity_matches_v29a(identity, gpu_id):
    expected = prereg.GPU_IDENTITIES_V29A[gpu_id]
    return (
        identity.get("physical_gpu_id") == gpu_id
        and identity.get("ray_gpu_id_canonical") == gpu_id
        and identity.get("cuda_visible_devices") == str(gpu_id)
        and identity.get("nvml_uuid") == expected["uuid"]
        and str(identity.get("pci_bus_id", "")).upper()
        == expected["pci_bus_id"].upper()
        and identity.get("total_bytes") == expected["total_bytes"]
    )


def _official_actor_pids_v29a(ray, workers):
    from ray.util.state import get_actor

    ray.get([worker.__ray_ready__.remote() for worker in workers.values()])
    result = {}
    for gpu_id, worker in workers.items():
        state = get_actor(worker._actor_id.hex())
        pid = None if state is None else getattr(state, "pid", None)
        status = None if state is None else getattr(state, "state", None)
        if status != "ALIVE" or isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise RuntimeError("V29A official actor PID identity changed")
        result[gpu_id] = pid
    if len(set(result.values())) != 4:
        raise RuntimeError("V29A official workers do not have four distinct processes")
    return result


def _monitor_four_tuning_futures_v29a(ray, futures, actor_pids_by_gpu):
    pending = list(futures)
    aggregate = {
        gpu_id: {
            "physical_gpu_id": gpu_id,
            "nvml_uuid": prereg.GPU_IDENTITIES_V29A[gpu_id]["uuid"],
            "sample_count": 0,
            "running_process_observed": False,
            "assigned_official_actor_pid_observed": False,
            "positive_gpu_utilization_observed": False,
            "maximum_gpu_utilization_percent": 0,
        }
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V29A
    }
    simultaneous_observation_count = 0
    while pending:
        observation = _observe_all_four_gpus_v29a()
        rows_by_gpu = {
            row["physical_gpu_id"]: row for row in observation["gpus"]
        }
        if set(rows_by_gpu) != set(prereg.PHYSICAL_GPU_IDS_V29A):
            raise RuntimeError("V29A in-flight physical-GPU surface changed")
        if all(
            actor_pids_by_gpu[gpu_id]
            in rows_by_gpu[gpu_id]["running_process_pids"]
            and rows_by_gpu[gpu_id]["gpu_utilization_percent"] > 0
            for gpu_id in prereg.PHYSICAL_GPU_IDS_V29A
        ):
            simultaneous_observation_count += 1
        for row in observation["gpus"]:
            item = aggregate[row["physical_gpu_id"]]
            item["sample_count"] += 1
            item["running_process_observed"] = (
                item["running_process_observed"]
                or row["running_process_count"] > 0
            )
            item["assigned_official_actor_pid_observed"] = (
                item["assigned_official_actor_pid_observed"]
                or actor_pids_by_gpu[row["physical_gpu_id"]]
                in row["running_process_pids"]
            )
            item["positive_gpu_utilization_observed"] = (
                item["positive_gpu_utilization_observed"]
                or row["gpu_utilization_percent"] > 0
            )
            item["maximum_gpu_utilization_percent"] = max(
                item["maximum_gpu_utilization_percent"],
                row["gpu_utilization_percent"],
            )
        ready, pending = ray.wait(pending, num_returns=1, timeout=0.25)
        if ready:
            # Surface an official worker exception immediately; do not replace
            # it with a later utilization-certificate failure.
            ray.get(ready)
    if simultaneous_observation_count < 1 or any(
        item["running_process_observed"] is not True
        or item["assigned_official_actor_pid_observed"] is not True
        or item["positive_gpu_utilization_observed"] is not True
        for item in aggregate.values()
    ):
        raise RuntimeError(
            "V29A did not observe an active process and positive utilization "
            "on every exact physical GPU during tuning"
        )
    return {
        "per_gpu": aggregate,
        "simultaneous_all_four_assigned_actor_pids_and_positive_utilization_observation_count": (
            simultaneous_observation_count
        ),
        "simultaneous_all_four_requirement_passed": True,
    }


def run_official_tuner_v29a(output_directory):
    import ray
    import torch
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    module = _load_official_tuner_v29a()
    if ray.is_initialized():
        raise RuntimeError("V29A requires a fresh Ray runtime")
    ray.init(num_gpus=4, include_dashboard=False)

    @ray.remote(num_gpus=1)
    class PlacementProbeV29A:
        def identity(self):
            import pynvml

            ids = ray.get_gpu_ids()
            if not isinstance(ids, list) or len(ids) != 1:
                raise RuntimeError("V29A probe Ray GPU allocation changed")
            physical = normalize_ray_gpu_id_v29a(ids[0])
            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(physical)
                uuid = pynvml.nvmlDeviceGetUUID(handle)
                pci = pynvml.nvmlDeviceGetPciInfo(handle).busId
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                if isinstance(uuid, bytes):
                    uuid = uuid.decode("ascii")
                if isinstance(pci, bytes):
                    pci = pci.decode("ascii")
            finally:
                pynvml.nvmlShutdown()
            return {
                "physical_gpu_id": physical,
                "ray_gpu_id_raw": ids[0],
                "ray_gpu_id_canonical": physical,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "nvml_uuid": str(uuid),
                "pci_bus_id": str(pci),
                "total_bytes": int(memory.total),
            }

    placement_groups = []
    workers = {}
    probes = []
    before_identities = {}
    after_identities = {}
    try:
        for _gpu_id in prereg.PHYSICAL_GPU_IDS_V29A:
            group = placement_group(
                [{"CPU": 1, "GPU": 1}], strategy="PACK"
            )
            ray.get(group.ready())
            placement_groups.append(group)
        strategies = [
            PlacementGroupSchedulingStrategy(
                placement_group=group,
                placement_group_capture_child_tasks=True,
            )
            for group in placement_groups
        ]
        probes = [
            PlacementProbeV29A.options(scheduling_strategy=strategy).remote()
            for strategy in strategies
        ]
        discovered = ray.get([probe.identity.remote() for probe in probes])
        strategies_by_gpu = {}
        for probe, identity, strategy in zip(probes, discovered, strategies):
            gpu_id = identity["ray_gpu_id_canonical"]
            if gpu_id in strategies_by_gpu or not _probe_identity_matches_v29a(
                identity, gpu_id
            ):
                raise RuntimeError("V29A placement probe identity changed")
            before_identities[gpu_id] = identity
            strategies_by_gpu[gpu_id] = strategy
            ray.kill(probe)
        probes = []
        if set(strategies_by_gpu) != set(prereg.PHYSICAL_GPU_IDS_V29A):
            raise RuntimeError("V29A exact four-GPU placement coverage changed")

        search_space = module.get_configs_compute_bound(False, [128, 128])
        if (
            len(search_space) != 1920
            or canonical_sha256(search_space) != prereg.SEARCH_SPACE_SHA256_V29A
        ):
            raise RuntimeError("V29A official FP8 search space changed")
        workers = {
            gpu_id: module.BenchmarkWorker.options(
                scheduling_strategy=strategies_by_gpu[gpu_id]
            ).remote(prereg.SELECTION_SEED_V29A)
            for gpu_id in prereg.PHYSICAL_GPU_IDS_V29A
        }
        actor_pids_by_gpu = _official_actor_pids_v29a(ray, workers)
        futures = [
            workers[gpu_id].tune.remote(
                prereg.BATCH_BY_GPU_V29A[gpu_id],
                256,
                1024,
                2048,
                8,
                torch.bfloat16,
                True,
                False,
                False,
                search_space,
                [128, 128],
                False,
            )
            for gpu_id in prereg.PHYSICAL_GPU_IDS_V29A
        ]
        utilization = _monitor_four_tuning_futures_v29a(
            ray, futures, actor_pids_by_gpu
        )
        selected = ray.get(futures)
        best_configs = {
            prereg.BATCH_BY_GPU_V29A[gpu_id]: module.sort_config(
                _validate_config_v29a(selected[gpu_id])
            )
            for gpu_id in prereg.PHYSICAL_GPU_IDS_V29A
        }
        module.save_configs(
            best_configs,
            256,
            1024,
            2048,
            8,
            torch.bfloat16,
            True,
            False,
            False,
            [128, 128],
            str(output_directory),
        )
        for worker in workers.values():
            ray.kill(worker)
        workers = {}
        probes = [
            PlacementProbeV29A.options(
                scheduling_strategy=strategies_by_gpu[gpu_id]
            ).remote()
            for gpu_id in prereg.PHYSICAL_GPU_IDS_V29A
        ]
        rediscovered = ray.get([probe.identity.remote() for probe in probes])
        for gpu_id, probe, identity in zip(
            prereg.PHYSICAL_GPU_IDS_V29A, probes, rediscovered
        ):
            if not _probe_identity_matches_v29a(identity, gpu_id):
                raise RuntimeError("V29A post-tuning placement identity changed")
            after_identities[gpu_id] = identity
            ray.kill(probe)
        probes = []
        return {
            "selected_configs": best_configs,
            "placement_identity_commitment_before_sha256": canonical_sha256(
                before_identities
            ),
            "placement_identity_commitment_after_sha256": canonical_sha256(
                after_identities
            ),
            "all_four_official_tune_futures_submitted_before_ray_get": True,
            "official_worker_count": 4,
            "configuration_count_per_worker": 1920,
            "inflight_physical_gpu_utilization": utilization,
            "all_four_distinct_uuid_workers_observed_active_and_positive": True,
            "official_actor_pid_map_commitment_sha256": canonical_sha256(
                actor_pids_by_gpu
            ),
            "prelaunch_idle_makes_observed_inflight_processes_tuner_owned": True,
        }
    finally:
        for worker in workers.values():
            try:
                ray.kill(worker)
            except BaseException:
                pass
        for probe in probes:
            try:
                ray.kill(probe)
            except BaseException:
                pass
        for group in placement_groups:
            try:
                remove_placement_group(group)
            except BaseException:
                pass
        ray.shutdown()


def validate_launch_arguments_v29a(args, implementation, recipe):
    if any(os.environ.get(key) for key in MOE_ENVIRONMENT_V29A):
        raise ValueError("V29A rejects external MoE tuning and config overrides")
    if args.v29a_dry_run and args.launch_v29a:
        raise ValueError("V29A dry-run and launch flags are mutually exclusive")
    if not args.v29a_dry_run and not args.launch_v29a:
        raise ValueError("V29A requires --v29a-dry-run or explicit --launch-v29a")
    if args.launch_v29a and os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3":
        raise ValueError("V29A real launch requires CUDA_VISIBLE_DEVICES=0,1,2,3")
    expected = (
        (args.expected_implementation_bundle_sha256, implementation["bundle_sha256"], "implementation"),
        (args.expected_recipe_sha256, recipe["content_sha256_before_self_field"], "recipe"),
    )
    for supplied, actual, label in expected:
        if args.launch_v29a and supplied is None:
            raise ValueError(f"V29A real launch requires expected {label} hash")
        if supplied is not None and supplied != actual:
            raise ValueError(f"V29A expected {label} hash changed")


def run_exact_v29a(preregistration, implementation, recipe):
    if (
        prereg.ATTEMPT_PATH_V29A.exists()
        or prereg.REPORT_PATH_V29A.parent.exists()
        or prereg.OUTPUT_DIRECTORY_V29A.exists()
    ):
        raise RuntimeError("V29A requires fresh exclusive output paths")
    environment = runtime_r2.certify_runtime_environment_r2()
    live_audit = live_cpu_disk_audit_v29a()
    prelaunch_idle = assert_all_four_gpus_idle_v29a()
    attempt = _seal({
        "schema": "vllm-moe-fp8-tuning-attempt-v29a",
        "status": "running",
        "phase": "before_four_concurrent_official_workers",
        "preregistration": recipe["preregistration"],
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "recipe_content_sha256": recipe["content_sha256_before_self_field"],
        "runtime_environment_certificate_sha256": environment[
            "content_sha256_before_self_field"
        ],
        "live_cpu_disk_audit": live_audit,
        "prelaunch_idle_certificate_sha256": prelaunch_idle[
            "content_sha256_before_self_field"
        ],
        "selected_table_written": False,
        "training_model_update_checkpoint_adoption_evaluation_or_dataset_promotion_applied": False,
        "dataset_train_evaluation_validation_heldout_ood_or_benchmark_surface_opened": False,
    })
    _assert_compact_v29a(attempt)
    _exclusive_write_json_v29a(prereg.ATTEMPT_PATH_V29A, attempt)
    final_idle = None
    try:
        prereg.OUTPUT_DIRECTORY_V29A.mkdir(parents=True, exist_ok=False)
        run_result = run_official_tuner_v29a(prereg.OUTPUT_DIRECTORY_V29A)
        final_idle = wait_for_final_gpu_idle_v29a()

        selected_table = validate_selected_table_v29a()
        if run_result["selected_configs"] != {
            int(key): value for key, value in selected_table["configs"].items()
        }:
            raise RuntimeError("V29A in-memory and persisted selected configs disagree")
        report = _seal({
            "schema": "vllm-moe-fp8-tuning-selection-report-v29a",
            "status": "complete_selection_not_evaluation",
            "preregistration": recipe["preregistration"],
            "implementation_bundle_sha256": implementation["bundle_sha256"],
            "recipe_content_sha256": recipe["content_sha256_before_self_field"],
            "runtime_environment_certificate_sha256": environment[
                "content_sha256_before_self_field"
            ],
            "live_cpu_disk_audit_content_sha256": live_audit[
                "content_sha256_before_self_field"
            ],
            "prelaunch_idle_certificate_sha256": prelaunch_idle[
                "content_sha256_before_self_field"
            ],
            "final_idle_certificate_sha256": final_idle[
                "content_sha256_before_self_field"
            ],
            "runtime_integrity": {
                **run_result,
                "all_four_gpus_idle_after_cleanup": final_idle["all_four_idle"],
                "official_source_unmodified": True,
                "only_official_OutOfResources_skip_active": True,
                "bf16_v27c_table_loaded_or_reused": False,
            },
            "selected_table": selected_table,
            "decision": "authorize_only_separate_fp8_table_evaluation_preregistration",
            "direct_adoption_training_model_update_checkpoint_evaluation_or_dataset_promotion_authorized": False,
            "direct_action_taken": False,
            "dataset_train_evaluation_validation_heldout_ood_or_benchmark_surface_opened": False,
            "raw_progress_timing_vectors_compiler_logs_or_search_results_persisted": False,
        })
        _assert_compact_v29a(report)
        _exclusive_write_json_v29a(prereg.REPORT_PATH_V29A, report)
        completed_attempt = _seal({
            **_without_self(attempt),
            "status": "complete",
            "phase": "after_compact_report_and_final_gpu_cleanup",
            "selected_table_written": True,
            "selected_table_file_sha256": selected_table["file_sha256"],
            "selected_table_content_sha256": selected_table["content_sha256"],
            "report_binding": {
                "path": str(prereg.REPORT_PATH_V29A),
                "file_sha256": file_sha256(prereg.REPORT_PATH_V29A),
                "content_sha256": report["content_sha256_before_self_field"],
            },
            "final_idle_certificate_sha256": final_idle[
                "content_sha256_before_self_field"
            ],
        })
        _assert_compact_v29a(completed_attempt)
        _rewrite_json_v29a(prereg.ATTEMPT_PATH_V29A, completed_attempt)
        return report
    except BaseException as failure:
        if final_idle is None:
            try:
                final_idle = wait_for_final_gpu_idle_v29a()
            except BaseException:
                final_idle = None
        failed_attempt = _seal({
            **_without_self(attempt),
            "status": "failed",
            "phase": "after_worker_cleanup_or_finalization_failure",
            "failure": {
                "exception_class": type(failure).__name__,
                "message_sha256": hashlib.sha256(
                    str(failure).encode("utf-8")
                ).hexdigest(),
                "raw_message_or_traceback_persisted": False,
            },
            "final_idle_certificate_sha256": (
                None if final_idle is None
                else final_idle["content_sha256_before_self_field"]
            ),
        })
        _assert_compact_v29a(failed_attempt)
        _rewrite_json_v29a(prereg.ATTEMPT_PATH_V29A, failed_attempt)
        raise failure


def _parser_v29a():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v29a-dry-run", action="store_true")
    parser.add_argument("--launch-v29a", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-recipe-sha256")
    return parser


def main(argv=None):
    args = _parser_v29a().parse_args(argv)
    preregistration = load_preregistration_v29a()
    implementation = implementation_identity_v29a()
    recipe = recipe_v29a(preregistration, implementation)
    validate_launch_arguments_v29a(args, implementation, recipe)
    if args.v29a_dry_run:
        value = _seal({
            "schema": "vllm-moe-fp8-tuning-dry-run-v29a",
            "preregistration_file_sha256": PREREG_FILE_SHA256_V29A,
            "preregistration_content_sha256": PREREG_CONTENT_SHA256_V29A,
            "implementation_bundle_sha256": implementation["bundle_sha256"],
            "recipe_content_sha256": recipe["content_sha256_before_self_field"],
            "expected_output_filename": prereg.EXPECTED_OUTPUT_FILENAME_V29A,
            "gpu_launched": False,
            "tuner_launched": False,
            "dataset_or_evaluation_surface_opened": False,
        })
        print(json.dumps(value, indent=2, sort_keys=True))
        return value
    return run_exact_v29a(preregistration, implementation, recipe)


if __name__ == "__main__":
    main(sys.argv[1:])
