#!/usr/bin/env python3
"""Fail-closed V29E FP8 selected-table synthetic-kernel evaluator."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import gc

import build_vllm_moe_fp8_selected_table_evaluation_retry_preregistration_v29e as prereg
import build_vllm_moe_fp8_selected_table_evaluation_preregistration_v29d as prereg_v29d
import build_vllm_moe_fp8_selected_table_v29d_failure_evidence_v29e as failure_v29e
import build_vllm_moe_fp8_tuning_preregistration_v29b as v29b
import run_eggroll_es_insertion_stability_v23a_retry_r2 as runtime_r2
import run_vllm_moe_fp8_tuning_v29b as runtime_v29b


ROOT = Path(__file__).resolve().parent
PREREG_PATH_V29E = prereg.OUTPUT_PATH_V29E
PREREG_FILE_SHA256_V29E = (
    "853a38e75bfe91baa21d0d4331dcfbd298f7828da529920dc3c244c81f908a1f"
)
PREREG_CONTENT_SHA256_V29E = (
    "5a8bb93c60631f5a1acb22d729c942a6f2630f8ad72b0698bc7c32ee5c3f089f"
)
PREREG_TEST_PATH_V29E = ROOT / (
    "test_build_vllm_moe_fp8_selected_table_evaluation_retry_preregistration_v29e.py"
)
RUNTIME_TEST_PATH_V29E = ROOT / (
    "test_run_vllm_moe_fp8_selected_table_evaluation_retry_v29e.py"
)
MOE_ENVIRONMENT_V29E = (
    "VLLM_TUNED_CONFIG_FOLDER",
    "VLLM_BATCH_INVARIANT",
    "VLLM_MOE_TUNE_CACHE_CLEAR_INTERVAL",
    "VLLM_ROCM_USE_AITER",
    "VLLM_ROCM_USE_AITER_MOE",
    "VLLM_USE_DEEP_GEMM",
    "VLLM_MOE_USE_DEEP_GEMM",
)
FORBIDDEN_COMPACT_KEYS_V29E = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "response", "responses", "text", "texts", "token_ids", "raw_pids",
    "timing_vectors", "input_tensor", "output_tensor", "training_rows",
    "evaluation_rows", "validation_rows", "heldout_rows", "ood_rows",
    "benchmark_rows", "traceback", "compiler_log", "progress_log",
}
IMPLEMENTATION_PATHS_V29E = {
    "v29e_preregistration_builder": Path(prereg.__file__).resolve(),
    "v29e_preregistration_tests": PREREG_TEST_PATH_V29E,
    "v29e_preregistration": PREREG_PATH_V29E,
    "v29e_runtime": Path(__file__).resolve(),
    "v29e_runtime_tests": RUNTIME_TEST_PATH_V29E,
    "v29d_preregistration_builder": Path(prereg_v29d.__file__).resolve(),
    "v29d_preregistration": prereg_v29d.OUTPUT_PATH_V29D,
    "v29d_runtime": ROOT / "run_vllm_moe_fp8_selected_table_evaluation_v29d.py",
    "v29d_failure_evidence_builder": Path(failure_v29e.__file__).resolve(),
    "v29d_failure_evidence_tests": ROOT / (
        "test_build_vllm_moe_fp8_selected_table_v29d_failure_evidence_v29e.py"
    ),
    "v29d_failure_evidence": prereg.FAILURE_EVIDENCE_PATH_V29E,
    "v29c_selection_evidence": prereg.EVIDENCE_PATH_V29E,
    "v29b_original_selected_table": prereg.TABLE_PATH_V29E,
    "official_vllm_025_tuner": v29b.OFFICIAL_TUNER_PATH_V29B,
    "installed_vllm_fused_moe": v29b.FUSED_MOE_PATH_V29B,
    "installed_vllm_fused_moe_config": v29b.FUSED_MOE_CONFIG_PATH_V29B,
    "installed_vllm_envs": v29b.VLLM_ENVS_PATH_V29B,
    "pinned_ray_actor_builtin": v29b.RAY_ACTOR_PATH_V29B,
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
    value.pop("content_sha256_before_self_field", None)
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


def _assert_compact_v29e(value):
    overlap = FORBIDDEN_COMPACT_KEYS_V29E & set(_recursive_keys(value))
    if overlap:
        raise RuntimeError(f"V29E compact output contains forbidden keys: {sorted(overlap)}")


def _exclusive_write_json_v29e(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def _rewrite_json_v29e(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_preregistration_v29e():
    value = json.loads(PREREG_PATH_V29E.read_text(encoding="utf-8"))
    if (
        file_sha256(PREREG_PATH_V29E) != PREREG_FILE_SHA256_V29E
        or value.get("content_sha256_before_self_field")
        != PREREG_CONTENT_SHA256_V29E
        or canonical_sha256(_without_self(value)) != PREREG_CONTENT_SHA256_V29E
    ):
        raise RuntimeError("V29E preregistration identity changed")
    return prereg.validate_preregistration_v29e(value)


def implementation_identity_v29e():
    files = {
        key: {
            "path": str(Path(path).resolve()),
            "file_sha256": file_sha256(path),
        }
        for key, path in IMPLEMENTATION_PATHS_V29E.items()
    }
    return {"files": files, "bundle_sha256": canonical_sha256(files)}


def recipe_v29e(preregistration, implementation):
    return _seal({
        "schema": "vllm-moe-fp8-selected-table-evaluation-recipe-v29e",
        "experiment_name": prereg.EXPERIMENT_NAME_V29E,
        "preregistration": {
            "path": str(PREREG_PATH_V29E),
            "file_sha256": PREREG_FILE_SHA256_V29E,
            "content_sha256": PREREG_CONTENT_SHA256_V29E,
        },
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "selection_evidence": copy.deepcopy(
            preregistration["selection_evidence"]
        ),
        "retry_of": copy.deepcopy(preregistration["retry_of"]),
        "sole_infrastructure_correction": copy.deepcopy(
            preregistration["sole_infrastructure_correction"]
        ),
        "selected_table": copy.deepcopy(preregistration["selected_table"]),
        "software_identity": copy.deepcopy(preregistration["software_identity"]),
        "model_identity": copy.deepcopy(preregistration["model_identity"]),
        "kernel_contract": copy.deepcopy(preregistration["kernel_contract"]),
        "schedule": copy.deepcopy(preregistration["schedule"]),
        "statistical_contract": copy.deepcopy(
            preregistration["statistical_contract"]
        ),
        "hardware_contract": copy.deepcopy(preregistration["hardware_contract"]),
        "persistence_contract": copy.deepcopy(
            preregistration["persistence_contract"]
        ),
        "authority": copy.deepcopy(preregistration["authority"]),
        "strict_synthetic_kernel_only": True,
    })


def validate_preregistration_commit_v29e(commit):
    if not isinstance(commit, str) or len(commit) != 40 or any(
        char not in "0123456789abcdef" for char in commit
    ):
        raise ValueError("V29E real launch requires exact lowercase preregistration commit")
    relative = PREREG_PATH_V29E.relative_to(ROOT).as_posix()
    raw = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
    )
    if hashlib.sha256(raw).hexdigest() != PREREG_FILE_SHA256_V29E:
        raise RuntimeError("V29E committed preregistration identity changed")
    return commit


def normalize_ray_gpu_id_v29e(value):
    if isinstance(value, bool):
        raise RuntimeError("V29E Ray GPU ID representation changed")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value in {"0", "1", "2", "3"}:
        result = int(value)
    else:
        raise RuntimeError("V29E Ray GPU ID representation changed")
    if result not in prereg.PHYSICAL_GPU_IDS_V29E:
        raise RuntimeError("V29E Ray GPU ID physical range changed")
    return result


def live_cpu_disk_audit_v29e():
    prereg.validate_static_inputs_v29e()
    base = runtime_v29b.live_cpu_disk_audit_v29b()
    return _seal({
        "schema": "vllm-moe-fp8-selected-table-live-cpu-disk-audit-v29e",
        "v29b_full_model_and_runtime_audit_content_sha256": (
            base["content_sha256_before_self_field"]
        ),
        "model_all_56_files_and_42_weight_shards_rehashed": True,
        "v29d_failure_evidence_commit": prereg.FAILURE_EVIDENCE_COMMIT_V29E,
        "v29d_failure_evidence_file_sha256": (
            prereg.FAILURE_EVIDENCE_FILE_SHA256_V29E
        ),
        "v29d_failure_evidence_content_sha256": (
            prereg.FAILURE_EVIDENCE_CONTENT_SHA256_V29E
        ),
        "v29d_preregistration_commit": failure_v29e.PREREG_COMMIT_V29E,
        "v29d_preregistration_file_sha256": failure_v29e.PREREG_FILE_SHA256_V29E,
        "v29d_preregistration_content_sha256": (
            failure_v29e.PREREG_CONTENT_SHA256_V29E
        ),
        "selection_evidence_commit": prereg.EVIDENCE_COMMIT_V29E,
        "selection_evidence_file_sha256": prereg.EVIDENCE_FILE_SHA256_V29E,
        "selection_evidence_content_sha256": prereg.EVIDENCE_CONTENT_SHA256_V29E,
        "selected_table_file_sha256": prereg.TABLE_FILE_SHA256_V29E,
        "selected_table_content_sha256": prereg.TABLE_CONTENT_SHA256_V29E,
        "dataset_train_evaluation_validation_heldout_ood_or_benchmark_surface_opened": False,
    })


def _observe_all_four_gpus_v29e():
    observation = runtime_v29b._observe_all_four_gpus_v29b()
    return {
        "gpus": [
            {**row, "v29e_identity_checked": True}
            for row in observation["gpus"]
        ],
        "all_four_idle": observation["all_four_idle"],
    }


def assert_all_four_gpus_idle_v29e():
    observation = _observe_all_four_gpus_v29e()
    if observation.get("all_four_idle") is not True:
        raise RuntimeError("V29E requires all four GPUs idle before claim or arm")
    return _seal({
        "schema": "vllm-moe-fp8-selected-table-idle-certificate-v29e",
        **observation,
    })


def wait_for_all_four_gpus_idle_v29e(*, timeout_seconds=30.0, interval_seconds=0.5):
    if timeout_seconds != 30.0 or interval_seconds != 0.5:
        raise RuntimeError("V29E GPU cleanup polling contract changed")
    start = time.monotonic()
    polls = 0
    while True:
        observation = _observe_all_four_gpus_v29e()
        polls += 1
        elapsed = time.monotonic() - start
        if observation.get("all_four_idle") is True:
            return _seal({
                "schema": "vllm-moe-fp8-selected-table-idle-certificate-v29e",
                **observation,
                "poll_count": polls,
                "elapsed_milliseconds": int(round(elapsed * 1000)),
                "bounded_async_cleanup_wait": True,
            })
        if elapsed >= timeout_seconds:
            raise RuntimeError("V29E GPU cleanup exceeded 30 seconds")
        time.sleep(min(interval_seconds, timeout_seconds - elapsed))


def _load_official_module_v29e():
    if file_sha256(v29b.OFFICIAL_TUNER_PATH_V29B) != v29b.OFFICIAL_TUNER_SHA256_V29B:
        raise RuntimeError("V29E official vLLM tuner identity changed")
    spec = importlib.util.spec_from_file_location(
        "vllm_benchmark_moe_fp8_v29e", v29b.OFFICIAL_TUNER_PATH_V29B,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.triton.__version__ != "3.6.0":
        raise RuntimeError("V29E Triton runtime version changed")
    return module


def _validate_config_v29e(config):
    allowed = {
        "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
        "num_warps", "num_stages",
    }
    if (
        not isinstance(config, dict) or set(config) != allowed
        or any(isinstance(item, bool) or not isinstance(item, int) for item in config.values())
    ):
        raise RuntimeError("V29E kernel config geometry changed")
    return config


def _tensor_byte_sha256_v29e(tensor):
    contiguous = tensor.detach().contiguous().view(-1).view(dtype=__import__("torch").uint8)
    raw = contiguous.cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _run_activity_witness_v29e(common_target_unix_ns):
    import torch

    if (
        isinstance(common_target_unix_ns, bool)
        or not isinstance(common_target_unix_ns, int)
        or common_target_unix_ns <= 0
    ):
        raise RuntimeError("V29E common activity witness target changed")
    shape = tuple(prereg.ACTIVITY_WITNESS_RECIPE_V29E["tensor_shape"])
    left = right = output = None
    iterations = 0
    started_unix_ns = None
    elapsed = None
    try:
        left = torch.ones(shape, dtype=torch.bfloat16, device="cuda")
        right = torch.ones(shape, dtype=torch.bfloat16, device="cuda")
        output = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
        torch.mm(left, right, out=output)
        torch.cuda.synchronize()
        while time.time_ns() < common_target_unix_ns:
            remaining = (common_target_unix_ns - time.time_ns()) / 1e9
            time.sleep(min(0.005, max(0.0, remaining)))
        started_unix_ns = time.time_ns()
        started = time.monotonic()
        while time.monotonic() - started < prereg.ACTIVITY_WITNESS_DURATION_SECONDS_V29E:
            torch.mm(left, right, out=output)
            # Bound the witness by completed GPU work rather than by an
            # arbitrarily deep queue of asynchronously submitted kernels.
            torch.cuda.synchronize()
            iterations += 1
        elapsed = time.monotonic() - started
        if iterations < 1 or elapsed < prereg.ACTIVITY_WITNESS_DURATION_SECONDS_V29E:
            raise RuntimeError("V29E CUDA activity witness duration changed")
    finally:
        del left, right, output
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return {
        "recipe_sha256": prereg.ACTIVITY_WITNESS_RECIPE_SHA256_V29E,
        "common_target_unix_ns": common_target_unix_ns,
        "started_unix_ns": started_unix_ns,
        "iteration_count": iterations,
        "elapsed_at_least_preregistered_duration": (
            elapsed is not None
            and elapsed >= prereg.ACTIVITY_WITNESS_DURATION_SECONDS_V29E
        ),
        "witness_tensors_deleted_cache_emptied_and_synchronized": True,
        "peak_memory_reset_occurs_after_return": True,
    }


def _worker_evaluate_v29e(
    physical_gpu, batch_size, seed, arm, tuned_config, empty_folder,
    common_activity_target_unix_ns,
):
    import pynvml
    import ray
    import torch

    ray_ids = ray.get_gpu_ids()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if (
        not isinstance(ray_ids, list) or len(ray_ids) != 1
        or normalize_ray_gpu_id_v29e(ray_ids[0]) != physical_gpu
        or visible != str(physical_gpu)
    ):
        raise RuntimeError("V29E exact Ray physical-GPU assignment changed")
    if arm not in {"default", "tuned"}:
        raise RuntimeError("V29E arm changed")
    if not Path(empty_folder).is_dir() or any(Path(empty_folder).iterdir()):
        raise RuntimeError("V29E default config directory is not exactly empty")
    torch.set_default_device("cuda")
    activity_witness = _run_activity_witness_v29e(
        common_activity_target_unix_ns
    )
    os.environ["VLLM_TUNED_CONFIG_FOLDER"] = str(empty_folder)
    module = _load_official_module_v29e()
    dtype = torch.bfloat16
    dtype_string = module._get_config_dtype_str(
        dtype, use_int8_w8a16=False, use_fp8_w8a8=True,
        use_int4_w4a16=False,
    )
    if module.get_moe_configs(256, 512, dtype_string, 128, 128) is not None:
        raise RuntimeError("V29E empty-folder official default lookup was not empty")
    if arm == "default":
        selected = module.get_default_config(
            batch_size, 256, 1024, 2048, 8, dtype_string, [128, 128],
        )
    else:
        selected = _validate_config_v29e(copy.deepcopy(tuned_config))
    captured = {}
    original_fused_experts = module.fused_experts

    def capture_fused_experts(*args, **kwargs):
        output = original_fused_experts(*args, **kwargs)
        captured["output"] = output
        return output

    module.fused_experts = capture_fused_experts
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        module.set_random_seed(seed)
        microseconds = module.benchmark_config(
            selected, batch_size, 256, 1024, 2048, 8, dtype,
            True, False, False, num_iters=prereg.OFFICIAL_NUM_ITERS_V29E,
            block_quant_shape=[128, 128], use_deep_gemm=False,
        )
        torch.cuda.synchronize()
        if "output" not in captured:
            raise RuntimeError("V29E official kernel output capture failed")
        output = captured["output"]
        output_sha256 = _tensor_byte_sha256_v29e(output)
        output_shape = list(output.shape)
        output_dtype = str(output.dtype)
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
    finally:
        module.fused_experts = original_fused_experts
    if not math.isfinite(microseconds) or microseconds <= 0:
        raise RuntimeError("V29E nonfinite kernel time")
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_gpu)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        pci = pynvml.nvmlDeviceGetPciInfo(handle).busId
        if isinstance(uuid, bytes):
            uuid = uuid.decode("ascii")
        if isinstance(pci, bytes):
            pci = pci.decode("ascii")
    finally:
        pynvml.nvmlShutdown()
    expected = v29b.GPU_IDENTITIES_V29B[physical_gpu]
    if (
        str(uuid) != expected["uuid"]
        or str(pci).upper() != expected["pci_bus_id"].upper()
        or torch.cuda.current_device() != 0
    ):
        raise RuntimeError("V29E worker exact UUID/PCI/runtime device changed")
    input_recipe = {
        "official_tuner_sha256": v29b.OFFICIAL_TUNER_SHA256_V29B,
        "seed": seed,
        "batch_size": batch_size,
        "experts": 256,
        "shard_intermediate_size": 1024,
        "hidden_size": 2048,
        "topk": 8,
        "dtype": "torch.bfloat16",
        "fp8_w8a8": True,
        "block_shape": [128, 128],
        "num_iters": prereg.OFFICIAL_NUM_ITERS_V29E,
    }
    return {
        "physical_gpu_id": physical_gpu,
        "nvml_uuid": str(uuid),
        "batch_size": batch_size,
        "seed": seed,
        "arm": arm,
        "selected_config": selected,
        "input_recipe_sha256": canonical_sha256(input_recipe),
        "output_sha256": output_sha256,
        "output_shape": output_shape,
        "output_dtype": output_dtype,
        "kernel_time_microseconds": float(microseconds),
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
        "activity_witness": activity_witness,
    }


def _probe_identity_matches_v29e(identity, gpu_id):
    expected = v29b.GPU_IDENTITIES_V29B[gpu_id]
    return (
        identity.get("physical_gpu_id") == gpu_id
        and identity.get("ray_gpu_id_canonical") == gpu_id
        and identity.get("cuda_visible_devices") == str(gpu_id)
        and identity.get("nvml_uuid") == expected["uuid"]
        and str(identity.get("pci_bus_id", "")).upper()
        == expected["pci_bus_id"].upper()
        and identity.get("total_bytes") == expected["total_bytes"]
    )


def _official_actor_pids_v29e(ray, workers):
    ray.get([worker.__ray_ready__.remote() for worker in workers.values()])
    values = ray.get([
        workers[gpu].__ray_call__.remote(lambda self: os.getpid())
        for gpu in prereg.PHYSICAL_GPU_IDS_V29E
    ])
    result = dict(zip(prereg.PHYSICAL_GPU_IDS_V29E, values))
    if (
        any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in values)
        or len(set(values)) != 4
    ):
        raise RuntimeError("V29E requires four distinct official worker PIDs")
    return result


def _monitor_four_futures_v29e(ray, futures, actor_pids_by_gpu):
    pending = list(futures)
    aggregate = {
        gpu: {
            "physical_gpu_id": gpu,
            "nvml_uuid": v29b.GPU_IDENTITIES_V29B[gpu]["uuid"],
            "sample_count": 0,
            "assigned_pid_observed": False,
            "positive_utilization_observed": False,
            "maximum_gpu_utilization_percent": 0,
        }
        for gpu in prereg.PHYSICAL_GPU_IDS_V29E
    }
    simultaneous = 0
    while pending:
        observation = _observe_all_four_gpus_v29e()
        rows = {row["physical_gpu_id"]: row for row in observation["gpus"]}
        if set(rows) != set(prereg.PHYSICAL_GPU_IDS_V29E):
            raise RuntimeError("V29E in-flight physical GPU surface changed")
        if all(
            actor_pids_by_gpu[gpu] in rows[gpu]["running_process_pids"]
            and rows[gpu]["gpu_utilization_percent"] > 0
            for gpu in prereg.PHYSICAL_GPU_IDS_V29E
        ):
            simultaneous += 1
        for gpu, row in rows.items():
            item = aggregate[gpu]
            item["sample_count"] += 1
            item["assigned_pid_observed"] |= (
                actor_pids_by_gpu[gpu] in row["running_process_pids"]
            )
            item["positive_utilization_observed"] |= (
                row["gpu_utilization_percent"] > 0
            )
            item["maximum_gpu_utilization_percent"] = max(
                item["maximum_gpu_utilization_percent"],
                row["gpu_utilization_percent"],
            )
        ready, pending = ray.wait(
            pending, num_returns=1,
            timeout=prereg.NVML_POLL_INTERVAL_SECONDS_V29E,
        )
        if ready:
            ray.get(ready)
    if simultaneous < 1 or any(
        item["assigned_pid_observed"] is not True
        or item["positive_utilization_observed"] is not True
        for item in aggregate.values()
    ):
        raise RuntimeError("V29E did not observe all four assigned PIDs with utilization")
    return {
        "per_gpu": aggregate,
        "simultaneous_all_four_assigned_pids_and_positive_utilization_count": simultaneous,
        "pass": True,
    }


def run_arm_v29e(arm, seed, tuned_configs, empty_folder):
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    if ray.is_initialized():
        raise RuntimeError("V29E requires a fresh Ray runtime per arm")
    ray.init(num_gpus=4, include_dashboard=False)

    @ray.remote(num_gpus=1)
    class PlacementProbeV29E:
        def identity(self):
            import pynvml

            ids = ray.get_gpu_ids()
            if not isinstance(ids, list) or len(ids) != 1:
                raise RuntimeError("V29E placement probe allocation changed")
            physical = normalize_ray_gpu_id_v29e(ids[0])
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
                "ray_gpu_id_canonical": physical,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "nvml_uuid": str(uuid),
                "pci_bus_id": str(pci),
                "total_bytes": int(memory.total),
            }

    @ray.remote(num_gpus=1)
    class EvaluationWorkerV29E:
        def evaluate(
            self, physical_gpu, batch_size, worker_seed, worker_arm, config,
            folder, common_activity_target_unix_ns,
        ):
            return _worker_evaluate_v29e(
                physical_gpu, batch_size, worker_seed, worker_arm, config,
                folder, common_activity_target_unix_ns,
            )

    groups = []
    probes = []
    workers = {}
    try:
        for _ in prereg.PHYSICAL_GPU_IDS_V29E:
            group = placement_group([{"CPU": 1, "GPU": 1}], strategy="PACK")
            ray.get(group.ready())
            groups.append(group)
        strategies = [
            PlacementGroupSchedulingStrategy(
                placement_group=group,
                placement_group_capture_child_tasks=True,
            )
            for group in groups
        ]
        probes = [
            PlacementProbeV29E.options(scheduling_strategy=strategy).remote()
            for strategy in strategies
        ]
        identities = ray.get([probe.identity.remote() for probe in probes])
        strategies_by_gpu = {}
        identity_by_gpu = {}
        for probe, identity, strategy in zip(probes, identities, strategies):
            gpu = identity["physical_gpu_id"]
            if gpu in strategies_by_gpu or not _probe_identity_matches_v29e(identity, gpu):
                raise RuntimeError("V29E placement probe identity changed")
            identity_by_gpu[gpu] = identity
            strategies_by_gpu[gpu] = strategy
            ray.kill(probe)
        probes = []
        if set(strategies_by_gpu) != set(prereg.PHYSICAL_GPU_IDS_V29E):
            raise RuntimeError("V29E exact four-GPU placement coverage changed")
        workers = {
            gpu: EvaluationWorkerV29E.options(
                scheduling_strategy=strategies_by_gpu[gpu]
            ).remote()
            for gpu in prereg.PHYSICAL_GPU_IDS_V29E
        }
        actor_pids = _official_actor_pids_v29e(ray, workers)
        common_activity_target_unix_ns = time.time_ns() + int(
            prereg.ACTIVITY_WITNESS_TARGET_OFFSET_SECONDS_V29E * 1e9
        )
        futures = [
            workers[gpu].evaluate.remote(
                gpu, prereg.BATCH_BY_GPU_V29E[gpu], seed, arm,
                tuned_configs[str(prereg.BATCH_BY_GPU_V29E[gpu])],
                str(empty_folder),
                common_activity_target_unix_ns,
            )
            for gpu in prereg.PHYSICAL_GPU_IDS_V29E
        ]
        utilization = _monitor_four_futures_v29e(ray, futures, actor_pids)
        results = ray.get(futures)
        if any(
            item.get("physical_gpu_id") != gpu
            or item.get("batch_size") != prereg.BATCH_BY_GPU_V29E[gpu]
            or item.get("seed") != seed
            or item.get("arm") != arm
            or item.get("nvml_uuid") != v29b.GPU_IDENTITIES_V29B[gpu]["uuid"]
            or item.get("activity_witness", {}).get("recipe_sha256")
            != prereg.ACTIVITY_WITNESS_RECIPE_SHA256_V29E
            or item.get("activity_witness", {}).get("common_target_unix_ns")
            != common_activity_target_unix_ns
            or item.get("activity_witness", {}).get(
                "elapsed_at_least_preregistered_duration"
            ) is not True
            or item.get("activity_witness", {}).get(
                "witness_tensors_deleted_cache_emptied_and_synchronized"
            ) is not True
            or item.get("activity_witness", {}).get(
                "peak_memory_reset_occurs_after_return"
            ) is not True
            for gpu, item in enumerate(results)
        ):
            raise RuntimeError("V29E worker result identity changed")
        return {
            "results": results,
            "placement_identity_commitment_sha256": canonical_sha256(identity_by_gpu),
            "actor_pid_map_commitment_sha256": canonical_sha256(actor_pids),
            "utilization": utilization,
            "all_four_futures_submitted_before_wait": True,
            "common_activity_target_commitment_sha256": canonical_sha256(
                common_activity_target_unix_ns
            ),
            "activity_witness_result_commitment_sha256": canonical_sha256([
                item["activity_witness"] for item in results
            ]),
            "activity_witness_excluded_and_peak_memory_reset_before_measurement": True,
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
        for group in groups:
            try:
                remove_placement_group(group)
            except BaseException:
                pass
        ray.shutdown()


def _quantile_v29e(values, probability, *, upper):
    ordered = sorted(values)
    if upper:
        index = max(0, math.ceil(probability * len(ordered)) - 1)
    else:
        index = min(len(ordered) - 1, math.floor(probability * len(ordered)))
    return ordered[index]


def summarize_v29e(records, *, bootstrap_resamples=None):
    if bootstrap_resamples is None:
        bootstrap_resamples = prereg.BOOTSTRAP_RESAMPLES_V29E
    if len(records) != prereg.REPETITIONS_V29E:
        raise RuntimeError("V29E paired repetition count changed")
    latency = {gpu: [] for gpu in prereg.PHYSICAL_GPU_IDS_V29E}
    vram = {gpu: [] for gpu in prereg.PHYSICAL_GPU_IDS_V29E}
    output_matches = []
    output_commitments = []
    for expected_repetition, record in enumerate(records):
        if record.get("repetition") != expected_repetition:
            raise RuntimeError("V29E repetition order changed")
        arms = record.get("arms", {})
        if set(arms) != {"default", "tuned"}:
            raise RuntimeError("V29E paired arms changed")
        default_by_gpu = {item["physical_gpu_id"]: item for item in arms["default"]}
        tuned_by_gpu = {item["physical_gpu_id"]: item for item in arms["tuned"]}
        if set(default_by_gpu) != set(prereg.PHYSICAL_GPU_IDS_V29E) or set(tuned_by_gpu) != set(prereg.PHYSICAL_GPU_IDS_V29E):
            raise RuntimeError("V29E paired GPU coverage changed")
        for gpu in prereg.PHYSICAL_GPU_IDS_V29E:
            default = default_by_gpu[gpu]
            tuned = tuned_by_gpu[gpu]
            if (
                default["seed"] != tuned["seed"]
                or default["input_recipe_sha256"] != tuned["input_recipe_sha256"]
                or default["batch_size"] != tuned["batch_size"]
            ):
                raise RuntimeError("V29E paired deterministic input identity changed")
            latency[gpu].append(
                default["kernel_time_microseconds"] / tuned["kernel_time_microseconds"]
            )
            allocated_ratio = (
                tuned["peak_memory_allocated_bytes"]
                / default["peak_memory_allocated_bytes"]
            )
            reserved_ratio = (
                tuned["peak_memory_reserved_bytes"]
                / default["peak_memory_reserved_bytes"]
            )
            vram[gpu].append(max(allocated_ratio, reserved_ratio))
            exact = (
                default["output_sha256"] == tuned["output_sha256"]
                and default["output_shape"] == tuned["output_shape"]
                and default["output_dtype"] == tuned["output_dtype"]
            )
            output_matches.append(exact)
            output_commitments.append({
                "repetition": expected_repetition,
                "gpu": gpu,
                "default_sha256": default["output_sha256"],
                "tuned_sha256": tuned["output_sha256"],
                "exact": exact,
            })
    point_latency = {gpu: statistics.median(values) for gpu, values in latency.items()}
    point_vram = {gpu: statistics.median(values) for gpu, values in vram.items()}
    point_global_latency = math.exp(statistics.fmean(
        math.log(point_latency[gpu]) for gpu in prereg.PHYSICAL_GPU_IDS_V29E
    ))
    point_global_vram = max(point_vram.values())
    draws = {
        **{f"gpu{gpu}_latency": [] for gpu in prereg.PHYSICAL_GPU_IDS_V29E},
        **{f"gpu{gpu}_vram": [] for gpu in prereg.PHYSICAL_GPU_IDS_V29E},
        "global_latency": [],
        "global_vram": [],
    }
    rng = random.Random(prereg.BOOTSTRAP_SEED_V29E)
    for _ in range(bootstrap_resamples):
        indices = [rng.randrange(prereg.REPETITIONS_V29E) for _ in range(prereg.REPETITIONS_V29E)]
        replicate_latency = {}
        replicate_vram = {}
        for gpu in prereg.PHYSICAL_GPU_IDS_V29E:
            replicate_latency[gpu] = statistics.median(latency[gpu][index] for index in indices)
            replicate_vram[gpu] = statistics.median(vram[gpu][index] for index in indices)
            draws[f"gpu{gpu}_latency"].append(replicate_latency[gpu])
            draws[f"gpu{gpu}_vram"].append(replicate_vram[gpu])
        draws["global_latency"].append(math.exp(statistics.fmean(
            math.log(replicate_latency[gpu])
            for gpu in prereg.PHYSICAL_GPU_IDS_V29E
        )))
        draws["global_vram"].append(max(replicate_vram.values()))
    alpha = prereg.PER_ENDPOINT_ALPHA_V29E
    per_gpu = {}
    for gpu in prereg.PHYSICAL_GPU_IDS_V29E:
        latency_lcb = _quantile_v29e(
            draws[f"gpu{gpu}_latency"], alpha, upper=False,
        )
        vram_ucb = _quantile_v29e(
            draws[f"gpu{gpu}_vram"], 1.0 - alpha, upper=True,
        )
        per_gpu[str(gpu)] = {
            "batch_size": prereg.BATCH_BY_GPU_V29E[gpu],
            "median_latency_speedup": point_latency[gpu],
            "familywise_latency_lower_bound": latency_lcb,
            "median_peak_vram_ratio": point_vram[gpu],
            "familywise_peak_vram_upper_bound": vram_ucb,
            "latency_gate_pass": latency_lcb >= 1.0,
            "peak_vram_gate_pass": vram_ucb <= 1.0,
        }
    global_latency_lcb = _quantile_v29e(
        draws["global_latency"], alpha, upper=False,
    )
    global_vram_ucb = _quantile_v29e(
        draws["global_vram"], 1.0 - alpha, upper=True,
    )
    exact_output_pass = all(output_matches) and len(output_matches) == 32
    passed = (
        exact_output_pass
        and all(item["latency_gate_pass"] for item in per_gpu.values())
        and all(item["peak_vram_gate_pass"] for item in per_gpu.values())
        and global_latency_lcb >= 1.0
        and global_vram_ucb <= 1.0
    )
    return _seal({
        "schema": "vllm-moe-fp8-selected-table-paired-summary-v29e",
        "repetitions": prereg.REPETITIONS_V29E,
        "bootstrap_seed": prereg.BOOTSTRAP_SEED_V29E,
        "bootstrap_resamples": bootstrap_resamples,
        "familywise_alpha": prereg.FAMILYWISE_ALPHA_V29E,
        "endpoint_count": prereg.ENDPOINT_COUNT_V29E,
        "per_endpoint_one_sided_alpha": alpha,
        "per_gpu": per_gpu,
        "global": {
            "latency_geometric_mean_speedup": point_global_latency,
            "familywise_latency_lower_bound": global_latency_lcb,
            "peak_vram_max_ratio": point_global_vram,
            "familywise_peak_vram_upper_bound": global_vram_ucb,
            "latency_gate_pass": global_latency_lcb >= 1.0,
            "peak_vram_gate_pass": global_vram_ucb <= 1.0,
        },
        "exact_output_equivalence": {
            "matched_pairs": sum(output_matches),
            "required_pairs": 32,
            "pass": exact_output_pass,
            "paired_output_digest_commitment_sha256": canonical_sha256(
                output_commitments
            ),
        },
        "pass": passed,
        "decision": (
            "authorize_only_separate_runtime_or_training_ab_preregistration"
            if passed else "retain_generic_defaults_and_reject_selected_table"
        ),
        "direct_table_adoption_model_update_training_checkpoint_dataset_promotion_authorized": False,
        "dataset_evaluation_validation_heldout_ood_or_benchmark_access_authorized": False,
        "raw_timing_memory_input_output_vectors_or_pids_persisted": False,
    })


def validate_launch_arguments_v29e(args, implementation, recipe):
    if any(os.environ.get(key) for key in MOE_ENVIRONMENT_V29E):
        raise ValueError("V29E rejects external MoE backend or config overrides")
    if args.v29e_dry_run and args.launch_v29e:
        raise ValueError("V29E dry-run and launch flags are mutually exclusive")
    if not args.v29e_dry_run and not args.launch_v29e:
        raise ValueError("V29E requires --v29e-dry-run or explicit --launch-v29e")
    if args.launch_v29e and os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3":
        raise ValueError("V29E real launch requires CUDA_VISIBLE_DEVICES=0,1,2,3")
    expected = (
        (args.expected_implementation_bundle_sha256, implementation["bundle_sha256"], "implementation"),
        (args.expected_recipe_sha256, recipe["content_sha256_before_self_field"], "recipe"),
    )
    for supplied, actual, label in expected:
        if args.launch_v29e and supplied is None:
            raise ValueError(f"V29E real launch requires expected {label} hash")
        if supplied is not None and supplied != actual:
            raise ValueError(f"V29E expected {label} hash changed")
    if args.launch_v29e:
        validate_preregistration_commit_v29e(args.expected_preregistration_commit)


def run_exact_v29e(preregistration, implementation, recipe):
    if (
        prereg.ATTEMPT_PATH_V29E.exists()
        or prereg.REPORT_PATH_V29E.parent.exists()
        or prereg.EMPTY_DEFAULT_DIRECTORY_V29E.exists()
    ):
        raise RuntimeError("V29E requires fresh exclusive output paths")
    environment = runtime_r2.certify_runtime_environment_r2()
    live_audit = live_cpu_disk_audit_v29e()
    prelaunch_idle = assert_all_four_gpus_idle_v29e()
    attempt = _seal({
        "schema": "vllm-moe-fp8-selected-table-evaluation-attempt-v29e",
        "status": "running",
        "phase": "before_first_counterbalanced_arm",
        "preregistration": recipe["preregistration"],
        "selection_evidence": recipe["selection_evidence"],
        "retry_of": recipe["retry_of"],
        "sole_infrastructure_correction": recipe[
            "sole_infrastructure_correction"
        ],
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
        "direct_action_taken": False,
        "dataset_evaluation_validation_heldout_ood_or_benchmark_surface_opened": False,
    })
    _assert_compact_v29e(attempt)
    _exclusive_write_json_v29e(prereg.ATTEMPT_PATH_V29E, attempt)
    records = []
    arm_integrity = []
    idle_commitments = []
    final_idle = None
    try:
        prereg.EMPTY_DEFAULT_DIRECTORY_V29E.mkdir(parents=True, exist_ok=False)
        tuned_configs = recipe["selected_table"]["exact_configs"]
        for scheduled in recipe["schedule"]["paired_counterbalanced_schedule"]:
            record = {
                "repetition": scheduled["repetition"],
                "seed": scheduled["seed"],
                "arms": {},
            }
            for arm in scheduled["arm_order"]:
                assert_all_four_gpus_idle_v29e()
                result = run_arm_v29e(
                    arm, scheduled["seed"], tuned_configs,
                    prereg.EMPTY_DEFAULT_DIRECTORY_V29E,
                )
                after_idle = wait_for_all_four_gpus_idle_v29e()
                final_idle = after_idle
                record["arms"][arm] = result["results"]
                idle_commitments.append(after_idle["content_sha256_before_self_field"])
                utilization = result["utilization"]
                arm_integrity.append({
                    "repetition": scheduled["repetition"],
                    "arm": arm,
                    "placement_identity_commitment_sha256": result[
                        "placement_identity_commitment_sha256"
                    ],
                    "actor_pid_map_commitment_sha256": result[
                        "actor_pid_map_commitment_sha256"
                    ],
                    "result_commitment_sha256": canonical_sha256(result["results"]),
                    "common_activity_target_commitment_sha256": result[
                        "common_activity_target_commitment_sha256"
                    ],
                    "activity_witness_result_commitment_sha256": result[
                        "activity_witness_result_commitment_sha256"
                    ],
                    "activity_witness_excluded_and_peak_memory_reset_before_measurement": result[
                        "activity_witness_excluded_and_peak_memory_reset_before_measurement"
                    ],
                    "simultaneous_all_four_observation_count": utilization[
                        "simultaneous_all_four_assigned_pids_and_positive_utilization_count"
                    ],
                    "per_gpu_utilization_commitment_sha256": canonical_sha256(
                        utilization["per_gpu"]
                    ),
                    "pass": utilization["pass"],
                })
            records.append(record)
        summary = summarize_v29e(records)
        report = _seal({
            "schema": "vllm-moe-fp8-selected-table-evaluation-report-v29e",
            "status": "complete_synthetic_kernel_evaluation",
            "preregistration": recipe["preregistration"],
            "selection_evidence": recipe["selection_evidence"],
            "retry_of": recipe["retry_of"],
            "sole_infrastructure_correction": recipe[
                "sole_infrastructure_correction"
            ],
            "selected_table": {
                "path": str(prereg.TABLE_PATH_V29E),
                "file_sha256": prereg.TABLE_FILE_SHA256_V29E,
                "content_sha256": prereg.TABLE_CONTENT_SHA256_V29E,
            },
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
                "arm_count": len(arm_integrity),
                "all_16_fresh_four_worker_arms_passed": (
                    len(arm_integrity) == 16
                    and all(item["pass"] for item in arm_integrity)
                ),
                "activity_witness_recipe_sha256": (
                    prereg.ACTIVITY_WITNESS_RECIPE_SHA256_V29E
                ),
                "official_num_iters": prereg.OFFICIAL_NUM_ITERS_V29E,
                "nvml_poll_interval_seconds": (
                    prereg.NVML_POLL_INTERVAL_SECONDS_V29E
                ),
                "all_16_common_start_activity_witnesses_excluded_and_peak_memory_reset": (
                    len(arm_integrity) == 16
                    and all(
                        item[
                            "activity_witness_excluded_and_peak_memory_reset_before_measurement"
                        ] is True
                        for item in arm_integrity
                    )
                ),
                "minimum_one_simultaneous_all_four_positive_observation_per_arm": (
                    len(arm_integrity) == 16
                    and all(
                        item["simultaneous_all_four_observation_count"] >= 1
                        for item in arm_integrity
                    )
                ),
                "arm_integrity_commitment_sha256": canonical_sha256(arm_integrity),
                "between_arm_and_final_idle_commitment_sha256": canonical_sha256(
                    idle_commitments
                ),
                "all_four_exact_gpus_finally_idle": final_idle["all_four_idle"],
                "raw_worker_pids_persisted": False,
            },
            "summary": summary,
            "direct_table_adoption_model_update_training_checkpoint_dataset_promotion_applied": False,
            "dataset_evaluation_validation_heldout_ood_or_benchmark_surface_opened": False,
            "raw_timing_memory_input_output_vectors_or_pids_persisted": False,
        })
        _assert_compact_v29e(report)
        _exclusive_write_json_v29e(prereg.REPORT_PATH_V29E, report)
        complete = _seal({
            **_without_self(attempt),
            "status": "complete",
            "phase": "after_compact_report_and_final_gpu_cleanup",
            "report_binding": {
                "path": str(prereg.REPORT_PATH_V29E),
                "file_sha256": file_sha256(prereg.REPORT_PATH_V29E),
                "content_sha256": report["content_sha256_before_self_field"],
            },
            "final_idle_certificate_sha256": final_idle[
                "content_sha256_before_self_field"
            ],
        })
        _assert_compact_v29e(complete)
        _rewrite_json_v29e(prereg.ATTEMPT_PATH_V29E, complete)
        return report
    except BaseException as failure:
        if final_idle is None:
            try:
                final_idle = wait_for_all_four_gpus_idle_v29e()
            except BaseException:
                final_idle = None
        failed = _seal({
            **_without_self(attempt),
            "status": "failed",
            "phase": "inside_counterbalanced_synthetic_kernel_evaluation",
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
        _assert_compact_v29e(failed)
        _rewrite_json_v29e(prereg.ATTEMPT_PATH_V29E, failed)
        raise


def _parser_v29e():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v29e-dry-run", action="store_true")
    parser.add_argument("--launch-v29e", action="store_true")
    parser.add_argument("--expected-preregistration-commit")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-recipe-sha256")
    return parser


def main(argv=None):
    args = _parser_v29e().parse_args(argv)
    preregistration = load_preregistration_v29e()
    implementation = implementation_identity_v29e()
    recipe = recipe_v29e(preregistration, implementation)
    validate_launch_arguments_v29e(args, implementation, recipe)
    if args.v29e_dry_run:
        value = _seal({
            "schema": "vllm-moe-fp8-selected-table-evaluation-dry-run-v29e",
            "preregistration_file_sha256": PREREG_FILE_SHA256_V29E,
            "preregistration_content_sha256": PREREG_CONTENT_SHA256_V29E,
            "implementation_bundle_sha256": implementation["bundle_sha256"],
            "recipe_content_sha256": recipe["content_sha256_before_self_field"],
            "repetitions": prereg.REPETITIONS_V29E,
            "bootstrap_resamples": prereg.BOOTSTRAP_RESAMPLES_V29E,
            "gpu_launched": False,
            "evaluation_launched": False,
            "dataset_evaluation_validation_heldout_ood_or_benchmark_surface_opened": False,
        })
        print(json.dumps(value, indent=2, sort_keys=True))
        return value
    return run_exact_v29e(preregistration, implementation, recipe)


if __name__ == "__main__":
    main(sys.argv[1:])
