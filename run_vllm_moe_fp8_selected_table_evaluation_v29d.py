#!/usr/bin/env python3
"""Fail-closed V29D FP8 selected-table synthetic-kernel evaluator."""

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

import build_vllm_moe_fp8_selected_table_evaluation_preregistration_v29d as prereg
import build_vllm_moe_fp8_tuning_preregistration_v29b as v29b
import run_eggroll_es_insertion_stability_v23a_retry_r2 as runtime_r2
import run_vllm_moe_fp8_tuning_v29b as runtime_v29b


ROOT = Path(__file__).resolve().parent
PREREG_PATH_V29D = prereg.OUTPUT_PATH_V29D
PREREG_FILE_SHA256_V29D = (
    "55519ce1150f3466817cbe3b7f6d22b4b79553755fc35c7e28a45b038dba7f42"
)
PREREG_CONTENT_SHA256_V29D = (
    "8bb3ecd02992f46cc2eb8a7f3144966ab5139f4b59fd1e179ef021a5d5687061"
)
PREREG_TEST_PATH_V29D = ROOT / (
    "test_build_vllm_moe_fp8_selected_table_evaluation_preregistration_v29d.py"
)
RUNTIME_TEST_PATH_V29D = ROOT / (
    "test_run_vllm_moe_fp8_selected_table_evaluation_v29d.py"
)
MOE_ENVIRONMENT_V29D = (
    "VLLM_TUNED_CONFIG_FOLDER",
    "VLLM_BATCH_INVARIANT",
    "VLLM_MOE_TUNE_CACHE_CLEAR_INTERVAL",
    "VLLM_ROCM_USE_AITER",
    "VLLM_ROCM_USE_AITER_MOE",
    "VLLM_USE_DEEP_GEMM",
    "VLLM_MOE_USE_DEEP_GEMM",
)
FORBIDDEN_COMPACT_KEYS_V29D = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "response", "responses", "text", "texts", "token_ids", "raw_pids",
    "timing_vectors", "input_tensor", "output_tensor", "training_rows",
    "evaluation_rows", "validation_rows", "heldout_rows", "ood_rows",
    "benchmark_rows", "traceback", "compiler_log", "progress_log",
}
IMPLEMENTATION_PATHS_V29D = {
    "v29d_preregistration_builder": Path(prereg.__file__).resolve(),
    "v29d_preregistration_tests": PREREG_TEST_PATH_V29D,
    "v29d_preregistration": PREREG_PATH_V29D,
    "v29d_runtime": Path(__file__).resolve(),
    "v29d_runtime_tests": RUNTIME_TEST_PATH_V29D,
    "v29c_selection_evidence": prereg.EVIDENCE_PATH_V29D,
    "v29b_original_selected_table": prereg.TABLE_PATH_V29D,
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


def _assert_compact_v29d(value):
    overlap = FORBIDDEN_COMPACT_KEYS_V29D & set(_recursive_keys(value))
    if overlap:
        raise RuntimeError(f"V29D compact output contains forbidden keys: {sorted(overlap)}")


def _exclusive_write_json_v29d(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def _rewrite_json_v29d(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_preregistration_v29d():
    value = json.loads(PREREG_PATH_V29D.read_text(encoding="utf-8"))
    if (
        file_sha256(PREREG_PATH_V29D) != PREREG_FILE_SHA256_V29D
        or value.get("content_sha256_before_self_field")
        != PREREG_CONTENT_SHA256_V29D
        or canonical_sha256(_without_self(value)) != PREREG_CONTENT_SHA256_V29D
    ):
        raise RuntimeError("V29D preregistration identity changed")
    return prereg.validate_preregistration_v29d(value)


def implementation_identity_v29d():
    files = {
        key: {
            "path": str(Path(path).resolve()),
            "file_sha256": file_sha256(path),
        }
        for key, path in IMPLEMENTATION_PATHS_V29D.items()
    }
    return {"files": files, "bundle_sha256": canonical_sha256(files)}


def recipe_v29d(preregistration, implementation):
    return _seal({
        "schema": "vllm-moe-fp8-selected-table-evaluation-recipe-v29d",
        "experiment_name": prereg.EXPERIMENT_NAME_V29D,
        "preregistration": {
            "path": str(PREREG_PATH_V29D),
            "file_sha256": PREREG_FILE_SHA256_V29D,
            "content_sha256": PREREG_CONTENT_SHA256_V29D,
        },
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "selection_evidence": copy.deepcopy(
            preregistration["selection_evidence"]
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


def validate_preregistration_commit_v29d(commit):
    if not isinstance(commit, str) or len(commit) != 40 or any(
        char not in "0123456789abcdef" for char in commit
    ):
        raise ValueError("V29D real launch requires exact lowercase preregistration commit")
    relative = PREREG_PATH_V29D.relative_to(ROOT).as_posix()
    raw = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
    )
    if hashlib.sha256(raw).hexdigest() != PREREG_FILE_SHA256_V29D:
        raise RuntimeError("V29D committed preregistration identity changed")
    return commit


def normalize_ray_gpu_id_v29d(value):
    if isinstance(value, bool):
        raise RuntimeError("V29D Ray GPU ID representation changed")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value in {"0", "1", "2", "3"}:
        result = int(value)
    else:
        raise RuntimeError("V29D Ray GPU ID representation changed")
    if result not in prereg.PHYSICAL_GPU_IDS_V29D:
        raise RuntimeError("V29D Ray GPU ID physical range changed")
    return result


def live_cpu_disk_audit_v29d():
    prereg.validate_static_inputs_v29d()
    base = runtime_v29b.live_cpu_disk_audit_v29b()
    return _seal({
        "schema": "vllm-moe-fp8-selected-table-live-cpu-disk-audit-v29d",
        "v29b_full_model_and_runtime_audit_content_sha256": (
            base["content_sha256_before_self_field"]
        ),
        "model_all_56_files_and_42_weight_shards_rehashed": True,
        "selection_evidence_commit": prereg.EVIDENCE_COMMIT_V29D,
        "selection_evidence_file_sha256": prereg.EVIDENCE_FILE_SHA256_V29D,
        "selection_evidence_content_sha256": prereg.EVIDENCE_CONTENT_SHA256_V29D,
        "selected_table_file_sha256": prereg.TABLE_FILE_SHA256_V29D,
        "selected_table_content_sha256": prereg.TABLE_CONTENT_SHA256_V29D,
        "dataset_train_evaluation_validation_heldout_ood_or_benchmark_surface_opened": False,
    })


def _observe_all_four_gpus_v29d():
    observation = runtime_v29b._observe_all_four_gpus_v29b()
    return {
        "gpus": [
            {**row, "v29d_identity_checked": True}
            for row in observation["gpus"]
        ],
        "all_four_idle": observation["all_four_idle"],
    }


def assert_all_four_gpus_idle_v29d():
    observation = _observe_all_four_gpus_v29d()
    if observation.get("all_four_idle") is not True:
        raise RuntimeError("V29D requires all four GPUs idle before claim or arm")
    return _seal({
        "schema": "vllm-moe-fp8-selected-table-idle-certificate-v29d",
        **observation,
    })


def wait_for_all_four_gpus_idle_v29d(*, timeout_seconds=30.0, interval_seconds=0.5):
    if timeout_seconds != 30.0 or interval_seconds != 0.5:
        raise RuntimeError("V29D GPU cleanup polling contract changed")
    start = time.monotonic()
    polls = 0
    while True:
        observation = _observe_all_four_gpus_v29d()
        polls += 1
        elapsed = time.monotonic() - start
        if observation.get("all_four_idle") is True:
            return _seal({
                "schema": "vllm-moe-fp8-selected-table-idle-certificate-v29d",
                **observation,
                "poll_count": polls,
                "elapsed_milliseconds": int(round(elapsed * 1000)),
                "bounded_async_cleanup_wait": True,
            })
        if elapsed >= timeout_seconds:
            raise RuntimeError("V29D GPU cleanup exceeded 30 seconds")
        time.sleep(min(interval_seconds, timeout_seconds - elapsed))


def _load_official_module_v29d():
    if file_sha256(v29b.OFFICIAL_TUNER_PATH_V29B) != v29b.OFFICIAL_TUNER_SHA256_V29B:
        raise RuntimeError("V29D official vLLM tuner identity changed")
    spec = importlib.util.spec_from_file_location(
        "vllm_benchmark_moe_fp8_v29d", v29b.OFFICIAL_TUNER_PATH_V29B,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.triton.__version__ != "3.6.0":
        raise RuntimeError("V29D Triton runtime version changed")
    return module


def _validate_config_v29d(config):
    allowed = {
        "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
        "num_warps", "num_stages",
    }
    if (
        not isinstance(config, dict) or set(config) != allowed
        or any(isinstance(item, bool) or not isinstance(item, int) for item in config.values())
    ):
        raise RuntimeError("V29D kernel config geometry changed")
    return config


def _tensor_byte_sha256_v29d(tensor):
    contiguous = tensor.detach().contiguous().view(-1).view(dtype=__import__("torch").uint8)
    raw = contiguous.cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _worker_evaluate_v29d(physical_gpu, batch_size, seed, arm, tuned_config, empty_folder):
    import pynvml
    import ray
    import torch

    ray_ids = ray.get_gpu_ids()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if (
        not isinstance(ray_ids, list) or len(ray_ids) != 1
        or normalize_ray_gpu_id_v29d(ray_ids[0]) != physical_gpu
        or visible != str(physical_gpu)
    ):
        raise RuntimeError("V29D exact Ray physical-GPU assignment changed")
    if arm not in {"default", "tuned"}:
        raise RuntimeError("V29D arm changed")
    if not Path(empty_folder).is_dir() or any(Path(empty_folder).iterdir()):
        raise RuntimeError("V29D default config directory is not exactly empty")
    os.environ["VLLM_TUNED_CONFIG_FOLDER"] = str(empty_folder)
    module = _load_official_module_v29d()
    torch.set_default_device("cuda")
    dtype = torch.bfloat16
    dtype_string = module._get_config_dtype_str(
        dtype, use_int8_w8a16=False, use_fp8_w8a8=True,
        use_int4_w4a16=False,
    )
    if module.get_moe_configs(256, 512, dtype_string, 128, 128) is not None:
        raise RuntimeError("V29D empty-folder official default lookup was not empty")
    if arm == "default":
        selected = module.get_default_config(
            batch_size, 256, 1024, 2048, 8, dtype_string, [128, 128],
        )
    else:
        selected = _validate_config_v29d(copy.deepcopy(tuned_config))
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
            True, False, False, num_iters=prereg.OFFICIAL_NUM_ITERS_V29D,
            block_quant_shape=[128, 128], use_deep_gemm=False,
        )
        torch.cuda.synchronize()
        if "output" not in captured:
            raise RuntimeError("V29D official kernel output capture failed")
        output = captured["output"]
        output_sha256 = _tensor_byte_sha256_v29d(output)
        output_shape = list(output.shape)
        output_dtype = str(output.dtype)
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
    finally:
        module.fused_experts = original_fused_experts
    if not math.isfinite(microseconds) or microseconds <= 0:
        raise RuntimeError("V29D nonfinite kernel time")
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
        raise RuntimeError("V29D worker exact UUID/PCI/runtime device changed")
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
        "num_iters": prereg.OFFICIAL_NUM_ITERS_V29D,
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
    }


def _probe_identity_matches_v29d(identity, gpu_id):
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


def _official_actor_pids_v29d(ray, workers):
    ray.get([worker.__ray_ready__.remote() for worker in workers.values()])
    values = ray.get([
        workers[gpu].__ray_call__.remote(lambda self: os.getpid())
        for gpu in prereg.PHYSICAL_GPU_IDS_V29D
    ])
    result = dict(zip(prereg.PHYSICAL_GPU_IDS_V29D, values))
    if (
        any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in values)
        or len(set(values)) != 4
    ):
        raise RuntimeError("V29D requires four distinct official worker PIDs")
    return result


def _monitor_four_futures_v29d(ray, futures, actor_pids_by_gpu):
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
        for gpu in prereg.PHYSICAL_GPU_IDS_V29D
    }
    simultaneous = 0
    while pending:
        observation = _observe_all_four_gpus_v29d()
        rows = {row["physical_gpu_id"]: row for row in observation["gpus"]}
        if set(rows) != set(prereg.PHYSICAL_GPU_IDS_V29D):
            raise RuntimeError("V29D in-flight physical GPU surface changed")
        if all(
            actor_pids_by_gpu[gpu] in rows[gpu]["running_process_pids"]
            and rows[gpu]["gpu_utilization_percent"] > 0
            for gpu in prereg.PHYSICAL_GPU_IDS_V29D
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
        ready, pending = ray.wait(pending, num_returns=1, timeout=0.25)
        if ready:
            ray.get(ready)
    if simultaneous < 1 or any(
        item["assigned_pid_observed"] is not True
        or item["positive_utilization_observed"] is not True
        for item in aggregate.values()
    ):
        raise RuntimeError("V29D did not observe all four assigned PIDs with utilization")
    return {
        "per_gpu": aggregate,
        "simultaneous_all_four_assigned_pids_and_positive_utilization_count": simultaneous,
        "pass": True,
    }


def run_arm_v29d(arm, seed, tuned_configs, empty_folder):
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    if ray.is_initialized():
        raise RuntimeError("V29D requires a fresh Ray runtime per arm")
    ray.init(num_gpus=4, include_dashboard=False)

    @ray.remote(num_gpus=1)
    class PlacementProbeV29D:
        def identity(self):
            import pynvml

            ids = ray.get_gpu_ids()
            if not isinstance(ids, list) or len(ids) != 1:
                raise RuntimeError("V29D placement probe allocation changed")
            physical = normalize_ray_gpu_id_v29d(ids[0])
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
    class EvaluationWorkerV29D:
        def evaluate(self, physical_gpu, batch_size, worker_seed, worker_arm, config, folder):
            return _worker_evaluate_v29d(
                physical_gpu, batch_size, worker_seed, worker_arm, config, folder,
            )

    groups = []
    probes = []
    workers = {}
    try:
        for _ in prereg.PHYSICAL_GPU_IDS_V29D:
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
            PlacementProbeV29D.options(scheduling_strategy=strategy).remote()
            for strategy in strategies
        ]
        identities = ray.get([probe.identity.remote() for probe in probes])
        strategies_by_gpu = {}
        identity_by_gpu = {}
        for probe, identity, strategy in zip(probes, identities, strategies):
            gpu = identity["physical_gpu_id"]
            if gpu in strategies_by_gpu or not _probe_identity_matches_v29d(identity, gpu):
                raise RuntimeError("V29D placement probe identity changed")
            identity_by_gpu[gpu] = identity
            strategies_by_gpu[gpu] = strategy
            ray.kill(probe)
        probes = []
        if set(strategies_by_gpu) != set(prereg.PHYSICAL_GPU_IDS_V29D):
            raise RuntimeError("V29D exact four-GPU placement coverage changed")
        workers = {
            gpu: EvaluationWorkerV29D.options(
                scheduling_strategy=strategies_by_gpu[gpu]
            ).remote()
            for gpu in prereg.PHYSICAL_GPU_IDS_V29D
        }
        actor_pids = _official_actor_pids_v29d(ray, workers)
        futures = [
            workers[gpu].evaluate.remote(
                gpu, prereg.BATCH_BY_GPU_V29D[gpu], seed, arm,
                tuned_configs[str(prereg.BATCH_BY_GPU_V29D[gpu])],
                str(empty_folder),
            )
            for gpu in prereg.PHYSICAL_GPU_IDS_V29D
        ]
        utilization = _monitor_four_futures_v29d(ray, futures, actor_pids)
        results = ray.get(futures)
        if any(
            item.get("physical_gpu_id") != gpu
            or item.get("batch_size") != prereg.BATCH_BY_GPU_V29D[gpu]
            or item.get("seed") != seed
            or item.get("arm") != arm
            or item.get("nvml_uuid") != v29b.GPU_IDENTITIES_V29B[gpu]["uuid"]
            for gpu, item in enumerate(results)
        ):
            raise RuntimeError("V29D worker result identity changed")
        return {
            "results": results,
            "placement_identity_commitment_sha256": canonical_sha256(identity_by_gpu),
            "actor_pid_map_commitment_sha256": canonical_sha256(actor_pids),
            "utilization": utilization,
            "all_four_futures_submitted_before_wait": True,
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


def _quantile_v29d(values, probability, *, upper):
    ordered = sorted(values)
    if upper:
        index = max(0, math.ceil(probability * len(ordered)) - 1)
    else:
        index = min(len(ordered) - 1, math.floor(probability * len(ordered)))
    return ordered[index]


def summarize_v29d(records, *, bootstrap_resamples=None):
    if bootstrap_resamples is None:
        bootstrap_resamples = prereg.BOOTSTRAP_RESAMPLES_V29D
    if len(records) != prereg.REPETITIONS_V29D:
        raise RuntimeError("V29D paired repetition count changed")
    latency = {gpu: [] for gpu in prereg.PHYSICAL_GPU_IDS_V29D}
    vram = {gpu: [] for gpu in prereg.PHYSICAL_GPU_IDS_V29D}
    output_matches = []
    output_commitments = []
    for expected_repetition, record in enumerate(records):
        if record.get("repetition") != expected_repetition:
            raise RuntimeError("V29D repetition order changed")
        arms = record.get("arms", {})
        if set(arms) != {"default", "tuned"}:
            raise RuntimeError("V29D paired arms changed")
        default_by_gpu = {item["physical_gpu_id"]: item for item in arms["default"]}
        tuned_by_gpu = {item["physical_gpu_id"]: item for item in arms["tuned"]}
        if set(default_by_gpu) != set(prereg.PHYSICAL_GPU_IDS_V29D) or set(tuned_by_gpu) != set(prereg.PHYSICAL_GPU_IDS_V29D):
            raise RuntimeError("V29D paired GPU coverage changed")
        for gpu in prereg.PHYSICAL_GPU_IDS_V29D:
            default = default_by_gpu[gpu]
            tuned = tuned_by_gpu[gpu]
            if (
                default["seed"] != tuned["seed"]
                or default["input_recipe_sha256"] != tuned["input_recipe_sha256"]
                or default["batch_size"] != tuned["batch_size"]
            ):
                raise RuntimeError("V29D paired deterministic input identity changed")
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
        math.log(point_latency[gpu]) for gpu in prereg.PHYSICAL_GPU_IDS_V29D
    ))
    point_global_vram = max(point_vram.values())
    draws = {
        **{f"gpu{gpu}_latency": [] for gpu in prereg.PHYSICAL_GPU_IDS_V29D},
        **{f"gpu{gpu}_vram": [] for gpu in prereg.PHYSICAL_GPU_IDS_V29D},
        "global_latency": [],
        "global_vram": [],
    }
    rng = random.Random(prereg.BOOTSTRAP_SEED_V29D)
    for _ in range(bootstrap_resamples):
        indices = [rng.randrange(prereg.REPETITIONS_V29D) for _ in range(prereg.REPETITIONS_V29D)]
        replicate_latency = {}
        replicate_vram = {}
        for gpu in prereg.PHYSICAL_GPU_IDS_V29D:
            replicate_latency[gpu] = statistics.median(latency[gpu][index] for index in indices)
            replicate_vram[gpu] = statistics.median(vram[gpu][index] for index in indices)
            draws[f"gpu{gpu}_latency"].append(replicate_latency[gpu])
            draws[f"gpu{gpu}_vram"].append(replicate_vram[gpu])
        draws["global_latency"].append(math.exp(statistics.fmean(
            math.log(replicate_latency[gpu])
            for gpu in prereg.PHYSICAL_GPU_IDS_V29D
        )))
        draws["global_vram"].append(max(replicate_vram.values()))
    alpha = prereg.PER_ENDPOINT_ALPHA_V29D
    per_gpu = {}
    for gpu in prereg.PHYSICAL_GPU_IDS_V29D:
        latency_lcb = _quantile_v29d(
            draws[f"gpu{gpu}_latency"], alpha, upper=False,
        )
        vram_ucb = _quantile_v29d(
            draws[f"gpu{gpu}_vram"], 1.0 - alpha, upper=True,
        )
        per_gpu[str(gpu)] = {
            "batch_size": prereg.BATCH_BY_GPU_V29D[gpu],
            "median_latency_speedup": point_latency[gpu],
            "familywise_latency_lower_bound": latency_lcb,
            "median_peak_vram_ratio": point_vram[gpu],
            "familywise_peak_vram_upper_bound": vram_ucb,
            "latency_gate_pass": latency_lcb >= 1.0,
            "peak_vram_gate_pass": vram_ucb <= 1.0,
        }
    global_latency_lcb = _quantile_v29d(
        draws["global_latency"], alpha, upper=False,
    )
    global_vram_ucb = _quantile_v29d(
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
        "schema": "vllm-moe-fp8-selected-table-paired-summary-v29d",
        "repetitions": prereg.REPETITIONS_V29D,
        "bootstrap_seed": prereg.BOOTSTRAP_SEED_V29D,
        "bootstrap_resamples": bootstrap_resamples,
        "familywise_alpha": prereg.FAMILYWISE_ALPHA_V29D,
        "endpoint_count": prereg.ENDPOINT_COUNT_V29D,
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


def validate_launch_arguments_v29d(args, implementation, recipe):
    if any(os.environ.get(key) for key in MOE_ENVIRONMENT_V29D):
        raise ValueError("V29D rejects external MoE backend or config overrides")
    if args.v29d_dry_run and args.launch_v29d:
        raise ValueError("V29D dry-run and launch flags are mutually exclusive")
    if not args.v29d_dry_run and not args.launch_v29d:
        raise ValueError("V29D requires --v29d-dry-run or explicit --launch-v29d")
    if args.launch_v29d and os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3":
        raise ValueError("V29D real launch requires CUDA_VISIBLE_DEVICES=0,1,2,3")
    expected = (
        (args.expected_implementation_bundle_sha256, implementation["bundle_sha256"], "implementation"),
        (args.expected_recipe_sha256, recipe["content_sha256_before_self_field"], "recipe"),
    )
    for supplied, actual, label in expected:
        if args.launch_v29d and supplied is None:
            raise ValueError(f"V29D real launch requires expected {label} hash")
        if supplied is not None and supplied != actual:
            raise ValueError(f"V29D expected {label} hash changed")
    if args.launch_v29d:
        validate_preregistration_commit_v29d(args.expected_preregistration_commit)


def run_exact_v29d(preregistration, implementation, recipe):
    if (
        prereg.ATTEMPT_PATH_V29D.exists()
        or prereg.REPORT_PATH_V29D.parent.exists()
        or prereg.EMPTY_DEFAULT_DIRECTORY_V29D.exists()
    ):
        raise RuntimeError("V29D requires fresh exclusive output paths")
    environment = runtime_r2.certify_runtime_environment_r2()
    live_audit = live_cpu_disk_audit_v29d()
    prelaunch_idle = assert_all_four_gpus_idle_v29d()
    attempt = _seal({
        "schema": "vllm-moe-fp8-selected-table-evaluation-attempt-v29d",
        "status": "running",
        "phase": "before_first_counterbalanced_arm",
        "preregistration": recipe["preregistration"],
        "selection_evidence": recipe["selection_evidence"],
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
    _assert_compact_v29d(attempt)
    _exclusive_write_json_v29d(prereg.ATTEMPT_PATH_V29D, attempt)
    records = []
    arm_integrity = []
    idle_commitments = []
    final_idle = None
    try:
        prereg.EMPTY_DEFAULT_DIRECTORY_V29D.mkdir(parents=True, exist_ok=False)
        tuned_configs = recipe["selected_table"]["exact_configs"]
        for scheduled in recipe["schedule"]["paired_counterbalanced_schedule"]:
            record = {
                "repetition": scheduled["repetition"],
                "seed": scheduled["seed"],
                "arms": {},
            }
            for arm in scheduled["arm_order"]:
                assert_all_four_gpus_idle_v29d()
                result = run_arm_v29d(
                    arm, scheduled["seed"], tuned_configs,
                    prereg.EMPTY_DEFAULT_DIRECTORY_V29D,
                )
                after_idle = wait_for_all_four_gpus_idle_v29d()
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
                    "simultaneous_all_four_observation_count": utilization[
                        "simultaneous_all_four_assigned_pids_and_positive_utilization_count"
                    ],
                    "per_gpu_utilization_commitment_sha256": canonical_sha256(
                        utilization["per_gpu"]
                    ),
                    "pass": utilization["pass"],
                })
            records.append(record)
        summary = summarize_v29d(records)
        report = _seal({
            "schema": "vllm-moe-fp8-selected-table-evaluation-report-v29d",
            "status": "complete_synthetic_kernel_evaluation",
            "preregistration": recipe["preregistration"],
            "selection_evidence": recipe["selection_evidence"],
            "selected_table": {
                "path": str(prereg.TABLE_PATH_V29D),
                "file_sha256": prereg.TABLE_FILE_SHA256_V29D,
                "content_sha256": prereg.TABLE_CONTENT_SHA256_V29D,
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
        _assert_compact_v29d(report)
        _exclusive_write_json_v29d(prereg.REPORT_PATH_V29D, report)
        complete = _seal({
            **_without_self(attempt),
            "status": "complete",
            "phase": "after_compact_report_and_final_gpu_cleanup",
            "report_binding": {
                "path": str(prereg.REPORT_PATH_V29D),
                "file_sha256": file_sha256(prereg.REPORT_PATH_V29D),
                "content_sha256": report["content_sha256_before_self_field"],
            },
            "final_idle_certificate_sha256": final_idle[
                "content_sha256_before_self_field"
            ],
        })
        _assert_compact_v29d(complete)
        _rewrite_json_v29d(prereg.ATTEMPT_PATH_V29D, complete)
        return report
    except BaseException as failure:
        if final_idle is None:
            try:
                final_idle = wait_for_all_four_gpus_idle_v29d()
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
        _assert_compact_v29d(failed)
        _rewrite_json_v29d(prereg.ATTEMPT_PATH_V29D, failed)
        raise


def _parser_v29d():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v29d-dry-run", action="store_true")
    parser.add_argument("--launch-v29d", action="store_true")
    parser.add_argument("--expected-preregistration-commit")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-recipe-sha256")
    return parser


def main(argv=None):
    args = _parser_v29d().parse_args(argv)
    preregistration = load_preregistration_v29d()
    implementation = implementation_identity_v29d()
    recipe = recipe_v29d(preregistration, implementation)
    validate_launch_arguments_v29d(args, implementation, recipe)
    if args.v29d_dry_run:
        value = _seal({
            "schema": "vllm-moe-fp8-selected-table-evaluation-dry-run-v29d",
            "preregistration_file_sha256": PREREG_FILE_SHA256_V29D,
            "preregistration_content_sha256": PREREG_CONTENT_SHA256_V29D,
            "implementation_bundle_sha256": implementation["bundle_sha256"],
            "recipe_content_sha256": recipe["content_sha256_before_self_field"],
            "repetitions": prereg.REPETITIONS_V29D,
            "bootstrap_resamples": prereg.BOOTSTRAP_RESAMPLES_V29D,
            "gpu_launched": False,
            "evaluation_launched": False,
            "dataset_evaluation_validation_heldout_ood_or_benchmark_surface_opened": False,
        })
        print(json.dumps(value, indent=2, sort_keys=True))
        return value
    return run_exact_v29d(preregistration, implementation, recipe)


if __name__ == "__main__":
    main(sys.argv[1:])
