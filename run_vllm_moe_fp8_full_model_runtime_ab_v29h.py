#!/usr/bin/env python3
"""Fail-closed V29H serialized-FP8 full-model selected-table runtime A/B."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import build_vllm_moe_fp8_full_model_runtime_ab_preregistration_v29h as prereg
import build_vllm_moe_fp8_tuning_preregistration_v29b as v29b
import run_eggroll_es_insertion_stability_v23a as runtime_v23a
import run_eggroll_es_insertion_stability_v23a_retry_r2 as runtime_r2
import run_vllm_moe_fp8_selected_table_evaluation_retry_v29e as runtime_v29e
import run_vllm_moe_fp8_tuning_v29b as runtime_v29b


ROOT = Path(__file__).resolve().parent
PREREG_PATH_V29H = prereg.OUTPUT_PATH_V29H
PREREG_FILE_SHA256_V29H = (
    "1fd08762b4e4926af9b9614768debe57177788bccf621d8d17ab83fe8393d7f2"
)
PREREG_CONTENT_SHA256_V29H = (
    "17e0cf1b7ea560e8e446d50bffcd97f6f110cda4ff0624abd5b37e6ce83908d8"
)
EXPERIMENT_NAME_V29H = "s6_v29h_serialized_fp8_full_model_runtime_ab"
OUTPUT_DIRECTORY_V29H = (ROOT / "experiments/eggroll_es_hpo/runs").resolve()
ATTEMPT_NAME_V29H = f".{EXPERIMENT_NAME_V29H}.launch_attempt.json"
REPORT_NAME_V29H = "serialized_fp8_full_model_runtime_ab_report_v29h.json"
PREREG_TEST_PATH_V29H = (
    ROOT / "test_build_vllm_moe_fp8_full_model_runtime_ab_preregistration_v29h.py"
).resolve()
RUNTIME_TEST_PATH_V29H = (
    ROOT / "test_run_vllm_moe_fp8_full_model_runtime_ab_v29h.py"
).resolve()
REQUIRED_PYTHON_V29H = (ROOT / "es-at-scale/.venv/bin/python").absolute()

IMPLEMENTATION_PATHS_V29H = {
    "v29h_preregistration_builder": Path(prereg.__file__).resolve(),
    "v29h_preregistration_tests": PREREG_TEST_PATH_V29H,
    "v29h_preregistration": PREREG_PATH_V29H,
    "v29h_runtime": Path(__file__).resolve(),
    "v29h_runtime_tests": RUNTIME_TEST_PATH_V29H,
    "v29f_positive_evidence_builder": (
        ROOT / "build_vllm_moe_fp8_selected_table_evaluation_positive_evidence_v29f.py"
    ),
    "v29f_positive_evidence_tests": (
        ROOT / "test_build_vllm_moe_fp8_selected_table_evaluation_positive_evidence_v29f.py"
    ),
    "v29f_positive_evidence": prereg.V29F_EVIDENCE_PATH_V29H,
    "v29e_preregistration": prereg.V29E_PREREG_PATH_V29H,
    "v29_selected_table": prereg.TABLE_PATH_V29H,
}
MOE_ENVIRONMENT_V29H = (
    "VLLM_TUNED_CONFIG_FOLDER", "VLLM_BATCH_INVARIANT",
    "VLLM_MOE_TUNE_CACHE_CLEAR_INTERVAL", "VLLM_ROCM_USE_AITER",
    "VLLM_ROCM_USE_AITER_MOE", "VLLM_USE_DEEP_GEMM",
    "VLLM_MOE_USE_DEEP_GEMM",
)
FORBIDDEN_ARGV_TOKENS_V29H = (
    "checkpoint", "update", "training", "dataset", "heldout", "validation",
    "ood", "benchmark", "eval", "promotion", "bf16-model",
)
FORBIDDEN_COMPACT_KEYS_V29H = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "prompt_token_ids", "token_ids", "text", "texts", "outputs", "logprobs",
    "elapsed_ns", "timing_vectors", "memory_samples", "peak_allocated_bytes",
    "peak_reserved_bytes", "pid", "pids", "bootstrap_draws",
    "bootstrap_replicates", "dataset_rows", "traceback",
}
ALLOWED_UNTRACKED_PREFIXES_V29H = ("experiments/dataset_probes/",)
ALLOWED_UNTRACKED_PATHS_V29H = frozenset({
    "experiments/gpu_utilization_v31_base_qwen36_train_reward_probe.jsonl",
})
CURATOR_UNTRACKED_PATTERN_V29H = re.compile(
    r"data/manual_reviews/context_merit_audit_v([0-9]+)/.+\Z"
)
WORKTREE_ALLOWLIST_CONTRACT_V29H = {
    "curator_snapshot_directories": (
        "data/manual_reviews/context_merit_audit_vN/ where N >= 390"
    ),
    "path_prefixes": list(ALLOWED_UNTRACKED_PREFIXES_V29H),
    "exact_paths": sorted(ALLOWED_UNTRACKED_PATHS_V29H),
    "tracked_changes_allowed": False,
    "other_untracked_paths_allowed": False,
}


canonical_sha256 = prereg.canonical_sha256
file_sha256 = prereg.file_sha256
_seal = runtime_v23a._seal_v23a


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _recursive_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).lower()
            yield from _recursive_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _recursive_keys(item)


def _assert_compact_v29h(value):
    overlap = FORBIDDEN_COMPACT_KEYS_V29H & set(_recursive_keys(value))
    if overlap:
        raise RuntimeError(
            f"V29H compact output contains forbidden keys: {sorted(overlap)}"
        )


def _assert_closed_argv_v29h(argv):
    for token in argv:
        lowered = str(token).lower()
        if any(item in lowered for item in FORBIDDEN_ARGV_TOKENS_V29H):
            raise ValueError(f"V29H rejects forbidden surface: {token}")


def _current_prereg_content_sha256_v29h():
    value = json.loads(PREREG_PATH_V29H.read_text(encoding="utf-8"))
    return value.get("content_sha256_before_self_field")


def load_preregistration_v29h():
    value = json.loads(PREREG_PATH_V29H.read_text(encoding="utf-8"))
    expected_content = PREREG_CONTENT_SHA256_V29H
    if expected_content is None:
        raise RuntimeError("V29H runtime requires frozen preregistration content hash")
    if (
        file_sha256(PREREG_PATH_V29H) != PREREG_FILE_SHA256_V29H
        or value.get("content_sha256_before_self_field") != expected_content
        or canonical_sha256(_without_self(value)) != expected_content
    ):
        raise RuntimeError("V29H preregistration identity changed")
    return prereg.validate_preregistration_v29h(value)


def implementation_identity_v29h():
    files = {
        key: {
            "path": str(Path(path).resolve()),
            "file_sha256": file_sha256(path),
        }
        for key, path in IMPLEMENTATION_PATHS_V29H.items()
    }
    return {"files": files, "bundle_sha256": canonical_sha256(files)}


def _is_allowed_untracked_v29h(relative):
    if relative in ALLOWED_UNTRACKED_PATHS_V29H or any(
        relative.startswith(prefix) for prefix in ALLOWED_UNTRACKED_PREFIXES_V29H
    ):
        return True
    match = CURATOR_UNTRACKED_PATTERN_V29H.fullmatch(relative)
    return match is not None and int(match.group(1)) >= 390


def validate_worktree_status_v29h(raw_status):
    if not isinstance(raw_status, str):
        raise TypeError("V29H worktree status must be text")
    allowed = []
    rejected = []
    for line in raw_status.splitlines():
        if len(line) < 4 or line[2] != " ":
            raise RuntimeError("V29H worktree status record changed")
        status, relative = line[:2], line[3:]
        if status == "??" and _is_allowed_untracked_v29h(relative):
            allowed.append(relative)
        else:
            rejected.append((status, relative))
    if rejected:
        raise RuntimeError(
            "V29H real launch requires committed-clean source outside the exact "
            "untracked allowlist"
        )
    return {
        "all_tracked_files_clean": True,
        "only_explicitly_allowlisted_untracked_paths_present": True,
        "allowed_untracked_entry_count": len(allowed),
        "allowed_untracked_entries_sha256": canonical_sha256(sorted(allowed)),
        "allowlist_contract_sha256": canonical_sha256(
            WORKTREE_ALLOWLIST_CONTRACT_V29H
        ),
    }


def certify_committed_source_v29h(implementation, expected_source_commit):
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    if len(head) != 40 or head != expected_source_commit:
        raise RuntimeError("V29H exact expected source commit changed")
    committed = {}
    for key, item in implementation["files"].items():
        path = Path(item["path"]).resolve()
        if not path.is_relative_to(ROOT):
            continue
        relative = path.relative_to(ROOT).as_posix()
        try:
            raw = subprocess.check_output(
                ["git", "show", f"{head}:{relative}"], cwd=ROOT,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"V29H real launch requires committed source: {relative}"
            ) from error
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item["file_sha256"]:
            raise RuntimeError(f"V29H source differs from HEAD: {relative}")
        committed[key] = digest
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT, text=True,
    )
    return _seal({
        "schema": "vllm-moe-fp8-full-model-committed-source-v29h",
        "git_head": head,
        "implementation_file_count": len(committed),
        "committed_implementation_sha256": canonical_sha256(committed),
        **validate_worktree_status_v29h(status),
    })


def live_cpu_disk_audit_v29h():
    prereg.validate_static_inputs_v29h()
    base = runtime_v29b.live_cpu_disk_audit_v29b()
    return _seal({
        "schema": "vllm-moe-fp8-full-model-live-cpu-disk-audit-v29h",
        "v29b_full_model_audit_sha256": base[
            "content_sha256_before_self_field"
        ],
        "all_56_model_files_and_42_weight_shards_rehashed": True,
        "model_all_files_fingerprint_sha256": (
            prereg.MODEL_ALL_FILES_FINGERPRINT_SHA256_V29H
        ),
        "selected_table_file_sha256": prereg.TABLE_FILE_SHA256_V29H,
        "selected_table_content_sha256": prereg.TABLE_CONTENT_SHA256_V29H,
        "v29f_evidence_file_sha256": prereg.V29F_EVIDENCE_FILE_SHA256_V29H,
        "dataset_or_semantic_content_opened": False,
    })


def recipe_v29h(preregistration, implementation):
    return _seal({
        "schema": "vllm-moe-fp8-full-model-runtime-ab-recipe-v29h",
        "experiment_name": EXPERIMENT_NAME_V29H,
        "preregistration": {
            "path": str(PREREG_PATH_V29H),
            "file_sha256": PREREG_FILE_SHA256_V29H,
            "content_sha256": PREREG_CONTENT_SHA256_V29H,
        },
        "authorization_basis": copy.deepcopy(
            preregistration["authorization_basis"]
        ),
        "serialized_fp8_model_contract": copy.deepcopy(
            preregistration["serialized_fp8_model_contract"]
        ),
        "selected_table_contract": copy.deepcopy(
            preregistration["selected_table_contract"]
        ),
        "arms": copy.deepcopy(preregistration["arms"]),
        "synthetic_request_contract": copy.deepcopy(
            preregistration["synthetic_request_contract"]
        ),
        "schedule": copy.deepcopy(preregistration["schedule"]),
        "request_budget": copy.deepcopy(preregistration["request_budget"]),
        "runtime_contract": copy.deepcopy(preregistration["runtime_contract"]),
        "exact_equivalence_contract": copy.deepcopy(
            preregistration["exact_equivalence_contract"]
        ),
        "statistical_contract": copy.deepcopy(
            preregistration["statistical_contract"]
        ),
        "gate_and_authority": copy.deepcopy(
            preregistration["gate_and_authority"]
        ),
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "required_python": str(REQUIRED_PYTHON_V29H),
        "fresh_exclusive_paths": {
            "attempt": str(OUTPUT_DIRECTORY_V29H / ATTEMPT_NAME_V29H),
            "run_directory": str(OUTPUT_DIRECTORY_V29H / EXPERIMENT_NAME_V29H),
            "report_name": REPORT_NAME_V29H,
            "per_group_empty_directories_are_fresh_temporary_and_removed": True,
        },
        "model_update_training_checkpoint_evaluation_or_dataset_action_allowed": False,
    })


def validate_runtime_v29h(args, preregistration, implementation, recipe):
    prereg.validate_preregistration_v29h(preregistration)
    conflicts = {
        key: os.environ.get(key) for key in MOE_ENVIRONMENT_V29H
        if os.environ.get(key) not in (None, "", "0")
    }
    if conflicts:
        raise ValueError("V29H parent MoE environment must be empty")
    if recipe.get("content_sha256_before_self_field") != canonical_sha256(
        _without_self(recipe)
    ):
        raise RuntimeError("V29H recipe identity changed")
    for expected, actual, label in (
        (args.expected_implementation_bundle_sha256,
         implementation["bundle_sha256"], "implementation"),
        (args.expected_recipe_sha256,
         recipe["content_sha256_before_self_field"], "recipe"),
    ):
        if not args.v29h_dry_run and expected is None:
            raise ValueError(f"V29H real launch requires expected {label} hash")
        if expected is not None and expected != actual:
            raise ValueError(f"V29H {label} hash changed")
    if not args.v29h_dry_run:
        if args.expected_source_commit is None:
            raise ValueError("V29H real launch requires expected source commit")
        if Path(sys.executable).absolute() != REQUIRED_PYTHON_V29H:
            raise RuntimeError(
                "V29H real launch requires ./es-at-scale/.venv/bin/python"
            )
        return certify_committed_source_v29h(
            implementation, args.expected_source_commit,
        )
    return None


def synthetic_requests_v29h(pair_index):
    if pair_index not in range(prereg.PAIR_COUNT_V29H):
        raise ValueError("V29H pair index changed")
    seed = prereg.PAIR_SEEDS_V29H[pair_index]
    requests = {}
    commitments = {}
    for gpu_id in prereg.PHYSICAL_GPU_IDS_V29H:
        length = prereg.PROMPT_TOKENS_BY_GPU_V29H[gpu_id]
        token_ids = [
            200 + ((position * 131 + seed * 17 + gpu_id * 53) % 1000)
            for position in range(length)
        ]
        requests[gpu_id] = {"prompt_token_ids": token_ids}
        commitments[str(gpu_id)] = {
            "length": length,
            "token_identity_sha256": canonical_sha256(token_ids),
        }
    return requests, _seal({
        "schema": "vllm-moe-fp8-synthetic-request-audit-v29h",
        "pair_index": pair_index,
        "synthetic_seed": seed,
        "per_gpu_commitments": commitments,
        "combined_prompt_tokens": sum(
            item["length"] for item in commitments.values()
        ),
        "dataset_tokenizer_decoding_or_semantic_content_opened": False,
        "raw_token_ids_persisted": False,
    })


def _sampling_params_v29h():
    from vllm import SamplingParams
    return SamplingParams(
        n=1, seed=43, temperature=0.0, top_p=1.0, max_tokens=1,
        logprobs=1, detokenize=False,
    )


def output_contract_v29h(output):
    candidates = getattr(output, "outputs", None)
    candidate = candidates[0] if isinstance(candidates, list) and len(candidates) == 1 else None
    ids = getattr(candidate, "token_ids", None)
    logprobs = getattr(candidate, "logprobs", None)
    cumulative = getattr(candidate, "cumulative_logprob", None)
    if (
        candidate is None or not isinstance(ids, list) or len(ids) != 1
        or isinstance(ids[0], bool) or not isinstance(ids[0], int)
        or not isinstance(logprobs, list) or len(logprobs) != 1
        or not isinstance(logprobs[0], dict) or ids[0] not in logprobs[0]
        or isinstance(cumulative, bool) or not isinstance(cumulative, (int, float))
        or not math.isfinite(float(cumulative))
    ):
        raise RuntimeError("V29H generated output geometry changed")
    selected = getattr(logprobs[0][ids[0]], "logprob", None)
    if (
        isinstance(selected, bool) or not isinstance(selected, (int, float))
        or not math.isfinite(float(selected))
    ):
        raise RuntimeError("V29H selected token logprob changed")
    return {
        "generated_token_id": ids[0],
        "selected_logprob": float(selected),
        "cumulative_logprob": float(cumulative),
        "generated_token_count": 1,
        "integer_output_shape": [1],
    }


def normalize_ray_gpu_id_v29h(value):
    return runtime_v29e.normalize_ray_gpu_id_v29e(value)


class FullModelGroupV29H:
    def __init__(self, arm, pair_index, empty_folder):
        if arm not in {"default_empty", "v29_selected_tuned"}:
            raise ValueError("V29H arm changed")
        self.arm = arm
        self.pair_index = pair_index
        self.empty_folder = Path(empty_folder).resolve()
        self.folder = (
            self.empty_folder if arm == "default_empty"
            else prereg.TABLE_PATH_V29H.parent.resolve()
        )
        self.engines = []
        self.groups = []
        self.ray = None
        self.identities = []

    def launch(self):
        if not self.empty_folder.is_dir() or any(self.empty_folder.iterdir()):
            raise RuntimeError("V29H default config folder is not fresh and empty")
        import ray
        from es_at_scale.trainer.es_trainer import ESNcclLLM
        from ray.util.placement_group import placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        if ray.is_initialized():
            raise RuntimeError("V29H requires fresh Ray runtime per group")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        for key in ("RAY_ADDRESS", "RAY_HEAD_IP", "RAY_GCS_SERVER_ADDRESS"):
            os.environ.pop(key, None)
        ray.init(address="local", include_dashboard=False)

        class PlacementProbeV29H:
            @staticmethod
            def identity():
                import pynvml
                ids = ray.get_gpu_ids()
                if not isinstance(ids, list) or len(ids) != 1:
                    raise RuntimeError("V29H placement probe allocation changed")
                physical = normalize_ray_gpu_id_v29h(ids[0])
                visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                if visible != str(physical):
                    raise RuntimeError("V29H placement visible GPU changed")
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
                    return {
                        "physical_gpu_id": physical,
                        "ray_gpu_id_raw": ids[0],
                        "cuda_visible_devices": visible,
                        "nvml_uuid": str(uuid),
                        "pci_bus_id": str(pci),
                        "total_bytes": int(memory.total),
                    }
                finally:
                    pynvml.nvmlShutdown()

        class ProfiledFP8LLMV29H(ESNcclLLM):
            def __init__(self, *args, expected_arm, expected_folder, **kwargs):
                self._v29h_arm = str(expected_arm)
                self._v29h_folder = str(expected_folder)
                super().__init__(*args, **kwargs)

            @staticmethod
            def _physical():
                ids = ray.get_gpu_ids()
                if not isinstance(ids, list) or len(ids) != 1:
                    raise RuntimeError("V29H actor GPU allocation changed")
                physical = normalize_ray_gpu_id_v29h(ids[0])
                if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
                    raise RuntimeError("V29H actor visible GPU changed")
                return physical, ids[0]

            def runtime_identity_v29h(self):
                import pynvml
                import torch
                import vllm.envs as vllm_envs
                import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe

                physical, raw = self._physical()
                actual_folder = os.environ.get("VLLM_TUNED_CONFIG_FOLDER")
                if (
                    actual_folder != self._v29h_folder
                    or vllm_envs.VLLM_TUNED_CONFIG_FOLDER != self._v29h_folder
                    or torch.cuda.device_count() != 1
                    or torch.cuda.current_device() != 0
                ):
                    raise RuntimeError("V29H actor runtime environment changed")
                fused_moe.get_moe_configs.cache_clear()
                configs = fused_moe.get_moe_configs(
                    256, 512, "fp8_w8a8", 128, 128,
                )
                names = tuple(sorted(item.name for item in Path(actual_folder).iterdir()))
                if self._v29h_arm == "default_empty":
                    if names or configs is not None:
                        raise RuntimeError("V29H default did not use generic FP8 fallback")
                    source = "generic_fp8_fallback_none"
                    config_sha = None
                else:
                    if (
                        names != (prereg.TABLE_PATH_V29H.name,)
                        or file_sha256(Path(actual_folder) / names[0])
                        != prereg.TABLE_FILE_SHA256_V29H
                        or not isinstance(configs, dict)
                        or canonical_sha256(configs)
                        != prereg.TABLE_LOADED_CONFIG_SHA256_V29H
                    ):
                        raise RuntimeError("V29H exact selected FP8 table was not loaded")
                    source = "exact_v29_selected_fp8_table"
                    config_sha = canonical_sha256(configs)
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
                    return {
                        "physical_gpu_id": physical,
                        "ray_gpu_id_raw": raw,
                        "cuda_visible_devices": str(physical),
                        "nvml_uuid": str(uuid),
                        "pci_bus_id": str(pci),
                        "total_bytes": int(memory.total),
                        "pid": os.getpid(),
                        "arm": self._v29h_arm,
                        "config_folder": actual_folder,
                        "config_source": source,
                        "config_content_sha256": config_sha,
                        "model_config_sha256": file_sha256(
                            prereg.MODEL_PATH_V29H / "config.json"
                        ),
                        "model_index_sha256": file_sha256(
                            prereg.MODEL_PATH_V29H / "model.safetensors.index.json"
                        ),
                    }
                finally:
                    pynvml.nvmlShutdown()

            @staticmethod
            def activity_witness_v29h(common_target_ns):
                import gc
                import torch
                left = right = output = None
                iterations = 0
                try:
                    left = torch.ones((4096, 4096), dtype=torch.bfloat16, device="cuda")
                    right = torch.ones((4096, 4096), dtype=torch.bfloat16, device="cuda")
                    output = torch.empty((4096, 4096), dtype=torch.bfloat16, device="cuda")
                    torch.mm(left, right, out=output)
                    torch.cuda.synchronize()
                    while time.time_ns() < common_target_ns:
                        remaining = (common_target_ns - time.time_ns()) / 1e9
                        time.sleep(min(0.005, max(remaining, 0.0)))
                    started = time.monotonic()
                    while time.monotonic() - started < 0.75:
                        torch.mm(left, right, out=output)
                        torch.cuda.synchronize()
                        iterations += 1
                finally:
                    del left, right, output
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                return {
                    "iteration_count": iterations,
                    "duration_requirement_passed": iterations > 0,
                    "tensors_deleted_cache_emptied_and_synchronized": True,
                }

            def generate_profiled_v29h(self, request, sampling_params):
                import threading

                import pynvml
                import torch
                physical, _raw = self._physical()
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(physical)
                stop = threading.Event()
                nvml_samples = []
                monitor_failures = []

                def monitor():
                    try:
                        while not stop.is_set():
                            nvml_samples.append(int(
                                pynvml.nvmlDeviceGetMemoryInfo(handle).used
                            ))
                            stop.wait(0.01)
                    except BaseException as error:
                        monitor_failures.append(type(error).__name__)

                torch.cuda.synchronize(0)
                torch.cuda.reset_peak_memory_stats(0)
                thread = threading.Thread(target=monitor, daemon=True)
                thread.start()
                try:
                    started = time.perf_counter_ns()
                    outputs = super().generate(
                        [request], sampling_params, use_tqdm=False
                    )
                    torch.cuda.synchronize(0)
                    elapsed = time.perf_counter_ns() - started
                finally:
                    stop.set()
                    thread.join()
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    nvml_samples.append(int(memory.used))
                    total_bytes = int(memory.total)
                    pynvml.nvmlShutdown()
                if (
                    not isinstance(outputs, list) or len(outputs) != 1
                    or elapsed <= 0 or monitor_failures or not nvml_samples
                ):
                    raise RuntimeError("V29H profiled generation changed")
                return {
                    "output": outputs[0],
                    "elapsed_ns": elapsed,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                    "peak_nvml_used_bytes": max(nvml_samples),
                    "nvml_total_bytes": total_bytes,
                }

        groups = [
            placement_group([{"GPU": 1, "CPU": 0}], strategy="PACK")
            for _ in prereg.PHYSICAL_GPU_IDS_V29H
        ]
        self.groups = groups
        self.ray = ray
        ray.get([group.ready() for group in groups])
        strategies = [
            PlacementGroupSchedulingStrategy(
                placement_group=group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=0,
            )
            for group in groups
        ]
        probes = [
            ray.remote(num_cpus=0, num_gpus=1, scheduling_strategy=strategy)(
                PlacementProbeV29H
            ).remote()
            for strategy in strategies
        ]
        identities = ray.get([probe.identity.remote() for probe in probes])
        strategies_by_gpu = {}
        for probe, strategy, identity in zip(probes, strategies, identities, strict=True):
            gpu = identity["physical_gpu_id"]
            expected = v29b.GPU_IDENTITIES_V29B[gpu]
            if (
                gpu in strategies_by_gpu
                or identity["nvml_uuid"] != expected["uuid"]
                or identity["pci_bus_id"].upper() != expected["pci_bus_id"].upper()
                or identity["total_bytes"] != expected["total_bytes"]
            ):
                raise RuntimeError("V29H placement probe identity changed")
            strategies_by_gpu[gpu] = strategy
            ray.kill(probe)
        if set(strategies_by_gpu) != set(prereg.PHYSICAL_GPU_IDS_V29H):
            raise RuntimeError("V29H placement probes did not cover GPUs 0..3")
        engines = []
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V29H:
            engines.append(ray.remote(
                num_cpus=0, num_gpus=1,
                scheduling_strategy=strategies_by_gpu[gpu_id],
            )(ProfiledFP8LLMV29H).options(runtime_env={
                "env_vars": {"VLLM_TUNED_CONFIG_FOLDER": str(self.folder)},
            }).remote(
                model=str(prereg.MODEL_PATH_V29H),
                expected_arm=self.arm,
                expected_folder=str(self.folder),
                tensor_parallel_size=1,
                worker_extension_cls=(
                    "es_at_scale.utils.worker_extension.WorkerExtension"
                ),
                dtype="bfloat16",
                enable_prefix_caching=False,
                enforce_eager=True,
                gpu_memory_utilization=0.82,
                max_model_len=4096,
                limit_mm_per_prompt={"image": 0, "video": 0},
                mm_processor_cache_gb=0,
                skip_mm_profiling=True,
                moe_backend="triton",
            ))
        self.engines = engines
        self.identities = ray.get([
            engine.runtime_identity_v29h.remote() for engine in engines
        ])

    def close(self):
        if self.ray is None:
            return
        from ray.util.placement_group import remove_placement_group
        for engine in self.engines:
            try:
                self.ray.kill(engine)
            except BaseException:
                pass
        for group in self.groups:
            try:
                remove_placement_group(group)
            except BaseException:
                pass
        self.ray.shutdown()

    def validate_identities(self):
        pids = {}
        compact = []
        expected_source = (
            "generic_fp8_fallback_none"
            if self.arm == "default_empty" else "exact_v29_selected_fp8_table"
        )
        expected_sha = (
            None if self.arm == "default_empty"
            else prereg.TABLE_LOADED_CONFIG_SHA256_V29H
        )
        for gpu_id, identity in enumerate(self.identities):
            expected = v29b.GPU_IDENTITIES_V29B[gpu_id]
            if (
                identity.get("physical_gpu_id") != gpu_id
                or normalize_ray_gpu_id_v29h(identity.get("ray_gpu_id_raw")) != gpu_id
                or identity.get("cuda_visible_devices") != str(gpu_id)
                or identity.get("nvml_uuid") != expected["uuid"]
                or str(identity.get("pci_bus_id", "")).upper()
                != expected["pci_bus_id"].upper()
                or identity.get("total_bytes") != expected["total_bytes"]
                or identity.get("arm") != self.arm
                or identity.get("config_folder") != str(self.folder)
                or identity.get("config_source") != expected_source
                or identity.get("config_content_sha256") != expected_sha
                or identity.get("model_config_sha256")
                != prereg.MODEL_CONFIG_SHA256_V29H
                or identity.get("model_index_sha256")
                != prereg.MODEL_INDEX_SHA256_V29H
                or isinstance(identity.get("pid"), bool)
                or not isinstance(identity.get("pid"), int)
                or identity["pid"] <= 0
            ):
                raise RuntimeError("V29H actor/model/config identity changed")
            pids[gpu_id] = identity["pid"]
            compact.append({
                "physical_gpu_id": gpu_id,
                "nvml_uuid": identity["nvml_uuid"],
                "pci_bus_id": identity["pci_bus_id"],
                "config_source": identity["config_source"],
                "config_content_sha256": identity["config_content_sha256"],
            })
        if len(set(pids.values())) != 4:
            raise RuntimeError("V29H requires four distinct actor processes")
        return pids, canonical_sha256(compact)

    def activity_witness(self, pids_by_gpu):
        common_target = time.time_ns() + 5_000_000_000
        futures = [
            engine.activity_witness_v29h.remote(common_target)
            for engine in self.engines
        ]
        activity = monitor_futures_v29h(self.ray, futures, pids_by_gpu)
        results = self.ray.get(futures)
        if any(
            item.get("duration_requirement_passed") is not True
            or item.get("tensors_deleted_cache_emptied_and_synchronized") is not True
            for item in results
        ):
            raise RuntimeError("V29H common-start activity witness changed")
        activity["witness_result_commitment_sha256"] = canonical_sha256(results)
        return activity

    def _generate_unprofiled(self, requests):
        sampling = _sampling_params_v29h()
        batches = self.ray.get([
            self.engines[gpu].generate.remote(
                [requests[gpu]], sampling, use_tqdm=False,
            )
            for gpu in prereg.PHYSICAL_GPU_IDS_V29H
        ])
        if any(not isinstance(batch, list) or len(batch) != 1 for batch in batches):
            raise RuntimeError("V29H unprofiled generation coverage changed")
        return [output_contract_v29h(batch[0]) for batch in batches]

    def _generate_profiled(self, requests):
        sampling = _sampling_params_v29h()
        reports = self.ray.get([
            self.engines[gpu].generate_profiled_v29h.remote(
                requests[gpu], sampling,
            )
            for gpu in prereg.PHYSICAL_GPU_IDS_V29H
        ])
        contracts = [output_contract_v29h(item["output"]) for item in reports]
        for item in reports:
            if (
                item["elapsed_ns"] <= 0 or item["peak_allocated_bytes"] <= 0
                or item["peak_reserved_bytes"] <= 0
                or item["peak_nvml_used_bytes"] <= 0
                or item["nvml_total_bytes"] <= 0
                or item["peak_nvml_used_bytes"] > item["nvml_total_bytes"]
            ):
                raise RuntimeError("V29H timing or peak memory changed")
        return contracts, reports

    def run(self, requests):
        pids, identity_commitment = self.validate_identities()
        activity = self.activity_witness(pids)
        warmup = self._generate_unprofiled(requests)
        warmup_commitment = canonical_sha256(warmup)
        references = self._generate_unprofiled(requests)
        repeat = self._generate_unprofiled(requests)
        if references != repeat:
            raise RuntimeError("V29H within-arm deterministic output changed")
        elapsed = np.empty((prereg.TIMING_CALLS_PER_ENGINE_V29H, 4), dtype=np.int64)
        allocated = np.empty_like(elapsed)
        reserved = np.empty_like(elapsed)
        nvml_fraction = np.empty_like(elapsed, dtype=np.float64)
        timed_commitments = []
        for call_index in range(prereg.TIMING_CALLS_PER_ENGINE_V29H):
            contracts, reports = self._generate_profiled(requests)
            if contracts != references:
                raise RuntimeError("V29H timed output differs from exact reference")
            timed_commitments.append(canonical_sha256(contracts))
            for gpu_id, report in enumerate(reports):
                elapsed[call_index, gpu_id] = report["elapsed_ns"]
                allocated[call_index, gpu_id] = report["peak_allocated_bytes"]
                reserved[call_index, gpu_id] = report["peak_reserved_bytes"]
                nvml_fraction[call_index, gpu_id] = (
                    report["peak_nvml_used_bytes"] / report["nvml_total_bytes"]
                )
        return {
            "arm": self.arm,
            "pair_index": self.pair_index,
            "elapsed_ns": elapsed,
            "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved,
            "peak_nvml_fraction": nvml_fraction,
            "reference_contracts": references,
            "reference_commitment_sha256": canonical_sha256(references),
            "warmup_commitment_sha256": warmup_commitment,
            "timed_commitments_sha256": canonical_sha256(timed_commitments),
            "actor_identity_commitment_sha256": identity_commitment,
            "activity": activity,
            "all_four_engines_generated_every_call": True,
        }


def monitor_futures_v29h(ray, futures, pids_by_gpu):
    import pynvml
    pending = list(futures)
    simultaneous = 0
    samples = 0
    peak_fraction = 0.0
    assigned_seen = {gpu: False for gpu in prereg.PHYSICAL_GPU_IDS_V29H}
    pynvml.nvmlInit()
    try:
        handles = {
            gpu: pynvml.nvmlDeviceGetHandleByIndex(gpu)
            for gpu in prereg.PHYSICAL_GPU_IDS_V29H
        }
        while pending:
            all_active = True
            for gpu, handle in handles.items():
                processes = []
                for name in (
                    "nvmlDeviceGetComputeRunningProcesses",
                    "nvmlDeviceGetGraphicsRunningProcesses",
                ):
                    function = getattr(pynvml, name, None)
                    if function is not None:
                        try:
                            processes.extend(function(handle))
                        except pynvml.NVMLError_NotSupported:
                            pass
                present = pids_by_gpu[gpu] in {int(item.pid) for item in processes}
                assigned_seen[gpu] |= present
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                peak_fraction = max(peak_fraction, memory.used / memory.total)
                all_active &= present and int(utilization.gpu) > 0
            samples += 1
            simultaneous += int(all_active)
            ready, pending = ray.wait(pending, num_returns=1, timeout=0.05)
            if ready:
                ray.get(ready)
    finally:
        pynvml.nvmlShutdown()
    if simultaneous < 1 or not all(assigned_seen.values()):
        raise RuntimeError("V29H did not observe all four assigned active actor PIDs")
    value = {
        "all_four_assigned_pids_and_positive_utilization_simultaneously": True,
        "sample_count": samples,
        "simultaneous_positive_sample_count": simultaneous,
        "peak_nvml_fraction": float(peak_fraction),
    }
    value["commitment_sha256"] = canonical_sha256(value)
    return value


def exact_pair_equivalence_v29h(default, tuned):
    components = {
        "reference_contracts_exact": (
            default["reference_contracts"] == tuned["reference_contracts"]
        ),
        "reference_commitment_exact": (
            default["reference_commitment_sha256"]
            == tuned["reference_commitment_sha256"]
        ),
        "timed_commitments_exact": (
            default["timed_commitments_sha256"]
            == tuned["timed_commitments_sha256"]
        ),
        "all_four_engines_generated_every_call": (
            default["all_four_engines_generated_every_call"] is True
            and tuned["all_four_engines_generated_every_call"] is True
        ),
    }
    return {**components, "pass": all(components.values())}


def _hierarchical_replicates_v29h(values, pair_draws, call_draws):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (
        prereg.PAIR_COUNT_V29H, prereg.TIMING_CALLS_PER_ENGINE_V29H,
    ) or not np.all(np.isfinite(values)):
        raise RuntimeError("V29H paired endpoint matrix changed")
    sampled = values[
        pair_draws[:, :, None], call_draws[:, None, :]
    ]
    return np.median(sampled, axis=(1, 2))


def performance_summary_v29h(pair_results):
    if tuple(pair_results) != prereg.PAIR_ORDER_V29H:
        raise RuntimeError("V29H pair result order changed")
    latency = np.empty((8, 7, 4), dtype=np.float64)
    vram = np.empty_like(latency)
    activity_peak = []
    order_global = {"default_first": [], "tuned_first": []}
    for pair_index, pair in enumerate(prereg.PAIR_ORDER_V29H):
        default = pair_results[pair]["arms"]["default_empty"]
        tuned = pair_results[pair]["arms"]["v29_selected_tuned"]
        latency[pair_index] = default["elapsed_ns"] / tuned["elapsed_ns"]
        vram[pair_index] = np.maximum(
            tuned["peak_allocated_bytes"] / default["peak_allocated_bytes"],
            tuned["peak_reserved_bytes"] / default["peak_reserved_bytes"],
        )
        activity_peak.extend([
            float(np.max(default["peak_nvml_fraction"])),
            float(np.max(tuned["peak_nvml_fraction"])),
        ])
        order = (
            "default_first"
            if prereg.ARM_ORDER_BY_PAIR_V29H[pair][0] == "default_empty"
            else "tuned_first"
        )
        order_global[order].append(float(np.exp(np.mean(np.log(
            np.median(latency[pair_index], axis=0)
        )))))
    pair_draws, call_draws, draw_sha = prereg.bootstrap_draw_plan_v29h()
    lower_q = prereg.PER_ENDPOINT_ALPHA_V29H
    upper_q = 1.0 - lower_q
    latency_endpoints = {}
    memory_endpoints = {}
    latency_replicates = []
    memory_replicates = []
    for gpu in prereg.PHYSICAL_GPU_IDS_V29H:
        latency_rep = _hierarchical_replicates_v29h(
            latency[:, :, gpu], pair_draws, call_draws,
        )
        memory_rep = _hierarchical_replicates_v29h(
            vram[:, :, gpu], pair_draws, call_draws,
        )
        latency_replicates.append(latency_rep)
        memory_replicates.append(memory_rep)
        point_latency = float(np.median(latency[:, :, gpu]))
        lcb = float(np.quantile(latency_rep, lower_q, method="linear"))
        point_memory = float(np.median(vram[:, :, gpu]))
        ucb = float(np.quantile(memory_rep, upper_q, method="linear"))
        latency_endpoints[str(gpu)] = {
            "prompt_tokens": prereg.PROMPT_TOKENS_BY_GPU_V29H[gpu],
            "median_default_over_tuned_ratio": point_latency,
            "familywise_lower_confidence_bound": lcb,
            "point_threshold": prereg.PER_GPU_LATENCY_POINT_MIN_V29H,
            "lower_bound_threshold": prereg.PER_GPU_LATENCY_LCB_MIN_V29H,
            "pass": bool(
                point_latency >= prereg.PER_GPU_LATENCY_POINT_MIN_V29H
                and lcb >= prereg.PER_GPU_LATENCY_LCB_MIN_V29H
            ),
        }
        memory_endpoints[str(gpu)] = {
            "median_tuned_over_default_ratio": point_memory,
            "familywise_upper_confidence_bound": ucb,
            "point_threshold": prereg.VRAM_POINT_RATIO_MAX_V29H,
            "upper_bound_threshold": prereg.VRAM_UCB_RATIO_MAX_V29H,
            "pass": bool(
                point_memory <= prereg.VRAM_POINT_RATIO_MAX_V29H
                and ucb <= prereg.VRAM_UCB_RATIO_MAX_V29H
            ),
        }
    latency_replicates = np.stack(latency_replicates, axis=1)
    memory_replicates = np.stack(memory_replicates, axis=1)
    per_gpu_latency_point = np.array([
        latency_endpoints[str(gpu)]["median_default_over_tuned_ratio"]
        for gpu in prereg.PHYSICAL_GPU_IDS_V29H
    ])
    per_gpu_memory_point = np.array([
        memory_endpoints[str(gpu)]["median_tuned_over_default_ratio"]
        for gpu in prereg.PHYSICAL_GPU_IDS_V29H
    ])
    global_latency_point = float(np.exp(np.mean(np.log(per_gpu_latency_point))))
    global_latency_rep = np.exp(np.mean(np.log(latency_replicates), axis=1))
    global_latency_lcb = float(np.quantile(
        global_latency_rep, lower_q, method="linear",
    ))
    global_memory_point = float(np.max(per_gpu_memory_point))
    global_memory_rep = np.max(memory_replicates, axis=1)
    global_memory_ucb = float(np.quantile(
        global_memory_rep, upper_q, method="linear",
    ))
    global_endpoints = {
        "full_model_latency": {
            "geometric_mean_speedup": global_latency_point,
            "familywise_lower_confidence_bound": global_latency_lcb,
            "point_threshold": prereg.GLOBAL_LATENCY_POINT_MIN_V29H,
            "lower_bound_threshold": prereg.GLOBAL_LATENCY_LCB_MIN_V29H,
            "pass": bool(
                global_latency_point >= prereg.GLOBAL_LATENCY_POINT_MIN_V29H
                and global_latency_lcb >= prereg.GLOBAL_LATENCY_LCB_MIN_V29H
            ),
        },
        "peak_vram": {
            "max_per_gpu_median_ratio": global_memory_point,
            "familywise_upper_confidence_bound": global_memory_ucb,
            "point_threshold": prereg.VRAM_POINT_RATIO_MAX_V29H,
            "upper_bound_threshold": prereg.VRAM_UCB_RATIO_MAX_V29H,
            "pass": bool(
                global_memory_point <= prereg.VRAM_POINT_RATIO_MAX_V29H
                and global_memory_ucb <= prereg.VRAM_UCB_RATIO_MAX_V29H
            ),
        },
    }
    return {
        "schema": "vllm-moe-fp8-full-model-performance-summary-v29h",
        "pair_count": 8,
        "matched_timing_calls_per_pair": 7,
        "latency_by_physical_gpu": latency_endpoints,
        "peak_vram_by_physical_gpu": memory_endpoints,
        "global": global_endpoints,
        "all_ten_performance_endpoints_passed": bool(
            all(item["pass"] for item in latency_endpoints.values())
            and all(item["pass"] for item in memory_endpoints.values())
            and all(item["pass"] for item in global_endpoints.values())
        ),
        "maximum_absolute_nvml_fraction": float(max(activity_peak)),
        "absolute_nvml_gate_passed": bool(
            max(activity_peak) <= prereg.MAX_ABSOLUTE_NVML_FRACTION_V29H
        ),
        "descriptive_arm_order_global_speed_medians": {
            key: float(np.median(values)) for key, values in order_global.items()
        },
        "bootstrap_draw_plan_sha256": draw_sha,
        "raw_timings_memory_or_bootstrap_replicates_persisted": False,
    }


def gate_v29h(equivalence, performance, integrity):
    passed = bool(
        equivalence["all_eight_pairs_exact"]
        and performance["all_ten_performance_endpoints_passed"]
        and performance["absolute_nvml_gate_passed"]
        and integrity["all_runtime_integrity_gates_passed"]
    )
    return _seal({
        "schema": "vllm-moe-fp8-full-model-authorization-gate-v29h",
        "pass": passed,
        "decision": (
            "authorize_only_exact_v29_table_in_a_separately_frozen_"
            "serialized_fp8_train_only_recipe_ab"
            if passed else "retain_empty_default_serialized_fp8_runtime"
        ),
        "all_eight_pairs_exact": equivalence["all_eight_pairs_exact"],
        "all_ten_performance_endpoints_passed": performance[
            "all_ten_performance_endpoints_passed"
        ],
        "absolute_nvml_gate_passed": performance["absolute_nvml_gate_passed"],
        "all_runtime_integrity_gates_passed": integrity[
            "all_runtime_integrity_gates_passed"
        ],
        "direct_table_or_recipe_adoption_authorized": False,
        "model_update_or_training_authorized": False,
        "checkpoint_write_authorized": False,
        "dataset_promotion_authorized": False,
        "evaluation_validation_heldout_ood_or_benchmark_access_authorized": False,
        "nontrain_runtime_reuse_authorized": False,
    })


def assert_all_four_idle_v29h():
    value = runtime_v29e.assert_all_four_gpus_idle_v29e()
    value = copy.deepcopy(value)
    value["schema"] = "vllm-moe-fp8-full-model-prelaunch-idle-v29h"
    return _seal(value)


def wait_between_groups_v29h(group_count):
    value = runtime_v29e.wait_for_all_four_gpus_idle_v29e()
    value = copy.deepcopy(value)
    value["schema"] = "vllm-moe-fp8-full-model-between-group-idle-v29h"
    value["completed_group_count"] = group_count
    return _seal(value)


def run_one_group_v29h(arm, pair_index, group_index, requests):
    controller = None
    result = None
    failure = None
    idle = None
    with tempfile.TemporaryDirectory(prefix="specialist_v29h_group_") as root:
        empty = Path(root) / "fresh_empty_config"
        empty.mkdir()
        try:
            controller = FullModelGroupV29H(arm, pair_index, empty)
            controller.launch()
            result = controller.run(requests)
        except BaseException as error:
            failure = error
        finally:
            if controller is not None:
                try:
                    controller.close()
                except BaseException as cleanup_error:
                    if failure is None:
                        failure = cleanup_error
        try:
            idle = wait_between_groups_v29h(group_index + 1)
        except BaseException as idle_error:
            if failure is None:
                failure = idle_error
    if failure is not None:
        raise failure
    if result is None or idle is None:
        raise RuntimeError("V29H group result or cleanup certificate missing")
    result["cleanup_certificate_sha256"] = idle[
        "content_sha256_before_self_field"
    ]
    return result, idle


def run_counterbalanced_v29h(preregistration, prelaunch_idle):
    del preregistration
    pair_results = {}
    group_audits = []
    request_audits = []
    latest_idle = prelaunch_idle
    group_index = 0
    for pair_index, pair in enumerate(prereg.PAIR_ORDER_V29H):
        requests, request_audit = synthetic_requests_v29h(pair_index)
        request_audits.append(request_audit["content_sha256_before_self_field"])
        arms = {}
        for arm in prereg.ARM_ORDER_BY_PAIR_V29H[pair]:
            result, latest_idle = run_one_group_v29h(
                arm, pair_index, group_index, requests,
            )
            arms[arm] = result
            group_audits.append({
                "pair": pair, "arm": arm, "group_index": group_index,
                "actor_identity_commitment_sha256": result[
                    "actor_identity_commitment_sha256"
                ],
                "activity_commitment_sha256": result["activity"][
                    "commitment_sha256"
                ],
                "reference_commitment_sha256": result[
                    "reference_commitment_sha256"
                ],
                "cleanup_certificate_sha256": result[
                    "cleanup_certificate_sha256"
                ],
            })
            group_index += 1
        equivalence = exact_pair_equivalence_v29h(
            arms["default_empty"], arms["v29_selected_tuned"],
        )
        if equivalence["pass"] is not True:
            raise RuntimeError(f"V29H exact output equivalence failed: {pair}")
        pair_results[pair] = {"arms": arms, "equivalence": equivalence}
    if group_index != prereg.ENGINE_GROUP_COUNT_V29H:
        raise RuntimeError("V29H engine group count changed")
    performance = performance_summary_v29h(pair_results)
    equivalence = {
        "schema": "vllm-moe-fp8-full-model-exact-equivalence-v29h",
        "pair_count": 8,
        "exact_pair_count": sum(
            item["equivalence"]["pass"] for item in pair_results.values()
        ),
        "all_eight_pairs_exact": all(
            item["equivalence"]["pass"] for item in pair_results.values()
        ),
        "component_pass_counts": {
            key: sum(item["equivalence"][key] for item in pair_results.values())
            for key in next(iter(pair_results.values()))["equivalence"]
            if key != "pass"
        },
        "paired_output_commitment_sha256": canonical_sha256([
            item["arms"]["default_empty"]["reference_commitment_sha256"]
            for item in pair_results.values()
        ]),
    }
    activity = [
        item["arms"][arm]["activity"]
        for item in pair_results.values()
        for arm in ("default_empty", "v29_selected_tuned")
    ]
    integrity = {
        "schema": "vllm-moe-fp8-full-model-runtime-integrity-v29h",
        "fresh_four_engine_group_count": group_index,
        "serialized_fp8_tp1_model_load_count": group_index * 4,
        "all_four_activity_group_count": sum(
            item["all_four_assigned_pids_and_positive_utilization_simultaneously"]
            for item in activity
        ),
        "minimum_activity_sample_count": min(
            item["sample_count"] for item in activity
        ),
        "minimum_simultaneous_positive_sample_count": min(
            item["simultaneous_positive_sample_count"] for item in activity
        ),
        "request_audit_bundle_sha256": canonical_sha256(request_audits),
        "group_audit_bundle_sha256": canonical_sha256(group_audits),
        "final_idle_certificate_sha256": latest_idle[
            "content_sha256_before_self_field"
        ],
        "all_16_groups_activity_config_identity_and_cleanup_passed": True,
        "all_four_finally_idle": latest_idle.get("all_four_idle") is True,
        "all_runtime_integrity_gates_passed": True,
        "dataset_tokenizer_decoding_or_semantic_content_opened": False,
        "raw_tokens_outputs_logprobs_timings_memory_samples_pids_or_draws_persisted": False,
    }
    integrity["all_runtime_integrity_gates_passed"] = bool(
        integrity["fresh_four_engine_group_count"] == 16
        and integrity["serialized_fp8_tp1_model_load_count"] == 64
        and integrity["all_four_activity_group_count"] == 16
        and integrity["all_four_finally_idle"]
    )
    return equivalence, performance, integrity


def _parser_v29h():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v29h-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-recipe-sha256")
    parser.add_argument("--expected-source-commit")
    return parser


def run_exact_v29h(
    preregistration, implementation, recipe, committed_source,
):
    environment = runtime_r2.certify_runtime_environment_r2()
    cpu_disk = live_cpu_disk_audit_v29h()
    attempt_path = OUTPUT_DIRECTORY_V29H / ATTEMPT_NAME_V29H
    run_dir = OUTPUT_DIRECTORY_V29H / EXPERIMENT_NAME_V29H
    report_path = run_dir / REPORT_NAME_V29H
    if attempt_path.exists() or run_dir.exists():
        raise RuntimeError("V29H requires fresh exclusive attempt and run paths")
    prelaunch_idle = assert_all_four_idle_v29h()
    attempt = _seal({
        "schema": "vllm-moe-fp8-full-model-durable-attempt-v29h",
        "status": "launching", "phase": "before_first_full_model_group",
        "recipe_sha256": recipe["content_sha256_before_self_field"],
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "committed_source_sha256": committed_source[
            "content_sha256_before_self_field"
        ],
        "runtime_environment_sha256": environment[
            "content_sha256_before_self_field"
        ],
        "cpu_disk_audit_sha256": cpu_disk["content_sha256_before_self_field"],
        "prelaunch_idle_sha256": prelaunch_idle[
            "content_sha256_before_self_field"
        ],
        "model_update_training_checkpoint_evaluation_or_dataset_action_applied": False,
    })
    runtime_v29e._exclusive_write_json_v29e(attempt_path, attempt)
    failure = None
    equivalence = performance = integrity = None
    try:
        equivalence, performance, integrity = run_counterbalanced_v29h(
            preregistration, prelaunch_idle,
        )
    except BaseException as error:
        failure = error
    if failure is not None:
        attempt.update({
            "status": "failed", "phase": "during_or_after_full_model_group_loop",
            "failure_type": type(failure).__name__,
            "failure_sha256": canonical_sha256({
                "type": type(failure).__name__, "repr": repr(failure),
            }),
            "report_exists_after_attempt": report_path.exists(),
            "model_update_training_checkpoint_evaluation_or_dataset_action_applied": False,
        })
        attempt = _seal(attempt)
        runtime_v29e._rewrite_json_v29e(attempt_path, attempt)
        raise failure
    gate = gate_v29h(equivalence, performance, integrity)
    report = _seal({
        "schema": "vllm-moe-fp8-full-model-runtime-ab-report-v29h",
        "status": "valid_completed_synthetic_train_runtime_only_ab",
        "recipe_sha256": recipe["content_sha256_before_self_field"],
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "committed_source_sha256": committed_source[
            "content_sha256_before_self_field"
        ],
        "prelaunch_idle_sha256": prelaunch_idle[
            "content_sha256_before_self_field"
        ],
        "equivalence": equivalence,
        "performance": performance,
        "runtime_integrity": integrity,
        "gate": gate,
        "model_update_training_checkpoint_evaluation_or_dataset_action_applied": False,
        "dataset_tokenizer_decoding_or_semantic_content_opened": False,
        "raw_tokens_outputs_logprobs_timings_memory_samples_pids_or_draws_persisted": False,
    })
    _assert_compact_v29h(report)
    runtime_v29e._exclusive_write_json_v29e(report_path, report)
    attempt.update({
        "status": "complete", "phase": "after_16_groups_and_compact_report",
        "report_binding": {
            "path": str(report_path), "file_sha256": file_sha256(report_path),
            "content_sha256": report["content_sha256_before_self_field"],
        },
        "final_idle_sha256": integrity["final_idle_certificate_sha256"],
    })
    attempt = _seal(attempt)
    runtime_v29e._rewrite_json_v29e(attempt_path, attempt)
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    _assert_closed_argv_v29h(argv)
    args = _parser_v29h().parse_args(argv)
    preregistration = load_preregistration_v29h()
    implementation = implementation_identity_v29h()
    recipe = recipe_v29h(preregistration, implementation)
    committed_source = validate_runtime_v29h(
        args, preregistration, implementation, recipe,
    )
    if args.v29h_dry_run:
        payload = _seal({
            "schema": "vllm-moe-fp8-full-model-runtime-ab-dry-run-v29h",
            "implementation_bundle_sha256": implementation["bundle_sha256"],
            "recipe_sha256": recipe["content_sha256_before_self_field"],
            "preregistration_file_sha256": PREREG_FILE_SHA256_V29H,
            "preregistration_content_sha256": PREREG_CONTENT_SHA256_V29H,
            "required_python": str(REQUIRED_PYTHON_V29H),
            "pair_count": 8,
            "fresh_four_tp1_engine_group_count": 16,
            "serialized_fp8_tp1_model_load_count": 64,
            "total_generation_request_budget": 640,
            "total_prompt_token_budget": 614_400,
            "synchronized_activity_witness_count": 64,
            "pass_authority": preregistration["gate_and_authority"][
                "pass_authority"
            ],
            "fresh_exclusive_and_committed_clean_real_launch_required": True,
            "gpu_launched": False,
            "dataset_or_semantic_content_opened": False,
            "model_update_training_checkpoint_evaluation_or_dataset_action_authorized": False,
        })
        _assert_compact_v29h(payload)
        print(json.dumps(payload, sort_keys=True))
        return payload
    return run_exact_v29h(
        preregistration, implementation, recipe, committed_source,
    )


if __name__ == "__main__":
    main()
