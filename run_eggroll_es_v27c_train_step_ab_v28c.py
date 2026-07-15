#!/usr/bin/env python3
"""Fail-closed runtime for the V28C BF16 V27C ES train-step A/B."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

import build_eggroll_es_insertion_model_seal_v23a as model_seal_v23a
import eggroll_es_v27c_train_step_ab_preregistration_v28c as prereg
import run_eggroll_es_anchor_variance_v10 as driver_v10
import run_eggroll_es_insertion_stability_v23a as runtime_v23a
import run_eggroll_es_insertion_stability_v23a_retry_r2 as runtime_r2
import run_eggroll_es_v27c_runtime_ab_v28a as runtime_v28a
import train_eggroll_es_specialist as base
import train_eggroll_es_specialist_anchor_v13 as anchor_v13


ROOT = Path(__file__).resolve().parent
PREREG_PATH_V28C = prereg.OUTPUT_PATH_V28C
PREREG_FILE_SHA256_V28C = (
    "52643d58267a1debe261090e2b25c7f9e9503a7c95bc85607c464f361d9ca144"
)
PREREG_CONTENT_SHA256_V28C = (
    "8158f1b8bdd04fde43b48369434484473b0b7686e31b15ed9c2e501108d5c1fd"
)
EXPERIMENT_NAME_V28C = "s6_v28c_v27c_bf16_no_update_es_train_step_ab"
OUTPUT_DIRECTORY_V28C = (ROOT / "experiments/eggroll_es_hpo/runs").resolve()
ATTEMPT_NAME_V28C = f".{EXPERIMENT_NAME_V28C}.launch_attempt.json"
REPORT_NAME_V28C = "v27c_bf16_train_step_ab_report_v28c.json"
TEST_PATH_V28C = (ROOT / "test_run_eggroll_es_v27c_train_step_ab_v28c.py").resolve()
PREREG_TEST_PATH_V28C = (
    ROOT / "test_eggroll_es_v27c_train_step_ab_preregistration_v28c.py"
).resolve()
REQUIRED_PYTHON_V28C = (
    ROOT / "es-at-scale/.venv/bin/python"
).absolute()
VLLM_FUSED_MOE_PATH_V28C = runtime_v28a.VLLM_FUSED_MOE_PATH_V28A
VLLM_FUSED_MOE_SHA256_V28C = runtime_v28a.VLLM_FUSED_MOE_SHA256_V28A
VLLM_ENVS_PATH_V28C = runtime_v28a.VLLM_ENVS_PATH_V28A
VLLM_ENVS_SHA256_V28C = runtime_v28a.VLLM_ENVS_SHA256_V28A

IMPLEMENTATION_PATHS_V28C = {
    "v28c_preregistration_module": Path(prereg.__file__).resolve(),
    "v28c_preregistration_tests": PREREG_TEST_PATH_V28C,
    "v28c_preregistration": PREREG_PATH_V28C,
    "v28c_runtime": Path(__file__).resolve(),
    "v28c_runtime_tests": TEST_PATH_V28C,
    "v13_trainer": Path(anchor_v13.__file__).resolve(),
    "v13_worker": ROOT / "eggroll_es_worker_v13.py",
    "v13_panel_manifest": prereg.PANEL_MANIFEST_V28C,
    "v28b_positive_evidence": prereg.POSITIVE_EVIDENCE_PATH_V28C,
    "v27c_tuned_table": prereg.TUNED_TABLE_PATH_V28C,
    "v23a_model_seal": prereg.MODEL_SEAL_PATH_V28C,
}
MOE_CONFOUNDING_ENV_V28C = (
    "VLLM_TUNED_CONFIG_FOLDER", "VLLM_ROCM_USE_AITER",
    "VLLM_ROCM_USE_AITER_MOE", "VLLM_USE_DEEP_GEMM",
    "VLLM_MOE_USE_DEEP_GEMM",
)
FORBIDDEN_ARGV_TOKENS_V28C = (
    "checkpoint", "update", "heldout", "validation", "ood", "benchmark",
    "eval", "save-model", "train-dataset", "promotion", "fp8",
)
FORBIDDEN_PERSISTED_KEYS_V28C = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "prompt_token_ids", "token_ids", "text", "texts", "responses",
    "coefficients", "weighted_central_response", "unweighted_central_response",
    "elapsed_ns", "timing_vectors", "raw_timings", "memory_samples",
    "raw_memory", "pid", "pids", "bootstrap_draws", "bootstrap_replicates",
    "diagnostic", "row_content", "row_sha256", "document_sha256",
}
ALLOWED_UNTRACKED_PREFIXES_V28C = ("experiments/dataset_probes/",)
ALLOWED_UNTRACKED_PATHS_V28C = frozenset({
    "experiments/gpu_utilization_v31_base_qwen36_train_reward_probe.jsonl",
})
CURATOR_UNTRACKED_PATTERN_V28C = re.compile(
    r"data/manual_reviews/context_merit_audit_v([0-9]+)/.+\Z"
)
WORKTREE_ALLOWLIST_CONTRACT_V28C = {
    "curator_snapshot_directories": (
        "data/manual_reviews/context_merit_audit_vN/ where N >= 390"
    ),
    "path_prefixes": list(ALLOWED_UNTRACKED_PREFIXES_V28C),
    "exact_paths": sorted(ALLOWED_UNTRACKED_PATHS_V28C),
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


def _assert_compact_v28c(value):
    overlap = FORBIDDEN_PERSISTED_KEYS_V28C & set(_recursive_keys(value))
    if overlap:
        raise RuntimeError(
            f"V28C compact output contains forbidden keys: {sorted(overlap)}"
        )


def _assert_train_only_argv_v28c(argv):
    for token in argv:
        lowered = str(token).lower()
        if any(item in lowered for item in FORBIDDEN_ARGV_TOKENS_V28C):
            raise ValueError(f"V28C rejects forbidden runtime surface: {token}")


def normalize_ray_gpu_id_v28c(value):
    if isinstance(value, bool):
        raise RuntimeError("V28C Ray GPU ID representation changed")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value in {"0", "1", "2", "3"}:
        result = int(value)
    else:
        raise RuntimeError("V28C Ray GPU ID representation changed")
    if result not in prereg.PHYSICAL_GPU_IDS_V28C:
        raise RuntimeError("V28C Ray GPU ID physical range changed")
    return result


def load_preregistration_v28c():
    value = json.loads(PREREG_PATH_V28C.read_text(encoding="utf-8"))
    if (
        file_sha256(PREREG_PATH_V28C) != PREREG_FILE_SHA256_V28C
        or value.get("content_sha256_before_self_field")
        != PREREG_CONTENT_SHA256_V28C
        or canonical_sha256(_without_self(value)) != PREREG_CONTENT_SHA256_V28C
    ):
        raise RuntimeError("V28C frozen preregistration identity changed")
    return prereg.validate_preregistration_v28c(value)


def implementation_identity_v28c():
    files = {
        key: {
            "path": str(Path(path).resolve()),
            "file_sha256": file_sha256(path),
        }
        for key, path in IMPLEMENTATION_PATHS_V28C.items()
    }
    return {
        "files": files,
        "overlay_bundle_sha256": canonical_sha256(files),
        "bundle_sha256": canonical_sha256(files),
    }


def _is_allowed_untracked_v28c(relative):
    if relative in ALLOWED_UNTRACKED_PATHS_V28C or any(
        relative.startswith(prefix) for prefix in ALLOWED_UNTRACKED_PREFIXES_V28C
    ):
        return True
    match = CURATOR_UNTRACKED_PATTERN_V28C.fullmatch(relative)
    return match is not None and int(match.group(1)) >= 390


def validate_worktree_status_v28c(raw_status):
    if not isinstance(raw_status, str):
        raise TypeError("V28C worktree status must be text")
    allowed_untracked = []
    rejected = []
    for line in raw_status.splitlines():
        if len(line) < 4 or line[2] != " ":
            raise RuntimeError("V28C worktree status record changed")
        status, relative = line[:2], line[3:]
        if status == "??" and _is_allowed_untracked_v28c(relative):
            allowed_untracked.append(relative)
        else:
            rejected.append({"status": status, "relative_path": relative})
    if rejected:
        raise RuntimeError(
            "V28C real launch requires committed-clean source outside the exact "
            "untracked allowlist"
        )
    return {
        "all_tracked_files_clean": True,
        "only_explicitly_allowlisted_untracked_paths_present": True,
        "allowed_untracked_entry_count": len(allowed_untracked),
        "allowed_untracked_entries_sha256": canonical_sha256(
            sorted(allowed_untracked)
        ),
        "allowlist_contract_sha256": canonical_sha256(
            WORKTREE_ALLOWLIST_CONTRACT_V28C
        ),
    }


def certify_real_launch_committed_source_v28c(
    implementation, expected_source_commit,
):
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    if len(head) != 40 or head != expected_source_commit:
        raise RuntimeError("V28C exact expected source commit changed")
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
                f"V28C real launch requires committed implementation: {relative}"
            ) from error
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item["file_sha256"]:
            raise RuntimeError(f"V28C source differs from HEAD: {relative}")
        committed[key] = digest
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT, text=True,
    )
    worktree = validate_worktree_status_v28c(status)
    return _seal({
        "schema": "eggroll-es-v28c-committed-clean-source-certificate",
        "git_head": head,
        "implementation_file_count": len(committed),
        "committed_implementation_sha256": canonical_sha256(committed),
        **worktree,
    })


def load_layer_bundle_v28c():
    plan_sha = driver_v10.MIDDLE_LATE_PLAN_SHA256_V10
    plan = anchor_v13.anchor_v11.FROZEN_STABILITY_PLANS_V11[plan_sha]
    return anchor_v13.load_frozen_layer_plan_v13(
        plan["path"],
        expected_file_sha256=plan["file_sha256"],
        expected_plan_sha256=plan_sha,
        expected_model_config_sha256=prereg.MODEL_CONFIG_SHA256_V28C,
    )


def live_cpu_disk_audit_v28c():
    table, evidence, base_seal = prereg.validate_static_inputs_v28c()
    if (
        file_sha256(VLLM_FUSED_MOE_PATH_V28C) != VLLM_FUSED_MOE_SHA256_V28C
        or file_sha256(VLLM_ENVS_PATH_V28C) != VLLM_ENVS_SHA256_V28C
    ):
        raise RuntimeError("V28C vLLM MoE loader dependency changed")
    all_files = model_seal_v23a._file_fingerprints(prereg.MODEL_PATH_V28C)
    if canonical_sha256(all_files) != (
        prereg.MODEL_BASE_ALL_FILES_FINGERPRINT_SHA256_V28C
    ):
        raise RuntimeError("V28C exact Qwen3.6 all-files fingerprint changed")
    shard_hashes = {
        name: all_files[name] for name in sorted(base_seal["shards"])
    }
    if shard_hashes != base_seal["shards"]:
        raise RuntimeError("V28C exact Qwen3.6 shard bytes changed")
    return _seal({
        "schema": "eggroll-es-v28c-live-cpu-disk-audit",
        "model_all_files_fingerprint_sha256": canonical_sha256(all_files),
        "model_shard_hash_audit_sha256": canonical_sha256(shard_hashes),
        "all_26_model_shard_bytes_and_sha256_match_committed_seal": True,
        "table_file_sha256": prereg.TUNED_TABLE_FILE_SHA256_V28C,
        "table_content_sha256": canonical_sha256(table),
        "positive_evidence_content_sha256": evidence[
            "content_sha256_before_self_field"
        ],
        "vllm_fused_moe_file_sha256": VLLM_FUSED_MOE_SHA256_V28C,
        "vllm_envs_file_sha256": VLLM_ENVS_SHA256_V28C,
        "nontrain_or_dataset_content_opened": False,
    })


def recipe_v28c(preregistration, layer_bundle, implementation):
    value = {
        "schema": "eggroll-es-v27c-bf16-train-step-ab-recipe-v28c",
        "experiment_name": EXPERIMENT_NAME_V28C,
        "preregistration": {
            "path": str(PREREG_PATH_V28C),
            "file_sha256": PREREG_FILE_SHA256_V28C,
            "content_sha256": PREREG_CONTENT_SHA256_V28C,
        },
        "authorization_basis": copy.deepcopy(
            preregistration["authorization_basis"]
        ),
        "model_contract": copy.deepcopy(preregistration["model_contract"]),
        "table_contract": copy.deepcopy(preregistration["table_contract"]),
        "arms": copy.deepcopy(preregistration["arms"]),
        "paired_counterbalanced_design": copy.deepcopy(
            preregistration["paired_counterbalanced_design"]
        ),
        "frozen_train_step_contract": copy.deepcopy(
            preregistration["frozen_train_step_contract"]
        ),
        "request_budget": copy.deepcopy(preregistration["request_budget"]),
        "runtime_contract": copy.deepcopy(preregistration["runtime_contract"]),
        "exact_equivalence_contract": copy.deepcopy(
            preregistration["exact_equivalence_contract"]
        ),
        "performance_analysis": copy.deepcopy(
            preregistration["performance_analysis"]
        ),
        "gate_and_authority": copy.deepcopy(
            preregistration["gate_and_authority"]
        ),
        "layer_plan": {
            key: layer_bundle[key] for key in (
                "path", "file_sha256", "plan_sha256", "model_config_sha256"
            )
        },
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "required_python": str(REQUIRED_PYTHON_V28C),
        "fresh_exclusive_paths": {
            "attempt": str(OUTPUT_DIRECTORY_V28C / ATTEMPT_NAME_V28C),
            "run_directory": str(OUTPUT_DIRECTORY_V28C / EXPERIMENT_NAME_V28C),
            "report_name": REPORT_NAME_V28C,
            "per_group_temporary_directories_removed_after_cleanup": True,
        },
        "model_update_allowed": False,
        "checkpoint_write_allowed": False,
        "evaluation_allowed": False,
        "dataset_promotion_allowed": False,
        "fp8_allowed": False,
    }
    return _seal(value)


def validate_runtime_v28c(args, preregistration, implementation, recipe):
    prereg.validate_preregistration_v28c(preregistration)
    conflicts = {
        key: os.environ.get(key) for key in MOE_CONFOUNDING_ENV_V28C
        if os.environ.get(key) not in (None, "")
    }
    if conflicts:
        raise ValueError("V28C parent requires all MoE override environment empty")
    if os.environ.get("VLLM_BATCH_INVARIANT") not in (None, "0"):
        raise ValueError("V28C rejects the vLLM batch-invariant backend swap")
    if recipe.get("content_sha256_before_self_field") != canonical_sha256(
        _without_self(recipe)
    ):
        raise ValueError("V28C runtime recipe identity changed")
    expected_values = (
        (args.expected_implementation_bundle_sha256,
         implementation["bundle_sha256"], "implementation"),
        (args.expected_recipe_sha256,
         recipe["content_sha256_before_self_field"], "recipe"),
    )
    for expected, actual, label in expected_values:
        if not args.v28c_dry_run and expected is None:
            raise ValueError(f"V28C real launch requires expected {label} hash")
        if expected is not None and expected != actual:
            raise ValueError(f"V28C {label} hash changed")
    if not args.v28c_dry_run:
        if args.expected_source_commit is None:
            raise ValueError("V28C real launch requires expected source commit")
        if Path(sys.executable).absolute() != REQUIRED_PYTHON_V28C:
            raise RuntimeError(
                "V28C real launch requires ./es-at-scale/.venv/bin/python"
            )
        return certify_real_launch_committed_source_v28c(
            implementation, args.expected_source_commit,
        )
    return None


def _actor_config_expected_sha_v28c():
    table = json.loads(prereg.TUNED_TABLE_PATH_V28C.read_text(encoding="utf-8"))
    table.pop("triton_version")
    return canonical_sha256({int(key): value for key, value in table.items()})


def load_runtime_trainer_v28c(arm, default_empty_directory, layer_bundle):
    if arm not in prereg.ARMS_V28C:
        raise ValueError("V28C arm changed")
    default_empty_directory = Path(default_empty_directory).resolve()
    if not default_empty_directory.is_dir() or any(default_empty_directory.iterdir()):
        raise RuntimeError("V28C dedicated default folder must be fresh and empty")
    folder = (
        default_empty_directory
        if arm == "default_empty" else prereg.TUNED_DIRECTORY_V28C
    )
    expected_config_sha = _actor_config_expected_sha_v28c()
    parent = anchor_v13.load_trainer(layer_bundle)

    class V28CTrainStepRuntimeTrainer(parent):
        def launch_engines(
            self, num_engines=4, n_gpu_per_vllm_engine=1,
            model_name="unused", precision="bfloat16",
        ):
            if (
                num_engines != 4 or n_gpu_per_vllm_engine != 1
                or Path(model_name).resolve() != prereg.MODEL_PATH_V28C.resolve()
                or precision != "bfloat16"
            ):
                raise ValueError("V28C requires exact BF16 model on four TP1 engines")
            import ray
            from es_at_scale.trainer.es_trainer import ESNcclLLM
            from ray.util.placement_group import placement_group
            from ray.util.scheduling_strategies import (
                PlacementGroupSchedulingStrategy,
            )

            class PlacementProbeV28C:
                @staticmethod
                def identity_v28c():
                    ids = ray.get_gpu_ids()
                    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                    if not isinstance(ids, list) or len(ids) != 1:
                        raise RuntimeError("V28C placement probe GPU coverage changed")
                    physical = normalize_ray_gpu_id_v28c(ids[0])
                    if visible != str(physical):
                        raise RuntimeError("V28C placement probe visible GPU changed")
                    return {
                        "physical_gpu_id": physical,
                        "ray_gpu_id_raw": ids[0],
                        "cuda_visible_devices": visible,
                    }

            class ProfiledESNcclLLMV28C(ESNcclLLM):
                def __init__(
                    self, *args, expected_arm, expected_model,
                    expected_folder, expected_config_content_sha256, **kwargs,
                ):
                    self._v28c_arm = str(expected_arm)
                    self._v28c_model = str(expected_model)
                    self._v28c_folder = str(expected_folder)
                    self._v28c_config_sha = str(expected_config_content_sha256)
                    super().__init__(*args, **kwargs)

                @staticmethod
                def _physical_gpu_v28c():
                    ids = ray.get_gpu_ids()
                    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                    if not isinstance(ids, list) or len(ids) != 1:
                        raise RuntimeError("V28C actor requires one Ray GPU ID")
                    physical = normalize_ray_gpu_id_v28c(ids[0])
                    if visible != str(physical):
                        raise RuntimeError("V28C actor Ray/visible GPU mismatch")
                    return physical, ids[0]

                def runtime_identity_v28c(self):
                    import pynvml
                    import torch
                    import vllm.envs as vllm_envs
                    import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe

                    physical, raw = self._physical_gpu_v28c()
                    actual_folder = os.environ.get("VLLM_TUNED_CONFIG_FOLDER")
                    if (
                        torch.cuda.device_count() != 1
                        or torch.cuda.current_device() != 0
                        or actual_folder != self._v28c_folder
                        or vllm_envs.VLLM_TUNED_CONFIG_FOLDER != self._v28c_folder
                    ):
                        raise RuntimeError("V28C actor device/config folder changed")
                    folder_path = Path(actual_folder)
                    names = tuple(sorted(item.name for item in folder_path.iterdir()))
                    fused_moe.get_moe_configs.cache_clear()
                    configs = fused_moe.get_moe_configs(256, 512, None)
                    if self._v28c_arm == "default_empty":
                        if names or configs is not None:
                            raise RuntimeError("V28C default did not use generic fallback")
                        source = "generic_fallback_none"
                        config_sha = None
                    else:
                        if (
                            names != (prereg.TUNED_FILENAME_V28C,)
                            or file_sha256(folder_path / names[0])
                            != prereg.TUNED_TABLE_FILE_SHA256_V28C
                            or not isinstance(configs, dict)
                            or canonical_sha256(configs) != self._v28c_config_sha
                        ):
                            raise RuntimeError("V28C actor did not load exact V27C table")
                        source = "exact_committed_v27c_table"
                        config_sha = self._v28c_config_sha
                    pynvml.nvmlInit()
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(physical)
                        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        uuid = pynvml.nvmlDeviceGetUUID(handle)
                        pci = pynvml.nvmlDeviceGetPciInfo(handle).busId
                        if isinstance(uuid, bytes):
                            uuid = uuid.decode("ascii")
                        if isinstance(pci, bytes):
                            pci = pci.decode("ascii")
                        return {
                            "schema": "eggroll-es-v28c-actor-runtime-identity",
                            "arm": self._v28c_arm,
                            "model_path": self._v28c_model,
                            "physical_gpu_id": physical,
                            "ray_gpu_id_raw": raw,
                            "cuda_visible_devices": str(physical),
                            "nvml_uuid": str(uuid),
                            "pci_bus_id": str(pci),
                            "total_bytes": int(memory.total),
                            "pid": os.getpid(),
                            "config_folder": actual_folder,
                            "config_source": source,
                            "config_content_sha256": config_sha,
                            "vllm_fused_moe_file_sha256": file_sha256(
                                Path(fused_moe.__file__).resolve()
                            ),
                            "vllm_envs_file_sha256": file_sha256(
                                Path(vllm_envs.__file__).resolve()
                            ),
                        }
                    finally:
                        pynvml.nvmlShutdown()

                @staticmethod
                def synchronize_v28c():
                    import torch
                    torch.cuda.synchronize(0)
                    return True

                @staticmethod
                def reset_peak_memory_v28c():
                    import torch
                    torch.cuda.synchronize(0)
                    torch.cuda.reset_peak_memory_stats(0)
                    return True

                @staticmethod
                def peak_memory_v28c():
                    import torch
                    torch.cuda.synchronize(0)
                    return {
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                    }

            pgs = [
                placement_group(
                    [{"GPU": 1, "CPU": 0}], strategy="PACK", lifetime="detached",
                )
                for _ in prereg.PHYSICAL_GPU_IDS_V28C
            ]
            ray.get([group.ready() for group in pgs])
            strategies = [
                PlacementGroupSchedulingStrategy(
                    placement_group=group,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=0,
                )
                for group in pgs
            ]
            probes = [
                ray.remote(num_cpus=0, num_gpus=1, scheduling_strategy=strategy)(
                    PlacementProbeV28C
                ).remote()
                for strategy in strategies
            ]
            probe_identities = ray.get([
                probe.identity_v28c.remote() for probe in probes
            ])
            strategy_by_gpu = {}
            for strategy, probe, identity in zip(
                strategies, probes, probe_identities, strict=True,
            ):
                physical = identity["physical_gpu_id"]
                if physical in strategy_by_gpu:
                    raise RuntimeError("V28C placement probes repeated a GPU")
                strategy_by_gpu[physical] = strategy
                ray.kill(probe)
            if set(strategy_by_gpu) != set(prereg.PHYSICAL_GPU_IDS_V28C):
                raise RuntimeError("V28C placement probes did not cover GPUs 0..3")
            engines = []
            for gpu_id in prereg.PHYSICAL_GPU_IDS_V28C:
                engines.append(ray.remote(
                    num_cpus=0, num_gpus=1,
                    scheduling_strategy=strategy_by_gpu[gpu_id],
                )(ProfiledESNcclLLMV28C).options(runtime_env={
                    "env_vars": {"VLLM_TUNED_CONFIG_FOLDER": str(folder)},
                }).remote(
                    model=str(prereg.MODEL_PATH_V28C),
                    expected_arm=arm,
                    expected_model=str(prereg.MODEL_PATH_V28C),
                    expected_folder=str(folder),
                    expected_config_content_sha256=expected_config_sha,
                    tensor_parallel_size=1,
                    worker_extension_cls=anchor_v13.WORKER_EXTENSION,
                    dtype="bfloat16",
                    enable_prefix_caching=False,
                    enforce_eager=True,
                    gpu_memory_utilization=0.82,
                    max_model_len=2048,
                    limit_mm_per_prompt={"image": 0, "video": 0},
                    mm_processor_cache_gb=0,
                    skip_mm_profiling=True,
                    moe_backend="triton",
                ))
            identities = ray.get([
                engine.runtime_identity_v28c.remote() for engine in engines
            ])
            self._v28c_probe_identities = probe_identities
            self._v28c_actor_identities = identities
            return engines, pgs

    return V28CTrainStepRuntimeTrainer


def _make_trainer_v28c(arm, empty_directory, layer_bundle, temp_output):
    trainer_class = load_runtime_trainer_v28c(
        arm, empty_directory, layer_bundle,
    )
    return trainer_class(
        model_name=str(prereg.MODEL_PATH_V28C), checkpoint=None, sigma=0.0003,
        alpha=0.0, population_size=32, reward_shaping="z-scores",
        num_iterations=1, max_tokens=1, batch_size=56, mini_batch_size=56,
        reward_function=base.specialist_reward,
        template_function=base.specialist_template,
        train_dataloader=[], eval_dataloader_dict={}, eval_freq=1,
        n_vllm_engines=4, n_gpu_per_vllm_engine=1, logging="none",
        global_seed=43, use_gpus="0,1,2,3",
        experiment_name="v28c_transient_engine_group",
        wandb_project="specialist-eggroll-es", save_best_models=False,
        reward_function_timeout=10, output_directory=str(temp_output),
    )


def validate_actor_identities_v28c(identities, arm, folder, baseline_identity):
    if not isinstance(identities, list) or len(identities) != 4:
        raise RuntimeError("V28C actor identity count changed")
    expected_source = (
        "generic_fallback_none"
        if arm == "default_empty" else "exact_committed_v27c_table"
    )
    expected_config_sha = (
        None if arm == "default_empty" else _actor_config_expected_sha_v28c()
    )
    pids = {}
    compact = []
    for gpu_id, identity in enumerate(identities):
        expected_physical = baseline_identity[gpu_id]
        if (
            identity.get("schema") != "eggroll-es-v28c-actor-runtime-identity"
            or identity.get("arm") != arm
            or identity.get("model_path") != str(prereg.MODEL_PATH_V28C)
            or identity.get("physical_gpu_id") != gpu_id
            or normalize_ray_gpu_id_v28c(identity.get("ray_gpu_id_raw")) != gpu_id
            or identity.get("cuda_visible_devices") != str(gpu_id)
            or identity.get("config_folder") != str(folder)
            or identity.get("config_source") != expected_source
            or identity.get("config_content_sha256") != expected_config_sha
            or identity.get("vllm_fused_moe_file_sha256")
            != VLLM_FUSED_MOE_SHA256_V28C
            or identity.get("vllm_envs_file_sha256") != VLLM_ENVS_SHA256_V28C
            or any(
                identity.get(key) != expected_physical[key]
                for key in ("nvml_uuid", "pci_bus_id", "total_bytes")
            )
            or isinstance(identity.get("pid"), bool)
            or not isinstance(identity.get("pid"), int)
            or identity["pid"] <= 0
        ):
            raise RuntimeError("V28C actor/model/config/physical mapping changed")
        pids[gpu_id] = identity["pid"]
        compact.append({
            "physical_gpu_id": gpu_id,
            "nvml_uuid": identity["nvml_uuid"],
            "pci_bus_id": identity["pci_bus_id"],
            "total_bytes": identity["total_bytes"],
            "config_source": identity["config_source"],
            "config_content_sha256": identity["config_content_sha256"],
        })
    if len(set(pids.values())) != 4:
        raise RuntimeError("V28C four engine actors do not have unique processes")
    return pids, canonical_sha256(compact)


def monitor_estimator_activity_v28c(pids_by_gpu, function):
    if set(pids_by_gpu) != set(prereg.PHYSICAL_GPU_IDS_V28C):
        raise RuntimeError("V28C activity monitor PID coverage changed")
    import pynvml

    stop = threading.Event()
    samples = []
    failures = []

    def monitor():
        try:
            pynvml.nvmlInit()
            handles = {
                gpu_id: pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                for gpu_id in prereg.PHYSICAL_GPU_IDS_V28C
            }
            try:
                while not stop.is_set():
                    row = {}
                    for gpu_id, handle in handles.items():
                        processes = []
                        for name in (
                            "nvmlDeviceGetComputeRunningProcesses",
                            "nvmlDeviceGetGraphicsRunningProcesses",
                        ):
                            callback = getattr(pynvml, name, None)
                            if callback is not None:
                                try:
                                    processes.extend(callback(handle))
                                except pynvml.NVMLError_NotSupported:
                                    pass
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        row[gpu_id] = {
                            "expected_process_present": pids_by_gpu[gpu_id]
                            in {int(item.pid) for item in processes},
                            "gpu_utilization_percent": int(utilization.gpu),
                            "used_bytes": int(memory.used),
                            "total_bytes": int(memory.total),
                        }
                    samples.append(row)
                    stop.wait(0.01)
            finally:
                pynvml.nvmlShutdown()
        except BaseException as error:
            failures.append(type(error).__name__)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        result = function()
    finally:
        stop.set()
        thread.join()
    if failures or not samples:
        raise RuntimeError("V28C PID-bound NVML activity monitor failed")
    qualifying = [
        row for row in samples
        if all(
            row[gpu_id]["expected_process_present"]
            and row[gpu_id]["gpu_utilization_percent"] > 0
            for gpu_id in prereg.PHYSICAL_GPU_IDS_V28C
        )
    ]
    if not qualifying:
        raise RuntimeError("V28C did not observe simultaneous activity on all four GPUs")
    peak_fraction = max(
        row[gpu_id]["used_bytes"] / row[gpu_id]["total_bytes"]
        for row in samples for gpu_id in prereg.PHYSICAL_GPU_IDS_V28C
    )
    compact_audit = {
        "all_four_expected_processes_and_positive_utilization_simultaneously": True,
        "sample_count": len(samples),
        "qualifying_sample_count": len(qualifying),
        "peak_nvml_fraction": float(peak_fraction),
    }
    compact_audit["commitment_sha256"] = canonical_sha256(compact_audit)
    return result, compact_audit


def exact_equivalence_v28c(default, tuned):
    default_analysis = default["analysis"]
    tuned_analysis = tuned["analysis"]
    panel_coefficients_exact = all(
        default_analysis["panel_analysis"][name]["coefficients"]
        == tuned_analysis["panel_analysis"][name]["coefficients"]
        for name in prereg.PANEL_NAMES_V28C
    )
    components = {
        "full_diagnostic_exact": default == tuned,
        "full_diagnostic_commitment_exact": (
            default["content_sha256_before_self_field"]
            == tuned["content_sha256_before_self_field"]
        ),
        "all_five_panel_coefficient_arrays_exact": panel_coefficients_exact,
        "all_response_and_analysis_payloads_exact": (
            default["responses"] == tuned["responses"]
            and default_analysis == tuned_analysis
        ),
        "robust_aggregate_coefficients_exact": (
            default_analysis["robust_optimization_aggregate"]["coefficients"]
            == tuned_analysis["robust_optimization_aggregate"]["coefficients"]
        ),
        "identity_guard_exact": (
            default["identity_audit"] == tuned["identity_audit"]
        ),
        "population_boundary_guard_exact": (
            default["population_boundary_audit_v4"]
            == tuned["population_boundary_audit_v4"]
        ),
        "panel_seed_and_common_random_number_contract_exact": all(
            default[key] == tuned[key] for key in (
                "perturbation_basis", "panel_contract", "common_random_numbers",
                "hardware_coverage",
            )
        ),
    }
    return {**components, "pass": all(components.values())}


def _bootstrap_median_bounds_v28c(values, draws):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (prereg.PAIR_COUNT_V28C,) or not np.all(np.isfinite(values)):
        raise RuntimeError("V28C paired endpoint vector changed")
    replicates = np.median(values[draws], axis=1)
    return (
        float(np.quantile(
            replicates, prereg.FAMILYWISE_LOWER_QUANTILE_V28C,
            method="linear",
        )),
        float(np.quantile(
            replicates, prereg.FAMILYWISE_UPPER_QUANTILE_V28C,
            method="linear",
        )),
    )


def performance_summary_v28c(pair_results):
    if tuple(pair_results) != prereg.PAIR_ORDER_V28C:
        raise RuntimeError("V28C paired result order changed")
    speed = []
    allocated = []
    reserved = []
    configure_speed = []
    estimator_speed = []
    peak_nvml_fractions = []
    order_speed = {"default_first": [], "tuned_first": []}
    for pair in prereg.PAIR_ORDER_V28C:
        result = pair_results[pair]
        default = result["arms"]["default_empty"]
        tuned = result["arms"]["v27c_tuned"]
        pair_speed = default["full_elapsed_ns"] / tuned["full_elapsed_ns"]
        speed.append(pair_speed)
        allocated.append(
            tuned["peak_allocated_bytes"] / default["peak_allocated_bytes"]
        )
        reserved.append(
            tuned["peak_reserved_bytes"] / default["peak_reserved_bytes"]
        )
        configure_speed.append(
            default["configure_elapsed_ns"] / tuned["configure_elapsed_ns"]
        )
        estimator_speed.append(
            default["estimate_elapsed_ns"] / tuned["estimate_elapsed_ns"]
        )
        peak_nvml_fractions.extend([
            default["activity"]["peak_nvml_fraction"],
            tuned["activity"]["peak_nvml_fraction"],
        ])
        order = (
            "default_first"
            if prereg.ARM_ORDER_BY_PAIR_V28C[pair][0] == "default_empty"
            else "tuned_first"
        )
        order_speed[order].append(pair_speed)
    draws, draw_sha = prereg.bootstrap_draw_plan_v28c()
    speed_lcb, _speed_ucb = _bootstrap_median_bounds_v28c(speed, draws)
    _alloc_lcb, allocated_ucb = _bootstrap_median_bounds_v28c(allocated, draws)
    _reserved_lcb, reserved_ucb = _bootstrap_median_bounds_v28c(reserved, draws)
    speed_median = float(np.median(speed))
    allocated_median = float(np.median(allocated))
    reserved_median = float(np.median(reserved))
    endpoints = {
        "complete_train_step_speed": {
            "median_default_over_tuned_ratio": speed_median,
            "familywise_lower_confidence_bound": speed_lcb,
            "point_threshold": prereg.SPEED_POINT_RATIO_MIN_V28C,
            "lower_bound_strict_threshold": prereg.SPEED_LCB_RATIO_MIN_V28C,
            "pass": bool(
                speed_median >= prereg.SPEED_POINT_RATIO_MIN_V28C
                and speed_lcb > prereg.SPEED_LCB_RATIO_MIN_V28C
            ),
        },
        "peak_torch_allocated": {
            "median_tuned_over_default_ratio": allocated_median,
            "familywise_upper_confidence_bound": allocated_ucb,
            "point_threshold": prereg.MEMORY_POINT_RATIO_MAX_V28C,
            "upper_bound_threshold": prereg.MEMORY_UCB_RATIO_MAX_V28C,
            "pass": bool(
                allocated_median <= prereg.MEMORY_POINT_RATIO_MAX_V28C
                and allocated_ucb <= prereg.MEMORY_UCB_RATIO_MAX_V28C
            ),
        },
        "peak_torch_reserved": {
            "median_tuned_over_default_ratio": reserved_median,
            "familywise_upper_confidence_bound": reserved_ucb,
            "point_threshold": prereg.MEMORY_POINT_RATIO_MAX_V28C,
            "upper_bound_threshold": prereg.MEMORY_UCB_RATIO_MAX_V28C,
            "pass": bool(
                reserved_median <= prereg.MEMORY_POINT_RATIO_MAX_V28C
                and reserved_ucb <= prereg.MEMORY_UCB_RATIO_MAX_V28C
            ),
        },
    }
    return {
        "schema": "eggroll-es-v28c-compact-paired-performance-summary",
        "pair_count": prereg.PAIR_COUNT_V28C,
        "endpoints": endpoints,
        "all_three_inferential_endpoints_passed": all(
            item["pass"] for item in endpoints.values()
        ),
        "absolute_peak_nvml_fraction_max_observed": float(
            max(peak_nvml_fractions)
        ),
        "absolute_peak_nvml_gate_passed": bool(
            max(peak_nvml_fractions) <= prereg.MAX_PEAK_NVML_FRACTION_V28C
        ),
        "descriptive_subphase_median_default_over_tuned_ratios": {
            "configure_train_panels_v13": float(np.median(configure_speed)),
            "estimate_train_panels_v13": float(np.median(estimator_speed)),
        },
        "descriptive_order_stratum_speed_medians": {
            key: float(np.median(values)) for key, values in order_speed.items()
        },
        "bootstrap_draw_plan_sha256": draw_sha,
        "raw_pair_vectors_or_bootstrap_replicates_persisted": False,
    }


def gate_v28c(equivalence, performance, runtime_integrity):
    passed = bool(
        equivalence["all_twelve_pairs_exact"]
        and performance["all_three_inferential_endpoints_passed"]
        and performance["absolute_peak_nvml_gate_passed"]
        and runtime_integrity["all_integrity_gates_passed"]
    )
    return _seal({
        "schema": "eggroll-es-v28c-authorization-gate",
        "pass": passed,
        "decision": (
            "authorize_only_exact_v27c_table_in_a_separately_frozen_"
            "bf16_train_only_training_recipe"
            if passed else "retain_empty_default_bf16_training_recipe"
        ),
        "all_twelve_pairs_exact": equivalence["all_twelve_pairs_exact"],
        "all_three_performance_endpoints_passed": performance[
            "all_three_inferential_endpoints_passed"
        ],
        "absolute_peak_nvml_gate_passed": performance[
            "absolute_peak_nvml_gate_passed"
        ],
        "all_runtime_integrity_gates_passed": runtime_integrity[
            "all_integrity_gates_passed"
        ],
        "direct_recipe_adoption_authorized": False,
        "model_update_authorized": False,
        "checkpoint_write_authorized": False,
        "evaluation_authorized": False,
        "dataset_promotion_authorized": False,
        "nontrain_reuse_authorized": False,
        "fp8_reuse_authorized": False,
    })


def _physical_identity_map_v28c(certificate):
    return runtime_v28a._physical_identity_map_v28a(certificate)


def assert_all_four_gpus_idle_v28c():
    value = runtime_v28a.assert_all_four_gpus_idle_v28a()
    value = copy.deepcopy(value)
    value["schema"] = "eggroll-es-v28c-prelaunch-all-gpu-idle-certificate"
    return _seal(value)


def wait_for_group_cleanup_v28c(expected_idle_certificate, group_index):
    value = runtime_v28a._wait_for_gpu_cleanup_idle_v28a(
        expected_idle_certificate, "interwave",
    )
    value = copy.deepcopy(value)
    value["schema"] = "eggroll-es-v28c-between-group-idle-certificate"
    value["completed_group_count"] = group_index + 1
    return _seal(value)


def _run_one_arm_v28c(
    pair, arm, group_index, panels, layer_bundle, baseline_identity,
    expected_idle_certificate,
):
    trainer = None
    failure = None
    result = None
    cleanup_failure = None
    idle = None
    with tempfile.TemporaryDirectory(prefix="specialist_v28c_group_") as temp_root:
        temp_root = Path(temp_root)
        empty_folder = temp_root / "fresh_empty_vllm_config"
        empty_folder.mkdir()
        transient_output = temp_root / "transient_trainer_output"
        folder = (
            empty_folder if arm == "default_empty"
            else prereg.TUNED_DIRECTORY_V28C
        )
        try:
            base.set_seed(43)
            trainer = _make_trainer_v28c(
                arm, empty_folder, layer_bundle, transient_output,
            )
            identities = trainer._v28c_actor_identities
            pids, identity_commitment = validate_actor_identities_v28c(
                identities, arm, folder, baseline_identity,
            )
            trainer._resolve([
                engine.reset_peak_memory_v28c.remote()
                for engine in trainer.engines
            ])
            trainer._resolve([
                engine.synchronize_v28c.remote() for engine in trainer.engines
            ])
            full_started = time.perf_counter_ns()
            configure_started = time.perf_counter_ns()
            configuration = trainer.configure_train_panels_v13(
                panels, frozen_layer_plan=layer_bundle,
            )
            configure_elapsed = time.perf_counter_ns() - configure_started
            estimate_started = time.perf_counter_ns()
            diagnostic, activity = monitor_estimator_activity_v28c(
                pids,
                lambda: trainer.estimate_train_panels_v13(
                    anchor_v13.PERTURBATION_SEEDS_V13,
                ),
            )
            estimate_elapsed = time.perf_counter_ns() - estimate_started
            trainer._resolve([
                engine.synchronize_v28c.remote() for engine in trainer.engines
            ])
            full_elapsed = time.perf_counter_ns() - full_started
            peaks = trainer._resolve([
                engine.peak_memory_v28c.remote() for engine in trainer.engines
            ])
            anchor_v13.validate_diagnostic_v13(diagnostic)
            if (
                full_elapsed <= 0 or configure_elapsed <= 0 or estimate_elapsed <= 0
                or full_elapsed < configure_elapsed + estimate_elapsed
                or len(peaks) != 4
                or any(
                    item["peak_allocated_bytes"] <= 0
                    or item["peak_reserved_bytes"] <= 0
                    for item in peaks
                )
            ):
                raise RuntimeError("V28C complete-step timing or memory profile changed")
            result = {
                "pair": pair,
                "arm": arm,
                "group_index": group_index,
                "full_elapsed_ns": full_elapsed,
                "configure_elapsed_ns": configure_elapsed,
                "estimate_elapsed_ns": estimate_elapsed,
                "peak_allocated_bytes": max(
                    item["peak_allocated_bytes"] for item in peaks
                ),
                "peak_reserved_bytes": max(
                    item["peak_reserved_bytes"] for item in peaks
                ),
                "configuration_commitment_sha256": canonical_sha256(configuration),
                "diagnostic": diagnostic,
                "diagnostic_commitment_sha256": diagnostic[
                    "content_sha256_before_self_field"
                ],
                "activity": activity,
                "actor_identity_commitment_sha256": identity_commitment,
            }
        except BaseException as error:
            failure = error
        finally:
            if trainer is not None:
                try:
                    base.close_trainer(trainer)
                except BaseException as error:
                    cleanup_failure = error
                    if failure is None:
                        failure = error
            try:
                import ray
                ray.shutdown()
            except BaseException as error:
                if failure is None:
                    failure = error
        try:
            idle = wait_for_group_cleanup_v28c(
                expected_idle_certificate, group_index,
            )
            if _physical_identity_map_v28c(idle) != baseline_identity:
                raise RuntimeError("V28C between-group physical identity changed")
        except BaseException as error:
            if failure is None:
                failure = error
    if failure is not None:
        raise failure
    if cleanup_failure is not None or idle is None or result is None:
        raise RuntimeError("V28C group cleanup/result finalization changed")
    result["cleanup_certificate_sha256"] = idle[
        "content_sha256_before_self_field"
    ]
    return result, idle


def run_counterbalanced_probe_v28c(
    preregistration, panels, layer_bundle, prelaunch_idle,
):
    baseline_identity = _physical_identity_map_v28c(prelaunch_idle)
    pair_results = {}
    group_audits = []
    current_idle = prelaunch_idle
    group_index = 0
    for pair in prereg.PAIR_ORDER_V28C:
        arms = {}
        for arm in prereg.ARM_ORDER_BY_PAIR_V28C[pair]:
            result, current_idle = _run_one_arm_v28c(
                pair, arm, group_index, panels, layer_bundle,
                baseline_identity, current_idle,
            )
            arms[arm] = result
            group_audits.append({
                "pair": pair,
                "arm": arm,
                "group_index": group_index,
                "diagnostic_commitment_sha256": result[
                    "diagnostic_commitment_sha256"
                ],
                "actor_identity_commitment_sha256": result[
                    "actor_identity_commitment_sha256"
                ],
                "activity_commitment_sha256": result["activity"][
                    "commitment_sha256"
                ],
                "cleanup_certificate_sha256": result[
                    "cleanup_certificate_sha256"
                ],
            })
            group_index += 1
        equivalence = exact_equivalence_v28c(
            arms["default_empty"]["diagnostic"],
            arms["v27c_tuned"]["diagnostic"],
        )
        if equivalence["pass"] is not True:
            raise RuntimeError(f"V28C exact train-step equivalence failed: {pair}")
        pair_results[pair] = {"arms": arms, "equivalence": equivalence}
    if group_index != 24:
        raise RuntimeError("V28C fresh engine-group count changed")
    performance = performance_summary_v28c(pair_results)
    equivalence = {
        "schema": "eggroll-es-v28c-compact-exact-equivalence",
        "pair_count": 12,
        "exact_pair_count": sum(
            result["equivalence"]["pass"] for result in pair_results.values()
        ),
        "all_twelve_pairs_exact": all(
            result["equivalence"]["pass"] for result in pair_results.values()
        ),
        "shared_diagnostic_commitment_sha256": (
            next(iter(pair_results.values()))["arms"]["default_empty"]
            ["diagnostic_commitment_sha256"]
        ),
        "component_pass_counts": {
            key: sum(
                result["equivalence"][key] for result in pair_results.values()
            )
            for key in next(iter(pair_results.values()))["equivalence"]
            if key != "pass"
        },
    }
    activity_values = [
        result["arms"][arm]["activity"]
        for result in pair_results.values() for arm in prereg.ARMS_V28C
    ]
    runtime_integrity = {
        "schema": "eggroll-es-v28c-compact-runtime-integrity",
        "fresh_engine_group_count": group_index,
        "all_four_activity_group_count": sum(
            item[
                "all_four_expected_processes_and_positive_utilization_simultaneously"
            ] for item in activity_values
        ),
        "minimum_activity_sample_count": min(
            item["sample_count"] for item in activity_values
        ),
        "minimum_qualifying_activity_sample_count": min(
            item["qualifying_sample_count"] for item in activity_values
        ),
        "group_audit_bundle_sha256": canonical_sha256(group_audits),
        "final_idle_certificate_sha256": current_idle[
            "content_sha256_before_self_field"
        ],
        "all_24_fresh_groups_and_cleanup_gates_passed": group_index == 24,
        "all_24_groups_observed_all_four_active": all(
            item[
                "all_four_expected_processes_and_positive_utilization_simultaneously"
            ] for item in activity_values
        ),
        "final_all_four_idle": current_idle["all_four_idle"] is True,
        "physical_gpu_identity_preserved_across_all_groups": (
            _physical_identity_map_v28c(current_idle) == baseline_identity
        ),
        "all_integrity_gates_passed": True,
        "raw_pids_timings_memory_samples_or_diagnostics_persisted": False,
    }
    runtime_integrity["all_integrity_gates_passed"] = all((
        runtime_integrity["all_24_fresh_groups_and_cleanup_gates_passed"],
        runtime_integrity["all_24_groups_observed_all_four_active"],
        runtime_integrity["final_all_four_idle"],
        runtime_integrity["physical_gpu_identity_preserved_across_all_groups"],
    ))
    return equivalence, performance, runtime_integrity


def _parser_v28c():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v28c-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-recipe-sha256")
    parser.add_argument("--expected-source-commit")
    return parser


def run_exact_v28c(
    preregistration, panels, layer_bundle, implementation, recipe,
    committed_clean_source,
):
    environment = runtime_r2.certify_runtime_environment_r2()
    cpu_disk = live_cpu_disk_audit_v28c()
    attempt_path = OUTPUT_DIRECTORY_V28C / ATTEMPT_NAME_V28C
    run_dir = OUTPUT_DIRECTORY_V28C / EXPERIMENT_NAME_V28C
    report_path = run_dir / REPORT_NAME_V28C
    if attempt_path.exists() or run_dir.exists():
        raise RuntimeError("V28C requires fresh exclusive attempt and run paths")
    prelaunch_idle = assert_all_four_gpus_idle_v28c()
    attempt = {
        "schema": "eggroll-es-v28c-durable-launch-attempt",
        "status": "launching",
        "phase": "before_first_fresh_engine_group",
        "recipe_sha256": recipe["content_sha256_before_self_field"],
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "committed_clean_source_certificate_sha256": committed_clean_source[
            "content_sha256_before_self_field"
        ],
        "runtime_environment_certificate_sha256": environment[
            "content_sha256_before_self_field"
        ],
        "live_cpu_disk_audit_sha256": cpu_disk[
            "content_sha256_before_self_field"
        ],
        "prelaunch_idle_certificate_sha256": prelaunch_idle[
            "content_sha256_before_self_field"
        ],
        "model_update_applied": False,
        "checkpoint_written": False,
        "evaluation_opened": False,
        "dataset_promotion_applied": False,
    }
    runtime_v23a._exclusive_write_json_v23a(attempt_path, attempt)
    if run_dir.exists():
        raise RuntimeError("V28C run directory appeared after exclusive claim")
    failure = None
    equivalence = performance = runtime_integrity = final_idle = None
    try:
        equivalence, performance, runtime_integrity = run_counterbalanced_probe_v28c(
            preregistration, panels, layer_bundle, prelaunch_idle,
        )
        final_idle = runtime_integrity["final_idle_certificate_sha256"]
    except BaseException as error:
        failure = error
    if failure is not None:
        attempt.update({
            "status": "failed",
            "phase": "during_or_after_fresh_engine_group_loop",
            "failure_type": type(failure).__name__,
            "failure_sha256": canonical_sha256({
                "type": type(failure).__name__, "repr": repr(failure),
            }),
            "report_exists_after_attempt": report_path.exists(),
            "model_update_applied": False,
            "checkpoint_written": False,
            "evaluation_opened": False,
            "dataset_promotion_applied": False,
        })
        runtime_v23a._rewrite_json_v23a(attempt_path, attempt)
        raise failure
    gate = gate_v28c(equivalence, performance, runtime_integrity)
    report = _seal({
        "schema": "eggroll-es-v27c-bf16-train-step-ab-report-v28c",
        "status": "valid_completed_train_only_no_update_runtime_ab",
        "recipe_sha256": recipe["content_sha256_before_self_field"],
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "committed_clean_source_certificate_sha256": committed_clean_source[
            "content_sha256_before_self_field"
        ],
        "prelaunch_idle_certificate_sha256": prelaunch_idle[
            "content_sha256_before_self_field"
        ],
        "final_idle_certificate_sha256": final_idle,
        "equivalence": equivalence,
        "performance": performance,
        "runtime_integrity": runtime_integrity,
        "gate": gate,
        "model_update_applied": False,
        "checkpoint_written": False,
        "evaluation_opened": False,
        "dataset_promotion_applied": False,
        "nontrain_surface_opened": False,
        "raw_rows_prompts_answers_tokens_coefficients_timings_memory_pids_or_draws_persisted": False,
    })
    _assert_compact_v28c(report)
    runtime_v23a._exclusive_write_json_v23a(report_path, report)
    attempt.update({
        "status": "complete",
        "phase": "after_24_group_cleanup_and_compact_report",
        "report_binding": {
            "path": str(report_path),
            "file_sha256": file_sha256(report_path),
            "content_sha256": report["content_sha256_before_self_field"],
        },
        "final_idle_certificate_sha256": final_idle,
    })
    runtime_v23a._rewrite_json_v23a(attempt_path, attempt)
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    _assert_train_only_argv_v28c(argv)
    args = _parser_v28c().parse_args(argv)
    preregistration = load_preregistration_v28c()
    panels = anchor_v13.load_panel_bundle_v13()
    layer_bundle = load_layer_bundle_v28c()
    implementation = implementation_identity_v28c()
    recipe = recipe_v28c(preregistration, layer_bundle, implementation)
    committed_clean_source = validate_runtime_v28c(
        args, preregistration, implementation, recipe,
    )
    if args.v28c_dry_run:
        payload = _seal({
            "schema": "eggroll-es-v27c-bf16-train-step-ab-dry-run-v28c",
            "implementation_bundle_sha256": implementation["bundle_sha256"],
            "recipe_sha256": recipe["content_sha256_before_self_field"],
            "preregistration_file_sha256": PREREG_FILE_SHA256_V28C,
            "preregistration_content_sha256": PREREG_CONTENT_SHA256_V28C,
            "required_python": str(REQUIRED_PYTHON_V28C),
            "source_allowlist_contract_sha256": canonical_sha256(
                WORKTREE_ALLOWLIST_CONTRACT_V28C
            ),
            "fresh_exclusive_attempt_and_run_paths_required": True,
            "real_launch_requires_committed_clean_expected_source_commit": True,
            "pair_count": prereg.PAIR_COUNT_V28C,
            "fresh_four_tp1_engine_group_count": 24,
            "total_generation_request_budget": (
                prereg.TOTAL_GENERATION_REQUESTS_V28C
            ),
            "pass_authority": preregistration["gate_and_authority"][
                "pass_authority"
            ],
            "model_update_checkpoint_evaluation_dataset_nontrain_and_fp8_authorized": False,
            "gpu_launched": False,
            "train_only_runtime_launched": False,
        })
        _assert_compact_v28c(payload)
        print(json.dumps(payload, sort_keys=True))
        return payload
    return run_exact_v28c(
        preregistration, panels, layer_bundle, implementation, recipe,
        committed_clean_source,
    )


if __name__ == "__main__":
    main()
