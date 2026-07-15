#!/usr/bin/env python3
"""Fail-closed future four-GPU train-only V27C task-runtime A/B launcher."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import build_eggroll_es_insertion_model_seal_v23a as model_seal_v23a
import eggroll_es_v27c_runtime_ab_preregistration_v28a as prereg
import run_eggroll_es_insertion_stability_v23a as runtime_v23a
import run_eggroll_es_insertion_stability_v23a_retry_r2 as runtime_r2


ROOT = Path(__file__).resolve().parent
PREREG_PATH_V28A = prereg.OUTPUT_PATH_V28A
PREREG_FILE_SHA256_V28A = (
    "3552010cb5310be3e55daa040339a2377b2c4e528d57919bfd575ca6d29ecc7e"
)
PREREG_CONTENT_SHA256_V28A = (
    "ca59ca74e63aef9855b8e88d16b392ccb3c0b342c7a4a7d865b544fbef704468"
)
EXPERIMENT_NAME_V28A = "s6_v28a_v27c_tuned_vs_empty_default_train_runtime_ab"
OUTPUT_DIRECTORY_V28A = (ROOT / "experiments/eggroll_es_hpo/runs").resolve()
ATTEMPT_NAME_V28A = f".{EXPERIMENT_NAME_V28A}.launch_attempt.json"
REPORT_NAME_V28A = "v27c_tuned_runtime_ab_report_v28a.json"
DEFAULT_EMPTY_DIRECTORY_NAME_V28A = ".v28a_fresh_empty_default_config"
TEST_PATH_V28A = (ROOT / "test_run_eggroll_es_v27c_runtime_ab_v28a.py").resolve()
PREREG_TEST_PATH_V28A = (
    ROOT / "test_eggroll_es_v27c_runtime_ab_preregistration_v28a.py"
).resolve()
VLLM_FUSED_MOE_PATH_V28A = Path(
    "/home/catid/specialist/es-at-scale/.venv/lib/python3.12/site-packages/"
    "vllm/model_executor/layers/fused_moe/fused_moe.py"
)
VLLM_FUSED_MOE_SHA256_V28A = (
    "72811a4e543cc6f415f184cb951b61522643cddc4d6456f61a2f8c1a53b2cf79"
)
VLLM_ENVS_PATH_V28A = Path(
    "/home/catid/specialist/es-at-scale/.venv/lib/python3.12/site-packages/"
    "vllm/envs.py"
)
VLLM_ENVS_SHA256_V28A = (
    "15ab853b73b26da5dc2808699138dff7217b72b9f661da274cc0c9f6c262f631"
)
IMPLEMENTATION_PATHS_V28A = {
    "v28a_preregistration_module": Path(prereg.__file__).resolve(),
    "v28a_preregistration_tests": PREREG_TEST_PATH_V28A,
    "v28a_preregistration": PREREG_PATH_V28A,
    "v28a_runtime": Path(__file__).resolve(),
    "v28a_runtime_tests": TEST_PATH_V28A,
    "v27d_positive_evidence_builder": (
        ROOT / "build_vllm_moe_tuning_evaluation_positive_evidence_v27d.py"
    ),
    "v27d_positive_evidence_tests": (
        ROOT / "test_build_vllm_moe_tuning_evaluation_positive_evidence_v27d.py"
    ),
    "v27d_positive_evidence": prereg.POSITIVE_EVIDENCE_PATH_V28A,
    "v27c_selection_evidence_builder": (
        ROOT / "build_vllm_moe_tuning_selection_evidence_v27c.py"
    ),
    "v27c_selection_evidence_tests": (
        ROOT / "test_build_vllm_moe_tuning_selection_evidence_v27c.py"
    ),
    "v27c_selection_evidence": (
        ROOT / "experiments/eggroll_es_hpo/S6_V27C_MOE_TUNING_SELECTION_EVIDENCE.json"
    ),
    "v27c_tuned_table": prereg.TUNED_TABLE_PATH_V28A,
    "v23a_model_seal_builder": ROOT / "build_eggroll_es_insertion_model_seal_v23a.py",
    "v23a_model_seal_tests": ROOT / "test_build_eggroll_es_insertion_model_seal_v23a.py",
    "v23a_model_seal": prereg.MODEL_SEAL_PATH_V28A,
}
MOE_CONFOUNDING_ENV_V28A = (
    "VLLM_ROCM_USE_AITER", "VLLM_ROCM_USE_AITER_MOE",
    "VLLM_USE_DEEP_GEMM", "VLLM_MOE_USE_DEEP_GEMM",
)
FORBIDDEN_PERSISTED_KEYS_V28A = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "prompt_token_ids", "token_ids", "text", "texts", "outputs", "responses",
    "scores", "score_arrays", "elapsed_ns", "timing_vectors", "memory_samples",
    "bootstrap_draws", "bootstrap_replicates", "row_content", "row_sha256",
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


def _assert_compact_v28a(value):
    overlap = FORBIDDEN_PERSISTED_KEYS_V28A & set(_recursive_keys(value))
    if overlap:
        raise RuntimeError(f"V28A compact output contains forbidden keys: {sorted(overlap)}")


def normalize_ray_gpu_id_v28a(value):
    if isinstance(value, bool):
        raise RuntimeError("V28A Ray GPU ID representation changed")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value in {"0", "1", "2", "3"}:
        result = int(value)
    else:
        raise RuntimeError("V28A Ray GPU ID representation changed")
    if result not in prereg.PHYSICAL_GPU_IDS_V28A:
        raise RuntimeError("V28A Ray GPU ID physical range changed")
    return result


def load_preregistration_v28a():
    value = json.loads(PREREG_PATH_V28A.read_text(encoding="utf-8"))
    if (
        file_sha256(PREREG_PATH_V28A) != PREREG_FILE_SHA256_V28A
        or value.get("content_sha256_before_self_field")
        != PREREG_CONTENT_SHA256_V28A
        or canonical_sha256(_without_self(value)) != PREREG_CONTENT_SHA256_V28A
    ):
        raise RuntimeError("V28A preregistration identity changed")
    value["waves"] = {
        wave: value["waves"][wave] for wave in prereg.WAVE_ORDER_V28A
    }
    return prereg.validate_preregistration_v28a(value)


def implementation_identity_v28a():
    inherited = runtime_r2.implementation_identity_r2()
    files = dict(inherited["files"])
    for key, path in IMPLEMENTATION_PATHS_V28A.items():
        files[key] = {
            "path": str(Path(path).resolve()),
            "file_sha256": file_sha256(path),
        }
    return {
        "files": files,
        "inherited_v23a_r2_bundle_sha256": inherited["bundle_sha256"],
        "v28a_overlay_bundle_sha256": canonical_sha256({
            key: files[key] for key in IMPLEMENTATION_PATHS_V28A
        }),
        "bundle_sha256": canonical_sha256(files),
    }


def recipe_v28a(preregistration, implementation):
    value = {
        "schema": "eggroll-es-v27c-tuned-runtime-ab-recipe-v28a",
        "experiment_name": EXPERIMENT_NAME_V28A,
        "preregistration": {
            "path": str(PREREG_PATH_V28A),
            "file_sha256": PREREG_FILE_SHA256_V28A,
            "content_sha256": PREREG_CONTENT_SHA256_V28A,
        },
        "positive_evidence_commit": prereg.POSITIVE_EVIDENCE_COMMIT_V28A,
        "model_contract": copy.deepcopy(preregistration["model_contract"]),
        "tuned_table_contract": copy.deepcopy(
            preregistration["tuned_table_contract"]
        ),
        "arms": copy.deepcopy(preregistration["arms"]),
        "only_intended_arm_difference": "VLLM_TUNED_CONFIG_FOLDER",
        "waves": copy.deepcopy(preregistration["waves"]),
        "pairing": copy.deepcopy(preregistration["pairing"]),
        "train_request_contract": copy.deepcopy(
            preregistration["train_request_contract"]
        ),
        "sampling_contract": copy.deepcopy(preregistration["sampling_contract"]),
        "runtime_contract": copy.deepcopy(preregistration["runtime_contract"]),
        "exact_output_equivalence": copy.deepcopy(
            preregistration["exact_output_equivalence"]
        ),
        "performance_analysis": copy.deepcopy(
            preregistration["performance_analysis"]
        ),
        "gate": copy.deepcopy(preregistration["gate"]),
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "runtime_dependency_files": {
            "vllm_fused_moe": {
                "path": str(VLLM_FUSED_MOE_PATH_V28A),
                "file_sha256": VLLM_FUSED_MOE_SHA256_V28A,
            },
            "vllm_envs": {
                "path": str(VLLM_ENVS_PATH_V28A),
                "file_sha256": VLLM_ENVS_SHA256_V28A,
            },
        },
        "runtime_environment_contract_r2": (
            runtime_r2.expected_environment_contract_r2()
        ),
        "fresh_exclusive_paths": {
            "attempt": str(OUTPUT_DIRECTORY_V28A / ATTEMPT_NAME_V28A),
            "run_directory": str(OUTPUT_DIRECTORY_V28A / EXPERIMENT_NAME_V28A),
            "report_name": REPORT_NAME_V28A,
            "default_empty_directory_name": DEFAULT_EMPTY_DIRECTORY_NAME_V28A,
        },
        "model_update_allowed": False,
        "checkpoint_write_allowed": False,
        "evaluation_allowed": False,
        "direct_recipe_adoption_allowed": False,
        "dataset_promotion_allowed": False,
        "contains_validation_heldout_ood_or_benchmark_content": False,
    }
    return _seal(value)


def validate_runtime_v28a(args, preregistration, implementation, recipe):
    prereg.validate_preregistration_v28a(preregistration)
    conflicts = {
        key: os.environ.get(key)
        for key in (
            "VLLM_TUNED_CONFIG_FOLDER", *MOE_CONFOUNDING_ENV_V28A,
        )
        if os.environ.get(key) not in (None, "")
    }
    if conflicts:
        raise ValueError("V28A parent requires all MoE override environment empty")
    if os.environ.get("VLLM_BATCH_INVARIANT") not in (None, "0"):
        raise ValueError("V28A rejects the vLLM batch-invariant backend swap")
    if recipe.get("content_sha256_before_self_field") != canonical_sha256(
        _without_self(recipe)
    ):
        raise ValueError("V28A runtime recipe identity changed")
    for expected, actual, label in (
        (
            args.expected_implementation_bundle_sha256,
            implementation["bundle_sha256"], "implementation",
        ),
        (
            args.expected_recipe_sha256,
            recipe["content_sha256_before_self_field"], "recipe",
        ),
    ):
        if not args.v28a_dry_run and expected is None:
            raise ValueError(f"V28A real launch requires expected {label} hash")
        if expected is not None and expected != actual:
            raise ValueError(f"V28A {label} hash changed")


def live_cpu_disk_audit_v28a():
    table, evidence, base_seal = prereg.validate_static_inputs_v28a()
    if (
        file_sha256(VLLM_FUSED_MOE_PATH_V28A) != VLLM_FUSED_MOE_SHA256_V28A
        or file_sha256(VLLM_ENVS_PATH_V28A) != VLLM_ENVS_SHA256_V28A
    ):
        raise RuntimeError("V28A vLLM MoE loader dependency changed")
    all_files = model_seal_v23a._file_fingerprints(prereg.MODEL_PATH_V28A)
    if canonical_sha256(all_files) != (
        prereg.MODEL_BASE_ALL_FILES_FINGERPRINT_SHA256_V28A
    ):
        raise RuntimeError("V28A exact Qwen3.6 all-files fingerprint changed")
    metadata = {
        name: all_files[name]["sha256"]
        for name in prereg.MODEL_METADATA_SHA256_V28A
    }
    if metadata != prereg.MODEL_METADATA_SHA256_V28A:
        raise RuntimeError("V28A exact Qwen3.6 metadata identity changed")
    shard_hashes = {
        name: all_files[name] for name in sorted(base_seal["shards"])
    }
    if shard_hashes != base_seal["shards"]:
        raise RuntimeError("V28A exact Qwen3.6 shard content identity changed")
    return _seal({
        "schema": "eggroll-es-v28a-live-cpu-disk-audit",
        "model_config_sha256": prereg.MODEL_CONFIG_SHA256_V28A,
        "model_index_sha256": prereg.MODEL_INDEX_SHA256_V28A,
        "model_weight_shard_manifest": prereg.MODEL_WEIGHT_SHARD_MANIFEST_V28A,
        "model_metadata_sha256": metadata,
        "model_seal_file_sha256": prereg.MODEL_SEAL_FILE_SHA256_V28A,
        "model_seal_content_sha256": prereg.MODEL_SEAL_CONTENT_SHA256_V28A,
        "model_all_files_fingerprint_sha256": canonical_sha256(all_files),
        "all_26_model_shard_bytes_and_sha256_match_committed_seal": True,
        "model_shard_hash_audit_sha256": canonical_sha256(shard_hashes),
        "tuned_table_file_sha256": prereg.TUNED_TABLE_FILE_SHA256_V28A,
        "tuned_table_content_sha256": canonical_sha256(table),
        "positive_evidence_content_sha256": evidence[
            "content_sha256_before_self_field"
        ],
        "positive_evidence_commit": prereg.POSITIVE_EVIDENCE_COMMIT_V28A,
        "vllm_fused_moe_file_sha256": VLLM_FUSED_MOE_SHA256_V28A,
        "vllm_envs_file_sha256": VLLM_ENVS_SHA256_V28A,
        "dataset_rows_or_nontrain_surfaces_opened": False,
    })


def _observe_all_four_gpus_v28a():
    import pynvml

    pynvml.nvmlInit()
    try:
        reports = []
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V28A:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            compute = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            try:
                graphics = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
            except pynvml.NVMLError:
                graphics = []
            process_count = len({int(item.pid) for item in [*compute, *graphics]})
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            pci_bus_id = pynvml.nvmlDeviceGetPciInfo(handle).busId
            if isinstance(uuid, bytes):
                uuid = uuid.decode("ascii")
            if isinstance(pci_bus_id, bytes):
                pci_bus_id = pci_bus_id.decode("ascii")
            reports.append({
                "physical_gpu_id": gpu_id,
                "nvml_uuid": str(uuid),
                "pci_bus_id": str(pci_bus_id),
                "total_bytes": int(memory.total),
                "running_process_count": process_count,
            })
    finally:
        pynvml.nvmlShutdown()
    return _seal({
        "schema": "eggroll-es-v28a-four-gpu-observation",
        "gpu_count": 4,
        "gpus": reports,
        "all_four_idle": all(
            item["running_process_count"] == 0 for item in reports
        ),
    })


def assert_all_four_gpus_idle_v28a():
    observation = _observe_all_four_gpus_v28a()
    if observation.get("all_four_idle") is not True:
        raise RuntimeError("V28A launch blocked because a GPU has an existing owner")
    value = copy.deepcopy(observation)
    value["schema"] = "eggroll-es-v28a-all-gpu-idle-certificate"
    return _seal(value)


def _physical_identity_map_v28a(certificate):
    result = {
        item["physical_gpu_id"]: {
            "nvml_uuid": item.get("nvml_uuid"),
            "pci_bus_id": item.get("pci_bus_id"),
            "total_bytes": item.get("total_bytes"),
        }
        for item in certificate.get("gpus", [])
    }
    if (
        set(result) != set(prereg.PHYSICAL_GPU_IDS_V28A)
        or any(not item["nvml_uuid"] or not item["pci_bus_id"]
               for item in result.values())
    ):
        raise RuntimeError("V28A physical-GPU identity certificate changed")
    return result


def _wait_for_gpu_cleanup_idle_v28a(
    expected_idle_certificate, cleanup_phase, *,
    timeout_seconds=30.0, interval_seconds=0.5,
):
    if cleanup_phase not in {"interwave", "final"}:
        raise RuntimeError("V28A GPU cleanup phase changed")
    if timeout_seconds != 30.0 or interval_seconds != 0.5:
        raise RuntimeError("V28A GPU cleanup polling contract changed")
    expected = _physical_identity_map_v28a(expected_idle_certificate)
    started = time.monotonic()
    polls = 0
    while True:
        observation = _observe_all_four_gpus_v28a()
        polls += 1
        if _physical_identity_map_v28a(observation) != expected:
            raise RuntimeError("V28A GPU cleanup physical-GPU identity changed")
        elapsed = time.monotonic() - started
        if observation.get("all_four_idle") is True:
            return _seal({
                "schema": f"eggroll-es-v28a-{cleanup_phase}-cleanup-idle-certificate",
                "cleanup_phase": cleanup_phase,
                "gpus": observation["gpus"],
                "all_four_idle": True,
                "poll_count": polls,
                "elapsed_milliseconds": int(round(elapsed * 1000.0)),
                "timeout_seconds": timeout_seconds,
                "interval_seconds": interval_seconds,
                "bounded_async_cleanup_wait": True,
            })
        remaining = timeout_seconds - elapsed
        if remaining <= 0:
            raise RuntimeError("V28A GPU cleanup exceeded 30 seconds")
        time.sleep(min(interval_seconds, remaining))


def wait_for_interwave_gpu_idle_v28a(expected_idle_certificate):
    return _wait_for_gpu_cleanup_idle_v28a(
        expected_idle_certificate, "interwave"
    )


def wait_for_final_gpu_idle_v28a(expected_idle_certificate):
    return _wait_for_gpu_cleanup_idle_v28a(
        expected_idle_certificate, "final"
    )


def _array_commitment_v28a(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    header = json.dumps({
        "shape": list(array.shape), "dtype": str(array.dtype),
    }, sort_keys=True, separators=(",", ":")).encode()
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def exact_output_equal_v28a(left, right):
    required = {
        "gold_mean_logprobs", "gold_dense_commitments", "generated_token_ids",
        "generated_text", "generated_selected_logprobs",
        "generated_cumulative_logprobs", "generated_decoded_tokens",
    }
    if set(left) != required or set(right) != required:
        raise RuntimeError("V28A exact output contract geometry changed")
    return all(output_equivalence_components_v28a(left, right).values())


def output_equivalence_components_v28a(left, right):
    required = {
        "gold_mean_logprobs", "gold_dense_commitments", "generated_token_ids",
        "generated_text", "generated_selected_logprobs",
        "generated_cumulative_logprobs", "generated_decoded_tokens",
    }
    if set(left) != required or set(right) != required:
        raise RuntimeError("V28A exact output contract geometry changed")
    return {
        "gold_mean_logprobs_exact": bool(np.array_equal(
            left["gold_mean_logprobs"], right["gold_mean_logprobs"]
        )),
        "gold_dense_commitments_exact": bool(
            left["gold_dense_commitments"] == right["gold_dense_commitments"]
        ),
        "generated_token_id_lists_exact": bool(
            left["generated_token_ids"] == right["generated_token_ids"]
        ),
        "generated_text_exact": bool(
            left["generated_text"] == right["generated_text"]
        ),
        "generated_selected_logprobs_exact": bool(np.array_equal(
            left["generated_selected_logprobs"],
            right["generated_selected_logprobs"],
        )),
        "generated_cumulative_logprobs_exact": bool(np.array_equal(
            left["generated_cumulative_logprobs"],
            right["generated_cumulative_logprobs"],
        )),
        "generated_decoded_tokens_exact": bool(
            left["generated_decoded_tokens"] == right["generated_decoded_tokens"]
        ),
    }


def output_commitment_v28a(value):
    return canonical_sha256({
        "gold_mean_logprobs_sha256": _array_commitment_v28a(
            value["gold_mean_logprobs"]
        ),
        "gold_dense_commitments_sha256": canonical_sha256(
            value["gold_dense_commitments"]
        ),
        "generated_token_ids_sha256": canonical_sha256(
            value["generated_token_ids"]
        ),
        "generated_text_sha256": canonical_sha256(value["generated_text"]),
        "generated_selected_logprobs_sha256": _array_commitment_v28a(
            value["generated_selected_logprobs"]
        ),
        "generated_cumulative_logprobs_sha256": _array_commitment_v28a(
            value["generated_cumulative_logprobs"]
        ),
        "generated_decoded_tokens_sha256": canonical_sha256(
            value["generated_decoded_tokens"]
        ),
    })


def performance_summary_v28a(elapsed_ns, processed_tokens, resident, peak_nvml):
    elapsed = np.asarray(elapsed_ns, dtype=np.int64)
    peaks = np.asarray(peak_nvml, dtype=np.int64)
    expected_cells = {
        f"{wave}_gpu_{gpu_id}"
        for wave in prereg.WAVE_ORDER_V28A
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V28A
    }
    if (
        elapsed.shape != (2, prereg.TIMING_REPETITIONS_V28A, 4)
        or peaks.shape != elapsed.shape
        or np.any(elapsed <= 0)
        or np.any(peaks <= 0)
        or isinstance(processed_tokens, bool)
        or not isinstance(processed_tokens, int)
        or processed_tokens <= 0
        or set(resident) != expected_cells
    ):
        raise RuntimeError("V28A timing or memory geometry changed")
    throughput = float(processed_tokens) / (elapsed.astype(np.float64) / 1e9)
    repetition_draws, gpu_draws, draw_sha256 = prereg.bootstrap_draw_plan_v28a()
    cells = {}
    memory = {}
    peak_matrix = np.empty_like(peaks)
    waves = {}
    for wave_index, wave in enumerate(prereg.WAVE_ORDER_V28A):
        waves[wave] = {
            "median_processed_tokens_per_second_all_gpus": float(
                np.median(throughput[wave_index])
            ),
            "minimum_processed_tokens_per_second_all_gpus": float(
                np.min(throughput[wave_index])
            ),
            "maximum_processed_tokens_per_second_all_gpus": float(
                np.max(throughput[wave_index])
            ),
        }
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V28A:
            cell = f"{wave}_gpu_{gpu_id}"
            arm = prereg.WAVE_ARM_ORDER_V28A[wave][gpu_id]
            values = throughput[wave_index, :, gpu_id]
            cells[cell] = {
                "wave": wave,
                "physical_gpu_id": gpu_id,
                "arm": arm,
                "median_processed_tokens_per_second": float(np.median(values)),
                "minimum_processed_tokens_per_second": float(np.min(values)),
                "maximum_processed_tokens_per_second": float(np.max(values)),
                "coefficient_of_variation": float(np.std(values) / np.mean(values)),
                "timing_commitment_sha256": _array_commitment_v28a(
                    elapsed[wave_index, :, gpu_id]
                ),
            }
            report = resident[cell]
            total = report.get("total_bytes")
            resident_bytes = report.get("used_bytes")
            timed = peaks[wave_index, :, gpu_id]
            if (
                isinstance(total, bool) or not isinstance(total, int) or total <= 0
                or isinstance(resident_bytes, bool)
                or not isinstance(resident_bytes, int)
                or not 0 < resident_bytes <= total
                or np.any(timed <= 0) or np.any(timed > total)
            ):
                raise RuntimeError("V28A resident or peak VRAM contract changed")
            observed = np.maximum(timed, resident_bytes)
            peak_matrix[wave_index, :, gpu_id] = observed
            maximum = int(np.max(observed))
            memory[cell] = {
                "wave": wave,
                "physical_gpu_id": gpu_id,
                "arm": arm,
                "resident_used_bytes": resident_bytes,
                "maximum_resident_or_timed_nvml_used_bytes": maximum,
                "total_bytes": total,
                "peak_nvml_fraction": float(maximum / total),
                "within_absolute_peak_fraction": bool(
                    maximum / total <= prereg.MAX_PEAK_NVML_FRACTION_V28A
                ),
                "peak_profile_commitment_sha256": _array_commitment_v28a(observed),
            }
    speed_ratios = np.empty((4, prereg.TIMING_REPETITIONS_V28A), dtype=np.float64)
    vram_ratios = np.empty_like(speed_ratios)
    pairs = {}
    order_speed = {"default_first": [], "tuned_first": []}
    order_vram = {"default_first": [], "tuned_first": []}
    for gpu_id, pair_name in zip(
        prereg.PHYSICAL_GPU_IDS_V28A, prereg.PAIR_ORDER_V28A, strict=True,
    ):
        default_wave_index = 0 if gpu_id in (0, 2) else 1
        tuned_wave_index = 1 - default_wave_index
        speed = (
            throughput[tuned_wave_index, :, gpu_id]
            / throughput[default_wave_index, :, gpu_id]
        )
        vram = (
            peak_matrix[tuned_wave_index, :, gpu_id].astype(np.float64)
            / peak_matrix[default_wave_index, :, gpu_id].astype(np.float64)
        )
        speed_ratios[gpu_id] = speed
        vram_ratios[gpu_id] = vram
        speed_replicates = np.median(speed[repetition_draws], axis=1)
        vram_replicates = np.median(vram[repetition_draws], axis=1)
        speed_median = float(np.median(speed))
        speed_lcb = float(np.quantile(
            speed_replicates, prereg.FAMILYWISE_LOWER_QUANTILE_V28A,
            method="linear",
        ))
        vram_median = float(np.median(vram))
        vram_ucb = float(np.quantile(
            vram_replicates, prereg.FAMILYWISE_UPPER_QUANTILE_V28A,
            method="linear",
        ))
        order = "default_first" if default_wave_index == 0 else "tuned_first"
        order_speed[order].extend(float(item) for item in speed)
        order_vram[order].extend(float(item) for item in vram)
        default_cell = (
            f"{prereg.WAVE_ORDER_V28A[default_wave_index]}_gpu_{gpu_id}"
        )
        tuned_cell = f"{prereg.WAVE_ORDER_V28A[tuned_wave_index]}_gpu_{gpu_id}"
        absolute_memory_pass = all(
            memory[cell]["within_absolute_peak_fraction"] is True
            for cell in (default_cell, tuned_cell)
        )
        speed_pass = bool(
            speed_lcb >= prereg.PER_GPU_THROUGHPUT_NONREGRESSION_V28A
            and speed_median >= prereg.PER_GPU_POINT_THROUGHPUT_MIN_V28A
        )
        vram_pass = bool(
            vram_ucb <= prereg.PER_GPU_VRAM_UCB_NONREGRESSION_V28A
            and vram_median <= prereg.PER_GPU_POINT_VRAM_RATIO_MAX_V28A
            and absolute_memory_pass
        )
        pairs[pair_name] = {
            "physical_gpu_id": gpu_id,
            "load_order": order,
            "default_cell": default_cell,
            "tuned_cell": tuned_cell,
            "median_tuned_over_default_throughput_ratio": speed_median,
            "familywise_throughput_lower_confidence_bound": speed_lcb,
            "throughput_nonregression_lcb_threshold": (
                prereg.PER_GPU_THROUGHPUT_NONREGRESSION_V28A
            ),
            "throughput_point_estimate_threshold": (
                prereg.PER_GPU_POINT_THROUGHPUT_MIN_V28A
            ),
            "throughput_pass": speed_pass,
            "median_tuned_over_default_peak_vram_ratio": vram_median,
            "familywise_peak_vram_upper_confidence_bound": vram_ucb,
            "peak_vram_ucb_threshold": prereg.PER_GPU_VRAM_UCB_NONREGRESSION_V28A,
            "peak_vram_point_estimate_threshold": (
                prereg.PER_GPU_POINT_VRAM_RATIO_MAX_V28A
            ),
            "absolute_peak_vram_pass": absolute_memory_pass,
            "vram_pass": vram_pass,
            "pass": bool(speed_pass and vram_pass),
        }
    global_speed_replicates = np.median(
        speed_ratios[gpu_draws[:, :, None], repetition_draws[:, None, :]],
        axis=(1, 2),
    )
    global_speed_median = float(np.median(speed_ratios))
    global_speed_lcb = float(np.quantile(
        global_speed_replicates, 0.05, method="linear",
    ))
    global_improvement_pass = bool(
        global_speed_median
        >= prereg.GLOBAL_POINT_THROUGHPUT_IMPROVEMENT_V28A
        and global_speed_lcb
        >= prereg.GLOBAL_LCB_THROUGHPUT_IMPROVEMENT_V28A
    )
    return {
        "processed_tokens_per_engine_call": processed_tokens,
        "cells": cells,
        "waves": waves,
        "pairs": pairs,
        "memory": memory,
        "global_task_throughput": {
            "median_tuned_over_default_ratio": global_speed_median,
            "hierarchical_bootstrap_lower_confidence_bound": global_speed_lcb,
            "point_improvement_threshold": (
                prereg.GLOBAL_POINT_THROUGHPUT_IMPROVEMENT_V28A
            ),
            "lower_bound_improvement_threshold": (
                prereg.GLOBAL_LCB_THROUGHPUT_IMPROVEMENT_V28A
            ),
            "pass": global_improvement_pass,
        },
        "load_order_strata": {
            order: {
                "physical_gpu_ids": [0, 2] if order == "default_first" else [1, 3],
                "median_tuned_over_default_throughput_ratio": float(
                    np.median(order_speed[order])
                ),
                "median_tuned_over_default_peak_vram_ratio": float(
                    np.median(order_vram[order])
                ),
            }
            for order in ("default_first", "tuned_first")
        },
        "bootstrap": {
            "seed": prereg.BOOTSTRAP_SEED_V28A,
            "repetitions": prereg.BOOTSTRAP_REPETITIONS_V28A,
            "draw_plan_sha256": draw_sha256,
            "raw_draws_replicates_timing_and_memory_vectors_persisted": False,
        },
    }


def evaluate_gate_v28a(equivalence, performance, runtime_integrity):
    expected_cells = {
        f"{wave}_gpu_{gpu_id}"
        for wave in prereg.WAVE_ORDER_V28A
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V28A
    }
    if (
        set(equivalence) != set(prereg.PAIR_ORDER_V28A)
        or set(performance.get("pairs", {})) != set(prereg.PAIR_ORDER_V28A)
        or set(performance.get("memory", {})) != expected_cells
        or runtime_integrity.get("all_integrity_audits_passed") is not True
    ):
        raise RuntimeError("V28A gate input contract changed")
    exact_pass = all(
        equivalence[pair].get("exact_output_equivalence") is True
        for pair in prereg.PAIR_ORDER_V28A
    )
    pair_speed_pass = all(
        performance["pairs"][pair]["throughput_pass"] is True
        for pair in prereg.PAIR_ORDER_V28A
    )
    pair_vram_pass = all(
        performance["pairs"][pair]["vram_pass"] is True
        for pair in prereg.PAIR_ORDER_V28A
    )
    absolute_vram_pass = all(
        performance["memory"][cell]["within_absolute_peak_fraction"] is True
        for cell in expected_cells
    )
    global_speed_pass = performance["global_task_throughput"]["pass"] is True
    passed = bool(
        exact_pass and pair_speed_pass and pair_vram_pass
        and absolute_vram_pass and global_speed_pass
    )
    return _seal({
        "schema": "eggroll-es-v28a-authorization-gate",
        "pass": passed,
        "decision": (
            "authorize_only_a_separately_frozen_train_only_training_recipe_A_B"
            if passed else "retain_empty_default_config_training_recipe"
        ),
        "all_runtime_integrity_audits_passed": True,
        "all_four_exact_output_pairs_passed": exact_pass,
        "all_four_per_gpu_throughput_gates_passed": pair_speed_pass,
        "global_task_throughput_improvement_gate_passed": global_speed_pass,
        "all_four_per_gpu_vram_gates_passed": pair_vram_pass,
        "all_eight_absolute_peak_vram_gates_passed": absolute_vram_pass,
        "direct_recipe_adoption_authorized": False,
        "model_update_authorized": False,
        "checkpoint_write_authorized": False,
        "evaluation_authorized": False,
        "dataset_promotion_authorized": False,
    })


class V28ProbeController:
    def __init__(self, preregistration, wave, default_empty_directory):
        if wave not in prereg.WAVE_ORDER_V28A:
            raise RuntimeError("V28A crossover wave changed")
        self.preregistration = preregistration
        self.wave = wave
        self.assignments = preregistration["waves"][wave]["gpu_assignments"]
        self.default_empty_directory = Path(default_empty_directory).resolve()
        self.engines = []
        self.placement_groups = []
        self.placement_probe_identities = []
        self._ray = None

    def _folder_for_arm(self, arm):
        if arm == "default_empty":
            return self.default_empty_directory
        if arm == "v27c_tuned":
            return prereg.TUNED_DIRECTORY_V28A
        raise RuntimeError("V28A unknown arm")

    def launch_engines(self):
        import ray
        from es_at_scale.trainer.es_trainer import ESNcclLLM
        from ray.util.placement_group import placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        class PlacementProbeV28A:
            @staticmethod
            def identity_v28a():
                ids = ray.get_gpu_ids()
                visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                if not isinstance(ids, list) or len(ids) != 1:
                    raise RuntimeError("V28A placement probe GPU coverage changed")
                physical = normalize_ray_gpu_id_v28a(ids[0])
                if visible != str(physical):
                    raise RuntimeError("V28A placement probe visible GPU changed")
                return {
                    "ray_gpu_id_raw": ids[0],
                    "ray_gpu_id_canonical": physical,
                    "cuda_visible_devices": visible,
                }

        class ProfiledESNcclLLMV28A(ESNcclLLM):
            def __init__(
                self, *args, probe_cell, probe_wave, probe_arm,
                expected_model_path, expected_config_folder,
                expected_tuned_config_content_sha256, **kwargs,
            ):
                self._v28a_probe_cell = str(probe_cell)
                self._v28a_probe_wave = str(probe_wave)
                self._v28a_probe_arm = str(probe_arm)
                self._v28a_expected_model_path = str(expected_model_path)
                self._v28a_expected_config_folder = str(expected_config_folder)
                self._v28a_expected_tuned_sha = str(
                    expected_tuned_config_content_sha256
                )
                super().__init__(*args, **kwargs)

            @staticmethod
            def _physical_gpu_v28a():
                ids = ray.get_gpu_ids()
                visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                if not isinstance(ids, list) or len(ids) != 1:
                    raise RuntimeError("V28A actor requires one Ray GPU ID")
                physical = normalize_ray_gpu_id_v28a(ids[0])
                if visible != str(physical):
                    raise RuntimeError("V28A actor Ray and visible GPU IDs disagree")
                return physical, ids[0]

            def runtime_identity_v28a(self):
                import pynvml
                import torch
                import vllm.envs as vllm_envs
                import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe

                physical, raw_ray_id = self._physical_gpu_v28a()
                actual_folder = os.environ.get("VLLM_TUNED_CONFIG_FOLDER")
                if (
                    torch.cuda.device_count() != 1
                    or torch.cuda.current_device() != 0
                    or actual_folder != self._v28a_expected_config_folder
                    or vllm_envs.VLLM_TUNED_CONFIG_FOLDER
                    != self._v28a_expected_config_folder
                ):
                    raise RuntimeError("V28A actor device or tuned-folder identity changed")
                folder = Path(actual_folder)
                names = tuple(sorted(item.name for item in folder.iterdir()))
                if self._v28a_probe_arm == "default_empty":
                    if names:
                        raise RuntimeError("V28A default config directory is not empty")
                elif names != (prereg.TUNED_FILENAME_V28A,):
                    raise RuntimeError("V28A tuned config directory contents changed")
                fused_moe.get_moe_configs.cache_clear()
                configs = fused_moe.get_moe_configs(256, 512, None)
                if self._v28a_probe_arm == "default_empty":
                    if configs is not None:
                        raise RuntimeError("V28A default actor did not use generic fallback")
                    config_source = "generic_fallback_none"
                    config_sha256 = None
                else:
                    if not isinstance(configs, dict):
                        raise RuntimeError("V28A tuned actor did not load exact V27C table")
                    config_sha256 = canonical_sha256(configs)
                    if config_sha256 != self._v28a_expected_tuned_sha:
                        raise RuntimeError("V28A tuned actor table content changed")
                    config_source = "exact_committed_v27c_table"
                pynvml.nvmlInit()
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(physical)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    uuid = pynvml.nvmlDeviceGetUUID(handle)
                    pci_bus_id = pynvml.nvmlDeviceGetPciInfo(handle).busId
                    if isinstance(uuid, bytes):
                        uuid = uuid.decode("ascii")
                    if isinstance(pci_bus_id, bytes):
                        pci_bus_id = pci_bus_id.decode("ascii")
                    return {
                        "schema": "eggroll-es-v28a-actor-runtime-identity",
                        "cell": self._v28a_probe_cell,
                        "wave": self._v28a_probe_wave,
                        "arm": self._v28a_probe_arm,
                        "model_path": self._v28a_expected_model_path,
                        "physical_gpu_id": physical,
                        "ray_gpu_id_raw": raw_ray_id,
                        "ray_gpu_id_canonical": physical,
                        "cuda_visible_devices": str(physical),
                        "nvml_uuid": str(uuid),
                        "pci_bus_id": str(pci_bus_id),
                        "torch_cuda_device_count": 1,
                        "torch_current_device": 0,
                        "used_bytes": int(info.used),
                        "total_bytes": int(info.total),
                        "vllm_tuned_config_folder": actual_folder,
                        "config_source": config_source,
                        "config_content_sha256": config_sha256,
                        "vllm_fused_moe_file_sha256": file_sha256(
                            Path(fused_moe.__file__).resolve()
                        ),
                        "vllm_envs_file_sha256": file_sha256(
                            Path(vllm_envs.__file__).resolve()
                        ),
                    }
                finally:
                    pynvml.nvmlShutdown()

            def generate_profiled_v28a(self, *args, **kwargs):
                import threading

                import pynvml
                import torch

                physical, _raw_ray_id = self._physical_gpu_v28a()
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(physical)
                stop = threading.Event()
                samples = []
                sample_failure = []

                def monitor():
                    try:
                        while not stop.is_set():
                            samples.append(int(
                                pynvml.nvmlDeviceGetMemoryInfo(handle).used
                            ))
                            stop.wait(0.01)
                    except BaseException as error:
                        sample_failure.append(type(error).__name__)

                thread = threading.Thread(target=monitor, daemon=True)
                torch.cuda.reset_peak_memory_stats(0)
                thread.start()
                try:
                    torch.cuda.synchronize()
                    started = time.perf_counter_ns()
                    outputs = super().generate(*args, **kwargs)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter_ns() - started
                finally:
                    stop.set()
                    thread.join()
                    samples.append(int(pynvml.nvmlDeviceGetMemoryInfo(handle).used))
                    pynvml.nvmlShutdown()
                if sample_failure or not samples or elapsed <= 0:
                    raise RuntimeError("V28A actor timing or NVML sampling failed")
                return {
                    "outputs": outputs,
                    "elapsed_ns": int(elapsed),
                    "peak_nvml_used_bytes": max(samples),
                    "peak_torch_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(0)
                    ),
                    "peak_torch_reserved_bytes": int(
                        torch.cuda.max_memory_reserved(0)
                    ),
                    "nvml_sample_count": len(samples),
                }

        if self.default_empty_directory.exists() is not True:
            raise RuntimeError("V28A fresh default directory is missing")
        if any(self.default_empty_directory.iterdir()):
            raise RuntimeError("V28A fresh default directory is not empty")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        for key in ("RAY_ADDRESS", "RAY_HEAD_IP", "RAY_GCS_SERVER_ADDRESS"):
            os.environ.pop(key, None)
        ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)
        pgs = [
            placement_group([{"GPU": 1, "CPU": 0}], strategy="PACK")
            for _ in prereg.PHYSICAL_GPU_IDS_V28A
        ]
        self.placement_groups = pgs
        self._ray = ray
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
            ray.remote(
                num_cpus=0, num_gpus=1, scheduling_strategy=strategy,
            )(PlacementProbeV28A).remote()
            for strategy in strategies
        ]
        identities = ray.get([probe.identity_v28a.remote() for probe in probes])
        strategies_by_gpu = {}
        for strategy, probe, identity in zip(strategies, probes, identities, strict=True):
            physical = identity.get("ray_gpu_id_canonical")
            if (
                physical in strategies_by_gpu
                or identity.get("cuda_visible_devices") != str(physical)
            ):
                raise RuntimeError("V28A placement probes do not cover unique GPUs")
            strategies_by_gpu[physical] = strategy
            ray.kill(probe)
        if set(strategies_by_gpu) != set(prereg.PHYSICAL_GPU_IDS_V28A):
            raise RuntimeError("V28A placement probes do not cover GPUs 0..3")
        self.placement_probe_identities = identities
        table = json.loads(prereg.TUNED_TABLE_PATH_V28A.read_text(encoding="utf-8"))
        table.pop("triton_version")
        expected_tuned_sha = canonical_sha256({int(key): item for key, item in table.items()})
        engines = []
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V28A:
            spec = self.assignments[str(gpu_id)]
            folder = self._folder_for_arm(spec["arm"])
            actor = ray.remote(
                num_cpus=0, num_gpus=1,
                scheduling_strategy=strategies_by_gpu[gpu_id],
            )(ProfiledESNcclLLMV28A).options(runtime_env={
                "env_vars": {"VLLM_TUNED_CONFIG_FOLDER": str(folder)},
            }).remote(
                model=spec["model_path"],
                probe_cell=spec["cell"],
                probe_wave=self.wave,
                probe_arm=spec["arm"],
                expected_model_path=spec["model_path"],
                expected_config_folder=str(folder),
                expected_tuned_config_content_sha256=expected_tuned_sha,
                tensor_parallel_size=1,
                dtype="bfloat16",
                enable_prefix_caching=False,
                enforce_eager=True,
                gpu_memory_utilization=0.82,
                max_model_len=2048,
                limit_mm_per_prompt={"image": 0, "video": 0},
                mm_processor_cache_gb=0,
                skip_mm_profiling=True,
                moe_backend="triton",
            )
            engines.append(actor)
        self.engines = engines

    def close(self):
        if self._ray is None:
            return
        from ray.util.placement_group import remove_placement_group
        for engine in self.engines:
            try:
                self._ray.kill(engine)
            except BaseException:
                pass
        for group in self.placement_groups:
            try:
                remove_placement_group(group)
            except BaseException:
                pass
        self._ray.shutdown()

    def _resolve(self, handles):
        return self._ray.get(handles)

    @staticmethod
    def prepare_train_requests_v28a():
        from transformers import AutoTokenizer

        bundle = runtime_v23a.mechanics_v23a.load_panel_bundle_v23a()
        runtime_v23a.anchor_v13.validate_panel_bundle_v13(bundle)
        if bundle["content_sha256_before_self_field"] != (
            prereg.PANEL_BUNDLE_CONTENT_SHA256_V28A
        ):
            raise RuntimeError("V28A frozen train-only panel bundle changed")
        tokenizer = AutoTokenizer.from_pretrained(prereg.MODEL_PATH_V28A)
        panels = {}
        requests = []
        token_audit = {}
        cursor = 0
        for panel_name in prereg.PANEL_NAMES_V28A:
            panel = bundle["panels"][panel_name]
            prompts = [
                runtime_v23a.base.specialist_template(item)
                for item in panel["questions"]
            ]
            dense = runtime_v23a.anchor_v4.prepare_gold_answer_items_v4(
                tokenizer, prompts, panel["answers"]
            )
            if len(dense) != 56:
                raise RuntimeError("V28A train panel request count changed")
            panels[panel_name] = {"slice": (cursor, cursor + 56), "dense": dense}
            requests.extend({"prompt_token_ids": item["prompt_token_ids"]}
                            for item in dense)
            cursor += 56
            token_audit[panel_name] = {
                "request_count": 56,
                "combined_token_count": sum(
                    len(item["prompt_token_ids"]) for item in dense
                ),
                "answer_token_count": sum(
                    item["answer_token_count"] for item in dense
                ),
                "maximum_combined_tokens": max(
                    len(item["prompt_token_ids"]) for item in dense
                ),
                "dense_contract_sha256": canonical_sha256(dense),
            }
        if cursor != prereg.REQUESTS_PER_ENGINE_V28A:
            raise RuntimeError("V28A aggregate request count changed")
        processed_tokens = sum(
            len(item["prompt_token_ids"]) + 1 for item in requests
        )
        audit = _seal({
            "schema": "eggroll-es-v28a-token-request-audit",
            "request_count": len(requests),
            "processed_tokens_per_engine_call": processed_tokens,
            "request_identity_sha256": canonical_sha256(requests),
            "panel_token_audit_sha256": canonical_sha256(token_audit),
            "all_requests_within_1024_token_cap": all(
                item["maximum_combined_tokens"] <= 1024
                for item in token_audit.values()
            ),
            "raw_train_content_or_token_ids_persisted": False,
        })
        return panels, requests, processed_tokens, audit

    @staticmethod
    def _sampling_params():
        from vllm import SamplingParams
        return SamplingParams(
            n=1, seed=43, temperature=0.0, top_p=1.0, max_tokens=1,
            prompt_logprobs=1, logprobs=1, detokenize=True,
        )

    def _generate(self, requests, *, profiled):
        if profiled:
            reports = self._resolve([
                engine.generate_profiled_v28a.remote(
                    requests, self._sampling_params(), use_tqdm=False
                ) for engine in self.engines
            ])
            batches = [item.get("outputs") for item in reports]
        else:
            reports = None
            batches = self._resolve([
                engine.generate.remote(
                    requests, self._sampling_params(), use_tqdm=False
                ) for engine in self.engines
            ])
        if len(batches) != 4 or any(
            not isinstance(batch, list) or len(batch) != len(requests)
            for batch in batches
        ):
            raise RuntimeError("V28A all-four-actor generation incomplete")
        return batches, reports

    @staticmethod
    def _generated_contract(batch):
        token_ids = []
        texts = []
        selected_logprobs = []
        cumulative_logprobs = []
        decoded_tokens = []
        for output in batch:
            candidates = getattr(output, "outputs", None)
            candidate = candidates[0] if candidates and len(candidates) == 1 else None
            ids = getattr(candidate, "token_ids", None)
            logprobs = getattr(candidate, "logprobs", None)
            text = getattr(candidate, "text", None)
            cumulative = getattr(candidate, "cumulative_logprob", None)
            if (
                candidate is None or not isinstance(ids, (list, tuple)) or len(ids) != 1
                or not isinstance(ids[0], int)
                or not isinstance(logprobs, list) or len(logprobs) != 1
                or not isinstance(logprobs[0], dict) or ids[0] not in logprobs[0]
                or not isinstance(text, str)
                or isinstance(cumulative, bool)
                or not isinstance(cumulative, (int, float))
            ):
                raise RuntimeError("V28A generated output contract changed")
            selected = logprobs[0][ids[0]]
            selected_value = getattr(selected, "logprob", None)
            decoded = getattr(selected, "decoded_token", None)
            if (
                isinstance(selected_value, bool)
                or not isinstance(selected_value, (int, float))
                or not math.isfinite(float(selected_value))
                or not math.isfinite(float(cumulative))
                or decoded is not None and not isinstance(decoded, str)
            ):
                raise RuntimeError("V28A generated logprob contract changed")
            token_ids.append([ids[0]])
            texts.append(text)
            selected_logprobs.append(float(selected_value))
            cumulative_logprobs.append(float(cumulative))
            decoded_tokens.append(decoded)
        return {
            "generated_token_ids": token_ids,
            "generated_text": texts,
            "generated_selected_logprobs": np.asarray(
                selected_logprobs, dtype=np.float64
            ),
            "generated_cumulative_logprobs": np.asarray(
                cumulative_logprobs, dtype=np.float64
            ),
            "generated_decoded_tokens": decoded_tokens,
        }

    def _score(self, panels, batches):
        result = {}
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V28A:
            cell = self.assignments[str(gpu_id)]["cell"]
            gold = np.empty((5, 56), dtype=np.float64)
            dense_commitments = []
            for panel_index, panel_name in enumerate(prereg.PANEL_NAMES_V28A):
                panel = panels[panel_name]
                start, stop = panel["slice"]
                reward, digest = runtime_v23a._score_panel_outputs_v23a(
                    panel["dense"], batches[gpu_id][start:stop]
                )
                gold[panel_index] = reward
                dense_commitments.append(digest)
            generated = self._generated_contract(batches[gpu_id])
            result[cell] = {
                "gold_mean_logprobs": gold,
                "gold_dense_commitments": dense_commitments,
                **generated,
            }
        return result

    def run_wave(self, panels, requests):
        identities = self._resolve([
            engine.runtime_identity_v28a.remote() for engine in self.engines
        ])
        table = json.loads(prereg.TUNED_TABLE_PATH_V28A.read_text(encoding="utf-8"))
        table.pop("triton_version")
        expected_tuned_sha = canonical_sha256({int(key): item for key, item in table.items()})
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V28A:
            item = identities[gpu_id]
            spec = self.assignments[str(gpu_id)]
            folder = self._folder_for_arm(spec["arm"])
            expected_source = (
                "generic_fallback_none" if spec["arm"] == "default_empty"
                else "exact_committed_v27c_table"
            )
            expected_config_sha = None if spec["arm"] == "default_empty" else expected_tuned_sha
            if (
                item.get("schema") != "eggroll-es-v28a-actor-runtime-identity"
                or item.get("cell") != spec["cell"]
                or item.get("wave") != self.wave
                or item.get("arm") != spec["arm"]
                or item.get("model_path") != spec["model_path"]
                or item.get("physical_gpu_id") != gpu_id
                or item.get("ray_gpu_id_canonical") != gpu_id
                or normalize_ray_gpu_id_v28a(item.get("ray_gpu_id_raw")) != gpu_id
                or item.get("cuda_visible_devices") != str(gpu_id)
                or item.get("vllm_tuned_config_folder") != str(folder)
                or item.get("config_source") != expected_source
                or item.get("config_content_sha256") != expected_config_sha
                or item.get("vllm_fused_moe_file_sha256")
                != VLLM_FUSED_MOE_SHA256_V28A
                or item.get("vllm_envs_file_sha256") != VLLM_ENVS_SHA256_V28A
                or item.get("torch_cuda_device_count") != 1
                or item.get("torch_current_device") != 0
            ):
                raise RuntimeError("V28A actor/model/config/physical-GPU mapping changed")
        warmup_commitments = []
        for _ in range(prereg.WARMUP_REPETITIONS_V28A):
            warmup_batches, _ = self._generate(requests, profiled=False)
            warmup = self._score(panels, warmup_batches)
            warmup_commitments.append(canonical_sha256({
                cell: output_commitment_v28a(value)
                for cell, value in warmup.items()
            }))
        resident_reports = self._resolve([
            engine.runtime_identity_v28a.remote() for engine in self.engines
        ])
        resident = {
            self.assignments[str(gpu_id)]["cell"]: {
                "used_bytes": resident_reports[gpu_id]["used_bytes"],
                "total_bytes": resident_reports[gpu_id]["total_bytes"],
            }
            for gpu_id in prereg.PHYSICAL_GPU_IDS_V28A
        }
        reference_batches, _ = self._generate(requests, profiled=False)
        reference = self._score(panels, reference_batches)
        repeat_batches, _ = self._generate(requests, profiled=False)
        repeat = self._score(panels, repeat_batches)
        for cell in reference:
            if not exact_output_equal_v28a(reference[cell], repeat[cell]):
                raise RuntimeError("V28A within-cell deterministic repeat changed")
        for arm in prereg.ARM_ORDER_V28A:
            matching = [
                self.assignments[str(gpu_id)]["cell"]
                for gpu_id in prereg.PHYSICAL_GPU_IDS_V28A
                if self.assignments[str(gpu_id)]["arm"] == arm
            ]
            if len(matching) != 2 or not exact_output_equal_v28a(
                reference[matching[0]], reference[matching[1]]
            ):
                raise RuntimeError("V28A same-arm duplicate cells disagree within wave")
        elapsed = np.empty((prereg.TIMING_REPETITIONS_V28A, 4), dtype=np.int64)
        peak_nvml = np.empty_like(elapsed)
        timed_output_commitments = []
        torch_profile_commitments = []
        for repetition in range(prereg.TIMING_REPETITIONS_V28A):
            batches, reports = self._generate(requests, profiled=True)
            observed = self._score(panels, batches)
            for gpu_id in prereg.PHYSICAL_GPU_IDS_V28A:
                cell = self.assignments[str(gpu_id)]["cell"]
                if not exact_output_equal_v28a(observed[cell], reference[cell]):
                    raise RuntimeError("V28A timed output differs from exact reference")
                report = reports[gpu_id]
                elapsed[repetition, gpu_id] = report["elapsed_ns"]
                peak_nvml[repetition, gpu_id] = report["peak_nvml_used_bytes"]
                torch_profile_commitments.append({
                    "cell": cell,
                    "repetition": repetition,
                    "allocated": report["peak_torch_allocated_bytes"],
                    "reserved": report["peak_torch_reserved_bytes"],
                    "nvml_sample_count": report["nvml_sample_count"],
                })
            timed_output_commitments.append(canonical_sha256({
                cell: output_commitment_v28a(value)
                for cell, value in observed.items()
            }))
        return {
            "wave": self.wave,
            "placement_probe_identities": self.placement_probe_identities,
            "actor_identities": identities,
            "reference": reference,
            "resident": resident,
            "elapsed_ns": elapsed,
            "peak_nvml": peak_nvml,
            "warmup_commitments": warmup_commitments,
            "timed_output_commitments": timed_output_commitments,
            "torch_profile_commitments": torch_profile_commitments,
            "within_wave_integrity": {
                "placement_groups_mapped_by_probe_not_creation_order": True,
                "exact_four_actor_physical_gpu_mapping": True,
                "exact_model_path_and_config_source_per_cell": True,
                "two_compile_warmup_calls_excluded": True,
                "within_cell_repeat_exact": True,
                "same_arm_duplicate_cells_exact_within_wave": True,
                "all_nine_timed_outputs_exact": True,
                "all_timing_and_nvml_profiles_complete": True,
            },
        }


def run_counterbalanced_probe_v28a(
    preregistration, prelaunch_gpu_idle_certificate, default_empty_directory,
):
    baseline_physical = _physical_identity_map_v28a(
        prelaunch_gpu_idle_certificate
    )
    panels, requests, processed_tokens, token_audit = (
        V28ProbeController.prepare_train_requests_v28a()
    )
    wave_results = {}
    interwave_idle_certificate = None
    final_idle_certificate = None
    for wave_index, wave in enumerate(prereg.WAVE_ORDER_V28A):
        controller = V28ProbeController(
            preregistration, wave, default_empty_directory
        )
        failure = None
        result = None
        try:
            controller.launch_engines()
            result = controller.run_wave(panels, requests)
        except BaseException as error:
            failure = error
        finally:
            try:
                controller.close()
            except BaseException as cleanup_error:
                if failure is None:
                    failure = cleanup_error
        if failure is not None:
            raise failure
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V28A:
            identity = result["actor_identities"][gpu_id]
            expected = baseline_physical[gpu_id]
            if any(
                identity.get(key) != expected[key]
                for key in ("nvml_uuid", "pci_bus_id", "total_bytes")
            ):
                raise RuntimeError(
                    "V28A actor does not match prelaunch physical-GPU identity"
                )
        wave_results[wave] = result
        if wave_index == 0:
            interwave_idle_certificate = wait_for_interwave_gpu_idle_v28a(
                prelaunch_gpu_idle_certificate
            )
            if _physical_identity_map_v28a(
                interwave_idle_certificate
            ) != baseline_physical:
                raise RuntimeError("V28A interwave physical-GPU identity changed")
        elif wave_index == 1:
            final_idle_certificate = wait_for_final_gpu_idle_v28a(
                prelaunch_gpu_idle_certificate
            )
            if _physical_identity_map_v28a(
                final_idle_certificate
            ) != baseline_physical:
                raise RuntimeError("V28A final physical-GPU identity changed")

    arm_appearances = {arm: [] for arm in prereg.ARM_ORDER_V28A}
    for wave in prereg.WAVE_ORDER_V28A:
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V28A:
            spec = preregistration["waves"][wave]["gpu_assignments"][str(gpu_id)]
            arm_appearances[spec["arm"]].append((wave, spec["cell"]))
    for arm, appearances in arm_appearances.items():
        if len(appearances) != 4:
            raise RuntimeError("V28A same-arm crossover appearance count changed")
        reference_wave, reference_cell = appearances[0]
        expected = wave_results[reference_wave]["reference"][reference_cell]
        for wave, cell in appearances[1:]:
            if not exact_output_equal_v28a(
                expected, wave_results[wave]["reference"][cell]
            ):
                raise RuntimeError(
                    f"V28A same-arm outputs disagree across waves: {arm}"
                )

    equivalence = {}
    for pair_name in prereg.PAIR_ORDER_V28A:
        spec = preregistration["pairing"]["physical_gpu_pairs"][pair_name]
        reference = wave_results[spec["reference_wave"]]["reference"][
            spec["reference_cell"]
        ]
        candidate = wave_results[spec["candidate_wave"]]["reference"][
            spec["candidate_cell"]
        ]
        components = output_equivalence_components_v28a(reference, candidate)
        exact = all(components.values())
        equivalence[pair_name] = {
            "physical_gpu_id": spec["physical_gpu_id"],
            "load_order": spec["load_order"],
            "exact_output_equivalence": exact,
            **components,
            "reference_commitment_sha256": output_commitment_v28a(reference),
            "candidate_commitment_sha256": output_commitment_v28a(candidate),
        }

    elapsed = np.stack([
        wave_results[wave]["elapsed_ns"] for wave in prereg.WAVE_ORDER_V28A
    ])
    peak_nvml = np.stack([
        wave_results[wave]["peak_nvml"] for wave in prereg.WAVE_ORDER_V28A
    ])
    resident = {}
    for wave in prereg.WAVE_ORDER_V28A:
        resident.update(wave_results[wave]["resident"])
    performance = performance_summary_v28a(
        elapsed, processed_tokens, resident, peak_nvml
    )
    runtime_integrity = {
        "exact_positive_evidence_commit_bound": True,
        "exact_bf16_qwen36_model_all_cells": True,
        "exact_two_wave_counterbalanced_arm_mapping": True,
        "placement_groups_mapped_by_string_int_canonicalized_probe": True,
        "all_four_physical_gpus_active_in_each_wave": True,
        "same_nvml_uuid_pci_and_total_memory_per_gpu_across_waves": True,
        "fresh_actor_engine_model_and_config_environment_each_wave": True,
        "bounded_interwave_cleanup_poll_completed": (
            interwave_idle_certificate.get("bounded_async_cleanup_wait") is True
        ),
        "interwave_cleanup_poll_count": interwave_idle_certificate.get("poll_count"),
        "interwave_cleanup_elapsed_milliseconds": (
            interwave_idle_certificate.get("elapsed_milliseconds")
        ),
        "bounded_final_async_cleanup_poll_completed": (
            final_idle_certificate.get("bounded_async_cleanup_wait") is True
        ),
        "final_cleanup_poll_count": final_idle_certificate.get("poll_count"),
        "final_cleanup_elapsed_milliseconds": (
            final_idle_certificate.get("elapsed_milliseconds")
        ),
        "all_four_gpus_idle_after_wave_b_cleanup": (
            final_idle_certificate.get("all_four_idle") is True
        ),
        "empty_default_generic_fallback_verified_all_four_appearances": True,
        "exact_committed_v27c_table_verified_all_four_appearances": True,
        "only_tuned_config_folder_differs_between_arms": True,
        "single_driver_materialization_and_tokenization_all_eight_cells": True,
        "model_load_compile_warmup_and_teardown_excluded": True,
        "within_cell_and_same_arm_outputs_exact": True,
        "cross_arm_output_equivalence_assessed_only_by_behavioral_gate": True,
        "all_nine_timed_outputs_exact_in_all_eight_cells": True,
        "update_checkpoint_evaluation_adoption_and_promotion_surfaces_closed": True,
    }
    runtime_integrity["all_integrity_audits_passed"] = all(
        value is True
        for value in runtime_integrity.values()
        if isinstance(value, bool)
    )
    gate = evaluate_gate_v28a(equivalence, performance, runtime_integrity)
    compact_waves = {}
    for wave in prereg.WAVE_ORDER_V28A:
        result = wave_results[wave]
        compact_waves[wave] = {
            "placement_probe_identity_sha256": canonical_sha256(
                result["placement_probe_identities"]
            ),
            "actor_identity_sha256": canonical_sha256(result["actor_identities"]),
            "reference_output_commitments_sha256": canonical_sha256({
                cell: output_commitment_v28a(value)
                for cell, value in result["reference"].items()
            }),
            "warmup_commitments_sha256": canonical_sha256(
                result["warmup_commitments"]
            ),
            "timed_output_commitments_sha256": canonical_sha256(
                result["timed_output_commitments"]
            ),
            "torch_profile_commitments_sha256": canonical_sha256(
                result["torch_profile_commitments"]
            ),
            "within_wave_integrity": result["within_wave_integrity"],
        }
    audit = _seal({
        "schema": "eggroll-es-v28a-compact-runtime-audit",
        "token_request_audit": token_audit,
        "waves": compact_waves,
        "interwave_idle_certificate": interwave_idle_certificate,
        "final_idle_certificate": final_idle_certificate,
        "runtime_integrity": runtime_integrity,
        "generation_call_count_per_engine_per_wave": 13,
        "total_engine_generation_calls_all_waves": 104,
        "total_generation_requests_all_engines_all_waves": 29_120,
        "raw_rows_prompts_answers_token_ids_text_logprobs_timings_memory_samples_or_bootstrap_replicates_persisted": False,
    })
    summary = _seal({
        "schema": "eggroll-es-v28a-task-runtime-summary",
        "equivalence": equivalence,
        "performance": performance,
        "gate": gate,
        "runtime_integrity": runtime_integrity,
        "direct_recipe_adoption_or_training_action_taken": False,
    })
    _assert_compact_v28a({"summary": summary, "audit": audit})
    return summary, audit


def _parser_v28a():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v28a-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-recipe-sha256")
    return parser


def run_exact_v28a(preregistration, implementation, recipe):
    environment = runtime_r2.certify_runtime_environment_r2()
    provenance = runtime_v23a._source_provenance_v23a(implementation)
    live_audit = live_cpu_disk_audit_v28a()
    attempt_path = OUTPUT_DIRECTORY_V28A / ATTEMPT_NAME_V28A
    run_directory = OUTPUT_DIRECTORY_V28A / EXPERIMENT_NAME_V28A
    report_path = run_directory / REPORT_NAME_V28A
    default_empty_directory = run_directory / DEFAULT_EMPTY_DIRECTORY_NAME_V28A
    if attempt_path.exists() or run_directory.exists():
        raise RuntimeError("V28A requires fresh exclusive attempt and run paths")
    gpu_idle = assert_all_four_gpus_idle_v28a()
    attempt = {
        "schema": "eggroll-es-v28a-durable-launch-attempt",
        "status": "launching",
        "phase": "before_fresh_empty_directory_and_actor_creation",
        "recipe": recipe,
        "source_provenance": provenance,
        "runtime_environment_certificate": environment,
        "live_cpu_disk_audit": live_audit,
        "gpu_idle_certificate": gpu_idle,
        "model_update_applied": False,
        "checkpoint_written": False,
        "evaluation_opened": False,
        "direct_recipe_adoption_applied": False,
        "dataset_promotion_applied": False,
    }
    _assert_compact_v28a(attempt)
    runtime_v23a._exclusive_write_json_v23a(attempt_path, attempt)
    failure = None
    result = None
    try:
        run_directory.mkdir(parents=True, exist_ok=False)
        default_empty_directory.mkdir(exist_ok=False)
        if any(default_empty_directory.iterdir()):
            raise RuntimeError("V28A dedicated default directory is not empty")
        result = run_counterbalanced_probe_v28a(
            preregistration, gpu_idle, default_empty_directory
        )
    except BaseException as error:
        failure = error
    if failure is not None:
        attempt.update({
            "status": "failed",
            "phase": "inside_v28a_train_only_runtime_ab",
            "failure_type": type(failure).__name__,
            "failure_sha256": canonical_sha256({
                "type": type(failure).__name__, "repr": repr(failure),
            }),
            "report_exists_after_attempt": report_path.exists(),
        })
        _assert_compact_v28a(attempt)
        runtime_v23a._rewrite_json_v23a(attempt_path, attempt)
        raise failure
    summary, runtime_audit = result
    report = _seal({
        "schema": "eggroll-es-v28a-task-runtime-report",
        "recipe": recipe,
        "summary": summary,
        "runtime_audit": runtime_audit,
        "implementation": implementation,
        "model_update_applied": False,
        "checkpoint_written": False,
        "evaluation_opened": False,
        "direct_recipe_adoption_applied": False,
        "dataset_promotion_applied": False,
        "direct_action_taken": False,
    })
    _assert_compact_v28a(report)
    runtime_v23a._exclusive_write_json_v23a(report_path, report)
    attempt.update({
        "status": "complete",
        "phase": "after_actor_cleanup_and_compact_report",
        "report_binding": {
            "path": str(report_path),
            "file_sha256": file_sha256(report_path),
            "content_sha256": report["content_sha256_before_self_field"],
        },
    })
    runtime_v23a._rewrite_json_v23a(attempt_path, attempt)
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    runtime_v23a._assert_train_only_argv_v23a(argv)
    args = _parser_v28a().parse_args(argv)
    preregistration = load_preregistration_v28a()
    implementation = implementation_identity_v28a()
    recipe = recipe_v28a(preregistration, implementation)
    validate_runtime_v28a(args, preregistration, implementation, recipe)
    if args.v28a_dry_run:
        payload = _seal({
            "schema": "eggroll-es-v28a-task-runtime-dry-run",
            "recipe": recipe,
            "implementation": implementation,
            "positive_evidence_commit": prereg.POSITIVE_EVIDENCE_COMMIT_V28A,
            "gpu_launched": False,
            "gpu_idle_check_executed": False,
            "dataset_rows_or_tokens_opened": False,
            "future_real_launch_requires_exact_post_commit_hashes": True,
            "future_real_launch_requires_immediate_preclaim_all_gpu_idle": True,
            "pass_authority_limited_to_separate_frozen_train_only_recipe_ab": True,
            "adoption_update_checkpoint_evaluation_or_promotion_authorized": False,
            "raw_train_or_output_content_persisted": False,
        })
        _assert_compact_v28a(payload)
        print(json.dumps(payload, sort_keys=True))
        return payload
    return run_exact_v28a(preregistration, implementation, recipe)


if __name__ == "__main__":
    main()
