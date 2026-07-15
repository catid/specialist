#!/usr/bin/env python3
"""Fail-closed future four-GPU train-only FP8-versus-V26 probe launcher."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

import audit_qwen36_fp8_routed_experts_bf16_backbone_v26 as audit_v26
import build_qwen36_fp8_routed_experts_bf16_backbone_v26 as builder_v26
import eggroll_es_v26_compat_speed_preregistration_v26a as prereg
import run_eggroll_es_insertion_stability_v23a as runtime_v23a
import run_eggroll_es_insertion_stability_v23a_retry_r2 as runtime_r2


ROOT = Path(__file__).resolve().parent
PREREG_PATH = prereg.OUTPUT_PATH_V26A
PREREG_FILE_SHA256 = "ffe596314299b13e0691ee3eaea8df0f123dd06a767b768edeab5c0a6bf6a4cf"
PREREG_CONTENT_SHA256 = "d8b3cc6f5837606f43c0fa26841b223a974d93fb4117c91fde553dae67af4767"
EXPERIMENT_NAME = "s6_v26a_fp8_vs_routed_experts_bf16_backbone_train_only_probe"
OUTPUT_DIRECTORY = (ROOT / "experiments/eggroll_es_hpo/runs").resolve()
ATTEMPT_NAME = f".{EXPERIMENT_NAME}.launch_attempt.json"
REPORT_NAME = "fp8_vs_v26_compat_speed_report_v26a.json"
TEST_PATH = (ROOT / "test_run_eggroll_es_v26_compat_speed_v26a.py").resolve()
PREREG_TEST_PATH = (
    ROOT / "test_eggroll_es_v26_compat_speed_preregistration_v26a.py"
).resolve()
IMPLEMENTATION_PATHS = {
    "v26_builder": Path(builder_v26.__file__).resolve(),
    "v26_builder_tests": ROOT / "test_build_qwen36_fp8_routed_experts_bf16_backbone_v26.py",
    "v26_auditor": Path(audit_v26.__file__).resolve(),
    "v26_auditor_tests": ROOT / "test_audit_qwen36_fp8_routed_experts_bf16_backbone_v26.py",
    "v26a_preregistration_module": Path(prereg.__file__).resolve(),
    "v26a_preregistration_tests": PREREG_TEST_PATH,
    "v26a_preregistration": PREREG_PATH,
    "v26a_runtime": Path(__file__).resolve(),
    "v26a_runtime_tests": TEST_PATH,
}
FORBIDDEN_PERSISTED_KEYS = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "prompt_token_ids", "token_ids", "outputs", "responses", "scores",
    "score_arrays", "elapsed_ns", "timing_vectors", "memory_samples",
    "bootstrap_draws", "bootstrap_replicates", "row_content", "row_sha256",
}


canonical_sha256 = prereg.canonical_sha256
file_sha256 = prereg.file_sha256
_seal = runtime_v23a._seal_v23a


def normalize_ray_gpu_id_v26a(value):
    if isinstance(value, bool):
        raise RuntimeError("V26A Ray GPU ID representation changed")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value in {"0", "1", "2", "3"}:
        result = int(value)
    else:
        raise RuntimeError("V26A Ray GPU ID representation changed")
    if result not in prereg.PHYSICAL_GPU_IDS_V26A:
        raise RuntimeError("V26A Ray GPU ID physical range changed")
    return result


def _without_self(value):
    return {
        key: item
        for key, item in value.items()
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


def _assert_compact(value):
    overlap = FORBIDDEN_PERSISTED_KEYS & set(_recursive_keys(value))
    if overlap:
        raise RuntimeError(f"V26A compact output contains forbidden keys: {sorted(overlap)}")


def load_preregistration_v26a():
    value = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    if (
        file_sha256(PREREG_PATH) != PREREG_FILE_SHA256
        or value.get("content_sha256_before_self_field") != PREREG_CONTENT_SHA256
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(value))
    ):
        raise RuntimeError("V26A preregistration identity changed")
    value["waves"] = {
        wave: value["waves"][wave] for wave in prereg.WAVE_ORDER_V26A
    }
    return prereg.validate_preregistration_v26a(value)


def implementation_identity_v26a():
    inherited = runtime_r2.implementation_identity_r2()
    files = dict(inherited["files"])
    for key, path in IMPLEMENTATION_PATHS.items():
        files[key] = {
            "path": str(Path(path).resolve()),
            "file_sha256": file_sha256(path),
        }
    return {
        "files": files,
        "inherited_v23a_r2_bundle_sha256": inherited["bundle_sha256"],
        "v26a_overlay_bundle_sha256": canonical_sha256({
            key: files[key] for key in IMPLEMENTATION_PATHS
        }),
        "bundle_sha256": canonical_sha256(files),
    }


def recipe_v26a(preregistration, implementation):
    value = {
        "schema": "eggroll-es-v26-model-compat-speed-runtime-recipe-v26a",
        "experiment_name": EXPERIMENT_NAME,
        "preregistration": {
            "path": str(PREREG_PATH),
            "file_sha256": PREREG_FILE_SHA256,
            "content_sha256": PREREG_CONTENT_SHA256,
        },
        "model_contract": copy.deepcopy(preregistration["model_contract"]),
        "waves": copy.deepcopy(preregistration["waves"]),
        "pairing": copy.deepcopy(preregistration["pairing"]),
        "train_request_contract": copy.deepcopy(
            preregistration["train_request_contract"]
        ),
        "sampling_contract": copy.deepcopy(preregistration["sampling_contract"]),
        "runtime_contract": copy.deepcopy(preregistration["runtime_contract"]),
        "equivalence_analysis": copy.deepcopy(
            preregistration["equivalence_analysis"]
        ),
        "speed_and_memory_analysis": copy.deepcopy(
            preregistration["speed_and_memory_analysis"]
        ),
        "gate": copy.deepcopy(preregistration["gate"]),
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "runtime_environment": {
            "certificate": "V23A retry R2 exact environment certificate required",
            "moe_backend": "triton",
            "external_moe_backend_overrides_rejected": True,
            "batch_invariant_backend_swap_rejected": True,
        },
        "fresh_exclusive_paths": {
            "attempt": str(OUTPUT_DIRECTORY / ATTEMPT_NAME),
            "run_directory": str(OUTPUT_DIRECTORY / EXPERIMENT_NAME),
            "report_name": REPORT_NAME,
        },
        "real_launch_requires_all_four_gpus_idle_before_attempt_claim": True,
        "model_update_allowed": False,
        "checkpoint_write_allowed": False,
        "evaluation_allowed": False,
        "direct_model_adoption_allowed": False,
        "contains_validation_ood_heldout_or_benchmark_content": False,
    }
    return _seal(value)


def validate_runtime_v26a(args, preregistration, implementation, recipe):
    prereg.validate_preregistration_v26a(preregistration)
    if any(
        os.environ.get(key) for key in runtime_v23a.MOE_OVERRIDE_ENVIRONMENT_V23A
    ):
        raise ValueError("V26A rejects external MoE backend overrides")
    if os.environ.get("VLLM_BATCH_INVARIANT") not in (None, "0"):
        raise ValueError("V26A rejects the vLLM batch-invariant backend swap")
    if recipe.get("content_sha256_before_self_field") != canonical_sha256(
        _without_self(recipe)
    ):
        raise ValueError("V26A runtime recipe identity changed")
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
        if not args.v26a_dry_run and expected is None:
            raise ValueError(f"V26A real launch requires expected {label} hash")
        if expected is not None and expected != actual:
            raise ValueError(f"V26A {label} hash changed")


def live_model_audit_v26a(preregistration):
    exact = audit_v26.audit_hybrid_checkpoint_v26(
        builder_v26.DEFAULT_BF16,
        prereg.FP8_MODEL_V26A,
        prereg.HYBRID_MODEL_V26A,
    )
    if (
        exact.get("content_sha256_before_self_field")
        != prereg.EXPECTED_V26_AUDIT_CONTENT_SHA256
        or exact.get("contains_dataset_or_evaluation_content") is not False
    ):
        raise RuntimeError("V26A exact hybrid audit identity changed")
    fp8_index = json.loads(
        (prereg.FP8_MODEL_V26A / "model.safetensors.index.json").read_text(
            encoding="utf-8"
        )
    )
    fp8_weights = builder_v26._weight_shard_file_manifest(
        prereg.FP8_MODEL_V26A, fp8_index["weight_map"].values()
    )
    if fp8_weights != prereg.FP8_WEIGHT_SHARD_MANIFEST_V26A:
        raise RuntimeError("V26A reference FP8 shard identity changed")
    return _seal({
        "schema": "eggroll-es-v26a-live-model-audit",
        "fp8_weight_shard_manifest": fp8_weights,
        "hybrid_audit_content_sha256": exact[
            "content_sha256_before_self_field"
        ],
        "hybrid_target_key_count": exact["target_key_count"],
        "hybrid_target_byte_count": exact["target_byte_count"],
        "all_backbone_tensors_exact_bf16": True,
        "all_routed_expert_tensors_and_scales_exact_fp8": True,
        "non_routed_fp8_scales_present": False,
        "contains_dataset_or_evaluation_content": False,
    })


def _observe_all_four_gpus_v26a():
    import pynvml

    pynvml.nvmlInit()
    try:
        reports = []
        for gpu_id in range(4):
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
        "schema": "eggroll-es-v26a-four-gpu-observation",
        "gpu_count": 4,
        "gpus": reports,
        "all_four_idle": all(
            item["running_process_count"] == 0 for item in reports
        ),
    })


def assert_all_four_gpus_idle_v26a():
    observation = _observe_all_four_gpus_v26a()
    if observation.get("all_four_idle") is not True:
        raise RuntimeError("V26A launch blocked because a GPU has an existing owner")
    value = copy.deepcopy(observation)
    value["schema"] = "eggroll-es-v26a-all-gpu-idle-certificate"
    return _seal(value)


def wait_for_interwave_gpu_idle_v26a(
    expected_idle_certificate, *, timeout_seconds=30.0, interval_seconds=0.5,
):
    if timeout_seconds != 30.0 or interval_seconds != 0.5:
        raise RuntimeError("V26A interwave idle polling contract changed")
    expected = {
        item["physical_gpu_id"]: {
            "nvml_uuid": item.get("nvml_uuid"),
            "pci_bus_id": item.get("pci_bus_id"),
            "total_bytes": item.get("total_bytes"),
        }
        for item in expected_idle_certificate.get("gpus", [])
    }
    if set(expected) != set(prereg.PHYSICAL_GPU_IDS_V26A):
        raise RuntimeError("V26A interwave baseline GPU identity changed")
    started = time.monotonic()
    polls = 0
    while True:
        observation = _observe_all_four_gpus_v26a()
        polls += 1
        observed = {
            item["physical_gpu_id"]: {
                "nvml_uuid": item.get("nvml_uuid"),
                "pci_bus_id": item.get("pci_bus_id"),
                "total_bytes": item.get("total_bytes"),
            }
            for item in observation.get("gpus", [])
        }
        if observed != expected:
            raise RuntimeError("V26A interwave physical-GPU identity changed")
        elapsed = time.monotonic() - started
        if observation.get("all_four_idle") is True:
            return _seal({
                "schema": "eggroll-es-v26a-interwave-cleanup-idle-certificate",
                "gpu_count": 4,
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
            raise RuntimeError("V26A interwave GPU cleanup did not finish within 30 seconds")
        time.sleep(min(interval_seconds, remaining))


def equivalence_metrics_v26a(reference_scores, candidate_scores, reference_tokens, candidate_tokens):
    reference = np.asarray(reference_scores, dtype=np.float64)
    candidate = np.asarray(candidate_scores, dtype=np.float64)
    reference_tokens = np.asarray(reference_tokens, dtype=np.int64)
    candidate_tokens = np.asarray(candidate_tokens, dtype=np.int64)
    if (
        reference.shape != (5, 56)
        or candidate.shape != reference.shape
        or reference_tokens.shape != (280,)
        or candidate_tokens.shape != reference_tokens.shape
        or not np.isfinite(reference).all()
        or not np.isfinite(candidate).all()
    ):
        raise RuntimeError("V26A equivalence input geometry changed")
    left = reference.reshape(-1)
    right = candidate.reshape(-1)
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        raise RuntimeError("V26A equivalence correlation has zero variance")
    delta = right - left
    absolute = np.abs(delta)
    metrics = {
        "absolute_mean_signed_logprob_delta": float(abs(np.mean(delta))),
        "mean_absolute_logprob_delta": float(np.mean(absolute)),
        "root_mean_square_logprob_delta": float(np.sqrt(np.mean(delta * delta))),
        "p99_absolute_logprob_delta": float(np.quantile(
            absolute, 0.99, method="linear"
        )),
        "maximum_absolute_logprob_delta": float(np.max(absolute)),
        "row_logprob_pearson": float(np.corrcoef(left, right)[0, 1]),
        "greedy_first_token_agreement": float(np.mean(
            reference_tokens == candidate_tokens
        )),
    }
    thresholds = prereg.EQUIVALENCE_THRESHOLDS_V26A
    passed = (
        metrics["absolute_mean_signed_logprob_delta"]
        <= thresholds["absolute_mean_signed_logprob_delta_max"]
        and metrics["mean_absolute_logprob_delta"]
        <= thresholds["mean_absolute_logprob_delta_max"]
        and metrics["root_mean_square_logprob_delta"]
        <= thresholds["root_mean_square_logprob_delta_max"]
        and metrics["p99_absolute_logprob_delta"]
        <= thresholds["p99_absolute_logprob_delta_max"]
        and metrics["maximum_absolute_logprob_delta"]
        <= thresholds["maximum_absolute_logprob_delta_max"]
        and metrics["row_logprob_pearson"]
        >= thresholds["row_logprob_pearson_min"]
        and metrics["greedy_first_token_agreement"]
        >= thresholds["greedy_first_token_agreement_min"]
    )
    return {"metrics": metrics, "thresholds": dict(thresholds), "pass": passed}


def speed_memory_summary_v26a(elapsed_ns, processed_tokens, resident, peak_nvml):
    elapsed = np.asarray(elapsed_ns, dtype=np.int64)
    peaks = np.asarray(peak_nvml, dtype=np.int64)
    expected_cells = {
        f"{wave}_gpu_{gpu_id}"
        for wave in prereg.WAVE_ORDER_V26A
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V26A
    }
    if (
        elapsed.shape != (2, prereg.TIMING_REPETITIONS_V26A, 4)
        or peaks.shape != elapsed.shape
        or np.any(elapsed <= 0)
        or np.any(peaks <= 0)
        or isinstance(processed_tokens, bool)
        or not isinstance(processed_tokens, int)
        or processed_tokens <= 0
        or set(resident) != expected_cells
    ):
        raise RuntimeError("V26A speed or memory geometry changed")
    throughput = float(processed_tokens) / (elapsed.astype(np.float64) / 1e9)
    draws, draw_sha = prereg.timing_bootstrap_draw_plan_v26a()
    cell_summary = {}
    memory_summary = {}
    wave_summary = {}
    for wave_index, wave in enumerate(prereg.WAVE_ORDER_V26A):
        wave_values = throughput[wave_index]
        wave_summary[wave] = {
            "median_processed_tokens_per_second_all_gpus": float(
                np.median(wave_values)
            ),
            "minimum_processed_tokens_per_second_all_gpus": float(
                np.min(wave_values)
            ),
            "maximum_processed_tokens_per_second_all_gpus": float(
                np.max(wave_values)
            ),
        }
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V26A:
            cell = f"{wave}_gpu_{gpu_id}"
            values = throughput[wave_index, :, gpu_id]
            backend = prereg.WAVE_BACKEND_ORDER_V26A[wave][gpu_id]
            cell_summary[cell] = {
                "wave": wave,
                "physical_gpu_id": gpu_id,
                "backend": backend,
                "median_processed_tokens_per_second": float(np.median(values)),
                "minimum_processed_tokens_per_second": float(np.min(values)),
                "maximum_processed_tokens_per_second": float(np.max(values)),
                "coefficient_of_variation": float(
                    np.std(values) / np.mean(values)
                ),
                "timing_commitment_sha256": hashlib.sha256(
                    np.ascontiguousarray(elapsed[wave_index, :, gpu_id]).tobytes(
                        order="C"
                    )
                ).hexdigest(),
            }
            report = resident[cell]
            total = report.get("total_bytes")
            resident_bytes = report.get("used_bytes")
            timed_peak = int(np.max(peaks[wave_index, :, gpu_id]))
            if (
                isinstance(total, bool) or not isinstance(total, int) or total <= 0
                or isinstance(resident_bytes, bool)
                or not isinstance(resident_bytes, int)
                or not 0 < resident_bytes <= total
                or not 0 < timed_peak <= total
            ):
                raise RuntimeError("V26A resident or peak VRAM contract changed")
            observed_peak = max(resident_bytes, timed_peak)
            memory_summary[cell] = {
                "wave": wave,
                "physical_gpu_id": gpu_id,
                "backend": backend,
                "resident_used_bytes": resident_bytes,
                "timed_peak_nvml_used_bytes": timed_peak,
                "peak_resident_or_timed_nvml_used_bytes": observed_peak,
                "total_bytes": total,
                "peak_nvml_fraction": float(observed_peak / total),
                "within_preregistered_peak_fraction": bool(
                    observed_peak / total <= prereg.MAX_PEAK_NVML_FRACTION_V26A
                ),
            }
    pairs = {}
    order_ratios = {"fp8_first": [], "hybrid_first": []}
    for gpu_id, pair_name in zip(
        prereg.PHYSICAL_GPU_IDS_V26A, prereg.PAIR_ORDER_V26A, strict=True,
    ):
        fp8_wave_index = 0 if gpu_id in (0, 2) else 1
        hybrid_wave_index = 1 - fp8_wave_index
        ratios = (
            throughput[hybrid_wave_index, :, gpu_id]
            / throughput[fp8_wave_index, :, gpu_id]
        )
        replicates = np.median(ratios[draws], axis=1)
        observed = float(np.median(ratios))
        lcb = float(np.quantile(
            replicates,
            prereg.SPEED_FAMILYWISE_QUANTILE_V26A,
            method="linear",
        ))
        order = "fp8_first" if fp8_wave_index == 0 else "hybrid_first"
        order_ratios[order].extend(float(item) for item in ratios)
        fp8_cell = f"{prereg.WAVE_ORDER_V26A[fp8_wave_index]}_gpu_{gpu_id}"
        hybrid_cell = (
            f"{prereg.WAVE_ORDER_V26A[hybrid_wave_index]}_gpu_{gpu_id}"
        )
        memory_pass = all(
            memory_summary[cell]["within_preregistered_peak_fraction"] is True
            for cell in (fp8_cell, hybrid_cell)
        )
        pairs[pair_name] = {
            "physical_gpu_id": gpu_id,
            "load_order": order,
            "fp8_cell": fp8_cell,
            "hybrid_cell": hybrid_cell,
            "median_hybrid_over_fp8_throughput_ratio": observed,
            "familywise_lower_confidence_bound": lcb,
            "noninferiority_ratio": prereg.SPEED_NONINFERIORITY_RATIO_V26A,
            "speed_pass": lcb >= prereg.SPEED_NONINFERIORITY_RATIO_V26A,
            "both_cell_peak_vram_pass": memory_pass,
            "pass": bool(
                lcb >= prereg.SPEED_NONINFERIORITY_RATIO_V26A and memory_pass
            ),
        }
    return {
        "processed_tokens_per_engine_call": processed_tokens,
        "cells": cell_summary,
        "waves": wave_summary,
        "pairs": pairs,
        "memory": memory_summary,
        "load_order_strata": {
            order: {
                "physical_gpu_ids": [0, 2] if order == "fp8_first" else [1, 3],
                "median_hybrid_over_fp8_throughput_ratio": float(
                    np.median(values)
                ),
            }
            for order, values in order_ratios.items()
        },
        "bootstrap": {
            "seed": prereg.TIMING_BOOTSTRAP_SEED_V26A,
            "repetitions": prereg.TIMING_BOOTSTRAP_REPETITIONS_V26A,
            "draw_plan_sha256": draw_sha,
            "raw_draws_replicates_and_timing_vectors_persisted": False,
        },
    }


def evaluate_gate_v26a(equivalence, performance, runtime_integrity):
    expected_cells = {
        f"{wave}_gpu_{gpu_id}"
        for wave in prereg.WAVE_ORDER_V26A
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V26A
    }
    if (
        set(equivalence) != set(prereg.PAIR_ORDER_V26A)
        or set(performance.get("pairs", {})) != set(prereg.PAIR_ORDER_V26A)
        or set(performance.get("memory", {})) != expected_cells
        or runtime_integrity.get("all_integrity_audits_passed") is not True
    ):
        raise RuntimeError("V26A gate input contract changed")
    passed = (
        all(equivalence[pair]["pass"] is True for pair in prereg.PAIR_ORDER_V26A)
        and all(performance["pairs"][pair]["pass"] is True
                for pair in prereg.PAIR_ORDER_V26A)
        and all(performance["memory"][cell][
            "within_preregistered_peak_fraction"
        ] is True for cell in expected_cells)
    )
    return _seal({
        "schema": "eggroll-es-v26a-authorization-gate",
        "pass": passed,
        "decision": (
            "authorize_only_a_separate_fresh_preregistered_train_only_training_A_B"
            if passed else "retain_existing_FP8_model_for_any_later_training_A_B"
        ),
        "all_runtime_integrity_audits_passed": True,
        "all_four_equivalence_pairs_passed": all(
            equivalence[pair]["pass"] is True for pair in prereg.PAIR_ORDER_V26A
        ),
        "all_four_speed_and_memory_pairs_passed": all(
            performance["pairs"][pair]["pass"] is True
            for pair in prereg.PAIR_ORDER_V26A
        ),
        "all_eight_peak_vram_limits_passed": all(
            performance["memory"][cell]["within_preregistered_peak_fraction"] is True
            for cell in expected_cells
        ),
        "direct_model_adoption_authorized": False,
        "model_update_authorized": False,
        "checkpoint_write_authorized": False,
        "evaluation_authorized": False,
        "dataset_promotion_authorized": False,
    })


class V26ProbeController:
    def __init__(self, preregistration, wave):
        if wave not in prereg.WAVE_ORDER_V26A:
            raise RuntimeError("V26A crossover wave changed")
        self.preregistration = preregistration
        self.wave = wave
        self.assignments = preregistration["waves"][wave]["gpu_assignments"]
        self.engines = []
        self.placement_groups = []
        self._ray = None

    def launch_engines(self):
        import ray
        from es_at_scale.trainer.es_trainer import ESNcclLLM
        from ray.util.placement_group import placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        class PlacementProbeV26A:
            @staticmethod
            def identity_v26a():
                ids = ray.get_gpu_ids()
                visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                if not isinstance(ids, list) or len(ids) != 1:
                    raise RuntimeError("V26A placement probe GPU coverage changed")
                physical = normalize_ray_gpu_id_v26a(ids[0])
                if visible != str(physical):
                    raise RuntimeError("V26A placement probe visible GPU changed")
                return {
                    "ray_gpu_id_raw": ids[0],
                    "ray_gpu_id_canonical": physical,
                    "cuda_visible_devices": visible,
                }

        class ProfiledESNcclLLMV26A(ESNcclLLM):
            def __init__(
                self, *args, probe_cell, probe_wave, probe_backend,
                expected_model_path, **kwargs,
            ):
                self._v26a_probe_cell = str(probe_cell)
                self._v26a_probe_wave = str(probe_wave)
                self._v26a_probe_backend = str(probe_backend)
                self._v26a_expected_model_path = str(expected_model_path)
                super().__init__(*args, **kwargs)

            @staticmethod
            def _physical_gpu_v26a():
                token = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                ids = ray.get_gpu_ids()
                if not isinstance(ids, list) or len(ids) != 1:
                    raise RuntimeError("V26A actor requires one physical GPU token")
                physical = normalize_ray_gpu_id_v26a(ids[0])
                if token != str(physical):
                    raise RuntimeError("V26A actor Ray and visible GPU IDs disagree")
                return physical

            def runtime_identity_v26a(self):
                import pynvml
                import torch

                physical = self._physical_gpu_v26a()
                if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
                    raise RuntimeError("V26A actor does not own exactly one CUDA device")
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
                        "schema": "eggroll-es-v26a-actor-runtime-identity",
                        "cell": self._v26a_probe_cell,
                        "wave": self._v26a_probe_wave,
                        "backend": self._v26a_probe_backend,
                        "model_path": self._v26a_expected_model_path,
                        "physical_gpu_id": physical,
                        "nvml_uuid": str(uuid),
                        "pci_bus_id": str(pci_bus_id),
                        "cuda_visible_devices": str(physical),
                        "torch_cuda_device_count": 1,
                        "torch_current_device": 0,
                        "used_bytes": int(info.used),
                        "total_bytes": int(info.total),
                    }
                finally:
                    pynvml.nvmlShutdown()

            def generate_profiled_v26a(self, *args, **kwargs):
                import threading
                import time

                import pynvml
                import torch

                physical = self._physical_gpu_v26a()
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
                    raise RuntimeError("V26A actor timing or NVML sampling failed")
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

        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        for key in ("RAY_ADDRESS", "RAY_HEAD_IP", "RAY_GCS_SERVER_ADDRESS"):
            os.environ.pop(key, None)
        ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)
        pgs = [
            placement_group([{"GPU": 1, "CPU": 0}], strategy="PACK")
            for _ in range(4)
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
            )(PlacementProbeV26A).remote()
            for strategy in strategies
        ]
        identities = ray.get([
            probe.identity_v26a.remote() for probe in probes
        ])
        strategies_by_gpu = {}
        for strategy, probe, identity in zip(strategies, probes, identities, strict=True):
            physical = identity.get("ray_gpu_id_canonical")
            if (
                physical in strategies_by_gpu
                or identity.get("cuda_visible_devices") != str(physical)
            ):
                raise RuntimeError("V26A placement groups do not cover unique GPUs")
            strategies_by_gpu[physical] = strategy
            ray.kill(probe)
        if set(strategies_by_gpu) != set(prereg.PHYSICAL_GPU_IDS_V26A):
            raise RuntimeError("V26A placement groups do not cover GPUs 0..3")
        engines = []
        for rank in prereg.PHYSICAL_GPU_IDS_V26A:
            spec = self.assignments[str(rank)]
            engines.append(ray.remote(
                num_cpus=0, num_gpus=1,
                scheduling_strategy=strategies_by_gpu[rank],
            )(ProfiledESNcclLLMV26A).remote(
                model=spec["model_path"],
                probe_cell=spec["cell"],
                probe_wave=self.wave,
                probe_backend=spec["backend"],
                expected_model_path=spec["model_path"],
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
            ))
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

    def _prepare_train_requests(self):
        from transformers import AutoTokenizer

        bundle = runtime_v23a.mechanics_v23a.load_panel_bundle_v23a()
        runtime_v23a.anchor_v13.validate_panel_bundle_v13(bundle)
        if bundle["content_sha256_before_self_field"] != (
            prereg.PANEL_BUNDLE_CONTENT_SHA256_V26A
        ):
            raise RuntimeError("V26A train-only panel bundle identity changed")
        tokenizer = AutoTokenizer.from_pretrained(prereg.FP8_MODEL_V26A)
        panels = {}
        requests = []
        cursor = 0
        token_audit = {}
        for panel_name in prereg.PANEL_NAMES_V26A:
            panel = bundle["panels"][panel_name]
            prompts = [
                runtime_v23a.base.specialist_template(item)
                for item in panel["questions"]
            ]
            dense = runtime_v23a.anchor_v4.prepare_gold_answer_items_v4(
                tokenizer, prompts, panel["answers"]
            )
            if len(dense) != 56:
                raise RuntimeError("V26A train panel request count changed")
            panels[panel_name] = {
                "slice": (cursor, cursor + 56), "dense": dense,
            }
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
        if cursor != prereg.REQUESTS_PER_ENGINE_V26A:
            raise RuntimeError("V26A aggregate request count changed")
        processed_tokens = sum(
            len(item["prompt_token_ids"]) + 1 for item in requests
        )
        return (
            panels,
            requests,
            processed_tokens,
            _seal({
                "schema": "eggroll-es-v26a-token-request-audit",
                "request_count": len(requests),
                "processed_tokens_per_engine_call": processed_tokens,
                "request_identity_sha256": canonical_sha256(requests),
                "panel_token_audit_sha256": canonical_sha256(token_audit),
                "all_requests_within_1024_token_cap": all(
                    item["maximum_combined_tokens"] <= 1024
                    for item in token_audit.values()
                ),
                "raw_train_content_or_token_ids_persisted": False,
            }),
        )

    @staticmethod
    def _sampling_params():
        from vllm import SamplingParams
        return SamplingParams(
            n=1, seed=43, temperature=0.0, top_p=1.0, max_tokens=1,
            prompt_logprobs=1, detokenize=False,
        )

    def _generate(self, requests, *, profiled):
        if profiled:
            reports = self._resolve([
                engine.generate_profiled_v26a.remote(
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
            raise RuntimeError("V26A all-four-actor generation incomplete")
        return batches, reports

    @staticmethod
    def _first_tokens(batch):
        values = []
        for output in batch:
            candidates = getattr(output, "outputs", None)
            token_ids = getattr(candidates[0], "token_ids", None) if candidates else None
            if not token_ids or not isinstance(token_ids[0], int):
                raise RuntimeError("V26A greedy first-token output changed")
            values.append(token_ids[0])
        return np.asarray(values, dtype=np.int64)

    def _score(self, panels, batches):
        scores = {}
        tokens = {}
        commitments = {}
        for rank in prereg.PHYSICAL_GPU_IDS_V26A:
            cell = self.assignments[str(rank)]["cell"]
            cell_scores = np.empty((5, 56), dtype=np.float64)
            hashes = []
            for panel_index, panel_name in enumerate(prereg.PANEL_NAMES_V26A):
                panel = panels[panel_name]
                start, stop = panel["slice"]
                reward, digest = runtime_v23a._score_panel_outputs_v23a(
                    panel["dense"], batches[rank][start:stop]
                )
                cell_scores[panel_index] = reward
                hashes.append(digest)
            scores[cell] = cell_scores
            tokens[cell] = V26ProbeController._first_tokens(batches[rank])
            commitments[cell] = hashes
        return scores, tokens, commitments

    def run_wave(self, panels, requests):
        identities = self._resolve([
            engine.runtime_identity_v26a.remote() for engine in self.engines
        ])
        for rank in prereg.PHYSICAL_GPU_IDS_V26A:
            item = identities[rank]
            spec = self.assignments[str(rank)]
            if (
                item.get("schema") != "eggroll-es-v26a-actor-runtime-identity"
                or item.get("cell") != spec["cell"]
                or item.get("wave") != self.wave
                or item.get("backend") != spec["backend"]
                or item.get("model_path") != spec["model_path"]
                or item.get("physical_gpu_id") != rank
                or not isinstance(item.get("nvml_uuid"), str)
                or not item["nvml_uuid"]
                or not isinstance(item.get("pci_bus_id"), str)
                or not item["pci_bus_id"]
                or item.get("cuda_visible_devices") != str(rank)
                or item.get("torch_cuda_device_count") != 1
                or item.get("torch_current_device") != 0
            ):
                raise RuntimeError("V26A actor/model/physical-GPU mapping changed")
        warmup, _ = self._generate(requests, profiled=False)
        self._score(panels, warmup)
        resident_reports = self._resolve([
            engine.runtime_identity_v26a.remote() for engine in self.engines
        ])
        resident = {
            self.assignments[str(index)]["cell"]: {
                "used_bytes": resident_reports[index]["used_bytes"],
                "total_bytes": resident_reports[index]["total_bytes"],
            }
            for index in prereg.PHYSICAL_GPU_IDS_V26A
        }
        reference_batches, _ = self._generate(requests, profiled=False)
        reference = self._score(panels, reference_batches)
        repeat_batches, _ = self._generate(requests, profiled=False)
        repeat = self._score(panels, repeat_batches)
        for index in range(3):
            for cell in reference[index]:
                left, right = reference[index][cell], repeat[index][cell]
                if index == 2:
                    equal = left == right
                else:
                    equal = np.array_equal(left, right)
                if not equal:
                    raise RuntimeError("V26A within-arm deterministic repeat changed")
        for backend in ("full_fp8", "fp8_routed_bf16_backbone_v26"):
            matching = [
                self.assignments[str(gpu_id)]["cell"]
                for gpu_id in prereg.PHYSICAL_GPU_IDS_V26A
                if self.assignments[str(gpu_id)]["backend"] == backend
            ]
            left, right = matching
            if (
                not np.array_equal(reference[0][left], reference[0][right])
                or not np.array_equal(reference[1][left], reference[1][right])
                or reference[2][left] != reference[2][right]
            ):
                raise RuntimeError("V26A duplicate backend replicas disagree")

        elapsed = np.empty((prereg.TIMING_REPETITIONS_V26A, 4), dtype=np.int64)
        peak_nvml = np.empty_like(elapsed)
        timing_commitments = []
        torch_peak_commitments = []
        for repetition in range(prereg.TIMING_REPETITIONS_V26A):
            batches, reports = self._generate(requests, profiled=True)
            observed = self._score(panels, batches)
            for index in prereg.PHYSICAL_GPU_IDS_V26A:
                cell = self.assignments[str(index)]["cell"]
                if (
                    not np.array_equal(observed[0][cell], reference[0][cell])
                    or not np.array_equal(observed[1][cell], reference[1][cell])
                    or observed[2][cell] != reference[2][cell]
                ):
                    raise RuntimeError("V26A timed output differs from deterministic reference")
                report = reports[index]
                elapsed[repetition, index] = report["elapsed_ns"]
                peak_nvml[repetition, index] = report["peak_nvml_used_bytes"]
                torch_peak_commitments.append({
                    "cell": cell,
                    "repetition": repetition,
                    "allocated": report["peak_torch_allocated_bytes"],
                    "reserved": report["peak_torch_reserved_bytes"],
                    "nvml_sample_count": report["nvml_sample_count"],
                })
            timing_commitments.append(canonical_sha256(observed[2]))

        return {
            "wave": self.wave,
            "identities": identities,
            "reference": reference,
            "elapsed_ns": elapsed,
            "peak_nvml": peak_nvml,
            "resident": resident,
            "timing_commitments": timing_commitments,
            "torch_peak_commitments": torch_peak_commitments,
            "within_wave_integrity": {
                "exact_four_actor_physical_gpu_mapping": True,
                "exact_model_path_and_backend_per_cell": True,
                "within_cell_repeats_exact": True,
                "same_backend_duplicate_cells_exact_within_wave": True,
                "all_seven_timed_outputs_equal_reference": True,
                "all_timing_and_nvml_profiles_complete": True,
            },
        }


def run_counterbalanced_probe_v26a(preregistration, prelaunch_gpu_idle_certificate):
    baseline_items = prelaunch_gpu_idle_certificate.get("gpus", [])
    if (
        prelaunch_gpu_idle_certificate.get("all_four_idle") is not True
        or [item.get("physical_gpu_id") for item in baseline_items]
        != list(prereg.PHYSICAL_GPU_IDS_V26A)
    ):
        raise RuntimeError("V26A prelaunch physical-GPU certificate changed")
    baseline_physical = {
        item["physical_gpu_id"]: {
            "nvml_uuid": item.get("nvml_uuid"),
            "pci_bus_id": item.get("pci_bus_id"),
            "total_bytes": item.get("total_bytes"),
        }
        for item in baseline_items
    }
    if any(
        not item["nvml_uuid"] or not item["pci_bus_id"]
        for item in baseline_physical.values()
    ):
        raise RuntimeError("V26A prelaunch physical-GPU identity incomplete")
    preparation_controller = V26ProbeController(
        preregistration, prereg.WAVE_ORDER_V26A[0]
    )
    panels, requests, processed_tokens, token_audit = (
        preparation_controller._prepare_train_requests()
    )
    wave_results = {}
    interwave_idle_certificate = None
    for wave_index, wave in enumerate(prereg.WAVE_ORDER_V26A):
        controller = V26ProbeController(preregistration, wave)
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
        wave_results[wave] = result
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V26A:
            identity = result["identities"][gpu_id]
            expected = baseline_physical[gpu_id]
            if any(
                identity.get(key) != expected[key]
                for key in ("nvml_uuid", "pci_bus_id", "total_bytes")
            ):
                raise RuntimeError(
                    "V26A actor does not match prelaunch physical-GPU identity"
                )
        if wave_index == 0:
            interwave_idle_certificate = wait_for_interwave_gpu_idle_v26a(
                prelaunch_gpu_idle_certificate
            )
            observed_physical = {
                item["physical_gpu_id"]: {
                    "nvml_uuid": item.get("nvml_uuid"),
                    "pci_bus_id": item.get("pci_bus_id"),
                    "total_bytes": item.get("total_bytes"),
                }
                for item in interwave_idle_certificate.get("gpus", [])
            }
            if observed_physical != baseline_physical:
                raise RuntimeError("V26A interwave physical-GPU identity changed")

    backend_appearances = {
        "full_fp8": [],
        "fp8_routed_bf16_backbone_v26": [],
    }
    for wave in prereg.WAVE_ORDER_V26A:
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V26A:
            spec = preregistration["waves"][wave]["gpu_assignments"][str(gpu_id)]
            backend_appearances[spec["backend"]].append((wave, spec["cell"]))
    for backend, appearances in backend_appearances.items():
        if len(appearances) != 4:
            raise RuntimeError("V26A backend crossover appearance count changed")
        reference_wave, reference_cell = appearances[0]
        expected = wave_results[reference_wave]["reference"]
        for wave, cell in appearances[1:]:
            observed = wave_results[wave]["reference"]
            if (
                not np.array_equal(
                    expected[0][reference_cell], observed[0][cell]
                )
                or not np.array_equal(
                    expected[1][reference_cell], observed[1][cell]
                )
                or expected[2][reference_cell] != observed[2][cell]
            ):
                raise RuntimeError(
                    f"V26A same-backend outputs disagree across waves: {backend}"
                )

    equivalence = {}
    for pair_name in prereg.PAIR_ORDER_V26A:
        spec = preregistration["pairing"]["physical_gpu_pairs"][pair_name]
        reference = wave_results[spec["reference_wave"]]["reference"]
        candidate = wave_results[spec["candidate_wave"]]["reference"]
        equivalence[pair_name] = equivalence_metrics_v26a(
            reference[0][spec["reference_cell"]],
            candidate[0][spec["candidate_cell"]],
            reference[1][spec["reference_cell"]],
            candidate[1][spec["candidate_cell"]],
        )
        equivalence[pair_name]["physical_gpu_id"] = spec["physical_gpu_id"]
        equivalence[pair_name]["load_order"] = spec["load_order"]

    elapsed = np.stack([
        wave_results[wave]["elapsed_ns"] for wave in prereg.WAVE_ORDER_V26A
    ])
    peak_nvml = np.stack([
        wave_results[wave]["peak_nvml"] for wave in prereg.WAVE_ORDER_V26A
    ])
    resident = {}
    for wave in prereg.WAVE_ORDER_V26A:
        resident.update(wave_results[wave]["resident"])
    performance = speed_memory_summary_v26a(
        elapsed, processed_tokens, resident, peak_nvml
    )
    runtime_integrity = {
        "exact_two_wave_counterbalanced_backend_mapping": True,
        "all_four_physical_gpus_active_in_each_wave": True,
        "fresh_actor_engine_and_model_load_boundary_between_waves": True,
        "all_four_gpus_idle_between_wave_teardown_and_next_load": (
            interwave_idle_certificate.get("all_four_idle") is True
        ),
        "bounded_interwave_async_cleanup_poll_completed": (
            interwave_idle_certificate.get("bounded_async_cleanup_wait") is True
        ),
        "interwave_cleanup_poll_count": interwave_idle_certificate.get(
            "poll_count"
        ),
        "interwave_cleanup_elapsed_milliseconds": interwave_idle_certificate.get(
            "elapsed_milliseconds"
        ),
        "same_nvml_uuid_pci_bus_and_total_memory_per_physical_gpu_across_waves": True,
        "model_load_and_teardown_time_excluded": True,
        "single_driver_tokenization_identical_all_eight_cells": True,
        "within_cell_repeats_exact": True,
        "same_backend_outputs_exact_across_all_four_appearances": True,
        "all_seven_timed_outputs_equal_reference_in_all_eight_cells": True,
        "all_timing_and_nvml_profiles_complete": True,
        "update_checkpoint_evaluation_and_adoption_surfaces_closed": True,
        "all_integrity_audits_passed": True,
    }
    gate = evaluate_gate_v26a(equivalence, performance, runtime_integrity)
    compact_wave_audits = {}
    for wave in prereg.WAVE_ORDER_V26A:
        result = wave_results[wave]
        compact_wave_audits[wave] = {
            "actor_identity_sha256": canonical_sha256(result["identities"]),
            "reference_dense_commitments_sha256": canonical_sha256(
                result["reference"][2]
            ),
            "timed_dense_commitments_sha256": canonical_sha256(
                result["timing_commitments"]
            ),
            "torch_peak_profile_commitment_sha256": canonical_sha256(
                result["torch_peak_commitments"]
            ),
            "within_wave_integrity": result["within_wave_integrity"],
        }
    audit = _seal({
        "schema": "eggroll-es-v26a-compact-runtime-audit",
        "token_request_audit": token_audit,
        "waves": compact_wave_audits,
        "interwave_idle_certificate": interwave_idle_certificate,
        "runtime_integrity": runtime_integrity,
        "generation_call_count_per_engine_per_wave": 10,
        "total_engine_generation_calls_all_waves": 80,
        "total_generation_requests_all_engines_all_waves": 22_400,
        "raw_outputs_scores_token_ids_timing_vectors_memory_samples_or_bootstrap_replicates_persisted": False,
    })
    summary = _seal({
        "schema": "eggroll-es-v26a-compat-speed-summary",
        "equivalence": equivalence,
        "performance": performance,
        "gate": gate,
        "runtime_integrity": runtime_integrity,
        "direct_model_adoption_or_training_action_taken": False,
    })
    _assert_compact({"summary": summary, "audit": audit})
    return summary, audit


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v26a-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-recipe-sha256")
    return parser


def run_exact_v26a(preregistration, implementation, recipe):
    environment = runtime_r2.certify_runtime_environment_r2()
    provenance = runtime_v23a._source_provenance_v23a(implementation)
    live_model = live_model_audit_v26a(preregistration)
    attempt_path = OUTPUT_DIRECTORY / ATTEMPT_NAME
    run_dir = OUTPUT_DIRECTORY / EXPERIMENT_NAME
    report_path = run_dir / REPORT_NAME
    if attempt_path.exists() or run_dir.exists():
        raise RuntimeError("V26A requires fresh exclusive attempt and run paths")
    gpu_idle = assert_all_four_gpus_idle_v26a()
    attempt = {
        "schema": "eggroll-es-v26a-durable-launch-attempt",
        "status": "launching",
        "phase": "before_actor_creation",
        "recipe": recipe,
        "source_provenance": provenance,
        "runtime_environment_certificate": environment,
        "gpu_idle_certificate": gpu_idle,
        "live_model_audit": live_model,
        "model_update_applied": False,
        "checkpoint_written": False,
        "evaluation_opened": False,
        "direct_model_adoption_applied": False,
    }
    _assert_compact(attempt)
    runtime_v23a._exclusive_write_json_v23a(attempt_path, attempt)
    failure = None
    result = None
    try:
        result = run_counterbalanced_probe_v26a(preregistration, gpu_idle)
    except BaseException as error:
        failure = error
    if failure is not None:
        attempt.update({
            "status": "failed",
            "phase": "inside_v26a_train_only_probe",
            "failure_type": type(failure).__name__,
            "failure_sha256": canonical_sha256({
                "type": type(failure).__name__, "repr": repr(failure),
            }),
            "report_exists_after_attempt": report_path.exists(),
        })
        _assert_compact(attempt)
        runtime_v23a._rewrite_json_v23a(attempt_path, attempt)
        raise failure
    summary, runtime_audit = result
    report = _seal({
        "schema": "eggroll-es-v26a-compat-speed-report",
        "recipe": recipe,
        "summary": summary,
        "runtime_audit": runtime_audit,
        "implementation": implementation,
        "model_update_applied": False,
        "checkpoint_written": False,
        "evaluation_opened": False,
        "direct_model_adoption_applied": False,
        "direct_action_taken": False,
    })
    _assert_compact(report)
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
    args = _parser().parse_args(argv)
    preregistration = load_preregistration_v26a()
    implementation = implementation_identity_v26a()
    recipe = recipe_v26a(preregistration, implementation)
    validate_runtime_v26a(args, preregistration, implementation, recipe)
    if args.v26a_dry_run:
        payload = _seal({
            "schema": "eggroll-es-v26a-compat-speed-dry-run",
            "recipe": recipe,
            "implementation": implementation,
            "gpu_launched": False,
            "gpu_idle_check_executed": False,
            "future_real_launch_requires_all_four_gpus_idle": True,
            "future_real_launch_requires_exact_post_commit_hashes": True,
            "pass_authority_limited_to_separate_train_only_training_ab": True,
            "direct_adoption_update_checkpoint_evaluation_authorized": False,
            "raw_train_or_output_content_persisted": False,
        })
        _assert_compact(payload)
        print(json.dumps(payload, sort_keys=True))
        return payload
    return run_exact_v26a(preregistration, implementation, recipe)


if __name__ == "__main__":
    main()
