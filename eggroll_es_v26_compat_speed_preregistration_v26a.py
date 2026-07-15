#!/usr/bin/env python3
"""Preregister a future train-only FP8-versus-V26 compatibility/speed probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CANONICAL_ROOT = Path("/home/catid/specialist")
OUTPUT_PATH_V26A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V26A_FP8_ROUTED_EXPERTS_COMPAT_SPEED_PREREGISTRATION.json"
)
SCHEMA_V26A = "eggroll-es-v26-model-compat-speed-preregistration-v26a"
FP8_MODEL_V26A = CANONICAL_ROOT / "models/Qwen3.6-35B-A3B-FP8"
HYBRID_MODEL_V26A = (
    CANONICAL_ROOT / "models/"
    "Qwen3.6-35B-A3B-FP8-routed-experts-BF16-backbone-v26"
)
HYBRID_PROVENANCE_V26A = (
    HYBRID_MODEL_V26A / "hybrid_routed_experts_bf16_backbone_manifest_v26.json"
)
PANEL_MANIFEST_V26A = (
    CANONICAL_ROOT / "experiments/eggroll_es_hpo/train_panel_sampling_v13/"
    "document_balanced_train_panels_v13.json"
)

FP8_CONFIG_SHA256_V26A = "570ef7ea45a7e1d3de2b1d3c70c4ac3562d0e768acdc195778cb4f4d95025845"
FP8_INDEX_SHA256_V26A = "6f176f344e41d35b17af12904e33401da5ebff3b49fccb8bfa0185bc2d50f9d6"
FP8_WEIGHT_SHARD_MANIFEST_V26A = {
    "file_count": 42,
    "total_bytes": 37_463_662_160,
    "manifest_sha256": "25ae972a0ac80b7875b5e041172d5ad572b522619040f4786a9facdf0e36e5dd",
}
HYBRID_CONFIG_SHA256_V26A = "e3c0081b616106017a26e2267bccbaefe1e221f41b729cfb1d89fddb3bfbf0b7"
HYBRID_INDEX_SHA256_V26A = "00209ac9b1eda53eada4af96d3f86c653150ea0e3229a6d10620507023ae4b72"
HYBRID_PROVENANCE_FILE_SHA256_V26A = "9a6860a673584d342fe64438539e33fffb8915b6b776c2b5386ffdb01ce6ad65"
HYBRID_PROVENANCE_CONTENT_SHA256_V26A = "e56b0162e0bd27f1690b68762290fcd247788c6f02fa2c22fe79e48153b42845"
HYBRID_WEIGHT_SHARD_MANIFEST_V26A = {
    "file_count": 67,
    "total_bytes": 38_898_892_464,
    "manifest_sha256": "6a4df77903cd680550724f988ef224eb4390a02088c7cf72987da7346e303539",
}
EXPECTED_V26_AUDIT_CONTENT_SHA256 = "52a21f4956cc386393d129897e59c59a578ca2f3e7818577e8974c7e1ce228d2"
V26_CODE_IDENTITIES = {
    "builder": {
        "path": "build_qwen36_fp8_routed_experts_bf16_backbone_v26.py",
        "file_sha256": "75ca45672c3e4496638ced92acab6c9a9dafa7510d0830c337f1b1fb3f48076a",
    },
    "auditor": {
        "path": "audit_qwen36_fp8_routed_experts_bf16_backbone_v26.py",
        "file_sha256": "b57fdef36ab0d742171449323c937bc441f723190b4cc8bb2862e33b420af5f0",
    },
    "builder_tests": {
        "path": "test_build_qwen36_fp8_routed_experts_bf16_backbone_v26.py",
        "file_sha256": "c03b184223f8b6a0938eb3c2bea26edefe0b5cccdf83bd7dca2a92d237e391ff",
    },
    "auditor_tests": {
        "path": "test_audit_qwen36_fp8_routed_experts_bf16_backbone_v26.py",
        "file_sha256": "67b4bdfd373a21e8091447845044ef5dcf2711e94d2efd729f8b4a5888da07ea",
    },
}
PANEL_MANIFEST_FILE_SHA256_V26A = "e555d9d6746cde6297cd3ab523b16dd7d78d81e2674447ee46d754ebfac52da7"
PANEL_MANIFEST_CONTENT_SHA256_V26A = "46cc98b694c98c1ee1c5456b855fb3b1db4534b3df2dcda69fc690a2d8a61bf5"
PANEL_BUNDLE_CONTENT_SHA256_V26A = "cc176a9b86c6447dcde8a11fd28d68c837d2119715126c57a3f37293fb0d492b"
TRAIN_SOURCE_FILE_SHA256_V26A = "f7127c38c7b540eaf9cf4349d1a1b8076e171da7f8ea43c11068ad1c311bb776"
TRAIN_SOURCE_ARROW_SHA256_V26A = "6b6fdfdd082f1de2bf1b4c78bd0a4154af5c709b26e46b0677dcde695d3b4cb6"

WAVE_ORDER_V26A = ("wave_a", "wave_b")
PHYSICAL_GPU_IDS_V26A = (0, 1, 2, 3)
PAIR_ORDER_V26A = tuple(f"physical_gpu_{gpu_id}" for gpu_id in PHYSICAL_GPU_IDS_V26A)
WAVE_BACKEND_ORDER_V26A = {
    "wave_a": (
        "full_fp8", "fp8_routed_bf16_backbone_v26",
        "full_fp8", "fp8_routed_bf16_backbone_v26",
    ),
    "wave_b": (
        "fp8_routed_bf16_backbone_v26", "full_fp8",
        "fp8_routed_bf16_backbone_v26", "full_fp8",
    ),
}
PANEL_NAMES_V26A = (
    "optimization_0", "optimization_1", "optimization_2",
    "train_screen_0", "train_screen_1",
)
REQUESTS_PER_ENGINE_V26A = 280
WARMUP_REPETITIONS_V26A = 1
DETERMINISM_REPETITIONS_V26A = 2
TIMING_REPETITIONS_V26A = 7
TIMING_BOOTSTRAP_REPETITIONS_V26A = 20_000
TIMING_BOOTSTRAP_SEED_V26A = 20260920
SPEED_FAMILYWISE_QUANTILE_V26A = 0.05 / 4
EQUIVALENCE_THRESHOLDS_V26A = {
    "absolute_mean_signed_logprob_delta_max": 0.02,
    "mean_absolute_logprob_delta_max": 0.05,
    "root_mean_square_logprob_delta_max": 0.10,
    "p99_absolute_logprob_delta_max": 0.25,
    "maximum_absolute_logprob_delta_max": 0.75,
    "row_logprob_pearson_min": 0.995,
    "greedy_first_token_agreement_min": 0.99,
}
SPEED_NONINFERIORITY_RATIO_V26A = 0.90
MAX_PEAK_NVML_FRACTION_V26A = 0.95


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _without_self(value):
    return {
        key: item
        for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def timing_bootstrap_draw_plan_v26a():
    generator = np.random.default_rng(TIMING_BOOTSTRAP_SEED_V26A)
    draws = generator.integers(
        0,
        TIMING_REPETITIONS_V26A,
        size=(TIMING_BOOTSTRAP_REPETITIONS_V26A, TIMING_REPETITIONS_V26A),
        dtype=np.int64,
    )
    header = {
        "shape": list(draws.shape),
        "dtype": "int64",
        "paired_repetition_indices_for_both_waves_and_all_four_physical_gpus": True,
    }
    digest = hashlib.sha256()
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)
    digest.update(draws.tobytes(order="C"))
    return draws, digest.hexdigest()


def _validate_conversion_contract_v26a():
    for item in V26_CODE_IDENTITIES.values():
        if file_sha256(CANONICAL_ROOT / item["path"]) != item["file_sha256"]:
            raise RuntimeError("V26 conversion code identity changed")
    if (
        file_sha256(FP8_MODEL_V26A / "config.json") != FP8_CONFIG_SHA256_V26A
        or file_sha256(FP8_MODEL_V26A / "model.safetensors.index.json")
        != FP8_INDEX_SHA256_V26A
        or file_sha256(HYBRID_MODEL_V26A / "config.json")
        != HYBRID_CONFIG_SHA256_V26A
        or file_sha256(HYBRID_MODEL_V26A / "model.safetensors.index.json")
        != HYBRID_INDEX_SHA256_V26A
        or file_sha256(HYBRID_PROVENANCE_V26A)
        != HYBRID_PROVENANCE_FILE_SHA256_V26A
    ):
        raise RuntimeError("V26 model config, index, or provenance identity changed")
    provenance = _load_json(HYBRID_PROVENANCE_V26A)
    if (
        provenance.get("content_sha256_before_self_field")
        != HYBRID_PROVENANCE_CONTENT_SHA256_V26A
        or provenance.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(provenance))
        or provenance.get("target_key_count") != 63_939
        or provenance.get("target_element_count") != 35_953_837_936
        or provenance.get("target_byte_count") != 38_890_114_784
        or provenance.get("removed_non_routed_scale_count") != 257
        or provenance.get("output_weight_shards")
        != HYBRID_WEIGHT_SHARD_MANIFEST_V26A
        or provenance.get("non_routed_fp8_scales_retained") is not False
        or provenance.get("contains_dataset_or_evaluation_content") is not False
    ):
        raise RuntimeError("V26 aggregate conversion contract changed")
    return provenance


def _validate_panel_manifest_v26a():
    if file_sha256(PANEL_MANIFEST_V26A) != PANEL_MANIFEST_FILE_SHA256_V26A:
        raise RuntimeError("V26A train-only panel manifest identity changed")
    value = _load_json(PANEL_MANIFEST_V26A)
    if (
        value.get("content_sha256_before_self_field")
        != PANEL_MANIFEST_CONTENT_SHA256_V26A
        or value.get("source", {}).get("jsonl_sha256")
        != TRAIN_SOURCE_FILE_SHA256_V26A
        or value.get("source", {}).get("arrow_sha256")
        != TRAIN_SOURCE_ARROW_SHA256_V26A
        or [item.get("name") for item in value.get("panels", [])]
        != list(PANEL_NAMES_V26A)
        or any(len(item.get("items", [])) != 56 for item in value.get("panels", []))
    ):
        raise RuntimeError("V26A train-only panel manifest content changed")
    return value


def build_preregistration_v26a():
    provenance = _validate_conversion_contract_v26a()
    _validate_panel_manifest_v26a()
    _draws, draw_sha256 = timing_bootstrap_draw_plan_v26a()
    fp8_path = str(FP8_MODEL_V26A)
    hybrid_path = str(HYBRID_MODEL_V26A)
    model_paths = {
        "full_fp8": fp8_path,
        "fp8_routed_bf16_backbone_v26": hybrid_path,
    }
    waves = {}
    for wave_index, wave in enumerate(WAVE_ORDER_V26A):
        assignments = {}
        for gpu_id, backend in zip(
            PHYSICAL_GPU_IDS_V26A, WAVE_BACKEND_ORDER_V26A[wave], strict=True,
        ):
            assignments[str(gpu_id)] = {
                "cell": f"{wave}_gpu_{gpu_id}",
                "engine_rank": gpu_id,
                "physical_gpu_id": gpu_id,
                "backend": backend,
                "model_path": model_paths[backend],
            }
        waves[wave] = {
            "wave_index": wave_index,
            "gpu_assignments": assignments,
            "all_four_gpus_active_concurrently": True,
            "fresh_engine_and_model_load_boundary_before_wave": True,
        }
    physical_pairs = {}
    for gpu_id in PHYSICAL_GPU_IDS_V26A:
        fp8_wave = next(
            wave for wave in WAVE_ORDER_V26A
            if WAVE_BACKEND_ORDER_V26A[wave][gpu_id] == "full_fp8"
        )
        hybrid_wave = next(
            wave for wave in WAVE_ORDER_V26A
            if WAVE_BACKEND_ORDER_V26A[wave][gpu_id]
            == "fp8_routed_bf16_backbone_v26"
        )
        physical_pairs[f"physical_gpu_{gpu_id}"] = {
            "physical_gpu_id": gpu_id,
            "reference_backend": "full_fp8",
            "reference_wave": fp8_wave,
            "reference_cell": f"{fp8_wave}_gpu_{gpu_id}",
            "candidate_backend": "fp8_routed_bf16_backbone_v26",
            "candidate_wave": hybrid_wave,
            "candidate_cell": f"{hybrid_wave}_gpu_{gpu_id}",
            "load_order": (
                "fp8_first" if fp8_wave == "wave_a" else "hybrid_first"
            ),
        }
    value = {
        "schema": SCHEMA_V26A,
        "experiment_name": "s6_v26a_fp8_vs_routed_experts_bf16_backbone_train_only_probe",
        "status": "preregistered_future_runtime_not_launched",
        "preregistered_before_any_v26a_train_reward_output": True,
        "strict_train_only": True,
        "model_contract": {
            "fp8": {
                "path": fp8_path,
                "config_sha256": FP8_CONFIG_SHA256_V26A,
                "index_sha256": FP8_INDEX_SHA256_V26A,
                "weight_shard_manifest": FP8_WEIGHT_SHARD_MANIFEST_V26A,
            },
            "hybrid_v26": {
                "path": hybrid_path,
                "config_sha256": HYBRID_CONFIG_SHA256_V26A,
                "index_sha256": HYBRID_INDEX_SHA256_V26A,
                "weight_shard_manifest": HYBRID_WEIGHT_SHARD_MANIFEST_V26A,
                "provenance_file_sha256": HYBRID_PROVENANCE_FILE_SHA256_V26A,
                "provenance_content_sha256": HYBRID_PROVENANCE_CONTENT_SHA256_V26A,
                "target_key_count": provenance["target_key_count"],
                "target_byte_count": provenance["target_byte_count"],
                "only_routed_experts_remain_fp8": True,
                "all_non_routed_pathways_exact_bf16": True,
                "non_routed_fp8_scale_count": 0,
            },
            "conversion_code_identities": V26_CODE_IDENTITIES,
            "required_exact_prelaunch_audit_content_sha256": (
                EXPECTED_V26_AUDIT_CONTENT_SHA256
            ),
        },
        "waves": waves,
        "pairing": {
            "pair_order": list(PAIR_ORDER_V26A),
            "physical_gpu_pairs": physical_pairs,
            "same_physical_gpu_compared_across_fresh_load_waves": True,
            "same_request_tokens_sampling_and_repetition_index_all_cells": True,
            "fp8_first_physical_gpu_ids": [0, 2],
            "hybrid_first_physical_gpu_ids": [1, 3],
            "wave_order_fixed_before_outputs": list(WAVE_ORDER_V26A),
        },
        "train_request_contract": {
            "panel_manifest_path": str(PANEL_MANIFEST_V26A),
            "panel_manifest_file_sha256": PANEL_MANIFEST_FILE_SHA256_V26A,
            "panel_manifest_content_sha256": PANEL_MANIFEST_CONTENT_SHA256_V26A,
            "panel_bundle_content_sha256": PANEL_BUNDLE_CONTENT_SHA256_V26A,
            "train_source_file_sha256": TRAIN_SOURCE_FILE_SHA256_V26A,
            "train_source_arrow_sha256": TRAIN_SOURCE_ARROW_SHA256_V26A,
            "panel_names": list(PANEL_NAMES_V26A),
            "rows_per_panel": 56,
            "request_count_per_engine_call": REQUESTS_PER_ENGINE_V26A,
            "driver_tokenizes_once_and_passes_identical_prompt_token_ids": True,
            "token_contract_is_hashed_and_aggregate_only": True,
            "maximum_prompt_plus_answer_tokens": 1024,
            "contains_validation_ood_heldout_or_benchmark_content": False,
        },
        "sampling_contract": {
            "n": 1,
            "seed": 43,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 1,
            "prompt_logprobs": 1,
            "detokenize": False,
            "prefix_caching": False,
            "enforce_eager": True,
        },
        "runtime_contract": {
            "wave_count": 2,
            "engine_count_per_wave": 4,
            "tensor_parallel_size_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "one_actor_and_one_complete_model_replica_per_physical_gpu_per_wave": True,
            "physical_gpu_identity_bound_by_nvml_index_uuid_and_pci_bus_across_waves": True,
            "placement_groups_bound_by_lightweight_string_int_canonicalized_gpu_probe": True,
            "placement_group_creation_order_never_used_as_physical_gpu_identity": True,
            "placement_groups_are_non_detached_and_removed_after_each_wave": True,
            "all_four_physical_gpus_active_concurrently_in_each_wave": True,
            "wave_backend_mapping": {
                wave: list(WAVE_BACKEND_ORDER_V26A[wave])
                for wave in WAVE_ORDER_V26A
            },
            "fresh_ray_actors_engines_and_model_loads_between_waves": True,
            "all_four_gpus_must_be_idle_after_wave_a_teardown_before_wave_b": True,
            "interwave_async_cleanup_idle_poll_timeout_seconds": 30.0,
            "interwave_async_cleanup_idle_poll_interval_seconds": 0.5,
            "interwave_poll_must_preserve_nvml_uuid_pci_and_total_memory": True,
            "prelaunch_idle_check_remains_immediate_and_unpolled": True,
            "same_moe_backend": "triton",
            "same_dtype": "bfloat16",
            "same_gpu_memory_utilization": 0.82,
            "warmup_repetitions": WARMUP_REPETITIONS_V26A,
            "determinism_repetitions": DETERMINISM_REPETITIONS_V26A,
            "timing_repetitions": TIMING_REPETITIONS_V26A,
            "generation_calls_per_engine_per_wave": (
                WARMUP_REPETITIONS_V26A
                + DETERMINISM_REPETITIONS_V26A
                + TIMING_REPETITIONS_V26A
            ),
            "total_engine_generation_calls_all_waves": 80,
            "total_generation_requests_all_engines_all_waves": 22_400,
            "model_load_initialization_and_teardown_time_excluded": True,
            "timing_clock": "actor-local time.perf_counter_ns with CUDA synchronization",
            "throughput_definition": (
                "fixed aggregate prompt tokens plus one output token per request divided by "
                "actor-local synchronized generation seconds"
            ),
            "resident_vram": "NVML used bytes after warmup before determinism and timing calls",
            "peak_vram": (
                "maximum of the post-warmup resident audit and 10ms actor-local NVML "
                "sampling across each timed generate call"
            ),
            "all_raw_outputs_scores_token_ids_timings_and_memory_samples_in_memory_only": True,
        },
        "equivalence_analysis": {
            "score": "per-example mean full gold-answer token logprob in natural-log units",
            "thresholds": EQUIVALENCE_THRESHOLDS_V26A,
            "threshold_justification": {
                "mean_absolute_0.05_nats": (
                    "caps the average likelihood-ratio displacement at exp(0.05)-1, about 5.13%"
                ),
                "rmse_0.10_and_p99_0.25_nats": (
                    "bound dispersion and 99th-percentile displacement while allowing expected "
                    "FP8-versus-BF16 kernel rounding"
                ),
                "max_0.75_nats": "rejects isolated catastrophic answer-token divergence",
                "pearson_0.995_and_token_agreement_0.99": (
                    "preserve row ordering and deterministic greedy behavior"
                ),
                "no_threshold_selected_after_outputs": True,
            },
            "exact_repeatability_within_each_wave_gpu_cell_required": True,
            "same_backend_score_arrays_and_token_ids_exact_across_all_four_appearances": True,
            "all_four_physical_gpu_pair_metrics_must_pass_independently": True,
        },
        "speed_and_memory_analysis": {
            "paired_timing_repetitions": TIMING_REPETITIONS_V26A,
            "paired_bootstrap_seed": TIMING_BOOTSTRAP_SEED_V26A,
            "paired_bootstrap_repetitions": TIMING_BOOTSTRAP_REPETITIONS_V26A,
            "paired_bootstrap_draw_plan_sha256": draw_sha256,
            "one_sided_familywise_quantile": SPEED_FAMILYWISE_QUANTILE_V26A,
            "hybrid_over_fp8_throughput_noninferiority_ratio": (
                SPEED_NONINFERIORITY_RATIO_V26A
            ),
            "maximum_peak_nvml_fraction": MAX_PEAK_NVML_FRACTION_V26A,
            "throughput_and_vram_recorded_for_every_wave_gpu_cell": True,
            "throughput_ratio_paired_by_physical_gpu_and_repetition_across_waves": True,
            "each_physical_gpu_speed_pair_gated_independently": True,
            "all_eight_wave_gpu_cells_peak_vram_gated": True,
            "fp8_first_and_hybrid_first_order_strata_reported_separately": True,
            "wave_level_summary_reported_to_expose_load_order_effects": True,
            "model_load_time_excluded_from_throughput": True,
            "timing_vectors_bootstrap_draws_replicates_and_memory_samples_not_persisted": True,
        },
        "gate": {
            "conjunctive_requirements": [
                "all runtime model device token and request integrity audits pass",
                "all equivalence thresholds pass on each of four physical-GPU pairs",
                "all four paired throughput familywise lower bounds are at least 0.90",
                "all eight wave-GPU cell peak NVML fractions are at most 0.95",
            ],
            "pass_authority": (
                "authorize_only_a_separate_fresh_preregistered_train_only_training_A_B"
            ),
            "failure_decision": "retain_existing_FP8_model_for_any_later_training_A_B",
            "direct_model_adoption_authorized": False,
            "model_update_authorized": False,
            "checkpoint_write_authorized": False,
            "evaluation_authorized": False,
            "dataset_promotion_authorized": False,
        },
        "required_runtime_adapter": {
            "status": "implemented_but_future_gpu_launch_not_yet_authorized",
            "must_verify_no_other_gpu_owner_before_attempt_claim": True,
            "gpu_idle_check_after_cpu_disk_audits_immediately_before_attempt_claim": True,
            "must_require_exact_post_commit_implementation_and_recipe_hashes": True,
            "must_persist_only_aggregate_metrics_hashes_and_integrity_booleans": True,
            "current_external_gpu_owner_blocks_launch": True,
        },
        "contains_validation_ood_heldout_or_benchmark_content": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def validate_preregistration_v26a(value):
    runtime = value.get("runtime_contract", {}) if isinstance(value, dict) else {}
    pairing = value.get("pairing", {}) if isinstance(value, dict) else {}
    waves = value.get("waves", {}) if isinstance(value, dict) else {}
    if (
        not isinstance(value, dict)
        or value.get("schema") != SCHEMA_V26A
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(value))
        or value.get("strict_train_only") is not True
        or value.get("contains_validation_ood_heldout_or_benchmark_content") is not False
        or runtime.get("wave_count") != 2
        or runtime.get("engine_count_per_wave") != 4
        or runtime.get("tensor_parallel_size_per_engine") != 1
        or runtime.get("all_four_physical_gpus_active_concurrently_in_each_wave") is not True
        or runtime.get(
            "physical_gpu_identity_bound_by_nvml_index_uuid_and_pci_bus_across_waves"
        ) is not True
        or runtime.get(
            "placement_groups_bound_by_lightweight_string_int_canonicalized_gpu_probe"
        ) is not True
        or runtime.get(
            "placement_group_creation_order_never_used_as_physical_gpu_identity"
        ) is not True
        or runtime.get("placement_groups_are_non_detached_and_removed_after_each_wave") is not True
        or runtime.get("fresh_ray_actors_engines_and_model_loads_between_waves") is not True
        or runtime.get("interwave_async_cleanup_idle_poll_timeout_seconds") != 30.0
        or runtime.get("interwave_async_cleanup_idle_poll_interval_seconds") != 0.5
        or runtime.get(
            "interwave_poll_must_preserve_nvml_uuid_pci_and_total_memory"
        ) is not True
        or runtime.get("prelaunch_idle_check_remains_immediate_and_unpolled") is not True
        or runtime.get("total_generation_requests_all_engines_all_waves") != 22_400
        or tuple(waves) != WAVE_ORDER_V26A
        or tuple(pairing.get("physical_gpu_pairs", {})) != PAIR_ORDER_V26A
        or pairing.get("wave_order_fixed_before_outputs") != list(WAVE_ORDER_V26A)
        or value.get("train_request_contract", {}).get(
            "contains_validation_ood_heldout_or_benchmark_content"
        ) is not False
        or value.get("gate", {}).get("direct_model_adoption_authorized") is not False
        or value.get("gate", {}).get("model_update_authorized") is not False
        or value.get("gate", {}).get("checkpoint_write_authorized") is not False
        or value.get("gate", {}).get("evaluation_authorized") is not False
        or value.get("gate", {}).get("dataset_promotion_authorized") is not False
        or value.get("required_runtime_adapter", {}).get(
            "gpu_idle_check_after_cpu_disk_audits_immediately_before_attempt_claim"
        ) is not True
    ):
        raise RuntimeError("V26A preregistration contract changed")
    for wave in WAVE_ORDER_V26A:
        assignments = waves[wave].get("gpu_assignments", {})
        if tuple(assignments) != tuple(str(item) for item in PHYSICAL_GPU_IDS_V26A):
            raise RuntimeError("V26A crossover GPU assignment changed")
        observed = tuple(
            assignments[str(gpu_id)].get("backend")
            for gpu_id in PHYSICAL_GPU_IDS_V26A
        )
        if observed != WAVE_BACKEND_ORDER_V26A[wave]:
            raise RuntimeError("V26A crossover backend order changed")
        for gpu_id in PHYSICAL_GPU_IDS_V26A:
            item = assignments[str(gpu_id)]
            if (
                item.get("physical_gpu_id") != gpu_id
                or item.get("engine_rank") != gpu_id
                or item.get("cell") != f"{wave}_gpu_{gpu_id}"
            ):
                raise RuntimeError("V26A crossover physical GPU mapping changed")
    for gpu_id, pair_name in zip(
        PHYSICAL_GPU_IDS_V26A, PAIR_ORDER_V26A, strict=True,
    ):
        pair = pairing["physical_gpu_pairs"][pair_name]
        if (
            pair.get("physical_gpu_id") != gpu_id
            or pair.get("reference_backend") != "full_fp8"
            or pair.get("candidate_backend")
            != "fp8_routed_bf16_backbone_v26"
            or pair.get("reference_wave") == pair.get("candidate_wave")
            or pair.get("load_order")
            != ("fp8_first" if gpu_id in (0, 2) else "hybrid_first")
        ):
            raise RuntimeError("V26A physical-GPU crossover pair changed")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH_V26A)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    value = validate_preregistration_v26a(build_preregistration_v26a())
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
            raise RuntimeError("persisted V26A preregistration differs")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return value


if __name__ == "__main__":
    main()
