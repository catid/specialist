#!/usr/bin/env python3
"""CPU-only preregistration for the V27C BF16 ES train-step runtime A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

import eggroll_es_v27c_runtime_ab_preregistration_v28a as v28a
import train_eggroll_es_specialist_anchor_v13 as anchor_v13


ROOT = Path(__file__).resolve().parent
CANONICAL_ROOT = Path("/home/catid/specialist")
OUTPUT_PATH_V28C = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V28C_V27C_TUNED_TRAIN_STEP_AB_PREREGISTRATION.json"
)
SCHEMA_V28C = "eggroll-es-v27c-bf16-train-step-ab-preregistration-v28c"

POSITIVE_EVIDENCE_PATH_V28C = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V28B_V27C_TUNED_RUNTIME_AB_POSITIVE_EVIDENCE.json"
)
POSITIVE_EVIDENCE_COMMIT_V28C = "fe96a88cc42ed7cae4e57ccb544b26942553421d"
POSITIVE_EVIDENCE_FILE_SHA256_V28C = (
    "034b34166324359687398dd2825a0a602e444360144fc645c51e5e399e972041"
)
POSITIVE_EVIDENCE_CONTENT_SHA256_V28C = (
    "d2601ec9636fd1df100018bc96d74adbdbc9fd2d4b1e0415cb20df683ae0326f"
)

MODEL_PATH_V28C = v28a.MODEL_PATH_V28A
MODEL_CONFIG_SHA256_V28C = v28a.MODEL_CONFIG_SHA256_V28A
MODEL_INDEX_SHA256_V28C = v28a.MODEL_INDEX_SHA256_V28A
MODEL_WEIGHT_SHARD_MANIFEST_V28C = v28a.MODEL_WEIGHT_SHARD_MANIFEST_V28A
MODEL_SEAL_PATH_V28C = v28a.MODEL_SEAL_PATH_V28A
MODEL_SEAL_COMMIT_V28C = v28a.MODEL_SEAL_COMMIT_V28A
MODEL_SEAL_FILE_SHA256_V28C = v28a.MODEL_SEAL_FILE_SHA256_V28A
MODEL_SEAL_CONTENT_SHA256_V28C = v28a.MODEL_SEAL_CONTENT_SHA256_V28A
MODEL_BASE_ALL_FILES_FINGERPRINT_SHA256_V28C = (
    v28a.MODEL_BASE_ALL_FILES_FINGERPRINT_SHA256_V28A
)

TUNED_DIRECTORY_V28C = v28a.TUNED_DIRECTORY_V28A
TUNED_FILENAME_V28C = v28a.TUNED_FILENAME_V28A
TUNED_TABLE_PATH_V28C = v28a.TUNED_TABLE_PATH_V28A
TUNED_TABLE_COMMIT_V28C = v28a.TUNED_TABLE_COMMIT_V28A
TUNED_TABLE_FILE_SHA256_V28C = (
    "128806798a5bf8a961a5bd0bc8765c82e8b73a116e6c7411e7aeba5522667562"
)
TUNED_TABLE_CONTENT_SHA256_V28C = (
    "a4f82f53b037f766536013bdc10c8ca1e49873603a8f44972ef8007ed406de84"
)

PANEL_MANIFEST_V28C = v28a.PANEL_MANIFEST_V28A
PANEL_MANIFEST_FILE_SHA256_V28C = v28a.PANEL_MANIFEST_FILE_SHA256_V28A
PANEL_MANIFEST_CONTENT_SHA256_V28C = v28a.PANEL_MANIFEST_CONTENT_SHA256_V28A
PANEL_BUNDLE_CONTENT_SHA256_V28C = v28a.PANEL_BUNDLE_CONTENT_SHA256_V28A
FROZEN_TRAIN_SOURCE_V28C = v28a.FROZEN_TRAIN_SOURCE_V28A
FROZEN_TRAIN_SOURCE_SHA256_V28C = v28a.FROZEN_TRAIN_SOURCE_SHA256_V28A
FROZEN_TRAIN_ARROW_SHA256_V28C = v28a.FROZEN_TRAIN_ARROW_SHA256_V28A
PANEL_NAMES_V28C = tuple(anchor_v13.PANEL_NAMES_V13)

FROZEN_V13_IMPLEMENTATION_SHA256_V28C = {
    "trainer_v13": "1a8a4145a85c183bb6121914357b7e6bce916b4f76a0693887ac41fa3a8c4c6e",
    "worker_v13": "5596bff9174e5e94e812181a51f8cc9f9b2a73f3a4cb58c45d5346147c8d6367",
}
FROZEN_V13_IMPLEMENTATION_PATHS_V28C = {
    "trainer_v13": ROOT / "train_eggroll_es_specialist_anchor_v13.py",
    "worker_v13": ROOT / "eggroll_es_worker_v13.py",
}

ARMS_V28C = ("default_empty", "v27c_tuned")
PAIR_COUNT_V28C = 12
PAIR_ORDER_V28C = tuple(f"pair_{index:02d}" for index in range(PAIR_COUNT_V28C))
ARM_ORDER_BY_PAIR_V28C = {
    pair: (
        ("default_empty", "v27c_tuned")
        if index % 2 == 0 else ("v27c_tuned", "default_empty")
    )
    for index, pair in enumerate(PAIR_ORDER_V28C)
}
PHYSICAL_GPU_IDS_V28C = (0, 1, 2, 3)
ENGINE_COUNT_V28C = 4
POPULATION_SIZE_V28C = 32
ROWS_PER_PANEL_V28C = 56
COMBINED_PANEL_REQUESTS_V28C = 280
BASE_PROBE_COUNT_PER_ARM_V28C = 2
SIGNED_WAVE_COUNT_PER_ARM_V28C = 16
PERTURBED_REQUESTS_PER_SIGNED_WAVE_V28C = 4 * COMBINED_PANEL_REQUESTS_V28C
REQUESTS_PER_ARM_V28C = (
    BASE_PROBE_COUNT_PER_ARM_V28C * COMBINED_PANEL_REQUESTS_V28C
    + SIGNED_WAVE_COUNT_PER_ARM_V28C * PERTURBED_REQUESTS_PER_SIGNED_WAVE_V28C
)
REQUESTS_PER_PAIR_V28C = len(ARMS_V28C) * REQUESTS_PER_ARM_V28C
TOTAL_GENERATION_REQUESTS_V28C = PAIR_COUNT_V28C * REQUESTS_PER_PAIR_V28C

BOOTSTRAP_SEED_V28C = 20261012
BOOTSTRAP_REPETITIONS_V28C = 50_000
INFERENTIAL_ENDPOINT_COUNT_V28C = 3
FAMILYWISE_ALPHA_V28C = 0.05
FAMILYWISE_LOWER_QUANTILE_V28C = (
    FAMILYWISE_ALPHA_V28C / INFERENTIAL_ENDPOINT_COUNT_V28C
)
FAMILYWISE_UPPER_QUANTILE_V28C = 1.0 - FAMILYWISE_LOWER_QUANTILE_V28C
SPEED_POINT_RATIO_MIN_V28C = 1.01
SPEED_LCB_RATIO_MIN_V28C = 1.0
MEMORY_POINT_RATIO_MAX_V28C = 1.01
MEMORY_UCB_RATIO_MAX_V28C = 1.02
MAX_PEAK_NVML_FRACTION_V28C = 0.95


canonical_sha256 = v28a.canonical_sha256
file_sha256 = v28a.file_sha256


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _load_json_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"V28C requires a JSON object: {path}")
    return value


def _committed_file_sha256(commit, path):
    relative = Path(path).resolve().relative_to(CANONICAL_ROOT).as_posix()
    raw = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=CANONICAL_ROOT,
    )
    return hashlib.sha256(raw).hexdigest()


def validate_static_inputs_v28c():
    if (
        file_sha256(MODEL_PATH_V28C / "config.json")
        != MODEL_CONFIG_SHA256_V28C
        or file_sha256(MODEL_PATH_V28C / "model.safetensors.index.json")
        != MODEL_INDEX_SHA256_V28C
        or v28a._model_shard_manifest_v28a()
        != MODEL_WEIGHT_SHARD_MANIFEST_V28C
    ):
        raise RuntimeError("V28C exact Qwen3.6-35B-A3B identity changed")
    model_config = _load_json_object(MODEL_PATH_V28C / "config.json")
    if (
        model_config.get("quantization_config") is not None
        or model_config.get("text_config", {}).get("dtype") != "bfloat16"
    ):
        raise RuntimeError("V28C is BF16-only and rejects every FP8 path")
    if (
        file_sha256(MODEL_SEAL_PATH_V28C) != MODEL_SEAL_FILE_SHA256_V28C
        or _committed_file_sha256(MODEL_SEAL_COMMIT_V28C, MODEL_SEAL_PATH_V28C)
        != MODEL_SEAL_FILE_SHA256_V28C
    ):
        raise RuntimeError("V28C committed model seal changed")
    seal = _load_json_object(MODEL_SEAL_PATH_V28C)
    base_seal = seal.get("arms", {}).get("base_middle_late", {})
    if (
        seal.get("content_sha256_before_self_field")
        != MODEL_SEAL_CONTENT_SHA256_V28C
        or canonical_sha256(_without_self(seal)) != MODEL_SEAL_CONTENT_SHA256_V28C
        or base_seal.get("all_files_fingerprint_sha256")
        != MODEL_BASE_ALL_FILES_FINGERPRINT_SHA256_V28C
    ):
        raise RuntimeError("V28C model-seal semantics changed")
    if (
        tuple(sorted(item.name for item in TUNED_DIRECTORY_V28C.iterdir()))
        != (TUNED_FILENAME_V28C,)
        or file_sha256(TUNED_TABLE_PATH_V28C)
        != TUNED_TABLE_FILE_SHA256_V28C
        or _committed_file_sha256(TUNED_TABLE_COMMIT_V28C, TUNED_TABLE_PATH_V28C)
        != TUNED_TABLE_FILE_SHA256_V28C
    ):
        raise RuntimeError("V28C exact committed V27C table changed")
    table = _load_json_object(TUNED_TABLE_PATH_V28C)
    if (
        canonical_sha256(table) != TUNED_TABLE_CONTENT_SHA256_V28C
        or table.get("triton_version") != "3.6.0"
        or {int(key) for key in table if key != "triton_version"}
        != {256, 512, 1024, 2048}
    ):
        raise RuntimeError("V28C V27C table semantics changed")
    if (
        file_sha256(POSITIVE_EVIDENCE_PATH_V28C)
        != POSITIVE_EVIDENCE_FILE_SHA256_V28C
        or _committed_file_sha256(
            POSITIVE_EVIDENCE_COMMIT_V28C, POSITIVE_EVIDENCE_PATH_V28C,
        ) != POSITIVE_EVIDENCE_FILE_SHA256_V28C
    ):
        raise RuntimeError("V28C V28B positive-evidence binding changed")
    evidence = _load_json_object(POSITIVE_EVIDENCE_PATH_V28C)
    decision = evidence.get("decision", {})
    if (
        evidence.get("schema") != "eggroll-es-v28a-runtime-positive-evidence-v28b"
        or evidence.get("status")
        != "valid_completed_train_only_runtime_ab_gate_passed"
        or evidence.get("content_sha256_before_self_field")
        != POSITIVE_EVIDENCE_CONTENT_SHA256_V28C
        or canonical_sha256(_without_self(evidence))
        != POSITIVE_EVIDENCE_CONTENT_SHA256_V28C
        or evidence.get("aggregate_result", {}).get("gate_passed") is not True
        or decision.get("pass_authority")
        != "authorize_only_a_separately_frozen_train_only_training_recipe_A_B"
        or decision.get("v27c_table_validated_only_for_exact_bf16_runtime_contract")
        is not True
        or decision.get("bf16_v27c_table_reuse_for_fp8_authorized") is not False
        or any(decision.get(key) is not False for key in (
            "model_update_authorized", "checkpoint_write_authorized",
            "evaluation_authorized", "dataset_promotion_authorized",
            "validation_heldout_ood_or_benchmark_open_authorized",
        ))
    ):
        raise RuntimeError("V28C V28B authority changed")
    if (
        file_sha256(PANEL_MANIFEST_V28C) != PANEL_MANIFEST_FILE_SHA256_V28C
        or file_sha256(FROZEN_TRAIN_SOURCE_V28C)
        != FROZEN_TRAIN_SOURCE_SHA256_V28C
        or any(
            file_sha256(FROZEN_V13_IMPLEMENTATION_PATHS_V28C[key]) != digest
            for key, digest in FROZEN_V13_IMPLEMENTATION_SHA256_V28C.items()
        )
    ):
        raise RuntimeError("V28C frozen production/V13 train-only inputs changed")
    return table, evidence, base_seal


def bootstrap_draw_plan_v28c():
    draws = np.random.default_rng(BOOTSTRAP_SEED_V28C).integers(
        0, PAIR_COUNT_V28C,
        size=(BOOTSTRAP_REPETITIONS_V28C, PAIR_COUNT_V28C),
        dtype=np.int64,
    )
    header = {
        "shape": list(draws.shape),
        "dtype": "int64",
        "same_paired_draws_all_three_inferential_endpoints": True,
    }
    digest = hashlib.sha256()
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
    digest.update(draws.tobytes(order="C"))
    return draws, digest.hexdigest()


def build_preregistration_v28c():
    table, evidence, base_seal = validate_static_inputs_v28c()
    _draws, draw_sha256 = bootstrap_draw_plan_v28c()
    pair_plan = {
        pair: {
            "pair_index": index,
            "arm_order": list(ARM_ORDER_BY_PAIR_V28C[pair]),
            "fresh_four_tp1_engine_group_per_arm": True,
            "same_panel_tokens_perturbation_seeds_and_order_both_arms": True,
        }
        for index, pair in enumerate(PAIR_ORDER_V28C)
    }
    value = {
        "schema": SCHEMA_V28C,
        "experiment_name": "s6_v28c_v27c_bf16_no_update_es_train_step_ab",
        "status": "preregistered_future_runtime_not_launched",
        "preregistered_before_any_v28c_gpu_output": True,
        "strict_train_only": True,
        "authorization_basis": {
            "path": str(POSITIVE_EVIDENCE_PATH_V28C),
            "commit": POSITIVE_EVIDENCE_COMMIT_V28C,
            "file_sha256": POSITIVE_EVIDENCE_FILE_SHA256_V28C,
            "content_sha256": POSITIVE_EVIDENCE_CONTENT_SHA256_V28C,
            "v28b_gate_passed": evidence["aggregate_result"]["gate_passed"],
            "v28b_pass_authority": evidence["decision"]["pass_authority"],
        },
        "model_contract": {
            "path": str(MODEL_PATH_V28C),
            "family": "Qwen3.6-35B-A3B",
            "config_sha256": MODEL_CONFIG_SHA256_V28C,
            "index_sha256": MODEL_INDEX_SHA256_V28C,
            "weight_shard_manifest": MODEL_WEIGHT_SHARD_MANIFEST_V28C,
            "model_seal": {
                "path": str(MODEL_SEAL_PATH_V28C),
                "commit": MODEL_SEAL_COMMIT_V28C,
                "file_sha256": MODEL_SEAL_FILE_SHA256_V28C,
                "content_sha256": MODEL_SEAL_CONTENT_SHA256_V28C,
                "base_all_files_fingerprint_sha256": base_seal[
                    "all_files_fingerprint_sha256"
                ],
            },
            "dtype": "bfloat16",
            "quantization_config": None,
            "fp8_or_other_quantized_model_paths_authorized": False,
        },
        "table_contract": {
            "directory": str(TUNED_DIRECTORY_V28C),
            "directory_contains_exactly_one_file": True,
            "filename": TUNED_FILENAME_V28C,
            "commit": TUNED_TABLE_COMMIT_V28C,
            "file_sha256": TUNED_TABLE_FILE_SHA256_V28C,
            "content_sha256": TUNED_TABLE_CONTENT_SHA256_V28C,
            "triton_version": table["triton_version"],
            "batch_keys": [256, 512, 1024, 2048],
        },
        "arms": {
            "default_empty": {
                "VLLM_TUNED_CONFIG_FOLDER": "fresh_exclusive_empty_directory",
                "expected_config_source": "generic_fallback_none",
            },
            "v27c_tuned": {
                "VLLM_TUNED_CONFIG_FOLDER": str(TUNED_DIRECTORY_V28C),
                "expected_config_source": "exact_committed_v27c_table",
            },
        },
        "only_intended_arm_difference": "VLLM_TUNED_CONFIG_FOLDER",
        "paired_counterbalanced_design": {
            "pair_count": PAIR_COUNT_V28C,
            "pair_order": list(PAIR_ORDER_V28C),
            "pairs": pair_plan,
            "default_first_pair_count": 6,
            "tuned_first_pair_count": 6,
            "no_posthoc_pair_or_order_adaptation": True,
        },
        "frozen_train_step_contract": {
            "implementation_file_sha256": FROZEN_V13_IMPLEMENTATION_SHA256_V28C,
            "panel_manifest_path": str(PANEL_MANIFEST_V28C),
            "panel_manifest_file_sha256": PANEL_MANIFEST_FILE_SHA256_V28C,
            "panel_manifest_content_sha256": PANEL_MANIFEST_CONTENT_SHA256_V28C,
            "panel_bundle_content_sha256": PANEL_BUNDLE_CONTENT_SHA256_V28C,
            "frozen_train_source_path": str(FROZEN_TRAIN_SOURCE_V28C),
            "frozen_train_source_file_sha256": FROZEN_TRAIN_SOURCE_SHA256_V28C,
            "frozen_train_source_arrow_sha256": FROZEN_TRAIN_ARROW_SHA256_V28C,
            "panel_names": list(PANEL_NAMES_V28C),
            "rows_per_panel": ROWS_PER_PANEL_V28C,
            "combined_panel_requests": COMBINED_PANEL_REQUESTS_V28C,
            "population_size": POPULATION_SIZE_V28C,
            "perturbation_basis_seed": anchor_v13.PERTURBATION_BASIS_SEED_V13,
            "perturbation_basis_sha256": anchor_v13.PERTURBATION_BASIS_SHA256_V13,
            "perturbation_seed_sha256": canonical_sha256(
                anchor_v13.PERTURBATION_SEEDS_V13
            ),
            "alpha": 0.0,
            "model_update_applied": False,
            "apply_seed_coefficients_called": False,
            "checkpoint_write_called": False,
            "same_fixed_panel_token_identities_seed_order_and_sampling_both_arms": True,
            "contains_validation_heldout_ood_or_benchmark_content": False,
        },
        "request_budget": {
            "base_probes_per_arm": BASE_PROBE_COUNT_PER_ARM_V28C,
            "base_probe_requests_each": COMBINED_PANEL_REQUESTS_V28C,
            "signed_waves_per_arm": SIGNED_WAVE_COUNT_PER_ARM_V28C,
            "engines_per_signed_wave": ENGINE_COUNT_V28C,
            "requests_per_engine_per_signed_wave": COMBINED_PANEL_REQUESTS_V28C,
            "requests_per_arm": REQUESTS_PER_ARM_V28C,
            "requests_per_pair": REQUESTS_PER_PAIR_V28C,
            "total_generation_requests": TOTAL_GENERATION_REQUESTS_V28C,
        },
        "runtime_contract": {
            "fresh_engine_group_count": PAIR_COUNT_V28C * len(ARMS_V28C),
            "engines_per_group": ENGINE_COUNT_V28C,
            "tensor_parallel_size_per_engine": 1,
            "physical_gpu_ids": list(PHYSICAL_GPU_IDS_V28C),
            "all_four_gpus_simultaneously_active_during_every_estimator_required": True,
            "all_four_activity_observed_by_pid_bound_nvml_monitor_in_memory": True,
            "fresh_actors_engines_model_and_config_environment_every_arm": True,
            "placement_groups_mapped_by_runtime_physical_gpu_probe": True,
            "placement_group_creation_order_does_not_define_gpu_identity": True,
            "moe_backend": "triton",
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.82,
            "max_model_len": 2048,
            "enable_prefix_caching": False,
            "default_directory_fresh_exclusive_and_empty_every_arm": True,
            "exact_config_source_and_table_verified_inside_every_actor": True,
            "prelaunch_committed_clean_source_contract": True,
            "prelaunch_all_four_idle_gate": True,
            "interarm_all_four_idle_and_identity_preserving_cleanup_gate": True,
            "final_all_four_idle_and_identity_preserving_cleanup_gate": True,
            "primary_elapsed_scope": (
                "configure_train_panels_v13_plus_estimate_train_panels_v13"
            ),
            "descriptive_subphases": [
                "configure_train_panels_v13", "estimate_train_panels_v13",
                "fresh_engine_launch_and_model_load_excluded_from_primary",
            ],
        },
        "exact_equivalence_contract": {
            "full_diagnostic_content_commitment_equal_within_every_pair": True,
            "all_five_panel_coefficient_arrays_exact": True,
            "robust_aggregate_coefficients_exact": True,
            "pre_post_base_probe_identity_exact": True,
            "population_boundary_and_exact_reference_guards_exact": True,
            "no_numeric_tolerance_or_posthoc_exception": True,
        },
        "performance_analysis": {
            "inferential_endpoints": [
                "default_over_tuned_complete_train_step_elapsed_ratio",
                "tuned_over_default_peak_torch_allocated_ratio",
                "tuned_over_default_peak_torch_reserved_ratio",
            ],
            "subphase_timings_descriptive_not_inferential": True,
            "bootstrap_seed": BOOTSTRAP_SEED_V28C,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS_V28C,
            "bootstrap_draw_plan_sha256": draw_sha256,
            "paired_resampling_unit": "fresh_engine_group_pair",
            "familywise_method": "one_sided_bonferroni_three_endpoints",
            "familywise_alpha": FAMILYWISE_ALPHA_V28C,
            "lower_quantile": FAMILYWISE_LOWER_QUANTILE_V28C,
            "upper_quantile": FAMILYWISE_UPPER_QUANTILE_V28C,
            "speed_point_ratio_min": SPEED_POINT_RATIO_MIN_V28C,
            "speed_lcb_ratio_strictly_greater_than": SPEED_LCB_RATIO_MIN_V28C,
            "memory_point_ratio_max": MEMORY_POINT_RATIO_MAX_V28C,
            "memory_ucb_ratio_max": MEMORY_UCB_RATIO_MAX_V28C,
            "absolute_peak_nvml_fraction_max": MAX_PEAK_NVML_FRACTION_V28C,
            "no_posthoc_threshold_endpoint_or_bootstrap_adaptation": True,
        },
        "gate_and_authority": {
            "all_runtime_integrity_exact_equivalence_activity_and_cleanup_gates_required": True,
            "all_three_preregistered_performance_endpoints_required": True,
            "pass_authority": (
                "authorize_only_exact_v27c_table_in_a_separately_frozen_"
                "bf16_train_only_training_recipe"
            ),
            "failure_decision": "retain_empty_default_bf16_training_recipe",
            "direct_recipe_adoption_authorized": False,
            "model_update_authorized": False,
            "checkpoint_write_authorized": False,
            "evaluation_authorized": False,
            "dataset_promotion_authorized": False,
            "validation_heldout_ood_or_benchmark_access_authorized": False,
            "nontrain_runtime_reuse_authorized": False,
            "fp8_reuse_authorized": False,
        },
        "persistence_contract": {
            "compact_aggregate_ratios_bounds_commitments_and_booleans_only": True,
            "rows_prompts_answers_text_token_ids_raw_coefficients_diagnostics": False,
            "raw_timings_memory_samples_pids_bootstrap_draws": False,
        },
        "contains_dataset_rows_questions_answers_or_document_content": False,
        "contains_validation_heldout_ood_or_benchmark_content": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def validate_preregistration_v28c(value):
    design = value.get("paired_counterbalanced_design", {}) if isinstance(value, dict) else {}
    train = value.get("frozen_train_step_contract", {}) if isinstance(value, dict) else {}
    runtime = value.get("runtime_contract", {}) if isinstance(value, dict) else {}
    authority = value.get("gate_and_authority", {}) if isinstance(value, dict) else {}
    budget = value.get("request_budget", {}) if isinstance(value, dict) else {}
    if (
        not isinstance(value, dict)
        or value.get("schema") != SCHEMA_V28C
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(value))
        or value.get("strict_train_only") is not True
        or design.get("pair_count") != 12
        or design.get("default_first_pair_count") != 6
        or design.get("tuned_first_pair_count") != 6
        or tuple(design.get("pair_order", ())) != PAIR_ORDER_V28C
        or train.get("alpha") != 0.0
        or train.get("model_update_applied") is not False
        or budget.get("requests_per_arm") != 18_480
        or budget.get("total_generation_requests") != 443_520
        or runtime.get("fresh_engine_group_count") != 24
        or runtime.get("engines_per_group") != 4
        or runtime.get("tensor_parallel_size_per_engine") != 1
        or runtime.get("moe_backend") != "triton"
        or runtime.get("dtype") != "bfloat16"
        or runtime.get("prelaunch_committed_clean_source_contract") is not True
        or authority.get("pass_authority")
        != (
            "authorize_only_exact_v27c_table_in_a_separately_frozen_"
            "bf16_train_only_training_recipe"
        )
        or any(authority.get(key) is not False for key in (
            "direct_recipe_adoption_authorized", "model_update_authorized",
            "checkpoint_write_authorized", "evaluation_authorized",
            "dataset_promotion_authorized",
            "validation_heldout_ood_or_benchmark_access_authorized",
            "nontrain_runtime_reuse_authorized", "fp8_reuse_authorized",
        ))
        or value.get("contains_dataset_rows_questions_answers_or_document_content")
        is not False
        or value.get("contains_validation_heldout_ood_or_benchmark_content")
        is not False
    ):
        raise RuntimeError("V28C preregistration contract changed")
    for pair in PAIR_ORDER_V28C:
        if tuple(design["pairs"][pair]["arm_order"]) != ARM_ORDER_BY_PAIR_V28C[pair]:
            raise RuntimeError("V28C pair counterbalancing changed")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH_V28C)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    value = validate_preregistration_v28c(build_preregistration_v28c())
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
            raise RuntimeError("persisted V28C preregistration differs")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return value


if __name__ == "__main__":
    main()
