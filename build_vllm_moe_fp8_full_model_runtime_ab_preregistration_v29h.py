#!/usr/bin/env python3
"""Preregister the V29H serialized-FP8 full-model selected-table runtime A/B."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

import build_vllm_moe_fp8_selected_table_evaluation_retry_preregistration_v29e as v29e
import build_vllm_moe_fp8_tuning_preregistration_v29b as v29b


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH_V29H = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V29H_FP8_FULL_MODEL_RUNTIME_AB_PREREGISTRATION.json"
)
SCHEMA_V29H = "vllm-moe-fp8-full-model-runtime-ab-preregistration-v29h"

V29F_EVIDENCE_PATH_V29H = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V29F_V29E_FP8_SELECTED_TABLE_EVALUATION_POSITIVE_EVIDENCE.json"
)
V29F_EVIDENCE_COMMIT_V29H = "5fe27ca6eb297a10c7a26aefc51b7e3dde28aa47"
V29F_EVIDENCE_FILE_SHA256_V29H = (
    "2fe8c0a2f244ac1d2594904d21e1b28b1383274f05c341cc839b06c7940b7258"
)
V29F_EVIDENCE_CONTENT_SHA256_V29H = (
    "1d3bd12b9b447303fb4f46083567c1ae7db723dddee261fbf36a76928de3a7f9"
)
V29E_PREREG_PATH_V29H = v29e.OUTPUT_PATH_V29E
V29E_PREREG_COMMIT_V29H = "02ee7b7a7a1b0fd33b1e5f3db5c95ea2b32a11e6"
V29E_PREREG_FILE_SHA256_V29H = (
    "853a38e75bfe91baa21d0d4331dcfbd298f7828da529920dc3c244c81f908a1f"
)
V29E_PREREG_CONTENT_SHA256_V29H = (
    "5a8bb93c60631f5a1acb22d729c942a6f2630f8ad72b0698bc7c32ee5c3f089f"
)

MODEL_PATH_V29H = v29b.MODEL_PATH_V29B
MODEL_CONFIG_SHA256_V29H = v29b.MODEL_CONFIG_SHA256_V29B
MODEL_INDEX_SHA256_V29H = v29b.MODEL_INDEX_SHA256_V29B
MODEL_WEIGHT_SHARDS_V29H = copy.deepcopy(v29b.MODEL_WEIGHT_SHARDS_V29B)
MODEL_ALL_FILES_SIZE_MANIFEST_V29H = copy.deepcopy(
    v29b.MODEL_ALL_FILES_SIZE_MANIFEST_V29B
)
MODEL_ALL_FILES_FINGERPRINT_SHA256_V29H = (
    v29b.MODEL_ALL_FILES_FINGERPRINT_SHA256_V29B
)

TABLE_PATH_V29H = v29e.TABLE_PATH_V29E
TABLE_COMMIT_V29H = "a203f4821c4a737310df75543353d21ce6cea978"
TABLE_FILE_SHA256_V29H = (
    "1a4ed0f44c6d7cc788baecd073107b4634db4c769f0820d10174f61117b25618"
)
TABLE_CONTENT_SHA256_V29H = (
    "d4a49735ccfd094d6e5a3ee763eca99ed355a51fd11a7c835bfecf9fafeaa50d"
)
TABLE_LOADED_CONFIG_SHA256_V29H = (
    "ebf00590ac51e66e52f5e99b933d1be72703fbbcc809cc2d585eca8d6b0c0a5d"
)
EXPECTED_CONFIGS_V29H = copy.deepcopy(v29e.v29d.EXPECTED_CONFIGS_V29D)

PHYSICAL_GPU_IDS_V29H = (0, 1, 2, 3)
PROMPT_TOKENS_BY_GPU_V29H = {0: 256, 1: 512, 2: 1024, 3: 2048}
PAIR_COUNT_V29H = 8
PAIR_ORDER_V29H = tuple(f"pair_{index:02d}" for index in range(PAIR_COUNT_V29H))
PAIR_SEEDS_V29H = tuple(2_026_101_400 + index for index in range(PAIR_COUNT_V29H))
ARM_ORDER_BY_PAIR_V29H = {
    pair: (
        ("default_empty", "v29_selected_tuned")
        if index % 2 == 0 else ("v29_selected_tuned", "default_empty")
    )
    for index, pair in enumerate(PAIR_ORDER_V29H)
}
WARMUP_CALLS_PER_ENGINE_V29H = 1
DETERMINISM_CALLS_PER_ENGINE_V29H = 2
TIMING_CALLS_PER_ENGINE_V29H = 7
CALLS_PER_ENGINE_GROUP_V29H = (
    WARMUP_CALLS_PER_ENGINE_V29H
    + DETERMINISM_CALLS_PER_ENGINE_V29H
    + TIMING_CALLS_PER_ENGINE_V29H
)
ENGINE_GROUP_COUNT_V29H = PAIR_COUNT_V29H * 2
MODEL_LOAD_COUNT_V29H = ENGINE_GROUP_COUNT_V29H * 4
TOTAL_GENERATION_REQUESTS_V29H = (
    ENGINE_GROUP_COUNT_V29H * 4 * CALLS_PER_ENGINE_GROUP_V29H
)
TOTAL_PROMPT_TOKENS_V29H = (
    ENGINE_GROUP_COUNT_V29H
    * CALLS_PER_ENGINE_GROUP_V29H
    * sum(PROMPT_TOKENS_BY_GPU_V29H.values())
)
TOTAL_GENERATED_TOKENS_V29H = TOTAL_GENERATION_REQUESTS_V29H
TOTAL_ACTIVITY_WITNESSES_V29H = ENGINE_GROUP_COUNT_V29H * 4

BOOTSTRAP_SEED_V29H = 20_261_014
BOOTSTRAP_RESAMPLES_V29H = 50_000
ENDPOINT_COUNT_V29H = 10
FAMILYWISE_ALPHA_V29H = 0.05
PER_ENDPOINT_ALPHA_V29H = FAMILYWISE_ALPHA_V29H / ENDPOINT_COUNT_V29H
PER_GPU_LATENCY_POINT_MIN_V29H = 0.99
PER_GPU_LATENCY_LCB_MIN_V29H = 0.98
GLOBAL_LATENCY_POINT_MIN_V29H = 1.002
GLOBAL_LATENCY_LCB_MIN_V29H = 0.99
VRAM_POINT_RATIO_MAX_V29H = 1.01
VRAM_UCB_RATIO_MAX_V29H = 1.02
MAX_ABSOLUTE_NVML_FRACTION_V29H = 0.95


canonical_sha256 = v29e.canonical_sha256
file_sha256 = v29e.file_sha256


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _load_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"V29H requires JSON object: {path}")
    return value


def _committed_file_sha256(commit, path):
    relative = Path(path).resolve().relative_to(ROOT).as_posix()
    raw = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
    )
    return hashlib.sha256(raw).hexdigest()


def validate_static_inputs_v29h():
    v29e.validate_static_inputs_v29e()
    expected = (
        (V29F_EVIDENCE_PATH_V29H, V29F_EVIDENCE_FILE_SHA256_V29H,
         V29F_EVIDENCE_COMMIT_V29H),
        (V29E_PREREG_PATH_V29H, V29E_PREREG_FILE_SHA256_V29H,
         V29E_PREREG_COMMIT_V29H),
        (TABLE_PATH_V29H, TABLE_FILE_SHA256_V29H, TABLE_COMMIT_V29H),
    )
    for path, digest, commit in expected:
        if (
            not Path(path).is_file() or Path(path).is_symlink()
            or file_sha256(path) != digest
            or _committed_file_sha256(commit, path) != digest
        ):
            raise RuntimeError(f"V29H committed input identity changed: {path}")
    evidence = _load_json(V29F_EVIDENCE_PATH_V29H)
    decision = evidence.get("decision", {})
    if (
        evidence.get("schema")
        != "vllm-moe-fp8-selected-table-evaluation-positive-evidence-v29f"
        or evidence.get("status")
        != "valid_completed_synthetic_kernel_evaluation_passed"
        or evidence.get("content_sha256_before_self_field")
        != V29F_EVIDENCE_CONTENT_SHA256_V29H
        or canonical_sha256(_without_self(evidence))
        != V29F_EVIDENCE_CONTENT_SHA256_V29H
        or evidence.get("aggregate_result", {}).get("pass") is not True
        or decision.get(
            "authorize_only_separate_fp8_runtime_or_training_ab_preregistration"
        ) is not True
        or decision.get("direct_selected_table_adoption_authorized") is not False
        or decision.get("full_model_or_bf16_training_path_integration_demonstrated")
        is not False
        or decision.get("training_or_model_update_authorized") is not False
        or decision.get(
            "dataset_evaluation_validation_heldout_ood_or_benchmark_open_authorized"
        ) is not False
    ):
        raise RuntimeError("V29H V29F evidence authority changed")
    prior = _load_json(V29E_PREREG_PATH_V29H)
    if (
        prior.get("content_sha256_before_self_field")
        != V29E_PREREG_CONTENT_SHA256_V29H
        or canonical_sha256(_without_self(prior))
        != V29E_PREREG_CONTENT_SHA256_V29H
        or prior.get("selected_table", {}).get("file_sha256")
        != TABLE_FILE_SHA256_V29H
        or prior.get("selected_table", {}).get("content_sha256")
        != TABLE_CONTENT_SHA256_V29H
        or prior.get("model_identity", {}).get("all_files_fingerprint_sha256")
        != MODEL_ALL_FILES_FINGERPRINT_SHA256_V29H
    ):
        raise RuntimeError("V29H V29E frozen identity changed")
    table = _load_json(TABLE_PATH_V29H)
    loaded = {int(key): value for key, value in table.items()
              if key != "triton_version"}
    if (
        table.get("triton_version") != "3.6.0"
        or {key: value for key, value in table.items() if key != "triton_version"}
        != EXPECTED_CONFIGS_V29H
        or canonical_sha256(table) != TABLE_CONTENT_SHA256_V29H
        or canonical_sha256(loaded) != TABLE_LOADED_CONFIG_SHA256_V29H
    ):
        raise RuntimeError("V29H exact V29 selected table changed")
    if (
        file_sha256(MODEL_PATH_V29H / "config.json")
        != MODEL_CONFIG_SHA256_V29H
        or file_sha256(MODEL_PATH_V29H / "model.safetensors.index.json")
        != MODEL_INDEX_SHA256_V29H
    ):
        raise RuntimeError("V29H serialized FP8 model metadata changed")
    config = _load_json(MODEL_PATH_V29H / "config.json")
    text = config.get("text_config", {})
    quant = config.get("quantization_config", {})
    if (
        config.get("architectures") != ["Qwen3_5MoeForConditionalGeneration"]
        or text.get("num_experts") != 256
        or text.get("num_experts_per_tok") != 8
        or text.get("hidden_size") != 2048
        or text.get("dtype") != "bfloat16"
        or quant.get("quant_method") != "fp8"
        or quant.get("fmt") != "e4m3"
        or quant.get("activation_scheme") != "dynamic"
        or quant.get("weight_block_size") != [128, 128]
    ):
        raise RuntimeError("V29H serialized block-FP8 model geometry changed")
    index = _load_json(MODEL_PATH_V29H / "model.safetensors.index.json")
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    size_records = [
        {"file": path.name, "bytes": path.stat().st_size}
        for path in sorted(MODEL_PATH_V29H.iterdir()) if path.is_file()
    ]
    shard_records = [
        {"file": name, "bytes": (MODEL_PATH_V29H / name).stat().st_size}
        for name in shard_names
    ]
    if (
        len(size_records) != MODEL_ALL_FILES_SIZE_MANIFEST_V29H["file_count"]
        or sum(item["bytes"] for item in size_records)
        != MODEL_ALL_FILES_SIZE_MANIFEST_V29H["total_bytes"]
        or canonical_sha256(size_records)
        != MODEL_ALL_FILES_SIZE_MANIFEST_V29H["size_manifest_sha256"]
        or len(shard_records) != MODEL_WEIGHT_SHARDS_V29H["file_count"]
        or sum(item["bytes"] for item in shard_records)
        != MODEL_WEIGHT_SHARDS_V29H["total_bytes"]
    ):
        raise RuntimeError("V29H serialized FP8 file-size surface changed")
    return evidence, prior, table


def schedule_v29h():
    return [
        {
            "pair": pair,
            "pair_index": index,
            "synthetic_seed": PAIR_SEEDS_V29H[index],
            "arm_order": list(ARM_ORDER_BY_PAIR_V29H[pair]),
        }
        for index, pair in enumerate(PAIR_ORDER_V29H)
    ]


def bootstrap_draw_plan_v29h():
    generator = np.random.default_rng(BOOTSTRAP_SEED_V29H)
    pair_draws = generator.integers(
        0, PAIR_COUNT_V29H,
        size=(BOOTSTRAP_RESAMPLES_V29H, PAIR_COUNT_V29H), dtype=np.int64,
    )
    call_draws = generator.integers(
        0, TIMING_CALLS_PER_ENGINE_V29H,
        size=(BOOTSTRAP_RESAMPLES_V29H, TIMING_CALLS_PER_ENGINE_V29H),
        dtype=np.int64,
    )
    header = {
        "pair_shape": list(pair_draws.shape),
        "call_shape": list(call_draws.shape),
        "dtype": "int64",
        "same_hierarchical_paired_draws_all_ten_endpoints": True,
    }
    digest = hashlib.sha256()
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)
    digest.update(pair_draws.tobytes(order="C"))
    digest.update(call_draws.tobytes(order="C"))
    return pair_draws, call_draws, digest.hexdigest()


def build_preregistration_v29h():
    evidence, prior, table = validate_static_inputs_v29h()
    _pair_draws, _call_draws, draw_sha = bootstrap_draw_plan_v29h()
    endpoints = [
        *[f"gpu{gpu}_full_model_latency_speedup"
          for gpu in PHYSICAL_GPU_IDS_V29H],
        *[f"gpu{gpu}_peak_vram_ratio" for gpu in PHYSICAL_GPU_IDS_V29H],
        "global_full_model_latency_geometric_mean_speedup",
        "global_peak_vram_max_ratio",
    ]
    value = {
        "schema": SCHEMA_V29H,
        "status": "preregistered_before_serialized_fp8_full_model_ab_not_launched",
        "strict_synthetic_train_runtime_only": True,
        "authorization_basis": {
            "v29f_evidence_path": str(V29F_EVIDENCE_PATH_V29H),
            "v29f_evidence_commit": V29F_EVIDENCE_COMMIT_V29H,
            "v29f_evidence_file_sha256": V29F_EVIDENCE_FILE_SHA256_V29H,
            "v29f_evidence_content_sha256": V29F_EVIDENCE_CONTENT_SHA256_V29H,
            "v29f_kernel_global_speedup": evidence["aggregate_result"]["global"][
                "latency_geometric_mean_speedup"
            ],
            "authorizes_only_this_separate_full_model_runtime_ab_preregistration": True,
        },
        "prior_preregistration": {
            "path": str(V29E_PREREG_PATH_V29H),
            "commit": V29E_PREREG_COMMIT_V29H,
            "file_sha256": V29E_PREREG_FILE_SHA256_V29H,
            "content_sha256": V29E_PREREG_CONTENT_SHA256_V29H,
            "strict_synthetic_kernel_only": prior["strict_synthetic_kernel_only"],
        },
        "serialized_fp8_model_contract": {
            "path": str(MODEL_PATH_V29H),
            "config_sha256": MODEL_CONFIG_SHA256_V29H,
            "index_sha256": MODEL_INDEX_SHA256_V29H,
            "weight_shards": MODEL_WEIGHT_SHARDS_V29H,
            "all_files": {
                **MODEL_ALL_FILES_SIZE_MANIFEST_V29H,
                "fingerprint_sha256": MODEL_ALL_FILES_FINGERPRINT_SHA256_V29H,
                "real_launch_rehashes_all_56_files_before_gpu_claim": True,
            },
            "quantization": {
                "quant_method": "fp8", "format": "e4m3",
                "activation_scheme": "dynamic", "weight_block_size": [128, 128],
                "activation_dtype": "bfloat16",
            },
            "full_model_loaded_independently_on_every_tp1_engine": True,
        },
        "selected_table_contract": {
            "path": str(TABLE_PATH_V29H),
            "commit": TABLE_COMMIT_V29H,
            "file_sha256": TABLE_FILE_SHA256_V29H,
            "content_sha256": TABLE_CONTENT_SHA256_V29H,
            "loaded_config_sha256": TABLE_LOADED_CONFIG_SHA256_V29H,
            "triton_version": table["triton_version"],
            "exact_configs": EXPECTED_CONFIGS_V29H,
            "dtype_selector": "fp8_w8a8",
            "block_shape_selector": [128, 128],
        },
        "arms": {
            "default_empty": {
                "VLLM_TUNED_CONFIG_FOLDER": "fresh_exclusive_empty_directory",
                "expected_config_source": "generic_fp8_fallback_none",
            },
            "v29_selected_tuned": {
                "VLLM_TUNED_CONFIG_FOLDER": str(TABLE_PATH_V29H.parent),
                "expected_config_source": "exact_v29_selected_fp8_table",
            },
        },
        "only_intended_arm_difference": "VLLM_TUNED_CONFIG_FOLDER",
        "synthetic_request_contract": {
            "construction": (
                "integer_token_ids_200_plus_position_and_fixed_pair_seed_modulo_1000"
            ),
            "prompt_tokens_by_exact_physical_gpu": {
                str(key): value for key, value in PROMPT_TOKENS_BY_GPU_V29H.items()
            },
            "one_request_per_engine_per_call": True,
            "same_exact_token_ids_sampling_and_call_order_within_every_pair": True,
            "sampling": {
                "n": 1, "seed": 43, "temperature": 0.0, "top_p": 1.0,
                "max_tokens": 1, "logprobs": 1, "detokenize": False,
            },
            "raw_token_ids_or_decoded_text_persisted": False,
            "dataset_files_tokenizers_decoded_text_or_semantic_content_opened": False,
        },
        "schedule": {
            "pair_count": PAIR_COUNT_V29H,
            "fixed_pair_seeds": list(PAIR_SEEDS_V29H),
            "paired_counterbalanced_schedule": schedule_v29h(),
            "default_first_count": 4,
            "tuned_first_count": 4,
            "fresh_four_tp1_engine_group_per_arm": True,
            "fresh_ray_runtime_and_full_model_loads_per_arm": True,
            "warmup_calls_per_engine": WARMUP_CALLS_PER_ENGINE_V29H,
            "determinism_calls_per_engine": DETERMINISM_CALLS_PER_ENGINE_V29H,
            "timing_calls_per_engine": TIMING_CALLS_PER_ENGINE_V29H,
            "no_posthoc_schedule_or_repetition_adaptation": True,
        },
        "request_budget": {
            "fresh_four_engine_groups": ENGINE_GROUP_COUNT_V29H,
            "full_model_tp1_loads": MODEL_LOAD_COUNT_V29H,
            "generation_calls_per_engine_per_group": CALLS_PER_ENGINE_GROUP_V29H,
            "total_generation_requests": TOTAL_GENERATION_REQUESTS_V29H,
            "total_prompt_tokens": TOTAL_PROMPT_TOKENS_V29H,
            "total_generated_tokens": TOTAL_GENERATED_TOKENS_V29H,
            "synchronized_activity_witnesses": TOTAL_ACTIVITY_WITNESSES_V29H,
        },
        "runtime_contract": {
            "physical_gpu_ids": list(PHYSICAL_GPU_IDS_V29H),
            "engines_per_group": 4,
            "tensor_parallel_size_per_engine": 1,
            "moe_backend": "triton",
            "dtype": "bfloat16",
            "quantization_from_serialized_checkpoint": "fp8_block_128x128",
            "gpu_memory_utilization": 0.82,
            "max_model_len": 4096,
            "enable_prefix_caching": False,
            "enforce_eager": True,
            "exact_config_source_verified_inside_every_actor": True,
            "placement_groups_mapped_by_physical_gpu_probe": True,
            "one_distinct_actor_pid_per_exact_gpu_uuid": True,
            "simultaneous_all_four_pid_bound_positive_activity_required_per_group": True,
            "all_four_idle_before_claim_between_every_group_and_after_cleanup": True,
            "launch_load_compile_warmup_determinism_and_cleanup_excluded_from_timing": True,
            "timing_clock": "actor_local_perf_counter_ns_with_cuda_synchronize",
            "peak_vram": "per_timed_call_torch_peak_allocated_and_reserved",
            "committed_clean_source_and_fresh_exclusive_paths_required": True,
        },
        "exact_equivalence_contract": {
            "generated_token_ids_exact": True,
            "selected_generated_token_logprob_exact": True,
            "cumulative_logprob_exact": True,
            "output_dtype_shape_and_commitment_exact": True,
            "all_determinism_and_timed_calls_match_within_arm": True,
            "default_and_tuned_exact_for_every_pair_gpu_and_call": True,
            "no_tolerance_or_posthoc_exception": True,
        },
        "statistical_contract": {
            "paired_unit": "fresh_engine_group_pair_with_seven_matched_calls",
            "bootstrap_seed": BOOTSTRAP_SEED_V29H,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES_V29H,
            "bootstrap_draw_plan_sha256": draw_sha,
            "bootstrap_statistic": "hierarchical_pair_then_call_resampled_median",
            "familywise_alpha": FAMILYWISE_ALPHA_V29H,
            "multiplicity": "Bonferroni_across_all_10_one_sided_endpoints",
            "per_endpoint_one_sided_alpha": PER_ENDPOINT_ALPHA_V29H,
            "endpoints": endpoints,
            "per_gpu_latency_point_ratio_min": PER_GPU_LATENCY_POINT_MIN_V29H,
            "per_gpu_latency_lcb_ratio_min": PER_GPU_LATENCY_LCB_MIN_V29H,
            "global_latency_point_ratio_min": GLOBAL_LATENCY_POINT_MIN_V29H,
            "global_latency_lcb_ratio_min": GLOBAL_LATENCY_LCB_MIN_V29H,
            "vram_point_ratio_max": VRAM_POINT_RATIO_MAX_V29H,
            "vram_ucb_ratio_max": VRAM_UCB_RATIO_MAX_V29H,
            "absolute_nvml_fraction_max": MAX_ABSOLUTE_NVML_FRACTION_V29H,
            "thresholds_selected_before_full_model_gpu_outputs": True,
        },
        "gate_and_authority": {
            "all_exact_equivalence_config_activity_identity_cleanup_and_performance_gates_required": True,
            "pass_authority": (
                "authorize_only_exact_v29_table_in_a_separately_frozen_"
                "serialized_fp8_train_only_recipe_ab"
            ),
            "failure_decision": "retain_empty_default_serialized_fp8_runtime",
            "direct_table_or_recipe_adoption_authorized": False,
            "model_update_or_training_authorized": False,
            "checkpoint_write_authorized": False,
            "dataset_promotion_authorized": False,
            "evaluation_validation_heldout_ood_or_benchmark_access_authorized": False,
            "nontrain_runtime_reuse_authorized": False,
            "bf16_table_reuse_authorized": False,
        },
        "persistence_contract": {
            "compact_aggregate_ratios_bounds_hashes_counts_and_booleans_only": True,
            "raw_tokens_outputs_logprobs_timings_memory_samples_pids_or_bootstrap_draws": False,
            "dataset_rows_prompts_answers_text_or_semantic_content": False,
        },
        "contains_dataset_rows_questions_answers_or_document_content": False,
        "contains_validation_heldout_ood_or_benchmark_content": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return validate_preregistration_v29h(value)


def validate_preregistration_v29h(value):
    schedule = value.get("schedule", {}) if isinstance(value, dict) else {}
    budget = value.get("request_budget", {}) if isinstance(value, dict) else {}
    runtime = value.get("runtime_contract", {}) if isinstance(value, dict) else {}
    authority = value.get("gate_and_authority", {}) if isinstance(value, dict) else {}
    if (
        not isinstance(value, dict)
        or value.get("schema") != SCHEMA_V29H
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(value))
        or value.get("strict_synthetic_train_runtime_only") is not True
        or schedule.get("pair_count") != 8
        or schedule.get("default_first_count") != 4
        or schedule.get("tuned_first_count") != 4
        or schedule.get("timing_calls_per_engine") != 7
        or budget.get("fresh_four_engine_groups") != 16
        or budget.get("full_model_tp1_loads") != 64
        or budget.get("total_generation_requests") != 640
        or budget.get("total_prompt_tokens") != 614_400
        or budget.get("total_generated_tokens") != 640
        or budget.get("synchronized_activity_witnesses") != 64
        or runtime.get("moe_backend") != "triton"
        or runtime.get("quantization_from_serialized_checkpoint")
        != "fp8_block_128x128"
        or authority.get("pass_authority")
        != (
            "authorize_only_exact_v29_table_in_a_separately_frozen_"
            "serialized_fp8_train_only_recipe_ab"
        )
        or any(authority.get(key) is not False for key in (
            "direct_table_or_recipe_adoption_authorized",
            "model_update_or_training_authorized", "checkpoint_write_authorized",
            "dataset_promotion_authorized",
            "evaluation_validation_heldout_ood_or_benchmark_access_authorized",
            "nontrain_runtime_reuse_authorized", "bf16_table_reuse_authorized",
        ))
        or value.get("contains_dataset_rows_questions_answers_or_document_content")
        is not False
        or value.get("contains_validation_heldout_ood_or_benchmark_content")
        is not False
    ):
        raise RuntimeError("V29H preregistration contract changed")
    for index, item in enumerate(schedule["paired_counterbalanced_schedule"]):
        pair = PAIR_ORDER_V29H[index]
        if (
            item["pair"] != pair
            or tuple(item["arm_order"]) != ARM_ORDER_BY_PAIR_V29H[pair]
            or item["synthetic_seed"] != PAIR_SEEDS_V29H[index]
        ):
            raise RuntimeError("V29H counterbalanced schedule changed")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH_V29H)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    value = build_preregistration_v29h()
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
            raise RuntimeError("persisted V29H preregistration differs")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return value


if __name__ == "__main__":
    main()
