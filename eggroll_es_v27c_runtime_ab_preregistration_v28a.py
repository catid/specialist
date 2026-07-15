#!/usr/bin/env python3
"""Preregister the train-only end-to-end V27C tuned-runtime A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CANONICAL_ROOT = Path("/home/catid/specialist")
OUTPUT_PATH_V28A = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V28A_V27C_TUNED_RUNTIME_AB_PREREGISTRATION.json"
)
SCHEMA_V28A = "eggroll-es-v27c-tuned-runtime-ab-preregistration-v28a"

MODEL_PATH_V28A = CANONICAL_ROOT / "models/Qwen3.6-35B-A3B"
MODEL_CONFIG_SHA256_V28A = (
    "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
)
MODEL_INDEX_SHA256_V28A = (
    "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"
)
MODEL_WEIGHT_SHARD_MANIFEST_V28A = {
    "file_count": 26,
    "total_bytes": 71_903_776_776,
    "manifest_sha256": (
        "2b0bbae17d50cbf1a6c80ebcb0198dde8a09bbf7d3f4dec6321b1c97b4d3b2b1"
    ),
}
MODEL_METADATA_SHA256_V28A = {
    "config.json": MODEL_CONFIG_SHA256_V28A,
    "generation_config.json": (
        "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"
    ),
    "model.safetensors.index.json": MODEL_INDEX_SHA256_V28A,
    "tokenizer_config.json": (
        "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b"
    ),
    "tokenizer.json": (
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    ),
    "chat_template.jinja": (
        "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"
    ),
}
MODEL_SEAL_PATH_V28A = CANONICAL_ROOT / (
    "experiments/eggroll_es_hpo/S6_V23A_INSERTION_MODEL_SEAL.json"
)
MODEL_SEAL_COMMIT_V28A = "00471e1c44b78813d02f0a8895c8e014a75dcc4e"
MODEL_SEAL_FILE_SHA256_V28A = (
    "96eeb236ea94678f57a530a27a471467d4b3d413d2e7be397e293b695cd4c440"
)
MODEL_SEAL_CONTENT_SHA256_V28A = (
    "d4cf795408967aefbc77f841c47e6fe2fbe3cefc14a4a0fdb4bf73b2701326f9"
)
MODEL_BASE_ALL_FILES_FINGERPRINT_SHA256_V28A = (
    "1a21a765e374e266037b3b7e5313a62a0de8ca37c00c0462b67c21af7e21f61e"
)

TUNED_DIRECTORY_V28A = CANONICAL_ROOT / (
    "experiments/vllm_moe_tuning/"
    "v025_rtx_pro_6000_bf16_tp1_exhaustive_v27c"
)
TUNED_FILENAME_V28A = (
    "E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_"
    "Max-Q_Workstation_Edition.json"
)
TUNED_TABLE_PATH_V28A = TUNED_DIRECTORY_V28A / TUNED_FILENAME_V28A
TUNED_TABLE_FILE_SHA256_V28A = (
    "128806798a5bf8a961a5bd0bc8765c82e8b73a116e6c7411e7aeba5522667562"
)
TUNED_TABLE_CONTENT_SHA256_V28A = (
    "a4f82f53b037f766536013bdc10c8ca1e49873603a8f44972ef8007ed406de84"
)
TUNED_TABLE_COMMIT_V28A = "27f5aae6d1e9cfd3c53dc4f01b92a5414da5c5c8"
ALLOWED_CONFIG_KEYS_V28A = {
    "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
    "num_warps", "num_stages",
}

POSITIVE_EVIDENCE_PATH_V28A = CANONICAL_ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V27D_MOE_TUNING_EVALUATION_POSITIVE_EVIDENCE.json"
)
POSITIVE_EVIDENCE_COMMIT_V28A = "76e3b10730e5a7be9c5f2c298bc75095abf5a9c8"
POSITIVE_EVIDENCE_FILE_SHA256_V28A = (
    "f99fee6719d3ccb1d7973d540bca844c645d21ff36e6608c8a9293cfaa91ee4c"
)
POSITIVE_EVIDENCE_CONTENT_SHA256_V28A = (
    "c6a58230dd221fbf1f18baa67144672fa37ad1fcb81e81af12cca9eaf0ec8b20"
)
POSITIVE_EVIDENCE_CODE_V28A = {
    "builder": {
        "path": "build_vllm_moe_tuning_evaluation_positive_evidence_v27d.py",
        "file_sha256": (
            "c94279cc63178014e6c218c731ebc5ee1d17bd435724c0463856b2e212c2be31"
        ),
    },
    "tests": {
        "path": "test_build_vllm_moe_tuning_evaluation_positive_evidence_v27d.py",
        "file_sha256": (
            "51d0c64d629caaf2e6e0fc2b29132377cb865cb2096846886550650958068b8b"
        ),
    },
}

PANEL_MANIFEST_V28A = CANONICAL_ROOT / (
    "experiments/eggroll_es_hpo/train_panel_sampling_v13/"
    "document_balanced_train_panels_v13.json"
)
PANEL_MANIFEST_FILE_SHA256_V28A = (
    "e555d9d6746cde6297cd3ab523b16dd7d78d81e2674447ee46d754ebfac52da7"
)
PANEL_MANIFEST_CONTENT_SHA256_V28A = (
    "46cc98b694c98c1ee1c5456b855fb3b1db4534b3df2dcda69fc690a2d8a61bf5"
)
PANEL_BUNDLE_CONTENT_SHA256_V28A = (
    "cc176a9b86c6447dcde8a11fd28d68c837d2119715126c57a3f37293fb0d492b"
)
FROZEN_TRAIN_SOURCE_V28A = Path(
    "/tmp/specialist-s6-candidate-guarded-ead1b21/train_qa_curated_v1.jsonl"
)
FROZEN_TRAIN_SOURCE_SHA256_V28A = (
    "f7127c38c7b540eaf9cf4349d1a1b8076e171da7f8ea43c11068ad1c311bb776"
)
FROZEN_TRAIN_ARROW_SHA256_V28A = (
    "6b6fdfdd082f1de2bf1b4c78bd0a4154af5c709b26e46b0677dcde695d3b4cb6"
)
PANEL_NAMES_V28A = (
    "optimization_0", "optimization_1", "optimization_2",
    "train_screen_0", "train_screen_1",
)

WAVE_ORDER_V28A = ("wave_a", "wave_b")
PHYSICAL_GPU_IDS_V28A = (0, 1, 2, 3)
ARM_ORDER_V28A = ("default_empty", "v27c_tuned")
WAVE_ARM_ORDER_V28A = {
    "wave_a": ("default_empty", "v27c_tuned", "default_empty", "v27c_tuned"),
    "wave_b": ("v27c_tuned", "default_empty", "v27c_tuned", "default_empty"),
}
PAIR_ORDER_V28A = tuple(f"physical_gpu_{item}" for item in PHYSICAL_GPU_IDS_V28A)
REQUESTS_PER_ENGINE_V28A = 280
WARMUP_REPETITIONS_V28A = 2
DETERMINISM_REPETITIONS_V28A = 2
TIMING_REPETITIONS_V28A = 9
BOOTSTRAP_REPETITIONS_V28A = 20_000
BOOTSTRAP_SEED_V28A = 20260928
FAMILYWISE_LOWER_QUANTILE_V28A = 0.05 / 4
FAMILYWISE_UPPER_QUANTILE_V28A = 1.0 - (0.05 / 4)
PER_GPU_THROUGHPUT_NONREGRESSION_V28A = 0.98
PER_GPU_POINT_THROUGHPUT_MIN_V28A = 1.0
GLOBAL_POINT_THROUGHPUT_IMPROVEMENT_V28A = 1.01
GLOBAL_LCB_THROUGHPUT_IMPROVEMENT_V28A = 1.0
PER_GPU_VRAM_UCB_NONREGRESSION_V28A = 1.02
PER_GPU_POINT_VRAM_RATIO_MAX_V28A = 1.01
MAX_PEAK_NVML_FRACTION_V28A = 0.95


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _load_json_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"V28A requires a JSON object: {path}")
    return value


def _committed_file_sha256(commit, path):
    relative = Path(path).resolve().relative_to(CANONICAL_ROOT).as_posix()
    raw = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=CANONICAL_ROOT,
    )
    return hashlib.sha256(raw).hexdigest()


def _model_shard_manifest_v28a():
    index = _load_json_object(MODEL_PATH_V28A / "model.safetensors.index.json")
    files = sorted(set(index.get("weight_map", {}).values()))
    entries = [
        {"name": name, "size_bytes": (MODEL_PATH_V28A / name).stat().st_size}
        for name in files
    ]
    return {
        "file_count": len(entries),
        "total_bytes": sum(item["size_bytes"] for item in entries),
        "manifest_sha256": canonical_sha256(entries),
    }


def validate_static_inputs_v28a():
    if (
        file_sha256(MODEL_PATH_V28A / "config.json")
        != MODEL_CONFIG_SHA256_V28A
        or file_sha256(MODEL_PATH_V28A / "model.safetensors.index.json")
        != MODEL_INDEX_SHA256_V28A
        or _model_shard_manifest_v28a() != MODEL_WEIGHT_SHARD_MANIFEST_V28A
    ):
        raise RuntimeError("V28A exact Qwen3.6 model identity changed")
    if (
        file_sha256(MODEL_SEAL_PATH_V28A) != MODEL_SEAL_FILE_SHA256_V28A
        or _committed_file_sha256(MODEL_SEAL_COMMIT_V28A, MODEL_SEAL_PATH_V28A)
        != MODEL_SEAL_FILE_SHA256_V28A
    ):
        raise RuntimeError("V28A exact committed model seal identity changed")
    model_seal = _load_json_object(MODEL_SEAL_PATH_V28A)
    base_seal = model_seal.get("arms", {}).get("base_middle_late", {})
    if (
        model_seal.get("content_sha256_before_self_field")
        != MODEL_SEAL_CONTENT_SHA256_V28A
        or canonical_sha256(_without_self(model_seal))
        != MODEL_SEAL_CONTENT_SHA256_V28A
        or base_seal.get("path") != str(MODEL_PATH_V28A)
        or base_seal.get("config_sha256") != MODEL_CONFIG_SHA256_V28A
        or base_seal.get("index_sha256") != MODEL_INDEX_SHA256_V28A
        or base_seal.get("shard_count") != 26
        or base_seal.get("all_files_fingerprint_sha256")
        != MODEL_BASE_ALL_FILES_FINGERPRINT_SHA256_V28A
        or set(base_seal.get("shards", {})) != {
            f"model-{index:05d}-of-00026.safetensors"
            for index in range(1, 27)
        }
    ):
        raise RuntimeError("V28A exact base-model seal semantics changed")
    if (
        tuple(sorted(item.name for item in TUNED_DIRECTORY_V28A.iterdir()))
        != (TUNED_FILENAME_V28A,)
        or file_sha256(TUNED_TABLE_PATH_V28A)
        != TUNED_TABLE_FILE_SHA256_V28A
        or _committed_file_sha256(TUNED_TABLE_COMMIT_V28A, TUNED_TABLE_PATH_V28A)
        != TUNED_TABLE_FILE_SHA256_V28A
    ):
        raise RuntimeError("V28A exact committed V27C table identity changed")
    table = _load_json_object(TUNED_TABLE_PATH_V28A)
    if (
        canonical_sha256(table) != TUNED_TABLE_CONTENT_SHA256_V28A
        or table.get("triton_version") != "3.6.0"
        or {int(key) for key in table if key != "triton_version"}
        != {256, 512, 1024, 2048}
        or any(
            set(config) != ALLOWED_CONFIG_KEYS_V28A
            or any(isinstance(item, bool) or not isinstance(item, int)
                   for item in config.values())
            for key, config in table.items() if key != "triton_version"
        )
    ):
        raise RuntimeError("V28A V27C table semantics changed")
    if (
        file_sha256(POSITIVE_EVIDENCE_PATH_V28A)
        != POSITIVE_EVIDENCE_FILE_SHA256_V28A
        or _committed_file_sha256(
            POSITIVE_EVIDENCE_COMMIT_V28A, POSITIVE_EVIDENCE_PATH_V28A,
        ) != POSITIVE_EVIDENCE_FILE_SHA256_V28A
    ):
        raise RuntimeError("V28A positive-evidence commit binding changed")
    evidence = _load_json_object(POSITIVE_EVIDENCE_PATH_V28A)
    if (
        evidence.get("content_sha256_before_self_field")
        != POSITIVE_EVIDENCE_CONTENT_SHA256_V28A
        or canonical_sha256(_without_self(evidence))
        != POSITIVE_EVIDENCE_CONTENT_SHA256_V28A
        or evidence.get("aggregate_result", {}).get("global_gate_passed") is not True
        or evidence.get("decision", {}).get(
            "authorize_separate_end_to_end_train_only_runtime_ab_preregistration"
        ) is not True
        or evidence.get("decision", {}).get("direct_recipe_adoption_authorized")
        is not False
        or evidence.get("contains_validation_heldout_ood_or_benchmark_content")
        is not False
    ):
        raise RuntimeError("V28A positive-evidence authority changed")
    for item in POSITIVE_EVIDENCE_CODE_V28A.values():
        if file_sha256(CANONICAL_ROOT / item["path"]) != item["file_sha256"]:
            raise RuntimeError("V28A positive-evidence code identity changed")
    if (
        file_sha256(PANEL_MANIFEST_V28A) != PANEL_MANIFEST_FILE_SHA256_V28A
        or file_sha256(FROZEN_TRAIN_SOURCE_V28A)
        != FROZEN_TRAIN_SOURCE_SHA256_V28A
    ):
        raise RuntimeError("V28A frozen V13 train-only input identity changed")
    return table, evidence, base_seal


def bootstrap_draw_plan_v28a():
    generator = np.random.default_rng(BOOTSTRAP_SEED_V28A)
    repetition_draws = generator.integers(
        0, TIMING_REPETITIONS_V28A,
        size=(BOOTSTRAP_REPETITIONS_V28A, TIMING_REPETITIONS_V28A),
        dtype=np.int64,
    )
    gpu_draws = generator.integers(
        0, len(PHYSICAL_GPU_IDS_V28A),
        size=(BOOTSTRAP_REPETITIONS_V28A, len(PHYSICAL_GPU_IDS_V28A)),
        dtype=np.int64,
    )
    header = {
        "repetition_shape": list(repetition_draws.shape),
        "gpu_shape": list(gpu_draws.shape),
        "dtype": "int64",
        "same_repetition_draws_all_physical_gpu_pairs": True,
        "hierarchical_global_bootstrap_gpu_then_repetition": True,
    }
    digest = hashlib.sha256()
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)
    digest.update(repetition_draws.tobytes(order="C"))
    digest.update(gpu_draws.tobytes(order="C"))
    return repetition_draws, gpu_draws, digest.hexdigest()


def build_preregistration_v28a():
    table, evidence, base_seal = validate_static_inputs_v28a()
    _repetition_draws, _gpu_draws, draw_sha256 = bootstrap_draw_plan_v28a()
    arms = {
        "default_empty": {
            "VLLM_TUNED_CONFIG_FOLDER": "fresh_dedicated_empty_directory",
            "expected_config_source": "generic_fallback_none",
        },
        "v27c_tuned": {
            "VLLM_TUNED_CONFIG_FOLDER": str(TUNED_DIRECTORY_V28A),
            "expected_config_source": "exact_committed_v27c_table",
            "table_file_sha256": TUNED_TABLE_FILE_SHA256_V28A,
            "table_content_sha256": TUNED_TABLE_CONTENT_SHA256_V28A,
        },
    }
    waves = {}
    for wave_index, wave in enumerate(WAVE_ORDER_V28A):
        assignments = {}
        for gpu_id, arm in zip(
            PHYSICAL_GPU_IDS_V28A, WAVE_ARM_ORDER_V28A[wave], strict=True,
        ):
            assignments[str(gpu_id)] = {
                "cell": f"{wave}_gpu_{gpu_id}",
                "physical_gpu_id": gpu_id,
                "arm": arm,
                "model_path": str(MODEL_PATH_V28A),
            }
        waves[wave] = {
            "wave_index": wave_index,
            "gpu_assignments": assignments,
            "all_four_gpus_active_concurrently": True,
            "fresh_ray_actors_engines_and_model_loads_before_wave": True,
        }
    pairs = {}
    for gpu_id in PHYSICAL_GPU_IDS_V28A:
        default_wave = next(
            wave for wave in WAVE_ORDER_V28A
            if WAVE_ARM_ORDER_V28A[wave][gpu_id] == "default_empty"
        )
        tuned_wave = next(
            wave for wave in WAVE_ORDER_V28A
            if WAVE_ARM_ORDER_V28A[wave][gpu_id] == "v27c_tuned"
        )
        pairs[f"physical_gpu_{gpu_id}"] = {
            "physical_gpu_id": gpu_id,
            "reference_arm": "default_empty",
            "reference_wave": default_wave,
            "reference_cell": f"{default_wave}_gpu_{gpu_id}",
            "candidate_arm": "v27c_tuned",
            "candidate_wave": tuned_wave,
            "candidate_cell": f"{tuned_wave}_gpu_{gpu_id}",
            "load_order": (
                "default_first" if default_wave == "wave_a" else "tuned_first"
            ),
        }
    value = {
        "schema": SCHEMA_V28A,
        "experiment_name": "s6_v28a_v27c_tuned_vs_empty_default_train_runtime_ab",
        "status": "preregistered_future_runtime_not_launched",
        "positive_evidence_commit": POSITIVE_EVIDENCE_COMMIT_V28A,
        "preregistered_before_any_v28a_task_runtime_output": True,
        "strict_train_only": True,
        "authorization_basis": {
            "positive_evidence_path": str(POSITIVE_EVIDENCE_PATH_V28A),
            "file_sha256": POSITIVE_EVIDENCE_FILE_SHA256_V28A,
            "content_sha256": POSITIVE_EVIDENCE_CONTENT_SHA256_V28A,
            "committed_at": POSITIVE_EVIDENCE_COMMIT_V28A,
            "kernel_global_geometric_mean_speedup": evidence[
                "aggregate_result"
            ]["global_geometric_mean_speedup"],
            "authorizes_only_this_separate_train_only_runtime_ab_preregistration": True,
        },
        "model_contract": {
            "path": str(MODEL_PATH_V28A),
            "config_sha256": MODEL_CONFIG_SHA256_V28A,
            "index_sha256": MODEL_INDEX_SHA256_V28A,
            "weight_shard_manifest": MODEL_WEIGHT_SHARD_MANIFEST_V28A,
            "metadata_file_sha256": MODEL_METADATA_SHA256_V28A,
            "committed_full_shard_model_seal": {
                "path": str(MODEL_SEAL_PATH_V28A),
                "commit": MODEL_SEAL_COMMIT_V28A,
                "file_sha256": MODEL_SEAL_FILE_SHA256_V28A,
                "content_sha256": MODEL_SEAL_CONTENT_SHA256_V28A,
                "base_all_files_fingerprint_sha256": base_seal[
                    "all_files_fingerprint_sha256"
                ],
                "base_shards": base_seal["shards"],
            },
            "same_exact_model_all_eight_wave_gpu_cells": True,
        },
        "tuned_table_contract": {
            "directory": str(TUNED_DIRECTORY_V28A),
            "directory_contains_exactly_one_file": True,
            "filename": TUNED_FILENAME_V28A,
            "file_sha256": TUNED_TABLE_FILE_SHA256_V28A,
            "content_sha256": TUNED_TABLE_CONTENT_SHA256_V28A,
            "commit": TUNED_TABLE_COMMIT_V28A,
            "triton_version": table["triton_version"],
            "batch_keys": [256, 512, 1024, 2048],
        },
        "arms": arms,
        "only_intended_arm_difference": "VLLM_TUNED_CONFIG_FOLDER",
        "waves": waves,
        "pairing": {
            "pair_order": list(PAIR_ORDER_V28A),
            "physical_gpu_pairs": pairs,
            "default_first_physical_gpu_ids": [0, 2],
            "tuned_first_physical_gpu_ids": [1, 3],
            "same_request_tokens_sampling_and_repetition_index_all_cells": True,
        },
        "train_request_contract": {
            "panel_manifest_path": str(PANEL_MANIFEST_V28A),
            "panel_manifest_file_sha256": PANEL_MANIFEST_FILE_SHA256_V28A,
            "panel_manifest_content_sha256": PANEL_MANIFEST_CONTENT_SHA256_V28A,
            "panel_bundle_content_sha256": PANEL_BUNDLE_CONTENT_SHA256_V28A,
            "frozen_train_source_path": str(FROZEN_TRAIN_SOURCE_V28A),
            "frozen_train_source_file_sha256": FROZEN_TRAIN_SOURCE_SHA256_V28A,
            "frozen_train_source_arrow_sha256": FROZEN_TRAIN_ARROW_SHA256_V28A,
            "panel_names": list(PANEL_NAMES_V28A),
            "rows_per_panel": 56,
            "request_count_per_engine_call": REQUESTS_PER_ENGINE_V28A,
            "driver_materializes_and_tokenizes_once_then_passes_identical_prompt_token_ids": True,
            "all_row_content_and_token_ids_memory_only": True,
            "maximum_prompt_plus_answer_tokens": 1024,
            "contains_validation_heldout_ood_or_benchmark_content": False,
        },
        "sampling_contract": {
            "n": 1, "seed": 43, "temperature": 0.0, "top_p": 1.0,
            "max_tokens": 1, "prompt_logprobs": 1, "logprobs": 1,
            "detokenize": True, "prefix_caching": False, "enforce_eager": True,
        },
        "runtime_contract": {
            "wave_count": 2,
            "engines_per_wave": 4,
            "tensor_parallel_size_per_engine": 1,
            "physical_gpu_ids": [0, 1, 2, 3],
            "placement_groups_discovered_by_lightweight_string_int_gpu_probe": True,
            "placement_group_creation_order_never_defines_physical_gpu": True,
            "placement_groups_non_detached_and_removed_after_each_wave": True,
            "one_actor_and_complete_model_per_physical_gpu_per_wave": True,
            "nvml_index_uuid_pci_and_total_memory_bound_across_both_waves": True,
            "fresh_actors_engines_model_load_and_config_environment_each_wave": True,
            "parent_moe_override_environment_must_be_empty": True,
            "per_actor_runtime_environment_differs_only_by_tuned_config_folder": True,
            "default_directory_must_be_fresh_dedicated_and_empty": True,
            "actor_must_verify_generic_none_or_exact_table_config_source": True,
            "moe_backend": "triton",
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.82,
            "max_model_len": 2048,
            "enable_prefix_caching": False,
            "limit_mm_per_prompt": {"image": 0, "video": 0},
            "mm_processor_cache_gb": 0,
            "skip_mm_profiling": True,
            "vllm_batch_invariant": False,
            "runtime_versions": {
                "ray": "2.56.0",
                "torch": "2.11.0+cu130",
                "vllm": "0.25.0",
            },
            "warmup_repetitions": WARMUP_REPETITIONS_V28A,
            "determinism_repetitions": DETERMINISM_REPETITIONS_V28A,
            "timing_repetitions": TIMING_REPETITIONS_V28A,
            "generation_calls_per_engine_per_wave": 13,
            "total_engine_generation_calls_all_waves": 104,
            "total_generation_requests_all_engines_all_waves": 29_120,
            "model_load_compile_warmup_and_teardown_excluded_from_timing": True,
            "timing_clock": "actor-local perf_counter_ns bracketed by CUDA synchronization",
            "throughput_definition": (
                "fixed aggregate prompt tokens plus one generated token per request divided "
                "by actor-local synchronized task-generation seconds"
            ),
            "timing_and_memory_paired_by_physical_gpu_and_repetition_across_waves": True,
            "peak_vram": (
                "maximum post-warmup resident audit and 10ms actor-local NVML samples "
                "during each timed task generation"
            ),
            "interwave_cleanup_poll_timeout_seconds": 30.0,
            "interwave_cleanup_poll_interval_seconds": 0.5,
            "interwave_poll_preserves_exact_physical_gpu_identity": True,
            "final_cleanup_poll_timeout_seconds": 30.0,
            "final_cleanup_poll_interval_seconds": 0.5,
            "final_poll_preserves_exact_physical_gpu_identity": True,
            "successful_report_requires_all_four_gpus_idle_after_wave_b_cleanup": True,
            "prelaunch_idle_check_after_all_cpu_disk_audits_immediately_before_claim": True,
        },
        "exact_output_equivalence": {
            "generated_token_id_lists_exact": True,
            "generated_text_exact": True,
            "generated_selected_token_logprob_exact": True,
            "generated_cumulative_logprob_exact": True,
            "full_gold_answer_dense_prompt_logprob_payload_exact": True,
            "within_cell_determinism_and_all_timed_calls_exact": True,
            "same_arm_exact_across_all_four_physical_gpu_appearances": True,
            "default_and_tuned_exact_on_every_physical_gpu_pair": True,
            "no_numeric_tolerance_or_posthoc_exception": True,
        },
        "performance_analysis": {
            "bootstrap_seed": BOOTSTRAP_SEED_V28A,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS_V28A,
            "bootstrap_draw_plan_sha256": draw_sha256,
            "per_gpu_familywise_lower_quantile": FAMILYWISE_LOWER_QUANTILE_V28A,
            "per_gpu_familywise_upper_quantile": FAMILYWISE_UPPER_QUANTILE_V28A,
            "per_gpu_throughput_lcb_nonregression_ratio": (
                PER_GPU_THROUGHPUT_NONREGRESSION_V28A
            ),
            "per_gpu_observed_median_throughput_ratio_min": (
                PER_GPU_POINT_THROUGHPUT_MIN_V28A
            ),
            "global_observed_median_throughput_improvement_ratio": (
                GLOBAL_POINT_THROUGHPUT_IMPROVEMENT_V28A
            ),
            "global_bootstrap_lcb_throughput_improvement_ratio": (
                GLOBAL_LCB_THROUGHPUT_IMPROVEMENT_V28A
            ),
            "per_gpu_peak_vram_ratio_familywise_ucb_max": (
                PER_GPU_VRAM_UCB_NONREGRESSION_V28A
            ),
            "per_gpu_observed_median_peak_vram_ratio_max": (
                PER_GPU_POINT_VRAM_RATIO_MAX_V28A
            ),
            "all_eight_cell_peak_nvml_fraction_max": MAX_PEAK_NVML_FRACTION_V28A,
            "default_first_and_tuned_first_strata_reported_separately": True,
            "wave_summaries_reported_to_expose_load_order_effects": True,
            "threshold_justification": {
                "throughput": (
                    "each GPU may lose at most 2% at the familywise lower bound, no GPU "
                    "may have a slower point estimate, and the global task median must improve "
                    "at least 1% with a nonnegative one-sided bootstrap lower bound"
                ),
                "vram": (
                    "each GPU allows at most 2% familywise sampling noise and 1% observed "
                    "peak increase while every cell remains below 95% of physical capacity"
                ),
                "selected_before_outputs": True,
            },
            "timings_memory_samples_and_bootstrap_replicates_memory_only": True,
        },
        "gate": {
            "all_exact_output_equivalence_checks_required": True,
            "all_four_per_gpu_throughput_nonregression_gates_required": True,
            "global_task_throughput_improvement_gate_required": True,
            "all_four_per_gpu_vram_nonregression_gates_required": True,
            "all_eight_absolute_peak_vram_gates_required": True,
            "pass_authority": (
                "authorize_only_a_separately_frozen_train_only_training_recipe_A_B"
            ),
            "failure_decision": "retain_empty_default_config_training_recipe",
            "direct_recipe_adoption_authorized": False,
            "model_update_authorized": False,
            "checkpoint_write_authorized": False,
            "evaluation_authorized": False,
            "dataset_promotion_authorized": False,
        },
        "persistence_contract": {
            "aggregate_metrics_hashes_and_integrity_booleans_only": True,
            "raw_rows_prompts_answers_token_ids_text_logprobs_timings_memory_samples_or_bootstrap_replicates": False,
        },
        "contains_dataset_rows_questions_answers_or_document_content": False,
        "contains_validation_heldout_ood_or_benchmark_content": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def validate_preregistration_v28a(value):
    runtime = value.get("runtime_contract", {}) if isinstance(value, dict) else {}
    if (
        not isinstance(value, dict)
        or value.get("schema") != SCHEMA_V28A
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(value))
        or value.get("strict_train_only") is not True
        or value.get("positive_evidence_commit") != POSITIVE_EVIDENCE_COMMIT_V28A
        or value.get("contains_dataset_rows_questions_answers_or_document_content")
        is not False
        or value.get("contains_validation_heldout_ood_or_benchmark_content")
        is not False
        or tuple(value.get("waves", {})) != WAVE_ORDER_V28A
        or tuple(value.get("pairing", {}).get("physical_gpu_pairs", {}))
        != PAIR_ORDER_V28A
        or runtime.get("engines_per_wave") != 4
        or runtime.get("tensor_parallel_size_per_engine") != 1
        or runtime.get("total_generation_requests_all_engines_all_waves")
        != 29_120
        or runtime.get(
            "prelaunch_idle_check_after_all_cpu_disk_audits_immediately_before_claim"
        ) is not True
        or runtime.get("interwave_cleanup_poll_timeout_seconds") != 30.0
        or runtime.get("interwave_cleanup_poll_interval_seconds") != 0.5
        or runtime.get("interwave_poll_preserves_exact_physical_gpu_identity")
        is not True
        or runtime.get("final_cleanup_poll_timeout_seconds") != 30.0
        or runtime.get("final_cleanup_poll_interval_seconds") != 0.5
        or runtime.get("final_poll_preserves_exact_physical_gpu_identity")
        is not True
        or runtime.get(
            "successful_report_requires_all_four_gpus_idle_after_wave_b_cleanup"
        ) is not True
        or value.get("gate", {}).get("direct_recipe_adoption_authorized") is not False
        or value.get("gate", {}).get("model_update_authorized") is not False
        or value.get("gate", {}).get("checkpoint_write_authorized") is not False
        or value.get("gate", {}).get("evaluation_authorized") is not False
        or value.get("gate", {}).get("dataset_promotion_authorized") is not False
    ):
        raise RuntimeError("V28A preregistration contract changed")
    for wave in WAVE_ORDER_V28A:
        assignments = value["waves"][wave]["gpu_assignments"]
        if tuple(assignments) != tuple(str(item) for item in PHYSICAL_GPU_IDS_V28A):
            raise RuntimeError("V28A physical-GPU assignment changed")
        if tuple(
            assignments[str(gpu_id)]["arm"] for gpu_id in PHYSICAL_GPU_IDS_V28A
        ) != WAVE_ARM_ORDER_V28A[wave]:
            raise RuntimeError("V28A counterbalanced arm order changed")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH_V28A)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    value = validate_preregistration_v28a(build_preregistration_v28a())
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
            raise RuntimeError("persisted V28A preregistration differs")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return value


if __name__ == "__main__":
    main()
