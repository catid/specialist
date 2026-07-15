#!/usr/bin/env python3
"""Preregister the V29D FP8 selected-table synthetic-kernel evaluation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import build_vllm_moe_fp8_tuning_preregistration_v29b as v29b


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH_V29D = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V29D_FP8_SELECTED_TABLE_EVALUATION_PREREGISTRATION.json"
)
EVIDENCE_PATH_V29D = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V29C_FP8_MOE_TUNING_SELECTION_POSITIVE_EVIDENCE.json"
)
TABLE_PATH_V29D = ROOT / (
    "experiments/vllm_moe_tuning/"
    "v025_rtx_pro_6000_fp8_w8a8_block128_tp1_exhaustive_v29b/"
    "E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_"
    "Max-Q_Workstation_Edition,dtype=fp8_w8a8,block_shape=[128,128].json"
)
RUN_DIRECTORY_V29D = ROOT / "experiments/eggroll_es_hpo/runs"
EXPERIMENT_NAME_V29D = (
    "s6_v29d_fp8_selected_table_paired_synthetic_kernel_evaluation"
)
ATTEMPT_PATH_V29D = (
    RUN_DIRECTORY_V29D / f".{EXPERIMENT_NAME_V29D}.launch_attempt.json"
)
REPORT_PATH_V29D = RUN_DIRECTORY_V29D / EXPERIMENT_NAME_V29D / (
    "fp8_selected_table_evaluation_report_v29d.json"
)
EMPTY_DEFAULT_DIRECTORY_V29D = (
    RUN_DIRECTORY_V29D / f".{EXPERIMENT_NAME_V29D}.empty_default_config"
)

EVIDENCE_COMMIT_V29D = "a203f4821c4a737310df75543353d21ce6cea978"
EVIDENCE_FILE_SHA256_V29D = (
    "47d1b09fb188dd1f8ff16314f1c20fe614f02b1cff067a1615a0d6f0f5ce2a7b"
)
EVIDENCE_CONTENT_SHA256_V29D = (
    "dc4d3b6d2b090e4e740f63de573875f331a456d6951b62cf49a003b1114ee02e"
)
TABLE_FILE_SHA256_V29D = (
    "1a4ed0f44c6d7cc788baecd073107b4634db4c769f0820d10174f61117b25618"
)
TABLE_CONTENT_SHA256_V29D = (
    "d4a49735ccfd094d6e5a3ee763eca99ed355a51fd11a7c835bfecf9fafeaa50d"
)
EXPECTED_CONFIGS_V29D = {
    "256": {
        "BLOCK_SIZE_K": 128, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128,
        "GROUP_SIZE_M": 64, "num_stages": 3, "num_warps": 4,
    },
    "512": {
        "BLOCK_SIZE_K": 128, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128,
        "GROUP_SIZE_M": 64, "num_stages": 2, "num_warps": 4,
    },
    "1024": {
        "BLOCK_SIZE_K": 128, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256,
        "GROUP_SIZE_M": 16, "num_stages": 3, "num_warps": 8,
    },
    "2048": {
        "BLOCK_SIZE_K": 128, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256,
        "GROUP_SIZE_M": 64, "num_stages": 3, "num_warps": 8,
    },
}
PHYSICAL_GPU_IDS_V29D = (0, 1, 2, 3)
BATCH_SIZES_V29D = (256, 512, 1024, 2048)
BATCH_BY_GPU_V29D = dict(zip(PHYSICAL_GPU_IDS_V29D, BATCH_SIZES_V29D))
REPETITIONS_V29D = 8
SEEDS_V29D = tuple(2_026_100_500 + item for item in range(REPETITIONS_V29D))
BOOTSTRAP_SEED_V29D = 20_261_005
BOOTSTRAP_RESAMPLES_V29D = 50_000
FAMILYWISE_ALPHA_V29D = 0.05
ENDPOINT_COUNT_V29D = 10
PER_ENDPOINT_ALPHA_V29D = FAMILYWISE_ALPHA_V29D / ENDPOINT_COUNT_V29D
OFFICIAL_NUM_ITERS_V29D = 100


canonical_sha256 = v29b.canonical_sha256
file_sha256 = v29b.file_sha256


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


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _commit_bytes(path):
    relative = Path(path).resolve().relative_to(ROOT).as_posix()
    return subprocess.check_output(
        ["git", "show", f"{EVIDENCE_COMMIT_V29D}:{relative}"], cwd=ROOT,
    )


def _load_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"V29D JSON object required: {path}")
    return value


def validate_static_inputs_v29d():
    expected = {
        EVIDENCE_PATH_V29D: EVIDENCE_FILE_SHA256_V29D,
        TABLE_PATH_V29D: TABLE_FILE_SHA256_V29D,
        v29b.OFFICIAL_TUNER_PATH_V29B: v29b.OFFICIAL_TUNER_SHA256_V29B,
        v29b.FUSED_MOE_PATH_V29B: v29b.FUSED_MOE_SHA256_V29B,
        v29b.FUSED_MOE_CONFIG_PATH_V29B: v29b.FUSED_MOE_CONFIG_SHA256_V29B,
        v29b.VLLM_ENVS_PATH_V29B: v29b.VLLM_ENVS_SHA256_V29B,
        v29b.RAY_ACTOR_PATH_V29B: v29b.RAY_ACTOR_SHA256_V29B,
        v29b.MODEL_PATH_V29B / "config.json": v29b.MODEL_CONFIG_SHA256_V29B,
        v29b.MODEL_PATH_V29B / "model.safetensors.index.json": (
            v29b.MODEL_INDEX_SHA256_V29B
        ),
    }
    for path, digest in expected.items():
        _require(
            Path(path).is_file() and not Path(path).is_symlink()
            and file_sha256(path) == digest,
            f"V29D static input identity changed: {path}",
        )
    for path, digest in (
        (EVIDENCE_PATH_V29D, EVIDENCE_FILE_SHA256_V29D),
        (TABLE_PATH_V29D, TABLE_FILE_SHA256_V29D),
    ):
        _require(
            hashlib.sha256(_commit_bytes(path)).hexdigest() == digest,
            f"V29D selection evidence commit binding changed: {path.name}",
        )
    evidence = _load_json(EVIDENCE_PATH_V29D)
    table = _load_json(TABLE_PATH_V29D)
    _require(
        evidence.get("schema")
        == "vllm-moe-fp8-tuning-selection-positive-evidence-v29c"
        and evidence.get("content_sha256_before_self_field")
        == EVIDENCE_CONTENT_SHA256_V29D
        and canonical_sha256(_without_self(evidence))
        == EVIDENCE_CONTENT_SHA256_V29D
        and evidence.get("decision", {}).get(
            "authorize_only_separate_fp8_table_evaluation_preregistration"
        ) is True
        and evidence.get("decision", {}).get(
            "selected_table_direct_adoption_authorized"
        ) is False,
        "V29D V29C selection evidence semantics changed",
    )
    _require(
        table.get("triton_version") == "3.6.0"
        and {key: value for key, value in table.items() if key != "triton_version"}
        == EXPECTED_CONFIGS_V29D
        and canonical_sha256(table) == TABLE_CONTENT_SHA256_V29D,
        "V29D exact selected table semantics changed",
    )
    config = _load_json(v29b.MODEL_PATH_V29B / "config.json")
    text = config.get("text_config", {})
    quant = config.get("quantization_config", {})
    _require(
        text.get("num_experts") == 256
        and text.get("num_experts_per_tok") == 8
        and text.get("hidden_size") == 2048
        and text.get("moe_intermediate_size") == 512
        and text.get("dtype") == "bfloat16"
        and quant.get("quant_method") == "fp8"
        and quant.get("weight_block_size") == [128, 128],
        "V29D exact Qwen FP8 geometry changed",
    )
    return evidence, table


def schedule_v29d():
    return [
        {
            "repetition": repetition,
            "seed": SEEDS_V29D[repetition],
            "arm_order": (
                ["default", "tuned"] if repetition % 2 == 0
                else ["tuned", "default"]
            ),
        }
        for repetition in range(REPETITIONS_V29D)
    ]


def build_preregistration_v29d():
    validate_static_inputs_v29d()
    endpoints = [
        *[f"gpu{gpu}_latency_speedup" for gpu in PHYSICAL_GPU_IDS_V29D],
        *[f"gpu{gpu}_peak_vram_ratio" for gpu in PHYSICAL_GPU_IDS_V29D],
        "global_latency_geometric_mean_speedup",
        "global_peak_vram_max_ratio",
    ]
    value = _seal({
        "schema": "vllm-moe-fp8-selected-table-evaluation-preregistration-v29d",
        "status": "preregistered_before_measurement_evaluation_not_launched",
        "strict_synthetic_kernel_only": True,
        "selection_evidence": {
            "commit": EVIDENCE_COMMIT_V29D,
            "path": str(EVIDENCE_PATH_V29D),
            "file_sha256": EVIDENCE_FILE_SHA256_V29D,
            "content_sha256": EVIDENCE_CONTENT_SHA256_V29D,
            "authorizes_only_this_separate_evaluation_preregistration": True,
        },
        "selected_table": {
            "path": str(TABLE_PATH_V29D),
            "file_sha256": TABLE_FILE_SHA256_V29D,
            "content_sha256": TABLE_CONTENT_SHA256_V29D,
            "triton_version": "3.6.0",
            "exact_configs": EXPECTED_CONFIGS_V29D,
        },
        "software_identity": {
            "versions": {
                "vllm": "0.25.0", "torch": "2.11.0+cu130",
                "triton": "3.6.0", "ray": "2.56.0",
            },
            "official_tuner_path": str(v29b.OFFICIAL_TUNER_PATH_V29B),
            "official_tuner_sha256": v29b.OFFICIAL_TUNER_SHA256_V29B,
            "installed_fused_moe_sha256": v29b.FUSED_MOE_SHA256_V29B,
            "installed_fused_moe_config_sha256": (
                v29b.FUSED_MOE_CONFIG_SHA256_V29B
            ),
            "installed_vllm_envs_sha256": v29b.VLLM_ENVS_SHA256_V29B,
            "pinned_ray_actor_sha256": v29b.RAY_ACTOR_SHA256_V29B,
        },
        "model_identity": {
            "path": str(v29b.MODEL_PATH_V29B),
            "config_sha256": v29b.MODEL_CONFIG_SHA256_V29B,
            "index_sha256": v29b.MODEL_INDEX_SHA256_V29B,
            "all_files_fingerprint_sha256": (
                v29b.MODEL_ALL_FILES_FINGERPRINT_SHA256_V29B
            ),
            "weight_shards": v29b.MODEL_WEIGHT_SHARDS_V29B,
        },
        "kernel_contract": {
            "dtype": "fp8_w8a8",
            "activation_dtype": "bfloat16",
            "block_shape": [128, 128],
            "experts": 256,
            "official_shard_intermediate_size": 1024,
            "hidden_size": 2048,
            "topk": 8,
            "use_deep_gemm": False,
            "official_num_iters": OFFICIAL_NUM_ITERS_V29D,
            "batches": list(BATCH_SIZES_V29D),
            "batch_by_exact_physical_gpu": {
                str(gpu): BATCH_BY_GPU_V29D[gpu]
                for gpu in PHYSICAL_GPU_IDS_V29D
            },
            "tuned_arm_uses_exact_selected_config": True,
            "default_arm_uses_verified_empty_config_folder_then_official_default": True,
            "same_seed_and_official_tensor_constructor_for_both_paired_arms": True,
            "output_equivalence": "exact_dtype_shape_and_byte_sha256",
            "peak_vram_metric": (
                "max(tuned/default torch.cuda.max_memory_allocated ratio, "
                "tuned/default torch.cuda.max_memory_reserved ratio)"
            ),
        },
        "schedule": {
            "repetitions": REPETITIONS_V29D,
            "fixed_seeds": list(SEEDS_V29D),
            "paired_counterbalanced_schedule": schedule_v29d(),
            "default_first_count": REPETITIONS_V29D // 2,
            "tuned_first_count": REPETITIONS_V29D // 2,
            "fresh_four_worker_ray_wave_per_arm": True,
            "one_exact_batch_per_exact_physical_gpu": True,
        },
        "statistical_contract": {
            "paired_unit": "repetition_with_same_seed_and_gpu",
            "bootstrap_seed": BOOTSTRAP_SEED_V29D,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES_V29D,
            "bootstrap_statistic": "median_with_shared_paired_indices",
            "familywise_alpha": FAMILYWISE_ALPHA_V29D,
            "multiplicity": "Bonferroni across all 10 one-sided endpoints",
            "per_endpoint_one_sided_alpha": PER_ENDPOINT_ALPHA_V29D,
            "endpoints": endpoints,
            "latency_speedup_definition": "default_microseconds/tuned_microseconds",
            "peak_vram_ratio_definition": (
                "max(tuned_peak_allocated/default_peak_allocated,"
                "tuned_peak_reserved/default_peak_reserved)"
            ),
            "global_latency_statistic": (
                "geometric mean of four per-GPU median latency speedups"
            ),
            "global_peak_vram_statistic": (
                "maximum of four per-GPU median peak-VRAM ratios"
            ),
            "latency_noninferiority_margin": 0.0,
            "peak_vram_regression_margin": 0.0,
            "all_four_per_gpu_latency_familywise_lower_bounds_must_be_at_least": 1.0,
            "global_latency_familywise_lower_bound_must_be_at_least": 1.0,
            "all_four_per_gpu_peak_vram_familywise_upper_bounds_must_be_at_most": 1.0,
            "global_peak_vram_familywise_upper_bound_must_be_at_most": 1.0,
            "all_exact_outputs_must_match": True,
            "all_gates_are_conjunctive": True,
        },
        "hardware_contract": {
            "physical_gpu_ids": list(PHYSICAL_GPU_IDS_V29D),
            "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
            "driver_version": "610.43.02",
            "identities": {
                str(key): value for key, value in v29b.GPU_IDENTITIES_V29B.items()
            },
            "one_distinct_worker_pid_per_exact_uuid": True,
            "simultaneous_all_four_assigned_pids_and_positive_utilization_required_per_arm": True,
            "all_four_idle_before_claim_between_arms_and_after_final_cleanup": True,
        },
        "persistence_contract": {
            "fresh_exclusive_attempt_path": str(ATTEMPT_PATH_V29D),
            "fresh_exclusive_report_path": str(REPORT_PATH_V29D),
            "fresh_exclusive_empty_default_directory": str(
                EMPTY_DEFAULT_DIRECTORY_V29D
            ),
            "raw_input_output_tensors_timing_vectors_or_pids_persisted": False,
            "compact_aggregate_hashes_counts_bounds_and_decision_only": True,
        },
        "authority": {
            "pass_authorizes_only_separate_runtime_or_training_ab_preregistration": True,
            "direct_table_adoption_authorized": False,
            "model_update_training_checkpoint_write_dataset_promotion_authorized": False,
            "dataset_evaluation_validation_heldout_ood_or_benchmark_access_authorized": False,
        },
        "contains_dataset_rows_questions_answers_or_document_content": False,
        "contains_validation_heldout_ood_or_benchmark_content": False,
    })
    validate_preregistration_v29d(value)
    return value


def validate_preregistration_v29d(value):
    stats = value.get("statistical_contract", {})
    schedule = value.get("schedule", {})
    authority = value.get("authority", {})
    orders = schedule.get("paired_counterbalanced_schedule", [])
    _require(
        value.get("strict_synthetic_kernel_only") is True
        and schedule.get("repetitions") == 8
        and len(orders) == 8
        and sum(item.get("arm_order") == ["default", "tuned"] for item in orders) == 4
        and sum(item.get("arm_order") == ["tuned", "default"] for item in orders) == 4
        and len(stats.get("endpoints", [])) == 10
        and stats.get("bootstrap_resamples") == 50_000
        and stats.get("all_gates_are_conjunctive") is True
        and authority.get("direct_table_adoption_authorized") is False
        and authority.get(
            "dataset_evaluation_validation_heldout_ood_or_benchmark_access_authorized"
        ) is False,
        "V29D preregistration schedule statistics or authority changed",
    )
    _require(
        value.get("content_sha256_before_self_field")
        == canonical_sha256(_without_self(value)),
        "V29D preregistration self hash changed",
    )
    return value


def _exclusive_write(path, value):
    path = Path(path).resolve()
    if path != OUTPUT_PATH_V29D.resolve() or path.exists():
        raise RuntimeError("V29D preregistration output must be fresh and exact")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH_V29D)
    args = parser.parse_args(argv)
    value = build_preregistration_v29d()
    if not args.dry_run:
        _exclusive_write(args.output, value)
    print(json.dumps({
        "schema": "vllm-moe-fp8-selected-table-evaluation-preregistration-build-v29d",
        "content_sha256": value["content_sha256_before_self_field"],
        "repetitions": REPETITIONS_V29D,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES_V29D,
        "gpu_launched": False,
    }, sort_keys=True))
    return value


if __name__ == "__main__":
    main()
