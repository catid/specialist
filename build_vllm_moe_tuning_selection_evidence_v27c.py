#!/usr/bin/env python3
"""Build compact aggregate evidence for the completed V27C selection run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TABLE_PATH = ROOT / (
    "experiments/vllm_moe_tuning/v025_rtx_pro_6000_bf16_tp1_exhaustive_v27c/"
    "E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_"
    "Max-Q_Workstation_Edition.json"
)
OUTPUT_PATH = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V27C_MOE_TUNING_SELECTION_EVIDENCE.json"
)
TABLE_FILE_SHA256 = "128806798a5bf8a961a5bd0bc8765c82e8b73a116e6c7411e7aeba5522667562"
TABLE_CONTENT_SHA256 = "a4f82f53b037f766536013bdc10c8ca1e49873603a8f44972ef8007ed406de84"
ALLOWED_CONFIG_KEYS = {
    "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
    "num_warps", "num_stages",
}


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_selected_table():
    if file_sha256(TABLE_PATH) != TABLE_FILE_SHA256:
        raise RuntimeError("V27C selected table file identity changed")
    value = json.loads(TABLE_PATH.read_text())
    if (
        canonical_sha256(value) != TABLE_CONTENT_SHA256
        or value.get("triton_version") != "3.6.0"
        or {int(key) for key in value if key != "triton_version"}
        != {256, 512, 1024, 2048}
        or any(
            set(config) != ALLOWED_CONFIG_KEYS
            or any(isinstance(item, bool) or not isinstance(item, int)
                   for item in config.values())
            for key, config in value.items() if key != "triton_version"
        )
    ):
        raise RuntimeError("V27C selected table content changed")
    return value


def build_evidence():
    table = load_selected_table()
    value = {
        "schema": "vllm-moe-tuning-selection-evidence-v27c",
        "status": "valid_completed_selection_not_evaluation",
        "selection_contract": {
            "v27a_evaluation_preregistration_commit": (
                "2572204429cf2b016d0a001086d309b582b97724"
            ),
            "v27b_retry_preregistration_commit": (
                "90ee9dd8275348e1bc5abcc5bbbf0caceec83bdd"
            ),
            "v27c_retry_preregistration_commit": (
                "1708f64fa06a40efdd97a10d60fa965bd1d33b78"
            ),
            "official_tuner_sha256": (
                "230151de56d177ac22920ff9dada010f4361933b34b54e18d5981367723a8bcf"
            ),
            "v27b_patched_tuner_sha256": (
                "2a9c31063532c0572500d9ed19db7df18d82cd8dad85cefb41eab2a6269df25e"
            ),
            "v27c_patched_tuner_sha256": (
                "ec1cfe2aea31a3aed59fc2322b8697855329f1d8cdf8c5c4d0310655e6399ec7"
            ),
            "model": "/home/catid/specialist/models/Qwen3.6-35B-A3B",
            "model_config_sha256": (
                "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
            ),
            "batch_sizes": [256, 512, 1024, 2048],
            "search_space_configurations_per_batch": 1920,
            "selection_seed": 20260715,
            "tp_size": 1,
            "dtype": "auto_bfloat16",
            "physical_gpu_ids": [0, 1, 2, 3],
        },
        "completion": {
            "exit_code": 0,
            "all_four_workers_completed_1920_of_1920": True,
            "selected_output_file_count": 1,
            "raw_selected_output_file_sha256": (
                "750fca006017b0c71bf148c5be48768d5cfee7f5ae936553d588622aabfbf5d2"
            ),
            "committed_table_is_canonical_reformat_with_exact_content_hash": True,
            "complete_log_sha256": (
                "cd79b8ee4aa53d095258c05c2293b076187fdc805aac0dab495947fbead5626c"
            ),
            "complete_log_bytes": 3_946_399,
            "pipeline_diagnostic_occurrences": 30,
            "traceback_occurrences": 0,
            "ray_task_error_occurrences": 0,
            "exact_runtime_error_filter_used": "PassManager::run failed",
            "all_other_runtime_errors_reraised": True,
        },
        "selected_table": {
            "relative_path": TABLE_PATH.relative_to(ROOT).as_posix(),
            "file_sha256": TABLE_FILE_SHA256,
            "content_sha256": TABLE_CONTENT_SHA256,
            "triton_version": table["triton_version"],
            "configs": {
                key: table[key] for key in ("256", "512", "1024", "2048")
            },
        },
        "next_gate": {
            "table_must_be_committed_before_launch": True,
            "fresh_paired_default_vs_tuned_kernel_evaluation_required": True,
            "evaluation_contract_commit": (
                "2572204429cf2b016d0a001086d309b582b97724"
            ),
            "evaluation_runtime_commit": (
                "4a7a22b922ad5ed5bbd63f64135cfe42c3f3a0fa"
            ),
            "evaluation_not_launched_by_this_artifact": True,
        },
        "authority": {
            "direct_recipe_adoption_authorized": False,
            "model_update_checkpoint_or_dataset_promotion_authorized": False,
            "validation_heldout_ood_or_benchmark_open_authorized": False,
            "selection_result_authorizes_only_preregistered_v27a_evaluation": True,
        },
        "contains_dataset_rows_questions_answers_or_document_content": False,
        "contains_validation_heldout_ood_or_benchmark_content": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


if __name__ == "__main__":
    print(json.dumps(build_evidence(), indent=2, sort_keys=True))
