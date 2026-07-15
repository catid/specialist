#!/usr/bin/env python3
"""Build compact positive evidence for the completed V29E FP8 kernel A/B."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PREREG_RELATIVE_PATH_V29F = (
    "experiments/eggroll_es_hpo/"
    "S6_V29E_FP8_SELECTED_TABLE_EVALUATION_RETRY_PREREGISTRATION.json"
)
SELECTION_RELATIVE_PATH_V29F = (
    "experiments/eggroll_es_hpo/"
    "S6_V29C_FP8_MOE_TUNING_SELECTION_POSITIVE_EVIDENCE.json"
)
FAILURE_RELATIVE_PATH_V29F = (
    "experiments/eggroll_es_hpo/"
    "S6_V29E_V29D_FP8_SELECTED_TABLE_INFRASTRUCTURE_FAILURE_EVIDENCE.json"
)
TABLE_RELATIVE_PATH_V29F = (
    "experiments/vllm_moe_tuning/"
    "v025_rtx_pro_6000_fp8_w8a8_block128_tp1_exhaustive_v29b/"
    "E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_"
    "Max-Q_Workstation_Edition,dtype=fp8_w8a8,block_shape=[128,128].json"
)
ATTEMPT_RELATIVE_PATH_V29F = (
    "experiments/eggroll_es_hpo/runs/"
    ".s6_v29e_fp8_selected_table_paired_synthetic_kernel_evaluation_retry."
    "launch_attempt.json"
)
REPORT_RELATIVE_PATH_V29F = (
    "experiments/eggroll_es_hpo/runs/"
    "s6_v29e_fp8_selected_table_paired_synthetic_kernel_evaluation_retry/"
    "fp8_selected_table_evaluation_report_v29e.json"
)
OUTPUT_RELATIVE_PATH_V29F = (
    "experiments/eggroll_es_hpo/"
    "S6_V29F_V29E_FP8_SELECTED_TABLE_EVALUATION_POSITIVE_EVIDENCE.json"
)

PREREG_PATH_V29F = ROOT / PREREG_RELATIVE_PATH_V29F
SELECTION_PATH_V29F = ROOT / SELECTION_RELATIVE_PATH_V29F
FAILURE_PATH_V29F = ROOT / FAILURE_RELATIVE_PATH_V29F
TABLE_PATH_V29F = ROOT / TABLE_RELATIVE_PATH_V29F
ATTEMPT_PATH_V29F = ROOT / ATTEMPT_RELATIVE_PATH_V29F
REPORT_PATH_V29F = ROOT / REPORT_RELATIVE_PATH_V29F
OUTPUT_PATH_V29F = ROOT / OUTPUT_RELATIVE_PATH_V29F

PREREG_COMMIT_V29F = "02ee7b7a7a1b0fd33b1e5f3db5c95ea2b32a11e6"
FAILURE_COMMIT_V29F = "756b026d6694ff5a379e3212e7a592ccf5c9981b"
SELECTION_COMMIT_V29F = "a203f4821c4a737310df75543353d21ce6cea978"
PREREG_FILE_SHA256_V29F = (
    "853a38e75bfe91baa21d0d4331dcfbd298f7828da529920dc3c244c81f908a1f"
)
PREREG_CONTENT_SHA256_V29F = (
    "5a8bb93c60631f5a1acb22d729c942a6f2630f8ad72b0698bc7c32ee5c3f089f"
)
FAILURE_FILE_SHA256_V29F = (
    "9320d617a93527f6005a91156cbef58c174619df496f44a1dc552c8673ac34e0"
)
FAILURE_CONTENT_SHA256_V29F = (
    "b9f9413e1dc4f174999c27b36287002f6de128fef0f378f805ae0392ab6127d4"
)
SELECTION_FILE_SHA256_V29F = (
    "47d1b09fb188dd1f8ff16314f1c20fe614f02b1cff067a1615a0d6f0f5ce2a7b"
)
SELECTION_CONTENT_SHA256_V29F = (
    "dc4d3b6d2b090e4e740f63de573875f331a456d6951b62cf49a003b1114ee02e"
)
TABLE_FILE_SHA256_V29F = (
    "1a4ed0f44c6d7cc788baecd073107b4634db4c769f0820d10174f61117b25618"
)
TABLE_CONTENT_SHA256_V29F = (
    "d4a49735ccfd094d6e5a3ee763eca99ed355a51fd11a7c835bfecf9fafeaa50d"
)
ATTEMPT_FILE_SHA256_V29F = (
    "a067795e34e15cdf1a7b6142d78c07c4b954f557c325a4e2d3eafd9896ec0f47"
)
ATTEMPT_CONTENT_SHA256_V29F = (
    "742a724ed8d280bfd5c6def451b266155515965410560c9e6fb0df7f9c123a1a"
)
REPORT_FILE_SHA256_V29F = (
    "ee341a270607d0ec022f21d9a0fdce0d4dd32f3eeecce6547e6e629cb975eeef"
)
REPORT_CONTENT_SHA256_V29F = (
    "b9e11f6c433d31cb32399329ba9d852a86464fdd1f889fb52ffcdf33232b4afb"
)
SUMMARY_CONTENT_SHA256_V29F = (
    "666d2213cedfdb23fe86e230a5f3a248fb25a37fa0cd1530ddd869fd581ca5fa"
)
IMPLEMENTATION_BUNDLE_SHA256_V29F = (
    "86e4aa8f93eab8b0e8c37ddc549406990469af456f6aaec010ac2b4ab84ad63e"
)
RECIPE_CONTENT_SHA256_V29F = (
    "907cd4c97bf6650e46383efb56851316d02ceddc37db2ff07234ee5114e3f7d2"
)
RUNTIME_ENVIRONMENT_SHA256_V29F = (
    "7acc87ffb6a50682f19309ea3059823d0fc96dfc459e30cb66d64e157a76f52a"
)
LIVE_CPU_DISK_AUDIT_SHA256_V29F = (
    "0bfb28bdb233c7b0c346466d8d992b8e53eaf228b7e024cd07663276abd00f87"
)
PRELAUNCH_IDLE_SHA256_V29F = (
    "68143e918231ddf78f350a4d8f47231a2c7397989aaa0fe37ed19addbc74ddeb"
)
FINAL_IDLE_SHA256_V29F = (
    "fb123fed0887770052250bc235555885d60d0dcc4bd42e2c8a36bf2593abaf4a"
)
ACTIVITY_WITNESS_SHA256_V29F = (
    "c131a7eb3402f35545e36d2e46665f360a50246ca6c205e24e06af467518f405"
)
ARM_INTEGRITY_SHA256_V29F = (
    "e41eef0317befc6d1fff5aa7f90b5dda8264b728eac1ed4aea61299c48b61c8a"
)
IDLE_COMMITMENT_SHA256_V29F = (
    "e3d008f563d3a6a371dbef3f62a979e98921e2bf62c617f8f1b2f75fbae3295c"
)
OUTPUT_COMMITMENT_SHA256_V29F = (
    "8da022a39b9ce1c46b839bd9add4b65955b5b1c4942db5bb29ba3ba3886a0a89"
)
MODEL_CONFIG_SHA256_V29F = (
    "570ef7ea45a7e1d3de2b1d3c70c4ac3562d0e768acdc195778cb4f4d95025845"
)
MODEL_INDEX_SHA256_V29F = (
    "6f176f344e41d35b17af12904e33401da5ebff3b49fccb8bfa0185bc2d50f9d6"
)

EXPECTED_CONFIGS_V29F = {
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
EXPECTED_PER_GPU_V29F = {
    "0": {
        "batch_size": 256,
        "familywise_latency_lower_bound": 1.0945357835730134,
        "familywise_peak_vram_upper_bound": 1.0,
        "latency_gate_pass": True,
        "median_latency_speedup": 1.1032971085776004,
        "median_peak_vram_ratio": 1.0,
        "peak_vram_gate_pass": True,
    },
    "1": {
        "batch_size": 512,
        "familywise_latency_lower_bound": 1.0573538139307819,
        "familywise_peak_vram_upper_bound": 1.0,
        "latency_gate_pass": True,
        "median_latency_speedup": 1.0602364900047896,
        "median_peak_vram_ratio": 1.0,
        "peak_vram_gate_pass": True,
    },
    "2": {
        "batch_size": 1024,
        "familywise_latency_lower_bound": 1.0022116378923651,
        "familywise_peak_vram_upper_bound": 1.0,
        "latency_gate_pass": True,
        "median_latency_speedup": 1.0053126775180914,
        "median_peak_vram_ratio": 1.0,
        "peak_vram_gate_pass": True,
    },
    "3": {
        "batch_size": 2048,
        "familywise_latency_lower_bound": 1.0048976158835112,
        "familywise_peak_vram_upper_bound": 1.0,
        "latency_gate_pass": True,
        "median_latency_speedup": 1.0082738818499424,
        "median_peak_vram_ratio": 1.0,
        "peak_vram_gate_pass": True,
    },
}
EXPECTED_GLOBAL_V29F = {
    "familywise_latency_lower_bound": 1.0391777692961988,
    "familywise_peak_vram_upper_bound": 1.0,
    "latency_gate_pass": True,
    "latency_geometric_mean_speedup": 1.0435030576977902,
    "peak_vram_gate_pass": True,
    "peak_vram_max_ratio": 1.0,
}
FORBIDDEN_KEYS_V29F = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "response", "responses", "token_ids", "training_rows", "evaluation_rows",
    "validation_rows", "heldout_rows", "ood_rows", "benchmark_rows",
    "timing_vectors", "raw_pids", "raw_worker_pids", "raw_tensors",
    "traceback", "progress_log", "compiler_log", "search_results",
}


def canonical_sha256(value):
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


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


def _seal(value):
    result = copy.deepcopy(value)
    result.pop("content_sha256_before_self_field", None)
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _load_json_object(path, label):
    path = Path(path)
    _require(
        path.is_file() and not path.is_symlink(),
        f"V29F {label} path changed",
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"V29F {label} must be a JSON object")
    return value


def _verify_self_hash(value, expected, label):
    _require(
        value.get("content_sha256_before_self_field") == expected
        and canonical_sha256(_without_self(value)) == expected,
        f"V29F {label} self hash changed",
    )


def _verify_committed_bytes(commit, relative_path, expected, label):
    raw = subprocess.check_output(
        ["git", "show", f"{commit}:{relative_path}"], cwd=ROOT,
    )
    _require(
        hashlib.sha256(raw).hexdigest() == expected,
        f"V29F committed {label} identity changed",
    )


def _recursive_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).lower()
            yield from _recursive_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _recursive_keys(item)


def _assert_compact_v29f(value):
    overlap = FORBIDDEN_KEYS_V29F & set(_recursive_keys(value))
    if overlap:
        raise RuntimeError(
            f"V29F positive evidence contains forbidden keys: {sorted(overlap)}"
        )


def _validate_semantics_v29f(
    preregistration, selection, failure, table, attempt, report,
):
    expected_attempt_keys = {
        "content_sha256_before_self_field",
        "dataset_evaluation_validation_heldout_ood_or_benchmark_surface_opened",
        "direct_action_taken", "final_idle_certificate_sha256",
        "implementation_bundle_sha256", "live_cpu_disk_audit_content_sha256",
        "phase", "prelaunch_idle_certificate_sha256", "preregistration",
        "recipe_content_sha256", "report_binding", "retry_of",
        "runtime_environment_certificate_sha256", "schema",
        "selection_evidence", "sole_infrastructure_correction", "status",
    }
    expected_report_keys = {
        "content_sha256_before_self_field",
        "dataset_evaluation_validation_heldout_ood_or_benchmark_surface_opened",
        "direct_table_adoption_model_update_training_checkpoint_dataset_promotion_applied",
        "final_idle_certificate_sha256", "implementation_bundle_sha256",
        "live_cpu_disk_audit_content_sha256", "prelaunch_idle_certificate_sha256",
        "preregistration", "raw_timing_memory_input_output_vectors_or_pids_persisted",
        "recipe_content_sha256", "retry_of", "runtime_environment_certificate_sha256",
        "runtime_integrity", "schema", "selected_table", "selection_evidence",
        "sole_infrastructure_correction", "status", "summary",
    }
    _require(
        set(attempt) == expected_attempt_keys and set(report) == expected_report_keys,
        "V29F compact attempt or report surface changed",
    )
    _require(
        preregistration.get("schema")
        == "vllm-moe-fp8-selected-table-evaluation-retry-preregistration-v29e"
        and preregistration.get("status")
        == "preregistered_before_exact_observability_retry_not_launched"
        and preregistration.get("strict_synthetic_kernel_only") is True
        and attempt.get("schema")
        == "vllm-moe-fp8-selected-table-evaluation-attempt-v29e"
        and attempt.get("status") == "complete"
        and attempt.get("phase") == "after_compact_report_and_final_gpu_cleanup"
        and report.get("schema")
        == "vllm-moe-fp8-selected-table-evaluation-report-v29e"
        and report.get("status") == "complete_synthetic_kernel_evaluation",
        "V29F lifecycle state changed",
    )

    prereg_binding = {
        "path": str(PREREG_PATH_V29F.resolve()),
        "file_sha256": PREREG_FILE_SHA256_V29F,
        "content_sha256": PREREG_CONTENT_SHA256_V29F,
    }
    selection_binding = {
        "commit": SELECTION_COMMIT_V29F,
        "path": str(SELECTION_PATH_V29F.resolve()),
        "file_sha256": SELECTION_FILE_SHA256_V29F,
        "content_sha256": SELECTION_CONTENT_SHA256_V29F,
        "authorizes_only_this_separate_evaluation_preregistration": True,
    }
    _require(
        attempt.get("preregistration") == prereg_binding
        and report.get("preregistration") == prereg_binding
        and attempt.get("selection_evidence") == selection_binding
        and report.get("selection_evidence") == selection_binding
        and preregistration.get("selection_evidence") == selection_binding,
        "V29F preregistration or selection binding changed",
    )
    _require(
        attempt.get("report_binding") == {
            "path": str(REPORT_PATH_V29F.resolve()),
            "file_sha256": REPORT_FILE_SHA256_V29F,
            "content_sha256": REPORT_CONTENT_SHA256_V29F,
        },
        "V29F attempt report binding changed",
    )

    shared_contracts = {
        "implementation_bundle_sha256": IMPLEMENTATION_BUNDLE_SHA256_V29F,
        "recipe_content_sha256": RECIPE_CONTENT_SHA256_V29F,
        "runtime_environment_certificate_sha256": RUNTIME_ENVIRONMENT_SHA256_V29F,
        "live_cpu_disk_audit_content_sha256": LIVE_CPU_DISK_AUDIT_SHA256_V29F,
        "prelaunch_idle_certificate_sha256": PRELAUNCH_IDLE_SHA256_V29F,
        "final_idle_certificate_sha256": FINAL_IDLE_SHA256_V29F,
    }
    _require(
        all(attempt.get(key) == value for key, value in shared_contracts.items())
        and all(report.get(key) == value for key, value in shared_contracts.items()),
        "V29F implementation recipe environment audit or idle binding changed",
    )
    _require(
        attempt.get("retry_of") == report.get("retry_of")
        == preregistration.get("retry_of")
        and attempt["retry_of"].get("v29d_failure_evidence_commit")
        == FAILURE_COMMIT_V29F
        and attempt["retry_of"].get("v29d_failure_evidence_file_sha256")
        == FAILURE_FILE_SHA256_V29F
        and attempt["retry_of"].get("v29d_failure_evidence_content_sha256")
        == FAILURE_CONTENT_SHA256_V29F
        and attempt["retry_of"].get("v29d_failure_was_activity_observability_only")
        is True
        and attempt["retry_of"].get("v29d_final_idle_certificate_present") is True
        and attempt["retry_of"].get("v29d_report_absent") is True,
        "V29F V29D retry lineage changed",
    )
    _require(
        attempt.get("sole_infrastructure_correction")
        == report.get("sole_infrastructure_correction")
        == preregistration.get("sole_infrastructure_correction")
        and attempt["sole_infrastructure_correction"].get(
            "common_start_activity_witness_recipe_sha256"
        ) == ACTIVITY_WITNESS_SHA256_V29F
        and attempt["sole_infrastructure_correction"].get(
            "selected_table_seeds_repetitions_arm_order_statistics_output_equivalence_latency_vram_and_authority_changed"
        ) is False,
        "V29F sole infrastructure correction changed",
    )

    table_with_version = copy.deepcopy(EXPECTED_CONFIGS_V29F)
    table_with_version["triton_version"] = "3.6.0"
    selected = preregistration.get("selected_table", {})
    _require(
        table == table_with_version
        and canonical_sha256(table) == TABLE_CONTENT_SHA256_V29F
        and selected.get("path") == str(TABLE_PATH_V29F.resolve())
        and selected.get("file_sha256") == TABLE_FILE_SHA256_V29F
        and selected.get("content_sha256") == TABLE_CONTENT_SHA256_V29F
        and selected.get("exact_configs") == EXPECTED_CONFIGS_V29F
        and selected.get("triton_version") == "3.6.0"
        and report.get("selected_table") == {
            "path": str(TABLE_PATH_V29F.resolve()),
            "file_sha256": TABLE_FILE_SHA256_V29F,
            "content_sha256": TABLE_CONTENT_SHA256_V29F,
        },
        "V29F selected table identity or content changed",
    )

    authority = preregistration.get("authority", {})
    kernel = preregistration.get("kernel_contract", {})
    statistics = preregistration.get("statistical_contract", {})
    schedule = preregistration.get("schedule", {})
    model = preregistration.get("model_identity", {})
    _require(
        authority == {
            "dataset_evaluation_validation_heldout_ood_or_benchmark_access_authorized": False,
            "direct_table_adoption_authorized": False,
            "model_update_training_checkpoint_write_dataset_promotion_authorized": False,
            "pass_authorizes_only_separate_runtime_or_training_ab_preregistration": True,
        }
        and kernel.get("dtype") == "fp8_w8a8"
        and kernel.get("activation_dtype") == "bfloat16"
        and kernel.get("block_shape") == [128, 128]
        and kernel.get("batch_by_exact_physical_gpu")
        == {"0": 256, "1": 512, "2": 1024, "3": 2048}
        and kernel.get("official_num_iters") == 1000
        and kernel.get("use_deep_gemm") is False
        and statistics.get("endpoints")
        == [
            "gpu0_latency_speedup", "gpu1_latency_speedup",
            "gpu2_latency_speedup", "gpu3_latency_speedup",
            "gpu0_peak_vram_ratio", "gpu1_peak_vram_ratio",
            "gpu2_peak_vram_ratio", "gpu3_peak_vram_ratio",
            "global_latency_geometric_mean_speedup", "global_peak_vram_max_ratio",
        ]
        and statistics.get("bootstrap_resamples") == 50_000
        and statistics.get("bootstrap_seed") == 20_261_005
        and statistics.get("familywise_alpha") == 0.05
        and statistics.get("per_endpoint_one_sided_alpha") == 0.005
        and schedule.get("repetitions") == 8
        and schedule.get("default_first_count") == 4
        and schedule.get("tuned_first_count") == 4
        and model.get("config_sha256") == MODEL_CONFIG_SHA256_V29F
        and model.get("index_sha256") == MODEL_INDEX_SHA256_V29F,
        "V29F frozen preregistration contract changed",
    )

    _require(
        selection.get("schema")
        == "vllm-moe-fp8-tuning-selection-positive-evidence-v29c"
        and selection.get("status") == "valid_completed_selection_not_evaluation"
        and selection.get("decision", {}).get("selection_positive_evidence_valid")
        is True
        and selection.get("decision", {}).get(
            "authorize_only_separate_fp8_table_evaluation_preregistration"
        ) is True
        and selection.get("decision", {}).get("selected_table_direct_adoption_authorized")
        is False
        and selection.get("artifacts", {}).get("original_official_selected_table", {})
        .get("file_sha256") == TABLE_FILE_SHA256_V29F,
        "V29F V29C selection evidence changed",
    )
    _require(
        failure.get("schema")
        == "vllm-moe-fp8-selected-table-infrastructure-failure-evidence-v29e"
        and failure.get("status")
        == "valid_failed_before_evaluation_report_observability_only"
        and failure.get("decision", {}).get(
            "authorize_only_exact_observability_retry_preregistration"
        ) is True
        and failure.get("decision", {}).get("selected_table_adoption_authorized")
        is False
        and failure.get("failure_boundary", {}).get(
            "activity_observability_gate_failed"
        ) is True
        and failure.get("failure_boundary", {}).get("kernel_statistical_evaluation_completed")
        is False
        and failure.get("failure_boundary", {}).get("final_idle_certificate_present")
        is True,
        "V29F V29D failure evidence changed",
    )

    integrity = report.get("runtime_integrity", {})
    _require(
        integrity == {
            "activity_witness_recipe_sha256": ACTIVITY_WITNESS_SHA256_V29F,
            "all_16_common_start_activity_witnesses_excluded_and_peak_memory_reset": True,
            "all_16_fresh_four_worker_arms_passed": True,
            "all_four_exact_gpus_finally_idle": True,
            "arm_count": 16,
            "arm_integrity_commitment_sha256": ARM_INTEGRITY_SHA256_V29F,
            "between_arm_and_final_idle_commitment_sha256": IDLE_COMMITMENT_SHA256_V29F,
            "minimum_one_simultaneous_all_four_positive_observation_per_arm": True,
            "nvml_poll_interval_seconds": 0.05,
            "official_num_iters": 1000,
            "raw_worker_pids_persisted": False,
        },
        "V29F runtime integrity changed",
    )
    summary = report.get("summary", {})
    _require(
        set(summary) == {
            "bootstrap_resamples", "bootstrap_seed",
            "content_sha256_before_self_field",
            "dataset_evaluation_validation_heldout_ood_or_benchmark_access_authorized",
            "decision",
            "direct_table_adoption_model_update_training_checkpoint_dataset_promotion_authorized",
            "endpoint_count", "exact_output_equivalence", "familywise_alpha",
            "global", "pass", "per_endpoint_one_sided_alpha", "per_gpu",
            "raw_timing_memory_input_output_vectors_or_pids_persisted",
            "repetitions", "schema",
        }
        and summary.get("schema")
        == "vllm-moe-fp8-selected-table-paired-summary-v29e"
        and summary.get("pass") is True
        and summary.get("repetitions") == 8
        and summary.get("endpoint_count") == 10
        and summary.get("bootstrap_resamples") == 50_000
        and summary.get("bootstrap_seed") == 20_261_005
        and summary.get("familywise_alpha") == 0.05
        and summary.get("per_endpoint_one_sided_alpha") == 0.005
        and summary.get("per_gpu") == EXPECTED_PER_GPU_V29F
        and summary.get("global") == EXPECTED_GLOBAL_V29F
        and summary.get("exact_output_equivalence") == {
            "matched_pairs": 32,
            "paired_output_digest_commitment_sha256": OUTPUT_COMMITMENT_SHA256_V29F,
            "pass": True,
            "required_pairs": 32,
        }
        and summary.get("decision")
        == "authorize_only_separate_runtime_or_training_ab_preregistration"
        and summary.get(
            "dataset_evaluation_validation_heldout_ood_or_benchmark_access_authorized"
        ) is False
        and summary.get(
            "direct_table_adoption_model_update_training_checkpoint_dataset_promotion_authorized"
        ) is False
        and summary.get("raw_timing_memory_input_output_vectors_or_pids_persisted")
        is False,
        "V29F aggregate result or authority changed",
    )
    _require(
        attempt.get("direct_action_taken") is False
        and attempt.get(
            "dataset_evaluation_validation_heldout_ood_or_benchmark_surface_opened"
        ) is False
        and report.get(
            "direct_table_adoption_model_update_training_checkpoint_dataset_promotion_applied"
        ) is False
        and report.get(
            "dataset_evaluation_validation_heldout_ood_or_benchmark_surface_opened"
        ) is False
        and report.get("raw_timing_memory_input_output_vectors_or_pids_persisted")
        is False
        and preregistration.get("contains_dataset_rows_questions_answers_or_document_content")
        is False
        and preregistration.get("contains_validation_heldout_ood_or_benchmark_content")
        is False,
        "V29F closed action or data surface changed",
    )
    return summary, integrity


def validate_bound_artifacts_v29f():
    expected_files = {
        PREREG_PATH_V29F: PREREG_FILE_SHA256_V29F,
        SELECTION_PATH_V29F: SELECTION_FILE_SHA256_V29F,
        FAILURE_PATH_V29F: FAILURE_FILE_SHA256_V29F,
        TABLE_PATH_V29F: TABLE_FILE_SHA256_V29F,
        ATTEMPT_PATH_V29F: ATTEMPT_FILE_SHA256_V29F,
        REPORT_PATH_V29F: REPORT_FILE_SHA256_V29F,
    }
    for path, expected in expected_files.items():
        _require(
            file_sha256(path) == expected,
            f"V29F file hash changed: {path.name}",
        )
    _verify_committed_bytes(
        PREREG_COMMIT_V29F, PREREG_RELATIVE_PATH_V29F,
        PREREG_FILE_SHA256_V29F, "preregistration",
    )
    _verify_committed_bytes(
        FAILURE_COMMIT_V29F, FAILURE_RELATIVE_PATH_V29F,
        FAILURE_FILE_SHA256_V29F, "V29D failure evidence",
    )
    _verify_committed_bytes(
        SELECTION_COMMIT_V29F, SELECTION_RELATIVE_PATH_V29F,
        SELECTION_FILE_SHA256_V29F, "V29C selection evidence",
    )
    _verify_committed_bytes(
        SELECTION_COMMIT_V29F, TABLE_RELATIVE_PATH_V29F,
        TABLE_FILE_SHA256_V29F, "selected table",
    )
    preregistration = _load_json_object(PREREG_PATH_V29F, "preregistration")
    selection = _load_json_object(SELECTION_PATH_V29F, "selection evidence")
    failure = _load_json_object(FAILURE_PATH_V29F, "failure evidence")
    table = _load_json_object(TABLE_PATH_V29F, "selected table")
    attempt = _load_json_object(ATTEMPT_PATH_V29F, "attempt")
    report = _load_json_object(REPORT_PATH_V29F, "report")
    _verify_self_hash(
        preregistration, PREREG_CONTENT_SHA256_V29F, "preregistration",
    )
    _verify_self_hash(selection, SELECTION_CONTENT_SHA256_V29F, "selection evidence")
    _verify_self_hash(failure, FAILURE_CONTENT_SHA256_V29F, "failure evidence")
    _verify_self_hash(attempt, ATTEMPT_CONTENT_SHA256_V29F, "attempt")
    _verify_self_hash(report, REPORT_CONTENT_SHA256_V29F, "report")
    _verify_self_hash(
        report.get("summary", {}), SUMMARY_CONTENT_SHA256_V29F, "summary",
    )
    _validate_semantics_v29f(
        preregistration, selection, failure, table, attempt, report,
    )
    return preregistration, selection, failure, table, attempt, report


def build_positive_evidence_v29f():
    _prereg, _selection, _failure, _table, attempt, report = (
        validate_bound_artifacts_v29f()
    )
    summary = report["summary"]
    integrity = report["runtime_integrity"]
    value = _seal({
        "schema": "vllm-moe-fp8-selected-table-evaluation-positive-evidence-v29f",
        "status": "valid_completed_synthetic_kernel_evaluation_passed",
        "contracts": {
            "v29e_preregistration_commit": PREREG_COMMIT_V29F,
            "v29e_preregistration_file_sha256": PREREG_FILE_SHA256_V29F,
            "v29e_preregistration_content_sha256": PREREG_CONTENT_SHA256_V29F,
            "v29e_implementation_bundle_sha256": IMPLEMENTATION_BUNDLE_SHA256_V29F,
            "v29e_recipe_content_sha256": RECIPE_CONTENT_SHA256_V29F,
            "runtime_environment_certificate_sha256": RUNTIME_ENVIRONMENT_SHA256_V29F,
            "live_cpu_disk_audit_content_sha256": LIVE_CPU_DISK_AUDIT_SHA256_V29F,
            "v29c_selection_commit": SELECTION_COMMIT_V29F,
            "v29d_failure_evidence_commit": FAILURE_COMMIT_V29F,
        },
        "artifacts": {
            "committed_preregistration": {
                "relative_path": PREREG_RELATIVE_PATH_V29F,
                "file_sha256": PREREG_FILE_SHA256_V29F,
                "content_sha256": PREREG_CONTENT_SHA256_V29F,
            },
            "committed_selection_evidence": {
                "relative_path": SELECTION_RELATIVE_PATH_V29F,
                "file_sha256": SELECTION_FILE_SHA256_V29F,
                "content_sha256": SELECTION_CONTENT_SHA256_V29F,
            },
            "committed_retry_failure_evidence": {
                "relative_path": FAILURE_RELATIVE_PATH_V29F,
                "file_sha256": FAILURE_FILE_SHA256_V29F,
                "content_sha256": FAILURE_CONTENT_SHA256_V29F,
            },
            "committed_selected_table": {
                "relative_path": TABLE_RELATIVE_PATH_V29F,
                "file_sha256": TABLE_FILE_SHA256_V29F,
                "content_sha256": TABLE_CONTENT_SHA256_V29F,
            },
            "ignored_live_attempt": {
                "relative_path": ATTEMPT_RELATIVE_PATH_V29F,
                "file_sha256": ATTEMPT_FILE_SHA256_V29F,
                "content_sha256": ATTEMPT_CONTENT_SHA256_V29F,
                "status": attempt["status"],
                "phase": attempt["phase"],
            },
            "ignored_live_report": {
                "relative_path": REPORT_RELATIVE_PATH_V29F,
                "file_sha256": REPORT_FILE_SHA256_V29F,
                "content_sha256": REPORT_CONTENT_SHA256_V29F,
                "summary_content_sha256": SUMMARY_CONTENT_SHA256_V29F,
            },
        },
        "aggregate_result": {
            "strict_synthetic_kernel_only": True,
            "physical_gpu_count": 4,
            "batch_by_physical_gpu": {
                gpu_id: item["batch_size"]
                for gpu_id, item in EXPECTED_PER_GPU_V29F.items()
            },
            "repetitions": summary["repetitions"],
            "counterbalanced_arm_count": integrity["arm_count"],
            "official_iterations_per_arm": integrity["official_num_iters"],
            "endpoint_count": summary["endpoint_count"],
            "familywise_alpha": summary["familywise_alpha"],
            "per_gpu": copy.deepcopy(EXPECTED_PER_GPU_V29F),
            "global": copy.deepcopy(EXPECTED_GLOBAL_V29F),
            "exact_output_equivalence": copy.deepcopy(
                summary["exact_output_equivalence"]
            ),
            "all_16_fresh_four_worker_arms_passed": integrity[
                "all_16_fresh_four_worker_arms_passed"
            ],
            "all_four_exact_gpus_finally_idle": integrity[
                "all_four_exact_gpus_finally_idle"
            ],
            "minimum_one_simultaneous_all_four_positive_observation_per_arm": integrity[
                "minimum_one_simultaneous_all_four_positive_observation_per_arm"
            ],
            "arm_integrity_commitment_sha256": ARM_INTEGRITY_SHA256_V29F,
            "between_arm_and_final_idle_commitment_sha256": IDLE_COMMITMENT_SHA256_V29F,
            "pass": summary["pass"],
        },
        "decision": {
            "synthetic_kernel_positive_evidence_valid": True,
            "authorize_only_separate_fp8_runtime_or_training_ab_preregistration": True,
            "full_model_or_bf16_training_path_integration_demonstrated": False,
            "direct_selected_table_adoption_authorized": False,
            "training_or_model_update_authorized": False,
            "checkpoint_write_authorized": False,
            "dataset_promotion_authorized": False,
            "dataset_evaluation_validation_heldout_ood_or_benchmark_open_authorized": False,
        },
        "contains_dataset_rows_questions_answers_or_document_content": False,
        "contains_validation_heldout_ood_or_benchmark_content": False,
        "raw_timing_memory_input_output_vectors_or_pids_persisted": False,
    })
    _assert_compact_v29f(value)
    return value


def _exclusive_write_json(path, value):
    path = Path(path).resolve()
    if path != OUTPUT_PATH_V29F.resolve():
        raise ValueError("V29F evidence output path changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise RuntimeError("V29F positive evidence already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH_V29F)
    args = parser.parse_args(argv)
    value = build_positive_evidence_v29f()
    if not args.dry_run:
        _exclusive_write_json(args.output, value)
    print(json.dumps({
        "schema": "vllm-moe-fp8-selected-table-evaluation-positive-evidence-build-v29f",
        "content_sha256": value["content_sha256_before_self_field"],
        "synthetic_kernel_pass": value["aggregate_result"]["pass"],
        "full_model_integration_demonstrated": False,
        "evaluation_authorized": False,
        "gpu_launched": False,
    }, sort_keys=True))
    return value


if __name__ == "__main__":
    main()
