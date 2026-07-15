#!/usr/bin/env python3
"""Preregister the exact V29E observability retry of V29D."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import build_vllm_moe_fp8_selected_table_evaluation_preregistration_v29d as v29d
import build_vllm_moe_fp8_selected_table_v29d_failure_evidence_v29e as failure


ROOT = Path(__file__).resolve().parent
EVIDENCE_PATH_V29E = v29d.EVIDENCE_PATH_V29D
EVIDENCE_COMMIT_V29E = v29d.EVIDENCE_COMMIT_V29D
EVIDENCE_FILE_SHA256_V29E = v29d.EVIDENCE_FILE_SHA256_V29D
EVIDENCE_CONTENT_SHA256_V29E = v29d.EVIDENCE_CONTENT_SHA256_V29D
TABLE_PATH_V29E = v29d.TABLE_PATH_V29D
TABLE_FILE_SHA256_V29E = v29d.TABLE_FILE_SHA256_V29D
TABLE_CONTENT_SHA256_V29E = v29d.TABLE_CONTENT_SHA256_V29D
PHYSICAL_GPU_IDS_V29E = v29d.PHYSICAL_GPU_IDS_V29D
BATCH_BY_GPU_V29E = v29d.BATCH_BY_GPU_V29D
SEEDS_V29E = v29d.SEEDS_V29D
REPETITIONS_V29E = v29d.REPETITIONS_V29D
BOOTSTRAP_SEED_V29E = v29d.BOOTSTRAP_SEED_V29D
BOOTSTRAP_RESAMPLES_V29E = v29d.BOOTSTRAP_RESAMPLES_V29D
FAMILYWISE_ALPHA_V29E = v29d.FAMILYWISE_ALPHA_V29D
ENDPOINT_COUNT_V29E = v29d.ENDPOINT_COUNT_V29D
PER_ENDPOINT_ALPHA_V29E = v29d.PER_ENDPOINT_ALPHA_V29D
OUTPUT_PATH_V29E = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V29E_FP8_SELECTED_TABLE_EVALUATION_RETRY_PREREGISTRATION.json"
)
FAILURE_EVIDENCE_PATH_V29E = failure.OUTPUT_PATH_V29E
FAILURE_EVIDENCE_COMMIT_V29E = "756b026d6694ff5a379e3212e7a592ccf5c9981b"
FAILURE_EVIDENCE_FILE_SHA256_V29E = (
    "9320d617a93527f6005a91156cbef58c174619df496f44a1dc552c8673ac34e0"
)
FAILURE_EVIDENCE_CONTENT_SHA256_V29E = (
    "b9f9413e1dc4f174999c27b36287002f6de128fef0f378f805ae0392ab6127d4"
)
EXPERIMENT_NAME_V29E = (
    "s6_v29e_fp8_selected_table_paired_synthetic_kernel_evaluation_retry"
)
RUN_DIRECTORY_V29E = ROOT / "experiments/eggroll_es_hpo/runs"
ATTEMPT_PATH_V29E = RUN_DIRECTORY_V29E / f".{EXPERIMENT_NAME_V29E}.launch_attempt.json"
REPORT_PATH_V29E = RUN_DIRECTORY_V29E / EXPERIMENT_NAME_V29E / (
    "fp8_selected_table_evaluation_report_v29e.json"
)
EMPTY_DEFAULT_DIRECTORY_V29E = (
    RUN_DIRECTORY_V29E / f".{EXPERIMENT_NAME_V29E}.empty_default_config"
)
OFFICIAL_NUM_ITERS_V29E = 1000
NVML_POLL_INTERVAL_SECONDS_V29E = 0.05
ACTIVITY_WITNESS_TARGET_OFFSET_SECONDS_V29E = 5.0
ACTIVITY_WITNESS_DURATION_SECONDS_V29E = 0.75
ACTIVITY_WITNESS_RECIPE_V29E = {
    "schema": "vllm-moe-fp8-common-start-cuda-activity-witness-v29e",
    "common_controller_target_offset_seconds": (
        ACTIVITY_WITNESS_TARGET_OFFSET_SECONDS_V29E
    ),
    "minimum_cuda_activity_seconds": ACTIVITY_WITNESS_DURATION_SECONDS_V29E,
    "tensor_shape": [4096, 4096],
    "tensor_dtype": "torch.bfloat16",
    "tensor_initialization": "torch.ones_without_rng",
    "operation": "torch.mm(left,right,out=output)",
    "warmup_iterations_before_common_target": 1,
    "synchronize_after_each_measured_iteration": True,
    "synchronize_after_warmup_and_after_witness": True,
    "delete_all_witness_tensors_then_gc_empty_cache_and_synchronize": True,
    "reset_peak_memory_stats_immediately_before_official_benchmark": True,
    "witness_excluded_from_latency_and_peak_vram_measurement": True,
}
ACTIVITY_WITNESS_RECIPE_SHA256_V29E = v29d.canonical_sha256(
    ACTIVITY_WITNESS_RECIPE_V29E
)
MAX_BATCH_MEMORY_AUDIT_V29E = {
    "batch_size": 2048,
    "official_num_iters": OFFICIAL_NUM_ITERS_V29E,
    "x_bytes": 8_388_608,
    "w1_fp8_bytes": 536_870_912,
    "w2_fp8_bytes": 268_435_456,
    "gating_output_bytes": 2_097_152_000,
    "input_gating_bytes": 2_097_152,
    "fp8_scale_and_scalar_bytes": 196_616,
    "ten_graph_outputs_upper_bound_bytes": 83_886_080,
    "audited_persistent_tensor_upper_bound_bytes": 2_997_026_824,
    "audited_persistent_tensor_upper_bound_gib": 2.7911987379193306,
    "required_upper_bound_gib": 5.0,
    "passes_required_upper_bound": True,
    "excludes_transient_kernel_workspace_measured_by_runtime_peak_gate": True,
}


canonical_sha256 = v29d.canonical_sha256
file_sha256 = v29d.file_sha256
_without_self = v29d._without_self
_seal = v29d._seal
_require = v29d._require


def validate_static_inputs_v29e():
    v29d.validate_static_inputs_v29d()
    failure.validate_v29d_failure_v29e()
    _require(
        FAILURE_EVIDENCE_PATH_V29E.is_file()
        and not FAILURE_EVIDENCE_PATH_V29E.is_symlink()
        and file_sha256(FAILURE_EVIDENCE_PATH_V29E)
        == FAILURE_EVIDENCE_FILE_SHA256_V29E,
        "V29E failure evidence file identity changed",
    )
    raw = subprocess.check_output(
        [
            "git", "show",
            f"{FAILURE_EVIDENCE_COMMIT_V29E}:"
            f"{FAILURE_EVIDENCE_PATH_V29E.relative_to(ROOT).as_posix()}",
        ],
        cwd=ROOT,
    )
    _require(
        hashlib.sha256(raw).hexdigest() == FAILURE_EVIDENCE_FILE_SHA256_V29E,
        "V29E committed failure evidence identity changed",
    )
    evidence = json.loads(FAILURE_EVIDENCE_PATH_V29E.read_text(encoding="utf-8"))
    _require(
        evidence.get("content_sha256_before_self_field")
        == FAILURE_EVIDENCE_CONTENT_SHA256_V29E
        and canonical_sha256(_without_self(evidence))
        == FAILURE_EVIDENCE_CONTENT_SHA256_V29E
        and evidence.get("decision", {}).get(
            "authorize_only_exact_observability_retry_preregistration"
        ) is True
        and evidence.get("decision", {}).get(
            "selected_table_adoption_authorized"
        ) is False,
        "V29E failure evidence semantics changed",
    )
    _require(
        MAX_BATCH_MEMORY_AUDIT_V29E["audited_persistent_tensor_upper_bound_bytes"]
        == sum((
            8_388_608, 536_870_912, 268_435_456, 2_097_152_000,
            2_097_152, 196_616, 83_886_080,
        ))
        and MAX_BATCH_MEMORY_AUDIT_V29E["audited_persistent_tensor_upper_bound_gib"]
        == MAX_BATCH_MEMORY_AUDIT_V29E[
            "audited_persistent_tensor_upper_bound_bytes"
        ] / (2**30)
        and MAX_BATCH_MEMORY_AUDIT_V29E["passes_required_upper_bound"] is True,
        "V29E 1000-iteration memory audit changed",
    )
    return evidence


def build_preregistration_v29e():
    validate_static_inputs_v29e()
    base = v29d.build_preregistration_v29d()
    value = copy.deepcopy(v29d._without_self(base))
    value.update({
        "schema": "vllm-moe-fp8-selected-table-evaluation-retry-preregistration-v29e",
        "status": "preregistered_before_exact_observability_retry_not_launched",
        "retry_of": {
            "v29d_preregistration_commit": failure.PREREG_COMMIT_V29E,
            "v29d_preregistration_file_sha256": failure.PREREG_FILE_SHA256_V29E,
            "v29d_preregistration_content_sha256": failure.PREREG_CONTENT_SHA256_V29E,
            "v29d_failed_attempt_file_sha256": failure.ATTEMPT_FILE_SHA256_V29E,
            "v29d_failed_attempt_content_sha256": failure.ATTEMPT_CONTENT_SHA256_V29E,
            "v29d_failure_evidence_path": str(FAILURE_EVIDENCE_PATH_V29E),
            "v29d_failure_evidence_commit": FAILURE_EVIDENCE_COMMIT_V29E,
            "v29d_failure_evidence_file_sha256": FAILURE_EVIDENCE_FILE_SHA256_V29E,
            "v29d_failure_evidence_content_sha256": FAILURE_EVIDENCE_CONTENT_SHA256_V29E,
            "v29d_failure_was_activity_observability_only": True,
            "v29d_report_absent": True,
            "v29d_final_idle_certificate_present": True,
        },
        "sole_infrastructure_correction": {
            "official_num_iters_before": v29d.OFFICIAL_NUM_ITERS_V29D,
            "official_num_iters_after": OFFICIAL_NUM_ITERS_V29E,
            "nvml_poll_interval_seconds_before": 0.25,
            "nvml_poll_interval_seconds_after": NVML_POLL_INTERVAL_SECONDS_V29E,
            "common_start_activity_witness_recipe": copy.deepcopy(
                ACTIVITY_WITNESS_RECIPE_V29E
            ),
            "common_start_activity_witness_recipe_sha256": (
                ACTIVITY_WITNESS_RECIPE_SHA256_V29E
            ),
            "max_batch_memory_audit": copy.deepcopy(MAX_BATCH_MEMORY_AUDIT_V29E),
            "selected_table_seeds_repetitions_arm_order_statistics_output_equivalence_latency_vram_and_authority_changed": False,
        },
    })
    value["kernel_contract"]["official_num_iters"] = OFFICIAL_NUM_ITERS_V29E
    value["schedule"].update({
        "common_start_activity_witness_before_every_official_measurement": True,
        "activity_witness_recipe_sha256": ACTIVITY_WITNESS_RECIPE_SHA256_V29E,
        "nvml_poll_interval_seconds": NVML_POLL_INTERVAL_SECONDS_V29E,
    })
    value["hardware_contract"].update({
        "activity_witness_exact_actor_pid_and_uuid_required": True,
        "minimum_simultaneous_all_four_positive_observations_per_arm": 1,
        "activity_witness_excluded_from_timing_and_peak_memory": True,
    })
    value["persistence_contract"] = {
        **value["persistence_contract"],
        "fresh_exclusive_attempt_path": str(ATTEMPT_PATH_V29E),
        "fresh_exclusive_report_path": str(REPORT_PATH_V29E),
        "fresh_exclusive_empty_default_directory": str(
            EMPTY_DEFAULT_DIRECTORY_V29E
        ),
    }
    value = _seal(value)
    validate_preregistration_v29e(value, base)
    return value


def validate_preregistration_v29e(value, base=None):
    if base is None:
        base = v29d.build_preregistration_v29d()
    expected_retry = {
        "v29d_preregistration_commit": failure.PREREG_COMMIT_V29E,
        "v29d_preregistration_file_sha256": failure.PREREG_FILE_SHA256_V29E,
        "v29d_preregistration_content_sha256": failure.PREREG_CONTENT_SHA256_V29E,
        "v29d_failed_attempt_file_sha256": failure.ATTEMPT_FILE_SHA256_V29E,
        "v29d_failed_attempt_content_sha256": failure.ATTEMPT_CONTENT_SHA256_V29E,
        "v29d_failure_evidence_path": str(FAILURE_EVIDENCE_PATH_V29E),
        "v29d_failure_evidence_commit": FAILURE_EVIDENCE_COMMIT_V29E,
        "v29d_failure_evidence_file_sha256": FAILURE_EVIDENCE_FILE_SHA256_V29E,
        "v29d_failure_evidence_content_sha256": FAILURE_EVIDENCE_CONTENT_SHA256_V29E,
        "v29d_failure_was_activity_observability_only": True,
        "v29d_report_absent": True,
        "v29d_final_idle_certificate_present": True,
    }
    expected_correction = {
        "official_num_iters_before": v29d.OFFICIAL_NUM_ITERS_V29D,
        "official_num_iters_after": OFFICIAL_NUM_ITERS_V29E,
        "nvml_poll_interval_seconds_before": 0.25,
        "nvml_poll_interval_seconds_after": NVML_POLL_INTERVAL_SECONDS_V29E,
        "common_start_activity_witness_recipe": ACTIVITY_WITNESS_RECIPE_V29E,
        "common_start_activity_witness_recipe_sha256": (
            ACTIVITY_WITNESS_RECIPE_SHA256_V29E
        ),
        "max_batch_memory_audit": MAX_BATCH_MEMORY_AUDIT_V29E,
        "selected_table_seeds_repetitions_arm_order_statistics_output_equivalence_latency_vram_and_authority_changed": False,
    }
    expected_kernel = copy.deepcopy(base["kernel_contract"])
    expected_kernel["official_num_iters"] = OFFICIAL_NUM_ITERS_V29E
    expected_schedule = copy.deepcopy(base["schedule"])
    expected_schedule.update({
        "common_start_activity_witness_before_every_official_measurement": True,
        "activity_witness_recipe_sha256": ACTIVITY_WITNESS_RECIPE_SHA256_V29E,
        "nvml_poll_interval_seconds": NVML_POLL_INTERVAL_SECONDS_V29E,
    })
    expected_hardware = copy.deepcopy(base["hardware_contract"])
    expected_hardware.update({
        "activity_witness_exact_actor_pid_and_uuid_required": True,
        "minimum_simultaneous_all_four_positive_observations_per_arm": 1,
        "activity_witness_excluded_from_timing_and_peak_memory": True,
    })
    expected_persistence = {
        **base["persistence_contract"],
        "fresh_exclusive_attempt_path": str(ATTEMPT_PATH_V29E),
        "fresh_exclusive_report_path": str(REPORT_PATH_V29E),
        "fresh_exclusive_empty_default_directory": str(
            EMPTY_DEFAULT_DIRECTORY_V29E
        ),
    }
    _require(
        value.get("schema")
        == "vllm-moe-fp8-selected-table-evaluation-retry-preregistration-v29e"
        and value.get("status")
        == "preregistered_before_exact_observability_retry_not_launched"
        and value.get("selection_evidence") == base.get("selection_evidence")
        and value.get("selected_table") == base.get("selected_table")
        and value.get("model_identity") == base.get("model_identity")
        and value.get("software_identity") == base.get("software_identity")
        and value.get("statistical_contract") == base.get("statistical_contract")
        and value.get("authority") == base.get("authority")
        and value.get("retry_of") == expected_retry
        and value.get("sole_infrastructure_correction") == expected_correction
        and value.get("kernel_contract") == expected_kernel
        and value.get("schedule") == expected_schedule
        and value.get("hardware_contract") == expected_hardware
        and value.get("persistence_contract") == expected_persistence,
        "V29E retry changed a non-observability V29D contract",
    )
    _require(
        value.get("content_sha256_before_self_field")
        == canonical_sha256(_without_self(value)),
        "V29E preregistration self hash changed",
    )
    return value


def _exclusive_write(path, value):
    path = Path(path).resolve()
    if path != OUTPUT_PATH_V29E.resolve() or path.exists():
        raise RuntimeError("V29E preregistration output must be fresh and exact")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH_V29E)
    args = parser.parse_args(argv)
    value = build_preregistration_v29e()
    if not args.dry_run:
        _exclusive_write(args.output, value)
    print(json.dumps({
        "schema": "vllm-moe-fp8-selected-table-evaluation-retry-preregistration-build-v29e",
        "content_sha256": value["content_sha256_before_self_field"],
        "official_num_iters": OFFICIAL_NUM_ITERS_V29E,
        "activity_witness_recipe_sha256": ACTIVITY_WITNESS_RECIPE_SHA256_V29E,
        "gpu_launched": False,
    }, sort_keys=True))
    return value


if __name__ == "__main__":
    main()
