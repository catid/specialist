#!/usr/bin/env python3
"""Build compact positive evidence for the completed V29B FP8 selection."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ATTEMPT_RELATIVE_PATH_V29C = (
    "experiments/eggroll_es_hpo/runs/"
    ".s6_v29b_fp8_w8a8_block128_moe_tuning_selection.launch_attempt.json"
)
REPORT_RELATIVE_PATH_V29C = (
    "experiments/eggroll_es_hpo/runs/"
    "s6_v29b_fp8_w8a8_block128_moe_tuning_selection/"
    "fp8_moe_tuning_selection_report_v29b.json"
)
PREREG_RELATIVE_PATH_V29C = (
    "experiments/eggroll_es_hpo/"
    "S6_V29B_FP8_MOE_TUNING_PREREGISTRATION.json"
)
LIVE_TABLE_RELATIVE_PATH_V29C = (
    "experiments/vllm_moe_tuning/"
    "v025_rtx_pro_6000_fp8_w8a8_block128_tp1_exhaustive_v29b/"
    "E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_"
    "Max-Q_Workstation_Edition,dtype=fp8_w8a8,block_shape=[128,128].json"
)
OUTPUT_RELATIVE_PATH_V29C = (
    "experiments/eggroll_es_hpo/"
    "S6_V29C_FP8_MOE_TUNING_SELECTION_POSITIVE_EVIDENCE.json"
)

ATTEMPT_PATH_V29C = ROOT / ATTEMPT_RELATIVE_PATH_V29C
REPORT_PATH_V29C = ROOT / REPORT_RELATIVE_PATH_V29C
PREREG_PATH_V29C = ROOT / PREREG_RELATIVE_PATH_V29C
LIVE_TABLE_PATH_V29C = ROOT / LIVE_TABLE_RELATIVE_PATH_V29C
OUTPUT_PATH_V29C = ROOT / OUTPUT_RELATIVE_PATH_V29C

ATTEMPT_FILE_SHA256_V29C = (
    "a7d2b85fdffa44fbb6e4f395ef1e749ea2fd6f23daa7cb4db683adbb14454456"
)
ATTEMPT_CONTENT_SHA256_V29C = (
    "0adfbf058eea46f3c55a10ac5ac61812b074fe24b6c772a382e04348e076681d"
)
REPORT_FILE_SHA256_V29C = (
    "3f24a1e5e520a4f5a199fc118d91eca6a741ff08ecf391f0c6a4b61f15118b62"
)
REPORT_CONTENT_SHA256_V29C = (
    "34fff582e872f0a7cb6a2e6ce8be645b746ec4a28530cadc3a9d1885f8d76eb3"
)
SELECTED_TABLE_FILE_SHA256_V29C = (
    "1a4ed0f44c6d7cc788baecd073107b4634db4c769f0820d10174f61117b25618"
)
SELECTED_TABLE_CONTENT_SHA256_V29C = (
    "d4a49735ccfd094d6e5a3ee763eca99ed355a51fd11a7c835bfecf9fafeaa50d"
)
PREREG_FILE_SHA256_V29C = (
    "1a9316f1d2a1a47e583b2027670626d3434ab0d0a235b26ae003058ae20b5369"
)
PREREG_CONTENT_SHA256_V29C = (
    "0dff2a5c6ac6e8312f72f0221140305aa06cf52a453d855188a329e2952ef7bf"
)
IMPLEMENTATION_BUNDLE_SHA256_V29C = (
    "fdcc961aa78da024efa4a9731c38a56b624df35786b4638621de86df97584fd5"
)
RECIPE_CONTENT_SHA256_V29C = (
    "835e4876620bca3691d172710128c378484a5d17925756a2eef7cc361a94b1c4"
)
RUNTIME_ENVIRONMENT_SHA256_V29C = (
    "7acc87ffb6a50682f19309ea3059823d0fc96dfc459e30cb66d64e157a76f52a"
)
LIVE_CPU_DISK_AUDIT_SHA256_V29C = (
    "33d634560c57e20b2d8e468f1821311d5382bb606c72c67599c194ecbee5e070"
)
PRELAUNCH_IDLE_SHA256_V29C = (
    "ab0a391b724e9578e474eb9d0c189915e8f5f0ecbb128fdb7ef74f30f409661b"
)
FINAL_IDLE_SHA256_V29C = (
    "d1e3f115d58b2ea99b82795eeaa87f6897a3e813ce1768161e06080d744eb211"
)
OFFICIAL_TUNER_SHA256_V29C = (
    "230151de56d177ac22920ff9dada010f4361933b34b54e18d5981367723a8bcf"
)
SEARCH_SPACE_SHA256_V29C = (
    "e9a5db2e566fab0e43bc47f7e92682debc830617199f15f9e0275e7eaed81c98"
)
PLACEMENT_IDENTITY_SHA256_V29C = (
    "08e5bbac970eac336cccdc88d010edf3fc73cc6cf3a86ff27064c8ad100c9776"
)
ACTOR_PID_MAP_SHA256_V29C = (
    "3d652d0c607fe03e3ad5c2944a6e4f615eee6f3ae84a7b2062e8370d63cac421"
)
EXPECTED_UUIDS_V29C = {
    "0": "GPU-4c394fc5-b18f-6622-ca94-f7fbd7112927",
    "1": "GPU-f10c2baf-536b-1d40-cd4b-25b202ae0ded",
    "2": "GPU-04cde663-7c53-2f18-3ec4-1699820e2640",
    "3": "GPU-972bf85d-1b32-2d1b-20f6-babc4c804999",
}
EXPECTED_BATCHES_V29C = (256, 512, 1024, 2048)
EXPECTED_CONFIGS_V29C = {
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
FORBIDDEN_KEYS_V29C = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "response", "responses", "training_rows", "evaluation_rows",
    "validation_rows", "heldout_rows", "ood_rows", "benchmark_rows",
    "progress_log", "compiler_log", "search_results", "timing_vectors",
    "raw_pids", "traceback", "message",
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
        f"V29C {label} path changed",
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"V29C {label} must be a JSON object")
    return value


def _verify_self_hash(value, expected, label):
    _require(
        value.get("content_sha256_before_self_field") == expected
        and canonical_sha256(_without_self(value)) == expected,
        f"V29C {label} self hash changed",
    )


def _verify_report_self_hash_v29c(report):
    normalized = copy.deepcopy(_without_self(report))
    selected = normalized.get("runtime_integrity", {}).get("selected_configs")
    _require(
        isinstance(selected, dict)
        and set(selected) == {str(item) for item in EXPECTED_BATCHES_V29C},
        "V29C report integer batch-key reconstruction changed",
    )
    normalized["runtime_integrity"]["selected_configs"] = {
        int(key): value for key, value in selected.items()
    }
    _require(
        report.get("content_sha256_before_self_field")
        == REPORT_CONTENT_SHA256_V29C
        and canonical_sha256(normalized) == REPORT_CONTENT_SHA256_V29C,
        "V29C report self hash changed",
    )


def _recursive_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).lower()
            yield from _recursive_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _recursive_keys(item)


def _assert_compact_v29c(value):
    overlap = FORBIDDEN_KEYS_V29C & set(_recursive_keys(value))
    if overlap:
        raise RuntimeError(
            f"V29C positive evidence contains forbidden keys: {sorted(overlap)}"
        )


def _validate_preregistration_v29c(preregistration):
    _verify_self_hash(
        preregistration, PREREG_CONTENT_SHA256_V29C, "preregistration",
    )
    authority = preregistration.get("authority", {})
    tuning = preregistration.get("tuning_contract", {})
    compiler = preregistration.get("compiler_error_policy", {})
    persistence = preregistration.get("persistence_contract", {})
    assignments = tuning.get("batch_assignment_by_physical_gpu", {})
    _require(
        preregistration.get("schema")
        == "vllm-moe-fp8-tuning-preregistration-v29b"
        and preregistration.get("strict_selection_only") is True
        and preregistration.get(
            "contains_dataset_rows_questions_answers_or_document_content"
        ) is False
        and preregistration.get(
            "contains_validation_heldout_ood_or_benchmark_content"
        ) is False
        and tuning.get("batch_sizes") == list(EXPECTED_BATCHES_V29C)
        and tuning.get("exhaustive_search_space_configurations_per_batch")
        == 1920
        and tuning.get("total_configurations_across_four_concurrent_workers")
        == 7680
        and tuning.get("search_space_sha256") == SEARCH_SPACE_SHA256_V29C
        and set(assignments) == set(EXPECTED_UUIDS_V29C)
        and all(
            assignments[str(gpu_id)].get("physical_gpu_id") == gpu_id
            and assignments[str(gpu_id)].get("batch_size") == batch
            and assignments[str(gpu_id)].get(
                "search_space_configuration_count"
            ) == 1920
            for gpu_id, batch in enumerate(EXPECTED_BATCHES_V29C)
        )
        and compiler.get("official_OutOfResources_skip_unchanged") is True
        and compiler.get("additional_CompilationError_skip_authorized") is False
        and compiler.get("additional_RuntimeError_skip_authorized") is False
        and persistence.get("contains_train_or_evaluation_rows") is False
        and persistence.get(
            "raw_progress_timing_vectors_compiler_logs_or_search_results_persisted"
        ) is False
        and authority.get("evaluation_authorized") is False
        and authority.get("training_authorized") is False
        and authority.get("model_update_authorized") is False
        and authority.get("checkpoint_write_authorized") is False
        and authority.get("dataset_promotion_authorized") is False
        and authority.get("selected_table_direct_adoption_authorized") is False,
        "V29C preregistration contract changed",
    )


def _validate_table_v29c(table):
    configs = {key: value for key, value in table.items() if key != "triton_version"}
    _require(
        table.get("triton_version") == "3.6.0"
        and configs == EXPECTED_CONFIGS_V29C
        and canonical_sha256(table) == SELECTED_TABLE_CONTENT_SHA256_V29C,
        "V29C selected table content changed",
    )
    return configs


def _validate_semantics_v29c(attempt, report, preregistration, live_table):
    _validate_preregistration_v29c(preregistration)
    live_configs = _validate_table_v29c(live_table)

    expected_prereg = {
        "path": str(PREREG_PATH_V29C.resolve()),
        "file_sha256": PREREG_FILE_SHA256_V29C,
        "content_sha256": PREREG_CONTENT_SHA256_V29C,
    }
    _require(
        attempt.get("schema") == "vllm-moe-fp8-tuning-attempt-v29b"
        and attempt.get("status") == "complete"
        and attempt.get("phase")
        == "after_compact_report_and_final_gpu_cleanup"
        and report.get("schema")
        == "vllm-moe-fp8-tuning-selection-report-v29b"
        and report.get("status") == "complete_selection_not_evaluation"
        and attempt.get("preregistration") == expected_prereg
        and report.get("preregistration") == expected_prereg
        and attempt.get("implementation_bundle_sha256")
        == IMPLEMENTATION_BUNDLE_SHA256_V29C
        and report.get("implementation_bundle_sha256")
        == IMPLEMENTATION_BUNDLE_SHA256_V29C
        and attempt.get("recipe_content_sha256") == RECIPE_CONTENT_SHA256_V29C
        and report.get("recipe_content_sha256") == RECIPE_CONTENT_SHA256_V29C
        and attempt.get("runtime_environment_certificate_sha256")
        == RUNTIME_ENVIRONMENT_SHA256_V29C
        and report.get("runtime_environment_certificate_sha256")
        == RUNTIME_ENVIRONMENT_SHA256_V29C,
        "V29C preregistration implementation or recipe binding changed",
    )
    _require(
        attempt.get("report_binding") == {
            "path": str(REPORT_PATH_V29C.resolve()),
            "file_sha256": REPORT_FILE_SHA256_V29C,
            "content_sha256": REPORT_CONTENT_SHA256_V29C,
        },
        "V29C attempt report binding changed",
    )

    live_audit = attempt.get("live_cpu_disk_audit", {})
    _verify_self_hash(live_audit, LIVE_CPU_DISK_AUDIT_SHA256_V29C, "live audit")
    _require(
        report.get("live_cpu_disk_audit_content_sha256")
        == LIVE_CPU_DISK_AUDIT_SHA256_V29C
        and live_audit.get("official_tuner_sha256")
        == OFFICIAL_TUNER_SHA256_V29C
        and live_audit.get("official_only_OutOfResources_skip_verified") is True
        and live_audit.get(
            "additional_compilation_or_runtime_exception_filter_present"
        ) is False
        and live_audit.get("dataset_train_evaluation_or_nontrain_surface_opened")
        is False
        and live_audit.get("model_all_42_weight_shards_rehashed") is True
        and live_audit.get("model_all_56_files_rehashed") is True,
        "V29C live CPU/disk audit changed",
    )
    _require(
        attempt.get("prelaunch_idle_certificate_sha256")
        == PRELAUNCH_IDLE_SHA256_V29C
        and report.get("prelaunch_idle_certificate_sha256")
        == PRELAUNCH_IDLE_SHA256_V29C
        and attempt.get("final_idle_certificate_sha256")
        == FINAL_IDLE_SHA256_V29C
        and report.get("final_idle_certificate_sha256")
        == FINAL_IDLE_SHA256_V29C
        and attempt.get("selected_table_written") is True
        and attempt.get("selected_table_file_sha256")
        == SELECTED_TABLE_FILE_SHA256_V29C
        and attempt.get("selected_table_content_sha256")
        == SELECTED_TABLE_CONTENT_SHA256_V29C,
        "V29C idle or selected-table attempt binding changed",
    )

    selected = report.get("selected_table", {})
    integrity = report.get("runtime_integrity", {})
    _require(
        selected.get("path") == str(LIVE_TABLE_PATH_V29C.resolve())
        and selected.get("file_sha256") == SELECTED_TABLE_FILE_SHA256_V29C
        and selected.get("content_sha256") == SELECTED_TABLE_CONTENT_SHA256_V29C
        and selected.get("triton_version") == "3.6.0"
        and selected.get("configs") == live_configs
        and integrity.get("selected_configs") == live_configs,
        "V29C selected table report binding changed",
    )

    inflight = integrity.get("inflight_physical_gpu_utilization", {})
    per_gpu = inflight.get("per_gpu", {})
    _require(
        integrity.get("official_worker_count") == 4
        and integrity.get("configuration_count_per_worker") == 1920
        and integrity.get("all_four_official_tune_futures_submitted_before_ray_get")
        is True
        and integrity.get("all_four_distinct_uuid_workers_observed_active_and_positive")
        is True
        and integrity.get("official_source_unmodified") is True
        and integrity.get("only_official_OutOfResources_skip_active") is True
        and integrity.get("bf16_v27c_table_loaded_or_reused") is False
        and integrity.get("all_four_gpus_idle_after_cleanup") is True
        and integrity.get("prelaunch_idle_makes_observed_inflight_processes_tuner_owned")
        is True
        and integrity.get("placement_identity_commitment_before_sha256")
        == PLACEMENT_IDENTITY_SHA256_V29C
        and integrity.get("placement_identity_commitment_after_sha256")
        == PLACEMENT_IDENTITY_SHA256_V29C
        and integrity.get("official_actor_pid_map_commitment_sha256")
        == ACTOR_PID_MAP_SHA256_V29C
        and set(per_gpu) == set(EXPECTED_UUIDS_V29C)
        and inflight.get(
            "simultaneous_all_four_assigned_actor_pids_and_positive_utilization_observation_count"
        ) == 4842
        and inflight.get("simultaneous_all_four_requirement_passed") is True,
        "V29C aggregate official tuning integrity changed",
    )
    for gpu_id, uuid in EXPECTED_UUIDS_V29C.items():
        item = per_gpu[gpu_id]
        _require(
            item == {
                "physical_gpu_id": int(gpu_id),
                "nvml_uuid": uuid,
                "assigned_official_actor_pid_observed": True,
                "running_process_observed": True,
                "positive_gpu_utilization_observed": True,
                "maximum_gpu_utilization_percent": 100,
                "sample_count": 10321,
            },
            f"V29C GPU {gpu_id} utilization certificate changed",
        )

    closed_attempt = (
        "training_model_update_checkpoint_adoption_evaluation_or_dataset_promotion_applied",
        "dataset_train_evaluation_validation_heldout_ood_or_benchmark_surface_opened",
    )
    _require(
        all(attempt.get(key) is False for key in closed_attempt)
        and report.get(
            "direct_adoption_training_model_update_checkpoint_evaluation_or_dataset_promotion_authorized"
        ) is False
        and report.get("direct_action_taken") is False
        and report.get(
            "dataset_train_evaluation_validation_heldout_ood_or_benchmark_surface_opened"
        ) is False
        and report.get(
            "raw_progress_timing_vectors_compiler_logs_or_search_results_persisted"
        ) is False
        and report.get("decision")
        == "authorize_only_separate_fp8_table_evaluation_preregistration",
        "V29C closed authority or decision changed",
    )
    return live_configs


def validate_bound_artifacts_v29c():
    expected_files = {
        ATTEMPT_PATH_V29C: ATTEMPT_FILE_SHA256_V29C,
        REPORT_PATH_V29C: REPORT_FILE_SHA256_V29C,
        PREREG_PATH_V29C: PREREG_FILE_SHA256_V29C,
        LIVE_TABLE_PATH_V29C: SELECTED_TABLE_FILE_SHA256_V29C,
    }
    for path, expected in expected_files.items():
        _require(file_sha256(path) == expected, f"V29C file hash changed: {path.name}")
    attempt = _load_json_object(ATTEMPT_PATH_V29C, "attempt")
    report = _load_json_object(REPORT_PATH_V29C, "report")
    preregistration = _load_json_object(PREREG_PATH_V29C, "preregistration")
    live_table = _load_json_object(LIVE_TABLE_PATH_V29C, "live selected table")
    _verify_self_hash(attempt, ATTEMPT_CONTENT_SHA256_V29C, "attempt")
    _verify_report_self_hash_v29c(report)
    configs = _validate_semantics_v29c(
        attempt, report, preregistration, live_table,
    )
    return attempt, report, preregistration, configs


def build_positive_evidence_v29c():
    attempt, report, _preregistration, _configs = validate_bound_artifacts_v29c()
    integrity = report["runtime_integrity"]
    inflight = integrity["inflight_physical_gpu_utilization"]
    per_gpu = inflight["per_gpu"]
    value = _seal({
        "schema": "vllm-moe-fp8-tuning-selection-positive-evidence-v29c",
        "status": "valid_completed_selection_not_evaluation",
        "contracts": {
            "v29b_preregistration_file_sha256": PREREG_FILE_SHA256_V29C,
            "v29b_preregistration_content_sha256": PREREG_CONTENT_SHA256_V29C,
            "v29b_implementation_bundle_sha256": IMPLEMENTATION_BUNDLE_SHA256_V29C,
            "v29b_recipe_content_sha256": RECIPE_CONTENT_SHA256_V29C,
            "runtime_environment_certificate_sha256": RUNTIME_ENVIRONMENT_SHA256_V29C,
            "live_cpu_disk_audit_content_sha256": LIVE_CPU_DISK_AUDIT_SHA256_V29C,
            "official_tuner_sha256": OFFICIAL_TUNER_SHA256_V29C,
            "search_space_sha256": SEARCH_SPACE_SHA256_V29C,
        },
        "artifacts": {
            "ignored_live_attempt": {
                "relative_path": ATTEMPT_RELATIVE_PATH_V29C,
                "file_sha256": ATTEMPT_FILE_SHA256_V29C,
                "content_sha256": ATTEMPT_CONTENT_SHA256_V29C,
                "status": attempt["status"],
                "phase": attempt["phase"],
            },
            "ignored_live_report": {
                "relative_path": REPORT_RELATIVE_PATH_V29C,
                "file_sha256": REPORT_FILE_SHA256_V29C,
                "content_sha256": REPORT_CONTENT_SHA256_V29C,
            },
            "original_official_selected_table": {
                "relative_path": LIVE_TABLE_RELATIVE_PATH_V29C,
                "file_sha256": SELECTED_TABLE_FILE_SHA256_V29C,
                "content_sha256": SELECTED_TABLE_CONTENT_SHA256_V29C,
                "must_be_force_added_at_exact_original_path_in_v29c_commit": True,
            },
        },
        "aggregate_result": {
            "official_worker_count": integrity["official_worker_count"],
            "configurations_per_worker": integrity[
                "configuration_count_per_worker"
            ],
            "total_configurations": (
                integrity["official_worker_count"]
                * integrity["configuration_count_per_worker"]
            ),
            "selected_batch_sizes": list(EXPECTED_BATCHES_V29C),
            "selected_table_file_sha256": SELECTED_TABLE_FILE_SHA256_V29C,
            "selected_table_content_sha256": SELECTED_TABLE_CONTENT_SHA256_V29C,
            "per_gpu": {
                gpu_id: {
                    "nvml_uuid": item["nvml_uuid"],
                    "assigned_official_actor_pid_observed": item[
                        "assigned_official_actor_pid_observed"
                    ],
                    "maximum_gpu_utilization_percent": item[
                        "maximum_gpu_utilization_percent"
                    ],
                    "sample_count": item["sample_count"],
                }
                for gpu_id, item in per_gpu.items()
            },
            "simultaneous_all_four_positive_observation_count": inflight[
                "simultaneous_all_four_assigned_actor_pids_and_positive_utilization_observation_count"
            ],
            "all_four_distinct_uuid_workers_active_and_positive": integrity[
                "all_four_distinct_uuid_workers_observed_active_and_positive"
            ],
            "all_four_official_futures_submitted_before_wait": integrity[
                "all_four_official_tune_futures_submitted_before_ray_get"
            ],
            "placement_identity_commitment_before_sha256": integrity[
                "placement_identity_commitment_before_sha256"
            ],
            "placement_identity_commitment_after_sha256": integrity[
                "placement_identity_commitment_after_sha256"
            ],
            "official_actor_pid_map_commitment_sha256": integrity[
                "official_actor_pid_map_commitment_sha256"
            ],
            "all_four_gpus_finally_idle": integrity[
                "all_four_gpus_idle_after_cleanup"
            ],
            "official_source_unmodified": integrity[
                "official_source_unmodified"
            ],
            "only_official_OutOfResources_skip_active": integrity[
                "only_official_OutOfResources_skip_active"
            ],
            "bf16_v27c_table_loaded_or_reused": integrity[
                "bf16_v27c_table_loaded_or_reused"
            ],
        },
        "decision": {
            "selection_positive_evidence_valid": True,
            "authorize_only_separate_fp8_table_evaluation_preregistration": True,
            "evaluation_authorized_by_this_evidence": False,
            "selected_table_direct_adoption_authorized": False,
            "training_or_model_update_authorized": False,
            "checkpoint_write_authorized": False,
            "dataset_promotion_authorized": False,
            "validation_heldout_ood_or_benchmark_open_authorized": False,
        },
        "contains_dataset_rows_questions_answers_or_document_content": False,
        "contains_validation_heldout_ood_or_benchmark_content": False,
        "raw_progress_timing_vectors_compiler_logs_search_results_or_pids_persisted": False,
    })
    _assert_compact_v29c(value)
    return value


def _exclusive_write_json(path, value):
    path = Path(path).resolve()
    if path != OUTPUT_PATH_V29C.resolve():
        raise ValueError("V29C evidence output path changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise RuntimeError("V29C positive evidence already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH_V29C)
    args = parser.parse_args(argv)
    value = build_positive_evidence_v29c()
    if not args.dry_run:
        _exclusive_write_json(args.output, value)
    print(json.dumps({
        "schema": "vllm-moe-fp8-tuning-positive-evidence-build-v29c",
        "content_sha256": value["content_sha256_before_self_field"],
        "selected_table_file_sha256": SELECTED_TABLE_FILE_SHA256_V29C,
        "total_configurations": value["aggregate_result"]["total_configurations"],
        "evaluation_authorized": False,
        "gpu_launched": False,
    }, sort_keys=True))
    return value


if __name__ == "__main__":
    main()
