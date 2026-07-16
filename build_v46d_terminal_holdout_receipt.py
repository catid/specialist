#!/usr/bin/env python3
"""Seal V46D's aggregate-only terminal receipt without protected-data access."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import run_matched_lora_candidate_eval_v44a as core


ROOT = Path(__file__).resolve().parent
PREREG = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "once_only_sft_v42i_sealed_holdout_eval_v46d.json"
).resolve()
REPORT = (
    ROOT / "experiments/eval_reports/"
    "once_only_sft_v42i_sealed_holdout_eval_v46d.json"
).resolve()
ATTEMPT = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    ".v46d_once_only_sft_v42i_sealed_holdout_eval.attempt.json"
).resolve()
COMPLETE = ATTEMPT.with_suffix(".complete.json")
GPU_LOG = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v46d_once_only_sft_v42i_sealed_holdout_eval/gpu_activity_v46d.jsonl"
).resolve()
OUTPUT = (
    ROOT / "experiments/eval_reports/"
    "v46d_once_only_sft_v42i_terminal_receipt.json"
).resolve()
EXPECTED = {
    "prereg_file": "80f2c165e8f60b129e776b47da73bd9f7dba15adb949214879bb11e00f608084",
    "prereg_content": "841f8c25234424927a12f19e5446da588d1e9907ed2634de2d2c1b6e072fe7cb",
    "report_file": "2aa501987500df5c9df931e8c2bb0f769a8de6559ab6755d1493c2d18b94ec21",
    "report_content": "33730688cb62a0fa6fbf5c96c402853c7a858480ee7c8af0a0405cb348534332",
    "attempt_file": "3446adc43afbab0e4afd098e9a20a31b0ee08c6daa041fb0b404461e6942d3c6",
    "attempt_content": "7e1a768d40e6141a8f73da69c147934718e4188f4140ef38019bbd7212f371a5",
    "complete_file": "cb44acc9a79ddfbc5a82b10623e3f68ab2a2e53695625389f9d7da10fd2eb55d",
    "complete_content": "426d9b04179ec7efc205c35f953a58cd2b36b65fa15eff890d25ec72f0658660",
    "gpu_log_file": "e4a6f4ec5f7277a9f71f4798d117e1c071c201b5a32fa68740e780f678edeef6",
    "holdout_file": "ab9a391e249910e876826dfab9c8e2f8e17a7b8695e6f018a3e515e5aa69603b",
}


def _compact_sha(value: dict) -> str:
    return core.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def _load_self_hashed(path: Path, file_sha: str, content_sha: str,
                      label: str) -> dict:
    if core.file_sha256(path) != file_sha:
        raise RuntimeError(f"V46D terminal receipt {label} file changed")
    value = json.loads(path.read_text())
    if (
        value.get("content_sha256_before_self_field") != content_sha
        or _compact_sha(value) != content_sha
    ):
        raise RuntimeError(f"V46D terminal receipt {label} content changed")
    return value


def build() -> dict:
    prereg = _load_self_hashed(
        PREREG, EXPECTED["prereg_file"], EXPECTED["prereg_content"], "prereg"
    )
    report = _load_self_hashed(
        REPORT, EXPECTED["report_file"], EXPECTED["report_content"], "report"
    )
    attempt = _load_self_hashed(
        ATTEMPT, EXPECTED["attempt_file"], EXPECTED["attempt_content"], "attempt"
    )
    complete = _load_self_hashed(
        COMPLETE, EXPECTED["complete_file"], EXPECTED["complete_content"],
        "complete attempt",
    )
    if core.file_sha256(GPU_LOG) != EXPECTED["gpu_log_file"]:
        raise RuntimeError("V46D terminal receipt GPU log changed")

    if (
        prereg.get("schema")
        != "once-only-fixed-sft-v42i-holdout-preregistration-v46d"
        or prereg.get("fixed_candidate_arm") != "sft_v42i"
        or prereg.get("candidate_selection_permitted") is not False
        or prereg.get("post_result_tuning_or_selection_permitted") is not False
        or prereg.get("holdout_access_count_before_preregistration") != 0
        or prereg.get("post_result_policy", {}).get(
            "holdout_may_not_be_reopened"
        ) is not True
        or prereg.get("post_result_policy", {}).get(
            "result_may_not_trigger_checkpoint_selection"
        ) is not True
        or prereg.get("post_result_policy", {}).get(
            "result_may_not_trigger_hyperparameter_tuning"
        ) is not True
        or prereg.get("post_result_policy", {}).get(
            "result_may_not_trigger_data_selection_or_resampling"
        ) is not True
    ):
        raise RuntimeError("V46D terminal receipt preregistered policy changed")

    gate = report.get("fixed_non_degradation_gate", {})
    access = report.get("single_access_receipt", {})
    gpu = report.get("gpu_activity", {})
    cleanup = report.get("placement_group_cleanup", {})
    if (
        report.get("schema")
        != "once-only-fixed-sft-v42i-holdout-aggregate-v46d"
        or report.get("status") != "complete_once_only_aggregate_no_selection"
        or report.get("runtime_revision") != "v46d_v45f_v46c_resolved"
        or report.get("preregistration", {}).get("file_sha256")
        != EXPECTED["prereg_file"]
        or report.get("preregistration", {}).get("content_sha256")
        != EXPECTED["prereg_content"]
        or report.get("fixed_candidate_arm") != "sft_v42i"
        or report.get("candidate_selection_performed") is not False
        or report.get("post_result_tuning_or_selection_permitted") is not False
        or report.get("base_duplicate_equivalence", {}).get("exact") is not True
        or report.get("holdout_opened") is not True
        or report.get("holdout_semantic_access_count") != 1
        or access.get("semantic_read_count") != 1
        or access.get("file_sha256") != EXPECTED["holdout_file"]
        or report.get("holdout_identity", {}).get("file_sha256")
        != EXPECTED["holdout_file"]
        or report.get("document_disjointness", {}).get(
            "validation_vs_heldout_document_intersection_count"
        ) != 0
        or report.get("raw_questions_answers_or_generations_persisted")
        is not False
        or gpu.get("all_four_attributed_positive_on_sealed_holdout") is not True
        or set(gpu.get("by_gpu", {})) != {"0", "1", "2", "3"}
        or any(
            item.get("resident_samples", 0) <= 0
            or item.get("positive_samples", 0) <= 0
            for item in gpu.get("by_gpu", {}).values()
        )
        or cleanup.get("engine_kill_count") != 4
        or cleanup.get("placement_group_remove_count") != 4
        or cleanup.get("all_four_gcs_states_removed") is not True
        or report.get("final_gpu_idle", {}).get(
            "all_four_compute_process_lists_empty"
        ) is not True
        or report.get("gpu_log", {}).get("file_sha256")
        != EXPECTED["gpu_log_file"]
        or gate.get("passed") is not False
        or gate.get("mean_reward_delta") != -0.0036501348892653135
        or gate.get("mean_reward_point_non_degradation") is not False
        or gate.get("exact_count_point_non_degradation") is not True
        or gate.get("nonzero_count_point_non_degradation") is not True
        or gate.get("no_protocol_or_leak_counter_increase") is not True
        or gate.get("candidate_selection_performed") is not False
    ):
        raise RuntimeError("V46D terminal receipt aggregate invariant changed")

    if (
        attempt.get("schema")
        != "once-only-fixed-sft-v42i-holdout-attempt-v46d"
        or attempt.get("status") != "launching_irrevocable_attempt"
        or attempt.get("phase") != "before_model_or_holdout_access"
        or attempt.get("holdout_semantic_access_count") != 0
        or attempt.get("holdout_opened") is not False
        or attempt.get("candidate_selection_performed") is not False
        or complete.get("schema")
        != "once-only-fixed-sft-v42i-holdout-attempt-v46d"
        or complete.get("status") != "complete_once_only_holdout_consumed"
        or complete.get("phase") != "aggregate_sealed"
        or complete.get("holdout_semantic_access_count") != 1
        or complete.get("holdout_opened") is not True
        or complete.get("candidate_selection_performed") is not False
        or complete.get("report_file_sha256") != EXPECTED["report_file"]
        or complete.get("preregistration_file_sha256")
        != EXPECTED["prereg_file"]
        or complete.get("preregistration_content_sha256")
        != EXPECTED["prereg_content"]
    ):
        raise RuntimeError("V46D terminal receipt attempt invariant changed")

    value = {
        "schema": "once-only-sft-v42i-holdout-terminal-receipt-v46d",
        "status": "terminal_holdout_consumed_gate_failed_no_retry",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bindings": {
            "preregistration": {
                "path": str(PREREG), "file_sha256": EXPECTED["prereg_file"],
                "content_sha256": EXPECTED["prereg_content"],
            },
            "aggregate_report": {
                "path": str(REPORT), "file_sha256": EXPECTED["report_file"],
                "content_sha256": EXPECTED["report_content"],
            },
            "attempt_before_access": {
                "path": str(ATTEMPT), "file_sha256": EXPECTED["attempt_file"],
                "content_sha256": EXPECTED["attempt_content"],
            },
            "attempt_complete": {
                "path": str(COMPLETE), "file_sha256": EXPECTED["complete_file"],
                "content_sha256": EXPECTED["complete_content"],
            },
            "gpu_log": {
                "path": str(GPU_LOG),
                "file_sha256": EXPECTED["gpu_log_file"],
            },
        },
        "audited_outcome": {
            "fixed_candidate": "sft_v42i",
            "candidate_selection_performed": False,
            "single_holdout_access_consumed": True,
            "holdout_semantic_access_count": 1,
            "fixed_gate_passed": False,
            "only_failed_point_gate": "mean_reward_point_non_degradation",
            "mean_reward_delta": -0.0036501348892653135,
            "exact_count_delta": 0,
            "nonzero_count_delta": 0,
            "protocol_or_leak_counter_increase": False,
            "reward_mean_delta_paired_item_bootstrap_95_ci": gate[
                "reward_mean_delta_paired_item_bootstrap_95_ci"
            ],
            "bootstrap_ci_is_informational": True,
            "three_base_duplicates_exact": True,
            "document_disjointness_verified": True,
            "raw_questions_answers_or_generations_persisted": False,
            "all_four_gpus_resident_and_positive": True,
            "all_four_gpus_idle_after_cleanup": True,
        },
        "terminal_policy": {
            "retry_or_reopen_holdout_permitted": False,
            "additional_holdout_evaluation_permitted": False,
            "use_result_for_checkpoint_selection_permitted": False,
            "use_result_for_candidate_selection_permitted": False,
            "use_result_for_hyperparameter_tuning_permitted": False,
            "use_result_for_dataset_selection_or_resampling_permitted": False,
            "use_result_for_training_recipe_tuning_permitted": False,
            "descriptive_reporting_of_preregistered_aggregate_permitted": True,
            "terminal_reason": (
                "the preregistered single access is consumed regardless of "
                "pass/fail outcome"
            ),
        },
        "protected_or_holdout_semantics_inspected_while_building_receipt": False,
        "holdout_reopened_or_rehashed_while_building_receipt": False,
        "aggregate_only_artifacts_inspected": True,
        "gpu_log_content_inspected": False,
    }
    value["content_sha256_before_self_field"] = core.canonical_sha256(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build()
    core.atomic_json(output, value)
    print(json.dumps({
        "path": str(output), "file_sha256": core.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "status": value["status"], "holdout_access_count": 1,
        "retry_permitted": False, "tuning_or_selection_permitted": False,
        "protected_semantics_inspected": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
