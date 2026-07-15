#!/usr/bin/env python3
"""Build compact infrastructure-failure evidence for the failed V29D launch."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ATTEMPT_RELATIVE_PATH_V29E = (
    "experiments/eggroll_es_hpo/runs/"
    ".s6_v29d_fp8_selected_table_paired_synthetic_kernel_evaluation."
    "launch_attempt.json"
)
REPORT_RELATIVE_PATH_V29E = (
    "experiments/eggroll_es_hpo/runs/"
    "s6_v29d_fp8_selected_table_paired_synthetic_kernel_evaluation/"
    "fp8_selected_table_evaluation_report_v29d.json"
)
PREREG_RELATIVE_PATH_V29E = (
    "experiments/eggroll_es_hpo/"
    "S6_V29D_FP8_SELECTED_TABLE_EVALUATION_PREREGISTRATION.json"
)
OUTPUT_RELATIVE_PATH_V29E = (
    "experiments/eggroll_es_hpo/"
    "S6_V29E_V29D_FP8_SELECTED_TABLE_INFRASTRUCTURE_FAILURE_EVIDENCE.json"
)
ATTEMPT_PATH_V29E = ROOT / ATTEMPT_RELATIVE_PATH_V29E
REPORT_PATH_V29E = ROOT / REPORT_RELATIVE_PATH_V29E
PREREG_PATH_V29E = ROOT / PREREG_RELATIVE_PATH_V29E
OUTPUT_PATH_V29E = ROOT / OUTPUT_RELATIVE_PATH_V29E

ATTEMPT_FILE_SHA256_V29E = (
    "1dabc51d07b0728f2f0492a4404fdd3b8ab61127d26d2c5f693dd8bcca4bb08f"
)
ATTEMPT_CONTENT_SHA256_V29E = (
    "3d7f133610cb8eca9a0ce5ffb930170f627b746d18ff56fce04f73c9c0a76142"
)
PREREG_COMMIT_V29E = "23d6bd0f7d90a7438488e55eb796928b9a0bfd31"
PREREG_FILE_SHA256_V29E = (
    "55519ce1150f3466817cbe3b7f6d22b4b79553755fc35c7e28a45b038dba7f42"
)
PREREG_CONTENT_SHA256_V29E = (
    "8bb3ecd02992f46cc2eb8a7f3144966ab5139f4b59fd1e179ef021a5d5687061"
)
FAILURE_MESSAGE_V29E = (
    "V29D did not observe all four assigned PIDs with utilization"
)
FAILURE_MESSAGE_SHA256_V29E = (
    "c0dd55cfb46132f2d48cdcaea5cb128d93f183d60e816fd9eff93b5a2b7c97c8"
)
IMPLEMENTATION_BUNDLE_SHA256_V29E = (
    "12207d122ea2a8ae1d1204361f152667645f70beac69962c758ced2937c78885"
)
RECIPE_CONTENT_SHA256_V29E = (
    "0f3c3eed8cc4abc34d7c478838073f64b64648f518e0503ba4da2c2e24b9e053"
)
RUNTIME_ENVIRONMENT_SHA256_V29E = (
    "7acc87ffb6a50682f19309ea3059823d0fc96dfc459e30cb66d64e157a76f52a"
)
LIVE_CPU_DISK_AUDIT_SHA256_V29E = (
    "122a690d0379084dd35a9947a480afcf9fb1671e8f634fabb33844bc61d71666"
)
PRELAUNCH_IDLE_SHA256_V29E = (
    "d1a73084e65e24ef8048a6fffeb9a6cb6f1b17fa1278e51cb4cba991590d3913"
)
FINAL_IDLE_SHA256_V29E = (
    "81b02885c550d90dd84edc32104c7bae658405c188bffa97583db3f285d23166"
)
V29C_COMMIT_V29E = "a203f4821c4a737310df75543353d21ce6cea978"
V29C_FILE_SHA256_V29E = (
    "47d1b09fb188dd1f8ff16314f1c20fe614f02b1cff067a1615a0d6f0f5ce2a7b"
)
V29C_CONTENT_SHA256_V29E = (
    "dc4d3b6d2b090e4e740f63de573875f331a456d6951b62cf49a003b1114ee02e"
)
FORBIDDEN_KEYS_V29E = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "response", "responses", "token_ids", "timing_vectors", "raw_pids",
    "raw_message", "traceback", "training_rows", "evaluation_rows",
    "validation_rows", "heldout_rows", "ood_rows", "benchmark_rows",
}


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def _recursive_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).lower()
            yield from _recursive_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _recursive_keys(item)


def _assert_compact_v29e(value):
    overlap = FORBIDDEN_KEYS_V29E & set(_recursive_keys(value))
    if overlap:
        raise RuntimeError(f"V29E failure evidence contains forbidden keys: {sorted(overlap)}")


def _verify_self(value, expected, label):
    _require(
        value.get("content_sha256_before_self_field") == expected
        and canonical_sha256(_without_self(value)) == expected,
        f"V29E {label} self hash changed",
    )


def validate_v29d_failure_v29e():
    _require(
        ATTEMPT_PATH_V29E.is_file()
        and not ATTEMPT_PATH_V29E.is_symlink()
        and file_sha256(ATTEMPT_PATH_V29E) == ATTEMPT_FILE_SHA256_V29E,
        "V29E V29D attempt file identity changed",
    )
    _require(
        not REPORT_PATH_V29E.exists(),
        "V29E requires the V29D report to be absent",
    )
    raw = subprocess.check_output(
        ["git", "show", f"{PREREG_COMMIT_V29E}:{PREREG_RELATIVE_PATH_V29E}"],
        cwd=ROOT,
    )
    _require(
        hashlib.sha256(raw).hexdigest() == PREREG_FILE_SHA256_V29E
        and file_sha256(PREREG_PATH_V29E) == PREREG_FILE_SHA256_V29E,
        "V29E committed V29D preregistration identity changed",
    )
    preregistration = json.loads(PREREG_PATH_V29E.read_text(encoding="utf-8"))
    _verify_self(preregistration, PREREG_CONTENT_SHA256_V29E, "V29D preregistration")
    attempt = json.loads(ATTEMPT_PATH_V29E.read_text(encoding="utf-8"))
    _verify_self(attempt, ATTEMPT_CONTENT_SHA256_V29E, "V29D attempt")
    _require(
        hashlib.sha256(FAILURE_MESSAGE_V29E.encode("utf-8")).hexdigest()
        == FAILURE_MESSAGE_SHA256_V29E,
        "V29E known observability failure message identity changed",
    )
    _require(
        attempt.get("schema")
        == "vllm-moe-fp8-selected-table-evaluation-attempt-v29d"
        and attempt.get("status") == "failed"
        and attempt.get("phase")
        == "inside_counterbalanced_synthetic_kernel_evaluation"
        and attempt.get("preregistration") == {
            "path": str(PREREG_PATH_V29E.resolve()),
            "file_sha256": PREREG_FILE_SHA256_V29E,
            "content_sha256": PREREG_CONTENT_SHA256_V29E,
        }
        and attempt.get("implementation_bundle_sha256")
        == IMPLEMENTATION_BUNDLE_SHA256_V29E
        and attempt.get("recipe_content_sha256") == RECIPE_CONTENT_SHA256_V29E
        and attempt.get("runtime_environment_certificate_sha256")
        == RUNTIME_ENVIRONMENT_SHA256_V29E
        and attempt.get("live_cpu_disk_audit_content_sha256")
        == LIVE_CPU_DISK_AUDIT_SHA256_V29E,
        "V29E V29D attempt contract binding changed",
    )
    _require(
        attempt.get("selection_evidence") == {
            "commit": V29C_COMMIT_V29E,
            "path": str(ROOT / (
                "experiments/eggroll_es_hpo/"
                "S6_V29C_FP8_MOE_TUNING_SELECTION_POSITIVE_EVIDENCE.json"
            )),
            "file_sha256": V29C_FILE_SHA256_V29E,
            "content_sha256": V29C_CONTENT_SHA256_V29E,
            "authorizes_only_this_separate_evaluation_preregistration": True,
        },
        "V29E V29C selection evidence binding changed",
    )
    failure = attempt.get("failure", {})
    _require(
        failure == {
            "exception_class": "RuntimeError",
            "message_sha256": FAILURE_MESSAGE_SHA256_V29E,
            "raw_message_or_traceback_persisted": False,
        }
        and attempt.get("prelaunch_idle_certificate_sha256")
        == PRELAUNCH_IDLE_SHA256_V29E
        and attempt.get("final_idle_certificate_sha256")
        == FINAL_IDLE_SHA256_V29E
        and attempt.get("direct_action_taken") is False
        and attempt.get(
            "dataset_evaluation_validation_heldout_ood_or_benchmark_surface_opened"
        ) is False,
        "V29E failure boundary idle cleanup or closed surface changed",
    )
    return attempt, preregistration


def build_failure_evidence_v29e():
    _attempt, _preregistration = validate_v29d_failure_v29e()
    value = _seal({
        "schema": "vllm-moe-fp8-selected-table-infrastructure-failure-evidence-v29e",
        "status": "valid_failed_before_evaluation_report_observability_only",
        "failed_attempt": {
            "relative_path": ATTEMPT_RELATIVE_PATH_V29E,
            "file_sha256": ATTEMPT_FILE_SHA256_V29E,
            "content_sha256": ATTEMPT_CONTENT_SHA256_V29E,
            "failure_exception_class": "RuntimeError",
            "failure_message_sha256": FAILURE_MESSAGE_SHA256_V29E,
            "phase": "inside_counterbalanced_synthetic_kernel_evaluation",
        },
        "contracts": {
            "v29d_preregistration_commit": PREREG_COMMIT_V29E,
            "v29d_preregistration_file_sha256": PREREG_FILE_SHA256_V29E,
            "v29d_preregistration_content_sha256": PREREG_CONTENT_SHA256_V29E,
            "v29d_implementation_bundle_sha256": IMPLEMENTATION_BUNDLE_SHA256_V29E,
            "v29d_recipe_content_sha256": RECIPE_CONTENT_SHA256_V29E,
            "v29c_selection_commit": V29C_COMMIT_V29E,
            "v29c_selection_file_sha256": V29C_FILE_SHA256_V29E,
            "v29c_selection_content_sha256": V29C_CONTENT_SHA256_V29E,
        },
        "failure_boundary": {
            "activity_observability_gate_failed": True,
            "kernel_statistical_evaluation_completed": False,
            "evaluation_report_written": False,
            "prelaunch_idle_certificate_sha256": PRELAUNCH_IDLE_SHA256_V29E,
            "final_idle_certificate_sha256": FINAL_IDLE_SHA256_V29E,
            "final_idle_certificate_present": True,
            "direct_table_adoption_or_action_taken": False,
            "dataset_or_nontrain_surface_opened": False,
        },
        "decision": {
            "v29d_result_available": False,
            "selected_table_adoption_authorized": False,
            "authorize_only_exact_observability_retry_preregistration": True,
            "training_model_update_checkpoint_dataset_promotion_authorized": False,
        },
        "raw_message_traceback_timing_output_or_pid_details_persisted": False,
        "contains_dataset_rows_questions_answers_or_document_content": False,
        "contains_validation_heldout_ood_or_benchmark_content": False,
    })
    _assert_compact_v29e(value)
    return value


def _exclusive_write(path, value):
    path = Path(path).resolve()
    if path != OUTPUT_PATH_V29E.resolve() or path.exists():
        raise RuntimeError("V29E failure evidence output must be fresh and exact")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH_V29E)
    args = parser.parse_args(argv)
    value = build_failure_evidence_v29e()
    if not args.dry_run:
        _exclusive_write(args.output, value)
    print(json.dumps({
        "schema": "vllm-moe-fp8-selected-table-failure-evidence-build-v29e",
        "content_sha256": value["content_sha256_before_self_field"],
        "evaluation_report_written": False,
        "selected_table_adoption_authorized": False,
        "gpu_launched": False,
    }, sort_keys=True))
    return value


if __name__ == "__main__":
    main()
