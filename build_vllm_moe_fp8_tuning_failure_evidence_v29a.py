#!/usr/bin/env python3
"""Build compact, fail-closed evidence for the immutable V29A failure."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import build_vllm_moe_fp8_tuning_preregistration_v29a as prereg


ROOT = Path(__file__).resolve().parent
ATTEMPT_RELATIVE_PATH_V29A = (
    "experiments/eggroll_es_hpo/runs/"
    ".s6_v29a_fp8_w8a8_block128_moe_tuning_selection.launch_attempt.json"
)
REPORT_RELATIVE_PATH_V29A = (
    "experiments/eggroll_es_hpo/runs/"
    "s6_v29a_fp8_w8a8_block128_moe_tuning_selection/"
    "fp8_moe_tuning_selection_report_v29a.json"
)
OUTPUT_DIRECTORY_RELATIVE_PATH_V29A = (
    "experiments/vllm_moe_tuning/"
    "v025_rtx_pro_6000_fp8_w8a8_block128_tp1_exhaustive_v29a"
)
RUNTIME_RELATIVE_PATH_V29A = "run_vllm_moe_fp8_tuning_v29a.py"
OUTPUT_PATH_V29A = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V29A_FP8_MOE_TUNING_INFRASTRUCTURE_FAILURE_EVIDENCE.json"
)
EXPECTED_ATTEMPT_FILE_SHA256_V29A = (
    "1c19e5f924ca4839a84d71bfd1af2e44945ef4871ca657e2c6d2bd7af71db7ee"
)
EXPECTED_ATTEMPT_CONTENT_SHA256_V29A = (
    "0c6fc60eecb0bd730c18fae66c36e7071080fae08511aadcd51540ea186cef2a"
)
EXPECTED_PREREG_FILE_SHA256_V29A = (
    "3fabaef03dde9944005937d4d975b4d546f330d66a426580720d0e3da5dc768d"
)
EXPECTED_PREREG_CONTENT_SHA256_V29A = (
    "bf605c113bec2730599160618bd8e1b483a4317b024f68a309b80c7ed436f8cb"
)
EXPECTED_IMPLEMENTATION_BUNDLE_SHA256_V29A = (
    "f4a282659dd899f9b9be1c4ecd6c55f07f5e2126f26179d48d91dc9940aafba3"
)
EXPECTED_RECIPE_CONTENT_SHA256_V29A = (
    "3fb6256df6f8f3b46808ef0649ff0d16505fd6fc4f5010441c3b184d8ed3c558"
)
EXPECTED_RUNTIME_ENVIRONMENT_CERTIFICATE_SHA256_V29A = (
    "7acc87ffb6a50682f19309ea3059823d0fc96dfc459e30cb66d64e157a76f52a"
)
EXPECTED_LIVE_AUDIT_CONTENT_SHA256_V29A = (
    "7285cda7663a42be6f41aacb2fe92a56df5ccb076e269f940252dea01322b806"
)
EXPECTED_PRELAUNCH_IDLE_CERTIFICATE_SHA256_V29A = (
    "491036be54f8230baec0ea7f173d7d7c20f537ab7fc8eb44f5b91cf80b37a931"
)
EXPECTED_FINAL_IDLE_CERTIFICATE_SHA256_V29A = (
    "505b96d3754a97ac2796c43987e16c9d7b2a54db8397b20622d668a468d7c8ce"
)
EXPECTED_RUNTIME_FILE_SHA256_V29A = (
    "532c06fe1a5c16ec4c85f8dda0ff0415a824ab488ff625a69823fc45706a2166"
)
EXPECTED_EXCEPTION_CLASS_V29A = "ServerUnavailable"
EXPECTED_MESSAGE_SHA256_V29A = (
    "fa15e5000aa7aea753bf68526312eded1a75026e9752ec130356d99207ca712a"
)
FORBIDDEN_KEYS_V29A = {
    "traceback", "message", "raw_message", "raw_traceback", "compiler_log",
    "progress_log", "search_results", "timing_vectors", "question",
    "questions", "answer", "answers", "prompt", "prompts", "responses",
    "training_rows", "evaluation_rows", "validation_rows", "heldout_rows",
    "ood_rows", "token_ids", "text", "texts",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _seal(value):
    value = copy.deepcopy(value)
    value.pop("content_sha256_before_self_field", None)
    value["content_sha256_before_self_field"] = prereg.canonical_sha256(value)
    return value


def _recursive_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).lower()
            yield from _recursive_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _recursive_keys(item)


def _assert_compact_v29a(value):
    overlap = FORBIDDEN_KEYS_V29A & set(_recursive_keys(value))
    if overlap:
        raise RuntimeError(
            f"V29A failure evidence contains forbidden keys: {sorted(overlap)}"
        )


def build_failure_evidence_v29a(
    attempt_path: Path,
    report_path: Path,
    output_directory: Path,
    runtime_path: Path,
):
    """Bind the compact evidence to the exact durable failed attempt."""
    attempt_path = Path(attempt_path).resolve()
    report_path = Path(report_path).resolve()
    output_directory = Path(output_directory).resolve()
    runtime_path = Path(runtime_path).resolve()
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    failure = attempt.get("failure", {})
    audit = attempt.get("live_cpu_disk_audit", {})
    runtime_source = runtime_path.read_text(encoding="utf-8")
    if (
        file_sha256(attempt_path) != EXPECTED_ATTEMPT_FILE_SHA256_V29A
        or attempt.get("content_sha256_before_self_field")
        != EXPECTED_ATTEMPT_CONTENT_SHA256_V29A
        or prereg.canonical_sha256(_without_self(attempt))
        != EXPECTED_ATTEMPT_CONTENT_SHA256_V29A
        or attempt.get("schema") != "vllm-moe-fp8-tuning-attempt-v29a"
        or attempt.get("status") != "failed"
        or attempt.get("phase")
        != "after_worker_cleanup_or_finalization_failure"
        or set(failure) != {
            "exception_class", "message_sha256",
            "raw_message_or_traceback_persisted",
        }
        or failure.get("exception_class") != EXPECTED_EXCEPTION_CLASS_V29A
        or failure.get("message_sha256") != EXPECTED_MESSAGE_SHA256_V29A
        or failure.get("raw_message_or_traceback_persisted") is not False
        or attempt.get("preregistration", {}).get("file_sha256")
        != EXPECTED_PREREG_FILE_SHA256_V29A
        or attempt.get("preregistration", {}).get("content_sha256")
        != EXPECTED_PREREG_CONTENT_SHA256_V29A
        or attempt.get("implementation_bundle_sha256")
        != EXPECTED_IMPLEMENTATION_BUNDLE_SHA256_V29A
        or attempt.get("recipe_content_sha256")
        != EXPECTED_RECIPE_CONTENT_SHA256_V29A
        or attempt.get("runtime_environment_certificate_sha256")
        != EXPECTED_RUNTIME_ENVIRONMENT_CERTIFICATE_SHA256_V29A
        or audit.get("content_sha256_before_self_field")
        != EXPECTED_LIVE_AUDIT_CONTENT_SHA256_V29A
        or prereg.canonical_sha256(_without_self(audit))
        != EXPECTED_LIVE_AUDIT_CONTENT_SHA256_V29A
        or attempt.get("prelaunch_idle_certificate_sha256")
        != EXPECTED_PRELAUNCH_IDLE_CERTIFICATE_SHA256_V29A
        or attempt.get("final_idle_certificate_sha256")
        != EXPECTED_FINAL_IDLE_CERTIFICATE_SHA256_V29A
        or attempt.get("selected_table_written") is not False
        or attempt.get(
            "training_model_update_checkpoint_adoption_evaluation_or_dataset_promotion_applied"
        ) is not False
        or attempt.get(
            "dataset_train_evaluation_validation_heldout_ood_or_benchmark_surface_opened"
        ) is not False
        or file_sha256(runtime_path) != EXPECTED_RUNTIME_FILE_SHA256_V29A
        or runtime_source.count("ray.init(num_gpus=4, include_dashboard=False)")
        != 1
        or runtime_source.count("from ray.util.state import get_actor") != 1
        or runtime_source.count("state = get_actor(worker._actor_id.hex())") != 1
        or runtime_source.index(
            "actor_pids_by_gpu = _official_actor_pids_v29a(ray, workers)"
        )
        >= runtime_source.index("futures = [")
        or not output_directory.is_dir()
        or any(output_directory.iterdir())
        or report_path.exists()
    ):
        raise RuntimeError("V29A infrastructure failure evidence changed")

    evidence = _seal({
        "schema": "vllm-moe-fp8-tuning-infrastructure-failure-evidence-v29a",
        "authority": {
            "infrastructure_failure_diagnosis_only": True,
            "retry_preregistration_allowed": True,
            "selected_table_adoption_allowed": False,
            "training_or_model_update_allowed": False,
            "evaluation_or_dataset_access_allowed": False,
        },
        "failed_attempt": {
            "relative_path": ATTEMPT_RELATIVE_PATH_V29A,
            "file_sha256": EXPECTED_ATTEMPT_FILE_SHA256_V29A,
            "content_sha256": EXPECTED_ATTEMPT_CONTENT_SHA256_V29A,
            "status": "failed",
            "phase": "after_worker_cleanup_or_finalization_failure",
            "failure_integrity": {
                "exception_class": EXPECTED_EXCEPTION_CLASS_V29A,
                "message_sha256": EXPECTED_MESSAGE_SHA256_V29A,
                "raw_message_or_traceback_persisted": False,
            },
            "preregistration_file_sha256": EXPECTED_PREREG_FILE_SHA256_V29A,
            "preregistration_content_sha256": (
                EXPECTED_PREREG_CONTENT_SHA256_V29A
            ),
            "implementation_bundle_sha256": (
                EXPECTED_IMPLEMENTATION_BUNDLE_SHA256_V29A
            ),
            "recipe_content_sha256": EXPECTED_RECIPE_CONTENT_SHA256_V29A,
            "runtime_environment_certificate_sha256": (
                EXPECTED_RUNTIME_ENVIRONMENT_CERTIFICATE_SHA256_V29A
            ),
            "live_cpu_disk_audit_content_sha256": (
                EXPECTED_LIVE_AUDIT_CONTENT_SHA256_V29A
            ),
            "prelaunch_idle_certificate_sha256": (
                EXPECTED_PRELAUNCH_IDLE_CERTIFICATE_SHA256_V29A
            ),
            "final_idle_certificate_sha256": (
                EXPECTED_FINAL_IDLE_CERTIFICATE_SHA256_V29A
            ),
        },
        "failure_boundary": {
            "runtime_relative_path": RUNTIME_RELATIVE_PATH_V29A,
            "runtime_file_sha256": EXPECTED_RUNTIME_FILE_SHA256_V29A,
            "ray_dashboard_disabled": True,
            "dashboard_dependent_state_api_used_for_actor_pid": True,
            "failure_before_official_tune_future_submission": True,
            "selected_table_written": False,
            "selected_output_directory_empty": True,
            "compact_report_relative_path": REPORT_RELATIVE_PATH_V29A,
            "compact_report_absent": True,
            "all_four_gpus_finally_idle": True,
            "cause_classification": (
                "dashboard_dependent_ray_state_api_unavailable_before_tuning"
            ),
        },
        "closed_surfaces": {
            "official_tuner_search_reached": False,
            "training_or_model_update_applied": False,
            "selected_table_adopted": False,
            "evaluation_or_dataset_surface_opened": False,
        },
        "raw_failure_payload_persisted": False,
    })
    _assert_compact_v29a(evidence)
    return evidence


def _parser_v29a():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--runtime-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH_V29A)
    return parser


def main(argv=None):
    args = _parser_v29a().parse_args(argv)
    value = build_failure_evidence_v29a(
        args.attempt_path,
        args.report_path,
        args.output_directory,
        args.runtime_path,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "file_sha256": file_sha256(args.output),
        "content_sha256": value["content_sha256_before_self_field"],
    }, sort_keys=True))
    return value


if __name__ == "__main__":
    main()
