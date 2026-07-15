#!/usr/bin/env python3
"""Build compact evidence for the immutable V23A R1 environment failure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import eggroll_es_insertion_stability_preregistration_v23a as prereg_v23a


ROOT = Path(__file__).resolve().parent
ATTEMPT_RELATIVE_PATH_R2 = (
    "experiments/eggroll_es_hpo/runs/"
    ".insertion_location_stability_v23a_authoritative_raw_seed_retry_r1."
    "launch_attempt.json"
)
REPORT_RELATIVE_PATH_R2 = (
    "experiments/eggroll_es_hpo/runs/"
    "insertion_location_stability_v23a_authoritative_raw_seed_retry_r1/"
    "insertion_location_stability_v23a_seed_retry_r1.json"
)
OUTPUT_PATH_R2 = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V23A_R1_RUNTIME_ENVIRONMENT_FAILURE_EVIDENCE_R2.json"
)
EXPECTED_ATTEMPT_FILE_SHA256_R2 = (
    "7862c63ed9346544b3cfb9b944f07282f7fcb129759ca0fa6f05e87a505debd0"
)
EXPECTED_ATTEMPT_CONTENT_SHA256_R2 = (
    "cfc18374362a5d6ba6c959f89b08c7ddd1f0117c50faeb2b177180dfc8094ca9"
)
EXPECTED_SOURCE_HEAD_R2 = "a3878b43d1c6beb1ee0e1476c0973a841c246708"
EXPECTED_SOURCE_IMPLEMENTATION_SHA256_R2 = (
    "856754221a74ca26f5a6b1bb0499c726b5e9453ce8ca305b2c66f7a46cdd9c65"
)
EXPECTED_SOURCE_PROVENANCE_CONTENT_SHA256_R2 = (
    "b317aaa3089929dc1c497037f797a20ef6c51562836f266cc21b258e87853117"
)
EXPECTED_RECIPE_CONTENT_SHA256_R2 = (
    "c49c97a45d73d8373bb3d53f7a891f324a22920ebb86bd99619e8e681da62ea0"
)
EXPECTED_FAILURE_TYPE_R2 = "ModuleNotFoundError"
EXPECTED_FAILURE_MESSAGE_R2 = "No module named 'vllm'"
FORBIDDEN_KEYS_R2 = {
    "traceback", "message", "question", "questions", "answer", "answers",
    "prompt", "prompts", "responses", "row_content", "token_ids",
    "model_repr",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _seal(value):
    value.pop("content_sha256_before_self_field", None)
    value["content_sha256_before_self_field"] = prereg_v23a.canonical_sha256(value)
    return value


def _recursive_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).lower()
            yield from _recursive_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _recursive_keys(item)


def _assert_compact(value):
    overlap = FORBIDDEN_KEYS_R2 & set(_recursive_keys(value))
    if overlap:
        raise RuntimeError(f"v23a-r2 evidence contains forbidden keys: {sorted(overlap)}")


def build_runtime_failure_evidence_r2(attempt_path: Path, report_path: Path):
    """Verify only the compact failure boundary and emit no verbose payload."""
    attempt_path = Path(attempt_path).resolve()
    report_path = Path(report_path).resolve()
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    failure = attempt.get("failure")
    source = attempt.get("source_provenance", {})
    recipe = attempt.get("recipe", {})
    retry_of = attempt.get("retry_of", {})
    if (
        file_sha256(attempt_path) != EXPECTED_ATTEMPT_FILE_SHA256_R2
        or attempt.get("content_sha256_before_self_field")
        != EXPECTED_ATTEMPT_CONTENT_SHA256_R2
        or attempt.get("content_sha256_before_self_field")
        != prereg_v23a.canonical_sha256(_without_self(attempt))
        or attempt.get("schema")
        != "eggroll-es-durable-launch-attempt-v23a-seed-retry-r1"
        or attempt.get("status") != "failed"
        or attempt.get("phase") != "inside_v23a_r1_train_only_runtime"
        or attempt.get("model_update_applied") is not False
        or attempt.get("nontrain_surface_opened") is not False
        or not isinstance(failure, dict)
        or failure.get("type") != EXPECTED_FAILURE_TYPE_R2
        or failure.get("message") != EXPECTED_FAILURE_MESSAGE_R2
        or source.get("git_head") != EXPECTED_SOURCE_HEAD_R2
        or source.get("implementation_bundle_sha256")
        != EXPECTED_SOURCE_IMPLEMENTATION_SHA256_R2
        or source.get("content_sha256_before_self_field")
        != EXPECTED_SOURCE_PROVENANCE_CONTENT_SHA256_R2
        or recipe.get("content_sha256_before_self_field")
        != EXPECTED_RECIPE_CONTENT_SHA256_R2
        or recipe.get("implementation_bundle_sha256")
        != EXPECTED_SOURCE_IMPLEMENTATION_SHA256_R2
        or retry_of.get("failure_evidence_content_sha256")
        != "95d8542cd46c3743f4cb4db99159b485e10798766779b43024572925beeed922"
        or report_path.exists()
    ):
        raise RuntimeError("V23A R1 runtime environment failure evidence changed")

    evidence = {
        "schema": "eggroll-es-v23a-r1-runtime-environment-failure-evidence-r2",
        "authority": {
            "train_only_failure_diagnosis": True,
            "model_selection_allowed": False,
            "evaluation_opened": False,
            "dataset_content_inspected": False,
        },
        "failed_attempt": {
            "relative_path": ATTEMPT_RELATIVE_PATH_R2,
            "file_sha256": EXPECTED_ATTEMPT_FILE_SHA256_R2,
            "content_sha256": EXPECTED_ATTEMPT_CONTENT_SHA256_R2,
            "source_git_head": EXPECTED_SOURCE_HEAD_R2,
            "source_implementation_bundle_sha256": (
                EXPECTED_SOURCE_IMPLEMENTATION_SHA256_R2
            ),
            "source_provenance_content_sha256": (
                EXPECTED_SOURCE_PROVENANCE_CONTENT_SHA256_R2
            ),
            "recipe_content_sha256": EXPECTED_RECIPE_CONTENT_SHA256_R2,
            "status": "failed",
            "phase": "inside_v23a_r1_train_only_runtime",
            "failure_type": EXPECTED_FAILURE_TYPE_R2,
            "missing_dependency": "vllm",
            "model_update_applied": False,
            "nontrain_surface_opened": False,
            "compact_report_relative_path": REPORT_RELATIVE_PATH_R2,
            "compact_report_absent": True,
        },
        "failure_boundary": {
            "failed_before_trainer_instance_creation": True,
            "failed_during_trainer_dependency_import": True,
            "engine_actor_creation_reached": False,
            "panel_loading_reached": False,
            "reference_scoring_reached": False,
            "perturbation_reached": False,
            "gpu_training_work_reached": False,
            "cause_classification": "runtime_interpreter_missing_required_dependency",
        },
        "original_seed_failure_evidence": {
            "file_sha256": (
                "1fae9b7244eff3b532eb3fa93e769b0085728309460804c683a452028d399438"
            ),
            "content_sha256": (
                "95d8542cd46c3743f4cb4db99159b485e10798766779b43024572925beeed922"
            ),
        },
        "traceback_or_model_repr_persisted": False,
        "row_or_response_content_persisted": False,
    }
    _assert_compact(evidence)
    return _seal(evidence)


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH_R2)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    evidence = build_runtime_failure_evidence_r2(args.attempt_path, args.report_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "file_sha256": file_sha256(args.output),
        "content_sha256": evidence["content_sha256_before_self_field"],
    }, sort_keys=True))
    return evidence


if __name__ == "__main__":
    main()
