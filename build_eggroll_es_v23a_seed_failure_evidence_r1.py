#!/usr/bin/env python3
"""Build compact evidence for the immutable V23A seed-domain failure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import eggroll_es_insertion_stability_preregistration_v23a as prereg_v23a


ROOT = Path(__file__).resolve().parent
ATTEMPT_RELATIVE_PATH_R1 = (
    "experiments/eggroll_es_hpo/runs/"
    ".insertion_location_stability_v23a_authoritative_raw.launch_attempt.json"
)
REPORT_RELATIVE_PATH_R1 = (
    "experiments/eggroll_es_hpo/runs/"
    "insertion_location_stability_v23a_authoritative_raw/"
    "insertion_location_stability_v23a.json"
)
OUTPUT_PATH_R1 = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V23A_SEED_DOMAIN_FAILURE_EVIDENCE_R1.json"
)
EXPECTED_ATTEMPT_FILE_SHA256_R1 = (
    "a1a49964eeffd6ce8ba4ea081ab2323e52cf80dc66523b80f9a6bcdf1b1cc8b0"
)
EXPECTED_ATTEMPT_CONTENT_SHA256_R1 = (
    "b6a4c152d1748496e104c5e9fbcfdf71f2d67a8460e86b9cc31160a872f5fc1a"
)
EXPECTED_SOURCE_HEAD_R1 = "cf384e7c438759c794ea9608865f196da7fd702b"
NUMPY_LEGACY_MAX_R1 = 2**32 - 1
TORCH_PREREG_MAX_R1 = 2**63 - 1
FORBIDDEN_KEYS_R1 = {
    "traceback", "message", "question", "questions", "answer", "answers",
    "prompt", "prompts", "responses", "row_content", "token_ids",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _without_self(value):
    return {key: item for key, item in value.items()
            if key != "content_sha256_before_self_field"}


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
    overlap = FORBIDDEN_KEYS_R1 & set(_recursive_keys(value))
    if overlap:
        raise RuntimeError(f"v23a-r1 evidence contains forbidden keys: {sorted(overlap)}")


def build_failure_evidence_r1(attempt_path: Path, report_path: Path):
    attempt_path = Path(attempt_path).resolve()
    report_path = Path(report_path).resolve()
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt_file_sha256 = file_sha256(attempt_path)
    failure = attempt.get("failure")
    failure_type = failure.get("type") if isinstance(failure, dict) else None
    failure_message = failure.get("message") if isinstance(failure, dict) else None
    seeds = prereg_v23a.perturbation_basis_v23a()["direction_seeds"]
    projections = [
        {"full_seed": seed, "numpy_legacy_seed": seed % (2**32)}
        for seed in seeds
    ]
    projected = [item["numpy_legacy_seed"] for item in projections]

    if (
        attempt_file_sha256 != EXPECTED_ATTEMPT_FILE_SHA256_R1
        or attempt.get("content_sha256_before_self_field")
        != EXPECTED_ATTEMPT_CONTENT_SHA256_R1
        or attempt.get("content_sha256_before_self_field")
        != prereg_v23a.canonical_sha256(_without_self(attempt))
        or attempt.get("schema") != "eggroll-es-durable-launch-attempt-v23a"
        or attempt.get("status") != "failed"
        or attempt.get("phase") != "inside_v23a_train_only_runtime"
        or attempt.get("model_update_applied") is not False
        or attempt.get("nontrain_surface_opened") is not False
        or attempt.get("source_provenance", {}).get("git_head")
        != EXPECTED_SOURCE_HEAD_R1
        or failure_type != "RayTaskError(ValueError)"
        or not isinstance(failure_message, str)
        or "np.random.seed(seed)" not in failure_message
        or "Seed must be between 0 and 2**32 - 1" not in failure_message
        or "eggroll_es_worker_v4.py" not in failure_message
        or "perturb_self_weights" not in failure_message
        or report_path.exists()
        or len(seeds) != 32
        or len(set(seeds)) != 32
        or not all(NUMPY_LEGACY_MAX_R1 < seed <= TORCH_PREREG_MAX_R1 for seed in seeds)
        or len(set(projected)) != 32
        or any(seed == 0 for seed in projected)
    ):
        raise RuntimeError("V23A durable seed-domain failure evidence changed")

    evidence = {
        "schema": "eggroll-es-v23a-seed-domain-failure-evidence-r1",
        "authority": {
            "train_only_failure_diagnosis": True,
            "model_selection_allowed": False,
            "evaluation_opened": False,
            "dataset_content_inspected": False,
        },
        "failed_attempt": {
            "relative_path": ATTEMPT_RELATIVE_PATH_R1,
            "file_sha256": attempt_file_sha256,
            "content_sha256": EXPECTED_ATTEMPT_CONTENT_SHA256_R1,
            "source_git_head": EXPECTED_SOURCE_HEAD_R1,
            "status": "failed",
            "phase": "inside_v23a_train_only_runtime",
            "failure_type": failure_type,
            "model_update_applied": False,
            "nontrain_surface_opened": False,
            "compact_report_relative_path": REPORT_RELATIVE_PATH_R1,
            "compact_report_absent": True,
        },
        "seed_domain": {
            "direction_count": len(seeds),
            "direction_seed_list_sha256": prereg_v23a.canonical_sha256(seeds),
            "minimum_direction_seed": min(seeds),
            "maximum_direction_seed": max(seeds),
            "all_direction_seeds_exceed_numpy_legacy_max": True,
            "all_direction_seeds_fit_positive_signed_63_bit": True,
            "numpy_legacy_max_inclusive": NUMPY_LEGACY_MAX_R1,
            "torch_preregistered_max_inclusive": TORCH_PREREG_MAX_R1,
            "numpy_projection_rule": "full_seed modulo 2**32",
            "full_to_numpy_projection_sha256": prereg_v23a.canonical_sha256(projections),
            "numpy_projection_unique_count": len(set(projected)),
            "numpy_projection_minimum": min(projected),
            "numpy_projection_maximum": max(projected),
            "numpy_projection_contains_zero": False,
        },
        "failure_boundary": {
            "failed_in_worker_seed_setup": True,
            "failed_at_numpy_legacy_seed_call": True,
            "first_preregistered_perturbation_must_fail_at_same_boundary": True,
            "selected_parameter_add_reached": False,
            "perturbed_generation_reached": False,
            "unperturbed_reference_generation_may_have_completed": True,
            "inference_basis": (
                "all 32 preregistered direction seeds exceed NumPy's legacy seed domain; "
                "the worker calls seed setup before selected-parameter iteration and the "
                "runtime calls perturbed generation only after perturbation returns"
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
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH_R1)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    evidence = build_failure_evidence_r1(args.attempt_path, args.report_path)
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
