#!/usr/bin/env python3
"""Seal the immutable R2 matched-context probe failure diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import eggroll_es_insertion_stability_preregistration_v23a as prereg_v23a


ROOT = Path(__file__).resolve().parent
ATTEMPT_RELATIVE_PATH_R3 = (
    "experiments/eggroll_es_hpo/runs/"
    ".insertion_location_stability_v23a_authoritative_raw_seed_retry_r2."
    "launch_attempt.json"
)
REPORT_RELATIVE_PATH_R3 = (
    "experiments/eggroll_es_hpo/runs/"
    "insertion_location_stability_v23a_authoritative_raw_seed_retry_r2/"
    "insertion_location_stability_v23a_seed_retry_r2.json"
)
OUTPUT_PATH_R3 = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V23A_R2_PROBE_CONTEXT_FAILURE_EVIDENCE_R3.json"
)
EXPECTED_ATTEMPT_FILE_SHA256_R3 = (
    "fe66592ad056b256d24dc05da15288c52e5100c1bc0b6fc1a1bbad85b14f1251"
)
EXPECTED_ATTEMPT_CONTENT_SHA256_R3 = (
    "3bab30a38e5331fafc8c4a7ad727572b142126f73cf30052a7c80b6b05118997"
)
EXPECTED_SOURCE_HEAD_R3 = "19cb1280da88a2896177c245ce32ae063b9e02fd"
EXPECTED_IMPLEMENTATION_SHA256_R3 = (
    "face2b9b781a8ac0c079fde0d09bae323afcd240ac154855b0d092d8970cbeb4"
)
EXPECTED_SOURCE_CONTENT_SHA256_R3 = (
    "4a58a3f1d54b020b68e1e488652fa0162d2d2aa7e03bcd283964433030105eb1"
)
EXPECTED_RECIPE_CONTENT_SHA256_R3 = (
    "1ee204ab7d7b4fbe42e8f80fd3562eda28a94da211aec601d5470cbb35f89b7a"
)
EXPECTED_RUNTIME_ENVIRONMENT_CONTENT_SHA256_R3 = (
    "7acc87ffb6a50682f19309ea3059823d0fc96dfc459e30cb66d64e157a76f52a"
)
EXPECTED_RUNTIME_DRIVER_SHA256_R3 = (
    "781223f28a0b37506734684f410e07074129f545363fb7eb202de719a18955b4"
)
EXPECTED_FAILURE_TYPE_R3 = "RuntimeError"
EXPECTED_FAILURE_MESSAGE_R3 = "v23a pre/post unperturbed reference probes drifted"
FORBIDDEN_KEYS_R3 = {
    "traceback", "message", "question", "questions", "answer", "answers",
    "prompt", "prompts", "responses", "row_content", "token_ids",
    "model_repr", "unit_scores", "score_arrays",
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
    overlap = FORBIDDEN_KEYS_R3 & set(_recursive_keys(value))
    if overlap:
        raise RuntimeError(f"v23a-r3 evidence contains forbidden keys: {sorted(overlap)}")


def _validate_bound_runtime_control_flow(attempt):
    source_entry = attempt.get("source_provenance", {}).get("files", {}).get(
        "runtime_driver_v23a", {}
    )
    runtime_path = ROOT / source_entry.get("relative_path", "missing")
    if (
        source_entry.get("file_sha256") != EXPECTED_RUNTIME_DRIVER_SHA256_R3
        or not runtime_path.is_file()
        or file_sha256(runtime_path) != EXPECTED_RUNTIME_DRIVER_SHA256_R3
    ):
        raise RuntimeError("V23A R2 runtime source binding changed")
    source = runtime_path.read_text(encoding="utf-8")
    ordered = (
        "reference_batches = self._generate_all_v23a(requests)",
        "reference_scores, reference_commitments, pre_probes = self._score_batches_v23a(",
        "restore_hashes.append(self._restore_verify_v23a())",
        "boundary = self._boundary_audit_v23a()",
        "first_request = [requests[0]]",
        "post_batches = self._generate_all_v23a(first_request)",
        "if pre_probes != post_probes:",
        'raise RuntimeError("v23a pre/post unperturbed reference probes drifted")',
    )
    offsets = [source.find(token) for token in ordered]
    if any(offset < 0 for offset in offsets) or offsets != sorted(offsets):
        raise RuntimeError("V23A R2 probe failure control flow changed")
    if (
        'if cursor != 280:' not in source
        or 'first_request = [requests[0]]' not in source
        or '[first_panel["dense_items"][0]], [batches[rank][0]]' not in source
    ):
        raise RuntimeError("V23A R2 probe batch-size diagnosis changed")


def build_probe_context_failure_evidence_r3(attempt_path: Path, report_path: Path):
    attempt_path = Path(attempt_path).resolve()
    report_path = Path(report_path).resolve()
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    failure = attempt.get("failure")
    source = attempt.get("source_provenance", {})
    recipe = attempt.get("recipe", {})
    runtime_environment = attempt.get("runtime_environment_certificate", {})
    if (
        file_sha256(attempt_path) != EXPECTED_ATTEMPT_FILE_SHA256_R3
        or attempt.get("content_sha256_before_self_field")
        != EXPECTED_ATTEMPT_CONTENT_SHA256_R3
        or attempt.get("content_sha256_before_self_field")
        != prereg_v23a.canonical_sha256(_without_self(attempt))
        or attempt.get("schema")
        != "eggroll-es-durable-launch-attempt-v23a-seed-retry-r2"
        or attempt.get("status") != "failed"
        or attempt.get("phase") != "inside_v23a_r2_train_only_runtime"
        or attempt.get("model_update_applied") is not False
        or attempt.get("nontrain_surface_opened") is not False
        or not isinstance(failure, dict)
        or failure.get("type") != EXPECTED_FAILURE_TYPE_R3
        or failure.get("message") != EXPECTED_FAILURE_MESSAGE_R3
        or source.get("git_head") != EXPECTED_SOURCE_HEAD_R3
        or source.get("implementation_bundle_sha256")
        != EXPECTED_IMPLEMENTATION_SHA256_R3
        or source.get("content_sha256_before_self_field")
        != EXPECTED_SOURCE_CONTENT_SHA256_R3
        or recipe.get("implementation_bundle_sha256")
        != EXPECTED_IMPLEMENTATION_SHA256_R3
        or recipe.get("content_sha256_before_self_field")
        != EXPECTED_RECIPE_CONTENT_SHA256_R3
        or runtime_environment.get("content_sha256_before_self_field")
        != EXPECTED_RUNTIME_ENVIRONMENT_CONTENT_SHA256_R3
        or report_path.exists()
    ):
        raise RuntimeError("V23A R2 probe context failure evidence changed")
    _validate_bound_runtime_control_flow(attempt)

    evidence = {
        "schema": "eggroll-es-v23a-r2-probe-context-failure-evidence-r3",
        "authority": {
            "train_only_failure_diagnosis": True,
            "model_selection_allowed": False,
            "evaluation_opened": False,
            "dataset_content_inspected": False,
        },
        "failed_attempt": {
            "relative_path": ATTEMPT_RELATIVE_PATH_R3,
            "file_sha256": EXPECTED_ATTEMPT_FILE_SHA256_R3,
            "content_sha256": EXPECTED_ATTEMPT_CONTENT_SHA256_R3,
            "source_git_head": EXPECTED_SOURCE_HEAD_R3,
            "source_implementation_bundle_sha256": EXPECTED_IMPLEMENTATION_SHA256_R3,
            "source_provenance_content_sha256": EXPECTED_SOURCE_CONTENT_SHA256_R3,
            "recipe_content_sha256": EXPECTED_RECIPE_CONTENT_SHA256_R3,
            "runtime_environment_content_sha256": (
                EXPECTED_RUNTIME_ENVIRONMENT_CONTENT_SHA256_R3
            ),
            "status": "failed",
            "phase": "inside_v23a_r2_train_only_runtime",
            "failure_type": EXPECTED_FAILURE_TYPE_R3,
            "failure_guard": "pre_post_unperturbed_reference_probe_exact_identity",
            "model_update_applied": False,
            "nontrain_surface_opened": False,
            "compact_report_relative_path": REPORT_RELATIVE_PATH_R3,
            "compact_report_absent": True,
        },
        "proven_control_flow_boundary": {
            "all_64_signed_waves_completed": True,
            "all_64_selected_restore_verifications_completed": True,
            "full_selected_and_unselected_population_boundary_audit_completed": True,
            "failure_occurred_after_full_weight_identity_audit": True,
            "compact_estimator_and_gate_construction_reached": False,
        },
        "probe_context_diagnosis": {
            "pre_probe_source": "request_0_scored_inside_280_request_batch",
            "post_probe_source": "same_request_0_scored_inside_1_request_batch",
            "request_identity_same": True,
            "generation_batch_shape_and_order_same": False,
            "exact_dense_hashes_across_mismatched_batch_contexts_are_not_a_weight_identity_test": True,
            "smallest_repair": "repeat_exact_full_280_request_context_before_and_after_population",
        },
        "r2_attempt_preserved_immutable": True,
        "traceback_or_model_repr_persisted": False,
        "row_response_or_score_content_persisted": False,
    }
    _assert_compact(evidence)
    return _seal(evidence)


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH_R3)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    evidence = build_probe_context_failure_evidence_r3(
        args.attempt_path, args.report_path
    )
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
