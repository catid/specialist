#!/usr/bin/env python3
"""Bind V23A R3's failed train-only insertion gate as compact evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import train_eggroll_es_insertion_stability_v23a as mechanics_v23a


ROOT = Path(__file__).resolve().parent
RUNS = (ROOT / "experiments/eggroll_es_hpo/runs").resolve()
ATTEMPT_PATH = RUNS / (
    ".insertion_location_stability_v23a_authoritative_raw_seed_retry_r3."
    "launch_attempt.json"
)
REPORT_PATH = (
    RUNS / "insertion_location_stability_v23a_authoritative_raw_seed_retry_r3"
    / "insertion_location_stability_v23a_seed_retry_r3.json"
)
OUTPUT_PATH = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V23A_INSERTION_NEGATIVE_AGGREGATE_EVIDENCE_R3.json"
)

ATTEMPT_FILE_SHA256 = "779d3e6113db1353bfa74a768348f1cdaf36d8735d8c24d70c130edd884c41d8"
ATTEMPT_CONTENT_SHA256 = "e53b5b4c0ddeee1757476cf1513753efac42dd60cb9334b1f0339276e1238e87"
REPORT_FILE_SHA256 = "fc62a4d4a9fae8e72eedad2881730061ff45f7218d323a25babd0ce2ad344d14"
REPORT_CONTENT_SHA256 = "852fb08c6b4709d19fc990e75fe66c5e807c67ca224a13610aa70246a04d815f"
SOURCE_COMMIT = "89e1563da58e15ab51d923025efb0017cb95ee72"
SOURCE_BUNDLE_CONTENT_SHA256 = "8fb0bbe343d1ce1b36e2f2eedefdc4f0bd059c5e791e66a0c7af4704478b6c8f"
IMPLEMENTATION_BUNDLE_SHA256 = "40ca3eea1948a9717da0180a65781694505f0db9fe61f0f13cf4c61e490f1755"
RECIPE_CONTENT_SHA256 = "b18e3e97736c78e229c2a26e918d3bc9b2e24ffb22f95f0934136b6dea4c6c4d"
RUNTIME_ENVIRONMENT_CONTENT_SHA256 = (
    "7acc87ffb6a50682f19309ea3059823d0fc96dfc459e30cb66d64e157a76f52a"
)

EXPECTED_LOCATION_RESULTS = {
    "insert_back_e005": {
        "all_sixteen_familywise_lcbs_passed": False,
        "base_runtime_integrity_passed": True,
        "candidate_runtime_integrity_passed": True,
        "gradient_pass_count": 0,
        "location_passed": False,
        "mean_familywise_lcb": -0.2644736778580434,
        "minimum_familywise_lcb": -0.5675335440721672,
        "reference_pass_count": 0,
    },
    "insert_front_e005": {
        "all_sixteen_familywise_lcbs_passed": False,
        "base_runtime_integrity_passed": True,
        "candidate_runtime_integrity_passed": True,
        "gradient_pass_count": 0,
        "location_passed": False,
        "mean_familywise_lcb": -0.5032856776922238,
        "minimum_familywise_lcb": -1.0654551915109935,
        "reference_pass_count": 0,
    },
    "insert_middle_e005": {
        "all_sixteen_familywise_lcbs_passed": False,
        "base_runtime_integrity_passed": True,
        "candidate_runtime_integrity_passed": True,
        "gradient_pass_count": 0,
        "location_passed": False,
        "mean_familywise_lcb": -0.5082781905408548,
        "minimum_familywise_lcb": -0.9314083067362329,
        "reference_pass_count": 0,
    },
}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value):
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def validate_completed_v23a_r3():
    if (
        file_sha256(ATTEMPT_PATH) != ATTEMPT_FILE_SHA256
        or file_sha256(REPORT_PATH) != REPORT_FILE_SHA256
    ):
        raise RuntimeError("V23A R3 attempt or report file identity changed")
    attempt = json.loads(ATTEMPT_PATH.read_text())
    report = json.loads(REPORT_PATH.read_text())
    if (
        attempt.get("schema")
        != "eggroll-es-durable-launch-attempt-v23a-seed-retry-r3"
        or attempt.get("status") != "complete"
        or attempt.get("phase") != "after_cleanup_and_compact_retry_report"
        or attempt.get("content_sha256_before_self_field")
        != ATTEMPT_CONTENT_SHA256
        or canonical_sha256(without_self(attempt)) != ATTEMPT_CONTENT_SHA256
        or attempt.get("model_update_applied") is not False
        or attempt.get("nontrain_surface_opened") is not False
        or report.get("schema")
        != "eggroll-es-insertion-location-stability-report-v23a-seed-retry-r3"
        or report.get("content_sha256_before_self_field")
        != REPORT_CONTENT_SHA256
        or canonical_sha256(without_self(report)) != REPORT_CONTENT_SHA256
        or report.get("model_update_applied") is not False
        or report.get("nontrain_surface_opened") is not False
        or report.get("direct_action_taken") is not False
    ):
        raise RuntimeError("V23A R3 completion or closure semantics changed")
    if attempt.get("report_binding") != {
        "path": str(REPORT_PATH),
        "file_sha256": REPORT_FILE_SHA256,
        "content_sha256": REPORT_CONTENT_SHA256,
    }:
        raise RuntimeError("V23A R3 report binding changed")
    source = attempt.get("source_provenance", {})
    implementation = report.get("implementation", {})
    if (
        source.get("git_head") != SOURCE_COMMIT
        or source.get("content_sha256_before_self_field")
        != SOURCE_BUNDLE_CONTENT_SHA256
        or canonical_sha256(without_self(source))
        != SOURCE_BUNDLE_CONTENT_SHA256
        or source.get("implementation_bundle_sha256")
        != IMPLEMENTATION_BUNDLE_SHA256
        or implementation.get("bundle_sha256")
        != IMPLEMENTATION_BUNDLE_SHA256
        or canonical_sha256(implementation.get("files"))
        != IMPLEMENTATION_BUNDLE_SHA256
    ):
        raise RuntimeError("V23A R3 committed implementation identity changed")
    for item in source.get("files", {}).values():
        raw = subprocess.check_output(
            ["git", "show", f"{SOURCE_COMMIT}:{item['relative_path']}"],
            cwd=ROOT,
        )
        if hashlib.sha256(raw).hexdigest() != item["file_sha256"]:
            raise RuntimeError("V23A R3 committed source file changed")

    recipe = report.get("recipe", {})
    environment = report.get("runtime_environment_certificate", {})
    if (
        recipe.get("content_sha256_before_self_field")
        != RECIPE_CONTENT_SHA256
        or canonical_sha256(without_self(recipe)) != RECIPE_CONTENT_SHA256
        or environment.get("content_sha256_before_self_field")
        != RUNTIME_ENVIRONMENT_CONTENT_SHA256
        or canonical_sha256(without_self(environment))
        != RUNTIME_ENVIRONMENT_CONTENT_SHA256
        or environment.get("completed_before_attempt_claim") is not True
        or environment.get("cuda_device_count") != 4
        or environment.get("dataset_or_evaluation_surface_opened") is not False
    ):
        raise RuntimeError("V23A R3 recipe or runtime environment changed")

    estimator = report.get("estimator", {})
    gate = report.get("gate", {})
    if (
        mechanics_v23a.evaluate_gate_v23a(estimator) != gate
        or gate.get("location_results") != EXPECTED_LOCATION_RESULTS
        or gate.get("passing_location_count") != 0
        or gate.get("selected_location_for_confirmation") is not None
        or gate.get("compatibility_gate_passed") is not False
        or gate.get("evaluation_authorized") is not False
        or gate.get("direct_model_update_authorized") is not False
        or gate.get("checkpoint_write_authorized") is not False
        or gate.get("dataset_promotion_authorized") is not False
        or gate.get("decision") != "retain_v13_base_middle_late_recipe"
    ):
        raise RuntimeError("V23A R3 train-only gate changed")

    integrity = estimator.get("runtime_integrity", {})
    expected_arms = {
        "base_middle_late", "insert_front_e005", "insert_middle_e005",
        "insert_back_e005",
    }
    if set(integrity) != expected_arms or not all(
        value.get("all_integrity_audits_passed") is True
        and value.get("all_sixty_four_signed_waves_complete") is True
        and value.get("exact_selected_reference_restored_every_signed_wave")
        is True
        and value.get("unselected_origin_unchanged") is True
        and value.get("pre_post_unperturbed_reference_probe_equal") is True
        for value in integrity.values()
    ):
        raise RuntimeError("V23A R3 per-arm runtime integrity changed")

    audit = report.get("runtime_audit", {})
    seed = audit.get("seed_domain_integrity_r1", {})
    guard = audit.get("matched_full_context_guard_r3", {})
    exact_fields = (
        "all_dense_commitments_exact", "all_four_score_arrays_exact",
        "all_probe_commitments_exact",
    )
    if (
        audit.get("signed_wave_count") != 64
        or audit.get("dense_result_commitment_count") != 1280
        or audit.get("per_unit_scores_persisted") is not False
        or audit.get("bootstrap_replicates_persisted") is not False
        or seed.get("all_four_workers_identical") is not True
        or seed.get("certificate_completed_before_reference_scoring") is not True
        or guard.get("pre_population_guard_completed_before_first_perturbation")
        is not True
        or guard.get("post_population_guard_completed_after_full_weight_audit")
        is not True
        or not all(
            guard["pre_population_exact_comparison"].get(key) is True
            for key in exact_fields
        )
        or not all(
            guard["post_population_exact_comparison"].get(key) is True
            for key in exact_fields
        )
        or guard.get("raw_scores_or_responses_persisted") is not False
        or estimator.get("unit_scores_persisted") is not False
        or estimator.get("bootstrap_draws_or_replicates_persisted") is not False
        or estimator.get("persisted_response_vectors_or_row_content") is not False
    ):
        raise RuntimeError("V23A R3 compact runtime audit changed")
    return attempt, report


def build_evidence():
    _attempt, report = validate_completed_v23a_r3()
    audit = report["runtime_audit"]
    gate = report["gate"]
    evidence = {
        "schema": "eggroll-es-v23a-insertion-negative-aggregate-evidence-r3",
        "attempt": {
            "path": str(ATTEMPT_PATH),
            "file_sha256": ATTEMPT_FILE_SHA256,
            "content_sha256": ATTEMPT_CONTENT_SHA256,
            "source_commit": SOURCE_COMMIT,
            "source_bundle_content_sha256": SOURCE_BUNDLE_CONTENT_SHA256,
        },
        "report": {
            "path": str(REPORT_PATH),
            "file_sha256": REPORT_FILE_SHA256,
            "content_sha256": REPORT_CONTENT_SHA256,
            "implementation_bundle_sha256": IMPLEMENTATION_BUNDLE_SHA256,
            "recipe_content_sha256": RECIPE_CONTENT_SHA256,
        },
        "runtime_integrity": {
            "all_four_arms_passed": True,
            "all_64_signed_waves_completed": True,
            "exact_restore_every_arm_every_wave": True,
            "unselected_origin_unchanged": True,
            "population_boundary_audit_completed": True,
            "seed_domain_certificate_all_four_workers_identical": True,
            "matched_context_reference_repeat_exact_before_population": True,
            "matched_context_reference_repeat_exact_after_population": True,
            "full_context_guard_phase_count": 3,
            "signed_wave_count": audit["signed_wave_count"],
            "requests_all_engines_all_signed_waves": audit[
                "requests_all_engines_all_signed_waves"
            ],
        },
        "aggregate_gate": {
            "family_hypothesis_count": gate["family_hypothesis_count"],
            "passing_location_count": gate["passing_location_count"],
            "selected_location_for_confirmation": gate[
                "selected_location_for_confirmation"
            ],
            "location_results": gate["location_results"],
            "decision": gate["decision"],
        },
        "closure": {
            "confirmation_authorized": False,
            "evaluation_authorized": False,
            "checkpoint_write_authorized": False,
            "model_update_authorized": False,
            "dataset_promotion_authorized": False,
            "retained_recipe": "v13_base_middle_late",
            "closed_hypotheses": [
                "insert_front_e005_layers_4_7",
                "insert_middle_e005_layers_20_23",
                "insert_back_e005_layers_40_43",
            ],
        },
        "scope_note": (
            "This closes only the three preregistered epsilon-0.05 insertion "
            "locations; it does not generalize to other insertion strengths, "
            "ranges, duplication counts, or backend representations."
        ),
        "contains_response_vectors_unit_scores_bootstrap_draws_or_replicates": False,
        "contains_dataset_rows_questions_answers_or_document_content": False,
        "contains_validation_ood_heldout_or_benchmark_content": False,
    }
    evidence["content_sha256_before_self_field"] = canonical_sha256(evidence)
    return evidence


def exclusive_write(path, value):
    path = Path(path).resolve()
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise ValueError(f"output already exists: {path}") from error
    with os.fdopen(descriptor, "wb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args(argv)
    exclusive_write(args.output, build_evidence())


if __name__ == "__main__":
    main()
