#!/usr/bin/env python3
"""Bind V14b's failed k=2 gate into compact aggregate evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import eggroll_es_hierarchical_preregistration_v14b as prereg_v14b
import run_eggroll_es_hierarchical_train_panels_v14b as driver_v14b
import train_eggroll_es_specialist_anchor_v14b as anchor_v14b


ROOT = Path(__file__).resolve().parent
ATTEMPT_PATH_V15 = driver_v14b._attempt_path_v14b().resolve()
REPORT_PATH_V15 = (
    driver_v14b.FROZEN_OUTPUT_DIRECTORY_V14B
    / driver_v14b.EXPERIMENT_NAME_V14B
    / driver_v14b.REPORT_NAME_V14B
).resolve()
OUTPUT_PATH_V15 = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V14B_K2_DISTINCT_ROW_NEGATIVE_AGGREGATE_EVIDENCE_V15.json"
).resolve()
ATTEMPT_FILE_SHA256_V15 = (
    "7c0a795442c277d51c8600c3c09eb7945bf26f70e476001b194e4650a927eaae"
)
ATTEMPT_CONTENT_SHA256_V15 = (
    "9bd36a4fbd0e3626593265885f6734ea91639d65d4db5a61c1b305da6463957c"
)
REPORT_FILE_SHA256_V15 = (
    "fef74086c0d1494593372e29325af688afa1f315c7f6ee1e2cb0a59150c84b9c"
)
REPORT_CONTENT_SHA256_V15 = (
    "b29886238418ccf35e9bc8b84dd8d25a2ebd1eab6e91319c72c02d22f092a5b5"
)
SOURCE_COMMIT_V15 = "564fb3008a5b6402b38d4c18a0264d0a105f31a4"
SOURCE_BUNDLE_CONTENT_SHA256_V15 = (
    "0e77bf89098f0a80608c8225cb9316b2b6c6fde2515ef34b3138e8be2381a134"
)
IMPLEMENTATION_BUNDLE_SHA256_V15 = (
    "86db8454d08201c8cffa3b602c16058ea769409d4c9e4164b4c2321f5a6aa9ff"
)
RECIPE_CONTENT_SHA256_V15 = (
    "41a56112ca6796a58240f41befde96a132c03930fa7fce83930bc99f2995554f"
)
DIAGNOSTIC_CONTENT_SHA256_V15 = (
    "2b9faabd62d89367e2456b553fe237c0f59b75dddfae18c9ceeb6ac9a221361c"
)
INTEGRITY_CONTENT_SHA256_V15 = (
    "011cdf23c24d5aabc8c77c704c4be7ae3078ddf56446848cb16ed71f31c83101"
)
CANDIDATE_CONTENT_SHA256_V15 = (
    "9c27b5c4ad4b54ce8cbd151474e7ec1c393936c76a28225dc32375817fc49e39"
)
AGGREGATE_COEFFICIENT_SHA256_V15 = (
    "72d6ffe4031a3e4dbdf4a01753054e2effdccb00f67f33de7cd7e2849d6a61e3"
)
EXPECTED_STABILITY_V15 = {
    "matched56_pairwise_cosine": {
        "count": 3, "median": 0.59336490603814,
        "worst": 0.3282531198822195,
    },
    "matched56_pairwise_sign_agreement": {
        "count": 3, "median": 0.6875, "worst": 0.59375,
    },
    "full_to_matched56_optimization_cosine": {
        "count": 3, "median": 0.7526394756957133,
        "worst": 0.7040159025964595,
    },
    "full_to_matched56_optimization_sign_agreement": {
        "count": 3, "median": 0.8125, "worst": 0.71875,
    },
    "crossfit_complement_to_screen_cosine": {
        "count": 2, "median": 0.4896964998727882,
        "worst": 0.4829485162505813,
    },
    "crossfit_complement_to_screen_sign_agreement": {
        "count": 2, "median": 0.796875, "worst": 0.78125,
    },
}
EXPECTED_FAILED_RULES_V15 = [
    "matched56_pairwise_cosine.worst_passed",
    "full_to_matched56_optimization_cosine.median_passed",
    "full_to_matched56_optimization_cosine.worst_passed",
    "full_to_matched56_optimization_sign_agreement.worst_passed",
]


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    return driver_v14b._canonical(value)


def validate_v14b_negative_v15(
    attempt_path=ATTEMPT_PATH_V15, report_path=REPORT_PATH_V15,
):
    attempt_path = Path(attempt_path).resolve()
    report_path = Path(report_path).resolve()
    if attempt_path != ATTEMPT_PATH_V15 or report_path != REPORT_PATH_V15:
        raise RuntimeError("v15 requires canonical V14b aggregate paths")
    if (
        _file_sha256(attempt_path) != ATTEMPT_FILE_SHA256_V15
        or _file_sha256(report_path) != REPORT_FILE_SHA256_V15
    ):
        raise RuntimeError("v15 V14b aggregate file identity changed")
    attempt = json.loads(attempt_path.read_text())
    report = json.loads(report_path.read_text())
    if (
        attempt.get("schema") != "eggroll-es-durable-launch-attempt-v14b"
        or attempt.get("status") != "complete"
        or attempt.get("phase") != "after_trainer_cleanup_and_report"
        or attempt.get("experiment_name") != prereg_v14b.EXPERIMENT_NAME_V14B
        or attempt.get("target_alpha_zero_only") is not True
        or attempt.get("model_update_applied") is not False
        or attempt.get("sealed_or_nontrain_surface_opened") is not False
        or attempt.get("report_exists_after_attempt") is not True
        or attempt.get("content_sha256_before_self_field")
        != ATTEMPT_CONTENT_SHA256_V15
        or attempt.get("content_sha256_before_self_field")
        != _canonical({
            key: value for key, value in attempt.items()
            if key != "content_sha256_before_self_field"
        })
        or report.get("schema") != "eggroll-es-k2-alpha-zero-report-v14b"
        or report.get("model_update_applied") is not False
        or report.get("sealed_or_nontrain_surface_opened") is not False
        or report.get("decision")
        != "train_only_k2_estimator_gate_no_model_update"
        or report.get("content_sha256_before_self_field")
        != REPORT_CONTENT_SHA256_V15
        or report.get("content_sha256_before_self_field")
        != _canonical({
            key: value for key, value in report.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v15 V14b completion semantics changed")
    if attempt.get("report_binding") != {
        "path": str(REPORT_PATH_V15),
        "file_sha256": REPORT_FILE_SHA256_V15,
        "content_sha256": REPORT_CONTENT_SHA256_V15,
    }:
        raise RuntimeError("v15 V14b report binding changed")
    source = attempt.get("source_provenance", {})
    implementation = report.get("implementation", {})
    if (
        source.get("git_head") != SOURCE_COMMIT_V15
        or source.get("implementation_bundle_sha256")
        != IMPLEMENTATION_BUNDLE_SHA256_V15
        or source.get("content_sha256_before_self_field")
        != SOURCE_BUNDLE_CONTENT_SHA256_V15
        or source.get("content_sha256_before_self_field")
        != _canonical({
            key: value for key, value in source.items()
            if key != "content_sha256_before_self_field"
        })
        or implementation.get("bundle_sha256")
        != IMPLEMENTATION_BUNDLE_SHA256_V15
        or implementation.get("bundle_sha256")
        != _canonical(implementation.get("files"))
    ):
        raise RuntimeError("v15 V14b committed source changed")
    for item in source.get("files", {}).values():
        raw = subprocess.check_output(
            ["git", "show", f"{SOURCE_COMMIT_V15}:{item['relative_path']}"],
            cwd=ROOT,
        )
        if (
            hashlib.sha256(raw).hexdigest() != item["file_sha256"]
            or _file_sha256(ROOT / item["relative_path"]) != item["file_sha256"]
        ):
            raise RuntimeError("v15 V14b source provenance changed")
    recipe = report.get("recipe", {})
    diagnostic = report.get("diagnostic", {})
    if (
        recipe.get("content_sha256_before_self_field")
        != RECIPE_CONTENT_SHA256_V15
        or recipe.get("content_sha256_before_self_field")
        != _canonical({
            key: value for key, value in recipe.items()
            if key != "content_sha256_before_self_field"
        })
        or recipe.get("alpha") != 0.0
        or recipe.get("model_update_allowed") is not False
        or recipe.get("moe_kernel_environment", {}).get(
            "vllm_tuned_config_folder"
        ) is not None
        or any(
            value is not None for value in recipe.get(
                "moe_kernel_environment", {}
            ).get("backend_selector_environment_overrides", {}).values()
        )
        or diagnostic.get("content_sha256_before_self_field")
        != DIAGNOSTIC_CONTENT_SHA256_V15
        or diagnostic.get("alpha") != 0.0
        or diagnostic.get("model_update_applied") is not False
        or diagnostic.get("applications") != []
    ):
        raise RuntimeError("v15 V14b runtime contract changed")
    anchor_v14b.validate_diagnostic_v14b(diagnostic)
    recomputed = anchor_v14b.analyze_responses_v14b(
        diagnostic["responses"], diagnostic["integrity_audits"],
    )
    if recomputed != diagnostic["analysis"]:
        raise RuntimeError("v15 V14b aggregate analysis changed")
    candidate = recomputed["candidate_summary"]
    gate = recomputed["promotion_gate"]
    if (
        diagnostic["integrity_audits"].get(
            "content_sha256_before_self_field"
        ) != INTEGRITY_CONTENT_SHA256_V15
        or candidate.get("content_sha256_before_self_field")
        != CANDIDATE_CONTENT_SHA256_V15
        or candidate.get("robust_aggregate", {}).get("coefficient_sha256")
        != AGGREGATE_COEFFICIENT_SHA256_V15
        or candidate.get("stability") != EXPECTED_STABILITY_V15
        or candidate.get("all_panel_spreads_nonzero") is not True
        or candidate.get("all_integrity_audits_passed") is not True
        or gate.get("eligible_for_train_only_estimator_confirmation") is not False
        or gate.get("eligible_for_model_update") is not False
        or gate.get("eligible_to_open_evaluation") is not False
        or gate.get("failure_decision")
        != "retain_v13_sampler_and_keep_all_eval_surfaces_closed"
        or gate.get("pass_decision") is not None
    ):
        raise RuntimeError("v15 V14b negative gate changed")
    return attempt, report, candidate, gate


def build_evidence_v15():
    attempt, report, candidate, gate = validate_v14b_negative_v15()
    diagnostic = report["diagnostic"]
    failed_rules = []
    for name, condition in gate["conditions"].items():
        for key, passed in condition.items():
            if key.endswith("passed") and passed is False:
                failed_rules.append(f"{name}.{key}")
    if failed_rules != EXPECTED_FAILED_RULES_V15:
        raise RuntimeError("v15 failed-rule identity changed")
    evidence = {
        "schema": "eggroll-es-v14b-k2-negative-aggregate-evidence-v15",
        "passed": True,
        "selection_surface": "frozen_train_aggregate_outputs_only",
        "contains_response_vectors_or_dense_result_hashes": False,
        "contains_source_rows_questions_answers_or_document_content": False,
        "contains_validation_ood_heldout_or_benchmark_content": False,
        "v14b_attempt": {
            "path": str(ATTEMPT_PATH_V15),
            "file_sha256": ATTEMPT_FILE_SHA256_V15,
            "content_sha256": ATTEMPT_CONTENT_SHA256_V15,
            "source_commit": SOURCE_COMMIT_V15,
            "source_bundle_content_sha256": SOURCE_BUNDLE_CONTENT_SHA256_V15,
        },
        "v14b_report": {
            "path": str(REPORT_PATH_V15),
            "file_sha256": REPORT_FILE_SHA256_V15,
            "content_sha256": REPORT_CONTENT_SHA256_V15,
            "implementation_bundle_sha256": IMPLEMENTATION_BUNDLE_SHA256_V15,
            "recipe_content_sha256": RECIPE_CONTENT_SHA256_V15,
            "diagnostic_content_sha256": DIAGNOSTIC_CONTENT_SHA256_V15,
        },
        "runtime_integrity": {
            "alpha": diagnostic["alpha"],
            "model_update_applied": diagnostic["model_update_applied"],
            "application_count": len(diagnostic["applications"]),
            "integrity_content_sha256": INTEGRITY_CONTENT_SHA256_V15,
            "all_integrity_audits_passed": all(
                value is True
                for key, value in diagnostic["integrity_audits"].items()
                if key != "content_sha256_before_self_field"
            ),
            "hardware": diagnostic["hardware_coverage"],
            "generation": diagnostic["generation_contract"],
            "population_boundary_audit_sha256": diagnostic[
                "population_boundary_audit_v4"
            ]["audit_sha256"],
            "panel_bundle_content_sha256": diagnostic[
                "panel_bundle_content_sha256"
            ],
            "perturbation_basis_sha256": diagnostic[
                "perturbation_basis"
            ]["basis_sha256"],
            "moe_backend": report["recipe"]["moe_kernel_environment"][
                "moe_backend"
            ],
            "tuned_config_folder": None,
        },
        "aggregate_gate": {
            "candidate_content_sha256": CANDIDATE_CONTENT_SHA256_V15,
            "robust_aggregate": candidate["robust_aggregate"],
            "stability": candidate["stability"],
            "baseline": prereg_v14b.BASELINE_STABILITY_V14B,
            "conditions": gate["conditions"],
            "failed_rules": failed_rules,
            "all_eight_spreads_nonzero": candidate[
                "all_panel_spreads_nonzero"
            ],
            "eligible_for_fresh_basis_confirmation": False,
            "eligible_for_model_update": False,
            "eligible_to_open_evaluation": False,
        },
        "decision": {
            "sampler": "retain_v13",
            "fresh_basis_k2_confirmation_authorized": False,
            "evaluation_surface_opened_or_authorized": False,
            "model_update_applied_or_authorized": False,
            "reason": "v14b_failed_its_preregistered_conjunctive_gate",
        },
        "next_step_constraint": (
            "V14b authorizes no follow-up experiment; any later train-only "
            "estimator requires a separate preregistration and cannot reuse "
            "this failed candidate as authority for confirmation, evaluation, "
            "or a model update"
        ),
    }
    evidence["content_sha256_before_self_field"] = _canonical(evidence)
    return evidence


def _exclusive_write(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("v15 compact negative evidence already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default=str(OUTPUT_PATH_V15))
    args = parser.parse_args(argv)
    output = Path(args.output_json).resolve()
    if output != OUTPUT_PATH_V15:
        raise ValueError("v15 compact evidence requires its canonical path")
    evidence = build_evidence_v15()
    _exclusive_write(output, evidence)
    print(json.dumps({
        "output": str(output), "file_sha256": _file_sha256(output),
        "content_sha256": evidence["content_sha256_before_self_field"],
    }, sort_keys=True))
    return evidence


if __name__ == "__main__":
    main()
