#!/usr/bin/env python3
"""Bind V14a's failed train-only gate into compact aggregate evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import eggroll_es_hierarchical_preregistration_v14a as prereg_v14a
import run_eggroll_es_hierarchical_train_panels_v14a as driver_v14a
import train_eggroll_es_specialist_anchor_v14a as anchor_v14a


ROOT = Path(__file__).resolve().parent
ATTEMPT_PATH_V14B = driver_v14a._attempt_path_v14a().resolve()
REPORT_PATH_V14B = (
    driver_v14a.FROZEN_OUTPUT_DIRECTORY_V14A
    / driver_v14a.EXPERIMENT_NAME_V14A
    / driver_v14a.REPORT_NAME_V14A
).resolve()
OUTPUT_PATH_V14B = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V14A_FULL_FRAME_NEGATIVE_AGGREGATE_EVIDENCE_V14B.json"
).resolve()
ATTEMPT_FILE_SHA256_V14B = (
    "f577c7ceb02605879f6c42cfd26c8672ca97a5d4aa4158c1c59df926dba06e2a"
)
ATTEMPT_CONTENT_SHA256_V14B = (
    "f43f1d4d1f6768841b774ed68efee7897a64b58a9cd121af276550b52ce5bcb2"
)
REPORT_FILE_SHA256_V14B = (
    "7ee4d14f421fda7e12998f35063d3b96982c369f93ac5147ba6b32fc1a25d97c"
)
REPORT_CONTENT_SHA256_V14B = (
    "792254113bccb2962f70b0eeeabb4341475ad7d1c61a7ada60e68c22de68b44c"
)
SOURCE_COMMIT_V14B = "0a3e2ab29b2104374cd680f36ba6bee4584f9a38"
SOURCE_BUNDLE_CONTENT_SHA256_V14B = (
    "5ac19d7c0a03223c7c53e608a618f86ec24b7afce2d178c42d85058996d4009d"
)
IMPLEMENTATION_BUNDLE_SHA256_V14B = (
    "92c682172c1d66d982c6e2cb33310b4661634c0ef0c0697bf4466564f43768a5"
)
RECIPE_CONTENT_SHA256_V14B = (
    "44111d43aec3a8b8401b791bd2c4512db4086d3cdfec16fcccfd3975eaf10cf0"
)
DIAGNOSTIC_CONTENT_SHA256_V14B = (
    "dd5c6f225242af25885a814485723dec3230660a28e0d61f0ec97f51666e1ea6"
)
BOUNDARY_AUDIT_SHA256_V14B = (
    "88796b4dff895a2cf22d6f4615b85ac5501449b7b86402f40651b07ae2403867"
)
CANDIDATE_CONTENT_SHA256_V14B = (
    "7ae2ee84b6d4435e8e0c411f4351ff8dc62834945090a27d9f7dc1c6d65144db"
)
AGGREGATE_COEFFICIENT_SHA256_V14B = (
    "6054cc3c27a53e81ef20bf58cb873fe4baf5ac27def0b1606578fa4674f0245a"
)
EXPECTED_STABILITY_V14B = {
    "matched56_pairwise_cosine": {
        "count": 3, "median": 0.4427854137278768,
        "worst": 0.26764423806475524,
    },
    "matched56_pairwise_sign_agreement": {
        "count": 3, "median": 0.65625, "worst": 0.59375,
    },
    "full_to_matched56_optimization_cosine": {
        "count": 3, "median": 0.7132916285087167,
        "worst": 0.6803342639761624,
    },
    "full_to_matched56_optimization_sign_agreement": {
        "count": 3, "median": 0.75, "worst": 0.75,
    },
    "crossfit_complement_to_screen_cosine": {
        "count": 2, "median": 0.5429050986548454,
        "worst": 0.508915138728717,
    },
    "crossfit_complement_to_screen_sign_agreement": {
        "count": 2, "median": 0.765625, "worst": 0.75,
    },
}


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    return driver_v14a._canonical(value)


def _aggregate_analysis_v14b(responses):
    """Recompute V14a from aggregate vectors without opening source rows."""
    anchor_v14a.validate_responses_v14a(responses)
    inputs = {
        "full_frame": responses["full_frame_sign_scores"],
        **{
            name: responses["matched56_sign_scores"][name]
            for name in prereg_v14a.PANEL_NAMES_V14A
        },
        **{
            f"complement_{name}": responses["complement_sign_scores"][name]
            for name in prereg_v14a.PANEL_NAMES_V14A[3:]
        },
    }
    vectors = {}
    standardization = {}
    for name, sign_scores in inputs.items():
        vectors[name], standardization[name] = anchor_v14a._standardize(
            anchor_v14a._central(sign_scores)
        )
    optimization = list(prereg_v14a.PANEL_NAMES_V14A[:3])
    pair_cosines = []
    pair_signs = []
    for left_index, left in enumerate(optimization):
        for right in optimization[left_index + 1:]:
            pair_cosines.append(anchor_v14a._cosine(
                vectors[left], vectors[right],
            ))
            pair_signs.append(anchor_v14a._sign_agreement(
                vectors[left], vectors[right],
            )["all_coordinate_fraction"])
    full_cosines = [
        anchor_v14a._cosine(vectors["full_frame"], vectors[name])
        for name in optimization
    ]
    full_signs = [
        anchor_v14a._sign_agreement(
            vectors["full_frame"], vectors[name],
        )["all_coordinate_fraction"]
        for name in optimization
    ]
    screen_cosines = []
    screen_signs = []
    for name in prereg_v14a.PANEL_NAMES_V14A[3:]:
        complement = vectors[f"complement_{name}"]
        screen_cosines.append(anchor_v14a._cosine(
            complement, vectors[name],
        ))
        screen_signs.append(anchor_v14a._sign_agreement(
            complement, vectors[name],
        )["all_coordinate_fraction"])
    stability = {
        "matched56_pairwise_cosine": anchor_v14a._metric_summary(
            pair_cosines
        ),
        "matched56_pairwise_sign_agreement": anchor_v14a._metric_summary(
            pair_signs
        ),
        "full_to_matched56_optimization_cosine": (
            anchor_v14a._metric_summary(full_cosines)
        ),
        "full_to_matched56_optimization_sign_agreement": (
            anchor_v14a._metric_summary(full_signs)
        ),
        "crossfit_complement_to_screen_cosine": (
            anchor_v14a._metric_summary(screen_cosines)
        ),
        "crossfit_complement_to_screen_sign_agreement": (
            anchor_v14a._metric_summary(screen_signs)
        ),
    }
    full = vectors["full_frame"]
    candidate = {
        "schema": "eggroll-es-full-frame-matched56-summary-v14a",
        "experiment_name": prereg_v14a.EXPERIMENT_NAME_V14A,
        "alpha": 0.0,
        "model_update_applied": False,
        "validation_ood_or_heldout_used": False,
        "perturbation_basis_sha256": (
            prereg_v14a.PERTURBATION_BASIS_SHA256_V14A
        ),
        "panel_identities": {
            "full_frame": prereg_v14a.FULL_FRAME_IDENTITY_V14A[
                "ordered_row_identity_sha256"
            ],
            **{
                name: prereg_v14a.PANEL_IDENTITIES_V14A[name][
                    "ordered_row_identity_sha256"
                ]
                for name in prereg_v14a.PANEL_NAMES_V14A
            },
        },
        "stability": stability,
        "all_panel_spreads_nonzero": all(
            not value["zero_spread"]
            for value in standardization.values()
        ),
        "robust_aggregate": {
            "coefficient_sha256": anchor_v14a.coefficient_sha256(
                anchor_v14a.PERTURBATION_SEEDS_V14A, full,
            ),
            "l2_norm": math.sqrt(math.fsum(value * value for value in full)),
            "nonzero_coordinate_count": sum(value != 0.0 for value in full),
        },
    }
    candidate["content_sha256_before_self_field"] = _canonical(candidate)
    gate = prereg_v14a.evaluate_candidate_v14a(candidate)
    return standardization, candidate, gate


def validate_v14a_negative_v14b(
    attempt_path=ATTEMPT_PATH_V14B, report_path=REPORT_PATH_V14B,
):
    attempt_path = Path(attempt_path).resolve()
    report_path = Path(report_path).resolve()
    if attempt_path != ATTEMPT_PATH_V14B or report_path != REPORT_PATH_V14B:
        raise RuntimeError("v14b requires canonical V14a aggregate paths")
    if (
        _file_sha256(attempt_path) != ATTEMPT_FILE_SHA256_V14B
        or _file_sha256(report_path) != REPORT_FILE_SHA256_V14B
    ):
        raise RuntimeError("v14b V14a aggregate file identity changed")
    attempt = json.loads(attempt_path.read_text())
    report = json.loads(report_path.read_text())
    if (
        attempt.get("schema") != "eggroll-es-durable-launch-attempt-v14a"
        or attempt.get("status") != "complete"
        or attempt.get("phase") != "after_trainer_cleanup_and_report"
        or attempt.get("experiment_name") != prereg_v14a.EXPERIMENT_NAME_V14A
        or attempt.get("target_alpha_zero_only") is not True
        or attempt.get("model_update_applied") is not False
        or attempt.get("sealed_or_nontrain_surface_opened") is not False
        or attempt.get("report_exists_after_attempt") is not True
        or attempt.get("content_sha256_before_self_field")
        != ATTEMPT_CONTENT_SHA256_V14B
        or attempt.get("content_sha256_before_self_field")
        != _canonical({
            key: value for key, value in attempt.items()
            if key != "content_sha256_before_self_field"
        })
        or report.get("schema")
        != "eggroll-es-full-frame-alpha-zero-report-v14a"
        or report.get("model_update_applied") is not False
        or report.get("sealed_or_nontrain_surface_opened") is not False
        or report.get("decision") != "train_only_sampler_gate_no_model_update"
        or report.get("content_sha256_before_self_field")
        != REPORT_CONTENT_SHA256_V14B
        or report.get("content_sha256_before_self_field")
        != _canonical({
            key: value for key, value in report.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v14b V14a aggregate completion semantics changed")
    binding = attempt.get("report_binding", {})
    if binding != {
        "path": str(REPORT_PATH_V14B),
        "file_sha256": REPORT_FILE_SHA256_V14B,
        "content_sha256": REPORT_CONTENT_SHA256_V14B,
    }:
        raise RuntimeError("v14b V14a report binding changed")
    source = attempt.get("source_provenance", {})
    implementation = report.get("implementation", {})
    if (
        source.get("git_head") != SOURCE_COMMIT_V14B
        or source.get("implementation_bundle_sha256")
        != IMPLEMENTATION_BUNDLE_SHA256_V14B
        or source.get("content_sha256_before_self_field")
        != SOURCE_BUNDLE_CONTENT_SHA256_V14B
        or source.get("content_sha256_before_self_field")
        != _canonical({
            key: value for key, value in source.items()
            if key != "content_sha256_before_self_field"
        })
        or implementation.get("bundle_sha256")
        != IMPLEMENTATION_BUNDLE_SHA256_V14B
        or implementation.get("bundle_sha256")
        != _canonical(implementation.get("files"))
    ):
        raise RuntimeError("v14b V14a committed implementation changed")
    for item in source.get("files", {}).values():
        relative = item["relative_path"]
        raw = subprocess.check_output(
            ["git", "show", f"{SOURCE_COMMIT_V14B}:{relative}"], cwd=ROOT,
        )
        if (
            hashlib.sha256(raw).hexdigest() != item["file_sha256"]
            or _file_sha256(ROOT / relative) != item["file_sha256"]
        ):
            raise RuntimeError("v14b V14a source provenance changed")
    recipe = report.get("recipe", {})
    diagnostic = report.get("diagnostic", {})
    if (
        recipe.get("content_sha256_before_self_field")
        != RECIPE_CONTENT_SHA256_V14B
        or recipe.get("content_sha256_before_self_field")
        != _canonical({
            key: value for key, value in recipe.items()
            if key != "content_sha256_before_self_field"
        })
        or recipe.get("alpha") != 0.0
        or recipe.get("model_update_allowed") is not False
        or recipe.get("implementation_bundle_sha256")
        != IMPLEMENTATION_BUNDLE_SHA256_V14B
        or recipe.get("hardware") != {
            "engine_count": 4, "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3], "complete_wave_required": True,
        }
        or recipe.get("generation") != {
            "prompts_per_engine_per_sign": 310,
            "generation_calls_per_engine_per_sign": 1,
            "matched_and_crossfit_responses_derived_without_generation": True,
        }
        or diagnostic.get("content_sha256_before_self_field")
        != DIAGNOSTIC_CONTENT_SHA256_V14B
        or diagnostic.get("content_sha256_before_self_field")
        != _canonical({
            key: value for key, value in diagnostic.items()
            if key != "content_sha256_before_self_field"
        })
        or diagnostic.get("alpha") != 0.0
        or diagnostic.get("model_update_applied") is not False
        or diagnostic.get("applications") != []
        or diagnostic.get("generation_contract") != recipe["generation"]
        or diagnostic.get("hardware_coverage") != {
            "engine_count": 4, "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3], "population_waves": 8,
            "signed_waves": 16, "partial_waves": 0,
            "all_engines_generate_every_signed_wave": True,
        }
    ):
        raise RuntimeError("v14b V14a runtime contract changed")
    identity = diagnostic.get("identity_audit", {})
    boundary = diagnostic.get("population_boundary_audit_v4", {})
    exact_checks = identity.get("exact_reference_checks")
    if (
        identity.get("passed") is not True
        or identity.get("pre_probe") != identity.get("post_probe")
        or not anchor_v14a.anchor_v4.anchor_v3.anchor_v2._all_collective_results(
            exact_checks,
            lambda value: (
                isinstance(value, dict) and value.get("passed") is True
            ),
        )
        or boundary.get("passed") is not True
        or boundary.get("audit_sha256") != BOUNDARY_AUDIT_SHA256_V14B
        or boundary.get("audit_sha256") != _canonical({
            key: value for key, value in boundary.items()
            if key != "audit_sha256"
        })
    ):
        raise RuntimeError("v14b V14a restoration audit changed")
    standardization, candidate, gate = _aggregate_analysis_v14b(
        diagnostic.get("responses")
    )
    analysis = diagnostic.get("analysis", {})
    if (
        analysis.get("standardization") != standardization
        or analysis.get("candidate_summary") != candidate
        or analysis.get("promotion_gate") != gate
        or candidate.get("content_sha256_before_self_field")
        != CANDIDATE_CONTENT_SHA256_V14B
        or candidate.get("robust_aggregate", {}).get("coefficient_sha256")
        != AGGREGATE_COEFFICIENT_SHA256_V14B
        or candidate.get("stability") != EXPECTED_STABILITY_V14B
        or candidate.get("all_panel_spreads_nonzero") is not True
        or gate.get("eligible_for_train_only_sampler_adoption") is not False
        or gate.get("eligible_for_model_update") is not False
        or gate.get("failure_decision")
        != "retain_v13_sampler_and_keep_eval_surfaces_closed"
    ):
        raise RuntimeError("v14b V14a aggregate gate changed")
    return attempt, report, candidate, gate


def build_evidence_v14b():
    attempt, report, candidate, gate = validate_v14a_negative_v14b()
    diagnostic = report["diagnostic"]
    conditions = gate["conditions"]
    failed_rules = []
    for name, condition in conditions.items():
        for key, passed in condition.items():
            if key.endswith("passed") and passed is False:
                failed_rules.append(f"{name}.{key}")
    evidence = {
        "schema": "eggroll-es-v14a-full-frame-negative-aggregate-evidence-v14b",
        "passed": True,
        "selection_surface": "frozen_train_aggregate_outputs_only",
        "contains_response_vectors_or_dense_result_hashes": False,
        "contains_source_rows_questions_answers_or_document_content": False,
        "contains_validation_ood_heldout_or_benchmark_content": False,
        "v14a_attempt": {
            "path": str(ATTEMPT_PATH_V14B),
            "file_sha256": ATTEMPT_FILE_SHA256_V14B,
            "content_sha256": ATTEMPT_CONTENT_SHA256_V14B,
            "source_commit": SOURCE_COMMIT_V14B,
            "source_bundle_content_sha256": (
                SOURCE_BUNDLE_CONTENT_SHA256_V14B
            ),
        },
        "v14a_report": {
            "path": str(REPORT_PATH_V14B),
            "file_sha256": REPORT_FILE_SHA256_V14B,
            "content_sha256": REPORT_CONTENT_SHA256_V14B,
            "implementation_bundle_sha256": (
                IMPLEMENTATION_BUNDLE_SHA256_V14B
            ),
            "recipe_content_sha256": RECIPE_CONTENT_SHA256_V14B,
            "diagnostic_content_sha256": DIAGNOSTIC_CONTENT_SHA256_V14B,
        },
        "runtime_integrity": {
            "alpha": diagnostic["alpha"],
            "model_update_applied": diagnostic["model_update_applied"],
            "application_count": len(diagnostic["applications"]),
            "identity_audit_passed": diagnostic["identity_audit"]["passed"],
            "pre_post_base_probe_equal": (
                diagnostic["identity_audit"]["pre_probe"]
                == diagnostic["identity_audit"]["post_probe"]
            ),
            "population_boundary_audit_sha256": BOUNDARY_AUDIT_SHA256_V14B,
            "hardware": diagnostic["hardware_coverage"],
            "generation": diagnostic["generation_contract"],
            "panel_bundle_content_sha256": diagnostic[
                "panel_bundle_content_sha256"
            ],
            "perturbation_basis_sha256": diagnostic[
                "perturbation_basis"
            ]["basis_sha256"],
        },
        "aggregate_gate": {
            "candidate_content_sha256": CANDIDATE_CONTENT_SHA256_V14B,
            "robust_aggregate": candidate["robust_aggregate"],
            "stability": candidate["stability"],
            "baseline": prereg_v14a.BASELINE_STABILITY_V14A,
            "conditions": conditions,
            "failed_rules": failed_rules,
            "all_eight_spreads_nonzero": candidate[
                "all_panel_spreads_nonzero"
            ],
            "eligible_for_train_only_sampler_adoption": False,
            "eligible_for_model_update": False,
        },
        "decision": {
            "sampler": "retain_v13",
            "row_draw_iteration_1_confirmation_authorized": False,
            "evaluation_surface_opened": False,
            "model_update_authorized": False,
            "reason": "v14a_failed_its_preregistered_conjunctive_gate",
        },
        "next_train_only_estimator_recommendation": {
            "name": "fresh_two_distinct_rows_per_multiline_document_mean",
            "status": "recommend_preregister_before_implementation_or_launch",
            "hypothesis_count": 1,
            "full_frame_documents": 310,
            "single_row_documents": 139,
            "multirow_documents": 171,
            "distinct_rows_per_multirow_document": 2,
            "unique_train_prompts_per_direction_sign": 481,
            "construction": (
                "freeze two distinct source-independent deterministic rows for "
                "every multirow document and the sole row for every singleton"
            ),
            "estimates": (
                "average row rewards within each document first, then compute "
                "the equal-document full-frame mean plus the same five matched56 "
                "HT means and two disjoint 254-document complement means"
            ),
            "generation": (
                "one identical 481-prompt generation batch per direction and "
                "sign; derive matched56 and crossfits without extra generation"
            ),
            "runtime": (
                "alpha zero, same frozen 32-direction basis, plus then minus, "
                "four TP=1 engines, exact restoration after every sign"
            ),
            "persistence": (
                "aggregate signed response vectors and dense-result hashes "
                "during the run; compact evidence retains metrics and hashes only"
            ),
            "gate_scope": (
                "the same full-to-matched56, matched56-pair, and disjoint-crossfit "
                "cosine/sign family with all thresholds frozen before launch"
            ),
            "authorization_if_passed": (
                "only a separately preregistered alpha-zero confirmation on a "
                "fresh 32-direction basis; no evaluation or model update"
            ),
            "why_not_multiplicity_search_now": (
                "testing k in {1,2,3,all} on one 794-row batch would select k on "
                "the same gate and require preregistered multiple-testing control "
                "plus an independent fresh-basis confirmation; k=2 is one cleaner "
                "cost-bounded hypothesis that directly targets row-choice variance"
            ),
            "current_authorization": (
                "recommendation only; do not launch, open evaluation, or update"
            ),
        },
    }
    evidence["content_sha256_before_self_field"] = _canonical(evidence)
    return evidence


def _exclusive_write(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("v14b compact negative evidence already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default=str(OUTPUT_PATH_V14B))
    args = parser.parse_args(argv)
    output = Path(args.output_json).resolve()
    if output != OUTPUT_PATH_V14B:
        raise ValueError("v14b compact evidence requires its canonical path")
    evidence = build_evidence_v14b()
    _exclusive_write(output, evidence)
    print(json.dumps({
        "output": str(output),
        "file_sha256": _file_sha256(output),
        "content_sha256": evidence["content_sha256_before_self_field"],
    }, sort_keys=True))
    return evidence


if __name__ == "__main__":
    main()
