#!/usr/bin/env python3
"""Build compact immutable negative evidence for completed V30A."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ATTEMPT_RELATIVE_PATH_V30B = (
    "experiments/eggroll_es_hpo/runs/"
    ".s6_v30a_paired_production_v389_train_only_runtime_basis20261003."
    "launch_attempt.json"
)
REPORT_RELATIVE_PATH_V30B = (
    "experiments/eggroll_es_hpo/runs/"
    "s6_v30a_paired_production_v389_train_only_runtime_basis20261003/"
    "paired_production_v389_compatibility_v30a.json"
)
OUTPUT_RELATIVE_PATH_V30B = (
    "experiments/eggroll_es_hpo/"
    "S6_V30B_V30A_PAIRED_V389_NEGATIVE_EVIDENCE.json"
)
ATTEMPT_PATH_V30B = ROOT / ATTEMPT_RELATIVE_PATH_V30B
REPORT_PATH_V30B = ROOT / REPORT_RELATIVE_PATH_V30B
OUTPUT_PATH_V30B = ROOT / OUTPUT_RELATIVE_PATH_V30B

ATTEMPT_FILE_SHA256_V30B = (
    "c0a8b9e4e7b4607fa37eebfde0ae57fc561716ed94e48b25f01fef481f2fe995"
)
ATTEMPT_CONTENT_SHA256_V30B = (
    "a93d1bd28eff0607c9ce4ff165e8b3ae24428995cc4fbbadc2a99ed32d41cc01"
)
REPORT_FILE_SHA256_V30B = (
    "784dbeece2ffa3eace520db8083d9f6c6b1d66bfc688257f37de5b38d2d0a5af"
)
REPORT_CONTENT_SHA256_V30B = (
    "1073fe301dab129215e2c7c8e8877ecdd9be4fc9c3f71d920a788d2ec2e381b3"
)
PREREGISTRATION_COMMIT_V30B = "c54cbf4cdea670f4044a6dc5fb035eb20face83c"
HARDENED_IMPLEMENTATION_COMMIT_V30B = (
    "5448c17a5217bd37dedd41e3a42b4fe5bd72d5a4"
)
LAUNCH_SOURCE_COMMIT_V30B = "a203f4821c4a737310df75543353d21ce6cea978"
CANDIDATE_INPUT_FREEZE_COMMIT_V30B = (
    "2e8a6b7d02fbc77a2442f6790fe0f80f1bebc02e"
)
PREREGISTRATION_RELATIVE_PATH_V30B = (
    "experiments/eggroll_es_hpo/"
    "S6_PAIRED_DATA_COMPAT_V30A_PREREGISTRATION.json"
)
RUNTIME_RELATIVE_PATH_V30B = "run_eggroll_es_paired_data_compat_v30a.py"
PREREGISTRATION_FILE_SHA256_V30B = (
    "543c90672961ba08e30a0bf87f0278a517257372f7564c84a848350e8951afcb"
)
RUNTIME_FILE_SHA256_V30B = (
    "2546bf81078e33b92908472dd434f35064749b8d0e73b559cf1dcd58700d22e3"
)
IMPLEMENTATION_BUNDLE_SHA256_V30B = (
    "2ff928d4b74fc018e38895481b33ed8c1b12e1ab0b7de77ef444e80050deb717"
)
RECIPE_CONTENT_SHA256_V30B = (
    "c78c103dc197c51d8dec9ac5b9958d47ca333fec4cbcf49a7df315c62e4724f4"
)
SOURCE_PROVENANCE_CONTENT_SHA256_V30B = (
    "d3669fa265f3e2cf4879670fd294da27d0de3ba6e0637256b1d429d871cf65d6"
)
COMMITTED_CLEAN_SOURCE_CONTENT_SHA256_V30B = (
    "09ef0135e222dac94b6a41477dbe670ea9371690efdfea21ef229b4d50a77ac4"
)
RUNTIME_ENVIRONMENT_CONTENT_SHA256_V30B = (
    "7acc87ffb6a50682f19309ea3059823d0fc96dfc459e30cb66d64e157a76f52a"
)
LIVE_MODEL_AUDIT_CONTENT_SHA256_V30B = (
    "f5cb0baf14dd17efc679a2baf7bc2377552f4b9cd6e423e1376b5b14b65623b6"
)
CONFIGURATION_CONTENT_SHA256_V30B = (
    "3c45c84cb97fbe7cf59edaaad635f16d6ca41e4a3988edcf894351e3f82e763a"
)
SUMMARY_CONTENT_SHA256_V30B = (
    "70c29d39971a1a7341cd365e484a63cfe25273e3ab6414ea694d7f83324e792a"
)
RUNTIME_AUDIT_CONTENT_SHA256_V30B = (
    "d757a40019bddeb715a1f8106b7ac90c0865d23fba020d31db4d2b438889c5c1"
)
GATE_CONTENT_SHA256_V30B = (
    "be250c10ddad448d65b46d71f8265f1ca95702e0692bfaea8944ddb31153b785"
)
PRELAUNCH_IDLE_SHA256_V30B = (
    "28c3f8d1bc9ab8fee79f18736af8c800655b8cf48852c8f5ba60151399708094"
)
FINAL_IDLE_SHA256_V30B = (
    "00740c6fd0661c2613bbdc3a0e3b02d902f22e9f036eb085e1b0c0d497b0c64a"
)

EXPECTED_ENDPOINTS_V30B = {
    "aggregate_to_optimization_cosine_median": {
        "candidate_v389_minus_production": -0.0833085058352121,
        "familywise_lcb": -0.2500635087576087,
    },
    "aggregate_to_optimization_cosine_worst": {
        "candidate_v389_minus_production": -0.05402994133497441,
        "familywise_lcb": -0.34971867133086976,
    },
    "aggregate_to_optimization_sign_agreement_median": {
        "candidate_v389_minus_production": 0.03125,
        "familywise_lcb": -0.125,
    },
    "aggregate_to_optimization_sign_agreement_worst": {
        "candidate_v389_minus_production": -0.0625,
        "familywise_lcb": -0.1875,
    },
    "optimization_pairwise_cosine_median": {
        "candidate_v389_minus_production": -0.02585786049274097,
        "familywise_lcb": -0.40331670683301735,
    },
    "optimization_pairwise_cosine_worst": {
        "candidate_v389_minus_production": -0.1476208521518517,
        "familywise_lcb": -0.44378228533250474,
    },
    "optimization_pairwise_sign_agreement_median": {
        "candidate_v389_minus_production": 0.0,
        "familywise_lcb": -0.21875,
    },
    "optimization_pairwise_sign_agreement_worst": {
        "candidate_v389_minus_production": -0.03125,
        "familywise_lcb": -0.25,
    },
    "train_screen_cosine_median": {
        "candidate_v389_minus_production": 0.07096498366367587,
        "familywise_lcb": -0.20262201264726712,
    },
    "train_screen_cosine_worst": {
        "candidate_v389_minus_production": 0.013487686321821957,
        "familywise_lcb": -0.2877068758549798,
    },
    "train_screen_sign_agreement_median": {
        "candidate_v389_minus_production": 0.0,
        "familywise_lcb": -0.15625,
    },
    "train_screen_sign_agreement_worst": {
        "candidate_v389_minus_production": -0.03125,
        "familywise_lcb": -0.1875,
    },
}
EXPECTED_PRODUCTION_ENDPOINTS_V30B = {
    "aggregate_to_optimization_cosine_median": 0.7498208586226849,
    "aggregate_to_optimization_cosine_worst": 0.7196986406021203,
    "aggregate_to_optimization_sign_agreement_median": 0.78125,
    "aggregate_to_optimization_sign_agreement_worst": 0.78125,
    "optimization_pairwise_cosine_median": 0.4060891168361095,
    "optimization_pairwise_cosine_worst": 0.3822483529282259,
    "optimization_pairwise_sign_agreement_median": 0.65625,
    "optimization_pairwise_sign_agreement_worst": 0.5625,
    "train_screen_cosine_median": 0.5011910171739168,
    "train_screen_cosine_worst": 0.4457360396699477,
    "train_screen_sign_agreement_median": 0.71875,
    "train_screen_sign_agreement_worst": 0.6875,
}
EXPECTED_CANDIDATE_ENDPOINTS_V30B = {
    name: EXPECTED_PRODUCTION_ENDPOINTS_V30B[name]
    + EXPECTED_ENDPOINTS_V30B[name]["candidate_v389_minus_production"]
    for name in EXPECTED_ENDPOINTS_V30B
}
EXPECTED_RUNTIME_INTEGRITY_KEYS_V30B = {
    "all_four_engines_scored_both_versions_every_signed_wave",
    "all_integrity_audits_passed",
    "alternating_version_order_complete",
    "exact_selected_restore_after_every_signed_wave",
    "fixed_side_representatives_every_direction_and_sign",
    "full_context_a_b_equal_before_first_perturbation",
    "full_context_a_c_equal_after_population_boundary",
    "same_resident_perturbation_scored_both_versions",
    "selected_and_unselected_population_boundary_passed",
    "token_and_request_identity_audits_passed",
}
FORBIDDEN_EVIDENCE_TERMS_V30B = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "token", "tokens", "response", "responses", "row", "rows", "unit",
    "units", "bootstrap", "heldout", "validation", "ood", "benchmark",
}


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


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
    _require(path.is_file() and not path.is_symlink(), f"V30B {label} path changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"V30B {label} must be a JSON object")
    return value


def _verify_self(value, expected, label):
    _require(
        value.get("content_sha256_before_self_field") == expected
        and canonical_sha256(_without_self(value)) == expected,
        f"V30B {label} self hash changed",
    )


def _all_recursive_strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).lower()
            yield from _all_recursive_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_recursive_strings(item)
    elif isinstance(value, str):
        yield value.lower()


def _assert_compact_v30b(value):
    offenders = []
    for text in _all_recursive_strings(value):
        words = set(text.replace("-", "_").replace("/", "_").split("_"))
        overlap = words & FORBIDDEN_EVIDENCE_TERMS_V30B
        if overlap:
            offenders.extend(sorted(overlap))
    if offenders:
        raise RuntimeError(
            f"V30B evidence contains forbidden detail terms: {sorted(set(offenders))}"
        )


def _git_show(commit, relative_path):
    return subprocess.check_output(
        ["git", "show", f"{commit}:{relative_path}"], cwd=ROOT,
    )


def validate_history_v30b():
    _require(
        hashlib.sha256(_git_show(
            PREREGISTRATION_COMMIT_V30B, PREREGISTRATION_RELATIVE_PATH_V30B,
        )).hexdigest() == PREREGISTRATION_FILE_SHA256_V30B,
        "V30B c54 V30A preregistration history changed",
    )
    for commit in (
        HARDENED_IMPLEMENTATION_COMMIT_V30B, LAUNCH_SOURCE_COMMIT_V30B,
    ):
        _require(
            hashlib.sha256(_git_show(commit, RUNTIME_RELATIVE_PATH_V30B)).hexdigest()
            == RUNTIME_FILE_SHA256_V30B,
            "V30B hardened or launch runtime history changed",
        )
    for older, newer in (
        (PREREGISTRATION_COMMIT_V30B, HARDENED_IMPLEMENTATION_COMMIT_V30B),
        (HARDENED_IMPLEMENTATION_COMMIT_V30B, LAUNCH_SOURCE_COMMIT_V30B),
    ):
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", older, newer], cwd=ROOT,
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _require(result.returncode == 0, "V30B source commit ancestry changed")


def _validate_source_and_implementation_v30b(attempt, report):
    provenance = attempt.get("source_provenance", {})
    clean = attempt.get("committed_clean_source_certificate", {})
    implementation = report.get("implementation", {})
    _verify_self(
        provenance, SOURCE_PROVENANCE_CONTENT_SHA256_V30B, "source provenance",
    )
    _verify_self(
        clean, COMMITTED_CLEAN_SOURCE_CONTENT_SHA256_V30B,
        "committed-clean source certificate",
    )
    _require(
        provenance.get("git_head") == LAUNCH_SOURCE_COMMIT_V30B
        and clean.get("git_head") == LAUNCH_SOURCE_COMMIT_V30B
        and clean.get("all_tracked_files_clean") is True
        and clean.get("only_explicitly_allowlisted_untracked_paths_present") is True
        and clean.get("allowed_untracked_entry_count") == 231
        and provenance.get("implementation_bundle_sha256")
        == IMPLEMENTATION_BUNDLE_SHA256_V30B
        and implementation.get("bundle_sha256")
        == IMPLEMENTATION_BUNDLE_SHA256_V30B,
        "V30B clean launch source or implementation binding changed",
    )
    source_files = provenance.get("files", {})
    implementation_files = implementation.get("files", {})
    _require(
        len(source_files) == 65
        and len(implementation_files) == 65
        and len(implementation.get("immutable_bound_files", {})) == 10
        and canonical_sha256(implementation_files)
        == IMPLEMENTATION_BUNDLE_SHA256_V30B
        and implementation.get("inherited_v23a_r2_bundle_sha256")
        == "4bbd31dbdb61366d6ca61c8f5955df3e59e1b34642cd12aca84b54153586d6a6"
        and implementation.get("v30a_overlay_bundle_sha256")
        == "188b6de48b63abf56f85c2d39138313ccfd8f59ee6b246505032529a2744b6f2",
        "V30B source implementation cardinality or aggregate hash changed",
    )
    normalized = {
        key: {
            "relative_path": Path(item["path"]).resolve().relative_to(ROOT).as_posix(),
            "file_sha256": item["file_sha256"],
        }
        for key, item in implementation_files.items()
    }
    _require(
        normalized == source_files,
        "V30B source provenance and implementation files disagree",
    )
    recipe = report.get("recipe", {})
    _verify_self(recipe, RECIPE_CONTENT_SHA256_V30B, "runtime recipe")
    _require(
        attempt.get("recipe") == recipe
        and recipe.get("implementation_bundle_sha256")
        == IMPLEMENTATION_BUNDLE_SHA256_V30B
        and recipe.get("preregistration", {}).get("file_sha256")
        == PREREGISTRATION_FILE_SHA256_V30B
        and recipe.get("preregistration", {}).get("candidate_freeze_commit")
        == CANDIDATE_INPUT_FREEZE_COMMIT_V30B,
        "V30B attempt report recipe or candidate freeze binding changed",
    )


def _validate_runtime_and_cardinality_v30b(attempt, report):
    environment = attempt.get("runtime_environment_certificate", {})
    live_model = attempt.get("live_model_audit", {})
    configuration = report.get("configuration", {})
    summary = report.get("summary", {})
    audit = report.get("runtime_audit", {})
    _verify_self(
        environment, RUNTIME_ENVIRONMENT_CONTENT_SHA256_V30B,
        "runtime environment",
    )
    _verify_self(live_model, LIVE_MODEL_AUDIT_CONTENT_SHA256_V30B, "live model")
    _verify_self(configuration, CONFIGURATION_CONTENT_SHA256_V30B, "configuration")
    _verify_self(summary, SUMMARY_CONTENT_SHA256_V30B, "summary")
    _verify_self(audit, RUNTIME_AUDIT_CONTENT_SHA256_V30B, "runtime audit")
    recipe = report["recipe"]
    accounting = recipe.get("request_accounting", {})
    frame = recipe.get("frame", {})
    panels = recipe.get("panels", {})
    _require(
        environment.get("completed_before_attempt_claim") is True
        and environment.get("cuda_device_count") == 4
        and environment.get("cuda_visible_devices") == "0,1,2,3"
        and environment.get("dataset_or_evaluation_surface_opened") is False
        and live_model.get("config_sha256")
        == "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
        and live_model.get("index_sha256")
        == "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
        "V30B runtime environment or live model integrity changed",
    )
    _require(
        frame.get("selected_paired_units") == 195
        and frame.get("reserve_paired_units") == 10
        and frame.get("shared_document_anchors") == 193
        and frame.get("joint_component_cross_side_anchors") == 2
        and frame.get("globally_disjoint_joint_units") is True
        and len(panels) == 5
        and all(item.get("paired_units") == 39 for item in panels.values())
        and accounting == {
            "full_context_phase_count": 3,
            "full_context_requests_all_engines": 4680,
            "full_context_requests_all_engines_per_phase": 1560,
            "paired_units_per_panel": 39,
            "panels": 5,
            "perturbed_requests_all_engines": 24960,
            "requests_all_engines_per_signed_wave": 1560,
            "requests_per_engine_per_signed_wave": 390,
            "requests_per_version_per_engine_call": 195,
            "signed_waves": 16,
            "total_generation_requests": 29640,
            "versions_per_signed_wave": 2,
        },
        "V30B durable panel or request cardinality changed",
    )
    runtime_integrity = summary.get("runtime_integrity", {})
    guard = audit.get("full_context_guard", {})
    _require(
        set(runtime_integrity) == EXPECTED_RUNTIME_INTEGRITY_KEYS_V30B
        and all(value is True for value in runtime_integrity.values())
        and audit.get("signed_wave_count") == 16
        and audit.get("perturbed_requests_all_engines") == 24960
        and audit.get("full_context_requests_all_engines") == 4680
        and audit.get("total_generation_requests") == 29640
        and audit.get("per_unit_scores_or_bootstrap_replicates_persisted") is False
        and audit.get("restore_checks_sha256")
        == "5141ebc7c5c1f9a291330e99233894267d783a8afd4c3ceef670445dbcc59b97"
        and audit.get("population_boundary_audit_sha256")
        == "0fafb9834bbc724795b72a2bfa7da36c3539f3ee093995180f8c3f7bd2087988"
        and guard.get("a_b_exact") == {
            "all_dense_result_commitments_exact": True,
            "all_version_engine_panel_score_arrays_exact": True,
        }
        and guard.get("a_c_exact") == {
            "all_dense_result_commitments_exact": True,
            "all_version_engine_panel_score_arrays_exact": True,
        }
        and guard.get("phase_a") == guard.get("phase_b") == guard.get("phase_c")
        and guard.get("all_four_engines_both_versions_each_phase") is True
        and guard.get("excluded_from_estimator_and_bootstrap") is True
        and guard.get("raw_scores_or_outputs_persisted") is False,
        "V30B restore boundary or full-context runtime integrity changed",
    )
    return summary


def _validate_gate_v30b(report, summary):
    gate = report.get("gate", {})
    _verify_self(gate, GATE_CONTENT_SHA256_V30B, "gate")
    paired = summary.get("paired_bootstrap", {})
    endpoints = paired.get("endpoints", {})
    normalized = {
        name: {
            "candidate_v389_minus_production": item.get(
                "candidate_v389_minus_production"
            ),
            "familywise_lcb": item.get("familywise_lcb"),
        }
        for name, item in endpoints.items()
    }
    _require(
        paired.get("repetitions") == 50_000
        and paired.get("seed") == 20_261_004
        and paired.get("one_sided_quantile") == 0.05 / 12
        and paired.get("raw_draws_or_replicates_persisted") is False
        and set(endpoints) == set(EXPECTED_ENDPOINTS_V30B)
        and all(item.get("noninferiority_margin") == 0.0 for item in endpoints.values())
        and normalized == EXPECTED_ENDPOINTS_V30B,
        "V30B exact 12 endpoint aggregates changed",
    )
    versions = summary.get("versions", {})
    _require(
        versions.get("production", {}).get("endpoint_values")
        == EXPECTED_PRODUCTION_ENDPOINTS_V30B
        and versions.get("candidate_v389", {}).get("endpoint_values")
        == EXPECTED_CANDIDATE_ENDPOINTS_V30B
        and versions.get("production", {}).get("all_panel_spreads_nonzero") is True
        and versions.get("candidate_v389", {}).get("all_panel_spreads_nonzero") is True,
        "V30B exact production or candidate point endpoints changed",
    )
    recomputed = all(
        item["familywise_lcb"] >= 0.0
        for item in EXPECTED_ENDPOINTS_V30B.values()
    )
    _require(
        recomputed is False
        and gate.get("pass") is False
        and gate.get("all_12_familywise_lcbs_nonnegative") is False
        and gate.get("all_runtime_integrity_audits_passed") is True
        and gate.get("decision") == "retain_production_dataset_and_v13_recipe"
        and gate.get("checkpoint_write_authorized") is False
        and gate.get("dataset_promotion_authorized") is False
        and gate.get("evaluation_authorized") is False
        and gate.get("model_update_authorized") is False,
        "V30B negative gate recomputation or decision changed",
    )


def _validate_closed_side_effects_v30b(attempt, report):
    _require(
        attempt.get("schema") == "eggroll-es-durable-launch-attempt-v30a"
        and attempt.get("status") == "complete"
        and attempt.get("phase") == "after_compact_report_and_final_gpu_cleanup"
        and report.get("schema") == "eggroll-es-paired-data-compat-report-v30a"
        and attempt.get("prelaunch_idle_certificate_sha256")
        == report.get("prelaunch_idle_certificate_sha256")
        == PRELAUNCH_IDLE_SHA256_V30B
        and attempt.get("final_idle_certificate_sha256")
        == report.get("final_idle_certificate_sha256")
        == FINAL_IDLE_SHA256_V30B
        and attempt.get("all_four_gpus_idle_after_cleanup") is True
        and report.get("all_four_gpus_idle_after_cleanup") is True,
        "V30B durable completion preclaim or final-idle integrity changed",
    )
    for value in (attempt, report):
        _require(
            value.get("checkpoint_written") is False
            and value.get("dataset_promotion_applied") is False
            and value.get("evaluation_opened") is False
            and value.get("model_update_applied") is False,
            "V30B forbidden mutation checkpoint evaluation or promotion changed",
        )
    _require(
        report.get("direct_action_taken") is False
        and report.get("configuration", {}).get("checkpoint_write_allowed") is False
        and report.get("configuration", {}).get("evaluation_allowed") is False
        and report.get("configuration", {}).get("model_update_allowed") is False
        and report.get("summary", {}).get(
            "persisted_response_vectors_rows_draws_or_replicates"
        ) is False,
        "V30B direct action configuration or persistence boundary changed",
    )


def validate_bound_artifacts_v30b():
    _require(
        file_sha256(ATTEMPT_PATH_V30B) == ATTEMPT_FILE_SHA256_V30B,
        "V30B attempt file hash changed",
    )
    _require(
        file_sha256(REPORT_PATH_V30B) == REPORT_FILE_SHA256_V30B,
        "V30B report file hash changed",
    )
    attempt = _load_json_object(ATTEMPT_PATH_V30B, "attempt")
    report = _load_json_object(REPORT_PATH_V30B, "report")
    _verify_self(attempt, ATTEMPT_CONTENT_SHA256_V30B, "attempt")
    _verify_self(report, REPORT_CONTENT_SHA256_V30B, "report")
    _require(
        attempt.get("report_binding") == {
            "path": str(REPORT_PATH_V30B.resolve()),
            "file_sha256": REPORT_FILE_SHA256_V30B,
            "content_sha256": REPORT_CONTENT_SHA256_V30B,
        },
        "V30B attempt report binding changed",
    )
    validate_history_v30b()
    _validate_source_and_implementation_v30b(attempt, report)
    summary = _validate_runtime_and_cardinality_v30b(attempt, report)
    _validate_gate_v30b(report, summary)
    _validate_closed_side_effects_v30b(attempt, report)
    return attempt, report


def build_negative_evidence_v30b():
    attempt, report = validate_bound_artifacts_v30b()
    summary = report["summary"]
    points = [
        item["candidate_v389_minus_production"]
        for item in EXPECTED_ENDPOINTS_V30B.values()
    ]
    lcbs = [item["familywise_lcb"] for item in EXPECTED_ENDPOINTS_V30B.values()]
    evidence = _seal({
        "schema": "eggroll-es-v30a-paired-v389-negative-evidence-v30b",
        "status": "valid_completed_negative_gate_no_action",
        "artifacts": {
            "durable_attempt": {
                "relative_path": ATTEMPT_RELATIVE_PATH_V30B,
                "file_sha256": ATTEMPT_FILE_SHA256_V30B,
                "content_sha256": ATTEMPT_CONTENT_SHA256_V30B,
            },
            "compact_report": {
                "relative_path": REPORT_RELATIVE_PATH_V30B,
                "file_sha256": REPORT_FILE_SHA256_V30B,
                "content_sha256": REPORT_CONTENT_SHA256_V30B,
            },
        },
        "source_history": {
            "preregistration_commit": PREREGISTRATION_COMMIT_V30B,
            "hardened_implementation_commit": HARDENED_IMPLEMENTATION_COMMIT_V30B,
            "clean_launch_source_commit": LAUNCH_SOURCE_COMMIT_V30B,
            "candidate_input_freeze_commit": CANDIDATE_INPUT_FREEZE_COMMIT_V30B,
            "implementation_bundle_sha256": IMPLEMENTATION_BUNDLE_SHA256_V30B,
            "recipe_content_sha256": RECIPE_CONTENT_SHA256_V30B,
            "source_provenance_content_sha256": (
                SOURCE_PROVENANCE_CONTENT_SHA256_V30B
            ),
            "committed_clean_source_content_sha256": (
                COMMITTED_CLEAN_SOURCE_CONTENT_SHA256_V30B
            ),
            "implementation_file_count": 65,
            "immutable_bound_file_count": 10,
        },
        "aggregate_execution": {
            "production_record_count": report["recipe"]["sources"]["production"]["rows"],
            "candidate_record_count": report["recipe"]["sources"]["candidate_v389"]["rows"],
            "selected_pair_count": 195,
            "reserve_pair_count": 10,
            "panel_count": 5,
            "pairs_per_panel": 39,
            "signed_wave_count": 16,
            "perturbed_request_count_all_engines": 24960,
            "full_context_request_count_all_engines": 4680,
            "total_generation_request_count": 29640,
            "runtime_environment_content_sha256": (
                RUNTIME_ENVIRONMENT_CONTENT_SHA256_V30B
            ),
            "live_model_audit_content_sha256": LIVE_MODEL_AUDIT_CONTENT_SHA256_V30B,
            "runtime_audit_content_sha256": RUNTIME_AUDIT_CONTENT_SHA256_V30B,
            "restore_checks_sha256": report["runtime_audit"]["restore_checks_sha256"],
            "population_boundary_audit_sha256": report["runtime_audit"][
                "population_boundary_audit_sha256"
            ],
            "prelaunch_idle_certificate_sha256": PRELAUNCH_IDLE_SHA256_V30B,
            "final_idle_certificate_sha256": FINAL_IDLE_SHA256_V30B,
            "all_runtime_restore_boundary_preclaim_and_final_idle_integrity_passed": True,
        },
        "endpoint_gate_aggregates": copy.deepcopy(EXPECTED_ENDPOINTS_V30B),
        "aggregate_result": {
            "endpoint_count": 12,
            "negative_point_estimate_count": sum(item < 0 for item in points),
            "zero_point_estimate_count": sum(item == 0 for item in points),
            "positive_point_estimate_count": sum(item > 0 for item in points),
            "negative_familywise_lcb_count": sum(item < 0 for item in lcbs),
            "best_familywise_lcb": max(lcbs),
            "worst_familywise_lcb": min(lcbs),
            "all_familywise_lcbs_nonnegative": all(item >= 0 for item in lcbs),
            "gate_pass": report["gate"]["pass"],
        },
        "side_effects": {
            "model_mutation_applied": False,
            "checkpoint_written": False,
            "nontrain_data_surface_opened": False,
            "dataset_promotion_applied": False,
            "direct_action_taken": False,
        },
        "decision": {
            "retain_production_dataset": True,
            "retain_v13_recipe": True,
            "v389_replacement_authorized": False,
            "direct_followup_authority": False,
        },
        "raw_or_record_level_details_persisted": False,
        "nontrain_data_content_persisted": False,
    })
    _assert_compact_v30b(evidence)
    return evidence


def _exclusive_write_json_v30b(path, value):
    path = Path(path).resolve()
    if path != OUTPUT_PATH_V30B.resolve():
        raise ValueError("V30B evidence output path changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise RuntimeError("V30B immutable negative evidence already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH_V30B)
    args = parser.parse_args(argv)
    value = build_negative_evidence_v30b()
    if not args.dry_run:
        _exclusive_write_json_v30b(args.output, value)
    print(json.dumps({
        "schema": "eggroll-es-v30a-negative-evidence-build-v30b",
        "content_sha256": value["content_sha256_before_self_field"],
        "endpoint_count": value["aggregate_result"]["endpoint_count"],
        "gate_pass": value["aggregate_result"]["gate_pass"],
        "direct_followup_authority": value["decision"]["direct_followup_authority"],
    }, sort_keys=True))
    return value


if __name__ == "__main__":
    main()
