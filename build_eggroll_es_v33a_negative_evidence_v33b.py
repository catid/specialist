#!/usr/bin/env python3
"""Build compact immutable negative evidence for completed V33A."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ATTEMPT_RELATIVE_PATH_V33B = (
    "experiments/eggroll_es_hpo/runs/"
    ".s6_v33a_paired_production_v364_train_only_runtime_basis20261008."
    "launch_attempt.json"
)
REPORT_RELATIVE_PATH_V33B = (
    "experiments/eggroll_es_hpo/runs/"
    "s6_v33a_paired_production_v364_train_only_runtime_basis20261008/"
    "paired_production_v364_compatibility_v33a.json"
)
OUTPUT_RELATIVE_PATH_V33B = (
    "experiments/eggroll_es_hpo/"
    "S6_V33B_V33A_PAIRED_V364_NEGATIVE_EVIDENCE.json"
)
PREREGISTRATION_RELATIVE_PATH_V33B = (
    "experiments/eggroll_es_hpo/"
    "S6_PAIRED_DATA_COMPAT_V33A_PREREGISTRATION.json"
)
RUNTIME_RELATIVE_PATH_V33B = "run_eggroll_es_paired_data_compat_v33a.py"
ATTEMPT_PATH_V33B = ROOT / ATTEMPT_RELATIVE_PATH_V33B
REPORT_PATH_V33B = ROOT / REPORT_RELATIVE_PATH_V33B
OUTPUT_PATH_V33B = ROOT / OUTPUT_RELATIVE_PATH_V33B

ATTEMPT_FILE_SHA256_V33B = (
    "a8f41e2d90a9e9a92cecabdac402beb23a16e2b1a2746c166794c4ca5f81236e"
)
ATTEMPT_CONTENT_SHA256_V33B = (
    "894bd2393b7052c3881d9dad9a0e221d7779f6c3254a2d866a66a1f5bab22e20"
)
REPORT_FILE_SHA256_V33B = (
    "278b1900e71147d713a38ce554981b3a04dd38f7de031175f5b314b068356144"
)
REPORT_CONTENT_SHA256_V33B = (
    "b5fc3b27e0b9703c5f0b62f02fc513e3ce6f57a5cd7aad5788b048ad0c44210d"
)
PREREGISTRATION_AND_SOURCE_COMMIT_V33B = (
    "59d9f873f7281ab821e69b5f070527f67a37ce16"
)
CANDIDATE_INPUT_FREEZE_COMMIT_V33B = (
    "de0d5518f5cffe2ee71d8fc6884e506f3c1f3272"
)
PREREGISTRATION_FILE_SHA256_V33B = (
    "c83e11376922ac273b5b6496ef60126a2cdc6ae044f6a9ab05d9481dc539bcda"
)
PREREGISTRATION_CONTENT_SHA256_V33B = (
    "f053c686721401d08b31f2619ba23511159e0a85cfbda2df606e0b00fa98bc61"
)
RUNTIME_FILE_SHA256_V33B = (
    "4197fa48d9d719b55548bbdccbff6c22eeb7b0e6dec359bbff007463ad36eb38"
)
IMPLEMENTATION_BUNDLE_SHA256_V33B = (
    "a37ec4758df1ab8fbf92229ecefef8515f1243667ae84ee88ebcbe184939a444"
)
RECIPE_CONTENT_SHA256_V33B = (
    "045bc6b732d4ba1a1c4096d473a5d8f4fb953b47a0bc2e227f4bb45536e8f429"
)
SOURCE_PROVENANCE_CONTENT_SHA256_V33B = (
    "9ae915e05b77033f889094a09a88c3d7227cf7810613ff7295fe0cd8f026d76b"
)
COMMITTED_CLEAN_SOURCE_CONTENT_SHA256_V33B = (
    "6b4432404875ba6c2a3611f97d07a4e2ffac1e6039fa29ec365b8687415cff93"
)
RUNTIME_ENVIRONMENT_CONTENT_SHA256_V33B = (
    "7acc87ffb6a50682f19309ea3059823d0fc96dfc459e30cb66d64e157a76f52a"
)
LIVE_MODEL_AUDIT_CONTENT_SHA256_V33B = (
    "5950046f10d32625cef45f7bce58f3dee87171c59900deb396333b97e0854c1e"
)
CONFIGURATION_CONTENT_SHA256_V33B = (
    "5b0736a6245dfd036ea1ddcff9e6fb99258df82b1e44a068fc172f4f6007bd26"
)
SUMMARY_CONTENT_SHA256_V33B = (
    "a0fa615ae1d050a14792a5a8491f24f6fa87713a65d5e1859f6ad639fd6f17f8"
)
RUNTIME_AUDIT_CONTENT_SHA256_V33B = (
    "1130d6546aebfb3aaac17493cc2d735d3928c2692e62c177ae6413858c7e2475"
)
GATE_CONTENT_SHA256_V33B = (
    "fcf0f66da80a52512b4f020e89c1610ff4304c20577d013dfc415fc58c3038cb"
)
PRELAUNCH_IDLE_SHA256_V33B = (
    "63436f339c5aa2430289d200b59ecd6043298981e79dcb7c89d0346be8a78de6"
)
FINAL_IDLE_SHA256_V33B = (
    "08e2eb8f27dccc599d217bf98b4fe4f5744320ecc62c8b60dd017720ac3e0242"
)

EXPECTED_ENDPOINTS_V33B = {
    "aggregate_to_optimization_cosine_median": {
        "candidate_v364_minus_production": -0.01369167726412035,
        "familywise_lcb": -0.1732666731165566,
    },
    "aggregate_to_optimization_cosine_worst": {
        "candidate_v364_minus_production": 0.06199547250387827,
        "familywise_lcb": -0.2064678054829425,
    },
    "aggregate_to_optimization_sign_agreement_median": {
        "candidate_v364_minus_production": 0.03125,
        "familywise_lcb": -0.078125,
    },
    "aggregate_to_optimization_sign_agreement_worst": {
        "candidate_v364_minus_production": 0.078125,
        "familywise_lcb": -0.109375,
    },
    "optimization_pairwise_cosine_median": {
        "candidate_v364_minus_production": 0.12113794170335956,
        "familywise_lcb": -0.1899069162463626,
    },
    "optimization_pairwise_cosine_worst": {
        "candidate_v364_minus_production": 0.12182987137579881,
        "familywise_lcb": -0.22716142938348097,
    },
    "optimization_pairwise_sign_agreement_median": {
        "candidate_v364_minus_production": 0.109375,
        "familywise_lcb": -0.109375,
    },
    "optimization_pairwise_sign_agreement_worst": {
        "candidate_v364_minus_production": 0.109375,
        "familywise_lcb": -0.125,
    },
    "train_screen_cosine_median": {
        "candidate_v364_minus_production": 0.14060859242255336,
        "familywise_lcb": -0.14027310456822809,
    },
    "train_screen_cosine_worst": {
        "candidate_v364_minus_production": 0.11089929984636981,
        "familywise_lcb": -0.2749971566674678,
    },
    "train_screen_sign_agreement_median": {
        "candidate_v364_minus_production": 0.078125,
        "familywise_lcb": -0.1015625,
    },
    "train_screen_sign_agreement_worst": {
        "candidate_v364_minus_production": 0.046875,
        "familywise_lcb": -0.140625,
    },
}
EXPECTED_PRODUCTION_ENDPOINTS_V33B = {
    "aggregate_to_optimization_cosine_median": 0.7077088060703811,
    "aggregate_to_optimization_cosine_worst": 0.579653575032599,
    "aggregate_to_optimization_sign_agreement_median": 0.78125,
    "aggregate_to_optimization_sign_agreement_worst": 0.6875,
    "optimization_pairwise_cosine_median": 0.22721332690112567,
    "optimization_pairwise_cosine_worst": 0.1592912207444499,
    "optimization_pairwise_sign_agreement_median": 0.515625,
    "optimization_pairwise_sign_agreement_worst": 0.46875,
    "train_screen_cosine_median": 0.3179422250247281,
    "train_screen_cosine_worst": 0.27239272085907845,
    "train_screen_sign_agreement_median": 0.59375,
    "train_screen_sign_agreement_worst": 0.546875,
}
EXPECTED_CANDIDATE_ENDPOINTS_V33B = {
    "aggregate_to_optimization_cosine_median": 0.6940171288062608,
    "aggregate_to_optimization_cosine_worst": 0.6416490475364772,
    "aggregate_to_optimization_sign_agreement_median": 0.8125,
    "aggregate_to_optimization_sign_agreement_worst": 0.765625,
    "optimization_pairwise_cosine_median": 0.34835126860448523,
    "optimization_pairwise_cosine_worst": 0.2811210921202487,
    "optimization_pairwise_sign_agreement_median": 0.625,
    "optimization_pairwise_sign_agreement_worst": 0.578125,
    "train_screen_cosine_median": 0.45855081744728143,
    "train_screen_cosine_worst": 0.38329202070544827,
    "train_screen_sign_agreement_median": 0.671875,
    "train_screen_sign_agreement_worst": 0.59375,
}
EXPECTED_RUNTIME_INTEGRITY_KEYS_V33B = {
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
FORBIDDEN_EVIDENCE_TERMS_V33B = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "token", "tokens", "response", "responses", "row", "rows",
    "record", "records", "bootstrap", "draw", "draws", "replicate",
    "replicates", "heldout", "validation", "ood", "benchmark",
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
    _require(path.is_file() and not path.is_symlink(), f"V33B {label} path changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"V33B {label} must be a JSON object")
    return value


def _verify_self(value, expected, label):
    _require(
        value.get("content_sha256_before_self_field") == expected
        and canonical_sha256(_without_self(value)) == expected,
        f"V33B {label} self hash changed",
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


def _assert_compact_v33b(value):
    offenders = []
    for text in _all_recursive_strings(value):
        words = set(text.replace("-", "_").replace("/", "_").split("_"))
        overlap = words & FORBIDDEN_EVIDENCE_TERMS_V33B
        if overlap:
            offenders.extend(sorted(overlap))
    if offenders:
        raise RuntimeError(
            f"V33B evidence contains forbidden detail terms: {sorted(set(offenders))}"
        )


def _git_show(commit, relative_path):
    return subprocess.check_output(
        ["git", "show", f"{commit}:{relative_path}"], cwd=ROOT,
    )


def validate_history_v33b():
    _require(
        hashlib.sha256(_git_show(
            PREREGISTRATION_AND_SOURCE_COMMIT_V33B,
            PREREGISTRATION_RELATIVE_PATH_V33B,
        )).hexdigest() == PREREGISTRATION_FILE_SHA256_V33B,
        "V33B committed preregistration history changed",
    )
    _require(
        hashlib.sha256(_git_show(
            PREREGISTRATION_AND_SOURCE_COMMIT_V33B,
            RUNTIME_RELATIVE_PATH_V33B,
        )).hexdigest() == RUNTIME_FILE_SHA256_V33B,
        "V33B committed runtime history changed",
    )


def _validate_source_and_implementation_v33b(attempt, report):
    provenance = attempt.get("source_provenance", {})
    clean = attempt.get("committed_clean_source_certificate", {})
    implementation = report.get("implementation", {})
    _verify_self(
        provenance, SOURCE_PROVENANCE_CONTENT_SHA256_V33B, "source provenance",
    )
    _verify_self(
        clean, COMMITTED_CLEAN_SOURCE_CONTENT_SHA256_V33B,
        "committed-clean source certificate",
    )
    _require(
        provenance.get("git_head") == PREREGISTRATION_AND_SOURCE_COMMIT_V33B
        and clean.get("git_head") == PREREGISTRATION_AND_SOURCE_COMMIT_V33B
        and clean.get("all_tracked_files_clean") is True
        and clean.get("only_explicitly_allowlisted_untracked_paths_present") is True
        and clean.get("allowed_untracked_entry_count") == 8
        and provenance.get("implementation_bundle_sha256")
        == IMPLEMENTATION_BUNDLE_SHA256_V33B
        and implementation.get("bundle_sha256")
        == IMPLEMENTATION_BUNDLE_SHA256_V33B
        and report.get("committed_clean_source_certificate_sha256")
        == COMMITTED_CLEAN_SOURCE_CONTENT_SHA256_V33B,
        "V33B clean launch source or implementation binding changed",
    )
    source_files = provenance.get("files", {})
    implementation_files = implementation.get("files", {})
    _require(
        len(source_files) == 68
        and len(implementation_files) == 68
        and len(implementation.get("immutable_bound_files", {})) == 12
        and canonical_sha256(implementation_files)
        == IMPLEMENTATION_BUNDLE_SHA256_V33B
        and implementation.get("inherited_v23a_r2_bundle_sha256")
        == "4bbd31dbdb61366d6ca61c8f5955df3e59e1b34642cd12aca84b54153586d6a6"
        and implementation.get("v33a_overlay_bundle_sha256")
        == "3ea2cf4ea28609ee6ffe315ddf9f21ce000000d5334661ecc7f4e3e8f0c01ddd",
        "V33B source implementation cardinality or aggregate hash changed",
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
        "V33B source provenance and implementation files disagree",
    )
    recipe = report.get("recipe", {})
    _verify_self(recipe, RECIPE_CONTENT_SHA256_V33B, "runtime recipe")
    _require(
        attempt.get("recipe") == recipe
        and recipe.get("implementation_bundle_sha256")
        == IMPLEMENTATION_BUNDLE_SHA256_V33B
        and recipe.get("preregistration", {}).get("file_sha256")
        == PREREGISTRATION_FILE_SHA256_V33B
        and recipe.get("preregistration", {}).get("content_sha256")
        == PREREGISTRATION_CONTENT_SHA256_V33B
        and recipe.get("preregistration", {}).get("candidate_freeze_commit")
        == CANDIDATE_INPUT_FREEZE_COMMIT_V33B,
        "V33B attempt report recipe or candidate freeze binding changed",
    )


def _validate_runtime_and_cardinality_v33b(attempt, report):
    environment = attempt.get("runtime_environment_certificate", {})
    live_model = attempt.get("live_model_audit", {})
    configuration = report.get("configuration", {})
    summary = report.get("summary", {})
    audit = report.get("runtime_audit", {})
    _verify_self(
        environment, RUNTIME_ENVIRONMENT_CONTENT_SHA256_V33B,
        "runtime environment",
    )
    _verify_self(live_model, LIVE_MODEL_AUDIT_CONTENT_SHA256_V33B, "live model")
    _verify_self(configuration, CONFIGURATION_CONTENT_SHA256_V33B, "configuration")
    _verify_self(summary, SUMMARY_CONTENT_SHA256_V33B, "summary")
    _verify_self(audit, RUNTIME_AUDIT_CONTENT_SHA256_V33B, "runtime audit")
    recipe = report["recipe"]
    accounting = recipe.get("request_accounting", {})
    frame = recipe.get("frame", {})
    panels = recipe.get("panels", {})
    perturbation = recipe.get("perturbation", {})
    _require(
        environment.get("completed_before_attempt_claim") is True
        and environment.get("cuda_device_count") == 4
        and environment.get("cuda_visible_devices") == "0,1,2,3"
        and environment.get("dataset_or_evaluation_surface_opened") is False
        and live_model.get("config_sha256")
        == "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
        and live_model.get("index_sha256")
        == "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
        "V33B runtime environment or live model integrity changed",
    )
    _require(
        frame.get("selected_paired_units") == 195
        and frame.get("reserve_paired_units") == 10
        and frame.get("shared_document_anchors") == 193
        and frame.get("joint_component_cross_side_anchors") == 2
        and frame.get("globally_disjoint_joint_units") is True
        and len(panels) == 5
        and all(item.get("paired_units") == 39 for item in panels.values())
        and perturbation.get("population_size") == 64
        and perturbation.get("engine_signed_direction_evaluations") == 128
        and perturbation.get("synchronized_four_engine_signed_waves") == 32
        and perturbation.get("direction_seed_list_sha256")
        == "4227e7c741175eb29f10c73b70f40e4442ebc6f2ca3d9f798dc7639cfe5a8e5f"
        and accounting == {
            "engine_signed_direction_evaluations": 128,
            "full_context_phase_count": 3,
            "full_context_requests_all_engines": 4680,
            "full_context_requests_all_engines_per_phase": 1560,
            "paired_units_per_panel": 39,
            "panels": 5,
            "perturbed_requests_all_engines": 49920,
            "requests_all_engines_per_signed_wave": 1560,
            "requests_per_engine_per_signed_wave": 390,
            "requests_per_version_per_engine_call": 195,
            "synchronized_four_engine_signed_waves": 32,
            "total_generation_requests": 54600,
            "versions_per_signed_wave": 2,
        },
        "V33B durable panel direction or request cardinality changed",
    )
    runtime_integrity = summary.get("runtime_integrity", {})
    guard = audit.get("full_context_guard", {})
    _require(
        set(runtime_integrity) == EXPECTED_RUNTIME_INTEGRITY_KEYS_V33B
        and all(value is True for value in runtime_integrity.values())
        and audit.get("engine_signed_direction_evaluation_count") == 128
        and audit.get("synchronized_four_engine_signed_wave_count") == 32
        and audit.get("perturbed_requests_all_engines") == 49920
        and audit.get("full_context_requests_all_engines") == 4680
        and audit.get("total_generation_requests") == 54600
        and audit.get("per_unit_scores_or_bootstrap_replicates_persisted") is False
        and audit.get("restore_checks_sha256")
        == "cc30e99c9412ed683c342e00bf659bb9e49f4ba8dc6cca13d6d0f9ae177ae903"
        and audit.get("population_boundary_audit_sha256")
        == "0fafb9834bbc724795b72a2bfa7da36c3539f3ee093995180f8c3f7bd2087988"
        and audit.get("signed_wave_schedule_sha256")
        == "a59438d09a5a186ad652aa9d3fed25cec758d3752a5d03940ea98b317f1763f5"
        and audit.get("token_audit_sha256")
        == "55f9923c866930905e019135eeeb3ab341d4ce7ef79511a790ae64ed73094569"
        and audit.get("fixed_request_identity_sha256")
        == "df35fdac75d6a71f64cba9ab2589c5254e529ea4d9cb1985b679a8454f4c5644"
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
        "V33B restore boundary or full-context runtime integrity changed",
    )
    return summary


def _validate_gate_v33b(report, summary):
    gate = report.get("gate", {})
    _verify_self(gate, GATE_CONTENT_SHA256_V33B, "gate")
    paired = summary.get("paired_bootstrap", {})
    endpoints = paired.get("endpoints", {})
    normalized = {
        name: {
            "candidate_v364_minus_production": item.get(
                "candidate_v364_minus_production"
            ),
            "familywise_lcb": item.get("familywise_lcb"),
        }
        for name, item in endpoints.items()
    }
    _require(
        paired.get("repetitions") == 50_000
        and paired.get("seed") == 20_261_009
        and paired.get("one_sided_quantile") == 0.05 / 12
        and paired.get("raw_draws_or_replicates_persisted") is False
        and set(endpoints) == set(EXPECTED_ENDPOINTS_V33B)
        and all(item.get("noninferiority_margin") == 0.0 for item in endpoints.values())
        and normalized == EXPECTED_ENDPOINTS_V33B,
        "V33B exact 12 endpoint aggregates changed",
    )
    versions = summary.get("versions", {})
    _require(
        versions.get("production", {}).get("endpoint_values")
        == EXPECTED_PRODUCTION_ENDPOINTS_V33B
        and versions.get("candidate_v364", {}).get("endpoint_values")
        == EXPECTED_CANDIDATE_ENDPOINTS_V33B
        and versions.get("production", {}).get("all_panel_spreads_nonzero") is True
        and versions.get("candidate_v364", {}).get("all_panel_spreads_nonzero")
        is True,
        "V33B exact production or candidate point endpoints changed",
    )
    points = [
        item["candidate_v364_minus_production"]
        for item in EXPECTED_ENDPOINTS_V33B.values()
    ]
    lcbs = [item["familywise_lcb"] for item in EXPECTED_ENDPOINTS_V33B.values()]
    _require(
        sum(item > 0 for item in points) == 11
        and sum(item == 0 for item in points) == 0
        and sum(item < 0 for item in points) == 1
        and sum(item < 0 for item in lcbs) == 12
        and max(lcbs) == -0.078125
        and min(lcbs) == -0.2749971566674678
        and gate.get("pass") is False
        and gate.get("all_12_familywise_lcbs_nonnegative") is False
        and gate.get("all_12_observed_point_deltas_nonnegative") is False
        and gate.get("all_runtime_integrity_audits_passed") is True
        and gate.get("decision") == "retain_production_dataset_and_v13_recipe"
        and gate.get("checkpoint_write_authorized") is False
        and gate.get("dataset_promotion_authorized") is False
        and gate.get("evaluation_authorized") is False
        and gate.get("model_update_authorized") is False,
        "V33B negative gate recomputation or decision changed",
    )


def _validate_closed_side_effects_v33b(attempt, report):
    _require(
        attempt.get("schema") == "eggroll-es-durable-launch-attempt-v33a"
        and attempt.get("status") == "complete"
        and attempt.get("phase") == "after_compact_report_and_final_gpu_cleanup"
        and report.get("schema") == "eggroll-es-paired-data-compat-report-v33a"
        and attempt.get("prelaunch_idle_certificate_sha256")
        == report.get("prelaunch_idle_certificate_sha256")
        == PRELAUNCH_IDLE_SHA256_V33B
        and attempt.get("final_idle_certificate_sha256")
        == report.get("final_idle_certificate_sha256")
        == FINAL_IDLE_SHA256_V33B
        and attempt.get("all_four_gpus_idle_after_cleanup") is True
        and report.get("all_four_gpus_idle_after_cleanup") is True,
        "V33B durable completion preclaim or final-idle integrity changed",
    )
    for value in (attempt, report):
        _require(
            value.get("checkpoint_written") is False
            and value.get("dataset_promotion_applied") is False
            and value.get("evaluation_opened") is False
            and value.get("model_update_applied") is False,
            "V33B forbidden mutation checkpoint evaluation or promotion changed",
        )
    _require(
        report.get("direct_action_taken") is False
        and report.get("configuration", {}).get("checkpoint_write_allowed") is False
        and report.get("configuration", {}).get("evaluation_allowed") is False
        and report.get("configuration", {}).get("model_update_allowed") is False
        and report.get("summary", {}).get(
            "persisted_response_vectors_rows_draws_or_replicates"
        ) is False
        and report.get("summary", {}).get("evaluation_opened") is False
        and report.get("summary", {}).get("model_update_applied") is False
        and report.get("runtime_audit", {}).get("evaluation_opened") is False
        and report.get("runtime_audit", {}).get("model_update_applied") is False,
        "V33B direct action configuration or persistence boundary changed",
    )


def validate_bound_artifacts_v33b():
    _require(
        file_sha256(ATTEMPT_PATH_V33B) == ATTEMPT_FILE_SHA256_V33B,
        "V33B attempt file hash changed",
    )
    _require(
        file_sha256(REPORT_PATH_V33B) == REPORT_FILE_SHA256_V33B,
        "V33B report file hash changed",
    )
    attempt = _load_json_object(ATTEMPT_PATH_V33B, "attempt")
    report = _load_json_object(REPORT_PATH_V33B, "report")
    _verify_self(attempt, ATTEMPT_CONTENT_SHA256_V33B, "attempt")
    _verify_self(report, REPORT_CONTENT_SHA256_V33B, "report")
    _require(
        attempt.get("report_binding") == {
            "path": str(REPORT_PATH_V33B.resolve()),
            "file_sha256": REPORT_FILE_SHA256_V33B,
            "content_sha256": REPORT_CONTENT_SHA256_V33B,
        },
        "V33B attempt report binding changed",
    )
    validate_history_v33b()
    _validate_source_and_implementation_v33b(attempt, report)
    summary = _validate_runtime_and_cardinality_v33b(attempt, report)
    _validate_gate_v33b(report, summary)
    _validate_closed_side_effects_v33b(attempt, report)
    return attempt, report


def build_negative_evidence_v33b():
    _attempt, report = validate_bound_artifacts_v33b()
    points = [
        item["candidate_v364_minus_production"]
        for item in EXPECTED_ENDPOINTS_V33B.values()
    ]
    lcbs = [item["familywise_lcb"] for item in EXPECTED_ENDPOINTS_V33B.values()]
    value = _seal({
        "schema": "eggroll-es-v33a-paired-v364-negative-evidence-v33b",
        "status": "valid_completed_negative_gate_no_action",
        "artifacts": {
            "durable_attempt": {
                "relative_path": ATTEMPT_RELATIVE_PATH_V33B,
                "file_sha256": ATTEMPT_FILE_SHA256_V33B,
                "content_sha256": ATTEMPT_CONTENT_SHA256_V33B,
            },
            "compact_report": {
                "relative_path": REPORT_RELATIVE_PATH_V33B,
                "file_sha256": REPORT_FILE_SHA256_V33B,
                "content_sha256": REPORT_CONTENT_SHA256_V33B,
            },
        },
        "source_history": {
            "preregistration_and_clean_launch_source_commit": (
                PREREGISTRATION_AND_SOURCE_COMMIT_V33B
            ),
            "candidate_input_freeze_commit": CANDIDATE_INPUT_FREEZE_COMMIT_V33B,
            "preregistration_file_sha256": PREREGISTRATION_FILE_SHA256_V33B,
            "preregistration_content_sha256": PREREGISTRATION_CONTENT_SHA256_V33B,
            "runtime_file_sha256": RUNTIME_FILE_SHA256_V33B,
            "implementation_bundle_sha256": IMPLEMENTATION_BUNDLE_SHA256_V33B,
            "recipe_content_sha256": RECIPE_CONTENT_SHA256_V33B,
            "source_provenance_content_sha256": SOURCE_PROVENANCE_CONTENT_SHA256_V33B,
            "committed_clean_source_content_sha256": (
                COMMITTED_CLEAN_SOURCE_CONTENT_SHA256_V33B
            ),
            "implementation_file_count": 68,
            "immutable_bound_file_count": 12,
        },
        "aggregate_execution": {
            "selected_pair_count": 195,
            "reserve_pair_count": 10,
            "panel_count": 5,
            "pairs_per_panel": 39,
            "direction_count": 64,
            "engine_signed_direction_evaluation_count": 128,
            "synchronized_four_engine_signed_wave_count": 32,
            "perturbed_request_count_all_engines": 49920,
            "full_context_request_count_all_engines": 4680,
            "total_generation_request_count": 54600,
            "runtime_environment_content_sha256": (
                RUNTIME_ENVIRONMENT_CONTENT_SHA256_V33B
            ),
            "live_model_audit_content_sha256": LIVE_MODEL_AUDIT_CONTENT_SHA256_V33B,
            "configuration_content_sha256": CONFIGURATION_CONTENT_SHA256_V33B,
            "summary_content_sha256": SUMMARY_CONTENT_SHA256_V33B,
            "runtime_audit_content_sha256": RUNTIME_AUDIT_CONTENT_SHA256_V33B,
            "gate_content_sha256": GATE_CONTENT_SHA256_V33B,
            "restore_checks_sha256": report["runtime_audit"]["restore_checks_sha256"],
            "population_boundary_audit_sha256": report["runtime_audit"][
                "population_boundary_audit_sha256"
            ],
            "prelaunch_idle_certificate_sha256": PRELAUNCH_IDLE_SHA256_V33B,
            "final_idle_certificate_sha256": FINAL_IDLE_SHA256_V33B,
            "all_runtime_integrity_guards_and_idle_boundaries_passed": True,
        },
        "aggregate_result": {
            "endpoint_count": 12,
            "negative_point_estimate_count": sum(item < 0 for item in points),
            "zero_point_estimate_count": sum(item == 0 for item in points),
            "positive_point_estimate_count": sum(item > 0 for item in points),
            "negative_familywise_lcb_count": sum(item < 0 for item in lcbs),
            "best_familywise_lcb": max(lcbs),
            "worst_familywise_lcb": min(lcbs),
            "all_observed_point_deltas_nonnegative": all(item >= 0 for item in points),
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
            "v364_replacement_authorized": False,
            "direct_followup_authority": False,
            "direct_confirmation_authority": False,
            "dataset_promotion_authority": False,
            "model_update_authority": False,
            "checkpoint_write_authority": False,
            "evaluation_authority": False,
            "nontrain_data_surface_authority": False,
        },
        "detailed_payloads_persisted": False,
        "nontrain_data_content_persisted": False,
    })
    _assert_compact_v33b(value)
    return value


def _exclusive_write_json_v33b(path, value):
    path = Path(path).resolve()
    if path != OUTPUT_PATH_V33B.resolve():
        raise ValueError("V33B evidence output path changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise RuntimeError("V33B immutable negative evidence already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH_V33B)
    args = parser.parse_args(argv)
    value = build_negative_evidence_v33b()
    if not args.dry_run:
        _exclusive_write_json_v33b(args.output, value)
    print(json.dumps({
        "schema": "eggroll-es-v33a-negative-evidence-build-v33b",
        "content_sha256": value["content_sha256_before_self_field"],
        "endpoint_count": value["aggregate_result"]["endpoint_count"],
        "gate_pass": value["aggregate_result"]["gate_pass"],
        "direct_followup_authority": value["decision"]["direct_followup_authority"],
    }, sort_keys=True))
    return value


if __name__ == "__main__":
    main()
