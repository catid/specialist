#!/usr/bin/env python3
"""Build aggregate-only negative evidence for the completed V22A raw run."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN_ROOT_V22A = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "production_v341_matched_replacement_v22a_authoritative_raw"
).resolve()
FAILED_LAUNCH_ID_V22A = "1784101254443005006-391898-c54723bc0b653705"
COMPLETE_LAUNCH_ID_V22A = "1784101310448295149-393068-b321f6b74812720d"
FAILED_ATTEMPT_PATH_V22A = (
    RUN_ROOT_V22A / "attempts" / f"{FAILED_LAUNCH_ID_V22A}.json"
).resolve()
COMPLETE_ATTEMPT_PATH_V22A = (
    RUN_ROOT_V22A / "attempts" / f"{COMPLETE_LAUNCH_ID_V22A}.json"
).resolve()
REPORT_PATH_V22A = (
    RUN_ROOT_V22A / "runs" / COMPLETE_LAUNCH_ID_V22A /
    "raw_v341_matched_replacement_attribution_v22a.json"
).resolve()
OUTPUT_PATH_V22A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V22A_V341_MATCHED_REPLACEMENT_NEGATIVE_EVIDENCE.json"
).resolve()

LAUNCH_GIT_HEAD_V22A = "d105841049b4fc01f8252599ad5a1cf271c2c156"
FAILED_ATTEMPT_FILE_SHA256_V22A = (
    "0de50a97af21ef6ff50724f56f21a65af7a776598740200a79e7bcab11f890a7"
)
FAILED_ATTEMPT_CONTENT_SHA256_V22A = (
    "5bddce1d1b5c15f38bcbbcc950961ad9f9b9459278a055baba3612f5d669352b"
)
COMPLETE_ATTEMPT_FILE_SHA256_V22A = (
    "ef5c81c9901e3548e92b4ab967b8b4b26dc1b2b56b9881106b274d7e6ec24452"
)
COMPLETE_ATTEMPT_CONTENT_SHA256_V22A = (
    "8b3f1c1e8c460f5742f097c81a165fa9f94363e4bc819ff14badbd72886abd2a"
)
REPORT_FILE_SHA256_V22A = (
    "4c1baebfdc5e646afe751c3183541768b154c3d871984a2c01313d0016a0487a"
)
REPORT_CONTENT_SHA256_V22A = (
    "2f83f198ad0ec6fa943b8cdac21c8c4bb86a2906c62e22e750001e2324aca59d"
)
SUMMARY_CONTENT_SHA256_V22A = (
    "c9721078c74778a516af2c38aa611527ebee8af5458afd90d8e670418b63c8ff"
)
RUNTIME_AUDIT_CONTENT_SHA256_V22A = (
    "2e10c78194c5ab94e63b0a81eb6cdd1215c9ecc40af1ddbb97514f55530cec7e"
)
RECIPE_CONTENT_SHA256_V22A = (
    "d03028e4586321077942ab7e0168bf2c9284f9c385b68854d439f2df9487ae87"
)
SOURCE_PROVENANCE_CONTENT_SHA256_V22A = (
    "590f0d8e792d650f31a0928796c94e51cb6a651db86788c619bfb7e4b184b743"
)
IMPLEMENTATION_BUNDLE_SHA256_V22A = (
    "7485078a7c6306ba191b285800487e5ece67e20c68311076f5986f0dfd8ea1e7"
)
INHERITED_V21A_BUNDLE_SHA256_V22A = (
    "8e4babb3c5663fbe5368a19213c48c608a635d849e438dda17329c1105ed2104"
)
PREREG_BUNDLE_SHA256_V22A = (
    "f98769694992ff7ba2343aec6636c6af953bff26a79a8ef1f9d831396038a5f0"
)
MECHANICS_BUNDLE_SHA256_V22A = (
    "12f69f90c1dd4e9804999ffdb0618cdd9c340a68873378cba8c724febfc41b04"
)
SOURCE_FILE_COUNT_V22A = 69
SOURCE_FILE_BINDINGS_SHA256_V22A = (
    "4e3d966cf7f6d240adca64facac74e5aca79f7d3145ba44d1d7625e0ee0a4920"
)
SIGNED_WAVE_SCHEDULE_SHA256_V22A = (
    "3b7849d6eb1fade0ee2c3a3b6247cc5d929fcd9e91c263f30b0538c3fa80575d"
)
DRAW_PLAN_CONTENT_SHA256_V22A = (
    "bb7fb2d5ca147142c0a8406fbe929944d964452deaeb2378f7a7286988fb7b2e"
)
PANEL_BUNDLE_CONTENT_SHA256_V22A = (
    "bda020933ff7b6abf3c9dd21e79e743d66bdf72f1ca23dbd31b5a96f9f571b0e"
)
GATE_SHA256_V22A = (
    "cf5e025cfc18b8d57e0e4c94bfb48571a72059a6d3ec8eedcdd3b7b55ecbd389"
)
ENDPOINT_PASS_STATUS_SHA256_V22A = (
    "4623fd723ebb6e260da3b4703bb13416c0efa76c33eab3cefe854a6ba695e519"
)
ARM_ESTIMATOR_COMMITMENTS_SHA256_V22A = (
    "8e3a70a1d7ae4b6314c6b637bd6edf0bfb7ae8f35c2ec91855c40bba5bb73220"
)

ENDPOINT_NAMES_V22A = (
    "optimization_pairwise_cosine_median",
    "optimization_pairwise_cosine_worst",
    "optimization_pairwise_sign_agreement_median",
    "optimization_pairwise_sign_agreement_worst",
    "aggregate_to_optimization_cosine_median",
    "aggregate_to_optimization_cosine_worst",
    "aggregate_to_optimization_sign_agreement_median",
    "aggregate_to_optimization_sign_agreement_worst",
    "train_screen_cosine_median",
    "train_screen_cosine_worst",
    "train_screen_sign_agreement_median",
    "train_screen_sign_agreement_worst",
)
ARMS_V22A = ("production_control", "v341_matched_replacement")
RUNTIME_INTEGRITY_KEYS_V22A = {
    "all_four_tp1_engines_every_signed_wave",
    "all_integrity_audits_passed",
    "all_ten_panels_every_direction_sign_and_arm",
    "all_thirty_two_signed_waves_complete",
    "counterbalanced_arm_order_complete",
    "exact_reference_restored_once_per_signed_wave",
    "population_boundary_audit_passed",
    "pre_post_raw_reference_probes_equal",
    "same_resident_perturbation_both_arms",
    "union_planner_called",
    "unselected_origin_audit_passed",
}
FORBIDDEN_CONTENT_KEYS_V22A = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "prompt_token_ids", "token_ids", "tokens", "row_sha256",
    "ordered_row_identity_sha256", "joint_ids",
    "ordered_joint_identity_sha256", "unit_scores", "responses",
    "coefficients", "bootstrap_replicates", "bootstrap_draws", "row_content",
    "heldout", "holdout", "validation", "ood", "benchmark", "eval",
}
SIDE_EFFECT_KEYS_V22A = (
    "union_planner_called", "model_update_applied", "checkpoint_written",
    "evaluation_surfaces_opened", "dataset_promotion_applied",
)
COMPLETE_ATTEMPT_KEYS_V22A = {
    "schema", "status", "phase", "experiment_name", "recipe",
    "source_provenance", *SIDE_EFFECT_KEYS_V22A, "launch_id", "run_directory",
    "report_exists_after_attempt", "report_binding",
    "content_sha256_before_self_field",
}
FAILED_ATTEMPT_KEYS_V22A = (
    COMPLETE_ATTEMPT_KEYS_V22A - {"report_binding"}
    | {"failure_type", "failure_sha256"}
)
REPORT_KEYS_V22A = {
    "schema", "recipe", "configuration", "runtime_audit", "summary", "gate",
    "implementation", *SIDE_EFFECT_KEYS_V22A, "content_sha256_before_self_field",
}
SUMMARY_KEYS_V22A = {
    "schema", "experiment_name", "alpha", "sigma", "runtime_integrity",
    "arms", "paired_bootstrap", *SIDE_EFFECT_KEYS_V22A,
    "persisted_response_vectors_or_row_content", "bootstrap_draws_persisted",
    "unit_scores_persisted", "content_sha256_before_self_field",
}
AUDIT_KEYS_V22A = {
    "schema", "fixed_request_identity_sha256", "token_boundary_audit_sha256",
    "pre_post_probe_identity_sha256", "signed_wave_schedule_sha256",
    "restore_checks_sha256", "dense_result_commitments_sha256",
    "population_boundary_audit_sha256", "unselected_origin_sha256",
    "unselected_origin_audit_sha256", "signed_wave_count", "panel_count",
    "requests_per_engine_per_signed_wave", "requests_per_engine_all_signed_waves",
    "requests_all_engines_all_signed_waves", "dense_result_commitment_count",
    "union_planner_called", "per_unit_scores_persisted",
    "bootstrap_replicates_persisted", "bootstrap_draws_persisted",
    "row_content_persisted", "content_sha256_before_self_field",
}
CONFIGURATION_KEYS_V22A = {
    "schema", "layer_plan_install_sha256", "reference_identity_sha256",
    "unselected_origin_sha256", "panel_bundle_content_sha256", "engine_count",
    "tp_per_engine", "gpu_ids", "union_planner_called",
    "train_only_raw_runtime_opened", "model_update_allowed",
    "checkpoint_write_allowed", "evaluation_surfaces_opened",
    "dataset_promotion_allowed",
}


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _is_digest(value):
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_finite_number(value):
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _recursive_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).lower()
            yield from _recursive_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _recursive_keys(item)


def _load_exact_run_files_v22a():
    attempts = sorted((RUN_ROOT_V22A / "attempts").glob("*.json"))
    reports = sorted((RUN_ROOT_V22A / "runs").glob("*/*.json"))
    run_dirs = sorted(path for path in (RUN_ROOT_V22A / "runs").iterdir() if path.is_dir())
    all_files = sorted(path for path in RUN_ROOT_V22A.rglob("*") if path.is_file())
    top_dirs = {path.name for path in RUN_ROOT_V22A.iterdir() if path.is_dir()}
    expected_files = [
        FAILED_ATTEMPT_PATH_V22A, COMPLETE_ATTEMPT_PATH_V22A, REPORT_PATH_V22A,
    ]
    expected_run_dirs = [
        RUN_ROOT_V22A / "runs" / FAILED_LAUNCH_ID_V22A,
        REPORT_PATH_V22A.parent,
    ]
    _require(
        attempts == expected_files[:2]
        and reports == [REPORT_PATH_V22A]
        and run_dirs == sorted(expected_run_dirs)
        and all_files == sorted(expected_files)
        and top_dirs == {"attempts", "checkpoints", "eval-output", "runs"},
        "v22a durable run cardinality changed",
    )
    _require(
        not any((RUN_ROOT_V22A / "checkpoints").iterdir())
        and not any((RUN_ROOT_V22A / "eval-output").iterdir())
        and not any(expected_run_dirs[0].iterdir())
        and not any(path.is_symlink() for path in RUN_ROOT_V22A.rglob("*")),
        "v22a forbidden durable side-effect path appeared",
    )
    _require(
        file_sha256(FAILED_ATTEMPT_PATH_V22A) == FAILED_ATTEMPT_FILE_SHA256_V22A
        and file_sha256(COMPLETE_ATTEMPT_PATH_V22A)
        == COMPLETE_ATTEMPT_FILE_SHA256_V22A
        and file_sha256(REPORT_PATH_V22A) == REPORT_FILE_SHA256_V22A,
        "v22a durable file bytes changed",
    )
    return tuple(
        json.loads(path.read_text(encoding="utf-8")) for path in expected_files
    )


def _validate_source_provenance_v22a(
    failed, complete, report, *, require_frozen_hashes, validate_launch_sources,
):
    source = complete.get("source_provenance", {})
    implementation = report.get("implementation", {})
    _require(
        failed.get("source_provenance") == source
        and set(source) == {
            "schema", "git_head", "files", "implementation_bundle_sha256",
            "content_sha256_before_self_field",
        }
        and source.get("schema") == "eggroll-es-raw-source-provenance-v22a"
        and source.get("git_head") == LAUNCH_GIT_HEAD_V22A
        and source.get("content_sha256_before_self_field")
        == canonical_sha256(_without_self(source))
        and set(implementation) == {
            "files", "inherited_v21a_bundle_sha256", "prereg_bundle_sha256",
            "mechanics_bundle_sha256", "bundle_sha256",
        }
        and implementation.get("inherited_v21a_bundle_sha256")
        == INHERITED_V21A_BUNDLE_SHA256_V22A
        and implementation.get("prereg_bundle_sha256")
        == PREREG_BUNDLE_SHA256_V22A
        and implementation.get("mechanics_bundle_sha256")
        == MECHANICS_BUNDLE_SHA256_V22A,
        "v22a source provenance or implementation envelope changed",
    )
    source_files = source.get("files", {})
    implementation_files = implementation.get("files", {})
    _require(
        isinstance(source_files, dict) and isinstance(implementation_files, dict)
        and len(source_files) == len(implementation_files) == SOURCE_FILE_COUNT_V22A
        and set(source_files) == set(implementation_files)
        and canonical_sha256(source_files) == SOURCE_FILE_BINDINGS_SHA256_V22A
        and implementation.get("bundle_sha256")
        == canonical_sha256(implementation_files)
        and source.get("implementation_bundle_sha256")
        == implementation.get("bundle_sha256"),
        "v22a implementation bundle or source coverage changed",
    )
    for key in sorted(implementation_files):
        implementation_item = implementation_files[key]
        source_item = source_files[key]
        _require(
            set(implementation_item) == {"path", "file_sha256"}
            and set(source_item) == {"relative_path", "file_sha256"}
            and _is_digest(implementation_item.get("file_sha256"))
            and source_item.get("file_sha256")
            == implementation_item.get("file_sha256"),
            "v22a individual source binding changed",
        )
        path = Path(implementation_item["path"]).resolve()
        try:
            relative = path.relative_to(ROOT).as_posix()
        except ValueError as error:
            raise RuntimeError("v22a source path escaped repository") from error
        _require(
            source_item.get("relative_path") == relative,
            "v22a relative source path binding changed",
        )
        if validate_launch_sources:
            try:
                raw = subprocess.check_output(
                    ["git", "show", f"{LAUNCH_GIT_HEAD_V22A}:{relative}"], cwd=ROOT
                )
            except subprocess.CalledProcessError as error:
                raise RuntimeError("v22a launch source is missing") from error
            _require(
                hashlib.sha256(raw).hexdigest() == implementation_item["file_sha256"]
                and path.is_file()
                and file_sha256(path) == implementation_item["file_sha256"],
                "v22a launch source file digest changed",
            )
    if require_frozen_hashes:
        _require(
            source["content_sha256_before_self_field"]
            == SOURCE_PROVENANCE_CONTENT_SHA256_V22A
            and implementation["bundle_sha256"] == IMPLEMENTATION_BUNDLE_SHA256_V22A,
            "v22a frozen source provenance identity changed",
        )
    return {
        "git_head": source["git_head"],
        "source_file_count": len(source_files),
        "source_file_bindings_sha256": canonical_sha256(source_files),
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "all_source_files_match_launch_head": bool(validate_launch_sources),
    }


def _validate_recipe_v22a(recipe, implementation_bundle, *, require_frozen_hashes):
    _require(
        recipe.get("schema")
        == "eggroll-es-authoritative-raw-v341-matched-replacement-recipe-v22a"
        and recipe.get("content_sha256_before_self_field")
        == canonical_sha256(_without_self(recipe))
        and recipe.get("experiment_name")
        == "production_v341_matched_replacement_v22a_authoritative_raw"
        and recipe.get("model") == str((ROOT / "models/Qwen3.6-35B-A3B").resolve())
        and recipe.get("layers") == [20, 21, 22, 23]
        and recipe.get("sigma") == 0.0003
        and recipe.get("alpha") == 0.0
        and recipe.get("population_size") == 64
        and recipe.get("implementation_bundle_sha256") == implementation_bundle
        and recipe.get("signed_wave_count") == 32
        and recipe.get("signed_wave_schedule_sha256")
        == SIGNED_WAVE_SCHEDULE_SHA256_V22A
        and recipe.get("same_perturbation_scores_both_arms_before_restore") is True
        and recipe.get("restore_once_after_both_arms") is True
        and recipe.get("draw_plan_content_sha256") == DRAW_PLAN_CONTENT_SHA256_V22A
        and recipe.get("panel_bundle_content_sha256") == PANEL_BUNDLE_CONTENT_SHA256_V22A,
        "v22a recipe identity or core recipe changed",
    )
    _require(
        recipe.get("analysis") == {
            "contrast_count": 1, "endpoint_count": 12,
            "bootstrap_repetitions": 50_000,
            "familywise_one_sided_quantile": 0.05 / 12,
            "noninferiority_margin": 0.0,
        }
        and recipe.get("authoritative_raw_scoring") == {
            "requests_per_engine_per_signed_wave": 480,
            "requests_by_arm": {
                "production_control": 240, "v341_matched_replacement": 240,
            },
            "requests_per_engine_all_signed_waves": 15_360,
            "requests_all_engines_all_signed_waves": 61_440,
            "dense_result_commitment_count": 2_560,
            "union_planner_called": False,
        }
        and recipe.get("hardware") == {
            "engine_count": 4, "tp_per_engine": 1, "gpu_ids": [0, 1, 2, 3],
            "all_four_engines_every_signed_wave": True,
        }
        and recipe.get("authority") == {
            "train_only_raw_runtime": True, "model_update_allowed": False,
            "checkpoint_write_allowed": False, "evaluation_allowed": False,
            "dataset_promotion_allowed": False,
        },
        "v22a recipe accounting hardware or authority changed",
    )
    _require(
        recipe.get("perturbation_basis_sha256")
        == "f68624388ac0549ac82ba3d1e64a317233c42f900502a6f5c6d6f07071b4c60e"
        and recipe.get("perturbation_seed_list_sha256")
        == "9faecdc81492052a6c466b0e986df9e31be0c0fccf24687a96ed604f2ef0f553"
        and recipe.get("mechanics_commit")
        == "365a7a307f933d973ecee43c73ee6c243369b801"
        and recipe.get("preregistration") == {
            "commit": "a5df84f9c31e9b3f8c7f601a807164df344dbff5",
            "file_sha256": (
                "b86a61e212af9862553119bc75f0c1bcfc264088af058585face1ac8f288e004"
            ),
            "content_sha256": (
                "ab0a72443a305bde922bf8fcb3cd9444a741489fc613f93e9d91eada3cc0ad08"
            ),
        }
        and recipe.get("moe_backend", {}).get("moe_backend") == "default_triton"
        and all(
            value is None
            for value in recipe.get("moe_backend", {}).get(
                "override_environment", {}
            ).values()
        ),
        "v22a basis preregistration or backend binding changed",
    )
    if require_frozen_hashes:
        _require(
            recipe["content_sha256_before_self_field"] == RECIPE_CONTENT_SHA256_V22A,
            "v22a frozen recipe content identity changed",
        )


def _recompute_gate_v22a(summary):
    _require(
        set(summary) == SUMMARY_KEYS_V22A
        and summary.get("schema")
        == "eggroll-es-raw-v341-matched-replacement-summary-v22a"
        and summary.get("content_sha256_before_self_field")
        == canonical_sha256(_without_self(summary))
        and summary.get("experiment_name")
        == "production_v341_matched_replacement_v22a_authoritative_raw"
        and summary.get("alpha") == 0.0 and summary.get("sigma") == 0.0003,
        "v22a compact summary identity changed",
    )
    integrity = summary.get("runtime_integrity", {})
    _require(
        set(integrity) == RUNTIME_INTEGRITY_KEYS_V22A
        and integrity.get("union_planner_called") is False
        and all(
            integrity.get(key) is True
            for key in RUNTIME_INTEGRITY_KEYS_V22A - {"union_planner_called"}
        ),
        "v22a runtime integrity contract changed",
    )
    arms = summary.get("arms", {})
    _require(
        set(arms) == set(ARMS_V22A)
        and all(
            set(arms[arm]) == {
                "all_panel_spreads_nonzero", "endpoint_values",
                "compact_estimator_sha256",
            }
            and arms[arm]["all_panel_spreads_nonzero"] is True
            and _is_digest(arms[arm]["compact_estimator_sha256"])
            and set(arms[arm]["endpoint_values"]) == set(ENDPOINT_NAMES_V22A)
            and all(
                _is_finite_number(value)
                for value in arms[arm]["endpoint_values"].values()
            )
            for arm in ARMS_V22A
        )
        and canonical_sha256({
            arm: arms[arm]["compact_estimator_sha256"] for arm in sorted(arms)
        }) == ARM_ESTIMATOR_COMMITMENTS_SHA256_V22A,
        "v22a compact arm endpoint coverage changed",
    )
    bootstrap = summary.get("paired_bootstrap", {})
    comparison = bootstrap.get("comparison", {})
    endpoint_items = comparison.get("endpoints", {})
    _require(
        set(bootstrap) == {
            "seed", "repetitions", "one_sided_quantile", "quantile_method",
            "draw_plan_content_sha256", "paired_same_draws_both_arms",
            "same_ht_coefficients_and_denominator_both_arms",
            "candidate_only_resampling_present", "whole_panel_block_resampling_used",
            "comparison",
        }
        and bootstrap.get("seed") == 20260824
        and bootstrap.get("repetitions") == 50_000
        and bootstrap.get("one_sided_quantile") == 0.05 / 12
        and bootstrap.get("quantile_method") == "linear"
        and bootstrap.get("draw_plan_content_sha256") == DRAW_PLAN_CONTENT_SHA256_V22A
        and bootstrap.get("paired_same_draws_both_arms") is True
        and bootstrap.get("same_ht_coefficients_and_denominator_both_arms") is True
        and bootstrap.get("candidate_only_resampling_present") is False
        and bootstrap.get("whole_panel_block_resampling_used") is False
        and set(comparison) == {"name", "treatment", "control", "endpoints"}
        and comparison.get("name") == "v341_matched_replacement_vs_production"
        and comparison.get("treatment") == "v341_matched_replacement"
        and comparison.get("control") == "production_control"
        and set(endpoint_items) == set(ENDPOINT_NAMES_V22A),
        "v22a paired bootstrap contract changed",
    )
    endpoint_status = {}
    for name in ENDPOINT_NAMES_V22A:
        item = endpoint_items[name]
        treatment = arms["v341_matched_replacement"]["endpoint_values"][name]
        control = arms["production_control"]["endpoint_values"][name]
        _require(
            set(item) == {
                "treatment_minus_control", "familywise_lcb",
                "noninferiority_margin",
            }
            and all(_is_finite_number(value) for value in item.values())
            and item["noninferiority_margin"] == 0.0
            and item["treatment_minus_control"] == treatment - control,
            "v22a observed delta LCB or zero-margin contract changed",
        )
        endpoint_status[name] = {
            "observed_passed": item["treatment_minus_control"] >= 0.0,
            "bootstrap_passed": item["familywise_lcb"] >= 0.0,
        }
    observed_count = sum(item["observed_passed"] for item in endpoint_status.values())
    bootstrap_count = sum(
        item["bootstrap_passed"] for item in endpoint_status.values()
    )
    expected_gate = {
        "schema": "eggroll-es-v341-matched-replacement-compatibility-gate-v22a",
        "observed_pass_count": observed_count,
        "bootstrap_pass_count": bootstrap_count,
        "all_twelve_observed_passed": observed_count == 12,
        "all_twelve_bootstrap_passed": bootstrap_count == 12,
        "all_panel_spreads_nonzero": True,
        "all_runtime_integrity_audits_passed": True,
        "compatibility_gate_passed": observed_count == 12 and bootstrap_count == 12,
        "decision": (
            "authorize_only_separate_fresh_basis_train_only_confirmation_preregistration"
            if observed_count == 12 and bootstrap_count == 12
            else "retain_production_dataset_and_v13_recipe"
        ),
        "dataset_promotion_authorized": False,
        "model_update_authorized": False,
        "evaluation_authorized": False,
    }
    _require(
        observed_count == 5 and bootstrap_count == 0
        and canonical_sha256(endpoint_status) == ENDPOINT_PASS_STATUS_SHA256_V22A
        and canonical_sha256(expected_gate) == GATE_SHA256_V22A,
        "v22a negative gate counts changed",
    )
    return expected_gate


def _validate_runtime_v22a(audit, configuration, *, require_frozen_hashes):
    _require(
        set(audit) == AUDIT_KEYS_V22A
        and audit.get("schema")
        == "eggroll-es-raw-v341-matched-replacement-runtime-audit-v22a"
        and audit.get("content_sha256_before_self_field")
        == canonical_sha256(_without_self(audit))
        and audit.get("signed_wave_count") == 32
        and audit.get("panel_count") == 10
        and audit.get("requests_per_engine_per_signed_wave") == 480
        and audit.get("requests_per_engine_all_signed_waves") == 15_360
        and audit.get("requests_all_engines_all_signed_waves") == 61_440
        and audit.get("dense_result_commitment_count") == 2_560
        and audit.get("signed_wave_schedule_sha256")
        == SIGNED_WAVE_SCHEDULE_SHA256_V22A
        and audit.get("union_planner_called") is False
        and all(audit.get(key) is False for key in (
            "per_unit_scores_persisted", "bootstrap_replicates_persisted",
            "bootstrap_draws_persisted", "row_content_persisted",
        ))
        and all(
            _is_digest(audit.get(key)) for key in AUDIT_KEYS_V22A
            if key.endswith("sha256")
        ),
        "v22a runtime audit or execution accounting changed",
    )
    _require(
        set(configuration) == CONFIGURATION_KEYS_V22A
        and configuration.get("schema")
        == "eggroll-es-authoritative-raw-runtime-configuration-v22a"
        and configuration.get("engine_count") == 4
        and configuration.get("tp_per_engine") == 1
        and configuration.get("gpu_ids") == [0, 1, 2, 3]
        and configuration.get("train_only_raw_runtime_opened") is True
        and configuration.get("union_planner_called") is False
        and all(configuration.get(key) is False for key in (
            "model_update_allowed", "checkpoint_write_allowed",
            "evaluation_surfaces_opened", "dataset_promotion_allowed",
        ))
        and configuration.get("panel_bundle_content_sha256")
        == PANEL_BUNDLE_CONTENT_SHA256_V22A
        and configuration.get("unselected_origin_sha256")
        == audit.get("unselected_origin_sha256"),
        "v22a runtime configuration or authority changed",
    )
    if require_frozen_hashes:
        _require(
            audit["content_sha256_before_self_field"]
            == RUNTIME_AUDIT_CONTENT_SHA256_V22A,
            "v22a frozen runtime audit identity changed",
        )


def validate_run_documents_v22a(
    failed, complete, report, *, require_frozen_hashes=True,
    validate_launch_sources=True,
):
    _require(
        isinstance(failed, dict) and set(failed) == FAILED_ATTEMPT_KEYS_V22A
        and failed.get("schema") == "eggroll-es-raw-attribution-attempt-v22a"
        and failed.get("status") == "failed"
        and failed.get("phase") == "inside_raw_runtime_or_cleanup"
        and failed.get("launch_id") == FAILED_LAUNCH_ID_V22A
        and Path(failed.get("run_directory", "")).resolve()
        == RUN_ROOT_V22A / "runs" / FAILED_LAUNCH_ID_V22A
        and failed.get("report_exists_after_attempt") is False
        and failed.get("failure_type") == "ModuleNotFoundError"
        and _is_digest(failed.get("failure_sha256"))
        and failed.get("content_sha256_before_self_field")
        == canonical_sha256(_without_self(failed)),
        "v22a fail-closed attempt envelope changed",
    )
    _require(
        isinstance(complete, dict) and set(complete) == COMPLETE_ATTEMPT_KEYS_V22A
        and complete.get("schema") == "eggroll-es-raw-attribution-attempt-v22a"
        and complete.get("status") == "complete"
        and complete.get("phase") == "after_cleanup_and_compact_report"
        and complete.get("launch_id") == COMPLETE_LAUNCH_ID_V22A
        and Path(complete.get("run_directory", "")).resolve() == REPORT_PATH_V22A.parent
        and complete.get("report_exists_after_attempt") is True
        and complete.get("content_sha256_before_self_field")
        == canonical_sha256(_without_self(complete))
        and isinstance(report, dict) and set(report) == REPORT_KEYS_V22A
        and report.get("schema")
        == "eggroll-es-authoritative-raw-v341-matched-replacement-report-v22a"
        and report.get("content_sha256_before_self_field")
        == canonical_sha256(_without_self(report)),
        "v22a completed attempt or report envelope changed",
    )
    binding = complete.get("report_binding", {})
    _require(
        set(binding) == {"path", "file_sha256", "content_sha256"}
        and Path(binding.get("path", "")).resolve() == REPORT_PATH_V22A
        and binding.get("file_sha256") == REPORT_FILE_SHA256_V22A
        and binding.get("content_sha256")
        == report.get("content_sha256_before_self_field")
        and failed.get("recipe") == complete.get("recipe") == report.get("recipe"),
        "v22a attempt-to-report binding changed",
    )
    source_audit = _validate_source_provenance_v22a(
        failed, complete, report, require_frozen_hashes=require_frozen_hashes,
        validate_launch_sources=validate_launch_sources,
    )
    recipe = report["recipe"]
    _validate_recipe_v22a(
        recipe, report["implementation"]["bundle_sha256"],
        require_frozen_hashes=require_frozen_hashes,
    )
    summary = report.get("summary", {})
    gate = _recompute_gate_v22a(summary)
    _require(report.get("gate") == gate, "v22a persisted gate differs from recomputation")
    audit = report.get("runtime_audit", {})
    configuration = report.get("configuration", {})
    _validate_runtime_v22a(audit, configuration, require_frozen_hashes=require_frozen_hashes)
    _require(
        all(attempt.get(key) is False for attempt in (failed, complete) for key in SIDE_EFFECT_KEYS_V22A)
        and all(report.get(key) is False for key in SIDE_EFFECT_KEYS_V22A)
        and all(summary.get(key) is False for key in (
            *SIDE_EFFECT_KEYS_V22A, "persisted_response_vectors_or_row_content",
            "bootstrap_draws_persisted", "unit_scores_persisted",
        ))
        and all(gate.get(key) is False for key in (
            "compatibility_gate_passed", "dataset_promotion_authorized",
            "model_update_authorized", "evaluation_authorized",
        )),
        "v22a raw-only no-mutation authority changed",
    )
    forbidden = sorted(
        FORBIDDEN_CONTENT_KEYS_V22A
        & set(_recursive_keys({"failed": failed, "complete": complete, "report": report}))
    )
    _require(not forbidden, "v22a aggregate documents contain forbidden content keys")
    if require_frozen_hashes:
        _require(
            failed["content_sha256_before_self_field"]
            == FAILED_ATTEMPT_CONTENT_SHA256_V22A
            and complete["content_sha256_before_self_field"]
            == COMPLETE_ATTEMPT_CONTENT_SHA256_V22A
            and report["content_sha256_before_self_field"] == REPORT_CONTENT_SHA256_V22A
            and summary["content_sha256_before_self_field"]
            == SUMMARY_CONTENT_SHA256_V22A,
            "v22a frozen attempt report or summary identity changed",
        )
    return {
        "source_audit": source_audit, "recipe": recipe,
        "runtime_audit": audit, "gate": gate,
        "forbidden_content_keys_found": forbidden,
    }


def build_evidence_v22a(*, validate_launch_sources=True):
    failed, complete, report = _load_exact_run_files_v22a()
    validated = validate_run_documents_v22a(
        failed, complete, report, require_frozen_hashes=True,
        validate_launch_sources=validate_launch_sources,
    )
    recipe = validated["recipe"]
    audit = validated["runtime_audit"]
    gate = validated["gate"]
    value = {
        "schema": "eggroll-es-v341-matched-replacement-negative-evidence-v22a",
        "status": "valid_completed_authoritative_raw_negative_compatibility_gate",
        "inputs": {
            "failed_attempt": {
                "path": str(FAILED_ATTEMPT_PATH_V22A),
                "file_sha256": FAILED_ATTEMPT_FILE_SHA256_V22A,
                "content_sha256": FAILED_ATTEMPT_CONTENT_SHA256_V22A,
                "launch_id": FAILED_LAUNCH_ID_V22A,
                "fail_closed": True,
            },
            "completed_attempt": {
                "path": str(COMPLETE_ATTEMPT_PATH_V22A),
                "file_sha256": COMPLETE_ATTEMPT_FILE_SHA256_V22A,
                "content_sha256": COMPLETE_ATTEMPT_CONTENT_SHA256_V22A,
                "launch_id": COMPLETE_LAUNCH_ID_V22A,
            },
            "report": {
                "path": str(REPORT_PATH_V22A),
                "file_sha256": REPORT_FILE_SHA256_V22A,
                "content_sha256": REPORT_CONTENT_SHA256_V22A,
            },
            "durable_cardinality": {
                "attempt_count": 2, "failed_attempt_count": 1,
                "completed_attempt_count": 1, "report_count": 1,
                "checkpoint_file_count": 0, "evaluation_file_count": 0,
            },
        },
        "source_provenance": validated["source_audit"],
        "recipe": {
            "content_sha256": recipe["content_sha256_before_self_field"],
            "model": recipe["model"], "layers": recipe["layers"],
            "sigma": recipe["sigma"], "alpha": recipe["alpha"],
            "population_size": recipe["population_size"],
            "perturbation_basis_sha256": recipe["perturbation_basis_sha256"],
            "perturbation_seed_list_sha256": recipe["perturbation_seed_list_sha256"],
            "draw_plan_content_sha256": recipe["draw_plan_content_sha256"],
            "panel_bundle_content_sha256": recipe["panel_bundle_content_sha256"],
            "layer_plan": copy.deepcopy(recipe["layer_plan"]),
            "moe_backend": copy.deepcopy(recipe["moe_backend"]),
        },
        "execution": {
            "engine_count": 4, "tp_per_engine": 1, "gpu_ids": [0, 1, 2, 3],
            "signed_wave_count": audit["signed_wave_count"],
            "requests_per_engine_per_signed_wave": audit[
                "requests_per_engine_per_signed_wave"
            ],
            "requests_per_engine_all_signed_waves": audit[
                "requests_per_engine_all_signed_waves"
            ],
            "requests_all_engines_all_signed_waves": audit[
                "requests_all_engines_all_signed_waves"
            ],
            "dense_result_commitment_count": audit["dense_result_commitment_count"],
            "signed_wave_schedule_sha256": audit["signed_wave_schedule_sha256"],
            "dense_result_commitments_sha256": audit[
                "dense_result_commitments_sha256"
            ],
            "restore_checks_sha256": audit["restore_checks_sha256"],
            "pre_post_probe_identity_sha256": audit[
                "pre_post_probe_identity_sha256"
            ],
            "all_runtime_integrity_audits_passed": True,
        },
        "analysis": {
            "comparison": "v341_matched_replacement_vs_production",
            "endpoint_count": 12, "bootstrap_repetitions": 50_000,
            "one_sided_quantile": 0.05 / 12, "quantile_method": "linear",
            "paired_same_draws_both_arms": True,
            "same_ht_coefficients_and_denominator_both_arms": True,
            "candidate_only_resampling_present": False,
            "noninferiority_margin": 0.0,
            "endpoint_pass_status_sha256": ENDPOINT_PASS_STATUS_SHA256_V22A,
            "observed_pass_count": gate["observed_pass_count"],
            "bootstrap_pass_count": gate["bootstrap_pass_count"],
        },
        "recomputed_gate": copy.deepcopy(gate),
        "decision": {
            "compatibility_gate_passed": False,
            "retain_production_dataset_and_v13_recipe": True,
            "candidate_v341_replacement_promotion_authorized": False,
            "model_update_authorized": False,
            "checkpoint_write_authorized": False,
            "evaluation_authorized": False,
            "dataset_promotion_authorized": False,
        },
        "aggregate_only_audit": {
            "forbidden_content_keys_found": [],
            "contains_row_question_answer_prompt_token_response_or_unit_scores": False,
            "contains_holdout_validation_ood_or_benchmark_content": False,
            "bootstrap_draws_or_replicates_persisted": False,
            "checkpoint_or_evaluation_files_created": False,
            "union_planner_called": False,
            "model_update_checkpoint_evaluation_or_promotion_applied": False,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return validate_evidence_v22a(value)


def validate_evidence_v22a(value):
    _require(
        isinstance(value, dict)
        and set(value) == {
            "schema", "status", "inputs", "source_provenance", "recipe",
            "execution", "analysis", "recomputed_gate", "decision",
            "aggregate_only_audit", "content_sha256_before_self_field",
        }
        and value.get("schema")
        == "eggroll-es-v341-matched-replacement-negative-evidence-v22a"
        and value.get("status")
        == "valid_completed_authoritative_raw_negative_compatibility_gate"
        and value.get("content_sha256_before_self_field")
        == canonical_sha256(_without_self(value))
        and value.get("inputs", {}).get("failed_attempt", {}).get("fail_closed")
        is True
        and value.get("inputs", {}).get("durable_cardinality") == {
            "attempt_count": 2, "failed_attempt_count": 1,
            "completed_attempt_count": 1, "report_count": 1,
            "checkpoint_file_count": 0, "evaluation_file_count": 0,
        }
        and value.get("source_provenance", {}).get("git_head")
        == LAUNCH_GIT_HEAD_V22A
        and value.get("source_provenance", {}).get("source_file_count") == 69
        and value.get("source_provenance", {}).get(
            "all_source_files_match_launch_head"
        ) is True
        and value.get("execution", {}).get("requests_all_engines_all_signed_waves")
        == 61_440
        and value.get("analysis", {}).get("observed_pass_count") == 5
        and value.get("analysis", {}).get("bootstrap_pass_count") == 0
        and value.get("analysis", {}).get("endpoint_pass_status_sha256")
        == ENDPOINT_PASS_STATUS_SHA256_V22A
        and value.get("recomputed_gate", {}).get("compatibility_gate_passed")
        is False
        and value.get("recomputed_gate", {}).get("decision")
        == "retain_production_dataset_and_v13_recipe"
        and value.get("decision", {}).get(
            "retain_production_dataset_and_v13_recipe"
        ) is True
        and all(value.get("decision", {}).get(key) is False for key in (
            "compatibility_gate_passed",
            "candidate_v341_replacement_promotion_authorized",
            "model_update_authorized", "checkpoint_write_authorized",
            "evaluation_authorized", "dataset_promotion_authorized",
        ))
        and value.get("aggregate_only_audit", {}).get(
            "forbidden_content_keys_found"
        ) == []
        and all(value.get("aggregate_only_audit", {}).get(key) is False for key in (
            "contains_row_question_answer_prompt_token_response_or_unit_scores",
            "contains_holdout_validation_ood_or_benchmark_content",
            "bootstrap_draws_or_replicates_persisted",
            "checkpoint_or_evaluation_files_created", "union_planner_called",
            "model_update_checkpoint_evaluation_or_promotion_applied",
        )),
        "v22a aggregate-only negative evidence changed",
    )
    return value


def _exclusive_write(path, value):
    path = Path(path).resolve()
    if path != OUTPUT_PATH_V22A:
        raise ValueError("v22a negative evidence output path changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("immutable v22a negative evidence already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH_V22A))
    args = parser.parse_args(argv)
    value = build_evidence_v22a(validate_launch_sources=True)
    _exclusive_write(Path(args.output), value)
    result = {
        "schema": "eggroll-es-v341-matched-replacement-negative-evidence-build-v22a",
        "path": str(OUTPUT_PATH_V22A),
        "file_sha256": file_sha256(OUTPUT_PATH_V22A),
        "content_sha256": value["content_sha256_before_self_field"],
        "gpu_launched": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
