#!/usr/bin/env python3
"""Build aggregate-only negative evidence for the completed V21A raw run."""

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
RUN_ROOT_V21A = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "production_v331_full_merged_patch_compatibility_v21a_authoritative_raw"
).resolve()
LAUNCH_ID_V21A = "1784097027797748644-308358-84a7d97fa2710610"
ATTEMPT_PATH_V21A = (
    RUN_ROOT_V21A / "attempts" / f"{LAUNCH_ID_V21A}.json"
).resolve()
REPORT_PATH_V21A = (
    RUN_ROOT_V21A / "runs" / LAUNCH_ID_V21A /
    "raw_production_v331_patch_attribution_v21a.json"
).resolve()
OUTPUT_PATH_V21A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V21A_PRODUCTION_V331_PATCH_NEGATIVE_EVIDENCE.json"
).resolve()

LAUNCH_GIT_HEAD_V21A = "0549c1fc77dbbbab90034ac3a32f081f1755d0a2"
ATTEMPT_FILE_SHA256_V21A = (
    "4fd9371f3bdd2100f1432249521ed9503a4f1262453657cf1cdd4fb18cb76ae6"
)
ATTEMPT_CONTENT_SHA256_V21A = (
    "7ff0469a9c59906a7d138cf43d8a26d70b40d7dadc6aae11e3f5dcf8110f2c97"
)
REPORT_FILE_SHA256_V21A = (
    "98b2d13350c8ed147fbd26108bd0d0137907bca7f1ddd0102371fd88b46ff29d"
)
REPORT_CONTENT_SHA256_V21A = (
    "35dfd2fae732c4ec5e8334d9f24107aca0c874bfaecefa08d6cf0c80974c3a41"
)
SUMMARY_CONTENT_SHA256_V21A = (
    "83693153cd87e0dfb2c6f873e5e3eb721964c670588cb8a567aa11a16e04bcd9"
)
RUNTIME_AUDIT_CONTENT_SHA256_V21A = (
    "044778623de2ae83eb55443255aa5e0ada688e247ff01116cb9ea30bcdaa214a"
)
RECIPE_CONTENT_SHA256_V21A = (
    "05808a0e3ff24aa77564e4465cf562c3ec52689017ecc7fb3e175d0410b34750"
)
SOURCE_PROVENANCE_CONTENT_SHA256_V21A = (
    "7eee3f44d0807a147ce86871e2f5f37731bdd9eef58aa994cef0abe2e3c1eac7"
)
IMPLEMENTATION_BUNDLE_SHA256_V21A = (
    "8e4babb3c5663fbe5368a19213c48c608a635d849e438dda17329c1105ed2104"
)
INHERITED_V19A_BUNDLE_SHA256_V21A = (
    "73ab67f230993810403171d2d18cc9abfbc4ab161a05356d87345c628ca0df4d"
)
PREREG_BUNDLE_SHA256_V21A = (
    "f029b557d1ee7f5beeaf1e7b56e385b6cf9c31b47871e94ca8c6aa3760ab56dd"
)
MECHANICS_BUNDLE_SHA256_V21A = (
    "fba28a9a9b5a5915319ff3077c27f8455be23af20e6283a8f838e24a005f04c8"
)
SOURCE_FILE_COUNT_V21A = 56
SIGNED_WAVE_SCHEDULE_SHA256_V21A = (
    "257a00f53a94d5262a7e9efef88a4b0a8c6c2a256322f47267954da2383ba414"
)
DRAW_PLAN_CONTENT_SHA256_V21A = (
    "b7964861ccd092f2c8c1177de5ff041f561e9a8f3a4d8b65bb2e4deb9aa7e820"
)

ENDPOINT_NAMES_V21A = (
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
ARMS_V21A = ("production_only", "production_plus_v331_patch")
RUNTIME_INTEGRITY_KEYS_V21A = {
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
FORBIDDEN_CONTENT_KEYS_V21A = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "prompt_token_ids", "token_ids", "tokens", "row_sha256",
    "ordered_row_identity_sha256", "joint_ids",
    "ordered_joint_identity_sha256", "unit_scores", "responses",
    "coefficients", "bootstrap_replicates", "bootstrap_draws", "row_content",
}

ATTEMPT_KEYS_V21A = {
    "schema", "status", "phase", "experiment_name", "recipe",
    "source_provenance", "union_planner_called", "model_update_applied",
    "checkpoint_written", "evaluation_surfaces_opened",
    "dataset_promotion_applied", "launch_id", "run_directory",
    "report_exists_after_attempt", "report_binding",
    "content_sha256_before_self_field",
}
REPORT_KEYS_V21A = {
    "schema", "recipe", "configuration", "runtime_audit", "summary", "gate",
    "implementation", "union_planner_called", "model_update_applied",
    "checkpoint_written", "evaluation_surfaces_opened",
    "dataset_promotion_applied", "content_sha256_before_self_field",
}
SUMMARY_KEYS_V21A = {
    "schema", "experiment_name", "alpha", "sigma", "runtime_integrity",
    "arms", "paired_bootstrap", "union_planner_called",
    "model_update_applied", "checkpoint_written", "evaluation_surfaces_opened",
    "dataset_promotion_applied", "persisted_response_vectors_or_row_content",
    "bootstrap_draws_persisted", "unit_scores_persisted",
    "content_sha256_before_self_field",
}
AUDIT_KEYS_V21A = {
    "schema", "fixed_request_identity_sha256", "token_boundary_audit_sha256",
    "pre_post_probe_identity_sha256", "signed_wave_schedule_sha256",
    "restore_checks_sha256", "dense_result_commitments_sha256",
    "population_boundary_audit_sha256", "unselected_origin_sha256",
    "unselected_origin_audit_sha256", "signed_wave_count", "panel_count",
    "requests_per_engine_per_signed_wave", "requests_per_engine_all_signed_waves",
    "dense_result_commitment_count", "union_planner_called",
    "per_unit_scores_persisted", "bootstrap_replicates_persisted",
    "bootstrap_draws_persisted", "row_content_persisted",
    "content_sha256_before_self_field",
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


def _load_exact_run_files_v21a():
    attempts = sorted((RUN_ROOT_V21A / "attempts").glob("*.json"))
    run_dirs = sorted(path for path in (RUN_ROOT_V21A / "runs").iterdir() if path.is_dir())
    reports = sorted((RUN_ROOT_V21A / "runs").glob("*/*.json"))
    all_files = sorted(path for path in RUN_ROOT_V21A.rglob("*") if path.is_file())
    _require(
        attempts == [ATTEMPT_PATH_V21A]
        and run_dirs == [REPORT_PATH_V21A.parent]
        and reports == [REPORT_PATH_V21A]
        and all_files == [ATTEMPT_PATH_V21A, REPORT_PATH_V21A],
        "v21a durable run file cardinality changed",
    )
    _require(
        file_sha256(ATTEMPT_PATH_V21A) == ATTEMPT_FILE_SHA256_V21A
        and file_sha256(REPORT_PATH_V21A) == REPORT_FILE_SHA256_V21A,
        "v21a durable attempt or report bytes changed",
    )
    return (
        json.loads(ATTEMPT_PATH_V21A.read_text(encoding="utf-8")),
        json.loads(REPORT_PATH_V21A.read_text(encoding="utf-8")),
    )


def _validate_source_provenance_v21a(
    attempt, report, *, require_frozen_hashes, validate_launch_sources,
):
    source = attempt.get("source_provenance", {})
    implementation = report.get("implementation", {})
    _require(
        set(source) == {
            "schema", "git_head", "files", "implementation_bundle_sha256",
            "content_sha256_before_self_field",
        }
        and source.get("schema") == "eggroll-es-raw-source-provenance-v21a"
        and source.get("git_head") == LAUNCH_GIT_HEAD_V21A
        and source.get("content_sha256_before_self_field")
        == canonical_sha256(_without_self(source))
        and set(implementation) == {
            "files", "inherited_v19a_bundle_sha256", "prereg_bundle_sha256",
            "mechanics_bundle_sha256", "bundle_sha256",
        }
        and implementation.get("inherited_v19a_bundle_sha256")
        == INHERITED_V19A_BUNDLE_SHA256_V21A
        and implementation.get("prereg_bundle_sha256")
        == PREREG_BUNDLE_SHA256_V21A
        and implementation.get("mechanics_bundle_sha256")
        == MECHANICS_BUNDLE_SHA256_V21A,
        "v21a source provenance or implementation structure changed",
    )
    source_files = source.get("files", {})
    implementation_files = implementation.get("files", {})
    _require(
        isinstance(source_files, dict)
        and isinstance(implementation_files, dict)
        and len(source_files) == len(implementation_files) == SOURCE_FILE_COUNT_V21A
        and set(source_files) == set(implementation_files)
        and implementation.get("bundle_sha256")
        == canonical_sha256(implementation_files)
        and source.get("implementation_bundle_sha256")
        == implementation.get("bundle_sha256"),
        "v21a implementation bundle or source coverage changed",
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
            "v21a individual source binding changed",
        )
        path = Path(implementation_item["path"]).resolve()
        try:
            relative = path.relative_to(ROOT).as_posix()
        except ValueError as error:
            raise RuntimeError("v21a source path escaped repository") from error
        _require(
            source_item.get("relative_path") == relative,
            "v21a relative source path binding changed",
        )
        if validate_launch_sources:
            try:
                raw = subprocess.check_output(
                    ["git", "show", f"{LAUNCH_GIT_HEAD_V21A}:{relative}"], cwd=ROOT
                )
            except subprocess.CalledProcessError as error:
                raise RuntimeError("v21a launch source is missing") from error
            _require(
                hashlib.sha256(raw).hexdigest() == implementation_item["file_sha256"]
                and path.is_file()
                and file_sha256(path) == implementation_item["file_sha256"],
                "v21a launch source file digest changed",
            )
    if require_frozen_hashes:
        _require(
            source["content_sha256_before_self_field"]
            == SOURCE_PROVENANCE_CONTENT_SHA256_V21A
            and implementation["bundle_sha256"]
            == IMPLEMENTATION_BUNDLE_SHA256_V21A,
            "v21a frozen source provenance identity changed",
        )
    return {
        "git_head": source["git_head"],
        "source_file_count": len(source_files),
        "source_file_bindings_sha256": canonical_sha256(source_files),
        "implementation_file_bindings_sha256": canonical_sha256(
            implementation_files
        ),
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "all_source_files_match_launch_head": bool(validate_launch_sources),
    }


def _validate_recipe_v21a(recipe, implementation_bundle, *, require_frozen_hashes):
    _require(
        recipe.get("schema")
        == "eggroll-es-authoritative-raw-production-v331-recipe-v21a"
        and recipe.get("content_sha256_before_self_field")
        == canonical_sha256(_without_self(recipe))
        and recipe.get("experiment_name")
        == "production_v331_full_merged_patch_compatibility_v21a_authoritative_raw"
        and recipe.get("model") == str((ROOT / "models/Qwen3.6-35B-A3B").resolve())
        and recipe.get("layers") == [20, 21, 22, 23]
        and recipe.get("sigma") == 0.0003
        and recipe.get("alpha") == 0.0
        and recipe.get("population_size") == 64
        and recipe.get("implementation_bundle_sha256") == implementation_bundle
        and recipe.get("signed_wave_count") == 32
        and recipe.get("signed_wave_schedule_sha256")
        == SIGNED_WAVE_SCHEDULE_SHA256_V21A
        and recipe.get("same_perturbation_scores_both_arms_before_restore") is True
        and recipe.get("restore_once_after_both_arms") is True
        and recipe.get("draw_plan_content_sha256") == DRAW_PLAN_CONTENT_SHA256_V21A,
        "v21a recipe identity or core recipe changed",
    )
    _require(
        recipe.get("analysis") == {
            "contrast_count": 1, "endpoint_count": 12,
            "bootstrap_repetitions": 50_000,
            "familywise_one_sided_quantile": 0.05 / 12,
            "noninferiority_margin": 0.0,
        }
        and recipe.get("authoritative_raw_scoring") == {
            "requests_per_engine_per_signed_wave": 540,
            "requests_by_arm": {
                "production_only": 240,
                "production_plus_v331_patch": 300,
            },
            "requests_per_engine_all_signed_waves": 17_280,
            "dense_result_commitment_count": 2_560,
            "union_planner_called": False,
        }
        and recipe.get("hardware") == {
            "engine_count": 4, "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "all_four_engines_every_signed_wave": True,
        }
        and recipe.get("authority") == {
            "train_only_raw_runtime": True, "model_update_allowed": False,
            "checkpoint_write_allowed": False, "evaluation_allowed": False,
            "dataset_promotion_allowed": False,
        },
        "v21a recipe accounting hardware or authority changed",
    )
    _require(
        recipe.get("perturbation_basis_sha256")
        == "65970861cd06b53e52cf848b2c8b8961160bf9c68f6b1b9f4935a88ba8d314d2"
        and recipe.get("perturbation_seed_list_sha256")
        == "b8456790fa704e10a50e332bc22bfeb7f981bdfa40c206494dcab8df2f1e9062"
        and recipe.get("panel_bundle_content_sha256")
        == "b565a3fee777d5a42414dd2f6c2a100e89f38c83c699e221fda7e6728875ad7e"
        and recipe.get("mechanics_commit")
        == "d97c552cfb99ffeaa50c1342df132adc41c31f2e"
        and recipe.get("preregistration") == {
            "commit": "74b9775e019a25f01d860e3508c9979646624db8",
            "file_sha256": (
                "d0a9284bb5a944b6f5059f4ed55a4616e2d00acaf2a8bdbba731603eb2126dbd"
            ),
            "content_sha256": (
                "f7d0814eeebf94e929421599b6ed66099db8d827aab26f67798aa73e06353dfc"
            ),
        }
        and recipe.get("moe_backend", {}).get("moe_backend") == "default_triton"
        and all(
            value is None
            for value in recipe.get("moe_backend", {}).get(
                "override_environment", {}
            ).values()
        ),
        "v21a basis preregistration or backend binding changed",
    )
    if require_frozen_hashes:
        _require(
            recipe["content_sha256_before_self_field"] == RECIPE_CONTENT_SHA256_V21A,
            "v21a frozen recipe content identity changed",
        )


def _recompute_gate_v21a(summary):
    _require(
        set(summary) == SUMMARY_KEYS_V21A
        and summary.get("schema")
        == "eggroll-es-raw-production-v331-attribution-summary-v21a"
        and summary.get("content_sha256_before_self_field")
        == canonical_sha256(_without_self(summary))
        and summary.get("experiment_name")
        == "production_v331_full_merged_patch_compatibility_v21a_authoritative_raw"
        and summary.get("alpha") == 0.0 and summary.get("sigma") == 0.0003,
        "v21a compact summary identity changed",
    )
    integrity = summary.get("runtime_integrity", {})
    _require(
        set(integrity) == RUNTIME_INTEGRITY_KEYS_V21A
        and integrity.get("union_planner_called") is False
        and all(
            integrity.get(key) is True
            for key in RUNTIME_INTEGRITY_KEYS_V21A - {"union_planner_called"}
        ),
        "v21a runtime integrity contract changed",
    )
    _require(
        set(summary.get("arms", {})) == set(ARMS_V21A)
        and all(
            set(summary["arms"][arm]) == {
                "all_panel_spreads_nonzero", "endpoint_values",
                "compact_estimator_sha256",
            }
            and summary["arms"][arm]["all_panel_spreads_nonzero"] is True
            and _is_digest(summary["arms"][arm]["compact_estimator_sha256"])
            and set(summary["arms"][arm]["endpoint_values"])
            == set(ENDPOINT_NAMES_V21A)
            and all(
                _is_finite_number(value)
                for value in summary["arms"][arm]["endpoint_values"].values()
            )
            for arm in ARMS_V21A
        ),
        "v21a compact arm endpoint coverage changed",
    )
    bootstrap = summary.get("paired_bootstrap", {})
    comparison = bootstrap.get("comparison", {})
    endpoint_items = comparison.get("endpoints", {})
    _require(
        set(bootstrap) == {
            "seed", "repetitions", "one_sided_quantile", "quantile_method",
            "draw_plan_content_sha256", "paired_same_draws_both_arms",
            "whole_panel_block_resampling_used", "comparison",
        }
        and bootstrap.get("seed") == 20260822
        and bootstrap.get("repetitions") == 50_000
        and bootstrap.get("one_sided_quantile") == 0.05 / 12
        and bootstrap.get("quantile_method") == "linear"
        and bootstrap.get("draw_plan_content_sha256")
        == DRAW_PLAN_CONTENT_SHA256_V21A
        and bootstrap.get("paired_same_draws_both_arms") is True
        and bootstrap.get("whole_panel_block_resampling_used") is False
        and set(comparison) == {"name", "treatment", "control", "endpoints"}
        and comparison.get("name") == "production_plus_v331_patch_vs_production"
        and comparison.get("treatment") == "production_plus_v331_patch"
        and comparison.get("control") == "production_only"
        and set(endpoint_items) == set(ENDPOINT_NAMES_V21A),
        "v21a paired bootstrap contract changed",
    )
    compact_endpoints = {}
    for name in ENDPOINT_NAMES_V21A:
        item = endpoint_items[name]
        treatment = summary["arms"]["production_plus_v331_patch"][
            "endpoint_values"
        ][name]
        control = summary["arms"]["production_only"]["endpoint_values"][name]
        recomputed_delta = treatment - control
        _require(
            set(item) == {
                "treatment_minus_control", "familywise_lcb",
                "noninferiority_margin",
            }
            and all(_is_finite_number(value) for value in item.values())
            and item["noninferiority_margin"] == 0.0
            and item["treatment_minus_control"] == recomputed_delta,
            "v21a observed delta LCB or zero-margin contract changed",
        )
        compact_endpoints[name] = {
            "treatment_minus_control": item["treatment_minus_control"],
            "familywise_lcb": item["familywise_lcb"],
            "noninferiority_margin": 0.0,
            "observed_passed": item["treatment_minus_control"] >= 0.0,
            "bootstrap_passed": item["familywise_lcb"] >= 0.0,
        }
    observed_count = sum(
        item["observed_passed"] for item in compact_endpoints.values()
    )
    bootstrap_count = sum(
        item["bootstrap_passed"] for item in compact_endpoints.values()
    )
    expected_gate = {
        "schema": "eggroll-es-production-v331-patch-compatibility-gate-v21a",
        "observed_pass_count": observed_count,
        "bootstrap_pass_count": bootstrap_count,
        "all_twelve_observed_passed": observed_count == 12,
        "all_twelve_bootstrap_passed": bootstrap_count == 12,
        "all_panel_spreads_nonzero": True,
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
        observed_count == 3 and bootstrap_count == 0
        and expected_gate["compatibility_gate_passed"] is False,
        "v21a negative gate counts changed",
    )
    return expected_gate, compact_endpoints


def _validate_runtime_audit_v21a(audit, *, require_frozen_hashes):
    _require(
        set(audit) == AUDIT_KEYS_V21A
        and audit.get("schema")
        == "eggroll-es-raw-production-v331-runtime-audit-v21a"
        and audit.get("content_sha256_before_self_field")
        == canonical_sha256(_without_self(audit))
        and audit.get("signed_wave_count") == 32
        and audit.get("panel_count") == 10
        and audit.get("requests_per_engine_per_signed_wave") == 540
        and audit.get("requests_per_engine_all_signed_waves") == 17_280
        and audit.get("dense_result_commitment_count") == 2_560
        and audit.get("signed_wave_schedule_sha256")
        == SIGNED_WAVE_SCHEDULE_SHA256_V21A
        and audit.get("union_planner_called") is False
        and all(audit.get(key) is False for key in (
            "per_unit_scores_persisted", "bootstrap_replicates_persisted",
            "bootstrap_draws_persisted", "row_content_persisted",
        ))
        and all(
            _is_digest(audit.get(key)) for key in AUDIT_KEYS_V21A
            if key.endswith("sha256")
        ),
        "v21a runtime audit or execution accounting changed",
    )
    if require_frozen_hashes:
        _require(
            audit["content_sha256_before_self_field"]
            == RUNTIME_AUDIT_CONTENT_SHA256_V21A,
            "v21a frozen runtime audit identity changed",
        )


def validate_run_documents_v21a(
    attempt, report, *, require_frozen_hashes=True, validate_launch_sources=True,
):
    _require(
        isinstance(attempt, dict) and set(attempt) == ATTEMPT_KEYS_V21A
        and attempt.get("schema") == "eggroll-es-raw-attribution-attempt-v21a"
        and attempt.get("status") == "complete"
        and attempt.get("phase") == "after_cleanup_and_compact_report"
        and attempt.get("launch_id") == LAUNCH_ID_V21A
        and Path(attempt.get("run_directory", "")).resolve() == REPORT_PATH_V21A.parent
        and attempt.get("report_exists_after_attempt") is True
        and attempt.get("content_sha256_before_self_field")
        == canonical_sha256(_without_self(attempt))
        and isinstance(report, dict) and set(report) == REPORT_KEYS_V21A
        and report.get("schema")
        == "eggroll-es-authoritative-raw-production-v331-report-v21a"
        and report.get("content_sha256_before_self_field")
        == canonical_sha256(_without_self(report)),
        "v21a durable attempt or report envelope changed",
    )
    binding = attempt.get("report_binding", {})
    _require(
        set(binding) == {"path", "file_sha256", "content_sha256"}
        and Path(binding.get("path", "")).resolve() == REPORT_PATH_V21A
        and binding.get("file_sha256") == REPORT_FILE_SHA256_V21A
        and binding.get("content_sha256")
        == report.get("content_sha256_before_self_field")
        and attempt.get("recipe") == report.get("recipe"),
        "v21a attempt-to-report binding changed",
    )
    source_audit = _validate_source_provenance_v21a(
        attempt, report, require_frozen_hashes=require_frozen_hashes,
        validate_launch_sources=validate_launch_sources,
    )
    recipe = report["recipe"]
    _validate_recipe_v21a(
        recipe, report["implementation"]["bundle_sha256"],
        require_frozen_hashes=require_frozen_hashes,
    )
    summary = report.get("summary", {})
    gate, endpoints = _recompute_gate_v21a(summary)
    _require(report.get("gate") == gate, "v21a persisted gate differs from recomputation")
    audit = report.get("runtime_audit", {})
    _validate_runtime_audit_v21a(audit, require_frozen_hashes=require_frozen_hashes)
    configuration = report.get("configuration", {})
    _require(
        configuration.get("engine_count") == 4
        and configuration.get("tp_per_engine") == 1
        and configuration.get("gpu_ids") == [0, 1, 2, 3]
        and configuration.get("train_only_raw_runtime_opened") is True
        and configuration.get("union_planner_called") is False
        and all(configuration.get(key) is False for key in (
            "model_update_allowed", "checkpoint_write_allowed",
            "evaluation_surfaces_opened", "dataset_promotion_allowed",
        ))
        and configuration.get("panel_bundle_content_sha256")
        == recipe["panel_bundle_content_sha256"]
        and configuration.get("unselected_origin_sha256")
        == audit["unselected_origin_sha256"],
        "v21a runtime configuration or authority changed",
    )
    _require(
        all(attempt.get(key) is False for key in (
            "union_planner_called", "model_update_applied", "checkpoint_written",
            "evaluation_surfaces_opened", "dataset_promotion_applied",
        ))
        and all(report.get(key) is False for key in (
            "union_planner_called", "model_update_applied", "checkpoint_written",
            "evaluation_surfaces_opened", "dataset_promotion_applied",
        ))
        and all(summary.get(key) is False for key in (
            "union_planner_called", "model_update_applied", "checkpoint_written",
            "evaluation_surfaces_opened", "dataset_promotion_applied",
            "persisted_response_vectors_or_row_content",
            "bootstrap_draws_persisted", "unit_scores_persisted",
        ))
        and all(gate.get(key) is False for key in (
            "compatibility_gate_passed", "dataset_promotion_authorized",
            "model_update_authorized", "evaluation_authorized",
        )),
        "v21a raw-only no-mutation authority changed",
    )
    forbidden = sorted(
        FORBIDDEN_CONTENT_KEYS_V21A
        & set(_recursive_keys({"attempt": attempt, "report": report}))
    )
    _require(not forbidden, "v21a aggregate documents contain forbidden content keys")
    if require_frozen_hashes:
        _require(
            attempt["content_sha256_before_self_field"]
            == ATTEMPT_CONTENT_SHA256_V21A
            and report["content_sha256_before_self_field"]
            == REPORT_CONTENT_SHA256_V21A
            and summary["content_sha256_before_self_field"]
            == SUMMARY_CONTENT_SHA256_V21A,
            "v21a frozen attempt report or summary identity changed",
        )
    return {
        "source_audit": source_audit,
        "recipe": recipe,
        "runtime_audit": audit,
        "gate": gate,
        "endpoints": endpoints,
        "forbidden_content_keys_found": forbidden,
    }


def build_evidence_v21a(*, validate_launch_sources=True):
    attempt, report = _load_exact_run_files_v21a()
    validated = validate_run_documents_v21a(
        attempt, report, require_frozen_hashes=True,
        validate_launch_sources=validate_launch_sources,
    )
    recipe = validated["recipe"]
    audit = validated["runtime_audit"]
    gate = validated["gate"]
    value = {
        "schema": "eggroll-es-production-v331-patch-negative-evidence-v21a",
        "status": "valid_completed_authoritative_raw_negative_compatibility_gate",
        "inputs": {
            "attempt": {
                "path": str(ATTEMPT_PATH_V21A),
                "file_sha256": ATTEMPT_FILE_SHA256_V21A,
                "content_sha256": ATTEMPT_CONTENT_SHA256_V21A,
            },
            "report": {
                "path": str(REPORT_PATH_V21A),
                "file_sha256": REPORT_FILE_SHA256_V21A,
                "content_sha256": REPORT_CONTENT_SHA256_V21A,
            },
            "launch_id": LAUNCH_ID_V21A,
        },
        "source_provenance": validated["source_audit"],
        "recipe": {
            "content_sha256": recipe["content_sha256_before_self_field"],
            "model": recipe["model"], "layers": recipe["layers"],
            "sigma": recipe["sigma"], "alpha": recipe["alpha"],
            "population_size": recipe["population_size"],
            "perturbation_basis_sha256": recipe["perturbation_basis_sha256"],
            "perturbation_seed_list_sha256": (
                recipe["perturbation_seed_list_sha256"]
            ),
            "draw_plan_content_sha256": recipe["draw_plan_content_sha256"],
            "panel_bundle_content_sha256": recipe["panel_bundle_content_sha256"],
            "layer_plan": copy.deepcopy(recipe["layer_plan"]),
            "moe_backend": copy.deepcopy(recipe["moe_backend"]),
        },
        "execution": {
            "engine_count": 4, "tp_per_engine": 1, "gpu_ids": [0, 1, 2, 3],
            "signed_wave_count": audit["signed_wave_count"],
            "requests_per_engine_per_signed_wave": (
                audit["requests_per_engine_per_signed_wave"]
            ),
            "requests_per_engine_all_signed_waves": (
                audit["requests_per_engine_all_signed_waves"]
            ),
            "dense_result_commitment_count": (
                audit["dense_result_commitment_count"]
            ),
            "signed_wave_schedule_sha256": audit["signed_wave_schedule_sha256"],
            "dense_result_commitments_sha256": (
                audit["dense_result_commitments_sha256"]
            ),
            "restore_checks_sha256": audit["restore_checks_sha256"],
            "pre_post_probe_identity_sha256": (
                audit["pre_post_probe_identity_sha256"]
            ),
            "all_runtime_integrity_audits_passed": True,
        },
        "analysis": {
            "comparison": "production_plus_v331_patch_vs_production",
            "endpoint_count": 12, "bootstrap_repetitions": 50_000,
            "one_sided_quantile": 0.05 / 12, "quantile_method": "linear",
            "paired_same_draws_both_arms": True,
            "noninferiority_margin": 0.0,
            "endpoints": validated["endpoints"],
            "observed_pass_count": gate["observed_pass_count"],
            "bootstrap_pass_count": gate["bootstrap_pass_count"],
        },
        "recomputed_gate": copy.deepcopy(gate),
        "decision": {
            "compatibility_gate_passed": False,
            "retain_production_dataset_and_v13_recipe": True,
            "candidate_v331_patch_promotion_authorized": False,
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
            "union_planner_called": False,
            "model_update_checkpoint_evaluation_or_promotion_applied": False,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return validate_evidence_v21a(value)


def validate_evidence_v21a(value):
    _require(
        isinstance(value, dict)
        and set(value) == {
            "schema", "status", "inputs", "source_provenance", "recipe",
            "execution", "analysis", "recomputed_gate", "decision",
            "aggregate_only_audit", "content_sha256_before_self_field",
        }
        and value.get("schema")
        == "eggroll-es-production-v331-patch-negative-evidence-v21a"
        and value.get("status")
        == "valid_completed_authoritative_raw_negative_compatibility_gate"
        and value.get("content_sha256_before_self_field")
        == canonical_sha256(_without_self(value))
        and value.get("inputs", {}).get("attempt", {}).get("file_sha256")
        == ATTEMPT_FILE_SHA256_V21A
        and value.get("inputs", {}).get("report", {}).get("file_sha256")
        == REPORT_FILE_SHA256_V21A
        and value.get("source_provenance", {}).get("git_head")
        == LAUNCH_GIT_HEAD_V21A
        and value.get("source_provenance", {}).get("source_file_count") == 56
        and value.get("source_provenance", {}).get(
            "all_source_files_match_launch_head"
        ) is True
        and value.get("analysis", {}).get("observed_pass_count") == 3
        and value.get("analysis", {}).get("bootstrap_pass_count") == 0
        and set(value.get("analysis", {}).get("endpoints", {}))
        == set(ENDPOINT_NAMES_V21A)
        and all(
            item.get("noninferiority_margin") == 0.0
            for item in value.get("analysis", {}).get("endpoints", {}).values()
        )
        and value.get("recomputed_gate", {}).get("compatibility_gate_passed")
        is False
        and value.get("recomputed_gate", {}).get("decision")
        == "retain_production_dataset_and_v13_recipe"
        and value.get("decision", {}).get(
            "retain_production_dataset_and_v13_recipe"
        ) is True
        and all(value.get("decision", {}).get(key) is False for key in (
            "compatibility_gate_passed", "candidate_v331_patch_promotion_authorized",
            "model_update_authorized", "checkpoint_write_authorized",
            "evaluation_authorized", "dataset_promotion_authorized",
        ))
        and value.get("aggregate_only_audit", {}).get(
            "forbidden_content_keys_found"
        ) == []
        and all(value.get("aggregate_only_audit", {}).get(key) is False for key in (
            "contains_row_question_answer_prompt_token_response_or_unit_scores",
            "contains_holdout_validation_ood_or_benchmark_content",
            "bootstrap_draws_or_replicates_persisted", "union_planner_called",
            "model_update_checkpoint_evaluation_or_promotion_applied",
        )),
        "v21a aggregate-only negative evidence changed",
    )
    return value


def _exclusive_write(path, value):
    path = Path(path).resolve()
    if path != OUTPUT_PATH_V21A:
        raise ValueError("v21a negative evidence output path changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("immutable v21a negative evidence already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH_V21A))
    args = parser.parse_args(argv)
    value = build_evidence_v21a(validate_launch_sources=True)
    _exclusive_write(Path(args.output), value)
    result = {
        "schema": "eggroll-es-production-v331-patch-negative-evidence-build-v21a",
        "path": str(OUTPUT_PATH_V21A),
        "file_sha256": file_sha256(OUTPUT_PATH_V21A),
        "content_sha256": value["content_sha256_before_self_field"],
        "gpu_launched": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
