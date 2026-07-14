#!/usr/bin/env python3
"""Minimal exact v8-data44 replay under a no-overwrite v9b envelope."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import run_eggroll_es_anchor_line_search as driver_v1
import run_eggroll_es_anchor_stability_v8 as driver_v8
import run_eggroll_es_anchor_replay_v9 as replay_v9
import train_eggroll_es_specialist_anchor_v8 as anchor_v8


ROOT = Path(__file__).resolve().parent
DATA_SEED_V9B = 44
REPLAY_COSINE_THRESHOLD_V9B = 0.99
EXPERIMENT_NAME_V9B = (
    "snapshot794_layer_v9b_middle_late_exact_replay_"
    "data44_basis20260714_control1"
)
EXPECTED_WRAPPER_RECIPE_SHA256_V9B = (
    "441879ad2bb8696bf18fccc32c05bae9f1e882ea4c2d558d7ba17ad0fc5b0e25"
)
V9B_POLICY = {
    "deterministic_replay_family_v9b": "exact_v8_data44_control",
    "stage_v9b": "deterministic_replay",
    "target_alpha_zero_only_v9b": True,
    "benchmark_selection_forbidden_v9b": True,
    "requires_failed_v8_family_v9b": True,
    "exact_raw_score_identity_required_v9b": True,
    "replay_coefficient_cosine_threshold_v9b": 0.99,
}
V9B_IMPLEMENTATION_PATHS = {
    **driver_v8.V8_IMPLEMENTATION_PATHS,
    "distributed_driver_v9b": Path(__file__).resolve(),
    "replay_reporter_v9b": ROOT / "report_eggroll_es_replay_v9b.py",
    "protocol_v9b": (
        ROOT / "experiments/eggroll_es_hpo/"
        "S6_DATA44_REPLAY_V9B_PROTOCOL.md"
    ),
    "contract_tests_v9b": ROOT / "test_eggroll_es_replay_v9b.py",
    "v9_evidence_validator": Path(replay_v9.__file__).resolve(),
    "v8_failed_report_v9b": replay_v9.V8_FAILED_REPORT_PATH_V9,
}
_ACTIVE_EXECUTION_V9B = None


def _failed_v8_seed44_evidence_v9b(path):
    family = replay_v9._v8_failed_evidence_v9(path)
    runs = family.get("runs", [])
    seed44 = [row for row in runs if row.get("data_seed") == 44]
    expected = replay_v9.ORIGINAL_V8_JOURNALS_V9[44]
    if (
        family.get("formal_v8_result", {}).get("passed") is not False
        or len(seed44) != 1
        or Path(seed44[0].get("journal", "")).resolve() != expected["path"]
        or seed44[0].get("journal_file_sha256") != expected["file_sha256"]
        or seed44[0].get("content_sha256") != expected["content_sha256"]
        or seed44[0].get("coefficient_sha256")
        != expected["coefficient_sha256"]
        or seed44[0].get("robust_plan_sha256")
        != expected["robust_plan_sha256"]
    ):
        raise ValueError("v9b exact original v8 data44 evidence changed")
    return {
        "schema": "eggroll-es-v8-data44-reference-binding-v9b",
        "failed_v8_family_binding_sha256": family["binding_sha256"],
        "reference": copy.deepcopy(seed44[0]),
    }


def wrapper_recipe_v9b(evidence):
    return {
        "schema": "eggroll-es-v8-data44-wrapper-recipe-v9b",
        "underlying_v8_recipe_sha256": (
            driver_v8.EXPECTED_RECIPE_SHA256_V8[44]
        ),
        "data_seed": 44,
        "actual_experiment_name": EXPERIMENT_NAME_V9B,
        "perturbation_basis_sha256": (
            driver_v8.PERTURBATION_BASIS_SHA256_V8
        ),
        "population_size": 32,
        "target_alphas": [0.0],
        "replay_cosine_threshold": 0.99,
        "reference_evidence": copy.deepcopy(evidence),
    }


def _replace_option(argv, option, value):
    argv = list(argv)
    positions = [index for index, item in enumerate(argv) if item == option]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"v9b requires one explicit {option}")
    argv[positions[0] + 1] = str(value)
    return argv


def validate_frozen_execution_cli_v9b(argv, bundle):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--v9b-stage", choices=("deterministic_replay",))
    parser.add_argument("--v9b-v8-failed-report")
    parser.add_argument("--v9b-perturbation-basis-seed", type=int)
    parser.add_argument("--v9b-dry-run", action="store_true")
    stage, actual_remaining = parser.parse_known_args(list(argv))
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument("--seed", type=int)
    options.add_argument("--population-size", type=int)
    options.add_argument("--target-alphas")
    options.add_argument("--experiment-name")
    actual, _ = options.parse_known_args(actual_remaining)
    if stage.v9b_stage != "deterministic_replay":
        raise ValueError("v9b requires --v9b-stage deterministic_replay")
    if actual.seed != 44:
        raise ValueError("v9b exact replay seed must be 44")
    if actual.population_size != 32:
        raise ValueError("v9b exact replay population must be 32")
    if driver_v1.parse_target_alphas(actual.target_alphas) != [0.0]:
        raise ValueError("v9b replay target must be exactly alpha zero")
    if actual.experiment_name != EXPERIMENT_NAME_V9B:
        raise ValueError("v9b no-overwrite experiment name changed")
    if (
        stage.v9b_perturbation_basis_seed
        != driver_v8.PERTURBATION_BASIS_SEED_V8
    ):
        raise ValueError("v9b perturbation basis seed changed")
    if stage.v9b_v8_failed_report is None:
        raise ValueError("v9b requires the failed v8 report")
    evidence = _failed_v8_seed44_evidence_v9b(
        stage.v9b_v8_failed_report
    )
    v8_argv = _replace_option(
        actual_remaining, "--experiment-name",
        driver_v8._experiment_name_v8(44),
    )
    v8_argv.extend([
        "--v8-stage", "stability",
        "--v8-v7-family-report", str(driver_v8.V7_REPORT_PATH_V8),
        "--v8-perturbation-basis-seed",
        str(driver_v8.PERTURBATION_BASIS_SEED_V8),
    ])
    v8_execution, _ = driver_v8.validate_frozen_execution_cli_v8(
        v8_argv, bundle,
    )
    recipe = wrapper_recipe_v9b(evidence)
    if driver_v1.canonical_sha256(recipe) != EXPECTED_WRAPPER_RECIPE_SHA256_V9B:
        raise RuntimeError("v9b wrapper recipe hash changed")
    return {
        "schema": "eggroll-es-v8-data44-replay-execution-v9b",
        "stage": "deterministic_replay", "arm": "middle_late",
        "data_seed": 44, "dry_run": stage.v9b_dry_run,
        "actual_experiment_name": EXPERIMENT_NAME_V9B,
        "reference_evidence": evidence,
        "wrapper_recipe": recipe,
        "underlying_v8_execution": v8_execution,
    }, actual_remaining


def set_active_v9b(bundle, execution):
    global _ACTIVE_EXECUTION_V9B
    if execution.get("data_seed") != 44:
        raise ValueError("v9b active execution changed seed")
    driver_v8.set_active_v8(bundle, execution["underlying_v8_execution"])
    _ACTIVE_EXECUTION_V9B = execution


def _v8_compatibility_journal_v9b(journal):
    compatible = copy.deepcopy(journal)
    compatible["schema"] = "eggroll-es-anchor-alpha-line-search-v8"
    compatible["snapshot"]["schema"] = (
        "eggroll-es-anchor-line-search-snapshot-v8"
    )
    compatible["snapshot"].pop("deterministic_replay_v9b", None)
    compatible["snapshot"]["implementation"] = {
        key: compatible["snapshot"]["implementation"][key]
        for key in driver_v8.V8_IMPLEMENTATION_PATHS
    }
    for key in V9B_POLICY:
        compatible["policy"].pop(key, None)
    compatible.pop("content_sha256_before_self_field", None)
    compatible["content_sha256_before_self_field"] = (
        driver_v1.canonical_sha256(compatible)
    )
    return compatible


def validate_completed_journal_v9b(journal):
    if (
        not isinstance(journal, dict)
        or journal.get("schema")
        != "eggroll-es-anchor-alpha-line-search-v9b"
        or journal.get("status") != "complete"
        or journal.get("in_progress") is not None
        or journal.get("targets") != [0.0]
        or journal.get("seeds") != driver_v8.PERTURBATION_SEEDS_V8
    ):
        raise RuntimeError("v9b journal is incomplete or changed basis/target")
    if journal.get("content_sha256_before_self_field") != (
        driver_v1.canonical_sha256({
            key: value for key, value in journal.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v9b journal content hash changed")
    if any(journal.get("policy", {}).get(key) != value
           for key, value in V9B_POLICY.items()):
        raise RuntimeError("v9b replay policy changed")
    snapshot = journal.get("snapshot", {})
    replay = snapshot.get("deterministic_replay_v9b")
    evidence = _failed_v8_seed44_evidence_v9b(
        replay_v9.V8_FAILED_REPORT_PATH_V9
    )
    recipe = wrapper_recipe_v9b(evidence)
    expected_replay = {
        "schema": "eggroll-es-v8-data44-replay-snapshot-v9b",
        "family": "exact_v8_data44_control",
        "stage": "deterministic_replay", "arm": "middle_late",
        "data_seed": 44,
        "actual_experiment_name": EXPERIMENT_NAME_V9B,
        "target_alphas": [0.0],
        "benchmark_treatment_applied": False,
        "selection_surface": "coefficient_and_raw_score_identity_only",
        "replay_cosine_threshold": 0.99,
        "reference_evidence": evidence,
        "wrapper_recipe": recipe,
    }
    if (
        snapshot.get("schema")
        != "eggroll-es-anchor-line-search-snapshot-v9b"
        or replay != expected_replay
        or driver_v1.canonical_sha256(recipe)
        != EXPECTED_WRAPPER_RECIPE_SHA256_V9B
    ):
        raise RuntimeError("v9b replay snapshot or recipe changed")
    actual_implementation = {
        key: anchor_v8.file_sha256(path)
        for key, path in V9B_IMPLEMENTATION_PATHS.items()
    }
    implementation = snapshot.get("implementation")
    if (
        not isinstance(implementation, dict)
        or set(implementation) != set(V9B_IMPLEMENTATION_PATHS)
        or implementation != actual_implementation
    ):
        raise RuntimeError("v9b implementation identity changed")
    inherited = driver_v8.validate_completed_journal_v8(
        _v8_compatibility_journal_v9b(journal)
    )
    if inherited["data_bootstrap_seed"] != 44:
        raise RuntimeError("v9b underlying v8 replay changed seed")
    return {
        **inherited, "stage": "deterministic_replay", "data_seed": 44,
        "content_sha256": journal["content_sha256_before_self_field"],
        "reference_evidence": evidence,
    }


def execute_line_search(*args, **kwargs):
    journal = driver_v8.execute_line_search(*args, **kwargs)
    execution = _ACTIVE_EXECUTION_V9B
    implementation = {
        key: anchor_v8.file_sha256(path)
        for key, path in V9B_IMPLEMENTATION_PATHS.items()
    }
    journal["schema"] = "eggroll-es-anchor-alpha-line-search-v9b"
    journal["snapshot"]["schema"] = (
        "eggroll-es-anchor-line-search-snapshot-v9b"
    )
    journal["snapshot"]["implementation"].update(implementation)
    journal["snapshot"]["deterministic_replay_v9b"] = {
        "schema": "eggroll-es-v8-data44-replay-snapshot-v9b",
        "family": "exact_v8_data44_control",
        "stage": "deterministic_replay", "arm": "middle_late",
        "data_seed": 44,
        "actual_experiment_name": EXPERIMENT_NAME_V9B,
        "target_alphas": [0.0],
        "benchmark_treatment_applied": False,
        "selection_surface": "coefficient_and_raw_score_identity_only",
        "replay_cosine_threshold": 0.99,
        "reference_evidence": copy.deepcopy(execution["reference_evidence"]),
        "wrapper_recipe": copy.deepcopy(execution["wrapper_recipe"]),
    }
    journal["policy"].update(V9B_POLICY)
    journal.pop("content_sha256_before_self_field", None)
    journal["content_sha256_before_self_field"] = (
        driver_v1.canonical_sha256(journal)
    )
    try:
        validate_completed_journal_v9b(journal)
    except Exception as error:
        journal["status"] = "failed"
        journal["failure"] = {
            "type": type(error).__name__, "message": str(error),
            "phase": "validating_complete_v9b_replay_audit",
        }
        journal.pop("content_sha256_before_self_field", None)
        driver_v1.atomic_write_json(kwargs["journal_path"], journal)
        raise
    driver_v1.atomic_write_json(kwargs["journal_path"], journal)
    return journal


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    bundle, remaining = anchor_v8.parse_frozen_layer_plan_cli_v8(argv)
    execution, actual_remaining = validate_frozen_execution_cli_v9b(
        remaining, bundle,
    )
    set_active_v9b(bundle, execution)
    if execution["dry_run"]:
        payload = {
            "schema": "eggroll-es-v8-data44-replay-dry-run-v9b",
            "data_seed": 44, "population_size": 32, "targets": [0.0],
            "actual_experiment_name": EXPERIMENT_NAME_V9B,
            "perturbation_basis_sha256": (
                driver_v8.PERTURBATION_BASIS_SHA256_V8
            ),
            "wrapper_recipe_sha256": driver_v1.canonical_sha256(
                execution["wrapper_recipe"]
            ),
        }
        print(json.dumps(payload, sort_keys=True))
        return payload
    old_argv = sys.argv
    old_anchor = driver_v1.anchor
    old_build = driver_v1.build_snapshot
    old_execute = driver_v1.execute_line_search
    sys.argv = [old_argv[0], *actual_remaining]
    driver_v1.anchor = anchor_v8
    driver_v1.build_snapshot = driver_v8.build_snapshot
    driver_v1.execute_line_search = execute_line_search
    try:
        driver_v1.main()
    finally:
        sys.argv = old_argv
        driver_v1.anchor = old_anchor
        driver_v1.build_snapshot = old_build
        driver_v1.execute_line_search = old_execute


if __name__ == "__main__":
    main()
