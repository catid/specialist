#!/usr/bin/env python3
"""Fail-closed exact data43 deterministic replay diagnostic v9."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import run_eggroll_es_anchor_line_search as driver_v1
import run_eggroll_es_anchor_line_search_v4 as driver_v4
import run_eggroll_es_anchor_line_search_v5 as driver_v5
import run_eggroll_es_anchor_line_search_v6 as driver_v6
import run_eggroll_es_anchor_stability_v8 as driver_v8
import report_eggroll_es_direction_stability_v8 as reporter_v8
import train_eggroll_es_specialist_anchor_v8 as anchor_v8


ROOT = Path(__file__).resolve().parent
DATA_SEED_V9 = 43
TARGETS_V9 = [0.0]
POPULATION_SIZE_V9 = driver_v8.POPULATION_SIZE_V8
PERTURBATION_BASIS_SEED_V9 = driver_v8.PERTURBATION_BASIS_SEED_V8
PERTURBATION_SEEDS_V9 = list(driver_v8.PERTURBATION_SEEDS_V8)
PERTURBATION_BASIS_SHA256_V9 = driver_v8.PERTURBATION_BASIS_SHA256_V8
REPLAY_COSINE_THRESHOLD_V9 = 0.99
MIDDLE_LATE_PLAN_SHA256_V9 = driver_v8.MIDDLE_LATE_PLAN_SHA256_V8
FROZEN_TRAIN_DATASET_V9 = driver_v8.FROZEN_TRAIN_DATASET_V8
FROZEN_EVAL_DATASET_V9 = driver_v8.FROZEN_EVAL_DATASET_V8
FROZEN_OUTPUT_DIRECTORY_V9 = driver_v8.FROZEN_OUTPUT_DIRECTORY_V8

V8_FAILED_REPORT_PATH_V9 = (
    ROOT / "experiments/eggroll_es_hpo/S6_SPLIT_SEED_POP32_V8_REPORT.json"
).resolve()
V8_FAILED_REPORT_FILE_SHA256_V9 = (
    "c1cbeb05ba25db1451d66c7d344f17cd86f535eb1a55d54584496029a0bca05d"
)
V8_FAILED_REPORT_CONTENT_SHA256_V9 = (
    "ae5c861705268ceadc9254c0ee8ddd367f6a2971ba45f35d495426b077a8c438"
)
ORIGINAL_V8_JOURNALS_V9 = {
    43: {
        "path": (
            ROOT / "experiments/eggroll_es_hpo/runs/"
            "snapshot794_layer_v8_middle_late_stability_"
            "data43_basis20260714/alpha_line_search.json"
        ).resolve(),
        "file_sha256": (
            "defd16591cbb137db09816a82c050716e24ec54ef90cb023339673df608e3da4"
        ),
        "content_sha256": (
            "98f14ba7525d2d33839421868b4c95a287f8334276a018c3a12dd76d20f54760"
        ),
        "coefficient_sha256": (
            "8c73a49ed5092aff908914a9c596995eac67043ccac724ef4a56a1e4ee71e8e6"
        ),
        "robust_plan_sha256": (
            "d32e92a71e3d13679e08fccc75ae6d56e1f4f5419e0b562615a4754ad88cec8e"
        ),
    },
    44: {
        "path": (
            ROOT / "experiments/eggroll_es_hpo/runs/"
            "snapshot794_layer_v8_middle_late_stability_"
            "data44_basis20260714/alpha_line_search.json"
        ).resolve(),
        "file_sha256": (
            "3f317ec14c47cd8fccea63dc09401e85416e8d4f0c2dfbb1d5e6d8f26385a30d"
        ),
        "content_sha256": (
            "f00f8f4d34c9885ed9904ad59d895bcb787d0fe1292977d7be8bf93841d68aa4"
        ),
        "coefficient_sha256": (
            "c63bf1e2081245a36b45704ac416675943bf2311b018f3489511d0ac231241a3"
        ),
        "robust_plan_sha256": (
            "66867658644c5ff406e72033eea61b5eb4aa007f67d5080e0da298d28f001c78"
        ),
    },
}
EXPECTED_V8_FAILED_EVIDENCE_BINDING_SHA256_V9 = (
    "7e5df754210699fdb46dfc4ede80c885ae38f0326649cf394519eddd54bb0e55"
)
EXPECTED_RECIPE_SHA256_V9 = (
    "a13aa3afec239f6c4f11429d1bcfd54b11c7a29d07632fdf7cc87df6c0f463ba"
)

V9_RECIPE_KEYS = {
    "model_name", "checkpoint", "train_dataset", "eval_dataset",
    "sigma", "population_size", "batch_size", "mini_batch_size",
    "max_tokens", "seed", "engine_count", "tp_per_engine", "gpu_ids",
    "eval_splits", "target_alphas", "anchor_prose_jsonl",
    "anchor_prose_report", "anchor_items_per_step",
    "anchor_max_input_tokens", "min_anchor_cosine", "ood_prose_jsonl",
    "ood_prose_max_input_tokens", "document_lcb_config_sha256",
    "reward_function_timeout", "output_directory", "experiment_name",
    "logging", "wandb_project", "data_seed", "replay_role",
    "perturbation_basis_seed", "perturbation_seed_count",
    "perturbation_basis_sha256", "v8_failed_evidence",
    "replay_cosine_threshold",
}
V9_IMPLEMENTATION_PATHS = {
    **driver_v8.V8_IMPLEMENTATION_PATHS,
    "distributed_driver_v9": Path(__file__).resolve(),
    "replay_reporter_v9": (
        ROOT / "report_eggroll_es_deterministic_replay_v9.py"
    ),
    "protocol_v9": (
        ROOT / "experiments/eggroll_es_hpo/"
        "S6_DETERMINISTIC_REPLAY_V9_PROTOCOL.md"
    ),
    "contract_tests_v9": ROOT / "test_eggroll_es_deterministic_replay_v9.py",
    "v8_failed_report": V8_FAILED_REPORT_PATH_V9,
}
_ACTIVE_LAYER_PLAN_BUNDLE = None
_ACTIVE_EXECUTION_V9 = None


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _v8_failed_evidence_v9(path):
    path = Path(path).resolve()
    driver_v8.offline_audit._assert_no_heldout(
        str(path), "v9 failed-v8-report path",
    )
    if (
        path != V8_FAILED_REPORT_PATH_V9
        or _file_sha256(path) != V8_FAILED_REPORT_FILE_SHA256_V9
    ):
        raise ValueError("v9 requires the exact failed v8 family report")
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("v9 cannot read the failed v8 report") from error
    if (
        report.get("content_sha256_before_self_field")
        != V8_FAILED_REPORT_CONTENT_SHA256_V9
        or report.get("content_sha256_before_self_field")
        != driver_v1.canonical_sha256({
            key: value for key, value in report.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise ValueError("v9 failed-v8 report content hash changed")
    expected_paths = [
        str(ORIGINAL_V8_JOURNALS_V9[seed]["path"])
        for seed in (43, 44)
    ]
    rebuilt = reporter_v8.build_report(expected_paths)
    if report != rebuilt:
        raise ValueError("v9 failed-v8 report differs from its strict journals")
    if (
        report.get("schema")
        != "eggroll-es-same-basis-direction-stability-report-v8"
        or report.get("preregistered_threshold") != 0.5
        or report.get("same_basis_coefficient_cosine")
        != 0.4276943787514416
        or report.get("passed") is not False
        or report.get("coverage", {}).get("data_bootstrap_seeds") != [43, 44]
        or report.get("coverage", {}).get("population_size") != 32
        or report.get("coverage", {}).get("target_alphas") != [0.0]
    ):
        raise ValueError("v9 requires the exact formal v8 failure decision")
    report_runs = report.get("runs")
    if not isinstance(report_runs, list) or len(report_runs) != 2:
        raise ValueError("v9 failed-v8 report omitted a strict journal")
    runs = []
    for row in sorted(report_runs, key=lambda item: item["data_bootstrap_seed"]):
        seed = row.get("data_bootstrap_seed")
        expected = ORIGINAL_V8_JOURNALS_V9.get(seed)
        if expected is None:
            raise ValueError("v9 failed-v8 report changed its seed set")
        journal_path = Path(row.get("journal", "")).resolve()
        if (
            journal_path != expected["path"]
            or _file_sha256(journal_path) != expected["file_sha256"]
            or row.get("journal_file_sha256") != expected["file_sha256"]
            or row.get("content_sha256") != expected["content_sha256"]
            or row.get("coefficient_sha256") != expected["coefficient_sha256"]
            or row.get("robust_plan_sha256") != expected["robust_plan_sha256"]
        ):
            raise ValueError("v9 strict v8 journal identity changed")
        runs.append({
            "data_seed": seed,
            "journal": str(journal_path),
            "journal_file_sha256": expected["file_sha256"],
            "content_sha256": expected["content_sha256"],
            "coefficient_sha256": expected["coefficient_sha256"],
            "robust_plan_sha256": expected["robust_plan_sha256"],
        })
    binding = {
        "schema": "eggroll-es-failed-v8-evidence-binding-v9",
        "report_path": str(path),
        "report_file_sha256": V8_FAILED_REPORT_FILE_SHA256_V9,
        "report_content_sha256": V8_FAILED_REPORT_CONTENT_SHA256_V9,
        "runs": runs,
        "formal_v8_result": {
            "same_basis_coefficient_cosine": 0.4276943787514416,
            "threshold": 0.5,
            "passed": False,
        },
        "v9_rationale": "exact_data43_replay_before_variance_reduction",
    }
    binding["binding_sha256"] = driver_v1.canonical_sha256(binding)
    if (
        binding["binding_sha256"]
        != EXPECTED_V8_FAILED_EVIDENCE_BINDING_SHA256_V9
    ):
        raise ValueError("v9 failed-v8 evidence binding changed")
    return binding


def validate_effective_anchor_api(module=anchor_v8):
    required = (
        "coefficient_sha256", "load_anchor_prose", "load_trainer",
        "load_frozen_layer_plan_v8", "validate_robust_plan_v8",
    )
    missing = [
        name for name in required if not callable(getattr(module, name, None))
    ]
    if missing:
        raise RuntimeError("v9 exact v8 anchor adapter API is incomplete")
    if module.WORKER_EXTENSION != (
        "eggroll_es_worker_v8.SplitSeedPopulation32AuditWorkerExtensionV8"
    ):
        raise RuntimeError("v9 replay did not retain the exact v8 worker")
    return required


def _experiment_name_v9():
    return (
        "snapshot794_layer_v9_middle_late_exact_replay_"
        "data43_basis20260714_retry1"
    )


def frozen_recipe_v9(v8_evidence):
    return {
        "model_name": str((ROOT / "models/Qwen3.6-35B-A3B").resolve()),
        "checkpoint": None,
        "train_dataset": str(FROZEN_TRAIN_DATASET_V9),
        "eval_dataset": str(FROZEN_EVAL_DATASET_V9),
        "sigma": 0.0003,
        "population_size": POPULATION_SIZE_V9,
        "batch_size": 64, "mini_batch_size": 64, "max_tokens": 32,
        "seed": DATA_SEED_V9,
        "engine_count": 4, "tp_per_engine": 1, "gpu_ids": [0, 1, 2, 3],
        "eval_splits": ["validation", "ood_qa"],
        "target_alphas": [0.0],
        "anchor_prose_jsonl": str(
            (ROOT / "data/general_prose_anchor_v1.jsonl").resolve()
        ),
        "anchor_prose_report": str(
            (ROOT / "data/general_prose_anchor_v1.report.json").resolve()
        ),
        "anchor_items_per_step": 128,
        "anchor_max_input_tokens": 512,
        "min_anchor_cosine": 0.8,
        "ood_prose_jsonl": str((ROOT / "data/ood_prose_v3.jsonl").resolve()),
        "ood_prose_max_input_tokens": 1024,
        "document_lcb_config_sha256": (
            anchor_v8.anchor_v7.anchor_v6.anchor_v5
            .DOCUMENT_LCB_CONFIG_SHA256_V5
        ),
        "reward_function_timeout": 10,
        "output_directory": str(FROZEN_OUTPUT_DIRECTORY_V9),
        "experiment_name": _experiment_name_v9(),
        "logging": "none", "wandb_project": "specialist-eggroll-es",
        "data_seed": DATA_SEED_V9,
        "replay_role": "exact_v8_data43_determinism_replay",
        "perturbation_basis_seed": PERTURBATION_BASIS_SEED_V9,
        "perturbation_seed_count": POPULATION_SIZE_V9,
        "perturbation_basis_sha256": PERTURBATION_BASIS_SHA256_V9,
        "v8_failed_evidence": copy.deepcopy(v8_evidence),
        "replay_cosine_threshold": REPLAY_COSINE_THRESHOLD_V9,
    }


def validate_frozen_execution_cli_v9(argv, bundle):
    inherited = driver_v5.validate_frozen_execution_cli_v5(argv)
    stage_parser = argparse.ArgumentParser(add_help=False)
    stage_parser.add_argument(
        "--v9-stage", choices=("deterministic_replay",),
    )
    stage_parser.add_argument("--v9-v8-failed-report")
    stage_parser.add_argument("--v9-perturbation-basis-seed", type=int)
    stage_parser.add_argument("--v9-dry-run", action="store_true")
    stage, remaining = stage_parser.parse_known_args(list(argv))
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--target-alphas")
    parser.add_argument("--train-dataset", default=str(FROZEN_TRAIN_DATASET_V9))
    parser.add_argument("--eval-dataset", default=str(FROZEN_EVAL_DATASET_V9))
    parser.add_argument("--reward-function-timeout", type=int, default=10)
    parser.add_argument(
        "--output-directory", default=str(FROZEN_OUTPUT_DIRECTORY_V9),
    )
    parser.add_argument("--experiment-name")
    parser.add_argument("--logging", default="none")
    parser.add_argument("--wandb-project", default="specialist-eggroll-es")
    recipe, _ = parser.parse_known_args(list(argv))
    metadata = anchor_v8.validate_frozen_layer_plan_bundle_v8(bundle)
    if stage.v9_stage != "deterministic_replay":
        raise ValueError("v9 requires --v9-stage deterministic_replay")
    if recipe.seed != DATA_SEED_V9:
        raise ValueError("v9 exact replay seed must be 43")
    if stage.v9_perturbation_basis_seed != PERTURBATION_BASIS_SEED_V9:
        raise ValueError("v9 perturbation basis seed changed")
    if recipe.target_alphas is None:
        raise ValueError("v9 requires explicit --target-alphas 0")
    targets = driver_v1.parse_target_alphas(recipe.target_alphas)
    if targets != TARGETS_V9:
        raise ValueError("v9 replay target must be exactly alpha zero")
    if (
        metadata["plan"] != "middle_late"
        or bundle["plan_sha256"] != MIDDLE_LATE_PLAN_SHA256_V9
        or recipe.population_size != POPULATION_SIZE_V9
        or recipe.batch_size != 64
        or Path(recipe.train_dataset).resolve() != FROZEN_TRAIN_DATASET_V9
        or Path(recipe.eval_dataset).resolve() != FROZEN_EVAL_DATASET_V9
        or recipe.reward_function_timeout != 10
        or Path(recipe.output_directory).resolve()
        != FROZEN_OUTPUT_DIRECTORY_V9
        or recipe.experiment_name != _experiment_name_v9()
        or recipe.logging != "none"
        or recipe.wandb_project != "specialist-eggroll-es"
    ):
        raise ValueError("v9 exact-replay runtime recipe changed")
    if stage.v9_v8_failed_report is None:
        raise ValueError("v9 requires the exact failed v8 report")
    evidence = _v8_failed_evidence_v9(stage.v9_v8_failed_report)
    frozen = frozen_recipe_v9(evidence)
    if driver_v1.canonical_sha256(frozen) != EXPECTED_RECIPE_SHA256_V9:
        raise RuntimeError("v9 frozen recipe hash changed")
    execution = {
        **inherited, **copy.deepcopy(frozen),
        "schema": "eggroll-es-deterministic-replay-execution-v9",
        "stage": "deterministic_replay", "arm": "middle_late",
        "dry_run": stage.v9_dry_run,
    }
    if {key: execution[key] for key in V9_RECIPE_KEYS} != frozen:
        raise RuntimeError("v9 effective replay recipe changed")
    return execution, remaining


def set_active_v9(bundle, execution):
    global _ACTIVE_LAYER_PLAN_BUNDLE, _ACTIVE_EXECUTION_V9
    metadata = anchor_v8.validate_frozen_layer_plan_bundle_v8(bundle)
    if execution.get("arm") != metadata["plan"]:
        raise ValueError("v9 execution and layer plan differ")
    driver_v5.set_active_layer_plan_bundle_v5(bundle)
    anchor_v8.set_default_layer_plan_bundle_v8(bundle)
    _ACTIVE_LAYER_PLAN_BUNDLE = bundle
    _ACTIVE_EXECUTION_V9 = execution


def build_snapshot(*args, **kwargs):
    bundle = _ACTIVE_LAYER_PLAN_BUNDLE
    execution = _ACTIVE_EXECUTION_V9
    metadata = anchor_v8.validate_frozen_layer_plan_bundle_v8(bundle)
    if not isinstance(execution, dict):
        raise RuntimeError("v9 has no active replay execution")
    snapshot = driver_v5.build_snapshot(*args, **kwargs)
    implementation = {
        key: anchor_v8.file_sha256(path)
        for key, path in V9_IMPLEMENTATION_PATHS.items()
    }
    snapshot["implementation"].update(implementation)
    snapshot["schema"] = "eggroll-es-anchor-line-search-snapshot-v9"
    snapshot["deterministic_replay_v9"] = {
        "schema": "eggroll-es-deterministic-replay-snapshot-v9",
        "family": "exact_v8_data43_replay",
        "stage": "deterministic_replay",
        "arm": "middle_late", "layers": metadata["layers"],
        "data_seed": DATA_SEED_V9,
        "reference_role": "original_v8_data43",
        "replay_role": "v9_exact_data43",
        "perturbation_basis_seed": PERTURBATION_BASIS_SEED_V9,
        "perturbation_seed_count": POPULATION_SIZE_V9,
        "perturbation_seeds": list(PERTURBATION_SEEDS_V9),
        "perturbation_basis_sha256": PERTURBATION_BASIS_SHA256_V9,
        "target_alphas": [0.0],
        "benchmark_treatment_applied": False,
        "selection_surface": "coefficient_and_raw_score_identity_only",
        "replay_cosine_threshold": REPLAY_COSINE_THRESHOLD_V9,
        "required_exact_identities": [
            "coefficient_sha256", "domain_scores_sha256",
            "anchor_scores_sha256",
        ],
        "v8_failed_evidence": copy.deepcopy(execution["v8_failed_evidence"]),
        "recipe": {
            key: copy.deepcopy(execution[key]) for key in V9_RECIPE_KEYS
        },
        "plan_sha256": bundle["plan_sha256"],
        "plan_file_sha256": bundle["file_sha256"],
        "source_unit_count": metadata["source_unit_count"],
        "runtime_selected_parameter_count": metadata[
            "runtime_selected_parameter_count"
        ],
        "selected_element_count": metadata["selected_element_count"],
        "selected_byte_count": metadata["selected_byte_count"],
        "implementation_bundle_sha256": driver_v1.canonical_sha256(
            implementation
        ),
    }
    return snapshot


V9_POLICY = {
    "deterministic_replay_family_v9": "exact_v8_data43_pop32",
    "stage_v9": "deterministic_replay",
    "target_alpha_zero_only_v9": True,
    "benchmark_selection_forbidden_v9": True,
    "same_perturbation_basis_required_v9": True,
    "requires_failed_v8_family_v9": True,
    "exact_raw_score_identity_required_v9": True,
    "replay_coefficient_cosine_threshold_v9": 0.99,
}
INHERITED_POLICY_V9 = {
    "alpha_order": "zero_then_strictly_increasing",
    "branching": False, "resume": False, "rollback": False,
    "selection_during_execution": False,
    "ood_qa_max_degradation": 0.0, "ood_prose_max_degradation": 0.0,
    "bf16_alpha_semantics": "path_dependent_monotonic_pilot",
    "direct_alpha_confirmation_required": True,
    "frozen_layer_plan_required": True,
    "dense_gold_reward_required": True,
    "document_lcb_anchor_required": True,
    "optimization_data": "train_and_anchor_only",
    "ood_validation_heldout_as_objective": False,
}


def _v8_compatibility_journal_v9(journal):
    compatible = copy.deepcopy(journal)
    compatible["schema"] = "eggroll-es-anchor-alpha-line-search-v8"
    snapshot = compatible["snapshot"]
    snapshot["schema"] = "eggroll-es-anchor-line-search-snapshot-v8"
    snapshot.pop("deterministic_replay_v9", None)
    implementation = {
        key: snapshot["implementation"][key]
        for key in driver_v8.V8_IMPLEMENTATION_PATHS
    }
    snapshot["implementation"] = implementation
    v7_evidence = driver_v8._v7_family_evidence_v8(
        driver_v8.V7_REPORT_PATH_V8
    )
    recipe_v8 = driver_v8.frozen_recipe_v8(DATA_SEED_V9, v7_evidence)
    spec = anchor_v8.FROZEN_STABILITY_PLANS_V8[
        MIDDLE_LATE_PLAN_SHA256_V9
    ]
    snapshot["split_seed_stability_v8"] = {
        "schema": "eggroll-es-split-seed-pop32-snapshot-v8",
        "family": "middle_late_same_basis_cross_data_seed",
        "stage": "stability", "arm": "middle_late",
        "layers": [20, 21, 22, 23],
        "data_bootstrap_seed_pair": [43, 44],
        "data_bootstrap_seed": DATA_SEED_V9,
        "perturbation_basis_seed": PERTURBATION_BASIS_SEED_V9,
        "perturbation_seed_count": POPULATION_SIZE_V9,
        "perturbation_seeds": list(PERTURBATION_SEEDS_V9),
        "perturbation_basis_sha256": PERTURBATION_BASIS_SHA256_V9,
        "target_alphas": [0.0],
        "benchmark_treatment_applied": False,
        "selection_surface": "same_basis_coefficients_only",
        "coefficient_cosine_threshold": 0.5,
        "recipe": recipe_v8,
        "plan_sha256": MIDDLE_LATE_PLAN_SHA256_V9,
        "plan_file_sha256": spec["file_sha256"],
        "source_unit_count": 35,
        "runtime_selected_parameter_count": 23,
        "selected_element_count": 142_999_552,
        "selected_byte_count": 285_999_104,
        "implementation_bundle_sha256": driver_v1.canonical_sha256(
            implementation
        ),
    }
    for key in V9_POLICY:
        compatible["policy"].pop(key, None)
    compatible["policy"].update({
        "split_seed_stability_family_v8": "middle_late_pop32_same_basis",
        "stage_v8": "stability", "target_alpha_zero_only_v8": True,
        "benchmark_selection_forbidden_v8": True,
        "same_perturbation_basis_required_v8": True,
        "cross_data_seed_coefficient_cosine_threshold_v8": 0.5,
        "requires_complete_v7_family_v8": True,
    })
    compatible.pop("content_sha256_before_self_field", None)
    compatible["content_sha256_before_self_field"] = (
        driver_v1.canonical_sha256(compatible)
    )
    return compatible


def _validate_inherited_zero_target_v9(journal):
    return driver_v8.validate_completed_journal_v8(
        _v8_compatibility_journal_v9(journal)
    )


def validate_completed_journal_v9(journal):
    if (
        not isinstance(journal, dict)
        or set(journal) != {
            "schema", "status", "in_progress", "policy", "targets",
            "trainer_configuration", "snapshot", "coefficient_plan",
            "states", "seeds", "content_sha256_before_self_field",
        }
        or journal.get("schema")
        != "eggroll-es-anchor-alpha-line-search-v9"
        or journal.get("status") != "complete"
        or journal.get("in_progress") is not None
        or journal.get("targets") != [0.0]
        or journal.get("seeds") != PERTURBATION_SEEDS_V9
    ):
        raise RuntimeError("v9 journal is incomplete or changed basis/target")
    if journal.get("content_sha256_before_self_field") != (
        driver_v1.canonical_sha256({
            key: value for key, value in journal.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v9 journal content hash changed")
    policy = journal.get("policy")
    if (
        not isinstance(policy, dict)
        or set(policy) != set(INHERITED_POLICY_V9) | set(V9_POLICY)
        or any(policy.get(key) != value for key, value in {
            **INHERITED_POLICY_V9, **V9_POLICY,
        }.items())
    ):
        raise RuntimeError("v9 deterministic-replay policy changed")
    snapshot = journal.get("snapshot")
    replay = (
        snapshot.get("deterministic_replay_v9")
        if isinstance(snapshot, dict) else None
    )
    expected_replay_keys = {
        "schema", "family", "stage", "arm", "layers", "data_seed",
        "reference_role", "replay_role", "perturbation_basis_seed",
        "perturbation_seed_count", "perturbation_seeds",
        "perturbation_basis_sha256", "target_alphas",
        "benchmark_treatment_applied", "selection_surface",
        "replay_cosine_threshold", "required_exact_identities",
        "v8_failed_evidence", "recipe", "plan_sha256",
        "plan_file_sha256", "source_unit_count",
        "runtime_selected_parameter_count", "selected_element_count",
        "selected_byte_count", "implementation_bundle_sha256",
    }
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("schema")
        != "eggroll-es-anchor-line-search-snapshot-v9"
        or set(snapshot) != {
            "schema", "train", "evaluations", "anchor",
            "fixed_train_batch", "implementation", "recipe",
            "distributed_update_v3", "frozen_layer_plan_v4",
            "dense_gold_reward_v4", "distributed_update_v4",
            "document_lcb_anchor_v5", "deterministic_replay_v9",
        }
        or not isinstance(replay, dict)
        or set(replay) != expected_replay_keys
        or replay.get("schema")
        != "eggroll-es-deterministic-replay-snapshot-v9"
        or replay.get("family") != "exact_v8_data43_replay"
        or replay.get("stage") != "deterministic_replay"
        or replay.get("arm") != "middle_late"
        or replay.get("layers") != [20, 21, 22, 23]
        or replay.get("data_seed") != DATA_SEED_V9
        or replay.get("reference_role") != "original_v8_data43"
        or replay.get("replay_role") != "v9_exact_data43"
        or replay.get("perturbation_basis_seed")
        != PERTURBATION_BASIS_SEED_V9
        or replay.get("perturbation_seed_count") != POPULATION_SIZE_V9
        or replay.get("perturbation_seeds") != PERTURBATION_SEEDS_V9
        or replay.get("perturbation_basis_sha256")
        != PERTURBATION_BASIS_SHA256_V9
        or replay.get("target_alphas") != [0.0]
        or replay.get("benchmark_treatment_applied") is not False
        or replay.get("selection_surface")
        != "coefficient_and_raw_score_identity_only"
        or replay.get("replay_cosine_threshold")
        != REPLAY_COSINE_THRESHOLD_V9
        or replay.get("required_exact_identities") != [
            "coefficient_sha256", "domain_scores_sha256",
            "anchor_scores_sha256",
        ]
        or replay.get("plan_sha256") != MIDDLE_LATE_PLAN_SHA256_V9
        or replay.get("source_unit_count") != 35
        or replay.get("runtime_selected_parameter_count") != 23
        or replay.get("selected_element_count") != 142_999_552
        or replay.get("selected_byte_count") != 285_999_104
    ):
        raise RuntimeError("v9 deterministic-replay snapshot changed")
    evidence = _v8_failed_evidence_v9(V8_FAILED_REPORT_PATH_V9)
    expected_recipe = frozen_recipe_v9(evidence)
    if (
        replay.get("v8_failed_evidence") != evidence
        or replay.get("recipe") != expected_recipe
        or driver_v1.canonical_sha256(expected_recipe)
        != EXPECTED_RECIPE_SHA256_V9
    ):
        raise RuntimeError("v9 persisted evidence or recipe changed")
    expected_snapshot_recipe = {
        key: expected_recipe[key] for key in (
            "model_name", "checkpoint", "sigma", "population_size",
            "batch_size", "mini_batch_size", "max_tokens", "seed",
            "min_anchor_cosine", "anchor_items_per_step", "target_alphas",
        )
    }
    if snapshot.get("recipe") != expected_snapshot_recipe:
        raise RuntimeError("v9 effective snapshot recipe changed")
    spec = anchor_v8.FROZEN_STABILITY_PLANS_V8[
        MIDDLE_LATE_PLAN_SHA256_V9
    ]
    if replay.get("plan_file_sha256") != spec["file_sha256"]:
        raise RuntimeError("v9 persisted layer-plan file changed")
    actual_implementation = {
        key: anchor_v8.file_sha256(path)
        for key, path in V9_IMPLEMENTATION_PATHS.items()
    }
    implementation = snapshot.get("implementation")
    if (
        not isinstance(implementation, dict)
        or set(implementation) != set(V9_IMPLEMENTATION_PATHS)
        or implementation != actual_implementation
        or replay.get("implementation_bundle_sha256")
        != driver_v1.canonical_sha256(actual_implementation)
    ):
        raise RuntimeError("v9 implementation identity changed")
    coefficient_plan = journal.get("coefficient_plan", {})
    for key in (
        "domain_scores_v5", "anchor_scores_v5",
        "document_lcb_anchor_v5", "robust_plan_binding_v5",
    ):
        if key not in coefficient_plan:
            raise RuntimeError("v9 replay omitted raw coefficient evidence")
    # The strict v8 completed validator owns the single permitted v6 scope.
    # Wrapping it here would be a forbidden reentrant legacy-audit scope.
    inherited = _validate_inherited_zero_target_v9(journal)
    if inherited["data_bootstrap_seed"] != DATA_SEED_V9:
        raise RuntimeError("v9 inherited data seed changed")
    return {
        **inherited,
        "arm": "middle_late", "stage": "deterministic_replay",
        "data_seed": DATA_SEED_V9,
        "perturbation_basis_sha256": PERTURBATION_BASIS_SHA256_V9,
        "v8_failed_evidence_binding_sha256": evidence["binding_sha256"],
        "content_sha256": journal["content_sha256_before_self_field"],
    }


def execute_line_search(*args, **kwargs):
    kwargs = dict(kwargs)
    execution = _ACTIVE_EXECUTION_V9
    expected_inherited_seeds = np.random.default_rng(
        seed=DATA_SEED_V9
    ).integers(
        0, 2**30, size=POPULATION_SIZE_V9, dtype=np.int64,
    ).tolist()
    if (
        not isinstance(execution, dict)
        or execution.get("data_seed") != DATA_SEED_V9
        or kwargs.get("seeds") != expected_inherited_seeds
    ):
        raise RuntimeError("v9 inherited driver population schedule changed")
    kwargs["seeds"] = list(PERTURBATION_SEEDS_V9)
    with driver_v6.scoped_legacy_audit_v6():
        journal = driver_v4.execute_line_search(*args, **kwargs)
    trainer = args[0] if args else kwargs["trainer"]
    plan = trainer._latest_anchor_plan
    anchor_v8.validate_robust_plan_v8(plan, recompute_numeric=True)
    binding_v5 = anchor_v8.anchor_v7.anchor_v6.anchor_v5.validate_robust_plan_v5(
        plan, recompute_numeric=True,
    )
    journal["schema"] = "eggroll-es-anchor-alpha-line-search-v9"
    journal["snapshot"]["schema"] = (
        "eggroll-es-anchor-line-search-snapshot-v9"
    )
    journal["policy"].update({
        "document_lcb_anchor_required": True,
        "optimization_data": "train_and_anchor_only",
        "ood_validation_heldout_as_objective": False,
        **V9_POLICY,
    })
    coefficient_plan = journal["coefficient_plan"]
    coefficient_plan["domain_scores_v5"] = list(plan["domain_scores"])
    coefficient_plan["anchor_scores_v5"] = list(plan["anchor_scores"])
    coefficient_plan["document_lcb_anchor_v5"] = plan[
        "document_lcb_anchor_v5"
    ]
    coefficient_plan["robust_plan_binding_v5"] = binding_v5
    journal.pop("content_sha256_before_self_field", None)
    journal["content_sha256_before_self_field"] = (
        driver_v1.canonical_sha256(journal)
    )
    try:
        validate_completed_journal_v9(journal)
    except Exception as error:
        journal["status"] = "failed"
        journal["failure"] = {
            "type": type(error).__name__, "message": str(error),
            "phase": "validating_complete_v9_replay_audit",
        }
        journal.pop("content_sha256_before_self_field", None)
        driver_v1.atomic_write_json(kwargs["journal_path"], journal)
        raise
    driver_v1.atomic_write_json(kwargs["journal_path"], journal)
    return journal


def main(argv=None):
    validate_effective_anchor_api()
    argv = list(sys.argv[1:] if argv is None else argv)
    bundle, remaining = anchor_v8.parse_frozen_layer_plan_cli_v8(argv)
    execution, remaining = validate_frozen_execution_cli_v9(remaining, bundle)
    set_active_v9(bundle, execution)
    if execution["dry_run"]:
        payload = {
            "schema": "eggroll-es-deterministic-replay-dry-run-v9",
            "arm": "middle_late", "stage": "deterministic_replay",
            "data_seed": DATA_SEED_V9,
            "perturbation_basis_seed": PERTURBATION_BASIS_SEED_V9,
            "perturbation_basis_sha256": PERTURBATION_BASIS_SHA256_V9,
            "population_size": POPULATION_SIZE_V9,
            "targets": [0.0],
            "replay_cosine_threshold": REPLAY_COSINE_THRESHOLD_V9,
            "recipe_sha256": driver_v1.canonical_sha256({
                key: execution[key] for key in V9_RECIPE_KEYS
            }),
            "v8_failed_evidence_binding_sha256": execution[
                "v8_failed_evidence"
            ]["binding_sha256"],
        }
        print(json.dumps(payload, sort_keys=True))
        return payload
    old_argv = sys.argv
    old_anchor = driver_v1.anchor
    old_build = driver_v1.build_snapshot
    old_execute = driver_v1.execute_line_search
    sys.argv = [old_argv[0], *remaining]
    driver_v1.anchor = anchor_v8
    driver_v1.build_snapshot = build_snapshot
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
