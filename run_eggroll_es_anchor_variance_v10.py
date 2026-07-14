#!/usr/bin/env python3
"""Fail-closed antithetic crossed D43/D44 x A43/A44 diagnostic v10."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import report_eggroll_es_deterministic_replay_v9 as reporter_v9
import run_eggroll_es_anchor_line_search as driver_v1
import run_eggroll_es_anchor_line_search_v4 as driver_v4
import run_eggroll_es_anchor_line_search_v5 as driver_v5
import run_eggroll_es_anchor_line_search_v6 as driver_v6
import run_eggroll_es_anchor_stability_v8 as driver_v8
import train_eggroll_es_specialist_anchor_v10 as anchor_v10


ROOT = Path(__file__).resolve().parent
DATA_SEED_V10 = 43
POPULATION_SIZE_V10 = 32
PERTURBATION_BASIS_SEED_V10 = driver_v8.PERTURBATION_BASIS_SEED_V8
PERTURBATION_SEEDS_V10 = list(driver_v8.PERTURBATION_SEEDS_V8)
PERTURBATION_BASIS_SHA256_V10 = driver_v8.PERTURBATION_BASIS_SHA256_V8
MIDDLE_LATE_PLAN_SHA256_V10 = driver_v8.MIDDLE_LATE_PLAN_SHA256_V8
FROZEN_TRAIN_DATASET_V10 = driver_v8.FROZEN_TRAIN_DATASET_V8
FROZEN_EVAL_DATASET_V10 = driver_v8.FROZEN_EVAL_DATASET_V8
FROZEN_OUTPUT_DIRECTORY_V10 = driver_v8.FROZEN_OUTPUT_DIRECTORY_V8
EXPERIMENT_NAME_V10 = (
    "snapshot794_layer_v10_middle_late_antithetic_cross_"
    "d43d44_a43a44_basis20260714"
)
V9_REPORT_PATH_V10 = (
    ROOT / "experiments/eggroll_es_hpo/S6_DETERMINISTIC_REPLAY_V9_REPORT.json"
).resolve()
V9_REPORT_FILE_SHA256_V10 = (
    "8dac33ad828a29021f107074a79aa536267349a6903f3f3b8d9a89146e7859a3"
)
V9_REPORT_CONTENT_SHA256_V10 = (
    "7871c2d959e9b70f9d0912a1af5571767da155fd3ffc5a3588ab6e8cbdb7a50f"
)
EXPECTED_V9_EVIDENCE_BINDING_SHA256_V10 = (
    "a036742002868326b43090adcbf7529260ae49265158f3999b5cc2d68c3b8ff5"
)
EXPECTED_RECIPE_SHA256_V10 = (
    "f25af4c9c6092ce0909458c22154b9038de455518dcbf1f8b7cd30cf141d76aa"
)
V10_RECIPE_KEYS = {
    "model_name", "checkpoint", "train_dataset", "eval_dataset", "sigma",
    "population_size", "base_direction_count", "unique_signed_direction_count",
    "actual_perturb_restore_cycle_count", "domain_signed_score_count",
    "anchor_signed_response_count",
    "batch_size", "domain_manifest_rows", "mini_batch_size", "max_tokens",
    "seed", "engine_count", "tp_per_engine", "gpu_ids", "eval_splits",
    "target_alphas", "anchor_prose_jsonl", "anchor_prose_report",
    "anchor_items_per_step", "anchor_max_input_tokens", "min_anchor_cosine",
    "anchor_generation_seeds", "ood_prose_jsonl",
    "ood_prose_max_input_tokens", "document_lcb_config_sha256",
    "reward_function_timeout", "output_directory", "experiment_name",
    "logging", "wandb_project", "perturbation_basis_seed",
    "perturbation_basis_sha256", "domain_manifests",
    "combined_domain_manifest_sha256", "v9_determinism_evidence",
}
V10_IMPLEMENTATION_PATHS = {
    **driver_v8.V8_IMPLEMENTATION_PATHS,
    "anchor_trainer": Path(anchor_v10.__file__).resolve(),
    "distributed_driver_v10": Path(__file__).resolve(),
    "distributed_trainer_v10": Path(anchor_v10.__file__).resolve(),
    "distributed_worker_v10": ROOT / "eggroll_es_worker_v10.py",
    "variance_reporter_v10": ROOT / "report_eggroll_es_variance_v10.py",
    "protocol_v10": (
        ROOT / "experiments/eggroll_es_hpo/"
        "S6_ANTITHETIC_CROSSED_V10_PROTOCOL.md"
    ),
    "contract_tests_v10": ROOT / "test_eggroll_es_variance_v10.py",
    "v9_determinism_report": V9_REPORT_PATH_V10,
    "v9_determinism_reporter": Path(reporter_v9.__file__).resolve(),
}
V10_POLICY = {
    "antithetic_crossed_family_v10": "D43_D44_x_A43_A44",
    "stage_v10": "variance",
    "target_alpha_zero_only_v10": True,
    "benchmark_selection_forbidden_v10": True,
    "paired_plus_minus_required_v10": True,
    "crossed_domain_anchor_required_v10": True,
    "requires_deterministic_v9_pass_v10": True,
}
INHERITED_POLICY_V10 = {
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
_ACTIVE_LAYER_PLAN_BUNDLE = None
_ACTIVE_EXECUTION_V10 = None
_ORIGINAL_BUILD_TRAIN_LOADER = driver_v1.base.build_train_loader


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _v9_determinism_evidence_v10(path):
    path = Path(path).resolve()
    driver_v8.offline_audit._assert_no_heldout(
        str(path), "v10 v9-report path",
    )
    if path != V9_REPORT_PATH_V10 or _file_sha256(path) != V9_REPORT_FILE_SHA256_V10:
        raise ValueError("v10 requires the exact passing v9 report")
    report = json.loads(path.read_text())
    if (
        report.get("content_sha256_before_self_field")
        != V9_REPORT_CONTENT_SHA256_V10
        or report.get("content_sha256_before_self_field")
        != driver_v1.canonical_sha256({
            key: value for key, value in report.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise ValueError("v10 v9 report content hash changed")
    runs = report.get("runs", [])
    if len(runs) != 2:
        raise ValueError("v10 v9 report changed run coverage")
    rebuilt = reporter_v9.build_report(
        runs[0]["journal"], runs[1]["journal"],
    )
    if rebuilt != report:
        raise ValueError("v10 v9 report differs from its strict journals")
    if (
        report.get("schema") != "eggroll-es-deterministic-replay-report-v9"
        or report.get("passed") is not True
        or report.get("all_exact_identities_match") is not True
        or report.get("cosines") != {
            "coefficient": 1.0,
            "standardized_anchor_score": 1.0,
            "standardized_domain_score": 1.0,
        }
    ):
        raise ValueError("v10 requires the exact deterministic v9 pass")
    binding = {
        "schema": "eggroll-es-v9-determinism-evidence-binding-v10",
        "report_path": str(path),
        "report_file_sha256": V9_REPORT_FILE_SHA256_V10,
        "report_content_sha256": V9_REPORT_CONTENT_SHA256_V10,
        "runs": [{
            key: row[key] for key in (
                "role", "journal", "journal_file_sha256", "content_sha256",
                "coefficient_sha256", "domain_scores_sha256",
                "anchor_scores_sha256", "robust_plan_sha256",
            )
        } for row in runs],
        "formal_v9_result": {
            "all_cosines": 1.0,
            "all_exact_identities_match": True,
            "passed": True,
        },
    }
    binding["binding_sha256"] = driver_v1.canonical_sha256(binding)
    if binding["binding_sha256"] != EXPECTED_V9_EVIDENCE_BINDING_SHA256_V10:
        raise ValueError("v10 v9 evidence binding changed")
    return binding


def frozen_recipe_v10(evidence):
    return {
        "model_name": str((ROOT / "models/Qwen3.6-35B-A3B").resolve()),
        "checkpoint": None,
        "train_dataset": str(FROZEN_TRAIN_DATASET_V10),
        "eval_dataset": str(FROZEN_EVAL_DATASET_V10),
        "sigma": 0.0003, "population_size": 32,
        "base_direction_count": 32, "unique_signed_direction_count": 64,
        "actual_perturb_restore_cycle_count": 128,
        "domain_signed_score_count": 128,
        "anchor_signed_response_count": 128,
        "batch_size": 128, "domain_manifest_rows": [64, 64],
        "mini_batch_size": 64, "max_tokens": 32, "seed": 43,
        "engine_count": 4, "tp_per_engine": 1, "gpu_ids": [0, 1, 2, 3],
        "eval_splits": ["validation", "ood_qa"], "target_alphas": [0.0],
        "anchor_prose_jsonl": str(
            (ROOT / "data/general_prose_anchor_v1.jsonl").resolve()
        ),
        "anchor_prose_report": str(
            (ROOT / "data/general_prose_anchor_v1.report.json").resolve()
        ),
        "anchor_items_per_step": 128, "anchor_max_input_tokens": 512,
        "min_anchor_cosine": 0.8, "anchor_generation_seeds": [43, 44],
        "ood_prose_jsonl": str((ROOT / "data/ood_prose_v3.jsonl").resolve()),
        "ood_prose_max_input_tokens": 1024,
        "document_lcb_config_sha256": anchor_v10.anchor_v5.DOCUMENT_LCB_CONFIG_SHA256_V5,
        "reward_function_timeout": 10,
        "output_directory": str(FROZEN_OUTPUT_DIRECTORY_V10),
        "experiment_name": EXPERIMENT_NAME_V10,
        "logging": "none", "wandb_project": "specialist-eggroll-es",
        "perturbation_basis_seed": PERTURBATION_BASIS_SEED_V10,
        "perturbation_basis_sha256": PERTURBATION_BASIS_SHA256_V10,
        "domain_manifests": copy.deepcopy(anchor_v10.DOMAIN_MANIFESTS_V10),
        "combined_domain_manifest_sha256": (
            anchor_v10.COMBINED_DOMAIN_MANIFEST_SHA256_V10
        ),
        "v9_determinism_evidence": copy.deepcopy(evidence),
    }


def _crossed_train_loader_v10(dataset, batch_size, seed):
    if int(batch_size) != 128 or int(seed) != 43:
        raise ValueError("v10 crossed loader requires combined batch128 seed43")
    combined_questions = []
    combined_answers = []
    for label in ("D43", "D44"):
        spec = anchor_v10.DOMAIN_MANIFESTS_V10[label]
        questions, answers = next(iter(_ORIGINAL_BUILD_TRAIN_LOADER(
            dataset, 64, spec["seed"],
        )))
        identity = driver_v1.canonical_sha256({
            "questions": list(questions), "answers": list(answers),
        })
        if len(questions) != 64 or identity != spec["sha256"]:
            raise RuntimeError(f"v10 frozen {label} manifest changed")
        combined_questions.extend(questions)
        combined_answers.extend(answers)
    combined_identity = driver_v1.canonical_sha256({
        "questions": combined_questions, "answers": combined_answers,
    })
    if combined_identity != anchor_v10.COMBINED_DOMAIN_MANIFEST_SHA256_V10:
        raise RuntimeError("v10 combined crossed manifest changed")
    return [(combined_questions, combined_answers)]


def validate_frozen_execution_cli_v10(argv, bundle):
    inherited = driver_v5.validate_frozen_execution_cli_v5(argv)
    stage_parser = argparse.ArgumentParser(add_help=False)
    stage_parser.add_argument("--v10-stage", choices=("variance",))
    stage_parser.add_argument("--v10-v9-report")
    stage_parser.add_argument("--v10-perturbation-basis-seed", type=int)
    stage_parser.add_argument("--v10-dry-run", action="store_true")
    stage, remaining = stage_parser.parse_known_args(list(argv))
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--population-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--target-alphas")
    parser.add_argument("--train-dataset", default=str(FROZEN_TRAIN_DATASET_V10))
    parser.add_argument("--eval-dataset", default=str(FROZEN_EVAL_DATASET_V10))
    parser.add_argument("--reward-function-timeout", type=int, default=10)
    parser.add_argument("--output-directory", default=str(FROZEN_OUTPUT_DIRECTORY_V10))
    parser.add_argument("--experiment-name")
    parser.add_argument("--logging", default="none")
    parser.add_argument("--wandb-project", default="specialist-eggroll-es")
    recipe, _ = parser.parse_known_args(list(argv))
    metadata = anchor_v10.validate_frozen_layer_plan_bundle_v10(bundle)
    if stage.v10_stage != "variance":
        raise ValueError("v10 requires --v10-stage variance")
    if recipe.seed != 43 or recipe.population_size != 32 or recipe.batch_size != 128:
        raise ValueError("v10 runtime requires seed43/pop32/combined-batch128")
    if driver_v1.parse_target_alphas(recipe.target_alphas) != [0.0]:
        raise ValueError("v10 target must be exactly alpha zero")
    if stage.v10_perturbation_basis_seed != PERTURBATION_BASIS_SEED_V10:
        raise ValueError("v10 perturbation basis seed changed")
    if (
        metadata["plan"] != "middle_late"
        or bundle["plan_sha256"] != MIDDLE_LATE_PLAN_SHA256_V10
        or Path(recipe.train_dataset).resolve() != FROZEN_TRAIN_DATASET_V10
        or Path(recipe.eval_dataset).resolve() != FROZEN_EVAL_DATASET_V10
        or recipe.reward_function_timeout != 10
        or Path(recipe.output_directory).resolve() != FROZEN_OUTPUT_DIRECTORY_V10
        or recipe.experiment_name != EXPERIMENT_NAME_V10
        or recipe.logging != "none"
        or recipe.wandb_project != "specialist-eggroll-es"
    ):
        raise ValueError("v10 antithetic-crossed runtime recipe changed")
    if stage.v10_v9_report is None:
        raise ValueError("v10 requires the passing v9 report")
    evidence = _v9_determinism_evidence_v10(stage.v10_v9_report)
    frozen = frozen_recipe_v10(evidence)
    if driver_v1.canonical_sha256(frozen) != EXPECTED_RECIPE_SHA256_V10:
        raise RuntimeError("v10 frozen recipe hash changed")
    execution = {
        **inherited, **copy.deepcopy(frozen),
        "schema": "eggroll-es-antithetic-crossed-execution-v10",
        "stage": "variance", "arm": "middle_late",
        "dry_run": stage.v10_dry_run,
    }
    if {key: execution[key] for key in V10_RECIPE_KEYS} != frozen:
        raise RuntimeError("v10 effective recipe changed")
    return execution, remaining


def set_active_v10(bundle, execution):
    global _ACTIVE_LAYER_PLAN_BUNDLE, _ACTIVE_EXECUTION_V10
    driver_v5.set_active_layer_plan_bundle_v5(bundle)
    anchor_v10.set_default_layer_plan_bundle_v10(bundle)
    _ACTIVE_LAYER_PLAN_BUNDLE = bundle
    _ACTIVE_EXECUTION_V10 = execution


def build_snapshot(*args, **kwargs):
    bundle = _ACTIVE_LAYER_PLAN_BUNDLE
    execution = _ACTIVE_EXECUTION_V10
    snapshot = driver_v5.build_snapshot(*args, **kwargs)
    if snapshot.get("fixed_train_batch") != {
        "rows": 128,
        "sha256": anchor_v10.COMBINED_DOMAIN_MANIFEST_SHA256_V10,
    }:
        raise RuntimeError("v10 combined fixed train batch changed")
    implementation = {
        key: anchor_v10.file_sha256(path)
        for key, path in V10_IMPLEMENTATION_PATHS.items()
    }
    snapshot["implementation"].update(implementation)
    snapshot["schema"] = "eggroll-es-anchor-line-search-snapshot-v10"
    snapshot["antithetic_crossed_v10"] = {
        "schema": "eggroll-es-antithetic-crossed-snapshot-v10",
        "family": "D43_D44_x_A43_A44", "stage": "variance",
        "arm": "middle_late", "layers": [20, 21, 22, 23],
        "data_seed": 43, "base_direction_count": 32,
        "unique_signed_direction_count": 64,
        "actual_perturb_restore_cycle_count": 128,
        "domain_signed_score_count": 128,
        "anchor_signed_response_count": 128,
        "sign_order": ["plus", "minus"],
        "domain_manifests": copy.deepcopy(anchor_v10.DOMAIN_MANIFESTS_V10),
        "combined_domain_manifest_sha256": (
            anchor_v10.COMBINED_DOMAIN_MANIFEST_SHA256_V10
        ),
        "anchor_generation_seeds": [43, 44],
        "perturbation_basis_seed": PERTURBATION_BASIS_SEED_V10,
        "perturbation_seeds": list(PERTURBATION_SEEDS_V10),
        "perturbation_basis_sha256": PERTURBATION_BASIS_SHA256_V10,
        "target_alphas": [0.0], "benchmark_treatment_applied": False,
        "selection_surface": "crossed_train_anchor_response_only",
        "v9_determinism_evidence": copy.deepcopy(
            execution["v9_determinism_evidence"]
        ),
        "recipe": {
            key: copy.deepcopy(execution[key]) for key in V10_RECIPE_KEYS
        },
        "plan_sha256": bundle["plan_sha256"],
        "plan_file_sha256": bundle["file_sha256"],
        "implementation_bundle_sha256": driver_v1.canonical_sha256(
            implementation
        ),
    }
    return snapshot


def _inherited_compatibility_v10(journal):
    compatible = copy.deepcopy(journal)
    compatible["schema"] = "eggroll-es-anchor-alpha-line-search-v8"
    compatible["snapshot"]["schema"] = (
        "eggroll-es-anchor-line-search-snapshot-v8"
    )
    compatible["snapshot"].pop("antithetic_crossed_v10", None)
    for key in V10_POLICY:
        compatible["policy"].pop(key, None)
    compatible["policy"].update({
        "split_seed_stability_family_v8": "middle_late_pop32_same_basis",
        "stage_v8": "stability", "target_alpha_zero_only_v8": True,
        "benchmark_selection_forbidden_v8": True,
        "same_perturbation_basis_required_v8": True,
        "cross_data_seed_coefficient_cosine_threshold_v8": 0.5,
        "requires_complete_v7_family_v8": True,
    })
    compatible["snapshot"]["split_seed_stability_v8"] = {}
    compatible.pop("content_sha256_before_self_field", None)
    compatible["content_sha256_before_self_field"] = (
        driver_v1.canonical_sha256(compatible)
    )
    return compatible


def validate_completed_journal_v10(journal):
    if (
        not isinstance(journal, dict)
        or journal.get("schema") != "eggroll-es-anchor-alpha-line-search-v10"
        or journal.get("status") != "complete"
        or journal.get("in_progress") is not None
        or journal.get("targets") != [0.0]
        or journal.get("seeds") != PERTURBATION_SEEDS_V10
    ):
        raise RuntimeError("v10 journal is incomplete or changed basis/target")
    if journal.get("content_sha256_before_self_field") != (
        driver_v1.canonical_sha256({
            key: value for key, value in journal.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v10 journal content hash changed")
    if journal.get("policy") != {**INHERITED_POLICY_V10, **V10_POLICY}:
        raise RuntimeError("v10 antithetic-crossed policy changed")
    snapshot = journal.get("snapshot", {})
    metadata = snapshot.get("antithetic_crossed_v10")
    evidence = _v9_determinism_evidence_v10(V9_REPORT_PATH_V10)
    expected_recipe = frozen_recipe_v10(evidence)
    if (
        snapshot.get("schema") != "eggroll-es-anchor-line-search-snapshot-v10"
        or not isinstance(metadata, dict)
        or metadata.get("schema") != "eggroll-es-antithetic-crossed-snapshot-v10"
        or metadata.get("family") != "D43_D44_x_A43_A44"
        or metadata.get("stage") != "variance"
        or metadata.get("data_seed") != 43
        or metadata.get("base_direction_count") != 32
        or metadata.get("unique_signed_direction_count") != 64
        or metadata.get("actual_perturb_restore_cycle_count") != 128
        or metadata.get("domain_signed_score_count") != 128
        or metadata.get("anchor_signed_response_count") != 128
        or metadata.get("sign_order") != ["plus", "minus"]
        or metadata.get("domain_manifests") != anchor_v10.DOMAIN_MANIFESTS_V10
        or metadata.get("combined_domain_manifest_sha256")
        != anchor_v10.COMBINED_DOMAIN_MANIFEST_SHA256_V10
        or metadata.get("anchor_generation_seeds") != [43, 44]
        or metadata.get("perturbation_seeds") != PERTURBATION_SEEDS_V10
        or metadata.get("perturbation_basis_sha256")
        != PERTURBATION_BASIS_SHA256_V10
        or metadata.get("target_alphas") != [0.0]
        or metadata.get("benchmark_treatment_applied") is not False
        or metadata.get("v9_determinism_evidence") != evidence
        or metadata.get("recipe") != expected_recipe
        or driver_v1.canonical_sha256(expected_recipe)
        != EXPECTED_RECIPE_SHA256_V10
    ):
        raise RuntimeError("v10 antithetic-crossed snapshot changed")
    if snapshot.get("fixed_train_batch") != {
        "rows": 128,
        "sha256": anchor_v10.COMBINED_DOMAIN_MANIFEST_SHA256_V10,
    }:
        raise RuntimeError("v10 fixed crossed batch identity changed")
    actual_implementation = {
        key: anchor_v10.file_sha256(path)
        for key, path in V10_IMPLEMENTATION_PATHS.items()
    }
    if (
        snapshot.get("implementation") != actual_implementation
        or metadata.get("implementation_bundle_sha256")
        != driver_v1.canonical_sha256(actual_implementation)
    ):
        raise RuntimeError("v10 implementation identity changed")
    plan = driver_v5._journal_plan_v5(
        driver_v8._v5_compatibility_journal_v8(
            _inherited_compatibility_v10(journal)
        )
    )
    plan["antithetic_cross_v10"] = journal["coefficient_plan"].get(
        "antithetic_cross_v10"
    )
    cross = anchor_v10.validate_antithetic_cross_v10(
        plan, recompute_numeric=True,
    )
    with driver_v6.scoped_legacy_audit_v6():
        inherited = driver_v8._validate_inherited_zero_target_v8(
            _inherited_compatibility_v10(journal)
        )
    if inherited["data_bootstrap_seed"] != 43:
        raise RuntimeError("v10 inherited execution changed seed")
    return {
        **inherited, "stage": "variance", "data_seed": 43,
        "cross": cross,
        "content_sha256": journal["content_sha256_before_self_field"],
    }


def execute_line_search(*args, **kwargs):
    kwargs = dict(kwargs)
    expected = np.random.default_rng(seed=43).integers(
        0, 2**30, size=32, dtype=np.int64,
    ).tolist()
    if kwargs.get("seeds") != expected:
        raise RuntimeError("v10 inherited population schedule changed")
    kwargs["seeds"] = list(PERTURBATION_SEEDS_V10)
    with driver_v6.scoped_legacy_audit_v6():
        journal = driver_v4.execute_line_search(*args, **kwargs)
    trainer = args[0] if args else kwargs["trainer"]
    plan = trainer._latest_anchor_plan
    anchor_v10.validate_antithetic_cross_v10(plan, recompute_numeric=True)
    anchor_v10.anchor_v8.validate_robust_plan_v8(plan, recompute_numeric=True)
    binding_v5 = anchor_v10.anchor_v5.validate_robust_plan_v5(
        plan, recompute_numeric=True,
    )
    journal["schema"] = "eggroll-es-anchor-alpha-line-search-v10"
    journal["snapshot"]["schema"] = "eggroll-es-anchor-line-search-snapshot-v10"
    journal["policy"].update({
        "document_lcb_anchor_required": True,
        "optimization_data": "train_and_anchor_only",
        "ood_validation_heldout_as_objective": False,
        **V10_POLICY,
    })
    coefficient_plan = journal["coefficient_plan"]
    coefficient_plan["domain_scores_v5"] = list(plan["domain_scores"])
    coefficient_plan["anchor_scores_v5"] = list(plan["anchor_scores"])
    coefficient_plan["document_lcb_anchor_v5"] = plan["document_lcb_anchor_v5"]
    coefficient_plan["robust_plan_binding_v5"] = binding_v5
    coefficient_plan["antithetic_cross_v10"] = copy.deepcopy(
        plan["antithetic_cross_v10"]
    )
    journal.pop("content_sha256_before_self_field", None)
    journal["content_sha256_before_self_field"] = driver_v1.canonical_sha256(journal)
    try:
        validate_completed_journal_v10(journal)
    except Exception as error:
        journal["status"] = "failed"
        journal["failure"] = {
            "type": type(error).__name__, "message": str(error),
            "phase": "validating_complete_v10_release_audit",
        }
        journal.pop("content_sha256_before_self_field", None)
        driver_v1.atomic_write_json(kwargs["journal_path"], journal)
        raise
    driver_v1.atomic_write_json(kwargs["journal_path"], journal)
    return journal


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    bundle, remaining = anchor_v10.parse_frozen_layer_plan_cli_v10(argv)
    execution, remaining = validate_frozen_execution_cli_v10(remaining, bundle)
    set_active_v10(bundle, execution)
    if execution["dry_run"]:
        payload = {
            "schema": "eggroll-es-antithetic-crossed-dry-run-v10",
            "data_seed": 43, "base_direction_count": 32,
            "unique_signed_direction_count": 64,
            "actual_perturb_restore_cycle_count": 128,
            "domain_signed_score_count": 128,
            "anchor_signed_response_count": 128,
            "domain_manifests": anchor_v10.DOMAIN_MANIFESTS_V10,
            "anchor_generation_seeds": [43, 44], "targets": [0.0],
            "recipe_sha256": driver_v1.canonical_sha256({
                key: execution[key] for key in V10_RECIPE_KEYS
            }),
        }
        print(json.dumps(payload, sort_keys=True))
        return payload
    old_argv = sys.argv
    old_anchor = driver_v1.anchor
    old_build = driver_v1.build_snapshot
    old_execute = driver_v1.execute_line_search
    old_loader = driver_v1.base.build_train_loader
    sys.argv = [old_argv[0], *remaining]
    driver_v1.anchor = anchor_v10
    driver_v1.build_snapshot = build_snapshot
    driver_v1.execute_line_search = execute_line_search
    driver_v1.base.build_train_loader = _crossed_train_loader_v10
    try:
        driver_v1.main()
    finally:
        sys.argv = old_argv
        driver_v1.anchor = old_anchor
        driver_v1.build_snapshot = old_build
        driver_v1.execute_line_search = old_execute
        driver_v1.base.build_train_loader = old_loader


if __name__ == "__main__":
    main()
