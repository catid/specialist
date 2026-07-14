#!/usr/bin/env python3
"""Fail-closed resident-sign exact-equivalence diagnostic v11."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import report_eggroll_es_variance_v10 as reporter_v10
import run_eggroll_es_anchor_line_search as driver_v1
import run_eggroll_es_anchor_line_search_v4 as driver_v4
import run_eggroll_es_anchor_line_search_v5 as driver_v5
import run_eggroll_es_anchor_line_search_v6 as driver_v6
import run_eggroll_es_anchor_stability_v8 as driver_v8
import run_eggroll_es_anchor_variance_v10 as driver_v10
import train_eggroll_es_specialist_anchor_v11 as anchor_v11


ROOT = Path(__file__).resolve().parent
DATA_SEED_V11 = 43
POPULATION_SIZE_V11 = 32
PERTURBATION_BASIS_SEED_V11 = driver_v10.PERTURBATION_BASIS_SEED_V10
PERTURBATION_SEEDS_V11 = list(driver_v10.PERTURBATION_SEEDS_V10)
PERTURBATION_BASIS_SHA256_V11 = driver_v10.PERTURBATION_BASIS_SHA256_V10
MIDDLE_LATE_PLAN_SHA256_V11 = driver_v10.MIDDLE_LATE_PLAN_SHA256_V10
FROZEN_TRAIN_DATASET_V11 = driver_v10.FROZEN_TRAIN_DATASET_V10
FROZEN_EVAL_DATASET_V11 = driver_v10.FROZEN_EVAL_DATASET_V10
FROZEN_OUTPUT_DIRECTORY_V11 = driver_v10.FROZEN_OUTPUT_DIRECTORY_V10
EXPERIMENT_NAME_V11 = (
    "snapshot794_layer_v11_middle_late_resident_sign_exact_v10_"
    "d43d44_a43a44_basis20260714"
)
V10_REPORT_PATH_V11 = (
    ROOT / "experiments/eggroll_es_hpo/S6_ANTITHETIC_CROSSED_V10_REPORT.json"
).resolve()
V10_REPORT_FILE_SHA256_V11 = (
    "a1a4528b98cee7c15323e654fe3dd6b57422c3dfba2f7ef23d95f10dee242a7f"
)
V10_REPORT_CONTENT_SHA256_V11 = (
    "1cbb920284e096e4ce744844aa098235f9c17dd85ac1e10a9bba19ae6db212a7"
)
V10_JOURNAL_PATH_V11 = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "snapshot794_layer_v10_middle_late_antithetic_cross_"
    "d43d44_a43a44_basis20260714/alpha_line_search.json"
).resolve()
V10_JOURNAL_FILE_SHA256_V11 = (
    "2708b563034367479da9b25f3fcd8bd556b0c2133f533b3b561fcfd46d9af5ee"
)
V10_JOURNAL_CONTENT_SHA256_V11 = (
    "3e68b1fb925378e31c9c4945de82d33c34f77c6abad585d0415ec456e78d71c7"
)
EXPECTED_V10_EVIDENCE_BINDING_SHA256_V11 = (
    "9fe976b369b8edb32b63018a5f270bb8aa033570fb704c930ce93082fb2207c5"
)
EXPECTED_RECIPE_SHA256_V11 = (
    "0e69dd27193a1f5c2eea9499d87b7370e74e8c33c84bd6f421d30b05bb0af4f7"
)
V11_RECIPE_KEYS = {
    "model_name", "checkpoint", "train_dataset", "eval_dataset", "sigma",
    "population_size", "base_direction_count", "unique_signed_direction_count",
    "actual_perturb_restore_cycle_count", "all_engine_sign_residency_count",
    "domain_signed_score_count", "anchor_signed_response_count", "batch_size",
    "domain_manifest_rows", "mini_batch_size", "max_tokens", "seed",
    "engine_count", "tp_per_engine", "gpu_ids", "eval_splits", "target_alphas",
    "anchor_prose_jsonl", "anchor_prose_report", "anchor_items_per_step",
    "anchor_max_input_tokens", "min_anchor_cosine", "anchor_generation_seeds",
    "ood_prose_jsonl", "ood_prose_max_input_tokens",
    "document_lcb_config_sha256", "reward_function_timeout", "output_directory",
    "experiment_name", "logging", "wandb_project", "perturbation_basis_seed",
    "perturbation_basis_sha256", "domain_manifests",
    "combined_domain_manifest_sha256", "resident_generation_order",
    "v9_determinism_evidence", "v10_equivalence_evidence",
}
V11_IMPLEMENTATION_PATHS = {
    **driver_v10.V10_IMPLEMENTATION_PATHS,
    "anchor_trainer": Path(anchor_v11.__file__).resolve(),
    "distributed_driver_v11": Path(__file__).resolve(),
    "distributed_trainer_v11": Path(anchor_v11.__file__).resolve(),
    "distributed_worker_v11": ROOT / "eggroll_es_worker_v11.py",
    "equivalence_reporter_v11": ROOT / "report_eggroll_es_equivalence_v11.py",
    "protocol_v11": (
        ROOT / "experiments/eggroll_es_hpo/"
        "S6_RESIDENT_SIGN_EQUIVALENCE_V11_PROTOCOL.md"
    ),
    "contract_tests_v11": ROOT / "test_eggroll_es_equivalence_v11.py",
    "v10_variance_report": V10_REPORT_PATH_V11,
    "v10_variance_journal": V10_JOURNAL_PATH_V11,
}
V11_POLICY = {
    "resident_sign_equivalence_family_v11": "D43_A43_A44_D44_per_sign",
    "stage_v11": "equivalence",
    "target_alpha_zero_only_v11": True,
    "benchmark_selection_forbidden_v11": True,
    "requires_exact_v10_response_equivalence_v11": True,
    "second_parent_call_engine_dispatch_forbidden_v11": True,
    "requires_complete_v10_pass_v11": True,
}
INHERITED_POLICY_V11 = copy.deepcopy(driver_v10.INHERITED_POLICY_V10)
_ACTIVE_LAYER_PLAN_BUNDLE = None
_ACTIVE_EXECUTION_V11 = None


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_v10_journal_v11():
    path = V10_JOURNAL_PATH_V11
    driver_v8.offline_audit._assert_no_heldout(str(path), "v11 v10 journal")
    if _file_sha256(path) != V10_JOURNAL_FILE_SHA256_V11:
        raise ValueError("v11 exact v10 journal file changed")
    journal = json.loads(path.read_text())
    if journal.get("content_sha256_before_self_field") != V10_JOURNAL_CONTENT_SHA256_V11:
        raise ValueError("v11 exact v10 journal content changed")
    driver_v10.validate_completed_journal_v10(journal)
    return journal


def _v10_equivalence_evidence_v11(path):
    path = Path(path).resolve()
    driver_v8.offline_audit._assert_no_heldout(str(path), "v11 v10 report")
    if path != V10_REPORT_PATH_V11 or _file_sha256(path) != V10_REPORT_FILE_SHA256_V11:
        raise ValueError("v11 requires the exact completed v10 report")
    report = json.loads(path.read_text())
    if (
        report.get("content_sha256_before_self_field")
        != V10_REPORT_CONTENT_SHA256_V11
        or report.get("content_sha256_before_self_field")
        != driver_v1.canonical_sha256({
            key: value for key, value in report.items()
            if key != "content_sha256_before_self_field"
        })
        or report.get("passed") is not True
        or report.get("journal") != str(V10_JOURNAL_PATH_V11)
        or report.get("journal_file_sha256") != V10_JOURNAL_FILE_SHA256_V11
        or report.get("journal_content_sha256") != V10_JOURNAL_CONTENT_SHA256_V11
        or report.get("cross_artifact_content_sha256")
        != anchor_v11.V10_EQUIVALENCE_TARGET_V11[
            "cross_artifact_content_sha256"
        ]
        or report.get("cell_coefficient_sha256")
        != anchor_v11.V10_EQUIVALENCE_TARGET_V11[
            "cell_coefficient_sha256"
        ]
    ):
        raise ValueError("v11 v10 report identity or pass changed")
    rebuilt = reporter_v10.build_report(V10_JOURNAL_PATH_V11)
    if rebuilt != report:
        raise ValueError("v11 v10 report differs from its exact journal")
    _load_v10_journal_v11()
    binding = {
        "schema": "eggroll-es-v10-equivalence-evidence-binding-v11",
        "report_path": str(path),
        "report_file_sha256": V10_REPORT_FILE_SHA256_V11,
        "report_content_sha256": V10_REPORT_CONTENT_SHA256_V11,
        "journal_path": str(V10_JOURNAL_PATH_V11),
        "journal_file_sha256": V10_JOURNAL_FILE_SHA256_V11,
        "journal_content_sha256": V10_JOURNAL_CONTENT_SHA256_V11,
        "cross_artifact_content_sha256": report[
            "cross_artifact_content_sha256"
        ],
        "cell_coefficient_sha256": copy.deepcopy(
            report["cell_coefficient_sha256"]
        ),
        "minimum_pairwise_coefficient_cosine": report[
            "minimum_pairwise_coefficient_cosine"
        ],
        "median_pairwise_coefficient_cosine": report[
            "median_pairwise_coefficient_cosine"
        ],
        "passed": True,
    }
    binding["binding_sha256"] = driver_v1.canonical_sha256(binding)
    if binding["binding_sha256"] != EXPECTED_V10_EVIDENCE_BINDING_SHA256_V11:
        raise ValueError("v11 v10 evidence binding changed")
    return binding


def frozen_recipe_v11(evidence):
    v9 = driver_v10._v9_determinism_evidence_v10(
        driver_v10.V9_REPORT_PATH_V10
    )
    recipe = driver_v10.frozen_recipe_v10(v9)
    recipe.update({
        "actual_perturb_restore_cycle_count": 64,
        "all_engine_sign_residency_count": 16,
        "experiment_name": EXPERIMENT_NAME_V11,
        "resident_generation_order": ["D43", "A43", "A44", "D44"],
        "v10_equivalence_evidence": copy.deepcopy(evidence),
    })
    return recipe


def validate_frozen_execution_cli_v11(argv, bundle):
    inherited = driver_v5.validate_frozen_execution_cli_v5(argv)
    stage_parser = argparse.ArgumentParser(add_help=False)
    stage_parser.add_argument("--v11-stage", choices=("equivalence",))
    stage_parser.add_argument("--v11-v10-report")
    stage_parser.add_argument("--v11-perturbation-basis-seed", type=int)
    stage_parser.add_argument("--v11-dry-run", action="store_true")
    stage, remaining = stage_parser.parse_known_args(list(argv))
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--population-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--target-alphas")
    parser.add_argument("--train-dataset", default=str(FROZEN_TRAIN_DATASET_V11))
    parser.add_argument("--eval-dataset", default=str(FROZEN_EVAL_DATASET_V11))
    parser.add_argument("--reward-function-timeout", type=int, default=10)
    parser.add_argument("--output-directory", default=str(FROZEN_OUTPUT_DIRECTORY_V11))
    parser.add_argument("--experiment-name")
    parser.add_argument("--logging", default="none")
    parser.add_argument("--wandb-project", default="specialist-eggroll-es")
    recipe, _ = parser.parse_known_args(list(argv))
    metadata = anchor_v11.validate_frozen_layer_plan_bundle_v11(bundle)
    if stage.v11_stage != "equivalence":
        raise ValueError("v11 requires --v11-stage equivalence")
    if recipe.seed != 43 or recipe.population_size != 32 or recipe.batch_size != 128:
        raise ValueError("v11 runtime requires seed43/pop32/combined-batch128")
    if driver_v1.parse_target_alphas(recipe.target_alphas) != [0.0]:
        raise ValueError("v11 target must be exactly alpha zero")
    if stage.v11_perturbation_basis_seed != PERTURBATION_BASIS_SEED_V11:
        raise ValueError("v11 perturbation basis seed changed")
    if (
        metadata["plan"] != "middle_late"
        or bundle["plan_sha256"] != MIDDLE_LATE_PLAN_SHA256_V11
        or Path(recipe.train_dataset).resolve() != FROZEN_TRAIN_DATASET_V11
        or Path(recipe.eval_dataset).resolve() != FROZEN_EVAL_DATASET_V11
        or recipe.reward_function_timeout != 10
        or Path(recipe.output_directory).resolve() != FROZEN_OUTPUT_DIRECTORY_V11
        or recipe.experiment_name != EXPERIMENT_NAME_V11
        or recipe.logging != "none"
        or recipe.wandb_project != "specialist-eggroll-es"
    ):
        raise ValueError("v11 resident-sign runtime recipe changed")
    if stage.v11_v10_report is None:
        raise ValueError("v11 requires the passing v10 report")
    evidence = _v10_equivalence_evidence_v11(stage.v11_v10_report)
    frozen = frozen_recipe_v11(evidence)
    if set(frozen) != V11_RECIPE_KEYS:
        raise RuntimeError("v11 frozen recipe key coverage changed")
    if driver_v1.canonical_sha256(frozen) != EXPECTED_RECIPE_SHA256_V11:
        raise RuntimeError("v11 frozen recipe hash changed")
    execution = {
        **inherited, **copy.deepcopy(frozen),
        "schema": "eggroll-es-resident-sign-execution-v11",
        "stage": "equivalence", "arm": "middle_late",
        "dry_run": stage.v11_dry_run,
    }
    if {key: execution[key] for key in V11_RECIPE_KEYS} != frozen:
        raise RuntimeError("v11 effective recipe changed")
    return execution, remaining


def set_active_v11(bundle, execution):
    global _ACTIVE_LAYER_PLAN_BUNDLE, _ACTIVE_EXECUTION_V11
    driver_v5.set_active_layer_plan_bundle_v5(bundle)
    anchor_v11.set_default_layer_plan_bundle_v11(bundle)
    _ACTIVE_LAYER_PLAN_BUNDLE = bundle
    _ACTIVE_EXECUTION_V11 = execution


def build_snapshot(*args, **kwargs):
    bundle = _ACTIVE_LAYER_PLAN_BUNDLE
    execution = _ACTIVE_EXECUTION_V11
    snapshot = driver_v5.build_snapshot(*args, **kwargs)
    if snapshot.get("fixed_train_batch") != {
        "rows": 128, "sha256": anchor_v11.COMBINED_DOMAIN_MANIFEST_SHA256_V11,
    }:
        raise RuntimeError("v11 combined fixed train batch changed")
    implementation = {
        key: anchor_v11.file_sha256(path)
        for key, path in V11_IMPLEMENTATION_PATHS.items()
    }
    snapshot["implementation"].update(implementation)
    snapshot["schema"] = "eggroll-es-anchor-line-search-snapshot-v11"
    snapshot["resident_sign_equivalence_v11"] = {
        "schema": "eggroll-es-resident-sign-snapshot-v11",
        "family": "D43_A43_A44_D44_per_sign", "stage": "equivalence",
        "arm": "middle_late", "layers": [20, 21, 22, 23],
        "data_seed": 43, "base_direction_count": 32,
        "unique_signed_direction_count": 64,
        "actual_perturb_restore_cycle_count": 64,
        "all_engine_sign_residency_count": 16,
        "domain_signed_score_count": 128,
        "anchor_signed_response_count": 128,
        "resident_generation_order": ["D43", "A43", "A44", "D44"],
        "second_parent_call": "validated_one_use_D44_cache_no_engine_dispatch",
        "domain_manifests": copy.deepcopy(anchor_v11.DOMAIN_MANIFESTS_V11),
        "anchor_generation_seeds": [43, 44],
        "perturbation_basis_seed": PERTURBATION_BASIS_SEED_V11,
        "perturbation_seeds": list(PERTURBATION_SEEDS_V11),
        "perturbation_basis_sha256": PERTURBATION_BASIS_SHA256_V11,
        "target_alphas": [0.0], "benchmark_treatment_applied": False,
        "v10_equivalence_evidence": copy.deepcopy(
            execution["v10_equivalence_evidence"]
        ),
        "recipe": {
            key: copy.deepcopy(execution[key]) for key in V11_RECIPE_KEYS
        },
        "plan_sha256": bundle["plan_sha256"],
        "plan_file_sha256": bundle["file_sha256"],
        "implementation_bundle_sha256": driver_v1.canonical_sha256(
            implementation
        ),
    }
    return snapshot


def _inherited_compatibility_v11(journal):
    compatible = copy.deepcopy(journal)
    compatible["schema"] = "eggroll-es-anchor-alpha-line-search-v8"
    compatible["snapshot"]["schema"] = "eggroll-es-anchor-line-search-snapshot-v8"
    compatible["snapshot"].pop("resident_sign_equivalence_v11", None)
    for key in V11_POLICY:
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
    compatible["content_sha256_before_self_field"] = driver_v1.canonical_sha256(
        compatible
    )
    return compatible


def _journal_plan_v11(journal):
    plan = driver_v5._journal_plan_v5(
        driver_v8._v5_compatibility_journal_v8(
            _inherited_compatibility_v11(journal)
        )
    )
    plan["resident_sign_cross_v11"] = journal["coefficient_plan"].get(
        "resident_sign_cross_v11"
    )
    return plan


def _v10_plan_v11(journal):
    plan = driver_v5._journal_plan_v5(
        driver_v8._v5_compatibility_journal_v8(
            driver_v10._inherited_compatibility_v10(journal)
        )
    )
    plan["antithetic_cross_v10"] = journal["coefficient_plan"].get(
        "antithetic_cross_v10"
    )
    return plan


def validate_completed_journal_v11(journal):
    if (
        not isinstance(journal, dict)
        or journal.get("schema") != "eggroll-es-anchor-alpha-line-search-v11"
        or journal.get("status") != "complete"
        or journal.get("in_progress") is not None
        or journal.get("targets") != [0.0]
        or journal.get("seeds") != PERTURBATION_SEEDS_V11
        or journal.get("content_sha256_before_self_field")
        != driver_v1.canonical_sha256({
            key: value for key, value in journal.items()
            if key != "content_sha256_before_self_field"
        })
        or journal.get("policy") != {**INHERITED_POLICY_V11, **V11_POLICY}
    ):
        raise RuntimeError("v11 journal identity, policy, or completion changed")
    snapshot = journal.get("snapshot", {})
    metadata = snapshot.get("resident_sign_equivalence_v11", {})
    evidence = _v10_equivalence_evidence_v11(V10_REPORT_PATH_V11)
    expected_recipe = frozen_recipe_v11(evidence)
    if (
        snapshot.get("schema") != "eggroll-es-anchor-line-search-snapshot-v11"
        or metadata.get("schema") != "eggroll-es-resident-sign-snapshot-v11"
        or metadata.get("family") != "D43_A43_A44_D44_per_sign"
        or metadata.get("stage") != "equivalence"
        or metadata.get("base_direction_count") != 32
        or metadata.get("unique_signed_direction_count") != 64
        or metadata.get("actual_perturb_restore_cycle_count") != 64
        or metadata.get("all_engine_sign_residency_count") != 16
        or metadata.get("domain_signed_score_count") != 128
        or metadata.get("anchor_signed_response_count") != 128
        or metadata.get("resident_generation_order")
        != ["D43", "A43", "A44", "D44"]
        or metadata.get("second_parent_call")
        != "validated_one_use_D44_cache_no_engine_dispatch"
        or metadata.get("domain_manifests") != anchor_v11.DOMAIN_MANIFESTS_V11
        or metadata.get("perturbation_seeds") != PERTURBATION_SEEDS_V11
        or metadata.get("perturbation_basis_sha256")
        != PERTURBATION_BASIS_SHA256_V11
        or metadata.get("target_alphas") != [0.0]
        or metadata.get("benchmark_treatment_applied") is not False
        or metadata.get("v10_equivalence_evidence") != evidence
        or metadata.get("recipe") != expected_recipe
        or driver_v1.canonical_sha256(expected_recipe) != EXPECTED_RECIPE_SHA256_V11
    ):
        raise RuntimeError("v11 resident-sign snapshot changed")
    implementation = {
        key: anchor_v11.file_sha256(path)
        for key, path in V11_IMPLEMENTATION_PATHS.items()
    }
    if (
        snapshot.get("implementation") != implementation
        or metadata.get("implementation_bundle_sha256")
        != driver_v1.canonical_sha256(implementation)
    ):
        raise RuntimeError("v11 implementation identity changed")
    plan = _journal_plan_v11(journal)
    resident = anchor_v11.validate_resident_cross_v11(
        plan, recompute_numeric=True,
    )
    v10_journal = _load_v10_journal_v11()
    equivalence = anchor_v11.compare_exact_v10_v11(
        _v10_plan_v11(v10_journal), plan,
    )
    with driver_v6.scoped_legacy_audit_v6():
        inherited = driver_v8._validate_inherited_zero_target_v8(
            _inherited_compatibility_v11(journal)
        )
    return {
        **inherited, "stage": "equivalence", "data_seed": 43,
        "resident": resident, "equivalence": equivalence,
        "content_sha256": journal["content_sha256_before_self_field"],
    }


def execute_line_search(*args, **kwargs):
    kwargs = dict(kwargs)
    expected = np.random.default_rng(seed=43).integers(
        0, 2**30, size=32, dtype=np.int64,
    ).tolist()
    if kwargs.get("seeds") != expected:
        raise RuntimeError("v11 inherited population schedule changed")
    kwargs["seeds"] = list(PERTURBATION_SEEDS_V11)
    with driver_v6.scoped_legacy_audit_v6():
        journal = driver_v4.execute_line_search(*args, **kwargs)
    trainer = args[0] if args else kwargs["trainer"]
    plan = trainer._latest_anchor_plan
    anchor_v11.validate_resident_cross_v11(plan, recompute_numeric=True)
    anchor_v11.anchor_v8.validate_robust_plan_v8(plan, recompute_numeric=True)
    binding_v5 = anchor_v11.anchor_v5.validate_robust_plan_v5(
        plan, recompute_numeric=True,
    )
    journal["schema"] = "eggroll-es-anchor-alpha-line-search-v11"
    journal["snapshot"]["schema"] = "eggroll-es-anchor-line-search-snapshot-v11"
    journal["policy"].update(V11_POLICY)
    coefficient_plan = journal["coefficient_plan"]
    coefficient_plan["domain_scores_v5"] = list(plan["domain_scores"])
    coefficient_plan["anchor_scores_v5"] = list(plan["anchor_scores"])
    coefficient_plan["document_lcb_anchor_v5"] = plan["document_lcb_anchor_v5"]
    coefficient_plan["robust_plan_binding_v5"] = binding_v5
    coefficient_plan["resident_sign_cross_v11"] = copy.deepcopy(
        plan["resident_sign_cross_v11"]
    )
    journal.pop("content_sha256_before_self_field", None)
    journal["content_sha256_before_self_field"] = driver_v1.canonical_sha256(
        journal
    )
    try:
        validate_completed_journal_v11(journal)
    except Exception as error:
        journal["status"] = "failed"
        journal["failure"] = {
            "type": type(error).__name__, "message": str(error),
            "phase": "validating_complete_v11_exact_equivalence",
        }
        journal.pop("content_sha256_before_self_field", None)
        driver_v1.atomic_write_json(kwargs["journal_path"], journal)
        raise
    driver_v1.atomic_write_json(kwargs["journal_path"], journal)
    return journal


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    bundle, remaining = anchor_v11.parse_frozen_layer_plan_cli_v11(argv)
    execution, remaining = validate_frozen_execution_cli_v11(remaining, bundle)
    set_active_v11(bundle, execution)
    if execution["dry_run"]:
        payload = {
            "schema": "eggroll-es-resident-sign-dry-run-v11",
            "data_seed": 43, "base_direction_count": 32,
            "unique_signed_direction_count": 64,
            "actual_perturb_restore_cycle_count": 64,
            "all_engine_sign_residency_count": 16,
            "domain_signed_score_count": 128,
            "anchor_signed_response_count": 128,
            "resident_generation_order": ["D43", "A43", "A44", "D44"],
            "targets": [0.0],
            "v10_report_content_sha256": V10_REPORT_CONTENT_SHA256_V11,
            "recipe_sha256": driver_v1.canonical_sha256({
                key: execution[key] for key in V11_RECIPE_KEYS
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
    driver_v1.anchor = anchor_v11
    driver_v1.build_snapshot = build_snapshot
    driver_v1.execute_line_search = execute_line_search
    driver_v1.base.build_train_loader = driver_v10._crossed_train_loader_v10
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
