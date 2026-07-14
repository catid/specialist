#!/usr/bin/env python3
"""Fail-closed V11c retry with the complete substituted-anchor API."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import run_eggroll_es_anchor_equivalence_v11b as driver_v11b
import run_eggroll_es_anchor_line_search as driver_v1
import run_eggroll_es_anchor_line_search_v4 as driver_v4
import run_eggroll_es_anchor_line_search_v5 as driver_v5
import run_eggroll_es_anchor_line_search_v6 as driver_v6
import train_eggroll_es_specialist_anchor_v11c as anchor_v11c


ROOT = Path(__file__).resolve().parent
EXPERIMENT_NAME_V11C = (
    "snapshot794_layer_v11c_middle_late_resident_sign_exact_v10_"
    "complete_api_d43d44_a43a44_basis20260714"
)
V11B_FAILURE_PATH_V11C = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_RESIDENT_SIGN_EQUIVALENCE_V11B_FAILURE.md"
).resolve()
V11B_FAILURE_FILE_SHA256_V11C = (
    "c517bfa02bcf1fb3a75bc063cf3d33c721c1d25f99a5c4f4964d520de53c1cfd"
)
V11B_FAILURE_COMMIT_V11C = (
    "013d042371b9e9b0d2a050b53072b392a86a2574"
)
V11B_FAILED_SOURCE_COMMIT_V11C = (
    "3cb02bc269dc4d36d64bc6ec7a63e6b8059afa12"
)
EXPECTED_V11B_FAILURE_BINDING_SHA256_V11C = (
    "cfac67c750bef3b7930a6d1a1a7e751bf17d0d33740cac7389cd5b8fb8de72ba"
)
EXPECTED_RECIPE_SHA256_V11C = (
    "9db34afa06ace48ae9a77ccc8012a38bf55b874c0e72fcdfa8ff80609111a065"
)
ANCHOR_RUNTIME_API_V11C = (
    "__file__", "coefficient_sha256", "load_anchor_prose", "load_trainer",
)
V11C_POLICY = {
    "resident_sign_api_retry_family_v11c": "V11b_exact_complete_anchor_facade",
    "stage_v11c": "equivalence_api_retry",
    "target_alpha_zero_only_v11c": True,
    "benchmark_selection_forbidden_v11c": True,
    "requires_exact_v11b_algorithm_v11c": True,
    "requires_bound_v11b_launch_failure_v11c": True,
    "complete_driver_v1_anchor_api_required_v11c": True,
    "pre_engine_anchor_api_preflight_required_v11c": True,
}
V11C_IMPLEMENTATION_PATHS = {
    "worker_v11c": ROOT / "eggroll_es_worker_v11c.py",
    "trainer_v11c": ROOT / "train_eggroll_es_specialist_anchor_v11c.py",
    "driver_v11c": Path(__file__).resolve(),
    "reporter_v11c": ROOT / "report_eggroll_es_equivalence_v11c.py",
    "tests_v11c": ROOT / "test_eggroll_es_equivalence_v11c.py",
    "protocol_v11c": (
        ROOT / "experiments/eggroll_es_hpo/"
        "S6_RESIDENT_SIGN_EQUIVALENCE_V11C_PROTOCOL.md"
    ),
    "v11b_failure_evidence": V11B_FAILURE_PATH_V11C,
}
_ACTIVE_EXECUTION_V11C = None


def _file_sha256(path):
    return driver_v1.file_sha256(path)


def _v11b_failure_evidence_v11c(path):
    path = Path(path).resolve()
    driver_v11b.driver_v11.driver_v8.offline_audit._assert_no_heldout(
        str(path), "v11c V11b launch failure evidence",
    )
    if (
        path != V11B_FAILURE_PATH_V11C
        or _file_sha256(path) != V11B_FAILURE_FILE_SHA256_V11C
    ):
        raise RuntimeError("v11c requires exact committed V11b failure evidence")
    text = path.read_text()
    required = (
        V11B_FAILED_SOURCE_COMMIT_V11C,
        "failed before engine",
        "creation, GPU allocation, journal creation",
        "AttributeError: module 'train_eggroll_es_specialist_anchor_v11b' has no attribute 'load_anchor_prose'",
        "launch-shaped regression that covers every anchor-module symbol",
    )
    if not all(fragment in text for fragment in required):
        raise RuntimeError("v11c V11b failure evidence semantics changed")
    binding = {
        "schema": "eggroll-es-v11b-launch-failure-evidence-v11c",
        "path": str(path),
        "file_sha256": V11B_FAILURE_FILE_SHA256_V11C,
        "commit": V11B_FAILURE_COMMIT_V11C,
        "failed_source_commit": V11B_FAILED_SOURCE_COMMIT_V11C,
        "failure_phase": "inherited_anchor_data_setup_before_engine_creation",
        "exception_type": "AttributeError",
        "missing_symbol": "load_anchor_prose",
        "gpu_allocation_started": False,
        "journal_created": False,
        "run_directory_created": False,
        "model_update_applied": False,
    }
    binding["binding_sha256"] = driver_v1.canonical_sha256(binding)
    if (
        EXPECTED_V11B_FAILURE_BINDING_SHA256_V11C is not None
        and binding["binding_sha256"]
        != EXPECTED_V11B_FAILURE_BINDING_SHA256_V11C
    ):
        raise RuntimeError("v11c V11b failure evidence binding changed")
    return binding


def audit_anchor_runtime_api_v11c(module, bundle):
    """Resolve every anchor symbol used by driver-v1 before engine creation."""
    surface = {}
    for name in ANCHOR_RUNTIME_API_V11C:
        if not hasattr(module, name):
            raise RuntimeError(f"v11c anchor runtime API is missing {name}")
        value = getattr(module, name)
        if name != "__file__" and not callable(value):
            raise RuntimeError(f"v11c anchor runtime API {name} is not callable")
        surface[name] = str(Path(value).resolve()) if name == "__file__" else True
    if Path(surface["__file__"]).resolve() != Path(module.__file__).resolve():
        raise RuntimeError("v11c anchor __file__ identity changed")
    anchor_data = module.load_anchor_prose(
        ROOT / "data/general_prose_anchor_v1.jsonl",
        ROOT / "data/general_prose_anchor_v1.report.json",
    )
    if len(anchor_data.get("rows", [])) != 128:
        raise RuntimeError("v11c anchor loader returned the wrong frozen row count")
    trainer_class = module.load_trainer(bundle)
    if not isinstance(trainer_class, type):
        raise RuntimeError("v11c load_trainer did not resolve a trainer class")
    seeds = list(driver_v11b.driver_v11.PERTURBATION_SEEDS_V11)
    coefficients = [0.0] * len(seeds)
    coefficient_identity = module.coefficient_sha256(seeds, coefficients)
    if (
        not isinstance(coefficient_identity, str)
        or len(coefficient_identity) != 64
        or coefficient_identity
        != module.coefficient_sha256(list(seeds), list(coefficients))
    ):
        raise RuntimeError("v11c coefficient_sha256 runtime API is unstable")
    audit = {
        "schema": "eggroll-es-anchor-runtime-api-preflight-v11c",
        "required_symbols": list(ANCHOR_RUNTIME_API_V11C),
        "resolved_symbols": surface,
        "anchor_rows": len(anchor_data["rows"]),
        "trainer_class_module": trainer_class.__module__,
        "trainer_class_name": trainer_class.__name__,
        "coefficient_probe_sha256": coefficient_identity,
        "engine_creation_attempted": False,
        "passed": True,
    }
    audit["content_sha256_before_self_field"] = driver_v1.canonical_sha256(audit)
    return audit


def _v11b_compat_execution_v11c(v10, failed_v11):
    recipe = driver_v11b.frozen_recipe_v11b(v10, failed_v11)
    return {
        **copy.deepcopy(recipe),
        "schema": "eggroll-es-resident-sign-execution-v11b",
        "stage": "equivalence_retry", "arm": "middle_late",
        "dry_run": False,
        "recipe_sha256": driver_v1.canonical_sha256(recipe),
        "v10_evidence": copy.deepcopy(v10),
    }


def frozen_recipe_v11c(v10, failed_v11, failed_v11b):
    recipe = driver_v11b.frozen_recipe_v11b(v10, failed_v11)
    recipe.update({
        "experiment_name": EXPERIMENT_NAME_V11C,
        "failed_v11b_evidence_v11c": copy.deepcopy(failed_v11b),
        "anchor_runtime_api_v11c": list(ANCHOR_RUNTIME_API_V11C),
    })
    return recipe


def validate_frozen_execution_cli_v11c(argv, bundle):
    inherited = driver_v5.validate_frozen_execution_cli_v5(argv)
    stage_parser = argparse.ArgumentParser(add_help=False)
    stage_parser.add_argument("--v11c-stage", choices=("equivalence_api_retry",))
    stage_parser.add_argument("--v11c-v10-report")
    stage_parser.add_argument("--v11c-failed-v11-journal")
    stage_parser.add_argument("--v11c-v11b-failure-evidence")
    stage_parser.add_argument("--v11c-perturbation-basis-seed", type=int)
    stage_parser.add_argument("--v11c-dry-run", action="store_true")
    stage, remaining = stage_parser.parse_known_args(list(argv))
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--population-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--target-alphas")
    parser.add_argument("--train-dataset", default=str(driver_v11b.driver_v11.FROZEN_TRAIN_DATASET_V11))
    parser.add_argument("--eval-dataset", default=str(driver_v11b.driver_v11.FROZEN_EVAL_DATASET_V11))
    parser.add_argument("--reward-function-timeout", type=int, default=10)
    parser.add_argument("--output-directory", default=str(driver_v11b.driver_v11.FROZEN_OUTPUT_DIRECTORY_V11))
    parser.add_argument("--experiment-name")
    parser.add_argument("--logging", default="none")
    parser.add_argument("--wandb-project", default="specialist-eggroll-es")
    runtime, _ = parser.parse_known_args(list(argv))
    metadata = anchor_v11c.validate_frozen_layer_plan_bundle_v11c(bundle)
    if stage.v11c_stage != "equivalence_api_retry":
        raise ValueError("v11c requires --v11c-stage equivalence_api_retry")
    if runtime.seed != 43 or runtime.population_size != 32 or runtime.batch_size != 128:
        raise ValueError("v11c requires seed43/pop32/combined-batch128")
    if driver_v1.parse_target_alphas(runtime.target_alphas) != [0.0]:
        raise ValueError("v11c target must be exactly alpha zero")
    if (
        stage.v11c_perturbation_basis_seed
        != driver_v11b.driver_v11.PERTURBATION_BASIS_SEED_V11
    ):
        raise ValueError("v11c perturbation basis seed changed")
    if (
        metadata["plan"] != "middle_late"
        or bundle["plan_sha256"]
        != driver_v11b.driver_v11.MIDDLE_LATE_PLAN_SHA256_V11
        or Path(runtime.train_dataset).resolve()
        != driver_v11b.driver_v11.FROZEN_TRAIN_DATASET_V11
        or Path(runtime.eval_dataset).resolve()
        != driver_v11b.driver_v11.FROZEN_EVAL_DATASET_V11
        or runtime.reward_function_timeout != 10
        or Path(runtime.output_directory).resolve()
        != driver_v11b.driver_v11.FROZEN_OUTPUT_DIRECTORY_V11
        or runtime.experiment_name != EXPERIMENT_NAME_V11C
        or runtime.logging != "none"
        or runtime.wandb_project != "specialist-eggroll-es"
    ):
        raise ValueError("v11c frozen runtime recipe changed")
    if any(value is None for value in (
        stage.v11c_v10_report, stage.v11c_failed_v11_journal,
        stage.v11c_v11b_failure_evidence,
    )):
        raise ValueError("v11c requires V10, V11, and V11b evidence")
    v10 = driver_v11b._bound_v10_evidence_v11b(stage.v11c_v10_report)
    failed_v11 = driver_v11b._failed_v11_evidence_v11b(
        stage.v11c_failed_v11_journal
    )
    failed_v11b = _v11b_failure_evidence_v11c(
        stage.v11c_v11b_failure_evidence
    )
    api_audit = audit_anchor_runtime_api_v11c(anchor_v11c, bundle)
    frozen = frozen_recipe_v11c(v10, failed_v11, failed_v11b)
    recipe_sha = driver_v1.canonical_sha256(frozen)
    if EXPECTED_RECIPE_SHA256_V11C is not None and recipe_sha != EXPECTED_RECIPE_SHA256_V11C:
        raise RuntimeError("v11c frozen recipe hash changed")
    execution = {
        **inherited, **copy.deepcopy(frozen),
        "schema": "eggroll-es-resident-sign-execution-v11c",
        "stage": "equivalence_api_retry", "arm": "middle_late",
        "dry_run": stage.v11c_dry_run,
        "recipe_sha256": recipe_sha,
        "v10_evidence": v10,
        "failed_v11_evidence": failed_v11,
        "anchor_runtime_api_preflight": api_audit,
    }
    return execution, remaining


def set_active_v11c(bundle, execution):
    global _ACTIVE_EXECUTION_V11C
    anchor_v11c.set_default_layer_plan_bundle_v11c(bundle)
    driver_v11b.set_active_v11b(
        bundle, _v11b_compat_execution_v11c(
            execution["v10_evidence"], execution["failed_v11_evidence"],
        ),
    )
    _ACTIVE_EXECUTION_V11C = execution


def _implementation_v11c():
    return {key: _file_sha256(path) for key, path in V11C_IMPLEMENTATION_PATHS.items()}


def build_snapshot(*args, **kwargs):
    snapshot = driver_v11b.build_snapshot(*args, **kwargs)
    execution = _ACTIVE_EXECUTION_V11C
    implementation = _implementation_v11c()
    snapshot["schema"] = "eggroll-es-anchor-line-search-snapshot-v11c"
    snapshot["resident_sign_api_retry_v11c"] = {
        "schema": "eggroll-es-resident-sign-api-retry-snapshot-v11c",
        "stage": "equivalence_api_retry", "arm": "middle_late",
        "anchor_runtime_api_preflight": copy.deepcopy(
            execution["anchor_runtime_api_preflight"]
        ),
        "v11b_failure_evidence": copy.deepcopy(
            execution["failed_v11b_evidence_v11c"]
        ),
        "recipe_sha256": execution["recipe_sha256"],
        "implementation": implementation,
        "implementation_bundle_sha256": driver_v1.canonical_sha256(
            implementation
        ),
    }
    return snapshot


def _compatibility_journal_v11c(journal):
    compatible = copy.deepcopy(journal)
    compatible["schema"] = "eggroll-es-anchor-alpha-line-search-v11b"
    compatible["snapshot"]["schema"] = (
        "eggroll-es-anchor-line-search-snapshot-v11b"
    )
    compatible["snapshot"].pop("resident_sign_api_retry_v11c", None)
    for key in V11C_POLICY:
        compatible["policy"].pop(key, None)
    compatible.pop("content_sha256_before_self_field", None)
    compatible["content_sha256_before_self_field"] = driver_v1.canonical_sha256(
        compatible
    )
    return compatible


def validate_completed_journal_v11c(journal):
    if (
        not isinstance(journal, dict)
        or journal.get("schema") != "eggroll-es-anchor-alpha-line-search-v11c"
        or journal.get("status") != "complete"
        or journal.get("in_progress") is not None
        or journal.get("targets") != [0.0]
        or journal.get("content_sha256_before_self_field")
        != driver_v1.canonical_sha256({
            key: value for key, value in journal.items()
            if key != "content_sha256_before_self_field"
        })
        or journal.get("policy") != {
            **driver_v11b.driver_v11.INHERITED_POLICY_V11,
            **driver_v11b.driver_v11.V11_POLICY,
            **driver_v11b.V11B_POLICY,
            **V11C_POLICY,
        }
    ):
        raise RuntimeError("v11c journal identity, policy, or completion changed")
    snapshot = journal.get("snapshot", {})
    retry = snapshot.get("resident_sign_api_retry_v11c", {})
    implementation = _implementation_v11c()
    evidence = _v11b_failure_evidence_v11c(V11B_FAILURE_PATH_V11C)
    v10 = driver_v11b._bound_v10_evidence_v11b(
        driver_v11b.driver_v11.V10_REPORT_PATH_V11
    )
    failed_v11 = driver_v11b._failed_v11_evidence_v11b(
        driver_v11b.FAILED_V11_JOURNAL_PATH_V11B
    )
    recipe = frozen_recipe_v11c(v10, failed_v11, evidence)
    api = retry.get("anchor_runtime_api_preflight", {})
    if (
        snapshot.get("schema") != "eggroll-es-anchor-line-search-snapshot-v11c"
        or retry.get("schema")
        != "eggroll-es-resident-sign-api-retry-snapshot-v11c"
        or retry.get("v11b_failure_evidence") != evidence
        or retry.get("recipe_sha256") != driver_v1.canonical_sha256(recipe)
        or driver_v1.canonical_sha256(recipe) != EXPECTED_RECIPE_SHA256_V11C
        or api.get("passed") is not True
        or api.get("required_symbols") != list(ANCHOR_RUNTIME_API_V11C)
        or api.get("engine_creation_attempted") is not False
        or api.get("content_sha256_before_self_field")
        != driver_v1.canonical_sha256({
            key: value for key, value in api.items()
            if key != "content_sha256_before_self_field"
        })
        or retry.get("implementation") != implementation
        or retry.get("implementation_bundle_sha256")
        != driver_v1.canonical_sha256(implementation)
    ):
        raise RuntimeError("v11c API retry snapshot changed")
    inherited = driver_v11b.validate_completed_journal_v11b(
        _compatibility_journal_v11c(journal)
    )
    return {
        **inherited,
        "stage": "equivalence_api_retry",
        "v11b_failure_binding_sha256": evidence["binding_sha256"],
        "anchor_runtime_api_preflight_sha256": retry[
            "anchor_runtime_api_preflight"
        ]["content_sha256_before_self_field"],
        "content_sha256": journal["content_sha256_before_self_field"],
    }


def execute_line_search(*args, **kwargs):
    kwargs = dict(kwargs)
    with driver_v6.scoped_legacy_audit_v6():
        journal = driver_v4.execute_line_search(*args, **kwargs)
    trainer = args[0] if args else kwargs["trainer"]
    plan = trainer._latest_anchor_plan
    anchor_v11c.anchor_v11b.anchor_v11.validate_resident_cross_v11(
        plan, recompute_numeric=True,
    )
    binding_v5 = (
        anchor_v11c.anchor_v11b.anchor_v11.anchor_v5.validate_robust_plan_v5(
            plan, recompute_numeric=True,
        )
    )
    journal["schema"] = "eggroll-es-anchor-alpha-line-search-v11c"
    journal["snapshot"]["schema"] = "eggroll-es-anchor-line-search-snapshot-v11c"
    journal["policy"].update(driver_v11b.driver_v11.V11_POLICY)
    journal["policy"].update(driver_v11b.V11B_POLICY)
    journal["policy"].update(V11C_POLICY)
    coefficient_plan = journal["coefficient_plan"]
    coefficient_plan["domain_scores_v5"] = list(plan["domain_scores"])
    coefficient_plan["anchor_scores_v5"] = list(plan["anchor_scores"])
    coefficient_plan["document_lcb_anchor_v5"] = plan["document_lcb_anchor_v5"]
    coefficient_plan["robust_plan_binding_v5"] = binding_v5
    coefficient_plan["resident_sign_cross_v11"] = copy.deepcopy(
        plan["resident_sign_cross_v11"]
    )
    coefficient_plan["dual_domain_manifest_binding_v11b"] = copy.deepcopy(
        plan["dual_domain_manifest_binding_v11b"]
    )
    journal.pop("content_sha256_before_self_field", None)
    journal["content_sha256_before_self_field"] = driver_v1.canonical_sha256(
        journal
    )
    try:
        validate_completed_journal_v11c(journal)
    except Exception as error:
        journal["status"] = "failed"
        journal["failure"] = {
            "type": type(error).__name__, "message": str(error),
            "phase": "validating_complete_v11c_exact_equivalence",
        }
        journal.pop("content_sha256_before_self_field", None)
        driver_v1.atomic_write_json(kwargs["journal_path"], journal)
        raise
    driver_v1.atomic_write_json(kwargs["journal_path"], journal)
    return journal


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    bundle, remaining = anchor_v11c.parse_frozen_layer_plan_cli_v11c(argv)
    execution, remaining = validate_frozen_execution_cli_v11c(remaining, bundle)
    set_active_v11c(bundle, execution)
    if execution["dry_run"]:
        payload = {
            "schema": "eggroll-es-resident-sign-complete-api-dry-run-v11c",
            "stage": "equivalence_api_retry",
            "anchor_runtime_api_preflight": execution[
                "anchor_runtime_api_preflight"
            ],
            "v11b_failure_binding_sha256": execution[
                "failed_v11b_evidence_v11c"
            ]["binding_sha256"],
            "v10_evidence_binding_sha256": execution["v10_evidence"][
                "binding_sha256"
            ],
            "recipe_sha256": execution["recipe_sha256"],
            "actual_perturb_restore_cycle_count": 64,
            "all_engine_sign_residency_count": 16,
            "gpu_ids": [0, 1, 2, 3],
        }
        print(json.dumps(payload, sort_keys=True))
        return payload
    # This launch-shaped pre-engine guard is deliberately repeated on the real
    # path immediately before driver-v1 performs data setup and engine creation.
    execution["anchor_runtime_api_preflight"] = audit_anchor_runtime_api_v11c(
        anchor_v11c, bundle,
    )
    old_argv = sys.argv
    old_anchor = driver_v1.anchor
    old_build = driver_v1.build_snapshot
    old_execute = driver_v1.execute_line_search
    old_loader = driver_v1.base.build_train_loader
    sys.argv = [old_argv[0], *remaining]
    driver_v1.anchor = anchor_v11c
    driver_v1.build_snapshot = build_snapshot
    driver_v1.execute_line_search = execute_line_search
    driver_v1.base.build_train_loader = driver_v11b._raw_crossed_train_loader_v11b
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
