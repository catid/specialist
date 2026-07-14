#!/usr/bin/env python3
"""Fail-closed V11b retry with dual raw/templated manifest identities."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import run_eggroll_es_anchor_equivalence_v11 as driver_v11
import run_eggroll_es_anchor_line_search as driver_v1
import run_eggroll_es_anchor_line_search_v4 as driver_v4
import run_eggroll_es_anchor_line_search_v5 as driver_v5
import run_eggroll_es_anchor_line_search_v6 as driver_v6
import run_eggroll_es_anchor_variance_v10 as driver_v10
import train_eggroll_es_specialist_anchor_v11b as anchor_v11b


ROOT = Path(__file__).resolve().parent
EXPERIMENT_NAME_V11B = (
    "snapshot794_layer_v11b_middle_late_resident_sign_exact_v10_"
    "dual_manifest_d43d44_a43a44_basis20260714"
)
FAILED_V11_JOURNAL_PATH_V11B = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "snapshot794_layer_v11_middle_late_resident_sign_exact_v10_"
    "d43d44_a43a44_basis20260714/alpha_line_search.json"
).resolve()
FAILED_V11_JOURNAL_FILE_SHA256_V11B = (
    "2c9680fc55b6f90ef306ecf7f7493c7f448f48c36790af9992ffe4236ed19924"
)
FAILED_V11_JOURNAL_CANONICAL_SHA256_V11B = (
    "e207345703fd4b9becae6e60ffc47e4c805d0f55a8c7a60d30976728541c9865"
)
EXPECTED_FAILED_EVIDENCE_BINDING_SHA256_V11B = (
    "b0955af23fd5cabb20f26b6a68b9672ad8fe68dff9500f4a3c91aa39bb82cdbb"
)
EXPECTED_RECIPE_SHA256_V11B = (
    "3ba91d4b8f089c5140638164897499506dff06e289a8607c8aaf9c7ad0f769c3"
)
V11B_POLICY = {
    "resident_sign_retry_family_v11b": "V11_exact_with_dual_raw_templated_manifests",
    "stage_v11b": "equivalence_retry",
    "target_alpha_zero_only_v11b": True,
    "benchmark_selection_forbidden_v11b": True,
    "requires_exact_v10_response_equivalence_v11b": True,
    "requires_bound_failed_v11_manifest_mismatch_v11b": True,
    "raw_and_templated_manifest_identities_required_v11b": True,
}
V11B_IMPLEMENTATION_PATHS = {
    "worker_v11b": ROOT / "eggroll_es_worker_v11b.py",
    "trainer_v11b": ROOT / "train_eggroll_es_specialist_anchor_v11b.py",
    "driver_v11b": Path(__file__).resolve(),
    "reporter_v11b": ROOT / "report_eggroll_es_equivalence_v11b.py",
    "tests_v11b": ROOT / "test_eggroll_es_equivalence_v11b.py",
    "protocol_v11b": (
        ROOT / "experiments/eggroll_es_hpo/"
        "S6_RESIDENT_SIGN_EQUIVALENCE_V11B_PROTOCOL.md"
    ),
}
_ACTIVE_BUNDLE_V11B = None
_ACTIVE_EXECUTION_V11B = None


def _file_sha256(path):
    return driver_v1.file_sha256(path)


def _failed_v11_evidence_v11b(path):
    path = Path(path).resolve()
    driver_v11.driver_v8.offline_audit._assert_no_heldout(
        str(path), "v11b failed V11 journal",
    )
    if (
        path != FAILED_V11_JOURNAL_PATH_V11B
        or _file_sha256(path) != FAILED_V11_JOURNAL_FILE_SHA256_V11B
    ):
        raise RuntimeError("v11b requires the exact failed V11 journal")
    journal = json.loads(path.read_text())
    if (
        driver_v1.canonical_sha256(journal)
        != FAILED_V11_JOURNAL_CANONICAL_SHA256_V11B
        or journal.get("schema") != "eggroll-es-anchor-alpha-line-search-v1"
        or journal.get("status") != "failed"
        or journal.get("coefficient_plan") is not None
        or journal.get("failure") != {
            "type": "RuntimeError",
            "message": "v11 captured D43 manifest changed",
            "completed_state_count": 1,
            "coefficient_plan_estimated": False,
        }
        or journal.get("in_progress") != {
            "state_index": 0,
            "target_alpha": 0.0,
            "phase": "estimating_fixed_coefficient_plan",
        }
    ):
        raise RuntimeError("v11b failed V11 evidence semantics changed")
    snapshot = journal.get("snapshot", {})
    resident = snapshot.get("resident_sign_equivalence_v11", {})
    if (
        snapshot.get("implementation", {}).get("distributed_trainer_v11")
        != _file_sha256(ROOT / "train_eggroll_es_specialist_anchor_v11.py")
        or resident.get("v10_equivalence_evidence", {}).get("passed") is not True
        or resident.get("domain_manifests")
        != anchor_v11b.RAW_DOMAIN_MANIFESTS_V11B
    ):
        raise RuntimeError("v11b failed V11 snapshot binding changed")
    binding = {
        "schema": "eggroll-es-failed-v11-manifest-evidence-v11b",
        "path": str(path),
        "file_sha256": FAILED_V11_JOURNAL_FILE_SHA256_V11B,
        "canonical_sha256": FAILED_V11_JOURNAL_CANONICAL_SHA256_V11B,
        "failure_type": "RuntimeError",
        "failure_message": "v11 captured D43 manifest changed",
        "failure_phase": "estimating_fixed_coefficient_plan",
        "raw_manifest_was_incorrectly_compared_to_templated_capture": True,
        "no_coefficient_plan_estimated": True,
        "completed_alpha_zero_state_count": 1,
    }
    binding["binding_sha256"] = driver_v1.canonical_sha256(binding)
    if (
        EXPECTED_FAILED_EVIDENCE_BINDING_SHA256_V11B is not None
        and binding["binding_sha256"]
        != EXPECTED_FAILED_EVIDENCE_BINDING_SHA256_V11B
    ):
        raise RuntimeError("v11b failed-run evidence binding changed")
    return binding


def frozen_recipe_v11b(v10_evidence, failed_evidence):
    failed_journal = json.loads(FAILED_V11_JOURNAL_PATH_V11B.read_text())
    recipe = copy.deepcopy(
        failed_journal["snapshot"]["resident_sign_equivalence_v11"]["recipe"]
    )
    if (
        driver_v1.canonical_sha256(recipe) != driver_v11.EXPECTED_RECIPE_SHA256_V11
        or recipe.get("v10_equivalence_evidence") != v10_evidence
    ):
        raise RuntimeError("v11b exact failed-run V11 recipe binding changed")
    recipe.update({
        "experiment_name": EXPERIMENT_NAME_V11B,
        "raw_domain_manifests_v11b": copy.deepcopy(
            anchor_v11b.RAW_DOMAIN_MANIFESTS_V11B
        ),
        "templated_domain_manifests_v11b": copy.deepcopy(
            anchor_v11b.TEMPLATED_DOMAIN_MANIFESTS_V11B
        ),
        "failed_v11_evidence_v11b": copy.deepcopy(failed_evidence),
    })
    return recipe


def _compat_execution_v11(v10_evidence):
    failed_journal = json.loads(FAILED_V11_JOURNAL_PATH_V11B.read_text())
    frozen = copy.deepcopy(
        failed_journal["snapshot"]["resident_sign_equivalence_v11"]["recipe"]
    )
    if (
        driver_v1.canonical_sha256(frozen)
        != driver_v11.EXPECTED_RECIPE_SHA256_V11
        or frozen.get("v10_equivalence_evidence") != v10_evidence
    ):
        raise RuntimeError("v11b V11 compatibility recipe changed")
    return {
        **copy.deepcopy(frozen),
        "schema": "eggroll-es-resident-sign-execution-v11",
        "stage": "equivalence", "arm": "middle_late", "dry_run": False,
    }


def _bound_v10_evidence_v11b(path):
    """Bind the exact V10 pass already deep-validated by failed V11."""
    path = Path(path).resolve()
    if (
        path != driver_v11.V10_REPORT_PATH_V11
        or _file_sha256(path) != driver_v11.V10_REPORT_FILE_SHA256_V11
    ):
        raise RuntimeError("v11b exact V10 report file changed")
    report = json.loads(path.read_text())
    if (
        report.get("passed") is not True
        or report.get("content_sha256_before_self_field")
        != driver_v11.V10_REPORT_CONTENT_SHA256_V11
        or report.get("content_sha256_before_self_field")
        != driver_v1.canonical_sha256({
            key: value for key, value in report.items()
            if key != "content_sha256_before_self_field"
        })
        or report.get("journal_file_sha256")
        != driver_v11.V10_JOURNAL_FILE_SHA256_V11
        or report.get("journal_content_sha256")
        != driver_v11.V10_JOURNAL_CONTENT_SHA256_V11
    ):
        raise RuntimeError("v11b exact V10 report content changed")
    failed_journal = json.loads(FAILED_V11_JOURNAL_PATH_V11B.read_text())
    evidence = copy.deepcopy(
        failed_journal["snapshot"]["resident_sign_equivalence_v11"]
        ["v10_equivalence_evidence"]
    )
    if (
        evidence.get("binding_sha256")
        != driver_v11.EXPECTED_V10_EVIDENCE_BINDING_SHA256_V11
        or evidence.get("report_path") != str(path)
        or evidence.get("report_file_sha256")
        != driver_v11.V10_REPORT_FILE_SHA256_V11
        or evidence.get("report_content_sha256")
        != driver_v11.V10_REPORT_CONTENT_SHA256_V11
        or evidence.get("journal_file_sha256")
        != driver_v11.V10_JOURNAL_FILE_SHA256_V11
        or evidence.get("journal_content_sha256")
        != driver_v11.V10_JOURNAL_CONTENT_SHA256_V11
        or evidence.get("passed") is not True
    ):
        raise RuntimeError("v11b failed-run V10 evidence binding changed")
    return evidence


def validate_frozen_execution_cli_v11b(argv, bundle):
    inherited = driver_v5.validate_frozen_execution_cli_v5(argv)
    stage_parser = argparse.ArgumentParser(add_help=False)
    stage_parser.add_argument("--v11b-stage", choices=("equivalence_retry",))
    stage_parser.add_argument("--v11b-v10-report")
    stage_parser.add_argument("--v11b-failed-v11-journal")
    stage_parser.add_argument("--v11b-perturbation-basis-seed", type=int)
    stage_parser.add_argument("--v11b-dry-run", action="store_true")
    stage, remaining = stage_parser.parse_known_args(list(argv))
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--population-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--target-alphas")
    parser.add_argument("--train-dataset", default=str(driver_v11.FROZEN_TRAIN_DATASET_V11))
    parser.add_argument("--eval-dataset", default=str(driver_v11.FROZEN_EVAL_DATASET_V11))
    parser.add_argument("--reward-function-timeout", type=int, default=10)
    parser.add_argument("--output-directory", default=str(driver_v11.FROZEN_OUTPUT_DIRECTORY_V11))
    parser.add_argument("--experiment-name")
    parser.add_argument("--logging", default="none")
    parser.add_argument("--wandb-project", default="specialist-eggroll-es")
    runtime, _ = parser.parse_known_args(list(argv))
    metadata = anchor_v11b.validate_frozen_layer_plan_bundle_v11b(bundle)
    if stage.v11b_stage != "equivalence_retry":
        raise ValueError("v11b requires --v11b-stage equivalence_retry")
    if runtime.seed != 43 or runtime.population_size != 32 or runtime.batch_size != 128:
        raise ValueError("v11b requires seed43/pop32/combined-batch128")
    if driver_v1.parse_target_alphas(runtime.target_alphas) != [0.0]:
        raise ValueError("v11b target must be exactly alpha zero")
    if stage.v11b_perturbation_basis_seed != driver_v11.PERTURBATION_BASIS_SEED_V11:
        raise ValueError("v11b perturbation basis seed changed")
    if (
        metadata["plan"] != "middle_late"
        or bundle["plan_sha256"] != driver_v11.MIDDLE_LATE_PLAN_SHA256_V11
        or Path(runtime.train_dataset).resolve() != driver_v11.FROZEN_TRAIN_DATASET_V11
        or Path(runtime.eval_dataset).resolve() != driver_v11.FROZEN_EVAL_DATASET_V11
        or runtime.reward_function_timeout != 10
        or Path(runtime.output_directory).resolve()
        != driver_v11.FROZEN_OUTPUT_DIRECTORY_V11
        or runtime.experiment_name != EXPERIMENT_NAME_V11B
        or runtime.logging != "none"
        or runtime.wandb_project != "specialist-eggroll-es"
    ):
        raise ValueError("v11b frozen runtime recipe changed")
    if stage.v11b_v10_report is None or stage.v11b_failed_v11_journal is None:
        raise ValueError("v11b requires V10 pass and failed V11 evidence")
    v10 = _bound_v10_evidence_v11b(stage.v11b_v10_report)
    failed = _failed_v11_evidence_v11b(stage.v11b_failed_v11_journal)
    frozen = frozen_recipe_v11b(v10, failed)
    recipe_sha = driver_v1.canonical_sha256(frozen)
    if EXPECTED_RECIPE_SHA256_V11B is not None and recipe_sha != EXPECTED_RECIPE_SHA256_V11B:
        raise RuntimeError("v11b frozen recipe hash changed")
    execution = {
        **inherited, **copy.deepcopy(frozen),
        "schema": "eggroll-es-resident-sign-execution-v11b",
        "stage": "equivalence_retry", "arm": "middle_late",
        "dry_run": stage.v11b_dry_run,
        "recipe_sha256": recipe_sha,
        "v10_evidence": v10,
    }
    return execution, remaining


def set_active_v11b(bundle, execution):
    global _ACTIVE_BUNDLE_V11B, _ACTIVE_EXECUTION_V11B
    anchor_v11b.set_default_layer_plan_bundle_v11b(bundle)
    driver_v11.set_active_v11(bundle, _compat_execution_v11(
        execution["v10_evidence"]
    ))
    _ACTIVE_BUNDLE_V11B = bundle
    _ACTIVE_EXECUTION_V11B = execution


def _implementation_v11b():
    return {
        key: _file_sha256(path)
        for key, path in V11B_IMPLEMENTATION_PATHS.items()
    }


def build_snapshot(*args, **kwargs):
    snapshot = driver_v11.build_snapshot(*args, **kwargs)
    execution = _ACTIVE_EXECUTION_V11B
    implementation = _implementation_v11b()
    snapshot["schema"] = "eggroll-es-anchor-line-search-snapshot-v11b"
    snapshot["resident_sign_retry_v11b"] = {
        "schema": "eggroll-es-resident-sign-retry-snapshot-v11b",
        "stage": "equivalence_retry", "arm": "middle_late",
        "raw_question_answer_manifests": copy.deepcopy(
            anchor_v11b.RAW_DOMAIN_MANIFESTS_V11B
        ),
        "templated_prompt_answer_manifests": copy.deepcopy(
            anchor_v11b.TEMPLATED_DOMAIN_MANIFESTS_V11B
        ),
        "raw_validation_surfaces": ["crossed_train_loader", "snapshot"],
        "templated_validation_surface": "trainer_capture_before_generation",
        "failed_v11_evidence": copy.deepcopy(
            execution["failed_v11_evidence_v11b"]
        ),
        "v10_equivalence_evidence": copy.deepcopy(execution["v10_evidence"]),
        "recipe_sha256": execution["recipe_sha256"],
        "implementation": implementation,
        "implementation_bundle_sha256": driver_v1.canonical_sha256(
            implementation
        ),
    }
    return snapshot


def _raw_crossed_train_loader_v11b(dataset, batch_size, seed):
    loader = driver_v10._crossed_train_loader_v10(dataset, batch_size, seed)
    questions, answers = next(iter(loader))
    for label, start in (("D43", 0), ("D44", 64)):
        identity = driver_v1.canonical_sha256({
            "questions": questions[start:start + 64],
            "answers": answers[start:start + 64],
        })
        if identity != anchor_v11b.RAW_DOMAIN_MANIFESTS_V11B[label]["sha256"]:
            raise RuntimeError(f"v11b raw {label} loader manifest changed")
    return [(questions, answers)]


def _compatibility_journal_v11b(journal):
    compatible = copy.deepcopy(journal)
    compatible["schema"] = "eggroll-es-anchor-alpha-line-search-v11"
    compatible["snapshot"]["schema"] = "eggroll-es-anchor-line-search-snapshot-v11"
    compatible["snapshot"].pop("resident_sign_retry_v11b", None)
    for key in V11B_POLICY:
        compatible["policy"].pop(key, None)
    compatible["coefficient_plan"].pop("dual_domain_manifest_binding_v11b", None)
    compatible.pop("content_sha256_before_self_field", None)
    compatible["content_sha256_before_self_field"] = driver_v1.canonical_sha256(
        compatible
    )
    return compatible


def validate_completed_journal_v11b(journal):
    if (
        not isinstance(journal, dict)
        or journal.get("schema") != "eggroll-es-anchor-alpha-line-search-v11b"
        or journal.get("status") != "complete"
        or journal.get("in_progress") is not None
        or journal.get("targets") != [0.0]
        or journal.get("content_sha256_before_self_field")
        != driver_v1.canonical_sha256({
            key: value for key, value in journal.items()
            if key != "content_sha256_before_self_field"
        })
        or journal.get("policy")
        != {**driver_v11.INHERITED_POLICY_V11, **driver_v11.V11_POLICY, **V11B_POLICY}
    ):
        raise RuntimeError("v11b journal identity, policy, or completion changed")
    snapshot = journal.get("snapshot", {})
    retry = snapshot.get("resident_sign_retry_v11b", {})
    implementation = _implementation_v11b()
    failed = _failed_v11_evidence_v11b(FAILED_V11_JOURNAL_PATH_V11B)
    v10 = _bound_v10_evidence_v11b(driver_v11.V10_REPORT_PATH_V11)
    recipe = frozen_recipe_v11b(v10, failed)
    if (
        snapshot.get("schema") != "eggroll-es-anchor-line-search-snapshot-v11b"
        or retry.get("schema")
        != "eggroll-es-resident-sign-retry-snapshot-v11b"
        or retry.get("raw_question_answer_manifests")
        != anchor_v11b.RAW_DOMAIN_MANIFESTS_V11B
        or retry.get("templated_prompt_answer_manifests")
        != anchor_v11b.TEMPLATED_DOMAIN_MANIFESTS_V11B
        or retry.get("failed_v11_evidence") != failed
        or retry.get("v10_equivalence_evidence") != v10
        or retry.get("recipe_sha256") != driver_v1.canonical_sha256(recipe)
        or retry.get("implementation") != implementation
        or retry.get("implementation_bundle_sha256")
        != driver_v1.canonical_sha256(implementation)
    ):
        raise RuntimeError("v11b retry snapshot changed")
    dual = journal.get("coefficient_plan", {}).get(
        "dual_domain_manifest_binding_v11b"
    )
    if (
        not isinstance(dual, dict)
        or dual.get("raw_question_answer_manifests")
        != anchor_v11b.RAW_DOMAIN_MANIFESTS_V11B
        or dual.get("templated_prompt_answer_manifests")
        != anchor_v11b.TEMPLATED_DOMAIN_MANIFESTS_V11B
        or dual.get("content_sha256_before_self_field")
        != driver_v1.canonical_sha256({
            key: value for key, value in dual.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v11b dual manifest plan binding changed")
    inherited = driver_v11.validate_completed_journal_v11(
        _compatibility_journal_v11b(journal)
    )
    return {
        **inherited,
        "stage": "equivalence_retry",
        "failed_v11_evidence_binding_sha256": failed["binding_sha256"],
        "dual_manifest_binding_sha256": dual[
            "content_sha256_before_self_field"
        ],
        "content_sha256": journal["content_sha256_before_self_field"],
    }


def execute_line_search(*args, **kwargs):
    kwargs = dict(kwargs)
    with driver_v6.scoped_legacy_audit_v6():
        journal = driver_v4.execute_line_search(*args, **kwargs)
    trainer = args[0] if args else kwargs["trainer"]
    plan = trainer._latest_anchor_plan
    anchor_v11b.anchor_v11.validate_resident_cross_v11(
        plan, recompute_numeric=True,
    )
    binding_v5 = anchor_v11b.anchor_v11.anchor_v5.validate_robust_plan_v5(
        plan, recompute_numeric=True,
    )
    journal["schema"] = "eggroll-es-anchor-alpha-line-search-v11b"
    journal["snapshot"]["schema"] = "eggroll-es-anchor-line-search-snapshot-v11b"
    journal["policy"].update(driver_v11.V11_POLICY)
    journal["policy"].update(V11B_POLICY)
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
        validate_completed_journal_v11b(journal)
    except Exception as error:
        journal["status"] = "failed"
        journal["failure"] = {
            "type": type(error).__name__, "message": str(error),
            "phase": "validating_complete_v11b_exact_equivalence",
        }
        journal.pop("content_sha256_before_self_field", None)
        driver_v1.atomic_write_json(kwargs["journal_path"], journal)
        raise
    driver_v1.atomic_write_json(kwargs["journal_path"], journal)
    return journal


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    bundle, remaining = anchor_v11b.parse_frozen_layer_plan_cli_v11b(argv)
    execution, remaining = validate_frozen_execution_cli_v11b(remaining, bundle)
    set_active_v11b(bundle, execution)
    if execution["dry_run"]:
        payload = {
            "schema": "eggroll-es-resident-sign-dual-manifest-dry-run-v11b",
            "stage": "equivalence_retry",
            "raw_domain_manifests": copy.deepcopy(
                anchor_v11b.RAW_DOMAIN_MANIFESTS_V11B
            ),
            "templated_domain_manifests": copy.deepcopy(
                anchor_v11b.TEMPLATED_DOMAIN_MANIFESTS_V11B
            ),
            "failed_v11_evidence_binding_sha256": execution[
                "failed_v11_evidence_v11b"
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
    old_argv = sys.argv
    old_anchor = driver_v1.anchor
    old_build = driver_v1.build_snapshot
    old_execute = driver_v1.execute_line_search
    old_loader = driver_v1.base.build_train_loader
    sys.argv = [old_argv[0], *remaining]
    driver_v1.anchor = anchor_v11b
    driver_v1.build_snapshot = build_snapshot
    driver_v1.execute_line_search = execute_line_search
    driver_v1.base.build_train_loader = _raw_crossed_train_loader_v11b
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
