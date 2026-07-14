#!/usr/bin/env python3
"""Fail-closed split-seed population-32 middle-late diagnostic v8."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import aggregate_eggroll_es_anchor_replications as offline_audit
import report_eggroll_es_direction_stability_v7 as reporter_v7
import run_eggroll_es_anchor_line_search as driver_v1
import run_eggroll_es_anchor_line_search_v4 as driver_v4
import run_eggroll_es_anchor_line_search_v5 as driver_v5
import run_eggroll_es_anchor_line_search_v6 as driver_v6
import run_eggroll_es_anchor_stability_v7 as driver_v7
import train_eggroll_es_specialist_anchor_v8 as anchor_v8


ROOT = Path(__file__).resolve().parent
STABILITY_TARGETS_V8 = [0.0]
DATA_BOOTSTRAP_SEEDS_V8 = (43, 44)
PERTURBATION_BASIS_SEED_V8 = 20260714
POPULATION_SIZE_V8 = 32
COEFFICIENT_COSINE_THRESHOLD_V8 = 0.5
MIDDLE_LATE_PLAN_SHA256_V8 = (
    "03745c603a6b48898b41afbd4d9121aef276d7e45ca1a3ae14607ec5d1042cb9"
)
PERTURBATION_SEEDS_V8 = np.random.default_rng(
    seed=PERTURBATION_BASIS_SEED_V8
).integers(
    0, 2**30, size=POPULATION_SIZE_V8, dtype=np.int64,
).tolist()
PERTURBATION_BASIS_SHA256_V8 = driver_v1.canonical_sha256({
    "schema": "eggroll-es-perturbation-basis-v8",
    "basis_seed": PERTURBATION_BASIS_SEED_V8,
    "population_size": POPULATION_SIZE_V8,
    "seeds": PERTURBATION_SEEDS_V8,
})
EXPECTED_PERTURBATION_BASIS_SHA256_V8 = (
    "29e7ceb1753c39b310a176d827e222b9a5b2c85edf9f2fef5c68b630b8fabc11"
)
FROZEN_TRAIN_DATASET_V8 = driver_v7.FROZEN_TRAIN_DATASET_V7
FROZEN_EVAL_DATASET_V8 = driver_v7.FROZEN_EVAL_DATASET_V7
FROZEN_OUTPUT_DIRECTORY_V8 = driver_v7.FROZEN_OUTPUT_DIRECTORY_V7
V7_REPORT_PATH_V8 = (
    ROOT / "experiments/eggroll_es_hpo/S6_DIRECTION_STABILITY_V7_REPORT.json"
).resolve()
V7_REPORT_FILE_SHA256_V8 = (
    "051428abf536a865bd54c9cea93c9b04dc55dabbcc12b64aebd4da4baa581252"
)
V7_REPORT_CONTENT_SHA256_V8 = (
    "3ff4031fde91ac1725868fb02932ad5551ba41f2bb114fd1945ca5c00cb5e941"
)
EXPECTED_V7_EVIDENCE_BINDING_SHA256_V8 = (
    "499ddd0df46cc40228d315ad0b6ffeec6472cdfebacabc3ef8eab1aa943a489b"
)
EXPECTED_RECIPE_SHA256_V8 = {
    43: "7e8db9ef4b4ad7a913a438e3e5d1e423569c426c91a7904c3274e6bea75567b0",
    44: "08c1efe6dd4112e50140d21efa2a0649336312288356882a0fa5bd851b9cf0b4",
}
V8_RECIPE_KEYS = {
    *driver_v7.V7_RECIPE_KEYS - {
        "v6_smoke_gate", "v6_pilot_family",
        "coefficient_cosine_threshold",
    },
    "data_bootstrap_seed", "perturbation_basis_seed",
    "perturbation_seed_count", "perturbation_basis_sha256",
    "v7_family_evidence", "coefficient_cosine_threshold",
}
V8_IMPLEMENTATION_PATHS = {
    **driver_v7.V7_IMPLEMENTATION_PATHS,
    "anchor_trainer": Path(anchor_v8.__file__).resolve(),
    "distributed_driver_v8": Path(__file__).resolve(),
    "distributed_trainer_v8": Path(anchor_v8.__file__).resolve(),
    "distributed_worker_v8": ROOT / "eggroll_es_worker_v8.py",
    "stability_reporter_v8": (
        ROOT / "report_eggroll_es_direction_stability_v8.py"
    ),
    "protocol_v8": (
        ROOT / "experiments/eggroll_es_hpo/"
        "S6_SPLIT_SEED_POP32_V8_PROTOCOL.md"
    ),
    "contract_tests_v8": ROOT / "test_eggroll_es_split_seed_v8.py",
    "v7_family_report": V7_REPORT_PATH_V8,
}
_ACTIVE_LAYER_PLAN_BUNDLE = None
_ACTIVE_EXECUTION_V8 = None


if PERTURBATION_BASIS_SHA256_V8 != EXPECTED_PERTURBATION_BASIS_SHA256_V8:
    raise RuntimeError("v8 frozen perturbation basis changed")


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _v7_family_evidence_v8(path):
    path = Path(path).resolve()
    offline_audit._assert_no_heldout(str(path), "v8 v7-report path")
    if (
        path != V7_REPORT_PATH_V8
        or _file_sha256(path) != V7_REPORT_FILE_SHA256_V8
    ):
        raise ValueError("v8 requires the exact completed v7 family report")
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("v8 cannot read the v7 family report") from error
    if (
        report.get("content_sha256_before_self_field")
        != V7_REPORT_CONTENT_SHA256_V8
        or report.get("content_sha256_before_self_field")
        != driver_v1.canonical_sha256({
            key: value for key, value in report.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise ValueError("v8 v7 family report content hash changed")
    arms = report.get("arms") if isinstance(report, dict) else None
    if not isinstance(arms, list) or len(arms) != 2:
        raise ValueError("v8 v7 report does not contain both arms")
    journal_paths = [
        run.get("journal")
        for arm in arms if isinstance(arm, dict)
        for run in arm.get("runs", []) if isinstance(run, dict)
    ]
    recomputed = reporter_v7.build_report(journal_paths)
    if report != recomputed:
        raise ValueError("v8 v7 report differs from its four completed journals")
    expected_decisions = {
        "front": (0.06870231114244911, False),
        "middle_late": (0.5019011191439507, True),
    }
    observed = {
        arm["arm"]: (arm["seed_slot_coefficient_cosine"], arm["passed"])
        for arm in arms
    }
    if (
        report.get("schema")
        != "eggroll-es-direction-stability-report-v7"
        or report.get("coverage") != {
            "arms": ["front", "middle_late"],
            "seeds": [43, 44], "target_alphas": [0.0],
        }
        or report.get("preregistered_threshold") != 0.5
        or observed != expected_decisions
        or report.get("all_arms_passed") is not False
    ):
        raise ValueError("v8 v7 family decision evidence changed")
    runs = []
    for arm in arms:
        for run in arm["runs"]:
            runs.append({
                "arm": arm["arm"], "seed": run["seed"],
                "journal": run["journal"],
                "journal_file_sha256": run["journal_file_sha256"],
                "content_sha256": run["content_sha256"],
                "coefficient_sha256": run["coefficient_sha256"],
                "robust_plan_sha256": run["robust_plan_sha256"],
            })
    binding = {
        "schema": "eggroll-es-v7-family-evidence-binding-v8",
        "report_path": str(path),
        "report_file_sha256": V7_REPORT_FILE_SHA256_V8,
        "report_content_sha256": V7_REPORT_CONTENT_SHA256_V8,
        "runs": runs,
        "formal_v7_result": {
            "front_passed": False,
            "middle_late_passed": True,
            "all_arms_passed": False,
        },
        "v8_rationale": "marginal_v7_pass_requires_same_basis_diagnostic",
    }
    binding["binding_sha256"] = driver_v1.canonical_sha256(binding)
    if (
        binding["binding_sha256"]
        != EXPECTED_V7_EVIDENCE_BINDING_SHA256_V8
    ):
        raise ValueError("v8 v7 family evidence binding changed")
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
        raise RuntimeError("v8 anchor adapter API is incomplete")
    if module.WORKER_EXTENSION != (
        "eggroll_es_worker_v8.SplitSeedPopulation32AuditWorkerExtensionV8"
    ):
        raise RuntimeError("v8 selected the wrong distributed worker")
    return required


def _experiment_name_v8(data_seed):
    return (
        "snapshot794_layer_v8_middle_late_stability_"
        f"data{data_seed}_basis{PERTURBATION_BASIS_SEED_V8}"
    )


def frozen_recipe_v8(data_seed, v7_evidence):
    return {
        "model_name": str(
            (ROOT / "models/Qwen3.6-35B-A3B").resolve()
        ),
        "checkpoint": None,
        "train_dataset": str(FROZEN_TRAIN_DATASET_V8),
        "eval_dataset": str(FROZEN_EVAL_DATASET_V8),
        "sigma": 0.0003,
        "population_size": POPULATION_SIZE_V8,
        "batch_size": 64, "mini_batch_size": 64, "max_tokens": 32,
        "seed": data_seed,
        "engine_count": 4, "tp_per_engine": 1, "gpu_ids": [0, 1, 2, 3],
        "eval_splits": ["validation", "ood_qa"],
        "target_alphas": [0.0],
        "anchor_prose_jsonl": str(
            (ROOT / "data/general_prose_anchor_v1.jsonl").resolve()
        ),
        "anchor_prose_report": str(
            (ROOT / "data/general_prose_anchor_v1.report.json").resolve()
        ),
        "anchor_items_per_step": 128, "anchor_max_input_tokens": 512,
        "min_anchor_cosine": 0.8,
        "ood_prose_jsonl": str(
            (ROOT / "data/ood_prose_v3.jsonl").resolve()
        ),
        "ood_prose_max_input_tokens": 1024,
        "document_lcb_config_sha256": (
            anchor_v8.anchor_v7.anchor_v6.anchor_v5.DOCUMENT_LCB_CONFIG_SHA256_V5
        ),
        "reward_function_timeout": 10,
        "output_directory": str(FROZEN_OUTPUT_DIRECTORY_V8),
        "experiment_name": _experiment_name_v8(data_seed),
        "logging": "none", "wandb_project": "specialist-eggroll-es",
        "data_bootstrap_seed": data_seed,
        "perturbation_basis_seed": PERTURBATION_BASIS_SEED_V8,
        "perturbation_seed_count": POPULATION_SIZE_V8,
        "perturbation_basis_sha256": PERTURBATION_BASIS_SHA256_V8,
        "v7_family_evidence": copy.deepcopy(v7_evidence),
        "coefficient_cosine_threshold": COEFFICIENT_COSINE_THRESHOLD_V8,
    }


def validate_frozen_execution_cli_v8(argv, bundle):
    inherited = driver_v5.validate_frozen_execution_cli_v5(argv)
    stage_parser = argparse.ArgumentParser(add_help=False)
    stage_parser.add_argument("--v8-stage", choices=("stability",))
    stage_parser.add_argument("--v8-v7-family-report")
    stage_parser.add_argument("--v8-perturbation-basis-seed", type=int)
    stage_parser.add_argument("--v8-dry-run", action="store_true")
    stage, remaining = stage_parser.parse_known_args(list(argv))
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--target-alphas")
    parser.add_argument("--train-dataset", default=str(FROZEN_TRAIN_DATASET_V8))
    parser.add_argument("--eval-dataset", default=str(FROZEN_EVAL_DATASET_V8))
    parser.add_argument("--reward-function-timeout", type=int, default=10)
    parser.add_argument("--output-directory", default=str(FROZEN_OUTPUT_DIRECTORY_V8))
    parser.add_argument("--experiment-name")
    parser.add_argument("--logging", default="none")
    parser.add_argument("--wandb-project", default="specialist-eggroll-es")
    recipe, _ = parser.parse_known_args(list(argv))
    metadata = anchor_v8.validate_frozen_layer_plan_bundle_v8(bundle)
    if stage.v8_stage != "stability":
        raise ValueError("v8 requires --v8-stage stability")
    if recipe.seed not in DATA_BOOTSTRAP_SEEDS_V8:
        raise ValueError("v8 data/bootstrap seed must be exactly 43 or 44")
    if stage.v8_perturbation_basis_seed != PERTURBATION_BASIS_SEED_V8:
        raise ValueError("v8 perturbation basis seed changed")
    if recipe.target_alphas is None:
        raise ValueError("v8 requires explicit --target-alphas 0")
    targets = driver_v1.parse_target_alphas(recipe.target_alphas)
    if targets != STABILITY_TARGETS_V8:
        raise ValueError("v8 stability target must be exactly alpha zero")
    if (
        metadata["plan"] != "middle_late"
        or bundle["plan_sha256"] != MIDDLE_LATE_PLAN_SHA256_V8
        or recipe.population_size != POPULATION_SIZE_V8
        or recipe.batch_size != 64
        or Path(recipe.train_dataset).resolve() != FROZEN_TRAIN_DATASET_V8
        or Path(recipe.eval_dataset).resolve() != FROZEN_EVAL_DATASET_V8
        or recipe.reward_function_timeout != 10
        or Path(recipe.output_directory).resolve()
        != FROZEN_OUTPUT_DIRECTORY_V8
        or recipe.experiment_name != _experiment_name_v8(recipe.seed)
        or recipe.logging != "none"
        or recipe.wandb_project != "specialist-eggroll-es"
    ):
        raise ValueError("v8 split-seed runtime recipe changed")
    if stage.v8_v7_family_report is None:
        raise ValueError("v8 requires the completed v7 family report")
    evidence = _v7_family_evidence_v8(stage.v8_v7_family_report)
    frozen = frozen_recipe_v8(recipe.seed, evidence)
    if driver_v1.canonical_sha256(frozen) != EXPECTED_RECIPE_SHA256_V8[
        recipe.seed
    ]:
        raise RuntimeError("v8 frozen recipe hash changed")
    execution = {
        **inherited, **copy.deepcopy(frozen),
        "schema": "eggroll-es-split-seed-pop32-execution-v8",
        "stage": "stability", "arm": "middle_late",
        "dry_run": stage.v8_dry_run,
    }
    if {key: execution[key] for key in V8_RECIPE_KEYS} != frozen:
        raise RuntimeError("v8 effective recipe changed")
    return execution, remaining


def set_active_v8(bundle, execution):
    global _ACTIVE_LAYER_PLAN_BUNDLE, _ACTIVE_EXECUTION_V8
    metadata = anchor_v8.validate_frozen_layer_plan_bundle_v8(bundle)
    if execution.get("arm") != metadata["plan"]:
        raise ValueError("v8 execution and layer plan differ")
    driver_v5.set_active_layer_plan_bundle_v5(bundle)
    anchor_v8.set_default_layer_plan_bundle_v8(bundle)
    _ACTIVE_LAYER_PLAN_BUNDLE = bundle
    _ACTIVE_EXECUTION_V8 = execution


def build_snapshot(*args, **kwargs):
    bundle = _ACTIVE_LAYER_PLAN_BUNDLE
    execution = _ACTIVE_EXECUTION_V8
    metadata = anchor_v8.validate_frozen_layer_plan_bundle_v8(bundle)
    if not isinstance(execution, dict):
        raise RuntimeError("v8 has no active execution")
    snapshot = driver_v5.build_snapshot(*args, **kwargs)
    implementation = {
        key: anchor_v8.file_sha256(path)
        for key, path in V8_IMPLEMENTATION_PATHS.items()
    }
    snapshot["implementation"].update(implementation)
    snapshot["schema"] = "eggroll-es-anchor-line-search-snapshot-v8"
    snapshot["split_seed_stability_v8"] = {
        "schema": "eggroll-es-split-seed-pop32-snapshot-v8",
        "family": "middle_late_same_basis_cross_data_seed",
        "stage": "stability", "arm": "middle_late",
        "layers": metadata["layers"],
        "data_bootstrap_seed_pair": [43, 44],
        "data_bootstrap_seed": execution["data_bootstrap_seed"],
        "perturbation_basis_seed": PERTURBATION_BASIS_SEED_V8,
        "perturbation_seed_count": POPULATION_SIZE_V8,
        "perturbation_seeds": list(PERTURBATION_SEEDS_V8),
        "perturbation_basis_sha256": PERTURBATION_BASIS_SHA256_V8,
        "target_alphas": [0.0],
        "benchmark_treatment_applied": False,
        "selection_surface": "same_basis_coefficients_only",
        "coefficient_cosine_threshold": COEFFICIENT_COSINE_THRESHOLD_V8,
        "recipe": {
            key: copy.deepcopy(execution[key]) for key in V8_RECIPE_KEYS
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


def _v5_compatibility_journal_v8(journal):
    compatible = copy.deepcopy(journal)
    compatible["schema"] = "eggroll-es-anchor-alpha-line-search-v5"
    compatible["snapshot"]["schema"] = (
        "eggroll-es-anchor-line-search-snapshot-v5"
    )
    compatible["snapshot"].pop("split_seed_stability_v8", None)
    for key in (
        "split_seed_stability_family_v8", "stage_v8",
        "target_alpha_zero_only_v8", "benchmark_selection_forbidden_v8",
        "same_perturbation_basis_required_v8",
        "cross_data_seed_coefficient_cosine_threshold_v8",
        "requires_complete_v7_family_v8",
    ):
        compatible["policy"].pop(key, None)
    compatible.pop("content_sha256_before_self_field", None)
    compatible["content_sha256_before_self_field"] = (
        driver_v1.canonical_sha256(compatible)
    )
    return compatible


def _validate_inherited_zero_target_v8(journal):
    compatible_v5 = _v5_compatibility_journal_v8(journal)
    plan = driver_v5._journal_plan_v5(compatible_v5)
    robust = anchor_v8.validate_robust_plan_v8(plan, recompute_numeric=True)
    snapshot_v5 = compatible_v5["snapshot"]
    robust_snapshot = snapshot_v5.get("document_lcb_anchor_v5")
    implementation = snapshot_v5.get("implementation")
    implementation_v5 = {
        key: implementation.get(key) for key in driver_v5.V5_IMPLEMENTATION_KEYS
    }
    if robust_snapshot != {
        "config": driver_v5.robust_anchor.document_lcb_config(),
        "config_sha256": driver_v5.robust_anchor.DOCUMENT_LCB_CONFIG_SHA256,
        "objective_source": "frozen_train_only_anchor_prose",
        "ood_validation_heldout_as_objective": False,
        "implementation_bundle_sha256": driver_v1.canonical_sha256(
            implementation_v5
        ),
    }:
        raise RuntimeError("v8 inherited v5 robust snapshot changed")
    compatible_v4 = driver_v5._v4_compatibility_journal_v5(compatible_v5)
    offline_audit._assert_no_heldout(compatible_v4)
    offline_audit._verify_content_hash(compatible_v4)
    legacy_snapshot = copy.deepcopy(compatible_v4["snapshot"])
    if legacy_snapshot.get("recipe", {}).get("target_alphas") != [0.0]:
        raise RuntimeError("v8 inherited snapshot is not alpha-zero only")
    legacy_snapshot["recipe"]["target_alphas"] = [0.0, 1.0]
    normalized = offline_audit._validate_snapshot(legacy_snapshot)
    normalized["targets"] = [0.0]
    normalized["recipe"]["target_alphas"] = [0.0]
    configuration = compatible_v4.get("trainer_configuration")
    for config_key, recipe_key in {
        "model_name": "model_name", "sigma": "sigma",
        "population_size": "population_size", "batch_size": "batch_size",
        "mini_batch_size": "mini_batch_size", "max_tokens": "max_tokens",
        "global_seed": "seed", "min_anchor_cosine": "min_anchor_cosine",
        "anchor_items_per_step": "anchor_items_per_step",
    }.items():
        if configuration.get(config_key) != normalized["recipe"].get(recipe_key):
            raise RuntimeError("v8 inherited trainer configuration changed")
    coefficient_plan = compatible_v4.get("coefficient_plan")
    coefficient_sha = coefficient_plan.get("coefficient_sha256")
    seeds = compatible_v4.get("seeds")
    if seeds != PERTURBATION_SEEDS_V8:
        raise RuntimeError("v8 journal differs from its fixed perturbation basis")
    distributed = offline_audit._validate_v4_distributed_provenance(
        coefficient_plan, coefficient_sha=coefficient_sha,
        targets=[0.0], snapshot=normalized, journal_seeds=seeds,
    )
    states = compatible_v4.get("states")
    if not isinstance(states, list) or len(states) != 1:
        raise RuntimeError("v8 must contain exactly one alpha-zero state")
    state = states[0]
    if (
        not isinstance(state, dict)
        or set(state) != {
            "state_index", "eval_iteration", "target_alpha",
            "alpha_increment", "coefficient_sha256", "qa",
            "ood_qa_gate", "ood_prose", "ood_prose_gate",
            "strict_guards_passed",
        }
        or state["state_index"] != 0 or state["eval_iteration"] != 0
        or float(state["target_alpha"]) != 0.0
        or float(state["alpha_increment"]) != 0.0
        or state["coefficient_sha256"] != coefficient_sha
    ):
        raise RuntimeError("v8 alpha-zero state identity changed")
    qa = {
        split: offline_audit._validate_qa_summary(
            state["qa"][split], f"v8 state 0 {split}",
        )
        for split in ("validation", "ood_qa")
    }
    prose = offline_audit._validate_prose_summary(
        state["ood_prose"], "v8 state 0",
    )
    for split in ("validation", "ood_qa"):
        expected = offline_audit.V4_FROZEN_S6_BASELINE[split]
        actual = qa[split]
        if (
            actual["rows"] != expected["rows"]
            or actual["exact"] != expected["exact"]
            or actual["nonzero"] != expected["nonzero"]
            or actual["mean_reward"] != expected["mean_reward"]
        ):
            raise RuntimeError("v8 alpha-zero S6 QA identity changed")
    expected_prose = offline_audit.V4_FROZEN_S6_BASELINE["ood_prose"]
    if (
        prose["item_count"] != expected_prose["item_count"]
        or prose["scored_token_count"] != expected_prose["scored_token_count"]
        or prose["mean_token_logprob"] != expected_prose["mean_token_logprob"]
    ):
        raise RuntimeError("v8 alpha-zero prose identity changed")
    qa_gate = offline_audit._validate_qa_gate(
        state["ood_qa_gate"], qa["ood_qa"], qa["ood_qa"], "v8 state 0",
    )
    prose_gate = offline_audit._validate_prose_gate(
        state["ood_prose_gate"], prose, prose, "v8 state 0",
    )
    if state["strict_guards_passed"] is not (
        qa_gate["passed"] and prose_gate["passed"]
    ):
        raise RuntimeError("v8 alpha-zero strict gate changed")
    return {
        "data_bootstrap_seed": normalized["seed"], "state_count": 1,
        "coefficient_sha256": coefficient_sha,
        "robust_plan_sha256": robust["robust_plan_sha256"],
        "distributed_update_v4": distributed,
    }


def validate_completed_journal_v8(journal):
    if (
        not isinstance(journal, dict)
        or set(journal) != {
            "schema", "status", "in_progress", "policy", "targets",
            "trainer_configuration", "snapshot", "coefficient_plan",
            "states", "seeds", "content_sha256_before_self_field",
        }
        or journal.get("schema")
        != "eggroll-es-anchor-alpha-line-search-v8"
        or journal.get("status") != "complete"
        or journal.get("in_progress") is not None
        or journal.get("targets") != [0.0]
        or journal.get("seeds") != PERTURBATION_SEEDS_V8
    ):
        raise RuntimeError("v8 journal is incomplete or changed basis/target")
    if journal.get("content_sha256_before_self_field") != (
        driver_v1.canonical_sha256({
            key: value for key, value in journal.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v8 journal content hash changed")
    policy = journal.get("policy")
    inherited_policy = {
        "alpha_order": "zero_then_strictly_increasing",
        "branching": False, "resume": False, "rollback": False,
        "selection_during_execution": False,
        "ood_qa_max_degradation": 0.0, "ood_prose_max_degradation": 0.0,
        "bf16_alpha_semantics": "path_dependent_monotonic_pilot",
        "direct_alpha_confirmation_required": True,
        "frozen_layer_plan_required": True, "dense_gold_reward_required": True,
        "document_lcb_anchor_required": True,
        "optimization_data": "train_and_anchor_only",
        "ood_validation_heldout_as_objective": False,
    }
    v8_policy = {
        "split_seed_stability_family_v8": "middle_late_pop32_same_basis",
        "stage_v8": "stability", "target_alpha_zero_only_v8": True,
        "benchmark_selection_forbidden_v8": True,
        "same_perturbation_basis_required_v8": True,
        "cross_data_seed_coefficient_cosine_threshold_v8": 0.5,
        "requires_complete_v7_family_v8": True,
    }
    if (
        not isinstance(policy, dict)
        or set(policy) != set(inherited_policy) | set(v8_policy)
        or any(policy.get(key) != value for key, value in {
            **inherited_policy, **v8_policy,
        }.items())
    ):
        raise RuntimeError("v8 split-seed policy changed")
    snapshot = journal.get("snapshot")
    stability = (
        snapshot.get("split_seed_stability_v8")
        if isinstance(snapshot, dict) else None
    )
    expected_stability_keys = {
        "schema", "family", "stage", "arm", "layers",
        "data_bootstrap_seed_pair", "data_bootstrap_seed",
        "perturbation_basis_seed", "perturbation_seed_count",
        "perturbation_seeds", "perturbation_basis_sha256",
        "target_alphas", "benchmark_treatment_applied",
        "selection_surface", "coefficient_cosine_threshold", "recipe",
        "plan_sha256", "plan_file_sha256", "source_unit_count",
        "runtime_selected_parameter_count", "selected_element_count",
        "selected_byte_count", "implementation_bundle_sha256",
    }
    if (
        snapshot.get("schema")
        != "eggroll-es-anchor-line-search-snapshot-v8"
        or set(snapshot) != {
            "schema", "train", "evaluations", "anchor",
            "fixed_train_batch", "implementation", "recipe",
            "distributed_update_v3", "frozen_layer_plan_v4",
            "dense_gold_reward_v4", "distributed_update_v4",
            "document_lcb_anchor_v5", "split_seed_stability_v8",
        }
        or not isinstance(stability, dict)
        or set(stability) != expected_stability_keys
        or stability.get("schema")
        != "eggroll-es-split-seed-pop32-snapshot-v8"
        or stability.get("family")
        != "middle_late_same_basis_cross_data_seed"
        or stability.get("stage") != "stability"
        or stability.get("arm") != "middle_late"
        or stability.get("layers") != [20, 21, 22, 23]
        or stability.get("data_bootstrap_seed_pair") != [43, 44]
        or stability.get("data_bootstrap_seed")
        not in DATA_BOOTSTRAP_SEEDS_V8
        or stability.get("perturbation_basis_seed")
        != PERTURBATION_BASIS_SEED_V8
        or stability.get("perturbation_seed_count") != POPULATION_SIZE_V8
        or stability.get("perturbation_seeds") != PERTURBATION_SEEDS_V8
        or stability.get("perturbation_basis_sha256")
        != PERTURBATION_BASIS_SHA256_V8
        or stability.get("target_alphas") != [0.0]
        or stability.get("benchmark_treatment_applied") is not False
        or stability.get("selection_surface")
        != "same_basis_coefficients_only"
        or stability.get("coefficient_cosine_threshold") != 0.5
        or stability.get("plan_sha256") != MIDDLE_LATE_PLAN_SHA256_V8
        or stability.get("source_unit_count") != 35
        or stability.get("runtime_selected_parameter_count") != 23
        or stability.get("selected_element_count") != 142_999_552
        or stability.get("selected_byte_count") != 285_999_104
    ):
        raise RuntimeError("v8 split-seed snapshot changed")
    evidence = _v7_family_evidence_v8(V7_REPORT_PATH_V8)
    expected_recipe = frozen_recipe_v8(
        stability["data_bootstrap_seed"], evidence,
    )
    if stability.get("recipe") != expected_recipe:
        raise RuntimeError("v8 persisted recipe changed")
    if driver_v1.canonical_sha256(expected_recipe) != EXPECTED_RECIPE_SHA256_V8[
        stability["data_bootstrap_seed"]
    ]:
        raise RuntimeError("v8 persisted recipe hash changed")
    expected_snapshot_recipe = {
        key: expected_recipe[key] for key in (
            "model_name", "checkpoint", "sigma", "population_size",
            "batch_size", "mini_batch_size", "max_tokens", "seed",
            "min_anchor_cosine", "anchor_items_per_step", "target_alphas",
        )
    }
    if snapshot.get("recipe") != expected_snapshot_recipe:
        raise RuntimeError("v8 effective snapshot recipe changed")
    spec = anchor_v8.FROZEN_STABILITY_PLANS_V8[
        MIDDLE_LATE_PLAN_SHA256_V8
    ]
    if stability.get("plan_file_sha256") != spec["file_sha256"]:
        raise RuntimeError("v8 persisted layer-plan file changed")
    actual_implementation = {
        key: anchor_v8.file_sha256(path)
        for key, path in V8_IMPLEMENTATION_PATHS.items()
    }
    implementation = snapshot.get("implementation")
    if (
        not isinstance(implementation, dict)
        or set(implementation) != set(V8_IMPLEMENTATION_PATHS)
        or {key: implementation.get(key) for key in V8_IMPLEMENTATION_PATHS}
        != actual_implementation
        or stability.get("implementation_bundle_sha256")
        != driver_v1.canonical_sha256(actual_implementation)
    ):
        raise RuntimeError("v8 implementation identity changed")
    with driver_v6.scoped_legacy_audit_v6():
        inherited = _validate_inherited_zero_target_v8(journal)
    if inherited["data_bootstrap_seed"] != stability["data_bootstrap_seed"]:
        raise RuntimeError("v8 inherited data/bootstrap seed changed")
    return {
        **inherited, "arm": "middle_late", "stage": "stability",
        "perturbation_basis_sha256": PERTURBATION_BASIS_SHA256_V8,
        "content_sha256": journal["content_sha256_before_self_field"],
    }


def execute_line_search(*args, **kwargs):
    kwargs = dict(kwargs)
    execution = _ACTIVE_EXECUTION_V8
    expected_inherited_seeds = np.random.default_rng(
        seed=execution["data_bootstrap_seed"]
    ).integers(
        0, 2**30, size=POPULATION_SIZE_V8, dtype=np.int64,
    ).tolist()
    if kwargs.get("seeds") != expected_inherited_seeds:
        raise RuntimeError("v8 inherited driver population seed schedule changed")
    kwargs["seeds"] = list(PERTURBATION_SEEDS_V8)
    with driver_v6.scoped_legacy_audit_v6():
        journal = driver_v4.execute_line_search(*args, **kwargs)
    trainer = args[0] if args else kwargs["trainer"]
    plan = trainer._latest_anchor_plan
    anchor_v8.validate_robust_plan_v8(plan, recompute_numeric=True)
    binding_v5 = anchor_v8.anchor_v7.anchor_v6.anchor_v5.validate_robust_plan_v5(
        plan, recompute_numeric=True,
    )
    journal["schema"] = "eggroll-es-anchor-alpha-line-search-v8"
    journal["snapshot"]["schema"] = (
        "eggroll-es-anchor-line-search-snapshot-v8"
    )
    journal["policy"].update({
        "document_lcb_anchor_required": True,
        "optimization_data": "train_and_anchor_only",
        "ood_validation_heldout_as_objective": False,
        "split_seed_stability_family_v8": "middle_late_pop32_same_basis",
        "stage_v8": "stability", "target_alpha_zero_only_v8": True,
        "benchmark_selection_forbidden_v8": True,
        "same_perturbation_basis_required_v8": True,
        "cross_data_seed_coefficient_cosine_threshold_v8": 0.5,
        "requires_complete_v7_family_v8": True,
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
        validate_completed_journal_v8(journal)
    except Exception as error:
        journal["status"] = "failed"
        journal["failure"] = {
            "type": type(error).__name__, "message": str(error),
            "phase": "validating_complete_v8_release_audit",
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
    execution, remaining = validate_frozen_execution_cli_v8(remaining, bundle)
    set_active_v8(bundle, execution)
    if execution["dry_run"]:
        payload = {
            "schema": "eggroll-es-split-seed-pop32-dry-run-v8",
            "arm": "middle_late", "stage": "stability",
            "data_bootstrap_seed": execution["data_bootstrap_seed"],
            "perturbation_basis_seed": PERTURBATION_BASIS_SEED_V8,
            "perturbation_basis_sha256": PERTURBATION_BASIS_SHA256_V8,
            "population_size": POPULATION_SIZE_V8, "targets": [0.0],
            "recipe_sha256": driver_v1.canonical_sha256({
                key: execution[key] for key in V8_RECIPE_KEYS
            }),
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
