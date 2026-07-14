#!/usr/bin/env python3
"""Fail-closed alpha-zero direction-stability protocol for S6 EGGROLL-ES."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

import aggregate_eggroll_es_anchor_replications as offline_audit
import run_eggroll_es_anchor_line_search as driver_v1
import run_eggroll_es_anchor_line_search_v4 as driver_v4
import run_eggroll_es_anchor_line_search_v5 as driver_v5
import run_eggroll_es_anchor_line_search_v6 as driver_v6
import train_eggroll_es_specialist_anchor_v7 as anchor_v7


ROOT = Path(__file__).resolve().parent
STABILITY_TARGETS_V7 = [0.0]
STABILITY_SEEDS_V7 = (43, 44)
STABILITY_ARMS_V7 = ("front", "middle_late")
COEFFICIENT_COSINE_THRESHOLD_V7 = 0.5
FROZEN_TRAIN_DATASET_V7 = driver_v6.FROZEN_TRAIN_DATASET_V6
FROZEN_EVAL_DATASET_V7 = driver_v6.FROZEN_EVAL_DATASET_V6
FROZEN_OUTPUT_DIRECTORY_V7 = driver_v6.FROZEN_OUTPUT_DIRECTORY_V6
SMOKE_GATE_PATH_V7 = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "snapshot794_layer_v6_smoke_gate_seed42.json"
).resolve()
SMOKE_GATE_FILE_SHA256_V7 = (
    "9669b75fc7e70965f8552ae15ffd6c471a2555eb855a831839d5334cb0474ea3"
)
SMOKE_GATE_BINDING_SHA256_V7 = (
    "3bdc4723fbfe069614138005afc42875ed0f0754cdea2197bd13f1e2d31fd6ee"
)
PILOT_EVIDENCE_PATH_V7 = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_DIRECTION_STABILITY_V7_PILOT_EVIDENCE.json"
).resolve()
PILOT_EVIDENCE_FILE_SHA256_V7 = (
    "af241c492b50b454119b5a5293f6e3ee8c936ad9735a22d3d7a44b5c980f7509"
)
V7_RECIPE_KEYS = {
    *driver_v6.V6_RECIPE_KEYS - {"smoke_gate"},
    "v6_smoke_gate", "v6_pilot_family",
    "coefficient_cosine_threshold",
}
V7_IMPLEMENTATION_PATHS = {
    **driver_v6.V6_IMPLEMENTATION_PATHS,
    "anchor_trainer": Path(anchor_v7.__file__).resolve(),
    "distributed_driver_v7": Path(__file__).resolve(),
    "distributed_trainer_v7": Path(anchor_v7.__file__).resolve(),
    "distributed_worker_v7": ROOT / "eggroll_es_worker_v7.py",
    "pilot_family_evidence_v7": PILOT_EVIDENCE_PATH_V7,
}
_ACTIVE_LAYER_PLAN_BUNDLE = None
_ACTIVE_EXECUTION_V7 = None


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _clean_smoke_evidence_v7(path):
    path = Path(path).resolve()
    if (
        path != SMOKE_GATE_PATH_V7
        or _file_sha256(path) != SMOKE_GATE_FILE_SHA256_V7
    ):
        raise ValueError("v7 requires the exact clean v6 four-smoke gate")
    binding = driver_v6._smoke_evidence_v6(path)
    if binding.get("binding_sha256") != SMOKE_GATE_BINDING_SHA256_V7:
        raise ValueError("v7 v6 smoke-gate binding changed")
    return binding


def _pilot_family_evidence_v7(path):
    path = Path(path).resolve()
    offline_audit._assert_no_heldout(str(path), "v7 pilot-evidence path")
    if (
        path != PILOT_EVIDENCE_PATH_V7
        or _file_sha256(path) != PILOT_EVIDENCE_FILE_SHA256_V7
    ):
        raise ValueError("v7 requires the exact four-pilot evidence file")
    evidence = json.loads(path.read_text())
    if (
        not isinstance(evidence, dict)
        or set(evidence) != {
            "schema", "smoke_gate", "pilots",
            "content_sha256_before_self_field",
        }
        or evidence.get("schema")
        != "eggroll-es-v6-four-pilot-family-evidence-v7"
        or evidence.get("content_sha256_before_self_field")
        != driver_v1.canonical_sha256({
            key: value for key, value in evidence.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise ValueError("v7 four-pilot evidence schema or hash changed")
    smoke = evidence.get("smoke_gate")
    if smoke != {
        "path": str(SMOKE_GATE_PATH_V7),
        "file_sha256": SMOKE_GATE_FILE_SHA256_V7,
        "binding_sha256": SMOKE_GATE_BINDING_SHA256_V7,
    }:
        raise ValueError("v7 pilot evidence names a different smoke gate")
    pilots = evidence.get("pilots")
    if not isinstance(pilots, list) or len(pilots) != 4:
        raise ValueError("v7 pilot evidence must contain four pilots")
    summaries = []
    for entry in pilots:
        if not isinstance(entry, dict) or set(entry) != {
            "arm", "journal", "file_sha256", "content_sha256",
            "coefficient_sha256", "robust_plan_sha256",
        }:
            raise ValueError("v7 pilot evidence entry fields changed")
        journal_path = Path(str(entry["journal"])).resolve()
        offline_audit._assert_no_heldout(
            str(journal_path), "v7 pilot journal path",
        )
        if _file_sha256(journal_path) != entry["file_sha256"]:
            raise ValueError("v7 pilot journal file hash changed")
        audit = driver_v6.validate_completed_journal_v6(
            json.loads(journal_path.read_text())
        )
        if (
            audit["arm"] != entry["arm"]
            or audit["stage"] != "pilot"
            or audit["seed"] != 42
            or audit["state_count"] != len(driver_v6.PILOT_TARGETS_V6)
            or any(audit[key] != entry[key] for key in (
                "content_sha256", "coefficient_sha256",
                "robust_plan_sha256",
            ))
        ):
            raise ValueError("v7 pilot evidence differs from completed journal")
        summaries.append(copy.deepcopy(entry))
    if [row["arm"] for row in summaries] != [
        "back", "front", "middle_early", "middle_late",
    ]:
        raise ValueError("v7 pilot evidence arm order or coverage changed")
    binding = {
        "schema": "eggroll-es-v6-four-pilot-family-binding-v7",
        "smoke_gate_binding_sha256": SMOKE_GATE_BINDING_SHA256_V7,
        "pilots": summaries,
        "evidence_file_sha256": PILOT_EVIDENCE_FILE_SHA256_V7,
    }
    binding["binding_sha256"] = driver_v1.canonical_sha256(binding)
    return binding


def validate_effective_anchor_api(module=anchor_v7):
    required = (
        "coefficient_sha256", "load_anchor_prose", "load_trainer",
        "load_frozen_layer_plan_v7", "validate_robust_plan_v7",
    )
    missing = [
        name for name in required if not callable(getattr(module, name, None))
    ]
    if missing:
        raise RuntimeError("v7 anchor adapter API is incomplete")
    if module.WORKER_EXTENSION != (
        "eggroll_es_worker_v7.DirectionStabilityAuditWorkerExtensionV7"
    ):
        raise RuntimeError("v7 selected the wrong distributed worker")
    return required


def _experiment_name_v7(arm, seed):
    return f"snapshot794_layer_v7_{arm}_stability_seed{seed}"


def frozen_recipe_v7(arm, seed, smoke_gate, pilot_family):
    return {
        "model_name": str(
            (ROOT / "models/Qwen3.6-35B-A3B").resolve()
        ),
        "checkpoint": None,
        "train_dataset": str(FROZEN_TRAIN_DATASET_V7),
        "eval_dataset": str(FROZEN_EVAL_DATASET_V7),
        "sigma": 0.0003,
        "population_size": 16,
        "batch_size": 64,
        "mini_batch_size": 64,
        "max_tokens": 32,
        "seed": seed,
        "engine_count": 4,
        "tp_per_engine": 1,
        "gpu_ids": [0, 1, 2, 3],
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
        "ood_prose_jsonl": str(
            (ROOT / "data/ood_prose_v3.jsonl").resolve()
        ),
        "ood_prose_max_input_tokens": 1024,
        "document_lcb_config_sha256": (
            anchor_v7.anchor_v6.anchor_v5.DOCUMENT_LCB_CONFIG_SHA256_V5
        ),
        "reward_function_timeout": 10,
        "output_directory": str(FROZEN_OUTPUT_DIRECTORY_V7),
        "experiment_name": _experiment_name_v7(arm, seed),
        "logging": "none",
        "wandb_project": "specialist-eggroll-es",
        "v6_smoke_gate": copy.deepcopy(smoke_gate),
        "v6_pilot_family": copy.deepcopy(pilot_family),
        "coefficient_cosine_threshold": COEFFICIENT_COSINE_THRESHOLD_V7,
    }


def validate_frozen_execution_cli_v7(argv, bundle):
    inherited = driver_v5.validate_frozen_execution_cli_v5(argv)
    stage_parser = argparse.ArgumentParser(add_help=False)
    stage_parser.add_argument("--v7-stage", choices=("stability",))
    stage_parser.add_argument("--v7-smoke-gate-json")
    stage_parser.add_argument("--v7-pilot-family-json")
    stage_parser.add_argument("--v7-dry-run", action="store_true")
    stage, remaining = stage_parser.parse_known_args(list(argv))
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--target-alphas")
    parser.add_argument("--train-dataset", default=str(FROZEN_TRAIN_DATASET_V7))
    parser.add_argument("--eval-dataset", default=str(FROZEN_EVAL_DATASET_V7))
    parser.add_argument("--reward-function-timeout", type=int, default=10)
    parser.add_argument("--output-directory", default=str(FROZEN_OUTPUT_DIRECTORY_V7))
    parser.add_argument("--experiment-name")
    parser.add_argument("--logging", default="none")
    parser.add_argument("--wandb-project", default="specialist-eggroll-es")
    recipe, _ = parser.parse_known_args(list(argv))
    metadata = anchor_v7.validate_frozen_layer_plan_bundle_v7(bundle)
    if stage.v7_stage != "stability":
        raise ValueError("v7 requires --v7-stage stability")
    if recipe.seed not in STABILITY_SEEDS_V7:
        raise ValueError("v7 stability seed must be exactly 43 or 44")
    if recipe.target_alphas is None:
        raise ValueError("v7 requires explicit --target-alphas 0")
    targets = driver_v1.parse_target_alphas(recipe.target_alphas)
    if targets != STABILITY_TARGETS_V7:
        raise ValueError("v7 stability target must be exactly alpha zero")
    if (
        metadata["plan"] not in STABILITY_ARMS_V7
        or recipe.population_size != 16
        or recipe.batch_size != 64
        or Path(recipe.train_dataset).resolve() != FROZEN_TRAIN_DATASET_V7
        or Path(recipe.eval_dataset).resolve() != FROZEN_EVAL_DATASET_V7
        or recipe.reward_function_timeout != 10
        or Path(recipe.output_directory).resolve()
        != FROZEN_OUTPUT_DIRECTORY_V7
        or recipe.experiment_name
        != _experiment_name_v7(metadata["plan"], recipe.seed)
        or recipe.logging != "none"
        or recipe.wandb_project != "specialist-eggroll-es"
    ):
        raise ValueError("v7 stability runtime recipe changed")
    if stage.v7_smoke_gate_json is None or stage.v7_pilot_family_json is None:
        raise ValueError("v7 requires v6 smoke and four-pilot evidence")
    smoke = _clean_smoke_evidence_v7(stage.v7_smoke_gate_json)
    pilots = _pilot_family_evidence_v7(stage.v7_pilot_family_json)
    frozen = frozen_recipe_v7(metadata["plan"], recipe.seed, smoke, pilots)
    execution = {
        **inherited,
        **copy.deepcopy(frozen),
        "schema": "eggroll-es-direction-stability-execution-v7",
        "stage": "stability",
        "arm": metadata["plan"],
        "dry_run": stage.v7_dry_run,
    }
    if {key: execution[key] for key in V7_RECIPE_KEYS} != frozen:
        raise RuntimeError("v7 effective recipe changed")
    return execution, remaining


def set_active_v7(bundle, execution):
    global _ACTIVE_LAYER_PLAN_BUNDLE, _ACTIVE_EXECUTION_V7
    anchor_v7.validate_frozen_layer_plan_bundle_v7(bundle)
    if execution.get("arm") != bundle["edge_split_v6"]["plan"]:
        raise ValueError("v7 execution and layer plan differ")
    driver_v5.set_active_layer_plan_bundle_v5(bundle)
    anchor_v7.set_default_layer_plan_bundle_v7(bundle)
    _ACTIVE_LAYER_PLAN_BUNDLE = bundle
    _ACTIVE_EXECUTION_V7 = execution


def build_snapshot(*args, **kwargs):
    bundle = _ACTIVE_LAYER_PLAN_BUNDLE
    execution = _ACTIVE_EXECUTION_V7
    metadata = anchor_v7.validate_frozen_layer_plan_bundle_v7(bundle)
    if not isinstance(execution, dict):
        raise RuntimeError("v7 has no active execution")
    snapshot = driver_v5.build_snapshot(*args, **kwargs)
    implementation = {
        key: anchor_v7.file_sha256(path)
        for key, path in V7_IMPLEMENTATION_PATHS.items()
    }
    snapshot["implementation"].update(implementation)
    snapshot["schema"] = "eggroll-es-anchor-line-search-snapshot-v7"
    snapshot["direction_stability_v7"] = {
        "schema": "eggroll-es-direction-stability-snapshot-v7",
        "family": "front_middle_late_cross_seed",
        "stage": "stability",
        "arm": metadata["plan"],
        "layers": metadata["layers"],
        "seed_pair": [43, 44],
        "seed": execution["seed"],
        "target_alphas": [0.0],
        "benchmark_treatment_applied": False,
        "selection_surface": "coefficient_direction_only",
        "coefficient_cosine_threshold": COEFFICIENT_COSINE_THRESHOLD_V7,
        "recipe": {
            key: copy.deepcopy(execution[key]) for key in V7_RECIPE_KEYS
        },
        "plan_sha256": bundle["plan_sha256"],
        "plan_file_sha256": bundle["file_sha256"],
        "implementation_bundle_sha256": driver_v1.canonical_sha256(
            implementation
        ),
    }
    return snapshot


def _v5_compatibility_journal_v7(journal):
    compatible = copy.deepcopy(journal)
    compatible["schema"] = "eggroll-es-anchor-alpha-line-search-v5"
    compatible["snapshot"]["schema"] = (
        "eggroll-es-anchor-line-search-snapshot-v5"
    )
    compatible["snapshot"].pop("direction_stability_v7", None)
    for key in (
        "direction_stability_family_v7", "stage_v7",
        "target_alpha_zero_only_v7", "benchmark_selection_forbidden_v7",
        "cross_seed_coefficient_cosine_threshold_v7",
        "requires_clean_v6_family_v7",
    ):
        compatible["policy"].pop(key, None)
    compatible.pop("content_sha256_before_self_field", None)
    compatible["content_sha256_before_self_field"] = (
        driver_v1.canonical_sha256(compatible)
    )
    return compatible


def _validate_inherited_zero_target_v7(journal):
    """Run inherited v4/v5 component audits without fabricating a treatment."""
    compatible_v5 = _v5_compatibility_journal_v7(journal)
    plan = driver_v5._journal_plan_v5(compatible_v5)
    robust = anchor_v7.validate_robust_plan_v7(plan, recompute_numeric=True)
    snapshot_v5 = compatible_v5["snapshot"]
    robust_snapshot = snapshot_v5.get("document_lcb_anchor_v5")
    implementation = snapshot_v5.get("implementation")
    implementation_v5 = {
        key: implementation.get(key) for key in driver_v5.V5_IMPLEMENTATION_KEYS
    }
    if (
        robust_snapshot != {
            "config": driver_v5.robust_anchor.document_lcb_config(),
            "config_sha256": driver_v5.robust_anchor.DOCUMENT_LCB_CONFIG_SHA256,
            "objective_source": "frozen_train_only_anchor_prose",
            "ood_validation_heldout_as_objective": False,
            "implementation_bundle_sha256": driver_v1.canonical_sha256(
                implementation_v5
            ),
        }
    ):
        raise RuntimeError("v7 inherited v5 robust snapshot changed")
    compatible_v4 = driver_v5._v4_compatibility_journal_v5(compatible_v5)
    offline_audit._assert_no_heldout(compatible_v4)
    offline_audit._verify_content_hash(compatible_v4)
    legacy_snapshot = copy.deepcopy(compatible_v4["snapshot"])
    if legacy_snapshot.get("recipe", {}).get("target_alphas") != [0.0]:
        raise RuntimeError("v7 inherited snapshot is not alpha-zero only")
    # The legacy snapshot parser predates direction-only runs and requires two
    # targets.  Adapt only that parser input; the actual journal/state/update
    # audit below remains the exact one-target artifact with zero applications.
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
            raise RuntimeError("v7 inherited trainer configuration changed")
    coefficient_plan = compatible_v4.get("coefficient_plan")
    coefficient_sha = coefficient_plan.get("coefficient_sha256")
    seeds = compatible_v4.get("seeds")
    expected_seeds = np.random.default_rng(
        seed=normalized["seed"]
    ).integers(0, 2**30, size=16, dtype=np.int64).tolist()
    if seeds != expected_seeds:
        raise RuntimeError("v7 population seeds differ from the global seed")
    distributed = offline_audit._validate_v4_distributed_provenance(
        coefficient_plan, coefficient_sha=coefficient_sha,
        targets=[0.0], snapshot=normalized, journal_seeds=seeds,
    )
    states = compatible_v4.get("states")
    if not isinstance(states, list) or len(states) != 1:
        raise RuntimeError("v7 must contain exactly one alpha-zero state")
    state = states[0]
    if (
        not isinstance(state, dict)
        or set(state) != {
            "state_index", "eval_iteration", "target_alpha",
            "alpha_increment", "coefficient_sha256", "qa",
            "ood_qa_gate", "ood_prose", "ood_prose_gate",
            "strict_guards_passed",
        }
        or state["state_index"] != 0
        or state["eval_iteration"] != 0
        or float(state["target_alpha"]) != 0.0
        or float(state["alpha_increment"]) != 0.0
        or state["coefficient_sha256"] != coefficient_sha
    ):
        raise RuntimeError("v7 alpha-zero state identity changed")
    qa = {
        split: offline_audit._validate_qa_summary(
            state["qa"][split], f"v7 state 0 {split}",
        )
        for split in ("validation", "ood_qa")
    }
    prose = offline_audit._validate_prose_summary(
        state["ood_prose"], "v7 state 0",
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
            raise RuntimeError("v7 alpha-zero S6 QA identity changed")
    expected_prose = offline_audit.V4_FROZEN_S6_BASELINE["ood_prose"]
    if (
        prose["item_count"] != expected_prose["item_count"]
        or prose["scored_token_count"] != expected_prose["scored_token_count"]
        or prose["mean_token_logprob"] != expected_prose["mean_token_logprob"]
    ):
        raise RuntimeError("v7 alpha-zero prose identity changed")
    qa_gate = offline_audit._validate_qa_gate(
        state["ood_qa_gate"], qa["ood_qa"], qa["ood_qa"], "v7 state 0",
    )
    prose_gate = offline_audit._validate_prose_gate(
        state["ood_prose_gate"], prose, prose, "v7 state 0",
    )
    if state["strict_guards_passed"] is not (
        qa_gate["passed"] and prose_gate["passed"]
    ):
        raise RuntimeError("v7 alpha-zero strict gate changed")
    return {
        "seed": normalized["seed"],
        "state_count": 1,
        "coefficient_sha256": coefficient_sha,
        "robust_plan_sha256": robust["robust_plan_sha256"],
        "distributed_update_v4": distributed,
    }


def validate_completed_journal_v7(journal):
    if (
        not isinstance(journal, dict)
        or set(journal) != {
            "schema", "status", "in_progress", "policy", "targets",
            "trainer_configuration", "snapshot", "coefficient_plan",
            "states", "seeds", "content_sha256_before_self_field",
        }
        or journal.get("schema")
        != "eggroll-es-anchor-alpha-line-search-v7"
        or journal.get("status") != "complete"
        or journal.get("in_progress") is not None
        or journal.get("targets") != [0.0]
    ):
        raise RuntimeError("v7 journal is incomplete or not alpha-zero only")
    if journal.get("content_sha256_before_self_field") != (
        driver_v1.canonical_sha256({
            key: value for key, value in journal.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v7 journal content hash changed")
    policy = journal.get("policy")
    inherited_policy_keys = {
        "alpha_order", "branching", "resume", "rollback",
        "selection_during_execution", "ood_qa_max_degradation",
        "ood_prose_max_degradation", "bf16_alpha_semantics",
        "direct_alpha_confirmation_required", "frozen_layer_plan_required",
        "dense_gold_reward_required", "document_lcb_anchor_required",
        "optimization_data", "ood_validation_heldout_as_objective",
    }
    required_policy = {
        "direction_stability_family_v7": "front_middle_late_cross_seed",
        "stage_v7": "stability",
        "target_alpha_zero_only_v7": True,
        "benchmark_selection_forbidden_v7": True,
        "cross_seed_coefficient_cosine_threshold_v7": 0.5,
        "requires_clean_v6_family_v7": True,
    }
    if (
        not isinstance(policy, dict)
        or set(policy) != inherited_policy_keys | set(required_policy)
        or any(policy.get(key) != value for key, value in {
            "alpha_order": "zero_then_strictly_increasing",
            "branching": False, "resume": False, "rollback": False,
            "selection_during_execution": False,
            "ood_qa_max_degradation": 0.0,
            "ood_prose_max_degradation": 0.0,
            "bf16_alpha_semantics": "path_dependent_monotonic_pilot",
            "direct_alpha_confirmation_required": True,
            "frozen_layer_plan_required": True,
            "dense_gold_reward_required": True,
            "document_lcb_anchor_required": True,
            "optimization_data": "train_and_anchor_only",
            "ood_validation_heldout_as_objective": False,
        }.items())
        or any(policy.get(key) != value for key, value in required_policy.items())
    ):
        raise RuntimeError("v7 stability policy changed")
    snapshot = journal.get("snapshot")
    stability = (
        snapshot.get("direction_stability_v7")
        if isinstance(snapshot, dict) else None
    )
    if (
        snapshot.get("schema")
        != "eggroll-es-anchor-line-search-snapshot-v7"
        or set(snapshot) != {
            "schema", "train", "evaluations", "anchor",
            "fixed_train_batch", "implementation", "recipe",
            "distributed_update_v3", "frozen_layer_plan_v4",
            "dense_gold_reward_v4", "distributed_update_v4",
            "document_lcb_anchor_v5", "direction_stability_v7",
        }
        or not isinstance(stability, dict)
        or set(stability) != {
            "schema", "family", "stage", "arm", "layers",
            "seed_pair", "seed", "target_alphas",
            "benchmark_treatment_applied", "selection_surface",
            "coefficient_cosine_threshold", "recipe", "plan_sha256",
            "plan_file_sha256", "implementation_bundle_sha256",
        }
        or stability.get("schema")
        != "eggroll-es-direction-stability-snapshot-v7"
        or stability.get("family") != "front_middle_late_cross_seed"
        or stability.get("stage") != "stability"
        or stability.get("arm") not in STABILITY_ARMS_V7
        or stability.get("seed") not in STABILITY_SEEDS_V7
        or stability.get("seed_pair") != [43, 44]
        or stability.get("target_alphas") != [0.0]
        or stability.get("benchmark_treatment_applied") is not False
        or stability.get("selection_surface")
        != "coefficient_direction_only"
        or stability.get("coefficient_cosine_threshold") != 0.5
    ):
        raise RuntimeError("v7 stability snapshot changed")
    smoke = _clean_smoke_evidence_v7(SMOKE_GATE_PATH_V7)
    pilots = _pilot_family_evidence_v7(PILOT_EVIDENCE_PATH_V7)
    expected_recipe = frozen_recipe_v7(
        stability["arm"], stability["seed"], smoke, pilots,
    )
    if stability.get("recipe") != expected_recipe:
        raise RuntimeError("v7 persisted recipe changed")
    expected_snapshot_recipe = {
        key: expected_recipe[key] for key in (
            "model_name", "checkpoint", "sigma", "population_size",
            "batch_size", "mini_batch_size", "max_tokens", "seed",
            "min_anchor_cosine", "anchor_items_per_step", "target_alphas",
        )
    }
    if snapshot.get("recipe") != expected_snapshot_recipe:
        raise RuntimeError("v7 effective snapshot recipe changed")
    spec = next(
        item for item in anchor_v7.FROZEN_STABILITY_PLANS_V7.values()
        if item["plan"] == stability["arm"]
    )
    expected_plan_sha256 = next(
        key for key, item in anchor_v7.FROZEN_STABILITY_PLANS_V7.items()
        if item["plan"] == stability["arm"]
    )
    if (
        stability.get("layers") != spec["layers"]
        or stability.get("plan_sha256") != expected_plan_sha256
        or stability.get("plan_file_sha256") != spec["file_sha256"]
    ):
        raise RuntimeError("v7 persisted layer plan changed")
    actual_implementation = {
        key: anchor_v7.file_sha256(path)
        for key, path in V7_IMPLEMENTATION_PATHS.items()
    }
    implementation = snapshot.get("implementation")
    if (
        not isinstance(implementation, dict)
        or set(implementation) != set(V7_IMPLEMENTATION_PATHS)
        or
        {key: implementation.get(key) for key in V7_IMPLEMENTATION_PATHS}
        != actual_implementation
        or stability.get("implementation_bundle_sha256")
        != driver_v1.canonical_sha256(actual_implementation)
    ):
        raise RuntimeError("v7 implementation identity changed")
    with driver_v6.scoped_legacy_audit_v6():
        inherited = _validate_inherited_zero_target_v7(journal)
    if inherited["seed"] != stability["seed"]:
        raise RuntimeError("v7 inherited seed changed")
    return {
        **inherited,
        "arm": stability["arm"],
        "stage": "stability",
        "content_sha256": journal["content_sha256_before_self_field"],
    }


def execute_line_search(*args, **kwargs):
    with driver_v6.scoped_legacy_audit_v6():
        journal = driver_v4.execute_line_search(*args, **kwargs)
    trainer = args[0] if args else kwargs["trainer"]
    plan = trainer._latest_anchor_plan
    anchor_v7.validate_robust_plan_v7(plan, recompute_numeric=True)
    binding_v5 = anchor_v7.anchor_v6.anchor_v5.validate_robust_plan_v5(
        plan, recompute_numeric=True,
    )
    journal["schema"] = "eggroll-es-anchor-alpha-line-search-v7"
    journal["snapshot"]["schema"] = (
        "eggroll-es-anchor-line-search-snapshot-v7"
    )
    journal["policy"].update({
        "document_lcb_anchor_required": True,
        "optimization_data": "train_and_anchor_only",
        "ood_validation_heldout_as_objective": False,
        "direction_stability_family_v7": "front_middle_late_cross_seed",
        "stage_v7": "stability",
        "target_alpha_zero_only_v7": True,
        "benchmark_selection_forbidden_v7": True,
        "cross_seed_coefficient_cosine_threshold_v7": 0.5,
        "requires_clean_v6_family_v7": True,
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
        validate_completed_journal_v7(journal)
    except Exception as error:
        journal["status"] = "failed"
        journal["failure"] = {
            "type": type(error).__name__, "message": str(error),
            "phase": "validating_complete_v7_release_audit",
        }
        journal.pop("content_sha256_before_self_field", None)
        driver_v1.atomic_write_json(kwargs["journal_path"], journal)
        raise
    driver_v1.atomic_write_json(kwargs["journal_path"], journal)
    return journal


def main(argv=None):
    validate_effective_anchor_api()
    argv = list(sys.argv[1:] if argv is None else argv)
    bundle, remaining = anchor_v7.parse_frozen_layer_plan_cli_v7(argv)
    execution, remaining = validate_frozen_execution_cli_v7(remaining, bundle)
    set_active_v7(bundle, execution)
    if execution["dry_run"]:
        payload = {
            "schema": "eggroll-es-direction-stability-dry-run-v7",
            "arm": execution["arm"], "seed": execution["seed"],
            "stage": execution["stage"], "targets": [0.0],
            "recipe_sha256": driver_v1.canonical_sha256({
                key: execution[key] for key in V7_RECIPE_KEYS
            }),
        }
        print(json.dumps(payload, sort_keys=True))
        return payload
    old_argv = sys.argv
    old_anchor = driver_v1.anchor
    old_build = driver_v1.build_snapshot
    old_execute = driver_v1.execute_line_search
    sys.argv = [old_argv[0], *remaining]
    driver_v1.anchor = anchor_v7
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
