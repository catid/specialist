#!/usr/bin/env python3
"""Fail-closed four-arm edge-split line search using the v5 document LCB."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import aggregate_eggroll_es_anchor_replications as offline_audit
import run_eggroll_es_anchor_line_search as driver_v1
import run_eggroll_es_anchor_line_search_v4 as driver_v4
import run_eggroll_es_anchor_line_search_v5 as driver_v5
import train_eggroll_es_specialist_anchor_v6 as anchor_v6


ROOT = Path(__file__).resolve().parent
SMOKE_TARGETS_V6 = [0.0, 0.00000078125]
PILOT_TARGETS_V6 = [
    0.0, 0.00000078125, 0.0000015625, 0.000003125, 0.00000625,
]
FROZEN_TRAIN_DATASET_V6 = Path(
    "/tmp/specialist-s6-candidate-guarded-ead1b21/dataset/train"
).resolve()
FROZEN_EVAL_DATASET_V6 = Path(
    "/tmp/specialist-s6-candidate-guarded-ead1b21/dataset/eval"
).resolve()
FROZEN_OUTPUT_DIRECTORY_V6 = (
    ROOT / "experiments/eggroll_es_hpo/runs"
).resolve()
V6_RECIPE_KEYS = {
    "model_name", "checkpoint", "train_dataset", "eval_dataset", "sigma",
    "population_size", "batch_size", "mini_batch_size", "max_tokens",
    "seed", "engine_count", "tp_per_engine", "gpu_ids", "eval_splits",
    "target_alphas", "anchor_prose_jsonl", "anchor_prose_report",
    "anchor_items_per_step", "anchor_max_input_tokens", "min_anchor_cosine",
    "ood_prose_jsonl", "ood_prose_max_input_tokens",
    "document_lcb_config_sha256", "reward_function_timeout",
    "output_directory", "experiment_name", "logging", "wandb_project",
    "smoke_gate",
}
V6_IMPLEMENTATION_PATHS = {
    "driver": ROOT / "run_eggroll_es_anchor_line_search.py",
    "anchor_trainer": Path(anchor_v6.__file__).resolve(),
    "base_trainer": ROOT / "train_eggroll_es_specialist.py",
    "projection": ROOT / "eggroll_es_anchor.py",
    "upstream_trainer": (
        ROOT / "es-at-scale/es_at_scale/trainer/es_trainer.py"
    ),
    "upstream_worker": (
        ROOT / "es-at-scale/es_at_scale/utils/worker_extension.py"
    ),
    "corrected_driver": ROOT / "run_eggroll_es_anchor_line_search_v2.py",
    "exact_worker": ROOT / "eggroll_es_worker_v2.py",
    "distributed_driver_v3": ROOT / "run_eggroll_es_anchor_line_search_v3.py",
    "distributed_trainer_v3": ROOT / "train_eggroll_es_specialist_anchor_v3.py",
    "distributed_worker_v3": ROOT / "eggroll_es_worker_v3.py",
    "distributed_driver_v4": ROOT / "run_eggroll_es_anchor_line_search_v4.py",
    "distributed_trainer_v4": ROOT / "train_eggroll_es_specialist_anchor_v4.py",
    "distributed_worker_v4": ROOT / "eggroll_es_worker_v4.py",
    "distributed_driver_v5": ROOT / "run_eggroll_es_anchor_line_search_v5.py",
    "distributed_trainer_v5": ROOT / "train_eggroll_es_specialist_anchor_v5.py",
    "robust_anchor_v5": ROOT / "eggroll_es_robust_anchor.py",
    "distributed_driver_v6": Path(__file__).resolve(),
    "distributed_trainer_v6": Path(anchor_v6.__file__).resolve(),
    "distributed_worker_v6": ROOT / "eggroll_es_worker_v6.py",
    "layer_plan_builder_v6": ROOT / "build_eggroll_es_edge_split_plans_v6.py",
    "offline_audit_v6": Path(offline_audit.__file__).resolve(),
    "anchor_trainer_v1": ROOT / "train_eggroll_es_specialist_anchor.py",
    "anchor_trainer_v2": ROOT / "train_eggroll_es_specialist_anchor_v2.py",
    "anchor_builder_v1": ROOT / "build_general_prose_anchor.py",
    "anchor_url_normalizer_v1": ROOT / "build_eval_v3.py",
    "base_reward_v1": ROOT / "es_train_acc.py",
    "base_reward_common_v1": ROOT / "es_common.py",
    "base_reward_layer_plan_v1": ROOT / "es_layer_plan.py",
    "base_reward_qa_quality_v1": ROOT / "qa_quality.py",
    "model_config_v6": ROOT / "models/Qwen3.6-35B-A3B/config.json",
    "layer_plan_front_v6": ROOT / "experiments/layer_plans/front_dense.json",
    "layer_plan_middle_early_v6": (
        ROOT / "experiments/layer_plans/middle_early_dense_v6.json"
    ),
    "layer_plan_middle_late_v6": (
        ROOT / "experiments/layer_plans/middle_late_dense_v6.json"
    ),
    "layer_plan_back_v6": ROOT / "experiments/layer_plans/back_dense.json",
    "protocol_v6": (
        ROOT / "experiments/eggroll_es_hpo/"
        "S6_EDGE_SPLIT_V6_DOCUMENT_LCB_PROTOCOL.md"
    ),
    "contract_tests_v6": ROOT / "test_eggroll_es_specialist_anchor_v6.py",
}
_LEGACY_AUDIT_GLOBAL_NAMES_V6 = (
    "V4_FROZEN_LAYER_PLANS", "V4_SOURCE_UNIT_COUNT",
    "V4_RUNTIME_PARAMETER_COUNT", "V4_SELECTED_ELEMENT_COUNT",
    "V4_SELECTED_BYTE_COUNT",
)
_EXPECTED_LEGACY_AUDIT_GLOBALS_SHA256_V6 = (
    "5d0a20f15b2f788500d65faabbf50fa9a70f0c45460717e83b099b5e6f33a079"
)
_V6_LEGACY_AUDIT_REPLACEMENTS = {
    "V4_FROZEN_LAYER_PLANS": {
        plan_sha256: {
            key: anchor_v6.worker_v6.FROZEN_LAYER_PLANS_V6[plan_sha256][key]
            for key in (
                "file_sha256", "checkpoint_to_runtime_mapping_sha256",
                "runtime_selected_name_sha256",
            )
        }
        for plan_sha256 in anchor_v6.FROZEN_EDGE_SPLIT_PLANS_V6
    },
    "V4_SOURCE_UNIT_COUNT": 35,
    "V4_RUNTIME_PARAMETER_COUNT": 23,
    "V4_SELECTED_ELEMENT_COUNT": 142_999_552,
    "V4_SELECTED_BYTE_COUNT": 285_999_104,
}
V6_LEGACY_AUDIT_SCOPE_POLICY = {
    "schema": "eggroll-es-v6-scoped-legacy-audit-policy-v1",
    "validator": (
        "aggregate_eggroll_es_anchor_replications.validate_journal"
    ),
    "original_globals_sha256": (
        _EXPECTED_LEGACY_AUDIT_GLOBALS_SHA256_V6
    ),
    "replacement_globals": copy.deepcopy(_V6_LEGACY_AUDIT_REPLACEMENTS),
    "scope": "entire_v5_execute_or_v6_offline_validation",
    "concurrency": "process_nonblocking_lock_no_reentrancy",
    "restoration": "exact_original_objects_in_finally",
}
V6_LEGACY_AUDIT_SCOPE_CONFIG_SHA256 = driver_v1.canonical_sha256(
    V6_LEGACY_AUDIT_SCOPE_POLICY
)
_V6_LEGACY_AUDIT_LOCK = threading.Lock()
_ACTIVE_LAYER_PLAN_BUNDLE = None
_ACTIVE_EXECUTION_V6 = None


def _legacy_audit_globals_v6():
    return {
        name: getattr(offline_audit, name)
        for name in _LEGACY_AUDIT_GLOBAL_NAMES_V6
    }


@contextmanager
def scoped_legacy_audit_v6():
    """Run the unchanged legacy validator against the exact v6 capacity."""
    if not _V6_LEGACY_AUDIT_LOCK.acquire(blocking=False):
        raise RuntimeError("v6 legacy-audit scope is concurrent or reentrant")
    originals = _legacy_audit_globals_v6()
    mutated = False
    try:
        if driver_v1.canonical_sha256(originals) != (
            _EXPECTED_LEGACY_AUDIT_GLOBALS_SHA256_V6
        ):
            raise RuntimeError("legacy v4 audit globals drifted before v6 scope")
        if driver_v1.canonical_sha256(
            V6_LEGACY_AUDIT_SCOPE_POLICY
        ) != V6_LEGACY_AUDIT_SCOPE_CONFIG_SHA256:
            raise RuntimeError("v6 legacy-audit scope policy drifted")
        if V6_LEGACY_AUDIT_SCOPE_POLICY["replacement_globals"] != (
            _V6_LEGACY_AUDIT_REPLACEMENTS
        ):
            raise RuntimeError("v6 legacy-audit replacements drifted")
        for name, value in _V6_LEGACY_AUDIT_REPLACEMENTS.items():
            setattr(offline_audit, name, copy.deepcopy(value))
        mutated = True
        if _legacy_audit_globals_v6() != _V6_LEGACY_AUDIT_REPLACEMENTS:
            raise RuntimeError("v6 legacy-audit scope installation failed")
        yield
    finally:
        if mutated:
            for name, value in originals.items():
                setattr(offline_audit, name, value)
            restored = _legacy_audit_globals_v6()
            if (
                any(restored[name] is not originals[name] for name in originals)
                or driver_v1.canonical_sha256(restored)
                != _EXPECTED_LEGACY_AUDIT_GLOBALS_SHA256_V6
            ):
                _V6_LEGACY_AUDIT_LOCK.release()
                raise RuntimeError("v6 failed to restore legacy audit globals")
        _V6_LEGACY_AUDIT_LOCK.release()


def validate_effective_anchor_api(module=anchor_v6):
    driver_v5.validate_effective_anchor_api(driver_v5.anchor_v5)
    required = (
        "coefficient_sha256", "load_anchor_prose", "load_trainer",
        "load_frozen_layer_plan_v6", "validate_robust_plan_v6",
    )
    missing = [
        name for name in required if not callable(getattr(module, name, None))
    ]
    if missing:
        raise RuntimeError(
            "v6 anchor adapter is missing callable API members: "
            + ", ".join(missing)
        )
    if module.WORKER_EXTENSION != (
        "eggroll_es_worker_v6.FrozenEdgeSplitAuditWorkerExtensionV6"
    ):
        raise RuntimeError("v6 selected the wrong distributed worker")
    return required


def set_active_layer_plan_bundle_v6(bundle, execution):
    global _ACTIVE_LAYER_PLAN_BUNDLE, _ACTIVE_EXECUTION_V6
    anchor_v6.validate_frozen_layer_plan_bundle_v6(bundle)
    if (
        not isinstance(execution, dict)
        or execution.get("arm") != bundle["edge_split_v6"]["plan"]
    ):
        raise ValueError("v6 execution does not match its frozen arm")
    driver_v5.set_active_layer_plan_bundle_v5(bundle)
    anchor_v6.set_default_layer_plan_bundle_v6(bundle)
    _ACTIVE_LAYER_PLAN_BUNDLE = bundle
    _ACTIVE_EXECUTION_V6 = execution


def _active_v6():
    bundle = _ACTIVE_LAYER_PLAN_BUNDLE
    execution = _ACTIVE_EXECUTION_V6
    anchor_v6.validate_frozen_layer_plan_bundle_v6(bundle)
    if not isinstance(execution, dict):
        raise RuntimeError("v6 has no validated execution stage")
    return bundle, execution


def _parse_targets_v6(raw):
    targets = driver_v1.parse_target_alphas(raw)
    if any(not math.isfinite(value) for value in targets):
        raise ValueError("v6 target alpha is non-finite")
    return targets


def _experiment_name_pattern_v6(arm, stage):
    return re.compile(
        rf"^snapshot794_layer_v6_{re.escape(arm)}_"
        rf"{stage}(?:_[a-z0-9]+)*_seed42$"
    )


def frozen_recipe_v6(arm, stage, experiment_name, smoke_gate):
    if stage not in {"smoke", "pilot"}:
        raise ValueError("v6 recipe stage is invalid")
    if _experiment_name_pattern_v6(arm, stage).fullmatch(
        str(experiment_name)
    ) is None:
        raise ValueError("v6 experiment name changed")
    population_size, batch_size = (
        (4, 8) if stage == "smoke" else (16, 64)
    )
    targets = SMOKE_TARGETS_V6 if stage == "smoke" else PILOT_TARGETS_V6
    return {
        "model_name": str(
            (anchor_v6.ROOT / "models/Qwen3.6-35B-A3B").resolve()
        ),
        "checkpoint": None,
        "train_dataset": str(FROZEN_TRAIN_DATASET_V6),
        "eval_dataset": str(FROZEN_EVAL_DATASET_V6),
        "sigma": 0.0003,
        "population_size": population_size,
        "batch_size": batch_size,
        "mini_batch_size": 64,
        "max_tokens": 32,
        "seed": 42,
        "engine_count": 4,
        "tp_per_engine": 1,
        "gpu_ids": [0, 1, 2, 3],
        "eval_splits": ["validation", "ood_qa"],
        "target_alphas": list(targets),
        "anchor_prose_jsonl": str(
            (anchor_v6.ROOT / "data/general_prose_anchor_v1.jsonl").resolve()
        ),
        "anchor_prose_report": str(
            (anchor_v6.ROOT / "data/general_prose_anchor_v1.report.json").resolve()
        ),
        "anchor_items_per_step": 128,
        "anchor_max_input_tokens": 512,
        "min_anchor_cosine": 0.8,
        "ood_prose_jsonl": str(
            (anchor_v6.ROOT / "data/ood_prose_v3.jsonl").resolve()
        ),
        "ood_prose_max_input_tokens": 1024,
        "document_lcb_config_sha256": (
            anchor_v6.anchor_v5.DOCUMENT_LCB_CONFIG_SHA256_V5
        ),
        "reward_function_timeout": 10,
        "output_directory": str(FROZEN_OUTPUT_DIRECTORY_V6),
        "experiment_name": experiment_name,
        "logging": "none",
        "wandb_project": "specialist-eggroll-es",
        "smoke_gate": copy.deepcopy(smoke_gate),
    }


def _smoke_evidence_v6(path):
    path = Path(path).resolve()
    offline_audit._assert_no_heldout(str(path), "v6 smoke-gate path")
    try:
        evidence = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("v6 smoke-gate evidence is unreadable") from error
    entries = evidence.get("arms") if isinstance(evidence, dict) else None
    if (
        evidence.get("schema") != "eggroll-es-edge-split-smoke-gate-v6"
        or set(evidence) != {"schema", "arms"}
        or not isinstance(entries, list)
        or len(entries) != 4
    ):
        raise ValueError("v6 smoke-gate evidence schema changed")
    offline_audit._assert_no_heldout(evidence, "v6 smoke-gate evidence")
    summaries = []
    seen_arms = set()
    seen_paths = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "arm", "journal", "content_sha256",
        }:
            raise ValueError("v6 smoke-gate entry fields changed")
        journal_path = Path(str(entry.get("journal", ""))).resolve()
        offline_audit._assert_no_heldout(
            str(journal_path), "v6 smoke journal path",
        )
        expected_hash = entry.get("content_sha256")
        if journal_path in seen_paths:
            raise ValueError("v6 smoke gate reused one journal")
        try:
            journal = json.loads(journal_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("v6 smoke journal is unreadable") from error
        audit = validate_completed_journal_v6(journal)
        arm = audit["arm"]
        if (
            entry.get("arm") != arm
            or expected_hash != audit["content_sha256"]
            or audit["stage"] != "smoke"
            or audit["state_count"] != len(SMOKE_TARGETS_V6)
        ):
            raise ValueError("v6 smoke evidence does not match its journal")
        seen_arms.add(arm)
        seen_paths.add(journal_path)
        summaries.append({
            "arm": arm,
            "content_sha256": audit["content_sha256"],
            "coefficient_sha256": audit["coefficient_sha256"],
            "robust_plan_sha256": audit["robust_plan_sha256"],
        })
    expected_arms = {
        spec["plan"]
        for spec in anchor_v6.FROZEN_EDGE_SPLIT_PLANS_V6.values()
    }
    if seen_arms != expected_arms:
        raise ValueError("v6 smoke evidence does not cover all four arms")
    summaries.sort(key=lambda row: row["arm"])
    summary = {
        "schema": "eggroll-es-edge-split-smoke-gate-binding-v6",
        "arms": summaries,
    }
    summary["binding_sha256"] = driver_v1.canonical_sha256(summary)
    return summary


def validate_frozen_execution_cli_v6(argv, bundle):
    inherited = driver_v5.validate_frozen_execution_cli_v5(argv)
    stage_parser = argparse.ArgumentParser(add_help=False)
    stage_parser.add_argument("--v6-stage", choices=("smoke", "pilot"))
    stage_parser.add_argument("--v6-smoke-gate-json")
    stage, remaining = stage_parser.parse_known_args(list(argv))
    recipe_parser = argparse.ArgumentParser(add_help=False)
    recipe_parser.add_argument("--population-size", type=int, default=8)
    recipe_parser.add_argument("--batch-size", type=int, default=64)
    recipe_parser.add_argument("--seed", type=int, default=42)
    recipe_parser.add_argument(
        "--train-dataset", default=str(FROZEN_TRAIN_DATASET_V6),
    )
    recipe_parser.add_argument(
        "--eval-dataset", default=str(FROZEN_EVAL_DATASET_V6),
    )
    recipe_parser.add_argument("--reward-function-timeout", type=int, default=10)
    recipe_parser.add_argument(
        "--output-directory", default=str(FROZEN_OUTPUT_DIRECTORY_V6),
    )
    recipe_parser.add_argument("--experiment-name")
    recipe_parser.add_argument("--logging", default="none")
    recipe_parser.add_argument(
        "--wandb-project", default="specialist-eggroll-es",
    )
    recipe_parser.add_argument(
        "--target-alphas", default="0,0.000025,0.00005,0.0001,0.00015",
    )
    recipe, _ = recipe_parser.parse_known_args(list(argv))
    if stage.v6_stage is None:
        raise ValueError("v6 requires an explicit --v6-stage")
    targets = _parse_targets_v6(recipe.target_alphas)
    metadata = anchor_v6.validate_frozen_layer_plan_bundle_v6(bundle)
    expected_name = _experiment_name_pattern_v6(
        metadata["plan"], stage.v6_stage,
    )
    if recipe.seed != 42:
        raise ValueError("v6 smoke and pilot recipes are frozen at seed 42")
    if (
        Path(recipe.train_dataset).resolve() != FROZEN_TRAIN_DATASET_V6
        or Path(recipe.eval_dataset).resolve() != FROZEN_EVAL_DATASET_V6
        or recipe.reward_function_timeout != 10
        or Path(recipe.output_directory).resolve()
        != FROZEN_OUTPUT_DIRECTORY_V6
        or recipe.logging != "none"
        or recipe.wandb_project != "specialist-eggroll-es"
        or not isinstance(recipe.experiment_name, str)
        or expected_name.fullmatch(recipe.experiment_name) is None
    ):
        raise ValueError("v6 runtime path, timeout, logging, or run name changed")
    smoke_gate = None
    if stage.v6_stage == "smoke":
        if (
            recipe.population_size != 4
            or recipe.batch_size != 8
            or targets != SMOKE_TARGETS_V6
            or stage.v6_smoke_gate_json is not None
        ):
            raise ValueError("v6 smoke recipe must be pop4/batch8/two-alpha")
    else:
        if (
            recipe.population_size != 16
            or recipe.batch_size != 64
            or targets != PILOT_TARGETS_V6
            or stage.v6_smoke_gate_json is None
        ):
            raise ValueError(
                "v6 pilot requires pop16/batch64/five-alpha and smoke evidence"
            )
        smoke_gate = _smoke_evidence_v6(stage.v6_smoke_gate_json)
    execution = {
        "schema": "eggroll-es-edge-split-execution-v6",
        "stage": stage.v6_stage,
        "arm": metadata["plan"],
        "paired_control": metadata["paired_control"],
        "population_size": recipe.population_size,
        "batch_size": recipe.batch_size,
        "seed": recipe.seed,
        "target_alphas": targets,
        "checkpoint": None,
        "train_dataset": str(FROZEN_TRAIN_DATASET_V6),
        "eval_dataset": str(FROZEN_EVAL_DATASET_V6),
        "reward_function_timeout": 10,
        "output_directory": str(FROZEN_OUTPUT_DIRECTORY_V6),
        "experiment_name": recipe.experiment_name,
        "logging": "none",
        "wandb_project": "specialist-eggroll-es",
        "smoke_gate": smoke_gate,
    }
    validated = {**inherited, **execution}
    persisted_recipe = {
        key: copy.deepcopy(validated[key]) for key in V6_RECIPE_KEYS
    }
    if persisted_recipe != frozen_recipe_v6(
        metadata["plan"], stage.v6_stage, recipe.experiment_name, smoke_gate,
    ):
        raise RuntimeError("v6 effective runtime recipe changed")
    return validated, remaining


def build_snapshot(*args, **kwargs):
    bundle, execution = _active_v6()
    snapshot = driver_v5.build_snapshot(*args, **kwargs)
    implementation = {
        key: anchor_v6.file_sha256(path)
        for key, path in V6_IMPLEMENTATION_PATHS.items()
    }
    snapshot["implementation"].update(implementation)
    snapshot["edge_split_v6"] = {
        "schema": "eggroll-es-edge-split-snapshot-v6",
        "family": "four_arm_capacity_matched_edge_split",
        "arm": execution["arm"],
        "paired_control": execution["paired_control"],
        "layers": list(bundle["edge_split_v6"]["layers"]),
        "source_unit_count": 35,
        "runtime_selected_parameter_count": 23,
        "selected_element_count": 142_999_552,
        "plan_file_sha256": bundle["file_sha256"],
        "plan_sha256": bundle["plan_sha256"],
        "stage": execution["stage"],
        "recipe": {
            key: copy.deepcopy(execution[key])
            for key in V6_RECIPE_KEYS
        },
        "document_lcb_config_sha256": (
            anchor_v6.anchor_v5.DOCUMENT_LCB_CONFIG_SHA256_V5
        ),
        "optimization_data": "train_and_anchor_only",
        "legacy_audit_scope_config_sha256": (
            V6_LEGACY_AUDIT_SCOPE_CONFIG_SHA256
        ),
        "implementation_bundle_sha256": driver_v1.canonical_sha256(
            implementation
        ),
    }
    return snapshot


def _v5_compatibility_journal_v6(journal):
    compatible = copy.deepcopy(journal)
    compatible["schema"] = "eggroll-es-anchor-alpha-line-search-v5"
    compatible["snapshot"]["schema"] = (
        "eggroll-es-anchor-line-search-snapshot-v5"
    )
    compatible["snapshot"].pop("edge_split_v6", None)
    compatible["policy"].pop("edge_split_family_v6", None)
    compatible["policy"].pop("pilot_requires_four_smokes_v6", None)
    compatible.pop("content_sha256_before_self_field", None)
    compatible["content_sha256_before_self_field"] = (
        driver_v1.canonical_sha256(compatible)
    )
    return compatible


def _validate_snapshot_v6(snapshot, journal):
    edge = snapshot.get("edge_split_v6") if isinstance(snapshot, dict) else None
    implementation = (
        snapshot.get("implementation") if isinstance(snapshot, dict) else None
    )
    if (
        snapshot.get("schema")
        != "eggroll-es-anchor-line-search-snapshot-v6"
        or set(snapshot) != {
            "schema", "train", "evaluations", "anchor",
            "fixed_train_batch", "implementation", "recipe",
            "distributed_update_v3", "frozen_layer_plan_v4",
            "dense_gold_reward_v4", "distributed_update_v4",
            "document_lcb_anchor_v5", "edge_split_v6",
        }
        or not isinstance(edge, dict)
        or set(edge) != {
            "schema", "family", "arm", "paired_control", "layers",
            "source_unit_count", "runtime_selected_parameter_count",
            "selected_element_count", "plan_file_sha256", "plan_sha256",
            "stage", "recipe", "document_lcb_config_sha256",
            "optimization_data", "legacy_audit_scope_config_sha256",
            "implementation_bundle_sha256",
        }
        or edge.get("schema") != "eggroll-es-edge-split-snapshot-v6"
        or edge.get("family") != "four_arm_capacity_matched_edge_split"
        or edge.get("source_unit_count") != 35
        or edge.get("runtime_selected_parameter_count") != 23
        or edge.get("selected_element_count") != 142_999_552
        or edge.get("document_lcb_config_sha256")
        != anchor_v6.anchor_v5.DOCUMENT_LCB_CONFIG_SHA256_V5
        or edge.get("optimization_data") != "train_and_anchor_only"
        or edge.get("legacy_audit_scope_config_sha256")
        != V6_LEGACY_AUDIT_SCOPE_CONFIG_SHA256
        or not isinstance(implementation, dict)
    ):
        raise RuntimeError("v6 snapshot contract changed")
    spec = anchor_v6.FROZEN_EDGE_SPLIT_PLANS_V6.get(edge.get("plan_sha256"))
    if (
        not isinstance(spec, dict)
        or edge.get("arm") != spec["plan"]
        or edge.get("paired_control") != spec["paired_control"]
        or edge.get("layers") != spec["layers"]
        or edge.get("plan_file_sha256") != spec["file_sha256"]
    ):
        raise RuntimeError("v6 snapshot selected an unfrozen arm")
    actual_implementation = {
        key: anchor_v6.file_sha256(path)
        for key, path in V6_IMPLEMENTATION_PATHS.items()
    }
    persisted_implementation = {
        key: implementation.get(key) for key in V6_IMPLEMENTATION_PATHS
    }
    if (
        persisted_implementation != actual_implementation
        or set(implementation) != set(V6_IMPLEMENTATION_PATHS)
        or edge.get("implementation_bundle_sha256")
        != driver_v1.canonical_sha256(persisted_implementation)
    ):
        raise RuntimeError("v6 implementation identity changed")
    recipe = edge.get("recipe")
    stage = edge.get("stage")
    expected_targets = (
        SMOKE_TARGETS_V6 if stage == "smoke" else PILOT_TARGETS_V6
    )
    expected_shape = (
        (4, 8) if stage == "smoke" else (16, 64)
    )
    try:
        expected_recipe = frozen_recipe_v6(
            edge.get("arm"), stage, recipe.get("experiment_name"),
            recipe.get("smoke_gate"),
        )
    except (AttributeError, ValueError) as error:
        raise RuntimeError("v6 persisted recipe changed") from error
    if (
        stage not in {"smoke", "pilot"}
        or not isinstance(recipe, dict)
        or set(recipe) != V6_RECIPE_KEYS
        or recipe != expected_recipe
        or (recipe.get("population_size"), recipe.get("batch_size"))
        != expected_shape
        or recipe.get("seed") != 42
        or recipe.get("target_alphas") != expected_targets
        or journal.get("targets") != expected_targets
        or snapshot.get("recipe", {}).get("population_size")
        != expected_shape[0]
        or snapshot.get("recipe", {}).get("batch_size") != expected_shape[1]
        or snapshot.get("recipe", {}).get("seed") != 42
        or snapshot.get("recipe", {}).get("target_alphas") != expected_targets
    ):
        raise RuntimeError("v6 persisted recipe changed")
    smoke_gate = recipe.get("smoke_gate")
    if stage == "smoke" and smoke_gate is not None:
        raise RuntimeError("v6 smoke unexpectedly depends on prior results")
    if stage == "pilot":
        arms = smoke_gate.get("arms") if isinstance(smoke_gate, dict) else None
        expected_arms = sorted(
            item["plan"]
            for item in anchor_v6.FROZEN_EDGE_SPLIT_PLANS_V6.values()
        )
        if (
            smoke_gate.get("schema")
            != "eggroll-es-edge-split-smoke-gate-binding-v6"
            or set(smoke_gate) != {
                "schema", "arms", "binding_sha256",
            }
            or not isinstance(arms, list)
            or [row.get("arm") for row in arms] != expected_arms
            or smoke_gate.get("binding_sha256")
            != driver_v1.canonical_sha256({
                key: value for key, value in smoke_gate.items()
                if key != "binding_sha256"
            })
            or any(
                not isinstance(row, dict)
                or set(row) != {
                    "arm", "content_sha256", "coefficient_sha256",
                    "robust_plan_sha256",
                }
                or offline_audit.HASH_PATTERN.fullmatch(str(row.get(key))) is None
                for row in arms
                for key in (
                    "content_sha256", "coefficient_sha256",
                    "robust_plan_sha256",
                )
            )
        ):
            raise RuntimeError("v6 pilot smoke-gate binding changed")
    return edge


def _validate_completed_journal_v6_scoped(journal):
    if (
        not isinstance(journal, dict)
        or set(journal) != {
            "schema", "status", "in_progress", "policy", "targets",
            "trainer_configuration", "snapshot", "coefficient_plan",
            "states", "seeds", "content_sha256_before_self_field",
        }
        or journal.get("schema")
        != "eggroll-es-anchor-alpha-line-search-v6"
        or journal.get("status") != "complete"
        or journal.get("in_progress") is not None
    ):
        raise RuntimeError("v6 journal is incomplete or has the wrong schema")
    if journal.get("content_sha256_before_self_field") != (
        driver_v1.canonical_sha256({
            key: value for key, value in journal.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v6 journal content hash changed")
    policy = journal.get("policy")
    if (
        not isinstance(policy, dict)
        or set(policy) != {
            "alpha_order", "branching", "resume", "rollback",
            "selection_during_execution", "ood_qa_max_degradation",
            "ood_prose_max_degradation", "bf16_alpha_semantics",
            "direct_alpha_confirmation_required",
            "frozen_layer_plan_required", "dense_gold_reward_required",
            "document_lcb_anchor_required", "optimization_data",
            "ood_validation_heldout_as_objective",
            "edge_split_family_v6", "pilot_requires_four_smokes_v6",
        }
        or policy.get("edge_split_family_v6")
        != "four_arm_capacity_matched"
        or policy.get("pilot_requires_four_smokes_v6") is not True
        or policy.get("ood_validation_heldout_as_objective") is not False
    ):
        raise RuntimeError("v6 journal policy changed")
    edge = _validate_snapshot_v6(journal.get("snapshot"), journal)
    coefficient_plan = journal.get("coefficient_plan")
    layer = (
        coefficient_plan.get("frozen_layer_plan_v4")
        if isinstance(coefficient_plan, dict) else None
    )
    if (
        not isinstance(layer, dict)
        or layer.get("plan_sha256") != edge["plan_sha256"]
    ):
        raise RuntimeError("v6 coefficient plan differs from its arm")
    try:
        v5_audit = driver_v5.validate_completed_journal_v5(
            _v5_compatibility_journal_v6(journal)
        )
    except Exception as error:
        raise RuntimeError(
            "v6 persisted journal failed its complete inherited v5 audit"
        ) from error
    return {
        **v5_audit,
        "arm": edge["arm"],
        "paired_control": edge["paired_control"],
        "stage": edge["stage"],
        "content_sha256": journal["content_sha256_before_self_field"],
    }


def validate_completed_journal_v6(journal):
    with scoped_legacy_audit_v6():
        return _validate_completed_journal_v6_scoped(journal)


def execute_line_search(*args, **kwargs):
    with scoped_legacy_audit_v6():
        journal = driver_v5.execute_line_search(*args, **kwargs)
    journal["schema"] = "eggroll-es-anchor-alpha-line-search-v6"
    journal["snapshot"]["schema"] = (
        "eggroll-es-anchor-line-search-snapshot-v6"
    )
    journal["policy"]["edge_split_family_v6"] = (
        "four_arm_capacity_matched"
    )
    journal["policy"]["pilot_requires_four_smokes_v6"] = True
    journal.pop("content_sha256_before_self_field", None)
    journal["content_sha256_before_self_field"] = (
        driver_v1.canonical_sha256(journal)
    )
    try:
        validate_completed_journal_v6(journal)
    except Exception as error:
        journal["status"] = "failed"
        journal["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "phase": "validating_complete_v6_release_audit",
        }
        journal.pop("content_sha256_before_self_field", None)
        driver_v1.atomic_write_json(kwargs["journal_path"], journal)
        raise
    driver_v1.atomic_write_json(kwargs["journal_path"], journal)
    return journal


def main(argv=None):
    validate_effective_anchor_api()
    argv = list(sys.argv[1:] if argv is None else argv)
    bundle, remaining = anchor_v6.parse_frozen_layer_plan_cli_v6(argv)
    execution, remaining = validate_frozen_execution_cli_v6(
        remaining, bundle,
    )
    set_active_layer_plan_bundle_v6(bundle, execution)
    old_argv = sys.argv
    old_anchor = driver_v1.anchor
    old_build = driver_v1.build_snapshot
    old_execute = driver_v1.execute_line_search
    sys.argv = [old_argv[0], *remaining]
    driver_v1.anchor = anchor_v6
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
