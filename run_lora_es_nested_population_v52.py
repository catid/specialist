#!/usr/bin/env python3
"""Fail-closed V52 nested P8-vs-P16 LoRA-ES train-only executor.

Dry-run validates only the self-contained preregistration and source bindings.
The live path is lazy: importing this module never imports torch, Ray, vLLM,
datasets, or any protected-evaluation runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lora_es_nested_population_v52 as design


ROOT = Path(__file__).resolve().parent
PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_nested_p8_vs_p16_v52_retry1.json"
).resolve()
RUN_DIR = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v52_matched_lora_es_nested_p8_vs_p16_retry1"
).resolve()
ATTEMPT = (
    RUN_DIR.parent / ".v52_matched_lora_es_nested_p8_vs_p16_retry1.attempt.json"
).resolve()
WORKER_EXTENSION_V52 = (
    "eggroll_es_worker_lora_v52.LoRAAdapterStateWorkerExtensionV52"
)
P8_SNAPSHOT = (RUN_DIR / "p8_candidate_v52").resolve()
P16_SNAPSHOT = (RUN_DIR / "p16_candidate_v52").resolve()
NUMERIC_CALIBRATION = (RUN_DIR / "numeric_calibration_v52.json").resolve()
ANCHOR_CALIBRATION = (RUN_DIR / "anchor_calibration_v52.json").resolve()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--preregistration", required=True)
    value.add_argument("--preregistration-sha256", required=True)
    value.add_argument("--preregistration-content-sha256", required=True)
    value.add_argument("--dry-run", action="store_true")
    value.add_argument(
        "--execute",
        action="store_true",
        help="enter the separately sealed GPU execution path",
    )
    return value


def _expected_artifacts_v52() -> dict:
    return {
        "attempt": str(ATTEMPT),
        "run_directory": str(RUN_DIR),
        "population": str(RUN_DIR / "nested_population_v52.json"),
        "p8_train_gate": str(RUN_DIR / "p8_train_gate_v52.json"),
        "p16_train_gate": str(RUN_DIR / "p16_train_gate_v52.json"),
        "p8_candidate_snapshot": str(P8_SNAPSHOT),
        "p16_candidate_snapshot": str(P16_SNAPSHOT),
        "numeric_calibration": str(NUMERIC_CALIBRATION),
        "anchor_calibration": str(ANCHOR_CALIBRATION),
        "ood_aggregate": str(RUN_DIR / "ood_first_aggregate_v52.json"),
        "shadow_aggregate": str(RUN_DIR / "document_disjoint_shadow_v52.json"),
        "report": str(RUN_DIR / "nested_population_report_v52.json"),
        "gpu_log": str(RUN_DIR / "gpu_activity_v52.jsonl"),
        "failure": str(RUN_DIR / "failure_v52.json"),
        "sealed_holdout_artifact": None,
    }


def _read_self_hashed_v52(path: Path, file_sha: str, content_sha: str) -> dict:
    if design.file_sha256_v52(path) != file_sha:
        raise RuntimeError(f"v52 sealed train contract file changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != content_sha
        or design.canonical_sha256_v52(compact) != content_sha
    ):
        raise RuntimeError(f"v52 sealed train contract content changed: {path}")
    return value


def load_generation_panel_v52() -> dict:
    value = _read_self_hashed_v52(
        design.TRAIN_GENERATION_PANEL_V52,
        design.SUBSET_FILE_SHA256_V52,
        design.SUBSET_CONTENT_SHA256_V52,
    )
    subset = value.get("subset", {})
    items = subset.get("items", [])
    rows = [item.get("row_sha256") for item in items]
    units = [item.get("unit_identity_sha256") for item in items]
    if (
        value.get("schema") != "sealed-v434-train-generation-panel-v52"
        or value.get("status") != "complete_before_v52_preregistration"
        or value.get("selected_rows") != 64
        or value.get("selected_conflict_units") != 64
        or value.get("model_outcomes_used_for_selection") is not False
        or value.get("protected_semantics_opened") is not False
        or value.get("shadow_ood_holdout_or_benchmark_opened") is not False
        or value.get("gpu_or_model_accessed") is not False
        or subset.get("schema") != "v52-v434-train-generation-panel"
        or subset.get("status")
        != "selected_content_free_before_v52_preregistration"
        or subset.get("model_outcomes_used_for_selection") is not False
        or len(items) != 64
        or [item.get("request_index") for item in items] != list(range(64))
        or len(set(rows)) != 64
        or len(set(units)) != 64
        or rows != subset.get("request_order_row_sha256")
        or subset.get("request_order_sha256")
        != design.canonical_sha256_v52(rows)
        or value.get("request_order_sha256")
        != design.REQUEST_ORDER_SHA256_V52
    ):
        raise RuntimeError("v52 v434 generation panel contract changed")
    return value


def load_train_bundle_v52() -> dict:
    from qa_quality import qa_pair_from_record

    membership = _read_self_hashed_v52(
        design.TRAIN_MEMBERSHIP_V52,
        design.MEMBERSHIP_SHA256_V52,
        design.MEMBERSHIP_CONTENT_SHA256_V52,
    )
    if (
        design.file_sha256_v52(design.TRAIN_DATASET_V52)
        != design.DATASET_SHA256_V52
        or membership.get("schema")
        != "v52-v434-train-row-conflict-unit-membership"
        or membership.get("status")
        != "complete_content_free_projection_before_v52_preregistration"
        or membership.get("rows") != 448
        or membership.get("conflict_units") != 208
        or membership.get("question_answer_evidence_or_text_persisted") is not False
        or membership.get("protected_semantics_opened") is not False
    ):
        raise RuntimeError("v52 v434 dataset or membership changed")
    rows = [
        json.loads(line) for line in design.TRAIN_DATASET_V52.read_text(
            encoding="utf-8"
        ).splitlines() if line
    ]
    members = membership.get("items", [])
    if len(rows) != 448 or len(members) != 448:
        raise RuntimeError("v52 v434 train row coverage changed")
    prepared = []
    for index, (row, member) in enumerate(zip(rows, members, strict=True)):
        row_sha = hashlib.sha256(json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        pair = qa_pair_from_record(row)
        if (
            pair is None or not pair[0] or not pair[1]
            or member.get("row_index") != index
            or member.get("row_sha256") != row_sha
            or not isinstance(member.get("unit_identity_sha256"), str)
            or not isinstance(member.get("row_count"), int)
            or member["row_count"] <= 0
        ):
            raise RuntimeError("v52 v434 train row identity or QA pair changed")
        prepared.append((row_sha, pair, member))
    weights = [
        1.0 / (208 * member["row_count"])
        for _row_sha, _pair, member in prepared
    ]
    if abs(sum(weights) - 1.0) > 1e-12:
        raise RuntimeError("v52 v434 equal-unit weights changed")
    result = {
        "schema": "eggroll-es-equal-unit-train-bundle-v52-v434",
        "dataset": {
            "path": str(design.TRAIN_DATASET_V52),
            "file_sha256": design.DATASET_SHA256_V52,
            "rows": 448,
            "ordered_row_sha256": membership["ordered_row_sha256"],
        },
        "train_membership": {
            "path": str(design.TRAIN_MEMBERSHIP_V52),
            "file_sha256": design.MEMBERSHIP_SHA256_V52,
            "content_sha256": design.MEMBERSHIP_CONTENT_SHA256_V52,
            "ordered_membership_sha256": membership[
                "ordered_membership_sha256"
            ],
        },
        "questions": [pair[0] for _row_sha, pair, _member in prepared],
        "answers": [pair[1] for _row_sha, pair, _member in prepared],
        "weights": weights,
        "row_sha256": [row_sha for row_sha, _pair, _member in prepared],
        "conflict_units": 208,
        "weight_identity_sha256": design.canonical_sha256_v52([{
            "row_sha256": row_sha,
            "unit_identity_sha256": member["unit_identity_sha256"],
            "unit_rows": member["row_count"],
        } for row_sha, _pair, member in prepared]),
        "unit_membership_v48b": [{
            "row_sha256": row_sha,
            "unit_identity_sha256": member["unit_identity_sha256"],
            "row_count": member["row_count"],
        } for row_sha, _pair, member in prepared],
    }
    result["content_sha256_before_self_field"] = (
        design.canonical_sha256_v52(result)
    )
    if (
        result["content_sha256_before_self_field"]
        != design.TRAIN_BUNDLE_CONTENT_SHA256_V52
    ):
        raise RuntimeError("v52 v434 train bundle content changed")
    return result


def augment_unit_membership_v52(bundle: dict) -> dict:
    if (
        bundle.get("content_sha256_before_self_field")
        != design.TRAIN_BUNDLE_CONTENT_SHA256_V52
        or len(bundle.get("unit_membership_v48b", [])) != 448
    ):
        raise RuntimeError("v52 v434 train bundle membership changed")
    result = dict(bundle)
    result["unit_membership_v43i"] = [{
        "unit_identity_sha256": item["unit_identity_sha256"],
        "row_count": item["row_count"],
    } for item in bundle["unit_membership_v48b"]]
    result["unit_membership_sha256_v43i"] = (
        design.canonical_sha256_v52([{
            "row_sha256": row_sha,
            "unit_identity_sha256": item["unit_identity_sha256"],
            "row_count": item["row_count"],
        } for row_sha, item in zip(
            bundle["row_sha256"], bundle["unit_membership_v48b"], strict=True,
        )])
    )
    return result


def verify_adapter_contract_v52() -> dict:
    observed = {
        "source_weights": design.file_sha256_v52(design.SOURCE_WEIGHTS_V52),
        "source_config": design.file_sha256_v52(design.SOURCE_CONFIG_V52),
        "staged_weights": design.file_sha256_v52(design.STAGED_WEIGHTS_V52),
        "staged_config": design.file_sha256_v52(design.STAGED_CONFIG_V52),
        "staged_manifest": design.file_sha256_v52(design.STAGED_MANIFEST_V52),
    }
    expected = {
        "source_weights": design.SOURCE_WEIGHTS_SHA256_V52,
        "source_config": design.SOURCE_CONFIG_SHA256_V52,
        "staged_weights": design.STAGED_WEIGHTS_SHA256_V52,
        "staged_config": design.STAGED_CONFIG_SHA256_V52,
        "staged_manifest": design.STAGED_MANIFEST_FILE_SHA256_V52,
    }
    stage = design.sealed_json_v52("v49d_stage_manifest")
    if (
        observed != expected
        or stage.get("status") != "complete_cpu_only_key_transform"
        or stage.get("transformed_identity", {}).get("sha256")
        != design.STAGED_TRANSFORMED_IDENTITY_SHA256_V52
    ):
        raise RuntimeError("v52 equal-v434 source/staged adapter contract changed")
    return {
        "source": str(design.SOURCE_V52),
        "staged": str(design.STAGED_V52),
        "file_sha256": observed,
        "canonical_fp32_master_sha256": design.MASTER_SHA256_V52,
        "bf16_runtime_values_sha256": design.MASTER_RUNTIME_SHA256_V52,
        "preinstall_postinstall_equality_required": True,
    }


def load_preregistration_v52(args) -> dict:
    path = Path(args.preregistration).resolve()
    if design.file_sha256_v52(path) != args.preregistration_sha256:
        raise RuntimeError("v52 preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    arms = design.scientific_arms_v52()
    recipe = value.get("fixed_recipe", {})
    launcher = value.get("launcher_fix", {})
    if (
        value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or design.canonical_sha256_v52(compact)
        != args.preregistration_content_sha256
        or value.get("schema")
        != "matched-lora-es-nested-p8-vs-p16-preregistration-v52"
        or value.get("retry_revision") != design.RETRY_REVISION_V52
        or value.get("status")
        != (
            "preregistered_after_content_free_v434_train_contract_and_before_"
            "v52_model_gpu_or_nontrain_semantic_access"
        )
        or value.get("gpu_launch_authorized") is not True
        or value.get("protected_semantic_access_authorized") is not False
        or value.get("sealed_holdout_access_authorized") is not False
        or value.get("arms") != arms
        or value.get("runtime") != design.RUNTIME_V52
        or value.get("single_scientific_variable")
        != design.assert_one_scientific_variable_v52(arms)
        or value.get("state_derivations") != design.state_derivations_v52()
        or value.get("compute_plan") != design.compute_plan_v52()
        or value.get("artifacts") != _expected_artifacts_v52()
        or launcher.get("required_python") != str(design.REQUIRED_PYTHON_V52)
        or launcher.get("change_scope")
        != "interpreter_and_fresh_retry_artifact_paths_only"
        or launcher.get("failure_before_model_or_gpu_actor_creation") is not True
        or launcher.get("science_seeds_master_data_and_gates_changed") is not False
        or value.get("retry_science_equivalence", {}).get("byte_equivalent")
        is not True
        or recipe.get("matched_initialization") != str(design.SOURCE_V52)
        or recipe.get("staged_initialization") != str(design.STAGED_V52)
        or recipe.get("source_weights_sha256")
        != design.SOURCE_WEIGHTS_SHA256_V52
        or recipe.get("source_config_sha256")
        != design.SOURCE_CONFIG_SHA256_V52
        or recipe.get("staged_weights_sha256")
        != design.STAGED_WEIGHTS_SHA256_V52
        or recipe.get("staged_manifest_file_sha256")
        != design.STAGED_MANIFEST_FILE_SHA256_V52
        or recipe.get("master_sha256") != design.MASTER_SHA256_V52
        or recipe.get("master_runtime_values_sha256")
        != design.MASTER_RUNTIME_SHA256_V52
        or recipe.get("dataset") != str(design.TRAIN_DATASET_V52)
        or recipe.get("dataset_sha256") != design.DATASET_SHA256_V52
        or recipe.get("train_bundle_content_sha256")
        != design.TRAIN_BUNDLE_CONTENT_SHA256_V52
        or recipe.get("membership") != str(design.TRAIN_MEMBERSHIP_V52)
        or recipe.get("membership_file_sha256")
        != design.MEMBERSHIP_SHA256_V52
        or recipe.get("generation_panel")
        != str(design.TRAIN_GENERATION_PANEL_V52)
        or recipe.get("generation_panel_file_sha256")
        != design.SUBSET_FILE_SHA256_V52
        or recipe.get("request_order_sha256")
        != design.REQUEST_ORDER_SHA256_V52
        or value.get("train_only_selection", {}).get("required_checks")
        != list(design.TRAIN_GATE_NAMES_V52)
        or value.get("ood_first_gate", {}).get(
            "prose_paired_document_bootstrap_95_lcb_minimum"
        ) != 0.0
        or value.get("document_disjoint_shadow_gate", {}).get(
            "required_zero_intersection_keys"
        ) != list(design.EDGE_IDENTITY_KEYS_V52)
        or value.get("stop_go_gates", {}).get("never_open_sealed_holdout")
        is not True
        or value.get("protected_semantics_opened") is not False
        or value.get("sealed_holdout_opened") is not False
    ):
        raise RuntimeError("v52 preregistration contract changed")
    return value


def require_live_interpreter_v52(executable: str | None = None) -> dict:
    observed = Path(executable or sys.executable).absolute()
    if observed != design.REQUIRED_PYTHON_V52:
        raise RuntimeError(
            "V52 live launch requires "
            f"{design.REQUIRED_PYTHON_V52}; observed {observed}"
        )
    return {
        "required_python": str(design.REQUIRED_PYTHON_V52),
        "observed_python": str(observed),
        "matched": True,
    }


def seal_runtime_state_consensus_v52(
    state_derivation: dict, actor_receipts: list[dict],
) -> dict:
    """Require one exact derived candidate/runtime identity on four actors."""
    if len(actor_receipts) != design.ACTORS_V52:
        raise RuntimeError("v52 runtime state actor coverage changed")
    candidates = set()
    runtimes = set()
    for rank, receipt in enumerate(actor_receipts):
        if (
            receipt.get("actor_rank") != rank
            or receipt.get("state_index") != state_derivation["state_index"]
            or receipt.get("seed") != state_derivation["seed"]
            or receipt.get("sigma") != state_derivation["sigma"]
            or receipt.get("sign") != state_derivation["sign"]
            or receipt.get("master_identity", {}).get("sha256")
            != design.MASTER_SHA256_V52
            or receipt.get("direct_from_pinned_fp32_master") is not True
            or receipt.get("cumulative_candidate_delta_used") is not False
        ):
            raise RuntimeError("v52 runtime state derivation changed")
        candidates.add(receipt.get("candidate_identity", {}).get("sha256"))
        runtimes.add(
            receipt.get("materialization", {}).get("runtime_values_sha256")
        )
    if (
        len(candidates) != 1 or len(runtimes) != 1
        or any(not isinstance(item, str) or len(item) != 64
               for item in candidates | runtimes)
    ):
        raise RuntimeError("v52 four-actor state identity consensus failed")
    result = {
        "schema": "four-actor-derived-state-consensus-v52",
        "state_derivation": state_derivation,
        "candidate_identity_sha256": next(iter(candidates)),
        "runtime_values_sha256": next(iter(runtimes)),
        "actors": actor_receipts,
        "scoring_authorized": True,
    }
    result["content_sha256"] = design.canonical_sha256_v52(result)
    return result


def validate_timing_coverage_v52(states: list[dict]) -> dict:
    if len(states) != 32:
        raise RuntimeError("v52 timing requires all 32 states")
    phase_counts = {phase: 0 for phase in design.PHASES_V52}
    for index, state in enumerate(states):
        if state.get("state_index") != index:
            raise RuntimeError("v52 timing state order changed")
        for phase in design.PHASES_V52:
            rows = state.get(phase, {}).get("actors", [])
            if (
                len(rows) != design.ACTORS_V52
                or [row.get("actor_rank") for row in rows]
                != list(range(design.ACTORS_V52))
                or any(
                    not isinstance(row.get("elapsed_ns"), int)
                    or row["elapsed_ns"] < 0 for row in rows
                )
            ):
                raise RuntimeError(f"v52 {phase} timing coverage changed")
            phase_counts[phase] += len(rows)
    result = {
        "schema": "complete-five-phase-actor-timing-v52",
        "states": 32,
        "actors_per_state": 4,
        "phase_actor_receipts": phase_counts,
        "total_actor_phase_receipts": sum(phase_counts.values()),
    }
    if result["total_actor_phase_receipts"] != 640:
        raise RuntimeError("v52 timing receipt total changed")
    return result


def _controller_timing_v52(started_ns: int, ended_ns: int) -> dict:
    return {
        "clock": "controller_monotonic_ns",
        "started_ns": started_ns,
        "ended_ns": ended_ns,
        "elapsed_ns": ended_ns - started_ns,
    }


@dataclass(frozen=True)
class _PendingStateV52:
    state: dict
    transition: dict
    generation: dict
    scoring_handles: Any


def _runtime_state_v52(derivation: dict, transition: dict) -> dict:
    consensus = transition.get("consensus", {})
    result = {
        **derivation,
        "candidate_identity_sha256": consensus.get(
            "candidate_identity_sha256"
        ),
        "runtime_values_sha256": consensus.get("runtime_values_sha256"),
    }
    if any(
        not isinstance(result[key], str) or len(result[key]) != 64
        for key in ("candidate_identity_sha256", "runtime_values_sha256")
    ):
        raise RuntimeError("v52 transition did not seal runtime state")
    return result


def _elided_restore_v52(
    previous: dict, current: dict, transition: dict,
) -> dict:
    actors = []
    transition_actors = transition.get("actors", [])
    if len(transition_actors) != design.ACTORS_V52:
        raise RuntimeError("v52 transition actor coverage changed")
    for rank, receipt in enumerate(transition_actors):
        if (
            receipt.get("actor_rank") != rank
            or receipt.get("previous_candidate_sha256")
            != previous["candidate_identity_sha256"]
            or receipt.get("candidate_identity", {}).get("sha256")
            != current["candidate_identity_sha256"]
            or receipt.get("intermediate_master_restore_elided") is not True
        ):
            raise RuntimeError("v52 elided restore proof changed")
        actors.append({
            "schema": "elided-intermediate-restore-timing-v52",
            "state_index": previous["state_index"],
            "actor_rank": rank,
            "mode": "elided_by_next_direct_pinned_master_transition",
            "elapsed_ns": 0,
            "next_state_index": current["state_index"],
            "previous_candidate_sha256": previous[
                "candidate_identity_sha256"
            ],
            "next_candidate_sha256": current[
                "candidate_identity_sha256"
            ],
            "exact_pinned_master_reconstruction_proved": True,
        })
    return {
        "schema": "elided-intermediate-restore-v52",
        "mode": "elided_by_next_direct_pinned_master_transition",
        "actors": actors,
    }


def _drain_pending_v52(
    pending: _PendingStateV52, restore: dict, operations,
) -> dict:
    actor_scores, score_timing, drain_timing = operations.resolve_scoring(
        pending.state, pending.scoring_handles,
    )
    return {
        "state": pending.state,
        "transition": pending.transition,
        "generation": pending.generation,
        "actor_scores": actor_scores,
        "score_timing": score_timing,
        "restore_timing": restore,
        "drain_timing": drain_timing,
    }


def run_direct_master_pipeline_v52(
    state_derivations: list[dict], operations,
) -> tuple[list[dict], dict]:
    """Run the accepted V51 one-slot schedule with runtime-sealed states."""
    if state_derivations != design.state_derivations_v52():
        raise ValueError("v52 signed state derivations changed")
    pending = None
    completed = []
    previous = None
    transition_attempted = False
    try:
        for derivation in state_derivations:
            transition_attempted = True
            transition = operations.transition(derivation, previous)
            state = _runtime_state_v52(derivation, transition)
            generation_handles = operations.launch_generation(state)
            generation = operations.wait_generation(
                state, generation_handles,
            )
            if pending is not None:
                completed.append(_drain_pending_v52(
                    pending,
                    _elided_restore_v52(
                        pending.state, state, transition,
                    ),
                    operations,
                ))
                pending = None
            scoring_handles = operations.submit_scoring(
                state, generation_handles,
            )
            pending = _PendingStateV52(
                state, transition, generation, scoring_handles,
            )
            previous = state
        final_restore = operations.final_restore("population_complete")
        transition_attempted = False
        if pending is not None:
            completed.append(_drain_pending_v52(
                pending, final_restore, operations,
            ))
            pending = None
        if [item["state"]["state_index"] for item in completed] != list(
            range(32)
        ):
            raise RuntimeError("v52 state completion order changed")
        return completed, final_restore
    except BaseException as error:
        if pending is not None:
            try:
                operations.cancel_scoring(
                    pending.state, pending.scoring_handles,
                )
            except BaseException as cancel_error:
                error.add_note(
                    "v52 scoring cancellation also failed: "
                    f"{type(cancel_error).__name__}: {cancel_error}"
                )
        if transition_attempted:
            try:
                operations.final_restore(
                    f"exception:{type(error).__name__}",
                )
            except BaseException as restore_error:
                combined = RuntimeError(
                    "v52 exact final pinned-master restore failed after "
                    f"{type(error).__name__}: {error}: {restore_error}"
                )
                combined.add_note(traceback.format_exc())
                raise combined from restore_error
        raise


class RayPopulationOperationsV52:
    """V51 generation/CPU-scoring overlap with dynamic V52 state hashes."""

    def __init__(
        self, trainer, bundle, dense_items, requests, anchors,
        master_sha: str, master_runtime_sha: str,
        *, v51, v48b, prior, v40a,
    ) -> None:
        import ray

        if v48b._PREPARED_FRAGILE is None or v48b._SEALED_SUBSET is None:
            raise RuntimeError("v52 immutable scoring inputs are incomplete")
        self.ray = ray
        self.trainer = trainer
        self.master_sha = master_sha
        self.master_runtime_sha = master_runtime_sha
        self.v48b = v48b
        self.prior = prior
        self.v40a = v40a
        self.plan = v48b.fused_requests_v48b(requests, anchors)
        self.params = v48b.sampling_params_for_plan_v48b(self.plan)
        if len(self.plan["requests"]) != 608 or len(self.params) != 608:
            raise RuntimeError("v52 population request coverage changed")
        actor_type = ray.remote(
            num_cpus=1, num_gpus=0, max_restarts=0, max_task_retries=0,
        )(v51.PopulationScoringActorV51)
        self.scorers = [
            actor_type.options(runtime_env={
                "env_vars": {"CUDA_VISIBLE_DEVICES": ""},
            }).remote(
                rank, bundle, dense_items, self.plan, anchors,
                list(v48b._PREPARED_FRAGILE), dict(v48b._SEALED_SUBSET),
            )
            for rank in range(design.ACTORS_V52)
        ]
        identities = ray.get([
            scorer.runtime_identity_v51.remote() for scorer in self.scorers
        ])
        expected = [{
            "schema": "timed-generation-boundary-cpu-scorer-v51",
            "actor_rank": rank,
            "gpu_ids": [],
            "cuda_visible_devices": "",
        } for rank in range(design.ACTORS_V52)]
        if identities != expected:
            self.close()
            raise RuntimeError("v52 CPU scorer identities changed")
        self.last_state_index = -1

    @staticmethod
    def _score_tag(state: dict, actor_rank: int) -> dict:
        return {
            "state_index": state["state_index"],
            "direction": state["direction"],
            "label": state["label"],
            "sign": state["sign"],
            "actor_rank": actor_rank,
        }

    def transition(self, derivation: dict, previous: dict | None) -> dict:
        expected_previous = (
            self.master_sha if previous is None
            else previous["candidate_identity_sha256"]
        )
        started_ns = time.monotonic_ns()
        values = self.trainer._resolve([
            self.trainer.engines[rank].collective_rpc.remote(
                "transition_derived_antithetic_from_pinned_master_v52",
                args=(
                    derivation["state_index"], derivation["seed"],
                    derivation["sigma"], derivation["sign"],
                    self.master_sha, expected_previous,
                ),
            )
            for rank in range(design.ACTORS_V52)
        ])
        ended_ns = time.monotonic_ns()
        if len(values) != design.ACTORS_V52 or any(
            len(item) != 1 for item in values
        ):
            raise RuntimeError("v52 transition actor coverage changed")
        actors = []
        for rank, value in enumerate(values):
            receipt = dict(value[0])
            receipt["actor_rank"] = rank
            receipt["elapsed_ns"] = receipt["timing"]["elapsed_ns"]
            actors.append(receipt)
        consensus = seal_runtime_state_consensus_v52(
            derivation, actors,
        )
        self.last_state_index = derivation["state_index"]
        return {
            "schema": "four-actor-derived-transition-timing-v52",
            "state_index": derivation["state_index"],
            "actors": actors,
            "consensus": consensus,
            "controller": _controller_timing_v52(started_ns, ended_ns),
        }

    def launch_generation(self, state: dict) -> dict:
        started_ns = time.monotonic_ns()
        refs = [
            self.trainer.engines[rank].generate.remote(
                self.plan["requests"], self.params, use_tqdm=False,
                lora_request=self.prior._lora_request(),
            )
            for rank in range(design.ACTORS_V52)
        ]
        return {"refs": refs, "started_ns": started_ns}

    def wait_generation(self, state: dict, handles: dict) -> dict:
        refs = handles.get("refs", [])
        if len(refs) != design.ACTORS_V52:
            raise RuntimeError("v52 generation handle coverage changed")
        by_ref = {ref: rank for rank, ref in enumerate(refs)}
        pending = list(refs)
        ready_ns = {}
        while pending:
            ready, pending = self.ray.wait(
                pending, num_returns=1, fetch_local=False,
            )
            ready_ns[by_ref[ready[0]]] = time.monotonic_ns()
        ended_ns = max(ready_ns.values())
        actors = [{
            "schema": "controller-observed-generation-timing-v52",
            "state_index": state["state_index"],
            "actor_rank": rank,
            "clock": "controller_monotonic_ns",
            "started_ns": handles["started_ns"],
            "ended_ns": ready_ns[rank],
            "elapsed_ns": ready_ns[rank] - handles["started_ns"],
        } for rank in range(design.ACTORS_V52)]
        return {
            "schema": "four-actor-generation-timing-v52",
            "state_index": state["state_index"],
            "actors": actors,
            "controller": _controller_timing_v52(
                handles["started_ns"], ended_ns,
            ),
        }

    def submit_scoring(self, state: dict, generation_handles: dict):
        refs = generation_handles["refs"]
        return [
            self.scorers[rank].score_v51.remote(
                self._score_tag(state, rank), refs[rank],
            )
            for rank in range(design.ACTORS_V52)
        ]

    def resolve_scoring(self, state: dict, scoring_handles):
        if len(scoring_handles) != design.ACTORS_V52:
            raise RuntimeError("v52 scoring handle coverage changed")
        started_ns = time.monotonic_ns()
        by_ref = {ref: rank for rank, ref in enumerate(scoring_handles)}
        pending = list(scoring_handles)
        ready_ns = {}
        while pending:
            ready, pending = self.ray.wait(
                pending, num_returns=1, fetch_local=False,
            )
            ready_ns[by_ref[ready[0]]] = time.monotonic_ns()
        rows = self.ray.get(scoring_handles)
        actors = []
        scores = []
        for rank, row in enumerate(rows):
            if (
                row.get("schema")
                != "timed-generation-boundary-actor-score-v51"
                or row.get("state") != self._score_tag(state, rank)
                or row.get("gpu_ids") != []
                or row.get("score", {}).get("actor_rank") != rank
                or row.get("timing", {}).get("actor_rank") != rank
            ):
                raise RuntimeError("v52 scoring receipt changed")
            timing = dict(row["timing"])
            timing["schema"] = "worker-score-timing-v52"
            actors.append(timing)
            scores.append(row["score"])
        score_timing = {
            "schema": "four-actor-score-timing-v52",
            "state_index": state["state_index"],
            "actors": actors,
        }
        drain_actors = [{
            "schema": "controller-observed-drain-timing-v52",
            "state_index": state["state_index"],
            "actor_rank": rank,
            "clock": "controller_monotonic_ns",
            "started_ns": started_ns,
            "ended_ns": ready_ns[rank],
            "elapsed_ns": ready_ns[rank] - started_ns,
        } for rank in range(design.ACTORS_V52)]
        return scores, score_timing, {
            "schema": "four-actor-drain-timing-v52",
            "state_index": state["state_index"],
            "actors": drain_actors,
            "controller": _controller_timing_v52(
                started_ns, max(ready_ns.values()),
            ),
        }

    def final_restore(self, reason: str) -> dict:
        started_ns = time.monotonic_ns()
        values = self.v40a._rpc_all(
            self.trainer, "restore_pinned_master_v51",
            (self.master_sha, self.master_runtime_sha, reason),
        )
        ended_ns = time.monotonic_ns()
        actors = []
        for rank, value in enumerate(values):
            receipt = dict(value)
            receipt["state_index"] = self.last_state_index
            receipt["actor_rank"] = rank
            receipt["elapsed_ns"] = receipt["timing"]["elapsed_ns"]
            if (
                receipt.get("restored") is not True
                or receipt.get("restored_identity", {}).get("sha256")
                != self.master_sha
                or receipt.get("materialization", {}).get(
                    "runtime_values_sha256"
                ) != self.master_runtime_sha
                or receipt.get("transaction_state_quiescent") is not True
            ):
                raise RuntimeError("v52 final exact restore changed")
            actors.append(receipt)
        if len(actors) != design.ACTORS_V52:
            raise RuntimeError("v52 final restore actor coverage changed")
        return {
            "schema": "four-actor-exact-final-restore-v52",
            "mode": "actual_exact_final_restore",
            "reason": reason,
            "actors": actors,
            "controller": _controller_timing_v52(started_ns, ended_ns),
        }

    def cancel_scoring(self, _state: dict, scoring_handles) -> None:
        for handle in scoring_handles:
            self.ray.cancel(handle, force=False)

    def close(self) -> None:
        for scorer in getattr(self, "scorers", []):
            self.ray.kill(scorer, no_restart=True)


def _common_random_plan_v52(receipts: list[dict], subset: dict) -> dict:
    expected = {
        (direction, sign, actor)
        for direction in range(16)
        for sign in ("plus", "minus")
        for actor in range(design.ACTORS_V52)
    }
    observed = {
        (row.get("direction"), row.get("sign"), row.get("actor_rank"))
        for row in receipts
    }
    if (
        len(receipts) != len(expected)
        or observed != expected
        or any(
            row.get("subset_content_sha256")
            != subset["content_sha256_before_self_field"]
            or row.get("request_order_sha256")
            != subset["request_order_sha256"]
            or row.get("generation_params")
            != subset["common_random_generation_params"]
            for row in receipts
        )
    ):
        raise RuntimeError("v52 common-random population plan changed")
    return {
        "signed_actor_state_receipts": len(receipts),
        "all_use_identical_selected_items_order_and_sampling": True,
        "subset_content_sha256": subset[
            "content_sha256_before_self_field"
        ],
        "request_order_sha256": subset["request_order_sha256"],
        "generation_params": subset["common_random_generation_params"],
    }


def replicated_population_v52(
    trainer, bundle, dense_items, requests, anchors,
    master_sha: str, master_runtime_sha: str,
    fresh_calibration_observed_maximum: float,
    *, v51, v48b, prior, v40a,
) -> dict:
    runtime = RayPopulationOperationsV52(
        trainer, bundle, dense_items, requests, anchors,
        master_sha, master_runtime_sha,
        v51=v51, v48b=v48b, prior=prior, v40a=v40a,
    )
    try:
        completed, final_restore = run_direct_master_pipeline_v52(
            design.state_derivations_v52(), runtime,
        )
    finally:
        runtime.close()
    scores = {
        label: [[None] * design.ACTORS_V52 for _ in range(16)]
        for label in ("plus", "minus")
    }
    receipts = []
    state_timings = []
    for item in completed:
        state = item["state"]
        if len(item["actor_scores"]) != design.ACTORS_V52:
            raise RuntimeError("v52 completed score coverage changed")
        for rank, score in enumerate(item["actor_scores"]):
            scores[state["label"]][state["direction"]][rank] = score
            receipts.append({
                "direction": state["direction"],
                "sign": state["label"],
                "actor_rank": rank,
                "subset_content_sha256": v48b._SEALED_SUBSET[
                    "subset"
                ]["content_sha256_before_self_field"],
                "request_order_sha256": v48b._SEALED_SUBSET[
                    "request_order_sha256"
                ],
                "generation_params": v48b._SEALED_SUBSET[
                    "common_random_generation_params"
                ],
            })
        state_timings.append({
            "state_index": state["state_index"],
            "state": state,
            "materialize": item["transition"],
            "generate": item["generation"],
            "score": item["score_timing"],
            "restore": item["restore_timing"],
            "drain": item["drain_timing"],
        })
    if any(
        item is None for label in scores.values()
        for direction in label for item in direction
    ):
        raise RuntimeError("v52 signed score matrix incomplete")
    timing = validate_timing_coverage_v52(state_timings)
    common = _common_random_plan_v52(
        receipts, v48b._SEALED_SUBSET["subset"],
    )
    arms = {}
    for name, population_size in (("p8", 8), ("p16", 16)):
        sign_scores = design.extract_arm_sign_scores_v52(
            scores, population_size,
        )
        central = prior._central_replicates(sign_scores["domain"])
        reliability = design.reliability_gate_v52(
            central, fresh_calibration_observed_maximum,
        )
        if reliability["passed"] is not True:
            raise RuntimeError(f"v52 {name} population reliability failed")
        projection = design.project_arm_v52(scores, population_size)
        arms[name] = {
            "population_size": population_size,
            "reliability": reliability,
            "projection": projection,
            "scale_plans": design.scale_plans_v52(projection),
        }
    return {
        "schema": "nested-direct-pinned-master-population-v52",
        "status": "complete_before_any_optimizer_update",
        "signed_scores": scores,
        "arms": arms,
        "common_random_plan": common,
        "timing": {
            "coverage": timing,
            "states": state_timings,
            "final_restore": final_restore,
        },
        "intermediate_master_restores_eliminated": 31,
        "actual_final_restore_actor_receipts": 4,
        "all_state_identities_sealed_by_four_actor_runtime_consensus": True,
        "final_exact_restore_passed": True,
        "protected_semantics_opened": False,
        "sealed_holdout_opened": False,
    }


def validate_final_aggregate_v52(value: dict) -> dict:
    """Validate aggregate-only stop/go evidence without raw protected rows."""
    if (
        value.get("raw_questions_answers_or_generations_persisted") is not False
        or value.get("sealed_holdout_opened") is not False
        or value.get("heldout_or_holdout_opened") is not False
    ):
        raise RuntimeError("v52 aggregate attempted protected/raw persistence")
    arms = {}
    for name in ("p8", "p16"):
        arm = value.get("arms", {}).get(name, {})
        train_results = arm.get("train_scale_results", [])
        ratio = design.selected_train_ratio_v52(train_results)
        ood_eligible = bool(ratio is not None and design.ood_eligible_v52(
            arm.get("ood_gate", {})
        ))
        shadow = arm.get("shadow_gate", {})
        shadow_disjoint = (
            design.document_disjoint_shadow_eligible_v52(shadow)
            if ood_eligible else False
        )
        arms[name] = {
            "selected_train_ratio": ratio,
            "ood_eligible": ood_eligible,
            "shadow_document_disjoint": shadow_disjoint,
            "shadow_better_than_master": bool(
                shadow_disjoint and shadow.get("better_than_master") is True
            ),
            "shadow_better_than_p8": bool(
                shadow_disjoint and shadow.get("better_than_p8") is True
            ),
        }
    result = {
        "schema": "nested-population-final-decision-v52",
        "p8": arms["p8"],
        "p16": arms["p16"],
        "population_size_hypothesis_passed": design.treatment_success_v52(arms),
        "sealed_holdout_opened": False,
    }
    result["content_sha256"] = design.canonical_sha256_v52(result)
    return result


def _expected_update_manifest_v52(
    prior, master: dict, reference_generation: int, update_sequence: int,
    seeds: list[int], coefficients: list[float], plan_id: str,
) -> tuple[str, str]:
    coefficient_sha = prior.worker_v3.coefficient_sha256_v3(
        seeds, coefficients,
    )
    manifest = {
        "schema": "canonical-lora-sharded-update-manifest-v41a",
        "seeds": seeds,
        "coefficients": coefficients,
        "coefficient_sha256": coefficient_sha,
        "population_size": len(seeds),
        "world_size": design.ACTORS_V52,
        "alpha": design.ALPHA_V52,
        "plan_id": plan_id,
        "expected_master_sha256": master["sha256"],
        "reference_generation": int(reference_generation),
        "update_sequence": int(update_sequence) + 1,
    }
    return prior.worker_v3.canonical_sha256_v3(manifest), coefficient_sha


def _prepare_execute_update_v52(
    trainer, prior, v40a, master: dict, reference_generation: int,
    update_sequence: int, seeds: list[int], coefficients: list[float],
    plan_id: str,
) -> dict:
    expected_manifest, coefficient_sha = _expected_update_manifest_v52(
        prior, master, reference_generation, update_sequence,
        seeds, coefficients, plan_id,
    )
    prepared = v40a._rpc_all(
        trainer, "prepare_sharded_adapter_update_v41a", (
            seeds, coefficients, coefficient_sha, len(seeds),
            design.ACTORS_V52, design.ALPHA_V52, plan_id,
            master["sha256"], reference_generation,
        ),
    )
    if (
        {item.get("manifest_sha256") for item in prepared}
        != {expected_manifest}
        or {item.get("rank") for item in prepared} != {0, 1, 2, 3}
        or len({
            design.canonical_sha256_v52(item.get("master_identity"))
            for item in prepared
        }) != 1
    ):
        raise RuntimeError("v52 prepared update consensus changed")
    executed = v40a._rpc_all(
        trainer, "execute_sharded_adapter_update_v41a",
        (expected_manifest,),
    )
    identities = [item["candidate_identity"] for item in executed]
    if len({design.canonical_sha256_v52(item) for item in identities}) != 1:
        raise RuntimeError("v52 update candidate differs across ranks")
    runtime_hashes = {
        item["materialization"]["runtime_values_sha256"]
        for item in executed
    }
    if len(runtime_hashes) != 1:
        raise RuntimeError("v52 update runtime differs across ranks")
    return {
        "coefficient_sha256": coefficient_sha,
        "manifest_sha256": expected_manifest,
        "prepared": prepared,
        "executed": executed,
        "candidate_identity": identities[0],
        "candidate_runtime_values_sha256": next(iter(runtime_hashes)),
        "master_committed": False,
    }


def _validate_abort_v52(aborted: dict, master: dict, runtime_sha: str) -> None:
    if (
        aborted.get("all_four_ranks_exact") is not True
        or aborted.get("restored_master_identity") != master
        or aborted.get("restored_runtime_values_sha256") != runtime_sha
        or len(aborted.get("workers", [])) != design.ACTORS_V52
        or len(aborted.get("state_certificates", []))
        != design.ACTORS_V52
    ):
        raise RuntimeError("v52 exact abort/readback changed")


def _save_pending_snapshot_v52(
    trainer, v40a, transaction: dict, output: Path,
) -> dict:
    snapshots = v40a._rpc_all(
        trainer, "save_pending_candidate_snapshot_v52", (
            str(output), transaction["manifest_sha256"],
            transaction["candidate_identity"]["sha256"],
            transaction["candidate_runtime_values_sha256"],
        ),
    )
    written = [item for item in snapshots if item.get("written") is True]
    if (
        len(written) != 1
        or written[0].get("rank") != 0
        or written[0].get("readback_verified") is not True
        or written[0].get("readback_identity")
        != transaction["candidate_identity"]
        or any(
            item.get("candidate_identity")
            != transaction["candidate_identity"]
            or item.get("materialization", {}).get(
                "runtime_values_sha256"
            ) != transaction["candidate_runtime_values_sha256"]
            or item.get("master_committed") is not False
            for item in snapshots
        )
    ):
        raise RuntimeError("v52 pending candidate snapshot consensus changed")
    return {
        "schema": "four-actor-uncommitted-snapshot-consensus-v52",
        "directory": str(output),
        "candidate_identity": transaction["candidate_identity"],
        "runtime_values_sha256": transaction[
            "candidate_runtime_values_sha256"
        ],
        "workers": snapshots,
        "written_rank": 0,
        "readback_verified": True,
        "master_committed": False,
    }


def _candidate_consensus_v52(
    trainer, prior, bundle, dense_items, requests, numeric_bounds,
) -> dict:
    records = prior._score_repeats(
        trainer, bundle, dense_items, requests,
        warmups=0, retained=prior.numeric.POST_UPDATE_REPEATS_V43G,
    )
    equivalence = prior.numeric.post_update_consensus_v43g(
        records, numeric_bounds,
    )
    return {
        "schema": "uncommitted-candidate-consensus-v52",
        "records": records,
        "equivalence": equivalence,
        "protected_semantics_opened": False,
    }


def evaluate_train_arm_v52(
    arm_name: str, trainer, prior, v40a, bundle, dense_items, requests,
    full_plan: dict, full_anchors: dict, reference_actors: list[dict],
    population_arm: dict, anchor_margins: dict, numeric_bounds: dict,
    master: dict, master_runtime_sha: str, reference_generation: int,
    update_sequence: int, snapshot: Path, transaction_tracker: dict,
) -> dict:
    population_size = population_arm["population_size"]
    seeds = list(design.P16_SEEDS_V52[:population_size])
    scale_results = []
    selected_snapshot = None
    for scale_plan in population_arm["scale_plans"]:
        ratio = scale_plan["target_norm_ratio"]
        plan_id = design.canonical_sha256_v52({
            "schema": "nested-population-update-plan-v52",
            "arm": arm_name,
            "population_size": population_size,
            "master_sha256": master["sha256"],
            "seeds": seeds,
            "scale_plan": scale_plan,
            "alpha": design.ALPHA_V52,
            "preference": "largest_strictly_passing_ratio",
        })
        expected_manifest, _ = _expected_update_manifest_v52(
            prior, master, reference_generation, update_sequence,
            seeds, scale_plan["coefficients"], plan_id,
        )
        transaction_tracker["value"] = {
            "manifest_sha256": expected_manifest,
        }
        transaction = _prepare_execute_update_v52(
            trainer, prior, v40a, master, reference_generation,
            update_sequence, seeds, scale_plan["coefficients"], plan_id,
        )
        transaction_tracker["value"] = transaction
        candidate_actors = prior._generate_fused_actor_scores(
            trainer, bundle, dense_items, full_plan, full_anchors,
        )
        gate = prior.fused.candidate_gate_v43i(
            reference_actors, candidate_actors, anchor_margins,
        )
        if set(gate.get("checks", {})) != set(
            design.TRAIN_GATE_NAMES_V52
        ):
            raise RuntimeError("v52 candidate gate inventory changed")
        consensus = None
        candidate_consensus_passed = False
        if gate.get("passed") is True and all(gate["checks"].values()):
            consensus = _candidate_consensus_v52(
                trainer, prior, bundle, dense_items, requests,
                numeric_bounds,
            )
            candidate_consensus_passed = (
                consensus["equivalence"].get("passed") is True
            )
        snapshot_receipt = None
        if candidate_consensus_passed:
            snapshot_receipt = _save_pending_snapshot_v52(
                trainer, v40a, transaction, snapshot,
            )
        aborted = prior._exact_abort_transaction(
            trainer, transaction["manifest_sha256"],
            master, master_runtime_sha,
        )
        _validate_abort_v52(aborted, master, master_runtime_sha)
        transaction_tracker["value"] = None
        passed = bool(
            gate.get("passed") is True
            and all(gate["checks"].values())
            and candidate_consensus_passed
        )
        scale_results.append({
            "target_norm_ratio": ratio,
            "scale_plan": scale_plan,
            "candidate_identity": transaction["candidate_identity"],
            "candidate_runtime_values_sha256": transaction[
                "candidate_runtime_values_sha256"
            ],
            "reference_actors": reference_actors,
            "candidate_actors": candidate_actors,
            "gate": gate,
            "checks": gate["checks"],
            "candidate_consensus": consensus,
            "candidate_consensus_passed": candidate_consensus_passed,
            "snapshot": snapshot_receipt,
            "exact_abort": aborted,
            "exact_abort_readback_passed": True,
            "passed": passed,
        })
        selected = design.selected_train_ratio_v52(scale_results)
        if passed:
            if selected != ratio or snapshot_receipt is None:
                raise RuntimeError("v52 passing scale selection changed")
            selected_snapshot = snapshot_receipt
            break
        if selected is not None:
            raise RuntimeError("v52 rejected scale selected unexpectedly")
    selected_ratio = design.selected_train_ratio_v52(scale_results)
    return {
        "schema": "nested-population-train-gate-v52",
        "arm": arm_name,
        "population_size": population_size,
        "seeds": seeds,
        "population_projection_content_sha256": population_arm[
            "projection"
        ]["content_sha256"],
        "reliability_content_sha256": population_arm[
            "reliability"
        ]["content_sha256"],
        "scale_results": scale_results,
        "selected_target_norm_ratio": selected_ratio,
        "selected_snapshot": selected_snapshot,
        "all_candidates_exactly_aborted_to_common_master": True,
        "master_committed": False,
        "protected_semantics_opened": False,
        "sealed_holdout_opened": False,
    }


@contextmanager
def patched_runtime_v52(prior):
    values = {
        "EXPERIMENT": "v52_matched_lora_es_nested_p8_vs_p16_retry1",
        "RUN_DIR": RUN_DIR,
        "ATTEMPT": ATTEMPT,
        "REPORT": RUN_DIR / "nested_population_report_v52.json",
        "GPU_LOG": RUN_DIR / "gpu_activity_v52.jsonl",
        "SNAPSHOT": P16_SNAPSHOT,
        "CALIBRATION_ARTIFACT": NUMERIC_CALIBRATION,
        "ANCHOR_CALIBRATION_ARTIFACT": ANCHOR_CALIBRATION,
        "WORKER_EXTENSION": WORKER_EXTENSION_V52,
        "SOURCE": design.SOURCE_V52,
        "SOURCE_WEIGHTS": design.SOURCE_WEIGHTS_V52,
        "SOURCE_CONFIG": design.SOURCE_CONFIG_V52,
        "STAGED": design.STAGED_V52,
        "STAGED_WEIGHTS": design.STAGED_WEIGHTS_V52,
        "STAGED_CONFIG": design.STAGED_CONFIG_V52,
        "STAGED_MANIFEST": design.STAGED_MANIFEST_V52,
        "DATASET": design.TRAIN_DATASET_V52,
        "DATASET_SHA256": design.DATASET_SHA256_V52,
        "SPLIT_MANIFEST": design.TRAIN_MEMBERSHIP_V52,
        "SPLIT_MANIFEST_SHA256": design.MEMBERSHIP_SHA256_V52,
        "TRAIN_BUNDLE_SHA256": design.TRAIN_BUNDLE_CONTENT_SHA256_V52,
    }
    saved = {key: getattr(prior, key) for key in values}
    for key, value in values.items():
        setattr(prior, key, value)
    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(prior, key, value)


def _persist_v52(prior, path: Path, value: dict) -> dict:
    return prior._persist_phase(path, value)


def _execute_v52(preregistration: dict) -> int:
    """Execute train-only V52; protected OOD/shadow evaluation stays separate."""
    import run_lora_es_generation_boundary_v48b as v48b
    import run_lora_es_transition_microbenchmark_v51 as v51

    prior = v48b.v43i
    v40a = prior.v40a
    population_path = RUN_DIR / "nested_population_v52.json"
    p8_gate_path = RUN_DIR / "p8_train_gate_v52.json"
    p16_gate_path = RUN_DIR / "p16_train_gate_v52.json"
    report_path = RUN_DIR / "nested_population_report_v52.json"
    gpu_log = RUN_DIR / "gpu_activity_v52.jsonl"
    failure_path = RUN_DIR / "failure_v52.json"
    if ATTEMPT.exists() or RUN_DIR.exists():
        raise RuntimeError("v52 requires fresh artifact paths")
    preflight = v40a.gpu_preflight()
    attempt = v40a.self_hashed({
        "schema": "nested-population-attempt-v52",
        "status": "launching_train_only",
        "phase": "before_train_semantics_model_or_gpu_load",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_content_sha256": preregistration[
            "content_sha256_before_self_field"
        ],
        "launcher": require_live_interpreter_v52(),
        "preflight": preflight,
        "protected_semantics_opened": False,
        "sealed_holdout_opened": False,
    })
    v40a.atomic_json(ATTEMPT, attempt)
    RUN_DIR.mkdir(parents=True)
    trainer = monitor = saved = None
    master = None
    master_runtime_sha = None
    transaction_tracker = {"value": None}
    emergency_abort = emergency_restore = None
    stop = threading.Event()
    failures: queue.Queue = queue.Queue()
    phase = v40a.Phase()
    started = time.monotonic()
    p8_gate = p16_gate = None
    numeric_calibration = anchor_calibration = None
    post_numeric_calibration = post_anchor_calibration = None
    adapter_contract = None
    cleanup = idle = gpu = None
    try:
        subset = load_generation_panel_v52()
        if subset["request_order_sha256"] != design.REQUEST_ORDER_SHA256_V52:
            raise RuntimeError("v52 frozen request order changed")
        v48b._SEALED_SUBSET = subset
        with v48b.patched_v43i_v48b(), patched_runtime_v52(prior):
            adapter_contract = verify_adapter_contract_v52()
            bundle = load_train_bundle_v52()
            if bundle["content_sha256_before_self_field"] != (
                design.TRAIN_BUNDLE_CONTENT_SHA256_V52
            ):
                raise RuntimeError("v52 frozen train bundle changed")
            bundle = augment_unit_membership_v52(bundle)
            anchor_bundle = prior.fused.load_anchor_bundle_v43i(
                prior.PROSE_ANCHOR, prior.PROSE_REPORT,
                prior.QA_ANCHOR, prior.QA_REPORT,
            )
            v40a.base.set_seed(prior.GLOBAL_SEED)
            trainer, saved = prior._make_trainer(preregistration)
            actor_ids = trainer._resolve([
                engine.runtime_identity_v40a.remote()
                for engine in trainer.engines
            ])
            pid_map = prior.prior._actor_pid_map(actor_ids)
            monitor = threading.Thread(
                target=v40a.monitor_gpus,
                args=(stop, phase, pid_map, gpu_log, failures),
                daemon=True,
            )
            monitor.start()
            dense_items, requests, panel_anchors, full_anchors = (
                v48b.prepare_v48b(trainer, bundle, anchor_bundle)
            )
            phase.value = "activate_matched_initialization"
            preinstall = prior._exact_base_score(
                trainer, bundle, dense_items, requests,
            )
            phase.value = "install_canonical_master"
            installations = v40a._rpc_all(
                trainer, "install_adapter_state_v41a", (
                    str(prior.SOURCE_WEIGHTS), str(prior.SOURCE_CONFIG),
                    v40a.file_sha256(prior.SOURCE_WEIGHTS),
                    v40a.file_sha256(prior.SOURCE_CONFIG),
                ),
            )
            certificates = v40a._rpc_all(
                trainer, "adapter_state_certificate_v41a",
            )
            masters = [item["current_identity"] for item in certificates]
            runtime_hashes = {
                item["materialization"]["runtime_values_sha256"]
                for item in certificates
            }
            if (
                any(
                    item.get("sha256") != design.MASTER_SHA256_V52
                    for item in masters
                )
                or runtime_hashes != {design.MASTER_RUNTIME_SHA256_V52}
            ):
                raise RuntimeError("v52 installed master identity changed")
            master = masters[0]
            master_runtime_sha = next(iter(runtime_hashes))
            reference_generations = {
                item["reference_generation"] for item in certificates
            }
            update_sequences = {
                item["update_sequence"] for item in certificates
            }
            if len(reference_generations) != 1 or update_sequences != {0}:
                raise RuntimeError("v52 initial transaction sequence changed")
            references = v40a._rpc_all(
                trainer, "capture_adapter_reference_v41a",
            )
            reference_generation = references[0]["reference_generation"]
            if {item["reference_generation"] for item in references} != {
                reference_generation
            }:
                raise RuntimeError("v52 reference generation differs by rank")
            postinstall = prior._exact_base_score(
                trainer, bundle, dense_items, requests,
            )
            if preinstall["consensus"] != postinstall["consensus"]:
                raise RuntimeError("v52 canonical install changed base score")

            phase.value = "shared_v434_numeric_calibration"
            numeric_calibration = prior._calibrate_numeric_path(
                trainer, bundle, dense_items, requests, master["sha256"],
            )
            numeric_bounds = numeric_calibration["bootstrap"]
            fresh_calibration_maximum = numeric_bounds[
                "equal_unit_mean"
            ]["observed_maximum_repeat_actor_spread"]
            post_numeric_calibration = prior._exact_base_score(
                trainer, bundle, dense_items, requests,
            )
            if post_numeric_calibration["consensus"] != postinstall["consensus"]:
                raise RuntimeError("v52 numeric calibration restore changed base score")

            phase.value = "shared_v434_anchor_calibration"
            anchor_calibration = prior._calibrate_anchor_path(
                trainer, full_anchors, master["sha256"], master_runtime_sha,
            )
            anchor_margins = anchor_calibration["calibrated_margins"]
            post_anchor_calibration = prior._exact_base_score(
                trainer, bundle, dense_items, requests,
            )
            if post_anchor_calibration["consensus"] != postinstall["consensus"]:
                raise RuntimeError("v52 anchor calibration restore changed base score")

            phase.value = "nested_p16_population_v52"
            population = replicated_population_v52(
                trainer, bundle, dense_items, requests, panel_anchors,
                master["sha256"], master_runtime_sha,
                fresh_calibration_maximum,
                v51=v51, v48b=v48b, prior=prior, v40a=v40a,
            )
            population_artifact = _persist_v52(
                prior, population_path, population,
            )
            phase.value = "verify_post_population_master"
            post_population = prior._exact_base_score(
                trainer, bundle, dense_items, requests,
            )
            if post_population["consensus"] != postinstall["consensus"]:
                raise RuntimeError("v52 population changed exact master score")

            full_plan = prior.fused.fused_requests_v43i(
                requests, full_anchors,
            )
            if len(full_plan["requests"]) != 896:
                raise RuntimeError("v52 full candidate request coverage changed")
            phase.value = "full_candidate_reference_score"
            reference_actors = prior._generate_fused_actor_scores(
                trainer, bundle, dense_items, full_plan, full_anchors,
            )
            phase.value = "p8_backtracking_train_gate"
            p8_gate = evaluate_train_arm_v52(
                "p8", trainer, prior, v40a, bundle, dense_items, requests,
                full_plan, full_anchors, reference_actors,
                population["arms"]["p8"], anchor_margins, numeric_bounds,
                master, master_runtime_sha, reference_generation, 0,
                P8_SNAPSHOT, transaction_tracker,
            )
            p8_artifact = _persist_v52(prior, p8_gate_path, p8_gate)
            phase.value = "p16_backtracking_train_gate"
            p16_gate = evaluate_train_arm_v52(
                "p16", trainer, prior, v40a, bundle, dense_items, requests,
                full_plan, full_anchors, reference_actors,
                population["arms"]["p16"], anchor_margins, numeric_bounds,
                master, master_runtime_sha, reference_generation, 0,
                P16_SNAPSHOT, transaction_tracker,
            )
            p16_artifact = _persist_v52(prior, p16_gate_path, p16_gate)

            phase.value = "verify_final_exact_master"
            final_score = prior._exact_base_score(
                trainer, bundle, dense_items, requests,
            )
            final_certificates = v40a._rpc_all(
                trainer, "adapter_state_certificate_v41a",
            )
            if (
                final_score["consensus"] != postinstall["consensus"]
                or any(
                    item["current_identity"] != master
                    or item["materialization"]["runtime_values_sha256"]
                    != master_runtime_sha
                    or item["update_sequence"] != 0
                    or item["reference_fresh"] is not True
                    for item in final_certificates
                )
            ):
                raise RuntimeError("v52 final master reconstruction changed")
            stop.set()
            monitor.join(timeout=10)
            if monitor.is_alive() or not failures.empty():
                raise RuntimeError("v52 GPU monitor failed") from (
                    failures.get() if not failures.empty() else None
                )
            gpu = v40a.summarize_gpu(gpu_log, pid_map)
            cleanup = v40a.cleanup_v38a.strict_close_trainer_v38a(trainer)
            trainer = None
            import ray
            ray.shutdown()
            idle = v40a.cleanup_v38a.wait_for_gpu_idle()

        passing = [
            name for name, gate in (("p8", p8_gate), ("p16", p16_gate))
            if gate["selected_target_norm_ratio"] is not None
        ]
        report = v40a.self_hashed({
            "schema": "nested-population-train-only-report-v52",
            "status": (
                "complete_candidates_saved_for_separate_ood_first_evaluation"
                if passing
                else "complete_no_train_gate_candidate_no_protected_access"
            ),
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_runtime_seconds": time.monotonic() - started,
            "preregistration_content_sha256": preregistration[
                "content_sha256_before_self_field"
            ],
            "installations": installations,
            "equal_v434_adapter_contract": adapter_contract,
            "initial_master_identity": master,
            "initial_runtime_values_sha256": master_runtime_sha,
            "initial_references": references,
            "score_audits": {
                "preinstall": preinstall,
                "postinstall": postinstall,
                "post_numeric_calibration": post_numeric_calibration,
                "post_anchor_calibration": post_anchor_calibration,
                "post_population": post_population,
                "final": final_score,
                "exact_master_score_preserved": True,
            },
            "final_state_certificates": final_certificates,
            "shared_fresh_calibration": {
                "numeric": {
                    "path": str(NUMERIC_CALIBRATION),
                    "file_sha256": v40a.file_sha256(NUMERIC_CALIBRATION),
                    "content_sha256": numeric_calibration[
                        "content_sha256_before_self_field"
                    ],
                },
                "anchor": {
                    "path": str(ANCHOR_CALIBRATION),
                    "file_sha256": v40a.file_sha256(ANCHOR_CALIBRATION),
                    "content_sha256": anchor_calibration[
                        "content_sha256_before_self_field"
                    ],
                },
                "master_sha256": master["sha256"],
                "shared_by_p8_and_p16": True,
                "historical_v48e_calibration_values_reused": False,
            },
            "population": {
                "path": str(population_path),
                "file_sha256": v40a.file_sha256(population_path),
                "content_sha256": population_artifact[
                    "content_sha256_before_self_field"
                ],
                "timing_coverage": population["timing"]["coverage"],
            },
            "train_gates": {
                "p8": {
                    "path": str(p8_gate_path),
                    "file_sha256": v40a.file_sha256(p8_gate_path),
                    "content_sha256": p8_artifact[
                        "content_sha256_before_self_field"
                    ],
                    "selected_target_norm_ratio": p8_gate[
                        "selected_target_norm_ratio"
                    ],
                },
                "p16": {
                    "path": str(p16_gate_path),
                    "file_sha256": v40a.file_sha256(p16_gate_path),
                    "content_sha256": p16_artifact[
                        "content_sha256_before_self_field"
                    ],
                    "selected_target_norm_ratio": p16_gate[
                        "selected_target_norm_ratio"
                    ],
                },
            },
            "train_gate_passing_arms": passing,
            "optimizer_master_committed": False,
            "all_candidates_exactly_aborted_to_common_master": True,
            "ood_first_evaluation_required_separately": bool(passing),
            "ood_or_shadow_evaluation_performed": False,
            "quality_or_promotion_conclusion_authorized": False,
            "gpu_activity": gpu,
            "cleanup": cleanup,
            "final_gpu_idle": idle,
            "gpu_log": {
                "path": str(gpu_log),
                "file_sha256": v40a.file_sha256(gpu_log),
            },
            "protected_semantics_opened": False,
            "sealed_holdout_opened": False,
        })
        v40a.atomic_json(report_path, report)
        print(json.dumps({
            "report": str(report_path),
            "report_sha256": v40a.file_sha256(report_path),
            "report_content_sha256": report[
                "content_sha256_before_self_field"
            ],
            "train_gate_passing_arms": passing,
            "optimizer_master_committed": False,
            "protected_semantics_opened": False,
        }, sort_keys=True))
        return 0
    except BaseException as error:
        emergency_abort_failure = emergency_restore_failure = None
        transaction = transaction_tracker.get("value")
        if (
            trainer is not None and transaction is not None
            and master is not None and master_runtime_sha is not None
        ):
            try:
                emergency_abort = prior._exact_abort_transaction(
                    trainer, transaction["manifest_sha256"],
                    master, master_runtime_sha,
                )
                transaction_tracker["value"] = None
            except BaseException as abort_error:
                emergency_abort_failure = {
                    "type": type(abort_error).__name__,
                    "message": str(abort_error),
                    "traceback": traceback.format_exc(),
                }
        if (
            trainer is not None and master is not None
            and master_runtime_sha is not None
            and emergency_abort_failure is None
        ):
            try:
                emergency_restore = v40a._rpc_all(
                    trainer, "restore_pinned_master_v51", (
                        master["sha256"], master_runtime_sha,
                        f"controller_exception:{type(error).__name__}",
                    ),
                )
            except BaseException as restore_error:
                emergency_restore_failure = {
                    "type": type(restore_error).__name__,
                    "message": str(restore_error),
                    "traceback": traceback.format_exc(),
                }
        stop.set()
        if monitor is not None:
            monitor.join(timeout=10)
        cleanup_failure = None
        if trainer is not None:
            try:
                cleanup = v40a.cleanup_v38a.strict_close_trainer_v38a(
                    trainer
                )
                trainer = None
                import ray
                ray.shutdown()
                idle = v40a.cleanup_v38a.wait_for_gpu_idle()
            except BaseException as cleanup_error:
                cleanup_failure = {
                    "type": type(cleanup_error).__name__,
                    "message": str(cleanup_error),
                    "traceback": traceback.format_exc(),
                }
        v40a.atomic_json(failure_path, v40a.self_hashed({
            "schema": "nested-population-failure-v52",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "p8_train_gate": p8_gate,
            "p16_train_gate": p16_gate,
            "emergency_exact_abort": emergency_abort,
            "emergency_exact_abort_failure": emergency_abort_failure,
            "emergency_exact_restore": emergency_restore,
            "emergency_exact_restore_failure": emergency_restore_failure,
            "strict_cleanup": cleanup,
            "strict_cleanup_failure": cleanup_failure,
            "final_gpu_idle": idle,
            "protected_semantics_opened": False,
            "sealed_holdout_opened": False,
        }))
        raise
    finally:
        if trainer is not None:
            try:
                v40a.base.close_trainer(trainer)
            except Exception:
                pass
        try:
            import ray
            ray.shutdown()
        except Exception:
            pass
        if saved is not None:
            v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION = saved


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.dry_run == args.execute:
        raise ValueError("v52 requires exactly one of --dry-run or --execute")
    prereg = load_preregistration_v52(args)
    if args.dry_run:
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "sole_scientific_variable": "antithetic_direction_population_size",
            "population_arms": [8, 16],
            "states": 32,
            "v51_schedule_only": True,
            "train_semantics_loaded": False,
            "protected_semantics_loaded": False,
            "model_imported_or_loaded": False,
            "ray_imported_or_initialized": False,
            "gpu_queried_or_loaded": False,
            "filesystem_writes": False,
            "sealed_holdout_opened": False,
        }, sort_keys=True))
        return 0
    require_live_interpreter_v52()
    return _execute_v52(prereg)


if __name__ == "__main__":
    raise SystemExit(main())
