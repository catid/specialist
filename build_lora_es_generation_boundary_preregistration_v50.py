#!/usr/bin/env python3
"""Seal the V50 scheduling-only derivative of the frozen V48B run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import run_lora_es_generation_boundary_v50 as runtime


def _copy(value):
    return json.loads(json.dumps(value))


def build_v50() -> dict:
    parent = runtime._load_parent_v50()
    value = {
        "schema": "matched-lora-es-generation-boundary-preregistration-v50",
        "status": "preregistered_before_train_only_launch",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_launch_authorized": True,
        "purpose": (
            "Overlap frozen V48B per-actor CPU scoring with the next signed "
            "state's generation using a one-state bounded buffer, without "
            "changing requests, objectives, exact restores, or gates."
        ),
        "protected_semantic_access_authorized": False,
        "shadow_ood_holdout_or_benchmark_authorized": False,
        "quality_selection_or_promotion_authorized": False,
        "access_contract": _copy(parent["access_contract"]),
        "parents": {
            "v48b_preregistration": {
                "path": str(runtime.PARENT_PREREGISTRATION),
                "file_sha256": (
                    runtime.PARENT_PREREGISTRATION_FILE_SHA256
                ),
                "content_sha256": (
                    runtime.PARENT_PREREGISTRATION_CONTENT_SHA256
                ),
                "objective_and_gate_contract_inherited_exactly": True,
            },
        },
        "recipe": _copy(parent["recipe"]),
        "generation_boundary_objective": _copy(
            parent["generation_boundary_objective"]
        ),
        "uncommitted_candidate_gate": _copy(
            parent["uncommitted_candidate_gate"]
        ),
        "runtime": _copy(parent["runtime"]),
        "population_scoring_schedule": {
            "cpu_scorer_actors": runtime.SCORER_ACTORS,
            "cpus_per_scorer_actor": 1,
            "gpus_per_scorer_actor": 0,
            "max_outstanding_scoring_states": (
                runtime.MAX_OUTSTANDING_SCORING_STATES
            ),
            "state_order": (
                "direction ascending; plus then minus; actor rank ascending"
            ),
            "restore_verified_before_next_materialization": True,
            "completion_order_affects_placement": False,
            "objective_or_gate_changed": False,
            "generation_outputs_passed_by_object_reference": True,
            "scorer_inputs_initialized_once_and_immutable": True,
        },
        "implementation_bindings": runtime.implementation_bindings_v50(),
        "artifacts": {
            "attempt": str(runtime.ATTEMPT),
            "run_directory": str(runtime.RUN_DIR),
            "report": str(runtime.REPORT),
            "gpu_log": str(runtime.GPU_LOG),
            "snapshot": str(runtime.SNAPSHOT),
            "population": str(runtime.RELIABILITY_ARTIFACT),
            "candidate_gate": str(runtime.CANDIDATE_GATE_ARTIFACT),
            "exact_abort": str(runtime.ABORT_ARTIFACT),
        },
        "required_gates": {
            **_copy(parent["required_gates"]),
            "one_scoring_state_queue_bound": True,
            "all_state_restores_verified_before_successor_materialization": True,
            "deterministic_actor_rank_placement": True,
            "cpu_scorers_have_zero_gpu_visibility": True,
            "V48B_population_numeric_equivalence": True,
        },
        "protected_semantics_opened": False,
        "shadow_ood_holdout_or_benchmark_opened": False,
        "current_fixed_holdout_cycle_eligible": False,
    }
    value["content_sha256_before_self_field"] = (
        runtime.v48b.v43i.v40a.canonical_sha256(value)
    )
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(runtime.PREREGISTRATION))
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_v50()
    runtime.v48b.v43i.v40a.atomic_json(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": runtime.v48b.v43i.v40a.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "gpu_launch_authorized": True,
        "objective_or_gate_changed": False,
        "protected_semantics_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
