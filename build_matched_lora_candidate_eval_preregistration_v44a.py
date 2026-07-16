#!/usr/bin/env python3
"""Freeze V44A before its one semantic read of shadow-dev and OOD gates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import run_matched_lora_candidate_eval_v44a as runtime


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_sft_hpo_es_fold3_ood_eval_v44a.json"
).resolve()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return result


def build() -> dict:
    stages = runtime.staged_adapter_bindings_v44a()
    if tuple(stages) != runtime.CANDIDATE_ARMS:
        raise RuntimeError("V44A staged candidate order changed")
    value = {
        "schema": "matched-lora-candidate-eval-preregistration-v44a",
        "status": "preregistered_before_single_semantic_access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_question": (
            "Which matched-init equal-unit LoRA update is best on document-"
            "disjoint fold-3 shadow-dev without QA or prose OOD degradation?"
        ),
        "arms": list(runtime.ARMS),
        "base_duplicate_arms": list(runtime.BASE_ARMS),
        "candidate_arms": list(runtime.CANDIDATE_ARMS),
        "candidate_definitions": [
            {
                "arm": arm,
                "adapter_id": runtime.ADAPTER_IDS_V44A[arm],
                "staged_directory": str(runtime.STAGED_BY_ARM[arm]),
                "source_weights_sha256": (
                    runtime.staging.EXPECTED_V44A[arm]["weights"]
                ),
                "training_family": (
                    "LoRA-ES" if arm == "lora_es_v43d" else "matched SFT"
                ),
            }
            for arm in runtime.CANDIDATE_ARMS
        ],
        "implementation_bindings": runtime.implementation_bindings_v44a(),
        "staged_adapters": stages,
        # These are prior content commitments only. build() intentionally does
        # not open, stat semantically, or hash any protected input.
        "single_access_inputs": runtime.PROTECTED_INPUTS_V44A,
        "raw_shadow_or_ood_content_opened_before_preregistration": False,
        "heldout_or_holdout_access_authorized": False,
        "runtime": {
            "model": str(runtime.MODEL),
            "precision": "bfloat16",
            "four_independent_tp1_engines": True,
            "physical_gpu_ids": list(runtime.GPU_IDS),
            "arm_wave_plan": [
                [{"arm": arm, "engine_index": engine} for arm, engine in wave]
                for wave in runtime.arm_wave_plan_v44a()
            ],
            "all_four_engines_receive_positive_work_each_evaluation_phase": True,
            "tuned_table_content_sha256": (
                "4c4a0d4bbb400ea1d881bea3aae144d6865c34199fbb67889eda9e92d3a2543d"
            ),
            "generation_seed": runtime.GENERATION_SEED,
            "bootstrap_seed": runtime.BOOTSTRAP_SEED,
            "bootstrap_samples": runtime.BOOTSTRAP_SAMPLES,
            "obsolete_full_weight_layer_plan_or_snapshot_path_authorized": False,
        },
        "shadow_protocol": {
            "fold": 3,
            "rows": 83,
            "conflict_units": 51,
            "document_disjoint_from_fold3_train_required": True,
            "equal_conflict_unit_weighting": True,
            "base_duplicate_exact_equivalence_required": True,
            "selection_candidates": list(runtime.CANDIDATE_ARMS),
            "selection_rule": (
                "lexicographic generated equal-unit mean reward, exact count, "
                "nonzero count, teacher-forced equal-unit mean answer logprob; "
                "frozen candidate-order final tie (V43D highest)"
            ),
            "shadow_improvement_over_base_a_required": True,
            "no_protocol_or_leak_counter_increase_required": True,
        },
        "ood_gates": {
            "selected_candidate_only": True,
            "qa_mean_reward_and_exact_count_non_degradation": True,
            "prose_point_and_paired_document_bootstrap_lcb_non_degradation": True,
            "ood_does_not_select_or_reorder_candidates": True,
        },
        "persistence": {
            "aggregate_report": str(runtime.REPORT),
            "raw_local_mode": "0600",
            "raw_local_git_eligible": False,
            "raw_questions_answers_or_generations_in_aggregate": False,
            "single_semantic_read_per_protected_input": True,
        },
        "fresh_artifacts": {
            "run_directory": str(runtime.RUN_DIR),
            "attempt": str(runtime.ATTEMPT),
            "gpu_log": str(runtime.GPU_LOG),
            "raw_local": str(runtime.RAW),
            "report": str(runtime.REPORT),
        },
    }
    value["content_sha256_before_self_field"] = runtime.canonical_sha256(value)
    return value


def main(argv: list[str] | None = None) -> int:
    output = Path(parser().parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build()
    runtime.atomic_json(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": runtime.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "protected_semantic_access_count": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
