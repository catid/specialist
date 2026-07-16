#!/usr/bin/env python3
"""Seal the matched V47C versus V49B replicated OOD-first evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import run_matched_lora_candidate_eval_v44a as core
import run_sft_v47c_vs_v49b_ood_first_v49c as runtime


OUTPUT = runtime.DEFAULT_PREREGISTRATION


def candidate_definition_v49c(logical: str) -> dict:
    stage = runtime.canonical_stage_binding_v49c(logical)
    if logical == "sft_v47c":
        source = {
            "training_family": (
                "matched equal-unit SFT on lineage-stable holdout-blind v430 refresh"
            ),
            "learning_rate": 5.5e-5,
            "completed_steps": 48,
            "source_weights_sha256": runtime.v47c_stage.EXPECTED["weights"],
            "source_config_sha256": runtime.v47c_stage.EXPECTED["config"],
            "source_report_file_sha256": runtime.v47c_stage.EXPECTED["report"],
            "source_report_content_sha256": (
                runtime.v47c_stage.EXPECTED["report_content"]
            ),
            "source_preregistration_file_sha256": (
                runtime.v47c_stage.EXPECTED["preregistration"]
            ),
            "source_attempt_file_sha256": runtime.v47c_stage.EXPECTED["attempt"],
            "training_weighting": "equal conflict-unit weights",
            "all_four_training_gpus_attributed_positive": True,
            "training_shadow_ood_holdout_opened": False,
        }
    elif logical == "sft_v49b":
        source = {
            "training_family": (
                "matched SFT with exact V49A source-balanced row weights on "
                "lineage-stable frozen fold membership"
            ),
            "learning_rate": 5.5e-5,
            "completed_steps": 48,
            "source_weights_sha256": runtime.v49b_stage.EXPECTED["weights"],
            "source_config_sha256": runtime.v49b_stage.EXPECTED["config"],
            "source_report_file_sha256": runtime.v49b_stage.EXPECTED["report"],
            "source_report_content_sha256": (
                runtime.v49b_stage.EXPECTED["report_content"]
            ),
            "source_preregistration_file_sha256": (
                runtime.v49b_stage.EXPECTED["preregistration"]
            ),
            "source_attempt_file_sha256": runtime.v49b_stage.EXPECTED["attempt"],
            "training_weighting": "exact V49A alternative per-row weights",
            "weight_identity_sha256": runtime.v49b_stage.EXPECTED_WEIGHT_IDENTITY,
            "per_row_identity_sha256": runtime.v49b_stage.EXPECTED_ROW_IDENTITY,
            "per_source_identity_sha256": runtime.v49b_stage.EXPECTED_SOURCE_IDENTITY,
            "per_category_identity_sha256": (
                runtime.v49b_stage.EXPECTED_CATEGORY_IDENTITY
            ),
            "all_four_training_gpus_attributed_positive": True,
            "training_shadow_ood_holdout_opened": False,
        }
    else:
        raise ValueError(logical)
    return {
        "logical_candidate": logical,
        "replica_arms": list(runtime.LOGICAL_REPLICAS_V49C[logical]),
        **source,
        "staged_directory": stage["directory"],
        "staged_weights_sha256": stage["weights_file_sha256"],
        "staged_config_sha256": stage["adapter_config_file_sha256"],
        "stage_manifest_file_sha256": stage["manifest_file_sha256"],
        "stage_manifest_content_sha256": stage["manifest_content_sha256"],
        "transformed_identity_sha256": stage["transformed_identity_sha256"],
        "replicas_use_identical_staged_bytes": True,
        "replica_outputs_required_bit_exact": False,
        "replica_adapter_ids": {
            arm: runtime.ADAPTER_IDS_V49C[arm]
            for arm in runtime.LOGICAL_REPLICAS_V49C[logical]
        },
    }


def build_v49c() -> dict:
    parent = runtime.parent_protocol_v47d_v49c()
    value = {
        key: item for key, item in parent.items()
        if key != "content_sha256_before_self_field"
    }
    stages = runtime.replica_stage_bindings_v49c()
    value.update({
        "schema": "sft-v47c-vs-v49b-replicated-ood-first-preregistration-v49c",
        "status": (
            "preregistered_after_v49b_train_completion_before_ood_first_evaluation"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_launch_authorized": True,
        "evaluation_launch_authorized": True,
        "heldout_or_holdout_access_authorized": False,
        "current_fixed_holdout_cycle_eligible": False,
        "future_cycle_diagnostic_only": True,
        "protected_semantics_inspected_during_revision": False,
        "purpose": (
            "Compare lineage-stable equal-unit V47C with exact V49A "
            "source-balanced V49B using the frozen replicated V47D OOD-first "
            "protocol, unchanged bundles, gates, prompts, sampling, and seed."
        ),
        "arms": list(runtime.ARMS_V49C),
        "base_duplicate_arms": list(runtime.BASE_ARMS_V49C),
        "logical_candidates": list(runtime.LOGICAL_CANDIDATES_V49C),
        "logical_candidate_replicas": {
            logical: list(replicas)
            for logical, replicas in runtime.LOGICAL_REPLICAS_V49C.items()
        },
        "candidate_arms": list(runtime.CANDIDATE_ARMS_V49C),
        "candidate_definitions": [
            candidate_definition_v49c(logical)
            for logical in runtime.LOGICAL_CANDIDATES_V49C
        ],
        "replica_staged_adapters": stages,
        "staged_adapters": stages,
        "implementation_bindings": runtime.implementation_bindings_v49c(),
        "extends_v47d_protocol": {
            "path": str(runtime.PARENT_PROTOCOL),
            "file_sha256": runtime.PARENT_PROTOCOL_FILE_SHA256,
            "content_sha256": runtime.PARENT_PROTOCOL_CONTENT_SHA256,
            "protected_commitments_reused_without_reopening": True,
            "bundles_gates_prompts_sampling_seed_unchanged": True,
            "four_base_two_candidate_replica_shape_unchanged": True,
        },
        "protected_semantics_opened_by_builder": False,
        "consumed_holdout_artifact_bound": False,
        "fresh_artifacts": {
            "run_directory": str(runtime.RUN_DIR),
            "attempt": str(runtime.ATTEMPT),
            "gpu_log": str(runtime.GPU_LOG),
            "raw_local": str(runtime.RAW),
            "report": str(runtime.REPORT),
        },
        "matched_comparison": {
            "primary_baseline": "sft_v47c",
            "candidate": "sft_v49b",
            "only_training_change": "per-row example weights",
            "same_initialization": True,
            "same_lora_surface": True,
            "same_learning_rate": True,
            "same_48_optimizer_steps": True,
            "same_prompt_encoding": True,
            "same_batches_and_schedule": True,
            "same_frozen_root_membership_and_document_disjointness": True,
            "v49b_accepted_train_text_projection": "through v434",
            "v47c_accepted_train_text_projection": "through v430",
            "post_v430_text_edits_are_lineage_stable_and_not_metric_selected": True,
            "v42i_role": "historical anchor through frozen V47D only",
            "fresh_v42i_arm_added": False,
            "reason_no_third_candidate": (
                "preserve exact V47D two-wave four-GPU replicated shape"
            ),
        },
        "replica_consensus_protocol_v49c": {
            "inherited_exactly_from": "V47D",
            "replicas_per_logical_candidate": 2,
            "replicas_share_exact_staged_adapter_bytes": True,
            "replicas_have_distinct_request_names_and_adapter_ids": True,
            "candidate_metric_or_generation_bit_equality_required": False,
            "each_replica_independently_ood_qa_gated": True,
            "each_replica_independently_ood_prose_bootstrap_lcb_gated": True,
            "each_replica_independently_protocol_gated": True,
            "logical_candidate_eligible_only_if_both_replicas_pass": True,
            "shadow_rank_uses_mean_of_two_replica_metric_vectors": True,
            "four_base_replicas_remain_exact_on_every_surface": True,
        },
        "selection_protocol_v49c": {
            "order": [
                "require exact four-base agreement on every surface",
                "independently OOD-gate both replicas of V47C and V49B",
                "exclude a logical candidate if either replica is ineligible",
                "mean the two shadow metric vectors for each eligible candidate",
                "rank only the immutable OOD-eligible set by the mean vector",
                "require the selected mean vector to improve over exact base",
            ],
            "ood_eligible_set_constructed_before_shadow_ranking": True,
            "rank_fields": list(runtime.RANK_FIELDS_V49C),
            "exact_tie_order": list(runtime.LOGICAL_CANDIDATES_V49C),
            "conservative_exact_tie_favors": "sft_v47c",
            "fallback_when_no_logical_candidate_is_eligible": "base_a",
        },
    })
    value["runtime"] = dict(value["runtime"])
    value["runtime"].update({
        "engine_count": 4,
        "physical_gpu_ids": [0, 1, 2, 3],
        "tensor_parallel_size_per_engine": 1,
        "two_full_fixed_waves": [
            [{"arm": arm, "engine_index": engine} for arm, engine in wave]
            for wave in runtime.arm_wave_plan_v49c()
        ],
        "every_gpu_receives_one_request_per_wave": True,
        "all_four_gpus_busy_in_every_evaluation_wave": True,
        "identical_prompts_sampling_params_and_seed_for_all_arms": True,
        "four_base_replicates_exact_required": True,
        "candidate_replicate_outputs_exact_required": False,
        "no_partial_third_wave": True,
    })
    value["shadow_protocol"] = dict(value["shadow_protocol"])
    value["shadow_protocol"].update({
        "selection_candidates": list(runtime.LOGICAL_CANDIDATES_V49C),
        "four_base_outputs_exact_equivalence_required": True,
        "candidate_replica_outputs_exact_equivalence_required": False,
        "logical_candidate_mean_replica_ranking": True,
        "ranking_occurs_only_after_all_replica_ood_gates": True,
        "uses_original_frozen_v412_shadow": True,
    })
    value["content_sha256_before_self_field"] = core.canonical_sha256(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_v49c()
    core.atomic_json(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": core.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "logical_candidates": value["logical_candidates"],
        "gpu_launch_authorized": True,
        "protected_semantic_access_count": 0,
        "heldout_or_holdout_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
