#!/usr/bin/env python3
"""Seal strict future-cycle V42I versus V47C replicated OOD-first evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import run_matched_lora_candidate_eval_v44a as core
import run_sft_v42i_vs_v47c_ood_first_v47d as runtime


OUTPUT = runtime.DEFAULT_PREREGISTRATION


def candidate_definition_v47d(logical: str) -> dict:
    stage = runtime.canonical_stage_binding_v47d(logical)
    if logical == "sft_v42i":
        source = {
            "training_family": "matched SFT on v412",
            "learning_rate": runtime.v42i_stage.SPEC["learning_rate"],
            "completed_steps": runtime.v42i_stage.SPEC["completed_steps"],
            "source_weights_sha256": runtime.v42i_stage.SPEC["weights"],
            "source_config_sha256": runtime.v42i_stage.SPEC["config"],
            "source_report_file_sha256": runtime.v42i_stage.SPEC["seal"],
            "source_report_content_sha256": runtime.v42i_stage.SPEC["seal_content"],
        }
    elif logical == "sft_v47c":
        source = {
            "training_family": "matched SFT on lineage-stable holdout-blind v430 refresh",
            "learning_rate": 5.5e-5,
            "completed_steps": 48,
            "source_weights_sha256": runtime.v47c_stage.EXPECTED["weights"],
            "source_config_sha256": runtime.v47c_stage.EXPECTED["config"],
            "source_report_file_sha256": runtime.v47c_stage.EXPECTED["report"],
            "source_report_content_sha256": runtime.v47c_stage.EXPECTED["report_content"],
            "source_preregistration_file_sha256": (
                runtime.v47c_stage.EXPECTED["preregistration"]
            ),
            "source_attempt_file_sha256": runtime.v47c_stage.EXPECTED["attempt"],
            "all_four_training_gpus_attributed_positive": True,
            "training_shadow_ood_holdout_opened": False,
            "lineage_stable_original_fold_assignment": True,
            "accepted_edit_successors_in_training": 42,
            "accepted_edit_successors_in_frozen_shadow": 9,
        }
    else:
        raise ValueError(logical)
    return {
        "logical_candidate": logical,
        "replica_arms": list(runtime.LOGICAL_REPLICAS_V47D[logical]),
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
            arm: runtime.ADAPTER_IDS_V47D[arm]
            for arm in runtime.LOGICAL_REPLICAS_V47D[logical]
        },
    }


def build_v47d() -> dict:
    parent = runtime.trusted.trusted.parent_preregistration_v46c()
    value = {
        key: item for key, item in parent.items()
        if key != "content_sha256_before_self_field"
    }
    stages = runtime.replica_stage_bindings_v47d()
    value.update({
        "schema": "sft-v42i-vs-v47c-replicated-ood-first-preregistration-v47d",
        "status": "preregistered_before_fresh_replicated_ood_first_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_launch_authorized": True,
        "evaluation_launch_authorized": True,
        "heldout_or_holdout_access_authorized": False,
        "current_fixed_holdout_cycle_eligible": False,
        "future_cycle_diagnostic_only": True,
        "protected_semantics_inspected_during_revision": False,
        "purpose": (
            "Compare exact base, V42I, and lineage-stable refreshed-data V47C "
            "under strict replicated OOD-first eligibility before ranking on "
            "the unchanged original document-disjoint shadow."
        ),
        "arms": list(runtime.ARMS_V47D),
        "base_duplicate_arms": list(runtime.BASE_ARMS_V47D),
        "logical_candidates": list(runtime.LOGICAL_CANDIDATES_V47D),
        "logical_candidate_replicas": {
            logical: list(replicas)
            for logical, replicas in runtime.LOGICAL_REPLICAS_V47D.items()
        },
        "candidate_arms": list(runtime.CANDIDATE_ARMS_V47D),
        "candidate_definitions": [
            candidate_definition_v47d(logical)
            for logical in runtime.LOGICAL_CANDIDATES_V47D
        ],
        "replica_staged_adapters": stages,
        "staged_adapters": stages,
        "implementation_bindings": runtime.implementation_bindings_v47d(),
        "extends_preregistration": {
            "path": str(runtime.trusted.trusted.PARENT_PREREGISTRATION),
            "file_sha256": runtime.trusted.trusted.PARENT_PREREGISTRATION_FILE_SHA256,
            "content_sha256": runtime.trusted.trusted.PARENT_PREREGISTRATION_CONTENT_SHA256,
            "protected_commitments_reused_without_reopening": True,
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
        "validation_label_policy": {
            "policy": "unchanged original frozen v412 fold-3 labels for both candidates",
            "evaluation_shadow_path": core.PROTECTED_INPUTS_V44A["shadow"]["path"],
            "evaluation_shadow_file_sha256": (
                core.PROTECTED_INPUTS_V44A["shadow"]["file_sha256"]
            ),
            "evaluation_shadow_rows": 83,
            "evaluation_shadow_conflict_units": 51,
            "accepted_edit_successors_in_v47c_training": 42,
            "accepted_shadow_side_edits": 9,
            "accepted_shadow_side_edit_units": 3,
            "shadow_side_edits_excluded_from_training": True,
            "shadow_side_edits_excluded_from_evaluation_label_refresh": True,
            "refreshed_shadow_used_as_evaluation_input": False,
            "refreshed_shadow_role": "content-addressed lineage proof only",
            "no_post_hoc_validation_target_change": True,
        },
        "edit_scope_audit_binding": {
            "path": str(runtime.EDIT_SCOPE_AUDIT),
            "file_sha256": runtime.EDIT_SCOPE_AUDIT_FILE_SHA256,
            "content_sha256": runtime.EDIT_SCOPE_AUDIT_CONTENT_SHA256,
            "schema": "specialist-v47c-edit-scope-audit-v1",
            "status": "complete_content_free_pre_evaluation_audit",
            "accepted_edit_decisions": 51,
            "edited_rows_entering_train": 42,
            "edited_units_entering_train": 34,
            "edited_rows_excluded_in_shadow": 9,
            "edited_units_excluded_in_shadow": 3,
            "fold_assignment_changes": 0,
            "manual_question_answer_or_evidence_inspection": False,
            "refreshed_shadow_authorized_as_selection_input": False,
        },
        "replica_consensus_protocol_v47d": {
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
        "selection_protocol_v47d": {
            "order": [
                "require exact four-base agreement on every surface",
                "independently OOD-gate both replicas of each logical candidate",
                "exclude a logical candidate if either replica is ineligible",
                "mean the two shadow metric vectors for each eligible candidate",
                "rank only the immutable OOD-eligible set by the mean vector",
                "require the selected mean vector to improve over exact base",
            ],
            "ood_eligible_set_constructed_before_shadow_ranking": True,
            "rank_fields": list(runtime.RANK_FIELDS_V47D),
            "exact_tie_order": list(runtime.LOGICAL_CANDIDATES_V47D),
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
            for wave in runtime.arm_wave_plan_v47d()
        ],
        "every_gpu_receives_one_request_per_wave": True,
        "all_four_gpus_busy_in_every_evaluation_wave": True,
        "identical_prompts_sampling_params_and_seed_for_all_arms": True,
        "four_base_replicates_exact_required": True,
        "candidate_replicate_outputs_exact_required": False,
    })
    value["shadow_protocol"] = dict(value["shadow_protocol"])
    value["shadow_protocol"].update({
        "selection_candidates": list(runtime.LOGICAL_CANDIDATES_V47D),
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
    value = build_v47d()
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
