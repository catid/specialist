#!/usr/bin/env python3
"""Seal replicated I/J/K V45E without protected-data access."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v45e as runtime


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_sft_ijk_replicated_ood_eligible_eval_v45e.json"
).resolve()


def candidate_definition_v45e(logical: str) -> dict:
    modules = {
        "sft_v42i": runtime.i_stage,
        "sft_v42j": runtime.j_stage,
        "sft_v42k": runtime.k_stage,
    }
    module = modules[logical]
    spec = module.SPEC
    stage = runtime.canonical_stage_binding_v45e(logical)
    return {
        "logical_candidate": logical,
        "replica_arms": list(runtime.LOGICAL_REPLICAS_V45E[logical]),
        "training_family": "matched SFT",
        "learning_rate": spec["learning_rate"],
        "completed_steps": spec["completed_steps"],
        "source_artifact_prefix": spec["artifact_prefix"],
        "source_weights_sha256": spec["weights"],
        "source_config_sha256": spec["config"],
        "source_report_file_sha256": spec["seal"],
        "source_report_content_sha256": spec["seal_content"],
        "staged_directory": str(runtime.STAGED_BY_LOGICAL_V45E[logical]),
        "staged_weights_sha256": stage["weights_file_sha256"],
        "stage_manifest_file_sha256": stage["manifest_file_sha256"],
        "stage_manifest_content_sha256": stage["manifest_content_sha256"],
        "transformed_identity_sha256": stage[
            "transformed_identity_sha256"
        ],
        "replicas_use_identical_staged_bytes": True,
        "replica_adapter_ids": {
            arm: runtime.ADAPTER_IDS_V45E[arm]
            for arm in runtime.LOGICAL_REPLICAS_V45E[logical]
        },
    }


def build() -> dict:
    prior = runtime.prior_preregistration_v45e()
    value = {
        key: item for key, item in prior.items()
        if key != "content_sha256_before_self_field"
    }
    value.update({
        "schema": "matched-lora-sft-ijk-replicated-ood-preregistration-v45e",
        "status": "preregistered_before_fresh_replicated_boundary_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arms": list(runtime.ARMS_V45E),
        "primary_base_duplicate_arms": list(runtime.PRIMARY_BASE_ARMS_V45E),
        "padding_base_arms": list(runtime.PADDING_BASE_ARMS_V45E),
        "all_base_arms": list(runtime.BASE_ARMS_V45E),
        "logical_candidates": list(runtime.LOGICAL_CANDIDATES_V45E),
        "logical_candidate_replicas": {
            logical: list(replicas)
            for logical, replicas in runtime.LOGICAL_REPLICAS_V45E.items()
        },
        "candidate_arms": list(runtime.CANDIDATE_ARMS_V45E),
        "candidate_definitions": [
            candidate_definition_v45e(logical)
            for logical in runtime.LOGICAL_CANDIDATES_V45E
        ],
        "replica_staged_adapters": runtime.replica_stage_bindings_v45e(),
        "implementation_bindings": runtime.implementation_bindings_v45e(),
        "extends_preregistration": {
            "path": str(runtime.V45D_PREREG),
            "file_sha256": runtime.V45D_PREREG_FILE_SHA256,
            "content_sha256": runtime.V45D_PREREG_CONTENT_SHA256,
            "protected_commitments_reused_without_reopening": True,
        },
        "v45d_aggregate_observed_before_v45e_preregistration": (
            runtime.prior_result_v45e()
        ),
        "protected_semantics_inspected_during_v45e_revision": False,
        "heldout_or_holdout_access_authorized": False,
        "fresh_artifacts": {
            "run_directory": str(runtime.RUN_DIR),
            "attempt": str(runtime.ATTEMPT),
            "gpu_log": str(runtime.GPU_LOG),
            "raw_local": str(runtime.RAW),
            "report": str(runtime.REPORT),
        },
        "replica_equivalence_protocol_v45e": {
            "every_logical_candidate_has_exactly_two_replicas": True,
            "replicas_share_exact_staged_directory_and_bytes": True,
            "replicas_have_distinct_request_names_and_adapter_ids": True,
            "metrics_must_be_exact_on_shadow_ood_qa_and_ood_prose": True,
            "raw_numeric_outputs_must_be_exact_on_all_three_splits": True,
            "any_replica_disagreement_fails_closed_before_selection": True,
        },
        "selection_protocol_v45e": {
            "order": [
                "require all base and logical-candidate replica equivalence",
                "construct OOD QA+prose eligibility per logical candidate",
                "exclude every ineligible logical candidate",
                "rank only eligible logical candidates on replica-A frozen shadow metrics",
                "apply shadow improvement requirement to the eligible winner",
            ],
            "per_logical_candidate_eligibility": {
                "replica_equivalence_required": True,
                "ood_qa_mean_reward_point_delta_gte_zero": True,
                "ood_qa_exact_count_delta_gte_zero": True,
                "ood_prose_point_delta_gte_zero": True,
                "ood_prose_paired_document_bootstrap_95_lcb_gte_zero": True,
                "shadow_protocol_or_leak_counter_increase_forbidden": True,
            },
            "shadow_rank_key": [
                "generated_equal_unit_mean_reward", "generated_exact_count",
                "generated_nonzero_count",
                "teacher_forced_equal_unit_mean_answer_logprob",
            ],
            "ood_eligible_set_constructed_before_shadow_ranking": True,
            "padding_base_arms_affect_eligibility_or_ranking": False,
            "no_eligible_candidate_behavior": (
                "select base_a operational sentinel and fail shadow improvement gate"
            ),
        },
    })
    value["runtime"] = dict(value["runtime"])
    value["runtime"].update({
        "three_full_fixed_waves": [
            [{"arm": arm, "engine_index": engine} for arm, engine in wave]
            for wave in runtime.arm_wave_plan_v45e()
        ],
        "every_gpu_receives_one_request_per_wave": True,
        "all_four_gpus_busy_in_every_evaluation_wave": True,
        "identical_prompts_sampling_params_and_seed_for_all_arms": True,
        "four_primary_base_replicates_exact_required": True,
        "two_padding_base_replicates_exact_required": True,
        "two_padding_base_replicates_excluded_from_ranking": True,
    })
    value["shadow_protocol"] = dict(value["shadow_protocol"])
    value["shadow_protocol"].update({
        "selection_candidates": list(runtime.LOGICAL_CANDIDATES_V45E),
        "six_base_outputs_exact_equivalence_required": True,
        "padding_base_arms_excluded_from_eligibility_and_ranking": list(
            runtime.PADDING_BASE_ARMS_V45E
        ),
        "logical_replica_equivalence_required": True,
    })
    value["content_sha256_before_self_field"] = core.canonical_sha256(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build()
    core.atomic_json(output, value)
    print(json.dumps({
        "path": str(output), "file_sha256": core.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "cpu_preflight_content_sha256": value[
            "cpu_preflight_expected"
        ]["content_sha256_before_self_field"],
        "logical_candidates": value["logical_candidates"],
        "protected_semantics_inspected_during_revision": False,
        "heldout_or_holdout_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
