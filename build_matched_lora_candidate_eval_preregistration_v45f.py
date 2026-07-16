#!/usr/bin/env python3
"""Seal V45F conservative replica consensus without protected-data access."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v45f as runtime


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_sft_ijk_replica_consensus_ood_eval_v45f.json"
).resolve()


def build() -> dict:
    prior = runtime.prior_preregistration_v45f()
    value = {
        key: item for key, item in prior.items()
        if key != "content_sha256_before_self_field"
    }
    value.update({
        "schema": "matched-lora-sft-ijk-replica-consensus-preregistration-v45f",
        "status": "preregistered_before_fresh_replica_consensus_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arms": list(runtime.prior.ARMS_V45E),
        "logical_candidates": list(runtime.prior.LOGICAL_CANDIDATES_V45E),
        "logical_candidate_replicas": {
            logical: list(replicas)
            for logical, replicas in runtime.prior.LOGICAL_REPLICAS_V45E.items()
        },
        "replica_staged_adapters": runtime.prior.replica_stage_bindings_v45e(),
        "implementation_bindings": runtime.implementation_bindings_v45f(),
        "extends_preregistration": {
            "path": str(runtime.V45E_PREREG),
            "file_sha256": runtime.V45E_PREREG_FILE_SHA256,
            "content_sha256": runtime.V45E_PREREG_CONTENT_SHA256,
            "protected_commitments_reused_without_reopening": True,
        },
        "v45e_failure_evidence": runtime.failure_evidence_v45f(),
        "protected_semantics_inspected_during_v45f_revision": False,
        "heldout_or_holdout_access_authorized": False,
        "fresh_artifacts": {
            "run_directory": str(runtime.RUN_DIR),
            "attempt": str(runtime.ATTEMPT),
            "gpu_log": str(runtime.GPU_LOG),
            "aggregate_receipts_no_raw_semantics": str(runtime.RAW),
            "report": str(runtime.REPORT),
        },
        "conservative_replica_consensus_v45f": {
            "candidate_replica_bit_exact_equivalence_required": False,
            "each_replica_independently_gated": True,
            "per_replica_gates": {
                "ood_qa_mean_reward_point_delta_gte_zero": True,
                "ood_qa_exact_count_delta_gte_zero": True,
                "ood_qa_paired_item_bootstrap_ci_informational": True,
                "ood_prose_mean_token_logprob_point_delta_gte_zero": True,
                "ood_prose_paired_document_bootstrap_95_lcb_gte_zero": True,
                "shadow_protocol_or_leak_counter_increase_forbidden": True,
            },
            "logical_candidate_eligible_iff_both_replicas_pass": True,
            "any_replica_gate_failure_excludes_logical_candidate": True,
            "ood_eligible_set_constructed_before_shadow_ranking": True,
            "shadow_rank_key_over_replica_means": [
                "generated_equal_unit_mean_reward",
                "generated_exact_count",
                "generated_nonzero_count",
                "teacher_forced_equal_unit_mean_answer_logprob",
            ],
            "per_replica_values_min_max_mean_and_range_reported": True,
            "padding_base_arms_affect_eligibility_or_ranking": False,
        },
        "base_protocol_v45f": {
            "four_primary_base_duplicates_exact": True,
            "two_padding_base_duplicates_exact": True,
            "all_six_base_outputs_exact_on_all_splits": True,
            "padding_base_arms": list(runtime.prior.PADDING_BASE_ARMS_V45E),
            "padding_base_arms_excluded_from_eligibility_and_ranking": True,
        },
        "raw_persistence_policy_v45f": {
            "raw_questions_answers_or_generations_persisted": False,
            "raw_semantics_cleared_after_in_memory_gates": True,
            "local_artifact_contains_only_aggregate_selection_and_access_receipts": True,
            "aggregate_report_contains_no_raw_content": True,
        },
    })
    value["runtime"] = dict(value["runtime"])
    value["runtime"].update({
        "three_full_fixed_waves": [
            [{"arm": arm, "engine_index": engine} for arm, engine in wave]
            for wave in runtime.prior.arm_wave_plan_v45e()
        ],
        "every_gpu_receives_one_request_per_wave": True,
        "all_four_gpus_busy_in_every_evaluation_wave": True,
        "identical_prompts_sampling_params_and_seed_for_all_arms": True,
        "existing_single_access_firewall_and_parser_preflight_reused": True,
    })
    value["shadow_protocol"] = dict(value["shadow_protocol"])
    value["shadow_protocol"].update({
        "selection_candidates": list(runtime.prior.LOGICAL_CANDIDATES_V45E),
        "six_base_outputs_exact_equivalence_required": True,
        "candidate_replica_bit_exact_equivalence_required": False,
        "replicated_mean_ranking_after_two-replica_ood_consensus": True,
        "padding_base_arms_excluded_from_eligibility_and_ranking": list(
            runtime.prior.PADDING_BASE_ARMS_V45E
        ),
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
        "protected_semantics_inspected_during_revision": False,
        "heldout_or_holdout_opened": False,
        "raw_semantics_will_be_persisted": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
