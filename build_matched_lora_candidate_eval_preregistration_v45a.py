#!/usr/bin/env python3
"""Seal V45A's all-arm OOD eligibility gates before fresh evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import build_matched_lora_candidate_eval_preregistration_v44c as prior_builder
import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v45a as runtime
import stage_candidate_adapters_vllm_v45a as staging


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_hpo_earlystop_ood_eligible_eval_v45a.json"
).resolve()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return result


def candidate_definition_v45a(arm: str) -> dict:
    if arm in staging.SOURCE_SPECS_V45A:
        spec = staging.SOURCE_SPECS_V45A[arm]
        return {
            "arm": arm,
            "training_family": "matched SFT",
            "learning_rate": spec["learning_rate"],
            "completed_steps": spec["completed_steps"],
            "source_artifact_prefix": spec["artifact_prefix"],
            "source_weights_sha256": spec["weights"],
            "staged_directory": str(runtime.STAGED_BY_ARM_V45A[arm]),
            "adapter_id": runtime.ADAPTER_IDS_V45A[arm],
        }
    expected = core.staging.EXPECTED_V44A[arm]
    learning_rates = {
        "sft_v42b": 1e-4, "sft_v42c": 3e-5,
        "sft_v42d": 1e-5, "lora_es_v43d": None,
    }
    return {
        "arm": arm,
        "training_family": (
            "LoRA-ES" if arm == "lora_es_v43d" else "matched SFT"
        ),
        "learning_rate": learning_rates[arm],
        "completed_steps": 1 if arm == "lora_es_v43d" else 48,
        "source_artifact_prefix": "post_update_step1" if arm == "lora_es_v43d" else "final",
        "source_weights_sha256": expected["weights"],
        "staged_directory": str(runtime.STAGED_BY_ARM_V45A[arm]),
        "adapter_id": runtime.ADAPTER_IDS_V45A[arm],
    }


def build() -> dict:
    value = {
        key: item for key, item in prior_builder.build().items()
        if key != "content_sha256_before_self_field"
    }
    cpu_preflight = runtime.parser_fix.offline_authorized_schema_audit_v44c()
    value.update({
        "schema": "matched-lora-ood-eligible-eval-preregistration-v45a",
        "status": "preregistered_before_fresh_ood_eligible_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arms": list(runtime.ARMS_V45A),
        "base_duplicate_arms": list(runtime.BASE_ARMS_V45A),
        "candidate_arms": list(runtime.CANDIDATE_ARMS_V45A),
        "candidate_definitions": [
            candidate_definition_v45a(arm)
            for arm in runtime.CANDIDATE_ARMS_V45A
        ],
        "implementation_bindings": runtime.implementation_bindings_v45a(),
        "staged_adapters": runtime.staged_adapter_bindings_v45a(),
        "cpu_preflight_expected": cpu_preflight,
        "v44c_observed_finding": runtime.v44c_finding_provenance_v45a(),
        "selection_metrics_observed_before_v45a_preregistration": True,
        "raw_shadow_or_ood_content_opened_before_preregistration": True,
        "heldout_or_holdout_access_authorized": False,
        "heldout_or_holdout_opened_before_preregistration": False,
        "fresh_artifacts": {
            "run_directory": str(runtime.RUN_DIR),
            "attempt": str(runtime.ATTEMPT),
            "gpu_log": str(runtime.GPU_LOG),
            "raw_local": str(runtime.RAW),
            "report": str(runtime.REPORT),
        },
        "selection_protocol_v45a": {
            "order": [
                "evaluate every arm with identical shadow, OOD QA, and OOD prose prompts/settings",
                "construct eligibility independently for every candidate",
                "exclude every ineligible candidate",
                "rank only the eligible set on frozen shadow metrics",
                "apply shadow improvement requirement to the eligible winner",
            ],
            "per_arm_eligibility": {
                "ood_qa_mean_reward_point_delta_gte_zero": True,
                "ood_qa_exact_count_delta_gte_zero": True,
                "ood_qa_paired_item_bootstrap_ci_reported_informationally": True,
                "ood_prose_mean_token_logprob_point_delta_gte_zero": True,
                "ood_prose_paired_document_bootstrap_95_lcb_gte_zero": True,
                "shadow_protocol_or_leak_counter_increase_forbidden": True,
            },
            "shadow_rank_key": [
                "generated_equal_unit_mean_reward", "generated_exact_count",
                "generated_nonzero_count",
                "teacher_forced_equal_unit_mean_answer_logprob",
            ],
            "candidate_order_affects_eligibility": False,
            "candidate_order_used_only_for_exact_final_ties": True,
            "unsafe_higher_shadow_arm_cannot_mask_safe_lower_shadow_arm": True,
            "no_eligible_candidate_behavior": "select base_a operational sentinel and fail shadow improvement gate",
        },
    })
    value["runtime"] = dict(value["runtime"])
    value["runtime"].update({
        "physical_gpu_ids": list(core.GPU_IDS),
        "four_independent_tp1_engines": True,
        "three_full_fixed_waves": [
            [{"arm": arm, "engine_index": engine} for arm, engine in wave]
            for wave in runtime.arm_wave_plan_v45a()
        ],
        "every_gpu_receives_one_request_per_wave": True,
        "identical_prompts_sampling_params_and_seed_for_all_arms": True,
        "generation_seed": core.GENERATION_SEED,
        "bootstrap_seed": core.BOOTSTRAP_SEED,
        "bootstrap_samples": core.BOOTSTRAP_SAMPLES,
        "protected_parser_preflight_before_model_creation": True,
        "protected_preflight_expected_content_sha256": cpu_preflight[
            "content_sha256_before_self_field"
        ],
    })
    value["shadow_protocol"] = dict(value["shadow_protocol"])
    value["shadow_protocol"].update({
        "selection_candidates": list(runtime.CANDIDATE_ARMS_V45A),
        "selection_rule": (
            "OOD-eligible-set first, then frozen lexicographic shadow ranking"
        ),
        "three_base_duplicate_exact_equivalence_required": True,
    })
    value["ood_gates"] = {
        "evaluated_for_every_candidate_before_shadow_ranking": True,
        "qa_mean_reward_and_exact_count_point_non_degradation": True,
        "qa_paired_item_bootstrap_cis_reported": True,
        "prose_point_and_paired_document_bootstrap_lcb_non_degradation": True,
        "per_arm_gate_table_persisted_in_aggregate": True,
        "ood_filters_candidates_and_does_not_rank_eligible_candidates": True,
    }
    value["content_sha256_before_self_field"] = core.canonical_sha256(value)
    return value


def main(argv: list[str] | None = None) -> int:
    output = Path(parser().parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build()
    core.atomic_json(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": core.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "cpu_preflight_content_sha256": value[
            "cpu_preflight_expected"
        ]["content_sha256_before_self_field"],
        "heldout_or_holdout_opened": False,
        "required_python": str(runtime.environment.EXPECTED_ENV_PREFIX / "bin/python"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
