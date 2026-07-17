#!/usr/bin/env python3
"""Prepare, but interlock, V42I's once-only sealed holdout evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import run_matched_lora_candidate_eval_v45d as environment_source
import run_once_only_holdout_eval_v46a as runtime


ROOT = Path(__file__).resolve().parent
OUTPUT = runtime.PREREG


def build() -> dict:
    raise RuntimeError(
        "V46A protected-QA preregistration is permanently quarantined with V1"
    )
    holdout = runtime.holdout_report_commitment_v46a()
    boundary = runtime.boundary_evidence_v46a()
    stage = runtime.staged_candidate_binding_v46a()
    environment = environment_source.prior.environment.environment_bindings_v44b()
    value = {
        "schema": "prepared-once-only-fixed-sft-v42i-holdout-preregistration-v46a",
        "status": "preregistered_before_once_only_holdout_access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "prepare a once-only aggregate evaluation of the current exact "
            "V42I SFT control against three exact base duplicates"
        ),
        "fixed_arms": list(runtime.ARMS_V46A),
        "base_duplicate_arms": list(runtime.BASE_ARMS_V46A),
        "fixed_candidate_arm": runtime.CANDIDATE_ARM_V46A,
        "candidate_selection_permitted": False,
        "post_result_tuning_or_selection_permitted": False,
        "holdout_access_authorized_once": True,
        "holdout_access_count_before_preregistration": 0,
        "holdout_commitment": holdout,
        "boundary_evidence": boundary,
        "fixed_staged_candidate": stage,
        "fixed_source_candidate": {
            "arm": "sft_v42i",
            "training_family": "matched SFT",
            "learning_rate": 5.5e-5,
            "completed_steps": 48,
            "source_directory": str(runtime.SOURCE_V42I),
            "source_report_path": str(runtime.SOURCE_REPORT_V42I),
            "source_report_file_sha256": (
                "3076ff21d7d7910cc9ae33f1c00c69b10d8e72c6c8366bb1029ceca17812cee6"
            ),
            "source_report_content_sha256": (
                "16d8898b6b81da33a6968c254e2d5c5684dd6a284ee0874b9f762bfc140b4341"
            ),
            "source_weights_sha256": (
                "9e83783c20dfb5eec91b7217d885270efed8aec216c80374444dcbc55fd7dab8"
            ),
            "source_config_sha256": (
                "0e8060efd40772233390f3f97ace489e473b2bc76572e7566b83afe3dd83cc51"
            ),
            "staged_weights_sha256": stage["weights_file_sha256"],
            "staged_config_sha256": stage["adapter_config_file_sha256"],
            "stage_manifest_file_sha256": stage["manifest_file_sha256"],
            "stage_manifest_content_sha256": stage[
                "manifest_content_sha256"
            ],
            "transformed_identity_sha256": stage[
                "transformed_identity_sha256"
            ],
            "all_tensor_bytes_preserved_exactly": True,
        },
        "report_hash_chain": {
            "v45a_v45b_v45c": (
                "bound by boundary_evidence -> prior_repetition_evidence; "
                "the prior evidence contains every aggregate file and content hash"
            ),
            "v45d": (
                "bound directly by boundary evidence file/content hash and "
                "internally by V45D aggregate file/content hash"
            ),
            "hash_chain_is_cryptographically_closed": True,
        },
        "fixed_gate": {
            "mean_reward_point_delta_gte_zero": True,
            "exact_count_point_delta_gte_zero": True,
            "nonzero_count_point_delta_gte_zero": True,
            "protocol_or_leak_counter_increase_forbidden": True,
            "teacher_forced_metric_informational_only": True,
            "paired_item_bootstrap_cis_informational_only": True,
            "candidate_selection_performed": False,
        },
        "stratified_reporting": {
            "dimensions": ["source", "quality_bucket"],
            "metrics": [
                "rows", "mean_reward", "exact_count", "exact_rate",
                "nonzero_count", "nonzero_rate",
            ],
            "strata_do_not_select_or_tune_candidate": True,
        },
        "single_access_policy": {
            "attempt_marker_written_before_any_holdout_access": True,
            "combined_eval_container_read_exactly_once": True,
            "expected_holdout_semantic_access_count": 1,
            "failure_after_access_consumes_attempt": True,
            "retry_after_failure_forbidden": True,
            "raw_questions_answers_or_generations_persisted": False,
            "aggregate_only_report": True,
        },
        "launch_interlock": {
            "resolved": False,
            "reason": "V43I OOD-first comparison is not yet resolved",
            "real_launch_permitted": False,
            "resolution_requires_new_runtime_and_preregistration_seal": True,
        },
        "runtime": {
            "model": str(runtime.core.MODEL),
            "precision": "bfloat16",
            "physical_gpu_ids": list(runtime.core.GPU_IDS),
            "four_independent_tp1_engines": True,
            "one_full_fixed_wave": [
                {"arm": arm, "engine_index": index}
                for index, arm in enumerate(runtime.ARMS_V46A)
            ],
            "all_four_gpus_receive_identical_prompt_batches": True,
            "all_four_gpus_must_be_resident_and_positive": True,
            "generation_seed": runtime.core.GENERATION_SEED,
            "bootstrap_seed": runtime.BOOTSTRAP_SEED_V46A,
            "bootstrap_samples": runtime.BOOTSTRAP_SAMPLES_V46A,
            "tuned_table_content_sha256": (
                "4c4a0d4bbb400ea1d881bea3aae144d6865c34199fbb67889eda9e92d3a2543d"
            ),
            "required_python_environment": environment,
            "dry_run_accesses_zero_holdout_semantics": True,
            "dry_run_does_not_stat_hash_or_open_holdout": True,
            "real_launch_currently_interlocked": True,
        },
        "implementation_bindings": runtime.nonprotected_bindings_v46a(),
        "fresh_artifacts": {
            "run_directory": str(runtime.RUN_DIR),
            "attempt": str(runtime.ATTEMPT),
            "gpu_log": str(runtime.GPU_LOG),
            "report": str(runtime.REPORT),
        },
        "pre_result_commitments": {
            "candidate_identity_is_fixed": True,
            "base_identity_is_fixed": True,
            "holdout_identity_and_split_are_fixed": True,
            "metrics_and_gate_are_fixed": True,
            "no_candidate_ranking": True,
        },
        "post_result_policy": {
            "result_is_terminal_for_this_frozen_holdout_cycle": True,
            "result_may_not_trigger_checkpoint_selection": True,
            "result_may_not_trigger_hyperparameter_tuning": True,
            "result_may_not_trigger_data_selection_or_resampling": True,
            "holdout_may_not_be_reopened": True,
        },
        "protected_semantics_accessed_while_building": False,
        "holdout_opened_or_hashed_while_building": False,
    }
    value["content_sha256_before_self_field"] = runtime.core.canonical_sha256(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build()
    runtime.core.atomic_json(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": runtime.core.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "fixed_candidate": value["fixed_candidate_arm"],
        "real_launch_permitted": False,
        "holdout_opened_or_hashed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
