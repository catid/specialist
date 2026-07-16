#!/usr/bin/env python3
"""Seal the replicated V59-versus-V434 OOD-only comparison."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import run_matched_lora_candidate_eval_v44a as core
import run_v59_vs_v434_replicated_ood_only_v60 as runtime


ROOT = Path(__file__).resolve().parent
OUTPUT = runtime.DEFAULT_PREREGISTRATION


def build() -> dict:
    """Build metadata only; protected OOD files are neither read nor hashed."""
    stages = runtime.replica_stage_bindings_v56()
    definitions = []
    for logical in runtime.LOGICAL_CANDIDATES:
        source = runtime._source_seal(logical)
        staged = runtime.canonical_stage_binding_v56(logical)
        definitions.append({
            "logical_candidate": logical,
            "replica_arms": list(runtime.LOGICAL_REPLICAS[logical]),
            "source_train_only_provenance": source,
            "source_weights_sha256": source["source_weights_sha256"],
            "source_config_sha256": source["source_config_sha256"],
            "staged_directory": staged["directory"],
            "staged_weights_sha256": staged["weights_file_sha256"],
            "staged_config_sha256": staged["adapter_config_file_sha256"],
            "stage_manifest_file_sha256": staged["manifest_file_sha256"],
            "stage_manifest_content_sha256": staged[
                "manifest_content_sha256"
            ],
            "transformed_identity_sha256": staged[
                "transformed_identity_sha256"
            ],
            "replicas_use_identical_staged_bytes": True,
            "replica_outputs_required_bit_exact": False,
        })
    value = {
        "schema": "v59-vs-v434-replicated-ood-only-v60",
        "status": "preregistered_before_single_ood_access",
        "evaluation_launch_authorized": True,
        "training_access_authorized": False,
        "shadow_access_authorized": False,
        "terminal_holdout_access_authorized": False,
        "promotion_authorized": False,
        "protected_semantics_opened_by_builder": False,
        "arms": list(runtime.ARMS),
        "base_determinism_arms": list(runtime.BASE_ARMS),
        "logical_candidates": list(runtime.LOGICAL_CANDIDATES),
        "logical_candidate_replicas": {
            name: list(replicas)
            for name, replicas in runtime.LOGICAL_REPLICAS.items()
        },
        "candidate_definitions": definitions,
        "staged_adapters": stages,
        "implementation_bindings": runtime.implementation_bindings_v56(),
        "single_access_inputs": dict(runtime.OOD_INPUTS),
        "input_scope": {
            "exact_labels": ["ood_qa", "ood_prose"],
            "same_content_addressed_inputs_for_every_arm": True,
            "same_qa_and_prose_parsers_for_every_arm": True,
            "shadow_bound": False,
            "terminal_holdout_bound": False,
            "training_runtime_bound_or_read": False,
            "semantic_reads_during_builder_or_dry_run": 0,
            "semantic_read_count_at_execute": 2,
            "source_faithful_single_access_required": True,
            "document_identity_definitions": {
                "ood_qa": (
                    "exact sealed OOD-QA JSONL file SHA256 for every row; "
                    "eval-v3 rows have no document_sha256 field"
                ),
                "ood_prose": "SHA256 of the exact prose text field",
            },
        },
        "train_identity_registry": {
            "path": str(runtime.REGISTRY),
            "file_sha256": runtime.REGISTRY_FILE_SHA256,
            "content_sha256": runtime.REGISTRY_CONTENT_SHA256,
            "train_rows": 448,
            "content_minimized": True,
            "runtime_train_row_reads": 0,
            "four_required_identity_domains": [
                "document_sha256",
                "normalized_url",
                "raw_lineage_identity_sha256",
                "semantic_cluster_identity",
            ],
            "all_four_domains_must_be_disjoint_before_model_creation": True,
            "semantic_comparison": {
                "rule": (
                    "pairwise frozen V13 lexical semantic thresholds against "
                    "stored train question/answer token feature sets"
                ),
                "equality_only_is_sufficient": False,
                "matching_identity": "exact train semantic_cluster_sha256",
            },
        },
        "runtime": {
            "model": str(core.MODEL),
            "engine_count": 4,
            "physical_gpu_ids": [0, 1, 2, 3],
            "tensor_parallel_size_per_engine": 1,
            "two_full_fixed_waves": [
                [{"arm": arm, "engine_index": engine}
                 for arm, engine in wave]
                for wave in runtime.arm_wave_plan_v56()
            ],
            "candidate_wave_is_interleaved_by_direct_replica_pair": True,
            "all_four_gpus_busy_in_every_wave": True,
            "every_gpu_receives_one_request_per_wave": True,
            "identical_prompts_sampling_parameters_and_seed_for_all_arms": True,
            "generation_seed": runtime.GENERATION_SEED,
            "tuned_table_file_content_sha256": (
                runtime.TUNED_TABLE_FILE_CONTENT_SHA256
            ),
            "tuned_table_content_sha256": (
                runtime.TUNED_TABLE_RUNTIME_PROJECTION_SHA256
            ),
            "tuned_table_projection_preflight": (
                runtime.tuned_table_projection_preflight_v56()
            ),
            "qa_bootstrap_samples_per_direct_replica": runtime.BOOTSTRAP_SAMPLES,
            "qa_bootstrap_seeds": dict(runtime.PAIR_BOOTSTRAP_SEEDS),
            "trainer_resolver_surface": (
                "current V40C resolver attached by V44A make_trainer_v44a"
            ),
            "trainer_resolver_capability_preflight": (
                runtime.resolver_surface_preflight_v56()
            ),
            "trainer_resolver_required_before_identity_rpc": True,
            "no_partial_or_third_wave": True,
        },
        "raw_base_determinism_controls": {
            "arms": list(runtime.BASE_ARMS),
            "ood_qa_aggregate_outputs_all_four_bit_exact": True,
            "ood_qa_raw_outputs_all_four_bit_exact": True,
            "ood_prose_aggregate_outputs_all_four_bit_exact": True,
            "ood_prose_raw_outputs_all_four_bit_exact": True,
            "failure_closed": True,
        },
        "direct_v59_vs_v434_gates": {
            "pairs": [
                {"pair": pair, "reference": reference,
                 "candidate": candidate}
                for pair, reference, candidate in runtime.DIRECT_PAIRS
            ],
            "both_v59_replicas_must_independently_pass": True,
            "generated_qa_point_gates": {
                "row_mean_reward_delta_minimum": 0.0,
                "equal_unit_f1_delta_minimum": 0.0,
                "exact_count_delta_minimum": 0,
                "nonzero_count_delta_minimum": 0,
                "answer_token_logprob_delta_minimum": 0.0,
            },
            "generated_qa_paired_item_bootstrap_gates": {
                "samples": runtime.BOOTSTRAP_SAMPLES,
                "percentiles": [0.025, 0.975],
                "per_pair_seeds": dict(runtime.PAIR_BOOTSTRAP_SEEDS),
                "generated_reward_f1_lcb_minimum": 0.0,
                "generated_exact_lcb_minimum": 0.0,
                "generated_nonzero_lcb_minimum": 0.0,
                "answer_logprob_lcb_minimum": 0.0,
            },
            "prose_gates": {
                "mean_token_logprob_delta_minimum": 0.0,
                "perplexity_delta_maximum": 0.0,
                "paired_document_bootstrap_lcb_minimum": 0.0,
            },
            "protocol_or_leak_counter_increase_allowed": False,
            "selection_or_promotion_role": "none; OOD evidence only",
        },
        "artifacts": {
            "run_directory": str(runtime.RUN_DIR),
            "attempt": str(runtime.ATTEMPT),
            "complete_attempt": str(
                runtime.ATTEMPT.with_suffix(".complete.json")
            ),
            "failure": str(runtime.FAILURE),
            "gpu_log": str(runtime.GPU_LOG),
            "raw_local": str(runtime.RAW),
            "aggregate_report": str(runtime.REPORT),
            "raw_local_mode": "0600",
            "raw_local_git_eligible": False,
            "aggregate_report_contains_no_question_answer_or_prose_text": True,
        },
        "access_firewall": {
            "ood_semantics_opened_during_preregistration": False,
            "ood_semantics_opened_during_dry_run": False,
            "shadow_semantics_opened": False,
            "terminal_holdout_semantics_opened": False,
            "training_dataset_opened": False,
            "gpu_accessed": False,
            "evaluation_launched": False,
        },
    }
    value["content_sha256_before_self_field"] = core.canonical_sha256(value)
    return value


def launch_command(
    path: Path, file_sha: str, content_sha: str, *, execute: bool = True
) -> list[str]:
    command = [
        str(ROOT / "es-at-scale/.venv/bin/python"),
        str(Path(runtime.__file__).resolve()),
        "--preregistration", str(path.resolve()),
        "--preregistration-sha256", file_sha,
        "--preregistration-content-sha256", content_sha,
    ]
    command.append("--execute" if execute else "--dry-run")
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build()
    core.atomic_json(output, value)
    file_sha = core.file_sha256(output)
    content_sha = value["content_sha256_before_self_field"]
    print(json.dumps({
        "path": str(output),
        "file_sha256": file_sha,
        "content_sha256": content_sha,
        "dry_run_command": shlex.join(launch_command(
            output, file_sha, content_sha, execute=False
        )),
        "execute_command": shlex.join(launch_command(
            output, file_sha, content_sha, execute=True
        )),
        "protected_semantic_access_count": 0,
        "train_rows_opened": 0,
        "shadow_opened": False,
        "terminal_holdout_opened": False,
        "gpu_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
