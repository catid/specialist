#!/usr/bin/env python3
"""Seal the fresh replicated v434-equal versus v440-equal OOD comparison."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import run_matched_lora_candidate_eval_v44a as core
import run_sft_v434_vs_v440_replicated_ood_only_v54a as runtime


ROOT = Path(__file__).resolve().parent
OUTPUT = runtime.DEFAULT_PREREGISTRATION


def build() -> dict:
    stages = runtime.replica_stage_bindings_v54a()
    definitions = []
    for logical in runtime.LOGICAL_CANDIDATES:
        source = runtime._source_seal(logical)
        staged = runtime.canonical_stage_binding_v54a(logical)
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
            "stage_manifest_content_sha256": staged["manifest_content_sha256"],
            "transformed_identity_sha256": staged[
                "transformed_identity_sha256"
            ],
            "replicas_use_identical_staged_bytes": True,
            "replica_outputs_required_bit_exact": False,
        })
    value = {
        "schema": "sft-v434-equal-vs-v440-equal-replicated-ood-only-v54a",
        "status": "preregistered_before_fresh_ood_only_launch",
        "evaluation_launch_authorized": True,
        "heldout_or_holdout_access_authorized": False,
        "shadow_access_authorized": False,
        "protected_semantics_opened_by_builder": False,
        "arms": list(runtime.ARMS),
        "base_duplicate_arms": list(runtime.BASE_ARMS),
        "logical_candidates": list(runtime.LOGICAL_CANDIDATES),
        "logical_candidate_replicas": {
            name: list(replicas)
            for name, replicas in runtime.LOGICAL_REPLICAS.items()
        },
        "candidate_definitions": definitions,
        "staged_adapters": stages,
        "implementation_bindings": runtime.implementation_bindings_v54a(),
        "single_access_inputs": dict(runtime.OOD_INPUTS),
        "input_scope": {
            "exact_labels": ["ood_qa", "ood_prose"],
            "same_content_addressed_inputs_for_every_arm": True,
            "same_qa_and_prose_parsers_for_every_arm": True,
            "shadow_or_split_manifest_bound": False,
            "sealed_holdout_or_heldout_bound": False,
            "semantic_reads_during_builder_or_dry_run": 0,
            "semantic_read_count_at_runtime": 2,
        },
        "protocol_parent": {
            "runtime": str(Path(runtime.parent.__file__).resolve()),
            "same_ood_qa_parser": True,
            "same_ood_prose_parser": True,
            "same_qa_and_prose_metrics": True,
            "same_generation_seed": True,
            "same_bootstrap_seed_and_samples": True,
            "same_replica_eligibility_gates": True,
            "only_candidate_change": "v434_source50 replaced by v440_equal",
        },
        "runtime": {
            "model": str(core.MODEL),
            "engine_count": 4,
            "physical_gpu_ids": [0, 1, 2, 3],
            "tensor_parallel_size_per_engine": 1,
            "two_full_fixed_waves": [
                [{"arm": arm, "engine_index": engine} for arm, engine in wave]
                for wave in runtime.arm_wave_plan_v54a()
            ],
            "all_four_gpus_busy_in_every_wave": True,
            "every_gpu_receives_one_request_per_wave": True,
            "identical_prompts_sampling_parameters_and_seed_for_all_arms": True,
            "generation_seed": core.GENERATION_SEED,
            "bootstrap_seed": runtime.BOOTSTRAP_SEED,
            "bootstrap_samples": runtime.BOOTSTRAP_SAMPLES,
            "tuned_table_content_sha256": (
                "4c4a0d4bbb400ea1d881bea3aae144d6865c34199fbb67889eda9e92d3a2543d"
            ),
            "no_partial_or_third_wave": True,
            "gpu_launch_requires_explicit_parent_handoff": True,
        },
        "ood_gates": {
            "four_base_outputs_exact_on_ood_qa_and_ood_prose_required": True,
            "each_candidate_replica_independently_gated": True,
            "both_replicas_required_for_logical_eligibility": True,
            "base_relative_qa_mean_reward_delta_minimum": 0.0,
            "base_relative_qa_exact_count_delta_minimum": 0,
            "base_relative_prose_point_non_degradation_required": True,
            "base_relative_prose_document_bootstrap_lcb_minimum": 0.0,
            "protocol_or_leak_counter_increase_allowed": False,
            "paired_qa_bootstrap_ci_role": "informational_not_a_gate",
        },
        "direct_v440_vs_v434_point_gates": {
            "comparison": "mean(v440_equal replicas)-mean(v434_equal replicas)",
            "mean_reward_delta_minimum": 0.0,
            "mean_exact_count_delta_minimum": 0,
            "mean_prose_token_logprob_delta_minimum": 0.0,
            "all_three_direct_gates_required": True,
            "bootstrap_role": "informational_not_a_direct_gate",
        },
        "artifacts": {
            "run_directory": str(runtime.RUN_DIR),
            "attempt": str(runtime.ATTEMPT),
            "complete_attempt": str(runtime.ATTEMPT.with_suffix(".complete.json")),
            "gpu_log": str(runtime.GPU_LOG),
            "raw_local": str(runtime.RAW),
            "report": str(runtime.REPORT),
        },
        "selection_firewall": {
            "this_phase_authorizes": "replicated OOD eligibility evidence only",
            "shadow_ranking_authorized": False,
            "holdout_evaluation_authorized": False,
            "promotion_authorized": False,
        },
        "access_firewall": {
            "ood_semantics_opened_during_preregistration": False,
            "shadow_semantics_opened": False,
            "holdout_or_heldout_opened": False,
            "gpu_accessed": False,
            "evaluation_launched": False,
        },
    }
    value["content_sha256_before_self_field"] = core.canonical_sha256(value)
    return value


def launch_command(path: Path, file_sha: str, content_sha: str) -> list[str]:
    return [
        str(ROOT / "es-at-scale/.venv/bin/python"),
        str(Path(runtime.__file__).resolve()),
        "--preregistration", str(path.resolve()),
        "--preregistration-sha256", file_sha,
        "--preregistration-content-sha256", content_sha,
    ]


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
        "launch_command": shlex.join(launch_command(output, file_sha, content_sha)),
        "protected_semantic_access_count": 0,
        "shadow_opened": False,
        "heldout_or_holdout_opened": False,
        "gpu_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
