#!/usr/bin/env python3
"""Seal eligible-only V434-equal replicated shadow evaluation."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import run_matched_lora_candidate_eval_v44a as core
import run_sft_v434_equal_replicated_shadow_only_v49e as runtime


def build() -> dict:
    proof = runtime.ood_recovery_proof_v49e()
    stages = runtime.replica_stage_bindings_v49e()
    value = {
        "schema": "v49e-v434-equal-replicated-shadow-only-preregistration",
        "status": "sealed_after_ood_eligibility_before_shadow_access",
        "evaluation_launch_authorized": True,
        "shadow_access_authorized": True,
        "heldout_or_holdout_access_authorized": False,
        "protected_semantics_opened_by_builder": False,
        "arms": list(runtime.ARMS),
        "base_duplicate_arms": list(runtime.BASE_ARMS),
        "logical_candidates": [runtime.LOGICAL_CANDIDATE],
        "logical_candidate_replicas": {
            runtime.LOGICAL_CANDIDATE: list(runtime.CANDIDATE_ARMS)
        },
        "excluded_arms": ["v434_source50_a", "v434_source50_b"],
        "source50_present_or_ranked": False,
        "one_full_fixed_wave": [
            {"arm": arm, "engine_index": engine}
            for arm, engine in runtime.arm_wave_plan_v49e()[0]
        ],
        "single_access_inputs": dict(runtime.SHADOW_INPUTS),
        "input_scope": {
            "exact_labels": ["shadow", "split_manifest"],
            "semantic_reads_during_builder_or_dry_run": 0,
            "semantic_read_count_at_runtime": 2,
            "ood_inputs_bound_or_reopened": False,
            "holdout_or_heldout_bound": False,
        },
        "ood_eligibility_proof": proof,
        "staged_adapters": stages,
        "implementation_bindings": runtime.implementation_bindings_v49e(),
        "shadow_protocol": {
            "fold": 3,
            "rows": 83,
            "conflict_units": 51,
            "uses_original_frozen_v412_shadow": True,
            "document_disjoint_from_fold3_train_required": True,
            "equal_conflict_unit_weighting": True,
            "exact_base_duplicate_equivalence_required": True,
            "candidate_replica_outputs_bit_exact_required": False,
            "rank_fields": list(runtime.RANK_FIELDS),
            "mean_two_replica_metric_vector": True,
            "lexicographic_improvement_over_exact_base_required": True,
            "each_replica_no_protocol_or_leak_counter_increase_required": True,
            "source50_excluded_before_shadow_open": True,
        },
        "runtime": {
            "model": str(core.MODEL),
            "engine_count": 4,
            "physical_gpu_ids": [0, 1, 2, 3],
            "tensor_parallel_size_per_engine": 1,
            "one_full_fixed_wave": True,
            "every_gpu_receives_one_arm": True,
            "all_four_gpus_busy_in_shadow": True,
            "identical_prompts_sampling_params_and_seed_for_all_arms": True,
            "generation_seed": core.GENERATION_SEED,
            "tuned_table_content_sha256": (
                "4c4a0d4bbb400ea1d881bea3aae144d6865c34199fbb67889eda9e92d3a2543d"
            ),
        },
        "artifacts": {
            "run_directory": str(runtime.RUN_DIR),
            "attempt": str(runtime.ATTEMPT),
            "gpu_log": str(runtime.GPU_LOG),
            "raw_local": str(runtime.RAW),
            "report": str(runtime.REPORT),
        },
        "selection_firewall": {
            "this_phase_authorizes": "eligible-only replicated shadow evidence",
            "eligible_set_frozen_before_shadow": [runtime.LOGICAL_CANDIDATE],
            "ineligible_candidates_excluded": ["v434_source50"],
            "holdout_evaluation_authorized": False,
            "selection_or_promotion_authorized": False,
        },
        "access_firewall": {
            "shadow_semantics_opened_during_preregistration": False,
            "split_semantics_opened_during_preregistration": False,
            "holdout_or_heldout_opened": False,
            "gpu_accessed": False,
            "evaluation_launched": False,
        },
    }
    value["content_sha256_before_self_field"] = core.canonical_sha256(value)
    return value


def launch_command(path: Path, file_sha: str, content_sha: str) -> list[str]:
    return [
        str(runtime.ROOT / "es-at-scale/.venv/bin/python"),
        str(Path(runtime.__file__).resolve()),
        "--preregistration", str(path.resolve()),
        "--preregistration-sha256", file_sha,
        "--preregistration-content-sha256", content_sha,
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(runtime.DEFAULT_PREREGISTRATION))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build()
    core.atomic_json(output, value)
    file_sha = core.file_sha256(output)
    content_sha = value["content_sha256_before_self_field"]
    print(json.dumps({
        "path": str(output), "file_sha256": file_sha,
        "content_sha256": content_sha,
        "launch_command": shlex.join(launch_command(output, file_sha, content_sha)),
        "protected_semantic_access_count": 0,
        "shadow_opened": False, "heldout_or_holdout_opened": False,
        "gpu_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
