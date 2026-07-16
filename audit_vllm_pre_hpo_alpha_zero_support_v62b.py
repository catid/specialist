#!/usr/bin/env python3
"""CPU-only installed-runtime support audit for the V62B evaluator path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import audit_vllm_pre_hpo_alpha_zero_support_v62a as v62a
import lora_es_pre_hpo_alpha_zero_calibration_v62b as analysis


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "vllm_pre_hpo_alpha_zero_support_audit_v62b.json"
).resolve()


file_sha256_v62b = v62a.file_sha256_v62a
canonical_sha256_v62b = v62a.canonical_sha256_v62a


def build_audit_v62b() -> dict:
    base = v62a.build_audit_v62a()
    supported = (
        base["status"] == "supported"
        and base["pre_hpo_alpha_zero_runtime_supported"] is True
        and base["requested_runtime_controls"]
        == analysis.RUNTIME_CONTROLS_V62B
        and base["generation_sampling_projection"] == {
            "seed": analysis.COMMON_GENERATION_SEED_V62B,
            **analysis.GENERATION_PARAMS_WITHOUT_SEED_V62B,
        }
        and base["intended_evaluator_projection"]["actors"]
        == analysis.ACTORS_V62B
        and base["intended_evaluator_projection"]["rows_per_actor_call"]
        == analysis.ROWS_V62B
        and base["global_batch_invariance_claimed"] is False
    )
    value = {
        "schema": "v62b-installed-vllm-pre-hpo-alpha-zero-support-audit",
        "status": "supported" if supported else "fail_closed_unsupported",
        "required_python": base["required_python"],
        "vllm_version": base["vllm_version"],
        "installed_vllm_source_file_sha256": (
            base["installed_vllm_source_file_sha256"]
        ),
        "requested_runtime_controls": dict(analysis.RUNTIME_CONTROLS_V62B),
        "scheduler_config_projection": base["scheduler_config_projection"],
        "generation_sampling_projection": (
            base["generation_sampling_projection"]
        ),
        "engine_args_fields_present": base["engine_args_fields_present"],
        "intended_evaluator_projection": {
            "base_model_family": "Qwen3.6-35B-A3B",
            "adapter_state": "V434",
            "physical_gpus": [0, 1, 2, 3],
            "actors": analysis.ACTORS_V62B,
            "tensor_parallel_size_per_actor": 1,
            "rows_per_actor_call": analysis.ROWS_V62B,
            "ranking_rows": analysis.RANKING_UNITS_V62B,
            "exact_sentinel_rows": analysis.EXACT_SENTINEL_UNITS_V62B,
            "unscored_warmup_periods": analysis.WARMUP_PERIODS_V62B,
            "scored_periods": analysis.SCORED_PERIODS_V62B,
            "total_sequential_periods": analysis.TOTAL_PERIODS_V62B,
            "scored_counterbalanced_blocks": analysis.SCORED_BLOCKS_V62B,
            "scored_pairs_per_actor": analysis.PAIRS_PER_ACTOR_V62B,
            "scored_replicas_per_conflict_unit": (
                analysis.REPLICAS_PER_UNIT_V62B
            ),
            "warmup_generation_completions_discarded": (
                analysis.WARMUP_GENERATION_COMPLETIONS_V62B
            ),
            "scored_generation_completions": (
                analysis.SCORED_GENERATION_COMPLETIONS_V62B
            ),
            "total_generation_completions": (
                analysis.TOTAL_GENERATION_COMPLETIONS_V62B
            ),
            "warmup_outputs_scored_or_persisted": False,
            "generation_only": True,
            "teacher_forced_requests": 0,
        },
        "pre_hpo_alpha_zero_runtime_supported": supported,
        "global_batch_invariance_claimed": False,
        "support_audit_authorizes_gpu_launch": False,
        "model_train_semantics_or_gpu_accessed": False,
        "filesystem_paths_mutated_outside_audit_output": False,
        "protected_semantics_opened": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256_v62b(value)
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_audit_v62b()
    v62a.v61e._exclusive_write(
        output,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps({
        "path": str(output),
        "file_sha256": file_sha256_v62b(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "status": value["status"],
        "support_audit_authorizes_gpu_launch": False,
        "model_train_semantics_or_gpu_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
