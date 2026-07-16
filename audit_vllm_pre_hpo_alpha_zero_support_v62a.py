#!/usr/bin/env python3
"""CPU-only installed-runtime audit for the V62A HPO evaluator path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import audit_vllm_fullbatch_fcfs_support_v61e as v61e


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "vllm_pre_hpo_alpha_zero_support_audit_v62a.json"
).resolve()
COMMON_GENERATION_SEED_V62A = 2_026_071_601
GENERATION_PARAMS_WITHOUT_SEED_V62A = {
    "n": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 64,
    "detokenize": True,
}
RUNTIME_CONTROLS_V62A = {
    "VLLM_BATCH_INVARIANT": False,
    "async_scheduling": False,
    "max_num_seqs": 68,
    "scheduling_policy": "fcfs",
    "enforce_eager": True,
}


file_sha256_v62a = v61e.file_sha256_v61e
canonical_sha256_v62a = v61e.canonical_sha256_v61e


def build_audit_v62a() -> dict:
    os.environ.pop("VLLM_BATCH_INVARIANT", None)
    base = v61e.build_audit_v61e()
    from vllm import SamplingParams
    from vllm.engine.arg_utils import EngineArgs

    sampling = SamplingParams(
        seed=COMMON_GENERATION_SEED_V62A,
        **GENERATION_PARAMS_WITHOUT_SEED_V62A,
    )
    fields = EngineArgs.__dataclass_fields__
    generation_projection = {
        "seed": sampling.seed,
        "n": sampling.n,
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "max_tokens": sampling.max_tokens,
        "detokenize": sampling.detokenize,
    }
    supported = (
        base["fullbatch_fcfs_controls_supported"] is True
        and base["requested_runtime_controls"] == RUNTIME_CONTROLS_V62A
        and generation_projection == {
            "seed": COMMON_GENERATION_SEED_V62A,
            **GENERATION_PARAMS_WITHOUT_SEED_V62A,
        }
        and all(key in fields for key in (
            "tensor_parallel_size",
            "enable_lora",
            "max_lora_rank",
            "max_loras",
            "max_cpu_loras",
            "max_num_seqs",
            "scheduling_policy",
            "async_scheduling",
            "enforce_eager",
        ))
    )
    value = {
        "schema": "v62a-installed-vllm-pre-hpo-alpha-zero-support-audit",
        "status": "supported" if supported else "fail_closed_unsupported",
        "required_python": base["required_python"],
        "vllm_version": base["vllm_version"],
        "installed_vllm_source_file_sha256": base["source_file_sha256"],
        "requested_runtime_controls": dict(RUNTIME_CONTROLS_V62A),
        "scheduler_config_projection": base["scheduler_config_projection"],
        "generation_sampling_projection": generation_projection,
        "engine_args_fields_present": {
            key: key in fields for key in (
                "tensor_parallel_size",
                "enable_lora",
                "max_lora_rank",
                "max_loras",
                "max_cpu_loras",
                "max_num_seqs",
                "scheduling_policy",
                "async_scheduling",
                "enforce_eager",
            )
        },
        "intended_evaluator_projection": {
            "base_model_family": "Qwen3.6-35B-A3B",
            "adapter_state": "V434",
            "physical_gpus": [0, 1, 2, 3],
            "actors": 4,
            "tensor_parallel_size_per_actor": 1,
            "rows_per_actor_call": 68,
            "ranking_rows": 64,
            "exact_sentinel_rows": 4,
            "sequential_periods": 4,
            "generation_only": True,
            "teacher_forced_requests": 0,
            "generation_completions": 1088,
        },
        "pre_hpo_alpha_zero_runtime_supported": supported,
        "global_batch_invariance_claimed": False,
        "support_audit_authorizes_gpu_launch": False,
        "model_train_semantics_or_gpu_accessed": False,
        "filesystem_paths_mutated_outside_audit_output": False,
        "protected_semantics_opened": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256_v62a(value)
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_audit_v62a()
    v61e._exclusive_write(
        output,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps({
        "path": str(output),
        "file_sha256": file_sha256_v62a(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "status": value["status"],
        "support_audit_authorizes_gpu_launch": False,
        "model_train_semantics_or_gpu_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
