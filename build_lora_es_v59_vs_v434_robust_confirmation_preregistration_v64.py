#!/usr/bin/env python3
"""Seal V64 before staged semantics, base-model load, or CUDA compute."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lora_es_nested_population_v52 as design_v52
import lora_es_v59_vs_v434_robust_confirmation_v64 as analysis
import run_lora_es_baseline_census_v61a as runtime_v61a
import run_lora_es_v59_vs_v434_robust_confirmation_v64 as runtime


def build_v64() -> dict:
    eligibility = runtime.verify_v62b_eligibility_v64()
    runtime.verify_adapter_artifacts_v64()
    support = runtime.installed_two_adapter_support_v64()
    value = {
        "schema": (
            "v64-v59-vs-v434-train-only-robust-confirmation-preregistration"
        ),
        "status": (
            "preregistered_before_train_semantics_model_or_cuda_compute"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "specific_v64_confirmation_gpu_launch_authorized": True,
        "eligibility_or_static_support_alone_authorizes_launch": False,
        "builder_or_dry_run_performed_cuda_compute_launch": False,
        "update_hpo_candidate_promotion_or_protected_access_authorized": False,
        "purpose": (
            "Run one fixed train-only, four-actor, generation-only robust "
            "confirmation of immutable V59 against immutable V434 after the "
            "sealed V62B evaluator calibration; no result authorizes an "
            "update, HPO, promotion, or protected evaluation."
        ),
        "scientific_scope": runtime.scientific_scope_v64(),
        "v62b_finalized_eligibility": eligibility,
        "installed_two_adapter_static_support": support,
        "base_model_artifact_expectation": (
            runtime.base_model_artifact_expectation_v64()
        ),
        "access_contract": runtime.access_contract_v64(),
        "fixed_confirmation_recipe": runtime.fixed_recipe_v64(),
        "primary_numeric_estimator": runtime.primary_estimator_v64(),
        "required_confirmation_gates": runtime.required_gates_v64(),
        "runtime": dict(design_v52.RUNTIME_V52),
        "required_python": str(design_v52.REQUIRED_PYTHON_V52),
        "implementation_bindings": runtime.implementation_bindings_v64(),
        "artifacts": runtime._artifacts_v64(),
        "required_integrity_gates": runtime.integrity_gates_v64(),
        "raw_question_answer_prompt_or_generation_text_may_be_persisted": False,
        "warmup_raw_output_or_generation_metric_may_be_persisted": False,
        "protected_semantics_opened": False,
        "ood_shadow_holdout_or_terminal_opened": False,
    }
    value["content_sha256_before_self_field"] = (
        analysis.canonical_sha256_v64(value)
    )
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(runtime.PREREGISTRATION))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_v64()
    runtime_v61a.atomic_json_v61a(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": runtime.file_sha256_v64(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "specific_v64_confirmation_gpu_launch_authorized": True,
        "update_hpo_candidate_promotion_or_protected_access_authorized": False,
        "builder_staged_dataset_or_base_model_accessed": False,
        "builder_model_directory_bytes_read": False,
        "builder_committed_model_seal_read": True,
        "builder_cuda_compute_initialized": False,
        "builder_zero_nvidia_device_node_reads_claimed": False,
        "builder_process_wide_zero_filesystem_writes_claimed": False,
        "protected_semantics_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
