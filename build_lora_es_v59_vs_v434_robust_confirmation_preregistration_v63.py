#!/usr/bin/env python3
"""Seal V63 V59-versus-V434 confirmation before any live access."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lora_es_nested_population_v52 as design_v52
import lora_es_v59_vs_v434_robust_confirmation_v63 as analysis
import run_lora_es_baseline_census_v61a as runtime_v61a
import run_lora_es_v59_vs_v434_robust_confirmation_v63 as runtime


def build_v63() -> dict:
    eligibility = runtime.verify_v62b_eligibility_v63()
    runtime.verify_adapter_artifacts_v63()
    support = runtime.installed_two_adapter_support_v63()
    value = {
        "schema": (
            "v63-v59-vs-v434-train-only-robust-confirmation-preregistration"
        ),
        "status": "preregistered_before_train_semantics_model_or_gpu_access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "specific_v63_confirmation_gpu_launch_authorized": True,
        "eligibility_or_static_support_alone_authorizes_launch": False,
        "builder_or_dry_run_performed_gpu_launch": False,
        "update_hpo_candidate_promotion_or_protected_access_authorized": False,
        "purpose": (
            "Run one fixed train-only, four-actor, generation-only robust "
            "confirmation of immutable V59 against immutable V434 after the "
            "sealed V62B evaluator calibration; no result authorizes an "
            "update, HPO, promotion, or protected evaluation."
        ),
        "scientific_scope": runtime.scientific_scope_v63(),
        "v62b_finalized_eligibility": eligibility,
        "installed_two_adapter_static_support": support,
        "access_contract": runtime.access_contract_v63(),
        "fixed_confirmation_recipe": runtime.fixed_recipe_v63(),
        "primary_numeric_estimator": runtime.primary_estimator_v63(),
        "required_confirmation_gates": runtime.required_gates_v63(),
        "runtime": dict(design_v52.RUNTIME_V52),
        "required_python": str(design_v52.REQUIRED_PYTHON_V52),
        "implementation_bindings": runtime.implementation_bindings_v63(),
        "artifacts": runtime._artifacts_v63(),
        "required_integrity_gates": runtime.integrity_gates_v63(),
        "raw_question_answer_prompt_or_generation_text_may_be_persisted": False,
        "warmup_raw_output_or_generation_metric_may_be_persisted": False,
        "protected_semantics_opened": False,
        "ood_shadow_holdout_or_terminal_opened": False,
    }
    value["content_sha256_before_self_field"] = (
        analysis.canonical_sha256_v63(value)
    )
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(runtime.PREREGISTRATION))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_v63()
    runtime_v61a.atomic_json_v61a(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": runtime.file_sha256_v63(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "specific_v63_confirmation_gpu_launch_authorized": True,
        "update_hpo_candidate_promotion_or_protected_access_authorized": False,
        "builder_staged_dataset_model_or_gpu_accessed": False,
        "protected_semantics_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
