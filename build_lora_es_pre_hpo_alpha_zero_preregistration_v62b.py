#!/usr/bin/env python3
"""Seal the outcome-agnostic V62B alpha-zero calibration before live access."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lora_es_nested_population_v52 as design_v52
import lora_es_pre_hpo_alpha_zero_calibration_v62b as analysis
import run_lora_es_baseline_census_v61a as runtime_v61a
import run_lora_es_pre_hpo_alpha_zero_calibration_v62b as runtime


def build_v62b() -> dict:
    runtime._read_support_audit_v62b()
    runtime._read_v62_methodology_v62b()
    value = {
        "schema": "v62b-v434-pre-hpo-alpha-zero-generation-preregistration",
        "status": "preregistered_before_train_semantics_model_or_gpu_access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "specific_alpha_zero_calibration_gpu_launch_authorized": True,
        "support_audit_alone_authorizes_gpu_launch": False,
        "builder_or_dry_run_performed_gpu_launch": False,
        "hpo_population_update_or_candidate_authorized": False,
        "ood_shadow_holdout_or_protected_access_authorized": False,
        "purpose": (
            "Run a fixed outcome-agnostic four-actor generation-only HPO "
            "evaluator calibration at alpha zero and unchanged V434 state, "
            "discard one counterbalanced warmup block, score exactly six "
            "counterbalanced blocks, and fail closed unless all three "
            "unchanged V62 gates pass."
        ),
        "v62_methodology_commit": analysis.V62_METHOD_COMMIT,
        "v62_numeric_audit_identities": dict(
            analysis.V62_NUMERIC_AUDIT_IDENTITIES
        ),
        "v62_preregistration_identities": dict(
            analysis.V62_PREREGISTRATION_IDENTITIES
        ),
        "scientific_scope": runtime.scientific_scope_v62b(),
        "installed_runtime_support_audit": (
            runtime.installed_support_binding_v62b()
        ),
        "access_contract": runtime.access_contract_v62b(),
        "fixed_calibration_recipe": runtime.fixed_recipe_v62b(),
        "primary_numeric_estimator": runtime.primary_estimator_v62b(),
        "required_alpha_zero_gates": runtime.required_gates_v62b(),
        "exact_sentinel_diagnostics": runtime.sentinel_policy_v62b(),
        "runtime": dict(design_v52.RUNTIME_V52),
        "required_python": str(design_v52.REQUIRED_PYTHON_V52),
        "implementation_bindings": runtime.implementation_bindings_v62b(),
        "artifacts": runtime._artifacts_v62b(),
        "required_integrity_gates": runtime.integrity_gates_v62b(),
        "raw_question_answer_or_generation_text_may_be_persisted": False,
        "warmup_raw_output_or_generation_metric_may_be_persisted": False,
        "protected_semantics_opened": False,
        "ood_shadow_holdout_or_terminal_opened": False,
    }
    value["content_sha256_before_self_field"] = (
        analysis.canonical_sha256_v62b(value)
    )
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(runtime.PREREGISTRATION))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_v62b()
    runtime_v61a.atomic_json_v61a(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": runtime_v61a.file_sha256_v61a(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "specific_alpha_zero_calibration_gpu_launch_authorized": True,
        "hpo_population_update_or_candidate_authorized": False,
        "builder_train_model_or_gpu_accessed": False,
        "protected_semantics_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
