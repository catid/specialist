#!/usr/bin/env python3
"""Seal V44C after the authorized schema-only parser audit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import build_matched_lora_candidate_eval_preregistration_v44a as prior_builder
import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v44c as retry


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_sft_hpo_es_fold3_ood_eval_v44c.json"
).resolve()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return result


def build() -> dict:
    cpu_preflight = retry.offline_authorized_schema_audit_v44c()
    value = {
        key: item for key, item in prior_builder.build().items()
        if key != "content_sha256_before_self_field"
    }
    value.update({
        "schema": "matched-lora-candidate-eval-preregistration-v44c",
        "status": "preregistered_after_schema_audit_before_fresh_parser_retry",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_bindings": retry.implementation_bindings_v44c(),
        "retry_of": retry.failed_launch_provenance_v44c(),
        "retry_reason": (
            "V44B exposed an explicit-field-only OOD QA schema that the strict "
            "serialized-text helper rejected; V44C accepts source question/answer "
            "fields when text is absent and still requires exact agreement when "
            "both representations exist"
        ),
        "cpu_preflight_expected": cpu_preflight,
        "raw_shadow_or_ood_content_opened_before_preregistration": True,
        "schema_only_authorized_inputs_opened_before_preregistration": True,
        "schema_audit_scope": [
            "record key inventories", "field types", "row counts",
            "strict QA parse agreement", "content-addressed aggregate identities",
        ],
        "evaluation_metrics_observed_before_v44c_preregistration": False,
        "heldout_or_holdout_access_authorized": False,
        "heldout_or_holdout_opened_during_schema_audit": False,
        "selection_rule_or_gate_changed_from_v44a": False,
        "scientific_contract_changed_from_v44a": False,
        "fresh_artifacts": {
            "run_directory": str(retry.RUN_DIR),
            "attempt": str(retry.ATTEMPT),
            "gpu_log": str(retry.GPU_LOG),
            "raw_local": str(retry.RAW),
            "report": str(retry.REPORT),
        },
    })
    value["runtime"] = dict(value["runtime"])
    value["runtime"].update({
        "required_python_environment": retry.env_retry.environment_bindings_v44b(),
        "protected_parser_preflight_before_model_creation": True,
        "protected_preflight_expected_content_sha256": cpu_preflight[
            "content_sha256_before_self_field"
        ],
        "all_four_authorized_inputs_parsed_once_before_model_creation": True,
    })
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
        "required_python": str(retry.env_retry.EXPECTED_ENV_PREFIX / "bin/python"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
