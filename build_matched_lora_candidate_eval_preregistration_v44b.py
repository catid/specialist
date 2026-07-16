#!/usr/bin/env python3
"""Seal the environment-bound fresh V44B retry before protected access."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import build_matched_lora_candidate_eval_preregistration_v44a as prior_builder
import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v44b as retry


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_sft_hpo_es_fold3_ood_eval_v44b.json"
).resolve()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return result


def build() -> dict:
    value = {
        key: item for key, item in prior_builder.build().items()
        if key != "content_sha256_before_self_field"
    }
    value.update({
        "schema": "matched-lora-candidate-eval-preregistration-v44b",
        "status": "preregistered_fresh_env_retry_before_single_semantic_access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_bindings": retry.implementation_bindings_v44b(),
        "retry_of": retry.failed_launch_provenance_v44b(),
        "retry_reason": (
            "V44A was invoked from the training-only environment without vLLM; "
            "V44B binds the es-at-scale environment and otherwise preserves the "
            "sealed arms, metrics, selection rule, OOD gates, and protected inputs"
        ),
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
    value["runtime"]["required_python_environment"] = (
        retry.environment_bindings_v44b()
    )
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
        "protected_semantic_access_count": 0,
        "required_python": str(retry.EXPECTED_ENV_PREFIX / "bin/python"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
