#!/usr/bin/env python3
"""Audit and stage sealed matched-init V42I final for compact V45D."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import stage_candidate_adapters_vllm_v45a as prior


ROOT = Path(__file__).resolve().parent
ARM = "sft_v42i"
SOURCE = (
    ROOT / "experiments/sft_controls/"
    "v42i_matched_init_equal_unit_fold3_v412_lr5p5e5/"
    "middle_late_r32_seed17_init20260715041/final"
).resolve()
REPORT = (
    ROOT / "experiments/sft_controls/"
    "v42i_matched_init_equal_unit_fold3_v412_lr5p5e5/runtime_report_v42i.json"
).resolve()
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
    "v42i_sft_qwen35_vllm_namespace_v45d"
).resolve()
SPEC = {
    "source": SOURCE,
    "report": REPORT,
    "output": OUTPUT,
    "artifact_prefix": "final",
    "weights": "9e83783c20dfb5eec91b7217d885270efed8aec216c80374444dcbc55fd7dab8",
    "config": "0e8060efd40772233390f3f97ace489e473b2bc76572e7566b83afe3dd83cc51",
    "seal": "3076ff21d7d7910cc9ae33f1c00c69b10d8e72c6c8366bb1029ceca17812cee6",
    "seal_content": "16d8898b6b81da33a6968c254e2d5c5684dd6a284ee0874b9f762bfc140b4341",
    "schema": "specialist-sft-matched-init-equal-unit-runtime-v42i",
    "status": "complete_train_only_lr5p5e5_state_sealed_shadow_unopened",
    "learning_rate": 5.5e-5,
    "learning_rate_label": "5.5e-5",
    "completed_steps": 48,
}


@contextmanager
def injected_v42i_v45d():
    previous = prior.SOURCE_SPECS_V45A.get(ARM)
    prior.SOURCE_SPECS_V45A[ARM] = SPEC
    try:
        yield
    finally:
        if previous is None:
            prior.SOURCE_SPECS_V45A.pop(ARM, None)
        else:
            prior.SOURCE_SPECS_V45A[ARM] = previous


def audit_source_v45d() -> dict:
    with injected_v42i_v45d():
        result = prior.audit_source_v45a(ARM)
    report = json.loads(REPORT.read_text())
    gpu = report.get("gpu_activity", {}).get("summary", {})
    if (
        report.get("recipe", {}).get("learning_rate") != 5.5e-5
        or report.get("recipe", {}).get("only_change_from_v42b")
        != "peak cosine-schedule learning rate 1e-4 -> 5.5e-5"
        or report.get("validation_ood_or_holdout_opened") is not False
        or report.get("artifacts", {}).get("output_file_sha256", {}).get(
            "final/adapter_model.safetensors"
        ) != SPEC["weights"]
        or report.get("artifacts", {}).get("output_file_sha256", {}).get(
            "final/adapter_config.json"
        ) != SPEC["config"]
    ):
        raise RuntimeError("V45D V42I train-only evidence changed")
    return result


def stage_v45d(output: Path | None = None) -> dict:
    audit_source_v45d()
    with injected_v42i_v45d():
        return prior.stage_one_v45a(ARM, output)


def main() -> int:
    manifest = stage_v45d()
    print(json.dumps({
        "directory": manifest["artifact"]["directory"],
        "weights_sha256": manifest["artifact"]["weights_file_sha256"],
        "manifest_content_sha256": manifest[
            "content_sha256_before_self_field"
        ],
        "transformed_identity_sha256": manifest[
            "transformed_identity"
        ]["sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
