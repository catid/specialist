#!/usr/bin/env python3
"""Audit and stage sealed matched-init V42K final for V45E."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import stage_candidate_adapters_vllm_v45a as prior


ROOT = Path(__file__).resolve().parent
ARM = "sft_v42k"
SOURCE = (
    ROOT / "experiments/sft_controls/"
    "v42k_matched_init_equal_unit_fold3_v412_lr5p375e5/"
    "middle_late_r32_seed17_init20260715041/final"
).resolve()
REPORT = (
    ROOT / "experiments/sft_controls/"
    "v42k_matched_init_equal_unit_fold3_v412_lr5p375e5/runtime_report_v42k.json"
).resolve()
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
    "v42k_sft_qwen35_vllm_namespace_v45e"
).resolve()
SPEC = {
    "source": SOURCE, "report": REPORT, "output": OUTPUT,
    "artifact_prefix": "final",
    "weights": "fab70b4f55413905cc826f7997f3033f51268092556a62e1e548a50892906812",
    "config": "6cd9117784c1693f6f36972283204abab585134519a57d41561b3c5e7e6da942",
    "seal": "7a68f837eafdac697623912f5de5033aea01be6f37b2f21b470480f1d85ab6dd",
    "seal_content": "099338f73319659f0819076fab7bc5ece4f730645c86580f4179830df9014b1d",
    "schema": "specialist-sft-matched-init-equal-unit-runtime-v42k",
    "status": "complete_train_only_lr5p375e5_state_sealed_shadow_unopened",
    "learning_rate": 5.375e-5, "learning_rate_label": "5.375e-5",
    "completed_steps": 48,
}


@contextmanager
def injected_v45e():
    previous = prior.SOURCE_SPECS_V45A.get(ARM)
    prior.SOURCE_SPECS_V45A[ARM] = SPEC
    try:
        yield
    finally:
        if previous is None:
            prior.SOURCE_SPECS_V45A.pop(ARM, None)
        else:
            prior.SOURCE_SPECS_V45A[ARM] = previous


def audit_source_v45e() -> dict:
    with injected_v45e():
        result = prior.audit_source_v45a(ARM)
    report = json.loads(REPORT.read_text())
    if (
        report.get("recipe", {}).get("learning_rate") != 5.375e-5
        or report.get("recipe", {}).get("only_change_from_v42b")
        != "peak cosine-schedule learning rate 1e-4 -> 5.375e-5"
        or report.get("validation_ood_or_holdout_opened") is not False
        or report.get("artifacts", {}).get("output_file_sha256", {}).get(
            "final/adapter_model.safetensors"
        ) != SPEC["weights"]
        or report.get("artifacts", {}).get("output_file_sha256", {}).get(
            "final/adapter_config.json"
        ) != SPEC["config"]
    ):
        raise RuntimeError("V45E V42K train-only evidence changed")
    return result


def stage_v45e(output: Path | None = None) -> dict:
    audit_source_v45e()
    with injected_v45e():
        return prior.stage_one_v45a(ARM, output)


def main() -> int:
    manifest = stage_v45e()
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
