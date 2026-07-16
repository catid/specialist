#!/usr/bin/env python3
"""Audit and stage sealed matched-init V42J final for V45E."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import stage_candidate_adapters_vllm_v45a as prior


ROOT = Path(__file__).resolve().parent
ARM = "sft_v42j"
SOURCE = (
    ROOT / "experiments/sft_controls/"
    "v42j_matched_init_equal_unit_fold3_v412_lr5p25e5/"
    "middle_late_r32_seed17_init20260715041/final"
).resolve()
REPORT = (
    ROOT / "experiments/sft_controls/"
    "v42j_matched_init_equal_unit_fold3_v412_lr5p25e5/runtime_report_v42j.json"
).resolve()
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
    "v42j_sft_qwen35_vllm_namespace_v45e"
).resolve()
SPEC = {
    "source": SOURCE, "report": REPORT, "output": OUTPUT,
    "artifact_prefix": "final",
    "weights": "5cf2d87cb4350ba27b2b93d428afb91a2ee37786aa52dce8a2342fe04c0605e3",
    "config": "e1d4e03fbf59491a6a10679b5241deccac890daad01ae5aa854977b3677c36ab",
    "seal": "5f92b0cd21d0da63b6d473314787db9cc69a6e8cb9077f0e4b157bea4a2276a9",
    "seal_content": "89ce05d941bf324b0a69f1606a2b59e3edddef38645ab915dfaf58a26ea986c5",
    "schema": "specialist-sft-matched-init-equal-unit-runtime-v42j",
    "status": "complete_train_only_lr5p25e5_state_sealed_shadow_unopened",
    "learning_rate": 5.25e-5, "learning_rate_label": "5.25e-5",
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
        report.get("recipe", {}).get("learning_rate") != 5.25e-5
        or report.get("recipe", {}).get("only_change_from_v42b")
        != "peak cosine-schedule learning rate 1e-4 -> 5.25e-5"
        or report.get("validation_ood_or_holdout_opened") is not False
        or report.get("artifacts", {}).get("output_file_sha256", {}).get(
            "final/adapter_model.safetensors"
        ) != SPEC["weights"]
        or report.get("artifacts", {}).get("output_file_sha256", {}).get(
            "final/adapter_config.json"
        ) != SPEC["config"]
    ):
        raise RuntimeError("V45E V42J train-only evidence changed")
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
