#!/usr/bin/env python3
"""Audit and stage sealed matched-init V42H in vLLM's namespace."""

from __future__ import annotations

import json
from pathlib import Path

import stage_candidate_adapters_vllm_v44a as prior


ROOT = Path(__file__).resolve().parent
ARM = "sft_v42h"
SOURCE = (
    ROOT / "experiments/sft_controls/"
    "v42h_matched_init_equal_unit_fold3_v412_lr6e5/"
    "middle_late_r32_seed17_init20260715041/final"
).resolve()
REPORT = (
    ROOT / "experiments/sft_controls/"
    "v42h_matched_init_equal_unit_fold3_v412_lr6e5/runtime_report_v42h.json"
).resolve()
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
    "v42h_sft_qwen35_vllm_namespace_v45c"
).resolve()
EXPECTED = {
    "weights": "a8e4b7c45490c18b4a3f2e9f88cef0a943da82ec69c129282040a47891c9296c",
    "config": "8da87ae343a460345b074da23be3dceddb263b459e64cd5db1d14b02a4905ef7",
    "report": "45fea5d0bae1658b4ef388a501238471ab894876b108cca9cd514143140fcc2b",
    "report_content": "275a645c4b7a8689649390c408c0767c3cd09c679b609e4c32d0e6dcf573088f",
}


def source_seal_v45c() -> dict:
    if prior.file_sha256_v44a(REPORT) != EXPECTED["report"]:
        raise RuntimeError("V45C V42H report file changed")
    value = json.loads(REPORT.read_text())
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    artifacts = value.get("artifacts", {}).get("output_file_sha256", {})
    recipe = value.get("recipe", {})
    initialization = recipe.get("initialization", {})
    if (
        value.get("content_sha256_before_self_field")
        != EXPECTED["report_content"]
        or prior.canonical_sha256_v44a(compact) != EXPECTED["report_content"]
        or value.get("schema")
        != "specialist-sft-matched-init-equal-unit-runtime-v42h"
        or value.get("status")
        != "complete_train_only_lr6e5_state_sealed_shadow_unopened"
        or value.get("validation_ood_or_holdout_opened") is not False
        or artifacts.get("final/adapter_model.safetensors")
        != EXPECTED["weights"]
        or artifacts.get("final/adapter_config.json") != EXPECTED["config"]
        or recipe.get("learning_rate") != 6e-5
        or recipe.get("only_change_from_v42b")
        != "peak cosine-schedule learning rate 1e-4 -> 6e-5"
        or recipe.get("loss_mode") != "equal_conflict_unit_answer_token_mean"
        or recipe.get("prompt_mode") != "es_exact"
        or recipe.get("world_size") != 4
        or recipe.get("target_layers") != "20,21,22,23"
        or initialization.get("sealed_initialization_seed") != 20260715041
        or initialization.get("tensor_count") != 70
        or initialization.get("elements") != 4_528_128
        or initialization.get("unscaled_fp32_master") is not True
    ):
        raise RuntimeError("V45C V42H matched train-only evidence changed")
    return {
        "schema": "matched-sft-success-seal-v45c",
        "report_file_sha256": EXPECTED["report"],
        "report_content_sha256": EXPECTED["report_content"],
        "learning_rate": 6e-5,
        "completed_steps": 48,
        "state_complete": True,
        "selection_data_opened": False,
        "heldout_or_holdout_opened": False,
    }


def _inject_v45c():
    old_candidate = prior.CANDIDATE_SPECS_V44A.get(ARM)
    old_expected = prior.EXPECTED_V44A.get(ARM)
    old_seal = prior._source_seal_v44a
    prior.CANDIDATE_SPECS_V44A[ARM] = (SOURCE, REPORT, OUTPUT)
    prior.EXPECTED_V44A[ARM] = {
        "weights": EXPECTED["weights"], "config": EXPECTED["config"]
    }
    prior._source_seal_v44a = lambda requested: (
        source_seal_v45c() if requested == ARM else old_seal(requested)
    )
    return old_candidate, old_expected, old_seal


def _restore_v45c(saved) -> None:
    old_candidate, old_expected, old_seal = saved
    if old_candidate is None:
        prior.CANDIDATE_SPECS_V44A.pop(ARM, None)
    else:
        prior.CANDIDATE_SPECS_V44A[ARM] = old_candidate
    if old_expected is None:
        prior.EXPECTED_V44A.pop(ARM, None)
    else:
        prior.EXPECTED_V44A[ARM] = old_expected
    prior._source_seal_v44a = old_seal


def audit_source_v45c() -> dict:
    saved = _inject_v45c()
    try:
        return prior.audit_source_v44a(ARM)
    finally:
        _restore_v45c(saved)


def stage_v45c(output: Path | None = None) -> dict:
    audit_source_v45c()
    saved = _inject_v45c()
    try:
        return prior.stage_one_v44a(
            ARM, Path(output).resolve() if output is not None else None
        )
    finally:
        _restore_v45c(saved)


def main() -> int:
    manifest = stage_v45c()
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
