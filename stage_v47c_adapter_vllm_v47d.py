#!/usr/bin/env python3
"""Audit and byte-exactly stage the completed V47C SFT adapter on CPU."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import stage_candidate_adapters_vllm_v44a as prior


ROOT = Path(__file__).resolve().parent
ARM = "sft_v47c"
RUN_ROOT = (
    ROOT / "experiments/sft_controls/"
    "v47c_lineage_stable_v430_fold3_lr5p5e5"
).resolve()
SOURCE = (RUN_ROOT / "middle_late_r32_seed17_init20260715041/final").resolve()
REPORT = (RUN_ROOT / "runtime_report_v47c.json").resolve()
PREREGISTRATION = (RUN_ROOT / "preregistration_v47c.json").resolve()
ATTEMPT = (RUN_ROOT / "attempt_v47c.json").resolve()
GPU_LOG = (RUN_ROOT / "gpu_activity_v47c.jsonl").resolve()
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
    "v47c_sft_qwen35_vllm_namespace_v47d"
).resolve()
EXPECTED = {
    "weights": "068317edda554abd05743579b9a6cecfea936e0740b215bbc6af8a52e985d8d3",
    "config": "6848af56fe7cbc9beb7806083c573747391b79095f66ee9bd322e76413bb4e3d",
    "report": "a92f99063b9ba4f75604b5d3ffcca1774f10bec0ec8a704bf9798e38d29c64af",
    "report_content": "d173883fc384758914a11d86207abd80923ee797ccf987b1650ffa00bf4095bb",
    "preregistration": "8b5abeb9530851e16dbaa9e48750a19c7e7e7f1aae51af1d5034f947e0bb31c0",
    "preregistration_content": "8b224522e208f4e01f426b3b261795754dd4048806d290c61a5535a535896eed",
    "attempt": "2e0bbde871529954f8f78a0619cdd02dbe619f473690807276c922436c60ae44",
    "attempt_content": "4873f01c9bce5e594e68b92de24f43765c5e0b4f090c2f9f459362d896aec1a0",
    "gpu_log": "47b3377e51e5c8bf898e5daef4f32dc46cdd8c36a901257366ec5a9c56a7dc40",
}


def _read_self_hashed(path: Path, file_sha: str, content_sha: str) -> dict:
    if prior.file_sha256_v44a(path) != file_sha:
        raise RuntimeError(f"V47D V47C seal changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != content_sha
        or prior.canonical_sha256_v44a(compact) != content_sha
    ):
        raise RuntimeError(f"V47D V47C self-hash changed: {path}")
    return value


def source_seal_v47d() -> dict:
    report = _read_self_hashed(REPORT, EXPECTED["report"], EXPECTED["report_content"])
    preregistration = _read_self_hashed(
        PREREGISTRATION, EXPECTED["preregistration"],
        EXPECTED["preregistration_content"],
    )
    attempt = _read_self_hashed(
        ATTEMPT, EXPECTED["attempt"], EXPECTED["attempt_content"]
    )
    artifacts = report.get("artifacts", {})
    output_hashes = artifacts.get("output_file_sha256", {})
    recipe = report.get("recipe", {})
    gpu = report.get("gpu_activity", {})
    proof = preregistration.get("fold_binding", {})
    if (
        report.get("schema")
        != "specialist-sft-lineage-stable-v430-runtime-v47c"
        or report.get("status")
        != "complete_train_only_state_sealed_shadow_ood_holdout_unopened"
        or report.get("validation_ood_or_holdout_opened") is not False
        or report.get("shadow_dev_opened") is not False
        or report.get("expected_optimizer_steps") != 48
        or report.get("output_validation", {}).get("final_global_step") != 48
        or report.get("output_validation", {}).get("final_epoch") != 3.0
        or recipe.get("learning_rate") != 5.5e-5
        or recipe.get("loss_mode") != "equal_conflict_unit_answer_token_mean"
        or recipe.get("prompt_mode") != "es_exact"
        or recipe.get("world_size") != 4
        or recipe.get("target_layers") != "20,21,22,23"
        or recipe.get("rank") != 32
        or recipe.get("max_steps") != 48
        or gpu.get("physical_gpu_ids") != [0, 1, 2, 3]
        or gpu.get("all_four_model_resident") is not True
        or gpu.get("all_four_positive_activity") is not True
        or len(gpu.get("by_gpu", {})) != 4
        or output_hashes.get("final/adapter_model.safetensors")
        != EXPECTED["weights"]
        or output_hashes.get("checkpoint-48/adapter_model.safetensors")
        != EXPECTED["weights"]
        or output_hashes.get("final/adapter_config.json") != EXPECTED["config"]
        or output_hashes.get("checkpoint-48/adapter_config.json")
        != EXPECTED["config"]
        or artifacts.get("gpu_log_sha256") != EXPECTED["gpu_log"]
        or prior.file_sha256_v44a(GPU_LOG) != EXPECTED["gpu_log"]
        or preregistration.get("schema")
        != "specialist-sft-lineage-stable-v430-preregistration-v47c"
        or preregistration.get("status") != "sealed_unlaunched_holdout_blind"
        or preregistration.get("contains_external_validation_ood_or_holdout_content")
        is not False
        or preregistration.get("recipe", {}).get("expected_optimizer_steps") != 48
        or preregistration.get("access_firewall", {}).get("eval_ood_holdout_opened")
        is not False
        or proof.get("train_rows") != 448
        or proof.get("train_conflict_units") != 208
        or proof.get("shadow_rows_aggregate_only") != 83
        or proof.get("shadow_conflict_units_aggregate_only") != 51
        or proof.get("original_root_membership_assignment_retained") is not True
        or proof.get("fold_assignment_changes") != 0
        or attempt.get("schema")
        != "specialist-sft-lineage-stable-v430-attempt-v47c"
        or attempt.get("status") != "complete"
        or attempt.get("returncode") != 0
        or attempt.get("final_report_sha256") != EXPECTED["report"]
    ):
        raise RuntimeError("V47D V47C train-only provenance changed")
    return {
        "schema": "v47c-train-only-success-provenance-v47d",
        "report_file_sha256": EXPECTED["report"],
        "report_content_sha256": EXPECTED["report_content"],
        "preregistration_file_sha256": EXPECTED["preregistration"],
        "preregistration_content_sha256": EXPECTED["preregistration_content"],
        "attempt_file_sha256": EXPECTED["attempt"],
        "attempt_content_sha256": EXPECTED["attempt_content"],
        "gpu_log_file_sha256": EXPECTED["gpu_log"],
        "completed_steps": 48,
        "final_epoch": 3.0,
        "state_complete": True,
        "all_four_training_gpus_attributed_positive": True,
        "lineage_stable_fold_assignment": True,
        "selection_data_opened": False,
        "shadow_ood_holdout_or_heldout_opened": False,
    }


@contextmanager
def injected_v47c_v47d():
    previous_candidate = prior.CANDIDATE_SPECS_V44A.get(ARM)
    previous_expected = prior.EXPECTED_V44A.get(ARM)
    previous_seal = prior._source_seal_v44a
    prior.CANDIDATE_SPECS_V44A[ARM] = (SOURCE, REPORT, OUTPUT)
    prior.EXPECTED_V44A[ARM] = {
        "weights": EXPECTED["weights"], "config": EXPECTED["config"],
    }
    prior._source_seal_v44a = lambda requested: (
        source_seal_v47d() if requested == ARM else previous_seal(requested)
    )
    try:
        yield
    finally:
        if previous_candidate is None:
            prior.CANDIDATE_SPECS_V44A.pop(ARM, None)
        else:
            prior.CANDIDATE_SPECS_V44A[ARM] = previous_candidate
        if previous_expected is None:
            prior.EXPECTED_V44A.pop(ARM, None)
        else:
            prior.EXPECTED_V44A[ARM] = previous_expected
        prior._source_seal_v44a = previous_seal


def audit_source_v47d() -> dict:
    source_seal_v47d()
    with injected_v47c_v47d():
        return prior.audit_source_v44a(ARM)


def stage_v47d(output: Path | None = None) -> dict:
    audit_source_v47d()
    with injected_v47c_v47d():
        return prior.stage_one_v44a(
            ARM, Path(output).resolve() if output is not None else None
        )


def main() -> int:
    manifest = stage_v47d()
    print(json.dumps({
        "directory": manifest["artifact"]["directory"],
        "weights_sha256": manifest["artifact"]["weights_file_sha256"],
        "config_sha256": manifest["artifact"]["adapter_config_file_sha256"],
        "manifest_file_sha256": prior.file_sha256_v44a(
            OUTPUT / "stage_manifest_v44a.json"
        ),
        "manifest_content_sha256": manifest["content_sha256_before_self_field"],
        "transformed_identity_sha256": manifest["transformed_identity"]["sha256"],
        "protected_semantic_access_count": 0,
        "heldout_or_holdout_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
