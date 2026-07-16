#!/usr/bin/env python3
"""Audit and byte-exactly stage the completed V47A SFT adapter on CPU."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import stage_candidate_adapters_vllm_v44a as prior


ROOT = Path(__file__).resolve().parent
ARM = "sft_v47a"
RUN_ROOT = (
    ROOT / "experiments/sft_controls/"
    "v47a_matched_init_equal_unit_fold3_v430_lr5p5e5"
).resolve()
SOURCE = (
    RUN_ROOT / "middle_late_r32_seed17_init20260715041/final"
).resolve()
REPORT = (RUN_ROOT / "runtime_report_v47a.json").resolve()
PREREGISTRATION = (RUN_ROOT / "preregistration_v47a.json").resolve()
ATTEMPT = (RUN_ROOT / "attempt_v47a.json").resolve()
GPU_LOG = (RUN_ROOT / "gpu_activity_v47a.jsonl").resolve()
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
    "v47a_sft_qwen35_vllm_namespace_v47b"
).resolve()
EXPECTED = {
    "weights": "a48b18e6356274043b764bf67b36a2d5d369424d6c7657115a06caa434ae8894",
    "config": "9ec17a6b05b1157d0f75fd6dc5ef4a2e5fc2b716e8bc14b99a33f5cc4f68b326",
    "report": "924089e3cbb6698d5ba0aae2d40f2985dabc157fa0c9d93324875248611d0c6a",
    "report_content": "741a78e73d769d8efa9fc9a9d2e7bf9b4df000ff57692fa77f7d57718f08bb59",
    "preregistration": "d99d7c5cc5bad01afccd954679b2a305f0eb6c74bacd055a26fc7a5eb245f589",
    "preregistration_content": "b120d65914db05e6f00b91d9db6a254a125adb27d6a7276e24ec65983ad29d58",
    "attempt": "07caa43273d5f033525a9401e31dfbae903c6ef3a533a3da1cce38c235537a0e",
    "attempt_content": "2e1e68c770fabd8c1b96b0644d1736eab0d8a4e79ba3d281dbc3002fedbc24d1",
    "gpu_log": "81a3f807256eaeac7b2c1d6ff32db4db5ba75a7e6d3f0781a886808f5775376f",
}


def _read_self_hashed(path: Path, file_sha: str, content_sha: str) -> dict:
    if prior.file_sha256_v44a(path) != file_sha:
        raise RuntimeError(f"V47B V47A seal changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != content_sha
        or prior.canonical_sha256_v44a(compact) != content_sha
    ):
        raise RuntimeError(f"V47B V47A self-hash changed: {path}")
    return value


def source_seal_v47b() -> dict:
    """Fail closed unless V47A remains the audited train-only final."""
    report = _read_self_hashed(
        REPORT, EXPECTED["report"], EXPECTED["report_content"]
    )
    preregistration = _read_self_hashed(
        PREREGISTRATION,
        EXPECTED["preregistration"],
        EXPECTED["preregistration_content"],
    )
    attempt = _read_self_hashed(
        ATTEMPT, EXPECTED["attempt"], EXPECTED["attempt_content"]
    )
    artifacts = report.get("artifacts", {})
    output_hashes = artifacts.get("output_file_sha256", {})
    recipe = report.get("recipe", {})
    gpu = report.get("gpu_activity", {})
    if (
        report.get("schema") != "specialist-sft-v430-refresh-runtime-v47a"
        or report.get("status")
        != "complete_train_only_state_sealed_shadow_ood_holdout_unopened"
        or report.get("validation_ood_or_holdout_opened") is not False
        or report.get("shadow_dev_opened") is not False
        or report.get("expected_optimizer_steps") != 48
        or report.get("output_validation", {}).get("final_global_step") != 48
        or report.get("output_validation", {}).get("final_epoch") != 3.0
        or recipe.get("learning_rate") != 5.5e-5
        or recipe.get("loss_mode")
        != "equal_conflict_unit_answer_token_mean"
        or recipe.get("prompt_mode") != "es_exact"
        or recipe.get("world_size") != 4
        or recipe.get("target_layers") != "20,21,22,23"
        or recipe.get("rank") != 32
        or recipe.get("max_steps") != 48
        or recipe.get("initialization", {}).get(
            "sealed_initialization_seed"
        ) != 20260715041
        or recipe.get("initialization", {}).get("tensor_count") != 70
        or recipe.get("initialization", {}).get("elements") != 4_528_128
        or recipe.get("initialization", {}).get("unscaled_fp32_master")
        is not True
        or gpu.get("physical_gpu_ids") != [0, 1, 2, 3]
        or gpu.get("all_four_model_resident") is not True
        or gpu.get("all_four_positive_activity") is not True
        or len(gpu.get("by_gpu", {})) != 4
        or output_hashes.get("final/adapter_model.safetensors")
        != EXPECTED["weights"]
        or output_hashes.get("checkpoint-48/adapter_model.safetensors")
        != EXPECTED["weights"]
        or output_hashes.get("final/adapter_config.json")
        != EXPECTED["config"]
        or output_hashes.get("checkpoint-48/adapter_config.json")
        != EXPECTED["config"]
        or artifacts.get("gpu_log_sha256") != EXPECTED["gpu_log"]
        or prior.file_sha256_v44a(GPU_LOG) != EXPECTED["gpu_log"]
        or preregistration.get("schema")
        != "specialist-sft-v430-refresh-preregistration-v47a"
        or preregistration.get("status") != "sealed_unlaunched_holdout_blind"
        or preregistration.get(
            "contains_external_validation_ood_or_holdout_content"
        ) is not False
        or preregistration.get("recipe", {}).get(
            "expected_optimizer_steps"
        ) != 48
        or preregistration.get("access_firewall", {}).get(
            "eval_ood_holdout_opened"
        ) is not False
        or attempt.get("schema")
        != "specialist-sft-v430-refresh-attempt-v47a"
        or attempt.get("status") != "complete"
        or attempt.get("returncode") != 0
        or attempt.get("final_report_sha256") != EXPECTED["report"]
    ):
        raise RuntimeError("V47B V47A train-only provenance changed")
    return {
        "schema": "v47a-train-only-success-provenance-v47b",
        "report_file_sha256": EXPECTED["report"],
        "report_content_sha256": EXPECTED["report_content"],
        "preregistration_file_sha256": EXPECTED["preregistration"],
        "preregistration_content_sha256": EXPECTED[
            "preregistration_content"
        ],
        "attempt_file_sha256": EXPECTED["attempt"],
        "attempt_content_sha256": EXPECTED["attempt_content"],
        "gpu_log_file_sha256": EXPECTED["gpu_log"],
        "completed_steps": 48,
        "final_epoch": 3.0,
        "state_complete": True,
        "all_four_training_gpus_attributed_positive": True,
        "selection_data_opened": False,
        "shadow_ood_holdout_or_heldout_opened": False,
    }


@contextmanager
def injected_v47a_v47b():
    previous_candidate = prior.CANDIDATE_SPECS_V44A.get(ARM)
    previous_expected = prior.EXPECTED_V44A.get(ARM)
    previous_seal = prior._source_seal_v44a
    prior.CANDIDATE_SPECS_V44A[ARM] = (SOURCE, REPORT, OUTPUT)
    prior.EXPECTED_V44A[ARM] = {
        "weights": EXPECTED["weights"],
        "config": EXPECTED["config"],
    }
    prior._source_seal_v44a = lambda requested: (
        source_seal_v47b()
        if requested == ARM else previous_seal(requested)
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


def audit_source_v47b() -> dict:
    source_seal_v47b()
    with injected_v47a_v47b():
        return prior.audit_source_v44a(ARM)


def stage_v47b(output: Path | None = None) -> dict:
    audit_source_v47b()
    with injected_v47a_v47b():
        return prior.stage_one_v44a(
            ARM, Path(output).resolve() if output is not None else None
        )


def main() -> int:
    manifest = stage_v47b()
    print(json.dumps({
        "directory": manifest["artifact"]["directory"],
        "weights_sha256": manifest["artifact"]["weights_file_sha256"],
        "config_sha256": manifest["artifact"][
            "adapter_config_file_sha256"
        ],
        "manifest_file_sha256": prior.file_sha256_v44a(
            OUTPUT / "stage_manifest_v44a.json"
        ),
        "manifest_content_sha256": manifest[
            "content_sha256_before_self_field"
        ],
        "transformed_identity_sha256": manifest[
            "transformed_identity"
        ]["sha256"],
        "protected_semantic_access_count": 0,
        "heldout_or_holdout_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
