#!/usr/bin/env python3
"""Audit and CPU-stage the completed V53A v440 adapter for vLLM."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import stage_candidate_adapters_vllm_v44a as prior


ROOT = Path(__file__).resolve().parent
ARM = "v440_equal"
RUN_ROOT = (
    ROOT / "experiments/sft_controls/v53a_v440_equal_lr5p5e5"
).resolve()
SOURCE = (RUN_ROOT / "v440_equal_r32_seed17_init20260715041/final").resolve()
REPORT = (RUN_ROOT / "runtime_report_v53a.json").resolve()
ATTEMPT = (RUN_ROOT / "attempt_v53a.json").resolve()
GPU_LOG = (RUN_ROOT / "gpu_activity_v53a.jsonl").resolve()
PREREGISTRATION = (RUN_ROOT / "preregistration_v53a.json").resolve()
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
    "v440_equal_sft_qwen35_vllm_namespace_v54a"
).resolve()
EXPECTED = {
    "weights": "d3f116432fff6639c0006981c9656c45573ef2ea68acdca7ba6b5d18f50dadb8",
    "config": "45c2329fad2e6ccdd5244ec456d12533875f972ca2adacc069cab3f9c90373e4",
    "report": "b1f770f90a44b4732aa3a9e56183cf233fda8e683d9e3c18b2d0cfa7345feae9",
    "report_content": "0744c760b8d491c2560cdd50b709ab2ded90be048279c3f08b2105e028d6c619",
    "attempt": "e16db97c6a5b25dcf5cb6d2785c43a2d774b296953f3634b6ab248d0f8dd10fe",
    "attempt_content": "b5a89b074fbe7f4d57a78e2c9d590a86e8a56025ad74a2e2decfbf933bc49458",
    "gpu_log": "8c7cbedebd73985328bffbddc5a08042d07bb5d7c69ea9853773f086c313ce8b",
    "preregistration": "a71ddb80ca370eb217af9025a543de0c9e474cb361816708fd7c234d2823f473",
    "preregistration_content": "b17536c3cbaedfa7cf758838c2e4d0449adf6f7da05fe0c8ab71aa9506721f92",
    "weight_identity": "2ac1bbf18fcfbb7bedd2fa24b6cd23c25807348f41bfd34f82b12d37eafdf50c",
    "dataset": "5d3de8f7bf3cfa802837cb65597f6d3bcd5906090fc54729fd1a1ef153c98021",
}


def _read_self_hashed(path: Path, file_sha: str, content_sha: str) -> dict:
    if prior.file_sha256_v44a(path) != file_sha:
        raise RuntimeError(f"V54A completion seal changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != content_sha
        or prior.canonical_sha256_v44a(compact) != content_sha
    ):
        raise RuntimeError(f"V54A self-hash changed: {path}")
    return value


def source_seal_v54a(arm: str = ARM) -> dict:
    if arm != ARM:
        raise ValueError(arm)
    report = _read_self_hashed(
        REPORT, EXPECTED["report"], EXPECTED["report_content"]
    )
    attempt = _read_self_hashed(
        ATTEMPT, EXPECTED["attempt"], EXPECTED["attempt_content"]
    )
    prereg = _read_self_hashed(
        PREREGISTRATION,
        EXPECTED["preregistration"],
        EXPECTED["preregistration_content"],
    )
    artifacts = report.get("artifacts", {})
    outputs = artifacts.get("output_file_sha256", {})
    recipe = report.get("recipe", {})
    gpu = report.get("gpu_activity", {})
    observed = report.get("observed_weighting_audit", {}).get("value", {})
    if (
        report.get("schema")
        != "specialist-sft-v440-equal-train-only-runtime-v53a"
        or report.get("status")
        != "complete_train_only_state_sealed_non_train_unopened"
        or report.get("validation_ood_or_holdout_opened") is not False
        or report.get("shadow_artifact_opened") is not False
        or report.get("expected_optimizer_steps") != 48
        or report.get("output_validation", {}).get("final_global_step") != 48
        or report.get("output_validation", {}).get("final_epoch") != 3.0
        or report.get("dataset", {}).get("sha256") != EXPECTED["dataset"]
        or recipe.get("learning_rate") != 5.5e-5
        or recipe.get("prompt_mode") != "es_exact"
        or recipe.get("world_size") != 4
        or recipe.get("target_layers") != "20,21,22,23"
        or recipe.get("rank") != 32
        or recipe.get("max_steps") != 48
        or observed.get("identity_sha256") != EXPECTED["weight_identity"]
        or gpu.get("physical_gpu_ids") != [0, 1, 2, 3]
        or gpu.get("all_four_model_resident") is not True
        or gpu.get("all_four_positive_activity") is not True
        or gpu.get("activity_attributed_to_torchrun_tree") is not True
        or outputs.get("final/adapter_model.safetensors")
        != EXPECTED["weights"]
        or outputs.get("checkpoint-48/adapter_model.safetensors")
        != EXPECTED["weights"]
        or outputs.get("final/adapter_config.json") != EXPECTED["config"]
        or artifacts.get("gpu_log_sha256") != EXPECTED["gpu_log"]
        or prior.file_sha256_v44a(GPU_LOG) != EXPECTED["gpu_log"]
        or prior.file_sha256_v44a(SOURCE / "adapter_model.safetensors")
        != EXPECTED["weights"]
        or prior.file_sha256_v44a(SOURCE / "adapter_config.json")
        != EXPECTED["config"]
        or prereg.get("schema")
        != "specialist-sft-v440-equal-train-only-preregistration-v53a"
        or prereg.get("evaluation_launch_authorized") is not False
        or prereg.get("access_firewall", {}).get("eval_ood_holdout_opened")
        is not False
        or attempt.get("schema")
        != "specialist-sft-v440-equal-train-only-attempt-v53a"
        or attempt.get("status") != "complete"
        or attempt.get("returncode") != 0
        or attempt.get("final_report_sha256") != EXPECTED["report"]
    ):
        raise RuntimeError("V54A v440 train-only provenance changed")
    return {
        "schema": "v54a-v440-train-only-success-provenance",
        "arm": ARM,
        "source_weights_sha256": EXPECTED["weights"],
        "source_config_sha256": EXPECTED["config"],
        "report_file_sha256": EXPECTED["report"],
        "report_content_sha256": EXPECTED["report_content"],
        "attempt_file_sha256": EXPECTED["attempt"],
        "attempt_content_sha256": EXPECTED["attempt_content"],
        "gpu_log_file_sha256": EXPECTED["gpu_log"],
        "preregistration_file_sha256": EXPECTED["preregistration"],
        "preregistration_content_sha256": EXPECTED["preregistration_content"],
        "weight_identity_sha256": EXPECTED["weight_identity"],
        "dataset_file_sha256": EXPECTED["dataset"],
        "completed_steps": 48,
        "all_four_training_gpus_attributed_positive": True,
        "shadow_ood_holdout_or_heldout_opened": False,
    }


@contextmanager
def _injected():
    previous_candidate = prior.CANDIDATE_SPECS_V44A.get(ARM)
    previous_expected = prior.EXPECTED_V44A.get(ARM)
    previous_seal = prior._source_seal_v44a
    prior.CANDIDATE_SPECS_V44A[ARM] = (SOURCE, REPORT, OUTPUT)
    prior.EXPECTED_V44A[ARM] = {
        "weights": EXPECTED["weights"],
        "config": EXPECTED["config"],
    }
    prior._source_seal_v44a = lambda requested: (
        source_seal_v54a(requested)
        if requested == ARM
        else previous_seal(requested)
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


def audit_source_v54a() -> dict:
    source_seal_v54a()
    with _injected():
        return prior.audit_source_v44a(ARM)


def stage_v54a() -> dict:
    audit_source_v54a()
    with _injected():
        return prior.stage_one_v44a(ARM, OUTPUT)


def main() -> int:
    manifest = stage_v54a()
    print(json.dumps({
        "directory": manifest["artifact"]["directory"],
        "weights_sha256": manifest["artifact"]["weights_file_sha256"],
        "config_sha256": manifest["artifact"]["adapter_config_file_sha256"],
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
        "gpu_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
