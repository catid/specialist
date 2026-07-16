#!/usr/bin/env python3
"""Audit and byte-exactly stage the completed V49B SFT adapter on CPU."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import stage_candidate_adapters_vllm_v44a as prior


ROOT = Path(__file__).resolve().parent
ARM = "sft_v49b"
RUN_ROOT = (
    ROOT / "experiments/sft_controls/"
    "v49b_source_balanced_v434_fold3_lr5p5e5"
).resolve()
SOURCE = (
    RUN_ROOT / "middle_late_r32_seed17_init20260715041_retry1/final"
).resolve()
REPORT = (RUN_ROOT / "runtime_report_v49b_retry1.json").resolve()
PREREGISTRATION = (RUN_ROOT / "preregistration_v49b_retry1.json").resolve()
ATTEMPT = (RUN_ROOT / "attempt_v49b_retry1.json").resolve()
GPU_LOG = (RUN_ROOT / "gpu_activity_v49b_retry1.jsonl").resolve()
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
    "v49b_sft_qwen35_vllm_namespace_v49c"
).resolve()

# These identities were sealed only after runtime_report_v49b_retry1 proved the
# exact preregistered 48-step train-only retry completed.  Leaving any value
# pending makes every audit/staging entry point fail closed.
EXPECTED = {
    "weights": "f1cb2d287f96c0b37d288783815a399d7ffa6c44554ec56e93e94b7970d80d83",
    "config": "941656d9035478ccceefc5693f1f022e8dc4f4366d8d4e8401ca4d7e5f72493f",
    "report": "961c0dbb6b37402927ff262ef7c623d8b4c2ea1b9feb0e0bac472351fc63cd1b",
    "report_content": "1ac883a4704d79b3f5bb6be3571b6f9bceabf77caaf6c945e4ff1ce7a4abf3a5",
    "preregistration": (
        "e1f9233e9f0892623bf7a74d3ccfccd7cd4a3be4e25866ee56658e084ff5545f"
    ),
    "preregistration_content": (
        "d5db1f7afe83dfa3b7c13e0e452e2af372e2f7429894a7f83250312289a42b05"
    ),
    "attempt": "4701455885175c6e32752ecddc6881be07fbae3c99eff592e0ea085d7e488a4d",
    "attempt_content": "1ada70165e9591a56015a94bc39fb749379dd729a566584088a42984db460d51",
    "gpu_log": "4ba2d47e76c58ab6c47419a6af6b60929822e0ac40d1b84ad1d5e4819e1a2eac",
}
EXPECTED_WEIGHT_IDENTITY = (
    "76dd9224cde643b2dd22123c2bd7952a830809a3ed84c977d61208da874de612"
)
EXPECTED_ROW_IDENTITY = (
    "12175f8a48150f2ee04942334a12d2255da73d1d20edfe5fe391f2f37313f90d"
)
EXPECTED_SOURCE_IDENTITY = (
    "3ab5d46ab944944137fabe1bd20db95000292c89804340f5d061b56bb1da77e0"
)
EXPECTED_CATEGORY_IDENTITY = (
    "e248f3d1eea9de0445248189bc4b9264447978d4736cb1b759b5a60c083f08d9"
)


def completion_identities_sealed_v49c() -> bool:
    return all(
        isinstance(value, str) and len(value) == 64 and "PENDING" not in value
        for value in EXPECTED.values()
    )


def _read_self_hashed(path: Path, file_sha: str, content_sha: str) -> dict:
    if not completion_identities_sealed_v49c():
        raise RuntimeError("V49C refuses V49B access before completion is sealed")
    if prior.file_sha256_v44a(path) != file_sha:
        raise RuntimeError(f"V49C V49B seal changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != content_sha
        or prior.canonical_sha256_v44a(compact) != content_sha
    ):
        raise RuntimeError(f"V49C V49B self-hash changed: {path}")
    return value


def source_seal_v49c() -> dict:
    """Require the complete train-only V49B receipt before reading weights."""
    report = _read_self_hashed(REPORT, EXPECTED["report"], EXPECTED["report_content"])
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
    observed = report.get("observed_weighting_audit", {}).get("value", {})
    prereg_recipe = preregistration.get("recipe", {})
    manifest = preregistration.get("input_manifest", {})
    disjoint = manifest.get("document_disjoint_membership", {})
    if (
        report.get("schema") != "specialist-sft-source-balanced-runtime-v49b"
        or report.get("status")
        != "complete_train_only_state_sealed_non_train_unopened"
        or report.get("validation_ood_or_holdout_opened") is not False
        or report.get("shadow_semantics_opened") is not False
        or report.get("expected_optimizer_steps") != 48
        or report.get("output_validation", {}).get("final_global_step") != 48
        or report.get("output_validation", {}).get("final_epoch") != 3.0
        or recipe.get("learning_rate") != 5.5e-5
        or recipe.get("loss_mode")
        != "V49A_source_balanced_answer_token_example_mean"
        or recipe.get("prompt_mode") != "es_exact"
        or recipe.get("world_size") != 4
        or recipe.get("target_layers") != "20,21,22,23"
        or recipe.get("rank") != 32
        or recipe.get("max_steps") != 48
        or observed.get("identity_sha256") != EXPECTED_WEIGHT_IDENTITY
        or observed.get("per_row_identity_sha256") != EXPECTED_ROW_IDENTITY
        or observed.get("per_source_identity_sha256") != EXPECTED_SOURCE_IDENTITY
        or observed.get("per_category_identity_sha256")
        != EXPECTED_CATEGORY_IDENTITY
        or observed.get("only_per_row_example_weights_changed") is not True
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
        != "specialist-sft-source-balanced-preregistration-v49b"
        or preregistration.get("status") != "sealed_unlaunched_train_only"
        or preregistration.get(
            "contains_external_validation_ood_or_holdout_content"
        ) is not False
        or prereg_recipe.get("expected_optimizer_steps") != 48
        or prereg_recipe.get("only_change_from_matched_parent")
        != "per-row example weights use the exact V49A alternative identity"
        or preregistration.get("access_firewall", {}).get(
            "eval_ood_holdout_opened"
        ) is not False
        or preregistration.get("dataset", {}).get("rows") != 448
        or disjoint.get("train_dev_conflict_unit_intersection") != 0
        or any(disjoint.get("train_dev_edge_identity_intersections", {}).values())
        or disjoint.get("non_train_rows_opened") is not False
        or attempt.get("schema")
        != "specialist-sft-source-balanced-attempt-v49b"
        or attempt.get("status") != "complete"
        or attempt.get("returncode") != 0
        or attempt.get("final_report_sha256") != EXPECTED["report"]
    ):
        raise RuntimeError("V49C V49B train-only provenance changed")
    return {
        "schema": "v49b-train-only-success-provenance-v49c",
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
        "exact_v49a_weight_identity": EXPECTED_WEIGHT_IDENTITY,
        "only_per_row_weights_changed_from_v47c": True,
        "selection_data_opened": False,
        "shadow_ood_holdout_or_heldout_opened": False,
    }


@contextmanager
def injected_v49b_v49c():
    previous_candidate = prior.CANDIDATE_SPECS_V44A.get(ARM)
    previous_expected = prior.EXPECTED_V44A.get(ARM)
    previous_seal = prior._source_seal_v44a
    prior.CANDIDATE_SPECS_V44A[ARM] = (SOURCE, REPORT, OUTPUT)
    prior.EXPECTED_V44A[ARM] = {
        "weights": EXPECTED["weights"], "config": EXPECTED["config"],
    }
    prior._source_seal_v44a = lambda requested: (
        source_seal_v49c() if requested == ARM else previous_seal(requested)
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


def audit_source_v49c() -> dict:
    source_seal_v49c()
    with injected_v49b_v49c():
        return prior.audit_source_v44a(ARM)


def stage_v49c(output: Path | None = None) -> dict:
    audit_source_v49c()
    with injected_v49b_v49c():
        return prior.stage_one_v44a(
            ARM, Path(output).resolve() if output is not None else None
        )


def main() -> int:
    manifest = stage_v49c()
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
