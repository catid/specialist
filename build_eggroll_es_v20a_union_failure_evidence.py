#!/usr/bin/env python3
"""Seal compact evidence for the failed V20A union equivalence gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import eggroll_es_nested_tier_interaction_preregistration_v20a as prereg_v20a


ROOT = Path(__file__).resolve().parent
ATTEMPT_PATH_V20A = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "snapshot_v298_production_patch_v20a_nested_tier_interactions_"
    "10x24_plus_nested_q1_alpha_zero_middle_late_fresh_basis_"
    "union_equivalence_gate/attempts/"
    "1784092157594340483-208250-d68c2a0f34c69c95.json"
).resolve()
OUTPUT_PATH_V20A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V20A_UNION_EQUIVALENCE_FAILURE_EVIDENCE.json"
).resolve()
ATTEMPT_FILE_SHA256_V20A = (
    "03f5ef75bb6ea6279fd39dc0b313fbd230743ac6d550f27937fee77c82b49c3e"
)
ATTEMPT_CONTENT_SHA256_V20A = (
    "ed5a7cb507a6cc25509a0bf7f62779808db1034c7eead45fbcb33a080deb6bac"
)
RECIPE_CONTENT_SHA256_V20A = (
    "aab6d8d15f669c0b807a491757717f8945913afcd2b38577e1c7147d6bcbcd30"
)
SOURCE_PROVENANCE_CONTENT_SHA256_V20A = (
    "bf5acf680cc2302a2ed577d2dc8ad141cdf29833c5c6531d7ad69f42550b9262"
)
IMPLEMENTATION_BUNDLE_SHA256_V20A = (
    "d704463ca520db2f1f74f1fe1057622b6a176763d83d8c312073ee9af5dd4e9e"
)
FAILURE_SHA256_V20A = (
    "ea7887b3a9cdb936849754b58337436288a1810c08da3362eba5faf52744d0c7"
)
SOURCE_GIT_HEAD_V20A = "5f744ec9e9175b68d33dba671f3cd58df994f7ec"

canonical_sha256 = prereg_v20a.canonical_sha256
file_sha256 = prereg_v20a.file_sha256


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def load_attempt_v20a():
    if file_sha256(ATTEMPT_PATH_V20A) != ATTEMPT_FILE_SHA256_V20A:
        raise RuntimeError("v20a union failure attempt bytes changed")
    attempt = json.loads(ATTEMPT_PATH_V20A.read_text(encoding="utf-8"))
    recipe = attempt.get("recipe", {})
    source = attempt.get("source_provenance", {})
    if (
        attempt.get("content_sha256_before_self_field")
        != ATTEMPT_CONTENT_SHA256_V20A
        or attempt.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(attempt))
        or recipe.get("content_sha256_before_self_field")
        != RECIPE_CONTENT_SHA256_V20A
        or recipe.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(recipe))
        or source.get("content_sha256_before_self_field")
        != SOURCE_PROVENANCE_CONTENT_SHA256_V20A
        or source.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(source))
        or attempt.get("status") != "failed"
        or attempt.get("phase") != "inside_union_equivalence_runtime"
        or attempt.get("failure_type") != "RuntimeError"
        or attempt.get("failure_sha256") != FAILURE_SHA256_V20A
        or attempt.get("report_exists_after_attempt") is not False
        or source.get("git_head") != SOURCE_GIT_HEAD_V20A
        or source.get("implementation_bundle_sha256")
        != IMPLEMENTATION_BUNDLE_SHA256_V20A
        or recipe.get("implementation_bundle_sha256")
        != IMPLEMENTATION_BUNDLE_SHA256_V20A
        or recipe.get("model_update_applied") is True
        or any(attempt.get(key) is not False for key in (
            "model_update_applied", "checkpoint_written",
            "evaluation_surfaces_opened", "dataset_promotion_applied",
            "attribution_run_opened",
        ))
    ):
        raise RuntimeError("v20a union failure attempt contract changed")
    for item in source.get("files", {}).values():
        raw = subprocess.check_output(
            ["git", "show", f"{SOURCE_GIT_HEAD_V20A}:{item['relative_path']}"],
            cwd=ROOT,
        )
        if hashlib.sha256(raw).hexdigest() != item["file_sha256"]:
            raise RuntimeError("v20a union failure source provenance changed")
    return attempt


def build_evidence_v20a():
    attempt = load_attempt_v20a()
    recipe = attempt["recipe"]
    value = {
        "schema": "eggroll-es-union-equivalence-failure-evidence-v20a",
        "status": "valid_failed_closed_reference_equivalence_gate",
        "input_attempt": {
            "path": str(ATTEMPT_PATH_V20A),
            "file_sha256": ATTEMPT_FILE_SHA256_V20A,
            "content_sha256": ATTEMPT_CONTENT_SHA256_V20A,
            "failure_sha256": FAILURE_SHA256_V20A,
            "source_git_head": SOURCE_GIT_HEAD_V20A,
            "source_provenance_content_sha256": (
                SOURCE_PROVENANCE_CONTENT_SHA256_V20A
            ),
            "recipe_content_sha256": RECIPE_CONTENT_SHA256_V20A,
            "implementation_bundle_sha256": IMPLEMENTATION_BUNDLE_SHA256_V20A,
        },
        "failure_boundary": {
            "state": "exact_reference",
            "perturbed_state_opened": False,
            "failure_type": "RuntimeError",
            "raw_union_bit_exact_equivalence_passed": False,
            "report_persisted": False,
            "cleanup_completed": True,
            "all_gpus_released_after_cleanup": True,
        },
        "frozen_recipe": {
            "model": recipe["model"],
            "layers": recipe["layers"],
            "sigma": recipe["sigma"],
            "alpha": recipe["alpha"],
            "population_size": recipe["population_size"],
            "hardware": recipe["hardware"],
            "moe_backend": recipe["moe_backend"],
        },
        "decision": {
            "union_scoring_authorized_for_v20a": False,
            "raw_arm_scoring_remains_authoritative": True,
            "next_runtime": "separately_committed_raw_only_v20a_train_attribution",
            "model_update_authorized": False,
            "checkpoint_write_authorized": False,
            "evaluation_authorized": False,
            "dataset_promotion_authorized": False,
        },
        "contains_scores_outputs_tokens_response_vectors_or_row_content": False,
        "contains_validation_ood_heldout_or_benchmark_content": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    validate_evidence_v20a(value)
    return value


def validate_evidence_v20a(value):
    if (
        value.get("schema")
        != "eggroll-es-union-equivalence-failure-evidence-v20a"
        or value.get("status")
        != "valid_failed_closed_reference_equivalence_gate"
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(value))
        or value.get("input_attempt", {}).get("file_sha256")
        != ATTEMPT_FILE_SHA256_V20A
        or value.get("failure_boundary", {}).get(
            "raw_union_bit_exact_equivalence_passed"
        ) is not False
        or value.get("failure_boundary", {}).get("perturbed_state_opened") is not False
        or value.get("failure_boundary", {}).get("report_persisted") is not False
        or value.get("decision", {}).get("union_scoring_authorized_for_v20a")
        is not False
        or value.get("decision", {}).get("raw_arm_scoring_remains_authoritative")
        is not True
        or any(value.get("decision", {}).get(key) is not False for key in (
            "model_update_authorized", "checkpoint_write_authorized",
            "evaluation_authorized", "dataset_promotion_authorized",
        ))
        or value.get(
            "contains_scores_outputs_tokens_response_vectors_or_row_content"
        ) is not False
        or value.get("contains_validation_ood_heldout_or_benchmark_content")
        is not False
    ):
        raise RuntimeError("v20a union failure evidence changed")
    return value


def _exclusive_write(path, value):
    path = Path(path).resolve()
    if path != OUTPUT_PATH_V20A:
        raise ValueError("v20a evidence output path changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("immutable v20a failure evidence already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH_V20A))
    args = parser.parse_args(argv)
    value = build_evidence_v20a()
    _exclusive_write(Path(args.output), value)
    result = {
        "schema": "eggroll-es-union-equivalence-failure-evidence-build-v20a",
        "path": str(OUTPUT_PATH_V20A),
        "file_sha256": file_sha256(OUTPUT_PATH_V20A),
        "content_sha256": value["content_sha256_before_self_field"],
        "gpu_launched": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
