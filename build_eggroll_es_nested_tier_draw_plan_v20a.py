#!/usr/bin/env python3
"""Seal the exact aggregate-only V20A paired-bootstrap draw plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

import eggroll_es_nested_tier_interaction_preregistration_v20a as prereg_v20a


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH_V20A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V20A_NESTED_TIER_BOOTSTRAP_DRAW_PLAN.json"
).resolve()
SEALED_FOUNDATION_COMMIT_V20A = "f8860e14c693020badf25985cb2ba6b4d4339e30"
PREREG_BUILDER_FILE_SHA256_V20A = (
    "53d8d1186c5164ad1f787d4eb8966aa0b759d2ce3d9f6f371108419475dadde5"
)
PREREG_FILE_SHA256_V20A = (
    "1a7f372bf6f2af6606acc4e6d4adbf9815b96931a532aa07348335a1416d6963"
)
PREREG_CONTENT_SHA256_V20A = (
    "9ce64a2d9cc91da2dd83aeb0d1e5adf3c0a3216e69613e6a61683c001909345e"
)

canonical_sha256 = prereg_v20a.canonical_sha256
file_sha256 = prereg_v20a.file_sha256


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _verify_foundation_v20a():
    expected = {
        Path(prereg_v20a.__file__).resolve(): PREREG_BUILDER_FILE_SHA256_V20A,
        prereg_v20a.OUTPUT_PATH_V20A: PREREG_FILE_SHA256_V20A,
    }
    for path, digest in expected.items():
        relative = path.relative_to(ROOT).as_posix()
        raw = subprocess.check_output(
            ["git", "show", f"{SEALED_FOUNDATION_COMMIT_V20A}:{relative}"],
            cwd=ROOT,
        )
        if hashlib.sha256(raw).hexdigest() != digest or file_sha256(path) != digest:
            raise RuntimeError(f"v20a draw-plan foundation changed: {relative}")
    persisted = json.loads(prereg_v20a.OUTPUT_PATH_V20A.read_text(encoding="utf-8"))
    if (
        persisted.get("content_sha256_before_self_field")
        != PREREG_CONTENT_SHA256_V20A
        or persisted != prereg_v20a.build_preregistration_v20a()
        or persisted.get("analysis", {}).get("hypothesis_count") != 60
        or persisted.get("analysis", {}).get("bootstrap", {}).get("seed")
        != prereg_v20a.BOOTSTRAP_SEED_V20A
        or persisted.get("analysis", {}).get("bootstrap", {}).get("repetitions")
        != prereg_v20a.BOOTSTRAP_REPETITIONS_V20A
    ):
        raise RuntimeError("v20a draw-plan preregistration changed")
    return persisted


def materialize_draw_arrays_v20a() -> dict:
    """Regenerate exact draws in memory; arrays must never be persisted."""
    _verify_foundation_v20a()
    rng = np.random.default_rng(prereg_v20a.BOOTSTRAP_SEED_V20A)
    base = np.empty((10, 4, 50_000, 6), dtype=np.uint8)
    for target in range(10):
        for category_index in range(4):
            base[target, category_index] = rng.integers(
                0, 6, size=(50_000, 6), dtype=np.uint8
            )
    candidate_source_offsets = np.empty((3, 10, 50_000), dtype=np.uint8)
    for tier_index in range(3):
        for role_indices in (np.arange(6), np.arange(6, 10)):
            for target in role_indices:
                candidate_source_offsets[tier_index, target] = rng.integers(
                    0,
                    len(role_indices),
                    size=50_000,
                    dtype=np.uint8,
                )
    return {
        "base": base,
        "candidate_source_offsets": candidate_source_offsets,
    }


def build_draw_plan_certificate_v20a() -> dict:
    arrays = materialize_draw_arrays_v20a()
    base = arrays["base"]
    candidate = arrays["candidate_source_offsets"]
    value = {
        "schema": "eggroll-es-nested-tier-bootstrap-draw-plan-v20a",
        "status": "exact_draw_commitments_only_no_runtime_authorization",
        "foundation": {
            "commit": SEALED_FOUNDATION_COMMIT_V20A,
            "preregistration_file_sha256": PREREG_FILE_SHA256_V20A,
            "preregistration_content_sha256": PREREG_CONTENT_SHA256_V20A,
        },
        "seed": prereg_v20a.BOOTSTRAP_SEED_V20A,
        "repetitions": prereg_v20a.BOOTSTRAP_REPETITIONS_V20A,
        "familywise_alpha": prereg_v20a.FAMILYWISE_ALPHA_V20A,
        "hypothesis_count": prereg_v20a.HYPOTHESIS_COUNT_V20A,
        "one_sided_quantile": (
            prereg_v20a.FAMILYWISE_ALPHA_V20A
            / prereg_v20a.HYPOTHESIS_COUNT_V20A
        ),
        "base_draws": {
            "shape": list(base.shape),
            "dtype": str(base.dtype),
            "bytes_sha256": hashlib.sha256(base.tobytes()).hexdigest(),
            "shared_across_all_four_arms": True,
        },
        "candidate_source_offsets": {
            "shape": list(candidate.shape),
            "dtype": str(candidate.dtype),
            "bytes_sha256": hashlib.sha256(candidate.tobytes()).hexdigest(),
            "tier_order": [1, 2, 3],
            "same_role_source_panels": True,
            "shared_by_every_nested_arm_containing_each_tier": True,
        },
        "fixed_panel_identities_every_replicate": True,
        "whole_panel_block_resampling_used": False,
        "draw_arrays_persisted": False,
        "contains_train_or_evaluation_content": False,
        "runtime_launch_authorized": False,
        "model_update_authorized": False,
        "checkpoint_write_authorized": False,
        "evaluation_authorized": False,
        "dataset_promotion_authorized": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    validate_draw_plan_certificate_v20a(value)
    return value


def validate_draw_plan_certificate_v20a(value):
    if (
        value.get("schema") != "eggroll-es-nested-tier-bootstrap-draw-plan-v20a"
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(value))
        or value.get("seed") != prereg_v20a.BOOTSTRAP_SEED_V20A
        or value.get("repetitions") != 50_000
        or value.get("hypothesis_count") != 60
        or value.get("one_sided_quantile") != 0.05 / 60
        or value.get("base_draws", {}).get("shape") != [10, 4, 50_000, 6]
        or value.get("candidate_source_offsets", {}).get("shape")
        != [3, 10, 50_000]
        or value.get("draw_arrays_persisted") is not False
        or value.get("whole_panel_block_resampling_used") is not False
        or value.get("runtime_launch_authorized") is not False
        or value.get("model_update_authorized") is not False
        or value.get("checkpoint_write_authorized") is not False
        or value.get("evaluation_authorized") is not False
        or value.get("dataset_promotion_authorized") is not False
    ):
        raise RuntimeError("v20a draw-plan certificate changed")
    return value


def _exclusive_write(path, value):
    path = Path(path).resolve()
    if path != OUTPUT_PATH_V20A:
        raise ValueError("v20a draw-plan output path changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("immutable v20a draw plan already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH_V20A))
    args = parser.parse_args(argv)
    value = build_draw_plan_certificate_v20a()
    _exclusive_write(Path(args.output), value)
    result = {
        "schema": "eggroll-es-nested-tier-bootstrap-draw-plan-build-v20a",
        "path": str(OUTPUT_PATH_V20A),
        "file_sha256": file_sha256(OUTPUT_PATH_V20A),
        "content_sha256": value["content_sha256_before_self_field"],
        "gpu_launched": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
