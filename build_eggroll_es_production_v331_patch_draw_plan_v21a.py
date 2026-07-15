#!/usr/bin/env python3
"""Seal exact paired/stratified V21A bootstrap draw commitments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

import eggroll_es_production_v331_patch_preregistration_v21a as prereg_v21a


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH_V21A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V21A_PRODUCTION_V331_PATCH_BOOTSTRAP_DRAW_PLAN.json"
).resolve()
PREREG_COMMIT_V21A = "74b9775e019a25f01d860e3508c9979646624db8"
PREREG_BUILDER_SHA256_V21A = (
    "06ffa648d9140ef2d5723d07bad7e1bb9fbde9684a1db86a0847afdc926c84df"
)
PREREG_TEST_SHA256_V21A = (
    "d46dabc63c90a0c37293f26fdc74cfe38e8b89b970dc8cb42a9fcf14a8cdfa0a"
)
PREREG_FILE_SHA256_V21A = (
    "d0a9284bb5a944b6f5059f4ed55a4616e2d00acaf2a8bdbba731603eb2126dbd"
)
PREREG_CONTENT_SHA256_V21A = (
    "f7d0814eeebf94e929421599b6ed66099db8d827aab26f67798aa73e06353dfc"
)
ROLE_PANEL_COUNTS_V21A = {"optimization": 6, "train_only_screen": 4}
CANDIDATE_SOURCE_SLOT_COUNTS_V21A = {
    "optimization": {
        "safety_consent": 12,
        "technique": 12,
        "equipment_material": 6,
        "resources_general": 6,
    },
    "train_only_screen": {
        "safety_consent": 8,
        "technique": 8,
        "equipment_material": 4,
        "resources_general": 4,
    },
}

canonical_sha256 = prereg_v21a.canonical_sha256
file_sha256 = prereg_v21a.file_sha256


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _verify_preregistration_v21a() -> dict:
    for path, digest in (
        (Path(prereg_v21a.__file__), PREREG_BUILDER_SHA256_V21A),
        (
            ROOT / "test_eggroll_es_production_v331_patch_preregistration_v21a.py",
            PREREG_TEST_SHA256_V21A,
        ),
        (prereg_v21a.OUTPUT_PATH_V21A, PREREG_FILE_SHA256_V21A),
    ):
        path = Path(path).resolve()
        relative = path.relative_to(ROOT).as_posix()
        raw = subprocess.check_output(
            ["git", "show", f"{PREREG_COMMIT_V21A}:{relative}"], cwd=ROOT
        )
        if hashlib.sha256(raw).hexdigest() != digest or file_sha256(path) != digest:
            raise RuntimeError(f"v21a draw-plan preregistration changed: {relative}")
    persisted = json.loads(prereg_v21a.OUTPUT_PATH_V21A.read_text())
    prereg_v21a.validate_preregistration_v21a(persisted)
    if (
        persisted["content_sha256_before_self_field"]
        != PREREG_CONTENT_SHA256_V21A
        or persisted["analysis"]["bootstrap"]["seed"]
        != prereg_v21a.BOOTSTRAP_SEED_V21A
        or persisted["analysis"]["bootstrap"]["repetitions"] != 50_000
        or persisted["analysis"]["hypothesis_count"] != 12
    ):
        raise RuntimeError("v21a draw-plan preregistration content changed")
    return persisted


def materialize_draw_arrays_v21a() -> dict:
    """Regenerate exact arrays in memory; never persist their values."""
    _verify_preregistration_v21a()
    rng = np.random.default_rng(prereg_v21a.BOOTSTRAP_SEED_V21A)
    base = np.empty((10, 4, 50_000, 6), dtype=np.uint8)
    for target in range(10):
        for category_index in range(4):
            base[target, category_index] = rng.integers(
                0, 6, size=(50_000, 6), dtype=np.uint8
            )
    candidate = {}
    quotas = prereg_v21a.frame_v21a.CANDIDATE_ONLY_TOPIC_QUOTAS_V21A
    for role, panel_count in ROLE_PANEL_COUNTS_V21A.items():
        candidate[role] = {}
        for topic, quota in quotas.items():
            source_count = CANDIDATE_SOURCE_SLOT_COUNTS_V21A[role][topic]
            candidate[role][topic] = rng.integers(
                0,
                source_count,
                size=(panel_count, 50_000, quota),
                dtype=np.uint8,
            )
    return {"base": base, "candidate": candidate}


def _array_contract(array: np.ndarray, **extra) -> dict:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        **extra,
    }


def build_draw_plan_certificate_v21a() -> dict:
    arrays = materialize_draw_arrays_v21a()
    candidate_contract = {}
    ordered_hashes = [hashlib.sha256(arrays["base"].tobytes()).hexdigest()]
    for role in ROLE_PANEL_COUNTS_V21A:
        candidate_contract[role] = {}
        for topic, quota in (
            prereg_v21a.frame_v21a.CANDIDATE_ONLY_TOPIC_QUOTAS_V21A.items()
        ):
            array = arrays["candidate"][role][topic]
            contract = _array_contract(
                array,
                source_slot_count=CANDIDATE_SOURCE_SLOT_COUNTS_V21A[role][topic],
                draws_per_target_panel_per_replicate=quota,
            )
            candidate_contract[role][topic] = contract
            ordered_hashes.append(contract["bytes_sha256"])
    value = {
        "schema": "eggroll-es-production-v331-patch-bootstrap-draw-plan-v21a",
        "status": "exact_draw_commitments_only_no_runtime_authority",
        "foundation": {
            "commit": PREREG_COMMIT_V21A,
            "preregistration_file_sha256": PREREG_FILE_SHA256_V21A,
            "preregistration_content_sha256": PREREG_CONTENT_SHA256_V21A,
        },
        "seed": prereg_v21a.BOOTSTRAP_SEED_V21A,
        "repetitions": 50_000,
        "familywise_alpha": 0.05,
        "hypothesis_count": 12,
        "one_sided_quantile": 0.05 / 12,
        "quantile_method": "linear",
        "base_draws": _array_contract(
            arrays["base"],
            panel_order=list(prereg_v21a.frame_v21a.PANEL_NAMES_V21A),
            category_order=list(prereg_v21a.frame_v21a.BASE_CATEGORIES_V21A),
            shared_across_both_arms=True,
        ),
        "candidate_only_draws": candidate_contract,
        "candidate_source_slot_counts": CANDIDATE_SOURCE_SLOT_COUNTS_V21A,
        "candidate_draws_same_role_and_topic": True,
        "ordered_array_commitment_sha256": canonical_sha256(ordered_hashes),
        "fixed_panel_identities_every_replicate": True,
        "paired_same_draws_both_arms": True,
        "whole_panel_block_resampling_used": False,
        "draw_arrays_persisted": False,
        "contains_train_or_evaluation_content": False,
        "runtime_launch_authorized": False,
        "gpu_launch_authorized": False,
        "model_update_authorized": False,
        "checkpoint_write_authorized": False,
        "evaluation_authorized": False,
        "dataset_promotion_authorized": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return validate_draw_plan_certificate_v21a(value)


def validate_draw_plan_certificate_v21a(value):
    candidate = value.get("candidate_only_draws", {})
    if (
        value.get("schema")
        != "eggroll-es-production-v331-patch-bootstrap-draw-plan-v21a"
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(value))
        or value.get("foundation", {}).get("commit") != PREREG_COMMIT_V21A
        or value.get("seed") != prereg_v21a.BOOTSTRAP_SEED_V21A
        or value.get("repetitions") != 50_000
        or value.get("hypothesis_count") != 12
        or value.get("one_sided_quantile") != 0.05 / 12
        or value.get("quantile_method") != "linear"
        or value.get("base_draws", {}).get("shape") != [10, 4, 50_000, 6]
        or set(candidate) != set(ROLE_PANEL_COUNTS_V21A)
        or any(
            set(candidate.get(role, {}))
            != set(prereg_v21a.frame_v21a.CANDIDATE_ONLY_TOPIC_QUOTAS_V21A)
            or any(
                candidate[role][topic].get("shape") != [
                    ROLE_PANEL_COUNTS_V21A[role],
                    50_000,
                    quota,
                ]
                or candidate[role][topic].get("source_slot_count")
                != CANDIDATE_SOURCE_SLOT_COUNTS_V21A[role][topic]
                for topic, quota in (
                    prereg_v21a.frame_v21a.CANDIDATE_ONLY_TOPIC_QUOTAS_V21A.items()
                )
            )
            for role in ROLE_PANEL_COUNTS_V21A
        )
        or value.get("candidate_draws_same_role_and_topic") is not True
        or value.get("paired_same_draws_both_arms") is not True
        or value.get("draw_arrays_persisted") is not False
        or value.get("whole_panel_block_resampling_used") is not False
        or any(value.get(key) is not False for key in (
            "runtime_launch_authorized", "gpu_launch_authorized",
            "model_update_authorized", "checkpoint_write_authorized",
            "evaluation_authorized", "dataset_promotion_authorized",
        ))
    ):
        raise RuntimeError("v21a bootstrap draw-plan certificate changed")
    return value


def _exclusive_write(path: Path, value: dict) -> None:
    path = Path(path).resolve()
    if path != OUTPUT_PATH_V21A:
        raise ValueError("v21a draw-plan output path changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("immutable v21a draw plan already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH_V21A))
    args = parser.parse_args(argv)
    value = build_draw_plan_certificate_v21a()
    _exclusive_write(Path(args.output), value)
    result = {
        "schema": "eggroll-es-production-v331-patch-bootstrap-draw-build-v21a",
        "path": str(OUTPUT_PATH_V21A),
        "file_sha256": file_sha256(OUTPUT_PATH_V21A),
        "content_sha256": value["content_sha256_before_self_field"],
        "gpu_launched": False,
        "runtime_launch_authorized": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
