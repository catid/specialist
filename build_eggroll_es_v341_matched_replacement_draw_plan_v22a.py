#!/usr/bin/env python3
"""Seal exact paired/stratified V22A bootstrap draw commitments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

import eggroll_es_v341_matched_replacement_preregistration_v22a as prereg_v22a


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH_V22A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V22A_V341_MATCHED_REPLACEMENT_BOOTSTRAP_DRAW_PLAN.json"
).resolve()
PREREG_COMMIT_V22A = "a5df84f9c31e9b3f8c7f601a807164df344dbff5"
PREREG_BUILDER_SHA256_V22A = (
    "f2d7e598446477a3c2ebc6f1c1da12225d34f63057866632f8cb324b4e098dbd"
)
PREREG_TEST_SHA256_V22A = (
    "621354feb9c2892fe552866c90a709dfbf082aecc221636547924fdcd9e44770"
)
PREREG_FILE_SHA256_V22A = (
    "b86a61e212af9862553119bc75f0c1bcfc264088af058585face1ac8f288e004"
)
PREREG_CONTENT_SHA256_V22A = (
    "ab0a72443a305bde922bf8fcb3cd9444a741489fc613f93e9d91eada3cc0ad08"
)

canonical_sha256 = prereg_v22a.canonical_sha256
file_sha256 = prereg_v22a.file_sha256


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _verify_preregistration_v22a() -> dict:
    for path, digest in (
        (Path(prereg_v22a.__file__), PREREG_BUILDER_SHA256_V22A),
        (
            ROOT / "test_eggroll_es_v341_matched_replacement_preregistration_v22a.py",
            PREREG_TEST_SHA256_V22A,
        ),
        (prereg_v22a.OUTPUT_PATH_V22A, PREREG_FILE_SHA256_V22A),
    ):
        path = Path(path).resolve()
        relative = path.relative_to(ROOT).as_posix()
        raw = subprocess.check_output(
            ["git", "show", f"{PREREG_COMMIT_V22A}:{relative}"], cwd=ROOT
        )
        if hashlib.sha256(raw).hexdigest() != digest or file_sha256(path) != digest:
            raise RuntimeError(f"v22a draw-plan preregistration changed: {relative}")
    persisted = json.loads(prereg_v22a.OUTPUT_PATH_V22A.read_text())
    prereg_v22a.validate_preregistration_v22a(persisted)
    bootstrap = persisted.get("analysis", {}).get("bootstrap", {})
    if (
        persisted["content_sha256_before_self_field"]
        != PREREG_CONTENT_SHA256_V22A
        or bootstrap.get("seed") != prereg_v22a.BOOTSTRAP_SEED_V22A
        or bootstrap.get("repetitions") != 50_000
        or bootstrap.get("one_sided_quantile") != 0.05 / 12
        or persisted.get("analysis", {}).get("hypothesis_count") != 12
        or persisted.get("frame_contract", {}).get(
            "candidate_only_components_excluded"
        ) != 54
    ):
        raise RuntimeError("v22a draw-plan preregistration content changed")
    return persisted


def materialize_draw_array_v22a() -> np.ndarray:
    """Regenerate exact arrays in memory; never persist their values."""
    _verify_preregistration_v22a()
    rng = np.random.default_rng(prereg_v22a.BOOTSTRAP_SEED_V22A)
    base = np.empty((10, 4, 50_000, 6), dtype=np.uint8)
    for panel_index in range(10):
        for category_index in range(4):
            base[panel_index, category_index] = rng.integers(
                0, 6, size=(50_000, 6), dtype=np.uint8
            )
    return base


def _array_contract(array: np.ndarray) -> dict:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "panel_order": list(prereg_v22a.frame_v22a.PANEL_NAMES_V22A),
        "category_order": list(prereg_v22a.frame_v22a.BASE_CATEGORIES_V22A),
        "source_slots_per_panel_category": 6,
        "draws_per_panel_category_per_replicate": 6,
        "shared_across_both_arms": True,
    }


def build_draw_plan_certificate_v22a() -> dict:
    base = materialize_draw_array_v22a()
    base_contract = _array_contract(base)
    value = {
        "schema": "eggroll-es-v341-matched-replacement-bootstrap-draw-plan-v22a",
        "status": "exact_draw_commitment_only_no_runtime_authority",
        "foundation": {
            "commit": PREREG_COMMIT_V22A,
            "preregistration_file_sha256": PREREG_FILE_SHA256_V22A,
            "preregistration_content_sha256": PREREG_CONTENT_SHA256_V22A,
        },
        "seed": prereg_v22a.BOOTSTRAP_SEED_V22A,
        "repetitions": 50_000,
        "familywise_alpha": 0.05,
        "hypothesis_count": 12,
        "one_sided_quantile": 0.05 / 12,
        "quantile_method": "linear",
        "base_draws": base_contract,
        "ordered_array_commitment_sha256": canonical_sha256([
            base_contract["bytes_sha256"]
        ]),
        "fixed_panel_identities_every_replicate": True,
        "fixed_stratum_identities_every_replicate": True,
        "paired_same_draws_both_arms": True,
        "same_ht_coefficients_and_denominator_both_arms": True,
        "candidate_only_draws_present": False,
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
    return validate_draw_plan_certificate_v22a(value)


def validate_draw_plan_certificate_v22a(value):
    expected_base = {
        "shape": [10, 4, 50_000, 6],
        "dtype": "uint8",
        "bytes_sha256": hashlib.sha256(
            materialize_draw_array_v22a().tobytes()
        ).hexdigest(),
        "panel_order": list(prereg_v22a.frame_v22a.PANEL_NAMES_V22A),
        "category_order": list(prereg_v22a.frame_v22a.BASE_CATEGORIES_V22A),
        "source_slots_per_panel_category": 6,
        "draws_per_panel_category_per_replicate": 6,
        "shared_across_both_arms": True,
    }
    expected = {
        "schema": "eggroll-es-v341-matched-replacement-bootstrap-draw-plan-v22a",
        "status": "exact_draw_commitment_only_no_runtime_authority",
        "foundation": {
            "commit": PREREG_COMMIT_V22A,
            "preregistration_file_sha256": PREREG_FILE_SHA256_V22A,
            "preregistration_content_sha256": PREREG_CONTENT_SHA256_V22A,
        },
        "seed": prereg_v22a.BOOTSTRAP_SEED_V22A,
        "repetitions": 50_000,
        "familywise_alpha": 0.05,
        "hypothesis_count": 12,
        "one_sided_quantile": 0.05 / 12,
        "quantile_method": "linear",
        "base_draws": expected_base,
        "ordered_array_commitment_sha256": canonical_sha256([
            expected_base["bytes_sha256"]
        ]),
        "fixed_panel_identities_every_replicate": True,
        "fixed_stratum_identities_every_replicate": True,
        "paired_same_draws_both_arms": True,
        "same_ht_coefficients_and_denominator_both_arms": True,
        "candidate_only_draws_present": False,
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
    if (
        _without_self(value) != expected
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(expected)
    ):
        raise RuntimeError("v22a bootstrap draw-plan certificate changed")
    return value


def _exclusive_write(path: Path, value: dict) -> None:
    path = Path(path).resolve()
    if path != OUTPUT_PATH_V22A:
        raise ValueError("v22a draw-plan output path changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("immutable v22a draw plan already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH_V22A))
    args = parser.parse_args(argv)
    value = build_draw_plan_certificate_v22a()
    _exclusive_write(Path(args.output), value)
    result = {
        "schema": "eggroll-es-v341-matched-replacement-bootstrap-draw-build-v22a",
        "path": str(OUTPUT_PATH_V22A),
        "file_sha256": file_sha256(OUTPUT_PATH_V22A),
        "content_sha256": value["content_sha256_before_self_field"],
        "base_array_bytes_sha256": value["base_draws"]["bytes_sha256"],
        "gpu_launched": False,
        "runtime_launch_authorized": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
