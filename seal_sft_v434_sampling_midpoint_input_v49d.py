#!/usr/bin/env python3
"""Seal the common V49D v434 train payload and both weighting audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import run_sft_train_only_control_v36a as engine
import seal_sft_source_balanced_input_v49b as v49b_input
import sft_v434_sampling_midpoint_weighting_v49d as weighting


ROOT = Path(__file__).resolve().parent
RUN_DIR = (
    ROOT / "experiments/sft_controls/v49d_v434_sampling_midpoint_lr5p5e5"
).resolve()
TRAIN = (RUN_DIR / "train_v434_fold3_v49d.jsonl").resolve()
WEIGHT_AUDITS = {
    "v434_equal": (RUN_DIR / "weighting_audit_v434_equal_v49d.json").resolve(),
    "v434_source50": (
        RUN_DIR / "weighting_audit_v434_source50_v49d.json"
    ).resolve(),
}
INPUT_MANIFEST = (RUN_DIR / "input_manifest_v49d.json").resolve()


def _json_bytes(value: dict) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def build_v49d_input() -> tuple[bytes, dict[str, dict], dict]:
    train_bytes, _v49b_audit, parent_manifest = v49b_input.build_v49b_input()
    rows = [json.loads(line) for line in train_bytes.splitlines() if line]
    computed = weighting.compute_v49d(rows)
    audits = {arm: complete for arm, (_weights, complete) in computed.items()}
    parent_disjoint = parent_manifest["document_disjoint_membership"]
    if (
        hashlib.sha256(train_bytes).hexdigest()
        != weighting.v49b.v49a.V434_TRAIN_SHA256
        or parent_disjoint["train_dev_conflict_unit_intersection"] != 0
        or any(parent_disjoint["train_dev_edge_identity_intersections"].values())
        or parent_disjoint["non_train_rows_opened"] is not False
    ):
        raise RuntimeError("V49D parent train or disjointness binding changed")
    audit_bindings = {}
    for arm in weighting.ARMS:
        audit = audits[arm]
        audit_bindings[arm] = {
            "path": str(WEIGHT_AUDITS[arm]),
            "file_sha256": hashlib.sha256(_json_bytes(audit)).hexdigest(),
            "content_sha256": audit["content_sha256_before_self_field"],
            "normalized_weight_sha256": audit["identity_sha256"],
            "trainer_weight_sha256": audit[
                "trainer_example_weight_identity_sha256"
            ],
            "per_row_sha256": audit["per_row_identity_sha256"],
            "per_source_sha256": audit["per_source_identity_sha256"],
            "per_category_sha256": audit["per_category_identity_sha256"],
        }
    manifest = {
        "schema": "specialist-v434-sampling-midpoint-input-manifest-v49d",
        "status": "sealed_train_only_before_launch",
        "dataset": {
            "path": str(TRAIN),
            "rows": 448,
            "conflict_units": 208,
            "file_sha256": weighting.v49b.v49a.V434_TRAIN_SHA256,
            "root_membership_sha256": weighting.v49b.ROOT_MEMBERSHIP_SHA256,
            "membership_exactly_frozen_v412_fold3_train": True,
            "same_exact_bytes_for_both_arms": True,
        },
        "weighting_audits": audit_bindings,
        "arm_order": list(weighting.ARMS),
        "controlled_contrast": {
            "only_difference": "per-row Trainer example weights",
            "equal_lambda": 0.0,
            "source50_lambda": 0.5,
            "source50_exact_multiplier_range": ["5/6", "5/4"],
            "v49a_full_parent_multiplier_range": ["2/3", "3/2"],
            "no_other_lambda_or_hpo_arm_authorized": True,
        },
        "document_disjoint_membership": dict(parent_disjoint),
        "parents": {
            "v49b_input_manifest_content_sha256": parent_manifest[
                "content_sha256_before_self_field"
            ],
            "v49a_design_file_sha256": weighting.v49b.V49A_DESIGN_FILE_SHA256,
            "v49a_design_content_sha256": (
                weighting.v49b.V49A_DESIGN_CONTENT_SHA256
            ),
            "v49a_full_weight_identity_sha256": (
                weighting.v49b.ALTERNATIVE_NORMALIZED_WEIGHT_SHA256
            ),
            "equal_weight_identity_sha256": (
                weighting.v49b.CURRENT_NORMALIZED_WEIGHT_SHA256
            ),
        },
        "access_firewall": {
            "train_semantics_opened": True,
            "shadow_semantics_opened": False,
            "eval_ood_holdout_semantics_opened": False,
            "non_train_rows_opened": False,
            "gpu_accessed": False,
            "training_launched": False,
            "evaluation_launched": False,
        },
    }
    manifest["content_sha256_before_self_field"] = engine.canonical_sha256(
        manifest
    )
    return train_bytes, audits, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(RUN_DIR))
    args = parser.parse_args(argv)
    if Path(args.run_dir).resolve() != RUN_DIR:
        raise ValueError("V49D sealing path is immutable")
    outputs = [TRAIN, INPUT_MANIFEST, *WEIGHT_AUDITS.values()]
    if any(path.exists() for path in outputs):
        raise FileExistsError("V49D sealed input output already exists")
    train_bytes, audits, manifest = build_v49d_input()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN.write_bytes(train_bytes)
    for arm, audit in audits.items():
        engine.atomic_write_json(WEIGHT_AUDITS[arm], audit)
    engine.atomic_write_json(INPUT_MANIFEST, manifest)
    print(json.dumps({
        "train": str(TRAIN),
        "train_file_sha256": engine.file_sha256(TRAIN),
        "weight_audit_file_sha256": {
            arm: engine.file_sha256(path)
            for arm, path in WEIGHT_AUDITS.items()
        },
        "manifest": str(INPUT_MANIFEST),
        "manifest_file_sha256": engine.file_sha256(INPUT_MANIFEST),
        "manifest_content_sha256": manifest[
            "content_sha256_before_self_field"
        ],
        "gpu_accessed": False,
        "non_train_semantics_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
