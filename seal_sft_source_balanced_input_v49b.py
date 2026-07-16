#!/usr/bin/env python3
"""Materialize the immutable v434 train projection and V49B weight audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_sft_train_only_control_v36a as engine
import sft_source_balanced_weighting_v49b as weighting


ROOT = Path(__file__).resolve().parent
RUN_DIR = (
    ROOT / "experiments/sft_controls/"
    "v49b_source_balanced_v434_fold3_lr5p5e5"
).resolve()
TRAIN = (RUN_DIR / "train_v434_fold3_v49b.jsonl").resolve()
WEIGHT_AUDIT = (RUN_DIR / "weighting_audit_v49b.json").resolve()
INPUT_MANIFEST = (RUN_DIR / "input_manifest_v49b.json").resolve()
FOLD_MANIFEST = (
    ROOT / "experiments/sft_controls/v37a_shadow_folds_v412/manifest_v37a.json"
).resolve()
FOLD_MANIFEST_FILE_SHA256 = (
    "7d2a8f2b86f9007aa2bfe8ae043be15647451cc4bbea53a18d5915085879ee9d"
)
FOLD_MANIFEST_CONTENT_SHA256 = (
    "3fcc2820e8dffe6a21198d0520365aace049735ac84bda179ea44bc8ad0881eb"
)


def build_v49b_input() -> tuple[bytes, dict, dict]:
    rows, membership = weighting.v49a.replay_v434_train_only()
    train_bytes = weighting.v49a.frozen.jsonl_bytes(rows)
    _weights, audit = weighting.compute_source_balanced_weights_v49b(rows)
    fold = json.loads(FOLD_MANIFEST.read_text(encoding="utf-8"))
    compact_fold = {
        key: item for key, item in fold.items()
        if key != "content_sha256_before_self_field"
    }
    fold_three = fold.get("folds", [None, None, None, {}])[3]
    if (
        engine.file_sha256(FOLD_MANIFEST) != FOLD_MANIFEST_FILE_SHA256
        or fold.get("content_sha256_before_self_field")
        != FOLD_MANIFEST_CONTENT_SHA256
        or engine.canonical_sha256(compact_fold)
        != FOLD_MANIFEST_CONTENT_SHA256
        or fold.get("schema") != "specialist-train-shadow-conflict-folds-v37a"
        or fold.get("status")
        != "frozen_train_derived_not_external_evaluation"
        or fold_three.get("fold") != 3
        or fold_three.get("train", {}).get("rows") != 448
        or fold_three.get("train", {}).get("conflict_units") != 208
        or fold_three.get("train_dev_conflict_unit_intersection") != 0
        or any(fold_three.get(
            "train_dev_edge_identity_intersections", {}
        ).values())
        or fold.get("policy", {}).get(
            "external_evaluation_ood_holdout_or_benchmark_opened"
        ) is not False
    ):
        raise RuntimeError("V49B frozen document-disjoint fold binding changed")
    if (
        membership["root_membership_sha256"]
        != weighting.ROOT_MEMBERSHIP_SHA256
        or membership["train_jsonl_sha256"]
        != weighting.v49a.V434_TRAIN_SHA256
    ):
        raise RuntimeError("V49B v434 train membership changed")
    audit_file_bytes = (
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    manifest = {
        "schema": "specialist-source-balanced-train-input-manifest-v49b",
        "status": "sealed_train_only_before_launch",
        "dataset": {
            "path": str(TRAIN),
            "rows": 448,
            "conflict_units": 208,
            "file_sha256": weighting.v49a.V434_TRAIN_SHA256,
            "root_membership_sha256": weighting.ROOT_MEMBERSHIP_SHA256,
            "membership_exactly_frozen_v412_fold3_train": True,
        },
        "weighting_audit": {
            "path": str(WEIGHT_AUDIT),
            "file_sha256": weighting.v49a.hashlib.sha256(
                audit_file_bytes
            ).hexdigest(),
            "content_sha256": audit["content_sha256_before_self_field"],
            "alternative_normalized_weight_sha256": (
                weighting.ALTERNATIVE_NORMALIZED_WEIGHT_SHA256
            ),
            "per_row_identity_sha256": audit["per_row_identity_sha256"],
            "per_source_identity_sha256": audit[
                "per_source_identity_sha256"
            ],
            "per_category_identity_sha256": audit[
                "per_category_identity_sha256"
            ],
        },
        "document_disjoint_membership": {
            "manifest_path": str(FOLD_MANIFEST),
            "manifest_file_sha256": FOLD_MANIFEST_FILE_SHA256,
            "manifest_content_sha256": FOLD_MANIFEST_CONTENT_SHA256,
            "confirmatory_fold": 3,
            "train_dev_conflict_unit_intersection": 0,
            "train_dev_edge_identity_intersections": dict(
                fold_three["train_dev_edge_identity_intersections"]
            ),
            "membership_replayed_by_content_free_root_identity_only": True,
            "non_train_rows_opened": False,
        },
        "v49a_parent": {
            "path": str(weighting.V49A_DESIGN),
            "file_sha256": weighting.V49A_DESIGN_FILE_SHA256,
            "content_sha256": weighting.V49A_DESIGN_CONTENT_SHA256,
            "alternative_weight_identity_inherited_exactly": True,
        },
        "access_firewall": {
            "train_semantics_opened": True,
            "shadow_semantics_opened": False,
            "eval_ood_holdout_semantics_opened": False,
            "gpu_accessed": False,
            "training_launched": False,
        },
    }
    manifest["content_sha256_before_self_field"] = engine.canonical_sha256(
        manifest
    )
    return train_bytes, audit, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(TRAIN))
    parser.add_argument("--weight-audit", default=str(WEIGHT_AUDIT))
    parser.add_argument("--manifest", default=str(INPUT_MANIFEST))
    args = parser.parse_args(argv)
    train = Path(args.train).resolve()
    audit_path = Path(args.weight_audit).resolve()
    manifest_path = Path(args.manifest).resolve()
    if any(path.exists() for path in (train, audit_path, manifest_path)):
        raise FileExistsError("V49B sealed input output already exists")
    train_bytes, audit, manifest = build_v49b_input()
    train.parent.mkdir(parents=True, exist_ok=True)
    train.write_bytes(train_bytes)
    engine.atomic_write_json(audit_path, audit)
    engine.atomic_write_json(manifest_path, manifest)
    print(json.dumps({
        "train": str(train),
        "train_file_sha256": engine.file_sha256(train),
        "weight_audit": str(audit_path),
        "weight_audit_file_sha256": engine.file_sha256(audit_path),
        "manifest": str(manifest_path),
        "manifest_file_sha256": engine.file_sha256(manifest_path),
        "manifest_content_sha256": manifest[
            "content_sha256_before_self_field"
        ],
        "gpu_accessed": False,
        "non_train_semantics_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
