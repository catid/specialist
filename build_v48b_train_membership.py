#!/usr/bin/env python3
"""Project V43I's sealed train bundle to content-free row/unit membership."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_lora_es_multi_anchor_v43i as v43i


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/datasets/"
    "v48b_train_row_conflict_unit_membership.json"
).resolve()


def build_membership_v48b() -> dict:
    bundle = v43i.equal_v38.load_equal_unit_train_bundle(
        v43i.DATASET, v43i.DATASET_SHA256,
        v43i.SPLIT_MANIFEST, v43i.SPLIT_MANIFEST_SHA256,
    )
    if bundle.get("content_sha256_before_self_field") != v43i.TRAIN_BUNDLE_SHA256:
        raise RuntimeError("v48b exact V43I train bundle changed")
    augmented = v43i.augment_unit_membership_v43i(bundle)
    items = [{
        "row_index": index,
        "row_sha256": row_sha,
        "unit_identity_sha256": membership["unit_identity_sha256"],
        "row_count": membership["row_count"],
    } for index, (row_sha, membership) in enumerate(zip(
        augmented["row_sha256"],
        augmented["unit_membership_v43i"],
        strict=True,
    ))]
    units = {}
    for item in items:
        unit = units.setdefault(item["unit_identity_sha256"], {
            "row_count": item["row_count"], "observed": 0,
        })
        if unit["row_count"] != item["row_count"]:
            raise RuntimeError("v48b conflict-unit multiplicity changed")
        unit["observed"] += 1
    if (
        len(items) != 448 or len(units) != 208
        or any(value["observed"] != value["row_count"] for value in units.values())
        or len({item["row_sha256"] for item in items}) != 448
    ):
        raise RuntimeError("v48b content-free membership coverage changed")
    result = {
        "schema": "v43i-train-row-conflict-unit-membership-v48b",
        "status": "complete_content_free_projection",
        "source": {
            "train_dataset_file_sha256": v43i.DATASET_SHA256,
            "content_free_split_commitment_file_sha256": (
                v43i.SPLIT_MANIFEST_SHA256
            ),
            "train_bundle_content_sha256": v43i.TRAIN_BUNDLE_SHA256,
            "v43i_unit_membership_sha256": augmented[
                "unit_membership_sha256_v43i"
            ],
        },
        "rows": 448,
        "conflict_units": 208,
        "items": items,
        "ordered_row_sha256": v43i.v40a.canonical_sha256(
            [item["row_sha256"] for item in items]
        ),
        "ordered_membership_sha256": v43i.v40a.canonical_sha256(items),
        "question_answer_evidence_or_text_persisted": False,
        "train_semantics_used_for_selection": False,
        "nontrain_semantics_opened": False,
        "protected_semantics_opened": False,
        "shadow_ood_holdout_or_benchmark_semantics_opened": False,
        "runtime_requires_original_split_commitment": False,
        "implementation": {
            "builder_file_sha256": v43i.v40a.file_sha256(
                Path(__file__).resolve()
            ),
            "v43i_runtime_file_sha256": v43i.v40a.file_sha256(
                Path(v43i.__file__).resolve()
            ),
            "equal_unit_loader_file_sha256": v43i.v40a.file_sha256(
                Path(v43i.equal_v38.__file__).resolve()
            ),
        },
    }
    result["content_sha256_before_self_field"] = v43i.v40a.canonical_sha256(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_membership_v48b()
    v43i.v40a.atomic_json(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": v43i.v40a.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "rows": value["rows"],
        "conflict_units": value["conflict_units"],
        "protected_semantics_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
