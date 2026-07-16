#!/usr/bin/env python3
"""Byte-stage the exact V43I train JSONL under a train-only runtime path."""

from __future__ import annotations

import json
import os
from pathlib import Path

import run_lora_es_multi_anchor_v43i as v43i


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/datasets/v48b_fold3_train_v412.jsonl"
).resolve()
MANIFEST = (
    ROOT / "experiments/eggroll_es_hpo/datasets/"
    "v48b_fold3_train_v412_stage_manifest.json"
).resolve()


def stage_v48b() -> dict:
    if OUTPUT.exists() or MANIFEST.exists():
        raise FileExistsError("v48b staged train input already exists")
    if v43i.v40a.file_sha256(v43i.DATASET) != v43i.DATASET_SHA256:
        raise RuntimeError("v48b V43I source train input changed")
    raw = v43i.DATASET.read_bytes()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp-{os.getpid()}")
    temporary.write_bytes(raw)
    try:
        os.link(temporary, OUTPUT)
    finally:
        temporary.unlink(missing_ok=True)
    if OUTPUT.read_bytes() != raw or v43i.v40a.file_sha256(OUTPUT) != v43i.DATASET_SHA256:
        OUTPUT.unlink(missing_ok=True)
        raise RuntimeError("v48b staged train bytes changed")
    value = {
        "schema": "v43i-train-input-byte-stage-v48b",
        "status": "complete_byte_exact_train_only_stage",
        "source_file_sha256": v43i.DATASET_SHA256,
        "artifact": {
            "path": str(OUTPUT),
            "file_sha256": v43i.DATASET_SHA256,
            "bytes": len(raw),
        },
        "byte_exact": True,
        "semantic_transform_performed": False,
        "nontrain_or_protected_input_opened": False,
        "runtime_path_contains_protected_surface_token": False,
        "implementation_file_sha256": v43i.v40a.file_sha256(
            Path(__file__).resolve()
        ),
    }
    value["content_sha256_before_self_field"] = v43i.v40a.canonical_sha256(value)
    v43i.v40a.atomic_json(MANIFEST, value)
    return value


def main() -> int:
    value = stage_v48b()
    print(json.dumps({
        "path": str(OUTPUT),
        "file_sha256": v43i.v40a.file_sha256(OUTPUT),
        "manifest": str(MANIFEST),
        "manifest_file_sha256": v43i.v40a.file_sha256(MANIFEST),
        "manifest_content_sha256": value["content_sha256_before_self_field"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
