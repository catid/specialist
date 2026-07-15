#!/usr/bin/env python3
"""Seal the exact train-only v331 candidate recovered from its source commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR_V331 = (
    ROOT / "experiments/eggroll_es_hpo/dataset_candidates/v331"
).resolve()
OUTPUT_PATH_V331 = OUTPUT_DIR_V331 / "train_qa_context_merit_v331.jsonl"
MANIFEST_PATH_V331 = OUTPUT_DIR_V331 / "manifest.json"

V331_SOURCE_COMMIT = "9d31e3407dd96f80011bdb2202e8aa9689b1d193"
V331_ROWS = 527
V331_SHA256 = "29889e18fd7a8bf313021ec4e8ad18adcc1752529fba247fdbde006513f7866b"
SOURCE_ARTIFACT_SHA256 = {
    "data/manual_reviews/context_merit_audit_v331/"
    "build_context_merit_audit_v331.py": (
        "a679766c0b017504c937371e60162c9d44719a544d470c7f7d9b36521a865434"
    ),
    "data/manual_reviews/context_merit_audit_v331/"
    "context_merit_audit_v331.jsonl": (
        "b468570c60c1c34814ae99321bb9ba5f3ff77fa2060fabd00f1acb258122a61b"
    ),
    "data/manual_reviews/context_merit_audit_v331/"
    "pending_curation_context_merit_v331.jsonl": (
        "120d2515286df3035cfd6d0119db4d6fea70f1d3768f0e85e1e4a62a2dcb94f8"
    ),
    "data/manual_reviews/context_merit_audit_v331/"
    "report_context_merit_v331.json": (
        "aa208207b0f03ad20d22d7d30ab151db99ba61f6844b04a7f2498369edf9a8cb"
    ),
    "data/manual_reviews/context_merit_audit_v331/"
    "test_context_merit_audit_v331.py": (
        "25867815e05cb967935c3760b6d7ead45d72802dd3d9d9540b527ba46ff91f8b"
    ),
}
SOURCE_REPORT_PATH = (
    "data/manual_reviews/context_merit_audit_v331/"
    "report_context_merit_v331.json"
)


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob(relative: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{V331_SOURCE_COMMIT}:{relative}"], cwd=ROOT
    )


def verify_source_commit_v331() -> dict[str, str]:
    observed = {
        relative: hashlib.sha256(_git_blob(relative)).hexdigest()
        for relative in SOURCE_ARTIFACT_SHA256
    }
    if observed != SOURCE_ARTIFACT_SHA256:
        raise RuntimeError("v331 source-commit artifact identity changed")
    report = json.loads(_git_blob(SOURCE_REPORT_PATH))
    projection = report.get("isolated_build_projection", {})
    sealed_policy = report.get("sealed_evaluation_policy", {})
    if (
        report.get("schema") != "context-merit-audit-report-v331"
        or projection.get("output_rows") != V331_ROWS
        or projection.get("output_sha256") != V331_SHA256
        or projection.get("repeat_dataset_byte_identical") is not True
        or sealed_policy.get("manual_worker_opened_eval_or_heldout_content")
        is not False
        or sealed_policy.get("manual_worker_received_eval_or_heldout_content")
        is not False
    ):
        raise RuntimeError("v331 source-commit projection certificate changed")
    return observed


def validate_candidate_snapshot_v331() -> None:
    if (
        file_sha256(OUTPUT_PATH_V331) != V331_SHA256
        or sum(1 for line in OUTPUT_PATH_V331.open("rb") if line.strip())
        != V331_ROWS
    ):
        raise RuntimeError("v331 train candidate snapshot identity changed")


def build_manifest_v331() -> dict:
    source_inventory = verify_source_commit_v331()
    validate_candidate_snapshot_v331()
    value = {
        "schema": "eggroll-es-immutable-train-candidate-manifest-v331",
        "status": "sealed_train_only_snapshot_no_runtime_or_evaluation_authority",
        "candidate": {
            "path": str(OUTPUT_PATH_V331),
            "rows": V331_ROWS,
            "file_sha256": V331_SHA256,
        },
        "provenance": {
            "source_commit": V331_SOURCE_COMMIT,
            "retrieval": (
                "detached_source_commit_projection_replay_then_exact_output_"
                "sha256_capture"
            ),
            "source_artifact_file_sha256": source_inventory,
            "source_artifact_inventory_sha256": canonical_sha256(
                source_inventory
            ),
            "ongoing_working_tree_curation_used": False,
        },
        "firewall": {
            "source_projection_automated_collision_filter_read_sealed_content": True,
            "v21a_builder_opened_heldout_validation_ood_eval_or_benchmark_content": False,
            "candidate_row_content_in_manifest": False,
            "runtime_launch_authorized": False,
            "model_update_authorized": False,
            "checkpoint_write_authorized": False,
            "evaluation_authorized": False,
            "dataset_promotion_authorized": False,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return validate_manifest_v331(value)


def validate_manifest_v331(value: dict) -> dict:
    without_self = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    firewall = value.get("firewall", {})
    if (
        value.get("schema")
        != "eggroll-es-immutable-train-candidate-manifest-v331"
        or value.get("candidate") != {
            "path": str(OUTPUT_PATH_V331),
            "rows": V331_ROWS,
            "file_sha256": V331_SHA256,
        }
        or value.get("provenance", {}).get("source_commit")
        != V331_SOURCE_COMMIT
        or value.get("provenance", {}).get(
            "ongoing_working_tree_curation_used"
        ) is not False
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(without_self)
        or firewall.get(
            "v21a_builder_opened_heldout_validation_ood_eval_or_benchmark_content"
        ) is not False
        or any(firewall.get(key) is not False for key in (
            "runtime_launch_authorized", "model_update_authorized",
            "checkpoint_write_authorized", "evaluation_authorized",
            "dataset_promotion_authorized",
        ))
    ):
        raise RuntimeError("v331 immutable train candidate manifest changed")
    return value


def _exclusive_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("immutable v331 manifest already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(MANIFEST_PATH_V331))
    args = parser.parse_args(argv)
    if Path(args.output).resolve() != MANIFEST_PATH_V331:
        raise ValueError("v331 manifest output path changed")
    manifest = build_manifest_v331()
    _exclusive_write(MANIFEST_PATH_V331, manifest)
    result = {
        "schema": "eggroll-es-train-candidate-seal-v331",
        "candidate_file_sha256": V331_SHA256,
        "manifest_file_sha256": file_sha256(MANIFEST_PATH_V331),
        "manifest_content_sha256": manifest["content_sha256_before_self_field"],
        "source_commit": V331_SOURCE_COMMIT,
        "gpu_launched": False,
        "runtime_launch_authorized": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
