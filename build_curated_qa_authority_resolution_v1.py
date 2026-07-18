#!/usr/bin/env python3
"""Seal retirement of legacy curated QA and the active V440 split authority.

The quarantined legacy evaluation collision sources are never resolved,
statted, hashed, counted, or opened here. This builder consumes only their
content-free incident receipt plus the safe source-split authority and its
materialized train/development projections.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parent
INCIDENT = (
    ROOT / "experiments/eggroll_es_hpo/incidents/"
    "legacy_eval_collision_access_20260717_v2.json"
).resolve()
SPLIT_AUTHORITY = (
    ROOT / "data/training_inventory/source_group_split_authority_v1.json"
).resolve()
OUTPUT = (
    ROOT / "data/training_inventory/curated_qa_authority_resolution_v1.json"
).resolve()
SCHEMA = "specialist-curated-qa-authority-resolution-v1"
EXPECTED_SPLIT_AUTHORITY_SHA256 = (
    "812111f7713f3a0d4fada42ab72e761be5b2175ab9301513e21845a2e03b8d15"
)
EXPECTED_INCIDENT_SHA256 = (
    "68b214cfad7326f15ad7f090130a14b60dab2e2ec478efbf32213e239d1097d4"
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_self_addressed(path: Path, expected: str) -> dict:
    _require(path.is_file() and not path.is_symlink(), f"unsafe receipt: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"receipt is not an object: {path}")
    unsigned = dict(value)
    declared = unsigned.pop("content_sha256_before_self_field", None)
    _require(declared == expected, f"unexpected content address: {path}")
    _require(canonical_sha256(unsigned) == declared, f"stale self address: {path}")
    return value


def _projection_receipt(authority: dict, partition: str, kind: str) -> dict:
    receipt = authority["materialized_train_development_projections"][partition][kind]
    path = (ROOT / receipt["path"]).resolve()
    _require(path.is_file() and not path.is_symlink(), f"unsafe projection: {path}")
    _require(file_sha256(path) == receipt["file_sha256"], f"stale projection: {path}")
    raw = path.read_bytes()
    _require(len(raw) == receipt["file_bytes"], f"projection size changed: {path}")
    _require(
        sum(1 for line in raw.splitlines() if line.strip()) == receipt["rows"],
        f"projection row count changed: {path}",
    )
    return {
        "partition": partition,
        "kind": kind,
        "path": receipt["path"],
        "file_sha256": receipt["file_sha256"],
        "file_bytes": receipt["file_bytes"],
        "rows": receipt["rows"],
        "qwen36_tokens": receipt["qwen36_tokens"],
    }


def build() -> dict:
    incident = _load_self_addressed(INCIDENT, EXPECTED_INCIDENT_SHA256)
    _require(
        incident["status"]
        == "confirmed_nonzero_legacy_evaluation_family_access_paths_quarantined",
        "legacy collision incident status changed",
    )
    quarantine = incident["quarantine"]
    _require(quarantine["source_reopen_stat_or_hash_prohibited"] is True,
             "legacy collision sources are not sealed no-reopen")
    _require(quarantine["entire_sources_quarantined_by_path_identity"] is True,
             "legacy collision sources are not fully quarantined")

    authority = _load_self_addressed(
        SPLIT_AUTHORITY, EXPECTED_SPLIT_AUTHORITY_SHA256
    )
    _require(authority["schema"] == "specialist-source-group-split-authority-v1",
             "source-split authority schema changed")
    invariants = authority["invariants"]
    for invariant in (
        "descendant_cross_split_overlap",
        "exact_content_cross_split_overlap",
        "near_duplicate_cross_split_overlap",
        "source_url_identity_cross_split_overlap",
    ):
        _require(invariants[invariant] is False, f"split overlap: {invariant}")
    _require(invariants["final_records_emitted"] is False,
             "final source records were emitted")
    _require(invariants["v440_final_semantic_records_emitted"] is False,
             "final V440 semantics were emitted")
    _require(invariants[
        "protected_holdout_ood_terminal_incident_or_manual_review_sources_opened"
    ] is False, "protected source was opened by split builder")

    assignments = authority["assignments"]
    v440_counts = {
        partition: assignments[partition]["v440_descendant_fact_count"]
        for partition in ("train", "development", "final")
    }
    _require(v440_counts == {"train": 382, "development": 74, "final": 60},
             "V440 split counts changed")
    _require(sum(v440_counts.values()) == 516, "V440 authority total changed")
    projections = [
        _projection_receipt(authority, partition, kind)
        for partition in ("train", "development")
        for kind in ("site_spans", "v440_qa")
    ]

    result = {
        "schema": SCHEMA,
        "status": "legacy_curated_qa_retired_active_v440_split_authority_verified",
        "authority": {
            "quarantined_legacy_evaluation_paths_resolved_statted_hashed_counted_or_opened": False,
            "protected_holdout_ood_terminal_or_manual_review_semantics_opened": False,
            "legacy_curated_qa_authorized_for_plan_training": False,
            "active_v440_train_projection_authorized_for_snapshot_construction": True,
            "training_launch_authorized": False,
        },
        "legacy_curated_qa": {
            "role": "retired_reproducibility_artifact_only",
            "production_rebuild_status": "intentionally_unavailable_without_new_independent_opaque_authorization",
            "reason": "the former collision sources are under irreversible no-reopen quarantine",
            "builder_fail_closed_interface_retained": "build_curated_qa.py",
            "incident_receipt": INCIDENT.relative_to(ROOT).as_posix(),
            "incident_content_sha256": EXPECTED_INCIDENT_SHA256,
        },
        "active_qa": {
            "authority_id": "qa-authority:v440-minus-url-index-logical-view",
            "source_split_authority": SPLIT_AUTHORITY.relative_to(ROOT).as_posix(),
            "source_split_content_sha256": EXPECTED_SPLIT_AUTHORITY_SHA256,
            "v440_rows_by_partition": v440_counts,
            "v440_rows_total": sum(v440_counts.values()),
            "final_records_emitted": False,
            "disjointness": {
                "descendant_fact": True,
                "exact_content": True,
                "near_duplicate": True,
                "source_group": True,
                "source_url_identity": True,
            },
            "materialized_safe_projections": projections,
            "deterministic_check_command": (
                ".venv/bin/python build_source_group_split_authority_v1.py --check"
            ),
        },
        "resolution": {
            "original_cli_documentation_mismatch_repaired": True,
            "synthetic_curated_builder_tests_required": True,
            "old_artifact_removed_from_active_training_authority": True,
            "opaque_legacy_boundary_not_forged_or_reconstructed": True,
            "current_training_qa_rebuild_is_deterministic": True,
        },
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    value = build()
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if args.check:
        _require(OUTPUT.is_file(), "curated-QA authority resolution is missing")
        _require(OUTPUT.read_bytes() == payload, "curated-QA authority resolution is stale")
    else:
        _atomic_write(OUTPUT, payload)
    print(json.dumps({
        "content_sha256": value["content_sha256_before_self_field"],
        "status": "checked" if args.check else "written",
        "v440_rows": value["active_qa"]["v440_rows_total"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
