#!/usr/bin/env python3
"""Seal the content-free receipt for the legacy QA collision-source access.

This builder deliberately works from fixed path strings and reported aggregate
facts.  It never resolves, stats, hashes, or opens either quarantined source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/incidents/"
    "legacy_eval_collision_access_20260717_v2.json"
)
SCHEMA = "specialist-legacy-evaluation-collision-access-incident-v2"
STATUS = "confirmed_nonzero_legacy_evaluation_family_access_paths_quarantined"
QUARANTINED_RELATIVE_PATHS = (
    "data/eval_qa.jsonl",
    "data/eval_qa_v2.jsonl",
)
FUTURE_EXCLUDED_ROLES = ("cpt", "dev", "qa", "sft", "train")
PRE_HARDENING_BUILDER_FILE_SHA256 = (
    "b12cb79907a50d05bde9269f3a98e43514da58e05fa1b85cb58bcfc692549384"
)
PRE_HARDENING_BUILDER_GIT_BLOB = "031f92c416f7afd50e252ea6e4d3e751b2b599d4"
PRE_HARDENING_BUILDER_COMMIT = "e49cded006506cf788169780d11f689aa024437f"


def canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def path_identity(relative_path: str) -> str:
    return canonical_sha256({
        "schema": "repository-relative-path-v1",
        "value": relative_path,
    })


def build_receipt() -> dict:
    identities = sorted(path_identity(path) for path in QUARANTINED_RELATIVE_PATHS)
    receipt = {
        "schema": SCHEMA,
        "status": STATUS,
        "recorded_at_utc": "2026-07-17T00:00:00+00:00",
        "classification": {
            "family": "legacy_evaluation_collision_sources",
            "irreversible_nonzero_access": True,
            "distinct_from_quarantined_v1_eval_v3": True,
            "distinct_from_v2_terminal_boundary": True,
        },
        "event": {
            "invocations": 1,
            "aggregate_eval_fact_count_reported": 505,
            "access_purpose": "collision_identity_construction_only",
            "raw_source_content_viewed_or_printed": False,
            "per_item_content_or_identifiers_viewed_or_printed": False,
            "model_execution_or_training_occurred": False,
        },
        "scope": {
            "touched_source_count": len(identities),
            "touched_source_path_identity_sha256": identities,
            "source_file_hashes": "unknown_not_computed",
            "holdout_source_touched": False,
            "ood_source_touched": False,
            "shadow_source_touched": False,
            "probe_source_touched": False,
        },
        "quarantine": {
            "entire_sources_quarantined_by_path_identity": True,
            "source_reopen_stat_or_hash_prohibited": True,
            "implicit_or_default_resolution_prohibited": True,
            "future_adaptation_roles_excluded": list(FUTURE_EXCLUDED_ROLES),
            "path_identity_only_no_byte_or_semantic_claim": True,
        },
        "pre_hardening_callsite": {
            "builder_path": "build_curated_qa.py",
            "builder_file_sha256": PRE_HARDENING_BUILDER_FILE_SHA256,
            "builder_git_blob": PRE_HARDENING_BUILDER_GIT_BLOB,
            "builder_commit": PRE_HARDENING_BUILDER_COMMIT,
            "implicit_eval_default_was_present": True,
        },
        "required_boundary": {
            "explicit_eval_choice_required": True,
            "synthetic_empty_eval_mode_is_content_free": True,
            "quarantined_path_resolution_rejected_before_eval_loader": True,
        },
    }
    receipt["content_sha256_before_self_field"] = canonical_sha256(receipt)
    return receipt


def validate_receipt(receipt: dict) -> None:
    compact = {
        key: value for key, value in receipt.items()
        if key != "content_sha256_before_self_field"
    }
    identities = sorted(path_identity(path) for path in QUARANTINED_RELATIVE_PATHS)
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("status") != STATUS
        or receipt.get("content_sha256_before_self_field")
        != canonical_sha256(compact)
        or receipt.get("classification", {}).get("irreversible_nonzero_access")
        is not True
        or receipt.get("event", {}).get("aggregate_eval_fact_count_reported")
        != 505
        or receipt.get("event", {}).get(
            "raw_source_content_viewed_or_printed"
        ) is not False
        or receipt.get("event", {}).get(
            "per_item_content_or_identifiers_viewed_or_printed"
        ) is not False
        or receipt.get("event", {}).get("model_execution_or_training_occurred")
        is not False
        or receipt.get("scope", {}).get("touched_source_count") != 2
        or receipt.get("scope", {}).get(
            "touched_source_path_identity_sha256"
        ) != identities
        or receipt.get("scope", {}).get("source_file_hashes")
        != "unknown_not_computed"
        or receipt.get("quarantine", {}).get(
            "source_reopen_stat_or_hash_prohibited"
        ) is not True
        or receipt.get("quarantine", {}).get("future_adaptation_roles_excluded")
        != list(FUTURE_EXCLUDED_ROLES)
        or receipt.get("required_boundary", {}).get(
            "quarantined_path_resolution_rejected_before_eval_loader"
        ) is not True
    ):
        raise RuntimeError("invalid legacy evaluation collision incident receipt")


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("incident receipt is not a JSON object")
    return value


def _write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    receipt = build_receipt()
    validate_receipt(receipt)
    if args.check:
        if _read_json(args.output) != receipt:
            raise RuntimeError("persisted incident receipt differs from rebuild")
    else:
        _write_exclusive(args.output, receipt)
    print(json.dumps({
        "content_sha256": receipt["content_sha256_before_self_field"],
        "quarantined_path_identities": 2,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
