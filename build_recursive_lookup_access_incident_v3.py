#!/usr/bin/env python3
"""Seal the content-free receipt for the recursive filename lookup incident.

All scope is represented by fixed namespace/path identities.  This builder
never enumerates, resolves, stats, hashes, or opens either implicated prefix.
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
    "recursive_lookup_access_20260717_v3.json"
)
SCHEMA = "specialist-recursive-filename-lookup-access-incident-v3"
STATUS = "confirmed_nonzero_prefix_wide_tool_access_evaluation_roles_quarantined"
ACCESSED_RELATIVE_PREFIXES = (
    "data/manual_reviews",
    "experiments/sft_controls",
)
KNOWN_MATCHING_RELATIVE_PATHS = (
    "experiments/sft_controls/v37a_shadow_folds_v412/fold_0_train.jsonl",
    "experiments/sft_controls/v37a_shadow_folds_v412/fold_2_train.jsonl",
)
SUPERSEDED_V2_FILE_SHA256 = (
    "3e01a95138356224e006c2661fec8dd4675b6fdb13d7ce43c2b1bcfbab656fb3"
)
SUPERSEDED_V2_CONTENT_SHA256 = (
    "4121c90c79cb2aacc8a927174e9fbbab6b38f983176c525ad0517b548edc0391"
)


def canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _identity(schema: str, value: str) -> str:
    return canonical_sha256({"schema": schema, "value": value})


def accessed_prefix_identities() -> list[str]:
    return sorted(
        _identity("repository-relative-path-prefix-v1", value)
        for value in ACCESSED_RELATIVE_PREFIXES
    )


def build_receipt() -> dict:
    receipt = {
        "schema": SCHEMA,
        "status": STATUS,
        "recorded_at_utc": "2026-07-17T00:00:00+00:00",
        "classification": {
            "irreversible_nonzero_tool_access": True,
            "access_scope": (
                "every_rg_readable_file_recursively_reachable_at_event_time_"
                "under_each_bound_prefix"
            ),
            "scope_is_not_limited_to_reported_filename_matches": True,
            "evaluation_shadow_dev_or_holdout_roles_within_scope_quarantined": True,
            "ordinary_training_role_access_does_not_convert_to_evaluation_evidence": True,
        },
        "event": {
            "tool_mode": "recursive_fixed_string_filename_only_lookup",
            "training_fact_identifiers_queried": 8,
            "recursive_scan_passes": 8,
            "file_open_count": "unknown_nonzero",
            "only_matching_filenames_printed": True,
            "raw_row_content_answers_urls_or_metrics_printed_or_inspected": False,
            "model_execution_or_training_occurred": False,
            "worker_stopped_after_detection": True,
        },
        "scope": {
            "accessed_prefix_count": 2,
            "accessed_prefix_identity_sha256": accessed_prefix_identities(),
            "individual_file_inventory": "not_enumerated_reopen_prohibited",
            "individual_file_hashes": "unknown_not_computed",
            "reported_matching_filename_count_at_least": 2,
            "reported_matching_path_identity_set_sha256": canonical_sha256(
                sorted(
                    _identity("repository-relative-path-v1", value)
                    for value in KNOWN_MATCHING_RELATIVE_PATHS
                )
            ),
        },
        "v2_impact": {
            "affected_revision_original_path": (
                "experiments/eggroll_es_hpo/preregistrations/"
                "recipe_evaluation_compute_contract_v2.json"
            ),
            "affected_revision_file_sha256": SUPERSEDED_V2_FILE_SHA256,
            "affected_revision_content_sha256": SUPERSEDED_V2_CONTENT_SHA256,
            "affected_dev_namespace_was_within_accessed_prefix": True,
            "affected_revision_status": "superseded_immutable_nonpromotable",
            "v2_terminal_source_namespace_within_accessed_prefix": False,
            "v2_terminal_source_opened_by_event": False,
            "v2_terminal_selection_remains_reserved": True,
            "fresh_dev_outside_accessed_prefixes_required": True,
        },
        "postincident_response": {
            "configured_path_metadata_resolution_may_have_occurred": True,
            "implicated_file_content_opened_or_hashed": False,
            "resolve_at_module_import_must_be_removed_before_reseal": True,
        },
        "quarantine": {
            "prefix_contents_may_not_supply_dev_shadow_holdout_or_evaluation": True,
            "prefix_file_inventory_reconstruction_prohibited": True,
            "source_reopen_stat_or_hash_for_incident_investigation_prohibited": True,
            "affected_v2_revision_may_not_be_promoted_or_reinterpreted": True,
        },
        "future_worker_policy": {
            "explicit_file_allowlist_required_before_lookup": True,
            "recursive_repository_or_dataset_prefix_scans_prohibited": True,
            "allowlist_may_not_include_dev_shadow_holdout_ood_probe_or_terminal": True,
            "filename_only_output_does_not_make_recursive_access_zero": True,
        },
    }
    receipt["content_sha256_before_self_field"] = canonical_sha256(receipt)
    return receipt


def validate_receipt(receipt: dict) -> None:
    compact = {
        key: value for key, value in receipt.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("status") != STATUS
        or receipt.get("content_sha256_before_self_field")
        != canonical_sha256(compact)
        or receipt.get("classification", {}).get(
            "irreversible_nonzero_tool_access"
        ) is not True
        or receipt.get("classification", {}).get(
            "scope_is_not_limited_to_reported_filename_matches"
        ) is not True
        or receipt.get("event", {}).get("recursive_scan_passes") != 8
        or receipt.get("event", {}).get("file_open_count") != "unknown_nonzero"
        or receipt.get("event", {}).get(
            "raw_row_content_answers_urls_or_metrics_printed_or_inspected"
        ) is not False
        or receipt.get("scope", {}).get("accessed_prefix_count") != 2
        or receipt.get("scope", {}).get("accessed_prefix_identity_sha256")
        != accessed_prefix_identities()
        or receipt.get("scope", {}).get("individual_file_inventory")
        != "not_enumerated_reopen_prohibited"
        or receipt.get("v2_impact", {}).get("affected_revision_file_sha256")
        != SUPERSEDED_V2_FILE_SHA256
        or receipt.get("v2_impact", {}).get("affected_revision_content_sha256")
        != SUPERSEDED_V2_CONTENT_SHA256
        or receipt.get("v2_impact", {}).get("affected_revision_status")
        != "superseded_immutable_nonpromotable"
        or receipt.get("v2_impact", {}).get("v2_terminal_source_opened_by_event")
        is not False
        or receipt.get("future_worker_policy", {}).get(
            "explicit_file_allowlist_required_before_lookup"
        ) is not True
    ):
        raise RuntimeError("invalid recursive lookup access incident receipt")


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
        "accessed_prefix_identities": 2,
        "content_sha256": receipt["content_sha256_before_self_field"],
        "file_inventory_reopened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
