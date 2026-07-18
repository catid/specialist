#!/usr/bin/env python3
"""Validate immutable metadata-only unexpected-source-read incident records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


LOG_PATH = Path(__file__).with_name("unexpected_source_read_incidents_v1.jsonl")
SCHEMA = "unexpected-source-read-incident-v1"
SAFE_SELECTION_RULE = "exclude quarantine in the file-selection command itself"
EXPECTED_FALSE_EXPOSURES = {
    "protected_text_read_or_printed",
    "development_text_read_or_printed",
    "final_text_read_or_printed",
    "holdout_text_read_or_printed",
    "ood_text_read_or_printed",
    "terminal_text_read_or_printed",
    "url_read_or_printed",
    "prompt_read_or_printed",
    "answer_read_or_printed",
    "completion_read_or_printed",
    "per_item_evaluation_metric_read_or_printed",
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def validate_record(record: dict) -> None:
    unsigned = dict(record)
    observed_address = unsigned.pop("content_sha256_before_self_field", None)
    if observed_address != canonical_sha256(unsigned):
        raise RuntimeError("incident content address changed")
    exposure = record.get("exposure")
    if (
        record.get("schema") != SCHEMA
        or record.get("status")
        != "recorded_irreversible_metadata_only_quarantine_boundary_read"
        or record.get("observed_return")
        != (
            "only guided_schema_sha256 metadata from quarantined training "
            "semantic-judge smoke reports"
        )
        or record.get("protected_source_touched") is not False
        or record.get("quarantine_boundary_read_irreversible") is not True
        or record.get("affected_quarantined_files_preserved") is not True
        or record.get("safe_future_selection_rule") != SAFE_SELECTION_RULE
        or not isinstance(exposure, dict)
        or set(exposure) != EXPECTED_FALSE_EXPOSURES
        or any(value is not False for value in exposure.values())
    ):
        raise RuntimeError("incident fail-closed metadata changed")
    paths = record.get("affected_quarantined_files")
    if (
        not isinstance(paths, list)
        or not paths
        or any("/quarantine/" not in value for value in paths)
        or len(paths) != len(set(paths))
    ):
        raise RuntimeError("incident quarantine preservation scope changed")


def main() -> int:
    if LOG_PATH.is_symlink() or not LOG_PATH.is_file():
        raise RuntimeError("unexpected-source-read incident log is missing")
    records = [
        json.loads(line)
        for line in LOG_PATH.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError("unexpected-source-read incident log is empty")
    for record in records:
        validate_record(record)
    print(json.dumps({"status": "valid", "incidents": len(records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
