#!/usr/bin/env python3
"""Deterministically promote reviewed append/replacement curation ledgers."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            fact_id = row.get("fact_id")
            if not isinstance(fact_id, str) or not fact_id:
                raise ValueError(
                    f"{path} line {line_number} has no fact_id")
            rows.append(row)
    return rows


DECISION_IDENTITY_FIELDS = (
    "action", "expected_question", "expected_answer", "question", "answer",
    "reason_code",
)


def decision_identity(row: dict) -> tuple:
    return tuple(row.get(field) for field in DECISION_IDENTITY_FIELDS)


def unique_rows(paths: list[Path], kind: str) -> tuple[list[dict], list[str]]:
    rows = []
    seen = {}
    confirmations = []
    for path in paths:
        for row in read_jsonl(path):
            fact_id = row["fact_id"]
            if fact_id in seen:
                if decision_identity(row) != decision_identity(seen[fact_id]):
                    raise ValueError(
                        f"conflicting {kind} fact_id {fact_id} across ledgers")
                confirmations.append(fact_id)
                continue
            seen[fact_id] = row
            rows.append(row)
    return rows, confirmations


def promote(base_rows: list[dict], append_rows: list[dict],
            replacement_rows: list[dict]) -> list[dict]:
    positions = {}
    for index, row in enumerate(base_rows):
        fact_id = row["fact_id"]
        if fact_id in positions:
            raise ValueError(f"duplicate base fact_id {fact_id}")
        positions[fact_id] = index

    promoted = list(base_rows)
    for replacement in replacement_rows:
        fact_id = replacement["fact_id"]
        if fact_id not in positions:
            raise ValueError(
                f"replacement fact_id {fact_id} is absent from base")
        previous = promoted[positions[fact_id]]
        for field in ("expected_question", "expected_answer"):
            if replacement.get(field) != previous.get(field):
                raise ValueError(
                    f"replacement fact_id {fact_id} changes {field}")
        promoted[positions[fact_id]] = replacement

    known = set(positions)
    for row in append_rows:
        fact_id = row["fact_id"]
        if fact_id in known:
            raise ValueError(
                f"append fact_id {fact_id} already exists, use replacement")
        known.add(fact_id)
        promoted.append(row)
    return promoted


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as destination:
        for row in rows:
            destination.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--append", type=Path, nargs="*", default=[])
    parser.add_argument("--replacements", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    base_rows = read_jsonl(args.base)
    base_info = {
        "path": str(args.base),
        "rows": len(base_rows),
        "sha256": file_sha256(args.base),
    }
    append_rows, append_confirmations = unique_rows(args.append, "append")
    replacement_rows, replacement_confirmations = unique_rows(
        args.replacements, "replacement")
    promoted = promote(base_rows, append_rows, replacement_rows)
    write_jsonl(args.output, promoted)
    action_counts = collections.Counter(row.get("action") for row in promoted)
    report = {
        "schema": "curation-ledger-promotion-v1",
        "base": base_info,
        "append_ledgers": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in args.append
        ],
        "replacement_ledgers": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in args.replacements
        ],
        "appended_rows": len(append_rows),
        "replacement_rows": len(replacement_rows),
        "duplicate_confirmations": {
            "append": append_confirmations,
            "replacement": replacement_confirmations,
        },
        "output": {"path": str(args.output), "rows": len(promoted),
                   "sha256": file_sha256(args.output)},
        "output_by_action": dict(sorted(action_counts.items())),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
