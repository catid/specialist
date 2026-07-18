#!/usr/bin/env python3
"""Prepare and seal independent manual semantic review of seed QA rows.

This tool never infers semantic correctness from lineage or an automated
score.  It first creates a content-addressed, metadata-only assignment
manifest.  Human reviewers then inspect the question, answer, and evidence in
the pinned source JSONL and author one decision for every assigned line.  Only
fully supported, useful, self-contained rows may receive ``pass``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
SOURCE = (
    ROOT
    / "data/training_inventory/high_information_domain_corpus_v1/seed_qa_train.jsonl"
)
OUTPUT_DIRECTORY = (
    ROOT
    / "data/training_inventory/high_information_domain_corpus_v1/"
    "seed_qa_semantic_review_v1"
)
ASSIGNMENTS = OUTPUT_DIRECTORY / "assignments.json"
DECISIONS_DIRECTORY = OUTPUT_DIRECTORY / "decisions"
DECISION_BUNDLE = OUTPUT_DIRECTORY / "decisions.jsonl"
AUTHORITY = (
    ROOT
    / "data/training_inventory/high_information_domain_corpus_v1/"
    "seed_qa_semantic_authority_v1.json"
)

SOURCE_SHA256 = "8775b94f57d73d1c0a6d86cbeae4c59a299b09ae3a80b50267fe4f7da1ec9b9a"
SOURCE_ROWS = 357
SOURCE_ASSISTANT_TOKENS = 9_153
CHUNK_SIZE = 30
ASSIGNMENT_SCHEMA = "seed-qa-semantic-review-assignments-v1"
DECISION_SCHEMA = "seed-qa-manual-semantic-decision-v1"
AUTHORITY_SCHEMA = "seed-qa-semantic-authority-v1"
REVIEW_METHOD = "manual_line_by_line_question_answer_evidence_v1"
REVIEWER_PREFIX = "codex-manual-review/"
HEX64 = set("0123456789abcdef")

DECISION_KEYS = {
    "schema",
    "source_line_number",
    "record_id",
    "source_record_sha256",
    "reviewer_id",
    "review_method",
    "reviewed_question_answer_and_evidence",
    "reviewer_independent_of_source_generator",
    "decision",
    "semantic_correctness_verified",
    "evidence_entails_entire_answer",
    "question_is_user_useful",
    "question_is_self_contained",
    "answer_is_direct_and_well_formed",
    "safety_qualification_is_adequate",
    "reason_code",
    "notes",
    "decision_content_sha256",
}
PASS_FLAGS = (
    "semantic_correctness_verified",
    "evidence_entails_entire_answer",
    "question_is_user_useful",
    "question_is_self_contained",
    "answer_is_direct_and_well_formed",
    "safety_qualification_is_adequate",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_hex64(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= HEX64
    )


def _self_address(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result[key] = canonical_sha256(result)
    return result


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _secure_regular_file(path: Path, purpose: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    _require(lexical.is_relative_to(ROOT), f"{purpose} escapes repository")
    current = Path(lexical.anchor)
    metadata = None
    for component in lexical.parts[1:]:
        current /= component
        metadata = current.lstat()
        _require(not stat.S_ISLNK(metadata.st_mode), f"{purpose} is symlinked")
    _require(
        metadata is not None and stat.S_ISREG(metadata.st_mode),
        f"{purpose} is not a regular file",
    )
    _require(metadata.st_nlink == 1, f"{purpose} is hard-linked")
    return lexical


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


def _json_payload(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _jsonl_payload(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(canonical_bytes(row) + b"\n" for row in rows)


def load_source(path: Path = SOURCE) -> list[dict[str, Any]]:
    path = _secure_regular_file(path, "seed QA source")
    _require(file_sha256(path) == SOURCE_SHA256, "seed QA source bytes changed")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, 1):
            _require(line.endswith("\n") and line.strip(), "seed QA JSONL changed")
            row = json.loads(line)
            _require(
                isinstance(row, dict)
                and row.get("schema") == "high-information-seed-qa-v1"
                and isinstance(row.get("record_id"), str)
                and row["record_id"]
                and isinstance(row.get("question"), str)
                and row["question"].strip()
                and isinstance(row.get("answer"), str)
                and row["answer"].strip()
                and isinstance(row.get("evidence"), str)
                and row["evidence"].strip()
                and _is_hex64(row.get("evidence_sha256")),
                f"seed QA row {line_number} contract changed",
            )
            _require(
                hashlib.sha256(row["evidence"].encode("utf-8")).hexdigest()
                == row["evidence_sha256"],
                f"seed QA row {line_number} evidence receipt changed",
            )
            rows.append(row)
    _require(len(rows) == SOURCE_ROWS, "seed QA row count changed")
    _require(
        len({row["record_id"] for row in rows}) == len(rows),
        "seed QA record IDs are not unique",
    )
    _require(
        sum(row.get("assistant_qwen36_token_count", -1) for row in rows)
        == SOURCE_ASSISTANT_TOKENS,
        "seed QA assistant-token accounting changed",
    )
    return rows


def build_assignments(rows: list[dict[str, Any]]) -> dict[str, Any]:
    chunks = []
    for start in range(0, len(rows), CHUNK_SIZE):
        selected = rows[start : start + CHUNK_SIZE]
        number = len(chunks)
        chunks.append(
            {
                "chunk_id": f"chunk-{number:03d}",
                "source_line_start": start + 1,
                "source_line_end": start + len(selected),
                "rows": len(selected),
                "assistant_qwen36_tokens": sum(
                    row["assistant_qwen36_token_count"] for row in selected
                ),
                "decision_path": _relative(
                    DECISIONS_DIRECTORY / f"chunk-{number:03d}.jsonl"
                ),
                "record_identity_commitment_sha256": canonical_sha256(
                    [
                        {
                            "source_line_number": start + offset + 1,
                            "record_id": row["record_id"],
                            "source_record_sha256": canonical_sha256(row),
                        }
                        for offset, row in enumerate(selected)
                    ]
                ),
            }
        )
    body = {
        "schema": ASSIGNMENT_SCHEMA,
        "status": "awaiting_manual_line_by_line_review",
        "source": {
            "path": _relative(SOURCE),
            "file_sha256": SOURCE_SHA256,
            "rows": len(rows),
            "assistant_qwen36_tokens": SOURCE_ASSISTANT_TOKENS,
        },
        "review_contract": {
            "method": REVIEW_METHOD,
            "source_content_must_be_inspected": ["question", "answer", "evidence"],
            "lineage_or_automated_scores_are_not_semantic_authority": True,
            "pass_requires_every_quality_flag_true": list(PASS_FLAGS),
            "unsupported_or_low_utility_rows_must_be_excluded": True,
            "reviewer_must_be_independent_of_source_generator": True,
            "protected_evaluation_content_opened": False,
        },
        "chunks": chunks,
    }
    return _self_address(body, "content_sha256_before_self_field")


def _decision_body(decision: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in decision.items() if key != "decision_content_sha256"}


def validate_decision(
    row: dict[str, Any], source_line_number: int, decision: Any
) -> dict[str, Any]:
    _require(isinstance(decision, dict), "manual decision is not an object")
    _require(set(decision) == DECISION_KEYS, "manual decision fields changed")
    _require(decision.get("schema") == DECISION_SCHEMA, "manual decision schema changed")
    _require(
        decision.get("source_line_number") == source_line_number
        and decision.get("record_id") == row["record_id"]
        and decision.get("source_record_sha256") == canonical_sha256(row),
        "manual decision source identity changed",
    )
    _require(
        isinstance(decision.get("reviewer_id"), str)
        and decision["reviewer_id"].startswith(REVIEWER_PREFIX)
        and len(decision["reviewer_id"]) > len(REVIEWER_PREFIX),
        "manual reviewer identity is invalid",
    )
    _require(decision.get("review_method") == REVIEW_METHOD, "manual review method changed")
    _require(
        decision.get("reviewed_question_answer_and_evidence") is True
        and decision.get("reviewer_independent_of_source_generator") is True,
        "manual content inspection or independence is absent",
    )
    _require(
        decision.get("decision") in {"pass", "exclude"},
        "manual semantic disposition is invalid",
    )
    _require(
        all(type(decision.get(flag)) is bool for flag in PASS_FLAGS),
        "manual quality flags must be exact booleans",
    )
    if decision["decision"] == "pass":
        _require(
            all(decision[flag] is True for flag in PASS_FLAGS),
            "a passing decision has an unproven quality flag",
        )
        _require(
            decision.get("reason_code") == "fully_supported_useful_seed_qa",
            "passing reason code changed",
        )
    else:
        _require(
            any(decision[flag] is False for flag in PASS_FLAGS),
            "an exclusion must identify a failed quality flag",
        )
        _require(
            decision.get("reason_code")
            in {
                "answer_not_fully_supported",
                "answer_factually_incorrect",
                "question_not_user_useful",
                "question_not_self_contained",
                "question_contains_unsupported_or_false_premise",
                "answer_not_direct_or_well_formed",
                "safety_qualification_inadequate",
                "multiple_quality_failures",
            },
            "exclusion reason code changed",
        )
    _require(
        isinstance(decision.get("notes"), str) and decision["notes"].strip(),
        "manual review notes are absent",
    )
    _require(
        decision.get("decision_content_sha256")
        == canonical_sha256(_decision_body(decision)),
        "manual decision content receipt changed",
    )
    return decision


def load_decisions(
    rows: list[dict[str, Any]], assignments: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decisions: list[dict[str, Any]] = []
    file_receipts = []
    for chunk in assignments["chunks"]:
        path = ROOT / chunk["decision_path"]
        path = _secure_regular_file(path, f"manual decision {chunk['chunk_id']}")
        chunk_rows = []
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for line in handle:
                _require(line.endswith("\n") and line.strip(), "manual decision JSONL changed")
                chunk_rows.append(json.loads(line))
        _require(len(chunk_rows) == chunk["rows"], "manual decision chunk row count changed")
        start = chunk["source_line_start"] - 1
        for offset, decision in enumerate(chunk_rows):
            source_index = start + offset
            decisions.append(validate_decision(rows[source_index], source_index + 1, decision))
        file_receipts.append(
            {
                "chunk_id": chunk["chunk_id"],
                "path": chunk["decision_path"],
                "file_sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "rows": len(chunk_rows),
            }
        )
    _require(len(decisions) == len(rows), "manual semantic decision coverage changed")
    _require(
        len({item["record_id"] for item in decisions}) == len(rows),
        "manual decisions contain duplicate record IDs",
    )
    return decisions, file_receipts


def build_authority(
    rows: list[dict[str, Any]],
    assignments: dict[str, Any],
    decisions: list[dict[str, Any]],
    decision_file_receipts: list[dict[str, Any]],
) -> tuple[bytes, dict[str, Any]]:
    admitted = [item for item in decisions if item["decision"] == "pass"]
    excluded = [item for item in decisions if item["decision"] == "exclude"]
    token_by_record = {
        row["record_id"]: row["assistant_qwen36_token_count"] for row in rows
    }
    admitted_tokens = sum(token_by_record[item["record_id"]] for item in admitted)
    excluded_tokens = sum(token_by_record[item["record_id"]] for item in excluded)
    bundle_payload = _jsonl_payload(decisions)
    body = {
        "schema": AUTHORITY_SCHEMA,
        "status": "sealed_passed",
        "semantic_correctness_verified": True,
        "eligible_for_training": bool(admitted),
        "source_rows": len(rows),
        "reviewed_rows": len(decisions),
        "training_rows_admitted": len(admitted),
        "excluded_rows": len(excluded),
        "source_assistant_qwen36_tokens": SOURCE_ASSISTANT_TOKENS,
        "assistant_qwen36_tokens": admitted_tokens,
        "excluded_assistant_qwen36_tokens": excluded_tokens,
        "replacement_generated_assistant_tokens_required": excluded_tokens,
        "source_dataset": {
            "path": _relative(SOURCE),
            "file_sha256": SOURCE_SHA256,
        },
        "assignments": {
            "path": _relative(ASSIGNMENTS),
            "content_sha256": assignments["content_sha256_before_self_field"],
        },
        "decision_bundle": {
            "path": _relative(DECISION_BUNDLE),
            "file_sha256": hashlib.sha256(bundle_payload).hexdigest(),
            "bytes": len(bundle_payload),
            "rows": len(decisions),
        },
        "decision_files": decision_file_receipts,
        "admitted_record_identity_commitment_sha256": canonical_sha256(
            [
                {
                    "record_id": item["record_id"],
                    "source_record_sha256": item["source_record_sha256"],
                    "decision_content_sha256": item["decision_content_sha256"],
                }
                for item in admitted
            ]
        ),
        "exclusion_ledger": [
            {
                "record_id": item["record_id"],
                "source_record_sha256": item["source_record_sha256"],
                "decision_content_sha256": item["decision_content_sha256"],
                "reason_code": item["reason_code"],
                "assistant_qwen36_tokens": token_by_record[item["record_id"]],
            }
            for item in excluded
        ],
        "review_contract": {
            "method": REVIEW_METHOD,
            "all_question_answer_evidence_triplets_manually_inspected": True,
            "lineage_treated_as_semantic_authority": False,
            "automated_score_treated_as_semantic_authority": False,
            "unresolved_rows": 0,
            "protected_evaluation_content_opened": False,
        },
    }
    _require(admitted_tokens + excluded_tokens == SOURCE_ASSISTANT_TOKENS, "authority token accounting changed")
    return bundle_payload, _self_address(body, "content_sha256_before_self_field")


def _load_assignments(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = build_assignments(rows)
    path = _secure_regular_file(ASSIGNMENTS, "manual review assignments")
    observed = json.loads(path.read_text(encoding="utf-8"))
    _require(observed == expected, "manual review assignments changed")
    return observed


def prepare(*, check: bool = False) -> dict[str, Any]:
    rows = load_source()
    assignments = build_assignments(rows)
    if check:
        observed = json.loads(_secure_regular_file(ASSIGNMENTS, "assignments").read_text(encoding="utf-8"))
        _require(observed == assignments, "assignment manifest changed")
    else:
        _atomic_write(ASSIGNMENTS, _json_payload(assignments))
    return assignments


def seal(*, check: bool = False) -> dict[str, Any]:
    rows = load_source()
    assignments = _load_assignments(rows)
    decisions, receipts = load_decisions(rows, assignments)
    bundle, authority = build_authority(rows, assignments, decisions, receipts)
    if check:
        _require(
            _secure_regular_file(DECISION_BUNDLE, "decision bundle").read_bytes()
            == bundle,
            "sealed decision bundle changed",
        )
        observed = json.loads(_secure_regular_file(AUTHORITY, "seed QA authority").read_text(encoding="utf-8"))
        _require(observed == authority, "seed QA semantic authority changed")
    else:
        _atomic_write(DECISION_BUNDLE, bundle)
        _atomic_write(AUTHORITY, _json_payload(authority))
    return authority


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--seal", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    value = prepare(check=args.check) if args.prepare else seal(check=args.check)
    rows = (
        value["reviewed_rows"]
        if "reviewed_rows" in value
        else value["source"]["rows"]
    )
    print(
        json.dumps(
            {
                "schema": value["schema"],
                "status": value["status"],
                "content_sha256": value["content_sha256_before_self_field"],
                "rows": rows,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
