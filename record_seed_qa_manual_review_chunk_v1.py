#!/usr/bin/env python3
"""Turn one manually authored seed-QA review annotation into sealed decisions.

The annotation is deliberately separate from the generated decision JSONL: a
reviewer must explicitly disposition every assigned record and leave a
record-specific note after reading its question, answer, and evidence.  This
tool validates that annotation against the pinned assignment and source, then
derives the verbose fail-closed decision contract consumed by the authority
builder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import build_seed_qa_semantic_authority_v1 as authority


ROOT = Path(__file__).resolve().parent
ANNOTATIONS_DIRECTORY = authority.OUTPUT_DIRECTORY / "annotations"
ANNOTATION_SCHEMA = "seed-qa-manual-review-annotation-v1"
ANNOTATION_KEYS = {
    "schema",
    "chunk_id",
    "reviewer_id",
    "review_method",
    "reviewed_question_answer_and_evidence",
    "reviewer_independent_of_source_generator",
    "records",
    "content_sha256_before_self_field",
}
RECORD_KEYS = {
    "source_line_number",
    "record_id",
    "decision",
    "failed_quality_flags",
    "reason_code",
    "notes",
}
EXCLUSION_REASONS = {
    "answer_not_fully_supported",
    "answer_factually_incorrect",
    "question_not_user_useful",
    "question_not_self_contained",
    "answer_not_direct_or_well_formed",
    "safety_qualification_inadequate",
    "multiple_quality_failures",
}


def _without_self_hash(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def load_annotation(path: Path) -> dict[str, Any]:
    secured = authority._secure_regular_file(path, "manual review annotation")
    value = json.loads(secured.read_text(encoding="utf-8", errors="strict"))
    authority._require(isinstance(value, dict), "annotation is not an object")
    authority._require(set(value) == ANNOTATION_KEYS, "annotation fields changed")
    authority._require(value.get("schema") == ANNOTATION_SCHEMA, "annotation schema changed")
    authority._require(
        value.get("content_sha256_before_self_field")
        == authority.canonical_sha256(_without_self_hash(value)),
        "annotation content receipt changed",
    )
    return value


def build_chunk_decisions(
    rows: list[dict[str, Any]],
    chunk: dict[str, Any],
    annotation: dict[str, Any],
) -> list[dict[str, Any]]:
    authority._require(
        annotation.get("chunk_id") == chunk["chunk_id"],
        "annotation chunk identity changed",
    )
    reviewer_id = annotation.get("reviewer_id")
    authority._require(
        isinstance(reviewer_id, str)
        and reviewer_id.startswith(authority.REVIEWER_PREFIX)
        and len(reviewer_id) > len(authority.REVIEWER_PREFIX),
        "annotation reviewer identity is invalid",
    )
    authority._require(
        annotation.get("review_method") == authority.REVIEW_METHOD,
        "annotation review method changed",
    )
    authority._require(
        annotation.get("reviewed_question_answer_and_evidence") is True
        and annotation.get("reviewer_independent_of_source_generator") is True,
        "annotation does not attest manual inspection and independence",
    )
    records = annotation.get("records")
    authority._require(
        isinstance(records, list) and len(records) == chunk["rows"],
        "annotation record coverage changed",
    )

    start = chunk["source_line_start"] - 1
    decisions: list[dict[str, Any]] = []
    for offset, item in enumerate(records):
        line_number = start + offset + 1
        source = rows[line_number - 1]
        authority._require(isinstance(item, dict), "annotation record is not an object")
        authority._require(set(item) == RECORD_KEYS, "annotation record fields changed")
        authority._require(
            item.get("source_line_number") == line_number
            and item.get("record_id") == source["record_id"],
            "annotation record order or identity changed",
        )
        decision = item.get("decision")
        authority._require(decision in {"pass", "exclude"}, "annotation decision is invalid")
        failed = item.get("failed_quality_flags")
        authority._require(
            isinstance(failed, list)
            and len(set(failed)) == len(failed)
            and all(flag in authority.PASS_FLAGS for flag in failed),
            "annotation failed-quality flags are invalid",
        )
        reason = item.get("reason_code")
        if decision == "pass":
            authority._require(not failed, "passing annotation identifies a failed flag")
            authority._require(
                reason == "fully_supported_useful_seed_qa",
                "passing annotation reason changed",
            )
        else:
            authority._require(bool(failed), "excluded annotation has no failed flag")
            authority._require(reason in EXCLUSION_REASONS, "exclusion reason changed")
        notes = item.get("notes")
        authority._require(
            isinstance(notes, str) and len(notes.strip()) >= 16,
            "annotation needs a substantive record-specific note",
        )

        body: dict[str, Any] = {
            "schema": authority.DECISION_SCHEMA,
            "source_line_number": line_number,
            "record_id": source["record_id"],
            "source_record_sha256": authority.canonical_sha256(source),
            "reviewer_id": reviewer_id,
            "review_method": authority.REVIEW_METHOD,
            "reviewed_question_answer_and_evidence": True,
            "reviewer_independent_of_source_generator": True,
            "decision": decision,
            **{
                flag: flag not in failed
                for flag in authority.PASS_FLAGS
            },
            "reason_code": reason,
            "notes": notes.strip(),
        }
        sealed = dict(body)
        sealed["decision_content_sha256"] = authority.canonical_sha256(body)
        decisions.append(authority.validate_decision(source, line_number, sealed))
    return decisions


def record_chunk(chunk_id: str, annotation_path: Path, *, check: bool = False) -> dict[str, Any]:
    rows = authority.load_source()
    assignments = authority._load_assignments(rows)
    matches = [chunk for chunk in assignments["chunks"] if chunk["chunk_id"] == chunk_id]
    authority._require(len(matches) == 1, "unknown or duplicate chunk identity")
    chunk = matches[0]
    expected_annotation = ANNOTATIONS_DIRECTORY / f"{chunk_id}.json"
    authority._require(
        Path(annotation_path).resolve() == expected_annotation.resolve(),
        "annotation path must be the canonical chunk path",
    )
    annotation = load_annotation(annotation_path)
    decisions = build_chunk_decisions(rows, chunk, annotation)
    payload = authority._jsonl_payload(decisions)
    output = ROOT / chunk["decision_path"]
    if check:
        authority._require(
            authority._secure_regular_file(output, "manual decision output").read_bytes()
            == payload,
            "manual decision output changed",
        )
    else:
        authority._atomic_write(output, payload)
    return {
        "schema": "seed-qa-manual-review-recording-result-v1",
        "status": "checked" if check else "recorded",
        "chunk_id": chunk_id,
        "rows": len(decisions),
        "passed": sum(item["decision"] == "pass" for item in decisions),
        "excluded": sum(item["decision"] == "exclude" for item in decisions),
        "output_path": output.relative_to(ROOT).as_posix(),
        "output_sha256": __import__("hashlib").sha256(payload).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-id", required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(record_chunk(args.chunk_id, args.annotation, check=args.check), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
