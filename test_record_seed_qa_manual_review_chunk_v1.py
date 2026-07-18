from __future__ import annotations

import copy

import pytest

import build_seed_qa_semantic_authority_v1 as authority
import record_seed_qa_manual_review_chunk_v1 as recorder


def _row(record_id: str) -> dict:
    return {
        "record_id": record_id,
        "question": "What is the safety fact?",
        "answer": "Monitor continuously.",
        "evidence": "Monitor continuously during the procedure.",
    }


def _annotation(records: list[dict]) -> dict:
    body = {
        "schema": recorder.ANNOTATION_SCHEMA,
        "chunk_id": "chunk-000",
        "reviewer_id": "codex-manual-review/test-reviewer",
        "review_method": authority.REVIEW_METHOD,
        "reviewed_question_answer_and_evidence": True,
        "reviewer_independent_of_source_generator": True,
        "records": records,
    }
    body["content_sha256_before_self_field"] = authority.canonical_sha256(body)
    return body


def _record(line: int, record_id: str, *, decision: str = "pass") -> dict:
    excluded = decision == "exclude"
    return {
        "source_line_number": line,
        "record_id": record_id,
        "decision": decision,
        "failed_quality_flags": ["question_is_user_useful"] if excluded else [],
        "reason_code": "question_not_user_useful" if excluded else "fully_supported_useful_seed_qa",
        "notes": "Evidence and answer were compared manually for this fixture.",
    }


def test_build_chunk_decisions_seals_pass_and_exclusion() -> None:
    rows = [_row("a"), _row("b")]
    chunk = {"chunk_id": "chunk-000", "source_line_start": 1, "rows": 2}
    decisions = recorder.build_chunk_decisions(
        rows,
        chunk,
        _annotation([_record(1, "a"), _record(2, "b", decision="exclude")]),
    )
    assert [item["decision"] for item in decisions] == ["pass", "exclude"]
    assert decisions[0]["question_is_user_useful"] is True
    assert decisions[1]["question_is_user_useful"] is False
    assert all(
        item["decision_content_sha256"]
        == authority.canonical_sha256(authority._decision_body(item))
        for item in decisions
    )


def test_annotation_must_cover_exact_source_order() -> None:
    rows = [_row("a"), _row("b")]
    chunk = {"chunk_id": "chunk-000", "source_line_start": 1, "rows": 2}
    with pytest.raises(RuntimeError, match="order or identity"):
        recorder.build_chunk_decisions(
            rows,
            chunk,
            _annotation([_record(1, "b"), _record(2, "a")]),
        )


def test_pass_cannot_hide_failed_quality_flag() -> None:
    rows = [_row("a")]
    chunk = {"chunk_id": "chunk-000", "source_line_start": 1, "rows": 1}
    annotation = _annotation([_record(1, "a")])
    annotation["records"][0]["failed_quality_flags"] = ["semantic_correctness_verified"]
    with pytest.raises(RuntimeError, match="passing annotation"):
        recorder.build_chunk_decisions(rows, chunk, annotation)


def test_note_must_be_substantive() -> None:
    rows = [_row("a")]
    chunk = {"chunk_id": "chunk-000", "source_line_start": 1, "rows": 1}
    annotation = _annotation([_record(1, "a")])
    annotation["records"][0]["notes"] = "okay"
    with pytest.raises(RuntimeError, match="substantive"):
        recorder.build_chunk_decisions(rows, chunk, annotation)


def test_unsupported_question_premise_can_be_explicitly_excluded() -> None:
    rows = [_row("a")]
    chunk = {"chunk_id": "chunk-000", "source_line_start": 1, "rows": 1}
    record = _record(1, "a", decision="exclude")
    record["failed_quality_flags"] = ["semantic_correctness_verified"]
    record["reason_code"] = "question_contains_unsupported_or_false_premise"
    decisions = recorder.build_chunk_decisions(rows, chunk, _annotation([record]))
    assert decisions[0]["decision"] == "exclude"
    assert decisions[0]["semantic_correctness_verified"] is False
