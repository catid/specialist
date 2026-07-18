from __future__ import annotations

import copy

import pytest

import build_seed_qa_semantic_authority_v1 as authority


def _row(number: int, tokens: int = 2) -> dict:
    evidence = f"Evidence supporting answer {number}."
    import hashlib

    return {
        "schema": "high-information-seed-qa-v1",
        "record_id": f"record-{number}",
        "question": f"Question {number}?",
        "answer": f"Answer {number}.",
        "evidence": evidence,
        "evidence_sha256": hashlib.sha256(evidence.encode()).hexdigest(),
        "assistant_qwen36_token_count": tokens,
    }


def _decision(row: dict, line: int, *, passed: bool = True) -> dict:
    value = {
        "schema": authority.DECISION_SCHEMA,
        "source_line_number": line,
        "record_id": row["record_id"],
        "source_record_sha256": authority.canonical_sha256(row),
        "reviewer_id": "codex-manual-review/synthetic-reviewer",
        "review_method": authority.REVIEW_METHOD,
        "reviewed_question_answer_and_evidence": True,
        "reviewer_independent_of_source_generator": True,
        "decision": "pass" if passed else "exclude",
        "semantic_correctness_verified": passed,
        "evidence_entails_entire_answer": passed,
        "question_is_user_useful": True,
        "question_is_self_contained": True,
        "answer_is_direct_and_well_formed": True,
        "safety_qualification_is_adequate": True,
        "reason_code": (
            "fully_supported_useful_seed_qa"
            if passed
            else "answer_not_fully_supported"
        ),
        "notes": "Synthetic manual-review fixture.",
    }
    value["decision_content_sha256"] = authority.canonical_sha256(value)
    return value


def test_manual_decision_rejects_forged_identity_and_unproven_pass():
    row = _row(1)
    valid = _decision(row, 1)
    assert authority.validate_decision(row, 1, valid) == valid

    forged = copy.deepcopy(valid)
    forged["source_record_sha256"] = "0" * 64
    forged["decision_content_sha256"] = authority.canonical_sha256(
        authority._decision_body(forged)
    )
    with pytest.raises(RuntimeError, match="source identity"):
        authority.validate_decision(row, 1, forged)

    unproven = copy.deepcopy(valid)
    unproven["evidence_entails_entire_answer"] = False
    unproven["decision_content_sha256"] = authority.canonical_sha256(
        authority._decision_body(unproven)
    )
    with pytest.raises(RuntimeError, match="unproven quality"):
        authority.validate_decision(row, 1, unproven)


def test_authority_accounts_for_explicit_exclusions_and_replacements(monkeypatch):
    rows = [_row(1, 2), _row(2, 3)]
    decisions = [_decision(rows[0], 1), _decision(rows[1], 2, passed=False)]
    monkeypatch.setattr(authority, "SOURCE_ASSISTANT_TOKENS", 5)
    assignments = {
        "content_sha256_before_self_field": "a" * 64,
        "chunks": [],
    }
    bundle, sealed = authority.build_authority(rows, assignments, decisions, [])
    assert bundle.endswith(b"\n")
    assert sealed["reviewed_rows"] == 2
    assert sealed["training_rows_admitted"] == 1
    assert sealed["excluded_rows"] == 1
    assert sealed["assistant_qwen36_tokens"] == 2
    assert sealed["excluded_assistant_qwen36_tokens"] == 3
    assert sealed["replacement_generated_assistant_tokens_required"] == 3
    assert sealed["review_contract"]["unresolved_rows"] == 0


def test_exclusion_requires_failed_quality_flag_and_reason():
    row = _row(1)
    excluded = _decision(row, 1, passed=False)
    assert authority.validate_decision(row, 1, excluded) == excluded

    unsupported_exclusion = copy.deepcopy(excluded)
    for flag in authority.PASS_FLAGS:
        unsupported_exclusion[flag] = True
    unsupported_exclusion["decision_content_sha256"] = authority.canonical_sha256(
        authority._decision_body(unsupported_exclusion)
    )
    with pytest.raises(RuntimeError, match="failed quality"):
        authority.validate_decision(row, 1, unsupported_exclusion)
