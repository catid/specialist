from __future__ import annotations

from copy import deepcopy

import pytest

import build_general_replay_final_authority_v1 as authority
from general_replay_v1 import assistant_token_count, canonical_sha256


class SyntheticQwenTokenizer:
    """Small assistant-mask-compatible tokenizer; no model or dataset access."""

    def convert_tokens_to_ids(self, token):
        return {"<|im_start|>": 1, "<|im_end|>": 6}[token]

    def encode(self, value, add_special_tokens=False):
        assert value == "\n" and add_special_tokens is False
        return [5]

    def decode(self, ids, skip_special_tokens=False):
        return {2: "system", 3: "user", 4: "assistant"}.get(ids[0], "x")

    def apply_chat_template(self, messages, **kwargs):
        result = []
        roles = {"system": 2, "user": 3, "assistant": 4}
        for index, message in enumerate(messages):
            result.extend([1, roles[message["role"]], 5])
            if message["role"] == "assistant":
                if index == len(messages) - 1:
                    result.extend([8, 9])
                result.extend([30] * 17)
            else:
                result.extend([20] * 8)
            result.extend([6, 7])
        if kwargs["add_generation_prompt"]:
            result.extend([1, 4, 5, 8, 9])
        return result


def synthetic_manual_row():
    tokenizer = SyntheticQwenTokenizer()
    response_sha256 = "a" * 64
    messages = [{"role": "user", "content": "Give one calm next step."}]
    assistant = {"role": "assistant", "content": "Pause, list the facts, and choose one reversible step."}
    count = assistant_token_count(tokenizer, [*messages, assistant], [])
    item = {
        "spec_id": "replay-spec-synthetic",
        "source_group_id": "synthetic-group-one",
        "prompt_identity_sha256": "b" * 64,
        "category": "ordinary_conversation",
        "messages": messages,
        "tools": [],
        "rubric_id": "ordinary-conversation-warm-practical-v1",
        "candidates": [{
            "assistant_message": assistant,
            "assistant_token_count": count,
            "candidate_index": 1,
            "response_sha256": response_sha256,
        }],
    }
    approval = {
        "schema": "general-replay-candidate-approval-v1",
        "spec_id": item["spec_id"],
        "response_sha256": response_sha256,
        "status": "approved",
        "rubric_id": item["rubric_id"],
        "reviewer": "synthetic-reviewer",
        "reviewed_at": "2026-07-18T00:00:00Z",
        "reason": "Synthetic fixture satisfies its synthetic rubric.",
    }
    row = authority.manual_candidate_to_row(
        item,
        approval,
        tokenizer,
        ledger_sha256="c" * 64,
        priority_manifest_sha256="d" * 64,
    )
    return tokenizer, row, count


def test_manual_conversion_and_exact_final_validation_are_assistant_only():
    tokenizer, row, count = synthetic_manual_row()
    audit = authority.validate_final_rows(
        [row], tokenizer, {"ordinary_conversation": count}
    )
    assert audit["rows"] == 1
    assert audit["assistant_tokens"] == count
    assert audit["unique_source_group_ids"] == 1
    assert row["assistant_mask"] == {
        "policy": "assistant_only_v1",
        "assistant_message_indices": [1],
        "system_tokens": False,
        "user_tokens": False,
        "tool_result_tokens": False,
    }
    assert row["lineage"]["source_kind"] == (
        "manual_approved_base_model_generation_v1"
    )
    assert row["verifier"]["decision"] == "approved"


def test_final_validation_fails_closed_on_duplicate_group_and_bad_lineage():
    tokenizer, row, count = synthetic_manual_row()
    duplicate = deepcopy(row)
    duplicate["row_id"] += "-other"
    duplicate["messages"][0]["content"] += " Another unique prompt."
    with pytest.raises(RuntimeError, match="duplicate row or source group"):
        authority.validate_final_rows(
            [row, duplicate], tokenizer, {"ordinary_conversation": count * 2}
        )

    forbidden = deepcopy(row)
    forbidden["lineage"]["source_kind"] = "unreviewed_base_generation"
    with pytest.raises(RuntimeError, match="unauthorized or protected lineage"):
        authority.validate_final_rows(
            [forbidden], tokenizer, {"ordinary_conversation": count}
        )


def test_self_hash_and_jsonl_parsing_fail_closed_on_drift():
    report = {"schema": "synthetic-report-v1", "rows": 2}
    expected = canonical_sha256(report)
    report["content_sha256_before_self_field"] = expected
    authority.validate_self_hash(report, expected, "synthetic report")

    drifted = deepcopy(report)
    drifted["rows"] = 3
    with pytest.raises(RuntimeError, match="self hash failed"):
        authority.validate_self_hash(drifted, expected, "synthetic report")
    with pytest.raises(ValueError, match="blank rows forbidden"):
        authority._load_jsonl(b'{"row":1}\n\n', "synthetic rows")


def test_manual_conversion_rejects_nonapproved_or_stale_decision():
    tokenizer, row, _ = synthetic_manual_row()
    item = {
        "spec_id": row["lineage"]["spec_id"],
        "source_group_id": row["source_group_id"],
        "prompt_identity_sha256": row["lineage"]["prompt_identity_sha256"],
        "category": row["category"],
        "messages": row["messages"][:-1],
        "tools": [],
        "rubric_id": row["verifier"]["rubric_id"],
        "candidates": [{
            "assistant_message": row["messages"][-1],
            "assistant_token_count": row["assistant_token_count"],
            "candidate_index": 1,
            "response_sha256": row["lineage"]["response_sha256"],
        }],
    }
    rejected = {
        "status": "rejected",
        "spec_id": item["spec_id"],
        "response_sha256": item["candidates"][0]["response_sha256"],
        "rubric_id": item["rubric_id"],
    }
    with pytest.raises(RuntimeError, match="does not address"):
        authority.manual_candidate_to_row(
            item,
            rejected,
            tokenizer,
            ledger_sha256="c" * 64,
            priority_manifest_sha256="d" * 64,
        )
