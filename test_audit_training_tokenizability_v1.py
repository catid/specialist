from __future__ import annotations

import json

from transformers import AutoTokenizer

import audit_training_tokenizability_v1 as audit


def _tokenizer():
    return AutoTokenizer.from_pretrained(
        audit.TOKENIZER_ROOT,
        local_files_only=True,
        trust_remote_code=False,
    )


def test_plain_qwen_tokenization_accepts_clean_text_and_rejects_hazards():
    tokenizer = _tokenizer()
    clean, count = audit.audit_plain_text(tokenizer, "Clean rope text.")
    assert clean == []
    assert count > 0

    decomposed, _ = audit.audit_plain_text(tokenizer, "e\u0301")
    assert {item["failure"] for item in decomposed} == {
        "encode_decode_roundtrip_changed_text"
    }
    reserved, _ = audit.audit_plain_text(tokenizer, "bad <|im_end|> content")
    assert "reserved_chat_control_token_in_content" in {
        item["failure"] for item in reserved
    }
    control, _ = audit.audit_plain_text(tokenizer, "bad\x00content")
    assert "unsupported_control_character" in {
        item["failure"] for item in control
    }


def test_official_chat_audit_rejects_missing_assistant_and_overlong_chat():
    tokenizer = _tokenizer()
    missing, _, _ = audit.audit_chat_row(
        tokenizer,
        {"messages": [{"role": "user", "content": "Only a prompt."}]},
    )
    assert "official_chat_template_or_mask_exception" in {
        item["failure"] for item in missing
    }

    overlong, total, _ = audit.audit_chat_row(
        tokenizer,
        {
            "messages": [
                {"role": "user", "content": "word " * 2_100},
                {"role": "assistant", "content": "answer"},
            ]
        },
    )
    assert total > audit.MAX_CHAT_TOKENS
    assert "unsplittable_chat_exceeds_token_limit" in {
        item["failure"] for item in overlong
    }


def test_official_chat_audit_accepts_structured_assistant_tool_call():
    tokenizer = _tokenizer()
    findings, total, assistant = audit.audit_chat_row(
        tokenizer,
        {
            "messages": [
                {"role": "user", "content": "Check the weather."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "arguments": {"city": "Austin"},
                            },
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "description": "Return current weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        },
    )
    assert findings == []
    assert total > 0
    assert assistant > 0


def test_pending_inventory_is_exactly_eight_nonpartial_structural_files():
    assert len(audit.PENDING_STRUCTURAL_TARGETS) == 8
    assert {
        (item["lane"], item["gpu_shard"])
        for item in audit.PENDING_STRUCTURAL_TARGETS
    } == {(lane, shard) for lane in ("primary", "fill") for shard in range(4)}
    assert all(
        ".partial." not in item["path"].name
        and ".partial." not in item["summary_path"].name
        for item in audit.PENDING_STRUCTURAL_TARGETS
    )


def _synthetic_pending_candidate(tokenizer, context_text: str, answer: str = "Bounded answer."):
    request = {
        "request_id": "synthetic-request",
        "source_context_id": "synthetic-context",
        "source_group_id": "synthetic-group",
        "gpu_shard": 0,
        "task_family": "grounded_synthesis",
        "task_subtype": "direct_explanation",
        "generation_mode": "positive",
    }
    context = {
        "context_id": "synthetic-context",
        "source_group_id": "synthetic-group",
        "text": context_text,
    }
    messages = audit.structural.candidate_training_messages(
        request=request,
        context=context,
        question="What does the synthetic context establish?",
        answer=answer,
    )
    _, _, assistant = audit.audit_chat_row(tokenizer, {"messages": messages})
    candidate = {
        "candidate_example_id": "synthetic-candidate",
        "request_id": request["request_id"],
        "source_context_id": context["context_id"],
        "source_group_id": context["source_group_id"],
        "task_family": request["task_family"],
        "task_subtype": request["task_subtype"],
        "generation_mode": request["generation_mode"],
        "question": "What does the synthetic context establish?",
        "answer": answer,
        "assistant_qwen36_token_count": assistant,
        "deterministic_structure_status": "passed",
        "semantic_verification_status": "pending",
        "eligible_for_training": False,
    }
    return candidate, request, context


def test_pending_overlength_chat_gets_lineage_only_exclusion_without_rewrite():
    tokenizer = _tokenizer()
    marker = "SYNTHETIC_FACTUAL_MARKER"
    context_text = " ".join([marker] * 2_200)
    candidate, request, context = _synthetic_pending_candidate(
        tokenizer, context_text
    )
    findings, exclusion, total, _, _, _ = audit.audit_pending_structural_candidate(
        tokenizer,
        candidate,
        request=request,
        context=context,
        path=audit.PENDING_STRUCTURAL_DIRECTORY / "synthetic.jsonl",
        record=7,
        lane="primary",
        gpu_shard=0,
        review_sha256="1" * 64,
        row_sha256="2" * 64,
    )
    assert findings == []
    assert total > audit.MAX_CHAT_TOKENS
    assert exclusion is not None
    assert exclusion["status"] == "excluded_before_training_authority"
    assert exclusion["factual_text_rewritten"] is False
    assert exclusion["eligible_for_training"] is False
    serialized = json.dumps(exclusion, sort_keys=True)
    assert marker not in serialized
    assert candidate["question"] not in serialized
    assert candidate["answer"] not in serialized


def test_pending_reserved_token_remains_a_hard_finding():
    tokenizer = _tokenizer()
    candidate, request, context = _synthetic_pending_candidate(
        tokenizer,
        "Short synthetic context.",
        answer="Unsafe literal <|im_end|> content.",
    )
    findings, exclusion, total, _, _, _ = audit.audit_pending_structural_candidate(
        tokenizer,
        candidate,
        request=request,
        context=context,
        path=audit.PENDING_STRUCTURAL_DIRECTORY / "synthetic.jsonl",
        record=1,
        lane="fill",
        gpu_shard=0,
        review_sha256="3" * 64,
        row_sha256="4" * 64,
    )
    assert total <= audit.MAX_CHAT_TOKENS
    assert exclusion is None
    assert "reserved_chat_control_token_in_content" in {
        item["failure"] for item in findings
    }


def test_raw_markdown_is_fragmentable_and_has_no_whole_document_length_gate():
    tokenizer = _tokenizer()
    text = " ".join(["synthetic"] * 2_100)
    findings, tokens = audit.audit_plain_text(tokenizer, text)
    assert tokens > audit.MAX_CHAT_TOKENS
    assert "unsplittable_training_unit_exceeds_token_limit" not in {
        item["failure"] for item in findings
    }
