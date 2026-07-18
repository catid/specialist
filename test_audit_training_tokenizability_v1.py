from __future__ import annotations

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
