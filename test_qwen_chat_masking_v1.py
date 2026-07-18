from __future__ import annotations

import pytest
from transformers import AutoTokenizer

import qwen_chat_masking_v1 as masking


TOKENIZER = "models/Qwen3.6-35B-A3B"


def _tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)


def test_real_official_template_single_turn_has_nonzero_assistant_labels():
    tokenizer = _tokenizer()
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is two plus two?"},
        {"role": "assistant", "content": "Four."},
    ]
    encoded = masking.encode_chat_assistant_only(tokenizer, messages)
    assert encoded["assistant_token_count"] > 0
    assert encoded["assistant_token_count"] < encoded["total_token_count"]
    supervised = [
        token for token, label in zip(encoded["input_ids"], encoded["labels"])
        if label != -100
    ]
    decoded = tokenizer.decode(supervised, skip_special_tokens=False)
    assert "Four." in decoded
    assert "two plus two" not in decoded
    assert encoded["assistant_spans"] == [
        {
            "message_index": 2,
            "token_start": encoded["assistant_spans"][0]["token_start"],
            "token_end": encoded["total_token_count"] - 1,
            "token_count": encoded["assistant_token_count"],
        }
    ]
    assert encoded["labels"][-1] == -100


def test_real_official_template_multiturn_labels_both_assistant_payloads_only():
    tokenizer = _tokenizer()
    messages = [
        {"role": "user", "content": "Name one primary color."},
        {"role": "assistant", "content": "Red."},
        {"role": "user", "content": "Name another."},
        {"role": "assistant", "content": "Blue."},
    ]
    encoded = masking.encode_chat_assistant_only(tokenizer, messages)
    assert len(encoded["assistant_spans"]) == 2
    supervised = tokenizer.decode(
        [token for token, mask in zip(encoded["input_ids"], encoded["assistant_mask"]) if mask],
        skip_special_tokens=False,
    )
    assert "Red." in supervised and "Blue." in supervised
    assert "primary color" not in supervised and "another" not in supervised


def test_real_official_template_tool_call_labels_only_assistant_tool_payload():
    tokenizer = _tokenizer()
    tools = [{
        "type": "function",
        "function": {
            "name": "synthetic_total",
            "description": "Compute a synthetic total.",
            "parameters": {
                "type": "object",
                "properties": {"values": {"type": "array"}},
                "required": ["values"],
            },
        },
    }]
    messages = [
        {"role": "user", "content": "Use the synthetic total tool."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "synthetic_total",
                    "arguments": {"values": [2, 3]},
                },
            }],
        },
    ]
    encoded = masking.encode_chat_assistant_only(
        tokenizer, messages, tools=tools
    )
    supervised = tokenizer.decode(
        [token for token, mask in zip(
            encoded["input_ids"], encoded["assistant_mask"]
        ) if mask],
        skip_special_tokens=False,
    )
    assert "<tool_call>" in supervised
    assert "synthetic_total" in supervised
    assert "Use the synthetic total tool" not in supervised


class UnstableTokenizer:
    def convert_tokens_to_ids(self, token):
        return {"<|im_start|>": 1, "<|im_end|>": 4}[token]

    def encode(self, value, add_special_tokens=False):
        assert value == "\n" and add_special_tokens is False
        return [3]

    def decode(self, ids, skip_special_tokens=False):
        return "user" if ids == [2] else "assistant"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is True
        if add_generation_prompt:
            return [1, 2, 3, 4, 1, 5, 3, 99]
        if len(messages) == 2:
            return [1, 2, 3, 4, 1, 5, 3, 6, 4]
        return [1, 2, 3, 4]


def test_template_prefix_mismatch_fails_closed():
    with pytest.raises(ValueError, match="generation prefix differs"):
        masking.encode_chat_assistant_only(
            UnstableTokenizer(),
            [
                {"role": "user", "content": "synthetic"},
                {"role": "assistant", "content": "answer"},
            ],
        )


def test_missing_or_nonfinal_assistant_is_rejected():
    tokenizer = _tokenizer()
    with pytest.raises(ValueError, match="no assistant"):
        masking.encode_chat_assistant_only(
            tokenizer, [{"role": "user", "content": "Only a prompt."}]
        )
    with pytest.raises(ValueError, match="must end"):
        masking.encode_chat_assistant_only(
            tokenizer,
            [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
                {"role": "user", "content": "Follow-up"},
            ],
        )
