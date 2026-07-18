#!/usr/bin/env python3
"""Official-template chat encoding with fail-closed assistant-only labels.

The bundled Qwen3.6 template does not contain Jinja ``generation`` blocks, so
Transformers' assistant-mask API returns an all-zero mask.  This module keeps
the official template byte-for-byte and derives assistant spans by exact prefix
alignment around each assistant turn.
"""

from __future__ import annotations

from typing import Any, Sequence


ALLOWED_ROLES = {"system", "user", "assistant", "tool"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def token_ids(value: Any) -> list[int]:
    if isinstance(value, dict) or hasattr(value, "keys"):
        _require("input_ids" in value, "tokenizer mapping omitted input_ids")
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    _require(isinstance(value, (list, tuple)), "tokenizer returned invalid token IDs")
    result = list(value)
    _require(
        all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in result),
        "tokenizer returned invalid token IDs",
    )
    return result


def _apply(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
    tools: Sequence[dict[str, Any]] | None,
) -> list[int]:
    kwargs = {}
    if tools:
        kwargs["tools"] = list(tools)
    return token_ids(
        tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            **kwargs,
        )
    )


def _rendered_role_blocks(tokenizer: Any, input_ids: list[int]) -> list[dict[str, Any]]:
    start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    newline_ids = token_ids(tokenizer.encode("\n", add_special_tokens=False))
    _require(
        isinstance(start_id, int)
        and isinstance(end_id, int)
        and start_id != end_id
        and len(newline_ids) == 1,
        "official ChatML boundary token identities changed",
    )
    newline_id = newline_ids[0]
    blocks = []
    cursor = 0
    while cursor < len(input_ids):
        try:
            start = input_ids.index(start_id, cursor)
        except ValueError:
            break
        try:
            end = input_ids.index(end_id, start + 1)
        except ValueError as exc:
            raise ValueError("rendered ChatML block omitted im_end") from exc
        try:
            header_end = input_ids.index(newline_id, start + 1, end)
        except ValueError as exc:
            raise ValueError("rendered ChatML block omitted role newline") from exc
        role = tokenizer.decode(
            input_ids[start + 1 : header_end], skip_special_tokens=False
        ).strip()
        _require(role in ALLOWED_ROLES, f"rendered unsupported ChatML role: {role!r}")
        blocks.append(
            {
                "role": role,
                "block_start": start,
                "content_start": header_end + 1,
                "end_token": end,
                "block_end": end + 1,
            }
        )
        cursor = end + 1
    _require(blocks, "official template rendered no ChatML role blocks")
    return blocks


def encode_chat_assistant_only(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    *,
    enable_thinking: bool = False,
    tools: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Encode a complete conversation and label only assistant payload suffixes.

    For each assistant turn, the prefix through the generated assistant header
    is rendered with ``add_generation_prompt=True``.  The conversation through
    the completed assistant turn is rendered without a generation prompt.  The
    exact aligned suffix is supervised.  Any template instability or empty
    assistant suffix fails closed.
    """

    _require(isinstance(messages, (list, tuple)) and messages, "chat messages are required")
    assistant_indices = []
    for index, message in enumerate(messages):
        _require(isinstance(message, dict), f"message {index} is not an object")
        role = message.get("role")
        _require(role in ALLOWED_ROLES, f"unsupported chat role at {index}: {role!r}")
        _require("content" in message, f"message {index} omitted content")
        if role == "assistant":
            assistant_indices.append(index)
    _require(assistant_indices, "chat has no assistant turn to supervise")
    _require(assistant_indices[-1] == len(messages) - 1, "training chat must end with an assistant turn")

    full_ids = _apply(
        tokenizer,
        messages,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
        tools=tools,
    )
    _require(full_ids, "official chat template produced no tokens")
    blocks = _rendered_role_blocks(tokenizer, full_ids)
    expected_roles = [message["role"] for message in messages]
    rendered_roles = [block["role"] for block in blocks]
    block_offset = 0
    if (
        tools
        and expected_roles[0] != "system"
        and rendered_roles == ["system", *expected_roles]
    ):
        # The official Qwen template creates a template-owned system block for
        # tool definitions when the conversation does not supply one.
        block_offset = 1
    else:
        _require(
            rendered_roles == expected_roles,
            f"rendered role blocks differ from messages: {rendered_roles!r} != {expected_roles!r}",
        )
    assistant_mask = [0] * len(full_ids)
    spans = []
    previous_end = 0
    for index in assistant_indices:
        block = blocks[index + block_offset]
        start = block["content_start"]
        end = block["block_end"]
        # The official non-thinking template inserts an empty thinking wrapper
        # only for the final assistant.  Exact generation-prefix alignment
        # excludes that template-owned wrapper from supervision.
        if index == assistant_indices[-1]:
            before = messages[:index]
            _require(before, "assistant cannot be the first training message")
            prefix_ids = _apply(
                tokenizer,
                before,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                tools=tools,
            )
            _require(
                full_ids[: len(prefix_ids)] == prefix_ids,
                f"assistant turn {index} generation prefix differs from full rendering",
            )
            start = len(prefix_ids)
        _require(start < end, f"assistant turn {index} has no aligned target tokens")
        _require(start >= previous_end, "assistant target spans overlap or regress")
        for position in range(start, end):
            _require(assistant_mask[position] == 0, "assistant target spans overlap")
            assistant_mask[position] = 1
        spans.append(
            {
                "message_index": index,
                "token_start": start,
                "token_end": end,
                "token_count": end - start,
            }
        )
        previous_end = end

    assistant_token_count = sum(assistant_mask)
    _require(assistant_token_count > 0, "assistant-only mask is empty")
    labels = [token if mask else -100 for token, mask in zip(full_ids, assistant_mask)]
    _require(
        sum(label != -100 for label in labels) == assistant_token_count,
        "assistant label count changed",
    )
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "assistant_mask": assistant_mask,
        "assistant_spans": spans,
        "total_token_count": len(full_ids),
        "assistant_token_count": assistant_token_count,
        "mask_method": "official_template_role_blocks_and_final_prefix_alignment_v1",
        "enable_thinking": enable_thinking,
    }
