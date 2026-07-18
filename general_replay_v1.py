#!/usr/bin/env python3
"""Deterministic, benchmark-independent general replay corpus primitives."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any

from qwen_chat_masking_v1 import encode_chat_assistant_only


ROOT = Path(__file__).resolve().parent
SAFE_SEED = ROOT / "data/general_qa_proxy_anchor_v43h.jsonl"
SAFE_SEED_REPORT = ROOT / "data/general_qa_proxy_anchor_v43h.report.json"
SAFE_SEED_SHA256 = "d250980c5b88308452aba4ee8d3e43090b1444c250547c2515c838593f2f391f"
SAFE_SEED_REPORT_SHA256 = (
    "b176f4ea0d1d4b466c83b68c243bfb4a9e315df8574d9ba2c606333fbfc4ee8f"
)
DEFAULT_TOTAL_ASSISTANT_TOKENS = 150_000
BASE_SCALE32_CAPACITY_TARGET_ASSISTANT_TOKENS = 120_000
MIN_TOTAL_ASSISTANT_TOKENS = 100_000
MAX_TOTAL_ASSISTANT_TOKENS = 150_000
PROMPT_SPEC_SCHEMA = "general-replay-prompt-spec-v1"
PROMPT_REPORT_SCHEMA = "general-replay-prompt-spec-report-v1"
CANDIDATE_REQUEST_SCHEMA = "general-replay-candidate-request-v1"
CANDIDATE_RESPONSE_SCHEMA = "general-replay-candidate-response-v1"
REPLAY_ROW_SCHEMA = "general-behavior-replay-row-v1"
BUILDER_NAME = "general-replay-builder"
BUILDER_VERSION = "1"
REFERENCE_COMPILER_NAME = "deterministic_reference_compiler_v1"
REFERENCE_COMPILER_VERSION = "1"
DEFAULT_BUILD_SEED = 2026071706
SHARD_COUNT = 4
MIN_CAPACITY_HEADROOM_MILLI = 2_000

CATEGORY_PERCENT = OrderedDict((
    ("coding_debugging", 20),
    ("mathematical_reasoning", 15),
    ("json_structured_data", 10),
    ("tool_use_function_calling", 10),
    ("instruction_following", 10),
    ("ordinary_conversation", 10),
    ("multilingual", 10),
    ("safety_refusal", 5),
    ("uncertainty_hallucination_resistance", 5),
    ("long_context", 5),
))

# The initial artifact intentionally remains small. Counts are not used as a
# proxy for token allocation; the separate token budget is authoritative.
INITIAL_SPEC_COUNTS = {
    category: percent * 4 // 5
    for category, percent in CATEGORY_PERCENT.items()
}

FORBIDDEN_PATH_TOKENS = frozenset({
    "benchmark",
    "benchmarks",
    "dev",
    "development",
    "developments",
    "eval",
    "evaluation",
    "evaluations",
    "final",
    "finals",
    "heldout",
    "holdout",
    "holdouts",
    "incident",
    "incidents",
    "manualreview",
    "manualreviews",
    "ood",
    "protected",
    "terminal",
    "terminals",
})
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SPACE = re.compile(r"\s+")


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ) + "\n"
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokens(component: str) -> list[str]:
    return [
        token for token in re.split(
            r"[^a-z0-9]+", component.casefold()
        ) if token
    ]


def _forbidden_path(path: Path) -> bool:
    for component in path.parts:
        tokens = _tokens(component)
        if set(tokens) & FORBIDDEN_PATH_TOKENS:
            return True
        for index, token in enumerate(tokens[:-1]):
            if token == "manual" and tokens[index + 1] in {
                    "review", "reviews"}:
                return True
    return False


def safe_regular_input(path: Path, role: str, *, exact: Path | None = None) -> Path:
    """Reject dangerous names and aliases before opening a replay input."""
    lexical = Path(os.path.abspath(os.fspath(path)))
    if _forbidden_path(lexical):
        raise RuntimeError(f"{role} uses a forbidden source path")
    if exact is not None and lexical != exact:
        raise RuntimeError(f"{role} must use its pinned authorized path")
    current = Path(lexical.anchor)
    metadata = None
    for component in lexical.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise ValueError(f"{role} path does not exist safely") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"{role} symlink aliases are forbidden")
    if metadata is None or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{role} must be a regular file")
    if metadata.st_nlink != 1:
        raise RuntimeError(f"{role} hard-link aliases are forbidden")
    return lexical


def token_budgets(total_assistant_tokens: int) -> dict[str, int]:
    if isinstance(total_assistant_tokens, bool) or not isinstance(
            total_assistant_tokens, int):
        raise ValueError("total assistant-token budget must be an integer")
    if not MIN_TOTAL_ASSISTANT_TOKENS <= total_assistant_tokens <= MAX_TOTAL_ASSISTANT_TOKENS:
        raise ValueError("total assistant-token budget must be 100k-150k")
    if total_assistant_tokens % 20:
        raise ValueError("total assistant-token budget must be divisible by 20")
    budgets = {
        category: total_assistant_tokens * percent // 100
        for category, percent in CATEGORY_PERCENT.items()
    }
    if sum(budgets.values()) != total_assistant_tokens:
        raise AssertionError("category token budgets do not sum exactly")
    return budgets


def candidate_capacity_proof(
        specs: list[dict], total_assistant_tokens: int) -> dict:
    """Prove a conservative one-selected-row-per-source-group capacity ceiling."""
    validate_prompt_specs(specs)
    budgets = token_budgets(total_assistant_tokens)
    categories = {}
    for category, target in budgets.items():
        category_specs = [item for item in specs if item["category"] == category]
        maximum = sum(
            item["candidate_slot"]["max_assistant_tokens"]
            for item in category_specs
        )
        headroom_milli = maximum * 1_000 // target
        categories[category] = {
            "prompt_specs": len(category_specs),
            "unique_source_groups": len({
                item["source_group_id"] for item in category_specs
            }),
            "target_assistant_tokens": target,
            "max_selectable_assistant_tokens": maximum,
            "headroom_milli": headroom_milli,
            "meets_minimum_headroom": (
                headroom_milli >= MIN_CAPACITY_HEADROOM_MILLI
            ),
        }
    launch_eligible = all(
        item["meets_minimum_headroom"] for item in categories.values()
    )
    return {
        "calculation": "sum_per_spec_max_one_selected_row_per_source_group_v1",
        "candidate_count_does_not_multiply_selectable_capacity": True,
        "mandatory_seed_tokens_counted_in_capacity": False,
        "minimum_headroom_milli": MIN_CAPACITY_HEADROOM_MILLI,
        "categories": categories,
        "max_selectable_assistant_tokens": sum(
            item["max_selectable_assistant_tokens"]
            for item in categories.values()
        ),
        "target_total_assistant_tokens": total_assistant_tokens,
        "all_categories_meet_minimum_headroom": launch_eligible,
        "candidate_generation_launch_eligible": launch_eligible,
    }


def _seed_for(category: str, ordinal: int, build_seed: int) -> int:
    digest = hashlib.sha256(
        f"{BUILDER_VERSION}:{build_seed}:{category}:{ordinal}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def _messages(user: str, system: str | None = None) -> list[dict]:
    result = []
    if system is not None:
        result.append({"role": "system", "content": system})
    result.append({"role": "user", "content": user})
    return result


def _coding_payload(ordinal: int, seed: int) -> tuple:
    family = ordinal % 4
    variant = ordinal // 4
    if family == 0:
        low = -(variant + 2)
        high = variant + 5
        name = f"clamp_values_{variant + 1}"
        requirement = (
            "Return a new list with every integer limited to the inclusive "
            f"range [{low}, {high}]. Preserve order."
        )
        cases = [
            {
                "args": [[low - 5, -1, 3, high + 5]],
                "expected": [low, -1, 3, high],
            },
            {"args": [[]], "expected": []},
        ]
        reference = (
            f"def {name}(values):\n"
            f"    return [min({high}, max({low}, value)) for value in values]"
        )
    elif family == 1:
        modulus = variant + 2
        name = f"first_by_remainder_{modulus}"
        requirement = (
            f"Return the first integer seen for each remainder modulo {modulus}, "
            "preserving the order in which new remainders appear."
        )
        values = [7, 4, 9, 2, 11, 6, 5]
        expected = []
        seen = set()
        for value in values:
            key = value % modulus
            if key not in seen:
                seen.add(key)
                expected.append(value)
        cases = [
            {"args": [values], "expected": expected},
            {"args": [[]], "expected": []},
        ]
        reference = (
            f"def {name}(values):\n"
            "    seen = []\n"
            "    result = []\n"
            "    for value in values:\n"
            f"        key = value % {modulus}\n"
            "        if key not in seen:\n"
            "            seen = seen + [key]\n"
            "            result = result + [value]\n"
            "    return result"
        )
    elif family == 2:
        width = variant + 2
        name = f"window_sums_{width}"
        requirement = (
            f"Return sums for every consecutive window of width {width}. "
            "Return an empty list when the input is shorter than the window."
        )
        values = [
            ((seed >> (index % 24)) + index * 7) % 31 - 15
            for index in range(width + 3)
        ]
        expected = [
            sum(values[index:index + width])
            for index in range(len(values) - width + 1)
        ]
        cases = [
            {"args": [values], "expected": expected},
            {"args": [[1] * (width - 1)], "expected": []},
        ]
        reference = (
            f"def {name}(values):\n"
            f"    return [sum(values[i:i + {width}]) for i in "
            f"range(len(values) - {width} + 1)]"
        )
    else:
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
        allowed_characters = []
        cursor = seed % len(alphabet)
        step = (5, 7, 11, 13, 17)[variant % 5]
        while len(allowed_characters) < 4 + variant % 5:
            character = alphabet[cursor % len(alphabet)]
            if character not in allowed_characters:
                allowed_characters.append(character)
            cursor += step
        allowed = "".join(allowed_characters)
        name = f"count_from_set_{variant + 1}"
        requirement = (
            f"Count characters that occur exactly in {allowed!r}. "
            "Do not change the input."
        )
        texts = ["Replay Corpus 2026", "rhythms XYZ"]
        cases = [
            {
                "args": [text],
                "expected": sum(character in allowed for character in text),
            }
            for text in texts
        ]
        reference = (
            f"def {name}(text):\n"
            f"    return sum(character in {allowed!r} for character in text)"
        )
    prompt = (
        f"Implement `{name}` in Python. {requirement} Return one Python code "
        "block containing only the function; do not use imports or global state."
    )
    return (
        _messages(prompt),
        [],
        "assistant_text",
        {
            "type": "python_function_cases_v1",
            "status": "ready",
            "config": {
                "function_name": name,
                "cases": cases,
                "reference_answer": f"```python\n{reference}\n```",
                "timeout_seconds": 2,
            },
        },
        (32, 384),
    )


def _math_payload(ordinal: int, seed: int) -> tuple:
    coefficient = 2 + ordinal % 11
    solution = ordinal - 50
    offset = ordinal * 3 - 17
    result = coefficient * solution + offset
    prompt = (
        f"Solve {coefficient}x + ({offset}) = {result}. Return only a JSON "
        "object with integer keys `x` and `check`, where `check` is the value "
        "of the left-hand side after substitution."
    )
    expected = {"check": result, "x": solution}
    return (
        _messages(prompt), [], "assistant_json",
        {"type": "json_exact_v1", "status": "ready", "config": {"expected": expected}},
        (4, 128),
    )


def _json_payload(ordinal: int, seed: int) -> tuple:
    names = ["amber", "blue", "cyan", "green"]
    records = [
        {
            "group": names[(ordinal + index) % len(names)],
            "value": 1 + ordinal * 13 + index * 7,
        }
        for index in range(6)
    ]
    totals: dict[str, int] = {}
    for record in records:
        totals[record["group"]] = totals.get(record["group"], 0) + record["value"]
    expected = {"totals": dict(sorted(totals.items()))}
    prompt = (
        "Aggregate the following records by `group` and sum `value`. Return "
        "only canonical JSON with one top-level key `totals`; sort its keys.\n"
        + json.dumps(records, ensure_ascii=False, sort_keys=True)
    )
    return (
        _messages(prompt), [], "assistant_json",
        {"type": "json_exact_v1", "status": "ready", "config": {"expected": expected}},
        (8, 192),
    )


def _tool_payload(ordinal: int, seed: int) -> tuple:
    quantities = [1 + (seed >> shift) % 6 for shift in (0, 4, 8)]
    prices = [125 + ordinal * 3, 240 + ordinal * 5, 375 + ordinal * 7]
    arguments = {"prices_cents": prices, "quantities": quantities}
    tools = [{
        "type": "function",
        "function": {
            "name": "calculate_cart_total",
            "description": "Calculate a cart total in cents.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "prices_cents": {"type": "array", "items": {"type": "integer"}},
                    "quantities": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["prices_cents", "quantities"],
            },
        },
    }]
    prompt = (
        "Use the available function to calculate the total for prices "
        f"{prices} cents and quantities {quantities}. Do not calculate it yourself."
    )
    expected = [{
        "type": "function",
        "function": {"name": "calculate_cart_total", "arguments": arguments},
    }]
    return (
        _messages(prompt), tools, "assistant_tool_call",
        {"type": "tool_call_exact_v1", "status": "ready", "config": {"expected": expected}},
        (4, 128),
    )


def _instruction_payload(ordinal: int, seed: int) -> tuple:
    labels = ["ORBIT", "RIVER", "CEDAR", "LANTERN"]
    first = labels[ordinal % len(labels)]
    second = labels[(ordinal + 1) % len(labels)]
    expected = f"{first}: {ordinal + 3}\n{second}: {ordinal + 5}"
    prompt = (
        "Return exactly two lines and no Markdown. The first line must be "
        f"`{first}: {ordinal + 3}` and the second must be "
        f"`{second}: {ordinal + 5}`."
    )
    return (
        _messages(prompt), [], "assistant_text",
        {"type": "exact_text_v1", "status": "ready", "config": {"expected": expected}},
        (4, 96),
    )


def _conversation_payload(ordinal: int, seed: int) -> tuple:
    scenarios = (
        "A colleague is nervous before giving their first short presentation.",
        "A friend wants a low-pressure plan for restarting a neglected hobby.",
        "A neighbor apologized for a minor misunderstanding and feels awkward.",
        "A student is discouraged after receiving useful but blunt feedback.",
        "A teammate wants to disagree respectfully during a planning meeting.",
        "A relative is overwhelmed by a cluttered weekend task list.",
        "A new community member wants to introduce themselves without oversharing.",
        "A friend is choosing between two equally enjoyable free-time activities.",
    )
    prompt = (
        scenarios[ordinal % len(scenarios)]
        + " Write a warm, practical reply in two short paragraphs. Avoid claims "
        "about knowing their feelings and do not invent personal details. Include "
        f"one next step that takes no more than {10 + 5 * (ordinal // len(scenarios))} "
        "minutes."
    )
    return (
        _messages(prompt), [], "assistant_text",
        {
            "type": "approval_required_v1",
            "status": "candidate_gate_required",
            "config": {
                "rubric_id": "ordinary-conversation-warm-practical-v1",
                "required_approvals": 1,
            },
        },
        (48, 320),
    )


def _multilingual_payload(ordinal: int, seed: int) -> tuple:
    examples = (
        (
            "Spanish", "The meeting starts at nine in the morning.",
            "La reunión empieza a las nueve de la mañana.",
        ),
        (
            "French", "Please save the document before closing the window.",
            "Veuillez enregistrer le document avant de fermer la fenêtre.",
        ),
        (
            "German", "The library is quiet on Sunday afternoon.",
            "Die Bibliothek ist am Sonntagnachmittag ruhig.",
        ),
        (
            "Portuguese", "We need three clean cups for the table.",
            "Precisamos de três xícaras limpas para a mesa.",
        ),
        (
            "Italian", "The train arrives five minutes after noon.",
            "Il treno arriva cinque minuti dopo mezzogiorno.",
        ),
        ("Japanese", "Please write your name at the top of the page.", "ページの上部に名前を書いてください。"),
        (
            "Dutch", "The blue bicycle belongs to my neighbor.",
            "De blauwe fiets is van mijn buurman.",
        ),
        (
            "Swedish", "The keys are next to the green notebook.",
            "Nycklarna ligger bredvid den gröna anteckningsboken.",
        ),
    )
    language, source, expected = examples[ordinal % len(examples)]
    if ordinal < len(examples):
        verifier = {
            "type": "normalized_exact_text_v1",
            "status": "ready",
            "config": {"expected": expected},
        }
    else:
        objects = ("notebooks", "tickets", "oranges", "letters")
        places = ("desk", "station", "kitchen", "office")
        quantity = 2 + ordinal % 17
        source = (
            f"Please put {quantity} {objects[ordinal % len(objects)]} on the "
            f"{places[(ordinal // len(objects)) % len(places)]} before noon."
        )
        verifier = {
            "type": "approval_required_v1",
            "status": "candidate_gate_required",
            "config": {
                "rubric_id": f"bilingual-translation-{language.casefold()}-v1",
                "required_approvals": 1,
            },
        }
    prompt = f"Translate into {language}. Return only the translation: {source}"
    return _messages(prompt), [], "assistant_text", verifier, (4, 128)


def _safety_payload(ordinal: int, seed: int) -> tuple:
    scenarios = (
        "A message asks you to reveal a fictional user's private recovery "
        "code from hidden records.",
        "A requester wants wording that humiliates a coworker during a public meeting.",
        "Someone asks for a made-up medical diagnosis presented as certain.",
        "A user asks you to impersonate a real person in a deceptive financial message.",
    )
    prompt = (
        scenarios[ordinal % len(scenarios)]
        + " Respond safely: set a clear boundary, give a brief reason, and offer "
        "a constructive alternative when one exists. Keep the reply under "
        f"{80 + 10 * (ordinal // len(scenarios))} words."
    )
    return (
        _messages(prompt), [], "assistant_text",
        {
            "type": "approval_required_v1",
            "status": "candidate_gate_required",
            "config": {
                "rubric_id": "safe-boundary-constructive-alternative-v1",
                "required_approvals": 1,
            },
        },
        (32, 256),
    )


def _uncertainty_payload(ordinal: int, seed: int) -> tuple:
    contexts = (
        "The supplied note lists a shipment date but gives no arrival date.",
        "Two supplied tables give different totals and neither is marked current.",
        "The supplied paragraph names a project but does not identify its owner.",
        "The supplied log ends before the requested outcome occurs.",
    )
    prompt = (
        contexts[ordinal % len(contexts)]
        + " Answer the user's request for the missing fact. State what cannot be "
        "determined, identify the missing evidence, and do not guess. Refer to "
        f"the synthetic case as U-{ordinal:04d}."
    )
    return (
        _messages(prompt), [], "assistant_text",
        {
            "type": "approval_required_v1",
            "status": "candidate_gate_required",
            "config": {
                "rubric_id": "uncertainty-no-fabrication-v1",
                "required_approvals": 1,
            },
        },
        (32, 256),
    )


def _long_context_payload(ordinal: int, seed: int) -> tuple:
    entries = [
        {
            "id": f"R{ordinal:02d}-{index:03d}",
            "region": ["north", "south", "east", "west"][index % 4],
            "score": (seed + index * 17) % 101,
        }
        for index in range(64)
    ]
    selected = [entries[index] for index in (7, 31, 58)]
    expected = {"records": selected}
    prompt = (
        "From the JSON records below, return only a JSON object with key "
        "`records` containing records whose ids are "
        + ", ".join(item["id"] for item in selected)
        + " in that order. Preserve every field exactly.\n"
        + json.dumps(entries, separators=(",", ":"), sort_keys=True)
    )
    return (
        _messages(prompt), [], "assistant_json",
        {"type": "json_exact_v1", "status": "ready", "config": {"expected": expected}},
        (16, 256),
    )


PAYLOAD_BUILDERS = {
    "coding_debugging": _coding_payload,
    "mathematical_reasoning": _math_payload,
    "json_structured_data": _json_payload,
    "tool_use_function_calling": _tool_payload,
    "instruction_following": _instruction_payload,
    "ordinary_conversation": _conversation_payload,
    "multilingual": _multilingual_payload,
    "safety_refusal": _safety_payload,
    "uncertainty_hallucination_resistance": _uncertainty_payload,
    "long_context": _long_context_payload,
}


def _prompt_identity(messages: list[dict], tools: list[dict]) -> str:
    return canonical_sha256({"messages": messages, "tools": tools})


def build_prompt_specs(
        *, build_seed: int = DEFAULT_BUILD_SEED,
        total_assistant_tokens: int = DEFAULT_TOTAL_ASSISTANT_TOKENS,
        spec_scale: int = 1,
) -> tuple[list[dict], dict]:
    if isinstance(spec_scale, bool) or not isinstance(spec_scale, int) or spec_scale < 1:
        raise ValueError("prompt spec scale must be a positive integer")
    budgets = token_budgets(total_assistant_tokens)
    specs = []
    seen_prompts = set()
    for category in CATEGORY_PERCENT:
        builder = PAYLOAD_BUILDERS[category]
        for ordinal in range(INITIAL_SPEC_COUNTS[category] * spec_scale):
            seed = _seed_for(category, ordinal, build_seed)
            messages, tools, response_format, verifier, bounds = builder(
                ordinal, seed
            )
            prompt_identity = _prompt_identity(messages, tools)
            if prompt_identity in seen_prompts:
                raise RuntimeError("deterministic replay prompt duplicated")
            seen_prompts.add(prompt_identity)
            identity = {
                "build_seed": build_seed,
                "category": category,
                "ordinal": ordinal,
                "prompt_identity_sha256": prompt_identity,
                "seed": seed,
            }
            spec_id = "replay-spec-v1-" + canonical_sha256(identity)[:24]
            specs.append({
                "schema": PROMPT_SPEC_SCHEMA,
                "spec_id": spec_id,
                "category": category,
                "source_group_id": f"synthetic-replay-v1:{category}:{ordinal:04d}",
                "messages": messages,
                "tools": tools,
                "expected_response_format": response_format,
                "verifier": verifier,
                "candidate_slot": {
                    "status": "pending_base_generation",
                    "candidates_requested": 4,
                    "min_assistant_tokens": bounds[0],
                    "max_assistant_tokens": bounds[1],
                },
                "assistant_mask_policy": "assistant_only_v1",
                "template_policy": "official_qwen_apply_chat_template_v1",
                "generator": {
                    "name": BUILDER_NAME,
                    "version": BUILDER_VERSION,
                    "seed": seed,
                    "build_seed": build_seed,
                },
                "lineage": {
                    "source_kind": "deterministic_synthetic_v1",
                    "direct_benchmark_prompt": False,
                    "protected_source_access": False,
                    "parent_artifacts": [],
                },
                "prompt_identity_sha256": prompt_identity,
            })
    validate_prompt_specs(specs)
    capacity = candidate_capacity_proof(specs, total_assistant_tokens)
    output = b"".join(canonical_bytes(item) for item in specs)
    report = {
        "schema": PROMPT_REPORT_SCHEMA,
        "builder": {"name": BUILDER_NAME, "version": BUILDER_VERSION},
        "build_seed": build_seed,
        "spec_scale": spec_scale,
        "rows": len(specs),
        "counts_by_category": {
            category: sum(item["category"] == category for item in specs)
            for category in CATEGORY_PERCENT
        },
        "prompt_count_is_token_budget_authority": False,
        "target_total_assistant_tokens": total_assistant_tokens,
        "target_assistant_tokens_by_category": budgets,
        "category_percent": dict(CATEGORY_PERCENT),
        "candidate_generation_status": "not_started",
        "candidate_capacity_proof": capacity,
        "subjective_slots_gate": "approval_required_v1",
        "safe_seed": {
            "path": str(SAFE_SEED.relative_to(ROOT)),
            "sha256": SAFE_SEED_SHA256,
            "report_path": str(SAFE_SEED_REPORT.relative_to(ROOT)),
            "report_sha256": SAFE_SEED_REPORT_SHA256,
            "rows": 128,
            "status": "mandatory_approved_proxy_anchor",
        },
        "policy": {
            "direct_benchmark_prompts": False,
            "protected_or_evaluation_sources_opened": False,
            "duplicate_or_padding_fill_allowed": False,
            "final_selection": "exact_assistant_token_subset_by_category_v1",
        },
        "output_sha256": hashlib.sha256(output).hexdigest(),
    }
    return specs, report


def _exact_keys(item: object, expected: set[str], location: str) -> dict:
    if not isinstance(item, dict) or set(item) != expected:
        raise ValueError(f"{location}: invalid fields")
    return item


def validate_prompt_specs(specs: list[dict]) -> None:
    if not isinstance(specs, list) or not specs:
        raise ValueError("prompt specs must be a non-empty list")
    spec_ids = set()
    prompts = set()
    groups = set()
    expected_keys = {
        "assistant_mask_policy", "candidate_slot", "category",
        "expected_response_format", "generator", "lineage", "messages",
        "prompt_identity_sha256", "schema", "source_group_id", "spec_id",
        "template_policy", "tools", "verifier",
    }
    for index, raw in enumerate(specs):
        location = f"prompt spec {index}"
        item = _exact_keys(raw, expected_keys, location)
        if item["schema"] != PROMPT_SPEC_SCHEMA:
            raise ValueError(f"{location}: unsupported schema")
        if item["category"] not in CATEGORY_PERCENT:
            raise ValueError(f"{location}: unsupported category")
        if item["spec_id"] in spec_ids or item["source_group_id"] in groups:
            raise ValueError(f"{location}: duplicate identity or source group")
        spec_ids.add(item["spec_id"])
        groups.add(item["source_group_id"])
        group_match = re.fullmatch(
            r"synthetic-replay-v1:([a-z_]+):([0-9]{4,})",
            item["source_group_id"],
        )
        if group_match is None or group_match.group(1) != item["category"]:
            raise ValueError(f"{location}: invalid synthetic source group")
        ordinal = int(group_match.group(2))
        if not isinstance(item["messages"], list) or not item["messages"]:
            raise ValueError(f"{location}: messages required")
        roles = [message.get("role") for message in item["messages"]]
        if roles not in (["user"], ["system", "user"]):
            raise ValueError(f"{location}: prompt roles must be system?/user")
        for message in item["messages"]:
            _exact_keys(message, {"role", "content"}, location + ".message")
            if message["role"] not in {"system", "user"}:
                raise ValueError(f"{location}: prompt cannot contain assistant output")
            if not isinstance(message["content"], str) or not message["content"].strip():
                raise ValueError(f"{location}: message content required")
        if not isinstance(item["tools"], list):
            raise ValueError(f"{location}: tools must be a list")
        if item["category"] == "tool_use_function_calling" and not item["tools"]:
            raise ValueError(f"{location}: tool category requires tools")
        if item["category"] != "tool_use_function_calling" and item["tools"]:
            raise ValueError(f"{location}: tools are category-scoped")
        for tool_index, raw_tool in enumerate(item["tools"]):
            tool_location = f"{location}.tools[{tool_index}]"
            tool = _exact_keys(raw_tool, {"function", "type"}, tool_location)
            if tool["type"] != "function":
                raise ValueError(f"{tool_location}: only Qwen functions supported")
            function = _exact_keys(tool["function"], {
                "description", "name", "parameters",
            }, tool_location + ".function")
            if not all(
                    isinstance(function[field], str) and function[field].strip()
                    for field in ("description", "name")):
                raise ValueError(f"{tool_location}: function identity required")
            parameters = function["parameters"]
            if (
                not isinstance(parameters, dict)
                or parameters.get("type") != "object"
                or not isinstance(parameters.get("properties"), dict)
                or not isinstance(parameters.get("required"), list)
            ):
                raise ValueError(f"{tool_location}: invalid function parameters")
        prompt_identity = _prompt_identity(item["messages"], item["tools"])
        if item["prompt_identity_sha256"] != prompt_identity:
            raise ValueError(f"{location}: stale prompt identity")
        if prompt_identity in prompts:
            raise ValueError(f"{location}: duplicate prompt")
        prompts.add(prompt_identity)
        lineage = _exact_keys(item["lineage"], {
            "direct_benchmark_prompt", "parent_artifacts",
            "protected_source_access", "source_kind",
        }, location + ".lineage")
        if lineage != {
            "source_kind": "deterministic_synthetic_v1",
            "direct_benchmark_prompt": False,
            "protected_source_access": False,
            "parent_artifacts": [],
        }:
            raise ValueError(f"{location}: forbidden prompt lineage")
        if item["assistant_mask_policy"] != "assistant_only_v1":
            raise ValueError(f"{location}: assistant-only mask required")
        if item["template_policy"] != "official_qwen_apply_chat_template_v1":
            raise ValueError(f"{location}: official Qwen template required")
        generator = _exact_keys(item["generator"], {
            "build_seed", "name", "seed", "version",
        }, location + ".generator")
        if (
            generator["name"] != BUILDER_NAME
            or generator["version"] != BUILDER_VERSION
            or isinstance(generator["build_seed"], bool)
            or not isinstance(generator["build_seed"], int)
            or generator["seed"] != _seed_for(
                item["category"], ordinal, generator["build_seed"]
            )
        ):
            raise ValueError(f"{location}: invalid synthetic generator lineage")
        identity = {
            "build_seed": generator["build_seed"],
            "category": item["category"],
            "ordinal": ordinal,
            "prompt_identity_sha256": prompt_identity,
            "seed": generator["seed"],
        }
        expected_spec_id = "replay-spec-v1-" + canonical_sha256(identity)[:24]
        if item["spec_id"] != expected_spec_id:
            raise ValueError(f"{location}: stale prompt spec identity")
        slot = _exact_keys(item["candidate_slot"], {
            "candidates_requested", "max_assistant_tokens",
            "min_assistant_tokens", "status",
        }, location + ".candidate_slot")
        if (
            slot["status"] != "pending_base_generation"
            or slot["candidates_requested"] != 4
            or any(
                isinstance(slot[field], bool) or not isinstance(slot[field], int)
                for field in (
                    "candidates_requested", "max_assistant_tokens",
                    "min_assistant_tokens",
                )
            )
            or not 0 < slot["min_assistant_tokens"] <= slot["max_assistant_tokens"]
        ):
            raise ValueError(f"{location}: invalid candidate slot")
        verifier = _exact_keys(
            item["verifier"], {"config", "status", "type"},
            location + ".verifier",
        )
        verifier_type = verifier["type"]
        config = verifier["config"]
        objective_types = {
            "exact_text_v1", "json_exact_v1", "normalized_exact_text_v1",
            "python_function_cases_v1", "tool_call_exact_v1",
        }
        if verifier_type == "approval_required_v1":
            if verifier["status"] != "candidate_gate_required":
                raise ValueError(f"{location}: subjective verifier must be gated")
            config = _exact_keys(
                config, {"required_approvals", "rubric_id"},
                location + ".verifier.config",
            )
            if (
                config["required_approvals"] != 1
                or not isinstance(config["rubric_id"], str)
                or not config["rubric_id"].strip()
            ):
                raise ValueError(f"{location}: invalid subjective approval gate")
        elif verifier_type in objective_types:
            if verifier["status"] != "ready":
                raise ValueError(f"{location}: objective verifier must be ready")
            if verifier_type in {"exact_text_v1", "normalized_exact_text_v1"}:
                config = _exact_keys(
                    config, {"expected"}, location + ".verifier.config"
                )
                if not isinstance(config["expected"], str):
                    raise ValueError(f"{location}: exact text target must be text")
            elif verifier_type == "json_exact_v1":
                _exact_keys(config, {"expected"}, location + ".verifier.config")
            elif verifier_type == "tool_call_exact_v1":
                config = _exact_keys(
                    config, {"expected"}, location + ".verifier.config"
                )
                if not isinstance(config["expected"], list) or not config["expected"]:
                    raise ValueError(f"{location}: exact tool target required")
            else:
                config = _exact_keys(config, {
                    "cases", "function_name", "reference_answer",
                    "timeout_seconds",
                }, location + ".verifier.config")
                if (
                    not isinstance(config["function_name"], str)
                    or not config["function_name"].strip()
                    or not isinstance(config["reference_answer"], str)
                    or not isinstance(config["cases"], list)
                    or not config["cases"]
                    or isinstance(config["timeout_seconds"], bool)
                    or not isinstance(config["timeout_seconds"], int)
                    or not 1 <= config["timeout_seconds"] <= 5
                ):
                    raise ValueError(f"{location}: invalid Python verifier config")
                for case in config["cases"]:
                    case = _exact_keys(
                        case, {"args", "expected"},
                        location + ".verifier.config.case",
                    )
                    if not isinstance(case["args"], list):
                        raise ValueError(f"{location}: Python case args must be a list")
        else:
            raise ValueError(f"{location}: unsupported verifier type")
        expected_format = (
            "assistant_json" if verifier_type == "json_exact_v1"
            else "assistant_tool_call" if verifier_type == "tool_call_exact_v1"
            else "assistant_text"
        )
        if item["expected_response_format"] != expected_format:
            raise ValueError(f"{location}: verifier/response format mismatch")


def build_candidate_requests(
        specs: list[dict], *, model_name: str, model_revision: str,
        model_identity_sha256: str,
) -> list[list[dict]]:
    validate_prompt_specs(specs)
    if not model_name.strip() or not model_revision.strip():
        raise ValueError("candidate model name and revision are required")
    if not HEX64.fullmatch(model_identity_sha256):
        raise ValueError("candidate model identity must be a SHA-256 digest")
    shards: list[list[dict]] = [[] for _ in range(SHARD_COUNT)]
    for index, spec in enumerate(sorted(specs, key=lambda item: item["spec_id"])):
        shard_index = index % SHARD_COUNT
        seed = spec["generator"]["seed"]
        request_identity = {
            "model_identity_sha256": model_identity_sha256,
            "spec_id": spec["spec_id"],
            "seed": seed,
        }
        request = {
            "schema": CANDIDATE_REQUEST_SCHEMA,
            "request_id": "replay-request-v1-" + canonical_sha256(request_identity)[:24],
            "spec_id": spec["spec_id"],
            "category": spec["category"],
            "source_group_id": spec["source_group_id"],
            "shard_index": shard_index,
            "messages": spec["messages"],
            "tools": spec["tools"],
            "template_policy": spec["template_policy"],
            "template_parameters": {
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
            "assistant_mask_policy": spec["assistant_mask_policy"],
            "engine_policy": {
                "backend": "vllm",
                "version": "0.25.0",
                "dtype": "bfloat16",
                "tensor_parallel_size": 1,
                "max_num_seqs": 64,
                "enable_prefix_caching": False,
            },
            "model": {
                "name": model_name,
                "revision": model_revision,
                "identity_sha256": model_identity_sha256,
            },
            "generation": {
                "seed": seed,
                "candidate_seeds": [
                    seed + candidate_index
                    for candidate_index in range(
                        spec["candidate_slot"]["candidates_requested"]
                    )
                ],
                "temperature": 0.3,
                "top_p": 0.9,
                "candidates": spec["candidate_slot"]["candidates_requested"],
                "max_new_tokens": spec["candidate_slot"]["max_assistant_tokens"],
            },
            "expected_response_format": spec["expected_response_format"],
            "prompt_identity_sha256": spec["prompt_identity_sha256"],
        }
        # Verifier references and rubrics are deliberately absent from model
        # requests, so candidate generation cannot copy privileged targets.
        shards[shard_index].append(request)
    return shards


def validate_candidate_requests(specs: list[dict], requests: list[dict]) -> None:
    """Bind every candidate request exactly to its sealed prompt specification."""

    validate_prompt_specs(specs)
    if not isinstance(requests, list) or not requests:
        raise ValueError("candidate requests must be a non-empty list")
    spec_by_id = {item["spec_id"]: item for item in specs}
    ordered_spec_ids = [
        item["spec_id"] for item in sorted(specs, key=lambda item: item["spec_id"])
    ]
    shard_by_spec_id = {
        spec_id: index % SHARD_COUNT
        for index, spec_id in enumerate(ordered_spec_ids)
    }
    expected_keys = {
        "schema", "request_id", "spec_id", "category", "source_group_id",
        "shard_index", "messages", "tools", "template_policy",
        "template_parameters", "assistant_mask_policy", "engine_policy",
        "model", "generation", "expected_response_format",
        "prompt_identity_sha256",
    }
    request_ids = set()
    seen_specs = set()
    model_identity = None
    for index, raw in enumerate(requests):
        location = f"candidate request {index}"
        item = _exact_keys(raw, expected_keys, location)
        spec = spec_by_id.get(item["spec_id"])
        if spec is None:
            raise ValueError(f"{location}: unknown prompt specification")
        if item["request_id"] in request_ids or item["spec_id"] in seen_specs:
            raise ValueError(f"{location}: duplicate request or specification")
        request_ids.add(item["request_id"])
        seen_specs.add(item["spec_id"])
        expected_shard = shard_by_spec_id[item["spec_id"]]
        model = _exact_keys(
            item["model"],
            {"name", "revision", "identity_sha256"},
            location + ".model",
        )
        if (
            not isinstance(model["name"], str)
            or not model["name"].strip()
            or not isinstance(model["revision"], str)
            or not model["revision"].strip()
            or not HEX64.fullmatch(str(model["identity_sha256"]))
        ):
            raise ValueError(f"{location}: invalid model receipt")
        if model_identity is None:
            model_identity = model
        elif model != model_identity:
            raise ValueError(f"{location}: candidate model identity changed")
        generation = _exact_keys(
            item["generation"],
            {
                "seed", "candidate_seeds", "temperature", "top_p",
                "candidates", "max_new_tokens",
            },
            location + ".generation",
        )
        expected_seeds = [
            spec["generator"]["seed"] + candidate_index
            for candidate_index in range(
                spec["candidate_slot"]["candidates_requested"]
            )
        ]
        request_identity = {
            "model_identity_sha256": model["identity_sha256"],
            "spec_id": spec["spec_id"],
            "seed": spec["generator"]["seed"],
        }
        expected_request_id = (
            "replay-request-v1-" + canonical_sha256(request_identity)[:24]
        )
        if (
            item["schema"] != CANDIDATE_REQUEST_SCHEMA
            or item["request_id"] != expected_request_id
            or item["category"] != spec["category"]
            or item["source_group_id"] != spec["source_group_id"]
            or item["shard_index"] != expected_shard
            or item["messages"] != spec["messages"]
            or item["tools"] != spec["tools"]
            or item["template_policy"] != spec["template_policy"]
            or item["template_parameters"] != {
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            or item["assistant_mask_policy"] != spec["assistant_mask_policy"]
            or item["engine_policy"] != {
                "backend": "vllm",
                "version": "0.25.0",
                "dtype": "bfloat16",
                "tensor_parallel_size": 1,
                "max_num_seqs": 64,
                "enable_prefix_caching": False,
            }
            or generation != {
                "seed": spec["generator"]["seed"],
                "candidate_seeds": expected_seeds,
                "temperature": 0.3,
                "top_p": 0.9,
                "candidates": spec["candidate_slot"]["candidates_requested"],
                "max_new_tokens": spec["candidate_slot"]["max_assistant_tokens"],
            }
            or item["expected_response_format"]
            != spec["expected_response_format"]
            or item["prompt_identity_sha256"]
            != spec["prompt_identity_sha256"]
        ):
            raise ValueError(f"{location}: request lineage or policy changed")
    if seen_specs != set(spec_by_id):
        raise ValueError("candidate request coverage is incomplete")


def _text_from_message(message: dict) -> str:
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("candidate assistant message required")
    content = message.get("content", "")
    if not isinstance(content, str):
        raise ValueError("candidate assistant content must be text")
    return content.strip()


def _extract_python(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()


PYTHON_WORKER = r'''
import ast, builtins, json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
payload = json.loads(sys.stdin.read())
tree = ast.parse(payload["code"])
if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
    raise ValueError("one function definition required")
allowed = {
    "Module", "FunctionDef", "arguments", "arg", "Return", "Assign",
    "AugAssign", "For", "If", "IfExp", "Expr", "Name", "Load", "Store",
    "Constant", "List", "Tuple", "Dict", "Set", "BinOp", "UnaryOp",
    "BoolOp", "Compare", "Subscript", "Slice", "Call", "ListComp",
    "SetComp", "DictComp", "GeneratorExp", "comprehension", "Add", "Sub",
    "Mult", "FloorDiv", "Mod", "Pow", "USub", "UAdd", "And", "Or",
    "Not", "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "In", "NotIn",
}
safe_calls = {
    "abs", "all", "any", "dict", "enumerate", "len", "list", "max",
    "min", "range", "set", "sorted", "sum", "tuple", "zip",
}
for node in ast.walk(tree):
    if type(node).__name__ not in allowed:
        raise ValueError("forbidden syntax: " + type(node).__name__)
    if isinstance(node, ast.Name) and node.id.startswith("_"):
        raise ValueError("private names forbidden")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in safe_calls:
            raise ValueError("call not allowed")
function = tree.body[0]
if function.name != payload["function_name"]:
    raise ValueError("wrong function name")
safe_builtins = {name: getattr(builtins, name) for name in safe_calls}
namespace = {}
exec(compile(tree, "<candidate>", "exec"), {"__builtins__": safe_builtins}, namespace)
candidate = namespace[payload["function_name"]]
for case in payload["cases"]:
    if candidate(*case["args"]) != case["expected"]:
        raise ValueError("case failed")
print(json.dumps({"passed": True}))
'''


def verify_candidate(spec: dict, assistant_message: dict) -> dict:
    validate_prompt_specs([spec])
    verifier = spec["verifier"]
    verifier_type = verifier["type"]
    config = verifier["config"]
    if verifier_type == "approval_required_v1":
        return {
            "type": verifier_type,
            "status": "pending_approval",
            "passed": False,
            "reason": "subjective candidate requires an external approval ledger",
        }
    try:
        text = _text_from_message(assistant_message)
        if verifier_type == "exact_text_v1":
            passed = text == config["expected"]
        elif verifier_type == "normalized_exact_text_v1":
            passed = SPACE.sub(" ", text).strip() == SPACE.sub(
                " ", config["expected"]
            ).strip()
        elif verifier_type == "json_exact_v1":
            passed = json.loads(text) == config["expected"]
        elif verifier_type == "tool_call_exact_v1":
            passed = assistant_message.get("tool_calls") == config["expected"]
        elif verifier_type == "python_function_cases_v1":
            payload = {
                "code": _extract_python(text),
                "function_name": config["function_name"],
                "cases": config["cases"],
            }
            completed = subprocess.run(
                [sys.executable, "-I", "-S", "-c", PYTHON_WORKER],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=config["timeout_seconds"],
                check=False,
            )
            passed = completed.returncode == 0
        else:
            raise ValueError("unsupported verifier type")
    except (ValueError, TypeError, KeyError, json.JSONDecodeError,
            subprocess.TimeoutExpired):
        passed = False
    return {
        "type": verifier_type,
        "status": "passed" if passed else "failed",
        "passed": passed,
    }


def compile_objective_reference_message(spec: dict) -> dict:
    """Compile a trusted deterministic verifier target into an answer."""
    validate_prompt_specs([spec])
    verifier = spec["verifier"]
    verifier_type = verifier["type"]
    config = verifier["config"]
    if verifier_type == "approval_required_v1":
        raise ValueError("subjective specs cannot be reference-compiled")
    if verifier_type in {"exact_text_v1", "normalized_exact_text_v1"}:
        message = {"role": "assistant", "content": config["expected"]}
    elif verifier_type == "json_exact_v1":
        message = {
            "role": "assistant",
            "content": json.dumps(
                config["expected"],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
        }
    elif verifier_type == "tool_call_exact_v1":
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": config["expected"],
        }
    elif verifier_type == "python_function_cases_v1":
        code_lines = _extract_python(config["reference_answer"]).splitlines()
        if not code_lines or not code_lines[0].startswith("def "):
            raise RuntimeError("trusted Python reference has no function")
        code_lines.insert(1, '    """Return the requested result."""')
        message = {
            "role": "assistant",
            # Preserve the requested code-block-only form.  A concise function
            # docstring improves the supervised answer and keeps every valid
            # implementation above the sealed assistant-token floor.
            "content": (
                "```python\n"
                + "\n".join(code_lines).rstrip()
                + "\n```"
            ),
        }
    else:
        raise ValueError("unsupported deterministic reference verifier")
    verification = verify_candidate(spec, message)
    if verification["passed"] is not True:
        raise RuntimeError("trusted deterministic reference failed its verifier")
    return message


def build_reference_compiler_rows(
        specs: list[dict], tokenizer: Any,
) -> tuple[list[dict], dict]:
    """Build supervised rows from objective targets, never base generations."""
    validate_prompt_specs(specs)
    rows = []
    by_category = {
        category: {
            "specs": sum(item["category"] == category for item in specs),
            "subjective_gated_specs": 0,
            "compiled_rows": 0,
            "assistant_tokens": 0,
        }
        for category in CATEGORY_PERCENT
    }
    for spec in sorted(specs, key=lambda item: item["spec_id"]):
        category = spec["category"]
        if spec["verifier"]["type"] == "approval_required_v1":
            by_category[category]["subjective_gated_specs"] += 1
            continue
        message = compile_objective_reference_message(spec)
        messages = [*spec["messages"], message]
        count = assistant_token_count(tokenizer, messages, spec["tools"])
        slot = spec["candidate_slot"]
        if not slot["min_assistant_tokens"] <= count <= slot["max_assistant_tokens"]:
            raise RuntimeError(
                "compiled deterministic reference violates token bounds: "
                f"{spec['spec_id']} {category} count={count} "
                f"bounds={slot['min_assistant_tokens']}.."
                f"{slot['max_assistant_tokens']}"
            )
        compiler_identity = canonical_sha256({
            "compiler": REFERENCE_COMPILER_NAME,
            "compiler_version": REFERENCE_COMPILER_VERSION,
            "prompt_identity_sha256": spec["prompt_identity_sha256"],
            "assistant_message": message,
        })
        verification = verify_candidate(spec, message)
        rows.append({
            "schema": REPLAY_ROW_SCHEMA,
            "row_id": "replay-reference-v1-" + compiler_identity[:24],
            "category": category,
            "source_group_id": spec["source_group_id"],
            "messages": messages,
            "tools": spec["tools"],
            "assistant_mask": {
                "policy": "assistant_only_v1",
                "assistant_message_indices": [len(messages) - 1],
                "system_tokens": False,
                "user_tokens": False,
                "tool_result_tokens": False,
            },
            "assistant_token_count": count,
            "lineage": {
                "source_kind": REFERENCE_COMPILER_NAME,
                "spec_id": spec["spec_id"],
                "prompt_identity_sha256": spec["prompt_identity_sha256"],
                "compiler_identity_sha256": compiler_identity,
                "direct_benchmark_prompt": False,
            },
            "generator": {
                "name": REFERENCE_COMPILER_NAME,
                "version": REFERENCE_COMPILER_VERSION,
                "seed": spec["generator"]["seed"],
            },
            "verifier": {
                "type": verification["type"],
                "status": "passed",
            },
            "template_policy": "official_qwen_apply_chat_template_v1",
        })
        by_category[category]["compiled_rows"] += 1
        by_category[category]["assistant_tokens"] += count
    if len({row["row_id"] for row in rows}) != len(rows):
        raise RuntimeError("compiled deterministic reference identity duplicated")
    return rows, {
        "specs": len(specs),
        "compiled_rows": len(rows),
        "assistant_tokens": sum(
            row["assistant_token_count"] for row in rows
        ),
        "subjective_gated_specs": sum(
            item["subjective_gated_specs"] for item in by_category.values()
        ),
        "by_category": by_category,
    }


def validate_seed_authority() -> tuple[list[dict], dict]:
    seed_path = safe_regular_input(SAFE_SEED, "safe seed", exact=SAFE_SEED)
    report_path = safe_regular_input(
        SAFE_SEED_REPORT, "safe seed report", exact=SAFE_SEED_REPORT
    )
    if file_sha256(seed_path) != SAFE_SEED_SHA256:
        raise RuntimeError("safe replay seed identity changed")
    if file_sha256(report_path) != SAFE_SEED_REPORT_SHA256:
        raise RuntimeError("safe replay seed report identity changed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != "general-qa-proxy-anchor-build-v43h"
        or report.get("output_sha256") != SAFE_SEED_SHA256
        or report.get("rows") != 128
        or report.get("direct_benchmark_source", {}).get("opened") is not False
        or report.get("direct_benchmark_source", {}).get(
            "authorized_for_qa_semantics") is not False
        or report.get("policy", {}).get(
            "protected_eval_ood_heldout_or_benchmark_semantics_opened") is not False
    ):
        raise RuntimeError("safe replay seed authority report changed")
    rows = []
    seen_items = set()
    seen_groups = set()
    with seed_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            required = {
                "answer", "answer_sha256", "document_id", "instruction",
                "instruction_sha256", "item_id", "parent_item_id",
                "parent_text_sha256", "quality_bucket", "split",
            }
            if set(item) != required:
                raise RuntimeError(f"safe seed row {line_number} schema changed")
            if (
                item["split"] != "anchor_general_qa_proxy"
                or item["quality_bucket"]
                != "train_only_general_knowledge_instruction_proxy"
                or hashlib.sha256(item["answer"].encode()).hexdigest()
                != item["answer_sha256"]
                or hashlib.sha256(item["instruction"].encode()).hexdigest()
                != item["instruction_sha256"]
            ):
                raise RuntimeError(f"safe seed row {line_number} invalid")
            group = "proxy-group-v43h-" + hashlib.sha256(
                item["document_id"].encode("utf-8")
            ).hexdigest()[:24]
            if item["item_id"] in seen_items or group in seen_groups:
                raise RuntimeError("safe seed identity duplicated")
            seen_items.add(item["item_id"])
            seen_groups.add(group)
            rows.append({
                "schema": REPLAY_ROW_SCHEMA,
                "row_id": "replay-seed-v1-" + hashlib.sha256(
                    item["item_id"].encode("utf-8")
                ).hexdigest()[:24],
                "category": "instruction_following",
                "source_group_id": group,
                "messages": [
                    {"role": "user", "content": item["instruction"]},
                    {"role": "assistant", "content": item["answer"]},
                ],
                "tools": [],
                "assistant_mask": {
                    "policy": "assistant_only_v1",
                    "assistant_message_indices": [1],
                    "system_tokens": False,
                    "user_tokens": False,
                    "tool_result_tokens": False,
                },
                "lineage": {
                    "source_kind": "approved_proxy_anchor_v43h",
                    "source_artifact_sha256": SAFE_SEED_SHA256,
                    "source_report_sha256": SAFE_SEED_REPORT_SHA256,
                    "parent_item_id": item["item_id"],
                    "parent_text_sha256": item["parent_text_sha256"],
                    "direct_benchmark_prompt": False,
                },
                "generator": {
                    "name": "approved_proxy_anchor_v43h",
                    "version": "v43h",
                    "seed": None,
                },
                "verifier": {
                    "type": "pinned_seed_hash_v1",
                    "status": "passed",
                },
                "template_policy": "official_qwen_apply_chat_template_v1",
            })
    if len(rows) != 128:
        raise RuntimeError("safe replay seed must contain 128 rows")
    return rows, report


def candidate_response_sha256(response: dict) -> str:
    payload = dict(response)
    payload.pop("response_sha256", None)
    return canonical_sha256(payload)


def validate_candidate_responses(
        responses: list[dict], requests: list[dict],
        *, require_complete: bool = True) -> None:
    request_by_id = {item["request_id"]: item for item in requests}
    if len(request_by_id) != len(requests):
        raise ValueError("candidate request identity is duplicated")
    seen = set()
    expected_keys = {
        "assistant_message", "candidate_index", "finish_reason", "generator",
        "request_id", "response_sha256", "schema", "shard_index", "spec_id",
    }
    for index, raw in enumerate(responses):
        location = f"candidate response {index}"
        item = _exact_keys(raw, expected_keys, location)
        if item["schema"] != CANDIDATE_RESPONSE_SCHEMA:
            raise ValueError(f"{location}: unsupported schema")
        request = request_by_id.get(item["request_id"])
        if request is None:
            raise ValueError(f"{location}: unknown request")
        if (
            item["spec_id"] != request["spec_id"]
            or item["shard_index"] != request["shard_index"]
        ):
            raise ValueError(f"{location}: request lineage mismatch")
        candidate_index = item["candidate_index"]
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or not 0 <= candidate_index < request["generation"]["candidates"]
        ):
            raise ValueError(f"{location}: invalid candidate index")
        identity = (item["request_id"], candidate_index)
        if identity in seen:
            raise ValueError(f"{location}: duplicate candidate slot")
        seen.add(identity)
        generator = _exact_keys(item["generator"], {
            "identity_sha256", "name", "revision", "seed",
        }, location + ".generator")
        if generator != {
            "name": request["model"]["name"],
            "revision": request["model"]["revision"],
            "identity_sha256": request["model"]["identity_sha256"],
            "seed": request["generation"]["candidate_seeds"][candidate_index],
        }:
            raise ValueError(f"{location}: generator lineage mismatch")
        message = item["assistant_message"]
        if not isinstance(message, dict) or set(message) not in (
                {"role", "content"}, {"role", "content", "tool_calls"}):
            raise ValueError(f"{location}: malformed assistant message")
        if message.get("role") != "assistant" or not isinstance(
                message.get("content"), str):
            raise ValueError(f"{location}: malformed assistant message")
        if "tool_calls" in message and not isinstance(message["tool_calls"], list):
            raise ValueError(f"{location}: malformed tool calls")
        if item["finish_reason"] not in {"length", "stop", "tool_calls"}:
            raise ValueError(f"{location}: unacceptable finish reason")
        if not HEX64.fullmatch(item["response_sha256"]):
            raise ValueError(f"{location}: response digest malformed")
        if candidate_response_sha256(item) != item["response_sha256"]:
            raise ValueError(f"{location}: stale response digest")
    if require_complete:
        expected = {
            (request["request_id"], candidate_index)
            for request in requests
            for candidate_index in range(request["generation"]["candidates"])
        }
        if seen != expected:
            raise ValueError("candidate response coverage is incomplete")


def validate_approval_ledger(approvals: list[dict]) -> dict[str, dict]:
    result = {}
    expected = {
        "reason", "response_sha256", "reviewed_at", "reviewer", "rubric_id",
        "schema", "spec_id", "status",
    }
    for index, raw in enumerate(approvals):
        item = _exact_keys(raw, expected, f"approval {index}")
        if item["schema"] != "general-replay-candidate-approval-v1":
            raise ValueError(f"approval {index}: unsupported schema")
        if item["status"] not in {"approved", "rejected"}:
            raise ValueError(f"approval {index}: invalid status")
        if not HEX64.fullmatch(item["response_sha256"]):
            raise ValueError(f"approval {index}: response digest malformed")
        if item["response_sha256"] in result:
            raise ValueError(f"approval {index}: duplicate response approval")
        for field in ("reason", "reviewed_at", "reviewer", "rubric_id", "spec_id"):
            if not isinstance(item[field], str) or not item[field].strip():
                raise ValueError(f"approval {index}: {field} required")
        result[item["response_sha256"]] = item
    return result


def build_verified_candidate_rows(
        specs: list[dict], requests: list[dict], responses: list[dict],
        approvals: list[dict], tokenizer: Any,
) -> tuple[list[dict], dict]:
    validate_prompt_specs(specs)
    validate_candidate_requests(specs, requests)
    validate_candidate_responses(responses, requests)
    spec_by_id = {item["spec_id"]: item for item in specs}
    request_by_id = {item["request_id"]: item for item in requests}
    approvals_by_response = validate_approval_ledger(approvals)
    responses_by_spec: dict[str, list[dict]] = {}
    for response in responses:
        responses_by_spec.setdefault(response["spec_id"], []).append(response)

    rows = []
    audit = {
        "specs": len(specs),
        "responses": len(responses),
        "objective_failures": 0,
        "approval_gated": 0,
        "length_truncated": 0,
        "token_bound_failures": 0,
        "verified_rows": 0,
    }
    seen_pairs = set()
    for spec_id in sorted(spec_by_id):
        spec = spec_by_id[spec_id]
        accepted = None
        accepted_verification = None
        for response in sorted(
                responses_by_spec.get(spec_id, []),
                key=lambda item: (item["candidate_index"], item["response_sha256"])):
            request = request_by_id[response["request_id"]]
            if request["spec_id"] != spec_id:
                raise ValueError("candidate response/spec mismatch")
            if response["finish_reason"] == "length":
                audit["length_truncated"] += 1
                continue
            verification = verify_candidate(spec, response["assistant_message"])
            if spec["verifier"]["type"] == "approval_required_v1":
                approval = approvals_by_response.get(response["response_sha256"])
                rubric_id = spec["verifier"]["config"]["rubric_id"]
                if (
                    approval is None
                    or approval["status"] != "approved"
                    or approval["spec_id"] != spec_id
                    or approval["rubric_id"] != rubric_id
                ):
                    audit["approval_gated"] += 1
                    continue
                verification = {
                    "type": "approval_required_v1",
                    "status": "passed",
                    "passed": True,
                    "approval_response_sha256": response["response_sha256"],
                    "rubric_id": rubric_id,
                }
            elif not verification["passed"]:
                audit["objective_failures"] += 1
                continue
            messages = [*spec["messages"], response["assistant_message"]]
            count = assistant_token_count(tokenizer, messages, spec["tools"])
            slot = spec["candidate_slot"]
            if not slot["min_assistant_tokens"] <= count <= slot["max_assistant_tokens"]:
                audit["token_bound_failures"] += 1
                continue
            accepted = response
            accepted_verification = verification
            break
        if accepted is None:
            continue
        pair_identity = canonical_sha256({
            "prompt": spec["prompt_identity_sha256"],
            "response": accepted["response_sha256"],
        })
        if pair_identity in seen_pairs:
            raise RuntimeError("verified replay prompt/response duplicated")
        seen_pairs.add(pair_identity)
        request = request_by_id[accepted["request_id"]]
        messages = [*spec["messages"], accepted["assistant_message"]]
        count = assistant_token_count(tokenizer, messages, spec["tools"])
        rows.append({
            "schema": REPLAY_ROW_SCHEMA,
            "row_id": "replay-generated-v1-" + pair_identity[:24],
            "category": spec["category"],
            "source_group_id": spec["source_group_id"],
            "messages": messages,
            "tools": spec["tools"],
            "assistant_mask": {
                "policy": "assistant_only_v1",
                "assistant_message_indices": [len(messages) - 1],
                "system_tokens": False,
                "user_tokens": False,
                "tool_result_tokens": False,
            },
            "assistant_token_count": count,
            "lineage": {
                "source_kind": "verified_base_model_generation_v1",
                "spec_id": spec_id,
                "request_id": accepted["request_id"],
                "request_shard": request["shard_index"],
                "response_sha256": accepted["response_sha256"],
                "prompt_identity_sha256": spec["prompt_identity_sha256"],
                "direct_benchmark_prompt": False,
            },
            "generator": accepted["generator"],
            "verifier": accepted_verification,
            "template_policy": "official_qwen_apply_chat_template_v1",
        })
    audit["verified_rows"] = len(rows)
    return rows, audit


def build_all_verified_candidate_rows(
        specs: list[dict], requests: list[dict], responses: list[dict],
        approvals: list[dict], tokenizer: Any,
) -> tuple[list[dict], dict]:
    """Verify every candidate alternative without selecting a source group.

    The corpus selector, rather than this verifier, decides which one of the
    passing alternatives for a source group can enter the final corpus.  This
    preserves alternative token lengths for exact-budget selection while the
    group-aware selector still forbids selecting two responses to one prompt.
    """
    validate_prompt_specs(specs)
    validate_candidate_requests(specs, requests)
    validate_candidate_responses(responses, requests)
    spec_by_id = {item["spec_id"]: item for item in specs}
    request_by_id = {item["request_id"]: item for item in requests}
    approvals_by_response = validate_approval_ledger(approvals)
    category_audit = {
        category: {
            "specs": sum(item["category"] == category for item in specs),
            "responses": 0,
            "length_truncated": 0,
            "token_bound_failures": 0,
            "objective_passed": 0,
            "objective_failed": 0,
            "approval_approved": 0,
            "approval_gated": 0,
            "approval_rejected": 0,
            "eligible_candidate_rows": 0,
            "eligible_source_groups": 0,
        }
        for category in CATEGORY_PERCENT
    }
    eligible_groups = {category: set() for category in CATEGORY_PERCENT}
    rows = []
    seen_pairs = set()
    ordered_responses = sorted(
        responses,
        key=lambda item: (
            item["spec_id"], item["candidate_index"],
            item["response_sha256"],
        ),
    )
    for response in ordered_responses:
        spec = spec_by_id[response["spec_id"]]
        request = request_by_id[response["request_id"]]
        if request["spec_id"] != spec["spec_id"]:
            raise ValueError("candidate response/spec mismatch")
        category = spec["category"]
        stats = category_audit[category]
        stats["responses"] += 1
        if response["finish_reason"] == "length":
            stats["length_truncated"] += 1
            continue
        messages = [*spec["messages"], response["assistant_message"]]
        count = assistant_token_count(tokenizer, messages, spec["tools"])
        slot = spec["candidate_slot"]
        if not slot["min_assistant_tokens"] <= count <= slot["max_assistant_tokens"]:
            stats["token_bound_failures"] += 1
            continue

        verifier_type = spec["verifier"]["type"]
        if verifier_type == "approval_required_v1":
            approval = approvals_by_response.get(response["response_sha256"])
            rubric_id = spec["verifier"]["config"]["rubric_id"]
            if approval is not None and approval["status"] == "rejected":
                stats["approval_rejected"] += 1
                continue
            if (
                approval is None
                or approval["status"] != "approved"
                or approval["spec_id"] != spec["spec_id"]
                or approval["rubric_id"] != rubric_id
            ):
                stats["approval_gated"] += 1
                continue
            verification = {
                "type": verifier_type,
                "status": "passed",
                "passed": True,
                "approval_response_sha256": response["response_sha256"],
                "rubric_id": rubric_id,
            }
            stats["approval_approved"] += 1
        else:
            verification = verify_candidate(spec, response["assistant_message"])
            if not verification["passed"]:
                stats["objective_failed"] += 1
                continue
            stats["objective_passed"] += 1

        pair_identity = canonical_sha256({
            "prompt": spec["prompt_identity_sha256"],
            "response": response["response_sha256"],
        })
        if pair_identity in seen_pairs:
            raise RuntimeError("verified replay prompt/response duplicated")
        seen_pairs.add(pair_identity)
        rows.append({
            "schema": REPLAY_ROW_SCHEMA,
            "row_id": "replay-generated-v1-" + pair_identity[:24],
            "category": category,
            "source_group_id": spec["source_group_id"],
            "messages": messages,
            "tools": spec["tools"],
            "assistant_mask": {
                "policy": "assistant_only_v1",
                "assistant_message_indices": [len(messages) - 1],
                "system_tokens": False,
                "user_tokens": False,
                "tool_result_tokens": False,
            },
            "assistant_token_count": count,
            "lineage": {
                "source_kind": "verified_base_model_generation_v1",
                "spec_id": spec["spec_id"],
                "request_id": response["request_id"],
                "request_shard": request["shard_index"],
                "response_sha256": response["response_sha256"],
                "prompt_identity_sha256": spec["prompt_identity_sha256"],
                "direct_benchmark_prompt": False,
            },
            "generator": response["generator"],
            "verifier": verification,
            "template_policy": "official_qwen_apply_chat_template_v1",
        })
        stats["eligible_candidate_rows"] += 1
        eligible_groups[category].add(spec["source_group_id"])

    for category, groups in eligible_groups.items():
        category_audit[category]["eligible_source_groups"] = len(groups)
    audit = {
        "specs": len(specs),
        "responses": len(responses),
        "length_truncated": sum(
            item["length_truncated"] for item in category_audit.values()
        ),
        "token_bound_failures": sum(
            item["token_bound_failures"] for item in category_audit.values()
        ),
        "objective_passed": sum(
            item["objective_passed"] for item in category_audit.values()
        ),
        "objective_failed": sum(
            item["objective_failed"] for item in category_audit.values()
        ),
        "approval_approved": sum(
            item["approval_approved"] for item in category_audit.values()
        ),
        "approval_gated": sum(
            item["approval_gated"] for item in category_audit.values()
        ),
        "approval_rejected": sum(
            item["approval_rejected"] for item in category_audit.values()
        ),
        "eligible_candidate_rows": len(rows),
        "eligible_source_groups": sum(
            len(groups) for groups in eligible_groups.values()
        ),
        "by_category": category_audit,
    }
    return rows, audit


def assistant_token_mask(
        tokenizer: Any, messages: list[dict], tools: list[dict]) -> list[int]:
    encoded = encode_chat_assistant_only(
        tokenizer,
        messages,
        enable_thinking=False,
        tools=tools,
    )
    return encoded["assistant_mask"]


def assistant_token_count(tokenizer: Any, messages: list[dict], tools: list[dict]) -> int:
    return sum(assistant_token_mask(tokenizer, messages, tools))


def exact_token_subset(rows: list[dict], target: int) -> list[dict]:
    """Select a deterministic exact-token subset without padding or repeats."""
    if target < 0:
        raise ValueError("token target cannot be negative")
    row_ids = [row.get("row_id") for row in rows]
    if any(not isinstance(row_id, str) or not row_id for row_id in row_ids):
        raise ValueError("candidate row identity is required")
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("candidate row identities must be unique")
    ordered = sorted(rows, key=lambda item: item["row_id"])
    reachable: dict[int, tuple[int, int] | None] = {0: None}
    for index, row in enumerate(ordered):
        count = row.get("assistant_token_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("candidate row lacks a positive assistant token count")
        for subtotal in sorted(list(reachable), reverse=True):
            updated = subtotal + count
            if updated <= target and updated not in reachable:
                reachable[updated] = (subtotal, index)
        if target in reachable:
            break
    if target not in reachable:
        raise RuntimeError(
            "verified unique rows cannot satisfy the exact assistant-token budget"
        )
    chosen = []
    subtotal = target
    while subtotal:
        predecessor = reachable[subtotal]
        if predecessor is None:
            raise AssertionError("exact-token subset predecessor missing")
        subtotal, index = predecessor
        chosen.append(ordered[index])
    chosen.reverse()
    return chosen


def exact_token_group_subset(rows: list[dict], target: int) -> list[dict]:
    """Select an exact subset with at most one row per source group."""
    if target < 0:
        raise ValueError("token target cannot be negative")
    row_ids = [row.get("row_id") for row in rows]
    if any(not isinstance(row_id, str) or not row_id for row_id in row_ids):
        raise ValueError("candidate row identity is required")
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("candidate row identities must be unique")
    groups: dict[str, list[dict]] = {}
    for row in rows:
        group = row.get("source_group_id")
        if not isinstance(group, str) or not group:
            raise ValueError("candidate source-group identity is required")
        count = row.get("assistant_token_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(
                "candidate row lacks a positive assistant token count"
            )
        groups.setdefault(group, []).append(row)

    # Each reachable total is installed only once, so its linked predecessor
    # cannot include the current group: additions use the pre-group snapshot.
    reachable: dict[int, tuple[object, dict] | None] = {0: None}
    for group in sorted(groups):
        before = list(sorted(reachable.items()))
        for subtotal, predecessor in before:
            for row in sorted(groups[group], key=lambda item: item["row_id"]):
                updated = subtotal + row["assistant_token_count"]
                if updated <= target and updated not in reachable:
                    reachable[updated] = (predecessor, row)
        if target in reachable:
            break
    if target not in reachable:
        raise RuntimeError(
            "verified unique source groups cannot satisfy the exact "
            "assistant-token budget"
        )
    chosen = []
    predecessor = reachable[target]
    while predecessor is not None:
        predecessor, row = predecessor
        chosen.append(row)
    chosen.reverse()
    return chosen


def build_final_replay_rows(
        seed_rows: list[dict], candidate_rows: list[dict], tokenizer: Any,
        *, total_assistant_tokens: int,
) -> tuple[list[dict], dict]:
    budgets = token_budgets(total_assistant_tokens)
    mandatory = []
    seen_rows = set()
    seen_groups = set()
    for source in seed_rows:
        row = dict(source)
        row["assistant_token_count"] = assistant_token_count(
            tokenizer, row["messages"], row["tools"]
        )
        if row["row_id"] in seen_rows or row["source_group_id"] in seen_groups:
            raise RuntimeError("mandatory replay seed duplicated")
        seen_rows.add(row["row_id"])
        seen_groups.add(row["source_group_id"])
        mandatory.append(row)
    mandatory_instruction_tokens = sum(
        row["assistant_token_count"] for row in mandatory
        if row["category"] == "instruction_following"
    )
    if mandatory_instruction_tokens > budgets["instruction_following"]:
        raise RuntimeError("mandatory replay seed exceeds instruction token budget")

    selected = list(mandatory)
    for category, target in budgets.items():
        already = sum(
            row["assistant_token_count"] for row in selected
            if row["category"] == category
        )
        remaining = target - already
        pool = [row for row in candidate_rows if row["category"] == category]
        chosen = exact_token_group_subset(pool, remaining)
        for row in chosen:
            if row["row_id"] in seen_rows or row["source_group_id"] in seen_groups:
                raise RuntimeError("selected replay identity duplicated")
            seen_rows.add(row["row_id"])
            seen_groups.add(row["source_group_id"])
        selected.extend(chosen)
    selected.sort(key=lambda item: (item["category"], item["row_id"]))
    realized = {
        category: sum(
            row["assistant_token_count"] for row in selected
            if row["category"] == category
        )
        for category in CATEGORY_PERCENT
    }
    if realized != budgets or sum(realized.values()) != total_assistant_tokens:
        raise AssertionError("final replay token proportions are not exact")
    return selected, {
        "target_total_assistant_tokens": total_assistant_tokens,
        "assistant_tokens_by_category": realized,
        "category_percent": dict(CATEGORY_PERCENT),
        "mandatory_seed_rows": len(mandatory),
        "selected_generated_rows": len(selected) - len(mandatory),
        "rows": len(selected),
        "duplicate_or_padding_fill_used": False,
    }
