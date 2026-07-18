#!/usr/bin/env python3
"""Build exact-token objective supplements for the 150k replay target.

The supplement is deliberately narrow: it creates only deterministic synthetic
coding, math, JSON, instruction-following, and exact-translation prompts.  It
never opens a benchmark, protected, evaluation, holdout, OOD, incident, or
manual-review source.  Every emitted answer is compiled from its verifier
target by the shared deterministic reference compiler and counted with the
official Qwen assistant-only mask.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from build_general_replay_corpus_v1 import load_qwen_tokenizer
from general_replay_v1 import (
    BUILDER_NAME,
    BUILDER_VERSION,
    PROMPT_SPEC_SCHEMA,
    ROOT,
    _prompt_identity,
    _seed_for,
    assistant_token_count,
    build_reference_compiler_rows,
    canonical_bytes,
    canonical_sha256,
    exact_token_group_subset,
    file_sha256,
    safe_regular_input,
    token_budgets,
    validate_prompt_specs,
    validate_seed_authority,
)
from prepare_general_replay_candidate_shards_v1 import load_prompt_artifacts
from run_general_replay_candidate_shard_v1 import (
    MODEL_DIRECTORY,
    MODEL_FILE_SHA256,
    MODEL_REVISION,
)


DIRECTORY = ROOT / "data/general_replay_v1"
DEFAULT_BASE_SPECS = DIRECTORY / "prompt_specs_v2_scale32.jsonl"
DEFAULT_BASE_SPEC_REPORT = DIRECTORY / "prompt_specs_v2_scale32.report.json"
DEFAULT_BASE_ROWS = DIRECTORY / "deterministic_reference_rows_v1_scale32.jsonl"
DEFAULT_BASE_ROW_REPORT = (
    DIRECTORY / "deterministic_reference_rows_v1_scale32.report.json"
)
DEFAULT_SPEC_OUTPUT = (
    DIRECTORY / "objective_supplement_prompt_specs_v1_150k.jsonl"
)
DEFAULT_ROW_OUTPUT = (
    DIRECTORY / "objective_supplement_reference_rows_v1_150k.jsonl"
)
DEFAULT_REPORT = DIRECTORY / "objective_supplement_v1_150k.report.json"
DEFAULT_TARGET = 150_000
SUPPLEMENT_BUILD_SEED = 2026071801
SUPPLEMENT_ORDINAL_BASE = 100_000

SCOPED_CATEGORIES = (
    "coding_debugging",
    "mathematical_reasoning",
    "json_structured_data",
    "instruction_following",
    "multilingual",
)

# These pools provide substantive headroom.  Exact token totals are selected
# only after every trusted answer has passed its verifier and official mask.
POOL_COUNTS = {
    "coding_debugging": 160,
    "mathematical_reasoning": 1_024,
    "json_structured_data": 128,
    "instruction_following": 768,
    "multilingual": 1_600,
}


def _messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


def _make_spec(
    *, category: str, ordinal: int, messages: list[dict],
    verifier: dict, response_format: str, bounds: tuple[int, int],
    build_seed: int = SUPPLEMENT_BUILD_SEED,
) -> dict:
    """Create a shared-schema synthetic spec with sealed deterministic lineage."""
    tools: list[dict] = []
    seed = _seed_for(category, ordinal, build_seed)
    prompt_identity = _prompt_identity(messages, tools)
    identity = {
        "build_seed": build_seed,
        "category": category,
        "ordinal": ordinal,
        "prompt_identity_sha256": prompt_identity,
        "seed": seed,
    }
    return {
        "schema": PROMPT_SPEC_SCHEMA,
        "spec_id": "replay-spec-v1-" + canonical_sha256(identity)[:24],
        "category": category,
        "source_group_id": (
            f"synthetic-replay-v1:{category}:{ordinal:06d}"
        ),
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
    }


def _coding_spec(index: int) -> dict:
    ordinal = SUPPLEMENT_ORDINAL_BASE + index
    family = index % 6
    variant = index // 6
    name = f"supplement_{family}_{variant}"
    if family == 0:
        offset = variant % 17 - 8
        prompt = (
            f"Implement `{name}` in Python. Return a new list formed by adding "
            f"{offset} to every integer in `values`, preserving order. Return "
            "one Python code block containing only the function; use no imports "
            "or global state."
        )
        cases = [
            {"args": [[-4, 0, 9]], "expected": [-4 + offset, offset, 9 + offset]},
            {"args": [[]], "expected": []},
        ]
        code = f"def {name}(values):\n    return [value + {offset} for value in values]"
    elif family == 1:
        threshold = variant % 29 - 14
        prompt = (
            f"Implement `{name}` in Python. Count the integers in `values` that "
            f"are strictly greater than {threshold}. Return one Python code block "
            "containing only the function; use no imports or global state."
        )
        values = [threshold - 2, threshold, threshold + 1, threshold + 7]
        cases = [
            {"args": [values], "expected": 2},
            {"args": [[]], "expected": 0},
        ]
        code = (
            f"def {name}(values):\n"
            f"    return sum(value > {threshold} for value in values)"
        )
    elif family == 2:
        divisor = 2 + variant % 11
        prompt = (
            f"Implement `{name}` in Python. Return, in original order, every "
            f"integer in `values` divisible by {divisor}. Return one Python code "
            "block containing only the function; use no imports or global state."
        )
        values = [divisor, divisor + 1, -2 * divisor, 3 * divisor]
        cases = [
            {"args": [values], "expected": [divisor, -2 * divisor, 3 * divisor]},
            {"args": [[]], "expected": []},
        ]
        code = (
            f"def {name}(values):\n"
            f"    return [value for value in values if value % {divisor} == 0]"
        )
    elif family == 3:
        prompt = (
            f"Implement `{name}` in Python. Return the running total after each "
            "integer in `values`; preserve order and return an empty list for an "
            "empty input. Return one Python code block containing only the "
            "function; use no imports or global state."
        )
        cases = [
            {"args": [[3, -1, 4]], "expected": [3, 2, 6]},
            {"args": [[]], "expected": []},
        ]
        code = (
            f"def {name}(values):\n"
            "    total = 0\n"
            "    result = []\n"
            "    for value in values:\n"
            "        total += value\n"
            "        result = result + [total]\n"
            "    return result"
        )
    elif family == 4:
        shift = 1 + variant % 7
        prompt = (
            f"Implement `{name}` in Python. Rotate nonempty `values` left by "
            f"{shift} positions, treating larger shifts cyclically; return an "
            "empty list for an empty input. Return one Python code block "
            "containing only the function; use no imports or global state."
        )
        values = [10, 20, 30, 40, 50]
        effective = shift % len(values)
        cases = [
            {"args": [values], "expected": values[effective:] + values[:effective]},
            {"args": [[]], "expected": []},
        ]
        code = (
            f"def {name}(values):\n"
            "    if not values:\n"
            "        return []\n"
            f"    shift = {shift} % len(values)\n"
            "    return values[shift:] + values[:shift]"
        )
    else:
        prompt = (
            f"Implement `{name}` in Python. Return the first occurrence of each "
            "integer in `values`, preserving order. Return one Python code block "
            "containing only the function; use no imports or global state."
        )
        cases = [
            {"args": [[4, 2, 4, 3, 2]], "expected": [4, 2, 3]},
            {"args": [[]], "expected": []},
        ]
        code = (
            f"def {name}(values):\n"
            "    result = []\n"
            "    for value in values:\n"
            "        if value not in result:\n"
            "            result = result + [value]\n"
            "    return result"
        )
    return _make_spec(
        category="coding_debugging",
        ordinal=ordinal,
        messages=_messages(prompt),
        response_format="assistant_text",
        verifier={
            "type": "python_function_cases_v1",
            "status": "ready",
            "config": {
                "function_name": name,
                "cases": cases,
                "reference_answer": f"```python\n{code}\n```",
                "timeout_seconds": 2,
            },
        },
        bounds=(1, 384),
    )


def _math_spec(index: int) -> dict:
    ordinal = SUPPLEMENT_ORDINAL_BASE + POOL_COUNTS["coding_debugging"] + index
    coefficient = 2 + index % 17
    solution = index - 600
    offset = (index * 19) % 2_003 - 1_001
    result = coefficient * solution + offset
    prompt = (
        f"Solve {coefficient}x + ({offset}) = {result}. Return only a JSON "
        "object with integer keys `x` and `check`, where `check` is the left "
        "side evaluated at your solution."
    )
    return _make_spec(
        category="mathematical_reasoning",
        ordinal=ordinal,
        messages=_messages(prompt),
        response_format="assistant_json",
        verifier={
            "type": "json_exact_v1",
            "status": "ready",
            "config": {"expected": {"check": result, "x": solution}},
        },
        bounds=(1, 128),
    )


def _json_spec(index: int) -> dict:
    ordinal = (
        SUPPLEMENT_ORDINAL_BASE
        + POOL_COUNTS["coding_debugging"]
        + POOL_COUNTS["mathematical_reasoning"]
        + index
    )
    labels = ("amber", "blue", "cedar", "delta", "ember", "fern")
    length = 2 + index % 7
    records = [
        {
            "label": labels[(index + position) % len(labels)],
            "value": 1 + ((index + 3) * (position + 5)) % 97,
        }
        for position in range(length)
    ]
    expected = {
        "count": length,
        "labels": sorted({record["label"] for record in records}),
        "total": sum(record["value"] for record in records),
    }
    prompt = (
        "Read the records below. Return only a JSON object with `count` equal "
        "to the number of records, `labels` containing the distinct labels in "
        "alphabetical order, and `total` equal to the sum of all values.\n"
        + json.dumps(records, ensure_ascii=False, sort_keys=True)
    )
    return _make_spec(
        category="json_structured_data",
        ordinal=ordinal,
        messages=_messages(prompt),
        response_format="assistant_json",
        verifier={
            "type": "json_exact_v1",
            "status": "ready",
            "config": {"expected": expected},
        },
        bounds=(1, 192),
    )


def _instruction_spec(index: int) -> dict:
    ordinal = (
        SUPPLEMENT_ORDINAL_BASE
        + POOL_COUNTS["coding_debugging"]
        + POOL_COUNTS["mathematical_reasoning"]
        + POOL_COUNTS["json_structured_data"]
        + index
    )
    family = index % 4
    number = 100 + index
    if family == 0:
        expected = f"ALPHA: {number}\nBETA: {number + 7}"
        prompt = (
            "Return exactly two lines and no Markdown. The first must be "
            f"`ALPHA: {number}` and the second must be `BETA: {number + 7}`."
        )
    elif family == 1:
        expected = f"status=ready; batch={number}; retries=0"
        prompt = (
            "Return exactly this single status line, with no surrounding text "
            f"or code fence: `status=ready; batch={number}; retries=0`"
        )
    elif family == 2:
        expected = f"READY: {number}"
        prompt = (
            "Return exactly one line and no Markdown. The line must be "
            f"`READY: {number}`."
        )
    else:
        expected = (
            f"ITEM {number}\nOWNER operations\nSTATE queued\nPRIORITY normal"
        )
        prompt = (
            "Return exactly four lines and no Markdown: "
            f"`ITEM {number}`, then `OWNER operations`, then `STATE queued`, "
            "then `PRIORITY normal`."
        )
    return _make_spec(
        category="instruction_following",
        ordinal=ordinal,
        messages=_messages(prompt),
        response_format="assistant_text",
        verifier={
            "type": "exact_text_v1",
            "status": "ready",
            "config": {"expected": expected},
        },
        bounds=(1, 128),
    )


# Every entry below is an explicitly authored source/translation pair.  The
# opaque code and room slots are copied verbatim, so deterministic expansion
# cannot introduce a translation judgment or an unreviewed lexical choice.
TRANSLATION_TEMPLATES = (
    ("Spanish", "Write code {code} on the blue card.", "Escribe el código {code} en la tarjeta azul."),
    ("Spanish", "Package {code} is beside the window.", "El paquete {code} está junto a la ventana."),
    ("Spanish", "Please take folder {code} to room {room}.", "Por favor, lleva la carpeta {code} a la sala {room}."),
    ("French", "Write code {code} on the blue card.", "Écrivez le code {code} sur la carte bleue."),
    ("French", "Package {code} is beside the window.", "Le colis {code} est à côté de la fenêtre."),
    ("French", "Please take folder {code} to room {room}.", "Veuillez apporter le dossier {code} à la salle {room}."),
    ("German", "Write code {code} on the blue card.", "Schreiben Sie den Code {code} auf die blaue Karte."),
    ("German", "Package {code} is beside the window.", "Das Paket {code} liegt neben dem Fenster."),
    ("German", "Please take folder {code} to room {room}.", "Bitte bringen Sie den Ordner {code} in Raum {room}."),
    ("Portuguese", "Write code {code} on the blue card.", "Escreva o código {code} no cartão azul."),
    ("Portuguese", "Package {code} is beside the window.", "O pacote {code} está ao lado da janela."),
    ("Portuguese", "Please take folder {code} to room {room}.", "Por favor, leve a pasta {code} para a sala {room}."),
    ("Italian", "Write code {code} on the blue card.", "Scriva il codice {code} sulla scheda blu."),
    ("Italian", "Package {code} is beside the window.", "Il pacco {code} è accanto alla finestra."),
    ("Italian", "Please take folder {code} to room {room}.", "Per favore, porti la cartella {code} nella stanza {room}."),
    ("Japanese", "Write code {code} on the blue card.", "青いカードにコード{code}を書いてください。"),
    ("Japanese", "Package {code} is beside the window.", "荷物{code}は窓のそばにあります。"),
    ("Japanese", "Please take folder {code} to room {room}.", "フォルダー{code}を{room}号室に持って行ってください。"),
    ("Dutch", "Write code {code} on the blue card.", "Schrijf code {code} op de blauwe kaart."),
    ("Dutch", "Package {code} is beside the window.", "Pakket {code} ligt naast het raam."),
    ("Dutch", "Please take folder {code} to room {room}.", "Breng map {code} alstublieft naar kamer {room}."),
    ("Swedish", "Write code {code} on the blue card.", "Skriv koden {code} på det blå kortet."),
    ("Swedish", "Package {code} is beside the window.", "Paketet {code} ligger bredvid fönstret."),
    ("Swedish", "Please take folder {code} to room {room}.", "Ta mappen {code} till rum {room}, tack."),
)


def _multilingual_spec(index: int) -> dict:
    ordinal = (
        SUPPLEMENT_ORDINAL_BASE
        + POOL_COUNTS["coding_debugging"]
        + POOL_COUNTS["mathematical_reasoning"]
        + POOL_COUNTS["json_structured_data"]
        + POOL_COUNTS["instruction_following"]
        + index
    )
    language, source_template, target_template = (
        TRANSLATION_TEMPLATES[index % len(TRANSLATION_TEMPLATES)]
    )
    code = f"{chr(65 + index % 26)}{chr(65 + (index // 26) % 26)}-{1_000 + index}"
    room = str(100 + index % 800)
    source = source_template.format(code=code, room=room)
    target = target_template.format(code=code, room=room)
    prompt = f"Translate into {language}. Return only the translation: {source}"
    return _make_spec(
        category="multilingual",
        ordinal=ordinal,
        messages=_messages(prompt),
        response_format="assistant_text",
        verifier={
            "type": "exact_text_v1",
            "status": "ready",
            "config": {"expected": target},
        },
        bounds=(1, 128),
    )


POOL_BUILDERS = {
    "coding_debugging": _coding_spec,
    "mathematical_reasoning": _math_spec,
    "json_structured_data": _json_spec,
    "instruction_following": _instruction_spec,
    "multilingual": _multilingual_spec,
}


def build_candidate_pool() -> list[dict]:
    specs = []
    for category in SCOPED_CATEGORIES:
        specs.extend(
            POOL_BUILDERS[category](index)
            for index in range(POOL_COUNTS[category])
        )
    validate_prompt_specs(specs)
    if any(
        spec["verifier"]["type"] == "approval_required_v1"
        for spec in specs
    ):
        raise AssertionError("objective supplement contains a subjective gate")
    return specs


def _load_base_rows(path: Path, report_path: Path) -> tuple[list[dict], dict]:
    sealed_path = safe_regular_input(path, "base deterministic reference rows")
    sealed_report_path = safe_regular_input(
        report_path, "base deterministic reference report"
    )
    report = json.loads(sealed_report_path.read_text(encoding="utf-8"))
    report_without_self_hash = dict(report)
    claimed_report_hash = report_without_self_hash.pop(
        "content_sha256_before_self_field", None
    )
    if (
        report.get("schema")
        != "general-replay-deterministic-reference-report-v1"
        or report.get("target_total_assistant_tokens") != DEFAULT_TARGET
        or report.get("output_sha256") != file_sha256(sealed_path)
        or claimed_report_hash != canonical_sha256(report_without_self_hash)
        or report.get("policy", {}).get("duplicate_or_padding_fill_used") is not False
        or report.get("policy", {}).get("subjective_rows_compiled") is not False
    ):
        raise RuntimeError("base deterministic reference authority changed")
    rows = []
    with sealed_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"base deterministic reference line {line_number} is invalid"
                ) from exc
    if len(rows) != report.get("rows"):
        raise RuntimeError("base deterministic reference row count changed")
    identities = [(row.get("row_id"), row.get("source_group_id")) for row in rows]
    if (
        any(not row_id or not group for row_id, group in identities)
        or len({row_id for row_id, _ in identities}) != len(rows)
        or len({group for _, group in identities}) != len(rows)
    ):
        raise RuntimeError("base deterministic reference identities changed")
    by_category = Counter()
    for row in rows:
        count = row.get("assistant_token_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise RuntimeError("base deterministic reference token count changed")
        by_category[row.get("category")] += count
    for category, item in report.get("categories", {}).items():
        if by_category[category] != item.get("assistant_tokens"):
            raise RuntimeError("base deterministic reference accounting changed")
    return rows, report


def _seed_token_counts(tokenizer: Any) -> Counter:
    seed_rows, _ = validate_seed_authority()
    result = Counter()
    for row in seed_rows:
        result[row["category"]] += assistant_token_count(
            tokenizer, row["messages"], row["tools"]
        )
    return result


def select_exact_supplement(
    pool_specs: list[dict], tokenizer: Any, required: dict[str, int],
) -> tuple[list[dict], list[dict], dict]:
    """Compile all pool targets, then select exact unique-group deficits."""
    pool_rows, compiler_audit = build_reference_compiler_rows(
        pool_specs, tokenizer
    )
    selected_rows = []
    category_audit = {}
    for category in SCOPED_CATEGORIES:
        target = required[category]
        category_pool = [
            row for row in pool_rows if row["category"] == category
        ]
        selected = exact_token_group_subset(category_pool, target)
        realized = sum(row["assistant_token_count"] for row in selected)
        if realized != target:
            raise AssertionError("supplement exact selection changed")
        selected_rows.extend(selected)
        category_audit[category] = {
            "pool_specs": POOL_COUNTS[category],
            "pool_rows": len(category_pool),
            "pool_assistant_tokens": sum(
                row["assistant_token_count"] for row in category_pool
            ),
            "required_supplement_tokens": target,
            "selected_rows": len(selected),
            "selected_assistant_tokens": realized,
            "exact_unique_source_group_subset_reached": True,
        }
    selected_spec_ids = {
        row["lineage"]["spec_id"] for row in selected_rows
    }
    selected_specs = [
        spec for spec in pool_specs if spec["spec_id"] in selected_spec_ids
    ]
    if len(selected_specs) != len(selected_rows):
        raise AssertionError("selected spec/reference mapping changed")
    selected_specs.sort(key=lambda item: item["spec_id"])
    selected_rows.sort(key=lambda item: item["row_id"])
    validate_prompt_specs(selected_specs)
    return selected_specs, selected_rows, {
        "compiler": compiler_audit,
        "categories": category_audit,
    }


def build_artifacts(
    *, base_specs: list[dict], base_rows: list[dict], base_row_report: dict,
    tokenizer: Any, target: int,
) -> tuple[bytes, bytes, bytes, dict]:
    if target != DEFAULT_TARGET:
        raise ValueError("objective supplement v1 is sealed to the 150k target")
    budgets = token_budgets(target)
    seed_tokens = _seed_token_counts(tokenizer)
    base_tokens = Counter()
    for row in base_rows:
        base_tokens[row["category"]] += row["assistant_token_count"]
    required = {
        category: budgets[category] - seed_tokens[category] - base_tokens[category]
        for category in SCOPED_CATEGORIES
    }
    if any(value <= 0 for value in required.values()):
        raise RuntimeError("scoped objective supplement is no longer required")

    pool_specs = build_candidate_pool()
    selected_specs, selected_rows, selection_audit = select_exact_supplement(
        pool_specs, tokenizer, required
    )
    # This catches source-group, prompt, and spec collisions with the base
    # before either supplement artifact is materialized.
    validate_prompt_specs([*base_specs, *selected_specs])

    supplement_tokens = Counter()
    for row in selected_rows:
        supplement_tokens[row["category"]] += row["assistant_token_count"]
    accounting = {}
    for category in SCOPED_CATEGORIES:
        realized = (
            seed_tokens[category]
            + base_tokens[category]
            + supplement_tokens[category]
        )
        accounting[category] = {
            "target_assistant_tokens": budgets[category],
            "mandatory_seed_tokens": seed_tokens[category],
            "base_reference_tokens": base_tokens[category],
            "supplement_reference_tokens": supplement_tokens[category],
            "realized_assistant_tokens": realized,
            "exact": realized == budgets[category],
        }
    if not all(item["exact"] for item in accounting.values()):
        raise AssertionError("scoped 150k category accounting is not exact")

    spec_bytes = b"".join(canonical_bytes(item) for item in selected_specs)
    row_bytes = b"".join(canonical_bytes(item) for item in selected_rows)
    report = {
        "schema": "general-replay-objective-supplement-report-v1",
        "status": "exact_scoped_category_capacity_ready",
        "target_total_assistant_tokens": target,
        "target_assistant_tokens_by_category": budgets,
        "scoped_categories": list(SCOPED_CATEGORIES),
        "all_scoped_category_targets_exact": True,
        "build_seed": SUPPLEMENT_BUILD_SEED,
        "ordinal_base": SUPPLEMENT_ORDINAL_BASE,
        "base_reference": {
            "output_sha256": base_row_report["output_sha256"],
            "rows": len(base_rows),
            "report_content_sha256": base_row_report[
                "content_sha256_before_self_field"
            ],
        },
        "outputs": {
            "prompt_specs": {
                "rows": len(selected_specs),
                "sha256": hashlib.sha256(spec_bytes).hexdigest(),
            },
            "reference_rows": {
                "rows": len(selected_rows),
                "assistant_tokens": sum(supplement_tokens.values()),
                "sha256": hashlib.sha256(row_bytes).hexdigest(),
            },
        },
        "selection": selection_audit,
        "accounting": accounting,
        "tokenizer": {
            "path": str(MODEL_DIRECTORY),
            "revision": MODEL_REVISION,
            "chat_template_sha256": MODEL_FILE_SHA256["chat_template.jinja"],
            "assistant_mask_policy": "official_qwen_assistant_only_v1",
        },
        "policy": {
            "deterministic_synthetic_sources_only": True,
            "direct_benchmark_prompts": False,
            "protected_or_evaluation_sources_opened": False,
            "subjective_rows_compiled": False,
            "same_objective_verifiers_reapplied": True,
            "duplicate_or_padding_fill_used": False,
            "one_row_per_unique_source_group": True,
            "multilingual_references": "hand_authored_exact_templates_v1",
        },
    }
    report["content_sha256_before_self_field"] = canonical_sha256(report)
    report_bytes = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return spec_bytes, row_bytes, report_bytes, report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--base-specs", type=Path, default=DEFAULT_BASE_SPECS)
    result.add_argument(
        "--base-spec-report", type=Path, default=DEFAULT_BASE_SPEC_REPORT
    )
    result.add_argument("--base-rows", type=Path, default=DEFAULT_BASE_ROWS)
    result.add_argument(
        "--base-row-report", type=Path, default=DEFAULT_BASE_ROW_REPORT
    )
    result.add_argument("--target-assistant-tokens", type=int, default=DEFAULT_TARGET)
    result.add_argument("--spec-output", type=Path, default=DEFAULT_SPEC_OUTPUT)
    result.add_argument("--row-output", type=Path, default=DEFAULT_ROW_OUTPUT)
    result.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    result.add_argument("--check", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    base_specs, base_spec_report = load_prompt_artifacts(
        args.base_specs, args.base_spec_report
    )
    base_rows, base_row_report = _load_base_rows(
        args.base_rows, args.base_row_report
    )
    if base_spec_report.get("output_sha256") != file_sha256(args.base_specs):
        raise RuntimeError("base prompt artifact identity changed")
    tokenizer = load_qwen_tokenizer(
        str(MODEL_DIRECTORY),
        MODEL_REVISION,
        MODEL_FILE_SHA256["chat_template.jinja"],
    )
    spec_bytes, row_bytes, report_bytes, report = build_artifacts(
        base_specs=base_specs,
        base_rows=base_rows,
        base_row_report=base_row_report,
        tokenizer=tokenizer,
        target=args.target_assistant_tokens,
    )
    outputs = (args.spec_output, args.row_output, args.report)
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError("supplement output paths must be disjoint")
    expected = (spec_bytes, row_bytes, report_bytes)
    if args.check:
        if any(path.read_bytes() != content for path, content in zip(outputs, expected)):
            raise RuntimeError("objective supplement artifacts changed")
        return 0
    if any(path.exists() for path in outputs):
        raise FileExistsError("objective supplement build requires fresh outputs")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    for path, content in zip(outputs, expected):
        path.write_bytes(content)
    print(json.dumps({
        "spec_output": str(args.spec_output),
        "spec_sha256": report["outputs"]["prompt_specs"]["sha256"],
        "row_output": str(args.row_output),
        "row_sha256": report["outputs"]["reference_rows"]["sha256"],
        "report": str(args.report),
        "report_content_sha256": report["content_sha256_before_self_field"],
        "rows": report["outputs"]["reference_rows"]["rows"],
        "assistant_tokens": report["outputs"]["reference_rows"]["assistant_tokens"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
