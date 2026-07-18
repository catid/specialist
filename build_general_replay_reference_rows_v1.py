#!/usr/bin/env python3
"""Compile trusted objective targets into verifier-checked replay rows."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from build_general_replay_corpus_v1 import load_qwen_tokenizer
from general_replay_v1 import (
    ROOT,
    assistant_token_count,
    build_reference_compiler_rows,
    canonical_bytes,
    canonical_sha256,
    exact_token_group_subset,
    token_budgets,
    validate_seed_authority,
)
from prepare_general_replay_candidate_shards_v1 import load_prompt_artifacts
from run_general_replay_candidate_shard_v1 import (
    MODEL_DIRECTORY,
    MODEL_FILE_SHA256,
    MODEL_REVISION,
)


DIRECTORY = ROOT / "data/general_replay_v1"
DEFAULT_SPECS = DIRECTORY / "prompt_specs_v2_scale32.jsonl"
DEFAULT_SPEC_REPORT = DIRECTORY / "prompt_specs_v2_scale32.report.json"
DEFAULT_OUTPUT = DIRECTORY / "deterministic_reference_rows_v1_scale32.jsonl"
DEFAULT_REPORT = DIRECTORY / "deterministic_reference_rows_v1_scale32.report.json"
DEFAULT_TARGET = 150_000


def build_artifacts(specs: list[dict], tokenizer, target: int) -> tuple[bytes, bytes, dict]:
    rows, compiler_audit = build_reference_compiler_rows(specs, tokenizer)
    seed_rows, _ = validate_seed_authority()
    seed_tokens = Counter()
    for row in seed_rows:
        seed_tokens[row["category"]] += assistant_token_count(
            tokenizer, row["messages"], row["tools"]
        )
    budgets = token_budgets(target)
    categories = {}
    for category, category_target in budgets.items():
        pool = [row for row in rows if row["category"] == category]
        required = category_target - seed_tokens[category]
        try:
            selected = exact_token_group_subset(pool, required)
            exact_reachable = True
            selected_rows = len(selected)
        except RuntimeError:
            exact_reachable = False
            selected_rows = None
        available = sum(row["assistant_token_count"] for row in pool)
        categories[category] = {
            **compiler_audit["by_category"][category],
            "target_assistant_tokens": category_target,
            "mandatory_seed_tokens": seed_tokens[category],
            "candidate_tokens_required_after_seed": required,
            "reference_capacity_shortfall_tokens": max(0, required - available),
            "exact_reference_only_budget_reachable": exact_reachable,
            "exact_reference_only_selected_rows": selected_rows,
        }
    output = b"".join(canonical_bytes(row) for row in rows)
    report = {
        "schema": "general-replay-deterministic-reference-report-v1",
        "status": (
            "exact_reference_capacity_ready"
            if all(item["exact_reference_only_budget_reachable"] for item in categories.values())
            else "capacity_audited_supplement_required"
        ),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "rows": len(rows),
        "assistant_tokens": compiler_audit["assistant_tokens"],
        "target_total_assistant_tokens": target,
        "categories": categories,
        "compiler": {
            "name": "deterministic_reference_compiler_v1",
            "version": "1",
            "objective_targets_only": True,
            "same_verifiers_reapplied": True,
        },
        "policy": {
            "base_generations_promoted": False,
            "subjective_rows_compiled": False,
            "subjective_autoapproval": False,
            "assistant_only_loss": True,
            "duplicate_or_padding_fill_used": False,
        },
        "tokenizer": {
            "path": str(MODEL_DIRECTORY),
            "revision": MODEL_REVISION,
            "chat_template_sha256": MODEL_FILE_SHA256["chat_template.jinja"],
        },
    }
    report["content_sha256_before_self_field"] = canonical_sha256(report)
    report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    return output, report_bytes, report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    result.add_argument("--spec-report", type=Path, default=DEFAULT_SPEC_REPORT)
    result.add_argument("--target-assistant-tokens", type=int, default=DEFAULT_TARGET)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    result.add_argument("--check", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    specs, spec_report = load_prompt_artifacts(args.specs, args.spec_report)
    tokenizer = load_qwen_tokenizer(
        str(MODEL_DIRECTORY), MODEL_REVISION,
        MODEL_FILE_SHA256["chat_template.jinja"],
    )
    output, report_bytes, report = build_artifacts(
        specs, tokenizer, args.target_assistant_tokens
    )
    if args.check:
        if args.output.read_bytes() != output or args.report.read_bytes() != report_bytes:
            raise RuntimeError("deterministic reference artifacts changed")
        return 0
    if args.output.exists() or args.report.exists():
        raise FileExistsError("deterministic reference build requires fresh outputs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_bytes(report_bytes)
    print(json.dumps({
        "output": str(args.output),
        "output_sha256": report["output_sha256"],
        "report": str(args.report),
        "report_content_sha256": report["content_sha256_before_self_field"],
        "rows": report["rows"],
        "status": report["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
