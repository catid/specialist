#!/usr/bin/env python3
"""Legacy candidate-response assembler and shared replay loading helpers.

Candidate-response-only assembly is retired: deterministic reference artifacts
are now mandatory for objective categories.  The CLI fails closed while shared
tokenizer and JSONL loaders remain available to the sealed v2 builders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from general_replay_v1 import (
    DEFAULT_TOTAL_ASSISTANT_TOKENS,
    ROOT,
    build_candidate_requests,
    build_final_replay_rows,
    build_all_verified_candidate_rows,
    canonical_bytes,
    canonical_sha256,
    file_sha256,
    safe_regular_input,
    validate_seed_authority,
)
from prepare_general_replay_candidate_shards_v1 import (
    DEFAULT_SPECS,
    DEFAULT_SPEC_REPORT,
    load_prompt_artifacts,
)


DEFAULT_DIRECTORY = ROOT / "data/general_replay_v1"
DEFAULT_OUTPUT = DEFAULT_DIRECTORY / "general_replay_corpus_v1.jsonl"
DEFAULT_REPORT = DEFAULT_DIRECTORY / "general_replay_corpus_v1.report.json"


def load_jsonl_inputs(paths: list[Path], role: str) -> tuple[list[dict], dict[str, str]]:
    rows = []
    hashes = {}
    for index, raw_path in enumerate(paths):
        path = safe_regular_input(raw_path, f"{role} {index}")
        hashes[str(path)] = file_sha256(path)
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{role} {index} line {line_number} is invalid JSON"
                    ) from exc
    return rows, dict(sorted(hashes.items()))


def load_qwen_tokenizer(
        tokenizer_path: str, tokenizer_revision: str,
        chat_template_sha256: str):
    if len(chat_template_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in chat_template_sha256):
        raise ValueError("chat-template identity must be a SHA-256 digest")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        revision=tokenizer_revision,
        local_files_only=True,
        trust_remote_code=False,
    )
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template.strip():
        raise RuntimeError("pinned tokenizer has no official chat template")
    actual = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
    if actual != chat_template_sha256:
        raise RuntimeError("official Qwen chat-template identity changed")
    return tokenizer


def build_corpus_artifacts(
        *, specs: list[dict], responses: list[dict], approvals: list[dict],
        tokenizer, candidate_model_name: str, candidate_model_revision: str,
        candidate_model_identity_sha256: str,
        total_assistant_tokens: int,
) -> tuple[bytes, dict, dict]:
    seed_rows, _ = validate_seed_authority()
    request_shards = build_candidate_requests(
        specs,
        model_name=candidate_model_name,
        model_revision=candidate_model_revision,
        model_identity_sha256=candidate_model_identity_sha256,
    )
    requests = [item for shard in request_shards for item in shard]
    candidate_rows, candidate_audit = build_all_verified_candidate_rows(
        specs, requests, responses, approvals, tokenizer
    )
    rows, budget_report = build_final_replay_rows(
        seed_rows,
        candidate_rows,
        tokenizer,
        total_assistant_tokens=total_assistant_tokens,
    )
    output = b"".join(canonical_bytes(item) for item in rows)
    return output, budget_report, candidate_audit


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    result.add_argument("--spec-report", type=Path, default=DEFAULT_SPEC_REPORT)
    result.add_argument("--responses", type=Path, nargs="+", required=True)
    result.add_argument("--approval-ledger", type=Path)
    result.add_argument("--tokenizer", required=True)
    result.add_argument("--tokenizer-revision", required=True)
    result.add_argument("--chat-template-sha256", required=True)
    result.add_argument("--candidate-model-name", required=True)
    result.add_argument("--candidate-model-revision", required=True)
    result.add_argument("--candidate-model-identity-sha256", required=True)
    result.add_argument(
        "--target-assistant-tokens",
        type=int,
        default=DEFAULT_TOTAL_ASSISTANT_TOKENS,
    )
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    result.add_argument("--check", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    raise RuntimeError(
        "candidate-response-only assembly is retired; use the exact 150k "
        "deterministic-reference final assembler"
    )


if __name__ == "__main__":
    raise SystemExit(main())
