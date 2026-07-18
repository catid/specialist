#!/usr/bin/env python3
"""Build compact, response-addressed packets for subjective human review."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from build_general_replay_corpus_v1 import load_jsonl_inputs, load_qwen_tokenizer
from general_replay_v1 import (
    ROOT,
    assistant_token_count,
    build_candidate_requests,
    build_reference_compiler_rows,
    canonical_bytes,
    canonical_sha256,
    token_budgets,
    validate_candidate_responses,
)
from prepare_general_replay_candidate_shards_v1 import load_prompt_artifacts
from run_general_replay_candidate_shard_v1 import (
    MODEL_DIRECTORY,
    MODEL_FILE_SHA256,
    MODEL_IDENTITY_SHA256,
    MODEL_NAME,
    MODEL_REVISION,
)


DIRECTORY = ROOT / "data/general_replay_v1"
DEFAULT_SPECS = DIRECTORY / "prompt_specs_v2_scale32.jsonl"
DEFAULT_SPEC_REPORT = DIRECTORY / "prompt_specs_v2_scale32.report.json"
DEFAULT_RESPONSES = [
    DIRECTORY / "candidate_responses_v1_scale32" / f"shard-{index:02d}.responses.jsonl"
    for index in range(4)
]
DEFAULT_OUTPUT_DIRECTORY = DIRECTORY / "approval_review_packet_v1_scale32"
DEFAULT_TARGET = 150_000
DEFAULT_CHUNK_SPECS = 8


def rubric_definition(rubric_id: str) -> dict:
    if rubric_id == "ordinary-conversation-warm-practical-v1":
        criteria = [
            "Directly addresses the user's stated situation.",
            "Uses a warm, respectful tone without inventing personal facts.",
            "Offers concrete and proportionate practical help.",
            "Avoids unsafe, deceptive, or needlessly overconfident advice.",
        ]
    elif rubric_id.startswith("bilingual-translation-") and rubric_id.endswith("-v1"):
        criteria = [
            "Preserves the source meaning, factual details, and tone.",
            "Uses the requested target language naturally and correctly.",
            "Adds no unsupported claims or unrequested commentary.",
            "Contains no material omission or mistranslation.",
        ]
    elif rubric_id == "safe-boundary-constructive-alternative-v1":
        criteria = [
            "Does not provide actionable harmful instructions.",
            "States a clear, proportionate boundary without moralizing.",
            "Offers a genuinely safer and relevant alternative.",
            "Does not leak the disallowed details through the explanation.",
        ]
    elif rubric_id == "uncertainty-no-fabrication-v1":
        criteria = [
            "Does not fabricate a fact, citation, event, or source.",
            "Clearly identifies what cannot be established from the prompt.",
            "Explains what evidence would resolve the uncertainty.",
            "Remains helpful without disguising a guess as fact.",
        ]
    else:
        raise ValueError(f"unknown subjective rubric {rubric_id!r}")
    return {
        "rubric_id": rubric_id,
        "decision": "approve only when every criterion passes",
        "criteria": criteria,
    }


def build_packet_artifacts(
        specs: list[dict], responses: list[dict], tokenizer,
        *, target: int, chunk_specs: int,
) -> tuple[dict[str, bytes], bytes, dict]:
    if isinstance(chunk_specs, bool) or not isinstance(chunk_specs, int) or chunk_specs < 1:
        raise ValueError("chunk spec count must be a positive integer")
    requests = [
        item
        for shard in build_candidate_requests(
            specs,
            model_name=MODEL_NAME,
            model_revision=MODEL_REVISION,
            model_identity_sha256=MODEL_IDENTITY_SHA256,
        )
        for item in shard
    ]
    validate_candidate_responses(responses, requests)
    responses_by_spec = defaultdict(list)
    for response in responses:
        responses_by_spec[response["spec_id"]].append(response)

    reference_rows, _ = build_reference_compiler_rows(specs, tokenizer)
    reference_tokens = defaultdict(int)
    for row in reference_rows:
        reference_tokens[row["category"]] += row["assistant_token_count"]
    budgets = token_budgets(target)
    records_by_category = defaultdict(list)
    skipped_by_category = defaultdict(lambda: {"length": 0, "token_bounds": 0})
    rubric_ids = set()
    for spec in sorted(specs, key=lambda item: item["spec_id"]):
        if spec["verifier"]["type"] != "approval_required_v1":
            continue
        rubric_id = spec["verifier"]["config"]["rubric_id"]
        rubric_ids.add(rubric_id)
        candidates = []
        for response in sorted(
                responses_by_spec[spec["spec_id"]],
                key=lambda item: (item["candidate_index"], item["response_sha256"])):
            if response["finish_reason"] == "length":
                skipped_by_category[spec["category"]]["length"] += 1
                continue
            count = assistant_token_count(
                tokenizer,
                [*spec["messages"], response["assistant_message"]],
                spec["tools"],
            )
            bounds = spec["candidate_slot"]
            if not bounds["min_assistant_tokens"] <= count <= bounds["max_assistant_tokens"]:
                skipped_by_category[spec["category"]]["token_bounds"] += 1
                continue
            candidates.append({
                "candidate_index": response["candidate_index"],
                "response_sha256": response["response_sha256"],
                "assistant_message": response["assistant_message"],
                "assistant_token_count": count,
            })
        if not candidates:
            continue
        records_by_category[spec["category"]].append({
            "schema": "general-replay-approval-review-item-v1",
            "spec_id": spec["spec_id"],
            "source_group_id": spec["source_group_id"],
            "prompt_identity_sha256": spec["prompt_identity_sha256"],
            "category": spec["category"],
            "messages": spec["messages"],
            "tools": spec["tools"],
            "expected_response_format": spec["expected_response_format"],
            "rubric_id": rubric_id,
            "required_approvals": spec["verifier"]["config"]["required_approvals"],
            "candidates": candidates,
        })

    artifacts = {}
    chunks = []
    categories = {}
    subjective_categories = sorted(records_by_category)
    for category in subjective_categories:
        records = sorted(
            records_by_category[category],
            key=lambda item: (
                -max(c["assistant_token_count"] for c in item["candidates"]),
                item["spec_id"],
            ),
        )
        remaining = max(0, budgets[category] - reference_tokens[category])
        maxima = sorted(
            (max(c["assistant_token_count"] for c in item["candidates"])
             for item in records),
            reverse=True,
        )
        covered = 0
        minimum_rows = 0
        for count in maxima:
            if covered >= remaining:
                break
            covered += count
            minimum_rows += 1
        capacity_sufficient = covered >= remaining
        category_chunks = []
        for offset in range(0, len(records), chunk_specs):
            chunk = records[offset:offset + chunk_specs]
            index = offset // chunk_specs
            name = f"{category}.chunk-{index:03d}.jsonl"
            content = b"".join(canonical_bytes(item) for item in chunk)
            artifacts[name] = content
            entry = {
                "path": name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "category": category,
                "specs": len(chunk),
                "candidates": sum(len(item["candidates"]) for item in chunk),
                "assistant_tokens_all_candidates": sum(
                    candidate["assistant_token_count"]
                    for item in chunk for candidate in item["candidates"]
                ),
            }
            chunks.append(entry)
            category_chunks.append(entry)
        categories[category] = {
            "target_assistant_tokens": budgets[category],
            "deterministic_reference_tokens": reference_tokens[category],
            "assistant_tokens_remaining_for_manual_review": remaining,
            "review_specs": len(records),
            "review_candidates": sum(len(item["candidates"]) for item in records),
            "review_max_one_per_source_group_tokens": sum(maxima),
            "review_capacity_shortfall_tokens": max(0, remaining - sum(maxima)),
            "theoretical_minimum_approved_rows_to_cover_remaining": (
                minimum_rows if capacity_sufficient else None
            ),
            "theoretical_minimum_chunks_to_cover_remaining": (
                math.ceil(minimum_rows / chunk_specs) if capacity_sufficient else None
            ),
            "chunks": len(category_chunks),
            "skipped_length_candidates": skipped_by_category[category]["length"],
            "skipped_token_bound_candidates": skipped_by_category[category]["token_bounds"],
            "capacity_lower_bound_ignores_review_outcomes": True,
        }
    manifest = {
        "schema": "general-replay-approval-review-packet-manifest-v1",
        "status": "pending_human_review",
        "target_total_assistant_tokens": target,
        "chunk_spec_capacity": chunk_specs,
        "categories": categories,
        "chunks": chunks,
        "rubrics": [rubric_definition(item) for item in sorted(rubric_ids)],
        "approval_ledger_contract": {
            "schema": "general-replay-candidate-approval-v1",
            "required_fields": [
                "schema", "spec_id", "response_sha256", "status",
                "rubric_id", "reviewer", "reviewed_at", "reason",
            ],
            "status_values": ["approved", "rejected"],
            "response_addressed": True,
        },
        "policy": {
            "auto_approvals_created": False,
            "approval_ledger_created": False,
            "length_truncated_candidates_excluded": True,
            "token_bound_failures_excluded": True,
            "direct_benchmark_or_protected_sources_used": False,
        },
    }
    manifest["content_sha256_before_self_field"] = canonical_sha256(manifest)
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    return artifacts, manifest_bytes, manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    result.add_argument("--spec-report", type=Path, default=DEFAULT_SPEC_REPORT)
    result.add_argument("--responses", type=Path, nargs="+", default=DEFAULT_RESPONSES)
    result.add_argument("--target-assistant-tokens", type=int, default=DEFAULT_TARGET)
    result.add_argument("--chunk-specs", type=int, default=DEFAULT_CHUNK_SPECS)
    result.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    result.add_argument("--check", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    specs, _ = load_prompt_artifacts(args.specs, args.spec_report)
    responses, _ = load_jsonl_inputs(args.responses, "candidate response")
    tokenizer = load_qwen_tokenizer(
        str(MODEL_DIRECTORY), MODEL_REVISION,
        MODEL_FILE_SHA256["chat_template.jinja"],
    )
    artifacts, manifest_bytes, manifest = build_packet_artifacts(
        specs, responses, tokenizer,
        target=args.target_assistant_tokens,
        chunk_specs=args.chunk_specs,
    )
    manifest_path = args.output_directory / "manifest.json"
    if args.check:
        for name, content in artifacts.items():
            if (args.output_directory / name).read_bytes() != content:
                raise RuntimeError(f"approval review chunk changed: {name}")
        if manifest_path.read_bytes() != manifest_bytes:
            raise RuntimeError("approval review manifest changed")
        return 0
    if args.output_directory.exists():
        raise FileExistsError("approval review packet build requires a fresh directory")
    args.output_directory.mkdir(parents=True)
    for name, content in artifacts.items():
        (args.output_directory / name).write_bytes(content)
    manifest_path.write_bytes(manifest_bytes)
    print(json.dumps({
        "output_directory": str(args.output_directory),
        "manifest": str(manifest_path),
        "manifest_content_sha256": manifest["content_sha256_before_self_field"],
        "chunks": len(artifacts),
        "review_candidates": sum(
            item["review_candidates"] for item in manifest["categories"].values()
        ),
        "status": manifest["status"],
        "auto_approvals_created": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
