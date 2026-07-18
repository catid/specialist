#!/usr/bin/env python3
"""Seal aggregate verification evidence for rejected base generations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_general_replay_corpus_v1 import load_jsonl_inputs, load_qwen_tokenizer
from general_replay_v1 import (
    ROOT,
    build_all_verified_candidate_rows,
    build_candidate_requests,
    canonical_sha256,
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
DEFAULT_OUTPUT = DIRECTORY / "generated_candidate_rejection_audit_v1.json"


def build_report(specs: list[dict], responses: list[dict], tokenizer,
                 response_hashes: dict[str, str], spec_report: dict) -> dict:
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
    _, audit = build_all_verified_candidate_rows(
        specs, requests, responses, [], tokenizer
    )
    categories = {}
    for category, stats in audit["by_category"].items():
        denominator = stats["objective_passed"] + stats["objective_failed"]
        categories[category] = {
            **stats,
            "objective_pass_rate_milli": (
                stats["objective_passed"] * 1_000 // denominator
                if denominator else None
            ),
        }
    report = {
        "schema": "general-replay-generated-candidate-rejection-audit-v1",
        "status": "sealed_rejected_training_source_evidence",
        "prompt_specs": {
            "sha256": spec_report["output_sha256"],
            "rows": spec_report["rows"],
        },
        "candidate_responses": response_hashes,
        "audit": {**audit, "by_category": categories},
        "candidate_model": {
            "name": MODEL_NAME,
            "revision": MODEL_REVISION,
            "identity_sha256": MODEL_IDENTITY_SHA256,
        },
        "policy": {
            "base_generations_promoted": False,
            "objective_training_source": "deterministic_reference_compiler_v1",
            "subjective_training_source_requires_human_approval": True,
            "candidate_text_embedded_in_audit": False,
            "aggregate_metrics_only": True,
        },
    }
    report["content_sha256_before_self_field"] = canonical_sha256(report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    result.add_argument("--spec-report", type=Path, default=DEFAULT_SPEC_REPORT)
    result.add_argument("--responses", type=Path, nargs="+", default=DEFAULT_RESPONSES)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--check", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    specs, spec_report = load_prompt_artifacts(args.specs, args.spec_report)
    responses, response_hashes = load_jsonl_inputs(
        args.responses, "candidate response"
    )
    tokenizer = load_qwen_tokenizer(
        str(MODEL_DIRECTORY), MODEL_REVISION,
        MODEL_FILE_SHA256["chat_template.jinja"],
    )
    report = build_report(
        specs, responses, tokenizer, response_hashes, spec_report
    )
    content = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    if args.check:
        if args.output.read_bytes() != content:
            raise RuntimeError("generated-candidate rejection audit changed")
        return 0
    if args.output.exists():
        raise FileExistsError("candidate audit build requires a fresh output")
    args.output.write_bytes(content)
    print(json.dumps({
        "output": str(args.output),
        "report_content_sha256": report["content_sha256_before_self_field"],
        "responses": report["audit"]["responses"],
        "status": report["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
