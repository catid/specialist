#!/usr/bin/env python3
"""Prepare four deterministic base-model candidate request shards; never run them."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from general_replay_v1 import (
    PROMPT_REPORT_SCHEMA,
    ROOT,
    SHARD_COUNT,
    build_candidate_requests,
    candidate_capacity_proof,
    canonical_bytes,
    canonical_sha256,
    safe_regular_input,
    validate_prompt_specs,
)


DEFAULT_DIRECTORY = ROOT / "data/general_replay_v1"
DEFAULT_SPECS = DEFAULT_DIRECTORY / "prompt_specs_v1_scale32.jsonl"
DEFAULT_SPEC_REPORT = DEFAULT_DIRECTORY / "prompt_specs_v1_scale32.report.json"
DEFAULT_OUTPUT_DIRECTORY = DEFAULT_DIRECTORY / "candidate_requests_v1_scale32"


def load_prompt_artifacts(spec_path: Path, report_path: Path) -> tuple[list[dict], dict]:
    spec_path = safe_regular_input(spec_path, "prompt spec artifact")
    report_path = safe_regular_input(report_path, "prompt spec report")
    raw = spec_path.read_bytes()
    specs = [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    validate_prompt_specs(specs)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != PROMPT_REPORT_SCHEMA
        or report.get("rows") != len(specs)
        or report.get("output_sha256") != hashlib.sha256(raw).hexdigest()
        or report.get("policy", {}).get("direct_benchmark_prompts") is not False
        or report.get("policy", {}).get(
            "protected_or_evaluation_sources_opened") is not False
    ):
        raise RuntimeError("prompt spec report does not authorize this artifact")
    target = report.get("target_total_assistant_tokens")
    proof = candidate_capacity_proof(specs, target)
    if (
        report.get("candidate_capacity_proof") != proof
        or proof["candidate_generation_launch_eligible"] is not True
    ):
        raise RuntimeError(
            "prompt pool lacks a verified per-category candidate-capacity proof"
        )
    return specs, report


def build_shard_artifacts(
        specs: list[dict], *, model_name: str, model_revision: str,
        model_identity_sha256: str,
        total_assistant_tokens: int) -> tuple[list[bytes], bytes, dict]:
    capacity = candidate_capacity_proof(specs, total_assistant_tokens)
    if capacity["candidate_generation_launch_eligible"] is not True:
        raise RuntimeError(
            "candidate request shards require at least 2x capacity per category"
        )
    shards = build_candidate_requests(
        specs,
        model_name=model_name,
        model_revision=model_revision,
        model_identity_sha256=model_identity_sha256,
    )
    shard_bytes = [
        b"".join(canonical_bytes(item) for item in shard)
        for shard in shards
    ]
    report = {
        "schema": "general-replay-candidate-request-shards-report-v1",
        "generation_launched": False,
        "shard_count": SHARD_COUNT,
        "requests": sum(len(shard) for shard in shards),
        "model": {
            "name": model_name,
            "revision": model_revision,
            "identity_sha256": model_identity_sha256,
        },
        "template_policy": "official_qwen_apply_chat_template_v1",
        "assistant_mask_policy": "assistant_only_v1",
        "engine_policy": {
            "backend": "vllm",
            "version": "0.25.0",
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "max_num_seqs": 64,
            "enable_prefix_caching": False,
        },
        "verifier_targets_in_requests": False,
        "candidate_capacity_proof_sha256": canonical_sha256(capacity),
        "candidate_capacity_by_category": capacity["categories"],
        "max_selectable_assistant_tokens": (
            capacity["max_selectable_assistant_tokens"]
        ),
        "target_total_assistant_tokens": total_assistant_tokens,
        "shards": [
            {
                "index": index,
                "rows": len(shards[index]),
                "sha256": hashlib.sha256(shard_bytes[index]).hexdigest(),
            }
            for index in range(SHARD_COUNT)
        ],
    }
    report["content_sha256_before_self_field"] = canonical_sha256(report)
    report_bytes = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return shard_bytes, report_bytes, report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    result.add_argument("--spec-report", type=Path, default=DEFAULT_SPEC_REPORT)
    result.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    result.add_argument("--model-name", required=True)
    result.add_argument("--model-revision", required=True)
    result.add_argument("--model-identity-sha256", required=True)
    result.add_argument("--check", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    specs, spec_report = load_prompt_artifacts(args.specs, args.spec_report)
    shard_bytes, report_bytes, report = build_shard_artifacts(
        specs,
        model_name=args.model_name,
        model_revision=args.model_revision,
        model_identity_sha256=args.model_identity_sha256,
        total_assistant_tokens=spec_report["target_total_assistant_tokens"],
    )
    paths = [
        args.output_directory / f"shard-{index:02d}.requests.jsonl"
        for index in range(SHARD_COUNT)
    ]
    report_path = args.output_directory / "request_shards.report.json"
    if args.check:
        for path, expected in zip(paths, shard_bytes):
            if path.read_bytes() != expected:
                raise RuntimeError(f"candidate request shard {path.name} changed")
        if report_path.read_bytes() != report_bytes:
            raise RuntimeError("candidate request shard report changed")
        return 0
    args.output_directory.mkdir(parents=True, exist_ok=True)
    if any(path.exists() for path in [*paths, report_path]):
        raise FileExistsError("candidate shard preparation requires fresh outputs")
    for path, content in zip(paths, shard_bytes):
        path.write_bytes(content)
    report_path.write_bytes(report_bytes)
    print(json.dumps({
        "generation_launched": False,
        "output_directory": str(args.output_directory),
        "prompt_spec_sha256": spec_report["output_sha256"],
        "requests": report["requests"],
        "report_content_sha256": report["content_sha256_before_self_field"],
        "shards": SHARD_COUNT,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
