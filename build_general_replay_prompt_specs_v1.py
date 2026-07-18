#!/usr/bin/env python3
"""Build deterministic general-replay prompt specifications and budget report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from general_replay_v1 import (
    BASE_SCALE32_CAPACITY_TARGET_ASSISTANT_TOKENS,
    DEFAULT_BUILD_SEED,
    ROOT,
    build_prompt_specs,
    canonical_bytes,
    canonical_sha256,
    validate_seed_authority,
)


DEFAULT_DIRECTORY = ROOT / "data/general_replay_v1"
DEFAULT_SPEC_SCALE = 32
DEFAULT_OUTPUT = DEFAULT_DIRECTORY / "prompt_specs_v1_scale32.jsonl"
DEFAULT_REPORT = DEFAULT_DIRECTORY / "prompt_specs_v1_scale32.report.json"


def build_artifacts(
        *, build_seed: int, total_assistant_tokens: int,
        spec_scale: int) -> tuple[bytes, bytes, dict]:
    validate_seed_authority()
    specs, report = build_prompt_specs(
        build_seed=build_seed,
        total_assistant_tokens=total_assistant_tokens,
        spec_scale=spec_scale,
    )
    output = b"".join(canonical_bytes(item) for item in specs)
    if hashlib.sha256(output).hexdigest() != report["output_sha256"]:
        raise AssertionError("prompt spec report/output identity mismatch")
    report = dict(report)
    report["content_sha256_before_self_field"] = canonical_sha256(report)
    report_bytes = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    return output, report_bytes, report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    result.add_argument("--build-seed", type=int, default=DEFAULT_BUILD_SEED)
    result.add_argument(
        "--target-assistant-tokens",
        type=int,
        default=BASE_SCALE32_CAPACITY_TARGET_ASSISTANT_TOKENS,
    )
    result.add_argument("--spec-scale", type=int, default=DEFAULT_SPEC_SCALE)
    result.add_argument("--check", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output, report_bytes, report = build_artifacts(
        build_seed=args.build_seed,
        total_assistant_tokens=args.target_assistant_tokens,
        spec_scale=args.spec_scale,
    )
    if args.output.resolve() == args.report.resolve():
        raise ValueError("prompt spec output and report paths must be disjoint")
    if args.check:
        if args.output.read_bytes() != output:
            raise RuntimeError("general replay prompt specs changed")
        if args.report.read_bytes() != report_bytes:
            raise RuntimeError("general replay prompt spec report changed")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.report.exists():
        raise FileExistsError("general replay prompt build requires fresh outputs")
    args.output.write_bytes(output)
    args.report.write_bytes(report_bytes)
    print(json.dumps({
        "output": str(args.output),
        "output_sha256": report["output_sha256"],
        "report": str(args.report),
        "report_content_sha256": report["content_sha256_before_self_field"],
        "rows": report["rows"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
