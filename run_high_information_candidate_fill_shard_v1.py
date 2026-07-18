#!/usr/bin/env python3
"""Generate an independently seeded, quality-focused domain deficit-fill pass.

This wrapper intentionally reuses the sealed Qwen candidate worker runtime and
request plan while changing three content-addressed surfaces: output paths,
sampling seeds, and the generation rubric.  Its outputs remain structural- and
semantic-unverified; they can never enter training merely because generation
completed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import run_high_information_candidate_shard_v1 as base


PASS_ID = "quality-deficit-fill-v1"
ROOT = Path(__file__).resolve().parent


def fill_request_seed(request_id: str) -> int:
    digest = hashlib.sha256(f"{PASS_ID}:{request_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def fill_shard_paths(shard_index: int, *, smoke: bool) -> dict[str, Path]:
    suffix = ".smoke" if smoke else ""
    stem = f"generation_fill_candidates_gpu{shard_index}{suffix}"
    return {
        "partial": base.corpus.OUTPUT_DIR / f"{stem}.partial.jsonl",
        "output": base.corpus.OUTPUT_DIR / f"{stem}.jsonl",
        "report": base.corpus.OUTPUT_DIR / f"{stem}.report.json",
        "telemetry": base.corpus.OUTPUT_DIR / f"{stem}.telemetry.json",
    }


def fill_generation_messages(request: dict, context: dict) -> list[dict[str, str]]:
    messages = base._ORIGINAL_GENERATION_MESSAGES(request, context)
    system = messages[0]["content"]
    system += (
        " This is an independently seeded quality deficit-fill pass. Prefer "
        "technique, mechanism, causal relationship, comparison, decision, "
        "safety, and meaningful lineage questions over bare era, date, public-"
        "identity, roster, or location trivia whenever the context affords a "
        "higher-value alternative. Every requested facet must be explicitly "
        "answered and supported by the cited excerpts. Avoid compound questions "
        "unless the answer and evidence cover every clause. Use one to three "
        "concise factual sentences per answer; never add padding or unsupported "
        "background. For application_scenario, construct a concrete novel "
        "scenario that genuinely requires applying the stated principle, and "
        "make the correct action or tradeoff explicit. For comparison_or_mechanism, "
        "state the relationship or mechanism rather than merely naming items. "
        "For misconception_correction, identify and correct the precise false "
        "premise. For calibrated negatives, explain exactly which requested "
        "information is absent or which scope transfer is unsupported."
    )
    messages[0] = {"role": "system", "content": system}
    return messages


def fill_sampling_kwargs(request: dict) -> dict[str, Any]:
    values = base._ORIGINAL_SAMPLING_KWARGS(request)
    values.update(
        {
            "temperature": 0.4,
            "top_p": 0.9,
            "seed": fill_request_seed(request["request_id"]),
        }
    )
    return values


def fill_make_candidate_record(
    request: dict,
    text: str,
    *,
    finish_reason: str | None,
    generated_token_count: int,
) -> dict:
    row = base._ORIGINAL_MAKE_CANDIDATE_RECORD(
        request,
        text,
        finish_reason=finish_reason,
        generated_token_count=generated_token_count,
    )
    row["generator"] = {
        **row["generator"],
        "temperature": 0.4,
        "top_p": 0.9,
        "seed": fill_request_seed(request["request_id"]),
        "generation_pass": PASS_ID,
    }
    row["content_sha256_before_self_field"] = base.candidate_content_sha256(row)
    return row


def configure_base() -> None:
    """Install the pass-specific surfaces before invoking the sealed runtime."""

    base.request_seed = fill_request_seed
    base.shard_paths = fill_shard_paths
    base.generation_messages = fill_generation_messages
    base.sampling_kwargs = fill_sampling_kwargs
    base.make_candidate_record = fill_make_candidate_record
    # The reused worker records Path(__file__) as its implementation receipt.
    # Point that global at this wrapper so the altered behavior is not mislabeled
    # as the original pass.
    base.__file__ = str(Path(__file__).resolve())


def main(argv: list[str] | None = None) -> int:
    configure_base()
    return base.main(argv)


# Preserve immutable references for wrapper calls and focused tests.  Assigning
# these names does not mutate worker behavior until configure_base() is called.
base._ORIGINAL_GENERATION_MESSAGES = base.generation_messages
base._ORIGINAL_SAMPLING_KWARGS = base.sampling_kwargs
base._ORIGINAL_MAKE_CANDIDATE_RECORD = base.make_candidate_record


if __name__ == "__main__":
    raise SystemExit(main())
