#!/usr/bin/env python3
"""Seal aggregate Qwen token-boundary evidence for V18A train sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

import build_eggroll_es_overlay_frame_v18a as frame_v18a
import train_eggroll_es_specialist as base


ROOT = Path(__file__).resolve().parent
MODEL_PATH_V18A = (ROOT / "models/Qwen3.6-35B-A3B").resolve()
OUTPUT_PATH_V18A = (
    ROOT / "experiments/eggroll_es_hpo/S6_TOKEN_LENGTH_AUDIT_V18A.json"
).resolve()
MAX_TOTAL_TOKENS_V18A = 1024
QUANTILES_V18A = (0.50, 0.90, 0.95, 0.99, 1.0)
QUANTILE_METHOD_V18A = "higher"
BASE_SPECIALIST_SHA256_V18A = (
    "bbffbf16747ec514c67e48daab696560eb3309f5a3edf0a700257969cad35c23"
)
TOKENIZER_FILES_V18A = {
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "tokenizer_config.json": "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b",
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    "chat_template.jinja": "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_tokenizer_v18a():
    expected = {
        MODEL_PATH_V18A / name: digest
        for name, digest in TOKENIZER_FILES_V18A.items()
    }
    expected[Path(base.__file__).resolve()] = BASE_SPECIALIST_SHA256_V18A
    if any(file_sha256(path) != digest for path, digest in expected.items()):
        raise RuntimeError("v18a tokenizer or prompt template identity changed")
    return AutoTokenizer.from_pretrained(
        MODEL_PATH_V18A, local_files_only=True,
    )


def _read_bound_rows_v18a(path: Path, expected_sha256: str, expected_rows: int):
    if file_sha256(path) != expected_sha256:
        raise RuntimeError("v18a token-audit train source identity changed")
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if len(rows) != expected_rows:
        raise RuntimeError("v18a token-audit train source row count changed")
    return rows


def _quantiles_v18a(values: list[int]) -> list[int]:
    result = np.quantile(
        np.asarray(values, dtype=np.int64),
        QUANTILES_V18A,
        method=QUANTILE_METHOD_V18A,
    )
    return [int(item) for item in result]


def audit_source_v18a(tokenizer, rows: list[dict]) -> dict:
    total_lengths = []
    answer_lengths = []
    mismatch_count = 0
    over_cap_count = 0
    for row in rows:
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not question:
            raise RuntimeError("v18a token-audit question contract changed")
        if not isinstance(answer, str) or not answer:
            raise RuntimeError("v18a token-audit answer contract changed")
        prompt = base.specialist_template(question)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        combined_ids = tokenizer.encode(prompt + answer, add_special_tokens=False)
        if combined_ids[:len(prompt_ids)] != prompt_ids:
            mismatch_count += 1
        answer_ids = combined_ids[len(prompt_ids):]
        total_lengths.append(len(combined_ids))
        answer_lengths.append(len(answer_ids))
        over_cap_count += len(combined_ids) > MAX_TOTAL_TOKENS_V18A
    return {
        "rows": len(rows),
        "tokenizer_boundary_mismatch_count": mismatch_count,
        "over_frozen_1024_total_token_cap_count": over_cap_count,
        "combined_prompt_answer_token_quantiles_p50_p90_p95_p99_max": (
            _quantiles_v18a(total_lengths)
        ),
        "answer_token_quantiles_p50_p90_p95_p99_max": _quantiles_v18a(
            answer_lengths
        ),
    }


def build_audit_v18a() -> dict:
    tokenizer = load_tokenizer_v18a()
    sources = {
        "candidate_v298": (
            frame_v18a.CANDIDATE_PATH_V18A,
            frame_v18a.CANDIDATE_SHA256_V18A,
            frame_v18a.CANDIDATE_ROWS_V18A,
        ),
        "production": (
            frame_v18a.PRODUCTION_PATH_V18A,
            frame_v18a.PRODUCTION_SHA256_V18A,
            frame_v18a.PRODUCTION_ROWS_V18A,
        ),
    }
    results = {}
    for name, (path, digest, rows) in sources.items():
        result = audit_source_v18a(
            tokenizer, _read_bound_rows_v18a(path, digest, rows)
        )
        result["path"] = str(path)
        result["file_sha256"] = digest
        results[name] = result
    value = {
        "schema": "eggroll-es-train-token-length-audit-v18a",
        "status": "complete_train_only_aggregate_evidence",
        "model": str(MODEL_PATH_V18A),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_file_sha256": dict(sorted(TOKENIZER_FILES_V18A.items())),
        "prompt_template": {
            "path": str(Path(base.__file__).resolve()),
            "file_sha256": BASE_SPECIALIST_SHA256_V18A,
        },
        "quantile_method": QUANTILE_METHOD_V18A,
        "max_total_prompt_answer_tokens": MAX_TOTAL_TOKENS_V18A,
        "sources": results,
        "firewall": {
            "train_only": True,
            "heldout_validation_ood_eval_or_benchmark_content_opened": False,
            "contains_question_answer_prompt_or_row_content": False,
            "gpu_launched": False,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    validate_audit_v18a(value)
    return value


def validate_audit_v18a(value: dict) -> dict:
    without_self = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("schema") != "eggroll-es-train-token-length-audit-v18a"
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(without_self)
        or value.get("quantile_method") != "higher"
        or value.get("sources", {}).get("candidate_v298", {}).get(
            "combined_prompt_answer_token_quantiles_p50_p90_p95_p99_max"
        ) != [67, 91, 104, 129, 144]
        or value.get("sources", {}).get("candidate_v298", {}).get(
            "answer_token_quantiles_p50_p90_p95_p99_max"
        ) != [19, 42, 52, 74, 86]
        or any(
            source.get("tokenizer_boundary_mismatch_count") != 0
            or source.get("over_frozen_1024_total_token_cap_count") != 0
            for source in value.get("sources", {}).values()
        )
        or value.get("firewall", {}).get(
            "heldout_validation_ood_eval_or_benchmark_content_opened"
        ) is not False
        or value.get("firewall", {}).get("gpu_launched") is not False
    ):
        raise RuntimeError("v18a train token-length audit changed")
    return value


def _exclusive_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("immutable v18a token audit already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH_V18A))
    args = parser.parse_args(argv)
    if Path(args.output).resolve() != OUTPUT_PATH_V18A:
        raise ValueError("v18a token audit output path changed")
    value = build_audit_v18a()
    _exclusive_write(OUTPUT_PATH_V18A, value)
    result = {
        "schema": "eggroll-es-train-token-length-audit-write-v18a",
        "path": str(OUTPUT_PATH_V18A),
        "file_sha256": file_sha256(OUTPUT_PATH_V18A),
        "content_sha256": value["content_sha256_before_self_field"],
        "gpu_launched": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
