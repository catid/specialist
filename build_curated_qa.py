#!/usr/bin/env python3
"""Merge validated QA tranches into one deterministic future-training set."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

from build_leakfree_qa import eval_facts
from qa_quality import (LOW_VALUE, has_protocol_tokens, leakage_reason,
                        normalize_text, qa_pair_from_record)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEFAULT_INPUTS = [
    DATA / "train_qa_verified_leakfree_v2.jsonl",
    DATA / "train_qa_manual_v1.jsonl",
    DATA / "rope_resource_qa_v1.jsonl",
    DATA / "rope_resource_factual_qa_v1.jsonl",
]
DEFAULT_OUTPUT = DATA / "train_qa_curated_v1.jsonl"
DEFAULT_REPORT = DATA / "train_qa_curated_v1.report.json"
DEFAULT_EVAL = [DATA / "eval_qa.jsonl", DATA / "eval_qa_v2.jsonl"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def merge(input_paths: list[Path], output_path: Path, report_path: Path,
          facts) -> dict:
    rows = []
    questions = {}
    pairs = {}
    fact_ids = {}
    counts_by_input = collections.Counter()
    counts_by_kind = collections.Counter()
    counts_by_source = collections.Counter()
    exclusions = []
    exclusion_reasons = collections.Counter()
    for path in input_paths:
        with path.open() as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                location = f"{path}:{line_number}"
                item = json.loads(line)
                try:
                    pair = qa_pair_from_record(item)
                except ValueError as exc:
                    raise ValueError(f"{location}: {exc}") from exc
                if pair is None:
                    raise ValueError(f"{location}: unsupported QA serialization")
                question, answer = pair
                if has_protocol_tokens(question) or has_protocol_tokens(answer):
                    raise ValueError(f"{location}: protocol token in QA")
                if LOW_VALUE.search(question):
                    reason = "low_value_or_time_sensitive"
                    exclusion_reasons[reason] += 1
                    exclusions.append({
                        "fact_id": item.get("fact_id"),
                        "input": portable_path(path),
                        "line": line_number,
                        "reason": reason,
                    })
                    continue
                leak = leakage_reason(question, answer, facts)
                if leak:
                    exclusion_reasons[leak] += 1
                    exclusions.append({
                        "fact_id": item.get("fact_id"),
                        "input": portable_path(path),
                        "line": line_number,
                        "reason": leak,
                    })
                    continue
                question_key = normalize_text(question)
                pair_key = (question_key, normalize_text(answer))
                if question_key in questions:
                    raise ValueError(
                        f"{location}: duplicate question also at {questions[question_key]}")
                if pair_key in pairs:
                    raise ValueError(
                        f"{location}: duplicate QA pair also at {pairs[pair_key]}")
                fact_id = item.get("fact_id")
                if not isinstance(fact_id, str) or not fact_id:
                    raise ValueError(f"{location}: missing fact_id")
                if fact_id in fact_ids:
                    raise ValueError(
                        f"{location}: duplicate fact_id also at {fact_ids[fact_id]}")
                questions[question_key] = location
                pairs[pair_key] = location
                fact_ids[fact_id] = location
                rows.append(item)
                counts_by_input[portable_path(path)] += 1
                counts_by_kind[item.get("kind", "unknown")] += 1
                counts_by_source[item.get("source", "unknown")] += 1

    rows.sort(key=lambda item: (
        normalize_text(qa_pair_from_record(item)[0]), item["fact_id"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output:
        for item in rows:
            output.write(json.dumps(item, ensure_ascii=False,
                                    sort_keys=True) + "\n")
    report = {
        "schema": "curated-training-qa-report-v1",
        "inputs": [portable_path(path) for path in input_paths],
        "input_sha256": {
            portable_path(path): file_sha256(path) for path in input_paths
        },
        "eval_fact_count": len(facts),
        "exclusions": exclusions,
        "output": portable_path(output_path),
        "output_sha256": file_sha256(output_path),
        "counts": {
            "by_input": dict(sorted(counts_by_input.items())),
            "by_kind": dict(sorted(counts_by_kind.items())),
            "by_source": dict(sorted(counts_by_source.items())),
            "excluded": len(exclusions),
            "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
            "output": len(rows),
            "unique_fact_ids": len(fact_ids),
            "unique_questions": len(questions),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path,
                        default=DEFAULT_INPUTS)
    parser.add_argument("--eval", nargs="+", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = merge(args.inputs, args.output, args.report, eval_facts(args.eval))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
