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
                        normalize_text, parse_qa, qa_pair_from_record,
                        stable_fact_id)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEFAULT_INPUTS = [
    DATA / "train_qa_verified_leakfree_v2.jsonl",
    DATA / "train_qa_manual_v1.jsonl",
    DATA / "rope_resource_qa_v1.jsonl",
    DATA / "rope_resource_factual_qa_v1.jsonl",
    DATA / "rope_resource_manual_v1.jsonl",
    DATA / "rope_topia_manual_v1.jsonl",
]
DEFAULT_OUTPUT = DATA / "train_qa_curated_v1.jsonl"
DEFAULT_REPORT = DATA / "train_qa_curated_v1.report.json"
DEFAULT_CURATIONS = [
    DATA / "train_qa_curated_v1.curation.jsonl",
    DATA / "train_qa_kinbakutoday.curation.jsonl",
]
DEFAULT_EVAL = [DATA / "eval_qa.jsonl", DATA / "eval_qa_v2.jsonl"]
CURATION_ACTIONS = {"drop", "edit"}


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


def require_text(item: dict, field: str, location: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: missing non-empty {field}")
    return value.strip()


def curation_paths(value: Path | list[Path] | None) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, Path):
        return [value]
    return list(value)


def load_curation(
        paths: Path | list[Path] | None,
) -> tuple[dict[str, dict], dict[str, Path]]:
    decisions = {}
    sources = {}
    for path in curation_paths(paths):
        with path.open() as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                location = "{}:{}".format(path, line_number)
                item = json.loads(line)
                action = require_text(item, "action", location)
                fact_id = require_text(item, "fact_id", location)
                require_text(item, "expected_question", location)
                require_text(item, "expected_answer", location)
                require_text(item, "reason", location)
                require_text(item, "reason_code", location)
                require_text(item, "reviewer", location)
                require_text(item, "reviewed_at", location)
                if action not in CURATION_ACTIONS:
                    raise ValueError(
                        f"{location}: unsupported action {action!r}")
                if fact_id in decisions:
                    raise ValueError(
                        f"{location}: duplicate curation fact_id {fact_id}")
                if action == "edit":
                    question = require_text(item, "question", location)
                    answer = require_text(item, "answer", location)
                    evidence = require_text(item, "evidence", location)
                    require_text(item, "evidence_url", location)
                    if not question.endswith("?"):
                        raise ValueError(
                            f"{location}: edited question must end with '?'")
                    if "\n" in question or "\n" in answer:
                        raise ValueError(
                            f"{location}: edited QA must be one line")
                    if has_protocol_tokens(
                            question) or has_protocol_tokens(answer):
                        raise ValueError(
                            f"{location}: protocol token in edited QA")
                    if normalize_text(answer) not in normalize_text(evidence):
                        raise ValueError(
                            f"{location}: edited answer must be extractive "
                            "from evidence"
                        )
                    rendered = f"Question: {question}\nAnswer: {answer}"
                    if parse_qa(rendered) != (question, answer):
                        raise ValueError(
                            f"{location}: edited QA does not round-trip")
                decisions[fact_id] = item
                sources[fact_id] = path
    return decisions, sources


def merge(input_paths: list[Path], output_path: Path, report_path: Path,
          facts, curation_path: Path | list[Path] | None = None) -> dict:
    paths = curation_paths(curation_path)
    decisions, decision_sources = load_curation(paths)
    applied_decisions = set()
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
                location = "{}:{}".format(path, line_number)
                item = json.loads(line)
                try:
                    pair = qa_pair_from_record(item)
                except ValueError as exc:
                    raise ValueError(f"{location}: {exc}") from exc
                if pair is None:
                    raise ValueError(
                        f"{location}: unsupported QA serialization")
                question, answer = pair
                original_fact_id = item.get("fact_id")
                decision = decisions.get(original_fact_id)
                if decision is not None:
                    if (question != decision["expected_question"] or
                            answer != decision["expected_answer"]):
                        raise ValueError(
                            f"{location}: stale curation decision for "
                            f"{original_fact_id}")
                    source_evidence_urls = {
                        item.get("url"),
                        item.get("evidence_url"),
                        item.get("url_evidence_url"),
                    }
                    if decision.get(
                            "evidence_url", item.get("url")
                    ) not in source_evidence_urls:
                        raise ValueError(
                            f"{location}: curation evidence URL disagrees "
                            "with source"
                        )
                    applied_decisions.add(original_fact_id)
                    action = decision["action"]
                    reason_code = decision["reason_code"]
                    if action == "drop":
                        reason = "manual_curation:{}".format(reason_code)
                        exclusion_reasons[reason] += 1
                        exclusions.append({
                            "fact_id": original_fact_id,
                            "input": portable_path(path),
                            "line": line_number,
                            "reason": reason,
                        })
                        continue
                    question = decision["question"].strip()
                    answer = decision["answer"].strip()
                    item = dict(item)
                    item.update({
                        "answer": answer,
                        "curation": {
                            "action": "edit",
                            "decision_file": portable_path(
                                decision_sources[original_fact_id]),
                            "original_fact_id": original_fact_id,
                            "reason": decision["reason"],
                            "reason_code": reason_code,
                            "reviewed_at": decision["reviewed_at"],
                            "reviewer": decision["reviewer"],
                        },
                        "evidence": decision["evidence"].strip(),
                        "evidence_url": decision["evidence_url"].strip(),
                        "fact_id": stable_fact_id(question, answer),
                        "quality_schema": "curated-qa-v1",
                        "question": question,
                        "text": f"Question: {question}\nAnswer: {answer}",
                    })
                # The source tranches contain several legacy prompt renderings,
                # including generator instructions.  Structured Q/A is the
                # validated contract, so render every accepted row canonically.
                item = dict(item)
                item.update({
                    "answer": answer,
                    "question": question,
                    "text": f"Question: {question}\nAnswer: {answer}",
                })
                if has_protocol_tokens(
                        question) or has_protocol_tokens(answer):
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
                        f"{location}: duplicate question also at "
                        f"{questions[question_key]}"
                    )
                if pair_key in pairs:
                    raise ValueError(
                        f"{location}: duplicate QA pair also at "
                        f"{pairs[pair_key]}"
                    )
                fact_id = item.get("fact_id")
                if not isinstance(fact_id, str) or not fact_id:
                    raise ValueError(f"{location}: missing fact_id")
                if fact_id in fact_ids:
                    raise ValueError(
                        f"{location}: duplicate fact_id also at "
                        f"{fact_ids[fact_id]}"
                    )
                questions[question_key] = location
                pairs[pair_key] = location
                fact_ids[fact_id] = location
                rows.append(item)
                counts_by_input[portable_path(path)] += 1
                counts_by_kind[item.get("kind", "unknown")] += 1
                counts_by_source[item.get("source", "unknown")] += 1

    unused_decisions = sorted(set(decisions) - applied_decisions)
    if unused_decisions:
        raise ValueError(
            "curation decisions did not match an input fact: " +
            ", ".join(unused_decisions))

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
        "curation": {
            "artifacts": [
                {
                    "path": portable_path(path),
                    "sha256": file_sha256(path),
                }
                for path in paths
            ],
            "by_action": dict(sorted(collections.Counter(
                decision["action"]
                for decision in decisions.values()).items())),
            "by_reason": dict(sorted(collections.Counter(
                decision["reason_code"]
                for decision in decisions.values()).items())),
            "decisions": len(decisions),
        },
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
    parser.add_argument("--curation", nargs="*", type=Path,
                        default=DEFAULT_CURATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = merge(
        args.inputs, args.output, args.report, eval_facts(args.eval),
        args.curation,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
