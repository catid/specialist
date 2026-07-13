#!/usr/bin/env python3
"""Validate manual QA decisions and emit deterministic training records."""
from __future__ import annotations

import argparse
import collections
import hashlib
import html
import json
import re
from pathlib import Path

from build_leakfree_qa import eval_facts
from qa_quality import (LOW_VALUE, has_protocol_tokens, leakage_reason,
                        normalize_text, parse_qa, stable_fact_id)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ACTIONS = {"keep", "edit", "drop", "add"}
CONTEXT_DEPENDENT = re.compile(
    r"\b(?:the|this) (?:article|excerpt|passage|text|context|source)\b|"
    r"\b(?:in (?:the|this) context|described as|mentioned above|"
    r"according to (?:the|this) (?:article|excerpt|passage|text|source))\b",
    re.IGNORECASE,
)


def read_jsonl(path: Path):
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                yield line_number, json.loads(line)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_whitespace(value: str) -> str:
    return " ".join(html.unescape(value).split()).casefold()


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def load_documents(raw_dir: Path):
    documents = {}
    for path in sorted(raw_dir.glob("*.json")):
        item = json.loads(path.read_text())
        url, text = item.get("url", ""), item.get("text", "")
        if url and text:
            if url in documents:
                raise ValueError(f"duplicate raw source URL: {url}")
            documents[url] = (path, item)
    return documents


def _require_string(decision: dict, field: str, location: str) -> str:
    value = decision.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: missing non-empty {field}")
    return value.strip()


def _validate_qa(question: str, answer: str, evidence: str, source_text: str,
                 facts, location: str):
    if not question.endswith("?"):
        raise ValueError(f"{location}: question must end with '?'")
    if not (10 <= len(question) <= 300 and 1 <= len(answer) <= 250):
        raise ValueError(f"{location}: question or answer has invalid length")
    if "\n" in question or "\n" in answer:
        raise ValueError(f"{location}: question and answer must be single-line")
    if has_protocol_tokens(question) or has_protocol_tokens(answer):
        raise ValueError(f"{location}: protocol token in manual QA")
    if CONTEXT_DEPENDENT.search(question):
        raise ValueError(f"{location}: question is context-dependent")
    if LOW_VALUE.search(question):
        raise ValueError(f"{location}: low-value or time-sensitive question")
    if compact_whitespace(evidence) not in compact_whitespace(source_text):
        raise ValueError(f"{location}: evidence is not a source-text excerpt")
    if normalize_text(answer) not in normalize_text(evidence):
        raise ValueError(f"{location}: answer must be extractive from evidence")
    leak = leakage_reason(question, answer, facts)
    if leak:
        raise ValueError(f"{location}: evaluation leakage ({leak})")
    rendered = f"Question: {question}\nAnswer: {answer}"
    if parse_qa(rendered) != (question, answer):
        raise ValueError(f"{location}: QA does not round-trip through parser")


def build(candidates_path: Path, review_paths: list[Path], raw_dir: Path,
          output_path: Path, facts, reviewed_candidates_path: Path | None = None):
    candidates = {}
    candidates_by_url = collections.defaultdict(set)
    for line_number, item in read_jsonl(candidates_path):
        fact_id = item.get("fact_id")
        if not fact_id or fact_id in candidates:
            raise ValueError(
                f"{candidates_path}:{line_number}: missing/duplicate fact_id")
        candidates[fact_id] = item
        candidates_by_url[item.get("url", "")].add(fact_id)

    documents = load_documents(raw_dir)
    covered = {}
    reviewed_urls = set()
    outputs = []
    counts = collections.Counter()
    reasons = collections.Counter()
    for review_path in sorted(review_paths):
        for line_number, decision in read_jsonl(review_path):
            location = f"{review_path}:{line_number}"
            action = decision.get("action")
            if action not in ACTIONS:
                raise ValueError(f"{location}: unsupported action {action!r}")
            ids = decision.get("candidate_fact_ids")
            if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
                raise ValueError(f"{location}: candidate_fact_ids must be a list")
            if action in {"keep", "edit", "drop"} and not ids:
                raise ValueError(f"{location}: {action} requires candidate IDs")
            if action == "keep" and len(ids) != 1:
                raise ValueError(f"{location}: keep requires exactly one candidate")
            if action == "add" and ids:
                raise ValueError(f"{location}: add cannot consume candidate IDs")
            for fact_id in ids:
                if fact_id not in candidates:
                    raise ValueError(f"{location}: unknown candidate {fact_id}")
                if fact_id in covered:
                    raise ValueError(
                        f"{location}: {fact_id} already consumed at {covered[fact_id]}")
                covered[fact_id] = location

            url = _require_string(decision, "url", location)
            source = _require_string(decision, "source", location)
            reviewer = _require_string(decision, "reviewer", location)
            batch = _require_string(decision, "batch", location)
            reason = _require_string(decision, "reason", location)
            reviewed_urls.add(url)
            if url not in documents:
                raise ValueError(f"{location}: source URL has no raw document")
            source_path, document = documents[url]
            if source != document.get("source", ""):
                raise ValueError(f"{location}: source label disagrees with raw document")
            for fact_id in ids:
                candidate = candidates[fact_id]
                if candidate.get("url") != url or candidate.get("source") != source:
                    raise ValueError(f"{location}: candidate comes from another source")

            counts[action] += 1
            counts["candidate_ids_" + action] += len(ids)
            reasons[reason] += 1
            if action == "drop":
                continue

            question = _require_string(decision, "question", location)
            answer = _require_string(decision, "answer", location)
            evidence = _require_string(decision, "evidence", location)
            if action == "keep":
                candidate = candidates[ids[0]]
                if (normalize_text(question), normalize_text(answer)) != (
                        normalize_text(candidate["question"]),
                        normalize_text(candidate["answer"])):
                    raise ValueError(f"{location}: keep changes its candidate QA")
            _validate_qa(question, answer, evidence, document["text"], facts,
                         location)
            outputs.append({
                "answer": answer,
                "document_sha256": hashlib.sha256(
                    document["text"].encode()).hexdigest(),
                "evidence": evidence,
                "fact_id": stable_fact_id(question, answer),
                "kind": "qa_manual",
                "quality_schema": "manual-qa-v1",
                "question": question,
                "review": {
                    "action": action,
                    "batch": batch,
                    "candidate_fact_ids": ids,
                    "reason": reason,
                    "reviewer": reviewer,
                },
                "source": source,
                "source_lineage": {"raw": portable_path(source_path)},
                "text": f"Question: {question}\nAnswer: {answer}",
                "url": url,
            })

    missing = sorted(
        fact_id for url in reviewed_urls for fact_id in candidates_by_url[url]
        if fact_id not in covered)
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(
            f"manual review does not cover {len(missing)} candidates: {preview}")

    if reviewed_candidates_path is not None:
        if reviewed_candidates_path.resolve() == candidates_path.resolve():
            raise ValueError("reviewed candidate subset cannot overwrite its input")
        reviewed_candidates_path.parent.mkdir(parents=True, exist_ok=True)
        subset = [item for item in candidates.values()
                  if item.get("url") in reviewed_urls]
        subset.sort(key=lambda item: (item.get("url", ""), item["fact_id"]))
        with reviewed_candidates_path.open("w") as destination:
            for item in subset:
                compact = {
                    "answer": item["answer"],
                    "fact_id": item["fact_id"],
                    "kind": "qa_manual_candidate",
                    "quality_schema": "manual-qa-candidate-v1",
                    "question": item["question"],
                    "source": item["source"],
                    "url": item["url"],
                }
                destination.write(
                    json.dumps(compact, ensure_ascii=False, sort_keys=True) + "\n")

    question_seen = {}
    fact_seen = {}
    for item in outputs:
        q_key = normalize_text(item["question"])
        if q_key in question_seen:
            raise ValueError(
                f"duplicate manual question: {item['question']!r}")
        question_seen[q_key] = item["fact_id"]
        if item["fact_id"] in fact_seen:
            raise ValueError(f"duplicate manual fact_id: {item['fact_id']}")
        fact_seen[item["fact_id"]] = item["question"]

    outputs.sort(key=lambda item: (item["url"], normalize_text(item["question"])))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output:
        for item in outputs:
            output.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    counts["reviewed_documents"] = len(reviewed_urls)
    counts["covered_candidates"] = len(covered)
    counts["output"] = len(outputs)
    report = {
        "schema": "manual-qa-report-v1",
        "candidate_input": portable_path(candidates_path),
        "candidate_sha256": file_sha256(candidates_path),
        "reviews": [portable_path(path) for path in sorted(review_paths)],
        "review_sha256": {
            portable_path(path): file_sha256(path)
            for path in sorted(review_paths)
        },
        "output": portable_path(output_path),
        "output_sha256": file_sha256(output_path),
        "counts": dict(sorted(counts.items())),
        "reasons": dict(sorted(reasons.items())),
    }
    if reviewed_candidates_path is not None:
        report["reviewed_candidates"] = portable_path(reviewed_candidates_path)
        report["reviewed_candidates_sha256"] = file_sha256(
            reviewed_candidates_path)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--raw-dir", type=Path, default=DATA / "raw")
    parser.add_argument("--eval", type=Path, nargs="+", default=[
        DATA / "eval_qa.jsonl", DATA / "eval_qa_v2.jsonl",
    ])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reviewed-candidates-output", type=Path,
        help="optionally save the compact candidate subset covered by reviews",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = build(
        args.candidates, args.reviews, args.raw_dir, args.output,
        eval_facts(args.eval), args.reviewed_candidates_output,
    )
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
