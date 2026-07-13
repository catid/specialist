#!/usr/bin/env python3
"""Build a deterministic, provenance-carrying QA set with eval leakage removed.

This is a hard pre-training gate, not an LLM judge.  It removes exact and
near-duplicate questions, same-entity/same-answer fact paraphrases, conflicting
answers, duplicate rendered examples, and obviously time-sensitive commerce
trivia.  Existing eval-v2 remains development-only because models have already
trained on leaked versions of some of its facts.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

from qa_quality import (EvalFact, LOW_VALUE, leakage_reason, normalize_text,
                        qa_pair_from_record, stable_fact_id)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def read_jsonl(path: Path):
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                yield line_number, json.loads(line)


def eval_facts(paths):
    facts = []
    for path in paths:
        for line_number, item in read_jsonl(path):
            question, answer = item["question"], item["answer"]
            item_id = item.get("item_id") or stable_fact_id(
                question, answer, f"{path.name}:{line_number}")
            facts.append(EvalFact(question, answer, item_id, item.get("split", "")))
    return facts


def raw_document_hashes(raw_dir: Path):
    hashes = {}
    if not raw_dir.exists():
        return hashes
    for path in sorted(raw_dir.glob("*.json")):
        try:
            item = json.loads(path.read_text())
            text = item.get("text", "")
            url = item.get("url", "")
            if url and text:
                hashes[url] = hashlib.sha256(text.encode()).hexdigest()
        except (OSError, json.JSONDecodeError):
            continue
    return hashes


def build(input_path: Path, output_path: Path, facts, document_hashes):
    counters = collections.Counter()
    candidates = []
    rendered_seen = set()
    for line_number, item in read_jsonl(input_path):
        counters["input"] += 1
        try:
            pair = qa_pair_from_record(item)
        except ValueError as exc:
            raise ValueError(f"{input_path}:{line_number}: {exc}") from exc
        if not pair:
            counters["unparseable"] += 1
            continue
        question, answer = pair
        reason = leakage_reason(question, answer, facts)
        if reason:
            counters[reason] += 1
            continue
        if LOW_VALUE.search(question):
            counters["low_value_or_time_sensitive"] += 1
            continue
        rendered_key = normalize_text(item.get("text", ""))
        if rendered_key in rendered_seen:
            counters["duplicate_rendered_text"] += 1
            continue
        rendered_seen.add(rendered_key)
        candidates.append((line_number, item, question, answer))

    # Equivalent questions with incompatible answers poison both likelihood
    # training and exact/F1 rewards. Drop every member of a conflicting group.
    answers_by_question = collections.defaultdict(set)
    for _line, _item, question, answer in candidates:
        answers_by_question[normalize_text(question)].add(normalize_text(answer))
    conflicts = {question for question, answers in answers_by_question.items()
                 if len(answers) > 1}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pair_seen = set()
    with output_path.open("w") as output:
        for line_number, item, question, answer in candidates:
            if normalize_text(question) in conflicts:
                counters["conflicting_answer"] += 1
                continue
            pair_key = (normalize_text(question), normalize_text(answer))
            if pair_key in pair_seen:
                counters["duplicate_qa_pair"] += 1
                continue
            pair_seen.add(pair_key)
            url = item.get("url", "")
            clean = dict(item)
            clean.update({
                # A fact ID identifies the normalized QA fact, not its source
                # row. Provenance remains in source_lineage/document_sha256.
                "fact_id": stable_fact_id(question, answer),
                "question": question,
                "answer": answer,
                "document_sha256": document_hashes.get(url),
                "quality_schema": "leakfree-qa-v2",
                "source_lineage": {
                    "input": str(input_path.resolve()),
                    "line": line_number,
                },
            })
            output.write(json.dumps(clean, ensure_ascii=False, sort_keys=True) + "\n")
            counters["output"] += 1
    return dict(sorted(counters.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval", type=Path, nargs="+", default=[
        DATA / "eval_qa.jsonl", DATA / "eval_qa_v2.jsonl",
    ])
    parser.add_argument("--raw-dir", type=Path, default=DATA / "raw")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must not overwrite --input")

    facts = eval_facts(args.eval)
    report = {
        "schema": "leakfree-qa-report-v2",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "eval_files": [str(path.resolve()) for path in args.eval],
        "eval_facts": len(facts),
        "counts": build(args.input, args.output, facts,
                        raw_document_hashes(args.raw_dir)),
    }
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
