#!/usr/bin/env python3
"""Convert the curated QA JSONL into ES-at-Scale DatasetDict artifacts."""

import argparse
import json
import shutil
from pathlib import Path

from datasets import Dataset, DatasetDict

from qa_quality import qa_pair_from_record


def load_training_rows(path):
    rows = []
    with Path(path).open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            pair = qa_pair_from_record(record)
            if pair is None:
                raise ValueError(
                    f"{path}, line {line_number}: unsupported QA record"
                )
            question, answer = pair
            rows.append({
                "question": question,
                "answer": answer,
                "fact_id": record.get("fact_id", f"line-{line_number}"),
            })
    if not rows:
        raise ValueError(f"{path}: no training rows")
    return rows


def load_eval_rows(path):
    splits = {}
    with Path(path).open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                split = record["split"]
                question = record["question"]
                answer = record["answer"]
            except KeyError as exc:
                raise ValueError(
                    f"{path}, line {line_number}: missing {exc.args[0]}"
                ) from exc
            splits.setdefault(split, []).append({
                "question": question,
                "answer": answer,
                "item_id": record.get("item_id", f"line-{line_number}"),
            })
    if not splits:
        raise ValueError(f"{path}: no evaluation rows")
    return splits


def build(train_jsonl, eval_jsonl, output):
    output = Path(output)
    train_rows = load_training_rows(train_jsonl)
    eval_splits = load_eval_rows(eval_jsonl)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    DatasetDict({"train": Dataset.from_list(train_rows)}).save_to_disk(
        output / "train"
    )
    DatasetDict({
        split: Dataset.from_list(rows)
        for split, rows in sorted(eval_splits.items())
    }).save_to_disk(output / "eval")
    manifest = {
        "schema": "eggroll-es-specialist-dataset-v1",
        "train_jsonl": str(Path(train_jsonl).resolve()),
        "eval_jsonl": str(Path(eval_jsonl).resolve()),
        "train_rows": len(train_rows),
        "eval_splits": {
            split: len(rows) for split, rows in sorted(eval_splits.items())
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-jsonl", type=Path,
        default=Path("data/train_qa_curated_v1.jsonl"),
    )
    parser.add_argument(
        "--eval-jsonl", type=Path, default=Path("data/eval_qa_v2.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/eggroll_es_specialist")
    )
    args = parser.parse_args()
    print(json.dumps(build(args.train_jsonl, args.eval_jsonl, args.output),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
