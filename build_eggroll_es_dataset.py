#!/usr/bin/env python3
"""Convert the curated QA JSONL into ES-at-Scale DatasetDict artifacts."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from datasets import Dataset, DatasetDict

from qa_quality import qa_pair_from_record


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def eval_paths(value):
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(path) for path in value]


def load_eval_inputs(paths):
    combined = {}
    item_ids = set()
    for path in eval_paths(paths):
        for split, rows in load_eval_rows(path).items():
            destination = combined.setdefault(split, [])
            for row in rows:
                item_id = row["item_id"]
                if item_id in item_ids:
                    raise ValueError(
                        f"duplicate evaluation item_id {item_id}")
                item_ids.add(item_id)
                destination.append(row)
    return combined


def build(train_jsonl, eval_jsonl, output):
    output = Path(output)
    train_rows = load_training_rows(train_jsonl)
    paths = eval_paths(eval_jsonl)
    eval_splits = load_eval_inputs(paths)
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
    eval_inputs = [
        {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
        }
        for path in paths
    ]
    manifest = {
        "schema": "eggroll-es-specialist-dataset-v1",
        "train_jsonl": str(Path(train_jsonl).resolve()),
        "train_jsonl_sha256": file_sha256(train_jsonl),
        "eval_inputs": eval_inputs,
        "eval_jsonl": (
            eval_inputs[0]["path"] if len(eval_inputs) == 1
            else [item["path"] for item in eval_inputs]
        ),
        "eval_jsonl_sha256": (
            eval_inputs[0]["sha256"] if len(eval_inputs) == 1
            else [item["sha256"] for item in eval_inputs]
        ),
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
        "--eval-jsonl", type=Path, nargs="+",
        default=[Path("data/eval_qa_v2.jsonl")],
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/eggroll_es_specialist")
    )
    args = parser.parse_args()
    print(json.dumps(build(args.train_jsonl, args.eval_jsonl, args.output),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
