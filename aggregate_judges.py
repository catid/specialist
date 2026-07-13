#!/usr/bin/env python3
"""Aggregate multiple judges over frozen, identical candidate answers."""
import argparse
import itertools
import json
from pathlib import Path


def load(path):
    rows = [json.loads(line) for line in open(path)]
    return {row.get("item_id") or row["q"]: row for row in rows}


def aggregate(paths):
    runs = [load(path) for path in paths]
    ids = sorted(set.intersection(*(set(run) for run in runs)))
    aligned = []
    mismatched_candidates = 0
    for item in ids:
        rows = [run[item] for run in runs]
        if len({row["q"] for row in rows}) != 1 or len({row["cand"] for row in rows}) != 1:
            mismatched_candidates += 1
            continue
        aligned.append(rows)
    pairwise = []
    for left, right in itertools.combinations(range(len(paths)), 2):
        agreements = sum(rows[left]["ok"] == rows[right]["ok"] for rows in aligned)
        pairwise.append({
            "left": str(Path(paths[left]).resolve()),
            "right": str(Path(paths[right]).resolve()),
            "agreement": agreements / max(len(aligned), 1),
            "disagreements": len(aligned) - agreements,
        })
    majority = [sum(bool(row["ok"]) for row in rows) > len(rows) / 2
                for rows in aligned]
    unanimous = sum(len({bool(row["ok"]) for row in rows}) == 1 for rows in aligned)
    return {
        "schema": "multi-judge-aggregate-v1",
        "inputs": [str(Path(path).resolve()) for path in paths],
        "common_items": len(ids),
        "aligned_identical_candidates": len(aligned),
        "candidate_mismatches_excluded": mismatched_candidates,
        "unanimous_fraction": unanimous / max(len(aligned), 1),
        "majority_accuracy": sum(majority) / max(len(majority), 1),
        "pairwise": pairwise,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.inputs) < 2:
        parser.error("at least two judge files are required")
    report = aggregate(args.inputs)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
