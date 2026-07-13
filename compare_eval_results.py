#!/usr/bin/env python3
"""Paired comparison of two eval JSONL files with bootstrap CI and McNemar."""
import argparse
import json
import math
import random
from pathlib import Path


def load(path):
    rows = [json.loads(line) for line in open(path)]
    return {row.get("item_id") or row["q"]: row for row in rows}


def exact_binomial_two_sided(successes, trials):
    if trials == 0:
        return 1.0
    tail = min(successes, trials - successes)
    probability = sum(math.comb(trials, k) for k in range(tail + 1)) / (2 ** trials)
    return min(1.0, 2 * probability)


def percentile(values, probability):
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def compare(left, right, bootstrap_samples=10000, seed=20260713):
    left_rows, right_rows = load(left), load(right)
    ids = sorted(set(left_rows) & set(right_rows))
    pairs = [(bool(left_rows[key]["ok"]), bool(right_rows[key]["ok"]))
             for key in ids if left_rows[key].get("q") == right_rows[key].get("q")]
    if not pairs:
        raise ValueError("result files have no aligned items")
    left_only = sum(a and not b for a, b in pairs)
    right_only = sum(b and not a for a, b in pairs)
    deltas = [float(b) - float(a) for a, b in pairs]
    rng = random.Random(seed)
    bootstraps = []
    for _ in range(bootstrap_samples):
        bootstraps.append(sum(deltas[rng.randrange(len(deltas))]
                              for _ in deltas) / len(deltas))
    return {
        "schema": "paired-eval-comparison-v1",
        "left": str(Path(left).resolve()),
        "right": str(Path(right).resolve()),
        "aligned_items": len(pairs),
        "left_accuracy": sum(a for a, _ in pairs) / len(pairs),
        "right_accuracy": sum(b for _, b in pairs) / len(pairs),
        "right_minus_left": sum(deltas) / len(deltas),
        "paired_bootstrap_95_ci": [
            percentile(bootstraps, 0.025), percentile(bootstraps, 0.975)],
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "mcnemar_exact_two_sided_p": exact_binomial_two_sided(
            left_only, left_only + right_only),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.left, args.right, args.bootstrap_samples, args.seed)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
