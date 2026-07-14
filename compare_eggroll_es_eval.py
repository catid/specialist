#!/usr/bin/env python3
"""Compare aligned ES-at-Scale evaluation arrays with paired statistics."""

import argparse
import hashlib
import json
import random
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path):
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a non-empty JSON array")
    required = {"prompt", "answer", "reward", "format"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required <= row.keys():
            missing = (
                required - set(row) if isinstance(row, dict) else required
            )
            raise ValueError(
                f"{path}, row {index}: missing {sorted(missing)}"
            )
    return rows


def percentile(values, probability):
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def compare(left_path, right_path, bootstrap_samples=20000, seed=20260714):
    left = load_rows(left_path)
    right = load_rows(right_path)
    if len(left) != len(right):
        raise ValueError("evaluation arrays have different lengths")

    pairs = []
    paired_items = []
    for index, (left_row, right_row) in enumerate(zip(left, right)):
        left_key = (left_row["prompt"], left_row["answer"])
        right_key = (right_row["prompt"], right_row["answer"])
        if left_key != right_key:
            raise ValueError(f"evaluation arrays diverge at row {index}")
        left_reward = float(left_row["reward"])
        right_reward = float(right_row["reward"])
        pairs.append((left_reward, right_reward))
        key = json.dumps(left_key, ensure_ascii=False, separators=(",", ":"))
        paired_items.append({
            "key_sha256": hashlib.sha256(key.encode()).hexdigest(),
            "left_exact": left_row["format"] == "exact",
            "left_reward": left_reward,
            "right_exact": right_row["format"] == "exact",
            "right_reward": right_reward,
        })

    deltas = [
        right_reward - left_reward for left_reward, right_reward in pairs
    ]
    rng = random.Random(seed)
    bootstraps = [
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(bootstrap_samples)
    ]
    left_rewards = [left_reward for left_reward, _ in pairs]
    right_rewards = [right_reward for _, right_reward in pairs]
    left_exact = sum(row["format"] == "exact" for row in left)
    right_exact = sum(row["format"] == "exact" for row in right)

    return {
        "schema": "eggroll-es-paired-eval-v2",
        "left": {
            "path": str(Path(left_path)),
            "sha256": file_sha256(left_path),
            "mean_reward": sum(left_rewards) / len(pairs),
            "exact": left_exact,
            "nonzero": sum(reward > 0 for reward in left_rewards),
        },
        "right": {
            "path": str(Path(right_path)),
            "sha256": file_sha256(right_path),
            "mean_reward": sum(right_rewards) / len(pairs),
            "exact": right_exact,
            "nonzero": sum(reward > 0 for reward in right_rewards),
        },
        "aligned_items": len(pairs),
        "right_minus_left_mean_reward": sum(deltas) / len(deltas),
        "paired_bootstrap_95_ci": [
            percentile(bootstraps, 0.025),
            percentile(bootstraps, 0.975),
        ],
        "right_wins": sum(delta > 0 for delta in deltas),
        "left_wins": sum(delta < 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "right_minus_left_exact": right_exact - left_exact,
        "bootstrap_samples": bootstrap_samples,
        # Retaining compact aligned rewards makes the paired statistics
        # independently auditable even when bulky raw model generations are
        # intentionally ignored.  Prompt/answer content is represented by a
        # stable hash so alignment can still be checked without duplicating it.
        "paired_items": paired_items,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.left, args.right, args.bootstrap_samples, args.seed)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
