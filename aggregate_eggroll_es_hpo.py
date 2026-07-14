#!/usr/bin/env python3
"""Select an ES candidate across seeds with non-degradation guard splits."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def scores(item, selection_split):
    return item.get("evaluation_scores", {
        selection_split: item["validation_score"],
    })


def canonical_sha256(value):
    rendered = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(rendered.encode()).hexdigest()


def dataset_identity(journal):
    dataset = journal["dataset"]
    # New journals pin the complete train/eval/manifest snapshot. Retain a
    # narrow fallback so historical journals can still be aggregated together,
    # but never let one train hash disguise different guarded eval artifacts.
    if "snapshot" in dataset:
        return {"snapshot_sha256": canonical_sha256(dataset["snapshot"])}
    return {"train_arrow_sha256": dataset["train_arrow_sha256"]}


def candidate_identity(item):
    return {
        key: item[key]
        for key in ("name", "sigma", "alpha", "steps")
        if key in item
    }


def aggregate(journals, selection_split, guard_splits,
              max_guard_degradation=0.0, min_positive_seed_fraction=0.6,
              risk_penalty=0.5):
    if not journals:
        raise ValueError("at least one seed journal is required")
    if not 0 <= min_positive_seed_fraction <= 1:
        raise ValueError("min_positive_seed_fraction must be in [0, 1]")
    if max_guard_degradation < 0 or risk_penalty < 0:
        raise ValueError(
            "guard degradation and risk penalty must be nonnegative")
    if len(set(guard_splits)) != len(guard_splits):
        raise ValueError("guard_splits contains duplicates")
    if selection_split in guard_splits:
        raise ValueError("selection split cannot also be a guard split")

    dataset_identities = {
        canonical_sha256(dataset_identity(journal)) for journal in journals
    }
    trainer_hashes = {journal["trainer_sha256"] for journal in journals}
    if len(dataset_identities) != 1 or len(trainer_hashes) != 1:
        raise ValueError("seed journals do not share dataset/trainer hashes")
    seeds = [journal["seed"] for journal in journals]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed journals contain duplicate seeds")
    candidate_sets = []
    for journal in journals:
        identities = {
            canonical_sha256(candidate_identity(item)): item["name"]
            for item in journal["results"]
        }
        if len(identities) != len(journal["results"]):
            raise ValueError("a seed journal contains duplicate candidates")
        candidate_sets.append(identities)
    if any(set(items) != set(candidate_sets[0])
           for items in candidate_sets[1:]):
        raise ValueError(
            "seed journals do not contain identical candidate settings")

    candidates = []
    for identity_hash, name in sorted(
            candidate_sets[0].items(), key=lambda item: item[1]):
        per_seed = []
        for journal in journals:
            result = next(
                item for item in journal["results"]
                if canonical_sha256(candidate_identity(item)) == identity_hash
            )
            baseline_scores = scores(journal["baseline"], selection_split)
            treatment_scores = scores(result, selection_split)
            deltas = {}
            for split in [selection_split, *guard_splits]:
                try:
                    baseline_score = float(baseline_scores[split])
                    treatment_score = float(treatment_scores[split])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"seed {journal['seed']} lacks numeric {split!r} "
                        "scores") from exc
                deltas[split] = treatment_score - baseline_score
            per_seed.append({
                "seed": journal["seed"],
                "baseline_scores": baseline_scores,
                "treatment_scores": treatment_scores,
                "deltas": deltas,
                "guard_passed": all(
                    deltas[split] >= -max_guard_degradation
                    for split in guard_splits
                ),
            })
        selection_deltas = [
            item["deltas"][selection_split] for item in per_seed
        ]
        mean_delta = statistics.mean(selection_deltas)
        std_delta = statistics.pstdev(selection_deltas)
        robust_score = mean_delta - risk_penalty * std_delta
        positive_fraction = sum(
            delta > 0 for delta in selection_deltas
        ) / len(selection_deltas)
        eligible = (
            mean_delta > 0
            and robust_score > 0
            and positive_fraction >= min_positive_seed_fraction
            and all(item["guard_passed"] for item in per_seed)
        )
        candidates.append({
            "name": name,
            "per_seed": per_seed,
            "mean_selection_delta": mean_delta,
            "median_selection_delta": statistics.median(selection_deltas),
            "std_selection_delta": std_delta,
            "min_selection_delta": min(selection_deltas),
            "max_selection_delta": max(selection_deltas),
            "positive_seed_fraction": positive_fraction,
            "robust_score": robust_score,
            "eligible": eligible,
        })
    eligible = [item for item in candidates if item["eligible"]]
    selected = (
        max(eligible, key=lambda item: item["robust_score"])["name"]
        if eligible else "baseline"
    )
    return {
        "schema": "eggroll-es-multiseed-selection-v1",
        "selection_split": selection_split,
        "guard_splits": guard_splits,
        "max_guard_degradation": max_guard_degradation,
        "min_positive_seed_fraction": min_positive_seed_fraction,
        "risk_penalty": risk_penalty,
        "seeds": seeds,
        "dataset_identity": dataset_identity(journals[0]),
        "dataset_train_arrow_sha256": journals[0]["dataset"].get(
            "train_arrow_sha256"
        ),
        "trainer_sha256": next(iter(trainer_hashes)),
        "candidates": candidates,
        "selected": selected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("journals", type=Path, nargs="+")
    parser.add_argument("--selection-split", default="validation")
    parser.add_argument("--guard-splits", default="ood_qa")
    parser.add_argument("--max-guard-degradation", type=float, default=0.0)
    parser.add_argument("--min-positive-seed-fraction", type=float,
                        default=0.6)
    parser.add_argument("--risk-penalty", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    journals = [json.loads(path.read_text()) for path in args.journals]
    result = aggregate(
        journals, args.selection_split,
        [item.strip() for item in args.guard_splits.split(",")
         if item.strip()],
        args.max_guard_degradation, args.min_positive_seed_fraction,
        args.risk_penalty,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
