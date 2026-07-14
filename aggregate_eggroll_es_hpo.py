#!/usr/bin/env python3
"""Select an ES candidate across seeds with non-degradation guard splits."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from run_eggroll_es_hpo import inspect_ood_prose_gate


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


def inspect_journal_ood_prose_policy(journal, expected_max_degradation):
    """Require the seed journal itself to pin the requested prose policy."""
    issues = []
    if journal.get("ood_prose_guard_enabled") is not True:
        issues.append("ood_prose_guard_enabled is not true")
    recorded = journal.get("max_ood_prose_degradation")
    if (
        isinstance(recorded, bool)
        or not isinstance(recorded, (int, float))
        or not math.isfinite(recorded)
    ):
        issues.append("max_ood_prose_degradation is not a finite number")
        recorded = None
    else:
        recorded = float(recorded)
        if recorded < 0:
            issues.append("max_ood_prose_degradation is negative")
        if recorded != expected_max_degradation:
            issues.append(
                "max_ood_prose_degradation does not match expected policy"
            )
    return {
        "valid": not issues,
        "issues": issues,
        "expected_max_degradation": expected_max_degradation,
        "recorded_max_degradation": recorded,
    }


def aggregate(journals, selection_split, guard_splits,
              max_guard_degradation=0.0, min_positive_seed_fraction=0.6,
              risk_penalty=0.5, max_guard_exact_loss=None,
              require_ood_prose_guard=False,
              expected_ood_prose_max_degradation=0.0):
    if not journals:
        raise ValueError("at least one seed journal is required")
    if not 0 <= min_positive_seed_fraction <= 1:
        raise ValueError("min_positive_seed_fraction must be in [0, 1]")
    if max_guard_degradation < 0 or risk_penalty < 0:
        raise ValueError(
            "guard degradation and risk penalty must be nonnegative")
    if max_guard_exact_loss is not None and max_guard_exact_loss < 0:
        raise ValueError("max guard exact loss must be nonnegative")
    if (
        isinstance(expected_ood_prose_max_degradation, bool)
        or not isinstance(
            expected_ood_prose_max_degradation, (int, float)
        )
        or not math.isfinite(expected_ood_prose_max_degradation)
        or expected_ood_prose_max_degradation < 0
    ):
        raise ValueError(
            "expected OOD prose max degradation must be a finite "
            "nonnegative number"
        )
    expected_ood_prose_max_degradation = float(
        expected_ood_prose_max_degradation
    )
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
            mean_guards_passed = all(
                deltas[split] >= -max_guard_degradation
                for split in guard_splits
            )
            if max_guard_exact_loss is None:
                exact_guards_passed = True
            else:
                baseline_details = journal["baseline"].get(
                    "evaluation_details", {})
                treatment_details = result.get("evaluation_details", {})
                exact_guards_passed = all(
                    split in baseline_details
                    and split in treatment_details
                    and treatment_details[split]["exact"] >= (
                        baseline_details[split]["exact"]
                        - max_guard_exact_loss
                    )
                    for split in guard_splits
                )
            if require_ood_prose_guard:
                journal_prose_policy = inspect_journal_ood_prose_policy(
                    journal, expected_ood_prose_max_degradation,
                )
                prose_gate = inspect_ood_prose_gate(
                    result.get("ood_prose_gate"),
                    expected_ood_prose_max_degradation,
                )
                declared_prose_pass = result.get(
                    "ood_prose_guard_passed"
                )
                declared_prose_valid = (
                    isinstance(declared_prose_pass, bool)
                    and declared_prose_pass == prose_gate["policy_passed"]
                )
                prose_guard_passed = (
                    journal_prose_policy["valid"]
                    and prose_gate["policy_passed"]
                    and declared_prose_valid
                )
            else:
                journal_prose_policy = None
                prose_gate = None
                declared_prose_pass = result.get(
                    "ood_prose_guard_passed"
                )
                declared_prose_valid = None
                prose_guard_passed = True
            per_seed.append({
                "seed": journal["seed"],
                "baseline_scores": baseline_scores,
                "treatment_scores": treatment_scores,
                "deltas": deltas,
                "mean_guards_passed": mean_guards_passed,
                "exact_guards_passed": exact_guards_passed,
                "ood_prose_guard_passed": prose_guard_passed,
                "ood_prose_declared_passed": declared_prose_pass,
                "ood_prose_declared_pass_valid": declared_prose_valid,
                "ood_prose_journal_policy": journal_prose_policy,
                "ood_prose_gate": prose_gate,
                "guard_passed": (
                    mean_guards_passed
                    and exact_guards_passed
                    and prose_guard_passed
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
        "max_guard_exact_loss": max_guard_exact_loss,
        "require_ood_prose_guard": require_ood_prose_guard,
        "expected_ood_prose_max_degradation": (
            expected_ood_prose_max_degradation
        ),
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
    parser.add_argument("--max-guard-exact-loss", type=int, default=0)
    parser.add_argument("--require-ood-prose-guard", action="store_true")
    parser.add_argument(
        "--expected-ood-prose-max-degradation", type=float, default=0.0,
        help=(
            "Required per-run OOD prose gate margin; runs recorded with a "
            "different margin are ineligible"
        ),
    )
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
        args.risk_penalty, args.max_guard_exact_loss,
        args.require_ood_prose_guard,
        args.expected_ood_prose_max_degradation,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
