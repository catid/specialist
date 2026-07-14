#!/usr/bin/env python3
"""Build the preregistered coefficient-only v7 cross-seed report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import run_eggroll_es_anchor_line_search as driver_v1
import run_eggroll_es_anchor_stability_v7 as stability_v7


ROOT = Path(__file__).resolve().parent
EXPECTED_COVERAGE = {
    (arm, seed)
    for arm in stability_v7.STABILITY_ARMS_V7
    for seed in stability_v7.STABILITY_SEEDS_V7
}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coefficient_cosine(left, right):
    if (
        not isinstance(left, list) or not isinstance(right, list)
        or len(left) != 16 or len(right) != 16
    ):
        raise ValueError("v7 coefficient vectors must each contain 16 values")
    left = [float(value) for value in left]
    right = [float(value) for value in right]
    if not all(math.isfinite(value) for value in (*left, *right)):
        raise ValueError("v7 coefficient vectors contain non-finite values")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("v7 coefficient cosine is undefined for a zero vector")
    return sum(a * b for a, b in zip(left, right)) / (
        left_norm * right_norm
    )


def _load_run(path):
    path = Path(path).resolve()
    stability_v7.offline_audit._assert_no_heldout(
        str(path), "v7 stability journal path",
    )
    try:
        journal = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read v7 stability journal {path}") from error
    audit = stability_v7.validate_completed_journal_v7(journal)
    coefficients = journal.get("coefficient_plan", {}).get("coefficients")
    if not isinstance(coefficients, list) or len(coefficients) != 16:
        raise ValueError("v7 completed journal omitted its 16 coefficients")
    return {
        "arm": audit["arm"],
        "seed": audit["seed"],
        "journal": str(path),
        "journal_file_sha256": file_sha256(path),
        "content_sha256": audit["content_sha256"],
        "coefficient_sha256": audit["coefficient_sha256"],
        "robust_plan_sha256": audit["robust_plan_sha256"],
        "coefficients": list(coefficients),
    }


def build_report(journal_paths):
    runs = [_load_run(path) for path in journal_paths]
    coverage = {(run["arm"], run["seed"]) for run in runs}
    if len(runs) != 4 or coverage != EXPECTED_COVERAGE:
        raise ValueError(
            "v7 report requires exactly front/middle_late at seeds 43 and 44"
        )
    if len({run["journal"] for run in runs}) != 4:
        raise ValueError("v7 report cannot reuse a journal path")
    arms = []
    for arm in stability_v7.STABILITY_ARMS_V7:
        pair = sorted(
            (run for run in runs if run["arm"] == arm),
            key=lambda run: run["seed"],
        )
        cosine = coefficient_cosine(
            pair[0]["coefficients"], pair[1]["coefficients"],
        )
        public_runs = [
            {key: value for key, value in run.items() if key != "coefficients"}
            for run in pair
        ]
        arms.append({
            "arm": arm,
            "seeds": [43, 44],
            "runs": public_runs,
            "seed_slot_coefficient_cosine": cosine,
            "threshold": stability_v7.COEFFICIENT_COSINE_THRESHOLD_V7,
            "passed": (
                cosine >= stability_v7.COEFFICIENT_COSINE_THRESHOLD_V7
            ),
        })
    implementation = {
        "reporter_v7": file_sha256(Path(__file__).resolve()),
        "stability_driver_v7": file_sha256(
            Path(stability_v7.__file__).resolve()
        ),
    }
    report = {
        "schema": "eggroll-es-direction-stability-report-v7",
        "metric": "seed_slot_coefficient_cosine",
        "metric_interpretation": (
            "cosine of projected ES coefficient vectors in population-slot "
            "order; this is a response-shape stability screen, not a "
            "parameter-space update cosine across different perturbation seeds"
        ),
        "preregistered_threshold": 0.5,
        "coverage": {
            "arms": ["front", "middle_late"],
            "seeds": [43, 44],
            "target_alphas": [0.0],
        },
        "selection_policy": (
            "coefficient_only_no_validation_ood_or_holdout_metrics_in_report"
        ),
        "arms": arms,
        "all_arms_passed": all(item["passed"] for item in arms),
        "implementation": implementation,
        "implementation_bundle_sha256": driver_v1.canonical_sha256(
            implementation
        ),
    }
    report["content_sha256_before_self_field"] = (
        driver_v1.canonical_sha256(report)
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    output_path = Path(args.output_json).resolve()
    stability_v7.offline_audit._assert_no_heldout(
        str(output_path), "v7 stability report output path",
    )
    if output_path.exists() or output_path.with_name(
        output_path.name + ".tmp"
    ).exists():
        raise ValueError("v7 stability report output already exists")
    report = build_report(args.journal)
    driver_v1.atomic_write_json(output_path, report)
    print(json.dumps({
        "output": str(output_path),
        "content_sha256": report["content_sha256_before_self_field"],
        "all_arms_passed": report["all_arms_passed"],
    }, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
