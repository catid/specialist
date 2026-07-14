#!/usr/bin/env python3
"""Build the coefficient-only same-basis v8 stability report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import run_eggroll_es_anchor_line_search as driver_v1
import run_eggroll_es_anchor_stability_v8 as stability_v8


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cosine(left, right):
    if (
        not isinstance(left, list) or not isinstance(right, list)
        or len(left) != stability_v8.POPULATION_SIZE_V8
        or len(right) != stability_v8.POPULATION_SIZE_V8
    ):
        raise ValueError("v8 coefficient vectors must each contain 32 values")
    left = [float(value) for value in left]
    right = [float(value) for value in right]
    if not all(math.isfinite(value) for value in (*left, *right)):
        raise ValueError("v8 cosine input contains non-finite values")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("v8 cosine is undefined for a zero vector")
    return sum(a * b for a, b in zip(left, right)) / (
        left_norm * right_norm
    )


def _standardize(values):
    values = [float(value) for value in values]
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    if std == 0.0:
        raise ValueError("v8 score vector has zero variance")
    return [(value - mean) / std for value in values]


def _load_run(path):
    path = Path(path).resolve()
    stability_v8.offline_audit._assert_no_heldout(
        str(path), "v8 stability journal path",
    )
    try:
        journal = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read v8 stability journal {path}") from error
    audit = stability_v8.validate_completed_journal_v8(journal)
    coefficient_plan = journal.get("coefficient_plan", {})
    coefficients = coefficient_plan.get("coefficients")
    domain_scores = coefficient_plan.get("domain_scores_v5")
    anchor_scores = coefficient_plan.get("anchor_scores_v5")
    projection = coefficient_plan.get("projection")
    if (
        not isinstance(coefficients, list)
        or len(coefficients) != stability_v8.POPULATION_SIZE_V8
        or not isinstance(domain_scores, list)
        or len(domain_scores) != stability_v8.POPULATION_SIZE_V8
        or not isinstance(anchor_scores, list)
        or len(anchor_scores) != stability_v8.POPULATION_SIZE_V8
        or not isinstance(projection, dict)
    ):
        raise ValueError("v8 completed journal omitted coefficient geometry")
    geometry = {
        "raw_domain_anchor_cosine": projection["domain_anchor_cosine_before"],
        "projected_anchor_cosine": projection["domain_anchor_cosine_after"],
        "projection_lambda": projection["projection_lambda"],
        "update_norm_ratio": projection["update_norm_ratio"],
        "projected_vs_raw_domain_cosine": cosine(
            coefficients, _standardize(domain_scores),
        ),
        "domain_mean": projection["domain_mean"],
        "domain_std": projection["domain_std"],
        "anchor_mean": projection["anchor_mean"],
        "anchor_std": projection["anchor_std"],
    }
    return {
        "arm": audit["arm"],
        "data_bootstrap_seed": audit["data_bootstrap_seed"],
        "perturbation_basis_sha256": audit["perturbation_basis_sha256"],
        "journal": str(path), "journal_file_sha256": file_sha256(path),
        "content_sha256": audit["content_sha256"],
        "coefficient_sha256": audit["coefficient_sha256"],
        "robust_plan_sha256": audit["robust_plan_sha256"],
        "geometry": geometry,
        "coefficients": list(coefficients),
    }


def build_report(journal_paths):
    runs = [_load_run(path) for path in journal_paths]
    if (
        len(runs) != 2
        or {run["data_bootstrap_seed"] for run in runs} != {43, 44}
        or {run["arm"] for run in runs} != {"middle_late"}
        or len({run["journal"] for run in runs}) != 2
        or {run["perturbation_basis_sha256"] for run in runs}
        != {stability_v8.PERTURBATION_BASIS_SHA256_V8}
    ):
        raise ValueError(
            "v8 report requires middle_late data seeds43/44 on one basis"
        )
    runs.sort(key=lambda run: run["data_bootstrap_seed"])
    coefficient_cosine = cosine(
        runs[0]["coefficients"], runs[1]["coefficients"],
    )
    public_runs = [
        {key: value for key, value in run.items() if key != "coefficients"}
        for run in runs
    ]
    implementation = {
        "reporter_v8": file_sha256(Path(__file__).resolve()),
        "stability_driver_v8": file_sha256(
            Path(stability_v8.__file__).resolve()
        ),
    }
    report = {
        "schema": "eggroll-es-same-basis-direction-stability-report-v8",
        "metric": "same_perturbation_basis_coefficient_cosine",
        "metric_interpretation": (
            "coefficient response-shape reproducibility across data/bootstrap "
            "seeds on one identical 32-perturbation basis"
        ),
        "preregistered_threshold": 0.5,
        "coverage": {
            "arm": "middle_late", "data_bootstrap_seeds": [43, 44],
            "perturbation_basis_seed": (
                stability_v8.PERTURBATION_BASIS_SEED_V8
            ),
            "perturbation_basis_sha256": (
                stability_v8.PERTURBATION_BASIS_SHA256_V8
            ),
            "population_size": 32, "target_alphas": [0.0],
        },
        "selection_policy": (
            "coefficient_and_train_anchor_geometry_only_no_benchmark_selection"
        ),
        "runs": public_runs,
        "same_basis_coefficient_cosine": coefficient_cosine,
        "passed": coefficient_cosine >= 0.5,
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
    output = Path(args.output_json).resolve()
    stability_v8.offline_audit._assert_no_heldout(
        str(output), "v8 stability report output path",
    )
    if output.exists() or output.with_name(output.name + ".tmp").exists():
        raise ValueError("v8 stability report output already exists")
    report = build_report(args.journal)
    driver_v1.atomic_write_json(output, report)
    print(json.dumps({
        "output": str(output), "passed": report["passed"],
        "same_basis_coefficient_cosine": report[
            "same_basis_coefficient_cosine"
        ],
        "content_sha256": report["content_sha256_before_self_field"],
    }, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
