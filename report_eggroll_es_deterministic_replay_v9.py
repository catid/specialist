#!/usr/bin/env python3
"""Compare the original v8 data43 run with its exact v9 replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import run_eggroll_es_anchor_line_search as driver_v1
import run_eggroll_es_anchor_stability_v8 as driver_v8
import run_eggroll_es_anchor_replay_v9 as replay_v9


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vector_sha256(values):
    return driver_v1.canonical_sha256([float(value) for value in values])


def cosine(left, right):
    if (
        not isinstance(left, list) or not isinstance(right, list)
        or len(left) != replay_v9.POPULATION_SIZE_V9
        or len(right) != replay_v9.POPULATION_SIZE_V9
    ):
        raise ValueError("v9 replay vectors must each contain 32 values")
    left = [float(value) for value in left]
    right = [float(value) for value in right]
    if not all(math.isfinite(value) for value in (*left, *right)):
        raise ValueError("v9 replay cosine input contains non-finite values")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("v9 replay cosine is undefined for a zero vector")
    return sum(a * b for a, b in zip(left, right)) / (
        left_norm * right_norm
    )


def standardize(values):
    values = [float(value) for value in values]
    if len(values) != replay_v9.POPULATION_SIZE_V9:
        raise ValueError("v9 raw score vector must contain 32 values")
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    if std == 0.0:
        raise ValueError("v9 raw score vector has zero variance")
    return [(value - mean) / std for value in values]


def _load_vectors(path, *, role):
    path = Path(path).resolve()
    replay_v9.driver_v8.offline_audit._assert_no_heldout(
        str(path), f"v9 {role} journal path",
    )
    try:
        journal = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read v9 {role} journal {path}") from error
    if role == "original":
        expected = replay_v9.ORIGINAL_V8_JOURNALS_V9[43]
        if path != expected["path"] or file_sha256(path) != expected[
            "file_sha256"
        ]:
            raise ValueError("v9 reporter requires the exact original data43 run")
        audit = driver_v8.validate_completed_journal_v8(journal)
        if (
            audit["data_bootstrap_seed"] != 43
            or audit["coefficient_sha256"] != expected["coefficient_sha256"]
            or audit["robust_plan_sha256"] != expected["robust_plan_sha256"]
            or audit["content_sha256"] != expected["content_sha256"]
        ):
            raise ValueError("v9 original data43 strict identity changed")
        data_seed = audit["data_bootstrap_seed"]
    elif role == "replay":
        audit = replay_v9.validate_completed_journal_v9(journal)
        data_seed = audit["data_seed"]
    else:
        raise ValueError("v9 replay role must be original or replay")
    coefficient_plan = journal.get("coefficient_plan", {})
    coefficients = coefficient_plan.get("coefficients")
    domain_scores = coefficient_plan.get("domain_scores_v5")
    anchor_scores = coefficient_plan.get("anchor_scores_v5")
    if any(
        not isinstance(values, list)
        or len(values) != replay_v9.POPULATION_SIZE_V9
        for values in (coefficients, domain_scores, anchor_scores)
    ):
        raise ValueError("v9 replay journal omitted coefficient/raw-score vectors")
    return {
        "role": role,
        "data_seed": data_seed,
        "journal": str(path),
        "journal_file_sha256": file_sha256(path),
        "content_sha256": audit["content_sha256"],
        "coefficient_sha256": audit["coefficient_sha256"],
        "robust_plan_sha256": audit["robust_plan_sha256"],
        "domain_scores_sha256": vector_sha256(domain_scores),
        "anchor_scores_sha256": vector_sha256(anchor_scores),
        "perturbation_basis_sha256": audit["perturbation_basis_sha256"],
        "coefficients": list(coefficients),
        "domain_scores": list(domain_scores),
        "anchor_scores": list(anchor_scores),
    }


def build_report(original_journal, replay_journal):
    evidence = replay_v9._v8_failed_evidence_v9(
        replay_v9.V8_FAILED_REPORT_PATH_V9
    )
    original = _load_vectors(original_journal, role="original")
    replay = _load_vectors(replay_journal, role="replay")
    if (
        original["data_seed"] != replay["data_seed"]
        or original["data_seed"] != replay_v9.DATA_SEED_V9
        or original["journal"] == replay["journal"]
        or {
            original["perturbation_basis_sha256"],
            replay["perturbation_basis_sha256"],
        } != {replay_v9.PERTURBATION_BASIS_SHA256_V9}
    ):
        raise ValueError("v9 report requires distinct exact-data43 same-basis runs")
    cosines = {
        "coefficient": cosine(
            original["coefficients"], replay["coefficients"],
        ),
        "standardized_domain_score": cosine(
            standardize(original["domain_scores"]),
            standardize(replay["domain_scores"]),
        ),
        "standardized_anchor_score": cosine(
            standardize(original["anchor_scores"]),
            standardize(replay["anchor_scores"]),
        ),
    }
    exact_identities = {
        "coefficient_sha256": (
            original["coefficient_sha256"] == replay["coefficient_sha256"]
        ),
        "domain_scores_sha256": (
            original["domain_scores_sha256"]
            == replay["domain_scores_sha256"]
        ),
        "anchor_scores_sha256": (
            original["anchor_scores_sha256"]
            == replay["anchor_scores_sha256"]
        ),
    }
    public_runs = []
    for run in (original, replay):
        public_runs.append({
            key: value for key, value in run.items()
            if key not in {"coefficients", "domain_scores", "anchor_scores"}
        })
    implementation = {
        "reporter_v9": file_sha256(Path(__file__).resolve()),
        "replay_driver_v9": file_sha256(Path(replay_v9.__file__).resolve()),
    }
    threshold = replay_v9.REPLAY_COSINE_THRESHOLD_V9
    report = {
        "schema": "eggroll-es-deterministic-replay-report-v9",
        "metric": "exact_data43_replay_identity_and_cosine",
        "metric_interpretation": (
            "same data/batch/reference/basis response reproducibility; "
            "standardized raw-score cosines remove irrelevant score offsets"
        ),
        "preregistered_cosine_threshold": threshold,
        "coverage": {
            "arm": "middle_late", "data_seed": 43,
            "perturbation_basis_seed": replay_v9.PERTURBATION_BASIS_SEED_V9,
            "perturbation_basis_sha256": (
                replay_v9.PERTURBATION_BASIS_SHA256_V9
            ),
            "population_size": 32, "target_alphas": [0.0],
            "roles": ["original", "replay"],
        },
        "selection_policy": (
            "coefficient_and_raw_train_anchor_identity_only_"
            "no_validation_ood_holdout_selection"
        ),
        "v8_failed_evidence_binding_sha256": evidence["binding_sha256"],
        "runs": public_runs,
        "cosines": cosines,
        "exact_identities": exact_identities,
        "all_exact_identities_match": all(exact_identities.values()),
        "all_cosines_pass": all(value >= threshold for value in cosines.values()),
        "passed": (
            all(exact_identities.values())
            and all(value >= threshold for value in cosines.values())
        ),
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
    parser.add_argument("--original-journal", required=True)
    parser.add_argument("--replay-journal", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output_json).resolve()
    replay_v9.driver_v8.offline_audit._assert_no_heldout(
        str(output), "v9 deterministic replay report output path",
    )
    if output.exists() or output.with_name(output.name + ".tmp").exists():
        raise ValueError("v9 deterministic replay report output already exists")
    report = build_report(args.original_journal, args.replay_journal)
    driver_v1.atomic_write_json(output, report)
    print(json.dumps({
        "output": str(output), "passed": report["passed"],
        "cosines": report["cosines"],
        "all_exact_identities_match": report[
            "all_exact_identities_match"
        ],
        "content_sha256": report["content_sha256_before_self_field"],
    }, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
