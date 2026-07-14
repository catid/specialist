#!/usr/bin/env python3
"""Report crossed antithetic response stability for v10."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

import report_eggroll_es_deterministic_replay_v9 as vector_tools
import run_eggroll_es_anchor_line_search as driver_v1
import run_eggroll_es_anchor_variance_v10 as variance_v10


def build_report(journal_path):
    path = Path(journal_path).resolve()
    variance_v10.driver_v8.offline_audit._assert_no_heldout(
        str(path), "v10 variance journal path",
    )
    journal = json.loads(path.read_text())
    audit = variance_v10.validate_completed_journal_v10(journal)
    cross = journal["coefficient_plan"]["antithetic_cross_v10"]
    cells = cross["cells"]
    names = sorted(cells)
    pairwise = {}
    for left, right in itertools.combinations(names, 2):
        pairwise[f"{left}__{right}"] = vector_tools.cosine(
            cells[left]["coefficients"], cells[right]["coefficients"],
        )
    domain_cosine = vector_tools.cosine(
        cross["central_domain_scores"]["D43"],
        cross["central_domain_scores"]["D44"],
    )
    anchor_cosine = vector_tools.cosine(
        cross["central_anchor_scores"]["A43"],
        cross["central_anchor_scores"]["A44"],
    )
    minimum = min(pairwise.values())
    median = statistics.median(pairwise.values())
    implementation = {
        "reporter_v10": vector_tools.file_sha256(Path(__file__).resolve()),
        "variance_driver_v10": vector_tools.file_sha256(
            Path(variance_v10.__file__).resolve()
        ),
    }
    report = {
        "schema": "eggroll-es-antithetic-crossed-report-v10",
        "metric": "four_cell_antithetic_coefficient_stability",
        "thresholds": {
            "minimum_pairwise_coefficient_cosine": 0.5,
            "median_pairwise_coefficient_cosine": 0.7,
        },
        "coverage": {
            "arm": "middle_late", "data_seed": 43,
            "domain_manifests": variance_v10.anchor_v10.DOMAIN_MANIFESTS_V10,
            "anchor_generation_seeds": [43, 44],
            "base_direction_count": 32,
            "unique_signed_direction_count": 64,
            "actual_perturb_restore_cycle_count": 128,
            "domain_signed_score_count": 128,
            "anchor_signed_response_count": 128,
            "target_alphas": [0.0],
            "perturbation_basis_sha256": (
                variance_v10.PERTURBATION_BASIS_SHA256_V10
            ),
        },
        "selection_policy": (
            "crossed_train_anchor_response_only_"
            "no_validation_ood_holdout_selection"
        ),
        "journal": str(path),
        "journal_file_sha256": vector_tools.file_sha256(path),
        "journal_content_sha256": audit["content_sha256"],
        "cross_artifact_content_sha256": audit["cross"]["content_sha256"],
        "cell_coefficient_sha256": audit["cross"][
            "cell_coefficient_sha256"
        ],
        "raw_central_domain_cosine": domain_cosine,
        "raw_central_anchor_cosine": anchor_cosine,
        "pairwise_cell_coefficient_cosines": pairwise,
        "minimum_pairwise_coefficient_cosine": minimum,
        "median_pairwise_coefficient_cosine": median,
        "passed": minimum >= 0.5 and median >= 0.7,
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
    parser.add_argument("--journal", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output_json).resolve()
    variance_v10.driver_v8.offline_audit._assert_no_heldout(
        str(output), "v10 variance report output path",
    )
    if output.exists() or output.with_name(output.name + ".tmp").exists():
        raise ValueError("v10 variance report output already exists")
    report = build_report(args.journal)
    driver_v1.atomic_write_json(output, report)
    print(json.dumps({
        "output": str(output), "passed": report["passed"],
        "minimum_pairwise_coefficient_cosine": report[
            "minimum_pairwise_coefficient_cosine"
        ],
        "median_pairwise_coefficient_cosine": report[
            "median_pairwise_coefficient_cosine"
        ],
        "content_sha256": report["content_sha256_before_self_field"],
    }, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
