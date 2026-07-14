#!/usr/bin/env python3
"""Report exact V11 resident-sign equivalence to the completed V10 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import report_eggroll_es_deterministic_replay_v9 as vector_tools
import run_eggroll_es_anchor_equivalence_v11 as equivalence_v11
import run_eggroll_es_anchor_line_search as driver_v1


def build_report(journal_path):
    path = Path(journal_path).resolve()
    equivalence_v11.driver_v8.offline_audit._assert_no_heldout(
        str(path), "v11 equivalence journal path",
    )
    journal = json.loads(path.read_text())
    audit = equivalence_v11.validate_completed_journal_v11(journal)
    cross = journal["coefficient_plan"]["resident_sign_cross_v11"]
    report = {
        "schema": "eggroll-es-resident-sign-equivalence-report-v11",
        "metric": "exact_raw_response_and_projected_plan_equivalence_to_v10",
        "passed": audit["equivalence"]["all_exact"] is True,
        "equivalence": audit["equivalence"],
        "coverage": {
            "arm": "middle_late", "data_seed": 43,
            "base_direction_count": 32,
            "unique_signed_direction_count": 64,
            "actual_perturb_restore_cycle_count": 64,
            "all_engine_sign_residency_count": 16,
            "domain_signed_score_count": 128,
            "anchor_signed_response_count": 128,
            "resident_generation_order": ["D43", "A43", "A44", "D44"],
            "target_alphas": [0.0],
        },
        "selection_policy": "exact_v10_equivalence_no_benchmark_selection",
        "v10_report": str(equivalence_v11.V10_REPORT_PATH_V11),
        "v10_report_file_sha256": (
            equivalence_v11.V10_REPORT_FILE_SHA256_V11
        ),
        "v10_report_content_sha256": (
            equivalence_v11.V10_REPORT_CONTENT_SHA256_V11
        ),
        "v10_journal_file_sha256": (
            equivalence_v11.V10_JOURNAL_FILE_SHA256_V11
        ),
        "v10_journal_content_sha256": (
            equivalence_v11.V10_JOURNAL_CONTENT_SHA256_V11
        ),
        "journal": str(path),
        "journal_file_sha256": vector_tools.file_sha256(path),
        "journal_content_sha256": audit["content_sha256"],
        "resident_artifact_content_sha256": cross[
            "content_sha256_before_self_field"
        ],
        "cell_coefficient_sha256": audit["resident"][
            "cell_coefficient_sha256"
        ],
    }
    report["content_sha256_before_self_field"] = driver_v1.canonical_sha256(
        report
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output_json).resolve()
    equivalence_v11.driver_v8.offline_audit._assert_no_heldout(
        str(output), "v11 equivalence report output path",
    )
    if output.exists() or output.with_name(output.name + ".tmp").exists():
        raise ValueError("v11 equivalence report output already exists")
    report = build_report(args.journal)
    driver_v1.atomic_write_json(output, report)
    print(json.dumps({
        "output": str(output), "passed": report["passed"],
        "content_sha256": report["content_sha256_before_self_field"],
        "equivalence_binding_sha256": report["equivalence"]["binding_sha256"],
    }, sort_keys=True))
    return report


if __name__ == "__main__":
    main()

