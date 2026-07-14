#!/usr/bin/env python3
"""Report V11c complete-facade exact equivalence to completed V10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import report_eggroll_es_deterministic_replay_v9 as vector_tools
import run_eggroll_es_anchor_equivalence_v11c as retry_v11c
import run_eggroll_es_anchor_line_search as driver_v1


def build_report(journal_path):
    path = Path(journal_path).resolve()
    retry_v11c.driver_v11b.driver_v11.driver_v8.offline_audit._assert_no_heldout(
        str(path), "v11c equivalence journal",
    )
    journal = json.loads(path.read_text())
    audit = retry_v11c.validate_completed_journal_v11c(journal)
    report = {
        "schema": "eggroll-es-resident-sign-equivalence-report-v11c",
        "metric": "exact_v10_equivalence_with_complete_anchor_runtime_api",
        "passed": audit["equivalence"]["all_exact"] is True,
        "equivalence": audit["equivalence"],
        "v11b_failure_binding_sha256": audit[
            "v11b_failure_binding_sha256"
        ],
        "anchor_runtime_api_preflight_sha256": audit[
            "anchor_runtime_api_preflight_sha256"
        ],
        "anchor_runtime_api": list(retry_v11c.ANCHOR_RUNTIME_API_V11C),
        "coverage": {
            "base_direction_count": 32,
            "unique_signed_direction_count": 64,
            "actual_perturb_restore_cycle_count": 64,
            "all_engine_sign_residency_count": 16,
            "domain_signed_score_count": 128,
            "anchor_signed_response_count": 128,
            "target_alphas": [0.0],
        },
        "selection_policy": "exact_v10_equivalence_no_benchmark_selection",
        "journal": str(path),
        "journal_file_sha256": vector_tools.file_sha256(path),
        "journal_content_sha256": audit["content_sha256"],
        "resident_artifact_content_sha256": audit["resident"][
            "content_sha256"
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
    retry_v11c.driver_v11b.driver_v11.driver_v8.offline_audit._assert_no_heldout(
        str(output), "v11c report output",
    )
    if output.exists() or output.with_name(output.name + ".tmp").exists():
        raise ValueError("v11c report output already exists")
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
