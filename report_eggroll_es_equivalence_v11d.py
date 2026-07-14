#!/usr/bin/env python3
"""Offline report for the V11d evidence-bound exact V11c retry."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import report_eggroll_es_equivalence_v11c as report_v11c
import run_eggroll_es_anchor_equivalence_v11d as retry_v11d


def build_report(journal_path, launch_attempt_path):
    journal_path = Path(journal_path).resolve()
    launch_attempt_path = Path(launch_attempt_path).resolve()
    offline = retry_v11d.driver_v11c.driver_v11b.driver_v11.driver_v8.offline_audit
    offline._assert_no_heldout(str(journal_path), "v11d equivalence journal")
    offline._assert_no_heldout(
        str(launch_attempt_path), "v11d launch-attempt evidence",
    )
    journal = json.loads(journal_path.read_text())
    launch_attempt = json.loads(launch_attempt_path.read_text())
    audit = retry_v11d.validate_completed_journal_v11d(journal)
    launch_audit = retry_v11d.validate_launch_attempt_v11d(launch_attempt)

    prior = retry_v11d._patch_v11c_retry_globals_v11d()
    try:
        report = report_v11c.build_report(journal_path)
    finally:
        retry_v11d._restore_v11c_retry_globals_v11d(prior)
    report = copy.deepcopy(report)
    report["schema"] = "eggroll-es-resident-sign-equivalence-report-v11d"
    report["metric"] = (
        "exact_v10_equivalence_with_durable_launch_evidence"
    )
    report["stage"] = "post_engine_pre_journal_exact_retry"
    report["v11d"] = {
        "experiment_name": retry_v11d.EXPERIMENT_NAME_V11D,
        "v11c_recipe_sha256": retry_v11d.EXPECTED_V11C_RECIPE_SHA256_V11D,
        "launch_attempt": str(launch_attempt_path),
        "launch_attempt_file_sha256": retry_v11d._file_sha256(
            launch_attempt_path
        ),
        "launch_attempt_content_sha256": launch_audit["content_sha256"],
        "v11c_failure_binding_sha256": launch_audit[
            "v11c_failure_binding_sha256"
        ],
        "journal_content_sha256": audit["content_sha256"],
    }
    report.pop("content_sha256_before_self_field", None)
    report["content_sha256_before_self_field"] = (
        retry_v11d.driver_v11c.driver_v1.canonical_sha256(report)
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", required=True)
    parser.add_argument("--launch-attempt", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output_json).resolve()
    offline = retry_v11d.driver_v11c.driver_v11b.driver_v11.driver_v8.offline_audit
    offline._assert_no_heldout(str(output), "v11d report output")
    if output.exists() or output.with_name(output.name + ".tmp").exists():
        raise ValueError("v11d report output already exists")
    report = build_report(args.journal, args.launch_attempt)
    retry_v11d.driver_v11c.driver_v1.atomic_write_json(output, report)
    print(json.dumps({
        "output": str(output),
        "passed": report["passed"],
        "content_sha256": report["content_sha256_before_self_field"],
        "launch_attempt_content_sha256": report["v11d"][
            "launch_attempt_content_sha256"
        ],
    }, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
