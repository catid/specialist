#!/usr/bin/env python3
"""Offline exact-equivalence report for the V11e forwarding retry."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import report_eggroll_es_equivalence_v11c as report_v11c
import run_eggroll_es_anchor_equivalence_v11e as retry_v11e


def build_report(journal_path, launch_attempt_path):
    journal_path = Path(journal_path).resolve()
    launch_attempt_path = Path(launch_attempt_path).resolve()
    runtime = retry_v11e._runtime_cli_v11e(
        list(retry_v11e.FROZEN_REAL_ARGV_V11E)
    )
    expected_attempt = retry_v11e._attempt_path_v11e(runtime).resolve()
    if launch_attempt_path != expected_attempt:
        raise RuntimeError("v11e reporter launch-attempt path changed")
    offline = retry_v11e._offline_audit_v11e()
    offline._assert_no_heldout(str(journal_path), "v11e equivalence journal")
    offline._assert_no_heldout(
        str(launch_attempt_path), "v11e launch-attempt evidence",
    )
    journal = json.loads(journal_path.read_text())
    attempt = json.loads(launch_attempt_path.read_text())
    # The inherited journal's exact schema permits only its explicit false
    # heldout sentinels. V11e-owned launch evidence has no legacy exception.
    offline._assert_no_heldout(attempt, "v11e loaded launch-attempt object")
    journal_audit = retry_v11e.validate_completed_journal_v11e(journal)
    attempt_audit = retry_v11e.validate_launch_attempt_v11e(
        attempt, journal_path,
    )

    prior = retry_v11e._patch_v11c_globals_v11e()
    try:
        report = report_v11c.build_report(journal_path)
    finally:
        retry_v11e._restore_v11c_globals_v11e(prior)
    report = copy.deepcopy(report)
    report["schema"] = "eggroll-es-resident-sign-equivalence-report-v11e"
    report["metric"] = (
        "exact_v10_equivalence_with_effective_anchor_cli_forwarding"
    )
    report["stage"] = "effective_downstream_cli_exact_retry"
    report["v11e"] = {
        "experiment_name": retry_v11e.EXPERIMENT_NAME_V11E,
        "v11c_recipe_sha256": retry_v11e.EXPECTED_V11C_RECIPE_SHA256_V11E,
        "launch_attempt": str(launch_attempt_path),
        "launch_attempt_file_sha256": retry_v11e._file_sha256(
            launch_attempt_path
        ),
        "launch_attempt_content_sha256": attempt_audit["content_sha256"],
        "v11d_failure_binding_sha256": attempt_audit[
            "v11d_failure_binding_sha256"
        ],
        "effective_downstream_cli_sha256": attempt_audit[
            "effective_downstream_cli_sha256"
        ],
        "v11c_implementation_bundle_sha256": attempt_audit[
            "v11c_implementation_bundle_sha256"
        ],
        "frozen_recipe_or_data_changed_from_v11d": False,
        "effective_cli_forwarding_corrected": True,
        "effective_cli_correction": copy.deepcopy(
            retry_v11e.EFFECTIVE_CLI_CORRECTION_V11E
        ),
        "journal_content_sha256": journal_audit["content_sha256"],
    }
    report.pop("content_sha256_before_self_field", None)
    report["content_sha256_before_self_field"] = retry_v11e._canonical(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", required=True)
    parser.add_argument("--launch-attempt", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output_json).resolve()
    retry_v11e._offline_audit_v11e()._assert_no_heldout(
        str(output), "v11e report output",
    )
    if output.exists() or output.with_name(output.name + ".tmp").exists():
        raise ValueError("v11e report output already exists")
    report = build_report(args.journal, args.launch_attempt)
    retry_v11e.driver_v11c.driver_v1.atomic_write_json(output, report)
    print(json.dumps({
        "output": str(output),
        "passed": report["passed"],
        "content_sha256": report["content_sha256_before_self_field"],
        "effective_downstream_cli_sha256": report["v11e"][
            "effective_downstream_cli_sha256"
        ],
    }, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
