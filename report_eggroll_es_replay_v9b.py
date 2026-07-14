#!/usr/bin/env python3
"""Compare original v8 data44 with the exact v9b control replay."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import report_eggroll_es_deterministic_replay_v9 as reporter_v9
import run_eggroll_es_anchor_line_search as driver_v1
import run_eggroll_es_anchor_replay_v9b as replay_v9b
import run_eggroll_es_anchor_stability_v8 as driver_v8


def _load(path, role):
    path = Path(path).resolve()
    replay_v9b.driver_v8.offline_audit._assert_no_heldout(
        str(path), f"v9b {role} journal path",
    )
    journal = json.loads(path.read_text())
    if role == "original":
        expected = replay_v9b.replay_v9.ORIGINAL_V8_JOURNALS_V9[44]
        if (
            path != expected["path"]
            or reporter_v9.file_sha256(path) != expected["file_sha256"]
        ):
            raise ValueError("v9b requires the exact original v8 data44 journal")
        audit = driver_v8.validate_completed_journal_v8(journal)
        if (
            audit["data_bootstrap_seed"] != 44
            or audit["content_sha256"] != expected["content_sha256"]
            or audit["coefficient_sha256"] != expected["coefficient_sha256"]
            or audit["robust_plan_sha256"] != expected["robust_plan_sha256"]
        ):
            raise ValueError("v9b original data44 identity changed")
    elif role == "replay":
        audit = replay_v9b.validate_completed_journal_v9b(journal)
    else:
        raise ValueError("v9b role must be original or replay")
    plan = journal.get("coefficient_plan", {})
    vectors = {
        "coefficients": plan.get("coefficients"),
        "domain_scores": plan.get("domain_scores_v5"),
        "anchor_scores": plan.get("anchor_scores_v5"),
    }
    if any(not isinstance(value, list) or len(value) != 32
           for value in vectors.values()):
        raise ValueError("v9b journal omitted a 32-value response vector")
    return {
        "role": role, "data_seed": 44, "journal": str(path),
        "journal_file_sha256": reporter_v9.file_sha256(path),
        "content_sha256": audit["content_sha256"],
        "coefficient_sha256": audit["coefficient_sha256"],
        "robust_plan_sha256": audit["robust_plan_sha256"],
        "domain_scores_sha256": reporter_v9.vector_sha256(
            vectors["domain_scores"]
        ),
        "anchor_scores_sha256": reporter_v9.vector_sha256(
            vectors["anchor_scores"]
        ),
        "perturbation_basis_sha256": audit["perturbation_basis_sha256"],
        **vectors,
    }


def build_report(original_journal, replay_journal):
    original = _load(original_journal, "original")
    replay = _load(replay_journal, "replay")
    if original["journal"] == replay["journal"]:
        raise ValueError("v9b requires distinct original and replay journals")
    cosines = {
        "coefficient": reporter_v9.cosine(
            original["coefficients"], replay["coefficients"],
        ),
        "standardized_domain_score": reporter_v9.cosine(
            reporter_v9.standardize(original["domain_scores"]),
            reporter_v9.standardize(replay["domain_scores"]),
        ),
        "standardized_anchor_score": reporter_v9.cosine(
            reporter_v9.standardize(original["anchor_scores"]),
            reporter_v9.standardize(replay["anchor_scores"]),
        ),
    }
    exact = {
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
    public = []
    for run in (original, replay):
        public.append({
            key: value for key, value in run.items()
            if key not in {"coefficients", "domain_scores", "anchor_scores"}
        })
    implementation = {
        "reporter_v9b": reporter_v9.file_sha256(Path(__file__).resolve()),
        "replay_driver_v9b": reporter_v9.file_sha256(
            Path(replay_v9b.__file__).resolve()
        ),
    }
    passed = all(exact.values()) and all(
        value >= replay_v9b.REPLAY_COSINE_THRESHOLD_V9B
        for value in cosines.values()
    )
    report = {
        "schema": "eggroll-es-v8-data44-replay-report-v9b",
        "preregistered_cosine_threshold": 0.99,
        "coverage": {
            "arm": "middle_late", "data_seed": 44,
            "population_size": 32, "target_alphas": [0.0],
            "perturbation_basis_sha256": (
                driver_v8.PERTURBATION_BASIS_SHA256_V8
            ),
        },
        "selection_policy": (
            "coefficient_and_raw_train_anchor_identity_only_"
            "no_validation_ood_holdout_selection"
        ),
        "runs": public, "cosines": cosines, "exact_identities": exact,
        "all_exact_identities_match": all(exact.values()),
        "passed": passed, "implementation": implementation,
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
    replay_v9b.driver_v8.offline_audit._assert_no_heldout(
        str(output), "v9b replay report output path",
    )
    if output.exists() or output.with_name(output.name + ".tmp").exists():
        raise ValueError("v9b report output already exists")
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
