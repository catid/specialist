#!/usr/bin/env python3
"""Mint compact, train-only aggregate evidence from completed V13b."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
from pathlib import Path

import run_eggroll_es_train_panels_v13 as driver_v13
import run_eggroll_es_train_panels_v13b as driver_v13b
import train_eggroll_es_specialist_anchor_v13 as anchor_v13


ROOT = Path(__file__).resolve().parent
ATTEMPT_PATH_V14A = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    ".snapshot794_layer_v13b_document_balanced_five_panel_alpha_zero_"
    "runtime_forwarded_resident_sign_basis20260714.launch_attempt.json"
).resolve()
REPORT_PATH_V14A = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "snapshot794_layer_v13b_document_balanced_five_panel_alpha_zero_"
    "runtime_forwarded_resident_sign_basis20260714/"
    "train_panel_diagnostic_v13.json"
).resolve()
OUTPUT_PATH_V14A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V13B_TRAIN_PANEL_AGGREGATE_EVIDENCE_V14A.json"
).resolve()
ATTEMPT_FILE_SHA256_V14A = (
    "00513c2e32d839c895a2cac5d4d00717be88b89bccfc9f841265c8cff9be6b6c"
)
ATTEMPT_CONTENT_SHA256_V14A = (
    "188c972e97d531d9ace9e3ae3ca17dee943e449795c6731552b8039cf04285c4"
)
REPORT_FILE_SHA256_V14A = (
    "d53832ab9d021aa4692cef038058014ca84501a772ee935195bf8dfeba85e753"
)
REPORT_CONTENT_SHA256_V14A = (
    "dfa8c73fae35d0b915dcb1f7c5ef2bca91415a551a178cc90ab534a0646939da"
)
SOURCE_COMMIT_V14A = "9b05db2c4afc544ac3a20f32bd5ab3cad6666d85"
IMPLEMENTATION_BUNDLE_SHA256_V14A = (
    "21bdbab4f22c3cb75f273266dc7caba1ffd3590148dfb204745eaf9f5a45a35b"
)


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    return driver_v13._canonical(value)


def _sign(value):
    return 1 if value > 0.0 else (-1 if value < 0.0 else 0)


def _cosine(left, right):
    numerator = math.fsum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise RuntimeError("v14a aggregate evidence encountered a zero vector")
    return numerator / (left_norm * right_norm)


def _sign_agreement(left, right):
    pairs = [(_sign(a), _sign(b)) for a, b in zip(left, right)]
    if any(a == 0 or b == 0 for a, b in pairs):
        raise RuntimeError("v14a aggregate evidence encountered a zero sign")
    return sum(a == b for a, b in pairs) / len(pairs)


def _summary(values):
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("v14a stability summary is incomplete or non-finite")
    return {
        "count": len(values),
        "median": float(statistics.median(values)),
        "worst": min(values),
    }


def validate_v13b_aggregate_v14a(
    attempt_path=ATTEMPT_PATH_V14A, report_path=REPORT_PATH_V14A,
):
    attempt_path = Path(attempt_path).resolve()
    report_path = Path(report_path).resolve()
    if attempt_path != ATTEMPT_PATH_V14A or report_path != REPORT_PATH_V14A:
        raise RuntimeError("v14a requires canonical V13b aggregate paths")
    if (
        _file_sha256(attempt_path) != ATTEMPT_FILE_SHA256_V14A
        or _file_sha256(report_path) != REPORT_FILE_SHA256_V14A
    ):
        raise RuntimeError("v14a V13b aggregate file identity changed")
    attempt = json.loads(attempt_path.read_text())
    report = json.loads(report_path.read_text())
    if (
        attempt.get("schema") != "eggroll-es-durable-launch-attempt-v13"
        or attempt.get("status") != "complete"
        or attempt.get("phase") != "after_trainer_cleanup_and_report"
        or attempt.get("experiment_name") != driver_v13b.EXPERIMENT_NAME_V13B
        or attempt.get("target_alpha_zero_only") is not True
        or attempt.get("model_update_applied") is not False
        or attempt.get("sealed_or_nontrain_surface_opened") is not False
        or attempt.get("report_exists_after_attempt") is not True
        or attempt.get("content_sha256_before_self_field")
        != ATTEMPT_CONTENT_SHA256_V14A
        or attempt.get("content_sha256_before_self_field")
        != _canonical({
            key: value for key, value in attempt.items()
            if key != "content_sha256_before_self_field"
        })
        or report.get("schema")
        != "eggroll-es-five-panel-alpha-zero-report-v13"
        or report.get("model_update_applied") is not False
        or report.get("sealed_or_nontrain_surface_opened") is not False
        or report.get("decision")
        != "diagnostic_only_no_promotion_interpretation"
        or report.get("content_sha256_before_self_field")
        != REPORT_CONTENT_SHA256_V14A
        or report.get("content_sha256_before_self_field")
        != _canonical({
            key: value for key, value in report.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v14a V13b aggregate completion semantics changed")
    binding = attempt.get("report_binding", {})
    if (
        Path(binding.get("path", "")).resolve() != report_path
        or binding.get("file_sha256") != REPORT_FILE_SHA256_V14A
        or binding.get("content_sha256") != REPORT_CONTENT_SHA256_V14A
    ):
        raise RuntimeError("v14a V13b report binding changed")
    anchor_v13.validate_diagnostic_v13(report.get("diagnostic"))
    source = attempt.get("source_provenance", {})
    if (
        source.get("git_head") != SOURCE_COMMIT_V14A
        or source.get("implementation_bundle_sha256")
        != IMPLEMENTATION_BUNDLE_SHA256_V14A
        or report.get("implementation", {}).get("bundle_sha256")
        != IMPLEMENTATION_BUNDLE_SHA256_V14A
        or report.get("recipe", {}).get("implementation_bundle_sha256")
        != IMPLEMENTATION_BUNDLE_SHA256_V14A
    ):
        raise RuntimeError("v14a V13b committed implementation changed")
    for item in source.get("files", {}).values():
        relative = item["relative_path"]
        raw = subprocess.check_output(
            ["git", "show", f"{SOURCE_COMMIT_V14A}:{relative}"], cwd=ROOT,
        )
        if (
            hashlib.sha256(raw).hexdigest() != item["file_sha256"]
            or _file_sha256(ROOT / relative) != item["file_sha256"]
        ):
            raise RuntimeError("v14a V13b source provenance changed")
    return attempt, report


def build_evidence_v14a():
    attempt, report = validate_v13b_aggregate_v14a()
    diagnostic = report["diagnostic"]
    analysis = diagnostic["analysis"]
    pairwise = list(analysis["optimization_pairwise"].values())
    aggregate = analysis["robust_optimization_aggregate"]["coefficients"]
    optimization = [
        analysis["panel_analysis"][name]["coefficients"]
        for name in anchor_v13.OPTIMIZATION_PANELS_V13
    ]
    screens = list(analysis["train_screen_transfer"].values())
    evidence = {
        "schema": "eggroll-es-v13b-train-panel-aggregate-evidence-v14a",
        "passed": True,
        "selection_surface": "frozen_train_panels_only",
        "contains_response_vectors_or_row_content": False,
        "contains_validation_ood_or_heldout_content": False,
        "v13b_attempt": {
            "path": str(ATTEMPT_PATH_V14A),
            "file_sha256": ATTEMPT_FILE_SHA256_V14A,
            "content_sha256": ATTEMPT_CONTENT_SHA256_V14A,
            "source_commit": SOURCE_COMMIT_V14A,
        },
        "v13b_report": {
            "path": str(REPORT_PATH_V14A),
            "file_sha256": REPORT_FILE_SHA256_V14A,
            "content_sha256": REPORT_CONTENT_SHA256_V14A,
            "implementation_bundle_sha256": (
                IMPLEMENTATION_BUNDLE_SHA256_V14A
            ),
            "recipe_content_sha256": report["recipe"][
                "content_sha256_before_self_field"
            ],
        },
        "runtime_integrity": {
            "alpha": diagnostic["alpha"],
            "model_update_applied": diagnostic["model_update_applied"],
            "identity_audit_passed": diagnostic["identity_audit"]["passed"],
            "population_boundary_audit_sha256": diagnostic[
                "population_boundary_audit_v4"
            ]["audit_sha256"],
            "panel_bundle_content_sha256": diagnostic[
                "panel_bundle_content_sha256"
            ],
            "perturbation_basis_sha256": diagnostic[
                "perturbation_basis"
            ]["basis_sha256"],
        },
        "stability": {
            "optimization_pairwise_cosine": _summary([
                item["cosine"] for item in pairwise
            ]),
            "optimization_pairwise_sign_agreement": _summary([
                item["sign_agreement"]["all_coordinate_fraction"]
                for item in pairwise
            ]),
            "aggregate_to_optimization_cosine": _summary([
                _cosine(aggregate, item) for item in optimization
            ]),
            "aggregate_to_optimization_sign_agreement": _summary([
                _sign_agreement(aggregate, item) for item in optimization
            ]),
            "train_screen_cosine": _summary([
                item["cosine_to_frozen_optimization_aggregate"]
                for item in screens
            ]),
            "train_screen_sign_agreement": _summary([
                item["sign_agreement_to_frozen_optimization_aggregate"][
                    "all_coordinate_fraction"
                ]
                for item in screens
            ]),
            "robust_aggregate": {
                "coefficient_sha256": analysis[
                    "robust_optimization_aggregate"
                ]["coefficient_sha256"],
                "l2_norm": analysis["robust_optimization_aggregate"][
                    "l2_norm"
                ],
                "nonzero_coordinate_count": analysis[
                    "robust_optimization_aggregate"
                ]["nonzero_coordinate_count"],
            },
            "all_panel_spreads_nonzero": all(
                not item["standardization"]["zero_spread"]
                for item in analysis["panel_analysis"].values()
            ),
        },
    }
    evidence["content_sha256_before_self_field"] = _canonical(evidence)
    return evidence


def _exclusive_write(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("v14a compact evidence already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default=str(OUTPUT_PATH_V14A))
    args = parser.parse_args(argv)
    output = Path(args.output_json).resolve()
    if output != OUTPUT_PATH_V14A:
        raise ValueError("v14a compact evidence requires its canonical path")
    evidence = build_evidence_v14a()
    _exclusive_write(output, evidence)
    print(json.dumps({
        "output": str(output),
        "file_sha256": _file_sha256(output),
        "content_sha256": evidence["content_sha256_before_self_field"],
    }, sort_keys=True))
    return evidence


if __name__ == "__main__":
    main()
