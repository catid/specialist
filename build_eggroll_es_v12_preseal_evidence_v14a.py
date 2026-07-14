#!/usr/bin/env python3
"""Mint compact aggregate evidence for V12's closed preseal candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import run_eggroll_es_anchor_line_search as driver_v1
import run_eggroll_es_consensus_candidate_v12 as driver_v12
import train_eggroll_es_specialist_anchor_v12 as anchor_v12


ROOT = Path(__file__).resolve().parent
REPORT_PATH_V14A = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "snapshot794_layer_v12_middle_late_consensus_preseal_c45c46/"
    "preseal_screen.json"
).resolve()
CANDIDATE_SEAL_PATH_V14A = REPORT_PATH_V14A.with_name("candidate_seal.json")
OUTPUT_PATH_V14A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V12_PRESEAL_NEGATIVE_AGGREGATE_EVIDENCE_V14A.json"
).resolve()
REPORT_FILE_SHA256_V14A = (
    "1cc6c07a251fcbee61150a3a688940bf1a765934db2cdb1822f37b5339477c4e"
)
REPORT_CONTENT_SHA256_V14A = (
    "001eac316bb74eb6b2949acade7167d9a6efedb1035f2767c5aedf254ecad831"
)
IMPLEMENTATION_BUNDLE_SHA256_V14A = (
    "65fd391ea76e94d3945906b8e94a8dfc5027d2dced7736b7def0ea78af0d69e1"
)
V11_EVIDENCE_CONTENT_SHA256_V14A = (
    "b6212a4bdafaf234f8445b11c18ef96e15526d450f89d366542c63fae2d15e8f"
)
CONSENSUS_COEFFICIENT_SHA256_V14A = (
    "1a85502564020048f634b0a8ced3952343f135f67ffd1faad1aaa697aebea8a8"
)


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_v12_preseal_v14a(path=REPORT_PATH_V14A):
    path = Path(path).resolve()
    if path != REPORT_PATH_V14A:
        raise RuntimeError("v14a requires the canonical V12 preseal report")
    if _file_sha256(path) != REPORT_FILE_SHA256_V14A:
        raise RuntimeError("v14a V12 preseal report file identity changed")
    report = json.loads(path.read_text())
    if (
        report.get("schema") != "eggroll-es-consensus-preseal-screen-v12"
        or report.get("stage") != "preseal"
        or report.get("status") != "no_eligible_candidate"
        or report.get("selected_alpha") is not None
        or report.get("fallback_selected") is not False
        or report.get("benchmark_content_opened_before_candidate_seal") is not False
        or report.get("heldout_opened") is not False
        or report.get("content_sha256_before_self_field")
        != REPORT_CONTENT_SHA256_V14A
        or report.get("content_sha256_before_self_field")
        != driver_v1.canonical_sha256({
            key: value for key, value in report.items()
            if key != "content_sha256_before_self_field"
        })
        or report.get("implementation", {}).get("bundle_sha256")
        != IMPLEMENTATION_BUNDLE_SHA256_V14A
        or report.get("implementation") != driver_v12.implementation_identity_v12()
        or report.get("v11_evidence_content_sha256")
        != V11_EVIDENCE_CONTENT_SHA256_V14A
        or report.get("consensus", {}).get("coefficient_sha256")
        != CONSENSUS_COEFFICIENT_SHA256_V14A
        or report.get("runtime_integrity", {}).get("identity_audit_passed")
        is not True
        or report.get("runtime_integrity", {}).get(
            "population_exact_restore_passed"
        ) is not True
        or CANDIDATE_SEAL_PATH_V14A.exists()
    ):
        raise RuntimeError("v14a V12 closed-preseal semantics changed")
    anchor_v12.validate_consensus_v12(report.get("consensus"))
    states = report.get("states")
    if (
        not isinstance(states, list)
        or [state.get("alpha") for state in states] != anchor_v12.ALPHA_GRID_V12
        or any(state.get("gate", {}).get("eligible") is not False for state in states)
        or states[0].get("application") is not None
        or any(state.get("application") is None for state in states[1:])
    ):
        raise RuntimeError("v14a V12 preseal state coverage changed")
    for state in states:
        gate = state["gate"]
        if (
            set(gate.get("screens", {})) != {"C45", "C46"}
            or set(gate.get("anchors", {})) != {"A43", "A44"}
            or gate.get("content_sha256_before_self_field")
            != driver_v1.canonical_sha256({
                key: value for key, value in gate.items()
                if key != "content_sha256_before_self_field"
            })
        ):
            raise RuntimeError("v14a V12 aggregate gate shape changed")
    return report


def build_evidence_v14a():
    report = validate_v12_preseal_v14a()
    states = []
    for state in report["states"]:
        gate = state["gate"]
        application = state["application"]
        states.append({
            "alpha": state["alpha"],
            "eligible": gate["eligible"],
            "screen_lcb": {
                name: gate["screens"][name]["lower_confidence_bound"]
                for name in ("C45", "C46")
            },
            "anchor_lcb": {
                name: gate["anchors"][name]["lower_confidence_bound"]
                for name in ("A43", "A44")
            },
            "gate_content_sha256": gate["content_sha256_before_self_field"],
            "application": (
                None if application is None else {
                    "target_alpha": application["target_alpha"],
                    "coefficient_sha256": application["coefficient_sha256"],
                    "manifest_sha256": application["manifest_sha256"],
                }
            ),
        })
    positive = states[1:]
    evidence = {
        "schema": "eggroll-es-v12-preseal-negative-aggregate-evidence-v14a",
        "passed": True,
        "decision": "close_v12_candidate_without_confirmation_or_release",
        "selection_surface": "train_screens_and_prose_anchors_only",
        "contains_response_documents_or_row_content": False,
        "contains_validation_ood_or_heldout_content": False,
        "report": {
            "path": str(REPORT_PATH_V14A),
            "file_sha256": REPORT_FILE_SHA256_V14A,
            "content_sha256": REPORT_CONTENT_SHA256_V14A,
            "implementation_bundle_sha256": IMPLEMENTATION_BUNDLE_SHA256_V14A,
            "v11_evidence_content_sha256": V11_EVIDENCE_CONTENT_SHA256_V14A,
        },
        "consensus_coefficient_sha256": CONSENSUS_COEFFICIENT_SHA256_V14A,
        "runtime_integrity": {
            "identity_audit_passed": True,
            "population_exact_restore_passed": True,
            "population_boundary_audit_sha256": report["runtime_integrity"][
                "population_boundary_audit_sha256"
            ],
        },
        "states": states,
        "aggregate_failure": {
            "positive_alpha_count": len(positive),
            "eligible_positive_alpha_count": sum(
                state["eligible"] is True for state in positive
            ),
            "all_positive_screen_lcbs_negative": all(
                value < 0.0 for state in positive
                for value in state["screen_lcb"].values()
            ),
            "all_positive_anchor_lcbs_negative": all(
                value < 0.0 for state in positive
                for value in state["anchor_lcb"].values()
            ),
            "candidate_seal_written": False,
            "heldout_opened": False,
            "benchmark_content_opened_before_candidate_seal": False,
        },
        "next_step_constraint": (
            "do_not_reuse_the_v12_consensus_candidate_or_alpha_grid; "
            "continue_only_with_a_fresh_train_only_direction_estimator"
        ),
    }
    evidence["content_sha256_before_self_field"] = driver_v1.canonical_sha256(
        evidence
    )
    return evidence


def _exclusive_write(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("v14a V12 negative evidence already exists") from error
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
        raise ValueError("v14a V12 negative evidence requires its canonical path")
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
