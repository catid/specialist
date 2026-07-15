#!/usr/bin/env python3
"""Bind V25A R1's valid, promising, but statistically unconfirmed result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS = (ROOT / "experiments/eggroll_es_hpo/runs").resolve()
ATTEMPT_PATH = RUNS / (
    ".s6_v25a_paired_production_v364_train_only_runtime_basis20260907_"
    "retry_r1.launch_attempt.json"
)
REPORT_PATH = RUNS / (
    "s6_v25a_paired_production_v364_train_only_runtime_basis20260907_"
    "retry_r1/paired_production_v364_compatibility_v25a_retry_r1.json"
)
OUTPUT_PATH = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V25A_PAIRED_PROMISING_UNCONFIRMED_EVIDENCE_R1.json"
)

ATTEMPT_FILE_SHA256 = "b4fc43b1542f017c571044f6c14c78ad9f904f78a444a5ab7b65721427a629b1"
ATTEMPT_CONTENT_SHA256 = "47cc122e823f385f8fc3f4bb01f6f96960bf77e5e0320e42a2a8235fe96dc343"
REPORT_FILE_SHA256 = "8ffba0cc153a85c8002f78135c3d04df634e0910325a53e30d7335880bfa993d"
REPORT_CONTENT_SHA256 = "0bb8e35eeb51dca8471814fd61655d5e14ee2e23fb5e9d82f7c2ab50a711730a"
SOURCE_COMMIT = "1afd2532765b01d0959b2f15760eba49ffe56d83"
SOURCE_CONTENT_SHA256 = "3e0d0b91e5ca96788c82ae259fe4bd05dda8529ded803d16bf2b94be0cab8e8e"
IMPLEMENTATION_BUNDLE_SHA256 = (
    "a973416264bfa0b49ab7dc46ad51c5f856fbdc93e9dd80c0cd184f7e664aabad"
)
OVERLAY_BUNDLE_SHA256 = (
    "bfa20f915dd7237a471f2db296f7c1e70ee163bf540979ec576f2002ad113ec5"
)
RECIPE_CONTENT_SHA256 = "8f9b5d874f552062b9d9ecf86a4831dca3ffdcb7299ec6c2875df40619ebe94a"
CONFIGURATION_CONTENT_SHA256 = (
    "9241ba83b280d2f2d727657e5ec49cecbc161b243724158e52ea33263f08eed2"
)
SUMMARY_CONTENT_SHA256 = "1e3bb13e1f3e4e056f52c1208adaed0d650130ea99941f3760fc2a2b06015cc9"
GATE_CONTENT_SHA256 = "d356792d72d742c0e07d2e533c1a0276b8d402ce22fba37af3ada40d77814f24"
RUNTIME_AUDIT_CONTENT_SHA256 = (
    "5640060210774811530748e1bc39ef4bb274707dc86ac3209c116f80d98f0abf"
)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value):
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def validate_completed_v25a_r1():
    if (
        file_sha256(ATTEMPT_PATH) != ATTEMPT_FILE_SHA256
        or file_sha256(REPORT_PATH) != REPORT_FILE_SHA256
    ):
        raise RuntimeError("V25A R1 attempt or report file identity changed")
    attempt = json.loads(ATTEMPT_PATH.read_text(encoding="utf-8"))
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if (
        attempt.get("schema") != "eggroll-es-durable-launch-attempt-v25a"
        or attempt.get("status") != "complete"
        or attempt.get("phase") != "after_cleanup_and_compact_report"
        or attempt.get("content_sha256_before_self_field")
        != ATTEMPT_CONTENT_SHA256
        or canonical_sha256(without_self(attempt)) != ATTEMPT_CONTENT_SHA256
        or report.get("schema")
        != "eggroll-es-paired-data-compat-report-v25a"
        or report.get("content_sha256_before_self_field")
        != REPORT_CONTENT_SHA256
        or canonical_sha256(without_self(report)) != REPORT_CONTENT_SHA256
    ):
        raise RuntimeError("V25A R1 completion identity changed")
    for value in (attempt, report):
        if (
            value.get("model_update_applied") is not False
            or value.get("checkpoint_written") is not False
            or value.get("evaluation_opened") is not False
            or value.get("dataset_promotion_applied") is not False
        ):
            raise RuntimeError("V25A R1 closure semantics changed")
    if report.get("direct_action_taken") is not False:
        raise RuntimeError("V25A R1 direct-action closure changed")
    if attempt.get("report_binding") != {
        "path": str(REPORT_PATH),
        "file_sha256": REPORT_FILE_SHA256,
        "content_sha256": REPORT_CONTENT_SHA256,
    }:
        raise RuntimeError("V25A R1 report binding changed")

    source = attempt.get("source_provenance", {})
    implementation = report.get("implementation", {})
    if (
        source.get("git_head") != SOURCE_COMMIT
        or source.get("content_sha256_before_self_field")
        != SOURCE_CONTENT_SHA256
        or canonical_sha256(without_self(source)) != SOURCE_CONTENT_SHA256
        or source.get("implementation_bundle_sha256")
        != IMPLEMENTATION_BUNDLE_SHA256
        or implementation.get("bundle_sha256")
        != IMPLEMENTATION_BUNDLE_SHA256
        or implementation.get("v25a_overlay_bundle_sha256")
        != OVERLAY_BUNDLE_SHA256
    ):
        raise RuntimeError("V25A R1 committed source identity changed")
    for item in source.get("files", {}).values():
        raw = subprocess.check_output(
            ["git", "show", f"{SOURCE_COMMIT}:{item['relative_path']}"],
            cwd=ROOT,
        )
        if hashlib.sha256(raw).hexdigest() != item["file_sha256"]:
            raise RuntimeError("V25A R1 committed source file changed")

    recipe = report.get("recipe", {})
    configuration = report.get("configuration", {})
    summary = report.get("summary", {})
    gate = report.get("gate", {})
    audit = report.get("runtime_audit", {})
    identities = (
        (recipe, RECIPE_CONTENT_SHA256),
        (configuration, CONFIGURATION_CONTENT_SHA256),
        (summary, SUMMARY_CONTENT_SHA256),
        (gate, GATE_CONTENT_SHA256),
        (audit, RUNTIME_AUDIT_CONTENT_SHA256),
    )
    if any(
        value.get("content_sha256_before_self_field") != expected
        or canonical_sha256(without_self(value)) != expected
        for value, expected in identities
    ):
        raise RuntimeError("V25A R1 aggregate component identity changed")
    endpoints = summary.get("paired_bootstrap", {}).get("endpoints", {})
    observed = [
        value.get("candidate_v364_minus_production")
        for value in endpoints.values()
    ]
    lcbs = [value.get("familywise_lcb") for value in endpoints.values()]
    integrity = summary.get("runtime_integrity", {})
    guard = audit.get("full_context_guard", {})
    if (
        len(endpoints) != 12
        or not all(isinstance(value, (int, float)) and value >= 0 for value in observed)
        or sum(value > 0 for value in observed) != 11
        or not all(isinstance(value, (int, float)) and value < 0 for value in lcbs)
        or gate.get("pass") is not False
        or gate.get("all_12_familywise_lcbs_nonnegative") is not False
        or gate.get("all_runtime_integrity_audits_passed") is not True
        or gate.get("decision") != "retain_production_dataset_and_v13_recipe"
        or not integrity or not all(integrity.values())
        or audit.get("signed_wave_count") != 16
        or audit.get("perturbed_requests_all_engines") != 24_960
        or audit.get("full_context_requests_all_engines") != 4_680
        or audit.get("total_generation_requests") != 29_640
        or not all(guard.get("a_b_exact", {}).values())
        or not all(guard.get("a_c_exact", {}).values())
        or audit.get("model_update_applied") is not False
        or audit.get("evaluation_opened") is not False
        or audit.get("per_unit_scores_or_bootstrap_replicates_persisted")
        is not False
    ):
        raise RuntimeError("V25A R1 aggregate result or integrity changed")
    return attempt, report


def build_evidence():
    _attempt, report = validate_completed_v25a_r1()
    summary = report["summary"]
    endpoints = summary["paired_bootstrap"]["endpoints"]
    observed = {
        key: value["candidate_v364_minus_production"]
        for key, value in endpoints.items()
    }
    lcbs = {key: value["familywise_lcb"] for key, value in endpoints.items()}
    evidence = {
        "schema": "eggroll-es-v25a-paired-promising-unconfirmed-evidence-r1",
        "status": "valid_completed_train_only_gate_failed_for_power",
        "attempt": {
            "path": str(ATTEMPT_PATH),
            "file_sha256": ATTEMPT_FILE_SHA256,
            "content_sha256": ATTEMPT_CONTENT_SHA256,
            "source_commit": SOURCE_COMMIT,
            "source_content_sha256": SOURCE_CONTENT_SHA256,
        },
        "report": {
            "path": str(REPORT_PATH),
            "file_sha256": REPORT_FILE_SHA256,
            "content_sha256": REPORT_CONTENT_SHA256,
            "implementation_bundle_sha256": IMPLEMENTATION_BUNDLE_SHA256,
            "overlay_bundle_sha256": OVERLAY_BUNDLE_SHA256,
            "recipe_content_sha256": RECIPE_CONTENT_SHA256,
        },
        "aggregate_result": {
            "endpoint_count": 12,
            "observed_candidate_minus_production_nonnegative": sum(
                value >= 0 for value in observed.values()
            ),
            "observed_candidate_minus_production_positive": sum(
                value > 0 for value in observed.values()
            ),
            "observed_delta_min": min(observed.values()),
            "observed_delta_max": max(observed.values()),
            "familywise_lcbs_nonnegative": sum(value >= 0 for value in lcbs.values()),
            "best_familywise_lcb": max(lcbs.values()),
            "worst_familywise_lcb": min(lcbs.values()),
            "familywise_alpha": 0.05,
            "bonferroni_endpoint_count": 12,
            "bootstrap_repetitions": 50_000,
        },
        "runtime_integrity": {
            "all_integrity_audits_passed": True,
            "all_four_gpus_scored_both_versions_each_wave": True,
            "all_16_signed_waves_complete": True,
            "all_29_640_generation_requests_accounted": True,
            "full_context_a_b_and_a_c_exact": True,
            "exact_restore_and_population_boundary_audit": True,
            "compact_persistence_only": True,
        },
        "decision": {
            "global_gate_passed": False,
            "reason": "all_observed_deltas_nonnegative_but_no_familywise_lcb_nonnegative",
            "retain_dataset": "production",
            "retain_recipe": "v13",
            "confirmation_authorized_by_this_gate": False,
            "dataset_promotion_authorized": False,
            "model_update_authorized": False,
            "checkpoint_write_authorized": False,
            "evaluation_authorized": False,
        },
        "scope_note": (
            "The exact v364 candidate is directionally promising on this frozen "
            "train-only panel, but the preregistered simultaneous noninferiority "
            "claim was not established. A new design must be independently "
            "preregistered and must not reinterpret this failed gate as a pass."
        ),
        "contains_response_vectors_unit_scores_or_bootstrap_replicates": False,
        "contains_dataset_rows_questions_answers_or_document_content": False,
        "contains_validation_ood_heldout_or_benchmark_content": False,
    }
    evidence["content_sha256_before_self_field"] = canonical_sha256(evidence)
    return evidence


def exclusive_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    evidence = build_evidence()
    if args.output:
        exclusive_write(args.output, evidence)
    else:
        print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
