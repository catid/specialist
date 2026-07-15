#!/usr/bin/env python3
"""Build compact aggregate-only evidence for the negative V18A patch gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import eggroll_es_production_overlay_scan_preregistration_v18a as prereg_v18a
import run_eggroll_es_production_patch_compat_v18a as driver_v18a
import train_eggroll_es_production_patch_compat_v18a as mechanics_v18a


ROOT = Path(__file__).resolve().parent
LAUNCH_ID_V18A = "1784085367124161970-80586-5e68ee14f55be587"
RUN_ROOT_V18A = (
    driver_v18a.FROZEN_OUTPUT_DIRECTORY_V18A
    / driver_v18a.EXPERIMENT_NAME_V18A
).resolve()
ATTEMPT_PATH_V18A = (
    RUN_ROOT_V18A / "attempts" / f"{LAUNCH_ID_V18A}.json"
).resolve()
REPORT_PATH_V18A = (
    RUN_ROOT_V18A / "runs" / LAUNCH_ID_V18A / driver_v18a.REPORT_NAME_V18A
).resolve()
OUTPUT_PATH_V18A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V18A_PRODUCTION_PATCH_COMPAT_NEGATIVE_EVIDENCE.json"
).resolve()

ATTEMPT_FILE_SHA256_V18A = (
    "44340402296a814e1890520bc8f10e2df2364df023a30a9f3b9e68c15282b02d"
)
ATTEMPT_CONTENT_SHA256_V18A = (
    "1b6701416f5e8fe8e14fdb2ee1134d67fcbd847d3d402d463535d76aabc5a817"
)
REPORT_FILE_SHA256_V18A = (
    "efcc2e2c2317a072937d4bc34c27ed2b22d6e33cc0ae0e7c5873aba457b4f14e"
)
REPORT_CONTENT_SHA256_V18A = (
    "4dfc8a653bbcacb2d0760439b0e2efc91db935d4198eb19898f5c0dc2b1a326a"
)
SUMMARY_CONTENT_SHA256_V18A = (
    "61daae6f2072e8f73c3ac37c88a23f62c333e819e4ae125fb8e62025b739388d"
)
GATE_CONTENT_SHA256_V18A = (
    "b0ddce7920c8dce23f7bda50ccdde8f432319eacf8423460045e7c01beaf38d4"
)
RUNTIME_AUDIT_CONTENT_SHA256_V18A = (
    "d19f74e9bdff9e20aebcb1b5cf0f8895d9370fd54ca6bed55d75d673d78a629b"
)
SOURCE_PROVENANCE_CONTENT_SHA256_V18A = (
    "23dc2170341c94167bad12511b3cf4ec5ccbcbb05d90ac877fc9ee59de0aa68f"
)
IMPLEMENTATION_BUNDLE_SHA256_V18A = (
    "3e16f53f979c7b1090ccc885a6cbf07f111b9f5a21d64b798cbad8a7ebc3874b"
)
RECIPE_CONTENT_SHA256_V18A = (
    "179fb1d0221fd793a2d35349ef8dcfe19d2e14e084d7a202044221e31e34fb3a"
)
EXPECTED_OBSERVED_PASS_COUNTS_V18A = {
    "patch_one_third": 5,
    "patch_two_thirds": 8,
    "patch_full": 7,
}

canonical_sha256 = prereg_v18a.canonical_sha256
file_sha256 = prereg_v18a.file_sha256


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _validate_source_provenance(attempt, report):
    source = attempt["source_provenance"]
    implementation = report["implementation"]
    if (
        source.get("schema") != "eggroll-es-committed-source-bundle-v18a"
        or source.get("content_sha256_before_self_field")
        != SOURCE_PROVENANCE_CONTENT_SHA256_V18A
        or source.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(source))
        or implementation.get("bundle_sha256")
        != IMPLEMENTATION_BUNDLE_SHA256_V18A
        or implementation.get("bundle_sha256")
        != canonical_sha256(implementation.get("files"))
        or source.get("implementation_bundle_sha256")
        != implementation.get("bundle_sha256")
        or set(source.get("files", {})) != set(implementation.get("files", {}))
        or len(source.get("files", {})) != 33
    ):
        raise RuntimeError("v18a compact source provenance changed")
    commit = source.get("git_head")
    subprocess.check_call(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=ROOT,
    )
    for key, item in implementation["files"].items():
        path = Path(item["path"]).resolve()
        relative = path.relative_to(ROOT).as_posix()
        raw = subprocess.check_output(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
        )
        digest = hashlib.sha256(raw).hexdigest()
        if (
            digest != item["file_sha256"]
            or source["files"].get(key) != {
                "relative_path": relative,
                "file_sha256": digest,
            }
        ):
            raise RuntimeError(
                "v18a launch source differs from committed provenance"
            )
    return source


def load_compact_inputs_v18a():
    """Load only the aggregate-only launch attempt and compact report."""
    if (
        file_sha256(ATTEMPT_PATH_V18A) != ATTEMPT_FILE_SHA256_V18A
        or file_sha256(REPORT_PATH_V18A) != REPORT_FILE_SHA256_V18A
    ):
        raise RuntimeError("v18a compact attempt or report file changed")
    attempt = json.loads(ATTEMPT_PATH_V18A.read_text(encoding="utf-8"))
    report = json.loads(REPORT_PATH_V18A.read_text(encoding="utf-8"))
    if (
        attempt.get("content_sha256_before_self_field")
        != ATTEMPT_CONTENT_SHA256_V18A
        or attempt.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(attempt))
        or report.get("content_sha256_before_self_field")
        != REPORT_CONTENT_SHA256_V18A
        or report.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(report))
        or report.get("summary", {}).get("content_sha256_before_self_field")
        != SUMMARY_CONTENT_SHA256_V18A
        or canonical_sha256(report.get("gate", {}))
        != GATE_CONTENT_SHA256_V18A
        or report.get("runtime_audit", {}).get(
            "content_sha256_before_self_field"
        ) != RUNTIME_AUDIT_CONTENT_SHA256_V18A
        or report.get("recipe", {}).get("content_sha256_before_self_field")
        != RECIPE_CONTENT_SHA256_V18A
    ):
        raise RuntimeError("v18a compact content binding changed")
    driver_v18a.validate_compact_report_v18a(
        report,
        expected_recipe=report["recipe"],
        expected_implementation=report["implementation"],
    )
    if mechanics_v18a.evaluate_patch_gate_v18a(report["summary"]) != report["gate"]:
        raise RuntimeError("v18a hardened gate does not recompute")
    expected_attempt_keys = {
        "schema", "status", "phase", "experiment_name", "launch_id",
        "run_directory", "recipe", "source_provenance",
        "model_update_applied", "checkpoint_written",
        "evaluation_surfaces_opened", "content_sha256_before_self_field",
        "report_exists_after_attempt", "report_binding",
    }
    if (
        set(attempt) != expected_attempt_keys
        or attempt.get("schema") != "eggroll-es-durable-launch-attempt-v18a"
        or attempt.get("status") != "complete"
        or attempt.get("phase")
        != "after_trainer_cleanup_and_compact_report"
        or attempt.get("launch_id") != LAUNCH_ID_V18A
        or attempt.get("recipe") != report.get("recipe")
        or attempt.get("model_update_applied") is not False
        or attempt.get("checkpoint_written") is not False
        or attempt.get("evaluation_surfaces_opened") is not False
        or attempt.get("report_exists_after_attempt") is not True
        or Path(attempt.get("run_directory")).resolve() != REPORT_PATH_V18A.parent
        or attempt.get("report_binding") != {
            "path": str(REPORT_PATH_V18A),
            "file_sha256": REPORT_FILE_SHA256_V18A,
            "content_sha256": REPORT_CONTENT_SHA256_V18A,
        }
    ):
        raise RuntimeError("v18a completed launch-attempt contract changed")
    source = _validate_source_provenance(attempt, report)
    return attempt, report, source


def build_evidence_v18a():
    _, report, source = load_compact_inputs_v18a()
    summary = report["summary"]
    gate = report["gate"]
    bootstrap = summary["paired_bootstrap"]
    production = summary["arms"]["production_only"]["endpoint_values"]
    arms = {}
    for arm in mechanics_v18a.ARMS_V18A[1:]:
        endpoints = {}
        for name in prereg_v18a.ENDPOINT_NAMES_V18A:
            candidate = float(summary["arms"][arm]["endpoint_values"][name])
            baseline = float(production[name])
            item = bootstrap["comparisons"][arm][name]
            delta = float(item["patch_minus_production"])
            lcb = float(item["familywise_lcb"])
            if (
                not all(math.isfinite(value) for value in (
                    candidate, baseline, delta, lcb,
                ))
                or not math.isclose(
                    candidate - baseline, delta, rel_tol=1e-12, abs_tol=1e-12,
                )
                or item["noninferiority_margin"] != 0.0
            ):
                raise RuntimeError(f"v18a endpoint {arm}/{name} does not recompute")
            endpoints[name] = {
                "patch_minus_production": delta,
                "familywise_lcb": lcb,
                "observed_noninferiority_pass": delta >= 0.0,
                "bootstrap_noninferiority_pass": lcb >= 0.0,
            }
        observed = sum(
            item["observed_noninferiority_pass"] for item in endpoints.values()
        )
        bootstrap_pass = sum(
            item["bootstrap_noninferiority_pass"] for item in endpoints.values()
        )
        gate_arm = gate["arms"][arm]
        if (
            len(endpoints) != 12
            or observed != EXPECTED_OBSERVED_PASS_COUNTS_V18A[arm]
            or bootstrap_pass != 0
            or gate_arm["observed_pass_count"] != observed
            or gate_arm["bootstrap_pass_count"] != bootstrap_pass
            or gate_arm["preregistered_gate_passed"] is not False
        ):
            raise RuntimeError(f"v18a negative gate changed for {arm}")
        arms[arm] = {
            "endpoints": endpoints,
            "observed_pass_count": observed,
            "bootstrap_pass_count": bootstrap_pass,
            "preregistered_gate_passed": False,
        }
    runtime_integrity = summary["runtime_integrity"]
    runtime_audit = report["runtime_audit"]
    if (
        bootstrap.get("seed") != prereg_v18a.BOOTSTRAP_SEED_V18A
        or bootstrap.get("repetitions") != 50_000
        or bootstrap.get("one_sided_quantile") != 0.05 / 36
        or any(value is not True for value in runtime_integrity.values())
        or runtime_audit.get("signed_wave_count") != 16
        or runtime_audit.get("requests_per_engine_per_signed_wave") != 1070
        or runtime_audit.get("per_unit_scores_persisted") is not False
        or runtime_audit.get("bootstrap_replicates_persisted") is not False
        or gate.get("any_patch_passed") is not False
        or gate.get("selected_largest_passing_patch") is not None
        or gate.get("decision") != "retain_production_dataset_and_v13_recipe"
        or report["recipe"].get("hardware") != {
            "engine_count": 4,
            "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "all_engines_every_signed_wave": True,
        }
    ):
        raise RuntimeError("v18a aggregate negative-gate contract changed")
    evidence = {
        "schema": "eggroll-es-production-patch-compat-negative-evidence-v18a",
        "status": "valid_completed_run_preregistered_gate_failed",
        "input_artifacts": {
            "launch_attempt": {
                "path": str(ATTEMPT_PATH_V18A),
                "file_sha256": ATTEMPT_FILE_SHA256_V18A,
                "content_sha256": ATTEMPT_CONTENT_SHA256_V18A,
            },
            "compact_report": {
                "path": str(REPORT_PATH_V18A),
                "file_sha256": REPORT_FILE_SHA256_V18A,
                "content_sha256": REPORT_CONTENT_SHA256_V18A,
            },
            "summary_content_sha256": SUMMARY_CONTENT_SHA256_V18A,
            "gate_content_sha256": GATE_CONTENT_SHA256_V18A,
        },
        "source_provenance": {
            "git_head": source["git_head"],
            "file_count": len(source["files"]),
            "content_sha256": SOURCE_PROVENANCE_CONTENT_SHA256_V18A,
            "implementation_bundle_sha256": IMPLEMENTATION_BUNDLE_SHA256_V18A,
            "all_files_match_launch_commit": True,
        },
        "bootstrap": {
            "seed": bootstrap["seed"],
            "repetitions": bootstrap["repetitions"],
            "one_sided_quantile": bootstrap["one_sided_quantile"],
            "familywise_alpha": 0.05,
            "arm_count": len(arms),
            "endpoint_count_per_arm": 12,
            "total_endpoint_count": 36,
            "paired_and_stratified": True,
            "replicates_persisted": False,
        },
        "arms": arms,
        "gate_summary": {
            "observed_pass_counts": {
                arm: value["observed_pass_count"] for arm, value in arms.items()
            },
            "bootstrap_pass_counts": {
                arm: value["bootstrap_pass_count"] for arm, value in arms.items()
            },
            "required_per_arm": 12,
            "all_rules_conjunctive": True,
            "any_patch_passed": False,
        },
        "runtime_integrity": {
            "checks": runtime_integrity,
            "signed_wave_count": runtime_audit["signed_wave_count"],
            "requests_per_engine_per_signed_wave": runtime_audit[
                "requests_per_engine_per_signed_wave"
            ],
            "fixed_request_identity_sha256": runtime_audit[
                "fixed_request_identity_sha256"
            ],
            "pre_post_probe_identity_sha256": runtime_audit[
                "pre_post_probe_identity_sha256"
            ],
            "restore_checks_sha256": runtime_audit["restore_checks_sha256"],
            "population_boundary_audit_sha256": runtime_audit[
                "population_boundary_audit_sha256"
            ],
            "all_four_gpus_observed_every_signed_wave": True,
            "exact_restoration_and_boundary_audit_passed": True,
            "per_unit_scores_persisted": False,
        },
        "backend": report["recipe"]["moe_backend"],
        "hardware": report["recipe"]["hardware"],
        "decision": {
            "retain_dataset": "production",
            "retain_recipe": "v13_middle_late_layers_20_23",
            "separate_train_only_recipe_preregistration_authorized": False,
            "dataset_promotion_authorized": False,
            "model_update_authorized": False,
            "evaluation_authorized": False,
        },
        "contains_response_vectors_or_row_content": False,
        "contains_validation_ood_heldout_or_benchmark_content": False,
        "interpretation": (
            "valid_train_only_run_but_all_three_patch_sizes_failed_zero_margin_"
            "familywise_compatibility_gate"
        ),
    }
    evidence["content_sha256_before_self_field"] = canonical_sha256(evidence)
    validate_evidence_v18a(evidence)
    return evidence


def validate_evidence_v18a(evidence):
    arms = evidence.get("arms", {}) if isinstance(evidence, dict) else {}
    arms_valid = (
        set(arms) == set(EXPECTED_OBSERVED_PASS_COUNTS_V18A)
        and all(
            set(value) == {
                "endpoints", "observed_pass_count", "bootstrap_pass_count",
                "preregistered_gate_passed",
            }
            and len(value["endpoints"]) == 12
            and value["observed_pass_count"]
            == EXPECTED_OBSERVED_PASS_COUNTS_V18A[arm]
            and value["bootstrap_pass_count"] == 0
            and value["preregistered_gate_passed"] is False
            and all(
                set(item) == {
                    "patch_minus_production", "familywise_lcb",
                    "observed_noninferiority_pass",
                    "bootstrap_noninferiority_pass",
                }
                and item["observed_noninferiority_pass"]
                is (float(item["patch_minus_production"]) >= 0.0)
                and item["bootstrap_noninferiority_pass"]
                is (float(item["familywise_lcb"]) >= 0.0)
                for item in value["endpoints"].values()
            )
            for arm, value in arms.items()
        )
    )
    if (
        not isinstance(evidence, dict)
        or set(evidence) != {
            "schema", "status", "input_artifacts", "source_provenance",
            "bootstrap", "arms", "gate_summary", "runtime_integrity",
            "backend", "hardware", "decision",
            "contains_response_vectors_or_row_content",
            "contains_validation_ood_heldout_or_benchmark_content",
            "interpretation", "content_sha256_before_self_field",
        }
        or evidence.get("schema")
        != "eggroll-es-production-patch-compat-negative-evidence-v18a"
        or evidence.get("status")
        != "valid_completed_run_preregistered_gate_failed"
        or evidence.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(evidence))
        or not arms_valid
        or evidence.get("gate_summary") != {
            "observed_pass_counts": EXPECTED_OBSERVED_PASS_COUNTS_V18A,
            "bootstrap_pass_counts": {
                arm: 0 for arm in EXPECTED_OBSERVED_PASS_COUNTS_V18A
            },
            "required_per_arm": 12,
            "all_rules_conjunctive": True,
            "any_patch_passed": False,
        }
        or evidence.get("decision") != {
            "retain_dataset": "production",
            "retain_recipe": "v13_middle_late_layers_20_23",
            "separate_train_only_recipe_preregistration_authorized": False,
            "dataset_promotion_authorized": False,
            "model_update_authorized": False,
            "evaluation_authorized": False,
        }
        or evidence.get("contains_response_vectors_or_row_content") is not False
        or evidence.get(
            "contains_validation_ood_heldout_or_benchmark_content"
        ) is not False
    ):
        raise RuntimeError("v18a compact negative evidence changed")
    prereg_v18a.prereg_v17a.validate_persisted_sha256_fields_v17a(evidence)
    return evidence


def _exclusive_write(path, value):
    path = Path(path).resolve()
    if path != OUTPUT_PATH_V18A:
        raise ValueError("v18a compact evidence output path changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("v18a compact evidence already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH_V18A))
    args = parser.parse_args(argv)
    if Path(args.output).resolve() != OUTPUT_PATH_V18A:
        raise ValueError("v18a compact evidence output path changed")
    evidence = build_evidence_v18a()
    _exclusive_write(args.output, evidence)
    result = {
        "schema": "eggroll-es-production-patch-evidence-write-v18a",
        "path": str(OUTPUT_PATH_V18A),
        "file_sha256": file_sha256(OUTPUT_PATH_V18A),
        "content_sha256": evidence["content_sha256_before_self_field"],
        "preregistered_gate_passed": False,
        "model_update_authorized": False,
        "evaluation_authorized": False,
        "gpu_launched": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
