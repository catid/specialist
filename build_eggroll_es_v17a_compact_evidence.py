#!/usr/bin/env python3
"""Build compact aggregate-only evidence for the negative V17A gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import eggroll_es_paired_data_compat_preregistration_v17a as prereg_v17a
import run_eggroll_es_paired_data_compat_v17a as driver_v17a


ROOT = Path(__file__).resolve().parent
ATTEMPT_PATH_V17A = driver_v17a._attempt_path_v17a().resolve()
REPORT_PATH_V17A = (
    driver_v17a.FROZEN_OUTPUT_DIRECTORY_V17A
    / driver_v17a.EXPERIMENT_NAME_V17A
    / driver_v17a.REPORT_NAME_V17A
).resolve()
OUTPUT_PATH_V17A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V17A_PAIRED_DATA_COMPAT_NEGATIVE_EVIDENCE.json"
).resolve()
ATTEMPT_FILE_SHA256_V17A = (
    "03953036139769564d2004f86f51255f8b7cdf1ff1a0bea09dce2164520ea837"
)
ATTEMPT_CONTENT_SHA256_V17A = (
    "156a1110eca6dbd2bd3ff4dfe6b64e98b5b40b3f4b6b8b2db217d7261913015e"
)
REPORT_FILE_SHA256_V17A = (
    "d1f0f03eb4a15ec473c571b8a1e29912ca362863903031559c92d12259cda3f4"
)
REPORT_CONTENT_SHA256_V17A = (
    "9f3cbb3f64d996d4a6d15928e849b321f1e5b5c5f7dc35a4389496348c55b9c1"
)
SUMMARY_CONTENT_SHA256_V17A = (
    "71b223dd347f9cdfcf1cc878d1b0f0ae7aa245b824a400cf7c836f1240348af0"
)
GATE_CONTENT_SHA256_V17A = (
    "2874754f1647f28690cf12816eea7fc6451bf3a6644d00901e6d8b5367a173b1"
)
RUNTIME_AUDIT_CONTENT_SHA256_V17A = (
    "accadc000cb8274b9000804e49df47e8c6d996212596edc50d8825c78f2bcf94"
)
SOURCE_PROVENANCE_CONTENT_SHA256_V17A = (
    "41f7093c98cd465639b81228d1b2aeb62e94aad1855102c217c7ac64c49bdfe7"
)
IMPLEMENTATION_BUNDLE_SHA256_V17A = (
    "bf299a62640747dfe1c9e2b4f40ef3654835259a95d80e457b6d30432c025aa6"
)
RECIPE_CONTENT_SHA256_V17A = (
    "97c633488032a83a349498caffb0aca0653fcd630d021837ee2f1bb8fd4bc51c"
)

canonical_sha256 = prereg_v17a.canonical_sha256
file_sha256 = prereg_v17a.file_sha256


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _validate_source_provenance(attempt, report):
    source = attempt["source_provenance"]
    implementation = report["implementation"]
    if (
        source.get("schema") != "eggroll-es-committed-source-bundle-v17a"
        or source.get("content_sha256_before_self_field")
        != SOURCE_PROVENANCE_CONTENT_SHA256_V17A
        or source.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(source))
        or implementation.get("bundle_sha256")
        != IMPLEMENTATION_BUNDLE_SHA256_V17A
        or implementation.get("bundle_sha256")
        != canonical_sha256(implementation.get("files"))
        or source.get("implementation_bundle_sha256")
        != implementation.get("bundle_sha256")
        or set(source.get("files", {})) != set(implementation.get("files", {}))
        or len(source.get("files", {})) != 32
    ):
        raise RuntimeError("v17a compact source provenance changed")
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
            raise RuntimeError("v17a launch source differs from committed provenance")
    return source


def load_compact_inputs_v17a():
    """Load only the two aggregate-only runtime artifacts."""
    if (
        file_sha256(ATTEMPT_PATH_V17A) != ATTEMPT_FILE_SHA256_V17A
        or file_sha256(REPORT_PATH_V17A) != REPORT_FILE_SHA256_V17A
    ):
        raise RuntimeError("v17a compact attempt or report file changed")
    attempt = json.loads(ATTEMPT_PATH_V17A.read_text(encoding="utf-8"))
    report = json.loads(REPORT_PATH_V17A.read_text(encoding="utf-8"))
    if (
        attempt.get("content_sha256_before_self_field")
        != ATTEMPT_CONTENT_SHA256_V17A
        or attempt.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(attempt))
        or report.get("content_sha256_before_self_field")
        != REPORT_CONTENT_SHA256_V17A
        or report.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(report))
        or report.get("summary", {}).get("content_sha256_before_self_field")
        != SUMMARY_CONTENT_SHA256_V17A
        or report.get("gate", {}).get("content_sha256_before_self_field")
        != GATE_CONTENT_SHA256_V17A
        or report.get("runtime_audit", {}).get(
            "content_sha256_before_self_field"
        ) != RUNTIME_AUDIT_CONTENT_SHA256_V17A
        or report.get("recipe", {}).get("content_sha256_before_self_field")
        != RECIPE_CONTENT_SHA256_V17A
    ):
        raise RuntimeError("v17a compact content binding changed")
    driver_v17a.validate_compact_report_v17a(
        report,
        expected_recipe=report["recipe"],
        expected_implementation=report["implementation"],
    )
    gate = prereg_v17a.evaluate_candidate_v17a(report["summary"])
    if gate != report["gate"]:
        raise RuntimeError("v17a hardened gate does not recompute")
    expected_attempt_keys = {
        "schema", "status", "phase", "experiment_name", "run_directory",
        "recipe", "source_provenance", "model_update_applied",
        "checkpoint_written", "evaluation_surfaces_opened",
        "content_sha256_before_self_field", "report_exists_after_attempt",
        "report_binding",
    }
    if (
        set(attempt) != expected_attempt_keys
        or attempt.get("schema") != "eggroll-es-durable-launch-attempt-v17a"
        or attempt.get("status") != "complete"
        or attempt.get("phase")
        != "after_trainer_cleanup_and_compact_report"
        or attempt.get("recipe") != report.get("recipe")
        or attempt.get("model_update_applied") is not False
        or attempt.get("checkpoint_written") is not False
        or attempt.get("evaluation_surfaces_opened") is not False
        or attempt.get("report_exists_after_attempt") is not True
        or Path(attempt.get("run_directory")).resolve() != REPORT_PATH_V17A.parent
        or attempt.get("report_binding") != {
            "path": str(REPORT_PATH_V17A),
            "file_sha256": REPORT_FILE_SHA256_V17A,
            "content_sha256": REPORT_CONTENT_SHA256_V17A,
        }
    ):
        raise RuntimeError("v17a completed launch-attempt contract changed")
    source = _validate_source_provenance(attempt, report)
    return attempt, report, source


def build_evidence_v17a():
    attempt, report, source = load_compact_inputs_v17a()
    summary = report["summary"]
    gate = report["gate"]
    bootstrap = summary["paired_bootstrap"]
    endpoints = {}
    for name in prereg_v17a.ENDPOINT_CONTRACT_V17A:
        production = float(
            summary["versions"]["production"]["endpoint_values"][name]
        )
        candidate = float(
            summary["versions"]["candidate_v283"]["endpoint_values"][name]
        )
        bootstrap_item = bootstrap["endpoints"][name]
        delta = float(bootstrap_item["candidate_minus_production"])
        lcb = float(bootstrap_item["familywise_lcb"])
        if (
            not all(math.isfinite(item) for item in (production, candidate, delta, lcb))
            or not math.isclose(
                candidate - production, delta, rel_tol=1e-12, abs_tol=1e-12,
            )
            or gate["observed_candidate_not_below_production"][name]
            != (candidate >= production)
            or gate["paired_noninferiority_results"][name] != (lcb >= 0.0)
            or bootstrap_item["noninferiority_margin"] != 0.0
        ):
            raise RuntimeError(f"v17a endpoint {name} does not recompute")
        endpoints[name] = {
            "candidate_minus_production": delta,
            "familywise_lcb": lcb,
            "observed_noninferiority_pass": candidate >= production,
            "bootstrap_noninferiority_pass": lcb >= 0.0,
        }
    observed_pass_count = sum(
        item["observed_noninferiority_pass"] for item in endpoints.values()
    )
    bootstrap_pass_count = sum(
        item["bootstrap_noninferiority_pass"] for item in endpoints.values()
    )
    runtime_integrity = summary["runtime_integrity"]
    runtime_audit = report["runtime_audit"]
    if (
        bootstrap.get("seed") != prereg_v17a.BOOTSTRAP_SEED_V17A
        or bootstrap.get("repetitions") != prereg_v17a.BOOTSTRAP_REPETITIONS_V17A
        or bootstrap.get("one_sided_quantile")
        != prereg_v17a.FAMILYWISE_ALPHA_V17A / len(endpoints)
        or len(endpoints) != 12
        or observed_pass_count != 9
        or bootstrap_pass_count != 0
        or gate.get("eligible_for_separate_v17b_preregistration") is not False
        or any(value is not True for value in runtime_integrity.values())
        or runtime_audit.get("signed_wave_count") != 16
        or runtime_audit.get("per_unit_scores_persisted") is not False
        or runtime_audit.get("bootstrap_replicates_persisted") is not False
        or report["recipe"].get("moe_backend")
        != driver_v17a._declared_moe_backend_v17a()
        or report["recipe"].get("hardware") != {
            "engine_count": 4,
            "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "all_engines_every_signed_wave": True,
        }
    ):
        raise RuntimeError("v17a aggregate negative-gate contract changed")
    evidence = {
        "schema": "eggroll-es-paired-data-compat-negative-evidence-v17a",
        "status": "valid_completed_run_preregistered_gate_failed",
        "input_artifacts": {
            "launch_attempt": {
                "path": str(ATTEMPT_PATH_V17A),
                "file_sha256": ATTEMPT_FILE_SHA256_V17A,
                "content_sha256": ATTEMPT_CONTENT_SHA256_V17A,
            },
            "compact_report": {
                "path": str(REPORT_PATH_V17A),
                "file_sha256": REPORT_FILE_SHA256_V17A,
                "content_sha256": REPORT_CONTENT_SHA256_V17A,
            },
            "summary_content_sha256": SUMMARY_CONTENT_SHA256_V17A,
            "gate_content_sha256": GATE_CONTENT_SHA256_V17A,
        },
        "source_provenance": {
            "git_head": source["git_head"],
            "file_count": len(source["files"]),
            "content_sha256": SOURCE_PROVENANCE_CONTENT_SHA256_V17A,
            "implementation_bundle_sha256": IMPLEMENTATION_BUNDLE_SHA256_V17A,
            "all_files_match_launch_commit": True,
        },
        "bootstrap": {
            "seed": bootstrap["seed"],
            "repetitions": bootstrap["repetitions"],
            "one_sided_quantile": bootstrap["one_sided_quantile"],
            "familywise_alpha": prereg_v17a.FAMILYWISE_ALPHA_V17A,
            "endpoint_count": len(endpoints),
            "paired_within_panel_stratum": True,
            "replicates_persisted": False,
        },
        "endpoints": endpoints,
        "gate_summary": {
            "observed_pass_count": observed_pass_count,
            "observed_required_count": len(endpoints),
            "bootstrap_pass_count": bootstrap_pass_count,
            "bootstrap_required_count": len(endpoints),
            "all_rules_conjunctive": True,
            "preregistered_gate_passed": False,
        },
        "runtime_integrity": {
            "checks": runtime_integrity,
            "signed_wave_count": runtime_audit["signed_wave_count"],
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
            "retain_recipe": "v13",
            "separate_v17b_preregistration_authorized": False,
            "dataset_promotion_authorized": False,
            "model_update_authorized": False,
            "evaluation_authorized": False,
        },
        "contains_response_vectors_or_row_content": False,
        "contains_validation_ood_heldout_or_benchmark_content": False,
        "interpretation": (
            "valid_train_only_run_but_v283_failed_zero_margin_familywise_"
            "compatibility_gate"
        ),
    }
    evidence["content_sha256_before_self_field"] = canonical_sha256(evidence)
    validate_evidence_v17a(evidence)
    return evidence


def validate_evidence_v17a(evidence):
    endpoints = evidence.get("endpoints", {}) if isinstance(evidence, dict) else {}
    endpoint_contract_valid = (
        set(endpoints) == set(prereg_v17a.ENDPOINT_CONTRACT_V17A)
        and all(
            set(item) == {
                "candidate_minus_production", "familywise_lcb",
                "observed_noninferiority_pass",
                "bootstrap_noninferiority_pass",
            }
            and isinstance(item["candidate_minus_production"], (int, float))
            and not isinstance(item["candidate_minus_production"], bool)
            and math.isfinite(float(item["candidate_minus_production"]))
            and isinstance(item["familywise_lcb"], (int, float))
            and not isinstance(item["familywise_lcb"], bool)
            and math.isfinite(float(item["familywise_lcb"]))
            and item["observed_noninferiority_pass"]
            is (float(item["candidate_minus_production"]) >= 0.0)
            and item["bootstrap_noninferiority_pass"]
            is (float(item["familywise_lcb"]) >= 0.0)
            for item in endpoints.values()
        )
        and sum(
            item["observed_noninferiority_pass"] for item in endpoints.values()
        ) == 9
        and sum(
            item["bootstrap_noninferiority_pass"] for item in endpoints.values()
        ) == 0
    )
    if (
        not isinstance(evidence, dict)
        or set(evidence) != {
            "schema", "status", "input_artifacts", "source_provenance",
            "bootstrap", "endpoints", "gate_summary", "runtime_integrity",
            "backend", "hardware", "decision",
            "contains_response_vectors_or_row_content",
            "contains_validation_ood_heldout_or_benchmark_content",
            "interpretation", "content_sha256_before_self_field",
        }
        or evidence.get("schema")
        != "eggroll-es-paired-data-compat-negative-evidence-v17a"
        or evidence.get("status")
        != "valid_completed_run_preregistered_gate_failed"
        or evidence.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(evidence))
        or not endpoint_contract_valid
        or evidence.get("gate_summary") != {
            "observed_pass_count": 9,
            "observed_required_count": 12,
            "bootstrap_pass_count": 0,
            "bootstrap_required_count": 12,
            "all_rules_conjunctive": True,
            "preregistered_gate_passed": False,
        }
        or evidence.get("decision") != {
            "retain_dataset": "production",
            "retain_recipe": "v13",
            "separate_v17b_preregistration_authorized": False,
            "dataset_promotion_authorized": False,
            "model_update_authorized": False,
            "evaluation_authorized": False,
        }
        or evidence.get("contains_response_vectors_or_row_content") is not False
        or evidence.get(
            "contains_validation_ood_heldout_or_benchmark_content"
        ) is not False
    ):
        raise RuntimeError("v17a compact negative evidence changed")
    prereg_v17a.validate_persisted_sha256_fields_v17a(evidence)
    return evidence


def _exclusive_write(path, value):
    path = Path(path).resolve()
    if path != OUTPUT_PATH_V17A:
        raise ValueError("v17a compact evidence output path changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("v17a compact evidence already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH_V17A))
    args = parser.parse_args(argv)
    if Path(args.output).resolve() != OUTPUT_PATH_V17A:
        raise ValueError("v17a compact evidence output path changed")
    evidence = build_evidence_v17a()
    _exclusive_write(args.output, evidence)
    result = {
        "schema": "eggroll-es-paired-data-compat-evidence-write-v17a",
        "path": str(OUTPUT_PATH_V17A),
        "file_sha256": file_sha256(OUTPUT_PATH_V17A),
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
