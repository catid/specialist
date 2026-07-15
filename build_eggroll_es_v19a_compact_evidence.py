#!/usr/bin/env python3
"""Build aggregate-only evidence for the negative V19A tier attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import eggroll_es_disjoint_tier_attribution_preregistration_v19a as prereg_v19a
import run_eggroll_es_disjoint_tier_attribution_v19a as driver_v19a
import train_eggroll_es_disjoint_tier_attribution_v19a as mechanics_v19a


ROOT = Path(__file__).resolve().parent
LAUNCH_ID_V19A = "1784089171975392026-148640-d2cca7b7fea6e833"
RUN_ROOT_V19A = (
    driver_v19a.FROZEN_OUTPUT_DIRECTORY_V19A
    / driver_v19a.EXPERIMENT_NAME_V19A
).resolve()
ATTEMPT_PATH_V19A = (
    RUN_ROOT_V19A / "attempts" / f"{LAUNCH_ID_V19A}.json"
).resolve()
REPORT_PATH_V19A = (
    RUN_ROOT_V19A / "runs" / LAUNCH_ID_V19A / driver_v19a.REPORT_NAME_V19A
).resolve()
OUTPUT_PATH_V19A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V19A_DISJOINT_TIER_ATTRIBUTION_NEGATIVE_EVIDENCE.json"
).resolve()

ATTEMPT_FILE_SHA256_V19A = (
    "f9142c2a00d92b2fff7054968688c940bf8cfdc69296d32a34dd9c41957f5f9d"
)
ATTEMPT_CONTENT_SHA256_V19A = (
    "909b2ca817c76a531fded34f3622dc02aa5cd03494967b42a8044784f19418c5"
)
REPORT_FILE_SHA256_V19A = (
    "bbffe518097c239f7ce613272077ae090a82bfd32de4b79b3f67b796e5ed13a6"
)
REPORT_CONTENT_SHA256_V19A = (
    "bcdb96107c12b374b343773947182537e9b370a4805c5ccb941d935ac0646e0d"
)
SUMMARY_CONTENT_SHA256_V19A = (
    "6fdb60e0c45508eaa574c319e457b6978d69a18127e9a955ccdf2eba38d0cad3"
)
GATE_CONTENT_SHA256_V19A = (
    "f905dfcdd0253b235b9825f8d00b8efa21fe6eb7d63483dfa1a5710db31b9d80"
)
RUNTIME_AUDIT_CONTENT_SHA256_V19A = (
    "3dad57318ad39aaa3baf64276386a7f4ce76e5bc238aed0ab9385cc226969be9"
)
SOURCE_PROVENANCE_CONTENT_SHA256_V19A = (
    "2b40010689f860ec18b2f9b419902a2234de2492d58219e7ac7b7af43fe063e1"
)
IMPLEMENTATION_BUNDLE_SHA256_V19A = (
    "73ab67f230993810403171d2d18cc9abfbc4ab161a05356d87345c628ca0df4d"
)
RECIPE_CONTENT_SHA256_V19A = (
    "4ba55599d8ac60de342af7a2f9874ba8d54710a4be88b82829703e531e556ff0"
)
EXPECTED_OBSERVED_PASS_COUNTS_V19A = {
    "patch_tier_1_only": 2,
    "patch_tier_2_only": 5,
    "patch_tier_3_only": 4,
}

canonical_sha256 = prereg_v19a.canonical_sha256
file_sha256 = prereg_v19a.file_sha256


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _validate_source_provenance(attempt, report):
    source = attempt["source_provenance"]
    implementation = report["implementation"]
    if (
        source.get("schema") != "eggroll-es-committed-source-bundle-v19a"
        or source.get("content_sha256_before_self_field")
        != SOURCE_PROVENANCE_CONTENT_SHA256_V19A
        or source.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(source))
        or implementation.get("bundle_sha256")
        != IMPLEMENTATION_BUNDLE_SHA256_V19A
        or implementation.get("bundle_sha256")
        != canonical_sha256(implementation.get("files"))
        or source.get("implementation_bundle_sha256")
        != implementation.get("bundle_sha256")
        or set(source.get("files", {})) != set(implementation.get("files", {}))
        or len(source.get("files", {})) != 43
    ):
        raise RuntimeError("v19a compact source provenance changed")
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
                "v19a launch source differs from committed provenance"
            )
    return source


def load_compact_inputs_v19a():
    """Load and validate only the attempt and aggregate-only compact report."""
    if (
        file_sha256(ATTEMPT_PATH_V19A) != ATTEMPT_FILE_SHA256_V19A
        or file_sha256(REPORT_PATH_V19A) != REPORT_FILE_SHA256_V19A
    ):
        raise RuntimeError("v19a compact attempt or report file changed")
    attempt = json.loads(ATTEMPT_PATH_V19A.read_text(encoding="utf-8"))
    report = json.loads(REPORT_PATH_V19A.read_text(encoding="utf-8"))
    if (
        attempt.get("content_sha256_before_self_field")
        != ATTEMPT_CONTENT_SHA256_V19A
        or attempt.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(attempt))
        or report.get("content_sha256_before_self_field")
        != REPORT_CONTENT_SHA256_V19A
        or report.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(report))
        or report.get("summary", {}).get("content_sha256_before_self_field")
        != SUMMARY_CONTENT_SHA256_V19A
        or canonical_sha256(report.get("gate", {}))
        != GATE_CONTENT_SHA256_V19A
        or report.get("runtime_audit", {}).get(
            "content_sha256_before_self_field"
        ) != RUNTIME_AUDIT_CONTENT_SHA256_V19A
        or report.get("recipe", {}).get("content_sha256_before_self_field")
        != RECIPE_CONTENT_SHA256_V19A
    ):
        raise RuntimeError("v19a compact content binding changed")
    driver_v19a.validate_compact_report_v19a(
        report,
        expected_recipe=report["recipe"],
        expected_implementation=report["implementation"],
    )
    if mechanics_v19a.evaluate_attribution_gate_v19a(
        report["summary"]
    ) != report["gate"]:
        raise RuntimeError("v19a hardened gate does not recompute")
    expected_attempt_keys = {
        "schema", "status", "phase", "experiment_name", "launch_id",
        "run_directory", "recipe", "source_provenance",
        "model_update_applied", "checkpoint_written",
        "evaluation_surfaces_opened", "dataset_promotion_applied",
        "content_sha256_before_self_field", "report_exists_after_attempt",
        "report_binding",
    }
    if (
        set(attempt) != expected_attempt_keys
        or attempt.get("schema") != "eggroll-es-durable-launch-attempt-v19a"
        or attempt.get("status") != "complete"
        or attempt.get("phase")
        != "after_trainer_cleanup_and_compact_report"
        or attempt.get("launch_id") != LAUNCH_ID_V19A
        or attempt.get("recipe") != report.get("recipe")
        or attempt.get("model_update_applied") is not False
        or attempt.get("checkpoint_written") is not False
        or attempt.get("evaluation_surfaces_opened") is not False
        or attempt.get("dataset_promotion_applied") is not False
        or attempt.get("report_exists_after_attempt") is not True
        or Path(attempt.get("run_directory")).resolve() != REPORT_PATH_V19A.parent
        or attempt.get("report_binding") != {
            "path": str(REPORT_PATH_V19A),
            "file_sha256": REPORT_FILE_SHA256_V19A,
            "content_sha256": REPORT_CONTENT_SHA256_V19A,
        }
    ):
        raise RuntimeError("v19a completed launch-attempt contract changed")
    source = _validate_source_provenance(attempt, report)
    return attempt, report, source


def build_evidence_v19a():
    _, report, source = load_compact_inputs_v19a()
    summary = report["summary"]
    gate = report["gate"]
    bootstrap = summary["paired_bootstrap"]
    production = summary["arms"]["production_only"]["endpoint_values"]
    arms = {}
    for arm in mechanics_v19a.ARMS_V19A[1:]:
        endpoints = {}
        for name in prereg_v19a.ENDPOINT_NAMES_V19A:
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
                raise RuntimeError(f"v19a endpoint {arm}/{name} changed")
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
            or observed != EXPECTED_OBSERVED_PASS_COUNTS_V19A[arm]
            or bootstrap_pass != 0
            or gate_arm["observed_pass_count"] != observed
            or gate_arm["bootstrap_pass_count"] != bootstrap_pass
            or gate_arm["preregistered_attribution_gate_passed"] is not False
        ):
            raise RuntimeError(f"v19a negative tier gate changed for {arm}")
        arms[arm] = {
            "endpoints": endpoints,
            "observed_pass_count": observed,
            "bootstrap_pass_count": bootstrap_pass,
            "preregistered_attribution_gate_passed": False,
        }
    runtime_integrity = summary["runtime_integrity"]
    runtime_audit = report["runtime_audit"]
    if (
        bootstrap.get("seed") != prereg_v19a.BOOTSTRAP_SEED_V19A
        or bootstrap.get("repetitions") != 50_000
        or bootstrap.get("one_sided_quantile") != 0.05 / 36
        or bootstrap.get("whole_panel_block_resampling_used") is not False
        or any(value is not True for value in runtime_integrity.values())
        or runtime_audit.get("signed_wave_count") != 16
        or runtime_audit.get("panel_count") != 10
        or runtime_audit.get("requests_per_engine_per_signed_wave") != 990
        or runtime_audit.get("dense_result_commitment_count") != 2560
        or runtime_audit.get("per_unit_scores_persisted") is not False
        or runtime_audit.get("bootstrap_replicates_persisted") is not False
        or runtime_audit.get("bootstrap_draws_persisted") is not False
        or runtime_audit.get("row_content_persisted") is not False
        or gate.get("any_tier_passed") is not False
        or gate.get("passing_tiers") != []
        or gate.get("decision") != "retain_production_dataset_and_v13_recipe"
        or report["recipe"].get("hardware") != {
            "engine_count": 4,
            "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "all_engines_every_signed_wave": True,
        }
    ):
        raise RuntimeError("v19a aggregate negative-gate contract changed")
    evidence = {
        "schema": "eggroll-es-disjoint-tier-attribution-negative-evidence-v19a",
        "status": "valid_completed_run_preregistered_gate_failed",
        "input_artifacts": {
            "launch_attempt": {
                "path": str(ATTEMPT_PATH_V19A),
                "file_sha256": ATTEMPT_FILE_SHA256_V19A,
                "content_sha256": ATTEMPT_CONTENT_SHA256_V19A,
            },
            "compact_report": {
                "path": str(REPORT_PATH_V19A),
                "file_sha256": REPORT_FILE_SHA256_V19A,
                "content_sha256": REPORT_CONTENT_SHA256_V19A,
            },
            "summary_content_sha256": SUMMARY_CONTENT_SHA256_V19A,
            "gate_content_sha256": GATE_CONTENT_SHA256_V19A,
        },
        "source_provenance": {
            "git_head": source["git_head"],
            "file_count": len(source["files"]),
            "content_sha256": SOURCE_PROVENANCE_CONTENT_SHA256_V19A,
            "implementation_bundle_sha256": IMPLEMENTATION_BUNDLE_SHA256_V19A,
            "all_files_match_launch_commit": True,
        },
        "bootstrap": {
            "seed": bootstrap["seed"],
            "repetitions": bootstrap["repetitions"],
            "one_sided_quantile": bootstrap["one_sided_quantile"],
            "familywise_alpha": 0.05,
            "tier_count": len(arms),
            "endpoint_count_per_tier": 12,
            "total_endpoint_count": 36,
            "paired_and_stratified": True,
            "whole_panel_block_resampling_used": False,
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
            "required_per_tier": 12,
            "all_rules_conjunctive": True,
            "any_tier_passed": False,
        },
        "runtime_integrity": {
            "checks": runtime_integrity,
            "signed_wave_count": runtime_audit["signed_wave_count"],
            "panel_count": runtime_audit["panel_count"],
            "requests_per_engine_per_signed_wave": runtime_audit[
                "requests_per_engine_per_signed_wave"
            ],
            "dense_result_commitment_count": runtime_audit[
                "dense_result_commitment_count"
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
            "separate_fresh_basis_train_only_confirmation_authorized": False,
            "dataset_promotion_authorized": False,
            "model_update_authorized": False,
            "evaluation_authorized": False,
        },
        "contains_response_vectors_or_row_content": False,
        "contains_validation_ood_heldout_or_benchmark_content": False,
        "interpretation": (
            "valid_train_only_run_but_each_disjoint_candidate_tier_failed_the_"
            "zero_margin_familywise_attribution_gate"
        ),
    }
    evidence["content_sha256_before_self_field"] = canonical_sha256(evidence)
    validate_evidence_v19a(evidence)
    return evidence


def validate_evidence_v19a(evidence):
    arms = evidence.get("arms", {}) if isinstance(evidence, dict) else {}
    arms_valid = (
        set(arms) == set(EXPECTED_OBSERVED_PASS_COUNTS_V19A)
        and all(
            set(value) == {
                "endpoints", "observed_pass_count", "bootstrap_pass_count",
                "preregistered_attribution_gate_passed",
            }
            and len(value["endpoints"]) == 12
            and value["observed_pass_count"]
            == EXPECTED_OBSERVED_PASS_COUNTS_V19A[arm]
            and value["bootstrap_pass_count"] == 0
            and value["preregistered_attribution_gate_passed"] is False
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
        != "eggroll-es-disjoint-tier-attribution-negative-evidence-v19a"
        or evidence.get("status")
        != "valid_completed_run_preregistered_gate_failed"
        or evidence.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(evidence))
        or not arms_valid
        or evidence.get("gate_summary") != {
            "observed_pass_counts": EXPECTED_OBSERVED_PASS_COUNTS_V19A,
            "bootstrap_pass_counts": {
                arm: 0 for arm in EXPECTED_OBSERVED_PASS_COUNTS_V19A
            },
            "required_per_tier": 12,
            "all_rules_conjunctive": True,
            "any_tier_passed": False,
        }
        or evidence.get("decision") != {
            "retain_dataset": "production",
            "retain_recipe": "v13_middle_late_layers_20_23",
            "separate_fresh_basis_train_only_confirmation_authorized": False,
            "dataset_promotion_authorized": False,
            "model_update_authorized": False,
            "evaluation_authorized": False,
        }
        or evidence.get("contains_response_vectors_or_row_content") is not False
        or evidence.get(
            "contains_validation_ood_heldout_or_benchmark_content"
        ) is not False
    ):
        raise RuntimeError("v19a compact negative evidence changed")
    prereg_v19a.prereg_v18a.prereg_v17a.validate_persisted_sha256_fields_v17a(
        evidence
    )
    return evidence


def _exclusive_write(path, value):
    path = Path(path).resolve()
    if path != OUTPUT_PATH_V19A:
        raise ValueError("v19a compact evidence output path changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("v19a compact evidence already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH_V19A))
    args = parser.parse_args(argv)
    if Path(args.output).resolve() != OUTPUT_PATH_V19A:
        raise ValueError("v19a compact evidence output path changed")
    evidence = build_evidence_v19a()
    _exclusive_write(args.output, evidence)
    result = {
        "schema": "eggroll-es-disjoint-tier-evidence-write-v19a",
        "path": str(OUTPUT_PATH_V19A),
        "file_sha256": file_sha256(OUTPUT_PATH_V19A),
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
