#!/usr/bin/env python3
"""Build aggregate-only evidence for the negative V20A nested-tier run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import eggroll_es_nested_tier_interaction_preregistration_v20a as prereg_v20a
import run_eggroll_es_raw_attribution_v20a as driver_v20a
import train_eggroll_es_nested_tier_interaction_v20a as mechanics_v20a


ROOT = Path(__file__).resolve().parent
LAUNCH_ID_V20A = "1784093226247966617-241051-deb4e94271f6cd46"
RUN_ROOT_V20A = (
    driver_v20a.FROZEN_OUTPUT_DIRECTORY_V20A
    / driver_v20a.EXPERIMENT_NAME_V20A
).resolve()
ATTEMPT_PATH_V20A = (
    RUN_ROOT_V20A / "attempts" / f"{LAUNCH_ID_V20A}.json"
).resolve()
REPORT_PATH_V20A = (
    RUN_ROOT_V20A / "runs" / LAUNCH_ID_V20A / driver_v20a.REPORT_NAME_V20A
).resolve()
OUTPUT_PATH_V20A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V20A_NESTED_TIER_ATTRIBUTION_NEGATIVE_EVIDENCE.json"
).resolve()

ATTEMPT_FILE_SHA256_V20A = (
    "a498e028f3e54b5b088b0a95ec90b3279586d8c13dcb4f925646cc73058fe593"
)
ATTEMPT_CONTENT_SHA256_V20A = (
    "226a63e31515244577e3c9cfcbd2583990dc66098b067794a2a7d212c8b86b74"
)
REPORT_FILE_SHA256_V20A = (
    "0b8a88e7304ac95dbc4a8173bed2bccb8857c663b49c528743af7455e16b2fe1"
)
REPORT_CONTENT_SHA256_V20A = (
    "ff430ea2bcba11ce7bdae8e2498d7d4fc79bc013adfb007e74dd3a26437d1815"
)
SUMMARY_CONTENT_SHA256_V20A = (
    "352d981a342af831e1eb6aa4d8014613481f77bb60335eb31c86c6ed3d804fcd"
)
GATE_CONTENT_SHA256_V20A = (
    "ff9a95ad9eac858337c898b82494bd19f58795e5c7a717b19de597b0a1b0ce9b"
)
RUNTIME_AUDIT_CONTENT_SHA256_V20A = (
    "637661e2a9529cf177f1ead4d5464201255abce69af3d85a8210e7f8784bcbfe"
)
SOURCE_PROVENANCE_CONTENT_SHA256_V20A = (
    "b54a2073ded1c4f48d57d8fc89e64ce2f296a96583f0bbdbb16a7d01b17c5003"
)
IMPLEMENTATION_BUNDLE_SHA256_V20A = (
    "2054eea64163b3257dbd4fac71b93ee80a2126e48ec6f865e463bea2ec2a055e"
)
RECIPE_CONTENT_SHA256_V20A = (
    "49b363115e5a9a26c5a370aa4816e17cb474931974e1aa6b66977122c9648454"
)
EXPECTED_OBSERVED_PASS_COUNTS_V20A = {
    "tier2_vs_production": 6,
    "tiers2_3_vs_production": 6,
    "all_tiers_vs_production": 11,
    "conditional_tier3_after_tier2": 6,
    "conditional_tier1_after_tiers2_3": 11,
}

canonical_sha256 = prereg_v20a.canonical_sha256
file_sha256 = prereg_v20a.file_sha256


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _validate_source_provenance(attempt, report):
    source = attempt["source_provenance"]
    implementation = report["implementation"]
    if (
        source.get("schema") != "eggroll-es-raw-attribution-source-bundle-v20a"
        or source.get("content_sha256_before_self_field")
        != SOURCE_PROVENANCE_CONTENT_SHA256_V20A
        or source.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(source))
        or implementation.get("bundle_sha256")
        != IMPLEMENTATION_BUNDLE_SHA256_V20A
        or implementation.get("bundle_sha256")
        != canonical_sha256(implementation.get("files"))
        or source.get("implementation_bundle_sha256")
        != implementation.get("bundle_sha256")
        or set(source.get("files", {})) != set(implementation.get("files", {}))
        or len(source.get("files", {})) != 7
    ):
        raise RuntimeError("v20a compact source provenance changed")
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
                "v20a launch source differs from committed provenance"
            )
    return source


def load_compact_inputs_v20a():
    """Load only the durable attempt and aggregate-only compact report."""
    if (
        file_sha256(ATTEMPT_PATH_V20A) != ATTEMPT_FILE_SHA256_V20A
        or file_sha256(REPORT_PATH_V20A) != REPORT_FILE_SHA256_V20A
    ):
        raise RuntimeError("v20a compact attempt or report file changed")
    attempt = json.loads(ATTEMPT_PATH_V20A.read_text(encoding="utf-8"))
    report = json.loads(REPORT_PATH_V20A.read_text(encoding="utf-8"))
    if (
        attempt.get("content_sha256_before_self_field")
        != ATTEMPT_CONTENT_SHA256_V20A
        or attempt.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(attempt))
        or report.get("content_sha256_before_self_field")
        != REPORT_CONTENT_SHA256_V20A
        or report.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(report))
        or report.get("summary", {}).get("content_sha256_before_self_field")
        != SUMMARY_CONTENT_SHA256_V20A
        or canonical_sha256(report.get("gate", {}))
        != GATE_CONTENT_SHA256_V20A
        or report.get("runtime_audit", {}).get(
            "content_sha256_before_self_field"
        ) != RUNTIME_AUDIT_CONTENT_SHA256_V20A
        or report.get("recipe", {}).get("content_sha256_before_self_field")
        != RECIPE_CONTENT_SHA256_V20A
    ):
        raise RuntimeError("v20a compact content binding changed")
    driver_v20a.validate_compact_report_v20a(
        report,
        expected_recipe=report["recipe"],
        expected_implementation=report["implementation"],
    )
    if mechanics_v20a.evaluate_attribution_gate_v20a(
        report["summary"]
    ) != report["gate"]:
        raise RuntimeError("v20a hardened gate does not recompute")
    expected_attempt_keys = {
        "schema", "status", "phase", "experiment_name", "launch_id",
        "run_directory", "recipe", "source_provenance",
        "model_update_applied", "checkpoint_written",
        "evaluation_surfaces_opened", "dataset_promotion_applied",
        "union_scoring_authorized_or_used",
        "content_sha256_before_self_field", "report_exists_after_attempt",
        "report_binding",
    }
    if (
        set(attempt) != expected_attempt_keys
        or attempt.get("schema") != "eggroll-es-raw-attribution-attempt-v20a"
        or attempt.get("status") != "complete"
        or attempt.get("phase") != "after_cleanup_and_compact_report"
        or attempt.get("launch_id") != LAUNCH_ID_V20A
        or attempt.get("recipe") != report.get("recipe")
        or attempt.get("model_update_applied") is not False
        or attempt.get("checkpoint_written") is not False
        or attempt.get("evaluation_surfaces_opened") is not False
        or attempt.get("dataset_promotion_applied") is not False
        or attempt.get("union_scoring_authorized_or_used") is not False
        or attempt.get("report_exists_after_attempt") is not True
        or Path(attempt.get("run_directory")).resolve() != REPORT_PATH_V20A.parent
        or attempt.get("report_binding") != {
            "path": str(REPORT_PATH_V20A),
            "file_sha256": REPORT_FILE_SHA256_V20A,
            "content_sha256": REPORT_CONTENT_SHA256_V20A,
        }
    ):
        raise RuntimeError("v20a completed launch-attempt contract changed")
    source = _validate_source_provenance(attempt, report)
    return attempt, report, source


def build_evidence_v20a():
    _, report, source = load_compact_inputs_v20a()
    summary = report["summary"]
    gate = report["gate"]
    bootstrap = summary["paired_bootstrap"]
    comparisons = {}
    for contrast, specification in prereg_v20a.CONTRASTS_V20A.items():
        treatment = specification["treatment"]
        control = specification["control"]
        treatment_values = summary["arms"][treatment]["endpoint_values"]
        control_values = summary["arms"][control]["endpoint_values"]
        endpoints = {}
        for name in prereg_v20a.ENDPOINT_NAMES_V20A:
            item = bootstrap["comparisons"][contrast][name]
            delta = float(item["treatment_minus_control"])
            lcb = float(item["familywise_lcb"])
            expected_delta = float(treatment_values[name]) - float(
                control_values[name]
            )
            if (
                not all(math.isfinite(value) for value in (delta, lcb))
                or not math.isclose(
                    expected_delta, delta, rel_tol=1e-12, abs_tol=1e-12,
                )
                or item["noninferiority_margin"] != 0.0
            ):
                raise RuntimeError(f"v20a endpoint {contrast}/{name} changed")
            endpoints[name] = {
                "treatment_minus_control": delta,
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
        gate_contrast = gate["contrasts"][contrast]
        if (
            len(endpoints) != 12
            or observed != EXPECTED_OBSERVED_PASS_COUNTS_V20A[contrast]
            or bootstrap_pass != 0
            or gate_contrast["observed_pass_count"] != observed
            or gate_contrast["bootstrap_pass_count"] != bootstrap_pass
            or gate_contrast["preregistered_contrast_gate_passed"] is not False
        ):
            raise RuntimeError(f"v20a negative contrast gate changed for {contrast}")
        comparisons[contrast] = {
            "treatment": treatment,
            "control": control,
            "purpose": specification["purpose"],
            "endpoints": endpoints,
            "observed_pass_count": observed,
            "bootstrap_pass_count": bootstrap_pass,
            "preregistered_contrast_gate_passed": False,
        }
    runtime_integrity = summary["runtime_integrity"]
    runtime_audit = report["runtime_audit"]
    configuration = report["configuration"]
    expected_checks = {
        "all_four_tp1_engines_every_signed_wave",
        "all_integrity_audits_passed",
        "all_sixteen_signed_waves_complete",
        "all_ten_panels_every_direction_sign_and_arm",
        "exact_reference_restored_once_per_signed_wave",
        "latin_arm_order_complete",
        "population_boundary_audit_passed",
        "pre_post_raw_reference_probes_equal",
        "union_scoring_called_or_used",
        "unselected_origin_audit_passed",
    }
    if (
        bootstrap.get("seed") != prereg_v20a.BOOTSTRAP_SEED_V20A
        or bootstrap.get("repetitions") != 50_000
        or bootstrap.get("one_sided_quantile") != 0.05 / 60
        or bootstrap.get("whole_panel_block_resampling_used") is not False
        or set(runtime_integrity) != expected_checks
        or any(
            value is not True
            for key, value in runtime_integrity.items()
            if key != "union_scoring_called_or_used"
        )
        or runtime_integrity["union_scoring_called_or_used"] is not False
        or runtime_audit.get("signed_wave_count") != 16
        or runtime_audit.get("panel_count") != 10
        or runtime_audit.get("requests_per_engine_per_signed_wave") != 1020
        or runtime_audit.get("dense_result_commitment_count") != 2560
        or runtime_audit.get("per_unit_scores_persisted") is not False
        or runtime_audit.get("bootstrap_replicates_persisted") is not False
        or runtime_audit.get("bootstrap_draws_persisted") is not False
        or runtime_audit.get("row_content_persisted") is not False
        or runtime_audit.get("union_scoring_called_or_used") is not False
        or gate.get("all_five_contrasts_passed") is not False
        or gate.get("decision") != "retain_production_dataset_and_v13_recipe"
        or any(gate.get(key) is not False for key in (
            "model_update_authorized", "evaluation_authorized",
            "dataset_promotion_authorized",
        ))
        or configuration.get("engine_count") != 4
        or configuration.get("tp_per_engine") != 1
        or configuration.get("gpu_ids") != [0, 1, 2, 3]
        or configuration.get("union_planner_called") is not False
        or configuration.get("union_scoring_authorized") is not False
    ):
        raise RuntimeError("v20a aggregate negative-gate contract changed")
    evidence = {
        "schema": "eggroll-es-nested-tier-attribution-negative-evidence-v20a",
        "status": "valid_completed_raw_run_preregistered_gate_failed",
        "input_artifacts": {
            "launch_attempt": {
                "path": str(ATTEMPT_PATH_V20A),
                "file_sha256": ATTEMPT_FILE_SHA256_V20A,
                "content_sha256": ATTEMPT_CONTENT_SHA256_V20A,
            },
            "compact_report": {
                "path": str(REPORT_PATH_V20A),
                "file_sha256": REPORT_FILE_SHA256_V20A,
                "content_sha256": REPORT_CONTENT_SHA256_V20A,
            },
            "summary_content_sha256": SUMMARY_CONTENT_SHA256_V20A,
            "gate_content_sha256": GATE_CONTENT_SHA256_V20A,
            "runtime_audit_content_sha256": RUNTIME_AUDIT_CONTENT_SHA256_V20A,
        },
        "source_provenance": {
            "git_head": source["git_head"],
            "file_count": len(source["files"]),
            "content_sha256": SOURCE_PROVENANCE_CONTENT_SHA256_V20A,
            "implementation_bundle_sha256": IMPLEMENTATION_BUNDLE_SHA256_V20A,
            "all_files_match_launch_commit": True,
        },
        "bootstrap": {
            "seed": bootstrap["seed"],
            "repetitions": bootstrap["repetitions"],
            "one_sided_quantile": bootstrap["one_sided_quantile"],
            "familywise_alpha": 0.05,
            "contrast_count": 5,
            "endpoint_count_per_contrast": 12,
            "total_endpoint_count": 60,
            "paired_and_stratified": True,
            "whole_panel_block_resampling_used": False,
            "replicates_persisted": False,
        },
        "contrasts": comparisons,
        "gate_summary": {
            "observed_pass_counts": {
                name: value["observed_pass_count"]
                for name, value in comparisons.items()
            },
            "bootstrap_pass_counts": {
                name: value["bootstrap_pass_count"]
                for name, value in comparisons.items()
            },
            "required_per_contrast": 12,
            "all_rules_conjunctive": True,
            "all_five_contrasts_passed": False,
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
            "unselected_origin_audit_sha256": runtime_audit[
                "unselected_origin_audit_sha256"
            ],
            "token_boundary_audit_sha256": runtime_audit[
                "token_boundary_audit_sha256"
            ],
            "all_four_gpus_observed_every_signed_wave": True,
            "exact_restoration_and_boundary_audits_passed": True,
            "per_unit_scores_persisted": False,
            "union_scoring_authorized_or_used": False,
        },
        "backend": report["recipe"]["moe_backend"],
        "hardware": report["recipe"]["hardware"],
        "decision": {
            "retain_dataset": "production",
            "retain_recipe": "v13_middle_late_layers_20_23",
            "separate_train_only_confirmation_authorized": False,
            "dataset_promotion_authorized": False,
            "model_update_authorized": False,
            "evaluation_authorized": False,
        },
        "contains_response_vectors_or_row_content": False,
        "contains_validation_ood_heldout_or_benchmark_content": False,
        "interpretation": (
            "valid_train_only_raw_run_but_all_five_nested_tier_contrasts_"
            "failed_the_zero_margin_familywise_attribution_gate"
        ),
    }
    evidence["content_sha256_before_self_field"] = canonical_sha256(evidence)
    validate_evidence_v20a(evidence)
    return evidence


def validate_evidence_v20a(evidence):
    contrasts = evidence.get("contrasts", {}) if isinstance(evidence, dict) else {}
    contrasts_valid = (
        set(contrasts) == set(EXPECTED_OBSERVED_PASS_COUNTS_V20A)
        and all(
            value["observed_pass_count"]
            == EXPECTED_OBSERVED_PASS_COUNTS_V20A[name]
            and value["bootstrap_pass_count"] == 0
            and value["preregistered_contrast_gate_passed"] is False
            and len(value["endpoints"]) == 12
            and all(
                item["observed_noninferiority_pass"]
                is (float(item["treatment_minus_control"]) >= 0.0)
                and item["bootstrap_noninferiority_pass"]
                is (float(item["familywise_lcb"]) >= 0.0)
                for item in value["endpoints"].values()
            )
            for name, value in contrasts.items()
        )
    )
    if (
        not isinstance(evidence, dict)
        or evidence.get("schema")
        != "eggroll-es-nested-tier-attribution-negative-evidence-v20a"
        or evidence.get("status")
        != "valid_completed_raw_run_preregistered_gate_failed"
        or evidence.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(evidence))
        or not contrasts_valid
        or evidence.get("gate_summary") != {
            "observed_pass_counts": EXPECTED_OBSERVED_PASS_COUNTS_V20A,
            "bootstrap_pass_counts": {
                name: 0 for name in EXPECTED_OBSERVED_PASS_COUNTS_V20A
            },
            "required_per_contrast": 12,
            "all_rules_conjunctive": True,
            "all_five_contrasts_passed": False,
        }
        or evidence.get("decision") != {
            "retain_dataset": "production",
            "retain_recipe": "v13_middle_late_layers_20_23",
            "separate_train_only_confirmation_authorized": False,
            "dataset_promotion_authorized": False,
            "model_update_authorized": False,
            "evaluation_authorized": False,
        }
        or evidence.get("contains_response_vectors_or_row_content") is not False
        or evidence.get(
            "contains_validation_ood_heldout_or_benchmark_content"
        ) is not False
    ):
        raise RuntimeError("v20a compact negative evidence changed")
    prereg_v20a.prereg_v19a.prereg_v18a.prereg_v17a.validate_persisted_sha256_fields_v17a(
        evidence
    )
    return evidence


def _exclusive_write(path, value):
    path = Path(path).resolve()
    if path != OUTPUT_PATH_V20A:
        raise ValueError("v20a compact evidence output path changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("v20a compact evidence already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH_V20A))
    args = parser.parse_args(argv)
    if Path(args.output).resolve() != OUTPUT_PATH_V20A:
        raise ValueError("v20a compact evidence output path changed")
    evidence = build_evidence_v20a()
    _exclusive_write(args.output, evidence)
    result = {
        "schema": "eggroll-es-nested-tier-evidence-write-v20a",
        "path": str(OUTPUT_PATH_V20A),
        "file_sha256": file_sha256(OUTPUT_PATH_V20A),
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
