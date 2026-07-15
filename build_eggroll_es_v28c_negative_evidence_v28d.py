#!/usr/bin/env python3
"""Build compact immutable negative evidence for completed V28C."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ATTEMPT_RELATIVE_PATH_V28D = (
    "experiments/eggroll_es_hpo/runs/"
    ".s6_v28c_v27c_bf16_no_update_es_train_step_ab.launch_attempt.json"
)
REPORT_RELATIVE_PATH_V28D = (
    "experiments/eggroll_es_hpo/runs/"
    "s6_v28c_v27c_bf16_no_update_es_train_step_ab/"
    "v27c_bf16_train_step_ab_report_v28c.json"
)
OUTPUT_RELATIVE_PATH_V28D = (
    "experiments/eggroll_es_hpo/"
    "S6_V28D_V28C_TUNED_TRAIN_STEP_NEGATIVE_EVIDENCE.json"
)
ATTEMPT_PATH_V28D = ROOT / ATTEMPT_RELATIVE_PATH_V28D
REPORT_PATH_V28D = ROOT / REPORT_RELATIVE_PATH_V28D
OUTPUT_PATH_V28D = ROOT / OUTPUT_RELATIVE_PATH_V28D

ATTEMPT_FILE_SHA256_V28D = (
    "c9c70a8e9a0b1674145462a333006bdd2fca74a9f2326f65363df9a42135f44b"
)
ATTEMPT_CONTENT_SHA256_V28D = (
    "8de598e472557fc088c599c1915e3ed44f7a524f67e5605a482efffcf0f29262"
)
REPORT_FILE_SHA256_V28D = (
    "59cfb5fe4768202d23f4918f9327ba539975f246f217f7c03cac39bcf9f0b7ac"
)
REPORT_CONTENT_SHA256_V28D = (
    "066ddddc704e2fffea4354b26b875bbb893cd1ff2d27c0e5f1b79d6540c876b9"
)
GATE_CONTENT_SHA256_V28D = (
    "a7c1e170f53b7de405896c51989f973b7b63fd950da0a2d8ac01bb34fb22475d"
)
IMPLEMENTATION_BUNDLE_SHA256_V28D = (
    "6674125d772e1fff2d6e096d12e05d8d21ab1a1f139b51a88a32d770ae7f4251"
)
RECIPE_CONTENT_SHA256_V28D = (
    "d96dbae6fe5904d7d827b20403089484aa02785e6c24993cdb3eabf2c71707f1"
)
COMMITTED_CLEAN_SOURCE_CONTENT_SHA256_V28D = (
    "3eeeded854e5c87918aa5073e32db8c78c460ede5794fc0bb5a43dbc56ca43ed"
)
RUNTIME_ENVIRONMENT_CONTENT_SHA256_V28D = (
    "7acc87ffb6a50682f19309ea3059823d0fc96dfc459e30cb66d64e157a76f52a"
)
LIVE_CPU_DISK_AUDIT_CONTENT_SHA256_V28D = (
    "9fae01835f523bcccf2eafab1f03639c96c21882331bb9847f378382d6e8527c"
)
PRELAUNCH_IDLE_CONTENT_SHA256_V28D = (
    "ee23ef349d191926577082ba9c88bf23cfc33598794c91b38331c3cd8e7960ce"
)
FINAL_IDLE_CONTENT_SHA256_V28D = (
    "93e9514dfdf46c6030886f9fe6e46ef660bd5de48049b3755a2d21ce2c2ef54a"
)
SHARED_DIAGNOSTIC_COMMITMENT_SHA256_V28D = (
    "fedc217e313cd7b9ea982760983b836dadc8e729669fcfbb0db635ba9957c5dd"
)
GROUP_AUDIT_BUNDLE_SHA256_V28D = (
    "cd6af184a48bb6102e3ac0277c70374f93a5360e7fc84348c5773bc177f5afbc"
)
BOOTSTRAP_DRAW_PLAN_SHA256_V28D = (
    "bed6a26d78a34cb432a5b6926e8dc1208e3b065dfdaf7ee9916a0a508d378ed8"
)

EQUIVALENCE_COMPONENTS_V28D = {
    "all_five_panel_coefficient_arrays_exact",
    "all_response_and_analysis_payloads_exact",
    "full_diagnostic_commitment_exact",
    "full_diagnostic_exact",
    "identity_guard_exact",
    "panel_seed_and_common_random_number_contract_exact",
    "population_boundary_guard_exact",
    "robust_aggregate_coefficients_exact",
}
ATTEMPT_KEYS_V28D = {
    "checkpoint_written",
    "committed_clean_source_certificate_sha256",
    "content_sha256_before_self_field",
    "dataset_promotion_applied",
    "evaluation_opened",
    "final_idle_certificate_sha256",
    "implementation_bundle_sha256",
    "live_cpu_disk_audit_sha256",
    "model_update_applied",
    "phase",
    "prelaunch_idle_certificate_sha256",
    "recipe_sha256",
    "report_binding",
    "runtime_environment_certificate_sha256",
    "schema",
    "status",
}
REPORT_KEYS_V28D = {
    "checkpoint_written",
    "committed_clean_source_certificate_sha256",
    "content_sha256_before_self_field",
    "dataset_promotion_applied",
    "equivalence",
    "evaluation_opened",
    "final_idle_certificate_sha256",
    "gate",
    "implementation_bundle_sha256",
    "model_update_applied",
    "nontrain_surface_opened",
    "performance",
    "prelaunch_idle_certificate_sha256",
    "raw_rows_prompts_answers_tokens_coefficients_timings_memory_pids_or_draws_persisted",
    "recipe_sha256",
    "runtime_integrity",
    "schema",
    "status",
}
FORBIDDEN_PAYLOAD_KEYS_V28D = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "text", "texts", "token", "tokens", "stdout", "stderr", "log",
    "logs", "pid", "pids", "timing", "timings", "memory_samples",
    "pair_vectors", "bootstrap_replicates", "bootstrap_draws",
}


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _seal(value):
    result = copy.deepcopy(value)
    result.pop("content_sha256_before_self_field", None)
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _load_json_object(path, label):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"V28D {label} path changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"V28D {label} must be a JSON object")
    return value


def _verify_self(value, expected, label):
    _require(
        value.get("content_sha256_before_self_field") == expected
        and canonical_sha256(_without_self(value)) == expected,
        f"V28D {label} self hash changed",
    )


def _recursive_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).lower()
            yield from _recursive_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _recursive_keys(item)


def _assert_compact_v28d(value):
    overlap = FORBIDDEN_PAYLOAD_KEYS_V28D & set(_recursive_keys(value))
    _require(
        not overlap,
        f"V28D evidence contains forbidden detailed payload keys: {sorted(overlap)}",
    )


def _validate_attempt_v28d(attempt, report):
    _require(set(attempt) == ATTEMPT_KEYS_V28D, "V28D attempt schema keys changed")
    _verify_self(attempt, ATTEMPT_CONTENT_SHA256_V28D, "attempt")
    _require(
        attempt.get("schema") == "eggroll-es-v28c-durable-launch-attempt"
        and attempt.get("status") == "complete"
        and attempt.get("phase") == "after_24_group_cleanup_and_compact_report"
        and attempt.get("implementation_bundle_sha256")
        == IMPLEMENTATION_BUNDLE_SHA256_V28D
        and attempt.get("recipe_sha256") == RECIPE_CONTENT_SHA256_V28D
        and attempt.get("committed_clean_source_certificate_sha256")
        == COMMITTED_CLEAN_SOURCE_CONTENT_SHA256_V28D
        and attempt.get("runtime_environment_certificate_sha256")
        == RUNTIME_ENVIRONMENT_CONTENT_SHA256_V28D
        and attempt.get("live_cpu_disk_audit_sha256")
        == LIVE_CPU_DISK_AUDIT_CONTENT_SHA256_V28D
        and attempt.get("prelaunch_idle_certificate_sha256")
        == PRELAUNCH_IDLE_CONTENT_SHA256_V28D
        and attempt.get("final_idle_certificate_sha256")
        == FINAL_IDLE_CONTENT_SHA256_V28D,
        "V28D attempt source, recipe, environment, or lifecycle binding changed",
    )
    binding = attempt.get("report_binding")
    _require(
        binding == {
            "path": str(REPORT_PATH_V28D.resolve()),
            "file_sha256": REPORT_FILE_SHA256_V28D,
            "content_sha256": REPORT_CONTENT_SHA256_V28D,
        },
        "V28D attempt report binding changed",
    )
    _require(
        report.get("implementation_bundle_sha256")
        == attempt["implementation_bundle_sha256"]
        and report.get("recipe_sha256") == attempt["recipe_sha256"]
        and report.get("committed_clean_source_certificate_sha256")
        == attempt["committed_clean_source_certificate_sha256"]
        and report.get("prelaunch_idle_certificate_sha256")
        == attempt["prelaunch_idle_certificate_sha256"]
        and report.get("final_idle_certificate_sha256")
        == attempt["final_idle_certificate_sha256"],
        "V28D attempt and report bindings disagree",
    )


def _validate_equivalence_v28d(report):
    equivalence = report.get("equivalence", {})
    components = equivalence.get("component_pass_counts", {})
    _require(
        set(equivalence) == {
            "schema", "pair_count", "exact_pair_count", "all_twelve_pairs_exact",
            "component_pass_counts", "shared_diagnostic_commitment_sha256",
        }
        and equivalence.get("schema")
        == "eggroll-es-v28c-compact-exact-equivalence"
        and equivalence.get("pair_count") == 12
        and equivalence.get("exact_pair_count") == 12
        and equivalence.get("all_twelve_pairs_exact") is True
        and set(components) == EQUIVALENCE_COMPONENTS_V28D
        and all(value == 12 for value in components.values())
        and equivalence.get("shared_diagnostic_commitment_sha256")
        == SHARED_DIAGNOSTIC_COMMITMENT_SHA256_V28D,
        "V28D exact 12-pair equivalence aggregate changed",
    )
    return equivalence


def _validate_runtime_integrity_v28d(report):
    integrity = report.get("runtime_integrity", {})
    _require(
        set(integrity) == {
            "schema",
            "all_24_fresh_groups_and_cleanup_gates_passed",
            "all_24_groups_observed_all_four_active",
            "all_four_activity_group_count",
            "all_integrity_gates_passed",
            "final_all_four_idle",
            "final_idle_certificate_sha256",
            "fresh_engine_group_count",
            "group_audit_bundle_sha256",
            "minimum_activity_sample_count",
            "minimum_qualifying_activity_sample_count",
            "physical_gpu_identity_preserved_across_all_groups",
            "raw_pids_timings_memory_samples_or_diagnostics_persisted",
        }
        and integrity.get("schema") == "eggroll-es-v28c-compact-runtime-integrity"
        and integrity.get("fresh_engine_group_count") == 24
        and integrity.get("all_four_activity_group_count") == 24
        and integrity.get("all_24_fresh_groups_and_cleanup_gates_passed") is True
        and integrity.get("all_24_groups_observed_all_four_active") is True
        and integrity.get("all_integrity_gates_passed") is True
        and integrity.get("final_all_four_idle") is True
        and integrity.get("physical_gpu_identity_preserved_across_all_groups")
        is True
        and integrity.get("raw_pids_timings_memory_samples_or_diagnostics_persisted")
        is False
        and integrity.get("minimum_activity_sample_count") == 3269
        and integrity.get("minimum_qualifying_activity_sample_count") == 2947
        and integrity.get("group_audit_bundle_sha256")
        == GROUP_AUDIT_BUNDLE_SHA256_V28D
        and integrity.get("final_idle_certificate_sha256")
        == FINAL_IDLE_CONTENT_SHA256_V28D,
        "V28D 24-group activity or cleanup integrity changed",
    )
    return integrity


def _validate_performance_v28d(report):
    performance = report.get("performance", {})
    endpoints = performance.get("endpoints", {})
    _require(
        set(performance) == {
            "schema",
            "absolute_peak_nvml_fraction_max_observed",
            "absolute_peak_nvml_gate_passed",
            "all_three_inferential_endpoints_passed",
            "bootstrap_draw_plan_sha256",
            "descriptive_order_stratum_speed_medians",
            "descriptive_subphase_median_default_over_tuned_ratios",
            "endpoints",
            "pair_count",
            "raw_pair_vectors_or_bootstrap_replicates_persisted",
        }
        and performance.get("schema")
        == "eggroll-es-v28c-compact-paired-performance-summary"
        and performance.get("pair_count") == 12
        and performance.get("bootstrap_draw_plan_sha256")
        == BOOTSTRAP_DRAW_PLAN_SHA256_V28D
        and performance.get("raw_pair_vectors_or_bootstrap_replicates_persisted")
        is False
        and performance.get("all_three_inferential_endpoints_passed") is False
        and set(endpoints) == {
            "complete_train_step_speed",
            "peak_torch_allocated",
            "peak_torch_reserved",
        },
        "V28D compact performance aggregate changed",
    )
    speed = endpoints["complete_train_step_speed"]
    expected_speed = {
        "familywise_lower_confidence_bound": 0.9945921239188809,
        "lower_bound_strict_threshold": 1.0,
        "median_default_over_tuned_ratio": 1.003176690951185,
        "pass": False,
        "point_threshold": 1.01,
    }
    _require(
        speed == expected_speed
        and speed["median_default_over_tuned_ratio"] < speed["point_threshold"]
        and speed["familywise_lower_confidence_bound"]
        <= speed["lower_bound_strict_threshold"],
        "V28D complete train-step sole performance failure changed",
    )
    expected_memory = {
        "familywise_upper_confidence_bound": 1.0,
        "median_tuned_over_default_ratio": 1.0,
        "pass": True,
        "point_threshold": 1.01,
        "upper_bound_threshold": 1.02,
    }
    _require(
        endpoints["peak_torch_allocated"] == expected_memory
        and endpoints["peak_torch_reserved"] == expected_memory
        and all(item.get("pass") is True for name, item in endpoints.items() if name != "complete_train_step_speed")
        and sum(item.get("pass") is False for item in endpoints.values()) == 1,
        "V28D memory endpoints or sole-failure count changed",
    )
    peak = performance.get("absolute_peak_nvml_fraction_max_observed")
    _require(
        math.isclose(peak, 0.8575398163188166, rel_tol=0.0, abs_tol=0.0)
        and peak <= 0.95
        and performance.get("absolute_peak_nvml_gate_passed") is True,
        "V28D absolute VRAM gate changed",
    )
    _require(
        performance.get("descriptive_order_stratum_speed_medians") == {
            "default_first": 1.003176690951185,
            "tuned_first": 1.0021323650492184,
        }
        and performance.get("descriptive_subphase_median_default_over_tuned_ratios")
        == {
            "configure_train_panels_v13": 0.9988170493574815,
            "estimate_train_panels_v13": 1.0045981914476685,
        },
        "V28D compact descriptive performance aggregate changed",
    )
    return performance


def _validate_closed_authority_v28d(attempt, report):
    gate = report.get("gate", {})
    _verify_self(gate, GATE_CONTENT_SHA256_V28D, "authorization gate")
    _require(
        set(gate) == {
            "schema",
            "absolute_peak_nvml_gate_passed",
            "all_runtime_integrity_gates_passed",
            "all_three_performance_endpoints_passed",
            "all_twelve_pairs_exact",
            "checkpoint_write_authorized",
            "content_sha256_before_self_field",
            "dataset_promotion_authorized",
            "decision",
            "direct_recipe_adoption_authorized",
            "evaluation_authorized",
            "fp8_reuse_authorized",
            "model_update_authorized",
            "nontrain_reuse_authorized",
            "pass",
        }
        and gate.get("schema") == "eggroll-es-v28c-authorization-gate"
        and gate.get("pass") is False
        and gate.get("decision") == "retain_empty_default_bf16_training_recipe"
        and gate.get("all_twelve_pairs_exact") is True
        and gate.get("all_runtime_integrity_gates_passed") is True
        and gate.get("absolute_peak_nvml_gate_passed") is True
        and gate.get("all_three_performance_endpoints_passed") is False
        and all(gate.get(key) is False for key in (
            "checkpoint_write_authorized",
            "dataset_promotion_authorized",
            "direct_recipe_adoption_authorized",
            "evaluation_authorized",
            "fp8_reuse_authorized",
            "model_update_authorized",
            "nontrain_reuse_authorized",
        )),
        "V28D closed authorization gate changed",
    )
    _require(
        all(attempt.get(key) is False for key in (
            "checkpoint_written", "dataset_promotion_applied",
            "evaluation_opened", "model_update_applied",
        ))
        and all(report.get(key) is False for key in (
            "checkpoint_written", "dataset_promotion_applied",
            "evaluation_opened", "model_update_applied", "nontrain_surface_opened",
            "raw_rows_prompts_answers_tokens_coefficients_timings_memory_pids_or_draws_persisted",
        )),
        "V28D forbidden side effect or detailed persistence changed",
    )
    return gate


def validate_bound_artifacts_v28d():
    _require(
        file_sha256(ATTEMPT_PATH_V28D) == ATTEMPT_FILE_SHA256_V28D,
        "V28D attempt file hash changed",
    )
    _require(
        file_sha256(REPORT_PATH_V28D) == REPORT_FILE_SHA256_V28D,
        "V28D report file hash changed",
    )
    attempt = _load_json_object(ATTEMPT_PATH_V28D, "attempt")
    report = _load_json_object(REPORT_PATH_V28D, "report")
    _require(set(report) == REPORT_KEYS_V28D, "V28D report schema keys changed")
    _verify_self(report, REPORT_CONTENT_SHA256_V28D, "report")
    _require(
        report.get("schema") == "eggroll-es-v27c-bf16-train-step-ab-report-v28c"
        and report.get("status") == "valid_completed_train_only_no_update_runtime_ab",
        "V28D report schema or completion status changed",
    )
    _validate_attempt_v28d(attempt, report)
    _validate_equivalence_v28d(report)
    _validate_runtime_integrity_v28d(report)
    _validate_performance_v28d(report)
    _validate_closed_authority_v28d(attempt, report)
    return attempt, report


def build_negative_evidence_v28d():
    attempt, report = validate_bound_artifacts_v28d()
    equivalence = report["equivalence"]
    integrity = report["runtime_integrity"]
    performance = report["performance"]
    speed = performance["endpoints"]["complete_train_step_speed"]
    memory = performance["endpoints"]["peak_torch_allocated"]
    value = {
        "schema": "eggroll-es-v28c-negative-evidence-v28d",
        "status": "sealed_completed_compact_negative_evidence",
        "artifacts": {
            "durable_attempt": {
                "relative_path": ATTEMPT_RELATIVE_PATH_V28D,
                "file_sha256": ATTEMPT_FILE_SHA256_V28D,
                "content_sha256": ATTEMPT_CONTENT_SHA256_V28D,
            },
            "compact_report": {
                "relative_path": REPORT_RELATIVE_PATH_V28D,
                "file_sha256": REPORT_FILE_SHA256_V28D,
                "content_sha256": REPORT_CONTENT_SHA256_V28D,
            },
        },
        "frozen_bindings": {
            "implementation_bundle_sha256": attempt["implementation_bundle_sha256"],
            "recipe_content_sha256": attempt["recipe_sha256"],
            "committed_clean_source_content_sha256": attempt[
                "committed_clean_source_certificate_sha256"
            ],
            "runtime_environment_content_sha256": attempt[
                "runtime_environment_certificate_sha256"
            ],
            "live_cpu_disk_audit_content_sha256": attempt[
                "live_cpu_disk_audit_sha256"
            ],
            "prelaunch_idle_content_sha256": attempt[
                "prelaunch_idle_certificate_sha256"
            ],
            "final_idle_content_sha256": attempt[
                "final_idle_certificate_sha256"
            ],
            "authorization_gate_content_sha256": GATE_CONTENT_SHA256_V28D,
        },
        "aggregate_execution": {
            "equivalence_pair_count": equivalence["pair_count"],
            "exact_equivalence_pair_count": equivalence["exact_pair_count"],
            "equivalence_component_count": len(equivalence["component_pass_counts"]),
            "all_equivalence_components_exact_for_all_pairs": True,
            "fresh_engine_group_count": integrity["fresh_engine_group_count"],
            "all_four_activity_group_count": integrity[
                "all_four_activity_group_count"
            ],
            "all_24_fresh_group_cleanup_gates_passed": True,
            "all_24_groups_observed_all_four_active": True,
            "final_all_four_idle": True,
            "physical_gpu_identity_preserved": True,
            "minimum_activity_sample_count": integrity[
                "minimum_activity_sample_count"
            ],
            "minimum_qualifying_activity_sample_count": integrity[
                "minimum_qualifying_activity_sample_count"
            ],
            "group_audit_bundle_sha256": integrity["group_audit_bundle_sha256"],
        },
        "aggregate_performance": {
            "inferential_endpoint_count": 3,
            "passing_endpoint_count": 2,
            "failing_endpoint_count": 1,
            "sole_failure": "complete_train_step_speed",
            "complete_train_step": {
                "median_default_over_tuned_ratio": speed[
                    "median_default_over_tuned_ratio"
                ],
                "point_threshold": speed["point_threshold"],
                "point_gate_passed": False,
                "familywise_lower_confidence_bound": speed[
                    "familywise_lower_confidence_bound"
                ],
                "lower_bound_strict_threshold": speed[
                    "lower_bound_strict_threshold"
                ],
                "lower_bound_gate_passed": False,
                "endpoint_passed": False,
            },
            "peak_torch_allocated": copy.deepcopy(memory),
            "peak_torch_reserved": copy.deepcopy(
                performance["endpoints"]["peak_torch_reserved"]
            ),
            "absolute_peak_nvml_fraction_observed": performance[
                "absolute_peak_nvml_fraction_max_observed"
            ],
            "absolute_peak_nvml_fraction_limit": 0.95,
            "absolute_peak_nvml_gate_passed": True,
            "memory_ratio_and_familywise_upper_bound_gates_passed": True,
            "bootstrap_draw_plan_sha256": performance[
                "bootstrap_draw_plan_sha256"
            ],
        },
        "decision": {
            "retain_empty_default_bf16_training_recipe": True,
            "v27c_tuned_recipe_direct_adoption_authority": False,
            "model_update_authority": False,
            "checkpoint_write_authority": False,
            "evaluation_authority": False,
            "dataset_authority": False,
            "fp8_authority": False,
            "nontrain_authority": False,
        },
        "side_effects": {
            "model_update_applied": False,
            "checkpoint_written": False,
            "evaluation_opened": False,
            "dataset_promotion_applied": False,
            "nontrain_surface_opened": False,
        },
        "input_scope": {
            "compact_attempt_and_report_aggregates_only": True,
            "raw_stdout_or_runtime_logs_opened": False,
            "dataset_or_semantic_content_opened": False,
            "detailed_runtime_payloads_persisted": False,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    _assert_compact_v28d(value)
    return value


def _exclusive_write_json_v28d(path, value):
    path = Path(path).resolve()
    if path != OUTPUT_PATH_V28D.resolve():
        raise ValueError("V28D evidence output path changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise RuntimeError("V28D evidence already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    value = build_negative_evidence_v28d()
    if not args.dry_run:
        _exclusive_write_json_v28d(OUTPUT_PATH_V28D, value)
    print(json.dumps({
        "schema": "eggroll-es-v28c-negative-evidence-build-v28d",
        "content_sha256": value["content_sha256_before_self_field"],
        "gate_pass": False,
        "retain_empty_default_bf16_training_recipe": True,
        "direct_adoption_authority": False,
        "gpu_launched": False,
        "dataset_or_semantic_content_opened": False,
    }, indent=2, sort_keys=True))
    return value


if __name__ == "__main__":
    main()
