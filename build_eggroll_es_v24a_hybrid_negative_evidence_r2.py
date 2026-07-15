#!/usr/bin/env python3
"""Bind V24A R1's failed train-only hybrid compatibility gate."""

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
    ".s6_v24a_hybrid_backend_train_only_compatibility_memory_retry_r1."
    "launch_attempt.json"
)
REPORT_PATH = RUNS / (
    "s6_v24a_hybrid_backend_train_only_compatibility_memory_retry_r1/"
    "hybrid_backend_compatibility_v24a_memory_retry_r1.json"
)
OUTPUT_PATH = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V24A_HYBRID_NEGATIVE_AGGREGATE_EVIDENCE_R2.json"
)

ATTEMPT_FILE_SHA256 = "31116ebde5d5722e1057f01206809069bf070304ec9ea98e5118bb5f966d9b56"
ATTEMPT_CONTENT_SHA256 = "8c50bc4c6e50b0012b1e56aab67094aa76bd9ea1c32faf617dfd3d20f51e0b05"
REPORT_FILE_SHA256 = "19fb30df59420517b6b1de0f6009992db0a58991bee23a784cad0d5c212a40be"
REPORT_CONTENT_SHA256 = "bb70e98062433be64eebcf6facf7dd572e2b05c9e970c94db9aea8a36f785a3d"
SOURCE_COMMIT = "d4d8d8978514784ba4b157811860521f91c9c234"
SOURCE_CONTENT_SHA256 = "04bef1d455351ed42c2481f75cdae6d3972bba2db79d53b247c0d6cfd5a91adf"
IMPLEMENTATION_BUNDLE_SHA256 = (
    "6e996e4a71f0acf160b7ca27ad19dc2e09a30863676ee21ea01796d5682d8d6d"
)
OVERLAY_BUNDLE_SHA256 = (
    "8bd35ea23d5f544a9e4830e9a23138e7b946db90ae1cd121d0fe582b3ed2ed27"
)
RECIPE_CONTENT_SHA256 = "759959c910fc0c0654dd7b4ba64d68e2cafe6afbda204c89c6dabd64292a1283"
RUNTIME_ENVIRONMENT_CONTENT_SHA256 = (
    "7acc87ffb6a50682f19309ea3059823d0fc96dfc459e30cb66d64e157a76f52a"
)
CONFIGURATION_CONTENT_SHA256 = (
    "4dbf9d958fbb9e183b25e21379284d3bc951b1f13ac2846f21312bec7ee31ce4"
)
RUNTIME_AUDIT_CONTENT_SHA256 = (
    "efa5e99c59e66e9b632c1cf644ddec8d992434e46f74b41945041b4dde4d3c03"
)
MEMORY_PREFLIGHT_CONTENT_SHA256 = (
    "22ca0ed44a7a24ea522b2df0f26e0dcaaf8b59a1d9567066e03e3a626a6b8eda"
)

EXPECTED_MODEL_LOAD_BYTES = {
    "bf16_a": 69_455_494_144,
    "hybrid_a": 35_986_854_400,
    "bf16_b": 69_455_494_144,
    "hybrid_b": 35_986_854_400,
}
EXPECTED_MEMORY_REDUCTION = 0.48187173896726543
EXPECTED_SPEED = {
    "pair_a": {
        "observed_bf16_over_hybrid_median": 1.0890458935158114,
        "familywise_lcb_headroom": 0.03190353650425881,
        "speed_pass": True,
    },
    "pair_b": {
        "observed_bf16_over_hybrid_median": 1.050405612961755,
        "familywise_lcb_headroom": -0.008087017895568567,
        "speed_pass": False,
    },
}


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


def validate_completed_v24a_r1():
    if (
        file_sha256(ATTEMPT_PATH) != ATTEMPT_FILE_SHA256
        or file_sha256(REPORT_PATH) != REPORT_FILE_SHA256
    ):
        raise RuntimeError("V24A R1 attempt or report file identity changed")
    attempt = json.loads(ATTEMPT_PATH.read_text(encoding="utf-8"))
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if (
        attempt.get("schema")
        != "eggroll-es-durable-launch-attempt-v24a-memory-retry-r1"
        or attempt.get("status") != "complete"
        or attempt.get("phase") != "after_cleanup_and_compact_report"
        or attempt.get("content_sha256_before_self_field")
        != ATTEMPT_CONTENT_SHA256
        or canonical_sha256(without_self(attempt)) != ATTEMPT_CONTENT_SHA256
        or attempt.get("model_update_applied") is not False
        or attempt.get("nontrain_surface_opened") is not False
        or report.get("schema")
        != "eggroll-es-hybrid-backend-compatibility-report-v24a-memory-retry-r1"
        or report.get("content_sha256_before_self_field")
        != REPORT_CONTENT_SHA256
        or canonical_sha256(without_self(report)) != REPORT_CONTENT_SHA256
        or report.get("model_update_applied") is not False
        or report.get("nontrain_surface_opened") is not False
        or report.get("direct_action_taken") is not False
    ):
        raise RuntimeError("V24A R1 completion or closure semantics changed")
    if attempt.get("report_binding") != {
        "path": str(REPORT_PATH),
        "file_sha256": REPORT_FILE_SHA256,
        "content_sha256": REPORT_CONTENT_SHA256,
    }:
        raise RuntimeError("V24A R1 report binding changed")

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
        or implementation.get("memory_retry_overlay_bundle_sha256")
        != OVERLAY_BUNDLE_SHA256
        or canonical_sha256(implementation.get("files"))
        != IMPLEMENTATION_BUNDLE_SHA256
    ):
        raise RuntimeError("V24A R1 committed implementation identity changed")
    for item in source.get("files", {}).values():
        raw = subprocess.check_output(
            ["git", "show", f"{SOURCE_COMMIT}:{item['relative_path']}"],
            cwd=ROOT,
        )
        if hashlib.sha256(raw).hexdigest() != item["file_sha256"]:
            raise RuntimeError("V24A R1 committed source file changed")

    recipe = report.get("recipe", {})
    environment = attempt.get("runtime_environment_certificate", {})
    configuration = report.get("configuration", {})
    if (
        recipe.get("content_sha256_before_self_field")
        != RECIPE_CONTENT_SHA256
        or canonical_sha256(without_self(recipe)) != RECIPE_CONTENT_SHA256
        or environment.get("content_sha256_before_self_field")
        != RUNTIME_ENVIRONMENT_CONTENT_SHA256
        or canonical_sha256(without_self(environment))
        != RUNTIME_ENVIRONMENT_CONTENT_SHA256
        or environment.get("completed_before_attempt_claim") is not True
        or environment.get("cuda_device_count") != 4
        or environment.get("dataset_or_evaluation_surface_opened") is not False
        or configuration.get("content_sha256_before_self_field")
        != CONFIGURATION_CONTENT_SHA256
        or canonical_sha256(without_self(configuration))
        != CONFIGURATION_CONTENT_SHA256
        or "nvml_resident_bytes" in configuration
        or configuration.get("nvml_excluded_from_all_gates") is not True
    ):
        raise RuntimeError("V24A R1 recipe, environment, or configuration changed")

    estimator = report.get("estimator_and_gate", {})
    preflight = estimator.get("memory_endpoint_preflight", {})
    pairs = estimator.get("pairs", {})
    if (
        estimator.get("schema")
        != "eggroll-es-hybrid-backend-compact-estimator-v24a-memory-retry-r1"
        or estimator.get("global_pass") is not False
        or estimator.get("memory_gate_uses_only_vllm_model_load_consumed_bytes")
        is not True
        or estimator.get("nvml_resident_bytes_retained_only_as_diagnostic")
        is not True
        or preflight.get("content_sha256_before_self_field")
        != MEMORY_PREFLIGHT_CONTENT_SHA256
        or canonical_sha256(without_self(preflight))
        != MEMORY_PREFLIGHT_CONTENT_SHA256
        or preflight.get("model_load_consumed_bytes")
        != EXPECTED_MODEL_LOAD_BYTES
        or preflight.get("duplicate_bf16_values_exact") is not True
        or preflight.get("duplicate_hybrid_values_exact") is not True
        or preflight.get("both_pairs_meet_threshold") is not True
        or preflight.get("nvml_excluded_from_gate") is not True
        or set(pairs) != {"pair_a", "pair_b"}
    ):
        raise RuntimeError("V24A R1 model-memory or global gate changed")
    for pair, expected_speed in EXPECTED_SPEED.items():
        value = pairs[pair]
        quality = value.get("quality_endpoints", {})
        memory = value.get("memory", {})
        if (
            len(quality) != 16
            or sum(
                item.get("familywise_lcb_headroom", float("-inf")) >= 0
                for item in quality.values()
            ) != 0
            or value.get("quality_pass") is not False
            or value.get("runtime_integrity_pass") is not True
            or value.get("pair_pass") is not False
            or value.get("speed_pass") is not expected_speed["speed_pass"]
            or value.get("speed_endpoint") != {
                "observed_bf16_over_hybrid_median": expected_speed[
                    "observed_bf16_over_hybrid_median"
                ],
                "familywise_lcb_headroom": expected_speed[
                    "familywise_lcb_headroom"
                ],
                "threshold": 1.05,
            }
            or memory.get("endpoint") != "vllm_model_load_consumed_bytes"
            or memory.get("model_load_reduction") != EXPECTED_MEMORY_REDUCTION
            or memory.get("threshold") != 0.40
            or memory.get("gate_input") is not True
            or value.get("memory_pass") is not True
            or value.get("nvml_resident_memory_diagnostic", {}).get(
                "gate_input"
            ) is not False
        ):
            raise RuntimeError(f"V24A R1 aggregate result changed for {pair}")

    audit = report.get("runtime_audit", {})
    guard = audit.get("matched_full_context_guard", {})
    if (
        audit.get("content_sha256_before_self_field")
        != RUNTIME_AUDIT_CONTENT_SHA256
        or canonical_sha256(without_self(audit))
        != RUNTIME_AUDIT_CONTENT_SHA256
        or audit.get("timed_signed_wave_count") != 64
        or audit.get("perturbed_requests_all_engines") != 71_680
        or audit.get("reference_requests_all_engines") != 1_120
        or audit.get("guard_requests_all_engines") != 2_240
        or audit.get("total_generation_requests") != 75_040
        or audit.get("model_update_applied") is not False
        or audit.get("nontrain_surface_opened") is not False
        or audit.get("nvml_resident_bytes_diagnostic_only") is not True
        or audit.get(
            "per_unit_scores_timing_vectors_raw_outputs_bootstrap_replicates_persisted"
        ) is not False
        or guard.get("memory_preflight_completed_before_phase_a") is not True
        or guard.get("same_request_list_object_value_shape_and_order_a_b_c")
        is not True
        or not all(guard.get("a_b_exact", {}).values())
        or not all(guard.get("a_c_exact", {}).values())
        or guard.get("raw_outputs_or_scores_persisted") is not False
    ):
        raise RuntimeError("V24A R1 compact runtime audit changed")
    return attempt, report


def build_evidence():
    _attempt, report = validate_completed_v24a_r1()
    estimator = report["estimator_and_gate"]
    pairs = estimator["pairs"]
    evidence = {
        "schema": "eggroll-es-v24a-hybrid-negative-aggregate-evidence-r2",
        "status": "valid_completed_train_only_gate_failed",
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
            "memory_retry_overlay_bundle_sha256": OVERLAY_BUNDLE_SHA256,
            "recipe_content_sha256": RECIPE_CONTENT_SHA256,
        },
        "memory_result": {
            "endpoint": "vllm_model_load_consumed_bytes",
            "bf16_bytes": EXPECTED_MODEL_LOAD_BYTES["bf16_a"],
            "hybrid_bytes": EXPECTED_MODEL_LOAD_BYTES["hybrid_a"],
            "reduction": EXPECTED_MEMORY_REDUCTION,
            "threshold": 0.40,
            "duplicate_bf16_exact": True,
            "duplicate_hybrid_exact": True,
            "both_pairs_passed": True,
            "nvml_excluded_from_gate": True,
        },
        "quality_result": {
            "endpoints_per_pair": 16,
            "familywise_endpoints_passed_per_pair": {
                pair: sum(
                    item["familywise_lcb_headroom"] >= 0
                    for item in pairs[pair]["quality_endpoints"].values()
                )
                for pair in ("pair_a", "pair_b")
            },
            "quality_pass_by_pair": {
                pair: pairs[pair]["quality_pass"]
                for pair in ("pair_a", "pair_b")
            },
            "best_familywise_lcb_headroom": max(
                item["familywise_lcb_headroom"]
                for item in pairs["pair_a"]["quality_endpoints"].values()
            ),
            "worst_familywise_lcb_headroom": min(
                item["familywise_lcb_headroom"]
                for item in pairs["pair_a"]["quality_endpoints"].values()
            ),
        },
        "speed_result": {
            pair: EXPECTED_SPEED[pair]
            for pair in ("pair_a", "pair_b")
        },
        "runtime_integrity": {
            "all_four_gpus_one_actor_each": True,
            "all_64_signed_waves_complete": True,
            "all_75_040_generation_requests_accounted": True,
            "exact_restore_and_population_boundary_audit": True,
            "matched_full_context_a_b_and_a_c_exact": True,
            "memory_preflight_before_reference_or_perturbation": True,
            "compact_persistence_only": True,
        },
        "decision": {
            "global_gate_passed": False,
            "confirmation_authorized": False,
            "evaluation_authorized": False,
            "model_update_authorized": False,
            "checkpoint_write_authorized": False,
            "dataset_promotion_authorized": False,
            "retain_backend": "bf16_qwen3.6_35b_a3b_v13_middle_late",
            "reject_backend_for_hpo": (
                "fp8_frozen_backbone_with_layers_20_23_bf16_v24"
            ),
            "reason": "both_duplicate_pairs_failed_quality_compatibility",
        },
        "scope_note": (
            "This closes only the exact V24 hybrid checkpoint and backend "
            "recipe. It does not generalize to other quantization formats, "
            "calibration methods, BF16 layer ranges, or mixed-precision designs."
        ),
        "contains_response_vectors_unit_scores_timing_vectors_or_bootstrap_replicates": False,
        "contains_dataset_rows_questions_answers_or_document_content": False,
        "contains_validation_ood_heldout_or_benchmark_content": False,
    }
    evidence["content_sha256_before_self_field"] = canonical_sha256(evidence)
    return evidence


def exclusive_write(path, value):
    path = Path(path).resolve()
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise ValueError(f"output already exists: {path}") from error
    with os.fdopen(descriptor, "wb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args(argv)
    exclusive_write(args.output, build_evidence())


if __name__ == "__main__":
    main()
