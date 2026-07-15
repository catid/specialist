#!/usr/bin/env python3
"""Build compact immutable negative evidence for completed V29H."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ATTEMPT_RELATIVE_PATH_V29I = (
    "experiments/eggroll_es_hpo/runs/"
    ".s6_v29h_serialized_fp8_full_model_runtime_ab.launch_attempt.json"
)
REPORT_RELATIVE_PATH_V29I = (
    "experiments/eggroll_es_hpo/runs/"
    "s6_v29h_serialized_fp8_full_model_runtime_ab/"
    "serialized_fp8_full_model_runtime_ab_report_v29h.json"
)
OUTPUT_RELATIVE_PATH_V29I = (
    "experiments/eggroll_es_hpo/"
    "S6_V29I_V29H_FP8_FULL_MODEL_RUNTIME_AB_NEGATIVE_EVIDENCE.json"
)
ATTEMPT_PATH_V29I = ROOT / ATTEMPT_RELATIVE_PATH_V29I
REPORT_PATH_V29I = ROOT / REPORT_RELATIVE_PATH_V29I
OUTPUT_PATH_V29I = ROOT / OUTPUT_RELATIVE_PATH_V29I

ATTEMPT_FILE_SHA256_V29I = (
    "0d3a176e75a9a947db92cee71e4294fd15ee273a2e516b15a9ab62d37557d9e1"
)
ATTEMPT_CONTENT_SHA256_V29I = (
    "0951f7be51736f40713b196165f0f1ffd6cd6f864fa4265eb32553854a29713d"
)
REPORT_FILE_SHA256_V29I = (
    "126212f8cd2e1ea400eb4036b96b1c8c71dcddd10c03e17ac1fc6b1035846e9e"
)
REPORT_CONTENT_SHA256_V29I = (
    "5d0ff6783f4b0fb104965a2fcee2141b803571cc46e42e3821fa61802bd35360"
)
GATE_CONTENT_SHA256_V29I = (
    "087754b7a46cf2635ee4947ce99cbf13a1395b0e25223c8f9ac4c3aea4b2919d"
)
COMMITTED_SOURCE_SHA256_V29I = (
    "00043862eb6ffe88fb0e9d31c934d995499ee00e4f5672a35dc9268e84a8d2d1"
)
IMPLEMENTATION_BUNDLE_SHA256_V29I = (
    "f6792ec94b9f3be36b528941301cfd7a41394089d9e90c1735bf59d13f6f4b6d"
)
RECIPE_CONTENT_SHA256_V29I = (
    "cac7e02d1b58ffdb67adf47ecf45c06a12b2b68d6317e101f0f777d905fbec2a"
)
RUNTIME_ENVIRONMENT_SHA256_V29I = (
    "7acc87ffb6a50682f19309ea3059823d0fc96dfc459e30cb66d64e157a76f52a"
)
CPU_DISK_AUDIT_SHA256_V29I = (
    "90b65c166a1d8a781aa3e5a0cf3320103ee0024df09bf7b7feb36593637b58fa"
)
PRELAUNCH_IDLE_SHA256_V29I = (
    "942aafe70a29196f650e8abf5e565690f56a72c5b3807e395a61baafde65f87a"
)
FINAL_IDLE_SHA256_V29I = (
    "2277bf71241b6062d994dda5d09dc815898beba3cdf150bdb40351213fafb183"
)
PAIRED_OUTPUT_COMMITMENT_SHA256_V29I = (
    "52e1fc289e24a88f8b397d49e5c438341cbf6a0bd64d41c26c8cbd6490eea258"
)
GROUP_AUDIT_BUNDLE_SHA256_V29I = (
    "51d0dd93451b3526cbca5f98173b7a719dd21f632899f1dba8c755b7d2894a6b"
)
REQUEST_AUDIT_BUNDLE_SHA256_V29I = (
    "bd3b8e7fb0318814a67ce53f14b2758f5eeb7dbddb95a962eddcdc5426e71433"
)
BOOTSTRAP_DRAW_PLAN_SHA256_V29I = (
    "34c02441fd2be849c5d5f240aa64e22311f32bd5527e9f12bbc42217b5c30789"
)

# These contracts are bound transitively by the exact V29H source,
# implementation, recipe, and CPU/disk-audit commitments above.  V29I never
# opens the preregistration, table, checkpoint, dataset, or runtime logs.
PREREGISTRATION_IDENTITY_V29I = {
    "relative_path": (
        "experiments/eggroll_es_hpo/"
        "S6_V29H_FP8_FULL_MODEL_RUNTIME_AB_PREREGISTRATION.json"
    ),
    "file_sha256": (
        "1fd08762b4e4926af9b9614768debe57177788bccf621d8d17ab83fe8393d7f2"
    ),
    "content_sha256": (
        "17e0cf1b7ea560e8e446d50bffcd97f6f110cda4ff0624abd5b37e6ce83908d8"
    ),
}
SELECTED_TABLE_IDENTITY_V29I = {
    "relative_path": (
        "experiments/vllm_moe_tuning/"
        "v025_rtx_pro_6000_fp8_w8a8_block128_tp1_exhaustive_v29b/"
        "E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_"
        "Max-Q_Workstation_Edition,dtype=fp8_w8a8,block_shape=[128,128].json"
    ),
    "commit": "a203f4821c4a737310df75543353d21ce6cea978",
    "file_sha256": (
        "1a4ed0f44c6d7cc788baecd073107b4634db4c769f0820d10174f61117b25618"
    ),
    "content_sha256": (
        "d4a49735ccfd094d6e5a3ee763eca99ed355a51fd11a7c835bfecf9fafeaa50d"
    ),
    "loaded_config_sha256": (
        "ebf00590ac51e66e52f5e99b933d1be72703fbbcc809cc2d585eca8d6b0c0a5d"
    ),
}
SERIALIZED_FP8_CHECKPOINT_IDENTITY_V29I = {
    "path": "/home/catid/specialist/models/Qwen3.6-35B-A3B-FP8",
    "quantization": "fp8_block_128x128",
    "config_sha256": (
        "570ef7ea45a7e1d3de2b1d3c70c4ac3562d0e768acdc195778cb4f4d95025845"
    ),
    "index_sha256": (
        "6f176f344e41d35b17af12904e33401da5ebff3b49fccb8bfa0185bc2d50f9d6"
    ),
    "weight_shards": {
        "file_count": 42,
        "total_bytes": 37_463_662_160,
        "manifest_sha256": (
            "25ae972a0ac80b7875b5e041172d5ad572b522619040f4786a9facdf0e36e5dd"
        ),
    },
    "all_files": {
        "file_count": 56,
        "total_bytes": 37_493_015_668,
        "size_manifest_sha256": (
            "46b80dca12b6cedfa9444cf7e7d13b03175a9e75a6abd775d235b9ad658737b0"
        ),
        "fingerprint_sha256": (
            "b3307a2ce16d029a5ca0b7fb2828070c1fe07c232d396f877ea6d9a3cc9d22c9"
        ),
    },
}

ATTEMPT_KEYS_V29I = {
    "committed_source_sha256",
    "content_sha256_before_self_field",
    "cpu_disk_audit_sha256",
    "final_idle_sha256",
    "implementation_bundle_sha256",
    "model_update_training_checkpoint_evaluation_or_dataset_action_applied",
    "phase",
    "prelaunch_idle_sha256",
    "recipe_sha256",
    "report_binding",
    "runtime_environment_sha256",
    "schema",
    "status",
}
REPORT_KEYS_V29I = {
    "committed_source_sha256",
    "content_sha256_before_self_field",
    "dataset_tokenizer_decoding_or_semantic_content_opened",
    "equivalence",
    "gate",
    "implementation_bundle_sha256",
    "model_update_training_checkpoint_evaluation_or_dataset_action_applied",
    "performance",
    "prelaunch_idle_sha256",
    "raw_tokens_outputs_logprobs_timings_memory_samples_pids_or_draws_persisted",
    "recipe_sha256",
    "runtime_integrity",
    "schema",
    "status",
}
FORBIDDEN_PAYLOAD_KEYS_V29I = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "text", "texts", "token_ids", "outputs", "logprobs", "stdout",
    "stderr", "logs", "pid", "pids", "timings", "memory_samples",
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
    _require(path.is_file() and not path.is_symlink(), f"V29I {label} path changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"V29I {label} must be a JSON object")
    return value


def _verify_self(value, expected, label):
    _require(
        value.get("content_sha256_before_self_field") == expected
        and canonical_sha256(_without_self(value)) == expected,
        f"V29I {label} self hash changed",
    )


def _recursive_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).lower()
            yield from _recursive_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _recursive_keys(item)


def _assert_compact_v29i(value):
    overlap = FORBIDDEN_PAYLOAD_KEYS_V29I & set(_recursive_keys(value))
    _require(
        not overlap,
        f"V29I evidence contains forbidden detailed payload keys: {sorted(overlap)}",
    )


def _validate_attempt_v29i(attempt, report):
    _require(set(attempt) == ATTEMPT_KEYS_V29I, "V29I attempt schema keys changed")
    _verify_self(attempt, ATTEMPT_CONTENT_SHA256_V29I, "attempt")
    _require(
        attempt.get("schema")
        == "vllm-moe-fp8-full-model-durable-attempt-v29h"
        and attempt.get("status") == "complete"
        and attempt.get("phase") == "after_16_groups_and_compact_report"
        and attempt.get("committed_source_sha256")
        == COMMITTED_SOURCE_SHA256_V29I
        and attempt.get("implementation_bundle_sha256")
        == IMPLEMENTATION_BUNDLE_SHA256_V29I
        and attempt.get("recipe_sha256") == RECIPE_CONTENT_SHA256_V29I
        and attempt.get("runtime_environment_sha256")
        == RUNTIME_ENVIRONMENT_SHA256_V29I
        and attempt.get("cpu_disk_audit_sha256") == CPU_DISK_AUDIT_SHA256_V29I
        and attempt.get("prelaunch_idle_sha256") == PRELAUNCH_IDLE_SHA256_V29I
        and attempt.get("final_idle_sha256") == FINAL_IDLE_SHA256_V29I,
        "V29I attempt source, implementation, recipe, or lifecycle binding changed",
    )
    _require(
        attempt.get("report_binding") == {
            "path": str(REPORT_PATH_V29I.resolve()),
            "file_sha256": REPORT_FILE_SHA256_V29I,
            "content_sha256": REPORT_CONTENT_SHA256_V29I,
        },
        "V29I attempt report binding changed",
    )
    _require(
        report.get("committed_source_sha256")
        == attempt["committed_source_sha256"]
        and report.get("implementation_bundle_sha256")
        == attempt["implementation_bundle_sha256"]
        and report.get("recipe_sha256") == attempt["recipe_sha256"]
        and report.get("prelaunch_idle_sha256")
        == attempt["prelaunch_idle_sha256"]
        and report.get("runtime_integrity", {}).get(
            "final_idle_certificate_sha256"
        ) == attempt["final_idle_sha256"],
        "V29I attempt and report bindings disagree",
    )


def _validate_equivalence_v29i(report):
    equivalence = report.get("equivalence", {})
    components = equivalence.get("component_pass_counts", {})
    _require(
        set(equivalence) == {
            "schema", "pair_count", "exact_pair_count", "all_eight_pairs_exact",
            "component_pass_counts", "paired_output_commitment_sha256",
        }
        and equivalence.get("schema")
        == "vllm-moe-fp8-full-model-exact-equivalence-v29h"
        and equivalence.get("pair_count") == 8
        and equivalence.get("exact_pair_count") == 8
        and equivalence.get("all_eight_pairs_exact") is True
        and components == {
            "all_four_engines_generated_every_call": 8,
            "reference_commitment_exact": 8,
            "reference_contracts_exact": 8,
            "timed_commitments_exact": 8,
        }
        and equivalence.get("paired_output_commitment_sha256")
        == PAIRED_OUTPUT_COMMITMENT_SHA256_V29I,
        "V29I exact 8-pair equivalence aggregate changed",
    )
    return equivalence


def _validate_runtime_integrity_v29i(report):
    integrity = report.get("runtime_integrity", {})
    _require(
        set(integrity) == {
            "schema",
            "all_16_groups_activity_config_identity_and_cleanup_passed",
            "all_four_activity_group_count",
            "all_four_finally_idle",
            "all_runtime_integrity_gates_passed",
            "dataset_tokenizer_decoding_or_semantic_content_opened",
            "final_idle_certificate_sha256",
            "fresh_four_engine_group_count",
            "group_audit_bundle_sha256",
            "minimum_activity_sample_count",
            "minimum_simultaneous_positive_sample_count",
            "raw_tokens_outputs_logprobs_timings_memory_samples_pids_or_draws_persisted",
            "request_audit_bundle_sha256",
            "serialized_fp8_tp1_model_load_count",
        }
        and integrity.get("schema")
        == "vllm-moe-fp8-full-model-runtime-integrity-v29h"
        and integrity.get(
            "all_16_groups_activity_config_identity_and_cleanup_passed"
        ) is True
        and integrity.get("fresh_four_engine_group_count") == 16
        and integrity.get("all_four_activity_group_count") == 16
        and integrity.get("serialized_fp8_tp1_model_load_count") == 64
        and integrity.get("all_four_finally_idle") is True
        and integrity.get("all_runtime_integrity_gates_passed") is True
        and integrity.get("dataset_tokenizer_decoding_or_semantic_content_opened")
        is False
        and integrity.get(
            "raw_tokens_outputs_logprobs_timings_memory_samples_pids_or_draws_persisted"
        ) is False
        and integrity.get("minimum_activity_sample_count") == 116
        and integrity.get("minimum_simultaneous_positive_sample_count") == 7
        and integrity.get("final_idle_certificate_sha256")
        == FINAL_IDLE_SHA256_V29I
        and integrity.get("group_audit_bundle_sha256")
        == GROUP_AUDIT_BUNDLE_SHA256_V29I
        and integrity.get("request_audit_bundle_sha256")
        == REQUEST_AUDIT_BUNDLE_SHA256_V29I,
        "V29I 16-group activity, config, identity, cleanup, or load aggregate changed",
    )
    return integrity


def _expected_latency_by_gpu_v29i():
    return {
        "0": {
            "familywise_lower_confidence_bound": 0.9675451811685479,
            "lower_bound_threshold": 0.98,
            "median_default_over_tuned_ratio": 0.9897305259896474,
            "pass": False,
            "point_threshold": 0.99,
            "prompt_tokens": 256,
        },
        "1": {
            "familywise_lower_confidence_bound": 0.9587325096052549,
            "lower_bound_threshold": 0.98,
            "median_default_over_tuned_ratio": 0.9864909829994344,
            "pass": False,
            "point_threshold": 0.99,
            "prompt_tokens": 512,
        },
        "2": {
            "familywise_lower_confidence_bound": 0.9377510410145846,
            "lower_bound_threshold": 0.98,
            "median_default_over_tuned_ratio": 0.9830635536075408,
            "pass": False,
            "point_threshold": 0.99,
            "prompt_tokens": 1024,
        },
        "3": {
            "familywise_lower_confidence_bound": 0.9595379002955822,
            "lower_bound_threshold": 0.98,
            "median_default_over_tuned_ratio": 0.9664193005888292,
            "pass": False,
            "point_threshold": 0.99,
            "prompt_tokens": 2048,
        },
    }


def _expected_peak_vram_by_gpu_v29i():
    endpoint = {
        "familywise_upper_confidence_bound": 1.0,
        "median_tuned_over_default_ratio": 1.0,
        "pass": True,
        "point_threshold": 1.01,
        "upper_bound_threshold": 1.02,
    }
    return {str(gpu): copy.deepcopy(endpoint) for gpu in range(4)}


def _validate_performance_v29i(report):
    performance = report.get("performance", {})
    _require(
        set(performance) == {
            "schema", "absolute_nvml_gate_passed",
            "all_ten_performance_endpoints_passed", "bootstrap_draw_plan_sha256",
            "descriptive_arm_order_global_speed_medians", "global",
            "latency_by_physical_gpu", "matched_timing_calls_per_pair",
            "maximum_absolute_nvml_fraction", "pair_count",
            "peak_vram_by_physical_gpu",
            "raw_timings_memory_or_bootstrap_replicates_persisted",
        }
        and performance.get("schema")
        == "vllm-moe-fp8-full-model-performance-summary-v29h"
        and performance.get("pair_count") == 8
        and performance.get("matched_timing_calls_per_pair") == 7
        and performance.get("bootstrap_draw_plan_sha256")
        == BOOTSTRAP_DRAW_PLAN_SHA256_V29I
        and performance.get("raw_timings_memory_or_bootstrap_replicates_persisted")
        is False
        and performance.get("all_ten_performance_endpoints_passed") is False,
        "V29I compact performance aggregate changed",
    )
    _require(
        performance.get("descriptive_arm_order_global_speed_medians") == {
            "default_first": 0.980778736792325,
            "tuned_first": 0.9818850569411415,
        },
        "V29I arm-order descriptive latency aggregate changed",
    )
    global_endpoints = performance.get("global", {})
    _require(
        set(global_endpoints) == {"full_model_latency", "peak_vram"}
        and global_endpoints.get("full_model_latency") == {
            "familywise_lower_confidence_bound": 0.9633065683884765,
            "geometric_mean_speedup": 0.9813847810295329,
            "lower_bound_threshold": 0.99,
            "pass": False,
            "point_threshold": 1.002,
        },
        "V29I global full-model latency failure changed",
    )
    expected_global_vram = {
        "familywise_upper_confidence_bound": 1.0,
        "max_per_gpu_median_ratio": 1.0,
        "pass": True,
        "point_threshold": 1.01,
        "upper_bound_threshold": 1.02,
    }
    _require(
        global_endpoints.get("peak_vram") == expected_global_vram
        and performance.get("peak_vram_by_physical_gpu")
        == _expected_peak_vram_by_gpu_v29i(),
        "V29I global or per-GPU VRAM endpoints changed",
    )
    latencies = performance.get("latency_by_physical_gpu", {})
    _require(
        latencies == _expected_latency_by_gpu_v29i()
        and all(endpoint["pass"] is False for endpoint in latencies.values()),
        "V29I four per-GPU latency failures changed",
    )
    _require(
        performance.get("maximum_absolute_nvml_fraction")
        == 0.7951821998835392
        and performance.get("absolute_nvml_gate_passed") is True,
        "V29I absolute NVML gate changed",
    )
    return performance


def _validate_closed_authority_v29i(attempt, report):
    gate = report.get("gate", {})
    _verify_self(gate, GATE_CONTENT_SHA256_V29I, "authorization gate")
    authority_keys = (
        "checkpoint_write_authorized",
        "dataset_promotion_authorized",
        "direct_table_or_recipe_adoption_authorized",
        "evaluation_validation_heldout_ood_or_benchmark_access_authorized",
        "model_update_or_training_authorized",
        "nontrain_runtime_reuse_authorized",
    )
    _require(
        set(gate) == {
            "schema", "absolute_nvml_gate_passed", "all_eight_pairs_exact",
            "all_runtime_integrity_gates_passed",
            "all_ten_performance_endpoints_passed", "checkpoint_write_authorized",
            "content_sha256_before_self_field", "dataset_promotion_authorized",
            "decision", "direct_table_or_recipe_adoption_authorized",
            "evaluation_validation_heldout_ood_or_benchmark_access_authorized",
            "model_update_or_training_authorized", "nontrain_runtime_reuse_authorized",
            "pass",
        }
        and gate.get("schema")
        == "vllm-moe-fp8-full-model-authorization-gate-v29h"
        and gate.get("pass") is False
        and gate.get("decision")
        == "retain_empty_default_serialized_fp8_runtime"
        and gate.get("all_eight_pairs_exact") is True
        and gate.get("all_runtime_integrity_gates_passed") is True
        and gate.get("absolute_nvml_gate_passed") is True
        and gate.get("all_ten_performance_endpoints_passed") is False
        and all(gate.get(key) is False for key in authority_keys),
        "V29I closed authorization gate changed",
    )
    side_effect_key = (
        "model_update_training_checkpoint_evaluation_or_dataset_action_applied"
    )
    _require(
        attempt.get(side_effect_key) is False
        and report.get(side_effect_key) is False
        and report.get("dataset_tokenizer_decoding_or_semantic_content_opened")
        is False
        and report.get(
            "raw_tokens_outputs_logprobs_timings_memory_samples_pids_or_draws_persisted"
        ) is False,
        "V29I forbidden side effect, semantic access, or detailed persistence changed",
    )
    return gate


def validate_bound_artifacts_v29i():
    _require(
        file_sha256(ATTEMPT_PATH_V29I) == ATTEMPT_FILE_SHA256_V29I,
        "V29I attempt file hash changed",
    )
    _require(
        file_sha256(REPORT_PATH_V29I) == REPORT_FILE_SHA256_V29I,
        "V29I report file hash changed",
    )
    attempt = _load_json_object(ATTEMPT_PATH_V29I, "attempt")
    report = _load_json_object(REPORT_PATH_V29I, "report")
    _require(set(report) == REPORT_KEYS_V29I, "V29I report schema keys changed")
    _verify_self(report, REPORT_CONTENT_SHA256_V29I, "report")
    _require(
        report.get("schema")
        == "vllm-moe-fp8-full-model-runtime-ab-report-v29h"
        and report.get("status")
        == "valid_completed_synthetic_train_runtime_only_ab",
        "V29I report schema or completion status changed",
    )
    _validate_attempt_v29i(attempt, report)
    _validate_equivalence_v29i(report)
    _validate_runtime_integrity_v29i(report)
    _validate_performance_v29i(report)
    _validate_closed_authority_v29i(attempt, report)
    return attempt, report


def build_negative_evidence_v29i():
    attempt, report = validate_bound_artifacts_v29i()
    equivalence = report["equivalence"]
    integrity = report["runtime_integrity"]
    performance = report["performance"]
    value = {
        "schema": "vllm-moe-fp8-full-model-negative-evidence-v29i",
        "status": "sealed_completed_compact_negative_evidence",
        "artifacts": {
            "durable_attempt": {
                "relative_path": ATTEMPT_RELATIVE_PATH_V29I,
                "file_sha256": ATTEMPT_FILE_SHA256_V29I,
                "content_sha256": ATTEMPT_CONTENT_SHA256_V29I,
            },
            "compact_report": {
                "relative_path": REPORT_RELATIVE_PATH_V29I,
                "file_sha256": REPORT_FILE_SHA256_V29I,
                "content_sha256": REPORT_CONTENT_SHA256_V29I,
            },
        },
        "frozen_bindings": {
            "committed_source_sha256": attempt["committed_source_sha256"],
            "implementation_bundle_sha256": attempt[
                "implementation_bundle_sha256"
            ],
            "recipe_content_sha256": attempt["recipe_sha256"],
            "runtime_environment_sha256": attempt[
                "runtime_environment_sha256"
            ],
            "cpu_disk_audit_sha256": attempt["cpu_disk_audit_sha256"],
            "prelaunch_idle_sha256": attempt["prelaunch_idle_sha256"],
            "final_idle_sha256": attempt["final_idle_sha256"],
            "authorization_gate_content_sha256": GATE_CONTENT_SHA256_V29I,
        },
        "transitively_frozen_contract_identities": {
            "binding_basis": {
                "committed_source_sha256": COMMITTED_SOURCE_SHA256_V29I,
                "implementation_bundle_sha256": IMPLEMENTATION_BUNDLE_SHA256_V29I,
                "recipe_content_sha256": RECIPE_CONTENT_SHA256_V29I,
                "cpu_disk_audit_sha256": CPU_DISK_AUDIT_SHA256_V29I,
            },
            "v29h_preregistration": copy.deepcopy(
                PREREGISTRATION_IDENTITY_V29I
            ),
            "selected_table": copy.deepcopy(SELECTED_TABLE_IDENTITY_V29I),
            "serialized_fp8_checkpoint": copy.deepcopy(
                SERIALIZED_FP8_CHECKPOINT_IDENTITY_V29I
            ),
        },
        "aggregate_execution": {
            "equivalence_pair_count": equivalence["pair_count"],
            "exact_equivalence_pair_count": equivalence["exact_pair_count"],
            "equivalence_component_count": len(
                equivalence["component_pass_counts"]
            ),
            "all_equivalence_components_exact_for_all_pairs": True,
            "paired_output_commitment_sha256": equivalence[
                "paired_output_commitment_sha256"
            ],
            "fresh_four_engine_group_count": integrity[
                "fresh_four_engine_group_count"
            ],
            "all_four_activity_group_count": integrity[
                "all_four_activity_group_count"
            ],
            "all_16_groups_activity_config_identity_and_cleanup_passed": True,
            "serialized_fp8_tp1_model_load_count": integrity[
                "serialized_fp8_tp1_model_load_count"
            ],
            "all_four_finally_idle": True,
            "minimum_activity_sample_count": integrity[
                "minimum_activity_sample_count"
            ],
            "minimum_simultaneous_positive_sample_count": integrity[
                "minimum_simultaneous_positive_sample_count"
            ],
            "group_audit_bundle_sha256": integrity[
                "group_audit_bundle_sha256"
            ],
            "request_audit_bundle_sha256": integrity[
                "request_audit_bundle_sha256"
            ],
        },
        "aggregate_performance": {
            "inferential_endpoint_count": 10,
            "passing_endpoint_count": 5,
            "failing_endpoint_count": 5,
            "all_five_latency_endpoints_failed": True,
            "all_five_vram_endpoints_passed": True,
            "global_full_model_latency": copy.deepcopy(
                performance["global"]["full_model_latency"]
            ),
            "latency_by_physical_gpu": copy.deepcopy(
                performance["latency_by_physical_gpu"]
            ),
            "global_peak_vram": copy.deepcopy(
                performance["global"]["peak_vram"]
            ),
            "peak_vram_by_physical_gpu": copy.deepcopy(
                performance["peak_vram_by_physical_gpu"]
            ),
            "maximum_absolute_nvml_fraction": performance[
                "maximum_absolute_nvml_fraction"
            ],
            "absolute_nvml_fraction_limit": 0.95,
            "absolute_nvml_gate_passed": True,
            "bootstrap_draw_plan_sha256": performance[
                "bootstrap_draw_plan_sha256"
            ],
        },
        "decision": {
            "retain_empty_default_serialized_fp8_runtime": True,
            "direct_table_or_recipe_adoption_authority": False,
            "model_update_or_training_authority": False,
            "checkpoint_write_authority": False,
            "evaluation_validation_heldout_ood_or_benchmark_access_authority": False,
            "dataset_promotion_authority": False,
            "nontrain_runtime_reuse_authority": False,
        },
        "side_effects": {
            "model_update_training_checkpoint_evaluation_or_dataset_action_applied": False,
            "dataset_tokenizer_decoding_or_semantic_content_opened": False,
            "detailed_runtime_payloads_persisted": False,
        },
        "input_scope": {
            "compact_attempt_and_report_aggregates_only": True,
            "preregistration_table_or_checkpoint_content_opened": False,
            "raw_stdout_or_runtime_logs_opened": False,
            "dataset_or_semantic_content_opened": False,
            "gpu_launched": False,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    _assert_compact_v29i(value)
    return value


def _exclusive_write_json_v29i(path, value):
    path = Path(path).resolve()
    if path != OUTPUT_PATH_V29I.resolve():
        raise ValueError("V29I evidence output path changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise RuntimeError("V29I evidence already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    value = build_negative_evidence_v29i()
    if not args.dry_run:
        _exclusive_write_json_v29i(OUTPUT_PATH_V29I, value)
    print(json.dumps({
        "schema": "vllm-moe-fp8-full-model-negative-evidence-build-v29i",
        "content_sha256": value["content_sha256_before_self_field"],
        "gate_pass": False,
        "retain_empty_default_serialized_fp8_runtime": True,
        "direct_adoption_authority": False,
        "gpu_launched": False,
        "dataset_or_semantic_content_opened": False,
    }, indent=2, sort_keys=True))
    return value


if __name__ == "__main__":
    main()
