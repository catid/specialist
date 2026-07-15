#!/usr/bin/env python3
"""Preregister the train-only V16 fused-MoE task timing/equivalence A/B."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

import train_eggroll_es_specialist_anchor_v13 as anchor_v13


ROOT = Path(__file__).resolve().parent
PREREGISTRATION_PATH_V16 = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_FUSED_MOE_TASK_AB_V16_PREREGISTRATION.json"
).resolve()
PROTOCOL_PATH_V16 = (
    ROOT / "experiments/eggroll_es_hpo/S6_FUSED_MOE_TASK_AB_V16_PROTOCOL.md"
).resolve()
TUNING_DIRECTORY_V16 = (
    ROOT / "experiments/vllm_moe_tuning/"
    "v026_rtx_pro_6000_bf16_tp1_pruned"
).resolve()
TUNED_CONFIG_PATH_V16 = TUNING_DIRECTORY_V16 / (
    "E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_"
    "Max-Q_Workstation_Edition.json"
)
BENCHMARK_EVIDENCE_PATH_V16 = TUNING_DIRECTORY_V16 / "benchmark_evidence.json"
README_PATH_V16 = TUNING_DIRECTORY_V16 / "README.md"
TUNING_TEST_PATH_V16 = ROOT / "test_vllm_moe_pruned_config_v026.py"
V13_EVIDENCE_PATH_V16 = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V13B_TRAIN_PANEL_AGGREGATE_EVIDENCE_V14A.json"
).resolve()

TUNED_CONFIG_FILE_SHA256_V16 = (
    "a6fbb265df9527d0024d531a1779dc19ecb416f80c873036d9713a2fdce9df2d"
)
BENCHMARK_EVIDENCE_FILE_SHA256_V16 = (
    "29a6c21492ef71d326590e959f7cc4228d0185abdac691938f43e9324826ab52"
)
README_FILE_SHA256_V16 = (
    "173cc846d64cbd0ddcb8a56416c16f77a29273e88993cee19f5f500deb57c012"
)
TUNING_TEST_FILE_SHA256_V16 = (
    "166fcf6bf409b3460da54ff847ded03a75930fcf527c3db768a61706b0c00438"
)
V13_EVIDENCE_FILE_SHA256_V16 = (
    "d367c9c4de1e1f3526ddb3dfba2f5bf24efc77cbccf951f7359eb1969fcd7b54"
)
V13_EVIDENCE_CONTENT_SHA256_V16 = (
    "06f662574013345a6c777af8688a38f3941286d9e11a427ed3342de53451b1e3"
)
V13_IMPLEMENTATION_HASHES_V16 = {
    "trainer_v13": "1a8a4145a85c183bb6121914357b7e6bce916b4f76a0693887ac41fa3a8c4c6e",
    "worker_v13": "5596bff9174e5e94e812181a51f8cc9f9b2a73f3a4cb58c45d5346147c8d6367",
    "driver_v13": "1fcd287c62084588d4264376eea01f216bef390561cbd5078ee2f77bac552ce0",
    "tests_v13": "16346a09e6d4e274919cece443c80c221a1f40d89d570b38c657217d58ebfa10",
}
V13_IMPLEMENTATION_PATHS_V16 = {
    "trainer_v13": ROOT / "train_eggroll_es_specialist_anchor_v13.py",
    "worker_v13": ROOT / "eggroll_es_worker_v13.py",
    "driver_v13": ROOT / "run_eggroll_es_train_panels_v13.py",
    "tests_v13": ROOT / "test_eggroll_es_train_panels_runtime_v13.py",
}
CONFIG_EVIDENCE_COMMIT_V16 = (
    "82cdf8e2babfa99aafdbd2eac1233571c56feca1"
)
END_TO_END_EVIDENCE_COMMIT_V16 = (
    "7cb80e6ba8310c8556e1093a77d80526072c7ad7"
)

EXPERIMENT_NAME_V16 = (
    "snapshot794_system_v16_fused_moe_default_vs_tuned_"
    "v13_five_panels_alpha_zero_basis20260714"
)
ARM_ORDER_V16 = ("default_triton", "tuned_triton")
TIMED_SIGNED_WAVE_COUNT_V16 = 16
MINIMUM_TOTAL_SPEEDUP_V16 = 1.05
MINIMUM_MEDIAN_PAIRED_WAVE_SPEEDUP_V16 = 1.05
MINIMUM_NONREGRESSIVE_WAVE_COUNT_V16 = 14
V13_PERTURBATION_BASIS_SHA256_V16 = (
    "29e7ceb1753c39b310a176d827e222b9a5b2c85edf9f2fef5c68b630b8fabc11"
)
V13_PANEL_BUNDLE_CONTENT_SHA256_V16 = (
    "cc176a9b86c6447dcde8a11fd28d68c837d2119715126c57a3f37293fb0d492b"
)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def load_bound_evidence_v16():
    expected = {
        TUNED_CONFIG_PATH_V16: TUNED_CONFIG_FILE_SHA256_V16,
        BENCHMARK_EVIDENCE_PATH_V16: BENCHMARK_EVIDENCE_FILE_SHA256_V16,
        README_PATH_V16: README_FILE_SHA256_V16,
        TUNING_TEST_PATH_V16: TUNING_TEST_FILE_SHA256_V16,
        V13_EVIDENCE_PATH_V16: V13_EVIDENCE_FILE_SHA256_V16,
        **{
            V13_IMPLEMENTATION_PATHS_V16[name]: digest
            for name, digest in V13_IMPLEMENTATION_HASHES_V16.items()
        },
    }
    if any(file_sha256(path) != digest for path, digest in expected.items()):
        raise RuntimeError("v16 sealed evidence or V13 implementation changed")
    benchmark = json.loads(BENCHMARK_EVIDENCE_PATH_V16.read_text())
    config = json.loads(TUNED_CONFIG_PATH_V16.read_text())
    v13 = json.loads(V13_EVIDENCE_PATH_V16.read_text())
    end_to_end = benchmark.get("end_to_end_triton_generation", {})
    if (
        benchmark.get("schema")
        != "specialist-vllm-fused-moe-pruned-benchmark-v1"
        or benchmark.get("status") != "end_to_end_verified_not_training_enabled"
        or end_to_end.get("exact_generated_text_and_token_identity") is not True
        or end_to_end.get("aggregate", {}).get("median_time_speedup")
        != 1.1704957009439074
        or end_to_end.get("aggregate", {}).get("worst_per_gpu_speedup")
        != 1.1646831453362367
        or end_to_end.get("timing_boundary")
        != "generation call only; model load, compile, warmup, and graph capture excluded"
        or set(config) != {
            "1", "16", "32", "64", "128", "256", "512", "1024",
            "2048", "4096", "8192", "16384",
        }
        or v13.get("schema")
        != "eggroll-es-v13b-train-panel-aggregate-evidence-v14a"
        or v13.get("content_sha256_before_self_field")
        != V13_EVIDENCE_CONTENT_SHA256_V16
        or canonical_sha256(_without_self(v13))
        != V13_EVIDENCE_CONTENT_SHA256_V16
        or v13.get("contains_response_vectors_or_row_content") is not False
        or v13.get("contains_validation_ood_or_heldout_content") is not False
    ):
        raise RuntimeError("v16 sealed evidence contract changed")
    return benchmark, config, v13


def build_preregistration_v16():
    benchmark, _config, v13 = load_bound_evidence_v16()
    end_to_end = benchmark["end_to_end_triton_generation"]
    seeds = list(anchor_v13.PERTURBATION_SEEDS_V13)
    if (
        len(seeds) != 32
        or len(set(seeds)) != 32
        or anchor_v13.PERTURBATION_BASIS_SHA256_V13
        != V13_PERTURBATION_BASIS_SHA256_V16
        or anchor_v13.PANEL_BUNDLE_CONTENT_SHA256_V13
        != V13_PANEL_BUNDLE_CONTENT_SHA256_V16
    ):
        raise RuntimeError("v16 V13 basis or panel identity changed")
    value = {
        "schema": "eggroll-es-fused-moe-task-ab-preregistration-v16",
        "status": "preregistered_runtime_not_yet_authorized",
        "experiment_name": EXPERIMENT_NAME_V16,
        "hypothesis_count": 1,
        "hypothesis": (
            "the exact committed RTX Pro fused-MoE config improves generation-"
            "only V13 five-panel diagnostic time without changing any dense "
            "output hash or compact estimator result"
        ),
        "separation": {
            "failed_v15b_architecture_result_used": False,
            "retained_model_recipe": "V13 middle-late layers 20-23",
            "systems_ab_only": True,
            "no_model_or_data_hpo": True,
        },
        "selection_surface": "exact_frozen_v13_train_panels_only",
        "contains_validation_ood_heldout_or_benchmark_content": False,
        "evidence": {
            "config_commit": CONFIG_EVIDENCE_COMMIT_V16,
            "end_to_end_commit": END_TO_END_EVIDENCE_COMMIT_V16,
            "tuned_config": {
                "path": str(TUNED_CONFIG_PATH_V16),
                "file_sha256": TUNED_CONFIG_FILE_SHA256_V16,
            },
            "benchmark": {
                "path": str(BENCHMARK_EVIDENCE_PATH_V16),
                "file_sha256": BENCHMARK_EVIDENCE_FILE_SHA256_V16,
                "sealed_median_time_speedup": end_to_end["aggregate"]
                ["median_time_speedup"],
                "sealed_worst_per_gpu_speedup": end_to_end["aggregate"]
                ["worst_per_gpu_speedup"],
                "exact_output_identity": True,
            },
            "v13": {
                "path": str(V13_EVIDENCE_PATH_V16),
                "file_sha256": V13_EVIDENCE_FILE_SHA256_V16,
                "content_sha256": V13_EVIDENCE_CONTENT_SHA256_V16,
                "stability": copy.deepcopy(v13["stability"]),
            },
        },
        "arms": {
            "order": list(ARM_ORDER_V16),
            "default_triton": {
                "moe_backend": "triton",
                "VLLM_TUNED_CONFIG_FOLDER": None,
            },
            "tuned_triton": {
                "moe_backend": "triton",
                "VLLM_TUNED_CONFIG_FOLDER": str(TUNING_DIRECTORY_V16),
                "config_file_sha256": TUNED_CONFIG_FILE_SHA256_V16,
            },
            "only_intended_difference": "exact_tuned_config_folder_activation",
            "same_model_panels_basis_generation_and_hardware": True,
        },
        "task": {
            "model": str((ROOT / "models/Qwen3.6-35B-A3B").resolve()),
            "layer_plan": "V13 middle-late layers 20-23",
            "alpha": 0.0,
            "model_update_allowed": False,
            "panel_bundle_content_sha256": V13_PANEL_BUNDLE_CONTENT_SHA256_V16,
            "panel_identities": copy.deepcopy(
                anchor_v13.PANEL_ORDERED_ROW_SHA256_V13
            ),
            "panel_names": list(anchor_v13.PANEL_NAMES_V13),
            "panel_size": 56,
            "perturbation_basis": {
                "basis_seed": anchor_v13.PERTURBATION_BASIS_SEED_V13,
                "basis_sha256": V13_PERTURBATION_BASIS_SHA256_V16,
                "seeds": seeds,
                "seed_sha256": canonical_sha256(seeds),
            },
            "sign_order": ["plus", "minus"],
            "generation_seed": 43,
            "temperature": 0.0,
            "max_tokens": 1,
            "prompts_per_direction_and_sign": 280,
        },
        "hardware": {
            "gpu_ids": [0, 1, 2, 3],
            "engine_count": 4,
            "tp_per_engine": 1,
            "all_four_gpus_required_every_signed_wave": True,
            "population_waves": 8,
            "signed_waves": TIMED_SIGNED_WAVE_COUNT_V16,
            "partial_waves_allowed": False,
        },
        "timing_protocol": {
            "clock": "time.perf_counter_ns",
            "boundary": (
                "only blocking four-engine generation resolve for each signed "
                "wave; excludes init, model load, graph capture, JIT warmup, "
                "perturb, restore, scoring, analysis, and cleanup"
            ),
            "unmeasured_warmup": (
                "one exact combined-five-panel reference generation call per "
                "engine after init and before timed signed waves"
            ),
            "paired_wave_order": (
                "same frozen basis order and plus-minus order in both arms"
            ),
            "timed_signed_wave_count_per_arm": TIMED_SIGNED_WAVE_COUNT_V16,
            "raw_content_persisted": False,
            "compact_wave_seconds_persisted": True,
        },
        "promotion_gate": {
            "all_rules_conjunctive": True,
            "exact_equivalence": {
                "diagnostic_content_sha256_equal": True,
                "dense_result_manifest_sha256_equal": True,
                "task_output_sha256_equal": True,
                "compact_estimator_equal": True,
            },
            "timing": {
                "minimum_total_generation_time_speedup": (
                    MINIMUM_TOTAL_SPEEDUP_V16
                ),
                "minimum_median_paired_wave_speedup": (
                    MINIMUM_MEDIAN_PAIRED_WAVE_SPEEDUP_V16
                ),
                "minimum_nonregressive_wave_count": (
                    MINIMUM_NONREGRESSIVE_WAVE_COUNT_V16
                ),
                "required_wave_count": TIMED_SIGNED_WAVE_COUNT_V16,
                "threshold_justification": (
                    "five_percent_is_a_material_conservative_fraction_of_the_"
                    "sealed_17.05_percent_median_and_16.47_percent_worst_gpu_"
                    "synthetic_gain_under_task_shape_shift"
                ),
                "post_hoc_shapes_repetitions_or_endpoints_allowed": False,
            },
            "all_integrity_restoration_and_hardware_audits_required": True,
            "pass_decision": (
                "authorize_only_opt_in_tuned_config_for_a_later_separately_"
                "preregistered_training_experiment"
            ),
            "failure_decision": (
                "keep_tuned_config_disabled_and_retain_default_triton_v13"
            ),
            "pass_does_not_authorize_model_update_or_evaluation": True,
        },
        "integrity": {
            "fresh_o_excl_attempt_and_run_required": True,
            "committed_source_bundle_required": True,
            "report_and_attempt_self_hashes_required": True,
            "exact_reference_restoration_after_every_sign": True,
            "pre_post_base_probe_must_match_exactly": True,
            "population_boundary_audit_required": True,
            "all_four_engines_must_participate_every_signed_wave": True,
            "persist_response_vectors_row_prompt_or_answer_content": False,
        },
        "firewall": {
            "forbidden": [
                "validation", "OOD", "heldout", "benchmark outcomes",
                "model update", "architecture change", "data HPO",
                "post-hoc timing endpoint or shape selection",
            ],
            "all_evaluation_surfaces_remain_closed": True,
        },
        "required_runtime_adapter": {
            "status": "not_yet_implemented",
            "runtime_not_authorized_by_this_preregistration_commit": True,
            "required_files": [
                "train_eggroll_es_fused_moe_task_ab_v16.py",
                "run_eggroll_es_fused_moe_task_ab_v16.py",
                "test_eggroll_es_fused_moe_task_ab_runtime_v16.py",
            ],
            "real_launch_requires_fresh_committed_implementation_bundle": True,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def _timing(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"wave_seconds", "total_seconds"}
        or not isinstance(value.get("wave_seconds"), list)
        or len(value["wave_seconds"]) != TIMED_SIGNED_WAVE_COUNT_V16
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            or float(item) <= 0.0
            for item in value["wave_seconds"]
        )
        or not math.isclose(
            float(value.get("total_seconds", -1.0)),
            math.fsum(float(item) for item in value["wave_seconds"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("v16 generation-only timing contract changed")


def evaluate_candidate_v16(candidate):
    preregistration = build_preregistration_v16()
    expected_keys = {
        "schema", "experiment_name", "alpha", "model_update_applied",
        "validation_ood_heldout_or_benchmark_used", "arm_order", "arms",
        "panel_bundle_content_sha256", "panel_identities",
        "perturbation_basis_sha256", "all_integrity_audits_passed",
        "persisted_response_vectors_or_row_content",
        "content_sha256_before_self_field",
    }
    if (
        not isinstance(candidate, dict)
        or set(candidate) != expected_keys
        or candidate.get("schema") != "eggroll-es-fused-moe-task-ab-summary-v16"
        or candidate.get("experiment_name") != EXPERIMENT_NAME_V16
        or candidate.get("alpha") != 0.0
        or candidate.get("model_update_applied") is not False
        or candidate.get("validation_ood_heldout_or_benchmark_used") is not False
        or candidate.get("arm_order") != list(ARM_ORDER_V16)
        or set(candidate.get("arms", {})) != set(ARM_ORDER_V16)
        or candidate.get("panel_bundle_content_sha256")
        != V13_PANEL_BUNDLE_CONTENT_SHA256_V16
        or candidate.get("panel_identities")
        != preregistration["task"]["panel_identities"]
        or candidate.get("perturbation_basis_sha256")
        != V13_PERTURBATION_BASIS_SHA256_V16
        or candidate.get("all_integrity_audits_passed") is not True
        or candidate.get("persisted_response_vectors_or_row_content") is not False
        or candidate.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(candidate))
    ):
        raise RuntimeError("v16 candidate summary contract changed")
    required_arm = {
        "diagnostic_content_sha256", "dense_result_manifest_sha256",
        "task_output_sha256", "compact_estimator", "generation_timing",
        "all_integrity_audits_passed", "persisted_raw_content",
    }
    for name in ARM_ORDER_V16:
        arm = candidate["arms"][name]
        if (
            not isinstance(arm, dict)
            or set(arm) != required_arm
            or any(
                not isinstance(arm.get(key), str) or len(arm[key]) != 64
                for key in (
                    "diagnostic_content_sha256",
                    "dense_result_manifest_sha256", "task_output_sha256",
                )
            )
            or arm.get("all_integrity_audits_passed") is not True
            or arm.get("persisted_raw_content") is not False
        ):
            raise RuntimeError("v16 candidate arm contract changed")
        _timing(arm["generation_timing"])
    default = candidate["arms"]["default_triton"]
    tuned = candidate["arms"]["tuned_triton"]
    equality = {
        "diagnostic_content_sha256_equal": (
            default["diagnostic_content_sha256"]
            == tuned["diagnostic_content_sha256"]
        ),
        "dense_result_manifest_sha256_equal": (
            default["dense_result_manifest_sha256"]
            == tuned["dense_result_manifest_sha256"]
        ),
        "task_output_sha256_equal": (
            default["task_output_sha256"] == tuned["task_output_sha256"]
        ),
        "compact_estimator_equal": (
            default["compact_estimator"] == tuned["compact_estimator"]
        ),
    }
    default_waves = default["generation_timing"]["wave_seconds"]
    tuned_waves = tuned["generation_timing"]["wave_seconds"]
    paired_speedups = [
        float(base) / float(candidate_time)
        for base, candidate_time in zip(default_waves, tuned_waves)
    ]
    total_speedup = (
        float(default["generation_timing"]["total_seconds"])
        / float(tuned["generation_timing"]["total_seconds"])
    )
    median_speedup = float(statistics.median(paired_speedups))
    nonregressive = sum(value >= 1.0 for value in paired_speedups)
    timing = {
        "total_generation_time_speedup": total_speedup,
        "median_paired_wave_speedup": median_speedup,
        "nonregressive_wave_count": nonregressive,
        "wave_count": len(paired_speedups),
        "total_speedup_passed": total_speedup >= MINIMUM_TOTAL_SPEEDUP_V16,
        "median_speedup_passed": (
            median_speedup >= MINIMUM_MEDIAN_PAIRED_WAVE_SPEEDUP_V16
        ),
        "nonregressive_wave_count_passed": (
            nonregressive >= MINIMUM_NONREGRESSIVE_WAVE_COUNT_V16
        ),
    }
    passed = all(equality.values()) and all(
        timing[key] for key in (
            "total_speedup_passed", "median_speedup_passed",
            "nonregressive_wave_count_passed",
        )
    )
    gate = {
        "schema": "eggroll-es-fused-moe-task-ab-gate-v16",
        "eligible_for_later_opt_in_training_preregistration": passed,
        "eligible_for_model_update": False,
        "eligible_to_open_evaluation": False,
        "exact_equivalence": equality,
        "timing": timing,
        "candidate_content_sha256": candidate[
            "content_sha256_before_self_field"
        ],
        "pass_decision": (
            "authorize_only_later_separate_opt_in_training_preregistration"
            if passed else None
        ),
        "failure_decision": (
            None if passed
            else "keep_tuned_config_disabled_and_retain_default_triton_v13"
        ),
    }
    gate["content_sha256_before_self_field"] = canonical_sha256(gate)
    return gate


def _exclusive_write(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("v16 preregistration already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default=str(PREREGISTRATION_PATH_V16))
    args = parser.parse_args(argv)
    output = Path(args.output_json).resolve()
    if output != PREREGISTRATION_PATH_V16:
        raise ValueError("v16 preregistration requires its canonical path")
    value = build_preregistration_v16()
    _exclusive_write(output, value)
    result = {
        "schema": "eggroll-es-fused-moe-task-ab-preregistration-write-v16",
        "path": str(output),
        "file_sha256": file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
