#!/usr/bin/env python3
"""Build compact aggregate positive evidence for the completed V28A runtime A/B."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH_V28B = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V28B_V27C_TUNED_RUNTIME_AB_POSITIVE_EVIDENCE.json"
)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def build_evidence_v28b():
    pairs = {
        "physical_gpu_0": {
            "load_order": "default_first",
            "median_tuned_over_default_throughput_ratio": 1.0056524358031131,
            "familywise_throughput_lower_confidence_bound": 1.0027792341048494,
            "median_tuned_over_default_peak_vram_ratio": 1.0,
            "familywise_peak_vram_upper_confidence_bound": 1.0,
            "exact_output_equivalence": True,
            "pass": True,
        },
        "physical_gpu_1": {
            "load_order": "tuned_first",
            "median_tuned_over_default_throughput_ratio": 1.016885538255543,
            "familywise_throughput_lower_confidence_bound": 1.0097532163108427,
            "median_tuned_over_default_peak_vram_ratio": 1.0,
            "familywise_peak_vram_upper_confidence_bound": 1.0,
            "exact_output_equivalence": True,
            "pass": True,
        },
        "physical_gpu_2": {
            "load_order": "default_first",
            "median_tuned_over_default_throughput_ratio": 1.0115147720375828,
            "familywise_throughput_lower_confidence_bound": 1.0004798704983742,
            "median_tuned_over_default_peak_vram_ratio": 1.0,
            "familywise_peak_vram_upper_confidence_bound": 1.0,
            "exact_output_equivalence": True,
            "pass": True,
        },
        "physical_gpu_3": {
            "load_order": "tuned_first",
            "median_tuned_over_default_throughput_ratio": 1.0203890375528333,
            "familywise_throughput_lower_confidence_bound": 1.0167190183704558,
            "median_tuned_over_default_peak_vram_ratio": 1.0,
            "familywise_peak_vram_upper_confidence_bound": 1.0,
            "exact_output_equivalence": True,
            "pass": True,
        },
    }
    value = {
        "schema": "eggroll-es-v28a-runtime-positive-evidence-v28b",
        "status": "valid_completed_train_only_runtime_ab_gate_passed",
        "contracts": {
            "v28a_preregistration_commit": (
                "73f6af9c78589d30f586af77841476ab8f197459"
            ),
            "preregistration_file_sha256": (
                "3552010cb5310be3e55daa040339a2377b2c4e528d57919bfd575ca6d29ecc7e"
            ),
            "preregistration_content_sha256": (
                "ca59ca74e63aef9855b8e88d16b392ccb3c0b342c7a4a7d865b544fbef704468"
            ),
            "implementation_bundle_sha256": (
                "a1fcb6640ae76f52f67712e94271c2de321b4b021285b734fc5e2e2be9ef027c"
            ),
            "v28a_overlay_bundle_sha256": (
                "375a5e725cc478beeb88b80bb9669c24fc43f71de752b477553fe0d565ef6d21"
            ),
            "inherited_v23a_r2_bundle_sha256": (
                "4bbd31dbdb61366d6ca61c8f5955df3e59e1b34642cd12aca84b54153586d6a6"
            ),
            "recipe_content_sha256": (
                "70d18ee97bac173c873b40d0ef71b6253b888a272b0d3086a337b3334d58c03b"
            ),
            "bootstrap_draw_plan_sha256": (
                "1077b8209d5f047ad4e34b2a05e07f40774b9bb953c66e69b615505fefb2a416"
            ),
        },
        "artifacts": {
            "attempt": {
                "path": (
                    "experiments/eggroll_es_hpo/runs/"
                    ".s6_v28a_v27c_tuned_vs_empty_default_train_runtime_ab."
                    "launch_attempt.json"
                ),
                "file_sha256": (
                    "4f0a5a8823060b1095ab52fb9ebd7b8b28f3f0eaee072c6f02e36e7cb1b173a5"
                ),
                "content_sha256": (
                    "162281aee9c92b130d524d6d99649831d7eccebab8351bb27bffdbd1cac35238"
                ),
                "status": "complete",
                "phase": "after_actor_cleanup_and_compact_report",
            },
            "report": {
                "path": (
                    "experiments/eggroll_es_hpo/runs/"
                    "s6_v28a_v27c_tuned_vs_empty_default_train_runtime_ab/"
                    "v27c_tuned_runtime_ab_report_v28a.json"
                ),
                "file_sha256": (
                    "94f2bf79633d4fedac6918e505985933623f213c4143a0d374748456b5a02f0a"
                ),
                "content_sha256": (
                    "892205cb9d807e8a47be6b74e55cb69d8167aabd2565bb963e11c4ff463a73cd"
                ),
                "summary_content_sha256": (
                    "ebacb6871ecaf4a3a9adcabe3a12caea8c60ed57eae66f6933d64b354aba2747"
                ),
                "runtime_audit_content_sha256": (
                    "f43d4f7ddbe5130318e2273444fcac735ec049b9205da59a34cc443b7319bee3"
                ),
            },
        },
        "aggregate_result": {
            "pairs": pairs,
            "exact_output_pair_count": 4,
            "shared_exact_output_commitment_sha256": (
                "de6e523465bec823f40a2672d779aeba1878f24cfeea772309cd628bee5d0108"
            ),
            "all_seven_output_components_exact_on_all_four_pairs": True,
            "global_task_throughput": {
                "median_tuned_over_default_ratio": 1.013367626851573,
                "hierarchical_bootstrap_lower_confidence_bound": (
                    1.0074561918588083
                ),
                "point_improvement_threshold": 1.01,
                "lower_bound_improvement_threshold": 1.0,
                "pass": True,
            },
            "all_four_per_gpu_throughput_gates_passed": True,
            "all_four_per_gpu_vram_gates_passed": True,
            "all_eight_absolute_peak_vram_gates_passed": True,
            "global_task_throughput_improvement_gate_passed": True,
            "all_runtime_integrity_audits_passed": True,
            "gate_passed": True,
        },
        "runtime_integrity": {
            "all_four_physical_gpus_active_in_each_wave": True,
            "exact_two_wave_counterbalanced_arm_mapping": True,
            "same_physical_gpu_identity_across_waves": True,
            "fresh_actor_engine_model_and_config_environment_each_wave": True,
            "empty_default_generic_fallback_verified_all_four_appearances": True,
            "exact_committed_v27c_table_verified_all_four_appearances": True,
            "bounded_interwave_cleanup_poll_completed": True,
            "bounded_final_async_cleanup_poll_completed": True,
            "all_four_gpus_idle_after_wave_b_cleanup": True,
            "total_engine_generation_calls_all_waves": 104,
            "total_generation_requests_all_engines_all_waves": 29_120,
            "raw_rows_prompts_answers_token_ids_text_logprobs_timings_memory_samples_or_bootstrap_replicates_persisted": False,
        },
        "decision": {
            "v27c_table_validated_only_for_exact_bf16_runtime_contract": True,
            "bf16_v27c_table_reuse_for_fp8_authorized": False,
            "pass_authority": (
                "authorize_only_a_separately_frozen_train_only_training_recipe_A_B"
            ),
            "direct_recipe_adoption_authorized": False,
            "model_update_authorized": False,
            "checkpoint_write_authorized": False,
            "evaluation_authorized": False,
            "dataset_promotion_authorized": False,
            "validation_heldout_ood_or_benchmark_open_authorized": False,
        },
        "contains_dataset_rows_questions_answers_or_document_content": False,
        "contains_validation_heldout_ood_or_benchmark_content": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


if __name__ == "__main__":
    print(json.dumps(build_evidence_v28b(), indent=2, sort_keys=True))
