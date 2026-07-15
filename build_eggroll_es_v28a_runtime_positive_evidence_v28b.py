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
ATTEMPT_PATH_V28B = ROOT / (
    "experiments/eggroll_es_hpo/runs/"
    ".s6_v28a_v27c_tuned_vs_empty_default_train_runtime_ab.launch_attempt.json"
)
REPORT_PATH_V28B = ROOT / (
    "experiments/eggroll_es_hpo/runs/"
    "s6_v28a_v27c_tuned_vs_empty_default_train_runtime_ab/"
    "v27c_tuned_runtime_ab_report_v28a.json"
)
ATTEMPT_FILE_SHA256_V28B = (
    "4f0a5a8823060b1095ab52fb9ebd7b8b28f3f0eaee072c6f02e36e7cb1b173a5"
)
ATTEMPT_CONTENT_SHA256_V28B = (
    "162281aee9c92b130d524d6d99649831d7eccebab8351bb27bffdbd1cac35238"
)
REPORT_FILE_SHA256_V28B = (
    "94f2bf79633d4fedac6918e505985933623f213c4143a0d374748456b5a02f0a"
)
REPORT_CONTENT_SHA256_V28B = (
    "892205cb9d807e8a47be6b74e55cb69d8167aabd2565bb963e11c4ff463a73cd"
)
SUMMARY_CONTENT_SHA256_V28B = (
    "ebacb6871ecaf4a3a9adcabe3a12caea8c60ed57eae66f6933d64b354aba2747"
)
RUNTIME_AUDIT_CONTENT_SHA256_V28B = (
    "f43d4f7ddbe5130318e2273444fcac735ec049b9205da59a34cc443b7319bee3"
)
RECIPE_CONTENT_SHA256_V28B = (
    "70d18ee97bac173c873b40d0ef71b6253b888a272b0d3086a337b3334d58c03b"
)
IMPLEMENTATION_BUNDLE_SHA256_V28B = (
    "a1fcb6640ae76f52f67712e94271c2de321b4b021285b734fc5e2e2be9ef027c"
)
PREREG_FILE_SHA256_V28B = (
    "3552010cb5310be3e55daa040339a2377b2c4e528d57919bfd575ca6d29ecc7e"
)
PREREG_CONTENT_SHA256_V28B = (
    "ca59ca74e63aef9855b8e88d16b392ccb3c0b342c7a4a7d865b544fbef704468"
)
PAIR_NAMES_V28B = tuple(f"physical_gpu_{gpu_id}" for gpu_id in range(4))
EXPECTED_PAIR_METRICS_V28B = {
    "physical_gpu_0": {
        "load_order": "default_first",
        "median_tuned_over_default_throughput_ratio": 1.0056524358031131,
        "familywise_throughput_lower_confidence_bound": 1.0027792341048494,
        "median_tuned_over_default_peak_vram_ratio": 1.0,
        "familywise_peak_vram_upper_confidence_bound": 1.0,
    },
    "physical_gpu_1": {
        "load_order": "tuned_first",
        "median_tuned_over_default_throughput_ratio": 1.016885538255543,
        "familywise_throughput_lower_confidence_bound": 1.0097532163108427,
        "median_tuned_over_default_peak_vram_ratio": 1.0,
        "familywise_peak_vram_upper_confidence_bound": 1.0,
    },
    "physical_gpu_2": {
        "load_order": "default_first",
        "median_tuned_over_default_throughput_ratio": 1.0115147720375828,
        "familywise_throughput_lower_confidence_bound": 1.0004798704983742,
        "median_tuned_over_default_peak_vram_ratio": 1.0,
        "familywise_peak_vram_upper_confidence_bound": 1.0,
    },
    "physical_gpu_3": {
        "load_order": "tuned_first",
        "median_tuned_over_default_throughput_ratio": 1.0203890375528333,
        "familywise_throughput_lower_confidence_bound": 1.0167190183704558,
        "median_tuned_over_default_peak_vram_ratio": 1.0,
        "familywise_peak_vram_upper_confidence_bound": 1.0,
    },
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


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _load_json_object(path):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"V28B artifact changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"V28B JSON object required: {path}")
    return value


def _verify_self_hash(value, expected, label):
    _require(
        isinstance(value, dict)
        and value.get("content_sha256_before_self_field") == expected
        and canonical_sha256(_without_self(value)) == expected,
        f"V28B {label} self-content hash changed",
    )


def _validate_semantics_v28b(attempt, report):
    expected_report_keys = {
        "checkpoint_written", "content_sha256_before_self_field",
        "dataset_promotion_applied", "direct_action_taken",
        "direct_recipe_adoption_applied", "evaluation_opened",
        "implementation", "model_update_applied", "recipe",
        "runtime_audit", "schema", "summary",
    }
    expected_attempt_keys = {
        "checkpoint_written", "content_sha256_before_self_field",
        "dataset_promotion_applied", "direct_recipe_adoption_applied",
        "evaluation_opened", "gpu_idle_certificate", "live_cpu_disk_audit",
        "model_update_applied", "phase", "recipe", "report_binding",
        "runtime_environment_certificate", "schema", "source_provenance",
        "status",
    }
    _require(set(report) == expected_report_keys, "V28B compact report surface changed")
    _require(set(attempt) == expected_attempt_keys, "V28B compact attempt surface changed")
    _require(
        report.get("schema") == "eggroll-es-v28a-task-runtime-report"
        and attempt.get("schema") == "eggroll-es-v28a-durable-launch-attempt"
        and attempt.get("status") == "complete"
        and attempt.get("phase") == "after_actor_cleanup_and_compact_report",
        "V28B completion state changed",
    )

    implementation = report.get("implementation", {})
    recipe = report.get("recipe", {})
    attempt_recipe = attempt.get("recipe", {})
    preregistration = recipe.get("preregistration", {})
    _require(
        implementation.get("bundle_sha256") == IMPLEMENTATION_BUNDLE_SHA256_V28B
        and recipe.get("implementation_bundle_sha256")
        == IMPLEMENTATION_BUNDLE_SHA256_V28B
        and attempt_recipe.get("implementation_bundle_sha256")
        == IMPLEMENTATION_BUNDLE_SHA256_V28B
        and attempt.get("source_provenance", {}).get("implementation_bundle_sha256")
        == IMPLEMENTATION_BUNDLE_SHA256_V28B
        and attempt_recipe == recipe,
        "V28B implementation or recipe cross-binding changed",
    )
    _require(
        preregistration.get("file_sha256") == PREREG_FILE_SHA256_V28B
        and preregistration.get("content_sha256") == PREREG_CONTENT_SHA256_V28B
        and attempt_recipe.get("preregistration") == preregistration,
        "V28B preregistration cross-binding changed",
    )
    binding = attempt.get("report_binding", {})
    _require(
        binding == {
            "path": str(REPORT_PATH_V28B.resolve()),
            "file_sha256": REPORT_FILE_SHA256_V28B,
            "content_sha256": REPORT_CONTENT_SHA256_V28B,
        },
        "V28B attempt report binding changed",
    )

    closed_report = (
        "direct_action_taken", "direct_recipe_adoption_applied",
        "model_update_applied", "checkpoint_written", "evaluation_opened",
        "dataset_promotion_applied",
    )
    closed_attempt = closed_report[1:]
    _require(
        all(report.get(key) is False for key in closed_report)
        and all(attempt.get(key) is False for key in closed_attempt)
        and recipe.get("direct_recipe_adoption_allowed") is False
        and recipe.get("model_update_allowed") is False
        and recipe.get("checkpoint_write_allowed") is False
        and recipe.get("evaluation_allowed") is False
        and recipe.get("dataset_promotion_allowed") is False
        and recipe.get("contains_validation_heldout_ood_or_benchmark_content")
        is False
        and attempt.get("live_cpu_disk_audit", {}).get(
            "dataset_rows_or_nontrain_surfaces_opened"
        ) is False,
        "V28B closed authority or train-only surface changed",
    )

    summary = report.get("summary", {})
    audit = report.get("runtime_audit", {})
    gate = summary.get("gate", {})
    expected_true_gate_fields = (
        "pass", "all_four_exact_output_pairs_passed",
        "all_four_per_gpu_throughput_gates_passed",
        "all_four_per_gpu_vram_gates_passed",
        "all_eight_absolute_peak_vram_gates_passed",
        "global_task_throughput_improvement_gate_passed",
        "all_runtime_integrity_audits_passed",
    )
    expected_false_gate_fields = (
        "direct_recipe_adoption_authorized", "model_update_authorized",
        "checkpoint_write_authorized", "evaluation_authorized",
        "dataset_promotion_authorized",
    )
    _require(
        gate.get("schema") == "eggroll-es-v28a-authorization-gate"
        and gate.get("decision")
        == "authorize_only_a_separately_frozen_train_only_training_recipe_A_B"
        and all(gate.get(key) is True for key in expected_true_gate_fields)
        and all(gate.get(key) is False for key in expected_false_gate_fields)
        and summary.get("direct_recipe_adoption_or_training_action_taken") is False,
        "V28B authorization gate changed",
    )

    equivalence = summary.get("equivalence", {})
    performance = summary.get("performance", {})
    performance_pairs = performance.get("pairs", {})
    _require(
        tuple(equivalence) == PAIR_NAMES_V28B
        and tuple(performance_pairs) == PAIR_NAMES_V28B,
        "V28B physical-GPU pair surface changed",
    )
    exact_component_fields = (
        "exact_output_equivalence", "gold_mean_logprobs_exact",
        "gold_dense_commitments_exact", "generated_token_id_lists_exact",
        "generated_text_exact", "generated_selected_logprobs_exact",
        "generated_cumulative_logprobs_exact", "generated_decoded_tokens_exact",
    )
    shared_commitment = (
        "de6e523465bec823f40a2672d779aeba1878f24cfeea772309cd628bee5d0108"
    )
    for gpu_id, pair_name in enumerate(PAIR_NAMES_V28B):
        exact = equivalence[pair_name]
        measured = performance_pairs[pair_name]
        expected = EXPECTED_PAIR_METRICS_V28B[pair_name]
        _require(
            exact.get("physical_gpu_id") == gpu_id
            and exact.get("load_order") == expected["load_order"]
            and all(exact.get(key) is True for key in exact_component_fields)
            and exact.get("reference_commitment_sha256") == shared_commitment
            and exact.get("candidate_commitment_sha256") == shared_commitment,
            f"V28B exact output equivalence changed: {pair_name}",
        )
        _require(
            measured.get("physical_gpu_id") == gpu_id
            and measured.get("load_order") == expected["load_order"]
            and measured.get("median_tuned_over_default_throughput_ratio")
            == expected["median_tuned_over_default_throughput_ratio"]
            and measured.get("familywise_throughput_lower_confidence_bound")
            == expected["familywise_throughput_lower_confidence_bound"]
            and measured.get("median_tuned_over_default_peak_vram_ratio")
            == expected["median_tuned_over_default_peak_vram_ratio"]
            and measured.get("familywise_peak_vram_upper_confidence_bound")
            == expected["familywise_peak_vram_upper_confidence_bound"]
            and measured.get("throughput_pass") is True
            and measured.get("vram_pass") is True
            and measured.get("absolute_peak_vram_pass") is True
            and measured.get("pass") is True,
            f"V28B performance pair changed: {pair_name}",
        )

    global_result = performance.get("global_task_throughput", {})
    _require(
        global_result == {
            "median_tuned_over_default_ratio": 1.013367626851573,
            "hierarchical_bootstrap_lower_confidence_bound": 1.0074561918588083,
            "point_improvement_threshold": 1.01,
            "lower_bound_improvement_threshold": 1.0,
            "pass": True,
        },
        "V28B global task throughput changed",
    )
    integrity = summary.get("runtime_integrity", {})
    required_integrity_true = (
        "all_four_gpus_idle_after_wave_b_cleanup",
        "all_four_physical_gpus_active_in_each_wave",
        "all_integrity_audits_passed",
        "all_nine_timed_outputs_exact_in_all_eight_cells",
        "bounded_final_async_cleanup_poll_completed",
        "bounded_interwave_cleanup_poll_completed",
        "cross_arm_output_equivalence_assessed_only_by_behavioral_gate",
        "empty_default_generic_fallback_verified_all_four_appearances",
        "exact_bf16_qwen36_model_all_cells",
        "exact_committed_v27c_table_verified_all_four_appearances",
        "exact_positive_evidence_commit_bound",
        "exact_two_wave_counterbalanced_arm_mapping",
        "fresh_actor_engine_model_and_config_environment_each_wave",
        "model_load_compile_warmup_and_teardown_excluded",
        "only_tuned_config_folder_differs_between_arms",
        "placement_groups_mapped_by_string_int_canonicalized_probe",
        "same_nvml_uuid_pci_and_total_memory_per_gpu_across_waves",
        "single_driver_materialization_and_tokenization_all_eight_cells",
        "update_checkpoint_evaluation_adoption_and_promotion_surfaces_closed",
        "within_cell_and_same_arm_outputs_exact",
    )
    _require(
        all(integrity.get(key) is True for key in required_integrity_true)
        and integrity.get("interwave_cleanup_poll_count") == 1
        and integrity.get("final_cleanup_poll_count") == 1
        and audit.get("runtime_integrity") == integrity
        and audit.get("generation_call_count_per_engine_per_wave") == 13
        and audit.get("total_engine_generation_calls_all_waves") == 104
        and audit.get("total_generation_requests_all_engines_all_waves") == 29_120
        and audit.get(
            "raw_rows_prompts_answers_token_ids_text_logprobs_timings_memory_samples_or_bootstrap_replicates_persisted"
        ) is False,
        "V28B runtime integrity or compact audit changed",
    )
    return attempt, report


def _validate_parsed_artifacts_v28b(attempt, report):
    _verify_self_hash(attempt, ATTEMPT_CONTENT_SHA256_V28B, "attempt")
    _verify_self_hash(report, REPORT_CONTENT_SHA256_V28B, "report")
    _verify_self_hash(report.get("recipe"), RECIPE_CONTENT_SHA256_V28B, "recipe")
    _verify_self_hash(report.get("summary"), SUMMARY_CONTENT_SHA256_V28B, "summary")
    _verify_self_hash(
        report.get("runtime_audit"), RUNTIME_AUDIT_CONTENT_SHA256_V28B,
        "runtime audit",
    )
    _verify_self_hash(
        report.get("summary", {}).get("gate"),
        "b989d3ea4107b7599083562996fa4a5560c4b2ef12a794e9e0eb187d554cc249",
        "gate",
    )
    return _validate_semantics_v28b(attempt, report)


def validate_bound_artifacts_v28b():
    _require(
        file_sha256(ATTEMPT_PATH_V28B) == ATTEMPT_FILE_SHA256_V28B,
        "V28B attempt file hash changed",
    )
    _require(
        file_sha256(REPORT_PATH_V28B) == REPORT_FILE_SHA256_V28B,
        "V28B report file hash changed",
    )
    attempt = _load_json_object(ATTEMPT_PATH_V28B)
    report = _load_json_object(REPORT_PATH_V28B)
    return _validate_parsed_artifacts_v28b(attempt, report)


def build_evidence_v28b():
    attempt, report = validate_bound_artifacts_v28b()
    implementation = report["implementation"]
    recipe = report["recipe"]
    summary = report["summary"]
    audit = report["runtime_audit"]
    source_pairs = summary["performance"]["pairs"]
    equivalence = summary["equivalence"]
    pairs = {
        pair_name: {
            "load_order": source_pairs[pair_name]["load_order"],
            "median_tuned_over_default_throughput_ratio": source_pairs[pair_name][
                "median_tuned_over_default_throughput_ratio"
            ],
            "familywise_throughput_lower_confidence_bound": source_pairs[pair_name][
                "familywise_throughput_lower_confidence_bound"
            ],
            "median_tuned_over_default_peak_vram_ratio": source_pairs[pair_name][
                "median_tuned_over_default_peak_vram_ratio"
            ],
            "familywise_peak_vram_upper_confidence_bound": source_pairs[pair_name][
                "familywise_peak_vram_upper_confidence_bound"
            ],
            "exact_output_equivalence": equivalence[pair_name][
                "exact_output_equivalence"
            ],
            "pass": source_pairs[pair_name]["pass"],
        }
        for pair_name in PAIR_NAMES_V28B
    }
    value = {
        "schema": "eggroll-es-v28a-runtime-positive-evidence-v28b",
        "status": "valid_completed_train_only_runtime_ab_gate_passed",
        "contracts": {
            "v28a_preregistration_commit": (
                "73f6af9c78589d30f586af77841476ab8f197459"
            ),
            "preregistration_file_sha256": (
                recipe["preregistration"]["file_sha256"]
            ),
            "preregistration_content_sha256": (
                recipe["preregistration"]["content_sha256"]
            ),
            "implementation_bundle_sha256": (
                implementation["bundle_sha256"]
            ),
            "v28a_overlay_bundle_sha256": (
                implementation["v28a_overlay_bundle_sha256"]
            ),
            "inherited_v23a_r2_bundle_sha256": (
                implementation["inherited_v23a_r2_bundle_sha256"]
            ),
            "recipe_content_sha256": (
                recipe["content_sha256_before_self_field"]
            ),
            "bootstrap_draw_plan_sha256": (
                summary["performance"]["bootstrap"]["draw_plan_sha256"]
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
                    attempt["content_sha256_before_self_field"]
                ),
                "status": attempt["status"],
                "phase": attempt["phase"],
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
                    report["content_sha256_before_self_field"]
                ),
                "summary_content_sha256": (
                    summary["content_sha256_before_self_field"]
                ),
                "runtime_audit_content_sha256": (
                    audit["content_sha256_before_self_field"]
                ),
            },
        },
        "aggregate_result": {
            "pairs": pairs,
            "exact_output_pair_count": 4,
            "shared_exact_output_commitment_sha256": (
                equivalence["physical_gpu_0"]["reference_commitment_sha256"]
            ),
            "all_seven_output_components_exact_on_all_four_pairs": all(
                item["exact_output_equivalence"] is True
                for item in equivalence.values()
            ),
            "global_task_throughput": {
                "median_tuned_over_default_ratio": summary["performance"][
                    "global_task_throughput"
                ]["median_tuned_over_default_ratio"],
                "hierarchical_bootstrap_lower_confidence_bound": (
                    summary["performance"]["global_task_throughput"][
                        "hierarchical_bootstrap_lower_confidence_bound"
                    ]
                ),
                "point_improvement_threshold": summary["performance"][
                    "global_task_throughput"
                ]["point_improvement_threshold"],
                "lower_bound_improvement_threshold": summary["performance"][
                    "global_task_throughput"
                ]["lower_bound_improvement_threshold"],
                "pass": summary["performance"]["global_task_throughput"]["pass"],
            },
            "all_four_per_gpu_throughput_gates_passed": summary["gate"][
                "all_four_per_gpu_throughput_gates_passed"
            ],
            "all_four_per_gpu_vram_gates_passed": summary["gate"][
                "all_four_per_gpu_vram_gates_passed"
            ],
            "all_eight_absolute_peak_vram_gates_passed": summary["gate"][
                "all_eight_absolute_peak_vram_gates_passed"
            ],
            "global_task_throughput_improvement_gate_passed": summary["gate"][
                "global_task_throughput_improvement_gate_passed"
            ],
            "all_runtime_integrity_audits_passed": summary["gate"][
                "all_runtime_integrity_audits_passed"
            ],
            "gate_passed": summary["gate"]["pass"],
        },
        "runtime_integrity": {
            "all_four_physical_gpus_active_in_each_wave": summary["runtime_integrity"][
                "all_four_physical_gpus_active_in_each_wave"
            ],
            "exact_two_wave_counterbalanced_arm_mapping": summary["runtime_integrity"][
                "exact_two_wave_counterbalanced_arm_mapping"
            ],
            "same_physical_gpu_identity_across_waves": summary["runtime_integrity"][
                "same_nvml_uuid_pci_and_total_memory_per_gpu_across_waves"
            ],
            "fresh_actor_engine_model_and_config_environment_each_wave": summary[
                "runtime_integrity"
            ]["fresh_actor_engine_model_and_config_environment_each_wave"],
            "empty_default_generic_fallback_verified_all_four_appearances": summary[
                "runtime_integrity"
            ]["empty_default_generic_fallback_verified_all_four_appearances"],
            "exact_committed_v27c_table_verified_all_four_appearances": summary[
                "runtime_integrity"
            ]["exact_committed_v27c_table_verified_all_four_appearances"],
            "bounded_interwave_cleanup_poll_completed": summary["runtime_integrity"][
                "bounded_interwave_cleanup_poll_completed"
            ],
            "bounded_final_async_cleanup_poll_completed": summary[
                "runtime_integrity"
            ]["bounded_final_async_cleanup_poll_completed"],
            "all_four_gpus_idle_after_wave_b_cleanup": summary["runtime_integrity"][
                "all_four_gpus_idle_after_wave_b_cleanup"
            ],
            "total_engine_generation_calls_all_waves": audit[
                "total_engine_generation_calls_all_waves"
            ],
            "total_generation_requests_all_engines_all_waves": audit[
                "total_generation_requests_all_engines_all_waves"
            ],
            "raw_rows_prompts_answers_token_ids_text_logprobs_timings_memory_samples_or_bootstrap_replicates_persisted": audit[
                "raw_rows_prompts_answers_token_ids_text_logprobs_timings_memory_samples_or_bootstrap_replicates_persisted"
            ],
        },
        "decision": {
            "v27c_table_validated_only_for_exact_bf16_runtime_contract": True,
            "bf16_v27c_table_reuse_for_fp8_authorized": False,
            "pass_authority": (
                "authorize_only_a_separately_frozen_train_only_training_recipe_A_B"
            ),
            "direct_recipe_adoption_authorized": summary["gate"][
                "direct_recipe_adoption_authorized"
            ],
            "model_update_authorized": summary["gate"]["model_update_authorized"],
            "checkpoint_write_authorized": summary["gate"][
                "checkpoint_write_authorized"
            ],
            "evaluation_authorized": summary["gate"]["evaluation_authorized"],
            "dataset_promotion_authorized": summary["gate"][
                "dataset_promotion_authorized"
            ],
            "validation_heldout_ood_or_benchmark_open_authorized": False,
        },
        "contains_dataset_rows_questions_answers_or_document_content": False,
        "contains_validation_heldout_ood_or_benchmark_content": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


if __name__ == "__main__":
    print(json.dumps(build_evidence_v28b(), indent=2, sort_keys=True))
