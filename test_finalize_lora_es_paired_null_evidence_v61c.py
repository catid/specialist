#!/usr/bin/env python3

import json

import pytest

import finalize_lora_es_paired_null_evidence_v61c as subject


def test_v61c_finalizer_rebuilds_and_verifies_all_sealed_numeric_evidence():
    value = subject.build_finalized_v61c()
    assert value["status"] == (
        "complete_numeric_only_evidence_verified_hpo_unauthorized"
    )
    assert value["source_hashes"]["evidence"] == {
        "file_sha256": subject.EVIDENCE_FILE_SHA256,
        "content_sha256": subject.EVIDENCE_CONTENT_SHA256,
    }
    verification = value["verification"]
    assert verification["all_source_file_and_self_hashes_verified"] is True
    assert verification[
        "stored_analysis_exactly_equals_independent_numeric_rebuild"
    ] is True
    assert verification["same_exact_v434_state_before_and_after_all_periods"] is True
    gpu = verification["gpu_activity_recomputed_from_numeric_log"]
    assert gpu["all_four_attributed_positive"] is True
    assert gpu["foreign_compute_process_observations"] == 0
    assert gpu["gpu_log_rows"] == 408
    assert set(gpu["by_gpu"]) == {"0", "1", "2", "3"}
    assert all(item["positive_samples"] > 0 for item in gpu["by_gpu"].values())


def test_v61c_finalizer_distinguishes_raw_and_normalized_null_widths():
    value = subject.build_finalized_v61c()
    null = value["ranking_primary_null"]
    assert null["raw_ci_halfwidths"] == {
        "generated_f1": 0.00039520537364525746,
        "teacher_logprob": 0.00034797419679617725,
    }
    assert null["frozen_practical_effect_scales"] == {
        "generated_f1": 0.01,
        "teacher_logprob": 0.001,
    }
    assert null["normalized_ci_halfwidths"] == {
        "generated_f1": 0.039520537364525744,
        "teacher_logprob": 0.34797419679617725,
    }
    assert null["raw_width_and_normalized_scale_are_not_interchangeable"] is True
    assert null["teacher_forced_logprob_primary_eligible"] is False
    assert null["teacher_eligibility_checks"] == {
        "teacher_null_interval_contains_zero": True,
        "teacher_absolute_point_within_limit": True,
        "teacher_interval_halfwidth_within_limit": True,
        "teacher_repeat_mean_absolute_delta_within_limit": False,
        "teacher_normalized_null_halfwidth_not_above_generated_f1": False,
    }


def test_v61c_finalizer_has_actor_pair_order_unit_and_metric_breakdown():
    value = subject.build_finalized_v61c()
    breakdown = value["ranking_metric_breakdown"]
    overall = breakdown["all_actors_pairs"]
    assert overall["generated_f1"]["comparisons"] == 512
    assert overall["generated_exact"]["zero_delta_count"] == 512
    assert overall["generated_nonzero"]["zero_delta_count"] == 512
    assert overall["all_512_generated_exact_deltas_zero"] is True
    assert overall["all_512_generated_nonzero_deltas_zero"] is True
    assert len(breakdown["by_actor"]) == 4
    assert all(item["generated_f1"]["comparisons"] == 128
               for item in breakdown["by_actor"])
    assert len(breakdown["by_pair"]) == 2
    assert all(item["generated_f1"]["comparisons"] == 256
               for item in breakdown["by_pair"])
    assert [item["candidate_after_reference"] for item in
            breakdown["by_candidate_temporal_order"]] == [False, True]
    assert all(item["teacher_logprob"]["comparisons"] == 256 for item in
               breakdown["by_candidate_temporal_order"])
    assert [item["request_type_order"] for item in
            breakdown["raw_period_actor_means"]] == [
        ["generation", "teacher_forced"],
        ["teacher_forced", "generation"],
        ["generation", "teacher_forced"],
        ["teacher_forced", "generation"],
    ]
    units = breakdown["unit_concentration"]
    assert units["units"] == 64
    assert units["units_with_nonzero_mean_paired_f1_delta"] == 6
    assert units["units_with_any_nonzero_replica_f1_delta"] == 7
    assert units["units_with_nonzero_mean_paired_teacher_logprob_delta"] == 37
    assert units["units_with_any_nonzero_replica_teacher_logprob_delta"] == 39


def test_v61c_finalizer_records_concentrated_sentinel_failure_without_relaxation():
    value = subject.build_finalized_v61c()
    sentinel = value["exact_sentinel"]
    assert sentinel["passed"] is False
    assert sentinel["individual_exact_flip_count"] == 5
    assert sentinel["units_with_any_exact_flip"] == 2
    assert sentinel["total_sentinel_units"] == 4
    assert sentinel["flip_count_histogram_over_affected_units"] == {1: 1, 4: 1}
    assert sorted(sentinel["flip_count_by_unit"].values()) == [1, 4]
    assert len(sentinel["flips"]) == 5
    conclusion = value["conclusion"]
    assert conclusion["teacher_logprob_rejected_as_v61_primary_fitness"] is True
    assert conclusion["exact_sentinel_gate_failed"] is True
    assert conclusion[
        "v61_hpo_update_selection_or_promotion_authorized"
    ] is False
    assert conclusion["thresholds_changed_after_outcomes"] is False
    assert conclusion["causal_source_of_nondeterminism_claimed"] is False
    assert conclusion["holdback_ood_shadow_or_terminal_access_authorized"] is False


def test_v61c_followup_is_proposal_only_and_persists_no_semantics():
    value = subject.build_finalized_v61c()
    followup = value["matched_followup_design"]
    assert followup["status"] == "proposal_only_requires_separate_preregistration"
    assert followup[
        "same_rows_labels_periods_metrics_bootstrap_and_frozen_thresholds"
    ] is True
    assert followup["required_runtime_controls"] == {
        "VLLM_BATCH_INVARIANT": "1",
        "async_scheduling": False,
        "actor_identity_must_verify_both": True,
        "fail_closed_if_qwen_gdn_or_mamba_backend_unsupported": True,
        "base_moe_and_lora_tuned_configs_bypassed": True,
        "batch_invariant_moe_default": {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "SPLIT_K": 1,
        },
        "lora_split_k": 1,
        "v27c_tuned_table_claimed_or_used": False,
        "expected_speed_tradeoff_not_a_correctness_error": True,
    }
    assert followup["hpo_or_protected_access_authorized"] is False
    assert followup["not_a_threshold_relaxation"] is True
    encoded = json.dumps(value, sort_keys=True)
    assert '"question"' not in encoded and '"answer"' not in encoded
    assert value[
        "raw_question_answer_prediction_or_generation_text_persisted"
    ] is False
    assert value["protected_semantics_opened"] is False


def test_v61c_self_hashed_reader_fails_closed_on_file_identity(tmp_path):
    value = {"schema": "synthetic"}
    value["content_sha256_before_self_field"] = (
        subject.analysis.canonical_sha256_v61c(value)
    )
    path = tmp_path / "value.json"
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    with pytest.raises(RuntimeError, match="input file changed"):
        subject._read_self_hashed_v61c(
            path, "0" * 64, value["content_sha256_before_self_field"],
        )
