import copy
import json

import pytest

import eggroll_es_lagged_replay_calibration_preregistration_v35a as prereg


def test_build_is_self_hashed_and_train_only():
    value = prereg.build_preregistration()
    assert value["content_sha256_before_self_field"] == (
        "27e046a70e849eb69f5da179ef263f065ca492a8d13e7891e7aa972a194fb531"
    )
    assert value["content_sha256_before_self_field"] == prereg.canonical_sha256(
        prereg.without_self(value)
    )
    firewall = value["strict_train_only_firewall"]
    assert firewall["nontrain_evaluation_surfaces_opened"] is False
    assert firewall["excluded_panels_not_generated_or_opened"] == [
        "train_screen_0", "train_screen_1",
    ]
    assert value["authority"]["open_nontrain_evaluation"] is False
    assert value["authority"]["apply_model_update_or_write_checkpoint"] is False


def test_materialized_preregistration_is_exact_rebuild():
    materialized = json.loads(prereg.OUTPUT_PATH.read_text(encoding="utf-8"))
    assert prereg.file_sha256(prereg.OUTPUT_PATH) == (
        "9f8790d99b4e3040304d858817015b451a2cce1abd69c531b611f7911b434272"
    )
    assert materialized == prereg.build_preregistration()


def test_calibration_uses_all_four_gpus_and_exact_equivalence():
    value = prereg.build_preregistration()
    hardware = value["hardware_and_generation"]
    assert hardware["gpu_ids"] == [0, 1, 2, 3]
    assert hardware["engine_count"] == 4
    assert hardware["tp_per_engine"] == 1
    assert hardware["requests_per_engine"] == 168
    assert hardware["total_requests"] == 672
    assert hardware["all_four_engines_generate_the_complete_optimization_union"] is True
    assert hardware["exact_tokens_and_logprobs_required_across_all_four_engines"] is True


def test_tier_counts_are_stratified_capped_and_have_audit_slack():
    value = prereg.build_preregistration()
    calibration = value["difficulty_calibration"]
    assert calibration["provisional_candidate_fraction"] == 0.5
    assert calibration["final_hard_tier_cap"] == 0.25
    assert calibration["provisional_candidates_per_panel"] == 29
    assert calibration["required_final_hard_rows_per_panel"] == 16
    assert calibration["provisional_candidates_total"] == 87
    assert calibration["required_final_hard_rows_total"] == 48
    assert calibration["tier_counts_per_panel"] == {
        "safety_consent": {
            "panel_rows": 9, "provisional_candidates": 5,
            "required_final_hard_rows": 3,
        },
        "technique": {
            "panel_rows": 16, "provisional_candidates": 8,
            "required_final_hard_rows": 4,
        },
        "equipment_material": {
            "panel_rows": 6, "provisional_candidates": 3,
            "required_final_hard_rows": 2,
        },
        "resources_general": {
            "panel_rows": 25, "provisional_candidates": 13,
            "required_final_hard_rows": 7,
        },
    }


def test_manual_gate_is_blinded_and_fail_closed():
    gate = prereg.build_preregistration()["blinded_manual_quality_gate"]
    assert "without_scores_ranks" in gate["reviewer_receives"]
    assert gate["one_decision_per_provisional_candidate"] == [
        "eligible", "ineligible",
    ]
    assert len(gate["eligible_requires_all"]) == 5
    assert gate["insufficient_eligible_candidates"] == (
        "fail_calibration_and_authorize_nothing"
    )
    assert gate["ineligible_rows_are_not_rewritten_or_replaced_inside_this_experiment"] is True


def test_future_hpo_is_separate_fixed_sequence_and_not_authorized_yet():
    value = prereg.build_preregistration()
    future = value["future_HPO_boundary"]
    assert future["recommended_fixed_sequence_replay_fractions"] == [0.1, 0.2]
    assert future["candidate_responses_must_use_a_fresh_direction_basis"] is True
    assert future["untouched_train_screens_used_only_after_tier_freeze"] is True
    assert future["fixed_sequence_familywise_noninferiority_gate_required"] is True
    assert value["authority"]["run_replay_HPO"] is False


def test_no_row_content_or_raw_scores_are_persisted():
    value = prereg.build_preregistration()
    encoded = json.dumps(value, sort_keys=True).lower()
    for forbidden in (
        '"question"', '"answer"', '"prompt"', '"raw_rewards"',
        '"logprobs"', '"tokens"', '"outputs"',
    ):
        assert forbidden not in encoded
    artifact = value["required_content_free_calibration_artifact"]
    assert artifact["raw_content_scores_tokens_logprobs_outputs_or_pids"] is False


def test_mutation_changes_seal():
    value = prereg.build_preregistration()
    changed = copy.deepcopy(value)
    changed["future_HPO_boundary"]["recommended_fixed_sequence_replay_fractions"].reverse()
    assert prereg.canonical_sha256(prereg.without_self(changed)) != value[
        "content_sha256_before_self_field"
    ]


def test_exclusive_write_rejects_wrong_or_existing_path(tmp_path):
    value = prereg.build_preregistration()
    with pytest.raises(ValueError):
        prereg.exclusive_write(tmp_path / "wrong.json", value)
    if prereg.OUTPUT_PATH.exists():
        with pytest.raises(ValueError):
            prereg.exclusive_write(prereg.OUTPUT_PATH, value)
