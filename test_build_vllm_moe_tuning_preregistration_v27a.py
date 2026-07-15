#!/usr/bin/env python3

import json

import build_vllm_moe_tuning_preregistration_v27a as prereg_v27a


def test_v27a_preregistration_is_deterministic_and_self_sealed():
    frozen = json.loads(prereg_v27a.OUTPUT_PATH.read_text())
    built = prereg_v27a.build_preregistration()
    assert built == frozen
    assert frozen["content_sha256_before_self_field"] == prereg_v27a.canonical_sha256({
        key: value for key, value in frozen.items()
        if key != "content_sha256_before_self_field"
    })


def test_v27a_evaluation_is_fresh_paired_and_uses_all_four_gpus():
    value = prereg_v27a.build_preregistration()
    evaluation = value["evaluation"]
    assert value["selection_is_not_evaluation"] is True
    assert evaluation["not_launched_by_this_artifact"] is True
    assert evaluation["fresh_seeds"] == prereg_v27a.EVALUATION_SEEDS
    assert len(evaluation["schedule"]) == 5
    for repetition, item in enumerate(evaluation["schedule"]):
        assert set(item["batch_to_physical_gpu"].values()) == {0, 1, 2, 3}
        expected = ["default", "tuned"] if repetition % 2 == 0 else ["tuned", "default"]
        assert item["arm_order"] == expected


def test_v27a_gate_cannot_directly_change_training_or_open_nontrain_data():
    value = prereg_v27a.build_preregistration()
    authority = value["authority"]
    assert authority[
        "pass_authorizes_only_separate_end_to_end_train_only_runtime_ab_preregistration"
    ] is True
    for key in (
        "direct_training_recipe_adoption_allowed", "model_update_allowed",
        "checkpoint_write_allowed", "dataset_promotion_allowed",
        "validation_heldout_ood_or_benchmark_open_allowed",
    ):
        assert authority[key] is False
    assert value["contains_dataset_rows_questions_answers_or_document_content"] is False
    assert value["contains_validation_heldout_ood_or_benchmark_content"] is False
