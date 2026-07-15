#!/usr/bin/env python3

import json

import build_vllm_moe_tuning_retry_preregistration_v27b as v27b


def test_v27b_retry_is_deterministic_and_self_sealed():
    frozen = json.loads(v27b.OUTPUT_PATH.read_text())
    built = v27b.build_preregistration()
    assert built == frozen
    assert frozen["content_sha256_before_self_field"] == v27b.canonical_sha256({
        key: value for key, value in frozen.items()
        if key != "content_sha256_before_self_field"
    })


def test_v27b_changes_only_invalid_compiler_exception_handling():
    value = v27b.build_preregistration()
    overlay = value["retry_overlay"]
    assert overlay["newly_skipped_exception_class_count"] == 1
    assert overlay["newly_skipped_exception_class"] == (
        "triton.compiler.errors.CompilationError"
    )
    assert overlay[
        "benchmark_timing_config_generation_and_selection_unchanged"
    ] is True
    assert overlay["all_other_exceptions_fail_closed"] is True


def test_v27b_does_not_recast_selection_as_evaluation_or_open_data():
    value = v27b.build_preregistration()
    assert value["evaluation"]["retry_output_is_selection_not_evaluation"] is True
    assert value["evaluation"]["evaluation_not_launched_by_this_artifact"] is True
    assert value["authority"][
        "direct_recipe_adoption_model_update_checkpoint_dataset_promotion"
    ] is False
    assert value["authority"]["validation_heldout_ood_or_benchmark_open"] is False
    assert value["contains_dataset_rows_questions_answers_or_document_content"] is False
