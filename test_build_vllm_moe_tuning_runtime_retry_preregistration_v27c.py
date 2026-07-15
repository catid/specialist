#!/usr/bin/env python3

import json

import build_vllm_moe_tuning_runtime_retry_preregistration_v27c as v27c


def test_v27c_retry_is_deterministic_and_self_sealed():
    frozen = json.loads(v27c.OUTPUT_PATH.read_text())
    built = v27c.build_preregistration()
    assert built == frozen
    assert frozen["content_sha256_before_self_field"] == v27c.canonical_sha256({
        key: value for key, value in frozen.items()
        if key != "content_sha256_before_self_field"
    })


def test_v27c_skips_only_the_exact_observed_mlir_runtime_error():
    value = v27c.build_preregistration()
    overlay = value["retry_overlay"]
    assert overlay["newly_skipped_exception_class"] == "builtins.RuntimeError"
    assert overlay["newly_skipped_exact_message"] == "PassManager::run failed"
    assert overlay["newly_skipped_exact_message_count"] == 1
    assert overlay["all_other_runtime_errors_reraised"] is True
    assert overlay[
        "benchmark_timing_config_generation_and_selection_unchanged"
    ] is True


def test_v27c_binds_failed_attempt_and_keeps_evaluation_closed():
    value = v27c.build_preregistration()
    failed = value["failed_selection_attempt"]
    assert failed["exit_code"] == 1
    assert failed["output_table_written"] is False
    assert failed["output_file_count"] == 0
    assert failed["ray_cause_message"] == "PassManager::run failed"
    assert value["evaluation"]["evaluation_not_launched_by_this_artifact"] is True
    assert value["authority"][
        "direct_recipe_adoption_model_update_checkpoint_dataset_promotion"
    ] is False
    assert value["authority"]["validation_heldout_ood_or_benchmark_open"] is False
    assert value["contains_dataset_rows_questions_answers_or_document_content"] is False
