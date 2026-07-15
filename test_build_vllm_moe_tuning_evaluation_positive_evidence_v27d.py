#!/usr/bin/env python3

import json

import build_vllm_moe_tuning_evaluation_positive_evidence_v27d as v27d


def test_v27d_positive_evidence_is_deterministic_and_self_sealed():
    frozen = json.loads(v27d.OUTPUT_PATH.read_text())
    built = v27d.build_evidence()
    assert built == frozen
    assert frozen["content_sha256_before_self_field"] == v27d.canonical_sha256({
        key: value for key, value in frozen.items()
        if key != "content_sha256_before_self_field"
    })


def test_v27d_all_preregistered_kernel_gates_passed():
    value = v27d.build_evidence()
    result = value["aggregate_result"]
    for batch in (256, 512, 1024, 2048):
        item = result[f"batch_{batch}"]
        assert item["median_speedup"] >= 1.0
        assert item["tuned_faster_repetitions"] == 5
        assert item["pass"] is True
    assert result["global_geometric_mean_speedup"] >= 1.03
    assert result["global_gate_passed"] is True


def test_v27d_pass_authority_is_narrow_and_data_free():
    value = v27d.build_evidence()
    decision = value["decision"]
    assert decision[
        "authorize_separate_end_to_end_train_only_runtime_ab_preregistration"
    ] is True
    assert decision["direct_recipe_adoption_authorized"] is False
    assert decision["validation_heldout_ood_or_benchmark_open_authorized"] is False
    assert value["contains_dataset_rows_questions_answers_or_document_content"] is False
