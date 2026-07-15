#!/usr/bin/env python3

import json

import build_eggroll_es_v26a_compat_speed_evidence as v26a


def test_v26a_evidence_is_deterministic_and_self_sealed():
    frozen = json.loads(v26a.OUTPUT_PATH.read_text())
    built = v26a.build_evidence()
    assert built == frozen
    assert frozen["content_sha256_before_self_field"] == v26a.canonical_sha256({
        key: value for key, value in frozen.items()
        if key != "content_sha256_before_self_field"
    })


def test_v26a_failed_only_behavioral_equivalence():
    result = v26a.build_evidence()["aggregate_result"]
    assert result["equivalence"]["passing_physical_gpu_pairs"] == 0
    assert result["equivalence"]["pass"] is False
    assert result["all_four_speed_noninferiority_pairs_passed"] is True
    assert result["all_eight_peak_vram_cells_passed"] is True
    assert result["sole_failed_gate"] == "behavioral_equivalence"
    assert result["overall_gate_passed"] is False


def test_v26a_failure_authority_is_narrow_and_data_free():
    value = v26a.build_evidence()
    decision = value["decision"]
    assert decision["retain_existing_full_fp8_model"] is True
    assert decision["authorize_hybrid_training_ab"] is False
    assert decision["model_update_or_checkpoint_authorized"] is False
    assert decision["validation_heldout_ood_or_benchmark_open_authorized"] is False
    assert value["contains_dataset_rows_questions_answers_or_document_content"] is False
