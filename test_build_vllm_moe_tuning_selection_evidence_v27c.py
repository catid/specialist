#!/usr/bin/env python3

import json

import build_vllm_moe_tuning_selection_evidence_v27c as v27c


def test_v27c_selection_evidence_is_deterministic_and_self_sealed():
    frozen = json.loads(v27c.OUTPUT_PATH.read_text())
    built = v27c.build_evidence()
    assert built == frozen
    assert frozen["content_sha256_before_self_field"] == v27c.canonical_sha256({
        key: value for key, value in frozen.items()
        if key != "content_sha256_before_self_field"
    })


def test_v27c_selected_table_is_exact_and_complete():
    table = v27c.load_selected_table()
    assert table["triton_version"] == "3.6.0"
    assert {int(key) for key in table if key != "triton_version"} == {
        256, 512, 1024, 2048,
    }
    assert v27c.file_sha256(v27c.TABLE_PATH) == v27c.TABLE_FILE_SHA256
    assert v27c.canonical_sha256(table) == v27c.TABLE_CONTENT_SHA256


def test_v27c_selection_does_not_bypass_fresh_evaluation():
    value = v27c.build_evidence()
    assert value["status"] == "valid_completed_selection_not_evaluation"
    assert value["completion"]["exit_code"] == 0
    assert value["completion"]["traceback_occurrences"] == 0
    assert value["next_gate"]["evaluation_not_launched_by_this_artifact"] is True
    assert value["authority"]["direct_recipe_adoption_authorized"] is False
    assert value["authority"][
        "validation_heldout_ood_or_benchmark_open_authorized"
    ] is False
