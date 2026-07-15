#!/usr/bin/env python3
"""CPU-only fail-closed tests for the V23A untouched model seal."""

import copy
import json

import pytest

import build_eggroll_es_insertion_model_seal_v23a as seal_v23a


EXPECTED_CONTENT_SHA256 = (
    "d4cf795408967aefbc77f841c47e6fe2fbe3cefc14a4a0fdb4bf73b2701326f9"
)
EXPECTED_FILE_SHA256 = (
    "96eeb236ea94678f57a530a27a471467d4b3d413d2e7be397e293b695cd4c440"
)


def load_seal():
    return json.loads(seal_v23a.OUTPUT_PATH_V23A.read_text(encoding="utf-8"))


def reseal(value):
    value["content_sha256_before_self_field"] = seal_v23a.canonical_sha256(
        seal_v23a._without_self(value)
    )


def test_v23a_full_model_audit_is_deterministic_and_persisted():
    built = seal_v23a.build_model_seal_v23a()
    persisted = load_seal()
    assert built == persisted
    assert built["content_sha256_before_self_field"] == EXPECTED_CONTENT_SHA256
    assert seal_v23a.file_sha256(seal_v23a.OUTPUT_PATH_V23A) == EXPECTED_FILE_SHA256


def test_v23a_exact_eligible_arms_layers_and_capacity_are_sealed():
    value = load_seal()
    assert value["arm_order"] == [
        "base_middle_late", "insert_front_e005",
        "insert_middle_e005", "insert_back_e005",
    ]
    assert {
        arm: item["target_layers"] for arm, item in value["arms"].items()
    } == {
        "base_middle_late": [20, 21, 22, 23],
        "insert_front_e005": [4, 5, 6, 7],
        "insert_middle_e005": [20, 21, 22, 23],
        "insert_back_e005": [40, 41, 42, 43],
    }
    assert value["capacity_match"]["selected_element_count_per_motif"] == 142_999_552
    assert all(item["shard_count"] == 26 for item in value["arms"].values())


def test_v23a_every_candidate_has_exact_complete_damping_and_copy_audit():
    arms = load_seal()["arms"]
    for arm in ("insert_front_e005", "insert_middle_e005", "insert_back_e005"):
        audit = arms[arm]["tensor_audit"]
        assert audit["inserted_tensor_count"] == 69
        assert audit["exact_undamped_copy_count"] == 57
        assert audit["exact_epsilon_damped_output_count"] == 12
        assert audit["attention_routed_and_shared_outputs_complete"] is True
        assert len(audit["per_layer"]) == 4
        for index, item in enumerate(audit["per_layer"]):
            expected_attention = (
                ".self_attn.o_proj.weight" if index == 3
                else ".linear_attn.out_proj.weight"
            )
            assert set(item["exact_scaled_suffixes"]) == {
                expected_attention,
                ".mlp.experts.down_proj",
                ".mlp.shared_expert.down_proj.weight",
            }


@pytest.mark.parametrize("plan", ["front", "middle", "back"])
def test_v23a_source_mapping_inserts_one_complete_motif(plan):
    mapping = seal_v23a._source_mapping(plan)
    assert len(mapping) == 44
    spec = next(
        item for item in seal_v23a.MODEL_SPECS_V23A.values()
        if item["plan"] == plan
    )
    assert [mapping[index] for index in spec["target_layers"]] == spec["source_layers"]
    assert [
        seal_v23a._expected_layer_types(40)[source] for source in mapping
    ] == seal_v23a._expected_layer_types(44)


def test_v23a_seal_rejects_resealed_eligibility_or_damping_tampering():
    for mutation in ("eligibility", "damping"):
        value = copy.deepcopy(load_seal())
        if mutation == "eligibility":
            value["eligibility"]["old_eval_probe_or_report_used"] = True
        else:
            value["arms"]["insert_front_e005"]["tensor_audit"][
                "attention_routed_and_shared_outputs_complete"
            ] = False
        reseal(value)
        with pytest.raises(RuntimeError, match="model seal changed"):
            seal_v23a.validate_model_seal_v23a(value)


def test_v23a_seal_contains_no_ineligible_model_or_artifact_path():
    value = load_seal()
    for arm in value["arms"].values():
        lowered = arm["path"].lower()
        assert not any(
            pattern in lowered for pattern in seal_v23a.INELIGIBLE_PATTERNS_V23A
        )
        assert arm["untouched_surgery_output_eligible"] is True
        assert arm["old_training_or_probe_artifact_used"] is False
        assert arm["gen20_or_gen30_master_used"] is False


def test_v23a_seal_write_is_scoped_and_immutable(tmp_path, monkeypatch):
    value = load_seal()
    output = tmp_path / "seal.json"
    monkeypatch.setattr(seal_v23a, "OUTPUT_PATH_V23A", output)
    seal_v23a._exclusive_write(output, value)
    assert json.loads(output.read_text(encoding="utf-8")) == value
    with pytest.raises(RuntimeError, match="already exists"):
        seal_v23a._exclusive_write(output, value)
    with pytest.raises(ValueError, match="output path changed"):
        seal_v23a._exclusive_write(tmp_path / "other.json", value)
