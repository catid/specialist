#!/usr/bin/env python3

import copy
import json

import pytest

import build_eggroll_es_v28a_runtime_positive_evidence_v28b as v28b


def test_v28b_positive_evidence_is_deterministic_and_self_sealed():
    frozen = json.loads(v28b.OUTPUT_PATH_V28B.read_text(encoding="utf-8"))
    built = v28b.build_evidence_v28b()
    assert built == frozen
    payload = copy.deepcopy(frozen)
    payload.pop("content_sha256_before_self_field")
    assert frozen["content_sha256_before_self_field"] == v28b.canonical_sha256(
        payload
    )


def test_v28b_binds_exact_prereg_implementation_recipe_attempt_and_report():
    value = v28b.build_evidence_v28b()
    contracts = value["contracts"]
    assert contracts["v28a_preregistration_commit"] == (
        "73f6af9c78589d30f586af77841476ab8f197459"
    )
    for key in (
        "preregistration_file_sha256",
        "preregistration_content_sha256",
        "implementation_bundle_sha256",
        "recipe_content_sha256",
    ):
        assert len(contracts[key]) == 64
    for artifact in value["artifacts"].values():
        assert len(artifact["file_sha256"]) == 64
        assert len(artifact["content_sha256"]) == 64


def test_v28b_all_preregistered_runtime_gates_passed():
    value = v28b.build_evidence_v28b()
    result = value["aggregate_result"]
    assert result["exact_output_pair_count"] == 4
    assert result["all_seven_output_components_exact_on_all_four_pairs"] is True
    for item in result["pairs"].values():
        assert item["exact_output_equivalence"] is True
        assert item["median_tuned_over_default_throughput_ratio"] >= 1.0
        assert item["familywise_throughput_lower_confidence_bound"] >= 0.98
        assert item["median_tuned_over_default_peak_vram_ratio"] <= 1.01
        assert item["familywise_peak_vram_upper_confidence_bound"] <= 1.02
        assert item["pass"] is True
    global_result = result["global_task_throughput"]
    assert global_result["median_tuned_over_default_ratio"] >= 1.01
    assert global_result["hierarchical_bootstrap_lower_confidence_bound"] >= 1.0
    assert result["gate_passed"] is True


def test_v28b_authority_is_narrow_data_free_and_does_not_transfer_bf16_table():
    value = v28b.build_evidence_v28b()
    decision = value["decision"]
    assert decision["v27c_table_validated_only_for_exact_bf16_runtime_contract"] is True
    assert decision["bf16_v27c_table_reuse_for_fp8_authorized"] is False
    for key in (
        "direct_recipe_adoption_authorized",
        "model_update_authorized",
        "checkpoint_write_authorized",
        "evaluation_authorized",
        "dataset_promotion_authorized",
        "validation_heldout_ood_or_benchmark_open_authorized",
    ):
        assert decision[key] is False
    assert value["contains_dataset_rows_questions_answers_or_document_content"] is False
    assert value["contains_validation_heldout_ood_or_benchmark_content"] is False


def test_v28b_live_compact_artifacts_are_cryptographically_and_semantically_bound():
    attempt, report = v28b.validate_bound_artifacts_v28b()
    assert attempt["report_binding"]["file_sha256"] == v28b.REPORT_FILE_SHA256_V28B
    assert report["implementation"]["bundle_sha256"] == (
        v28b.IMPLEMENTATION_BUNDLE_SHA256_V28B
    )
    assert report["summary"]["gate"]["pass"] is True


@pytest.mark.parametrize(
    ("path_attribute", "source_attribute", "message"),
    (
        ("ATTEMPT_PATH_V28B", "ATTEMPT_PATH_V28B", "attempt file hash changed"),
        ("REPORT_PATH_V28B", "REPORT_PATH_V28B", "report file hash changed"),
    ),
)
def test_v28b_changed_artifact_bytes_fail_closed(
    monkeypatch, tmp_path, path_attribute, source_attribute, message,
):
    source = getattr(v28b, source_attribute)
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b"\n")
    monkeypatch.setattr(v28b, path_attribute, changed)
    with pytest.raises(RuntimeError, match=message):
        v28b.validate_bound_artifacts_v28b()


def _parsed_compact_artifacts():
    return (
        json.loads(v28b.ATTEMPT_PATH_V28B.read_text(encoding="utf-8")),
        json.loads(v28b.REPORT_PATH_V28B.read_text(encoding="utf-8")),
    )


def test_v28b_changed_gate_fails_semantic_validation():
    attempt, report = _parsed_compact_artifacts()
    report["summary"]["gate"]["pass"] = False
    with pytest.raises(RuntimeError, match="authorization gate changed"):
        v28b._validate_semantics_v28b(attempt, report)


def test_v28b_changed_metric_fails_semantic_validation():
    attempt, report = _parsed_compact_artifacts()
    report["summary"]["performance"]["global_task_throughput"][
        "median_tuned_over_default_ratio"
    ] = 99.0
    with pytest.raises(RuntimeError, match="global task throughput changed"):
        v28b._validate_semantics_v28b(attempt, report)


def test_v28b_changed_attempt_report_binding_fails_semantic_validation():
    attempt, report = _parsed_compact_artifacts()
    attempt["report_binding"]["content_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="attempt report binding changed"):
        v28b._validate_semantics_v28b(attempt, report)


def test_v28b_changed_self_hashed_payload_fails_before_semantics():
    attempt, report = _parsed_compact_artifacts()
    report["summary"]["gate"]["pass"] = False
    with pytest.raises(RuntimeError, match="report self-content hash changed"):
        v28b._validate_parsed_artifacts_v28b(attempt, report)
