#!/usr/bin/env python3
"""Aggregate-only tests for the V17A negative compatibility evidence."""

import copy
import json

import pytest

import build_eggroll_es_v17a_compact_evidence as evidence_v17a


def test_v17a_evidence_rebuilds_exactly_from_only_two_compact_inputs():
    value = evidence_v17a.build_evidence_v17a()
    assert json.loads(evidence_v17a.OUTPUT_PATH_V17A.read_text()) == value
    assert evidence_v17a.ATTEMPT_PATH_V17A.name.endswith("launch_attempt.json")
    assert evidence_v17a.REPORT_PATH_V17A.name == "paired_data_compat_v17a.json"
    assert value["input_artifacts"]["launch_attempt"]["file_sha256"] == (
        evidence_v17a.ATTEMPT_FILE_SHA256_V17A
    )
    assert value["input_artifacts"]["compact_report"]["file_sha256"] == (
        evidence_v17a.REPORT_FILE_SHA256_V17A
    )


def test_v17a_all_twelve_conjunctive_endpoints_recompute_negative_gate():
    value = evidence_v17a.build_evidence_v17a()
    assert len(value["endpoints"]) == 12
    assert sum(
        item["observed_noninferiority_pass"]
        for item in value["endpoints"].values()
    ) == 9
    assert sum(
        item["bootstrap_noninferiority_pass"]
        for item in value["endpoints"].values()
    ) == 0
    assert all(
        item["observed_noninferiority_pass"]
        == (item["candidate_minus_production"] >= 0.0)
        and item["bootstrap_noninferiority_pass"]
        == (item["familywise_lcb"] >= 0.0)
        for item in value["endpoints"].values()
    )
    assert value["gate_summary"] == {
        "observed_pass_count": 9,
        "observed_required_count": 12,
        "bootstrap_pass_count": 0,
        "bootstrap_required_count": 12,
        "all_rules_conjunctive": True,
        "preregistered_gate_passed": False,
    }
    assert value["bootstrap"]["seed"] == 20260719
    assert value["bootstrap"]["repetitions"] == 20_000
    assert value["bootstrap"]["one_sided_quantile"] == 0.05 / 12


def test_v17a_runtime_provenance_restoration_backend_and_gpu_evidence_are_bound():
    value = evidence_v17a.build_evidence_v17a()
    assert value["source_provenance"]["file_count"] == 32
    assert value["source_provenance"]["all_files_match_launch_commit"] is True
    assert all(value["runtime_integrity"]["checks"].values())
    assert value["runtime_integrity"]["signed_wave_count"] == 16
    assert value["runtime_integrity"][
        "all_four_gpus_observed_every_signed_wave"
    ] is True
    assert value["runtime_integrity"][
        "exact_restoration_and_boundary_audit_passed"
    ] is True
    assert value["hardware"] == {
        "engine_count": 4,
        "tp_per_engine": 1,
        "gpu_ids": [0, 1, 2, 3],
        "all_engines_every_signed_wave": True,
    }
    assert value["backend"]["moe_backend"] == "default_triton"
    assert set(value["backend"]["override_environment"].values()) == {None}


def test_v17a_negative_evidence_never_authorizes_update_or_eval_and_is_compact():
    value = evidence_v17a.build_evidence_v17a()
    assert value["decision"] == {
        "retain_dataset": "production",
        "retain_recipe": "v13",
        "separate_v17b_preregistration_authorized": False,
        "dataset_promotion_authorized": False,
        "model_update_authorized": False,
        "evaluation_authorized": False,
    }
    assert value["contains_response_vectors_or_row_content"] is False
    assert value[
        "contains_validation_ood_heldout_or_benchmark_content"
    ] is False
    serialized = json.dumps(value, sort_keys=True)
    for forbidden in (
        '"questions"', '"answers"', '"prompt_token_ids"', '"unit_scores"',
        '"responses"', '"coefficients"', '"bootstrap_replicates"',
    ):
        assert forbidden not in serialized

    tampered = copy.deepcopy(value)
    first = next(iter(tampered["endpoints"]))
    tampered["endpoints"][first]["bootstrap_noninferiority_pass"] = True
    tampered["content_sha256_before_self_field"] = evidence_v17a.canonical_sha256({
        key: item for key, item in tampered.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="negative evidence changed"):
        evidence_v17a.validate_evidence_v17a(tampered)
