#!/usr/bin/env python3
"""Aggregate-only tests for the V19A negative attribution evidence."""

import copy
import json

import pytest

import build_eggroll_es_v19a_compact_evidence as evidence_v19a


def test_v19a_evidence_rebuilds_exactly_from_only_two_compact_inputs():
    value = evidence_v19a.build_evidence_v19a()
    assert json.loads(evidence_v19a.OUTPUT_PATH_V19A.read_text()) == value
    assert value["input_artifacts"]["launch_attempt"]["file_sha256"] == (
        evidence_v19a.ATTEMPT_FILE_SHA256_V19A
    )
    assert value["input_artifacts"]["compact_report"]["file_sha256"] == (
        evidence_v19a.REPORT_FILE_SHA256_V19A
    )


def test_v19a_three_by_twelve_conjunctive_endpoints_recompute_negative_gate():
    value = evidence_v19a.build_evidence_v19a()
    assert {
        arm: item["observed_pass_count"]
        for arm, item in value["arms"].items()
    } == evidence_v19a.EXPECTED_OBSERVED_PASS_COUNTS_V19A
    assert all(
        len(item["endpoints"]) == 12
        and item["bootstrap_pass_count"] == 0
        and item["preregistered_attribution_gate_passed"] is False
        for item in value["arms"].values()
    )
    assert value["bootstrap"]["seed"] == 20260728
    assert value["bootstrap"]["repetitions"] == 50_000
    assert value["bootstrap"]["one_sided_quantile"] == 0.05 / 36
    assert value["bootstrap"]["whole_panel_block_resampling_used"] is False


def test_v19a_runtime_provenance_backend_and_all_four_gpus_are_bound():
    value = evidence_v19a.build_evidence_v19a()
    assert value["source_provenance"]["file_count"] == 43
    assert value["source_provenance"]["all_files_match_launch_commit"] is True
    assert all(value["runtime_integrity"]["checks"].values())
    assert value["runtime_integrity"]["signed_wave_count"] == 16
    assert value["runtime_integrity"]["panel_count"] == 10
    assert value["runtime_integrity"][
        "requests_per_engine_per_signed_wave"
    ] == 990
    assert value["runtime_integrity"]["dense_result_commitment_count"] == 2560
    assert value["runtime_integrity"][
        "all_four_gpus_observed_every_signed_wave"
    ] is True
    assert value["hardware"] == {
        "engine_count": 4,
        "tp_per_engine": 1,
        "gpu_ids": [0, 1, 2, 3],
        "all_engines_every_signed_wave": True,
    }
    assert value["backend"]["moe_backend"] == "default_triton"
    assert set(value["backend"]["override_environment"].values()) == {None}


def test_v19a_negative_evidence_never_authorizes_update_or_eval_and_is_compact():
    value = evidence_v19a.build_evidence_v19a()
    assert value["decision"] == {
        "retain_dataset": "production",
        "retain_recipe": "v13_middle_late_layers_20_23",
        "separate_fresh_basis_train_only_confirmation_authorized": False,
        "dataset_promotion_authorized": False,
        "model_update_authorized": False,
        "evaluation_authorized": False,
    }
    assert value["contains_response_vectors_or_row_content"] is False
    assert value["contains_validation_ood_heldout_or_benchmark_content"] is False
    serialized = json.dumps(value, sort_keys=True)
    for forbidden in (
        '"questions"', '"answers"', '"prompt_token_ids"', '"unit_scores"',
        '"responses"', '"coefficients"', '"bootstrap_replicates"',
        '"bootstrap_draws"', '"row_content"',
    ):
        assert forbidden not in serialized

    tampered = copy.deepcopy(value)
    arm = tampered["arms"]["patch_tier_2_only"]
    first = next(iter(arm["endpoints"]))
    arm["endpoints"][first]["bootstrap_noninferiority_pass"] = True
    tampered["content_sha256_before_self_field"] = evidence_v19a.canonical_sha256({
        key: item for key, item in tampered.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="negative evidence changed"):
        evidence_v19a.validate_evidence_v19a(tampered)
