#!/usr/bin/env python3
"""Aggregate-only tests for the V18A negative compatibility evidence."""

import copy
import json

import pytest

import build_eggroll_es_v18a_compact_evidence as evidence_v18a


def test_v18a_evidence_rebuilds_exactly_from_only_two_compact_inputs():
    value = evidence_v18a.build_evidence_v18a()
    assert json.loads(evidence_v18a.OUTPUT_PATH_V18A.read_text()) == value
    assert evidence_v18a.ATTEMPT_PATH_V18A.name.endswith(".json")
    assert evidence_v18a.REPORT_PATH_V18A.name == (
        "production_patch_compat_v18a.json"
    )
    assert value["input_artifacts"]["launch_attempt"]["file_sha256"] == (
        evidence_v18a.ATTEMPT_FILE_SHA256_V18A
    )
    assert value["input_artifacts"]["compact_report"]["file_sha256"] == (
        evidence_v18a.REPORT_FILE_SHA256_V18A
    )


def test_v18a_three_by_twelve_conjunctive_endpoints_recompute_negative_gate():
    value = evidence_v18a.build_evidence_v18a()
    assert set(value["arms"]) == {
        "patch_one_third", "patch_two_thirds", "patch_full",
    }
    assert {
        arm: item["observed_pass_count"]
        for arm, item in value["arms"].items()
    } == evidence_v18a.EXPECTED_OBSERVED_PASS_COUNTS_V18A
    assert all(
        len(item["endpoints"]) == 12
        and item["bootstrap_pass_count"] == 0
        and item["preregistered_gate_passed"] is False
        for item in value["arms"].values()
    )
    assert value["bootstrap"]["seed"] == 20260724
    assert value["bootstrap"]["repetitions"] == 50_000
    assert value["bootstrap"]["one_sided_quantile"] == 0.05 / 36


def test_v18a_runtime_provenance_restoration_backend_and_gpu_evidence_are_bound():
    value = evidence_v18a.build_evidence_v18a()
    assert value["source_provenance"]["file_count"] == 33
    assert value["source_provenance"]["all_files_match_launch_commit"] is True
    assert all(value["runtime_integrity"]["checks"].values())
    assert value["runtime_integrity"]["signed_wave_count"] == 16
    assert value["runtime_integrity"][
        "requests_per_engine_per_signed_wave"
    ] == 1070
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


def test_v18a_negative_evidence_never_authorizes_update_or_eval_and_is_compact():
    value = evidence_v18a.build_evidence_v18a()
    assert value["decision"] == {
        "retain_dataset": "production",
        "retain_recipe": "v13_middle_late_layers_20_23",
        "separate_train_only_recipe_preregistration_authorized": False,
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
    arm = tampered["arms"]["patch_one_third"]
    first = next(iter(arm["endpoints"]))
    arm["endpoints"][first]["bootstrap_noninferiority_pass"] = True
    tampered["content_sha256_before_self_field"] = evidence_v18a.canonical_sha256({
        key: item for key, item in tampered.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="negative evidence changed"):
        evidence_v18a.validate_evidence_v18a(tampered)
