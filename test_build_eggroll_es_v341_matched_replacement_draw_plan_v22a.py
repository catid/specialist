#!/usr/bin/env python3
"""Offline tests for the exact V22A paired bootstrap draw plan."""

import copy
import json

import numpy as np
import pytest

import build_eggroll_es_v341_matched_replacement_draw_plan_v22a as draw_v22a


@pytest.fixture(scope="module")
def array_v22a():
    return draw_v22a.materialize_draw_array_v22a()


def test_v22a_draw_plan_binds_exact_preregistration():
    persisted = json.loads(draw_v22a.OUTPUT_PATH_V22A.read_text())
    assert persisted["foundation"] == {
        "commit": draw_v22a.PREREG_COMMIT_V22A,
        "preregistration_file_sha256": draw_v22a.PREREG_FILE_SHA256_V22A,
        "preregistration_content_sha256": draw_v22a.PREREG_CONTENT_SHA256_V22A,
    }
    assert draw_v22a.validate_draw_plan_certificate_v22a(persisted) == persisted


def test_v22a_draw_array_has_exact_stratified_shape_domain_and_order(array_v22a):
    assert array_v22a.shape == (10, 4, 50_000, 6)
    assert array_v22a.dtype == np.uint8
    assert int(array_v22a.min()) == 0
    assert int(array_v22a.max()) == 5
    value = json.loads(draw_v22a.OUTPUT_PATH_V22A.read_text())
    assert value["base_draws"]["panel_order"] == list(
        draw_v22a.prereg_v22a.frame_v22a.PANEL_NAMES_V22A
    )
    assert value["base_draws"]["category_order"] == list(
        draw_v22a.prereg_v22a.frame_v22a.BASE_CATEGORIES_V22A
    )


def test_v22a_draw_regeneration_is_byte_exact_and_values_are_not_persisted(
    array_v22a,
):
    second = draw_v22a.materialize_draw_array_v22a()
    assert array_v22a.tobytes() == second.tobytes()
    serialized = draw_v22a.OUTPUT_PATH_V22A.read_text()
    assert '"draw_arrays_persisted": false' in serialized
    assert '"values"' not in serialized


def test_v22a_draw_plan_is_exact_paired_50k_bonferroni_without_candidate_draws():
    value = json.loads(draw_v22a.OUTPUT_PATH_V22A.read_text())
    assert value["seed"] == draw_v22a.prereg_v22a.BOOTSTRAP_SEED_V22A
    assert value["repetitions"] == 50_000
    assert value["hypothesis_count"] == 12
    assert value["one_sided_quantile"] == 0.05 / 12
    assert value["quantile_method"] == "linear"
    assert value["paired_same_draws_both_arms"] is True
    assert value["same_ht_coefficients_and_denominator_both_arms"] is True
    assert value["candidate_only_draws_present"] is False
    assert value["whole_panel_block_resampling_used"] is False


def test_v22a_draw_plan_has_no_runtime_update_eval_or_promotion_authority():
    value = json.loads(draw_v22a.OUTPUT_PATH_V22A.read_text())
    assert value["contains_train_or_evaluation_content"] is False
    for key in (
        "runtime_launch_authorized", "gpu_launch_authorized",
        "model_update_authorized", "checkpoint_write_authorized",
        "evaluation_authorized", "dataset_promotion_authorized",
    ):
        assert value[key] is False


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["base_draws"].update({"shape": [1]}),
        lambda value: value["base_draws"].update({"bytes_sha256": "0" * 64}),
        lambda value: value.update({"one_sided_quantile": 0.05 / 11}),
        lambda value: value.update({"paired_same_draws_both_arms": False}),
        lambda value: value.update({"candidate_only_draws_present": True}),
        lambda value: value.update({"runtime_launch_authorized": True}),
    ),
)
def test_v22a_draw_plan_rejects_hash_shape_gate_pairing_and_authority_tampering(
    mutation,
):
    original = json.loads(draw_v22a.OUTPUT_PATH_V22A.read_text())
    tampered = copy.deepcopy(original)
    mutation(tampered)
    tampered["content_sha256_before_self_field"] = draw_v22a.canonical_sha256({
        key: item for key, item in tampered.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="draw-plan certificate changed"):
        draw_v22a.validate_draw_plan_certificate_v22a(tampered)
