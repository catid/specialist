#!/usr/bin/env python3
"""Pure offline tests for the exact V21A paired bootstrap draw plan."""

import copy
import json

import numpy as np
import pytest

import build_eggroll_es_production_v331_patch_draw_plan_v21a as draw_v21a


@pytest.fixture(scope="module")
def arrays_v21a():
    return draw_v21a.materialize_draw_arrays_v21a()


def test_v21a_draw_plan_binds_exact_preregistration():
    persisted = json.loads(draw_v21a.OUTPUT_PATH_V21A.read_text())
    assert persisted["foundation"] == {
        "commit": draw_v21a.PREREG_COMMIT_V21A,
        "preregistration_file_sha256": draw_v21a.PREREG_FILE_SHA256_V21A,
        "preregistration_content_sha256": draw_v21a.PREREG_CONTENT_SHA256_V21A,
    }
    assert draw_v21a.validate_draw_plan_certificate_v21a(persisted) == persisted


def test_v21a_draw_arrays_have_exact_stratified_shapes_and_domains(arrays_v21a):
    base = arrays_v21a["base"]
    assert base.shape == (10, 4, 50_000, 6)
    assert base.dtype == np.uint8
    assert int(base.min()) == 0 and int(base.max()) == 5
    quotas = draw_v21a.prereg_v21a.frame_v21a.CANDIDATE_ONLY_TOPIC_QUOTAS_V21A
    for role, panel_count in draw_v21a.ROLE_PANEL_COUNTS_V21A.items():
        for topic, quota in quotas.items():
            values = arrays_v21a["candidate"][role][topic]
            source_count = draw_v21a.CANDIDATE_SOURCE_SLOT_COUNTS_V21A[role][topic]
            assert values.shape == (panel_count, 50_000, quota)
            assert values.dtype == np.uint8
            assert int(values.min()) == 0
            assert int(values.max()) == source_count - 1


def test_v21a_draw_regeneration_is_byte_exact_and_arrays_are_not_persisted(arrays_v21a):
    second = draw_v21a.materialize_draw_arrays_v21a()
    assert arrays_v21a["base"].tobytes() == second["base"].tobytes()
    for role in draw_v21a.ROLE_PANEL_COUNTS_V21A:
        for topic in arrays_v21a["candidate"][role]:
            assert (
                arrays_v21a["candidate"][role][topic].tobytes()
                == second["candidate"][role][topic].tobytes()
            )
    serialized = draw_v21a.OUTPUT_PATH_V21A.read_text()
    assert '"draw_arrays_persisted": false' in serialized
    assert '"values"' not in serialized


def test_v21a_draw_plan_is_paired_50k_bonferroni_and_no_authority():
    value = json.loads(draw_v21a.OUTPUT_PATH_V21A.read_text())
    assert value["repetitions"] == 50_000
    assert value["hypothesis_count"] == 12
    assert value["one_sided_quantile"] == 0.05 / 12
    assert value["quantile_method"] == "linear"
    assert value["paired_same_draws_both_arms"] is True
    assert value["whole_panel_block_resampling_used"] is False
    for key in (
        "runtime_launch_authorized", "gpu_launch_authorized",
        "model_update_authorized", "checkpoint_write_authorized",
        "evaluation_authorized", "dataset_promotion_authorized",
    ):
        assert value[key] is False


def test_v21a_draw_plan_rejects_shape_pairing_and_authority_tampering():
    original = json.loads(draw_v21a.OUTPUT_PATH_V21A.read_text())
    for mutation in (
        lambda value: value["base_draws"].update({"shape": [1]}),
        lambda value: value.update({"paired_same_draws_both_arms": False}),
        lambda value: value.update({"runtime_launch_authorized": True}),
    ):
        tampered = copy.deepcopy(original)
        mutation(tampered)
        tampered["content_sha256_before_self_field"] = draw_v21a.canonical_sha256({
            key: item for key, item in tampered.items()
            if key != "content_sha256_before_self_field"
        })
        with pytest.raises(RuntimeError, match="draw-plan certificate changed"):
            draw_v21a.validate_draw_plan_certificate_v21a(tampered)
