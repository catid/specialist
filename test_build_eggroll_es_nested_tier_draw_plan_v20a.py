#!/usr/bin/env python3
"""Offline tests for the sealed V20A bootstrap draw plan."""

import copy
import hashlib
import json

import pytest

import build_eggroll_es_nested_tier_draw_plan_v20a as draw_v20a


def test_v20a_draw_plan_is_deterministic_persisted_and_exact():
    first = draw_v20a.build_draw_plan_certificate_v20a()
    second = draw_v20a.build_draw_plan_certificate_v20a()
    persisted = json.loads(draw_v20a.OUTPUT_PATH_V20A.read_text())
    assert first == second == persisted
    assert draw_v20a.file_sha256(draw_v20a.OUTPUT_PATH_V20A) == (
        "9032aa332f9b143d5f00dde7a522b5085163c6727a0c13bf8670664c24a99fff"
    )
    assert first["content_sha256_before_self_field"] == (
        "2eb5de70d60be3178ea8f27ffcf7a54293fdfa3c4ed412cfb0f0093e7c5fae28"
    )
    assert first["foundation"]["commit"] == (
        "f8860e14c693020badf25985cb2ba6b4d4339e30"
    )


def test_v20a_draw_arrays_match_committed_hashes_and_nested_sharing():
    certificate = draw_v20a.build_draw_plan_certificate_v20a()
    arrays = draw_v20a.materialize_draw_arrays_v20a()
    assert list(arrays["base"].shape) == [10, 4, 50_000, 6]
    assert list(arrays["candidate_source_offsets"].shape) == [3, 10, 50_000]
    assert hashlib.sha256(arrays["base"].tobytes()).hexdigest() == (
        certificate["base_draws"]["bytes_sha256"]
    )
    assert hashlib.sha256(
        arrays["candidate_source_offsets"].tobytes()
    ).hexdigest() == certificate["candidate_source_offsets"]["bytes_sha256"]
    assert certificate["base_draws"]["shared_across_all_four_arms"] is True
    assert certificate["candidate_source_offsets"][
        "shared_by_every_nested_arm_containing_each_tier"
    ] is True
    assert certificate["candidate_source_offsets"]["tier_order"] == [1, 2, 3]


def test_v20a_draw_plan_is_content_free_fail_closed_and_opens_no_authority():
    value = draw_v20a.build_draw_plan_certificate_v20a()
    assert value["repetitions"] == 50_000
    assert value["hypothesis_count"] == 60
    assert value["one_sided_quantile"] == 0.05 / 60
    assert value["draw_arrays_persisted"] is False
    assert value["contains_train_or_evaluation_content"] is False
    assert value["runtime_launch_authorized"] is False
    assert value["model_update_authorized"] is False
    assert value["checkpoint_write_authorized"] is False
    assert value["evaluation_authorized"] is False
    assert value["dataset_promotion_authorized"] is False
    tampered = copy.deepcopy(value)
    tampered["draw_arrays_persisted"] = True
    tampered["content_sha256_before_self_field"] = draw_v20a.canonical_sha256({
        key: item for key, item in tampered.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="certificate changed"):
        draw_v20a.validate_draw_plan_certificate_v20a(tampered)
