#!/usr/bin/env python3

import copy
import math

import pytest

import eggroll_es_hierarchical_preregistration_v14a as prereg_v14a
import eggroll_es_paired_distinct_row_sampler_v14b as sampler
import eggroll_es_train_panel_sampler_v13 as sampler_v13


@pytest.fixture(scope="module")
def frozen():
    rows, source_sha256 = sampler_v13.load_frozen_train()
    assert source_sha256 == sampler_v13.SOURCE_SHA256
    return rows


def test_v14b_k2_full_frame_is_exact_without_replacement(frozen):
    full = sampler.build_full_frame(frozen)
    assert sampler.validate_full_frame(full, frozen)
    assert full == sampler.build_full_frame(frozen)
    assert full["documents"] == 310
    assert full["single_row_documents"] == 139
    assert full["multirow_documents"] == 171
    assert full["prompts"] == 481
    assert sum(item["selected_row_count"] for item in full["items"]) == 481
    assert all(
        len({row["row_index"] for row in item["selected_rows"]})
        == item["selected_row_count"]
        == min(2, item["document_row_count"])
        for item in full["items"]
    )


def test_v14b_full_frame_identities_and_order_are_frozen(frozen):
    full = sampler.build_full_frame(frozen)
    assert full["frame_sha256"] == (
        "ce50531881f4b7044bf82fc3e8fd52d603ba53041fccc4b934aa307840862d6c"
    )
    assert full["ordered_document_identity_sha256"] == (
        "3574c320bee17d7df2f31b2cc35d0ee018e903702c1dc1c3dda2778c77e05c0f"
    )
    assert full["ordered_prompt_identity_sha256"] == (
        "b3e7c0eb24a04377fc7727cb1972fefdca016dd112c0076068f2527677866ba9"
    )
    assert full["content_sha256_before_self_field"] == (
        "9d9fc31e928948cae12d7dc4b5ffedfd9def8482a4e6af8e82dc7dcfce7cb3d4"
    )
    assert [
        row["prompt_position"]
        for item in full["items"] for row in item["selected_rows"]
    ] == list(range(481))


def test_v14b_reuses_the_exact_five_disjoint_v14_document_allocations(frozen):
    full = sampler.build_full_frame(frozen)
    panels = sampler.build_matched_panels(frozen, full)
    assert sampler.validate_matched_panels(panels, frozen, full)
    assert panels == sampler.build_matched_panels(frozen, full)
    _rows, _v14a_full, v14a_panels = prereg_v14a.materialize_panels_v14a()
    assert tuple(panels) == sampler.PANEL_NAMES
    assert all(panel["documents"] == 56 for panel in panels.values())
    assert [panels[name]["prompts"] for name in panels] == [92, 81, 87, 88, 86]
    allocated = [
        item["document_sha256"]
        for panel in panels.values() for item in panel["items"]
    ]
    assert len(allocated) == len(set(allocated)) == 280
    for name, panel in panels.items():
        assert [item["document_sha256"] for item in panel["items"]] == [
            item["document_sha256"] for item in v14a_panels[name]["items"]
        ]
        assert math.fsum(
            item["equal_document_ht_weight"] for item in panel["items"]
        ) == pytest.approx(310.0)


def test_v14b_screen_complements_are_exact_disjoint_254_document_sets(frozen):
    full = sampler.build_full_frame(frozen)
    panels = sampler.build_matched_panels(frozen, full)
    complements = sampler.build_screen_complements(full, panels)
    assert sampler.validate_screen_complements(complements, full, panels)
    assert complements == sampler.build_screen_complements(full, panels)
    assert tuple(complements) == sampler.PANEL_NAMES[3:]
    assert [complements[name]["prompts"] for name in complements] == [393, 395]
    all_documents = {item["document_sha256"] for item in full["items"]}
    for name, complement in complements.items():
        screen = {item["document_sha256"] for item in panels[name]["items"]}
        kept = {item["document_sha256"] for item in complement["items"]}
        assert complement["documents"] == len(kept) == 254
        assert screen.isdisjoint(kept)
        assert screen | kept == all_documents


def test_v14b_reduction_averages_rows_before_equal_document_aggregation(frozen):
    full = sampler.build_full_frame(frozen)
    prompt_values = [float(index) for index in range(481)]
    document_values = sampler.document_means(prompt_values, full)
    for item in full["items"]:
        positions = [row["prompt_position"] for row in item["selected_rows"]]
        assert document_values[item["document_sha256"]] == pytest.approx(
            math.fsum(prompt_values[position] for position in positions)
            / len(positions)
        )
    assert sampler.full_frame_mean(document_values, full) == pytest.approx(
        math.fsum(document_values.values()) / 310.0
    )
    panels = sampler.build_matched_panels(frozen, full)
    for panel in panels.values():
        expected = math.fsum(
            item["equal_document_ht_weight"]
            * document_values[item["document_sha256"]]
            for item in panel["items"]
        ) / 310.0
        assert sampler.matched_panel_mean(document_values, panel) == pytest.approx(
            expected
        )


def test_v14b_crn_schedule_has_one_frozen_481_prompt_order(frozen):
    full = sampler.build_full_frame(frozen)
    schedule = sampler.common_random_number_schedule(full, range(32))
    assert len(schedule) == 64
    assert {item["sign"] for item in schedule} == {"plus", "minus"}
    assert {item["prompt_count"] for item in schedule} == {481}
    assert {item["ordered_prompt_identity_sha256"] for item in schedule} == {
        full["ordered_prompt_identity_sha256"]
    }
    assert [
        (item["wave_index"], item["sign"], item["engine_index"])
        for item in schedule[:8]
    ] == [
        (0, "plus", 0), (0, "plus", 1),
        (0, "plus", 2), (0, "plus", 3),
        (0, "minus", 0), (0, "minus", 1),
        (0, "minus", 2), (0, "minus", 3),
    ]
    assert [item["execution_index"] for item in schedule] == list(range(64))


def test_v14b_bad_counts_nonfinite_rewards_and_tampering_fail_closed(frozen):
    full = sampler.build_full_frame(frozen)
    with pytest.raises(ValueError, match="count"):
        sampler.document_means([0.0] * 480, full)
    with pytest.raises(ValueError, match="non-finite"):
        sampler.document_means([math.nan] * 481, full)
    tampered = copy.deepcopy(full)
    tampered["items"][0]["selected_rows"][0]["row_index"] = (
        tampered["items"][1]["selected_rows"][0]["row_index"]
    )
    with pytest.raises(ValueError, match="content hash"):
        sampler.validate_full_frame(tampered, frozen)

    reordered = copy.deepcopy(full)
    multi = next(
        item for item in reordered["items"] if item["selected_row_count"] == 2
    )
    multi["selected_rows"].reverse()
    for rank, selected in enumerate(multi["selected_rows"]):
        selected["selection_rank"] = rank
    reordered["content_sha256_before_self_field"] = sampler._canonical({
        key: value for key, value in reordered.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(ValueError, match="without-replacement"):
        sampler.validate_full_frame(reordered, frozen)

    changed_frame = copy.deepcopy(full)
    changed_frame["frame_sha256"] = "0" * 64
    changed_frame["content_sha256_before_self_field"] = sampler._canonical({
        key: value for key, value in changed_frame.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(ValueError, match="allocation"):
        sampler.validate_full_frame(changed_frame, frozen)

    panels = sampler.build_matched_panels(frozen, full)
    tampered_panels = copy.deepcopy(panels)
    tampered_panels["optimization_0"]["items"][0][
        "equal_document_ht_weight"
    ] += 1.0
    tampered_panels["optimization_0"][
        "content_sha256_before_self_field"
    ] = sampler._canonical({
        key: value for key, value in tampered_panels["optimization_0"].items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(ValueError, match="allocation"):
        sampler.validate_matched_panels(tampered_panels, frozen, full)
    complements = sampler.build_screen_complements(full, panels)
    tampered_complements = copy.deepcopy(complements)
    tampered_complements["train_screen_0"]["items"][0][
        "document_sha256"
    ] = next(iter({
        item["document_sha256"] for item in panels["train_screen_0"]["items"]
    }))
    tampered_complements["train_screen_0"][
        "content_sha256_before_self_field"
    ] = sampler._canonical({
        key: value
        for key, value in tampered_complements["train_screen_0"].items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(ValueError, match="changed"):
        sampler.validate_screen_complements(
            tampered_complements, full, panels,
        )
