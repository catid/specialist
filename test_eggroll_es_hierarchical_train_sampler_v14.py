#!/usr/bin/env python3

import copy
import math

import pytest

import eggroll_es_hierarchical_train_sampler_v14 as sampler
import eggroll_es_train_panel_sampler_v13 as v13


@pytest.fixture(scope="module")
def frozen():
    rows, _ = v13.load_frozen_train()
    return rows


def test_frozen_frame_keeps_rows_but_balances_documents(frozen):
    documents, frame = sampler.build_document_frame(frozen)
    assert frame["rows"] == 794
    assert frame["documents"] == len(documents) == 310
    assert frame["maximum_rows_per_document"] == 27
    assert sum(item["row_count"] for item in documents) == 794


def test_iteration_panel_is_exact_rebuild_and_crn(frozen):
    panel = sampler.build_iteration_panel(frozen, 0)
    assert sampler.validate_panel(panel, frozen)
    assert panel == sampler.build_iteration_panel(frozen, 0)
    assert panel["rows"] == 56
    assert len({item["document_sha256"] for item in panel["items"]}) == 56
    schedule = sampler.common_random_number_schedule(panel, range(32))
    assert len(schedule) == 64
    assert {row["sign"] for row in schedule} == {"plus", "minus"}
    assert len({row["ordered_row_identity_sha256"] for row in schedule}) == 1


def test_adjacent_panels_rotate_documents_and_rows(frozen):
    panels = [sampler.build_iteration_panel(frozen, index) for index in range(5)]
    for stratum in sampler.STRATA:
        selected = [
            item["document_sha256"]
            for panel in panels
            for item in panel["items"] if item["stratum"] == stratum
        ]
        assert len(selected) == len(set(selected))
    documents, _ = sampler.build_document_frame(frozen)
    multi = next(item for item in documents if item["row_count"] >= 3)
    chosen = {
        sampler._choose_row(multi, frozen, iteration)
        for iteration in range(24)
    }
    assert len(chosen) >= 2


def test_probabilities_and_ht_weights_target_equal_documents(frozen):
    panel = sampler.build_iteration_panel(frozen, 7)
    for item in panel["items"]:
        assert item["within_document_row_selection_probability"] == pytest.approx(
            1.0 / item["document_row_count"]
        )
        assert item["joint_row_selection_probability"] == pytest.approx(
            item["document_selection_probability"]
            / item["document_row_count"]
        )
        assert item["equal_document_ht_weight"] == pytest.approx(
            1.0 / item["document_selection_probability"]
        )
    assert sum(
        item["equal_document_ht_weight"] for item in panel["items"]
    ) == pytest.approx(310.0)
    assert sampler.equal_document_mean([1.0] * 56, panel) == 1.0


def test_ht_mean_uses_document_not_row_count_weight(frozen):
    panel = sampler.build_iteration_panel(frozen, 3)
    values = [float(item["document_row_count"]) for item in panel["items"]]
    expected = sum(
        item["equal_document_ht_weight"] * value
        for item, value in zip(panel["items"], values)
    ) / 310.0
    assert sampler.equal_document_mean(values, panel) == pytest.approx(expected)
    with pytest.raises(ValueError, match="count"):
        sampler.equal_document_mean(values[:-1], panel)
    with pytest.raises(ValueError, match="non-finite"):
        sampler.equal_document_mean([math.nan] * 56, panel)


def test_tampering_and_bad_iterations_fail_closed(frozen):
    panel = sampler.build_iteration_panel(frozen, 1)
    tampered = copy.deepcopy(panel)
    tampered["items"][0]["row_index"] = tampered["items"][1]["row_index"]
    with pytest.raises(ValueError, match="content hash"):
        sampler.validate_panel(tampered, frozen)
    for value in (-1, True, 1.5):
        with pytest.raises(ValueError, match="iteration"):
            sampler.build_iteration_panel(frozen, value)


def test_policy_disables_adaptive_replay_and_eval_selection(frozen):
    policy = sampler.build_policy(frozen)
    assert policy["hard_replay"] == {
        "enabled": False,
        "fraction": 0.0,
        "maximum_fraction": 0.25,
        "enable_only_with": (
            "lagged out-of-fold train-only difficulty and exact propensities"
        ),
    }
    assert policy["selection_firewall"]["forbidden"] == [
        "validation", "OOD", "heldout", "benchmark outcomes",
    ]


def test_materialization_reads_only_frozen_train():
    result = sampler.materialize_iteration(2)
    assert len(result["questions"]) == len(result["answers"]) == 56
    assert len(result["weights"]) == 56
    assert result["source_sha256"] == v13.SOURCE_SHA256
