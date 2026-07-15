#!/usr/bin/env python3
"""Offline tests for exact nested-arm request unioning."""

import copy

import pytest

import eggroll_es_union_request_plan_v20a as union_v20a


ARMS = ("production_only", "tier_2", "tiers_2_3", "all_tiers")
PANELS = ("optimization_0", "train_screen_0")


def _item(value, example_index):
    tokens = [1, int(value), 2]
    return {
        "example_index": example_index,
        "prompt_sha256": f"{value + 10:064x}",
        "answer_sha256": f"{value + 20:064x}",
        "prompt_token_count": 2,
        "answer_token_start": 2,
        "answer_token_count": 1,
        "prompt_token_ids": tokens,
        "prompt_token_ids_sha256": union_v20a.canonical_sha256(tokens),
        "eos_appended": False,
    }


def _prepared():
    values = {
        "production_only": ((10, 11, 12), (20, 21, 22)),
        "tier_2": ((10, 13, 12, 30), (20, 23, 22, 31)),
        "tiers_2_3": ((10, 13, 14, 30, 32), (20, 23, 24, 31, 33)),
        "all_tiers": ((15, 13, 14, 30, 32, 34), (25, 23, 24, 31, 33, 35)),
    }
    prepared = {}
    for arm in ARMS:
        panels = {}
        flat = []
        cursor = 0
        for panel, panel_values in zip(PANELS, values[arm]):
            items = [_item(value, index) for index, value in enumerate(panel_values)]
            panels[panel] = {
                "dense_items": items,
                "slice": (cursor, cursor + len(items)),
            }
            flat.extend(items)
            cursor += len(items)
        prepared[arm] = {"panels": panels, "prompt_items": flat}
    return prepared


def test_v20a_union_plan_is_lossless_compact_and_deterministic():
    prepared = _prepared()
    runtime, audit = union_v20a.build_union_request_plan_v20a(
        prepared, ARMS, PANELS
    )
    rebuilt_runtime, rebuilt_audit = union_v20a.build_union_request_plan_v20a(
        copy.deepcopy(prepared), ARMS, PANELS
    )
    assert rebuilt_runtime == runtime
    assert rebuilt_audit == audit
    assert audit["raw_request_count"] == 36
    assert audit["unique_request_count"] == 18
    assert audit["eliminated_duplicate_request_count"] == 18
    assert audit["contains_token_ids_or_row_content"] is False
    assert "prompt_token_ids" not in audit
    assert union_v20a.validate_union_request_plan_v20a(
        runtime, audit, prepared
    ) == audit


def test_v20a_union_outputs_reconstruct_every_arm_and_panel_order():
    prepared = _prepared()
    runtime, _audit = union_v20a.build_union_request_plan_v20a(
        prepared, ARMS, PANELS
    )
    outputs = [f"output-{index}" for index in range(len(runtime["union_prompt_items"]))]
    for arm in ARMS:
        rebuilt = union_v20a.reconstruct_arm_outputs_v20a(
            runtime, arm, PANELS, outputs
        )
        for panel in PANELS:
            expected_tokens = [
                item["prompt_token_ids"]
                for item in prepared[arm]["panels"][panel]["dense_items"]
            ]
            actual_tokens = [
                runtime["union_prompt_items"][index]["prompt_token_ids"]
                for index in runtime["arm_panel_union_indices"][arm][panel]
            ]
            assert actual_tokens == expected_tokens
            assert len(rebuilt[panel]) == len(expected_tokens)


def test_v20a_union_plan_rejects_hash_collision_slice_and_mapping_tampering():
    prepared = _prepared()
    collision = copy.deepcopy(prepared)
    item = collision["all_tiers"]["panels"]["optimization_0"]["dense_items"][-1]
    item["prompt_token_ids_sha256"] = collision["production_only"]["panels"][
        "optimization_0"
    ]["dense_items"][0]["prompt_token_ids_sha256"]
    collision["all_tiers"]["prompt_items"][-12] = item
    with pytest.raises(ValueError, match="token identity changed"):
        union_v20a.build_union_request_plan_v20a(collision, ARMS, PANELS)

    malformed = _prepared()
    malformed["tier_2"]["panels"]["optimization_0"]["slice"] = (0, 99)
    with pytest.raises(ValueError, match="panel slice changed"):
        union_v20a.build_union_request_plan_v20a(malformed, ARMS, PANELS)

    prepared = _prepared()
    runtime, audit = union_v20a.build_union_request_plan_v20a(
        prepared, ARMS, PANELS
    )
    runtime["arm_panel_union_indices"]["tier_2"]["optimization_0"][0] = 999
    with pytest.raises(RuntimeError, match="identities changed|reconstruction failed"):
        union_v20a.validate_union_request_plan_v20a(runtime, audit, prepared)


def test_v20a_v19_train_only_materialization_has_exact_440_request_union():
    import train_eggroll_es_disjoint_tier_attribution_v19a as mechanics_v19a

    bundle = mechanics_v19a.load_panel_bundle_v19a()
    prepared = {}
    for arm in mechanics_v19a.ARMS_V19A:
        panels = {}
        flat = []
        cursor = 0
        for panel in mechanics_v19a.PANEL_NAMES_V19A:
            row_hashes = bundle["panels"][panel]["arms"][arm]["row_sha256"]
            items = []
            for index, digest in enumerate(row_hashes):
                tokens = [int(digest[:8], 16), int(digest[8:16], 16)]
                items.append({
                    "prompt_token_ids": tokens,
                    "prompt_token_ids_sha256": union_v20a.canonical_sha256(tokens),
                })
            panels[panel] = {
                "dense_items": items,
                "slice": (cursor, cursor + len(items)),
            }
            flat.extend(items)
            cursor += len(items)
        prepared[arm] = {"panels": panels, "prompt_items": flat}
    _runtime, audit = union_v20a.build_union_request_plan_v20a(
        prepared, mechanics_v19a.ARMS_V19A, mechanics_v19a.PANEL_NAMES_V19A
    )
    assert audit["raw_request_count"] == 990
    # The per-panel union is 447 requests; seven candidate-only requests are
    # also identical across panels, so the global signed-wave union is 440.
    assert sum(audit["per_panel_unique_request_count"].values()) == 447
    assert audit["unique_request_count"] == 440
    assert audit["eliminated_duplicate_request_count"] == 550
    assert set(audit["per_panel_unique_request_count"].values()) == {44, 45}
