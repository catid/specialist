#!/usr/bin/env python3

import hashlib

import pytest

import lora_es_generation_boundary_sampling_v48a as subject


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _synthetic_inputs():
    rows, memberships, evidence = [], [], []
    for unit_index in range(subject.TRAIN_CONFLICT_UNITS_V48A):
        count = 3 if unit_index < 32 else 2
        unit = _sha(f"unit-{unit_index}")
        kind = unit_index % 4
        for row_index in range(count):
            row_sha = _sha(f"row-{unit_index}-{row_index}")
            rows.append(row_sha)
            memberships.append({
                "row_sha256": row_sha,
                "unit_identity_sha256": unit,
                "row_count": count,
            })
            actors = []
            for actor in range(4):
                if kind == 0:
                    f1 = 0.4 if actor < 2 else 0.6
                    exact, nonzero = 0, 1
                    prediction = _sha(f"unstable-{actor}")
                elif kind == 1:
                    f1, exact, nonzero = 0.5, 0, 1
                    prediction = _sha("partial")
                elif kind == 2:
                    f1, exact, nonzero = 1.0, 1, 1
                    prediction = _sha("exact")
                else:
                    f1, exact, nonzero = 0.0, 0, 0
                    prediction = _sha("zero")
                actors.append({
                    "actor_rank": actor, "f1": f1, "exact": exact,
                    "nonzero": nonzero, "prediction_sha256": prediction,
                })
            evidence.append({"row_sha256": row_sha, "actors": actors})
    assert len(rows) == subject.TRAIN_ROWS_V48A
    bundle = {
        "row_sha256": rows,
        "unit_membership_v48a": memberships,
        "train_bundle_content_sha256": _sha("train-bundle"),
    }
    base = {
        "schema": "train-only-four-actor-base-generation-evidence-v48a",
        "generation_params": dict(subject.GENERATION_PARAMS_V48A),
        "train_bundle_content_sha256": _sha("train-bundle"),
        "matched_master_sha256": _sha("master"),
        "rows": evidence,
        "protected_semantics_opened": False,
        "shadow_ood_holdout_or_benchmark_opened": False,
    }
    return bundle, base


def _population(order):
    return {
        "plus": [[float(order[d]) + 0.001 * actor for actor in range(4)]
                 for d in range(8)],
        "minus": [[-float(order[d]) + 0.001 * actor for actor in range(4)]
                  for d in range(8)],
    }


def test_subset_is_deterministic_equal_unit_and_not_oversampled_v48a():
    bundle, base = _synthetic_inputs()
    first = subject.build_fragile_subset_v48a(bundle, base)
    permutation = list(reversed(range(subject.TRAIN_ROWS_V48A)))
    shuffled_bundle = {
        **bundle,
        "row_sha256": [bundle["row_sha256"][i] for i in permutation],
        "unit_membership_v48a": [
            bundle["unit_membership_v48a"][i] for i in permutation
        ],
    }
    shuffled_base = {**base, "rows": list(reversed(base["rows"]))}
    second = subject.build_fragile_subset_v48a(shuffled_bundle, shuffled_base)
    assert first == second
    assert first["selected_rows"] == 64
    assert first["selected_conflict_units"] == 64
    assert len({item["unit_identity_sha256"] for item in first["items"]}) == 64
    assert {item["equal_conflict_unit_weight"] for item in first["items"]} == {
        1 / 64
    }
    assert first["stratum_counts"] == {
        "unstable": 22, "partial": 22, "exact": 10, "zero": 10,
    }
    assert first["teacher_forced_domain_sampling_changed"] is False
    assert first["rows_duplicated_or_oversampled_in_domain_objective"] is False


def test_common_random_plan_covers_every_signed_actor_state_v48a():
    subset = subject.build_fragile_subset_v48a(*_synthetic_inputs())
    receipts = [{
        "direction": direction, "sign": sign, "actor_rank": actor,
        "subset_content_sha256": subset["content_sha256_before_self_field"],
        "request_order_sha256": subset["request_order_sha256"],
        "generation_params": dict(subject.GENERATION_PARAMS_V48A),
    } for direction in range(8) for sign in ("plus", "minus")
      for actor in range(4)]
    result = subject.assert_common_random_plan_v48a(receipts, subset)
    assert result["signed_actor_state_receipts"] == 64
    assert result["all_use_identical_selected_items_order_and_sampling"] is True
    receipts[-1]["request_order_sha256"] = _sha("wrong")
    with pytest.raises(RuntimeError, match="common-random"):
        subject.assert_common_random_plan_v48a(receipts, subset)


def test_fragile_score_is_equal_unit_direct_generated_f1_v48a():
    subset = subject.build_fragile_subset_v48a(*_synthetic_inputs())
    metrics = [{
        "row_sha256": item["row_sha256"],
        "f1": index / 63, "exact": int(index == 63),
        "nonzero": int(index > 0), "prediction_sha256": _sha(f"p-{index}"),
    } for index, item in enumerate(subset["items"])]
    score = subject.score_fragile_items_v48a(subset, metrics)
    assert score["equal_conflict_unit_mean_f1"] == pytest.approx(0.5)
    assert score["selected_conflict_units"] == 64
    assert score["exact_count"] == 1
    assert score["nonzero_count"] == 63


def test_generated_f1_is_primary_and_domain_is_also_constrained_v48a():
    result = subject.direct_generation_objective_v48a(
        _population([1, 2, 3, 4, 5, 6, 7, 8]),
        _population([8, 1, 7, 2, 6, 3, 5, 4]),
        _population([1, 3, 2, 4, 6, 5, 8, 7]),
        _population([2, 1, 4, 3, 5, 7, 6, 8]),
    )
    assert result["fragile_generated_f1_is_primary_not_halfspace_only"] is True
    assert result["projection_anchors"] == [
        "domain", "fragile_generation_f1", "prose_lm",
        "qa_answer_logprob",
    ]
    assert result["projection"]["diagnostics"][
        "all_anchor_halfspaces_satisfied"
    ] is True
    assert result["projection"]["diagnostics"][
        "anchor_directional_derivatives_after"
    ]["fragile_generation_f1"] >= -subject.multi_anchor.FEASIBILITY_TOLERANCE_V43H


def test_zero_generated_f1_spread_fails_closed_v48a():
    flat = {"plus": [[1.0] * 4 for _ in range(8)],
            "minus": [[1.0] * 4 for _ in range(8)]}
    with pytest.raises(RuntimeError, match="generated-F1 objective has zero"):
        subject.direct_generation_objective_v48a(
            _population(range(8)), flat,
            _population([1, 3, 2, 4, 6, 5, 8, 7]),
            _population([2, 1, 4, 3, 5, 7, 6, 8]),
        )


def test_protected_runtime_paths_are_rejected_v48a():
    subject.validate_runtime_paths_v48a([
        "/train/fold_3_train.jsonl", "/train/unit_membership.json",
    ])
    for token in subject.FORBIDDEN_RUNTIME_PATH_TOKENS_V48A:
        with pytest.raises(ValueError, match="protected runtime path"):
            subject.validate_runtime_paths_v48a([f"/data/{token}.jsonl"])


def test_malformed_actor_or_membership_evidence_fails_closed_v48a():
    bundle, base = _synthetic_inputs()
    base["rows"][0]["actors"].pop()
    with pytest.raises(ValueError, match="actor or row coverage"):
        subject.build_fragile_subset_v48a(bundle, base)
