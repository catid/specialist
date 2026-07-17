from __future__ import annotations

import copy
import math

import pytest

import eggroll_es_mirrored_v66 as mirrored_v66
import lora_es_robust_consensus_v43g as robust_v43g
import reward_shaping_ablation_v1 as shaping


DIRECTION_SEEDS = (101, 202, 303, 404)
CONTRACT_SHA = "a" * 64


def population_records(
    *,
    training_seed=1701,
    generation_index=0,
    population_id="population-a",
    prompt_ids=("prompt-a", "prompt-b"),
    reward_fn=None,
):
    if reward_fn is None:
        reward_fn = lambda prompt, direction, sign: (
            10.0
            + prompt * 0.5
            + direction
            + sign * (prompt + 1) * (direction + 1) * 0.05
        )
    records = []
    for prompt_index, prompt_id in enumerate(prompt_ids):
        for direction_index, direction_seed in enumerate(DIRECTION_SEEDS):
            for sign in (1, -1):
                records.append({
                    "dataset_role": "train",
                    "training_seed": training_seed,
                    "generation_index": generation_index,
                    "population_id": population_id,
                    "evaluation_contract_sha256": CONTRACT_SHA,
                    "prompt_group_id": prompt_id,
                    "repeat_index": 0,
                    "evaluation_seed": 9000 + prompt_index,
                    "direction_index": direction_index,
                    "direction_seed": direction_seed,
                    "sign": sign,
                    "reward": reward_fn(prompt_index, direction_index, sign),
                })
    return records


def shape(records, method):
    return shaping.shape_reward_population_v1(
        records, DIRECTION_SEEDS, CONTRACT_SHA, method
    )


def test_raw_rewards_and_direct_pair_differences_are_exactly_same_estimator():
    records = population_records()
    comparison = shaping.compare_reward_shaping_v1(
        records, DIRECTION_SEEDS, CONTRACT_SHA
    )
    raw = comparison["arms"]["raw_rewards"]
    paired = comparison["arms"]["antithetic_pair_difference"]
    assert raw["direction_coefficients"] == pytest.approx(
        [0.15, 0.30, 0.45, 0.60]
    )
    assert raw["direction_coefficients"] == paired["direction_coefficients"]
    assert comparison["raw_and_direct_pair_coefficients_exactly_equal"] is True
    assert (
        comparison["raw_and_direct_pair_are_one_gradient_estimator_not_two_independent_arms"]
        is True
    )


def test_centered_rank_matches_existing_v43g_midrank_semantics():
    values = [1.0, 1.0, 2.0, 4.0, 4.0, 9.0]
    assert shaping.centered_ranks_v1(values) == robust_v43g.centered_ranks_v43g(
        values
    )
    assert shaping.centered_ranks_v1([3.0] * 8) == [0.0] * 8
    ranks = shaping.centered_ranks_v1(values)
    assert ranks[0] == ranks[1]
    assert ranks[3] == ranks[4]


def test_zscores_are_prompt_local_not_pooled_across_different_scales():
    first = []
    for direction in range(4):
        for sign in (1, -1):
            first.append(direction * 2.0 + sign * (direction + 1) * 0.25)

    def reward_fn(prompt, direction, sign):
        value = direction * 2.0 + sign * (direction + 1) * 0.25
        return value if prompt == 0 else 1_000.0 * value + 777.0

    records = population_records(reward_fn=reward_fn)
    result = shape(records, "within_prompt_centered_zscore")
    group_coefficients = [
        [pair["coefficient"] for pair in group["pairs"]]
        for group in result["prompt_groups"]
    ]
    # The +1e-8 current-recipe guard makes scaled groups differ only by its
    # expected tiny scale effect, not by a pooled mean or variance.
    assert group_coefficients[0] == pytest.approx(
        group_coefficients[1], rel=2e-8, abs=2e-8
    )
    assert result["statistics_never_pool_across_prompt_groups"] is True


def test_exact_zero_spread_produces_zero_zscores_and_zero_midrank_update():
    records = population_records(reward_fn=lambda *_: 4.25)
    zscore = shape(records, "within_prompt_centered_zscore")
    ranks = shape(records, "within_prompt_centered_rank")
    assert zscore["zero_update"] is True
    assert ranks["zero_update"] is True
    assert zscore["direction_coefficients"] == [0.0] * 4
    assert ranks["direction_coefficients"] == [0.0] * 4
    assert all(
        group["statistics"]["zero_spread"] is True
        for group in zscore["prompt_groups"] + ranks["prompt_groups"]
    )


@pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
def test_nonfinite_reward_fails_before_any_statistic(nonfinite):
    records = population_records()
    records[3]["reward"] = nonfinite
    with pytest.raises(ValueError, match="reward must be finite"):
        shape(records, "within_prompt_centered_zscore")
    with pytest.raises(ValueError, match="reward must be finite"):
        shape(records, "within_prompt_centered_rank")


def test_finite_inputs_that_overflow_pair_or_variance_fail_closed():
    records = population_records()
    records[0]["reward"] = 1e308
    records[1]["reward"] = -1e308
    with pytest.raises(RuntimeError, match="coefficient overflowed"):
        shape(records, "raw_rewards")
    with pytest.raises(RuntimeError, match="variance overflowed"):
        shape(records, "within_prompt_centered_zscore")


def test_cosine_is_stable_for_extreme_finite_coefficients():
    left = [1e308, -1e308, 5e307]
    assert shaping.coefficient_cosine_v1(left, left) == pytest.approx(1.0)
    assert shaping.coefficient_cosine_v1(left, [-value for value in left]) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("dataset_role", "dev", "train records only"),
        ("training_seed", 1702, "crossed a role/seed/generation/population"),
        ("generation_index", 1, "crossed a role/seed/generation/population"),
        ("population_id", "population-b", "crossed a role/seed/generation/population"),
    ],
)
def test_cross_boundary_population_mixing_fails_closed(field, replacement, match):
    records = population_records()
    records[0][field] = replacement
    with pytest.raises(ValueError, match=match):
        shape(records, "within_prompt_centered_rank")


def test_extra_semantic_or_holdout_payload_field_is_rejected():
    records = population_records()
    records[0]["question"] = "protected semantics must not enter this path"
    with pytest.raises(ValueError, match="semantic or holdout payloads are forbidden"):
        shape(records, "raw_rewards")


def test_missing_duplicate_and_misidentified_candidate_fail_closed():
    records = population_records()
    with pytest.raises(RuntimeError, match="complete mirrored population"):
        shape(records[:-1], "raw_rewards")

    duplicate = records + [copy.deepcopy(records[0])]
    with pytest.raises(ValueError, match="duplicate candidate"):
        shape(duplicate, "raw_rewards")

    wrong_seed = copy.deepcopy(records)
    wrong_seed[0]["direction_seed"] += 1
    with pytest.raises(RuntimeError, match="direction seed identity changed"):
        shape(wrong_seed, "raw_rewards")


def test_record_order_cannot_change_grouping_or_coefficients():
    records = population_records()
    forward = shape(records, "within_prompt_centered_rank")
    reverse = shape(list(reversed(records)), "within_prompt_centered_rank")
    assert forward == reverse


def test_one_outlier_cannot_contaminate_any_other_prompt_group():
    clean = population_records(prompt_ids=("prompt-a", "prompt-b", "prompt-c"))
    contaminated = copy.deepcopy(clean)
    contaminated[0]["reward"] = 1e12
    result = shaping.outlier_sensitivity_v1(
        clean, contaminated, DIRECTION_SEEDS, CONTRACT_SHA
    )
    for diagnostic in result["method_diagnostics"].values():
        assert diagnostic["unchanged_prompt_group_count"] == 2
        assert diagnostic["only_contaminated_prompt_group_changed"] is True
    raw_delta = result["method_diagnostics"]["raw_rewards"][
        "coefficient_l2_delta"
    ]
    rank_delta = result["method_diagnostics"]["within_prompt_centered_rank"][
        "coefficient_l2_delta"
    ]
    z_delta = result["method_diagnostics"]["within_prompt_centered_zscore"][
        "coefficient_l2_delta"
    ]
    assert raw_delta > 1e11
    assert rank_delta < 2.0
    assert z_delta < 3.0
    assert result["raw_and_direct_pair_remain_exactly_equivalent"] is True


def test_outlier_diagnostic_rejects_more_than_one_changed_reward():
    clean = population_records()
    contaminated = copy.deepcopy(clean)
    contaminated[0]["reward"] += 1.0
    contaminated[1]["reward"] += 1.0
    with pytest.raises(ValueError, match="exactly one changed reward"):
        shaping.outlier_sensitivity_v1(
            clean, contaminated, DIRECTION_SEEDS, CONTRACT_SHA
        )


def test_multi_seed_stability_checks_seals_seed_labels_and_shared_arm():
    outputs = {}
    for seed in (1701, 1702, 1703):
        outputs[seed] = shape(
            population_records(
                training_seed=seed,
                population_id=f"population-{seed}",
                reward_fn=lambda prompt, direction, sign, seed=seed: (
                    direction + sign * (direction + 1) * (1 + seed % 3) / 10
                ),
            ),
            "within_prompt_centered_rank",
        )
    result = shaping.multi_seed_stability_v1(outputs)
    assert result["training_seeds"] == [1701, 1702, 1703]
    assert len(result["pairwise"]) == 3
    assert result["zero_vector_pair_count"] == 0

    mislabeled = dict(outputs)
    mislabeled[9999] = mislabeled.pop(1701)
    with pytest.raises(ValueError, match="seed label changed"):
        shaping.multi_seed_stability_v1(mislabeled)

    tampered = copy.deepcopy(outputs)
    tampered[1701]["direction_coefficients"][0] += 1.0
    with pytest.raises(RuntimeError, match="output identity changed"):
        shaping.multi_seed_stability_v1(tampered)


def test_direct_pair_output_matches_current_v66_central_difference_reduction():
    payload = mirrored_v66.common_evaluation_payload_v66(
        [{"opaque_prompt_id": "prompt-a"}],
        {"temperature": 0.0, "max_tokens": 1},
        {"schema": "train-only-reward-v1"},
        9000,
    )
    plan = mirrored_v66.mirrored_population_plan_v66(
        DIRECTION_SEEDS, 0.0006, payload
    )
    records = population_records(prompt_ids=("prompt-a",))
    reward_lookup = {
        (record["direction_index"], record["sign"]): record["reward"]
        for record in records
    }
    signed_rewards = []
    for wave in plan["waves"]:
        for assignment in wave:
            signed_rewards.append({
                "pair_id": assignment["pair_id"],
                "sign": assignment["sign"],
                "direction_index": assignment["direction_index"],
                "direction_seed": assignment["direction_seed"],
                "evaluation_contract_sha256": assignment[
                    "evaluation_contract_sha256"
                ],
                "reward": reward_lookup[
                    (assignment["direction_index"], assignment["sign"])
                ],
            })
    v66 = mirrored_v66.pair_difference_update_v66(
        plan, signed_rewards, learning_rate=0.00015
    )
    direct = shape(records, "antithetic_pair_difference")
    assert direct["direction_coefficients"] == v66["coefficients"]
