#!/usr/bin/env python3

from __future__ import annotations

import copy
import json

import pytest

import eggroll_es_decode_robustness_v68 as subject
import eggroll_es_multiobjective_trust_region_v67 as trust_v67


CANDIDATE_A = "a" * 64
CANDIDATE_B = "b" * 64
BLOCKS = [
    {"role": "train", "prompt_block_sha256": "c" * 64, "prompt_count": 2},
    {"role": "dev", "prompt_block_sha256": "d" * 64, "prompt_count": 3},
    {"role": "ood_qa", "prompt_block_sha256": "e" * 64, "prompt_count": 1},
]


def _plan() -> dict:
    return subject.build_ablation_plan([CANDIDATE_B, CANDIDATE_A], BLOCKS)


def _reward(assignment: dict) -> float:
    return 0.8 if assignment["candidate_checkpoint_sha256"] == CANDIDATE_A else 0.6


def _cell(assignment: dict, reward: float | None = None) -> dict:
    if reward is None:
        reward = _reward(assignment)
    output_identity = {
        "candidate": assignment["candidate_checkpoint_sha256"],
        "training_seed": assignment["training_seed"],
        "mode": assignment["decode_mode"],
        "role": assignment["role"],
    }
    if assignment["decode_mode"] == "seeded_stochastic":
        output_identity["repeat"] = assignment["repeat_index"]
    output_sha = subject.canonical_sha256(output_identity)
    value = {
        "schema": "specialist-decode-cell-receipt-v68",
        "assignment_id": assignment["assignment_id"],
        "candidate_checkpoint_sha256": assignment[
            "candidate_checkpoint_sha256"
        ],
        "training_seed": assignment["training_seed"],
        "decode_mode": assignment["decode_mode"],
        "repeat_index": assignment["repeat_index"],
        "role": assignment["role"],
        "prompt_block_sha256": assignment["prompt_block_sha256"],
        "evaluation_payload_sha256": assignment["evaluation_payload_sha256"],
        "decode_seed": assignment["decode_seed"],
        "scheduled_completions": assignment["scheduled_completions"],
        "scheduled_generated_tokens": assignment["scheduled_generated_tokens"],
        "realized_completions": assignment["scheduled_completions"],
        "realized_generated_tokens": assignment["scheduled_generated_tokens"],
        "length_finish_count": assignment["scheduled_completions"],
        "early_stop_count": 0,
        "output_inventory_sha256": output_sha,
        "judge_metrics_sha256": subject.canonical_sha256({
            "output": output_sha,
            "judge": assignment["judge_contract_sha256"],
        }),
        "unit_balanced_reward": reward,
        "all_item_hard_gates_passed": True,
        "wall_seconds": 1.0,
        "raw_output_access_count": 0,
        "raw_output_persisted": False,
        "protected_access_count": 0,
        "protected_source_opened": False,
    }
    value["content_sha256_before_self_field"] = subject.canonical_sha256(value)
    return value


def _cells(plan: dict) -> list[dict]:
    return [_cell(assignment) for assignment in plan["assignments"]]


def _trust_evidence() -> dict:
    role = {
        "weighted_unit_mean_delta": 0.01,
        "weighted_unit_mean_delta_95_lcb": 0.0,
        "component_delta_95_lcb": {
            name: 0.0 for name in trust_v67.COMPONENT_ORDER
        },
        "candidate_hard_event_counts": {
            name: 0 for name in trust_v67.HARD_EVENT_ORDER
        },
    }
    return {
        "schema": "specialist-train-dev-ood-evidence-v67",
        "contract_content_sha256": (
            trust_v67.EXPECTED_EVALUATION_CONTRACT_CONTENT_SHA256
        ),
        "protected_access_count": 0,
        "protected_source_opened": False,
        "roles_evaluated": list(trust_v67.SAFE_EVALUATION_ROLES),
        "train": copy.deepcopy(role),
        "dev": copy.deepcopy(role),
        "ood": {
            "qa_mean_reward_delta_95_lcb": 0.0,
            "qa_exact_count_delta": 0,
            "prose_mean_token_logprob_delta_95_lcb": 0.0,
        },
    }


def _trust_bindings(*, fail_candidate=None, fail_seed=None) -> list[dict]:
    result = []
    for candidate in (CANDIDATE_A, CANDIDATE_B):
        for seed in subject.TRAINING_SEEDS:
            evidence = _trust_evidence()
            if candidate == fail_candidate and seed == fail_seed:
                evidence["ood"]["qa_exact_count_delta"] = -2
            receipt = trust_v67.evaluate_trust_region(evidence)
            result.append(subject.bind_trust_receipt(candidate, seed, receipt))
    return result


def _rehash(value: dict) -> None:
    value.pop("content_sha256_before_self_field", None)
    value["content_sha256_before_self_field"] = subject.canonical_sha256(value)


def _find_cell(cells: list[dict], **matches) -> dict:
    return next(
        cell for cell in cells
        if all(cell[key] == value for key, value in matches.items())
    )


def test_plan_pairs_exact_candidate_independent_payloads_and_budgets():
    plan = _plan()
    subject.validate_ablation_plan(plan)
    assert plan["candidate_checkpoint_sha256s"] == [CANDIDATE_A, CANDIDATE_B]
    assert plan["training_seeds"] == [1701, 1702, 1703]
    assert len(plan["assignments"]) == 144
    grouped = {}
    for item in plan["assignments"]:
        key = (
            item["training_seed"], item["decode_mode"], item["repeat_index"],
            item["role"],
        )
        grouped.setdefault(key, []).append(item)
    assert all(len(items) == 2 for items in grouped.values())
    assert all(
        len({item["decode_seed"] for item in items}) == 1
        and len({item["decode_contract_sha256"] for item in items}) == 1
        and len({item["evaluation_payload_sha256"] for item in items}) == 1
        and len({item["judge_contract_sha256"] for item in items}) == 1
        for items in grouped.values()
    )
    budget_hashes = {
        subject.canonical_sha256(budget)
        for modes in plan["candidate_mode_budgets"].values()
        for budget in modes.values()
    }
    assert len(budget_hashes) == 1
    assert plan["raw_prompt_or_output_access_count"] == 0
    assert plan["protected_access_count"] == 0


def test_decode_payloads_freeze_fixed_length_and_mode_only_differences():
    greedy = subject.decode_contract("greedy", 123)
    stochastic = subject.decode_contract("seeded_stochastic", 123)
    differing = {
        key for key in greedy if greedy[key] != stochastic[key]
    }
    assert differing == {"temperature", "top_p", "top_k"}
    for payload in (greedy, stochastic):
        assert payload["max_tokens"] == payload["min_tokens"] == 64
        assert payload["ignore_eos"] is True
        assert payload["detokenize"] is False
        assert payload["seed"] == 123


@pytest.mark.parametrize("seeds", (
    [1701], [1701, 1702, 1703, 1704], [1701, 1703, 1702],
))
def test_unsealed_missing_or_reordered_training_seed_is_rejected(seeds):
    with pytest.raises(ValueError, match="sealed training seeds"):
        subject.build_ablation_plan(
            [CANDIDATE_A, CANDIDATE_B], BLOCKS, training_seeds=seeds
        )


def test_candidate_specific_payload_tampering_invalidates_sealed_plan():
    plan = _plan()
    changed = copy.deepcopy(plan)
    changed["assignments"][0]["evaluation_payload_sha256"] = "f" * 64
    changed["plan_sha256"] = subject.canonical_sha256({
        key: value for key, value in changed.items() if key != "plan_sha256"
    })
    with pytest.raises(RuntimeError, match="common payload changed"):
        subject.validate_ablation_plan(changed)


def test_passing_replicated_analysis_selects_same_candidate_all_cells():
    plan = _plan()
    result = subject.analyze_ablation(plan, _cells(plan), _trust_bindings())
    assert result["promotion_eligible"] is True
    assert result["selected_candidate_checkpoint_sha256"] == CANDIDATE_A
    assert subject.require_selection(result) == CANDIDATE_A
    assert all(
        stratum["stable_winner"] == CANDIDATE_A
        for stratum in result["rank_stability"]
    )
    assert result["reward_statistics"][CANDIDATE_A]["seeded_stochastic"][
        "count"
    ] == 12
    assert result["throughput"][CANDIDATE_A]["greedy"][
        "actor_generated_tokens"
    ] > 0


def test_greedy_tie_blocks_arbitrary_selection_even_with_high_aggregate():
    plan = _plan()
    cells = _cells(plan)
    for cell in cells:
        if cell["decode_mode"] == "greedy" and cell["role"] == "dev":
            cell["unit_balanced_reward"] = 0.99
            _rehash(cell)
    result = subject.analyze_ablation(plan, cells, _trust_bindings())
    assert result["promotion_eligible"] is False
    assert any(
        gate.endswith(":greedy:tie_free")
        for gate in result["failed_hard_gates"]
    )
    assert result["selected_candidate_checkpoint_sha256"] is None


def test_greedy_output_nondeterminism_blocks_promotion():
    plan = _plan()
    cells = _cells(plan)
    changed = _find_cell(
        cells,
        candidate_checkpoint_sha256=CANDIDATE_A,
        training_seed=1701,
        decode_mode="greedy",
        repeat_index=3,
        role="train",
    )
    changed["output_inventory_sha256"] = "f" * 64
    changed["judge_metrics_sha256"] = "1" * 64
    _rehash(changed)
    result = subject.analyze_ablation(plan, cells, _trust_bindings())
    assert result["promotion_eligible"] is False
    assert result["hard_gate_checks"][
        "greedy_output_and_judge_identity_replicated"
    ] is False
    assert result["greedy_identity_failures"] == [{
        "candidate_checkpoint_sha256": CANDIDATE_A,
        "training_seed": 1701,
        "role": "train",
    }]


def test_seed_leakage_in_cell_and_unsealed_trust_seed_are_rejected():
    plan = _plan()
    cells = _cells(plan)
    cells[0]["decode_seed"] += 1
    _rehash(cells[0])
    with pytest.raises(RuntimeError, match="sealed decode_seed"):
        subject.analyze_ablation(plan, cells, _trust_bindings())

    receipt = trust_v67.evaluate_trust_region(_trust_evidence())
    with pytest.raises(ValueError, match="unsealed trust-receipt seed"):
        subject.bind_trust_receipt(CANDIDATE_A, 1704, receipt)


def test_unequal_early_stop_and_realized_tokens_block_promotion():
    plan = _plan()
    cells = _cells(plan)
    changed = cells[0]
    changed["realized_generated_tokens"] -= 1
    changed["length_finish_count"] -= 1
    changed["early_stop_count"] = 1
    _rehash(changed)
    result = subject.analyze_ablation(plan, cells, _trust_bindings())
    assert result["promotion_eligible"] is False
    assert result["hard_gate_checks"]["all_cell_protocol_gates"] is False
    assert result["hard_gate_checks"][
        "observed_candidate_mode_token_budgets_equal"
    ] is False
    assert {item["failure"] for item in result["protocol_failures"]} == {
        "generated_token_count_exact", "length_finish_count_exact", "no_early_stop"
    }


def test_stochastic_outlier_cannot_override_three_of_four_modal_ranking():
    plan = _plan()
    cells = _cells(plan)
    for cell in cells:
        candidate = cell["candidate_checkpoint_sha256"]
        if cell["decode_mode"] == "greedy":
            cell["unit_balanced_reward"] = 0.6 if candidate == CANDIDATE_A else 0.5
        elif cell["repeat_index"] < 3:
            cell["unit_balanced_reward"] = 0.6 if candidate == CANDIDATE_A else 0.5
        else:
            cell["unit_balanced_reward"] = 0.0 if candidate == CANDIDATE_A else 1.0
        _rehash(cell)
    result = subject.analyze_ablation(plan, cells, _trust_bindings())
    stats = result["reward_statistics"]
    assert stats[CANDIDATE_B]["seeded_stochastic"]["mean"] > (
        stats[CANDIDATE_A]["seeded_stochastic"]["mean"]
    )
    assert result["promotion_eligible"] is True
    assert result["selected_candidate_checkpoint_sha256"] == CANDIDATE_A
    assert all(
        item["modal_ranking_repeats"] == 3
        for item in result["rank_stability"]
        if item["decode_mode"] == "seeded_stochastic"
    )


def test_one_unstable_stochastic_seed_blocks_aggregate_masking():
    plan = _plan()
    cells = _cells(plan)
    for cell in cells:
        if (
            cell["training_seed"] == 1702
            and cell["decode_mode"] == "seeded_stochastic"
            and cell["repeat_index"] >= 2
        ):
            cell["unit_balanced_reward"] = (
                0.1 if cell["candidate_checkpoint_sha256"] == CANDIDATE_A else 0.9
            )
            _rehash(cell)
    result = subject.analyze_ablation(plan, cells, _trust_bindings())
    assert result["promotion_eligible"] is False
    assert result["hard_gate_checks"][
        "rank:1702:seeded_stochastic:replicated_stability"
    ] is False
    assert result["selected_candidate_checkpoint_sha256"] is None


def test_one_selected_candidate_ood_failure_blocks_all_other_good_aggregates():
    plan = _plan()
    result = subject.analyze_ablation(
        plan,
        _cells(plan),
        _trust_bindings(fail_candidate=CANDIDATE_A, fail_seed=1703),
    )
    assert result["selected_candidate_checkpoint_sha256"] == CANDIDATE_A
    assert result["promotion_eligible"] is False
    assert result["hard_gate_checks"][
        "selected_candidate_passes_train_dev_ood_all_seeds"
    ] is False
    assert result["trust_region_status"][CANDIDATE_A]["1703"]["passed"] is False


def test_one_cell_item_gate_failure_cannot_be_masked_by_reward():
    plan = _plan()
    cells = _cells(plan)
    cells[0]["unit_balanced_reward"] = 1.0
    cells[0]["all_item_hard_gates_passed"] = False
    _rehash(cells[0])
    result = subject.analyze_ablation(plan, cells, _trust_bindings())
    assert result["promotion_eligible"] is False
    assert "multiobjective_item_gates" in {
        item["failure"] for item in result["protocol_failures"]
    }


def test_raw_output_or_protected_access_fails_immediately():
    plan = _plan()
    cells = _cells(plan)
    cells[0]["raw_output_access_count"] = 1
    _rehash(cells[0])
    with pytest.raises(RuntimeError, match="raw candidate output access"):
        subject.analyze_ablation(plan, cells, _trust_bindings())

    cells = _cells(plan)
    cells[0]["protected_source_opened"] = True
    _rehash(cells[0])
    with pytest.raises(RuntimeError, match="protected output access"):
        subject.analyze_ablation(plan, cells, _trust_bindings())


def test_output_text_token_ids_or_unknown_metrics_cannot_enter_receipt():
    plan = _plan()
    cells = _cells(plan)
    cells[0]["generated_text"] = "must not be accepted"
    _rehash(cells[0])
    with pytest.raises(ValueError, match="keys changed"):
        subject.analyze_ablation(plan, cells, _trust_bindings())


def test_missing_duplicate_or_foreign_assignment_fails_closed():
    plan = _plan()
    cells = _cells(plan)
    with pytest.raises(ValueError, match="exactly cover"):
        subject.analyze_ablation(plan, cells[:-1], _trust_bindings())
    with pytest.raises(ValueError, match="duplicate"):
        subject.analyze_ablation(
            plan, cells[:-1] + [copy.deepcopy(cells[0])], _trust_bindings()
        )


def test_mutated_analysis_cannot_be_promoted():
    plan = _plan()
    result = subject.analyze_ablation(plan, _cells(plan), _trust_bindings())
    result["promotion_eligible"] = False
    with pytest.raises(RuntimeError, match="invalid or mutated"):
        subject.require_selection(result)


def test_preregistration_is_content_addressed_and_persisted_exactly():
    first = subject.build_preregistration()
    second = subject.build_preregistration()
    assert first == second
    assert first["generated_token_budget"][
        "candidate_and_mode_totals_must_be_exactly_equal"
    ] is True
    assert first["rank_and_selection_stability"][
        "same_winner_required_every_seed_and_mode"
    ] is True
    assert first["content_sha256_before_self_field"] == subject.canonical_sha256({
        key: value for key, value in first.items()
        if key != "content_sha256_before_self_field"
    })
    subject.validate_preregistration(json.loads(
        subject.PREREGISTRATION.read_text(encoding="utf-8")
    ))
