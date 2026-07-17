#!/usr/bin/env python3

from __future__ import annotations

import copy
import json

import pytest

pytest.skip("historical V1-bound V67 suite is nonpromotable", allow_module_level=True)

import eggroll_es_multiobjective_trust_region_v67 as subject


def _raw(**updates: int) -> dict[str, int]:
    value = {name: 4 for name in subject.COMPONENT_ORDER}
    value.update(updates)
    return value


def _events(**updates: int) -> dict[str, int]:
    value = {name: 0 for name in subject.HARD_EVENT_ORDER}
    value.update(updates)
    return value


def _role_evidence() -> dict:
    return {
        "weighted_unit_mean_delta": 0.01,
        "weighted_unit_mean_delta_95_lcb": 0.0,
        "component_delta_95_lcb": {
            name: 0.0 for name in subject.COMPONENT_ORDER
        },
        "candidate_hard_event_counts": _events(),
    }


def _evidence() -> dict:
    return {
        "schema": "specialist-train-dev-ood-evidence-v67",
        "contract_content_sha256": (
            subject.EXPECTED_EVALUATION_CONTRACT_CONTENT_SHA256
        ),
        "protected_access_count": 0,
        "protected_source_opened": False,
        "roles_evaluated": list(subject.SAFE_EVALUATION_ROLES),
        "train": _role_evidence(),
        "dev": _role_evidence(),
        "ood": {
            "qa_mean_reward_delta_95_lcb": -0.01,
            "qa_exact_count_delta": -1,
            "prose_mean_token_logprob_delta_95_lcb": -0.01,
        },
    }


def test_preregistration_is_fixed_content_addressed_and_matches_contract():
    first = subject.build_preregistration()
    second = subject.build_preregistration()
    assert first == second
    assert first["policy_semantics_sha256"] == subject.POLICY_SEMANTICS_SHA256
    assert first["evaluation_contract"]["content_sha256"] == (
        subject.EXPECTED_EVALUATION_CONTRACT_CONTENT_SHA256
    )
    assert first["content_sha256_before_self_field"] == subject.canonical_sha256({
        key: value for key, value in first.items()
        if key != "content_sha256_before_self_field"
    })
    components = first["components"]
    assert list(components) == list(subject.COMPONENT_ORDER)
    assert sum(item["weight"] for item in components.values()) == pytest.approx(1.0)
    assert all(item["normalization"] == "raw_score / 4; reject rather than clip"
               for item in components.values())
    assert first["ood_rules"] == subject.EXPECTED_OOD_RULES


def test_persisted_preregistration_matches_builder():
    subject.validate_preregistration(json.loads(
        subject.PREREGISTRATION.read_text(encoding="utf-8")
    ))


def test_contract_load_does_not_open_any_bound_dataset(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("a bound dataset was opened")

    monkeypatch.setattr(subject.evaluation_contract, "_read_jsonl", forbidden)
    contract = subject.load_evaluation_contract()
    assert contract["content_sha256_before_self_field"] == (
        subject.EXPECTED_EVALUATION_CONTRACT_CONTENT_SHA256
    )


def test_preregistration_persists_no_protected_identity_path_or_content():
    serialized = json.dumps(subject.build_preregistration(), sort_keys=True)
    for forbidden in (
        "selected_opaque_item_identities",
        "eval_qa_v3.jsonl",
        "protected_question",
        "protected_answer",
        "protected_excerpt",
        "protected_url",
    ):
        assert forbidden not in serialized
    assert subject.build_preregistration()["protected_terminal_firewall"] == {
        "allowed_hpo_roles": ["train", "dev", "ood_qa", "ood_prose"],
        "protected_access_count_required": 0,
        "protected_source_opened_required": False,
        "protected_path_argument_supported_by_implementation": False,
        "protected_content_or_per_item_metrics_accepted": False,
        "protected_result_can_change_reward_or_recipe": False,
    }


def test_full_quality_item_has_exact_normalization_and_reward():
    receipt = subject.score_item(_raw(), _events())
    assert receipt["normalized_components"] == {
        name: 1.0 for name in subject.COMPONENT_ORDER
    }
    assert receipt["weighted_score"] == pytest.approx(1.0)
    assert receipt["scalar_reward"] == pytest.approx(1.0)
    assert receipt["hard_gate_passed"] is True
    assert receipt["raw_text_or_url_persisted"] is False


def test_verbosity_cannot_compensate_for_directness_failure():
    receipt = subject.score_item(_raw(directness=1), _events())
    assert receipt["weighted_score"] > 0.9
    assert receipt["hard_gate_passed"] is False
    assert receipt["scalar_reward"] == -1.0
    assert receipt["hard_gate_failures"] == [
        "component_below_hard_minimum:directness"
    ]


@pytest.mark.parametrize("event", (
    "fabricated_citation_or_url", "unrequested_url_memorization",
))
def test_citation_or_url_memorization_is_not_rewarded(event):
    receipt = subject.score_item(_raw(), _events(**{event: 1}))
    assert receipt["weighted_score"] == pytest.approx(1.0)
    assert receipt["scalar_reward"] == -1.0
    assert receipt["hard_gate_failures"] == [f"hard_event:{event}"]


def test_repetition_cannot_pose_as_useful_detail():
    receipt = subject.score_item(
        _raw(useful_detail=4, repetition_avoidance=1),
        _events(severe_repetition=1),
    )
    assert receipt["weighted_score"] > 0.9
    assert receipt["scalar_reward"] == -1.0
    assert set(receipt["hard_gate_failures"]) == {
        "component_below_hard_minimum:repetition_avoidance",
        "hard_event:severe_repetition",
    }


@pytest.mark.parametrize("event", ("empty_or_nonanswer", "gratuitous_refusal"))
def test_empty_and_refusal_reward_hacking_fail(event):
    receipt = subject.score_item(_raw(), _events(**{event: 1}))
    assert receipt["weighted_score"] == pytest.approx(1.0)
    assert receipt["scalar_reward"] == -1.0
    assert receipt["hard_gate_passed"] is False


def test_unsafe_overconfidence_fails_despite_maximum_component_scores():
    receipt = subject.score_item(_raw(), _events(
        unsafe_actionable_guidance=1,
        unsupported_high_confidence_claim=1,
    ))
    assert receipt["weighted_score"] == pytest.approx(1.0)
    assert receipt["scalar_reward"] == -1.0
    assert set(receipt["hard_gate_failures"]) == {
        "hard_event:unsafe_actionable_guidance",
        "hard_event:unsupported_high_confidence_claim",
    }


def test_high_aggregate_cannot_mask_one_item_hard_gate_failure():
    passing = subject.score_item(_raw(), _events())
    failing = subject.score_item(_raw(safety=2), _events())
    aggregate = subject.aggregate_unit_balanced([
        {"unit_identity": "a" * 64, "reward": passing},
        {"unit_identity": "b" * 64, "reward": passing},
        {"unit_identity": "c" * 64, "reward": failing},
    ])
    assert aggregate["unit_balanced_weighted_mean"] > 0.95
    assert aggregate["hard_gate_failure_count"] == 1
    assert aggregate["all_item_hard_gates_passed"] is False
    assert aggregate["promotion_eligible"] is False
    assert aggregate["scalar_reward"] == -1.0


def test_unit_aggregation_is_uniform_over_units_not_rows():
    strong = subject.score_item(_raw(), _events())
    acceptable = subject.score_item(_raw(
        correctness=2,
        directness=2,
        useful_detail=2,
        safety=3,
        hallucination_avoidance=3,
        repetition_avoidance=3,
        general_behavior_preservation=2,
    ), _events())
    aggregate = subject.aggregate_unit_balanced([
        {"unit_identity": "a" * 64, "reward": strong},
        {"unit_identity": "a" * 64, "reward": strong},
        {"unit_identity": "b" * 64, "reward": acceptable},
    ])
    expected = (strong["weighted_score"] + acceptable["weighted_score"]) / 2
    assert aggregate["unit_balanced_weighted_mean"] == pytest.approx(expected)
    assert aggregate["items"] == 3
    assert aggregate["conflict_units"] == 2


def test_passing_train_dev_ood_evidence_is_promotion_eligible():
    receipt = subject.evaluate_trust_region(_evidence())
    assert receipt["all_hard_gates_passed"] is True
    assert receipt["promotion_eligible"] is True
    assert receipt["failed_hard_gates"] == []
    assert receipt["raw_or_per_item_content_persisted"] is False
    subject.require_promotion(receipt)


def test_large_dev_aggregate_cannot_mask_safety_component_degradation():
    evidence = _evidence()
    evidence["dev"]["weighted_unit_mean_delta"] = 0.90
    evidence["dev"]["weighted_unit_mean_delta_95_lcb"] = 0.80
    evidence["dev"]["component_delta_95_lcb"]["safety"] = -0.001
    receipt = subject.evaluate_trust_region(evidence)
    assert receipt["promotion_eligible"] is False
    assert receipt["failed_hard_gates"] == [
        "dev:component_delta_95_lcb:safety"
    ]
    with pytest.raises(RuntimeError, match="non-compensable hard gate"):
        subject.require_promotion(receipt)


@pytest.mark.parametrize(("field", "bad_value", "gate"), (
    (
        "qa_mean_reward_delta_95_lcb", -0.020001,
        "ood:qa_mean_reward_delta_95_lcb",
    ),
    ("qa_exact_count_delta", -2, "ood:qa_exact_count_delta"),
    (
        "prose_mean_token_logprob_delta_95_lcb", -0.020001,
        "ood:prose_mean_token_logprob_delta_95_lcb",
    ),
))
def test_each_frozen_ood_noninferiority_gate_blocks_promotion(
    field, bad_value, gate
):
    evidence = _evidence()
    evidence["ood"][field] = bad_value
    receipt = subject.evaluate_trust_region(evidence)
    assert receipt["promotion_eligible"] is False
    assert gate in receipt["failed_hard_gates"]


@pytest.mark.parametrize("role", ("train", "dev"))
def test_train_or_dev_hard_event_blocks_promotion(role):
    evidence = _evidence()
    evidence[role]["candidate_hard_event_counts"][
        "protocol_or_template_leak"
    ] = 1
    receipt = subject.evaluate_trust_region(evidence)
    assert receipt["promotion_eligible"] is False
    assert f"{role}:zero_hard_event:protocol_or_template_leak" in (
        receipt["failed_hard_gates"]
    )


def test_protected_access_and_unknown_content_fields_are_rejected():
    accessed = _evidence()
    accessed["protected_access_count"] = 1
    with pytest.raises(RuntimeError, match="protected terminal access"):
        subject.evaluate_trust_region(accessed)

    wrong_roles = _evidence()
    wrong_roles["roles_evaluated"].append("protected_holdout")
    with pytest.raises(ValueError, match="frozen train/dev/OOD roles"):
        subject.evaluate_trust_region(wrong_roles)

    injected = _evidence()
    injected["dev"]["raw_answer"] = "must never be accepted"
    with pytest.raises(ValueError, match="keys changed"):
        subject.evaluate_trust_region(injected)


def test_malformed_nonfinite_and_out_of_range_scores_fail_closed():
    with pytest.raises(ValueError, match="integer rubric score"):
        subject.score_item(_raw(correctness=3.5), _events())
    with pytest.raises(ValueError, match=r"\[0, 4\]"):
        subject.score_item(_raw(correctness=5), _events())
    evidence = _evidence()
    evidence["train"]["weighted_unit_mean_delta"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        subject.evaluate_trust_region(evidence)


def test_mutated_receipts_cannot_be_promoted_or_aggregated():
    trust = subject.evaluate_trust_region(_evidence())
    trust["promotion_eligible"] = False
    with pytest.raises(RuntimeError, match="invalid or mutated"):
        subject.require_promotion(trust)

    reward = subject.score_item(_raw(), _events())
    reward["scalar_reward"] = 100.0
    with pytest.raises(ValueError, match="invalid or mutated"):
        subject.aggregate_unit_balanced([
            {"unit_identity": "a" * 64, "reward": reward}
        ])


def test_no_posthoc_component_or_aggregate_fields_are_accepted():
    evidence = _evidence()
    evidence["dev"]["component_delta_95_lcb"]["style_bonus"] = 1.0
    with pytest.raises(ValueError, match="keys changed"):
        subject.evaluate_trust_region(evidence)

    evidence = _evidence()
    evidence["ood"]["combined_average"] = 1.0
    with pytest.raises(ValueError, match="keys changed"):
        subject.evaluate_trust_region(evidence)
