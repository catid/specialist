#!/usr/bin/env python3

import copy
import hashlib
import json

import pytest

import eggroll_es_back_plan_confirmation_preregistration_v15b as prereg


def _candidate():
    frozen = prereg.build_preregistration_v15b()
    control = copy.deepcopy(
        frozen["promotion_gate"]["historical_v13_baseline"]
    )
    back = copy.deepcopy(frozen["promotion_gate"]["v15a_back_reference"])
    value = {
        "schema": "eggroll-es-back-plan-confirmation-summary-v15b",
        "experiment_name": prereg.EXPERIMENT_NAME_V15B,
        "alpha": 0.0,
        "model_update_applied": False,
        "validation_ood_heldout_or_benchmark_used": False,
        "perturbation_basis_sha256": prereg.PERTURBATION_BASIS_SHA256_V15B,
        "panel_bundle_content_sha256": frozen["estimator"][
            "panel_bundle_content_sha256"
        ],
        "panel_identities": frozen["estimator"]["ordered_panel_identities"],
        "arm_order": list(prereg.ARM_ORDER_V15B),
        "arms": {
            "middle_late": {
                "plan_sha256": frozen["paired_architecture"]["arms"]
                ["middle_late"]["plan_sha256"],
                "stability": control,
                "robust_aggregate": {
                    "coefficient_sha256": "a" * 64,
                    "l2_norm": 4.0,
                    "nonzero_coordinate_count": 32,
                },
            },
            "back": {
                "plan_sha256": frozen["paired_architecture"]["arms"]
                ["back"]["plan_sha256"],
                "stability": back,
                "robust_aggregate": {
                    "coefficient_sha256": "b" * 64,
                    "l2_norm": 4.5,
                    "nonzero_coordinate_count": 32,
                },
            },
        },
        "all_panel_spreads_nonzero": {"middle_late": True, "back": True},
        "all_integrity_audits_passed": True,
    }
    value["content_sha256_before_self_field"] = prereg.canonical_sha256(value)
    return value


def _reseal(value):
    value["content_sha256_before_self_field"] = prereg.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def test_v15b_authorization_scope_basis_and_firewall_are_exact():
    value = prereg.build_preregistration_v15b()
    assert value["authorization"]["paired_rerun_within_authorization"]
    assert value["authorization"]["second_architecture_hypothesis_added"] is False
    assert value["paired_architecture"]["candidate_arm"] == "back"
    assert value["paired_architecture"]["control_can_be_promoted"] is False
    assert value["runtime"]["alpha"] == 0.0
    assert value["runtime"]["model_update_allowed"] is False
    assert value["contains_validation_ood_heldout_or_benchmark_content"] is False
    basis = prereg.validate_perturbation_basis_v15b()
    assert len(basis["seeds"]) == len(set(basis["seeds"])) == 32
    assert prereg.canonical_sha256(basis) == prereg.PERTURBATION_BASIS_SHA256_V15B
    assert prereg.PERTURBATION_BASIS_SHA256_V15B != (
        prereg.V15A_PERTURBATION_BASIS_SHA256_V15B
    )


def test_v15b_gate_passes_all_three_predeclared_families_and_no_more():
    gate = prereg.evaluate_candidate_v15b(_candidate())
    assert gate["eligible_for_separate_back_plan_train_update_preregistration"]
    assert gate["eligible_for_model_update"] is False
    assert gate["eligible_to_open_evaluation"] is False
    assert gate["pass_decision"] == (
        "preregister_back_plan_nonzero_alpha_train_update_experiment"
    )
    for family in (
        "absolute_v13", "paired_middle_late_control",
        "v15a_replication_stability",
    ):
        assert all(
            flag
            for condition in gate["conditions"][family].values()
            for label, flag in condition.items()
            if label.endswith("passed")
        )


def test_v15b_sorted_json_candidate_replays_by_explicit_arm_order():
    persisted = json.loads(json.dumps(_candidate(), sort_keys=True))
    assert tuple(persisted["arms"]) == ("back", "middle_late")
    gate = prereg.evaluate_candidate_v15b(persisted)
    assert gate["eligible_for_separate_back_plan_train_update_preregistration"]


@pytest.mark.parametrize("endpoint", ["median", "worst"])
def test_v15b_absolute_and_paired_cosine_margin_is_conjunctive(endpoint):
    candidate = _candidate()
    metric = "aggregate_to_optimization_cosine"
    baseline = prereg.build_preregistration_v15b()["promotion_gate"]
    baseline = baseline["historical_v13_baseline"][metric][endpoint]
    candidate["arms"]["back"]["stability"][metric][endpoint] = (
        baseline + prereg.COSINE_MINIMUM_IMPROVEMENT_V15B - 1e-6
    )
    if endpoint == "worst":
        candidate["arms"]["back"]["stability"][metric]["median"] = max(
            candidate["arms"]["back"]["stability"][metric]["median"],
            candidate["arms"]["back"]["stability"][metric]["worst"],
        )
    _reseal(candidate)
    gate = prereg.evaluate_candidate_v15b(candidate)
    assert not gate["eligible_for_separate_back_plan_train_update_preregistration"]
    assert not gate["conditions"]["absolute_v13"][metric][
        f"{endpoint}_passed"
    ]


def test_v15b_v15a_stability_tolerance_is_fixed_and_binding():
    candidate = _candidate()
    metric = "train_screen_cosine"
    reference = prereg.build_preregistration_v15b()["promotion_gate"]
    reference = reference["v15a_back_reference"][metric]["median"]
    candidate["arms"]["back"]["stability"][metric]["median"] = (
        reference - prereg.COSINE_V15A_REPLICATION_TOLERANCE_V15B - 1e-6
    )
    _reseal(candidate)
    gate = prereg.evaluate_candidate_v15b(candidate)
    assert gate["conditions"]["absolute_v13"][metric]["median_passed"]
    assert gate["conditions"]["paired_middle_late_control"][metric][
        "median_passed"
    ]
    assert not gate["conditions"]["v15a_replication_stability"][metric][
        "median_passed"
    ]
    assert not gate["eligible_for_separate_back_plan_train_update_preregistration"]


def test_v15b_update_or_integrity_tampering_fails_closed():
    candidate = _candidate()
    candidate["model_update_applied"] = True
    _reseal(candidate)
    with pytest.raises(RuntimeError, match="summary contract"):
        prereg.evaluate_candidate_v15b(candidate)

    candidate = _candidate()
    candidate["all_integrity_audits_passed"] = False
    _reseal(candidate)
    with pytest.raises(RuntimeError, match="summary contract"):
        prereg.evaluate_candidate_v15b(candidate)


def test_v15b_frozen_preregistration_file_is_exact_rebuild():
    frozen = json.loads(prereg.PREREGISTRATION_PATH_V15B.read_text())
    assert frozen == prereg.build_preregistration_v15b()
    assert frozen["content_sha256_before_self_field"] == (
        "0a4efb1a8a07cd194876d0942e77b188e4463b86cd6076606bccfb92f054f720"
    )
    assert hashlib.sha256(
        prereg.PREREGISTRATION_PATH_V15B.read_bytes()
    ).hexdigest() == (
        "5b90f16961c94d3a04b72ae29860094f7f1e6e8bad793780967c27448e0ba57f"
    )
