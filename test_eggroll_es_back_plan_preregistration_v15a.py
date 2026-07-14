#!/usr/bin/env python3

import copy
import hashlib
import json

import pytest

import eggroll_es_back_plan_preregistration_v15a as prereg


def _candidate():
    control = copy.deepcopy(prereg.V13_BASELINE_STABILITY_V15A)
    back = copy.deepcopy(control)
    for name in back:
        if name.endswith("_cosine"):
            back[name]["median"] += prereg.COSINE_MINIMUM_IMPROVEMENT_V15A
            back[name]["worst"] += prereg.COSINE_MINIMUM_IMPROVEMENT_V15A
    panels = prereg.validate_v13_estimator_v15a()
    value = {
        "schema": "eggroll-es-back-plan-stability-summary-v15a",
        "experiment_name": prereg.EXPERIMENT_NAME_V15A,
        "alpha": 0.0,
        "model_update_applied": False,
        "validation_ood_heldout_or_benchmark_used": False,
        "perturbation_basis_sha256": prereg.PERTURBATION_BASIS_SHA256_V15A,
        "panel_bundle_content_sha256": (
            prereg.anchor_v13.PANEL_BUNDLE_CONTENT_SHA256_V13
        ),
        "panel_identities": {
            name: panel["ordered_row_identity_sha256"]
            for name, panel in panels["panels"].items()
        },
        "arm_order": list(prereg.ARM_ORDER_V15A),
        "arms": {
            "middle_late": {
                "plan_sha256": prereg.LAYER_PLANS_V15A["middle_late"][
                    "plan_sha256"
                ],
                "stability": control,
                "robust_aggregate": {
                    "coefficient_sha256": "a" * 64,
                    "l2_norm": 4.5,
                    "nonzero_coordinate_count": 32,
                },
            },
            "back": {
                "plan_sha256": prereg.LAYER_PLANS_V15A["back"]["plan_sha256"],
                "stability": back,
                "robust_aggregate": {
                    "coefficient_sha256": "b" * 64,
                    "l2_norm": 4.7,
                    "nonzero_coordinate_count": 32,
                },
            },
        },
        "all_panel_spreads_nonzero": {"middle_late": True, "back": True},
        "all_integrity_audits_passed": True,
    }
    value["content_sha256_before_self_field"] = prereg._canonical(value)
    return value


def _reseal(value):
    value["content_sha256_before_self_field"] = prereg._canonical({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def _walk_forbidden(value):
    forbidden = {
        "responses", "coefficients", "questions", "answers", "documents",
        "validation", "ood", "heldout", "benchmark",
    }
    if isinstance(value, dict):
        assert not forbidden.intersection(value)
        for item in value.values():
            _walk_forbidden(item)
    elif isinstance(value, list):
        for item in value:
            _walk_forbidden(item)


def test_v15a_binds_v13_baseline_and_both_negative_evidence_artifacts():
    v13 = prereg.load_v13_evidence_v15a()
    v14a = prereg.load_v14a_negative_v15a()
    v14b = prereg.load_v14b_negative_v15a()
    assert {
        name: v13["stability"][name]
        for name in prereg.V13_BASELINE_STABILITY_V15A
    } == prereg.V13_BASELINE_STABILITY_V15A
    assert v14a["decision"]["sampler"] == "retain_v13"
    assert v14b["decision"]["sampler"] == "retain_v13"
    assert v14a["aggregate_gate"][
        "eligible_for_train_only_sampler_adoption"
    ] is False
    assert v14b["aggregate_gate"][
        "eligible_for_fresh_basis_confirmation"
    ] is False
    _walk_forbidden(v13)
    _walk_forbidden(v14a)
    _walk_forbidden(v14b)


def test_v15a_exact_v13_five_panel_estimator_is_unchanged():
    bundle = prereg.validate_v13_estimator_v15a()
    assert bundle["source"]["rows"] == 794
    assert tuple(bundle["panels"]) == prereg.anchor_v13.PANEL_NAMES_V13
    assert [len(panel["questions"]) for panel in bundle["panels"].values()] == [
        56, 56, 56, 56, 56,
    ]
    assert bundle["content_sha256_before_self_field"] == (
        "cc176a9b86c6447dcde8a11fd28d68c837d2119715126c57a3f37293fb0d492b"
    )


def test_v15a_capacity_matched_back_and_middle_late_plans_are_exact():
    bundles = prereg.validate_layer_plans_v15a()
    assert tuple(bundles) == prereg.ARM_ORDER_V15A
    assert bundles["middle_late"]["edge_split_v6"]["layers"] == [20, 21, 22, 23]
    assert bundles["back"]["edge_split_v6"]["layers"] == [36, 37, 38, 39]
    assert all(
        prereg.anchor_v6.FROZEN_RUNTIME_EXPECTATIONS_V6[
            prereg.LAYER_PLANS_V15A[name]["plan_sha256"]
        ] == prereg.CAPACITY_V15A
        for name in prereg.ARM_ORDER_V15A
    )


def test_v15a_basis_is_fresh_exact_and_shared_by_both_arms():
    basis = prereg.validate_perturbation_basis_v15a()
    assert basis["basis_seed"] == 20260715
    assert len(basis["seeds"]) == len(set(basis["seeds"])) == 32
    assert prereg._canonical(basis) == prereg.PERTURBATION_BASIS_SHA256_V15A
    assert prereg.PERTURBATION_BASIS_SHA256_V15A != (
        prereg.PREVIOUS_PERTURBATION_BASIS_SHA256_V15A
    )
    value = prereg.build_preregistration_v15a()
    assert value["paired_architecture"]["same_fresh_basis_both_arms"] is True
    assert value["runtime"]["perturbation_basis"] == basis


def test_v15a_preregistration_is_one_paired_train_only_hypothesis():
    value = prereg.build_preregistration_v15a()
    assert value["status"] == "preregistered_not_launch_authorized"
    assert value["hypothesis_count"] == 1
    assert value["selection_surface"] == "exact_frozen_v13_train_panels_only"
    assert value["contains_validation_ood_heldout_or_benchmark_content"] is False
    assert value["paired_architecture"]["arm_order"] == ["middle_late", "back"]
    assert value["runtime"]["alpha"] == 0.0
    assert value["runtime"]["model_update_allowed"] is False
    assert value["runtime"]["gpu_ids"] == [0, 1, 2, 3]
    assert value["runtime"]["all_four_engines_required_every_signed_wave"]
    assert value["v13_estimator"]["no_full_frame_or_disjoint_crossfit_claim"]
    assert value["v13_estimator"]["native_endpoints_only"] == list(
        prereg.METRIC_COUNTS_V15A
    )
    assert value["required_runtime_adapter"]["status"] == "not_yet_implemented"


def test_v15a_gate_passes_only_material_absolute_and_paired_improvement():
    gate = prereg.evaluate_candidate_v15a(_candidate())
    assert gate["eligible_for_fresh_basis_back_plan_confirmation"] is True
    assert gate["eligible_for_model_update"] is False
    assert gate["eligible_to_open_evaluation"] is False
    assert gate["pass_decision"] == (
        "preregister_back_plan_alpha_zero_confirmation_on_another_fresh_basis"
    )

    tied = _candidate()
    tied["arms"]["back"]["stability"] = copy.deepcopy(
        tied["arms"]["middle_late"]["stability"]
    )
    _reseal(tied)
    tied_gate = prereg.evaluate_candidate_v15a(tied)
    assert tied_gate["eligible_for_fresh_basis_back_plan_confirmation"] is False
    assert tied_gate["failure_decision"] == (
        "retain_v13_middle_late_and_keep_all_eval_surfaces_closed"
    )


@pytest.mark.parametrize(
    "metric", [name for name in prereg.METRIC_COUNTS_V15A if name.endswith("_cosine")]
)
def test_v15a_every_cosine_median_and_worst_requires_fixed_margin(metric):
    for endpoint in ("median", "worst"):
        candidate = _candidate()
        candidate["arms"]["back"]["stability"][metric][endpoint] -= 1e-6
        _reseal(candidate)
        gate = prereg.evaluate_candidate_v15a(candidate)
        assert gate["eligible_for_fresh_basis_back_plan_confirmation"] is False
        assert gate["conditions"]["absolute_v13"][metric][
            f"{endpoint}_passed"
        ] is False
        assert gate["conditions"]["paired_middle_late_control"][metric][
            f"{endpoint}_passed"
        ] is False


@pytest.mark.parametrize(
    "metric", [name for name in prereg.METRIC_COUNTS_V15A if name.endswith("sign_agreement")]
)
def test_v15a_every_sign_median_and_worst_forbids_regression(metric):
    for endpoint in ("median", "worst"):
        candidate = _candidate()
        candidate["arms"]["back"]["stability"][metric][endpoint] -= 1 / 32
        _reseal(candidate)
        gate = prereg.evaluate_candidate_v15a(candidate)
        assert gate["eligible_for_fresh_basis_back_plan_confirmation"] is False
        assert gate["conditions"]["absolute_v13"][metric][
            f"{endpoint}_passed"
        ] is False
        assert gate["conditions"]["paired_middle_late_control"][metric][
            f"{endpoint}_passed"
        ] is False


def test_v15a_candidate_contract_fails_closed_on_update_plan_or_hash_tampering():
    candidate = _candidate()
    candidate["model_update_applied"] = True
    _reseal(candidate)
    with pytest.raises(RuntimeError, match="summary contract"):
        prereg.evaluate_candidate_v15a(candidate)

    candidate = _candidate()
    candidate["arms"]["back"]["plan_sha256"] = "0" * 64
    _reseal(candidate)
    with pytest.raises(RuntimeError, match="arm contract"):
        prereg.evaluate_candidate_v15a(candidate)

    candidate = _candidate()
    candidate["content_sha256_before_self_field"] = "0" * 64
    with pytest.raises(RuntimeError, match="summary contract"):
        prereg.evaluate_candidate_v15a(candidate)


def test_v15a_front_plus_back_is_separate_and_never_adaptive_fallback():
    value = prereg.build_preregistration_v15a()
    separate = value["separate_future_hypothesis"]
    assert separate == {
        "front_plus_back_is_part_of_v15a": False,
        "front_plus_back_authorized_on_v15a_failure": False,
        "front_plus_back_authorized_on_v15a_pass": False,
        "requirement": (
            "a separate preregistration, basis, implementation, and gate "
            "before any front-plus-back experiment"
        ),
    }
    assert "front plus back result selection" in value["firewall"]["forbidden"]


def test_v15a_preregistration_file_is_exact_rebuild():
    frozen = json.loads(prereg.PREREGISTRATION_PATH_V15A.read_text())
    assert frozen == prereg.build_preregistration_v15a()
    assert frozen["content_sha256_before_self_field"] == (
        "dda0f49e470cf5bb550f80d27a2389d069c8d064975c18b581261054462bb7c7"
    )
    assert hashlib.sha256(prereg.PREREGISTRATION_PATH_V15A.read_bytes()).hexdigest() == (
        "ad86f388ff4effbc195a3fd60d6d32c430a83026a331a18d625d477d390f3b88"
    )
