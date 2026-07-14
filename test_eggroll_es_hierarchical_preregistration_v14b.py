#!/usr/bin/env python3

import copy
import hashlib
import json

import pytest

import eggroll_es_hierarchical_preregistration_v14a as prereg_v14a
import eggroll_es_hierarchical_preregistration_v14b as prereg


def _candidate(delta=0.01):
    stability = copy.deepcopy(prereg.BASELINE_STABILITY_V14B)
    for name in (
        "matched56_pairwise_cosine",
        "crossfit_complement_to_screen_cosine",
    ):
        stability[name]["median"] += delta
        stability[name]["worst"] += delta
    value = {
        "schema": "eggroll-es-paired-distinct-row-summary-v14b",
        "experiment_name": prereg.EXPERIMENT_NAME_V14B,
        "alpha": 0.0,
        "model_update_applied": False,
        "validation_ood_or_heldout_used": False,
        "perturbation_basis_sha256": prereg.PERTURBATION_BASIS_SHA256_V14B,
        "panel_identities": prereg.candidate_panel_identities_v14b(),
        "stability": stability,
        "all_panel_spreads_nonzero": True,
        "robust_aggregate": {
            "coefficient_sha256": "a" * 64,
            "l2_norm": 4.0,
            "nonzero_coordinate_count": 32,
        },
        "all_integrity_audits_passed": True,
    }
    value["content_sha256_before_self_field"] = prereg._canonical(value)
    return value


def _reseal(value):
    value["content_sha256_before_self_field"] = prereg._canonical({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def test_v14b_binds_exact_v14a_negative_and_v13_baseline_evidence():
    negative = prereg.load_v14a_negative_evidence_v14b()
    baseline = prereg.load_v13_baseline_evidence_v14b()
    assert negative["decision"]["sampler"] == "retain_v13"
    assert negative["decision"][
        "row_draw_iteration_1_confirmation_authorized"
    ] is False
    assert negative["contains_response_vectors_or_dense_result_hashes"] is False
    assert baseline["content_sha256_before_self_field"] == (
        prereg.V13_BASELINE_EVIDENCE_CONTENT_SHA256_V14B
    )
    assert prereg.BASELINE_STABILITY_V14B == prereg_v14a.BASELINE_STABILITY_V14A


def test_v14b_materialization_binds_481_prompts_and_exact_crossfits():
    rows, full, panels, complements = prereg.materialize_sampler_v14b()
    del rows
    assert full["documents"] == 310
    assert full["single_row_documents"] == 139
    assert full["multirow_documents"] == 171
    assert full["prompts"] == 481
    assert full["ordered_prompt_identity_sha256"] == (
        prereg.FULL_FRAME_IDENTITY_V14B["ordered_prompt_identity_sha256"]
    )
    documents = [
        item["document_sha256"]
        for panel in panels.values() for item in panel["items"]
    ]
    assert len(documents) == len(set(documents)) == 280
    assert [panels[name]["prompts"] for name in panels] == [92, 81, 87, 88, 86]
    assert [complements[name]["documents"] for name in complements] == [254, 254]
    assert [complements[name]["prompts"] for name in complements] == [393, 395]


def test_v14b_preregistration_freezes_single_hypothesis_recipe_and_integrity():
    value = prereg.build_preregistration_v14b()
    assert value["status"] == "preregistered_not_launch_authorized"
    assert value["sampling"]["hypothesis_count"] == 1
    assert value["sampling"]["generation_prompt_count_per_direction_and_sign"] == 481
    assert value["sampling"]["distinct_rows_per_multirow_document"] == 2
    assert "without-replacement" in value["sampling"][
        "without_replacement_algorithm"
    ]
    assert value["full_frame"][
        "document_means_computed_before_equal_document_mean"
    ] is True
    assert value["runtime"]["alpha"] == 0.0
    assert value["runtime"]["model_update_allowed"] is False
    assert value["runtime"]["layer_plan"]["plan_sha256"] == (
        prereg.LAYER_PLAN_SHA256_V14B
    )
    assert value["runtime"]["perturbation_basis_sha256"] == (
        prereg_v14a.PERTURBATION_BASIS_SHA256_V14A
    )
    assert value["integrity"]["restore_in_finally_after_every_sign_required"]
    assert value["integrity"]["persist_per_document_reward_vectors"] is False
    assert value["required_runtime_adapter"]["status"] == "not_yet_implemented"


def test_v14b_gate_passes_only_v13_baseline_conjunctive_improvement():
    gate = prereg.evaluate_candidate_v14b(_candidate())
    assert gate["eligible_for_train_only_estimator_confirmation"] is True
    assert gate["eligible_for_model_update"] is False
    assert gate["eligible_to_open_evaluation"] is False
    assert gate["pass_decision"] == (
        "preregister_fresh_basis_k2_alpha_zero_confirmation"
    )
    tied = prereg.evaluate_candidate_v14b(_candidate(delta=0.0))
    assert tied["eligible_for_train_only_estimator_confirmation"] is False
    assert tied["conditions"]["matched56_pairwise_cosine"]["comparison"] == (
        "strictly_greater"
    )


def test_v14b_gate_rejects_any_sign_screen_or_integrity_regression():
    candidate = _candidate()
    candidate["stability"]["matched56_pairwise_sign_agreement"]["worst"] -= 0.01
    _reseal(candidate)
    assert prereg.evaluate_candidate_v14b(candidate)[
        "eligible_for_train_only_estimator_confirmation"
    ] is False

    candidate = _candidate()
    candidate["stability"]["crossfit_complement_to_screen_cosine"]["worst"] = (
        prereg.BASELINE_STABILITY_V14B[
            "crossfit_complement_to_screen_cosine"
        ]["worst"]
    )
    _reseal(candidate)
    assert prereg.evaluate_candidate_v14b(candidate)[
        "eligible_for_train_only_estimator_confirmation"
    ] is False

    candidate = _candidate()
    candidate["all_integrity_audits_passed"] = False
    _reseal(candidate)
    with pytest.raises(RuntimeError, match="contract changed"):
        prereg.evaluate_candidate_v14b(candidate)


@pytest.mark.parametrize("name", tuple(prereg.BASELINE_STABILITY_V14B))
def test_v14b_each_strict_and_nonlower_boundary_is_exact(name):
    baseline = prereg.BASELINE_STABILITY_V14B[name]
    strict = name in {
        "matched56_pairwise_cosine",
        "crossfit_complement_to_screen_cosine",
    }
    candidate = _candidate()
    candidate["stability"][name] = copy.deepcopy(baseline)
    _reseal(candidate)
    gate = prereg.evaluate_candidate_v14b(candidate)
    condition = gate["conditions"][name]
    assert condition["comparison"] == (
        "strictly_greater" if strict else "not_lower"
    )
    assert condition["median_passed"] is (not strict)
    assert condition["worst_passed"] is (not strict)
    if not strict:
        below = _candidate()
        below["stability"][name]["worst"] = baseline["worst"] - 1e-6
        _reseal(below)
        assert prereg.evaluate_candidate_v14b(below)["conditions"][name][
            "worst_passed"
        ] is False


def test_v14b_candidate_self_hash_identities_and_aggregate_fail_closed():
    candidate = _candidate()
    candidate["content_sha256_before_self_field"] = "0" * 64
    with pytest.raises(RuntimeError, match="contract changed"):
        prereg.evaluate_candidate_v14b(candidate)

    candidate = _candidate()
    candidate["panel_identities"]["full_frame"] = "0" * 64
    _reseal(candidate)
    with pytest.raises(RuntimeError, match="contract changed"):
        prereg.evaluate_candidate_v14b(candidate)

    for key, value in (("l2_norm", float("nan")), ("l2_norm", 0.0),
                       ("nonzero_coordinate_count", 31)):
        candidate = _candidate()
        candidate["robust_aggregate"][key] = value
        _reseal(candidate)
        with pytest.raises(RuntimeError, match="robust aggregate"):
            prereg.evaluate_candidate_v14b(candidate)


def test_v14b_evidence_sampler_and_layer_plan_hashes_fail_closed(monkeypatch):
    monkeypatch.setattr(
        prereg, "V14A_NEGATIVE_EVIDENCE_FILE_SHA256_V14B", "0" * 64,
    )
    with pytest.raises(RuntimeError, match="evidence file identity"):
        prereg.load_v14a_negative_evidence_v14b()
    monkeypatch.undo()

    monkeypatch.setattr(prereg, "SAMPLER_FILE_SHA256_V14B", "0" * 64)
    with pytest.raises(RuntimeError, match="sampler implementation"):
        prereg.materialize_sampler_v14b()
    monkeypatch.undo()

    monkeypatch.setattr(prereg, "LAYER_PLAN_FILE_SHA256_V14B", "0" * 64)
    with pytest.raises(RuntimeError, match="layer plan file"):
        prereg.build_preregistration_v14b()


def test_v14b_firewall_pass_and_failure_decisions_never_open_eval_or_update():
    value = prereg.build_preregistration_v14b()
    assert value["promotion_gate"]["failure_decision"] == (
        "retain_v13_sampler_and_keep_all_eval_surfaces_closed"
    )
    assert value["promotion_gate"]["pass_decision"] == (
        "authorize_only_a_separately_preregistered_k2_alpha_zero_"
        "confirmation_on_a_fresh_32_direction_basis"
    )
    assert value["promotion_gate"][
        "pass_does_not_authorize_model_update_or_evaluation"
    ] is True
    assert value["firewall"]["v14a_status"] == (
        "closed_failed_gate_no_row_draw1_confirmation"
    )
    assert value["firewall"]["current_sampler"] == "retain_v13"


def test_v14b_preregistration_file_is_exact_rebuild():
    path = prereg.PREREGISTRATION_PATH_V14B
    frozen = json.loads(path.read_text())
    assert frozen == prereg.build_preregistration_v14b()
    assert frozen["content_sha256_before_self_field"] == prereg._canonical({
        key: value for key, value in frozen.items()
        if key != "content_sha256_before_self_field"
    })
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FILE_SHA256_V14B
    assert frozen["content_sha256_before_self_field"] == CONTENT_SHA256_V14B


FILE_SHA256_V14B = (
    "dcab1a49befebc8b67bbb9a80b866e876438a1e960e37953dff8c742b4e2c8ec"
)
CONTENT_SHA256_V14B = (
    "0963d1a8e18a97af949b94762292536c606279610c7c239e445d85e5be2c3216"
)
