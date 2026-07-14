#!/usr/bin/env python3

import copy
import hashlib
import json

import pytest

import eggroll_es_hierarchical_preregistration_v14a as prereg


def _file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(delta=0.01):
    stability = copy.deepcopy(prereg.BASELINE_STABILITY_V14A)
    for name in (
        "matched56_pairwise_cosine",
        "crossfit_complement_to_screen_cosine",
    ):
        stability[name]["median"] += delta
        stability[name]["worst"] += delta
    value = {
        "schema": "eggroll-es-full-frame-matched56-summary-v14a",
        "experiment_name": prereg.EXPERIMENT_NAME_V14A,
        "alpha": 0.0,
        "model_update_applied": False,
        "validation_ood_or_heldout_used": False,
        "perturbation_basis_sha256": prereg.PERTURBATION_BASIS_SHA256_V14A,
        "panel_identities": {
            "full_frame": prereg.FULL_FRAME_IDENTITY_V14A[
                "ordered_row_identity_sha256"
            ],
            **{
                name: identity["ordered_row_identity_sha256"]
                for name, identity in prereg.PANEL_IDENTITIES_V14A.items()
            },
        },
        "stability": stability,
        "all_panel_spreads_nonzero": True,
        "robust_aggregate": {
            "coefficient_sha256": "a" * 64,
            "l2_norm": 4.0,
            "nonzero_coordinate_count": 32,
        },
    }
    value["content_sha256_before_self_field"] = prereg._canonical(value)
    return value


def _reseal(value):
    value["content_sha256_before_self_field"] = prereg._canonical({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def test_v14a_compact_evidence_is_exact_and_contains_no_response_vectors():
    evidence = prereg.load_evidence_v14a()
    assert evidence["contains_response_vectors_or_row_content"] is False
    assert evidence["contains_validation_ood_or_heldout_content"] is False
    assert evidence["stability"]["optimization_pairwise_cosine"] == {
        "count": 3,
        "median": 0.47411088498906484,
        "worst": 0.3900621868364503,
    }

    forbidden_keys = {"responses", "coefficients", "questions", "answers"}

    def walk(value):
        if isinstance(value, dict):
            assert not forbidden_keys.intersection(value)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(evidence)


def test_v14a_v12_negative_evidence_closes_candidate_without_eval_surface():
    evidence = prereg.load_v12_negative_evidence_v14a()
    assert evidence["decision"] == (
        "close_v12_candidate_without_confirmation_or_release"
    )
    assert evidence["aggregate_failure"] == {
        "all_positive_anchor_lcbs_negative": True,
        "all_positive_screen_lcbs_negative": True,
        "benchmark_content_opened_before_candidate_seal": False,
        "candidate_seal_written": False,
        "eligible_positive_alpha_count": 0,
        "heldout_opened": False,
        "positive_alpha_count": 2,
    }
    assert evidence["contains_response_documents_or_row_content"] is False
    assert evidence["contains_validation_ood_or_heldout_content"] is False


def test_v14a_preregistration_file_is_exact_rebuild():
    path = prereg.PREREGISTRATION_PATH_V14A
    assert _file_sha256(path) == (
        "d27052ee26d9ba5dd4383491b3d093d0a2f9469ddb4a073909a2b6590e0cba3e"
    )
    frozen = json.loads(path.read_text())
    assert frozen == prereg.build_preregistration_v14a()
    assert frozen["content_sha256_before_self_field"] == (
        "e610c4bd83449b6b9cb3a0055f8e099ebae32ff6827aa64c6521d74705bda59d"
    )
    assert frozen["status"] == "preregistered_not_launch_authorized"
    assert frozen["required_runtime_adapter"]["status"] == "not_yet_implemented"
    assert frozen["firewall"]["v11f_status"] == (
        "immutable_failed_superseded_by_completed_v11g"
    )
    assert frozen["firewall"]["v12_candidate_status"] == (
        "closed_no_eligible_alpha_no_confirmation_no_release"
    )
    assert frozen["sampling"]["generation_prompt_count_per_direction_and_sign"] == 310
    assert frozen["sampling"]["full_frame_document_selection_variance"] == 0.0


def test_v14a_full_frame_and_matched_panels_are_exact_and_disjoint():
    _rows, full_frame, panels = prereg.materialize_panels_v14a()
    documents = [
        item["document_sha256"]
        for panel in panels.values() for item in panel["items"]
    ]
    assert len(documents) == len(set(documents)) == 280
    assert full_frame["rows"] == 310
    assert len({item["document_sha256"] for item in full_frame["items"]}) == 310
    assert [
        panels[name]["document_allocation_iteration"] for name in panels
    ] == list(range(5))
    assert all(panel["rows"] == 56 for panel in panels.values())
    assert all(
        sum(item["equal_document_ht_weight"] for item in panel["items"])
        == pytest.approx(310.0)
        for panel in panels.values()
    )


def test_v14a_numeric_gate_passes_only_strict_cosine_improvement():
    candidate = _candidate()
    gate = prereg.evaluate_candidate_v14a(candidate)
    assert gate["eligible_for_train_only_sampler_adoption"] is True
    assert gate["eligible_for_model_update"] is False

    tied = _candidate(delta=0.0)
    tied_gate = prereg.evaluate_candidate_v14a(tied)
    assert tied_gate["eligible_for_train_only_sampler_adoption"] is False
    assert tied_gate["conditions"]["matched56_pairwise_cosine"][
        "comparison"
    ] == "strictly_greater"


def test_v14a_numeric_gate_rejects_any_sign_or_screen_regression():
    candidate = _candidate()
    candidate["stability"]["matched56_pairwise_sign_agreement"][
        "worst"
    ] -= 0.01
    _reseal(candidate)
    gate = prereg.evaluate_candidate_v14a(candidate)
    assert gate["eligible_for_train_only_sampler_adoption"] is False

    candidate = _candidate()
    candidate["stability"]["crossfit_complement_to_screen_cosine"]["worst"] = (
        prereg.BASELINE_STABILITY_V14A[
            "crossfit_complement_to_screen_cosine"
        ]["worst"]
    )
    _reseal(candidate)
    gate = prereg.evaluate_candidate_v14a(candidate)
    assert gate["eligible_for_train_only_sampler_adoption"] is False


def test_v14a_candidate_contract_rejects_update_or_panel_tampering():
    candidate = _candidate()
    candidate["model_update_applied"] = True
    _reseal(candidate)
    with pytest.raises(RuntimeError, match="contract changed"):
        prereg.evaluate_candidate_v14a(candidate)

    candidate = _candidate()
    candidate["panel_identities"]["optimization_0"] = "0" * 64
    _reseal(candidate)
    with pytest.raises(RuntimeError, match="contract changed"):
        prereg.evaluate_candidate_v14a(candidate)
