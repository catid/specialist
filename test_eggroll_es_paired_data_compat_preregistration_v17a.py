#!/usr/bin/env python3
"""Offline tests for the V17A paired data-only preregistration."""

import copy
import json

import pytest

import eggroll_es_paired_data_compat_preregistration_v17a as prereg


def synthetic_candidate(delta=0.01, lcb=0.001):
    production = {
        name: 0.4 + 0.01 * index
        for index, name in enumerate(prereg.ENDPOINT_CONTRACT_V17A)
    }
    candidate = {name: value + delta for name, value in production.items()}
    value = {
        "schema": "eggroll-es-paired-data-compat-summary-v17a",
        "experiment_name": prereg.EXPERIMENT_NAME_V17A,
        "alpha": 0.0,
        "sigma": 0.0003,
        "model_update_applied": False,
        "evaluation_surfaces_opened": False,
        "frame_content_sha256": prereg.FRAME_CONTENT_SHA256_V17A,
        "perturbation_basis_sha256": (
            prereg.anchor_v13.PERTURBATION_BASIS_SHA256_V13
        ),
        "runtime_integrity": {
            "all_four_engines_every_signed_wave": True,
            "fixed_side_batch_identity_every_direction_and_sign": True,
            "same_resident_perturbation_both_versions": True,
            "alternating_version_order_complete": True,
            "exact_reference_restoration_passed": True,
            "pre_post_base_probes_equal_both_versions": True,
            "population_boundary_audit_passed": True,
            "tokenizer_and_prompt_logprob_contract_passed": True,
            "all_integrity_audits_passed": True,
        },
        "versions": {
            "production": {
                "all_panel_spreads_nonzero": True,
                "endpoint_values": production,
                "compact_estimator_sha256": "1" * 64,
            },
            "candidate_v283": {
                "all_panel_spreads_nonzero": True,
                "endpoint_values": candidate,
                "compact_estimator_sha256": "2" * 64,
            },
        },
        "paired_bootstrap": {
            "seed": prereg.BOOTSTRAP_SEED_V17A,
            "repetitions": prereg.BOOTSTRAP_REPETITIONS_V17A,
            "one_sided_quantile": (
                prereg.FAMILYWISE_ALPHA_V17A
                / len(prereg.ENDPOINT_CONTRACT_V17A)
            ),
            "endpoints": {
                name: {
                    "candidate_minus_production": delta,
                    "familywise_lcb": lcb,
                    "noninferiority_margin": 0.0,
                }
                for name in prereg.ENDPOINT_CONTRACT_V17A
            },
        },
        "cross_dataset_direction_similarity_diagnostic": {
            "used_for_gate": False,
            "content_sha256": "3" * 64,
        },
        "persisted_response_vectors_or_row_content": False,
    }
    value["content_sha256_before_self_field"] = prereg.canonical_sha256(value)
    return value


def reseal(value):
    value["content_sha256_before_self_field"] = prereg.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })
    return value


def test_v17a_preregistration_binds_corrected_frame_full_answer_and_no_update():
    value = prereg.build_preregistration_v17a()
    assert json.loads(prereg.PREREGISTRATION_PATH_V17A.read_text()) == value
    assert value["inputs"]["joint_frame"] == {
        "path": str(prereg.FRAME_PATH_V17A),
        "file_sha256": prereg.FRAME_FILE_SHA256_V17A,
        "content_sha256": prereg.FRAME_CONTENT_SHA256_V17A,
        "joint_components": 276,
        "paired_units": 195,
        "selected_units": 190,
        "reserve_units": 5,
        "paired_strata": {
            "safety_consent": 70, "technique": 41,
            "equipment_material": 13, "resources_general": 71,
        },
    }
    scoring = value["scoring"]
    assert scoring["prompt_contains_full_gold_answer"] is True
    assert scoring["scored_positions"] == "all_aligned_answer_tokens_only"
    assert scoring["max_tokens"] == 1
    assert scoring["max_tokens_role"] == (
        "dummy_generation_trigger_not_answer_cap"
    )
    assert scoring["token_length_audit"][
        "over_frozen_1024_total_token_cap_count"
    ] == {"production": 0, "candidate_v283": 0}
    assert value["frozen_recipe"]["alpha"] == 0.0
    assert value["frozen_recipe"]["model_update_allowed"] is False
    assert value["required_runtime_adapter"][
        "runtime_not_authorized_by_this_preregistration_commit"
    ] is True
    bootstrap = value["analysis"]["bootstrap"]
    assert bootstrap["preserve_per_panel_stratum_counts"] == [14, 8, 2, 14]
    assert bootstrap["reserve_units_used"] is False
    assert bootstrap["persist_per_unit_scores_or_bootstrap_replicates"] is False
    assert bootstrap["recompute_each_replicate"] == (
        "Horvitz_Thompson_panel_scores_then_32_coefficients_then_"
        "all_nonlinear_median_and_worst_endpoints"
    )


def test_v17a_gate_requires_all_twelve_observed_and_bootstrap_endpoints():
    passed = prereg.evaluate_candidate_v17a(synthetic_candidate())
    assert passed["eligible_for_separate_v17b_preregistration"] is True
    assert all(passed["observed_candidate_not_below_production"].values())
    assert all(passed["paired_noninferiority_results"].values())
    assert passed["eligible_for_dataset_promotion"] is False
    assert passed["eligible_for_model_update"] is False
    assert passed["eligible_to_open_evaluation"] is False

    observed_failure = synthetic_candidate()
    name = next(iter(prereg.ENDPOINT_CONTRACT_V17A))
    observed_failure["versions"]["candidate_v283"]["endpoint_values"][name] = 0.0
    observed_failure["paired_bootstrap"]["endpoints"][name][
        "candidate_minus_production"
    ] = -observed_failure["versions"]["production"]["endpoint_values"][name]
    reseal(observed_failure)
    failed = prereg.evaluate_candidate_v17a(observed_failure)
    assert failed["eligible_for_separate_v17b_preregistration"] is False
    assert failed["failure_decision"] == "retain_production_dataset_and_v13_recipe"

    lcb_failure = synthetic_candidate()
    lcb_failure["paired_bootstrap"]["endpoints"][name]["familywise_lcb"] = -1e-9
    reseal(lcb_failure)
    failed = prereg.evaluate_candidate_v17a(lcb_failure)
    assert failed["eligible_for_separate_v17b_preregistration"] is False


def test_v17a_cross_dataset_similarity_cannot_change_gate():
    first = synthetic_candidate()
    second = copy.deepcopy(first)
    second["cross_dataset_direction_similarity_diagnostic"][
        "content_sha256"
    ] = "9" * 64
    reseal(second)
    assert prereg.evaluate_candidate_v17a(first)[
        "eligible_for_separate_v17b_preregistration"
    ] is True
    assert prereg.evaluate_candidate_v17a(second)[
        "eligible_for_separate_v17b_preregistration"
    ] is True


def test_v17a_rejects_diagnostic_extra_content_and_non_lowercase_sha_fields():
    diagnostic_extra = synthetic_candidate()
    diagnostic_extra["cross_dataset_direction_similarity_diagnostic"][
        "raw_similarity"
    ] = 0.99
    reseal(diagnostic_extra)
    with pytest.raises(RuntimeError, match="diagnostic contract"):
        prereg.evaluate_candidate_v17a(diagnostic_extra)

    for bad_digest in ("A" * 64, "g" * 64, "0" * 63):
        bad_sha = synthetic_candidate()
        bad_sha["versions"]["production"][
            "compact_estimator_sha256"
        ] = bad_digest
        reseal(bad_sha)
        with pytest.raises(RuntimeError, match="SHA-256|stability contract"):
            prereg.evaluate_candidate_v17a(bad_sha)


def test_v17a_rejects_update_surface_integrity_and_contract_tampering():
    for mutation in ("update", "integrity", "extra"):
        value = synthetic_candidate()
        if mutation == "update":
            value["model_update_applied"] = True
        elif mutation == "integrity":
            value["runtime_integrity"][
                "fixed_side_batch_identity_every_direction_and_sign"
            ] = False
        else:
            value["unexpected"] = True
        reseal(value)
        with pytest.raises(RuntimeError, match="contract|integrity"):
            prereg.evaluate_candidate_v17a(value)
