import copy
import json

import pytest

import eggroll_es_v401_replacement_fraction_preregistration_v34b as prereg


def test_exact_fresh_basis_and_synchronized_schedule():
    value = prereg.build_preregistration()
    basis = value["frozen_recipe"]["perturbation_basis"]
    assert len(basis["direction_seeds"]) == len(set(basis["direction_seeds"])) == 64
    assert basis["direction_seed_list_sha256"] == "e1e45bc1965360d78a9a367c884b29fbd0c09bf00a2f83c36571fe09dc41bdf5"
    assert basis["basis_content_sha256"] not in prereg.PRIOR_BASIS_CONTENT_SHA256.values()
    schedule = basis["signed_population_schedule"]
    assert len(schedule) == 32
    assert all(len(item["engine_direction_seeds"]) == 4 for item in schedule)
    assert sum(item["resident_source_order"][0] == "production" for item in schedule) == 16


def test_fraction_algebra_and_budget_add_no_requests():
    value = prereg.build_preregistration()
    hpo = value["replacement_fraction_hpo"]
    budget = value["hardware_and_budget"]
    assert hpo["fractions_in_fixed_test_order"] == [0.05, 0.1, 0.2, 0.4, 1.0]
    assert hpo["fraction_model_requests"] == budget["fraction_specific_requests"] == 0
    assert budget["perturbed_requests"] == 32 * 4 * 390 == 49_920
    assert budget["full_context_requests"] == 3 * 4 * 390 == 4_680
    assert budget["total_generation_requests"] == 54_600


def test_fixed_sequence_and_multiplicity_are_frozen():
    value = prereg.build_preregistration()
    analysis = value["analysis"]
    gate = value["fixed_sequence_gate"]
    assert len(analysis["endpoints"]) == 12
    assert analysis["bootstrap"]["within_fraction_bonferroni_quantile"] == 0.05 / 12
    assert analysis["bootstrap"]["draw_plan_sha256"] == "458d4bdaf9e8f990258712e561699c630ab8e1091f9919979492081c283d5dec"
    assert gate["stop_at_first_failure"] is True
    assert gate["fractions_after_first_failure_are_not_authorized_or_interpreted"] is True
    assert gate["direct_dataset_promotion_model_update_checkpoint_write_or_eval_authorized"] is False


def test_preregistration_is_train_only_and_content_free():
    value = prereg.build_preregistration()
    assert value["content_sha256_before_self_field"] == prereg.canonical_sha256(
        prereg._without_self(value)
    )
    firewall = value["strict_train_only_firewall"]
    assert firewall["validation_heldout_ood_eval_or_benchmark_opened"] is False
    assert firewall["dataset_rows_questions_answers_or_document_text_persisted"] is False
    encoded = json.dumps(value).lower()
    for forbidden in ('"question"', '"answer"', '"prompt"', '"unit_scores"'):
        assert forbidden not in encoded


def test_mutating_fraction_order_changes_seal():
    value = prereg.build_preregistration()
    changed = copy.deepcopy(value)
    changed["replacement_fraction_hpo"]["fractions_in_fixed_test_order"].reverse()
    assert prereg.canonical_sha256(prereg._without_self(changed)) != value[
        "content_sha256_before_self_field"
    ]


def test_exclusive_write_rejects_wrong_or_existing_path(tmp_path):
    value = prereg.build_preregistration()
    with pytest.raises(ValueError):
        prereg.exclusive_write(tmp_path / "wrong.json", value)
    if prereg.OUTPUT_PATH.exists():
        with pytest.raises(ValueError):
            prereg.exclusive_write(prereg.OUTPUT_PATH, value)
