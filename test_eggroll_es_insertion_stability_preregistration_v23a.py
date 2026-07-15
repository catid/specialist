#!/usr/bin/env python3
"""CPU-only fail-closed tests for V23A insertion preregistration."""

import copy
import hashlib
import json

import pytest

import eggroll_es_insertion_stability_preregistration_v23a as prereg_v23a


EXPECTED_CONTENT_SHA256 = (
    "de43c14ae4fc325dfd23351fd1021dd029282b93f6cb6a17c4073c2ac72cc281"
)
EXPECTED_FILE_SHA256 = (
    "6dfdf59ed6e9be494fdbd2450eca296d2e334bfac54b46fb59309f7be62ccf57"
)
EXPECTED_PLAN_FILE_SHA256 = {
    "base_middle_late": "4dca69212f2eee1c5882a7f3de944e3dadd8de94b42e58e7d7495547e8b1c747",
    "insert_front_e005": "4c0f451545f01ec53cc17800e24f331d1038bbc3ce96fe352a9cc2b96c822e29",
    "insert_middle_e005": "9f343cb136a5d4883ae81878ecec005e028b7f5e492ca0cc64b1f9e1945c112a",
    "insert_back_e005": "21a0100d2bf729ce5ce88ea83ea668086cfb512ff5684413050d03d796c7820e",
}


def load_preregistration():
    return json.loads(prereg_v23a.OUTPUT_PATH_V23A.read_text(encoding="utf-8"))


def reseal(value):
    value["content_sha256_before_self_field"] = prereg_v23a.canonical_sha256(
        prereg_v23a._without_self(value)
    )


def test_v23a_preregistration_is_exact_deterministic_and_persisted():
    built = prereg_v23a.build_preregistration_v23a()
    persisted = load_preregistration()
    assert built == persisted
    assert built["content_sha256_before_self_field"] == EXPECTED_CONTENT_SHA256
    assert prereg_v23a.file_sha256(prereg_v23a.OUTPUT_PATH_V23A) == EXPECTED_FILE_SHA256
    for arm, expected in EXPECTED_PLAN_FILE_SHA256.items():
        path = built["arms"][arm]["layer_plan"]["path"]
        assert prereg_v23a.file_sha256(path) == expected


def test_v23a_basis_is_fresh_exact_and_counterbalanced():
    basis = prereg_v23a.perturbation_basis_v23a()
    schedule = prereg_v23a.signed_wave_schedule_v23a()
    assert len(basis["direction_seeds"]) == len(set(basis["direction_seeds"])) == 32
    assert prereg_v23a.canonical_sha256(basis) == (
        prereg_v23a.PERTURBATION_BASIS_SHA256_V23A
    )
    assert prereg_v23a.canonical_sha256(basis["direction_seeds"]) == (
        prereg_v23a.PERTURBATION_SEED_LIST_SHA256_V23A
    )
    assert prereg_v23a.canonical_sha256(schedule) == (
        prereg_v23a.SIGNED_WAVE_SCHEDULE_SHA256_V23A
    )
    assert prereg_v23a.PERTURBATION_BASIS_SHA256_V23A not in set(
        prereg_v23a.PRIOR_BASIS_CONTENT_SHA256_V23A.values()
    )
    assert len(schedule) == 64
    for index in range(32):
        pair = [item for item in schedule if item["direction_index"] == index]
        assert {item["sign"] for item in pair} == {"plus", "minus"}
        assert len({item["direction_seed"] for item in pair}) == 1


def test_v23a_four_fixed_rank_arms_have_exact_models_motifs_and_equal_capacity():
    value = load_preregistration()
    assert value["runtime"]["engine_arm_mapping"] == {
        "0": "base_middle_late", "1": "insert_front_e005",
        "2": "insert_middle_e005", "3": "insert_back_e005",
    }
    assert {
        arm: item["target_layers"] for arm, item in value["arms"].items()
    } == {
        "base_middle_late": [20, 21, 22, 23],
        "insert_front_e005": [4, 5, 6, 7],
        "insert_middle_e005": [20, 21, 22, 23],
        "insert_back_e005": [40, 41, 42, 43],
    }
    shape_hashes = {
        item["layer_plan"]["logical_shape_order_sha256"]
        for item in value["arms"].values()
    }
    assert len(shape_hashes) == 1
    assert all(
        item["layer_plan"]["num_units"] == 35
        and item["layer_plan"]["selected_element_count"] == 142_999_552
        for item in value["arms"].values()
    )


def test_v23a_exact_v13_panels_and_request_accounting_are_frozen():
    value = load_preregistration()
    panel = value["panel_contract"]
    runtime = value["runtime"]
    assert panel["optimization_panels"] == [
        "optimization_0", "optimization_1", "optimization_2"
    ]
    assert panel["untouched_train_screen_panels"] == [
        "train_screen_0", "train_screen_1"
    ]
    assert panel["rows_per_panel"] == 56 and panel["panel_count"] == 5
    assert runtime["requests_per_engine_per_signed_wave"] == 280
    assert runtime["requests_all_engines_per_signed_wave"] == 1_120
    assert runtime["requests_per_engine_all_signed_waves"] == 17_920
    assert runtime["requests_all_engines_all_signed_waves"] == 71_680
    assert runtime["unperturbed_reference_requests_all_engines"] == 1_120
    assert runtime["all_four_gpus_score_every_signed_wave"] is True


def test_v23a_reference_compatibility_and_global_multiplicity_are_conjunctive():
    value = load_preregistration()
    reference = value["reference_compatibility"]
    analysis = value["analysis"]
    assert reference["endpoints"] == list(prereg_v23a.REFERENCE_ENDPOINTS_V23A)
    assert reference["zero_noninferiority_margin"] is True
    assert reference["location_cannot_advance_if_any_reference_endpoint_fails"] is True
    assert analysis["candidate_location_count"] == 3
    assert analysis["endpoint_count_per_location"] == 16
    assert analysis["family_hypothesis_count"] == 48
    assert analysis["one_sided_familywise_quantile"] == 0.05 / 48
    assert analysis["multiplicity_covers_all_three_locations_and_all_endpoints"] is True


def test_v23a_pass_can_only_authorize_permuted_fresh_train_confirmation():
    value = load_preregistration()
    initial = value["runtime"]["engine_arm_mapping"]
    confirmation = value["gate"]["confirmation_engine_arm_mapping"]
    assert set(initial.values()) == set(confirmation.values())
    assert all(initial[rank] != confirmation[rank] for rank in initial)
    assert value["gate"]["pass_authority"] == (
        "authorize_only_separate_fresh_basis_train_only_confirmation"
    )
    assert value["gate"]["no_location_pass_decision"] == (
        "retain_v13_base_middle_late_recipe"
    )
    assert all(value["gate"][key] is False for key in (
        "direct_model_update_authorized", "checkpoint_write_authorized",
        "evaluation_authorized", "dataset_promotion_authorized",
    ))


def test_v23a_inputs_are_aggregate_only_and_old_insertion_artifacts_are_excluded():
    value = load_preregistration()
    assert not (
        prereg_v23a.FORBIDDEN_CONTENT_KEYS_V23A
        & set(prereg_v23a._recursive_keys(value))
    )
    assert all(value["excluded_inputs"].values())
    paths = json.dumps({
        "model_seal": value["model_seal"], "arms": value["arms"],
        "aggregate_evidence": value["aggregate_evidence"],
    }).lower()
    for forbidden in ("es_location", "gen20", "gen30", "eval_reports", ".jsonl"):
        assert forbidden not in paths


@pytest.mark.parametrize("mutation", [
    "extra", "basis", "mapping", "multiplicity", "reference", "authority",
])
def test_v23a_rejects_resealed_preregistration_tampering(mutation):
    value = copy.deepcopy(load_preregistration())
    if mutation == "extra":
        value["harmless"] = 1
    elif mutation == "basis":
        value["fresh_basis"]["basis_content_sha256"] = "0" * 64
    elif mutation == "mapping":
        value["runtime"]["engine_arm_mapping"]["0"] = "insert_front_e005"
    elif mutation == "multiplicity":
        value["analysis"]["family_hypothesis_count"] = 16
    elif mutation == "reference":
        value["reference_compatibility"][
            "location_cannot_advance_if_any_reference_endpoint_fails"
        ] = False
    else:
        value["authority"]["model_update_allowed"] = True
    reseal(value)
    with pytest.raises(RuntimeError, match="preregistration changed"):
        prereg_v23a.validate_preregistration_v23a(value)


def test_v23a_output_writer_is_exclusive(tmp_path):
    raw = b"{}\n"
    path = tmp_path / "artifact.json"
    prereg_v23a._exclusive_write(path, raw)
    assert path.read_bytes() == raw
    with pytest.raises(RuntimeError, match="already exists"):
        prereg_v23a._exclusive_write(path, raw)
