#!/usr/bin/env python3
"""Contract tests for the train-only V24A hybrid backend preregistration."""

import copy
import json

import pytest

import eggroll_es_hybrid_backend_preregistration_v24a as prereg
import eggroll_es_insertion_stability_preregistration_v23a as prereg_v23a


@pytest.fixture(scope="module")
def persisted():
    value = json.loads(prereg.PREREGISTRATION_PATH_V24A.read_text())
    return prereg.validate_preregistration_v24a(value)


def test_v24a_persisted_preregistration_is_exact_rebuild(persisted):
    assert persisted == prereg.build_preregistration_v24a()
    assert persisted["schema"] == prereg.SCHEMA_V24A
    assert persisted["content_sha256_before_self_field"] == prereg.canonical_sha256({
        key: value for key, value in persisted.items()
        if key != "content_sha256_before_self_field"
    })
    assert prereg.file_sha256(prereg.PREREGISTRATION_PATH_V24A) == (
        "20b65b6d8c849580782be9a348a6c8ed705135058e91c21d7b26546e51fb6756"
    )


def test_v24a_binds_complete_real_hybrid_audit_and_models(persisted):
    audit = persisted["hybrid_checkpoint_audit"]
    assert audit == {
        "path": "/home/catid/specialist/experiments/eggroll_es_hpo/S6_V24_HYBRID_CHECKPOINT_AUDIT.json",
        "file_sha256": prereg.AUDIT_FILE_SHA256_V24A,
        "content_sha256": prereg.AUDIT_CONTENT_SHA256_V24A,
        "selected_partition_exact_bf16": True,
        "retained_partition_exact_fp8": True,
        "unaffected_files_exact_hardlinks": True,
        "selected_unit_count": 35,
        "selected_element_count": 142_999_552,
    }
    contract = persisted["model_contract"]
    assert contract["same_exact_35_unit_bf16_selected_partition"] is True
    assert contract["only_frozen_complement_backend_differs"] is True
    assert contract["bf16"]["config_sha256"] == prereg.BF16_CONFIG_SHA256_V24A
    assert contract["hybrid"]["config_sha256"] == prereg.HYBRID_CONFIG_SHA256_V24A
    assert contract["hybrid"]["overlay_sha256"] == prereg.HYBRID_OVERLAY_SHA256_V24A


def test_v24a_fresh_basis_is_unique_deterministic_and_distinct_from_v23(persisted):
    seeds = prereg.perturbation_seeds_v24a()
    schedule = prereg.signed_wave_schedule_v24a()
    assert len(seeds) == len(set(seeds)) == 32
    assert len(schedule) == 64
    for direction in range(32):
        expected = ["plus", "minus"] if direction % 2 == 0 else ["minus", "plus"]
        assert [schedule[2 * direction + offset]["sign"] for offset in range(2)] == expected
        assert all(schedule[2 * direction + offset]["direction_seed"] == seeds[direction]
                   for offset in range(2))
    assert persisted["fresh_basis"]["direction_seed_list_sha256"] == prereg.canonical_sha256(seeds)
    assert persisted["fresh_basis"]["signed_wave_schedule_sha256"] == prereg.canonical_sha256(schedule)
    prior = json.loads(
        prereg_v23a.OUTPUT_PATH_V23A.read_text(encoding="utf-8")
    )
    assert persisted["fresh_basis"]["basis_content_sha256"] != (
        prior["fresh_basis"]["basis_content_sha256"]
    )
    assert seeds != prereg_v23a.perturbation_basis_v23a()["direction_seeds"]
    assert persisted["fresh_basis"]["distinct_from_every_listed_prior_basis"] is True
    assert persisted["fresh_basis"]["basis_content_sha256"] not in set(
        persisted["fresh_basis"]["prior_basis_content_sha256"].values()
    )


def test_v24a_uses_two_matched_pairs_and_all_four_gpus_every_wave(persisted):
    assert set(persisted["arms"]) == set(prereg.ARM_ORDER_V24A)
    assert persisted["pairing"]["pairs"] == {
        "pair_a": {"bf16": "bf16_a", "hybrid": "hybrid_a"},
        "pair_b": {"bf16": "bf16_b", "hybrid": "hybrid_b"},
    }
    runtime = persisted["runtime"]
    assert runtime["engine_arm_mapping"] == {
        "0": "bf16_a", "1": "hybrid_a", "2": "bf16_b", "3": "hybrid_b",
    }
    assert runtime["gpu_ids"] == [0, 1, 2, 3]
    assert runtime["engine_count"] == 4 and runtime["tp_per_engine"] == 1
    assert runtime["all_four_gpus_score_every_signed_wave"] is True
    assert runtime["perturbed_requests_all_engines"] == 71_680
    assert runtime["unperturbed_pre_post_requests_all_engines"] == 2_240
    assert runtime["total_generation_requests"] == 73_920


def test_v24a_multiplicity_covers_every_pair_quality_and_speed_endpoint(persisted):
    analysis = persisted["analysis"]
    assert len(analysis["quality_endpoint_thresholds"]) == 16
    assert analysis["quality_endpoint_count_per_pair"] == 16
    assert analysis["pair_count"] == 2
    assert analysis["speed_endpoint_count"] == 2
    assert analysis["family_hypothesis_count"] == 2 * 16 + 2 == 34
    assert analysis["one_sided_familywise_quantile"] == pytest.approx(0.05 / 34)
    assert analysis["bootstrap_repetitions"] == 50_000
    assert analysis["paired_direction_and_stratified_train_row_bootstrap"] is True
    assert analysis["speedup_threshold_per_pair"] == 1.05
    assert analysis["memory_reduction_threshold_per_pair"] == 0.40


def test_v24a_gate_only_authorizes_separate_mapping_reversed_confirmation(persisted):
    gate = persisted["gate"]
    assert gate["confirmation_engine_backend_mapping"] == {
        "0": "hybrid", "1": "bf16", "2": "hybrid", "3": "bf16",
    }
    assert gate["confirmation_basis_seed"] == 20260901
    assert gate["confirmation_bootstrap_seed"] == 20260902
    assert gate["pass_authority"] == (
        "authorize_only_separate_fresh_basis_train_only_mapping_reversed_confirmation"
    )
    assert all(gate[key] is False for key in (
        "direct_backend_substitution_authorized", "model_update_authorized",
        "checkpoint_write_authorized", "evaluation_authorized",
        "dataset_promotion_authorized",
    ))
    assert all(persisted["authority"][key] is False for key in (
        "model_update_allowed", "checkpoint_write_allowed", "evaluation_allowed",
        "dataset_promotion_allowed", "backend_adoption_allowed",
    ))


def test_v24a_rejects_any_preregistration_mutation(persisted):
    changed = copy.deepcopy(persisted)
    changed["analysis"]["speedup_threshold_per_pair"] = 1.0
    with pytest.raises(RuntimeError, match="content changed"):
        prereg.validate_preregistration_v24a(changed)
