#!/usr/bin/env python3

import json

import numpy as np
import pytest

import eggroll_es_paired_data_compat_preregistration_v33a as prereg_v33a


def test_v33a_preregistration_is_exact_and_deterministic():
    frozen = json.loads(prereg_v33a.PREREGISTRATION_PATH.read_text())
    built = prereg_v33a.build_preregistration()
    assert built == frozen
    assert prereg_v33a.file_sha256(prereg_v33a.PREREGISTRATION_PATH) == (
        "c83e11376922ac273b5b6496ef60126a2cdc6ae044f6a9ab05d9481dc539bcda"
    )
    assert frozen["content_sha256_before_self_field"] == (
        "f053c686721401d08b31f2619ba23511159e0a85cfbda2df606e0b00fa98bc61"
    )
    assert frozen["content_sha256_before_self_field"] == (
        prereg_v33a.canonical_sha256(prereg_v33a.without_self(frozen))
    )
    prereg_v33a.validate_sha_fields(frozen)


def test_v33a_fresh_basis_and_schedule_are_fully_pinned():
    frozen = prereg_v33a.build_preregistration()
    basis = frozen["frozen_recipe"]["perturbation_basis"]
    regenerated = [
        int(value) for value in np.random.default_rng(20261008).integers(
            0, 2**30, size=64, dtype=np.int64,
        )
    ]
    assert basis["direction_seeds"] == regenerated
    assert basis["direction_seed_list_sha256"] == (
        "4227e7c741175eb29f10c73b70f40e4442ebc6f2ca3d9f798dc7639cfe5a8e5f"
    )
    assert basis["signed_population_schedule_sha256"] == (
        "d85c0e18c3853485b1adc9e068c3a63346dd605230854e5ead14db71f8492011"
    )
    schedule = basis["signed_population_schedule"]
    assert len(schedule) == 32
    assert [item["sign"] for item in schedule] == ["plus", "minus"] * 16
    assert all(len(item["engine_direction_seeds"]) == 4 for item in schedule)
    assert basis["basis_content_sha256"] not in set(
        prereg_v33a.PRIOR_BASIS_HASHES.values()
    )


def test_v33a_bootstrap_draws_multiplicity_and_gate_are_predeclared():
    frozen = prereg_v33a.build_preregistration()
    bootstrap = frozen["analysis"]["bootstrap"]
    assert bootstrap["seed"] == 20261009
    assert bootstrap["repetitions"] == 50_000
    assert bootstrap["draw_plan_sha256"] == (
        "ef44acfe80d9afab5e17621eb62acc09572a6c3486f1f0a61dd6848ab6398b37"
    )
    assert bootstrap["draw_plan_sha256"] == prereg_v33a.bootstrap_draw_plan_sha256()
    assert bootstrap["multiplicity"] == "Bonferroni_over_12_endpoints"
    assert bootstrap["one_sided_quantile"] == pytest.approx(0.05 / 12)
    assert frozen["analysis"]["endpoint_count"] == 12
    assert frozen["analysis"]["candidate_minus_production_noninferiority_margin"] == 0.0
    assert frozen["analysis"]["all_endpoints_conjunctive"] is True
    assert frozen["analysis"][
        "all_12_observed_point_deltas_must_be_nonnegative"
    ] is True
    assert bootstrap["draw_plan_sha256"] != (
        "44569a4a813d0b736b6c093b7c2b5e1ffd4b1a353398b98cc14dabe4a718f7c2"
    )


def test_v33a_binds_205_unit_frame_and_keeps_nontrain_surfaces_closed():
    frozen = prereg_v33a.build_preregistration()
    joint = frozen["inputs"]["joint_frame"]
    assert joint["paired_units"] == 205
    assert joint["selected_units"] == 195
    assert joint["reserve_units"] == 10
    assert frozen["hardware"]["requests_per_engine_per_signed_wave"] == 390
    assert frozen["hardware"]["population_waves"] == 16
    assert frozen["hardware"]["synchronized_four_engine_signed_waves"] == 32
    assert frozen["hardware"]["engine_signed_direction_evaluations"] == 128
    assert frozen["hardware"]["perturbed_requests_all_engines"] == 49_920
    assert frozen["hardware"]["full_context_requests_all_engines"] == 4_680
    assert frozen["hardware"]["total_generation_requests"] == 54_600
    assert frozen["strict_train_only"]["selection_surface"] == (
        "paired_train_panels_only"
    )
    for key in (
        "validation_opened", "ood_opened", "heldout_opened", "benchmark_opened",
        "model_update_or_checkpoint_write", "dataset_promotion",
    ):
        assert frozen["strict_train_only"][key] is False
    assert frozen["required_runtime_adapter"]["runtime_launch_authorized"] is False
    assert frozen["promotion_gate"][
        "pass_does_not_authorize_dataset_promotion_update_or_evaluation"
    ] is True
    baseline = frozen["audited_prior_aggregate_evidence"]
    assert baseline["version"] == "v25a_r1_promising_unconfirmed"
    assert baseline["commit"] == "1fb518c2c3eed467eed47fa4a9b498cfdca05c24"
    assert baseline["v25a_global_gate_passed"] is False
    assert baseline["v25a_confirmation_authorized"] is False
    assert baseline["v25a_not_reinterpreted_as_pass_or_confirmation"] is True
    assert baseline["all_12_conjunctive_familywise_endpoints_preserved"] is True
    assert baseline["perturbation_or_bootstrap_draw_reuse"] is False
    assert baseline["optimization_panels"] == 3
    assert baseline["untouched_train_screens"] == 2
    assert baseline["prior_population_size"] == 32
    assert baseline["new_population_size"] == 64
    assert baseline["bootstrap_repetitions"] == 50_000
    assert baseline[
        "posthoc_power_margin_multiplicity_or_threshold_change"
    ] is False


def test_v33a_exclusive_write_rejects_reuse(tmp_path, monkeypatch):
    output = tmp_path / "prereg.json"
    monkeypatch.setattr(prereg_v33a, "PREREGISTRATION_PATH", output)
    prereg_v33a.exclusive_write(output, {"bound": True})
    with pytest.raises(ValueError, match="already exists"):
        prereg_v33a.exclusive_write(output, {"bound": True})
