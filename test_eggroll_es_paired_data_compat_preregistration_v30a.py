#!/usr/bin/env python3

import json

import numpy as np
import pytest

import eggroll_es_paired_data_compat_preregistration_v30a as prereg_v30a


def test_v30a_preregistration_is_exact_and_deterministic():
    frozen = json.loads(prereg_v30a.PREREGISTRATION_PATH.read_text())
    built = prereg_v30a.build_preregistration()
    assert built == frozen
    assert frozen["content_sha256_before_self_field"] == (
        "dbb0f0bd704cc7c598856f94045550c30b11c2ab3ca5881abcf3b44fab5b1423"
    )
    assert frozen["content_sha256_before_self_field"] == (
        prereg_v30a.canonical_sha256(prereg_v30a.without_self(frozen))
    )
    prereg_v30a.validate_sha_fields(frozen)


def test_v30a_fresh_basis_and_schedule_are_fully_pinned():
    frozen = prereg_v30a.build_preregistration()
    basis = frozen["frozen_recipe"]["perturbation_basis"]
    regenerated = [
        int(value) for value in np.random.default_rng(20261003).integers(
            0, 2**30, size=32, dtype=np.int64,
        )
    ]
    assert basis["direction_seeds"] == regenerated
    assert basis["direction_seed_list_sha256"] == (
        "29d165336769bbd89ae3eebf56c8d74f5b4fe603b226506eed1c632dd630b7af"
    )
    assert basis["signed_population_schedule_sha256"] == (
        "410d13c940d8f83451c3744ae449854c65d90a7f9c5e46191a9339968d6c91e1"
    )
    schedule = basis["signed_population_schedule"]
    assert len(schedule) == 16
    assert [item["sign"] for item in schedule] == ["plus", "minus"] * 8
    assert all(len(item["engine_direction_seeds"]) == 4 for item in schedule)
    assert basis["basis_content_sha256"] not in set(
        prereg_v30a.PRIOR_BASIS_HASHES.values()
    )


def test_v30a_bootstrap_draws_multiplicity_and_gate_are_predeclared():
    frozen = prereg_v30a.build_preregistration()
    bootstrap = frozen["analysis"]["bootstrap"]
    assert bootstrap["seed"] == 20261004
    assert bootstrap["repetitions"] == 50_000
    assert bootstrap["draw_plan_sha256"] == (
        "dbea000043a713114c150d07e11b813d1cbf00dcca9c418527d9a81c94b94ad5"
    )
    assert bootstrap["draw_plan_sha256"] == prereg_v30a.bootstrap_draw_plan_sha256()
    assert bootstrap["multiplicity"] == "Bonferroni_over_12_endpoints"
    assert bootstrap["one_sided_quantile"] == pytest.approx(0.05 / 12)
    assert frozen["analysis"]["endpoint_count"] == 12
    assert frozen["analysis"]["candidate_minus_production_noninferiority_margin"] == 0.0
    assert frozen["analysis"]["all_endpoints_conjunctive"] is True
    assert bootstrap["draw_plan_sha256"] != (
        "44569a4a813d0b736b6c093b7c2b5e1ffd4b1a353398b98cc14dabe4a718f7c2"
    )


def test_v30a_binds_205_unit_frame_and_keeps_nontrain_surfaces_closed():
    frozen = prereg_v30a.build_preregistration()
    joint = frozen["inputs"]["joint_frame"]
    assert joint["paired_units"] == 205
    assert joint["selected_units"] == 195
    assert joint["reserve_units"] == 10
    assert frozen["hardware"]["requests_per_engine_per_signed_wave"] == 390
    assert frozen["hardware"]["perturbed_requests_all_engines"] == 24_960
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
    baseline = frozen["audited_baseline"]
    assert baseline["version"] == "v25a"
    assert baseline["all_12_conjunctive_familywise_endpoints_preserved"] is True
    assert baseline["perturbation_or_bootstrap_draw_reuse"] is False
    assert baseline["optimization_panels"] == 3
    assert baseline["untouched_train_screens"] == 2
    assert baseline["population_size"] == 32
    assert baseline["bootstrap_repetitions"] == 50_000
    assert baseline[
        "posthoc_power_margin_multiplicity_or_threshold_change"
    ] is False


def test_v30a_exclusive_write_rejects_reuse(tmp_path, monkeypatch):
    output = tmp_path / "prereg.json"
    monkeypatch.setattr(prereg_v30a, "PREREGISTRATION_PATH", output)
    prereg_v30a.exclusive_write(output, {"bound": True})
    with pytest.raises(ValueError, match="already exists"):
        prereg_v30a.exclusive_write(output, {"bound": True})
