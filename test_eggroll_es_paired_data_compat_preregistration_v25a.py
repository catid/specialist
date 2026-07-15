#!/usr/bin/env python3

import json

import numpy as np
import pytest

import eggroll_es_paired_data_compat_preregistration_v25a as prereg_v25a


def test_v25a_preregistration_is_exact_and_deterministic():
    frozen = json.loads(prereg_v25a.PREREGISTRATION_PATH.read_text())
    built = prereg_v25a.build_preregistration()
    assert built == frozen
    assert frozen["content_sha256_before_self_field"] == (
        "0b5dfc076304bb8eb8bddd4f0f0d9d7754a0220c01839188b7be82484525a748"
    )
    assert frozen["content_sha256_before_self_field"] == (
        prereg_v25a.canonical_sha256(prereg_v25a.without_self(frozen))
    )
    prereg_v25a.validate_sha_fields(frozen)


def test_v25a_fresh_basis_and_schedule_are_fully_pinned():
    frozen = prereg_v25a.build_preregistration()
    basis = frozen["frozen_recipe"]["perturbation_basis"]
    regenerated = [
        int(value) for value in np.random.default_rng(20260907).integers(
            0, 2**30, size=32, dtype=np.int64,
        )
    ]
    assert basis["direction_seeds"] == regenerated
    assert basis["direction_seed_list_sha256"] == (
        "3bf870bf5aa8e5db0da17554fdb76845be591a761b4a7f1e57592b2a46d4be22"
    )
    assert basis["signed_population_schedule_sha256"] == (
        "d33931bfcb4aab568ceda1ae6c0eeffcfb8bc3a0f9037d2f4a5bd5d418133b73"
    )
    schedule = basis["signed_population_schedule"]
    assert len(schedule) == 16
    assert [item["sign"] for item in schedule] == ["plus", "minus"] * 8
    assert all(len(item["engine_direction_seeds"]) == 4 for item in schedule)
    assert basis["basis_content_sha256"] not in set(
        prereg_v25a.PRIOR_BASIS_HASHES.values()
    )


def test_v25a_bootstrap_draws_multiplicity_and_gate_are_predeclared():
    frozen = prereg_v25a.build_preregistration()
    bootstrap = frozen["analysis"]["bootstrap"]
    assert bootstrap["seed"] == 20260908
    assert bootstrap["repetitions"] == 50_000
    assert bootstrap["draw_plan_sha256"] == (
        "44569a4a813d0b736b6c093b7c2b5e1ffd4b1a353398b98cc14dabe4a718f7c2"
    )
    assert bootstrap["draw_plan_sha256"] == prereg_v25a.bootstrap_draw_plan_sha256()
    assert bootstrap["multiplicity"] == "Bonferroni_over_12_endpoints"
    assert bootstrap["one_sided_quantile"] == pytest.approx(0.05 / 12)
    assert frozen["analysis"]["endpoint_count"] == 12
    assert frozen["analysis"]["candidate_minus_production_noninferiority_margin"] == 0.0
    assert frozen["analysis"]["all_endpoints_conjunctive"] is True


def test_v25a_binds_205_unit_frame_and_keeps_nontrain_surfaces_closed():
    frozen = prereg_v25a.build_preregistration()
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


def test_v25a_exclusive_write_rejects_reuse(tmp_path, monkeypatch):
    output = tmp_path / "prereg.json"
    monkeypatch.setattr(prereg_v25a, "PREREGISTRATION_PATH", output)
    prereg_v25a.exclusive_write(output, {"bound": True})
    with pytest.raises(ValueError, match="already exists"):
        prereg_v25a.exclusive_write(output, {"bound": True})
