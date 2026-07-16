#!/usr/bin/env python3

import json

import numpy as np

import build_lora_es_robust_paired_hpo_preview_v61 as builder
import lora_es_robust_paired_hpo_v61 as subject


def _metrics(units=64, f1=0.5):
    value = np.zeros((units, 4, 2, 3), dtype=np.float64)
    value[..., 0] = f1
    value[..., 2] = float(f1 > 0.0)
    value[..., 1] = float(f1 == 1.0)
    return value


def test_v61_panels_are_conflict_unit_disjoint_and_exact_is_sentinel_only():
    strata = builder._read_sealed_v61(
        builder.V61A_STRATA,
        subject.V61A_STRATA_FILE_SHA256,
        subject.V61A_STRATA_CONTENT_SHA256,
    )
    panels = subject.build_panels_v61(strata)
    assert len(panels["ranking"]) == 64
    assert len(panels["untouched_holdback"]) == 50
    assert len(panels["exact_sentinel"]) == 4
    assert len(panels["unused_reserve"]) == 90
    roles = [{item["unit_identity_sha256"] for item in panels[key]} for key in (
        "ranking", "untouched_holdback", "exact_sentinel", "unused_reserve",
    )]
    assert len(set.union(*roles)) == 208
    assert all(not roles[left] & roles[right] for left in range(4)
               for right in range(left + 1, 4))
    assert all(item["base_exact_actor_count"] == 0 for item in panels["ranking"])
    assert sorted(item["base_exact_actor_count"] for item in panels["exact_sentinel"]) == [2, 4, 4, 4]
    assert panels["gpu_launch_authorized"] is False
    assert panels[
        "v61a_baseline_model_outcomes_used_for_train_only_stratification"
    ] is True
    assert panels["future_candidate_outcomes_used_for_panel_selection"] is False
    assert panels["protected_or_holdback_outcomes_used"] is False
    assert panels["train_only_adaptive_design"] is True


def test_v61_paired_bootstrap_is_deterministic_and_zero_for_identical_states():
    value = _metrics()
    first = subject.paired_unit_actor_bootstrap_v61(value, value, replicates=256)
    second = subject.paired_unit_actor_bootstrap_v61(value, value, replicates=256)
    assert first == second
    assert first["robust_generation_fitness"] == 0.0
    assert set(first["lower_confidence_bounds"].values()) == {0.0}


def test_v61_paired_bootstrap_rewards_improvement_and_penalizes_instability():
    reference = _metrics()
    improved = _metrics(f1=0.6)
    good = subject.paired_unit_actor_bootstrap_v61(
        reference, improved, replicates=256,
    )
    assert good["lower_confidence_bounds"]["f1_delta"] > 0.09
    assert good["robust_generation_fitness"] > 0.0
    unstable = _metrics(f1=0.6)
    unstable[:, 0, 0, 0] = 0.1
    bad = subject.paired_unit_actor_bootstrap_v61(
        reference, unstable, replicates=256,
    )
    assert bad["lower_confidence_bounds"]["stability_improvement"] < 0.0
    assert bad["robust_generation_fitness"] < good["robust_generation_fitness"]


def test_v61_exact_sentinel_is_strict_per_unit():
    reference = _metrics(units=4, f1=1.0)
    candidate = reference.copy()
    assert subject.exact_sentinel_gate_v61(reference, candidate)["passed"] is True
    candidate[0, 0, 0, 0] = 0.5
    candidate[0, 0, 0, 1] = 0.0
    assert subject.exact_sentinel_gate_v61(reference, candidate)["passed"] is False


def test_v61_preview_binds_censuses_and_cannot_launch():
    value = builder.build_preview_v61()
    assert value["gpu_launch_authorized"] is False
    assert value["new_launchable_preregistration_required"] is True
    assert value["fixed_model_optimizer_recipe"]["sigma"] == 0.0048
    assert value["fixed_model_optimizer_recipe"]["population_size"] == 16
    assert value["paired_population_fitness"]["exact_in_population_composite"] is False
    assert value["protected_phase_contract"]["protected_path_open_count"] == 0
    assert value["adaptive_design_provenance"] == {
        "v61a_baseline_model_outcomes_used_for_train_only_stratification": True,
        "future_candidate_outcomes_used_for_panel_selection": False,
        "protected_or_holdback_outcomes_used": False,
        "train_only_adaptive_design": True,
    }
    encoded = json.dumps(value, sort_keys=True)
    assert '"question"' not in encoded and '"answer"' not in encoded
