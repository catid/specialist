#!/usr/bin/env python3

import json

import analyze_lora_es_generated_f1_gate_v62 as audit
import build_lora_es_generated_f1_gate_preregistration_v62 as builder


def test_v62_arithmetic_rebuild_matches_all_finalized_primary_intervals():
    value = audit.build_audit_v62()
    runs = value["runs"]
    assert [runs[name]["estimator_alternatives"]["arithmetic_mean_8"][
        "contains_zero"
    ] for name in ("v61c", "v61d", "v61e")] == [True, True, True]
    assert [runs[name]["estimator_alternatives"]["arithmetic_mean_8"][
        "halfwidth"
    ] for name in ("v61c", "v61d", "v61e")] == [
        0.00039520537364525746,
        0.0007477014243197278,
        0.0006345017752690909,
    ]
    assert [runs[name]["estimator_alternatives"]["arithmetic_mean_8"][
        "point"
    ] for name in ("v61c", "v61d", "v61e")] == [
        0.00025793744545295736,
        -0.00016923934613666746,
        0.00020058517989776778,
    ]


def test_v62_quantifies_sparse_erasure_and_rejects_consensus_primary():
    value = audit.build_audit_v62()
    pooled = value["pooled_alternative_quantification"]
    assert pooled["arithmetic_nonzero_unit_outcomes"] == 19
    assert pooled["median_8_arithmetic_nonzero_units_erased"] == 16
    assert pooled["median_8_erasure_fraction"] == 16 / 19
    assert pooled["trim_mean_6_arithmetic_nonzero_units_erased"] == 3
    assert pooled["three_of_four_actor_sign_consensus_units"] == 2
    assert pooled["units_with_any_nonzero_replica"] == 28
    assert pooled["four_of_four_actor_sign_consensus_units"] == 0
    assert pooled["winsorized_run_level_sign_mismatch_count"] == 1
    selected = value["method_selection"]
    assert selected["primary_estimator"] == (
        "arithmetic_mean_8_conflict_unit_cluster"
    )
    assert selected["median_8_rejected_as_primary_due_sparse_signal_erasure"]


def test_v62_actor_influence_uses_like_for_like_null_calibration():
    value = audit.build_audit_v62()
    runs = value["runs"]
    assert [runs[name]["actor_influence"][
        "maximum_absolute_leave_one_actor_out_shift"
    ] for name in ("v61c", "v61d", "v61e")] == [
        0.00012697480507903975,
        0.0004788587653010674,
        0.0012119648781783704,
    ]
    calibrated = value["calibrated_noise_and_signal"]
    assert calibrated["signal_over_maximum_ci_halfwidth"] > 2.0
    assert calibrated["v61e_passes_future_ci_width_gate"] is True
    assert calibrated["v61e_passes_future_actor_influence_gate"] is True
    assert calibrated["future_pre_hpo_max_actor_leave_one_out_shift"] == (
        0.0012119648781783704
    )
    assert calibrated["future_actor_influence_threshold_calibration"] == (
        "maximum_like_for_like_finalized_v61c_v61d_v61e_null_"
        "leave_one_actor_out_shift"
    )
    assert calibrated["future_actor_influence_threshold_rule"] == (
        "conservative_observed_maximum_no_distributional_quantile_claim"
    )
    assert calibrated[
        "v53_actor_spread_definition_proven_equivalent_to_v62_loo"
    ] is False
    assert calibrated["v53_actor_spread_used_as_v62_loo_threshold"] is False
    assert all(
        item["proposed_population_fitness_at_null"]["value"] < 0.0
        for item in runs.values()
    )


def test_v62_preregisters_f1_primary_and_moves_exact_to_full_census():
    value = builder.build_preregistration_v62()
    estimator = value["estimator_selection"]
    assert estimator["within_unit_aggregation"] == (
        "arithmetic_mean_all_8_replicas"
    )
    assert estimator["population_fitness_formula"] == (
        "generated_f1_primary_cluster_bootstrap_lcb_minus_1.0_times_"
        "maximum_absolute_leave_one_actor_out_shift"
    )
    diagnostics = value["population_diagnostics_not_abort_gates"]
    assert diagnostics["generated_exact"]["any_single_flip_aborts_population"] is False
    assert diagnostics["generated_exact"][
        "any_per_unit_eight_of_eight_equality_failure_aborts_population"
    ] is False
    assert diagnostics["generated_exact"]["used_in_population_fitness"] is False
    assert diagnostics["teacher_forced_logprob"]["used_in_population_fitness"] is False
    sentinel = diagnostics["frozen_exact_sentinel"]
    stable = sentinel["baseline_stable"]
    assert stable["unit_count"] == 3
    assert stable["per_unit_consensus_statistic"] == (
        "strict_majority_exact_pass_count_at_least_5_of_8"
    )
    assert stable["used_as_population_abort_gate"] is False
    assert len(stable["unit_identity_sha256"]) == 3
    assert stable["unit_identity_sha256"] == list(
        builder.STABLE_EXACT_UNIT_SHA256
    )
    stress = sentinel["actor_unstable_stress"]
    assert stress["unit_count"] == 1
    assert stress["role"] == "diagnostic_stress_unit_only"
    assert stress["excluded_from_stable_consensus_aggregate"] is True
    assert stress["used_as_population_abort_gate"] is False
    assert stress["unit_identity_sha256"] == (
        builder.ACTOR_UNSTABLE_EXACT_UNIT_SHA256
    )
    assert sentinel["used_in_population_fitness"] is False
    commit = value["pre_materialization_master_commit_gate"]
    assert commit["eligible_train_full_census_required"] is True
    assert commit["counterbalanced_pairs_per_actor"] == 2
    assert commit[
        "ephemeral_in_memory_candidate_state_required_for_scoring"
    ] is True
    assert commit[
        "persistent_candidate_artifact_or_snapshot_before_gate"
    ] is False
    assert commit[
        "failure_discards_ephemeral_candidate_before_any_persistent_"
        "artifact_or_master_commit"
    ] is True
    assert commit["master_commit_before_gate"] is False
    assert commit["aggregate_exact_noninferiority_margin"] == 0
    assert commit[
        "aggregate_exact_candidate_total_must_be_at_least_reference_total"
    ] is True
    assert commit["individual_exact_flips_are_reported_but_do_not_individually_abort"]


def test_v62_is_self_hashed_numeric_only_and_authorizes_no_launch_or_access():
    numeric = audit.build_audit_v62()
    value = builder.build_preregistration_v62()
    assert numeric["authorization"] == {
        "hpo_launch_authorized": False,
        "candidate_materialization_or_master_commit_authorized": False,
        "holdback_ood_shadow_terminal_or_protected_access_authorized": False,
    }
    access = value["access_and_authorization"]
    assert access["gpu_launch_authorized"] is False
    assert access["hpo_population_launch_authorized"] is False
    assert access["holdback_ood_shadow_terminal_or_protected_access_authorized"] is False
    assert access["separate_runtime_preregistration_required_before_any_launch"]
    encoded = json.dumps(value, sort_keys=True)
    assert '"question"' not in encoded and '"answer"' not in encoded
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    assert value["content_sha256_before_self_field"] == (
        audit.canonical_sha256_v62(compact)
    )
