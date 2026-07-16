#!/usr/bin/env python3

import build_lora_es_generation_boundary_design_v48a as builder
import lora_es_generation_boundary_sampling_v48a as boundary
import run_lora_es_multi_anchor_v43i as v43i


def test_v48a_design_is_fail_closed_pending_train_only_evidence():
    value = builder.build_design_v48a()
    assert value["status"] == (
        "sealed_cpu_design_pending_train_only_base_evidence"
    )
    assert value["gpu_launch_authorized"] is False
    assert value["protected_semantics_opened"] is False
    assert value["shadow_ood_holdout_or_benchmark_opened"] is False
    assert value["access_contract"]["runtime_paths_sealed_now"] == []
    assert value["access_contract"]["v46d_or_any_holdout_artifact_bound"] is False
    assert value["content_sha256_before_self_field"] == v43i.v40a.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def test_v48a_design_makes_f1_primary_and_a_projection_anchor():
    value = builder.build_design_v48a()
    objective = value["objective_protocol"]
    assert objective["fragile_generated_f1_is_direct_primary_objective"] is True
    assert objective["fragile_generated_f1_is_not_only_a_halfspace"] is True
    assert objective["simultaneous_projection_anchors"] == [
        "domain", "fragile_generation_f1", "prose_lm",
        "qa_answer_logprob",
    ]
    assert objective["fail_closed_if_fragile_f1_population_spread_is_zero"] is True


def test_v48a_compute_accounting_and_common_random_contract():
    value = builder.build_design_v48a()
    population = value["population_protocol"]
    assert population["signed_actor_states"] == 64
    assert population["new_requests_per_actor_state"] == 608
    assert population["extra_fragile_generation_completions"] == 4096
    assert population["extra_worst_case_decode_tokens"] == 262_144
    assert population["generation_decode_work_multiplier_vs_v43i"] == 3.0
    assert population["common_random_plan_receipts_required"] == 64
    subset = value["fragile_subset_protocol"]
    assert subset["selected_conflict_units"] == boundary.FRAGILE_SUBSET_UNITS_V48A
    assert subset["maximum_rows_per_selected_conflict_unit"] == 1
    assert subset["naive_row_oversampling"] is False


def test_v48a_implementation_and_v43m_noop_are_hash_bound():
    value = builder.build_design_v48a()
    assert value["implementation_bindings"] == builder.implementation_bindings_v48a()
    assert value["problem_audit"]["v43m_f1_halfspace_nonbinding"] is True
    assert value["problem_audit"]["v43m_coefficients_changed"] is False
