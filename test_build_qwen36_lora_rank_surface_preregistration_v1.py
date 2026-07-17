#!/usr/bin/env python3

from __future__ import annotations

import copy
import json

import pytest

import build_qwen36_lora_rank_surface_preregistration_v1 as subject


@pytest.fixture(scope="module")
def built() -> dict:
    return subject.build_preregistration()


@pytest.fixture(scope="module")
def geometry() -> dict:
    return subject.topology_v70.build_architecture_manifest()


def test_persisted_json_and_report_match_deterministic_build(built):
    body = copy.deepcopy(built)
    claimed = body.pop("content_sha256_before_self_field")
    assert claimed == subject.canonical_sha256(body)
    assert subject.OUTPUT.read_text(encoding="ascii") == subject.render(built)
    assert subject.REPORT.read_text(encoding="utf-8") == subject.render_report(built)
    subject.validate_preregistration(built)


def test_cpu_authority_and_sealed_upstreams_are_exact(built):
    assert built["authority"] == {
        "builder_cpu_metadata_and_header_audit_only": True,
        "builder_raw_model_or_adapter_tensor_materialized": False,
        "builder_dataset_row_or_protected_content_opened": False,
        "builder_gpu_launch_or_model_update_performed": False,
        "live_hpo_or_scored_evaluation_authorized": False,
        "known_v1_protected_access_incident_is_not_reset": True,
    }
    upstream = built["upstream_contracts"]
    assert len(upstream["implementation_files"]) == 6
    assert len(upstream["json_contracts"]) == 6
    assert {
        item["sha256"] for item in upstream["implementation_files"]
    } == set(subject.UPSTREAM_FILES.values())
    incident = upstream["known_protected_access_incident"]
    assert incident["content_sha256"] == (
        subject.EXPECTED_ACCESS_INCIDENT_CONTENT_SHA256
    )
    assert incident["v1_access_nonzero"] is True
    assert incident["entire_v1_source_quarantined"] is True
    firewall = built["protected_terminal_firewall"]
    assert firewall["builder_protected_source_opened"] is False
    assert firewall["known_v1_access_count_is_nonzero"] is True
    assert firewall["v1_source_eligible_for_future_terminal_evaluation"] is False
    assert firewall[
        "fresh_untouched_v2_required_before_any_live_hpo_or_terminal_evaluation"
    ] is True


def test_access_incident_is_content_addressed_nonzero_and_fail_closed():
    incident = json.loads(subject.ACCESS_INCIDENT.read_text(encoding="ascii"))
    body = copy.deepcopy(incident)
    claimed = body.pop("content_sha256_before_self_field")
    assert claimed == subject.EXPECTED_ACCESS_INCIDENT_CONTENT_SHA256
    assert claimed == subject.canonical_sha256(body)
    assert subject.file_sha256(subject.ACCESS_INCIDENT) == (
        subject.EXPECTED_ACCESS_INCIDENT_FILE_SHA256
    )
    event = incident["event"]
    assert event["v1_protected_access_count_is_nonzero"] is True
    assert event["raw_source_file_parse_count"] == 10
    assert event["rows_materialized_per_raw_source_parse"] == 59
    assert event[
        "rows_materialized_total_in_test_process_including_repeated_parses"
    ] == 590
    assert event["selected_protected_rows_returned_by_successful_claimed_loader"] == 12
    assert len(incident["known_selected_opaque_item_identities"]) == 12
    assert incident["quarantine"]["v1_source_may_not_be_used_for_future_terminal_evaluation"] is True
    assert incident["quarantine"]["reset_or_relabel_v1_access_count_to_zero_prohibited"] is True
    escape = incident["output_escape_audit"]
    assert escape["protected_raw_rows_materialized_inside_test_process"] is True
    assert escape["protected_row_text_in_agent_or_tool_output"] is False
    assert escape["protected_row_text_persisted_by_specialist_0j5_27_artifacts"] is False
    assert escape["protected_row_text_used_for_model_input_training_reward_or_selection"] is False


def test_rank_only_ladder_has_exact_logical_and_packed_geometry(built):
    expected = {
        "full_current_r8": (8, 1_132_032, 1_230_336, 98_304),
        "full_current_r16": (16, 2_264_064, 2_460_672, 196_608),
        "full_current_r32_control": (32, 4_528_128, 4_921_344, 393_216),
        "full_current_r64": (64, 9_056_256, 9_842_688, 786_432),
    }
    for arm_id, values in expected.items():
        surface = built["arms"][arm_id]["surface"]
        assert (
            surface["rank"],
            surface["logical_trainable_parameters"],
            surface["active_packed_elements"],
            surface["packing_duplicate_elements"],
        ) == values
        assert surface["module_count"] == 35
        assert surface["peft_tensor_count"] == 70
        assert surface["runtime_view_count"] == 82
        assert surface["families"] == list(subject.ALL_FAMILIES)
        assert surface["selected_layers"] == [20, 21, 22, 23]


def test_current_control_reproduces_v70_inventory_exactly(built):
    control = built["arms"][subject.CONTROL_ARM]["surface"]
    v70 = subject.topology_v70.build_arm_specs()[
        "current_middle_late_motif_r32"
    ]["surface"]
    assert (
        control["module_count"],
        control["peft_tensor_count"],
        control["runtime_view_count"],
        control["logical_trainable_parameters"],
        control["active_packed_elements"],
        control["packing_duplicate_elements"],
    ) == (
        v70["module_count"],
        v70["tensor_count"],
        v70["runtime_view_count"],
        v70["parameter_count"],
        v70["runtime_allocated_elements"],
        v70["runtime_packing_duplicate_elements"],
    ) == (35, 70, 82, 4_528_128, 4_921_344, 393_216)
    assert control["family_parameter_counts"] == {
        "router_gate": 294_912,
        "shared_attention_gdn": 3_250_176,
        "shared_expert_projections": 983_040,
    }


def test_family_only_cohort_matches_v69_and_does_not_change_rank_or_layers(built):
    expected = {
        "shared_expert_only_r32": (12, 24, 983_040, 983_040, 0),
        "attention_gdn_only_r32": (19, 50, 3_250_176, 3_643_392, 393_216),
        "frozen_router_dense_r32": (31, 74, 4_233_216, 4_626_432, 393_216),
        subject.CONTROL_ARM: (35, 82, 4_528_128, 4_921_344, 393_216),
    }
    for arm_id, values in expected.items():
        surface = built["arms"][arm_id]["surface"]
        assert surface["rank"] == 32
        assert surface["selected_layers"] == [20, 21, 22, 23]
        assert (
            surface["module_count"],
            surface["runtime_view_count"],
            surface["logical_trainable_parameters"],
            surface["active_packed_elements"],
            surface["packing_duplicate_elements"],
        ) == values


def test_each_target_and_aggregate_account_for_vllm_slices_and_rank_padding(built):
    for arm in built["arms"].values():
        surface = arm["surface"]
        for target in surface["targets"]:
            rank = target["rank"]
            out_features, in_features = target["base_shape"]
            slices = len(target["runtime_slices"])
            assert target["logical_parameters"] == rank * (
                in_features + out_features
            )
            assert target["active_packed_elements"] == (
                slices * rank * in_features + out_features * rank
            )
            assert target["packing_duplicate_elements"] == (
                (slices - 1) * rank * in_features
            )
        assert surface["active_packed_elements"] == (
            surface["logical_trainable_parameters"]
            + surface["packing_duplicate_elements"]
        )
        assert surface["rank_padding_elements"] == 0
        assert arm["dedicated_engine_policy"]["runtime_max_lora_rank"] == (
            surface["rank"]
        )
    assert built["arms"]["full_current_r8"][
        "shared_rank64_engine_diagnostic_excluded_from_pareto"
    ]["rank_padding_elements"] == 8_612_352
    assert built["arms"]["full_current_r16"][
        "shared_rank64_engine_diagnostic_excluded_from_pareto"
    ]["rank_padding_elements"] == 7_382_016
    assert built["arms"][subject.CONTROL_ARM][
        "shared_rank64_engine_diagnostic_excluded_from_pareto"
    ]["rank_padding_elements"] == 4_921_344


def test_fp32_master_runtime_transfer_and_checkpoint_ledgers_are_exact(built):
    for arm in built["arms"].values():
        surface = arm["surface"]
        ledger = arm["byte_ledger"]
        master = 4 * surface["logical_trainable_parameters"]
        runtime = 2 * surface["active_packed_elements"]
        assert ledger["persistent_bytes"]["canonical_fp32_master"] == master
        assert ledger["persistent_bytes"]["active_bf16_runtime_payload"] == runtime
        transfer = ledger["per_materialization_transfer_payload_bytes"]
        assert transfer["runtime_install_h2d"] == runtime
        assert transfer["single_flat_exact_audit_d2h"] == runtime
        assert transfer["prepack_fp32_master_h2d_if_current_gpu_candidate_path"] == master
        assert transfer["owned_fp32_candidate_d2h_if_current_gpu_candidate_path"] == master
        projected = ledger["one_update_projected_current_path"]
        assert projected["audited_runtime_materializations"] == 33
        assert projected["runtime_install_and_restore_h2d_bytes"] == 33 * runtime
        assert projected["runtime_exact_audit_d2h_bytes"] == 33 * runtime
        assert projected["candidate_fp32_master_h2d_bytes"] == 32 * master
        assert projected["owned_candidate_fp32_d2h_bytes"] == 32 * master
        checkpoints = ledger["checkpoint_tensor_and_step_payload_bytes"]
        assert checkpoints["sgd_no_slot"] == master + 8
        assert checkpoints["momentum_one_slot"] == 2 * master + 8
        assert checkpoints["adamw_two_slots"] == 3 * master + 8
        assert checkpoints["whole_file_bytes_must_be_measured_not_inferred"] is True
    control = built["arms"][subject.CONTROL_ARM]["byte_ledger"]
    assert control["one_update_projected_current_path"] == {
        "signed_candidate_installs": 32,
        "final_canonical_restore": 1,
        "audited_runtime_materializations": 33,
        "runtime_install_and_restore_h2d_bytes": 324_808_704,
        "runtime_exact_audit_d2h_bytes": 324_808_704,
        "candidate_fp32_master_h2d_bytes": 579_600_384,
        "owned_candidate_fp32_d2h_bytes": 579_600_384,
        "projection_is_not_a_live_counter_receipt": True,
    }


def test_rank_and_family_effects_are_preregistered_as_separate_cohorts(built):
    separation = built["cohort_separation"]
    assert tuple(separation["rank_only_fixed_layers_and_families"]) == (
        "full_current_r8",
        "full_current_r16",
        subject.CONTROL_ARM,
        "full_current_r64",
    )
    assert tuple(separation["family_only_fixed_layers_and_rank32"]) == (
        "shared_expert_only_r32",
        "attention_gdn_only_r32",
        "frozen_router_dense_r32",
        subject.CONTROL_ARM,
    )
    assert separation["cross_cohort_difference_may_not_be_attributed_to_rank_alone"] is True
    assert separation["no_arm_changes_layer_topology"] is True
    assert separation["v70_topology_winner_dependency"][
        "this_protocol_may_not_promote_another_topology"
    ] is True
    assert built["budget_summary"]["distinct_logical_parameter_budget_count"] == 7


def test_initialization_does_not_hide_compression_as_an_optimizer_effect(built):
    arms = built["arms"]
    assert arms[subject.CONTROL_ARM]["initialization"][
        "function_preserving_from_current_rank32"
    ] is True
    assert arms["full_current_r64"]["initialization"][
        "function_preserving_from_current_rank32"
    ] is True
    for arm_id in ("full_current_r8", "full_current_r16"):
        initialization = arms[arm_id]["initialization"]
        assert "truncated SVD" in initialization["method"]
        assert initialization["compression_or_family_removal_is_part_of_estimand"] is True
        assert initialization["rank8_or_rank16_may_not_be_called_a_pure_optimizer_effect"] is True


def test_all_arm_seed_schedules_have_equal_rollout_and_gpu_second_contract(built):
    compute = built["compute_and_schedule"]
    assert compute["training_seeds"] == [1701, 1702, 1703]
    assert compute["optimization_generated_rollouts_each_arm_seed"] == 2_048
    assert compute["evaluation_generated_rollouts_each_arm_seed"] == 171
    assert compute["evaluation_breakdown"] == {
        "source_disjoint_dev_generated": 83,
        "ood_qa_generated": 24,
        "ood_prose_teacher_forced": 16,
        "data_free_systems_generated": 64,
    }
    for seed, schedule in compute["counterbalanced_schedule_by_seed"].items():
        assert int(seed) in compute["training_seeds"]
        assert len(schedule) == len(subject.ARM_ORDER)
        assert set(schedule) == set(subject.ARM_ORDER)
    assert compute["exact_rollout_and_crn_identity_equality_required"] is True
    assert compute[
        "every_arm_seed_charged_gpu_seconds_must_match_same_seed_control_within_tolerance"
    ] is True
    assert compute["charged_gpu_second_relative_tolerance_within_cohort"] == 0.02
    assert compute["artificial_idle_padding_prohibited"] is True
    assert compute["adaptive_extra_training_to_fill_time_prohibited"] is True
    assert compute["all_four_physical_gpus_useful_each_training_and_evaluation_phase"] is True


def test_three_seed_quality_ood_and_pareto_gates_fail_closed(built):
    gates = built["three_seed_source_disjoint_quality_gates"]
    assert gates["evaluation_contract_content_sha256"] == (
        subject.trust_v67.EXPECTED_EVALUATION_CONTRACT_CONTENT_SHA256
    )
    assert gates["evaluation_contract_v1_binding_quarantined_not_live_eligible"] is True
    assert gates["fresh_v2_contract_rebind_required_before_any_live_hpo"] is True
    assert gates["trust_policy_semantics_sha256"] == (
        subject.trust_v67.POLICY_SEMANTICS_SHA256
    )
    assert gates["common_frozen_pre_es_reference"] == (
        "exact_current_rank32_adapter_before_any_arm_training"
    )
    assert gates["paired_pareto_control_arm"] == subject.CONTROL_ARM
    assert gates["all_three_seeds_required_each_arm"] is True
    assert gates["weighted_dev_delta_95_lcb_minimum"] == -0.01
    assert gates["weighted_dev_point_delta_minimum_exclusive"] == 0.0
    assert gates["ood_qa_reward_delta_95_lcb_minimum"] == -0.02
    assert gates["ood_qa_exact_count_delta_minimum"] == -1
    assert gates["ood_prose_token_logprob_delta_95_lcb_minimum"] == -0.02
    assert gates["safety_and_hallucination_component_lcb_minimum"] == 0.0
    assert gates["new_hard_safety_or_quality_event_count_maximum"] == 0
    pareto = built["pareto_selection"]
    assert pareto["eligible_only_after_all_live_compute_activity_surface_and_quality_gates"] is True
    assert pareto["scalarization_or_post_hoc_axis_weights_prohibited"] is True
    assert pareto["select_only_pareto_nondominated_surface"] is True
    assert pareto["current_full_control_reported_even_if_dominated"] is True


@pytest.mark.parametrize(
    ("rank", "families", "maximum", "message"),
    [
        (4, subject.ALL_FAMILIES, None, "rank must be one of"),
        (32, (), None, "nonempty canonical"),
        (32, ("unknown",), None, "nonempty canonical"),
        (32, tuple(reversed(subject.ALL_FAMILIES)), None, "nonempty canonical"),
        (32, subject.ALL_FAMILIES, 16, "cannot be smaller"),
    ],
)
def test_invalid_rank_family_or_runtime_capacity_is_rejected(
    geometry, rank, families, maximum, message
):
    with pytest.raises(ValueError, match=message):
        subject.build_surface(
            geometry,
            rank=rank,
            families=families,
            runtime_max_lora_rank=maximum,
        )


def test_rehashed_or_semantically_mutated_preregistration_is_rejected(built):
    changed = copy.deepcopy(built)
    changed["arms"][subject.CONTROL_ARM]["surface"][
        "active_packed_elements"
    ] -= 1
    changed.pop("content_sha256_before_self_field")
    changed["content_sha256_before_self_field"] = subject.canonical_sha256(changed)
    with pytest.raises(RuntimeError, match="preregistration is stale"):
        subject.validate_preregistration(changed)


def test_upstream_hash_drift_fails_closed(monkeypatch):
    changed = dict(subject.UPSTREAM_FILES)
    changed["eggroll_es_front_tail_topology_v70.py"] = "0" * 64
    monkeypatch.setattr(subject, "UPSTREAM_FILES", changed)
    with pytest.raises(RuntimeError, match="upstream implementation changed"):
        subject.validate_upstream_contracts()


def test_json_is_finite_and_outstanding_work_keeps_bead_in_progress(built):
    rendered = subject.render(built)
    assert json.loads(rendered)["schema"] == subject.SCHEMA
    assert "NaN" not in rendered
    assert "Infinity" not in rendered
    outstanding = built["outstanding_before_task_acceptance"]
    assert outstanding["task_must_remain_in_progress"] is True
    assert outstanding[
        "specialist_0j5_30_fresh_evaluation_v2_complete"
    ] is False
    assert outstanding["twenty_one_arm_seed_runs_complete"] is False
    assert outstanding["live_packed_allocation_and_transfer_ledgers_complete"] is False
    assert outstanding["three_seed_quality_ood_and_reward_snr_complete"] is False
    assert outstanding["pareto_frontier_frozen"] is False
