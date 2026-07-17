import json
import math

import pytest

import build_structured_es_preregistration_v1 as builder
import structured_es_oracle_v1 as contract


@pytest.fixture(scope="module")
def built():
    return builder.build_preregistration_v1()


def test_builder_is_deterministic_and_cpu_contract_is_valid(built):
    assert builder.build_preregistration_v1() == built
    result = contract.validate_preregistration_v1(built)
    assert result == {
        "status": "sealed_cpu_correctness_runtime_dependencies_pending",
        "content_sha256": (
            "de119d12099c381c299ff6de7484d882e805661c7223205943204d19e2e0b405"
        ),
        "arm_count": 6,
    }


def test_checked_in_artifact_is_exact_builder_output(built):
    checked = json.loads(builder.OUTPUT.read_text(encoding="utf-8"))
    assert checked == built
    assert checked["content_sha256_before_self_field"] == (
        "de119d12099c381c299ff6de7484d882e805661c7223205943204d19e2e0b405"
    )
    assert builder.file_sha256_v1(builder.OUTPUT) == (
        "1daa8372cef13613736534b6539eceea112616e32abbf0fa7ec30b457b3aeb4b"
    )


def test_builder_reuses_sealed_optimizer_sigma_and_v66d_telemetry(built):
    sources = built["source_contracts"]
    assert sources["fp32_optimizer_sigma"]["file_sha256"] == (
        contract.OPTIMIZER_CONTRACT_FILE_SHA256_V1
    )
    assert sources["fp32_optimizer_sigma"]["content_sha256"] == (
        contract.OPTIMIZER_CONTRACT_CONTENT_SHA256_V1
    )
    assert sources["fp32_optimizer_sigma"][
        "selected_fixed_optimizer_for_isolation"
    ] == "sgd"
    assert sources["fp32_optimizer_sigma"][
        "selected_fixed_sigma_mode_for_isolation"
    ] == "global"
    assert sources["fp32_optimizer_sigma"][
        "selection_used_empirical_optimizer_results"
    ] is False

    telemetry = sources["v66d_accepted_telemetry"]
    assert telemetry["report_file_sha256"] == contract.V66D_REPORT_FILE_SHA256_V1
    assert telemetry["report_content_sha256"] == (
        contract.V66D_REPORT_CONTENT_SHA256_V1
    )
    assert telemetry["gpu_telemetry_file_sha256"] == (
        contract.V66D_TELEMETRY_FILE_SHA256_V1
    )
    assert telemetry["actor_log_file_sha256"] == (
        contract.V66D_ACTOR_LOG_FILE_SHA256_V1
    )
    assert telemetry["all_16_signed_candidates_actor_receipted"] is True
    assert telemetry["all_four_gpu_batches_acknowledged"] is True
    assert telemetry["exact_abort_restore_and_final_idle"] is True
    assert telemetry["live_artifacts_opened_by_this_builder"] is False


def test_builder_reads_only_dense_metadata_and_seals_the_real_model_surface(built):
    dense = built["surfaces"]["dense_fullweight"]
    assert dense["metadata_only_no_tensor_payload_loaded"] is True
    assert dense["shard_count"] == 26
    assert dense["tensor_count"] == 1_045
    assert dense["elements"] == 35_951_822_704
    assert dense["bf16_bytes"] == 71_903_645_408
    assert dense["fp32_master_bytes"] == 143_807_290_816
    assert dense["maximum_tensor_elements"] == 536_870_912
    assert dense["dtype_element_counts"] == {"BF16": 35_951_822_704}
    assert dense["ordered_tensor_metadata_sha256"] == (
        "a408d963dd8f5e02fbcce79eb7d6f5ff3a8c0d0fb2da484e25fe7fce451ea2ef"
    )
    assert dense["index_file_sha256"] == contract.MODEL_INDEX_FILE_SHA256_V1
    assert dense["config_file_sha256"] == contract.MODEL_CONFIG_FILE_SHA256_V1
    assert dense["systems_anchor_only"] is True
    assert dense["quality_causal_comparison_to_lora"] is False


def test_dense_anchor_scratch_capacity_and_bandwidth_cost_are_explicit(built):
    memory = built["memory_bandwidth_contract"]
    dense = memory["dense_fullweight_system_anchor"]
    assert dense == {
        "whole_surface_noise_elements_allocated": 0,
        "whole_surface_candidate_elements_allocated": 0,
        "inplace_tensor_streaming_required": True,
        "bf16_model_bytes_per_replica": 71_903_645_408,
        "fp32_master_bytes_per_replica": 143_807_290_816,
        "four_replica_fp32_master_bytes": 575_229_163_264,
        "maximum_single_tensor_fp32_noise_bytes": 2_147_483_648,
        "weighted_update_scratch_ceiling_bytes": 4_294_967_296,
        "candidate_parameter_write_bytes_per_signed_candidate": 71_903_645_408,
        "surface_ratio_vs_lora": pytest.approx(7939.665730297377),
        "capacity_preflight_required_before_launch": True,
    }
    assert (
        contract.SIGNED_CANDIDATES_PER_UPDATE_V1
        * dense["candidate_parameter_write_bytes_per_signed_candidate"]
        == 1_150_458_326_528
    )
    assert memory[
        "common_candidate_runtime_write_bytes_per_lora_signed_candidate"
    ] == 9_842_688
    assert memory[
        "structured_does_not_reduce_runtime_install_bytes_without_fusion"
    ] is True


def test_matched_lora_arms_have_equal_surface_compute_and_only_rank_changes(built):
    lora_arms = [
        item for item in built["arms"]
        if item["causal_group"] == "matched_lora_space"
    ]
    assert len(lora_arms) == 5
    assert {item["surface"] for item in lora_arms} == {"matched_lora_4528128"}
    assert {(item["method"], item["structured_rank"]) for item in lora_arms} == {
        ("iid_absolute_index", None),
        *{
            ("structured_outer_product", rank)
            for rank in contract.STRUCTURED_RANKS_V1
        },
    }
    compute = built["compute_contract"]
    assert compute["systems_replicate_seed"] == 1701
    assert compute["directions_per_update"] == 8
    assert compute["signed_candidates_per_update"] == 16
    assert compute["train_units_per_candidate"] == 64
    assert compute["rollouts_per_systems_arm_seed"] == 1_024
    assert compute["quality_replicate_seeds"] == [1701, 1702, 1703]
    assert compute["quality_sigma_schedule"] == [0.0006, 0.0003]
    assert compute["quality_rollouts_per_lora_arm_seed"] == 2_048
    assert compute["fixed_optimizer"] == "sgd"
    assert compute["fixed_sigma_mode"] == "global"
    assert math.isclose(compute["update_budget_ratio"], 0.0005)


def test_artifact_does_not_authorize_gpu_eval_protected_or_commit_actions(built):
    assert built["authorization"] == {
        "cpu_correctness": True,
        "gpu_launch": False,
        "train_identity_use": True,
        "dev_or_ood": False,
        "protected_holdout": False,
        "live_run_read": False,
        "candidate_commit": False,
        "promotion": False,
    }
    assert built["status"] == "sealed_cpu_correctness_runtime_dependencies_pending"
    assert built["dependencies"]["production_streaming_worker_complete"] is False
    assert built["dependencies"]["optimizer_phase_pcie_profile_complete"] is False
    assert built["dependencies"]["dense_fullweight_capacity_preflight_complete"] is False
