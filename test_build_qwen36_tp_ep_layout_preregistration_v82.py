import copy
import json

import pytest

import build_qwen36_tp_ep_layout_preregistration_v82 as v82


def test_v82_generated_artifacts_are_current_and_self_hashed():
    value = v82.build_preregistration_v82()
    body = copy.deepcopy(value)
    claimed = body.pop("content_sha256_before_self_field")
    assert claimed == v82.canonical_sha256_v82(body)
    assert v82.OUTPUT.read_text(encoding="ascii") == v82.render_json_v82(value)
    assert v82.REPORT.read_text(encoding="utf-8") == v82.render_report_v82(value)


def test_v82_is_cpu_only_and_waits_for_dependencies():
    value = v82.build_preregistration_v82()
    authority = value["authority"]
    assert authority == {
        "cpu_file_inspection_only": True,
        "gpu_api_or_model_launch_performed_by_builder": False,
        "dataset_or_prompt_rows_opened_by_builder": False,
        "model_or_adapter_update_performed": False,
        "checkpoint_or_layout_promotion_performed": False,
        "live_launch_authorized": False,
        "live_launch_waits_for_beads": ["specialist-0j5.14", "specialist-0j5.15"],
        "site_packages_or_es_at_scale_modified": False,
    }
    assert value["status"] == "cpu_preregistered_live_launch_not_authorized"


def test_v82_admits_control_tp4_tp_only_ep_and_wide_ep_agrs():
    value = v82.build_preregistration_v82()
    arms = value["admitted_arms"]
    assert [arm["arm_id"] for arm in arms] == [
        "replicated_4xtp1_control",
        "single_tp4_tensor_sharded",
        "single_tp4_ep4_tp_collective",
        "wide_dp4_tp1_ep4_agrs",
    ]
    assert [arm["candidate_concurrency"] for arm in arms] == [4, 1, 1, 1]
    assert [arm["signed_candidate_waves"] for arm in arms] == [4, 16, 16, 16]
    assert arms[0]["parallel"] == {
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "enable_expert_parallel": False,
        "all2all_backend": "allgather_reducescatter",
        "all2all_active": False,
    }
    assert arms[1]["parallel"]["tensor_parallel_size"] == 4
    assert arms[1]["parallel"]["enable_expert_parallel"] is False
    assert arms[2]["parallel"] == {
        "tensor_parallel_size": 4,
        "data_parallel_size": 1,
        "enable_expert_parallel": True,
        "all2all_backend": "allgather_reducescatter",
        "all2all_active": False,
        "enable_ep_weight_filter": False,
        "enable_eplb": False,
        "expert_placement_strategy": "linear",
    }
    assert arms[2]["expected_moe_rank_contract"]["use_all2all_kernels"] is False
    assert arms[3]["parallel"] == {
        "tensor_parallel_size": 1,
        "data_parallel_size": 4,
        "data_parallel_size_local": 4,
        "enable_expert_parallel": True,
        "all2all_backend": "allgather_reducescatter",
        "all2all_active": True,
        "enable_ep_weight_filter": False,
        "enable_eplb": False,
        "expert_placement_strategy": "linear",
    }
    assert arms[3]["expected_moe_rank_contract"]["use_all2all_kernels"] is True
    assert arms[3]["expected_moe_rank_contract"][
        "use_ag_rs_all2all_kernels"
    ] is True
    assert arms[3]["expected_moe_rank_contract"][
        "dense_and_kv_state_remain_tp1_replicated"
    ] is True
    assert all(arm["worker_rank_count"] == 4 for arm in arms)


def test_v82_rejects_recipe_incompatible_and_unattested_backends():
    value = v82.build_preregistration_v82()
    rejected = {arm["arm_id"]: arm for arm in value["explicitly_rejected_arms"]}
    assert set(rejected) == {
        "dp4_tp1_wide_ep_four_distinct_one_slot",
        "wide_dp4_tp1_ep4_deepep",
        "wide_dp4_tp1_ep4_nixl",
        "wide_dp4_tp1_ep4_pplx_or_naive",
        "wide_dp4_tp1_ep4_flashinfer_nvlink",
        "ep_weight_filter_as_memory_arm",
    }
    assert all(arm["admitted"] is False for arm in rejected.values())
    assert "different candidate values" in rejected[
        "dp4_tp1_wide_ep_four_distinct_one_slot"
    ]["reason"]
    assert "no fallback" in rejected["wide_dp4_tp1_ep4_deepep"]["reason"]
    assert "silent rewrite is forbidden" in rejected[
        "wide_dp4_tp1_ep4_pplx_or_naive"
    ]["reason"]
    assert "3D fused" in rejected["ep_weight_filter_as_memory_arm"]["reason"]


def test_v82_static_qwen_support_captures_kv_replication_caveat():
    model = v82.build_preregistration_v82()["installed_support_evidence"]["model"]
    assert model["architecture"] == "Qwen3_5MoeForConditionalGeneration"
    assert model["layers"] == 40
    assert model["experts"] == 256
    assert model["ep4_local_experts_per_rank"] == 64
    assert model["tp4_static_divisibility_passed"] is True
    assert model["attention_kv_heads_per_tp1_rank"] == 2
    assert model["attention_kv_heads_per_tp4_rank"] == 1
    assert model["attention_kv_elements_per_token_ratio_tp4_rank_to_tp1_rank"] == 0.5
    assert model["mamba_state_shape_requires_live_rank_attestation"] is True
    assert model["ep_weight_filter_memory_or_io_savings_assumed"] is False


def test_v82_preserves_v66d_candidate_and_workload_identity():
    workload = v82.build_preregistration_v82()["common_workload"]
    state = workload["candidate_state"]
    assert state["canonical_fp32_master_sha256"] == (
        "eea2d60e19530ba99e9ac4bc50f2806b20aa13ed30e159bad63a0144d0cb81b6"
    )
    assert state["canonical_runtime_bf16_values_sha256"] == (
        "a1353c47bc11f02a9b67d7859d6670b07d6754c285ac4f357255878c09384f5b"
    )
    assert len(state["direction_seeds_exact"]) == 8
    assert state["signed_population_size"] == 16
    assert state["sigma"] == 0.0006
    assert workload["prompt_panel"]["rows_per_candidate"] == 64
    assert workload["request_outputs_total"] == 1024
    assert workload["generated_tokens_upper_bound"] == 1024
    assert workload["decode"] == {
        "detokenize": False,
        "max_tokens": 1,
        "n": 1,
        "prompt_logprobs": 1,
        "temperature": 0.0,
        "top_p": 1.0,
    }


def test_v82_requires_rank_ownership_restore_and_all_four_useful():
    contract = v82.build_preregistration_v82()["runtime_attestation"]
    pre = contract["fail_closed_before_first_generation"]
    assert pre["physical_gpu_ids_exact"] == [0, 1, 2, 3]
    assert pre["resolved_all2all_backend_exact_no_rewrite"] is True
    assert pre["forbid_any_fallback_warning_or_backend_substitution"] is True
    ownership = contract["model_and_rank_ownership"]
    assert ownership["expert_map_and_local_expert_range_per_layer_and_rank"] is True
    assert ownership["attention_kv_heads_and_cache_block_shapes_per_rank"] is True
    assert ownership["mamba_conv_and_ssm_state_dtype_shape_bytes_per_rank"] is True
    restore = contract["lora_identity_and_restore"]
    assert restore["canonical_fp32_master_restore_sha256_exact"] is True
    assert restore["gathered_runtime_bf16_values_restore_sha256_exact"] is True
    useful = contract["useful_gpu_work"]
    assert useful["actor_cuda_event_elapsed_positive_per_candidate_per_rank"] is True
    assert useful["resident_but_idle_rank_fails"] is True
    trace = contract["collective_trace"]
    assert trace["tp4_ep_requires_zero_agrs_dispatch_and_combine_collectives"] is True
    assert trace[
        "wide_dp4_ep_requires_observed_agrs_dispatch_and_combine_collectives"
    ] is True


def test_v82_reports_vram_hbm_nccl_pcie_and_gpu_normalized_throughput():
    measurement = v82.build_preregistration_v82()["measurement"]
    assert measurement["vram_and_cache"][
        "per_rank_peak_and_phase_peak_memory_used_mib"
    ] is True
    bandwidth = measurement["bandwidth_and_communication"]
    assert bandwidth[
        "memory_utilization_is_activity_proxy_not_hbm_bytes_per_second"
    ] is True
    assert bandwidth["nsys_nccl_message_bytes_and_duration_per_rank_required"] is True
    assert bandwidth["pcie_rx_tx_integrals_are_left_rectangle_estimates"] is True
    assert bandwidth["ncu_profile_not_used_for_timing"] is True
    throughput = measurement["throughput"]
    assert throughput["rollouts_per_gpu_second"] is True
    assert throughput["active_gpu_seconds_sum"] is True
    assert throughput["end_to_end_population_wall_seconds"] is True


def test_v82_rejects_slower_arm_and_retains_semantic_ood_gates():
    gates = v82.build_preregistration_v82()["selection_and_quality"]
    system = gates["system_selection"]
    assert system[
        "any_arm_with_point_throughput_below_control_is_slower_and_rejected"
    ] is True
    assert system["rollouts_per_gpu_second_point_ratio_vs_control_min"] == 1.0
    assert system["per_gpu_peak_vram_reduction_fraction_min_for_memory_claim"] == 0.1
    assert gates["source_disjoint_semantic"]["paired_95pct_lower_bound_min"] == -0.002
    assert gates["protected_ood_one_shot_after_layout_freeze"][
        "paired_95pct_lower_bound_min"
    ] == -0.005
    assert gates["promotion"]["default"] is False


def test_v82_fails_closed_if_installed_vllm_identity_changes(monkeypatch):
    changed = dict(v82.VLLM_FILES)
    key = "vllm/config/parallel.py"
    changed[key] = "0" * 64
    monkeypatch.setattr(v82, "VLLM_FILES", changed)
    with pytest.raises(RuntimeError, match="installed vLLM source changed"):
        v82.build_preregistration_v82()


def test_v82_fails_closed_if_parent_identity_changes(monkeypatch):
    monkeypatch.setattr(v82, "V79_FILE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="sealed file changed"):
        v82.build_preregistration_v82()


def test_v82_json_contains_no_nonfinite_values():
    encoded = v82.render_json_v82(v82.build_preregistration_v82())
    decoded = json.loads(encoded)
    assert decoded["schema"] == v82.SCHEMA
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
