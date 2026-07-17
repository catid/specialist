import json

import build_qwen36_adapter_transport_cuda_preregistration_v81a as builder
import eggroll_es_adapter_transport_precision_v81 as transport
import run_qwen36_adapter_transport_paired_v81a as runner


def test_v81a_preregistration_binds_additive_and_accepted_sources_exactly():
    value = builder.build_preregistration_v81a()
    accepted = value["bindings"]["accepted_lifecycle"]
    additive = value["bindings"]["additive_implementation"]
    assert accepted["worker_v71"]["file_sha256"] == (
        "6167ecd24332e0384e50f4bfa34893112623c2291f35e89989d8c3a527b9fcaa"
    )
    assert accepted["worker_v72"]["file_sha256"] == (
        "547d525edfd51412abb3a4980ddc4a55730ad0eb09987ec202ce2ce8f701a2c2"
    )
    assert accepted["v73b_preregistration"]["content_sha256"] == (
        "50b51ee2d71dc85d024c4a63cc57183d2b9c2d925c18924a719dacb1fb61dc94"
    )
    assert accepted["v73b_postrun"]["content_sha256"] == (
        "21689f75ecaaf583aedde50ad293ce3a9b5644009d62c2bc4624637280a651e7"
    )
    assert additive["cuda_worker_subclass"]["path"] == (
        "eggroll_es_worker_lora_pinned_transport_v81a.py"
    )
    assert additive["prospective_pair_runner"]["path"] == (
        "run_qwen36_adapter_transport_paired_v81a.py"
    )


def test_cuda_contract_has_one_exact_bank_and_no_device_staging_or_d2d():
    value = builder.build_preregistration_v81a()
    cuda = value["cuda_integration"]
    assert cuda["pinned_host_bank_count_per_actor"] == 1
    assert cuda["pinned_host_bank_elements"] == 4_921_344
    assert cuda["pinned_host_bank_bytes_per_actor"] == 9_842_688
    assert cuda["pinned_host_bank_bytes_all_four_actors"] == 39_370_752
    assert cuda["runtime_view_count"] == 82
    assert cuda["h2d_bytes_per_transition"] == 9_842_688
    assert cuda["h2d_calls_per_transition"] == 82
    assert cuda["temporary_device_publication_staging_bytes"] == 0
    assert cuda["device_to_device_payload_bytes"] == 0
    assert cuda["consumer_stream_wait_event_before_activation"] is True
    assert cuda["event_synchronize_before_v71_exact_readback"] is True
    assert cuda["event_fence_before_generation"] is True
    assert cuda["double_buffering"] is False


def test_cpu_fp32_authority_and_v71_v72_restore_contract_are_immutable():
    value = builder.build_preregistration_v81a()
    authority = value["canonical_authority"]
    lifecycle = value["accepted_lifecycle_contract"]
    assert authority["location"] == "cpu"
    assert authority["dtype"] == "float32"
    assert authority["tensor_count"] == 70
    assert authority["elements"] == 4_528_128
    assert authority["bytes_per_actor"] == 18_112_512
    assert authority["execution_transport_may_mutate_these_roles"] is False
    assert lifecycle["v71_exact_runtime_base_master_audits_retained"] is True
    assert lifecycle["v72_host_bank_ownership_one_two_one_retained"] is True
    assert lifecycle["restore_unknown_or_partial_state"] == (
        "exact_master_or_terminal_poison"
    )
    assert lifecycle["sole_resident_vllm_adapter_slot_count"] == 1


def test_memlock_limit_is_only_capacity_context_not_pinning_proof():
    value = builder.build_preregistration_v81a()
    preflight = value["memlock_preflight"]
    assert preflight["required_bytes_per_actor_process"] == 9_842_688
    assert preflight["four_actor_aggregate_context_bytes"] == 39_370_752
    assert preflight["each_actor_inherited_limit_capacity_check_passed"] is True
    assert preflight["four_bank_sum_below_reported_limit_context"] is True
    assert preflight["limit_is_not_proof_of_successful_pinning"] is True
    assert preflight["live_each_actor_tensor_is_pinned_required"] is True


def test_prospective_run_is_four_pair_four_gpu_v73c_gated_and_nonpromotable():
    value = builder.build_preregistration_v81a()
    paired = value["prospective_pair_run"]
    gate = value["launch_gate_pending"]
    evaluation = value["evaluation_boundary"]
    assert paired["current_launch_authorized"] is False
    assert paired["minimum_counterbalanced_pairs"] == 3
    assert paired["frozen_pair_count"] == 4
    assert paired["pair_order"] == [
        list(item) for item in runner.PAIR_ORDER_V81A
    ]
    assert paired["physical_gpu_ids_every_arm"] == [0, 1, 2, 3]
    assert paired["systems_only_nonpromotable"] is True
    assert gate["required_schema"] == runner.GATE_SCHEMA_V81A
    assert gate["required_bead"] == "specialist-0j5.32"
    assert gate["gate_file_present_now"] is False
    assert evaluation["this_runner_is_systems_only"] is True
    assert evaluation["quality_evaluation_in_this_artifact"] is False
    assert evaluation["quarantined_v1_resolved"] is False
    assert evaluation["legacy_evaluation_contract_imported"] is False
    assert "sealed_V2" in evaluation["quality_or_OOD_promotion_requires"]


def test_collective_coalescing_is_explicitly_out_of_scope():
    value = builder.build_preregistration_v81a()
    exclusions = value["scope_exclusions"]
    assert exclusions["fp32_collective_coalescing"] == "specialist-0j5.36"
    assert exclusions["collective_bucket_or_dtype_change_in_v81a"] is False
    assert exclusions["noise_or_update_generation_change"] is False
    assert exclusions["execution_dtype_change"] is False
    assert exclusions["persistent_device_VRAM_reduction_claim"] is False
    assert exclusions["shared_v71_fused_exact_audit_buffer_change"] is False


def test_preregistration_is_canonical_self_hashed_and_output_is_current():
    value = builder.build_preregistration_v81a()
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    assert value["content_sha256_before_self_field"] == (
        transport.canonical_sha256_v81(compact)
    )
    assert json.loads(builder.OUTPUT.read_text(encoding="ascii")) == value
