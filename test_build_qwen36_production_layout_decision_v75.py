import copy
import json

import pytest

import build_qwen36_production_layout_decision_v75 as decision_v75


def _valid_fp8_promotion(decision):
    return {
        "schema": decision_v75.FP8_PROMOTION_SCHEMA_V75,
        "decision_content_sha256": decision["content_sha256_before_self_field"],
        "v74_receipts": 4,
        "v74_immutable_run_inventory_sha256": (
            decision_v75.EXPECTED_V74_INVENTORY_SHA256
        ),
        "v74_capacity_gate_passed": True,
        "physical_gpus": [0, 1, 2, 3],
        "gpu_memory_utilization": 0.50,
        "resolved_quantization": "fp8",
        "weight_block_size": [128, 128],
        "generation_peak_mib": 52_738,
        "full_es_peak_mib": 84_138,
        "kv_cache_tokens": 157_286,
        "paired_replicates": 3,
        "median_runtime_ratio_to_bf16": 1.09,
        "source_disjoint_dev_semantic_reward_noninferiority": True,
        "ood_all_registered_noninferiority_conditions": True,
        "all_four_gpu_activity_and_cleanup_idle": True,
        "reference_restore_exact_at_token_hash_level": True,
        "zero_unresolved_kernel_or_tactic_fallbacks": True,
        "routed_expert_method_count_attested": True,
        "exact_adapter_candidate_restore_update_receipts": True,
        "protected_terminal_opened": False,
    }


def test_builder_matches_machine_readable_artifact_exactly():
    built = decision_v75.build_decision_v75()
    observed = json.loads(decision_v75.OUTPUT.read_text(encoding="utf-8"))
    assert observed == built
    assert built["schema"] == decision_v75.SCHEMA_V75
    assert built["status"] == "provisional_not_final_benchmark_authority"
    assert built["content_sha256_before_self_field"] == (
        "5dd23d1effbecec2068d8e21d7f8bf9e5afab85a9e8a58d38a913e835c0e0ed5"
    )
    compact = dict(built)
    compact.pop("content_sha256_before_self_field")
    assert decision_v75.canonical_sha256_v75(compact) == (
        built["content_sha256_before_self_field"]
    )


def test_bound_evidence_aggregates_and_v74_capacity_only_status():
    built = decision_v75.build_decision_v75()
    v73 = built["evidence"]["paired_precision_v73"]
    v74 = built["evidence"]["right_sized_fp8_capacity_v74"]
    assert v73["bf16"]["median_runtime_seconds"] == 45.19571384298615
    assert v73["fp8_serialized"]["median_runtime_seconds"] == 49.39261133299442
    assert v73["fp8_serialized"]["runtime_ratio_to_bf16"] == pytest.approx(
        1.0928605200172001
    )
    assert v73["bf16"]["peak_memory_mib"] == 83_820
    assert v73["fp8_serialized"]["peak_memory_mib"] == 83_878
    assert v74["immutable_run_inventory_sha256"] == (
        "13dc3991ec440e273359455fdf970f0025f3048deeb74d73219d29dd845ee04c"
    )
    assert v74["receipts"] == 4
    assert v74["physical_gpus"] == [0, 1, 2, 3]
    assert v74["median_runtime_seconds"] == 48.95789764399524
    assert v74["steady_and_peak_memory_mib"] == 52_738
    assert v74["physical_headroom_mib"] == 45_149
    assert v74["peak_saving_vs_v73_fp8_0p82_mib"] == 31_140
    assert v74["kv_cache_tokens"] == 157_286
    assert v74["capacity_gate_passed"] is True
    assert v74["promotion_gate_passed"] is False
    assert v74["scored_evaluation_or_training_authorized"] is False
    assert "reference_repeat_not_exact_4_to_5_of_68_rows" in (
        v74["promotion_blockers_observed"]
    )


def test_capacity_reservation_and_layout_choices_are_explicit():
    built = decision_v75.build_decision_v75()
    capacity = built["capacity_reservation"]
    assert capacity == {
        "device_total_mib": 97_887,
        "full_es_observed_peak_mib": 84_138,
        "minimum_reserved_full_es_headroom_mib": 13_749,
        "candidate_update_delta_mib": 926,
        "minimum_pre_es_generation_headroom_mib": 14_675,
        "maximum_pre_es_generation_peak_mib": 83_212,
        "maximum_full_es_peak_mib": 84_138,
        "fp8_rightsized_gpu_memory_utilization": 0.50,
        "fp8_nominal_engine_budget_mib_at_0p50": 48_943.5,
        "fp8_rightsized_observed_peak_mib": 52_738,
        "fp8_rightsized_observed_headroom_mib": 45_149,
        "minimum_kv_cache_tokens": 139_264,
        "flat_shadow_buffer_reserved_mib": 0.0,
        "host_memory_reservation_status": "blocked_pending_specialist-0j5.19",
    }
    safe = built["safe_default"]["layout"]
    fp8 = built["conditional_fp8_challenger"]["layout"]
    for layout in (safe, fp8):
        assert layout["memory"]["max_loras"] == 1
        assert layout["memory"]["max_num_seqs"] == 68
        assert layout["collective_layout"]["parameter_boundaries"] == (
            "native_23_tensor"
        )
        assert layout["collective_layout"]["flat_shadow_buffer"] is False
        assert layout["state"]["dense_full_weight_master_on_lora_path"] is False
        assert layout["state"]["mode"] == "external_lora_canonical_fp32"
    assert safe["memory"]["gpu_memory_utilization"] == 0.82
    assert fp8["memory"]["gpu_memory_utilization"] == 0.50
    assert fp8["precision_requirements"]["weight_block_size"] == [128, 128]
    assert built["conditional_fp8_challenger"]["status"] == "blocked_not_promoted"


def test_rejections_blockers_and_deferred_work_are_not_silently_promoted():
    built = decision_v75.build_decision_v75()
    rejected = {item["id"] for item in built["rejected_alternatives"]}
    assert rejected == {
        "dual_gpu_lora_slots_v66",
        "global_static_max_num_seqs_48_or_32",
        "flat_collective_shadow_buffer",
        "fp8_with_gpu_memory_utilization_0p82",
        "int4",
    }
    assert built["blocking_beads_before_finalization"] == [
        "specialist-0j5.15",
        "specialist-0j5.18",
        "specialist-0j5.19",
        "specialist-0j5.21",
        "specialist-0j5.22",
    ]
    deferred = {item["id"] for item in built["deferred_not_rejected"]}
    assert deferred == {
        "compiled_cuda_graph_execution",
        "fused_candidate_noise_update_and_audit",
        "shared_or_streamed_host_master",
    }


def test_bf16_confirmation_consumes_exact_decision_and_layout():
    decision = decision_v75.validate_decision_v75()
    request = decision_v75.consumer_request_v75(decision, "bf16")
    authorization = decision_v75.authorize_consumer_request_v75(
        request, purpose="confirmation"
    )
    assert authorization["authorized"] is True
    assert authorization["precision_arm"] == "bf16"
    assert authorization["layout_unchanged"] is True
    assert authorization["final_benchmark_authorized"] is False


def test_provisional_decision_always_rejects_final_benchmark():
    decision = decision_v75.validate_decision_v75()
    request = decision_v75.consumer_request_v75(decision, "bf16")
    with pytest.raises(RuntimeError, match="remains provisional"):
        decision_v75.authorize_consumer_request_v75(
            request, purpose="final_benchmark"
        )


def test_fp8_confirmation_requires_every_registered_gate():
    decision = decision_v75.validate_decision_v75()
    request = decision_v75.consumer_request_v75(decision, "fp8_serialized")
    with pytest.raises(RuntimeError, match="absent"):
        decision_v75.authorize_consumer_request_v75(
            request, purpose="confirmation"
        )
    promotion = _valid_fp8_promotion(decision)
    authorization = decision_v75.authorize_consumer_request_v75(
        request, purpose="confirmation", promotion=promotion
    )
    assert authorization["precision_arm"] == "fp8_serialized"
    assert authorization["final_benchmark_authorized"] is False


@pytest.mark.parametrize(
    "field",
    [
        "v74_capacity_gate_passed",
        "source_disjoint_dev_semantic_reward_noninferiority",
        "ood_all_registered_noninferiority_conditions",
        "all_four_gpu_activity_and_cleanup_idle",
        "reference_restore_exact_at_token_hash_level",
        "zero_unresolved_kernel_or_tactic_fallbacks",
        "routed_expert_method_count_attested",
        "exact_adapter_candidate_restore_update_receipts",
    ],
)
def test_fp8_boolean_promotion_gates_fail_closed(field):
    decision = decision_v75.build_decision_v75()
    promotion = _valid_fp8_promotion(decision)
    promotion[field] = False
    with pytest.raises(RuntimeError, match="identity or boolean gate"):
        decision_v75.validate_fp8_promotion_v75(promotion, decision)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generation_peak_mib", 83_213),
        ("full_es_peak_mib", 84_139),
        ("kv_cache_tokens", 139_263),
        ("paired_replicates", 2),
        ("median_runtime_ratio_to_bf16", 1.1000001),
        ("median_runtime_ratio_to_bf16", float("nan")),
        ("median_runtime_ratio_to_bf16", float("inf")),
    ],
)
def test_fp8_numeric_promotion_gates_fail_closed(field, value):
    decision = decision_v75.build_decision_v75()
    promotion = _valid_fp8_promotion(decision)
    promotion[field] = value
    with pytest.raises(RuntimeError, match="numeric gate|threshold failed"):
        decision_v75.validate_fp8_promotion_v75(promotion, decision)


def test_consumer_rejects_tampering_and_unknown_fields():
    decision = decision_v75.validate_decision_v75()
    request = decision_v75.consumer_request_v75(decision, "bf16")
    tampered = copy.deepcopy(request)
    tampered["layout"]["memory"]["max_loras"] = 2
    with pytest.raises(RuntimeError, match="layout changed"):
        decision_v75.authorize_consumer_request_v75(
            tampered, purpose="confirmation"
        )
    promotion = _valid_fp8_promotion(decision)
    promotion["unregistered_claim"] = True
    with pytest.raises(RuntimeError, match="schema changed"):
        decision_v75.validate_fp8_promotion_v75(promotion, decision)
    changed = copy.deepcopy(decision)
    changed["capacity_reservation"]["minimum_reserved_full_es_headroom_mib"] -= 1
    with pytest.raises(RuntimeError, match="unvalidated"):
        decision_v75.consumer_request_v75(changed)


def test_all_evidence_validators_run_cpu_only():
    v73 = decision_v75.validate_v73_evidence_v75()
    assert decision_v75.validate_v74_evidence_v75(v73)["capacity_gate_passed"]
    assert decision_v75.validate_prior_layout_evidence_v75()[
        "one_slot_eager_median_seconds"
    ] == 46.089602033549454
    assert decision_v75.validate_collective_evidence_v75()["decision"] == (
        "retain_native_parameter_boundaries_no_flat_shadow"
    )
