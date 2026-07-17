import copy
import json

import pytest

import build_qwen36_resident_lora_slot_decision_v80 as decision_v80


EXPECTED_DECISION_SHA = (
    "4c255e41d930e2338f5c3a9d02a9cf99671be3b0b7bbdfa4690a7d39aea86b6d"
)
EXPECTED_PREREG_SHA = (
    "1723fe1be88cdca1c73decd38bfc2a1614a78332ae7670b834396f70816cc4e7"
)


def test_machine_artifacts_match_deterministic_builds_exactly():
    decision = decision_v80.build_decision_v80()
    prereg = decision_v80.build_preregistration_v80()
    observed_decision = json.loads(
        decision_v80.DECISION_OUTPUT.read_text(encoding="utf-8")
    )
    observed_prereg = json.loads(
        decision_v80.PREREG_OUTPUT.read_text(encoding="utf-8")
    )
    assert observed_decision == decision
    assert observed_prereg == prereg
    assert decision["content_sha256_before_self_field"] == EXPECTED_DECISION_SHA
    assert prereg["content_sha256_before_self_field"] == EXPECTED_PREREG_SHA
    compact = dict(decision)
    compact.pop("content_sha256_before_self_field")
    assert decision_v80.canonical_sha256_v80(compact) == EXPECTED_DECISION_SHA


def test_primary_runtime_and_conservative_throughput_are_exact():
    arms = decision_v80.build_decision_v80()["evidence"]["primary_arms"]
    assert arms["max_loras_1_eager"]["runtime"]["median_seconds"] == (
        46.089602033549454
    )
    assert arms["max_loras_2_eager"]["runtime"]["median_seconds"] == (
        80.09551188797923
    )
    assert arms["max_loras_1_graph"]["runtime"]["median_seconds"] == (
        23.550330316007603
    )
    assert arms["max_loras_2_graph"]["runtime"]["median_seconds"] == (
        31.965402808011277
    )
    for arm in arms.values():
        runtime = arm["runtime"]
        assert runtime["recorded_call_output_tokens_per_actor"] == 34_816
        assert runtime["timer_includes_two_warmup_generations"] is True
        assert runtime["timer_includes_first_adapter_loads"] is True
        assert runtime["adjacent_adapter_state_transitions_inside_timer"] == 6
        assert runtime["isolated_switch_latency_measured"] is False


def test_runtime_comparison_is_complete_workload_not_switch_latency():
    comparisons = decision_v80.build_decision_v80()["evidence"]["comparisons"]
    eager = comparisons["eager_max_loras_2_vs_1"]
    graph = comparisons["graph_max_loras_2_vs_1"]
    assert eager["median_workload_runtime_increase_percent"] == pytest.approx(
        73.78217288506042
    )
    assert graph["median_workload_runtime_increase_percent"] == pytest.approx(
        35.73229071136972
    )
    assert eager["recorded_call_token_rate_decrease_percent"] == pytest.approx(
        42.45669832535698
    )
    assert graph["recorded_call_token_rate_decrease_percent"] == pytest.approx(
        26.325563743231363
    )
    assert "not an isolated" in eager["interpretation"]


def test_vram_delta_kv_displacement_and_fixed_budget_plateaus_are_separate():
    memory = decision_v80.build_decision_v80()["evidence"]["comparisons"][
        "memory_and_capacity"
    ]
    assert memory["model_load_allocation_increase_gib"] == 3.54
    assert memory["model_load_allocation_increase_mib_approximate"] == 3624.96
    assert memory["eager_kv_token_reduction_percent"] == pytest.approx(
        51.76499310661764
    )
    assert memory["graph_kv_token_reduction_percent"] == pytest.approx(
        55.00030517578125
    )
    assert memory["eager_device_resident_mode_mib"] == {
        "max_loras_1": 83_820.0,
        "max_loras_2": 83_254.0,
        "max_loras_2_minus_1": -566.0,
    }
    assert memory["graph_device_resident_mode_mib"] == {
        "max_loras_1": 81_944.0,
        "max_loras_2": 81_186.0,
        "max_loras_2_minus_1": -758.0,
    }
    assert "not the incremental slot cost" in memory["nvml_plateau_interpretation"]


def test_output_behavior_rejects_exact_restore_and_quality_claims():
    built = decision_v80.build_decision_v80()
    arms = built["evidence"]["primary_arms"]
    assert arms["max_loras_2_graph"]["output_behavior"][
        "candidate_repeat_changed_rows_distribution"
    ] == {"0": 7, "3": 1}
    assert arms["max_loras_1_eager"]["output_behavior"][
        "actors_with_any_reference_repeat_change"
    ] == 11
    for arm in arms.values():
        behavior = arm["output_behavior"]
        assert behavior["decoded_text_or_reward_persisted"] is False
        assert behavior["semantic_quality_equivalence_measured"] is False
    cache = built["cache_and_restore_disposition"]
    assert cache["prefix_cache_reuse_mechanism_enabled"] is False
    assert cache["absence_of_all_stale_candidate_state_proved"] is False
    assert cache["exact_token_repeat_restore_achieved_by_every_actor"] is False


def test_later_one_slot_graph_replications_are_bound_but_not_postselected_into_primary():
    evidence = decision_v80.build_decision_v80()["evidence"]
    primary = evidence["primary_arms"]["max_loras_1_graph"]
    supplemental = evidence["supplemental_arms"][
        "max_loras_1_graph_later_replications"
    ]
    assert primary["run_count"] == 3
    assert primary["actor_count"] == 12
    assert supplemental["run_count"] == 5
    assert supplemental["actor_count"] == 20
    assert supplemental["runtime"]["median_seconds"] == 23.282616868498735
    assert supplemental["output_behavior"][
        "candidate_repeat_changed_rows_distribution"
    ] == {"0": 20}


def test_installed_source_proves_cpu_lru_and_one_base_not_duplicate_base():
    audit = decision_v80.build_decision_v80()["installed_vllm_source_audit"]
    claims = audit["claims_supported_by_source"]
    assert claims == {
        "registered_adapter_capacity_is_max_cpu_loras": True,
        "active_gpu_slot_capacity_is_max_loras": True,
        "already_registered_adapter_skips_disk_load_and_is_touched": True,
        "one_active_slot_evicts_oldest_then_copies_registered_weights": True,
        "linear_and_fused_moe_lora_buffers_scale_with_max_loras": True,
        "lora_wrappers_retain_one_base_layer_reference": True,
        "max_loras_2_duplicates_the_gpu_base_model": False,
    }
    assert "no allocator-level breakdown" in audit["causal_limit"]


def test_retained_layout_is_the_supported_cpu_resident_lazy_alternative():
    decision = decision_v80.build_decision_v80()["decision"]
    assert decision["tested_max_loras_2_resident_gpu_slots"] == "rejected"
    assert decision["retained_supported_layout"] == {
        "max_loras": 1,
        "max_cpu_loras": 2,
        "base_model_instances_per_actor": 1,
        "registered_cpu_adapter_capacity": 2,
        "active_gpu_adapter_slot_capacity": 1,
        "after_first_load_disk_reload_for_two_alternating_adapters": False,
        "gpu_slot_activation_copies_registered_adapter_weights": True,
        "prefix_caching": False,
    }
    assert "_load_adapter" in decision["load_inplace_assessment"]
    assert "No supported safe implementation" in (
        decision["custom_compact_gpu_buffer_assessment"]
    )


def test_missing_switch_traffic_and_quality_are_null_not_imputed():
    limits = decision_v80.build_decision_v80()["measurement_limits"]
    for field in (
        "isolated_adapter_switch_latency_seconds",
        "pcie_rx_bytes_per_switch",
        "pcie_tx_bytes_per_switch",
        "hbm_bytes_per_switch",
        "decoded_text_or_reward_quality_equivalence",
        "direct_gpu_lora_tensor_restore_digest",
    ):
        assert limits[field] is None
    assert "NVML files lack PCIe fields" in limits["why_not_inferred"]


def test_future_challenger_contract_fails_closed_and_has_no_run_authority():
    prereg = decision_v80.build_preregistration_v80()
    gates = prereg["promotion_gates"]
    assert gates["model_load_allocation_gib_not_above"] == 68.24
    assert gates["eager_gpu_kv_tokens_not_below"] == 139_264
    assert gates["exact_adapter_tensor_restore"] is True
    assert gates["zero_missing_switch_or_pcie_measurements"] is True
    assert all(value is False for value in prereg["authority"].values())
    assert "never imputed" in prereg["failure_policy"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_loras", 2),
        ("max_cpu_loras", 1),
        ("base_model_instances_per_actor", 2),
        ("prefix_caching", True),
        ("custom_gpu_slot_buffer", True),
        ("decision_content_sha256", "0" * 64),
    ],
)
def test_layout_consumer_rejects_drift_and_rejected_options(field, value):
    decision = decision_v80.validate_decision_v80()
    request = decision_v80.retained_layout_request_v80(decision)
    request[field] = value
    with pytest.raises(RuntimeError, match="changed|rejected"):
        decision_v80.authorize_layout_request_v80(request)


def test_layout_consumer_only_authorizes_selection_not_launch_or_training():
    decision = decision_v80.validate_decision_v80()
    request = decision_v80.retained_layout_request_v80(decision)
    result = decision_v80.authorize_layout_request_v80(request)
    assert result == {
        "layout_selection_authorized": True,
        "gpu_launch_authorized": False,
        "training_authorized": False,
        "decision_content_sha256": EXPECTED_DECISION_SHA,
        "max_loras": 1,
        "max_cpu_loras": 2,
    }


def test_bead_closes_as_rejection_without_claiming_acceptance():
    built = decision_v80.build_decision_v80()
    assert built["status"] == "closed_negative_two_gpu_resident_slots_rejected"
    assert built["bead_disposition"]["close_as_negative_result"] is True
    assert built["bead_disposition"]["acceptance_criteria_claimed_complete"] is False
    assert all(value is False for value in built["authority"].values())


def test_rehashed_semantically_tampered_receipt_is_rejected(tmp_path, monkeypatch):
    source = (
        decision_v80.ROOT
        / decision_v80.PRIMARY_ARMS["max_loras_1_eager"][0]
        / "gpu_0_receipt.json"
    )
    value = json.loads(source.read_text(encoding="utf-8"))
    value.pop("content_sha256_before_self_field")
    value["runtime"]["max_cpu_loras"] = 3
    value["content_sha256_before_self_field"] = decision_v80.canonical_sha256_v80(
        value
    )
    path = tmp_path / "tampered_receipt.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(decision_v80, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="receipt contract changed"):
        decision_v80._validate_receipt_v80(path, 1, False)


def test_source_hash_drift_fails_closed(monkeypatch):
    contract = decision_v80.VLLM_SOURCE_ATTESTATIONS["config/lora.py"]
    monkeypatch.setitem(contract, "sha256", "0" * 64)
    with pytest.raises(RuntimeError, match="installed vLLM source changed"):
        decision_v80.audit_vllm_source_v80()
