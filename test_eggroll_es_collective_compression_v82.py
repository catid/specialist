#!/usr/bin/env python3

import copy
import json
import math
import struct
import subprocess
import sys

import pytest

import benchmark_eggroll_es_collective_compression_v82 as benchmark
import build_qwen36_collective_compression_preregistration_v82 as builder
import eggroll_es_collective_compression_v82 as subject


def _seal(body):
    return {**body, "content_sha256": subject.canonical_sha256_v82(body)}


def _gate(collective_rank=2):
    identifiers = ["audit", "update_collective", "generation", "restore"]
    identifiers.remove("update_collective")
    identifiers.insert(collective_rank - 1, "update_collective")
    ranked = [
        {
            "rank": rank,
            "id": identifier,
            "unoverlapped_wall_seconds": float(10 - rank),
            "link_bytes": rank * 100,
            "hbm_bytes": rank * 200,
            "measurement_sha256": f"{rank}" * 64,
        }
        for rank, identifier in enumerate(identifiers, start=1)
    ]
    body = {
        "schema": subject.GATE_SCHEMA_V82,
        "dependency_beads": {
            bead: {
                "status": "closed",
                "acceptance_passed": True,
                "evidence_sha256": character * 64,
            }
            for bead, character in (
                ("specialist-0j5.14", "a"),
                ("specialist-0j5.18", "b"),
                ("specialist-0j5.19", "c"),
                ("specialist-0j5.32", "d"),
            )
        },
        "post_optimization_profile": {
            "profile_sha256": "e" * 64,
            "recipe_layout_sha256": "f" * 64,
            "measured_after_beads": ["specialist-0j5.18", "specialist-0j5.19"],
            "ranked_residual_bottlenecks": ranked,
            "collective_link_bytes_measured": True,
            "collective_time_measured": True,
            "hbm_bytes_measured": True,
            "all_four_gpus_attributed": True,
            "protected_content_opened": False,
        },
        "decision": (
            "launch_fp32_vs_bf16_ablation"
            if collective_rank <= 3
            else "close_not_applicable_without_gpu_ablation"
        ),
    }
    return _seal(body)


def _live(prereg_sha, decision="promote_bf16_error_feedback"):
    accounting = subject.collective_byte_accounting_v82()["bf16_error_feedback"]
    systems = []
    for index, seed in enumerate(subject.REGISTERED_SEEDS_V82):
        systems.append({
            "seed": seed,
            "counterbalanced_order": "AB" if index % 2 == 0 else "BA",
            "physical_gpus": [0, 1, 2, 3],
            "all_gpus_useful_each_arm": True,
            "cleanup_idle_each_arm": True,
            "fp32_link_bytes": 1000,
            "bf16_link_bytes": 520,
            "fp32_collective_seconds": 2.0,
            "bf16_collective_seconds": 1.5,
            "fp32_throughput_updates_per_second": 1.0,
            "bf16_throughput_updates_per_second": 1.1,
            "bf16_peak_staging_bytes": accounting[
                "maximum_native_tensor_bf16_staging_bytes_per_rank"
            ],
            "bf16_peak_residual_bytes": accounting[
                "transaction_peak_residual_bytes_per_rank"
            ],
            "hbm_bytes_measured": True,
        })
    body = {
        "schema": subject.LIVE_SCHEMA_V82,
        "preregistration_content_sha256": prereg_sha,
        "gate_content_sha256": "f" * 64,
        "training_seeds": list(subject.REGISTERED_SEEDS_V82),
        "arms": ["fp32_control", "bf16_error_feedback"],
        "paired_systems_results": systems,
        "quality_results": {
            "source_disjoint_contract_passed": True,
            "protected_holdout_opened": False,
            "three_seed_dev_paired_lcb": 0.0,
            "positive_dev_seeds": 2,
            "ood_qa_reward_lcb": -0.02,
            "ood_qa_exact_delta": -1,
            "ood_prose_logprob_lcb": -0.02,
            "all_ood_noninferiority_conditions_passed": True,
        },
        "residual_results": {
            "all_finite": True,
            "local_conservation_bit_exact": True,
            "resume_replay_bit_exact": True,
            "rollback_bit_exact": True,
            "rank_local_residual_receipts_complete": True,
            "maximum_absolute_error": 0.001,
            "maximum_registered_bound": 0.01,
            "canonical_master_dtype": "float32",
            "optimizer_dtype": "float32",
            "communication_dtype": "bfloat16",
        },
        "integrity": {
            "native_23_tensor_order_exact": True,
            "flat_shadow_allocated": False,
            "fp8_collective_attempted": False,
            "fp32_fallback_bit_exact": True,
            "unknown_collective_outcome_restored_or_poisoned": True,
            "checkpoint_master_residual_atomic": True,
            "protocol_or_leak_counter_increased": False,
        },
        "promotion_decision": decision,
    }
    return _seal(body)


def test_v82_preregistration_rebuilds_and_capabilities_are_fail_closed():
    frozen = json.loads(builder.OUTPUT.read_text(encoding="utf-8"))
    assert builder.build_preregistration_v82() == frozen
    assert builder.validate_preregistration_v82(frozen) == frozen
    capability = frozen["local_capabilities"]
    assert capability["torch_runtime_version_from_bound_source"] == "2.11.0+cu130"
    assert capability["nccl_distribution_version"] == "2.28.9"
    assert capability["bf16"]["ordinary_all_reduce_arm_registered"] is True
    assert capability["fp8"]["ordinary_all_reduce_arm_registered"] is False
    assert capability["fp8"]["launch_forbidden"] is True
    assert "Unsupported Float8" in capability["fp8"]["rejection_message"]


def test_v82_builder_imports_neither_torch_nor_cuda():
    code = """
import json, sys
import build_qwen36_collective_compression_preregistration_v82 as b
b.build_preregistration_v82()
print(json.dumps({'torch': 'torch' in sys.modules}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=builder.ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout) == {"torch": False}


def test_v82_v68_keeps_native_boundaries_and_no_flat_shadow():
    evidence = builder.inspect_v68_layout_evidence_v82()
    assert evidence["decision"] == "retain_native_23_tensor_boundaries"
    assert evidence["flat_shadow_bytes_per_rank"] == 571_998_208
    assert evidence["flat_shadow_forbidden_for_v82"] is True
    assert evidence["paired_run_native_over_flat_median_speeds"] == [
        1.003526527791839,
        0.9977076489242296,
    ]


def test_v82_byte_accounting_exposes_link_hbm_and_vram_tradeoff():
    value = subject.collective_byte_accounting_v82()
    assert value["native_tensor_count"] == 23
    assert value["total_elements_per_rank"] == 142_999_552
    assert value["fp32_control"]["collective_payload_bytes_per_rank"] == 571_998_208
    bf16 = value["bf16_error_feedback"]
    assert bf16["collective_payload_bytes_per_rank"] == 285_999_104
    assert bf16["nominal_ring_bus_bytes_per_rank"] == 428_998_656
    assert bf16["transaction_peak_residual_bytes_per_rank"] == 1_143_996_416
    assert bf16["maximum_native_tensor_bf16_staging_bytes_per_rank"] == 50_331_648
    assert bf16["incremental_transaction_peak_bytes_per_rank"] == 1_194_328_064
    assert bf16["fused_prepare_hbm_bytes_per_rank_per_update_lower_bound"] == 2_001_993_728
    assert value["link_bytes_are_projection_not_measurement"] is True


def test_v82_bf16_rne_tie_and_error_bound_are_deterministic():
    lower_tie = subject.bf16_roundtrip_v82(1.00390625)
    upper_tie = subject.bf16_roundtrip_v82(1.01171875)
    assert lower_tie["decoded"] == 1.0
    assert upper_tie["decoded"] == 1.015625
    for value in (0.0, -0.0, 1.0e-37, -2.7182817, 3.0e38):
        result = subject.bf16_roundtrip_v82(value)
        assert math.isfinite(result["decoded"])
        assert result["absolute_error"] <= result["absolute_error_bound"]
    with pytest.raises(ValueError):
        subject.bf16_roundtrip_v82(3.4e38)


def test_v82_error_feedback_conserves_compensated_fp32_bits():
    residual = 0.0
    for update in (0.1, -0.2, 1.0e-7, 15.75, -3.14159):
        value = subject.prepare_bf16_element_v82(update, residual)
        reconstructed = subject.float32_v82(
            value["transmitted"] + value["next_residual"]
        )
        assert subject.float32_bits_v82(reconstructed) == subject.float32_bits_v82(
            value["compensated"]
        )
        assert value["absolute_error"] <= value["absolute_error_bound"]
        residual = value["next_residual"]


def test_v82_exact_fp32_fallback_preserves_input_bits_and_rejects_residual():
    values = [0.1, -0.0, 2.5, -7.0]
    result = subject.prepare_fp32_fallback_shard_v82(values, [0.0] * len(values))
    assert result["input_bits"] == result["transmitted_bits"]
    assert result["conversion_or_rescaling_used"] is False
    assert result["next_residual_bits"] == ["00000000"] * len(values)
    assert result["input_bits"][1] == "80000000"
    with pytest.raises(RuntimeError, match="cannot consume residual"):
        subject.prepare_fp32_fallback_shard_v82(values, [0.0, 0.0, 1e-7, 0.0])


def test_v82_residual_resume_commit_and_rollback_are_bit_exact():
    state = subject.new_residual_state_v82({"a": 4, "b": 3})
    updates = {"a": [0.1, 0.2, 0.3, 0.4], "b": [-0.1, -0.2, 0.7]}
    transaction = subject.prepare_residual_transaction_v82(state, updates, "step-1")
    committed = subject.commit_residual_transaction_v82(state, transaction)
    payload = subject.serialize_residual_checkpoint_v82(committed)
    resumed = subject.resume_residual_checkpoint_v82(payload)
    assert resumed == committed
    next_a = subject.prepare_residual_transaction_v82(committed, updates, "step-2")
    next_b = subject.prepare_residual_transaction_v82(resumed, updates, "step-2")
    assert next_a == next_b
    assert subject.rollback_residual_transaction_v82(committed, transaction) == state
    assert subject.rollback_residual_transaction_v82(state, transaction) == state


def test_v82_residual_state_corruption_order_and_unknown_generation_fail_closed():
    state = subject.new_residual_state_v82({"a": 2, "b": 1})
    corrupt = copy.deepcopy(state)
    corrupt["residual_bits_by_shard"][0][0] = "00000001"
    with pytest.raises(RuntimeError, match="identity"):
        subject.validate_residual_state_v82(corrupt)
    with pytest.raises(RuntimeError, match="order"):
        subject.prepare_residual_transaction_v82(
            state, {"b": [0.1], "a": [0.2, 0.3]}, "bad-order"
        )
    transaction = subject.prepare_residual_transaction_v82(
        state, {"a": [0.1, 0.2], "b": [0.3]}, "step"
    )
    unknown = subject.new_residual_state_v82({"a": 2, "b": 1}, version=9)
    with pytest.raises(RuntimeError, match="unknown generation"):
        subject.rollback_residual_transaction_v82(unknown, transaction)


def test_v82_seed_sharding_and_rank_reduction_are_order_deterministic():
    shards = subject.strided_seed_shards_v82(range(8))
    assert [item["indices"] for item in shards] == [
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    vectors = {3: [4.0], 1: [2.0], 0: [1.0], 2: [3.0]}
    assert subject.deterministic_rank_sum_v82(vectors) == (10.0,)
    with pytest.raises(ValueError, match="coverage"):
        subject.deterministic_rank_sum_v82({0: [1.0]})


def test_v82_antithetic_algebra_sign_swap_negates_coefficients():
    pairs = [
        {"direction_seed": 11, "reward_plus": 1.5, "reward_minus": 0.5},
        {"direction_seed": 12, "reward_plus": -1.0, "reward_minus": 2.0},
    ]
    value = subject.antithetic_coefficients_v82(pairs)
    swapped = subject.antithetic_coefficients_v82([
        {
            "direction_seed": item["direction_seed"],
            "reward_plus": item["reward_minus"],
            "reward_minus": item["reward_plus"],
        }
        for item in pairs
    ])
    assert value["coefficients"] == [1.0, -3.0]
    assert swapped["coefficients"] == [-1.0, 3.0]
    assert value["unpaired_reward_centering_used"] is False


def test_v82_dependency_gate_launches_only_for_post_optimization_top_three():
    for rank in (1, 2, 3):
        gate = _gate(rank)
        assert subject.validate_bottleneck_gate_v82(gate) == gate
        assert subject.live_launch_authorized_v82(gate) is True
    gate = _gate(4)
    assert subject.live_launch_authorized_v82(gate) is False
    assert gate["decision"] == "close_not_applicable_without_gpu_ablation"


def test_v82_dependency_gate_rejects_incomplete_or_unmeasured_evidence():
    mutations = (
        lambda item: item["dependency_beads"]["specialist-0j5.18"].__setitem__(
            "status", "in_progress"
        ),
        lambda item: item["post_optimization_profile"].__setitem__(
            "collective_link_bytes_measured", False
        ),
        lambda item: item["post_optimization_profile"].__setitem__(
            "protected_content_opened", True
        ),
        lambda item: item.__setitem__("decision", "launch_fp32_vs_bf16_ablation"),
    )
    for mutate in mutations:
        changed = copy.deepcopy(_gate(4))
        mutate(changed)
        body = {key: value for key, value in changed.items() if key != "content_sha256"}
        changed["content_sha256"] = subject.canonical_sha256_v82(body)
        with pytest.raises(RuntimeError):
            subject.validate_bottleneck_gate_v82(changed)


def test_v82_prospective_runner_fails_gate_before_importing_torch(tmp_path):
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps(_gate(4)), encoding="utf-8")
    code = f"""
import json, sys
import benchmark_eggroll_es_collective_compression_v82 as b
try:
    b.load_authorized_gate_v82(b.Path({str(gate)!r}))
except RuntimeError:
    pass
print(json.dumps({{'torch': 'torch' in sys.modules}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=builder.ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout) == {"torch": False}


def test_v82_preflight_block_order_and_summary_are_counterbalanced():
    blocks = []
    for seed in subject.REGISTERED_SEEDS_V82:
        for pair_index, order in enumerate(benchmark.block_orders_v82()):
            for arm in order:
                blocks.append({
                    "seed": seed,
                    "pair_index": pair_index,
                    "arm": arm,
                    "seconds": 2.0 if arm == "fp32_control" else 1.8,
                })
    summary = benchmark.summarize_blocks_v82(blocks)
    assert all(row["bf16_faster_pairs"] == 6 for row in summary["by_seed"])
    assert all(
        row["bf16_over_fp32_speed_median"] == pytest.approx(2.0 / 1.8)
        for row in summary["by_seed"]
    )
    blocks[0], blocks[1] = blocks[1], blocks[0]
    with pytest.raises(RuntimeError, match="order"):
        benchmark.summarize_blocks_v82(blocks)


def test_v82_live_evidence_requires_three_seed_quality_ood_and_systems_gain():
    prereg_sha = "1" * 64
    evidence = _live(prereg_sha)
    assert subject.validate_live_evidence_v82(evidence, prereg_sha) == evidence
    for mutate in (
        lambda item: item["quality_results"].__setitem__(
            "ood_qa_reward_lcb", -0.020001
        ),
        lambda item: item["residual_results"].__setitem__(
            "rollback_bit_exact", False
        ),
        lambda item: item["integrity"].__setitem__(
            "fp8_collective_attempted", True
        ),
        lambda item: item["paired_systems_results"][0].__setitem__(
            "bf16_peak_residual_bytes", 1
        ),
    ):
        changed = copy.deepcopy(evidence)
        mutate(changed)
        body = {key: value for key, value in changed.items() if key != "content_sha256"}
        changed["content_sha256"] = subject.canonical_sha256_v82(body)
        with pytest.raises(RuntimeError):
            subject.validate_live_evidence_v82(changed, prereg_sha)


def test_v82_promotion_rejects_payload_saving_without_throughput_gain():
    prereg_sha = "2" * 64
    evidence = _live(prereg_sha)
    for row in evidence["paired_systems_results"]:
        row["bf16_throughput_updates_per_second"] = 0.9
    body = {key: value for key, value in evidence.items() if key != "content_sha256"}
    evidence["content_sha256"] = subject.canonical_sha256_v82(body)
    with pytest.raises(RuntimeError, match="systems improvement"):
        subject.validate_live_evidence_v82(evidence, prereg_sha)
    evidence["promotion_decision"] = "retain_fp32_control"
    body = {key: value for key, value in evidence.items() if key != "content_sha256"}
    evidence["content_sha256"] = subject.canonical_sha256_v82(body)
    assert subject.validate_live_evidence_v82(evidence, prereg_sha) == evidence


def test_v82_float32_checkpoint_rejects_nonfinite_duplicate_and_hash_drift():
    state = subject.new_residual_state_v82({"x": 1})
    payload = subject.serialize_residual_checkpoint_v82(state)
    duplicate = payload.replace(b'{"canonical_master', b'{"schema":"x","canonical_master', 1)
    with pytest.raises(RuntimeError):
        subject.resume_residual_checkpoint_v82(duplicate)
    changed = json.loads(payload)
    changed["residual_bits_by_shard"][0][0] = f"{struct.unpack('<I', struct.pack('<f', math.inf))[0]:08x}"
    changed_payload = (json.dumps(changed, sort_keys=True) + "\n").encode()
    with pytest.raises((ValueError, RuntimeError)):
        subject.resume_residual_checkpoint_v82(changed_payload)
