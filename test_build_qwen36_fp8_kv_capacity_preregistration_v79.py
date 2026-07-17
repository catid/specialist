import inspect
import subprocess

import build_qwen36_fp8_kv_capacity_preregistration_v79 as subject


def test_derivation_selects_lowest_safe_point_on_sealed_grid():
    value = subject.derive_utilization_v79()
    assert value["selected"]["gpu_memory_utilization"] == 0.485
    assert value["selected"]["projected_block_aligned_tokens"] == 161_792
    assert value["next_lower_rejected"]["gpu_memory_utilization"] == 0.484
    assert value["next_lower_rejected"][
        "projected_block_aligned_tokens"
    ] < 161_792
    assert value["minimum_margin_tokens"] == 4_096


def test_builder_binds_current_v76_and_v78_r3_not_historical_r1():
    value = subject.build_preregistration_v79()
    sealed = value["sealed_evidence"]
    assert sealed["v76_bf16_kv"]["inventory"]["bundle_sha256"] == (
        subject.V76_RUN_BUNDLE_SHA256
    )
    assert sealed["v78_fp8_per_token_head_r3"]["inventory"][
        "bundle_sha256"
    ] == subject.V78_RUN_BUNDLE_SHA256
    assert sealed["v78_fp8_per_token_head_r3"][
        "capacity_tokens_per_actor"
    ] == [198_656] * 4
    assert sealed["v78_r1_historical_only_not_launch_ancestry"][
        "run_bundle_sha256"
    ] != subject.V78_RUN_BUNDLE_SHA256


def test_contract_binds_cache_output_semantic_ood_logs_and_cleanup():
    value = subject.build_preregistration_v79()
    gates = value["live_acceptance"]
    assert gates["capacity"]["minimum_tokens_per_actor"] == 161_792
    assert gates["hybrid_cache"]["mamba_ssm_cache_dtype_exact"] == "float32"
    assert gates["hybrid_cache"]["attention_backend_log_exact"] == "TRITON_ATTN"
    assert gates["output"]["candidate_repeat_exact_at_token_hash_level"] is True
    assert gates["semantic"]["source_disjoint_paired_evaluation_required"] is True
    assert gates["protected_ood"]["no_retuning_after_open"] is True
    assert gates["performance_and_memory"][
        "per_actor_generated_tokens_per_second_required"
    ] is True
    assert gates["performance_and_memory"][
        "per_actor_median_p95_and_max_call_latency_required"
    ] is True
    assert gates["performance_and_memory"][
        "sampled_pcie_rx_tx_byte_integrals_required"
    ] is True
    assert gates["performance_and_memory"][
        "hbm_bytes_per_second_must_not_be_inferred_from_memory_utilization"
    ] is True
    assert "CUDA out of memory" in gates["logs"]["forbidden"]
    assert gates["cleanup"]["minimum_consecutive_post_exit_idle_batches"] == 2
    assert gates["promotion"][
        "scored_training_checkpoint_or_layout_promotion_default"
    ] is False


def test_parameter_residency_and_physical_memory_are_sealed():
    value = subject.build_preregistration_v79()
    sealed = value["sealed_evidence"]
    residency = sealed["current_parameter_residency"]
    assert residency["total_parameter_count"] == 813
    assert residency["total_logical_bytes"] == 35_712_084_096
    assert list(residency["components"]) == ["language"]
    assert sealed["hardware"]["memory_total_mib_per_gpu"] == 97_887
    assert sealed["hardware"]["physical_gpu_ids"] == [0, 1, 2, 3]


def test_preregistration_self_hash_and_authority():
    value = subject.build_preregistration_v79()
    claimed = value.pop("content_sha256_before_self_field")
    assert subject.canonical_sha256_v79(value) == claimed
    assert value["authority"]["gpu_launch_performed_by_this_build"] is False
    assert value["authority"]["dataset_or_protected_data_opened"] is False
    assert value["authority"]["scored_or_training_authority"] is False


def test_builder_is_cpu_only_and_launcher_is_syntax_valid():
    source = inspect.getsource(subject)
    assert "import torch" not in source
    assert "import vllm" not in source
    assert "nvidia-smi" not in source
    subprocess.run(
        ["bash", "-n", str(subject.LAUNCHER)],
        check=True,
        capture_output=True,
        text=True,
    )
    launcher = subject.LAUNCHER.read_text(encoding="utf-8")
    assert "for gpu in 0 1 2 3" in launcher
    assert "--sample-interval-seconds 0.5" in launcher
    assert "--require-pcie" in launcher
    assert "--precision-arm fp8_serialized" in launcher
    monitor = subject.MONITOR.read_text(encoding="utf-8")
    assert "utilization.memory" in monitor
    assert "nvmlDeviceGetPcieThroughput" in monitor
    assert "hbm_bytes_per_second_inferred" in monitor


def test_checked_artifact_matches_builder():
    assert subject.main.__name__ == "main"
    expected = subject.build_preregistration_v79()
    actual = subject._json(subject.OUTPUT)
    assert actual == expected
