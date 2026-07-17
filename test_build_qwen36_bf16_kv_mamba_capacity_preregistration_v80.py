import inspect
import subprocess

import build_qwen36_bf16_kv_mamba_capacity_preregistration_v80 as subject


def test_derivation_selects_lowest_safe_point_on_sealed_grid():
    value = subject.derive_utilization_v80()
    assert value["selected"]["gpu_memory_utilization"] == 0.479
    assert value["selected"]["projected_complete_context_tokens"] == 161_792
    assert value["selected"]["projected_full_2048_token_contexts"] == 79
    assert value["next_lower_rejected"]["gpu_memory_utilization"] == 0.478
    assert value["next_lower_rejected"][
        "projected_complete_context_tokens"
    ] == 157_696
    assert value["minimum_margin_tokens_over_v76"] == 4_096


def test_builder_binds_current_v76_and_reviewed_v78c_r1():
    value = subject.build_preregistration_v80()
    sealed = value["sealed_evidence"]
    assert sealed["v76_control"]["inventory"]["bundle_sha256"] == (
        subject.V76_RUN_BUNDLE_SHA256
    )
    v78c = sealed["v78c_r1"]
    assert v78c["inventory"]["bundle_sha256"] == (
        subject.V78C_RUN_BUNDLE_SHA256
    )
    assert v78c["inventory"]["file_count"] == 9
    assert v78c["capacity_tokens_per_actor"] == [218_843] * 4
    assert v78c["max_concurrency_per_actor"] == [106.86] * 4
    assert v78c["attention_backend"] == "FLASH_ATTN"
    assert v78c["model_config_fp32_ssm_override_warning_actor_count"] == 4


def test_contract_binds_cache_output_semantic_ood_logs_and_cleanup():
    value = subject.build_preregistration_v80()
    gates = value["live_acceptance"]
    assert gates["capacity"]["minimum_tokens_per_actor"] == 161_792
    assert gates["capacity"]["minimum_full_2048_token_contexts_per_actor"] == 79
    assert gates["hybrid_cache"]["mamba_ssm_cache_dtype_exact"] == "bfloat16"
    assert gates["hybrid_cache"]["effective_attention_kv_dtype"] == "bfloat16"
    assert gates["hybrid_cache"]["attention_backend_log_exact"] == "FLASH_ATTN"
    assert gates["output"]["candidate_repeat_exact_at_token_hash_level"] is True
    assert gates["semantic"]["source_disjoint_paired_evaluation_required"] is True
    assert gates["semantic"]["explicit_mamba_ssm_override_requires_strict_review"] is True
    assert gates["protected_ood"]["no_retuning_after_open"] is True
    perf = gates["performance_and_memory"]
    assert perf["per_actor_generated_tokens_per_second_required"] is True
    assert perf["per_actor_median_p95_and_max_call_latency_required"] is True
    assert perf["sample_interval_seconds_max"] == 1.0
    assert perf["sampled_pcie_rx_tx_byte_integrals_required"] is True
    assert perf["hbm_bytes_per_second_must_not_be_inferred_from_memory_utilization"] is True
    assert "CUDA out of memory" in gates["logs"]["forbidden"]
    assert subject.OVERRIDE_WARNING in gates["logs"]["required_once_per_actor"]
    assert gates["cleanup"]["minimum_consecutive_post_exit_idle_batches"] == 3
    assert gates["cleanup"]["post_exit_memory_used_mib_max"] == 4
    assert gates["promotion"][
        "scored_training_checkpoint_or_layout_promotion_default"
    ] is False


def test_parameter_residency_physical_memory_and_projection_are_sealed():
    value = subject.build_preregistration_v80()
    sealed = value["sealed_evidence"]
    residency = sealed["v78c_r1"]["parameter_residency"]
    assert residency["total_parameter_count"] == 813
    assert residency["total_logical_bytes"] == 35_712_084_096
    assert list(residency["components"]) == ["language"]
    assert sealed["hardware"]["memory_total_mib_per_gpu"] == 97_887
    assert sealed["hardware"]["physical_gpu_ids"] == [0, 1, 2, 3]
    perf = value["live_acceptance"]["performance_and_memory"]
    assert perf["peak_memory_used_mib_per_actor_max"] == 50_808
    assert perf["projected_peak_memory_used_mib_not_a_gate"] == 48_752.373
    assert perf["minimum_physical_headroom_mib_per_actor"] == 47_079


def test_preregistration_self_hash_and_authority():
    value = subject.build_preregistration_v80()
    claimed = value.pop("content_sha256_before_self_field")
    assert subject.common.canonical_sha256_v79(value) == claimed
    assert value["authority"]["gpu_launch_performed_by_this_build"] is False
    assert value["authority"]["dataset_or_protected_data_opened"] is False
    assert value["authority"]["scored_or_training_authority"] is False


def test_builder_is_cpu_only_and_launcher_and_monitor_are_bound():
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
    assert "--max-cleanup-wait-seconds 60" in launcher
    assert "--require-pcie" in launcher
    assert "--precision-arm fp8_serialized" in launcher
    monitor = subject.MONITOR.read_text(encoding="utf-8")
    assert "utilization.memory" in monitor
    assert "nvmlDeviceGetPcieThroughput" in monitor
    assert "hbm_bytes_per_second_inferred" in monitor
    assert subject.common.file_sha256_v79(subject.MONITOR) == (
        subject.SOURCE_SHA256[subject.MONITOR.name]
    )


def test_checked_artifact_matches_builder():
    expected = subject.build_preregistration_v80()
    actual = subject._json(subject.OUTPUT)
    assert actual == expected
