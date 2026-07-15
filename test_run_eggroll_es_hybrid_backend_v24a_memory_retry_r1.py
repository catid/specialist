import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import run_eggroll_es_hybrid_backend_v24a_memory_retry_r1 as runtime


def _prereg():
    return runtime.base.load_preregistration_v24a()


def _failure():
    return runtime.validate_original_memory_failure_r1()


def _valid_model_load_bytes():
    return {
        "bf16_a": 69_459_804_160,
        "hybrid_a": 35_991_420_928,
        "bf16_b": 69_459_804_160,
        "hybrid_b": 35_991_420_928,
    }


def test_v24a_r1_recipe_changes_only_memory_instrumentation_and_fresh_paths():
    frozen = _prereg()
    implementation = {"bundle_sha256": "a" * 64}
    retry = runtime.recipe_r1(frozen, implementation, _failure())
    inherited = runtime.base.recipe_v24a(frozen, implementation)
    for key in (
        "arms", "pairing", "fresh_basis", "panel_contract", "runtime",
        "matched_full_context_guard", "timing_contract", "restoration_contract",
        "hybrid_checkpoint_audit", "model_contract", "authority",
    ):
        assert retry[key] == inherited[key]
    assert retry["analysis"]["bootstrap_repetitions"] == 50_000
    assert retry["analysis"]["quality_endpoint_thresholds"] == inherited[
        "analysis"
    ]["quality_endpoint_thresholds"]
    assert retry["analysis"]["speedup_threshold_per_pair"] == 1.05
    assert retry["analysis"]["memory_reduction_threshold_per_pair"] == 0.40
    assert "model_load_consumed_bytes" in retry["analysis"][
        "memory_reduction_definition"
    ]
    assert retry["memory_endpoint_repair_r1"][
        "preflight_before_reference_a_b_c_or_first_perturbation"
    ] is True
    assert retry["nvml_memory_contract"]["diagnostic_only"] is True
    assert retry["seed_domain_repair"]["worker_extension"] == (
        runtime.WORKER_EXTENSION_R1
    )
    assert retry["content_sha256_before_self_field"] == runtime.canonical_sha256(
        runtime._without_self(retry)
    )


def test_v24a_r1_paths_are_fresh_disjoint_and_unclaimed():
    retry_paths = {
        runtime.OUTPUT_DIRECTORY_R1 / runtime.ATTEMPT_NAME_R1,
        runtime.OUTPUT_DIRECTORY_R1 / runtime.EXPERIMENT_NAME_R1,
        runtime.OUTPUT_DIRECTORY_R1 / runtime.EXPERIMENT_NAME_R1
        / runtime.REPORT_NAME_R1,
    }
    earlier = {
        runtime.ORIGINAL_ATTEMPT_PATH_R1,
        runtime.ORIGINAL_REPORT_PATH_R1.parent,
        runtime.ORIGINAL_REPORT_PATH_R1,
    }
    assert retry_paths.isdisjoint(earlier)
    assert all(not path.exists() for path in retry_paths)


def test_v24a_r1_model_load_preflight_is_exact_and_fail_closed():
    values = _valid_model_load_bytes()
    preflight = runtime.mechanics_r1.validate_model_load_memory_preflight_r1(values)
    assert preflight["duplicate_bf16_values_exact"] is True
    assert preflight["duplicate_hybrid_values_exact"] is True
    assert preflight["both_pairs_meet_threshold"] is True
    assert preflight["reduction_by_pair"]["pair_a"] > 0.40
    assert preflight["nvml_excluded_from_gate"] is True
    for arm in ("bf16_b", "hybrid_b"):
        changed = dict(values)
        changed[arm] += 1
        with pytest.raises(RuntimeError, match="duplicate"):
            runtime.mechanics_r1.validate_model_load_memory_preflight_r1(changed)
    changed = dict(values)
    changed["hybrid_a"] = changed["hybrid_b"] = 50_000_000_000
    with pytest.raises(RuntimeError, match="below 0.40"):
        runtime.mechanics_r1.validate_model_load_memory_preflight_r1(changed)
    changed = dict(values)
    changed["bf16_a"] = True
    with pytest.raises(RuntimeError, match="coverage changed"):
        runtime.mechanics_r1.validate_model_load_memory_preflight_r1(changed)


def test_v24a_r1_memory_preflight_precedes_all_requests_guards_and_perturbations():
    configure = inspect.getsource(
        runtime.HybridBackendMemoryRetryMixinV24AR1.configure_hybrid_backend_v24a
    )
    estimate = inspect.getsource(
        runtime.HybridBackendMemoryRetryMixinV24AR1.estimate_hybrid_backend_v24a
    )
    assert configure.index('"model_load_memory_v24a_r1"') < configure.index(
        "validate_model_load_memory_preflight_r1"
    )
    assert configure.index("validate_model_load_memory_preflight_r1") < configure.index(
        "return _seal(configured)"
    )
    assert 'configured.pop("nvml_resident_bytes", None)' in configure
    assert '"nvml_resident_bytes_diagnostic"' in configure
    assert '"nvml_excluded_from_all_gates": True' in configure
    assert estimate.index('hasattr(self, "_v24a_r1_memory_preflight")') < estimate.index(
        "_prepared_requests_v24a()"
    )
    assert estimate.index("_prepared_requests_v24a()") < estimate.index("phase_a =")
    assert estimate.index("phase_a =") < estimate.index("for wave in schedule:")


def test_v24a_r1_loader_uses_exact_worker_models_timing_and_diagnostic_nvml():
    source = inspect.getsource(runtime.load_runtime_trainer_r1)
    assert "class TimedESNcclLLMV24AR1" in source
    assert source.count("torch.cuda.synchronize()") == 2
    assert "time.perf_counter_ns()" in source
    assert "pynvml.nvmlDeviceGetMemoryInfo" in source
    assert "worker_extension_cls=WORKER_EXTENSION_R1" in source
    assert 'model=preregistration["arms"][arm]["model_path"]' in source
    assert "for rank, arm in enumerate(prereg.ARM_ORDER_V24A)" in source
    assert "tensor_parallel_size=1" in source
    assert "gpu_memory_utilization=0.82" in source
    assert 'moe_backend="triton"' in source


def test_v24a_r1_compact_estimator_gates_model_load_not_equalized_nvml(monkeypatch):
    mechanics = runtime.mechanics_r1
    repetitions = 16
    monkeypatch.setattr(mechanics.base, "REPETITIONS", repetitions)
    monkeypatch.setattr(
        mechanics.base.anchor_v13, "validate_panel_bundle_v13", lambda _bundle: None,
    )
    strata = []
    for stratum in mechanics.base.anchor_v13.panel_sampler.STRATA:
        strata.extend(
            [stratum]
            * mechanics.base.anchor_v13.panel_sampler.STRATUM_QUOTAS[stratum]
        )
    panel_bundle = {
        "panels": {
            panel: {"weights": [310.0 / 56.0] * 56, "strata": strata}
            for panel in runtime.prereg.PANEL_NAMES_V24A
        }
    }
    direction_signal = np.linspace(-1.0, 1.0, 32, dtype=np.float64)
    unit_scores = {}
    reference_scores = {}
    reference = np.tile(np.linspace(0.1, 0.9, 56), (5, 1))
    for arm in runtime.prereg.ARM_ORDER_V24A:
        values = np.empty((5, 2, 32, 56), dtype=np.float64)
        for panel in range(5):
            values[panel, 0] = direction_signal[:, None] + panel * 0.01
            values[panel, 1] = -direction_signal[:, None] + panel * 0.01
        unit_scores[arm] = values
        reference_scores[arm] = reference.copy()
    timings = np.ones((64, 4), dtype=np.float64)
    timings[:, 0] = timings[:, 2] = 2.0
    nvml = {
        "bf16_a": 86_491_004_928,
        "hybrid_a": 86_516_170_752,
        "bf16_b": 86_491_004_928,
        "hybrid_b": 86_516_170_752,
    }
    integrity = {
        arm: {key: True for key in mechanics.RUNTIME_INTEGRITY_KEYS_R1}
        for arm in runtime.prereg.ARM_ORDER_V24A
    }
    result = mechanics.build_compact_summary_v24a_memory_retry_r1(
        unit_scores,
        reference_scores,
        timings,
        _valid_model_load_bytes(),
        nvml,
        panel_bundle,
        integrity,
    )
    assert result["global_pass"] is True
    assert result["bootstrap"]["repetitions"] == repetitions
    for pair in runtime.prereg.PAIR_ORDER_V24A:
        assert result["pairs"][pair]["memory"]["gate_input"] is True
        assert "model_load_consumed_bytes" in result["pairs"][pair]["memory"][
            "endpoint"
        ]
        diagnostic = result["pairs"][pair]["nvml_resident_memory_diagnostic"]
        assert diagnostic["gate_input"] is False
        assert diagnostic[
            "excluded_from_quality_speed_memory_and_global_pass"
        ] is True
        assert "bf16_resident_bytes" not in result["pairs"][pair]["memory"]
    assert result["memory_gate_uses_only_vllm_model_load_consumed_bytes"] is True
    assert result["nvml_resident_bytes_retained_only_as_diagnostic"] is True
    assert "nvml_resident_bytes" not in result


def test_v24a_r1_preserves_64_wave_guard_restore_and_compact_boundaries():
    source = inspect.getsource(
        runtime.HybridBackendMemoryRetryMixinV24AR1.estimate_hybrid_backend_v24a
    )
    positions = [source.index(token) for token in (
        "reference_batches", "phase_a =", "repeat_batches", "phase_b =",
        "if not all(pre_equal.values())", "for wave in schedule:",
        "boundary_sha =", "post_batches", "phase_c =",
        "if not all(post_equal.values())",
    )]
    assert positions == sorted(positions)
    assert "len(restore_hashes) != 64" in source
    assert "len(dense_hashes) != 1_280" in source
    assert "perturbed_requests_all_engines\": 71_680" in source
    assert "total_generation_requests\": 75_040" in source
    assert "_assert_compact_persistence_v23a" in source


def test_v24a_r1_real_launch_hashes_and_environment_fail_closed(monkeypatch):
    frozen = _prereg()
    implementation = {"bundle_sha256": "a" * 64}
    recipe = {"content_sha256_before_self_field": "b" * 64}
    with pytest.raises(ValueError, match="requires expected implementation"):
        runtime.validate_runtime_r1(
            SimpleNamespace(
                v24a_r1_dry_run=False,
                expected_implementation_bundle_sha256=None,
                expected_recipe_sha256=None,
            ),
            frozen,
            implementation,
            recipe,
        )
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with pytest.raises(ValueError, match="batch-invariant backend swap"):
        runtime.validate_runtime_r1(
            SimpleNamespace(
                v24a_r1_dry_run=True,
                expected_implementation_bundle_sha256=None,
                expected_recipe_sha256=None,
            ),
            frozen,
            implementation,
            recipe,
        )


def test_v24a_r1_provenance_environment_evidence_and_o_excl_precede_trainer():
    source = inspect.getsource(runtime.run_exact_r1)
    assert source.index("certify_runtime_environment_r2()") < source.index(
        "_exclusive_write_json_v23a(attempt_path, attempt)"
    )
    assert source.index("_source_provenance_v23a") < source.index(
        "_exclusive_write_json_v23a(attempt_path, attempt)"
    )
    assert source.index("_exclusive_write_json_v23a(attempt_path, attempt)") < source.index(
        "_make_trainer_r1"
    )
    assert source.index("if run_dir.exists()") < source.index("_make_trainer_r1")
    main_source = inspect.getsource(runtime.main)
    assert main_source.index("validate_original_memory_failure_r1()") < main_source.index(
        "implementation_identity_r1()"
    )
    assert main_source.index("validate_original_memory_failure_r1()") < main_source.index(
        "run_exact_r1"
    )


def test_v24a_r1_update_eval_checkpoint_and_union_surfaces_remain_closed():
    frozen = _prereg()
    recipe = runtime.recipe_r1(
        frozen, {"bundle_sha256": "a" * 64}, _failure()
    )
    assert all(recipe["authority"][key] is False for key in (
        "model_update_allowed", "checkpoint_write_allowed", "evaluation_allowed",
        "dataset_promotion_allowed", "backend_adoption_allowed",
    ))
    factory = inspect.getsource(runtime._make_trainer_r1)
    assert "checkpoint=None" in factory
    assert "alpha=0.0" in factory
    assert "eval_dataloader_dict={}" in factory
    assert "save_best_models=False" in factory


def test_v24a_r1_dry_run_constructs_no_trainer_or_gpu(monkeypatch, capsys):
    frozen = _prereg()
    failure = _failure()
    monkeypatch.setattr(runtime, "validate_original_memory_failure_r1", lambda: failure)
    monkeypatch.setattr(runtime.base, "load_preregistration_v24a", lambda: frozen)
    monkeypatch.setattr(
        runtime,
        "implementation_identity_r1",
        lambda: {"bundle_sha256": "a" * 64, "files": {}},
    )
    monkeypatch.setattr(
        runtime,
        "_make_trainer_r1",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("trainer created")
        ),
    )
    value = runtime.main(["--v24a-r1-dry-run"])
    assert json.loads(capsys.readouterr().out) == value
    assert value["gpu_launched"] is False
    assert value["memory_preflight_precedes_a_b_c_and_first_perturbation"] is True
    assert value["nvml_is_diagnostic_only"] is True
    assert value[
        "real_launch_requires_exact_post_cherry_pick_implementation_and_recipe_hashes"
    ] is True
    assert value["content_sha256_before_self_field"] == runtime.canonical_sha256(
        runtime._without_self(value)
    )
