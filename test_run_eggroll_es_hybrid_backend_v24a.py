import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import eggroll_es_hybrid_backend_preregistration_v24a as prereg
import run_eggroll_es_hybrid_backend_v24a as runtime
import train_eggroll_es_hybrid_backend_v24a as mechanics


def _prereg():
    return runtime.load_preregistration_v24a()


def test_v24a_recipe_exactly_binds_models_arms_basis_and_all_four_gpus():
    frozen = _prereg()
    recipe = runtime.recipe_v24a(frozen, {"bundle_sha256": "a" * 64})
    assert recipe["arms"] == frozen["arms"]
    assert recipe["fresh_basis"] == frozen["fresh_basis"]
    assert recipe["panel_contract"] == frozen["panel_contract"]
    assert recipe["analysis"] == frozen["analysis"]
    assert recipe["gate"] == frozen["gate"]
    assert recipe["runtime"]["engine_arm_mapping"] == {
        "0": "bf16_a", "1": "hybrid_a", "2": "bf16_b", "3": "hybrid_b",
    }
    assert recipe["model_contract"]["bf16"]["path"] == (
        "/home/catid/specialist/models/Qwen3.6-35B-A3B"
    )
    assert recipe["model_contract"]["hybrid"]["path"].endswith(
        "Qwen3.6-35B-A3B-FP8-middle-late-BF16-v24"
    )
    assert recipe["content_sha256_before_self_field"] == runtime.canonical_sha256(
        runtime._without_self(recipe)
    )


def test_v24a_exact_64_wave_seed_repair_and_request_accounting():
    frozen = _prereg()
    seeds = prereg.perturbation_seeds_v24a()
    schedule = prereg.signed_wave_schedule_v24a()
    contract = runtime.seed_projection_contract_v24a(frozen)
    assert len(seeds) == len(set(seeds)) == 32
    assert len(schedule) == 64
    assert runtime.canonical_sha256(schedule) == frozen["fresh_basis"][
        "signed_wave_schedule_sha256"
    ]
    assert contract["direction_seed_list_sha256"] == frozen["fresh_basis"][
        "direction_seed_list_sha256"
    ]
    projections = [
        {"full_seed": seed, "numpy_legacy_seed": seed % (2**32)} for seed in seeds
    ]
    assert contract["full_to_numpy_projection_sha256"] == runtime.canonical_sha256(
        projections
    )
    recipe = runtime.recipe_v24a(frozen, {"bundle_sha256": "a" * 64})
    guard = recipe["matched_full_context_guard"]
    assert guard["reference_requests_all_engines"] == 1_120
    assert guard["guard_requests_all_engines"] == 2_240
    assert guard["total_generation_requests_including_guards"] == 75_040
    assert frozen["runtime"]["perturbed_requests_all_engines"] == 71_680


def test_v24a_full_context_guard_order_is_fail_closed_and_excluded_from_timing():
    source = inspect.getsource(runtime.HybridBackendRuntimeMixinV24A.estimate_hybrid_backend_v24a)
    positions = [source.index(token) for token in (
        "reference_batches", "phase_a =", "repeat_batches", "phase_b =",
        "if not all(pre_equal.values())", "for wave in schedule:",
        "boundary_sha =", "post_batches", "phase_c =",
        "if not all(post_equal.values())",
    )]
    assert positions == sorted(positions)
    assert "self._generate_v24a(requests, timed=True)" in source
    assert source.count("self._generate_v24a(requests, timed=False)") == 3
    recipe = runtime.recipe_v24a(_prereg(), {"bundle_sha256": "a" * 64})
    timing = recipe["timing_contract"]
    assert timing["only_64_perturbed_full_context_calls_timed"] is True
    assert timing["a_b_c_guard_calls_excluded"] is True
    assert recipe["matched_full_context_guard"][
        "guard_calls_excluded_from_quality_speed_and_bootstrap_endpoints"
    ] is True


def test_v24a_actor_adapter_uses_internal_timing_nvml_and_exact_models():
    source = inspect.getsource(runtime.load_runtime_trainer_v24a)
    assert "class TimedESNcclLLMV24A" in source
    assert "time.perf_counter_ns()" in source
    assert source.count("torch.cuda.synchronize()") == 2
    assert "pynvml.nvmlDeviceGetMemoryInfo" in source
    assert 'model=preregistration["arms"][arm]["model_path"]' in source
    assert "for rank, arm in enumerate(prereg.ARM_ORDER_V24A)" in source
    assert "worker_extension_cls=WORKER_EXTENSION" in source
    assert "tensor_parallel_size=1" in source
    assert 'moe_backend="triton"' in source


def test_v24a_restore_and_unselected_audit_are_after_every_wave_and_population():
    source = inspect.getsource(runtime.HybridBackendRuntimeMixinV24A)
    loop = inspect.getsource(runtime.HybridBackendRuntimeMixinV24A.estimate_hybrid_backend_v24a)
    assert "restore_hashes.append(self._restore_v24a())" in loop
    assert "len(restore_hashes) != 64" in loop
    assert loop.index("restore_hashes.append") < loop.index("boundary_sha = self._boundary_v24a()")
    assert '"audit_population_completion_v4"' in source
    recipe = runtime.recipe_v24a(_prereg(), {"bundle_sha256": "a" * 64})
    restoration = recipe["restoration_contract"]
    assert restoration[
        "selected_reference_copy_restore_and_exact_hash_every_signed_wave"
    ] is True
    assert restoration[
        "full_selected_and_unselected_byte_audit_after_all_64_waves"
    ] is True


def test_v24a_metric_endpoints_match_identical_backends():
    directions = np.arange(160, dtype=np.float64).reshape(5, 32) + 1.0
    values = mechanics._metric_endpoints(directions, directions.copy())
    assert set(values) == set(tuple(prereg.QUALITY_ENDPOINT_THRESHOLDS_V24A)[:12])
    for name, value in values.items():
        if "cosine" in name or "sign_agreement" in name:
            assert float(value) == pytest.approx(1.0)


def test_v24a_compact_estimator_executes_bootstrap_without_persisting_raw_arrays(
    monkeypatch,
):
    repetitions = 16
    monkeypatch.setattr(mechanics, "REPETITIONS", repetitions)
    monkeypatch.setattr(
        mechanics.anchor_v13, "validate_panel_bundle_v13", lambda _bundle: None,
    )
    strata = []
    for stratum in mechanics.anchor_v13.panel_sampler.STRATA:
        strata.extend(
            [stratum] * mechanics.anchor_v13.panel_sampler.STRATUM_QUOTAS[stratum]
        )
    panel_bundle = {
        "panels": {
            panel: {"weights": [310.0 / 56.0] * 56, "strata": strata}
            for panel in prereg.PANEL_NAMES_V24A
        }
    }
    direction_signal = np.linspace(-1.0, 1.0, 32, dtype=np.float64)
    unit_scores = {}
    reference_scores = {}
    reference = np.tile(np.linspace(0.1, 0.9, 56), (5, 1))
    for arm in prereg.ARM_ORDER_V24A:
        values = np.empty((5, 2, 32, 56), dtype=np.float64)
        for panel in range(5):
            values[panel, 0] = direction_signal[:, None] + panel * 0.01
            values[panel, 1] = -direction_signal[:, None] + panel * 0.01
        unit_scores[arm] = values
        reference_scores[arm] = reference.copy()
    timings = np.ones((64, 4), dtype=np.float64)
    timings[:, 0] = timings[:, 2] = 2.0
    resident_bytes = {
        "bf16_a": 100, "hybrid_a": 50, "bf16_b": 100, "hybrid_b": 50,
    }
    integrity = {
        arm: {key: True for key in mechanics.RUNTIME_INTEGRITY_KEYS}
        for arm in prereg.ARM_ORDER_V24A
    }
    result = mechanics.build_compact_summary_v24a(
        unit_scores, reference_scores, timings, resident_bytes,
        panel_bundle, integrity,
    )
    assert result["global_pass"] is True
    assert result["bootstrap"]["repetitions"] == repetitions
    assert result["raw_scores_timing_vectors_bootstrap_draws_or_replicates_persisted"] is False
    assert not ({"unit_scores", "bootstrap_draws", "bootstrap_replicates"} & set(
        runtime.runtime_v23a._recursive_keys_v23a(result)
    ))


def test_v24a_real_launch_hashes_and_backend_environment_fail_closed(monkeypatch):
    frozen = _prereg()
    implementation = {"bundle_sha256": "a" * 64}
    recipe = {"content_sha256_before_self_field": "b" * 64}
    with pytest.raises(ValueError, match="requires expected implementation"):
        runtime.validate_runtime_v24a(
            SimpleNamespace(
                v24a_dry_run=False,
                expected_implementation_bundle_sha256=None,
                expected_recipe_sha256=None,
            ), frozen, implementation, recipe,
        )
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with pytest.raises(ValueError, match="batch-invariant backend swap"):
        runtime.validate_runtime_v24a(
            SimpleNamespace(
                v24a_dry_run=True,
                expected_implementation_bundle_sha256=None,
                expected_recipe_sha256=None,
            ), frozen, implementation, recipe,
        )


def test_v24a_update_eval_checkpoint_and_union_surfaces_are_closed():
    frozen = _prereg()
    recipe = runtime.recipe_v24a(frozen, {"bundle_sha256": "a" * 64})
    assert all(recipe["authority"][key] is False for key in (
        "model_update_allowed", "checkpoint_write_allowed",
        "evaluation_allowed", "dataset_promotion_allowed", "backend_adoption_allowed",
    ))
    for name in (
        "train_step", "apply_seed_coefficients", "eval_step", "fit",
        "configure_anchor", "configure_train_panels_v13",
    ):
        with pytest.raises(RuntimeError, match="closes update checkpoint evaluation"):
            getattr(runtime.HybridBackendRuntimeMixinV24A, name)()
    factory = inspect.getsource(runtime._make_trainer_v24a)
    assert "checkpoint=None" in factory
    assert "alpha=0.0" in factory
    assert "eval_dataloader_dict={}" in factory
    assert "save_best_models=False" in factory


def test_v24a_fresh_paths_source_provenance_and_o_excl_claim_precede_trainer():
    source = inspect.getsource(runtime.run_exact_v24a)
    assert source.index("_source_provenance_v23a") < source.index(
        "_exclusive_write_json_v23a(attempt_path, attempt)"
    )
    assert source.index("validate_live_contract_v24a") < source.index(
        "_exclusive_write_json_v23a(attempt_path, attempt)"
    )
    assert source.index("_exclusive_write_json_v23a(attempt_path, attempt)") < source.index(
        "_make_trainer_v24a"
    )
    assert source.index("if run_dir.exists()") < source.index("_make_trainer_v24a")
    assert "fresh_run_reservation_race" in source
    attempt = runtime.OUTPUT_DIRECTORY / runtime.ATTEMPT_NAME
    report = runtime.OUTPUT_DIRECTORY / runtime.EXPERIMENT_NAME / runtime.REPORT_NAME
    assert attempt != report
    assert not attempt.exists() and not report.exists()


def test_v24a_dry_run_constructs_no_trainer_or_gpu(monkeypatch, capsys):
    frozen = _prereg()
    monkeypatch.setattr(runtime, "load_preregistration_v24a", lambda: frozen)
    monkeypatch.setattr(
        runtime, "implementation_identity_v24a",
        lambda: {"bundle_sha256": "a" * 64, "files": {}},
    )
    monkeypatch.setattr(
        runtime, "_make_trainer_v24a",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("trainer created")),
    )
    value = runtime.main(["--v24a-dry-run"])
    assert json.loads(capsys.readouterr().out) == value
    assert value["gpu_launched"] is False
    assert value["matched_full_context_guard_active"] is True
    assert value["content_sha256_before_self_field"] == runtime.canonical_sha256(
        runtime._without_self(value)
    )
