import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pytest

import run_eggroll_es_v401_replacement_fraction_v34c as runtime


def _foundation():
    preregistration = runtime.load_preregistration()
    panel_bundle = runtime.mechanics_v34b.materialize_paired_panel_bundle()
    layer_bundle = runtime.load_layer_bundle(preregistration)
    implementation = runtime.implementation_identity()
    recipe = runtime.recipe_v34c(
        preregistration, panel_bundle, layer_bundle, implementation
    )
    return preregistration, panel_bundle, layer_bundle, implementation, recipe


def test_v34c_binds_every_frozen_v34b_file_to_exact_commit():
    bindings = runtime.verify_frozen_v34b()
    assert runtime.V34B_COMMIT == "b254d4bdae0bb3fcb98d015c155393df9cca2d5d"
    for name in runtime.V34B_BOUND_FILES:
        assert bindings[name]["commit"] == runtime.V34B_COMMIT
        assert bindings[name]["file_sha256"] == runtime.V34B_BOUND_FILES[name][1]
    assert runtime.frame_v34b.CANDIDATE_SHA256 == (
        "8e29826dd389171c69f5eb6f43781f900345974c3d4d11274268e86c6145693b"
    )
    assert runtime.frame_v34b.PRODUCTION_SHA256 == (
        "62e7ae28c86a458d4d33bf3f73f1b91b873c86e3f70ce87706a7394d1f391507"
    )


def test_v34c_reuses_exact_v33a_runtime_and_worker_files():
    bindings = runtime.verify_frozen_v34b()
    for name, (_path, digest) in runtime.REUSED_V33A_FILES.items():
        assert bindings[name]["file_sha256"] == digest


def test_v34c_seed_projection_is_exact_fresh_v34b_basis():
    preregistration = runtime.load_preregistration()
    projection = runtime.seed_projection_contract(preregistration)
    assert projection["direction_count"] == 64
    assert projection["direction_seed_list_sha256"] == (
        "e1e45bc1965360d78a9a367c884b29fbd0c09bf00a2f83c36571fe09dc41bdf5"
    )
    assert projection["numpy_projection_unique_count"] == 64
    assert projection["worker_extension"] == runtime.WORKER_EXTENSION


def test_v34c_recipe_scores_two_raw_arms_and_adds_zero_fraction_requests():
    _prereg, _panels, _layer, _implementation, recipe = _foundation()
    assert recipe["sources"]["raw_arms_scored"] == [
        "production", "candidate_v401"
    ]
    assert recipe["sources"][
        "fraction_specific_raw_arm_or_model_request_count"
    ] == 0
    assert recipe["fraction_analysis"]["fractions_in_fixed_test_order"] == [
        0.05, 0.1, 0.2, 0.4, 1.0
    ]
    assert recipe["fraction_analysis"]["derived_algebraically_by_v34b_mechanics"] is True
    assert recipe["fraction_analysis"]["stop_at_first_failure"] is True


def test_v34c_recipe_exact_four_engine_schedule_and_request_budget():
    _prereg, _panels, _layer, _implementation, recipe = _foundation()
    perturbation = recipe["perturbation"]
    assert perturbation["population_size"] == 64
    assert perturbation["synchronized_signed_wave_count"] == 32
    assert perturbation["engine_signed_direction_evaluation_count"] == 128
    assert perturbation["all_four_tp1_engines_every_wave"] is True
    budget = recipe["request_budget"]
    assert budget["perturbed_requests"] == 49_920
    assert budget["full_context_requests"] == 4_680
    assert budget["fraction_specific_requests"] == 0
    assert budget["total_generation_requests"] == 54_600


def test_v34c_recipe_freezes_restore_boundary_origin_abc_activity_cleanup():
    _prereg, _panels, _layer, _implementation, recipe = _foundation()
    assert recipe["guards"] == {
        "full_context_A_B_before_population_and_C_after": True,
        "A_equals_B_and_A_equals_C_exact": True,
        "exact_selected_restore_after_each_signed_wave": True,
        "full_partition_population_boundary": True,
        "unselected_origin_required_by_full_partition_audit": True,
        "all_four_engine_activity_each_wave": True,
        "preclaim_and_final_all_four_gpu_idle": True,
        "fail_closed_cleanup": True,
    }


def test_v34c_recipe_closes_all_direct_authority_and_is_self_hashed():
    _prereg, _panels, _layer, _implementation, recipe = _foundation()
    assert recipe["content_sha256_before_self_field"] == runtime.canonical_sha256(
        runtime._without_self(recipe)
    )
    authority = recipe["authority"]
    assert authority["pass_authorizes_only_separately_frozen_train_only_recipe"] is True
    assert all(
        value is False
        for key, value in authority.items()
        if key != "pass_authorizes_only_separately_frozen_train_only_recipe"
    )


def test_v34c_allowlist_accepts_only_current_curator_and_exact_background_paths():
    result = runtime.validate_worktree_status(
        "?? data/manual_reviews/context_merit_audit_v407/a.json\n"
        "?? experiments/dataset_probes/a.json\n"
        "?? experiments/gpu_utilization_v31_base_qwen36_train_reward_probe.jsonl\n"
    )
    assert result["all_tracked_files_clean"] is True
    assert result["allowed_untracked_entry_count"] == 3
    for status in (
        " M run.py\n",
        "?? unknown.txt\n",
        "?? data/manual_reviews/context_merit_audit_v389/a.json\n",
    ):
        with pytest.raises(RuntimeError, match="committed-clean"):
            runtime.validate_worktree_status(status)


def test_v34c_cli_closes_nontrain_and_mutation_surfaces():
    for token in (
        "--validation=x", "--holdout=x", "--ood=x", "--benchmark=x",
        "--checkpoint=x", "--update=x", "--promotion=x",
    ):
        with pytest.raises(ValueError, match="forbidden runtime surface"):
            runtime.assert_train_only_argv([token])


def test_v34c_real_launch_requires_all_three_exact_cli_bindings():
    prereg, _panels, _layer, implementation, recipe = _foundation()
    base = {
        "v34c_dry_run": False,
        "expected_source_commit": None,
        "expected_implementation_bundle_sha256": None,
        "expected_recipe_sha256": None,
    }
    with pytest.raises(ValueError, match="implementation bundle"):
        runtime.validate_runtime_args(
            argparse.Namespace(**base), prereg, implementation, recipe
        )
    base["expected_implementation_bundle_sha256"] = implementation["bundle_sha256"]
    with pytest.raises(ValueError, match="recipe"):
        runtime.validate_runtime_args(
            argparse.Namespace(**base), prereg, implementation, recipe
        )
    base["expected_recipe_sha256"] = recipe["content_sha256_before_self_field"]
    with pytest.raises(ValueError, match="source commit"):
        runtime.validate_runtime_args(
            argparse.Namespace(**base), prereg, implementation, recipe
        )


def test_v34c_dry_run_binds_exact_hashes_without_gpu_launch(capsys):
    value = runtime.main(["--v34c-dry-run"])
    output = json.loads(capsys.readouterr().out)
    assert output == value
    assert value["schema"] == "eggroll-es-v401-replacement-fraction-gpu-dry-run-v34c"
    assert value["frozen_v34b_commit"] == runtime.V34B_COMMIT
    assert value["gpu_launched"] is False
    assert value["runtime_launched"] is False
    assert value["request_budget"]["total_generation_requests"] == 54_600
    assert value["fractions_derived_algebraically_with_zero_requests"] is True


def test_v34c_dry_run_expected_hashes_fail_closed(capsys):
    first = runtime.main(["--v34c-dry-run"])
    capsys.readouterr()
    result = runtime.main([
        "--v34c-dry-run",
        "--expected-implementation-bundle-sha256",
        first["implementation_bundle_sha256"],
        "--expected-recipe-sha256",
        first["recipe_sha256"],
    ])
    assert result["gpu_launched"] is False
    capsys.readouterr()
    with pytest.raises(ValueError, match="implementation bundle"):
        runtime.main([
            "--v34c-dry-run",
            "--expected-implementation-bundle-sha256",
            "0" * 64,
        ])


def test_v34c_phase_equality_checks_both_raw_sources():
    values = {
        source: np.ones((4, 5, 39), dtype=np.float64)
        for source in runtime.mechanics_v34b.SOURCES
    }
    phase = (values, {source: ["x"] for source in values})
    assert all(runtime._phase_equal(phase, copy.deepcopy(phase)).values())
    changed = copy.deepcopy(phase)
    changed[0]["candidate_v401"][0, 0, 0] = 2.0
    assert runtime._phase_equal(phase, changed)[
        "all_source_engine_panel_score_arrays_exact"
    ] is False


def test_v34c_capture_contract_covers_32_waves_and_all_requests(monkeypatch):
    instance = object.__new__(runtime.ReplacementFractionRuntimeMixinV34C)
    scores = {
        source: np.full((4, 5, 39), index + 1.0, dtype=np.float64)
        for index, source in enumerate(runtime.mechanics_v34b.SOURCES)
    }
    phase = (scores, {source: [source] for source in scores})
    monkeypatch.setattr(
        instance,
        "_prepared_fixed_batches_v34c",
        lambda: ({}, {"production": "a", "candidate_v401": "b"}, {}),
    )
    monkeypatch.setattr(instance, "_full_context_phase_v34c", lambda prepared: copy.deepcopy(phase))

    def wave(item, prepared, unit_scores, commitments):
        sign = runtime.mechanics_v34b.SIGNS.index(item["sign"])
        for engine, direction in enumerate(item["engine_direction_indices"]):
            unit_scores[:, :, sign, direction] = float(engine + 1)
        commitments.extend(["x"] * 40)
        return {
            "signed_wave_index": item["signed_wave_index"],
            "restore_sha256": "0" * 64,
            "source_activity_sha256": "1" * 64,
            "all_four_engines_active_for_both_sources": True,
        }

    monkeypatch.setattr(instance, "_run_signed_wave_v34c", wave)
    monkeypatch.setattr(instance, "_population_boundary_v34c", lambda: "2" * 64)
    unit_scores, audit = instance.capture_raw_arms_v34c()
    assert unit_scores.shape == (2, 5, 2, 64, 39)
    assert np.isfinite(unit_scores).all()
    assert audit["synchronized_signed_wave_count"] == 32
    assert audit["all_four_tp1_engines_both_sources_every_wave"] is True
    assert audit["perturbed_request_count"] == 49_920
    assert audit["full_context_request_count"] == 4_680
    assert audit["total_generation_request_count"] == 54_600
    assert audit["fraction_specific_request_count"] == 0


def test_v34c_fail_closed_runtime_always_closes_and_waits_for_idle(
    monkeypatch, tmp_path,
):
    prereg, panels, layer, implementation, recipe = _foundation()
    events = []
    attempts = []

    class FakeTrainer:
        def configure_replacement_fraction_v34c(self, *args):
            return {"configured": True}

        def capture_raw_arms_v34c(self):
            raise RuntimeError("synthetic capture failure")

    monkeypatch.setattr(runtime.runtime_r2, "certify_runtime_environment_r2", lambda: {
        "content_sha256_before_self_field": "1" * 64
    })
    monkeypatch.setattr(runtime, "validate_live_model", lambda prereg: {
        "content_sha256_before_self_field": "2" * 64
    })
    monkeypatch.setattr(runtime.runtime_v33a, "assert_all_four_gpus_idle_v33a", lambda: {
        "content_sha256_before_self_field": "3" * 64
    })
    monkeypatch.setattr(runtime.runtime_v33a, "wait_for_final_gpu_idle_v33a", lambda cert: (
        events.append("final_idle") or {"content_sha256_before_self_field": "4" * 64}
    ))
    monkeypatch.setattr(runtime, "make_trainer", lambda *args: FakeTrainer())
    monkeypatch.setattr(runtime.runtime_v23a.base, "close_trainer", lambda trainer: events.append("close"))
    monkeypatch.setattr(runtime, "_write_attempt", lambda value: attempts.append(copy.deepcopy(value)))
    monkeypatch.setattr(runtime, "_rewrite_attempt", lambda value: attempts.append(copy.deepcopy(value)))
    monkeypatch.setattr(runtime, "ATTEMPT_PATH", tmp_path / "attempt.json")
    monkeypatch.setattr(runtime, "RUN_DIRECTORY", tmp_path / "run")
    committed = {"content_sha256_before_self_field": "5" * 64}
    with pytest.raises(RuntimeError, match="synthetic capture failure"):
        runtime.run_exact_v34c(
            prereg, panels, layer, implementation, recipe, committed
        )
    assert events == ["close", "final_idle"]
    assert attempts[-1]["status"] == "failed"
    assert attempts[-1]["all_four_gpus_idle_after_cleanup"] is True


def test_v34c_integrity_is_set_only_after_cleanup_and_all_gates_true():
    value = runtime.runtime_integrity_after_cleanup()
    assert set(value) == runtime.mechanics_v34b.RUNTIME_INTEGRITY_KEYS
    assert all(value.values())


def test_v34c_postcleanup_recheck_rehashes_sources_and_frozen_bundle():
    foundation = _foundation()
    value = runtime.recheck_postcleanup_bindings(*foundation)
    assert value["performed_after_gpu_cleanup_before_fraction_analysis"] is True
    assert value["source_file_sha256"] == {
        "production": runtime.frame_v34b.PRODUCTION_SHA256,
        "candidate_v401": runtime.frame_v34b.CANDIDATE_SHA256,
        "candidate_manifest_v401": runtime.frame_v34b.CANDIDATE_MANIFEST_SHA256,
        "preregistration_v34b": runtime.mechanics_v34b.PREREGISTRATION_FILE_SHA256,
    }
    assert value["implementation_bundle_sha256"] == foundation[3]["bundle_sha256"]
    assert value["recipe_sha256"] == foundation[4]["content_sha256_before_self_field"]


def test_v34c_closed_trainer_surfaces_raise():
    for name in (
        "train_step", "fit", "eval_step", "apply_seed_coefficients",
        "evaluate_population_on_batch",
    ):
        with pytest.raises(RuntimeError, match="closes update checkpoint"):
            getattr(runtime.ReplacementFractionRuntimeMixinV34C, name)()


def test_v34c_compact_guard_rejects_semantic_or_raw_payload_keys():
    for key in (
        "questions", "answers", "prompt_token_ids", "unit_scores",
        "responses", "coefficients", "bootstrap_draws", "pids", "timings",
    ):
        with pytest.raises(RuntimeError, match="forbidden keys"):
            runtime.assert_compact({key: []})


def test_v34c_source_contains_real_four_tp1_launch_and_no_direct_update_surface():
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "tensor_parallel_size=1" in source
    assert "for rank in range(4)" in source
    assert '"perturb_self_weights"' in source
    assert '"restore_self_weights_exact"' in source
    assert '"audit_population_completion_v4"' in source
    assert "mechanics_v34b.build_compact_summary" in source
    assert "model_update_applied\": False" in source
