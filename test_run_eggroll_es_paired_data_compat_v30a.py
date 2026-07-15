import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

import run_eggroll_es_paired_data_compat_v30a as runtime


def _inputs():
    preregistration = runtime.load_preregistration_v30a()
    panel_bundle = runtime.mechanics_v30a.load_paired_panel_bundle_v30a()
    layer_bundle = runtime.load_layer_bundle_v30a(preregistration)
    return preregistration, panel_bundle, layer_bundle


def test_v30a_candidate_frame_and_preregistration_are_exactly_bound():
    identities = runtime._verify_bound_files_v30a()
    assert identities["preregistration_v30a"]["file_sha256"] == (
        "543c90672961ba08e30a0bf87f0278a517257372f7564c84a848350e8951afcb"
    )
    assert identities["frame_manifest_v30a"]["file_sha256"] == (
        "dbc7ed06c815b32107448fe61ef26c1dc1bcd47325bc520243e874f341240d04"
    )
    assert identities["candidate_v389"]["file_sha256"] == (
        "4b6da77e7e1ae3d1145b3f2d29c7774b6aad2b4cb520fcea9a48af93d4322388"
    )
    assert all(set(item) == {"relative_path", "file_sha256"} for item in identities.values())


def test_v30a_recipe_exactly_binds_frame_schedule_bootstrap_and_requests():
    preregistration, panel_bundle, layer_bundle = _inputs()
    recipe = runtime.recipe_v30a(
        preregistration, panel_bundle, layer_bundle, {"bundle_sha256": "a" * 64}
    )
    assert recipe["frame"]["selected_paired_units"] == 195
    assert recipe["frame"]["reserve_paired_units"] == 10
    assert recipe["frame"]["shared_document_anchors"] == 193
    assert recipe["frame"]["joint_component_cross_side_anchors"] == 2
    assert [panel["paired_units"] for panel in recipe["panels"].values()] == [39] * 5
    assert recipe["perturbation"]["basis_seed"] == 20261003
    assert recipe["perturbation"]["signed_waves"] == 16
    assert recipe["perturbation"][
        "all_four_engines_score_both_versions_every_signed_wave"
    ] is True
    assert recipe["bootstrap"]["repetitions"] == 50_000
    assert recipe["bootstrap"]["draw_plan_sha256"] == (
        "dbea000043a713114c150d07e11b813d1cbf00dcca9c418527d9a81c94b94ad5"
    )
    assert recipe["request_accounting"] == {
        "paired_units_per_panel": 39,
        "panels": 5,
        "requests_per_version_per_engine_call": 195,
        "versions_per_signed_wave": 2,
        "requests_per_engine_per_signed_wave": 390,
        "requests_all_engines_per_signed_wave": 1_560,
        "signed_waves": 16,
        "perturbed_requests_all_engines": 24_960,
        "full_context_requests_all_engines_per_phase": 1_560,
        "full_context_phase_count": 3,
        "full_context_requests_all_engines": 4_680,
        "total_generation_requests": 29_640,
    }
    assert recipe["content_sha256_before_self_field"] == runtime.canonical_sha256(
        runtime._without_self(recipe)
    )


def test_v30a_loader_uses_all_four_fixed_gpus_same_model_and_v23_worker():
    source = inspect.getsource(runtime.load_runtime_trainer_v30a)
    assert "for rank in range(4)" in source
    assert "tensor_parallel_size=1" in source
    assert "model=model" in source
    assert "worker_extension_cls=WORKER_EXTENSION_V30A" in source
    assert "enable_prefix_caching=False" in source
    assert "enforce_eager=True" in source
    assert 'moe_backend="triton"' in source
    configure = inspect.getsource(
        runtime.PairedDataCompatRuntimeMixinV30A.configure_paired_data_compat_v30a
    )
    assert 'report.get("cuda_visible_devices") != str(rank)' in configure
    assert '"seed_projection_certificate_v23a_r1"' in configure
    assert '"install_layer_plan_v4"' in configure


def test_v30a_estimate_orders_a_b_population_boundary_c_and_exact_accounting():
    source = inspect.getsource(
        runtime.PairedDataCompatRuntimeMixinV30A.estimate_paired_data_compat_v30a
    )
    positions = [source.index(token) for token in (
        "phase_a =", "phase_b =", "a_b_equal =", "for schedule_item in schedule:",
        "boundary_sha256 =", "phase_c =", "a_c_equal =",
        "build_compact_estimator_summary_v30a",
    )]
    assert positions == sorted(positions)
    assert "len(dense_commitments) != 640" in source
    assert "len(restore_hashes) != 16" in source
    assert '"perturbed_requests_all_engines": 24_960' in source
    assert '"full_context_requests_all_engines": 4_680' in source
    assert '"total_generation_requests": 29_640' in source
    assert "token_audit_sha256" in source
    run_wave = inspect.getsource(
        runtime.PairedDataCompatRuntimeMixinV30A._run_signed_wave_v30a
    )
    assert "execute_paired_resident_signed_wave_v30a" in run_wave


def test_v30a_restore_unwraps_each_tp1_result_and_rebinds_every_rank():
    class Rpc:
        def __init__(self, rank):
            self.rank = rank

        def remote(self, method, args):
            return method, self.rank, args

    class Harness(runtime.PairedDataCompatRuntimeMixinV30A):
        def __init__(self):
            self.engines = [SimpleNamespace(collective_rpc=Rpc(rank)) for rank in range(4)]
            self._v30a_references = [
                {"selected": {"sha256": f"selected-{rank}"}}
                for rank in range(4)
            ]

        @staticmethod
        def _resolve(tokens):
            results = []
            for method, rank, _args in tokens:
                if method == "restore_self_weights_exact":
                    results.append([True])
                else:
                    selected = {"sha256": f"selected-{rank}"}
                    results.append([{
                        "schema": "eggroll-es-selected-exact-reference-check-v4",
                        "passed": True,
                        "reference_generation": 1,
                        "reference": selected,
                        "current": selected,
                    }])
            return results

    harness = Harness()
    assert len(harness._restore_and_verify_v30a()) == 64
    harness._v30a_references[3]["selected"] = {"sha256": "wrong"}
    with pytest.raises(RuntimeError, match="exact selected restore changed"):
        harness._restore_and_verify_v30a()


def test_v30a_configuration_requires_full_seed_and_fresh_reference_certificates():
    configure_source = inspect.getsource(
        runtime.PairedDataCompatRuntimeMixinV30A.configure_paired_data_compat_v30a
    )
    for token in (
        'report.get("python_random_receives_full_seed") is not True',
        'report.get("torch_cuda_all_receives_full_seed") is not True',
    ):
        assert token in configure_source
    reference_source = inspect.getsource(runtime._validate_exact_references_v30a)
    for token in (
        'reference.get("reference_generation") != 1',
        'reference.get("fresh_for_population") is not True',
        'selected[rank].get("reference")',
        'reference["identity"].get("selected")',
    ):
        assert token in reference_source
    restore_source = inspect.getsource(
        runtime.PairedDataCompatRuntimeMixinV30A._restore_and_verify_v30a
    )
    for invalid_worker_v4_field_check in (
        'reference.get("rank")', 'reference.get("world_size")',
        'selected[rank].get("rank")', 'selected[rank].get("world_size")',
        'item.get("rank")', 'item.get("world_size")',
    ):
        assert invalid_worker_v4_field_check not in reference_source
        assert invalid_worker_v4_field_check not in restore_source


def test_v30a_reference_validation_matches_rankless_worker_v4_schema():
    selected_identity = {"schema": "partition", "sha256": "a" * 64}
    full_identity = {
        "schema": "eggroll-es-partitioned-weight-state-v4",
        "sha256": "b" * 64,
        "selected": selected_identity,
    }
    references = [{
        "schema": "eggroll-es-selected-exact-reference-state-v4",
        "reference_generation": 1,
        "fresh_for_population": True,
        "identity": full_identity,
    } for _ in range(4)]
    selected = [{
        "schema": "eggroll-es-selected-exact-reference-check-v4",
        "passed": True,
        "reference_generation": 1,
        "reference": selected_identity,
        "current": selected_identity,
    } for _ in range(4)]
    assert runtime._validate_exact_references_v30a(references, selected) == (
        runtime.canonical_sha256(selected_identity)
    )
    assert all("rank" not in item and "world_size" not in item for item in references)
    assert all("rank" not in item and "world_size" not in item for item in selected)
    selected[3] = {**selected[3], "current": {"sha256": "c" * 64}}
    with pytest.raises(RuntimeError, match="certificate changed"):
        runtime._validate_exact_references_v30a(references, selected)


def test_v30a_full_context_phase_equality_is_exact_and_content_free():
    scores = {
        version: np.arange(780, dtype=np.float64).reshape(4, 5, 39)
        for version in runtime.mechanics_v30a.VERSIONS_V30A
    }
    phase = (scores, {version: ["a"] * 20 for version in scores})
    assert all(runtime._phase_equal(phase, phase).values())
    identity = runtime._phase_identity(phase)
    assert set(identity) == {"score_arrays_sha256", "dense_commitments_sha256"}
    changed = (
        {key: value.copy() for key, value in scores.items()},
        phase[1],
    )
    changed[0]["candidate_v389"][0, 0, 0] += 1.0
    assert runtime._phase_equal(phase, changed)[
        "all_version_engine_panel_score_arrays_exact"
    ] is False


def test_v30a_paths_are_disjoint_and_fresh_or_completed_exactly():
    attempt = runtime.OUTPUT_DIRECTORY_V30A / runtime.ATTEMPT_NAME_V30A
    run_dir = runtime.OUTPUT_DIRECTORY_V30A / runtime.EXPERIMENT_NAME_V30A
    report = run_dir / runtime.REPORT_NAME_V30A
    assert len({attempt, run_dir, report}) == 3
    if not attempt.exists() and not run_dir.exists() and not report.exists():
        return
    assert attempt.exists() and run_dir.is_dir() and report.is_file()
    ledger = json.loads(attempt.read_text())
    assert ledger["status"] == "complete"
    assert ledger["phase"] == "after_cleanup_and_compact_report"
    assert ledger["report_binding"] == {
        "path": str(report),
        "file_sha256": runtime.file_sha256(report),
        "content_sha256": json.loads(report.read_text())[
            "content_sha256_before_self_field"
        ],
    }


def test_v30a_real_launch_hash_and_environment_are_fail_closed(monkeypatch):
    preregistration, panel_bundle, layer_bundle = _inputs()
    implementation = {"bundle_sha256": "a" * 64}
    recipe = runtime.recipe_v30a(
        preregistration, panel_bundle, layer_bundle, implementation
    )
    with pytest.raises(ValueError, match="requires expected implementation"):
        runtime.validate_runtime_v30a(
            SimpleNamespace(
                v30a_dry_run=False,
                expected_implementation_bundle_sha256=None,
                expected_recipe_sha256=None,
            ),
            preregistration,
            implementation,
            recipe,
        )
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with pytest.raises(ValueError, match="batch-invariant backend swap"):
        runtime.validate_runtime_v30a(
            SimpleNamespace(
                v30a_dry_run=True,
                expected_implementation_bundle_sha256=None,
                expected_recipe_sha256=None,
            ),
            preregistration,
            implementation,
            recipe,
        )


def test_v30a_commit_certificate_is_real_launch_only(monkeypatch):
    preregistration, panel_bundle, layer_bundle = _inputs()
    implementation = {"bundle_sha256": "a" * 64}
    recipe = runtime.recipe_v30a(
        preregistration, panel_bundle, layer_bundle, implementation
    )
    calls = []
    monkeypatch.setattr(
        runtime,
        "_certify_real_launch_committed_source_v30a",
        lambda: calls.append("certified"),
    )

    runtime.validate_runtime_v30a(
        SimpleNamespace(
            v30a_dry_run=True,
            expected_implementation_bundle_sha256=None,
            expected_recipe_sha256=None,
        ),
        preregistration,
        implementation,
        recipe,
    )
    assert calls == []

    runtime.validate_runtime_v30a(
        SimpleNamespace(
            v30a_dry_run=False,
            expected_implementation_bundle_sha256=implementation["bundle_sha256"],
            expected_recipe_sha256=recipe["content_sha256_before_self_field"],
        ),
        preregistration,
        implementation,
        recipe,
    )
    assert calls == ["certified"]


def test_v30a_update_eval_checkpoint_promotion_and_union_surfaces_are_closed():
    preregistration, panel_bundle, layer_bundle = _inputs()
    recipe = runtime.recipe_v30a(
        preregistration, panel_bundle, layer_bundle, {"bundle_sha256": "a" * 64}
    )
    assert all(recipe["authority"][key] is False for key in (
        "model_update_allowed", "checkpoint_write_allowed", "evaluation_allowed",
        "dataset_promotion_allowed",
    ))
    for name in (
        "train_step", "apply_seed_coefficients", "eval_step", "fit",
        "configure_anchor", "configure_train_panels_v13",
    ):
        with pytest.raises(RuntimeError, match="closes update checkpoint evaluation"):
            getattr(runtime.PairedDataCompatRuntimeMixinV30A, name)()
    factory = inspect.getsource(runtime._make_trainer_v30a)
    assert "checkpoint=None" in factory
    assert "alpha=0.0" in factory
    assert "eval_dataloader_dict={}" in factory
    assert "save_best_models=False" in factory


def test_v30a_attempt_claim_provenance_environment_and_model_precede_trainer():
    source = inspect.getsource(runtime.run_exact_v30a)
    assert source.index("certify_runtime_environment_r2()") < source.index(
        "_exclusive_write_json_v23a(attempt_path, attempt)"
    )
    assert source.index("_source_provenance_v23a") < source.index(
        "_exclusive_write_json_v23a(attempt_path, attempt)"
    )
    assert source.index("validate_live_model_v30a") < source.index(
        "_exclusive_write_json_v23a(attempt_path, attempt)"
    )
    assert source.index("_exclusive_write_json_v23a(attempt_path, attempt)") < source.index(
        "_make_trainer_v30a"
    )
    assert source.index("if run_dir.exists()") < source.index("_make_trainer_v30a")


def test_v30a_forbidden_argv_and_compact_persistence_fail_closed():
    for token in ("--validation-file=x", "--ood", "--heldout", "--eval"):
        with pytest.raises(ValueError, match="forbidden runtime surface"):
            runtime._assert_train_only_argv([token])
    with pytest.raises(RuntimeError, match="forbidden keys"):
        runtime._assert_compact({"unit_scores": []})


def test_v30a_dry_run_constructs_no_trainer_or_gpu(monkeypatch, capsys):
    preregistration, panel_bundle, layer_bundle = _inputs()
    monkeypatch.setattr(runtime, "load_preregistration_v30a", lambda: preregistration)
    monkeypatch.setattr(
        runtime.mechanics_v30a,
        "load_paired_panel_bundle_v30a",
        lambda: panel_bundle,
    )
    monkeypatch.setattr(runtime, "load_layer_bundle_v30a", lambda _value: layer_bundle)
    monkeypatch.setattr(
        runtime,
        "implementation_identity_v30a",
        lambda: {"bundle_sha256": "a" * 64, "files": {}},
    )
    monkeypatch.setattr(
        runtime,
        "_make_trainer_v30a",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("trainer created")
        ),
    )
    value = runtime.main(["--v30a-dry-run"])
    assert json.loads(capsys.readouterr().out) == value
    assert value["gpu_launched"] is False
    assert value[
        "pass_authority_limited_to_separate_full_v389_train_only_hpo_preregistration"
    ] is True
    assert value[
        "dataset_promotion_model_update_checkpoint_and_evaluation_authorized"
    ] is False
    assert value["row_response_score_or_bootstrap_content_persisted"] is False
