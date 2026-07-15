import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

import run_eggroll_es_paired_data_compat_v33a as runtime


def _inputs():
    preregistration = runtime.load_preregistration_v33a()
    panel_bundle = runtime.mechanics_v33a.load_paired_panel_bundle_v33a()
    layer_bundle = runtime.load_layer_bundle_v33a(preregistration)
    return preregistration, panel_bundle, layer_bundle


def test_v33a_candidate_frame_and_preregistration_are_exactly_bound():
    identities = runtime._verify_bound_files_v33a()
    assert identities["preregistration_v33a"]["file_sha256"] == (
        "c83e11376922ac273b5b6496ef60126a2cdc6ae044f6a9ab05d9481dc539bcda"
    )
    assert identities["frame_manifest_v33a"]["file_sha256"] == (
        "8c8ab38e03949e01982701b58e0f273c23305e67b114f7c607e7e4d83d10666a"
    )
    assert identities["candidate_v364"]["file_sha256"] == (
        "874b77dd8ef988bb24d4b13999ddebc2068b053eab208def9bb1e23e7138c36a"
    )
    assert identities["v25a_promising_unconfirmed_aggregate"]["file_sha256"] == (
        "311966e1b8b03d2354289fde9ef010f47709862d11fdec9e236d7f4de0a9a65c"
    )
    assert all(set(item) == {"relative_path", "file_sha256"} for item in identities.values())


def test_v33a_recipe_exactly_binds_frame_schedule_bootstrap_and_requests():
    preregistration, panel_bundle, layer_bundle = _inputs()
    recipe = runtime.recipe_v33a(
        preregistration, panel_bundle, layer_bundle, {"bundle_sha256": "a" * 64}
    )
    assert recipe["frame"]["selected_paired_units"] == 195
    assert recipe["frame"]["reserve_paired_units"] == 10
    assert recipe["frame"]["shared_document_anchors"] == 193
    assert recipe["frame"]["joint_component_cross_side_anchors"] == 2
    assert [panel["paired_units"] for panel in recipe["panels"].values()] == [39] * 5
    assert recipe["perturbation"]["basis_seed"] == 20261008
    assert recipe["perturbation"]["population_size"] == 64
    assert recipe["perturbation"]["synchronized_four_engine_signed_waves"] == 32
    assert recipe["perturbation"]["engine_signed_direction_evaluations"] == 128
    assert recipe["perturbation"][
        "all_four_engines_score_both_versions_every_signed_wave"
    ] is True
    assert recipe["bootstrap"]["repetitions"] == 50_000
    assert recipe["bootstrap"]["draw_plan_sha256"] == (
        "ef44acfe80d9afab5e17621eb62acc09572a6c3486f1f0a61dd6848ab6398b37"
    )
    assert recipe["request_accounting"] == {
        "paired_units_per_panel": 39,
        "panels": 5,
        "requests_per_version_per_engine_call": 195,
        "versions_per_signed_wave": 2,
        "requests_per_engine_per_signed_wave": 390,
        "requests_all_engines_per_signed_wave": 1_560,
        "synchronized_four_engine_signed_waves": 32,
        "engine_signed_direction_evaluations": 128,
        "perturbed_requests_all_engines": 49_920,
        "full_context_requests_all_engines_per_phase": 1_560,
        "full_context_phase_count": 3,
        "full_context_requests_all_engines": 4_680,
        "total_generation_requests": 54_600,
    }
    assert recipe["content_sha256_before_self_field"] == runtime.canonical_sha256(
        runtime._without_self(recipe)
    )


def test_v33a_loader_uses_all_four_fixed_gpus_same_model_and_v23_worker():
    source = inspect.getsource(runtime.load_runtime_trainer_v33a)
    assert "for rank in range(4)" in source
    assert "tensor_parallel_size=1" in source
    assert "model=model" in source
    assert "worker_extension_cls=WORKER_EXTENSION_V33A" in source
    assert "enable_prefix_caching=False" in source
    assert "enforce_eager=True" in source
    assert 'moe_backend="triton"' in source
    configure = inspect.getsource(
        runtime.PairedDataCompatRuntimeMixinV33A.configure_paired_data_compat_v33a
    )
    assert 'report.get("cuda_visible_devices") != str(rank)' in configure
    assert '"seed_projection_certificate_v33a"' in configure
    assert '"install_layer_plan_v4"' in configure


def test_v33a_all_engine_generation_submits_four_before_resolve():
    events = []

    class Generate:
        def __init__(self, rank):
            self.rank = rank

        def remote(self, items, _sampling, use_tqdm):
            events.append(("submit", self.rank, len(items), use_tqdm))
            return self.rank

    class Harness(runtime.PairedDataCompatRuntimeMixinV33A):
        def __init__(self):
            self.engines = [
                SimpleNamespace(generate=Generate(rank)) for rank in range(4)
            ]
            self._v33a_fixed_request_identity = {
                "production": runtime.canonical_sha256([[1]] * 195)
            }

        @staticmethod
        def _dense_sampling_params_v4(_seed):
            return object()

        @staticmethod
        def _resolve(tokens):
            events.append(("resolve", tuple(tokens)))
            return [[object()] * 195 for _ in tokens]

    prepared = {
        "production": {
            "prompt_items": [{"prompt_token_ids": [1]} for _ in range(195)]
        }
    }
    batches = Harness()._generate_version_all_engines_v33a(
        "production", prepared
    )
    assert len(batches) == 4
    assert events[:4] == [
        ("submit", rank, 195, False) for rank in range(4)
    ]
    assert events[4] == ("resolve", (0, 1, 2, 3))


def test_v33a_fresh_schedule_seeds_reach_each_worker_without_bypass():
    events = []

    class Rpc:
        def __init__(self, rank):
            self.rank = rank

        def remote(self, method, args):
            events.append((self.rank, method, args))
            return [None]

    class Harness(runtime.PairedDataCompatRuntimeMixinV33A):
        def __init__(self):
            self.engines = [
                SimpleNamespace(collective_rpc=Rpc(rank)) for rank in range(4)
            ]

        @staticmethod
        def _resolve(tokens):
            return tokens

    schedule_item = runtime.mechanics_v33a.resident_signed_wave_schedule_v33a()[7]
    Harness()._perturb_signed_wave_v33a(
        schedule_item["engine_direction_seeds"], schedule_item["negate"]
    )
    assert events == [
        (
            rank,
            "perturb_self_weights",
            (
                schedule_item["engine_direction_seeds"][rank],
                0.0003,
                schedule_item["negate"],
            ),
        )
        for rank in range(4)
    ]


def test_v33a_estimate_orders_a_b_population_boundary_c_and_exact_accounting():
    source = inspect.getsource(
        runtime.PairedDataCompatRuntimeMixinV33A.estimate_paired_data_compat_v33a
    )
    positions = [source.index(token) for token in (
        "phase_a =", "phase_b =", "a_b_equal =", "for schedule_item in schedule:",
        "boundary_sha256 =", "phase_c =", "a_c_equal =",
        "build_compact_estimator_summary_v33a",
    )]
    assert positions == sorted(positions)
    assert "len(dense_commitments) != 1_280" in source
    assert "len(restore_hashes) != 32" in source
    assert '"perturbed_requests_all_engines": 49_920' in source
    assert '"full_context_requests_all_engines": 4_680' in source
    assert '"total_generation_requests": 54_600' in source
    assert "token_audit_sha256" in source
    run_wave = inspect.getsource(
        runtime.PairedDataCompatRuntimeMixinV33A._run_signed_wave_v33a
    )
    assert "execute_paired_resident_signed_wave_v33a" in run_wave


def test_v33a_restore_unwraps_each_tp1_result_and_rebinds_every_rank():
    class Rpc:
        def __init__(self, rank):
            self.rank = rank

        def remote(self, method, args):
            return method, self.rank, args

    class Harness(runtime.PairedDataCompatRuntimeMixinV33A):
        def __init__(self):
            self.engines = [SimpleNamespace(collective_rpc=Rpc(rank)) for rank in range(4)]
            self._v33a_references = [
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
    assert len(harness._restore_and_verify_v33a()) == 64
    harness._v33a_references[3]["selected"] = {"sha256": "wrong"}
    with pytest.raises(RuntimeError, match="exact selected restore changed"):
        harness._restore_and_verify_v33a()


def test_v33a_configuration_requires_full_seed_and_fresh_reference_certificates():
    configure_source = inspect.getsource(
        runtime.PairedDataCompatRuntimeMixinV33A.configure_paired_data_compat_v33a
    )
    for token in (
        'report.get("python_random_receives_full_seed") is not True',
        'report.get("torch_cuda_all_receives_full_seed") is not True',
    ):
        assert token in configure_source
    reference_source = inspect.getsource(runtime._validate_exact_references_v33a)
    for token in (
        'reference.get("reference_generation") != 1',
        'reference.get("fresh_for_population") is not True',
        'selected[rank].get("reference")',
        'reference["identity"].get("selected")',
    ):
        assert token in reference_source
    restore_source = inspect.getsource(
        runtime.PairedDataCompatRuntimeMixinV33A._restore_and_verify_v33a
    )
    for invalid_worker_v4_field_check in (
        'reference.get("rank")', 'reference.get("world_size")',
        'selected[rank].get("rank")', 'selected[rank].get("world_size")',
        'item.get("rank")', 'item.get("world_size")',
    ):
        assert invalid_worker_v4_field_check not in reference_source
        assert invalid_worker_v4_field_check not in restore_source


def test_v33a_reference_validation_matches_rankless_worker_v4_schema():
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
    assert runtime._validate_exact_references_v33a(references, selected) == (
        runtime.canonical_sha256(selected_identity)
    )
    assert all("rank" not in item and "world_size" not in item for item in references)
    assert all("rank" not in item and "world_size" not in item for item in selected)
    selected[3] = {**selected[3], "current": {"sha256": "c" * 64}}
    with pytest.raises(RuntimeError, match="certificate changed"):
        runtime._validate_exact_references_v33a(references, selected)


def test_v33a_full_context_phase_equality_is_exact_and_content_free():
    scores = {
        version: np.arange(780, dtype=np.float64).reshape(4, 5, 39)
        for version in runtime.mechanics_v33a.VERSIONS_V33A
    }
    phase = (scores, {version: ["a"] * 20 for version in scores})
    assert all(runtime._phase_equal(phase, phase).values())
    identity = runtime._phase_identity(phase)
    assert set(identity) == {"score_arrays_sha256", "dense_commitments_sha256"}
    changed = (
        {key: value.copy() for key, value in scores.items()},
        phase[1],
    )
    changed[0]["candidate_v364"][0, 0, 0] += 1.0
    assert runtime._phase_equal(phase, changed)[
        "all_version_engine_panel_score_arrays_exact"
    ] is False


def test_v33a_paths_are_disjoint_and_fresh_or_completed_exactly():
    attempt = runtime.OUTPUT_DIRECTORY_V33A / runtime.ATTEMPT_NAME_V33A
    run_dir = runtime.OUTPUT_DIRECTORY_V33A / runtime.EXPERIMENT_NAME_V33A
    report = run_dir / runtime.REPORT_NAME_V33A
    assert len({attempt, run_dir, report}) == 3
    if not attempt.exists() and not run_dir.exists() and not report.exists():
        return
    assert attempt.exists() and run_dir.is_dir() and report.is_file()
    ledger = json.loads(attempt.read_text())
    assert ledger["status"] == "complete"
    assert ledger["phase"] == "after_compact_report_and_final_gpu_cleanup"
    assert ledger["all_four_gpus_idle_after_cleanup"] is True
    assert len(ledger["final_idle_certificate_sha256"]) == 64
    assert ledger["report_binding"] == {
        "path": str(report),
        "file_sha256": runtime.file_sha256(report),
        "content_sha256": json.loads(report.read_text())[
            "content_sha256_before_self_field"
        ],
    }


def test_v33a_real_launch_hash_and_environment_are_fail_closed(monkeypatch):
    preregistration, panel_bundle, layer_bundle = _inputs()
    implementation = {"bundle_sha256": "a" * 64}
    recipe = runtime.recipe_v33a(
        preregistration, panel_bundle, layer_bundle, implementation
    )
    with pytest.raises(ValueError, match="requires expected implementation"):
        runtime.validate_runtime_v33a(
            SimpleNamespace(
                v33a_dry_run=False,
                expected_implementation_bundle_sha256=None,
                expected_recipe_sha256=None,
            ),
            preregistration,
            implementation,
            recipe,
        )
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with pytest.raises(ValueError, match="batch-invariant backend swap"):
        runtime.validate_runtime_v33a(
            SimpleNamespace(
                v33a_dry_run=True,
                expected_implementation_bundle_sha256=None,
                expected_recipe_sha256=None,
            ),
            preregistration,
            implementation,
            recipe,
        )


def test_v33a_commit_certificate_is_real_launch_only(monkeypatch):
    preregistration, panel_bundle, layer_bundle = _inputs()
    implementation = {"bundle_sha256": "a" * 64}
    recipe = runtime.recipe_v33a(
        preregistration, panel_bundle, layer_bundle, implementation
    )
    calls = []
    certificate = {"schema": "committed-clean"}
    monkeypatch.setattr(
        runtime,
        "_certify_real_launch_committed_source_v33a",
        lambda: calls.append("certified") or certificate,
    )

    dry_actual = runtime.validate_runtime_v33a(
        SimpleNamespace(
            v33a_dry_run=True,
            expected_implementation_bundle_sha256=None,
            expected_recipe_sha256=None,
        ),
        preregistration,
        implementation,
        recipe,
    )
    assert calls == []
    assert dry_actual is None

    actual = runtime.validate_runtime_v33a(
        SimpleNamespace(
            v33a_dry_run=False,
            expected_implementation_bundle_sha256=implementation["bundle_sha256"],
            expected_recipe_sha256=recipe["content_sha256_before_self_field"],
        ),
        preregistration,
        implementation,
        recipe,
    )
    assert calls == ["certified"]
    assert actual is certificate


def test_v33a_worktree_allowlist_is_explicit_and_fail_closed():
    allowed = "\n".join((
        "?? data/manual_reviews/context_merit_audit_v393/part.jsonl",
        "?? experiments/dataset_probes/probe.json",
        "?? experiments/gpu_utilization_v31_base_qwen36_train_reward_probe.jsonl",
    ))
    certificate = runtime._validate_worktree_status_v33a(allowed)
    assert certificate["all_tracked_files_clean"] is True
    assert certificate[
        "only_explicitly_allowlisted_untracked_paths_present"
    ] is True
    assert certificate["allowed_untracked_entry_count"] == 3
    assert len(certificate["allowlist_contract_sha256"]) == 64

    for rejected in (
        "?? data/manual_reviews/context_merit_audit_v364/late.jsonl",
        "?? unexpected-user-file.txt",
        " M run_eggroll_es_paired_data_compat_v33a.py",
        "M  test_run_eggroll_es_paired_data_compat_v33a.py",
    ):
        with pytest.raises(RuntimeError, match="explicit untracked allowlist"):
            runtime._validate_worktree_status_v33a(rejected)


def test_v33a_update_eval_checkpoint_promotion_and_union_surfaces_are_closed():
    preregistration, panel_bundle, layer_bundle = _inputs()
    recipe = runtime.recipe_v33a(
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
            getattr(runtime.PairedDataCompatRuntimeMixinV33A, name)()
    factory = inspect.getsource(runtime._make_trainer_v33a)
    assert "checkpoint=None" in factory
    assert "alpha=0.0" in factory
    assert "population_size=64" in factory
    assert "eval_dataloader_dict={}" in factory
    assert "save_best_models=False" in factory


def test_v33a_attempt_claim_provenance_environment_and_model_precede_trainer():
    source = inspect.getsource(runtime.run_exact_v33a)
    assert source.index("certify_runtime_environment_r2()") < source.index(
        "_exclusive_write_json_v23a(attempt_path, attempt)"
    )
    assert source.index("_source_provenance_v23a") < source.index(
        "_exclusive_write_json_v23a(attempt_path, attempt)"
    )
    assert source.index("validate_live_model_v33a") < source.index(
        "_exclusive_write_json_v23a(attempt_path, attempt)"
    )
    assert source.index("assert_all_four_gpus_idle_v33a()") > source.index(
        "validate_live_model_v33a"
    )
    assert source.index("assert_all_four_gpus_idle_v33a()") < source.index(
        "_exclusive_write_json_v23a(attempt_path, attempt)"
    )
    assert source.index("_exclusive_write_json_v23a(attempt_path, attempt)") < source.index(
        "_make_trainer_v33a"
    )
    assert source.index("if run_dir.exists()") < source.index("_make_trainer_v33a")
    assert source.index("base.close_trainer(trainer)") < source.rindex(
        "wait_for_final_gpu_idle_v33a(prelaunch_idle)"
    )
    assert source.rindex("wait_for_final_gpu_idle_v33a(prelaunch_idle)") < source.index(
        '"schema": "eggroll-es-paired-data-compat-report-v33a"'
    )


def _gpu_observation(running_counts=(0, 0, 0, 0), identity_suffix=""):
    return {
        "gpus": [
            {
                "physical_gpu_id": index,
                "name": runtime.EXPECTED_GPU_NAME_V33A,
                "driver_version": runtime.EXPECTED_DRIVER_VERSION_V33A,
                "nvml_uuid": runtime.EXPECTED_GPU_IDENTITIES_V33A[index][
                    "nvml_uuid"
                ] + identity_suffix,
                "pci_bus_id": runtime.EXPECTED_GPU_IDENTITIES_V33A[index][
                    "pci_bus_id"
                ],
                "total_bytes": runtime.EXPECTED_GPU_IDENTITIES_V33A[index][
                    "total_bytes"
                ],
                "running_process_count": running_counts[index],
            }
            for index in range(4)
        ],
        "all_four_idle": all(value == 0 for value in running_counts),
    }


def test_v33a_prelaunch_and_bounded_final_idle_are_fail_closed(monkeypatch):
    monkeypatch.setattr(
        runtime, "_observe_all_four_gpus_v33a", lambda: _gpu_observation()
    )
    prelaunch = runtime.assert_all_four_gpus_idle_v33a()
    assert prelaunch["all_four_idle"] is True

    observations = iter((
        _gpu_observation((1, 1, 1, 1)),
        _gpu_observation(),
    ))
    monkeypatch.setattr(
        runtime, "_observe_all_four_gpus_v33a", lambda: next(observations)
    )
    monkeypatch.setattr(runtime.time, "sleep", lambda _seconds: None)
    final = runtime.wait_for_final_gpu_idle_v33a(prelaunch)
    assert final["all_four_idle"] is True
    assert final["poll_count"] == 2
    assert final["bounded_async_cleanup_wait"] is True

    monkeypatch.setattr(
        runtime,
        "_observe_all_four_gpus_v33a",
        lambda: _gpu_observation((1, 0, 0, 0)),
    )
    with pytest.raises(RuntimeError, match="idle before claim"):
        runtime.assert_all_four_gpus_idle_v33a()

    monkeypatch.setattr(
        runtime,
        "_observe_all_four_gpus_v33a",
        lambda: _gpu_observation(identity_suffix="-changed"),
    )
    with pytest.raises(RuntimeError, match="physical GPU identity changed"):
        runtime.wait_for_final_gpu_idle_v33a(prelaunch)


def test_v33a_durable_success_and_failure_both_bind_final_idle(
    monkeypatch, tmp_path,
):
    prelaunch = runtime._seal({
        "schema": "eggroll-es-v33a-prelaunch-idle-certificate",
        **_gpu_observation(),
    })
    final_idle = runtime._seal({
        "schema": "eggroll-es-v33a-final-idle-certificate",
        **_gpu_observation(),
        "poll_count": 1,
        "elapsed_milliseconds": 0,
        "bounded_async_cleanup_wait": True,
    })
    committed_clean = runtime._seal({
        "schema": "eggroll-es-committed-clean-source-certificate-v33a",
        "git_head": "a" * 40,
        "all_tracked_files_clean": True,
        "only_explicitly_allowlisted_untracked_paths_present": True,
        "allowed_untracked_entry_count": 0,
        "allowed_untracked_entries_sha256": runtime.canonical_sha256([]),
        "allowlist_contract_sha256": runtime.canonical_sha256(
            runtime.WORKTREE_ALLOWLIST_CONTRACT_V33A
        ),
    })
    implementation = {"bundle_sha256": "b" * 64, "files": {}}
    recipe = runtime._seal({"schema": "test-recipe"})
    gate = {"schema": "test-gate", "pass": False}
    closed = []

    monkeypatch.setattr(runtime, "OUTPUT_DIRECTORY_V33A", tmp_path)
    monkeypatch.setattr(
        runtime.runtime_r2,
        "certify_runtime_environment_r2",
        lambda: {"schema": "environment"},
    )
    monkeypatch.setattr(
        runtime.runtime_v23a,
        "_source_provenance_v23a",
        lambda _implementation: {"git_head": "a" * 40},
    )
    monkeypatch.setattr(
        runtime, "validate_live_model_v33a", lambda _prereg: {"schema": "model"}
    )
    monkeypatch.setattr(
        runtime, "assert_all_four_gpus_idle_v33a", lambda: prelaunch
    )
    monkeypatch.setattr(
        runtime,
        "wait_for_final_gpu_idle_v33a",
        lambda _prelaunch: final_idle,
    )
    monkeypatch.setattr(runtime.runtime_v23a.base, "set_seed", lambda _seed: None)
    monkeypatch.setattr(
        runtime.runtime_v23a.base,
        "close_trainer",
        lambda trainer: closed.append(trainer),
    )
    monkeypatch.setattr(
        runtime.mechanics_v33a, "evaluate_candidate_v33a", lambda _summary: gate
    )

    class Trainer:
        def __init__(self, failure=None):
            self.failure = failure

        @staticmethod
        def configure_paired_data_compat_v33a(*_args):
            return {"schema": "configuration"}

        def estimate_paired_data_compat_v33a(self):
            if self.failure is not None:
                raise self.failure
            return {"schema": "summary"}, gate, {"schema": "audit"}

    def install_trainer(failure=None):
        run_dir = tmp_path / runtime.EXPERIMENT_NAME_V33A
        trainer = Trainer(failure)

        def make_trainer(*_args):
            run_dir.mkdir()
            return trainer

        monkeypatch.setattr(
            runtime, "_make_trainer_v33a", make_trainer
        )
        return trainer

    successful_trainer = install_trainer()
    report = runtime.run_exact_v33a(
        {}, {}, {}, implementation, recipe, committed_clean
    )
    attempt_path = tmp_path / runtime.ATTEMPT_NAME_V33A
    attempt = json.loads(attempt_path.read_text())
    assert report["all_four_gpus_idle_after_cleanup"] is True
    assert report["final_idle_certificate_sha256"] == final_idle[
        "content_sha256_before_self_field"
    ]
    assert attempt["status"] == "complete"
    assert attempt["phase"] == "after_compact_report_and_final_gpu_cleanup"
    assert attempt["final_idle_certificate_sha256"] == final_idle[
        "content_sha256_before_self_field"
    ]
    assert closed == [successful_trainer]

    attempt_path.unlink()
    report_path = (
        tmp_path
        / runtime.EXPERIMENT_NAME_V33A
        / runtime.REPORT_NAME_V33A
    )
    report_path.unlink()
    report_path.parent.rmdir()
    failed_trainer = install_trainer(RuntimeError("synthetic failure"))
    with pytest.raises(RuntimeError, match="synthetic failure"):
        runtime.run_exact_v33a(
            {}, {}, {}, implementation, recipe, committed_clean
        )
    failed_attempt = json.loads(attempt_path.read_text())
    assert failed_attempt["status"] == "failed"
    assert failed_attempt["phase"] == "after_cleanup_or_final_idle_failure"
    assert failed_attempt["all_four_gpus_idle_after_cleanup"] is True
    assert failed_attempt["final_idle_certificate_sha256"] == final_idle[
        "content_sha256_before_self_field"
    ]
    assert failed_attempt["report_exists_after_attempt"] is False
    assert closed == [successful_trainer, failed_trainer]

    attempt_path.unlink()
    report_path.parent.rmdir()
    finalization_trainer = install_trainer()
    exclusive_write = runtime.runtime_v23a._exclusive_write_json_v23a

    def fail_report_write(path, value):
        if path.name == runtime.REPORT_NAME_V33A:
            raise OSError("synthetic report failure")
        return exclusive_write(path, value)

    monkeypatch.setattr(
        runtime.runtime_v23a,
        "_exclusive_write_json_v23a",
        fail_report_write,
    )
    with pytest.raises(OSError, match="synthetic report failure"):
        runtime.run_exact_v33a(
            {}, {}, {}, implementation, recipe, committed_clean
        )
    finalization_attempt = json.loads(attempt_path.read_text())
    assert finalization_attempt["status"] == "failed"
    assert finalization_attempt["phase"] == (
        "after_final_idle_during_compact_report_finalization"
    )
    assert finalization_attempt["all_four_gpus_idle_after_cleanup"] is True
    assert finalization_attempt["final_idle_certificate_sha256"] == final_idle[
        "content_sha256_before_self_field"
    ]
    assert finalization_attempt["report_exists_after_attempt"] is False
    assert closed == [
        successful_trainer, failed_trainer, finalization_trainer,
    ]


def test_v33a_forbidden_argv_and_compact_persistence_fail_closed():
    for token in ("--validation-file=x", "--ood", "--heldout", "--eval"):
        with pytest.raises(ValueError, match="forbidden runtime surface"):
            runtime._assert_train_only_argv([token])
    with pytest.raises(RuntimeError, match="forbidden keys"):
        runtime._assert_compact({"unit_scores": []})
    with pytest.raises(RuntimeError, match="forbidden keys"):
        runtime._assert_compact({"unit_ids": []})


def test_v33a_dry_run_constructs_no_trainer_or_gpu(monkeypatch, capsys):
    preregistration, panel_bundle, layer_bundle = _inputs()
    monkeypatch.setattr(runtime, "load_preregistration_v33a", lambda: preregistration)
    monkeypatch.setattr(
        runtime.mechanics_v33a,
        "load_paired_panel_bundle_v33a",
        lambda: panel_bundle,
    )
    monkeypatch.setattr(runtime, "load_layer_bundle_v33a", lambda _value: layer_bundle)
    monkeypatch.setattr(
        runtime,
        "implementation_identity_v33a",
        lambda: {"bundle_sha256": "a" * 64, "files": {}},
    )
    monkeypatch.setattr(
        runtime,
        "_make_trainer_v33a",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("trainer created")
        ),
    )
    value = runtime.main(["--v33a-dry-run"])
    assert json.loads(capsys.readouterr().out) == value
    assert value["gpu_launched"] is False
    assert value["train_only_runtime_launched"] is False
    assert value[
        "validation_heldout_ood_benchmark_or_evaluation_opened"
    ] is False
    assert value[
        "pass_authority_limited_to_separate_fresh_basis_v364_confirmation_preregistration"
    ] is True
    assert value[
        "dataset_promotion_model_update_checkpoint_and_evaluation_authorized"
    ] is False
    assert value["row_response_score_or_bootstrap_content_persisted"] is False
