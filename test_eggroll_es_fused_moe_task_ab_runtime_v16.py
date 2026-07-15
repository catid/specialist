#!/usr/bin/env python3
"""No-GPU tests for the V16 fused-MoE task A/B runtime."""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest

import eggroll_es_fused_moe_task_ab_preregistration_v16 as prereg_v16
import run_eggroll_es_fused_moe_task_ab_v16 as driver_v16
import train_eggroll_es_fused_moe_task_ab_v16 as trainer_v16
import train_eggroll_es_specialist_anchor_v11 as anchor_v11
import train_eggroll_es_specialist_anchor_v13 as anchor_v13


PLAN_SHA = driver_v16.driver_v13.driver_v10.MIDDLE_LATE_PLAN_SHA256_V10


def load_bundle():
    spec = anchor_v11.FROZEN_STABILITY_PLANS_V11[PLAN_SHA]
    return anchor_v13.load_frozen_layer_plan_v13(
        spec["path"], expected_file_sha256=spec["file_sha256"],
        expected_plan_sha256=PLAN_SHA,
        expected_model_config_sha256=anchor_v11.MODEL_CONFIG_SHA256_V11,
    )


def cli(extra=None):
    bundle = load_bundle()
    return [
        "--layer-plan-json", bundle["path"],
        "--expected-layer-plan-file-sha256", bundle["file_sha256"],
        "--expected-layer-plan-sha256", bundle["plan_sha256"],
        "--expected-model-config-sha256", bundle["model_config_sha256"],
        "--v16-dry-run",
        *(extra or []),
    ]


def synthetic_arm(seconds=1.0, marker="a"):
    return {
        "diagnostic_content_sha256": marker * 64,
        "dense_result_manifest_sha256": "b" * 64,
        "task_output_sha256": "c" * 64,
        "compact_estimator": {
            "stability": {"synthetic": True},
            "robust_aggregate": {"coefficient_sha256": "d" * 64},
        },
        "generation_timing": {
            "wave_seconds": [float(seconds)] * 16,
            "total_seconds": float(seconds) * 16,
        },
        "all_integrity_audits_passed": True,
        "persisted_raw_content": False,
    }


def synthetic_diagnostic():
    responses = {}
    base_values = [float(index - 15) for index in range(32)]
    for panel_index, name in enumerate(anchor_v13.PANEL_NAMES_V13):
        central = [
            (1.0 + 0.05 * panel_index) * value for value in base_values
        ]
        weighted = {
            "plus": [10.0 + value for value in central],
            "minus": [10.0 - value for value in central],
        }
        unweighted = {
            "plus": [20.0 + 0.5 * value for value in central],
            "minus": [20.0 - 0.5 * value for value in central],
        }
        responses[name] = {
            "weighted_sign_scores": weighted,
            "unweighted_sign_scores": unweighted,
            "stratum_sign_scores": {
                stratum: {
                    sign: list(unweighted[sign])
                    for sign in anchor_v13.SIGNS_V13
                }
                for stratum in anchor_v13.panel_sampler.STRATA
            },
            "weighted_stratum_contributions": {
                stratum: {
                    sign: [
                        value
                        * anchor_v13.STRATUM_POPULATION_V13[stratum] / 310.0
                        for value in weighted[sign]
                    ]
                    for sign in anchor_v13.SIGNS_V13
                }
                for stratum in anchor_v13.panel_sampler.STRATA
            },
            "dense_result_sha256": {
                sign: [
                    anchor_v13.canonical_sha256([name, sign, index])
                    for index in range(32)
                ]
                for sign in anchor_v13.SIGNS_V13
            },
        }
    probe = {"schema": "synthetic-reference-probe", "sha256": "e" * 64}
    return {
        "content_sha256_before_self_field": "f" * 64,
        "alpha": 0.0,
        "applications": [],
        "model_update_applied": False,
        "responses": responses,
        "analysis": anchor_v13.analyze_panel_responses_v13(responses),
        "identity_audit": {"pre_probe": probe, "post_probe": copy.deepcopy(probe)},
        "panel_contract": {
            name: {"ordered_row_identity_sha256": "1" * 64}
            for name in anchor_v13.PANEL_NAMES_V13
        },
        "population_boundary_audit_v4": {"passed": True},
        "hardware_coverage": {
            "engine_count": 4, "tp_per_engine": 1, "gpu_ids": [0, 1, 2, 3],
            "population_waves": 8, "signed_waves": 16, "partial_waves": 0,
            "all_engines_generate_every_signed_wave": True,
        },
    }


def test_v16_clock_wraps_only_generation_resolve_and_preserves_v13_order():
    source = inspect.getsource(trainer_v16.TimedTaskABMixinV16.estimate_task_ab_v16)
    perturb = source.index('"perturb_self_weights"')
    started = source.index("started_ns = time.perf_counter_ns()")
    generation = source.index(".generate.remote(", started)
    resolved = source.index("completed_ns = time.perf_counter_ns()")
    restore = source.index("self._restore_all_engines_exact()", resolved)
    scoring = source.index("_score_panel_outputs_v13", restore)
    assert perturb < started < generation < resolved < restore < scoring
    assert source.count("time.perf_counter_ns()") == 2
    assert source.index("warmup = self._resolve") < perturb
    assert source.index("pre_probe = self._base_probe_v13") < perturb
    assert source.index('"content_sha256_before_self_field"') < source.index(
        '"clock": "time.perf_counter_ns"'
    )


def test_v16_timing_is_excluded_from_all_equivalence_hashes(monkeypatch):
    diagnostic = synthetic_diagnostic()
    monkeypatch.setattr(anchor_v13, "validate_diagnostic_v13", lambda value: value)
    first = trainer_v16.compact_arm_v16(diagnostic, {
        "clock": "time.perf_counter_ns",
        "boundary": "blocking_four_engine_generation_resolve_only",
        "warmup_generation_calls_per_engine": 1,
        "wave_seconds": [1.0] * 16,
        "total_seconds": 16.0,
    })
    second = trainer_v16.compact_arm_v16(diagnostic, {
        "clock": "time.perf_counter_ns",
        "boundary": "blocking_four_engine_generation_resolve_only",
        "warmup_generation_calls_per_engine": 1,
        "wave_seconds": [2.0] * 16,
        "total_seconds": 32.0,
    })
    hash_keys = (
        "diagnostic_content_sha256", "dense_result_manifest_sha256",
        "task_output_sha256", "compact_estimator",
    )
    assert {key: first[key] for key in hash_keys} == {
        key: second[key] for key in hash_keys
    }
    assert first["generation_timing"] != second["generation_timing"]


def test_v16_trainer_keeps_v13_mro_and_blocks_historical_entrypoints():
    trainer_class = trainer_v16.load_trainer(load_bundle())
    assert trainer_v16.TimedTaskABMixinV16 in trainer_class.__mro__
    controller = object.__new__(trainer_v16.TimedTaskABMixinV16)
    with pytest.raises(RuntimeError, match="task A/B entrypoint"):
        controller.configure_train_panels_v13(None)
    with pytest.raises(RuntimeError, match="timed task A/B"):
        controller.estimate_train_panels_v13([])
    with pytest.raises(RuntimeError, match="forbids model updates"):
        anchor_v13.TrainPanelDiagnosticMixinV13.apply_seed_coefficients(
            object(), {}, 0.0,
        )


def test_v16_dry_run_reads_no_task_inputs_and_launches_no_gpu(monkeypatch, capsys):
    monkeypatch.setattr(
        driver_v16.driver_v13, "validate_arrow_train_v13",
        lambda *_args: pytest.fail("dry run opened train Arrow"),
    )
    monkeypatch.setattr(
        driver_v16.anchor_v13, "load_panel_bundle_v13",
        lambda *_args: pytest.fail("dry run opened task panels"),
    )
    monkeypatch.setattr(
        driver_v16, "_make_trainer_v16",
        lambda *_args: pytest.fail("dry run constructed trainer"),
    )
    payload = driver_v16.main(cli())
    assert payload["gpu_launched"] is False
    assert payload["fresh_process_per_arm"] is True
    assert payload["recipe"]["task"]["alpha"] == 0.0
    assert payload["recipe"]["hardware"]["gpu_ids"] == [0, 1, 2, 3]
    assert "fused-moe-task-ab-dry-run-v16" in capsys.readouterr().out


def test_v16_arm_environments_have_exactly_one_difference(monkeypatch):
    for name in (
        driver_v16.TUNED_ENV_NAME_V16, *driver_v16.MOE_CONFOUNDING_ENV_V16,
    ):
        monkeypatch.delenv(name, raising=False)
    base = {"PATH": "/bin", "SAME": "yes"}
    default = driver_v16._arm_environment_v16(
        "default_triton", base_environment=base,
    )
    tuned = driver_v16._arm_environment_v16(
        "tuned_triton", base_environment=default,
    )
    binding = driver_v16._arm_environment_difference_v16(default, tuned)
    assert binding["only_difference"] == "VLLM_TUNED_CONFIG_FOLDER"
    assert default.get("VLLM_TUNED_CONFIG_FOLDER") is None
    assert tuned["VLLM_TUNED_CONFIG_FOLDER"] == str(
        prereg_v16.TUNING_DIRECTORY_V16
    )
    changed = dict(tuned, EXTRA="confound")
    with pytest.raises(RuntimeError, match="more than tuned folder"):
        driver_v16._arm_environment_difference_v16(default, changed)


def test_v16_tuned_internal_child_accepts_only_its_exact_folder(monkeypatch):
    implementation = driver_v16.implementation_identity_v16()
    args = driver_v16._parser_v16().parse_args([
        "--expected-implementation-bundle-sha256",
        implementation["bundle_sha256"],
        "--expected-recipe-sha256", "a" * 64,
        "--internal-arm-worker", "tuned_triton",
        "--internal-arm-token", "b" * 64,
        "--internal-arm-output", "/tmp/synthetic-v16-arm.json",
    ])
    monkeypatch.setenv(
        driver_v16.TUNED_ENV_NAME_V16,
        str(prereg_v16.TUNING_DIRECTORY_V16),
    )
    driver_v16.validate_runtime_v16(args, load_bundle(), implementation)
    monkeypatch.setenv(driver_v16.MOE_CONFOUNDING_ENV_V16[0], "1")
    with pytest.raises(ValueError, match="confounding MoE override"):
        driver_v16.validate_runtime_v16(args, load_bundle(), implementation)


@pytest.mark.parametrize("name", driver_v16.MOE_CONFOUNDING_ENV_V16)
def test_v16_rejects_backend_confounders(name, monkeypatch):
    monkeypatch.setenv(name, "1")
    with pytest.raises(ValueError, match="override environment unset"):
        driver_v16.validate_runtime_v16(
            driver_v16._parser_v16().parse_args(["--v16-dry-run"]),
            load_bundle(), driver_v16.implementation_identity_v16(),
        )


def test_v16_parent_runs_sequential_fresh_processes_and_compact_gate(
    tmp_path, monkeypatch,
):
    calls = []
    monkeypatch.setattr(driver_v16, "FROZEN_OUTPUT_DIRECTORY_V16", tmp_path)
    monkeypatch.setattr(
        driver_v16, "_source_provenance_v16",
        lambda implementation: {
            "schema": "synthetic-source",
            "implementation_bundle_sha256": implementation["bundle_sha256"],
        },
    )
    bundle = load_bundle()
    implementation = driver_v16.implementation_identity_v16()
    recipe = driver_v16.recipe_v16(bundle, implementation)

    def fake_child(_args, _bundle, _implementation, _recipe, arm, path, env):
        calls.append((arm, dict(env)))
        compact = synthetic_arm(1.0 if arm == "default_triton" else 0.8)
        envelope = {
            "schema": "eggroll-es-fused-moe-task-arm-v16",
            "arm": arm,
            "recipe_sha256": recipe["content_sha256_before_self_field"],
            "implementation_bundle_sha256": implementation["bundle_sha256"],
            "fresh_process_arm_worker": True,
            "moe_backend": "triton",
            "vllm_tuned_config_folder": (
                None if arm == "default_triton"
                else str(prereg_v16.TUNING_DIRECTORY_V16)
            ),
            "compact_arm": compact,
            "persisted_raw_content": False,
        }
        driver_v16._exclusive_write_json(path, envelope)

    monkeypatch.setattr(driver_v16, "_run_fresh_arm_process_v16", fake_child)
    args = driver_v16._parser_v16().parse_args([])
    args.output_directory = str(tmp_path)
    report = driver_v16.run_exact_v16(
        args, bundle, implementation, recipe,
    )
    assert [arm for arm, _environment in calls] == list(
        prereg_v16.ARM_ORDER_V16
    )
    driver_v16._arm_environment_difference_v16(calls[0][1], calls[1][1])
    assert report["gate"][
        "eligible_for_later_opt_in_training_preregistration"
    ] is True
    assert report["gate"]["eligible_for_model_update"] is False
    assert report["gate"]["eligible_to_open_evaluation"] is False
    serialized = json.dumps(report)
    assert '"responses"' not in serialized
    assert '"questions"' not in serialized
    assert '"answers"' not in serialized


def test_v16_failed_second_child_is_durable_and_does_not_run_more(
    tmp_path, monkeypatch,
):
    calls = []
    monkeypatch.setattr(driver_v16, "FROZEN_OUTPUT_DIRECTORY_V16", tmp_path)
    monkeypatch.setattr(
        driver_v16, "_source_provenance_v16",
        lambda implementation: {"schema": "synthetic-source"},
    )
    bundle = load_bundle()
    implementation = driver_v16.implementation_identity_v16()
    recipe = driver_v16.recipe_v16(bundle, implementation)

    def fail_child(_args, _bundle, _implementation, _recipe, arm, _path, _env):
        calls.append(arm)
        if arm == "tuned_triton":
            raise RuntimeError("synthetic tuned child failure")

    monkeypatch.setattr(driver_v16, "_run_fresh_arm_process_v16", fail_child)
    args = driver_v16._parser_v16().parse_args([])
    args.output_directory = str(tmp_path)
    with pytest.raises(RuntimeError, match="synthetic tuned child failure"):
        driver_v16.run_exact_v16(args, bundle, implementation, recipe)
    attempt = json.loads(driver_v16._attempt_path_v16().read_text())
    assert calls == ["default_triton", "tuned_triton"]
    assert attempt["status"] == "failed"
    assert attempt["model_update_applied"] is False
    assert attempt["evaluation_opened"] is False
    assert attempt["raw_content_persisted"] is False


def test_v16_internal_worker_closes_trainer_and_persists_only_compact(
    tmp_path, monkeypatch,
):
    calls = []
    bundle = load_bundle()
    implementation = driver_v16.implementation_identity_v16()
    recipe = driver_v16.recipe_v16(bundle, implementation)
    output = tmp_path / "arm.json"

    class Trainer:
        def configure_task_ab_v16(self, panels, *, frozen_layer_plan):
            calls.append("configure")
            assert panels == {"synthetic": True}
            assert frozen_layer_plan == bundle

        def estimate_task_ab_v16(self, seeds):
            calls.append("estimate")
            assert seeds == anchor_v13.PERTURBATION_SEEDS_V13
            return {"raw": "memory-only"}, {"timing": True}

    monkeypatch.setattr(
        driver_v16.driver_v13, "validate_arrow_train_v13", lambda _path: True,
    )
    monkeypatch.setattr(
        driver_v16.anchor_v13, "load_panel_bundle_v13",
        lambda: {"synthetic": True},
    )
    monkeypatch.setattr(driver_v16, "_make_trainer_v16", lambda *_args: Trainer())
    monkeypatch.setattr(
        driver_v16.trainer_v16, "compact_arm_v16",
        lambda diagnostic, timing: (
            calls.append((diagnostic, timing)) or synthetic_arm()
        ),
    )
    monkeypatch.setattr(
        driver_v16.base, "close_trainer", lambda _trainer: calls.append("close"),
    )
    monkeypatch.delenv(driver_v16.TUNED_ENV_NAME_V16, raising=False)
    args = driver_v16._parser_v16().parse_args([
        "--expected-implementation-bundle-sha256",
        implementation["bundle_sha256"],
        "--expected-recipe-sha256", recipe["content_sha256_before_self_field"],
        "--internal-arm-worker", "default_triton",
        "--internal-arm-token", driver_v16._arm_token_v16(
            recipe["content_sha256_before_self_field"],
            implementation["bundle_sha256"], "default_triton",
        ),
        "--internal-arm-output", str(output),
    ])
    driver_v16._run_arm_worker_v16(args, bundle, implementation, recipe)
    assert calls[-1] == "close"
    persisted = json.loads(output.read_text())
    assert persisted["persisted_raw_content"] is False
    assert "memory-only" not in output.read_text()


def test_v16_real_child_command_has_exact_pins_and_no_dry_flag(
    tmp_path, monkeypatch,
):
    captured = {}
    bundle = load_bundle()
    implementation = driver_v16.implementation_identity_v16()
    recipe = driver_v16.recipe_v16(bundle, implementation)
    monkeypatch.setattr(
        driver_v16.subprocess, "run",
        lambda command, **kwargs: captured.update(
            {"command": command, "kwargs": kwargs}
        ),
    )
    environment = {"PATH": "/bin"}
    driver_v16._run_fresh_arm_process_v16(
        driver_v16._parser_v16().parse_args([]), bundle, implementation,
        recipe, "default_triton", tmp_path / "compact.json", environment,
    )
    command = captured["command"]
    assert "--v16-dry-run" not in command
    assert command.count("--internal-arm-worker") == 1
    assert command[command.index("--internal-arm-worker") + 1] == "default_triton"
    assert implementation["bundle_sha256"] in command
    assert recipe["content_sha256_before_self_field"] in command
    assert captured["kwargs"]["env"] is environment
    assert captured["kwargs"]["check"] is True
