#!/usr/bin/env python3
"""CPU-only fail-closed tests for the V19A attribution runtime."""

import copy
import json

import numpy as np
import pytest

import run_eggroll_es_disjoint_tier_attribution_v19a as driver_v19a


PLAN_SHA = driver_v19a.driver_v13.driver_v10.MIDDLE_LATE_PLAN_SHA256_V10


def load_layer_bundle():
    spec = driver_v19a.anchor_v11.FROZEN_STABILITY_PLANS_V11[PLAN_SHA]
    return driver_v19a.anchor_v13.load_frozen_layer_plan_v13(
        spec["path"],
        expected_file_sha256=spec["file_sha256"],
        expected_plan_sha256=PLAN_SHA,
        expected_model_config_sha256=driver_v19a.anchor_v11.MODEL_CONFIG_SHA256_V11,
    )


def cli(extra=None):
    spec = driver_v19a.anchor_v11.FROZEN_STABILITY_PLANS_V11[PLAN_SHA]
    return [
        "--layer-plan-json", str(spec["path"]),
        "--expected-layer-plan-file-sha256", spec["file_sha256"],
        "--expected-layer-plan-sha256", PLAN_SHA,
        "--expected-model-config-sha256",
        driver_v19a.anchor_v11.MODEL_CONFIG_SHA256_V11,
        "--v19a-dry-run",
        *(extra or []),
    ]


def runtime_inputs():
    layer = load_layer_bundle()
    implementation = driver_v19a.implementation_identity_v19a()
    preregistration, panels = driver_v19a._load_bound_inputs_v19a(layer)
    recipe = driver_v19a.recipe_v19a(
        layer,
        preregistration,
        panels,
        implementation,
        driver_v19a._declared_moe_backend_v19a(),
    )
    return layer, panels, implementation, recipe


def synthetic_summary():
    endpoint_names = driver_v19a.prereg_v19a.ENDPOINT_NAMES_V19A
    arms = {
        arm: {
            "all_panel_spreads_nonzero": True,
            "endpoint_values": {
                name: 0.5 + 0.001 * endpoint_index
                for endpoint_index, name in enumerate(endpoint_names)
            },
            "compact_estimator_sha256": f"{arm_index + 1:064x}",
        }
        for arm_index, arm in enumerate(driver_v19a.mechanics_v19a.ARMS_V19A)
    }
    value = {
        "schema": "eggroll-es-disjoint-tier-attribution-summary-v19a",
        "experiment_name": driver_v19a.EXPERIMENT_NAME_V19A,
        "alpha": 0.0,
        "sigma": 0.0003,
        "model_update_applied": False,
        "evaluation_surfaces_opened": False,
        "dataset_promotion_applied": False,
        "frame_content_sha256": (
            driver_v19a.mechanics_v19a.FRAME_CERTIFICATE_CONTENT_SHA256_V19A
        ),
        "perturbation_basis_sha256": (
            driver_v19a.prereg_v19a.PERTURBATION_BASIS_SHA256_V19A
        ),
        "runtime_integrity": {"all_integrity_audits_passed": True},
        "arms": arms,
        "paired_bootstrap": {
            "seed": driver_v19a.prereg_v19a.BOOTSTRAP_SEED_V19A,
            "repetitions": 50_000,
            "one_sided_quantile": 0.05 / 36,
            "draw_plan_content_sha256": (
                driver_v19a.mechanics_v19a.BOOTSTRAP_DRAW_PLAN_CONTENT_SHA256_V19A
            ),
            "whole_panel_block_resampling_used": False,
            "comparisons": {
                arm: {
                    name: {
                        "patch_minus_production": 0.01,
                        "familywise_lcb": 0.001,
                        "noninferiority_margin": 0.0,
                    }
                    for name in endpoint_names
                }
                for arm in driver_v19a.mechanics_v19a.ARMS_V19A[1:]
            },
        },
        "persisted_response_vectors_or_row_content": False,
        "bootstrap_draws_persisted": False,
        "unit_scores_persisted": False,
    }
    value["content_sha256_before_self_field"] = driver_v19a.canonical_sha256(value)
    return value


def synthetic_configuration():
    return {
        "schema": "eggroll-es-disjoint-tier-runtime-configuration-v19a",
        "layer_plan_install_sha256": "4" * 64,
        "reference_identity_sha256": "5" * 64,
        "unselected_origin_sha256": "6" * 64,
        "panel_bundle_content_sha256": (
            driver_v19a.mechanics_v19a.PANEL_BUNDLE_CONTENT_SHA256_V19A
        ),
        "worker_extension": (
            "eggroll_es_worker_v13.TrainPanelDiagnosticWorkerExtensionV13"
        ),
        "engine_count": 4,
        "tp_per_engine": 1,
        "gpu_ids": [0, 1, 2, 3],
        "model_update_allowed": False,
        "checkpoint_write_allowed": False,
        "evaluation_surfaces_opened": False,
        "dataset_promotion_allowed": False,
    }


def synthetic_runtime_audit():
    value = {
        "schema": "eggroll-es-disjoint-tier-runtime-compact-audit-v19a",
        "fixed_request_identity_sha256": "7" * 64,
        "token_boundary_audit_sha256": "8" * 64,
        "pre_post_probe_identity_sha256": "9" * 64,
        "signed_wave_schedule_sha256": "a" * 64,
        "restore_checks_sha256": "b" * 64,
        "dense_result_commitments_sha256": "c" * 64,
        "population_boundary_audit_sha256": "d" * 64,
        "unselected_origin_sha256": "6" * 64,
        "unselected_origin_audit_sha256": "e" * 64,
        "signed_wave_count": 16,
        "panel_count": 10,
        "requests_per_engine_per_signed_wave": 990,
        "dense_result_commitment_count": 2560,
        "per_unit_scores_persisted": False,
        "bootstrap_replicates_persisted": False,
        "bootstrap_draws_persisted": False,
        "row_content_persisted": False,
    }
    value["content_sha256_before_self_field"] = driver_v19a.canonical_sha256(value)
    return value


def test_v19a_dry_run_binds_exact_artifacts_recipe_and_counts(monkeypatch, capsys):
    monkeypatch.setattr(
        driver_v19a,
        "_make_trainer_v19a",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry run constructed a trainer")
        ),
    )
    first = driver_v19a.main(cli())
    first_output = capsys.readouterr().out
    second = driver_v19a.main(cli())
    second_output = capsys.readouterr().out
    assert first == second
    assert first_output == second_output
    assert first["gpu_launched"] is False
    assert first["recipe"]["hardware"] == {
        "engine_count": 4,
        "tp_per_engine": 1,
        "gpu_ids": [0, 1, 2, 3],
        "all_engines_every_signed_wave": True,
    }
    assert first["recipe"]["scoring"][
        "requests_per_engine_per_signed_wave_all_arms"
    ] == 990
    assert first["recipe"]["scoring"]["dense_result_commitment_count"] == 2560
    assert first["recipe"]["perturbation"]["layers"] == [20, 21, 22, 23]
    assert first["recipe"]["perturbation"]["seeds"] == (
        driver_v19a.prereg_v19a.PERTURBATION_SEEDS_V19A
    )
    assert first["recipe"]["bootstrap_draw_plan_content_sha256"] == (
        "e1842969819e6d26836aafe7d4640d2cb0f6540530007b60a9ce950103b0865a"
    )
    assert first["recipe"]["materialized_panel_bundle_content_sha256"] == (
        "b9bfc1868f5e2a6f54cd9531e0b759872020d2bc8fb9e8a8a287b548293d4f06"
    )
    assert first["recipe"]["moe_backend"] == {
        "moe_backend": "default_triton",
        "override_environment": {
            name: None for name in driver_v19a.MOE_OVERRIDE_ENVIRONMENT_V19A
        },
        "v16_task_ab_decision_retained": "default_triton",
    }
    assert first["implementation"]["files"]["trainer_mechanics_v19a"][
        "file_sha256"
    ] == "38db195e18f8b2dd9483c77a7e93d8026d8fb6da2319b158b38a6e2218b17924"
    for key in (
        "model_update_allowed",
        "checkpoint_write_allowed",
        "evaluation_surfaces_opened",
        "dataset_promotion_allowed",
    ):
        assert first["recipe"][key] is False


@pytest.mark.parametrize("name", driver_v19a.MOE_OVERRIDE_ENVIRONMENT_V19A)
def test_v19a_rejects_all_five_moe_overrides_before_assembly(name, monkeypatch):
    monkeypatch.setenv(name, "synthetic-confound")
    monkeypatch.setattr(
        driver_v19a,
        "implementation_identity_v19a",
        lambda: (_ for _ in ()).throw(
            AssertionError("override reached runtime assembly")
        ),
    )
    with pytest.raises(ValueError, match="every MoE backend override unset"):
        driver_v19a.main(cli())


def test_v19a_rejects_hash_tampering_forbidden_argv_and_unknown_controls():
    layer, _panels, implementation, recipe = runtime_inputs()
    args = driver_v19a._parser_v19a().parse_args(["--v19a-dry-run"])
    args.expected_recipe_sha256 = "0" * 64
    with pytest.raises(ValueError, match="recipe hash changed"):
        driver_v19a.validate_runtime_v19a(args, layer, implementation, recipe)
    args.expected_recipe_sha256 = recipe["content_sha256_before_self_field"]
    args.expected_implementation_bundle_sha256 = "0" * 64
    with pytest.raises(ValueError, match="implementation bundle hash changed"):
        driver_v19a.validate_runtime_v19a(args, layer, implementation, recipe)
    for forbidden in ("--checkpoint", "--validation-json", "--eval-dataset"):
        with pytest.raises(ValueError, match="forbidden runtime surface"):
            driver_v19a.main(cli([forbidden, "/tmp/x"]))
    with pytest.raises(SystemExit):
        driver_v19a.main(cli(["--alpha", "0.1"]))


class FakeSignedWaveController(
    driver_v19a.DisjointTierAttributionRuntimeMixinV19A
):
    def __init__(self, fail_arm=None):
        self.events = []
        self.fail_arm = fail_arm

    def _perturb_signed_wave_v19a(self, engine_seeds, negate):
        self.events.append(("perturb", list(engine_seeds), bool(negate)))

    def _score_resident_arm_v19a(
        self, arm, _prepared, _schedule_item, _unit_scores, dense_commitments,
    ):
        self.events.append(("score", arm))
        if arm == self.fail_arm:
            raise RuntimeError("synthetic resident arm failure")
        dense_commitments.append(arm)
        return arm

    def _restore_and_verify_signed_wave_v19a(self):
        self.events.append(("restore_verify",))
        return "f" * 64


def test_v19a_wave_scores_latin_order_and_restores_exactly_once_on_failure():
    schedule = driver_v19a.resident_signed_wave_schedule_v19a()
    unit_scores = {
        arm: np.empty((10, 2, 32, request_count))
        for arm, request_count in (
            driver_v19a.frame_v19a.ARM_REQUESTS_PER_PANEL_V19A.items()
        )
    }
    controller = FakeSignedWaveController()
    assert controller._run_signed_wave_v19a(
        schedule[3], {}, unit_scores, []
    ) == "f" * 64
    assert controller.events == [
        ("perturb", schedule[3]["engine_seeds"], True),
        *(("score", arm) for arm in schedule[3]["resident_arm_order"]),
        ("restore_verify",),
    ]
    failing_arm = schedule[0]["resident_arm_order"][2]
    failed = FakeSignedWaveController(fail_arm=failing_arm)
    with pytest.raises(RuntimeError, match="synthetic resident arm failure"):
        failed._run_signed_wave_v19a(schedule[0], {}, unit_scores, [])
    assert sum(event[0] == "perturb" for event in failed.events) == 1
    assert sum(event[0] == "restore_verify" for event in failed.events) == 1
    assert failed.events[-1] == ("restore_verify",)
    for malformed in ({}, {**schedule[-1], "signed_wave_index": -1}):
        with pytest.raises(RuntimeError, match="schedule item changed"):
            controller._run_signed_wave_v19a(malformed, {}, unit_scores, [])


def test_v19a_population_and_unselected_origin_audit_is_bound_and_tamper_evident():
    install = {
        "plan_sha256": "1" * 64,
        "unselected_origin_sha256": "2" * 64,
    }
    boundary = {
        "schema": "eggroll-es-population-boundary-audit-v4",
        "engine_count": 4,
        "runtime_mapping": copy.deepcopy(install),
        "unselected_origin_sha256": "2" * 64,
        "worker_reports": [
            {"rank": rank, "passed": True, **install} for rank in range(4)
        ],
        "passed": True,
    }
    boundary["audit_sha256"] = driver_v19a.canonical_sha256(boundary)
    audit, digest = driver_v19a._validate_population_boundary_v19a(
        boundary, install
    )
    assert audit["passed"] is True
    assert audit["unselected_origin_sha256"] == install[
        "unselected_origin_sha256"
    ]
    assert digest == driver_v19a.canonical_sha256(audit)
    tampered = copy.deepcopy(boundary)
    tampered["worker_reports"][-1]["unselected_origin_sha256"] = "0" * 64
    tampered["audit_sha256"] = driver_v19a.canonical_sha256({
        key: value for key, value in tampered.items() if key != "audit_sha256"
    })
    with pytest.raises(RuntimeError, match="unselected-origin audit failed"):
        driver_v19a._validate_population_boundary_v19a(tampered, install)


class FakeExactAccountingController(
    driver_v19a.DisjointTierAttributionRuntimeMixinV19A
):
    def __init__(self, commitments_per_wave=160):
        self.commitments_per_wave = commitments_per_wave
        self.wave_calls = 0
        self._v19a_panel_bundle = {"synthetic": True}
        self._v4_layer_plan_install = {
            "plan_sha256": "1" * 64,
            "unselected_origin_sha256": "2" * 64,
        }

    def _prepared_fixed_batches_v19a(self):
        self._v19a_token_boundary_audit_sha256 = "3" * 64
        return {}, {arm: f"{index + 4:064x}" for index, arm in enumerate(
            driver_v19a.mechanics_v19a.ARMS_V19A
        )}

    def _base_probe_v19a(self, _prepared):
        return {arm: "8" * 64 for arm in driver_v19a.mechanics_v19a.ARMS_V19A}

    def _run_signed_wave_v19a(
        self, _schedule_item, _prepared, unit_scores, dense_commitments,
    ):
        self.wave_calls += 1
        for values in unit_scores.values():
            values.fill(0.25)
        dense_commitments.extend(
            f"{self.wave_calls:02x}" * 32
            for _ in range(self.commitments_per_wave)
        )
        return f"{self.wave_calls:064x}"

    def _population_boundary_audit_v4(self, _iteration):
        install = self._v4_layer_plan_install
        value = {
            "schema": "eggroll-es-population-boundary-audit-v4",
            "engine_count": 4,
            "runtime_mapping": copy.deepcopy(install),
            "unselected_origin_sha256": install["unselected_origin_sha256"],
            "worker_reports": [
                {"rank": rank, "passed": True, **install} for rank in range(4)
            ],
            "passed": True,
        }
        value["audit_sha256"] = driver_v19a.canonical_sha256(value)
        return value


def test_v19a_coordinator_enforces_16_waves_2560_commitments_and_all_shapes(
    monkeypatch,
):
    compact = synthetic_summary()

    def compact_builder(unit_scores, panel_bundle):
        assert panel_bundle == {"synthetic": True}
        assert {
            arm: values.shape
            for arm, values in unit_scores.items()
        } == {
            arm: (10, 2, 32, request_count)
            for arm, request_count in (
                driver_v19a.frame_v19a.ARM_REQUESTS_PER_PANEL_V19A.items()
            )
        }
        assert all(np.isfinite(values).all() for values in unit_scores.values())
        return {
            "arms": compact["arms"],
            "paired_bootstrap": compact["paired_bootstrap"],
        }

    monkeypatch.setattr(
        driver_v19a.mechanics_v19a,
        "build_compact_estimator_summary_v19a",
        compact_builder,
    )
    controller = FakeExactAccountingController()
    summary, gate, audit = controller.estimate_disjoint_tier_attribution_v19a(
        driver_v19a.prereg_v19a.PERTURBATION_SEEDS_V19A
    )
    assert controller.wave_calls == 16
    assert audit["signed_wave_count"] == 16
    assert audit["panel_count"] == 10
    assert audit["requests_per_engine_per_signed_wave"] == 990
    assert audit["dense_result_commitment_count"] == 2560
    assert audit["unselected_origin_sha256"] == "2" * 64
    assert summary["runtime_integrity"]["all_integrity_audits_passed"] is True
    assert gate == driver_v19a.mechanics_v19a.evaluate_attribution_gate_v19a(summary)

    incomplete = FakeExactAccountingController(commitments_per_wave=159)
    with pytest.raises(RuntimeError, match="population capture is incomplete"):
        incomplete.estimate_disjoint_tier_attribution_v19a(
            driver_v19a.prereg_v19a.PERTURBATION_SEEDS_V19A
        )


def test_v19a_compact_report_rejects_hidden_content_and_authority():
    _layer, _panels, implementation, recipe = runtime_inputs()
    summary = synthetic_summary()
    report = {
        "schema": "eggroll-es-disjoint-tier-attribution-report-v19a",
        "recipe": recipe,
        "configuration": synthetic_configuration(),
        "runtime_audit": synthetic_runtime_audit(),
        "summary": summary,
        "gate": driver_v19a.mechanics_v19a.evaluate_attribution_gate_v19a(summary),
        "implementation": implementation,
        "model_update_applied": False,
        "checkpoint_written": False,
        "evaluation_surfaces_opened": False,
        "dataset_promotion_applied": False,
    }
    report["content_sha256_before_self_field"] = driver_v19a.canonical_sha256(report)
    assert driver_v19a.validate_compact_report_v19a(
        report,
        expected_recipe=recipe,
        expected_implementation=implementation,
    ) == report
    serialized = json.dumps(report, sort_keys=True)
    for forbidden in (
        '"questions"', '"answers"', '"prompt_token_ids"', '"unit_scores"',
        '"responses"', '"coefficients"', '"bootstrap_draws"', '"row_sha256"',
    ):
        assert forbidden not in serialized

    hidden = copy.deepcopy(report)
    hidden["runtime_audit"]["unit_scores"] = [0.1]
    hidden["runtime_audit"]["content_sha256_before_self_field"] = (
        driver_v19a.canonical_sha256({
            key: value
            for key, value in hidden["runtime_audit"].items()
            if key != "content_sha256_before_self_field"
        })
    )
    hidden["content_sha256_before_self_field"] = driver_v19a.canonical_sha256({
        key: value
        for key, value in hidden.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="compact report contract"):
        driver_v19a.validate_compact_report_v19a(
            hidden,
            expected_recipe=recipe,
            expected_implementation=implementation,
        )

    authority = copy.deepcopy(report)
    authority["gate"]["model_update_authorized"] = True
    authority["content_sha256_before_self_field"] = driver_v19a.canonical_sha256({
        key: value
        for key, value in authority.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="compact report contract"):
        driver_v19a.validate_compact_report_v19a(
            authority,
            expected_recipe=recipe,
            expected_implementation=implementation,
        )


def test_v19a_fake_real_path_uses_o_excl_unique_paths_and_cleanup(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(driver_v19a, "FROZEN_OUTPUT_DIRECTORY_V19A", tmp_path)
    layer, panels, implementation, recipe = runtime_inputs()
    summary = synthetic_summary()
    gate = driver_v19a.mechanics_v19a.evaluate_attribution_gate_v19a(summary)

    class FakeTrainer:
        def configure_disjoint_tier_attribution_v19a(self, panel_bundle, **_kwargs):
            assert panel_bundle["content_sha256_before_self_field"] == (
                driver_v19a.mechanics_v19a.PANEL_BUNDLE_CONTENT_SHA256_V19A
            )
            return synthetic_configuration()

        def estimate_disjoint_tier_attribution_v19a(self, seeds):
            assert seeds == driver_v19a.prereg_v19a.PERTURBATION_SEEDS_V19A
            return summary, gate, synthetic_runtime_audit()

    launch_ids = iter(("success-one", "success-two"))
    monkeypatch.setattr(driver_v19a, "_new_launch_id_v19a", lambda: next(launch_ids))
    monkeypatch.setattr(
        driver_v19a,
        "_source_provenance_v19a",
        lambda current: {
            "schema": "synthetic-committed-source-v19a",
            "implementation_bundle_sha256": current["bundle_sha256"],
            "content_sha256_before_self_field": "f" * 64,
        },
    )
    monkeypatch.setattr(driver_v19a, "_make_trainer_v19a", lambda _value: FakeTrainer())
    closed = []
    monkeypatch.setattr(
        driver_v19a.base, "close_trainer", lambda trainer: closed.append(trainer)
    )
    first = driver_v19a.run_exact_v19a(layer, panels, implementation, recipe)
    second = driver_v19a.run_exact_v19a(layer, panels, implementation, recipe)
    assert first == second
    assert len(closed) == 2
    root = tmp_path / driver_v19a.EXPERIMENT_NAME_V19A
    assert len(list((root / "attempts").glob("*.json"))) == 2
    reports = list((root / "runs").glob(f"*/{driver_v19a.REPORT_NAME_V19A}"))
    assert len(reports) == 2
    assert all(json.loads(path.read_text()) == first for path in reports)


def test_v19a_failure_attempt_hashes_error_and_closes_trainer(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(driver_v19a, "FROZEN_OUTPUT_DIRECTORY_V19A", tmp_path)
    layer, panels, implementation, recipe = runtime_inputs()

    class FailingTrainer:
        def configure_disjoint_tier_attribution_v19a(self, *_args, **_kwargs):
            raise RuntimeError("synthetic sensitive failure payload")

    monkeypatch.setattr(driver_v19a, "_new_launch_id_v19a", lambda: "failure")
    monkeypatch.setattr(
        driver_v19a,
        "_source_provenance_v19a",
        lambda current: {
            "schema": "synthetic-committed-source-v19a",
            "implementation_bundle_sha256": current["bundle_sha256"],
            "content_sha256_before_self_field": "e" * 64,
        },
    )
    trainer = FailingTrainer()
    monkeypatch.setattr(driver_v19a, "_make_trainer_v19a", lambda _value: trainer)
    closed = []
    monkeypatch.setattr(
        driver_v19a.base, "close_trainer", lambda value: closed.append(value)
    )
    with pytest.raises(RuntimeError, match="synthetic sensitive failure payload"):
        driver_v19a.run_exact_v19a(layer, panels, implementation, recipe)
    assert closed == [trainer]
    attempt_path = next(
        (tmp_path / driver_v19a.EXPERIMENT_NAME_V19A / "attempts").glob("*.json")
    )
    attempt = json.loads(attempt_path.read_text())
    assert attempt["status"] == "failed"
    assert attempt["failure_type"] == "RuntimeError"
    assert len(attempt["failure_sha256"]) == 64
    assert "failure_message" not in attempt
    assert "traceback" not in attempt
    assert "sensitive failure payload" not in json.dumps(attempt)


def test_v19a_mutation_checkpoint_and_evaluation_surfaces_are_closed():
    trainer = object.__new__(driver_v19a.DisjointTierAttributionRuntimeMixinV19A)
    for method in (
        "configure_production_patch_compat_v18a",
        "estimate_production_patch_compat_v18a",
        "estimate_step_coefficients",
        "apply_seed_coefficients",
        "train_step",
        "evaluate_handle",
        "evaluate_population_on_batch",
        "eval_step",
        "fit",
    ):
        with pytest.raises(RuntimeError, match="closes every update"):
            getattr(trainer, method)()
