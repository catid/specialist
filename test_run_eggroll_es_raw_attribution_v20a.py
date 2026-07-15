#!/usr/bin/env python3
"""CPU-only fail-closed tests for authoritative raw V20A attribution."""

import copy
import json

import numpy as np
import pytest

import run_eggroll_es_raw_attribution_v20a as driver_v20a


PLAN_SHA = (
    driver_v20a.equivalence_v20a.driver_v19a.driver_v13.driver_v10
    .MIDDLE_LATE_PLAN_SHA256_V10
)


def load_layer_bundle():
    spec = driver_v20a.anchor_v11.FROZEN_STABILITY_PLANS_V11[PLAN_SHA]
    return driver_v20a.anchor_v13.load_frozen_layer_plan_v13(
        spec["path"],
        expected_file_sha256=spec["file_sha256"],
        expected_plan_sha256=PLAN_SHA,
        expected_model_config_sha256=driver_v20a.anchor_v11.MODEL_CONFIG_SHA256_V11,
    )


def cli(extra=None):
    spec = driver_v20a.anchor_v11.FROZEN_STABILITY_PLANS_V11[PLAN_SHA]
    return [
        "--layer-plan-json", str(spec["path"]),
        "--expected-layer-plan-file-sha256", spec["file_sha256"],
        "--expected-layer-plan-sha256", PLAN_SHA,
        "--expected-model-config-sha256", driver_v20a.anchor_v11.MODEL_CONFIG_SHA256_V11,
        "--v20a-raw-dry-run",
        *(extra or []),
    ]


def runtime_inputs():
    layer = load_layer_bundle()
    implementation = driver_v20a.implementation_identity_v20a()
    preregistration, panels, evidence = driver_v20a._load_bound_inputs_v20a(layer)
    recipe = driver_v20a.recipe_v20a(
        layer,
        preregistration,
        panels,
        evidence,
        implementation,
        driver_v20a.equivalence_v20a._declared_moe_backend_v20a(),
    )
    return layer, panels, implementation, recipe


def synthetic_compact():
    endpoint_names = driver_v20a.prereg_v20a.ENDPOINT_NAMES_V20A
    arms = {
        arm: {
            "all_panel_spreads_nonzero": True,
            "endpoint_values": {name: 1.0 for name in endpoint_names},
            "compact_estimator_sha256": f"{index + 1:064x}",
        }
        for index, arm in enumerate(driver_v20a.mechanics_v20a.ARMS_V20A)
    }
    comparisons = {
        contrast: {
            name: {
                "treatment_minus_control": 0.25,
                "familywise_lcb": 0.1,
                "noninferiority_margin": 0.0,
            }
            for name in endpoint_names
        }
        for contrast in driver_v20a.mechanics_v20a.CONTRASTS_V20A
    }
    return {
        "arms": arms,
        "paired_bootstrap": {
            "seed": driver_v20a.prereg_v20a.BOOTSTRAP_SEED_V20A,
            "repetitions": 50_000,
            "one_sided_quantile": 0.05 / 60,
            "draw_plan_content_sha256": (
                driver_v20a.mechanics_v20a.DRAW_PLAN_CONTENT_SHA256_V20A
            ),
            "whole_panel_block_resampling_used": False,
            "comparisons": comparisons,
        },
    }


def synthetic_summary():
    compact = synthetic_compact()
    value = {
        "schema": "eggroll-es-raw-nested-tier-attribution-summary-v20a",
        "experiment_name": driver_v20a.EXPERIMENT_NAME_V20A,
        "alpha": 0.0,
        "sigma": 0.0003,
        "runtime_integrity": {
            "all_four_tp1_engines_every_signed_wave": True,
            "all_ten_panels_every_direction_sign_and_arm": True,
            "all_sixteen_signed_waves_complete": True,
            "latin_arm_order_complete": True,
            "exact_reference_restored_once_per_signed_wave": True,
            "pre_post_raw_reference_probes_equal": True,
            "population_boundary_audit_passed": True,
            "unselected_origin_audit_passed": True,
            "union_scoring_called_or_used": False,
            "all_integrity_audits_passed": True,
        },
        "arms": compact["arms"],
        "paired_bootstrap": compact["paired_bootstrap"],
        "union_scoring_authorized_or_used": False,
        "model_update_applied": False,
        "checkpoint_written": False,
        "evaluation_surfaces_opened": False,
        "dataset_promotion_applied": False,
        "persisted_response_vectors_or_row_content": False,
        "bootstrap_draws_persisted": False,
        "unit_scores_persisted": False,
    }
    value["content_sha256_before_self_field"] = driver_v20a.canonical_sha256(value)
    return value


def synthetic_configuration():
    return {
        "schema": "eggroll-es-raw-attribution-runtime-configuration-v20a",
        "layer_plan_install_sha256": "1" * 64,
        "reference_identity_sha256": "2" * 64,
        "unselected_origin_sha256": "3" * 64,
        "panel_bundle_content_sha256": (
            driver_v20a.mechanics_v20a.PANEL_BUNDLE_CONTENT_SHA256_V20A
        ),
        "engine_count": 4,
        "tp_per_engine": 1,
        "gpu_ids": [0, 1, 2, 3],
        "model_update_allowed": False,
        "checkpoint_write_allowed": False,
        "evaluation_surfaces_opened": False,
        "dataset_promotion_allowed": False,
        "union_scoring_authorized": False,
        "union_planner_called": False,
        "train_only_attribution_runtime_opened": True,
    }


def synthetic_runtime_audit():
    value = {
        "schema": "eggroll-es-raw-attribution-runtime-audit-v20a",
        "fixed_request_identity_sha256": "4" * 64,
        "token_boundary_audit_sha256": "5" * 64,
        "pre_post_probe_identity_sha256": "6" * 64,
        "signed_wave_schedule_sha256": "7" * 64,
        "restore_checks_sha256": "8" * 64,
        "dense_result_commitments_sha256": "9" * 64,
        "population_boundary_audit_sha256": "a" * 64,
        "unselected_origin_sha256": "b" * 64,
        "unselected_origin_audit_sha256": "c" * 64,
        "signed_wave_count": 16,
        "panel_count": 10,
        "requests_per_engine_per_signed_wave": 1020,
        "dense_result_commitment_count": 2560,
        "union_scoring_called_or_used": False,
        "per_unit_scores_persisted": False,
        "bootstrap_replicates_persisted": False,
        "bootstrap_draws_persisted": False,
        "row_content_persisted": False,
    }
    value["content_sha256_before_self_field"] = driver_v20a.canonical_sha256(value)
    return value


def test_raw_v20a_dry_run_binds_exact_runtime_and_opens_no_gpu(monkeypatch, capsys):
    monkeypatch.setattr(
        driver_v20a,
        "_make_trainer_v20a",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry run constructed a trainer")
        ),
    )
    first = driver_v20a.main(cli())
    first_output = capsys.readouterr().out
    second = driver_v20a.main(cli())
    second_output = capsys.readouterr().out
    assert first == second
    assert first_output == second_output
    assert first["gpu_launched"] is False
    assert first["union_scoring_authorized_or_used"] is False
    recipe = first["recipe"]
    assert recipe["model"].endswith("Qwen3.6-35B-A3B")
    assert recipe["layers"] == [20, 21, 22, 23]
    assert recipe["alpha"] == 0.0
    assert recipe["signed_wave_count"] == 16
    assert recipe["authoritative_raw_scoring"] == {
        "requests_per_engine_per_signed_wave": 1020,
        "requests_by_arm": {
            "production_only": 240,
            "patch_tier_2_only": 250,
            "patch_tiers_2_3": 260,
            "patch_all_tiers": 270,
        },
        "dense_result_commitment_count": 2560,
    }
    assert recipe["union_scoring"]["authorized"] is False
    assert recipe["union_scoring"]["planner_called_by_runtime"] is False
    assert recipe["hardware"] == {
        "engine_count": 4,
        "tp_per_engine": 1,
        "gpu_ids": [0, 1, 2, 3],
        "all_four_engines_every_signed_wave": True,
    }
    assert recipe["moe_backend"] == (
        driver_v20a.equivalence_v20a._declared_moe_backend_v20a()
    )


@pytest.mark.parametrize("name", driver_v20a.MOE_OVERRIDE_ENVIRONMENT_V20A)
def test_raw_v20a_rejects_every_moe_override(name, monkeypatch):
    monkeypatch.setenv(name, "synthetic-confound")
    with pytest.raises(ValueError, match="every MoE backend override unset"):
        driver_v20a.main(cli())


def test_raw_v20a_rejects_hash_tampering_forbidden_and_unknown_controls():
    layer, _panels, implementation, recipe = runtime_inputs()
    args = driver_v20a._parser_v20a().parse_args(["--v20a-raw-dry-run"])
    args.expected_recipe_sha256 = "0" * 64
    with pytest.raises(ValueError, match="recipe hash changed"):
        driver_v20a.validate_runtime_v20a(args, layer, implementation, recipe)
    args.expected_recipe_sha256 = recipe["content_sha256_before_self_field"]
    args.expected_implementation_bundle_sha256 = "0" * 64
    with pytest.raises(ValueError, match="implementation bundle hash changed"):
        driver_v20a.validate_runtime_v20a(args, layer, implementation, recipe)
    for forbidden in ("--checkpoint", "--validation-json", "--eval-dataset"):
        with pytest.raises(ValueError, match="forbidden runtime surface"):
            driver_v20a.main(cli([forbidden, "/tmp/x"]))
    with pytest.raises(SystemExit):
        driver_v20a.main(cli(["--alpha", "0.1"]))


def test_raw_v20a_schedule_is_exact_and_latin_balanced():
    schedule = driver_v20a.resident_signed_wave_schedule_v20a()
    assert len(schedule) == 16
    assert [item["signed_wave_index"] for item in schedule] == list(range(16))
    assert sum(len(item["engine_seeds"]) for item in schedule) == 64
    for sign in ("plus", "minus"):
        signed = [item for item in schedule if item["sign"] == sign]
        for arm in driver_v20a.mechanics_v20a.ARMS_V20A:
            assert sorted(item["resident_arm_order"].index(arm) for item in signed) == [
                0, 0, 1, 1, 2, 2, 3, 3,
            ]
    with pytest.raises(RuntimeError, match="perturbation basis changed"):
        driver_v20a.resident_signed_wave_schedule_v20a(list(range(32)))


class FakeWaveController(driver_v20a.RawNestedTierAttributionRuntimeMixinV20A):
    def __init__(self, fail_arm=None):
        self.events = []
        self.fail_arm = fail_arm

    def _perturb_signed_wave_v19a(self, seeds, negate):
        self.events.append(("perturb", tuple(seeds), negate))

    def _score_resident_arm_v20a(self, arm, *_args):
        self.events.append(("score", arm))
        if arm == self.fail_arm:
            raise RuntimeError("synthetic raw arm failure")
        return arm

    def _restore_and_verify_signed_wave_v19a(self):
        self.events.append(("restore",))
        return "d" * 64


def test_raw_v20a_wave_scores_resident_order_and_restores_once_even_on_failure():
    item = driver_v20a.resident_signed_wave_schedule_v20a()[0]
    controller = FakeWaveController()
    assert controller._run_signed_wave_v20a(item, {}, {}, []) == "d" * 64
    assert [event[1] for event in controller.events if event[0] == "score"] == (
        item["resident_arm_order"]
    )
    assert controller.events[-1] == ("restore",)
    assert sum(event[0] == "restore" for event in controller.events) == 1

    failing = FakeWaveController(fail_arm=item["resident_arm_order"][1])
    with pytest.raises(RuntimeError, match="synthetic raw arm failure"):
        failing._run_signed_wave_v20a(item, {}, {}, [])
    assert failing.events[-1] == ("restore",)
    assert sum(event[0] == "restore" for event in failing.events) == 1


class FakeFullController(driver_v20a.RawNestedTierAttributionRuntimeMixinV20A):
    def __init__(self):
        self._v20a_panel_bundle = {"synthetic": True}
        self._v20a_token_boundary_audit_sha256 = "e" * 64
        self._v4_layer_plan_install = {
            "plan_sha256": "f" * 64,
            "unselected_origin_sha256": "1" * 64,
        }
        self.wave_indices = []

    def _prepared_fixed_batches_v20a(self):
        return {"synthetic": True}, {"fixed": True}

    def _raw_reference_probe_v20a(self, _prepared):
        return "2" * 64

    def _run_signed_wave_v20a(self, item, _prepared, unit_scores, commitments):
        self.wave_indices.append(item["signed_wave_index"])
        for values in unit_scores.values():
            values.fill(float(item["signed_wave_index"] + 1))
        commitments.extend(
            f"{item['signed_wave_index'] * 160 + index:064x}"
            for index in range(160)
        )
        return f"{item['signed_wave_index']:064x}"

    def _population_boundary_audit_v4(self, _iteration):
        install = self._v4_layer_plan_install
        value = {
            "engine_count": 4,
            "runtime_mapping": copy.deepcopy(install),
            "unselected_origin_sha256": install["unselected_origin_sha256"],
            "worker_reports": [
                {"rank": rank, "passed": True, **install} for rank in range(4)
            ],
            "passed": True,
        }
        value["audit_sha256"] = driver_v20a.canonical_sha256(value)
        return value


def test_raw_v20a_full_controller_captures_16x4x10_and_runs_5x12_gate(monkeypatch):
    observed_shapes = {}

    def compact_builder(unit_scores, panel_bundle):
        assert panel_bundle == {"synthetic": True}
        observed_shapes.update({arm: values.shape for arm, values in unit_scores.items()})
        return synthetic_compact()

    monkeypatch.setattr(
        driver_v20a.mechanics_v20a,
        "build_compact_estimator_summary_v20a",
        compact_builder,
    )
    controller = FakeFullController()
    summary, gate, audit = controller.estimate_raw_attribution_v20a(
        driver_v20a.prereg_v20a.PERTURBATION_SEEDS_V20A
    )
    assert controller.wave_indices == list(range(16))
    assert observed_shapes == {
        arm: (
            10, 2, 32,
            driver_v20a.frame_v20a.ARM_REQUESTS_PER_PANEL_V20A[arm],
        )
        for arm in driver_v20a.mechanics_v20a.ARMS_V20A
    }
    assert len(gate["contrasts"]) == 5
    assert all(
        result["observed_pass_count"] == result["bootstrap_pass_count"] == 12
        for result in gate["contrasts"].values()
    )
    assert summary["runtime_integrity"]["all_integrity_audits_passed"] is True
    assert audit["signed_wave_count"] == 16
    assert audit["dense_result_commitment_count"] == 2560
    assert audit["union_scoring_called_or_used"] is False


def make_report(implementation, recipe):
    summary = synthetic_summary()
    report = {
        "schema": "eggroll-es-authoritative-raw-attribution-report-v20a",
        "recipe": recipe,
        "configuration": synthetic_configuration(),
        "runtime_audit": synthetic_runtime_audit(),
        "summary": summary,
        "gate": driver_v20a.mechanics_v20a.evaluate_attribution_gate_v20a(summary),
        "implementation": implementation,
        "union_scoring_authorized_or_used": False,
        "model_update_applied": False,
        "checkpoint_written": False,
        "evaluation_surfaces_opened": False,
        "dataset_promotion_applied": False,
    }
    report["content_sha256_before_self_field"] = driver_v20a.canonical_sha256(report)
    return report


def reseal_report(report):
    audit = report.get("runtime_audit")
    if isinstance(audit, dict):
        audit["content_sha256_before_self_field"] = driver_v20a.canonical_sha256(
            driver_v20a._without_self(audit)
        )
    summary = report.get("summary")
    if isinstance(summary, dict):
        summary["content_sha256_before_self_field"] = driver_v20a.canonical_sha256(
            driver_v20a._without_self(summary)
        )
    report["content_sha256_before_self_field"] = driver_v20a.canonical_sha256(
        driver_v20a._without_self(report)
    )


def test_raw_v20a_compact_report_rejects_hidden_content_authority_and_union():
    _layer, _panels, implementation, recipe = runtime_inputs()
    report = make_report(implementation, recipe)
    assert driver_v20a.validate_compact_report_v20a(
        report, expected_recipe=recipe, expected_implementation=implementation
    ) == report
    serialized = json.dumps(report, sort_keys=True)
    for forbidden in (
        '"questions"', '"answers"', '"prompt_token_ids"', '"unit_scores"',
        '"responses"', '"bootstrap_replicates"', '"bootstrap_draws"',
    ):
        assert forbidden not in serialized

    hidden = copy.deepcopy(report)
    hidden["runtime_audit"]["prompt_token_ids"] = [1, 2]
    reseal_report(hidden)
    with pytest.raises(RuntimeError, match="compact report changed"):
        driver_v20a.validate_compact_report_v20a(
            hidden, expected_recipe=recipe, expected_implementation=implementation
        )

    union = copy.deepcopy(report)
    union["configuration"]["union_scoring_authorized"] = True
    reseal_report(union)
    with pytest.raises(RuntimeError, match="compact report changed"):
        driver_v20a.validate_compact_report_v20a(
            union, expected_recipe=recipe, expected_implementation=implementation
        )


def test_raw_v20a_fake_real_path_is_durable_and_cleanup_is_mandatory(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(driver_v20a, "FROZEN_OUTPUT_DIRECTORY_V20A", tmp_path)
    layer, panels, implementation, recipe = runtime_inputs()
    summary = synthetic_summary()
    gate = driver_v20a.mechanics_v20a.evaluate_attribution_gate_v20a(summary)

    class FakeTrainer:
        def configure_raw_attribution_v20a(self, panel_bundle, **_kwargs):
            assert panel_bundle["content_sha256_before_self_field"] == (
                driver_v20a.mechanics_v20a.PANEL_BUNDLE_CONTENT_SHA256_V20A
            )
            return synthetic_configuration()

        def estimate_raw_attribution_v20a(self, seeds):
            assert seeds == driver_v20a.prereg_v20a.PERTURBATION_SEEDS_V20A
            return summary, gate, synthetic_runtime_audit()

    monkeypatch.setattr(
        driver_v20a,
        "_source_provenance_v20a",
        lambda current: {
            "schema": "synthetic-committed-source-v20a",
            "implementation_bundle_sha256": current["bundle_sha256"],
            "content_sha256_before_self_field": "d" * 64,
        },
    )
    trainer = FakeTrainer()
    monkeypatch.setattr(driver_v20a, "_make_trainer_v20a", lambda _value: trainer)
    closed = []
    monkeypatch.setattr(driver_v20a.base, "close_trainer", closed.append)
    report = driver_v20a.run_exact_v20a(layer, panels, implementation, recipe)
    assert closed == [trainer]
    root = tmp_path / driver_v20a.EXPERIMENT_NAME_V20A
    attempt_path = next((root / "attempts").glob("*.json"))
    report_path = next((root / "runs").glob(f"*/{driver_v20a.REPORT_NAME_V20A}"))
    attempt = json.loads(attempt_path.read_text())
    assert attempt["status"] == "complete"
    assert attempt["report_binding"]["file_sha256"] == driver_v20a.file_sha256(
        report_path
    )
    assert json.loads(report_path.read_text()) == report


def test_raw_v20a_failure_hashes_error_and_writes_no_report(tmp_path, monkeypatch):
    monkeypatch.setattr(driver_v20a, "FROZEN_OUTPUT_DIRECTORY_V20A", tmp_path)
    layer, panels, implementation, recipe = runtime_inputs()

    class FailingTrainer:
        def configure_raw_attribution_v20a(self, *_args, **_kwargs):
            raise RuntimeError("synthetic sensitive raw failure")

    trainer = FailingTrainer()
    monkeypatch.setattr(
        driver_v20a,
        "_source_provenance_v20a",
        lambda current: {
            "schema": "synthetic-committed-source-v20a",
            "implementation_bundle_sha256": current["bundle_sha256"],
            "content_sha256_before_self_field": "e" * 64,
        },
    )
    monkeypatch.setattr(driver_v20a, "_make_trainer_v20a", lambda _value: trainer)
    closed = []
    monkeypatch.setattr(driver_v20a.base, "close_trainer", closed.append)
    with pytest.raises(RuntimeError, match="synthetic sensitive raw failure"):
        driver_v20a.run_exact_v20a(layer, panels, implementation, recipe)
    assert closed == [trainer]
    root = tmp_path / driver_v20a.EXPERIMENT_NAME_V20A
    attempt = json.loads(next((root / "attempts").glob("*.json")).read_text())
    assert attempt["status"] == "failed"
    assert attempt["failure_type"] == "RuntimeError"
    assert len(attempt["failure_sha256"]) == 64
    assert "failure_message" not in attempt
    assert "traceback" not in attempt
    assert "sensitive raw failure" not in json.dumps(attempt)
    assert not list((root / "runs").glob(f"*/{driver_v20a.REPORT_NAME_V20A}"))


def test_raw_v20a_union_update_and_evaluation_surfaces_are_closed():
    trainer = object.__new__(driver_v20a.RawNestedTierAttributionRuntimeMixinV20A)
    for method in (
        "configure_union_equivalence_v20a", "run_union_equivalence_v20a",
        "_generate_union_v20a", "_equivalence_state_v20a",
        "build_union_request_plan_v20a",
    ):
        with pytest.raises(RuntimeError, match="permanently disables union"):
            getattr(trainer, method)()
    for method in (
        "configure_disjoint_tier_attribution_v19a",
        "estimate_disjoint_tier_attribution_v19a",
        "build_compact_estimator_summary_v20a",
        "train_step", "apply_seed_coefficients", "fit", "eval_step",
        "evaluate_handle", "evaluate_population_on_batch",
    ):
        with pytest.raises(RuntimeError, match="closes attribution"):
            getattr(trainer, method)()
