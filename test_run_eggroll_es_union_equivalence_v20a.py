#!/usr/bin/env python3
"""CPU-only fail-closed tests for the V20A union equivalence runtime."""

import copy
import json

import numpy as np
import pytest

import run_eggroll_es_union_equivalence_v20a as driver_v20a


PLAN_SHA = driver_v20a.driver_v19a.driver_v13.driver_v10.MIDDLE_LATE_PLAN_SHA256_V10


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
        "--v20a-dry-run",
        *(extra or []),
    ]


def runtime_inputs():
    layer = load_layer_bundle()
    implementation = driver_v20a.implementation_identity_v20a()
    preregistration, panels = driver_v20a._load_bound_inputs_v20a(layer)
    recipe = driver_v20a.recipe_v20a(
        layer,
        preregistration,
        panels,
        implementation,
        driver_v20a._declared_moe_backend_v20a(),
    )
    return layer, panels, implementation, recipe


def state_summary(name="exact_reference"):
    return {
        "state": name,
        "raw_requests_per_engine": 1020,
        "unique_union_requests_per_engine": 450,
        "eliminated_duplicate_requests_per_engine": 570,
        "raw_to_unique_ratio": 1020 / 450,
        "union_audit_content_sha256": "1" * 64,
        "entry_count": 160,
        "raw_dense_commitments_sha256": "2" * 64,
        "union_dense_commitments_sha256": "2" * 64,
        "per_unit_score_bytes_sha256": "3" * 64,
        "all_per_arm_panel_scores_bit_exact": True,
        "all_dense_commitments_bit_exact": True,
        "scores_or_outputs_persisted": False,
    }


def synthetic_summary():
    value = {
        "schema": "eggroll-es-union-equivalence-summary-v20a",
        "experiment_name": driver_v20a.EXPERIMENT_NAME_V20A,
        "reference_equivalence": state_summary(),
        "perturbed_equivalence": state_summary("preregistered_perturbed_plus_wave_0"),
        "exact_reference_restored_after_perturbed_wave": True,
        "all_four_tp1_engines_both_states": True,
        "raw_arm_scoring_authoritative": True,
        "union_scoring_authorized_for_later_v20a_train_only_attribution": True,
        "v20a_attribution_run_authorized": False,
        "model_update_applied": False,
        "checkpoint_written": False,
        "evaluation_surfaces_opened": False,
        "dataset_promotion_applied": False,
        "scores_outputs_tokens_or_row_content_persisted": False,
    }
    value["content_sha256_before_self_field"] = driver_v20a.canonical_sha256(value)
    return value


def synthetic_configuration():
    return {
        "schema": "eggroll-es-union-equivalence-runtime-configuration-v20a",
        "layer_plan_install_sha256": "4" * 64,
        "reference_identity_sha256": "5" * 64,
        "unselected_origin_sha256": "6" * 64,
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
        "attribution_runtime_opened": False,
    }


def synthetic_runtime_audit():
    value = {
        "schema": "eggroll-es-union-equivalence-runtime-audit-v20a",
        "fixed_request_identity_sha256": "7" * 64,
        "token_boundary_audit_sha256": "8" * 64,
        "union_plan_content_sha256": "9" * 64,
        "restore_check_sha256": "a" * 64,
        "population_boundary_audit_sha256": "b" * 64,
        "unselected_origin_sha256": "6" * 64,
        "unselected_origin_audit_sha256": "c" * 64,
        "unselected_origin_audit_passed": True,
        "equivalence_state_count": 2,
        "engine_count": 4,
        "dense_comparisons_per_state": 160,
        "raw_requests_per_engine_per_state": 1020,
        "unique_union_requests_per_engine_per_state": 450,
        "tokens_scores_outputs_or_row_content_persisted": False,
    }
    value["content_sha256_before_self_field"] = driver_v20a.canonical_sha256(value)
    return value


def test_v20a_dry_run_binds_phases_recipe_and_opens_no_gpu(monkeypatch, capsys):
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
    assert first["attribution_run_opened"] is False
    recipe = first["recipe"]
    assert recipe["model"].endswith("Qwen3.6-35B-A3B")
    assert recipe["layers"] == [20, 21, 22, 23]
    assert recipe["alpha"] == 0.0
    assert recipe["raw_arm_scoring"]["requests_per_engine"] == 1020
    assert recipe["equivalence_signed_wave"] == driver_v20a.equivalence_signed_wave_v20a()
    assert recipe["hardware"] == {
        "engine_count": 4,
        "tp_per_engine": 1,
        "gpu_ids": [0, 1, 2, 3],
        "all_four_engines_both_equivalence_states": True,
    }
    assert recipe["moe_backend"] == driver_v20a._declared_moe_backend_v20a()
    assert recipe["authority"]["may_launch_v20a_attribution"] is False
    assert recipe["authority"]["model_update_allowed"] is False
    assert first["implementation"]["files"]["trainer_mechanics_v20a"][
        "file_sha256"
    ] == "52774f35de92421772c86ee53d79d2f8b9db7e21e8d4690cdf3fd1163eabee34"


@pytest.mark.parametrize("name", driver_v20a.MOE_OVERRIDE_ENVIRONMENT_V20A)
def test_v20a_rejects_all_moe_overrides_before_assembly(name, monkeypatch):
    monkeypatch.setenv(name, "synthetic-confound")
    with pytest.raises(ValueError, match="every MoE backend override unset"):
        driver_v20a.main(cli())


def test_v20a_rejects_hash_tampering_forbidden_and_unknown_controls():
    layer, _panels, implementation, recipe = runtime_inputs()
    args = driver_v20a._parser_v20a().parse_args(["--v20a-dry-run"])
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


def test_v20a_dense_comparison_is_bit_exact_and_rejects_one_bit_drift():
    raw = {
        (engine, arm, panel): (
            np.asarray([engine + 0.25, engine + 0.5], dtype=np.float64),
            f"{engine + 1:064x}",
        )
        for engine in range(4)
        for arm in driver_v20a.mechanics_v20a.ARMS_V20A
        for panel in driver_v20a.mechanics_v20a.PANEL_NAMES_V20A
    }
    union = copy.deepcopy(raw)
    audit = driver_v20a.compare_dense_equivalence_v20a(raw, union)
    assert audit["entry_count"] == 160
    assert audit["all_per_arm_panel_scores_bit_exact"] is True
    assert audit["all_dense_commitments_bit_exact"] is True
    assert audit["scores_or_outputs_persisted"] is False
    key = next(iter(union))
    union[key][0].view(np.uint64)[0] ^= np.uint64(1)
    with pytest.raises(RuntimeError, match="not bit-exact"):
        driver_v20a.compare_dense_equivalence_v20a(raw, union)


def _prepared_1020():
    prepared = {}
    for arm in driver_v20a.mechanics_v20a.ARMS_V20A:
        count = driver_v20a.frame_v20a.ARM_REQUESTS_PER_PANEL_V20A[arm]
        panels = {}
        flat = []
        cursor = 0
        for panel_index, panel in enumerate(driver_v20a.mechanics_v20a.PANEL_NAMES_V20A):
            items = []
            for index in range(count):
                # Base positions share identities across arms; nested extras are distinct.
                token_value = panel_index * 100 + (index if index < 24 else 1000 + index)
                tokens = [1, token_value, 2]
                items.append({
                    "prompt_token_ids": tokens,
                    "prompt_token_ids_sha256": driver_v20a.canonical_sha256(tokens),
                })
            panels[panel] = {
                "dense_items": items,
                "slice": (cursor, cursor + len(items)),
            }
            flat.extend(items)
            cursor += len(items)
        prepared[arm] = {"panels": panels, "prompt_items": flat}
    return prepared


class FakeEquivalenceController(driver_v20a.UnionScoringEquivalenceRuntimeMixinV20A):
    def __init__(self, fail_perturbed=False):
        self.events = []
        self.fail_perturbed = fail_perturbed
        self._v4_layer_plan_install = {
            "plan_sha256": "1" * 64,
            "unselected_origin_sha256": "2" * 64,
        }

    def _prepared_fixed_batches_v20a(self):
        self._v20a_token_boundary_audit_sha256 = "3" * 64
        return _prepared_1020(), {arm: f"{i + 4:064x}" for i, arm in enumerate(
            driver_v20a.mechanics_v20a.ARMS_V20A
        )}

    def _equivalence_state_v20a(self, state, *_args):
        self.events.append(("equivalence", state))
        if self.fail_perturbed and state.startswith("preregistered"):
            raise RuntimeError("synthetic perturbed mismatch")
        return state_summary(state)

    def _perturb_signed_wave_v19a(self, seeds, negate):
        self.events.append(("perturb", list(seeds), negate))

    def _restore_and_verify_signed_wave_v19a(self):
        self.events.append(("restore",))
        return "a" * 64

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


def test_v20a_equivalence_controller_uses_two_states_and_restores_on_failure():
    controller = FakeEquivalenceController()
    summary, gate, audit = controller.run_union_equivalence_v20a()
    assert [event[0] for event in controller.events] == [
        "equivalence", "perturb", "equivalence", "restore",
    ]
    assert summary["exact_reference_restored_after_perturbed_wave"] is True
    assert gate[
        "union_scoring_authorized_for_later_v20a_train_only_attribution"
    ] is True
    assert gate["v20a_attribution_run_authorized"] is False
    assert audit["equivalence_state_count"] == 2
    assert audit["raw_requests_per_engine_per_state"] == 1020

    failing = FakeEquivalenceController(fail_perturbed=True)
    with pytest.raises(RuntimeError, match="synthetic perturbed mismatch"):
        failing.run_union_equivalence_v20a()
    assert failing.events[-1] == ("restore",)
    assert sum(event[0] == "restore" for event in failing.events) == 1


def test_v20a_compact_report_rejects_hidden_content_and_authority():
    _layer, _panels, implementation, recipe = runtime_inputs()
    summary = synthetic_summary()
    report = {
        "schema": "eggroll-es-union-equivalence-report-v20a",
        "recipe": recipe,
        "configuration": synthetic_configuration(),
        "runtime_audit": synthetic_runtime_audit(),
        "summary": summary,
        "gate": driver_v20a.evaluate_equivalence_gate_v20a(summary),
        "implementation": implementation,
        "model_update_applied": False,
        "checkpoint_written": False,
        "evaluation_surfaces_opened": False,
        "dataset_promotion_applied": False,
        "attribution_run_opened": False,
    }
    report["content_sha256_before_self_field"] = driver_v20a.canonical_sha256(report)
    assert driver_v20a.validate_compact_report_v20a(
        report, expected_recipe=recipe, expected_implementation=implementation
    ) == report
    serialized = json.dumps(report, sort_keys=True)
    for forbidden in (
        '"questions"', '"answers"', '"prompt_token_ids"', '"unit_scores"',
        '"responses"', '"union_prompt_items"', '"arm_panel_union_indices"',
    ):
        assert forbidden not in serialized
    hidden = copy.deepcopy(report)
    hidden["runtime_audit"]["prompt_token_ids"] = [1, 2]
    hidden["runtime_audit"]["content_sha256_before_self_field"] = (
        driver_v20a.canonical_sha256({
            key: value for key, value in hidden["runtime_audit"].items()
            if key != "content_sha256_before_self_field"
        })
    )
    hidden["content_sha256_before_self_field"] = driver_v20a.canonical_sha256({
        key: value for key, value in hidden.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="compact equivalence report changed"):
        driver_v20a.validate_compact_report_v20a(
            hidden, expected_recipe=recipe, expected_implementation=implementation
        )


def test_v20a_fake_real_path_is_durable_and_cleanup_is_mandatory(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(driver_v20a, "FROZEN_OUTPUT_DIRECTORY_V20A", tmp_path)
    layer, panels, implementation, recipe = runtime_inputs()
    summary = synthetic_summary()
    gate = driver_v20a.evaluate_equivalence_gate_v20a(summary)

    class FakeTrainer:
        def configure_union_equivalence_v20a(self, panel_bundle, **_kwargs):
            assert panel_bundle["content_sha256_before_self_field"] == (
                driver_v20a.mechanics_v20a.PANEL_BUNDLE_CONTENT_SHA256_V20A
            )
            return synthetic_configuration()

        def run_union_equivalence_v20a(self):
            return summary, gate, synthetic_runtime_audit()

    monkeypatch.setattr(
        driver_v20a,
        "_source_provenance_v20a",
        lambda current: {
            "schema": "synthetic-committed-source-v20a",
            "implementation_bundle_sha256": current["bundle_sha256"],
            "content_sha256_before_self_field": "f" * 64,
        },
    )
    monkeypatch.setattr(driver_v20a, "_make_trainer_v20a", lambda _value: FakeTrainer())
    closed = []
    monkeypatch.setattr(
        driver_v20a.base,
        "close_trainer",
        lambda trainer: closed.append(trainer),
    )
    report = driver_v20a.run_exact_v20a(layer, panels, implementation, recipe)
    assert len(closed) == 1
    root = tmp_path / driver_v20a.EXPERIMENT_NAME_V20A
    attempt_path = next((root / "attempts").glob("*.json"))
    report_path = next((root / "runs").glob(f"*/{driver_v20a.REPORT_NAME_V20A}"))
    attempt = json.loads(attempt_path.read_text())
    assert attempt["status"] == "complete"
    assert attempt["report_binding"]["file_sha256"] == driver_v20a.file_sha256(
        report_path
    )
    assert json.loads(report_path.read_text()) == report


def test_v20a_failure_attempt_hashes_error_and_writes_no_report(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(driver_v20a, "FROZEN_OUTPUT_DIRECTORY_V20A", tmp_path)
    layer, panels, implementation, recipe = runtime_inputs()

    class FailingTrainer:
        def configure_union_equivalence_v20a(self, *_args, **_kwargs):
            raise RuntimeError("synthetic sensitive equivalence failure")

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
    monkeypatch.setattr(driver_v20a.base, "close_trainer", lambda value: closed.append(value))
    with pytest.raises(RuntimeError, match="synthetic sensitive equivalence failure"):
        driver_v20a.run_exact_v20a(layer, panels, implementation, recipe)
    assert closed == [trainer]
    root = tmp_path / driver_v20a.EXPERIMENT_NAME_V20A
    attempt_path = next((root / "attempts").glob("*.json"))
    attempt = json.loads(attempt_path.read_text())
    assert attempt["status"] == "failed"
    assert attempt["failure_type"] == "RuntimeError"
    assert len(attempt["failure_sha256"]) == 64
    assert "failure_message" not in attempt
    assert "traceback" not in attempt
    assert "sensitive equivalence failure" not in json.dumps(attempt)
    assert not list((root / "runs").glob(f"*/{driver_v20a.REPORT_NAME_V20A}"))


def test_v20a_mutation_attribution_and_evaluation_surfaces_are_closed():
    trainer = object.__new__(driver_v20a.UnionScoringEquivalenceRuntimeMixinV20A)
    for method in (
        "configure_disjoint_tier_attribution_v19a",
        "estimate_disjoint_tier_attribution_v19a",
        "build_compact_estimator_summary_v20a",
        "train_step", "apply_seed_coefficients", "fit", "eval_step",
        "evaluate_handle", "evaluate_population_on_batch",
    ):
        with pytest.raises(RuntimeError, match="closes attribution"):
            getattr(trainer, method)()
