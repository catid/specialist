import copy
import inspect
import json

import numpy as np
import pytest

import eggroll_es_insertion_stability_preregistration_v23a as prereg_v23a
import run_eggroll_es_insertion_stability_v23a as runtime_v23a


def _prereg():
    return runtime_v23a._load_preregistration_v23a()


def test_v23a_recipe_is_exact_four_arm_train_only_contract():
    prereg = _prereg()
    implementation = {"bundle_sha256": "a" * 64}
    recipe = runtime_v23a.recipe_v23a(prereg, implementation)
    assert list(recipe["arms"]) == list(prereg_v23a.ARM_ORDER_V23A)
    assert [recipe["arms"][arm]["engine_rank"] for arm in prereg_v23a.ARM_ORDER_V23A] == [0, 1, 2, 3]
    assert len({recipe["arms"][arm]["model_path"] for arm in prereg_v23a.ARM_ORDER_V23A}) == 4
    assert recipe["runtime"]["requests_all_engines_all_signed_waves"] == 71_680
    assert recipe["post_restore_probe_requests_all_engines"] == 4
    assert recipe["authority"] == {
        "train_only_raw_scoring": True,
        "model_update_allowed": False,
        "checkpoint_write_allowed": False,
        "evaluation_allowed": False,
        "dataset_promotion_allowed": False,
    }
    assert recipe["content_sha256_before_self_field"] == runtime_v23a.canonical_sha256(
        runtime_v23a._without_self_v23a(recipe)
    )


def test_v23a_device_identity_requires_exact_distinct_physical_mapping():
    reports = [{
        "schema": "eggroll-es-runtime-device-identity-v23a",
        "rank": rank, "world_size": 4, "arm": arm,
        "cuda_visible_devices": str(rank), "runtime_cuda_device": 0,
        "device_name": "GPU", "device_total_memory": 80_000_000_000,
        "update_surfaces_closed": True,
    } for rank, arm in enumerate(prereg_v23a.ARM_ORDER_V23A)]
    result = runtime_v23a._validate_device_identities_v23a(reports)
    assert result["gpu_ids"] == [0, 1, 2, 3]
    duplicate = copy.deepcopy(reports)
    duplicate[3]["cuda_visible_devices"] = "2"
    with pytest.raises(RuntimeError, match="physical-GPU"):
        runtime_v23a._validate_device_identities_v23a(duplicate)


def test_v23a_install_validator_is_arm_specific_and_exact():
    prereg = _prereg()
    for rank, arm in enumerate(prereg_v23a.ARM_ORDER_V23A):
        binding = runtime_v23a._expected_plan_binding_v23a(arm, prereg)
        frozen = runtime_v23a.worker_v23a.FROZEN_LAYER_PLANS_V23A[
            binding["layer_plan_sha256"]
        ]
        report = {
            "schema": "eggroll-es-layer-plan-installed-v4", "installed": True,
            "idempotent": False, "rank": rank, "world_size": 4,
            "plan": frozen["plan"], "selected_byte_count": frozen["selected_byte_count"],
            "selected_parameter_manifest_sha256": "b" * 64,
            "unselected_origin_sha256": "c" * 64, **binding,
        }
        compact = runtime_v23a._validate_install_v23a(report, arm, rank, prereg)
        assert compact["selected_element_count"] == 142_999_552
        wrong = dict(report, layer_plan_sha256="0" * 64)
        with pytest.raises(RuntimeError, match="installation changed"):
            runtime_v23a._validate_install_v23a(wrong, arm, rank, prereg)


def test_v23a_cross_arm_logical_shape_order_is_reproved_at_runtime():
    prereg = _prereg()
    digest = runtime_v23a._validate_cross_arm_logical_contract_v23a(prereg)
    assert digest == prereg["cross_arm_perturbation_contract"]["logical_shape_order_sha256"]
    wrong = copy.deepcopy(prereg)
    wrong["arms"]["insert_back_e005"]["layer_plan"]["logical_shape_order_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="logical perturbation geometry"):
        runtime_v23a._validate_cross_arm_logical_contract_v23a(wrong)


def test_v23a_perturbation_completion_requires_four_tp1_acknowledgements():
    runtime_v23a._validate_perturbation_results_v23a([[None]] * 4)
    with pytest.raises(RuntimeError, match="all four"):
        runtime_v23a._validate_perturbation_results_v23a([[None]] * 3)
    with pytest.raises(RuntimeError, match="all four"):
        runtime_v23a._validate_perturbation_results_v23a([[None], [None], [], [None]])


def test_v23a_runtime_closes_every_inherited_action_surface():
    closed = runtime_v23a.InsertionLocationRuntimeMixinV23A._closed_surface_v23a
    for name in (
        "configure_anchor", "configure_train_panels_v13", "estimate_train_panels_v13",
        "estimate_step_coefficients", "apply_seed_coefficients", "train_step",
        "evaluate_handle", "evaluate_population_on_batch", "eval_step", "fit",
    ):
        assert getattr(runtime_v23a.InsertionLocationRuntimeMixinV23A, name) is closed
    with pytest.raises(RuntimeError, match="closes all"):
        closed()


def test_v23a_argv_and_compact_persistence_fail_closed():
    for token in ("--checkpoint=x", "--validation-path=x", "--union-mode"):
        with pytest.raises(ValueError, match="forbidden runtime surface"):
            runtime_v23a._assert_train_only_argv_v23a([token])
    runtime_v23a._assert_compact_persistence_v23a({"hash": "a" * 64})
    with pytest.raises(RuntimeError, match="forbidden keys"):
        runtime_v23a._assert_compact_persistence_v23a({"unit_scores": []})


class _Remote:
    def remote(self, *_args, **_kwargs):
        return [None]


class _Engine:
    collective_rpc = _Remote()


class _SyntheticRuntime(runtime_v23a.InsertionLocationRuntimeMixinV23A):
    def __init__(self):
        self.engines = [_Engine() for _ in range(4)]
        self.generate_calls = []
        self.score_calls = 0
        self._v23a_panel_bundle = object()

    @staticmethod
    def _resolve(values):
        return values

    def _prepared_train_requests_v23a(self):
        first = prereg_v23a.anchor_v13.PANEL_NAMES_V13[0]
        panels = {first: {"dense_items": [object()]}}
        return panels, list(range(280)), "1" * 64

    def _generate_all_v23a(self, requests):
        self.generate_calls.append(len(requests))
        return [[object() for _ in requests] for _ in range(4)]

    def _score_batches_v23a(self, _panels, _batches):
        self.score_calls += 1
        values = {
            arm: np.full((5, 56), self.score_calls + rank, dtype=np.float64)
            for rank, arm in enumerate(prereg_v23a.ARM_ORDER_V23A)
        }
        commitments = {arm: [f"{arm}-{self.score_calls}-{index}" for index in range(5)]
                       for arm in prereg_v23a.ARM_ORDER_V23A}
        probes = {arm: runtime_v23a.canonical_sha256({"probe": 1})
                  for arm in prereg_v23a.ARM_ORDER_V23A}
        return values, commitments, probes

    @staticmethod
    def _restore_verify_v23a():
        return "2" * 64

    @staticmethod
    def _boundary_audit_v23a():
        return {"passed": True, "reports_sha256": "3" * 64}


def test_v23a_synthetic_runtime_executes_all_64_four_gpu_waves(monkeypatch):
    fake = _SyntheticRuntime()
    compact = {
        "schema": "synthetic", "comparisons": {},
        "content_sha256_before_self_field": "old",
    }
    monkeypatch.setattr(runtime_v23a.mechanics_v23a, "build_compact_estimator_summary_v23a",
                        lambda *_args: copy.deepcopy(compact))
    monkeypatch.setattr(runtime_v23a.mechanics_v23a, "evaluate_gate_v23a",
                        lambda _summary: {"direct_model_update_authorized": False})
    monkeypatch.setattr(runtime_v23a.anchor_v4, "score_gold_answer_outputs_v4",
                        lambda *_args: {"probe": 1})
    estimator, gate, audit = fake.estimate_insertion_stability_v23a()
    assert fake.generate_calls == [280] * 65 + [1]
    assert fake.score_calls == 65
    assert audit["signed_wave_count"] == 64
    assert audit["requests_all_engines_all_signed_waves"] == 71_680
    assert audit["dense_result_commitment_count"] == 1_280
    assert audit["post_restore_probe_requests_all_engines"] == 4
    assert gate["direct_model_update_authorized"] is False
    assert all(
        integrity["all_integrity_audits_passed"] is True
        for integrity in estimator["runtime_integrity"].values()
    )


def test_v23a_heterogeneous_launcher_keeps_model_choice_inside_arm_loop():
    source = inspect.getsource(runtime_v23a.load_runtime_trainer_v23a)
    assert "for rank, arm in enumerate(prereg_v23a.ARM_ORDER_V23A)" in source
    assert '"model": preregistration["arms"][arm]["model_path"]' in source
    assert '"tensor_parallel_size": 1' in source
    assert '"worker_extension_cls": WORKER_EXTENSION_V23A' in source
    assert '"moe_backend": "triton"' in source


def test_v23a_dry_run_is_gpu_closed_and_hash_bound(capsys):
    value = runtime_v23a.main(["--v23a-dry-run"])
    captured = json.loads(capsys.readouterr().out)
    assert value == captured
    assert value["gpu_launched"] is False
    assert value["real_launch_requires_committed_bundle_and_recipe_hashes"] is True
    assert value["recipe"]["runtime"]["engine_arm_mapping"] == prereg_v23a.ENGINE_ARM_MAPPING_V23A
    assert value["content_sha256_before_self_field"] == runtime_v23a.canonical_sha256(
        runtime_v23a._without_self_v23a(value)
    )
