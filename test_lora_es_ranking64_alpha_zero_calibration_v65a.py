#!/usr/bin/env python3
"""CPU-only numeric and preregistration tests for V65A."""

from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest

import build_lora_es_ranking64_alpha_zero_preregistration_v65a as builder
import lora_es_ranking64_alpha_zero_calibration_v65a as subject


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _metric(request_index: int, f1: float = 0.5, exact: int = 0) -> dict:
    return {
        "request_index": request_index,
        "row_sha256": _sha(f"row-{request_index}"),
        "unit_identity_sha256": _sha(f"unit-{request_index}"),
        "f1": float(f1),
        "exact": int(exact),
        "nonzero": int(float(f1) > 0.0),
    }


def _scored(f1: float = 0.5) -> list:
    return [[[
        _metric(request_index, f1)
        for request_index in range(64)
    ] for _actor in range(4)] for _period in range(4)]


def _master_receipt() -> dict:
    return {
        "schema": "synthetic-exact-v434-master-receipt-v65a",
        "canonical_fp32_master_sha256": (
            subject.population65.design52.MASTER_SHA256_V52
        ),
        "bf16_runtime_values_sha256": (
            subject.population65.design52.MASTER_RUNTIME_SHA256_V52
        ),
        "four_actor_consensus": True,
    }


def _receipts(kind: str) -> list[dict]:
    master = _master_receipt()
    return [{
        "period_kind": kind,
        "period_index": index,
        "before": copy.deepcopy(master),
        "after": copy.deepcopy(master),
        "identical_v434_state": True,
    } for index in range(4)]


@lru_cache(maxsize=1)
def _real_installations_v65a() -> list[dict]:
    path = builder.ROOT / (
        "experiments/eggroll_es_hpo/runs/"
        "v61a_v434_train_only_baseline_census/baseline_census_report_v61a.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    installations = value.get("installations")
    if not isinstance(installations, list) or len(installations) != 4:
        raise RuntimeError("test V61A installation fixture changed")
    return installations


def _evidence_inputs_v65a(runtime) -> dict:
    design52 = subject.population65.design52
    real_installations = _real_installations_v65a()
    master_identity = copy.deepcopy(real_installations[0]["canonical_identity"])
    identity_sha = subject.canonical_sha256_v65a(master_identity)
    installed = {
        "canonical_fp32_master_sha256": design52.MASTER_SHA256_V52,
        "canonical_master_identity_sha256": identity_sha,
        "four_actor_certificate_sha256": _sha("certificates"),
        "bf16_runtime_values_sha256": design52.MASTER_RUNTIME_SHA256_V52,
    }
    aggregate = {
        "schema": "v65a-read-only-four-actor-master-slot-consensus",
        "canonical_fp32_master_sha256": design52.MASTER_SHA256_V52,
        "canonical_master_identity_sha256": identity_sha,
        "bf16_runtime_values_sha256": design52.MASTER_RUNTIME_SHA256_V52,
        "runtime_view_count_per_actor": 82,
        "runtime_elements_per_actor": 4_921_344,
        "runtime_dtype": "torch.bfloat16",
        "base_inventory_sha256": subject.BASE_INVENTORY_SHA256_V65A,
        "four_actor_exact_read_only_consensus": True,
    }
    timing = {
        "clock": "worker_monotonic_ns", "started_ns": 10,
        "ended_ns": 12, "elapsed_ns": 2,
    }
    write_receipts = []
    read_receipts = []
    for kind in ("unscored_warmup", "scored"):
        for index in range(4):
            write_actors = [{
                "schema": "exact-master-slot-write-v65a",
                "period_kind": kind,
                "period_index": index,
                "master_identity": copy.deepcopy(master_identity),
                "materialization": {
                    **copy.deepcopy(
                        real_installations[actor]["materialization"]
                    ),
                    "phase": "v65a_exact_master_slot_write",
                },
                "base_identity": {
                    **copy.deepcopy(real_installations[actor]["base_identity"]),
                    "phase": "v65a_exact_master_slot_write",
                },
                "transaction_state_quiescent": True,
                "timing": copy.deepcopy(timing),
            } for actor in range(4)]
            write_receipts.append({
                "period_kind": kind,
                "period_index": index,
                "pre_write_master": copy.deepcopy(installed),
                "post_write_master": copy.deepcopy(installed),
                "actors": write_actors,
                "actor_receipts_sha256": subject.canonical_sha256_v65a(
                    write_actors
                ),
            })
            for edge in ("before_generation", "after_generation"):
                read_actors = [{
                    "schema": "read-only-exact-master-slot-v65a",
                    "period_kind": kind,
                    "period_index": index,
                    "edge": edge,
                    "master_identity": copy.deepcopy(master_identity),
                    "runtime_view_count": 82,
                    "runtime_elements": 4_921_344,
                    "runtime_dtype": "torch.bfloat16",
                    "runtime_values_sha256": design52.MASTER_RUNTIME_SHA256_V52,
                    "active_lora_ids": [1],
                    "active_manager_cache_lora_ids": [1],
                    "base_identity": {
                        **copy.deepcopy(
                            real_installations[actor]["base_identity"]
                        ),
                        "phase": "v65a_read_only_slot_receipt",
                    },
                    "transaction_state_quiescent": True,
                    "slot_read_only_no_weight_write_or_reset": True,
                    "timing": copy.deepcopy(timing),
                } for actor in range(4)]
                read_receipts.append({
                    "period_kind": kind,
                    "period_index": index,
                    "edge": edge,
                    "aggregate": copy.deepcopy(aggregate),
                    "actors": read_actors,
                    "actor_receipts_sha256": subject.canonical_sha256_v65a(
                        read_actors
                    ),
                })
    active = [{
        "schema": "v65a-effective-active-lora-receipt",
        "expected_lora_int_id": 1,
        "active_lora_ids": [1],
        "active_manager_cache_lora_ids": [1],
        "loaded_cpu_cache_lora_ids": [1],
        "active_slot_index": 0,
        "max_loras": 1,
        "max_cpu_loras": 2,
        "max_lora_rank": 32,
        "extra_or_candidate_adapter_loaded": False,
        "staged_v434_applied_receipt": {
            "schema": "v64-effective-applied-lora-receipt",
            "expected_lora_int_id": 1,
            "active_lora_ids": [1],
            "active_manager_cache_lora_ids": [1],
            "loaded_cpu_cache_lora_ids": [1],
            "active_slot_index": 0,
            "facade_type": "LRUCacheWorkerLoRAManager",
            "manager_type": "LRUCacheLoRAModelManager",
            "staged_weights_file_sha256": design52.STAGED_WEIGHTS_SHA256_V52,
            "canonical_fp32_state_sha256": design52.MASTER_SHA256_V52,
            "canonical_ordered_key_sha256": (
                design52.MASTER_ORDERED_KEY_SHA256_V52
            ),
            "canonical_tensor_count": 70,
            "canonical_elements": 4_528_128,
            "registered_lora_module_count": 23,
            "matched_live_lora_module_count": 23,
            "unmatched_registered_lora_module_count": 0,
            "runtime_module_manifest_sha256": (
                subject.RUNTIME_MODULE_MANIFEST_SHA256_V65A
            ),
            "source_linked_runtime_view_count": 82,
            "source_linked_runtime_elements": 4_921_344,
            "source_linked_runtime_dtype": "torch.bfloat16",
            "source_linked_runtime_values_sha256": (
                design52.MASTER_RUNTIME_SHA256_V52
            ),
            "registered_slot_view_count": 82,
            "registered_slot_records_sha256": (
                subject.REGISTERED_SLOT_RECORDS_SHA256_V65A
            ),
            "exact_staged_fp32_to_gpu_slot_equality": True,
            "exact_registered_postpack_to_gpu_slot_equality": True,
            "active_matches_expected": True,
            "max_loras": 1,
            "max_cpu_loras": 2,
        },
    } for _actor in range(4)]
    installations = copy.deepcopy(real_installations)
    state = lambda kind: [{
        "period_kind": kind,
        "period_index": index,
        "before": copy.deepcopy(aggregate),
        "after": copy.deepcopy(aggregate),
        "identical_v434_state": True,
    } for index in range(4)]
    return {
        "panel": {"content_sha256_before_self_field": _sha("panel")},
        "input_receipt": {},
        "actor_identities": [{"actor": index} for index in range(4)],
        "worker_identities": [{"worker": index} for index in range(4)],
        "active_lora_receipts": active,
        "installations": installations,
        "installed_master": installed,
        "warmup_state_receipts": state("unscored_warmup"),
        "scored_state_receipts": state("scored"),
        "slot_write_receipts": write_receipts,
        "read_only_slot_receipts": read_receipts,
        "final_master_state": copy.deepcopy(installed),
        "scored_periods": _scored(),
    }


def _implementation_entries() -> dict:
    return {
        "numeric_analysis_v65a": builder.ROOT / (
            "lora_es_ranking64_alpha_zero_calibration_v65a.py"
        ),
        "preregistration_builder_v65a": builder.ROOT / (
            "build_lora_es_ranking64_alpha_zero_preregistration_v65a.py"
        ),
        "hash_only_panel_design_v65": builder.ROOT / (
            "lora_es_robust_sampling_population_v65.py"
        ),
        "base_model_byte_receipt_runtime_v64": builder.ROOT / (
            "run_lora_es_v59_vs_v434_robust_confirmation_v64.py"
        ),
    }


def test_v65a_schedule_counts_and_exact_engine_controls():
    assert subject.ROWS_V65A == 64
    assert subject.WARMUP_PERIODS_V65A == subject.SCORED_PERIODS_V65A == 4
    assert subject.PAIR_PERIODS_V65A == ((0, 1), (2, 3))
    assert subject.REPLICAS_PER_UNIT_V65A == 8
    assert subject.WARMUP_GENERATION_COMPLETIONS_V65A == 1_024
    assert subject.SCORED_GENERATION_COMPLETIONS_V65A == 1_024
    assert subject.TOTAL_GENERATION_COMPLETIONS_V65A == 2_048
    assert subject.LABEL_PLAN_V65A == {
        str(actor): ["candidate", "reference", "reference", "candidate"]
        for actor in range(4)
    }
    assert subject.ENGINE_CONTROLS_V65A == {
        "tensor_parallel_size": 1,
        "dtype": "torch.bfloat16",
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.82,
        "max_loras": 1,
        "max_cpu_loras": 2,
        "max_lora_rank": 32,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "enforce_eager": True,
        "async_scheduling": False,
        "max_num_seqs": 64,
        "max_num_batched_tokens": 8192,
        "scheduling_policy": "fcfs",
        "VLLM_BATCH_INVARIANT": False,
    }


def test_v65a_runtime_uses_explicit_v434_request_and_read_only_edge_hashes():
    import eggroll_es_worker_lora_v65a as worker65a
    import run_lora_es_ranking64_alpha_zero_calibration_v65a as runtime65a

    request_source = inspect.getsource(runtime65a.lora_request_v65a)
    execute_source = inspect.getsource(runtime65a.execute_v65a)
    read_only_source = inspect.getsource(
        worker65a.LoRAAdapterStateWorkerExtensionV65A.
        read_only_exact_master_slot_v65a
    )
    assert "design52.STAGED_V52" in request_source
    assert "v434_ranking64_alpha_zero_v65a" in request_source
    assert "prior._lora_request()" not in execute_source
    assert "verify_adapter_contract_v52" in execute_source
    assert "_materialize_v41a" not in read_only_source
    assert "copy_(" not in read_only_source
    assert "slot_read_only_no_weight_write_or_reset" in read_only_source
    assert "construction_state" in execute_source
    assert 'construction_state.get("trainer")' in execute_source
    assert execute_source.index("post_generation_integrity") < (
        execute_source.index("numeric_reduction")
    )


def test_v65a_build_evidence_accepts_exact_order_and_rejects_receipt_reorder():
    import run_lora_es_ranking64_alpha_zero_calibration_v65a as runtime65a

    values = _evidence_inputs_v65a(runtime65a)
    evidence = runtime65a.build_evidence_v65a(**values)
    assert evidence["exact_master_slot_write_receipts"] == (
        values["slot_write_receipts"]
    )
    assert evidence["read_only_live_slot_receipts"] == (
        values["read_only_slot_receipts"]
    )
    changed = copy.deepcopy(values)
    changed["slot_write_receipts"][0], changed["slot_write_receipts"][1] = (
        changed["slot_write_receipts"][1],
        changed["slot_write_receipts"][0],
    )
    with pytest.raises(RuntimeError):
        runtime65a.build_evidence_v65a(**changed)
    changed = copy.deepcopy(values)
    changed["read_only_slot_receipts"][1]["edge"] = "before_generation"
    with pytest.raises(RuntimeError):
        runtime65a.build_evidence_v65a(**changed)


def test_v65a_constructor_failure_cleanup_always_shutdowns_and_proves_idle():
    import run_lora_es_ranking64_alpha_zero_calibration_v65a as runtime65a

    events = []

    class Cleanup:
        def strict_close_trainer_v38a(self, _trainer):
            events.append("strict")
            raise RuntimeError("synthetic partial constructor")

        def wait_for_gpu_idle(self):
            events.append("idle")
            return {"all_four_compute_process_lists_empty": True}

    class Ray:
        @staticmethod
        def shutdown():
            events.append("shutdown")

    class Pool:
        def terminate(self):
            events.append("pool_terminate")

        def join(self):
            events.append("pool_join")

    partial = SimpleNamespace(mp_pool=Pool())
    cleanup, idle, pool, errors = runtime65a._failure_cleanup_v65a(
        SimpleNamespace(cleanup_v38a=Cleanup()), partial, Ray,
    )
    assert cleanup is None
    assert idle == {"all_four_compute_process_lists_empty": True}
    assert pool == {
        "attempted_after_incomplete_strict_cleanup": True,
        "pool_found": True,
        "terminate_succeeded": True,
        "join_succeeded": True,
    }
    assert len(errors) == 1
    assert events == [
        "strict", "pool_terminate", "pool_join", "shutdown", "idle",
    ]

    events.clear()
    cleanup, idle, pool, errors = runtime65a._failure_cleanup_v65a(
        SimpleNamespace(cleanup_v38a=Cleanup()), None, Ray,
    )
    assert cleanup is None
    assert idle == {"all_four_compute_process_lists_empty": True}
    assert pool["pool_found"] is False
    assert errors == []
    assert events == ["shutdown", "idle"]


def test_v65a_scored_validator_checks_identity_order_and_returns_unit_first():
    scored = _scored()
    array = subject.validate_scored_periods_v65a(scored)
    assert array.shape == (64, 4, 4, 3)
    assert np.all(array[..., 0] == 0.5)

    changed = copy.deepcopy(scored)
    changed[3][2][10]["request_index"] = 9
    with pytest.raises(ValueError):
        subject.validate_scored_periods_v65a(changed)
    changed = copy.deepcopy(scored)
    changed[1][1][4]["row_sha256"] = _sha("different")
    with pytest.raises(ValueError):
        subject.validate_scored_periods_v65a(changed)
    changed = copy.deepcopy(scored)
    changed[0][0][0]["exact"] = 1
    with pytest.raises(ValueError):
        subject.validate_scored_periods_v65a(changed)


def test_v65a_identical_state_analysis_passes_zero_null_without_exact_gate():
    result = subject.analyze_scored_periods_v65a(_scored())
    primary = result["primary_cluster_bootstrap"]
    assert primary["paired_replicas_per_unit_preserved_and_averaged"] == 8
    assert primary["resampled_axis"] == "conflict_unit_only"
    for name in (
        "generated_f1_delta", "generated_nonzero_delta",
        "stability_improvement", "joint_composite",
    ):
        interval = primary["intervals"][name]
        assert interval == {
            "point": 0.0, "lcb": 0.0, "ucb": 0.0,
            "halfwidth": 0.0, "contains_zero": True, "null_radius": 0.0,
        }
    gate = result["required_alpha_zero_gate"]
    assert gate["passed"] is True
    assert gate["exact_or_sentinel_gate_applied"] is False
    assert result["exact_sentinel_logic_present"] is False


def test_v65a_cluster_bootstrap_uses_joint_composite_distribution():
    reference = np.zeros((64, 4, 2, 3), dtype=np.float64)
    candidate = np.zeros_like(reference)
    reference[..., 0] = 0.5
    reference[..., 2] = 1.0
    candidate[..., 0] = 0.5
    candidate[..., 2] = 1.0
    # Correlated unit effects must be combined per unit before resampling.
    candidate[:32, ..., 0] = 0.6
    candidate[32:, ..., 0] = 0.4
    indices = subject.frozen_bootstrap_indices_v65a()
    result = subject.cluster_bootstrap_v65a(
        reference, candidate, bootstrap_indices=indices,
    )
    f1_units = np.r_[np.full(32, 0.1), np.full(32, -0.1)]
    composite_units = 0.8 * f1_units
    samples = np.mean(composite_units[indices], axis=1)
    ordered = np.sort(samples)
    lower_index = int(np.floor(
        subject.ONE_SIDED_ALPHA_V65A * (len(ordered) - 1)
    ))
    expected_lcb = ordered[lower_index]
    expected_ucb = ordered[len(ordered) - 1 - lower_index]
    joint = result["intervals"]["joint_composite"]
    assert joint["lcb"] == pytest.approx(expected_lcb)
    assert joint["ucb"] == pytest.approx(expected_ucb)
    assert joint["null_radius"] == pytest.approx(
        max(abs(expected_lcb), abs(expected_ucb))
    )
    assert result["joint_distribution_bootstrapped_before_quantiles"] is True
    temporal = result["temporal_pair_joint_composite_intervals"]
    assert set(temporal) == {"pair_0", "pair_1"}
    assert result["B_C_pass"] == max(
        temporal["pair_0"]["null_radius"],
        temporal["pair_1"]["null_radius"],
    )


def test_v65a_metric_types_exact_implication_and_axis_orientation():
    changed = _scored()
    changed[0][0][0]["exact"] = True
    with pytest.raises(ValueError):
        subject.validate_scored_periods_v65a(changed)
    changed = _scored()
    changed[0][0][0]["exact"] = 1
    changed[0][0][0]["f1"] = 0.9999999995
    with pytest.raises(ValueError):
        subject.validate_scored_periods_v65a(changed)

    scored = _scored()
    for period, f1 in enumerate((0.4, 0.5, 0.6, 0.7)):
        for actor in range(4):
            for unit in range(64):
                scored[period][actor][unit]["f1"] = f1
                scored[period][actor][unit]["nonzero"] = 1
    paired = subject.paired_replicas_v65a(scored)
    assert paired.shape == (64, 4, 2, 3)
    assert np.allclose(paired[:, :, 0, 0], -0.1)
    assert np.allclose(paired[:, :, 1, 0], 0.1)


def test_v65a_null_bound_transfer_is_frozen_and_gate_conditional():
    result = subject.analyze_scored_periods_v65a(_scored())
    transfer = result["future_v65_null_bound_transfer"]
    primary = result["primary_cluster_bootstrap"]
    assert transfer["outcome_independent_field_mapping"] == (
        subject.FUTURE_V65_NULL_BOUND_TRANSFER_V65A
    )
    assert transfer["bounds"] == {
        "B_F": primary["intervals"]["generated_f1_delta"]["null_radius"],
        "B_C": primary["intervals"]["joint_composite"]["null_radius"],
        "B_S": primary["intervals"]["stability_improvement"]["null_radius"],
        "B_C_pass": primary["B_C_pass"],
    }
    assert transfer[
        "rebind_or_launch_eligible_only_if_required_alpha_zero_gate_passed"
    ] is True
    assert transfer[
        "failed_required_alpha_zero_gate_forbids_bound_rebinding_and_v65_launch"
    ] is False
    assert transfer["required_future_v65_spread_gates"][
        "stability_coefficient_when_gate_not_met"
    ] == 0.0


def test_v65a_temporal_pair_radii_are_sealed_before_maximum():
    reference = np.zeros((64, 4, 2, 3), dtype=np.float64)
    candidate = np.zeros_like(reference)
    reference[..., 0] = 0.5
    reference[..., 2] = 1.0
    candidate[..., 0] = 0.5
    candidate[..., 2] = 1.0
    candidate[:32, :, 0, 0] = 0.52
    candidate[32:, :, 0, 0] = 0.48
    candidate[:32, :, 1, 0] = 0.60
    candidate[32:, :, 1, 0] = 0.40
    result = subject.cluster_bootstrap_v65a(reference, candidate)
    temporal = result["temporal_pair_joint_composite_intervals"]
    assert temporal["pair_1"]["null_radius"] > temporal["pair_0"]["null_radius"]
    assert result["B_C_pass"] == temporal["pair_1"]["null_radius"]
    assert result["B_C_pass_definition"] == (
        "max(pair_0.null_radius,pair_1.null_radius)"
    )


def test_v65a_gate_thresholds_are_inclusive_and_composite_is_required():
    primary = {
        "intervals": {
            "generated_f1_delta": {
                "contains_zero": True,
                "halfwidth": subject.MAX_PRIMARY_CI_HALFWIDTH_V65A,
            },
            "joint_composite": {"contains_zero": True},
            "stability_improvement": {"contains_zero": True},
        }
    }
    actor = {
        "maximum_absolute_leave_one_actor_out_shift": (
            subject.MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V65A
        )
    }
    assert subject.gate_v65a(primary, actor)["passed"] is True
    changed = copy.deepcopy(primary)
    changed["intervals"]["joint_composite"]["contains_zero"] = False
    assert subject.gate_v65a(changed, actor)["passed"] is False
    changed = copy.deepcopy(primary)
    changed["intervals"]["generated_f1_delta"]["halfwidth"] = np.nextafter(
        subject.MAX_PRIMARY_CI_HALFWIDTH_V65A, np.inf,
    )
    assert subject.gate_v65a(changed, actor)["passed"] is False


def test_v65a_state_receipts_require_exact_master_around_all_eight_periods():
    warmup = _receipts("unscored_warmup")
    scored = _receipts("scored")
    assert subject.validate_state_receipts_v65a(
        warmup, scored, _master_receipt(),
    ) == (warmup, scored)
    changed = copy.deepcopy(warmup)
    changed[2]["after"]["bf16_runtime_values_sha256"] = _sha("changed")
    with pytest.raises(ValueError):
        subject.validate_state_receipts_v65a(
            changed, scored, _master_receipt(),
        )


def test_v65a_builder_is_deterministic_exact64_and_measurement_only(tmp_path):
    panel_path = tmp_path / "panel.json"
    first_panel, first = builder.build_v65a(
        ranking_panel_output=panel_path,
        implementation_entry_paths=_implementation_entries(),
        _test_only_allow_nonproduction_entry_paths=True,
    )
    second_panel, second = builder.build_v65a(
        ranking_panel_output=panel_path,
        implementation_entry_paths=_implementation_entries(),
        _test_only_allow_nonproduction_entry_paths=True,
    )
    assert first_panel == second_panel
    assert first == second
    assert first["schema"] == (
        "v65a-ranking64-alpha-zero-calibration-preregistration"
    )
    assert first["status"] == (
        "sealed_before_v65a_train_semantics_model_ray_or_gpu_access"
    )
    authorization = first["authorization"]
    assert authorization["gpu_launch"] is True
    assert authorization["alpha_zero_calibration"] is True
    assert authorization["physical_gpu_ids"] == [0, 1, 2, 3]
    for forbidden in (
        "projection", "optimizer_update", "adapter_update", "candidate",
        "candidate_snapshot", "hpo_population", "train_holdback",
        "exact_sentinel", "ood_shadow", "protected_semantics",
        "terminal_holdout", "promotion",
    ):
        assert authorization[forbidden] is False
    assert first["content_sha256_before_self_field"] == (
        subject.self_content_sha256_v65a(first)
    )
    assert first["ranking_panel"]["file_sha256"] == (
        builder.payload_sha256_v65a(first_panel)
    )
    assert json.dumps(first_panel).lower().find('"question"') == -1
    assert json.dumps(first_panel).lower().find('"answer"') == -1


def test_v65a_builder_binds_prefix_schedule_engine_receipts_and_cluster_gate():
    panel, sources = builder.sealed_source_bindings_v65a()
    implementation = builder.implementation_bindings_v65a(
        _implementation_entries()
    )
    value = builder.build_preregistration_v65a(
        panel, sources, implementation,
        implementation_entry_paths=_implementation_entries(),
        _test_only_allow_nonproduction_entry_paths=True,
    )
    access = value["access_contract"]
    assert access["decode_exactly_first_64_v61c_ranking_rows"] is True
    assert access["decode_v61c_row_64_or_later"] is False
    assert access["ranking_prefix_bytes"] == 136_848
    assert access["ranking_prefix_sha256"] == (
        "8259894003268a2fafed6a9a66ce3e604d5eb76cdf19a1c1c759e5ffc5916c70"
    )
    assert access["source_file_size_metadata_bytes"] == 144_481
    assert access["live_authorized_prefix_pread_count"] == 2
    assert access["postrun_prefix_integrity_pread_decodes_semantics"] is False
    recipe = value["fixed_calibration_recipe"]
    assert recipe["lora_request"] == {
        "name": "v434_ranking64_alpha_zero_v65a",
        "integer_id": 1,
        "path": str(subject.population65.design52.STAGED_V52),
    }
    assert recipe["unscored_warmup_periods"] == 4
    assert recipe["scored_periods"] == 4
    assert recipe["pairs_per_actor"] == 2
    assert recipe["paired_replicas_per_unit"] == 8
    assert recipe["warmup_generation_completions_discarded"] == 1_024
    assert recipe["scored_generation_completions"] == 1_024
    assert recipe["total_generation_completions"] == 2_048
    assert recipe["runtime_determinism_controls"] == (
        subject.ENGINE_CONTROLS_V65A
    )
    assert recipe["candidate_before_reference_pairs_per_actor"] == 1
    assert recipe["candidate_after_reference_pairs_per_actor"] == 1
    receipt = recipe["sanitized_live_engine_and_cache_receipt"]
    assert receipt["required_from_every_actor"] is True
    assert receipt["active_lora_ids_exactly"] == [1]
    assert receipt["extra_or_candidate_adapter_loaded"] is False
    materialization = recipe["exact_master_rematerialization"]
    assert materialization["rpc"] == "rematerialize_exact_master_v65a"
    assert materialization[
        "required_before_every_warmup_and_scored_period"
    ] is True
    assert materialization["period_slot_write_receipts_required"] == 8
    assert materialization["read_only_live_slot_receipts_required"] == 16
    assert materialization[
        "after_generation_receipt_may_write_or_reset_slot"
    ] is False
    numeric = value["numeric_analysis_contract"]
    assert numeric[
        "within_unit_actor_pair_replicas_preserved_and_averaged"
    ] == 8
    assert numeric["resampled_axis"] == "conflict_unit_only"
    assert numeric[
        "joint_composite_distribution_bootstrapped_before_quantiles"
    ] is True
    assert numeric["joint_composite_weights"] == {
        "f1_delta": 0.8,
        "nonzero_delta": 0.2,
        "stability_improvement": 0.25,
    }
    assert set(numeric["temporal_pair_joint_composite_intervals"]) == {
        "pair_0", "pair_1",
    }
    assert numeric["B_C_pass_definition"] == (
        "max(pair_0.null_radius,pair_1.null_radius)"
    )
    transfer = numeric["future_v65_null_bound_transfer"]
    assert transfer["outcome_independent_field_mapping"] == (
        subject.FUTURE_V65_NULL_BOUND_TRANSFER_V65A
    )
    assert transfer[
        "rebind_or_launch_requires_required_alpha_zero_gate_passed"
    ] is True
    assert transfer[
        "failed_required_alpha_zero_gate_forbids_bound_rebinding_and_v65_launch"
    ] is True
    source = sources["v62b_batch68_methodology_and_threshold_source"]
    assert source["VLLM_BATCH_INVARIANT"] is False
    assert source["authorizes_exact_64_calibration_or_population"] is False
    assert value["required_integrity_gates"][
        "adapter_source_and_stage_contract_reverified_unchanged_postcleanup"
    ] is True


def test_v65a_builder_rejects_rehashed_panel_scope_tamper():
    panel, sources = builder.sealed_source_bindings_v65a()
    implementation = builder.implementation_bindings_v65a(
        _implementation_entries()
    )
    changed = copy.deepcopy(panel)
    changed["ranking_units"] = 68
    changed["content_sha256_before_self_field"] = (
        subject.canonical_sha256_v65a({
            key: item for key, item in changed.items()
            if key != "content_sha256_before_self_field"
        })
    )
    with pytest.raises(RuntimeError):
        builder.build_preregistration_v65a(
            changed, sources, implementation,
            implementation_entry_paths=_implementation_entries(),
            _test_only_allow_nonproduction_entry_paths=True,
        )


def test_v65a_builder_requires_full_runtime_external_execution_closure():
    panel, sources = builder.sealed_source_bindings_v65a()
    implementation = builder.implementation_bindings_v65a()
    external_keys = {
        f"entry__v64_runtime__{name}"
        for name in builder.runtime64.WORKER_EXECUTION_PATHS_V64
    }
    assert external_keys
    assert external_keys.issubset(implementation)
    assert builder.REQUIRED_IMPLEMENTATION_BINDING_KEYS_V65A.issubset(
        implementation
    )
    assert implementation[
        "entry__v64_runtime__upstream_es_trainer"
    ]["path"] == str(
        builder.ROOT / "es-at-scale/es_at_scale/trainer/es_trainer.py"
    )
    assert implementation[
        "entry__v64_runtime__ray_worker_sitecustomize"
    ]["path"] == str(builder.ROOT / "eggroll_es_compat/sitecustomize.py")

    changed = copy.deepcopy(implementation)
    changed.pop("entry__v64_runtime__upstream_es_trainer")
    with pytest.raises(RuntimeError):
        builder.build_preregistration_v65a(panel, sources, changed)


def test_v65a_builder_output_is_launchable_by_runtime_loader(tmp_path):
    panel_path = tmp_path / "ranking-panel.json"
    preregistration_path = tmp_path / "preregistration.json"
    _panel, preregistration = builder.build_v65a(
        ranking_panel_output=panel_path,
    )
    preregistration_path.write_bytes(builder.json_payload_v65a(preregistration))
    runtime = importlib.import_module(
        "run_lora_es_ranking64_alpha_zero_calibration_v65a"
    )
    args = SimpleNamespace(
        preregistration=str(preregistration_path),
        preregistration_sha256=subject.population65.file_sha256_v65(
            preregistration_path
        ),
        preregistration_content_sha256=preregistration[
            "content_sha256_before_self_field"
        ],
    )
    assert runtime.load_preregistration_v65a(args) == preregistration


def test_v65a_runtime_loader_rejects_rehashed_missing_or_changed_binding(
    tmp_path,
):
    _panel, preregistration = builder.build_v65a(
        ranking_panel_output=tmp_path / "ranking-panel.json",
    )
    runtime = importlib.import_module(
        "run_lora_es_ranking64_alpha_zero_calibration_v65a"
    )
    variants = []
    missing = copy.deepcopy(preregistration)
    missing["implementation_bindings"].pop(
        "entry__v64_runtime__upstream_es_trainer"
    )
    variants.append(("missing", missing))
    changed = copy.deepcopy(preregistration)
    changed["implementation_bindings"][
        "entry__v64_runtime__ray_worker_sitecustomize"
    ]["file_sha256"] = _sha("changed-sitecustomize")
    variants.append(("changed", changed))
    for label, value in variants:
        value["implementation_closure_manifest_sha256"] = (
            subject.canonical_sha256_v65a({
                key: binding["file_sha256"]
                for key, binding in sorted(
                    value["implementation_bindings"].items()
                )
            })
        )
        value.pop("content_sha256_before_self_field")
        value["content_sha256_before_self_field"] = (
            subject.canonical_sha256_v65a(value)
        )
        path = tmp_path / f"tampered-{label}.json"
        path.write_bytes(builder.json_payload_v65a(value))
        args = SimpleNamespace(
            preregistration=str(path),
            preregistration_sha256=(
                subject.population65.file_sha256_v65(path)
            ),
            preregistration_content_sha256=value[
                "content_sha256_before_self_field"
            ],
        )
        with pytest.raises(RuntimeError):
            runtime.load_preregistration_v65a(args)
