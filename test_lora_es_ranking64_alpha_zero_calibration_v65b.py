#!/usr/bin/env python3
"""CPU-only adversarial tests for the prospective V65B calibration."""

from __future__ import annotations

import copy
import hashlib
import json
import queue
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import build_lora_es_ranking64_alpha_zero_preregistration_v65b as builder
import eggroll_es_worker_lora_v65b as worker
import lora_es_ranking64_alpha_zero_calibration_v65b as subject
import run_lora_es_ranking64_alpha_zero_calibration_v65b as runtime


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _metric(index: int, f1: float = 0.5) -> dict:
    return {
        "request_index": index,
        "row_sha256": _sha(f"row-{index}"),
        "unit_identity_sha256": _sha(f"unit-{index}"),
        "f1": float(f1),
        "exact": int(f1 == 1.0),
        "nonzero": int(f1 > 0.0),
    }


def _scored(f1: float = 0.5) -> list:
    return [[[
        _metric(index, f1) for index in range(64)
    ] for _actor in range(4)] for _period in range(72)]


def _interval(
    *, lower: float = -0.0001, upper: float = 0.0001,
    point: float = 0.0,
) -> dict:
    return {
        "point": float(point),
        "lcb": float(lower),
        "ucb": float(upper),
        "halfwidth": float(0.5 * (upper - lower)),
        "contains_zero": bool(lower <= 0.0 <= upper),
        "null_radius": float(max(abs(lower), abs(upper))),
    }


def _group() -> dict:
    return {
        "generated_f1_delta": _interval(),
        "joint_composite": _interval(),
        "stability_improvement": _interval(),
    }


def _primary() -> dict:
    return {
        "schema": "v65b-ranking64-high-rep-cluster-bootstrap",
        "intervals": {
            **_group(),
            "generated_nonzero_delta": _interval(),
            "generated_exact_delta_diagnostic": _interval(),
        },
        "temporal_pass_intervals": {
            "early_position": _group(), "late_position": _group(),
        },
        "run_half_intervals": {
            "first_run_half": _group(), "second_run_half": _group(),
        },
        "orientation_effect_interval": _interval(),
        "early_minus_late_joint_interval": _interval(),
        "B_C_pass": 0.0001,
        "six_epoch_intervals": {f"epoch_{index}": _group()
                                for index in range(6)},
        "six_epoch_intervals_are_sealed_non_gating_diagnostics": True,
        "superblock_influence": {"sealed_non_gating_diagnostic": True},
    }


def _actor(shift: float = 0.0) -> dict:
    return {
        "schema": "v65b-ranking64-actor-influence",
        "maximum_absolute_leave_one_actor_out_shift": float(shift),
    }


def _gpu_row(reason: int = 4, *, phase: str | None = None) -> dict:
    phase = phase or "scored_period_0_generation_all_actors"
    forbidden = reason & runtime.FORBIDDEN_CLOCK_REASON_MASK_V65B
    return {
        "sampled_at_utc": "2026-07-16T00:00:00+00:00",
        "phase": phase,
        "generation_phase": runtime.is_generation_phase_v65b(phase),
        "gpu": 0,
        "expected_pid": 100,
        "compute_pids": [100],
        "foreign_compute_pids": [],
        "utilization_percent": 90,
        "memory_used_mib": 1000,
        "temperature_c": 80,
        "power_draw_mw": 250000,
        "clock_event_reasons_bitmask": reason,
        "clock_event_reasons_hex": f"0x{reason:016x}",
        "allowed_diagnostic_reasons_bitmask": (
            reason & runtime.ALLOWED_DIAGNOSTIC_CLOCK_REASON_MASK_V65B
        ),
        "forbidden_hardware_or_thermal_reasons_bitmask": forbidden,
        "hardware_or_thermal_slowdown_active": bool(forbidden),
    }


def test_schedule_is_exact_six_subset_crossover_and_position_balanced():
    value = subject.validate_schedule_v65b()
    assert value["scored_periods"] == 72
    assert value["adjacent_blocks"] == 36
    assert value["four_period_superblocks"] == 18
    assert value[
        "each_actor_candidate_and_reference_first_nine_times_per_position"
    ] is True
    for actor in range(4):
        orders = value["actor_orders"][str(actor)]
        assert orders.count("candidate_first") == 18
        assert orders[0::2].count("candidate_first") == 9
        assert orders[1::2].count("candidate_first") == 9
    for period in range(72):
        labels = [subject.label_v65b(actor, period) for actor in range(4)]
        assert labels.count("candidate") == 2
        assert labels.count("reference") == 2


def test_schedule_rejects_label_plan_not_bound_to_canonical_function(monkeypatch):
    forged = copy.deepcopy(subject.LABEL_PLAN_V65B)
    forged["0"], forged["2"] = forged["2"], forged["0"]
    monkeypatch.setattr(subject, "LABEL_PLAN_V65B", forged)
    with pytest.raises(RuntimeError, match="label plan"):
        subject.validate_schedule_v65b()


def test_schedule_rejects_globally_balanced_but_half_confounded_cycle(monkeypatch):
    sequence = (
        [(0, 1)] * 3 + [(2, 3)] * 3 + [(0, 2)] * 3
        + [(0, 3)] * 3 + [(1, 2)] * 3 + [(1, 3)] * 3
    )

    def confounded(block):
        base = frozenset(sequence[block // 2])
        return base if block % 2 == 0 else frozenset(range(4)) - base

    monkeypatch.setattr(subject, "_forward_actors_v65b", confounded)
    monkeypatch.setattr(subject, "LABEL_PLAN_V65B", {
        str(actor): [subject.label_v65b(actor, period) for period in range(72)]
        for actor in range(4)
    })
    with pytest.raises(RuntimeError, match="run-half subset balance"):
        subject.validate_schedule_v65b()


def test_pairing_recovers_candidate_minus_reference_for_every_cell():
    scored = _scored(0.0)
    for period in range(72):
        for actor in range(4):
            f1 = 0.75 if subject.label_v65b(actor, period) == "candidate" else 0.25
            scored[period][actor] = [_metric(index, f1) for index in range(64)]
    reference, candidate = subject.paired_state_metrics_v65b(scored)
    assert reference.shape == (64, 4, 36, 3)
    assert candidate.shape == (64, 4, 36, 3)
    np.testing.assert_array_equal(candidate[..., 0] - reference[..., 0], 0.5)


def test_scored_validator_rejects_duplicate_identities_uppercase_and_nonfinite():
    duplicate = _scored()
    for period in duplicate:
        for actor in period:
            for metric in actor:
                metric["row_sha256"] = "0" * 64
                metric["unit_identity_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="not 64 unique"):
        subject.validate_scored_periods_v65b(duplicate)

    uppercase = _scored()
    uppercase[0][0][0]["row_sha256"] = "A" * 64
    with pytest.raises(ValueError, match="identity"):
        subject.validate_scored_periods_v65b(uppercase)

    nonfinite = _scored()
    nonfinite[0][0][0]["f1"] = float("nan")
    with pytest.raises(ValueError):
        subject.validate_scored_periods_v65b(nonfinite)


def test_frozen_bootstrap_identity_and_integer_dtype_are_enforced():
    indices = subject.frozen_bootstrap_indices_v65b()
    assert indices.shape == (65_536, 64)
    assert hashlib.sha256(
        indices.astype("<i8", copy=False).tobytes(order="C")
    ).hexdigest() == subject.BOOTSTRAP_INDEX_MATRIX_SHA256_V65B
    floated = indices.astype(np.float64)
    floated += 0.1
    zeros = np.zeros((64, 4, 36, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="integer typed"):
        subject.cluster_bootstrap_v65b(
            zeros, zeros, bootstrap_indices=floated,
        )


def test_v65b_interval_uses_v65b_alpha_not_v65a_global(monkeypatch):
    values = np.linspace(-1.0, 1.0, 64)
    # One constant-unit bootstrap row for each possible unit.
    indices = np.repeat(np.arange(64, dtype=np.int64)[:, None], 64, axis=1)
    original = subject._interval_v65b(values, indices)
    monkeypatch.setattr(subject, "BOOTSTRAP_ALPHA_V65B", 0.49)
    changed = subject._interval_v65b(values, indices)
    assert changed["lcb"] != original["lcb"]
    assert changed["ucb"] != original["ucb"]


@pytest.mark.parametrize("malformation", [
    "empty_passes", "empty_halves", "bool_actor", "negative_actor",
    "nonfinite_actor", "inconsistent_halfwidth",
])
def test_gate_rejects_malformed_or_vacuous_inputs(malformation):
    primary = _primary()
    actor = _actor()
    if malformation == "empty_passes":
        primary["temporal_pass_intervals"] = {}
    elif malformation == "empty_halves":
        primary["run_half_intervals"] = {}
    elif malformation == "bool_actor":
        actor["maximum_absolute_leave_one_actor_out_shift"] = False
    elif malformation == "negative_actor":
        actor["maximum_absolute_leave_one_actor_out_shift"] = -1.0
    elif malformation == "nonfinite_actor":
        actor["maximum_absolute_leave_one_actor_out_shift"] = float("-inf")
    else:
        primary["intervals"]["generated_f1_delta"]["halfwidth"] = -1.0
    with pytest.raises(ValueError):
        subject.gate_v65b(primary, actor)


def _nonzero_interval() -> dict:
    return _interval(lower=0.001, upper=0.0012, point=0.0011)


@pytest.mark.parametrize(("name", "mutate", "expected_check"), [
    ("pooled_f1_zero", lambda p, a: p["intervals"].__setitem__(
        "generated_f1_delta", _nonzero_interval()),
     "generated_f1_primary_interval_contains_zero"),
    ("pooled_joint_zero", lambda p, a: p["intervals"].__setitem__(
        "joint_composite", _nonzero_interval()),
     "joint_composite_interval_contains_zero"),
    ("pooled_stability_zero", lambda p, a: p["intervals"].__setitem__(
        "stability_improvement", _nonzero_interval()),
     "stability_improvement_interval_contains_zero"),
    ("pooled_width", lambda p, a: p["intervals"].__setitem__(
        "generated_f1_delta", _interval(lower=-0.001, upper=0.001)),
     "generated_f1_primary_ci_halfwidth_within_fixed_limit"),
    ("actor", lambda p, a: a.__setitem__(
        "maximum_absolute_leave_one_actor_out_shift", 1.0),
     "actor_leave_one_out_shift_within_fixed_limit"),
    ("pass_f1_zero", lambda p, a: p["temporal_pass_intervals"]
     ["early_position"].__setitem__("generated_f1_delta", _nonzero_interval()),
     "both_temporal_pass_f1_intervals_contain_zero"),
    ("pass_joint_zero", lambda p, a: p["temporal_pass_intervals"]
     ["late_position"].__setitem__("joint_composite", _nonzero_interval()),
     "both_temporal_pass_joint_intervals_contain_zero"),
    ("pass_width", lambda p, a: p["temporal_pass_intervals"]
     ["early_position"].__setitem__(
         "generated_f1_delta", _interval(lower=-0.001, upper=0.001)),
     "both_temporal_pass_f1_halfwidths_within_fixed_limit"),
    ("orientation", lambda p, a: p.__setitem__(
        "orientation_effect_interval", _nonzero_interval()),
     "orientation_effect_interval_contains_zero"),
    ("pass_difference", lambda p, a: p.__setitem__(
        "early_minus_late_joint_interval", _nonzero_interval()),
     "early_minus_late_joint_interval_contains_zero"),
    ("half_stability", lambda p, a: p["run_half_intervals"]
     ["second_run_half"].__setitem__(
         "stability_improvement", _nonzero_interval()),
     "both_run_half_f1_joint_and_stability_intervals_contain_zero"),
    ("half_width", lambda p, a: p["run_half_intervals"]
     ["first_run_half"].__setitem__(
         "generated_f1_delta", _interval(lower=-0.001, upper=0.001)),
     "both_run_half_f1_halfwidths_within_fixed_limit"),
])
def test_every_claimed_gate_can_fail_closed(name, mutate, expected_check):
    del name
    primary, actor = _primary(), _actor()
    mutate(primary, actor)
    gate = subject.gate_v65b(primary, actor)
    assert gate["checks"][expected_check] is False
    assert gate["passed"] is False
    assert set(gate["checks"]) == subject.GATE_CHECK_KEYS_V65B


def test_epoch_and_superblock_diagnostics_are_sealed_but_non_gating():
    primary, actor = _primary(), _actor()
    expected = subject.gate_v65b(primary, actor)
    primary["six_epoch_intervals"]["epoch_0"][
        "generated_f1_delta"
    ] = _nonzero_interval()
    primary["superblock_influence"][
        "maximum_absolute_leave_one_superblock_out_shift"
    ] = 999.0
    assert subject.gate_v65b(primary, actor) == expected


def test_future_bounds_reject_forged_gate_and_nonfinite_bounds():
    primary = _primary()
    primary["orientation_effect_interval"] = _nonzero_interval()
    gate = subject.gate_v65b(primary, _actor())
    forged = copy.deepcopy(gate)
    forged["checks"]["orientation_effect_interval_contains_zero"] = True
    forged["passed"] = True
    with pytest.raises(ValueError, match="not linked"):
        subject.future_v65_null_bounds_v65b(primary, forged, _actor())

    primary = _primary()
    gate = subject.gate_v65b(primary, _actor())
    primary["B_C_pass"] = float("nan")
    with pytest.raises(ValueError, match="bound"):
        subject.future_v65_null_bounds_v65b(primary, gate, _actor())

    primary = _primary()
    gate = subject.gate_v65b(primary, _actor())
    primary["B_C_pass"] = 0.00009
    with pytest.raises(ValueError, match="bound"):
        subject.future_v65_null_bounds_v65b(primary, gate, _actor())


def test_future_bounds_require_exact_gate_checks_and_do_not_authorize_v65():
    primary, actor = _primary(), _actor()
    gate = subject.gate_v65b(primary, actor)
    observed = subject.future_v65_null_bounds_v65b(primary, gate, actor)
    assert observed["eligible_for_future_separate_preregistration"] is True
    assert observed["v65_population_launch_authorized"] is False
    broken = copy.deepcopy(gate)
    broken["checks"].pop(next(iter(broken["checks"])))
    with pytest.raises(ValueError):
        subject.future_v65_null_bounds_v65b(primary, broken, actor)

    influential = _actor(1.0)
    failed = subject.gate_v65b(primary, influential)
    forged = copy.deepcopy(failed)
    forged["checks"]["actor_leave_one_out_shift_within_fixed_limit"] = True
    forged["passed"] = True
    with pytest.raises(ValueError, match="not linked"):
        subject.future_v65_null_bounds_v65b(primary, forged, influential)


@pytest.mark.parametrize("period_index", [True, 1.9, -0.1, "1"])
def test_worker_rejects_coercible_period_indices(period_index):
    with pytest.raises(ValueError, match="exact types"):
        worker.LoRAAdapterStateWorkerExtensionV65B._period_v65b(
            "scored", period_index,
        )


def test_worker_rejects_string_coercible_period_and_edge_coordinates():
    class StringLike:
        def __str__(self):
            return "scored"

    with pytest.raises(ValueError, match="exact types"):
        worker.LoRAAdapterStateWorkerExtensionV65B._period_v65b(
            StringLike(), 0,
        )
    extension = object.__new__(worker.LoRAAdapterStateWorkerExtensionV65B)
    with pytest.raises(ValueError, match="exact sealed string"):
        extension.read_only_exact_master_slot_v65b(
            "scored", 0, StringLike(),
        )


def test_worker_accepts_only_exact_in_range_periods():
    assert worker.LoRAAdapterStateWorkerExtensionV65B._period_v65b(
        "unscored_warmup", 7,
    ) == ("unscored_warmup", 7)
    assert worker.LoRAAdapterStateWorkerExtensionV65B._period_v65b(
        "scored", 71,
    ) == ("scored", 71)
    with pytest.raises(ValueError):
        worker.LoRAAdapterStateWorkerExtensionV65B._period_v65b("scored", 72)


def test_r1_planning_recomputes_relevant_f1_and_naive_iid_values():
    value = builder.r1_planning_binding_v65b()
    recomputed = value["recomputed_planning_inputs"]
    assert recomputed["naive_iid_minimum_paired_replicas"] == pytest.approx(
        43.44359896859038, abs=1e-15,
    )
    assert recomputed[
        "projected_pair0_generated_f1_halfwidth_at_72"
    ] == pytest.approx(0.0006738575138486831, abs=1e-18)
    assert recomputed[
        "pair0_joint_projection_is_nongating_heuristic_because_stability_is_nonlinear"
    ] is True
    assert value["request_prompt_token_ids_sha256"] == (
        builder.R1_REQUEST_PROMPT_TOKEN_IDS_SHA256
    )


def test_builder_exact_reader_uses_one_duplicate_safe_byte_snapshot(
    tmp_path, monkeypatch,
):
    value = {"schema": "test", "numeric": 1}
    value["content_sha256_before_self_field"] = (
        builder.base.population65.self_content_sha256_v65(value)
    )
    path = tmp_path / "exact.json"
    payload = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(payload)
    file_hash = hashlib.sha256(payload).hexdigest()
    content_hash = value["content_sha256_before_self_field"]
    original_read_bytes = Path.read_bytes
    reads = 0

    def one_read(self):
        nonlocal reads
        reads += 1
        if reads > 1:
            raise AssertionError("sealed artifact was read more than once")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", one_read)
    assert builder._read_exact(path, file_hash, content_hash) == value
    assert reads == 1

    monkeypatch.setattr(Path, "read_bytes", original_read_bytes)
    duplicate = b'{"schema":"test","schema":"forged"}\n'
    path.write_bytes(duplicate)
    with pytest.raises(RuntimeError, match="duplicate JSON key"):
        builder._read_exact(
            path, hashlib.sha256(duplicate).hexdigest(), "0" * 64,
        )


def test_r1_planning_rejects_broken_evidence_analysis_finalizer_link(monkeypatch):
    original = builder._read_exact

    def forged(path, file_sha256, content_sha256):
        value = copy.deepcopy(original(path, file_sha256, content_sha256))
        if Path(path) == builder.R1_ANALYSIS:
            value["source_evidence_content_sha256"] = "0" * 64
        return value

    monkeypatch.setattr(builder, "_read_exact", forged)
    with pytest.raises(RuntimeError, match="R1 planning source"):
        builder.r1_planning_binding_v65b()


@pytest.mark.parametrize("field", [
    "f1_limit", "actor_limit", "alpha", "weights", "generation",
])
def test_r1_guard_rejects_threshold_or_recipe_reinterpretation(monkeypatch, field):
    if field == "f1_limit":
        monkeypatch.setattr(subject, "MAX_PRIMARY_CI_HALFWIDTH_V65B", 1.0)
    elif field == "actor_limit":
        monkeypatch.setattr(subject, "MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V65B", 1.0)
    elif field == "alpha":
        monkeypatch.setattr(subject, "BOOTSTRAP_ALPHA_V65B", 0.49)
    elif field == "weights":
        monkeypatch.setattr(subject, "COMPOSITE_WEIGHTS_V65B", {
            "f1_delta": 1.0, "nonzero_delta": 0.0,
            "stability_improvement": 0.0,
        })
    else:
        changed = dict(subject.GENERATION_PARAMS_WITHOUT_SEED_V65B)
        changed["temperature"] = 0.1
        monkeypatch.setattr(subject, "GENERATION_PARAMS_WITHOUT_SEED_V65B", changed)
    with pytest.raises(RuntimeError, match="R1 planning source"):
        builder.r1_planning_binding_v65b()


def test_builder_rejects_forged_panel_and_source_bindings():
    panel, sources = builder.base.sealed_source_bindings_v65a()
    bindings = builder.implementation_bindings_v65b()
    forged_panel = copy.deepcopy(panel)
    forged_panel["items"][0]["row_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="input changed"):
        builder.build_preregistration_v65b(
            forged_panel, sources, bindings,
        )
    forged_sources = copy.deepcopy(sources)
    forged_sources["forged"] = {"question": "raw semantics"}
    with pytest.raises(RuntimeError, match="input changed"):
        builder.build_preregistration_v65b(
            panel, forged_sources, bindings,
        )


def test_full_builder_seals_freshness_joint_gate_health_and_non_authority():
    _panel, prereg = builder.build_v65b()
    recipe = prereg["fixed_recipe"]
    numeric = prereg["numeric_contract"]
    assert recipe["warmups_run_in_fresh_v65b_process"] is True
    assert recipe["r1_warmup_or_engine_state_transferred"] is False
    assert recipe["same_exact_64_unit_and_request_order_as_r1"] is True
    assert numeric[
        "early_and_late_position_joint_intervals_must_contain_zero"
    ] is True
    assert recipe["gpu_hardware_health_monitor"][
        "forbidden_generation_clock_event_reason_mask"
    ] == 232
    assert recipe[
        "fixed_actor_rpc_generation_and_construction_timeouts"
    ] == (
        runtime.ACTOR_WAIT_TIMEOUT_CONTRACT_V65B
    )
    assert prereg["success_directly_authorizes_v65_population"] is False


@pytest.mark.parametrize("reason", [0, 1, 2, 4, 16, 256, 279])
def test_runtime_clock_reason_classification_allows_diagnostic_bits(reason):
    pid_map = {0: 100, 1: 101, 2: 102, 3: 103}
    assert runtime._validate_gpu_row_v65b(_gpu_row(reason), pid_map)[
        "hardware_or_thermal_slowdown_active"
    ] is False


def test_runtime_clock_reason_and_pid_attribution_fail_closed():
    pid_map = {0: 100, 1: 101, 2: 102, 3: 103}
    for reason in (8, 32, 64, 128):
        with pytest.raises(RuntimeError, match="sample value"):
            runtime._validate_gpu_row_v65b(_gpu_row(reason), pid_map)
        with pytest.raises(RuntimeError, match="sample value"):
            runtime._validate_gpu_row_v65b(_gpu_row(reason | 4), pid_map)
    with pytest.raises(RuntimeError):
        runtime._validate_gpu_row_v65b(_gpu_row(512), pid_map)
    foreign = _gpu_row(0)
    foreign["compute_pids"] = [100, 999]
    foreign["foreign_compute_pids"] = [999]
    with pytest.raises(RuntimeError):
        runtime._validate_gpu_row_v65b(foreign, pid_map)
    forged_foreign = _gpu_row(0)
    forged_foreign["compute_pids"] = [100, 999]
    forged_foreign["foreign_compute_pids"] = []
    with pytest.raises(RuntimeError):
        runtime._validate_gpu_row_v65b(forged_foreign, pid_map)
    float_expected_pid = _gpu_row(0)
    float_expected_pid["expected_pid"] = 100.0
    with pytest.raises(RuntimeError, match="sample value"):
        runtime._validate_gpu_row_v65b(float_expected_pid, pid_map)


def test_runtime_requires_lowercase_preregistration_hashes():
    with pytest.raises(RuntimeError, match="invalid SHA"):
        runtime._require_lower_sha256_v65b("A" * 64, "test")
    assert runtime._require_lower_sha256_v65b("a" * 64, "test") == "a" * 64


def test_fixed_actor_wait_timeout_is_applied_and_timeout_is_fail_closed():
    class FakeGetTimeoutError(Exception):
        pass

    class FakeRay:
        exceptions = SimpleNamespace(GetTimeoutError=FakeGetTimeoutError)

        def __init__(self, fail=False):
            self.fail = fail
            self.observed_timeout = None

        def get(self, handles, timeout=None):
            self.observed_timeout = timeout
            if self.fail:
                raise FakeGetTimeoutError("synthetic timeout")
            return list(handles)

    trainer = SimpleNamespace()
    fake = FakeRay()
    receipt = runtime.install_fixed_actor_wait_timeout_v65b(trainer, fake)
    assert receipt == runtime.ACTOR_WAIT_TIMEOUT_CONTRACT_V65B
    assert trainer._resolve([1, 2]) == [1, 2]
    assert fake.observed_timeout == 300.0

    trainer = SimpleNamespace()
    fake = FakeRay(fail=True)
    runtime.install_fixed_actor_wait_timeout_v65b(trainer, fake)
    with pytest.raises(
        runtime.V65BActorWaitTimeout, match="actor wait expired",
    ):
        trainer._resolve([1])
    assert runtime.ACTOR_WAIT_TIMEOUT_CONTRACT_V65B[
        "timeout_retry_drop_reorder_or_early_stop"
    ] is False
    assert runtime.ACTOR_WAIT_TIMEOUT_CONTRACT_V65B[
        "timeout_enters_strict_engine_cleanup_and_four_gpu_idle_proof"
    ] is True


def test_inherited_constructor_ray_get_is_bounded(monkeypatch):
    class FakeGetTimeoutError(Exception):
        pass

    class FakeRay:
        exceptions = SimpleNamespace(GetTimeoutError=FakeGetTimeoutError)

        def __init__(self):
            self.timeouts = []

        def get(self, handles, timeout=None):
            self.timeouts.append(timeout)
            return list(handles)

    fake_ray = FakeRay()
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    def inherited_constructor(_prereg, _prior, _state):
        import ray

        assert ray.get([1, 2]) == [1, 2]
        return "trainer", "saved"

    monkeypatch.setattr(runtime.base65a, "make_trainer_v65a", inherited_constructor)
    assert runtime.make_trainer_v65b({}, object(), {}) == ("trainer", "saved")
    assert fake_ray.timeouts == [
        runtime.FIXED_ACTOR_WAIT_TIMEOUT_SECONDS_V65B
    ]


def test_whole_constructor_watchdog_expires_fail_closed():
    with pytest.raises(TimeoutError, match="synthetic construction expired"):
        with runtime._fixed_sigalrm_watchdog_v65b(
            0.02, "synthetic construction",
        ):
            time.sleep(0.1)


def test_watchdog_rejects_late_success_after_inner_timeout_is_caught():
    with pytest.raises(TimeoutError, match="swallowed watchdog expired"):
        with runtime._fixed_sigalrm_watchdog_v65b(
            0.02, "swallowed watchdog",
        ):
            try:
                time.sleep(0.04)
            except BaseException:
                pass
            time.sleep(0.01)


def test_watchdog_pending_repeat_cannot_interrupt_disarm(monkeypatch):
    previous_handler = object()
    installed = {signal.SIGALRM: previous_handler}

    def fake_signal(number, handler):
        previous = installed.get(number)
        installed[number] = handler
        return previous

    def fake_setitimer(_kind, seconds, interval=0.0):
        del interval
        if seconds == 0.0:
            installed[signal.SIGALRM](signal.SIGALRM, None)

    monkeypatch.setattr(runtime.signal, "getitimer", lambda _kind: (0.0, 0.0))
    monkeypatch.setattr(runtime.signal, "getsignal", installed.get)
    monkeypatch.setattr(runtime.signal, "signal", fake_signal)
    monkeypatch.setattr(runtime.signal, "setitimer", fake_setitimer)
    with runtime._fixed_sigalrm_watchdog_v65b(10.0, "synthetic disarm"):
        pass
    assert installed[signal.SIGALRM] is previous_handler


def test_termination_handlers_cannot_be_replaced_and_never_exit_zero():
    previous_int = signal.getsignal(signal.SIGINT)
    previous_term = signal.getsignal(signal.SIGTERM)
    state = runtime.install_fixed_termination_handlers_v65b()
    try:
        installed = signal.getsignal(signal.SIGTERM)
        assert installed is state["handler"]
        assert signal.signal(signal.SIGTERM, signal.SIG_DFL) is installed
        assert signal.getsignal(signal.SIGTERM) is installed
        with pytest.raises(runtime.V65BTerminationRequested):
            installed(signal.SIGTERM, None)
        with pytest.raises(runtime.V65BTerminationRequested):
            installed(signal.SIGTERM, None)
        with pytest.raises(runtime.V65BTerminationRequested):
            runtime.raise_if_termination_requested_v65b(state)
        state["cleanup_started"] = True
        # A repeated signal is absorbed only after top-level cleanup owns it.
        assert installed(signal.SIGTERM, None) is None
    finally:
        runtime.restore_termination_handlers_v65b(state)
    assert signal.getsignal(signal.SIGINT) is previous_int
    assert signal.getsignal(signal.SIGTERM) is previous_term


def test_execute_installs_termination_handler_before_guarded_live_path(
    monkeypatch,
):
    previous_term = signal.getsignal(signal.SIGTERM)
    observed = []

    def guarded(_preregistration, _args, state):
        observed.append(signal.getsignal(signal.SIGTERM) is state["handler"])
        raise RuntimeError("synthetic early failure")

    monkeypatch.setattr(runtime, "_execute_v65b_guarded", guarded)
    with pytest.raises(RuntimeError, match="synthetic early failure"):
        runtime.execute_v65b({}, SimpleNamespace())
    assert observed == [True]
    assert signal.getsignal(signal.SIGTERM) is previous_term


def test_sanitized_cli_never_emits_raw_exception_text(monkeypatch, capsys):
    raw = "sensitive prompt and generation text"
    monkeypatch.setattr(
        runtime, "main", lambda _argv=None: (_ for _ in ()).throw(
            ValueError(raw)
        ),
    )
    assert runtime.sanitized_cli_v65b([]) == 1
    captured = capsys.readouterr()
    assert raw not in captured.err
    value = json.loads(captured.err)
    assert value["type"] == "ValueError"
    assert value["message_sha256"] == hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def test_reward_pool_shutdown_is_bounded_and_exact():
    class FakeProcess:
        def __init__(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def kill(self):
            self.alive = False

    class FakePool:
        def __init__(self, hang=False):
            self._pool = [FakeProcess() for _ in range(8)]
            self.hang = hang

        def terminate(self):
            for process in self._pool:
                process.alive = False

        def join(self):
            if self.hang:
                time.sleep(0.1)

    trainer = SimpleNamespace(mp_pool=FakePool())
    receipt = runtime.bounded_reward_pool_shutdown_v65b(
        trainer, timeout_seconds=0.1,
    )
    assert receipt["worker_process_count"] == 8
    assert receipt["all_workers_stopped"] is True
    assert isinstance(trainer.mp_pool, runtime._ClosedRewardPoolV65B)

    trainer = SimpleNamespace(mp_pool=FakePool(hang=True))
    with pytest.raises(TimeoutError, match="reward-pool join expired"):
        runtime.bounded_reward_pool_shutdown_v65b(
            trainer, timeout_seconds=0.01,
        )
    assert isinstance(trainer.mp_pool, runtime._ClosedRewardPoolV65B)


def test_reward_pool_shutdown_retains_live_pool_reference_after_failed_fallback():
    class UnkillableProcess:
        def is_alive(self):
            return True

        def kill(self):
            raise RuntimeError("synthetic kill failure")

        def join(self, timeout=None):
            del timeout

    class FailingPool:
        def __init__(self):
            self._pool = [UnkillableProcess() for _ in range(8)]

        def terminate(self):
            raise RuntimeError("synthetic terminate failure")

        def join(self):
            raise AssertionError("join must not follow failed terminate")

    pool = FailingPool()
    trainer = SimpleNamespace(mp_pool=pool)
    with pytest.raises(RuntimeError, match="fallback did not stop"):
        runtime.bounded_reward_pool_shutdown_v65b(
            trainer, timeout_seconds=0.1,
        )
    assert trainer.mp_pool is pool
    with pytest.raises(RuntimeError, match="fallback did not stop"):
        runtime.bounded_reward_pool_shutdown_v65b(
            trainer, timeout_seconds=0.1,
        )
    assert trainer.mp_pool is pool

def test_live_timeout_constants_must_equal_sealed_recipe(monkeypatch):
    _panel, prereg = builder.build_v65b()
    forged = copy.deepcopy(runtime.ACTOR_WAIT_TIMEOUT_CONTRACT_V65B)
    forged["seconds"] = 1.0
    monkeypatch.setattr(runtime, "ACTOR_WAIT_TIMEOUT_CONTRACT_V65B", forged)
    with pytest.raises(RuntimeError, match="sealed timeout/GPU"):
        runtime.validate_preregistered_runtime_contract_v65b(prereg)


def test_generation_wait_polls_monitor_failure_before_actor_deadline():
    failures = queue.Queue()

    class FakeRay:
        def __init__(self):
            self.waits = 0
            self.get_called = False

        def wait(self, pending, **_kwargs):
            self.waits += 1
            failures.put(RuntimeError("synthetic thermal slowdown"))
            return [], pending

        def get(self, _handles, timeout=None):
            del timeout
            self.get_called = True
            return []

    fake = FakeRay()
    with pytest.raises(RuntimeError, match="GPU health monitor failed"):
        runtime.resolve_generation_with_health_v65b(
            SimpleNamespace(), [1, 2, 3, 4], failures, None, fake,
        )
    assert fake.waits == 1
    assert fake.get_called is False


def test_worker_timing_rejects_negative_or_zero_elapsed_values():
    assert runtime._valid_worker_timing_v65b({
        "clock": "worker_monotonic_ns", "started_ns": 1,
        "ended_ns": 2, "elapsed_ns": 1,
    })
    assert not runtime._valid_worker_timing_v65b({
        "clock": "worker_monotonic_ns", "started_ns": -2,
        "ended_ns": -1, "elapsed_ns": 1,
    })
    assert not runtime._valid_worker_timing_v65b({
        "clock": "worker_monotonic_ns", "started_ns": 1,
        "ended_ns": 1, "elapsed_ns": 0,
    })


def test_runtime_gpu_json_parser_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "duplicate-gpu.jsonl"
    path.write_text('{"gpu":0,"gpu":1}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate GPU JSON key"):
        runtime._parse_gpu_rows_v65b(path)


def test_failure_publication_is_exclusive_and_idempotent(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    failure = run_dir / "failure.json"
    finalized = run_dir / "finalized.json"
    report = run_dir / "report.json"
    monkeypatch.setattr(runtime, "RUN_DIR", run_dir)
    monkeypatch.setattr(runtime, "FAILURE", failure)
    monkeypatch.setattr(runtime, "FINALIZED", finalized)
    monkeypatch.setattr(runtime, "REPORT", report)
    first = {"schema": "first"}
    second = {"schema": "second"}
    assert runtime.publish_failure_if_absent_v65b(first) is True
    assert runtime.publish_failure_if_absent_v65b(second) is False
    assert json.loads(failure.read_text(encoding="utf-8"))["schema"] == "first"

    failure.unlink()
    finalized.write_text("finalized\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="terminal claim"):
        runtime.publish_failure_if_absent_v65b(second)
    assert not failure.exists()

    finalized.unlink()
    report.write_text("report\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="durable report"):
        runtime.publish_failure_if_absent_v65b(second)
    assert not failure.exists()


def test_terminal_artifact_lock_wait_is_bounded(tmp_path, monkeypatch):
    operations = []

    def contended(_descriptor, operation):
        operations.append(operation)
        if operation != runtime.fcntl.LOCK_UN:
            raise BlockingIOError("synthetic contention")

    monkeypatch.setattr(runtime.fcntl, "flock", contended)
    monkeypatch.setattr(
        runtime, "FIXED_ACTOR_WAIT_TIMEOUT_SECONDS_V65B", 0.001,
    )
    with pytest.raises(TimeoutError, match="artifact lock wait expired"):
        with runtime.terminal_artifact_lock_v65b(tmp_path):
            raise AssertionError("contended lock must not yield")
    assert operations[-1] == runtime.fcntl.LOCK_UN


def test_best_effort_console_failure_cannot_demote_report(monkeypatch):
    def broken_print(*_args, **_kwargs):
        raise BrokenPipeError("closed consumer")

    monkeypatch.setattr("builtins.print", broken_print)
    assert runtime.best_effort_json_print_v65b({"status": "complete"}) is False


def test_run_directory_is_an_atomic_single_controller_claim(
    tmp_path, monkeypatch,
):
    run_dir = tmp_path / "nested" / "run"
    attempt = tmp_path / "attempt.json"
    monkeypatch.setattr(runtime, "RUN_DIR", run_dir)
    monkeypatch.setattr(runtime, "ATTEMPT", attempt)
    state = {}
    assert runtime.claim_fresh_run_directory_v65b(state) == run_dir
    assert state["run_directory_claimed"] is True
    assert run_dir.is_dir()
    with pytest.raises(RuntimeError, match="fresh artifact paths"):
        runtime.claim_fresh_run_directory_v65b()


def test_edge_actor_bindings_accept_permuted_physical_gpu_order():
    identities = [
        {"pid": 1000 + rank, "physical_gpu_id": gpu}
        for rank, gpu in enumerate((2, 0, 3, 1))
    ]
    trainer = SimpleNamespace()
    bindings = runtime.install_edge_actor_bindings_v65b(
        trainer, identities,
    )
    assert [
        binding["controller_actor_rank"] for binding in bindings
    ] == [0, 1, 2, 3]
    assert [
        binding["controller_physical_gpu_id"] for binding in bindings
    ] == [2, 0, 3, 1]
    assert [
        binding["controller_expected_pid"] for binding in bindings
    ] == [1000, 1001, 1002, 1003]
    assert trainer._v65b_edge_actor_bindings == bindings


def test_worker_receipts_emit_intrinsic_pid_and_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(worker.os, "getpid", lambda: 12345)
    identity = (
        worker.LoRAAdapterStateWorkerExtensionV65B
        ._intrinsic_worker_identity_v65b()
    )
    assert identity == {
        "worker_pid": 12345,
        "worker_physical_gpu_id": 2,
        "worker_cuda_visible_devices": "2",
    }


def test_canonical_exact_equality_rejects_python_numeric_aliases():
    assert runtime._canonical_equal_v65b({"value": 1}, {"value": 1})
    assert not runtime._canonical_equal_v65b({"value": True}, {"value": 1})
    assert not runtime._canonical_equal_v65b({"value": 1.0}, {"value": 1})


def test_preflight_injects_fixed_nvidia_smi_subprocess_timeout(monkeypatch):
    import subprocess

    observed = []

    def fake_run(*_args, **kwargs):
        observed.append(kwargs.get("timeout"))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    class FakeV40A:
        @staticmethod
        def gpu_preflight():
            subprocess.run(["nvidia-smi"], check=True)
            return {"ok": True}

    assert runtime.gpu_preflight_with_fixed_watchdog_v65b(FakeV40A()) == {
        "ok": True
    }
    assert observed == [runtime.FIXED_NVIDIA_SMI_SUBPROCESS_TIMEOUT_SECONDS_V65B]
