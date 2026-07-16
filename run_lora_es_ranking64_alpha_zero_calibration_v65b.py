#!/usr/bin/env python3
"""Run the prospective high-rep exact-64 alpha-zero calibration (V65B).

This is a fixed calibration, not an HPO population.  It performs eight fresh
unscored warmups followed by the sealed 72-period crossover, never changes the
V434 adapter, persists numeric/hash-only evidence, and never authorizes V65.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import queue
import signal
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import build_lora_es_ranking64_alpha_zero_preregistration_v65b as builder65b
import lora_es_nested_population_v52 as design52
import lora_es_ranking64_alpha_zero_calibration_v65b as analysis65b
import lora_es_robust_sampling_population_v65 as design65
import run_lora_es_baseline_census_v61a as runtime61a
import run_lora_es_nested_population_v52 as runtime52
import run_lora_es_paired_null_calibration_v61c as runtime61c
import run_lora_es_ranking64_alpha_zero_calibration_v65a as base65a
import run_lora_es_robust_sampling_population_v65 as runtime65
import run_lora_es_v59_vs_v434_robust_confirmation_v64 as runtime64


ROOT = Path(__file__).resolve().parent
EXPERIMENT = builder65b.EXPERIMENT
RUN_DIR = builder65b.RUN_DIR
ATTEMPT = Path(builder65b.artifacts_v65b()["attempt"])
EVIDENCE = Path(builder65b.artifacts_v65b()["evidence"])
ANALYSIS = Path(builder65b.artifacts_v65b()["analysis"])
REPORT = Path(builder65b.artifacts_v65b()["report"])
FAILURE = Path(builder65b.artifacts_v65b()["failure"])
FINALIZED = Path(builder65b.artifacts_v65b()["finalized"])
GPU_LOG = Path(builder65b.artifacts_v65b()["gpu_log"])
PREREGISTRATION = builder65b.PREREGISTRATION_OUTPUT
WORKER_EXTENSION_V65B = (
    "eggroll_es_worker_lora_v65b.LoRAAdapterStateWorkerExtensionV65B"
)
RUNTIME_CONTROLS_V65B = dict(analysis65b.ENGINE_CONTROLS_V65B)
FIXED_ACTOR_WAIT_TIMEOUT_SECONDS_V65B = 300.0
FIXED_CONSTRUCTION_WATCHDOG_SECONDS_V65B = 660.0
FIXED_REWARD_POOL_SHUTDOWN_TIMEOUT_SECONDS_V65B = 12.0
FIXED_GPU_PREFLIGHT_WATCHDOG_SECONDS_V65B = 30.0
FIXED_NVIDIA_SMI_SUBPROCESS_TIMEOUT_SECONDS_V65B = 10.0
FIXED_CLEANUP_AND_IDLE_WATCHDOG_SECONDS_V65B = 120.0
FIXED_FINAL_RAY_SHUTDOWN_WATCHDOG_SECONDS_V65B = 30.0
GPU_SAMPLE_INTERVAL_SECONDS_V65B = 0.5
GPU_MONITOR_FAILURE_POLL_SECONDS_V65B = 0.25
GPU_MONITOR_READY_TIMEOUT_SECONDS_V65B = 30.0
MAXIMUM_GPU_SAMPLE_CYCLE_GAP_SECONDS_V65B = 2.0
REPORT_WALL_CLOCK_TOLERANCE_SECONDS_V65B = 5.0
FOUR_ACTOR_CERTIFICATE_SHA256_V65B = (
    "00a49e786aa1e8b4bed0e0241235a4be118d757af5ee27ef479498e08af60000"
)
ACTOR_WAIT_TIMEOUT_CONTRACT_V65B = {
    "schema": (
        "v65b-fixed-construction-actor-rpc-generation-and-"
        "pool-shutdown-timeouts"
    ),
    "seconds": FIXED_ACTOR_WAIT_TIMEOUT_SECONDS_V65B,
    "whole_trainer_construction_watchdog_seconds": (
        FIXED_CONSTRUCTION_WATCHDOG_SECONDS_V65B
    ),
    "reward_pool_shutdown_timeout_seconds": (
        FIXED_REWARD_POOL_SHUTDOWN_TIMEOUT_SECONDS_V65B
    ),
    "gpu_preflight_watchdog_seconds": FIXED_GPU_PREFLIGHT_WATCHDOG_SECONDS_V65B,
    "nvidia_smi_subprocess_timeout_seconds": (
        FIXED_NVIDIA_SMI_SUBPROCESS_TIMEOUT_SECONDS_V65B
    ),
    "strict_cleanup_and_idle_watchdog_seconds": (
        FIXED_CLEANUP_AND_IDLE_WATCHDOG_SECONDS_V65B
    ),
    "final_ray_shutdown_watchdog_seconds": (
        FIXED_FINAL_RAY_SHUTDOWN_WATCHDOG_SECONDS_V65B
    ),
    "scope": [
        "whole_trainer_construction_watchdog",
        "placement_group_ready_waits",
        "constructor_collective_rpc_waits",
        "actor_identity_waits", "collective_rpc_waits",
        "four_actor_generation_waits",
        "reward_pool_terminate_and_join",
        "gpu_preflight_and_nvidia_smi_queries",
        "strict_cleanup_and_four_gpu_idle_proof",
        "final_unconditional_ray_shutdown",
        "terminal_artifact_lock_waits",
    ],
    "timeout_retry_drop_reorder_or_early_stop": False,
    "timeout_invalidates_calibration": True,
    "timeout_enters_strict_engine_cleanup_and_four_gpu_idle_proof": True,
    "successful_run_requires_all_after_generation_and_final_exact_state_receipts": True,
}

# NVML current-clock-event reason bits.  Idle, user/application clocks, a
# software power cap, sync boost, and display clocks are diagnostic only.
# Hardware slowdown, software/hardware thermal slowdown, and external power
# brake are fail-closed during any generation phase.
CLOCK_REASON_IDLE_V65B = 1
CLOCK_REASON_APPLICATION_CLOCKS_V65B = 2
CLOCK_REASON_SOFTWARE_POWER_CAP_V65B = 4
CLOCK_REASON_HARDWARE_SLOWDOWN_V65B = 8
CLOCK_REASON_SYNC_BOOST_V65B = 16
CLOCK_REASON_SOFTWARE_THERMAL_V65B = 32
CLOCK_REASON_HARDWARE_THERMAL_V65B = 64
CLOCK_REASON_HARDWARE_POWER_BRAKE_V65B = 128
CLOCK_REASON_DISPLAY_CLOCKS_V65B = 256
FORBIDDEN_CLOCK_REASON_MASK_V65B = (
    CLOCK_REASON_HARDWARE_SLOWDOWN_V65B
    | CLOCK_REASON_SOFTWARE_THERMAL_V65B
    | CLOCK_REASON_HARDWARE_THERMAL_V65B
    | CLOCK_REASON_HARDWARE_POWER_BRAKE_V65B
)
ALLOWED_DIAGNOSTIC_CLOCK_REASON_MASK_V65B = (
    CLOCK_REASON_IDLE_V65B
    | CLOCK_REASON_APPLICATION_CLOCKS_V65B
    | CLOCK_REASON_SOFTWARE_POWER_CAP_V65B
    | CLOCK_REASON_SYNC_BOOST_V65B
    | CLOCK_REASON_DISPLAY_CLOCKS_V65B
)
KNOWN_CLOCK_REASON_MASK_V65B = (
    ALLOWED_DIAGNOSTIC_CLOCK_REASON_MASK_V65B
    | FORBIDDEN_CLOCK_REASON_MASK_V65B
)
GPU_HARDWARE_HEALTH_CONTRACT_V65B = {
    "sample_interval_seconds": GPU_SAMPLE_INTERVAL_SECONDS_V65B,
    "controller_failure_poll_seconds": GPU_MONITOR_FAILURE_POLL_SECONDS_V65B,
    "initial_cycle_ready_timeout_seconds": (
        GPU_MONITOR_READY_TIMEOUT_SECONDS_V65B
    ),
    "generation_wait_aborts_on_monitor_failure_before_actor_deadline": True,
    "maximum_consecutive_sample_cycle_gap_seconds": (
        MAXIMUM_GPU_SAMPLE_CYCLE_GAP_SECONDS_V65B
    ),
    "sample_fields": [
        "physical_gpu", "expected_and_compute_pids",
        "utilization_percent", "memory_used_mib", "temperature_c",
        "power_draw_mw", "nvml_current_clock_event_reasons_bitmask",
    ],
    "foreign_compute_pid_fails_closed_in_every_phase": True,
    "generation_phase_count": 80,
    "forbidden_generation_clock_event_reason_mask": (
        FORBIDDEN_CLOCK_REASON_MASK_V65B
    ),
    "forbidden_generation_reasons": {
        "hardware_slowdown": CLOCK_REASON_HARDWARE_SLOWDOWN_V65B,
        "software_thermal_slowdown": CLOCK_REASON_SOFTWARE_THERMAL_V65B,
        "hardware_thermal_slowdown": CLOCK_REASON_HARDWARE_THERMAL_V65B,
        "hardware_external_power_brake": (
            CLOCK_REASON_HARDWARE_POWER_BRAKE_V65B
        ),
    },
    "diagnostic_nonfailing_reasons": {
        "idle": CLOCK_REASON_IDLE_V65B,
        "application_or_user_clocks": CLOCK_REASON_APPLICATION_CLOCKS_V65B,
        "software_power_cap": CLOCK_REASON_SOFTWARE_POWER_CAP_V65B,
        "sync_boost": CLOCK_REASON_SYNC_BOOST_V65B,
        "display_clock_setting": CLOCK_REASON_DISPLAY_CLOCKS_V65B,
    },
    "each_generation_phase_requires_attributed_positive_activity_on_all_four_gpus": True,
    "each_generation_phase_requires_a_simultaneous_positive_four_gpu_cycle": True,
    "thermal_or_hardware_slowdown_observations_required": 0,
}
GPU_SAMPLE_KEYS_V65B = frozenset({
    "sampled_at_utc", "phase", "generation_phase", "gpu",
    "expected_pid", "compute_pids", "foreign_compute_pids",
    "utilization_percent", "memory_used_mib", "temperature_c",
    "power_draw_mw", "clock_event_reasons_bitmask",
    "clock_event_reasons_hex", "allowed_diagnostic_reasons_bitmask",
    "forbidden_hardware_or_thermal_reasons_bitmask",
    "hardware_or_thermal_slowdown_active",
})


class V65BTerminationRequested(BaseException):
    """Non-successful, cleanup-owning response to SIGINT or SIGTERM."""


class V65BActorWaitTimeout(TimeoutError):
    """A timeout originating specifically from Ray actor or generation work."""


def install_fixed_termination_handlers_v65b() -> dict:
    """Prevent inherited handlers from converting interruption into exit 0."""
    original_signal = signal.signal
    protected = (signal.SIGINT, signal.SIGTERM)
    previous = {number: signal.getsignal(number) for number in protected}
    state = {
        "original_signal": original_signal,
        "previous": previous,
        "requested": None,
        "cleanup_started": False,
        "run_directory_claimed": False,
    }

    def terminate(number, _frame):
        # Once the top-level exception path explicitly owns cleanup, repeated
        # scheduler signals must not abort the four-GPU idle proof.  Before
        # then, keep raising: a nested catch must never consume the request.
        if state["cleanup_started"]:
            return
        state["requested"] = int(number)
        raise V65BTerminationRequested(
            f"v65b termination signal {int(number)} requested strict cleanup"
        )

    for number in protected:
        original_signal(number, terminate)

    def guarded_signal(number, handler):
        if number in protected:
            # Match signal.signal's return contract while keeping the V65B
            # handler installed through inherited trainer construction.
            return signal.getsignal(number)
        return original_signal(number, handler)

    signal.signal = guarded_signal
    state["handler"] = terminate
    return state


def raise_if_termination_requested_v65b(state: dict | None) -> None:
    if state is not None and state.get("requested") is not None:
        raise V65BTerminationRequested(
            f"v65b termination signal {state['requested']} remained pending"
        )


def restore_termination_handlers_v65b(state: dict | None) -> None:
    if state is None:
        return
    original_signal = state["original_signal"]
    signal.signal = original_signal
    for number, handler in state["previous"].items():
        original_signal(number, handler)


def artifacts_v65b() -> dict:
    return dict(builder65b.artifacts_v65b())


def best_effort_json_print_v65b(value: dict) -> bool:
    """Never demote durable evidence because a log consumer disappeared."""
    try:
        print(json.dumps(value, sort_keys=True))
    except OSError:
        return False
    return True


def claim_fresh_run_directory_v65b(
    termination_state: dict | None = None,
) -> Path:
    """Atomically give one controller exclusive ownership of this run."""
    if ATTEMPT.exists() or RUN_DIR.exists():
        raise RuntimeError("v65b requires fresh artifact paths")
    try:
        RUN_DIR.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise RuntimeError("v65b run directory was claimed concurrently") from error
    if termination_state is not None:
        termination_state["run_directory_claimed"] = True
    return RUN_DIR


@contextmanager
def terminal_artifact_lock_v65b(directory: Path):
    """Serialize terminal claims under the fixed actor-wait deadline."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(Path(directory), flags)
    try:
        deadline = (
            time.monotonic() + FIXED_ACTOR_WAIT_TIMEOUT_SECONDS_V65B
        )
        while True:
            try:
                fcntl.flock(
                    descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
                break
            except BlockingIOError as error:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "v65b fixed terminal artifact lock wait expired"
                    ) from error
                time.sleep(min(0.05, remaining))
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def publish_failure_if_absent_v65b(value: dict) -> bool:
    """Publish at most one failure while excluding finalizer publication."""
    with terminal_artifact_lock_v65b(RUN_DIR):
        if FINALIZED.exists():
            raise RuntimeError(
                "v65b finalized success already owns the terminal claim"
            )
        if REPORT.exists():
            raise RuntimeError(
                "v65b durable report already owns the runtime terminal claim"
            )
        if FAILURE.exists():
            return False
        runtime61a.atomic_json_v61a(FAILURE, value)
        return True


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--preregistration", required=True)
    value.add_argument("--preregistration-sha256", required=True)
    value.add_argument("--preregistration-content-sha256", required=True)
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--execute", action="store_true")
    return value


def _require_lower_sha256_v65b(value: object, name: str) -> str:
    if (
        not isinstance(value, str) or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"v65b invalid SHA-256: {name}")
    return value


def _exact_int_v65b(value: object, expected: int) -> bool:
    return type(expected) is int and type(value) is int and value == expected


def _exact_active_lora_one_v65b(value: object) -> bool:
    return (
        isinstance(value, list) and len(value) == 1
        and _exact_int_v65b(value[0], 1)
    )


def _canonical_equal_v65b(left: object, right: object) -> bool:
    try:
        return (
            design65.canonical_sha256_v65(left)
            == design65.canonical_sha256_v65(right)
        )
    except (TypeError, ValueError):
        return False


def _json_without_duplicates_v65b(path: Path) -> object:
    return _json_bytes_without_duplicates_v65b(
        Path(path).read_bytes(), Path(path),
    )


def _json_bytes_without_duplicates_v65b(
    payload: bytes, source_name: object,
) -> object:
    def reject(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise RuntimeError(
                    f"v65b duplicate JSON key in {source_name}: {key}"
                )
            value[key] = item
        return value

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"v65b non-UTF-8 JSON: {source_name}") from error
    return json.loads(text, object_pairs_hook=reject)


def load_preregistration_v65b(args) -> dict:
    path = Path(args.preregistration).resolve()
    file_sha = _require_lower_sha256_v65b(
        args.preregistration_sha256, "preregistration file",
    )
    content_sha = _require_lower_sha256_v65b(
        args.preregistration_content_sha256, "preregistration content",
    )
    payload = path.read_bytes()
    if (
        path != PREREGISTRATION.resolve()
        or hashlib.sha256(payload).hexdigest() != file_sha
    ):
        raise RuntimeError("v65b preregistration path or file changed")
    value = _json_bytes_without_duplicates_v65b(payload, path)
    _panel, expected = builder65b.build_v65b()
    if (
        not _canonical_equal_v65b(value, expected)
        or value.get("content_sha256_before_self_field") != content_sha
        or design65.self_content_sha256_v65(value) != content_sha
        or value.get("artifacts") != artifacts_v65b()
    ):
        raise RuntimeError("v65b preregistration contract changed")
    validate_preregistered_runtime_contract_v65b(value)
    return value


def validate_preregistered_runtime_contract_v65b(preregistration: dict) -> dict:
    """Bind live timeout/monitor constants to the exact sealed recipe."""
    recipe = preregistration.get("fixed_recipe", {}) \
        if isinstance(preregistration, dict) else {}
    if (
        not _canonical_equal_v65b(
            recipe.get("fixed_actor_rpc_generation_and_construction_timeouts"),
            ACTOR_WAIT_TIMEOUT_CONTRACT_V65B,
        )
        or not _canonical_equal_v65b(
            recipe.get("gpu_hardware_health_monitor"),
            GPU_HARDWARE_HEALTH_CONTRACT_V65B,
        )
        or recipe.get("report_wall_clock_reconciliation_tolerance_seconds")
        != REPORT_WALL_CLOCK_TOLERANCE_SECONDS_V65B
        or recipe.get("exact_master_rematerialization", {}).get(
            "four_actor_certificate_sha256"
        ) != FOUR_ACTOR_CERTIFICATE_SHA256_V65B
    ):
        raise RuntimeError("v65b sealed timeout/GPU runtime contract changed")
    return {
        "sealed_timeout_contract_equals_live_runtime": True,
        "sealed_gpu_monitor_contract_equals_live_runtime": True,
        "sealed_report_clock_tolerance_equals_live_runtime": True,
        "sealed_four_actor_master_certificate_equals_live_runtime": True,
    }


def _raise_ray_timeout_v65b(ray_module, error: BaseException, context: str):
    timeout_type = getattr(
        getattr(ray_module, "exceptions", None), "GetTimeoutError", (),
    )
    if timeout_type and isinstance(error, timeout_type):
        raise V65BActorWaitTimeout(
            f"v65b fixed {context} expired"
        ) from error
    raise error


def _bounded_ray_get_v65b(
    ray_module, original_get, handles, *, timeout=None, context: str,
):
    if timeout is None:
        effective = FIXED_ACTOR_WAIT_TIMEOUT_SECONDS_V65B
    else:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0.0
        ):
            raise RuntimeError("v65b invalid caller-supplied Ray timeout")
        effective = min(
            float(timeout), FIXED_ACTOR_WAIT_TIMEOUT_SECONDS_V65B,
        )
    try:
        return original_get(handles, timeout=effective)
    except BaseException as error:
        _raise_ray_timeout_v65b(ray_module, error, context)


@contextmanager
def _fixed_sigalrm_watchdog_v65b(seconds: float, context: str):
    """Interrupt and reject work that reaches or outlives a fixed deadline."""
    if (
        threading.current_thread() is not threading.main_thread()
        or not hasattr(signal, "SIGALRM")
        or not hasattr(signal, "setitimer")
    ):
        raise RuntimeError("v65b construction watchdog requires main-thread SIGALRM")
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(float(seconds))
        or float(seconds) <= 0.0
    ):
        raise RuntimeError("v65b invalid construction watchdog deadline")
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer != (0.0, 0.0):
        raise RuntimeError("v65b refuses to replace an existing real-time watchdog")
    previous_handler = signal.getsignal(signal.SIGALRM)

    deadline = time.monotonic() + float(seconds)
    state = {"expired": False, "disarming": False}

    def expired(_signum, _frame):
        if state["disarming"]:
            return
        state["expired"] = True
        raise TimeoutError(f"v65b fixed {context} expired")

    signal.signal(signal.SIGALRM, expired)
    # Repeat after the deadline so a nested catch cannot permanently disarm
    # the watchdog.  The elapsed-time check below also prevents false success
    # when the first TimeoutError was swallowed before the block returned.
    signal.setitimer(signal.ITIMER_REAL, float(seconds), 0.1)
    try:
        yield
    finally:
        # A pending repeat may be delivered during teardown.  Make the
        # handler idempotent first, then restore the timer and handler through
        # nested finally blocks so one restoration failure cannot skip the
        # other.
        state["disarming"] = True
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        finally:
            signal.signal(signal.SIGALRM, previous_handler)
        if state["expired"] or time.monotonic() >= deadline:
            raise TimeoutError(f"v65b fixed {context} expired")


@contextmanager
def _fixed_constructor_ray_get_v65b(ray_module):
    """Bound Ray waits reached inside the inherited constructor only."""
    original_get = ray_module.get

    def bounded_get(handles, *, timeout=None):
        return _bounded_ray_get_v65b(
            ray_module, original_get, handles, timeout=timeout,
            context="trainer-construction Ray wait",
        )

    ray_module.get = bounded_get
    try:
        yield
    finally:
        ray_module.get = original_get


@contextmanager
def _tracked_reward_pool_factory_v65b(construction_state: dict):
    """Expose a Pool object before its initializer can partially stall."""
    import multiprocessing
    import sys
    from multiprocessing.pool import Pool as PoolClass

    original_factory = multiprocessing.Pool
    module_name = "es_at_scale.trainer.es_trainer"
    original_module_pool = None
    original_module = sys.modules.get(module_name)
    if original_module is not None:
        original_module_pool = original_module.Pool

    class TrackedRewardPoolV65B(PoolClass):
        def __new__(cls, *args, **kwargs):
            value = super().__new__(cls)
            construction_state["partial_reward_pool"] = value
            return value

    def factory(*args, **kwargs):
        return TrackedRewardPoolV65B(*args, **kwargs)

    multiprocessing.Pool = factory
    if original_module is not None:
        original_module.Pool = factory
    try:
        yield
    finally:
        multiprocessing.Pool = original_factory
        loaded_module = sys.modules.get(module_name)
        if loaded_module is not None:
            loaded_module.Pool = (
                original_module_pool
                if original_module_pool is not None else original_factory
            )


@contextmanager
def _fixed_subprocess_timeout_v65b():
    import subprocess

    original_run = subprocess.run

    def bounded_run(*args, **kwargs):
        supplied = kwargs.get("timeout")
        if supplied is None:
            kwargs["timeout"] = FIXED_NVIDIA_SMI_SUBPROCESS_TIMEOUT_SECONDS_V65B
        else:
            kwargs["timeout"] = min(
                float(supplied), FIXED_NVIDIA_SMI_SUBPROCESS_TIMEOUT_SECONDS_V65B,
            )
        return original_run(*args, **kwargs)

    subprocess.run = bounded_run
    try:
        yield
    finally:
        subprocess.run = original_run


def make_trainer_v65b(preregistration, prior, construction_state=None):
    """Construct V65A's audited engine shape under the fresh V65B namespace."""
    if construction_state is None:
        construction_state = {}
    saved_base = (
        base65a.EXPERIMENT, base65a.RUN_DIR, base65a.WORKER_EXTENSION_V65A,
    )
    base65a.EXPERIMENT = EXPERIMENT
    base65a.RUN_DIR = RUN_DIR
    base65a.WORKER_EXTENSION_V65A = WORKER_EXTENSION_V65B
    try:
        with _fixed_sigalrm_watchdog_v65b(
            FIXED_CONSTRUCTION_WATCHDOG_SECONDS_V65B,
            "whole trainer construction watchdog",
        ):
            import ray

            with _fixed_constructor_ray_get_v65b(ray), \
                    _tracked_reward_pool_factory_v65b(construction_state):
                return base65a.make_trainer_v65a(
                    preregistration, prior, construction_state,
                )
    finally:
        (
            base65a.EXPERIMENT, base65a.RUN_DIR,
            base65a.WORKER_EXTENSION_V65A,
        ) = saved_base


def install_fixed_actor_wait_timeout_v65b(trainer, ray_module=None) -> dict:
    """Replace every post-construction actor wait with one fixed deadline."""
    if ray_module is None:
        import ray as ray_module

    def resolve(handles):
        return _bounded_ray_get_v65b(
            ray_module, ray_module.get, handles,
            context="postconstruction actor wait",
        )

    trainer._resolve = resolve
    receipt = dict(ACTOR_WAIT_TIMEOUT_CONTRACT_V65B)
    trainer._v65b_actor_wait_timeout_receipt = receipt
    return receipt


class _ClosedRewardPoolV65B:
    """No-op replacement after the real reward pool is conclusively stopped."""

    def close(self):
        return None

    def terminate(self):
        return None

    def join(self):
        return None


def _call_with_deadline_v65b(function, timeout: float, context: str) -> None:
    outcome: queue.Queue = queue.Queue(maxsize=1)

    def invoke():
        try:
            function()
        except BaseException as error:
            outcome.put((False, error))
        else:
            outcome.put((True, None))

    worker = threading.Thread(target=invoke, daemon=True)
    worker.start()
    worker.join(timeout=max(0.0, float(timeout)))
    if worker.is_alive():
        raise TimeoutError(f"v65b fixed {context} expired")
    succeeded, error = outcome.get_nowait()
    if not succeeded:
        raise error


def bounded_reward_pool_shutdown_v65b(
    trainer, *, timeout_seconds: float | None = None,
) -> dict:
    """Terminate and join the otherwise-unbounded legacy Pool fail-closed."""
    timeout = (
        FIXED_REWARD_POOL_SHUTDOWN_TIMEOUT_SECONDS_V65B
        if timeout_seconds is None else float(timeout_seconds)
    )
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise RuntimeError("v65b invalid reward-pool shutdown timeout")
    pool = getattr(trainer, "mp_pool", None) if trainer is not None else None
    if isinstance(pool, _ClosedRewardPoolV65B):
        return {
            "schema": "v65b-bounded-reward-pool-shutdown",
            "timeout_seconds": timeout,
            "pool_found": False,
            "worker_process_count": 0,
            "terminate_completed_before_deadline": True,
            "join_completed_before_deadline": True,
            "forced_worker_kill_count": 0,
            "all_workers_stopped": True,
        }
    if pool is None:
        if trainer is not None:
            trainer.mp_pool = _ClosedRewardPoolV65B()
        return {
            "schema": "v65b-bounded-reward-pool-shutdown",
            "timeout_seconds": timeout,
            "pool_found": False,
            "worker_process_count": 0,
            "terminate_completed_before_deadline": True,
            "join_completed_before_deadline": True,
            "forced_worker_kill_count": 0,
            "all_workers_stopped": True,
        }

    processes = list(getattr(pool, "_pool", []))
    deadline = time.monotonic() + timeout
    terminate_completed = join_completed = False
    forced_kills = 0
    try:
        _call_with_deadline_v65b(
            pool.terminate,
            min(timeout * 0.4, deadline - time.monotonic()),
            "reward-pool terminate",
        )
        terminate_completed = True
        _call_with_deadline_v65b(
            pool.join, min(timeout * 0.4, deadline - time.monotonic()),
            "reward-pool join",
        )
        join_completed = True
    except BaseException as original_error:
        for process in processes:
            try:
                if process.is_alive():
                    _call_with_deadline_v65b(
                        process.kill, deadline - time.monotonic(),
                        "reward-pool worker kill",
                    )
                    forced_kills += 1
            except BaseException:
                pass
        for process in processes:
            try:
                remaining = max(0.0, deadline - time.monotonic())
                _call_with_deadline_v65b(
                    lambda process=process, remaining=remaining: process.join(
                        timeout=remaining,
                    ),
                    remaining, "reward-pool worker join",
                )
            except BaseException:
                pass
        try:
            stopped_after_fallback = all(
                not process.is_alive() for process in processes
            )
        except BaseException:
            stopped_after_fallback = False
        if not stopped_after_fallback:
            # Keep the only controller-owned reference to the live pool.  A
            # later failure-cleanup pass must be able to retry termination;
            # replacing it here would both leak the workers and make the next
            # pass falsely report that no pool exists.
            raise RuntimeError(
                "v65b reward-pool fallback did not stop every worker"
            ) from original_error
        if trainer is not None:
            trainer.mp_pool = _ClosedRewardPoolV65B()
        raise original_error
    stopped = all(not process.is_alive() for process in processes)
    if stopped and trainer is not None:
        trainer.mp_pool = _ClosedRewardPoolV65B()
    receipt = {
        "schema": "v65b-bounded-reward-pool-shutdown",
        "timeout_seconds": timeout,
        "pool_found": True,
        "worker_process_count": len(processes),
        "terminate_completed_before_deadline": terminate_completed,
        "join_completed_before_deadline": join_completed,
        "forced_worker_kill_count": forced_kills,
        "all_workers_stopped": stopped,
    }
    if (
        len(processes) != 8
        or not terminate_completed or not join_completed
        or forced_kills != 0 or not stopped
    ):
        raise RuntimeError("v65b reward pool did not stop exactly before deadline")
    return receipt


def lora_request_v65b(prior):
    from vllm.lora.request import LoRARequest

    return LoRARequest(
        "v434_ranking64_alpha_zero_v65b", 1, str(design52.STAGED_V52),
        base_model_name=str(prior.v40a.MODEL),
    )


def is_generation_phase_v65b(value: object) -> bool:
    return (
        isinstance(value, str)
        and (
            value.startswith("unscored_warmup_")
            or value.startswith("scored_period_")
        )
        and value.endswith("_generation_all_actors")
    )


def _nvml_clock_reasons_v65b(pynvml, handle) -> int:
    getter = getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None)
    if getter is None:
        raise RuntimeError("v65b NVML clock-event reason API unavailable")
    value = getter(handle)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError("v65b invalid NVML clock-event reasons")
    return int(value)


def monitor_gpus_v65b(
    stop, phase, expected_pids, path, failures, ready,
) -> None:
    """Record attribution and health; fail closed during generation."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handles = {
            gpu: pynvml.nvmlDeviceGetHandleByIndex(gpu) for gpu in range(4)
        }
        with Path(path).open("x", encoding="utf-8") as output:
            first_cycle = True
            while not stop.is_set():
                sampled = datetime.now(timezone.utc).isoformat()
                current_phase = str(phase.value)
                generation = is_generation_phase_v65b(current_phase)
                for gpu, handle in handles.items():
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    pids = sorted({int(item.pid) for item in processes})
                    foreign = [pid for pid in pids if pid != expected_pids[gpu]]
                    if foreign:
                        raise RuntimeError(
                            f"v65b foreign GPU compute process on GPU {gpu}"
                        )
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    reasons = _nvml_clock_reasons_v65b(pynvml, handle)
                    if reasons & ~KNOWN_CLOCK_REASON_MASK_V65B:
                        raise RuntimeError("v65b unknown NVML clock-event reason")
                    forbidden = reasons & FORBIDDEN_CLOCK_REASON_MASK_V65B
                    row = {
                        "sampled_at_utc": sampled,
                        "phase": current_phase,
                        "generation_phase": generation,
                        "gpu": gpu,
                        "expected_pid": expected_pids[gpu],
                        "compute_pids": pids,
                        "foreign_compute_pids": foreign,
                        "utilization_percent": int(utilization.gpu),
                        "memory_used_mib": int(memory.used // (1024 * 1024)),
                        "temperature_c": int(pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU,
                        )),
                        "power_draw_mw": int(
                            pynvml.nvmlDeviceGetPowerUsage(handle)
                        ),
                        "clock_event_reasons_bitmask": reasons,
                        "clock_event_reasons_hex": f"0x{reasons:016x}",
                        "allowed_diagnostic_reasons_bitmask": (
                            reasons & ALLOWED_DIAGNOSTIC_CLOCK_REASON_MASK_V65B
                        ),
                        "forbidden_hardware_or_thermal_reasons_bitmask": (
                            forbidden
                        ),
                        "hardware_or_thermal_slowdown_active": bool(forbidden),
                    }
                    output.write(json.dumps(row, sort_keys=True) + "\n")
                    if generation and forbidden:
                        output.flush()
                        raise RuntimeError(
                            "v65b thermal or hardware slowdown during generation"
                        )
                output.flush()
                if first_cycle:
                    first_cycle = False
                    ready.set()
                stop.wait(GPU_SAMPLE_INTERVAL_SECONDS_V65B)
    except BaseException as error:
        failures.put(error)
    finally:
        ready.set()
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _raise_monitor_failure_v65b(failures: queue.Queue) -> None:
    if not failures.empty():
        raise RuntimeError("v65b GPU health monitor failed") from failures.get()


def resolve_generation_with_health_v65b(
    trainer, handles, failures: queue.Queue, termination_state,
    ray_module=None,
):
    """Wait in fixed-order while polling fatal monitor/signal state."""
    if ray_module is None:
        import ray as ray_module
    handles = list(handles)
    if len(handles) != 4:
        raise RuntimeError("v65b four-actor generation handle count changed")
    pending = list(handles)
    deadline = time.monotonic() + FIXED_ACTOR_WAIT_TIMEOUT_SECONDS_V65B
    while pending:
        _raise_monitor_failure_v65b(failures)
        raise_if_termination_requested_v65b(termination_state)
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise V65BActorWaitTimeout(
                "v65b fixed four-actor generation wait expired"
            )
        _ready, pending = ray_module.wait(
            pending, num_returns=len(pending),
            timeout=min(GPU_MONITOR_FAILURE_POLL_SECONDS_V65B, remaining),
            fetch_local=False,
        )
    _raise_monitor_failure_v65b(failures)
    raise_if_termination_requested_v65b(termination_state)
    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        raise V65BActorWaitTimeout(
            "v65b fixed four-actor generation wait expired"
        )
    return _bounded_ray_get_v65b(
        ray_module, ray_module.get, handles, timeout=remaining,
        context="four-actor generation wait",
    )


def _validate_gpu_row_v65b(row: object, pid_map: dict[int, int]) -> dict:
    if not isinstance(row, dict) or set(row) != GPU_SAMPLE_KEYS_V65B:
        raise RuntimeError("v65b GPU sample schema changed")
    gpu = row.get("gpu")
    reasons = row.get("clock_event_reasons_bitmask")
    forbidden = row.get("forbidden_hardware_or_thermal_reasons_bitmask")
    allowed = row.get("allowed_diagnostic_reasons_bitmask")
    compute = row.get("compute_pids")
    foreign = row.get("foreign_compute_pids")
    numeric_nonnegative = (
        "utilization_percent", "memory_used_mib", "temperature_c",
        "power_draw_mw", "clock_event_reasons_bitmask",
        "allowed_diagnostic_reasons_bitmask",
        "forbidden_hardware_or_thermal_reasons_bitmask",
    )
    if (
        type(gpu) is not int or gpu not in range(4)
        or type(row.get("expected_pid")) is not int
        or row.get("expected_pid") != pid_map[gpu]
        or not isinstance(compute, list) or compute != sorted(set(compute))
        or any(type(pid) is not int or pid <= 0 for pid in compute)
        or not isinstance(foreign, list)
        or foreign != [pid for pid in compute if pid != pid_map[gpu]]
        or foreign
        or any(type(value) is not int or value < 0
               for key in numeric_nonnegative for value in [row.get(key)])
        or row.get("utilization_percent") > 100
        or row.get("generation_phase") is not is_generation_phase_v65b(
            row.get("phase")
        )
        or row.get("clock_event_reasons_hex") != f"0x{reasons:016x}"
        or reasons & ~KNOWN_CLOCK_REASON_MASK_V65B
        or allowed != reasons & ALLOWED_DIAGNOSTIC_CLOCK_REASON_MASK_V65B
        or forbidden != reasons & FORBIDDEN_CLOCK_REASON_MASK_V65B
        or row.get("hardware_or_thermal_slowdown_active") is not bool(forbidden)
        or row.get("generation_phase") is True and forbidden != 0
    ):
        raise RuntimeError("v65b GPU sample value changed")
    try:
        sampled = datetime.fromisoformat(row.get("sampled_at_utc"))
    except (TypeError, ValueError) as error:
        raise RuntimeError("v65b invalid GPU sample timestamp") from error
    if sampled.tzinfo is None or sampled.utcoffset() is None:
        raise RuntimeError("v65b timezone-naive GPU sample timestamp")
    return row


def _gpu_sample_cycles_v65b(rows: list[dict]) -> list[dict]:
    cycles = []
    seen = set()
    for row in rows:
        sampled = row["sampled_at_utc"]
        if not cycles or cycles[-1]["sampled_at_utc"] != sampled:
            if sampled in seen:
                raise RuntimeError("v65b GPU sample cycle timestamp reappeared")
            seen.add(sampled)
            cycles.append({"sampled_at_utc": sampled, "rows": []})
        cycles[-1]["rows"].append(row)
    parsed = []
    for cycle in cycles:
        values = cycle["rows"]
        if (
            len(values) != 4
            or [row["gpu"] for row in values] != list(range(4))
            or len({row["phase"] for row in values}) != 1
            or len({row["generation_phase"] for row in values}) != 1
        ):
            raise RuntimeError("v65b incomplete or incoherent four-GPU sample cycle")
        parsed.append(datetime.fromisoformat(cycle["sampled_at_utc"]))
        cycle["phase"] = values[0]["phase"]
        cycle["generation_phase"] = values[0]["generation_phase"]
    gaps = [
        (current - previous).total_seconds()
        for previous, current in zip(parsed, parsed[1:])
    ]
    if (
        any(gap <= 0.0 for gap in gaps)
        or any(gap > MAXIMUM_GPU_SAMPLE_CYCLE_GAP_SECONDS_V65B for gap in gaps)
    ):
        raise RuntimeError("v65b GPU monitor sampling cadence changed")
    return cycles


def _parse_gpu_rows_v65b(path: Path) -> list[dict]:
    rows = []

    def reject_constant(value):
        raise RuntimeError(f"v65b non-finite GPU JSON constant: {value}")

    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1,
    ):
        if not line:
            continue

        def reject_duplicates(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise RuntimeError(
                        f"v65b duplicate GPU JSON key at row {line_number}: {key}"
                    )
                value[key] = item
            return value

        try:
            rows.append(json.loads(
                line, object_pairs_hook=reject_duplicates,
                parse_constant=reject_constant,
            ))
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"v65b malformed GPU JSON row {line_number}"
            ) from error
    if not rows:
        raise RuntimeError("v65b GPU log is empty")
    return rows


def gpu_period_phase_summary_v65b(path: Path, pid_map: dict[int, int]) -> dict:
    rows = [
        _validate_gpu_row_v65b(row, pid_map)
        for row in _parse_gpu_rows_v65b(path)
    ]
    cycles = _gpu_sample_cycles_v65b(rows)
    phases = [
        f"unscored_warmup_{index}_generation_all_actors"
        for index in range(analysis65b.WARMUP_PERIODS_V65B)
    ] + [
        f"scored_period_{index}_generation_all_actors"
        for index in range(analysis65b.SCORED_PERIODS_V65B)
    ]
    phase_index = {phase: index for index, phase in enumerate(phases)}
    observed_generation_indices = []
    for cycle in cycles:
        if cycle["generation_phase"]:
            if cycle["phase"] not in phase_index:
                raise RuntimeError("v65b unknown generation phase in GPU log")
            observed_generation_indices.append(phase_index[cycle["phase"]])
    if (
        observed_generation_indices != sorted(observed_generation_indices)
        or set(observed_generation_indices) != set(range(80))
    ):
        raise RuntimeError("v65b warmup/scored GPU phase order changed")
    by_phase = {}
    for phase in phases:
        phase_cycles = [cycle for cycle in cycles if cycle["phase"] == phase]
        simultaneous = [
            cycle for cycle in phase_cycles
            if all(
                pid_map[row["gpu"]] in row["compute_pids"]
                and row["utilization_percent"] > 0
                for row in cycle["rows"]
            )
        ]
        if not simultaneous:
            raise RuntimeError(
                f"v65b no simultaneous positive four-GPU cycle in {phase}"
            )
        by_gpu = {}
        for gpu in range(4):
            selected = [
                row for row in rows
                if row["phase"] == phase and row["gpu"] == gpu
            ]
            resident = [
                row for row in selected
                if pid_map[gpu] in row["compute_pids"]
            ]
            if (
                not resident
                or not any(row["utilization_percent"] > 0 for row in resident)
                or any(row["foreign_compute_pids"] for row in selected)
                or any(row["forbidden_hardware_or_thermal_reasons_bitmask"]
                       for row in selected)
            ):
                raise RuntimeError(
                    f"v65b GPU {gpu} activity or health failed in {phase}"
                )
            by_gpu[str(gpu)] = {
                "samples": len(selected),
                "resident_samples": len(resident),
                "positive_resident_samples": sum(
                    row["utilization_percent"] > 0 for row in resident
                ),
                "peak_utilization_percent": max(
                    row["utilization_percent"] for row in resident
                ),
                "peak_memory_used_mib": max(
                    row["memory_used_mib"] for row in resident
                ),
                "peak_temperature_c": max(
                    row["temperature_c"] for row in resident
                ),
                "peak_power_draw_mw": max(
                    row["power_draw_mw"] for row in resident
                ),
                "allowed_diagnostic_clock_reason_observations": sum(
                    bool(row["allowed_diagnostic_reasons_bitmask"])
                    for row in resident
                ),
                "forbidden_hardware_or_thermal_reason_observations": 0,
            }
        by_phase[phase] = by_gpu
    return {
        "schema": "v65b-per-period-four-gpu-activity-and-health",
        "generation_phases": 80,
        "all_eighty_generation_phases_positive_on_all_four_gpus": True,
        "all_eighty_generation_phases_have_a_simultaneous_positive_four_gpu_cycle": True,
        "complete_four_gpu_sample_cycles": len(cycles),
        "maximum_consecutive_sample_cycle_gap_seconds": (
            max(
                (datetime.fromisoformat(current["sampled_at_utc"])
                 - datetime.fromisoformat(previous["sampled_at_utc"]))
                .total_seconds()
                for previous, current in zip(cycles, cycles[1:])
            ) if len(cycles) > 1 else 0.0
        ),
        "sample_cycle_gap_within_sealed_limit": True,
        "foreign_compute_process_observations": 0,
        "thermal_or_hardware_slowdown_observations_in_generation": 0,
        "allowed_idle_app_clock_or_software_power_cap_reasons_are_nonfailing": True,
        "by_phase": by_phase,
    }


MASTER_CERTIFICATE_KEYS_V65B = frozenset({
    "canonical_fp32_master_sha256", "canonical_master_identity_sha256",
    "four_actor_certificate_sha256", "bf16_runtime_values_sha256",
})
READ_AGGREGATE_KEYS_V65B = frozenset({
    "schema", "canonical_fp32_master_sha256",
    "canonical_master_identity_sha256", "bf16_runtime_values_sha256",
    "runtime_view_count_per_actor", "runtime_elements_per_actor",
    "runtime_dtype", "base_inventory_sha256",
    "four_actor_exact_read_only_consensus",
})
CONTROLLER_ACTOR_BINDING_KEYS_V65B = frozenset({
    "controller_actor_rank", "controller_expected_pid",
    "controller_physical_gpu_id",
})
INTRINSIC_WORKER_IDENTITY_KEYS_V65B = frozenset({
    "worker_pid", "worker_physical_gpu_id", "worker_cuda_visible_devices",
})


def _validate_master_certificate_v65b(receipt: object) -> dict:
    if (
        not isinstance(receipt, dict)
        or set(receipt) != MASTER_CERTIFICATE_KEYS_V65B
        or base65a._master_core_v65a(receipt) != {
            "canonical_fp32_master_sha256": design52.MASTER_SHA256_V52,
            "canonical_master_identity_sha256": (
                analysis65b.v65a.MASTER_IDENTITY_OBJECT_SHA256_V65A
            ),
            "bf16_runtime_values_sha256": (
                design52.MASTER_RUNTIME_SHA256_V52
            ),
        }
    ):
        raise RuntimeError("v65b exact four-actor master certificate changed")
    if (
        _require_lower_sha256_v65b(
            receipt.get("four_actor_certificate_sha256"),
            "four-actor master certificate",
        ) != FOUR_ACTOR_CERTIFICATE_SHA256_V65B
    ):
        raise RuntimeError("v65b four-actor master certificate hash changed")
    return receipt


def _validate_read_aggregate_v65b(aggregate: object) -> dict:
    if (
        not isinstance(aggregate, dict)
        or set(aggregate) != READ_AGGREGATE_KEYS_V65B
        or aggregate.get("schema")
        != "v65b-read-only-four-actor-master-slot-consensus"
        or base65a._master_core_v65a(aggregate) != {
            "canonical_fp32_master_sha256": design52.MASTER_SHA256_V52,
            "canonical_master_identity_sha256": (
                analysis65b.v65a.MASTER_IDENTITY_OBJECT_SHA256_V65A
            ),
            "bf16_runtime_values_sha256": (
                design52.MASTER_RUNTIME_SHA256_V52
            ),
        }
        or not _exact_int_v65b(
            aggregate.get("runtime_view_count_per_actor"), 82,
        )
        or not _exact_int_v65b(
            aggregate.get("runtime_elements_per_actor"), 4_921_344,
        )
        or aggregate.get("runtime_dtype") != "torch.bfloat16"
        or aggregate.get("base_inventory_sha256")
        != analysis65b.v65a.BASE_INVENTORY_SHA256_V65A
        or aggregate.get("four_actor_exact_read_only_consensus") is not True
    ):
        raise RuntimeError("v65b exact read-only aggregate changed")
    return aggregate


def _valid_worker_timing_v65b(value: object) -> bool:
    return (
        base65a._valid_worker_timing_v65a(value)
        and value["started_ns"] >= 0
        and value["ended_ns"] >= 0
        and value["elapsed_ns"] > 0
    )


def install_edge_actor_bindings_v65b(trainer, identities: list[dict]) -> list:
    if not isinstance(identities, list) or len(identities) != 4:
        raise RuntimeError("v65b edge actor identity coverage changed")
    bindings = []
    for rank, identity in enumerate(identities):
        physical_gpu_id = (
            identity.get("physical_gpu_id")
            if isinstance(identity, dict) else None
        )
        if (
            not isinstance(identity, dict)
            or type(identity.get("pid")) is not int
            or identity["pid"] <= 0
            or type(physical_gpu_id) is not int
            or physical_gpu_id not in range(4)
        ):
            raise RuntimeError("v65b edge actor identity changed")
        bindings.append({
            "controller_actor_rank": rank,
            "controller_expected_pid": identity.get("pid"),
            "controller_physical_gpu_id": physical_gpu_id,
        })
    if (
        {item["controller_actor_rank"] for item in bindings} != set(range(4))
        or {item["controller_expected_pid"] for item in bindings}
        != {identity["pid"] for identity in identities}
        or {item["controller_physical_gpu_id"] for item in bindings}
        != set(range(4))
    ):
        raise RuntimeError("v65b edge actor bindings are not one-to-one")
    trainer._v65b_edge_actor_bindings = bindings
    return bindings


def _bind_actor_receipts_v65b(trainer, actors: object) -> list[dict]:
    bindings = getattr(trainer, "_v65b_edge_actor_bindings", None)
    if (
        not isinstance(actors, list) or len(actors) != 4
        or not isinstance(bindings, list) or len(bindings) != 4
    ):
        raise RuntimeError("v65b controller actor binding coverage changed")
    result = []
    for rank, (actor, binding) in enumerate(
        zip(actors, bindings, strict=True),
    ):
        if (
            not isinstance(actor, dict)
            or binding.get("controller_actor_rank") != rank
            or type(actor.get("worker_pid")) is not int
            or actor.get("worker_pid") != binding.get("controller_expected_pid")
            or not _exact_int_v65b(
                actor.get("worker_physical_gpu_id"),
                binding.get("controller_physical_gpu_id"),
            )
            or actor.get("worker_cuda_visible_devices")
            != str(binding.get("controller_physical_gpu_id"))
        ):
            raise RuntimeError("v65b controller actor binding changed")
        result.append({**actor, **binding})
    return result


def _controller_actor_binding_valid_v65b(
    actor: dict, actor_identities: list[dict], rank: int,
) -> bool:
    identity = actor_identities[rank]
    return (
        _exact_int_v65b(actor.get("controller_actor_rank"), rank)
        and type(actor.get("controller_expected_pid")) is int
        and actor.get("controller_expected_pid") == identity.get("pid")
        and _exact_int_v65b(
            actor.get("controller_physical_gpu_id"),
            identity.get("physical_gpu_id"),
        )
        and type(actor.get("worker_pid")) is int
        and actor.get("worker_pid") == identity.get("pid")
        and _exact_int_v65b(
            actor.get("worker_physical_gpu_id"),
            identity.get("physical_gpu_id"),
        )
        and actor.get("worker_cuda_visible_devices")
        == str(identity.get("physical_gpu_id"))
    )


def _slot_write_v65b(trainer, v40a, period_kind, period_index):
    pre_write = runtime61c._assert_v434_certificates_v61c(
        v40a._rpc_all(trainer, "adapter_state_certificate_v52")
    )
    actors = v40a._rpc_all(
        trainer, "rematerialize_exact_master_v65b", (
            period_kind, period_index, design52.MASTER_SHA256_V52,
            design52.MASTER_RUNTIME_SHA256_V52,
        ),
    )
    if (
        len(actors) != 4
        or any(
            not isinstance(receipt, dict)
            or receipt.get("schema") != "exact-master-slot-write-v65b"
            or receipt.get("period_kind") != period_kind
            or not _exact_int_v65b(receipt.get("period_index"), period_index)
            or receipt.get("master_identity", {}).get("sha256")
            != design52.MASTER_SHA256_V52
            or design65.canonical_sha256_v65(receipt.get("master_identity"))
            != analysis65b.v65a.MASTER_IDENTITY_OBJECT_SHA256_V65A
            or receipt.get("materialization", {}).get("storage_layout_sha256")
            != analysis65b.v65a.MATERIALIZATION_STORAGE_LAYOUT_SHA256_V65A
            or receipt.get("materialization", {}).get("runtime_values_sha256")
            != design52.MASTER_RUNTIME_SHA256_V52
            or receipt.get("base_identity", {}).get("inventory_sha256")
            != analysis65b.v65a.BASE_INVENTORY_SHA256_V65A
            or receipt.get("transaction_state_quiescent") is not True
            for receipt in actors
        )
    ):
        raise RuntimeError("v65b exact-master slot-write receipt changed")
    actors = _bind_actor_receipts_v65b(trainer, actors)
    post_write = runtime61c._assert_v434_certificates_v61c(
        v40a._rpc_all(trainer, "adapter_state_certificate_v52")
    )
    if (
        _validate_master_certificate_v65b(pre_write)
        != _validate_master_certificate_v65b(post_write)
    ):
        raise RuntimeError("v65b exact master changed during slot write")
    return post_write, {
        "period_kind": period_kind,
        "period_index": period_index,
        "pre_write_master": pre_write,
        "post_write_master": post_write,
        "actors": actors,
        "actor_receipts_sha256": design65.canonical_sha256_v65(actors),
    }


def _read_only_slot_v65b(trainer, v40a, period_kind, period_index, edge):
    actors = v40a._rpc_all(
        trainer, "read_only_exact_master_slot_v65b",
        (period_kind, period_index, edge),
    )
    if (
        len(actors) != 4
        or any(
            not isinstance(receipt, dict)
            or receipt.get("schema") != "read-only-exact-master-slot-v65b"
            or receipt.get("period_kind") != period_kind
            or not _exact_int_v65b(receipt.get("period_index"), period_index)
            or receipt.get("edge") != edge
            or receipt.get("master_identity", {}).get("sha256")
            != design52.MASTER_SHA256_V52
            or design65.canonical_sha256_v65(receipt.get("master_identity"))
            != analysis65b.v65a.MASTER_IDENTITY_OBJECT_SHA256_V65A
            or not _exact_int_v65b(receipt.get("runtime_view_count"), 82)
            or not _exact_int_v65b(
                receipt.get("runtime_elements"), 4_921_344,
            )
            or receipt.get("runtime_dtype") != "torch.bfloat16"
            or receipt.get("runtime_values_sha256")
            != design52.MASTER_RUNTIME_SHA256_V52
            or not _exact_active_lora_one_v65b(
                receipt.get("active_lora_ids")
            )
            or not _exact_active_lora_one_v65b(
                receipt.get("active_manager_cache_lora_ids")
            )
            or receipt.get("base_identity", {}).get("inventory_sha256")
            != analysis65b.v65a.BASE_INVENTORY_SHA256_V65A
            or receipt.get("transaction_state_quiescent") is not True
            or receipt.get("slot_read_only_no_weight_write_or_reset") is not True
            for receipt in actors
        )
    ):
        raise RuntimeError("v65b read-only live slot receipt changed")
    actors = _bind_actor_receipts_v65b(trainer, actors)
    identities = [receipt["master_identity"] for receipt in actors]
    inventories = [
        receipt["base_identity"]["inventory_sha256"] for receipt in actors
    ]
    if (
        len({design65.canonical_sha256_v65(value) for value in identities}) != 1
        or len(set(inventories)) != 1
    ):
        raise RuntimeError("v65b read-only actor consensus changed")
    aggregate = {
        "schema": "v65b-read-only-four-actor-master-slot-consensus",
        "canonical_fp32_master_sha256": design52.MASTER_SHA256_V52,
        "canonical_master_identity_sha256": design65.canonical_sha256_v65(
            identities[0]
        ),
        "bf16_runtime_values_sha256": design52.MASTER_RUNTIME_SHA256_V52,
        "runtime_view_count_per_actor": 82,
        "runtime_elements_per_actor": 4_921_344,
        "runtime_dtype": "torch.bfloat16",
        "base_inventory_sha256": inventories[0],
        "four_actor_exact_read_only_consensus": True,
    }
    _validate_read_aggregate_v65b(aggregate)
    return aggregate, {
        "period_kind": period_kind,
        "period_index": period_index,
        "edge": edge,
        "aggregate": aggregate,
        "actors": actors,
        "actor_receipts_sha256": design65.canonical_sha256_v65(actors),
    }


def _validate_state_receipts_v65b(warmup, scored, expected, reads) -> None:
    _validate_master_certificate_v65b(expected)
    if not isinstance(reads, list) or len(reads) != 160:
        raise RuntimeError("v65b state/read linkage coverage changed")
    ordinal = 0
    for values, kind, count in (
        (warmup, "unscored_warmup", 8), (scored, "scored", 72),
    ):
        if not isinstance(values, list) or len(values) != count:
            raise RuntimeError("v65b state-receipt coverage changed")
        for index, receipt in enumerate(values):
            if (
                not isinstance(receipt, dict)
                or set(receipt) != {
                    "period_kind", "period_index", "before", "after",
                    "identical_v434_state",
                }
                or receipt.get("period_kind") != kind
                or not _exact_int_v65b(receipt.get("period_index"), index)
                or _validate_read_aggregate_v65b(receipt.get("before"))
                != reads[2 * ordinal].get("aggregate")
                or _validate_read_aggregate_v65b(receipt.get("after"))
                != reads[2 * ordinal + 1].get("aggregate")
                or receipt.get("before") != receipt.get("after")
                or receipt.get("identical_v434_state") is not True
            ):
                raise RuntimeError("v65b state receipt changed")
            ordinal += 1


def _validate_period_receipts_v65b(
    writes, reads, installed, actor_identities,
) -> None:
    _validate_master_certificate_v65b(installed)
    coordinates = [
        (kind, index)
        for kind, count in (("unscored_warmup", 8), ("scored", 72))
        for index in range(count)
    ]
    if not isinstance(writes, list) or len(writes) != 80:
        raise RuntimeError("v65b slot-write receipt coverage changed")
    for receipt, (kind, index) in zip(writes, coordinates, strict=True):
        actors = receipt.get("actors") if isinstance(receipt, dict) else None
        if (
            not isinstance(receipt, dict)
            or set(receipt) != {
                "period_kind", "period_index", "pre_write_master",
                "post_write_master", "actors", "actor_receipts_sha256",
            }
            or receipt.get("period_kind") != kind
            or not _exact_int_v65b(receipt.get("period_index"), index)
            or _validate_master_certificate_v65b(
                receipt.get("pre_write_master")
            ) != installed
            or _validate_master_certificate_v65b(
                receipt.get("post_write_master")
            ) != installed
            or not isinstance(actors, list) or len(actors) != 4
            or receipt.get("actor_receipts_sha256")
            != design65.canonical_sha256_v65(actors)
            or any(
                set(actor) != {
                    "schema", "period_kind", "period_index",
                    "master_identity", "materialization", "base_identity",
                    "transaction_state_quiescent", "timing",
                    *CONTROLLER_ACTOR_BINDING_KEYS_V65B,
                    *INTRINSIC_WORKER_IDENTITY_KEYS_V65B,
                }
                or actor.get("schema") != "exact-master-slot-write-v65b"
                or actor.get("period_kind") != kind
                or not _exact_int_v65b(actor.get("period_index"), index)
                or not _controller_actor_binding_valid_v65b(
                    actor, actor_identities, rank,
                )
                or not _valid_worker_timing_v65b(actor.get("timing"))
                for rank, actor in enumerate(actors) if isinstance(actor, dict)
            )
            or any(not isinstance(actor, dict) for actor in actors)
            or len({
                tuple(actor["timing"][key] for key in (
                    "started_ns", "ended_ns", "elapsed_ns",
                )) for actor in actors
            }) != 4
        ):
            raise RuntimeError("v65b ordered slot-write receipt changed")
    edges = [
        (kind, index, edge)
        for kind, index in coordinates
        for edge in ("before_generation", "after_generation")
    ]
    if not isinstance(reads, list) or len(reads) != 160:
        raise RuntimeError("v65b read-only receipt coverage changed")
    for receipt, (kind, index, edge) in zip(reads, edges, strict=True):
        actors = receipt.get("actors") if isinstance(receipt, dict) else None
        aggregate = receipt.get("aggregate") if isinstance(receipt, dict) else None
        if (
            not isinstance(receipt, dict)
            or set(receipt) != {
                "period_kind", "period_index", "edge", "aggregate", "actors",
                "actor_receipts_sha256",
            }
            or receipt.get("period_kind") != kind
            or not _exact_int_v65b(receipt.get("period_index"), index)
            or receipt.get("edge") != edge
            or _validate_read_aggregate_v65b(aggregate) != aggregate
            or not isinstance(actors, list) or len(actors) != 4
            or receipt.get("actor_receipts_sha256")
            != design65.canonical_sha256_v65(actors)
            or any(
                not isinstance(actor, dict)
                or set(actor) != {
                    "schema", "period_kind", "period_index", "edge",
                    "master_identity", "runtime_view_count",
                    "runtime_elements", "runtime_dtype",
                    "runtime_values_sha256", "active_lora_ids",
                    "active_manager_cache_lora_ids", "base_identity",
                    "transaction_state_quiescent",
                    "slot_read_only_no_weight_write_or_reset", "timing",
                    *CONTROLLER_ACTOR_BINDING_KEYS_V65B,
                    *INTRINSIC_WORKER_IDENTITY_KEYS_V65B,
                }
                or actor.get("schema") != "read-only-exact-master-slot-v65b"
                or actor.get("period_kind") != kind
                or not _exact_int_v65b(actor.get("period_index"), index)
                or actor.get("edge") != edge
                or not _controller_actor_binding_valid_v65b(
                    actor, actor_identities, rank,
                )
                or actor.get("slot_read_only_no_weight_write_or_reset") is not True
                or not _valid_worker_timing_v65b(actor.get("timing"))
                for rank, actor in enumerate(actors)
            )
            or len({
                tuple(actor["timing"][key] for key in (
                    "started_ns", "ended_ns", "elapsed_ns",
                )) for actor in actors
            }) != 4
        ):
            raise RuntimeError("v65b ordered read-only receipt changed")
    for ordinal, write in enumerate(writes):
        before = reads[2 * ordinal]
        after = reads[2 * ordinal + 1]
        for rank in range(4):
            write_timing = write["actors"][rank]["timing"]
            before_timing = before["actors"][rank]["timing"]
            after_timing = after["actors"][rank]["timing"]
            if (
                write_timing["ended_ns"] > before_timing["started_ns"]
                or before_timing["ended_ns"] > after_timing["started_ns"]
            ):
                raise RuntimeError("v65b within-period receipt timing changed")
    previous_ended = [None] * 4
    seen_timings = [set() for _ in range(4)]
    for ordinal, write in enumerate(writes):
        operations = (write, reads[2 * ordinal], reads[2 * ordinal + 1])
        for operation in operations:
            for rank, actor in enumerate(operation["actors"]):
                timing = actor["timing"]
                identity = (
                    timing["started_ns"], timing["ended_ns"],
                    timing["elapsed_ns"],
                )
                if (
                    identity in seen_timings[rank]
                    or previous_ended[rank] is not None
                    and timing["started_ns"] <= previous_ended[rank]
                ):
                    raise RuntimeError(
                        "v65b global period-edge receipt timing changed"
                    )
                seen_timings[rank].add(identity)
                previous_ended[rank] = timing["ended_ns"]


def build_evidence_v65b(
    *, panel, input_receipt, actor_identities, worker_identities,
    active_lora_receipts, installations, installed_master,
    warmup_state_receipts, scored_state_receipts, slot_write_receipts,
    read_only_slot_receipts, final_master_state, scored_periods,
    actor_wait_timeout_receipt,
) -> dict:
    numeric = analysis65b.validate_scored_periods_v65b(scored_periods)
    if numeric.shape != (64, 72, 4, 3):
        raise RuntimeError("v65b evidence coverage changed")
    if not _canonical_equal_v65b(
        actor_wait_timeout_receipt, ACTOR_WAIT_TIMEOUT_CONTRACT_V65B,
    ):
        raise RuntimeError("v65b fixed timeout receipt changed")
    base65a._validate_active_lora_receipts_value_v65a(active_lora_receipts)
    base65a._validate_installations_v65a(installations)
    if (
        _validate_master_certificate_v65b(final_master_state)
        != _validate_master_certificate_v65b(installed_master)
    ):
        raise RuntimeError("v65b final exact V434 master state changed")
    _validate_period_receipts_v65b(
        slot_write_receipts, read_only_slot_receipts, installed_master,
        actor_identities,
    )
    _validate_state_receipts_v65b(
        warmup_state_receipts, scored_state_receipts, installed_master,
        read_only_slot_receipts,
    )
    value = {
        "schema": "v65b-ranking64-high-rep-generation-evidence",
        "status": "complete_fixed_fresh_warmup_and_72_period_characterization",
        "panel_content_sha256": panel["content_sha256_before_self_field"],
        "authorized_input_receipt": input_receipt,
        "canonical_fp32_master_sha256": design52.MASTER_SHA256_V52,
        "bf16_runtime_values_sha256": design52.MASTER_RUNTIME_SHA256_V52,
        "row_count": 64,
        "actor_count": 4,
        "unscored_warmup_period_count": 8,
        "scored_period_count": 72,
        "adjacent_block_count": 36,
        "four_period_superblock_count": 18,
        "paired_replicas_per_unit": 144,
        "label_plan": dict(analysis65b.LABEL_PLAN_V65B),
        "schedule": analysis65b.validate_schedule_v65b(),
        "runtime_determinism_controls": dict(RUNTIME_CONTROLS_V65B),
        "fixed_actor_wait_timeout": actor_wait_timeout_receipt,
        "actor_wait_timeout_observed": False,
        "actor_runtime_identities": actor_identities,
        "worker_runtime_identities": worker_identities,
        "active_lora_receipts": active_lora_receipts,
        "active_lora_receipts_sha256": design65.canonical_sha256_v65(
            active_lora_receipts
        ),
        "initial_installations": installations,
        "installed_master_state": installed_master,
        "final_restored_master_state": final_master_state,
        "warmup_state_receipts": warmup_state_receipts,
        "warmup_state_receipts_sha256": design65.canonical_sha256_v65(
            warmup_state_receipts
        ),
        "scored_state_receipts": scored_state_receipts,
        "scored_state_receipts_sha256": design65.canonical_sha256_v65(
            scored_state_receipts
        ),
        "exact_master_slot_write_receipts": slot_write_receipts,
        "exact_master_slot_write_receipts_sha256": (
            design65.canonical_sha256_v65(slot_write_receipts)
        ),
        "read_only_live_slot_receipts": read_only_slot_receipts,
        "read_only_live_slot_receipts_sha256": (
            design65.canonical_sha256_v65(read_only_slot_receipts)
        ),
        "scored_periods": scored_periods,
        "numeric_scored_periods_sha256": design65.canonical_sha256_v65(
            scored_periods
        ),
        "fresh_process_warmup_state_from_r1_transferred": False,
        "warmup_generation_completions_discarded": 2_048,
        "scored_generation_completions": 18_432,
        "total_generation_completions": 20_480,
        "generation_only": True,
        "warmup_raw_outputs_persisted": False,
        "warmup_generation_metrics_computed_or_persisted": False,
        "adaptive_retry_drop_reorder_or_early_stop_performed": False,
        "timeout_retry_drop_reorder_or_early_stop_performed": False,
        "alpha": 0.0,
        "sigma_or_direction": None,
        "adapter_update_candidate_hpo_or_projection_performed": False,
        "holdback_sentinel_reserve_ood_terminal_or_protected_opened": False,
        "raw_question_answer_prompt_or_generation_text_persisted": False,
        "v65_population_launch_authorized": False,
    }
    return base65a._self_hashed_v65a(value)


def gpu_preflight_with_fixed_watchdog_v65b(v40a) -> dict:
    with _fixed_sigalrm_watchdog_v65b(
        FIXED_GPU_PREFLIGHT_WATCHDOG_SECONDS_V65B,
        "GPU preflight watchdog",
    ), _fixed_subprocess_timeout_v65b():
        return v40a.gpu_preflight()


def _execute_v65b_guarded(
    preregistration: dict, args, termination_state: dict,
) -> int:
    import run_lora_es_generation_boundary_v48b as v48b

    prior = v48b.v43i
    v40a = prior.v40a
    validate_preregistered_runtime_contract_v65b(preregistration)
    expectation = runtime64.base_model_artifact_expectation_v64()
    pre_model_receipt = runtime64.verify_base_model_artifacts_v64(expectation)
    claim_fresh_run_directory_v65b(termination_state)
    try:
        preflight = gpu_preflight_with_fixed_watchdog_v65b(v40a)
    except BaseException as error:
        termination_state["cleanup_started"] = True
        publish_failure_if_absent_v65b(base65a._self_hashed_v65a({
            "schema": "v65b-fixed-gpu-preflight-failure",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "type": type(error).__name__,
            "message_sha256": hashlib.sha256(
                str(error).encode("utf-8")
            ).hexdigest(),
            "fixed_actor_wait_timeout": dict(ACTOR_WAIT_TIMEOUT_CONTRACT_V65B),
            "model_ray_or_gpu_loaded": False,
            "nvidia_smi_subprocess_timeout_applied": True,
            "raw_error_message_or_traceback_persisted": False,
            "v65_population_launch_authorized": False,
        }))
        raise
    attempt = base65a._self_hashed_v65a({
        "schema": "v65b-ranking64-high-rep-alpha-zero-attempt",
        "status": "launching_fixed_high_rep_exact64_calibration_only",
        "phase": "before_authorized_semantics_model_ray_or_gpu_load",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_file_sha256": args.preregistration_sha256,
        "preregistration_content_sha256": args.preregistration_content_sha256,
        "runtime_determinism_controls": dict(RUNTIME_CONTROLS_V65B),
        "base_model_artifact_receipt": pre_model_receipt,
        "preflight": preflight,
        "fixed_unscored_warmup_periods": 8,
        "fixed_scored_periods": 72,
        "fixed_adjacent_blocks": 36,
        "fixed_paired_replicas_per_unit": 144,
        "fresh_process_warmup_state_from_r1_transferred": False,
        "gpu_hardware_health_monitor_required": True,
        "forbidden_clock_event_reason_mask": FORBIDDEN_CLOCK_REASON_MASK_V65B,
        "fixed_actor_wait_timeout": dict(ACTOR_WAIT_TIMEOUT_CONTRACT_V65B),
        "model_loaded_or_gpu_compute_started": False,
        "candidate_hpo_update_projection_or_protected_access": False,
        "v65_population_launch_authorized": False,
    })
    runtime61a.atomic_json_v61a(ATTEMPT, attempt)
    trainer = monitor = saved = None
    stop = threading.Event()
    monitor_ready = threading.Event()
    failures: queue.Queue = queue.Queue()
    phase = v40a.Phase()
    started = time.monotonic()
    cleanup = idle = pool_cleanup = None
    construction_state = {}
    monitor_stop_timeout_observed = False
    try:
        adapter_contract = runtime52.verify_adapter_contract_v52()
        panel, _panel61c = runtime65._load_hash_only_panel_v65(preregistration)
        v40a.base.set_seed(runtime65.GENERATION_SEED_V65)
        trainer, saved = make_trainer_v65b(
            preregistration, prior, construction_state,
        )
        raise_if_termination_requested_v65b(termination_state)
        actor_wait_timeout_receipt = install_fixed_actor_wait_timeout_v65b(
            trainer,
        )
        actor_identities = trainer._resolve([
            engine.runtime_identity_v65a.remote() for engine in trainer.engines
        ])
        pid_map = base65a.validate_actor_identities_v65a(
            actor_identities,
            preregistration["runtime"]["tuned_table_content_sha256"],
        )
        install_edge_actor_bindings_v65b(trainer, actor_identities)
        worker_identities = v40a._rpc_all(trainer, "runtime_identity_v40a")
        if v40a.validate_identities(actor_identities, worker_identities) != pid_map:
            raise RuntimeError("v65b actor/worker identity mapping changed")
        monitor = threading.Thread(
            target=monitor_gpus_v65b,
            args=(stop, phase, pid_map, GPU_LOG, failures, monitor_ready),
            daemon=True,
        )
        monitor.start()
        if not monitor_ready.wait(GPU_MONITOR_READY_TIMEOUT_SECONDS_V65B):
            raise TimeoutError("v65b GPU health monitor readiness expired")
        _raise_monitor_failure_v65b(failures)
        requests, params, answers, input_receipt = (
            runtime65.prepare_ranking_requests_v65(trainer, prior, panel)
        )
        if input_receipt.get("request_prompt_token_ids_sha256") != (
            preregistration.get("access_contract", {}).get(
                "request_prompt_token_ids_sha256"
            )
        ):
            raise RuntimeError("v65b submitted prompt-token identity changed")
        request = lora_request_v65b(prior)
        if (
            getattr(request, "lora_name", None)
            != "v434_ranking64_alpha_zero_v65b"
            or not _exact_int_v65b(getattr(request, "lora_int_id", None), 1)
            or Path(getattr(request, "lora_path", "")).resolve()
            != design52.STAGED_V52
        ):
            raise RuntimeError("v65b LoRA request identity changed")
        input_receipt.update({
            "submitted_request_batch_size": 64,
            "runtime_determinism_controls": dict(RUNTIME_CONTROLS_V65B),
            "lora_adapter_request_name": "v434_ranking64_alpha_zero_v65b",
            "lora_adapter_request_id": 1,
            "lora_adapter_request_path": str(design52.STAGED_V52),
        })
        phase.value = "activate_v434_lora_slot_all_actors"
        if v40a._rpc_all(trainer, "add_lora", (request,)) != [True] * 4:
            raise RuntimeError("v65b four-actor LoRA activation failed")
        phase.value = "install_exact_v434_master_all_actors"
        installations = v40a._rpc_all(
            trainer, "install_adapter_state_v41a", (
                str(design52.SOURCE_WEIGHTS_V52),
                str(design52.SOURCE_CONFIG_V52),
                design52.SOURCE_WEIGHTS_SHA256_V52,
                design52.SOURCE_CONFIG_SHA256_V52,
            ),
        )
        base65a._validate_installations_v65a(installations)
        installed = runtime61c._assert_v434_certificates_v61c(
            v40a._rpc_all(trainer, "adapter_state_certificate_v52")
        )
        active_lora_receipts = base65a._active_lora_receipts_v65a(
            trainer, v40a,
        )

        warmup_state_receipts = []
        scored_state_receipts = []
        slot_write_receipts = []
        read_only_slot_receipts = []
        for period_index in range(8):
            phase.value = (
                f"unscored_warmup_{period_index}_exact_master_slot_write"
            )
            _write_certificate, write_receipt = _slot_write_v65b(
                trainer, v40a, "unscored_warmup", period_index,
            )
            slot_write_receipts.append(write_receipt)
            before, read_receipt = _read_only_slot_v65b(
                trainer, v40a, "unscored_warmup", period_index,
                "before_generation",
            )
            read_only_slot_receipts.append(read_receipt)
            _raise_monitor_failure_v65b(failures)
            raise_if_termination_requested_v65b(termination_state)
            phase.value = (
                f"unscored_warmup_{period_index}_generation_all_actors"
            )
            batches = base65a._validate_batches_v65a(
                resolve_generation_with_health_v65b(trainer, [
                    engine.generate.remote(
                        requests, params, use_tqdm=False, lora_request=request,
                    ) for engine in trainer.engines
                ], failures, termination_state),
                "unscored_warmup",
            )
            _raise_monitor_failure_v65b(failures)
            raise_if_termination_requested_v65b(termination_state)
            phase.value = (
                f"unscored_warmup_{period_index}_post_generation_integrity"
            )
            after, read_receipt = _read_only_slot_v65b(
                trainer, v40a, "unscored_warmup", period_index,
                "after_generation",
            )
            read_only_slot_receipts.append(read_receipt)
            warmup_state_receipts.append(base65a._state_receipt_v65a(
                "unscored_warmup", period_index, before, after, installed,
            ))
            del batches

        scored_periods = []
        for period_index in range(72):
            phase.value = f"scored_period_{period_index}_exact_master_slot_write"
            _write_certificate, write_receipt = _slot_write_v65b(
                trainer, v40a, "scored", period_index,
            )
            slot_write_receipts.append(write_receipt)
            before, read_receipt = _read_only_slot_v65b(
                trainer, v40a, "scored", period_index, "before_generation",
            )
            read_only_slot_receipts.append(read_receipt)
            _raise_monitor_failure_v65b(failures)
            raise_if_termination_requested_v65b(termination_state)
            phase.value = f"scored_period_{period_index}_generation_all_actors"
            batches = base65a._validate_batches_v65a(
                resolve_generation_with_health_v65b(trainer, [
                    engine.generate.remote(
                        requests, params, use_tqdm=False, lora_request=request,
                    ) for engine in trainer.engines
                ], failures, termination_state),
                "scored",
            )
            _raise_monitor_failure_v65b(failures)
            raise_if_termination_requested_v65b(termination_state)
            phase.value = (
                f"scored_period_{period_index}_post_generation_integrity"
            )
            after, read_receipt = _read_only_slot_v65b(
                trainer, v40a, "scored", period_index, "after_generation",
            )
            read_only_slot_receipts.append(read_receipt)
            phase.value = f"scored_period_{period_index}_numeric_reduction"
            scored_periods.append(base65a._score_period_v65a(
                panel["items"], answers, batches,
            ))
            del batches
            scored_state_receipts.append(base65a._state_receipt_v65a(
                "scored", period_index, before, after, installed,
            ))
            phase.value = f"scored_period_{period_index}_complete"

        phase.value = "final_exact_master_restoration_certificate"
        final_master_state = runtime61c._assert_v434_certificates_v61c(
            v40a._rpc_all(trainer, "adapter_state_certificate_v52")
        )
        evidence = build_evidence_v65b(
            panel=panel, input_receipt=input_receipt,
            actor_identities=actor_identities,
            worker_identities=worker_identities,
            active_lora_receipts=active_lora_receipts,
            installations=installations, installed_master=installed,
            warmup_state_receipts=warmup_state_receipts,
            scored_state_receipts=scored_state_receipts,
            slot_write_receipts=slot_write_receipts,
            read_only_slot_receipts=read_only_slot_receipts,
            final_master_state=final_master_state,
            scored_periods=scored_periods,
            actor_wait_timeout_receipt=actor_wait_timeout_receipt,
        )
        stop.set()
        monitor.join(timeout=10)
        _raise_monitor_failure_v65b(failures)
        if monitor.is_alive():
            raise RuntimeError("v65b GPU health monitor did not stop")
        gpu = v40a.summarize_gpu(GPU_LOG, pid_map)
        gpu_phases = gpu_period_phase_summary_v65b(GPU_LOG, pid_map)
        pool_cleanup = bounded_reward_pool_shutdown_v65b(trainer)
        import ray

        with _fixed_sigalrm_watchdog_v65b(
            FIXED_CLEANUP_AND_IDLE_WATCHDOG_SECONDS_V65B,
            "strict cleanup and four-GPU idle proof",
        ), _fixed_subprocess_timeout_v65b():
            cleanup = v40a.cleanup_v38a.strict_close_trainer_v38a(trainer)
            ray.shutdown()
            idle = v40a.cleanup_v38a.wait_for_gpu_idle()
        trainer = None
        base65a._validate_postrun_integrity_v65a(
            gpu, cleanup, idle, pid_map,
        )
        post_prefix_receipt = base65a._authorized_prefix_postrun_receipt_v65a()
        post_model_receipt = runtime64.verify_base_model_artifacts_v64(expectation)
        if post_model_receipt != pre_model_receipt:
            raise RuntimeError("v65b base-model bytes changed during calibration")
        post_adapter_contract = runtime52.verify_adapter_contract_v52()
        if post_adapter_contract != adapter_contract:
            raise RuntimeError("v65b adapter bytes changed during calibration")
        evidence = base65a._self_hashed_v65a({
            **evidence,
            "adapter_artifact_contract_prelaunch": adapter_contract,
            "adapter_artifact_contract_postcleanup": post_adapter_contract,
            "adapter_artifact_contract_unchanged": True,
        })
        null_analysis = analysis65b.analyze_scored_periods_v65b(scored_periods)
        null_analysis = base65a._self_hashed_v65a({
            **null_analysis,
            "source_evidence_content_sha256": evidence[
                "content_sha256_before_self_field"
            ],
        })
        runtime61a.atomic_json_v61a(EVIDENCE, evidence)
        runtime61a.atomic_json_v61a(ANALYSIS, null_analysis)
        raise_if_termination_requested_v65b(termination_state)
        gate = null_analysis["required_alpha_zero_gate"]
        completed_at_utc = datetime.now(timezone.utc).isoformat()
        wall_runtime_seconds = time.monotonic() - started
        report = base65a._self_hashed_v65a({
            "schema": "v65b-ranking64-high-rep-alpha-zero-report",
            "status": (
                "complete_gate_passed_population_still_unauthorized"
                if gate["passed"] else "complete_gate_failed_closed"
            ),
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": completed_at_utc,
            "wall_runtime_seconds": wall_runtime_seconds,
            "preregistration_file_sha256": args.preregistration_sha256,
            "preregistration_content_sha256": args.preregistration_content_sha256,
            "attempt": {
                "path": str(ATTEMPT),
                "file_sha256": design65.file_sha256_v65(ATTEMPT),
                "content_sha256": attempt["content_sha256_before_self_field"],
            },
            "evidence": {
                "path": str(EVIDENCE),
                "file_sha256": design65.file_sha256_v65(EVIDENCE),
                "content_sha256": evidence["content_sha256_before_self_field"],
            },
            "analysis": {
                "path": str(ANALYSIS),
                "file_sha256": design65.file_sha256_v65(ANALYSIS),
                "content_sha256": null_analysis[
                    "content_sha256_before_self_field"
                ],
                "required_alpha_zero_gate": gate,
            },
            "runtime_determinism_controls": dict(RUNTIME_CONTROLS_V65B),
            "fixed_actor_wait_timeout": actor_wait_timeout_receipt,
            "actor_wait_timeout_observed": False,
            "actor_runtime_identities": actor_identities,
            "base_model_prelaunch_artifact_receipt": pre_model_receipt,
            "base_model_postrun_artifact_receipt": post_model_receipt,
            "adapter_artifact_contract_prelaunch": adapter_contract,
            "adapter_artifact_contract_postcleanup": post_adapter_contract,
            "adapter_artifact_contract_unchanged": True,
            "authorized_prefix_postrun_receipt": post_prefix_receipt,
            "gpu_activity": gpu,
            "gpu_period_phases_and_hardware_health": gpu_phases,
            "gpu_log_file_sha256": design65.file_sha256_v65(GPU_LOG),
            "bounded_reward_pool_cleanup": pool_cleanup,
            "cleanup": cleanup,
            "final_gpu_idle": idle,
            "warmup_generation_completions_discarded": 2_048,
            "scored_generation_completions": 18_432,
            "total_generation_completions": 20_480,
            "raw_question_answer_prompt_or_generation_text_persisted": False,
            "candidate_hpo_update_projection_or_promotion_performed": False,
            "holdback_sentinel_reserve_ood_terminal_or_protected_opened": False,
            "fresh_process_warmup_state_from_r1_transferred": False,
            "thermal_or_hardware_slowdown_observed_during_generation": False,
            "v65_population_launch_authorized": False,
        })
        # From this point onward no signal or broken console may demote a
        # fully cleaned, durable run into a contradictory failure claim.
        termination_state["cleanup_started"] = True
        runtime61a.atomic_json_v61a(REPORT, report)
        best_effort_json_print_v65b({
            "report_file_sha256": design65.file_sha256_v65(REPORT),
            "report_content_sha256": report[
                "content_sha256_before_self_field"
            ],
            "evidence_file_sha256": design65.file_sha256_v65(EVIDENCE),
            "analysis_file_sha256": design65.file_sha256_v65(ANALYSIS),
            "required_alpha_zero_gate_passed": gate["passed"],
            "all_eighty_generation_phases_positive_on_all_four_gpus": True,
            "thermal_or_hardware_slowdown_observations_in_generation": 0,
            "v65_population_launch_authorized": False,
        })
        return 0
    except BaseException as error:
        if termination_state is not None:
            termination_state["cleanup_started"] = True
        stop.set()
        monitor_cleanup_errors = []
        if monitor is not None:
            monitor.join(timeout=10)
            if monitor.is_alive():
                monitor_stop_timeout_observed = True
                monitor_cleanup_errors.append(TimeoutError(
                    "v65b fixed failure-path GPU monitor stop expired"
                ))
        if trainer is None:
            trainer = construction_state.get("trainer")
        if saved is None:
            saved = construction_state.get("saved")
        partial_reward_pool = construction_state.get("partial_reward_pool")
        if (
            trainer is not None
            and getattr(trainer, "mp_pool", None) is None
            and partial_reward_pool is not None
        ):
            trainer.mp_pool = partial_reward_pool
        bounded_cleanup_errors = list(monitor_cleanup_errors)
        try:
            observed_pool_cleanup = bounded_reward_pool_shutdown_v65b(trainer)
            if pool_cleanup is None:
                pool_cleanup = observed_pool_cleanup
        except BaseException as cleanup_error:
            bounded_cleanup_errors.append(cleanup_error)
        observed_cleanup = observed_idle = None
        partial_pool_cleanup = {
            "attempted_after_incomplete_strict_cleanup": True,
            "pool_found": False,
            "terminate_succeeded": False,
            "join_succeeded": False,
        }
        cleanup_errors = []
        try:
            with _fixed_sigalrm_watchdog_v65b(
                FIXED_CLEANUP_AND_IDLE_WATCHDOG_SECONDS_V65B,
                "failure cleanup and four-GPU idle proof",
            ), _fixed_subprocess_timeout_v65b():
                (
                    observed_cleanup, observed_idle,
                    partial_pool_cleanup, cleanup_errors,
                ) = base65a._failure_cleanup_v65a(
                    v40a, trainer, strict_already_complete=cleanup is not None,
                )
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        cleanup_errors = bounded_cleanup_errors + cleanup_errors
        if observed_cleanup is not None:
            cleanup = observed_cleanup
        if observed_idle is not None:
            idle = observed_idle
        timeout_observed = (
            isinstance(error, TimeoutError)
            or any(isinstance(item, TimeoutError) for item in cleanup_errors)
        )
        actor_wait_timeout_observed = (
            isinstance(error, V65BActorWaitTimeout)
            or any(
                isinstance(item, V65BActorWaitTimeout)
                for item in cleanup_errors
            )
        )
        trainer = None
        construction_state["trainer"] = None
        publish_failure_if_absent_v65b(base65a._self_hashed_v65a({
            "schema": "v65b-ranking64-high-rep-alpha-zero-failure",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "type": type(error).__name__,
            "message_sha256": hashlib.sha256(
                str(error).encode("utf-8")
            ).hexdigest(),
            "cleanup": cleanup,
            "bounded_reward_pool_cleanup": pool_cleanup,
            "partial_constructor_pool_cleanup": partial_pool_cleanup,
            "final_gpu_idle": idle,
            "cleanup_errors": base65a._sanitized_errors_v65a(cleanup_errors),
            "ray_shutdown_attempted_even_without_complete_trainer": True,
            "four_gpu_idle_proof_attempted": True,
            "gpu_monitor_stop_timeout_observed": monitor_stop_timeout_observed,
            "raw_error_message_or_traceback_persisted": False,
            "runtime_determinism_controls": dict(RUNTIME_CONTROLS_V65B),
            "fixed_actor_wait_timeout": dict(
                ACTOR_WAIT_TIMEOUT_CONTRACT_V65B
            ),
            "actor_wait_timeout_observed": actor_wait_timeout_observed,
            "timeout_invalidated_calibration": timeout_observed,
            "adaptive_retry_drop_reorder_or_early_stop_performed": False,
            "candidate_hpo_update_projection_or_promotion_performed": False,
            "holdback_sentinel_reserve_ood_terminal_or_protected_opened": False,
            "raw_question_answer_prompt_or_generation_text_persisted": False,
            "v65_population_launch_authorized": False,
        }))
        if cleanup_errors:
            raise RuntimeError(
                "v65b failed and strict four-GPU cleanup also failed"
            ) from cleanup_errors[0]
        raise
    finally:
        if trainer is not None:
            try:
                with _fixed_sigalrm_watchdog_v65b(
                    FIXED_CLEANUP_AND_IDLE_WATCHDOG_SECONDS_V65B,
                    "final trainer close",
                ):
                    v40a.base.close_trainer(trainer)
            except BaseException:
                pass
        try:
            import ray

            with _fixed_sigalrm_watchdog_v65b(
                FIXED_FINAL_RAY_SHUTDOWN_WATCHDOG_SECONDS_V65B,
                "final unconditional Ray shutdown",
            ):
                ray.shutdown()
        except BaseException:
            pass
        if saved is not None:
            v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION = saved


def execute_v65b(preregistration: dict, args) -> int:
    """Protect the entire live path, including preflight and attempt write."""
    termination_state = install_fixed_termination_handlers_v65b()
    try:
        return _execute_v65b_guarded(
            preregistration, args, termination_state,
        )
    except BaseException as error:
        termination_state["cleanup_started"] = True
        if termination_state.get("run_directory_claimed"):
            try:
                publish_failure_if_absent_v65b(
                    base65a._self_hashed_v65a({
                        "schema": "v65b-early-live-path-failure",
                        "failed_at_utc": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "type": type(error).__name__,
                        "message_sha256": hashlib.sha256(
                            str(error).encode("utf-8")
                        ).hexdigest(),
                        "raw_error_message_or_traceback_persisted": False,
                        "runtime_determinism_controls": dict(
                            RUNTIME_CONTROLS_V65B
                        ),
                        "v65_population_launch_authorized": False,
                    })
                )
            except BaseException as publication_error:
                # REPORT/FINALIZED legitimately outrank a late failure.  Any
                # other publication problem remains attached to the primary
                # nonzero failure without replacing it.
                if not REPORT.exists() and not FINALIZED.exists():
                    error.add_note(
                        "early failure publication also failed: "
                        f"{type(publication_error).__name__}"
                    )
        raise
    finally:
        restore_termination_handlers_v65b(termination_state)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    preregistration = load_preregistration_v65b(args)
    if args.dry_run:
        if args.execute:
            raise RuntimeError("v65b dry-run and execute are mutually exclusive")
        print(json.dumps({
            "schema": preregistration["schema"],
            "content_sha256": preregistration[
                "content_sha256_before_self_field"
            ],
            "authorized_semantic_rows_opened": 0,
            "model_ray_or_gpu_loaded": False,
            "filesystem_writes": False,
            "unscored_warmup_periods": 8,
            "scored_periods": 72,
            "paired_replicas_per_unit": 144,
            "candidate_hpo_update_projection_or_protected_access": False,
            "v65_population_launch_authorized": False,
        }, sort_keys=True))
        return 0
    if not args.execute:
        raise RuntimeError("v65b live path requires --execute")
    if os.environ.get("VLLM_BATCH_INVARIANT") not in (None, "0"):
        raise RuntimeError("v65b requires VLLM_BATCH_INVARIANT absent or false")
    runtime52.require_live_interpreter_v52()
    return execute_v65b(preregistration, args)


def sanitized_cli_v65b(argv: list[str] | None = None) -> int:
    """Emit only numeric/hash-safe failure metadata at the CLI boundary."""
    try:
        return main(argv)
    except (Exception, V65BTerminationRequested) as error:
        payload = {
            "status": "failed",
            "type": type(error).__name__,
            "message_sha256": hashlib.sha256(
                str(error).encode("utf-8")
            ).hexdigest(),
            "raw_error_message_or_traceback_persisted": False,
        }
        try:
            sys.stderr.write(json.dumps(payload, sort_keys=True) + "\n")
            sys.stderr.flush()
        except OSError:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(sanitized_cli_v65b())
