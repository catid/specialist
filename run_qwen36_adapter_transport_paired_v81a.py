#!/usr/bin/env python3
"""Prospective systems-only paired runner contract for V81A transport.

This module contains no model or dataset import.  It validates the future
V73C phase-evidence launch gate, constructs a four-pair counterbalanced
schedule, validates compact arm receipts, and analyzes them.  The current V81A
preregistration does not authorize execution; a later additive launcher must
inject an arm executor only after the gate is sealed.

Quality/HPO promotion is deliberately out of scope.  The quarantined V1
evaluation source is rejected recursively.  A separate V2-bound successor is
required if the systems arm earns a quality comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
PREREGISTRATION = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_adapter_transport_cuda_v81a.json"
)
PAIR_ROOT = ROOT / (
    "experiments/eggroll_es_hpo/runs/"
    "v81a_adapter_transport_counterbalanced"
)

CONTROL_ARM_V81A = "control_v71_v72_sync_temporary_device"
CHALLENGER_ARM_V81A = "challenger_v81a_pinned_direct"
ARMS_V81A = (CONTROL_ARM_V81A, CHALLENGER_ARM_V81A)
PHYSICAL_GPU_IDS_V81A = (0, 1, 2, 3)
PAIR_ORDER_V81A = (
    (CONTROL_ARM_V81A, CHALLENGER_ARM_V81A),
    (CHALLENGER_ARM_V81A, CONTROL_ARM_V81A),
    (CONTROL_ARM_V81A, CHALLENGER_ARM_V81A),
    (CHALLENGER_ARM_V81A, CONTROL_ARM_V81A),
)
MINIMUM_PAIR_COUNT_V81A = 3
RUNTIME_VIEWS_V81A = 82
RUNTIME_BYTES_V81A = 9_842_688
MAXIMUM_RUNTIME_VIEW_BYTES_V81A = 524_288

GATE_SCHEMA_V81A = "qwen36-adapter-transport-systems-live-gate-v81a"
ARM_RECEIPT_SCHEMA_V81A = "qwen36-adapter-transport-arm-receipt-v81a"
REPORT_SCHEMA_V81A = "qwen36-adapter-transport-paired-report-v81a"

FORBIDDEN_V1_SURFACES_V81A = (
    "data/eval_qa_v3.jsonl",
    "recipe_evaluation_compute_contract_v1.json",
    "recipe_evaluation_contract_v1.py",
    "specialist-recipe-evaluation-compute-contract-v1",
)


def canonical_sha256_v81a(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def file_sha256_v81a(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_v81a(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha_v81a(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"v81a {label} is not a lowercase SHA-256")
    return value


def _git_commit_v81a(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"v81a {label} is not a full lowercase Git commit")
    return value


def _exact_int_v81a(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RuntimeError(f"v81a {label} must be an integer >= {minimum}")
    return value


def _recursive_strings_v81a(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _recursive_strings_v81a(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _recursive_strings_v81a(item)


def reject_quarantined_v1_surface_v81a(value: Any) -> None:
    for text in _recursive_strings_v81a(value):
        normalized = text.replace("\\", "/")
        if any(surface in normalized for surface in FORBIDDEN_V1_SURFACES_V81A):
            raise RuntimeError("v81a systems runner resolved quarantined V1")


def _without_self_v81a(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def load_preregistration_v81a(
    path: Path = PREREGISTRATION,
    *,
    expected_file_sha256: str | None = None,
    expected_content_sha256: str | None = None,
) -> dict[str, Any]:
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="ascii"))
    claimed = value.get("content_sha256_before_self_field")
    _sha_v81a(claimed, "preregistration content identity")
    _require_v81a(
        canonical_sha256_v81a(_without_self_v81a(value)) == claimed,
        "v81a preregistration self-hash changed",
    )
    if expected_file_sha256 is not None:
        _require_v81a(
            file_sha256_v81a(path) == _sha_v81a(
                expected_file_sha256, "preregistration file identity"
            ),
            "v81a preregistration file identity changed",
        )
    if expected_content_sha256 is not None:
        _require_v81a(
            claimed == _sha_v81a(
                expected_content_sha256, "preregistration expected content identity"
            ),
            "v81a preregistration content identity changed",
        )
    _require_v81a(
        value.get("schema")
        == "qwen36-adapter-transport-cuda-preregistration-v81a"
        and value.get("authority", {}).get("gpu_launch") is False
        and value.get("authority", {}).get("quality_or_hpo_promotion") is False,
        "v81a current preregistration authority changed",
    )
    reject_quarantined_v1_surface_v81a(value)
    return value


def validate_bound_regular_file_v81a(binding: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(binding, Mapping):
        raise RuntimeError("v81a evidence binding is not a mapping")
    path = Path(str(binding.get("path", ""))).resolve()
    expected = _sha_v81a(binding.get("file_sha256"), "evidence file identity")
    _require_v81a(
        path.is_file() and not path.is_symlink(),
        f"v81a sealed evidence is absent or non-regular: {path}",
    )
    _require_v81a(
        file_sha256_v81a(path) == expected,
        f"v81a sealed evidence changed: {path}",
    )
    return {"path": str(path), "file_sha256": expected}


def validate_launch_gate_v81a(
    gate: Mapping[str, Any],
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(gate, Mapping):
        raise RuntimeError("v81a launch gate must be a mapping")
    reject_quarantined_v1_surface_v81a(gate)
    claimed = _sha_v81a(
        gate.get("content_sha256_before_self_field"), "launch gate identity"
    )
    _require_v81a(
        canonical_sha256_v81a(_without_self_v81a(gate)) == claimed,
        "v81a launch gate self-hash changed",
    )
    prereg_claimed = preregistration["content_sha256_before_self_field"]
    _require_v81a(
        gate.get("schema") == GATE_SCHEMA_V81A
        and gate.get("status") == "sealed_after_v73c_phase_evidence"
        and gate.get("gpu_launch_authorized") is True
        and gate.get("systems_only_nonpromotable") is True
        and gate.get("quality_evaluation_authorized") is False
        and gate.get("hpo_or_recipe_promotion_authorized") is False
        and gate.get("checkpoint_or_model_update_authorized") is False
        and gate.get("protected_content_access_authorized") is False
        and gate.get("quarantined_v1_resolved") is False
        and gate.get("v2_bound_successor_required_for_quality") is True,
        "v81a launch gate widened systems-only authority",
    )
    prereg_binding = gate.get("preregistration")
    _require_v81a(
        isinstance(prereg_binding, Mapping)
        and prereg_binding.get("content_sha256") == prereg_claimed
        and _sha_v81a(
            prereg_binding.get("file_sha256"), "gate preregistration file identity"
        ) == file_sha256_v81a(PREREGISTRATION),
        "v81a launch gate binds another preregistration",
    )
    source_commit = _git_commit_v81a(gate.get("source_commit"), "source commit")
    current_head = os.environ.get("SPECIALIST_V81A_EXPECTED_SOURCE_COMMIT")
    if current_head is not None:
        _require_v81a(
            current_head == source_commit,
            "v81a launch environment source commit changed",
        )

    phase = gate.get("v73c_phase_evidence")
    _require_v81a(
        isinstance(phase, Mapping)
        and phase.get("bead") == "specialist-0j5.32"
        and phase.get("complete") is True
        and phase.get("timeline_profile_passed") is True
        and phase.get("all_four_physical_gpus") == list(PHYSICAL_GPU_IDS_V81A)
        and phase.get("all_exact_phases_nonoverlapping") is True
        and phase.get("materialize_restore_update_abort_cleanup_covered") is True
        and phase.get("final_idle") is True
        and phase.get("protected_content_opened") is False
        and phase.get("profiled_timing_used_as_speed_claim") is False,
        "v81a launch lacks sealed V73C phase evidence",
    )
    bindings = phase.get("bindings")
    _require_v81a(
        isinstance(bindings, list) and len(bindings) >= 3,
        "v81a V73C evidence binding set is incomplete",
    )
    bound = [validate_bound_regular_file_v81a(item) for item in bindings]
    _require_v81a(
        gate.get("pair_schedule") == [list(order) for order in PAIR_ORDER_V81A]
        and gate.get("physical_gpu_ids") == list(PHYSICAL_GPU_IDS_V81A),
        "v81a launch gate pair/GPU schedule changed",
    )
    return {
        "schema": "qwen36-adapter-transport-validated-live-gate-v81a",
        "gate_content_sha256": claimed,
        "source_commit": source_commit,
        "v73c_bound_files": bound,
        "systems_only_nonpromotable": True,
        "quarantined_v1_resolved": False,
    }


def prospective_schedule_v81a() -> list[dict[str, Any]]:
    rows = []
    for pair_index, order in enumerate(PAIR_ORDER_V81A):
        pair_id = canonical_sha256_v81a({
            "schema": "qwen36-adapter-transport-pair-id-v81a",
            "pair_index": pair_index,
            "order": list(order),
            "physical_gpu_ids": list(PHYSICAL_GPU_IDS_V81A),
        })
        for order_position, arm in enumerate(order):
            rows.append({
                "pair_index": pair_index,
                "pair_id": pair_id,
                "order_position": order_position,
                "arm": arm,
                "physical_gpu_ids": list(PHYSICAL_GPU_IDS_V81A),
                "run_directory": str(
                    PAIR_ROOT / f"pair_{pair_index}" / arm
                ),
            })
    return rows


def _positive_int_vector_v81a(
    value: Any, expected_length: int, label: str
) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != expected_length
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        raise RuntimeError(f"v81a {label} vector changed")
    return value


def validate_arm_receipt_v81a(
    receipt: Mapping[str, Any],
    schedule_row: Mapping[str, Any],
    gate_identity: str,
) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise RuntimeError("v81a arm receipt is not a mapping")
    reject_quarantined_v1_surface_v81a(receipt)
    claimed = _sha_v81a(
        receipt.get("content_sha256_before_self_field"), "arm receipt identity"
    )
    _require_v81a(
        canonical_sha256_v81a(_without_self_v81a(receipt)) == claimed,
        "v81a arm receipt self-hash changed",
    )
    _require_v81a(
        receipt.get("schema") == ARM_RECEIPT_SCHEMA_V81A
        and receipt.get("pair_index") == schedule_row["pair_index"]
        and receipt.get("pair_id") == schedule_row["pair_id"]
        and receipt.get("order_position") == schedule_row["order_position"]
        and receipt.get("arm") == schedule_row["arm"]
        and receipt.get("gate_content_sha256") == gate_identity
        and receipt.get("physical_gpu_ids") == list(PHYSICAL_GPU_IDS_V81A)
        and receipt.get("systems_only_nonpromotable") is True
        and receipt.get("protected_content_opened") is False
        and receipt.get("checkpoint_model_update_or_promotion_performed") is False,
        "v81a arm coordinate or authority changed",
    )
    exact = receipt.get("exact_lifecycle")
    _require_v81a(
        isinstance(exact, Mapping)
        and exact.get("v71_exact_audits_passed") is True
        and exact.get("v72_one_two_one_ownership_passed") is True
        and exact.get("same_live_dual_compiler_exact") is True
        and exact.get("reward_values_cross_run_bitwise_oracle") is False
        and exact.get("update_aborted_without_commit") is True
        and exact.get("restore_or_poison_passed") is True
        and exact.get("final_master_exact") is True
        and exact.get("final_runtime_exact") is True,
        "v81a exact V71/V72/V73B lifecycle receipt changed",
    )
    for field in (
        "population_plan_sha256",
        "candidate_projection_set_sha256",
        "candidate_audit_set_sha256",
        "same_live_reward_sha256",
        "same_live_update_sha256",
        "final_master_sha256",
        "final_runtime_sha256",
    ):
        _sha_v81a(exact.get(field), f"arm {field}")

    transport = receipt.get("transport")
    transitions = _exact_int_v81a(
        transport.get("transition_count") if isinstance(transport, Mapping) else None,
        "transition count",
        1,
    )
    expected_pinned = schedule_row["arm"] == CHALLENGER_ARM_V81A
    _require_v81a(
        transport.get("runtime_view_count") == RUNTIME_VIEWS_V81A
        and transport.get("h2d_bytes_per_transition") == RUNTIME_BYTES_V81A
        and transport.get("h2d_calls_per_transition") == RUNTIME_VIEWS_V81A
        and transport.get("temporary_device_publication_staging_bytes_per_transition")
        == (0 if expected_pinned else MAXIMUM_RUNTIME_VIEW_BYTES_V81A)
        and transport.get("maximum_view_bytes")
        == MAXIMUM_RUNTIME_VIEW_BYTES_V81A
        and transport.get("d2d_bytes_per_transition")
        == (0 if expected_pinned else RUNTIME_BYTES_V81A)
        and transport.get("pinned_host_bank_count") == (1 if expected_pinned else 0)
        and transport.get("pinned_host_bank_bytes")
        == (RUNTIME_BYTES_V81A if expected_pinned else 0)
        and transport.get("actual_bank_is_pinned") is expected_pinned
        and transport.get("event_fenced_before_audit_and_generation")
        is expected_pinned,
        "v81a arm transport byte/event ledger changed",
    )

    measurements = receipt.get("measurements")
    _require_v81a(
        isinstance(measurements, Mapping)
        and measurements.get("phase_source") == "sealed_v73c_method"
        and measurements.get("all_four_useful_gpu_attribution") is True
        and measurements.get("profiler_overhead_used_as_speed_claim") is False,
        "v81a phase measurement provenance changed",
    )
    by_gpu = measurements.get("by_gpu")
    _require_v81a(
        isinstance(by_gpu, Mapping)
        and set(by_gpu) == {str(item) for item in PHYSICAL_GPU_IDS_V81A},
        "v81a measurement GPU coverage changed",
    )
    for gpu in PHYSICAL_GPU_IDS_V81A:
        item = by_gpu[str(gpu)]
        _positive_int_vector_v81a(
            item.get("transition_elapsed_ns"), transitions,
            f"GPU {gpu} transition elapsed",
        )
        for field in (
            "peak_allocated_vram_bytes",
            "peak_reserved_vram_bytes",
            "pcie_rx_bytes",
            "pcie_tx_bytes",
            "useful_cuda_event_elapsed_ns",
        ):
            _exact_int_v81a(item.get(field), f"GPU {gpu} {field}", 1)
        hbm = item.get("hbm_activity")
        _require_v81a(
            item.get("physical_gpu_id") == gpu
            and _exact_int_v81a(item.get("actor_pid"), "actor PID", 1) > 0
            and isinstance(hbm, Mapping)
            and hbm.get("metric_kind") in {
                "tool_defined_bandwidth_sample",
                "exact_integrated_bytes",
            }
            and isinstance(hbm.get("value"), (int, float))
            and not isinstance(hbm.get("value"), bool)
            and math.isfinite(float(hbm["value"]))
            and float(hbm["value"]) > 0
            and hbm.get("units") in {"bytes", "tool_defined_units"}
            and hbm.get("exact_bytes")
            is (hbm.get("metric_kind") == "exact_integrated_bytes"),
            "v81a actor/GPU useful-work attribution changed",
        )
    cleanup = receipt.get("cleanup")
    _require_v81a(
        isinstance(cleanup, Mapping)
        and cleanup.get("all_four_actor_final_idle") is True
        and cleanup.get("all_four_gpu_process_lists_empty") is True
        and cleanup.get("ray_shutdown") is True
        and cleanup.get("pinned_banks_released") is expected_pinned
        and cleanup.get("failure") is None,
        "v81a arm cleanup receipt changed",
    )
    return dict(receipt)


def _geometric_mean_v81a(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise RuntimeError("v81a geometric mean input changed")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def analyze_paired_receipts_v81a(
    receipts: Sequence[Mapping[str, Any]],
    gate_identity: str,
) -> dict[str, Any]:
    schedule = prospective_schedule_v81a()
    _require_v81a(
        len(PAIR_ORDER_V81A) >= MINIMUM_PAIR_COUNT_V81A
        and len(receipts) == len(schedule),
        "v81a paired receipt coverage changed",
    )
    validated = [
        validate_arm_receipt_v81a(receipt, row, gate_identity)
        for receipt, row in zip(receipts, schedule, strict=True)
    ]
    transition_ratios = []
    hbm_ratios = []
    pcie_ratios = []
    peak_vram_ratios = []
    pair_receipts = []
    for pair_index in range(len(PAIR_ORDER_V81A)):
        pair = [
            item for item in validated if item["pair_index"] == pair_index
        ]
        _require_v81a(len(pair) == 2, "v81a pair arm coverage changed")
        by_arm = {item["arm"]: item for item in pair}
        _require_v81a(
            set(by_arm) == set(ARMS_V81A), "v81a pair arm identity changed"
        )
        control = by_arm[CONTROL_ARM_V81A]
        challenger = by_arm[CHALLENGER_ARM_V81A]
        control_exact = control["exact_lifecycle"]
        challenger_exact = challenger["exact_lifecycle"]
        # Cross-run reward floats are deliberately excluded.  Deterministic
        # plan/projection/final identities remain exact paired oracles.
        paired_fields = (
            "population_plan_sha256",
            "candidate_projection_set_sha256",
            "final_master_sha256",
            "final_runtime_sha256",
        )
        _require_v81a(
            all(control_exact[field] == challenger_exact[field]
                for field in paired_fields),
            "v81a deterministic paired lifecycle identity changed",
        )
        _require_v81a(
            control["transport"]["transition_count"]
            == challenger["transport"]["transition_count"],
            "v81a paired transition count changed",
        )
        for gpu in PHYSICAL_GPU_IDS_V81A:
            a = control["measurements"]["by_gpu"][str(gpu)]
            b = challenger["measurements"]["by_gpu"][str(gpu)]
            control_elapsed = statistics.fmean(a["transition_elapsed_ns"])
            challenger_elapsed = statistics.fmean(b["transition_elapsed_ns"])
            transition_ratios.append(control_elapsed / challenger_elapsed)
            _require_v81a(
                a["hbm_activity"]["metric_kind"]
                == b["hbm_activity"]["metric_kind"]
                and a["hbm_activity"]["units"]
                == b["hbm_activity"]["units"],
                "v81a paired HBM activity units changed",
            )
            hbm_ratios.append(
                float(a["hbm_activity"]["value"])
                / float(b["hbm_activity"]["value"])
            )
            pcie_ratios.append(a["pcie_rx_bytes"] / b["pcie_rx_bytes"])
            peak_vram_ratios.append(
                b["peak_reserved_vram_bytes"] / a["peak_reserved_vram_bytes"]
            )
        pair_receipts.append({
            "pair_index": pair_index,
            "order": list(PAIR_ORDER_V81A[pair_index]),
            "deterministic_exact_fields": list(paired_fields),
            "arm_local_same_live_reward_hashes": {
                arm: by_arm[arm]["exact_lifecycle"]["same_live_reward_sha256"]
                for arm in ARMS_V81A
            },
            "cross_run_reward_float_equality_required": False,
        })

    transition_geomean = _geometric_mean_v81a(transition_ratios)
    hbm_geomean = _geometric_mean_v81a(hbm_ratios)
    pcie_geomean = _geometric_mean_v81a(pcie_ratios)
    peak_vram_geomean = _geometric_mean_v81a(peak_vram_ratios)
    systems_candidate_pass = (
        transition_geomean > 1.0
        and hbm_geomean > 1.0
        and pcie_geomean >= 1.0
        and peak_vram_geomean <= 1.0
    )
    summary = {
        "schema": REPORT_SCHEMA_V81A,
        "status": "complete_systems_only_nonpromotable",
        "pair_count": len(PAIR_ORDER_V81A),
        "pair_order": [list(order) for order in PAIR_ORDER_V81A],
        "physical_gpu_ids": list(PHYSICAL_GPU_IDS_V81A),
        "gate_content_sha256": _sha_v81a(gate_identity, "analysis gate identity"),
        "pairs": pair_receipts,
        "systems_metrics": {
            "control_over_challenger_transition_geomean": transition_geomean,
            "control_over_challenger_hbm_activity_geomean": hbm_geomean,
            "control_over_challenger_pcie_rx_geomean": pcie_geomean,
            "challenger_over_control_peak_reserved_vram_geomean": (
                peak_vram_geomean
            ),
            "h2d_bytes_equal": True,
            "h2d_calls_equal": True,
            "challenger_d2d_bytes_zero": True,
            "challenger_temporary_device_publication_staging_bytes_zero": True,
            "systems_candidate_pass": systems_candidate_pass,
            "systems_pass_rules": {
                "transition_time_strict_improvement": True,
                "HBM_activity_strict_improvement_same_units": True,
                "PCIe_RX_no_increase": True,
                "peak_reserved_VRAM_no_increase": True,
            },
        },
        "decision_authority": {
            "systems_signal_complete": True,
            "direct_worker_or_recipe_promotion_authorized": False,
            "quality_or_hpo_claim_authorized": False,
            "checkpoint_or_model_update_authorized": False,
            "v2_bound_quality_successor_may_be_preregistered": (
                systems_candidate_pass
            ),
            "quarantined_v1_resolved": False,
        },
    }
    summary["content_sha256_before_self_field"] = canonical_sha256_v81a(summary)
    return summary


def execute_prospective_schedule_v81a(
    gate: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    arm_executor: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Run the frozen order with an additive, gate-authorized arm executor.

    The current CLI never supplies an executor.  This injection point is for a
    later committed V73C-bound launcher and is tested with CPU-only fakes.
    """
    validated_gate = validate_launch_gate_v81a(gate, preregistration)
    if not callable(arm_executor):
        raise TypeError("v81a prospective arm executor must be callable")
    receipts = []
    for row in prospective_schedule_v81a():
        run_directory = Path(row["run_directory"])
        _require_v81a(
            not run_directory.exists(),
            f"v81a prospective run path is not fresh: {run_directory}",
        )
        receipt = arm_executor({
            **row,
            "gate_content_sha256": validated_gate["gate_content_sha256"],
            "systems_only_nonpromotable": True,
            "quarantined_v1_resolved": False,
        })
        receipts.append(validate_arm_receipt_v81a(
            receipt, row, validated_gate["gate_content_sha256"]
        ))
    return receipts


def _load_receipts_v81a(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for item in prospective_schedule_v81a():
        path = Path(directory) / f"pair_{item['pair_index']}" / (
            f"{item['order_position']}_{item['arm']}.json"
        )
        rows.append(json.loads(path.read_text(encoding="ascii")))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--preregistration", default=str(PREREGISTRATION))
    parser.add_argument("--preregistration-file-sha256")
    parser.add_argument("--preregistration-content-sha256")
    parser.add_argument("--gate")
    parser.add_argument("--receipts")
    args = parser.parse_args(argv)
    preregistration = load_preregistration_v81a(
        Path(args.preregistration),
        expected_file_sha256=args.preregistration_file_sha256,
        expected_content_sha256=args.preregistration_content_sha256,
    )
    if args.dry_run:
        print(json.dumps({
            "schema": "qwen36-adapter-transport-prospective-schedule-v81a",
            "status": "blocked_until_additive_v73c_evidence_gate",
            "schedule": prospective_schedule_v81a(),
            "systems_only_nonpromotable": True,
            "quality_requires_separate_sealed_v2_successor": True,
            "quarantined_v1_resolved": False,
            "model_ray_gpu_or_protected_access": False,
        }, sort_keys=True))
        return 0
    if args.execute:
        raise RuntimeError(
            "v81a current artifact cannot execute; a committed additive V73C-"
            "bound arm executor and live gate are required"
        )
    if args.gate is None or args.receipts is None:
        raise ValueError("v81a analysis requires --gate and --receipts")
    gate = json.loads(Path(args.gate).read_text(encoding="ascii"))
    validated = validate_launch_gate_v81a(gate, preregistration)
    report = analyze_paired_receipts_v81a(
        _load_receipts_v81a(Path(args.receipts)),
        validated["gate_content_sha256"],
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
