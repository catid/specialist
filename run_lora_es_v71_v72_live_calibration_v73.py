#!/usr/bin/env python3
"""Live-only V71/V72 calibration adapter over the accepted V66d flow.

Import and ``--dry-run`` are CPU-only.  ``--execute`` retains V66d's sealed
train-only schedule and actor CUDA/GPU telemetry while inserting V71's exact
candidate/population/update acceptance barriers and V72's one-master state
receipts.  It never commits, checkpoints, promotes, or opens protected data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import build_lora_es_v71_v72_live_calibration_preregistration_v73 as builder
import eggroll_es_audit_contract_v71 as audit_v71
import eggroll_es_host_state_contract_v72 as host_v72
import eggroll_es_worker_lora_v66 as state_v66
import run_lora_es_mirrored_calibration_v66 as v66
import run_lora_es_mirrored_calibration_v66d as v66d
import train_eggroll_es_specialist as base


ROOT = Path(__file__).resolve().parent
PREREGISTRATION = builder.OUTPUT
RUN_DIR = builder.RUN
_ARTIFACTS = builder.artifacts_v73()
ATTEMPT = Path(_ARTIFACTS["attempt"]).resolve()
GPU_LOG = Path(_ARTIFACTS["gpu_log"]).resolve()
GPU_WORK_LOG = Path(_ARTIFACTS["actor_cuda_work_log"]).resolve()
HOST_SAMPLES = Path(_ARTIFACTS["host_process_samples"]).resolve()
HOST_SUMMARY = Path(_ARTIFACTS["host_process_summary"]).resolve()
POPULATION = Path(_ARTIFACTS["population"]).resolve()
UPDATE = Path(_ARTIFACTS["update"]).resolve()
AUDIT_TRAFFIC = Path(_ARTIFACTS["audit_traffic"]).resolve()
EQUIVALENCE = Path(_ARTIFACTS["equivalence"]).resolve()
REPORT = Path(_ARTIFACTS["report"]).resolve()
FAILURE = Path(_ARTIFACTS["failure"]).resolve()

WORKER_EXTENSION_V73 = (
    "eggroll_es_worker_lora_v72.LoRAAdapterStateWorkerExtensionV72"
)
REQUIRED_WORKER_ENDPOINTS_V73 = tuple(
    builder.build_preregistration_v73()["runtime"][
        "required_worker_endpoints"
    ]
)
NOISE_PROTOCOL_V73 = state_v66.NOISE_PROTOCOL_V66
WORLD_SIZE_V73 = 4
SIGNED_CANDIDATES_V73 = 16
CANDIDATES_PER_ACTOR_V73 = 4
HOST_SAMPLE_INTERVAL_SECONDS_V73 = 0.5


def artifacts_v73():
    return dict(_ARTIFACTS)


def _write_self_hashed_v73(path: Path, value: dict) -> dict:
    result = dict(value)
    result.pop("content_sha256_before_self_field", None)
    result["content_sha256_before_self_field"] = (
        v66.mirrored.canonical_sha256_v66(result)
    )
    payload = (
        json.dumps(result, ensure_ascii=True, allow_nan=False, indent=2,
                   sort_keys=True) + "\n"
    ).encode("ascii")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    return result


def validate_lora_worker_contract_v73():
    contract = base.validate_worker_state_mode(
        base.TRAINER_STATE_MODE_EXTERNAL_WORKER,
        WORKER_EXTENSION_V73,
        REQUIRED_WORKER_ENDPOINTS_V73,
    )
    extension = contract.pop("resolved_worker_extension")
    contract["resolved_class"] = (
        f"{extension.__module__}.{extension.__qualname__}"
    )
    contract["dense_full_weight_master_install_authorized"] = False
    contract["v71_exact_audit_required"] = True
    contract["v72_one_master_ownership_required"] = True
    return contract


def load_preregistration_v73(args):
    path = Path(args.preregistration).resolve()
    if v66.file_sha256_v66(path) != args.preregistration_sha256:
        raise RuntimeError("v73 preregistration file identity changed")
    value = json.loads(path.read_text(encoding="ascii"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or v66.mirrored.canonical_sha256_v66(compact)
        != args.preregistration_content_sha256
        or value != builder.build_preregistration_v73()
        or value.get("schema")
        != "lora-es-v71-v72-qwen36-live-calibration-preregistration-v73"
        or value.get("status")
        != "sealed_cpu_only_before_v73_train_model_ray_gpu_or_protected_access"
        or value.get("artifacts") != artifacts_v73()
    ):
        raise RuntimeError("v73 preregistration content changed")
    return value


def load_accepted_control_values_v73():
    builder.accepted_v66d_control_v73()
    values = {}
    for name in ("population", "update", "report"):
        path = Path(builder.CONTROL_FILES_V73[name]["path"])
        values[name] = json.loads(path.read_text(encoding="ascii"))
    actor_path = Path(
        builder.CONTROL_FILES_V73["actor_cuda_work_log"]["path"]
    )
    values["actor_rows"] = [
        json.loads(line)
        for line in actor_path.read_text(encoding="ascii").splitlines()
        if line
    ]
    return values


_STATUS_BYTES_V73 = (
    "VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem", "VmLck", "VmPin",
)
_NODE_PATTERN_V73 = re.compile(r"^N([0-9]+)=([0-9]+)$")


def _status_value_v73(value: str, key: str) -> int:
    parts = value.split()
    if key == "Threads":
        if len(parts) != 1:
            raise RuntimeError("v73 process thread count changed")
        return int(parts[0])
    if len(parts) != 2 or parts[1] != "kB":
        raise RuntimeError(f"v73 process status unit changed: {key}")
    return int(parts[0]) * 1024


def process_snapshot_v73(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
    include_numa: bool,
) -> dict:
    pid = int(pid)
    root = Path(proc_root) / str(pid)
    status_raw = (root / "status").read_text(encoding="ascii")
    status = {}
    for line in status_raw.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            status[key] = value.strip()
    required = set(_STATUS_BYTES_V73[:-1]) | {"Threads"}
    if not required.issubset(status):
        raise RuntimeError("v73 required process status fields are absent")
    status_values = {
        key: _status_value_v73(status[key], key)
        for key in _STATUS_BYTES_V73[:-1]
    }
    status_values["VmPin"] = (
        _status_value_v73(status["VmPin"], "VmPin")
        if "VmPin" in status else 0
    )
    status_values["Threads"] = _status_value_v73(
        status["Threads"], "Threads"
    )

    stat_raw = (root / "stat").read_text(encoding="ascii").strip()
    close = stat_raw.rfind(")")
    if close < 0:
        raise RuntimeError("v73 process stat command field changed")
    fields = stat_raw[close + 2:].split()
    if len(fields) < 22:
        raise RuntimeError("v73 process stat field count changed")
    minor_faults = int(fields[7])
    major_faults = int(fields[9])
    if minor_faults < 0 or major_faults < 0:
        raise RuntimeError("v73 process fault counters changed")

    result = {
        "pid": pid,
        "status_bytes": status_values,
        "minor_faults": minor_faults,
        "major_faults": major_faults,
        "numa_included": bool(include_numa),
    }
    if include_numa:
        node_pages = {}
        counters = {}
        lines = 0
        numa_raw = (root / "numa_maps").read_text(encoding="ascii")
        for line in numa_raw.splitlines():
            if not line:
                continue
            lines += 1
            for token in line.split()[1:]:
                match = _NODE_PATTERN_V73.fullmatch(token)
                if match:
                    node = str(int(match.group(1)))
                    node_pages[node] = node_pages.get(node, 0) + int(
                        match.group(2)
                    )
                    continue
                if "=" not in token:
                    continue
                key, raw = token.split("=", 1)
                if key in {"anon", "dirty", "mapped", "active", "mapmax"}:
                    try:
                        counters[key] = counters.get(key, 0) + int(raw)
                    except ValueError:
                        pass
        if lines == 0 or not node_pages:
            raise RuntimeError("v73 NUMA map contained no node placement")
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        result["numa"] = {
            "map_lines": lines,
            "page_size_bytes": page_size,
            "node_pages": dict(sorted(node_pages.items(), key=lambda x: int(x[0]))),
            "node_bytes": {
                key: pages * page_size for key, pages in sorted(
                    node_pages.items(), key=lambda x: int(x[0])
                )
            },
            "counters": dict(sorted(counters.items())),
        }
    return result


def _seal_sample_row_v73(value):
    result = dict(value)
    result["row_sha256"] = v66.mirrored.canonical_sha256_v66(result)
    return result


def summarize_host_rows_v73(rows, bindings):
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("v73 host telemetry rows are absent")
    expected = {
        int(item["actor_rank"]): {
            "actor_rank": int(item["actor_rank"]),
            "worker_pid": int(item["worker_pid"]),
            "physical_gpu_id": int(item["physical_gpu_id"]),
        }
        for item in bindings
    }
    if set(expected) != set(range(WORLD_SIZE_V73)):
        raise RuntimeError("v73 host telemetry actor bindings changed")
    by_rank = {rank: [] for rank in expected}
    prior_index = -1
    for row in rows:
        compact = {key: item for key, item in row.items() if key != "row_sha256"}
        if (
            row.get("row_sha256")
            != v66.mirrored.canonical_sha256_v66(compact)
            or int(row.get("sample_index", -1)) <= prior_index
        ):
            raise RuntimeError("v73 host telemetry row seal/order changed")
        prior_index = int(row["sample_index"])
        binding = row.get("binding")
        rank = int(binding.get("actor_rank", -1)) if isinstance(binding, dict) else -1
        if rank not in expected or binding != expected[rank]:
            raise RuntimeError("v73 host telemetry binding changed")
        by_rank[rank].append(row)
    actors = {}
    all_nodes = set()
    for rank, selected in by_rank.items():
        if not selected:
            raise RuntimeError("v73 host telemetry omitted an actor")
        boundaries = {row["boundary"] for row in selected}
        if not {"monitor_start", "pre_cleanup"}.issubset(boundaries):
            raise RuntimeError("v73 host telemetry missed lifecycle boundaries")
        minors = [row["process"]["minor_faults"] for row in selected]
        majors = [row["process"]["major_faults"] for row in selected]
        if minors != sorted(minors) or majors != sorted(majors):
            raise RuntimeError("v73 process fault counters moved backwards")
        numa_rows = [row for row in selected if row["process"]["numa_included"]]
        if not numa_rows:
            raise RuntimeError("v73 host telemetry omitted NUMA placement")
        nodes = sorted({
            node
            for row in numa_rows
            for node in row["process"]["numa"]["node_pages"]
        }, key=int)
        all_nodes.update(nodes)
        peak_nodes = {
            node: max(
                row["process"]["numa"]["node_pages"].get(node, 0)
                for row in numa_rows
            )
            for node in nodes
        }
        statuses = [row["process"]["status_bytes"] for row in selected]
        actors[str(rank)] = {
            "binding": expected[rank],
            "sample_count": len(selected),
            "numa_sample_count": len(numa_rows),
            "first_monotonic_ns": selected[0]["monotonic_ns"],
            "last_monotonic_ns": selected[-1]["monotonic_ns"],
            "rss_bytes_first": statuses[0]["VmRSS"],
            "rss_bytes_last": statuses[-1]["VmRSS"],
            "rss_bytes_peak_sampled": max(item["VmRSS"] for item in statuses),
            "hwm_bytes_peak_reported": max(item["VmHWM"] for item in statuses),
            "rss_anon_bytes_peak": max(item["RssAnon"] for item in statuses),
            "rss_file_bytes_peak": max(item["RssFile"] for item in statuses),
            "rss_shmem_bytes_peak": max(item["RssShmem"] for item in statuses),
            "locked_bytes_peak": max(item["VmLck"] for item in statuses),
            "pinned_bytes_peak": max(item["VmPin"] for item in statuses),
            "minor_faults_first": minors[0],
            "minor_faults_last": minors[-1],
            "minor_faults_delta": minors[-1] - minors[0],
            "major_faults_first": majors[0],
            "major_faults_last": majors[-1],
            "major_faults_delta": majors[-1] - majors[0],
            "numa_nodes_observed": nodes,
            "peak_pages_by_numa_node": peak_nodes,
        }
    return {
        "schema": "eggroll-es-four-actor-host-process-summary-v73",
        "actor_count": WORLD_SIZE_V73,
        "sample_count": len(rows),
        "all_actor_rank_pid_gpu_bindings_exact": True,
        "all_fault_counters_monotonic": True,
        "numa_nodes_observed": sorted(all_nodes, key=int),
        "actors": actors,
    }


class HostProcessMonitorV73:
    def __init__(self, path, bindings, phase_getter, interval_seconds=0.5):
        self.path = Path(path).resolve()
        self.bindings = [dict(item) for item in bindings]
        self.phase_getter = phase_getter
        self.interval_seconds = float(interval_seconds)
        if self.interval_seconds <= 0.0:
            raise ValueError("v73 host sample interval must be positive")
        self.rows = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._handle = None
        self._failure = None

    def start(self):
        if self._thread is not None or self.path.exists():
            raise RuntimeError("v73 host monitor is not fresh")
        self._handle = self.path.open("xb")
        self.sample_now("monitor_start", include_numa=True)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _phase(self):
        phase, generation = self.phase_getter()
        return str(phase), int(generation)

    def sample_now(self, boundary, *, include_numa):
        if self._failure is not None:
            raise RuntimeError("v73 host monitor previously failed") from self._failure
        with self._lock:
            phase, generation = self._phase()
            captured_ns = time.monotonic_ns()
            for binding in self.bindings:
                process = process_snapshot_v73(
                    binding["worker_pid"], include_numa=include_numa
                )
                row = _seal_sample_row_v73({
                    "schema": "eggroll-es-actor-host-process-sample-v73",
                    "sample_index": len(self.rows),
                    "monotonic_ns": captured_ns,
                    "phase": phase,
                    "phase_generation": generation,
                    "boundary": str(boundary),
                    "binding": dict(binding),
                    "process": process,
                })
                self.rows.append(row)
                payload = (
                    json.dumps(row, ensure_ascii=True, allow_nan=False,
                               separators=(",", ":"), sort_keys=True) + "\n"
                ).encode("ascii")
                self._handle.write(payload)

    def _loop(self):
        while not self._stop.wait(self.interval_seconds):
            try:
                self.sample_now("periodic", include_numa=False)
            except BaseException as error:
                self._failure = error
                self._stop.set()
                return

    def finish(self, extra):
        primary = None
        try:
            self.sample_now("pre_cleanup", include_numa=True)
        except BaseException as error:
            primary = error
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive() and primary is None:
                primary = RuntimeError("v73 host monitor did not stop")
        if self._failure is not None and primary is None:
            primary = RuntimeError("v73 host monitor background sampling failed")
            primary.__cause__ = self._failure
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None
        if primary is not None:
            raise primary
        result = summarize_host_rows_v73(self.rows, self.bindings)
        result.update({
            "sample_interval_seconds": self.interval_seconds,
            "sample_log": {
                "path": str(self.path),
                "file_sha256": v66.file_sha256_v66(self.path),
                "rows": len(self.rows),
            },
            **extra,
        })
        return result


def candidate_audit_consensus_v73(matrices):
    if len(matrices) != WORLD_SIZE_V73:
        raise RuntimeError("v73 candidate audit actor coverage changed")
    candidate_hashes = []
    transition_hashes = []
    by_rank = []
    for rank, receipt in enumerate(matrices):
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema")
            != "eggroll-es-candidate-audit-matrix-v71"
            or receipt.get("candidate_count") != CANDIDATES_PER_ACTOR_V73
            or receipt.get("all_rewards_provisional") is not True
            or len(receipt.get("candidate_audit_sha256", []))
            != CANDIDATES_PER_ACTOR_V73
            or len(receipt.get("transition_audit_sha256", []))
            != CANDIDATES_PER_ACTOR_V73
            or len(set(receipt["candidate_audit_sha256"]))
            != CANDIDATES_PER_ACTOR_V73
        ):
            raise RuntimeError("v73 candidate audit matrix changed")
        candidate_hashes.extend(receipt["candidate_audit_sha256"])
        transition_hashes.extend(receipt["transition_audit_sha256"])
        by_rank.append({
            "rank": rank,
            "candidate_audit_sha256": list(
                receipt["candidate_audit_sha256"]
            ),
            "transition_audit_sha256": list(
                receipt["transition_audit_sha256"]
            ),
        })
    if (
        len(set(candidate_hashes)) != SIGNED_CANDIDATES_V73
        or len(set(transition_hashes)) != SIGNED_CANDIDATES_V73
    ):
        raise RuntimeError("v73 aggregate candidate audit coverage changed")
    result = {
        "schema": "eggroll-es-four-actor-candidate-audit-consensus-v73",
        "actor_count": WORLD_SIZE_V73,
        "candidate_count": SIGNED_CANDIDATES_V73,
        "candidate_count_per_actor": CANDIDATES_PER_ACTOR_V73,
        "all_rewards_provisional": True,
        "by_rank": by_rank,
    }
    result["consensus_sha256"] = v66.mirrored.canonical_sha256_v66(result)
    return result


def population_acceptance_consensus_v73(receipts, audit_consensus):
    if len(receipts) != WORLD_SIZE_V73:
        raise RuntimeError("v73 population acceptance actor coverage changed")
    tokens = []
    compact = []
    for rank, receipt in enumerate(receipts):
        expected = audit_consensus["by_rank"][rank][
            "candidate_audit_sha256"
        ]
        token = receipt.get("acceptance_sha256") if isinstance(receipt, dict) else None
        unsealed = {
            key: item for key, item in receipt.items()
            if key != "acceptance_sha256"
        } if isinstance(receipt, dict) else {}
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema")
            != "eggroll-es-population-reward-acceptance-v71"
            or receipt.get("candidate_audit_sha256") != expected
            or receipt.get("candidate_count") != CANDIDATES_PER_ACTOR_V73
            or receipt.get("rewards_accepted") is not True
            or receipt.get("update_authorized") is not False
            or not isinstance(token, str) or len(token) != 64
            or token != audit_v71.canonical_sha256_v71(unsealed)
        ):
            raise RuntimeError("v73 population acceptance receipt changed")
        tokens.append(token)
        compact.append({
            "rank": rank,
            "acceptance_sha256": token,
            "boundary_audit_sha256": receipt["boundary_audit_sha256"],
        })
    if len(set(tokens)) != WORLD_SIZE_V73:
        raise RuntimeError("v73 rank-local population tokens collapsed")
    return {
        "schema": "eggroll-es-rank-local-population-acceptance-v73",
        "tokens": tokens,
        "by_rank": compact,
        "rank_local_not_broadcast": True,
    }


def validate_residency_receipts_v73(receipts, phase, banks):
    if len(receipts) != WORLD_SIZE_V73:
        raise RuntimeError("v73 residency actor coverage changed")
    expected_bytes = int(banks) * host_v72.MASTER_BYTES_V72
    for receipt in receipts:
        compact = {
            key: item for key, item in receipt.items()
            if key != "receipt_sha256"
        }
        if (
            receipt.get("receipt_sha256")
            != host_v72.canonical_sha256_v72(compact)
            or receipt.get("phase") != phase
            or receipt.get("unique_owned_bank_count") != banks
            or receipt.get("unique_owned_tensor_bytes") != expected_bytes
            or receipt.get("reference_tensor_bank_present") is not False
            or receipt.get("full_state_clone_bytes_for_observation") != 0
        ):
            raise RuntimeError("v73 host-state residency receipt changed")
    return {
        "phase": phase,
        "actor_count": WORLD_SIZE_V73,
        "unique_owned_bank_count_per_actor": banks,
        "unique_owned_tensor_bytes_per_actor": expected_bytes,
        "aggregate_actor_tensor_bytes": expected_bytes * WORLD_SIZE_V73,
        "receipt_sha256_by_rank": [
            receipt["receipt_sha256"] for receipt in receipts
        ],
    }


def validate_audit_traffic_v73(receipts):
    if len(receipts) != WORLD_SIZE_V73:
        raise RuntimeError("v73 audit traffic actor coverage changed")
    expected = {
        "h2d_bytes": 14 * audit_v71.RUNTIME_LORA_BYTES_V71,
        "lora_d2h_bytes": 20 * audit_v71.RUNTIME_LORA_BYTES_V71,
        # Population/update acceptance are trust boundaries; the exact abort
        # additionally hashes the frozen base while proving rollback.
        "base_d2h_bytes": 3 * audit_v71.BASE_BYTES_V71,
        "master_validation_host_copy_bytes": 0,
        "lora_d2h_calls": 20,
        "exact_base_audits": 3,
    }
    expected["total_device_transfer_bytes"] = (
        expected["h2d_bytes"]
        + expected["lora_d2h_bytes"]
        + expected["base_d2h_bytes"]
    )
    compact = []
    reported_models = []
    for rank, receipt in enumerate(receipts):
        observed = receipt.get("observed") if isinstance(receipt, dict) else None
        if (
            receipt.get("schema")
            != "eggroll-es-worker-audit-traffic-receipt-v71"
            or not isinstance(observed, dict)
            or any(observed.get(key) != value for key, value in expected.items())
            or receipt.get("completed_boundaries")
            != ["population_reward_acceptance", "update_acceptance"]
            or not receipt.get("population_acceptance_sha256")
            or receipt.get("terminal_poisoned") is not False
        ):
            raise RuntimeError("v73 exact-audit traffic receipt changed")
        compact.append({
            "rank": rank,
            "observed": observed,
            "completed_boundaries": receipt["completed_boundaries"],
            "population_acceptance_sha256": receipt[
                "population_acceptance_sha256"
            ],
        })
        reported_models.append(receipt.get("byte_accounted_model"))
    if any(model != reported_models[0] for model in reported_models[1:]):
        raise RuntimeError("v73 worker byte-accounted models differ by rank")
    aggregate = {
        key: value * WORLD_SIZE_V73 for key, value in expected.items()
        if key.endswith("bytes") or key.endswith("calls")
        or key.endswith("audits")
    }
    outside_counter_per_actor = {
        # V72 bootstraps the immutable-base registry before the V71 counters
        # exist.  Each of four local candidates copies the FP32 master H2D and
        # the owned candidate D2H.  The one update copies its reduced FP32
        # delta D2H.  These are deterministic code-path bytes, not silently
        # relabelled as observed worker-counter traffic.
        "install_base_bootstrap_d2h_bytes": audit_v71.BASE_BYTES_V71,
        "candidate_fp32_master_h2d_bytes": (
            CANDIDATES_PER_ACTOR_V73 * audit_v71.MASTER_BYTES_V71
        ),
        "candidate_fp32_owned_state_d2h_bytes": (
            CANDIDATES_PER_ACTOR_V73 * audit_v71.MASTER_BYTES_V71
        ),
        "update_reduced_fp32_delta_d2h_bytes": audit_v71.MASTER_BYTES_V71,
    }
    outside_counter_per_actor["total_bytes"] = sum(
        outside_counter_per_actor.values()
    )
    known_total_per_actor = (
        expected["total_device_transfer_bytes"]
        + outside_counter_per_actor["total_bytes"]
    )
    accepted_v66d_audit_d2h = (
        60 * audit_v71.BASE_BYTES_V71
        + 120 * audit_v71.RUNTIME_LORA_BYTES_V71
    )
    v73_audit_d2h = WORLD_SIZE_V73 * (
        expected["base_d2h_bytes"] + expected["lora_d2h_bytes"]
    )
    return {
        "schema": "eggroll-es-four-actor-audit-traffic-v73",
        "per_actor_expected_and_observed": expected,
        "aggregate_expected_and_observed": aggregate,
        "known_code_path_device_transfer_outside_worker_counter_per_actor": (
            outside_counter_per_actor
        ),
        "known_code_path_device_transfer_total": {
            "per_actor_bytes": known_total_per_actor,
            "all_actor_bytes": known_total_per_actor * WORLD_SIZE_V73,
        },
        "accepted_v66d_measured_audit_d2h_comparison": {
            "accepted_v66d_base_plus_lora_d2h_bytes": (
                accepted_v66d_audit_d2h
            ),
            "v73_base_plus_lora_d2h_bytes": v73_audit_d2h,
            "saved_bytes": accepted_v66d_audit_d2h - v73_audit_d2h,
            "saved_fraction": (
                (accepted_v66d_audit_d2h - v73_audit_d2h)
                / accepted_v66d_audit_d2h
            ),
            "comparison_scope": (
                "immutable accepted V66d measured/code-ledger base plus LoRA "
                "D2H lower bound versus exact V73 worker-counter expectation"
            ),
        },
        "worker_reported_full_commit_final_projection_model": (
            reported_models[0]
        ),
        "scope": (
            "worker counters exactly cover V71/V72 LoRA materialization and "
            "exact-audit transfers; deterministic install/candidate/update "
            "state transfers are byte-accounted separately; generation, "
            "all-reduce GPU traffic, and allocator traffic are excluded"
        ),
        "by_rank": compact,
        "exact_match": True,
    }


def compact_population_v73(execution):
    materializations = []
    materialization_times = []
    for receipt in execution["materialization_receipts"]:
        if (
            receipt.get("master_identity", {}).get("sha256")
            != v66.MASTER_SHA256_V66
            or receipt.get("master_validation_clone_bytes") != 0
        ):
            raise RuntimeError("v73 V71 candidate master invariant changed")
        materializations.append({
            "direction_seed": receipt["direction_seed"],
            "sigma": receipt["sigma"],
            "sign": receipt["sign"],
            "pair_id": receipt["pair_id"],
            "evaluation_contract_sha256": receipt[
                "evaluation_contract_sha256"
            ],
            "noise_protocol": NOISE_PROTOCOL_V73,
            "master_unchanged": True,
            "exact_restore_required": receipt["exact_restore_required"],
            "master_sha256": receipt["master_identity"]["sha256"],
            "candidate_sha256": receipt["candidate_identity"]["sha256"],
            "runtime_values_sha256": receipt["materialization"][
                "runtime_values_sha256"
            ],
        })
        materialization_times.append({
            "direction_seed": receipt["direction_seed"],
            "sign": receipt["sign"],
            "elapsed_ns": receipt["timing"]["elapsed_ns"],
        })
    restorations = []
    restore_times = []
    for receipt in execution["restore_receipts"]:
        restorations.append({
            "reason": receipt["reason"],
            "restored_master_sha256": receipt["restored_identity"]["sha256"],
            "runtime_values_sha256": receipt["materialization"][
                "runtime_values_sha256"
            ],
            "algebraic_bf16_restore_used": receipt[
                "algebraic_bf16_restore_used"
            ],
            "terminal_poisoned": receipt["terminal_poisoned"],
        })
        restore_times.append({
            "reason": receipt["reason"],
            "elapsed_ns": receipt["timing"]["elapsed_ns"],
        })
    if len(materializations) != 16 or len(restorations) != 16:
        raise RuntimeError("v73 mirrored population receipt coverage changed")
    if _ACTIVE_CONTEXT_V73 is not None:
        _ACTIVE_CONTEXT_V73.worker_operation_times = {
            "candidate_materialization": materialization_times,
            "exact_restore": restore_times,
        }
    return {
        "schema": "v66-mirrored-qwen36-population-evidence",
        "plan_sha256": execution["plan_sha256"],
        "evaluation_contract_sha256": execution[
            "evaluation_contract_sha256"
        ],
        "signed_rewards": execution["signed_rewards"],
        "materializations": materializations,
        "restorations": restorations,
        "signed_reward_sha256": v66.mirrored.canonical_sha256_v66(
            execution["signed_rewards"]
        ),
        "all_submitted_work_drained": execution[
            "all_submitted_work_drained"
        ],
        "all_four_actors_restored_after_every_wave": execution[
            "all_four_actors_restored_after_every_wave"
        ],
        "raw_questions_answers_or_outputs_persisted": False,
        "protected_dev_ood_or_holdout_opened": False,
    }


_POPULATION_EQUIVALENCE_FIELDS_V73 = (
    "plan_sha256", "evaluation_contract_sha256", "signed_rewards",
    "materializations", "restorations", "signed_reward_sha256",
    "all_submitted_work_drained", "all_four_actors_restored_after_every_wave",
    "input_receipt", "plan", "install_master_consensus_sha256",
    "installation_count", "raw_questions_answers_or_outputs_persisted",
    "protected_dev_ood_or_holdout_opened",
)


def population_equivalence_v73(candidate, control):
    changed = [
        key for key in _POPULATION_EQUIVALENCE_FIELDS_V73
        if candidate.get(key) != control.get(key)
    ]
    if changed:
        raise RuntimeError(
            f"v73 population differs from accepted V66d: {changed}"
        )
    result = {
        "schema": "eggroll-es-v66d-population-equivalence-v73",
        "fields": list(_POPULATION_EQUIVALENCE_FIELDS_V73),
        "candidate_count": len(candidate["signed_rewards"]),
        "candidate_identity_count": len(candidate["materializations"]),
        "reward_bit_exact": True,
        "candidate_and_runtime_identity_exact": True,
        "restore_identity_exact": True,
        "accepted_population_file_sha256": builder.CONTROL_FILES_V73[
            "population"
        ]["file_sha256"],
    }
    result["equivalence_sha256"] = v66.mirrored.canonical_sha256_v66(result)
    return result


_UPDATE_EQUIVALENCE_FIELDS_V73 = (
    "coefficient_l2", "nonzero_pair_differences", "direction_count",
    "worker_alpha", "worker_population_size", "effective_noise_scale",
    "coefficient_sha256", "prepared_rank_shards",
    "candidate_master_sha256", "candidate_runtime_values_sha256",
    "candidate_differs_from_master", "candidate_runtime_differs_from_master",
    "master_committed", "all_four_abort_receipts_exact",
    "checkpoint_snapshot_or_promotion_performed", "plan_sha256",
    "protected_dev_ood_or_holdout_opened",
)


def update_equivalence_v73(candidate, control):
    changed = [
        key for key in _UPDATE_EQUIVALENCE_FIELDS_V73
        if candidate.get(key) != control.get(key)
    ]
    if changed:
        raise RuntimeError(f"v73 update differs from accepted V66d: {changed}")
    result = {
        "schema": "eggroll-es-v66d-update-equivalence-v73",
        "fields": list(_UPDATE_EQUIVALENCE_FIELDS_V73),
        "coefficients_exact": True,
        "prepared_shards_exact": True,
        "candidate_and_runtime_identity_exact": True,
        "abort_semantics_exact": True,
        # V72 installs reference generation 1 and the inherited controller
        # performs one explicit capture, so its manifest metadata generation
        # intentionally differs from the older V66d generation counter.  The
        # coefficients, shards, candidate bytes, runtime bytes, and abort are
        # the equivalence surface; the differing manifest is recorded.
        "manifest_sha256_exact_required": False,
        "accepted_manifest_sha256": control["manifest_sha256"],
        "candidate_manifest_sha256": candidate["manifest_sha256"],
        "manifest_metadata_delta": "v72_reference_generation",
        "accepted_update_file_sha256": builder.CONTROL_FILES_V73[
            "update"
        ]["file_sha256"],
    }
    result["equivalence_sha256"] = v66.mirrored.canonical_sha256_v66(result)
    return result


def actor_work_equivalence_v73(candidate_rows, control_rows):
    fields = (
        "wave_index", "engine_rank", "direction_seed", "sign", "pair_id",
        "evaluation_contract_sha256", "work_id", "output_cardinality",
    )

    def compact(rows):
        if len(rows) != SIGNED_CANDIDATES_V73:
            raise RuntimeError("v73 actor work receipt count changed")
        values = [{key: row.get(key) for key in fields} for row in rows]
        return sorted(values, key=lambda item: item["work_id"])

    candidate = compact(candidate_rows)
    control = compact(control_rows)
    if candidate != control:
        raise RuntimeError("v73 actor work assignments/cardinality differ from V66d")
    result = {
        "schema": "eggroll-es-v66d-actor-work-equivalence-v73",
        "fields": list(fields),
        "receipt_count": SIGNED_CANDIDATES_V73,
        "work_assignment_and_cardinality_exact": True,
        "worker_pid_gpu_and_timing_expected_dynamic": True,
        "semantic_rows_sha256": v66.mirrored.canonical_sha256_v66(candidate),
    }
    result["equivalence_sha256"] = v66.mirrored.canonical_sha256_v66(result)
    return result


def _rpc_all_v73(trainer, method, args=()):
    values = trainer._resolve([
        engine.collective_rpc.remote(method, args=args)
        for engine in trainer.engines
    ])
    if len(values) != WORLD_SIZE_V73 or any(
        not isinstance(value, list) or len(value) != 1 for value in values
    ):
        raise RuntimeError(f"v73 incomplete four-actor RPC: {method}")
    return [value[0] for value in values]


def _rpc_ranked_v73(trainer, method, args_by_rank):
    if len(args_by_rank) != WORLD_SIZE_V73:
        raise RuntimeError(f"v73 rank-local RPC coverage changed: {method}")
    values = trainer._resolve([
        trainer.engines[rank].collective_rpc.remote(
            method, args=tuple(args_by_rank[rank])
        )
        for rank in range(WORLD_SIZE_V73)
    ])
    if len(values) != WORLD_SIZE_V73 or any(
        not isinstance(value, list) or len(value) != 1 for value in values
    ):
        raise RuntimeError(f"v73 incomplete rank-local RPC: {method}")
    return [value[0] for value in values]


class IntegrationContextV73(v66d.LiveContextV66D):
    def __init__(self):
        super().__init__()
        if _PENDING_CONTROL_V73 is None or _PENDING_PREREGISTRATION_V73 is None:
            raise RuntimeError("v73 sealed context inputs are absent")
        self.control = _PENDING_CONTROL_V73
        self.preregistration = _PENDING_PREREGISTRATION_V73
        self.phase = None
        self.trainer = None
        self.host_monitor = None
        self.host_summary = None
        self.host_failure = None
        self.phase_transitions = []
        self.controller_operations = []
        self.worker_operation_times = {}
        self.population_equivalence = None
        self.update_equivalence = None
        self.candidate_audit_consensus = None
        self.population_acceptance = None
        self.update_invariants = None
        self.audit_traffic = None
        self.residency = {}
        self.certificate_calls = 0

    def current_phase(self):
        if self.phase is None:
            return "unbound", 0
        return self.phase.snapshot()

    def attach_phase(self, phase):
        if self.phase is not None and self.phase is not phase:
            raise RuntimeError("v73 phase handshake was replaced")
        self.phase = phase
        self._start_host_monitor_if_ready()

    def attach_runtime(self, trainer, phase):
        self.trainer = trainer
        self.attach_phase(phase)

    def capture_bindings(self, actor, worker, legacy_validator):
        value = super().capture_bindings(actor, worker, legacy_validator)
        self._start_host_monitor_if_ready()
        return value

    def _start_host_monitor_if_ready(self):
        if self.host_monitor is not None or self.phase is None or self.bindings is None:
            return
        self.host_monitor = HostProcessMonitorV73(
            HOST_SAMPLES,
            self.bindings,
            self.current_phase,
            HOST_SAMPLE_INTERVAL_SECONDS_V73,
        )
        self.host_monitor.start()

    def phase_acknowledged(self, before, after, started_ns, ended_ns):
        self.phase_transitions.append({
            "from": str(before),
            "to": str(after),
            "started_ns": int(started_ns),
            "ended_ns": int(ended_ns),
            "elapsed_ns": int(ended_ns - started_ns),
        })
        if self.host_monitor is not None:
            self.host_monitor.sample_now(
                f"phase_ack:{after}", include_numa=True
            )

    def record_operation(self, name, started_ns, ended_ns, **metadata):
        self.controller_operations.append({
            "operation": str(name),
            "started_ns": int(started_ns),
            "ended_ns": int(ended_ns),
            "elapsed_ns": int(ended_ns - started_ns),
            **metadata,
        })

    def capture_host_boundary(self, boundary):
        if self.host_monitor is None:
            raise RuntimeError("v73 host monitor was not started")
        self.host_monitor.sample_now(str(boundary), include_numa=True)

    def finalize_host_before_cleanup(self):
        if self.host_summary is not None or self.host_failure is not None:
            return
        if self.host_monitor is None:
            self.host_failure = RuntimeError("v73 host monitor never started")
            return
        try:
            summary = self.host_monitor.finish({
                "phase_transitions": list(self.phase_transitions),
                "controller_operations": list(self.controller_operations),
                "worker_operation_times": self.worker_operation_times,
                "sampling_stopped_before_actor_cleanup": True,
            })
            self.host_summary = _write_self_hashed_v73(HOST_SUMMARY, summary)
        except BaseException as error:
            self.host_failure = error

    def require_host_summary(self):
        if self.host_failure is not None:
            raise RuntimeError("v73 host telemetry failed") from self.host_failure
        if self.host_summary is None:
            raise RuntimeError("v73 host telemetry summary is absent")
        return self.host_summary


_ACTIVE_CONTEXT_V73: IntegrationContextV73 | None = None
_PENDING_CONTROL_V73: dict | None = None
_PENDING_PREREGISTRATION_V73: dict | None = None


class CompatiblePhaseHandshakeV73(v66d.CompatiblePhaseHandshakeV66D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if _ACTIVE_CONTEXT_V73 is None:
            raise RuntimeError("v73 active context is absent at phase creation")
        _ACTIVE_CONTEXT_V73.attach_phase(self)

    @property
    def value(self):
        return self.snapshot()[0]

    @value.setter
    def value(self, phase):
        current, _ = self.snapshot()
        if phase == current:
            return
        started_ns = time.monotonic_ns()
        self.transition(phase, timeout_seconds=15.0)
        ended_ns = time.monotonic_ns()
        if _ACTIVE_CONTEXT_V73 is None:
            raise RuntimeError("v73 context disappeared during phase transition")
        _ACTIVE_CONTEXT_V73.phase_acknowledged(
            current, phase, started_ns, ended_ns
        )


class RayMirroredCallbacksV73(v66d.RayMirroredCallbacksV66D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if _ACTIVE_CONTEXT_V73 is None:
            raise RuntimeError("v73 active context is absent at callback creation")
        _ACTIVE_CONTEXT_V73.attach_runtime(self.trainer, self.phase)


def _population_accept_then_update_v73(original, plan, signed_rewards, learning_rate):
    context = _ACTIVE_CONTEXT_V73
    if (
        context is None
        or context.trainer is None
        or context.population_equivalence is None
        or context.population_acceptance is not None
    ):
        raise RuntimeError("v73 population acceptance dependency changed")
    context.phase.value = "candidate_audit_matrix_all_actors"
    started = time.monotonic_ns()
    matrices = _rpc_all_v73(
        context.trainer, "candidate_audit_matrix_v71"
    )
    audit_consensus = candidate_audit_consensus_v73(matrices)
    ended = time.monotonic_ns()
    context.record_operation(
        "candidate_audit_matrix", started, ended,
        candidate_count=SIGNED_CANDIDATES_V73,
    )
    context.candidate_audit_consensus = audit_consensus

    context.phase.value = "population_reward_acceptance_all_actors"
    started = time.monotonic_ns()
    acceptances = _rpc_ranked_v73(
        context.trainer,
        "accept_population_rewards_v71",
        [
            (item["candidate_audit_sha256"],)
            for item in audit_consensus["by_rank"]
        ],
    )
    population_acceptance = population_acceptance_consensus_v73(
        acceptances, audit_consensus
    )
    ended = time.monotonic_ns()
    context.record_operation(
        "population_reward_acceptance", started, ended,
        actor_count=WORLD_SIZE_V73,
    )
    context.population_acceptance = population_acceptance
    context.capture_host_boundary("population_rewards_accepted")

    started = time.monotonic_ns()
    update = original(plan, signed_rewards, learning_rate)
    expected = original(
        context.control["population"]["plan"],
        context.control["population"]["signed_rewards"],
        context.preregistration["fixed_recipe"]["learning_rate"],
    )
    if update != expected:
        raise RuntimeError("v73 pair-difference update input differs from V66d")
    ended = time.monotonic_ns()
    context.record_operation("pair_difference_update_math", started, ended)
    return update


def _execute_and_abort_nonzero_update_v73(
    trainer,
    update,
    master_sha,
    reference_generation,
    phase,
):
    context = _ACTIVE_CONTEXT_V73
    if (
        context is None
        or context.population_acceptance is None
        or context.trainer is not trainer
    ):
        raise RuntimeError("v73 update preceded population acceptance")
    coefficient_l2 = math.sqrt(math.fsum(
        value * value for value in update["coefficients"]
    ))
    nonzero_pairs = sum(value != 0.0 for value in update["coefficients"])
    if coefficient_l2 == 0.0 or nonzero_pairs == 0:
        raise RuntimeError("v73 Qwen calibration produced a zero pair update")
    prepared = executed = aborts = None
    manifest_sha = None
    primary_error = None
    try:
        phase.value = "pair_difference_update_accept_prepare_all_actors"
        started = time.monotonic_ns()
        tokens = context.population_acceptance["tokens"]
        common = (
            update["direction_seeds"],
            update["coefficients"],
            update["coefficient_sha256"],
            update["worker_population_size"],
            WORLD_SIZE_V73,
            update["worker_alpha"],
            "v66-qwen36-nonzero-calibration",
            master_sha,
            reference_generation,
        )
        prepared = _rpc_ranked_v73(
            trainer,
            "prepare_sharded_adapter_update_v71",
            [(tokens[rank], *common) for rank in range(WORLD_SIZE_V73)],
        )
        manifests = {item.get("manifest_sha256") for item in prepared}
        if (
            len(manifests) != 1
            or None in manifests
            or {item.get("rank") for item in prepared} != set(range(4))
            or any(
                item.get("population_acceptance_sha256") != tokens[rank]
                or item.get("rollback_aliases_master") is not True
                or item.get("rollback_clone_bytes") != 0
                for rank, item in enumerate(prepared)
            )
        ):
            raise RuntimeError("v73 V72 update prepare consensus changed")
        manifest_sha = next(iter(manifests))
        update_tokens = [
            item.get("update_acceptance_sha256") for item in prepared
        ]
        if (
            any(not isinstance(item, str) or len(item) != 64 for item in update_tokens)
            or len(set(update_tokens)) != WORLD_SIZE_V73
            or any(
                update_tokens[rank] != audit_v71.canonical_sha256_v71({
                    key: value for key, value in prepared[rank].items()
                    if key != "update_acceptance_sha256"
                })
                for rank in range(WORLD_SIZE_V73)
            )
        ):
            raise RuntimeError("v73 rank-local update acceptance tokens changed")
        ended = time.monotonic_ns()
        context.record_operation(
            "update_acceptance_and_prepare", started, ended,
            manifest_sha256=manifest_sha,
        )

        phase.value = "pair_difference_update_execute_all_actors"
        started = time.monotonic_ns()
        executed = _rpc_all_v73(
            trainer, "execute_sharded_adapter_update_v41a", (manifest_sha,)
        )
        candidate_hashes = {
            item.get("candidate_identity", {}).get("sha256") for item in executed
        }
        runtime_hashes = {
            item.get("materialization", {}).get("runtime_values_sha256")
            for item in executed
        }
        control_update = context.control["update"]
        if (
            candidate_hashes != {control_update["candidate_master_sha256"]}
            or runtime_hashes
            != {control_update["candidate_runtime_values_sha256"]}
            or any(
                item.get("master_committed") is not False
                or item.get("full_state_validation_clone_bytes") != 0
                or item.get("pinned_host_staging_used") is not False
                or item.get("max_host_update_staging_bytes")
                > host_v72.MAX_MASTER_TENSOR_BYTES_V72
                for item in executed
            )
        ):
            raise RuntimeError("v73 V72 update candidate differs from V66d")
        residency = _rpc_all_v73(trainer, "host_state_residency_v72")
        context.residency["update_executed"] = validate_residency_receipts_v73(
            residency, "executed_candidate_retained", 2
        )
        context.capture_host_boundary("update_executed_two_banks")
        ended = time.monotonic_ns()
        context.record_operation(
            "update_execute_and_residency", started, ended,
            candidate_master_sha256=next(iter(candidate_hashes)),
        )
        context.update_invariants = {
            "population_acceptance_sha256_by_rank": tokens,
            "update_acceptance_sha256_by_rank": update_tokens,
            "manifest_sha256": manifest_sha,
            "candidate_master_sha256": next(iter(candidate_hashes)),
            "candidate_runtime_values_sha256": next(iter(runtime_hashes)),
            "rank_local_tokens_not_broadcast": True,
        }
    except BaseException as error:
        primary_error = error
    finally:
        try:
            phase.value = "pair_difference_update_abort_all_actors"
            started = time.monotonic_ns()
            aborts = _rpc_all_v73(
                trainer,
                "abort_mirrored_update_if_present_v66",
                (master_sha, "v73_nonzero_calibration_no_commit", manifest_sha),
            )
            ended = time.monotonic_ns()
            context.record_operation(
                "exact_update_abort", started, ended,
                expected_manifest_sha256=manifest_sha,
            )
        except BaseException as abort_error:
            raise RuntimeError(
                "v73 could not prove four-actor exact update abort"
            ) from abort_error
    if aborts is None or any(
        not isinstance(item, dict)
        or item.get("restored_identity", {}).get("sha256") != master_sha
        or item.get("terminal_poisoned") is not False
        for item in aborts
    ):
        raise RuntimeError(
            "v73 update abort master consensus changed after primary failure"
        ) from primary_error
    if primary_error is not None:
        raise primary_error
    residency = _rpc_all_v73(trainer, "host_state_residency_v72")
    context.residency["post_abort"] = validate_residency_receipts_v73(
        residency, "quiescent_one_master", 1
    )
    context.capture_host_boundary("post_abort_one_bank")
    return {
        "schema": "v73-nonzero-qwen36-pair-difference-update-receipt",
        "coefficient_l2": coefficient_l2,
        "nonzero_pair_differences": nonzero_pairs,
        "direction_count": len(update["coefficients"]),
        "worker_alpha": update["worker_alpha"],
        "worker_population_size": update["worker_population_size"],
        "effective_noise_scale": update["effective_noise_scale"],
        "coefficient_sha256": update["coefficient_sha256"],
        "manifest_sha256": manifest_sha,
        "prepared_rank_shards": [{
            "rank": item["rank"],
            "shard_indices": item["shard_indices"],
            "shard_seeds": item["shard_seeds"],
        } for item in prepared],
        "candidate_master_sha256": executed[0]["candidate_identity"]["sha256"],
        "candidate_runtime_values_sha256": executed[0]["materialization"][
            "runtime_values_sha256"
        ],
        "candidate_differs_from_master": True,
        "candidate_runtime_differs_from_master": True,
        "master_committed": False,
        "all_four_abort_receipts_exact": True,
        "checkpoint_snapshot_or_promotion_performed": False,
        "population_acceptance_rank_local": True,
        "update_acceptance_rank_local": True,
        "v72_two_bank_peak_observed": True,
    }


def _artifact_reference_v73(path, value):
    return {
        "path": str(path),
        "file_sha256": v66.file_sha256_v66(path),
        "content_sha256": value["content_sha256_before_self_field"],
    }


@contextmanager
def _restore_legacy_surface_v73(saved_legacy):
    global _ACTIVE_CONTEXT_V73, _PENDING_CONTROL_V73
    global _PENDING_PREREGISTRATION_V73
    try:
        yield
    finally:
        for name, value in saved_legacy.items():
            setattr(v66d, name, value)
        _ACTIVE_CONTEXT_V73 = None
        _PENDING_CONTROL_V73 = None
        _PENDING_PREREGISTRATION_V73 = None


@contextmanager
def patched_live_v73(preregistration, control):
    import run_lora_topology_probe_v40a as v40a

    global _ACTIVE_CONTEXT_V73, _PENDING_CONTROL_V73
    global _PENDING_PREREGISTRATION_V73
    if _ACTIVE_CONTEXT_V73 is not None:
        raise RuntimeError("v73 live patch is not reentrant")
    _PENDING_CONTROL_V73 = control
    _PENDING_PREREGISTRATION_V73 = preregistration
    legacy_names = {
        "RUN_DIR": RUN_DIR,
        "ATTEMPT": ATTEMPT,
        "GPU_LOG": GPU_LOG,
        "GPU_WORK_LOG": GPU_WORK_LOG,
        "POPULATION": POPULATION,
        "UPDATE": UPDATE,
        "REPORT": REPORT,
        "FAILURE": FAILURE,
        "WORKER_EXTENSION_V66D": WORKER_EXTENSION_V73,
        "LiveContextV66D": IntegrationContextV73,
    }
    saved_legacy = {name: getattr(v66d, name) for name in legacy_names}
    for name, value in legacy_names.items():
        setattr(v66d, name, value)
    base_write = v66._write_self_hashed_v66
    base_atomic_json = v40a.atomic_json
    with _restore_legacy_surface_v73(saved_legacy), \
            v66d.patched_live_v66d() as context:
        if not isinstance(context, IntegrationContextV73):
            raise RuntimeError("v73 did not receive its integration context")
        _ACTIVE_CONTEXT_V73 = context
        saved_inner = {
            "phase": v40a.Phase,
            "atomic_json": v40a.atomic_json,
            "callbacks": v66._RayMirroredCallbacksV66,
            "compact": v66._compact_population_v66,
            "update_execute": v66._execute_and_abort_nonzero_update_v66,
            "rpc_all": v66._rpc_all_v66,
            "writer": v66._write_self_hashed_v66,
            "pair_update": v66.mirrored.pair_difference_update_v66,
            "cleanup": v40a.cleanup_v38a.strict_close_trainer_v38a,
        }
        original_rpc_all = saved_inner["rpc_all"]
        original_pair_update = saved_inner["pair_update"]
        original_cleanup = saved_inner["cleanup"]

        def atomic_json_v73(path, value):
            if Path(path).resolve() == ATTEMPT:
                compact = {
                    key: item for key, item in value.items()
                    if key != "content_sha256_before_self_field"
                }
                compact.update({
                    "schema": "v73-v71-v72-qwen36-calibration-attempt",
                    "status": (
                        "launching_train_only_exact_equivalence_no_commit"
                    ),
                    "accepted_v66d_control": preregistration[
                        "accepted_v66d_control"
                    ],
                    "host_process_telemetry_required": True,
                    "v71_population_update_acceptance_required": True,
                    "v72_one_master_ownership_required": True,
                })
                value = v40a.self_hashed(compact)
            return base_atomic_json(path, value)

        def rpc_all_v73(trainer, method, args=()):
            values = original_rpc_all(trainer, method, args)
            if method != "mirrored_adapter_state_certificate_v66":
                return values
            context.certificate_calls += 1
            residency = original_rpc_all(
                trainer, "host_state_residency_v72", ()
            )
            label = (
                "post_install" if context.certificate_calls == 1
                else "final_quiescent"
            )
            context.residency[label] = validate_residency_receipts_v73(
                residency, "quiescent_one_master", 1
            )
            context.capture_host_boundary(f"{label}_one_bank")
            if context.certificate_calls == 2:
                traffic = original_rpc_all(
                    trainer, "audit_traffic_receipt_v71", (16,)
                )
                context.audit_traffic = validate_audit_traffic_v73(traffic)
            elif context.certificate_calls > 2:
                raise RuntimeError("v73 final certificate call count changed")
            return values

        def pair_update_v73(plan, signed_rewards, learning_rate):
            return _population_accept_then_update_v73(
                original_pair_update,
                plan,
                signed_rewards,
                learning_rate,
            )

        def close_trainer_v73(trainer):
            context.finalize_host_before_cleanup()
            return original_cleanup(trainer)

        def write_v73(path, value):
            resolved = Path(path).resolve()
            value = dict(value)
            if resolved == POPULATION:
                context.population_equivalence = population_equivalence_v73(
                    value, control["population"]
                )
                value["schema"] = "v73-v71-v72-qwen36-population-evidence"
                value["v71_candidate_rewards_provisional"] = True
                value["v66d_population_equivalence"] = (
                    context.population_equivalence
                )
            elif resolved == UPDATE:
                context.update_equivalence = update_equivalence_v73(
                    value, control["update"]
                )
                value["schema"] = "v73-v71-v72-qwen36-update-evidence"
                value["v66d_update_equivalence"] = context.update_equivalence
                value["v71_v72_update_invariants"] = context.update_invariants
                value["host_state_residency"] = context.residency
            elif resolved == REPORT:
                host_summary = context.require_host_summary()
                if (
                    context.certificate_calls != 2
                    or context.audit_traffic is None
                    or context.population_equivalence is None
                    or context.update_equivalence is None
                    or context.candidate_audit_consensus is None
                    or context.population_acceptance is None
                ):
                    raise RuntimeError("v73 live acceptance evidence is incomplete")
                actor_work_equivalence = actor_work_equivalence_v73(
                    list(context.receipts.values()), control["actor_rows"]
                )
                audit_artifact = _write_self_hashed_v73(AUDIT_TRAFFIC, {
                    **context.audit_traffic,
                    "candidate_audit_consensus": (
                        context.candidate_audit_consensus
                    ),
                    "population_acceptance": context.population_acceptance,
                    "update_invariants": context.update_invariants,
                    "host_state_residency": context.residency,
                })
                equivalence_artifact = _write_self_hashed_v73(EQUIVALENCE, {
                    "schema": "eggroll-es-accepted-v66d-equivalence-v73",
                    "accepted_control": preregistration[
                        "accepted_v66d_control"
                    ],
                    "population": context.population_equivalence,
                    "update": context.update_equivalence,
                    "actor_work": actor_work_equivalence,
                    "actor_work_assignment_control_sha256": (
                        preregistration["accepted_v66d_control"]
                        ["semantic_anchor"]["actor_work_id_sha256"]
                    ),
                    "candidate_reward_update_exact": True,
                    "master_committed": False,
                })
                value.update({
                    "schema": "v73-v71-v72-qwen36-calibration-report",
                    "status": (
                        "complete_exact_v66d_equivalence_no_commit_profiled"
                    ),
                    "beads": [
                        "specialist-0j5.19", "specialist-0j5.21",
                        "specialist-0j5.20",
                    ],
                    "host_process_summary": _artifact_reference_v73(
                        HOST_SUMMARY, host_summary
                    ),
                    "audit_traffic": _artifact_reference_v73(
                        AUDIT_TRAFFIC, audit_artifact
                    ),
                    "accepted_v66d_equivalence": _artifact_reference_v73(
                        EQUIVALENCE, equivalence_artifact
                    ),
                    "actor_cuda_work_log": context.ensure_summary(
                        GPU_LOG,
                        {
                            item["physical_gpu_id"]: item["worker_pid"]
                            for item in context.bindings
                        },
                    )["actor_cuda_work_log"],
                    "phase_transitions": context.phase_transitions,
                    "controller_operations": context.controller_operations,
                    "all_v71_rewards_accepted_before_update_math": True,
                    "all_v72_state_generations_one_or_two_bank_bounded": True,
                })
            elif resolved == FAILURE:
                value.update({
                    "schema": "v73-v71-v72-qwen36-calibration-failure",
                    "partial_actor_cuda_work_log": {
                        "path": str(GPU_WORK_LOG),
                        "exists": GPU_WORK_LOG.exists(),
                        "rows": len(context.receipts),
                    },
                    "partial_host_process_log": {
                        "path": str(HOST_SAMPLES),
                        "exists": HOST_SAMPLES.exists(),
                        "rows": (
                            len(context.host_monitor.rows)
                            if context.host_monitor is not None else 0
                        ),
                    },
                    "partial_phase_transitions": context.phase_transitions,
                    "partial_controller_operations": context.controller_operations,
                })
            return base_write(path, value)

        v40a.Phase = CompatiblePhaseHandshakeV73
        v40a.atomic_json = atomic_json_v73
        v40a.cleanup_v38a.strict_close_trainer_v38a = close_trainer_v73
        v66._RayMirroredCallbacksV66 = RayMirroredCallbacksV73
        v66._compact_population_v66 = compact_population_v73
        v66._execute_and_abort_nonzero_update_v66 = (
            _execute_and_abort_nonzero_update_v73
        )
        v66._rpc_all_v66 = rpc_all_v73
        v66._write_self_hashed_v66 = write_v73
        v66.mirrored.pair_difference_update_v66 = pair_update_v73
        try:
            yield context
        finally:
            context.finalize_host_before_cleanup()
            v40a.Phase = saved_inner["phase"]
            v40a.atomic_json = saved_inner["atomic_json"]
            v40a.cleanup_v38a.strict_close_trainer_v38a = saved_inner["cleanup"]
            v66._RayMirroredCallbacksV66 = saved_inner["callbacks"]
            v66._compact_population_v66 = saved_inner["compact"]
            v66._execute_and_abort_nonzero_update_v66 = saved_inner[
                "update_execute"
            ]
            v66._rpc_all_v66 = saved_inner["rpc_all"]
            v66._write_self_hashed_v66 = saved_inner["writer"]
            v66.mirrored.pair_difference_update_v66 = saved_inner["pair_update"]
            _ACTIVE_CONTEXT_V73 = None
def parser_v73():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", default=str(PREREGISTRATION))
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--preregistration-content-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv=None):
    args = parser_v73().parse_args(argv)
    if args.dry_run == args.execute:
        raise ValueError("v73 requires exactly one of --dry-run or --execute")
    preregistration = load_preregistration_v73(args)
    worker_contract = validate_lora_worker_contract_v73()
    if args.dry_run:
        recipe = preregistration["fixed_recipe"]
        print(json.dumps({
            "schema": preregistration["schema"],
            "model": "Qwen3.6-35B-A3B",
            "four_tp1_engines": True,
            "direction_count": len(recipe["direction_seeds"]),
            "signed_population_size": 2 * len(recipe["direction_seeds"]),
            "train_only_rows_per_candidate": 64,
            "expected_artifacts": artifacts_v73(),
            "worker_contract": worker_contract,
            "accepted_v66d_control": preregistration[
                "accepted_v66d_control"
            ],
            "dependency_order": recipe["integration_v73"][
                "dependency_order"
            ],
            "rank_local_population_and_update_tokens": True,
            "candidate_reward_update_equivalence": "exact",
            "host_rss_numa_fault_and_phase_telemetry": True,
            "train_semantics_model_ray_or_gpu_loaded": False,
            "filesystem_writes": False,
            "protected_dev_ood_or_holdout_opened": False,
            "checkpoint_snapshot_commit_or_promotion_authorized": False,
        }, sort_keys=True))
        return 0
    if Path(sys.executable).absolute() != v66.REQUIRED_PYTHON_V66:
        raise RuntimeError(
            f"v73 requires {v66.REQUIRED_PYTHON_V66}; observed {sys.executable}"
        )
    control = load_accepted_control_values_v73()
    with patched_live_v73(preregistration, control):
        return v66.execute_v66(preregistration, args)


if __name__ == "__main__":
    raise SystemExit(main())
