#!/usr/bin/env python3
"""Deterministic fail-closed postrun analysis for the V73B LoRA gate.

This module is CPU-only.  It opens only sealed run/control artifacts and never
imports torch, Ray, vLLM, a model, or a dataset.  Timing evidence is split into
four scopes so unlike measurements are not conflated:

* sampled telemetry epoch windows (comparable between V66d and V73B),
* exact V73B controller phase windows,
* V73B per-worker materialize/restore RPC durations, and
* per-candidate CUDA-event generation durations (comparable across runs).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
RUN = ROOT / (
    "experiments/eggroll_es_hpo/runs/"
    "v73b_lora_es_same_live_qwen36_calibration"
)
V66D_RUN = ROOT / (
    "experiments/eggroll_es_hpo/runs/"
    "v66d_lora_es_mirrored_crn_qwen36_calibration"
)
PREREGISTRATION = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "lora_es_v71_v72_same_live_calibration_v73b.json"
)
ATTEMPT = RUN.parent / ".v73b_lora_es_same_live_qwen36_calibration.attempt.json"
OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/"
    "qwen36_v73b_lora_calibration_postrun_20260717.json"
)
MARKDOWN = ROOT / (
    "experiments/eggroll_es_hpo/"
    "qwen36_v73b_lora_calibration_postrun_20260717.md"
)

RUN_FILES = {
    "actor_cuda_work_receipts_v73b.jsonl": (
        "30df82d21b28c7d5c94ede785c69c10896bad9d07db52ccf065e3031181d7013"
    ),
    "exact_audit_traffic_v73b.json": (
        "388fb6f544254c94e0c0ae11956932757834894103784f9e43d6b76e1bb3cb20"
    ),
    "gpu_activity_v73b.jsonl": (
        "7abc096625404f56271096d7785cadf5982519d53066267c89416099094c7b4c"
    ),
    "host_process_samples_v73b.jsonl": (
        "2157d882ecc6b4f0e3ee8feaab5141041676114777fb5b8c4c9ad64cd45c5706"
    ),
    "host_process_summary_v73b.json": (
        "3e4e59013f9d515121d46dc55154769ce33cb2c8cd33d0f039d9166ab806f75d"
    ),
    "mirrored_calibration_report_v73b.json": (
        "ba1b0a76dd0a9955b5e3f779d1ef440037b12b4689b3b1ab640d1ee1a4cff44a"
    ),
    "mirrored_population_v73b.json": (
        "89ba761585d503f5362b11cd1a369e871f4d2993a21a1640513de2e87d30d427"
    ),
    "pair_difference_update_v73b.json": (
        "487676d8debf114f86042c120eeb420f31c69f66d3d7c39f507c13e4c6337739"
    ),
    "same_live_equivalence_v73b.json": (
        "29f2df631f7ba758c1413e21416e06fc92d1e31d9ce29f5edc205ce81a9f14a9"
    ),
}
STATIC_FILES = {
    PREREGISTRATION: (
        "9c5ce43c36e08e038ee33e86380ab6c287ae1b4bcba4c80def623daeb00f7ed9"
    ),
    ATTEMPT: (
        "74e40ec760bd3221c7c940e44d24c1dc43284be685fd4fc40ba200ca00fb4900"
    ),
    V66D_RUN / "mirrored_calibration_report_v66d.json": (
        "12a5e854856d28bd8439cf3ed004664317086f8d117ae08e78b59f857f6102bb"
    ),
    V66D_RUN / "mirrored_population_v66d.json": (
        "9d172d15f82a54c697b8b860ff3131733d59006f1e4b790b5b9b87ded679e9d4"
    ),
    V66D_RUN / "pair_difference_update_v66d.json": (
        "f958f90b26c5b2afa4a81b03a0ab91c12d9684c2ce236bbb658d674e7a5eeffd"
    ),
    V66D_RUN / "actor_cuda_work_receipts_v66d.jsonl": (
        "aa10617c347b7ce5449165580dd4eaa98bb5131cfde5fcf9cda1134b380390e0"
    ),
    V66D_RUN / "gpu_activity_v66d.jsonl": (
        "a31d9c4cfe6507ca642c061c14cdb40b8ebe35b6ea81783a2199df2bb3c0e475"
    ),
}
SELF_HASHED_RUN_JSON = (
    "exact_audit_traffic_v73b.json",
    "host_process_summary_v73b.json",
    "mirrored_calibration_report_v73b.json",
    "mirrored_population_v73b.json",
    "pair_difference_update_v73b.json",
    "same_live_equivalence_v73b.json",
)

PREREG_FILE_SHA256 = STATIC_FILES[PREREGISTRATION]
PREREG_CONTENT_SHA256 = (
    "50b51ee2d71dc85d024c4a63cc57183d2b9c2d925c18924a719dacb1fb61dc94"
)
MASTER_SHA256 = "eea2d60e19530ba99e9ac4bc50f2806b20aa13ed30e159bad63a0144d0cb81b6"
MASTER_RUNTIME_SHA256 = (
    "a1353c47bc11f02a9b67d7859d6670b07d6754c285ac4f357255878c09384f5b"
)
LIVE_REWARD_SHA256 = (
    "0122298c844c59665b46b9e09b4b9249fb065e49e0c7629ef9d2ad9aacea76c2"
)
LIVE_COEFFICIENT_SHA256 = (
    "005182fc01f44066ce9728cbefcaca905b08c79cb1d59d39532bc9d154c3bc14"
)
LIVE_CANDIDATE_SHA256 = (
    "f3bcdb9de5d9b815a1dc5f1f8678d02b476755521847649971a0ed9c796c068a"
)
LIVE_RUNTIME_SHA256 = (
    "819563416319fbf66590cd6603b9eb21305c9c0880aaf9d1323f452ecd53e3ef"
)
GPU_IDS = (0, 1, 2, 3)
SIGNED_CANDIDATES = 16
WAVES = 4
ROWS_PER_CANDIDATE = 64
MEMORY_TOTAL_MIB = 97_887
MASTER_BYTES = 18_112_512

EXPECTED_PER_ACTOR_TRAFFIC = {
    "base_d2h_bytes": 857_997_312,
    "exact_base_audits": 3,
    "h2d_bytes": 137_797_632,
    "lora_d2h_bytes": 196_853_760,
    "lora_d2h_calls": 20,
    "master_validation_host_copy_bytes": 0,
    "total_device_transfer_bytes": 1_192_648_704,
}
EXPECTED_AGGREGATE_TRAFFIC = {
    "base_d2h_bytes": 3_431_989_248,
    "exact_base_audits": 12,
    "h2d_bytes": 551_190_528,
    "lora_d2h_bytes": 787_415_040,
    "lora_d2h_calls": 80,
    "master_validation_host_copy_bytes": 0,
    "total_device_transfer_bytes": 4_770_594_816,
}
V66D_D2H_BYTES = 18_341_068_800
V73B_D2H_BYTES = 4_219_404_288
SAVED_D2H_BYTES = 14_121_664_512


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="ascii"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="ascii").splitlines(), 1
    ):
        _require(bool(line), f"blank JSONL row: {path}:{line_number}")
        row = json.loads(line)
        _require(isinstance(row, dict), f"JSONL object required: {path}:{line_number}")
        rows.append(row)
    _require(bool(rows), f"empty JSONL artifact: {path}")
    return rows


def validate_self_hash(value: Mapping[str, Any], label: str) -> str:
    body = copy.deepcopy(dict(value))
    claimed = body.pop("content_sha256_before_self_field", None)
    _require(
        isinstance(claimed, str) and claimed == canonical_sha256(body),
        f"self hash changed: {label}",
    )
    return claimed


def validate_embedded_hash(
    value: Mapping[str, Any], field: str, label: str
) -> str:
    body = copy.deepcopy(dict(value))
    claimed = body.pop(field, None)
    _require(
        isinstance(claimed, str) and claimed == canonical_sha256(body),
        f"embedded hash changed: {label}:{field}",
    )
    return claimed


def validate_static_identity() -> dict[str, Any]:
    _require(RUN.is_dir() and not RUN.is_symlink(), "V73B run directory changed")
    observed_names = sorted(path.name for path in RUN.iterdir())
    _require(
        observed_names == sorted(RUN_FILES)
        and all((RUN / name).is_file() and not (RUN / name).is_symlink()
                for name in observed_names),
        "V73B run inventory changed",
    )
    files = []
    for name, expected in RUN_FILES.items():
        path = RUN / name
        actual = file_sha256(path)
        _require(actual == expected, f"V73B artifact file hash changed: {name}")
        files.append({
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": actual,
        })
    for path, expected in STATIC_FILES.items():
        _require(path.is_file() and not path.is_symlink(), f"static input missing: {path}")
        _require(file_sha256(path) == expected, f"static input hash changed: {path}")
    _require(not (RUN / "failure_v73b.json").exists(), "V73B failure artifact appeared")

    preregistration = _load_json(PREREGISTRATION)
    validate_self_hash(preregistration, "V73B preregistration")
    _require(
        preregistration.get("schema")
        == "lora-es-v71-v72-qwen36-same-live-calibration-preregistration-v73b"
        and preregistration.get("content_sha256_before_self_field")
        == PREREG_CONTENT_SHA256
        and preregistration.get("acceptance", {}).get(
            "canonical_and_independent_compiler_outputs_whole_mapping_exact"
        )
        is True
        and preregistration.get("acceptance", {}).get(
            "historical_reward_values_are_diagnostic_only"
        )
        is True,
        "V73B preregistration semantics changed",
    )
    attempt = _load_json(ATTEMPT)
    validate_self_hash(attempt, "V73B launch attempt")
    _require(
        attempt.get("preregistration_file_sha256") == PREREG_FILE_SHA256
        and attempt.get("preregistration_content_sha256") == PREREG_CONTENT_SHA256
        and attempt.get("checkpoint_snapshot_or_promotion_authorized") is False
        and attempt.get("protected_dev_ood_or_holdout_opened") is False
        and attempt.get("v71_population_update_acceptance_required") is True
        and attempt.get("v72_one_master_ownership_required") is True,
        "V73B launch attempt authority changed",
    )
    for name in SELF_HASHED_RUN_JSON:
        validate_self_hash(_load_json(RUN / name), name)
    return {
        "preregistration": {
            "path": str(PREREGISTRATION.relative_to(ROOT)),
            "file_sha256": PREREG_FILE_SHA256,
            "content_sha256": PREREG_CONTENT_SHA256,
        },
        "attempt": {
            "path": str(ATTEMPT.relative_to(ROOT)),
            "file_sha256": STATIC_FILES[ATTEMPT],
            "content_sha256": attempt["content_sha256_before_self_field"],
            "inherited_v73_launch_journal_schema": attempt["schema"],
            "final_v73b_report_is_authoritative": True,
        },
        "run_inventory": {
            "file_count": len(files),
            "files": sorted(files, key=lambda row: row["path"]),
            "bundle_sha256": canonical_sha256(
                sorted(files, key=lambda row: row["path"])
            ),
        },
        "all_json_artifact_self_hashes_valid": True,
        "failure_artifact_absent": True,
    }


def _semantic_actor_row(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "wave_index", "engine_rank", "direction_seed", "sign", "pair_id",
        "evaluation_contract_sha256", "work_id", "output_cardinality",
    )
    return {field: row.get(field) for field in fields}


def validate_actor_receipts(
    path: Path,
    *,
    control_path: Path | None = None,
) -> dict[str, Any]:
    rows = _load_jsonl(path)
    _require(len(rows) == SIGNED_CANDIDATES, "actor receipt count changed")
    by_wave: dict[int, list[dict[str, Any]]] = defaultdict(list)
    bindings = {}
    elapsed = []
    for row in rows:
        body = copy.deepcopy(row)
        claim = body.pop("receipt_sha256", None)
        _require(
            row.get("schema") == "eggroll-es-actor-cuda-work-receipt-v66d"
            and isinstance(claim, str)
            and claim == canonical_sha256(body),
            "actor CUDA receipt self hash changed",
        )
        wave = row.get("wave_index")
        rank = row.get("engine_rank")
        gpu = row.get("physical_gpu_id")
        pid = row.get("worker_pid")
        cardinality = row.get("output_cardinality", {})
        event = row.get("cuda_event", {})
        _require(
            wave in range(WAVES)
            and rank in GPU_IDS
            and gpu in GPU_IDS
            and isinstance(pid, int) and pid > 1
            and cardinality
            == {
                "generated_tokens": ROWS_PER_CANDIDATE,
                "prompt_tokens": 4823,
                "request_outputs": ROWS_PER_CANDIDATE,
                "samples": ROWS_PER_CANDIDATE,
            }
            and event.get("backend") == "torch.cuda.Event"
            and event.get("start_recorded") is True
            and event.get("end_recorded") is True
            and event.get("end_synchronized") is True
            and isinstance(event.get("elapsed_ms"), (int, float))
            and math.isfinite(event["elapsed_ms"])
            and event["elapsed_ms"] > 0
            and isinstance(event.get("worker_monotonic_elapsed_ns"), int)
            and event["worker_monotonic_elapsed_ns"] > 0,
            "actor CUDA receipt semantics changed",
        )
        _require(
            rank not in bindings or bindings[rank] == (gpu, pid),
            "actor rank/PID/GPU binding changed",
        )
        bindings[rank] = (gpu, pid)
        by_wave[wave].append(row)
        elapsed.append(float(event["elapsed_ms"]))
    _require(
        sorted(by_wave) == list(range(WAVES))
        and all(
            len(by_wave[wave]) == 4
            and {row["engine_rank"] for row in by_wave[wave]} == set(GPU_IDS)
            and {row["physical_gpu_id"] for row in by_wave[wave]} == set(GPU_IDS)
            for wave in range(WAVES)
        )
        and len({row["work_id"] for row in rows}) == SIGNED_CANDIDATES
        and set(bindings) == set(GPU_IDS)
        and len({value[0] for value in bindings.values()}) == 4
        and len({value[1] for value in bindings.values()}) == 4,
        "actor receipt wave coverage changed",
    )
    if control_path is not None:
        control = _load_jsonl(control_path)
        observed_semantics = sorted(
            (_semantic_actor_row(row) for row in rows),
            key=lambda row: row["work_id"],
        )
        control_semantics = sorted(
            (_semantic_actor_row(row) for row in control),
            key=lambda row: row["work_id"],
        )
        _require(
            observed_semantics == control_semantics,
            "actor work assignment/cardinality differs from V66d",
        )
    wave_max = [
        max(row["cuda_event"]["elapsed_ms"] for row in by_wave[wave])
        for wave in range(WAVES)
    ]
    return {
        "row_count": len(rows),
        "all_row_self_hashes_valid": True,
        "unique_work_ids": len({row["work_id"] for row in rows}),
        "bindings": [
            {
                "actor_rank": rank,
                "physical_gpu_id": bindings[rank][0],
                "worker_pid": bindings[rank][1],
            }
            for rank in GPU_IDS
        ],
        "semantic_equivalence_to_v66d": control_path is not None,
        "cuda_event_ms": {
            "median": statistics.median(elapsed),
            "mean": statistics.fmean(elapsed),
            "minimum": min(elapsed),
            "maximum": max(elapsed),
            "critical_path_max_by_wave": wave_max,
            "critical_path_sum_seconds": sum(wave_max) / 1000.0,
        },
        "rows": rows,
    }


def validate_gpu_telemetry(
    path: Path,
    bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = _load_jsonl(path)
    expected = {
        int(row["physical_gpu_id"]): int(row["worker_pid"])
        for row in bindings
    }
    _require(
        [row.get("sequence") for row in rows] == list(range(len(rows)))
        and len(rows) % 4 == 0,
        "GPU telemetry sequence changed",
    )
    per_gpu: dict[int, list[dict[str, Any]]] = defaultdict(list)
    batches = []
    for offset in range(0, len(rows), 4):
        batch = rows[offset:offset + 4]
        _require(
            {row.get("gpu") for row in batch} == set(GPU_IDS)
            and len({row.get("monotonic_ns") for row in batch}) == 1
            and len({row.get("phase") for row in batch}) == 1
            and len({row.get("phase_epoch") for row in batch}) == 1,
            "incomplete four-GPU telemetry batch",
        )
        batches.append(batch)
        for row in batch:
            gpu = row["gpu"]
            _require(
                row.get("schema") == "eggroll-es-four-gpu-phase-telemetry-v66"
                and row.get("actor_rank") == gpu
                and row.get("expected_pid") == expected[gpu]
                and row.get("compute_pids") == [expected[gpu]]
                and row.get("foreign_compute_pids") == []
                and row.get("memory_total_mib") == MEMORY_TOTAL_MIB
                and isinstance(row.get("gpu_utilization_percent"), int)
                and 0 <= row["gpu_utilization_percent"] <= 100
                and isinstance(row.get("memory_used_mib"), int)
                and 0 < row["memory_used_mib"] <= MEMORY_TOTAL_MIB
                and isinstance(row.get("pcie_rx_kib_per_second"), int)
                and row["pcie_rx_kib_per_second"] >= 0
                and isinstance(row.get("pcie_tx_kib_per_second"), int)
                and row["pcie_tx_kib_per_second"] >= 0,
                "GPU telemetry contract changed",
            )
            per_gpu[gpu].append(row)

    epoch_rows = []
    for batch in batches:
        row = batch[0]
        key = (row["phase_epoch"], row["phase"])
        if not epoch_rows or epoch_rows[-1]["key"] != key:
            epoch_rows.append({
                "key": key,
                "first_monotonic_ns": row["monotonic_ns"],
                "last_monotonic_ns": row["monotonic_ns"],
                "batch_count": 0,
            })
        epoch_rows[-1]["last_monotonic_ns"] = row["monotonic_ns"]
        epoch_rows[-1]["batch_count"] += 1
    _require(
        [row["key"][0] for row in epoch_rows]
        == list(range(len(epoch_rows))),
        "GPU telemetry phase epochs changed",
    )
    phases = []
    for index, row in enumerate(epoch_rows):
        next_first = (
            epoch_rows[index + 1]["first_monotonic_ns"]
            if index + 1 < len(epoch_rows)
            else row["last_monotonic_ns"]
        )
        phases.append({
            "phase_epoch": row["key"][0],
            "phase": row["key"][1],
            "batch_count": row["batch_count"],
            "sample_span_seconds": (
                row["last_monotonic_ns"] - row["first_monotonic_ns"]
            ) / 1e9,
            "epoch_upper_window_seconds": (
                next_first - row["first_monotonic_ns"]
            ) / 1e9,
        })
    return {
        "row_count": len(rows),
        "complete_four_gpu_batches": len(batches),
        "foreign_compute_process_observations": 0,
        "all_four_gpus_positive": all(
            any(row["gpu_utilization_percent"] > 0 for row in per_gpu[gpu])
            for gpu in GPU_IDS
        ),
        "per_gpu": {
            str(gpu): {
                "expected_pid": expected[gpu],
                "peak_memory_used_mib": max(
                    row["memory_used_mib"] for row in per_gpu[gpu]
                ),
                "peak_gpu_utilization_percent": max(
                    row["gpu_utilization_percent"] for row in per_gpu[gpu]
                ),
                "positive_utilization_rows": sum(
                    row["gpu_utilization_percent"] > 0 for row in per_gpu[gpu]
                ),
            }
            for gpu in GPU_IDS
        },
        "phase_epochs": phases,
    }


def _rebuild_host_actor(
    rows: Sequence[Mapping[str, Any]], binding: Mapping[str, Any]
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["sample_index"])
    statuses = [row["process"]["status_bytes"] for row in ordered]
    minor = [row["process"]["minor_faults"] for row in ordered]
    major = [row["process"]["major_faults"] for row in ordered]
    _require(
        all(right >= left for left, right in zip(minor, minor[1:]))
        and all(right >= left for left, right in zip(major, major[1:])),
        "host fault counter moved backwards",
    )
    numa_rows = [row for row in ordered if row["process"]["numa_included"]]
    nodes = sorted({
        node
        for row in numa_rows
        for node in row["process"]["numa"]["node_pages"]
    })
    peak_pages = {
        node: max(
            row["process"]["numa"]["node_pages"].get(node, 0)
            for row in numa_rows
        )
        for node in nodes
    }
    return {
        "binding": dict(binding),
        "sample_count": len(ordered),
        "first_monotonic_ns": ordered[0]["monotonic_ns"],
        "last_monotonic_ns": ordered[-1]["monotonic_ns"],
        "rss_bytes_first": statuses[0]["VmRSS"],
        "rss_bytes_last": statuses[-1]["VmRSS"],
        "rss_bytes_peak_sampled": max(row["VmRSS"] for row in statuses),
        "hwm_bytes_peak_reported": max(row["VmHWM"] for row in statuses),
        "rss_anon_bytes_peak": max(row["RssAnon"] for row in statuses),
        "rss_file_bytes_peak": max(row["RssFile"] for row in statuses),
        "rss_shmem_bytes_peak": max(row["RssShmem"] for row in statuses),
        "locked_bytes_peak": max(row["VmLck"] for row in statuses),
        "pinned_bytes_peak": max(row["VmPin"] for row in statuses),
        "minor_faults_first": minor[0],
        "minor_faults_last": minor[-1],
        "minor_faults_delta": minor[-1] - minor[0],
        "major_faults_first": major[0],
        "major_faults_last": major[-1],
        "major_faults_delta": major[-1] - major[0],
        "numa_sample_count": len(numa_rows),
        "numa_nodes_observed": nodes,
        "peak_pages_by_numa_node": peak_pages,
    }


def validate_host_telemetry(
    samples_path: Path,
    summary: Mapping[str, Any],
    bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = _load_jsonl(samples_path)
    expected_binding = {
        int(row["actor_rank"]): dict(row) for row in bindings
    }
    _require(
        [row.get("sample_index") for row in rows] == list(range(len(rows))),
        "host sample sequence changed",
    )
    by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        body = copy.deepcopy(row)
        claim = body.pop("row_sha256", None)
        rank = row.get("binding", {}).get("actor_rank")
        process = row.get("process", {})
        status = process.get("status_bytes", {})
        _require(
            row.get("schema") == "eggroll-es-actor-host-process-sample-v73"
            and isinstance(claim, str) and claim == canonical_sha256(body)
            and rank in GPU_IDS
            and row.get("binding") == expected_binding[rank]
            and process.get("pid") == expected_binding[rank]["worker_pid"]
            and set(status)
            == {
                "VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem",
                "VmLck", "VmPin", "Threads",
            }
            and all(isinstance(value, int) and value >= 0 for value in status.values())
            and isinstance(process.get("minor_faults"), int)
            and process["minor_faults"] >= 0
            and isinstance(process.get("major_faults"), int)
            and process["major_faults"] >= 0,
            "host process sample changed",
        )
        if process.get("numa_included"):
            numa = process.get("numa", {})
            _require(
                numa.get("page_size_bytes") == 4096
                and all(
                    numa["node_bytes"][node] == pages * 4096
                    for node, pages in numa["node_pages"].items()
                ),
                "host NUMA sample changed",
            )
        by_rank[rank].append(row)
    rebuilt = {
        str(rank): _rebuild_host_actor(
            by_rank[rank], expected_binding[rank]
        )
        for rank in GPU_IDS
    }
    _require(
        summary.get("schema") == "eggroll-es-four-actor-host-process-summary-v73"
        and summary.get("actor_count") == 4
        and summary.get("sample_count") == len(rows)
        and summary.get("sample_interval_seconds") == 0.5
        and summary.get("sampling_stopped_before_actor_cleanup") is True
        and summary.get("all_actor_rank_pid_gpu_bindings_exact") is True
        and summary.get("all_fault_counters_monotonic") is True
        and summary.get("actors") == rebuilt
        and summary.get("numa_nodes_observed") == ["0"]
        and summary.get("sample_log")
        == {
            "path": str(samples_path),
            "file_sha256": RUN_FILES[samples_path.name],
            "rows": len(rows),
        },
        "host process summary changed",
    )
    worker_times = summary.get("worker_operation_times", {})
    materialize = worker_times.get("candidate_materialization")
    restore = worker_times.get("exact_restore")
    _require(
        isinstance(materialize, list) and len(materialize) == 16
        and isinstance(restore, list) and len(restore) == 16
        and all(row.get("elapsed_ns", 0) > 0 for row in materialize + restore)
        and [row["reason"] for row in restore]
        == [f"wave_{wave}_finalize" for wave in range(4) for _ in range(4)],
        "worker operation timing coverage changed",
    )
    materialize_seconds = [row["elapsed_ns"] / 1e9 for row in materialize]
    restore_seconds = [row["elapsed_ns"] / 1e9 for row in restore]
    return {
        "sample_count": len(rows),
        "all_sample_row_self_hashes_valid": True,
        "all_fault_counters_monotonic": True,
        "numa_nodes_observed": ["0"],
        "actors": rebuilt,
        "maximum_sampled_rss_bytes": max(
            row["rss_bytes_peak_sampled"] for row in rebuilt.values()
        ),
        "maximum_reported_hwm_bytes": max(
            row["hwm_bytes_peak_reported"] for row in rebuilt.values()
        ),
        "maximum_minor_fault_delta": max(
            row["minor_faults_delta"] for row in rebuilt.values()
        ),
        "maximum_major_fault_delta": max(
            row["major_faults_delta"] for row in rebuilt.values()
        ),
        "maximum_locked_bytes": max(
            row["locked_bytes_peak"] for row in rebuilt.values()
        ),
        "maximum_pinned_bytes": max(
            row["pinned_bytes_peak"] for row in rebuilt.values()
        ),
        "worker_rpc_seconds": {
            "candidate_materialization": {
                "median": statistics.median(materialize_seconds),
                "mean": statistics.fmean(materialize_seconds),
                "minimum": min(materialize_seconds),
                "maximum": max(materialize_seconds),
                "critical_path_max_sum": sum(
                    max(materialize_seconds[index:index + 4])
                    for index in range(0, 16, 4)
                ),
            },
            "exact_restore": {
                "median": statistics.median(restore_seconds),
                "mean": statistics.fmean(restore_seconds),
                "minimum": min(restore_seconds),
                "maximum": max(restore_seconds),
                "critical_path_max_sum": sum(
                    max(restore_seconds[index:index + 4])
                    for index in range(0, 16, 4)
                ),
            },
        },
    }


def _pair_coefficients(
    plan: Mapping[str, Any], rewards: Sequence[Mapping[str, Any]]
) -> list[float]:
    assignments = [item for wave in plan["waves"] for item in wave]
    expected = {
        (item["pair_id"], item["sign"]): item for item in assignments
    }
    observed = {}
    for row in rewards:
        key = (row.get("pair_id"), row.get("sign"))
        reward = row.get("reward")
        _require(
            key in expected and key not in observed
            and isinstance(reward, (int, float)) and not isinstance(reward, bool)
            and math.isfinite(float(reward))
            and all(
                row.get(field) == expected[key][field]
                for field in (
                    "direction_index", "direction_seed",
                    "evaluation_contract_sha256",
                )
            ),
            "live reward matrix changed",
        )
        observed[key] = float(reward)
    _require(set(observed) == set(expected), "live reward matrix incomplete")
    coefficients = []
    for direction_index, _seed in enumerate(plan["direction_seeds"]):
        matching = [
            item for item in assignments
            if item["direction_index"] == direction_index
        ]
        pair_ids = {item["pair_id"] for item in matching}
        _require(len(matching) == 2 and len(pair_ids) == 1, "pair plan changed")
        pair_id = next(iter(pair_ids))
        coefficients.append(observed[(pair_id, 1)] - observed[(pair_id, -1)])
    return coefficients


def _validate_residency(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "post_install": ("quiescent_one_master", 1, MASTER_BYTES),
        "update_executed": ("executed_candidate_retained", 2, 2 * MASTER_BYTES),
        "post_abort": ("quiescent_one_master", 1, MASTER_BYTES),
        "final_quiescent": ("quiescent_one_master", 1, MASTER_BYTES),
    }
    result = {}
    for name, (phase, banks, bytes_per_actor) in expected.items():
        row = value.get(name)
        _require(
            isinstance(row, dict)
            and row.get("phase") == phase
            and row.get("actor_count") == 4
            and row.get("unique_owned_bank_count_per_actor") == banks
            and row.get("unique_owned_tensor_bytes_per_actor") == bytes_per_actor
            and row.get("aggregate_actor_tensor_bytes") == 4 * bytes_per_actor
            and isinstance(row.get("receipt_sha256_by_rank"), list)
            and len(row["receipt_sha256_by_rank"]) == 4
            and all(
                isinstance(item, str) and len(item) == 64
                for item in row["receipt_sha256_by_rank"]
            ),
            f"host state residency changed: {name}",
        )
        result[name] = {
            "phase": phase,
            "banks_per_actor": banks,
            "bytes_per_actor": bytes_per_actor,
            "aggregate_actor_bytes": 4 * bytes_per_actor,
        }
    installed_receipts = value["post_install"]["receipt_sha256_by_rank"]
    executed_receipts = value["update_executed"]["receipt_sha256_by_rank"]
    _require(
        len(set(installed_receipts)) == 1
        and len(set(executed_receipts)) == 1
        and executed_receipts != installed_receipts
        and value["post_abort"]["receipt_sha256_by_rank"]
        == installed_receipts
        and value["final_quiescent"]["receipt_sha256_by_rank"]
        == installed_receipts,
        "exact abort receipt identity changed",
    )
    result["receipt_identity"] = {
        "all_four_install_master_receipts_exact": True,
        "all_four_executed_candidate_receipts_exact": True,
        "executed_candidate_differs_from_installed_master": True,
        "all_four_post_abort_receipts_exact_to_install": True,
        "all_four_final_receipts_exact_to_install": True,
    }
    return result


def validate_semantics(
    artifacts: Mapping[str, Mapping[str, Any]],
    actor: Mapping[str, Any],
    host: Mapping[str, Any],
) -> dict[str, Any]:
    population = artifacts["population"]
    update = artifacts["update"]
    equivalence = artifacts["equivalence"]
    traffic = artifacts["traffic"]
    report = artifacts["report"]
    host_summary = artifacts["host_summary"]
    control_population = artifacts["control_population"]
    control_update = artifacts["control_update"]

    _require(
        report.get("schema") == "v73b-v71-v72-qwen36-calibration-report"
        and report.get("status")
        == "complete_same_live_update_equivalence_no_commit_profiled"
        and report.get("preregistration_file_sha256") == PREREG_FILE_SHA256
        and report.get("preregistration_content_sha256") == PREREG_CONTENT_SHA256,
        "V73B final report identity changed",
    )
    references = {
        "population": ("mirrored_population_v73b.json", population),
        "nonzero_update": ("pair_difference_update_v73b.json", update),
        "same_live_equivalence": ("same_live_equivalence_v73b.json", equivalence),
        "audit_traffic": ("exact_audit_traffic_v73b.json", traffic),
        "host_process_summary": ("host_process_summary_v73b.json", host_summary),
    }
    for field, (name, value) in references.items():
        _require(
            report.get(field)
            == {
                "path": str(RUN / name),
                "file_sha256": RUN_FILES[name],
                "content_sha256": value["content_sha256_before_self_field"],
            },
            f"report artifact reference changed: {field}",
        )
    _require(
        report.get("actor_cuda_work_log")
        == {
            "path": str(RUN / "actor_cuda_work_receipts_v73b.jsonl"),
            "file_sha256": RUN_FILES["actor_cuda_work_receipts_v73b.jsonl"],
            "rows": 16,
        }
        and report.get("gpu_log_file_sha256")
        == RUN_FILES["gpu_activity_v73b.jsonl"]
        and report.get("population", {}).get("content_sha256")
        == update.get("population_content_sha256"),
        "report JSONL or population linkage changed",
    )
    gpu_waves = report.get("gpu_waves", {})
    actor_rows_by_key = {
        (row["wave_index"], row["physical_gpu_id"]): row
        for row in actor["rows"]
    }
    _require(
        gpu_waves.get("passed") is True
        and gpu_waves.get("waves") == 4
        and gpu_waves.get("signed_candidates") == 16
        and gpu_waves.get("actor_cuda_event_receipt_for_every_candidate") is True
        and gpu_waves.get("actor_pid_gpu_binding_exact") is True
        and gpu_waves.get("all_four_physical_gpus_useful_in_every_wave") is True
        and gpu_waves.get("nonzero_output_and_token_cardinality_every_candidate")
        is True
        and set(gpu_waves.get("by_wave", {})) == {"0", "1", "2", "3"}
        and all(
            set(gpu_waves["by_wave"][str(wave)]) == {"0", "1", "2", "3"}
            and all(
                gpu_waves["by_wave"][str(wave)][str(gpu)].get(
                    "actor_cuda_work_receipt_sha256"
                )
                == actor_rows_by_key[(wave, gpu)]["receipt_sha256"]
                and gpu_waves["by_wave"][str(wave)][str(gpu)].get(
                    "cuda_event_elapsed_ms"
                )
                == actor_rows_by_key[(wave, gpu)]["cuda_event"]["elapsed_ms"]
                and gpu_waves["by_wave"][str(wave)][str(gpu)].get(
                    "output_cardinality"
                )
                == actor_rows_by_key[(wave, gpu)]["output_cardinality"]
                and gpu_waves["by_wave"][str(wave)][str(gpu)].get(
                    "useful_work_attributed"
                )
                is True
                for gpu in GPU_IDS
            )
            for wave in range(WAVES)
        ),
        "report four-wave GPU attribution changed",
    )
    gpu_activity = report.get("gpu_activity", {})
    _require(
        gpu_activity.get("all_four_attributed_positive") is True
        and set(gpu_activity.get("by_gpu", {})) == {"0", "1", "2", "3"}
        and all(
            gpu_activity["by_gpu"][str(gpu)].get("actor_cuda_work_receipts") == 4
            and gpu_activity["by_gpu"][str(gpu)].get("useful_work_attributed")
            is True
            for gpu in GPU_IDS
        )
        and report.get("compute_ledger", {}).get("physical_gpus") == [0, 1, 2, 3]
        and report["compute_ledger"].get("optimization_generated_rollouts") == 1024
        and report["compute_ledger"].get("teacher_forced_answer_tokens") == 26_384
        and report["compute_ledger"].get("checkpoint_count") == 0,
        "report GPU activity or compute ledger changed",
    )

    _require(
        population.get("schema") == "v73b-v71-v72-qwen36-population-evidence"
        and population.get("signed_reward_sha256") == LIVE_REWARD_SHA256
        and canonical_sha256(population["signed_rewards"]) == LIVE_REWARD_SHA256
        and len(population.get("materializations", [])) == 16
        and len(population.get("restorations", [])) == 16
        and population.get("all_submitted_work_drained") is True
        and population.get("all_four_actors_restored_after_every_wave") is True
        and population.get("v71_candidate_rewards_provisional") is False
        and population.get("v71_live_rewards_accepted_before_update_math") is True,
        "V73B population completion changed",
    )
    plan = population["plan"]
    plan_body = copy.deepcopy(plan)
    plan_sha = plan_body.pop("plan_sha256", None)
    _require(
        plan_sha == canonical_sha256(plan_body)
        and population.get("plan_sha256") == plan_sha,
        "population plan hash changed",
    )
    coefficients = _pair_coefficients(plan, population["signed_rewards"])
    coefficient_sha = canonical_sha256({
        "seeds": plan["direction_seeds"],
        "coefficients": coefficients,
    })
    _require(
        coefficient_sha == LIVE_COEFFICIENT_SHA256
        and update.get("coefficient_sha256") == coefficient_sha
        and update.get("coefficient_l2")
        == math.sqrt(math.fsum(value * value for value in coefficients))
        and update.get("nonzero_pair_differences")
        == sum(value != 0.0 for value in coefficients)
        == 8
        and update.get("direction_count") == 8
        and update.get("worker_alpha") == 0.125
        and update.get("worker_population_size") == 8
        and update.get("effective_noise_scale") == 0.015625,
        "same-live coefficient derivation changed",
    )
    exact_fields = population["same_live_population_equivalence"][
        "exact_non_reward_fields"
    ]
    reward_metadata = lambda row: {
        key: value for key, value in row.items() if key != "reward"
    }
    _require(
        [reward_metadata(row) for row in population["signed_rewards"]]
        == [reward_metadata(row) for row in control_population["signed_rewards"]],
        "live reward assignment metadata differs from V66d",
    )
    reward_deltas = [
        float(live["reward"]) - float(old["reward"])
        for live, old in zip(
            population["signed_rewards"],
            control_population["signed_rewards"],
            strict=True,
        )
    ]

    def preference_signs(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        pairs: dict[str, dict[int, float]] = defaultdict(dict)
        for row in rows:
            pairs[str(row["pair_id"])][int(row["sign"])] = float(row["reward"])
        _require(
            len(pairs) == 8 and all(set(value) == {-1, 1} for value in pairs.values()),
            "reward preference pair coverage changed",
        )
        return {
            pair_id: (values[1] > values[-1]) - (values[1] < values[-1])
            for pair_id, values in pairs.items()
        }

    live_signs = preference_signs(population["signed_rewards"])
    old_signs = preference_signs(control_population["signed_rewards"])
    recomputed_drift = {
        "acceptance_threshold_applied": False,
        "exact_value_match_count": sum(value == 0.0 for value in reward_deltas),
        "l2": math.sqrt(math.fsum(value * value for value in reward_deltas)),
        "maximum_absolute": max(abs(value) for value in reward_deltas),
        "mean_absolute": (
            math.fsum(abs(value) for value in reward_deltas) / len(reward_deltas)
        ),
        "pair_count": len(live_signs),
        "pair_preference_sign_match_count": sum(
            live_signs[pair_id] == old_signs[pair_id] for pair_id in live_signs
        ),
    }
    _require(
        all(population.get(field) == control_population.get(field)
            for field in exact_fields),
        "non-reward population equivalence changed",
    )
    for row in population["materializations"]:
        _require(
            row.get("master_sha256") == MASTER_SHA256
            and row.get("master_unchanged") is True
            and row.get("exact_restore_required") is True,
            "candidate materialization master invariant changed",
        )
    for row in population["restorations"]:
        _require(
            row
            == {
                "algebraic_bf16_restore_used": False,
                "reason": row["reason"],
                "restored_master_sha256": MASTER_SHA256,
                "runtime_values_sha256": MASTER_RUNTIME_SHA256,
                "terminal_poisoned": False,
            },
            "population exact restore changed",
        )

    pop_equiv = population["same_live_population_equivalence"]
    compiler = update["same_live_compiler_equivalence"]
    update_equiv = update["same_live_update_equivalence"]
    validate_embedded_hash(pop_equiv, "equivalence_sha256", "population equivalence")
    validate_embedded_hash(compiler, "equivalence_sha256", "compiler equivalence")
    validate_embedded_hash(
        update_equiv["live_update_execution"],
        "consensus_sha256", "live update consensus",
    )
    validate_embedded_hash(update_equiv, "equivalence_sha256", "update equivalence")
    validate_embedded_hash(equivalence["actor_work"], "equivalence_sha256", "actor work")
    _require(
        pop_equiv == equivalence.get("population")
        and update_equiv == equivalence.get("update")
        and pop_equiv.get("historical_reward_bit_exact") is False
        and pop_equiv.get("historical_reward_values_are_diagnostic_only") is True
        and pop_equiv.get("live_reward_matrix_complete_finite_and_unique") is True
        and pop_equiv.get("reward_assignment_metadata_exact_to_accepted_v66d") is True
        and pop_equiv.get("historical_reward_delta", {}).get(
            "acceptance_threshold_applied"
        )
        is False
        and pop_equiv.get("historical_reward_delta") == recomputed_drift
        and compiler.get("same_live_reward_object_used_for_both") is True
        and compiler.get("whole_result_mapping_exact") is True
        and compiler.get("coefficient_sha256") == coefficient_sha
        and update_equiv.get("canonical_and_independent_compilers_exact") is True
        and update_equiv.get("four_actor_candidate_and_runtime_identity_exact") is True
        and update_equiv.get("historical_reward_derived_update_identity_required") is False
        and equivalence.get("same_live_canonical_and_independent_compilers_exact") is True
        and equivalence.get("four_actor_live_update_identity_consensus") is True,
        "same-live equivalence contract changed",
    )
    execution = update_equiv["live_update_execution"]
    _require(
        execution.get("actor_count") == 4
        and execution.get("all_actor_identities_exact") is True
        and execution.get("candidate_master_sha256") == LIVE_CANDIDATE_SHA256
        and execution.get("candidate_runtime_values_sha256") == LIVE_RUNTIME_SHA256
        and execution.get("candidate_differs_from_master") is True
        and execution.get("runtime_differs_from_master") is True
        and update.get("candidate_master_sha256") == LIVE_CANDIDATE_SHA256
        and update.get("candidate_runtime_values_sha256") == LIVE_RUNTIME_SHA256,
        "live candidate execution consensus changed",
    )
    static_update_fields = update_equiv["static_fields_exact_to_accepted_v66d"]
    _require(
        all(update.get(field) == control_update.get(field)
            for field in static_update_fields)
        and update_equiv.get("historical_candidate_master_sha256")
        == control_update.get("candidate_master_sha256")
        and update_equiv.get("historical_candidate_runtime_values_sha256")
        == control_update.get("candidate_runtime_values_sha256"),
        "static update equivalence to V66d changed",
    )

    _require(
        traffic.get("schema") == "eggroll-es-four-actor-audit-traffic-v73"
        and traffic.get("exact_match") is True
        and traffic.get("per_actor_expected_and_observed")
        == EXPECTED_PER_ACTOR_TRAFFIC
        and traffic.get("aggregate_expected_and_observed")
        == EXPECTED_AGGREGATE_TRAFFIC
        and len(traffic.get("by_rank", [])) == 4
        and {row.get("rank") for row in traffic["by_rank"]} == set(GPU_IDS)
        and all(
            row.get("observed") == {
                **EXPECTED_PER_ACTOR_TRAFFIC,
                "cheap_transition_checks": 21,
                "master_cache_hits": 28,
            }
            and row.get("completed_boundaries")
            == ["population_reward_acceptance", "update_acceptance"]
            for row in traffic["by_rank"]
        ),
        "exact audit traffic receipt changed",
    )
    comparison = traffic["accepted_v66d_measured_audit_d2h_comparison"]
    saved_fraction = SAVED_D2H_BYTES / V66D_D2H_BYTES
    _require(
        comparison.get("accepted_v66d_base_plus_lora_d2h_bytes")
        == V66D_D2H_BYTES
        and comparison.get("v73_base_plus_lora_d2h_bytes") == V73B_D2H_BYTES
        and comparison.get("saved_bytes") == SAVED_D2H_BYTES
        and comparison.get("saved_fraction") == saved_fraction
        and traffic.get("known_code_path_device_transfer_total")
        == {"per_actor_bytes": 1_641_660_416, "all_actor_bytes": 6_566_641_664},
        "V66d/V73B traffic comparison changed",
    )
    candidate_audits = traffic.get("candidate_audit_consensus", {})
    validate_embedded_hash(
        candidate_audits, "consensus_sha256", "candidate audit consensus"
    )
    population_acceptance = traffic.get("population_acceptance", {})
    update_invariants = traffic.get("update_invariants", {})
    population_tokens = population_acceptance.get("tokens")
    update_tokens = update_invariants.get("update_acceptance_sha256_by_rank")
    _require(
        candidate_audits.get("actor_count") == 4
        and candidate_audits.get("candidate_count") == 16
        and candidate_audits.get("candidate_count_per_actor") == 4
        and candidate_audits.get("all_rewards_provisional") is True
        and len({
            item
            for row in candidate_audits.get("by_rank", [])
            for item in row.get("candidate_audit_sha256", [])
        }) == 16
        and population_acceptance.get("rank_local_not_broadcast") is True
        and isinstance(population_tokens, list) and len(population_tokens) == 4
        and len(set(population_tokens)) == 4
        and population_tokens
        == update_invariants.get("population_acceptance_sha256_by_rank")
        and update_invariants.get("rank_local_tokens_not_broadcast") is True
        and isinstance(update_tokens, list) and len(update_tokens) == 4
        and len(set(update_tokens)) == 4
        and update_invariants.get("candidate_master_sha256")
        == LIVE_CANDIDATE_SHA256
        and update_invariants.get("candidate_runtime_values_sha256")
        == LIVE_RUNTIME_SHA256,
        "rank-local candidate/population/update acceptance changed",
    )
    residency = _validate_residency(traffic["host_state_residency"])
    _require(
        update.get("host_state_residency")
        == {
            key: traffic["host_state_residency"][key]
            for key in ("post_install", "update_executed", "post_abort")
        },
        "update/audit residency linkage changed",
    )

    unsafe_true = (
        report.get("checkpoint_snapshot_or_promotion_performed"),
        report.get("protected_dev_ood_or_holdout_opened"),
        report.get("raw_questions_answers_or_outputs_persisted"),
        population.get("protected_dev_ood_or_holdout_opened"),
        population.get("raw_questions_answers_or_outputs_persisted"),
        update.get("checkpoint_snapshot_or_promotion_performed"),
        update.get("protected_dev_ood_or_holdout_opened"),
        update.get("master_committed"),
        equivalence.get("master_committed"),
    )
    _require(
        all(value is False for value in unsafe_true)
        and update.get("all_four_abort_receipts_exact") is True
        and update.get("candidate_differs_from_master") is True
        and update.get("candidate_runtime_differs_from_master") is True
        and update.get("population_acceptance_rank_local") is True
        and update.get("update_acceptance_rank_local") is True
        and update.get("v72_two_bank_peak_observed") is True
        and report.get("final_master_sha256") == MASTER_SHA256
        and report.get("final_runtime_values_sha256") == MASTER_RUNTIME_SHA256
        and report.get("all_v71_rewards_accepted_before_update_math") is True
        and report.get("all_v72_state_generations_one_or_two_bank_bounded") is True
        and report.get("historical_reward_floats_used_as_acceptance_gate") is False
        and report.get("same_live_reward_vector_used_by_both_compilers") is True
        and report.get("same_live_compiler_output_whole_mapping_exact") is True,
        "no-commit/exact-abort/protected authority changed",
    )
    _require(
        report.get("base_model_prelaunch_artifact_receipt")
        == report.get("base_model_postrun_artifact_receipt")
        and report["base_model_prelaunch_artifact_receipt"].get(
            "all_26_weight_shard_bytes_and_sha256_verified"
        )
        is True
        and report["base_model_prelaunch_artifact_receipt"].get(
            "all_14_non_weight_file_bytes_and_sha256_verified"
        )
        is True,
        "base model pre/post artifact identity changed",
    )
    _require(
        host_summary.get("phase_transitions") == report.get("phase_transitions")
        and host_summary.get("controller_operations")
        == report.get("controller_operations"),
        "host/report timing linkage changed",
    )
    return {
        "same_live": {
            "live_reward_sha256": LIVE_REWARD_SHA256,
            "coefficient_sha256": coefficient_sha,
            "canonical_and_independent_compilers_whole_mapping_exact": True,
            "historical_reward_bits_required": False,
            "historical_reward_drift": pop_equiv["historical_reward_delta"],
            "live_candidate_master_sha256": LIVE_CANDIDATE_SHA256,
            "live_candidate_runtime_sha256": LIVE_RUNTIME_SHA256,
            "four_actor_live_identity_consensus": True,
        },
        "traffic": {
            "per_actor_worker_counter": EXPECTED_PER_ACTOR_TRAFFIC,
            "aggregate_worker_counter": EXPECTED_AGGREGATE_TRAFFIC,
            "v66d_base_plus_lora_d2h_bytes": V66D_D2H_BYTES,
            "v73b_base_plus_lora_d2h_bytes": V73B_D2H_BYTES,
            "saved_d2h_bytes": SAVED_D2H_BYTES,
            "saved_d2h_fraction": saved_fraction,
            "saved_d2h_percent": 100.0 * saved_fraction,
            "known_code_path_total_device_transfer_all_actors_bytes": 6_566_641_664,
            "worker_counter_exact_match_all_four_actors": True,
        },
        "host_state_residency": residency,
        "host_process": host,
        "safety": {
            "all_16_population_candidates_exact_to_v66d": True,
            "all_16_population_restores_exact": True,
            "all_16_actor_work_receipts_exact_to_v66d": actor[
                "semantic_equivalence_to_v66d"
            ],
            "all_four_update_aborts_exact": True,
            "final_master_and_runtime_exact": True,
            "master_committed": False,
            "checkpoint_or_promotion_performed": False,
            "protected_dev_ood_or_holdout_opened": False,
            "raw_questions_answers_or_outputs_persisted": False,
            "base_model_pre_post_identity_exact": True,
        },
    }


def validate_cleanup(report: Mapping[str, Any]) -> dict[str, Any]:
    cleanup = report.get("cleanup", {})
    before = cleanup.get("before")
    after = cleanup.get("after")
    _require(
        isinstance(before, list) and len(before) == 4
        and isinstance(after, list) and len(after) == 4
        and all(row.get("state") == "CREATED" for row in before)
        and all(row.get("state") == "REMOVED" for row in after)
        and {row["placement_group_id"] for row in before}
        == {row["placement_group_id"] for row in after}
        and cleanup.get("all_four_gcs_states_removed") is True
        and cleanup.get("engine_kill_count") == 4
        and cleanup.get("placement_group_remove_count") == 4
        and cleanup.get("driver_scoped_non_detached_by_construction") is True
        and report.get("final_gpu_idle", {}).get(
            "all_four_compute_process_lists_empty"
        )
        is True,
        "V73B cleanup/final idle changed",
    )
    return {
        "placement_groups_created_then_removed": 4,
        "engine_kill_count": 4,
        "all_four_gcs_states_removed": True,
        "all_four_compute_process_lists_empty": True,
    }


def _phase_map(telemetry: Mapping[str, Any]) -> dict[str, float]:
    return {
        row["phase"]: row["epoch_upper_window_seconds"]
        for row in telemetry["phase_epochs"]
    }


def _controller_windows(
    transitions: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    result = {}
    for left, right in zip(transitions, transitions[1:]):
        _require(
            left["ended_ns"] <= right["started_ns"]
            and left["elapsed_ns"] == left["ended_ns"] - left["started_ns"],
            "controller phase transition timing changed",
        )
        result[left["to"]] = (right["started_ns"] - left["ended_ns"]) / 1e9
    return result


def timing_analysis(
    report: Mapping[str, Any],
    v66d_report: Mapping[str, Any],
    actor: Mapping[str, Any],
    v66d_actor: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    v66d_telemetry: Mapping[str, Any],
    host: Mapping[str, Any],
) -> dict[str, Any]:
    v73_phase = _phase_map(telemetry)
    v66_phase = _phase_map(v66d_telemetry)

    def four_sum(mapping: Mapping[str, float], template: str) -> float:
        return sum(mapping[template.format(wave)] for wave in range(4))

    common = {}
    for label, template in (
        ("candidate_materialization", "mirrored_wave_{}_materialize_all_actors"),
        ("generation", "mirrored_wave_{}_generation_all_actors"),
        ("exact_restore", "wave_{}_finalize_restore_all_actors"),
    ):
        old = four_sum(v66_phase, template)
        new = four_sum(v73_phase, template)
        common[label] = {
            "v66d_sampled_epoch_upper_window_seconds": old,
            "v73b_sampled_epoch_upper_window_seconds": new,
            "v73b_minus_v66d_seconds": new - old,
            "observed_reduction_fraction": 1.0 - new / old,
        }
    for label, old_name, new_name in (
        (
            "canonical_install",
            "install_canonical_v434_master_all_actors",
            "install_canonical_v434_master_all_actors",
        ),
        (
            "update_prepare",
            "pair_difference_update_prepare_all_actors",
            "pair_difference_update_accept_prepare_all_actors",
        ),
        (
            "update_execute",
            "pair_difference_update_execute_all_actors",
            "pair_difference_update_execute_all_actors",
        ),
        (
            "update_abort_observed_span",
            "pair_difference_update_abort_all_actors",
            "pair_difference_update_abort_all_actors",
        ),
    ):
        old, new = v66_phase[old_name], v73_phase[new_name]
        common[label] = {
            "v66d_sampled_epoch_upper_window_seconds": old,
            "v73b_sampled_epoch_upper_window_seconds": new,
            "v73b_minus_v66d_seconds": new - old,
            "observed_reduction_fraction": 1.0 - new / old,
        }
    # The final abort has no following epoch; both values are sampled first-to-
    # last spans.  Prepare is not semantically identical because V73B includes
    # exact V71 acceptance.  Preserve those qualifications in machine data.
    common["update_abort_observed_span"]["scope_qualification"] = (
        "last_phase_first_to_last_sample_span_not_full_cleanup_window"
    )
    common["update_prepare"]["scope_qualification"] = (
        "v73b_includes_v71_rank_local_update_acceptance"
    )

    def end_to_end(field: str, old: float, new: float) -> dict[str, float]:
        return {
            f"v66d_{field}": old,
            f"v73b_{field}": new,
            "v73b_minus_v66d": new - old,
            "observed_reduction_fraction": 1.0 - new / old,
        }

    v66_cuda = v66d_actor["cuda_event_ms"]
    v73_cuda = actor["cuda_event_ms"]
    cuda = {
        "scope": "per_candidate_actor_cuda_event_generation_only",
        "v66d": v66_cuda,
        "v73b": v73_cuda,
        "median_delta_fraction": v73_cuda["median"] / v66_cuda["median"] - 1.0,
        "mean_delta_fraction": v73_cuda["mean"] / v66_cuda["mean"] - 1.0,
        "critical_path_sum_delta_fraction": (
            v73_cuda["critical_path_sum_seconds"]
            / v66_cuda["critical_path_sum_seconds"] - 1.0
        ),
        "generation_speedup_observed": False,
    }
    exact_windows = _controller_windows(report["phase_transitions"])
    return {
        "scope_rules": {
            "sampled_epoch_upper_window": (
                "first telemetry timestamp in a phase through first timestamp "
                "in the next phase; monitor-granularity upper window"
            ),
            "controller_phase_window": (
                "phase-ack completion through start of the next phase transition; "
                "high-resolution V73B only"
            ),
            "worker_rpc_duration": (
                "per-worker method receipt; four actors overlap and durations "
                "must not be summed as wall time"
            ),
            "actor_cuda_event": "generation CUDA work only",
            "cross_run_inference": (
                "single accepted run per implementation; observed differences "
                "are descriptive and not a causal benchmark"
            ),
        },
        "common_scope_sampled_epoch_windows": common,
        "end_to_end": {
            "wall": end_to_end(
                "wall_runtime_seconds",
                v66d_report["wall_runtime_seconds"],
                report["wall_runtime_seconds"],
            ),
            "model_resident_per_gpu": end_to_end(
                "model_resident_seconds_per_gpu",
                v66d_report["compute_ledger"]["model_resident_seconds_per_gpu"],
                report["compute_ledger"]["model_resident_seconds_per_gpu"],
            ),
            "charged_gpu": end_to_end(
                "charged_gpu_seconds",
                v66d_report["compute_ledger"]["charged_gpu_seconds"],
                report["compute_ledger"]["charged_gpu_seconds"],
            ),
        },
        "actor_cuda_generation": cuda,
        "v73b_controller_phase_windows_seconds": exact_windows,
        "v73b_worker_rpc_seconds": host["worker_rpc_seconds"],
        "interpretation": {
            "generation_cuda_critical_path_was_faster": False,
            "state_and_audit_windows_materially_reduced": True,
            "overall_wall_time_materially_reduced": True,
            "do_not_attribute_overall_gain_to_model_inference": True,
        },
    }


def analyze_finalized() -> dict[str, Any]:
    static = validate_static_identity()
    artifacts = {
        "traffic": _load_json(RUN / "exact_audit_traffic_v73b.json"),
        "host_summary": _load_json(RUN / "host_process_summary_v73b.json"),
        "report": _load_json(RUN / "mirrored_calibration_report_v73b.json"),
        "population": _load_json(RUN / "mirrored_population_v73b.json"),
        "update": _load_json(RUN / "pair_difference_update_v73b.json"),
        "equivalence": _load_json(RUN / "same_live_equivalence_v73b.json"),
        "control_report": _load_json(
            V66D_RUN / "mirrored_calibration_report_v66d.json"
        ),
        "control_population": _load_json(
            V66D_RUN / "mirrored_population_v66d.json"
        ),
        "control_update": _load_json(
            V66D_RUN / "pair_difference_update_v66d.json"
        ),
    }
    for name in ("control_report", "control_population", "control_update"):
        validate_self_hash(artifacts[name], name)

    actor = validate_actor_receipts(
        RUN / "actor_cuda_work_receipts_v73b.jsonl",
        control_path=V66D_RUN / "actor_cuda_work_receipts_v66d.jsonl",
    )
    v66d_actor = validate_actor_receipts(
        V66D_RUN / "actor_cuda_work_receipts_v66d.jsonl"
    )
    telemetry = validate_gpu_telemetry(
        RUN / "gpu_activity_v73b.jsonl", actor["bindings"]
    )
    v66d_telemetry = validate_gpu_telemetry(
        V66D_RUN / "gpu_activity_v66d.jsonl", v66d_actor["bindings"]
    )
    _require(
        telemetry["row_count"] == 372
        and telemetry["complete_four_gpu_batches"] == 93
        and telemetry["all_four_gpus_positive"] is True
        and all(
            row["peak_memory_used_mib"] == 84_140
            for row in telemetry["per_gpu"].values()
        ),
        "V73B useful four-GPU telemetry changed",
    )
    host = validate_host_telemetry(
        RUN / "host_process_samples_v73b.jsonl",
        artifacts["host_summary"],
        actor["bindings"],
    )
    _require(
        host["sample_count"] == 412
        and all(row["sample_count"] == 103 for row in host["actors"].values()),
        "V73B host telemetry cardinality changed",
    )
    semantics = validate_semantics(artifacts, actor, host)
    cleanup = validate_cleanup(artifacts["report"])
    timing = timing_analysis(
        artifacts["report"], artifacts["control_report"],
        actor, v66d_actor, telemetry, v66d_telemetry, host,
    )
    gpu_peak = max(
        row["peak_memory_used_mib"] for row in telemetry["per_gpu"].values()
    )
    v66d_gpu_peak = max(
        row["peak_memory_used_mib"]
        for row in v66d_telemetry["per_gpu"].values()
    )
    result = {
        "schema": "qwen36-v73b-lora-calibration-postrun-analysis-v1",
        "status": "accepted_same_live_v71_v72_lora_gate",
        "passed": True,
        "static_identity": static,
        "artifact_validation": {
            "all_json_artifact_self_hashes_valid": True,
            "all_16_actor_receipt_self_hashes_valid": True,
            "all_412_host_sample_self_hashes_valid": True,
            "all_file_hashes_and_cross_references_valid": True,
            "v73b_actor_receipts": {
                key: value for key, value in actor.items() if key != "rows"
            },
            "v73b_gpu_telemetry": telemetry,
        },
        **semantics,
        "vram": {
            "v66d_peak_memory_used_mib_per_gpu": v66d_gpu_peak,
            "v73b_peak_memory_used_mib_per_gpu": gpu_peak,
            "observed_delta_mib": gpu_peak - v66d_gpu_peak,
            "v73b_minimum_physical_headroom_mib": MEMORY_TOTAL_MIB - gpu_peak,
            "interpretation": (
                "V72 is a host-state ownership optimization; sampled GPU peak "
                "is effectively unchanged at this telemetry granularity"
            ),
        },
        "timing": timing,
        "cleanup": cleanup,
        "task_disposition": {
            "specialist-0j5.29": "close_live_gate_fully_satisfied",
            "specialist-0j5.21": (
                "close_lora_exact_audit_traffic_and_live_final_boundary_satisfied"
            ),
            "specialist-0j5.14": (
                "keep_open_cupti_hbm_cuda_allocator_and_checkpoint_phases_missing"
            ),
            "specialist-0j5.19": (
                "keep_open_lora_subarm_accepted_but_dense_fullweight_shared_master_"
                "and_live_checkpoint_work_unproven"
            ),
        },
        "authority": {
            "analysis_is_cpu_only": True,
            "dataset_model_ray_or_gpu_opened_by_analyzer": False,
            "checkpoint_snapshot_commit_or_promotion_authorized": False,
            "protected_dev_ood_or_holdout_opened": False,
            "dense_fullweight_ownership_inferred_from_lora": False,
        },
        "analyzer": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "file_sha256": file_sha256(Path(__file__).resolve()),
        },
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def render_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True
    ) + "\n"


def render_markdown(value: Mapping[str, Any]) -> str:
    traffic = value["traffic"]
    host = value["host_process"]
    timing = value["timing"]
    common = timing["common_scope_sampled_epoch_windows"]
    wall = timing["end_to_end"]["wall"]
    cuda = timing["actor_cuda_generation"]
    vram = value["vram"]
    drift = value["same_live"]["historical_reward_drift"]
    return f"""# Qwen3.6 V73B LoRA live-gate postrun analysis

Date: 2026-07-17 UTC

Verdict: **accepted** for the V71 exact-audit plus V72 LoRA host-ownership
gate.  The run is train-only evidence: no model update was committed, no
checkpoint or promotion occurred, and no protected dev/OOD/holdout data was
opened.

## Exact acceptance

- All self-hashes passed for the launch journal and six JSON run artifacts.
- All 16 actor CUDA receipts and all 412 host-process samples passed their
  per-row hashes.  The 16 work assignments/cardinalities exactly match V66d,
  with four candidates on each of four physical GPUs over four waves.
- The live reward matrix was finite, complete, and assignment-exact.  Its
  digest is `{value['same_live']['live_reward_sha256']}`.
- The canonical V66 compiler and independent one-pass compiler produced the
  same complete result mapping from that same live vector.  Coefficient digest:
  `{value['same_live']['coefficient_sha256']}`.
- All four actors agreed on candidate `{value['same_live']['live_candidate_master_sha256']}`
  and runtime `{value['same_live']['live_candidate_runtime_sha256']}`.
- All 16 population restores and all four update aborts returned exactly to
  the original master/runtime; final cleanup removed all four placement groups
  and left all four compute-process lists empty.

Historical reward bits remained diagnostic, not an acceptance threshold:
maximum absolute drift was {drift['maximum_absolute']:.12g}, and 5/8 pair
preference signs matched V66d.  No post-hoc tolerance was applied.

## Traffic and memory

| Measurement | V66d | V73B | Observed change |
|---|---:|---:|---:|
| Base + LoRA D2H bytes | {traffic['v66d_base_plus_lora_d2h_bytes']:,} | {traffic['v73b_base_plus_lora_d2h_bytes']:,} | -{traffic['saved_d2h_percent']:.4f}% |
| Peak GPU memory / GPU | {vram['v66d_peak_memory_used_mib_per_gpu']:,} MiB | {vram['v73b_peak_memory_used_mib_per_gpu']:,} MiB | {vram['observed_delta_mib']:+,} MiB |
| Wall runtime | {wall['v66d_wall_runtime_seconds']:.3f} s | {wall['v73b_wall_runtime_seconds']:.3f} s | -{100 * wall['observed_reduction_fraction']:.2f}% |

Worker counters exactly matched the preregistered values on every actor:
{traffic['aggregate_worker_counter']['total_device_transfer_bytes']:,} bytes
aggregate, with zero repeated master-validation host-copy bytes.  Including
explicitly accounted transfers outside those worker counters gives
{traffic['known_code_path_total_device_transfer_all_actors_bytes']:,} bytes.

V72 ownership followed one/two/one/one banks per actor: 18,112,512 bytes after
install, 36,225,024 bytes with the executed candidate retained, 18,112,512
bytes after abort, and 18,112,512 bytes at final quiescence.  Host telemetry
sampled all actors on NUMA node 0; maximum sampled RSS was
{host['maximum_sampled_rss_bytes']:,} bytes, maximum reported HWM was
{host['maximum_reported_hwm_bytes']:,} bytes, major-fault delta was zero, and
fault counters were monotonic.  Pinned bytes remained
{host['maximum_pinned_bytes']:,}; this run does not validate a future pinned
staging arm.

## Timing scopes

| Comparable sampled epoch window | V66d | V73B | Reduction |
|---|---:|---:|---:|
| Canonical install | {common['canonical_install']['v66d_sampled_epoch_upper_window_seconds']:.3f} s | {common['canonical_install']['v73b_sampled_epoch_upper_window_seconds']:.3f} s | {100 * common['canonical_install']['observed_reduction_fraction']:.1f}% |
| Four materialization waves | {common['candidate_materialization']['v66d_sampled_epoch_upper_window_seconds']:.3f} s | {common['candidate_materialization']['v73b_sampled_epoch_upper_window_seconds']:.3f} s | {100 * common['candidate_materialization']['observed_reduction_fraction']:.1f}% |
| Four generation waves | {common['generation']['v66d_sampled_epoch_upper_window_seconds']:.3f} s | {common['generation']['v73b_sampled_epoch_upper_window_seconds']:.3f} s | {100 * common['generation']['observed_reduction_fraction']:.1f}% |
| Four restore waves | {common['exact_restore']['v66d_sampled_epoch_upper_window_seconds']:.3f} s | {common['exact_restore']['v73b_sampled_epoch_upper_window_seconds']:.3f} s | {100 * common['exact_restore']['observed_reduction_fraction']:.1f}% |
| Update execute | {common['update_execute']['v66d_sampled_epoch_upper_window_seconds']:.3f} s | {common['update_execute']['v73b_sampled_epoch_upper_window_seconds']:.3f} s | {100 * common['update_execute']['observed_reduction_fraction']:.1f}% |
| Final abort sampled span | {common['update_abort_observed_span']['v66d_sampled_epoch_upper_window_seconds']:.3f} s | {common['update_abort_observed_span']['v73b_sampled_epoch_upper_window_seconds']:.3f} s | {100 * common['update_abort_observed_span']['observed_reduction_fraction']:.1f}% |

The table uses the same monitor-derived epoch-window definition for both runs.
It is not interchangeable with per-worker RPC duration.  V73B's worker RPC
medians were {host['worker_rpc_seconds']['candidate_materialization']['median']:.3f}
seconds for candidate materialization and
{host['worker_rpc_seconds']['exact_restore']['median']:.3f} seconds for restore;
four actors overlap, so these durations are not summed as serial wall time.
The final-abort row is a first-to-last sampled span because there is no next
phase epoch, and V73B update-prepare includes a V71 acceptance boundary absent
from V66d.

Generation itself was not faster.  The actor CUDA critical-path sum was
{cuda['v66d']['critical_path_sum_seconds']:.3f} seconds in V66d and
{cuda['v73b']['critical_path_sum_seconds']:.3f} seconds in V73B
({100 * cuda['critical_path_sum_delta_fraction']:+.2f}%).  The observed overall
gain therefore aligns with state/audit-window reductions, not faster model
inference.  This is one accepted run per implementation, so timing differences
are descriptive rather than a causal replicated benchmark.

## Task disposition

- Close `specialist-0j5.29`: every live acceptance condition passed.
- Close `specialist-0j5.21`: the LoRA exact-audit traffic reduction, exact
  final boundary, abort, and cleanup gates passed.
- Keep `specialist-0j5.14` open: CUPTI/equivalent HBM bandwidth, CUDA allocator
  decomposition, and live checkpoint-phase evidence remain absent.
- Keep `specialist-0j5.19` open: this accepts the LoRA one/two/one ownership
  sub-arm only.  It does not establish dense full-weight shared/mmap ownership
  or a live checkpoint path.

Canonical machine analysis content SHA-256:
`{value['content_sha256_before_self_field']}`.
"""


def _atomic_write(path: Path, payload: str) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload.encode("ascii"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--markdown", default=str(MARKDOWN))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    value = analyze_finalized()
    json_payload = render_json(value)
    markdown_payload = render_markdown(value)
    output = Path(args.output).resolve()
    markdown = Path(args.markdown).resolve()
    if args.check:
        _require(output.is_file(), f"postrun JSON missing: {output}")
        _require(markdown.is_file(), f"postrun Markdown missing: {markdown}")
        _require(
            output.read_text(encoding="ascii") == json_payload,
            f"postrun JSON stale: {output}",
        )
        _require(
            markdown.read_text(encoding="ascii") == markdown_payload,
            f"postrun Markdown stale: {markdown}",
        )
    else:
        if output.exists() or markdown.exists():
            raise FileExistsError("postrun output already exists; use --check")
        _atomic_write(output, json_payload)
        _atomic_write(markdown, markdown_payload)
    print(json.dumps({
        "passed": value["passed"],
        "content_sha256": value["content_sha256_before_self_field"],
        "output": str(output),
        "markdown": str(markdown),
        "checked": bool(args.check),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
