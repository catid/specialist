#!/usr/bin/env python3
"""Fail-closed, CPU-only analysis of finalized V66d ES telemetry.

This module never imports torch, Ray, vLLM, pynvml, or dataset code.  It binds
the sealed V66d implementation, validates the durable actor/NVML receipts, and
reports two deliberately different forms of bandwidth evidence:

* sampled NVML PCIe integrals (estimates over the sampled intervals); and
* implementation/metadata-derived transfer byte lower bounds.

The latter are lower bounds, not measurements.  Keeping those claims separate
prevents a sparse throughput counter from being presented as an exact byte
count and prevents a source audit from being presented as observed traffic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import eggroll_es_gpu_telemetry_v66 as telemetry


SCHEMA_V71 = "eggroll-es-v66d-phase-memory-analysis-v71"
PREREGISTRATION_SCHEMA_V71 = (
    "eggroll-es-v66d-phase-memory-analysis-preregistration-v71"
)
ROOT = Path(__file__).resolve().parent
RUN_DIRECTORY_V66D = ROOT / (
    "experiments/eggroll_es_hpo/runs/"
    "v66d_lora_es_mirrored_crn_qwen36_calibration"
)
PREREGISTRATION_V71 = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "v66d_phase_memory_analysis_v71.json"
)
STAGE_MANIFEST = ROOT / (
    "experiments/eggroll_es_hpo/staged_adapters/"
    "v434_equal_sft_qwen35_vllm_namespace_v49d/stage_manifest_v44a.json"
)
TOPOLOGY_REPORT = ROOT / (
    "experiments/eggroll_es_hpo/runs/"
    "v40c_v37_lora_topology_probe_tuned_projection_retry/"
    "lora_topology_report_v40c.json"
)

CANONICAL_TENSORS = 70
CANONICAL_ELEMENTS = 4_528_128
CANONICAL_DTYPE_BYTES = 4
RUNTIME_VIEWS = 82
RUNTIME_ELEMENTS = 4_921_344
RUNTIME_DTYPE_BYTES = 2
BASE_LAYER_TENSORS = 23
BASE_LAYER_ELEMENTS = 142_999_552
BASE_LAYER_BYTES = 285_999_104

EXPECTED_STATIC_SHA256 = {
    "eggroll_es_gpu_telemetry_v66.py": (
        "6c086f55ea2e5f66c1c01056b3a6acee7b13074ac9af00c693bea7848cd08224"
    ),
    "eggroll_es_worker_lora_v66d.py": (
        "807af52dab4b842f0a74b33cf083f8b67a7e2f4bd0329eb1ca08c4ffb3831ed6"
    ),
    "run_lora_es_mirrored_calibration_v66d.py": (
        "3c3c10ea6e33190c47f7bbebbbbb49eeecf84ee2486501f258f2c97302adf015"
    ),
    "eggroll_es_mirrored_v66.py": (
        "06b8cfd775051e1a20d30f969442cdcdc1b2b56ad10c1291d287f031a47594ad"
    ),
    "run_lora_es_mirrored_calibration_v66.py": (
        "b0dd2cbbbcb7eeaf040fe4c569aaaa6d565157417ac707896d8176503c6f1bc9"
    ),
    "eggroll_es_worker_lora_v66.py": (
        "ee976fbc56a720c5c2a5e52d86c7a02d1e8c7414ed7383952d3da03e72944a03"
    ),
    "eggroll_es_worker_lora_v41a.py": (
        "cc40337eba30fe0748996c22dcbf3914b8c12249f6e2e47d6128aadee575494c"
    ),
    "eggroll_es_worker_lora_topology_v40a.py": (
        "d487ff4657a2a6fbd7d2d0f9200a63547dd79b258cc488b93382fdf26b52cf05"
    ),
    "experiments/eggroll_es_hpo/preregistrations/"
    "lora_es_mirrored_calibration_v66d.json": (
        "3269f7138d74266538cc3b0f31e1a904808f8f3751dde5a7a9456e93b13314b0"
    ),
}
EXPECTED_STAGE_MANIFEST_SHA256 = (
    "e30ba44563b5db56f4a487b26f4e2310fd3755b15f8db69d9400facd8baa3813"
)
EXPECTED_STAGE_CONTENT_SHA256 = (
    "ea328ada018e1c0d182d329d2a9cb81f8f0375aef93738f3b9c0a00f63c82da3"
)
EXPECTED_WEIGHTS_SHA256 = (
    "7a41d921c6988dc62dca092230ed5ccfd5d6568a600503c87ff086cb2763485a"
)
EXPECTED_CONFIG_SHA256 = (
    "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5"
)
EXPECTED_TOPOLOGY_REPORT_SHA256 = (
    "7672f835239b91e66a03a512ad9fbe3cbbaff31783a7b1a26bdd136d98b55050"
)
EXPECTED_TOPOLOGY_CONTENT_SHA256 = (
    "9394ed06a80fffb1c4cc1532ac59741ab4c4a1c5a481136f29415b463eaf747d"
)
EXPECTED_V66D_PREREG_CONTENT_SHA256 = (
    "2f8e23b643507b594b05719966da1b9bcc64a2b7f412021066df4c6418144531"
)


def canonical_sha256_v71(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def file_sha256_v71(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_self_hashed_json(path: Path, expected_schema: str | None = None) -> dict:
    value = _load_json(path)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact is not an object: {path}")
    claimed = value.get("content_sha256_before_self_field")
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        not isinstance(claimed, str)
        or claimed != canonical_sha256_v71(compact)
        or (expected_schema is not None and value.get("schema") != expected_schema)
    ):
        raise RuntimeError(f"self-hashed JSON identity changed: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    result = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="ascii").splitlines(), start=1,
    ):
        if not line:
            raise RuntimeError(f"blank JSONL row at {path}:{line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"JSONL row is not an object at {path}:{line_number}")
        result.append(value)
    if not result:
        raise RuntimeError(f"JSONL artifact is empty: {path}")
    return result


def verify_static_contract_v71(root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    observed = {}
    for relative, expected in EXPECTED_STATIC_SHA256.items():
        path = root / relative
        actual = file_sha256_v71(path)
        if actual != expected:
            raise RuntimeError(f"sealed implementation identity changed: {relative}")
        observed[relative] = actual
    v66d_prereg = _load_self_hashed_json(
        root / (
            "experiments/eggroll_es_hpo/preregistrations/"
            "lora_es_mirrored_calibration_v66d.json"
        ),
        "lora-es-mirrored-crn-qwen36-calibration-preregistration-v66d",
    )
    if (
        v66d_prereg["content_sha256_before_self_field"]
        != EXPECTED_V66D_PREREG_CONTENT_SHA256
    ):
        raise RuntimeError("V66d preregistration content identity changed")
    return {
        "schema": "v66d-static-analysis-contract-v71",
        "files": observed,
        "v66d_preregistration_content_sha256": (
            EXPECTED_V66D_PREREG_CONTENT_SHA256
        ),
    }


def load_adapter_metadata_v71(
    stage_manifest_path: Path = STAGE_MANIFEST,
    topology_report_path: Path = TOPOLOGY_REPORT,
) -> dict:
    stage_manifest_path = Path(stage_manifest_path).resolve()
    topology_report_path = Path(topology_report_path).resolve()
    if file_sha256_v71(stage_manifest_path) != EXPECTED_STAGE_MANIFEST_SHA256:
        raise RuntimeError("V434 stage manifest file identity changed")
    stage = _load_self_hashed_json(
        stage_manifest_path, "candidate-lora-vllm-stage-manifest-v44a",
    )
    if stage["content_sha256_before_self_field"] != EXPECTED_STAGE_CONTENT_SHA256:
        raise RuntimeError("V434 stage manifest content identity changed")
    records = stage.get("tensor_mapping_records")
    if not isinstance(records, list) or len(records) != CANONICAL_TENSORS:
        raise RuntimeError("V434 canonical tensor mapping count changed")
    if any(
        not isinstance(item, dict)
        or item.get("dtype") != "torch.float32"
        or isinstance(item.get("elements"), bool)
        or not isinstance(item.get("elements"), int)
        or item["elements"] <= 0
        for item in records
    ):
        raise RuntimeError("V434 canonical tensor metadata changed")
    canonical_elements = sum(item["elements"] for item in records)
    if canonical_elements != CANONICAL_ELEMENTS:
        raise RuntimeError("V434 canonical tensor element count changed")
    directory = stage_manifest_path.parent
    weights = directory / "adapter_model.safetensors"
    config = directory / "adapter_config.json"
    if (
        stage.get("artifact", {}).get("weights_file_sha256")
        != EXPECTED_WEIGHTS_SHA256
        or stage.get("artifact", {}).get("adapter_config_file_sha256")
        != EXPECTED_CONFIG_SHA256
        or file_sha256_v71(weights) != EXPECTED_WEIGHTS_SHA256
        or file_sha256_v71(config) != EXPECTED_CONFIG_SHA256
    ):
        raise RuntimeError("V434 staged adapter byte identity changed")

    if file_sha256_v71(topology_report_path) != EXPECTED_TOPOLOGY_REPORT_SHA256:
        raise RuntimeError("V40 runtime topology report file identity changed")
    topology = _load_self_hashed_json(
        topology_report_path, "lora-topology-probe-report-v40a",
    )
    if topology["content_sha256_before_self_field"] != EXPECTED_TOPOLOGY_CONTENT_SHA256:
        raise RuntimeError("V40 runtime topology report content identity changed")
    inventory = topology.get("inventory")
    base = inventory.get("base_layer_weights") if isinstance(inventory, dict) else None
    if (
        not isinstance(inventory, dict)
        or inventory.get("peft_tensor_count") != CANONICAL_TENSORS
        or inventory.get("peft_elements") != CANONICAL_ELEMENTS
        or inventory.get("peft_dtype_counts") != {"torch.float32": CANONICAL_TENSORS}
        or inventory.get("runtime_view_count") != RUNTIME_VIEWS
        or inventory.get("runtime_active_allocated_elements") != RUNTIME_ELEMENTS
        or inventory.get("runtime_dtypes") != ["torch.bfloat16"]
        or inventory.get("runtime_module_count") != BASE_LAYER_TENSORS
        or not isinstance(base, dict)
        or base.get("tensor_count") != BASE_LAYER_TENSORS
        or base.get("elements") != BASE_LAYER_ELEMENTS
        or base.get("bytes") != BASE_LAYER_BYTES
    ):
        raise RuntimeError("V40 adapter/runtime/base topology geometry changed")
    return {
        "schema": "v434-vllm-adapter-transfer-metadata-v71",
        "canonical_tensor_count": CANONICAL_TENSORS,
        "canonical_elements": CANONICAL_ELEMENTS,
        "canonical_dtype": "torch.float32",
        "canonical_bytes": CANONICAL_ELEMENTS * CANONICAL_DTYPE_BYTES,
        "runtime_view_count": RUNTIME_VIEWS,
        "runtime_elements": RUNTIME_ELEMENTS,
        "runtime_dtype": "torch.bfloat16",
        "runtime_bytes": RUNTIME_ELEMENTS * RUNTIME_DTYPE_BYTES,
        "base_layer_tensor_count": BASE_LAYER_TENSORS,
        "base_layer_elements": BASE_LAYER_ELEMENTS,
        "base_layer_bytes": BASE_LAYER_BYTES,
        "stage_manifest_file_sha256": EXPECTED_STAGE_MANIFEST_SHA256,
        "stage_manifest_content_sha256": EXPECTED_STAGE_CONTENT_SHA256,
        "weights_file_sha256": EXPECTED_WEIGHTS_SHA256,
        "config_file_sha256": EXPECTED_CONFIG_SHA256,
        "runtime_topology_report_file_sha256": EXPECTED_TOPOLOGY_REPORT_SHA256,
        "runtime_topology_report_content_sha256": EXPECTED_TOPOLOGY_CONTENT_SHA256,
    }


def _validate_adapter_metadata_v71(metadata: Mapping[str, Any]) -> dict:
    expected = load_adapter_metadata_v71()
    if not isinstance(metadata, Mapping) or dict(metadata) != expected:
        raise RuntimeError("adapter transfer metadata is incomplete or changed")
    return expected


def expected_phases_v71(wave_count: int) -> list[str]:
    if isinstance(wave_count, bool) or not isinstance(wave_count, int) or wave_count <= 0:
        raise RuntimeError("V66d wave count must be a positive exact integer")
    phases = [
        "activate_v434_lora_slot_all_actors",
        "install_canonical_v434_master_all_actors",
    ]
    for wave in range(wave_count):
        phases.extend([
            f"mirrored_wave_{wave}_materialize_all_actors",
            f"mirrored_wave_{wave}_generation_all_actors",
            f"wave_{wave}_finalize_restore_all_actors",
        ])
    phases.extend([
        "pair_difference_update_prepare_all_actors",
        "pair_difference_update_execute_all_actors",
        "pair_difference_update_abort_all_actors",
    ])
    return phases


def _sampled_integral(rows: Sequence[dict], key: str, multiplier: float) -> dict:
    values = [row[key] for row in rows]
    if len(rows) < 2 or any(value is None for value in values):
        return {
            "supported_samples": sum(value is not None for value in values),
            "total_samples": len(values),
            "covered_seconds": 0.0,
            "sampled_left_rectangle_estimate": None,
        }
    total = 0.0
    for previous, current in zip(rows, rows[1:]):
        elapsed = (current["monotonic_ns"] - previous["monotonic_ns"]) / 1e9
        if elapsed <= 0.0:
            raise RuntimeError("telemetry sample time did not increase")
        total += float(previous[key]) * multiplier * elapsed
    return {
        "supported_samples": len(values),
        "total_samples": len(values),
        "covered_seconds": (
            rows[-1]["monotonic_ns"] - rows[0]["monotonic_ns"]
        ) / 1e9,
        "sampled_left_rectangle_estimate": int(total),
    }


def _mean(values: Sequence[int]) -> float:
    return math.fsum(values) / len(values)


def _validate_rows_and_phases_v71(
    rows: Sequence[dict], bindings: Sequence[dict], wave_count: int,
) -> tuple[list[dict], list[str], dict[str, list[dict]]]:
    bindings = telemetry.canonical_actor_bindings_v66(bindings)
    by_gpu = telemetry.binding_by_gpu_v66(bindings)
    if not isinstance(rows, (list, tuple)) or not rows or len(rows) % 4:
        raise RuntimeError("V66d telemetry must contain complete four-row batches")
    validated = [telemetry._validate_sample_v66(row, by_gpu) for row in rows]
    if [row["sequence"] for row in validated] != list(range(len(validated))):
        raise RuntimeError("V66d telemetry sequence is not exact and contiguous")

    batches = []
    prior_ns = None
    for offset in range(0, len(validated), 4):
        batch = validated[offset:offset + 4]
        if (
            [row["gpu"] for row in batch] != list(telemetry.GPU_IDS_V66)
            or len({row["monotonic_ns"] for row in batch}) != 1
            or len({row["sampled_at_utc"] for row in batch}) != 1
            or len({row["phase"] for row in batch}) != 1
            or len({row["phase_epoch"] for row in batch}) != 1
        ):
            raise RuntimeError("V66d telemetry batch is not one exact four-GPU tick")
        monotonic_ns = batch[0]["monotonic_ns"]
        if prior_ns is not None and monotonic_ns <= prior_ns:
            raise RuntimeError("V66d telemetry batch clock is not strictly increasing")
        prior_ns = monotonic_ns
        sampled = batch[0]["sampled_at_utc"]
        try:
            parsed = datetime.fromisoformat(sampled)
        except (TypeError, ValueError) as error:
            raise RuntimeError("V66d sampled UTC timestamp is invalid") from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise RuntimeError("V66d sampled timestamp is not timezone-aware")
        for row in batch:
            pid = row["expected_pid"]
            memory = row["process_memory_mib"]
            value = memory.get(str(pid), memory.get(pid))
            if (
                pid not in row["compute_pids"]
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                or value > row["memory_used_mib"]
            ):
                raise RuntimeError("V66d expected process residency/memory changed")
        batches.append(batch)

    observed = []
    observed_epochs = []
    for batch in batches:
        phase, epoch = batch[0]["phase"], batch[0]["phase_epoch"]
        if not observed or observed[-1] != phase:
            observed.append(phase)
            observed_epochs.append(epoch)
        elif observed_epochs[-1] != epoch:
            raise RuntimeError("V66d phase label was reused with another epoch")
    required = expected_phases_v71(wave_count)
    expected_observed = (["setup"] if observed and observed[0] == "setup" else []) + required
    if observed != expected_observed:
        raise RuntimeError("V66d phase order/coverage changed")
    expected_epochs = ([0] if expected_observed[0] == "setup" else []) + list(
        range(1, len(required) + 1)
    )
    if observed_epochs != expected_epochs:
        raise RuntimeError("V66d phase epochs are not the sealed handshake sequence")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in validated:
        grouped[row["phase"]].append(row)
    return validated, observed, grouped


def _phase_statistics_v71(
    bindings: Sequence[dict], phases: Sequence[str],
    grouped: Mapping[str, Sequence[dict]],
) -> dict:
    bindings = telemetry.canonical_actor_bindings_v66(bindings)
    by_gpu = telemetry.binding_by_gpu_v66(bindings)
    result = {}
    phase_first = {
        phase: min(row["monotonic_ns"] for row in grouped[phase])
        for phase in phases
    }
    for index, phase in enumerate(phases):
        selected = list(grouped[phase])
        timestamps = sorted({row["monotonic_ns"] for row in selected})
        next_first = phase_first[phases[index + 1]] if index + 1 < len(phases) else None
        per_gpu = {}
        for gpu in telemetry.GPU_IDS_V66:
            gpu_rows = sorted(
                (row for row in selected if row["gpu"] == gpu),
                key=lambda row: row["monotonic_ns"],
            )
            binding = by_gpu[gpu]
            pid = binding["worker_pid"]
            process_memory = [
                row["process_memory_mib"].get(
                    str(pid), row["process_memory_mib"].get(pid),
                )
                for row in gpu_rows
            ]
            rx = _sampled_integral(
                gpu_rows, "pcie_rx_kib_per_second", 1024.0,
            )
            tx = _sampled_integral(
                gpu_rows, "pcie_tx_kib_per_second", 1024.0,
            )
            energy = _sampled_integral(gpu_rows, "power_draw_mw", 0.001)
            per_gpu[str(gpu)] = {
                "actor_rank": binding["actor_rank"],
                "worker_pid": pid,
                "samples": len(gpu_rows),
                "resident_samples": len(gpu_rows),
                "positive_gpu_samples": sum(
                    row["gpu_utilization_percent"] > 0 for row in gpu_rows
                ),
                "gpu_utilization_percent_mean": _mean([
                    row["gpu_utilization_percent"] for row in gpu_rows
                ]),
                "gpu_utilization_percent_peak": max(
                    row["gpu_utilization_percent"] for row in gpu_rows
                ),
                "memory_utilization_percent_mean": _mean([
                    row["memory_utilization_percent"] for row in gpu_rows
                ]),
                "memory_utilization_percent_peak": max(
                    row["memory_utilization_percent"] for row in gpu_rows
                ),
                "memory_used_mib_min": min(
                    row["memory_used_mib"] for row in gpu_rows
                ),
                "memory_used_mib_mean": _mean([
                    row["memory_used_mib"] for row in gpu_rows
                ]),
                "memory_used_mib_peak": max(
                    row["memory_used_mib"] for row in gpu_rows
                ),
                "memory_total_mib": min(
                    row["memory_total_mib"] for row in gpu_rows
                ),
                "minimum_headroom_mib": min(
                    row["memory_total_mib"] - row["memory_used_mib"]
                    for row in gpu_rows
                ),
                "expected_process_memory_mib_min": min(process_memory),
                "expected_process_memory_mib_mean": _mean(process_memory),
                "expected_process_memory_mib_peak": max(process_memory),
                "pcie_rx_bytes": rx,
                "pcie_tx_bytes": tx,
                "energy_joules": energy,
                "power_draw_mw_mean": _mean([
                    row["power_draw_mw"] for row in gpu_rows
                ]),
                "power_draw_mw_peak": max(
                    row["power_draw_mw"] for row in gpu_rows
                ),
            }
        result[phase] = {
            "phase_epoch": selected[0]["phase_epoch"],
            "complete_four_gpu_batches": len(timestamps),
            "first_sample_monotonic_ns": timestamps[0],
            "last_sample_monotonic_ns": timestamps[-1],
            "observed_sample_span_seconds": (
                timestamps[-1] - timestamps[0]
            ) / 1e9,
            "phase_work_upper_bound_seconds": (
                (next_first - timestamps[0]) / 1e9
                if next_first is not None else None
            ),
            "duration_semantics": (
                "first acknowledged sample to first acknowledged sample of next phase; "
                "an upper bound on controller work plus next sampling-barrier latency"
                if next_first is not None
                else "last phase has only a sampled-span lower observation"
            ),
            "per_gpu": per_gpu,
            "aggregate": {
                "gpu_utilization_percent_mean_all_samples": _mean([
                    row["gpu_utilization_percent"] for row in selected
                ]),
                "gpu_utilization_percent_peak": max(
                    row["gpu_utilization_percent"] for row in selected
                ),
                "memory_utilization_percent_mean_all_samples": _mean([
                    row["memory_utilization_percent"] for row in selected
                ]),
                "memory_used_mib_peak_per_gpu": max(
                    row["memory_used_mib"] for row in selected
                ),
                "minimum_headroom_mib_per_gpu": min(
                    row["memory_total_mib"] - row["memory_used_mib"]
                    for row in selected
                ),
                "sampled_pcie_rx_bytes_sum_gpus": sum(
                    item["pcie_rx_bytes"]["sampled_left_rectangle_estimate"] or 0
                    for item in per_gpu.values()
                ),
                "sampled_pcie_tx_bytes_sum_gpus": sum(
                    item["pcie_tx_bytes"]["sampled_left_rectangle_estimate"] or 0
                    for item in per_gpu.values()
                ),
                "sampled_energy_joules_sum_gpus": sum(
                    item["energy_joules"]["sampled_left_rectangle_estimate"] or 0
                    for item in per_gpu.values()
                ),
            },
        }
    return result


def _actor_event_summary_v71(
    receipts: Sequence[dict], bindings: Sequence[dict], assignments: Sequence[dict],
    phase_statistics: Mapping[str, dict],
) -> tuple[dict, dict]:
    validated = telemetry.validate_actor_work_receipts_v66d(
        receipts, bindings, assignments, expected_request_outputs=64,
    )
    for receipt in validated:
        event = receipt["cuda_event"]
        event_ns = float(event["elapsed_ms"]) * 1_000_000.0
        wall_ns = event["worker_monotonic_elapsed_ns"]
        tolerance_ns = max(5_000_000.0, wall_ns * 0.02)
        if event_ns > wall_ns + tolerance_ns:
            raise RuntimeError("CUDA event elapsed time exceeds worker wall evidence")
    by_wave: dict[int, list[dict]] = defaultdict(list)
    by_gpu: dict[int, list[dict]] = defaultdict(list)
    binding_by_rank = {
        item["actor_rank"]: item
        for item in telemetry.canonical_actor_bindings_v66(bindings)
    }
    for receipt in validated:
        by_wave[receipt["wave_index"]].append(receipt)
        gpu = binding_by_rank[receipt["engine_rank"]]["physical_gpu_id"]
        by_gpu[gpu].append(receipt)
    wave_summary = {}
    for wave, items in sorted(by_wave.items()):
        elapsed = [float(item["cuda_event"]["elapsed_ms"]) for item in items]
        phase = f"mirrored_wave_{wave}_generation_all_actors"
        upper = phase_statistics[phase]["phase_work_upper_bound_seconds"]
        if upper is None or max(elapsed) / 1000.0 > upper + 0.005:
            raise RuntimeError("actor CUDA event does not fit generation phase window")
        wave_summary[str(wave)] = {
            "receipts": len(items),
            "cuda_event_elapsed_ms_min": min(elapsed),
            "cuda_event_elapsed_ms_mean": math.fsum(elapsed) / len(elapsed),
            "cuda_event_elapsed_ms_max_parallel_critical_path": max(elapsed),
            "cuda_event_elapsed_ms_sum_actor_work": math.fsum(elapsed),
            "generation_phase_work_upper_bound_seconds": upper,
            "request_outputs": sum(
                item["output_cardinality"]["request_outputs"] for item in items
            ),
            "generated_tokens": sum(
                item["output_cardinality"]["generated_tokens"] for item in items
            ),
            "prompt_tokens": sum(
                item["output_cardinality"]["prompt_tokens"] for item in items
            ),
        }
    per_gpu = {}
    for gpu, items in sorted(by_gpu.items()):
        elapsed = [float(item["cuda_event"]["elapsed_ms"]) for item in items]
        per_gpu[str(gpu)] = {
            "receipts": len(items),
            "cuda_event_elapsed_ms_sum": math.fsum(elapsed),
            "cuda_event_elapsed_ms_mean": math.fsum(elapsed) / len(elapsed),
            "generated_tokens": sum(
                item["output_cardinality"]["generated_tokens"] for item in items
            ),
        }
    return {
        "schema": "v66d-actor-event-coverage-v71",
        "passed": True,
        "receipt_count": len(validated),
        "waves": len(by_wave),
        "receipts_per_wave": 4,
        "receipts_per_gpu": len(by_wave),
        "cuda_event_parallel_critical_path_ms_sum_waves": math.fsum(
            item["cuda_event_elapsed_ms_max_parallel_critical_path"]
            for item in wave_summary.values()
        ),
        "output_cardinality_exact": True,
        "event_within_worker_wall_clock": True,
        "by_wave": wave_summary,
        "by_gpu": per_gpu,
    }, {item["work_id"]: item for item in validated}


def _transfer_lower_bounds_v71(metadata: Mapping[str, Any], wave_count: int) -> dict:
    metadata = _validate_adapter_metadata_v71(metadata)
    actors = 4
    canonical = metadata["canonical_bytes"]
    runtime = metadata["runtime_bytes"]
    base = metadata["base_layer_bytes"]
    operations = []

    def add(
        phase: str, operation: str, invocations: int, *,
        materializations: int = 0,
        canonical_h2d: int = 0,
        canonical_d2h: int = 0,
    ) -> None:
        h2d = canonical_h2d + materializations * runtime
        d2h = canonical_d2h + materializations * (2 * runtime + base)
        operations.append({
            "phase": phase,
            "operation": operation,
            "actor_invocations": invocations,
            "runtime_materialization_and_audit_calls_per_invocation": materializations,
            "h2d_bytes_per_invocation_lower_bound": h2d,
            "d2h_bytes_per_invocation_lower_bound": d2h,
            "h2d_bytes_total_lower_bound": invocations * h2d,
            "d2h_bytes_total_lower_bound": invocations * d2h,
        })

    install = "install_canonical_v434_master_all_actors"
    add(install, "canonical_install_materialize_and_exact_audit", actors, materializations=1)
    add(install, "post_install_state_certificate_materialize_and_exact_audit", actors, materializations=1)
    add(install, "reference_capture_materialize_and_exact_audit", actors, materializations=1)
    for wave in range(wave_count):
        add(
            f"mirrored_wave_{wave}_materialize_all_actors",
            "fp32_master_to_candidate_device_candidate_back_then_runtime_exact_audit",
            actors,
            materializations=1,
            canonical_h2d=canonical,
            canonical_d2h=canonical,
        )
        add(
            f"wave_{wave}_finalize_restore_all_actors",
            "immutable_master_runtime_restore_and_exact_audit",
            actors,
            materializations=1,
        )
    add(
        "pair_difference_update_execute_all_actors",
        "reduced_fp32_update_to_cpu_then_candidate_runtime_exact_audit",
        actors,
        materializations=1,
        canonical_d2h=canonical,
    )
    abort = "pair_difference_update_abort_all_actors"
    add(abort, "pending_update_rollback_runtime_exact_audit", actors, materializations=1)
    add(abort, "unconditional_master_restore_runtime_exact_audit", actors, materializations=1)
    add(abort, "final_state_certificate_runtime_exact_audit", actors, materializations=1)

    materializations = sum(
        item["actor_invocations"]
        * item["runtime_materialization_and_audit_calls_per_invocation"]
        for item in operations
    )
    candidate_invocations = actors * wave_count
    components = {
        "base_layer_exact_hash_d2h": materializations * base,
        "runtime_equal_and_sha256_readback_d2h": materializations * 2 * runtime,
        "candidate_fp32_return_d2h": candidate_invocations * canonical,
        "reduced_update_fp32_return_d2h": actors * canonical,
        "runtime_materialization_h2d": materializations * runtime,
        "candidate_fp32_master_h2d": candidate_invocations * canonical,
    }
    total_h2d = sum(item["h2d_bytes_total_lower_bound"] for item in operations)
    total_d2h = sum(item["d2h_bytes_total_lower_bound"] for item in operations)
    if (
        sum(value for key, value in components.items() if key.endswith("_h2d"))
        != total_h2d
        or sum(value for key, value in components.items() if key.endswith("_d2h"))
        != total_d2h
    ):
        raise RuntimeError("transfer lower-bound component ledger does not balance")
    return {
        "schema": "v66d-code-derived-transfer-byte-lower-bounds-v71",
        "claim": (
            "minimum logical bytes implied by the sealed successful call graph; "
            "allocator, protocol, duplicate, cache, and vLLM activation traffic excluded"
        ),
        "canonical_fp32_bytes": canonical,
        "runtime_bf16_bytes": runtime,
        "base_layer_exact_audit_bytes": base,
        "successful_runtime_materialization_and_audit_calls": materializations,
        "h2d_bytes_total_lower_bound": total_h2d,
        "d2h_bytes_total_lower_bound": total_d2h,
        "bidirectional_bytes_total_lower_bound": total_h2d + total_d2h,
        "component_lower_bounds": components,
        "operations": operations,
        "not_counted": [
            "vLLM add_lora activation/cache initialization traffic",
            "allocator and PCIe transaction overhead",
            "CPU-only canonical master clones and hashes",
            "NCCL peer or NVLink traffic",
            "model load, KV-cache allocation, generation, cleanup, and checkpoint traffic",
            "failure-only repair or rollback paths",
        ],
    }


def analyze_records_v71(
    rows: Sequence[dict],
    bindings: Sequence[dict],
    assignments: Sequence[dict],
    receipts: Sequence[dict],
    adapter_metadata: Mapping[str, Any],
    completion_proof: Mapping[str, Any],
) -> dict:
    required_completion = {
        "report_schema": "v66d-mirrored-crn-qwen36-calibration-report",
        "report_status": "complete_nonzero_train_only_no_commit_actor_attributed",
        "master_committed": False,
        "all_four_abort_receipts_exact": True,
        "final_exact_master_restored": True,
        "checkpoint_count": 0,
        "protected_dev_ood_or_holdout_opened": False,
    }
    if not isinstance(completion_proof, Mapping) or dict(completion_proof) != required_completion:
        raise RuntimeError("successful V66d completion proof is absent or changed")
    wave_indices = sorted({item["wave_index"] for item in assignments})
    if wave_indices != list(range(len(wave_indices))):
        raise RuntimeError("V66d assignment waves are not contiguous")
    wave_count = len(wave_indices)
    validated_rows, phases, grouped = _validate_rows_and_phases_v71(
        rows, bindings, wave_count,
    )
    phase_statistics = _phase_statistics_v71(
        bindings, phases, grouped,
    )
    actor_events, _receipt_by_work = _actor_event_summary_v71(
        receipts, bindings, assignments, phase_statistics,
    )
    mirrored = telemetry.summarize_mirrored_waves_v66d(
        validated_rows,
        bindings,
        assignments,
        receipts,
        expected_request_outputs=64,
    )
    transfers = _transfer_lower_bounds_v71(adapter_metadata, wave_count)
    overall_per_gpu = {}
    for gpu in telemetry.GPU_IDS_V66:
        gpu_rows = [row for row in validated_rows if row["gpu"] == gpu]
        binding = next(
            item for item in bindings if item["physical_gpu_id"] == gpu
        )
        pid = binding["worker_pid"]
        process_memory = [
            row["process_memory_mib"].get(
                str(pid), row["process_memory_mib"].get(pid),
            )
            for row in gpu_rows
        ]
        overall_per_gpu[str(gpu)] = {
            "actor_rank": binding["actor_rank"],
            "worker_pid": pid,
            "samples": len(gpu_rows),
            "peak_memory_used_mib": max(row["memory_used_mib"] for row in gpu_rows),
            "peak_expected_process_memory_mib": max(process_memory),
            "minimum_headroom_mib": min(
                row["memory_total_mib"] - row["memory_used_mib"] for row in gpu_rows
            ),
            "peak_gpu_utilization_percent": max(
                row["gpu_utilization_percent"] for row in gpu_rows
            ),
            "peak_memory_utilization_percent": max(
                row["memory_utilization_percent"] for row in gpu_rows
            ),
            "peak_power_draw_mw": max(row["power_draw_mw"] for row in gpu_rows),
        }
    phase_rx = sum(
        item["aggregate"]["sampled_pcie_rx_bytes_sum_gpus"]
        for item in phase_statistics.values()
    )
    phase_tx = sum(
        item["aggregate"]["sampled_pcie_tx_bytes_sum_gpus"]
        for item in phase_statistics.values()
    )
    return {
        "schema": SCHEMA_V71,
        "passed": True,
        "analysis_mode": "CPU-only durable artifact analysis",
        "telemetry_rows": len(validated_rows),
        "complete_four_gpu_batches": len(validated_rows) // 4,
        "phase_count": len(phases),
        "phase_order": phases,
        "actor_event_coverage": actor_events,
        "mirrored_work_attribution": mirrored,
        "phase_statistics": phase_statistics,
        "overall_per_gpu": overall_per_gpu,
        "sampled_pcie_integrals": {
            "claim": (
                "left-rectangle estimates over adjacent samples within each phase; "
                "not exact byte counts or lower bounds"
            ),
            "rx_bytes_sum_gpus": phase_rx,
            "tx_bytes_sum_gpus": phase_tx,
        },
        "transfer_byte_lower_bounds": transfers,
        "completion_proof": required_completion,
        "foreign_compute_process_observations": 0,
        "protected_dev_ood_or_holdout_opened": False,
    }


def analyze_finalized_run_v71(
    run_directory: Path = RUN_DIRECTORY_V66D,
    *,
    verify_static: bool = True,
) -> dict:
    run_directory = Path(run_directory).resolve()
    report_path = run_directory / "mirrored_calibration_report_v66d.json"
    population_path = run_directory / "mirrored_population_v66d.json"
    update_path = run_directory / "pair_difference_update_v66d.json"
    gpu_log_path = run_directory / "gpu_activity_v66d.jsonl"
    actor_log_path = run_directory / "actor_cuda_work_receipts_v66d.jsonl"
    if (run_directory / "failure_v66d.json").exists():
        raise RuntimeError("V66d failure artifact coexists with claimed success")
    static_contract = verify_static_contract_v71() if verify_static else None
    report = _load_self_hashed_json(
        report_path, "v66d-mirrored-crn-qwen36-calibration-report",
    )
    population = _load_self_hashed_json(
        population_path, "v66-mirrored-qwen36-population-evidence",
    )
    update = _load_self_hashed_json(
        update_path, "v66-nonzero-qwen36-pair-difference-update-receipt",
    )
    if (
        report.get("status")
        != "complete_nonzero_train_only_no_commit_actor_attributed"
        or report.get("protected_dev_ood_or_holdout_opened") is not False
        or report.get("checkpoint_snapshot_or_promotion_performed") is not False
        or report.get("compute_ledger", {}).get("checkpoint_count") != 0
        or report.get("nonzero_update", {}).get("master_committed") is not False
        or update.get("master_committed") is not False
        or update.get("all_four_abort_receipts_exact") is not True
        or update.get("candidate_differs_from_master") is not True
        or update.get("candidate_runtime_differs_from_master") is not True
        or report.get("preregistration_content_sha256")
        != EXPECTED_V66D_PREREG_CONTENT_SHA256
    ):
        raise RuntimeError("V66d successful completion semantics changed")
    expected_files = {
        population_path: report["population"]["file_sha256"],
        update_path: report["nonzero_update"]["file_sha256"],
        gpu_log_path: report["gpu_log_file_sha256"],
        actor_log_path: report["actor_cuda_work_log"]["file_sha256"],
    }
    for path, expected_hash in expected_files.items():
        if file_sha256_v71(path) != expected_hash:
            raise RuntimeError(f"V66d report-bound artifact changed: {path.name}")
    if (
        Path(report["population"]["path"]).resolve() != population_path
        or Path(report["nonzero_update"]["path"]).resolve() != update_path
        or Path(report["actor_cuda_work_log"]["path"]).resolve() != actor_log_path
    ):
        raise RuntimeError("V66d report artifact paths changed")
    assignments = [
        item for wave in population["plan"]["waves"] for item in wave
    ]
    bindings = report["gpu_waves"]["bindings"]
    rows = _read_jsonl(gpu_log_path)
    receipts = _read_jsonl(actor_log_path)
    completion = {
        "report_schema": report["schema"],
        "report_status": report["status"],
        "master_committed": report["nonzero_update"]["master_committed"],
        "all_four_abort_receipts_exact": update["all_four_abort_receipts_exact"],
        "final_exact_master_restored": (
            report["final_master_sha256"]
            == population["install_master_consensus_sha256"]
        ),
        "checkpoint_count": report["compute_ledger"]["checkpoint_count"],
        "protected_dev_ood_or_holdout_opened": report[
            "protected_dev_ood_or_holdout_opened"
        ],
    }
    analysis = analyze_records_v71(
        rows,
        bindings,
        assignments,
        receipts,
        load_adapter_metadata_v71(),
        completion,
    )
    expected_wave_summary = dict(report["gpu_waves"])
    expected_wave_summary.pop("actor_cuda_work_log", None)
    if analysis["mirrored_work_attribution"] != expected_wave_summary:
        raise RuntimeError("V66d report wave summary differs from raw artifacts")
    charged = report["compute_ledger"]["charged_gpu_seconds"]
    resident = report["compute_ledger"]["model_resident_seconds_per_gpu"]
    if not math.isclose(charged, 4.0 * resident, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError("V66d charged GPU-second accounting changed")
    analysis.update({
        "artifact_receipts": {
            path.name: {
                "file_sha256": file_sha256_v71(path),
                "bytes": path.stat().st_size,
            }
            for path in (
                report_path, population_path, update_path, gpu_log_path,
                actor_log_path,
            )
        },
        "report_content_sha256": report["content_sha256_before_self_field"],
        "population_content_sha256": population[
            "content_sha256_before_self_field"
        ],
        "update_content_sha256": update["content_sha256_before_self_field"],
        "charged_gpu_seconds": charged,
        "model_resident_seconds_per_gpu": resident,
        "checkpoint_phase_observed": False,
        "checkpoint_phase_reason": (
            "sealed calibration performs no commit, snapshot, promotion, or checkpoint"
        ),
        "static_contract": static_contract,
    })
    analysis["content_sha256_before_self_field"] = canonical_sha256_v71(analysis)
    return analysis


def build_preregistration_v71() -> dict:
    compact = {
        "schema": PREREGISTRATION_SCHEMA_V71,
        "status": "sealed_postrun_analysis_contract_before_v71_evidence_publication",
        "bead": "specialist-0j5.14",
        "input_run_schema": "v66d-mirrored-crn-qwen36-calibration-report",
        "input_run_status": "complete_nonzero_train_only_no_commit_actor_attributed",
        "static_file_sha256": EXPECTED_STATIC_SHA256,
        "adapter_metadata": {
            "stage_manifest_file_sha256": EXPECTED_STAGE_MANIFEST_SHA256,
            "stage_manifest_content_sha256": EXPECTED_STAGE_CONTENT_SHA256,
            "weights_file_sha256": EXPECTED_WEIGHTS_SHA256,
            "config_file_sha256": EXPECTED_CONFIG_SHA256,
            "runtime_topology_report_file_sha256": EXPECTED_TOPOLOGY_REPORT_SHA256,
            "runtime_topology_report_content_sha256": EXPECTED_TOPOLOGY_CONTENT_SHA256,
            "canonical_tensors": CANONICAL_TENSORS,
            "canonical_elements": CANONICAL_ELEMENTS,
            "runtime_views": RUNTIME_VIEWS,
            "runtime_elements": RUNTIME_ELEMENTS,
            "base_layer_elements": BASE_LAYER_ELEMENTS,
        },
        "analysis_contract": {
            "exact_contiguous_four_gpu_batches": True,
            "exact_phase_epoch_sequence": True,
            "actor_pid_gpu_binding_and_residency_every_sample": True,
            "self_hashed_cuda_receipt_every_signed_candidate": True,
            "cuda_event_must_fit_worker_and_phase_wall_evidence": True,
            "per_phase_per_gpu_memory_utilization_power_and_pcie": True,
            "sampled_pcie_integrals_are_estimates_not_lower_bounds": True,
            "code_and_tensor_metadata_transfer_bytes_are_lower_bounds_not_measurements": True,
            "protected_dev_ood_or_holdout_opened": False,
            "gpu_or_dataset_access_by_analyzer": False,
        },
        "claim_limits": [
            "NVML does not split model, KV cache, adapter, scratch, or allocator pools",
            "no checkpoint phase exists in the sealed no-commit calibration",
            "no host RSS, NUMA, CPU allocator, CUDA allocator, or NCCL byte receipt exists",
            "phase duration is bracketed by sampling handshakes, not exact controller events",
        ],
    }
    return {
        **compact,
        "content_sha256_before_self_field": canonical_sha256_v71(compact),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", type=Path, default=RUN_DIRECTORY_V66D)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-preregistration", action="store_true")
    args = parser.parse_args(argv)
    if args.check_preregistration:
        if _load_json(PREREGISTRATION_V71) != build_preregistration_v71():
            raise RuntimeError("V71 analysis preregistration changed")
        print(json.dumps({
            "preregistration": str(PREREGISTRATION_V71),
            "content_sha256": build_preregistration_v71()[
                "content_sha256_before_self_field"
            ],
            "passed": True,
        }, sort_keys=True))
        return 0
    analysis = analyze_finalized_run_v71(args.run_directory)
    payload = json.dumps(analysis, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
