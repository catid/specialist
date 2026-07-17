#!/usr/bin/env python3
"""Seal the V66 Qwen3.6 resident-LoRA-slot decision (CPU only).

This builder reads immutable synthetic receipts, logs, and telemetry.  It does
not import torch/vLLM, initialize CUDA, open a dataset, or authorize a training
or evaluation run.  The decision is deliberately negative: the tested second
GPU LoRA slot is rejected, while vLLM's existing two-entry CPU LRU cache plus
one preallocated GPU slot is retained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
DECISION_OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/decisions/"
    "qwen36_resident_lora_slot_decision_v80.json"
)
PREREG_OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_resident_lora_slot_challenger_v80.json"
)

SCHEMA_V80 = "qwen36-resident-lora-slot-decision-v80"
PREREG_SCHEMA_V80 = "qwen36-resident-lora-slot-challenger-preregistration-v80"
ONE_SLOT_SCHEMA = "v63-synthetic-two-adapter-switch-feasibility-probe"
TWO_SLOT_SCHEMA = "v66-synthetic-two-resident-adapter-switch-probe"

CALL_PLAN = [
    "reference", "candidate", "candidate", "reference",
    "reference", "candidate", "candidate", "reference",
]
ADAPTER_IDENTITIES = {
    "candidate": {
        "config_sha256": (
            "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5"
        ),
        "weights_sha256": (
            "c2665b60928b16120a2b98fdf137fafd250644852c86a02d797689f02105c6c8"
        ),
    },
    "reference": {
        "config_sha256": (
            "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5"
        ),
        "weights_sha256": (
            "7a41d921c6988dc62dca092230ed5ccfd5d6568a600503c87ff086cb2763485a"
        ),
    },
}

PRIMARY_ARMS = {
    "max_loras_1_eager": (
        "experiments/eggroll_es_hpo/runs/"
        "v66_four_gpu_adapter_switch_memory_baseline",
        "experiments/eggroll_es_hpo/runs/"
        "v66_four_gpu_adapter_switch_memory_eager_r2",
        "experiments/eggroll_es_hpo/runs/"
        "v66_four_gpu_adapter_switch_memory_eager_r3",
    ),
    "max_loras_1_graph": (
        "experiments/eggroll_es_hpo/runs/"
        "v66_four_gpu_adapter_switch_memory_graph",
        "experiments/eggroll_es_hpo/runs/"
        "v66_four_gpu_adapter_switch_memory_graph_r2",
        "experiments/eggroll_es_hpo/runs/"
        "v66_four_gpu_adapter_switch_memory_graph_r3",
    ),
    "max_loras_2_eager": (
        "experiments/eggroll_es_hpo/runs/"
        "v66_four_gpu_resident_adapter_switch_eager",
    ),
    "max_loras_2_graph": (
        "experiments/eggroll_es_hpo/runs/"
        "v66_four_gpu_resident_adapter_switch_graph",
        "experiments/eggroll_es_hpo/runs/"
        "v66_four_gpu_resident_adapter_switch_graph_r2",
    ),
}

# These later one-slot graph replications are bound as stability support.  They
# are not silently added to the already reported primary runtime comparator.
SUPPLEMENTAL_ARMS = {
    "max_loras_1_graph_later_replications": tuple(
        "experiments/eggroll_es_hpo/runs/"
        f"v66_four_gpu_adapter_switch_memory_graph_r{index}"
        for index in range(4, 9)
    ),
}

EXPECTED_DIRECTORY_INVENTORIES = {
    "v66_four_gpu_adapter_switch_memory_baseline": (
        9, "7939ce31e6a152a2fc1cafd9dc140d010afe85569f3463923de7747198351ebc"
    ),
    "v66_four_gpu_adapter_switch_memory_eager_r2": (
        9, "17136d7163d6588964719b71d4fe7caa62413dbf236c6d6bfc7705805cf1441e"
    ),
    "v66_four_gpu_adapter_switch_memory_eager_r3": (
        9, "2317cfa42f5f03b8f3c76468d5b9c324c148dbb47228e993fe20227a60f2ca89"
    ),
    "v66_four_gpu_adapter_switch_memory_graph": (
        9, "6fb24dc550f068e6a9f7f795fc68114b5b6d882241830c6bba2363308e4d6b72"
    ),
    "v66_four_gpu_adapter_switch_memory_graph_r2": (
        9, "235b87aee6637e8501358f5777fd7fd6cb767244f0485a4e416e81d20b024cb3"
    ),
    "v66_four_gpu_adapter_switch_memory_graph_r3": (
        9, "342adb1021bf4798fe8785eda238e6ae257f617b77d912b97b07f66efcfb37bc"
    ),
    "v66_four_gpu_adapter_switch_memory_graph_r4": (
        10, "b7b94f6cc0650adeef11a2e8aeb93e56aac062ab97a5069afa94f8586964d11c"
    ),
    "v66_four_gpu_adapter_switch_memory_graph_r5": (
        10, "a50b39157e9ae3575abc977300ee16f73f47787a59a9321d810994e448dda32f"
    ),
    "v66_four_gpu_adapter_switch_memory_graph_r6": (
        10, "85fe81255bda0dfcb205de65bbc882a7eaad6e941d4a80706ee53c5c4a260801"
    ),
    "v66_four_gpu_adapter_switch_memory_graph_r7": (
        10, "9b95841c7aaaa247f08616f3bc7fafd449ed35e54c948f6dce284555064241a9"
    ),
    "v66_four_gpu_adapter_switch_memory_graph_r8": (
        10, "71cfe7176eba6c10f3710c84d405cee3ef140988e452592b6df835e15f1490ff"
    ),
    "v66_four_gpu_resident_adapter_switch_eager": (
        10, "f29bf39457b23fadd4d6ccefbef9714180b3bb8ed58f5ad1581b4fef895a864a"
    ),
    "v66_four_gpu_resident_adapter_switch_graph": (
        10, "d8c5be27c1b7cce440a46ddb6913aeb0556f9b0f7db8c098ff4eb9d1d04a813e"
    ),
    "v66_four_gpu_resident_adapter_switch_graph_r2": (
        10, "9c950e5264341475aae72cfceb07146a4e0e74bbee93e5e97c9326eafead52e1"
    ),
}

BOUND_IMPLEMENTATION_FILES = {
    "probe_vllm_two_adapter_switch_v63.py": (
        "115774a63f54480fa4796f24f5b47a82fda1c2a761db4cf3ae0b6b83e85165d6"
    ),
    "probe_vllm_resident_adapter_switch_v66.py": (
        "64548c62de27cbd967e8d29abdbacd58fd79f4c3fb37ff9a829a86c612c1840f"
    ),
    "experiments/eggroll_es_hpo/"
    "qwen36_adapter_switch_memory_v66_20260717.md": (
        "cd934c7b9e2adb4f95be4f4104aaf01f90f71909c9bdb16fd33efc5524b8330d"
    ),
}

VLLM_ROOT = Path(
    "es-at-scale/.venv/lib/python3.12/site-packages/vllm"
)
VLLM_SOURCE_ATTESTATIONS = {
    "lora/model_manager.py": {
        "sha256": (
            "13201a06e17cccffb30c90bf3d268dfbf901567623b03454003afb5a922ae45a"
        ),
        "required_fragments": [
            "self.capacity, self.deactivate_adapter",
            "self.lora_slots, self._deactivate_adapter",
            "return self.lora_config.max_cpu_loras",
            "return self.lora_config.max_loras",
            "module.set_lora(",
            "self._active_adapters.remove_oldest()",
            "from_layer(\n                    module,\n                    self.lora_slots,",
        ],
    },
    "lora/worker_manager.py": {
        "sha256": (
            "1f7bb394ca92e21a3f0e250943196dd679c145f16fb67cc64e00e011e8e1447a"
        ),
        "required_fragments": [
            "lora_request.lora_int_id not in self.list_adapters()",
            "or lora_request.load_inplace",
            "If the lora is already loaded, just touch it",
            "self._adapter_manager.activate_adapter(lora_request.lora_int_id)",
        ],
    },
    "lora/layers/base_linear.py": {
        "sha256": (
            "040d75b4a76cb97f28c507453ab8aea68252c35b8b64d9f83ce40d32e2a3a495"
        ),
        "required_fragments": [
            "self.base_layer = base_layer",
            "torch.zeros(\n                max_loras,",
            ".copy_(\n            lora_a, non_blocking=True",
            ".copy_(\n            lora_b, non_blocking=True",
        ],
    },
    "lora/layers/fused_moe.py": {
        "sha256": (
            "62dcf28f2af0906b420c75bba171dcf4e5a392e02dc9860c32fad88f46b95b7e"
        ),
        "required_fragments": [
            "self.base_layer = base_layer",
            "max_loras,\n                    self.local_num_experts,",
            "self.w13_lora_a_stacked",
            "self.w2_lora_b_stacked",
        ],
    },
    "config/lora.py": {
        "sha256": (
            "031bade79427ba8d1746be968cfa9c343e7c4dda1e7fdb5059a5775cb34874b5"
        ),
        "required_fragments": [
            "max_loras: int = Field(default=1, ge=1)",
            "Maximum number of LoRAs to store in CPU memory",
            "self.max_cpu_loras < self.max_loras",
        ],
    },
}


def canonical_sha256_v80(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def file_sha256_v80(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _validate_self_hash(value: Mapping[str, Any], label: str) -> None:
    compact = dict(value)
    claimed = compact.pop("content_sha256_before_self_field", None)
    if not isinstance(claimed, str) or canonical_sha256_v80(compact) != claimed:
        raise RuntimeError(f"self hash changed: {label}")


def directory_inventory_v80(relative: str) -> tuple[list[dict], str]:
    directory = ROOT / relative
    if not directory.is_dir():
        raise RuntimeError(f"evidence directory absent: {relative}")
    rows = []
    for path in sorted(item for item in directory.iterdir() if item.is_file()):
        rows.append({
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256_v80(path),
        })
    return rows, canonical_sha256_v80(rows)


def validate_bound_files_v80() -> list[dict]:
    result = []
    for relative, expected in BOUND_IMPLEMENTATION_FILES.items():
        path = ROOT / relative
        observed = file_sha256_v80(path)
        if observed != expected:
            raise RuntimeError(f"bound implementation changed: {relative}")
        result.append({"path": relative, "sha256": observed})
    return result


def audit_vllm_source_v80() -> dict:
    files = []
    for relative, contract in VLLM_SOURCE_ATTESTATIONS.items():
        path = ROOT / VLLM_ROOT / relative
        observed = file_sha256_v80(path)
        if observed != contract["sha256"]:
            raise RuntimeError(f"installed vLLM source changed: {relative}")
        source = path.read_text(encoding="utf-8")
        for fragment in contract["required_fragments"]:
            if fragment not in source:
                raise RuntimeError(
                    f"installed vLLM semantic fragment absent: {relative}"
                )
        files.append({
            "path": str(VLLM_ROOT / relative),
            "sha256": observed,
            "required_fragment_count": len(contract["required_fragments"]),
        })
    return {
        "vllm_version_from_live_receipts": "0.25.0",
        "files": files,
        "claims_supported_by_source": {
            "registered_adapter_capacity_is_max_cpu_loras": True,
            "active_gpu_slot_capacity_is_max_loras": True,
            "already_registered_adapter_skips_disk_load_and_is_touched": True,
            "one_active_slot_evicts_oldest_then_copies_registered_weights": True,
            "linear_and_fused_moe_lora_buffers_scale_with_max_loras": True,
            "lora_wrappers_retain_one_base_layer_reference": True,
            "max_loras_2_duplicates_the_gpu_base_model": False,
        },
        "causal_limit": (
            "Source proves slot-scaled LoRA buffers, including fused-MoE "
            "buffers, but the live evidence has no allocator-level breakdown "
            "that assigns every byte of the 3.54 GiB increase."
        ),
    }


def _validate_receipt_v80(path: Path, max_loras: int, graph: bool) -> dict:
    value = _load_json(path)
    _validate_self_hash(value, str(path.relative_to(ROOT)))
    expected_schema = ONE_SLOT_SCHEMA if max_loras == 1 else TWO_SLOT_SCHEMA
    runtime = value.get("runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError(f"runtime absent: {path}")
    exact_runtime = {
        "VLLM_BATCH_INVARIANT": False,
        "async_scheduling": False,
        "cuda_graphs_enabled": graph,
        "enforce_eager": not graph,
        "max_cpu_loras": 2,
        "max_loras": max_loras,
        "max_num_seqs": 68,
        "scheduling_policy": "fcfs",
        "vllm_version": "0.25.0",
    }
    if (
        value.get("schema") != expected_schema
        or any(runtime.get(key) != expected for key, expected in exact_runtime.items())
        or value.get("adapter_identities") != ADAPTER_IDENTITIES
        or value.get("call_plan") != CALL_PLAN
        or value.get("engine_shutdown_completed") is not True
        or value.get("source_dataset_rows_opened") != 0
        or value.get("prompt_or_generation_text_persisted") is not False
        or value.get("token_ids_persisted") is not False
        or value.get("adapter_update_or_hpo_performed") is not False
        or value.get("protected_ood_shadow_or_terminal_opened") is not False
    ):
        raise RuntimeError(f"receipt contract changed: {path}")
    if max_loras == 2 and (
        runtime.get("both_adapters_have_resident_gpu_slots") is not True
        or value.get("single_variable_change_from_v63") != {"max_loras": [1, 2]}
    ):
        raise RuntimeError(f"two-slot attestation changed: {path}")
    elapsed = value.get("wall_runtime_seconds_excluding_model_load_and_cleanup")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) <= 0
    ):
        raise RuntimeError(f"invalid elapsed time: {path}")
    calls = value.get("calls")
    if not isinstance(calls, list) or len(calls) != 8:
        raise RuntimeError(f"call coverage changed: {path}")
    total_tokens = 0
    for index, call in enumerate(calls):
        if (
            call.get("call_index") != index
            or call.get("label") != CALL_PLAN[index]
            or "elapsed_seconds" in call
            or not isinstance(call.get("rows"), list)
            or len(call["rows"]) != 68
        ):
            raise RuntimeError(f"call semantics changed: {path}")
        if canonical_sha256_v80(call["rows"]) != call.get("rows_sha256"):
            raise RuntimeError(f"call row hash changed: {path}")
        for row in call["rows"]:
            token_hash = row.get("token_ids_sha256")
            token_count = row.get("token_count")
            if (
                not isinstance(token_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", token_hash)
                or token_count != 64
            ):
                raise RuntimeError(f"token-hash receipt changed: {path}")
            total_tokens += token_count
    for field in (
        "reference_within_state_changed_rows",
        "candidate_within_state_changed_rows",
        "between_state_differing_rows",
    ):
        if not isinstance(value.get(field), int) or not 0 <= value[field] <= 68:
            raise RuntimeError(f"output metric invalid: {path}")
    value["_elapsed"] = float(elapsed)
    value["_recorded_call_output_tokens"] = total_tokens
    return value


def _one_regex(text: str, pattern: str, label: str, cast: Any) -> Any:
    matches = re.findall(pattern, text)
    if len(matches) != 1:
        raise RuntimeError(f"expected one {label}, found {len(matches)}")
    return cast(matches[0].replace(",", ""))


def _parse_log_v80(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    if (
        "'enable_prefix_caching': False" not in text
        or "enable_prefix_caching=False" not in text
        or "Checkpoint size: 66.97 GiB" not in text
    ):
        raise RuntimeError(f"cache/model log contract changed: {path}")
    return {
        "model_load_gib": _one_regex(
            text, r"Model loading took ([0-9.]+) GiB", "model load", float
        ),
        "available_kv_gib": _one_regex(
            text,
            r"Available KV cache memory: ([0-9.]+) GiB",
            "available KV",
            float,
        ),
        "gpu_kv_tokens": _one_regex(
            text, r"GPU KV cache size: ([0-9,]+) tokens", "KV tokens", int
        ),
        "maximum_2048_token_concurrency": _one_regex(
            text,
            r"Maximum concurrency for 2,048 tokens per request: ([0-9.]+)x",
            "maximum concurrency",
            float,
        ),
        "prefix_caching_enabled": False,
    }


def _mode(values: Sequence[int]) -> int:
    if not values:
        raise RuntimeError("empty telemetry series")
    counts = Counter(values)
    maximum = max(counts.values())
    return min(value for value, count in counts.items() if count == maximum)


def _parse_nvml_v80(path: Path) -> dict:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    rows = [next(csv.reader([line]), []) for line in lines if line]
    # The early monitor changed query shape during several runs. Nine-column
    # rows are timestamp,index,uuid,memory.used,...; compact seven-column rows
    # are either the same timestamped shape or index,uuid,memory.used,... .
    # Three-column uuid,pid,memory rows are process telemetry. Preserve both
    # schemas rather than pretending their values are interchangeable.
    series_by_schema: dict[str, dict[str, list[int]]] = {}
    for row in rows:
        if len(row) == 9:
            schema, identity, used = (
                "device_memory_used_mib", str(int(row[1].strip())), int(row[3].strip())
            )
        elif len(row) == 7 and "T" in row[0]:
            schema, identity, used = (
                "device_memory_used_mib", str(int(row[1].strip())), int(row[3].strip())
            )
        elif len(row) == 7 and row[1].strip().startswith("GPU-"):
            schema, identity, used = (
                "device_memory_used_mib", str(int(row[0].strip())), int(row[2].strip())
            )
        elif len(row) == 3 and row[0].strip().startswith("GPU-"):
            schema, identity, used = (
                "compute_process_memory_used_mib", row[0].strip(), int(row[2].strip())
            )
        else:
            continue
        series_by_schema.setdefault(schema, {}).setdefault(identity, []).append(used)
    if not series_by_schema:
        raise RuntimeError(f"unrecognized telemetry schema: {path}")
    schema_summaries = {}
    for schema, series in sorted(series_by_schema.items()):
        summaries = []
        for identity, samples in sorted(series.items()):
            resident = [sample for sample in samples if sample > 1024]
            if not resident:
                raise RuntimeError(f"no resident samples: {path}")
            summaries.append({
                "identity": identity,
                "resident_sample_count": len(resident),
                "resident_mode_mib": _mode(resident),
                "resident_peak_mib": max(resident),
            })
        if len(summaries) != 4:
            raise RuntimeError(f"four-GPU telemetry coverage changed: {path}")
        schema_summaries[schema] = summaries
    return {
        "schema_summaries": schema_summaries,
        "pcie_rx_tx_fields_present": False,
    }


def _counts(values: Sequence[int]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(values).items())}


def _aggregate_arm_v80(name: str, directories: Sequence[str]) -> dict:
    expected_max_loras = 2 if name.startswith("max_loras_2") else 1
    graph = "graph" in name
    receipts = []
    logs = []
    nvml = []
    inventories = []
    for relative in directories:
        directory = ROOT / relative
        inventory, inventory_hash = directory_inventory_v80(relative)
        expected_count, expected_hash = EXPECTED_DIRECTORY_INVENTORIES[directory.name]
        if len(inventory) != expected_count or inventory_hash != expected_hash:
            raise RuntimeError(f"immutable V66 inventory changed: {relative}")
        inventories.append({
            "directory": relative,
            "file_count": len(inventory),
            "inventory_sha256": inventory_hash,
        })
        receipt_paths = sorted(directory.glob("gpu_*_receipt.json"))
        log_paths = sorted(directory.glob("gpu_*.log"))
        if len(receipt_paths) != 4 or len(log_paths) != 4:
            raise RuntimeError(f"four actor files required: {relative}")
        receipts.extend(
            _validate_receipt_v80(path, expected_max_loras, graph)
            for path in receipt_paths
        )
        logs.extend(_parse_log_v80(path) for path in log_paths)
        nvml.append(_parse_nvml_v80(directory / "nvidia_smi_samples.log"))
    elapsed = [item["_elapsed"] for item in receipts]
    recorded_tokens = [item["_recorded_call_output_tokens"] for item in receipts]
    if set(recorded_tokens) != {34_816}:
        raise RuntimeError(f"recorded output token coverage changed: {name}")
    schemas = sorted({schema for item in nvml for schema in item["schema_summaries"]})
    nvml_modes = {
        schema: [
            gpu["resident_mode_mib"]
            for item in nvml if schema in item["schema_summaries"]
            for gpu in item["schema_summaries"][schema]
        ]
        for schema in schemas
    }
    nvml_peaks = {
        schema: [
            gpu["resident_peak_mib"]
            for item in nvml if schema in item["schema_summaries"]
            for gpu in item["schema_summaries"][schema]
        ]
        for schema in schemas
    }
    median_elapsed = statistics.median(elapsed)
    return {
        "directories": list(directories),
        "directory_inventories": inventories,
        "run_count": len(directories),
        "actor_count": len(receipts),
        "physical_gpu_coverage_per_run": [0, 1, 2, 3],
        "runtime": {
            "median_seconds": median_elapsed,
            "minimum_seconds": min(elapsed),
            "maximum_seconds": max(elapsed),
            "recorded_call_output_tokens_per_actor": 34_816,
            "recorded_call_tokens_per_timed_second_lower_bound": (
                34_816 / median_elapsed
            ),
            "nominal_four_actor_lower_bound_tokens_per_second": (
                4 * 34_816 / median_elapsed
            ),
            "timer_includes_two_warmup_generations": True,
            "timer_includes_eight_recorded_generations": True,
            "timer_includes_first_adapter_loads": True,
            "adjacent_adapter_state_transitions_inside_timer": 6,
            "isolated_switch_latency_measured": False,
        },
        "engine_memory": {
            "model_load_gib_values": sorted({item["model_load_gib"] for item in logs}),
            "median_model_load_gib": statistics.median(
                item["model_load_gib"] for item in logs
            ),
            "available_kv_gib_values": sorted(
                {item["available_kv_gib"] for item in logs}
            ),
            "median_available_kv_gib": statistics.median(
                item["available_kv_gib"] for item in logs
            ),
            "gpu_kv_token_values": sorted({item["gpu_kv_tokens"] for item in logs}),
            "median_gpu_kv_tokens": statistics.median(
                item["gpu_kv_tokens"] for item in logs
            ),
            "prefix_caching_enabled": False,
            "nvml": {
                "schemas": schemas,
                "median_resident_mode_mib_by_schema": {
                    schema: statistics.median(values)
                    for schema, values in nvml_modes.items()
                },
                "maximum_resident_peak_mib_by_schema": {
                    schema: max(values) for schema, values in nvml_peaks.items()
                },
                "pcie_rx_tx_fields_present": False,
                "cross_schema_comparison_authorized": False,
            },
        },
        "output_behavior": {
            "reference_repeat_changed_rows_distribution": _counts([
                item["reference_within_state_changed_rows"] for item in receipts
            ]),
            "candidate_repeat_changed_rows_distribution": _counts([
                item["candidate_within_state_changed_rows"] for item in receipts
            ]),
            "first_reference_candidate_differing_rows_distribution": _counts([
                item["between_state_differing_rows"] for item in receipts
            ]),
            "actors_with_any_reference_repeat_change": sum(
                item["reference_within_state_changed_rows"] > 0
                for item in receipts
            ),
            "actors_with_any_candidate_repeat_change": sum(
                item["candidate_within_state_changed_rows"] > 0
                for item in receipts
            ),
            "decoded_text_or_reward_persisted": False,
            "semantic_quality_equivalence_measured": False,
        },
        "safety": {
            "source_dataset_rows_opened": 0,
            "protected_holdout_or_ood_opened": False,
            "adapter_update_or_hpo_performed": False,
            "all_receipts_report_engine_shutdown": True,
        },
    }


def collect_evidence_v80() -> dict:
    primary = {
        name: _aggregate_arm_v80(name, directories)
        for name, directories in PRIMARY_ARMS.items()
    }
    supplemental = {
        name: _aggregate_arm_v80(name, directories)
        for name, directories in SUPPLEMENTAL_ARMS.items()
    }
    one_eager = primary["max_loras_1_eager"]
    one_graph = primary["max_loras_1_graph"]
    two_eager = primary["max_loras_2_eager"]
    two_graph = primary["max_loras_2_graph"]

    def comparison(one: dict, two: dict) -> dict:
        one_time = one["runtime"]["median_seconds"]
        two_time = two["runtime"]["median_seconds"]
        one_rate = one["runtime"]["recorded_call_tokens_per_timed_second_lower_bound"]
        two_rate = two["runtime"]["recorded_call_tokens_per_timed_second_lower_bound"]
        return {
            "median_workload_runtime_increase_percent": (two_time / one_time - 1) * 100,
            "recorded_call_token_rate_decrease_percent": (1 - two_rate / one_rate) * 100,
            "interpretation": (
                "Complete switched-workload comparison only; it is not an "
                "isolated adapter-switch latency estimate."
            ),
        }

    model_delta = round((
        two_eager["engine_memory"]["median_model_load_gib"]
        - one_eager["engine_memory"]["median_model_load_gib"]
    ), 2)
    one_eager_device = one_eager["engine_memory"]["nvml"][
        "median_resident_mode_mib_by_schema"
    ]["device_memory_used_mib"]
    two_eager_device = two_eager["engine_memory"]["nvml"][
        "median_resident_mode_mib_by_schema"
    ]["device_memory_used_mib"]
    one_graph_device = one_graph["engine_memory"]["nvml"][
        "median_resident_mode_mib_by_schema"
    ]["device_memory_used_mib"]
    two_graph_device = two_graph["engine_memory"]["nvml"][
        "median_resident_mode_mib_by_schema"
    ]["device_memory_used_mib"]
    return {
        "primary_arms": primary,
        "supplemental_arms": supplemental,
        "comparisons": {
            "eager_max_loras_2_vs_1": comparison(one_eager, two_eager),
            "graph_max_loras_2_vs_1": comparison(one_graph, two_graph),
            "memory_and_capacity": {
                "model_load_allocation_increase_gib": model_delta,
                "model_load_allocation_increase_mib_approximate": model_delta * 1024,
                "eager_device_resident_mode_mib": {
                    "max_loras_1": one_eager_device,
                    "max_loras_2": two_eager_device,
                    "max_loras_2_minus_1": two_eager_device - one_eager_device,
                },
                "graph_device_resident_mode_mib": {
                    "max_loras_1": one_graph_device,
                    "max_loras_2": two_graph_device,
                    "max_loras_2_minus_1": two_graph_device - one_graph_device,
                },
                "eager_available_kv_reduction_percent": (
                    1
                    - two_eager["engine_memory"]["median_available_kv_gib"]
                    / one_eager["engine_memory"]["median_available_kv_gib"]
                ) * 100,
                "eager_kv_token_reduction_percent": (
                    1
                    - two_eager["engine_memory"]["median_gpu_kv_tokens"]
                    / one_eager["engine_memory"]["median_gpu_kv_tokens"]
                ) * 100,
                "graph_available_kv_reduction_percent": (
                    1
                    - two_graph["engine_memory"]["median_available_kv_gib"]
                    / one_graph["engine_memory"]["median_available_kv_gib"]
                ) * 100,
                "graph_kv_token_reduction_percent": (
                    1
                    - two_graph["engine_memory"]["median_gpu_kv_tokens"]
                    / one_graph["engine_memory"]["median_gpu_kv_tokens"]
                ) * 100,
                "gpu_memory_utilization_in_every_arm": 0.82,
                "nvml_plateau_interpretation": (
                    "The engine budget is fixed at 0.82, so total NVML plateaus "
                    "are lower for max_loras=2 because the larger LoRA allocation "
                    "displaces KV; they are not the incremental slot cost. The "
                    "device-memory rows above are comparable. Legacy process-memory "
                    "rows are retained only as corroboration; cross-schema "
                    "subtraction is forbidden."
                ),
            },
        },
    }


def build_preregistration_v80() -> dict:
    value = {
        "schema": PREREG_SCHEMA_V80,
        "status": "sealed_before_any_future_compact_gpu_slot_challenger",
        "purpose": (
            "Fail-closed requirements for any future custom compact/in-place "
            "GPU-resident alternative to the retained one-GPU-slot CPU-LRU layout."
        ),
        "fixed_reference_layout": {
            "max_loras": 1,
            "max_cpu_loras": 2,
            "base_model_instances_per_actor": 1,
            "enable_prefix_caching": False,
            "gpu_memory_utilization": 0.82,
            "max_num_seqs": 68,
            "max_model_len": 2048,
            "dtype": "bfloat16",
            "vllm_version": "0.25.0",
        },
        "minimum_replication": {
            "physical_gpus_per_run": [0, 1, 2, 3],
            "sealed_runs_per_arm": 3,
            "counterbalanced_arm_order_required": True,
            "same_process_matched_arms_preferred": True,
        },
        "required_measurements": {
            "adapter_state": [
                "exact_direct_tensor_digest_after_every_install_and_restore",
                "exact_slot_identity_and_zeroing_for_unpopulated_slices",
                "no_base_weight_pointer_or_digest_change",
            ],
            "cache_isolation": [
                "prefix_cache_disabled_or_adapter_identity_namespaced",
                "fresh_request_kv_ownership_proved_across_candidate_switches",
                "no_stale_mapping_or_graph_specialization_crosses_adapter_identity",
            ],
            "timing": [
                "synchronized_isolated_cpu_to_gpu_activation_latency",
                "synchronized_restore_latency",
                "full_generation_workload_runtime_separate_from_switch_timers",
            ],
            "traffic": [
                "phase_aligned_pcie_rx_bytes_and_tx_bytes",
                "host_source_and_gpu_destination_buffer_bytes",
                "achieved_hbm_bytes_or_explicitly_unavailable",
            ],
            "memory": [
                "allocator_delta_for_base_lora_kv_graph_and_workspace",
                "peak_device_memory_and_minimum_physical_headroom",
                "kv_tokens_and_supported_full_length_concurrency",
            ],
            "behavior_and_quality": [
                "matched_token_hash_repeatability",
                "source_disjoint_dev_semantic_reward_noninferiority",
                "all_registered_ood_noninferiority_conditions",
            ],
        },
        "promotion_gates": {
            "no_duplicate_gpu_base_state": True,
            "exact_adapter_tensor_restore": True,
            "no_stale_kv_prefix_mapping_or_graph_state": True,
            "median_full_workload_runtime_no_slower_than_reference": True,
            "model_load_allocation_gib_not_above": 68.24,
            "eager_gpu_kv_tokens_not_below": 139_264,
            "semantic_and_ood_noninferiority": True,
            "cleanup_idle_and_all_four_gpu_attribution": True,
            "zero_missing_switch_or_pcie_measurements": True,
        },
        "failure_policy": (
            "Any missing, non-finite, unbound, post-selected, or cross-schema "
            "measurement rejects the challenger; it is never imputed from the "
            "V66 complete-workload timer."
        ),
        "authority": {
            "gpu_launch_authorized": False,
            "training_authorized": False,
            "dataset_access_authorized": False,
            "protected_evaluation_authorized": False,
            "custom_vllm_patch_authorized": False,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256_v80(value)
    return value


def build_decision_v80() -> dict:
    evidence = collect_evidence_v80()
    source_audit = audit_vllm_source_v80()
    prereg = build_preregistration_v80()
    value = {
        "schema": SCHEMA_V80,
        "status": "closed_negative_two_gpu_resident_slots_rejected",
        "decision_kind": "post_hoc_live_evidence_seal_not_a_new_gpu_experiment",
        "bead": "specialist-0j5.16",
        "bound_implementation_files": validate_bound_files_v80(),
        "future_challenger_preregistration": {
            "path": str(PREREG_OUTPUT.relative_to(ROOT)),
            "schema": PREREG_SCHEMA_V80,
            "content_sha256": prereg["content_sha256_before_self_field"],
        },
        "evidence": evidence,
        "installed_vllm_source_audit": source_audit,
        "decision": {
            "tested_max_loras_2_resident_gpu_slots": "rejected",
            "retained_supported_layout": {
                "max_loras": 1,
                "max_cpu_loras": 2,
                "base_model_instances_per_actor": 1,
                "registered_cpu_adapter_capacity": 2,
                "active_gpu_adapter_slot_capacity": 1,
                "after_first_load_disk_reload_for_two_alternating_adapters": False,
                "gpu_slot_activation_copies_registered_adapter_weights": True,
                "prefix_caching": False,
            },
            "base_state_conclusion": (
                "Neither tested layout duplicates the base model per adapter. "
                "max_loras=2 expands preallocated linear and fused-MoE LoRA "
                "buffers around one base_layer reference."
            ),
            "safe_existing_lazy_cpu_resident_alternative": (
                "vLLM max_loras=1,max_cpu_loras=2 keeps both adapter objects in "
                "the registered CPU LRU after first load, evicts the prior active "
                "GPU slot on a switch, and copies the selected registered weights "
                "into that one slot."
            ),
            "load_inplace_assessment": (
                "Not an optimization here: load_inplace explicitly re-enters "
                "_load_adapter and replaces the registered object."
            ),
            "custom_compact_gpu_buffer_assessment": (
                "No supported safe implementation was found in the installed "
                "vLLM path. It remains a new implementation subject to the "
                "forward challenger preregistration, not unfinished work in this bead."
            ),
            "rejection_reasons": [
                "3.54_gib_higher_model_lora_allocation",
                "51_to_55_percent_kv_capacity_displacement",
                "73.8_percent_eager_complete_workload_slowdown",
                "35.7_percent_graph_complete_workload_slowdown",
                "candidate_repeat_instability_observed_in_one_of_eight_graph_actors",
                "no_semantic_quality_equivalence_evidence",
            ],
        },
        "measurement_limits": {
            "isolated_adapter_switch_latency_seconds": None,
            "pcie_rx_bytes_per_switch": None,
            "pcie_tx_bytes_per_switch": None,
            "hbm_bytes_per_switch": None,
            "decoded_text_or_reward_quality_equivalence": None,
            "direct_gpu_lora_tensor_restore_digest": None,
            "why_not_inferred": (
                "The only timer spans two warmups, first adapter loads, eight "
                "recorded generations, and six state transitions. NVML files "
                "lack PCIe fields. Token hashes expose numerical behavior but "
                "not semantic quality or direct tensor identity."
            ),
        },
        "cache_and_restore_disposition": {
            "prefix_cache_disabled_in_all_56_primary_and_supplemental_actors": True,
            "prefix_cache_reuse_mechanism_enabled": False,
            "direct_stale_kv_or_mapping_instrumentation_present": False,
            "absence_of_all_stale_candidate_state_proved": False,
            "exact_token_repeat_restore_achieved_by_every_actor": False,
            "interpretation": (
                "Prefix-cache reuse was disabled, removing that cache path, but "
                "there was no direct KV/mapping instrumentation and nonzero repeat "
                "differences mean exact output restore was not established. This "
                "strengthens rejection; it does not promote graph execution or a "
                "custom buffer."
            ),
        },
        "bead_disposition": {
            "close_as_negative_result": True,
            "acceptance_criteria_claimed_complete": False,
            "closure_reason": (
                "The concrete resident-two-GPU-slot proposal is decisively "
                "rejected. The supported one-slot/two-CPU-cache layout already "
                "avoids base duplication and repeated disk loads. Unmeasured "
                "custom-buffer work is a separately preregistered future challenger."
            ),
        },
        "authority": {
            "gpu_launch_authorized": False,
            "training_authorized": False,
            "dataset_access_authorized": False,
            "protected_evaluation_authorized": False,
            "v66_artifact_mutation_authorized": False,
            "site_package_mutation_authorized": False,
            "custom_vllm_patch_authorized": False,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256_v80(value)
    return value


def validate_decision_v80(path: Path = DECISION_OUTPUT) -> dict:
    observed = _load_json(path)
    _validate_self_hash(observed, str(path.relative_to(ROOT)))
    expected = build_decision_v80()
    if observed != expected:
        raise RuntimeError("resident-slot decision differs from deterministic build")
    prereg = _load_json(PREREG_OUTPUT)
    _validate_self_hash(prereg, str(PREREG_OUTPUT.relative_to(ROOT)))
    if prereg != build_preregistration_v80():
        raise RuntimeError("resident-slot challenger preregistration changed")
    return observed


def retained_layout_request_v80(decision: Mapping[str, Any]) -> dict:
    return {
        "schema": "qwen36-resident-lora-slot-layout-request-v80",
        "decision_content_sha256": decision.get("content_sha256_before_self_field"),
        "max_loras": 1,
        "max_cpu_loras": 2,
        "base_model_instances_per_actor": 1,
        "prefix_caching": False,
        "custom_gpu_slot_buffer": False,
    }


def authorize_layout_request_v80(request: Mapping[str, Any]) -> dict:
    decision = validate_decision_v80()
    expected = retained_layout_request_v80(decision)
    if dict(request) != expected:
        raise RuntimeError("layout request changed or attempts a rejected slot layout")
    return {
        "layout_selection_authorized": True,
        "gpu_launch_authorized": False,
        "training_authorized": False,
        "decision_content_sha256": decision["content_sha256_before_self_field"],
        "max_loras": 1,
        "max_cpu_loras": 2,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        decision = validate_decision_v80()
        print(json.dumps({
            "status": "valid",
            "decision": str(DECISION_OUTPUT.relative_to(ROOT)),
            "content_sha256": decision["content_sha256_before_self_field"],
        }, sort_keys=True))
        return 0
    prereg = build_preregistration_v80()
    decision = build_decision_v80()
    _write_json(PREREG_OUTPUT, prereg)
    _write_json(DECISION_OUTPUT, decision)
    print(json.dumps({
        "decision": str(DECISION_OUTPUT.relative_to(ROOT)),
        "decision_content_sha256": decision["content_sha256_before_self_field"],
        "preregistration": str(PREREG_OUTPUT.relative_to(ROOT)),
        "preregistration_content_sha256": prereg[
            "content_sha256_before_self_field"
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
