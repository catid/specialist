#!/usr/bin/env python3
"""Build the provisional Qwen3.6 production-layout decision (CPU only).

The builder consumes only sealed JSON/log receipts.  It intentionally keeps a
safe BF16 layout and a conditional serialized-FP8 challenger separate.  No
boolean in this document promotes FP8, CUDA graphs, or an unfinished memory
optimization to the final benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import analyze_v66d_phase_memory_v71 as memory_v71


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/decisions/"
    "qwen36_production_layout_provisional_v75.json"
)
SCHEMA_V75 = "qwen36-memory-efficient-production-layout-decision-v75"
FP8_PROMOTION_SCHEMA_V75 = "qwen36-fp8-layout-promotion-evidence-v75"
RECIPE_CONTRACT = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "recipe_evaluation_compute_contract_v1.json"
)
V67_PRECISION_CONTRACT = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_quantized_base_ablation_v67.json"
)

EXPECTED_RECIPE_CONTRACT_FILE_SHA256 = (
    "04af81499067e2feb0186c0a61e4c1af10f838a8eb7deec6dd41cd192748cacf"
)
EXPECTED_RECIPE_CONTRACT_CONTENT_SHA256 = (
    "2442c0c2be3ac4c883612f400f8f213ce3bc82ef96e03fad1ef10ec3b7d11fad"
)
EXPECTED_V67_FILE_SHA256 = (
    "e1bd808ce102882c0a3c13d4eab337ebce5ed70011ec8eefdbe8a727bbdc2724"
)
EXPECTED_V67_CONTENT_SHA256 = (
    "cdef17b02c2f77516b17225562edbed2c8faab41cbdb359a6a58ee2fec8b5236"
)
EXPECTED_V73_IMPLEMENTATION_SHA256 = (
    "43661c32cd8d06deef6d8e2f0d83d889b00f554748b94c3345e2b2052cac66a9"
)
EXPECTED_V73_INVENTORY_SHA256 = (
    "ab948ff5be8a9fede83107bbf01b4cde51b5367980b3489141595af3592f1762"
)
EXPECTED_V74_IMPLEMENTATION_SHA256 = (
    "c4434b1c720c9c209b3638abc6202ec864e18d7f5233963f7360a88fedbd5a63"
)
EXPECTED_V74_INVENTORY_SHA256 = (
    "13dc3991ec440e273359455fdf970f0025f3048deeb74d73219d29dd845ee04c"
)
EXPECTED_ONE_SLOT_INVENTORY_SHA256 = (
    "dc7aecfc5fd5a84a673cc1d9593c0e733b1e30a4b3fd0f71dd0d55aa1c171b93"
)
EXPECTED_DUAL_SLOT_INVENTORY_SHA256 = (
    "aec63377f7fbf35ad18759ea70064cb02020bb6ba5547489184e8ae12ee8a14d"
)
EXPECTED_SEQUENCE_CAP_INVENTORY_SHA256 = (
    "42f3e5ca3342d70288c175e3d9ea3ad991b75ae1e9fd5787d8230de0086c920e"
)
COLLECTIVE_BENCHMARK_SHA256 = {
    "experiments/eggroll_es_hpo/runs/"
    "v68_collective_inprocess_paired_12x64/benchmark.json": (
        "eeef875c72dc0bd1a6135e8cd4425e9a7833f5c34a77d8a2714f0077c05af5ff"
    ),
    "experiments/eggroll_es_hpo/runs/"
    "v68_collective_inprocess_paired_12x64_r2/benchmark.json": (
        "e4259651d63a676f00ca17427c135ab6d7c209ab93793855df50322b18fbfa64"
    ),
}

V73_DIRECTORIES = (
    "experiments/eggroll_es_hpo/runs/v73_quantized_base_paired_wave1",
    "experiments/eggroll_es_hpo/runs/v73_quantized_base_paired_wave2",
)
V74_DIRECTORY = (
    "experiments/eggroll_es_hpo/runs/v74_fp8_rightsized_050_r1"
)
ONE_SLOT_DIRECTORIES = (
    "experiments/eggroll_es_hpo/runs/v66_four_gpu_adapter_switch_memory_baseline",
    "experiments/eggroll_es_hpo/runs/v66_four_gpu_adapter_switch_memory_eager_r2",
    "experiments/eggroll_es_hpo/runs/v66_four_gpu_adapter_switch_memory_eager_r3",
    "experiments/eggroll_es_hpo/runs/v66_four_gpu_adapter_switch_memory_graph",
    "experiments/eggroll_es_hpo/runs/v66_four_gpu_adapter_switch_memory_graph_r2",
    "experiments/eggroll_es_hpo/runs/v66_four_gpu_adapter_switch_memory_graph_r3",
)
DUAL_SLOT_DIRECTORIES = (
    "experiments/eggroll_es_hpo/runs/v66_four_gpu_resident_adapter_switch_eager",
    "experiments/eggroll_es_hpo/runs/v66_four_gpu_resident_adapter_switch_graph",
    "experiments/eggroll_es_hpo/runs/v66_four_gpu_resident_adapter_switch_graph_r2",
)
SEQUENCE_CAP_DIRECTORIES = (
    "experiments/eggroll_es_hpo/runs/v67_four_gpu_static32_adapter_switch_graph",
    "experiments/eggroll_es_hpo/runs/v67_four_gpu_static32_adapter_switch_graph_r2",
    "experiments/eggroll_es_hpo/runs/v68_four_gpu_static48_adapter_switch_graph",
)

DEVICE_TOTAL_MIB = 97_887
V71_FULL_ES_PEAK_MIB = 84_138
V71_SETUP_PEAK_MIB = 83_212
V71_MINIMUM_HEADROOM_MIB = 13_749
V71_CANDIDATE_UPDATE_DELTA_MIB = 926
FP8_TARGET_GPU_MEMORY_UTILIZATION = 0.50
BF16_CURRENT_GPU_MEMORY_UTILIZATION = 0.82
MAX_NUM_SEQS = 68
MAX_MODEL_LEN = 2048
MINIMUM_KV_TOKENS = MAX_NUM_SEQS * MAX_MODEL_LEN


def canonical_sha256_v75(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")).hexdigest()


def file_sha256_v75(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"artifact is not a JSON object: {path}")
    return value


def _validate_self_hash(value: Mapping[str, Any], label: str) -> None:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} is not an object")
    compact = dict(value)
    claimed = compact.pop("content_sha256_before_self_field", None)
    if not isinstance(claimed, str) or claimed != canonical_sha256_v75(compact):
        raise RuntimeError(f"{label} self hash changed")


def _load_contract(
    path: Path, file_sha256: str, content_sha256: str, schema: str,
) -> dict:
    if file_sha256_v75(path) != file_sha256:
        raise RuntimeError(f"contract file identity changed: {path}")
    value = _load_json(path)
    _validate_self_hash(value, path.name)
    if (
        value.get("schema") != schema
        or value.get("content_sha256_before_self_field") != content_sha256
    ):
        raise RuntimeError(f"contract content changed: {path}")
    return value


def _inventory(
    directories: Sequence[str], *, include_v73_nvml: bool = False,
) -> list[dict]:
    result = []
    for relative in directories:
        directory = ROOT / relative
        receipts = sorted(directory.glob("gpu_*_receipt.json"))
        if len(receipts) != 4:
            raise RuntimeError(f"four receipt coverage changed: {relative}")
        for path in receipts:
            result.append({
                "path": str(path.relative_to(ROOT)),
                "file_sha256": file_sha256_v75(path),
            })
        if include_v73_nvml:
            path = directory / "nvidia_smi_samples.log"
            result.append({
                "path": str(path.relative_to(ROOT)),
                "file_sha256": file_sha256_v75(path),
            })
    return result


def _check_inventory(items: Sequence[dict], expected: str, label: str) -> str:
    observed = canonical_sha256_v75(list(items))
    if observed != expected:
        raise RuntimeError(f"{label} evidence inventory changed")
    return observed


def _receipt_runtimes(directories: Sequence[str]) -> list[dict]:
    result = []
    for relative in directories:
        for path in sorted((ROOT / relative).glob("gpu_*_receipt.json")):
            receipt = _load_json(path)
            _validate_self_hash(receipt, str(path.relative_to(ROOT)))
            runtime = receipt.get("runtime")
            elapsed = receipt.get(
                "wall_runtime_seconds_excluding_model_load_and_cleanup"
            )
            if (
                not isinstance(runtime, dict)
                or isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(float(elapsed))
                or float(elapsed) <= 0.0
                or receipt.get("engine_shutdown_completed") is not True
                or receipt.get("adapter_update_or_hpo_performed") is not False
            ):
                raise RuntimeError("adapter-switch receipt semantics changed")
            result.append({"receipt": receipt, "elapsed": float(elapsed)})
    return result


def validate_prior_layout_evidence_v75() -> dict:
    one_inventory = _inventory(ONE_SLOT_DIRECTORIES)
    dual_inventory = _inventory(DUAL_SLOT_DIRECTORIES)
    cap_inventory = _inventory(SEQUENCE_CAP_DIRECTORIES)
    _check_inventory(
        one_inventory, EXPECTED_ONE_SLOT_INVENTORY_SHA256, "one-slot",
    )
    _check_inventory(
        dual_inventory, EXPECTED_DUAL_SLOT_INVENTORY_SHA256, "dual-slot",
    )
    _check_inventory(
        cap_inventory, EXPECTED_SEQUENCE_CAP_INVENTORY_SHA256, "sequence-cap",
    )
    one = _receipt_runtimes(ONE_SLOT_DIRECTORIES)
    dual = _receipt_runtimes(DUAL_SLOT_DIRECTORIES)
    caps = _receipt_runtimes(SEQUENCE_CAP_DIRECTORIES)

    one_eager = [
        item["elapsed"] for item in one
        if item["receipt"]["runtime"]["cuda_graphs_enabled"] is False
    ]
    one_graph = [
        item["elapsed"] for item in one
        if item["receipt"]["runtime"]["cuda_graphs_enabled"] is True
    ]
    dual_eager = [
        item["elapsed"] for item in dual
        if item["receipt"]["runtime"]["cuda_graphs_enabled"] is False
    ]
    dual_graph = [
        item["elapsed"] for item in dual
        if item["receipt"]["runtime"]["cuda_graphs_enabled"] is True
    ]
    if (
        len(one_eager) != 12 or len(one_graph) != 12
        or len(dual_eager) != 4 or len(dual_graph) != 8
        or any(item["receipt"]["runtime"]["max_loras"] != 1 for item in one)
        or any(item["receipt"]["runtime"]["max_loras"] != 2 for item in dual)
    ):
        raise RuntimeError("one/dual-slot evidence coverage changed")
    by_cap: dict[int, list[float]] = {32: [], 48: []}
    for item in caps:
        cap = item["receipt"]["runtime"]["max_num_seqs"]
        if cap not in by_cap:
            raise RuntimeError("sequence-cap evidence arm changed")
        by_cap[cap].append(item["elapsed"])
    if len(by_cap[32]) != 8 or len(by_cap[48]) != 4:
        raise RuntimeError("sequence-cap evidence coverage changed")
    result = {
        "one_slot_inventory_sha256": EXPECTED_ONE_SLOT_INVENTORY_SHA256,
        "dual_slot_inventory_sha256": EXPECTED_DUAL_SLOT_INVENTORY_SHA256,
        "sequence_cap_inventory_sha256": EXPECTED_SEQUENCE_CAP_INVENTORY_SHA256,
        "one_slot_eager_median_seconds": statistics.median(one_eager),
        "dual_slot_eager_median_seconds": statistics.median(dual_eager),
        "dual_slot_eager_regression_fraction": (
            statistics.median(dual_eager) / statistics.median(one_eager) - 1.0
        ),
        "one_slot_graph_median_seconds": statistics.median(one_graph),
        "dual_slot_graph_median_seconds": statistics.median(dual_graph),
        "dual_slot_graph_regression_fraction": (
            statistics.median(dual_graph) / statistics.median(one_graph) - 1.0
        ),
        "cap68_graph_median_seconds": statistics.median(one_graph),
        "cap48_graph_median_seconds": statistics.median(by_cap[48]),
        "cap48_regression_fraction": (
            statistics.median(by_cap[48]) / statistics.median(one_graph) - 1.0
        ),
        "cap32_graph_median_seconds": statistics.median(by_cap[32]),
        "cap32_regression_fraction": (
            statistics.median(by_cap[32]) / statistics.median(one_graph) - 1.0
        ),
    }
    expected = {
        "one_slot_eager_median_seconds": 46.089602033549454,
        "dual_slot_eager_median_seconds": 80.09551188797923,
        "one_slot_graph_median_seconds": 23.550330316007603,
        "dual_slot_graph_median_seconds": 31.965402808011277,
        "cap68_graph_median_seconds": 23.550330316007603,
        "cap48_graph_median_seconds": 28.988969927013386,
        "cap32_graph_median_seconds": 31.97206194201135,
    }
    if any(result[key] != value for key, value in expected.items()):
        raise RuntimeError("prior production-layout aggregate changed")
    return result


_MODEL_LOAD = re.compile(
    r"Model loading took ([0-9]+\.[0-9]+) GiB memory and ([0-9]+\.[0-9]+) seconds"
)
_KV_GIB = re.compile(r"Available KV cache memory: ([0-9]+\.[0-9]+) GiB")
_KV_TOKENS = re.compile(r"GPU KV cache size: ([0-9,]+) tokens")
_CONCURRENCY = re.compile(
    r"Maximum concurrency for 2,048 tokens per request: ([0-9]+\.[0-9]+)x"
)
_ACTOR = re.compile(r"v73-wave([12])-(bf16|fp8_serialized)-gpu-([0-3])")
_V74_ACTOR = re.compile(r"v74-fp8-050-r1-gpu-([0-3])")


def _one_match(pattern: re.Pattern[str], text: str, label: str) -> re.Match[str]:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(f"V73 log lacks one exact {label}")
    return matches[0]


def _parse_nvml_v73(path: Path) -> dict[int, dict]:
    batches: dict[str, list[tuple[int, int, int, int, float]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            raise RuntimeError("V73 NVML row shape changed")
        timestamp, gpu, _uuid, memory, util, memory_util, power = fields
        try:
            row = (int(gpu), int(memory), int(util), int(memory_util), float(power))
        except ValueError as error:
            raise RuntimeError("V73 NVML numeric field changed") from error
        if (
            row[0] not in (0, 1, 2, 3)
            or row[1] < 0 or not 0 <= row[2] <= 100
            or not 0 <= row[3] <= 100 or row[4] < 0.0
        ):
            raise RuntimeError("V73 NVML value changed")
        batches.setdefault(timestamp, []).append(row)
    if not batches or any(
        len(batch) != 4 or [row[0] for row in batch] != [0, 1, 2, 3]
        for batch in batches.values()
    ):
        raise RuntimeError("V73 NVML four-GPU batch coverage changed")
    result = {}
    for gpu in range(4):
        rows = [row for batch in batches.values() for row in batch if row[0] == gpu]
        resident = [row[1] for row in rows if row[1] > 1024]
        if (
            not resident or max(row[2] for row in rows) <= 0
            or rows[0][1] > 16 or rows[-1][1] > 16
            or rows[-1][2] != 0 or rows[-1][3] != 0
        ):
            raise RuntimeError("V73 useful-work or cleanup-idle evidence changed")
        result[gpu] = {
            "samples": len(rows),
            "steady_memory_mib": Counter(resident).most_common(1)[0][0],
            "peak_memory_mib": max(resident),
            "peak_gpu_utilization_percent": max(row[2] for row in rows),
            "peak_memory_utilization_percent": max(row[3] for row in rows),
            "final_memory_mib": rows[-1][1],
            "cleanup_idle": True,
        }
    return result


def validate_v73_evidence_v75() -> dict:
    implementation = ROOT / "probe_vllm_quantized_adapter_switch_v73.py"
    if file_sha256_v75(implementation) != EXPECTED_V73_IMPLEMENTATION_SHA256:
        raise RuntimeError("V73 implementation identity changed")
    inventory = _inventory(V73_DIRECTORIES, include_v73_nvml=True)
    _check_inventory(inventory, EXPECTED_V73_INVENTORY_SHA256, "V73")
    runtimes = {"bf16": [], "fp8_serialized": []}
    records = {"bf16": [], "fp8_serialized": []}
    all_arm_gpus = {"bf16": set(), "fp8_serialized": set()}
    wave_arm_counts = {1: Counter(), 2: Counter()}
    nvml_by_wave = {}
    for relative in V73_DIRECTORIES:
        wave = int(relative[-1])
        directory = ROOT / relative
        nvml = _parse_nvml_v73(directory / "nvidia_smi_samples.log")
        nvml_by_wave[wave] = nvml
        for path in sorted(directory.glob("gpu_*_receipt.json")):
            value = _load_json(path)
            _validate_self_hash(value, str(path.relative_to(ROOT)))
            actor = _ACTOR.fullmatch(str(value.get("actor_label")))
            arm = value.get("precision_arm")
            runtime = value.get("runtime")
            certificate = value.get("resolved_precision_certificate")
            gates = value.get("preflight_gates")
            if (
                value.get("schema")
                != "v73-qwen36-precision-lora-switch-preflight"
                or actor is None or int(actor.group(1)) != wave
                or actor.group(2) != arm
                or not isinstance(runtime, dict)
                or runtime.get("vllm_version") != "0.25.0"
                or runtime.get("max_loras") != 1
                or runtime.get("max_cpu_loras") != 2
                or runtime.get("max_num_seqs") != MAX_NUM_SEQS
                or runtime.get("enforce_eager") is not True
                or runtime.get("cuda_graphs_enabled") is not False
                or not isinstance(certificate, dict)
                or not isinstance(gates, dict)
                or gates.get("load_generate_switch_cleanup_passed") is not True
                or gates.get("candidate_repeat_exact_at_token_hash_level") is not True
                or gates.get("scored_evaluation_or_training_authorized") is not False
                or value.get("engine_shutdown_completed") is not True
                or value.get("source_dataset_rows_opened") != 0
                or value.get("protected_ood_shadow_or_terminal_opened") is not False
                or value.get("prompt_or_generation_text_persisted") is not False
                or value.get("token_ids_persisted") is not False
            ):
                raise RuntimeError("V73 receipt contract changed")
            gpu = int(actor.group(3))
            expected_quant = None if arm == "bf16" else "fp8"
            expected_block = None if arm == "bf16" else [128, 128]
            if (
                certificate.get("quant_config_name") != expected_quant
                or certificate.get("weight_block_size") != expected_block
                or certificate.get("resolved_from_live_engine") is not True
            ):
                raise RuntimeError("V73 resolved precision changed")
            elapsed = float(value["wall_runtime_seconds_excluding_model_load_and_cleanup"])
            if not math.isfinite(elapsed) or elapsed <= 0.0:
                raise RuntimeError("V73 runtime changed")
            text = path.with_name(path.name.replace("_receipt.json", ".log")).read_text(
                encoding="utf-8"
            )
            load = _one_match(_MODEL_LOAD, text, "model-load receipt")
            kv = _one_match(_KV_GIB, text, "KV-memory receipt")
            tokens = _one_match(_KV_TOKENS, text, "KV-token receipt")
            concurrency = _one_match(_CONCURRENCY, text, "concurrency receipt")
            row = {
                "wave": wave,
                "gpu": gpu,
                "runtime_seconds": elapsed,
                "model_load_gib": float(load.group(1)),
                "model_load_seconds": float(load.group(2)),
                "kv_cache_gib": float(kv.group(1)),
                "kv_cache_tokens": int(tokens.group(1).replace(",", "")),
                "maximum_2048_token_concurrency": float(concurrency.group(1)),
                "steady_memory_mib": nvml[gpu]["steady_memory_mib"],
                "peak_memory_mib": nvml[gpu]["peak_memory_mib"],
                "reference_repeat_changed_rows": value[
                    "reference_within_state_changed_rows"
                ],
                "candidate_repeat_changed_rows": value[
                    "candidate_within_state_changed_rows"
                ],
                "between_adapter_changed_rows": value[
                    "between_state_differing_rows"
                ],
                "receipt_content_sha256": value[
                    "content_sha256_before_self_field"
                ],
            }
            runtimes[arm].append(elapsed)
            records[arm].append(row)
            all_arm_gpus[arm].add(gpu)
            wave_arm_counts[wave][arm] += 1
    if (
        any(len(values) != 4 for values in records.values())
        or any(gpus != {0, 1, 2, 3} for gpus in all_arm_gpus.values())
        or any(counts != {"bf16": 2, "fp8_serialized": 2}
               for counts in wave_arm_counts.values())
    ):
        raise RuntimeError("V73 paired/counterbalanced coverage changed")
    bf16_median = statistics.median(runtimes["bf16"])
    fp8_median = statistics.median(runtimes["fp8_serialized"])
    if (
        bf16_median != 45.19571384298615
        or fp8_median != 49.39261133299442
        or {item["model_load_gib"] for item in records["bf16"]} != {68.24}
        or {item["model_load_gib"] for item in records["fp8_serialized"]}
        != {36.93}
        or {item["kv_cache_gib"] for item in records["bf16"]} != {6.87}
        or {item["kv_cache_gib"] for item in records["fp8_serialized"]}
        != {38.14}
        or {item["kv_cache_tokens"] for item in records["bf16"]}
        != {139_264}
        or {item["kv_cache_tokens"] for item in records["fp8_serialized"]}
        != {775_372}
        or {item["peak_memory_mib"] for item in records["bf16"]} != {83_820}
        or {item["peak_memory_mib"] for item in records["fp8_serialized"]}
        != {83_878}
    ):
        raise RuntimeError("V73 paired precision aggregate changed")
    return {
        "schema": "v73-paired-precision-evidence-summary-v75",
        "inventory_sha256": EXPECTED_V73_INVENTORY_SHA256,
        "implementation_file_sha256": EXPECTED_V73_IMPLEMENTATION_SHA256,
        "waves": 2,
        "actors_per_arm": 4,
        "every_arm_observed_on_every_physical_gpu": True,
        "all_four_gpus_useful_and_cleanup_idle_each_wave": True,
        "fixed_gpu_memory_utilization": 0.82,
        "bf16": {
            "median_runtime_seconds": bf16_median,
            "runtime_range_seconds": [min(runtimes["bf16"]), max(runtimes["bf16"])],
            "model_load_gib": 68.24,
            "kv_cache_gib": 6.87,
            "kv_cache_tokens": 139_264,
            "peak_memory_mib": 83_820,
            "reference_repeat_changed_rows_range": [1, 2],
            "candidate_repeat_changed_rows": 0,
            "between_adapter_changed_rows_range": [6, 7],
        },
        "fp8_serialized": {
            "median_runtime_seconds": fp8_median,
            "runtime_range_seconds": [
                min(runtimes["fp8_serialized"]),
                max(runtimes["fp8_serialized"]),
            ],
            "runtime_ratio_to_bf16": fp8_median / bf16_median,
            "runtime_regression_fraction": fp8_median / bf16_median - 1.0,
            "model_load_gib": 36.93,
            "model_load_reduction_gib": round(68.24 - 36.93, 2),
            "kv_cache_gib": 38.14,
            "kv_cache_increase_gib": round(38.14 - 6.87, 2),
            "kv_cache_tokens": 775_372,
            "peak_memory_mib": 83_878,
            "reference_repeat_changed_rows_range": [4, 6],
            "candidate_repeat_changed_rows": 0,
            "between_adapter_changed_rows_range": [13, 15],
        },
        "fixed_0p82_budget_converts_weight_savings_to_kv_not_nvml_peak_savings": True,
        "scored_evaluation_or_training_authorized": False,
        "protected_data_opened": False,
    }


def validate_v74_evidence_v75(v73: Mapping[str, Any]) -> dict:
    """Validate the immutable right-sized FP8 capacity run.

    V74 is deliberately capacity evidence, not precision-promotion evidence.
    In particular, it has one diagnostic run, no scored examples, non-exact
    reference repeats, and unresolved kernel/method attestations.
    """
    implementation = ROOT / "probe_vllm_fp8_rightsized_v74.py"
    if file_sha256_v75(implementation) != EXPECTED_V74_IMPLEMENTATION_SHA256:
        raise RuntimeError("V74 implementation identity changed")
    directory = ROOT / V74_DIRECTORY
    expected_names = {
        "actor_pids.txt", "nvidia_smi_samples.log",
        *(f"gpu_{gpu}.log" for gpu in range(4)),
        *(f"gpu_{gpu}_receipt.json" for gpu in range(4)),
    }
    paths = sorted(path for path in directory.iterdir() if path.is_file())
    if {path.name for path in paths} != expected_names:
        raise RuntimeError("V74 immutable run inventory shape changed")
    inventory = [{
        "path": str(path.relative_to(ROOT)),
        "file_sha256": file_sha256_v75(path),
    } for path in paths]
    _check_inventory(inventory, EXPECTED_V74_INVENTORY_SHA256, "V74")

    nvml = _parse_nvml_v73(directory / "nvidia_smi_samples.log")
    if (
        {item["samples"] for item in nvml.values()} != {259}
        or {item["steady_memory_mib"] for item in nvml.values()} != {52_738}
        or {item["peak_memory_mib"] for item in nvml.values()} != {52_738}
        or {item["peak_gpu_utilization_percent"] for item in nvml.values()}
        != {100}
    ):
        raise RuntimeError("V74 NVML capacity aggregate changed")

    runtimes = []
    reference_changes = []
    between_changes = []
    gpus = set()
    log_values = []
    for path in sorted(directory.glob("gpu_*_receipt.json")):
        value = _load_json(path)
        _validate_self_hash(value, str(path.relative_to(ROOT)))
        actor = _V74_ACTOR.fullmatch(str(value.get("actor_label")))
        runtime = value.get("runtime")
        memory_certificate = value.get("resolved_memory_budget_certificate")
        precision_certificate = value.get("resolved_precision_certificate")
        gates = value.get("preflight_gates")
        authority = value.get("authority")
        if (
            value.get("schema")
            != "v74-qwen36-fp8-rightsized-vram-preflight"
            or actor is None
            or not isinstance(runtime, Mapping)
            or runtime.get("gpu_memory_utilization") != 0.50
            or runtime.get("vllm_version") != "0.25.0"
            or runtime.get("max_loras") != 1
            or runtime.get("max_cpu_loras") != 2
            or runtime.get("max_num_seqs") != MAX_NUM_SEQS
            or runtime.get("enforce_eager") is not True
            or runtime.get("cuda_graphs_enabled") is not False
            or runtime.get("resolved_quantization") != "fp8"
            or runtime.get("serialized_fp8_checkpoint") is not True
            or memory_certificate != {
                "gpu_memory_utilization": 0.50,
                "resolved_from_live_engine": True,
            }
            or not isinstance(precision_certificate, Mapping)
            or precision_certificate.get("quant_config_name") != "fp8"
            or precision_certificate.get("weight_block_size") != [128, 128]
            or precision_certificate.get("resolved_from_live_engine") is not True
            or gates != {
                "candidate_changes_output": True,
                "candidate_repeat_exact_at_token_hash_level": True,
                "load_generate_switch_cleanup_passed": True,
                "reference_restore_exact_at_token_hash_level": False,
                "routed_expert_method_count_pending_worker_attestation": True,
                "scored_evaluation_or_training_authorized": False,
            }
            or authority != {
                "data_free_capacity_and_switch_diagnostic_only": True,
                "scored_evaluation_training_checkpoint_or_promotion_allowed": False,
            }
            or value.get("candidate_within_state_changed_rows") != 0
            or value.get("engine_shutdown_completed") is not True
            or value.get("adapter_update_or_hpo_performed") is not False
            or value.get("source_dataset_rows_opened") != 0
            or value.get("protected_ood_shadow_or_terminal_opened") is not False
            or value.get("prompt_or_generation_text_persisted") is not False
            or value.get("token_ids_persisted") is not False
        ):
            raise RuntimeError("V74 receipt contract changed")
        gpu = int(actor.group(1))
        if path.name != f"gpu_{gpu}_receipt.json":
            raise RuntimeError("V74 actor/file GPU binding changed")
        elapsed = value.get("wall_runtime_seconds_excluding_model_load_and_cleanup")
        if (
            isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed)) or float(elapsed) <= 0.0
        ):
            raise RuntimeError("V74 runtime changed")
        log = path.with_name(f"gpu_{gpu}.log").read_text(encoding="utf-8")
        load = _one_match(_MODEL_LOAD, log, "V74 model-load receipt")
        kv = _one_match(_KV_GIB, log, "V74 KV-memory receipt")
        tokens = _one_match(_KV_TOKENS, log, "V74 KV-token receipt")
        concurrency = _one_match(_CONCURRENCY, log, "V74 concurrency receipt")
        required_unresolved_log_evidence = (
            "Auto-disabled DeepGemm",
            "DeepGEMM E8M0 enabled on current platform",
            "Using default MoE config. Performance might be sub-optimal!",
            "No FlashInfer autotune cache entries found.Falling back to default tactics.",
            "v025_rtx_pro_6000_bf16_tp1_exhaustive_v27c",
        )
        if (
            any(fragment not in log for fragment in required_unresolved_log_evidence)
            or log.count("No LoRA kernel configs found") < 6
        ):
            raise RuntimeError("V74 unresolved kernel/fallback evidence changed")
        runtimes.append(float(elapsed))
        reference_changes.append(value["reference_within_state_changed_rows"])
        between_changes.append(value["between_state_differing_rows"])
        gpus.add(gpu)
        log_values.append({
            "model_load_gib": float(load.group(1)),
            "kv_cache_gib": float(kv.group(1)),
            "kv_cache_tokens": int(tokens.group(1).replace(",", "")),
            "maximum_2048_token_concurrency": float(concurrency.group(1)),
        })
    median_runtime = statistics.median(runtimes)
    if (
        gpus != {0, 1, 2, 3}
        or len(runtimes) != 4
        or median_runtime != 48.95789764399524
        or set(reference_changes) != {4, 5}
        or set(between_changes) != {12, 13, 14, 15}
        or {item["model_load_gib"] for item in log_values} != {36.93}
        or {item["kv_cache_gib"] for item in log_values} != {7.73}
        or {item["kv_cache_tokens"] for item in log_values} != {157_286}
        or {item["maximum_2048_token_concurrency"] for item in log_values}
        != {76.80}
    ):
        raise RuntimeError("V74 right-sized aggregate changed")
    peak = 52_738
    return {
        "schema": "v74-fp8-rightsized-capacity-summary-v75",
        "implementation_file_sha256": EXPECTED_V74_IMPLEMENTATION_SHA256,
        "immutable_run_inventory_sha256": EXPECTED_V74_INVENTORY_SHA256,
        "receipts": 4,
        "physical_gpus": [0, 1, 2, 3],
        "all_four_gpus_useful_and_cleanup_idle": True,
        "gpu_memory_utilization": 0.50,
        "median_runtime_seconds": median_runtime,
        "runtime_range_seconds": [min(runtimes), max(runtimes)],
        "runtime_ratio_to_v73_bf16_median": (
            median_runtime / v73["bf16"]["median_runtime_seconds"]
        ),
        "runtime_ratio_to_v73_fp8_median": (
            median_runtime / v73["fp8_serialized"]["median_runtime_seconds"]
        ),
        "model_load_gib": 36.93,
        "kv_cache_gib": 7.73,
        "kv_cache_tokens": 157_286,
        "maximum_2048_token_concurrency": 76.80,
        "steady_and_peak_memory_mib": peak,
        "physical_headroom_mib": DEVICE_TOTAL_MIB - peak,
        "peak_saving_vs_v73_fp8_0p82_mib": (
            v73["fp8_serialized"]["peak_memory_mib"] - peak
        ),
        "capacity_gate_passed": True,
        "promotion_gate_passed": False,
        "promotion_blockers_observed": [
            "reference_repeat_not_exact_4_to_5_of_68_rows",
            "single_capacity_replicate_not_three_paired_replicates",
            "full_es_peak_and_exact_update_receipts_not_measured",
            "deepgemm_cutlass_path_attestation_contradictory",
            "default_moe_and_flashinfer_tactic_fallbacks",
            "inherited_bf16_moe_tuning_path",
            "routed_expert_method_count_pending",
            "source_disjoint_semantic_and_ood_gates_not_run",
        ],
        "scored_evaluation_or_training_authorized": False,
        "protected_data_opened": False,
    }


def validate_collective_evidence_v75() -> dict:
    run_medians = []
    artifacts = []
    for relative, expected_sha in COLLECTIVE_BENCHMARK_SHA256.items():
        path = ROOT / relative
        actual = file_sha256_v75(path)
        if actual != expected_sha:
            raise RuntimeError("V68 paired collective artifact changed")
        value = _load_json(path)
        results = value.get("results")
        if (
            value.get("schema") != "eggroll-es-in-process-paired-collective-layout-v68"
            or value.get("backend") != "nccl"
            or value.get("world_size") != 4
            or value.get("runtime_parameter_count") != 23
            or value.get("elements_per_rank") != 142_999_552
            or value.get("bytes_per_layout_per_rank") != 571_998_208
            or value.get("both_layouts_resident_bytes_per_rank") != 1_143_996_416
            or value.get("pair_count") != 12
            or value.get("data_or_model_opened") is not False
            or not isinstance(results, list) or len(results) != 4
            or {item.get("rank") for item in results} != {0, 1, 2, 3}
            or {item.get("gpu") for item in results} != {0, 1, 2, 3}
        ):
            raise RuntimeError("V68 paired collective contract changed")
        medians = [
            float(item["summary"]["native_over_flat_median_speed"])
            for item in results
        ]
        run_medians.append(statistics.median(medians))
        artifacts.append({"path": relative, "file_sha256": actual})
    if run_medians != [1.003526527791839, 0.9977076489242296]:
        raise RuntimeError("V68 paired collective aggregate changed")
    return {
        "schema": "v68-native-versus-flat-collective-summary-v75",
        "artifacts": artifacts,
        "paired_run_native_over_flat_median_speeds": run_medians,
        "flat_shadow_extra_bytes_per_rank": 571_998_208,
        "flat_shadow_extra_mib_per_rank": 545.5,
        "decision": "retain_native_parameter_boundaries_no_flat_shadow",
    }


def _safe_default_layout_v75(v67: Mapping[str, Any]) -> dict:
    arm = v67["arms"]["bf16"]
    return {
        "layout_id": "qwen36-bf16-eager-one-slot-cap68-native-v75",
        "precision_arm": "bf16",
        "model_path": arm["checkpoint"]["path"],
        "model_config_sha256": arm["checkpoint"]["config_sha256"],
        "model_index_sha256": arm["checkpoint"]["index_sha256"],
        "resolved_quantization": None,
        "serialized_quantization": False,
        "dtype": "bfloat16",
        "canonical_adapter_dtype": "float32",
        "runtime_adapter_dtype": "bfloat16",
        "vllm_version": "0.25.0",
        "execution": {
            "enforce_eager": True,
            "cuda_graphs_enabled": False,
            "compilation_promoted": False,
            "async_scheduling": False,
            "scheduling_policy": "fcfs",
        },
        "parallelism": {
            "physical_gpus": [0, 1, 2, 3],
            "engines": 4,
            "tensor_parallel_size": 1,
            "one_candidate_per_actor_per_mirrored_wave": True,
        },
        "memory": {
            "gpu_memory_utilization": BF16_CURRENT_GPU_MEMORY_UTILIZATION,
            "max_model_len": MAX_MODEL_LEN,
            "max_num_seqs": MAX_NUM_SEQS,
            "max_loras": 1,
            "max_cpu_loras": 2,
            "max_lora_rank": 32,
            "enable_prefix_caching": False,
            "kv_cache_dtype": "auto",
        },
        "collective_layout": {
            "backend": "nccl",
            "parameter_boundaries": "native_23_tensor",
            "flat_shadow_buffer": False,
            "micro_bucket_elements": None,
        },
        "state": {
            "mode": "external_lora_canonical_fp32",
            "dense_full_weight_master_on_lora_path": False,
            "single_gpu_lora_slot": True,
            "candidate_and_restore_semantics": "v66d_exact_master_or_poison",
            "audit_policy": "current_full_exact_until_v71_audit_ablation_passes",
        },
    }


def _fp8_challenger_layout_v75(v67: Mapping[str, Any]) -> dict:
    arm = v67["arms"]["fp8_serialized"]
    result = _safe_default_layout_v75(v67)
    result.update({
        "layout_id": "qwen36-fp8-serialized-eager-one-slot-cap68-native-rightsized-v75",
        "precision_arm": "fp8_serialized",
        "model_path": arm["checkpoint"]["path"],
        "model_config_sha256": arm["checkpoint"]["config_sha256"],
        "model_index_sha256": arm["checkpoint"]["index_sha256"],
        "resolved_quantization": "fp8",
        "serialized_quantization": True,
    })
    result["memory"] = dict(result["memory"])
    result["memory"]["gpu_memory_utilization"] = (
        FP8_TARGET_GPU_MEMORY_UTILIZATION
    )
    result["precision_requirements"] = {
        "weight_block_size": [128, 128],
        "starting_moe_tuning_table": "empty_default",
        "online_requantization_forbidden": True,
        "bf16_or_rejected_v29_moe_table_reuse_forbidden": True,
    }
    return result


def build_decision_v75() -> dict:
    raise RuntimeError(
        "V75 decision is historical and transitively bound to quarantined "
        "evaluation V1; create a V2 successor"
    )
    recipe = _load_contract(
        RECIPE_CONTRACT,
        EXPECTED_RECIPE_CONTRACT_FILE_SHA256,
        EXPECTED_RECIPE_CONTRACT_CONTENT_SHA256,
        "specialist-recipe-evaluation-compute-contract-v1",
    )
    v67 = _load_contract(
        V67_PRECISION_CONTRACT,
        EXPECTED_V67_FILE_SHA256,
        EXPECTED_V67_CONTENT_SHA256,
        "qwen36-quantized-base-ablation-preregistration-v67",
    )
    v71 = memory_v71.analyze_finalized_run_v71()
    if (
        v71["content_sha256_before_self_field"]
        != "a4503e18cb6185ee872ad571d24a51fd8a7ac5154e1fd69793fba629842166e6"
        or {item["peak_memory_used_mib"] for item in v71["overall_per_gpu"].values()}
        != {V71_FULL_ES_PEAK_MIB}
        or {item["minimum_headroom_mib"] for item in v71["overall_per_gpu"].values()}
        != {V71_MINIMUM_HEADROOM_MIB}
    ):
        raise RuntimeError("V71 production-capacity evidence changed")
    v73 = validate_v73_evidence_v75()
    v74 = validate_v74_evidence_v75(v73)
    prior = validate_prior_layout_evidence_v75()
    collective = validate_collective_evidence_v75()
    ood = recipe["score_aggregation"]["ood_noninferiority"]
    compact = {
        "schema": SCHEMA_V75,
        "status": "provisional_not_final_benchmark_authority",
        "bead": "specialist-0j5.20",
        "decision_class": "production_layout_framework",
        "authority": {
            "cpu_planning_and_confirmation_runs_allowed": True,
            "final_recipe_benchmark_consumption_allowed": False,
            "precision_promotion_allowed_by_this_document": False,
            "protected_terminal_access_allowed": False,
            "dataset_or_protected_content_opened_by_builder": False,
        },
        "evidence": {
            "recipe_evaluation_contract": {
                "file_sha256": EXPECTED_RECIPE_CONTRACT_FILE_SHA256,
                "content_sha256": EXPECTED_RECIPE_CONTRACT_CONTENT_SHA256,
            },
            "precision_preregistration_v67": {
                "file_sha256": EXPECTED_V67_FILE_SHA256,
                "content_sha256": EXPECTED_V67_CONTENT_SHA256,
            },
            "phase_memory_profile_v71": {
                "analysis_content_sha256": v71[
                    "content_sha256_before_self_field"
                ],
                "v66d_report_file_sha256": v71["artifact_receipts"][
                    "mirrored_calibration_report_v66d.json"
                ]["file_sha256"],
                "charged_gpu_seconds": v71["charged_gpu_seconds"],
                "all_four_gpus_useful": True,
                "cleanup_idle": True,
            },
            "paired_precision_v73": v73,
            "right_sized_fp8_capacity_v74": v74,
            "one_dual_slot_and_sequence_caps": prior,
            "native_flat_collectives": collective,
        },
        "capacity_reservation": {
            "device_total_mib": DEVICE_TOTAL_MIB,
            "full_es_observed_peak_mib": V71_FULL_ES_PEAK_MIB,
            "minimum_reserved_full_es_headroom_mib": V71_MINIMUM_HEADROOM_MIB,
            "candidate_update_delta_mib": V71_CANDIDATE_UPDATE_DELTA_MIB,
            "minimum_pre_es_generation_headroom_mib": (
                V71_MINIMUM_HEADROOM_MIB + V71_CANDIDATE_UPDATE_DELTA_MIB
            ),
            "maximum_pre_es_generation_peak_mib": V71_SETUP_PEAK_MIB,
            "maximum_full_es_peak_mib": V71_FULL_ES_PEAK_MIB,
            "fp8_rightsized_gpu_memory_utilization": (
                FP8_TARGET_GPU_MEMORY_UTILIZATION
            ),
            "fp8_nominal_engine_budget_mib_at_0p50": (
                DEVICE_TOTAL_MIB * FP8_TARGET_GPU_MEMORY_UTILIZATION
            ),
            "fp8_rightsized_observed_peak_mib": v74[
                "steady_and_peak_memory_mib"
            ],
            "fp8_rightsized_observed_headroom_mib": v74[
                "physical_headroom_mib"
            ],
            "minimum_kv_cache_tokens": MINIMUM_KV_TOKENS,
            "flat_shadow_buffer_reserved_mib": 0.0,
            "host_memory_reservation_status": "blocked_pending_specialist-0j5.19",
        },
        "safe_default": {
            "status": "selected_for_confirmation_not_final_benchmark",
            "layout": _safe_default_layout_v75(v67),
            "reason_codes": [
                "v66d_full_es_exact_restore_and_four_gpu_activity_passed",
                "v71_minimum_13749_mib_headroom_observed",
                "v73_bf16_is_9p286_percent_faster_at_matched_0p82_budget",
                "v74_fp8_0p50_capacity_passed_but_precision_promotion_failed",
                "precision_semantics_remain_unscored",
            ],
        },
        "conditional_fp8_challenger": {
            "status": "blocked_not_promoted",
            "layout": _fp8_challenger_layout_v75(v67),
            "capacity_probe_v74": {
                "status": "passed_capacity_only_not_precision_promotion",
                "schema": "v74-qwen36-fp8-rightsized-vram-preflight",
                "single_variable_change": {
                    "gpu_memory_utilization": [0.82, 0.50]
                },
                "receipt_count_required": 4,
                "physical_gpus_required": [0, 1, 2, 3],
                "current_receipts_consumed": v74["receipts"],
                "immutable_run_inventory_sha256": v74[
                    "immutable_run_inventory_sha256"
                ],
                "observed_peak_mib": v74["steady_and_peak_memory_mib"],
                "observed_headroom_mib": v74["physical_headroom_mib"],
                "observed_kv_cache_tokens": v74["kv_cache_tokens"],
            },
            "all_promotion_gates_required": {
                "v74_load_generate_switch_cleanup_all_four": True,
                "resolved_serialized_fp8_block_128x128": True,
                "actual_generation_peak_mib_maximum": V71_SETUP_PEAK_MIB,
                "actual_full_es_peak_mib_maximum": V71_FULL_ES_PEAK_MIB,
                "kv_cache_tokens_minimum": MINIMUM_KV_TOKENS,
                "paired_replicates_minimum": 3,
                "median_runtime_ratio_to_bf16_maximum": 1.10,
                "source_disjoint_dev_semantic_reward_noninferiority": True,
                "ood_all_registered_noninferiority_conditions": ood,
                "all_four_gpu_activity_and_cleanup_idle": True,
                "reference_restore_exact_at_token_hash_level": True,
                "zero_unresolved_kernel_or_tactic_fallbacks": True,
                "routed_expert_method_count_attested": True,
                "exact_adapter_candidate_restore_update_receipts": True,
                "protected_terminal_opened": False,
            },
            "v73_observation": {
                "median_runtime_ratio_to_bf16": v73["fp8_serialized"][
                    "runtime_ratio_to_bf16"
                ],
                "model_load_reduction_gib": v73["fp8_serialized"][
                    "model_load_reduction_gib"
                ],
                "kv_cache_increase_gib_at_fixed_0p82": v73["fp8_serialized"][
                    "kv_cache_increase_gib"
                ],
                "nvml_peak_saving_mib_at_fixed_0p82": (
                    v73["bf16"]["peak_memory_mib"]
                    - v73["fp8_serialized"]["peak_memory_mib"]
                ),
                "semantic_equivalence_known": False,
            },
            "v74_observation": {
                "capacity_gate_passed": True,
                "promotion_gate_passed": False,
                "median_runtime_seconds": v74["median_runtime_seconds"],
                "runtime_ratio_to_v73_bf16_median": v74[
                    "runtime_ratio_to_v73_bf16_median"
                ],
                "peak_memory_mib": v74["steady_and_peak_memory_mib"],
                "physical_headroom_mib": v74["physical_headroom_mib"],
                "kv_cache_tokens": v74["kv_cache_tokens"],
                "promotion_blockers_observed": v74[
                    "promotion_blockers_observed"
                ],
            },
        },
        "retained_choices": [
            {
                "dimension": "lora_gpu_slots",
                "choice": 1,
                "alternative": 2,
                "evidence": {
                    "dual_eager_regression_fraction": prior[
                        "dual_slot_eager_regression_fraction"
                    ],
                    "dual_graph_regression_fraction": prior[
                        "dual_slot_graph_regression_fraction"
                    ],
                },
            },
            {
                "dimension": "max_num_seqs",
                "choice": 68,
                "alternatives": [48, 32],
                "evidence": {
                    "cap48_regression_fraction": prior["cap48_regression_fraction"],
                    "cap32_regression_fraction": prior["cap32_regression_fraction"],
                },
            },
            {
                "dimension": "collective_parameter_layout",
                "choice": "native_23_tensor",
                "alternative": "flat_shadow",
                "evidence": {
                    "paired_run_speed_ratios": collective[
                        "paired_run_native_over_flat_median_speeds"
                    ],
                    "flat_shadow_extra_mib_per_rank": 545.5,
                },
            },
            {
                "dimension": "full_weight_master_on_lora_path",
                "choice": False,
                "alternative": True,
                "reason": "external LoRA state mode forbids dense full-weight master",
            },
        ],
        "rejected_alternatives": [
            {
                "id": "dual_gpu_lora_slots_v66",
                "status": "rejected_current_vllm_hardware",
                "reopen_rule": "new compact implementation with bounded allocation and paired win",
            },
            {
                "id": "global_static_max_num_seqs_48_or_32",
                "status": "rejected_as_production_default",
                "reopen_rule": "real-length bucketing without a lower global cap",
            },
            {
                "id": "flat_collective_shadow_buffer",
                "status": "rejected_no_material_speed_win_and_545p5_mib_per_rank",
                "reopen_rule": "same-process paired material benefit exceeding memory cost",
            },
            {
                "id": "fp8_with_gpu_memory_utilization_0p82",
                "status": "rejected_as_production_capacity_layout",
                "reopen_rule": "none; use the right-sized 0p50 challenger",
            },
            {
                "id": "int4",
                "status": "blocked_unsupported_not_silently_fallback",
                "reopen_rule": "new serialized checkpoint and new preregistration",
            },
        ],
        "deferred_not_rejected": [
            {
                "id": "compiled_cuda_graph_execution",
                "reason": "throughput positive but token behavior changed; semantic/OOD and kernel gates pending",
                "beads": ["specialist-0j5.17", "specialist-0j5.22"],
            },
            {
                "id": "fused_candidate_noise_update_and_audit",
                "reason": "V71 identifies bandwidth target but live exactness/speed evidence pending",
                "beads": ["specialist-0j5.18", "specialist-0j5.21"],
            },
            {
                "id": "shared_or_streamed_host_master",
                "reason": "host allocation/fault evidence pending",
                "beads": ["specialist-0j5.19"],
            },
        ],
        "blocking_beads_before_finalization": [
            "specialist-0j5.15",
            "specialist-0j5.18",
            "specialist-0j5.19",
            "specialist-0j5.21",
            "specialist-0j5.22",
        ],
        "open_dependencies_with_provisional_current_choice": [
            "specialist-0j5.16", "specialist-0j5.17",
        ],
        "finalization": {
            "all_blocking_beads_must_close": True,
            "selected_layout_must_be_rebuilt_with_final_artifact_hashes": True,
            "three_seed_confirmation_required": True,
            "all_seed_ood_gates_required": True,
            "pooled_dev_primary_paired_95_lcb_minimum": 0.0,
            "four_gpu_activity_and_cleanup_required": True,
            "headroom_reservation_must_hold": True,
            "protected_terminal_not_used_for_layout_selection": True,
            "final_benchmark_must_bind_decision_content_sha256_unchanged": True,
        },
        "consumer_contract": {
            "request_schema": "qwen36-production-layout-consumer-request-v75",
            "must_include_decision_content_sha256": True,
            "must_equal_selected_layout_exactly": True,
            "conditional_fp8_requires_promotion_evidence_schema": (
                FP8_PROMOTION_SCHEMA_V75
            ),
            "final_benchmark_rejected_while_status_is_provisional": True,
        },
    }
    return {
        **compact,
        "content_sha256_before_self_field": canonical_sha256_v75(compact),
    }


def validate_decision_v75(value: Mapping[str, Any] | None = None) -> dict:
    raise RuntimeError(
        "V75 historical decision is nonpromotable after V1 quarantine"
    )
    expected = build_decision_v75()
    observed = _load_json(OUTPUT) if value is None else dict(value)
    if observed != expected:
        raise RuntimeError("V75 production-layout decision changed")
    return expected


def consumer_request_v75(decision: Mapping[str, Any], arm: str = "bf16") -> dict:
    if dict(decision) != build_decision_v75():
        raise RuntimeError("consumer used an unvalidated V75 decision")
    if arm == "bf16":
        layout = decision["safe_default"]["layout"]
    elif arm == "fp8_serialized":
        layout = decision["conditional_fp8_challenger"]["layout"]
    else:
        raise RuntimeError("consumer requested an unknown precision arm")
    return {
        "schema": "qwen36-production-layout-consumer-request-v75",
        "decision_content_sha256": decision[
            "content_sha256_before_self_field"
        ],
        "layout": layout,
    }


def validate_fp8_promotion_v75(
    promotion: Mapping[str, Any], decision: Mapping[str, Any],
) -> dict:
    raise RuntimeError(
        "V75 promotion is permanently disabled after evaluation V1 quarantine"
    )
    required = {
        "schema": FP8_PROMOTION_SCHEMA_V75,
        "decision_content_sha256": decision["content_sha256_before_self_field"],
        "v74_receipts": 4,
        "v74_immutable_run_inventory_sha256": EXPECTED_V74_INVENTORY_SHA256,
        "v74_capacity_gate_passed": True,
        "physical_gpus": [0, 1, 2, 3],
        "gpu_memory_utilization": 0.50,
        "resolved_quantization": "fp8",
        "weight_block_size": [128, 128],
        "generation_peak_mib": V71_SETUP_PEAK_MIB,
        "full_es_peak_mib": V71_FULL_ES_PEAK_MIB,
        "kv_cache_tokens": MINIMUM_KV_TOKENS,
        "paired_replicates": 3,
        "median_runtime_ratio_to_bf16": 1.10,
        "source_disjoint_dev_semantic_reward_noninferiority": True,
        "ood_all_registered_noninferiority_conditions": True,
        "all_four_gpu_activity_and_cleanup_idle": True,
        "reference_restore_exact_at_token_hash_level": True,
        "zero_unresolved_kernel_or_tactic_fallbacks": True,
        "routed_expert_method_count_attested": True,
        "exact_adapter_candidate_restore_update_receipts": True,
        "protected_terminal_opened": False,
    }
    if not isinstance(promotion, Mapping):
        raise RuntimeError("FP8 promotion evidence is absent")
    observed = dict(promotion)
    if set(observed) != set(required):
        raise RuntimeError("FP8 promotion evidence schema changed")
    exact_fields = {
        key: value for key, value in required.items()
        if key not in {
            "generation_peak_mib", "full_es_peak_mib", "kv_cache_tokens",
            "paired_replicates", "median_runtime_ratio_to_bf16",
        }
    }
    if any(observed.get(key) != value for key, value in exact_fields.items()):
        raise RuntimeError("FP8 promotion identity or boolean gate changed")
    numeric = (
        observed.get("generation_peak_mib"),
        observed.get("full_es_peak_mib"),
        observed.get("kv_cache_tokens"),
        observed.get("paired_replicates"),
        observed.get("median_runtime_ratio_to_bf16"),
    )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in numeric
    ):
        raise RuntimeError("FP8 promotion numeric gate changed")
    if (
        observed["generation_peak_mib"] <= 0
        or observed["full_es_peak_mib"] < observed["generation_peak_mib"]
        or observed["generation_peak_mib"] > V71_SETUP_PEAK_MIB
        or observed["full_es_peak_mib"] > V71_FULL_ES_PEAK_MIB
        or not isinstance(observed["kv_cache_tokens"], int)
        or observed["kv_cache_tokens"] < MINIMUM_KV_TOKENS
        or not isinstance(observed["paired_replicates"], int)
        or observed["paired_replicates"] < 3
        or observed["median_runtime_ratio_to_bf16"] <= 0.0
        or observed["median_runtime_ratio_to_bf16"] > 1.10
    ):
        raise RuntimeError("FP8 promotion threshold failed")
    return observed


def authorize_consumer_request_v75(
    request: Mapping[str, Any],
    *,
    purpose: str,
    promotion: Mapping[str, Any] | None = None,
) -> dict:
    decision = validate_decision_v75()
    if purpose not in {"confirmation", "final_benchmark"}:
        raise RuntimeError("V75 consumer purpose changed")
    if purpose == "final_benchmark":
        raise RuntimeError("V75 remains provisional and cannot authorize final benchmark")
    if not isinstance(request, Mapping) or set(request) != {
        "schema", "decision_content_sha256", "layout",
    }:
        raise RuntimeError("V75 consumer request schema changed")
    if (
        request["schema"] != decision["consumer_contract"]["request_schema"]
        or request["decision_content_sha256"]
        != decision["content_sha256_before_self_field"]
    ):
        raise RuntimeError("V75 consumer decision identity changed")
    safe = decision["safe_default"]["layout"]
    fp8 = decision["conditional_fp8_challenger"]["layout"]
    if request["layout"] == safe:
        if promotion is not None:
            raise RuntimeError("BF16 safe default must not consume FP8 promotion")
        arm = "bf16"
    elif request["layout"] == fp8:
        validate_fp8_promotion_v75(promotion, decision)
        arm = "fp8_serialized"
    else:
        raise RuntimeError("V75 consumer layout changed")
    return {
        "schema": "qwen36-production-layout-consumer-authorization-v75",
        "authorized": True,
        "purpose": "confirmation",
        "precision_arm": arm,
        "decision_content_sha256": decision["content_sha256_before_self_field"],
        "layout_unchanged": True,
        "final_benchmark_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    value = build_decision_v75()
    if args.check:
        validate_decision_v75()
        print(json.dumps({
            "passed": True,
            "path": str(OUTPUT),
            "content_sha256": value["content_sha256_before_self_field"],
            "status": value["status"],
        }, sort_keys=True))
    else:
        print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
