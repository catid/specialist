#!/usr/bin/env python3
"""CPU-only parameter-matched Qwen3.6 LoRA-ES topology contract.

This module reads configuration, JSON commitments, and safetensors headers.
It never materializes a model tensor, accepts no dataset path, and invokes no
GPU API.  GPU evidence is accepted later only as aggregate/hash receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from safetensors import safe_open

import eggroll_es_moe_targeting_v69 as moe_v69
import eggroll_es_multiobjective_trust_region_v67 as trust_v67
import recipe_evaluation_contract_v1 as evaluation_v1


ROOT = Path(__file__).resolve().parent
MODEL = (ROOT / "models/Qwen3.6-35B-A3B").resolve()
MODEL_CONFIG = (MODEL / "config.json").resolve()
MODEL_INDEX = (MODEL / "model.safetensors.index.json").resolve()
REFERENCE_ADAPTER = moe_v69.REFERENCE_ADAPTER
REFERENCE_ADAPTER_CONFIG = moe_v69.REFERENCE_ADAPTER_CONFIG
REFERENCE_ADAPTER_WEIGHTS = moe_v69.REFERENCE_ADAPTER_WEIGHTS
PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_front_tail_lora_topology_v70.json"
).resolve()

EXPECTED_MODEL_CONFIG_SHA256 = moe_v69.EXPECTED_MODEL_CONFIG_SHA256
EXPECTED_MODEL_INDEX_SHA256 = moe_v69.EXPECTED_MODEL_INDEX_SHA256
EXPECTED_ADAPTER_CONFIG_SHA256 = moe_v69.EXPECTED_ADAPTER_CONFIG_SHA256
EXPECTED_ADAPTER_WEIGHTS_SHA256 = moe_v69.EXPECTED_ADAPTER_WEIGHTS_SHA256
EXPECTED_MOE_GEOMETRY_MODULE_SHA256 = (
    "a1f0e4ab2a6e65b05a5b20ec84db473d28c97cc497e42c1eed311b334355cc49"
)
LORA_RUNTIME_MAPPING = (
    ROOT / "eggroll_es_worker_lora_topology_v40a.py"
).resolve()
EXPECTED_LORA_RUNTIME_MAPPING_SHA256 = (
    "d487ff4657a2a6fbd7d2d0f9200a63547dd79b258cc488b93382fdf26b52cf05"
)
V40_RUNTIME_MAPPING_REPORT = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v40c_v37_lora_topology_probe_tuned_projection_retry/"
    "lora_topology_report_v40c.json"
).resolve()
EXPECTED_V40_REPORT_FILE_SHA256 = (
    "7672f835239b91e66a03a512ad9fbe3cbbaff31783a7b1a26bdd136d98b55050"
)
EXPECTED_V40_REPORT_CONTENT_SHA256 = (
    "9394ed06a80fffb1c4cc1532ac59741ab4c4a1c5a481136f29415b463eaf747d"
)

V23_MODEL_SEAL = (
    ROOT / "experiments/eggroll_es_hpo/S6_V23A_INSERTION_MODEL_SEAL.json"
).resolve()
V23_NEGATIVE_EVIDENCE = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V23A_INSERTION_NEGATIVE_AGGREGATE_EVIDENCE_R3.json"
).resolve()
EXPECTED_V23_MODEL_SEAL_FILE_SHA256 = (
    "96eeb236ea94678f57a530a27a471467d4b3d413d2e7be397e293b695cd4c440"
)
EXPECTED_V23_MODEL_SEAL_CONTENT_SHA256 = (
    "d4cf795408967aefbc77f841c47e6fe2fbe3cefc14a4a0fdb4bf73b2701326f9"
)
EXPECTED_V23_NEGATIVE_FILE_SHA256 = (
    "c1c3dfc8a191832914ba6475b86cedb390a6542e073ee4456ab4c354164aa25d"
)
EXPECTED_V23_NEGATIVE_CONTENT_SHA256 = (
    "51736876f895bb29c1f9f57845f4f2d4ef1f439b4b854bc9b17599aa35320417"
)

NUM_LAYERS = 40
MOTIF_WIDTH = 4
MOTIF_STARTS = tuple(range(0, NUM_LAYERS, MOTIF_WIDTH))
MOTIF_TYPES = (
    "linear_attention", "linear_attention", "linear_attention",
    "full_attention",
)
NUM_EXPERTS = 256
EXPERTS_PER_TOKEN = 8
TARGET_FAMILIES = (
    moe_v69.FAMILY_SHARED_SEQUENCE,
    moe_v69.FAMILY_SHARED_EXPERT,
    moe_v69.FAMILY_ROUTER,
)
RUNTIME_PACKED_MODULES = {
    "q_proj": ("qkv_proj", (0,)),
    "k_proj": ("qkv_proj", (1,)),
    "v_proj": ("qkv_proj", (2,)),
    "gate_proj": ("gate_up_proj", (0,)),
    "up_proj": ("gate_up_proj", (1,)),
    "in_proj_qkv": ("in_proj_qkvz", (0, 1, 2)),
    "in_proj_z": ("in_proj_qkvz", (3,)),
    "in_proj_b": ("in_proj_ba", (0,)),
    "in_proj_a": ("in_proj_ba", (1,)),
}

ARM_ORDER = (
    "early_contiguous_motif_r32",
    "late_contiguous_motif_r32",
    "symmetric_early_late_motifs_r16",
    "current_middle_late_motif_r32",
    "matched_distributed_motifs_r8",
)
CONTROL_ARM = "current_middle_late_motif_r32"
ARM_DEFINITIONS = {
    "early_contiguous_motif_r32": {
        "motif_starts": (0,), "rank": 32, "topology": "contiguous_early",
    },
    "late_contiguous_motif_r32": {
        "motif_starts": (36,), "rank": 32, "topology": "contiguous_late",
    },
    "symmetric_early_late_motifs_r16": {
        "motif_starts": (0, 36), "rank": 16,
        "topology": "symmetric_early_plus_late",
    },
    "current_middle_late_motif_r32": {
        "motif_starts": (20,), "rank": 32,
        "topology": "current_middle_late_control",
    },
    "matched_distributed_motifs_r8": {
        "motif_starts": (0, 12, 24, 36), "rank": 8,
        "topology": "distributed_aligned_motif_control",
    },
}
TRAINING_SEEDS = (1701, 1702, 1703)
SCHEDULE_BY_SEED = {
    1701: (
        CONTROL_ARM,
        "early_contiguous_motif_r32",
        "late_contiguous_motif_r32",
        "symmetric_early_late_motifs_r16",
        "matched_distributed_motifs_r8",
    ),
    1702: (
        "late_contiguous_motif_r32",
        "symmetric_early_late_motifs_r16",
        "matched_distributed_motifs_r8",
        CONTROL_ARM,
        "early_contiguous_motif_r32",
    ),
    1703: (
        "matched_distributed_motifs_r8",
        CONTROL_ARM,
        "early_contiguous_motif_r32",
        "late_contiguous_motif_r32",
        "symmetric_early_late_motifs_r16",
    ),
}

TRAINABLE_PARAMETER_BUDGET = 4_528_128
DIRECTION_COUNT = 16
SIGNED_POPULATION = 32
UPDATE_STEPS = 1
PERTURBATION_SCALAR_DRAWS = 72_450_048
OPTIMIZATION_ROLLOUTS_PER_RUN = 2_048
PRIMARY_EVALUATION_ROLLOUTS_PER_RUN = 83
LAYER_SENSITIVITY_ROLLOUTS_PER_RUN = 64
TOTAL_EVALUATION_ROLLOUTS_PER_RUN = 147
MIN_ROUTING_TOKENS_PER_LAYER = 4_096
MIN_ACTIVE_EXPERTS_PER_LAYER = 192
MIN_ROUTING_ENTROPY = 0.80
MAX_EXPERT_LOAD_CV = 2.0
MAX_TOPK_ROUTING_DISAGREEMENT = 0.20
MAX_ROUTING_JSD = 0.05
MIN_SAFE_HEADROOM_BYTES = 4 * 1024**3
MAX_PEAK_RESERVED_DELTA_VS_CONTROL_BYTES = 2 * 1024**3
MIN_THROUGHPUT_RATIO_VS_CONTROL = 0.95
MAX_MEMORY_TRAFFIC_RATIO_VS_CONTROL = 1.05

_HEX64 = re.compile(r"[0-9a-f]{64}")


def canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_keys(value: object, keys: Iterable[str], label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    expected = set(keys)
    if set(value) != expected:
        raise ValueError(
            f"{label} keys changed; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _guard(path: Path, expected_sha256: str, label: str) -> None:
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise RuntimeError(f"{label} bytes changed: {observed}")


def _without_self(value: dict) -> dict:
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _runtime_target(logical: str) -> tuple[str, tuple[int, ...]]:
    prefix, separator, leaf = logical.rpartition(".")
    if not separator:
        raise RuntimeError("LoRA logical module lost its namespace")
    replacement, slices = RUNTIME_PACKED_MODULES.get(leaf, (leaf, (0,)))
    return f"{prefix}.{replacement}", tuple(slices)


def _all_target_records(index: dict, layer_types: list[str]) -> list[dict]:
    by_shard = defaultdict(list)
    for layer in range(NUM_LAYERS):
        for family in TARGET_FAMILIES:
            for suffix in moe_v69._suffixes(layer_types[layer])[family]:
                key = f"model.language_model.layers.{layer}.{suffix}"
                shard = index["weight_map"].get(key)
                if shard is None:
                    raise RuntimeError(f"required topology target absent: {key}")
                by_shard[shard].append((key, layer, family, suffix))
    records = []
    for shard, items in sorted(by_shard.items()):
        with safe_open(MODEL / shard, framework="pt", device="cpu") as handle:
            for key, layer, family, suffix in items:
                view = handle.get_slice(key)
                shape = list(view.get_shape())
                if len(shape) != 2:
                    raise RuntimeError("active LoRA topology target is not 2D")
                logical = key.replace("model.language_model.", "model.")
                logical = logical[:-len(".weight")]
                runtime_target, slices = _runtime_target(logical)
                records.append({
                    "base_key": key,
                    "checkpoint_shard": shard,
                    "layer": layer,
                    "motif_position": layer % MOTIF_WIDTH,
                    "layer_type": layer_types[layer],
                    "family": family,
                    "suffix": suffix,
                    "shape": shape,
                    "dtype": view.get_dtype(),
                    "peft_logical_module": logical,
                    "runtime_target": runtime_target,
                    "runtime_slices": list(slices),
                })
    return sorted(records, key=lambda item: item["base_key"])


def build_architecture_manifest() -> dict:
    _guard(MODEL_CONFIG, EXPECTED_MODEL_CONFIG_SHA256, "model config")
    _guard(MODEL_INDEX, EXPECTED_MODEL_INDEX_SHA256, "model index")
    _guard(
        REFERENCE_ADAPTER_CONFIG, EXPECTED_ADAPTER_CONFIG_SHA256,
        "reference adapter config",
    )
    _guard(
        REFERENCE_ADAPTER_WEIGHTS, EXPECTED_ADAPTER_WEIGHTS_SHA256,
        "reference adapter weights",
    )
    _guard(
        Path(moe_v69.__file__).resolve(), EXPECTED_MOE_GEOMETRY_MODULE_SHA256,
        "MoE geometry module",
    )
    _guard(
        LORA_RUNTIME_MAPPING, EXPECTED_LORA_RUNTIME_MAPPING_SHA256,
        "LoRA runtime mapping",
    )
    _guard(
        V40_RUNTIME_MAPPING_REPORT, EXPECTED_V40_REPORT_FILE_SHA256,
        "LoRA runtime mapping report",
    )
    runtime_report = json.loads(
        V40_RUNTIME_MAPPING_REPORT.read_text(encoding="utf-8")
    )
    runtime_inventory = runtime_report.get("inventory", {})
    if (
        runtime_report.get("schema") != "lora-topology-probe-report-v40a"
        or runtime_report.get("status") != "complete_train_only_four_gpu"
        or runtime_report.get("synthetic_prompt_only") is not True
        or runtime_report.get("dataset_or_evaluation_accessed") is not False
        or runtime_report.get("content_sha256_before_self_field")
        != EXPECTED_V40_REPORT_CONTENT_SHA256
        or canonical_sha256(_without_self(runtime_report))
        != EXPECTED_V40_REPORT_CONTENT_SHA256
        or runtime_inventory.get("peft_elements") != 4_528_128
        or runtime_inventory.get("peft_tensor_count") != 70
        or runtime_inventory.get("runtime_active_allocated_elements")
        != 4_921_344
        or runtime_inventory.get("runtime_view_count") != 82
        or runtime_inventory.get("runtime_dtypes") != ["torch.bfloat16"]
    ):
        raise RuntimeError("audited LoRA-to-vLLM mapping evidence changed")
    current = moe_v69.build_geometry_manifest()
    moe_v69.validate_geometry_manifest(current)
    config = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    text = config["text_config"]
    layer_types = text["layer_types"]
    index = json.loads(MODEL_INDEX.read_text(encoding="utf-8"))
    expected_types = list(MOTIF_TYPES) * (NUM_LAYERS // MOTIF_WIDTH)
    if (
        config.get("architectures") != ["Qwen3_5MoeForConditionalGeneration"]
        or text.get("num_hidden_layers") != NUM_LAYERS
        or layer_types != expected_types
        or text.get("num_experts") != NUM_EXPERTS
        or text.get("num_experts_per_tok") != EXPERTS_PER_TOKEN
        or len(index.get("weight_map", {})) != moe_v69.EXPECTED_MODEL_INDEX_KEYS
    ):
        raise RuntimeError("Qwen3.6 hybrid architecture changed")
    records = _all_target_records(index, layer_types)
    if len(records) != 350:
        raise RuntimeError("40-layer active LoRA target inventory changed")

    signatures = {}
    for position in range(MOTIF_WIDTH):
        baseline = []
        for record in records:
            if record["layer"] != 20 + position:
                continue
            baseline.append({
                key: record[key] for key in (
                    "motif_position", "layer_type", "family", "suffix",
                    "shape", "dtype", "runtime_slices",
                )
            })
        signatures[str(position)] = sorted(
            baseline, key=lambda item: (item["family"], item["suffix"])
        )
        for layer in range(position, NUM_LAYERS, MOTIF_WIDTH):
            observed = []
            for record in records:
                if record["layer"] != layer:
                    continue
                observed.append({
                    key: record[key] for key in (
                        "motif_position", "layer_type", "family", "suffix",
                        "shape", "dtype", "runtime_slices",
                    )
                })
            observed.sort(key=lambda item: (item["family"], item["suffix"]))
            if observed != signatures[str(position)]:
                raise RuntimeError(
                    f"layer {layer} no longer matches motif position {position}"
                )

    manifest = {
        "schema": "qwen36-hybrid-lora-topology-geometry-v70",
        "model_config_sha256": EXPECTED_MODEL_CONFIG_SHA256,
        "model_index_sha256": EXPECTED_MODEL_INDEX_SHA256,
        "reference_adapter_config_sha256": EXPECTED_ADAPTER_CONFIG_SHA256,
        "reference_adapter_weights_sha256": EXPECTED_ADAPTER_WEIGHTS_SHA256,
        "reference_adapter_parameter_count": current[
            "reference_adapter_parameter_count"
        ],
        "current_active_module_count": 35,
        "current_active_tensor_count": 70,
        "current_layers": [20, 21, 22, 23],
        "num_layers": NUM_LAYERS,
        "layer_types": layer_types,
        "motif_starts": list(MOTIF_STARTS),
        "motif_types": list(MOTIF_TYPES),
        "shape_compatible_motif_ranges": [
            {
                "start": start,
                "end_exclusive": start + MOTIF_WIDTH,
                "layers": list(range(start, start + MOTIF_WIDTH)),
                "semantic_mergeability_proven": False,
            }
            for start in MOTIF_STARTS
        ],
        "all_target_records": records,
        "all_target_record_count": len(records),
        "all_target_record_inventory_sha256": canonical_sha256(records),
        "motif_position_signatures": signatures,
        "motif_position_signature_sha256": canonical_sha256(signatures),
        "lora_runtime_mapping_file_sha256": (
            EXPECTED_LORA_RUNTIME_MAPPING_SHA256
        ),
        "lora_runtime_mapping_report_content_sha256": (
            EXPECTED_V40_REPORT_CONTENT_SHA256
        ),
        "reference_runtime_allocated_elements": 4_921_344,
        "reference_runtime_view_count": 82,
        "raw_model_tensor_materialized": False,
        "protected_or_training_content_opened": False,
    }
    manifest["content_sha256_before_self_field"] = canonical_sha256(manifest)
    return manifest


def validate_architecture_manifest(manifest: dict) -> None:
    if manifest != build_architecture_manifest():
        raise RuntimeError("topology key, layer, shape, or runtime alias drift")


def _surface(geometry: dict, motif_starts: Iterable[int], rank: int) -> dict:
    starts = tuple(motif_starts)
    if (
        not starts or len(starts) != len(set(starts))
        or starts != tuple(sorted(starts))
        or any(start not in MOTIF_STARTS for start in starts)
        or isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0
    ):
        raise ValueError("surface requires unique aligned motifs and positive rank")
    layers = tuple(
        layer for start in starts for layer in range(start, start + MOTIF_WIDTH)
    )
    selected = [
        record for record in geometry["all_target_records"]
        if record["layer"] in layers
    ]
    targets = []
    expanded_runtime_slices = set()
    for record in selected:
        out_features, in_features = record["shape"]
        target = {
            "base_key": record["base_key"],
            "layer": record["layer"],
            "motif_position": record["motif_position"],
            "layer_type": record["layer_type"],
            "family": record["family"],
            "peft_logical_module": record["peft_logical_module"],
            "runtime_target": record["runtime_target"],
            "runtime_slices": record["runtime_slices"],
            "base_shape": record["shape"],
            "rank": rank,
            "lora_a_shape": [rank, in_features],
            "lora_b_shape": [out_features, rank],
            "parameters": rank * (in_features + out_features),
        }
        for slice_index in target["runtime_slices"]:
            signature = (target["runtime_target"], slice_index)
            if signature in expanded_runtime_slices:
                raise RuntimeError("LoRA logical modules collide at runtime")
            expanded_runtime_slices.add(signature)
        targets.append(target)
    targets.sort(key=lambda item: item["base_key"])
    layer_parameter_counts = {
        str(layer): sum(
            item["parameters"] for item in targets if item["layer"] == layer
        )
        for layer in layers
    }
    layer_module_counts = {
        str(layer): sum(item["layer"] == layer for item in targets)
        for layer in layers
    }
    family_counts = {
        family: sum(
            item["parameters"] for item in targets if item["family"] == family
        )
        for family in TARGET_FAMILIES
    }
    runtime_allocated_elements = sum(
        len(item["runtime_slices"]) * math.prod(item["lora_a_shape"])
        + math.prod(item["lora_b_shape"])
        for item in targets
    )
    logical_parameter_count = sum(item["parameters"] for item in targets)
    surface = {
        "motif_starts": list(starts),
        "motif_ranges": [
            [start, start + MOTIF_WIDTH] for start in starts
        ],
        "selected_layers": list(layers),
        "selected_layer_types": [
            geometry["layer_types"][layer] for layer in layers
        ],
        "rank": rank,
        "lora_alpha": 2 * rank,
        "effective_lora_scaling": 2.0,
        "module_count": len(targets),
        "tensor_count": 2 * len(targets),
        "parameter_count": logical_parameter_count,
        "runtime_allocated_elements": runtime_allocated_elements,
        "runtime_packing_duplicate_elements": (
            runtime_allocated_elements - logical_parameter_count
        ),
        "layer_parameter_counts": layer_parameter_counts,
        "layer_module_counts": layer_module_counts,
        "family_parameter_counts": family_counts,
        "runtime_view_count": 2 * len(expanded_runtime_slices),
        "runtime_mapping_sha256": canonical_sha256([
            {
                "peft_logical_module": item["peft_logical_module"],
                "runtime_target": item["runtime_target"],
                "runtime_slices": item["runtime_slices"],
            }
            for item in targets
        ]),
        "targets": targets,
        "arbitrary_individual_layers_are_mergeable_units": False,
    }
    surface["surface_sha256"] = canonical_sha256(surface)
    expected_modules = 35 * len(starts)
    expected_family_counts = {
        moe_v69.FAMILY_SHARED_SEQUENCE: 3_250_176,
        moe_v69.FAMILY_SHARED_EXPERT: 983_040,
        moe_v69.FAMILY_ROUTER: 294_912,
    }
    if (
        len(layers) != MOTIF_WIDTH * len(starts)
        or surface["selected_layer_types"] != list(MOTIF_TYPES) * len(starts)
        or surface["module_count"] != expected_modules
        or surface["parameter_count"] != TRAINABLE_PARAMETER_BUDGET
        or surface["runtime_allocated_elements"] != 4_921_344
        or surface["runtime_packing_duplicate_elements"] != 393_216
        or surface["family_parameter_counts"] != expected_family_counts
        or rank * len(starts) != 32
    ):
        raise RuntimeError("topology surface lost exact budget or motif coverage")
    return surface


def build_arm_specs(geometry: dict | None = None) -> dict[str, dict]:
    if geometry is None:
        geometry = build_architecture_manifest()
    validate_architecture_manifest(geometry)
    result = {}
    for arm_id in ARM_ORDER:
        definition = ARM_DEFINITIONS[arm_id]
        surface = _surface(
            geometry, definition["motif_starts"], definition["rank"]
        )
        result[arm_id] = {
            "arm_id": arm_id,
            "topology": definition["topology"],
            "surface": surface,
            "router_policy": "tuned_equal_family_budget_all_arms",
            "canonical_master_dtype": "float32",
            "runtime_adapter_dtype": "bfloat16",
            "runtime_max_lora_rank": definition["rank"],
            "rank_padding_elements_allowed": 0,
            "es_update_budget": {
                "direction_count": DIRECTION_COUNT,
                "signed_population": SIGNED_POPULATION,
                "update_steps": UPDATE_STEPS,
                "perturbation_scalar_draws": PERTURBATION_SCALAR_DRAWS,
            },
            "optimization_rollouts_per_run": OPTIMIZATION_ROLLOUTS_PER_RUN,
            "primary_evaluation_rollouts_per_run": (
                PRIMARY_EVALUATION_ROLLOUTS_PER_RUN
            ),
            "layer_sensitivity_rollouts_per_run": (
                LAYER_SENSITIVITY_ROLLOUTS_PER_RUN
            ),
            "total_evaluation_rollouts_per_run": (
                TOTAL_EVALUATION_ROLLOUTS_PER_RUN
            ),
        }
    identities = {
        (
            spec["surface"]["parameter_count"],
            spec["surface"]["runtime_allocated_elements"],
            canonical_sha256(spec["surface"]["family_parameter_counts"]),
            canonical_sha256(spec["es_update_budget"]),
            spec["optimization_rollouts_per_run"],
            spec["total_evaluation_rollouts_per_run"],
        )
        for spec in result.values()
    }
    if len(identities) != 1:
        raise RuntimeError("topology arms are no longer exactly budget matched")
    return result


def build_custom_surface(motif_starts: Iterable[int], rank: int) -> dict:
    geometry = build_architecture_manifest()
    return _surface(geometry, tuple(motif_starts), rank)


def build_insertion_continuation_registry() -> dict:
    _guard(V23_MODEL_SEAL, EXPECTED_V23_MODEL_SEAL_FILE_SHA256, "V23 model seal")
    _guard(
        V23_NEGATIVE_EVIDENCE, EXPECTED_V23_NEGATIVE_FILE_SHA256,
        "V23 negative evidence",
    )
    seal = json.loads(V23_MODEL_SEAL.read_text(encoding="utf-8"))
    negative = json.loads(V23_NEGATIVE_EVIDENCE.read_text(encoding="utf-8"))
    if (
        seal.get("schema") != "eggroll-es-insertion-model-seal-v23a"
        or seal.get("content_sha256_before_self_field")
        != EXPECTED_V23_MODEL_SEAL_CONTENT_SHA256
        or canonical_sha256(_without_self(seal))
        != EXPECTED_V23_MODEL_SEAL_CONTENT_SHA256
        or negative.get("schema")
        != "eggroll-es-v23a-insertion-negative-aggregate-evidence-r3"
        or negative.get("content_sha256_before_self_field")
        != EXPECTED_V23_NEGATIVE_CONTENT_SHA256
        or canonical_sha256(_without_self(negative))
        != EXPECTED_V23_NEGATIVE_CONTENT_SHA256
    ):
        raise RuntimeError("V23 insertion evidence content changed")
    runtime = negative.get("runtime_integrity", {})
    closure = negative.get("closure", {})
    if (
        runtime.get("all_four_arms_passed") is not True
        or runtime.get("exact_restore_every_arm_every_wave") is not True
        or runtime.get("unselected_origin_unchanged") is not True
        or closure.get("model_update_authorized") is not False
        or closure.get("confirmation_authorized") is not False
        or negative.get("aggregate_gate", {}).get(
            "selected_location_for_confirmation"
        ) is not None
    ):
        raise RuntimeError("V23 insertion runtime or closure semantics changed")
    mapping = {
        "insert_front_e005": {
            "source_layers": [0, 1, 2, 3],
            "target_layers": [4, 5, 6, 7],
            "runtime_mapping_sha256": (
                "9c51558f4e134249997aacff139b035860ccf888d6458fc0ab26edaa6c79f80f"
            ),
            "closed_hypothesis": "insert_front_e005_layers_4_7",
        },
        "insert_middle_e005": {
            "source_layers": [16, 17, 18, 19],
            "target_layers": [20, 21, 22, 23],
            "runtime_mapping_sha256": (
                "7ebd15cb9c04cfe8dab67009dc2af5c0054131a11c15a6c8e83f277ffef4585c"
            ),
            "closed_hypothesis": "insert_middle_e005_layers_20_23",
        },
        "insert_back_e005": {
            "source_layers": [36, 37, 38, 39],
            "target_layers": [40, 41, 42, 43],
            "runtime_mapping_sha256": (
                "cdfeae8da8b703d48355dfc182188f850d3a1a9cda01e7989437c791682c3b5a"
            ),
            "closed_hypothesis": "insert_back_e005_layers_40_43",
        },
    }
    closed = set(closure.get("closed_hypotheses", ()))
    candidates = {}
    for arm_id, binding in mapping.items():
        arm = seal["arms"][arm_id]
        if (
            arm.get("source_layers") != binding["source_layers"]
            or arm.get("target_layers") != binding["target_layers"]
            or arm.get("epsilon") != 0.05
            or arm.get("untouched_surgery_output_eligible") is not True
            or binding["closed_hypothesis"] not in closed
        ):
            raise RuntimeError("V23 insertion checkpoint binding changed")
        candidates[arm_id] = {
            "source_layers": binding["source_layers"],
            "target_layers": binding["target_layers"],
            "epsilon": 0.05,
            "config_sha256": arm["config_sha256"],
            "index_sha256": arm["index_sha256"],
            "source_map_sha256": arm["source_map_sha256"],
            "runtime_mapping_sha256": binding["runtime_mapping_sha256"],
            "checkpoint_shapes_and_vllm_mapping_exact": True,
            "prior_train_only_result": "closed_negative_v23a_r3",
            "continuation_launch_authorized": False,
            "semantic_mergeability_proven": False,
        }
    value = {
        "schema": "qwen36-motif-insertion-continuation-registry-v70",
        "v23_model_seal_content_sha256": (
            EXPECTED_V23_MODEL_SEAL_CONTENT_SHA256
        ),
        "v23_negative_evidence_content_sha256": (
            EXPECTED_V23_NEGATIVE_CONTENT_SHA256
        ),
        "candidates": candidates,
        "only_complete_aligned_motifs_are_shape_compatible": True,
        "all_existing_epsilon_005_hypotheses_remain_closed": True,
        "fresh_strength_checkpoint_mapping_and_preregistration_required": True,
        "topology_ablation_cannot_reopen_insertion_automatically": True,
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def bind_quality_receipt(arm_id: str, seed: int, receipt: dict) -> dict:
    if arm_id not in ARM_ORDER:
        raise ValueError("unregistered topology arm")
    if seed not in TRAINING_SEEDS:
        raise ValueError("unregistered topology seed")
    value = {
        "arm_id": arm_id,
        "training_seed": seed,
        "trust_receipt": receipt,
        "trust_receipt_content_sha256": receipt.get(
            "content_sha256_before_self_field"
        ),
    }
    value["binding_sha256"] = canonical_sha256(value)
    return value


def _quality_observation(binding: dict, arm_id: str, seed: int) -> dict:
    _exact_keys(binding, (
        "arm_id", "training_seed", "trust_receipt",
        "trust_receipt_content_sha256", "binding_sha256",
    ), "topology quality binding")
    if binding["binding_sha256"] != canonical_sha256({
        key: value for key, value in binding.items()
        if key != "binding_sha256"
    }):
        raise RuntimeError("topology quality binding changed")
    if binding["arm_id"] != arm_id or binding["training_seed"] != seed:
        raise RuntimeError("quality receipt is bound to another arm or seed")
    receipt = binding["trust_receipt"]
    if binding["trust_receipt_content_sha256"] != receipt.get(
        "content_sha256_before_self_field"
    ):
        raise RuntimeError("quality trust receipt hash binding changed")
    try:
        trust_v67.require_promotion(receipt)
        passed = True
    except RuntimeError as error:
        if "non-compensable hard gate" not in str(error):
            raise
        passed = False
    metrics = receipt.get("component_and_aggregate_metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError("quality trust receipt lost aggregate metrics")
    return {
        "passed": passed,
        "train": metrics.get("train"),
        "dev": metrics.get("dev"),
        "ood": metrics.get("ood"),
    }


_ACTIVITY_KEYS = (
    "registered_surface_sha256", "registered_runtime_mapping_sha256",
    "registered_module_count", "registered_tensor_count",
    "canonical_master_elements", "updated_parameter_count",
    "nonzero_update_elements", "update_l2_by_family",
    "runtime_adapter_allocated_elements", "runtime_adapter_dtype_bytes",
    "rank_padding_elements", "runtime_max_lora_rank",
    "inactive_target_count", "unsupported_target_count",
    "effective_lora_scaling",
)
_SENSITIVITY_KEYS = (
    "module_count", "parameter_count", "nonzero_update_elements",
    "update_l2", "router_update_l2", "masked_reward_delta",
    "sensitivity_rollouts",
)
_ROUTING_KEYS = (
    "router_weight_sha256_before", "router_weight_sha256_after",
    "router_update_l2", "observation_tokens", "active_experts",
    "normalized_entropy", "expert_load_cv",
    "topk_disagreement_rate_from_base", "routing_jsd_mean_from_base",
    "route_inventory_sha256",
)
_SYSTEM_KEYS = (
    "device_total_bytes", "peak_allocated_bytes", "peak_reserved_bytes",
    "safe_headroom_bytes", "adapter_resident_bytes", "generated_tokens",
    "wall_seconds", "generated_tokens_per_second", "memory_traffic_bytes",
    "memory_traffic_seconds", "memory_bandwidth_gib_per_second",
)
_ATTEMPT_KEYS = (
    "arm", "gpu_residency_intervals", "optimization_generated_rollouts",
    "evaluation_generated_rollouts", "generated_tokens",
    "teacher_forced_tokens", "sft_nonpadding_tokens",
)


def _schedule() -> list[tuple[int, str]]:
    result = []
    for seed in TRAINING_SEEDS:
        result.extend((seed, arm_id) for arm_id in SCHEDULE_BY_SEED[seed])
    if (
        len(result) != len(TRAINING_SEEDS) * len(ARM_ORDER)
        or len(set(result)) != len(result)
        or any(set(SCHEDULE_BY_SEED[seed]) != set(ARM_ORDER)
               for seed in TRAINING_SEEDS)
    ):
        raise RuntimeError("topology counterbalanced schedule changed")
    return result


def _validated_compute_attempt(
    attempt: dict,
    *,
    arm_id: str,
    start: float,
    end: float,
) -> dict:
    _exact_keys(attempt, _ATTEMPT_KEYS, "topology compute attempt")
    if attempt["arm"] != arm_id:
        raise RuntimeError("compute attempt charged to another topology arm")
    intervals = attempt["gpu_residency_intervals"]
    if not isinstance(intervals, list) or len(intervals) != 4:
        raise ValueError("compute attempt requires exactly four GPU intervals")
    by_gpu = {}
    for interval in intervals:
        _exact_keys(
            interval, ("physical_gpu_id", "start_s", "end_s"),
            "GPU residency interval",
        )
        gpu = _integer(interval["physical_gpu_id"], "physical GPU id")
        interval_start = _finite(interval["start_s"], "GPU interval start")
        interval_end = _finite(interval["end_s"], "GPU interval end")
        if gpu in by_gpu or gpu not in (0, 1, 2, 3):
            raise ValueError("GPU residency IDs must be exactly 0,1,2,3")
        by_gpu[gpu] = (interval_start, interval_end)
    if set(by_gpu) != {0, 1, 2, 3}:
        raise ValueError("compute attempt does not cover all four GPUs")
    if any(
        not math.isclose(interval_start, start, abs_tol=1e-9)
        or not math.isclose(interval_end, end, abs_tol=1e-9)
        for interval_start, interval_end in by_gpu.values()
    ):
        raise RuntimeError("GPU residency does not match the sealed run interval")
    optimization = _integer(
        attempt["optimization_generated_rollouts"],
        "optimization generated rollouts",
    )
    evaluation = _integer(
        attempt["evaluation_generated_rollouts"],
        "evaluation generated rollouts",
    )
    generated = _integer(
        attempt["generated_tokens"], "generated tokens", 1
    )
    _integer(attempt["teacher_forced_tokens"], "teacher-forced tokens")
    _integer(attempt["sft_nonpadding_tokens"], "SFT nonpadding tokens")
    if (
        optimization != OPTIMIZATION_ROLLOUTS_PER_RUN
        or evaluation != TOTAL_EVALUATION_ROLLOUTS_PER_RUN
    ):
        raise RuntimeError("topology rollout budget changed")
    evaluation_v1.charge_compute_attempt(attempt)
    return {"generated_tokens": generated}


def _run_checks(evidence: dict, spec: dict) -> tuple[dict, dict]:
    _exact_keys(evidence, (
        "schema", "arm_id", "training_seed", "launch_sequence_index",
        "surface_sha256", "started_monotonic_s", "ended_monotonic_s",
        "aggregate_quality_reward", "target_activity",
        "layerwise_sensitivity", "routing_by_layer", "system_metrics",
        "es_update_budget", "compute_attempt", "quality_trust_binding",
        "raw_output_access_count", "protected_access_count",
        "protected_source_opened",
    ), "topology run evidence")
    if evidence["schema"] != "qwen36-front-tail-topology-run-evidence-v70":
        raise ValueError("topology run evidence schema changed")
    arm_id = evidence["arm_id"]
    seed = evidence["training_seed"]
    if arm_id != spec["arm_id"] or seed not in TRAINING_SEEDS:
        raise RuntimeError("topology evidence arm or seed changed")
    schedule = _schedule()
    expected_index = schedule.index((seed, arm_id))
    if evidence["launch_sequence_index"] != expected_index:
        raise RuntimeError("topology launch sequence binding changed")
    if evidence["surface_sha256"] != spec["surface"]["surface_sha256"]:
        raise RuntimeError("topology surface identity changed")
    start = _finite(evidence["started_monotonic_s"], "topology run start")
    end = _finite(evidence["ended_monotonic_s"], "topology run end")
    if end <= start:
        raise ValueError("topology run interval must be positive")
    reward = _finite(
        evidence["aggregate_quality_reward"], "aggregate quality reward"
    )
    if reward < -1.0 or reward > 1.0:
        raise ValueError("aggregate quality reward must be in [-1,1]")
    raw_access = _integer(
        evidence["raw_output_access_count"], "raw output access count"
    )
    protected_access = _integer(
        evidence["protected_access_count"], "protected access count"
    )
    if (
        raw_access != 0 or protected_access != 0
        or evidence["protected_source_opened"] is not False
    ):
        raise RuntimeError("raw or protected output access is prohibited")

    surface = spec["surface"]
    activity = _exact_keys(
        evidence["target_activity"], _ACTIVITY_KEYS, "target activity"
    )
    registered_surface = _sha(
        activity["registered_surface_sha256"], "registered surface"
    )
    registered_mapping = _sha(
        activity["registered_runtime_mapping_sha256"],
        "registered runtime mapping",
    )
    registered_modules = _integer(
        activity["registered_module_count"], "registered modules"
    )
    registered_tensors = _integer(
        activity["registered_tensor_count"], "registered tensors"
    )
    master_elements = _integer(
        activity["canonical_master_elements"], "canonical master elements"
    )
    updated_parameters = _integer(
        activity["updated_parameter_count"], "updated parameters"
    )
    nonzero_updates = _integer(
        activity["nonzero_update_elements"], "nonzero update elements"
    )
    runtime_elements = _integer(
        activity["runtime_adapter_allocated_elements"],
        "runtime adapter allocated elements",
    )
    runtime_dtype_bytes = _integer(
        activity["runtime_adapter_dtype_bytes"], "runtime adapter dtype bytes", 1
    )
    rank_padding = _integer(
        activity["rank_padding_elements"], "rank padding elements"
    )
    runtime_rank = _integer(
        activity["runtime_max_lora_rank"], "runtime max LoRA rank", 1
    )
    inactive = _integer(
        activity["inactive_target_count"], "inactive target count"
    )
    unsupported = _integer(
        activity["unsupported_target_count"], "unsupported target count"
    )
    scaling = _finite(
        activity["effective_lora_scaling"], "effective LoRA scaling"
    )
    family_l2_raw = _exact_keys(
        activity["update_l2_by_family"], TARGET_FAMILIES,
        "family update L2",
    )
    family_l2 = {
        family: _finite(value, f"family update L2 {family}")
        for family, value in family_l2_raw.items()
    }
    checks = {
        "registered_surface_exact": registered_surface == surface["surface_sha256"],
        "runtime_mapping_exact": (
            registered_mapping == surface["runtime_mapping_sha256"]
        ),
        "module_and_tensor_coverage_exact": (
            registered_modules == surface["module_count"]
            and registered_tensors == surface["tensor_count"]
        ),
        "logical_parameter_budget_exact": (
            master_elements == updated_parameters
            == surface["parameter_count"] == TRAINABLE_PARAMETER_BUDGET
        ),
        "runtime_allocation_has_no_rank_padding": (
            runtime_elements == surface["runtime_allocated_elements"]
            and rank_padding == 0
            and runtime_rank == surface["rank"]
            and runtime_dtype_bytes == 2
        ),
        "all_target_families_updated": (
            0 < nonzero_updates <= surface["parameter_count"]
            and all(value > 0.0 for value in family_l2.values())
        ),
        "no_inactive_or_unsupported_targets": inactive == unsupported == 0,
        "lora_scaling_equal_across_ranks": math.isclose(
            scaling, surface["effective_lora_scaling"], abs_tol=1e-12
        ),
        "es_update_budget_exact": evidence["es_update_budget"]
        == spec["es_update_budget"],
    }

    selected_layer_keys = tuple(
        str(layer) for layer in surface["selected_layers"]
    )
    sensitivity = _exact_keys(
        evidence["layerwise_sensitivity"], selected_layer_keys,
        "layerwise sensitivity",
    )
    sensitivity_observed = {}
    sensitivity_nonzero_total = 0
    sensitivity_rollout_total = 0
    for layer_key in selected_layer_keys:
        row = _exact_keys(
            sensitivity[layer_key], _SENSITIVITY_KEYS,
            f"layerwise sensitivity {layer_key}",
        )
        module_count = _integer(row["module_count"], "layer module count")
        parameter_count = _integer(
            row["parameter_count"], "layer parameter count"
        )
        nonzero = _integer(
            row["nonzero_update_elements"], "layer nonzero update elements"
        )
        update_l2 = _finite(row["update_l2"], "layer update L2")
        router_l2 = _finite(row["router_update_l2"], "layer router update L2")
        reward_delta = _finite(
            row["masked_reward_delta"], "layer masked reward delta"
        )
        rollouts = _integer(
            row["sensitivity_rollouts"], "layer sensitivity rollouts", 1
        )
        expected_parameters = surface["layer_parameter_counts"][layer_key]
        expected_modules = surface["layer_module_counts"][layer_key]
        expected_rollouts = (
            LAYER_SENSITIVITY_ROLLOUTS_PER_RUN
            // len(surface["selected_layers"])
        )
        checks[f"layer_{layer_key}_activity_and_budget_exact"] = (
            module_count == expected_modules
            and parameter_count == expected_parameters
            and 0 < nonzero <= expected_parameters
            and update_l2 > 0.0 and router_l2 > 0.0
            and -2.0 <= reward_delta <= 2.0
            and rollouts == expected_rollouts
        )
        sensitivity_nonzero_total += nonzero
        sensitivity_rollout_total += rollouts
        sensitivity_observed[layer_key] = {
            "module_count": module_count,
            "parameter_count": parameter_count,
            "nonzero_update_elements": nonzero,
            "update_l2": update_l2,
            "router_update_l2": router_l2,
            "masked_reward_delta": reward_delta,
            "sensitivity_rollouts": rollouts,
        }
    checks["layerwise_sensitivity_complete_and_nonpadding"] = (
        sensitivity_nonzero_total == nonzero_updates
        and sensitivity_rollout_total == LAYER_SENSITIVITY_ROLLOUTS_PER_RUN
    )

    routing = _exact_keys(
        evidence["routing_by_layer"], selected_layer_keys, "routing by layer"
    )
    routing_observed = {}
    for layer_key in selected_layer_keys:
        row = _exact_keys(
            routing[layer_key], _ROUTING_KEYS, f"routing layer {layer_key}"
        )
        before = _sha(row["router_weight_sha256_before"], "router before")
        after = _sha(row["router_weight_sha256_after"], "router after")
        router_l2 = _finite(row["router_update_l2"], "router update L2")
        tokens = _integer(row["observation_tokens"], "routing tokens")
        active = _integer(row["active_experts"], "active experts")
        entropy = _finite(row["normalized_entropy"], "routing entropy")
        load_cv = _finite(row["expert_load_cv"], "expert load CV")
        disagreement = _finite(
            row["topk_disagreement_rate_from_base"], "routing disagreement"
        )
        jsd = _finite(row["routing_jsd_mean_from_base"], "routing JSD")
        route_hash = _sha(row["route_inventory_sha256"], "route inventory")
        checks[f"layer_{layer_key}_router_policy_and_drift_bounded"] = (
            before != after and router_l2 > 0.0
            and math.isclose(
                router_l2,
                sensitivity_observed[layer_key]["router_update_l2"],
                rel_tol=1e-9, abs_tol=1e-12,
            )
            and tokens >= MIN_ROUTING_TOKENS_PER_LAYER
            and MIN_ACTIVE_EXPERTS_PER_LAYER <= active <= NUM_EXPERTS
            and MIN_ROUTING_ENTROPY <= entropy <= 1.0
            and 0.0 <= load_cv <= MAX_EXPERT_LOAD_CV
            and 0.0 <= disagreement <= MAX_TOPK_ROUTING_DISAGREEMENT
            and 0.0 <= jsd <= MAX_ROUTING_JSD
        )
        routing_observed[layer_key] = {
            "observation_tokens": tokens,
            "active_experts": active,
            "normalized_entropy": entropy,
            "expert_load_cv": load_cv,
            "topk_disagreement_rate_from_base": disagreement,
            "routing_jsd_mean_from_base": jsd,
            "router_update_l2": router_l2,
            "route_inventory_sha256": route_hash,
        }

    attempt_observed = _validated_compute_attempt(
        evidence["compute_attempt"], arm_id=arm_id, start=start, end=end
    )
    system = _exact_keys(
        evidence["system_metrics"], _SYSTEM_KEYS, "system metrics"
    )
    device_total = _integer(
        system["device_total_bytes"], "device total bytes", 1
    )
    peak_allocated = _integer(
        system["peak_allocated_bytes"], "peak allocated bytes", 1
    )
    peak_reserved = _integer(
        system["peak_reserved_bytes"], "peak reserved bytes", 1
    )
    safe_headroom = _integer(
        system["safe_headroom_bytes"], "safe headroom bytes"
    )
    adapter_bytes = _integer(
        system["adapter_resident_bytes"], "adapter resident bytes", 1
    )
    generated_tokens = _integer(
        system["generated_tokens"], "system generated tokens", 1
    )
    wall_seconds = _finite(system["wall_seconds"], "system wall seconds")
    tokens_per_second = _finite(
        system["generated_tokens_per_second"], "generated tokens per second"
    )
    traffic_bytes = _integer(
        system["memory_traffic_bytes"], "memory traffic bytes", 1
    )
    traffic_seconds = _finite(
        system["memory_traffic_seconds"], "memory traffic seconds"
    )
    bandwidth = _finite(
        system["memory_bandwidth_gib_per_second"],
        "memory bandwidth GiB/s",
    )
    checks.update({
        "throughput_receipt_consistent": (
            wall_seconds > 0.0
            and generated_tokens == attempt_observed["generated_tokens"]
            and math.isclose(
                wall_seconds, end - start, rel_tol=1e-9, abs_tol=1e-9
            )
            and math.isclose(
                tokens_per_second, generated_tokens / wall_seconds,
                rel_tol=1e-9, abs_tol=1e-9,
            )
        ),
        "vram_receipt_consistent_and_safe": (
            peak_allocated <= peak_reserved <= device_total
            and safe_headroom == device_total - peak_reserved
            and safe_headroom >= MIN_SAFE_HEADROOM_BYTES
            and adapter_bytes
            == surface["runtime_allocated_elements"] * runtime_dtype_bytes
        ),
        "memory_bandwidth_receipt_consistent": (
            0.0 < traffic_seconds <= wall_seconds
            and math.isclose(
                bandwidth,
                traffic_bytes / traffic_seconds / (1024**3),
                rel_tol=1e-9, abs_tol=1e-12,
            )
        ),
    })
    quality = _quality_observation(
        evidence["quality_trust_binding"], arm_id, seed
    )
    return checks, {
        "arm_id": arm_id,
        "training_seed": seed,
        "launch_sequence_index": expected_index,
        "started_monotonic_s": start,
        "ended_monotonic_s": end,
        "aggregate_quality_reward": reward,
        "quality": quality,
        "layerwise_sensitivity": sensitivity_observed,
        "routing_by_layer": routing_observed,
        "system_metrics": {
            "device_total_bytes": device_total,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "safe_headroom_bytes": safe_headroom,
            "adapter_resident_bytes": adapter_bytes,
            "generated_tokens": generated_tokens,
            "wall_seconds": wall_seconds,
            "generated_tokens_per_second": tokens_per_second,
            "memory_traffic_bytes": traffic_bytes,
            "memory_traffic_seconds": traffic_seconds,
            "memory_bandwidth_gib_per_second": bandwidth,
        },
    }


def analyze_topology(run_evidence: list[dict]) -> dict:
    geometry = build_architecture_manifest()
    specs = build_arm_specs(geometry)
    schedule = _schedule()
    if (
        not isinstance(run_evidence, list)
        or len(run_evidence) != len(schedule)
    ):
        raise ValueError("topology evidence must cover every sealed arm and seed")
    by_key = {}
    for evidence in run_evidence:
        if not isinstance(evidence, dict):
            raise ValueError("topology run evidence must be objects")
        key = (evidence.get("training_seed"), evidence.get("arm_id"))
        if key not in schedule or key in by_key:
            raise ValueError("topology evidence is missing, duplicate, or foreign")
        by_key[key] = evidence
    if set(by_key) != set(schedule):
        raise ValueError("topology evidence does not cover the sealed schedule")

    integrity_checks = {}
    observations = {arm_id: {} for arm_id in ARM_ORDER}
    attempts = []
    previous_end = None
    for expected_index, (seed, arm_id) in enumerate(schedule):
        evidence = by_key[(seed, arm_id)]
        checks, observed = _run_checks(evidence, specs[arm_id])
        observations[arm_id][str(seed)] = observed
        for name, passed in checks.items():
            integrity_checks[f"{arm_id}:seed{seed}:{name}"] = passed
        start = observed["started_monotonic_s"]
        end = observed["ended_monotonic_s"]
        integrity_checks[f"schedule_index_{expected_index}:nonoverlap"] = (
            previous_end is None or start >= previous_end
        )
        previous_end = end
        attempts.append(evidence["compute_attempt"])

    totals = evaluation_v1.aggregate_compute_ledger(attempts)
    compute_match = evaluation_v1.validate_compute_match(
        totals,
        mode="compute_matched_quality",
        contract=trust_v67.load_evaluation_contract(),
    )
    expected_work = {
        "optimization_generated_rollouts": (
            len(TRAINING_SEEDS) * OPTIMIZATION_ROLLOUTS_PER_RUN
        ),
        "evaluation_generated_rollouts": (
            len(TRAINING_SEEDS) * TOTAL_EVALUATION_ROLLOUTS_PER_RUN
        ),
    }
    integrity_checks["all_arms_four_gpu_compute_matched"] = (
        compute_match["passed"] is True and set(totals) == set(ARM_ORDER)
        and all(
            totals[arm_id][key] == value
            for arm_id in ARM_ORDER
            for key, value in expected_work.items()
        )
    )
    integrity_checks["all_arm_parameter_family_and_perturbation_budgets_equal"] = (
        len({
            (
                spec["surface"]["parameter_count"],
                canonical_sha256(spec["surface"]["family_parameter_counts"]),
                canonical_sha256(spec["es_update_budget"]),
            )
            for spec in specs.values()
        }) == 1
    )

    current = observations[CONTROL_ARM]
    current_quality_pass = all(
        current[str(seed)]["quality"]["passed"] for seed in TRAINING_SEEDS
    )
    candidate_gates = {}
    eligible_candidates = []
    reward_deltas = {}
    for arm_id in ARM_ORDER:
        if arm_id == CONTROL_ARM:
            continue
        candidate = observations[arm_id]
        deltas = [
            candidate[str(seed)]["aggregate_quality_reward"]
            - current[str(seed)]["aggregate_quality_reward"]
            for seed in TRAINING_SEEDS
        ]
        reward_deltas[arm_id] = {
            "by_seed": {
                str(seed): deltas[index]
                for index, seed in enumerate(TRAINING_SEEDS)
            },
            "mean": math.fsum(deltas) / len(deltas),
            "positive_seed_count": sum(delta > 0.0 for delta in deltas),
        }
        quality_pass = all(
            candidate[str(seed)]["quality"]["passed"]
            for seed in TRAINING_SEEDS
        )
        throughput_ratios = {
            str(seed): (
                candidate[str(seed)]["system_metrics"][
                    "generated_tokens_per_second"
                ]
                / current[str(seed)]["system_metrics"][
                    "generated_tokens_per_second"
                ]
            )
            for seed in TRAINING_SEEDS
        }
        peak_deltas = {
            str(seed): (
                candidate[str(seed)]["system_metrics"]["peak_reserved_bytes"]
                - current[str(seed)]["system_metrics"]["peak_reserved_bytes"]
            )
            for seed in TRAINING_SEEDS
        }
        traffic_ratios = {
            str(seed): (
                candidate[str(seed)]["system_metrics"]["memory_traffic_bytes"]
                / current[str(seed)]["system_metrics"]["memory_traffic_bytes"]
            )
            for seed in TRAINING_SEEDS
        }
        gates = {
            "train_dev_ood_trust_all_seeds": quality_pass,
            "paired_reward_mean_strict_improvement": (
                reward_deltas[arm_id]["mean"] > 0.0
            ),
            "paired_reward_improves_at_least_two_seeds": (
                reward_deltas[arm_id]["positive_seed_count"] >= 2
            ),
            "throughput_noninferior_each_seed": all(
                value >= MIN_THROUGHPUT_RATIO_VS_CONTROL
                for value in throughput_ratios.values()
            ),
            "peak_vram_delta_bounded_each_seed": all(
                value <= MAX_PEAK_RESERVED_DELTA_VS_CONTROL_BYTES
                for value in peak_deltas.values()
            ),
            "memory_traffic_noninferior_each_seed": all(
                value <= MAX_MEMORY_TRAFFIC_RATIO_VS_CONTROL
                for value in traffic_ratios.values()
            ),
        }
        candidate_gates[arm_id] = {
            "checks": gates,
            "throughput_ratio_vs_control_by_seed": throughput_ratios,
            "peak_reserved_delta_vs_control_bytes_by_seed": peak_deltas,
            "memory_traffic_ratio_vs_control_by_seed": traffic_ratios,
            "eligible": all(gates.values()),
        }
        if all(gates.values()):
            eligible_candidates.append(arm_id)

    integrity_passed = all(integrity_checks.values())
    selected = None
    if integrity_passed and current_quality_pass:
        selected = CONTROL_ARM
        if eligible_candidates:
            selected = max(
                eligible_candidates,
                key=lambda arm_id: (
                    reward_deltas[arm_id]["mean"],
                    -ARM_ORDER.index(arm_id),
                ),
            )
    result = {
        "schema": "qwen36-front-tail-lora-topology-analysis-v70",
        "geometry_content_sha256": geometry[
            "content_sha256_before_self_field"
        ],
        "arm_surface_sha256s": {
            arm_id: specs[arm_id]["surface"]["surface_sha256"]
            for arm_id in ARM_ORDER
        },
        "run_observations_by_arm_and_seed": observations,
        "compute_totals": totals,
        "compute_match": compute_match,
        "integrity_checks": dict(sorted(integrity_checks.items())),
        "failed_integrity_checks": sorted(
            name for name, passed in integrity_checks.items() if not passed
        ),
        "current_control_train_dev_ood_passed_all_seeds": (
            current_quality_pass
        ),
        "candidate_reward_deltas": reward_deltas,
        "candidate_promotion_gates": candidate_gates,
        "selected_hpo_topology": selected,
        "integrity_passed": integrity_passed,
        "hpo_selection_eligible": selected is not None,
        "protected_terminal_evaluation_performed": False,
        "protected_terminal_promotion_eligible": False,
        "selection_is_provisional_until_live_dependency_and_terminal_gate": True,
        "posthoc_unregistered_range_selection_allowed": False,
        "insertion_or_duplication_continuation_authorized": False,
        "raw_output_or_protected_content_persisted": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def require_hpo_selection(result: dict) -> str:
    compact = _without_self(result)
    checks = result.get("integrity_checks")
    integrity = (
        isinstance(checks, dict) and bool(checks)
        and all(isinstance(value, bool) for value in checks.values())
        and all(checks.values())
    )
    selected = result.get("selected_hpo_topology")
    eligible = integrity and selected in ARM_ORDER
    if (
        result.get("schema")
        != "qwen36-front-tail-lora-topology-analysis-v70"
        or result.get("content_sha256_before_self_field")
        != canonical_sha256(compact)
        or result.get("integrity_passed") is not integrity
        or result.get("hpo_selection_eligible") is not eligible
        or result.get("protected_terminal_evaluation_performed") is not False
        or result.get("protected_terminal_promotion_eligible") is not False
        or result.get("insertion_or_duplication_continuation_authorized")
        is not False
        or result.get("raw_output_or_protected_content_persisted") is not False
    ):
        raise RuntimeError("invalid or mutated topology analysis")
    if not eligible:
        raise RuntimeError("topology analysis failed integrity or control trust")
    return selected


def build_preregistration() -> dict:
    geometry = build_architecture_manifest()
    specs = build_arm_specs(geometry)
    insertion = build_insertion_continuation_registry()
    value = {
        "schema": "qwen36-front-tail-lora-topology-preregistration-v70",
        "status": (
            "cpu_protocol_sealed_pending_specialist_0j5_14_and_live_evidence"
        ),
        "created_at_utc": "2026-07-17T00:00:00+00:00",
        "purpose": (
            "Test the front/tail topology prior at exact LoRA parameter, "
            "target-family, perturbation, rollout, and optimizer semantics."
        ),
        "evaluation_contract_content_sha256": (
            trust_v67.EXPECTED_EVALUATION_CONTRACT_CONTENT_SHA256
        ),
        "multiobjective_policy_semantics_sha256": (
            trust_v67.POLICY_SEMANTICS_SHA256
        ),
        "architecture": {
            "geometry_content_sha256": geometry[
                "content_sha256_before_self_field"
            ],
            "all_target_record_inventory_sha256": geometry[
                "all_target_record_inventory_sha256"
            ],
            "motif_position_signature_sha256": geometry[
                "motif_position_signature_sha256"
            ],
            "num_layers": NUM_LAYERS,
            "layer_types": geometry["layer_types"],
            "motif_starts": list(MOTIF_STARTS),
            "motif_types": list(MOTIF_TYPES),
            "active_target_records_all_layers": 350,
            "current_active_modules_layers_20_23": 35,
            "current_active_tensors_layers_20_23": 70,
            "current_adapter_parameters": 4_528_128,
            "audited_vllm_mapping_report_content_sha256": (
                EXPECTED_V40_REPORT_CONTENT_SHA256
            ),
            "audited_runtime_allocated_elements_current_surface": 4_921_344,
            "audited_runtime_view_count_current_surface": 82,
            "shape_compatible_motif_ranges": geometry[
                "shape_compatible_motif_ranges"
            ],
            "shape_compatibility_does_not_prove_semantic_mergeability": True,
            "arbitrary_layers_are_not_copy_merge_or_insertion_units": True,
        },
        "arms": {
            arm_id: {
                "topology": spec["topology"],
                "motif_starts": spec["surface"]["motif_starts"],
                "motif_ranges": spec["surface"]["motif_ranges"],
                "selected_layers": spec["surface"]["selected_layers"],
                "selected_layer_types": spec["surface"][
                    "selected_layer_types"
                ],
                "rank": spec["surface"]["rank"],
                "lora_alpha": spec["surface"]["lora_alpha"],
                "effective_lora_scaling": spec["surface"][
                    "effective_lora_scaling"
                ],
                "module_count": spec["surface"]["module_count"],
                "tensor_count": spec["surface"]["tensor_count"],
                "parameter_count": spec["surface"]["parameter_count"],
                "runtime_allocated_elements": spec["surface"][
                    "runtime_allocated_elements"
                ],
                "runtime_packing_duplicate_elements": spec["surface"][
                    "runtime_packing_duplicate_elements"
                ],
                "layer_parameter_counts": spec["surface"][
                    "layer_parameter_counts"
                ],
                "layer_module_counts": spec["surface"][
                    "layer_module_counts"
                ],
                "family_parameter_counts": spec["surface"][
                    "family_parameter_counts"
                ],
                "runtime_mapping_sha256": spec["surface"][
                    "runtime_mapping_sha256"
                ],
                "surface_sha256": spec["surface"]["surface_sha256"],
                "runtime_max_lora_rank": spec["runtime_max_lora_rank"],
                "rank_padding_elements_allowed": 0,
                "router_policy": spec["router_policy"],
                "es_update_budget": spec["es_update_budget"],
                "optimization_rollouts_per_run": (
                    spec["optimization_rollouts_per_run"]
                ),
                "primary_evaluation_rollouts_per_run": (
                    spec["primary_evaluation_rollouts_per_run"]
                ),
                "layer_sensitivity_rollouts_per_run": (
                    spec["layer_sensitivity_rollouts_per_run"]
                ),
                "total_evaluation_rollouts_per_run": (
                    spec["total_evaluation_rollouts_per_run"]
                ),
            }
            for arm_id, spec in specs.items()
        },
        "counterbalanced_schedule": {
            "seed_order": list(TRAINING_SEEDS),
            "arms_by_seed": {
                str(seed): list(SCHEDULE_BY_SEED[seed])
                for seed in TRAINING_SEEDS
            },
            "global_launch_sequence": [
                {"training_seed": seed, "arm_id": arm_id}
                for seed, arm_id in _schedule()
            ],
            "all_runs_use_all_four_physical_gpus": True,
            "all_prior_runs_finish_before_next_start": True,
            "posthoc_arm_or_range_addition_allowed": False,
        },
        "exact_matching": {
            "trainable_parameters_each_arm": TRAINABLE_PARAMETER_BUDGET,
            "runtime_bfloat16_allocated_elements_each_arm": 4_921_344,
            "runtime_packing_duplicate_elements_each_arm": 393_216,
            "runtime_packing_duplication_is_counted_not_rank_padding": True,
            "family_parameter_counts_each_arm": {
                moe_v69.FAMILY_SHARED_SEQUENCE: 3_250_176,
                moe_v69.FAMILY_SHARED_EXPERT: 983_040,
                moe_v69.FAMILY_ROUTER: 294_912,
            },
            "direction_count_each_run": DIRECTION_COUNT,
            "signed_population_each_run": SIGNED_POPULATION,
            "update_steps_each_run": UPDATE_STEPS,
            "perturbation_scalar_draws_each_run": PERTURBATION_SCALAR_DRAWS,
            "optimization_rollouts_each_run": OPTIMIZATION_ROLLOUTS_PER_RUN,
            "evaluation_rollouts_each_run": TOTAL_EVALUATION_ROLLOUTS_PER_RUN,
            "optimizer_sigma_sampling_and_initialization_identical": True,
            "alpha_over_rank_each_arm": 2.0,
            "rank_padding_may_not_count_as_trainable_budget": True,
        },
        "layerwise_sensitivity": {
            "total_masked_rollouts_each_arm_seed": (
                LAYER_SENSITIVITY_ROLLOUTS_PER_RUN
            ),
            "rollouts_are_divided_equally_over_selected_layers": True,
            "every_selected_layer_reports_module_and_parameter_count": True,
            "every_selected_layer_reports_nonzero_count_and_update_l2": True,
            "every_selected_layer_reports_router_update_l2": True,
            "every_selected_layer_reports_masked_reward_delta": True,
            "sensitivity_is_diagnostic_not_posthoc_range_authority": True,
        },
        "router_confound_controls": {
            "router_tuned_in_every_arm": True,
            "router_parameters_each_arm": 294_912,
            "router_scaling_equal": True,
            "router_before_after_hash_and_positive_update_required_each_layer": True,
            "minimum_routing_tokens_per_layer": MIN_ROUTING_TOKENS_PER_LAYER,
            "minimum_active_experts_per_layer": MIN_ACTIVE_EXPERTS_PER_LAYER,
            "minimum_normalized_entropy": MIN_ROUTING_ENTROPY,
            "maximum_expert_load_cv": MAX_EXPERT_LOAD_CV,
            "maximum_topk_disagreement": MAX_TOPK_ROUTING_DISAGREEMENT,
            "maximum_routing_jsd": MAX_ROUTING_JSD,
        },
        "quality_and_system_gates": {
            "train_dev_ood_trust_required_all_three_seeds": True,
            "paired_reward_mean_strict_improvement_over_current": True,
            "paired_reward_positive_at_least_two_of_three_seeds": True,
            "minimum_throughput_ratio_each_seed": (
                MIN_THROUGHPUT_RATIO_VS_CONTROL
            ),
            "maximum_peak_reserved_delta_bytes_each_seed": (
                MAX_PEAK_RESERVED_DELTA_VS_CONTROL_BYTES
            ),
            "maximum_memory_traffic_ratio_each_seed": (
                MAX_MEMORY_TRAFFIC_RATIO_VS_CONTROL
            ),
            "minimum_safe_headroom_bytes": MIN_SAFE_HEADROOM_BYTES,
            "compute_mode": "compute_matched_quality",
            "gpu_second_relative_tolerance": 0.02,
        },
        "insertion_and_duplication": insertion,
        "fail_closed_cases": [
            "checkpoint key shape motif or runtime alias drift",
            "inactive missing foreign or unsupported target module",
            "logical rank padded to fake equal trainable parameters",
            "router budget scaling activity or routing drift differs",
            "parameter perturbation rollout or four-GPU compute mismatch",
            "missing layerwise sensitivity or nonzero-update coverage",
            "post-hoc arm start range motif or rank choice",
            "arbitrary layers treated as semantic merge units",
            "closed V23 epsilon-0.05 insertion hypothesis reopened",
            "raw output or protected content accessed during HPO",
        ],
        "protected_terminal_firewall": {
            "protected_access_count_required_during_hpo": 0,
            "protected_source_opened_required_during_hpo": False,
            "protected_content_or_per_item_metrics_accepted": False,
            "terminal_evaluation_performed_by_this_protocol": False,
            "terminal_promotion_eligible_before_downstream_gate": False,
        },
        "outstanding_before_task_acceptance": {
            "dependency_issue": "specialist-0j5.14",
            "phase_separated_vram_and_bandwidth_profile_complete": False,
            "multi_seed_gpu_runs_complete": False,
            "protected_terminal_evaluation_complete": False,
            "task_must_remain_in_progress": True,
        },
        "implementation": {
            "module": str(Path(__file__).resolve()),
            "module_file_sha256": file_sha256(Path(__file__).resolve()),
        },
        "raw_model_tensor_materialized": False,
        "protected_or_training_content_opened": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def validate_preregistration(value: dict) -> None:
    if value != build_preregistration():
        raise RuntimeError("front/tail topology preregistration changed")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--check", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    value = build_preregistration()
    if args.check:
        validate_preregistration(json.loads(
            PREREGISTRATION.read_text(encoding="utf-8")
        ))
    else:
        print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
