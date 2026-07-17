#!/usr/bin/env python3
"""CPU-only Qwen3.6 LoRA-ES MoE targeting and routing-drift contract.

Only safetensors headers and aggregate/hash receipts are inspected.  No model
tensor is materialized, no dataset path is accepted, and no GPU API is used.
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

import eggroll_es_multiobjective_trust_region_v67 as trust_v67
import recipe_evaluation_contract_v1 as evaluation_v1


ROOT = Path(__file__).resolve().parent
MODEL = (ROOT / "models/Qwen3.6-35B-A3B").resolve()
MODEL_CONFIG = (MODEL / "config.json").resolve()
MODEL_INDEX = (MODEL / "model.safetensors.index.json").resolve()
REFERENCE_ADAPTER = (
    ROOT / "experiments/sft_controls/"
    "v49d_v434_sampling_midpoint_lr5p5e5/"
    "v434_equal_r32_seed17_init20260715041/final"
).resolve()
REFERENCE_ADAPTER_CONFIG = (REFERENCE_ADAPTER / "adapter_config.json").resolve()
REFERENCE_ADAPTER_WEIGHTS = (
    REFERENCE_ADAPTER / "adapter_model.safetensors"
).resolve()
PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_moe_lora_targeting_v69.json"
).resolve()

EXPECTED_MODEL_CONFIG_SHA256 = (
    "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
)
EXPECTED_MODEL_INDEX_SHA256 = (
    "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"
)
EXPECTED_ADAPTER_CONFIG_SHA256 = (
    "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5"
)
EXPECTED_ADAPTER_WEIGHTS_SHA256 = (
    "0d6efd4d5be626f41cdd711843f799b9b3c09e9ecf6a7a8e6e9aeeff09f6dc5b"
)
EXPECTED_MODEL_INDEX_KEYS = 1045
EXPECTED_MODEL_TENSOR_BYTES = 71_903_645_408
TARGET_LAYERS = (20, 21, 22, 23)
NUM_EXPERTS = 256
EXPERTS_PER_TOKEN = 8
RANK_DEFAULT = 32

FAMILY_SHARED_SEQUENCE = "shared_attention_gdn"
FAMILY_SHARED_EXPERT = "shared_expert_projections"
FAMILY_ROUTED_EXPERT = "routed_expert_projections"
FAMILY_ROUTER = "router_gate"
FAMILY_SHARED_SCALAR_GATE = "shared_expert_scalar_gate"
FAMILY_ORDER = (
    FAMILY_SHARED_SEQUENCE,
    FAMILY_SHARED_EXPERT,
    FAMILY_ROUTED_EXPERT,
    FAMILY_ROUTER,
    FAMILY_SHARED_SCALAR_GATE,
)
SUPPORTED_FAMILIES = frozenset((
    FAMILY_SHARED_SEQUENCE, FAMILY_SHARED_EXPERT, FAMILY_ROUTER,
))

ARM_ORDER = (
    "frozen_router_attention_gdn_r32",
    "frozen_router_shared_expert_r32",
    "frozen_router_dense_r32",
    "router_tuned_parameter_matched",
)
PROMOTABLE_CONTRAST = (
    "frozen_router_dense_r32", "router_tuned_parameter_matched",
)

DIRECTION_COUNT = 16
SIGNED_POPULATION = 32
UPDATE_STEPS = 1
MIN_ROUTING_OBSERVATION_TOKENS_PER_LAYER = 4096
MIN_ACTIVE_EXPERTS_PER_LAYER = 192
MIN_NORMALIZED_ROUTING_ENTROPY = 0.80
MAX_EXPERT_LOAD_CV = 2.0
MIN_COVERAGE_DELTA_FROM_BASE = -8
ROUTING_THRESHOLDS = {
    "frozen_router": {
        "topk_disagreement_rate_maximum": 0.10,
        "probability_total_variation_mean_maximum": 0.10,
        "routing_jsd_mean_maximum": 0.02,
    },
    "router_tuned": {
        "topk_disagreement_rate_maximum": 0.20,
        "probability_total_variation_mean_maximum": 0.20,
        "routing_jsd_mean_maximum": 0.05,
    },
}

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
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
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


def _signed_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _file_guard(path: Path, expected: str, label: str) -> None:
    observed = file_sha256(path)
    if observed != expected:
        raise RuntimeError(f"{label} bytes changed: {observed}")


def _suffixes(layer_type: str) -> dict[str, tuple[str, ...]]:
    if layer_type == "linear_attention":
        sequence = (
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.out_proj.weight",
        )
    elif layer_type == "full_attention":
        sequence = tuple(
            f"self_attn.{name}_proj.weight" for name in ("q", "k", "v", "o")
        )
    else:
        raise RuntimeError("unsupported Qwen layer type")
    return {
        FAMILY_SHARED_SEQUENCE: sequence,
        FAMILY_SHARED_EXPERT: tuple(
            f"mlp.shared_expert.{name}_proj.weight"
            for name in ("gate", "up", "down")
        ),
        FAMILY_ROUTED_EXPERT: (
            "mlp.experts.gate_up_proj", "mlp.experts.down_proj",
        ),
        FAMILY_ROUTER: ("mlp.gate.weight",),
        FAMILY_SHARED_SCALAR_GATE: ("mlp.shared_expert_gate.weight",),
    }


def _selected_base_records(
    index: dict, layer_types: list[str]
) -> list[dict]:
    weight_map = index["weight_map"]
    records = []
    by_file = defaultdict(list)
    for layer in TARGET_LAYERS:
        for family in FAMILY_ORDER:
            for suffix in _suffixes(layer_types[layer])[family]:
                key = f"model.language_model.layers.{layer}.{suffix}"
                if key not in weight_map:
                    raise RuntimeError(f"required model target absent: {key}")
                by_file[weight_map[key]].append((key, layer, family))
    for filename, items in sorted(by_file.items()):
        path = MODEL / filename
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key, layer, family in items:
                view = handle.get_slice(key)
                records.append({
                    "base_key": key,
                    "checkpoint_shard": filename,
                    "layer": layer,
                    "layer_type": layer_types[layer],
                    "family": family,
                    "shape": list(view.get_shape()),
                    "ndim": len(view.get_shape()),
                    "dtype": view.get_dtype(),
                })
    return sorted(records, key=lambda item: item["base_key"])


def _adapter_records() -> list[dict]:
    records = []
    with safe_open(
        REFERENCE_ADAPTER_WEIGHTS, framework="pt", device="cpu"
    ) as handle:
        for key in sorted(handle.keys()):
            view = handle.get_slice(key)
            records.append({
                "key": key,
                "shape": list(view.get_shape()),
                "dtype": view.get_dtype(),
                "elements": math.prod(view.get_shape()),
            })
    return records


def _adapter_keys_for_base(record: dict, rank: int) -> tuple[dict, dict]:
    if record["ndim"] != 2:
        raise ValueError("standard LoRA requires a 2D base surface")
    out_features, in_features = record["shape"]
    logical = record["base_key"].replace(
        "model.language_model.", "model."
    )
    if not logical.endswith(".weight"):
        raise RuntimeError("2D LoRA base key lost its weight suffix")
    logical = logical[:-len(".weight")]
    prefix = f"base_model.model.{logical}"
    return (
        {
            "key": f"{prefix}.lora_A.weight",
            "shape": [rank, in_features],
            "elements": rank * in_features,
        },
        {
            "key": f"{prefix}.lora_B.weight",
            "shape": [out_features, rank],
            "elements": out_features * rank,
        },
    )


def build_geometry_manifest() -> dict:
    _file_guard(MODEL_CONFIG, EXPECTED_MODEL_CONFIG_SHA256, "model config")
    _file_guard(MODEL_INDEX, EXPECTED_MODEL_INDEX_SHA256, "model index")
    _file_guard(
        REFERENCE_ADAPTER_CONFIG, EXPECTED_ADAPTER_CONFIG_SHA256,
        "reference adapter config",
    )
    _file_guard(
        REFERENCE_ADAPTER_WEIGHTS, EXPECTED_ADAPTER_WEIGHTS_SHA256,
        "reference adapter weights",
    )
    config = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    text = config["text_config"]
    layer_types = text["layer_types"]
    index = json.loads(MODEL_INDEX.read_text(encoding="utf-8"))
    adapter_config = json.loads(
        REFERENCE_ADAPTER_CONFIG.read_text(encoding="utf-8")
    )
    if (
        config.get("architectures") != ["Qwen3_5MoeForConditionalGeneration"]
        or text.get("num_hidden_layers") != 40
        or len(layer_types) != 40
        or [layer_types[layer] for layer in TARGET_LAYERS]
        != ["linear_attention"] * 3 + ["full_attention"]
        or text.get("num_experts") != NUM_EXPERTS
        or text.get("num_experts_per_tok") != EXPERTS_PER_TOKEN
        or len(index.get("weight_map", {})) != EXPECTED_MODEL_INDEX_KEYS
        or int(index.get("metadata", {}).get("total_size", -1))
        != EXPECTED_MODEL_TENSOR_BYTES
    ):
        raise RuntimeError("Qwen3.6 model geometry changed")
    expected_targets = {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
        "down_proj", "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z",
        "out_proj",
    }
    expected_parameters = {
        f"model.layers.{layer}.mlp.gate.weight" for layer in TARGET_LAYERS
    }
    if (
        adapter_config.get("r") != RANK_DEFAULT
        or adapter_config.get("lora_alpha") != 64
        or adapter_config.get("layers_to_transform") != list(TARGET_LAYERS)
        or set(adapter_config.get("target_modules", ())) != expected_targets
        or set(adapter_config.get("target_parameters", ()))
        != expected_parameters
    ):
        raise RuntimeError("reference adapter targeting contract changed")

    base_records = _selected_base_records(index, layer_types)
    adapter_records = _adapter_records()
    adapter_by_key = {item["key"]: item for item in adapter_records}
    family_summaries = {}
    covered_adapter_keys = set()
    for family in FAMILY_ORDER:
        records = [item for item in base_records if item["family"] == family]
        supported = family in SUPPORTED_FAMILIES
        expected_adapter = []
        rank32_parameters = 0
        hypothetical_rank32_parameters = 0
        for record in records:
            if record["ndim"] == 2:
                out_features, in_features = record["shape"]
                hypothetical_rank32_parameters += RANK_DEFAULT * (
                    out_features + in_features
                )
            elif record["ndim"] == 3:
                experts, out_features, in_features = record["shape"]
                hypothetical_rank32_parameters += (
                    experts * RANK_DEFAULT * (out_features + in_features)
                )
            else:
                raise RuntimeError("unclassified target tensor dimensionality")
        if supported:
            for record in records:
                a_record, b_record = _adapter_keys_for_base(
                    record, RANK_DEFAULT
                )
                for expected in (a_record, b_record):
                    actual = adapter_by_key.get(expected["key"])
                    if actual is None:
                        raise RuntimeError(
                            f"active reference target absent: {expected['key']}"
                        )
                    if (
                        actual["shape"] != expected["shape"]
                        or actual["dtype"] != "F32"
                    ):
                        raise RuntimeError("reference adapter key geometry changed")
                    covered_adapter_keys.add(expected["key"])
                    expected_adapter.append(expected)
                    rank32_parameters += expected["elements"]
        reason = None
        if family == FAMILY_ROUTED_EXPERT:
            reason = (
                "packed per-expert checkpoint targets are 3D; current PEFT/"
                "EGGROLL/vLLM LoRA surfaces require 2D A/B matrices"
            )
        elif family == FAMILY_SHARED_SCALAR_GATE:
            reason = (
                "2D base key exists but is absent from the sealed adapter and "
                "has no audited EGGROLL-to-vLLM runtime mapping"
            )
        family_summaries[family] = {
            "base_module_count": len(records),
            "base_record_sha256": canonical_sha256(records),
            "base_shapes": dict(sorted(Counter(
                str(item["shape"]) for item in records
            ).items())),
            "base_ndim_values": sorted({item["ndim"] for item in records}),
            "reference_adapter_tensor_count": len(expected_adapter),
            "rank32_lora_parameters": rank32_parameters,
            "hypothetical_rank32_lora_parameters": (
                hypothetical_rank32_parameters
            ),
            "launch_supported": supported,
            "unsupported_reason": reason,
        }
    if (
        len(base_records) != 47
        or len(adapter_records) != 70
        or set(adapter_by_key) != covered_adapter_keys
        or sum(item["elements"] for item in adapter_records) != 4_528_128
    ):
        raise RuntimeError("model/adapter target inventory changed")
    manifest = {
        "schema": "qwen36-moe-lora-key-geometry-v69",
        "model_config_sha256": EXPECTED_MODEL_CONFIG_SHA256,
        "model_index_sha256": EXPECTED_MODEL_INDEX_SHA256,
        "adapter_config_sha256": EXPECTED_ADAPTER_CONFIG_SHA256,
        "adapter_weights_sha256": EXPECTED_ADAPTER_WEIGHTS_SHA256,
        "target_layers": list(TARGET_LAYERS),
        "target_layer_types": [layer_types[layer] for layer in TARGET_LAYERS],
        "num_experts": NUM_EXPERTS,
        "experts_per_token": EXPERTS_PER_TOKEN,
        "base_records": base_records,
        "base_record_inventory_sha256": canonical_sha256(base_records),
        "reference_adapter_records": adapter_records,
        "reference_adapter_record_inventory_sha256": canonical_sha256(
            adapter_records
        ),
        "families": family_summaries,
        "reference_adapter_parameter_count": 4_528_128,
        "raw_model_tensor_materialized": False,
        "protected_or_training_data_opened": False,
    }
    manifest["content_sha256_before_self_field"] = canonical_sha256(manifest)
    return manifest


def validate_geometry_manifest(manifest: dict) -> None:
    if manifest != build_geometry_manifest():
        raise RuntimeError("Qwen3.6 target key alias, shape, or coverage drift")


def _surface(
    geometry: dict,
    families: Iterable[str],
    *,
    rank_overrides: dict[str, int] | None = None,
) -> dict:
    selected_families = tuple(families)
    if (
        not selected_families
        or len(set(selected_families)) != len(selected_families)
        or any(family not in FAMILY_ORDER for family in selected_families)
    ):
        raise ValueError("target families must be unique registered families")
    unsupported = [
        family for family in selected_families
        if not geometry["families"][family]["launch_supported"]
    ]
    if unsupported:
        details = {
            family: geometry["families"][family]["unsupported_reason"]
            for family in unsupported
        }
        raise RuntimeError(f"unsupported LoRA target families: {details}")
    overrides = dict(rank_overrides or {})
    records = [
        item for item in geometry["base_records"]
        if item["family"] in selected_families
    ]
    base_keys = {item["base_key"] for item in records}
    if not set(overrides).issubset(base_keys):
        raise ValueError("rank override references an absent target key")
    targets = []
    for record in records:
        rank = overrides.get(record["base_key"], RANK_DEFAULT)
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError("LoRA target rank must be a positive integer")
        a_record, b_record = _adapter_keys_for_base(record, rank)
        targets.append({
            "base_key": record["base_key"],
            "family": record["family"],
            "base_shape": record["shape"],
            "rank": rank,
            "lora_a_shape": a_record["shape"],
            "lora_b_shape": b_record["shape"],
            "parameters": a_record["elements"] + b_record["elements"],
        })
    targets.sort(key=lambda item: item["base_key"])
    surface = {
        "families": list(selected_families),
        "module_count": len(targets),
        "tensor_count": 2 * len(targets),
        "parameter_count": sum(item["parameters"] for item in targets),
        "rank_histogram": dict(sorted(Counter(
            str(item["rank"]) for item in targets
        ).items())),
        "family_parameter_counts": {
            family: sum(
                item["parameters"] for item in targets
                if item["family"] == family
            )
            for family in selected_families
        },
        "targets": targets,
    }
    surface["surface_sha256"] = canonical_sha256(surface)
    return surface


def build_arm_specs(geometry: dict | None = None) -> dict[str, dict]:
    if geometry is None:
        geometry = build_geometry_manifest()
    validate_geometry_manifest(geometry)
    z_out_overrides = {
        record["base_key"]: 24
        for record in geometry["base_records"]
        if record["layer"] in (20, 21, 22)
        and record["family"] == FAMILY_SHARED_SEQUENCE
        and record["base_key"].endswith((
            ".linear_attn.in_proj_z.weight",
            ".linear_attn.out_proj.weight",
        ))
    }
    definitions = {
        "frozen_router_attention_gdn_r32": {
            "families": (FAMILY_SHARED_SEQUENCE,),
            "overrides": {},
            "router_policy": "frozen_router",
            "phase": "frozen_router",
        },
        "frozen_router_shared_expert_r32": {
            "families": (FAMILY_SHARED_EXPERT,),
            "overrides": {},
            "router_policy": "frozen_router",
            "phase": "frozen_router",
        },
        "frozen_router_dense_r32": {
            "families": (FAMILY_SHARED_SEQUENCE, FAMILY_SHARED_EXPERT),
            "overrides": {},
            "router_policy": "frozen_router",
            "phase": "frozen_router",
        },
        "router_tuned_parameter_matched": {
            "families": (
                FAMILY_SHARED_SEQUENCE, FAMILY_SHARED_EXPERT, FAMILY_ROUTER,
            ),
            "overrides": z_out_overrides,
            "router_policy": "router_tuned",
            "phase": "router_tuned_after_frozen_router",
        },
    }
    expected_counts = {
        "frozen_router_attention_gdn_r32": 3_250_176,
        "frozen_router_shared_expert_r32": 983_040,
        "frozen_router_dense_r32": 4_233_216,
        "router_tuned_parameter_matched": 4_233_216,
    }
    result = {}
    for order, arm_id in enumerate(ARM_ORDER):
        definition = definitions[arm_id]
        surface = _surface(
            geometry, definition["families"],
            rank_overrides=definition["overrides"],
        )
        if surface["parameter_count"] != expected_counts[arm_id]:
            raise RuntimeError("sealed arm parameter count changed")
        result[arm_id] = {
            "arm_id": arm_id,
            "launch_sequence_index": order,
            "phase": definition["phase"],
            "router_policy": definition["router_policy"],
            "surface": surface,
            "es_update_budget": {
                "direction_count": DIRECTION_COUNT,
                "signed_population": SIGNED_POPULATION,
                "update_steps": UPDATE_STEPS,
                "perturbation_scalar_draws": (
                    DIRECTION_COUNT * surface["parameter_count"]
                ),
            },
            "launch_authorized_after_cpu_preflight": True,
        }
    if (
        result[PROMOTABLE_CONTRAST[0]]["surface"]["parameter_count"]
        != result[PROMOTABLE_CONTRAST[1]]["surface"]["parameter_count"]
        or result[PROMOTABLE_CONTRAST[1]]["surface"]["rank_histogram"]
        != {"24": 6, "32": 29}
    ):
        raise RuntimeError("router contrast is no longer exactly parameter matched")
    return result


def build_custom_surface(families: Iterable[str], rank: int = 32) -> dict:
    geometry = build_geometry_manifest()
    selected_families = tuple(families)
    return _surface(
        geometry, selected_families, rank_overrides={
            record["base_key"]: rank
            for record in geometry["base_records"]
            if record["family"] in set(selected_families)
        },
    )


def bind_quality_receipt(arm_id: str, training_seed: int, receipt: dict) -> dict:
    if arm_id not in ARM_ORDER:
        raise ValueError("unregistered targeting arm")
    if training_seed not in (1701, 1702, 1703):
        raise ValueError("unsealed targeting quality seed")
    value = {
        "arm_id": arm_id,
        "training_seed": training_seed,
        "trust_receipt": receipt,
        "trust_receipt_content_sha256": receipt.get(
            "content_sha256_before_self_field"
        ),
    }
    value["binding_sha256"] = canonical_sha256(value)
    return value


def _quality_status(bindings: list[dict], arm_id: str) -> dict[int, dict]:
    if not isinstance(bindings, list):
        raise ValueError("quality trust bindings must be a list")
    result = {}
    for binding in bindings:
        _exact_keys(binding, (
            "arm_id", "training_seed", "trust_receipt",
            "trust_receipt_content_sha256", "binding_sha256",
        ), "quality trust binding")
        compact = {
            key: value for key, value in binding.items()
            if key != "binding_sha256"
        }
        if binding["binding_sha256"] != canonical_sha256(compact):
            raise RuntimeError("quality trust binding changed")
        if binding["arm_id"] != arm_id:
            raise RuntimeError("quality trust receipt bound to another arm")
        seed = binding["training_seed"]
        if seed not in (1701, 1702, 1703) or seed in result:
            raise ValueError("quality trust seeds are unsealed or duplicated")
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
        result[seed] = {
            "passed": passed,
            "train": metrics.get("train"),
            "dev": metrics.get("dev"),
            "ood": metrics.get("ood"),
        }
    if set(result) != {1701, 1702, 1703}:
        raise ValueError("quality trust receipts must cover all sealed seeds")
    return result


_ACTIVITY_KEYS = (
    "registered_surface_sha256", "registered_module_count",
    "updated_parameter_count", "nonzero_update_elements",
    "update_l2_by_family", "unsupported_target_count",
)
_ROUTING_KEYS = (
    "router_weight_sha256_before", "router_weight_sha256_after",
    "router_update_l2", "observation_tokens_per_layer",
    "active_experts_per_layer", "normalized_entropy_per_layer",
    "expert_load_cv_per_layer", "coverage_delta_from_base_per_layer",
    "topk_disagreement_rate_from_base",
    "probability_total_variation_mean_from_base", "routing_jsd_mean_from_base",
    "route_inventory_sha256",
)
_THROUGHPUT_KEYS = (
    "generated_tokens", "wall_seconds", "generated_tokens_per_second",
)


def _layer_map(value: dict, label: str, validator) -> dict:
    _exact_keys(value, tuple(str(layer) for layer in TARGET_LAYERS), label)
    return {
        str(layer): validator(value[str(layer)], f"{label}.{layer}")
        for layer in TARGET_LAYERS
    }


def _arm_checks(evidence: dict, spec: dict) -> tuple[dict, dict]:
    _exact_keys(evidence, (
        "schema", "arm_id", "surface_sha256", "launch_sequence_index",
        "phase", "started_monotonic_s", "ended_monotonic_s",
        "aggregate_quality_reward", "target_activity", "routing_metrics",
        "throughput", "es_update_budget", "compute_attempts",
        "quality_trust_bindings",
        "raw_output_access_count", "protected_access_count",
        "protected_source_opened",
    ), "targeting arm evidence")
    if evidence["schema"] != "qwen36-moe-targeting-arm-evidence-v69":
        raise ValueError("targeting arm evidence schema changed")
    for key in ("arm_id", "launch_sequence_index", "phase"):
        expected = spec[key]
        if evidence[key] != expected:
            raise RuntimeError(f"targeting arm changed sealed {key}")
    if evidence["surface_sha256"] != spec["surface"]["surface_sha256"]:
        raise RuntimeError("targeting arm surface identity changed")
    start = _finite(evidence["started_monotonic_s"], "arm start")
    end = _finite(evidence["ended_monotonic_s"], "arm end")
    if end <= start:
        raise ValueError("targeting arm interval must be positive")
    reward = _finite(evidence["aggregate_quality_reward"], "aggregate reward")
    if reward < -1.0 or reward > 1.0:
        raise ValueError("aggregate reward must be in [-1, 1]")
    raw_access = _integer(
        evidence["raw_output_access_count"], "raw output access count"
    )
    protected_access = _integer(
        evidence["protected_access_count"], "protected access count"
    )
    if (
        raw_access != 0
        or protected_access != 0
        or evidence["protected_source_opened"] is not False
    ):
        raise RuntimeError("raw or protected output access is prohibited")

    activity = _exact_keys(evidence["target_activity"], _ACTIVITY_KEYS, "activity")
    registered_surface = _sha(
        activity["registered_surface_sha256"], "registered surface"
    )
    registered_modules = _integer(
        activity["registered_module_count"], "registered module count"
    )
    updated_parameters = _integer(
        activity["updated_parameter_count"], "updated parameter count"
    )
    nonzero_updates = _integer(
        activity["nonzero_update_elements"], "nonzero update elements"
    )
    unsupported_targets = _integer(
        activity["unsupported_target_count"], "unsupported target count"
    )
    family_l2 = _exact_keys(
        activity["update_l2_by_family"],
        spec["surface"]["families"], "family update L2",
    )
    family_l2 = {
        family: _finite(value, f"update L2 {family}")
        for family, value in family_l2.items()
    }
    checks = {
        "surface_registered_exactly": (
            registered_surface == spec["surface"]["surface_sha256"]
        ),
        "all_target_modules_registered": (
            registered_modules == spec["surface"]["module_count"]
        ),
        "updated_parameter_count_exact": (
            updated_parameters == spec["surface"]["parameter_count"]
        ),
        "target_update_is_active": (
            0 < nonzero_updates <= spec["surface"]["parameter_count"]
            and all(value > 0.0 for value in family_l2.values())
        ),
        "no_unsupported_target_registered": unsupported_targets == 0,
        "es_update_budget_exact": evidence["es_update_budget"]
        == spec["es_update_budget"],
    }

    routing = _exact_keys(evidence["routing_metrics"], _ROUTING_KEYS, "routing")
    before = _sha(routing["router_weight_sha256_before"], "router before")
    after = _sha(routing["router_weight_sha256_after"], "router after")
    update_l2 = _finite(routing["router_update_l2"], "router update L2")
    tokens = _layer_map(
        routing["observation_tokens_per_layer"], "routing tokens",
        lambda value, label: _integer(value, label),
    )
    active = _layer_map(
        routing["active_experts_per_layer"], "active experts",
        lambda value, label: _integer(value, label),
    )
    entropy = _layer_map(
        routing["normalized_entropy_per_layer"], "routing entropy", _finite,
    )
    load_cv = _layer_map(
        routing["expert_load_cv_per_layer"], "expert load CV", _finite,
    )
    coverage_delta = _layer_map(
        routing["coverage_delta_from_base_per_layer"], "coverage delta",
        _signed_integer,
    )
    disagreement = _finite(
        routing["topk_disagreement_rate_from_base"], "top-k disagreement"
    )
    tv = _finite(
        routing["probability_total_variation_mean_from_base"], "routing TV"
    )
    jsd = _finite(routing["routing_jsd_mean_from_base"], "routing JSD")
    _sha(routing["route_inventory_sha256"], "route inventory")
    threshold = ROUTING_THRESHOLDS[spec["router_policy"]]
    checks.update({
        "routing_observation_tokens_sufficient": all(
            value >= MIN_ROUTING_OBSERVATION_TOKENS_PER_LAYER
            for value in tokens.values()
        ),
        "expert_coverage_sufficient": all(
            MIN_ACTIVE_EXPERTS_PER_LAYER <= value <= NUM_EXPERTS
            for value in active.values()
        ),
        "expert_coverage_noninferior": all(
            value >= MIN_COVERAGE_DELTA_FROM_BASE
            for value in coverage_delta.values()
        ),
        "routing_entropy_not_collapsed": all(
            MIN_NORMALIZED_ROUTING_ENTROPY <= value <= 1.0
            for value in entropy.values()
        ),
        "expert_load_cv_bounded": all(
            0.0 <= value <= MAX_EXPERT_LOAD_CV for value in load_cv.values()
        ),
        "topk_routing_drift_bounded": (
            0.0 <= disagreement <= threshold["topk_disagreement_rate_maximum"]
        ),
        "routing_probability_tv_bounded": (
            0.0 <= tv
            <= threshold["probability_total_variation_mean_maximum"]
        ),
        "routing_jsd_bounded": (
            0.0 <= jsd <= threshold["routing_jsd_mean_maximum"]
        ),
    })
    if spec["router_policy"] == "frozen_router":
        checks["router_parameter_change_matches_policy"] = (
            before == after and update_l2 == 0.0
        )
    else:
        checks["router_parameter_change_matches_policy"] = (
            before != after and update_l2 > 0.0
            and family_l2[FAMILY_ROUTER] > 0.0
        )

    throughput = _exact_keys(
        evidence["throughput"], _THROUGHPUT_KEYS, "throughput"
    )
    generated_tokens = _integer(
        throughput["generated_tokens"], "throughput generated tokens", 1
    )
    wall_seconds = _finite(throughput["wall_seconds"], "throughput wall seconds")
    tokens_per_second = _finite(
        throughput["generated_tokens_per_second"],
        "throughput generated tokens per second",
    )
    attempts = evidence["compute_attempts"]
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("each targeting arm requires compute attempts")
    charged_generated_tokens = sum(
        _integer(
            attempt.get("generated_tokens", 0),
            "compute attempt generated tokens",
        )
        for attempt in attempts
    )
    checks["throughput_receipt_consistent"] = (
        wall_seconds > 0.0
        and generated_tokens == charged_generated_tokens
        and math.isclose(wall_seconds, end - start, rel_tol=1e-9, abs_tol=1e-9)
        and math.isclose(
            tokens_per_second, generated_tokens / wall_seconds,
            rel_tol=1e-9, abs_tol=1e-9,
        )
    )
    quality = _quality_status(evidence["quality_trust_bindings"], spec["arm_id"])
    checks["train_dev_ood_trust_passes_all_seeds"] = all(
        item["passed"] for item in quality.values()
    )
    return checks, {
        "started_monotonic_s": start,
        "ended_monotonic_s": end,
        "aggregate_quality_reward": reward,
        "target_families": list(spec["surface"]["families"]),
        "family_parameter_counts": dict(
            spec["surface"]["family_parameter_counts"]
        ),
        "quality_by_seed": {
            str(seed): value for seed, value in sorted(quality.items())
        },
        "throughput": {
            "generated_tokens": generated_tokens,
            "wall_seconds": wall_seconds,
            "generated_tokens_per_second": tokens_per_second,
        },
        "routing": {
            "observation_tokens_per_layer": tokens,
            "active_experts_per_layer": active,
            "normalized_entropy_per_layer": entropy,
            "expert_load_cv_per_layer": load_cv,
            "coverage_delta_from_base_per_layer": coverage_delta,
            "topk_disagreement_rate_from_base": disagreement,
            "probability_total_variation_mean_from_base": tv,
            "routing_jsd_mean_from_base": jsd,
            "router_weight_changed": before != after,
            "router_update_l2": update_l2,
        },
    }


def analyze_targeting(arm_evidence: list[dict]) -> dict:
    geometry = build_geometry_manifest()
    specs = build_arm_specs(geometry)
    if not isinstance(arm_evidence, list) or len(arm_evidence) != len(ARM_ORDER):
        raise ValueError("targeting evidence must contain every sealed arm")
    by_id = {}
    for evidence in arm_evidence:
        arm_id = evidence.get("arm_id") if isinstance(evidence, dict) else None
        if arm_id not in specs or arm_id in by_id:
            raise ValueError("targeting evidence arm is missing, duplicate, or foreign")
        by_id[arm_id] = evidence
    if set(by_id) != set(ARM_ORDER):
        raise ValueError("targeting evidence does not cover the sealed arms")

    checks = {}
    observed = {}
    all_attempts = []
    previous_end = None
    for expected_index, arm_id in enumerate(ARM_ORDER):
        evidence = by_id[arm_id]
        arm_checks, arm_observed = _arm_checks(evidence, specs[arm_id])
        observed[arm_id] = arm_observed
        for name, passed in arm_checks.items():
            checks[f"{arm_id}:{name}"] = passed
        start, end = (
            arm_observed["started_monotonic_s"],
            arm_observed["ended_monotonic_s"],
        )
        checks[f"{arm_id}:launch_order"] = (
            previous_end is None or start >= previous_end
        )
        previous_end = end
        attempts = evidence["compute_attempts"]
        if any(attempt.get("arm") != arm_id for attempt in attempts):
            raise RuntimeError("compute attempt is charged to another arm")
        all_attempts.extend(attempts)

    totals = evaluation_v1.aggregate_compute_ledger(all_attempts)
    compute_match = evaluation_v1.validate_compute_match(
        totals, mode="compute_matched_quality",
        contract=trust_v67.load_evaluation_contract(),
    )
    checks["four_gpu_compute_accounted_and_matched"] = compute_match["passed"]
    frozen_spec, router_spec = (
        specs[PROMOTABLE_CONTRAST[0]], specs[PROMOTABLE_CONTRAST[1]]
    )
    checks["promotable_parameter_counts_exactly_matched"] = (
        frozen_spec["surface"]["parameter_count"]
        == router_spec["surface"]["parameter_count"] == 4_233_216
    )
    checks["promotable_direction_and_update_budgets_exactly_matched"] = (
        frozen_spec["es_update_budget"] == router_spec["es_update_budget"]
    )
    passed = all(checks.values())
    selected = None
    if passed:
        selected = max(
            PROMOTABLE_CONTRAST,
            key=lambda arm_id: (
                observed[arm_id]["aggregate_quality_reward"],
                -PROMOTABLE_CONTRAST.index(arm_id),
            ),
        )
    result = {
        "schema": "qwen36-moe-lora-targeting-analysis-v69",
        "geometry_content_sha256": geometry["content_sha256_before_self_field"],
        "arm_surface_sha256s": {
            arm_id: specs[arm_id]["surface"]["surface_sha256"]
            for arm_id in ARM_ORDER
        },
        "arm_observations": observed,
        "compute_totals": totals,
        "compute_match": compute_match,
        "hard_gate_checks": dict(sorted(checks.items())),
        "failed_hard_gates": sorted(
            name for name, check in checks.items() if not check
        ),
        "selected_promotable_arm": selected,
        "all_hard_gates_passed": passed,
        "promotion_eligible": passed,
        "aggregate_quality_cannot_mask_router_or_target_gate": True,
        "raw_output_or_protected_content_persisted": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def require_selection(result: dict) -> str:
    compact = {
        key: value for key, value in result.items()
        if key != "content_sha256_before_self_field"
    }
    checks = result.get("hard_gate_checks")
    passed = (
        isinstance(checks, dict) and bool(checks)
        and all(isinstance(value, bool) for value in checks.values())
        and all(checks.values())
    )
    selected = result.get("selected_promotable_arm")
    if (
        result.get("schema") != "qwen36-moe-lora-targeting-analysis-v69"
        or result.get("content_sha256_before_self_field")
        != canonical_sha256(compact)
        or result.get("all_hard_gates_passed") is not passed
        or result.get("promotion_eligible") is not passed
        or result.get("raw_output_or_protected_content_persisted") is not False
        or (passed and selected not in PROMOTABLE_CONTRAST)
    ):
        raise RuntimeError("invalid or mutated MoE targeting analysis")
    if not passed:
        raise RuntimeError("MoE targeting ablation failed a hard gate")
    return selected


def build_preregistration() -> dict:
    geometry = build_geometry_manifest()
    specs = build_arm_specs(geometry)
    value = {
        "schema": "qwen36-moe-lora-targeting-preregistration-v69",
        "status": "sealed_before_any_targeting_ablation_model_output",
        "created_at_utc": "2026-07-17T00:00:00+00:00",
        "evaluation_contract_content_sha256": (
            trust_v67.EXPECTED_EVALUATION_CONTRACT_CONTENT_SHA256
        ),
        "multiobjective_policy_semantics_sha256": (
            trust_v67.POLICY_SEMANTICS_SHA256
        ),
        "geometry": {
            "content_sha256": geometry["content_sha256_before_self_field"],
            "base_record_inventory_sha256": geometry[
                "base_record_inventory_sha256"
            ],
            "reference_adapter_record_inventory_sha256": geometry[
                "reference_adapter_record_inventory_sha256"
            ],
            "reference_adapter_parameter_count": geometry[
                "reference_adapter_parameter_count"
            ],
            "families": geometry["families"],
            "model_config_sha256": EXPECTED_MODEL_CONFIG_SHA256,
            "model_index_sha256": EXPECTED_MODEL_INDEX_SHA256,
            "adapter_config_sha256": EXPECTED_ADAPTER_CONFIG_SHA256,
            "adapter_weights_sha256": EXPECTED_ADAPTER_WEIGHTS_SHA256,
        },
        "arms": {
            arm_id: {
                "launch_sequence_index": spec["launch_sequence_index"],
                "phase": spec["phase"],
                "router_policy": spec["router_policy"],
                "families": spec["surface"]["families"],
                "module_count": spec["surface"]["module_count"],
                "tensor_count": spec["surface"]["tensor_count"],
                "parameter_count": spec["surface"]["parameter_count"],
                "rank_histogram": spec["surface"]["rank_histogram"],
                "family_parameter_counts": spec["surface"][
                    "family_parameter_counts"
                ],
                "surface_sha256": spec["surface"]["surface_sha256"],
                "es_update_budget": spec["es_update_budget"],
            }
            for arm_id, spec in specs.items()
        },
        "ordering": {
            "launch_order": list(ARM_ORDER),
            "all_frozen_router_arms_finish_before_router_tuned_start": True,
            "router_result_nonpromotable_if_any_frozen_arm_is_incomplete_or_failed": True,
        },
        "promotable_contrast": {
            "arms": list(PROMOTABLE_CONTRAST),
            "parameter_count_each": 4_233_216,
            "rank_matching": (
                "frozen dense rank32 versus router rank32 plus dense rank32, "
                "except six linear-attention in_proj_z/out_proj modules rank24"
            ),
            "direction_count_each": DIRECTION_COUNT,
            "signed_population_each": SIGNED_POPULATION,
            "update_steps_each": UPDATE_STEPS,
            "perturbation_scalar_draws_each": 67_731_456,
        },
        "target_activity_gates": {
            "registered_surface_and_module_count_exact": True,
            "updated_parameter_count_exact": True,
            "nonzero_update_elements_required": True,
            "positive_update_l2_each_target_family": True,
            "unsupported_target_count_required": 0,
        },
        "routing_observation": {
            "layers": list(TARGET_LAYERS),
            "experts": NUM_EXPERTS,
            "experts_per_token": EXPERTS_PER_TOKEN,
            "minimum_tokens_per_layer": (
                MIN_ROUTING_OBSERVATION_TOKENS_PER_LAYER
            ),
            "minimum_active_experts_per_layer": MIN_ACTIVE_EXPERTS_PER_LAYER,
            "minimum_normalized_entropy": MIN_NORMALIZED_ROUTING_ENTROPY,
            "maximum_expert_load_cv": MAX_EXPERT_LOAD_CV,
            "minimum_coverage_delta_from_base": MIN_COVERAGE_DELTA_FROM_BASE,
            "thresholds_by_router_policy": ROUTING_THRESHOLDS,
            "router_weight_hash_and_update_l2_must_match_policy": True,
        },
        "quality_and_compute": {
            "train_dev_ood_trust_required_each_of_three_seeds_each_arm": True,
            "four_physical_gpu_intervals_required_each_attempt": True,
            "compute_mode": "compute_matched_quality",
            "gpu_second_relative_tolerance": 0.02,
            "evaluation_rollout_counts_exactly_equal": True,
            "aggregate_quality_never_overrides_target_or_routing_gate": True,
            "per_arm_target_families_quality_ood_and_throughput_reported": True,
        },
        "unsupported_surfaces": {
            FAMILY_ROUTED_EXPERT: geometry["families"][
                FAMILY_ROUTED_EXPERT
            ]["unsupported_reason"],
            FAMILY_SHARED_SCALAR_GATE: geometry["families"][
                FAMILY_SHARED_SCALAR_GATE
            ]["unsupported_reason"],
        },
        "adversarial_cases": [
            "checkpoint/adapter alias or key drift invalidates geometry",
            "absent or zero-update target invalidates an arm",
            "unequal parameter, direction, or update budget blocks comparison",
            "router mutation or routing collapse cannot hide in aggregate quality",
            "packed 3D routed-expert LoRA is launch-ineligible",
            "raw output or protected terminal access is prohibited",
        ],
        "implementation": {
            "module": str(Path(__file__).resolve()),
            "module_file_sha256": file_sha256(Path(__file__).resolve()),
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def validate_preregistration(value: dict) -> None:
    if value != build_preregistration():
        raise RuntimeError("MoE targeting preregistration changed")


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
