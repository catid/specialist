#!/usr/bin/env python3
"""CPU-only Qwen3.6 LoRA rank/target-surface Pareto preregistration.

This builder reads only sealed JSON metadata and safetensors headers through
the already-reviewed V69/V70 geometry builders.  It does not materialize model
or adapter tensor payloads, import a GPU runtime, open dataset rows, or
authorize a live launch.

Two causal cohorts are deliberately separate:

* the rank cohort fixes layers and target families while changing rank; and
* the family cohort fixes layers and rank while changing target families.

The current rank-32 adapter is shared as the correctness/control arm.  A
different layer topology selected by V70 requires a new preregistration; these
results may not be relabeled as evidence for that different topology.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import eggroll_es_front_tail_topology_v70 as topology_v70
import eggroll_es_moe_targeting_v69 as moe_v69
import eggroll_es_multiobjective_trust_region_v67 as trust_v67
import recipe_evaluation_contract_v1 as evaluation_v1


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_lora_rank_surface_pareto_v1.json"
)
REPORT = ROOT / (
    "experiments/eggroll_es_hpo/"
    "qwen36_lora_rank_surface_pareto_v1_cpu_evidence_20260717.md"
)
ACCESS_INCIDENT = ROOT / (
    "experiments/eggroll_es_hpo/incidents/"
    "protected_holdout_access_20260717_v1.json"
)
EXPECTED_ACCESS_INCIDENT_FILE_SHA256 = (
    "e20d2129a72fc2d314002a5448a8e8332296b0975e345f40140e6895247978ae"
)
EXPECTED_ACCESS_INCIDENT_CONTENT_SHA256 = (
    "df8856617f5facd6fedac21ab7b653681d2a38484f8e7d93f493bccd47932301"
)
SCHEMA = "qwen36-lora-rank-surface-pareto-preregistration-v1"

UPSTREAM_FILES = {
    "eggroll_es_front_tail_topology_v70.py": (
        "9e2ac60f51940cfcb2c3b5f625d859d2eb8477c2f06e1bd03385d81eb0b0f8d7"
    ),
    "eggroll_es_moe_targeting_v69.py": (
        "a1f0e4ab2a6e65b05a5b20ec84db473d28c97cc497e42c1eed311b334355cc49"
    ),
    "eggroll_es_multiobjective_trust_region_v67.py": (
        "0992691aeb9891d67904124acc7b4bfc062a46367282c1040144e6754b1677a6"
    ),
    "recipe_evaluation_contract_v1.py": (
        "e3f57e9290298e2510118e8c9f10c835618fa12206197462e7ae5a0b7ab68c25"
    ),
    "eggroll_es_audit_contract_v71.py": (
        "cc80ac0e1bf3c9db83e3275df16ea1479273d92a40240496163543643bd0eaa8"
    ),
    "eggroll_es_adapter_transport_precision_v81.py": (
        "08e8b74cd8a79ab7615a89877e184cbea50e8ddfbd95bdbfefccc50c13f28bfc"
    ),
}
UPSTREAM_JSON = {
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_front_tail_lora_topology_v70.json": {
        "schema": "qwen36-front-tail-lora-topology-preregistration-v70",
        "file_sha256": (
            "4da6b721a3d62653ab227522822b4f18c84e0a025530faf3193d186167b91dbb"
        ),
        "content_sha256": (
            "69b9e07b96b060d8c6b3a2854fe5cf79e39f4706b77f11460e48dd931743e88a"
        ),
    },
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_moe_lora_targeting_v69.json": {
        "schema": "qwen36-moe-lora-targeting-preregistration-v69",
        "file_sha256": (
            "ce9e7ba9ce2da2f0c74bd452763b944d80b2b0687575d60ca88757bb9c679541"
        ),
        "content_sha256": (
            "562c7b45974b085ab56c58dd41ddc9996c25dc4763b468a1f4a078976f914a47"
        ),
    },
    "experiments/eggroll_es_hpo/preregistrations/"
    "recipe_evaluation_compute_contract_v1.json": {
        "schema": "specialist-recipe-evaluation-compute-contract-v1",
        "file_sha256": (
            "04af81499067e2feb0186c0a61e4c1af10f838a8eb7deec6dd41cd192748cacf"
        ),
        "content_sha256": (
            "2442c0c2be3ac4c883612f400f8f213ce3bc82ef96e03fad1ef10ec3b7d11fad"
        ),
    },
    "experiments/eggroll_es_hpo/preregistrations/"
    "multiobjective_reward_ood_trust_region_v67.json": {
        "schema": "specialist-multiobjective-ood-trust-preregistration-v67",
        "file_sha256": (
            "5001148bc27fb7550dc6b40336ed2d32d9296f2b17f6372a075a02beeee6bf7d"
        ),
        "content_sha256": (
            "feb96e50782f99d736d15e92cddb3ed32a0defad4af07281b5044f5c33e79e11"
        ),
    },
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_exact_audit_traffic_v71.json": {
        "schema": "eggroll-es-exact-audit-traffic-contract-v71",
        "file_sha256": (
            "8747e9ca3c022b593bdfcf445881106d5410c3496f0135bcd2a663f07ca55240"
        ),
        "content_sha256": (
            "14c7afe2fd370798a26641f6950e92592be67e5b2e2e5fabfc442b76462c2f99"
        ),
    },
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_adapter_transport_precision_v81.json": {
        "schema": "qwen36-adapter-transport-precision-preregistration-v81",
        "file_sha256": (
            "cb8f31d981d4d3471adbfb78d962fb7ae028e56e7364013874dcbc0d181e1c25"
        ),
        "content_sha256": (
            "db01b11ca0fc1565cc81a00cd0a6426f80ae840f907eb0446fda5fb0908eeb80"
        ),
    },
}

CURRENT_LAYERS = (20, 21, 22, 23)
ALL_FAMILIES = (
    moe_v69.FAMILY_SHARED_SEQUENCE,
    moe_v69.FAMILY_SHARED_EXPERT,
    moe_v69.FAMILY_ROUTER,
)
RANKS = (8, 16, 32, 64)
CONTROL_ARM = "full_current_r32_control"
ARM_ORDER = (
    CONTROL_ARM,
    "full_current_r8",
    "full_current_r16",
    "full_current_r64",
    "shared_expert_only_r32",
    "attention_gdn_only_r32",
    "frozen_router_dense_r32",
)
ARM_DEFINITIONS = {
    CONTROL_ARM: {"rank": 32, "families": ALL_FAMILIES},
    "full_current_r8": {"rank": 8, "families": ALL_FAMILIES},
    "full_current_r16": {"rank": 16, "families": ALL_FAMILIES},
    "full_current_r64": {"rank": 64, "families": ALL_FAMILIES},
    "shared_expert_only_r32": {
        "rank": 32,
        "families": (moe_v69.FAMILY_SHARED_EXPERT,),
    },
    "attention_gdn_only_r32": {
        "rank": 32,
        "families": (moe_v69.FAMILY_SHARED_SEQUENCE,),
    },
    "frozen_router_dense_r32": {
        "rank": 32,
        "families": (
            moe_v69.FAMILY_SHARED_SEQUENCE,
            moe_v69.FAMILY_SHARED_EXPERT,
        ),
    },
}
RANK_COHORT = (
    "full_current_r8", "full_current_r16", CONTROL_ARM, "full_current_r64",
)
FAMILY_COHORT = (
    "shared_expert_only_r32", "attention_gdn_only_r32",
    "frozen_router_dense_r32", CONTROL_ARM,
)
TRAINING_SEEDS = (1701, 1702, 1703)
SCHEDULES = {
    "1701": list(ARM_ORDER),
    "1702": list(ARM_ORDER[2:] + ARM_ORDER[:2]),
    "1703": list(ARM_ORDER[4:] + ARM_ORDER[:4]),
}

DIRECTION_COUNT = 16
SIGNED_POPULATION = 32
ROLLOUTS_PER_SIGNED_CANDIDATE = 64
OPTIMIZATION_ROLLOUTS = SIGNED_POPULATION * ROLLOUTS_PER_SIGNED_CANDIDATE
DEV_GENERATED_ROLLOUTS = 83
OOD_QA_GENERATED_ROLLOUTS = 24
OOD_PROSE_TEACHER_FORCED_ITEMS = 16
SYSTEMS_SYNTHETIC_ROLLOUTS = 64
EVALUATION_GENERATED_ROLLOUTS = (
    DEV_GENERATED_ROLLOUTS
    + OOD_QA_GENERATED_ROLLOUTS
    + SYSTEMS_SYNTHETIC_ROLLOUTS
)
GPU_SECOND_CEILING = 14_400
GPU_SECOND_RELATIVE_TOLERANCE = 0.02
FP32_BYTES = 4
BF16_BYTES = 2
UINT64_BYTES = 8


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _without_self(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def validate_upstream_contracts() -> dict[str, Any]:
    raise RuntimeError(
        "rank-surface V1 upstream contracts are quarantined; create a V2 successor"
    )
    files = []
    for relative, expected in UPSTREAM_FILES.items():
        path = ROOT / relative
        actual = file_sha256(path)
        _require(actual == expected, f"upstream implementation changed: {relative}")
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": actual})
    json_contracts = []
    for relative, expected in UPSTREAM_JSON.items():
        path = ROOT / relative
        actual = file_sha256(path)
        _require(actual == expected["file_sha256"], f"upstream JSON changed: {relative}")
        value = json.loads(path.read_text(encoding="ascii"))
        body = _without_self(value)
        _require(
            value.get("schema") == expected["schema"]
            and value.get("content_sha256_before_self_field")
            == expected["content_sha256"]
            and canonical_sha256(body) == expected["content_sha256"],
            f"upstream JSON self hash changed: {relative}",
        )
        json_contracts.append({"path": relative, **expected})
    incident_file_sha256 = file_sha256(ACCESS_INCIDENT)
    _require(
        incident_file_sha256 == EXPECTED_ACCESS_INCIDENT_FILE_SHA256,
        "protected-access incident record changed",
    )
    incident = json.loads(ACCESS_INCIDENT.read_text(encoding="ascii"))
    incident_body = _without_self(incident)
    _require(
        incident.get("schema")
        == "specialist-protected-holdout-access-incident-v1"
        and incident.get("content_sha256_before_self_field")
        == EXPECTED_ACCESS_INCIDENT_CONTENT_SHA256
        and canonical_sha256(incident_body)
        == EXPECTED_ACCESS_INCIDENT_CONTENT_SHA256
        and incident["event"]["v1_protected_access_count_is_nonzero"] is True
        and incident["quarantine"][
            "v1_source_may_not_be_used_for_future_terminal_evaluation"
        ] is True,
        "protected-access incident semantics changed",
    )
    # Invoke the reviewed validators without opening their bound dataset paths.
    evaluation_contract = trust_v67.load_evaluation_contract()
    return {
        "implementation_files": files,
        "json_contracts": json_contracts,
        "known_protected_access_incident": {
            "path": str(ACCESS_INCIDENT.relative_to(ROOT)),
            "file_sha256": incident_file_sha256,
            "content_sha256": EXPECTED_ACCESS_INCIDENT_CONTENT_SHA256,
            "bead": "specialist-0j5.30",
            "v1_access_nonzero": True,
            "entire_v1_source_quarantined": True,
        },
        "bundle_sha256": canonical_sha256({
            "implementation_files": files,
            "json_contracts": json_contracts,
            "access_incident_file_sha256": incident_file_sha256,
            "access_incident_content_sha256": (
                EXPECTED_ACCESS_INCIDENT_CONTENT_SHA256
            ),
        }),
    }


def _rank(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in RANKS:
        raise ValueError(f"{label} must be one of {RANKS}")
    return value


def _families(values: Iterable[str]) -> tuple[str, ...]:
    result = tuple(values)
    if (
        not result
        or len(result) != len(set(result))
        or any(family not in ALL_FAMILIES for family in result)
        or result != tuple(family for family in ALL_FAMILIES if family in result)
    ):
        raise ValueError("families must be a nonempty canonical supported subset")
    return result


def build_surface(
    geometry: Mapping[str, Any],
    *,
    rank: int,
    families: Iterable[str],
    runtime_max_lora_rank: int | None = None,
) -> dict[str, Any]:
    rank = _rank(rank, "rank")
    families = _families(families)
    maximum_rank = rank if runtime_max_lora_rank is None else _rank(
        runtime_max_lora_rank, "runtime_max_lora_rank"
    )
    if maximum_rank < rank:
        raise ValueError("runtime_max_lora_rank cannot be smaller than adapter rank")
    selected = [
        record
        for record in geometry["all_target_records"]
        if record["layer"] in CURRENT_LAYERS and record["family"] in families
    ]
    _require(bool(selected), "surface selected no registered modules")
    targets = []
    runtime_slice_identities = set()
    for record in selected:
        out_features, in_features = record["shape"]
        slices = tuple(record["runtime_slices"])
        _require(len(slices) >= 1, "runtime target lost every packed slice")
        for slice_index in slices:
            identity = (record["runtime_target"], slice_index)
            _require(identity not in runtime_slice_identities, "runtime slice collision")
            runtime_slice_identities.add(identity)
        logical = rank * (in_features + out_features)
        active = len(slices) * rank * in_features + out_features * rank
        reserved = (
            len(slices) * maximum_rank * in_features
            + out_features * maximum_rank
        )
        targets.append({
            "base_key": record["base_key"],
            "layer": record["layer"],
            "layer_type": record["layer_type"],
            "family": record["family"],
            "peft_logical_module": record["peft_logical_module"],
            "runtime_target": record["runtime_target"],
            "runtime_slices": list(slices),
            "base_shape": record["shape"],
            "rank": rank,
            "runtime_max_lora_rank": maximum_rank,
            "lora_a_shape": [rank, in_features],
            "lora_b_shape": [out_features, rank],
            "logical_parameters": logical,
            "active_packed_elements": active,
            "packing_duplicate_elements": active - logical,
            "rank_padding_elements": reserved - active,
            "reserved_packed_elements": reserved,
        })
    targets.sort(key=lambda item: item["base_key"])
    logical_parameters = sum(item["logical_parameters"] for item in targets)
    active_packed = sum(item["active_packed_elements"] for item in targets)
    reserved_packed = sum(item["reserved_packed_elements"] for item in targets)
    surface = {
        "selected_layers": list(CURRENT_LAYERS),
        "families": list(families),
        "rank": rank,
        "lora_alpha": 2 * rank,
        "alpha_over_rank": 2.0,
        "runtime_max_lora_rank": maximum_rank,
        "module_count": len(targets),
        "peft_tensor_count": 2 * len(targets),
        "runtime_view_count": 2 * len(runtime_slice_identities),
        "logical_trainable_parameters": logical_parameters,
        "active_packed_elements": active_packed,
        "packing_duplicate_elements": active_packed - logical_parameters,
        "rank_padding_elements": reserved_packed - active_packed,
        "reserved_packed_elements_selected_wrappers": reserved_packed,
        "family_parameter_counts": {
            family: sum(
                item["logical_parameters"]
                for item in targets
                if item["family"] == family
            )
            for family in families
        },
        "layer_parameter_counts": {
            str(layer): sum(
                item["logical_parameters"]
                for item in targets
                if item["layer"] == layer
            )
            for layer in CURRENT_LAYERS
        },
        "active_module_inventory_sha256": canonical_sha256(targets),
        "targets": targets,
    }
    surface["surface_sha256"] = canonical_sha256(surface)
    return surface


def build_byte_ledger(surface: Mapping[str, Any]) -> dict[str, Any]:
    parameters = surface["logical_trainable_parameters"]
    active_packed = surface["active_packed_elements"]
    reserved_packed = surface["reserved_packed_elements_selected_wrappers"]
    master = parameters * FP32_BYTES
    active_runtime = active_packed * BF16_BYTES
    reserved_runtime = reserved_packed * BF16_BYTES
    candidate_installs = SIGNED_POPULATION
    audited_materializations = candidate_installs + 1  # final canonical restore
    ledger = {
        "element_dtypes": {
            "canonical_master": "float32",
            "checkpoint_tensor_payload": "float32",
            "runtime_execution_views": "bfloat16",
        },
        "logical_elements": {
            "canonical_master": parameters,
            "active_packed_runtime": active_packed,
            "packing_duplicates": surface["packing_duplicate_elements"],
            "rank_padding_selected_wrappers": surface["rank_padding_elements"],
            "reserved_packed_selected_wrappers": reserved_packed,
        },
        "persistent_bytes": {
            "canonical_fp32_master": master,
            "active_bf16_runtime_payload": active_runtime,
            "reserved_bf16_selected_wrapper_capacity": reserved_runtime,
            "packing_duplicate_bf16_payload": (
                surface["packing_duplicate_elements"] * BF16_BYTES
            ),
            "rank_padding_bf16_payload": (
                surface["rank_padding_elements"] * BF16_BYTES
            ),
        },
        "host_transaction_line_items_not_assumed_simultaneous": {
            "canonical_master": master,
            "pending_update_master": master,
            "rollback_snapshot_if_not_aliased": master,
            "single_candidate_fp32_transient": master,
        },
        "per_materialization_transfer_payload_bytes": {
            "runtime_install_h2d": active_runtime,
            "single_flat_exact_audit_d2h": active_runtime,
            "prepack_fp32_master_h2d_if_current_gpu_candidate_path": master,
            "owned_fp32_candidate_d2h_if_current_gpu_candidate_path": master,
        },
        "one_update_projected_current_path": {
            "signed_candidate_installs": candidate_installs,
            "final_canonical_restore": 1,
            "audited_runtime_materializations": audited_materializations,
            "runtime_install_and_restore_h2d_bytes": (
                audited_materializations * active_runtime
            ),
            "runtime_exact_audit_d2h_bytes": (
                audited_materializations * active_runtime
            ),
            "candidate_fp32_master_h2d_bytes": candidate_installs * master,
            "owned_candidate_fp32_d2h_bytes": candidate_installs * master,
            "projection_is_not_a_live_counter_receipt": True,
        },
        "checkpoint_tensor_and_step_payload_bytes": {
            "sgd_no_slot": master + UINT64_BYTES,
            "momentum_one_slot": 2 * master + UINT64_BYTES,
            "adamw_two_slots": 3 * master + UINT64_BYTES,
            "three_seed_sgd_no_slot": 3 * (master + UINT64_BYTES),
            "safetensors_header_json_and_manifest_bytes": None,
            "whole_file_bytes_must_be_measured_not_inferred": True,
        },
        "allocation_claim_boundary": {
            "active_payload_and_selected_wrapper_rank_padding_are_exact_geometry": True,
            "unselected_wrapper_or_allocator_reservation_bytes": None,
            "full_device_slot_allocation_requires_live_audit": True,
            "logical_checkpoint_bytes_are_not_vram": True,
        },
    }
    ledger["ledger_sha256"] = canonical_sha256(ledger)
    return ledger


def _initialization(arm_id: str, rank: int, families: tuple[str, ...]) -> dict[str, Any]:
    if arm_id == CONTROL_ARM:
        method = "exact_current_rank32_adapter_bytes"
        function_preserving = True
        compression_part_of_estimand = False
    elif rank == 64 and families == ALL_FAMILIES:
        method = (
            "zero_expand_each_rank32_A_and_B_into_rank64_with_alpha128; "
            "old factors occupy the leading 32 coordinates"
        )
        function_preserving = True
        compression_part_of_estimand = False
    elif rank in (8, 16) and families == ALL_FAMILIES:
        method = (
            "modulewise truncated SVD of the effective rank32 delta with "
            "canonical singular-vector signs and alpha_over_rank_2"
        )
        function_preserving = False
        compression_part_of_estimand = True
    else:
        method = (
            "retain exact current factors for included modules and omit every "
            "excluded family module"
        )
        function_preserving = False
        compression_part_of_estimand = True
    return {
        "method": method,
        "function_preserving_from_current_rank32": function_preserving,
        "compression_or_family_removal_is_part_of_estimand": (
            compression_part_of_estimand
        ),
        "pre_es_hash_only_forward_equivalence_required_if_function_preserving": (
            function_preserving
        ),
        "pre_es_source_disjoint_quality_and_ood_baseline_required": True,
        "rank8_or_rank16_may_not_be_called_a_pure_optimizer_effect": (
            rank in (8, 16)
        ),
        "tensor_conversion_performed_by_cpu_preregistration": False,
    }


def build_arms(geometry: Mapping[str, Any]) -> dict[str, Any]:
    arms = {}
    for order, arm_id in enumerate(ARM_ORDER):
        definition = ARM_DEFINITIONS[arm_id]
        rank = definition["rank"]
        families = tuple(definition["families"])
        surface = build_surface(
            geometry, rank=rank, families=families,
            runtime_max_lora_rank=rank,
        )
        shared_rank64 = build_surface(
            geometry, rank=rank, families=families,
            runtime_max_lora_rank=64,
        )
        arms[arm_id] = {
            "arm_id": arm_id,
            "launch_sequence_index": order,
            "surface": surface,
            "byte_ledger": build_byte_ledger(surface),
            "initialization": _initialization(arm_id, rank, families),
            "dedicated_engine_policy": {
                "runtime_max_lora_rank": rank,
                "rank_padding_elements_expected": 0,
                "separate_engine_required": True,
                "reason": (
                    "a shared rank64 engine would pad lower-rank selected "
                    "wrappers and obscure the deployable memory point"
                ),
            },
            "shared_rank64_engine_diagnostic_excluded_from_pareto": {
                "reserved_selected_wrapper_elements": shared_rank64[
                    "reserved_packed_elements_selected_wrappers"
                ],
                "rank_padding_elements": shared_rank64[
                    "rank_padding_elements"
                ],
                "rank_padding_bytes_bf16": (
                    shared_rank64["rank_padding_elements"] * BF16_BYTES
                ),
                "quality_or_memory_selection_eligible": False,
            },
            "es_budget": {
                "direction_count": DIRECTION_COUNT,
                "signed_population": SIGNED_POPULATION,
                "optimization_generated_rollouts": OPTIMIZATION_ROLLOUTS,
                "perturbation_scalar_draws": (
                    DIRECTION_COUNT * surface["logical_trainable_parameters"]
                ),
            },
        }
    control = arms[CONTROL_ARM]["surface"]
    current_v70 = topology_v70.build_arm_specs()[
        "current_middle_late_motif_r32"
    ]["surface"]
    _require(
        control["module_count"] == current_v70["module_count"] == 35
        and control["peft_tensor_count"] == current_v70["tensor_count"] == 70
        and control["runtime_view_count"] == current_v70["runtime_view_count"] == 82
        and control["logical_trainable_parameters"]
        == current_v70["parameter_count"] == 4_528_128
        and control["active_packed_elements"]
        == current_v70["runtime_allocated_elements"] == 4_921_344
        and control["packing_duplicate_elements"]
        == current_v70["runtime_packing_duplicate_elements"] == 393_216,
        "current full rank32 control lost exact V70 geometry",
    )
    v69 = moe_v69.build_arm_specs()
    expected_family = {
        "shared_expert_only_r32": v69["frozen_router_shared_expert_r32"][
            "surface"
        ]["parameter_count"],
        "attention_gdn_only_r32": v69["frozen_router_attention_gdn_r32"][
            "surface"
        ]["parameter_count"],
        "frozen_router_dense_r32": v69["frozen_router_dense_r32"][
            "surface"
        ]["parameter_count"],
    }
    _require(
        all(
            arms[arm_id]["surface"]["logical_trainable_parameters"] == count
            for arm_id, count in expected_family.items()
        ),
        "family-only cohort lost exact V69 parameter geometry",
    )
    return arms


def build_preregistration() -> dict[str, Any]:
    raise RuntimeError(
        "rank-surface V1 preregistration is historical/nonpromotable; create a "
        "V2 successor"
    )
    upstream = validate_upstream_contracts()
    geometry = topology_v70.build_architecture_manifest()
    topology_v70.validate_architecture_manifest(geometry)
    arms = build_arms(geometry)
    parameter_budgets = {
        arm_id: arms[arm_id]["surface"]["logical_trainable_parameters"]
        for arm_id in ARM_ORDER
    }
    packed_budgets = {
        arm_id: arms[arm_id]["surface"]["active_packed_elements"]
        for arm_id in ARM_ORDER
    }
    value = {
        "schema": SCHEMA,
        "status": "cpu_preregistered_quarantined_pending_evaluation_v2_and_live_dependencies",
        "created_at_utc": "2026-07-17T00:00:00+00:00",
        "bead": "specialist-0j5.27",
        "purpose": (
            "Measure the source-disjoint quality, reward-SNR, VRAM, packed "
            "allocation, transfer, checkpoint, and GPU-second Pareto frontier "
            "for Qwen3.6 LoRA rank and target-surface size."
        ),
        "authority": {
            "builder_cpu_metadata_and_header_audit_only": True,
            "builder_raw_model_or_adapter_tensor_materialized": False,
            "builder_dataset_row_or_protected_content_opened": False,
            "builder_gpu_launch_or_model_update_performed": False,
            "live_hpo_or_scored_evaluation_authorized": False,
            "known_v1_protected_access_incident_is_not_reset": True,
        },
        "upstream_contracts": upstream,
        "geometry": {
            "model_config_sha256": geometry["model_config_sha256"],
            "model_index_sha256": geometry["model_index_sha256"],
            "reference_adapter_config_sha256": geometry[
                "reference_adapter_config_sha256"
            ],
            "reference_adapter_weights_sha256": geometry[
                "reference_adapter_weights_sha256"
            ],
            "all_target_record_count": geometry["all_target_record_count"],
            "all_target_record_inventory_sha256": geometry[
                "all_target_record_inventory_sha256"
            ],
            "current_layers": list(CURRENT_LAYERS),
            "current_full_module_count": 35,
            "current_full_peft_tensor_count": 70,
            "current_full_runtime_view_count": 82,
            "current_full_logical_parameters": 4_528_128,
            "current_full_active_packed_elements": 4_921_344,
            "current_full_packing_duplicate_elements": 393_216,
            "audited_runtime_mapping_report_content_sha256": geometry[
                "lora_runtime_mapping_report_content_sha256"
            ],
        },
        "cohort_separation": {
            "rank_only_fixed_layers_and_families": list(RANK_COHORT),
            "family_only_fixed_layers_and_rank32": list(FAMILY_COHORT),
            "control_shared_between_cohorts": CONTROL_ARM,
            "cross_cohort_difference_may_not_be_attributed_to_rank_alone": True,
            "no_arm_changes_layer_topology": True,
            "v70_topology_winner_dependency": {
                "bead": "specialist-0j5.6",
                "live_selection_complete": False,
                "if_winner_is_not_current_layers_20_23": (
                    "new preregistration must instantiate the rank ladder on "
                    "the exact winning module inventory before promotion"
                ),
                "this_protocol_may_not_promote_another_topology": True,
            },
            "v69_family_selection_dependency": {
                "bead": "specialist-0j5.7",
                "cpu_contract_complete": True,
                "live_family_evidence_complete": False,
                "routed_3d_expert_targets_remain_unsupported": True,
            },
        },
        "arm_order": list(ARM_ORDER),
        "arms": arms,
        "budget_summary": {
            "logical_parameter_budgets": parameter_budgets,
            "active_packed_element_budgets": packed_budgets,
            "distinct_logical_parameter_budget_count": len(set(parameter_budgets.values())),
            "full_current_control_retained": CONTROL_ARM,
            "logical_parameters_are_not_packed_allocation_or_vram": True,
            "family_subset_full_slot_vram_may_not_shrink": (
                "unselected wrappers may remain allocated; live allocator and "
                "slot receipts decide, not active payload arithmetic"
            ),
        },
        "compute_and_schedule": {
            "training_seeds": list(TRAINING_SEEDS),
            "counterbalanced_schedule_by_seed": SCHEDULES,
            "optimization_generated_rollouts_each_arm_seed": OPTIMIZATION_ROLLOUTS,
            "rollouts_per_signed_candidate": ROLLOUTS_PER_SIGNED_CANDIDATE,
            "signed_population": SIGNED_POPULATION,
            "direction_count": DIRECTION_COUNT,
            "evaluation_generated_rollouts_each_arm_seed": (
                EVALUATION_GENERATED_ROLLOUTS
            ),
            "evaluation_breakdown": {
                "source_disjoint_dev_generated": DEV_GENERATED_ROLLOUTS,
                "ood_qa_generated": OOD_QA_GENERATED_ROLLOUTS,
                "ood_prose_teacher_forced": OOD_PROSE_TEACHER_FORCED_ITEMS,
                "data_free_systems_generated": SYSTEMS_SYNTHETIC_ROLLOUTS,
            },
            "exact_rollout_and_crn_identity_equality_required": True,
            "charged_gpu_seconds_include_load_install_generate_update_eval_checkpoint_failures": True,
            "charged_gpu_second_ceiling_each_arm_seed": GPU_SECOND_CEILING,
            "charged_gpu_second_relative_tolerance_within_cohort": (
                GPU_SECOND_RELATIVE_TOLERANCE
            ),
            "every_arm_seed_charged_gpu_seconds_must_match_same_seed_control_within_tolerance": True,
            "simultaneous_rollout_and_gpu_second_match_required_for_causal_quality_claim": True,
            "artificial_idle_padding_prohibited": True,
            "adaptive_extra_training_to_fill_time_prohibited": True,
            "all_four_physical_gpus_useful_each_training_and_evaluation_phase": True,
        },
        "live_active_surface_audit": {
            "every_actor_reports_exact_active_module_inventory_sha256": True,
            "every_actor_reports_peft_keys_shapes_rank_alpha_and_dtype": True,
            "every_actor_reports_runtime_target_slice_and_view_shapes": True,
            "module_tensor_view_and_updated_parameter_counts_exact": True,
            "positive_finite_update_l2_each_registered_family_and_layer": True,
            "unsupported_or_unregistered_target_count_exact": 0,
            "runtime_active_packed_elements_exact": True,
            "packing_duplicate_elements_exact": True,
            "rank_padding_elements_exact": True,
            "unselected_wrapper_slot_bytes_and_allocator_reservation_measured": True,
            "candidate_install_h2d_d2h_and_hbm_bytes_and_calls_measured": True,
            "master_pending_rollback_optimizer_and_checkpoint_bytes_measured": True,
            "whole_checkpoint_file_hash_and_bytes_required": True,
            "peak_vram_safe_headroom_throughput_and_cleanup_required": True,
            "cache_or_slot_state_may_not_cross_arm_candidate_identity": True,
        },
        "reward_snr_reporting": {
            "mirrored_pair_reward_differences_persisted_as_numeric_receipts": True,
            "required_each_arm_seed": [
                "finite_pair_count",
                "mean_pair_difference",
                "standard_deviation_pair_difference",
                "standard_error",
                "absolute_mean_over_standard_deviation_snr",
                "positive_pair_fraction",
                "bootstrap_95pct_interval",
                "rank_utility_entropy",
            ],
            "zero_variance_or_nonfinite_snr_fails_closed": True,
            "snr_is_reported_not_used_to_override_quality_or_ood_hard_gates": True,
        },
        "three_seed_source_disjoint_quality_gates": {
            "evaluation_contract_content_sha256": (
                trust_v67.EXPECTED_EVALUATION_CONTRACT_CONTENT_SHA256
            ),
            "evaluation_contract_v1_binding_quarantined_not_live_eligible": True,
            "fresh_v2_contract_rebind_required_before_any_live_hpo": True,
            "trust_policy_semantics_sha256": trust_v67.POLICY_SEMANTICS_SHA256,
            "required_roles": ["train", "dev", "ood_qa", "ood_prose"],
            "all_three_seeds_required_each_arm": True,
            "common_frozen_pre_es_reference": (
                "exact_current_rank32_adapter_before_any_arm_training"
            ),
            "paired_pareto_control_arm": CONTROL_ARM,
            "dev_rows": DEV_GENERATED_ROLLOUTS,
            "ood_qa_rows": OOD_QA_GENERATED_ROLLOUTS,
            "ood_prose_rows": OOD_PROSE_TEACHER_FORCED_ITEMS,
            "weighted_dev_delta_95_lcb_minimum": -0.01,
            "weighted_dev_point_delta_minimum_exclusive": 0.0,
            "ood_qa_reward_delta_95_lcb_minimum": -0.02,
            "ood_qa_exact_count_delta_minimum": -1,
            "ood_prose_token_logprob_delta_95_lcb_minimum": -0.02,
            "safety_and_hallucination_component_lcb_minimum": 0.0,
            "new_hard_safety_or_quality_event_count_maximum": 0,
            "every_v67_component_event_and_aggregate_gate_required": True,
            "pre_es_compressed_or_subset_initialization_reported_separately": True,
            "aggregate_score_cannot_compensate_for_failed_component_or_event": True,
        },
        "pareto_selection": {
            "eligible_only_after_all_live_compute_activity_surface_and_quality_gates": True,
            "axes_minimize": [
                "logical_fp32_master_bytes",
                "active_packed_bf16_bytes",
                "live_peak_vram_bytes",
                "candidate_install_h2d_bytes",
                "checkpoint_bytes",
                "charged_gpu_seconds",
            ],
            "axes_maximize": [
                "three_seed_source_disjoint_dev_reward",
                "rollouts_per_gpu_second",
                "generated_tokens_per_second",
                "reward_snr",
            ],
            "ood_and_safety_are_hard_gates_not_tradeable_axes": True,
            "dominance": (
                "no worse on every measured axis and strictly better on at "
                "least one; missing or incomparable evidence cannot dominate"
            ),
            "scalarization_or_post_hoc_axis_weights_prohibited": True,
            "select_only_pareto_nondominated_surface": True,
            "current_full_control_reported_even_if_dominated": True,
            "protected_terminal_access_before_recipe_freeze": False,
            "production_promotion_by_this_cpu_contract": False,
        },
        "protected_terminal_firewall": {
            "builder_protected_source_opened": False,
            "known_v1_access_count_is_nonzero": True,
            "known_access_incident_bead": "specialist-0j5.30",
            "known_access_incident_content_sha256": (
                EXPECTED_ACCESS_INCIDENT_CONTENT_SHA256
            ),
            "entire_v1_source_quarantined": True,
            "v1_source_eligible_for_future_terminal_evaluation": False,
            "fresh_untouched_v2_required_before_any_live_hpo_or_terminal_evaluation": True,
            "protected_content_or_per_item_metrics_accepted": False,
            "one_terminal_aggregate_only_after_surface_and_recipe_freeze": True,
            "no_retry_or_retuning_after_protected_open": True,
        },
        "outstanding_before_task_acceptance": {
            "specialist_0j5_30_fresh_evaluation_v2_complete": False,
            "specialist_0j5_14_phase_profile_complete": False,
            "specialist_0j5_6_live_topology_selection_complete": False,
            "v69_live_family_evidence_complete": False,
            "twenty_one_arm_seed_runs_complete": False,
            "live_packed_allocation_and_transfer_ledgers_complete": False,
            "three_seed_quality_ood_and_reward_snr_complete": False,
            "pareto_frontier_frozen": False,
            "protected_terminal_evaluation_complete": False,
            "task_must_remain_in_progress": True,
        },
        "implementation": {
            "builder": str(Path(__file__).resolve()),
            "builder_file_sha256": file_sha256(Path(__file__).resolve()),
        },
    }
    _require(
        value["budget_summary"]["distinct_logical_parameter_budget_count"] >= 4,
        "fewer than four LoRA parameter budgets",
    )
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def validate_preregistration(value: Mapping[str, Any]) -> None:
    raise RuntimeError(
        "rank-surface V1 artifact is nonpromotable after evaluation V1 quarantine"
    )
    expected = build_preregistration()
    _require(dict(value) == expected, "rank/surface preregistration is stale")


def render(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"


def render_report(value: Mapping[str, Any]) -> str:
    rows = []
    for arm_id in value["arm_order"]:
        arm = value["arms"][arm_id]
        surface = arm["surface"]
        bytes_ = arm["byte_ledger"]["persistent_bytes"]
        padding = arm[
            "shared_rank64_engine_diagnostic_excluded_from_pareto"
        ]["rank_padding_bytes_bf16"]
        rows.append(
            "| {arm} | {rank} | {modules} | {logical:,} | {packed:,} | "
            "{duplicate:,} | {master:,} | {runtime:,} | {padding:,} |".format(
                arm=arm_id,
                rank=surface["rank"],
                modules=surface["module_count"],
                logical=surface["logical_trainable_parameters"],
                packed=surface["active_packed_elements"],
                duplicate=surface["packing_duplicate_elements"],
                master=bytes_["canonical_fp32_master"],
                runtime=bytes_["active_bf16_runtime_payload"],
                padding=padding,
            )
        )
    control = value["arms"][CONTROL_ARM]["byte_ledger"]
    control_projection = control["one_update_projected_current_path"]
    control_checkpoints = control["checkpoint_tensor_and_step_payload_bytes"]
    gates = value["three_seed_source_disjoint_quality_gates"]
    compute = value["compute_and_schedule"]
    blockers = value["outstanding_before_task_acceptance"]
    return "\n".join([
        "# Qwen3.6 LoRA rank/target-surface Pareto CPU evidence (V1)",
        "",
        "Status: **CPU preregistered and quarantined; live launch and promotion are not authorized.**",
        "",
        "This artifact defines seven LoRA parameter and active-packed-payload "
        "budgets around the exact current adapter control. The builder reads "
        "sealed metadata and safetensors headers only and opens no dataset "
        "row, tensor payload, or GPU.",
        "",
        "A broad upstream regression-test invocation outside the builder did "
        "open the V1 protected source. This is recorded as a real nonzero "
        "access event in `specialist-0j5.30` and the content-addressed incident "
        "record. The entire V1 source (all 59 rows, including all 18 legacy "
        "heldout candidates) is quarantined. No protected row text escaped "
        "the test process or entered these artifacts, but V1 may not be reset "
        "to zero or used for terminal evaluation. A fresh untouched V2 "
        "contract and rebind are mandatory before any live HPO.",
        "",
        "## Exact geometry and byte budgets",
        "",
        "| Arm | Rank | Modules | Logical params | Active packed elems | "
        "Packing duplicates | FP32 master bytes | Active BF16 bytes | "
        "BF16 padding in shared-r64 diagnostic |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "The deployable measurements use a dedicated engine whose "
        "`max_lora_rank` equals the arm rank, so selected-wrapper rank padding "
        "is exactly zero. The final column is a diagnostic showing how much "
        "selected-wrapper padding a shared rank-64 engine would add; it is "
        "excluded from Pareto selection. Unselected wrappers and allocator "
        "reservations are deliberately not inferred from payload arithmetic "
        "and require live receipts.",
        "",
        "The rank-only cohort fixes layers 20-23 and all supported families at "
        "ranks 8/16/32/64. The family-only cohort fixes layers 20-23 and rank "
        "32 while testing shared-expert, attention/GDN, dense-without-router, "
        "and the full control surfaces. Cross-cohort comparisons cannot be "
        "called rank effects. No arm changes layer topology.",
        "",
        "Rank 64 is a function-preserving zero expansion of the current rank-32 "
        "factors. Ranks 8 and 16 use deterministic modulewise truncated SVD, "
        "so their compression loss is part of the estimand and must be "
        "reported before ES. If the V70 topology task selects anything other "
        "than layers 20-23, this entire ladder must be preregistered again on "
        "the winning inventory.",
        "",
        "## Control-arm transfer and checkpoint ledger",
        "",
        f"- Canonical FP32 master: {control['persistent_bytes']['canonical_fp32_master']:,} bytes.",
        f"- Active packed BF16 payload: {control['persistent_bytes']['active_bf16_runtime_payload']:,} bytes, including {control['persistent_bytes']['packing_duplicate_bf16_payload']:,} bytes of vLLM packing duplicates.",
        f"- Each audited materialization projects {control['per_materialization_transfer_payload_bytes']['runtime_install_h2d']:,} bytes H2D plus {control['per_materialization_transfer_payload_bytes']['single_flat_exact_audit_d2h']:,} bytes D2H for the packed runtime view.",
        f"- The current candidate path additionally projects {control['per_materialization_transfer_payload_bytes']['prepack_fp32_master_h2d_if_current_gpu_candidate_path']:,} FP32-master bytes H2D and {control['per_materialization_transfer_payload_bytes']['owned_fp32_candidate_d2h_if_current_gpu_candidate_path']:,} bytes D2H per signed candidate.",
        f"- One 32-candidate update plus canonical restore projects {control_projection['runtime_install_and_restore_h2d_bytes']:,} packed H2D bytes and {control_projection['runtime_exact_audit_d2h_bytes']:,} audit D2H bytes. Candidate-master H2D and owned-candidate D2H are {control_projection['candidate_fp32_master_h2d_bytes']:,} bytes each.",
        f"- Logical checkpoint payloads including the uint64 step are {control_checkpoints['sgd_no_slot']:,} bytes (SGD), {control_checkpoints['momentum_one_slot']:,} bytes (momentum), and {control_checkpoints['adamw_two_slots']:,} bytes (AdamW). Safetensors header/manifest and whole-file bytes must be measured.",
        "",
        "Every live actor must report exact module identities, PEFT/runtime "
        "shapes and slices, rank/alpha/dtype, updated parameter counts, packed "
        "elements, duplicates, padding, install/audit traffic, checkpoint "
        "bytes, allocator reservations, peak VRAM, safe headroom, throughput, "
        "cleanup, and positive finite update norms by selected family/layer.",
        "",
        "## Compute, validation, and selection",
        "",
        f"Each arm uses seeds {compute['training_seeds']}, exactly {compute['optimization_generated_rollouts_each_arm_seed']:,} optimization rollouts, {compute['evaluation_generated_rollouts_each_arm_seed']:,} generated evaluation rollouts, and a charged ceiling of {compute['charged_gpu_second_ceiling_each_arm_seed']:,} GPU-seconds. Both rollout/CRN identity and charged GPU-seconds within {100 * compute['charged_gpu_second_relative_tolerance_within_cohort']:.0f}% are required; idle padding and adaptive extra work are prohibited.",
        "",
        f"Against the common frozen pre-ES rank-32 reference, source-disjoint gates require all three seeds: dev reward 95% LCB >= {gates['weighted_dev_delta_95_lcb_minimum']} with point delta > {gates['weighted_dev_point_delta_minimum_exclusive']}; OOD-QA reward LCB >= {gates['ood_qa_reward_delta_95_lcb_minimum']} and exact-count delta >= {gates['ood_qa_exact_count_delta_minimum']}; OOD prose token-logprob LCB >= {gates['ood_prose_token_logprob_delta_95_lcb_minimum']}; safety/hallucination component LCBs >= {gates['safety_and_hallucination_component_lcb_minimum']}; and zero new hard events. The trained rank-32 full arm is the paired Pareto control, not the frozen baseline. These V1-bound thresholds are design evidence only until they are rebound unchanged to a fresh V2 boundary; protected data remains terminal and unavailable during HPO.",
        "",
        "Only arms passing every compute, audit, quality, OOD, and safety gate "
        "may enter a nondominated Pareto set over measured quality, bytes, "
        "VRAM, transfer, throughput, SNR, and GPU-seconds. Scalarization and "
        "post-hoc weights are prohibited; the current full control remains in "
        "the report even if dominated.",
        "",
        "## Outstanding dependencies",
        "",
        *[
            f"- `{name}`: {str(state).lower()}"
            for name, state in sorted(blockers.items())
        ],
        "",
        "No empirical quality, VRAM, bandwidth, throughput, SNR, or Pareto "
        "winner is claimed by this CPU preregistration.",
        "",
        f"Canonical content SHA-256: `{value['content_sha256_before_self_field']}`",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render(build_preregistration())
    value = json.loads(expected)
    expected_report = render_report(value)
    if args.check:
        _require(OUTPUT.is_file(), f"missing preregistration: {OUTPUT}")
        _require(
            OUTPUT.read_text(encoding="ascii") == expected,
            "persisted rank/surface preregistration is stale",
        )
        _require(REPORT.is_file(), f"missing CPU evidence report: {REPORT}")
        _require(
            REPORT.read_text(encoding="utf-8") == expected_report,
            "persisted CPU evidence report is stale",
        )
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(expected, encoding="ascii")
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(expected_report, encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "report": str(REPORT),
        "content_sha256": value["content_sha256_before_self_field"],
        "status": value["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
