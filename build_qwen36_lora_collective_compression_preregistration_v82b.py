#!/usr/bin/env python3
"""Build the additive V82B correction for LoRA collective compression.

This CPU-only builder binds the sealed 70-tensor initialization manifest and
the exact V72 production update source.  It records V82 and V75 as immutable
wrong-scope evidence, supersedes only their collective-layout conclusions,
and does not authorize a GPU launch, model access, data access, or promotion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import eggroll_es_collective_compression_v82b as oracle


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_lora_collective_compression_v82b.json"
)
REPORT = ROOT / (
    "experiments/eggroll_es_hpo/"
    "qwen36_lora_collective_compression_v82b_cpu_evidence_20260717.md"
)
INCIDENT = ROOT / (
    "experiments/eggroll_es_hpo/"
    "qwen36_collective_scope_incident_v82b_20260717.md"
)

SCHEMA_V82B = "qwen36-lora-collective-compression-correction-v82b"
CREATED_AT_UTC_V82B = "2026-07-17T00:00:00+00:00"
ORACLE = ROOT / "eggroll_es_collective_compression_v82b.py"
ORACLE_FILE_SHA256 = (
    "4fef02306ef6519a328ba024e4c4d5eec568695d7a08a80ef0bec8796ffdeb35"
)
MANIFEST = ROOT / (
    "experiments/eggroll_es_hpo/initial_adapters/"
    "matched_lora_initialization_v41a_seed20260715041/"
    "initialization_manifest_v41a.json"
)
MANIFEST_FILE_SHA256 = (
    "a2fa79e6ac06f75743d3fee8f5c0b1aabe6bb83b52b05910ed6460438e2640a2"
)
MANIFEST_CONTENT_SHA256 = (
    "5f885b415302c4e748e19f4d535f1e57ff87f785370f653b345cbdfafda3224b"
)
INITIALIZATION_BUILDER = ROOT / "build_matched_lora_initialization_v41a.py"
INITIALIZATION_BUILDER_SHA256 = (
    "2d1bb1b36d1aec160e6a50972c02c2d667e055e09ec5fd8015c1dff1b523a512"
)
WORKER_V41A = ROOT / "eggroll_es_worker_lora_v41a.py"
WORKER_V41A_SHA256 = (
    "cc40337eba30fe0748996c22dcbf3914b8c12249f6e2e47d6128aadee575494c"
)
WORKER_V72 = ROOT / "eggroll_es_worker_lora_v72.py"
WORKER_V72_SHA256 = (
    "547d525edfd51412abb3a4980ddc4a55730ad0eb09987ec202ce2ce8f701a2c2"
)

V82_COMMIT = "7bfb666c5afa63d199b83de2e8f670f7e7857999"
V82_ARTIFACT = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_collective_compression_v82.json"
)
V82_ARTIFACT_FILE_SHA256 = (
    "ca727884ceda45652a2d3d67677107e3e8582173bd8c96ee556b96bf6521e1f6"
)
V82_ARTIFACT_CONTENT_SHA256 = (
    "42f96cdaf7ada94f1e696aecc7d8254f25ee3e3b4fbce8400ba8218f4abf0cc3"
)
V82_SOURCE_BINDINGS = {
    "eggroll_es_collective_compression_v82.py": (
        "1b205a846dde09da73f5f36477f68e643f0d2f3ae89b16eb7918db46db03022d"
    ),
    "benchmark_eggroll_es_collective_compression_v82.py": (
        "24a2b8d94d79845e54c7608869feca72fe2e5c0dbdf00a46110a1a45f18b3888"
    ),
    "build_qwen36_collective_compression_preregistration_v82.py": (
        "bbef59904bf958f119b4d127abd4bb737004c183a142bf336cd56959c19d48de"
    ),
    "test_eggroll_es_collective_compression_v82.py": (
        "ddf87716f2c0a00ffd7dd7a120f82d87662fec0223009e9a79738fc9c2de2c93"
    ),
    "experiments/eggroll_es_hpo/"
    "qwen36_collective_compression_v82_cpu_evidence_20260717.md": (
        "fb3a0c911eb34f5c7cfff53256e26d7dffcacd2e81a8ae441c41e18671eedfdc"
    ),
}

V75_ARTIFACT = ROOT / (
    "experiments/eggroll_es_hpo/decisions/"
    "qwen36_production_layout_provisional_v75.json"
)
V75_ARTIFACT_FILE_SHA256 = (
    "f960df73f7082b2e5583692c1649b2057f06e012268530eb4529f2bc616a1ebb"
)
V75_ARTIFACT_CONTENT_SHA256 = (
    "5dd23d1effbecec2068d8e21d7f8bf9e5afab85a9e8a58d38a913e835c0e0ed5"
)
V75_BUILDER = ROOT / "build_qwen36_production_layout_decision_v75.py"
V75_BUILDER_FILE_SHA256 = (
    "922c5d15f64591d0ab56eda141a10a02dd4cff78bf82abd6501cc112ab97276f"
)

LEGACY_ELEMENTS = 142_999_552
LEGACY_TENSOR_COUNT = 23
LEGACY_MAX_TENSOR_ELEMENTS = 25_165_824


def file_sha256_v82b(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_v82b(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json_v82b(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    _require_v82b(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _validate_self_hash_v82b(
    value: Mapping[str, Any], expected: str, label: str
) -> None:
    body = copy.deepcopy(dict(value))
    claimed = body.pop("content_sha256_before_self_field", None)
    _require_v82b(
        claimed == expected and oracle.canonical_sha256_v82b(body) == expected,
        f"{label} self hash changed",
    )


def inspect_lora_update_scope_v82b() -> dict[str, Any]:
    bindings = (
        (ORACLE, ORACLE_FILE_SHA256, "V82B oracle"),
        (MANIFEST, MANIFEST_FILE_SHA256, "V41A initialization manifest"),
        (
            INITIALIZATION_BUILDER,
            INITIALIZATION_BUILDER_SHA256,
            "V41A initialization builder",
        ),
        (WORKER_V41A, WORKER_V41A_SHA256, "V41A LoRA worker"),
        (WORKER_V72, WORKER_V72_SHA256, "V72 LoRA worker"),
    )
    for path, expected, label in bindings:
        _require_v82b(
            path.is_file()
            and not path.is_symlink()
            and file_sha256_v82b(path) == expected,
            f"sealed {label} changed",
        )
    manifest = _load_json_v82b(MANIFEST)
    _validate_self_hash_v82b(
        manifest, MANIFEST_CONTENT_SHA256, "V41A initialization manifest"
    )
    surface = manifest.get("surface", {})
    _require_v82b(
        manifest.get("schema") == "matched-lora-initialization-manifest-v41a"
        and manifest.get("dataset_or_training_examples_accessed") is False
        and manifest.get("gpu_accessed") is False
        and manifest.get("shadow_ood_holdout_or_heldout_accessed") is False
        and surface.get("schema") == "matched-lora-peft-surface-v41a"
        and surface.get("sha256") == oracle.EXPECTED_SURFACE_SHA256_V82B
        and surface.get("tensor_count") == oracle.TENSOR_COUNT_V82B
        and surface.get("module_count") == oracle.MODULE_COUNT_V82B
        and surface.get("elements") == oracle.TOTAL_ELEMENTS_V82B
        and manifest.get("tensor_identity", {}).get("ordered_key_sha256")
        == oracle.EXPECTED_ORDERED_KEY_SHA256_V82B,
        "sealed V41A LoRA surface changed",
    )
    ordered_manifest = oracle.validate_ordered_shape_manifest_v82b(
        surface["records"]
    )
    v41_text = WORKER_V41A.read_text(encoding="utf-8")
    v72_text = WORKER_V72.read_text(encoding="utf-8")
    _require_v82b(
        "EXPECTED_TENSOR_COUNT_V41A = 70" in v41_text
        and "EXPECTED_MASTER_ELEMENTS_V41A = 4_528_128" in v41_text
        and "for key, master in self._v41_master.items():" in v72_text
        and "self.inter_pg.all_reduce(" in v72_text
        and "accumulator, out_tensor=accumulator, stream=stream" in v72_text,
        "canonical LoRA update source semantics changed",
    )
    return {
        "schema": "qwen36-canonical-lora-update-scope-v82b",
        "source_bindings": {
            "initialization_manifest": {
                "path": str(MANIFEST.relative_to(ROOT)),
                "file_sha256": MANIFEST_FILE_SHA256,
                "content_sha256": MANIFEST_CONTENT_SHA256,
                "surface_sha256": oracle.EXPECTED_SURFACE_SHA256_V82B,
                "ordered_key_sha256": oracle.EXPECTED_ORDERED_KEY_SHA256_V82B,
            },
            "initialization_builder": {
                "path": INITIALIZATION_BUILDER.name,
                "file_sha256": INITIALIZATION_BUILDER_SHA256,
            },
            "canonical_state_worker": {
                "path": WORKER_V41A.name,
                "file_sha256": WORKER_V41A_SHA256,
            },
            "production_update_worker": {
                "path": WORKER_V72.name,
                "file_sha256": WORKER_V72_SHA256,
                "method": "execute_sharded_adapter_update_v41a",
                "communicator": "self.inter_pg.all_reduce",
            },
        },
        "canonical_master": {
            "dtype": "float32",
            "tensor_count": oracle.TENSOR_COUNT_V82B,
            "module_count": oracle.MODULE_COUNT_V82B,
            "elements": oracle.TOTAL_ELEMENTS_V82B,
            "bytes": oracle.TOTAL_ELEMENTS_V82B * oracle.FP32_BYTES_V82B,
            "ordered_shape_manifest_sha256": oracle.canonical_sha256_v82b(
                ordered_manifest
            ),
            "ordered_shape_manifest": ordered_manifest,
        },
        "update_execution": {
            "native_boundaries": "canonical_70_peft_tensors_in_sorted_key_order",
            "collective_calls_per_actor_per_update": 70,
            "one_fresh_fp32_accumulator_per_tensor": True,
            "accumulator_reduced_in_place": True,
            "reduced_tensor_copied_to_cpu_fp32_before_master_add": True,
            "flat_selected_base_weight_surface_used": False,
        },
    }


def inspect_immutable_incident_lineage_v82b() -> dict[str, Any]:
    _require_v82b(
        file_sha256_v82b(V82_ARTIFACT) == V82_ARTIFACT_FILE_SHA256,
        "immutable V82 artifact changed",
    )
    v82 = _load_json_v82b(V82_ARTIFACT)
    _validate_self_hash_v82b(v82, V82_ARTIFACT_CONTENT_SHA256, "V82 artifact")
    source_rows = []
    for relative, expected in V82_SOURCE_BINDINGS.items():
        path = ROOT / relative
        _require_v82b(
            path.is_file()
            and not path.is_symlink()
            and file_sha256_v82b(path) == expected,
            f"immutable V82 source changed: {relative}",
        )
        source_rows.append({"path": relative, "file_sha256": expected})
    fixed = v82.get("fixed_update_surface", {})
    accounting = v82.get("byte_accounting", {})
    _require_v82b(
        v82.get("schema") == "qwen36-collective-compression-preregistration-v82"
        and fixed.get("native_tensor_count") == LEGACY_TENSOR_COUNT
        and fixed.get("elements_per_rank") == LEGACY_ELEMENTS
        and max(fixed.get("native_tensor_elements", []))
        == LEGACY_MAX_TENSOR_ELEMENTS
        and accounting.get("fp32_control", {}).get(
            "collective_payload_bytes_per_rank"
        )
        == 571_998_208
        and accounting.get("bf16_error_feedback", {}).get(
            "incremental_transaction_peak_bytes_per_rank"
        )
        == 1_194_328_064,
        "V82 wrong-scope evidence no longer matches the incident",
    )
    benchmark_text = (
        ROOT / "benchmark_eggroll_es_collective_compression_v82.py"
    ).read_text(encoding="utf-8")
    _require_v82b(
        "import torch.distributed as dist" in benchmark_text
        and "dist.all_reduce(update)" in benchmark_text
        and "dist.all_reduce(q)" in benchmark_text,
        "V82 benchmark backend mismatch evidence changed",
    )

    _require_v82b(
        file_sha256_v82b(V75_ARTIFACT) == V75_ARTIFACT_FILE_SHA256
        and file_sha256_v82b(V75_BUILDER) == V75_BUILDER_FILE_SHA256,
        "immutable V75 artifact or builder changed",
    )
    v75 = _load_json_v82b(V75_ARTIFACT)
    _validate_self_hash_v82b(v75, V75_ARTIFACT_CONTENT_SHA256, "V75 artifact")
    safe = v75.get("safe_default", {}).get("layout", {}).get(
        "collective_layout", {}
    )
    challenger = v75.get("conditional_fp8_challenger", {}).get(
        "layout", {}
    ).get("collective_layout", {})
    retained = [
        row
        for row in v75.get("retained_choices", [])
        if row.get("dimension") == "collective_parameter_layout"
    ]
    _require_v82b(
        v75.get("schema")
        == "qwen36-memory-efficient-production-layout-decision-v75"
        and v75.get("status") == "provisional_not_final_benchmark_authority"
        and safe.get("parameter_boundaries") == "native_23_tensor"
        and challenger.get("parameter_boundaries") == "native_23_tensor"
        and len(retained) == 1
        and retained[0].get("choice") == "native_23_tensor",
        "V75 collective-layout contamination changed",
    )
    return {
        "schema": "qwen36-collective-scope-incident-lineage-v82b",
        "v82": {
            "commit": V82_COMMIT,
            "artifact": {
                "path": str(V82_ARTIFACT.relative_to(ROOT)),
                "file_sha256": V82_ARTIFACT_FILE_SHA256,
                "content_sha256": V82_ARTIFACT_CONTENT_SHA256,
            },
            "source_bindings": source_rows,
            "immutable_wrong_scope_evidence": True,
            "historical_result_rewritten": False,
            "wrong_surface": {
                "tensor_count": LEGACY_TENSOR_COUNT,
                "elements": LEGACY_ELEMENTS,
                "maximum_tensor_elements": LEGACY_MAX_TENSOR_ELEMENTS,
                "actual_identity": "selected_full_weight_middle_late_surface",
                "not_identity": "canonical_PEFT_LoRA_master",
            },
            "additional_backend_mismatch": {
                "production": "self.inter_pg.all_reduce_PyNccl_style_communicator",
                "prospective_v82_benchmark": "torch.distributed.all_reduce_ProcessGroupNCCL",
                "torch_dtype_capability_evidence_proves_canonical_communicator": False,
            },
            "live_arm_or_promotion_authority": False,
        },
        "v75": {
            "artifact": {
                "path": str(V75_ARTIFACT.relative_to(ROOT)),
                "file_sha256": V75_ARTIFACT_FILE_SHA256,
                "content_sha256": V75_ARTIFACT_CONTENT_SHA256,
                "builder_file_sha256": V75_BUILDER_FILE_SHA256,
            },
            "immutable_artifact_mutated": False,
            "contaminated_fields": [
                "safe_default.layout.collective_layout.parameter_boundaries",
                "conditional_fp8_challenger.layout.collective_layout.parameter_boundaries",
                "retained_choices[collective_parameter_layout]",
            ],
            "collective_layout_claim_superseded": True,
            "superseding_boundary_identity": "canonical_70_peft_tensor",
            "collective_layout_promotable_before_rebuilt_decision": False,
            "consumer_authority_for_collective_layout": False,
            "noncollective_dimensions_re_evaluated_by_v82b": False,
        },
    }


def build_preregistration_v82b() -> dict[str, Any]:
    scope = inspect_lora_update_scope_v82b()
    incident = inspect_immutable_incident_lineage_v82b()
    source_records = [
        {
            "key": row["key"],
            "module": row["module"],
            "role": row["role"],
            "shape": row["shape"],
            "dtype": "torch.float32",
            "elements": row["elements"],
        }
        for row in scope["canonical_master"]["ordered_shape_manifest"]
    ]
    accounting = oracle.collective_byte_accounting_v82b(source_records)
    old = _load_json_v82b(V82_ARTIFACT)["byte_accounting"]
    fp32 = accounting["fp32_control"]
    compressed = accounting["bf16_error_feedback_hypothetical"]
    projection = accounting["nominal_projection"]
    scope_ratio = LEGACY_ELEMENTS / oracle.TOTAL_ELEMENTS_V82B
    _require_v82b(
        fp32["payload_bytes_per_actor_per_update"] == 18_112_512
        and fp32["nominal_ring_bus_bytes_per_actor_per_update"] == 27_168_768
        and compressed["payload_bytes_per_actor_per_update"] == 9_056_256
        and compressed["nominal_ring_bus_bytes_per_actor_per_update"]
        == 13_584_384
        and compressed["transaction_two_residual_banks_bytes_per_actor"]
        == 36_225_024
        and compressed["maximum_bf16_gpu_staging_bytes"] == 524_288
        and compressed["incremental_transaction_peak_gpu_bytes_per_actor"]
        == 36_749_312
        and compressed[
            "fused_prepare_hbm_bytes_per_actor_per_update_lower_bound"
        ]
        == 63_393_792
        and compressed[
            "incremental_local_hbm_bytes_per_actor_per_update_lower_bound_"
            "versus_fp32_excluding_nccl"
        ]
        == 54_337_536
        and projection["ring_bus_bytes_saved_per_actor_per_update"]
        == 13_584_384,
        "V82B corrected byte identities changed",
    )
    v66d = _load_json_v82b(V82_ARTIFACT)["prior_evidence"]["phase_profile"][
        "immutable_v66d_observation"
    ]
    _require_v82b(
        v66d["logical_reduced_fp32_return_bytes_all_actors"]
        == fp32["payload_bytes_per_actor_per_update"] * oracle.WORLD_SIZE_V82B
        and v66d["update_execute_observed_span_seconds"] == 4.202,
        "V66D logical LoRA payload binding changed",
    )
    body = {
        "schema": SCHEMA_V82B,
        "status": (
            "additive_wrong_scope_correction_live_compression_not_materially_"
            "justified_no_gpu_launch"
        ),
        "created_at_utc": CREATED_AT_UTC_V82B,
        "beads": {
            "incident": "specialist-nen.31",
            "superseded_compression_experiment": "specialist-0j5.28",
        },
        "authority": {
            "cpu_file_inspection_only": True,
            "gpu_or_model_launched": False,
            "dataset_training_example_dev_ood_or_holdout_opened": False,
            "adapter_update_or_checkpoint_written": False,
            "live_compression_arm_registered": False,
            "training_checkpoint_layout_or_runtime_promotion_authorized": False,
        },
        "purpose": (
            "Correct V82/V75 collective scope to the exact production LoRA "
            "master and decide whether compression merits any live arm."
        ),
        "oracle": {
            "schema": oracle.SCHEMA_V82B,
            "path": ORACLE.name,
            "file_sha256": ORACLE_FILE_SHA256,
            "imports_torch_or_cuda": False,
        },
        "canonical_lora_update_scope": scope,
        "immutable_incident_lineage": incident,
        "corrected_byte_accounting": accounting,
        "legacy_to_corrected_comparison": {
            "legacy_elements": LEGACY_ELEMENTS,
            "corrected_elements": oracle.TOTAL_ELEMENTS_V82B,
            "legacy_over_corrected_element_ratio": scope_ratio,
            "legacy_tensor_count": LEGACY_TENSOR_COUNT,
            "corrected_tensor_count": oracle.TENSOR_COUNT_V82B,
            "legacy_fp32_payload_bytes_per_actor": old["fp32_control"][
                "collective_payload_bytes_per_rank"
            ],
            "corrected_fp32_payload_bytes_per_actor": fp32[
                "payload_bytes_per_actor_per_update"
            ],
            "legacy_nominal_ring_savings_bytes_per_actor": old[
                "nominal_ring_bus_bytes_saved_per_rank"
            ],
            "corrected_nominal_ring_savings_bytes_per_actor": projection[
                "ring_bus_bytes_saved_per_actor_per_update"
            ],
            "legacy_incremental_transaction_peak_bytes_per_actor": old[
                "bf16_error_feedback"
            ]["incremental_transaction_peak_bytes_per_rank"],
            "corrected_incremental_transaction_peak_bytes_per_actor": compressed[
                "incremental_transaction_peak_gpu_bytes_per_actor"
            ],
            "legacy_maximum_bf16_staging_bytes_per_actor": old[
                "bf16_error_feedback"
            ]["maximum_native_tensor_bf16_staging_bytes_per_rank"],
            "corrected_maximum_bf16_staging_bytes_per_actor": compressed[
                "maximum_bf16_gpu_staging_bytes"
            ],
            "all_legacy_v82_live_and_promotion_conclusions_superseded": True,
        },
        "materiality_reassessment": {
            "v66d_observed_update_execute_span_seconds": v66d[
                "update_execute_observed_span_seconds"
            ],
            "v66d_logical_fp32_payload_all_actors_bytes": v66d[
                "logical_reduced_fp32_return_bytes_all_actors"
            ],
            "v66d_payload_matches_corrected_lora_surface_exactly": True,
            "isolated_unoverlapped_pynccl_collective_time_measured": False,
            "canonical_pynccl_link_bytes_measured": False,
            "canonical_pynccl_hbm_bytes_measured": False,
            "nominal_ring_savings_mib_per_actor_per_update": (
                projection["ring_bus_bytes_saved_per_actor_per_update"] / (1 << 20)
            ),
            "nominal_ring_savings_mib_all_actors_per_update": (
                projection["ring_bus_bytes_saved_all_actors_per_update"] / (1 << 20)
            ),
            "incremental_peak_gpu_mib_per_actor": compressed[
                "incremental_transaction_peak_gpu_bytes_per_actor"
            ]
            / (1 << 20),
            "incremental_local_hbm_mib_per_actor_per_update_lower_bound": compressed[
                "incremental_local_hbm_bytes_per_actor_per_update_lower_bound_"
                "versus_fp32_excluding_nccl"
            ]
            / (1 << 20),
            "local_hbm_increment_to_nominal_ring_savings_ratio": compressed[
                "incremental_local_hbm_bytes_per_actor_per_update_lower_bound_"
                "versus_fp32_excluding_nccl"
            ]
            / projection["ring_bus_bytes_saved_per_actor_per_update"],
            "collective_call_latency_reduced_by_dtype_compression": False,
            "materiality_established": False,
            "current_decision": "retain_exact_fp32_control_no_live_compression_arm",
            "reason": (
                "The corrected payload is 31.58x smaller than V82 assumed; "
                "70 small native calls retain launch latency, and the minimum "
                "incremental local HBM traffic is 4x the nominal ring saving. "
                "No isolated canonical PyNccl time/link/HBM measurement exists."
            ),
        },
        "future_reconsideration_gate": {
            "registered_action": (
                "profile_the_unchanged_fp32_v72_update_only_if_existing_"
                "post_optimization_profiling_dependencies_authorize_it"
            ),
            "profile_schema": oracle.PROFILE_SCHEMA_V82B,
            "profile_seeds": list(oracle.PROFILE_SEEDS_V82B),
            "compression_arm_must_not_execute_during_profile": True,
            "required_measurements": [
                "unoverlapped canonical PyNccl collective seconds",
                "update execute seconds",
                "collective bottleneck rank",
                "link bytes",
                "HBM bytes",
                "all-four-GPU attribution and cleanup",
            ],
            "minimum_median_collective_fraction_of_update": 0.05,
            "minimum_median_perfect_half_payload_speedup_upper_bound_fraction": 0.01,
            "top_three_replicates_minimum": 2,
            "all_thresholds_required_before_implementing_live_compression": True,
            "profile_or_compression_gpu_launch_authorized_by_v82b": False,
            "quality_or_promotion_authorized": False,
        },
        "v75_additive_correction": {
            "immutable_v75_result_rewritten": False,
            "collective_layout_field_superseded": True,
            "native_23_tensor_not_the_lora_update_layout": True,
            "correct_native_layout": "canonical_70_peft_tensor",
            "v75_collective_layout_not_promotable_until_rebuilt": True,
            "v75_noncollective_fields_not_rejudged_here": True,
        },
        "correctness_and_safety": {
            "fp32_control_remains_exact_production_path": True,
            "V82_bit_or_transaction_oracle_authorizes_live_use": False,
            "BF16_support_through_canonical_PyNccl_communicator_proven": False,
            "source_disjoint_semantic_gate_run": False,
            "protected_ood_gate_run": False,
            "promotion_authorized": False,
        },
    }
    return {
        **body,
        "content_sha256_before_self_field": oracle.canonical_sha256_v82b(body),
    }


def validate_preregistration_v82b(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = build_preregistration_v82b()
    _require_v82b(dict(value) == expected, "V82B correction contract changed")
    return copy.deepcopy(expected)


def render_json_v82b(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"


def render_report_v82b(value: Mapping[str, Any]) -> str:
    accounting = value["corrected_byte_accounting"]
    fp32 = accounting["fp32_control"]
    bf16 = accounting["bf16_error_feedback_hypothetical"]
    projection = accounting["nominal_projection"]
    comparison = value["legacy_to_corrected_comparison"]
    materiality = value["materiality_reassessment"]
    manifest = value["canonical_lora_update_scope"]["canonical_master"]
    return f"""# Qwen3.6 LoRA collective compression correction V82B

## Decision

Do not launch or implement a live compression arm from V82.  The exact V72
production update reduces 70 canonical PEFT FP32 tensors totaling 4,528,128
elements.  V82 modeled a different 23-tensor, 142,999,552-element selected
base-weight surface, overestimating the update surface by
{comparison['legacy_over_corrected_element_ratio']:.2f}x.  Commit `{V82_COMMIT}`
remains immutable wrong-scope evidence; none of its live or promotion
conclusions are carried forward.

The corrected nominal ring saving is only
{materiality['nominal_ring_savings_mib_per_actor_per_update']:.3f} MiB per
actor per update ({materiality['nominal_ring_savings_mib_all_actors_per_update']:.3f}
MiB across four actors).  A transactional BF16 error-feedback design adds
{materiality['incremental_peak_gpu_mib_per_actor']:.3f} MiB peak GPU state and
at least {materiality['incremental_local_hbm_mib_per_actor_per_update_lower_bound']:.3f}
MiB local HBM traffic per actor, excluding NCCL internals.  That HBM increment
is {materiality['local_hbm_increment_to_nominal_ring_savings_ratio']:.1f}x the
nominal ring-byte saving.  Dtype compression also leaves all 70 collective
launches intact.  No isolated canonical PyNccl collective time, link bytes, or
HBM bytes have been measured, so materiality is not established.

## Exact source-bound surface

- Initialization manifest: `{MANIFEST_FILE_SHA256}`
- Manifest content: `{MANIFEST_CONTENT_SHA256}`
- Surface identity: `{oracle.EXPECTED_SURFACE_SHA256_V82B}`
- Ordered-key identity: `{oracle.EXPECTED_ORDERED_KEY_SHA256_V82B}`
- Ordered-shape identity: `{manifest['ordered_shape_manifest_sha256']}`
- V41A worker: `{WORKER_V41A_SHA256}`
- V72 production worker: `{WORKER_V72_SHA256}`
- Tensor/module/element count: {manifest['tensor_count']} / {manifest['module_count']} / {manifest['elements']:,}

The full ordered 70-row key/shape manifest is embedded in the machine-readable
preregistration.  The V72 method `execute_sharded_adapter_update_v41a` loops
`self._v41_master.items()`, creates one same-shape FP32 accumulator, calls
`self.inter_pg.all_reduce` in place, and copies each reduced delta to CPU FP32.

## Corrected byte formulas

For `E = 4,528,128`, world size `N = 4`, FP32 width 4, and BF16 width 2:

| Quantity | Formula | Bytes | MiB |
|---|---|---:|---:|
| FP32 payload / actor | `4E` | {fp32['payload_bytes_per_actor_per_update']:,} | {fp32['payload_bytes_per_actor_per_update']/(1<<20):.3f} |
| BF16 payload / actor | `2E` | {bf16['payload_bytes_per_actor_per_update']:,} | {bf16['payload_bytes_per_actor_per_update']/(1<<20):.3f} |
| FP32 nominal ring / actor | `4E * 2(N-1)/N = 6E` | {fp32['nominal_ring_bus_bytes_per_actor_per_update']:,} | {fp32['nominal_ring_bus_bytes_per_actor_per_update']/(1<<20):.3f} |
| BF16 nominal ring / actor | `2E * 2(N-1)/N = 3E` | {bf16['nominal_ring_bus_bytes_per_actor_per_update']:,} | {bf16['nominal_ring_bus_bytes_per_actor_per_update']/(1<<20):.3f} |
| Nominal ring saving / actor | `3E` | {projection['ring_bus_bytes_saved_per_actor_per_update']:,} | {projection['ring_bus_bytes_saved_per_actor_per_update']/(1<<20):.3f} |
| Two FP32 residual banks | `8E` | {bf16['transaction_two_residual_banks_bytes_per_actor']:,} | {bf16['transaction_two_residual_banks_bytes_per_actor']/(1<<20):.3f} |
| Maximum BF16 staging | `2 * 262,144` | {bf16['maximum_bf16_gpu_staging_bytes']:,} | {bf16['maximum_bf16_gpu_staging_bytes']/(1<<20):.3f} |
| Incremental transaction peak | `8E + max_stage` | {bf16['incremental_transaction_peak_gpu_bytes_per_actor']:,} | {bf16['incremental_transaction_peak_gpu_bytes_per_actor']/(1<<20):.3f} |
| Fused prepare HBM lower bound | `read update 4E + read residual 4E + write residual 4E + write BF16 2E = 14E` | {bf16['fused_prepare_hbm_bytes_per_actor_per_update_lower_bound']:,} | {bf16['fused_prepare_hbm_bytes_per_actor_per_update_lower_bound']/(1<<20):.3f} |
| Incremental local HBM vs FP32 | `14E + BF16 D2H read 2E - FP32 D2H read 4E = 12E` | {bf16['incremental_local_hbm_bytes_per_actor_per_update_lower_bound_versus_fp32_excluding_nccl']:,} | {bf16['incremental_local_hbm_bytes_per_actor_per_update_lower_bound_versus_fp32_excluding_nccl']/(1<<20):.3f} |

Ring bytes are a projection, not a measurement of the canonical communicator;
NCCL-internal HBM traffic is deliberately excluded.

## V75 correction and next gate

V75 remains immutable and was already provisional.  Its safe-default,
conditional-FP8, and retained-choice `native_23_tensor` collective fields are
now explicitly superseded.  They are not promotable until V75 is rebuilt with
the canonical 70-tensor PEFT layout.  V82B does not rejudge V75's unrelated
precision, cache, scheduling, or residency fields.

Only an unchanged-FP32 V72 profile may reopen the question.  Across the three
registered seeds it must measure exact canonical PyNccl collective time, link
bytes, HBM bytes, attribution, and cleanup; the collective must be at least 5%
of update execution, have at least a 1% perfect-half-payload speedup upper
bound, and rank in the top three in at least two replicates.  V82B authorizes
neither that GPU profile nor a compression arm by itself.

Dataset/protected/model/GPU access by this builder: none.  Semantic/OOD gates
were not run, and promotion is false.

- V82B content SHA-256: `{value['content_sha256_before_self_field']}`
"""


def render_incident_v82b(value: Mapping[str, Any]) -> str:
    comparison = value["legacy_to_corrected_comparison"]
    incident = value["immutable_incident_lineage"]
    return f"""# Incident: collective update surface scope contamination

## Summary

V82 commit `{V82_COMMIT}` used the 23-tensor selected base-weight surface
({comparison['legacy_elements']:,} elements) as though it were the production
LoRA update.  The canonical V72 path actually reduces 70 FP32 PEFT master
tensors ({comparison['corrected_elements']:,} elements).  V82 is retained
unchanged as wrong-scope evidence; this additive V82B artifact supersedes its
collective byte, VRAM, HBM, benchmark, live-arm, and promotion conclusions.

The V82 prospective benchmark also used `torch.distributed.all_reduce`, while
production calls `self.inter_pg.all_reduce`.  Its bound ProcessGroupNCCL dtype
inspection therefore does not prove BF16 support or performance through the
canonical PyNccl-style communicator.

## Downstream impact

V75 artifact `{incident['v75']['artifact']['content_sha256']}` copied V68's
`native_23_tensor` choice into both layouts and its retained-choice list.  V75
is not modified, and its noncollective decisions are not assessed here.  Its
collective-layout field is superseded and cannot be promoted until a rebuilt
decision binds the canonical 70-tensor PEFT manifest.

## Containment

- `specialist-nen.31` records this incident and blocks `specialist-0j5.28`.
- No V82 compression code or receipt may authorize a GPU run or promotion.
- Exact FP32 V72 remains the only authorized update path.
- A future compression implementation is not registered unless a separately
  authorized, data-free, unchanged-FP32 canonical profile first passes the
  prospective V82B materiality thresholds.
- No dataset, protected evaluation content, model, or GPU was opened for this
  correction.

V82B content SHA-256: `{value['content_sha256_before_self_field']}`
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    value = build_preregistration_v82b()
    rendered = render_json_v82b(value)
    report = render_report_v82b(value)
    incident = render_incident_v82b(value)
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(rendered, encoding="ascii")
        REPORT.write_text(report, encoding="ascii")
        INCIDENT.write_text(incident, encoding="ascii")
    if args.check:
        _require_v82b(
            OUTPUT.read_text(encoding="ascii") == rendered,
            "V82B preregistration is stale",
        )
        _require_v82b(
            REPORT.read_text(encoding="ascii") == report,
            "V82B report is stale",
        )
        _require_v82b(
            INCIDENT.read_text(encoding="ascii") == incident,
            "V82B incident note is stale",
        )
    if not args.write and not args.check:
        print(rendered, end="")
    else:
        print(
            json.dumps(
                {
                    "output": str(OUTPUT),
                    "report": str(REPORT),
                    "incident": str(INCIDENT),
                    "content_sha256": value[
                        "content_sha256_before_self_field"
                    ],
                    "live_compression_arm_registered": False,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
