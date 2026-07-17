#!/usr/bin/env python3
"""Build the CPU-only fused structured-runtime preregistration V72."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import eggroll_es_fused_structured_runtime_v72 as fused
import structured_es_oracle_v1 as oracle


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_fused_structured_runtime_v72.json"
)
SCHEMA = "qwen36-fused-structured-runtime-preregistration-v72"

STRUCTURED_PREREG = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "structured_es_lora_comparison_v1.json"
)
V71_AUDIT_PREREG = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_exact_audit_traffic_v71.json"
)
V61_TOPOLOGY_REPORT = ROOT / (
    "experiments/eggroll_es_hpo/runs/"
    "v61a_v434_train_only_baseline_census/baseline_census_report_v61a.json"
)
ORACLE = ROOT / "structured_es_oracle_v1.py"
V71_AUDIT = ROOT / "eggroll_es_audit_contract_v71.py"
IMPLEMENTATION = ROOT / "eggroll_es_fused_structured_runtime_v72.py"

EXPECTED_FILES = {
    STRUCTURED_PREREG: (
        "1daa8372cef13613736534b6539eceea112616e32abbf0fa7ec30b457b3aeb4b"
    ),
    V71_AUDIT_PREREG: (
        "8747e9ca3c022b593bdfcf445881106d5410c3496f0135bcd2a663f07ca55240"
    ),
    V61_TOPOLOGY_REPORT: (
        "89aa6b70b6150cc5abafa6ebddaffebf1751fb6001c136fdd0dd40dd29ad2878"
    ),
    ORACLE: fused.EXPECTED_ORACLE_FILE_SHA256_V72,
    V71_AUDIT: fused.EXPECTED_V71_AUDIT_FILE_SHA256_V72,
}
EXPECTED_CONTENT = {
    STRUCTURED_PREREG: (
        "de119d12099c381c299ff6de7484d882e805661c7223205943204d19e2e0b405"
    ),
    V71_AUDIT_PREREG: (
        "14c7afe2fd370798a26641f6950e92592be67e5b2e2e5fabfc442b76462c2f99"
    ),
}


def file_sha256_v72(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_v72(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"v72 JSON source is not an object: {path}")
    return value


def _load_self_hashed_v72(path: Path, schema: str) -> dict:
    value = _load_json_v72(path)
    compact = dict(value)
    claimed = compact.pop("content_sha256_before_self_field", None)
    if (
        value.get("schema") != schema
        or claimed != EXPECTED_CONTENT[path]
        or fused.canonical_sha256_v72(compact) != claimed
    ):
        raise RuntimeError(f"v72 sealed contract changed: {path}")
    return value


def _production_projection_v72() -> tuple[dict, dict]:
    report = _load_json_v72(V61_TOPOLOGY_REPORT)
    installations = report.get("installations")
    if not isinstance(installations, list) or len(installations) != 4:
        raise RuntimeError("v72 four-GPU topology installation coverage changed")
    assignment_sha = {
        item.get("assignment_sha256") for item in installations
        if isinstance(item, Mapping)
    }
    assignment_values = [item.get("assignments") for item in installations]
    materializations = [item.get("materialization") for item in installations]
    if (
        assignment_sha
        != {"bac008805d7fc7c6279c47255d8d1563b0be978cb21109e8c013114f143e09df"}
        or any(value != assignment_values[0] for value in assignment_values[1:])
        or any(value != materializations[0] for value in materializations[1:])
    ):
        raise RuntimeError("v72 production runtime topology differs across GPUs")
    materialization = materializations[0]
    if (
        not isinstance(materialization, Mapping)
        or materialization.get("source_tensor_count") != 70
        or materialization.get("source_elements") != 4_528_128
        or materialization.get("runtime_view_count") != 82
        or materialization.get("runtime_elements") != 4_921_344
        or materialization.get("runtime_dtype") != "torch.bfloat16"
        or materialization.get("b_scale") != 2.0
    ):
        raise RuntimeError("v72 production materialization shape changed")
    manifest = fused.build_runtime_projection_manifest_v72(
        assignment_values[0], b_scale=2.0
    )
    fused.validate_runtime_projection_manifest_v72(
        manifest, require_production_shape=True
    )
    return manifest, {
        "path": str(V61_TOPOLOGY_REPORT),
        "file_sha256": EXPECTED_FILES[V61_TOPOLOGY_REPORT],
        "four_physical_gpu_installations": True,
        "assignment_sha256": next(iter(assignment_sha)),
        "assignments_identical_across_gpus": True,
        "source_or_runtime_tensor_payload_opened": False,
    }


def _byte_ledgers_v72(manifest: Mapping[str, Any]) -> dict[str, dict]:
    result = {
        "iid_absolute_index": fused.production_byte_ledger_v72(
            manifest,
            method="iid_absolute_index",
            structured_rank=None,
        )
    }
    for rank in oracle.STRUCTURED_RANKS_V1:
        result[f"structured_rank_{rank}"] = fused.production_byte_ledger_v72(
            manifest,
            method="structured_outer_product",
            structured_rank=rank,
        )
    return result


def build_preregistration_v72() -> dict:
    for path, expected in EXPECTED_FILES.items():
        observed = file_sha256_v72(path)
        if observed != expected:
            raise RuntimeError(
                f"v72 sealed source changed: {path}: {observed} != {expected}"
            )
    structured = _load_self_hashed_v72(
        STRUCTURED_PREREG, oracle.SCHEMA_V1
    )
    v71 = _load_self_hashed_v72(
        V71_AUDIT_PREREG, "eggroll-es-exact-audit-traffic-contract-v71"
    )
    if (
        structured["authorization"]["gpu_launch"] is not False
        or structured["authorization"]["protected_holdout"] is not False
        or v71["rules"]["unknown_or_partial_rpc_requires_exact_restore_or_poison"]
        is not True
        or v71["rules"]["update_acceptance_after_exact_boundary"] is not True
    ):
        raise RuntimeError("v72 source safety authority changed")
    manifest, topology_source = _production_projection_v72()
    ledgers = _byte_ledgers_v72(manifest)
    compact = {
        "schema": SCHEMA,
        "status": "sealed_cpu_oracle_production_cuda_integration_pending",
        "bead": "specialist-0j5.18",
        "purpose": (
            "Fuse absolute-index IID/rank-k structured perturbation generation "
            "with bounded direct writes into the production 82-view BF16 LoRA "
            "layout and stream weighted updates into a rollback-owned FP32 master."
        ),
        "authority": {
            "cpu_oracle_and_fault_tests": True,
            "gpu_launch": False,
            "dataset_or_protected_content_access": False,
            "training_or_scored_evaluation": False,
            "candidate_or_update_promotion": False,
        },
        "implementation_bindings": {
            "implementation": {
                "path": str(IMPLEMENTATION),
                "file_sha256": file_sha256_v72(IMPLEMENTATION),
                "schema": fused.SCHEMA_V72,
            },
            "structured_es_oracle": {
                "path": str(ORACLE),
                "file_sha256": EXPECTED_FILES[ORACLE],
                "rng_algorithm": oracle.RNG_ALGORITHM_V1,
            },
            "v71_audit_contract": {
                "path": str(V71_AUDIT),
                "file_sha256": EXPECTED_FILES[V71_AUDIT],
                "schema": "eggroll-es-exact-audit-traffic-contract-v71",
            },
            "structured_preregistration": {
                "path": str(STRUCTURED_PREREG),
                "file_sha256": EXPECTED_FILES[STRUCTURED_PREREG],
                "content_sha256": EXPECTED_CONTENT[STRUCTURED_PREREG],
            },
            "v71_audit_preregistration": {
                "path": str(V71_AUDIT_PREREG),
                "file_sha256": EXPECTED_FILES[V71_AUDIT_PREREG],
                "content_sha256": EXPECTED_CONTENT[V71_AUDIT_PREREG],
            },
            "production_topology": topology_source,
        },
        "production_projection": {
            "manifest": manifest,
            "source_tensor_count": 70,
            "source_elements": 4_528_128,
            "runtime_view_count": 82,
            "runtime_elements": 4_921_344,
            "runtime_bf16_bytes": 9_842_688,
            "packed_a_full_duplication": True,
            "packed_b_disjoint_split_and_scale_2": True,
        },
        "sealed_counter_schedule": {
            "algorithm": oracle.RNG_ALGORITHM_V1,
            "counter_tuple": [
                "direction_seed",
                "full_canonical_source_tensor_key",
                "iid_element_or_structured_left_right_rank_domain",
                "tensor_local_absolute_element_or_factor_ordinal",
            ],
            "global_source_traversal": "source keys ascending then local absolute index",
            "shard_range": "floor(total*rank/world_size) half-open",
            "chunk_order_independent": True,
            "seed_coefficient_reduction_order": "direction seed ascending",
            "structured_component_reduction_order": "rank component ascending FP32",
            "overlap_gap_or_world_size_change": "fail_closed",
        },
        "direct_runtime_candidate_lifecycle": {
            "candidate_begin": [
                "V71 cached immutable-master invariant",
                "V71 runtime object/storage/version invariant",
            ],
            "materialize": (
                "generate one absolute-index FP32 chunk and write every intersecting "
                "packed BF16 runtime span directly"
            ),
            "pre_generation": (
                "coverage plus V71 object/storage/version checks; a fail-closed "
                "content sentinel is rebound so an early exact audit rejects; "
                "reward remains provisional; no D2H"
            ),
            "post_generation": (
                "one exact BF16 readback validates streamed per-chunk digests and "
                "rebinds the V71 runtime registry"
            ),
            "restore": (
                "stream immutable FP32 master into all runtime spans, exact-audit "
                "master and BF16 runtime once, otherwise terminal poison"
            ),
            "unknown_or_partial_rpc": "exact full restore or terminal poison",
        },
        "streamed_update_lifecycle": {
            "precondition": "exact V71 update_acceptance_sha256",
            "reduction": (
                "fixed seed-ascending FP32 accumulation into bounded chunks"
            ),
            "output": (
                "one full FP32 pending master owned by the transaction; this is "
                "persistent rollback/commit output, not scratch"
            ),
            "commit": "provisional until exact V71 commit boundary",
            "final": "rollback retained until exact V71 final boundary",
            "unknown_or_partial_rpc": (
                "discard pending output and exact-audit original, or terminal poison"
            ),
            "final_update_maximum_ulp_vs_cpu_oracle": 2,
        },
        "production_byte_ledgers": ledgers,
        "registered_cpu_acceptance": {
            "full_chunk_shard_reconstruction": "exact for IID and ranks 1/4/8/16",
            "packed_projection": "exact A duplication and B split/scale",
            "candidate_runtime_values": "exact BF16 versus oracle",
            "update_final_maximum_ulp": 2,
            "overlap_and_gap_rejected": True,
            "partial_candidate_restore_or_poison": True,
            "partial_update_original_exact_or_poison": True,
            "commit_and_final_boundaries_required": True,
            "whole_surface_noise_candidate_or_update_allocations": 0,
        },
        "promotion_blockers": [
            "CUDA absolute-index generator is not implemented or attested",
            "direct GPU runtime-view writer is not integrated into the V71 worker",
            "four-GPU candidate/update/restore receipts are absent",
            "direct H2D/D2H and HBM traffic counters are absent",
            "paired throughput and phase-time improvement is unmeasured",
            "Qwen reward/semantic/OOD equivalence is unmeasured",
        ],
        "required_live_experiment": {
            "physical_gpus": [0, 1, 2, 3],
            "paired_replicates_minimum": 3,
            "candidate_count_per_update": 16,
            "methods": [
                "iid_absolute_index",
                "structured_rank_1",
                "structured_rank_4",
                "structured_rank_8",
                "structured_rank_16",
            ],
            "must_bind_v71_audit_schedule_unchanged": True,
            "must_report": [
                "candidate materialize/restore/update elapsed time",
                "H2D, D2H, HBM, scratch, and peak VRAM bytes",
                "per-GPU useful work and cleanup idle",
                "exact candidate/restore and <=2 ULP update certificates",
                "paired reward, semantic, and OOD noninferiority",
            ],
            "promotion_authorized_by_preregistration": False,
        },
    }
    return {
        **compact,
        "content_sha256_before_self_field": fused.canonical_sha256_v72(compact),
    }


def validate_preregistration_v72(
    value: Mapping[str, Any] | None = None,
    *,
    launch: bool = False,
) -> dict:
    expected = build_preregistration_v72()
    observed = _load_json_v72(OUTPUT) if value is None else dict(value)
    if observed != expected:
        raise RuntimeError("v72 fused structured-runtime preregistration changed")
    if launch:
        raise RuntimeError(
            "v72 production CUDA integration/live receipts are pending; launch forbidden"
        )
    return expected


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args(argv)
    value = build_preregistration_v72()
    if args.launch:
        validate_preregistration_v72(value, launch=True)
    if args.check:
        validate_preregistration_v72()
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
