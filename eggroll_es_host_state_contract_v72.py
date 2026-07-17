#!/usr/bin/env python3
"""CPU contract for canonical LoRA host-state ownership and publication V72."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from eggroll_es_audit_contract_v71 import OwnedMasterIdentityCacheV71


SCHEMA_V72 = "eggroll-es-versioned-host-state-contract-v72"
MASTER_BYTES_V72 = 18_112_512
MASTER_ELEMENTS_V72 = 4_528_128
MASTER_TENSORS_V72 = 70
MAX_MASTER_TENSOR_BYTES_V72 = 1_048_576
WORLD_SIZE_V72 = 4
DENSE_FP32_MASTER_BYTES_PER_ACTOR_V72 = 143_807_290_816


def canonical_sha256_v72(value) -> str:
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


class ImmutableStateLeaseV72:
    """Retain one owned tensor map by reference with exact drift detection."""

    def __init__(self, role, generation, tensors, identity):
        role = str(role)
        if not role:
            raise ValueError("v72 state lease role is empty")
        if type(generation) is not int or generation < 0:
            raise ValueError("v72 state lease generation is invalid")
        self.role = role
        self.generation = generation
        self.tensors = tensors
        self.identity = identity
        self.cache = OwnedMasterIdentityCacheV71(tensors, identity=identity)

    def cheap_certificate(self, phase):
        cached = self.cache.cached_identity(self.tensors, phase)
        if cached["sha256"] != self.identity["sha256"]:
            raise RuntimeError("v72 leased state identity changed")
        return {
            "schema": "eggroll-es-immutable-state-lease-v72",
            "role": self.role,
            "generation": self.generation,
            "identity_sha256": self.identity["sha256"],
            "tensor_count": self.identity["tensor_count"],
            "elements": self.identity["elements"],
            "bytes": self.identity["bytes"],
            "aliases_owned_mapping": True,
            "ownership_clone_bytes": 0,
            "validation_clone_bytes": 0,
            "cheap_invariant": cached["cheap_invariant"],
        }

    def exact_certificate(self, boundary):
        exact = self.cache.exact_audit(self.tensors, boundary)
        if exact["sha256"] != self.identity["sha256"]:
            raise RuntimeError("v72 leased state exact identity changed")
        return {
            "schema": "eggroll-es-exact-immutable-state-lease-v72",
            "role": self.role,
            "generation": self.generation,
            "identity_sha256": self.identity["sha256"],
            "bytes": self.identity["bytes"],
            "ownership_clone_bytes": 0,
            "validation_clone_bytes": 0,
            "exact_content_audit": exact["exact_content_audit"],
        }


def _phase_bytes_v72(multiplier):
    return int(multiplier * MASTER_BYTES_V72)


def state_residency_account_v72(world_size=WORLD_SIZE_V72):
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("v72 world size must be positive")
    baseline = {
        "quiescent_after_install": _phase_bytes_v72(2),
        "prepared_persistent": _phase_bytes_v72(3),
        "executed_persistent": _phase_bytes_v72(4),
        "execute_validation_peak": _phase_bytes_v72(5),
        "commit_clone_peak": _phase_bytes_v72(7),
        "committed_rollback_retained": _phase_bytes_v72(3),
        "final_quiescent": _phase_bytes_v72(2),
        "checkpoint_tensor_residency_peak": _phase_bytes_v72(4),
    }
    proposed = {
        "quiescent_after_install": _phase_bytes_v72(1),
        "prepared_persistent": _phase_bytes_v72(1),
        "executed_persistent": _phase_bytes_v72(2),
        "execute_validation_peak": _phase_bytes_v72(2),
        "commit_clone_peak": _phase_bytes_v72(2),
        "committed_rollback_retained": _phase_bytes_v72(2),
        "final_quiescent": _phase_bytes_v72(1),
        "checkpoint_tensor_residency_peak": (
            MASTER_BYTES_V72 + MAX_MASTER_TENSOR_BYTES_V72
        ),
    }
    baseline_peak = max(baseline.values())
    proposed_peak = max(proposed.values())
    clone_passes = {
        "install": {"baseline": 8, "proposed": 1},
        "reference_capture": {"baseline": 3, "proposed": 0},
        "prepare": {"baseline": 2, "proposed": 0},
        "execute": {"baseline": 2, "proposed": 0},
        "commit": {"baseline": 7, "proposed": 0},
        "final": {"baseline": 0, "proposed": 0},
        "checkpoint": {"baseline": 4, "proposed": 0},
    }
    baseline_clone_passes = sum(
        item["baseline"] for item in clone_passes.values()
    )
    proposed_clone_passes = sum(
        item["proposed"] for item in clone_passes.values()
    )
    per_actor_saved = (
        baseline_clone_passes - proposed_clone_passes
    ) * MASTER_BYTES_V72
    result = {
        "schema": "eggroll-es-host-state-byte-rss-account-v72",
        "world_size": world_size,
        "master": {
            "tensor_count": MASTER_TENSORS_V72,
            "elements": MASTER_ELEMENTS_V72,
            "bytes": MASTER_BYTES_V72,
            "max_tensor_bytes": MAX_MASTER_TENSOR_BYTES_V72,
        },
        "logical_tensor_residency_bytes_per_actor": {
            "baseline": baseline,
            "proposed": proposed,
        },
        "peak": {
            "baseline_bytes_per_actor": baseline_peak,
            "proposed_bytes_per_actor": proposed_peak,
            "saved_bytes_per_actor": baseline_peak - proposed_peak,
            "saved_fraction": (
                (baseline_peak - proposed_peak) / baseline_peak
            ),
            "aggregate_actor_rss_sum_saved_bytes": (
                (baseline_peak - proposed_peak) * world_size
            ),
        },
        "steady_quiescent": {
            "baseline_bytes_per_actor": baseline["final_quiescent"],
            "proposed_bytes_per_actor": proposed["final_quiescent"],
            "saved_bytes_per_actor": MASTER_BYTES_V72,
            "aggregate_actor_rss_sum_saved_bytes": (
                MASTER_BYTES_V72 * world_size
            ),
        },
        "full_state_tensor_copy_passes_one_commit_lifecycle": {
            "by_phase": clone_passes,
            "baseline": baseline_clone_passes,
            "proposed": proposed_clone_passes,
            "saved": baseline_clone_passes - proposed_clone_passes,
            "saved_copy_bytes_per_actor": per_actor_saved,
            "saved_copy_bytes_all_actors": per_actor_saved * world_size,
        },
        "not_counted": [
            "hash serialization bytes",
            "safetensors page-cache residency",
            "per-tensor reduced-update staging",
            "GPU model, KV cache, and runtime LoRA buffers",
        ],
    }
    result["content_sha256_before_self_field"] = canonical_sha256_v72(result)
    return result


def build_contract_v72(world_size=WORLD_SIZE_V72):
    residency = state_residency_account_v72(world_size)
    result = {
        "schema": SCHEMA_V72,
        "status": "cpu_contract_no_gpu_or_dataset_or_protected_access",
        "world_size": world_size,
        "rules": {
            "reference_is_identity_only": True,
            "prepared_rollback_aliases_immutable_master": True,
            "executed_candidate_is_second_owned_generation": True,
            "commit_moves_candidate_ownership_without_clone": True,
            "old_master_retained_until_exact_final": True,
            "snapshot_writes_immutable_master_without_clone": True,
            "snapshot_streams_one_tensor_for_readback": True,
            "snapshot_directory_publication_is_atomic": True,
            "partial_generation_cannot_be_published_or_adopted": True,
            "unknown_state_requires_exact_restore_or_poison": True,
        },
        "residency": residency,
        "pinning_decision": {
            "selected": False,
            "reason": (
                "the runtime consumes converted/split BF16 views rather than "
                "the FP32 master directly; CPU-only evidence cannot prove a "
                "locked-memory or asynchronous-copy benefit"
            ),
            "future_live_arm": (
                "one NUMA-local reusable BF16 staging buffer per actor, with "
                "event-proved completion and a measured locked-memory cap"
            ),
        },
        "dense_full_weight_decision": {
            "implemented": False,
            "fp32_master_bytes_per_actor": DENSE_FP32_MASTER_BYTES_PER_ACTOR_V72,
            "four_actor_private_copy_bytes": (
                DENSE_FP32_MASTER_BYTES_PER_ACTOR_V72 * world_size
            ),
            "reason": (
                "cross-process mmap/shared-memory ownership, NUMA locality, "
                "page-fault behavior, and partial-generation recovery require "
                "a capacity-qualified live design; they are not inferred from "
                "the safe canonical-LoRA ownership proof"
            ),
        },
        "gpu_launch_performed": False,
        "dataset_or_protected_access_performed": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256_v72(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the CPU-only V72 host-state ownership contract."
    )
    parser.add_argument("--world-size", type=int, default=WORLD_SIZE_V72)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    value = build_contract_v72(args.world_size)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return value


if __name__ == "__main__":
    main()
