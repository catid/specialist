#!/usr/bin/env python3
"""CPU contract for lower-traffic exact LoRA/base audits.

V71 keeps every state transition fail closed while distinguishing cheap
object/storage/version invariants from content audits that necessarily move
bytes to the host.  Rewards remain provisional until the population boundary
audit, and update/commit/final acceptance each has an exact trust boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


SCHEMA_V71 = "eggroll-es-exact-audit-traffic-contract-v71"
MASTER_BYTES_V71 = 18_112_512
RUNTIME_LORA_BYTES_V71 = 9_842_688
BASE_BYTES_V71 = 285_999_104
CHEAP_EDGES_V71 = (
    "materialize",
    "pre_generation",
    "post_generation",
    "restore",
)
EXACT_BOUNDARIES_V71 = (
    "population_reward_acceptance",
    "update_acceptance",
    "commit",
    "final",
)
CHECKPOINT_BOUNDARY_V71 = "checkpoint"


def canonical_sha256_v71(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def tensor_sha256_v71(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _tensor_metadata_v71(tensor: torch.Tensor) -> dict:
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError("v71 invariant target is not a tensor")
    return {
        "object_id": id(tensor),
        "storage_data_ptr": int(tensor.untyped_storage().data_ptr()),
        "storage_offset": int(tensor.storage_offset()),
        "storage_bytes": int(tensor.untyped_storage().nbytes()),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "elements": int(tensor.numel()),
        "element_size": int(tensor.element_size()),
        "version": int(tensor._version),
    }


class TensorInvariantRegistryV71:
    """Track identity cheaply and content exactly at explicit boundaries."""

    def __init__(
        self,
        label: str,
        tensors: dict[str, torch.Tensor],
        *,
        precomputed_sha256: dict[str, str] | None = None,
    ):
        if not isinstance(label, str) or not label:
            raise ValueError("v71 invariant label is empty")
        if not isinstance(tensors, dict) or not tensors:
            raise RuntimeError("v71 invariant tensor map is empty")
        self.label = label
        self._mapping_id = id(tensors)
        self._keys = tuple(sorted(tensors))
        if len(self._keys) != len(set(self._keys)):
            raise RuntimeError("v71 invariant keys are duplicated")
        self._metadata = {
            key: _tensor_metadata_v71(tensors[key]) for key in self._keys
        }
        if precomputed_sha256 is None:
            self._sha256 = {
                key: tensor_sha256_v71(tensors[key]) for key in self._keys
            }
        else:
            if set(precomputed_sha256) != set(self._keys):
                raise RuntimeError("v71 precomputed hash coverage changed")
            self._sha256 = {}
            for key in self._keys:
                value = str(precomputed_sha256[key])
                if (
                    len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise RuntimeError("v71 precomputed tensor hash changed")
                self._sha256[key] = value
        self._content_identity = canonical_sha256_v71([
            {
                "key": key,
                "shape": self._metadata[key]["shape"],
                "dtype": self._metadata[key]["dtype"],
                "elements": self._metadata[key]["elements"],
                "sha256": self._sha256[key],
            }
            for key in self._keys
        ])
        self._bytes = sum(
            item["elements"] * item["element_size"]
            for item in self._metadata.values()
        )

    @property
    def content_identity(self):
        return self._content_identity

    @property
    def bytes(self):
        return self._bytes

    def _validate_mapping_v71(self, tensors):
        if (
            not isinstance(tensors, dict)
            or id(tensors) != self._mapping_id
            or tuple(sorted(tensors)) != self._keys
        ):
            raise RuntimeError(f"v71 {self.label} object/key mapping drifted")

    def cheap_certificate(self, tensors, phase: str):
        self._validate_mapping_v71(tensors)
        phase = str(phase)
        if not phase:
            raise ValueError("v71 cheap invariant phase is empty")
        for key in self._keys:
            current = _tensor_metadata_v71(tensors[key])
            if current != self._metadata[key]:
                raise RuntimeError(
                    f"v71 {self.label} object/storage/version drifted at {phase}: {key}"
                )
        return {
            "schema": "eggroll-es-cheap-tensor-invariant-v71",
            "label": self.label,
            "phase": phase,
            "passed": True,
            "tensor_count": len(self._keys),
            "bytes_not_read_back": self._bytes,
            "checks": [
                "object_id", "storage_data_ptr", "storage_offset",
                "storage_bytes", "shape", "stride", "dtype", "device",
                "elements", "element_size", "version",
            ],
            "d2h_bytes": 0,
        }

    def exact_certificate(self, tensors, boundary: str):
        cheap = self.cheap_certificate(tensors, boundary)
        observed = {
            key: tensor_sha256_v71(tensors[key]) for key in self._keys
        }
        if observed != self._sha256:
            changed = [key for key in self._keys if observed[key] != self._sha256[key]]
            raise RuntimeError(
                f"v71 {self.label} exact content drifted at {boundary}: {changed}"
            )
        return {
            "schema": "eggroll-es-exact-tensor-content-audit-v71",
            "label": self.label,
            "boundary": str(boundary),
            "passed": True,
            "tensor_count": len(self._keys),
            "content_identity": self._content_identity,
            "d2h_bytes": self._bytes,
            "cheap_invariant": cheap,
        }

    def rebind_controlled_write(
        self,
        tensors,
        phase: str,
        *,
        expected_sha256: dict[str, str] | None = None,
    ):
        """Advance expected versions after a worker-owned, verified write."""
        self._validate_mapping_v71(tensors)
        metadata = {}
        for key in self._keys:
            current = _tensor_metadata_v71(tensors[key])
            prior = dict(self._metadata[key])
            prior.pop("version")
            version = current.pop("version")
            if current != prior:
                raise RuntimeError(
                    f"v71 {self.label} storage changed during controlled write: {key}"
                )
            current["version"] = version
            metadata[key] = current
        observed = (
            {key: tensor_sha256_v71(tensors[key]) for key in self._keys}
            if expected_sha256 is None else dict(expected_sha256)
        )
        if set(observed) != set(self._keys):
            raise RuntimeError("v71 controlled-write hash coverage changed")
        self._metadata = metadata
        self._sha256 = observed
        self._content_identity = canonical_sha256_v71([
            {
                "key": key,
                "shape": metadata[key]["shape"],
                "dtype": metadata[key]["dtype"],
                "elements": metadata[key]["elements"],
                "sha256": observed[key],
            }
            for key in self._keys
        ])
        return {
            "schema": "eggroll-es-controlled-version-rebind-v71",
            "label": self.label,
            "phase": str(phase),
            "tensor_count": len(self._keys),
            "content_identity": self._content_identity,
        }


class OwnedMasterIdentityCacheV71:
    """Cache an owned immutable FP32 master without validation clones."""

    def __init__(self, tensors: dict[str, torch.Tensor], identity: dict | None = None):
        precomputed = None
        if identity is not None:
            records = identity.get("tensors") if isinstance(identity, dict) else None
            if not isinstance(records, list):
                raise RuntimeError("v71 precomputed master identity changed")
            precomputed = {
                item["key"]: item["sha256"]
                for item in records
                if isinstance(item, dict) and "key" in item and "sha256" in item
            }
            if len(precomputed) != len(records):
                raise RuntimeError("v71 precomputed master records changed")
        self._registry = TensorInvariantRegistryV71(
            "owned_fp32_master",
            tensors,
            precomputed_sha256=precomputed,
        )
        if any(
            tensor.dtype != torch.float32 or tensor.device.type != "cpu"
            for tensor in tensors.values()
        ):
            raise RuntimeError("v71 owned master must be CPU FP32")
        self._identity = {
            "schema": "eggroll-es-owned-fp32-master-identity-v71",
            "sha256": self._registry.content_identity,
            "tensor_count": len(tensors),
            "elements": sum(int(tensor.numel()) for tensor in tensors.values()),
            "bytes": self._registry.bytes,
        }
        if identity is not None:
            if (
                identity.get("sha256") != self._identity["sha256"]
                or int(identity.get("tensor_count", -1))
                != self._identity["tensor_count"]
                or int(identity.get("elements", -1)) != self._identity["elements"]
                or int(identity.get("bytes", -1)) != self._identity["bytes"]
            ):
                raise RuntimeError("v71 precomputed master identity mismatch")
        self._cache_hits = 0

    @property
    def sha256(self):
        return self._identity["sha256"]

    def cached_identity(self, tensors, phase: str):
        cheap = self._registry.cheap_certificate(tensors, phase)
        self._cache_hits += 1
        return {
            **self._identity,
            "cache_hits": self._cache_hits,
            "validation_clone_bytes": 0,
            "cheap_invariant": cheap,
        }

    def exact_audit(self, tensors, boundary: str):
        exact = self._registry.exact_certificate(tensors, boundary)
        return {
            **self._identity,
            "exact_content_audit": exact,
            "validation_clone_bytes": 0,
        }


def canonical_audit_schedule_v71(candidate_count=16):
    if type(candidate_count) is not int or candidate_count <= 0:
        raise ValueError("v71 candidate count must be positive")
    events = []
    for candidate in range(candidate_count):
        for edge in CHEAP_EDGES_V71[:3]:
            events.append({
                "event": "cheap_transition",
                "candidate": candidate,
                "edge": edge,
            })
        events.append({
            "event": "exact_lora",
            "candidate": candidate,
            "edge": "post_generation",
            "single_d2h_readback": True,
        })
        events.append({
            "event": "cheap_transition",
            "candidate": candidate,
            "edge": "restore",
        })
    events.extend([
        {"event": "exact_base", "boundary": "population_reward_acceptance"},
        {"event": "exact_master", "boundary": "population_reward_acceptance"},
        {"event": "reward_acceptance", "candidate_count": candidate_count},
        {"event": "exact_base", "boundary": "update_acceptance"},
        {"event": "exact_master", "boundary": "update_acceptance"},
        {"event": "update_acceptance"},
        {"event": "commit_state_write", "acceptance": "provisional"},
        {"event": "exact_base", "boundary": "commit"},
        {"event": "exact_master", "boundary": "commit"},
        {"event": "exact_lora", "boundary": "commit",
         "single_d2h_readback": True},
        {"event": "commit_acceptance"},
        {"event": "final_state_write", "acceptance": "provisional"},
        {"event": "exact_base", "boundary": "final"},
        {"event": "exact_master", "boundary": "final"},
        {"event": "exact_lora", "boundary": "final",
         "single_d2h_readback": True},
        {"event": "final_acceptance"},
    ])
    return events


def validate_audit_schedule_v71(events, candidate_count=16):
    expected = canonical_audit_schedule_v71(candidate_count)
    if events != expected:
        raise RuntimeError("v71 audit schedule changed or acceptance moved early")
    return {
        "schema": "eggroll-es-audit-schedule-certificate-v71",
        "passed": True,
        "candidate_count": candidate_count,
        "cheap_transition_checks": candidate_count * len(CHEAP_EDGES_V71),
        "per_candidate_exact_lora_checks": candidate_count,
        "exact_base_boundaries": list(EXACT_BOUNDARIES_V71),
        "rewards_provisional_until_population_boundary": True,
        "update_provisional_until_update_boundary": True,
        "commit_and_final_exact": True,
        "schedule_sha256": canonical_sha256_v71(events),
    }


def traffic_account_v71(candidate_count=16, world_size=4):
    """Byte-account the safety-equivalent V41/V66 and proposed V71 paths."""
    if type(candidate_count) is not int or candidate_count <= 0:
        raise ValueError("v71 traffic candidate count must be positive")
    if (
        type(world_size) is not int
        or world_size <= 0
        or candidate_count % world_size != 0
    ):
        raise ValueError(
            "v71 world size must be positive and divide candidate count"
        )
    # One lifecycle contains candidate+restore materializations for every
    # signed candidate and one materialization each at update execution,
    # commit, and finalization.  Each candidate also gets a post-generation
    # LoRA audit; commit and final each add one boundary audit.  The V41-style
    # baseline reads LoRA once for equality and again for SHA.  V71 stages the
    # disjoint views contiguously on device and reads them to host exactly once.
    candidates_per_actor = candidate_count // world_size
    materializations = 2 * candidate_count + 3 * world_size
    logical_lora_verifications = 3 * candidate_count + 5 * world_size

    # V41 hashes the base at every transition.  A safety-equivalent lifecycle
    # therefore hashes it at candidate materialize/post-generation/restore,
    # update execution, both explicit pre-acceptance boundaries, and twice at
    # commit/final (post-write check plus boundary audit).  V71 leaves only the
    # four named trust boundaries as content reads.  Seven current V66 master
    # validations per signed candidate clone the whole FP32 state; update-path
    # ownership clones are deliberately excluded because they are state, not
    # validation, traffic.
    baseline_base_audits = 3 * candidate_count + 7 * world_size
    proposed_base_audits = len(EXACT_BOUNDARIES_V71) * world_size
    baseline = {
        "h2d_bytes": materializations * RUNTIME_LORA_BYTES_V71,
        "lora_d2h_bytes": (
            2 * logical_lora_verifications * RUNTIME_LORA_BYTES_V71
        ),
        "base_d2h_bytes": baseline_base_audits * BASE_BYTES_V71,
        "master_validation_host_copy_bytes": (
            7 * candidate_count * MASTER_BYTES_V71
        ),
        "lora_d2h_calls": 2 * logical_lora_verifications,
        "base_exact_audits": baseline_base_audits,
        "gpu_staging_read_write_bytes": 0,
        "peak_fused_staging_vram_bytes": 0,
        "expected_value_concat_host_copy_bytes": 0,
    }
    proposed = {
        "h2d_bytes": materializations * RUNTIME_LORA_BYTES_V71,
        "lora_d2h_bytes": (
            logical_lora_verifications * RUNTIME_LORA_BYTES_V71
        ),
        "base_d2h_bytes": proposed_base_audits * BASE_BYTES_V71,
        "master_validation_host_copy_bytes": 0,
        "lora_d2h_calls": logical_lora_verifications,
        "base_exact_audits": proposed_base_audits,
        "gpu_staging_read_write_bytes": (
            2 * logical_lora_verifications * RUNTIME_LORA_BYTES_V71
        ),
        "peak_fused_staging_vram_bytes": RUNTIME_LORA_BYTES_V71,
        # Equality is performed segment-by-segment against the existing
        # expected tensors, not against a second concatenated host buffer.
        "expected_value_concat_host_copy_bytes": 0,
    }
    for item in (baseline, proposed):
        item["total_device_transfer_bytes"] = (
            item["h2d_bytes"]
            + item["lora_d2h_bytes"]
            + item["base_d2h_bytes"]
        )
        item["total_host_copy_or_device_transfer_bytes"] = (
            item["total_device_transfer_bytes"]
            + item["master_validation_host_copy_bytes"]
        )
    device_saved = (
        baseline["total_device_transfer_bytes"]
        - proposed["total_device_transfer_bytes"]
    )
    all_saved = (
        baseline["total_host_copy_or_device_transfer_bytes"]
        - proposed["total_host_copy_or_device_transfer_bytes"]
    )
    result = {
        "schema": "eggroll-es-audit-byte-account-v71",
        "candidate_count": candidate_count,
        "world_size": world_size,
        "candidates_per_actor": candidates_per_actor,
        "constants": {
            "master_bytes": MASTER_BYTES_V71,
            "runtime_lora_bytes": RUNTIME_LORA_BYTES_V71,
            "base_bytes": BASE_BYTES_V71,
        },
        "baseline": baseline,
        "proposed": proposed,
        "savings": {
            "device_transfer_bytes": device_saved,
            "device_transfer_fraction": (
                device_saved / baseline["total_device_transfer_bytes"]
            ),
            "host_copy_or_device_transfer_bytes": all_saved,
            "host_copy_or_device_transfer_fraction": (
                all_saved
                / baseline["total_host_copy_or_device_transfer_bytes"]
            ),
            "lora_d2h_bytes": (
                baseline["lora_d2h_bytes"] - proposed["lora_d2h_bytes"]
            ),
            "base_d2h_bytes": (
                baseline["base_d2h_bytes"] - proposed["base_d2h_bytes"]
            ),
            "master_validation_host_copy_bytes": baseline[
                "master_validation_host_copy_bytes"
            ],
        },
        "scope": (
            "aggregate across all actors for one mirrored population plus one "
            "update/commit/final lifecycle; audit/materialization traffic only; "
            "excludes generation, noise construction, collectives, and "
            "state-ownership copies"
        ),
        "operation_counts": {
            "materializations": materializations,
            "logical_lora_verifications": logical_lora_verifications,
            "exact_base_boundaries": proposed_base_audits,
            "exact_base_boundaries_per_actor": len(EXACT_BOUNDARIES_V71),
            "baseline_candidate_master_validations": 7 * candidate_count,
        },
    }
    if candidate_count == 16 and world_size == 4:
        observed_base_audits = 60
        observed_materializations = 60
        result["measured_v66d_current_path"] = {
            "source": (
                "experiments/eggroll_es_hpo/"
                "qwen36_v66d_phase_memory_analysis_v71_20260717.md"
            ),
            "actor_count": 4,
            "candidate_count": 16,
            "base_exact_audits": observed_base_audits,
            "base_d2h_bytes": observed_base_audits * BASE_BYTES_V71,
            "materialization_calls": observed_materializations,
            "lora_equality_plus_sha_d2h_calls": 2 * observed_materializations,
            "lora_equality_plus_sha_d2h_bytes": (
                2 * observed_materializations * RUNTIME_LORA_BYTES_V71
            ),
            "base_plus_lora_d2h_bytes": (
                observed_base_audits * BASE_BYTES_V71
                + 2 * observed_materializations * RUNTIME_LORA_BYTES_V71
            ),
            "interpretation": (
                "measured/code-ledger lower bound for the immutable V66d run; "
                "the safety-equivalent tables above include the new explicit "
                "reward/update/commit/final audit schedule"
            ),
        }
    result["content_sha256_before_self_field"] = canonical_sha256_v71(result)
    return result


def build_contract_v71(candidate_count=16, world_size=4):
    schedule = canonical_audit_schedule_v71(candidate_count)
    result = {
        "schema": SCHEMA_V71,
        "status": "cpu_contract_no_gpu_or_protected_access",
        "candidate_count": candidate_count,
        "world_size": world_size,
        "rules": {
            "cheap_every_transition": True,
            "cheap_fields": [
                "object", "storage", "pointer", "metadata", "version",
            ],
            "single_lora_d2h_for_equality_and_sha": True,
            "reward_acceptance_after_exact_population_boundary": True,
            "update_acceptance_after_exact_boundary": True,
            "commit_and_final_exact_boundaries": True,
            "checkpoint_exact_boundary": True,
            "owned_master_identity_cached_without_validation_clone": True,
            "unknown_or_partial_rpc_requires_exact_restore_or_poison": True,
        },
        "schedule": schedule,
        "schedule_certificate": validate_audit_schedule_v71(
            schedule, candidate_count
        ),
        "traffic": traffic_account_v71(candidate_count, world_size),
        "protected_dev_ood_or_holdout_opened": False,
        "gpu_launch_performed": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256_v71(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the CPU-only V71 exact-audit traffic contract."
    )
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = build_contract_v71(args.candidate_count, args.world_size)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
