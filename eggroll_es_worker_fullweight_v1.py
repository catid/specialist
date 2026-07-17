#!/usr/bin/env python3
"""Canonical full-weight ES state for the legacy ES-at-Scale adapter.

The upstream worker mutates BF16 parameters in place, restarts the same RNG
stream for every tensor, and tries to restore a perturbation by subtraction.
This extension instead owns an ordered CPU FP32 master on every TP=1 engine.
Candidates and restores are complete materializations from that immutable
master.  The legacy rank-zero update endpoint is a pure prepare step.  The
existing all-engine broadcast phase is overridden to shard population seeds,
all-reduce FP32 update tensors, and change every canonical master identically,
without losing sub-BF16 residuals.

Noise uses one private FP32 generator per population seed.  A generator is
seeded once and advanced across the complete, sorted runtime parameter
manifest.  The manifest is part of every state identity, so tensor boundaries
cannot silently change the frozen stream schedule.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "es-at-scale"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from es_at_scale.utils.worker_extension import WorkerExtension  # noqa: E402


NOISE_SCHEDULE_V1 = "torch-fp32-single-generator-sorted-manifest-v1"
CHECKPOINT_SCHEMA_V1 = "eggroll-es-canonical-full-weight-checkpoint-v1"
STATE_SCHEMA_V1 = "eggroll-es-canonical-full-weight-state-v1"
SUPPORTED_RUNTIME_DTYPES_V1 = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
}


def canonical_sha256_v1(value) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _require_seed_v1(seed) -> int:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, numbers.Integral)
        or int(seed) < 0
        or int(seed) >= 2**63
    ):
        raise ValueError("full-weight noise seed must be an integer in [0, 2^63)")
    return int(seed)


def _require_nonnegative_metadata_v1(value, label) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or int(value) < 0
    ):
        raise RuntimeError(f"canonical checkpoint {label} is invalid")
    return int(value)


def validate_update_inputs_v1(seeds, coefficients, alpha, population_size):
    """Validate the legacy update algebra without requiring P % world == 0."""
    if (
        isinstance(population_size, bool)
        or not isinstance(population_size, numbers.Integral)
        or int(population_size) <= 0
    ):
        raise ValueError("population size must be a positive integer")
    population_size = int(population_size)
    if not isinstance(seeds, (list, tuple)) or not isinstance(
        coefficients, (list, tuple)
    ):
        raise TypeError("seeds and coefficients must be lists or tuples")
    if len(seeds) != population_size or len(coefficients) != population_size:
        raise ValueError("seed and coefficient counts must match population")
    canonical_seeds = [_require_seed_v1(seed) for seed in seeds]
    if len(set(canonical_seeds)) != population_size:
        raise ValueError("population seeds must be unique")
    canonical_coefficients = []
    for coefficient in coefficients:
        if isinstance(coefficient, bool):
            raise ValueError("boolean ES coefficient is forbidden")
        try:
            value = float(coefficient)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("ES coefficient is not numeric") from error
        if not math.isfinite(value):
            raise ValueError("ES coefficients must be finite")
        canonical_coefficients.append(value)
    if isinstance(alpha, bool):
        raise ValueError("boolean ES alpha is forbidden")
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("ES alpha must be finite and nonnegative")
    return canonical_seeds, canonical_coefficients, alpha, population_size


def seed_shard_v1(seeds, coefficients, rank, world_size):
    """Return an uneven-safe, strided partition of the population."""
    rank = int(rank)
    world_size = int(world_size)
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("seed shard rank/world size is invalid")
    if len(seeds) != len(coefficients):
        raise ValueError("seed and coefficient lengths differ")
    indices = list(range(rank, len(seeds), world_size))
    return {
        "indices": indices,
        "seeds": [int(seeds[index]) for index in indices],
        "coefficients": [float(coefficients[index]) for index in indices],
    }


def _new_generator_v1(device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(_require_seed_v1(seed))
    return generator


def draw_noise_v1(parameter, generator: torch.Generator) -> torch.Tensor:
    """Draw the next FP32 segment from one persistent full-model stream."""
    return torch.randn(
        parameter.shape,
        dtype=torch.float32,
        device=parameter.device,
        generator=generator,
    )


def _tensor_hash_update_v1(digest, record, tensor):
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise RuntimeError("state hashing requires a contiguous CPU tensor")
    metadata = json.dumps(
        record,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    digest.update(len(metadata).to_bytes(8, "little"))
    digest.update(metadata)
    raw = tensor.reshape(-1).view(torch.uint8)
    byte_count = int(raw.numel())
    digest.update(byte_count.to_bytes(8, "little"))
    view = memoryview(raw.numpy())
    block_bytes = 64 * 1024 * 1024
    for start in range(0, byte_count, block_bytes):
        digest.update(view[start : start + block_bytes])
    return byte_count


def _state_identity_v1(manifest, named_tensors, *, state, dtype_override=None):
    tensors = dict(named_tensors)
    digest = hashlib.sha256()
    total_bytes = 0
    for record in manifest:
        name = record["name"]
        if name not in tensors:
            raise RuntimeError(f"state is missing parameter {name!r}")
        tensor = tensors[name]
        expected_dtype = (
            dtype_override if dtype_override is not None else record["dtype"]
        )
        if str(tensor.dtype) != str(expected_dtype):
            raise RuntimeError(f"state dtype changed for {name!r}")
        if list(tensor.shape) != record["shape"]:
            raise RuntimeError(f"state shape changed for {name!r}")
        total_bytes += _tensor_hash_update_v1(
            digest,
            {
                "name": name,
                "shape": record["shape"],
                "dtype": str(tensor.dtype),
                "offset": record["offset"],
                "elements": record["elements"],
            },
            tensor,
        )
    if set(tensors) != {item["name"] for item in manifest}:
        raise RuntimeError("state contains parameters outside the manifest")
    return _finished_state_identity_v1(
        manifest, digest, total_bytes, state=state,
    )


def _finished_state_identity_v1(manifest, digest, total_bytes, *, state):
    return {
        "schema": STATE_SCHEMA_V1,
        "state": str(state),
        "sha256": digest.hexdigest(),
        "manifest_sha256": canonical_sha256_v1(manifest),
        "parameter_count": len(manifest),
        "elements": sum(item["elements"] for item in manifest),
        "bytes": total_bytes,
    }


def _projection_identity_v1(manifest, parameters, master, *, state):
    """Hash a master projection one tensor at a time without a full mirror."""
    digest = hashlib.sha256()
    total_bytes = 0
    for record, (name, parameter) in zip(manifest, parameters, strict=True):
        projection = master[name].to(dtype=parameter.dtype).contiguous()
        total_bytes += _tensor_hash_update_v1(
            digest,
            {
                "name": name,
                "shape": record["shape"],
                "dtype": str(projection.dtype),
                "offset": record["offset"],
                "elements": record["elements"],
            },
            projection,
        )
        del projection
    return _finished_state_identity_v1(
        manifest, digest, total_bytes, state=state,
    )


class CanonicalFullWeightWorkerExtensionV1(WorkerExtension):
    """FP32-master, exact-restore replacement for the legacy full-weight worker."""

    def _communicator_v1(self):
        communicator = getattr(self, "inter_pg", None)
        if communicator is None:
            raise RuntimeError("inter-engine communicator is not initialized")
        rank = int(getattr(communicator, "rank", -1))
        world_size = int(getattr(communicator, "world_size", -1))
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise RuntimeError("inter-engine communicator metadata is invalid")
        if getattr(communicator, "available", True) is not True:
            raise RuntimeError("inter-engine communicator is unavailable")
        if getattr(communicator, "disabled", False) is not False:
            raise RuntimeError("inter-engine communicator is disabled")
        tp_world_size = int(getattr(self, "world_size", 1))
        if tp_world_size != 1:
            raise RuntimeError(
                "canonical full-weight v1 requires TP=1; TP-aware global noise "
                "and checkpoint publication are not implemented"
            )
        return {"rank": rank, "world_size": world_size, "tp_world_size": 1}

    def _parameter_layout_v1(self):
        model = self.model_runner.model
        canonical_raw = list(model.named_parameters())
        try:
            all_raw = list(model.named_parameters(remove_duplicate=False))
        except TypeError:
            all_raw = canonical_raw
        if not canonical_raw:
            raise RuntimeError("full-weight model has no parameters")
        all_names = set()
        aliases_by_object = {}
        parameter_by_object = {}
        for raw_name, parameter in all_raw:
            name = str(raw_name)
            if not name or name in all_names:
                raise RuntimeError(
                    f"duplicate or empty parameter name: {name!r}"
                )
            all_names.add(name)
            aliases_by_object.setdefault(id(parameter), []).append(name)
            parameter_by_object[id(parameter)] = parameter
        canonical_by_object = {}
        for raw_name, parameter in canonical_raw:
            name = str(raw_name)
            if id(parameter) in canonical_by_object:
                raise RuntimeError(
                    f"default named_parameters retained an alias: {name!r}"
                )
            canonical_by_object[id(parameter)] = name
        if set(canonical_by_object) != set(parameter_by_object):
            raise RuntimeError(
                "default named_parameters did not cover every parameter object"
            )
        raw = sorted(
            (
                canonical_name,
                parameter_by_object[object_id],
                sorted(
                    name for name in aliases_by_object[object_id]
                    if name != canonical_name
                ),
            )
            for object_id, canonical_name in canonical_by_object.items()
        )
        storages = set()
        device = None
        parameters = []
        manifest = []
        offset = 0
        for raw_name, parameter, aliases in raw:
            name = str(raw_name)
            if parameter.dtype not in SUPPORTED_RUNTIME_DTYPES_V1:
                raise RuntimeError(
                    f"unsupported full-weight runtime dtype for {name!r}: "
                    f"{parameter.dtype}"
                )
            if not parameter.is_contiguous():
                raise RuntimeError(f"noncontiguous parameter is unsupported: {name!r}")
            parameter_device = torch.device(parameter.device)
            if device is None:
                device = parameter_device
            elif parameter_device != device:
                raise RuntimeError("all TP=1 runtime parameters must share one device")
            if parameter.numel() > 0:
                storage = (
                    parameter_device.type,
                    parameter_device.index,
                    int(parameter.untyped_storage().data_ptr()),
                )
                if storage in storages:
                    raise RuntimeError(
                        "distinct parameter objects with overlapping storage are "
                        f"unsupported: {name!r}"
                    )
                storages.add(storage)
            elements = int(parameter.numel())
            record = {
                "name": name,
                "aliases": aliases,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "elements": elements,
                "runtime_bytes": elements * int(parameter.element_size()),
                "offset": offset,
            }
            parameters.append((name, parameter))
            manifest.append(record)
            offset += elements
        return parameters, manifest, device

    def _require_installed_v1(self):
        if getattr(self, "_fw_v1_installed", False) is not True:
            raise RuntimeError("canonical full-weight state is not installed")
        if getattr(self, "_fw_v1_poisoned", False):
            raise RuntimeError("canonical full-weight worker state is poisoned")

    def _validated_state_v1(self):
        self._require_installed_v1()
        parameters, manifest, device = self._parameter_layout_v1()
        if canonical_sha256_v1(manifest) != self._fw_v1_manifest_sha256:
            raise RuntimeError("runtime parameter manifest changed")
        master = self._fw_v1_master
        if set(master) != {item["name"] for item in manifest}:
            raise RuntimeError("canonical FP32 master namespace changed")
        for record in manifest:
            tensor = master[record["name"]]
            if (
                tensor.device.type != "cpu"
                or tensor.dtype != torch.float32
                or not tensor.is_contiguous()
                or list(tensor.shape) != record["shape"]
            ):
                raise RuntimeError(
                    f"canonical FP32 master metadata changed: {record['name']!r}"
                )
        return parameters, manifest, device, master

    def _require_quiescent_v1(self):
        self._require_installed_v1()
        if getattr(self, "_fw_v1_active_candidate", None) is not None:
            raise RuntimeError("a full-weight candidate transaction is active")
        if getattr(self, "_fw_v1_update_phase", None) is not None:
            raise RuntimeError("a full-weight update or broadcast is active")

    def _require_digest_consensus_v1(
        self, master_sha256, runtime_sha256, src_rank,
    ):
        """Use the live communicator to compare post-broadcast identities."""
        communicator = self._communicator_v1()
        local = bytes.fromhex(str(master_sha256)) + bytes.fromhex(
            str(runtime_sha256)
        )
        if len(local) != 64:
            raise RuntimeError("full-weight consensus identity is not SHA-256")
        own = torch.tensor(
            list(local), dtype=torch.uint8, device=self._fw_v1_device,
        )
        expected = own.clone()
        received = self.inter_pg.broadcast(
            expected,
            src=int(src_rank),
            stream=(
                torch.cuda.current_stream(torch.device(self._fw_v1_device))
                if torch.device(self._fw_v1_device).type == "cuda" else None
            ),
        )
        if received is not None:
            expected = received
        if not torch.equal(own, expected):
            raise RuntimeError("full-weight replicas disagree after FP32 broadcast")
        return {
            "source_rank": int(src_rank),
            "master_sha256": str(master_sha256),
            "runtime_sha256": str(runtime_sha256),
            "passed": True,
            "world_size": communicator["world_size"],
        }

    def install_full_weight_master_v1(self):
        """Capture/promote one canonical FP32 master after checkpoint loading."""
        if getattr(self, "_fw_v1_installed", False):
            raise RuntimeError("canonical full-weight installation is one-shot")
        communicator = self._communicator_v1()
        parameters, manifest, device = self._parameter_layout_v1()
        preloaded = getattr(self, "_fw_v1_preloaded_master", None)
        master = {}
        master_digest = hashlib.sha256()
        runtime_digest = hashlib.sha256()
        master_bytes = 0
        runtime_bytes = 0
        for record, (name, parameter) in zip(manifest, parameters, strict=True):
            if record["name"] != name:
                raise RuntimeError("parameter/manifest order changed during install")
            if preloaded is None:
                runtime = parameter.detach().to(device="cpu", copy=True).contiguous()
                canonical = runtime.to(dtype=torch.float32).contiguous()
            else:
                if name not in preloaded:
                    raise RuntimeError(f"loaded FP32 checkpoint is missing {name!r}")
                canonical = preloaded[name]
                if (
                    canonical.device.type != "cpu"
                    or canonical.dtype != torch.float32
                    or list(canonical.shape) != record["shape"]
                ):
                    raise RuntimeError(
                        f"loaded FP32 checkpoint metadata changed for {name!r}"
                    )
                canonical = canonical.contiguous()
                runtime = canonical.to(dtype=parameter.dtype).contiguous()
                parameter.data.copy_(runtime, non_blocking=False)
            master[name] = canonical
            master_bytes += _tensor_hash_update_v1(
                master_digest,
                {
                    "name": name,
                    "shape": record["shape"],
                    "dtype": str(torch.float32),
                    "offset": record["offset"],
                    "elements": record["elements"],
                },
                canonical,
            )
            runtime_bytes += _tensor_hash_update_v1(
                runtime_digest,
                {
                    "name": name,
                    "shape": record["shape"],
                    "dtype": str(runtime.dtype),
                    "offset": record["offset"],
                    "elements": record["elements"],
                },
                runtime,
            )
            del runtime
        if preloaded is not None and set(preloaded) != set(master):
            raise RuntimeError("loaded FP32 checkpoint contains unknown parameters")
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device)
        master_identity = _finished_state_identity_v1(
            manifest, master_digest, master_bytes, state="canonical_fp32",
        )
        runtime_identity = _finished_state_identity_v1(
            manifest, runtime_digest, runtime_bytes, state="runtime_projection",
        )
        if hasattr(self, "_fw_v1_preloaded_master"):
            del self._fw_v1_preloaded_master
        self._fw_v1_master = master
        self._fw_v1_manifest = manifest
        self._fw_v1_manifest_sha256 = canonical_sha256_v1(manifest)
        self._fw_v1_device = str(device)
        self._fw_v1_master_identity = master_identity
        self._fw_v1_runtime_identity = runtime_identity
        self._fw_v1_generation = int(
            getattr(self, "_fw_v1_preloaded_generation", 0)
        )
        self._fw_v1_update_sequence = int(
            getattr(self, "_fw_v1_preloaded_update_sequence", 0)
        )
        self._fw_v1_active_candidate = None
        self._fw_v1_last_restored = None
        self._fw_v1_update_phase = None
        self._fw_v1_pending_update = None
        self._fw_v1_last_update = None
        self._fw_v1_poisoned = False
        self._fw_v1_installed = True
        return {
            "schema": "eggroll-es-canonical-full-weight-install-v1",
            "installed": True,
            "communicator": communicator,
            "noise_schedule": NOISE_SCHEDULE_V1,
            "manifest_sha256": self._fw_v1_manifest_sha256,
            "master_generation": self._fw_v1_generation,
            "master_identity": master_identity,
            "runtime_identity": runtime_identity,
        }

    def materialize_full_weight_candidate_v1(
        self,
        transaction_id,
        state_index,
        seed,
        sigma,
        sign,
        expected_master_sha256,
    ):
        """Materialize master + sign*sigma*noise; master remains immutable."""
        self._require_quiescent_v1()
        transaction_id = str(transaction_id)
        if not transaction_id:
            raise ValueError("candidate transaction id is empty")
        state_index = int(state_index)
        if state_index < 0:
            raise ValueError("candidate state index must be nonnegative")
        seed = _require_seed_v1(seed)
        sigma = float(sigma)
        sign = int(sign)
        if not math.isfinite(sigma) or sigma < 0.0 or sign not in (-1, 1):
            raise ValueError("candidate requires finite sigma>=0 and sign +/-1")
        if str(expected_master_sha256) != self._fw_v1_master_identity["sha256"]:
            raise RuntimeError("candidate expected a different FP32 master")
        parameters, manifest, device, master = self._validated_state_v1()
        active = {
            "transaction_id": transaction_id,
            "state_index": state_index,
            "seed": seed,
            "sigma": sigma,
            "sigma_hex": sigma.hex(),
            "sign": sign,
            "phase": "materializing",
            "master_generation": self._fw_v1_generation,
            "master_sha256": self._fw_v1_master_identity["sha256"],
            "manifest_sha256": self._fw_v1_manifest_sha256,
        }
        # Record the transaction before the first write.  A partial write can
        # therefore always be repaired by restore_or_readback_full_weight_v1.
        self._fw_v1_active_candidate = active
        generator = _new_generator_v1(device, seed)
        try:
            with torch.no_grad():
                for record, (name, parameter) in zip(
                    manifest, parameters, strict=True,
                ):
                    if record["name"] != name:
                        raise RuntimeError("candidate manifest order changed")
                    candidate = master[name].to(
                        device=parameter.device, dtype=torch.float32,
                        copy=True,
                    )
                    noise = draw_noise_v1(parameter, generator)
                    candidate.add_(noise, alpha=sign * sigma)
                    parameter.data.copy_(
                        candidate.to(dtype=parameter.dtype), non_blocking=False,
                    )
                    del candidate, noise
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize(device)
        except BaseException:
            active["phase"] = "partial_materialization"
            raise
        active["phase"] = "candidate_materialized"
        receipt = {
            "schema": "eggroll-es-full-weight-candidate-v1",
            **active,
            "noise_schedule": NOISE_SCHEDULE_V1,
            "derived_candidate_sha256": canonical_sha256_v1({
                "master_sha256": active["master_sha256"],
                "manifest_sha256": active["manifest_sha256"],
                "noise_schedule": NOISE_SCHEDULE_V1,
                "seed": seed,
                "sigma_hex": sigma.hex(),
                "sign": sign,
            }),
        }
        active["receipt"] = receipt
        return receipt

    def restore_or_readback_full_weight_v1(
        self, transaction_id, expected_master_sha256,
    ):
        """Idempotently repair any candidate/partial candidate from master."""
        self._require_installed_v1()
        transaction_id = str(transaction_id)
        expected_master_sha256 = str(expected_master_sha256)
        if expected_master_sha256 != self._fw_v1_master_identity["sha256"]:
            raise RuntimeError("restore expected a different FP32 master")
        active = getattr(self, "_fw_v1_active_candidate", None)
        last = getattr(self, "_fw_v1_last_restored", None)
        if active is None:
            if (
                isinstance(last, dict)
                and last.get("transaction_id") == transaction_id
                and last.get("master_sha256") == expected_master_sha256
            ):
                return {**last, "idempotent_readback": True}
            raise RuntimeError("restore transaction is not active or completed")
        if active.get("transaction_id") != transaction_id:
            raise RuntimeError("restore transaction id does not match active candidate")
        if active.get("master_sha256") != expected_master_sha256:
            raise RuntimeError("active candidate master identity changed")
        parameters, manifest, device, master = self._validated_state_v1()
        active["phase"] = "restoring"
        runtime_digest = hashlib.sha256()
        runtime_bytes = 0
        try:
            with torch.no_grad():
                for record, (name, parameter) in zip(
                    manifest, parameters, strict=True,
                ):
                    runtime = master[name].to(dtype=parameter.dtype).contiguous()
                    parameter.data.copy_(runtime, non_blocking=False)
                    runtime_bytes += _tensor_hash_update_v1(
                        runtime_digest,
                        {
                            "name": name,
                            "shape": record["shape"],
                            "dtype": str(runtime.dtype),
                            "offset": record["offset"],
                            "elements": record["elements"],
                        },
                        runtime,
                    )
                    del runtime
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize(device)
            runtime_identity = _finished_state_identity_v1(
                manifest, runtime_digest, runtime_bytes,
                state="runtime_projection",
            )
        except BaseException:
            active["phase"] = "restore_failed"
            self._fw_v1_poisoned = True
            raise
        if runtime_identity != self._fw_v1_runtime_identity:
            self._fw_v1_poisoned = True
            raise RuntimeError("restored runtime projection identity changed")
        receipt = {
            "schema": "eggroll-es-full-weight-exact-restore-v1",
            "restored": True,
            "transaction_id": transaction_id,
            "master_generation": self._fw_v1_generation,
            "master_sha256": expected_master_sha256,
            "runtime_identity": runtime_identity,
            "manifest_sha256": self._fw_v1_manifest_sha256,
            "algebraic_native_restore_used": False,
            "idempotent_readback": False,
            "prior_candidate": dict(active),
        }
        self._fw_v1_active_candidate = None
        self._fw_v1_last_restored = receipt
        return receipt

    def perturb_self_weights(self, seed, noise_scale, negate=False):
        """Compatibility endpoint; all mutation uses the canonical protocol."""
        sign = -1 if bool(negate) else 1
        seed = _require_seed_v1(seed)
        sigma = float(noise_scale)
        transaction_id = canonical_sha256_v1({
            "schema": "legacy-full-weight-candidate-transaction-v1",
            "generation": int(getattr(self, "_fw_v1_generation", -1)),
            "seed": seed,
            "sigma_hex": sigma.hex(),
            "sign": sign,
        })
        return self.materialize_full_weight_candidate_v1(
            transaction_id,
            0,
            seed,
            sigma,
            sign,
            self._fw_v1_master_identity["sha256"],
        )

    def restore_self_weights(self, seed, sigma):
        """Compatibility endpoint; subtraction is deliberately impossible."""
        self._require_installed_v1()
        active = getattr(self, "_fw_v1_active_candidate", None)
        if not isinstance(active, dict):
            raise RuntimeError("no canonical full-weight candidate is active")
        if active.get("seed") != _require_seed_v1(seed):
            raise RuntimeError("restore seed does not match active candidate")
        sigma = float(sigma)
        if active.get("sigma_hex") != sigma.hex():
            raise RuntimeError("restore sigma does not match active candidate")
        return self.restore_or_readback_full_weight_v1(
            active["transaction_id"], active["master_sha256"],
        )

    def update_weights_from_seeds(
        self, seeds, coeffs, alpha, population_size,
    ):
        """Prepare the legacy payload for an all-engine sharded FP32 update."""
        self._require_quiescent_v1()
        communicator = self._communicator_v1()
        if communicator["rank"] != 0:
            raise RuntimeError("legacy update endpoint is rank-zero only")
        seeds, coeffs, alpha, population_size = validate_update_inputs_v1(
            seeds, coeffs, alpha, population_size,
        )
        coefficient_sha256 = canonical_sha256_v1({
            "seeds": seeds, "coefficients": coeffs,
        })
        pending = {
            "schema": "eggroll-es-full-weight-pending-update-v1",
            "seeds": seeds,
            "coefficients": coeffs,
            "coefficient_sha256": coefficient_sha256,
            "alpha": alpha,
            "population_size": population_size,
            "master_generation_before": self._fw_v1_generation,
            "update_sequence": self._fw_v1_update_sequence + 1,
            "manifest_sha256": self._fw_v1_manifest_sha256,
            "noise_schedule": NOISE_SCHEDULE_V1,
        }
        pending["manifest_sha256_v1"] = canonical_sha256_v1(pending)
        self._fw_v1_pending_update = pending
        self._fw_v1_update_phase = "awaiting_sharded_update"
        return {
            "schema": "eggroll-es-full-weight-update-prepared-v1",
            "prepared": True,
            **{
                key: value for key, value in pending.items()
                if key != "schema"
            },
        }

    def _broadcast_update_payload_v1(self, src_rank, communicator, device):
        """Broadcast rank zero's prepared numeric payload to every replica."""
        rank = communicator["rank"]
        stream = (
            torch.cuda.current_stream(device)
            if device.type == "cuda" else None
        )

        def broadcast(tensor):
            received = self.inter_pg.broadcast(
                tensor, src=int(src_rank), stream=stream,
            )
            return tensor if received is None else received

        if rank == src_rank:
            pending = getattr(self, "_fw_v1_pending_update", None)
            if not isinstance(pending, dict):
                raise RuntimeError("source has no prepared full-weight update")
            header = torch.tensor([
                pending["population_size"],
                pending["master_generation_before"],
                pending["update_sequence"],
            ], dtype=torch.int64, device=device)
            scalar = torch.tensor(
                [pending["alpha"]], dtype=torch.float64, device=device,
            )
        else:
            header = torch.empty(3, dtype=torch.int64, device=device)
            scalar = torch.empty(1, dtype=torch.float64, device=device)
        header = broadcast(header)
        scalar = broadcast(scalar)
        population_size = int(header[0].item())
        if population_size <= 0:
            raise RuntimeError("broadcast population size is invalid")
        if rank == src_rank:
            seed_tensor = torch.tensor(
                pending["seeds"], dtype=torch.int64, device=device,
            )
            coefficient_tensor = torch.tensor(
                pending["coefficients"], dtype=torch.float64, device=device,
            )
        else:
            seed_tensor = torch.empty(
                population_size, dtype=torch.int64, device=device,
            )
            coefficient_tensor = torch.empty(
                population_size, dtype=torch.float64, device=device,
            )
        seed_tensor = broadcast(seed_tensor)
        coefficient_tensor = broadcast(coefficient_tensor)
        seeds, coefficients, alpha, population_size = validate_update_inputs_v1(
            seed_tensor.cpu().tolist(),
            coefficient_tensor.cpu().tolist(),
            float(scalar[0].item()),
            population_size,
        )
        if int(header[1].item()) != self._fw_v1_generation:
            raise RuntimeError("broadcast update used a stale master generation")
        if int(header[2].item()) != self._fw_v1_update_sequence + 1:
            raise RuntimeError("broadcast update sequence is stale or skipped")
        return {
            "seeds": seeds,
            "coefficients": coefficients,
            "alpha": alpha,
            "population_size": population_size,
            "master_generation_before": int(header[1].item()),
            "update_sequence": int(header[2].item()),
            "coefficient_sha256": canonical_sha256_v1({
                "seeds": seeds, "coefficients": coefficients,
            }),
        }

    def broadcast_all_weights(self, src_rank: int):
        """Run a seed-sharded all-GPU update and seal identical FP32 masters."""
        self._require_installed_v1()
        communicator = self._communicator_v1()
        src_rank = int(src_rank)
        if src_rank != 0:
            raise ValueError("canonical full-weight v1 requires source rank zero")
        rank = communicator["rank"]
        if getattr(self, "_fw_v1_active_candidate", None) is not None:
            raise RuntimeError("cannot broadcast while a candidate is active")
        phase = getattr(self, "_fw_v1_update_phase", None)
        if rank == src_rank and phase != "awaiting_sharded_update":
            raise RuntimeError(
                "source has no prepared sharded update to execute"
            )
        if rank != src_rank and phase is not None:
            raise RuntimeError("destination worker is not quiescent for broadcast")
        parameters, manifest, device, master = self._validated_state_v1()
        self._fw_v1_update_phase = "receiving_sharded_update_payload"
        try:
            plan = self._broadcast_update_payload_v1(
                src_rank, communicator, device,
            )
            shard = seed_shard_v1(
                plan["seeds"], plan["coefficients"],
                rank, communicator["world_size"],
            )
            generators = [
                _new_generator_v1(device, seed) for seed in shard["seeds"]
            ]
            scale = plan["alpha"] / plan["population_size"]
            master_changed = 0
            runtime_changed = 0
            update_nonzero = 0
            update_l2_squared = 0.0
            update_max_abs = 0.0
            self._fw_v1_update_phase = "executing_sharded_fp32_update"
            with torch.no_grad():
                for record, (name, parameter) in zip(
                    manifest, parameters, strict=True,
                ):
                    accumulator = torch.zeros_like(
                        parameter.data, dtype=torch.float32,
                    )
                    for generator, coefficient in zip(
                        generators, shard["coefficients"], strict=True,
                    ):
                        noise = draw_noise_v1(parameter, generator)
                        accumulator.add_(noise, alpha=coefficient)
                        del noise
                    reduced = self.inter_pg.all_reduce(
                        accumulator,
                        out_tensor=accumulator,
                        stream=(
                            torch.cuda.current_stream(parameter.device)
                            if parameter.is_cuda else None
                        ),
                    )
                    if reduced is not None:
                        accumulator = reduced
                    if (
                        accumulator.dtype != torch.float32
                        or accumulator.device != parameter.device
                        or tuple(accumulator.shape) != tuple(parameter.shape)
                    ):
                        raise RuntimeError(
                            "FP32 all-reduce returned incompatible update metadata"
                        )
                    accumulator.mul_(scale)
                    update_nonzero += int(
                        torch.count_nonzero(accumulator).item()
                    )
                    update_norm = float(
                        torch.linalg.vector_norm(accumulator).item()
                    )
                    update_l2_squared += update_norm * update_norm
                    if accumulator.numel():
                        update_max_abs = max(
                            update_max_abs,
                            float(accumulator.abs().max().item()),
                        )
                    delta = accumulator.to(device="cpu").contiguous()
                    previous_master = master[name]
                    next_master = previous_master.add(delta)
                    master_changed += int(torch.count_nonzero(
                        next_master != previous_master
                    ).item())
                    previous_runtime = parameter.detach().clone()
                    next_runtime = next_master.to(
                        device=parameter.device, dtype=parameter.dtype,
                    )
                    runtime_changed += int(torch.count_nonzero(
                        next_runtime != previous_runtime
                    ).item())
                    master[name] = next_master.contiguous()
                    parameter.data.copy_(next_runtime, non_blocking=False)
                    del accumulator, delta, previous_runtime, next_runtime
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize(device)
            self._fw_v1_update_phase = "hashing_post_update_state"
            master_identity = _state_identity_v1(
                manifest,
                master.items(),
                state="canonical_fp32",
                dtype_override=torch.float32,
            )
            runtime_identity = _projection_identity_v1(
                manifest, parameters, master, state="runtime_projection",
            )
            self._fw_v1_update_phase = "sealing_replica_consensus"
            consensus = self._require_digest_consensus_v1(
                master_identity["sha256"], runtime_identity["sha256"], src_rank,
            )
            self._fw_v1_update_phase = "committing_sharded_update"
            self._fw_v1_master_identity = master_identity
            self._fw_v1_runtime_identity = runtime_identity
            self._fw_v1_generation += 1
            self._fw_v1_update_sequence += 1
            self._fw_v1_pending_update = None
            self._fw_v1_last_restored = None
            receipt = {
                "schema": "eggroll-es-full-weight-sharded-update-v1",
                "updated": True,
                "rank": rank,
                "world_size": communicator["world_size"],
                "source_rank": src_rank,
                "master_generation": self._fw_v1_generation,
                "update_sequence": self._fw_v1_update_sequence,
                "manifest_sha256": self._fw_v1_manifest_sha256,
                "master_identity": master_identity,
                "runtime_identity": runtime_identity,
                "replica_consensus": consensus,
                "noise_schedule": NOISE_SCHEDULE_V1,
                "coefficient_sha256": plan["coefficient_sha256"],
                "alpha": plan["alpha"],
                "population_size": plan["population_size"],
                "scale": scale,
                "shard_indices": shard["indices"],
                "shard_seeds": shard["seeds"],
                "master_changed_elements": master_changed,
                "runtime_changed_elements": runtime_changed,
                "update_nonzero_elements": update_nonzero,
                "master_update_l2": math.sqrt(update_l2_squared),
                "master_update_max_abs": update_max_abs,
            }
            self._fw_v1_last_update = receipt
            self._fw_v1_update_phase = None
            return receipt
        except BaseException:
            failed_phase = self._fw_v1_update_phase
            if failed_phase == "sealing_replica_consensus":
                terminal_phase = "terminal_identity_disagreement"
            elif failed_phase == "committing_sharded_update":
                terminal_phase = "terminal_commit_failure"
            else:
                terminal_phase = "terminal_partial_sharded_update"
            self._fw_v1_update_phase = terminal_phase
            self._fw_v1_poisoned = True
            raise

    def full_weight_state_certificate_v1(self):
        """Read-only live-state audit; this endpoint never repairs mutation."""
        self._require_quiescent_v1()
        try:
            parameters, manifest, device, master = self._validated_state_v1()
            master_identity = _state_identity_v1(
                manifest,
                master.items(),
                state="canonical_fp32",
                dtype_override=torch.float32,
            )
        except BaseException:
            self._fw_v1_poisoned = True
            raise
        if master_identity != self._fw_v1_master_identity:
            self._fw_v1_poisoned = True
            raise RuntimeError("canonical FP32 master identity changed")
        live_digest = hashlib.sha256()
        expected_digest = hashlib.sha256()
        live_bytes = 0
        expected_bytes = 0
        for record, (name, parameter) in zip(
            manifest, parameters, strict=True,
        ):
            current = parameter.detach().to(
                device="cpu", copy=True,
            ).contiguous()
            projection = master[name].to(dtype=parameter.dtype).contiguous()
            if not torch.equal(current, projection):
                self._fw_v1_poisoned = True
                raise RuntimeError(
                    f"live runtime differs from canonical projection: {name!r}"
                )
            live_bytes += _tensor_hash_update_v1(
                live_digest,
                {
                    "name": name,
                    "shape": record["shape"],
                    "dtype": str(current.dtype),
                    "offset": record["offset"],
                    "elements": record["elements"],
                },
                current,
            )
            expected_bytes += _tensor_hash_update_v1(
                expected_digest,
                {
                    "name": name,
                    "shape": record["shape"],
                    "dtype": str(projection.dtype),
                    "offset": record["offset"],
                    "elements": record["elements"],
                },
                projection,
            )
            del current, projection
        live_identity = _finished_state_identity_v1(
            manifest, live_digest, live_bytes, state="live_runtime",
        )
        expected_identity = _finished_state_identity_v1(
            manifest, expected_digest, expected_bytes,
            state="runtime_projection",
        )
        if expected_identity["sha256"] != self._fw_v1_runtime_identity["sha256"]:
            self._fw_v1_poisoned = True
            raise RuntimeError("canonical runtime projection identity changed")
        return {
            "schema": "eggroll-es-full-weight-read-only-certificate-v1",
            "passed": True,
            "read_only": True,
            "rank": self._communicator_v1()["rank"],
            "device": str(device),
            "master_generation": self._fw_v1_generation,
            "update_sequence": self._fw_v1_update_sequence,
            "manifest_sha256": self._fw_v1_manifest_sha256,
            "master_identity": master_identity,
            "live_runtime_identity": live_identity,
            "runtime_projection_identity": expected_identity,
            "noise_schedule": NOISE_SCHEDULE_V1,
        }

    def save_self_weights_to_disk(self, filepath):
        """Persist the resumable FP32 master, never a lossy BF16-only state."""
        self._require_quiescent_v1()
        if self._communicator_v1()["rank"] != 0:
            raise RuntimeError("only inter-engine rank zero may publish a checkpoint")
        try:
            _parameters, manifest, _device, master = self._validated_state_v1()
            master_identity = _state_identity_v1(
                manifest,
                master.items(),
                state="canonical_fp32",
                dtype_override=torch.float32,
            )
        except BaseException:
            self._fw_v1_poisoned = True
            raise
        if master_identity != self._fw_v1_master_identity:
            self._fw_v1_poisoned = True
            raise RuntimeError("cannot save a changed canonical FP32 master")
        payload = {
            "schema": CHECKPOINT_SCHEMA_V1,
            "manifest": self._fw_v1_manifest,
            "manifest_sha256": self._fw_v1_manifest_sha256,
            "master_generation": self._fw_v1_generation,
            "update_sequence": self._fw_v1_update_sequence,
            "master_identity": master_identity,
            "noise_schedule": NOISE_SCHEDULE_V1,
            "state_dict": master,
        }
        torch.save(payload, filepath)
        return {
            "schema": "eggroll-es-full-weight-checkpoint-saved-v1",
            "path": os.fspath(filepath),
            "master_identity": master_identity,
            "master_generation": self._fw_v1_generation,
            "update_sequence": self._fw_v1_update_sequence,
            "resumable_fp32_master": True,
        }

    def load_weights_from_disk(self, filepath):
        """Load a V1 FP32 checkpoint or explicitly promote a legacy state."""
        if getattr(self, "_fw_v1_installed", False):
            raise RuntimeError("checkpoint loading after master install is forbidden")
        self._communicator_v1()
        try:
            payload = torch.load(filepath, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(filepath, map_location="cpu")
        if not isinstance(payload, dict):
            raise RuntimeError("full-weight checkpoint root is not a dictionary")
        if payload.get("schema") == CHECKPOINT_SCHEMA_V1:
            state = payload.get("state_dict")
            if not isinstance(state, dict):
                raise RuntimeError("canonical checkpoint has no FP32 state_dict")
            if payload.get("noise_schedule") != NOISE_SCHEDULE_V1:
                raise RuntimeError("canonical checkpoint noise schedule changed")
            claimed_master_identity = payload.get("master_identity")
            if not isinstance(claimed_master_identity, dict):
                raise RuntimeError("canonical checkpoint has no master identity")
            claimed_manifest = payload.get("manifest")
            if canonical_sha256_v1(claimed_manifest) != payload.get(
                "manifest_sha256"
            ):
                raise RuntimeError("canonical checkpoint manifest identity changed")
            self._fw_v1_preloaded_generation = (
                _require_nonnegative_metadata_v1(
                    payload.get("master_generation"), "master generation",
                )
            )
            self._fw_v1_preloaded_update_sequence = (
                _require_nonnegative_metadata_v1(
                    payload.get("update_sequence"), "update sequence",
                )
            )
            if (
                self._fw_v1_preloaded_generation
                != self._fw_v1_preloaded_update_sequence
            ):
                raise RuntimeError(
                    "canonical checkpoint generation and update sequence differ"
                )
        else:
            # Legacy upstream checkpoints are plain native-dtype state dicts.
            # Promotion is explicit in the returned receipt; after this point
            # future snapshots preserve the FP32 residual.
            state = payload
            claimed_manifest = None
            claimed_master_identity = None
            self._fw_v1_preloaded_generation = 0
            self._fw_v1_preloaded_update_sequence = 0
        parameters, manifest, _device = self._parameter_layout_v1()
        if claimed_manifest is not None and claimed_manifest != manifest:
            raise RuntimeError("checkpoint runtime manifest differs from live model")
        if set(state) != {name for name, _ in parameters}:
            raise RuntimeError("checkpoint parameter namespace differs from live model")
        master = {}
        with torch.no_grad():
            for name, parameter in parameters:
                tensor = state[name]
                if not isinstance(tensor, torch.Tensor):
                    raise RuntimeError(f"checkpoint value is not a tensor: {name!r}")
                if tuple(tensor.shape) != tuple(parameter.shape):
                    raise RuntimeError(f"checkpoint shape changed: {name!r}")
                if (
                    claimed_master_identity is not None
                    and tensor.dtype != torch.float32
                ):
                    raise RuntimeError(
                        f"canonical checkpoint tensor is not FP32: {name!r}"
                    )
                canonical = tensor.detach().to(
                    device="cpu", dtype=torch.float32,
                ).contiguous()
                master[name] = canonical
            if claimed_master_identity is not None:
                actual_master_identity = _state_identity_v1(
                    manifest,
                    master.items(),
                    state="canonical_fp32",
                    dtype_override=torch.float32,
                )
                if actual_master_identity != claimed_master_identity:
                    raise RuntimeError(
                        "canonical checkpoint master identity changed"
                    )
            for name, parameter in parameters:
                canonical = master[name]
                parameter.data.copy_(
                    canonical.to(
                        device=parameter.device, dtype=parameter.dtype,
                    ),
                    non_blocking=False,
                )
        if torch.cuda.is_available() and parameters[0][1].is_cuda:
            torch.cuda.synchronize(parameters[0][1].device)
        self._fw_v1_preloaded_master = master
        return {
            "schema": "eggroll-es-full-weight-checkpoint-loaded-v1",
            "canonical_v1": payload.get("schema") == CHECKPOINT_SCHEMA_V1,
            "legacy_native_promoted": payload.get("schema") != CHECKPOINT_SCHEMA_V1,
            "manifest_sha256": canonical_sha256_v1(manifest),
            "master_generation": self._fw_v1_preloaded_generation,
            "update_sequence": self._fw_v1_preloaded_update_sequence,
        }
