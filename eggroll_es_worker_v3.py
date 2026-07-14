"""Distributed, fail-closed worker methods for anchored EGGROLL-ES v3.

V2 made population restoration byte exact by retaining a CPU copy of every
committed parameter on every engine.  V3 keeps that safety property while
parallelising the much more expensive seed update: every one of four TP=1
engines regenerates an equal, disjoint seed shard, FP32 accumulators are
summed through the already-initialised inter-engine PyNccl communicator, and
each engine applies the same reduced update locally.

The old exact reference is deliberately retained across a resident monotonic
alpha search.  It is fresh for population perturbations only until the first
accepted update; after that it is rollback-only.  A later population estimate
must explicitly capture one new reference, but alpha increments do not.
"""

import hashlib
import json
import math
import numbers

import torch

from eggroll_es_worker_v2 import ExactAuditWorkerExtension


REQUIRED_ENGINE_COUNT = 4


def canonical_sha256_v3(value):
    raw = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def coefficient_sha256_v3(seeds, coefficients):
    """Match the frozen trainer's coefficient identity byte for byte."""
    return canonical_sha256_v3({
        "seeds": list(seeds),
        "coefficients": list(coefficients),
    })


def validate_seed_coefficients_v3(
    seeds, coefficients, population_size, world_size,
):
    """Return canonical numeric lists after strict distributed validation."""
    population_size = int(population_size)
    world_size = int(world_size)
    if world_size != REQUIRED_ENGINE_COUNT:
        raise ValueError(
            f"anchored v3 requires exactly {REQUIRED_ENGINE_COUNT} engines"
        )
    if population_size <= 0 or population_size % world_size != 0:
        raise ValueError(
            "population size must be positive and divisible by engine count"
        )
    if len(seeds) != population_size or len(coefficients) != population_size:
        raise ValueError("seed and coefficient lists must match population")
    canonical_seeds = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
            raise ValueError("population seed is not an integer")
        canonical = int(seed)
        if canonical < 0:
            raise ValueError("population seeds must be nonnegative integers")
        canonical_seeds.append(canonical)
    if len(set(canonical_seeds)) != population_size:
        raise ValueError("population seeds must be unique")
    canonical_coefficients = []
    for coefficient in coefficients:
        if isinstance(coefficient, bool):
            raise ValueError("boolean population coefficient is forbidden")
        try:
            canonical = float(coefficient)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("population coefficient is not numeric") from error
        if not math.isfinite(canonical):
            raise ValueError("population coefficients must be finite")
        canonical_coefficients.append(canonical)
    return canonical_seeds, canonical_coefficients


def seed_shard_v3(seeds, coefficients, rank, world_size):
    """Return the deterministic, strided shard owned by one engine rank."""
    rank = int(rank)
    world_size = int(world_size)
    if rank < 0 or rank >= world_size:
        raise ValueError("inter-engine communicator rank is out of range")
    indices = list(range(rank, len(seeds), world_size))
    return {
        "indices": indices,
        "seeds": [int(seeds[index]) for index in indices],
        "coefficients": [float(coefficients[index]) for index in indices],
    }


def accumulate_seed_terms_v3(parameter, seeds, coefficients):
    """Regenerate one seed shard and accumulate it in FP32.

    Noise is generated in the model parameter dtype before conversion to FP32,
    exactly matching the frozen upstream update algebra.
    """
    if len(seeds) != len(coefficients):
        raise ValueError("local seed and coefficient shard lengths differ")
    accumulator = torch.zeros_like(parameter.data, dtype=torch.float32)
    for seed, coefficient in zip(seeds, coefficients):
        generator = torch.Generator(device=parameter.device)
        generator.manual_seed(int(seed))
        noise = torch.randn(
            parameter.shape,
            dtype=parameter.dtype,
            device=parameter.device,
            generator=generator,
        )
        accumulator.add_(noise.to(torch.float32), alpha=float(coefficient))
        del noise
    return accumulator


def update_manifest_v3(
    *,
    coefficient_sha256,
    population_size,
    world_size,
    reference_generation,
    plan_id,
    update_sequence,
    previous_alpha,
    target_alpha,
    expected_base_sha256,
):
    return {
        "schema": "eggroll-es-distributed-update-manifest-v3",
        "coefficient_sha256": str(coefficient_sha256),
        "population_size": int(population_size),
        "world_size": int(world_size),
        "reference_generation": int(reference_generation),
        "plan_id": str(plan_id),
        "update_sequence": int(update_sequence),
        "previous_alpha": float(previous_alpha),
        "target_alpha": float(target_alpha),
        "expected_base_sha256": str(expected_base_sha256),
    }


class DistributedExactAuditWorkerExtensionV3(ExactAuditWorkerExtension):
    """Exact population restore plus sharded, audited FP32 ES updates."""

    def _require_no_pending_update_v3(self):
        pending = getattr(self, "_v3_pending_update", None)
        if pending is not None:
            raise RuntimeError("a distributed v3 update is still pending")

    def _require_fresh_population_reference_v3(self):
        if not getattr(self, "_v3_reference_fresh", False):
            raise RuntimeError(
                "exact reference is stale for population use; capture the "
                "current committed weights before perturbing or restoring"
            )

    def _communicator_state_v3(self, expected_world_size):
        expected_world_size = int(expected_world_size)
        if expected_world_size != REQUIRED_ENGINE_COUNT:
            raise RuntimeError(
                f"anchored v3 requires exactly {REQUIRED_ENGINE_COUNT} engines"
            )
        communicator = getattr(self, "inter_pg", None)
        if communicator is None:
            raise RuntimeError("inter-engine communicator is not initialised")
        world_size = int(getattr(communicator, "world_size", -1))
        rank = int(getattr(communicator, "rank", -1))
        if world_size != expected_world_size:
            raise RuntimeError("inter-engine communicator world size changed")
        if rank < 0 or rank >= world_size:
            raise RuntimeError("inter-engine communicator rank is invalid")
        if getattr(communicator, "available", None) is not True:
            raise RuntimeError("inter-engine PyNccl communicator is unavailable")
        if getattr(communicator, "disabled", None) is not False:
            raise RuntimeError("inter-engine PyNccl communicator is disabled")
        # V3 initially supports one complete model replica per engine.  With
        # TP>1, parameter names/shapes differ by TP rank and need a different
        # shard manifest and final-identity protocol.
        tp_world_size = int(getattr(self, "world_size", 1))
        if tp_world_size != 1:
            raise RuntimeError("anchored v3 initially requires TP=1 per engine")
        return {
            "rank": rank,
            "world_size": world_size,
            "tp_world_size": tp_world_size,
            "available": True,
            "disabled": False,
        }

    def save_self_exact_reference(self, chunk_bytes=64 * 1024 * 1024):
        """Capture one new population base and retire any update session."""
        self._require_no_pending_update_v3()
        identity = super().save_self_exact_reference(chunk_bytes=chunk_bytes)
        generation = int(getattr(self, "_v3_reference_generation", 0)) + 1
        self._v3_reference_generation = generation
        self._v3_reference_fresh = True
        self._v3_reference_identity = dict(identity)
        self._v3_current_identity = dict(identity)
        self._v3_update_session = None
        self._v3_update_sequence = 0
        self._v3_accepted_alpha = 0.0
        self._v3_pending_update = None
        return {
            "schema": "eggroll-es-exact-reference-state-v3",
            "reference_generation": generation,
            "fresh_for_population": True,
            "identity": dict(identity),
        }

    def restore_self_weights_exact(self):
        self._require_no_pending_update_v3()
        self._require_fresh_population_reference_v3()
        return super().restore_self_weights_exact()

    def verify_self_exact_reference(self, chunk_bytes=64 * 1024 * 1024):
        self._require_no_pending_update_v3()
        self._require_fresh_population_reference_v3()
        result = super().verify_self_exact_reference(chunk_bytes=chunk_bytes)
        result = dict(result)
        result["schema"] = "eggroll-es-exact-reference-check-v3"
        result["reference_generation"] = self._v3_reference_generation
        return result

    def perturb_self_weights(self, seed, noise_scale, negate=False):
        self._require_no_pending_update_v3()
        self._require_fresh_population_reference_v3()
        return super().perturb_self_weights(seed, noise_scale, negate)

    def inspect_distributed_update_state_v3(self, expected_world_size):
        communicator = self._communicator_state_v3(expected_world_size)
        identity = getattr(self, "_v3_current_identity", None)
        reference = getattr(self, "_v3_reference_identity", None)
        if not isinstance(identity, dict) or not isinstance(reference, dict):
            raise RuntimeError("v3 exact reference has not been captured")
        return {
            "schema": "eggroll-es-distributed-worker-state-v3",
            "communicator": communicator,
            "reference_generation": int(self._v3_reference_generation),
            "reference_fresh_for_population": bool(self._v3_reference_fresh),
            "reference_identity": dict(reference),
            "current_identity": dict(identity),
            "update_session": self._v3_update_session,
            "update_sequence": int(self._v3_update_sequence),
            "accepted_alpha": float(self._v3_accepted_alpha),
            "pending": self._v3_pending_update is not None,
        }

    def _allocation_readiness_preflight_v3(self):
        """Rehearse the largest local update allocation before any collective.

        The accumulation peak for one parameter is its FP32 accumulator, one
        parameter-dtype noise tensor, and the transient FP32 result of
        ``noise.to(torch.float32)``.  Allocating and touching that full trio
        during prepare is local-only: it creates no NCCL
        operation, consumes no RNG state, and changes no model value.  If one
        replica cannot make the allocation, its prepare RPC fails while every
        other replica is still outside the collective loop.
        """
        largest = None
        parameter_count = 0
        for name, parameter in self.model_runner.model.named_parameters():
            parameter_count += 1
            footprint = int(parameter.numel()) * (
                2 * torch.empty((), dtype=torch.float32).element_size()
                + int(parameter.element_size())
            )
            candidate = (footprint, str(name), parameter)
            if largest is None or candidate[:2] > largest[:2]:
                largest = candidate
        if largest is None:
            raise RuntimeError("distributed update model has no parameters")

        footprint, name, parameter = largest
        accumulator = None
        native_noise = None
        converted_noise = None
        try:
            accumulator = torch.empty_like(
                parameter.data, dtype=torch.float32,
            )
            native_noise = torch.empty_like(parameter.data)
            converted_noise = torch.empty_like(
                parameter.data, dtype=torch.float32,
            )
            # Touch both allocations so asynchronous CUDA allocation/kernel
            # failures surface during prepare, not after another rank enters
            # its first all-reduce.
            accumulator.zero_()
            native_noise.zero_()
            converted_noise.zero_()
            if parameter.is_cuda:
                torch.cuda.synchronize(parameter.device)
        finally:
            # Free all rehearsal tensors before prepare returns.  The explicit
            # synchronization above ensures their touch kernels have completed;
            # no scratch tensor survives into the collective phase.
            del accumulator, native_noise, converted_noise
        return {
            "schema": "eggroll-es-local-allocation-preflight-v3",
            "passed": True,
            "parameter_count": parameter_count,
            "largest_parameter_name": name,
            "largest_parameter_shape": list(parameter.shape),
            "parameter_dtype": str(parameter.dtype),
            "accumulator_dtype": "torch.float32",
            "simulated_peak_temporary_bytes": footprint,
            "scratch_freed_before_collectives": True,
            "collectives_created": False,
            "rng_consumed": False,
            "weights_changed": False,
        }

    def prepare_sharded_seed_update_v3(
        self,
        seeds,
        coefficients,
        coefficient_sha256,
        population_size,
        expected_world_size,
        reference_generation,
        plan_id,
        update_sequence,
        previous_alpha,
        target_alpha,
        expected_base_sha256,
    ):
        """Validate a complete manifest before any rank enters a collective."""
        self._require_no_pending_update_v3()
        communicator = self._communicator_state_v3(expected_world_size)
        seeds, coefficients = validate_seed_coefficients_v3(
            seeds, coefficients, population_size, expected_world_size,
        )
        actual_coefficient_sha = coefficient_sha256_v3(seeds, coefficients)
        if actual_coefficient_sha != coefficient_sha256:
            raise ValueError("coefficient identity changed before update")
        reference_generation = int(reference_generation)
        if reference_generation != int(self._v3_reference_generation):
            raise RuntimeError("distributed update used a stale reference generation")
        expected_base_sha256 = str(expected_base_sha256)
        current_sha = self._v3_current_identity.get("sha256")
        if current_sha != expected_base_sha256:
            raise RuntimeError("distributed update base hash changed")
        update_sequence = int(update_sequence)
        if update_sequence != int(self._v3_update_sequence) + 1:
            raise RuntimeError("distributed update sequence is stale or skipped")
        if self._v3_update_sequence == 0 and not self._v3_reference_fresh:
            raise RuntimeError(
                "first distributed update cannot use a stale population reference"
            )
        if self._v3_update_sequence > 0 and self._v3_reference_fresh:
            raise RuntimeError(
                "continued distributed update unexpectedly has a fresh reference"
            )
        previous_alpha = float(previous_alpha)
        target_alpha = float(target_alpha)
        if (
            not math.isfinite(previous_alpha)
            or not math.isfinite(target_alpha)
            or target_alpha <= previous_alpha
            or previous_alpha != float(self._v3_accepted_alpha)
        ):
            raise ValueError("distributed alpha transition is not monotonic")
        plan_id = str(plan_id)
        if not plan_id:
            raise ValueError("distributed update plan id is empty")
        if self._v3_update_session not in (None, plan_id):
            raise RuntimeError("a different distributed update plan is active")
        if self._v3_update_sequence > 0 and self._v3_update_session != plan_id:
            raise RuntimeError("distributed update session changed mid-search")

        manifest = update_manifest_v3(
            coefficient_sha256=coefficient_sha256,
            population_size=population_size,
            world_size=expected_world_size,
            reference_generation=reference_generation,
            plan_id=plan_id,
            update_sequence=update_sequence,
            previous_alpha=previous_alpha,
            target_alpha=target_alpha,
            expected_base_sha256=expected_base_sha256,
        )
        manifest_sha = canonical_sha256_v3(manifest)
        shard = seed_shard_v3(
            seeds,
            coefficients,
            communicator["rank"],
            communicator["world_size"],
        )
        expected_count = int(population_size) // int(expected_world_size)
        if len(shard["indices"]) != expected_count:
            raise RuntimeError("distributed seed shard is not balanced")
        allocation_preflight = self._allocation_readiness_preflight_v3()
        self._v3_pending_update = {
            "phase": "prepared",
            "manifest": manifest,
            "manifest_sha256": manifest_sha,
            "shard": shard,
        }
        return {
            "schema": "eggroll-es-distributed-update-prepared-v3",
            "prepared": True,
            "manifest_sha256": manifest_sha,
            "rank": communicator["rank"],
            "world_size": communicator["world_size"],
            "shard_indices": list(shard["indices"]),
            "shard_seeds": list(shard["seeds"]),
            "shard_pair_sha256": canonical_sha256_v3({
                "seeds": shard["seeds"],
                "coefficients": shard["coefficients"],
            }),
            "base_sha256": current_sha,
            "reference_generation": reference_generation,
            "update_sequence": update_sequence,
            "allocation_preflight": allocation_preflight,
        }

    def execute_prepared_seed_update_v3(self, manifest_sha256):
        """All-reduce every FP32 parameter accumulator, then hash weights."""
        pending = getattr(self, "_v3_pending_update", None)
        if not isinstance(pending, dict) or pending.get("phase") != "prepared":
            raise RuntimeError("distributed update was not prepared")
        if pending.get("manifest_sha256") != manifest_sha256:
            raise RuntimeError("prepared distributed manifest changed")
        manifest = pending["manifest"]
        communicator_state = self._communicator_state_v3(
            manifest["world_size"],
        )
        shard = pending["shard"]
        parameter_count = 0
        reduced_element_count = 0
        try:
            with torch.no_grad():
                for _, parameter in self.model_runner.model.named_parameters():
                    accumulator = accumulate_seed_terms_v3(
                        parameter, shard["seeds"], shard["coefficients"],
                    )
                    if accumulator.dtype != torch.float32:
                        raise RuntimeError("local update accumulator is not FP32")
                    stream = (
                        torch.cuda.current_stream()
                        if accumulator.is_cuda else None
                    )
                    reduced = self.inter_pg.all_reduce(
                        accumulator,
                        out_tensor=accumulator,
                        stream=stream,
                    )
                    if reduced is None:
                        raise RuntimeError("PyNccl all-reduce returned no tensor")
                    if (
                        reduced.dtype != torch.float32
                        or reduced.device != parameter.device
                        or tuple(reduced.shape) != tuple(parameter.shape)
                    ):
                        raise RuntimeError(
                            "PyNccl all-reduce returned an incompatible tensor"
                        )
                    scale = (
                        float(manifest["target_alpha"])
                        - float(manifest["previous_alpha"])
                    ) / float(manifest["population_size"])
                    # Preserve the upstream fixed algebra: scale the complete
                    # direction in FP32, then cast only once at parameter add.
                    reduced.mul_(scale)
                    parameter.data.add_(reduced.to(parameter.dtype))
                    parameter_count += 1
                    reduced_element_count += int(reduced.numel())
                    del accumulator, reduced
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            final_identity = super().weight_state_sha256()
        except Exception:
            # Retain and use the old exact reference only for fail-closed abort.
            ExactAuditWorkerExtension.restore_self_weights_exact(self)
            self._v3_current_identity = dict(self._v3_reference_identity)
            self._v3_reference_fresh = True
            self._v3_pending_update = None
            raise
        pending["phase"] = "executed"
        pending["final_identity"] = dict(final_identity)
        return {
            "schema": "eggroll-es-distributed-update-executed-v3",
            "executed": True,
            "manifest_sha256": manifest_sha256,
            "rank": communicator_state["rank"],
            "world_size": communicator_state["world_size"],
            "parameter_count": parameter_count,
            "reduced_element_count": reduced_element_count,
            "collective_dtype": "torch.float32",
            "final_identity": dict(final_identity),
        }

    def commit_prepared_seed_update_v3(
        self, manifest_sha256, expected_final_sha256,
    ):
        """Accept an update only after the controller compares every rank."""
        pending = getattr(self, "_v3_pending_update", None)
        if not isinstance(pending, dict) or pending.get("phase") != "executed":
            raise RuntimeError("distributed update was not executed")
        if pending.get("manifest_sha256") != manifest_sha256:
            raise RuntimeError("executed distributed manifest changed")
        final_identity = pending.get("final_identity")
        if final_identity.get("sha256") != expected_final_sha256:
            raise RuntimeError("distributed final weight identity changed")
        manifest = pending["manifest"]
        self._v3_current_identity = dict(final_identity)
        self._v3_update_session = manifest["plan_id"]
        self._v3_update_sequence = int(manifest["update_sequence"])
        self._v3_accepted_alpha = float(manifest["target_alpha"])
        # The retained reference is now rollback-only.  Population operations
        # fail until save_self_exact_reference captures the accepted weights.
        self._v3_reference_fresh = False
        self._v3_pending_update = None
        return {
            "schema": "eggroll-es-distributed-update-committed-v3",
            "committed": True,
            "manifest_sha256": manifest_sha256,
            "rank": int(self.inter_pg.rank),
            "final_sha256": expected_final_sha256,
            "reference_generation": int(self._v3_reference_generation),
            "reference_fresh_for_population": False,
            "update_sequence": int(self._v3_update_sequence),
            "accepted_alpha": float(self._v3_accepted_alpha),
        }

    def abort_distributed_update_v3(self, plan_id, reference_generation):
        """Restore the retained population base after any update failure."""
        if int(reference_generation) != int(self._v3_reference_generation):
            raise RuntimeError("cannot abort through a stale reference generation")
        active_plan = self._v3_update_session
        pending = getattr(self, "_v3_pending_update", None)
        pending_plan = (
            pending.get("manifest", {}).get("plan_id")
            if isinstance(pending, dict) else None
        )
        if (
            active_plan is not None
            or pending_plan is not None
        ) and str(plan_id) not in (active_plan, pending_plan):
            raise RuntimeError("abort plan does not match active update session")
        ExactAuditWorkerExtension.restore_self_weights_exact(self)
        verification = ExactAuditWorkerExtension.verify_self_exact_reference(self)
        self._v3_current_identity = dict(self._v3_reference_identity)
        self._v3_reference_fresh = True
        self._v3_update_session = None
        self._v3_update_sequence = 0
        self._v3_accepted_alpha = 0.0
        self._v3_pending_update = None
        return {
            "schema": "eggroll-es-distributed-update-abort-v3",
            "aborted": True,
            "rank": int(self.inter_pg.rank),
            "reference_generation": int(self._v3_reference_generation),
            "restored_identity": dict(verification["current"]),
        }
