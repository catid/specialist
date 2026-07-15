#!/usr/bin/env python3
"""Seed-domain repair worker for the exclusive V23A retry R1."""

from __future__ import annotations

import hashlib
import json
import random

import numpy as np
import torch

import eggroll_es_worker_v23a as worker_v23a


NUMPY_LEGACY_MODULUS_R1 = 2**32
MAX_PREREGISTERED_TORCH_SEED_R1 = 2**63 - 1


def project_numpy_legacy_seed_r1(full_seed: int) -> int:
    """Project an admitted full Torch seed into RandomState's 32-bit domain."""
    if isinstance(full_seed, bool) or not isinstance(full_seed, int):
        raise TypeError("v23a-r1 seed must be an integer, not a coercible value")
    if not 0 <= full_seed <= MAX_PREREGISTERED_TORCH_SEED_R1:
        raise ValueError("v23a-r1 seed is outside the preregistered positive 63-bit domain")
    return full_seed % NUMPY_LEGACY_MODULUS_R1


def canonical_sha256_r1(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class InsertionLocationAuditWorkerExtensionV23ARetryR1(
    worker_v23a.InsertionLocationAuditWorkerExtensionV23A,
):
    """Preserve Torch's full perturbation seed; narrow only legacy NumPy RNG."""

    def _set_seed(self, seed):
        full_seed = seed
        numpy_seed = project_numpy_legacy_seed_r1(full_seed)

        # Keep the original full seed for Python and every Torch RNG.  In
        # particular, CUDA Philox streams can distinguish the high 31 bits.
        self.local_seed = full_seed
        self.local_numpy_seed_r1 = numpy_seed
        random.seed(full_seed)
        np.random.seed(numpy_seed)
        torch.manual_seed(full_seed)
        torch.cuda.manual_seed_all(full_seed)

    def seed_projection_certificate_v23a_r1(
        self, direction_seeds, expected_seed_list_sha256, expected_projection_sha256,
    ):
        if (
            not isinstance(direction_seeds, list)
            or len(direction_seeds) != 32
            or len(set(direction_seeds)) != 32
        ):
            raise RuntimeError("v23a-r1 direction seed coverage changed")
        projections = [
            {
                "full_seed": seed,
                "numpy_legacy_seed": project_numpy_legacy_seed_r1(seed),
            }
            for seed in direction_seeds
        ]
        projected = [item["numpy_legacy_seed"] for item in projections]
        seed_list_sha256 = canonical_sha256_r1(direction_seeds)
        projection_sha256 = canonical_sha256_r1(projections)
        if (
            seed_list_sha256 != expected_seed_list_sha256
            or projection_sha256 != expected_projection_sha256
            or len(set(projected)) != 32
            or any(seed == 0 for seed in projected)
        ):
            raise RuntimeError("v23a-r1 seed projection certificate changed")
        return {
            "schema": "eggroll-es-v23a-seed-projection-worker-certificate-r1",
            "direction_count": 32,
            "direction_seed_list_sha256": seed_list_sha256,
            "full_to_numpy_projection_sha256": projection_sha256,
            "numpy_projection_unique_count": 32,
            "numpy_projection_contains_zero": False,
            "python_random_receives_full_seed": True,
            "torch_global_receives_full_seed": True,
            "torch_cuda_all_receives_full_seed": True,
            "explicit_torch_generator_receives_full_seed": True,
            "only_numpy_legacy_seed_is_projected": True,
        }
