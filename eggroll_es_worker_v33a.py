#!/usr/bin/env python3
"""64-direction seed certificate overlay for the V33A train-only runtime."""

from __future__ import annotations

import eggroll_es_worker_v23a_retry_r1 as worker_r1


class PairedDataCompatWorkerExtensionV33A(
    worker_r1.InsertionLocationAuditWorkerExtensionV23ARetryR1,
):
    """Keep the repaired full-seed behavior and certify all 64 V33A seeds."""

    def seed_projection_certificate_v33a(
        self, direction_seeds, expected_seed_list_sha256,
        expected_projection_sha256,
    ):
        if (
            not isinstance(direction_seeds, list)
            or len(direction_seeds) != 64
            or len(set(direction_seeds)) != 64
        ):
            raise RuntimeError("v33a direction seed coverage changed")
        projections = [
            {
                "full_seed": seed,
                "numpy_legacy_seed": worker_r1.project_numpy_legacy_seed_r1(seed),
            }
            for seed in direction_seeds
        ]
        projected = [item["numpy_legacy_seed"] for item in projections]
        seed_list_sha256 = worker_r1.canonical_sha256_r1(direction_seeds)
        projection_sha256 = worker_r1.canonical_sha256_r1(projections)
        if (
            seed_list_sha256 != expected_seed_list_sha256
            or projection_sha256 != expected_projection_sha256
            or len(set(projected)) != 64
            or any(seed == 0 for seed in projected)
        ):
            raise RuntimeError("v33a seed projection certificate changed")
        return {
            "schema": "eggroll-es-v33a-seed-projection-worker-certificate",
            "direction_count": 64,
            "direction_seed_list_sha256": seed_list_sha256,
            "full_to_numpy_projection_sha256": projection_sha256,
            "numpy_projection_unique_count": 64,
            "numpy_projection_contains_zero": False,
            "python_random_receives_full_seed": True,
            "torch_global_receives_full_seed": True,
            "torch_cuda_all_receives_full_seed": True,
            "explicit_torch_generator_receives_full_seed": True,
            "only_numpy_legacy_seed_is_projected": True,
        }
