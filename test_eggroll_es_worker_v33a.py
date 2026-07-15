#!/usr/bin/env python3

import pytest

import eggroll_es_paired_data_compat_preregistration_v33a as prereg
import eggroll_es_worker_v33a as worker


def test_v33a_worker_certifies_all_64_fresh_direction_seeds():
    seeds = prereg.perturbation_seeds()
    projections = [
        {
            "full_seed": seed,
            "numpy_legacy_seed": worker.worker_r1.project_numpy_legacy_seed_r1(seed),
        }
        for seed in seeds
    ]
    extension = object.__new__(worker.PairedDataCompatWorkerExtensionV33A)
    result = extension.seed_projection_certificate_v33a(
        seeds,
        worker.worker_r1.canonical_sha256_r1(seeds),
        worker.worker_r1.canonical_sha256_r1(projections),
    )
    assert result["direction_count"] == 64
    assert result["numpy_projection_unique_count"] == 64
    assert result["only_numpy_legacy_seed_is_projected"] is True


def test_v33a_worker_rejects_partial_or_changed_seed_projection():
    seeds = prereg.perturbation_seeds()
    extension = object.__new__(worker.PairedDataCompatWorkerExtensionV33A)
    with pytest.raises(RuntimeError, match="coverage"):
        extension.seed_projection_certificate_v33a(seeds[:32], "0" * 64, "0" * 64)
    with pytest.raises(RuntimeError, match="certificate"):
        extension.seed_projection_certificate_v33a(seeds, "0" * 64, "0" * 64)
