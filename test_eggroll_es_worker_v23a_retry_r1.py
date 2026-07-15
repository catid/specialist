import inspect
import os

import pytest
import torch

import eggroll_es_worker_v4 as worker_v4
import eggroll_es_worker_v23a_retry_r1 as worker_r1
import eggroll_es_insertion_stability_preregistration_v23a as prereg_v23a


FULL_SEED = 7_100_699_430_629_063_086
LOW32_SEED = 3_547_773_358


def test_v23a_r1_projects_only_numpy_and_preserves_full_torch_seed(monkeypatch):
    calls = []
    monkeypatch.setattr(worker_r1.random, "seed", lambda seed: calls.append(("python", seed)))
    monkeypatch.setattr(worker_r1.np.random, "seed", lambda seed: calls.append(("numpy", seed)))
    monkeypatch.setattr(worker_r1.torch, "manual_seed", lambda seed: calls.append(("torch", seed)))
    monkeypatch.setattr(
        worker_r1.torch.cuda, "manual_seed_all",
        lambda seed: calls.append(("torch_cuda_all", seed)),
    )
    worker = object.__new__(worker_r1.InsertionLocationAuditWorkerExtensionV23ARetryR1)
    worker._set_seed(FULL_SEED)
    assert worker.local_seed == FULL_SEED
    assert worker.local_numpy_seed_r1 == LOW32_SEED
    assert calls == [
        ("python", FULL_SEED),
        ("numpy", LOW32_SEED),
        ("torch", FULL_SEED),
        ("torch_cuda_all", FULL_SEED),
    ]


def test_v23a_r1_seed_projection_is_fail_closed_and_exact():
    assert worker_r1.project_numpy_legacy_seed_r1(FULL_SEED) == LOW32_SEED
    assert worker_r1.project_numpy_legacy_seed_r1(2**32) == 0
    assert worker_r1.project_numpy_legacy_seed_r1(2**63 - 1) == 2**32 - 1
    for invalid in (True, 1.0, "1", None):
        with pytest.raises(TypeError):
            worker_r1.project_numpy_legacy_seed_r1(invalid)
    for invalid in (-1, 2**63):
        with pytest.raises(ValueError):
            worker_r1.project_numpy_legacy_seed_r1(invalid)


def test_v23a_r1_explicit_parameter_generators_still_receive_full_seed():
    source = inspect.getsource(worker_v4.LayerRestrictedExactAuditWorkerExtensionV4.perturb_self_weights)
    assert "self._set_seed(int(seed))" in source
    assert "generator.manual_seed(int(seed))" in source
    assert source.index("self._set_seed(int(seed))") < source.index("parameter.data.add_")
    assert source.index("generator.manual_seed(int(seed))") < source.index("parameter.data.add_")


def test_v23a_r1_worker_certifies_all_preregistered_projections_before_scoring():
    seeds = prereg_v23a.perturbation_basis_v23a()["direction_seeds"]
    mapping = [
        {"full_seed": seed, "numpy_legacy_seed": seed % (2**32)}
        for seed in seeds
    ]
    worker = object.__new__(worker_r1.InsertionLocationAuditWorkerExtensionV23ARetryR1)
    report = worker.seed_projection_certificate_v23a_r1(
        seeds, prereg_v23a.canonical_sha256(seeds), prereg_v23a.canonical_sha256(mapping)
    )
    assert report["direction_count"] == 32
    assert report["numpy_projection_unique_count"] == 32
    assert report["numpy_projection_contains_zero"] is False
    assert report["explicit_torch_generator_receives_full_seed"] is True
    assert report["only_numpy_legacy_seed_is_projected"] is True
    with pytest.raises(RuntimeError, match="certificate changed"):
        worker.seed_projection_certificate_v23a_r1(
            seeds, "0" * 64, prereg_v23a.canonical_sha256(mapping)
        )


@pytest.mark.skipif(
    os.environ.get("V23A_R1_RUN_CUDA_SEED_CONTRACT") != "1" or not torch.cuda.is_available(),
    reason="explicit opt-in only; normal CPU contract tests must not launch a GPU",
)
def test_v23a_r1_cuda_full_seed_stream_differs_from_projected_stream():
    full = torch.Generator(device="cuda").manual_seed(FULL_SEED)
    projected = torch.Generator(device="cuda").manual_seed(LOW32_SEED)
    full_sample = torch.randn(64, device="cuda", generator=full)
    projected_sample = torch.randn(64, device="cuda", generator=projected)
    assert not torch.equal(full_sample, projected_sample)
