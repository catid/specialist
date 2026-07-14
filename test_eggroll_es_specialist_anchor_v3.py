import json
import math
from types import SimpleNamespace

import pytest
import torch

from eggroll_es_worker_v3 import (
    DistributedExactAuditWorkerExtensionV3,
    accumulate_seed_terms_v3,
    canonical_sha256_v3,
    coefficient_sha256_v3,
    seed_shard_v3,
    update_manifest_v3,
    validate_seed_coefficients_v3,
)
import run_eggroll_es_anchor_line_search_v3 as line_search_v3
import train_eggroll_es_specialist_anchor_v3 as anchor_v3
from train_eggroll_es_specialist_anchor_v3 import (
    DistributedAnchoredStepMixinV3,
    validate_executed_updates_v3,
    validate_prepared_shards_v3,
)


def test_four_seed_shards_are_balanced_disjoint_and_serially_equivalent():
    parameter = torch.nn.Parameter(torch.tensor(
        [0.25, -0.5, 1.0, 2.0], dtype=torch.bfloat16,
    ))
    seeds = [11, 22, 33, 44, 55, 66, 77, 88]
    coefficients = [0.5, -0.25, 1.0, -1.5, 0.75, 0.1, -0.4, 0.8]
    shards = [
        seed_shard_v3(seeds, coefficients, rank, 4)
        for rank in range(4)
    ]
    assert [shard["indices"] for shard in shards] == [
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    distributed = sum((
        accumulate_seed_terms_v3(
            parameter, shard["seeds"], shard["coefficients"],
        )
        for shard in shards
    ), torch.zeros_like(parameter, dtype=torch.float32))
    serial = accumulate_seed_terms_v3(parameter, seeds, coefficients)
    assert torch.allclose(distributed, serial, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "seeds,coefficients,population,match",
    [
        ([1, 2, 3, 4, 5, 6], [1.0] * 6, 6, "divisible"),
        ([1] * 8, [1.0] * 8, 8, "unique"),
        (list(range(8)), [1.0] * 7 + [math.inf], 8, "finite"),
    ],
)
def test_seed_coefficient_validation_fails_closed(
    seeds, coefficients, population, match,
):
    with pytest.raises(ValueError, match=match):
        validate_seed_coefficients_v3(
            seeds, coefficients, population, world_size=4,
        )


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(
            [0.00166, -0.25, 1.0], dtype=torch.bfloat16,
        ))


class FakeInterEngineGroup:
    def __init__(self, rank=0, world_size=4):
        self.rank = rank
        self.world_size = world_size
        self.available = True
        self.disabled = False
        self.calls = []

    def all_reduce(self, tensor, out_tensor=None, stream=None):
        self.calls.append((tensor.dtype, stream))
        assert out_tensor is tensor
        return out_tensor


def make_worker(rank=0):
    worker = object.__new__(DistributedExactAuditWorkerExtensionV3)
    worker.model_runner = SimpleNamespace(model=TinyModel())
    worker.inter_pg = FakeInterEngineGroup(rank=rank)
    worker.world_size = 1
    worker.save_self_exact_reference(chunk_bytes=2)
    return worker


def prepare_worker(worker, target_alpha=0.1):
    seeds = [11, 22, 33, 44, 55, 66, 77, 88]
    coefficients = [0.5, -0.25, 1.0, -1.5, 0.75, 0.1, -0.4, 0.8]
    digest = coefficient_sha256_v3(seeds, coefficients)
    report = worker.prepare_sharded_seed_update_v3(
        seeds,
        coefficients,
        digest,
        8,
        4,
        worker._v3_reference_generation,
        "plan-a",
        1,
        0.0,
        target_alpha,
        worker._v3_current_identity["sha256"],
    )
    return report


def test_worker_collective_update_keeps_reference_for_abort_but_marks_it_stale():
    worker = make_worker()
    initial = worker.model_runner.model.weight.detach().clone()
    prepared = prepare_worker(worker)
    assert prepared["shard_indices"] == [0, 4]
    executed = worker.execute_prepared_seed_update_v3(
        prepared["manifest_sha256"],
    )
    assert worker.inter_pg.calls == [(torch.float32, None)]
    committed = worker.commit_prepared_seed_update_v3(
        prepared["manifest_sha256"],
        executed["final_identity"]["sha256"],
    )
    assert committed["reference_fresh_for_population"] is False
    assert not torch.equal(worker.model_runner.model.weight, initial)
    with pytest.raises(RuntimeError, match="stale for population"):
        worker.restore_self_weights_exact()
    with pytest.raises(RuntimeError, match="stale for population"):
        worker.perturb_self_weights(123, 0.001)
    aborted = worker.abort_distributed_update_v3(
        "plan-a", worker._v3_reference_generation,
    )
    assert aborted["aborted"] is True
    assert torch.equal(worker.model_runner.model.weight, initial)


def test_worker_prepare_rehearses_allocation_without_rng_weights_or_collective():
    worker = make_worker()
    initial = worker.model_runner.model.weight.detach().clone()
    rng_before = torch.random.get_rng_state().clone()
    prepared = prepare_worker(worker)
    preflight = prepared["allocation_preflight"]
    assert preflight == {
        "schema": "eggroll-es-local-allocation-preflight-v3",
        "passed": True,
        "parameter_count": 1,
        "largest_parameter_name": "weight",
        "largest_parameter_shape": [3],
        "parameter_dtype": "torch.bfloat16",
        "accumulator_dtype": "torch.float32",
        "simulated_peak_temporary_bytes": 30,
        "scratch_freed_before_collectives": True,
        "collectives_created": False,
        "rng_consumed": False,
        "weights_changed": False,
    }
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert torch.equal(worker.model_runner.model.weight, initial)
    assert worker.inter_pg.calls == []


def test_worker_prepare_rejects_parameterless_model_before_collective():
    worker = make_worker()
    worker.model_runner.model = torch.nn.Module()
    with pytest.raises(RuntimeError, match="model has no parameters"):
        prepare_worker(worker)
    assert worker._v3_pending_update is None
    assert worker.inter_pg.calls == []


def test_worker_rejects_wrong_communicator_world_size_before_prepare():
    worker = make_worker()
    worker.inter_pg.world_size = 3
    with pytest.raises(RuntimeError, match="world size changed"):
        prepare_worker(worker)
    assert worker._v3_pending_update is None


def test_worker_rejects_changed_coefficient_identity_before_collective():
    worker = make_worker()
    seeds = [11, 22, 33, 44, 55, 66, 77, 88]
    coefficients = [0.5, -0.25, 1.0, -1.5, 0.75, 0.1, -0.4, 0.8]
    with pytest.raises(ValueError, match="coefficient identity changed"):
        worker.prepare_sharded_seed_update_v3(
            seeds,
            coefficients,
            "0" * 64,
            8,
            4,
            worker._v3_reference_generation,
            "plan-a",
            1,
            0.0,
            0.1,
            worker._v3_current_identity["sha256"],
        )
    assert worker._v3_pending_update is None


def prepared_reports(
    seeds, coefficients=None, *, manifest="manifest", base="base", sequence=1,
):
    if coefficients is None:
        coefficients = [0.1] * len(seeds)
    reports = []
    for rank in range(4):
        shard = seed_shard_v3(seeds, coefficients, rank, 4)
        reports.append({
            "prepared": True,
            "manifest_sha256": manifest,
            "rank": rank,
            "world_size": 4,
            "shard_indices": shard["indices"],
            "shard_seeds": shard["seeds"],
            "shard_pair_sha256": canonical_sha256_v3({
                "seeds": shard["seeds"],
                "coefficients": shard["coefficients"],
            }),
            "base_sha256": base,
            "reference_generation": 1,
            "update_sequence": sequence,
            "allocation_preflight": {
                "schema": "eggroll-es-local-allocation-preflight-v3",
                "passed": True,
                "parameter_count": 1,
                "largest_parameter_name": "weight",
                "largest_parameter_shape": [3],
                "parameter_dtype": "torch.bfloat16",
                "collectives_created": False,
                "rng_consumed": False,
                "weights_changed": False,
                "accumulator_dtype": "torch.float32",
                "simulated_peak_temporary_bytes": 30,
                "scratch_freed_before_collectives": True,
            },
        })
    return reports


def test_controller_rejects_overlapping_prepared_shards():
    seeds = list(range(8))
    coefficients = [0.1] * len(seeds)
    reports = prepared_reports(seeds, coefficients)
    reports[3]["shard_indices"] = list(reports[2]["shard_indices"])
    reports[3]["shard_seeds"] = list(reports[2]["shard_seeds"])
    with pytest.raises(RuntimeError, match="expected stride"):
        validate_prepared_shards_v3(
            reports, seeds, coefficients, "manifest", 1, "base", 1,
        )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("update_sequence", 2, "update sequences differ"),
        ("shard_pair_sha256", "0" * 64, "shard identity differs"),
    ],
)
def test_controller_binds_prepared_sequence_and_seed_coefficient_pair(
    field, value, match,
):
    seeds = list(range(8))
    coefficients = [0.1] * len(seeds)
    reports = prepared_reports(seeds, coefficients)
    reports[2][field] = value
    with pytest.raises(RuntimeError, match=match):
        validate_prepared_shards_v3(
            reports, seeds, coefficients, "manifest", 1, "base", 1,
        )


def executed_reports(final_hashes, manifest="manifest"):
    return [{
        "executed": True,
        "manifest_sha256": manifest,
        "world_size": 4,
        "rank": rank,
        "collective_dtype": "torch.float32",
        "parameter_count": 2,
        "reduced_element_count": 7,
        "final_identity": {
            "schema": "weights",
            "sha256": final_hashes[rank],
            "parameter_count": 2,
            "total_bytes": 14,
        },
    } for rank in range(4)]


def test_controller_requires_identical_final_weight_hashes():
    with pytest.raises(RuntimeError, match="hashes differ"):
        validate_executed_updates_v3(
            executed_reports(["same", "same", "same", "different"]),
            "manifest",
        )


class FakeCoordinator(DistributedAnchoredStepMixinV3):
    def __init__(self, mismatched_final=False, tampered_manifest=False):
        self.population_size = 8
        self._v3_reference_generation = 1
        self._v3_reference_identity = {"sha256": "reference"}
        self._v3_current_identity = {"sha256": "base"}
        self._v3_reference_fresh = True
        self._v3_update_sequence = 0
        self._v3_accepted_alpha = 0.0
        self._v3_active_plan_id = "plan-a"
        self.mismatched_final = mismatched_final
        self.tampered_manifest = tampered_manifest
        self.calls = []
        self.persisted = []

    def _persist_anchor_plan(self, plan):
        self.persisted.append(json.loads(json.dumps(plan)))

    def _rpc_all_engines_v3(self, method, args):
        self.calls.append(method)
        if method == "prepare_sharded_seed_update_v3":
            seeds = args[0]
            coefficients = args[1]
            base = args[-1]
            self.pending_target = args[-2]
            manifest = update_manifest_v3(
                coefficient_sha256=args[2],
                population_size=args[3],
                world_size=args[4],
                reference_generation=args[5],
                plan_id=args[6],
                update_sequence=args[7],
                previous_alpha=args[8],
                target_alpha=args[9],
                expected_base_sha256=args[10],
            )
            manifest_sha = canonical_sha256_v3(manifest)
            if self.tampered_manifest:
                manifest_sha = "0" * 64
            return prepared_reports(
                seeds,
                coefficients,
                manifest=manifest_sha,
                base=base,
                sequence=args[7],
            )
        if method == "execute_prepared_seed_update_v3":
            hashes = ["final"] * 4
            if self.mismatched_final:
                hashes[-1] = "drift"
            return executed_reports(hashes, manifest=args[0])
        if method == "commit_prepared_seed_update_v3":
            return [{
                "committed": True,
                "manifest_sha256": args[0],
                "final_sha256": "final",
                "reference_fresh_for_population": False,
                "update_sequence": self._v3_update_sequence + 1,
                "rank": rank,
            } for rank in range(4)]
        if method == "inspect_distributed_update_state_v3":
            return [{
                "pending": False,
                "communicator": {
                    "rank": rank,
                    "world_size": 4,
                    "tp_world_size": 1,
                    "available": True,
                    "disabled": False,
                },
                "reference_generation": 1,
                "reference_fresh_for_population": False,
                "reference_identity": {"sha256": "reference"},
                "current_identity": {
                    "schema": "weights",
                    "sha256": "final",
                    "parameter_count": 2,
                    "total_bytes": 14,
                },
                "update_session": "plan-a",
                "update_sequence": self._v3_update_sequence + 1,
                "accepted_alpha": self.pending_target,
            } for rank in range(4)]
        if method == "abort_distributed_update_v3":
            return [{
                "aborted": True,
                "restored_identity": {"sha256": "reference"},
            } for _ in range(4)]
        raise AssertionError(method)


def make_plan(trainer):
    seeds = list(range(8))
    coefficients = [0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 0.75, -0.75]
    plan = {
        "identity_audit": {"passed": True},
        "seeds": seeds,
        "coefficients": coefficients,
        "coefficient_sha256": coefficient_sha256_v3(seeds, coefficients),
        "distributed_update_v3": {
            "plan_id": "plan-a",
            "reference_generation": 1,
            "reference_identity": {"sha256": "reference"},
        },
        "applied_alpha": 0.0,
        "applications": [],
    }
    trainer._latest_anchor_plan = plan
    return plan


def test_resident_updates_use_all_engines_without_reference_recapture():
    trainer = FakeCoordinator()
    plan = make_plan(trainer)
    trainer.apply_seed_coefficients(plan, 0.1)
    # The fake final hash is the base identity for the second increment.
    assert trainer._v3_current_identity["sha256"] == "final"
    trainer.apply_seed_coefficients(plan, 0.2)
    assert trainer.calls == [
        "prepare_sharded_seed_update_v3",
        "execute_prepared_seed_update_v3",
        "commit_prepared_seed_update_v3",
        "inspect_distributed_update_state_v3",
    ] * 2
    assert "save_self_exact_reference" not in trainer.calls
    assert [item["update_sequence"] for item in plan["applications"]] == [1, 2]
    assert all(
        canonical_sha256_v3(item["manifest"])
        == item["manifest_sha256"]
        for item in plan["applications"]
    )
    assert all(
        item["reference_recaptured"] is False
        for item in plan["applications"]
    )


def test_final_hash_mismatch_aborts_to_retained_exact_reference():
    trainer = FakeCoordinator(mismatched_final=True)
    plan = make_plan(trainer)
    with pytest.raises(RuntimeError, match="hashes differ"):
        trainer.apply_seed_coefficients(plan, 0.1)
    assert trainer.calls[-1] == "abort_distributed_update_v3"
    assert plan["applied_alpha"] == 0.0
    assert plan["distributed_update_v3"]["last_failure"][
        "aborted_to_reference"
    ] is True


def test_controller_rejects_unanimous_but_wrong_worker_manifest_before_execute():
    trainer = FakeCoordinator(tampered_manifest=True)
    plan = make_plan(trainer)
    with pytest.raises(RuntimeError, match="controller expectation"):
        trainer.apply_seed_coefficients(plan, 0.1)
    assert trainer.calls == [
        "prepare_sharded_seed_update_v3",
        "abort_distributed_update_v3",
    ]


def test_v3_anchor_plan_persistence_is_atomic_and_keeps_provenance(
    tmp_path, monkeypatch,
):
    trainer = object.__new__(DistributedAnchoredStepMixinV3)
    trainer.logging_dir = str(tmp_path)
    trainer._pending_identity_audit = {
        "schema": "eggroll-es-alpha-zero-identity-audit-v2",
        "passed": True,
    }
    calls = []
    real_atomic_write = anchor_v3.anchor_v2._atomic_write_json

    def recording_atomic_write(path, value):
        calls.append((path, json.loads(json.dumps(value))))
        return real_atomic_write(path, value)

    monkeypatch.setattr(
        anchor_v3.anchor_v2, "_atomic_write_json", recording_atomic_write,
    )
    plan = {
        "schema": "eggroll-es-anchored-seed-plan-v1",
        "iteration": 2,
        "distributed_update_v3": {
            "schema": "eggroll-es-distributed-seed-plan-v3",
            "plan_id": "plan-a",
        },
    }
    trainer._persist_anchor_plan(plan)
    path = tmp_path / "anchor-plan-iteration-3.json"
    assert len(calls) == 1
    assert calls[0][0] == path
    assert json.loads(path.read_text()) == plan
    assert plan["identity_audit"]["passed"] is True
    assert plan["distributed_update_v3"]["plan_id"] == "plan-a"
    assert not path.with_name(path.name + ".tmp").exists()


class ProbeCoordinator(DistributedAnchoredStepMixinV3):
    def __init__(self):
        self.engines = [object(), object(), object(), object()]
        self.global_seed = 42

    def _sampling_params(self, **kwargs):
        return kwargs

    def _resolve(self, value):
        return value


def test_identity_probe_reports_truthful_nonempty_engine_coverage(monkeypatch):
    prompt_counts = []

    def fake_dispatch(engines, prompts, sampling, resolve):
        prompt_counts.append(len(prompts))
        return [SimpleNamespace(reward=index) for index in range(len(prompts))]

    monkeypatch.setattr(
        anchor_v3.anchor_v2.anchor_v1,
        "dispatch_eval_batch",
        fake_dispatch,
    )
    monkeypatch.setattr(anchor_v3.anchor_v2, "domain_output_sha256", lambda x: "d")
    monkeypatch.setattr(
        anchor_v3.anchor_v2, "anchor_output_sha256", lambda items, x: "a",
    )
    trainer = ProbeCoordinator()
    result = trainer._identity_probe(
        ["a", "b", "c", "d", "e"],
        object(),
        [
            {"prompt_token_ids": [1, 2]},
            {"prompt_token_ids": [3, 4]},
        ],
        0,
    )
    assert prompt_counts == [5, 2]
    assert result["dispatch"] == "strided_engine_shards_separate_calls"
    assert result["engine_coverage"] == {
        "configured_engines": 4,
        "domain_nonempty_engines": 4,
        "anchor_nonempty_engines": 2,
        "domain_uses_all_engines": True,
        "anchor_uses_all_engines": False,
    }


def test_identity_probe_rejects_domain_batch_that_leaves_a_gpu_idle():
    trainer = ProbeCoordinator()
    with pytest.raises(RuntimeError, match="cover all four engines"):
        trainer._identity_probe(["a", "b", "c"], object(), [], 0)


def test_resident_v3_wrapper_exposes_effective_anchor_api():
    assert line_search_v3.validate_effective_anchor_api() == (
        "coefficient_sha256", "load_anchor_prose", "load_trainer",
    )
    coefficients = [0.5, -0.5]
    assert anchor_v3.coefficient_sha256([11, 22], coefficients) == (
        coefficient_sha256_v3([11, 22], coefficients)
    )
    assert canonical_sha256_v3({"safe": 1})


def test_v3_snapshot_binds_inherited_v2_implementation_identities(monkeypatch):
    monkeypatch.setattr(
        line_search_v3,
        "_V1_BUILD_SNAPSHOT",
        lambda *args, **kwargs: {"implementation": {}},
    )
    snapshot = line_search_v3.build_snapshot()
    implementation = snapshot["implementation"]
    assert implementation["corrected_driver"] == anchor_v3.file_sha256(
        line_search_v3.Path(line_search_v3.driver_v2.__file__).resolve()
    )
    assert implementation["exact_worker"] == anchor_v3.file_sha256(
        anchor_v3.ROOT / "eggroll_es_worker_v2.py"
    )
    assert implementation["distributed_driver_v3"]
    assert implementation["distributed_trainer_v3"]
    assert implementation["distributed_worker_v3"]


def test_v3_wrapper_persists_canonical_coefficient_values():
    seeds = [11, 22, 33, 44]
    coefficients = [0.5, -0.5, 1.0, -1.0]
    coefficient_sha = coefficient_sha256_v3(seeds, coefficients)
    journal = {
        "seeds": list(seeds),
        "coefficient_plan": {"coefficient_sha256": coefficient_sha},
    }
    plan = {
        "seeds": list(seeds),
        "coefficients": list(coefficients),
        "coefficient_sha256": coefficient_sha,
    }
    assert line_search_v3.bind_coefficient_values_v3(journal, plan) == (
        coefficient_sha
    )
    assert journal["coefficient_plan"]["coefficients"] == coefficients


def test_v3_wrapper_rejects_seed_or_coefficient_identity_drift():
    seeds = [11, 22, 33, 44]
    coefficients = [0.5, -0.5, 1.0, -1.0]
    coefficient_sha = coefficient_sha256_v3(seeds, coefficients)
    plan = {
        "seeds": list(seeds),
        "coefficients": list(coefficients),
        "coefficient_sha256": coefficient_sha,
    }
    journal = {
        "seeds": [11, 22, 33, 99],
        "coefficient_plan": {"coefficient_sha256": coefficient_sha},
    }
    with pytest.raises(RuntimeError, match="journal seeds differ"):
        line_search_v3.bind_coefficient_values_v3(journal, plan)

    journal["seeds"] = list(seeds)
    plan["coefficients"][0] += 0.5
    with pytest.raises(RuntimeError, match="coefficient identity changed"):
        line_search_v3.bind_coefficient_values_v3(journal, plan)
