from __future__ import annotations

from collections import Counter, defaultdict
from types import SimpleNamespace

import pytest
import torch

import eggroll_es_mirrored_v66 as mirrored
import eggroll_es_worker_lora_v41a as state_v41a
import eggroll_es_worker_lora_v66 as worker_v66
from eggroll_es_worker_v3 import coefficient_sha256_v3
from test_eggroll_es_worker_lora_v41a import (
    CONFIG,
    WEIGHTS,
    FakePG,
    fake_manager,
    source_master,
)


def evaluation_payload():
    return mirrored.common_evaluation_payload_v66(
        prompt_block=[
            {"request_id": "row-a", "prompt_token_ids": [1, 2, 3]},
            {"request_id": "row-b", "prompt_token_ids": [4, 5]},
        ],
        decode_contract={
            "temperature": 0.0,
            "max_tokens": 64,
            "seed_source": "evaluation_seed",
        },
        judge_contract={
            "schema": "train-only-source-grounded-score-v1",
            "weights_sha256": "a" * 64,
        },
        evaluation_seed=2026071702,
    )


def mirrored_plan():
    return mirrored.mirrored_population_plan_v66(
        [101, 102, 103, 104, 105, 106, 107, 108],
        0.0006,
        evaluation_payload(),
    )


def make_v66_worker(monkeypatch):
    monkeypatch.setattr(state_v41a, "EXPECTED_BASE_ELEMENTS_V41A", 23)
    monkeypatch.setattr(state_v41a, "EXPECTED_BASE_BYTES_V41A", 46)
    master = source_master()
    manager = fake_manager(master)
    value = object.__new__(worker_v66.LoRAAdapterStateWorkerExtensionV66)
    value.device = torch.device("cpu")
    value.model_runner = SimpleNamespace(
        lora_manager=SimpleNamespace(_adapter_manager=manager)
    )
    value.inter_pg = FakePG(rank=0)
    installed = value.install_adapter_state_v41a(
        WEIGHTS,
        CONFIG,
        state_v41a.file_sha256_v41a(WEIGHTS),
        state_v41a.file_sha256_v41a(CONFIG),
    )
    return value, manager, installed


def test_exact_antithetic_noise_is_deterministic_elementwise_negation():
    tensor = torch.zeros((17, 19), dtype=torch.float32)
    plus = worker_v66.signed_noise_like_v66(
        tensor, "layer.lora_A.weight", 9182, 1, "cpu"
    )
    minus = worker_v66.signed_noise_like_v66(
        tensor, "layer.lora_A.weight", 9182, -1, "cpu"
    )
    replay = worker_v66.signed_noise_like_v66(
        tensor, "layer.lora_A.weight", 9182, 1, "cpu"
    )
    other_key = worker_v66.signed_noise_like_v66(
        tensor, "layer.lora_B.weight", 9182, 1, "cpu"
    )
    assert torch.equal(plus, replay)
    assert torch.equal(plus, minus.neg())
    assert not torch.equal(plus, other_key)


def test_plan_pairs_common_random_conditions_and_balances_every_rank():
    plan = mirrored_plan()
    assignments = [item for wave in plan["waves"] for item in wave]
    assert plan["direction_count"] == 8
    assert plan["signed_population_size"] == 16
    assert plan["wave_count"] == 4
    assert all({item["engine_rank"] for item in wave} == {0, 1, 2, 3}
               for wave in plan["waves"])
    assert Counter(item["engine_rank"] for item in assignments) == {
        0: 4, 1: 4, 2: 4, 3: 4,
    }
    for rank in range(4):
        signs = [item["sign"] for item in assignments
                 if item["engine_rank"] == rank]
        assert Counter(signs) == {1: 2, -1: 2}

    pairs = defaultdict(list)
    for assignment in assignments:
        pairs[assignment["pair_id"]].append(assignment)
    assert len(pairs) == 8
    for pair in pairs.values():
        assert {item["sign"] for item in pair} == {1, -1}
        assert len({item["direction_seed"] for item in pair}) == 1
        assert len({item["prompt_block_sha256"] for item in pair}) == 1
        assert len({item["decode_contract_sha256"] for item in pair}) == 1
        assert len({item["judge_contract_sha256"] for item in pair}) == 1
        assert len({item["evaluation_seed"] for item in pair}) == 1
        assert len({item["wave_index"] for item in pair}) == 1


@pytest.mark.parametrize(
    "seeds",
    ([1, 2, 3], [1, 2, 3, 4, 4], [1, 2, 3, 4, 5, 6]),
)
def test_plan_rejects_unbalanced_or_duplicate_direction_populations(seeds):
    with pytest.raises(ValueError):
        mirrored.mirrored_population_plan_v66(
            seeds, 0.001, evaluation_payload()
        )


def test_pair_difference_compiles_exact_central_difference_worker_algebra():
    plan = mirrored_plan()
    differences = [0.5, -1.25, 2.0, 0.0, 0.125, -0.25, 3.5, -4.0]
    rewards = []
    for wave in plan["waves"]:
        for assignment in wave:
            baseline = 10.0 + assignment["direction_index"]
            difference = differences[assignment["direction_index"]]
            reward = baseline + (difference / 2.0) * assignment["sign"]
            rewards.append({
                "pair_id": assignment["pair_id"],
                "sign": assignment["sign"],
                "direction_index": assignment["direction_index"],
                "direction_seed": assignment["direction_seed"],
                "evaluation_contract_sha256": assignment[
                    "evaluation_contract_sha256"
                ],
                "reward": reward,
            })
    update = mirrored.pair_difference_update_v66(
        plan, list(reversed(rewards)), learning_rate=0.0003
    )
    assert update["coefficients"] == differences
    assert update["direction_seeds"] == plan["direction_seeds"]
    assert update["worker_population_size"] == 8
    assert update["worker_alpha"] == pytest.approx(0.0003 / (2 * 0.0006))
    assert update["effective_noise_scale"] == pytest.approx(
        0.0003 / (2 * 8 * 0.0006)
    )
    epsilon = [2.0, -1.0, 0.5, 3.0, -2.5, 0.75, 1.25, -0.125]
    worker_delta = update["effective_noise_scale"] * sum(
        coefficient * noise
        for coefficient, noise in zip(differences, epsilon, strict=True)
    )
    direct_delta = 0.0003 / (2 * 8 * 0.0006) * sum(
        coefficient * noise
        for coefficient, noise in zip(differences, epsilon, strict=True)
    )
    assert worker_delta == direct_delta


class ImmediateHandle:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error


def resolve_immediate(handle):
    if handle.error is not None:
        raise handle.error
    return handle.value


def test_executor_submits_full_waves_shares_one_payload_and_restores_all():
    payload = evaluation_payload()
    plan = mirrored_plan()
    materialized = []
    evaluated = []
    restored = []
    payload_objects = []

    def submit_materialize(assignment):
        materialized.append((assignment["wave_index"], assignment["engine_rank"]))
        return ImmediateHandle({
            "pair_id": assignment["pair_id"],
            "direction_seed": assignment["direction_seed"],
            "sign": assignment["sign"],
            "evaluation_contract_sha256": assignment[
                "evaluation_contract_sha256"
            ],
        })

    def submit_evaluate(assignment, shared_payload):
        evaluated.append((assignment["wave_index"], assignment["engine_rank"]))
        payload_objects.append(shared_payload)
        return ImmediateHandle(
            assignment["direction_index"] + assignment["sign"] * 0.25
        )

    def submit_restore(rank, reason):
        restored.append((rank, reason))
        return ImmediateHandle({"restored": True, "terminal_poisoned": False})

    result = mirrored.execute_mirrored_plan_v66(
        plan,
        payload,
        submit_materialize,
        submit_evaluate,
        submit_restore,
        resolve_immediate,
    )
    assert result["all_submitted_work_drained"] is True
    assert len(result["signed_rewards"]) == 16
    assert len(result["restore_receipts"]) == 16
    assert all(value is payload_objects[0] for value in payload_objects)
    for wave_index in range(4):
        assert {rank for wave, rank in materialized if wave == wave_index} == {
            0, 1, 2, 3,
        }
        assert {rank for wave, rank in evaluated if wave == wave_index} == {
            0, 1, 2, 3,
        }
        assert {rank for rank, reason in restored
                if reason == f"wave_{wave_index}_finalize"} == {0, 1, 2, 3}


def test_executor_drains_partial_failure_then_restores_every_actor_before_raise():
    payload = evaluation_payload()
    plan = mirrored_plan()
    resolved = []
    restored = []

    def submit_materialize(assignment):
        return ImmediateHandle({
            "pair_id": assignment["pair_id"],
            "direction_seed": assignment["direction_seed"],
            "sign": assignment["sign"],
            "evaluation_contract_sha256": assignment[
                "evaluation_contract_sha256"
            ],
        })

    def submit_evaluate(assignment, _shared_payload):
        error = RuntimeError("synthetic interrupted generation") \
            if assignment["engine_rank"] == 1 else None
        return ImmediateHandle(1.0, error)

    def submit_restore(rank, reason):
        restored.append((rank, reason))
        return ImmediateHandle({"restored": True, "terminal_poisoned": False})

    def resolving(handle):
        resolved.append(handle)
        return resolve_immediate(handle)

    with pytest.raises(
        mirrored.MirroredExecutionErrorV66,
        match="all four actors restored",
    ):
        mirrored.execute_mirrored_plan_v66(
            plan,
            payload,
            submit_materialize,
            submit_evaluate,
            submit_restore,
            resolving,
        )
    # Four materializations, four evaluations (including handles after the
    # failing one), then four restores were all resolved in the failed wave.
    assert len(resolved) == 12
    assert {rank for rank, _reason in restored} == {0, 1, 2, 3}


def test_worker_materializes_both_signs_and_restores_exact_master(monkeypatch):
    value, _manager, installed = make_v66_worker(monkeypatch)
    master_sha = installed["canonical_identity"]["sha256"]
    contract_sha = evaluation_payload()["contract"][
        "evaluation_contract_sha256"
    ]
    pair_id = "b" * 64
    candidates = []
    for sign in (1, -1):
        receipt = value.materialize_mirrored_adapter_v66(
            7721, 0.0006, sign, pair_id, contract_sha, master_sha
        )
        candidates.append(receipt["candidate_identity"]["sha256"])
        restored = value.restore_mirrored_adapter_v66(
            master_sha, f"sign_{sign}", pair_id
        )
        assert restored["restored_identity"]["sha256"] == master_sha
        assert restored["algebraic_bf16_restore_used"] is False
    assert candidates[0] != candidates[1]
    # Cleanup is deliberately idempotent when a materialize RPC may have failed
    # before returning a receipt to its controller.
    retry = value.restore_mirrored_adapter_v66(
        master_sha, "uncertain_rpc_readback"
    )
    assert retry["restored"] is True
    assert retry["prior_transaction"] is None


def test_pair_difference_receipt_feeds_current_four_rank_lora_update(monkeypatch):
    value, _manager, installed = make_v66_worker(monkeypatch)
    plan = mirrored_plan()
    rewards = []
    for wave in plan["waves"]:
        for assignment in wave:
            rewards.append({
                "pair_id": assignment["pair_id"],
                "sign": assignment["sign"],
                "direction_index": assignment["direction_index"],
                "direction_seed": assignment["direction_seed"],
                "evaluation_contract_sha256": assignment[
                    "evaluation_contract_sha256"
                ],
                "reward": float(assignment["sign"] * (
                    assignment["direction_index"] + 1
                )),
            })
    update = mirrored.pair_difference_update_v66(
        plan, rewards, learning_rate=0.0003
    )
    assert update["coefficient_sha256"] == coefficient_sha256_v3(
        update["direction_seeds"], update["coefficients"]
    )
    prepared = value.prepare_sharded_adapter_update_v41a(
        update["direction_seeds"],
        update["coefficients"],
        update["coefficient_sha256"],
        update["worker_population_size"],
        4,
        update["worker_alpha"],
        "mirrored-v66-test-plan",
        installed["canonical_identity"]["sha256"],
        value._v41_reference_generation,
    )
    assert prepared["rank"] == 0
    assert prepared["shard_indices"] == [0, 4]
    assert prepared["shard_seeds"] == [101, 105]
    rollback = value.abort_sharded_adapter_update_v41a(
        prepared["manifest_sha256"]
    )
    assert rollback["rolled_back"] is True


def test_partial_candidate_write_is_repaired_before_original_error_escapes(
    monkeypatch,
):
    value, manager, installed = make_v66_worker(monkeypatch)
    master_sha = installed["canonical_identity"]["sha256"]
    original_runtime_sha = installed["materialization"][
        "runtime_values_sha256"
    ]
    contract_sha = evaluation_payload()["contract"][
        "evaluation_contract_sha256"
    ]
    real_materialize = value._materialize_v41a
    calls = 0

    def partial_once(tensors, phase):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_name = sorted(manager.modules)[0]
            manager.modules[first_name].lora_a_stacked[0][0, 0].fill_(7)
            raise RuntimeError("synthetic partial runtime write")
        return real_materialize(tensors, phase)

    value._materialize_v41a = partial_once
    with pytest.raises(RuntimeError, match="synthetic partial runtime write"):
        value.materialize_mirrored_adapter_v66(
            8821, 0.0006, 1, "c" * 64, contract_sha, master_sha
        )
    assert calls == 2
    assert value._v41_active_perturbation is None
    assert value._v66_candidate_transaction is None
    assert getattr(value, "_v66_terminal_poison", None) is None
    assert value._v41_active_materialization[
        "runtime_values_sha256"
    ] == original_runtime_sha
    certificate = value.mirrored_adapter_state_certificate_v66()
    assert certificate["current_identity"]["sha256"] == master_sha
    assert certificate["terminal_poisoned"] is False


def test_restore_failure_terminally_poisoned_blocks_inherited_state(monkeypatch):
    value, _manager, installed = make_v66_worker(monkeypatch)
    master_sha = installed["canonical_identity"]["sha256"]
    contract_sha = evaluation_payload()["contract"][
        "evaluation_contract_sha256"
    ]
    value.materialize_mirrored_adapter_v66(
        9921, 0.0006, -1, "d" * 64, contract_sha, master_sha
    )

    def failed_restore(_tensors, _phase):
        raise RuntimeError("synthetic restore write failure")

    value._materialize_v41a = failed_restore
    with pytest.raises(RuntimeError, match="actor poisoned"):
        value.restore_mirrored_adapter_v66(master_sha, "injected_failure")
    assert value._v66_terminal_poison["requires_actor_recreation"] is True
    with pytest.raises(RuntimeError, match="terminally poisoned"):
        value.adapter_state_certificate_v41a()
