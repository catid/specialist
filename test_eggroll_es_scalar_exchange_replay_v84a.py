import copy
import json
from pathlib import Path

import pytest
import torch

import eggroll_es_fused_structured_runtime_v72 as fused_v72
import eggroll_es_scalar_exchange_replay_v84a as subject
import build_qwen36_lora_scalar_exchange_replay_preregistration_v84a as builder


V82B_PREREG = Path(
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_lora_collective_compression_v82b.json"
)


def source_records():
    value = json.loads(V82B_PREREG.read_text(encoding="utf-8"))
    return value["canonical_lora_update_scope"]["canonical_master"][
        "ordered_shape_manifest"
    ]


def pair_rows():
    return [
        {"seed": seed, "coefficient": coefficient}
        for seed, coefficient in zip(
            [101, 211, 307, 401, 503, 601, 701, 809],
            [0.75, -0.5, 0.25, -0.125, 0.0625, -0.03125, 0.015625, -0.0078125],
            strict=True,
        )
    ]


def local_shards(pairs=None):
    pairs = pair_rows() if pairs is None else pairs
    # Deliberately noncanonical local and rank order.  Consensus must sort by
    # rank for ownership and globally by seed for replay arithmetic.
    return [
        subject.build_local_shard_v84a(2, [pairs[5], pairs[4]]),
        subject.build_local_shard_v84a(0, [pairs[1], pairs[0]]),
        subject.build_local_shard_v84a(3, [pairs[7], pairs[6]]),
        subject.build_local_shard_v84a(1, [pairs[3], pairs[2]]),
    ]


def plan(method="iid_absolute_index", structured_rank=None, pairs=None):
    pairs = pair_rows() if pairs is None else pairs
    return subject.build_update_plan_v84a(
        seeds=sorted(row["seed"] for row in pairs),
        pairs=pairs,
        method=method,
        sigma=0.0006,
        step_size=0.0005,
        structured_rank=structured_rank,
        population_acceptance_sha256="a" * 64,
        expected_master_sha256="b" * 64,
        plan_id="synthetic-v84a-plan",
        update_sequence=7,
    )


def consensus(method="iid_absolute_index", structured_rank=None):
    return subject.seal_consensus_v84a(
        local_shards(), plan(method, structured_rank)
    )


def master():
    # Rank 16 is legal while keeping the absolute-index oracle cheap.
    value = torch.arange(16 * 16, dtype=torch.float32).reshape(16, 16)
    return {"synthetic.matrix": value.mul_(1.0e-4).contiguous()}


def test_exact_network_rng_scratch_and_hbm_tradeoff_is_source_bound():
    value = subject.byte_rng_accounting_v84a(source_records())
    network = value["network_projection"]
    hbm = value[
        "explicit_hbm_projection_excluding_rng_and_collective_internals"
    ]

    assert network["native_collective_calls_per_actor"] == 70
    assert network["native_nominal_ring_bus_bytes_per_actor"] == 27_168_768
    assert network["pair_allgather_bus_bytes_per_actor"] == 96
    assert network["digest_allgather_bus_bytes_per_actor"] == 96
    assert network["scalar_total_bus_bytes_per_actor"] == 192
    assert network["scalar_total_bus_bytes_all_actors"] == 768
    assert network["nominal_bus_bytes_saved_per_actor"] == 27_168_576
    assert network["native_to_scalar_nominal_bus_ratio"] == 141_504

    assert hbm["native_source_equivalent_bytes_per_actor"] == 217_350_144
    assert hbm["replay_source_equivalent_bytes_per_actor"] == 941_850_624
    assert hbm["replay_incremental_bytes_per_actor"] == 724_500_480
    assert hbm["replay_source_equivalent_multiplier"] == pytest.approx(13 / 3)
    assert hbm["fused_no_dense_noise_accumulator_bytes_per_actor"] == 652_050_432
    assert hbm[
        "ideal_fused_final_update_write_plus_d2h_read_floor_bytes_per_actor"
    ] == 36_225_024

    rng = value["rng_and_canonical_work"]
    iid = rng["iid_absolute_index:rank=None"]
    assert iid["native_balanced_rng_normals_per_actor"] == 9_056_256
    assert iid["replay_rng_normals_per_actor"] == 36_225_024
    assert iid["replay_rng_normals_all_actors"] == 144_900_096
    assert iid["replay_rng_multiplier_per_actor_and_all_actors"] == 4.0
    expected = {
        1: (287_488, 1_149_952, 295_040),
        4: (1_149_952, 4_599_808, 393_728),
        8: (2_299_904, 9_199_616, 525_312),
        16: (4_599_808, 18_399_232, 788_480),
    }
    for rank, (native, replay, scratch) in expected.items():
        row = rng[f"structured_outer_product:rank={rank}"]
        assert row["native_balanced_rng_normals_per_actor"] == native
        assert row["replay_rng_normals_per_actor"] == replay
        assert row["maximum_streamed_update_scratch_bytes"] == scratch
        assert row["replay_rng_multiplier_per_actor_and_all_actors"] == 4.0

    assert value["scratch_and_residency"][
        "whole_surface_noise_elements_allocated"
    ] == 0
    assert value["authority"]["live_scalar_arm_authorized"] is False
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    assert value["content_sha256"] == subject.canonical_sha256_v84a(body)


def test_antithetic_reward_collapse_is_exact_and_fail_closed():
    signed = []
    for index, seed in enumerate([101, 211, 307, 401, 503, 601, 701, 809]):
        signed.extend([
            {"seed": seed, "sign": -1, "reward": -0.25 * index},
            {"seed": seed, "sign": 1, "reward": 1.0 + 0.5 * index},
        ])
    collapsed = subject.collapse_antithetic_rewards_v84a(list(reversed(signed)))
    for row in collapsed:
        source = [item for item in signed if item["seed"] == row["seed"]]
        epsilon = torch.tensor(0.375, dtype=torch.float32)
        signed_sum = sum(
            item["reward"] * item["sign"] * epsilon for item in source
        )
        assert signed_sum == row["coefficient"] * epsilon

    duplicate = copy.deepcopy(signed)
    duplicate[1]["sign"] = -1
    with pytest.raises(RuntimeError, match="duplicate antithetic sign"):
        subject.collapse_antithetic_rewards_v84a(duplicate)
    nonfinite = copy.deepcopy(signed)
    nonfinite[0]["reward"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        subject.collapse_antithetic_rewards_v84a(nonfinite)


def test_consensus_sorts_and_binds_every_rank_pair_and_retry_identity():
    value = consensus()
    assert [row["seed"] for row in value["proposal"]["ordered_pairs"]] \
        == sorted(row["seed"] for row in pair_rows())
    assert [row["rank"] for row in value["rank_digest_views"]] == [0, 1, 2, 3]
    assert value["all_rank_consensus"] is True
    assert subject.validate_consensus_v84a(value) == value

    reordered = subject.seal_consensus_v84a(
        list(reversed(local_shards())), plan()
    )
    assert reordered == value

    changed = subject.build_update_plan_v84a(
        seeds=[row["seed"] for row in pair_rows()],
        pairs=pair_rows(),
        method="iid_absolute_index",
        sigma=0.0006,
        step_size=0.0006,
        structured_rank=None,
        population_acceptance_sha256="a" * 64,
        expected_master_sha256="b" * 64,
        plan_id="synthetic-v84a-plan",
        update_sequence=7,
    )
    changed_consensus = subject.seal_consensus_v84a(local_shards(), changed)
    assert changed_consensus["content_sha256"] != value["content_sha256"]


@pytest.mark.parametrize("failure", ["duplicate", "missing", "nonfinite", "digest"])
def test_consensus_duplicate_missing_nonfinite_and_split_brain_fail_closed(failure):
    shards = local_shards()
    expected_plan = plan()
    views = None
    if failure == "duplicate":
        bad = pair_rows()
        bad[7] = copy.deepcopy(bad[0])
        shards = local_shards(bad)
    elif failure == "missing":
        shards[0] = copy.deepcopy(shards[0])
        shards[0]["pairs"].pop()
    elif failure == "nonfinite":
        with pytest.raises(ValueError, match="finite"):
            subject.build_local_shard_v84a(0, [(1, float("inf")), (2, 1.0)])
        return
    else:
        valid = subject.seal_consensus_v84a(shards, expected_plan)
        views = copy.deepcopy(valid["rank_digest_views"])
        views[2]["proposal_sha256"] = "f" * 64

    message = "duplicate|missing|two directions|consensus"
    with pytest.raises((RuntimeError, ValueError), match=message):
        subject.seal_consensus_v84a(shards, expected_plan, views)


@pytest.mark.parametrize(
    "method,structured_rank",
    [("iid_absolute_index", None)]
    + [("structured_outer_product", rank) for rank in (1, 4, 8, 16)],
)
def test_fake_four_rank_replay_matches_dense_reference_within_v72_ulp_gate(
    method, structured_rank
):
    source = master()
    shards = local_shards()
    update_plan = plan(method, structured_rank)
    sealed = subject.seal_consensus_v84a(shards, update_plan)
    dense = subject.native_dense_reference_v84a(source, shards, update_plan)
    identities = []
    for rank in range(4):
        transaction = subject.ScalarReplayTransactionV84A(
            source,
            sealed,
            rank,
            authority=subject.SYNTHETIC_AUTHORITY_V84A,
            chunk_elements=37,
        )
        receipt = transaction.execute_v84a(transaction.transaction_sha256)
        replay = transaction.pending_master
        certificate = fused_v72.validate_update_ulp_v72(
            replay, dense, maximum_ulp=subject.MAXIMUM_UPDATE_ULP_V84A
        )
        assert certificate["passed"] is True
        assert certificate["observed_maximum_ulp"] <= 2
        assert receipt["dense_full_noise_allocated"] is False
        assert receipt["whole_surface_noise_elements_allocated"] == 0
        assert receipt["whole_surface_update_elements_allocated"] == 0
        identities.append(receipt["pending_master_identity"])
    assert identities == [identities[0]] * 4


def test_transaction_restore_finalize_poison_and_stale_retry_semantics():
    source = master()
    sealed = consensus()

    restored = subject.ScalarReplayTransactionV84A(
        source, sealed, 0, authority=subject.SYNTHETIC_AUTHORITY_V84A,
        chunk_elements=31,
    )
    receipt = restored.execute_v84a(restored.transaction_sha256)
    restored.commit_provisional_v84a(
        receipt["pending_master_identity"]["sha256"]
    )
    result = restored.restore_v84a("synthetic rejection")
    assert result["restored"] is True
    assert restored.current_master is restored.original_master
    assert restored.phase == "restored_exact_original"

    finalized = subject.ScalarReplayTransactionV84A(
        source, sealed, 1, authority=subject.SYNTHETIC_AUTHORITY_V84A,
        chunk_elements=31,
    )
    receipt = finalized.execute_v84a(finalized.transaction_sha256)
    candidate_sha = receipt["pending_master_identity"]["sha256"]
    finalized.commit_provisional_v84a(candidate_sha)
    assert finalized.finalize_v84a(candidate_sha)["finalized"] is True

    poisoned = subject.ScalarReplayTransactionV84A(
        source, sealed, 2, authority=subject.SYNTHETIC_AUTHORITY_V84A,
        chunk_elements=31,
    )
    calls = 0

    def fail_second_chunk(_key, _start, _end):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic replay failure")

    original_identity = subject.mapping_identity_v84a(source)
    with pytest.raises(RuntimeError, match="synthetic replay failure"):
        poisoned.execute_v84a(
            poisoned.transaction_sha256, chunk_hook=fail_second_chunk
        )
    assert poisoned.phase == "terminal_poison"
    assert poisoned.current_master is poisoned.original_master
    assert subject.mapping_identity_v84a(source) == original_identity
    with pytest.raises(RuntimeError, match="stale replay/retry"):
        poisoned.execute_v84a(poisoned.transaction_sha256)

    wrong_resume = subject.ScalarReplayTransactionV84A(
        source, sealed, 3, authority=subject.SYNTHETIC_AUTHORITY_V84A
    )
    with pytest.raises(RuntimeError, match="transaction identity changed"):
        wrong_resume.execute_v84a("0" * 64)
    assert wrong_resume.phase == "ready"


def test_synthetic_helper_has_no_live_authority():
    with pytest.raises(RuntimeError, match="live authority is absent"):
        subject.ScalarReplayTransactionV84A(
            master(), consensus(), 0, authority="live"
        )
    value = subject.byte_rng_accounting_v84a(source_records())
    assert value["authority"] == {
        "synthetic_cpu_only": True,
        "gpu_or_model_opened": False,
        "dataset_or_evaluation_opened": False,
        "live_scalar_arm_authorized": False,
        "quality_or_speed_claim": False,
        "promotion_authorized": False,
    }


def test_preregistration_and_report_are_current_self_hashed_and_unauthorized():
    prereg_bytes, report_bytes = builder.expected_bytes()
    assert builder.OUTPUT.read_bytes() == prereg_bytes
    assert builder.REPORT.read_bytes() == report_bytes
    value = json.loads(prereg_bytes)
    body = {
        key: item
        for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    assert value["content_sha256_before_self_field"] \
        == subject.canonical_sha256_v84a(body)
    assert value["prospective_v73d_gate"] == {
        "accepted_v73d_profile_present": False,
        "dense_collective_ranked_top_three": False,
        "minimum_replicates_with_dense_collective_top_three": 2,
        "required_replicates": 3,
        "live_scalar_arm_authorized": False,
        "paired_live_implementation_or_benchmark_authorized": False,
        "rule": (
            "no live implementation, model load, GPU run, or paired benchmark "
            "unless an accepted V73D profile first ranks the canonical dense "
            "FP32 collective in the top three in at least two registered replicates"
        ),
    }
    assert value["decision"]["current_path"] == "retain_native_exact_fp32_v72"
    assert value["decision"]["promotion_authorized"] is False
    assert value["authority"]["dataset_training_eval_ood_holdout_or_probe_opened"] is False
