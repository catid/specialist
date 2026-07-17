import copy
import hashlib
import math
import random

import pytest
import torch

import eggroll_es_fused_structured_runtime_v72 as subject
import structured_es_oracle_v1 as oracle


METHODS = [
    ("iid_absolute_index", None),
    *[("structured_outer_product", rank) for rank in oracle.STRUCTURED_RANKS_V1],
]


def _sha(label):
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _packed_assignments():
    return [
        {
            "peft_key": "unit.packed.lora_A.weight",
            "runtime_module": "runtime.packed",
            "runtime_shape": [2, 3],
            "segment_count": 2,
            "segment_index": 0,
            "side": "A",
            "slice_index": 0,
            "slot": 0,
            "source_shape": [2, 3],
        },
        {
            "peft_key": "unit.packed.lora_A.weight",
            "runtime_module": "runtime.packed",
            "runtime_shape": [2, 3],
            "segment_count": 2,
            "segment_index": 1,
            "side": "A",
            "slice_index": 1,
            "slot": 0,
            "source_shape": [2, 3],
        },
        {
            "peft_key": "unit.packed.lora_B.weight",
            "runtime_module": "runtime.packed",
            "runtime_shape": [1, 2],
            "segment_count": 2,
            "segment_index": 0,
            "side": "B",
            "slice_index": 0,
            "slot": 0,
            "source_shape": [4, 2],
        },
        {
            "peft_key": "unit.packed.lora_B.weight",
            "runtime_module": "runtime.packed",
            "runtime_shape": [3, 2],
            "segment_count": 2,
            "segment_index": 1,
            "side": "B",
            "slice_index": 1,
            "slot": 0,
            "source_shape": [4, 2],
        },
    ]


def _packed_fixture():
    plan = subject.build_runtime_projection_manifest_v72(
        _packed_assignments(), b_scale=2.0
    )
    master = {
        "unit.packed.lora_A.weight": (
            torch.arange(6, dtype=torch.float32).reshape(2, 3) / 10.0
        ),
        "unit.packed.lora_B.weight": (
            torch.arange(8, dtype=torch.float32).reshape(4, 2) / 10.0
        ),
    }
    views = subject.allocate_runtime_views_v72(plan)
    return plan, master, views


def _single_fixture(shape=(19, 23)):
    assignments = [{
        "peft_key": "unit.matrix.lora_A.weight",
        "runtime_module": "runtime.matrix",
        "runtime_shape": list(shape),
        "segment_count": 1,
        "segment_index": 0,
        "side": "A",
        "slice_index": 0,
        "slot": 0,
        "source_shape": list(shape),
    }]
    plan = subject.build_runtime_projection_manifest_v72(assignments, b_scale=2.0)
    master = {
        "unit.matrix.lora_A.weight": torch.linspace(
            -0.25, 0.25, math.prod(shape), dtype=torch.float32
        ).reshape(shape)
    }
    views = subject.allocate_runtime_views_v72(plan)
    return plan, master, views


def _resign_manifest(value):
    value["content_sha256_before_self_field"] = subject.canonical_sha256_v72({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })
    return value


def _run_candidate(method, rank, world_size, chunk_elements, shard_order=None):
    plan, master, views = _single_fixture()
    runtime = subject.FusedStructuredRuntimeV72(master, views, plan)
    runtime.begin_candidate_v72(
        method=method,
        seed=1701,
        sigma=0.01,
        sign=-1,
        structured_rank=rank,
    )
    order = list(range(world_size)) if shard_order is None else list(shard_order)
    for shard in order:
        runtime.write_candidate_shard_v72(
            world_size=world_size, rank=shard, chunk_elements=chunk_elements
        )
    provisional = runtime.complete_candidate_write_v72()
    receipt = runtime.post_generation_exact_audit_v72()
    expected = master["unit.matrix.lora_A.weight"].view(-1).add(
        oracle.noise_chunk_v1(
            method, 1701, "unit.matrix.lora_A.weight", [19, 23],
            0, 19 * 23, rank,
        ),
        alpha=-0.01,
    ).to(torch.bfloat16).reshape(19, 23)
    assert torch.equal(views["runtime.matrix|A|slice=0|slot=0"], expected)
    assert provisional["reward_provisional"] is True
    assert receipt["reward_accepted"] is False
    return runtime, receipt


def test_projection_manifest_models_packed_a_duplication_and_b_splitting():
    plan, master, views = _packed_fixture()
    assert plan["source_tensor_count"] == 2
    assert plan["source_elements"] == 14
    assert plan["runtime_view_count"] == 4
    assert plan["runtime_elements"] == 20
    runtime = subject.FusedStructuredRuntimeV72(master, views, plan)
    assert runtime.phase == "quiescent"
    assert torch.equal(
        views["runtime.packed|A|slice=0|slot=0"],
        master["unit.packed.lora_A.weight"].to(torch.bfloat16),
    )
    assert torch.equal(
        views["runtime.packed|A|slice=0|slot=0"],
        views["runtime.packed|A|slice=1|slot=0"],
    )
    expected_b = master["unit.packed.lora_B.weight"].mul(2.0).to(torch.bfloat16)
    assert torch.equal(
        views["runtime.packed|B|slice=0|slot=0"], expected_b[:1]
    )
    assert torch.equal(
        views["runtime.packed|B|slice=1|slot=0"], expected_b[1:]
    )


@pytest.mark.parametrize("method,rank", METHODS)
def test_candidate_is_exact_across_full_chunk_and_shard_reconstruction(method, rank):
    runs = [
        _run_candidate(method, rank, 1, 437),
        _run_candidate(method, rank, 4, 17, [2, 0, 3, 1]),
        _run_candidate(method, rank, 7, 13, [6, 1, 4, 0, 5, 2, 3]),
    ]
    identities = {
        receipt["runtime_exact_audit"]["runtime_identity_sha256"]
        for _runtime, receipt in runs
    }
    assert len(identities) == 1
    for runtime, _receipt in runs:
        runtime.begin_restore_v72()
        for shard in reversed(range(4)):
            runtime.write_restore_shard_v72(
                world_size=4, rank=shard, chunk_elements=11
            )
        restored = runtime.finish_restore_v72()
        assert restored["restored"] is True
        assert runtime.phase == "quiescent"


def test_candidate_and_restore_byte_ledger_has_no_whole_surface_scratch():
    runtime, _receipt = _run_candidate(
        "structured_outer_product", 4, 4, 17, [3, 0, 2, 1]
    )
    runtime.begin_restore_v72()
    for shard in range(4):
        runtime.write_restore_shard_v72(world_size=4, rank=shard, chunk_elements=19)
    runtime.finish_restore_v72()
    ledger = runtime.transaction_receipt_v72()["byte_ledger"]
    assert ledger["candidate_runtime_write_bytes"] == 19 * 23 * 2
    assert ledger["post_generation_exact_audit_d2h_bytes"] == 19 * 23 * 2
    assert ledger["restore_runtime_write_bytes"] == 19 * 23 * 2
    assert ledger["restore_exact_audit_d2h_bytes"] == 19 * 23 * 2
    assert ledger["maximum_candidate_scratch_bytes"] < 19 * 23 * 4
    assert ledger["whole_surface_noise_elements_allocated"] == 0
    assert ledger["whole_surface_candidate_elements_allocated"] == 0


def test_duplicate_candidate_shard_is_terminally_poisoned():
    plan, master, views = _single_fixture()
    runtime = subject.FusedStructuredRuntimeV72(master, views, plan)
    runtime.begin_candidate_v72(
        method="iid_absolute_index", seed=1, sigma=0.01, sign=1
    )
    runtime.write_candidate_shard_v72(world_size=4, rank=0, chunk_elements=13)
    with pytest.raises(RuntimeError, match="overlap"):
        runtime.write_candidate_shard_v72(world_size=4, rank=0, chunk_elements=13)
    assert runtime.poisoned is True
    assert runtime.phase == "terminal_poison"
    with pytest.raises(RuntimeError, match="terminally poisoned"):
        runtime.begin_candidate_v72(
            method="iid_absolute_index", seed=2, sigma=0.01, sign=1
        )


def test_candidate_gap_is_rejected_and_terminally_poisoned():
    plan, master, views = _single_fixture()
    runtime = subject.FusedStructuredRuntimeV72(master, views, plan)
    runtime.begin_candidate_v72(
        method="iid_absolute_index", seed=1, sigma=0.01, sign=1
    )
    runtime.write_candidate_shard_v72(world_size=4, rank=0)
    with pytest.raises(RuntimeError, match="coverage is incomplete"):
        runtime.complete_candidate_write_v72()
    assert runtime.poisoned is True


def test_world_size_change_mid_candidate_is_terminal():
    plan, master, views = _single_fixture()
    runtime = subject.FusedStructuredRuntimeV72(master, views, plan)
    runtime.begin_candidate_v72(
        method="iid_absolute_index", seed=1, sigma=0.01, sign=1
    )
    runtime.write_candidate_shard_v72(world_size=4, rank=0)
    with pytest.raises(RuntimeError, match="world size changed"):
        runtime.write_candidate_shard_v72(world_size=7, rank=1)
    assert runtime.poisoned is True


def test_post_generation_corruption_is_detected_before_reward_acceptance():
    plan, master, views = _single_fixture()
    runtime = subject.FusedStructuredRuntimeV72(master, views, plan)
    runtime.begin_candidate_v72(
        method="structured_outer_product", seed=1701, sigma=0.01, sign=1,
        structured_rank=4,
    )
    for shard in range(4):
        runtime.write_candidate_shard_v72(world_size=4, rank=shard, chunk_elements=17)
    runtime.complete_candidate_write_v72()
    key = "runtime.matrix|A|slice=0|slot=0"
    views[key].view(-1)[7] = views[key].view(-1)[7] + torch.tensor(
        1.0, dtype=torch.bfloat16
    )
    with pytest.raises(RuntimeError, match="version drifted|differs from stream"):
        runtime.post_generation_exact_audit_v72()
    assert runtime.poisoned is True


def test_unversioned_runtime_corruption_is_caught_by_stream_digest():
    plan, master, views = _single_fixture()
    runtime = subject.FusedStructuredRuntimeV72(master, views, plan)
    runtime.begin_candidate_v72(
        method="iid_absolute_index", seed=1701, sigma=0.01, sign=1
    )
    runtime.write_candidate_shard_v72(world_size=1, rank=0, chunk_elements=17)
    runtime.complete_candidate_write_v72()
    key = "runtime.matrix|A|slice=0|slot=0"
    views[key].data.view(-1)[7] = views[key].data.view(-1)[7] + torch.tensor(
        1.0, dtype=torch.bfloat16
    )
    with pytest.raises(RuntimeError, match="differs from stream"):
        runtime.post_generation_exact_audit_v72()
    assert runtime.poisoned is True


def test_partial_unknown_candidate_repairs_exactly():
    plan, master, views = _packed_fixture()
    runtime = subject.FusedStructuredRuntimeV72(master, views, plan)
    canonical = {key: value.clone() for key, value in views.items()}
    runtime.begin_candidate_v72(
        method="structured_outer_product", seed=1701, sigma=0.01, sign=-1,
        structured_rank=1,
    )
    runtime.write_candidate_shard_v72(world_size=4, rank=0, chunk_elements=2)
    receipt = runtime.repair_after_uncertain_candidate_v72(
        world_size=4, chunk_elements=3
    )
    assert receipt["restored"] is True
    assert runtime.poisoned is False
    assert runtime.phase == "quiescent"
    assert all(torch.equal(views[key], canonical[key]) for key in canonical)


def test_failed_uncertain_restore_terminally_poisons():
    plan, master, views = _single_fixture()
    runtime = subject.FusedStructuredRuntimeV72(master, views, plan)
    runtime.begin_candidate_v72(
        method="iid_absolute_index", seed=1, sigma=0.01, sign=1
    )
    runtime.write_candidate_shard_v72(world_size=4, rank=0)
    with pytest.raises(RuntimeError, match="runtime poisoned"):
        runtime.repair_after_uncertain_candidate_v72(
            world_size=4, inject_failure_after_shards=2
        )
    assert runtime.poisoned is True
    assert runtime.transaction_receipt_v72()["oracle_transaction"]["poisoned"] is True


def test_master_corruption_is_caught_by_v71_cheap_invariant_and_poisoned():
    plan, master, views = _single_fixture()
    runtime = subject.FusedStructuredRuntimeV72(master, views, plan)
    master["unit.matrix.lora_A.weight"].view(-1)[0].add_(1.0)
    with pytest.raises(RuntimeError, match="object/storage/version drifted"):
        runtime.begin_candidate_v72(
            method="iid_absolute_index", seed=1, sigma=0.01, sign=1
        )
    assert runtime.poisoned is True


def test_runtime_storage_rebinding_is_caught_before_candidate():
    plan, master, views = _single_fixture()
    runtime = subject.FusedStructuredRuntimeV72(master, views, plan)
    key = next(iter(views))
    runtime.runtime_views[key] = runtime.runtime_views[key].clone()
    with pytest.raises(RuntimeError, match="object/storage/version drifted"):
        runtime.begin_candidate_v72(
            method="iid_absolute_index", seed=1, sigma=0.01, sign=1
        )
    assert runtime.poisoned is True


def test_manifest_rejects_tamper_gap_overlap_alias_and_wrong_shape():
    plan, _master, _views = _packed_fixture()
    tampered = copy.deepcopy(plan)
    tampered["runtime_bytes"] -= 2
    with pytest.raises(RuntimeError, match="identity changed"):
        subject.validate_runtime_projection_manifest_v72(tampered)

    gap = copy.deepcopy(plan)
    b = [item for item in gap["projections"] if item["side"] == "B"]
    b[1]["source_start"] += 1
    b[1]["runtime_shape"] = [2, 2]
    b[1]["source_end"] -= 1
    gap["runtime_elements"] -= 2
    gap["runtime_bytes"] -= 4
    with pytest.raises(RuntimeError, match="extent/scale|gap or overlap"):
        subject.validate_runtime_projection_manifest_v72(_resign_manifest(gap))

    unaligned = copy.deepcopy(plan)
    b = [item for item in unaligned["projections"] if item["side"] == "B"]
    b[0]["source_end"] = 3
    b[0]["runtime_shape"] = [1, 3]
    b[1]["source_start"] = 3
    b[1]["source_end"] = 8
    b[1]["runtime_shape"] = [1, 5]
    with pytest.raises(RuntimeError, match="extent/scale"):
        subject.validate_runtime_projection_manifest_v72(_resign_manifest(unaligned))

    duplicate = copy.deepcopy(_packed_assignments())
    duplicate[1]["slice_index"] = 0
    with pytest.raises(RuntimeError, match="duplicated"):
        subject.build_runtime_projection_manifest_v72(duplicate, b_scale=2.0)

    bad_a = copy.deepcopy(_packed_assignments())
    bad_a[0]["runtime_shape"] = [1, 3]
    with pytest.raises(RuntimeError, match="duplicate the full source"):
        subject.build_runtime_projection_manifest_v72(bad_a, b_scale=2.0)


@pytest.mark.parametrize("method,rank", METHODS)
def test_streamed_update_matches_independent_oracle_across_shards(method, rank):
    plan, master, _views = _single_fixture()
    kwargs = {
        "method": method,
        "seeds": [1704, 1701, 1703, 1702],
        "coefficients": [0.25, -0.75, 0.5, -0.125],
        "sigma": 0.03,
        "step_size": 0.002,
        "structured_rank": rank,
    }
    reference = subject.cpu_oracle_reference_update_v72(
        master, plan, **kwargs, chunk_elements=31
    )
    identities = set()
    for world_size, chunk, order in (
        (1, 437, [0]),
        (4, 17, [3, 1, 0, 2]),
        (7, 13, [4, 0, 6, 2, 1, 5, 3]),
    ):
        update = subject.StreamedMasterUpdateV72(
            master, plan, **kwargs,
            v71_update_acceptance_sha256=_sha("update-acceptance"),
        )
        for shard in order:
            update.write_shard_v72(
                world_size=world_size, rank=shard, chunk_elements=chunk
            )
        receipt = update.finish_v72()
        certificate = subject.validate_update_ulp_v72(
            update.pending_master, reference
        )
        assert certificate["observed_maximum_ulp"] == 0
        assert receipt["dense_noise_materialized"] is False
        assert receipt["dense_update_materialized"] is False
        assert receipt["committed"] is False
        identities.add(receipt["pending_master_identity"]["sha256"])
    assert len(identities) == 1


def test_update_seed_order_is_canonical():
    plan, master, _views = _single_fixture()
    pairs = [(9, 0.5), (2, -0.25), (7, 0.125)]
    outputs = []
    for ordered in (pairs, list(reversed(pairs))):
        update = subject.StreamedMasterUpdateV72(
            master, plan,
            method="structured_outer_product",
            seeds=[item[0] for item in ordered],
            coefficients=[item[1] for item in ordered],
            sigma=0.1,
            step_size=0.01,
            structured_rank=4,
            v71_update_acceptance_sha256=_sha("accept"),
        )
        for shard in range(4):
            update.write_shard_v72(world_size=4, rank=shard, chunk_elements=19)
        outputs.append(update.finish_v72()["pending_master_identity"]["sha256"])
    assert outputs[0] == outputs[1]


def test_update_gap_and_overlap_leave_original_exactly_repairable():
    plan, master, _views = _single_fixture()
    original = {key: value.clone() for key, value in master.items()}
    gap = subject.StreamedMasterUpdateV72(
        master, plan,
        method="iid_absolute_index", seeds=[1], coefficients=[0.5],
        sigma=0.1, step_size=0.01,
        v71_update_acceptance_sha256=_sha("accept"),
    )
    gap.write_shard_v72(world_size=4, rank=0)
    with pytest.raises(RuntimeError, match="coverage"):
        gap.finish_v72()
    aborted = gap.abort_v72("incomplete_rpc")
    assert aborted["pending_output_discarded"] is True
    assert all(torch.equal(master[key], original[key]) for key in master)

    overlap = subject.StreamedMasterUpdateV72(
        master, plan,
        method="iid_absolute_index", seeds=[1], coefficients=[0.5],
        sigma=0.1, step_size=0.01,
        v71_update_acceptance_sha256=_sha("accept"),
    )
    overlap.write_shard_v72(world_size=4, rank=0)
    with pytest.raises(RuntimeError, match="overlaps"):
        overlap.write_shard_v72(world_size=4, rank=0)
    assert overlap.abort_v72("duplicate_rpc")["terminal_poisoned"] is False


def test_update_fault_injection_discards_pending_and_preserves_master(monkeypatch):
    plan, master, _views = _single_fixture()
    original = {key: value.clone() for key, value in master.items()}
    update = subject.StreamedMasterUpdateV72(
        master, plan,
        method="iid_absolute_index", seeds=[1, 2], coefficients=[0.5, -0.2],
        sigma=0.1, step_size=0.01,
        v71_update_acceptance_sha256=_sha("accept"),
    )
    calls = 0
    real = oracle.noise_chunk_v1

    def fault(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected generator failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(oracle, "noise_chunk_v1", fault)
    with pytest.raises(RuntimeError, match="injected generator"):
        update.write_shard_v72(world_size=1, rank=0, chunk_elements=17)
    receipt = update.abort_v72("generator_failure")
    assert receipt["terminal_poisoned"] is False
    assert all(torch.equal(master[key], original[key]) for key in master)


def _finished_update():
    plan, master, _views = _single_fixture()
    update = subject.StreamedMasterUpdateV72(
        master, plan,
        method="structured_outer_product", seeds=[1, 2], coefficients=[0.5, -0.2],
        sigma=0.1, step_size=0.01, structured_rank=4,
        v71_update_acceptance_sha256=_sha("accept"),
    )
    for shard in [2, 0, 3, 1]:
        update.write_shard_v72(world_size=4, rank=shard, chunk_elements=17)
    update.finish_v72()
    return update


def test_update_commit_requires_v71_acceptance_commit_and_final_boundaries():
    update = _finished_update()
    with pytest.raises(RuntimeError, match="identity changed"):
        update.commit_provisional_v72(_sha("wrong"))
    provisional = update.commit_provisional_v72(_sha("accept"))
    assert provisional["accepted"] is False
    assert provisional["rollback_retained"] is True
    with pytest.raises(RuntimeError, match="final boundary is out of order"):
        update.finalize_v72(_sha("final"))
    commit = update.accept_commit_boundary_v72(_sha("commit"))
    assert commit["commit_accepted"] is True
    final = update.finalize_v72(_sha("final"))
    assert final["rollback_released"] is True
    assert update.phase == "finalized"


def test_provisional_commit_can_abort_to_exact_original():
    update = _finished_update()
    original_identity = subject._mapping_identity_v72(update.original_master)
    update.commit_provisional_v72(_sha("accept"))
    receipt = update.abort_v72("controller_consensus_failure")
    assert receipt["original_master_sha256"] == original_identity["sha256"]
    assert update.current_master is update.original_master
    assert update.poisoned is False


def test_corrupt_rollback_master_causes_terminal_poison():
    update = _finished_update()
    update.commit_provisional_v72(_sha("accept"))
    next(iter(update.original_master.values())).view(-1)[0].add_(1.0)
    with pytest.raises(RuntimeError, match="transaction poisoned"):
        update.abort_v72("rollback_corruption")
    assert update.poisoned is True


def test_update_ulp_gate_accepts_two_and_rejects_three():
    expected = {"x": torch.tensor([1.0, -1.0, 0.0], dtype=torch.float32)}
    two = {"x": expected["x"].clone()}
    for _ in range(2):
        two["x"][0] = torch.nextafter(
            two["x"][0], torch.tensor(float("inf"), dtype=torch.float32)
        )
    certificate = subject.validate_update_ulp_v72(two, expected)
    assert certificate["observed_maximum_ulp"] == 2
    three = {"x": two["x"].clone()}
    three["x"][0] = torch.nextafter(
        three["x"][0], torch.tensor(float("inf"), dtype=torch.float32)
    )
    with pytest.raises(RuntimeError, match="exceeds"):
        subject.validate_update_ulp_v72(three, expected)


def test_update_rejects_duplicate_seeds_nonfinite_and_bad_acceptance():
    plan, master, _views = _single_fixture()
    kwargs = {
        "master": master,
        "manifest": plan,
        "method": "iid_absolute_index",
        "seeds": [1, 1],
        "coefficients": [0.5, 0.25],
        "sigma": 0.1,
        "step_size": 0.01,
        "v71_update_acceptance_sha256": _sha("accept"),
    }
    with pytest.raises(ValueError, match="unique"):
        subject.StreamedMasterUpdateV72(**kwargs)
    kwargs["seeds"] = [1, 2]
    kwargs["coefficients"] = [0.5, float("nan")]
    with pytest.raises(ValueError, match="finite"):
        subject.StreamedMasterUpdateV72(**kwargs)
    kwargs["coefficients"] = [0.5, 0.25]
    kwargs["v71_update_acceptance_sha256"] = "bad"
    with pytest.raises(ValueError, match="SHA-256"):
        subject.StreamedMasterUpdateV72(**kwargs)
