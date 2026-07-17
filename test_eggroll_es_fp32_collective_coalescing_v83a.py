import copy
import json
from pathlib import Path

import pytest
import torch

import eggroll_es_fp32_collective_coalescing_v83a as subject


ROOT = Path(__file__).resolve().parent
V82B = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_lora_collective_compression_v82b.json"
)


def source_records():
    value = json.loads(V82B.read_text(encoding="ascii"))
    return value["canonical_lora_update_scope"]["canonical_master"][
        "ordered_shape_manifest"
    ]


def plan(choice="bounded_4mib"):
    return subject.build_bucket_plan_v83a(source_records(), choice)


def master_for(value):
    result = {}
    for bucket in value["buckets"]:
        for entry in bucket["entries"]:
            result[entry["key"]] = torch.full(
                entry["shape"],
                float(entry["source_ordinal"] % 13) / 8.0,
                dtype=torch.float32,
            )
    return result


def local_tensor(entry, rank):
    base = torch.arange(entry["elements"], dtype=torch.float32).remainder_(17)
    base.add_(float(entry["source_ordinal"] * 32 + rank))
    return base.reshape(entry["shape"]).contiguous()


def native_update(value, scale):
    result = {}
    for bucket in value["buckets"]:
        for entry in bucket["entries"]:
            reduced = torch.zeros(entry["shape"], dtype=torch.float32)
            for rank in range(subject.WORLD_SIZE_V83A):
                reduced.add_(local_tensor(entry, rank))
            result[entry["key"]] = reduced.mul(scale).contiguous()
    return result


class FakeFourRankPG:
    def __init__(self, value, rank, trace, fail_call=None, return_clone=False):
        self.plan = value
        self.rank = rank
        self.world_size = 4
        self.trace = trace
        self.fail_call = fail_call
        self.return_clone = return_clone
        self.calls = 0

    def all_reduce(self, tensor, *, out_tensor, stream):
        assert out_tensor is tensor
        assert stream == "synthetic-stream"
        bucket = self.plan["buckets"][self.calls]
        expected_local = torch.empty(bucket["elements"], dtype=torch.float32)
        expected_reduced = torch.zeros(bucket["elements"], dtype=torch.float32)
        for entry in bucket["entries"]:
            destination = slice(entry["bucket_start"], entry["bucket_end"])
            expected_local[destination].copy_(
                local_tensor(entry, self.rank).reshape(-1)
            )
            for rank in range(self.world_size):
                expected_reduced[destination].add_(
                    local_tensor(entry, rank).reshape(-1)
                )
        assert torch.equal(tensor, expected_local)
        call = self.calls
        self.calls += 1
        self.trace.append(("all_reduce", call, stream, tensor.data_ptr()))
        if call == self.fail_call:
            raise RuntimeError("synthetic collective failure")
        tensor.copy_(expected_reduced)
        return tensor.clone() if self.return_clone else tensor


class FakeEvent:
    def __init__(self, trace, fail):
        self.trace = trace
        self.fail = fail

    def record(self, stream):
        self.trace.append(("event_record", stream))
        if self.fail == "record":
            raise RuntimeError("synthetic event record failure")

    def synchronize(self):
        self.trace.append(("event_synchronize",))
        if self.fail == "synchronize":
            raise RuntimeError("synthetic event synchronize failure")


def make_transaction(value, rank=0, fail_call=None, return_clone=False, event_fail=None):
    trace = []
    communicator = FakeFourRankPG(
        value,
        rank,
        trace,
        fail_call=fail_call,
        return_clone=return_clone,
    )
    transaction = subject.ExactFP32CoalescedUpdateV83A(
        master_for(value),
        value,
        communicator,
        "synthetic-stream",
        lambda: FakeEvent(trace, event_fail),
        authority=subject.SYNTHETIC_AUTHORITY_V83A,
    )
    return transaction, communicator, trace


def test_all_sealed_choices_cover_ordered_surface_without_gaps_or_overlaps():
    expected = {
        "flat_all_18112512b": (1, 4_528_128, 18_112_512),
        "bounded_8mib": (3, 2_093_056, 8_372_224),
        "bounded_4mib": (5, 1_048_576, 4_194_304),
        "bounded_2mib": (10, 516_096, 2_064_384),
    }
    for choice, values in expected.items():
        value = plan(choice)
        assert subject.validate_bucket_plan_v83a(value) == value
        assert (
            value["coalesced_collective_calls"],
            value["maximum_bucket_elements"],
            value["maximum_bucket_bytes"],
        ) == values
        assert value["tensor_count"] == 70
        assert value["total_elements"] == 4_528_128
        assert value["total_bytes"] == 18_112_512
        assert value["ordered_key_sha256"] \
            == subject.EXPECTED_ORDERED_KEY_SHA256_V83A
        global_cursor = 0
        ordinals = []
        keys = []
        for bucket_ordinal, bucket in enumerate(value["buckets"]):
            assert bucket["ordinal"] == bucket_ordinal
            assert bucket["elements"] <= value["capacity_elements"]
            bucket_cursor = 0
            for entry in bucket["entries"]:
                assert entry["global_start"] == global_cursor
                assert entry["global_end"] == global_cursor + entry["elements"]
                assert entry["bucket_start"] == bucket_cursor
                assert entry["bucket_end"] == bucket_cursor + entry["elements"]
                global_cursor = entry["global_end"]
                bucket_cursor = entry["bucket_end"]
                ordinals.append(entry["source_ordinal"])
                keys.append(entry["key"])
            assert bucket_cursor == bucket["elements"]
        assert global_cursor == 4_528_128
        assert ordinals == list(range(70))
        assert len(keys) == len(set(keys)) == 70


def test_payload_staging_and_hbm_formulas_are_exact_and_not_speed_claims():
    for choice, _capacity in subject.BUCKET_CHOICES_V83A:
        value = plan(choice)
        accounting = value["byte_accounting"]
        assert accounting["fp32_payload_bytes_per_actor_per_update"] == 18_112_512
        assert accounting["nominal_ring_bus_bytes_per_actor_per_update"] == 27_168_768
        assert accounting["network_payload_change_versus_native_bytes"] == 0
        assert accounting["materialized_pack_read_plus_write_hbm_bytes"] \
            == 36_225_024
        assert accounting["materialized_gpu_unpack_read_plus_write_hbm_bytes"] \
            == 36_225_024
        assert accounting["conservative_pack_plus_gpu_unpack_hbm_bytes"] \
            == 72_450_048
        assert accounting["unchanged_d2h_source_hbm_read_bytes"] == 18_112_512
        assert accounting["direct_fill_extra_pack_or_gpu_unpack_hbm_bytes"] == 0
        assert accounting["direct_fill_zero_is_design_target_not_live_measurement"] \
            is True


@pytest.mark.parametrize(
    "mutation",
    ("global_gap", "bucket_overlap", "duplicate_key", "shape", "hash"),
)
def test_plan_mutations_fail_closed(mutation):
    value = plan()
    changed = copy.deepcopy(value)
    if mutation == "global_gap":
        changed["buckets"][0]["entries"][1]["global_start"] += 1
    elif mutation == "bucket_overlap":
        changed["buckets"][0]["entries"][1]["bucket_start"] -= 1
    elif mutation == "duplicate_key":
        changed["buckets"][0]["entries"][1]["key"] = changed["buckets"][0][
            "entries"
        ][0]["key"]
    elif mutation == "shape":
        changed["buckets"][0]["entries"][0]["shape"][0] += 1
    else:
        changed["content_sha256"] = "0" * 64
    with pytest.raises((RuntimeError, ValueError)):
        subject.validate_bucket_plan_v83a(changed)


def test_master_storage_alias_is_rejected():
    value = plan()
    master = master_for(value)
    entries = [entry for bucket in value["buckets"] for entry in bucket["entries"]]
    assert entries[0]["shape"] == entries[2]["shape"]
    master[entries[2]["key"]] = master[entries[0]["key"]]
    communicator = FakeFourRankPG(value, 0, [])
    with pytest.raises(RuntimeError, match="storage aliases"):
        subject.ExactFP32CoalescedUpdateV83A(
            master,
            value,
            communicator,
            "synthetic-stream",
            lambda: FakeEvent([], None),
            authority=subject.SYNTHETIC_AUTHORITY_V83A,
        )


@pytest.mark.parametrize("rank", range(4))
def test_fake_four_rank_exact_native_reduction_unpack_candidate_and_restore(rank):
    value = plan("bounded_4mib")
    transaction, communicator, trace = make_transaction(value, rank=rank)
    original_identity = transaction.original_identity
    update = native_update(value, scale=0.125)
    expected_candidate = {
        key: transaction.original_master[key].add(update[key]).contiguous()
        for key in transaction.original_master
    }
    receipt = transaction.execute_v83a(
        lambda entry: local_tensor(entry, rank), scale=0.125
    )
    assert communicator.calls == value["coalesced_collective_calls"]
    assert receipt["collective_dtype"] == "torch.float32"
    assert receipt["update_identity"] == subject.mapping_identity_v83a(update)
    assert receipt["candidate_identity"] \
        == subject.mapping_identity_v83a(expected_candidate)
    assert all(
        torch.equal(transaction.pending_candidate[key], expected_candidate[key])
        for key in expected_candidate
    )
    assert all(
        transaction.pending_candidate[key].data_ptr()
        != transaction.original_master[key].data_ptr()
        for key in transaction.original_master
    )
    expected_trace_names = []
    for _bucket in value["buckets"]:
        expected_trace_names.extend(
            ["all_reduce", "event_record", "event_synchronize"]
        )
    assert [row[0] for row in trace] == expected_trace_names
    commit = transaction.commit_provisional_v83a(
        receipt["candidate_identity"]["sha256"]
    )
    assert commit["committed"] is True
    assert transaction.current_master is transaction.pending_candidate
    restored = transaction.restore_v83a("outer_controller_rejected_candidate")
    assert restored["restored_identity"] == original_identity
    assert transaction.current_master is transaction.original_master
    assert subject.mapping_identity_v83a(transaction.current_master) == original_identity
    with pytest.raises(RuntimeError, match="stale transaction"):
        transaction.execute_v83a(
            lambda entry: local_tensor(entry, rank), scale=0.125
        )


@pytest.mark.parametrize(
    "mode",
    ("source_alias", "collective", "return_alias", "event_record", "event_sync"),
)
def test_partial_failure_preserves_original_poisons_and_rejects_stale_retry(mode):
    value = plan("bounded_2mib")
    transaction, _communicator, _trace = make_transaction(
        value,
        rank=0,
        fail_call=1 if mode == "collective" else None,
        return_clone=mode == "return_alias",
        event_fail=(
            "record" if mode == "event_record" else
            "synchronize" if mode == "event_sync" else None
        ),
    )
    original = transaction.original_identity
    first_key = next(iter(transaction.original_master))

    def producer(entry):
        if mode == "source_alias" and entry["key"] == first_key:
            return transaction.original_master[first_key]
        return local_tensor(entry, 0)

    with pytest.raises(RuntimeError):
        transaction.execute_v83a(producer, scale=0.125)
    assert transaction.phase == "poisoned"
    assert transaction.pending_candidate is None
    assert transaction.current_master is transaction.original_master
    assert subject.mapping_identity_v83a(transaction.current_master) == original
    with pytest.raises(RuntimeError, match="stale transaction"):
        transaction.execute_v83a(producer, scale=0.125)


def test_synthetic_helper_has_no_live_authority_or_cuda_fallback():
    value = plan()
    communicator = FakeFourRankPG(value, 0, [])
    with pytest.raises(RuntimeError, match="live authority is absent"):
        subject.ExactFP32CoalescedUpdateV83A(
            master_for(value),
            value,
            communicator,
            "synthetic-stream",
            lambda: FakeEvent([], None),
            authority="live",
        )
