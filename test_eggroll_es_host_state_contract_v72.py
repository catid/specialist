import json

import pytest
import torch

import eggroll_es_audit_contract_v71 as audit_v71
import eggroll_es_host_state_contract_v72 as contract


def _identity(tensors):
    records = [{
        "key": key,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "elements": int(tensor.numel()),
        "sha256": audit_v71.tensor_sha256_v71(tensor),
    } for key, tensor in sorted(tensors.items())]
    return {
        "schema": "test-state",
        "sha256": audit_v71.canonical_sha256_v71(records),
        "tensor_count": len(records),
        "elements": sum(item["elements"] for item in records),
        "bytes": sum(item["elements"] * 4 for item in records),
        "tensors": records,
    }


def test_immutable_lease_aliases_without_clone_and_detects_normal_mutation():
    tensors = {"weight": torch.arange(8, dtype=torch.float32)}
    identity = _identity(tensors)
    lease = contract.ImmutableStateLeaseV72(
        "rollback", 3, tensors, identity
    )

    receipt = lease.cheap_certificate("prepare")
    assert lease.tensors is tensors
    assert receipt["aliases_owned_mapping"] is True
    assert receipt["ownership_clone_bytes"] == 0
    assert receipt["validation_clone_bytes"] == 0

    tensors["weight"][0].add_(1)
    with pytest.raises(RuntimeError, match="object/storage/version drifted"):
        lease.cheap_certificate("execute")


def test_immutable_lease_exactly_detects_version_bypassing_mutation():
    tensors = {"weight": torch.arange(8, dtype=torch.float32)}
    lease = contract.ImmutableStateLeaseV72(
        "rollback", 3, tensors, _identity(tensors)
    )
    version = tensors["weight"]._version
    tensors["weight"].data[0].add_(1)

    assert tensors["weight"]._version == version
    assert lease.cheap_certificate("transition")["ownership_clone_bytes"] == 0
    with pytest.raises(RuntimeError, match="exact content drifted"):
        lease.exact_certificate("commit")


def test_byte_and_rss_accounting_is_exact_for_four_actors():
    value = contract.state_residency_account_v72(4)
    master = contract.MASTER_BYTES_V72
    peak = value["peak"]
    copies = value["full_state_tensor_copy_passes_one_commit_lifecycle"]

    assert peak["baseline_bytes_per_actor"] == 7 * master
    assert peak["proposed_bytes_per_actor"] == 2 * master
    assert peak["saved_bytes_per_actor"] == 5 * master
    assert peak["aggregate_actor_rss_sum_saved_bytes"] == 20 * master
    assert copies["baseline"] == 26
    assert copies["proposed"] == 1
    assert copies["saved"] == 25
    assert copies["saved_copy_bytes_per_actor"] == 452_812_800
    assert copies["saved_copy_bytes_all_actors"] == 1_811_251_200
    assert value["steady_quiescent"][
        "aggregate_actor_rss_sum_saved_bytes"
    ] == 72_450_048


def test_contract_rejects_speculative_pinning_and_dense_sharing_claims():
    value = contract.build_contract_v72()
    assert value["pinning_decision"]["selected"] is False
    assert value["dense_full_weight_decision"]["implemented"] is False
    assert value["dense_full_weight_decision"][
        "four_actor_private_copy_bytes"
    ] == 575_229_163_264
    assert value["rules"]["partial_generation_cannot_be_published_or_adopted"]
    assert value["gpu_launch_performed"] is False
    assert value["dataset_or_protected_access_performed"] is False


def test_cli_writes_a_self_hashed_cpu_contract(tmp_path):
    output = tmp_path / "contract.json"
    value = contract.main(["--output", str(output)])
    observed = json.loads(output.read_text(encoding="utf-8"))
    assert observed == value
    content_sha = observed.pop("content_sha256_before_self_field")
    assert contract.canonical_sha256_v72(observed) == content_sha
