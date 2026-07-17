import copy
import json

import pytest
import torch

import eggroll_es_audit_contract_v71 as audit


def test_normal_one_element_base_mutation_fails_cheap_version_guard():
    tensors = {"base": torch.arange(8, dtype=torch.bfloat16)}
    registry = audit.TensorInvariantRegistryV71("base", tensors)
    assert registry.cheap_certificate(tensors, "before")["d2h_bytes"] == 0

    tensors["base"][3].add_(1)

    with pytest.raises(RuntimeError, match="object/storage/version drifted"):
        registry.cheap_certificate(tensors, "after_one_element_write")


def test_version_bypassing_one_element_base_mutation_fails_exact_boundary():
    tensors = {"base": torch.arange(8, dtype=torch.bfloat16)}
    registry = audit.TensorInvariantRegistryV71("base", tensors)
    version = tensors["base"]._version

    tensors["base"].data[3].add_(1)

    assert tensors["base"]._version == version
    assert registry.cheap_certificate(tensors, "transition")["passed"] is True
    with pytest.raises(RuntimeError, match="exact content drifted"):
        registry.exact_certificate(tensors, "population_reward_acceptance")


def test_object_storage_and_mapping_drift_are_independently_rejected():
    original = torch.arange(8, dtype=torch.float32)
    tensors = {"weight": original}
    registry = audit.TensorInvariantRegistryV71("master", tensors)

    tensors["weight"] = original.clone()
    with pytest.raises(RuntimeError, match="object/storage/version drifted"):
        registry.cheap_certificate(tensors, "object_replacement")

    tensors = {"weight": torch.arange(8, dtype=torch.float32)}
    registry = audit.TensorInvariantRegistryV71("master", tensors)
    replacement_storage = torch.arange(16, dtype=torch.float32)
    tensors["weight"].set_(replacement_storage.untyped_storage(), 0, (8,), (1,))
    with pytest.raises(RuntimeError, match="object/storage/version drifted"):
        registry.cheap_certificate(tensors, "storage_replacement")

    tensors = {"weight": torch.arange(8, dtype=torch.float32)}
    registry = audit.TensorInvariantRegistryV71("master", tensors)
    copied_mapping = dict(tensors)
    with pytest.raises(RuntimeError, match="object/key mapping drifted"):
        registry.cheap_certificate(copied_mapping, "mapping_replacement")


def test_owned_master_cache_has_zero_validation_clones_and_exactly_catches_data_write():
    tensors = {"master": torch.arange(16, dtype=torch.float32).reshape(4, 4)}
    cache = audit.OwnedMasterIdentityCacheV71(tensors)

    first = cache.cached_identity(tensors, "candidate_0")
    second = cache.cached_identity(tensors, "candidate_1")
    assert first["validation_clone_bytes"] == 0
    assert second["validation_clone_bytes"] == 0
    assert second["cache_hits"] == 2
    assert second["cheap_invariant"]["d2h_bytes"] == 0

    version = tensors["master"]._version
    tensors["master"].data[0, 0].add_(1)
    assert tensors["master"]._version == version
    assert cache.cached_identity(tensors, "candidate_2")["sha256"] == cache.sha256
    with pytest.raises(RuntimeError, match="exact content drifted"):
        cache.exact_audit(tensors, "update_acceptance")


def test_precomputed_master_identity_avoids_a_second_hash_pass(monkeypatch):
    tensors = {"master": torch.arange(4, dtype=torch.float32)}
    first = audit.OwnedMasterIdentityCacheV71(tensors)
    identity = {
        "sha256": first.sha256,
        "tensor_count": 1,
        "elements": 4,
        "bytes": 16,
        "tensors": [{
            "key": "master",
            "shape": [4],
            "dtype": "torch.float32",
            "elements": 4,
            "sha256": audit.tensor_sha256_v71(tensors["master"]),
        }],
    }

    def forbidden_hash(_tensor):
        raise AssertionError("precomputed identity unexpectedly re-read content")

    monkeypatch.setattr(audit, "tensor_sha256_v71", forbidden_hash)
    adopted = audit.OwnedMasterIdentityCacheV71(tensors, identity=identity)
    assert adopted.cached_identity(tensors, "adopt")["sha256"] == first.sha256


def test_schedule_keeps_every_acceptance_behind_its_exact_boundary():
    events = audit.canonical_audit_schedule_v71(4)
    certificate = audit.validate_audit_schedule_v71(events, 4)
    assert certificate["exact_base_boundaries"] == list(audit.EXACT_BOUNDARIES_V71)

    def position(event, **fields):
        return next(
            index for index, item in enumerate(events)
            if item["event"] == event
            and all(item.get(key) == value for key, value in fields.items())
        )

    assert position("exact_base", boundary="population_reward_acceptance") \
        < position("reward_acceptance")
    assert position("exact_base", boundary="update_acceptance") \
        < position("update_acceptance")
    assert position("commit_state_write") \
        < position("exact_base", boundary="commit") \
        < position("commit_acceptance")
    assert position("final_state_write") \
        < position("exact_base", boundary="final") \
        < position("final_acceptance")

    tampered = copy.deepcopy(events)
    exact = tampered.pop(position("exact_base", boundary="update_acceptance"))
    tampered.insert(position("update_acceptance"), exact)
    with pytest.raises(RuntimeError, match="acceptance moved early"):
        audit.validate_audit_schedule_v71(tampered, 4)


def test_traffic_account_includes_four_base_boundaries_and_fused_staging_cost():
    count = 16
    value = audit.traffic_account_v71(count)
    baseline = value["baseline"]
    proposed = value["proposed"]
    operations = value["operation_counts"]

    assert value["world_size"] == 4
    assert value["candidates_per_actor"] == 4
    assert operations["materializations"] == 2 * count + 3 * 4
    assert operations["logical_lora_verifications"] == 3 * count + 5 * 4
    assert operations["exact_base_boundaries_per_actor"] == 4
    assert proposed["base_exact_audits"] == 4 * 4
    assert proposed["base_d2h_bytes"] == 4 * 4 * audit.BASE_BYTES_V71
    assert proposed["lora_d2h_calls"] * 2 == baseline["lora_d2h_calls"]
    assert proposed["lora_d2h_bytes"] * 2 == baseline["lora_d2h_bytes"]
    assert proposed["h2d_bytes"] == baseline["h2d_bytes"]
    assert proposed["master_validation_host_copy_bytes"] == 0
    assert proposed["peak_fused_staging_vram_bytes"] \
        == audit.RUNTIME_LORA_BYTES_V71
    assert proposed["gpu_staging_read_write_bytes"] \
        == 2 * proposed["lora_d2h_bytes"]
    assert value["savings"]["device_transfer_bytes"] > 0
    assert value["savings"]["host_copy_or_device_transfer_bytes"] > 0
    measured = value["measured_v66d_current_path"]
    assert measured["base_exact_audits"] == 60
    assert measured["base_d2h_bytes"] == 17_159_946_240
    assert measured["lora_equality_plus_sha_d2h_calls"] == 120
    assert measured["lora_equality_plus_sha_d2h_bytes"] == 1_181_122_560


def test_contract_is_cpu_only_and_unknown_rpc_is_fail_closed():
    value = audit.build_contract_v71(4)
    assert value["status"] == "cpu_contract_no_gpu_or_protected_access"
    assert value["protected_dev_ood_or_holdout_opened"] is False
    assert value["gpu_launch_performed"] is False
    assert value["rules"]["unknown_or_partial_rpc_requires_exact_restore_or_poison"]
    assert value["rules"]["checkpoint_exact_boundary"] is True


def test_cli_writes_self_hashed_contract_without_gpu_or_protected_access(tmp_path):
    output = tmp_path / "contract.json"
    result = audit.main([
        "--candidate-count", "4",
        "--world-size", "4",
        "--output", str(output),
    ])
    observed = json.loads(output.read_text(encoding="utf-8"))
    assert observed == result
    content_sha = observed.pop("content_sha256_before_self_field")
    assert audit.canonical_sha256_v71(observed) == content_sha
    assert result["gpu_launch_performed"] is False
    assert result["protected_dev_ood_or_holdout_opened"] is False
