import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

import stage_matched_lora_initialization_vllm_v41b as stage


def test_key_transform_is_exact_and_one_way_v41b():
    source = (
        "base_model.model.model.layers.23.self_attn.q_proj.lora_A.weight"
    )
    target = (
        "base_model.model.model.language_model.layers.23."
        "self_attn.q_proj.lora_A.weight"
    )
    assert stage.transform_key_v41b(source) == target
    with pytest.raises(ValueError, match="outside the sealed namespace"):
        stage.transform_key_v41b("model.layers.23.self_attn.q_proj.lora_A.weight")
    with pytest.raises(ValueError, match="exactly once"):
        stage.transform_key_v41b(target)


def test_sealed_source_audit_v41b():
    source = stage.audit_sealed_source_v41b()
    assert len(source["tensors"]) == stage.EXPECTED_TENSOR_COUNT_V41B
    assert sum(t.numel() for t in source["tensors"].values()) == stage.EXPECTED_ELEMENTS_V41B
    assert source["file_hashes"] == {
        "weights": stage.EXPECTED_SOURCE_WEIGHTS_SHA256_V41B,
        "config": stage.EXPECTED_SOURCE_CONFIG_SHA256_V41B,
        "manifest": stage.EXPECTED_SOURCE_MANIFEST_FILE_SHA256_V41B,
    }
    assert source["manifest_content_sha256"] == (
        stage.EXPECTED_SOURCE_MANIFEST_CONTENT_SHA256_V41B
    )
    assert source["source_tensor_identity_sha256"] == (
        stage.EXPECTED_SOURCE_TENSOR_IDENTITY_SHA256_V41B
    )
    assert all(
        item["target_key"] == stage.transform_key_v41b(item["source_key"])
        for item in source["records"]
    )


def _assert_staged_artifact_v41b(directory: Path) -> dict:
    source = stage.audit_sealed_source_v41b()
    weights = directory / "adapter_model.safetensors"
    config = directory / "adapter_config.json"
    manifest_path = directory / "stage_manifest_v41b.json"
    assert config.read_bytes() == source["config_bytes"]
    assert stage.file_sha256_v41b(config) == stage.EXPECTED_SOURCE_CONFIG_SHA256_V41B

    raw_manifest = json.loads(manifest_path.read_text())
    content_hash = raw_manifest.pop("content_sha256_before_self_field")
    assert content_hash == stage.canonical_sha256_v41b(raw_manifest)
    assert raw_manifest["transform"] == {
        "operation": "key_prefix_replacement_only",
        "source_prefix": stage.SOURCE_PREFIX_V41B,
        "target_prefix": stage.TARGET_PREFIX_V41B,
        "tensor_arithmetic_performed": False,
        "tensor_cast_performed": False,
        "tensor_bytes_preserved_exactly": True,
        "adapter_config_copied_byte_exact": True,
    }
    assert not raw_manifest["dataset_or_training_examples_accessed"]
    assert not raw_manifest["shadow_ood_holdout_or_heldout_accessed"]
    assert not raw_manifest["gpu_accessed"]
    assert not raw_manifest["evaluation_performed"]

    with safe_open(weights, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        assert handle.metadata() == {"format": "pt"}
    assert len(keys) == stage.EXPECTED_TENSOR_COUNT_V41B
    assert all(key.startswith(stage.TARGET_PREFIX_V41B) for key in keys)
    assert not any(key.startswith(stage.SOURCE_PREFIX_V41B) for key in keys)

    target_tensors = load_file(weights, device="cpu")
    records = raw_manifest["tensor_mapping_records"]
    assert len(records) == stage.EXPECTED_TENSOR_COUNT_V41B
    for record in records:
        source_tensor = source["tensors"][record["source_key"]]
        target_tensor = target_tensors[record["target_key"]]
        assert target_tensor.dtype == torch.float32
        assert torch.equal(target_tensor, source_tensor)
        assert stage.tensor_sha256_v41b(target_tensor) == record["source_tensor_sha256"]
        assert record["target_tensor_sha256"] == record["source_tensor_sha256"]
        assert record["tensor_bytes_preserved_exactly"] is True
    assert raw_manifest["transformed_identity"][
        "all_tensor_bytes_preserved_exactly"
    ] is True
    return {
        "weights_sha256": stage.file_sha256_v41b(weights),
        "manifest_file_sha256": stage.file_sha256_v41b(manifest_path),
        "manifest_content_sha256": content_hash,
        "transformed_identity": raw_manifest["transformed_identity"],
    }


def test_stage_is_byte_exact_and_whole_file_deterministic_v41b(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    stage.stage_artifact_v41b(first)
    stage.stage_artifact_v41b(second)
    first_identity = _assert_staged_artifact_v41b(first)
    second_identity = _assert_staged_artifact_v41b(second)
    assert first_identity == second_identity
    assert (first / "adapter_model.safetensors").read_bytes() == (
        second / "adapter_model.safetensors"
    ).read_bytes()
    assert (first / "adapter_config.json").read_bytes() == (
        second / "adapter_config.json"
    ).read_bytes()
    assert (first / "stage_manifest_v41b.json").read_bytes() == (
        second / "stage_manifest_v41b.json"
    ).read_bytes()


def test_stage_refuses_overwrite_v41b(tmp_path):
    output = tmp_path / "already-there"
    output.mkdir()
    with pytest.raises(FileExistsError):
        stage.stage_artifact_v41b(output)


def test_cli_exposes_no_source_seed_data_or_eval_override_v41b():
    parser = stage.parser_v41b()
    assert parser.parse_args([]).output == str(stage.DEFAULT_OUTPUT_V41B)
    with pytest.raises(SystemExit):
        parser.parse_args(["--source", "/tmp/unsealed"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--seed", "1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset", "/tmp/data"])
