import hashlib
import json
import math

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

import build_matched_lora_initialization_v41a as build


V37_WEIGHTS = (
    build.ROOT / "experiments/sft_controls/v37a_equal_unit_fold3_v412/"
    "middle_late_r32_seed17/final/adapter_model.safetensors"
).resolve()


def test_v41a_surface_exactly_matches_v37a_peft_inventory():
    specs = {item["key"]: item for item in build.tensor_specs_v41a()}
    with safe_open(V37_WEIGHTS, framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == set(specs)
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            assert list(tensor.shape) == specs[key]["shape"]
            assert tensor.dtype == torch.float32
    assert len(specs) == 70
    assert len({item["module"] for item in specs.values()}) == 35
    assert sum(item["elements"] for item in specs.values()) == 4_528_128
    assert sum(item["elements"] for item in specs.values() if item["role"] == "A") == 2_359_296
    assert sum(item["elements"] for item in specs.values() if item["role"] == "B") == 2_168_832


def test_v41a_source_config_is_byte_exact_v37a_contract():
    raw, config = build.validate_source_config_v41a()
    assert hashlib.sha256(raw).hexdigest() == build.SOURCE_CONFIG_SHA256_V41A
    assert config["r"] == 32
    assert config["lora_alpha"] == 64
    assert config["layers_to_transform"] == [20, 21, 22, 23]
    assert config["init_lora_weights"] is True


def test_v41a_initialization_is_deterministic_kaiming_a_and_exact_zero_b():
    first, first_records, first_guard = build.build_tensors_v41a()
    second, second_records, second_guard = build.build_tensors_v41a()
    assert first_records == second_records
    assert first_guard == second_guard
    assert build.tensor_identity_v41a(first_records) == build.tensor_identity_v41a(
        second_records
    )
    for spec in build.tensor_specs_v41a():
        key = spec["key"]
        assert torch.equal(first[key], second[key])
        if spec["role"] == "A":
            assert torch.count_nonzero(first[key]).item() > 0
            bound = 1.0 / math.sqrt(spec["shape"][1])
            assert float(first[key].abs().max()) <= bound
        else:
            assert torch.count_nonzero(first[key]).item() == 0
            assert first[key].view(torch.uint8).count_nonzero().item() == 0
    assert first_guard["passed"] is True
    assert first_guard["module_pairs"] == 35
    assert first_guard["zero_zero_pairs"] == 0


def test_v41a_zero_zero_guard_rejects_bilinear_degeneracy():
    tensors, _records, _guard = build.build_tensors_v41a()
    bad = {key: value.clone() for key, value in tensors.items()}
    a_key = next(key for key in sorted(bad) if ".lora_A." in key)
    bad[a_key].zero_()
    with pytest.raises(RuntimeError, match="forbidden zero/zero antithetic"):
        build.validate_zero_zero_degeneracy_guard_v41a(bad)


def test_v41a_guard_rejects_nonzero_b_initialization():
    tensors, _records, _guard = build.build_tensors_v41a()
    bad = {key: value.clone() for key, value in tensors.items()}
    b_key = next(key for key in sorted(bad) if ".lora_B." in key)
    bad[b_key][0, 0] = 1.0
    with pytest.raises(RuntimeError, match="nonzero-A/exact-zero-B"):
        build.validate_zero_zero_degeneracy_guard_v41a(bad)


def test_v41a_zero_zero_degeneracy_witness_is_exact():
    witness = build.zero_zero_degeneracy_witness_v41a()
    assert witness == {
        "schema": "matched-lora-zero-zero-degeneracy-witness-v41a",
        "passed": True,
        "plus_equals_minus": True,
        "central_nonzero_elements": 0,
        "sigma": 0.125,
    }


def test_v41a_artifact_is_reproducible_self_hashed_and_byte_verified(tmp_path):
    first_dir = tmp_path / "first_v41a"
    second_dir = tmp_path / "second_v41a"
    first = build.build_artifact_v41a(first_dir)
    second = build.build_artifact_v41a(second_dir)

    for manifest, directory in ((first, first_dir), (second, second_dir)):
        weights = directory / "adapter_model.safetensors"
        config = directory / "adapter_config.json"
        manifest_path = directory / "initialization_manifest_v41a.json"
        assert weights.is_file() and config.is_file() and manifest_path.is_file()
        reopened_manifest = json.loads(manifest_path.read_text())
        content = reopened_manifest.pop("content_sha256_before_self_field")
        assert content == build.canonical_sha256_v41a(reopened_manifest)
        assert content == manifest["content_sha256_before_self_field"]
        assert manifest["readback"]["verified"] is True
        assert manifest["readback"]["tensor_identity"] == manifest["tensor_identity"]
        assert manifest["artifact"]["weights_file_sha256"] == build.file_sha256_v41a(
            weights
        )
        assert manifest["artifact"]["adapter_config_file_sha256"] == (
            build.SOURCE_CONFIG_SHA256_V41A
        )
        assert config.read_bytes() == build.SOURCE_CONFIG_V41A.read_bytes()
        loaded = load_file(weights, device="cpu")
        assert len(loaded) == 70
        build.validate_zero_zero_degeneracy_guard_v41a(loaded)

    assert first["tensor_identity"] == second["tensor_identity"]
    assert first["artifact"]["weights_file_sha256"] == second["artifact"][
        "weights_file_sha256"
    ]
    assert first["content_sha256_before_self_field"] == second[
        "content_sha256_before_self_field"
    ]


def test_v41a_seed_is_sealed_and_has_no_cli_override():
    parser = build.parser_v41a()
    with pytest.raises(SystemExit):
        parser.parse_args(["--seed", "1"])
    assert build.SEALED_INITIALIZATION_SEED_V41A == 20_260_715_041
