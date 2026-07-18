import json
from pathlib import Path
import re

import audit_qwen36_architecture_v1 as audit


ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "training_protocol/qwen36_architecture_contract_v1.json"


def load_contract():
    value = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    audit.validate_contract(value)
    return value


def test_contract_is_data_free_and_records_exact_physical_host():
    value = load_contract()
    assert value["authority"] == {
        "dataset_access": False,
        "gpu_cuda_visibility_in_audit": False,
        "optimizer_creation": False,
        "protected_evaluation_access": False,
        "training": False,
        "weight_loading": False,
        "weight_or_adapter_update": False,
    }
    assert value["acceptance"]["training_launched"] is False
    assert value["hardware"]["expected"] == {
        "compute_capability": "12.0",
        "count": 4,
        "memory_total_mib": 97_887,
        "name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    }
    assert [gpu["index"] for gpu in value["hardware"]["gpus"]] == [0, 1, 2, 3]
    assert {gpu["memory_total_mib"] for gpu in value["hardware"]["gpus"]} == {97_887}


def test_checkpoint_geometry_and_installed_package_state_are_pinned():
    value = load_contract()
    audit_commit = value["repository"]["audit_base_commit"]
    assert re.fullmatch(r"[0-9a-f]{40}", audit_commit)
    assert audit._command("git", "cat-file", "-t", audit_commit) == "commit"
    assert value["repository"]["submodules"] == [
        {
            "commit": "574a9d134da1ffce2a8bb812019899e5c96b588a",
            "description": "(v1.0.0-2-g574a9d1)",
            "path": "es-at-scale",
            "status": " ",
        }
    ]
    checkpoint = value["checkpoint"]
    assert checkpoint["config_sha256"] == "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
    assert checkpoint["index_sha256"] == "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"
    assert checkpoint["declared_model_type"] == "qwen3_5_moe"
    assert checkpoint["declared_architectures"] == ["Qwen3_5MoeForConditionalGeneration"]
    assert checkpoint["tensor_count"] == 1_045
    assert checkpoint["weight_file_count"] == 26
    assert checkpoint["geometry"] == {
        "full_attention_layers": 10,
        "hidden_size": 2_048,
        "layer_types": [
            "full_attention" if index % 4 == 3 else "linear_attention"
            for index in range(40)
        ],
        "linear_attention_layers": 30,
        "mtp_num_hidden_layers": 1,
        "num_experts": 256,
        "num_experts_per_token": 8,
        "num_hidden_layers": 40,
        "routed_expert_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "vision_depth": 27,
        "vision_hidden_size": 1_152,
        "vision_intermediate_size": 4_304,
        "vocab_size": 248_320,
    }
    packages = value["software"]["packages"]
    assert packages["torch"]["version"] == "2.11.0"
    assert packages["transformers"]["version"] == "5.12.1"
    assert packages["peft"]["version"] == "0.19.1"
    assert packages["accelerate"]["version"] == "1.14.0"
    assert packages["triton"]["version"] == "3.6.0"
    assert packages["fla-core"]["version"] == "0.5.1"
    assert packages["flash-linear-attention"]["version"] == "0.5.1"
    assert packages["causal-conv1d"]["version"] == "1.6.2.post1"
    for missing in ("trl", "unsloth", "flash-attn"):
        assert packages[missing]["installed"] is False
        assert packages[missing]["version"] is None
    optional = value["software"]["optional_training_packages"]
    assert optional["fast_linear_attention"].startswith(
        "installed_runtime_validation_delegated_to_"
    )
    assert optional["causal_conv1d"].startswith(
        "installed_runtime_validation_delegated_to_"
    )


def test_auto_causal_loader_is_proven_text_only_with_exact_prefix_conversion():
    value = load_contract()
    dispatch = value["architecture"]["auto_causal_lm_dispatch"]
    assert dispatch["input_is_composite_config"] is True
    assert dispatch["auto_factory_extracts_text_config"] is True
    assert dispatch["result_has_visual_module"] is False
    assert dispatch["result_has_composite_language_model_namespace"] is False
    assert dispatch["mapped_language_keys_equal_runtime_parameters"] is True
    assert dispatch["result_parameter_count"] == dispatch["checkpoint_language_tensor_count"] == 693
    assert dispatch["effective_prefix_mapping"] == {
        "checkpoint": "model.language_model.<suffix>",
        "runtime": "model.<suffix>",
        "lm_head": "unchanged",
    }
    assert dispatch["built_in_conversion_mapping"][0]["source_patterns"] == [
        r"^model\.language_model\.(.+)$"
    ]
    assert dispatch["built_in_conversion_mapping"][0]["target_patterns"] == [r"model.\1"]
    assert dispatch["ignored_checkpoint_prefixes"] == {
        "model.visual.*": 333,
        "mtp.*": 19,
    }
    assert value["architecture"]["training_modality_decision"]["decision"] == (
        "text_only_exclude_vision_and_mtp"
    )


def test_every_layer_has_exact_fused_shared_and_frozen_gate_names():
    value = load_contract()
    architecture = value["architecture"]
    layers = architecture["layers"]
    assert len(layers) == 40
    expected_shared = set()
    expected_routed = set()
    expected_routers = set()
    expected_shared_gates = set()
    for index, layer in enumerate(layers):
        runtime_mlp = f"model.layers.{index}.mlp"
        assert layer["index"] == index
        parameters = {item["role"]: item for item in layer["parameters"]}
        assert parameters["router"]["runtime_name"] == f"{runtime_mlp}.gate.weight"
        assert parameters["router"]["shape"] == [256, 2_048]
        assert parameters["router"]["train_with_lora"] is False
        assert parameters["routed_gate_up"]["runtime_name"] == (
            f"{runtime_mlp}.experts.gate_up_proj"
        )
        assert parameters["routed_gate_up"]["shape"] == [256, 1_024, 2_048]
        assert parameters["routed_down"]["runtime_name"] == (
            f"{runtime_mlp}.experts.down_proj"
        )
        assert parameters["routed_down"]["shape"] == [256, 2_048, 512]
        assert parameters["shared_expert_gate"]["runtime_name"] == (
            f"{runtime_mlp}.shared_expert_gate.weight"
        )
        assert parameters["shared_expert_gate"]["train_with_lora"] is False

        expected_shared.update(
            f"{runtime_mlp}.shared_expert.{projection}"
            for projection in ("gate_proj", "up_proj", "down_proj")
        )
        expected_routed.update(
            f"{runtime_mlp}.experts.{parameter}"
            for parameter in ("gate_up_proj", "down_proj")
        )
        expected_routers.add(f"{runtime_mlp}.gate.weight")
        expected_shared_gates.add(f"{runtime_mlp}.shared_expert_gate.weight")

    target = architecture["adapter_target_contract"]
    assert set(target["shared_expert_target_modules"]) == expected_shared
    assert set(target["routed_expert_target_parameters"]) == expected_routed
    assert set(architecture["explicitly_frozen"]["router_parameters"]) == expected_routers
    assert set(architecture["explicitly_frozen"]["shared_expert_gate_parameters"]) == (
        expected_shared_gates
    )


def test_target_manifest_contains_only_literal_complete_runtime_names():
    target = load_contract()["architecture"]["adapter_target_contract"]
    assert target["matching_mode"] == "literal_full_name_membership_only"
    assert target["suffix_guessing_allowed"] is False
    assert target["regex_targeting_allowed"] is False
    names = target["shared_expert_target_modules"] + target["routed_expert_target_parameters"]
    assert len(names) == len(set(names)) == 200
    assert all(name.startswith("model.layers.") for name in names)
    assert not any(any(char in name for char in "*[]()?$^|\\") for name in names)
    assert target["require_pre_attach_exact_set_equality"] is True
    assert target["require_post_attach_observed_exact_set_equality"] is True
