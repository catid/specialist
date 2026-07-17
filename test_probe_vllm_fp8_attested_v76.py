import pytest
import torch

import probe_vllm_fp8_attested_v76 as subject


def audit():
    records = [{
        "name": f"model.layers.{index}.mlp.experts",
        "module_class": "RoutedExperts",
        "quant_method_class": "Fp8MoEMethod",
        "quant_method_module": "vllm.fp8",
        "runtime_quant_wrapper_class": "FusedMoEModularMethod",
        "runtime_quant_wrapper_module": "vllm.fused_moe",
        "weight_block_size": [128, 128],
        "block_quant": True,
        "fp8_backend_class": "CutlassBackend",
        "fp8_backend_module": "vllm.cutlass",
        "fp8_backend_name": "TRITON",
        "fp8_backend_value": "triton",
        "experts_implementation_class": "TritonExperts",
        "experts_implementation_module": "vllm.triton_moe",
        "w13_dtype": "torch.float8_e4m3fn",
        "w2_dtype": "torch.float8_e4m3fn",
    } for index in range(40)]
    return {
        "schema": "v76-fp8-routed-model-runtime-audit",
        "fp8_moe_method_count": 40,
        "fp8_quant_reference_count": 80,
        "fp8_quant_references_sha256": "synthetic-wrapper-reference-hash",
        "fp8_moe_names_sha256": (
            subject.base.base.base.base.base.canonical_sha256(
                [item["name"] for item in records]
            )
        ),
        "fp8_moe_backend_class_counts": {"CutlassBackend": 40},
        "fp8_moe_records": records,
        "routed_like_module_count": 40,
        "routed_like_without_fp8_method": [],
        "moe_runner_module_count": 40,
        "moe_runner_modules": [{
            "name": f"model.layers.{index}.mlp.experts",
            "module_class": "MoERunner",
            "quant_method_class": None,
        } for index in range(40)],
    }


def v75_receipt():
    value = {
        "schema": subject.base.SCHEMA_V75,
        "runtime": {
            "starting_moe_tuning_table": "fresh_empty_default",
            "enable_flashinfer_autotune": False,
        },
        "preflight_gates": {
            "routed_expert_method_count_pending_worker_attestation": True,
        },
    }
    value["content_sha256_before_self_field"] = (
        subject.base.base.base.base.base.canonical_sha256(value)
    )
    return value


def workaround():
    return {
        "schema": "v76-explicit-deepgemm-disable-ordering-workaround",
        "VLLM_USE_DEEP_GEMM": "0",
        "quant_config_use_deep_gemm_before_post_init": False,
        "upstream_source_modified": False,
    }


def test_audit_requires_all_40_exact_block_fp8_methods():
    assert subject.validate_runtime_audit_v76(audit())[
        "fp8_moe_method_count"
    ] == 40
    changed = audit()
    changed["fp8_moe_records"][0]["weight_block_size"] = [64, 128]
    with pytest.raises(RuntimeError, match="attestation failed"):
        subject.validate_runtime_audit_v76(changed)


def test_upgrade_seals_worker_attestation_and_workaround():
    upgraded = subject.upgraded_receipt_v76(
        v75_receipt(), audit(), workaround()
    )
    assert upgraded["preflight_gates"][
        "fp8_routed_expert_method_count_must_equal_40"
    ] is True
    assert upgraded["preflight_gates"][
        "routed_expert_method_count_pending_worker_attestation"
    ] is False
    claimed = upgraded.pop("content_sha256_before_self_field")
    assert subject.base.base.base.base.base.canonical_sha256(upgraded) == claimed


def test_upgrade_rejects_parent_or_workaround_drift():
    parent = v75_receipt()
    parent["runtime"]["enable_flashinfer_autotune"] = True
    with pytest.raises(RuntimeError, match="receipt changed"):
        subject.upgraded_receipt_v76(parent, audit(), workaround())
    changed = workaround()
    changed["VLLM_USE_DEEP_GEMM"] = "1"
    with pytest.raises(RuntimeError, match="workaround certificate"):
        subject.upgraded_receipt_v76(v75_receipt(), audit(), changed)


def test_parameter_residency_separates_text_visual_and_mtp():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Linear(3, 2, bias=False)
            self.visual = torch.nn.Linear(2, 2, bias=False)
            self.mtp = torch.nn.Linear(2, 1, bias=False)

    value = subject.parameter_residency_v76(Tiny())
    assert value["components"]["language"]["logical_bytes"] == 24
    assert value["components"]["visual"]["logical_bytes"] == 16
    assert value["components"]["mtp"]["logical_bytes"] == 8
    assert value["total_logical_bytes"] == 48


def test_main_enables_only_process_local_callback_serialization(monkeypatch):
    class StopAfterEnvironment(RuntimeError):
        pass

    def stop_import(name, *args, **kwargs):
        if name == "vllm":
            assert subject.os.environ["VLLM_USE_DEEP_GEMM"] == "0"
            assert subject.os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] == "1"
            raise StopAfterEnvironment
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.delenv("VLLM_ALLOW_INSECURE_SERIALIZATION", raising=False)
    monkeypatch.setattr("builtins.__import__", stop_import)
    monkeypatch.setattr(subject.base.base, "validate_argv_v74", lambda _: None)
    with pytest.raises(StopAfterEnvironment):
        subject.main()
