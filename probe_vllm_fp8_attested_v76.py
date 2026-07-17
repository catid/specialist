#!/usr/bin/env python3
"""Attest routed FP8 methods and apply the V75 DeepGemm ordering workaround."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import probe_vllm_fp8_clean_preflight_v75 as base


SCHEMA_V76 = "v76-qwen36-fp8-routed-runtime-attestation"


def parameter_residency_v76(model: object) -> dict:
    components: dict[str, dict] = {}
    names: dict[str, list[str]] = {}
    for name, parameter in model.named_parameters():
        dotted = f".{name}."
        if ".visual." in dotted:
            component = "visual"
        elif ".mtp." in dotted:
            component = "mtp"
        elif ".language_model." in dotted or name.startswith("lm_head."):
            component = "language"
        else:
            component = "other"
        record = components.setdefault(component, {
            "parameter_count": 0,
            "logical_bytes": 0,
            "dtype_counts": {},
            "device_counts": {},
        })
        nbytes = int(parameter.numel()) * int(parameter.element_size())
        dtype = str(parameter.dtype)
        device = str(parameter.device)
        record["parameter_count"] += 1
        record["logical_bytes"] += nbytes
        record["dtype_counts"][dtype] = record["dtype_counts"].get(dtype, 0) + 1
        record["device_counts"][device] = record["device_counts"].get(device, 0) + 1
        names.setdefault(component, []).append(name)
    for component, record in components.items():
        record["dtype_counts"] = dict(sorted(record["dtype_counts"].items()))
        record["device_counts"] = dict(sorted(record["device_counts"].items()))
        record["parameter_names_sha256"] = (
            base.base.base.base.base.canonical_sha256(sorted(names[component]))
        )
    components = dict(sorted(components.items()))
    return {
        "schema": "v76-live-named-parameter-residency",
        "components": components,
        "total_parameter_count": sum(
            item["parameter_count"] for item in components.values()
        ),
        "total_logical_bytes": sum(
            item["logical_bytes"] for item in components.values()
        ),
        "named_parameters_remove_duplicate_default": True,
    }


def audit_fp8_routed_runtime_v76(model: object) -> dict:
    records = []
    fp8_references = []
    routed_like = []
    moe_runners = []
    for name, module in model.named_modules():
        quant = getattr(module, "quant_method", None)
        quant_class = type(quant).__name__ if quant is not None else None
        if name.endswith(".mlp.experts"):
            moe_runners.append({
                "name": name,
                "module_class": type(module).__name__,
                "quant_method_class": quant_class,
            })
        if type(module).__name__ == "RoutedExperts":
            routed_like.append({
                "name": name,
                "module_class": type(module).__name__,
                "quant_method_class": quant_class,
            })
        wrapped_quant = getattr(quant, "old_quant_method", None)
        fp8_quant = (
            quant if quant_class == "Fp8MoEMethod"
            else wrapped_quant
            if type(wrapped_quant).__name__ == "Fp8MoEMethod"
            else None
        )
        if fp8_quant is None:
            continue
        fp8_references.append({
            "name": name,
            "module_class": type(module).__name__,
            "runtime_quant_wrapper_class": quant_class,
        })
        if type(module).__name__ != "RoutedExperts":
            continue
        backend = getattr(fp8_quant, "fp8_backend", None)
        experts_cls = getattr(fp8_quant, "experts_cls", None)
        w13 = getattr(module, "w13_weight", None)
        w2 = getattr(module, "w2_weight", None)
        records.append({
            "name": name,
            "module_class": type(module).__name__,
            "quant_method_class": type(fp8_quant).__name__,
            "quant_method_module": type(fp8_quant).__module__,
            "runtime_quant_wrapper_class": quant_class,
            "runtime_quant_wrapper_module": type(quant).__module__,
            "weight_block_size": list(
                getattr(fp8_quant, "weight_block_size", [])
            ),
            "block_quant": getattr(fp8_quant, "block_quant", None),
            "fp8_backend_class": type(backend).__name__ if backend is not None else None,
            "fp8_backend_module": type(backend).__module__ if backend is not None else None,
            "fp8_backend_name": getattr(backend, "name", None),
            "fp8_backend_value": getattr(backend, "value", None),
            "experts_implementation_class": (
                experts_cls.__name__ if experts_cls is not None else None
            ),
            "experts_implementation_module": (
                experts_cls.__module__ if experts_cls is not None else None
            ),
            "w13_dtype": str(getattr(w13, "dtype", None)),
            "w2_dtype": str(getattr(w2, "dtype", None)),
        })
    records.sort(key=lambda item: item["name"])
    fp8_references.sort(key=lambda item: item["name"])
    routed_like.sort(key=lambda item: item["name"])
    moe_runners.sort(key=lambda item: item["name"])
    return {
        "schema": "v76-fp8-routed-model-runtime-audit",
        "fp8_moe_method_count": len(records),
        "fp8_quant_reference_count": len(fp8_references),
        "fp8_quant_references_sha256": base.base.base.base.base.canonical_sha256(
            fp8_references
        ),
        "fp8_moe_names_sha256": base.base.base.base.base.canonical_sha256(
            [item["name"] for item in records]
        ),
        "fp8_moe_backend_class_counts": dict(sorted(Counter(
            item["fp8_backend_class"] for item in records
        ).items())),
        "fp8_moe_records": records,
        "routed_like_module_count": len(routed_like),
        "routed_like_without_fp8_method": [
            item for item in routed_like
            if item["name"] not in {record["name"] for record in records}
        ],
        "moe_runner_module_count": len(moe_runners),
        "moe_runner_modules": moe_runners,
        "parameter_residency": parameter_residency_v76(model),
    }


def validate_runtime_audit_v76(value: dict) -> dict:
    records = value.get("fp8_moe_records") if isinstance(value, dict) else None
    if (
        value.get("schema") != "v76-fp8-routed-model-runtime-audit"
        or value.get("fp8_moe_method_count") != 40
        or value.get("fp8_quant_reference_count") != 80
        or value.get("routed_like_module_count") != 40
        or value.get("moe_runner_module_count") != 40
        or value.get("routed_like_without_fp8_method") != []
        or not isinstance(records, list)
        or len(records) != 40
        or len({item.get("name") for item in records}) != 40
        or any(
            item.get("quant_method_class") != "Fp8MoEMethod"
            or item.get("module_class") != "RoutedExperts"
            or item.get("runtime_quant_wrapper_class")
            != "FusedMoEModularMethod"
            or item.get("weight_block_size") != [128, 128]
            or item.get("block_quant") is not True
            or item.get("w13_dtype") != "torch.float8_e4m3fn"
            or item.get("w2_dtype") != "torch.float8_e4m3fn"
            for item in records
        )
    ):
        records_for_summary = records if isinstance(records, list) else []
        summary = {
            "schema": value.get("schema") if isinstance(value, dict) else None,
            "fp8_moe_method_count": (
                value.get("fp8_moe_method_count")
                if isinstance(value, dict) else None
            ),
            "fp8_quant_reference_count": (
                value.get("fp8_quant_reference_count")
                if isinstance(value, dict) else None
            ),
            "routed_like_module_count": (
                value.get("routed_like_module_count")
                if isinstance(value, dict) else None
            ),
            "moe_runner_module_count": (
                value.get("moe_runner_module_count")
                if isinstance(value, dict) else None
            ),
            "routed_like_without_fp8_method_count": len(
                value.get("routed_like_without_fp8_method", [])
            ) if isinstance(value, dict) else None,
            "routed_like_quant_method_classes": sorted({
                str(item.get("quant_method_class"))
                for item in value.get("routed_like_without_fp8_method", [])
            }) if isinstance(value, dict) else None,
            "first_routed_like": (
                value.get("routed_like_without_fp8_method", [None])[0]
                if isinstance(value, dict)
                and value.get("routed_like_without_fp8_method") else None
            ),
            "record_count": len(records_for_summary),
            "quant_method_classes": sorted({
                str(item.get("quant_method_class"))
                for item in records_for_summary
            }),
            "weight_block_sizes": sorted({
                json.dumps(item.get("weight_block_size"), sort_keys=True)
                for item in records_for_summary
            }),
            "block_quant_values": sorted({
                str(item.get("block_quant")) for item in records_for_summary
            }),
            "w13_dtypes": sorted({
                str(item.get("w13_dtype")) for item in records_for_summary
            }),
            "w2_dtypes": sorted({
                str(item.get("w2_dtype")) for item in records_for_summary
            }),
            "backend_counts": (
                value.get("fp8_moe_backend_class_counts")
                if isinstance(value, dict) else None
            ),
            "first_record": records_for_summary[0]
            if records_for_summary else None,
        }
        raise RuntimeError(
            "V76 routed FP8 runtime attestation failed: "
            + json.dumps(summary, sort_keys=True)
        )
    expected_hash = base.base.base.base.base.canonical_sha256(
        [item["name"] for item in records]
    )
    if value.get("fp8_moe_names_sha256") != expected_hash:
        raise RuntimeError("V76 routed FP8 name identity changed")
    return value


def upgraded_receipt_v76(value: dict, audit: dict, workaround: dict) -> dict:
    if (
        not isinstance(value, dict)
        or value.get("schema") != base.SCHEMA_V75
        or value.get("runtime", {}).get("starting_moe_tuning_table")
        != "fresh_empty_default"
        or value.get("runtime", {}).get("enable_flashinfer_autotune") is not False
    ):
        raise RuntimeError("V76 underlying V75 receipt changed")
    claimed = value.get("content_sha256_before_self_field")
    original = dict(value)
    original.pop("content_sha256_before_self_field", None)
    if base.base.base.base.base.canonical_sha256(original) != claimed:
        raise RuntimeError("V76 underlying V75 receipt identity changed")
    audit = validate_runtime_audit_v76(audit)
    if workaround != {
        "schema": "v76-explicit-deepgemm-disable-ordering-workaround",
        "VLLM_USE_DEEP_GEMM": "0",
        "quant_config_use_deep_gemm_before_post_init": False,
        "upstream_source_modified": False,
    }:
        raise RuntimeError("V76 DeepGemm workaround certificate changed")
    result = dict(original)
    result["schema"] = SCHEMA_V76
    result["routed_fp8_runtime_attestation"] = audit
    result["deepgemm_ordering_workaround"] = dict(workaround)
    result["preflight_gates"] = dict(result["preflight_gates"])
    result["preflight_gates"].update({
        "fp8_routed_expert_method_count_must_equal_40": True,
        "unexpected_unquantized_routed_expert_prefixes_empty": True,
        "routed_expert_method_count_pending_worker_attestation": False,
    })
    result["content_sha256_before_self_field"] = (
        base.base.base.base.base.canonical_sha256(result)
    )
    return result


def publish_v76(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary = path.with_name(f".{path.name}.v76-{os.getpid()}")
    with temporary.open("x", encoding="ascii") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    output = base.base.validate_argv_v74(sys.argv[1:])
    prior = os.environ.get("VLLM_USE_DEEP_GEMM")
    prior_insecure_serialization = os.environ.get(
        "VLLM_ALLOW_INSECURE_SERIALIZATION"
    )
    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    # apply_model must ship this module-local, read-only audit callback to the
    # colocated worker.  The callback and worker are both trusted local code.
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    import vllm
    from vllm.config import VllmConfig

    original_llm = vllm.LLM
    original_get_quant = VllmConfig._get_quantization_config
    audits: list[dict] = []

    def explicit_get_quantization(model_config, load_config):
        quant = original_get_quant(model_config, load_config)
        if (
            getattr(model_config, "quantization", None) == "fp8"
            and os.environ.get("VLLM_USE_DEEP_GEMM") == "0"
            and hasattr(quant, "use_deep_gemm")
        ):
            quant.use_deep_gemm = False
        return quant

    def attested_llm(*args, **kwargs):
        engine = original_llm(*args, **kwargs)
        values = engine.apply_model(audit_fp8_routed_runtime_v76)
        if not isinstance(values, list) or len(values) != 1:
            raise RuntimeError("V76 apply_model worker cardinality changed")
        audits.append(validate_runtime_audit_v76(values[0]))
        return engine

    VllmConfig._get_quantization_config = staticmethod(explicit_get_quantization)
    vllm.LLM = attested_llm
    try:
        status = base.main()
    finally:
        VllmConfig._get_quantization_config = original_get_quant
        vllm.LLM = original_llm
        if prior is None:
            os.environ.pop("VLLM_USE_DEEP_GEMM", None)
        else:
            os.environ["VLLM_USE_DEEP_GEMM"] = prior
        if prior_insecure_serialization is None:
            os.environ.pop("VLLM_ALLOW_INSECURE_SERIALIZATION", None)
        else:
            os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = (
                prior_insecure_serialization
            )
    if status != 0 or not output.is_file() or len(audits) != 1:
        raise RuntimeError("V76 underlying routed-attestation probe failed")
    workaround = {
        "schema": "v76-explicit-deepgemm-disable-ordering-workaround",
        "VLLM_USE_DEEP_GEMM": "0",
        "quant_config_use_deep_gemm_before_post_init": False,
        "upstream_source_modified": False,
    }
    upgraded = upgraded_receipt_v76(
        json.loads(output.read_text(encoding="utf-8")), audits[0], workaround
    )
    publish_v76(output, upgraded)
    print(json.dumps({
        "schema": SCHEMA_V76,
        "output": str(output),
        "content_sha256": upgraded["content_sha256_before_self_field"],
        "wall_runtime_seconds": upgraded[
            "wall_runtime_seconds_excluding_model_load_and_cleanup"
        ],
        "preflight_gates": upgraded["preflight_gates"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
