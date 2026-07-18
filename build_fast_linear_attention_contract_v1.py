#!/usr/bin/env python3
"""Build a synthetic-only fast linear-attention runtime contract.

The parent process audits installed distributions and source receipts, then
launches three isolated probes per physical GPU.  No model, checkpoint,
tokenizer, dataset, or evaluation source is opened.  A compiled-kernel crash
is contained to its child process and becomes an explicit fallback reason.
"""

from __future__ import annotations

import argparse
import copy
import concurrent.futures
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent
ARCHITECTURE_CONTRACT = (
    ROOT / "training_protocol/qwen36_architecture_contract_v1.json"
).resolve()
OUTPUT = (
    ROOT / "training_protocol/fast_linear_attention_contract_v1.json"
).resolve()

SCHEMA = "specialist-fast-linear-attention-contract-v1"
CORE_DISTRIBUTIONS = ("torch", "transformers", "peft", "triton")
FAST_DISTRIBUTIONS = {
    "causal-conv1d": "1.6.2.post1",
    "fla-core": "0.5.1",
    "flash-linear-attention": "0.5.1",
}
RUNTIME_DISTRIBUTIONS = (*CORE_DISTRIBUTIONS, *FAST_DISTRIBUTIONS, "tilelang")
GPU_INDICES = (0, 1, 2, 3)
WORKER_MARKER = "FAST_LINEAR_ATTENTION_RESULT="
WARMUP_ITERATIONS = 2
BENCHMARK_ITERATIONS = 12
FULL_MODULE_BENCHMARK_ITERATIONS = {128: 6, 2048: 4}
BF16_TOLERANCES = {
    "causal_conv1d_forward": {"atol": 0.03, "rtol": 0.03},
    "causal_conv1d_gradients": {"atol": 0.03, "rtol": 0.03},
    "gated_delta_chunk_forward": {"atol": 0.03, "rtol": 0.03},
    "gated_delta_chunk_gradients": {"atol": 0.03, "rtol": 0.03},
    "gated_delta_recurrent_forward": {"atol": 0.03, "rtol": 0.03},
    "gated_delta_recurrent_gradients": {"atol": 0.03, "rtol": 0.03},
    "qwen35_moe_gated_delta_module_forward": {"atol": 0.03, "rtol": 0.03},
    "qwen35_moe_gated_delta_module_gradients": {"atol": 0.03, "rtol": 0.03},
}
QWEN35_MOE_GATED_DELTA_GEOMETRY = {
    "hidden_size": 2048,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
}
HYBRID_TRAINING_BINDINGS = {
    "causal_conv1d_fn": None,
    "causal_conv1d_update": (
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe."
        "torch_causal_conv1d_update"
    ),
    "chunk_gated_delta_rule": (
        "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule"
    ),
    "recurrent_gated_delta_rule": (
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe."
        "torch_recurrent_gated_delta_rule"
    ),
}
SYNTHETIC_WORKLOADS = {
    "causal_conv1d": {
        "dtype": "torch.bfloat16",
        "seed_base": 19019,
        "shape": {"batch": 2, "channels": 64, "sequence": 128, "width": 4},
        "distributions": {
            "x": "cpu_generator_uniform[-0.5,0.5]_then_bfloat16_cuda",
            "weight": "cpu_generator_uniform[-0.25,0.25]_then_bfloat16_cuda",
            "bias": "cpu_generator_uniform[-0.1,0.1]_then_bfloat16_cuda",
            "upstream_gradient": "cpu_generator_uniform[-0.5,0.5]_then_bfloat16_cuda",
        },
        "activation": "silu",
    },
    "gated_delta_chunk": {
        "dtype": "torch.bfloat16",
        "seed_base": 19119,
        "shape": {
            "batch": 1,
            "sequence": 64,
            "heads": 2,
            "key_head_dim": 32,
            "value_head_dim": 32,
        },
        "distributions": {
            "query_key_value": "cpu_generator_uniform[-0.5,0.5]_then_bfloat16_cuda",
            "g": "negative_softplus(cpu_generator_uniform[-1,1])_then_bfloat16_cuda",
            "beta": "sigmoid(cpu_generator_uniform[-1,1])_then_bfloat16_cuda",
            "upstream_gradient": "cpu_generator_uniform[-0.5,0.5]_then_bfloat16_cuda",
        },
        "use_qk_l2norm_in_kernel": True,
        "output_final_state": False,
    },
    "gated_delta_recurrent": {
        "dtype": "torch.bfloat16",
        "seed_base": 19219,
        "shape": {
            "batch": 1,
            "sequence": 8,
            "heads": 2,
            "key_head_dim": 32,
            "value_head_dim": 32,
        },
        "distributions": {
            "query_key_value": "cpu_generator_uniform[-0.5,0.5]_then_bfloat16_cuda",
            "g": "negative_softplus(cpu_generator_uniform[-1,1])_then_bfloat16_cuda",
            "beta": "sigmoid(cpu_generator_uniform[-1,1])_then_bfloat16_cuda",
            "upstream_gradient": "cpu_generator_uniform[-0.5,0.5]_then_bfloat16_cuda",
        },
        "use_qk_l2norm_in_kernel": True,
        "output_final_state": True,
    },
    "qwen35_moe_gated_delta_module": {
        "dtype": "torch.bfloat16",
        "seed_base": 19319,
        "shape": {
            "batch": 1,
            "sequence_lengths": [128, 2048],
            **QWEN35_MOE_GATED_DELTA_GEOMETRY,
        },
        "distributions": {
            "module_parameters": (
                "installed_Qwen3_5MoeGatedDeltaNet_constructor_after_"
                "torch_manual_seed_19319_then_deepcopy"
            ),
            "hidden_states": (
                "cpu_generator_uniform[-0.1,0.1]_seed_"
                "(19319+sequence_length)_then_bfloat16_cuda"
            ),
            "upstream_gradient": (
                "same_cpu_generator_next_uniform[-0.1,0.1]_then_"
                "bfloat16_cuda"
            ),
        },
        "layer_idx": 0,
        "cache_params": None,
        "attention_mask": None,
        "normalization_binding": "installed_module_binding_unchanged",
        "reference_bindings": {
            "causal_conv1d_fn": None,
            "causal_conv1d_update": (
                "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe."
                "torch_causal_conv1d_update"
            ),
            "chunk_gated_delta_rule": (
                "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe."
                "torch_chunk_gated_delta_rule"
            ),
            "recurrent_gated_delta_rule": (
                "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe."
                "torch_recurrent_gated_delta_rule"
            ),
        },
        "hybrid_bindings": HYBRID_TRAINING_BINDINGS,
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _display_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _source_receipt(path: Path, role: str) -> dict:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise RuntimeError(f"source receipt is not a file: {resolved}")
    return {
        "role": role,
        "path": _display_path(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _load_self_addressed(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    declared = value.get("content_sha256_before_self_field")
    unsigned = dict(value)
    unsigned.pop("content_sha256_before_self_field", None)
    if (
        not isinstance(declared, str)
        or not re.fullmatch(r"[0-9a-f]{64}", declared)
        or canonical_sha256(unsigned) != declared
    ):
        raise RuntimeError(f"invalid content address: {path}")
    return value


def _dist_info_file(distribution: importlib.metadata.Distribution, name: str) -> Path | None:
    for item in distribution.files or ():
        if Path(str(item)).name == name and ".dist-info" in str(item):
            path = Path(distribution.locate_file(item)).resolve()
            if path.is_file():
                return path
    return None


def _distribution_receipt(name: str) -> dict:
    distribution = importlib.metadata.distribution(name)
    metadata_path = _dist_info_file(distribution, "METADATA")
    record_path = _dist_info_file(distribution, "RECORD")
    direct_url_path = _dist_info_file(distribution, "direct_url.json")
    direct_url = None
    vcs_commit = None
    if direct_url_path is not None:
        direct_url = json.loads(direct_url_path.read_text(encoding="utf-8"))
        vcs_commit = direct_url.get("vcs_info", {}).get("commit_id")
    return {
        "requested_name": name,
        "canonical_name": distribution.metadata.get("Name"),
        "version": distribution.version,
        "direct_url": direct_url,
        "vcs_commit": vcs_commit,
        "dist_info_path": _display_path(Path(distribution._path)),
        "metadata_sha256": (
            file_sha256(metadata_path) if metadata_path is not None else None
        ),
        "record_sha256": file_sha256(record_path) if record_path is not None else None,
    }


def _verify_recorded_sources(receipts: dict[str, dict]) -> dict:
    checks = {}
    for role, receipt in sorted(receipts.items()):
        path = ROOT / receipt["path"]
        observed = file_sha256(path)
        checks[role] = {
            "path": receipt["path"],
            "expected_sha256": receipt["sha256"],
            "observed_sha256": observed,
            "matches": observed == receipt["sha256"],
        }
    return checks


def _audit_environment(architecture: dict) -> tuple[dict, dict]:
    distributions = {
        name: _distribution_receipt(name) for name in RUNTIME_DISTRIBUTIONS
    }
    baseline_packages = architecture["software"]["packages"]
    package_checks = {}
    for name in CORE_DISTRIBUTIONS:
        expected = baseline_packages[name]
        actual = distributions[name]
        package_checks[name] = {
            "expected_version": expected["version"],
            "observed_version": actual["version"],
            "version_matches": actual["version"] == expected["version"],
            "expected_direct_url": expected["direct_url"],
            "observed_direct_url": actual["direct_url"],
            "direct_url_matches": actual["direct_url"] == expected["direct_url"],
            "expected_vcs_commit": expected["vcs_commit"],
            "observed_vcs_commit": actual["vcs_commit"],
            "vcs_commit_matches": actual["vcs_commit"] == expected["vcs_commit"],
        }

    baseline_sources = {}
    for role, receipt in architecture["architecture"]["implementation_sources"].items():
        baseline_sources[f"architecture:{role}"] = receipt
    for role, receipt in architecture["software"]["peft_implementation_sources"].items():
        baseline_sources[f"peft:{role}"] = receipt
    baseline_source_checks = _verify_recorded_sources(baseline_sources)

    import torch

    torch_runtime = architecture["software"]["torch_runtime"]
    torch_checks = {
        "expected_runtime_version": torch_runtime["version"],
        "observed_runtime_version": torch.__version__,
        "runtime_version_matches": torch.__version__ == torch_runtime["version"],
        "expected_git_version": torch_runtime["git_version"],
        "observed_git_version": torch.version.git_version,
        "git_version_matches": torch.version.git_version
        == torch_runtime["git_version"],
        "expected_compiled_cuda": torch_runtime["compiled_cuda"],
        "observed_compiled_cuda": torch.version.cuda,
        "compiled_cuda_matches": torch.version.cuda
        == torch_runtime["compiled_cuda"],
    }
    fast_version_checks = {
        name: {
            "expected": expected,
            "observed": distributions[name]["version"],
            "matches": distributions[name]["version"] == expected,
        }
        for name, expected in FAST_DISTRIBUTIONS.items()
    }
    all_core_preserved = (
        all(
            all(
                check[key]
                for key in (
                    "version_matches",
                    "direct_url_matches",
                    "vcs_commit_matches",
                )
            )
            for check in package_checks.values()
        )
        and all(check["matches"] for check in baseline_source_checks.values())
        and all(
            torch_checks[key]
            for key in (
                "runtime_version_matches",
                "git_version_matches",
                "compiled_cuda_matches",
            )
        )
    )
    all_fast_versions_exact = all(
        check["matches"] for check in fast_version_checks.values()
    )

    source_paths = {
        "builder": Path(__file__).resolve(),
        "torch_init": Path(importlib.import_module("torch").__file__).resolve(),
        "triton_init": Path(importlib.import_module("triton").__file__).resolve(),
        "transformers_init": Path(
            importlib.import_module("transformers").__file__
        ).resolve(),
        "peft_init": Path(importlib.import_module("peft").__file__).resolve(),
        "causal_conv1d_init": Path(
            importlib.import_module("causal_conv1d").__file__
        ).resolve(),
        "causal_conv1d_interface": Path(
            importlib.import_module("causal_conv1d.causal_conv1d_interface").__file__
        ).resolve(),
        "causal_conv1d_cpp_functions": Path(
            importlib.import_module("causal_conv1d.cpp_functions").__file__
        ).resolve(),
        "causal_conv1d_cuda_extension": Path(
            importlib.import_module("causal_conv1d_cuda").__file__
        ).resolve(),
        "fla_init": Path(importlib.import_module("fla").__file__).resolve(),
        "fla_gated_delta_init": Path(
            importlib.import_module("fla.ops.gated_delta_rule").__file__
        ).resolve(),
        "fla_gated_delta_chunk": Path(
            importlib.import_module("fla.ops.gated_delta_rule.chunk").__file__
        ).resolve(),
        "fla_gated_delta_fused_recurrent": Path(
            importlib.import_module(
                "fla.ops.gated_delta_rule.fused_recurrent"
            ).__file__
        ).resolve(),
        "tilelang_init": Path(importlib.import_module("tilelang").__file__).resolve(),
        "transformers_qwen35_moe": Path(
            importlib.import_module(
                "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
            ).__file__
        ).resolve(),
        "transformers_qwen35_moe_configuration": Path(
            importlib.import_module(
                "transformers.models.qwen3_5_moe.configuration_qwen3_5_moe"
            ).__file__
        ).resolve(),
    }
    source_receipts = {
        role: _source_receipt(path, role) for role, path in source_paths.items()
    }
    audit = {
        "distributions": distributions,
        "core_baseline_checks": package_checks,
        "baseline_source_checks": baseline_source_checks,
        "torch_runtime_checks": torch_checks,
        "fast_distribution_version_checks": fast_version_checks,
        "all_core_distributions_preserved": all_core_preserved,
        "all_fast_distribution_versions_exact": all_fast_versions_exact,
        "source_receipts": source_receipts,
    }
    return audit, source_receipts


def _fast_path_availability() -> dict:
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen
    from transformers.utils.import_utils import (
        is_causal_conv1d_available,
        is_flash_linear_attention_available,
    )

    bindings = {
        "causal_conv1d_fn": qwen.causal_conv1d_fn,
        "causal_conv1d_update": qwen.causal_conv1d_update,
        "chunk_gated_delta_rule": qwen.chunk_gated_delta_rule,
        "fused_recurrent_gated_delta_rule": qwen.fused_recurrent_gated_delta_rule,
        "FusedRMSNormGated": qwen.FusedRMSNormGated,
    }
    return {
        "transformers_is_causal_conv1d_available": is_causal_conv1d_available(),
        "transformers_is_flash_linear_attention_available": (
            is_flash_linear_attention_available()
        ),
        "qwen35_moe_module_is_fast_path_available": bool(
            qwen.is_fast_path_available
        ),
        "bindings": {
            name: {
                "available": value is not None,
                "module": getattr(value, "__module__", None),
                "qualname": getattr(value, "__qualname__", None),
            }
            for name, value in bindings.items()
        },
        "all_required_bindings_imported": all(
            value is not None for value in bindings.values()
        ),
        "availability_is_import_only_not_runtime_validation": True,
    }


def _gpu_inventory() -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows = []
    for row in csv.reader(completed.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 6:
            raise RuntimeError(f"unexpected nvidia-smi row: {row!r}")
        rows.append({
            "index": int(row[0]),
            "name": row[1],
            "uuid": row[2],
            "memory_total_mib": int(row[3]),
            "compute_capability": row[4],
            "driver_version": row[5],
        })
    if [item["index"] for item in rows] != list(GPU_INDICES):
        raise RuntimeError("exact physical four-GPU inventory changed")
    return rows


def _validate_gpu_baseline(inventory: list[dict], architecture: dict) -> dict:
    expected = architecture["hardware"]["gpus"]
    checks = []
    for actual, baseline in zip(inventory, expected, strict=True):
        check = {
            "index": actual["index"],
            "name_matches": actual["name"] == baseline["name"],
            "memory_matches": actual["memory_total_mib"]
            == baseline["memory_total_mib"],
            "compute_capability_matches": actual["compute_capability"]
            == baseline["compute_capability"],
            "driver_matches": actual["driver_version"]
            == baseline["driver_version"],
        }
        check["all_match"] = all(
            value for key, value in check.items() if key.endswith("_matches")
        )
        checks.append(check)
    return {
        "per_gpu": checks,
        "all_match_architecture_contract": len(inventory) == len(expected)
        and all(item["all_match"] for item in checks),
    }


def _cpu_uniform(
    torch: Any,
    shape: tuple[int, ...],
    low: float,
    high: float,
    seed: int,
    device: str = "cuda",
) -> Any:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.rand(shape, generator=generator, dtype=torch.float32)
    value = value * (high - low) + low
    return value.to(dtype=torch.bfloat16, device=device)


def _comparison_metrics(
    torch: Any,
    observed: Any,
    reference: Any,
    tolerance: dict,
) -> dict:
    observed_float = observed.detach().float()
    reference_float = reference.detach().float()
    difference = (observed_float - reference_float).abs()
    denominator = torch.maximum(
        observed_float.abs(), reference_float.abs()
    ).clamp_min(1e-5)
    return {
        "shape": list(observed.shape),
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
        "max_symmetric_relative_error": float((difference / denominator).max().item()),
        "atol": tolerance["atol"],
        "rtol": tolerance["rtol"],
        "passed": bool(
            torch.allclose(
                observed_float,
                reference_float,
                atol=tolerance["atol"],
                rtol=tolerance["rtol"],
            )
        ),
    }


def _gpu_properties(torch: Any, physical_index: int) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("worker must see exactly one CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    return {
        "physical_index": physical_index,
        "worker_visible_index": 0,
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": properties.total_memory,
    }


def _benchmark(
    torch: Any,
    step: Callable[[], None],
    *,
    tokens_per_iteration: int,
    warmup_iterations: int = WARMUP_ITERATIONS,
    benchmark_iterations: int = BENCHMARK_ITERATIONS,
) -> dict:
    if warmup_iterations < 1 or benchmark_iterations < 1:
        raise ValueError("benchmark iteration counts must be positive")
    for _ in range(warmup_iterations):
        step()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(benchmark_iterations):
        step()
    stop.record()
    stop.synchronize()
    elapsed_ms = float(start.elapsed_time(stop))
    seconds = elapsed_ms / 1000.0
    return {
        "mode": "forward_plus_backward",
        "warmup_iterations": warmup_iterations,
        "measured_iterations": benchmark_iterations,
        "tokens_per_iteration": tokens_per_iteration,
        "total_elapsed_ms": elapsed_ms,
        "milliseconds_per_iteration": elapsed_ms / benchmark_iterations,
        "tokens_per_second": (
            tokens_per_iteration * benchmark_iterations / seconds
        ),
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _make_causal_inputs(torch: Any) -> tuple[list[Any], Any]:
    spec = SYNTHETIC_WORKLOADS["causal_conv1d"]["shape"]
    base = SYNTHETIC_WORKLOADS["causal_conv1d"]["seed_base"]
    x = _cpu_uniform(
        torch,
        (spec["batch"], spec["channels"], spec["sequence"]),
        -0.5,
        0.5,
        base,
    ).requires_grad_()
    weight = _cpu_uniform(
        torch,
        (spec["channels"], spec["width"]),
        -0.25,
        0.25,
        base + 1,
    ).requires_grad_()
    bias = _cpu_uniform(
        torch,
        (spec["channels"],),
        -0.1,
        0.1,
        base + 2,
    ).requires_grad_()
    upstream = _cpu_uniform(
        torch,
        (spec["batch"], spec["channels"], spec["sequence"]),
        -0.5,
        0.5,
        base + 3,
    )
    return [x, weight, bias], upstream


def _run_causal(torch: Any, backend: str) -> tuple[Any, dict[str, Any]]:
    import torch.nn.functional as functional

    tensors, upstream = _make_causal_inputs(torch)
    x, weight, bias = tensors
    if backend == "fast":
        from causal_conv1d import causal_conv1d_fn

        output = causal_conv1d_fn(
            x=x,
            weight=weight,
            bias=bias,
            activation="silu",
        )
    elif backend == "torch_fallback":
        output = functional.silu(
            functional.conv1d(
                x,
                weight.unsqueeze(1),
                bias,
                padding=weight.shape[-1] - 1,
                groups=x.shape[1],
            )[..., : x.shape[-1]]
        )
    else:
        raise ValueError(backend)
    (output.float() * upstream.float()).sum().backward()
    gradients = {
        name: tensor.grad.detach()
        for name, tensor in zip(("x", "weight", "bias"), tensors, strict=True)
    }
    return output.detach(), gradients


def _causal_step(torch: Any, backend: str) -> Callable[[], None]:
    import torch.nn.functional as functional

    tensors, upstream = _make_causal_inputs(torch)
    x, weight, bias = tensors

    def step() -> None:
        for tensor in tensors:
            tensor.grad = None
        if backend == "fast":
            from causal_conv1d import causal_conv1d_fn

            output = causal_conv1d_fn(
                x=x,
                weight=weight,
                bias=bias,
                activation="silu",
            )
        else:
            output = functional.silu(
                functional.conv1d(
                    x,
                    weight.unsqueeze(1),
                    bias,
                    padding=weight.shape[-1] - 1,
                    groups=x.shape[1],
                )[..., : x.shape[-1]]
            )
        (output.float() * upstream.float()).sum().backward()

    return step


def _make_gated_inputs(torch: Any, workload: str) -> tuple[list[Any], Any]:
    import torch.nn.functional as functional

    spec = SYNTHETIC_WORKLOADS[workload]["shape"]
    base = SYNTHETIC_WORKLOADS[workload]["seed_base"]
    shape = (
        spec["batch"],
        spec["sequence"],
        spec["heads"],
        spec["key_head_dim"],
    )
    query = _cpu_uniform(torch, shape, -0.5, 0.5, base).requires_grad_()
    key = _cpu_uniform(torch, shape, -0.5, 0.5, base + 1).requires_grad_()
    value_shape = (*shape[:-1], spec["value_head_dim"])
    value = _cpu_uniform(
        torch, value_shape, -0.5, 0.5, base + 2
    ).requires_grad_()
    gate_shape = shape[:-1]
    raw_g = _cpu_uniform(torch, gate_shape, -1.0, 1.0, base + 3)
    g = (-functional.softplus(raw_g.float())).to(torch.bfloat16)
    g = g.detach().requires_grad_()
    raw_beta = _cpu_uniform(torch, gate_shape, -1.0, 1.0, base + 4)
    beta = torch.sigmoid(raw_beta.float()).to(torch.bfloat16)
    beta = beta.detach().requires_grad_()
    upstream = _cpu_uniform(torch, value_shape, -0.5, 0.5, base + 5)
    return [query, key, value, g, beta], upstream


def _run_chunk(torch: Any, backend: str) -> tuple[Any, dict[str, Any]]:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        torch_chunk_gated_delta_rule,
    )

    tensors, upstream = _make_gated_inputs(torch, "gated_delta_chunk")
    query, key, value, g, beta = tensors
    function = (
        chunk_gated_delta_rule
        if backend == "fast"
        else torch_chunk_gated_delta_rule
    )
    output, final_state = function(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    if final_state is not None:
        raise RuntimeError("training chunk unexpectedly returned final state")
    (output.float() * upstream.float()).sum().backward()
    gradients = {
        name: tensor.grad.detach()
        for name, tensor in zip(
            ("query", "key", "value", "g", "beta"), tensors, strict=True
        )
    }
    return output.detach(), gradients


def _chunk_step(torch: Any, backend: str) -> Callable[[], None]:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        torch_chunk_gated_delta_rule,
    )

    tensors, upstream = _make_gated_inputs(torch, "gated_delta_chunk")
    query, key, value, g, beta = tensors
    function = (
        chunk_gated_delta_rule
        if backend == "fast"
        else torch_chunk_gated_delta_rule
    )

    def step() -> None:
        for tensor in tensors:
            tensor.grad = None
        output, _ = function(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        (output.float() * upstream.float()).sum().backward()

    return step


def _run_recurrent_forward(torch: Any, backend: str) -> tuple[Any, Any]:
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        torch_recurrent_gated_delta_rule,
    )

    tensors, _ = _make_gated_inputs(torch, "gated_delta_recurrent")
    tensors = [tensor.detach() for tensor in tensors]
    query, key, value, g, beta = tensors
    function = (
        fused_recurrent_gated_delta_rule
        if backend == "fast"
        else torch_recurrent_gated_delta_rule
    )
    return function(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )


def _run_recurrent_backward(torch: Any, backend: str) -> dict[str, Any]:
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        torch_recurrent_gated_delta_rule,
    )

    tensors, upstream = _make_gated_inputs(torch, "gated_delta_recurrent")
    query, key, value, g, beta = tensors
    function = (
        fused_recurrent_gated_delta_rule
        if backend == "fast"
        else torch_recurrent_gated_delta_rule
    )
    output, _ = function(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    (output.float() * upstream.float()).sum().backward()
    return {
        name: tensor.grad.detach()
        for name, tensor in zip(
            ("query", "key", "value", "g", "beta"), tensors, strict=True
        )
    }


def _callable_path(value: Any) -> str | None:
    if value is None:
        return None
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if module is None or qualname is None:
        raise RuntimeError(f"binding has no stable callable identity: {value!r}")
    return f"{module}.{qualname}"


def _configure_qwen35_moe_gated_delta_modules(
    root_module: Any,
    *,
    policy: str,
) -> dict:
    """Apply an exact kernel policy to every gated-delta module below root."""
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen

    modules = [
        module
        for module in root_module.modules()
        if isinstance(module, qwen.Qwen3_5MoeGatedDeltaNet)
    ]
    if not modules:
        raise RuntimeError("root contains no Qwen3_5MoeGatedDeltaNet modules")
    if policy not in {"hybrid_training", "torch_reference"}:
        raise ValueError(policy)
    for module in modules:
        module.causal_conv1d_fn = None
        module.causal_conv1d_update = qwen.torch_causal_conv1d_update
        module.chunk_gated_delta_rule = (
            qwen.chunk_gated_delta_rule
            if policy == "hybrid_training"
            else qwen.torch_chunk_gated_delta_rule
        )
        module.recurrent_gated_delta_rule = qwen.torch_recurrent_gated_delta_rule

    observed = {
        "causal_conv1d_fn": _callable_path(modules[0].causal_conv1d_fn),
        "causal_conv1d_update": _callable_path(modules[0].causal_conv1d_update),
        "chunk_gated_delta_rule": _callable_path(
            modules[0].chunk_gated_delta_rule
        ),
        "recurrent_gated_delta_rule": _callable_path(
            modules[0].recurrent_gated_delta_rule
        ),
    }
    expected = (
        HYBRID_TRAINING_BINDINGS
        if policy == "hybrid_training"
        else SYNTHETIC_WORKLOADS["qwen35_moe_gated_delta_module"][
            "reference_bindings"
        ]
    )
    if observed != expected:
        raise RuntimeError(
            f"{policy} binding identity mismatch: {observed!r} != {expected!r}"
        )
    if any(
        {
            "causal_conv1d_fn": _callable_path(module.causal_conv1d_fn),
            "causal_conv1d_update": _callable_path(module.causal_conv1d_update),
            "chunk_gated_delta_rule": _callable_path(
                module.chunk_gated_delta_rule
            ),
            "recurrent_gated_delta_rule": _callable_path(
                module.recurrent_gated_delta_rule
            ),
        }
        != expected
        for module in modules
    ):
        raise RuntimeError(f"{policy} was not applied uniformly to every module")
    return {
        "policy": policy,
        "matched_module_count": len(modules),
        "all_matched_modules_configured": True,
        "bindings": observed,
        "normalization_binding_changed": False,
    }


def apply_qwen35_moe_training_hybrid(root_module: Any) -> dict:
    """Public training setup hook for the validated mixed-kernel policy."""
    return _configure_qwen35_moe_gated_delta_modules(
        root_module,
        policy="hybrid_training",
    )


def _make_qwen35_moe_gated_delta_modules(torch: Any) -> tuple[Any, Any, dict]:
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeGatedDeltaNet,
    )

    seed = SYNTHETIC_WORKLOADS["qwen35_moe_gated_delta_module"]["seed_base"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config = Qwen3_5MoeTextConfig(
        dtype=torch.bfloat16,
        **QWEN35_MOE_GATED_DELTA_GEOMETRY,
    )
    reference = Qwen3_5MoeGatedDeltaNet(config, layer_idx=0).to(
        device="cuda",
        dtype=torch.bfloat16,
    )
    hybrid = copy.deepcopy(reference)
    reference_bindings = _configure_qwen35_moe_gated_delta_modules(
        reference,
        policy="torch_reference",
    )
    hybrid_bindings = apply_qwen35_moe_training_hybrid(hybrid)
    realized_geometry = {
        "hidden_size": reference.hidden_size,
        "linear_conv_kernel_dim": reference.conv_kernel_size,
        "linear_key_head_dim": reference.head_k_dim,
        "linear_value_head_dim": reference.head_v_dim,
        "linear_num_key_heads": reference.num_k_heads,
        "linear_num_value_heads": reference.num_v_heads,
        "hidden_act": reference.activation,
        "rms_norm_eps": reference.layer_norm_epsilon,
    }
    if realized_geometry != QWEN35_MOE_GATED_DELTA_GEOMETRY:
        raise RuntimeError(
            "synthetic full-module geometry does not match the declared geometry"
        )
    if any(
        not torch.equal(reference.state_dict()[name], hybrid.state_dict()[name])
        for name in reference.state_dict()
    ):
        raise RuntimeError("deep-copied synthetic module parameters differ")
    return reference, hybrid, {
        "config_class": (
            "transformers.models.qwen3_5_moe.configuration_qwen3_5_moe."
            "Qwen3_5MoeTextConfig"
        ),
        "module_class": (
            "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe."
            "Qwen3_5MoeGatedDeltaNet"
        ),
        "realized_geometry": realized_geometry,
        "reference": reference_bindings,
        "hybrid": hybrid_bindings,
        "synthetic_parameter_state_copied_exactly": True,
        "pretrained_or_adapter_weights_loaded": False,
    }


def _make_qwen35_moe_module_inputs(
    torch: Any,
    sequence_length: int,
) -> tuple[Any, Any]:
    spec = SYNTHETIC_WORKLOADS["qwen35_moe_gated_delta_module"]
    shape = spec["shape"]
    if sequence_length not in shape["sequence_lengths"]:
        raise ValueError(sequence_length)
    generator = torch.Generator(device="cpu").manual_seed(
        spec["seed_base"] + sequence_length
    )
    tensor_shape = (shape["batch"], sequence_length, shape["hidden_size"])
    hidden_states = torch.rand(
        tensor_shape,
        generator=generator,
        dtype=torch.float32,
    )
    hidden_states = hidden_states * 0.2 - 0.1
    upstream = torch.rand(
        tensor_shape,
        generator=generator,
        dtype=torch.float32,
    )
    upstream = upstream * 0.2 - 0.1
    return (
        hidden_states.to(device="cuda", dtype=torch.bfloat16),
        upstream.to(device="cuda", dtype=torch.bfloat16),
    )


def _run_qwen35_moe_module(
    torch: Any,
    module: Any,
    hidden_states: Any,
    upstream: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    module.zero_grad(set_to_none=True)
    leaf_hidden_states = hidden_states.detach().clone().requires_grad_()
    output = module(
        leaf_hidden_states,
        cache_params=None,
        attention_mask=None,
    )
    (output.float() * upstream.float()).sum().backward()
    parameter_gradients = {}
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"full-module parameter has no gradient: {name}")
        parameter_gradients[name] = parameter.grad.detach().clone()
    if leaf_hidden_states.grad is None:
        raise RuntimeError("full-module hidden states have no gradient")
    return (
        output.detach(),
        leaf_hidden_states.grad.detach().clone(),
        parameter_gradients,
    )


def _qwen35_moe_module_step(
    torch: Any,
    module: Any,
    sequence_length: int,
) -> Callable[[], None]:
    hidden_states, upstream = _make_qwen35_moe_module_inputs(
        torch,
        sequence_length,
    )
    hidden_states.requires_grad_()

    def step() -> None:
        module.zero_grad(set_to_none=True)
        hidden_states.grad = None
        output = module(
            hidden_states,
            cache_params=None,
            attention_mask=None,
        )
        (output.float() * upstream.float()).sum().backward()

    return step


def _qwen35_moe_module_parity(
    torch: Any,
    reference: Any,
    hybrid: Any,
    sequence_length: int,
) -> dict:
    hidden_states, upstream = _make_qwen35_moe_module_inputs(
        torch,
        sequence_length,
    )
    reference_output, reference_input_gradient, reference_gradients = (
        _run_qwen35_moe_module(
            torch,
            reference,
            hidden_states,
            upstream,
        )
    )
    hybrid_output, hybrid_input_gradient, hybrid_gradients = (
        _run_qwen35_moe_module(
            torch,
            hybrid,
            hidden_states,
            upstream,
        )
    )
    if reference_gradients.keys() != hybrid_gradients.keys():
        raise RuntimeError("reference and hybrid parameter names differ")
    forward = _comparison_metrics(
        torch,
        hybrid_output,
        reference_output,
        BF16_TOLERANCES["qwen35_moe_gated_delta_module_forward"],
    )
    input_gradient = _comparison_metrics(
        torch,
        hybrid_input_gradient,
        reference_input_gradient,
        BF16_TOLERANCES["qwen35_moe_gated_delta_module_gradients"],
    )
    parameter_gradients = {
        name: _comparison_metrics(
            torch,
            hybrid_gradients[name],
            reference_gradients[name],
            BF16_TOLERANCES["qwen35_moe_gated_delta_module_gradients"],
        )
        for name in sorted(hybrid_gradients)
    }
    return {
        "sequence_length": sequence_length,
        "forward": forward,
        "input_gradient": input_gradient,
        "parameter_gradients": parameter_gradients,
        "all_parameter_gradients_present": True,
        "all_passed": forward["passed"]
        and input_gradient["passed"]
        and all(item["passed"] for item in parameter_gradients.values()),
    }


def _worker_qwen35_moe_hybrid_module(physical_index: int) -> dict:
    import torch

    reference, hybrid, module_contract = _make_qwen35_moe_gated_delta_modules(
        torch
    )
    sequence_lengths = SYNTHETIC_WORKLOADS["qwen35_moe_gated_delta_module"][
        "shape"
    ]["sequence_lengths"]
    parity = {
        str(sequence_length): _qwen35_moe_module_parity(
            torch,
            reference,
            hybrid,
            sequence_length,
        )
        for sequence_length in sequence_lengths
    }
    reference.zero_grad(set_to_none=True)
    hybrid.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    benchmarks = {}
    for sequence_length in sequence_lengths:
        reference_benchmark = _benchmark(
            torch,
            _qwen35_moe_module_step(torch, reference, sequence_length),
            tokens_per_iteration=sequence_length,
            benchmark_iterations=FULL_MODULE_BENCHMARK_ITERATIONS[
                sequence_length
            ],
        )
        hybrid_benchmark = _benchmark(
            torch,
            _qwen35_moe_module_step(torch, hybrid, sequence_length),
            tokens_per_iteration=sequence_length,
            benchmark_iterations=FULL_MODULE_BENCHMARK_ITERATIONS[
                sequence_length
            ],
        )
        benchmarks[str(sequence_length)] = {
            "hybrid_training": hybrid_benchmark,
            "torch_reference": reference_benchmark,
            "hybrid_throughput_speedup": (
                hybrid_benchmark["tokens_per_second"]
                / reference_benchmark["tokens_per_second"]
            ),
            "hybrid_peak_allocated_bytes_ratio": (
                hybrid_benchmark["peak_allocated_bytes"]
                / reference_benchmark["peak_allocated_bytes"]
            ),
        }
    return {
        "probe": "qwen35_moe_hybrid_module",
        "gpu": _gpu_properties(torch, physical_index),
        "status": "ok",
        "module_contract": module_contract,
        "parity": parity,
        "all_sequence_parity_passed": all(
            item["all_passed"] for item in parity.values()
        ),
        "benchmarks": benchmarks,
    }


def _worker_causal_fast(physical_index: int) -> dict:
    import torch

    result = {
        "probe": "causal_fast",
        "gpu": _gpu_properties(torch, physical_index),
        "status": "ok",
    }
    reference_output, reference_gradients = _run_causal(torch, "torch_fallback")
    try:
        fast_output, fast_gradients = _run_causal(torch, "fast")
        forward = _comparison_metrics(
            torch,
            fast_output,
            reference_output,
            BF16_TOLERANCES["causal_conv1d_forward"],
        )
        gradients = {
            name: _comparison_metrics(
                torch,
                fast_gradients[name],
                reference_gradients[name],
                BF16_TOLERANCES["causal_conv1d_gradients"],
            )
            for name in fast_gradients
        }
        result["parity"] = {
            "forward": forward,
            "gradients": gradients,
            "all_passed": forward["passed"]
            and all(item["passed"] for item in gradients.values()),
        }
        result["benchmark"] = _benchmark(
            torch,
            _causal_step(torch, "fast"),
            tokens_per_iteration=2 * 128,
        )
    except BaseException as error:
        result["status"] = "fast_kernel_error"
        result["error"] = {
            "type": type(error).__name__,
            "message": str(error)[:1000],
        }
        result["parity"] = {
            "status": "not_comparable_fast_kernel_returned_no_output_or_gradients"
        }
        result["benchmark"] = None
    return result


def _worker_causal_fallback(physical_index: int) -> dict:
    import torch

    return {
        "probe": "causal_fallback",
        "gpu": _gpu_properties(torch, physical_index),
        "status": "ok",
        "benchmark": _benchmark(
            torch,
            _causal_step(torch, "torch_fallback"),
            tokens_per_iteration=2 * 128,
        ),
    }


def _worker_gated_delta(physical_index: int) -> dict:
    import torch

    reference_output, reference_gradients = _run_chunk(torch, "torch_fallback")
    fast_output, fast_gradients = _run_chunk(torch, "fast")
    chunk_forward = _comparison_metrics(
        torch,
        fast_output,
        reference_output,
        BF16_TOLERANCES["gated_delta_chunk_forward"],
    )
    chunk_gradients = {
        name: _comparison_metrics(
            torch,
            fast_gradients[name],
            reference_gradients[name],
            BF16_TOLERANCES["gated_delta_chunk_gradients"],
        )
        for name in fast_gradients
    }

    fast_recurrent_output, fast_recurrent_state = _run_recurrent_forward(
        torch, "fast"
    )
    reference_recurrent_output, reference_recurrent_state = (
        _run_recurrent_forward(torch, "torch_fallback")
    )
    recurrent_forward = {
        "output": _comparison_metrics(
            torch,
            fast_recurrent_output,
            reference_recurrent_output,
            BF16_TOLERANCES["gated_delta_recurrent_forward"],
        ),
        "final_state": _comparison_metrics(
            torch,
            fast_recurrent_state,
            reference_recurrent_state,
            BF16_TOLERANCES["gated_delta_recurrent_forward"],
        ),
    }
    recurrent_backward: dict[str, Any]
    try:
        reference_recurrent_gradients = _run_recurrent_backward(
            torch, "torch_fallback"
        )
        fast_recurrent_gradients = _run_recurrent_backward(torch, "fast")
        recurrent_backward = {
            "status": "compared",
            "gradients": {
                name: _comparison_metrics(
                    torch,
                    fast_recurrent_gradients[name],
                    reference_recurrent_gradients[name],
                    BF16_TOLERANCES["gated_delta_recurrent_gradients"],
                )
                for name in fast_recurrent_gradients
            },
        }
        recurrent_backward["all_passed"] = all(
            item["passed"] for item in recurrent_backward["gradients"].values()
        )
    except BaseException as error:
        recurrent_backward = {
            "status": "fast_backward_unsupported",
            "error": {
                "type": type(error).__name__,
                "message": str(error)[:1000],
            },
            "gradients": None,
            "all_passed": False,
        }

    return {
        "probe": "gated_delta",
        "gpu": _gpu_properties(torch, physical_index),
        "status": "ok",
        "chunk_training_kernel_parity": {
            "forward": chunk_forward,
            "gradients": chunk_gradients,
            "all_passed": chunk_forward["passed"]
            and all(item["passed"] for item in chunk_gradients.values()),
        },
        "fused_recurrent_kernel_parity": {
            "forward": recurrent_forward,
            "forward_all_passed": all(
                item["passed"] for item in recurrent_forward.values()
            ),
            "backward": recurrent_backward,
        },
        "benchmarks": {
            "fast": _benchmark(
                torch,
                _chunk_step(torch, "fast"),
                tokens_per_iteration=64,
            ),
            "torch_fallback": _benchmark(
                torch,
                _chunk_step(torch, "torch_fallback"),
                tokens_per_iteration=64,
            ),
        },
    }


WORKER_FUNCTIONS = {
    "causal_fast": _worker_causal_fast,
    "causal_fallback": _worker_causal_fallback,
    "gated_delta": _worker_gated_delta,
    "qwen35_moe_hybrid_module": _worker_qwen35_moe_hybrid_module,
}


def _run_worker_process(probe: str, physical_index: int) -> dict:
    environment = dict(os.environ)
    environment.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(physical_index),
        "PYTHONHASHSEED": "0",
        "TOKENIZERS_PARALLELISM": "false",
    })
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-probe",
        probe,
        "--physical-gpu-index",
        str(physical_index),
    ]
    try:
        completed = subprocess.run(
            command,
            env=environment,
            capture_output=True,
            text=True,
            timeout=360,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "probe": probe,
            "status": "worker_timeout",
            "returncode": None,
            "stdout_sha256": hashlib.sha256(
                (error.stdout or "").encode("utf-8", errors="replace")
            ).hexdigest(),
            "stderr_sha256": hashlib.sha256(
                (error.stderr or "").encode("utf-8", errors="replace")
            ).hexdigest(),
        }
    stdout = completed.stdout
    stderr = completed.stderr
    process_receipt = {
        "returncode": completed.returncode,
        "termination_signal": (
            -completed.returncode if completed.returncode < 0 else None
        ),
        "termination_signal_name": (
            signal.Signals(-completed.returncode).name
            if completed.returncode < 0
            else None
        ),
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
    }
    marker_lines = [
        line[len(WORKER_MARKER) :]
        for line in stdout.splitlines()
        if line.startswith(WORKER_MARKER)
    ]
    if completed.returncode != 0 or len(marker_lines) != 1:
        return {
            "probe": probe,
            "status": "worker_process_error",
            **process_receipt,
            "error_tail": (stderr or stdout)[-1500:],
        }
    value = json.loads(marker_lines[0])
    value["process_receipt"] = process_receipt
    return value


def _selection_decision(
    environment: dict,
    availability: dict,
    gpu_results: list[dict],
) -> dict:
    hybrid_failures = []
    all_fast_rejection_reasons = []
    if not environment["all_core_distributions_preserved"]:
        hybrid_failures.append("core_distribution_or_source_drift")
    if not environment["all_fast_distribution_versions_exact"]:
        hybrid_failures.append("fast_distribution_version_drift")
    if not availability["all_required_bindings_imported"] or not availability[
        "qwen35_moe_module_is_fast_path_available"
    ]:
        hybrid_failures.append("transformers_qwen35_moe_fast_bindings_unavailable")
    for item in gpu_results:
        index = item["gpu"]["index"]
        causal_fast = item["probes"]["causal_fast"]
        causal_fallback = item["probes"]["causal_fallback"]
        gated = item["probes"]["gated_delta"]
        hybrid = item["probes"].get("qwen35_moe_hybrid_module", {})
        if causal_fast.get("status") != "ok" or not causal_fast.get(
            "parity", {}
        ).get("all_passed", False):
            all_fast_rejection_reasons.append(
                f"gpu_{index}_causal_conv1d_fast_runtime_or_parity_failed"
            )
        if causal_fallback.get("status") != "ok":
            hybrid_failures.append(
                f"gpu_{index}_causal_conv1d_fallback_benchmark_failed"
            )
        if gated.get("status") != "ok" or not gated.get(
            "chunk_training_kernel_parity", {}
        ).get("all_passed", False):
            hybrid_failures.append(
                f"gpu_{index}_gated_delta_chunk_parity_failed"
            )
        recurrent = gated.get("fused_recurrent_kernel_parity", {})
        if not recurrent.get("forward_all_passed", False):
            all_fast_rejection_reasons.append(
                f"gpu_{index}_gated_delta_recurrent_forward_failed"
            )
        if not recurrent.get("backward", {}).get("all_passed", False):
            all_fast_rejection_reasons.append(
                f"gpu_{index}_gated_delta_recurrent_backward_unavailable_or_failed"
            )
        if not gated.get("benchmarks", {}).get("fast"):
            hybrid_failures.append(
                f"gpu_{index}_gated_delta_fast_benchmark_failed"
            )
        if hybrid.get("status") != "ok":
            hybrid_failures.append(
                f"gpu_{index}_hybrid_full_module_worker_failed"
            )
            continue
        if not hybrid.get("all_sequence_parity_passed", False):
            hybrid_failures.append(
                f"gpu_{index}_hybrid_full_module_parity_failed"
            )
        module_contract = hybrid.get("module_contract", {})
        if module_contract.get("hybrid", {}).get("bindings") != (
            HYBRID_TRAINING_BINDINGS
        ) or not module_contract.get("hybrid", {}).get(
            "all_matched_modules_configured", False
        ):
            hybrid_failures.append(
                f"gpu_{index}_hybrid_binding_contract_failed"
            )
        for sequence_length in (128, 2048):
            benchmark = hybrid.get("benchmarks", {}).get(str(sequence_length), {})
            if not benchmark.get("hybrid_training") or not benchmark.get(
                "torch_reference"
            ):
                hybrid_failures.append(
                    f"gpu_{index}_hybrid_seq_{sequence_length}_benchmark_failed"
                )
        sequence_2048 = hybrid.get("benchmarks", {}).get("2048", {})
        if (
            sequence_2048.get("hybrid_throughput_speedup", 0.0) < 1.10
            and sequence_2048.get(
                "hybrid_peak_allocated_bytes_ratio", float("inf")
            )
            > 0.95
        ):
            hybrid_failures.append(
                f"gpu_{index}_hybrid_seq_2048_not_materially_better"
            )
    hybrid_failures = sorted(set(hybrid_failures))
    all_fast_rejection_reasons = sorted(set(all_fast_rejection_reasons))
    selected = "hybrid_training" if not hybrid_failures else "torch_fallback"
    return {
        "selected": selected,
        "selected_training_path": (
            "torch_causal_conv1d_plus_fla_chunk_gated_delta_plus_torch_recurrent"
            if selected == "hybrid_training"
            else "installed_transformers_all_torch_fallback"
        ),
        "selected_bindings": (
            HYBRID_TRAINING_BINDINGS if selected == "hybrid_training" else None
        ),
        "fast_path_import_available": availability[
            "qwen35_moe_module_is_fast_path_available"
        ],
        "fast_path_runtime_validated_on_all_four_gpus": False,
        "hybrid_training_path_runtime_validated_on_all_four_gpus": selected
        == "hybrid_training",
        "all_fast_path_rejection_reasons": all_fast_rejection_reasons,
        "hybrid_path_validation_failures": hybrid_failures,
        "reasons": (
            all_fast_rejection_reasons
            if selected == "hybrid_training"
            else hybrid_failures + all_fast_rejection_reasons
        ),
        "material_improvement_gate": {
            "sequence_length": 2048,
            "minimum_throughput_speedup": 1.10,
            "or_maximum_peak_allocated_bytes_ratio": 0.95,
            "passed_on_all_four_gpus": selected == "hybrid_training",
        },
        "fast_components_selected": (
            ["chunk_gated_delta_rule"]
            if selected == "hybrid_training"
            else []
        ),
        "torch_fallback_components_selected": (
            [
                "causal_conv1d_fn",
                "causal_conv1d_update",
                "recurrent_gated_delta_rule",
            ]
            if selected == "hybrid_training"
            else [
                "causal_conv1d_fn",
                "causal_conv1d_update",
                "chunk_gated_delta_rule",
                "recurrent_gated_delta_rule",
            ]
        ),
        "fallback_is_installed_transformers_torch_implementation": selected
        == "torch_fallback",
        "training_launch_authorized": False,
    }


def construct() -> dict:
    architecture = _load_self_addressed(ARCHITECTURE_CONTRACT)
    architecture_hidden_size = architecture["checkpoint"]["geometry"][
        "hidden_size"
    ]
    if architecture_hidden_size != QWEN35_MOE_GATED_DELTA_GEOMETRY[
        "hidden_size"
    ]:
        raise RuntimeError("full-module hidden size drifted from architecture contract")
    environment, _ = _audit_environment(architecture)
    availability = _fast_path_availability()
    gpu_inventory = _gpu_inventory()
    baseline_gpu_checks = _validate_gpu_baseline(gpu_inventory, architecture)
    def probe_gpu(gpu: dict) -> dict:
        index = gpu["index"]
        probes = {
            probe: _run_worker_process(probe, index)
            for probe in (
                "causal_fast",
                "causal_fallback",
                "gated_delta",
                "qwen35_moe_hybrid_module",
            )
        }
        return {"gpu": gpu, "probes": probes}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        gpu_results = list(executor.map(probe_gpu, gpu_inventory))
    decision = _selection_decision(environment, availability, gpu_results)
    value = {
        "schema": SCHEMA,
        "status": (
            "complete_hybrid_training_selected"
            if decision["selected"] == "hybrid_training"
            else "complete_torch_fallback_selected"
        ),
        "purpose": (
            "synthetic-only Qwen3.5-MoE fast-kernel parity, integrity, and "
            "four-GPU throughput contract"
        ),
        "baseline_architecture_contract": {
            "path": ARCHITECTURE_CONTRACT.relative_to(ROOT).as_posix(),
            "file_sha256": file_sha256(ARCHITECTURE_CONTRACT),
            "content_sha256": architecture["content_sha256_before_self_field"],
            "schema": architecture["schema"],
        },
        "environment_integrity": environment,
        "transformers_qwen35_moe_fast_path": availability,
        "synthetic_workloads": SYNTHETIC_WORKLOADS,
        "declared_bfloat16_tolerances": BF16_TOLERANCES,
        "hybrid_training_policy": {
            "scope": "every_Qwen3_5MoeGatedDeltaNet_below_model_root",
            "application_hook": (
                "build_fast_linear_attention_contract_v1."
                "apply_qwen35_moe_training_hybrid"
            ),
            "bindings": HYBRID_TRAINING_BINDINGS,
            "normalization_binding_changed": False,
            "geometry": QWEN35_MOE_GATED_DELTA_GEOMETRY,
            "geometry_provenance": {
                "hidden_size": (
                    "baseline_architecture_contract.checkpoint.geometry.hidden_size"
                ),
                "linear_attention_fields": (
                    "installed_Qwen3_5MoeTextConfig_explicit_defaults_and_"
                    "realized_Qwen3_5MoeGatedDeltaNet_attributes"
                ),
            },
            "sequence_lengths_validated": [128, 2048],
            "training_only_no_cache_parity_scope": True,
            "single_token_decode_not_selected_for_fast_recurrent": True,
        },
        "gpu_execution": {
            "selection_mode": (
                "parallel_one_isolated_worker_per_physical_gpu_with_sequential_probes"
            ),
            "worker_cuda_visible_device_count": 1,
            "warmup_iterations": WARMUP_ITERATIONS,
            "benchmark_iterations": BENCHMARK_ITERATIONS,
            "full_module_benchmark_iterations": (
                FULL_MODULE_BENCHMARK_ITERATIONS
            ),
            "gpu_baseline_checks": baseline_gpu_checks,
            "per_gpu_results": gpu_results,
        },
        "selected_fast_or_fallback": decision,
        "authority": {
            "synthetic_tensors_only": True,
            "model_or_adapter_weights_loaded": False,
            "datasets_or_training_rows_opened": False,
            "protected_holdout_ood_terminal_incident_or_manual_review_opened": False,
            "optimizer_created": False,
            "training_launched": False,
            "gpu_mutation_beyond_ephemeral_synthetic_allocations": False,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def _validate_checked_contract(value: dict, *, check_current: bool) -> None:
    if value.get("schema") != SCHEMA:
        raise RuntimeError("unsupported fast linear-attention contract schema")
    declared = value.get("content_sha256_before_self_field")
    unsigned = dict(value)
    unsigned.pop("content_sha256_before_self_field", None)
    if canonical_sha256(unsigned) != declared:
        raise RuntimeError("fast linear-attention contract content address mismatch")
    authority = value.get("authority", {})
    if authority != {
        "synthetic_tensors_only": True,
        "model_or_adapter_weights_loaded": False,
        "datasets_or_training_rows_opened": False,
        "protected_holdout_ood_terminal_incident_or_manual_review_opened": False,
        "optimizer_created": False,
        "training_launched": False,
        "gpu_mutation_beyond_ephemeral_synthetic_allocations": False,
    }:
        raise RuntimeError("synthetic-only authority boundary changed")
    results = value["gpu_execution"]["per_gpu_results"]
    if [item["gpu"]["index"] for item in results] != list(GPU_INDICES):
        raise RuntimeError("contract does not contain all four physical GPUs")
    expected_decision = _selection_decision(
        value["environment_integrity"],
        value["transformers_qwen35_moe_fast_path"],
        results,
    )
    if value["selected_fast_or_fallback"] != expected_decision:
        raise RuntimeError("fast-or-fallback decision is not reproducible")
    if check_current:
        architecture = _load_self_addressed(ARCHITECTURE_CONTRACT)
        if value["baseline_architecture_contract"] != {
            "path": ARCHITECTURE_CONTRACT.relative_to(ROOT).as_posix(),
            "file_sha256": file_sha256(ARCHITECTURE_CONTRACT),
            "content_sha256": architecture["content_sha256_before_self_field"],
            "schema": architecture["schema"],
        }:
            raise RuntimeError("architecture contract receipt changed")
        environment, current_sources = _audit_environment(architecture)
        if environment["distributions"] != value["environment_integrity"][
            "distributions"
        ]:
            raise RuntimeError("installed distribution receipts changed")
        if current_sources != value["environment_integrity"]["source_receipts"]:
            raise RuntimeError("installed implementation source receipts changed")
        if _gpu_inventory() != [item["gpu"] for item in results]:
            raise RuntimeError("physical GPU inventory changed")


def build(*, check: bool = False) -> dict:
    if check:
        if not OUTPUT.is_file():
            raise RuntimeError(f"checked contract is missing: {OUTPUT}")
        value = json.loads(OUTPUT.read_text(encoding="utf-8"))
        _validate_checked_contract(value, check_current=True)
        return value
    value = construct()
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{OUTPUT.name}.", dir=OUTPUT.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, OUTPUT)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return value


def _worker_main(probe: str, physical_index: int) -> None:
    function = WORKER_FUNCTIONS[probe]
    try:
        value = function(physical_index)
    except BaseException as error:
        value = {
            "probe": probe,
            "status": "worker_python_error",
            "error": {
                "type": type(error).__name__,
                "message": str(error)[:1500],
            },
        }
    print(WORKER_MARKER + json.dumps(value, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--worker-probe", choices=sorted(WORKER_FUNCTIONS))
    parser.add_argument("--physical-gpu-index", type=int)
    arguments = parser.parse_args()
    if arguments.worker_probe is not None:
        if arguments.physical_gpu_index not in GPU_INDICES:
            raise RuntimeError("worker requires a valid physical GPU index")
        _worker_main(arguments.worker_probe, arguments.physical_gpu_index)
        return
    value = build(check=arguments.check)
    print(json.dumps({
        "output": OUTPUT.relative_to(ROOT).as_posix(),
        "content_sha256": value["content_sha256_before_self_field"],
        "selected": value["selected_fast_or_fallback"]["selected"],
        "gpu_results": len(value["gpu_execution"]["per_gpu_results"]),
        "training_launched": value["authority"]["training_launched"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
