#!/usr/bin/env python3
"""Isolated CPU RSS/copy benchmark for the V72 canonical LoRA state model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

import eggroll_es_host_state_contract_v72 as contract


SCHEMA = "eggroll-es-host-state-cpu-benchmark-v72"


def _rss_bytes_v72():
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            return int(fields[1]) * 1024
    raise RuntimeError("v72 VmRSS is unavailable")


def _child_peak_v72(bank_count):
    torch.set_num_threads(1)
    start_rss = _rss_bytes_v72()
    banks = []
    elements = contract.MASTER_ELEMENTS_V72
    for index in range(bank_count):
        tensor = torch.empty(elements, dtype=torch.float32)
        tensor.fill_(float(index + 1))
        banks.append(tensor)
    observed_rss = _rss_bytes_v72()
    checksum = float(sum(tensor[0].item() for tensor in banks))
    return {
        "bank_count": bank_count,
        "elements_per_bank": elements,
        "bytes_per_bank": contract.MASTER_BYTES_V72,
        "exact_tensor_bytes": bank_count * contract.MASTER_BYTES_V72,
        "rss_before_bytes": start_rss,
        "rss_after_bytes": observed_rss,
        "rss_delta_bytes": observed_rss - start_rss,
        "checksum": checksum,
    }


def _child_copy_v72(copy_passes):
    torch.set_num_threads(1)
    source = torch.arange(
        contract.MASTER_ELEMENTS_V72, dtype=torch.float32
    )
    started_ns = time.perf_counter_ns()
    checksum = 0.0
    for _index in range(copy_passes):
        copied = source.clone()
        checksum += float(copied[0].item() + copied[-1].item())
        del copied
    elapsed_ns = time.perf_counter_ns() - started_ns
    copied_bytes = copy_passes * contract.MASTER_BYTES_V72
    return {
        "copy_passes": copy_passes,
        "bytes_per_pass": contract.MASTER_BYTES_V72,
        "copied_bytes": copied_bytes,
        "elapsed_ns": elapsed_ns,
        "effective_gib_per_second": (
            copied_bytes / (1024 ** 3) / (elapsed_ns / 1_000_000_000)
        ),
        "checksum": checksum,
    }


def _child_result_v72(mode):
    if mode == "baseline_peak":
        result = _child_peak_v72(7)
    elif mode == "proposed_peak":
        result = _child_peak_v72(2)
    elif mode == "baseline_copy":
        result = _child_copy_v72(26)
    elif mode == "proposed_copy":
        result = _child_copy_v72(1)
    else:
        raise ValueError("v72 child benchmark mode changed")
    return {"mode": mode, **result}


def validate_benchmark_v72(value):
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise RuntimeError("v72 host benchmark schema changed")
    compact = {
        key: item for key, item in value.items()
        if key not in {"content_sha256_before_self_field", "validation"}
    }
    if value.get("content_sha256_before_self_field") != (
        contract.canonical_sha256_v72(compact)
    ):
        raise RuntimeError("v72 host benchmark content hash changed")
    arms = value.get("arms")
    if not isinstance(arms, dict) or set(arms) != {
        "baseline_peak", "proposed_peak", "baseline_copy", "proposed_copy"
    }:
        raise RuntimeError("v72 host benchmark arms changed")
    baseline_peak = arms["baseline_peak"]
    proposed_peak = arms["proposed_peak"]
    baseline_copy = arms["baseline_copy"]
    proposed_copy = arms["proposed_copy"]
    if (
        baseline_peak.get("bank_count") != 7
        or proposed_peak.get("bank_count") != 2
        or baseline_peak.get("exact_tensor_bytes")
        != 7 * contract.MASTER_BYTES_V72
        or proposed_peak.get("exact_tensor_bytes")
        != 2 * contract.MASTER_BYTES_V72
        or baseline_peak.get("rss_delta_bytes", 0)
        <= proposed_peak.get("rss_delta_bytes", 0)
        or proposed_peak.get("rss_delta_bytes", 0) <= 0
        or baseline_copy.get("copy_passes") != 26
        or proposed_copy.get("copy_passes") != 1
        or baseline_copy.get("copied_bytes")
        != 26 * contract.MASTER_BYTES_V72
        or proposed_copy.get("copied_bytes") != contract.MASTER_BYTES_V72
        or baseline_copy.get("elapsed_ns", 0) <= 0
        or proposed_copy.get("elapsed_ns", 0) <= 0
    ):
        raise RuntimeError("v72 host benchmark invariant changed")
    return {
        "schema": "eggroll-es-host-state-cpu-benchmark-certificate-v72",
        "passed": True,
        "exact_peak_tensor_bytes_saved": (
            baseline_peak["exact_tensor_bytes"]
            - proposed_peak["exact_tensor_bytes"]
        ),
        "observed_rss_delta_saved_bytes": (
            baseline_peak["rss_delta_bytes"]
            - proposed_peak["rss_delta_bytes"]
        ),
        "exact_copy_bytes_saved": (
            baseline_copy["copied_bytes"] - proposed_copy["copied_bytes"]
        ),
        "timing_is_diagnostic_not_a_live_acceptance_result": True,
    }


def run_benchmark_v72():
    arms = {}
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    for mode in (
        "baseline_peak", "proposed_peak", "baseline_copy", "proposed_copy"
    ):
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", mode],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        arms[mode] = json.loads(completed.stdout)
    value = {
        "schema": SCHEMA,
        "status": "cpu_synthetic_allocator_diagnostic",
        "arms": arms,
        "gpu_visible_to_children": False,
        "gpu_launch_performed": False,
        "dataset_or_protected_access_performed": False,
        "interpretation": (
            "exact tensor/copy bytes are normative; VmRSS and elapsed time are "
            "isolated-process allocator diagnostics, not Qwen live acceptance"
        ),
    }
    value["content_sha256_before_self_field"] = (
        contract.canonical_sha256_v72(value)
    )
    value["validation"] = validate_benchmark_v72(value)
    # Validation is derived after the self-hash and intentionally excluded
    # from it so a verifier can remove this field and reproduce the seal.
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark V72 canonical LoRA host-state residency."
    )
    parser.add_argument(
        "--child",
        choices=(
            "baseline_peak", "proposed_peak", "baseline_copy", "proposed_copy"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.child:
        print(json.dumps(_child_result_v72(args.child), sort_keys=True))
        return None
    value = run_benchmark_v72()
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return value


if __name__ == "__main__":
    main()
