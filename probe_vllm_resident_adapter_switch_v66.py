#!/usr/bin/env python3
"""Run the V63 synthetic switch probe with both adapters GPU-resident.

This is a narrow diagnostic wrapper: it changes only ``max_loras`` from one to
two, preserving the model, prompts, call order, decoding, CPU adapter capacity,
and cleanup path of the bound V63 probe.  The receipt is upgraded and re-hashed
after the underlying probe exits.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import probe_vllm_two_adapter_switch_v63 as base


SCHEMA_V66 = "v66-synthetic-two-resident-adapter-switch-probe"


def output_argument_v66(argv):
    positions = [index for index, value in enumerate(argv) if value == "--output"]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise RuntimeError("resident-adapter probe requires one --output path")
    return Path(argv[positions[0] + 1]).resolve()


def upgraded_receipt_v66(value):
    if (
        not isinstance(value, dict)
        or value.get("schema")
        != "v63-synthetic-two-adapter-switch-feasibility-probe"
        or value.get("runtime", {}).get("max_loras") != 1
        or value.get("runtime", {}).get("max_cpu_loras") != 2
        or value.get("engine_shutdown_completed") is not True
        or value.get("adapter_update_or_hpo_performed") is not False
    ):
        raise RuntimeError("underlying V63 probe receipt changed")
    claimed = value.get("content_sha256_before_self_field")
    original = dict(value)
    original.pop("content_sha256_before_self_field", None)
    if base.base.canonical_sha256(original) != claimed:
        raise RuntimeError("underlying V63 probe receipt identity changed")
    result = dict(original)
    result["schema"] = SCHEMA_V66
    result["runtime"] = dict(result["runtime"])
    result["runtime"]["max_loras"] = 2
    result["runtime"]["both_adapters_have_resident_gpu_slots"] = True
    result["single_variable_change_from_v63"] = {"max_loras": [1, 2]}
    result["content_sha256_before_self_field"] = base.base.canonical_sha256(
        result
    )
    return result


def publish_upgraded_receipt_v66(path, value):
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n"
    )
    temporary = path.with_name(f".{path.name}.resident-v66-{os.getpid()}")
    with temporary.open("x", encoding="ascii") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main():
    output = output_argument_v66(sys.argv[1:])
    import vllm

    original_llm = vllm.LLM

    def resident_llm(*args, **kwargs):
        if kwargs.get("max_loras") != 1 or kwargs.get("max_cpu_loras") != 2:
            raise RuntimeError("V63 adapter-capacity baseline changed")
        kwargs = dict(kwargs)
        kwargs["max_loras"] = 2
        return original_llm(*args, **kwargs)

    vllm.LLM = resident_llm
    try:
        status = base.main()
    finally:
        vllm.LLM = original_llm
    if status != 0 or not output.is_file():
        raise RuntimeError("underlying resident-adapter probe failed")
    upgraded = upgraded_receipt_v66(json.loads(output.read_text(encoding="utf-8")))
    publish_upgraded_receipt_v66(output, upgraded)
    print(json.dumps({
        "schema": SCHEMA_V66,
        "output": str(output),
        "content_sha256": upgraded["content_sha256_before_self_field"],
        "wall_runtime_seconds": upgraded[
            "wall_runtime_seconds_excluding_model_load_and_cleanup"
        ],
        "reference_changed": upgraded[
            "reference_within_state_changed_rows"
        ],
        "candidate_changed": upgraded[
            "candidate_within_state_changed_rows"
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
