#!/usr/bin/env python3
"""Isolate vLLM compilation from CUDA graphs in the V63 switch probe.

The established V63 probe ties ``enforce_eager=False`` to its ``--graph``
flag.  This wrapper uses that flag only to permit compilation, then explicitly
sets ``VLLM_COMPILE`` with ``CUDAGraphMode.NONE``.  It upgrades the receipt
only after checking the resolved in-process vLLM configuration.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import probe_vllm_two_adapter_switch_v63 as base


SCHEMA_V72 = "v72-synthetic-two-adapter-compile-without-cudagraph-probe"


def output_argument_v72(argv: list[str]) -> Path:
    positions = [index for index, value in enumerate(argv) if value == "--output"]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise RuntimeError("V72 requires exactly one --output path")
    if "--graph" not in argv:
        raise RuntimeError("V72 requires --graph to disable enforce_eager")
    return Path(argv[positions[0] + 1]).resolve()


def resolved_compilation_v72(engine: object) -> dict:
    llm_engine = getattr(engine, "llm_engine", None)
    vllm_config = getattr(llm_engine, "vllm_config", None)
    config = getattr(vllm_config, "compilation_config", None)
    mode = getattr(getattr(config, "mode", None), "name", None)
    cudagraph = getattr(getattr(config, "cudagraph_mode", None), "name", None)
    if mode != "VLLM_COMPILE" or cudagraph != "NONE":
        raise RuntimeError(
            "V72 resolved configuration is not VLLM_COMPILE without CUDA graphs"
        )
    return {
        "compilation_mode": mode,
        "cudagraph_mode": cudagraph,
        "resolved_from_live_engine": True,
    }


def upgraded_receipt_v72(value: dict, resolved: dict) -> dict:
    if (
        not isinstance(value, dict)
        or value.get("schema")
        != "v63-synthetic-two-adapter-switch-feasibility-probe"
        or value.get("runtime", {}).get("enforce_eager") is not False
        or value.get("runtime", {}).get("cuda_graphs_enabled") is not True
        or value.get("engine_shutdown_completed") is not True
        or value.get("adapter_update_or_hpo_performed") is not False
    ):
        raise RuntimeError("V72 underlying V63 receipt changed")
    claimed = value.get("content_sha256_before_self_field")
    original = dict(value)
    original.pop("content_sha256_before_self_field", None)
    if base.base.canonical_sha256(original) != claimed:
        raise RuntimeError("V72 underlying V63 receipt identity changed")
    if resolved != {
        "compilation_mode": "VLLM_COMPILE",
        "cudagraph_mode": "NONE",
        "resolved_from_live_engine": True,
    }:
        raise RuntimeError("V72 resolved compilation certificate changed")

    result = dict(original)
    result["schema"] = SCHEMA_V72
    result["runtime"] = dict(result["runtime"])
    result["runtime"].update({
        "cuda_graphs_enabled": False,
        "compilation_mode": "VLLM_COMPILE",
        "cudagraph_mode": "NONE",
        "underlying_graph_flag_only_disabled_enforce_eager": True,
    })
    result["resolved_compilation_certificate"] = dict(resolved)
    result["single_variable_isolation"] = {
        "relative_to_v63_eager": "enable VLLM compilation, keep CUDA graphs off",
        "relative_to_v63_compiled_graph": "disable CUDA graphs, keep VLLM compilation",
    }
    result["content_sha256_before_self_field"] = base.base.canonical_sha256(result)
    return result


def publish_v72(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary = path.with_name(f".{path.name}.v72-{os.getpid()}")
    with temporary.open("x", encoding="ascii") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    output = output_argument_v72(sys.argv[1:])
    import vllm

    original_llm = vllm.LLM
    observed: list[dict] = []

    def compile_no_graph_llm(*args, **kwargs):
        if kwargs.get("enforce_eager") is not False:
            raise RuntimeError("V72 underlying eager contract changed")
        if "compilation_config" in kwargs:
            raise RuntimeError("V72 underlying compilation contract changed")
        kwargs = dict(kwargs)
        kwargs["compilation_config"] = {
            "mode": "VLLM_COMPILE",
            "cudagraph_mode": "NONE",
        }
        engine = original_llm(*args, **kwargs)
        observed.append(resolved_compilation_v72(engine))
        return engine

    vllm.LLM = compile_no_graph_llm
    try:
        status = base.main()
    finally:
        vllm.LLM = original_llm
    if status != 0 or not output.is_file() or len(observed) != 1:
        raise RuntimeError("V72 underlying compile-only probe failed")
    upgraded = upgraded_receipt_v72(
        json.loads(output.read_text(encoding="utf-8")), observed[0]
    )
    publish_v72(output, upgraded)
    print(json.dumps({
        "schema": SCHEMA_V72,
        "output": str(output),
        "content_sha256": upgraded["content_sha256_before_self_field"],
        "wall_runtime_seconds": upgraded[
            "wall_runtime_seconds_excluding_model_load_and_cleanup"
        ],
        "between_state_differing": upgraded["between_state_differing_rows"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
