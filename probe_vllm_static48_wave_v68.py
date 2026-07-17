#!/usr/bin/env python3
"""Run the V63 adapter-switch probe with a 48-sequence scheduler cap."""

from __future__ import annotations

import json
import sys

import probe_vllm_static_wave_v67 as wave32


base = wave32.base
SCHEMA_V68 = "v68-synthetic-static-48-sequence-wave-switch-probe"


def upgraded_receipt_v68(value):
    if (
        not isinstance(value, dict)
        or value.get("schema")
        != "v63-synthetic-two-adapter-switch-feasibility-probe"
        or value.get("runtime", {}).get("max_num_seqs") != 68
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
    result["schema"] = SCHEMA_V68
    result["runtime"] = dict(result["runtime"])
    result["runtime"]["max_num_seqs"] = 48
    result["runtime"]["submitted_prompt_count"] = base.base.ROWS
    result["runtime"]["bounded_internal_scheduling_waves"] = True
    result["single_variable_change_from_v63"] = {"max_num_seqs": [68, 48]}
    result["content_sha256_before_self_field"] = base.base.canonical_sha256(
        result
    )
    return result


def main():
    output = wave32.output_argument_v67(sys.argv[1:])
    import vllm

    original_llm = vllm.LLM

    def static_wave_llm(*args, **kwargs):
        if kwargs.get("max_num_seqs") != 68:
            raise RuntimeError("V63 sequence-capacity baseline changed")
        kwargs = dict(kwargs)
        kwargs["max_num_seqs"] = 48
        return original_llm(*args, **kwargs)

    vllm.LLM = static_wave_llm
    try:
        status = base.main()
    finally:
        vllm.LLM = original_llm
    if status != 0 or not output.is_file():
        raise RuntimeError("underlying static-wave probe failed")
    upgraded = upgraded_receipt_v68(json.loads(output.read_text(encoding="utf-8")))
    wave32.publish_upgraded_receipt_v67(output, upgraded)
    print(json.dumps({
        "schema": SCHEMA_V68,
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
