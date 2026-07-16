#!/usr/bin/env python3
"""Build the immutable V56P synthetic-only topology preflight contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import run_v55b_candidate_topology_preflight_v56p as runtime
import run_lora_topology_probe_v40a as parent


def build_v56p() -> dict:
    config = json.loads((runtime.stage.SOURCE / "adapter_config.json").read_text())
    tuned = json.loads(parent.TUNED_FILE.read_text())
    if (
        config.get("r") != 32
        or config.get("lora_alpha") != 64
        or config.get("bias") != "none"
        or config.get("lora_dropout") != 0.0
        or config.get("layers_to_transform") != [20, 21, 22, 23]
    ):
        raise RuntimeError("V56P exact V55B adapter configuration changed")
    value = {
        "schema": "v55b-candidate-topology-preflight-preregistration-v56p",
        "status": "preregistered_before_four_gpu_launch",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "Prove the fixed V55B LoRA-ES snapshot loads on all four TP1 "
            "Qwen3.6 actors, has identical resident topology and deterministic "
            "synthetic forwards, responds to one reversible resident LoRA "
            "mutation, and cleans up exactly before the OOD-only gate."
        ),
        "implementation_bindings": runtime.implementation_bindings_v56p(),
        "adapter_binding": runtime.stage_binding_v56p(),
        "runtime": {
            "physical_gpu_ids": list(parent.GPU_IDS),
            "engine_count": 4,
            "tensor_parallel_size": 1,
            "all_four_actors_concurrent": True,
            "model": str(parent.MODEL),
            "source_adapter": str(runtime.stage.SOURCE),
            "staged_adapter": str(runtime.stage.OUTPUT),
            "adapter_rank": 32,
            "adapter_alpha": 64,
            "enable_lora": True,
            "max_loras": 1,
            "max_cpu_loras": 1,
            "tuned_folder": str(parent.TUNED_FOLDER),
            "tuned_table_content_sha256": runtime.canonical_sha256(tuned),
            "synthetic_prompt_sha256": hashlib.sha256(
                parent.SYNTHETIC_PROMPT.encode()
            ).hexdigest(),
        },
        "probe": {
            "peft_tensor_count_expected": 70,
            "peft_elements_expected": 4_528_128,
            "four_actor_forward_consensus_required": True,
            "resident_topology_semantic_consensus_required": True,
            "mutation": {
                "target": "model.layers.23.self_attn.o_proj.lora_B[0,0]",
                "requested_delta": 1.0,
                "one_element_only": True,
                "must_change_forward_without_reload": True,
                "must_restore_exact_original_forward": True,
            },
            "all_four_gpu_attributed_positive_activity_required": True,
            "exact_four_engine_cleanup_required": True,
        },
        "artifacts": {
            "run_directory": str(runtime.RUN_DIR),
            "attempt": str(runtime.ATTEMPT),
            "report": str(runtime.REPORT),
            "gpu_log": str(runtime.GPU_LOG),
        },
        "dataset_or_evaluation_access_authorized": False,
        "dataset_shadow_ood_holdout_or_heldout_paths_bound": False,
        "synthetic_prompt_only": True,
        "quality_claim_authorized": False,
        "terminal_holdout_access_authorized": False,
    }
    value["content_sha256_before_self_field"] = runtime.canonical_sha256(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(runtime.DEFAULT_PREREGISTRATION))
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    parent.atomic_json(output, build_v56p())
    value = json.loads(output.read_text())
    print(json.dumps({
        "path": str(output),
        "file_sha256": runtime.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "dataset_or_evaluation_accessed": False,
        "gpu_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
