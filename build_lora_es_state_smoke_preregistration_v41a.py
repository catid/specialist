#!/usr/bin/env python3
"""Build the fail-closed V41A canonical LoRA-ES state-smoke contract."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import run_lora_es_state_smoke_v41a as runtime


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "canonical_lora_es_state_smoke_v41a_retry_r2.json"
).resolve()
PARENT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "v37_lora_topology_probe_tuned_projection_retry_v40c.json"
).resolve()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    parent_content = parent.get("content_sha256_before_self_field")
    compact_parent = {
        key: value for key, value in parent.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        parent.get("schema") != "lora-topology-preregistration-v40c"
        or parent_content != runtime.v40a.canonical_sha256(compact_parent)
    ):
        raise RuntimeError("v41a parent V40C preregistration changed")
    tuned = parent["runtime"]["tuned_table_content_sha256"]
    value = {
        "schema": "canonical-lora-es-state-preregistration-v41a",
        "status": "preregistered_before_four_gpu_smoke",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Prove canonical FP32 LoRA state materialization, antithetic forward "
            "effect, exact restore, base immutability, and PEFT snapshot readback "
            "on four concurrent TP1 actors before any LoRA-ES training run."
        ),
        "dataset_or_evaluation_access_authorized": False,
        "sealed_holdout_opened": False,
        "selection_or_quality_claim_authorized": False,
        "parent_v40c": {
            "path": str(PARENT),
            "file_sha256": runtime.v40a.file_sha256(PARENT),
            "content_sha256": parent_content,
            "tuned_table_content_sha256": tuned,
        },
        "runtime": {
            "model": str(runtime.v40a.MODEL),
            "source_adapter": str(runtime.v40a.ADAPTER),
            "staged_adapter": str(runtime.v40a.STAGED_ADAPTER),
            "snapshot": str(runtime.SNAPSHOT),
            "worker_extension": runtime.WORKER_EXTENSION,
            "physical_gpu_ids": [0, 1, 2, 3],
            "engine_count": 4,
            "tensor_parallel_size": 1,
            "all_four_actors_concurrent": True,
            "canonical_dtype": "float32",
            "runtime_dtype": "bfloat16",
            "canonical_tensor_count": 70,
            "canonical_elements": 4_528_128,
            "runtime_module_count": 23,
            "runtime_view_count": 82,
            "runtime_elements": 4_921_344,
            "adapter_rank": 32,
            "adapter_alpha": 64,
            "seed": runtime.SEED,
            "sigma": runtime.SIGMA,
            "sign_order": [1, -1],
            "synthetic_prompt_sha256": runtime.v40a.hashlib.sha256(
                runtime.v40a.SYNTHETIC_PROMPT.encode("utf-8")
            ).hexdigest(),
            "tuned_folder": str(runtime.v40a.TUNED_FOLDER),
            "tuned_table_content_sha256": tuned,
        },
        "required_gates": {
            "source_forward_equals_canonical_install_forward": True,
            "plus_and_minus_each_change_forward": True,
            "plus_and_minus_are_distinguishable": True,
            "exact_restore_recovers_source_forward_after_each_sign": True,
            "relevant_base_weights_unchanged_at_every_state_boundary": True,
            "all_four_gpus_have_attributed_positive_activity": True,
            "one_rank_writes_snapshot_and_exact_readback_passes": True,
            "final_all_gpu_idle_cleanup": True,
        },
        "implementation_bindings": runtime._bindings(),
        "artifacts": {
            "attempt": str(runtime.ATTEMPT),
            "run_directory": str(runtime.RUN_DIR),
            "report": str(runtime.REPORT),
            "gpu_log": str(runtime.GPU_LOG),
            "snapshot": str(runtime.SNAPSHOT),
        },
    }
    value = runtime.v40a.self_hashed(value)
    runtime.v40a.atomic_json(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": runtime.v40a.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
