#!/usr/bin/env python3
"""Seal the V40B resolver-only retry before using any GPU."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import build_lora_topology_probe_preregistration_v40a as build_v40a
import run_lora_topology_probe_v40a as v40a
import run_lora_topology_probe_v40b as v40b


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "v37_lora_topology_probe_resolver_retry_v40b.json"
).resolve()


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return value


def build() -> dict:
    prior = build_v40a.build()
    paths = {
        "retry_runtime": Path(v40b.__file__).resolve(),
        "runtime_v40a": Path(v40a.__file__).resolve(),
        "worker": ROOT / "eggroll_es_worker_lora_topology_v40a.py",
        "adapter_weights": v40a.ADAPTER_FILE,
        "adapter_config": v40a.ADAPTER / "adapter_config.json",
        "staged_adapter_weights": v40a.STAGED_ADAPTER_FILE,
        "staged_adapter_config": v40a.STAGED_ADAPTER / "adapter_config.json",
        "stage_manifest": v40a.STAGE_MANIFEST,
        "stage_runtime": ROOT / "stage_v37_adapter_for_vllm_v40a.py",
        "model_config": v40a.MODEL / "config.json",
        "model_index": v40a.MODEL / "model.safetensors.index.json",
        "tuned_table": v40a.TUNED_FILE,
        "base_runtime": ROOT / "train_eggroll_es_specialist.py",
        "cleanup_runtime": ROOT / "run_eggroll_es_equal_unit_v38a.py",
    }
    bindings = {key: v40a.file_sha256(path) for key, path in paths.items()}
    bindings["model_shards_content_sha256"] = v40a.MODEL_SHARDS_CONTENT_SHA256
    value = {
        key: item for key, item in prior.items()
        if key != "content_sha256_before_self_field"
    }
    value.update({
        "schema": "lora-topology-preregistration-v40b",
        "status": "preregistered_before_gpu_retry",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_bindings": bindings,
        "retry_of": {
            "experiment": v40a.EXPERIMENT,
            "failure": str(v40a.RUN_DIR / "failure_v40a.json"),
            "failure_sha256": v40a.file_sha256(v40a.RUN_DIR / "failure_v40a.json"),
            "phase": "post_model_load_pre_adapter_activation",
            "reason": "base trainer lacks later ES _resolve convenience method",
            "scientific_contract_changed": False,
            "coordinator_change": "resolve Ray object refs with ray.get",
        },
        "artifacts": {
            "run_directory": str(v40b.RUN_DIR), "attempt": str(v40b.ATTEMPT),
            "report": str(v40b.REPORT), "gpu_log": str(v40b.GPU_LOG),
        },
    })
    value["content_sha256_before_self_field"] = v40a.canonical_sha256(value)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    v40a.atomic_json(output, build())
    content = json.loads(output.read_text())["content_sha256_before_self_field"]
    print(json.dumps({
        "path": str(output), "file_sha256": v40a.file_sha256(output),
        "content_sha256": content,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
