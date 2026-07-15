#!/usr/bin/env python3
"""Seal the V40C tuned-table runtime-projection retry."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import build_lora_topology_probe_preregistration_v40b as build_v40b
import run_lora_topology_probe_v40a as v40a
import run_lora_topology_probe_v40b as v40b
import run_lora_topology_probe_v40c as v40c


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (ROOT / "experiments/eggroll_es_hpo/preregistrations/"
                  "v37_lora_topology_probe_tuned_projection_retry_v40c.json").resolve()


def parser():
    value = argparse.ArgumentParser(); value.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return value


def build():
    prior = build_v40b.build()
    paths = {
        "retry_runtime": Path(v40c.__file__).resolve(),
        "runtime_v40a": Path(v40a.__file__).resolve(),
        "resolver_runtime_v40b": Path(v40b.__file__).resolve(),
        "worker": ROOT / "eggroll_es_worker_lora_topology_v40a.py",
        "adapter_weights": v40a.ADAPTER_FILE,
        "adapter_config": v40a.ADAPTER / "adapter_config.json",
        "staged_adapter_weights": v40a.STAGED_ADAPTER_FILE,
        "staged_adapter_config": v40a.STAGED_ADAPTER / "adapter_config.json",
        "stage_manifest": v40a.STAGE_MANIFEST,
        "stage_runtime": ROOT / "stage_v37_adapter_for_vllm_v40a.py",
        "model_config": v40a.MODEL / "config.json", "model_index": v40a.MODEL / "model.safetensors.index.json",
        "tuned_table": v40a.TUNED_FILE, "base_runtime": ROOT / "train_eggroll_es_specialist.py",
        "cleanup_runtime": ROOT / "run_eggroll_es_equal_unit_v38a.py",
    }
    bindings = {key: v40a.file_sha256(path) for key, path in paths.items()}
    bindings["model_shards_content_sha256"] = v40a.MODEL_SHARDS_CONTENT_SHA256
    tuned = json.loads(v40a.TUNED_FILE.read_text())
    file_content = v40a.canonical_sha256(tuned)
    tuned.pop("triton_version", None)
    projection = v40a.canonical_sha256({int(key): item for key, item in tuned.items()})
    value = {key: item for key, item in prior.items()
             if key != "content_sha256_before_self_field"}
    value.update({
        "schema": "lora-topology-preregistration-v40c",
        "status": "preregistered_before_tuned_projection_retry",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_bindings": bindings,
        "retry_of": {
            "experiment": v40b.EXPERIMENT,
            "failure": str(v40b.RUN_DIR / "failure_v40a.json"),
            "failure_sha256": v40a.file_sha256(v40b.RUN_DIR / "failure_v40a.json"),
            "phase": "post_model_load_pre_adapter_activation",
            "reason": "vLLM removes triton_version and normalizes keys in returned table",
            "scientific_contract_changed": False,
        },
        "artifacts": {"run_directory": str(v40c.RUN_DIR), "attempt": str(v40c.ATTEMPT),
                      "report": str(v40c.REPORT), "gpu_log": str(v40c.GPU_LOG)},
    })
    value["runtime"]["tuned_table_file_content_sha256"] = file_content
    value["runtime"]["tuned_table_content_sha256"] = projection
    value["content_sha256_before_self_field"] = v40a.canonical_sha256(value)
    return value


def main(argv=None):
    output = Path(parser().parse_args(argv).output).resolve()
    if output.exists(): raise FileExistsError(output)
    value = build(); v40a.atomic_json(output, value)
    print(json.dumps({"path": str(output), "file_sha256": v40a.file_sha256(output),
                      "content_sha256": value["content_sha256_before_self_field"]}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
