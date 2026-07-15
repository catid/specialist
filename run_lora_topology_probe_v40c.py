#!/usr/bin/env python3
"""Fresh V40C retry with the exact vLLM-normalized V27C table identity."""

from __future__ import annotations

import json
from pathlib import Path

import run_lora_topology_probe_v40a as v40a
import run_lora_topology_probe_v40b as v40b


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "v40c_v37_lora_topology_probe_tuned_projection_retry"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
REPORT = (RUN_DIR / "lora_topology_report_v40c.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v40c.jsonl").resolve()


def load_preregistration_v40c(args) -> dict:
    path = Path(args.preregistration).resolve()
    if v40a.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("v40c preregistration file identity changed")
    value = json.loads(path.read_text())
    content = value.get("content_sha256_before_self_field")
    if (
        content != args.preregistration_content_sha256
        or content != v40a.canonical_sha256({
            key: item for key, item in value.items()
            if key != "content_sha256_before_self_field"
        })
        or value.get("schema") != "lora-topology-preregistration-v40c"
        or value.get("status") != "preregistered_before_tuned_projection_retry"
        or value.get("dataset_or_evaluation_access_authorized") is not False
    ):
        raise RuntimeError("v40c preregistration content changed")
    paths = {
        "retry_runtime": Path(__file__).resolve(),
        "runtime_v40a": Path(v40a.__file__).resolve(),
        "resolver_runtime_v40b": Path(v40b.__file__).resolve(),
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
    observed = {key: v40a.file_sha256(path) for key, path in paths.items()}
    observed["model_shards_content_sha256"] = v40a.MODEL_SHARDS_CONTENT_SHA256
    if observed != value.get("implementation_bindings"):
        raise RuntimeError("v40c implementation or artifact binding changed")
    tuned = json.loads(v40a.TUNED_FILE.read_text())
    file_content = v40a.canonical_sha256(tuned)
    tuned.pop("triton_version", None)
    runtime_projection = v40a.canonical_sha256({int(key): item for key, item in tuned.items()})
    if (
        value["runtime"].get("tuned_table_file_content_sha256") != file_content
        or value["runtime"].get("tuned_table_content_sha256") != runtime_projection
    ):
        raise RuntimeError("v40c tuned table projection identity changed")
    return value


def make_trainer_v40c(prereg):
    trainer = _ORIGINAL_MAKE_TRAINER(prereg)
    import ray
    trainer._resolve = lambda handles: ray.get(handles)
    return trainer


_ORIGINAL_MAKE_TRAINER = v40a.make_trainer


def main(argv=None):
    saved = {
        "EXPERIMENT": v40a.EXPERIMENT, "RUN_DIR": v40a.RUN_DIR,
        "ATTEMPT": v40a.ATTEMPT, "REPORT": v40a.REPORT,
        "GPU_LOG": v40a.GPU_LOG, "load": v40a.load_preregistration,
        "make": v40a.make_trainer,
    }
    v40a.EXPERIMENT = EXPERIMENT; v40a.RUN_DIR = RUN_DIR
    v40a.ATTEMPT = ATTEMPT; v40a.REPORT = REPORT; v40a.GPU_LOG = GPU_LOG
    v40a.load_preregistration = load_preregistration_v40c
    v40a.make_trainer = make_trainer_v40c
    try:
        return v40a.main(argv)
    finally:
        v40a.EXPERIMENT = saved["EXPERIMENT"]; v40a.RUN_DIR = saved["RUN_DIR"]
        v40a.ATTEMPT = saved["ATTEMPT"]; v40a.REPORT = saved["REPORT"]
        v40a.GPU_LOG = saved["GPU_LOG"]
        v40a.load_preregistration = saved["load"]; v40a.make_trainer = saved["make"]


if __name__ == "__main__":
    raise SystemExit(main())
