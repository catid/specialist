#!/usr/bin/env python3
"""Synthetic-only four-GPU topology preflight for the fixed V59 adapter."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import run_lora_topology_probe_v40a as parent
import stage_v59_candidate_vllm_v60 as stage


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "v60p_v59_candidate_topology_preflight"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
REPORT = (RUN_DIR / "lora_topology_report_v60p.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v60p.jsonl").resolve()
DEFAULT_PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "v59_candidate_topology_preflight_v60p.json"
).resolve()
BUILDER = (ROOT / "build_v59_candidate_topology_preflight_v60p.py").resolve()
TESTS = (ROOT / "test_v59_candidate_topology_preflight_v60p.py").resolve()

EXPECTED = {
    "source_weights": "c2665b60928b16120a2b98fdf137fafd250644852c86a02d797689f02105c6c8",
    "source_config": "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5",
    "source_evidence": "136bb6d5e93a0e2a8187a670ae2e8760bed1dc08396dadeba76a6c053674d4ab",
    "staged_weights": "e189cfb9fcd6c55700babae91111266825522fc46ddbd53fc0574786711eccae",
    "staged_config": "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5",
    "stage_manifest": "c0e8c322f8c373dadd66506a2ad3859b7f9f18e56e56ff7b387999fb060f3d0e",
    "stage_manifest_content": "4b56da407f9a764157e6c9970ad1cc0b3695ad8e6ea9ee691cf73e274ba26bf6",
    "transformed_identity": "17426568952a5d23490886db52f9b2f8e648ebdf267b98789616dceb31d1a6c6",
    "candidate": "1713987fcad93f3e6368a309415faf5de2f4230eaf3c44baf23b8e9a2edf2a3d",
}


def file_sha256(path: Path) -> str:
    return parent.file_sha256(Path(path))


def canonical_sha256(value: object) -> str:
    return parent.canonical_sha256(value)


def stage_binding_v60p() -> dict:
    """Fail closed unless source, transformed bytes, and manifest are exact."""
    manifest_path = stage.OUTPUT / "stage_manifest_v44a.json"
    observed = {
        "source_weights": file_sha256(stage.SOURCE / "adapter_model.safetensors"),
        "source_config": file_sha256(stage.SOURCE / "adapter_config.json"),
        "source_evidence": file_sha256(stage.EVIDENCE),
        "staged_weights": file_sha256(stage.OUTPUT / "adapter_model.safetensors"),
        "staged_config": file_sha256(stage.OUTPUT / "adapter_config.json"),
        "stage_manifest": file_sha256(manifest_path),
    }
    if observed != {name: EXPECTED[name] for name in observed}:
        raise RuntimeError("V60P V59 adapter binding changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in manifest.items()
        if key != "content_sha256_before_self_field"
    }
    identity = manifest.get("transformed_identity", {})
    source = manifest.get("source", {})
    seal = source.get("seal", {})
    if (
        manifest.get("content_sha256_before_self_field")
        != EXPECTED["stage_manifest_content"]
        or canonical_sha256(compact) != EXPECTED["stage_manifest_content"]
        or manifest.get("schema") != "candidate-lora-vllm-stage-manifest-v44a"
        or manifest.get("status") != "complete_cpu_only_key_transform"
        or manifest.get("arm") != stage.ARM
        or manifest.get("dataset_or_evaluation_accessed") is not False
        or manifest.get("shadow_ood_holdout_or_heldout_accessed") is not False
        or manifest.get("gpu_accessed") is not False
        or source.get("weights_file_sha256") != EXPECTED["source_weights"]
        or source.get("adapter_config_file_sha256") != EXPECTED["source_config"]
        or seal.get("evidence_file_sha256") != EXPECTED["source_evidence"]
        or seal.get("canonical_candidate_sha256") != EXPECTED["candidate"]
        or seal.get("all_nine_train_endpoint_gates_passed") is not True
        or seal.get("four_actor_consensus_passed") is not True
        or seal.get(
            "wrapper_telemetry_false_negative_did_not_change_science"
        ) is not True
        or identity.get("sha256") != EXPECTED["transformed_identity"]
        or identity.get("tensor_count") != 70
        or identity.get("elements") != 4_528_128
        or identity.get("all_tensor_bytes_preserved_exactly") is not True
    ):
        raise RuntimeError("V60P V59 staged provenance changed")
    return {
        **observed,
        "stage_manifest_content": EXPECTED["stage_manifest_content"],
        "transformed_identity": EXPECTED["transformed_identity"],
        "candidate": EXPECTED["candidate"],
        "tensor_count": 70,
        "elements": 4_528_128,
        "all_tensor_bytes_preserved_exactly": True,
        "all_nine_train_endpoint_gates_passed": True,
        "dataset_or_evaluation_accessed": False,
    }


def implementation_bindings_v60p() -> dict:
    paths = {
        "runtime": Path(__file__).resolve(),
        "builder": BUILDER,
        "tests": TESTS,
        "parent_runtime": Path(parent.__file__).resolve(),
        "worker": ROOT / "eggroll_es_worker_lora_topology_v40a.py",
        "source_weights": stage.SOURCE / "adapter_model.safetensors",
        "source_config": stage.SOURCE / "adapter_config.json",
        "source_evidence": stage.EVIDENCE,
        "staged_weights": stage.OUTPUT / "adapter_model.safetensors",
        "staged_config": stage.OUTPUT / "adapter_config.json",
        "stage_manifest": stage.OUTPUT / "stage_manifest_v44a.json",
        "stage_runtime": Path(stage.__file__).resolve(),
        "canonical_stage_runtime": ROOT / "stage_candidate_adapters_vllm_v44a.py",
        "model_config": parent.MODEL / "config.json",
        "model_index": parent.MODEL / "model.safetensors.index.json",
        "tuned_table": parent.TUNED_FILE,
        "base_runtime": ROOT / "train_eggroll_es_specialist.py",
        "cleanup_runtime": ROOT / "run_eggroll_es_equal_unit_v38a.py",
    }
    result = {name: file_sha256(path) for name, path in paths.items()}
    result["model_shards_content_sha256"] = parent.MODEL_SHARDS_CONTENT_SHA256
    return result


def load_preregistration_v60p(args) -> dict:
    path = Path(args.preregistration).resolve()
    if file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V60P preregistration file changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        content != args.preregistration_content_sha256
        or content != canonical_sha256(compact)
        or value.get("schema")
        != "v59-candidate-topology-preflight-preregistration-v60p"
        or value.get("status") != "preregistered_before_four_gpu_launch"
        or value.get("implementation_bindings") != implementation_bindings_v60p()
        or value.get("adapter_binding") != stage_binding_v60p()
        or value.get("dataset_or_evaluation_access_authorized") is not False
        or value.get("synthetic_prompt_only") is not True
        or value.get("quality_claim_authorized") is not False
        or value.get("terminal_holdout_access_authorized") is not False
        or value.get("runtime", {}).get("physical_gpu_ids") != [0, 1, 2, 3]
        or value.get("runtime", {}).get("engine_count") != 4
        or value.get("runtime", {}).get("tuned_table_content_sha256")
        != "4c4a0d4bbb400ea1d881bea3aae144d6865c34199fbb67889eda9e92d3a2543d"
    ):
        raise RuntimeError("V60P preregistration content changed")
    forbidden = ("heldout", "holdout", "shadow", "ood", "train.jsonl")
    serialized = json.dumps(value["implementation_bindings"]).lower()
    if any(term in serialized for term in forbidden):
        raise RuntimeError("V60P bound a forbidden data or evaluation path")
    return value


def lora_request_v60p():
    from vllm.lora.request import LoRARequest
    return LoRARequest(
        "v59_fragile_priority_ratio0p25_v60p", 1, str(stage.OUTPUT),
        base_model_name=str(parent.MODEL),
    )


def attach_resolver_v60p(trainer, resolver):
    if not callable(resolver):
        raise TypeError("V60P resolver must be callable")
    trainer._resolve = lambda handles: resolver(handles)
    if not callable(getattr(trainer, "_resolve", None)):
        raise RuntimeError("V60P trainer resolver installation failed")
    return trainer


_ORIGINAL_MAKE_TRAINER = parent.make_trainer


def make_trainer_v60p(preregistration: dict):
    trainer = _ORIGINAL_MAKE_TRAINER(preregistration)
    import ray
    return attach_resolver_v60p(trainer, ray.get)


@contextmanager
def patched_parent_v60p():
    values = {
        "ADAPTER": stage.SOURCE,
        "ADAPTER_FILE": stage.SOURCE / "adapter_model.safetensors",
        "STAGED_ADAPTER": stage.OUTPUT,
        "STAGED_ADAPTER_FILE": stage.OUTPUT / "adapter_model.safetensors",
        "STAGE_MANIFEST": stage.OUTPUT / "stage_manifest_v44a.json",
        "EXPERIMENT": EXPERIMENT,
        "RUN_DIR": RUN_DIR,
        "ATTEMPT": ATTEMPT,
        "REPORT": REPORT,
        "GPU_LOG": GPU_LOG,
        "load_preregistration": load_preregistration_v60p,
        "_lora_request": lora_request_v60p,
        "make_trainer": make_trainer_v60p,
    }
    previous = {name: getattr(parent, name) for name in values}
    for name, value in values.items():
        setattr(parent, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(parent, name, value)


def main(argv: list[str] | None = None) -> int:
    with patched_parent_v60p():
        return parent.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
