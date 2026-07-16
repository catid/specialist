#!/usr/bin/env python3
"""Run a sealed four-GPU synthetic-only preflight for the V55B adapter."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import run_lora_topology_probe_v40a as parent
import stage_v55b_candidate_vllm_v56 as stage


ROOT = Path(__file__).resolve().parent
FAILED_EXPERIMENT = "v56p_v55b_candidate_topology_preflight"
FAILED_RUN_DIR = (
    ROOT / "experiments/eggroll_es_hpo/runs" / FAILED_EXPERIMENT
).resolve()
FAILED_ATTEMPT = (
    FAILED_RUN_DIR.parent / f".{FAILED_EXPERIMENT}.attempt.json"
).resolve()
FAILED_FAILURE = (FAILED_RUN_DIR / "failure_v40a.json").resolve()
FAILED_API_EXPERIMENT = "v56p_v55b_candidate_topology_preflight_retry1"
FAILED_API_RUN_DIR = (
    ROOT / "experiments/eggroll_es_hpo/runs" / FAILED_API_EXPERIMENT
).resolve()
FAILED_API_ATTEMPT = (
    FAILED_API_RUN_DIR.parent / f".{FAILED_API_EXPERIMENT}.attempt.json"
).resolve()
FAILED_API_FAILURE = (FAILED_API_RUN_DIR / "failure_v40a.json").resolve()
EXPERIMENT = "v56p_v55b_candidate_topology_preflight_retry2"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
REPORT = (RUN_DIR / "lora_topology_report_v56p.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v56p.jsonl").resolve()
DEFAULT_PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "v55b_candidate_topology_preflight_v56p_retry2.json"
).resolve()
BUILDER = (ROOT / "build_v55b_candidate_topology_preflight_v56p.py").resolve()
TESTS = (ROOT / "test_v55b_candidate_topology_preflight_v56p.py").resolve()

EXPECTED = {
    "source_weights": "d13a62107b29ca2a17682d2fa0d2eb424ef3eb90ad8aafc0bc0f5c5786b7bf9c",
    "source_config": "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5",
    "evidence": "3db9ad07e7d028a347dfc9e6a90a5ddcc3c87763439d13b80a1b33d86a0af96f",
    "staged_weights": "e30ab8173b4f979e6a5b4621908042cce66411246f3e242541ded945acaa7608",
    "staged_config": "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5",
    "stage_manifest": "43d779ddf1c37d11c0016159739832c431ea223b8b5e9345e29ac3e3ed341483",
    "stage_manifest_content": "a39359184a96b52008eabbce50f0f83ab1bda39513f6ed6c731dbdcdb093c6ce",
    "transformed_identity": "1638967e98ef1b677a651c49f6b97abb1656c4fce1b6776423aaccc3662e3cf5",
    "candidate": "78fa46a9f77387f7872a09202658e471b2c03969687abcbccc253f5f194980fc",
    "failed_attempt": "e3c587b1952316ae67ac7b226ba10da923d5715523a01e427e8dbc21165112a7",
    "failed_attempt_content": "1b377e4350ddcc48eb095a51c0f06b1f5b01bc8646c69be9ea8caab072b7b482",
    "failed_failure": "022b32b491766d6fff3b89806e4e4e51290c4eb145b3dd69fdb81ab2483a3ef2",
    "failed_failure_content": "60d806e3f37ea3da2d29fbb5091cfb66d93296dab97e7628a2e25ae7601323a5",
    "failed_api_attempt": "0ada9a1a78fc6158b5eed071289c4febf5507c6ad9491b9daa2f5d8cffcd5bb9",
    "failed_api_attempt_content": "665f2113b4c146b95bb1ff0896c3a2fe5e25ff5fd892b8c1e149b5a693af49c3",
    "failed_api_failure": "6e76e50827ecd552fff79057e15e9a574a6f05aaa149c4181a4fc7ea79e2f0e0",
    "failed_api_failure_content": "e575761b60b0a2b7c89855abfebb54f5a0bd3d4b5ba22dc1d217ab9214a809c6",
}


def file_sha256(path: Path) -> str:
    return parent.file_sha256(Path(path))


def canonical_sha256(value: object) -> str:
    return parent.canonical_sha256(value)


def stage_binding_v56p() -> dict:
    manifest_path = stage.OUTPUT / "stage_manifest_v44a.json"
    observed = {
        "source_weights": file_sha256(stage.SOURCE / "adapter_model.safetensors"),
        "source_config": file_sha256(stage.SOURCE / "adapter_config.json"),
        "evidence": file_sha256(stage.EVIDENCE),
        "staged_weights": file_sha256(stage.OUTPUT / "adapter_model.safetensors"),
        "staged_config": file_sha256(stage.OUTPUT / "adapter_config.json"),
        "stage_manifest": file_sha256(manifest_path),
    }
    if observed != {
        key: EXPECTED[key] for key in observed
    }:
        raise RuntimeError("V56P V55B adapter binding changed")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    identity = value.get("transformed_identity", {})
    source = value.get("source", {})
    seal = source.get("seal", {})
    if (
        content != EXPECTED["stage_manifest_content"]
        or content != canonical_sha256(compact)
        or value.get("schema") != "candidate-lora-vllm-stage-manifest-v44a"
        or value.get("status") != "complete_cpu_only_key_transform"
        or value.get("arm") != stage.ARM
        or value.get("dataset_or_evaluation_accessed") is not False
        or value.get("shadow_ood_holdout_or_heldout_accessed") is not False
        or value.get("gpu_accessed") is not False
        or source.get("weights_file_sha256") != EXPECTED["source_weights"]
        or source.get("adapter_config_file_sha256") != EXPECTED["source_config"]
        or seal.get("evidence_file_sha256") != EXPECTED["evidence"]
        or seal.get("canonical_candidate_sha256") != EXPECTED["candidate"]
        or seal.get("all_nine_train_endpoint_gates_passed") is not True
        or identity.get("sha256") != EXPECTED["transformed_identity"]
        or identity.get("tensor_count") != 70
        or identity.get("elements") != 4_528_128
        or identity.get("all_tensor_bytes_preserved_exactly") is not True
    ):
        raise RuntimeError("V56P staged V55B provenance changed")
    return {
        **observed,
        "stage_manifest_content": content,
        "transformed_identity": identity["sha256"],
        "candidate": seal["canonical_candidate_sha256"],
        "tensor_count": identity["tensor_count"],
        "elements": identity["elements"],
        "all_tensor_bytes_preserved_exactly": True,
        "all_nine_train_endpoint_gates_passed": True,
        "dataset_or_evaluation_accessed": False,
    }


def implementation_bindings_v56p() -> dict:
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
        "failed_attempt": FAILED_ATTEMPT,
        "failed_failure": FAILED_FAILURE,
        "failed_api_attempt": FAILED_API_ATTEMPT,
        "failed_api_failure": FAILED_API_FAILURE,
    }
    result = {key: file_sha256(path) for key, path in paths.items()}
    result["model_shards_content_sha256"] = parent.MODEL_SHARDS_CONTENT_SHA256
    return result


def recovery_binding_v56p() -> dict:
    if (
        file_sha256(FAILED_ATTEMPT) != EXPECTED["failed_attempt"]
        or file_sha256(FAILED_FAILURE) != EXPECTED["failed_failure"]
    ):
        raise RuntimeError("V56P failed launch evidence changed")
    attempt = json.loads(FAILED_ATTEMPT.read_text(encoding="utf-8"))
    failure = json.loads(FAILED_FAILURE.read_text(encoding="utf-8"))
    if (
        file_sha256(FAILED_API_ATTEMPT) != EXPECTED["failed_api_attempt"]
        or file_sha256(FAILED_API_FAILURE) != EXPECTED["failed_api_failure"]
    ):
        raise RuntimeError("V56P failed API-drift evidence changed")
    api_attempt = json.loads(FAILED_API_ATTEMPT.read_text(encoding="utf-8"))
    api_failure = json.loads(FAILED_API_FAILURE.read_text(encoding="utf-8"))
    attempt_content = attempt.pop("content_sha256_before_self_field", None)
    failure_content = failure.pop("content_sha256_before_self_field", None)
    api_attempt_content = api_attempt.pop(
        "content_sha256_before_self_field", None
    )
    api_failure_content = api_failure.pop(
        "content_sha256_before_self_field", None
    )
    if (
        attempt_content != EXPECTED["failed_attempt_content"]
        or attempt_content != canonical_sha256(attempt)
        or failure_content != EXPECTED["failed_failure_content"]
        or failure_content != canonical_sha256(failure)
        or attempt.get("status") != "launching"
        or attempt.get("phase") != "before_model_launch"
        or attempt.get("dataset_or_evaluation_accessed") is not False
        or failure.get("type") != "ModuleNotFoundError"
        or failure.get("message") != "No module named 'vllm'"
        or failure.get("dataset_or_evaluation_accessed") is not False
        or api_attempt_content != EXPECTED["failed_api_attempt_content"]
        or api_attempt_content != canonical_sha256(api_attempt)
        or api_failure_content != EXPECTED["failed_api_failure_content"]
        or api_failure_content != canonical_sha256(api_failure)
        or api_attempt.get("phase") != "before_model_launch"
        or api_attempt.get("dataset_or_evaluation_accessed") is not False
        or api_failure.get("type") != "AttributeError"
        or api_failure.get("message")
        != "'TopologyTrainerV40A' object has no attribute '_resolve'"
        or api_failure.get("dataset_or_evaluation_accessed") is not False
    ):
        raise RuntimeError("V56P failed launch provenance changed")
    return {
        "failed_experiment": FAILED_EXPERIMENT,
        "failed_attempt_file_sha256": EXPECTED["failed_attempt"],
        "failed_attempt_content_sha256": EXPECTED["failed_attempt_content"],
        "failed_failure_file_sha256": EXPECTED["failed_failure"],
        "failed_failure_content_sha256": EXPECTED["failed_failure_content"],
        "failure_type": "ModuleNotFoundError",
        "failure_message": "No module named 'vllm'",
        "failed_before_model_creation": True,
        "gpu_accessed": False,
        "dataset_or_evaluation_accessed": False,
        "recovery_interpreter": str(
            ROOT / "es-at-scale/.venv/bin/python"
        ),
        "api_drift_failed_experiment": FAILED_API_EXPERIMENT,
        "api_drift_attempt_file_sha256": EXPECTED["failed_api_attempt"],
        "api_drift_attempt_content_sha256": EXPECTED[
            "failed_api_attempt_content"
        ],
        "api_drift_failure_file_sha256": EXPECTED["failed_api_failure"],
        "api_drift_failure_content_sha256": EXPECTED[
            "failed_api_failure_content"
        ],
        "api_drift_failure_type": "AttributeError",
        "api_drift_failed_after_four_model_loads": True,
        "api_drift_cleanup_returned_all_gpus_to_4_mib": True,
        "resolver_recovery": "attach ray.get as trainer._resolve before use",
    }


def load_preregistration_v56p(args) -> dict:
    path = Path(args.preregistration).resolve()
    if file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V56P preregistration file changed")
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
        != "v55b-candidate-topology-preflight-preregistration-v56p"
        or value.get("status") != "preregistered_before_four_gpu_launch"
        or value.get("implementation_bindings") != implementation_bindings_v56p()
        or value.get("adapter_binding") != stage_binding_v56p()
        or value.get("recovery") != recovery_binding_v56p()
        or value.get("dataset_or_evaluation_access_authorized") is not False
        or value.get("synthetic_prompt_only") is not True
        or value.get("quality_claim_authorized") is not False
        or value.get("runtime", {}).get("physical_gpu_ids") != [0, 1, 2, 3]
        or value.get("runtime", {}).get("engine_count") != 4
        or value.get("runtime", {}).get("tuned_table_content_sha256")
        != canonical_sha256(json.loads(parent.TUNED_FILE.read_text()))
    ):
        raise RuntimeError("V56P preregistration content changed")
    forbidden = ("heldout", "holdout", "shadow", "ood", "train.jsonl")
    serialized = json.dumps(value["implementation_bindings"]).lower()
    if any(term in serialized for term in forbidden):
        raise RuntimeError("V56P bound a forbidden data/evaluation path")
    return value


def lora_request_v56p():
    from vllm.lora.request import LoRARequest
    return LoRARequest(
        "v55b_maximin_ratio0p25_v56p",
        1,
        str(stage.OUTPUT),
        base_model_name=str(parent.MODEL),
    )


def attach_resolver_v56p(trainer, resolver):
    """Install the current V40C resolver surface before any caller uses it."""
    if not callable(resolver):
        raise TypeError("V56P resolver must be callable")
    trainer._resolve = lambda handles: resolver(handles)
    if not callable(getattr(trainer, "_resolve", None)):
        raise RuntimeError("V56P trainer resolver installation failed")
    return trainer


_ORIGINAL_MAKE_TRAINER = parent.make_trainer


def make_trainer_v56p(preregistration: dict):
    trainer = _ORIGINAL_MAKE_TRAINER(preregistration)
    import ray
    return attach_resolver_v56p(trainer, ray.get)


@contextmanager
def _patched_parent_v56p():
    names = {
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
        "load_preregistration": load_preregistration_v56p,
        "_lora_request": lora_request_v56p,
        "make_trainer": make_trainer_v56p,
    }
    previous = {name: getattr(parent, name) for name in names}
    for name, value in names.items():
        setattr(parent, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(parent, name, value)


def main(argv: list[str] | None = None) -> int:
    with _patched_parent_v56p():
        return parent.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
