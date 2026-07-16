#!/usr/bin/env python3
"""Environment-bound fresh-path retry of the sealed V44A science protocol."""

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path

import run_matched_lora_candidate_eval_v44a as core


ROOT = Path(__file__).resolve().parent
EXPECTED_ENV_PREFIX = (ROOT / "es-at-scale/.venv").resolve()
FAILED_ATTEMPT_V44A = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    ".v44a_matched_lora_sft_es_fold3_ood_eval.attempt.json"
).resolve()
FAILED_RUN_V44A = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v44a_matched_lora_sft_es_fold3_ood_eval"
).resolve()
FAILED_FAILURE_V44A = FAILED_RUN_V44A / "failure_v44a.json"

EXPERIMENT = "v44b_matched_lora_sft_es_fold3_ood_eval_retry_env"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
RAW = (RUN_DIR / "raw_items_v44b.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v44b.jsonl").resolve()
REPORT = (
    ROOT / "experiments/eval_reports/"
    "matched_lora_sft_hpo_es_fold3_ood_eval_v44b.json"
).resolve()

FAILED_EXPECTED_V44A = {
    "attempt_file_sha256": (
        "40289a3314adb8f47b4c8d5f3d9e22b2d1d81aacb4fd50e8d8b1c13bf152f4f5"
    ),
    "attempt_content_sha256": (
        "72311635422848d4115d3dfd37f90a2a03bc1d124578d8e02f5f399305b5410c"
    ),
    "failure_file_sha256": (
        "f0635bbad947a0d41bb3d670eaeecdac70e8b345d931dd84fb32673811f5e664"
    ),
    "failure_content_sha256": (
        "958c3b1b2bf9e46cadcdc522fad2bd8d51afb7d90a7d62f53015b336bed964b1"
    ),
}


def _read_self_hashed_nonprotected_v44b(path: Path, file_sha: str,
                                         content_sha: str) -> dict:
    if core.file_sha256(path) != file_sha:
        raise RuntimeError(f"V44B retry provenance file changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != content_sha
        or core.canonical_sha256(compact) != content_sha
    ):
        raise RuntimeError(f"V44B retry provenance self-hash changed: {path}")
    return value


def failed_launch_provenance_v44b() -> dict:
    attempt = _read_self_hashed_nonprotected_v44b(
        FAILED_ATTEMPT_V44A,
        FAILED_EXPECTED_V44A["attempt_file_sha256"],
        FAILED_EXPECTED_V44A["attempt_content_sha256"],
    )
    failure = _read_self_hashed_nonprotected_v44b(
        FAILED_FAILURE_V44A,
        FAILED_EXPECTED_V44A["failure_file_sha256"],
        FAILED_EXPECTED_V44A["failure_content_sha256"],
    )
    if (
        attempt.get("phase") != "before_model_or_protected_semantic_access"
        or attempt.get("protected_semantic_access_count") != 0
        or attempt.get("heldout_or_holdout_opened") is not False
        or failure.get("type") != "ModuleNotFoundError"
        or failure.get("message") != "No module named 'vllm'"
        or failure.get("heldout_or_holdout_opened") is not False
        or "trainer, saved = make_trainer_v44a(prereg)" not in failure.get(
            "traceback", ""
        )
    ):
        raise RuntimeError("V44B failed-launch boundary changed")
    return {
        **FAILED_EXPECTED_V44A,
        "attempt": str(FAILED_ATTEMPT_V44A),
        "failure": str(FAILED_FAILURE_V44A),
        "failure_before_model_creation": True,
        "protected_semantic_access_count": 0,
        "heldout_or_holdout_opened": False,
        "scientific_observations_produced": False,
    }


def environment_bindings_v44b() -> dict:
    import es_at_scale
    import ray
    import torch
    import vllm

    prefix = Path(sys.prefix).resolve()
    if prefix != EXPECTED_ENV_PREFIX:
        raise RuntimeError(
            f"V44B requires {EXPECTED_ENV_PREFIX}/bin/python; got prefix {prefix}"
        )
    packages = {
        name: importlib.metadata.version(name)
        for name in ("vllm", "ray", "torch", "safetensors")
    }
    modules = {
        "vllm": str(Path(vllm.__file__).resolve()),
        "ray": str(Path(ray.__file__).resolve()),
        "torch": str(Path(torch.__file__).resolve()),
        "es_at_scale": str(Path(es_at_scale.__file__).resolve()),
    }
    expected_root = (ROOT / "es-at-scale").resolve()
    if not Path(modules["es_at_scale"]).is_relative_to(expected_root):
        raise RuntimeError("V44B es_at_scale import is outside the cloned repository")
    return {
        "sys_executable": str(Path(sys.executable).resolve()),
        "sys_prefix": str(prefix),
        "python_version": sys.version,
        "packages": packages,
        "module_files": modules,
        "vllm_importable": True,
        "four_tp1_runtime_environment": True,
    }


def implementation_bindings_v44b() -> dict:
    return {
        "retry_runtime": core.file_sha256(Path(__file__).resolve()),
        "core": core.implementation_bindings_v44a(),
        "worker_extension": core.file_sha256(
            ROOT / "eggroll_es_worker_lora_topology_v40a.py"
        ),
        "environment": environment_bindings_v44b(),
        "failed_launch": failed_launch_provenance_v44b(),
    }


def load_preregistration_v44b(args) -> dict:
    path = Path(args.preregistration).resolve()
    if core.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V44B preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        content != args.preregistration_content_sha256
        or content != core.canonical_sha256(compact)
        or value.get("schema")
        != "matched-lora-candidate-eval-preregistration-v44b"
        or value.get("status")
        != "preregistered_fresh_env_retry_before_single_semantic_access"
        or value.get("heldout_or_holdout_access_authorized") is not False
        or value.get("raw_shadow_or_ood_content_opened_before_preregistration")
        is not False
        or value.get("single_access_inputs") != core.PROTECTED_INPUTS_V44A
        or value.get("staged_adapters") != core.staged_adapter_bindings_v44a()
        or value.get("implementation_bindings") != implementation_bindings_v44b()
        or value.get("runtime", {}).get("tuned_table_content_sha256")
        != "4c4a0d4bbb400ea1d881bea3aae144d6865c34199fbb67889eda9e92d3a2543d"
    ):
        raise RuntimeError("V44B preregistration content changed")
    core._forbid_holdout_v44a(
        item["path"] for item in value["single_access_inputs"].values()
    )
    # As in V44A, protected inputs are neither opened nor hashed here.
    return value


def main(argv: list[str] | None = None) -> int:
    # Fail before preregistration parsing if the wrong Python environment is used.
    environment_bindings_v44b()
    saved = {
        "EXPERIMENT": core.EXPERIMENT,
        "RUN_DIR": core.RUN_DIR,
        "ATTEMPT": core.ATTEMPT,
        "RAW": core.RAW,
        "GPU_LOG": core.GPU_LOG,
        "REPORT": core.REPORT,
        "load": core.load_preregistration,
    }
    core.EXPERIMENT = EXPERIMENT
    core.RUN_DIR = RUN_DIR
    core.ATTEMPT = ATTEMPT
    core.RAW = RAW
    core.GPU_LOG = GPU_LOG
    core.REPORT = REPORT
    core.load_preregistration = load_preregistration_v44b
    try:
        return core.main(argv)
    finally:
        core.EXPERIMENT = saved["EXPERIMENT"]
        core.RUN_DIR = saved["RUN_DIR"]
        core.ATTEMPT = saved["ATTEMPT"]
        core.RAW = saved["RAW"]
        core.GPU_LOG = saved["GPU_LOG"]
        core.REPORT = saved["REPORT"]
        core.load_preregistration = saved["load"]


if __name__ == "__main__":
    raise SystemExit(main())
