#!/usr/bin/env python3
"""CPU-only installed-vLLM audit for V61E full-batch FCFS controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VLLM_ROOT = (
    ROOT / "es-at-scale/.venv/lib/python3.12/site-packages/vllm"
).resolve()
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "vllm_fullbatch_fcfs_support_audit_v61e.json"
).resolve()
SOURCE_PATHS = {
    "envs": VLLM_ROOT / "envs.py",
    "scheduler_config": VLLM_ROOT / "config/scheduler.py",
    "model_config": VLLM_ROOT / "config/model.py",
    "vllm_config": VLLM_ROOT / "config/vllm.py",
    "engine_arg_utils": VLLM_ROOT / "engine/arg_utils.py",
    "fused_moe": VLLM_ROOT / "model_executor/layers/fused_moe/fused_moe.py",
    "lora_triton_utils": VLLM_ROOT / "lora/ops/triton_ops/utils.py",
}


def file_sha256_v61e(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256_v61e(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")).hexdigest()


def build_audit_v61e() -> dict:
    os.environ.pop("VLLM_BATCH_INVARIANT", None)
    import vllm
    import vllm.envs as envs
    from vllm.config.scheduler import SchedulerConfig
    from vllm.engine.arg_utils import EngineArgs

    scheduler = SchedulerConfig.default_factory(
        max_model_len=2048,
        is_encoder_decoder=False,
        max_num_seqs=68,
        policy="fcfs",
        async_scheduling=False,
    )
    fields = EngineArgs.__dataclass_fields__
    controls_supported = (
        envs.VLLM_BATCH_INVARIANT is False
        and scheduler.max_num_seqs == 68
        and scheduler.policy == "fcfs"
        and scheduler.async_scheduling is False
        and scheduler.get_scheduler_cls().__name__ == "Scheduler"
        and all(key in fields for key in (
            "max_num_seqs",
            "scheduling_policy",
            "async_scheduling",
            "enforce_eager",
        ))
    )
    value = {
        "schema": "v61e-installed-vllm-fullbatch-fcfs-support-audit",
        "status": "supported" if controls_supported else "fail_closed_unsupported",
        "required_python": str(Path(os.sys.executable).resolve()),
        "vllm_version": str(vllm.__version__),
        "source_file_sha256": {
            key: file_sha256_v61e(path) for key, path in SOURCE_PATHS.items()
        },
        "requested_runtime_controls": {
            "VLLM_BATCH_INVARIANT": False,
            "async_scheduling": False,
            "max_num_seqs": 68,
            "scheduling_policy": "fcfs",
            "enforce_eager": True,
        },
        "batch_invariant_environment_resolved_false": (
            envs.VLLM_BATCH_INVARIANT is False
        ),
        "scheduler_config_projection": {
            "async_scheduling": scheduler.async_scheduling,
            "max_num_seqs": scheduler.max_num_seqs,
            "policy": scheduler.policy,
            "scheduler_class": scheduler.get_scheduler_cls().__name__,
        },
        "engine_args_fields_present": {
            key: key in fields for key in (
                "max_num_seqs",
                "scheduling_policy",
                "async_scheduling",
                "enforce_eager",
            )
        },
        "fullbatch_fcfs_controls_supported": controls_supported,
        "effective_request_batch_size": 68,
        "v61c_v27c_tuned_table_runtime_identity_must_remain_exact": True,
        "global_batch_invariance_claimed": False,
        "gpu_model_or_train_semantics_accessed": False,
        "gpu_launch_authorized_by_audit": controls_supported,
        "protected_semantics_opened": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256_v61e(value)
    return value


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_audit_v61e()
    _exclusive_write(
        output,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps({
        "path": str(output),
        "file_sha256": file_sha256_v61e(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "status": value["status"],
        "gpu_launch_authorized_by_audit": value[
            "gpu_launch_authorized_by_audit"
        ],
        "gpu_model_or_train_semantics_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
