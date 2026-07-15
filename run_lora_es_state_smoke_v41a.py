#!/usr/bin/env python3
"""Preregistered four-GPU canonical LoRA-ES state smoke for V41A.

This run is deliberately train-only and synthetic-prompt-only.  It proves that
the canonical FP32 PEFT state can drive the resident vLLM LoRA buffers, that an
antithetic perturbation changes the next forward without an adapter reload,
that exact restoration recovers the original forward, and that a standard
PEFT snapshot can be written and reopened.  It does not read validation, OOD,
or holdout data and it makes no quality claim.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import run_lora_topology_probe_v40a as v40a
import run_lora_topology_probe_v40c as v40c


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "v41a_canonical_lora_es_state_smoke_retry_r2"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
REPORT = (RUN_DIR / "lora_es_state_smoke_report_v41a_retry_r2.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v41a_retry_r2.jsonl").resolve()
SNAPSHOT = (RUN_DIR / "canonical_adapter_snapshot_v41a_retry_r2").resolve()
WORKER_EXTENSION = (
    "eggroll_es_worker_lora_v41a.LoRAAdapterStateWorkerExtensionV41A"
)
SEED = 41_202_607_15
SIGMA = 0.0003


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--preregistration", required=True)
    result.add_argument("--preregistration-sha256", required=True)
    result.add_argument("--preregistration-content-sha256", required=True)
    result.add_argument("--dry-run", action="store_true")
    return result


def _bindings() -> dict[str, str]:
    paths = {
        "runtime": Path(__file__).resolve(),
        "builder": ROOT / "build_lora_es_state_smoke_preregistration_v41a.py",
        "worker": ROOT / "eggroll_es_worker_lora_v41a.py",
        "runtime_v40a": Path(v40a.__file__).resolve(),
        "resolver_runtime_v40c": Path(v40c.__file__).resolve(),
        "adapter_weights": v40a.ADAPTER_FILE,
        "adapter_config": v40a.ADAPTER / "adapter_config.json",
        "staged_adapter_weights": v40a.STAGED_ADAPTER_FILE,
        "staged_adapter_config": v40a.STAGED_ADAPTER / "adapter_config.json",
        "stage_manifest": v40a.STAGE_MANIFEST,
        "model_config": v40a.MODEL / "config.json",
        "model_index": v40a.MODEL / "model.safetensors.index.json",
        "tuned_table": v40a.TUNED_FILE,
        "base_runtime": ROOT / "train_eggroll_es_specialist.py",
        "cleanup_runtime": ROOT / "run_eggroll_es_equal_unit_v38a.py",
    }
    observed = {key: v40a.file_sha256(path) for key, path in paths.items()}
    observed["model_shards_content_sha256"] = v40a.MODEL_SHARDS_CONTENT_SHA256
    return observed


def load_preregistration(args: argparse.Namespace) -> dict:
    path = Path(args.preregistration).resolve()
    if v40a.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("v41a preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    compact = {key: item for key, item in value.items()
               if key != "content_sha256_before_self_field"}
    if (
        content != args.preregistration_content_sha256
        or content != v40a.canonical_sha256(compact)
        or value.get("schema") != "canonical-lora-es-state-preregistration-v41a"
        or value.get("status") != "preregistered_before_four_gpu_smoke"
        or value.get("dataset_or_evaluation_access_authorized") is not False
        or value.get("sealed_holdout_opened") is not False
        or value.get("implementation_bindings") != _bindings()
    ):
        raise RuntimeError("v41a preregistration contract changed")
    runtime = value.get("runtime", {})
    if (
        runtime.get("seed") != SEED or runtime.get("sigma") != SIGMA
        or runtime.get("physical_gpu_ids") != [0, 1, 2, 3]
        or runtime.get("worker_extension") != WORKER_EXTENSION
        or runtime.get("source_adapter") != str(v40a.ADAPTER)
        or runtime.get("staged_adapter") != str(v40a.STAGED_ADAPTER)
        or runtime.get("snapshot") != str(SNAPSHOT)
        or runtime.get("tuned_table_content_sha256")
        != value.get("parent_v40c", {}).get("tuned_table_content_sha256")
    ):
        raise RuntimeError("v41a runtime recipe changed")
    forbidden = ("heldout", "holdout", "ood", "eval_qa", "shadow")
    serialized = json.dumps(value, sort_keys=True).lower()
    # The two negative boolean field names are allowed; no bound path may use them.
    for key in ("source_adapter", "staged_adapter", "snapshot", "model"):
        bound = str(runtime.get(key, "")).lower()
        if any(token in bound for token in forbidden):
            raise RuntimeError(f"v41a forbidden data/evaluation path in {key}")
    if "dataset_or_evaluation_access_authorized" not in serialized:
        raise RuntimeError("v41a access prohibition is missing")
    return value


def _actor_pid_map(actor_ids: list[dict]) -> dict[int, int]:
    result = {}
    for item in actor_ids:
        gpu = int(item["physical_gpu_id"])
        pid = int(item["pid"])
        if gpu in result:
            raise RuntimeError("v41a duplicate physical GPU actor")
        result[gpu] = pid
    if set(result) != set(v40a.GPU_IDS) or len(set(result.values())) != 4:
        raise RuntimeError("v41a physical GPU/PID coverage changed")
    return result


def _semantic_consensus(values: list[dict], phase: str) -> str:
    if len(values) != 4:
        raise RuntimeError(f"v41a incomplete four-rank result at {phase}")
    hashes = [v40a.canonical_sha256(value) for value in values]
    if len(set(hashes)) != 1:
        raise RuntimeError(f"v41a semantic rank disagreement at {phase}")
    return hashes[0]


def _master_identity(values: list[dict], field: str) -> dict:
    identities = [value[field] for value in values]
    _semantic_consensus(identities, field)
    return identities[0]


def _make_trainer(prereg: dict):
    saved = {
        "experiment": v40a.EXPERIMENT,
        "run_dir": v40a.RUN_DIR,
        "worker": v40a.WORKER_EXTENSION,
    }
    v40a.EXPERIMENT = EXPERIMENT
    v40a.RUN_DIR = RUN_DIR
    v40a.WORKER_EXTENSION = WORKER_EXTENSION
    try:
        trainer = v40c.make_trainer_v40c(prereg)
    except Exception:
        v40a.EXPERIMENT = saved["experiment"]
        v40a.RUN_DIR = saved["run_dir"]
        v40a.WORKER_EXTENSION = saved["worker"]
        raise
    return trainer, saved


def _restore_v40_globals(saved: dict) -> None:
    v40a.EXPERIMENT = saved["experiment"]
    v40a.RUN_DIR = saved["run_dir"]
    v40a.WORKER_EXTENSION = saved["worker"]


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    prereg = load_preregistration(args)
    if args.dry_run:
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "dataset_or_evaluation_accessed": False,
            "sealed_holdout_opened": False,
        }, sort_keys=True))
        return 0
    if ATTEMPT.exists() or RUN_DIR.exists():
        raise RuntimeError("v41a requires fresh artifact paths")
    preflight = v40a.gpu_preflight()
    attempt = v40a.self_hashed({
        "schema": "canonical-lora-es-state-attempt-v41a",
        "status": "launching", "phase": "before_model_launch",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_file_sha256": args.preregistration_sha256,
        "preregistration_content_sha256": args.preregistration_content_sha256,
        "preflight": preflight, "dataset_or_evaluation_accessed": False,
        "sealed_holdout_opened": False,
    })
    v40a.atomic_json(ATTEMPT, attempt)
    RUN_DIR.mkdir(parents=True)
    trainer = monitor = saved = None
    stop = threading.Event()
    failures: queue.Queue = queue.Queue()
    phase = v40a.Phase()
    started = time.monotonic()
    try:
        v40a.base.set_seed(SEED)
        trainer, saved = _make_trainer(prereg)
        actor_ids = trainer._resolve([
            engine.runtime_identity_v40a.remote() for engine in trainer.engines
        ])
        pid_map = _actor_pid_map(actor_ids)
        monitor = threading.Thread(
            target=v40a.monitor_gpus,
            args=(stop, phase, pid_map, GPU_LOG, failures), daemon=True,
        )
        monitor.start()

        phase.value = "activate_source_adapter"
        baseline = v40a.probe_forward(trainer)
        phase.value = "install_canonical_fp32_master"
        installs = v40a._rpc_all(trainer, "install_adapter_state_v41a", (
            str(v40a.ADAPTER_FILE), str(v40a.ADAPTER / "adapter_config.json"),
            v40a.file_sha256(v40a.ADAPTER_FILE),
            v40a.file_sha256(v40a.ADAPTER / "adapter_config.json"),
        ))
        install_consensus = _semantic_consensus(installs, "install")
        installed_forward = v40a.probe_forward(trainer)
        if installed_forward["consensus_output_sha256"] != baseline["consensus_output_sha256"]:
            raise RuntimeError("v41a canonical install changed the source-adapter forward")

        certificates = v40a._rpc_all(trainer, "adapter_state_certificate_v41a")
        certificate_consensus = _semantic_consensus(certificates, "certificate")
        master = _master_identity(certificates, "current_identity")
        references = v40a._rpc_all(trainer, "capture_adapter_reference_v41a")
        reference_consensus = _semantic_consensus(references, "reference")

        perturbations = {}
        forwards = {}
        restorations = {}
        for sign, label in ((1, "plus"), (-1, "minus")):
            phase.value = f"antithetic_{label}"
            values = v40a._rpc_all(
                trainer, "materialize_antithetic_adapter_v41a",
                (SEED, SIGMA, sign, master["sha256"]),
            )
            perturbations[label] = {
                "consensus_sha256": _semantic_consensus(values, label),
                "values": values,
            }
            forwards[label] = v40a.probe_forward(trainer)
            phase.value = f"restore_after_{label}"
            restored = v40a._rpc_all(trainer, "restore_adapter_master_v41a")
            restorations[label] = {
                "consensus_sha256": _semantic_consensus(restored, f"restore_{label}"),
                "values": restored,
            }
            after = v40a.probe_forward(trainer)
            restorations[label]["forward"] = after
            if after["consensus_output_sha256"] != baseline["consensus_output_sha256"]:
                raise RuntimeError(f"v41a exact restore failed after {label}")

        base_hash = baseline["consensus_output_sha256"]
        plus_hash = forwards["plus"]["consensus_output_sha256"]
        minus_hash = forwards["minus"]["consensus_output_sha256"]
        if base_hash in {plus_hash, minus_hash} or plus_hash == minus_hash:
            raise RuntimeError("v41a antithetic forwards were not distinguishable")

        phase.value = "snapshot"
        snapshots = v40a._rpc_all(
            trainer, "save_adapter_snapshot_v41a", (str(SNAPSHOT), master["sha256"]),
        )
        written = [value for value in snapshots if value.get("written")]
        if len(written) != 1 or not written[0].get("readback_verified"):
            raise RuntimeError("v41a snapshot writer/readback coverage changed")

        stop.set()
        monitor.join(timeout=10)
        if monitor.is_alive() or not failures.empty():
            raise RuntimeError("v41a GPU monitor failed") from (
                failures.get() if not failures.empty() else None
            )
        gpu = v40a.summarize_gpu(GPU_LOG, pid_map)
        cleanup = v40a.cleanup_v38a.strict_close_trainer_v38a(trainer)
        trainer = None
        import ray
        ray.shutdown()
        idle = v40a.cleanup_v38a.wait_for_gpu_idle()
        report = v40a.self_hashed({
            "schema": "canonical-lora-es-state-smoke-report-v41a",
            "status": "complete_train_only_four_gpu",
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_runtime_seconds": time.monotonic() - started,
            "preregistration": {
                "path": str(Path(args.preregistration).resolve()),
                "file_sha256": args.preregistration_sha256,
                "content_sha256": args.preregistration_content_sha256,
            },
            "actor_identities": actor_ids, "physical_gpu_pid_map": pid_map,
            "canonical_master_identity": master,
            "install_consensus_sha256": install_consensus,
            "certificate_consensus_sha256": certificate_consensus,
            "reference_consensus_sha256": reference_consensus,
            "forward_probe": {
                "baseline": baseline, "installed": installed_forward,
                "plus": forwards["plus"], "minus": forwards["minus"],
                "plus_changed_forward": True, "minus_changed_forward": True,
                "plus_minus_distinguishable": True,
                "restored_exact_forward_after_each_sign": True,
            },
            "perturbations": perturbations, "restorations": restorations,
            "snapshot_results": snapshots, "gpu_activity": gpu,
            "placement_group_cleanup": cleanup, "final_gpu_idle": idle,
            "preflight": preflight,
            "gpu_log": {"path": str(GPU_LOG), "file_sha256": v40a.file_sha256(GPU_LOG)},
            "dataset_or_evaluation_accessed": False,
            "sealed_holdout_opened": False, "synthetic_prompt_only": True,
            "quality_or_promotion_conclusion_authorized": False,
        })
        v40a.atomic_json(REPORT, report)
        print(json.dumps({
            "report": str(REPORT), "report_sha256": v40a.file_sha256(REPORT),
            "canonical_elements": master["elements"],
            "antithetic_forward_effect": True, "snapshot_readback": True,
        }, sort_keys=True))
        return 0
    except BaseException as error:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=10)
        v40a.atomic_json(RUN_DIR / "failure_v41a.json", v40a.self_hashed({
            "schema": "canonical-lora-es-state-smoke-failure-v41a",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(),
            "dataset_or_evaluation_accessed": False,
            "sealed_holdout_opened": False,
        }))
        raise
    finally:
        if trainer is not None:
            try:
                v40a.base.close_trainer(trainer)
            except Exception:
                pass
        try:
            import ray
            ray.shutdown()
        except Exception:
            pass
        if saved is not None:
            _restore_v40_globals(saved)


if __name__ == "__main__":
    raise SystemExit(main())
