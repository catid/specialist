#!/usr/bin/env python3
"""Fail-closed four-GPU runtime for the V38A equal-unit ES update."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pynvml

import run_sft_train_only_control_v36a as evidence
import train_eggroll_es_equal_unit_v38a as trainer_v38a
import train_eggroll_es_specialist as base
import train_eggroll_es_specialist_anchor_v13 as anchor_v13


ROOT = Path(__file__).resolve().parent
MODEL = (ROOT / "models/Qwen3.6-35B-A3B").resolve()
DATASET = (
    ROOT / "experiments/sft_controls/v37a_shadow_folds_v412/fold_3_train.jsonl"
).resolve()
SPLIT_MANIFEST = (
    ROOT / "experiments/sft_controls/v37a_shadow_folds_v412/manifest_v37a.json"
).resolve()
LAYER_PLAN = (ROOT / "experiments/layer_plans/middle_late_dense_v6.json").resolve()
EXPERIMENT = "v38a_equal_unit_fold3_pop32_antithetic_nonzero"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.launch_attempt.json").resolve()
REPORT = RUN_DIR / "equal_unit_update_report_v38a.json"
GPU_LOG = RUN_DIR / "gpu_activity_v38a.jsonl"
SNAPSHOT = RUN_DIR / "selected_runtime_snapshot_v38a.safetensors"
EXPECTED_GPU_IDS = (0, 1, 2, 3)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--preregistration", required=True)
    result.add_argument("--preregistration-sha256", required=True)
    result.add_argument("--preregistration-content-sha256", required=True)
    result.add_argument("--dry-run", action="store_true")
    return result


def load_preregistration(args) -> dict:
    path = Path(args.preregistration).resolve()
    if evidence.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("v38a preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    if (
        content != args.preregistration_content_sha256
        or content != evidence.canonical_sha256({
            key: item for key, item in value.items()
            if key != "content_sha256_before_self_field"
        })
        or value.get("schema") != "eggroll-es-equal-unit-preregistration-v38a"
        or value.get("status") != "preregistered_not_yet_run"
        or value.get("shadow_dev_external_eval_ood_or_holdout_opened") is not False
    ):
        raise RuntimeError("v38a preregistration content changed")
    bindings = value["implementation_bindings"]
    observed = {
        "runtime": evidence.file_sha256(Path(__file__).resolve()),
        "trainer": evidence.file_sha256(Path(trainer_v38a.__file__).resolve()),
        "worker": evidence.file_sha256(ROOT / "eggroll_es_worker_v38a.py"),
        "dataset": evidence.file_sha256(DATASET),
        "split_manifest": evidence.file_sha256(SPLIT_MANIFEST),
        "layer_plan": evidence.file_sha256(LAYER_PLAN),
        "model_config": evidence.file_sha256(MODEL / "config.json"),
        "model_index": evidence.file_sha256(
            MODEL / "model.safetensors.index.json"
        ),
        "trainer_v13": evidence.file_sha256(
            ROOT / "train_eggroll_es_specialist_anchor_v13.py"
        ),
        "trainer_v11c": evidence.file_sha256(
            ROOT / "train_eggroll_es_specialist_anchor_v11c.py"
        ),
        "worker_v11c": evidence.file_sha256(ROOT / "eggroll_es_worker_v11c.py"),
        "trainer_v4": evidence.file_sha256(
            ROOT / "train_eggroll_es_specialist_anchor_v4.py"
        ),
        "worker_v4": evidence.file_sha256(ROOT / "eggroll_es_worker_v4.py"),
        "worker_v3": evidence.file_sha256(ROOT / "eggroll_es_worker_v3.py"),
    }
    if observed != bindings:
        raise RuntimeError("v38a implementation binding changed")
    expected_paths = {
        "run_directory": str(RUN_DIR), "attempt": str(ATTEMPT),
        "report": str(REPORT), "gpu_log": str(GPU_LOG),
        "selected_runtime_snapshot": str(SNAPSHOT),
    }
    if value.get("artifacts") != expected_paths:
        raise RuntimeError("v38a artifact contract changed")
    return value


def gpu_preflight() -> dict:
    query = subprocess.run(
        [
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ], text=True, capture_output=True, check=True,
    ).stdout.strip()
    memory = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ], text=True, capture_output=True, check=True,
    ).stdout.strip().splitlines()
    used = {int(line.split(",")[0]): int(line.split(",")[1]) for line in memory}
    if query or set(used) != set(EXPECTED_GPU_IDS) or any(v > 2048 for v in used.values()):
        raise RuntimeError("v38a requires four exclusive idle GPUs")
    return {"compute_process_query_empty": True, "memory_used_mib": used}


def _physical_id(identity: dict) -> int:
    value = identity.get("cuda_visible_devices")
    if value not in {"0", "1", "2", "3"}:
        raise RuntimeError("v38a Ray actor physical GPU identity changed")
    return int(value)


def validate_worker_identities(items: list[dict]) -> dict[int, int]:
    if len(items) != 4:
        raise RuntimeError("v38a worker identity coverage changed")
    mapping = {}
    ranks = []
    for item in items:
        if (
            item.get("schema") != "eggroll-es-worker-runtime-identity-v38a"
            or not isinstance(item.get("pid"), int)
            or item["pid"] <= 0
            or item.get("cuda_current_device") != 0
        ):
            raise RuntimeError("v38a worker identity changed")
        physical = _physical_id(item)
        if physical in mapping:
            raise RuntimeError("v38a workers repeated a physical GPU")
        mapping[physical] = item["pid"]
        ranks.append(item["inter_engine_rank"])
    if set(mapping) != set(EXPECTED_GPU_IDS) or sorted(ranks) != list(EXPECTED_GPU_IDS):
        raise RuntimeError("v38a worker GPU/rank coverage changed")
    return mapping


def monitor_gpus(
    stop: threading.Event, expected_pids: dict[int, int], path: Path,
    failures: queue.Queue,
) -> None:
    try:
        pynvml.nvmlInit()
        handles = {
            gpu: pynvml.nvmlDeviceGetHandleByIndex(gpu) for gpu in EXPECTED_GPU_IDS
        }
        with path.open("x", encoding="utf-8") as output:
            while not stop.is_set():
                sampled = datetime.now(timezone.utc).isoformat()
                for gpu, handle in handles.items():
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    pids = sorted({int(item.pid) for item in processes})
                    foreign = [pid for pid in pids if pid != expected_pids[gpu]]
                    if foreign:
                        raise RuntimeError(f"foreign GPU processes on {gpu}: {foreign}")
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    row = {
                        "sampled_at_utc": sampled, "gpu": gpu,
                        "expected_pid": expected_pids[gpu], "compute_pids": pids,
                        "foreign_compute_pids": foreign,
                        "utilization_percent": int(utilization.gpu),
                        "memory_used_mib": int(memory.used // (1024 * 1024)),
                        "temperature_c": int(pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )),
                    }
                    output.write(json.dumps(row, sort_keys=True) + "\n")
                output.flush()
                stop.wait(0.5)
    except BaseException as error:
        failures.put(error)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def summarize_gpu_log(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    result = {}
    for gpu in EXPECTED_GPU_IDS:
        selected = [row for row in rows if row["gpu"] == gpu]
        resident = [row for row in selected if row["expected_pid"] in row["compute_pids"]]
        result[str(gpu)] = {
            "samples": len(selected),
            "resident_samples": len(resident),
            "positive_utilization_samples": sum(
                row["utilization_percent"] > 0 for row in resident
            ),
            "peak_utilization_percent": max(
                (row["utilization_percent"] for row in resident), default=0
            ),
            "peak_memory_used_mib": max(
                (row["memory_used_mib"] for row in resident), default=0
            ),
            "peak_temperature_c": max(
                (row["temperature_c"] for row in resident), default=0
            ),
            "mean_resident_utilization_percent": (
                sum(row["utilization_percent"] for row in resident) / len(resident)
                if resident else 0.0
            ),
        }
    if any(
        item["resident_samples"] <= 0 or item["positive_utilization_samples"] <= 0
        for item in result.values()
    ):
        raise RuntimeError("v38a did not observe attributed activity on all GPUs")
    return {"all_four_attributed_positive": True, "by_gpu": result}


def validate_sealed_update(update: dict) -> dict:
    application = update.get("application", {})
    manifest = application.get("manifest", {})
    final = application.get("final_identity", {})
    snapshot_reports = update.get("snapshot_reports", [])
    written = [item for item in snapshot_reports if item.get("written") is True]
    coefficients = update.get("coefficients", [])
    if (
        update.get("status") != "one_nonzero_update_sealed_train_only"
        or update.get("alpha") != 0.00015
        or update.get("sigma") != 0.0003
        or len(coefficients) != 32
        or not any(value != 0.0 for value in coefficients)
        or update.get("standardization", {}).get("zero_spread") is not False
        or application.get("target_alpha") != 0.00015
        or application.get("update_sequence") != 1
        or final.get("sha256") == manifest.get("expected_base_sha256")
        or final.get("unselected", {}).get("sha256")
        != manifest.get("unselected_origin_sha256")
        or len(application.get("commits", [])) != 4
        or any(item.get("committed") is not True for item in application["commits"])
        or len(application.get("post_commit_states", [])) != 4
        or len(written) != 1
        or written[0].get("rank") != 0
        or written[0].get("tensor_count") != 23
        or written[0].get("tensor_elements") != 142_999_552
    ):
        raise RuntimeError("v38a sealed update gate failed")
    return {
        "coefficient_spread_nonzero": True,
        "selected_final_identity_differs_from_base": True,
        "unselected_identity_equals_origin": True,
        "four_rank_commit_consensus": True,
        "selected_snapshot_inventory_exact": True,
    }


def wait_for_gpu_idle(timeout_seconds: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last = None
    while time.monotonic() < deadline:
        last = subprocess.run(
            [
                "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ], text=True, capture_output=True, check=True,
        ).stdout.strip()
        if not last:
            return {"all_four_compute_process_lists_empty": True}
        time.sleep(0.5)
    raise RuntimeError(f"v38a GPU processes survived cleanup: {last}")


def make_trainer(bundle):
    trainer_class = trainer_v38a.load_trainer(bundle)
    return trainer_class(
        model_name=str(MODEL), checkpoint=None, sigma=0.0003, alpha=0.0,
        population_size=32, reward_shaping="z-scores", num_iterations=1,
        max_tokens=1, batch_size=448, mini_batch_size=448,
        reward_function=base.specialist_reward,
        template_function=base.specialist_template,
        train_dataloader=[], eval_dataloader_dict={}, eval_freq=1,
        n_vllm_engines=4, n_gpu_per_vllm_engine=1, logging="none",
        global_seed=43, use_gpus="0,1,2,3", experiment_name=EXPERIMENT,
        wandb_project="specialist-eggroll-es", save_best_models=False,
        reward_function_timeout=10, output_directory=str(RUN_DIR.parent),
    )


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    prereg = load_preregistration(args)
    train_bundle = trainer_v38a.load_equal_unit_train_bundle(
        DATASET, prereg["implementation_bindings"]["dataset"],
        SPLIT_MANIFEST, prereg["implementation_bindings"]["split_manifest"],
    )
    layer_bundle = anchor_v13.load_frozen_layer_plan_v13(
        LAYER_PLAN,
        expected_file_sha256=prereg["implementation_bindings"]["layer_plan"],
        expected_plan_sha256=prereg["recipe"]["layer_plan_sha256"],
        expected_model_config_sha256=prereg["implementation_bindings"]["model_config"],
    )
    if train_bundle["content_sha256_before_self_field"] != prereg[
        "recipe"
    ]["train_bundle_content_sha256"]:
        raise RuntimeError("v38a frozen train bundle changed")
    if args.dry_run:
        print(json.dumps({
            "preregistration_content_sha256": prereg[
                "content_sha256_before_self_field"
            ],
            "train_bundle_content_sha256": train_bundle[
                "content_sha256_before_self_field"
            ],
            "layer_plan_sha256": layer_bundle["plan_sha256"],
        }, indent=2, sort_keys=True))
        return 0
    if ATTEMPT.exists() or RUN_DIR.exists():
        raise RuntimeError("v38a requires fresh attempt and run paths")
    preflight = gpu_preflight()
    attempt = evidence.self_hashed({
        "schema": "eggroll-es-equal-unit-attempt-v38a",
        "status": "launching", "phase": "before_trainer_creation",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_content_sha256": prereg[
            "content_sha256_before_self_field"
        ],
        "preflight": preflight,
        "shadow_dev_external_eval_ood_or_holdout_opened": False,
    })
    evidence.atomic_write_json(ATTEMPT, attempt)
    trainer = None
    monitor = None
    stop = threading.Event()
    failures: queue.Queue = queue.Queue()
    started = time.monotonic()
    try:
        base.set_seed(43)
        trainer = make_trainer(layer_bundle)
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        configured = trainer.configure_equal_unit_v38a(
            train_bundle, frozen_layer_plan=layer_bundle,
        )
        pid_map = validate_worker_identities(configured["worker_identities"])
        monitor = threading.Thread(
            target=monitor_gpus, args=(stop, pid_map, GPU_LOG, failures),
            daemon=True,
        )
        monitor.start()
        update_started = time.monotonic()
        update = trainer.estimate_apply_and_snapshot_v38a(SNAPSHOT)
        update_seconds = time.monotonic() - update_started
        stop.set()
        monitor.join(timeout=10)
        if monitor.is_alive() or not failures.empty():
            raise RuntimeError("v38a GPU activity monitor failed") from (
                failures.get() if not failures.empty() else None
            )
        gpu = summarize_gpu_log(GPU_LOG)
        if update["snapshot_file_sha256"] != evidence.file_sha256(SNAPSHOT):
            raise RuntimeError("v38a selected snapshot changed after save")
        update_gates = validate_sealed_update(update)
        base.close_trainer(trainer)
        trainer = None
        import ray
        ray.shutdown()
        final_idle = wait_for_gpu_idle()
        report = evidence.self_hashed({
            "schema": "eggroll-es-equal-unit-runtime-report-v38a",
            "status": "complete_one_nonzero_update_state_sealed",
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_runtime_seconds_through_state_seal": time.monotonic() - started,
            "resident_population_update_snapshot_seconds": update_seconds,
            "preregistration": {
                "path": str(Path(args.preregistration).resolve()),
                "file_sha256": args.preregistration_sha256,
                "content_sha256": args.preregistration_content_sha256,
            },
            "configuration": configured,
            "update": update,
            "update_gates": update_gates,
            "gpu_activity": gpu,
            "preflight": preflight,
            "final_idle": final_idle,
            "artifacts": {
                "selected_runtime_snapshot": str(SNAPSHOT),
                "selected_runtime_snapshot_sha256": evidence.file_sha256(SNAPSHOT),
                "gpu_log": str(GPU_LOG),
                "gpu_log_sha256": evidence.file_sha256(GPU_LOG),
            },
            "shadow_dev_external_eval_ood_or_holdout_opened": False,
            "promotion_decision_authorized": False,
        })
        evidence.atomic_write_json(REPORT, report)
        attempt.update({
            "status": "complete", "phase": "state_sealed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "report": str(REPORT), "report_sha256": evidence.file_sha256(REPORT),
        })
        evidence.atomic_write_json(ATTEMPT, evidence.self_hashed(attempt))
        print(REPORT)
        return 0
    except BaseException as error:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=10)
        attempt.update({
            "status": "failed", "phase": "runtime",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "failure": {
                "type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(),
            },
        })
        evidence.atomic_write_json(ATTEMPT, evidence.self_hashed(attempt))
        raise
    finally:
        if trainer is not None:
            try:
                base.close_trainer(trainer)
            except Exception:
                pass
        try:
            import ray
            ray.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
