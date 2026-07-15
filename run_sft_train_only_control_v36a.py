#!/usr/bin/env python3
"""Fail-closed four-GPU BF16 LoRA SFT runtime control.

This runner is deliberately train-only.  It binds one content-addressed JSONL
projection, launches the existing assistant-only SFT implementation under
four-process DDP, records GPU activity, and writes a self-hashed report.  It
does not accept an evaluation path and it makes no promotion decision.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from qa_quality import qa_pair_from_record


ROOT = Path(__file__).resolve().parent
BASE_MODEL = (ROOT / "models/Qwen3.6-35B-A3B").resolve()
SFT_SCRIPT = (ROOT / "sft_lora.py").resolve()
EXPECTED_GPU_IDS = (0, 1, 2, 3)
FORBIDDEN_DATA_NAME_PARTS = ("eval", "validation", "holdout", "heldout", "ood")
EXPECTED_ENCODING_AUDIT = {
    "prompt_mode": "es_exact",
    "eos_appended": False,
    "train_prompt_tokens": 26_379,
    "train_answer_tokens": 13_439,
    "train_rows": 531,
}
EXPECTED_TRAINABLE_INVENTORY = {
    "target_layers": [20, 21, 22, 23],
    "tensor_count": 70,
    "elements": 4_528_128,
    "identity_sha256": "46dbf26cf2a5c39cdb68d4858b5a6de89fa9ecb3c3007c459bba09b1cd6cec3c",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def self_hashed(value: dict) -> dict:
    value = dict(value)
    value.pop("content_sha256_before_self_field", None)
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def validate_train_dataset(path: Path, expected_sha256: str, expected_rows: int) -> dict:
    path = path.resolve()
    lowered_parts = {part.casefold() for part in path.parts}
    if any(
        forbidden in part
        for part in lowered_parts
        for forbidden in FORBIDDEN_DATA_NAME_PARTS
    ):
        raise ValueError("v36a train dataset path has a forbidden evaluation marker")
    observed_sha256 = file_sha256(path)
    if observed_sha256 != expected_sha256:
        raise ValueError("v36a train dataset SHA-256 changed")
    rows = 0
    fact_ids: set[str] = set()
    document_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at train line {line_number}") from exc
            pair = qa_pair_from_record(record)
            if pair is None:
                raise ValueError(f"unsupported QA record at train line {line_number}")
            question, answer = pair
            if not question.strip() or not answer.strip():
                raise ValueError(f"empty QA field at train line {line_number}")
            fact_id = record.get("fact_id")
            document_sha256 = record.get("document_sha256")
            if not isinstance(fact_id, str) or not fact_id:
                raise ValueError(f"missing fact identity at train line {line_number}")
            if fact_id in fact_ids:
                raise ValueError(f"duplicate fact identity at train line {line_number}")
            if not isinstance(document_sha256, str) or len(document_sha256) != 64:
                raise ValueError(f"missing document identity at train line {line_number}")
            fact_ids.add(fact_id)
            document_ids.add(document_sha256)
            rows += 1
    if rows != expected_rows:
        raise ValueError("v36a train dataset row count changed")
    return {
        "path": str(path),
        "sha256": observed_sha256,
        "rows": rows,
        "unique_fact_ids": len(fact_ids),
        "unique_documents": len(document_ids),
    }


def build_train_command(args: argparse.Namespace) -> list[str]:
    return [
        str((ROOT / ".venv/bin/torchrun").resolve()),
        "--standalone",
        "--nproc-per-node=4",
        str(SFT_SCRIPT),
        "--data",
        str(Path(args.dataset).resolve()),
        "--out",
        str(Path(args.output_dir).resolve()),
        "--epochs",
        str(args.epochs),
        "--rank",
        str(args.rank),
        "--lora-dropout",
        str(args.lora_dropout),
        "--grad-accum",
        str(args.grad_accum),
        "--max-length",
        str(args.max_length),
        "--save-steps",
        str(args.save_steps),
        "--data-sha256",
        args.dataset_sha256,
        "--data-rows",
        str(args.dataset_rows),
        "--expected-world-size",
        "4",
        "--per-device-batch-size",
        str(args.per_device_batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--seed",
        str(args.seed),
        "--prompt-mode",
        args.prompt_mode,
        "--loss-mode",
        args.loss_mode,
        "--target-layers",
        args.target_layers,
        "--expected-trainable-elements",
        str(args.expected_trainable_elements),
        "--expected-trainable-tensors",
        str(args.expected_trainable_tensors),
        "--attn-implementation",
        args.attn_implementation,
    ]


def validate_preregistration(args: argparse.Namespace, command: list[str]) -> dict:
    path = Path(args.preregistration).resolve()
    if file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("v36a preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content_hash = value.get("content_sha256_before_self_field")
    if content_hash != args.preregistration_content_sha256:
        raise RuntimeError("v36a preregistration content identity changed")
    if content_hash != canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }):
        raise RuntimeError("v36a preregistration self-hash does not recompute")
    if (
        value.get("schema")
        != "specialist-sft-train-only-control-preregistration-v36a"
        or value.get("status") != "preregistered_not_yet_run"
        or value.get("contains_validation_ood_or_holdout_content") is not False
        or value.get("recipe", {}).get("command") != command
        or value.get("dataset", {}).get("sha256") != args.dataset_sha256
        or value.get("dataset", {}).get("rows") != args.dataset_rows
    ):
        raise RuntimeError("v36a preregistration recipe contract changed")
    expected_paths = {
        "output_dir": str(Path(args.output_dir).resolve()),
        "stdout_log": str(Path(args.stdout_log).resolve()),
        "gpu_log": str(Path(args.gpu_log).resolve()),
        "report": str(Path(args.report).resolve()),
        "attempt_report": str(Path(args.attempt_report).resolve()),
    }
    if value.get("artifacts") != expected_paths:
        raise RuntimeError("v36a preregistration artifact paths changed")
    implementation = value.get("implementation", {})
    model = value.get("model", {})
    comparison = value.get("comparison_binding", {})
    observed = {
        "runner": file_sha256(Path(__file__).resolve()),
        "sft": file_sha256(SFT_SCRIPT),
        "model_config": file_sha256(BASE_MODEL / "config.json"),
        "model_index": file_sha256(BASE_MODEL / "model.safetensors.index.json"),
        "layer_plan": file_sha256(Path(comparison["eggroll_es_layer_plan"])),
    }
    expected = {
        "runner": implementation.get("runner_sha256"),
        "sft": implementation.get("sft_sha256"),
        "model_config": model.get("config_sha256"),
        "model_index": model.get("index_sha256"),
        "layer_plan": comparison.get("eggroll_es_layer_plan_sha256"),
    }
    if observed != expected:
        raise RuntimeError("v36a preregistration implementation binding changed")
    return {
        "path": str(path),
        "file_sha256": args.preregistration_sha256,
        "content_sha256": content_hash,
        "implementation_bindings": observed,
    }


def _gpu_uuid_to_index() -> dict[str, int]:
    completed = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ], check=True, capture_output=True, text=True,
    )
    result = {}
    for line in completed.stdout.splitlines():
        index, uuid = [field.strip() for field in line.split(",", 1)]
        result[uuid] = int(index)
    if tuple(sorted(result.values())) != EXPECTED_GPU_IDS:
        raise RuntimeError("v36a physical GPU UUID mapping changed")
    return result


def compute_pids_by_gpu(uuid_to_index: dict[str, int]) -> dict[int, set[int]]:
    result = {gpu: set() for gpu in EXPECTED_GPU_IDS}
    completed = subprocess.run(
        [
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ], check=True, capture_output=True, text=True,
    )
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        uuid, pid = [field.strip() for field in line.split(",", 1)]
        if uuid not in uuid_to_index:
            raise RuntimeError("v36a observed compute on an unexpected GPU UUID")
        result[uuid_to_index[uuid]].add(int(pid))
    return result


def descendant_pids(root_pid: int) -> set[int]:
    parents = {}
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat_path.read_text().split()
            parents[int(fields[0])] = int(fields[3])
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return descendants


def assert_gpu_exclusive(max_prelaunch_memory_mib: int = 2048) -> dict:
    uuid_to_index = _gpu_uuid_to_index()
    compute = compute_pids_by_gpu(uuid_to_index)
    if any(compute.values()):
        raise RuntimeError("v36a found a foreign compute process before launch")
    completed = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ], check=True, capture_output=True, text=True,
    )
    memory = {}
    for line in completed.stdout.splitlines():
        gpu, used = [field.strip() for field in line.split(",", 1)]
        memory[int(gpu)] = int(used)
    if tuple(sorted(memory)) != EXPECTED_GPU_IDS or any(
        used > max_prelaunch_memory_mib for used in memory.values()
    ):
        raise RuntimeError("v36a GPUs are not idle enough for an exclusive launch")
    return {
        "compute_pids_by_gpu": {str(gpu): [] for gpu in EXPECTED_GPU_IDS},
        "memory_used_mib_by_gpu": {str(gpu): memory[gpu] for gpu in EXPECTED_GPU_IDS},
        "maximum_allowed_memory_mib": max_prelaunch_memory_mib,
    }


def parse_gpu_csv(line: str) -> dict:
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 6:
        raise ValueError("unexpected nvidia-smi CSV shape")
    return {
        "timestamp": fields[0],
        "gpu": int(fields[1]),
        "utilization_percent": int(fields[2]),
        "memory_used_mib": int(fields[3]),
        "power_watts": float(fields[4]),
        "temperature_c": int(fields[5]),
    }


def poll_gpus(
    stop: threading.Event,
    output_path: Path,
    errors: queue.Queue,
    child_state: dict,
) -> None:
    query = (
        "timestamp,index,utilization.gpu,memory.used,power.draw,temperature.gpu"
    )
    try:
        uuid_to_index = _gpu_uuid_to_index()
        with output_path.open("w", encoding="utf-8") as output:
            while not stop.is_set():
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--query-gpu={query}",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                sampled_at = datetime.now(timezone.utc).isoformat()
                root_pid = child_state.get("root_pid")
                descendants = descendant_pids(root_pid) if root_pid else set()
                compute = compute_pids_by_gpu(uuid_to_index)
                rows = []
                for line in completed.stdout.splitlines():
                    if not line.strip():
                        continue
                    sample = parse_gpu_csv(line)
                    sample["sampled_at_utc"] = sampled_at
                    pids = compute[sample["gpu"]]
                    sample["compute_pids"] = sorted(pids)
                    sample["attributed_compute_pids"] = sorted(pids & descendants)
                    sample["foreign_compute_pids"] = sorted(pids - descendants)
                    if sample["foreign_compute_pids"]:
                        raise RuntimeError(
                            "v36a observed a foreign GPU compute process during training"
                        )
                    rows.append(sample)
                    output.write(json.dumps(sample, sort_keys=True) + "\n")
                output.flush()
                if tuple(sorted(row["gpu"] for row in rows)) != EXPECTED_GPU_IDS:
                    raise RuntimeError("v36a GPU monitor did not observe exactly four GPUs")
                stop.wait(0.5)
    except BaseException as exc:  # preserve monitor failure for the coordinator
        errors.put(exc)


def read_gpu_samples(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_gpu_samples(samples: list[dict]) -> dict:
    by_gpu = {gpu: [] for gpu in EXPECTED_GPU_IDS}
    for sample in samples:
        gpu = sample.get("gpu")
        if gpu not in by_gpu:
            raise ValueError("v36a observed an unexpected physical GPU")
        by_gpu[gpu].append(sample)
    if any(not values for values in by_gpu.values()):
        raise ValueError("v36a GPU activity evidence is incomplete")
    summaries = {}
    for gpu, values in by_gpu.items():
        resident = [row for row in values if row["memory_used_mib"] >= 60_000]
        summaries[str(gpu)] = {
            "samples": len(values),
            "resident_samples": len(resident),
            "positive_utilization_samples": sum(
                row["utilization_percent"] > 0 for row in values
            ),
            "attributed_activity_samples": sum(
                row["utilization_percent"] > 0
                and bool(row.get("attributed_compute_pids"))
                for row in values
            ),
            "high_utilization_samples": sum(
                row["utilization_percent"] >= 80 for row in values
            ),
            "peak_utilization_percent": max(row["utilization_percent"] for row in values),
            "peak_memory_used_mib": max(row["memory_used_mib"] for row in values),
            "peak_power_watts": max(row["power_watts"] for row in values),
            "peak_temperature_c": max(row["temperature_c"] for row in values),
            "resident_mean_utilization_percent": (
                sum(row["utilization_percent"] for row in resident) / len(resident)
                if resident
                else None
            ),
        }
    all_four_positive = all(
        summaries[str(gpu)]["attributed_activity_samples"] > 0
        for gpu in EXPECTED_GPU_IDS
    )
    all_four_resident = all(
        summaries[str(gpu)]["resident_samples"] > 0 for gpu in EXPECTED_GPU_IDS
    )
    return {
        "physical_gpu_ids": list(EXPECTED_GPU_IDS),
        "by_gpu": summaries,
        "all_four_positive_activity": all_four_positive,
        "all_four_model_resident": all_four_resident,
        "activity_attributed_to_torchrun_tree": all_four_positive,
    }


def extract_train_metrics(log_text: str) -> dict:
    for line in reversed(log_text.splitlines()):
        if "'train_runtime'" not in line:
            continue
        start = line.find("{")
        end = line.rfind("}")
        if start < 0 or end <= start:
            continue
        value = ast.literal_eval(line[start : end + 1])
        if not isinstance(value, dict) or "train_runtime" not in value:
            continue
        result = {}
        for key, raw in value.items():
            if key in {
                "train_runtime",
                "train_samples_per_second",
                "train_steps_per_second",
                "train_loss",
                "epoch",
            }:
                result[key] = float(raw)
        return result
    raise ValueError("v36a Trainer metrics were not found in the training log")


def extract_json_event(log_text: str, key: str) -> dict:
    marker = json.dumps(key) + ":"
    observed = []
    for line in log_text.splitlines():
        marker_index = line.find(marker)
        if marker_index < 0:
            continue
        start = line.rfind("{", 0, marker_index)
        end = line.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            value = json.loads(line[start:end + 1])
        except json.JSONDecodeError:
            continue
        if key in value:
            observed.append(value[key])
    if not observed or any(value != observed[0] for value in observed[1:]):
        raise ValueError(f"v36a child event {key} is missing or inconsistent")
    return {"value": observed[0], "emission_count": len(observed)}


def validate_output_artifacts(output_dir: Path, expected_steps: int) -> dict:
    final = output_dir / "final"
    config_path = final / "adapter_config.json"
    adapter_path = final / "adapter_model.safetensors"
    if not config_path.is_file() or not adapter_path.is_file():
        raise RuntimeError("v36a final LoRA adapter is incomplete")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_modules = {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
        "down_proj", "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z",
        "out_proj",
    }
    expected_parameters = {
        f"model.layers.{layer}.mlp.gate.weight" for layer in (20, 21, 22, 23)
    }
    if (
        config.get("r") != 32
        or config.get("lora_alpha") != 64
        or config.get("lora_dropout") != 0.0
        or config.get("layers_to_transform") != [20, 21, 22, 23]
        or set(config.get("target_modules", [])) != expected_modules
        or set(config.get("target_parameters", [])) != expected_parameters
    ):
        raise RuntimeError("v36a saved adapter configuration changed")
    checkpoints = sorted(
        int(path.name.removeprefix("checkpoint-"))
        for path in output_dir.glob("checkpoint-*")
        if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
    )
    expected_checkpoints = [19, 38, expected_steps]
    state_path = output_dir / f"checkpoint-{expected_steps}" / "trainer_state.json"
    if checkpoints != expected_checkpoints or not state_path.is_file():
        raise RuntimeError("v36a checkpoint schedule changed")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("global_step") != expected_steps or float(state.get("epoch")) != 3.0:
        raise RuntimeError("v36a final optimizer step count changed")
    return {
        "adapter_config": config,
        "checkpoint_steps": checkpoints,
        "final_global_step": state["global_step"],
        "final_epoch": state["epoch"],
    }


def terminate_process_group(process: subprocess.Popen | None) -> bool:
    if process is None or process.poll() is not None:
        return False
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    except ProcessLookupError:
        pass
    return True


def hash_output_files(output_dir: Path) -> dict[str, str]:
    result = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            result[str(path.relative_to(output_dir))] = file_sha256(path)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--dataset", required=True)
    result.add_argument("--dataset-sha256", required=True)
    result.add_argument("--dataset-rows", required=True, type=int)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--stdout-log", required=True)
    result.add_argument("--gpu-log", required=True)
    result.add_argument("--report", required=True)
    result.add_argument("--attempt-report", required=True)
    result.add_argument("--preregistration", required=True)
    result.add_argument("--preregistration-sha256", required=True)
    result.add_argument("--preregistration-content-sha256", required=True)
    result.add_argument("--epochs", type=float, default=3.0)
    result.add_argument("--rank", type=int, default=32)
    result.add_argument("--lora-dropout", type=float, default=0.0)
    result.add_argument("--grad-accum", type=int, default=1)
    result.add_argument("--per-device-batch-size", type=int, default=7)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--prompt-mode", choices=("es_exact",), default="es_exact")
    result.add_argument(
        "--loss-mode", choices=("example_mean",), default="example_mean"
    )
    result.add_argument("--target-layers", default="20,21,22,23")
    result.add_argument("--expected-trainable-elements", type=int, default=4_528_128)
    result.add_argument("--expected-trainable-tensors", type=int, default=70)
    result.add_argument("--max-length", type=int, default=1024)
    result.add_argument("--save-steps", type=int, default=100)
    result.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="sdpa",
    )
    result.add_argument("--dry-run", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if tuple(range(int(os.environ.get("WORLD_SIZE", "4")))) != EXPECTED_GPU_IDS:
        raise RuntimeError("v36a runtime contract requires world size four")
    dataset = validate_train_dataset(
        Path(args.dataset), args.dataset_sha256, args.dataset_rows
    )
    command = build_train_command(args)
    preregistration = validate_preregistration(args, command)
    if args.dry_run:
        print(json.dumps({
            "dataset": dataset,
            "command": command,
            "preregistration": preregistration,
        }, indent=2))
        return 0

    output_dir = Path(args.output_dir).resolve()
    stdout_log = Path(args.stdout_log).resolve()
    gpu_log = Path(args.gpu_log).resolve()
    report_path = Path(args.report).resolve()
    attempt_path = Path(args.attempt_report).resolve()
    for path in (stdout_log, gpu_log, report_path, attempt_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    if attempt_path.exists() or report_path.exists():
        raise RuntimeError("v36a attempt or final report path already exists")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("v36a output directory is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    prelaunch = assert_gpu_exclusive()
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    stop = threading.Event()
    monitor_errors: queue.Queue = queue.Queue()
    child_state = {"root_pid": None}
    monitor = None
    environment = dict(os.environ)
    environment.update({
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    process = None
    returncode = None
    phase = "prepared"
    attempt = {
        "schema": "specialist-sft-train-only-attempt-v36a",
        "status": "launch_prepared",
        "started_at_utc": started_at,
        "phase": phase,
        "dataset": dataset,
        "preregistration": preregistration,
        "prelaunch_gpu_exclusivity": prelaunch,
        "command": command,
        "output_dir": str(output_dir),
    }
    atomic_write_json(attempt_path, self_hashed(attempt))
    try:
        with stdout_log.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            child_state["root_pid"] = process.pid
            phase = "child_running"
            attempt.update({
                "status": "child_running", "phase": phase,
                "torchrun_pid": process.pid,
            })
            atomic_write_json(attempt_path, self_hashed(attempt))
            monitor = threading.Thread(
                target=poll_gpus,
                args=(stop, gpu_log, monitor_errors, child_state),
                name="v36a-gpu-monitor",
                daemon=True,
            )
            monitor.start()
            assert process.stdout is not None
            for line in process.stdout:
                output.write(line)
                output.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            returncode = process.wait()
            phase = "child_complete"
        if returncode != 0:
            raise RuntimeError(f"v36a SFT child failed with exit code {returncode}")
        stop.set()
        if monitor is not None:
            monitor.join(timeout=10)
            if monitor.is_alive():
                raise RuntimeError("v36a GPU monitor failed to stop")
        if not monitor_errors.empty():
            raise RuntimeError("v36a GPU monitor failed") from monitor_errors.get()

        log_text = stdout_log.read_text(encoding="utf-8")
        metrics = extract_train_metrics(log_text)
        encoding_event = extract_json_event(log_text, "encoding_audit")
        inventory_event = extract_json_event(log_text, "trainable_inventory")
        if encoding_event["value"] != EXPECTED_ENCODING_AUDIT:
            raise RuntimeError("v36a observed encoding audit changed")
        if inventory_event["value"] != EXPECTED_TRAINABLE_INVENTORY:
            raise RuntimeError("v36a observed trainable inventory changed")
        gpu_summary = summarize_gpu_samples(read_gpu_samples(gpu_log))
        if not gpu_summary["all_four_positive_activity"]:
            raise RuntimeError("v36a did not record attributed activity on every GPU")
        if not gpu_summary["all_four_model_resident"]:
            raise RuntimeError("v36a did not record model residency on every GPU")
        output_validation = validate_output_artifacts(output_dir, 57)
        output_files = hash_output_files(output_dir)
        post_preregistration = validate_preregistration(args, command)
        if post_preregistration != preregistration:
            raise RuntimeError("v36a preregistration changed during the run")
        phase = "evidence_validated"

        report = {
        "schema": "specialist-sft-train-only-runtime-control-v36a",
        "status": "complete_train_only_no_promotion_decision",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_runtime_seconds": time.monotonic() - start,
        "selection_surface": "train_only",
        "validation_ood_or_holdout_opened": False,
        "dataset": dataset,
        "preregistration": preregistration,
        "model": {
            "path": str(BASE_MODEL),
            "config_sha256": file_sha256(BASE_MODEL / "config.json"),
            "index_sha256": file_sha256(BASE_MODEL / "model.safetensors.index.json"),
        },
        "implementation": {
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "sft_script": str(SFT_SCRIPT),
            "sft_script_sha256": file_sha256(SFT_SCRIPT),
        },
        "recipe": {
            "world_size": 4,
            "physical_gpu_ids": list(EXPECTED_GPU_IDS),
            "epochs": args.epochs,
            "rank": args.rank,
            "lora_alpha": 2 * args.rank,
            "lora_dropout": args.lora_dropout,
            "grad_accum": args.grad_accum,
            "per_device_batch_size": args.per_device_batch_size,
            "effective_global_batch_size": (
                4 * args.per_device_batch_size * args.grad_accum
            ),
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "max_length": args.max_length,
            "bf16": True,
            "gradient_checkpointing": True,
            "attn_implementation": args.attn_implementation,
            "prompt_mode": args.prompt_mode,
            "loss_mode": args.loss_mode,
            "target_layers": args.target_layers,
            "expected_trainable_elements": args.expected_trainable_elements,
            "expected_trainable_tensors": args.expected_trainable_tensors,
            "command": command,
        },
        "trainer_metrics": metrics,
        "observed_encoding_audit": encoding_event,
        "observed_trainable_inventory": inventory_event,
        "gpu_activity": gpu_summary,
        "prelaunch_gpu_exclusivity": prelaunch,
        "output_validation": output_validation,
        "artifacts": {
            "output_dir": str(output_dir),
            "output_file_sha256": output_files,
            "stdout_log": str(stdout_log),
            "stdout_log_sha256": file_sha256(stdout_log),
            "gpu_log": str(gpu_log),
            "gpu_log_sha256": file_sha256(gpu_log),
        },
        "interpretation": (
            "runtime and train-loss control only; no quality, validation, OOD, "
            "holdout, or promotion conclusion is authorized"
        ),
        }
        atomic_write_json(report_path, self_hashed(report))
        attempt.update({
            "status": "complete", "phase": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": returncode,
            "final_report": str(report_path),
            "final_report_sha256": file_sha256(report_path),
        })
        atomic_write_json(attempt_path, self_hashed(attempt))
        print(f"wrote {report_path}")
        return 0
    except BaseException as exc:
        terminated = terminate_process_group(process)
        stop.set()
        if monitor is not None:
            monitor.join(timeout=10)
        failure = dict(attempt)
        failure.update({
            "status": "failed",
            "phase": phase,
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": returncode if returncode is not None else (
                process.poll() if process is not None else None
            ),
            "child_terminated_by_runner": terminated,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "retry_policy": "new content-addressed attempt directory required",
        })
        atomic_write_json(attempt_path, self_hashed(failure))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
