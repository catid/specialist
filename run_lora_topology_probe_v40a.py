#!/usr/bin/env python3
"""Preregistered, train-only four-GPU V37 LoRA topology probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import subprocess
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pynvml

import run_eggroll_es_equal_unit_v38a as cleanup_v38a
import run_sft_train_only_control_v36a as hashing
import train_eggroll_es_specialist as base


ROOT = Path(__file__).resolve().parent
MODEL = (ROOT / "models/Qwen3.6-35B-A3B").resolve()
ADAPTER = (
    ROOT / "experiments/sft_controls/v37a_equal_unit_fold3_v412/"
    "middle_late_r32_seed17/final"
).resolve()
ADAPTER_FILE = (ADAPTER / "adapter_model.safetensors").resolve()
STAGED_ADAPTER = (
    ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
    "v37a_qwen35_vllm_namespace_v40a"
).resolve()
STAGED_ADAPTER_FILE = (STAGED_ADAPTER / "adapter_model.safetensors").resolve()
STAGE_MANIFEST = (STAGED_ADAPTER / "stage_manifest_v40a.json").resolve()
TUNED_FOLDER = (
    ROOT / "experiments/vllm_moe_tuning/"
    "v025_rtx_pro_6000_bf16_tp1_exhaustive_v27c"
).resolve()
TUNED_FILE = (
    TUNED_FOLDER /
    "E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition.json"
).resolve()
EXPERIMENT = "v40a_v37_lora_topology_probe"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
REPORT = (RUN_DIR / "lora_topology_report_v40a.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v40a.jsonl").resolve()
GPU_IDS = (0, 1, 2, 3)
MODEL_SHARDS_CONTENT_SHA256 = (
    "af8ea3a900c04e97d2d8e3146b8e23be5ee3e6548dea20440020b2f43ee6656e"
)
WORKER_EXTENSION = (
    "eggroll_es_worker_lora_topology_v40a.LoRATopologyWorkerExtensionV40A"
)
SYNTHETIC_PROMPT = (
    "<|im_start|>user\nState one general safety principle for learning a "
    "new physical skill.<|im_end|>\n<|im_start|>assistant\n"
)


def file_sha256(path: Path) -> str:
    return hashing.file_sha256(Path(path))


def canonical_sha256(value) -> str:
    return hashing.canonical_sha256(value)


def self_hashed(value: dict) -> dict:
    result = dict(value)
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.link(temporary, path)
    temporary.unlink()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--preregistration", required=True)
    result.add_argument("--preregistration-sha256", required=True)
    result.add_argument("--preregistration-content-sha256", required=True)
    result.add_argument("--dry-run", action="store_true")
    return result


def load_preregistration(args) -> dict:
    path = Path(args.preregistration).resolve()
    if file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("v40a preregistration file identity changed")
    value = json.loads(path.read_text())
    content = value.get("content_sha256_before_self_field")
    if (
        content != args.preregistration_content_sha256
        or content != canonical_sha256({
            key: item for key, item in value.items()
            if key != "content_sha256_before_self_field"
        })
        or value.get("schema") != "lora-topology-preregistration-v40a"
        or value.get("status") != "preregistered_before_gpu_launch"
        or value.get("dataset_or_evaluation_access_authorized") is not False
    ):
        raise RuntimeError("v40a preregistration content changed")
    paths = {
        "runtime": Path(__file__).resolve(),
        "worker": ROOT / "eggroll_es_worker_lora_topology_v40a.py",
        "adapter_weights": ADAPTER_FILE,
        "adapter_config": ADAPTER / "adapter_config.json",
        "staged_adapter_weights": STAGED_ADAPTER_FILE,
        "staged_adapter_config": STAGED_ADAPTER / "adapter_config.json",
        "stage_manifest": STAGE_MANIFEST,
        "stage_runtime": ROOT / "stage_v37_adapter_for_vllm_v40a.py",
        "model_config": MODEL / "config.json",
        "model_index": MODEL / "model.safetensors.index.json",
        "tuned_table": TUNED_FILE,
        "base_runtime": ROOT / "train_eggroll_es_specialist.py",
        "cleanup_runtime": ROOT / "run_eggroll_es_equal_unit_v38a.py",
    }
    observed = {key: file_sha256(path) for key, path in paths.items()}
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())
    shard_names = sorted(set(index["weight_map"].values()))
    if len(shard_names) != 26 or not all((MODEL / name).is_file() for name in shard_names):
        raise RuntimeError("v40a base model shard inventory changed")
    observed["model_shards_content_sha256"] = MODEL_SHARDS_CONTENT_SHA256
    if observed != value.get("implementation_bindings"):
        raise RuntimeError("v40a implementation or artifact binding changed")
    forbidden = ("heldout", "holdout", "shadow", "ood", "train.jsonl")
    for bound in paths.values():
        if any(item in str(Path(bound).resolve()).lower() for item in forbidden):
            raise RuntimeError("v40a bound a forbidden data/evaluation path")
    return value


def gpu_preflight() -> dict:
    processes = subprocess.run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits",
    ], text=True, capture_output=True, check=True).stdout.strip()
    rows = subprocess.run([
        "nvidia-smi", "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ], text=True, capture_output=True, check=True).stdout.strip().splitlines()
    memory = {int(row.split(",")[0]): int(row.split(",")[1]) for row in rows}
    if processes or set(memory) != set(GPU_IDS) or any(v > 2048 for v in memory.values()):
        raise RuntimeError("v40a requires four exclusive idle physical GPUs")
    return {"compute_process_query_empty": True, "memory_used_mib": memory}


def normalize_gpu_id(value) -> int:
    number = float(value)
    if not number.is_integer() or int(number) not in GPU_IDS:
        raise RuntimeError("v40a invalid Ray GPU id")
    return int(number)


def make_trainer(prereg: dict):
    parent = base.load_trainer()
    expected_tuned_content = prereg["runtime"]["tuned_table_content_sha256"]

    class TopologyTrainerV40A(parent):
        def launch_engines(self, num_engines=4, n_gpu_per_vllm_engine=1,
                           model_name="unused", precision="bfloat16"):
            if int(num_engines) != 4 or int(n_gpu_per_vllm_engine) != 1:
                raise RuntimeError("v40a requires four TP1 engines")
            import ray
            from ray.util.placement_group import placement_group
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
            from es_at_scale.trainer.es_trainer import ESNcclLLM

            class TopologyLLMV40A(ESNcclLLM):
                def runtime_identity_v40a(self):
                    import torch
                    import vllm.envs as vllm_envs
                    import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe
                    raw = ray.get_gpu_ids()
                    if len(raw) != 1:
                        raise RuntimeError("v40a actor does not own exactly one GPU")
                    physical = normalize_gpu_id(raw[0])
                    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                    folder = os.environ.get("VLLM_TUNED_CONFIG_FOLDER")
                    if (
                        visible != str(physical) or torch.cuda.device_count() != 1
                        or torch.cuda.current_device() != 0
                        or folder != str(TUNED_FOLDER)
                        or vllm_envs.VLLM_TUNED_CONFIG_FOLDER != str(TUNED_FOLDER)
                    ):
                        raise RuntimeError("v40a actor device/tuning binding changed")
                    fused_moe.get_moe_configs.cache_clear()
                    config = fused_moe.get_moe_configs(256, 512, None)
                    if canonical_sha256(config) != expected_tuned_content:
                        raise RuntimeError("v40a exact V27C tuned table was not loaded")
                    return {
                        "schema": "lora-topology-actor-identity-v40a",
                        "pid": os.getpid(), "physical_gpu_id": physical,
                        "cuda_visible_devices": visible, "cuda_current_device": 0,
                        "tuned_folder": folder,
                        "tuned_table_content_sha256": canonical_sha256(config),
                    }

            pgs = [placement_group([{"GPU": 1, "CPU": 0}], strategy="PACK")
                   for _ in GPU_IDS]
            ray.get([pg.ready() for pg in pgs])
            strategies = [PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_capture_child_tasks=True,
                placement_group_bundle_index=0,
            ) for pg in pgs]
            kwargs = {
                "model": str(MODEL), "tensor_parallel_size": 1,
                "worker_extension_cls": WORKER_EXTENSION,
                "dtype": precision, "enable_prefix_caching": False,
                "enforce_eager": True, "gpu_memory_utilization": 0.82,
                "max_model_len": 2048,
                "limit_mm_per_prompt": {"image": 0, "video": 0},
                "mm_processor_cache_gb": 0, "skip_mm_profiling": True,
                "moe_backend": "triton", "enable_lora": True,
                "max_lora_rank": 32, "max_loras": 1, "max_cpu_loras": 1,
            }
            engines = [ray.remote(
                num_cpus=0, num_gpus=1, scheduling_strategy=strategy,
            )(TopologyLLMV40A).options(runtime_env={
                "env_vars": {"VLLM_TUNED_CONFIG_FOLDER": str(TUNED_FOLDER)},
            }).remote(**kwargs) for strategy in strategies]
            return engines, pgs

    return TopologyTrainerV40A(
        model_name=str(MODEL), checkpoint=None, sigma=0.0, alpha=0.0,
        population_size=4, reward_shaping="z-scores", num_iterations=0,
        max_tokens=4, batch_size=1, mini_batch_size=1,
        reward_function=base.specialist_reward,
        template_function=lambda value: value,
        train_dataloader=[], eval_dataloader_dict={}, eval_freq=1,
        n_vllm_engines=4, n_gpu_per_vllm_engine=1, logging="none",
        global_seed=20_260_715, use_gpus="0,1,2,3",
        experiment_name=EXPERIMENT, wandb_project="none",
        save_best_models=False, reward_function_timeout=10,
        output_directory=str(RUN_DIR.parent),
    )


def _rpc_all(trainer, method: str, args=()) -> list[dict]:
    values = trainer._resolve([
        engine.collective_rpc.remote(method, args=args) for engine in trainer.engines
    ])
    if len(values) != 4 or any(len(value) != 1 for value in values):
        raise RuntimeError(f"v40a incomplete collective RPC: {method}")
    return [value[0] for value in values]


def validate_identities(actor: list[dict], worker: list[dict]) -> dict[int, int]:
    if len(actor) != 4 or len(worker) != 4:
        raise RuntimeError("v40a identity coverage changed")
    worker_by_pid = {item["pid"]: item for item in worker}
    mapping = {}
    for item in actor:
        gpu, pid = item["physical_gpu_id"], item["pid"]
        if gpu in mapping or pid not in worker_by_pid:
            raise RuntimeError("v40a actor/worker identity mismatch")
        if worker_by_pid[pid]["cuda_visible_devices"] != str(gpu):
            raise RuntimeError("v40a actor/worker physical GPU mismatch")
        mapping[gpu] = pid
    if set(mapping) != set(GPU_IDS) or len(set(mapping.values())) != 4:
        raise RuntimeError("v40a physical GPU/PID mapping incomplete")
    return mapping


class Phase:
    value = "setup"


def monitor_gpus(stop, phase, expected_pids, path, failures):
    try:
        pynvml.nvmlInit()
        handles = {gpu: pynvml.nvmlDeviceGetHandleByIndex(gpu) for gpu in GPU_IDS}
        with Path(path).open("x") as output:
            while not stop.is_set():
                sampled = datetime.now(timezone.utc).isoformat()
                for gpu, handle in handles.items():
                    pids = sorted({int(item.pid) for item in
                                   pynvml.nvmlDeviceGetComputeRunningProcesses(handle)})
                    foreign = [pid for pid in pids if pid != expected_pids[gpu]]
                    if foreign:
                        raise RuntimeError(f"v40a foreign GPU process: {gpu} {foreign}")
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    output.write(json.dumps({
                        "sampled_at_utc": sampled, "phase": phase.value, "gpu": gpu,
                        "expected_pid": expected_pids[gpu], "compute_pids": pids,
                        "foreign_compute_pids": foreign,
                        "utilization_percent": int(util.gpu),
                        "memory_used_mib": int(memory.used // 2**20),
                    }, sort_keys=True) + "\n")
                output.flush()
                stop.wait(0.5)
    except BaseException as error:
        failures.put(error)
    finally:
        try: pynvml.nvmlShutdown()
        except Exception: pass


def summarize_gpu(path: Path, expected_pids: dict[int, int]) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    result = {}
    for gpu in GPU_IDS:
        selected = [row for row in rows if row["gpu"] == gpu]
        resident = [row for row in selected if expected_pids[gpu] in row["compute_pids"]]
        if not resident or not any(row["utilization_percent"] > 0 for row in resident):
            raise RuntimeError(f"v40a GPU {gpu} lacked attributed positive activity")
        result[str(gpu)] = {
            "expected_pid": expected_pids[gpu], "samples": len(selected),
            "resident_samples": len(resident),
            "positive_samples": sum(row["utilization_percent"] > 0 for row in resident),
            "mean_resident_utilization_percent": math.fsum(
                row["utilization_percent"] for row in resident) / len(resident),
            "peak_utilization_percent": max(row["utilization_percent"] for row in resident),
            "peak_memory_used_mib": max(row["memory_used_mib"] for row in resident),
        }
    return {"all_four_attributed_positive": True, "by_gpu": result}


def _lora_request():
    from vllm.lora.request import LoRARequest
    return LoRARequest(
        "v37a_final_vllm_namespace_v40a", 1, str(STAGED_ADAPTER),
        base_model_name=str(MODEL),
    )


def _logprob_value(value):
    if value is None:
        return None
    if hasattr(value, "logprob"):
        return float(value.logprob)
    return float(value)


def output_record(output) -> dict:
    prompt_rows = []
    for row in output.prompt_logprobs or []:
        if row is None:
            prompt_rows.append(None)
        else:
            prompt_rows.append(sorted(
                (int(token), _logprob_value(value)) for token, value in row.items()
            ))
    samples = []
    for sample in output.outputs:
        logprobs = []
        for row in sample.logprobs or []:
            logprobs.append(sorted(
                (int(token), _logprob_value(value)) for token, value in row.items()
            ))
        samples.append({
            "token_ids": list(sample.token_ids),
            "cumulative_logprob": float(sample.cumulative_logprob or 0.0),
            "logprobs": logprobs,
        })
    return {
        "prompt_token_ids": list(output.prompt_token_ids),
        "prompt_logprobs": prompt_rows, "outputs": samples,
    }


def probe_forward(trainer) -> dict:
    from vllm import SamplingParams
    prompt_ids = trainer.tokenizer.encode(SYNTHETIC_PROMPT, add_special_tokens=False)
    params = SamplingParams(
        n=1, seed=20_260_715, temperature=0.0, top_p=1.0,
        max_tokens=4, prompt_logprobs=5, logprobs=5, detokenize=False,
    )
    batches = trainer._resolve([
        engine.generate.remote(
            [{"prompt_token_ids": list(prompt_ids)}], params,
            use_tqdm=False, lora_request=_lora_request(),
        ) for engine in trainer.engines
    ])
    if len(batches) != 4 or any(len(batch) != 1 for batch in batches):
        raise RuntimeError("v40a forward probe coverage changed")
    records = [output_record(batch[0]) for batch in batches]
    hashes = [canonical_sha256(record) for record in records]
    if len(set(hashes)) != 1:
        raise RuntimeError("v40a four actors returned different synthetic forwards")
    return {
        "synthetic_prompt_sha256": hashlib.sha256(SYNTHETIC_PROMPT.encode()).hexdigest(),
        "prompt_token_count": len(prompt_ids),
        "prompt_token_ids_sha256": canonical_sha256(prompt_ids),
        "actor_output_sha256": hashes,
        "consensus_output_sha256": hashes[0],
        "records": records,
    }


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    prereg = load_preregistration(args)
    if args.dry_run:
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "dataset_or_evaluation_accessed": False,
        }, sort_keys=True))
        return 0
    if ATTEMPT.exists() or RUN_DIR.exists():
        raise RuntimeError("v40a requires fresh artifact paths")
    preflight = gpu_preflight()
    attempt = self_hashed({
        "schema": "lora-topology-attempt-v40a", "status": "launching",
        "phase": "before_model_launch", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_file_sha256": args.preregistration_sha256,
        "preregistration_content_sha256": args.preregistration_content_sha256,
        "preflight": preflight, "dataset_or_evaluation_accessed": False,
    })
    atomic_json(ATTEMPT, attempt)
    RUN_DIR.mkdir(parents=True)
    trainer = monitor = None
    stop = threading.Event(); failures = queue.Queue(); phase = Phase()
    started = time.monotonic()
    try:
        base.set_seed(20_260_715)
        trainer = make_trainer(prereg)
        actor_ids = trainer._resolve([
            engine.runtime_identity_v40a.remote() for engine in trainer.engines
        ])
        worker_ids = _rpc_all(trainer, "runtime_identity_v40a")
        pid_map = validate_identities(actor_ids, worker_ids)
        monitor = threading.Thread(
            target=monitor_gpus, args=(stop, phase, pid_map, GPU_LOG, failures),
            daemon=True,
        ); monitor.start()

        phase.value = "activate_baseline"
        baseline = probe_forward(trainer)
        phase.value = "inventory"
        inventories = _rpc_all(trainer, "inventory_lora_topology_v40a", (
            str(ADAPTER_FILE), file_sha256(ADAPTER_FILE),
        ))
        content = [item["content_sha256_before_self_field"] for item in inventories]
        # Storage pointers and PIDs are actor-local, but all semantic fields must agree.
        def semantic(item):
            value = json.loads(json.dumps(item))
            value.pop("pid", None); value.pop("content_sha256_before_self_field", None)
            for section in ("mapping", "runtime_views"):
                for row in value.get(section, []):
                    row.pop("storage_data_ptr", None)
                    for view in row.get("runtime_views", []):
                        view.pop("storage_data_ptr", None)
            for row in value["base_layer_weights"]["tensors"]:
                row.pop("storage_data_ptr", None)
            return value
        semantic_hashes = [canonical_sha256(semantic(item)) for item in inventories]
        if len(set(semantic_hashes)) != 1:
            raise RuntimeError("v40a actor LoRA topology differs")

        phase.value = "mutate"
        mutations = _rpc_all(trainer, "mutate_one_lora_element_v40a", (1.0,))
        mutated = probe_forward(trainer)
        if mutated["consensus_output_sha256"] == baseline["consensus_output_sha256"]:
            raise RuntimeError("v40a resident one-element mutation had no forward effect")

        phase.value = "restore"
        restorations = _rpc_all(trainer, "restore_one_lora_element_v40a")
        if not all(item["restored_exactly"] for item in restorations):
            raise RuntimeError("v40a LoRA restoration was not exact")
        restored = probe_forward(trainer)
        if restored["consensus_output_sha256"] != baseline["consensus_output_sha256"]:
            raise RuntimeError("v40a restored forward did not return to baseline")

        stop.set(); monitor.join(timeout=10)
        if monitor.is_alive() or not failures.empty():
            raise RuntimeError("v40a GPU monitor failed") from (
                failures.get() if not failures.empty() else None
            )
        gpu = summarize_gpu(GPU_LOG, pid_map)
        cleanup = cleanup_v38a.strict_close_trainer_v38a(trainer)
        trainer = None
        import ray
        ray.shutdown()
        idle = cleanup_v38a.wait_for_gpu_idle()
        report = self_hashed({
            "schema": "lora-topology-probe-report-v40a",
            "status": "complete_train_only_four_gpu",
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_runtime_seconds": time.monotonic() - started,
            "preregistration": {
                "path": str(Path(args.preregistration).resolve()),
                "file_sha256": args.preregistration_sha256,
                "content_sha256": args.preregistration_content_sha256,
            },
            "actor_identities": actor_ids, "worker_identities": worker_ids,
            "physical_gpu_pid_map": pid_map,
            "adapter": {
                "source_path": str(ADAPTER),
                "source_weights_sha256": file_sha256(ADAPTER_FILE),
                "staged_path": str(STAGED_ADAPTER),
                "staged_weights_sha256": file_sha256(STAGED_ADAPTER_FILE),
                "stage_manifest_sha256": file_sha256(STAGE_MANIFEST),
                "rank": 32, "alpha": 64, "max_loras": 1, "active_id": 1,
            },
            "inventory_semantic_consensus_sha256": semantic_hashes[0],
            "actor_inventory_content_sha256": content,
            "inventory": semantic(inventories[0]),
            "forward_equivalence_probe": {
                "baseline": baseline, "mutated": mutated, "restored": restored,
                "mutations": mutations, "restorations": restorations,
                "mutation_changed_subsequent_forward_without_reload": True,
                "restoration_recovered_exact_forward": True,
            },
            "gpu_activity": gpu, "placement_group_cleanup": cleanup,
            "final_gpu_idle": idle, "preflight": preflight,
            "gpu_log": {"path": str(GPU_LOG), "file_sha256": file_sha256(GPU_LOG)},
            "dataset_or_evaluation_accessed": False,
            "synthetic_prompt_only": True,
        })
        atomic_json(REPORT, report)
        print(json.dumps({
            "report": str(REPORT), "report_sha256": file_sha256(REPORT),
            "runtime_active_allocated_elements": report["inventory"][
                "runtime_active_allocated_elements"],
            "peft_elements": report["inventory"]["peft_elements"],
            "mutation_forward_effect": True,
        }, sort_keys=True))
        return 0
    except BaseException as error:
        stop.set()
        if monitor is not None: monitor.join(timeout=10)
        atomic_json(RUN_DIR / "failure_v40a.json", self_hashed({
            "schema": "lora-topology-probe-failure-v40a",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(),
            "dataset_or_evaluation_accessed": False,
        }))
        raise
    finally:
        if trainer is not None:
            try: base.close_trainer(trainer)
            except Exception: pass
        try:
            import ray
            ray.shutdown()
        except Exception: pass


if __name__ == "__main__":
    raise SystemExit(main())
