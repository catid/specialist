#!/usr/bin/env python3
"""Preregistered four-arm aggregate-only V38A ES/V37A SFT evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import random
import subprocess
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pynvml

import build_train_shadow_folds_v37a as shadow
import qa_quality
import run_eggroll_es_equal_unit_v38a as run_v38a
import run_sft_train_only_control_v36a as hashing
import train_eggroll_es_equal_unit_v38a as train_v38a
import train_eggroll_es_specialist as base
import train_eggroll_es_specialist_anchor_v13 as anchor_v13


ROOT = Path(__file__).resolve().parent
MODEL = (ROOT / "models/Qwen3.6-35B-A3B").resolve()
SFT_ADAPTER = (
    ROOT / "experiments/sft_controls/v37a_equal_unit_fold3_v412/"
    "middle_late_r32_seed17/final"
).resolve()
SFT_REPORT = (
    ROOT / "experiments/sft_controls/v37a_equal_unit_fold3_v412/"
    "runtime_report_v37a.json"
).resolve()
ES_RUN = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v38a_equal_unit_fold3_pop32_antithetic_nonzero"
).resolve()
ES_REPORT = (ES_RUN / "equal_unit_update_report_v38a.json").resolve()
ES_SNAPSHOT = (ES_RUN / "selected_runtime_snapshot_v38a.safetensors").resolve()
SHADOW = (
    ROOT / "experiments/sft_controls/v37a_shadow_folds_v412/"
    "fold_3_shadow_dev.jsonl"
).resolve()
SPLIT_MANIFEST = (
    ROOT / "experiments/sft_controls/v37a_shadow_folds_v412/manifest_v37a.json"
).resolve()
OOD_QA = (ROOT / "data/ood_qa_v3.jsonl").resolve()
OOD_PROSE = (ROOT / "data/ood_prose_v3.jsonl").resolve()
LAYER_PLAN = (ROOT / "experiments/layer_plans/middle_late_dense_v6.json").resolve()
TUNED_FOLDER = (
    ROOT / "experiments/vllm_moe_tuning/"
    "v025_rtx_pro_6000_bf16_tp1_exhaustive_v27c"
).resolve()
TUNED_FILE = (
    TUNED_FOLDER /
    "E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition.json"
).resolve()
EXPERIMENT = "v39a_sealed_es_sft_base_fold3_eval"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
RAW = (RUN_DIR / "raw_items_v39a.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v39a.jsonl").resolve()
REPORT = (
    ROOT / "experiments/eval_reports/"
    "sealed_es_sft_base_fold3_eval_v39a.json"
).resolve()
ARMS = ("base_a", "base_b", "sft_v37a", "es_v38a")
GPU_IDS = (0, 1, 2, 3)
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20_260_715
GENERATION_SEED = 20_260_715


def canonical_sha256(value) -> str:
    return hashing.canonical_sha256(value)


def file_sha256(path: Path) -> str:
    return hashing.file_sha256(Path(path))


def model_shard_manifest() -> list[dict]:
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()))
    if len(names) != 26:
        raise RuntimeError("v39a base-model shard inventory changed")
    return [{
        "name": name,
        "bytes": (MODEL / name).stat().st_size,
        "file_sha256": file_sha256(MODEL / name),
    } for name in names]


def atomic_json(path: Path, value: dict, mode: int | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    if mode is not None:
        os.chmod(temporary, mode)
    os.link(temporary, path)
    temporary.unlink()


def self_hashed(value: dict) -> dict:
    result = dict(value)
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def _forbid_holdout(paths) -> None:
    for path in paths:
        lowered = str(Path(path).resolve()).lower()
        if "heldout" in lowered or "holdout" in lowered:
            raise RuntimeError("v39a holdout path access is forbidden")


class SingleAccessFirewall:
    def __init__(self, expected: dict[str, dict]):
        self.expected = expected
        self.receipts: dict[str, dict] = {}
        _forbid_holdout(item["path"] for item in expected.values())

    def _read(self, label: str) -> tuple[Path, bytes]:
        if label in self.receipts or label not in self.expected:
            raise RuntimeError(f"v39a single-access violation: {label}")
        item = self.expected[label]
        path = Path(item["path"]).resolve()
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != item["file_sha256"]:
            raise RuntimeError(f"v39a {label} identity changed")
        self.receipts[label] = {
            "path": str(path), "file_sha256": actual,
            "bytes": len(raw), "semantic_read_count": 1,
        }
        return path, raw

    def jsonl(self, label: str) -> list[dict]:
        _path, raw = self._read(label)
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
        if not rows or not all(isinstance(row, dict) for row in rows):
            raise RuntimeError(f"v39a {label} is not nonempty JSONL")
        self.receipts[label]["rows"] = len(rows)
        return rows

    def json(self, label: str) -> dict:
        _path, raw = self._read(label)
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError(f"v39a {label} is not a JSON object")
        return value


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
        raise RuntimeError("v39a preregistration file identity changed")
    value = json.loads(path.read_text())
    content = value.get("content_sha256_before_self_field")
    if (
        content != args.preregistration_content_sha256
        or content != canonical_sha256({
            key: item for key, item in value.items()
            if key != "content_sha256_before_self_field"
        })
        or value.get("schema") != "sealed-candidate-eval-preregistration-v39a"
        or value.get("status") != "preregistered_before_shadow_access"
        or value.get("heldout_or_holdout_access_authorized") is not False
    ):
        raise RuntimeError("v39a preregistration content changed")
    paths = {
        "runtime": Path(__file__).resolve(),
        "model_config": MODEL / "config.json",
        "model_index": MODEL / "model.safetensors.index.json",
        "sft_adapter_config": SFT_ADAPTER / "adapter_config.json",
        "sft_adapter_weights": SFT_ADAPTER / "adapter_model.safetensors",
        "sft_report": SFT_REPORT,
        "es_report": ES_REPORT,
        "es_snapshot": ES_SNAPSHOT,
        "shadow": SHADOW,
        "split_manifest": SPLIT_MANIFEST,
        "ood_qa": OOD_QA,
        "ood_prose": OOD_PROSE,
        "layer_plan": LAYER_PLAN,
        "tuned_table": TUNED_FILE,
        "worker_v38a": ROOT / "eggroll_es_worker_v38a.py",
        "trainer_v38a": ROOT / "train_eggroll_es_equal_unit_v38a.py",
        "runtime_v38a": ROOT / "run_eggroll_es_equal_unit_v38a.py",
        "qa_quality": ROOT / "qa_quality.py",
        "reward": ROOT / "train_eggroll_es_specialist.py",
    }
    observed = {key: file_sha256(path) for key, path in paths.items()}
    observed["model_shards_content_sha256"] = canonical_sha256(
        model_shard_manifest()
    )
    if observed != value.get("implementation_bindings"):
        raise RuntimeError("v39a bound implementation or artifact changed")
    _forbid_holdout(paths.values())
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
        raise RuntimeError("v39a requires four exclusive idle physical GPUs")
    return {"compute_process_query_empty": True, "memory_used_mib": memory}


def normalize_gpu_id(value) -> int:
    number = float(value)
    if not number.is_integer() or int(number) not in GPU_IDS:
        raise RuntimeError("v39a invalid Ray GPU id")
    return int(number)


def create_trainer(layer_bundle, prereg):
    parent = train_v38a.load_trainer(layer_bundle)
    expected_tuned_content = prereg["runtime"]["tuned_table_content_sha256"]

    class EvaluationTrainerV39A(parent):
        def launch_engines(self, num_engines=4, n_gpu_per_vllm_engine=1,
                           model_name="unused", precision="bfloat16"):
            if int(num_engines) != 4 or int(n_gpu_per_vllm_engine) != 1:
                raise RuntimeError("v39a exact four-engine TP1 design changed")
            import ray
            from ray.util.placement_group import placement_group
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
            from es_at_scale.trainer.es_trainer import ESNcclLLM

            class EvaluationLLMV39A(ESNcclLLM):
                def runtime_identity_v39a(self):
                    import torch
                    import vllm.envs as vllm_envs
                    import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe
                    raw_ids = ray.get_gpu_ids()
                    if len(raw_ids) != 1:
                        raise RuntimeError("v39a actor does not own exactly one GPU")
                    physical = normalize_gpu_id(raw_ids[0])
                    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                    folder = os.environ.get("VLLM_TUNED_CONFIG_FOLDER")
                    if (
                        visible != str(physical) or torch.cuda.device_count() != 1
                        or torch.cuda.current_device() != 0
                        or folder != str(TUNED_FOLDER)
                        or vllm_envs.VLLM_TUNED_CONFIG_FOLDER != str(TUNED_FOLDER)
                    ):
                        raise RuntimeError("v39a actor device/tuned-folder mismatch")
                    fused_moe.get_moe_configs.cache_clear()
                    config = fused_moe.get_moe_configs(256, 512, None)
                    if not isinstance(config, dict) or canonical_sha256(config) != expected_tuned_content:
                        raise RuntimeError("v39a actor did not load exact tuned table")
                    pynvml.nvmlInit()
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(physical)
                        uuid = pynvml.nvmlDeviceGetUUID(handle)
                        pci = pynvml.nvmlDeviceGetPciInfo(handle).busId
                        total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
                        if isinstance(uuid, bytes):
                            uuid = uuid.decode("ascii")
                        if isinstance(pci, bytes):
                            pci = pci.decode("ascii")
                    finally:
                        pynvml.nvmlShutdown()
                    return {
                        "schema": "sealed-candidate-actor-identity-v39a",
                        "pid": os.getpid(), "physical_gpu_id": physical,
                        "cuda_visible_devices": visible, "cuda_current_device": 0,
                        "nvml_uuid": str(uuid), "pci_bus_id": str(pci),
                        "total_bytes": int(total), "tuned_folder": folder,
                        "tuned_table_content_sha256": canonical_sha256(config),
                    }

            pgs = [placement_group([{"GPU": 1, "CPU": 0}], strategy="PACK")
                   for _ in range(4)]
            ray.get([pg.ready() for pg in pgs])
            strategies = [PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_capture_child_tasks=True,
                placement_group_bundle_index=0,
            ) for pg in pgs]
            kwargs = {
                "model": str(MODEL), "tensor_parallel_size": 1,
                "worker_extension_cls": train_v38a.WORKER_EXTENSION,
                "dtype": precision, "enable_prefix_caching": False,
                "enforce_eager": True, "gpu_memory_utilization": 0.82,
                "max_model_len": 2048, "limit_mm_per_prompt": {"image": 0, "video": 0},
                "mm_processor_cache_gb": 0, "skip_mm_profiling": True,
                "moe_backend": "triton", "enable_lora": True,
                "max_lora_rank": 32, "max_loras": 1, "max_cpu_loras": 1,
            }
            engines = [ray.remote(
                num_cpus=0, num_gpus=1, scheduling_strategy=strategy,
            )(EvaluationLLMV39A).options(runtime_env={
                "env_vars": {"VLLM_TUNED_CONFIG_FOLDER": str(TUNED_FOLDER)},
            }).remote(**kwargs) for strategy in strategies]
            return engines, pgs

    return EvaluationTrainerV39A(
        model_name=str(MODEL), checkpoint=None, sigma=0.0003, alpha=0.0,
        population_size=32, reward_shaping="z-scores", num_iterations=1,
        max_tokens=64, batch_size=83, mini_batch_size=83,
        reward_function=base.specialist_reward,
        template_function=base.specialist_template,
        train_dataloader=[], eval_dataloader_dict={}, eval_freq=1,
        n_vllm_engines=4, n_gpu_per_vllm_engine=1, logging="none",
        global_seed=GENERATION_SEED, use_gpus="0,1,2,3",
        experiment_name=EXPERIMENT, wandb_project="none", save_best_models=False,
        reward_function_timeout=10, output_directory=str(RUN_DIR.parent),
    )


def validate_actor_identities(items: list[dict], worker_items: list[dict]) -> dict[int, int]:
    if len(items) != 4 or len(worker_items) != 4:
        raise RuntimeError("v39a identity coverage changed")
    by_pid = {item["pid"]: item for item in worker_items}
    mapping = {}
    for item in items:
        if item.get("schema") != "sealed-candidate-actor-identity-v39a":
            raise RuntimeError("v39a actor identity schema changed")
        physical, pid = item["physical_gpu_id"], item["pid"]
        if physical in mapping or pid not in by_pid:
            raise RuntimeError("v39a actor/worker identity mismatch")
        if by_pid[pid].get("cuda_visible_devices") != str(physical):
            raise RuntimeError("v39a actor/worker physical GPU mismatch")
        mapping[physical] = pid
    if set(mapping) != set(GPU_IDS) or len(set(mapping.values())) != 4:
        raise RuntimeError("v39a physical GPU mapping incomplete")
    return mapping


class Phase:
    def __init__(self): self.value = "setup"


def monitor_gpus(stop, phase, expected_pids, path, failures):
    try:
        pynvml.nvmlInit()
        handles = {gpu: pynvml.nvmlDeviceGetHandleByIndex(gpu) for gpu in GPU_IDS}
        with Path(path).open("x") as output:
            while not stop.is_set():
                sampled = datetime.now(timezone.utc).isoformat()
                for gpu, handle in handles.items():
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    pids = sorted({int(item.pid) for item in processes})
                    foreign = [pid for pid in pids if pid != expected_pids[gpu]]
                    if foreign:
                        raise RuntimeError(f"v39a foreign GPU process: {gpu} {foreign}")
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    output.write(json.dumps({
                        "sampled_at_utc": sampled, "phase": phase.value,
                        "gpu": gpu, "expected_pid": expected_pids[gpu],
                        "compute_pids": pids, "foreign_compute_pids": foreign,
                        "utilization_percent": int(utilization.gpu),
                        "memory_used_mib": int(memory.used // 2**20),
                    }, sort_keys=True) + "\n")
                output.flush(); stop.wait(0.5)
    except BaseException as error:
        failures.put(error)
    finally:
        try: pynvml.nvmlShutdown()
        except Exception: pass


def summarize_gpu(path: Path, expected_pids: dict[int, int]) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    result = {}
    required_phases = ("shadow", "ood_qa", "ood_prose")
    for gpu in GPU_IDS:
        selected = [row for row in rows if row["gpu"] == gpu]
        if not selected or any(row["foreign_compute_pids"] for row in selected):
            raise RuntimeError("v39a GPU monitor coverage failed")
        if any(row["expected_pid"] != expected_pids[gpu] for row in selected):
            raise RuntimeError("v39a GPU PID attribution changed")
        phases = {}
        for label in required_phases:
            phase_rows = [row for row in selected if row["phase"] == label]
            resident = [row for row in phase_rows if expected_pids[gpu] in row["compute_pids"]]
            positive = [row for row in resident if row["utilization_percent"] > 0]
            if not resident or not positive:
                raise RuntimeError(f"v39a GPU {gpu} inactive in {label}")
            phases[label] = {
                "samples": len(phase_rows), "resident_samples": len(resident),
                "positive_samples": len(positive),
                "peak_utilization_percent": max(row["utilization_percent"] for row in resident),
                "peak_memory_used_mib": max(row["memory_used_mib"] for row in resident),
                "mean_resident_utilization_percent": math.fsum(
                    row["utilization_percent"] for row in resident) / len(resident),
            }
        result[str(gpu)] = {"expected_pid": expected_pids[gpu], "phases": phases}
    return {"all_four_attributed_positive_each_phase": True, "by_gpu": result}


def qa_pair(row: dict) -> tuple[str, str]:
    pair = qa_quality.qa_pair_from_record(row)
    if pair is None:
        question, answer = row.get("question"), row.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            raise RuntimeError("v39a QA row is invalid")
        pair = (question.strip(), answer.strip())
    if not pair[0] or not pair[1] or qa_quality.has_protocol_tokens(pair[0]) or qa_quality.has_protocol_tokens(pair[1]):
        raise RuntimeError("v39a QA protocol content changed")
    return pair


def shadow_bundle(rows: list[dict], manifest: dict) -> dict:
    if len(rows) != 83:
        raise RuntimeError("v39a shadow row count changed")
    commitments = {}
    for unit in manifest.get("content_free_unit_commitments", []):
        if unit.get("fold") != 3: continue
        for row_sha in unit["row_sha256"]:
            commitments[row_sha] = (unit["unit_identity_sha256"], unit["row_count"])
    row_hashes = [shadow.row_sha256(row) for row in rows]
    if set(row_hashes) != set(commitments) or len(commitments) != 83:
        raise RuntimeError("v39a shadow commitments changed")
    if len({unit for unit, _count in commitments.values()}) != 51:
        raise RuntimeError("v39a shadow unit count changed")
    pairs = [qa_pair(row) for row in rows]
    weights = [1.0 / (51 * commitments[row_sha][1]) for row_sha in row_hashes]
    if not math.isclose(math.fsum(weights), 1.0, abs_tol=1e-15, rel_tol=0.0):
        raise RuntimeError("v39a shadow unit weights changed")
    return {"questions": [p[0] for p in pairs], "answers": [p[1] for p in pairs],
            "item_sha256": row_hashes, "weights": weights, "rows": 83, "units": 51}


def qa_bundle(rows: list[dict]) -> dict:
    pairs = [qa_pair(row) for row in rows]
    item_hashes = [canonical_sha256({"question": q, "answer": a}) for q, a in pairs]
    if len(set(item_hashes)) != len(item_hashes):
        raise RuntimeError("v39a OOD QA identities repeated")
    return {"questions": [p[0] for p in pairs], "answers": [p[1] for p in pairs],
            "item_sha256": item_hashes, "weights": [1.0 / len(rows)] * len(rows),
            "rows": len(rows), "units": len(rows)}


def _lora_request():
    from vllm.lora.request import LoRARequest
    return LoRARequest("v37a_sft", 1, str(SFT_ADAPTER), base_model_name=str(MODEL))


def _arm_requests(engines, prompts, params):
    handles = []
    for arm, engine in zip(ARMS, engines, strict=True):
        kwargs = {"use_tqdm": False}
        if arm == "sft_v37a": kwargs["lora_request"] = _lora_request()
        handles.append(engine.generate.remote(list(prompts), params, **kwargs))
    return handles


def protocol_counters(question: str, response: str) -> dict:
    normalized_q = qa_quality.normalize_text(question)
    normalized_r = qa_quality.normalize_text(response)
    return {
        "protocol_token_emission": int(qa_quality.has_protocol_tokens(response)),
        "prompt_echo": int(len(normalized_q) >= 10 and normalized_q in normalized_r),
        "empty_extracted_answer": int(not base.extract_answer(response)),
    }


def evaluate_qa(trainer, bundle: dict, raw_sink: dict, label: str) -> dict:
    import ray
    from vllm import SamplingParams
    prompts = [base.specialist_template(q) for q in bundle["questions"]]
    dense_items = anchor_v13.anchor_v4.prepare_gold_answer_items_v4(
        trainer.tokenizer, prompts, bundle["answers"],
    )
    dense_prompts = [{"prompt_token_ids": item["prompt_token_ids"]} for item in dense_items]
    teacher = ray.get(_arm_requests(trainer.engines, dense_prompts, SamplingParams(
        n=1, seed=GENERATION_SEED, temperature=0.0, top_p=1.0,
        max_tokens=1, prompt_logprobs=1, detokenize=False,
    )))
    generated = ray.get(_arm_requests(trainer.engines, prompts, SamplingParams(
        n=1, seed=GENERATION_SEED, temperature=0.0, top_p=1.0,
        max_tokens=64,
    )))
    result = {}
    for arm, teacher_outputs, generated_outputs in zip(ARMS, teacher, generated, strict=True):
        dense = anchor_v13.anchor_v4.score_gold_answer_outputs_v4(dense_items, teacher_outputs)
        if len(generated_outputs) != bundle["rows"]:
            raise RuntimeError("v39a generated QA result count changed")
        records = []
        raw_rows = []
        for index, (question, answer, identity, dense_row, output) in enumerate(zip(
            bundle["questions"], bundle["answers"], bundle["item_sha256"],
            dense["examples"], generated_outputs, strict=True,
        )):
            if len(output.outputs) != 1:
                raise RuntimeError("v39a generated QA multiplicity changed")
            response = output.outputs[0].text
            fmt, reward = base.specialist_reward(response, answer)
            counters = protocol_counters(question, response)
            records.append({
                "item_sha256": identity,
                "teacher_mean_answer_token_logprob": dense_row["mean_answer_token_logprob"],
                "generated_reward": float(reward), "generated_format": fmt,
                "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                **counters,
            })
            raw_rows.append({
                "item_index": index, "item_sha256": identity,
                "question": question, "answer": answer, "response": response,
                "teacher": dense_row, "format": fmt, "reward": float(reward),
                "counters": counters,
            })
        weights = bundle["weights"]
        teacher_values = [row["teacher_mean_answer_token_logprob"] for row in records]
        rewards = [row["generated_reward"] for row in records]
        counters = {key: sum(row[key] for row in records) for key in (
            "protocol_token_emission", "prompt_echo", "empty_extracted_answer")}
        result[arm] = {
            "rows": bundle["rows"], "units": bundle["units"],
            "teacher_forced_equal_unit_mean_answer_logprob": math.fsum(
                w * value for w, value in zip(weights, teacher_values)),
            "generated_equal_unit_mean_reward": math.fsum(
                w * value for w, value in zip(weights, rewards)),
            "generated_row_mean_reward": math.fsum(rewards) / len(rewards),
            "generated_exact_count": sum(row["generated_format"] == "exact" for row in records),
            "generated_nonzero_count": sum(value > 0.0 for value in rewards),
            "protocol_leak_counters": counters,
            "numeric_item_manifest_sha256": canonical_sha256(records),
        }
        raw_sink.setdefault(label, {})[arm] = raw_rows
    if result["base_a"] != result["base_b"]:
        raise RuntimeError(f"v39a base duplicate equivalence failed on {label}")
    return result


def selection_key(metrics: dict, arm: str) -> tuple:
    return (
        metrics["generated_equal_unit_mean_reward"],
        metrics["generated_exact_count"], metrics["generated_nonzero_count"],
        metrics["teacher_forced_equal_unit_mean_answer_logprob"],
        1 if arm == "es_v38a" else 0,
    )


def select_candidate(shadow_metrics: dict) -> dict:
    selected = max(("sft_v37a", "es_v38a"),
                   key=lambda arm: selection_key(shadow_metrics[arm], arm))
    candidate, baseline = shadow_metrics[selected], shadow_metrics["base_a"]
    no_protocol_increase = all(
        candidate["protocol_leak_counters"][key]
        <= baseline["protocol_leak_counters"][key]
        for key in baseline["protocol_leak_counters"]
    )
    passed = (
        selection_key(candidate, selected)[:-1]
        > selection_key(baseline, "base_a")[:-1]
        and no_protocol_increase
    )
    return {
        "selected_arm": selected,
        "rule": "lexicographic generated mean, exact, nonzero, teacher logprob; ES final tie",
        "shadow_improvement_gate_passed": passed,
        "no_protocol_or_leak_counter_increase": no_protocol_increase,
    }


def evaluate_prose(trainer, rows, raw_sink):
    import ray
    from vllm import SamplingParams
    items = base.prepare_ood_prose_items(rows, trainer.tokenizer, 1024)
    prompts = [{"prompt_token_ids": item["prompt_token_ids"]} for item in items]
    batches = ray.get(_arm_requests(trainer.engines, prompts, SamplingParams(
        n=1, seed=GENERATION_SEED, temperature=0.0, top_p=1.0,
        max_tokens=1, prompt_logprobs=1, detokenize=False,
    )))
    detailed, aggregate = {}, {}
    for arm, outputs in zip(ARMS, batches, strict=True):
        value = base.summarize_ood_prose(items, outputs)
        detailed[arm] = value
        raw_sink.setdefault("ood_prose", {})[arm] = value["items"]
        compact = [{
            "text_sha256": row["text_sha256"],
            "token_ids_sha256": row["token_ids_sha256"],
            "scored_token_count": row["scored_token_count"],
            "sum_token_logprob": row["sum_token_logprob"],
        } for row in value["items"]]
        aggregate[arm] = {
            "item_count": value["item_count"],
            "scored_token_count": value["scored_token_count"],
            "mean_token_logprob": value["mean_token_logprob"],
            "numeric_item_manifest_sha256": canonical_sha256(compact),
        }
    if aggregate["base_a"] != aggregate["base_b"]:
        raise RuntimeError("v39a prose base duplicate equivalence failed")
    return aggregate, detailed


def prose_gate(baseline, candidate) -> dict:
    comparison = base.compare_ood_prose(
        baseline, candidate, max_degradation=0.0,
        bootstrap_samples=BOOTSTRAP_SAMPLES, bootstrap_seed=BOOTSTRAP_SEED,
    )
    comparison["point_non_degradation_passed"] = comparison["delta"] >= 0.0
    comparison["bootstrap_lcb_non_degradation_passed"] = (
        comparison["paired_document_bootstrap_95_ci"][0] >= 0.0
    )
    comparison["passed"] = (
        comparison["point_non_degradation_passed"]
        and comparison["bootstrap_lcb_non_degradation_passed"]
    )
    return comparison


def qa_ood_gate(baseline: dict, candidate: dict) -> dict:
    return {
        "mean_reward_delta": (
            candidate["generated_row_mean_reward"] - baseline["generated_row_mean_reward"]
        ),
        "exact_count_delta": candidate["generated_exact_count"] - baseline["generated_exact_count"],
        "mean_reward_non_degradation_passed": (
            candidate["generated_row_mean_reward"] >= baseline["generated_row_mean_reward"]
        ),
        "exact_non_degradation_passed": (
            candidate["generated_exact_count"] >= baseline["generated_exact_count"]
        ),
        "passed": (
            candidate["generated_row_mean_reward"] >= baseline["generated_row_mean_reward"]
            and candidate["generated_exact_count"] >= baseline["generated_exact_count"]
        ),
    }


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    prereg = load_preregistration(args)
    if args.dry_run:
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "shadow_semantically_opened": False, "heldout_opened": False,
        }, sort_keys=True))
        return 0
    if ATTEMPT.exists() or RUN_DIR.exists() or REPORT.exists():
        raise RuntimeError("v39a requires fresh artifact paths")
    preflight = gpu_preflight()
    attempt = self_hashed({
        "schema": "sealed-candidate-eval-attempt-v39a", "status": "launching",
        "phase": "before_model_or_shadow_access",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_file_sha256": args.preregistration_sha256,
        "preregistration_content_sha256": args.preregistration_content_sha256,
        "preflight": preflight, "heldout_or_holdout_opened": False,
    })
    atomic_json(ATTEMPT, attempt)
    RUN_DIR.mkdir(parents=True)
    trainer = monitor = None
    stop = threading.Event(); failures = queue.Queue(); phase = Phase()
    raw_sink = {"schema": "sealed-candidate-raw-local-v39a"}
    started = time.monotonic()
    try:
        base.set_seed(GENERATION_SEED)
        layer = anchor_v13.load_frozen_layer_plan_v13(
            LAYER_PLAN,
            expected_file_sha256=prereg["implementation_bindings"]["layer_plan"],
            expected_plan_sha256=prereg["runtime"]["layer_plan_sha256"],
            expected_model_config_sha256=prereg["implementation_bindings"]["model_config"],
        )
        trainer = create_trainer(layer, prereg)
        configured = trainer.configure_equal_unit_v38a(
            {"content_sha256_before_self_field": "v39a-evaluation-only"},
            frozen_layer_plan=layer,
        )
        actor_ids = trainer._resolve([
            engine.runtime_identity_v39a.remote() for engine in trainer.engines
        ])
        pid_map = validate_actor_identities(actor_ids, configured["worker_identities"])
        monitor = threading.Thread(
            target=monitor_gpus,
            args=(stop, phase, pid_map, GPU_LOG, failures), daemon=True,
        ); monitor.start()
        es_report = json.loads(ES_REPORT.read_text())
        expected_final = es_report["update"]["application"]["final_identity"]
        es_loaded = trainer._resolve([
            trainer.engines[3].collective_rpc.remote(
                "load_selected_snapshot_v38a",
                args=(str(ES_SNAPSHOT), file_sha256(ES_SNAPSHOT), expected_final),
            )
        ])[0][0]
        if es_loaded.get("current_identity") != expected_final:
            raise RuntimeError("v39a ES snapshot load identity changed")
        firewall = SingleAccessFirewall(prereg["single_access_inputs"])
        manifest = firewall.json("split_manifest")
        phase.value = "shadow"
        shadow_metrics = evaluate_qa(
            trainer, shadow_bundle(firewall.jsonl("shadow"), manifest),
            raw_sink, "shadow",
        )
        selection = select_candidate(shadow_metrics)
        phase.value = "ood_qa"
        ood_qa_metrics = evaluate_qa(
            trainer, qa_bundle(firewall.jsonl("ood_qa")), raw_sink, "ood_qa",
        )
        phase.value = "ood_prose"
        ood_prose_metrics, ood_prose_details = evaluate_prose(
            trainer, firewall.jsonl("ood_prose"), raw_sink,
        )
        selected = selection["selected_arm"]
        qa_gate = qa_ood_gate(ood_qa_metrics["base_a"], ood_qa_metrics[selected])
        prose = prose_gate(ood_prose_details["base_a"], ood_prose_details[selected])
        final_gate = {
            "shadow_improvement": selection["shadow_improvement_gate_passed"],
            "ood_qa_no_degradation": qa_gate["passed"],
            "ood_prose_point_and_lcb_no_degradation": prose["passed"],
        }
        final_gate["passed"] = all(final_gate.values())
        raw_sink["selection"] = selection
        raw_sink["access_receipts"] = firewall.receipts
        atomic_json(RAW, raw_sink, mode=0o600)
        raw_sha = file_sha256(RAW)
        stop.set(); monitor.join(timeout=10)
        if monitor.is_alive() or not failures.empty():
            raise RuntimeError("v39a GPU monitor failed") from (
                failures.get() if not failures.empty() else None
            )
        gpu = summarize_gpu(GPU_LOG, pid_map)
        cleanup = run_v38a.strict_close_trainer_v38a(trainer)
        trainer = None
        import ray
        ray.shutdown()
        idle = run_v38a.wait_for_gpu_idle()
        report = self_hashed({
            "schema": "sealed-candidate-eval-aggregate-v39a",
            "status": "complete_aggregate_only_no_heldout_access",
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_runtime_seconds": time.monotonic() - started,
            "preregistration": {
                "path": str(Path(args.preregistration).resolve()),
                "file_sha256": args.preregistration_sha256,
                "content_sha256": args.preregistration_content_sha256,
            },
            "arms": list(ARMS), "actor_identities": actor_ids,
            "es_snapshot_load": es_loaded,
            "shadow": shadow_metrics, "selection": selection,
            "ood_qa": ood_qa_metrics, "ood_qa_gate": qa_gate,
            "ood_prose": ood_prose_metrics, "ood_prose_gate": prose,
            "final_gate": final_gate,
            "single_access_receipts": firewall.receipts,
            "gpu_activity": gpu, "placement_group_cleanup": cleanup,
            "final_gpu_idle": idle, "preflight": preflight,
            "raw_local_artifact": {
                "path": str(RAW), "file_sha256": raw_sha,
                "git_eligible": False, "raw_content_in_aggregate": False,
            },
            "gpu_log": {"path": str(GPU_LOG), "file_sha256": file_sha256(GPU_LOG)},
            "heldout_or_holdout_opened": False,
            "raw_questions_answers_or_generations_persisted_in_aggregate": False,
        })
        atomic_json(REPORT, report)
        attempt.update({
            "status": "complete", "phase": "aggregate_sealed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "report": str(REPORT), "report_sha256": file_sha256(REPORT),
        })
        attempt.pop("content_sha256_before_self_field", None)
        atomic_json(ATTEMPT.with_suffix(".complete.json"), self_hashed(attempt))
        print(json.dumps({
            "report": str(REPORT), "report_sha256": file_sha256(REPORT),
            "selected_arm": selected, "final_gate_passed": final_gate["passed"],
        }, sort_keys=True))
        return 0
    except BaseException as error:
        stop.set()
        if monitor is not None: monitor.join(timeout=10)
        failure = self_hashed({
            "schema": "sealed-candidate-eval-failure-v39a",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(),
            "heldout_or_holdout_opened": False,
        })
        atomic_json(RUN_DIR / "failure_v39a.json", failure)
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
