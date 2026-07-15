#!/usr/bin/env python3
"""Fail-closed four-GPU evaluator for the preregistered V27A MoE table."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PREREG_PATH = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V27A_MOE_TUNING_EVALUATION_PREREGISTRATION.json"
)
PREREG_COMMIT = "2572204429cf2b016d0a001086d309b582b97724"
PREREG_FILE_SHA256 = "8e2571fc4fd364ff6f80122ce614a6f4f6ac413f684969da7e215e851c4ee47e"
PREREG_CONTENT_SHA256 = "9259d596c69918fd17a81573e3fbc0a74f3b024d5cb60cff2600c183862e92b8"
OFFICIAL_TUNER_PATH = Path("/tmp/benchmark_moe_v025.py")
OFFICIAL_TUNER_SHA256 = "230151de56d177ac22920ff9dada010f4361933b34b54e18d5981367723a8bcf"
MODEL_PATH = ROOT / "models/Qwen3.6-35B-A3B"
MODEL_CONFIG_SHA256 = "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
BATCH_SIZES = (256, 512, 1024, 2048)
GPU_UUIDS = (
    "GPU-4c394fc5-b18f-6622-ca94-f7fbd7112927",
    "GPU-f10c2baf-536b-1d40-cd4b-25b202ae0ded",
    "GPU-04cde663-7c53-2f18-3ec4-1699820e2640",
    "GPU-972bf85d-1b32-2d1b-20f6-babc4c804999",
)
EXPECTED_TUNED_FILENAME = (
    "E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_"
    "Max-Q_Workstation_Edition.json"
)
RUNS = ROOT / "experiments/eggroll_es_hpo/runs"
EXPERIMENT_NAME = "s6_v27a_moe_tuned_vs_default_fresh_paired_kernel_evaluation"
ATTEMPT_PATH = RUNS / f".{EXPERIMENT_NAME}.launch_attempt.json"
REPORT_PATH = RUNS / EXPERIMENT_NAME / "moe_tuning_evaluation_v27a.json"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value):
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def seal(value):
    value = dict(value)
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def exclusive_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def rewrite(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_preregistration():
    raw = subprocess.check_output(
        ["git", "show", f"{PREREG_COMMIT}:{PREREG_PATH.relative_to(ROOT)}"],
        cwd=ROOT,
    )
    if (
        hashlib.sha256(raw).hexdigest() != PREREG_FILE_SHA256
        or file_sha256(PREREG_PATH) != PREREG_FILE_SHA256
    ):
        raise RuntimeError("V27A preregistration file changed")
    value = json.loads(PREREG_PATH.read_text())
    if (
        value.get("content_sha256_before_self_field") != PREREG_CONTENT_SHA256
        or canonical_sha256(without_self(value)) != PREREG_CONTENT_SHA256
        or value.get("status")
        != "preregistered_before_tuning_output_observed_evaluation_not_launched"
    ):
        raise RuntimeError("V27A preregistration semantics changed")
    return value


def implementation_bundle():
    paths = {
        "runtime": Path(__file__).resolve(),
        "runtime_tests": ROOT / "test_run_vllm_moe_tuning_evaluation_v27a.py",
        "prereg_builder": ROOT / "build_vllm_moe_tuning_preregistration_v27a.py",
        "prereg_tests": ROOT / "test_build_vllm_moe_tuning_preregistration_v27a.py",
        "preregistration": PREREG_PATH,
    }
    files = {
        key: {
            "relative_path": path.relative_to(ROOT).as_posix(),
            "file_sha256": file_sha256(path),
        }
        for key, path in paths.items()
    }
    return {"files": files, "bundle_sha256": canonical_sha256(files)}


def validate_tuned_config(path, expected_sha256):
    path = Path(path).resolve()
    relative = path.relative_to(ROOT)
    if (
        path.name != EXPECTED_TUNED_FILENAME
        or file_sha256(path) != expected_sha256
    ):
        raise RuntimeError("V27A tuned config file identity changed")
    value = json.loads(path.read_text())
    if value.get("triton_version") != "3.6.0":
        raise RuntimeError("V27A tuned config Triton version changed")
    keys = {int(key) for key in value if key != "triton_version"}
    if keys != set(BATCH_SIZES):
        raise RuntimeError("V27A tuned config batch coverage changed")
    allowed = {
        "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
        "num_warps", "num_stages",
    }
    if any(
        set(config) != allowed
        or any(isinstance(item, bool) or not isinstance(item, int) for item in config.values())
        for key, config in value.items() if key != "triton_version"
    ):
        raise RuntimeError("V27A tuned config geometry changed")
    result = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(relative)], cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("V27A tuned config differs from committed HEAD")
    tracked = subprocess.check_output(
        ["git", "ls-files", "--error-unmatch", str(relative)],
        cwd=ROOT,
        stderr=subprocess.DEVNULL,
    )
    if not tracked:
        raise RuntimeError("V27A tuned config is not committed")
    return value


def hardware_identity():
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,driver_version,uuid",
        "--format=csv,noheader,nounits",
    ], text=True)
    rows = []
    for line in raw.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 4:
            raise RuntimeError("V27A nvidia-smi hardware row changed")
        rows.append({
            "index": int(fields[0]), "name": fields[1],
            "driver_version": fields[2], "uuid": fields[3],
        })
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3"
        or len(rows) != 4
        or [item["index"] for item in rows] != [0, 1, 2, 3]
        or tuple(item["uuid"] for item in rows) != GPU_UUIDS
        or any(
            item["name"]
            != "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
            or item["driver_version"] != "610.43.02"
            for item in rows
        )
    ):
        raise RuntimeError("V27A exact four-GPU hardware identity changed")
    return rows


def _load_official_module(path=OFFICIAL_TUNER_PATH):
    if file_sha256(path) != OFFICIAL_TUNER_SHA256:
        raise RuntimeError("V27A official vLLM tuner identity changed")
    spec = importlib.util.spec_from_file_location("vllm_benchmark_moe_v025", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _worker_evaluate(physical_gpu, batch_size, seed, config_folder):
    """Executed inside a one-GPU Ray actor; returns hardware and timing only."""
    import ray
    import torch

    module = _load_official_module()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    ray_ids = ray.get_gpu_ids()
    if visible != str(physical_gpu) or ray_ids != [physical_gpu]:
        raise RuntimeError("V27A Ray physical-GPU assignment changed")
    torch.set_default_device("cuda")
    module.set_random_seed(seed)
    os.environ["VLLM_TUNED_CONFIG_FOLDER"] = str(config_folder)
    dtype = torch.bfloat16
    dtype_string = module._get_config_dtype_str(
        dtype, use_int8_w8a16=False, use_fp8_w8a8=False,
        use_int4_w4a16=False,
    )
    config = module.get_moe_configs(256, 512, dtype_string)
    if config is None:
        selected = module.get_default_config(
            batch_size, 256, 1024, 2048, 8, dtype_string, None,
        )
        source = "default"
    else:
        selected = config[min(config, key=lambda key: abs(key - batch_size))]
        source = "tuned"
    microseconds = module.benchmark_config(
        selected, batch_size, 256, 1024, 2048, 8, dtype,
        False, False, False, num_iters=100, block_quant_shape=None,
        use_deep_gemm=False,
    )
    if not math.isfinite(microseconds) or microseconds <= 0:
        raise RuntimeError("V27A nonfinite kernel time")
    return {
        "physical_gpu": physical_gpu,
        "cuda_visible_devices": visible,
        "runtime_cuda_device": torch.cuda.current_device(),
        "gpu_name": torch.cuda.get_device_name(),
        "batch_size": batch_size,
        "seed": seed,
        "config_source": source,
        "config": selected,
        "kernel_time_microseconds": microseconds,
    }


def run_arm(arm, config_folder, seed):
    import ray

    if ray.is_initialized():
        raise RuntimeError("V27A requires a fresh Ray runtime per arm")
    ray.init(num_gpus=4, include_dashboard=False)

    @ray.remote(num_gpus=1)
    class Worker:
        def identity(self):
            ids = ray.get_gpu_ids()
            return {"ray_gpu_ids": ids, "visible": os.environ.get("CUDA_VISIBLE_DEVICES")}

        def evaluate(self, physical_gpu, batch_size, worker_seed, folder):
            return _worker_evaluate(physical_gpu, batch_size, worker_seed, folder)

    try:
        workers = [Worker.remote() for _ in range(4)]
        identities = ray.get([worker.identity.remote() for worker in workers])
        by_gpu = {}
        for worker, identity in zip(workers, identities):
            ids = identity.get("ray_gpu_ids")
            visible = identity.get("visible")
            if (
                not isinstance(ids, list) or len(ids) != 1
                or visible != str(ids[0]) or ids[0] in by_gpu
            ):
                raise RuntimeError("V27A Ray one-actor-per-GPU identity changed")
            by_gpu[int(ids[0])] = worker
        if set(by_gpu) != {0, 1, 2, 3}:
            raise RuntimeError("V27A Ray four-GPU coverage changed")
        results = ray.get([
            by_gpu[gpu].evaluate.remote(gpu, batch, seed, str(config_folder))
            for gpu, batch in enumerate(BATCH_SIZES)
        ])
    finally:
        ray.shutdown()
    expected_source = arm
    if any(
        result.get("physical_gpu") != gpu
        or result.get("batch_size") != BATCH_SIZES[gpu]
        or result.get("runtime_cuda_device") != 0
        or result.get("gpu_name")
        != "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
        or result.get("config_source") != expected_source
        for gpu, result in enumerate(results)
    ):
        raise RuntimeError("V27A arm result identity changed")
    return results


def summarize(records):
    by_batch = {}
    for batch in BATCH_SIZES:
        pairs = []
        for record in records:
            default = next(x for x in record["default"] if x["batch_size"] == batch)
            tuned = next(x for x in record["tuned"] if x["batch_size"] == batch)
            pairs.append(default["kernel_time_microseconds"] / tuned["kernel_time_microseconds"])
        median = statistics.median(pairs)
        by_batch[str(batch)] = {
            "paired_speedups": pairs,
            "median_speedup": median,
            "tuned_faster_repetitions": sum(value > 1.0 for value in pairs),
            "pass": median >= 1.0 and sum(value > 1.0 for value in pairs) >= 4,
        }
    geometric_mean = math.exp(statistics.fmean(
        math.log(value["median_speedup"]) for value in by_batch.values()
    ))
    passed = all(value["pass"] for value in by_batch.values()) and geometric_mean >= 1.03
    return seal({
        "schema": "vllm-moe-tuning-paired-summary-v27a",
        "batches": by_batch,
        "global_geometric_mean_speedup": geometric_mean,
        "pass": passed,
        "decision": (
            "authorize_separate_end_to_end_train_only_runtime_ab_preregistration"
            if passed else "discard_tuned_table_and_keep_vllm_generic_defaults"
        ),
        "direct_recipe_adoption_authorized": False,
        "model_update_checkpoint_dataset_promotion_or_nontrain_evaluation_authorized": False,
    })


def run_evaluation(tuned_path, tuned_sha256, expected_implementation):
    prereg = load_preregistration()
    implementation = implementation_bundle()
    if implementation["bundle_sha256"] != expected_implementation:
        raise RuntimeError("V27A implementation bundle changed")
    tuned_config = validate_tuned_config(tuned_path, tuned_sha256)
    hardware = hardware_identity()
    if file_sha256(MODEL_PATH / "config.json") != MODEL_CONFIG_SHA256:
        raise RuntimeError("V27A model config changed")
    default_folder = (RUNS / ".v27a_empty_default_config").resolve()
    if default_folder.exists() and any(default_folder.iterdir()):
        raise RuntimeError("V27A default config folder is not empty")
    default_folder.mkdir(parents=True, exist_ok=True)
    if ATTEMPT_PATH.exists() or REPORT_PATH.parent.exists():
        raise RuntimeError("V27A requires fresh exclusive output paths")
    attempt = seal({
        "schema": "vllm-moe-tuning-evaluation-attempt-v27a",
        "status": "launching",
        "phase": "before_first_arm",
        "preregistration_content_sha256": PREREG_CONTENT_SHA256,
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "tuned_config_file_sha256": tuned_sha256,
        "hardware_identity_sha256": canonical_sha256(hardware),
        "model_update_applied": False,
        "checkpoint_written": False,
        "dataset_surface_opened": False,
        "nontrain_evaluation_surface_opened": False,
    })
    exclusive_write(ATTEMPT_PATH, attempt)
    records = []
    try:
        for item in prereg["evaluation"]["schedule"]:
            record = {"repetition": item["repetition"], "seed": item["seed"]}
            for arm in item["arm_order"]:
                folder = default_folder if arm == "default" else Path(tuned_path).parent
                record[arm] = run_arm(arm, folder, item["seed"])
            records.append(record)
        summary = summarize(records)
        report = seal({
            "schema": "vllm-moe-tuning-evaluation-report-v27a",
            "preregistration_content_sha256": PREREG_CONTENT_SHA256,
            "implementation": implementation,
            "tuned_config": {
                "path": str(Path(tuned_path).resolve()),
                "file_sha256": tuned_sha256,
                "content_sha256": canonical_sha256(tuned_config),
            },
            "hardware": hardware,
            "records": records,
            "summary": summary,
            "model_update_applied": False,
            "checkpoint_written": False,
            "dataset_surface_opened": False,
            "nontrain_evaluation_surface_opened": False,
            "direct_recipe_adoption_applied": False,
        })
        exclusive_write(REPORT_PATH, report)
        attempt.update({
            "status": "complete",
            "phase": "after_compact_report",
            "report_binding": {
                "path": str(REPORT_PATH),
                "file_sha256": file_sha256(REPORT_PATH),
                "content_sha256": report["content_sha256_before_self_field"],
            },
        })
        attempt = seal(without_self(attempt))
        rewrite(ATTEMPT_PATH, attempt)
        return report
    except BaseException as error:
        attempt.update({
            "status": "failed", "phase": "inside_paired_kernel_evaluation",
            "failure_type": type(error).__name__,
            "failure_sha256": canonical_sha256(repr(error)),
        })
        attempt = seal(without_self(attempt))
        rewrite(ATTEMPT_PATH, attempt)
        raise


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--tuned-config", type=Path)
    value.add_argument("--expected-tuned-config-sha256")
    value.add_argument("--expected-implementation-bundle-sha256")
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    prereg = load_preregistration()
    implementation = implementation_bundle()
    if args.dry_run:
        print(json.dumps(seal({
            "schema": "vllm-moe-tuning-evaluation-dry-run-v27a",
            "preregistration_content_sha256": prereg["content_sha256_before_self_field"],
            "implementation": implementation,
            "gpu_launched": False,
            "evaluation_launched": False,
        }), sort_keys=True))
        return
    if not all((
        args.tuned_config, args.expected_tuned_config_sha256,
        args.expected_implementation_bundle_sha256,
    )):
        raise ValueError("V27A real launch requires all exact expected identities")
    run_evaluation(
        args.tuned_config, args.expected_tuned_config_sha256,
        args.expected_implementation_bundle_sha256,
    )


if __name__ == "__main__":
    main()
