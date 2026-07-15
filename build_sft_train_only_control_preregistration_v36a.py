#!/usr/bin/env python3
"""Build the immutable train-only V36A sparse-LoRA control preregistration."""

from __future__ import annotations

import json
from pathlib import Path

import run_sft_train_only_control_v36a as runtime


ROOT = Path(__file__).resolve().parent
RUN_DIR = (ROOT / "experiments/sft_controls/v36a_v412").resolve()
DATASET = RUN_DIR / "train_v412.jsonl"
OUTPUT_DIR = RUN_DIR / "middle_late_r32_seed17"
PREREGISTRATION = RUN_DIR / "preregistration_v36a.json"
ARTIFACT_TAG = "v36a"
LAYER_PLAN = (ROOT / "experiments/layer_plans/middle_late_dense_v6.json").resolve()

EXPECTED = {
    "dataset_sha256": "44a5482e49073d23a35bdc4c574d6c52c9e8d7f946559dfa722dda16eec5882b",
    "dataset_rows": 531,
    "dataset_documents": 289,
    "runner_sha256": "e9823359d82aeb61d7fd1f17f24ea22fc46680cc122c55232b1913d765d1d49d",
    "sft_sha256": "ce574db4137764e52402d12ee6a4b2081f7339fada85bc328564f95dbf908769",
    "model_config_sha256": "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
    "model_index_sha256": "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
    "layer_plan_sha256": "d65d702969dcec7a56ca4fcf461d402c44642966191a57c2ef092ec339e3e3df",
    "trainable_elements": 4_528_128,
    "trainable_tensors": 70,
}


def _arguments():
    return runtime.parser().parse_args([
        "--dataset", str(DATASET),
        "--dataset-sha256", EXPECTED["dataset_sha256"],
        "--dataset-rows", str(EXPECTED["dataset_rows"]),
        "--output-dir", str(OUTPUT_DIR),
        "--stdout-log", str(RUN_DIR / f"stdout_{ARTIFACT_TAG}.log"),
        "--gpu-log", str(RUN_DIR / f"gpu_activity_{ARTIFACT_TAG}.jsonl"),
        "--report", str(RUN_DIR / f"runtime_report_{ARTIFACT_TAG}.json"),
        "--attempt-report", str(RUN_DIR / f"attempt_{ARTIFACT_TAG}.json"),
        "--preregistration", str(PREREGISTRATION),
        "--preregistration-sha256", "PENDING",
        "--preregistration-content-sha256", "PENDING",
        "--epochs", "3",
        "--rank", "32",
        "--lora-dropout", "0",
        "--grad-accum", "1",
        "--per-device-batch-size", "7",
        "--learning-rate", "1e-4",
        "--seed", "17",
        "--max-length", "1024",
        "--save-steps", "19",
        "--attn-implementation", "sdpa",
        "--prompt-mode", "es_exact",
        "--loss-mode", "example_mean",
        "--target-layers", "20,21,22,23",
        "--expected-trainable-elements", str(EXPECTED["trainable_elements"]),
        "--expected-trainable-tensors", str(EXPECTED["trainable_tensors"]),
    ])


def build():
    args = _arguments()
    dataset = runtime.validate_train_dataset(
        DATASET, EXPECTED["dataset_sha256"], EXPECTED["dataset_rows"]
    )
    observed = {
        "dataset_documents": dataset["unique_documents"],
        "runner_sha256": runtime.file_sha256(ROOT / "run_sft_train_only_control_v36a.py"),
        "sft_sha256": runtime.file_sha256(ROOT / "sft_lora.py"),
        "model_config_sha256": runtime.file_sha256(runtime.BASE_MODEL / "config.json"),
        "model_index_sha256": runtime.file_sha256(
            runtime.BASE_MODEL / "model.safetensors.index.json"
        ),
        "layer_plan_sha256": runtime.file_sha256(LAYER_PLAN),
    }
    for key, value in observed.items():
        if value != EXPECTED[key]:
            raise RuntimeError(f"v36a preregistration binding changed: {key}")
    layer_plan = json.loads(LAYER_PLAN.read_text())
    if layer_plan.get("layers") != [20, 21, 22, 23] or layer_plan.get("num_units") != 35:
        raise RuntimeError("v36a retained ES layer plan changed")
    result = {
        "schema": "specialist-sft-train-only-control-preregistration-v36a",
        "status": "preregistered_not_yet_run",
        "experiment_name": "sft_train_only_v36a_v412_middle_late_r32_seed17",
        "selection_surface": "train_only_runtime_and_training_loss",
        "contains_validation_ood_or_holdout_content": False,
        "dataset": dataset,
        "implementation": {
            "runner": str((ROOT / "run_sft_train_only_control_v36a.py").resolve()),
            "runner_sha256": EXPECTED["runner_sha256"],
            "sft": str((ROOT / "sft_lora.py").resolve()),
            "sft_sha256": EXPECTED["sft_sha256"],
        },
        "model": {
            "path": str(runtime.BASE_MODEL),
            "config_sha256": EXPECTED["model_config_sha256"],
            "index_sha256": EXPECTED["model_index_sha256"],
        },
        "artifacts": {
            "output_dir": str(OUTPUT_DIR),
            "stdout_log": str(RUN_DIR / f"stdout_{ARTIFACT_TAG}.log"),
            "gpu_log": str(RUN_DIR / f"gpu_activity_{ARTIFACT_TAG}.jsonl"),
            "report": str(RUN_DIR / f"runtime_report_{ARTIFACT_TAG}.json"),
            "attempt_report": str(RUN_DIR / f"attempt_{ARTIFACT_TAG}.json"),
        },
        "comparison_binding": {
            "eggroll_es_layer_plan": str(LAYER_PLAN),
            "eggroll_es_layer_plan_sha256": EXPECTED["layer_plan_sha256"],
            "eggroll_es_layers": [20, 21, 22, 23],
            "eggroll_es_dense_tensor_count": 35,
            "eggroll_es_dense_elements": 142_999_552,
            "sft_parameterization": "rank_32_LoRA_over_the_same_35_weight_units",
            "sft_trainable_tensors": EXPECTED["trainable_tensors"],
            "sft_trainable_elements": EXPECTED["trainable_elements"],
            "remaining_mismatch": (
                "LoRA is a low-rank subspace, while ES perturbs full dense tensors; "
                "the train corpus is row-uniform rather than an equal-document panel"
            ),
        },
        "recipe": {
            "world_size": 4,
            "physical_gpu_ids": [0, 1, 2, 3],
            "epochs": 3.0,
            "per_device_batch_size": 7,
            "gradient_accumulation_steps": 1,
            "effective_global_batch_size": 28,
            "expected_optimizer_steps": 57,
            "equal_microbatch_sizes": True,
            "distributed_sampler_presentations_per_epoch": 532,
            "source_rows_per_epoch": 531,
            "distributed_padding_duplicates_per_epoch": 1,
            "learning_rate": 1e-4,
            "scheduler": "cosine",
            "warmup_ratio": 0.03,
            "bf16": True,
            "gradient_checkpointing": True,
            "attention": "sdpa",
            "prompt": "exact EGGROLL-ES specialist_template(question)+raw_answer",
            "eos_appended_or_scored": False,
            "loss": "mean answer-token NLL per example, then mean examples",
            "expected_encoding_audit": runtime.EXPECTED_ENCODING_AUDIT,
            "expected_trainable_inventory": runtime.EXPECTED_TRAINABLE_INVENTORY,
            "seed": 17,
            "command": runtime.build_train_command(args),
        },
        "required_runtime_evidence": {
            "all_four_gpus_positive_activity": True,
            "gpu_activity_attributed_to_torchrun_process_tree": True,
            "exclusive_idle_gpu_preflight": True,
            "all_four_gpus_model_resident": True,
            "exact_dataset_binding": True,
            "exact_trainable_inventory": True,
            "durable_attempt_record_and_failure_cleanup": True,
            "trainer_metrics": [
                "train_runtime", "train_samples_per_second",
                "train_steps_per_second", "train_loss", "epoch",
            ],
            "no_evaluation_data_argument": True,
        },
        "decision_policy": {
            "authorized": "runtime and train-loss characterization only",
            "forbidden": [
                "quality superiority claim", "model promotion", "dataset promotion",
                "validation access", "OOD access", "holdout access",
            ],
            "next_gate": (
                "freeze a document-disjoint train shadow split and compare SFT and "
                "a nonzero ES update under equal wall-clock budgets"
            ),
        },
    }
    result["content_sha256_before_self_field"] = runtime.canonical_sha256(result)
    return result


def main():
    result = build()
    PREREGISTRATION.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(PREREGISTRATION)
    print(result["content_sha256_before_self_field"])


if __name__ == "__main__":
    main()
