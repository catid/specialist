#!/usr/bin/env python3
"""Freeze the confirmatory fold-3 equal-unit sparse-SFT arm."""

from __future__ import annotations

import json
from pathlib import Path

import build_train_shadow_folds_v37a as shadow
import run_sft_equal_unit_v37a as launcher
import run_sft_train_only_control_v36a as engine


ROOT = Path(__file__).resolve().parent
RUN_DIR = (ROOT / "experiments/sft_controls/v37a_equal_unit_fold3_v412").resolve()
DATASET = (shadow.OUTPUT_DIR / "fold_3_train.jsonl").resolve()
MANIFEST = shadow.MANIFEST
LAYER_PLAN = (ROOT / "experiments/layer_plans/middle_late_dense_v6.json").resolve()
OUTPUT_DIR = RUN_DIR / "middle_late_r32_seed17"
PREREGISTRATION = RUN_DIR / "preregistration_v37a.json"
EXPECTED = {
    "dataset_sha256": "97fc920ac39f67536df26977de951e8c34bf8486eb8f42fbb0a67687f025a92a",
    "dataset_rows": 448,
    "dataset_documents": 234,
    "conflict_units": 208,
    "weight_identity_sha256": "631199dc13d240434f7b0a9ea94c0848c315d83b12fada0be3a7189e57a85b06",
    "manifest_file_sha256": "7d2a8f2b86f9007aa2bfe8ae043be15647451cc4bbea53a18d5915085879ee9d",
    "manifest_content_sha256": "3fcc2820e8dffe6a21198d0520365aace049735ac84bda179ea44bc8ad0881eb",
    "builder_sha256": "29fb6e94eb7bca470d5d2ef32dd9bc74a755eaa0624a14a74a1a22ded070a6a8",
    "launcher_sha256": "73fe9e0402f4e8515ade143e6528d1ae5c0f6b0d8feed1ef7101af6f3df58ead",
    "engine_sha256": "e9823359d82aeb61d7fd1f17f24ea22fc46680cc122c55232b1913d765d1d49d",
    "sft_sha256": "f6f38312e5e4baf19923ac4a52ecb30813bfd5c8be4f561a6d9c38076a0159b4",
    "model_config_sha256": "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
    "model_index_sha256": "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
    "layer_plan_sha256": "d65d702969dcec7a56ca4fcf461d402c44642966191a57c2ef092ec339e3e3df",
}


def arguments() -> object:
    return launcher.parser().parse_args([
        "--dataset", str(DATASET),
        "--dataset-sha256", EXPECTED["dataset_sha256"],
        "--dataset-rows", str(EXPECTED["dataset_rows"]),
        "--expected-conflict-units", str(EXPECTED["conflict_units"]),
        "--expected-weight-identity-sha256", EXPECTED["weight_identity_sha256"],
        "--output-dir", str(OUTPUT_DIR),
        "--stdout-log", str(RUN_DIR / "stdout_v37a.log"),
        "--gpu-log", str(RUN_DIR / "gpu_activity_v37a.jsonl"),
        "--report", str(RUN_DIR / "runtime_report_v37a.json"),
        "--attempt-report", str(RUN_DIR / "attempt_v37a.json"),
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
        "--save-steps", "16",
        "--attn-implementation", "sdpa",
        "--prompt-mode", "es_exact",
        "--loss-mode", "example_mean",
        "--target-layers", "20,21,22,23",
        "--expected-trainable-elements", "4528128",
        "--expected-trainable-tensors", "70",
    ])


def build() -> dict:
    args = arguments()
    dataset = engine.validate_train_dataset(
        DATASET, EXPECTED["dataset_sha256"], EXPECTED["dataset_rows"],
    )
    observed = {
        "dataset_documents": dataset["unique_documents"],
        "manifest_file_sha256": engine.file_sha256(MANIFEST),
        "builder_sha256": engine.file_sha256(ROOT / "build_train_shadow_folds_v37a.py"),
        "launcher_sha256": engine.file_sha256(ROOT / "run_sft_equal_unit_v37a.py"),
        "engine_sha256": engine.file_sha256(
            ROOT / "run_sft_train_only_control_v36a.py"
        ),
        "sft_sha256": engine.file_sha256(ROOT / "sft_lora_equal_unit_v37a.py"),
        "model_config_sha256": engine.file_sha256(engine.BASE_MODEL / "config.json"),
        "model_index_sha256": engine.file_sha256(
            engine.BASE_MODEL / "model.safetensors.index.json"
        ),
        "layer_plan_sha256": engine.file_sha256(LAYER_PLAN),
    }
    for key, value in observed.items():
        if value != EXPECTED[key]:
            raise RuntimeError(f"v37a preregistration binding changed: {key}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest["content_sha256_before_self_field"]
        != EXPECTED["manifest_content_sha256"]
        or manifest["selection_firewall"]["confirmatory_fold"] != 3
    ):
        raise RuntimeError("v37a split-manifest contract changed")
    fold = manifest["folds"][3]
    if (
        fold["train"]["sha256"] != EXPECTED["dataset_sha256"]
        or fold["train"]["rows"] != 448
        or fold["train"]["conflict_units"] != 208
        or fold["shadow_dev"]["rows"] != 83
        or fold["shadow_dev"]["conflict_units"] != 51
        or any(fold["train_dev_edge_identity_intersections"].values())
    ):
        raise RuntimeError("v37a confirmatory fold aggregate changed")
    command = launcher.build_train_command(args)
    result = {
        "schema": "specialist-sft-equal-unit-preregistration-v37a",
        "status": "preregistered_not_yet_run",
        "experiment_name": "sft_equal_unit_v37a_fold3_v412_middle_late_r32_seed17",
        "contains_external_validation_ood_or_holdout_content": False,
        "dataset": dataset,
        "implementation": {
            "split_builder": str((ROOT / "build_train_shadow_folds_v37a.py").resolve()),
            "split_builder_sha256": EXPECTED["builder_sha256"],
            "launcher": str((ROOT / "run_sft_equal_unit_v37a.py").resolve()),
            "launcher_sha256": EXPECTED["launcher_sha256"],
            "engine": str((ROOT / "run_sft_train_only_control_v36a.py").resolve()),
            "engine_sha256": EXPECTED["engine_sha256"],
            "sft": str((ROOT / "sft_lora_equal_unit_v37a.py").resolve()),
            "sft_sha256": EXPECTED["sft_sha256"],
        },
        "model": {
            "path": str(engine.BASE_MODEL),
            "config_sha256": EXPECTED["model_config_sha256"],
            "index_sha256": EXPECTED["model_index_sha256"],
        },
        "artifacts": {
            "output_dir": str(OUTPUT_DIR),
            "stdout_log": str(RUN_DIR / "stdout_v37a.log"),
            "gpu_log": str(RUN_DIR / "gpu_activity_v37a.jsonl"),
            "report": str(RUN_DIR / "runtime_report_v37a.json"),
            "attempt_report": str(RUN_DIR / "attempt_v37a.json"),
        },
        "comparison_binding": {
            "split_manifest": str(MANIFEST),
            "split_manifest_file_sha256": EXPECTED["manifest_file_sha256"],
            "split_manifest_content_sha256": EXPECTED["manifest_content_sha256"],
            "confirmatory_fold": 3,
            "shadow_dev_rows": 83,
            "shadow_dev_conflict_units": 51,
            "shadow_dev_opened_during_preregistration": False,
            "eggroll_es_layer_plan": str(LAYER_PLAN),
            "eggroll_es_layer_plan_sha256": EXPECTED["layer_plan_sha256"],
            "eggroll_es_layers": [20, 21, 22, 23],
            "eggroll_es_dense_tensor_count": 35,
            "eggroll_es_dense_elements": 142_999_552,
            "sft_trainable_tensors": 70,
            "sft_trainable_elements": 4_528_128,
        },
        "recipe": {
            "world_size": 4,
            "physical_gpu_ids": [0, 1, 2, 3],
            "epochs": 3.0,
            "per_device_batch_size": 7,
            "effective_global_batch_size": 28,
            "rows_per_epoch": 448,
            "distributed_padding_rows_per_epoch": 0,
            "optimizer_steps_per_epoch": 16,
            "expected_optimizer_steps": 48,
            "learning_rate": 1e-4,
            "scheduler": "cosine",
            "warmup_ratio": 0.03,
            "seed": 17,
            "bf16": True,
            "gradient_checkpointing": True,
            "attention": "sdpa",
            "prompt": "exact EGGROLL-ES specialist_template(question)+raw_answer",
            "eos_appended_or_scored": False,
            "loss": (
                "mean answer-token NLL per row multiplied by "
                "448/(208*unit_row_count), then ordinary global-batch mean"
            ),
            "renormalize_weights_within_batch": False,
            "conflict_units": 208,
            "weight_identity_sha256": EXPECTED["weight_identity_sha256"],
            "expected_encoding_audit": launcher.EXPECTED_ENCODING_AUDIT,
            "expected_weighting_audit": launcher.EXPECTED_WEIGHTING_AUDIT,
            "expected_trainable_inventory": engine.EXPECTED_TRAINABLE_INVENTORY,
            "command": command,
        },
        "selection_firewall": {
            "shadow_dev_access": (
                "exactly once only after this SFT state and the nonzero ES state "
                "are both sealed"
            ),
            "forbidden_before_state_seal": [
                "fold-3 shadow-dev score", "fold 0/1/2/4 optimization",
                "external validation", "OOD", "holdout", "model promotion",
            ],
            "this_arm_authorizes": "training state and runtime evidence only",
        },
    }
    result["content_sha256_before_self_field"] = engine.canonical_sha256(result)
    return result


def main() -> None:
    result = build()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if PREREGISTRATION.exists():
        raise RuntimeError("v37a preregistration already exists")
    engine.atomic_write_json(PREREGISTRATION, result)
    print(PREREGISTRATION)
    print(result["content_sha256_before_self_field"])


if __name__ == "__main__":
    main()
