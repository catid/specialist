#!/usr/bin/env python3
"""Seal the fresh matched-init equal-weight v440 SFT control V53A."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import build_train_refresh_v440_v53a as refresh
import run_sft_train_only_control_v36a as engine
import run_sft_v440_equal_matched_init_v53a as launcher
import sft_lora_equal_unit_matched_init_v42a as source_contract
import sft_lora_equal_unit_matched_init_v42b as loader_contract
import sft_lora_equal_unit_matched_init_v47a as schedule


ROOT = Path(__file__).resolve().parent
RUN_DIR = (
    ROOT / "experiments/sft_controls/v53a_v440_equal_lr5p5e5"
).resolve()
OUTPUT_DIR = (RUN_DIR / "v440_equal_r32_seed17_init20260715041").resolve()
PREREGISTRATION = (RUN_DIR / "preregistration_v53a.json").resolve()
ARTIFACTS = {
    "output_dir": str(OUTPUT_DIR),
    "stdout_log": str((RUN_DIR / "stdout_v53a.log").resolve()),
    "gpu_log": str((RUN_DIR / "gpu_activity_v53a.jsonl").resolve()),
    "report": str((RUN_DIR / "runtime_report_v53a.json").resolve()),
    "attempt_report": str((RUN_DIR / "attempt_v53a.json").resolve()),
}


def argument_vector(
    preregistration_sha256: str = "PENDING",
    preregistration_content_sha256: str = "PENDING",
) -> list[str]:
    return [
        "--dataset", str(refresh.TRAIN),
        "--dataset-sha256", refresh.EXPECTED["train_sha256"],
        "--dataset-rows", "448",
        "--expected-conflict-units", "208",
        "--expected-weight-identity-sha256",
        launcher.EXPECTED_WEIGHTING_AUDIT["identity_sha256"],
        "--initial-adapter", str(source_contract.INITIAL_ADAPTER_V42A),
        "--initial-adapter-weights-sha256",
        source_contract.INITIAL_WEIGHTS_SHA256_V42A,
        "--initial-adapter-config-sha256",
        source_contract.INITIAL_CONFIG_SHA256_V42A,
        "--initial-adapter-manifest-sha256",
        source_contract.INITIAL_MANIFEST_SHA256_V42A,
        "--initial-adapter-manifest-content-sha256",
        source_contract.INITIAL_MANIFEST_CONTENT_SHA256_V42A,
        "--initial-adapter-tensor-identity-sha256",
        source_contract.INITIAL_TENSOR_IDENTITY_SHA256_V42A,
        "--output-dir", ARTIFACTS["output_dir"],
        "--stdout-log", ARTIFACTS["stdout_log"],
        "--gpu-log", ARTIFACTS["gpu_log"],
        "--report", ARTIFACTS["report"],
        "--attempt-report", ARTIFACTS["attempt_report"],
        "--preregistration", str(PREREGISTRATION),
        "--preregistration-sha256", preregistration_sha256,
        "--preregistration-content-sha256",
        preregistration_content_sha256,
        "--epochs", "3",
        "--max-steps", "48",
        "--rank", "32",
        "--lora-dropout", "0",
        "--grad-accum", "1",
        "--per-device-batch-size", "7",
        "--learning-rate", "5.5e-5",
        "--seed", "17",
        "--max-length", "1024",
        "--save-steps", "16",
        "--attn-implementation", "sdpa",
        "--prompt-mode", "es_exact",
        "--loss-mode", "example_mean",
        "--target-layers", "20,21,22,23",
        "--expected-trainable-elements", "4528128",
        "--expected-trainable-tensors", "70",
    ]


def arguments():
    return launcher.parser().parse_args(argument_vector())


def _recipe_identity(initialization: dict, loader: dict) -> str:
    return engine.canonical_sha256({
        "initialization_tensor_identity_sha256": initialization[
            "tensor_identity_sha256"
        ],
        "adapter_loader": loader,
        "world_size": 4,
        "physical_gpu_ids": [0, 1, 2, 3],
        "epochs_argument": 3.0,
        "per_device_batch_size": 7,
        "effective_global_batch_size": 28,
        "dataloader_drop_last": True,
        "optimizer_steps_per_dataloader_epoch": 16,
        "explicit_max_steps_cap": 48,
        "expected_optimizer_steps": 48,
        "row_equivalent_passes": 3.0,
        "learning_rate": 5.5e-5,
        "scheduler": "cosine",
        "warmup_ratio": 0.03,
        "training_seed": 17,
        "target_layers": [20, 21, 22, 23],
        "rank": 32,
        "lora_alpha": 64,
        "save_steps": 16,
        "max_length": 1024,
        "prompt_mode": "es_exact",
        "attn_implementation": "sdpa",
        "schedule_audit": schedule.schedule_audit_v47a(48),
    })


def build() -> dict:
    args = arguments()
    dataset = engine.validate_train_dataset(
        refresh.TRAIN, refresh.EXPECTED["train_sha256"], 448
    )
    manifest = launcher.validate_refresh_manifest_v53a()
    initialization = source_contract.validate_initialization_artifact_v42a(
        source_contract.INITIAL_ADAPTER_V42A
    )
    loader = loader_contract.expected_loader_audit_v42b()
    command = launcher.build_train_command(args)
    recipe_identity = _recipe_identity(initialization, loader)
    bindings = launcher.implementation_bindings_v53a()
    result = {
        "schema": "specialist-sft-v440-equal-train-only-preregistration-v53a",
        "status": "sealed_unlaunched_train_only",
        "experiment_name": "sft_v53a_v440_equal_fold3_lr5p5e5",
        "training_launch_authorized": True,
        "evaluation_launch_authorized": False,
        "contains_external_validation_ood_or_holdout_content": False,
        "dataset": dataset,
        "initialization": initialization,
        "adapter_loader": loader,
        "implementation": {
            "file_sha256_bindings": bindings,
            "launcher": str(Path(launcher.__file__).resolve()),
            "sft_script": str(launcher.SFT_SCRIPT),
            "refresh_builder": str(Path(refresh.__file__).resolve()),
            "refresh_manifest": str(refresh.MANIFEST),
        },
        "v440_fold_binding": {
            "projection_rows": refresh.EXPECTED["projection_rows"],
            "projection_sha256": refresh.EXPECTED["projection_sha256"],
            "train_rows": refresh.EXPECTED["train_rows"],
            "train_sha256": refresh.EXPECTED["train_sha256"],
            "train_conflict_units": refresh.EXPECTED["train_conflict_units"],
            "root_membership_sha256": refresh.EXPECTED[
                "root_membership_sha256"
            ],
            "manifest_path": str(refresh.MANIFEST),
            "manifest_file_sha256": launcher.MANIFEST_FILE_SHA256,
            "manifest_content_sha256": manifest[
                "content_sha256_before_self_field"
            ],
            "membership_exactly_frozen_v412_fold3_train": True,
            "fold_assignment_changes": 0,
        },
        "v49d_equal_recipe_parent": {
            "path": str(
                ROOT / "experiments/sft_controls/"
                "v49d_v434_sampling_midpoint_lr5p5e5/preregistration_v49d.json"
            ),
            "fresh_from_same_matched_initialization": True,
            "same_training_surface_schedule_and_step_budget": True,
            "changed_only": (
                "accepted train-side text cleanup from v434 through v440 and "
                "the corresponding deterministic equal-weight identity"
            ),
        },
        "artifacts": dict(ARTIFACTS),
        "recipe": {
            "command": command,
            "world_size": 4,
            "physical_gpu_ids": [0, 1, 2, 3],
            "all_four_gpu_activity_and_residency_required": True,
            "epochs_argument": 3.0,
            "per_device_batch_size": 7,
            "effective_global_batch_size": 28,
            "dataloader_drop_last": True,
            "optimizer_steps_per_dataloader_epoch": 16,
            "explicit_max_steps_cap": 48,
            "max_steps_is_terminal_authority": True,
            "expected_optimizer_steps": 48,
            "examples_emitted_per_dataloader_epoch": 448,
            "total_examples_emitted": 1344,
            "row_equivalent_passes": 3.0,
            "complete_all_row_passes": 3.0,
            "learning_rate": launcher.LEARNING_RATE,
            "scheduler": "cosine",
            "warmup_ratio": 0.03,
            "training_seed": 17,
            "initialization_seed": source_contract.INITIAL_SEED_V42A,
            "target_layers": [20, 21, 22, 23],
            "rank": 32,
            "lora_alpha": 64,
            "save_steps": 16,
            "expected_checkpoints": [16, 32, 48],
            "max_length": 1024,
            "expected_encoding_audit": launcher.EXPECTED_ENCODING_AUDIT,
            "expected_weighting_audit": launcher.EXPECTED_WEIGHTING_AUDIT,
            "expected_schedule_audit": schedule.schedule_audit_v47a(48),
            "expected_trainable_inventory": engine.EXPECTED_TRAINABLE_INVENTORY,
            "prompt_mode": "es_exact",
            "loss": "equal-conflict-unit answer-token mean",
            "attn_implementation": "sdpa",
            "common_v49d_equal_recipe_identity_sha256": recipe_identity,
        },
        "access_firewall": {
            "training_input": "exact v440 frozen fold-3 train roots only",
            "shadow_artifact_opened": False,
            "eval_ood_holdout_opened": False,
            "external_metric_or_protected_result_used": False,
            "post_training_evaluation_authorized": False,
        },
        "selection_firewall": {
            "this_run_authorizes": "one train-only fresh v440 SFT state",
            "shadow_ood_holdout_feedback_authorized": False,
            "recipe_change_after_seal_authorized": False,
            "promotion_authorized": False,
        },
    }
    result["content_sha256_before_self_field"] = engine.canonical_sha256(result)
    return result


def launch_command(file_sha256: str, content_sha256: str) -> list[str]:
    return [
        str(ROOT / ".venv/bin/python"),
        str(Path(launcher.__file__).resolve()),
        *argument_vector(file_sha256, content_sha256),
    ]


def main() -> None:
    result = build()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if PREREGISTRATION.exists():
        raise FileExistsError("V53A preregistration already exists")
    engine.atomic_write_json(PREREGISTRATION, result)
    file_sha = engine.file_sha256(PREREGISTRATION)
    content_sha = result["content_sha256_before_self_field"]
    print(json.dumps({
        "path": str(PREREGISTRATION),
        "file_sha256": file_sha,
        "content_sha256": content_sha,
        "launch_command": shlex.join(launch_command(file_sha, content_sha)),
        "gpu_accessed": False,
        "training_launched": False,
        "protected_inputs_opened": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
