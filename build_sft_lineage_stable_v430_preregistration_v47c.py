#!/usr/bin/env python3
"""Seal the unlaunched lineage-stable v430 SFT control V47C."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import build_train_refresh_lineage_stable_v430_v47c as refresh
import run_sft_equal_unit_matched_init_v47c as launcher
import run_sft_train_only_control_v36a as engine
import sft_lora_equal_unit_matched_init_v42a as source_contract
import sft_lora_equal_unit_matched_init_v42b as sft_v42b
import sft_lora_equal_unit_matched_init_v47a as sft


ROOT = Path(__file__).resolve().parent
RUN_DIR = (
    ROOT / "experiments/sft_controls/"
    "v47c_lineage_stable_v430_fold3_lr5p5e5"
).resolve()
OUTPUT_DIR = RUN_DIR / "middle_late_r32_seed17_init20260715041"
PREREGISTRATION = RUN_DIR / "preregistration_v47c.json"
MANIFEST_FILE_SHA256 = "1cc42172c4a7ad32027aee8320803974a5ab1d6bd8153ae6a43df14c19ab05cc"


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
        "--output-dir", str(OUTPUT_DIR),
        "--stdout-log", str(RUN_DIR / "stdout_v47c.log"),
        "--gpu-log", str(RUN_DIR / "gpu_activity_v47c.jsonl"),
        "--report", str(RUN_DIR / "runtime_report_v47c.json"),
        "--attempt-report", str(RUN_DIR / "attempt_v47c.json"),
        "--preregistration", str(PREREGISTRATION),
        "--preregistration-sha256", preregistration_sha256,
        "--preregistration-content-sha256", preregistration_content_sha256,
        "--epochs", "3", "--max-steps", "48", "--rank", "32",
        "--lora-dropout", "0", "--grad-accum", "1",
        "--per-device-batch-size", "7", "--learning-rate", "5.5e-5",
        "--seed", "17", "--max-length", "1024", "--save-steps", "16",
        "--attn-implementation", "sdpa", "--prompt-mode", "es_exact",
        "--loss-mode", "example_mean", "--target-layers", "20,21,22,23",
        "--expected-trainable-elements", "4528128",
        "--expected-trainable-tensors", "70",
    ]


def arguments():
    return launcher.parser().parse_args(argument_vector())


def build() -> dict:
    args = arguments()
    dataset = engine.validate_train_dataset(
        refresh.TRAIN, refresh.EXPECTED["train_sha256"], 448
    )
    manifest = json.loads(refresh.MANIFEST.read_text(encoding="utf-8"))
    manifest_content = manifest.get("content_sha256_before_self_field")
    proof = manifest["lineage_stability_proof"]
    if (
        engine.file_sha256(refresh.MANIFEST) != MANIFEST_FILE_SHA256
        or manifest_content != refresh.EXPECTED["manifest_content_sha256"]
        or manifest_content != engine.canonical_sha256({
            key: item for key, item in manifest.items()
            if key != "content_sha256_before_self_field"
        })
        or manifest["fold"]["train"]["sha256"]
        != refresh.EXPECTED["train_sha256"]
        or any(manifest["fold"]["train_dev_edge_identity_intersections"].values())
        or proof["fold_assignment_changes"] != 0
        or proof["root_membership_sets_identical"] is not True
        or proof["unit_row_multiplicities_identical"] is not True
    ):
        raise RuntimeError("V47C lineage-stable manifest changed")
    initialization = source_contract.validate_initialization_artifact_v42a(
        source_contract.INITIAL_ADAPTER_V42A
    )
    loader = sft_v42b.expected_loader_audit_v42b()
    command = launcher.build_train_command(args)
    bindings = launcher.implementation_bindings()
    result = {
        "schema": "specialist-sft-lineage-stable-v430-preregistration-v47c",
        "status": "sealed_unlaunched_holdout_blind",
        "experiment_name": "sft_v47c_lineage_stable_v430_fold3_lr5p5e5",
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
        "v42i_recipe_parent": {
            "path": str(launcher.V42I_PREREGISTRATION),
            "file_sha256": bindings["v42i_preregistration"],
            "content_sha256": "22a63bc8b5da313a3cce3fae48a7fd626efd65b5869a41ff8329ec4534d0280b",
            "inherited_exactly": [
                "sealed initialization and direct canonical loader",
                "rank 32 alpha 64 LoRA surface on layers 20,21,22,23",
                "learning rate 5.5e-5 and cosine schedule with 0.03 warmup ratio",
                "448 rows, global batch 28, three complete passes, 48 steps",
                "seed 17, max length 1024, save every 16 steps",
                "bf16 base, FP32 trainable adapter, SDPA, gradient checkpointing",
                "equal-conflict-unit answer-token objective and es_exact prompt",
            ],
            "changed_only": (
                "accepted v413-v430 text edits within the exact original v412 "
                "root-membership fold assignment; explicit max_steps=48 restates "
                "the realized V42I optimizer budget"
            ),
        },
        "artifacts": {
            "output_dir": str(OUTPUT_DIR),
            "stdout_log": str(RUN_DIR / "stdout_v47c.log"),
            "gpu_log": str(RUN_DIR / "gpu_activity_v47c.jsonl"),
            "report": str(RUN_DIR / "runtime_report_v47c.json"),
            "attempt_report": str(RUN_DIR / "attempt_v47c.json"),
        },
        "fold_binding": {
            "projection_sha256": refresh.EXPECTED["projection_sha256"],
            "manifest_path": str(refresh.MANIFEST),
            "manifest_file_sha256": bindings["refresh_manifest"],
            "manifest_content_sha256": manifest_content,
            "confirmatory_fold": 3,
            "train_rows": 448, "train_conflict_units": 208,
            "shadow_rows_aggregate_only": 83,
            "shadow_conflict_units_aggregate_only": 51,
            "original_root_membership_assignment_retained": True,
            "unit_membership_changes": 0, "unit_multiplicity_changes": 0,
            "fold_assignment_changes": 0,
            "all_edge_identity_intersections": 0,
        },
        "recipe": {
            "command": command,
            "world_size": 4, "physical_gpu_ids": [0, 1, 2, 3],
            "all_four_gpu_activity_and_residency_required": True,
            "epochs_argument": 3.0,
            "per_device_batch_size": 7, "effective_global_batch_size": 28,
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
            "scheduler": "cosine", "warmup_ratio": 0.03,
            "training_seed": 17,
            "initialization_seed": source_contract.INITIAL_SEED_V42A,
            "target_layers": [20, 21, 22, 23], "rank": 32,
            "lora_alpha": 64, "save_steps": 16,
            "expected_checkpoints": [16, 32, 48],
            "expected_encoding_audit": launcher.EXPECTED_ENCODING_AUDIT,
            "expected_weighting_audit": launcher.EXPECTED_WEIGHTING_AUDIT,
            "expected_schedule_audit": sft.schedule_audit_v47a(48),
            "expected_trainable_inventory": engine.EXPECTED_TRAINABLE_INVENTORY,
            "prompt_mode": "es_exact",
            "loss": "equal-conflict-unit answer-token mean",
            "attn_implementation": "sdpa",
        },
        "access_firewall": {
            "training_input": "lineage-stable refreshed fold-3 train file only",
            "shadow_dev_opened_during_preregistration": False,
            "shadow_dev_opened_during_training": False,
            "eval_ood_holdout_opened": False,
            "external_metric_or_protected_result_used": False,
            "post_training_evaluation_authorized": False,
        },
        "selection_firewall": {
            "this_run_authorizes": "one train-only V47C state and runtime evidence",
            "shadow_ood_holdout_feedback_authorized": False,
            "recipe_change_after_seal_authorized": False,
        },
    }
    result["content_sha256_before_self_field"] = engine.canonical_sha256(result)
    return result


def launch_command(file_sha256: str, content_sha256: str) -> list[str]:
    return [
        str(ROOT / ".venv/bin/python"), str(Path(launcher.__file__).resolve()),
        *argument_vector(file_sha256, content_sha256),
    ]


def main() -> None:
    result = build()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if PREREGISTRATION.exists():
        raise RuntimeError("V47C preregistration already exists")
    engine.atomic_write_json(PREREGISTRATION, result)
    file_sha256 = engine.file_sha256(PREREGISTRATION)
    content_sha256 = result["content_sha256_before_self_field"]
    print(json.dumps({
        "path": str(PREREGISTRATION), "file_sha256": file_sha256,
        "content_sha256": content_sha256,
        "launch_command": shlex.join(launch_command(file_sha256, content_sha256)),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
