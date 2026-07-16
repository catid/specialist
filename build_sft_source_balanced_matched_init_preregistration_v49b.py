#!/usr/bin/env python3
"""Seal the matched V49B source-balanced SFT arm."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import run_sft_source_balanced_matched_init_v49b as launcher
import run_sft_train_only_control_v36a as engine
import seal_sft_source_balanced_input_v49b as sealed_input
import sft_lora_equal_unit_matched_init_v42a as source_contract
import sft_lora_equal_unit_matched_init_v42b as sft_v42b
import sft_lora_equal_unit_matched_init_v47a as schedule


ROOT = Path(__file__).resolve().parent
RUN_DIR = sealed_input.RUN_DIR
OUTPUT_DIR = RUN_DIR / "middle_late_r32_seed17_init20260715041_retry1"
PREREGISTRATION = RUN_DIR / "preregistration_v49b_retry1.json"
V47C_PREREGISTRATION = (
    ROOT / "experiments/sft_controls/"
    "v47c_lineage_stable_v430_fold3_lr5p5e5/preregistration_v47c.json"
).resolve()
V47C_PREREGISTRATION_FILE_SHA256 = (
    "8b5abeb9530851e16dbaa9e48750a19c7e7e7f1aae51af1d5034f947e0bb31c0"
)
V47C_PREREGISTRATION_CONTENT_SHA256 = (
    "8b224522e208f4e01f426b3b261795754dd4048806d290c61a5535a535896eed"
)
V42I_PREREGISTRATION_FILE_SHA256 = (
    "54e21c50a9a2601054f69765835be3eb15757697f6779c0b806b40e341accf9e"
)
V42I_PREREGISTRATION_CONTENT_SHA256 = (
    "22a63bc8b5da313a3cce3fae48a7fd626efd65b5869a41ff8329ec4534d0280b"
)


def argument_vector(
    preregistration_sha256: str = "PENDING",
    preregistration_content_sha256: str = "PENDING",
) -> list[str]:
    return [
        "--dataset", str(sealed_input.TRAIN),
        "--dataset-sha256", launcher.weighting.v49a.V434_TRAIN_SHA256,
        "--dataset-rows", "448",
        "--expected-conflict-units", "208",
        "--expected-weight-identity-sha256",
        launcher.weighting.ALTERNATIVE_NORMALIZED_WEIGHT_SHA256,
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
        "--stdout-log", str(RUN_DIR / "stdout_v49b_retry1.log"),
        "--gpu-log", str(RUN_DIR / "gpu_activity_v49b_retry1.jsonl"),
        "--report", str(RUN_DIR / "runtime_report_v49b_retry1.json"),
        "--attempt-report", str(RUN_DIR / "attempt_v49b_retry1.json"),
        "--preregistration", str(PREREGISTRATION),
        "--preregistration-sha256", preregistration_sha256,
        "--preregistration-content-sha256",
        preregistration_content_sha256,
        "--epochs", "3", "--max-steps", "48", "--rank", "32",
        "--lora-dropout", "0", "--grad-accum", "1",
        "--per-device-batch-size", "7",
        "--learning-rate", "5.5e-5",
        "--seed", "17", "--max-length", "1024",
        "--save-steps", "16", "--attn-implementation", "sdpa",
        "--prompt-mode", "es_exact", "--loss-mode", "example_mean",
        "--target-layers", "20,21,22,23",
        "--expected-trainable-elements", "4528128",
        "--expected-trainable-tensors", "70",
    ]


def arguments():
    return launcher.parser().parse_args(argument_vector())


def _validate_parent(path: Path, file_sha: str, content_sha: str) -> dict:
    if engine.file_sha256(path) != file_sha:
        raise RuntimeError("V49B matched SFT parent file changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("content_sha256_before_self_field") != content_sha
        or content_sha != engine.canonical_sha256({
            key: item for key, item in value.items()
            if key != "content_sha256_before_self_field"
        })
        or value.get("contains_external_validation_ood_or_holdout_content")
        is not False
    ):
        raise RuntimeError("V49B matched SFT parent content changed")
    return value


def build() -> dict:
    args = arguments()
    dataset = engine.validate_train_dataset(
        sealed_input.TRAIN,
        launcher.weighting.v49a.V434_TRAIN_SHA256,
        448,
    )
    input_manifest = launcher.validate_input_manifest_v49b()
    complete_audit, compact_audit = launcher.load_weighting_audit_v49b()
    v47c_parent = _validate_parent(
        V47C_PREREGISTRATION,
        V47C_PREREGISTRATION_FILE_SHA256,
        V47C_PREREGISTRATION_CONTENT_SHA256,
    )
    _validate_parent(
        launcher.v47c.V42I_PREREGISTRATION,
        V42I_PREREGISTRATION_FILE_SHA256,
        V42I_PREREGISTRATION_CONTENT_SHA256,
    )
    initialization = source_contract.validate_initialization_artifact_v42a(
        source_contract.INITIAL_ADAPTER_V42A
    )
    loader = sft_v42b.expected_loader_audit_v42b()
    command = launcher.build_train_command(args)
    bindings = launcher.implementation_bindings_v49b()
    result = {
        "schema": "specialist-sft-source-balanced-preregistration-v49b",
        "status": "sealed_unlaunched_train_only",
        "experiment_name": "sft_v49b_source_balanced_v434_fold3_lr5p5e5_retry1",
        "contains_external_validation_ood_or_holdout_content": False,
        "dataset": dataset,
        "input_manifest": {
            "path": str(sealed_input.INPUT_MANIFEST),
            "file_sha256": launcher.INPUT_MANIFEST_FILE_SHA256,
            "content_sha256": launcher.INPUT_MANIFEST_CONTENT_SHA256,
            "document_disjoint_membership": input_manifest[
                "document_disjoint_membership"
            ],
        },
        "weighting_audit": {
            "path": str(sealed_input.WEIGHT_AUDIT),
            "file_sha256": launcher.WEIGHT_AUDIT_FILE_SHA256,
            "content_sha256": launcher.WEIGHT_AUDIT_CONTENT_SHA256,
            "runtime_compact": compact_audit,
            "per_row_records": len(complete_audit["per_row"]),
            "per_source_records": len(complete_audit["per_source"]),
            "per_category_records": len(complete_audit["per_category"]),
        },
        "initialization": initialization,
        "adapter_loader": loader,
        "implementation": {
            "file_sha256_bindings": bindings,
            "launcher": str(Path(launcher.__file__).resolve()),
            "sft_script": str(launcher.SFT_SCRIPT),
            "weighting_runtime": str(Path(launcher.weighting.__file__).resolve()),
            "input_sealer": str(Path(sealed_input.__file__).resolve()),
        },
        "matched_recipe_parents": {
            "v47c": {
                "path": str(V47C_PREREGISTRATION),
                "file_sha256": V47C_PREREGISTRATION_FILE_SHA256,
                "content_sha256": V47C_PREREGISTRATION_CONTENT_SHA256,
                "runtime_schedule_surface_inherited": True,
            },
            "v42i": {
                "path": str(launcher.v47c.V42I_PREREGISTRATION),
                "file_sha256": V42I_PREREGISTRATION_FILE_SHA256,
                "content_sha256": V42I_PREREGISTRATION_CONTENT_SHA256,
                "initialization_lora_lr_and_optimizer_budget_inherited": True,
            },
            "v49a": {
                "path": str(launcher.weighting.V49A_DESIGN),
                "file_sha256": launcher.weighting.V49A_DESIGN_FILE_SHA256,
                "content_sha256": launcher.weighting.V49A_DESIGN_CONTENT_SHA256,
                "alternative_weight_identity_inherited_exactly": True,
            },
        },
        "artifacts": {
            "output_dir": str(OUTPUT_DIR),
            "stdout_log": str(RUN_DIR / "stdout_v49b_retry1.log"),
            "gpu_log": str(RUN_DIR / "gpu_activity_v49b_retry1.jsonl"),
            "report": str(RUN_DIR / "runtime_report_v49b_retry1.json"),
            "attempt_report": str(RUN_DIR / "attempt_v49b_retry1.json"),
        },
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
            "expected_weighting_audit": compact_audit,
            "expected_schedule_audit": schedule.schedule_audit_v47a(48),
            "expected_trainable_inventory": engine.EXPECTED_TRAINABLE_INVENTORY,
            "prompt_mode": "es_exact",
            "loss": "V49A weighted answer-token mean per example",
            "attn_implementation": "sdpa",
            "only_change_from_matched_parent": (
                "per-row example weights use the exact V49A alternative identity"
            ),
        },
        "data_refresh_binding": {
            "projection": "accepted train-only edits replayed through v434",
            "train_jsonl_sha256": launcher.weighting.v49a.V434_TRAIN_SHA256,
            "root_membership_sha256": launcher.weighting.ROOT_MEMBERSHIP_SHA256,
            "rows_added_or_dropped": 0,
            "fold_membership_changed": False,
            "document_disjoint_commitment_changed": False,
        },
        "access_firewall": {
            "training_input": (
                "sealed v434 projection of frozen v412 fold-3 train roots only"
            ),
            "shadow_semantics_opened_during_preregistration": False,
            "shadow_semantics_opened_during_training": False,
            "eval_ood_holdout_opened": False,
            "external_metric_or_protected_result_used": False,
            "post_training_evaluation_authorized": False,
        },
        "selection_firewall": {
            "this_run_authorizes": "one train-only V49B state and runtime evidence",
            "shadow_ood_holdout_feedback_authorized": False,
            "recipe_change_after_seal_authorized": False,
            "promotion_authorized": False,
        },
    }
    # Explicitly compare invariant controls to the committed matched parent.
    parent_recipe = v47c_parent["recipe"]
    invariants = {
        "world_size": 4,
        "per_device_batch_size": 7,
        "effective_global_batch_size": 28,
        "explicit_max_steps_cap": 48,
        "expected_optimizer_steps": 48,
        "learning_rate": 5.5e-5,
        "target_layers": [20, 21, 22, 23],
        "rank": 32,
        "lora_alpha": 64,
        "save_steps": 16,
        "prompt_mode": "es_exact",
        "attn_implementation": "sdpa",
    }
    if any(parent_recipe.get(key) != value for key, value in invariants.items()):
        raise RuntimeError("V49B V47C/V42I invariant recipe changed")
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
    if PREREGISTRATION.exists():
        raise FileExistsError("V49B preregistration already exists")
    engine.atomic_write_json(PREREGISTRATION, result)
    file_sha = engine.file_sha256(PREREGISTRATION)
    content_sha = result["content_sha256_before_self_field"]
    print(json.dumps({
        "path": str(PREREGISTRATION),
        "file_sha256": file_sha,
        "content_sha256": content_sha,
        "launch_command": shlex.join(launch_command(file_sha, content_sha)),
        "gpu_accessed": False,
        "non_train_semantics_opened": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
