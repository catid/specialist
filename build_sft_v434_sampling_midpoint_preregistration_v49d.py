#!/usr/bin/env python3
"""Seal both fresh, content-matched V49D SFT training arms."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import run_sft_train_only_control_v36a as engine
import run_sft_v434_sampling_midpoint_matched_init_v49d as launcher
import seal_sft_v434_sampling_midpoint_input_v49d as sealed_input
import sft_lora_equal_unit_matched_init_v42a as source_contract
import sft_lora_equal_unit_matched_init_v42b as sft_v42b
import sft_lora_equal_unit_matched_init_v47a as schedule


ROOT = Path(__file__).resolve().parent
RUN_DIR = sealed_input.RUN_DIR
PREREGISTRATION = (RUN_DIR / "preregistration_v49d.json").resolve()
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
ARTIFACTS = {
    "v434_equal": {
        "output_dir": str((RUN_DIR / "v434_equal_r32_seed17_init20260715041").resolve()),
        "stdout_log": str((RUN_DIR / "stdout_v434_equal.log").resolve()),
        "gpu_log": str((RUN_DIR / "gpu_activity_v434_equal.jsonl").resolve()),
        "report": str((RUN_DIR / "runtime_report_v434_equal.json").resolve()),
        "attempt_report": str((RUN_DIR / "attempt_v434_equal.json").resolve()),
    },
    "v434_source50": {
        "output_dir": str((RUN_DIR / "v434_source50_r32_seed17_init20260715041").resolve()),
        "stdout_log": str((RUN_DIR / "stdout_v434_source50.log").resolve()),
        "gpu_log": str((RUN_DIR / "gpu_activity_v434_source50.jsonl").resolve()),
        "report": str((RUN_DIR / "runtime_report_v434_source50.json").resolve()),
        "attempt_report": str((RUN_DIR / "attempt_v434_source50.json").resolve()),
    },
}


def argument_vector(
    arm: str,
    preregistration_sha256: str = "PENDING",
    preregistration_content_sha256: str = "PENDING",
) -> list[str]:
    if arm not in launcher.ARMS:
        raise ValueError(arm)
    artifacts = ARTIFACTS[arm]
    return [
        "--arm", arm,
        "--dataset", str(sealed_input.TRAIN),
        "--dataset-sha256", launcher.weighting.v49b.v49a.V434_TRAIN_SHA256,
        "--dataset-rows", "448",
        "--expected-conflict-units", "208",
        "--expected-weight-identity-sha256",
        launcher.weighting.EXPECTED[arm]["normalized_weight_sha256"],
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
        "--output-dir", artifacts["output_dir"],
        "--stdout-log", artifacts["stdout_log"],
        "--gpu-log", artifacts["gpu_log"],
        "--report", artifacts["report"],
        "--attempt-report", artifacts["attempt_report"],
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


def arguments(arm: str):
    return launcher.parser().parse_args(argument_vector(arm))


def _validate_v47c_parent() -> dict:
    if engine.file_sha256(V47C_PREREGISTRATION) != V47C_PREREGISTRATION_FILE_SHA256:
        raise RuntimeError("V49D matched V47C parent file changed")
    value = json.loads(V47C_PREREGISTRATION.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    if (
        content != V47C_PREREGISTRATION_CONTENT_SHA256
        or content != engine.canonical_sha256({
            key: item for key, item in value.items()
            if key != "content_sha256_before_self_field"
        })
        or value.get("contains_external_validation_ood_or_holdout_content")
        is not False
    ):
        raise RuntimeError("V49D matched V47C parent content changed")
    return value


def _common_recipe_identity(initialization: dict, loader: dict) -> str:
    return engine.canonical_sha256({
        "dataset_sha256": launcher.weighting.v49b.v49a.V434_TRAIN_SHA256,
        "dataset_rows": 448,
        "conflict_units": 208,
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
        "encoding_audit": launcher.EXPECTED_ENCODING_AUDIT,
        "schedule_audit": schedule.schedule_audit_v47a(48),
    })


def build() -> dict:
    dataset = engine.validate_train_dataset(
        sealed_input.TRAIN,
        launcher.weighting.v49b.v49a.V434_TRAIN_SHA256,
        448,
    )
    input_manifest = launcher.validate_input_manifest_v49d()
    parent = _validate_v47c_parent()
    initialization = source_contract.validate_initialization_artifact_v42a(
        source_contract.INITIAL_ADAPTER_V42A
    )
    loader = sft_v42b.expected_loader_audit_v42b()
    recipe_identity = _common_recipe_identity(initialization, loader)
    bindings = launcher.implementation_bindings_v49d()
    arms = {}
    for arm_name in launcher.ARMS:
        args = arguments(arm_name)
        complete, compact = launcher.load_weighting_audit_v49d(arm_name)
        command = launcher.build_train_command(args)
        arms[arm_name] = {
            "weighting_audit": {
                "path": str(sealed_input.WEIGHT_AUDITS[arm_name]),
                "file_sha256": launcher.WEIGHT_AUDIT_HASHES[arm_name]["file"],
                "content_sha256": launcher.WEIGHT_AUDIT_HASHES[arm_name]["content"],
                "runtime_compact": compact,
                "per_row_records": len(complete["per_row"]),
                "per_source_records": len(complete["per_source"]),
                "per_category_records": len(complete["per_category"]),
            },
            "artifacts": dict(ARTIFACTS[arm_name]),
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
                "expected_weighting_audit": compact,
                "expected_schedule_audit": schedule.schedule_audit_v47a(48),
                "expected_trainable_inventory": engine.EXPECTED_TRAINABLE_INVENTORY,
                "prompt_mode": "es_exact",
                "loss": "weighted answer-token mean per example",
                "attn_implementation": "sdpa",
                "common_recipe_identity_sha256": recipe_identity,
            },
        }
    result = {
        "schema": "specialist-v434-sampling-midpoint-preregistration-v49d",
        "status": "sealed_unlaunched_train_only",
        "experiment_name": "sft_v49d_v434_equal_vs_source50_lr5p5e5",
        "training_launch_authorized": True,
        "evaluation_launch_authorized": False,
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
        "initialization": initialization,
        "adapter_loader": loader,
        "implementation": {
            "file_sha256_bindings": bindings,
            "launcher": str(Path(launcher.__file__).resolve()),
            "weighting_runtime": str(Path(launcher.weighting.__file__).resolve()),
            "input_sealer": str(Path(sealed_input.__file__).resolve()),
        },
        "training_arm_order": list(launcher.ARMS),
        "training_arms": arms,
        "matched_control_contract": {
            "recipe_identity_sha256": recipe_identity,
            "same_exact_v434_training_bytes": True,
            "same_initial_adapter_bytes_and_tensor_identity": True,
            "same_lora_surface_rank_alpha_and_dropout": True,
            "same_learning_rate_optimizer_scheduler_and_warmup": True,
            "same_seed_prompt_encoding_batches_and_dataloader_order": True,
            "same_48_steps_and_three_complete_row_passes": True,
            "same_four_gpu_ddp_topology": True,
            "only_permitted_difference": "per-row Trainer example weights",
            "equal_lambda": 0.0,
            "source50_lambda": 0.5,
            "source50_exact_multiplier_range": ["5/6", "5/4"],
            "v49a_full_parent_multiplier_range": ["2/3", "3/2"],
            "no_other_lambda_or_hpo_arm_authorized": True,
        },
        "matched_recipe_parent": {
            "path": str(V47C_PREREGISTRATION),
            "file_sha256": V47C_PREREGISTRATION_FILE_SHA256,
            "content_sha256": V47C_PREREGISTRATION_CONTENT_SHA256,
            "runtime_schedule_surface_inherited": True,
            "parent_recipe_identity_checked": True,
        },
        "future_evaluation": {
            "template_path": str(launcher.FUTURE_EVAL_BUILDER.parent /
                "experiments/eggroll_es_hpo/preregistrations/"
                "sft_v434_equal_vs_source50_replicated_ood_first_template_v49d.json"
            ),
            "template_not_yet_a_launch_manifest": True,
            "requires_both_training_completion_receipts": True,
            "evaluation_authorized_by_this_preregistration": False,
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
            "this_run_authorizes": "two fixed train-only V49D states",
            "training_order": ["v434_equal", "v434_source50"],
            "shadow_ood_holdout_feedback_between_arms_authorized": False,
            "recipe_change_between_arms_authorized": False,
            "promotion_authorized": False,
        },
    }
    parent_recipe = parent["recipe"]
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
        raise RuntimeError("V49D V47C/V42I invariant recipe changed")
    equal_recipe = arms["v434_equal"]["recipe"]
    source_recipe = arms["v434_source50"]["recipe"]
    excluded = {"command", "expected_weighting_audit"}
    if (
        {k: v for k, v in equal_recipe.items() if k not in excluded}
        != {k: v for k, v in source_recipe.items() if k not in excluded}
        or equal_recipe["command"][0:4] == source_recipe["command"][0:4]
    ):
        # Commands must differ at the SFT script, while all recipe fields other
        # than the weight audit and command remain byte-for-byte equal.
        raise RuntimeError("V49D arm-control isolation changed")
    result["content_sha256_before_self_field"] = engine.canonical_sha256(result)
    return result


def launch_command(arm: str, file_sha256: str, content_sha256: str) -> list[str]:
    return [
        str(ROOT / ".venv/bin/python"),
        str(Path(launcher.__file__).resolve()),
        *argument_vector(arm, file_sha256, content_sha256),
    ]


def main() -> None:
    result = build()
    if PREREGISTRATION.exists():
        raise FileExistsError("V49D preregistration already exists")
    engine.atomic_write_json(PREREGISTRATION, result)
    file_sha = engine.file_sha256(PREREGISTRATION)
    content_sha = result["content_sha256_before_self_field"]
    print(json.dumps({
        "path": str(PREREGISTRATION),
        "file_sha256": file_sha,
        "content_sha256": content_sha,
        "launch_commands": {
            arm: shlex.join(launch_command(arm, file_sha, content_sha))
            for arm in launcher.ARMS
        },
        "gpu_accessed": False,
        "training_launched": False,
        "non_train_semantics_opened": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
