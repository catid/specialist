#!/usr/bin/env python3
"""Fail-closed four-GPU V47C SFT launcher on lineage-stable v430 fold 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_train_refresh_lineage_stable_v430_v47c as refresh
import run_sft_equal_unit_matched_init_v42a as v42a
import run_sft_equal_unit_matched_init_v42b as v42b
import run_sft_train_only_control_v36a as engine
import sft_lora_equal_unit_matched_init_v42a as source_contract
import sft_lora_equal_unit_matched_init_v42b as sft_v42b
import sft_lora_equal_unit_matched_init_v47a as sft


ROOT = Path(__file__).resolve().parent
SFT_SCRIPT = (ROOT / "sft_lora_equal_unit_matched_init_v47a.py").resolve()
V42A_SFT_SCRIPT = (ROOT / "sft_lora_equal_unit_matched_init_v42a.py").resolve()
V42B_LOADER_SCRIPT = (ROOT / "sft_lora_equal_unit_matched_init_v42b.py").resolve()
OBJECTIVE_SCRIPT = (ROOT / "sft_lora_equal_unit_v37a.py").resolve()
LEARNING_RATE = 5.5e-5
EXPECTED_STEPS = 48
V42I_PREREGISTRATION = (
    ROOT / "experiments/sft_controls/"
    "v42i_matched_init_equal_unit_fold3_v412_lr5p5e5/"
    "preregistration_v42i.json"
).resolve()
EXPECTED_ENCODING_AUDIT = {
    "prompt_mode": "es_exact",
    "eos_appended": False,
    "train_prompt_tokens": 22_340,
    "train_answer_tokens": 11_675,
    "train_rows": 448,
}
EXPECTED_WEIGHTING_AUDIT = {
    "schema": "specialist-equal-conflict-unit-weighting-v37a",
    "rows": 448,
    "conflict_units": 208,
    "ordinary_row_mean_weight": 1.0,
    "minimum_row_weight": 0.05128205128205128,
    "maximum_row_weight": 2.1538461538461537,
    "unit_objective_mass": 0.004807692307692308,
    "identity_sha256": "d8a373ee35920d5325b94a1037b8e605eccc7c3aa803625735d31b1b864b6bcb",
}


def parser() -> argparse.ArgumentParser:
    result = v42b.parser()
    result.add_argument("--max-steps", required=True, type=int)
    return result


def validate_recipe(args: argparse.Namespace) -> None:
    if (
        args.epochs != 3.0
        or args.max_steps != EXPECTED_STEPS
        or args.rank != 32
        or args.lora_dropout != 0.0
        or args.grad_accum != 1
        or args.per_device_batch_size != 7
        or args.learning_rate != LEARNING_RATE
        or args.seed != 17
        or args.max_length != 1024
        or args.save_steps != 16
        or args.attn_implementation != "sdpa"
        or args.prompt_mode != "es_exact"
        or args.loss_mode != "example_mean"
        or args.target_layers != "20,21,22,23"
        or args.expected_trainable_elements != 4_528_128
        or args.expected_trainable_tensors != 70
        or args.dataset_rows != 448
        or args.expected_conflict_units != 208
        or args.expected_weight_identity_sha256
        != EXPECTED_WEIGHTING_AUDIT["identity_sha256"]
    ):
        raise ValueError("V47C exact V42I recipe or lineage-stable data changed")
    source_contract.validate_recipe_arguments_v42a(argparse.Namespace(
        rank=args.rank,
        target_layers=args.target_layers,
        expected_trainable_elements=args.expected_trainable_elements,
        expected_trainable_tensors=args.expected_trainable_tensors,
        expected_world_size=4,
        initial_adapter=args.initial_adapter,
        initial_adapter_weights_sha256=args.initial_adapter_weights_sha256,
        initial_adapter_config_sha256=args.initial_adapter_config_sha256,
        initial_adapter_manifest_sha256=args.initial_adapter_manifest_sha256,
        initial_adapter_manifest_content_sha256=(
            args.initial_adapter_manifest_content_sha256
        ),
        initial_adapter_tensor_identity_sha256=(
            args.initial_adapter_tensor_identity_sha256
        ),
    ))


def build_train_command(args: argparse.Namespace) -> list[str]:
    validate_recipe(args)
    return [
        str((ROOT / ".venv/bin/torchrun").resolve()),
        "--standalone", "--nproc-per-node=4", str(SFT_SCRIPT),
        "--data", str(Path(args.dataset).resolve()),
        "--data-sha256", args.dataset_sha256,
        "--data-rows", str(args.dataset_rows),
        "--expected-conflict-units", str(args.expected_conflict_units),
        "--expected-weight-identity-sha256", args.expected_weight_identity_sha256,
        "--initial-adapter", str(Path(args.initial_adapter).resolve()),
        "--initial-adapter-weights-sha256", args.initial_adapter_weights_sha256,
        "--initial-adapter-config-sha256", args.initial_adapter_config_sha256,
        "--initial-adapter-manifest-sha256", args.initial_adapter_manifest_sha256,
        "--initial-adapter-manifest-content-sha256",
        args.initial_adapter_manifest_content_sha256,
        "--initial-adapter-tensor-identity-sha256",
        args.initial_adapter_tensor_identity_sha256,
        "--out", str(Path(args.output_dir).resolve()),
        "--epochs", str(args.epochs),
        "--max-steps", str(args.max_steps),
        "--rank", str(args.rank),
        "--per-device-batch-size", str(args.per_device_batch_size),
        "--learning-rate", str(args.learning_rate),
        "--seed", str(args.seed),
        "--max-length", str(args.max_length),
        "--save-steps", str(args.save_steps),
        "--target-layers", args.target_layers,
        "--expected-trainable-elements", str(args.expected_trainable_elements),
        "--expected-trainable-tensors", str(args.expected_trainable_tensors),
        "--expected-world-size", "4",
        "--attn-implementation", args.attn_implementation,
    ]


def implementation_bindings() -> dict:
    return {
        "launcher": engine.file_sha256(Path(__file__).resolve()),
        "engine": engine.file_sha256(Path(engine.__file__).resolve()),
        "sft": engine.file_sha256(SFT_SCRIPT),
        "sft_v42a_source": engine.file_sha256(V42A_SFT_SCRIPT),
        "sft_v42b_loader": engine.file_sha256(V42B_LOADER_SCRIPT),
        "objective": engine.file_sha256(OBJECTIVE_SCRIPT),
        "refresh_builder": engine.file_sha256(Path(refresh.__file__).resolve()),
        "refresh_manifest": engine.file_sha256(refresh.MANIFEST),
        "v42i_preregistration": engine.file_sha256(V42I_PREREGISTRATION),
        "model_config": engine.file_sha256(engine.BASE_MODEL / "config.json"),
        "model_index": engine.file_sha256(
            engine.BASE_MODEL / "model.safetensors.index.json"
        ),
    }


def validate_preregistration(args, command) -> dict:
    path = Path(args.preregistration).resolve()
    if engine.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V47C preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    expected_artifacts = {
        "output_dir": str(Path(args.output_dir).resolve()),
        "stdout_log": str(Path(args.stdout_log).resolve()),
        "gpu_log": str(Path(args.gpu_log).resolve()),
        "report": str(Path(args.report).resolve()),
        "attempt_report": str(Path(args.attempt_report).resolve()),
    }
    if (
        content != args.preregistration_content_sha256
        or content != engine.canonical_sha256({
            key: item for key, item in value.items()
            if key != "content_sha256_before_self_field"
        })
        or value.get("schema")
        != "specialist-sft-lineage-stable-v430-preregistration-v47c"
        or value.get("status") != "sealed_unlaunched_holdout_blind"
        or value.get("contains_external_validation_ood_or_holdout_content")
        is not False
        or value.get("dataset", {}).get("path")
        != str(Path(args.dataset).resolve())
        or value.get("dataset", {}).get("sha256") != args.dataset_sha256
        or value.get("dataset", {}).get("rows") != args.dataset_rows
        or value.get("recipe", {}).get("command") != command
        or value.get("recipe", {}).get("explicit_max_steps_cap") != 48
        or value.get("recipe", {}).get("expected_optimizer_steps") != 48
        or value.get("recipe", {}).get("expected_encoding_audit")
        != EXPECTED_ENCODING_AUDIT
        or value.get("recipe", {}).get("expected_weighting_audit")
        != EXPECTED_WEIGHTING_AUDIT
        or value.get("recipe", {}).get("expected_schedule_audit")
        != sft.schedule_audit_v47a(EXPECTED_STEPS)
        or value.get("artifacts") != expected_artifacts
        or value.get("access_firewall", {}).get("shadow_dev_opened_during_training")
        is not False
        or value.get("access_firewall", {}).get("eval_ood_holdout_opened")
        is not False
    ):
        raise RuntimeError("V47C sealed preregistration contract changed")
    observed = implementation_bindings()
    if observed != value["implementation"]["file_sha256_bindings"]:
        raise RuntimeError("V47C implementation binding changed")
    initialization = source_contract.validate_initialization_artifact_v42a(
        Path(args.initial_adapter)
    )
    loader = sft_v42b.expected_loader_audit_v42b()
    if value.get("initialization") != initialization or value.get(
        "adapter_loader"
    ) != loader:
        raise RuntimeError("V47C matched initialization or loader changed")
    return {
        "path": str(path), "file_sha256": args.preregistration_sha256,
        "content_sha256": content, "implementation_bindings": observed,
        "source_initialization": initialization, "adapter_loader": loader,
    }


def rewrite_evidence(args: argparse.Namespace) -> None:
    report_path = Path(args.report).resolve()
    attempt_path = Path(args.attempt_report).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    log_text = Path(args.stdout_log).read_text(encoding="utf-8")
    weighting = engine.extract_json_event(log_text, "weighting_audit")
    initialization = engine.extract_json_event(log_text, "source_initialization_audit")
    loader = engine.extract_json_event(log_text, "initialization_loader_audit_v42b")
    schedule = engine.extract_json_event(log_text, "schedule_audit_v47a")
    expected_initialization = v42a.expected_initialization_runtime_audit_v42a()
    expected_loader = sft_v42b.expected_loader_audit_v42b()
    if (
        weighting["value"] != EXPECTED_WEIGHTING_AUDIT
        or initialization["value"] != expected_initialization
        or loader["value"] != expected_loader
        or schedule["value"] != sft.schedule_audit_v47a(EXPECTED_STEPS)
    ):
        raise RuntimeError("V47C runtime audit changed")
    report.pop("content_sha256_before_self_field", None)
    report.update({
        "schema": "specialist-sft-lineage-stable-v430-runtime-v47c",
        "status": "complete_train_only_state_sealed_shadow_ood_holdout_unopened",
        "selection_surface": "lineage_stable_refreshed_fold_3_train_only",
        "validation_ood_or_holdout_opened": False,
        "shadow_dev_opened": False,
        "observed_weighting_audit": weighting,
        "source_initialization": {
            "expected": expected_initialization, "observed": initialization,
            "all_ddp_rank_emissions_identical": True,
        },
        "adapter_loader": {
            "expected": expected_loader, "observed": loader,
            "all_ddp_rank_emissions_identical": True,
        },
        "explicit_step_cap": schedule,
        "expected_optimizer_steps": EXPECTED_STEPS,
        "refresh_manifest": str(refresh.MANIFEST),
    })
    report["recipe"].update({
        "learning_rate": LEARNING_RATE,
        "loss_mode": "equal_conflict_unit_answer_token_mean",
        "initialization": expected_initialization["source"],
        "adapter_loader": expected_loader,
        "max_steps": EXPECTED_STEPS,
        "recipe_parent": "V42I exact hyperparameters and initialization",
    })
    engine.atomic_write_json(report_path, engine.self_hashed(report))
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt.pop("content_sha256_before_self_field", None)
    attempt.update({
        "schema": "specialist-sft-lineage-stable-v430-attempt-v47c",
        "source_initialization": expected_initialization["source"],
        "adapter_loader": expected_loader,
        "final_report_sha256": engine.file_sha256(report_path),
    })
    engine.atomic_write_json(attempt_path, engine.self_hashed(attempt))


def main(argv: list[str] | None = None) -> int:
    originals = (
        engine.SFT_SCRIPT, engine.EXPECTED_ENCODING_AUDIT, engine.parser,
        engine.build_train_command, engine.validate_preregistration,
        engine.validate_output_artifacts,
    )
    engine.SFT_SCRIPT = SFT_SCRIPT
    engine.EXPECTED_ENCODING_AUDIT = EXPECTED_ENCODING_AUDIT
    engine.parser = parser
    engine.build_train_command = build_train_command
    engine.validate_preregistration = validate_preregistration
    engine.validate_output_artifacts = v42a.validate_output_artifacts
    try:
        result = engine.main(argv)
    finally:
        (
            engine.SFT_SCRIPT, engine.EXPECTED_ENCODING_AUDIT, engine.parser,
            engine.build_train_command, engine.validate_preregistration,
            engine.validate_output_artifacts,
        ) = originals
    effective = parser().parse_args(argv) if argv is not None else parser().parse_args()
    if result == 0 and not effective.dry_run:
        rewrite_evidence(effective)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
