#!/usr/bin/env python3
"""Fail-closed V37A equal-unit SFT launcher built on the audited V36 engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_sft_train_only_control_v36a as engine


ROOT = Path(__file__).resolve().parent
SFT_SCRIPT = (ROOT / "sft_lora_equal_unit_v37a.py").resolve()
EXPECTED_STEPS = 48
EXPECTED_CHECKPOINTS = [16, 32, 48]
EXPECTED_ENCODING_AUDIT = {
    "prompt_mode": "es_exact",
    "eos_appended": False,
    "train_prompt_tokens": 22_340,
    "train_answer_tokens": 11_420,
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
    "identity_sha256": "631199dc13d240434f7b0a9ea94c0848c315d83b12fada0be3a7189e57a85b06",
}
BASE_PARSER = engine.parser


def parser() -> argparse.ArgumentParser:
    result = BASE_PARSER()
    result.add_argument("--expected-conflict-units", required=True, type=int)
    result.add_argument("--expected-weight-identity-sha256", required=True)
    return result


def build_train_command(args: argparse.Namespace) -> list[str]:
    return [
        str((ROOT / ".venv/bin/torchrun").resolve()),
        "--standalone",
        "--nproc-per-node=4",
        str(SFT_SCRIPT),
        "--data", str(Path(args.dataset).resolve()),
        "--data-sha256", args.dataset_sha256,
        "--data-rows", str(args.dataset_rows),
        "--expected-conflict-units", str(args.expected_conflict_units),
        "--expected-weight-identity-sha256",
        args.expected_weight_identity_sha256,
        "--out", str(Path(args.output_dir).resolve()),
        "--epochs", str(args.epochs),
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


def validate_preregistration(args: argparse.Namespace, command: list[str]) -> dict:
    path = Path(args.preregistration).resolve()
    if engine.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("v37a preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content_hash = value.get("content_sha256_before_self_field")
    if (
        content_hash != args.preregistration_content_sha256
        or content_hash != engine.canonical_sha256({
            key: item for key, item in value.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v37a preregistration content identity changed")
    expected_paths = {
        "output_dir": str(Path(args.output_dir).resolve()),
        "stdout_log": str(Path(args.stdout_log).resolve()),
        "gpu_log": str(Path(args.gpu_log).resolve()),
        "report": str(Path(args.report).resolve()),
        "attempt_report": str(Path(args.attempt_report).resolve()),
    }
    if (
        value.get("schema")
        != "specialist-sft-equal-unit-preregistration-v37a"
        or value.get("status") != "preregistered_not_yet_run"
        or value.get("contains_external_validation_ood_or_holdout_content") is not False
        or value.get("recipe", {}).get("command") != command
        or value.get("dataset", {}).get("sha256") != args.dataset_sha256
        or value.get("dataset", {}).get("rows") != args.dataset_rows
        or value.get("artifacts") != expected_paths
    ):
        raise RuntimeError("v37a preregistration contract changed")
    implementation = value["implementation"]
    model = value["model"]
    comparison = value["comparison_binding"]
    observed = {
        "launcher": engine.file_sha256(Path(__file__).resolve()),
        "engine": engine.file_sha256(Path(engine.__file__).resolve()),
        "sft": engine.file_sha256(SFT_SCRIPT),
        "model_config": engine.file_sha256(engine.BASE_MODEL / "config.json"),
        "model_index": engine.file_sha256(
            engine.BASE_MODEL / "model.safetensors.index.json"
        ),
        "layer_plan": engine.file_sha256(Path(comparison["eggroll_es_layer_plan"])),
        "split_manifest": engine.file_sha256(Path(comparison["split_manifest"])),
    }
    expected = {
        "launcher": implementation["launcher_sha256"],
        "engine": implementation["engine_sha256"],
        "sft": implementation["sft_sha256"],
        "model_config": model["config_sha256"],
        "model_index": model["index_sha256"],
        "layer_plan": comparison["eggroll_es_layer_plan_sha256"],
        "split_manifest": comparison["split_manifest_file_sha256"],
    }
    if observed != expected:
        raise RuntimeError("v37a preregistration implementation binding changed")
    return {
        "path": str(path),
        "file_sha256": args.preregistration_sha256,
        "content_sha256": content_hash,
        "implementation_bindings": observed,
    }


def validate_output_artifacts(output_dir: Path, ignored_steps: int) -> dict:
    del ignored_steps
    final = output_dir / "final"
    config_path = final / "adapter_config.json"
    adapter_path = final / "adapter_model.safetensors"
    if not config_path.is_file() or not adapter_path.is_file():
        raise RuntimeError("v37a final LoRA adapter is incomplete")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_modules = {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
        "down_proj", "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z",
        "out_proj",
    }
    expected_parameters = {
        f"model.layers.{layer}.mlp.gate.weight" for layer in (20, 21, 22, 23)
    }
    if (
        config.get("r") != 32
        or config.get("lora_alpha") != 64
        or config.get("lora_dropout") != 0.0
        or config.get("layers_to_transform") != [20, 21, 22, 23]
        or set(config.get("target_modules", [])) != expected_modules
        or set(config.get("target_parameters", [])) != expected_parameters
    ):
        raise RuntimeError("v37a saved adapter configuration changed")
    checkpoints = sorted(
        int(path.name.removeprefix("checkpoint-"))
        for path in output_dir.glob("checkpoint-*")
        if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
    )
    state_path = output_dir / "checkpoint-48/trainer_state.json"
    if checkpoints != EXPECTED_CHECKPOINTS or not state_path.is_file():
        raise RuntimeError("v37a checkpoint schedule changed")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("global_step") != EXPECTED_STEPS or float(state.get("epoch")) != 3.0:
        raise RuntimeError("v37a final optimizer step count changed")
    return {
        "adapter_config": config,
        "checkpoint_steps": checkpoints,
        "final_global_step": state["global_step"],
        "final_epoch": state["epoch"],
    }


def rewrite_evidence(args: argparse.Namespace) -> None:
    report_path = Path(args.report).resolve()
    attempt_path = Path(args.attempt_report).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    weighting = engine.extract_json_event(
        Path(args.stdout_log).read_text(encoding="utf-8"), "weighting_audit"
    )
    if weighting["value"] != EXPECTED_WEIGHTING_AUDIT:
        raise RuntimeError("v37a observed weighting audit changed")
    report.pop("content_sha256_before_self_field", None)
    report.update({
        "schema": "specialist-sft-equal-unit-runtime-v37a",
        "status": "complete_train_only_states_sealed_shadow_unopened",
        "selection_surface": "fold_3_train_only",
        "validation_ood_or_holdout_opened": False,
        "observed_weighting_audit": weighting,
        "expected_optimizer_steps": EXPECTED_STEPS,
        "implementation": {
            "launcher": str(Path(__file__).resolve()),
            "launcher_sha256": engine.file_sha256(Path(__file__).resolve()),
            "engine": str(Path(engine.__file__).resolve()),
            "engine_sha256": engine.file_sha256(Path(engine.__file__).resolve()),
            "sft_script": str(SFT_SCRIPT),
            "sft_script_sha256": engine.file_sha256(SFT_SCRIPT),
        },
        "interpretation": (
            "equal-unit SFT state is sealed; no shadow-dev, external validation, "
            "OOD, holdout, quality, or promotion conclusion is authorized"
        ),
    })
    report["recipe"]["loss_mode"] = "equal_conflict_unit_answer_token_mean"
    engine.atomic_write_json(report_path, engine.self_hashed(report))
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt.pop("content_sha256_before_self_field", None)
    attempt.update({
        "schema": "specialist-sft-equal-unit-attempt-v37a",
        "final_report_sha256": engine.file_sha256(report_path),
    })
    engine.atomic_write_json(attempt_path, engine.self_hashed(attempt))


def main(argv: list[str] | None = None) -> int:
    engine.SFT_SCRIPT = SFT_SCRIPT
    engine.EXPECTED_ENCODING_AUDIT = EXPECTED_ENCODING_AUDIT
    engine.parser = parser
    engine.build_train_command = build_train_command
    engine.validate_preregistration = validate_preregistration
    engine.validate_output_artifacts = validate_output_artifacts
    result = engine.main(argv)
    if result == 0 and argv is not None and "--dry-run" not in argv:
        args = parser().parse_args(argv)
        rewrite_evidence(args)
    elif result == 0 and argv is None:
        import sys
        if "--dry-run" not in sys.argv[1:]:
            rewrite_evidence(parser().parse_args())
    return result


if __name__ == "__main__":
    raise SystemExit(main())
