#!/usr/bin/env python3
"""Fail-closed V42A matched-initialization equal-unit SFT launcher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file

import run_sft_train_only_control_v36a as engine
import sft_lora_equal_unit_matched_init_v42a as sft


ROOT = Path(__file__).resolve().parent
SFT_SCRIPT = (ROOT / "sft_lora_equal_unit_matched_init_v42a.py").resolve()
OBJECTIVE_SCRIPT = (ROOT / "sft_lora_equal_unit_v37a.py").resolve()
EXPECTED_STEPS_V42A = 48
EXPECTED_CHECKPOINTS_V42A = [16, 32, 48]
EXPECTED_ENCODING_AUDIT_V42A = {
    "prompt_mode": "es_exact",
    "eos_appended": False,
    "train_prompt_tokens": 22_340,
    "train_answer_tokens": 11_420,
    "train_rows": 448,
}
EXPECTED_WEIGHTING_AUDIT_V42A = {
    "schema": "specialist-equal-conflict-unit-weighting-v37a",
    "rows": 448,
    "conflict_units": 208,
    "ordinary_row_mean_weight": 1.0,
    "minimum_row_weight": 0.05128205128205128,
    "maximum_row_weight": 2.1538461538461537,
    "unit_objective_mass": 0.004807692307692308,
    "identity_sha256": (
        "631199dc13d240434f7b0a9ea94c0848c315d83b12fada0be3a7189e57a85b06"
    ),
}
BASE_PARSER_V42A = engine.parser


def parser() -> argparse.ArgumentParser:
    result = BASE_PARSER_V42A()
    result.add_argument("--expected-conflict-units", required=True, type=int)
    result.add_argument("--expected-weight-identity-sha256", required=True)
    result.add_argument("--initial-adapter", required=True)
    result.add_argument("--initial-adapter-weights-sha256", required=True)
    result.add_argument("--initial-adapter-config-sha256", required=True)
    result.add_argument("--initial-adapter-manifest-sha256", required=True)
    result.add_argument(
        "--initial-adapter-manifest-content-sha256", required=True
    )
    result.add_argument(
        "--initial-adapter-tensor-identity-sha256", required=True
    )
    return result


def _validate_runner_recipe_v42a(args: argparse.Namespace) -> None:
    if (
        args.epochs != 3.0
        or args.rank != sft.EXPECTED_RANK_V42A
        or args.lora_dropout != 0.0
        or args.grad_accum != 1
        or args.per_device_batch_size != 7
        or args.learning_rate != 1e-4
        or args.seed != 17
        or args.max_length != 1024
        or args.save_steps != 16
        or args.attn_implementation != "sdpa"
        or args.prompt_mode != "es_exact"
        or args.loss_mode != "example_mean"
        or args.target_layers != "20,21,22,23"
        or args.expected_trainable_elements != sft.EXPECTED_ELEMENTS_V42A
        or args.expected_trainable_tensors != sft.EXPECTED_TENSORS_V42A
        or args.expected_conflict_units != 208
        or args.expected_weight_identity_sha256
        != EXPECTED_WEIGHTING_AUDIT_V42A["identity_sha256"]
    ):
        raise ValueError("V42A runner recipe changed")
    child_args = argparse.Namespace(
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
    )
    sft.validate_recipe_arguments_v42a(child_args)


def build_train_command(args: argparse.Namespace) -> list[str]:
    _validate_runner_recipe_v42a(args)
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
        "--initial-adapter", str(Path(args.initial_adapter).resolve()),
        "--initial-adapter-weights-sha256",
        args.initial_adapter_weights_sha256,
        "--initial-adapter-config-sha256",
        args.initial_adapter_config_sha256,
        "--initial-adapter-manifest-sha256",
        args.initial_adapter_manifest_sha256,
        "--initial-adapter-manifest-content-sha256",
        args.initial_adapter_manifest_content_sha256,
        "--initial-adapter-tensor-identity-sha256",
        args.initial_adapter_tensor_identity_sha256,
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


def expected_initialization_runtime_audit_v42a() -> dict:
    source = sft.validate_initialization_artifact_v42a(
        sft.INITIAL_ADAPTER_V42A
    )
    loaded = load_file(
        str(sft.INITIAL_ADAPTER_V42A / "adapter_model.safetensors"),
        device="cpu",
    )
    loaded_audit = sft.validate_loaded_adapter_state_v42a(
        loaded, sft.INITIAL_ADAPTER_V42A, source
    )
    return {
        "schema": "specialist-matched-lora-initialization-runtime-v42a",
        "source": source,
        "loaded": loaded_audit,
    }


def validate_preregistration(
    args: argparse.Namespace, command: list[str]
) -> dict:
    path = Path(args.preregistration).resolve()
    if engine.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V42A preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content_hash = value.get("content_sha256_before_self_field")
    if (
        content_hash != args.preregistration_content_sha256
        or content_hash != engine.canonical_sha256({
            key: item for key, item in value.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("V42A preregistration content identity changed")
    expected_paths = {
        "output_dir": str(Path(args.output_dir).resolve()),
        "stdout_log": str(Path(args.stdout_log).resolve()),
        "gpu_log": str(Path(args.gpu_log).resolve()),
        "report": str(Path(args.report).resolve()),
        "attempt_report": str(Path(args.attempt_report).resolve()),
    }
    if (
        value.get("schema")
        != "specialist-sft-matched-init-equal-unit-preregistration-v42a"
        or value.get("status") != "preregistered_not_yet_run"
        or value.get("contains_external_validation_ood_or_holdout_content")
        is not False
        or value.get("recipe", {}).get("command") != command
        or value.get("dataset", {}).get("sha256") != args.dataset_sha256
        or value.get("dataset", {}).get("rows") != args.dataset_rows
        or value.get("artifacts") != expected_paths
    ):
        raise RuntimeError("V42A preregistration contract changed")

    implementation = value["implementation"]
    model = value["model"]
    comparison = value["comparison_binding"]
    observed = {
        "launcher": engine.file_sha256(Path(__file__).resolve()),
        "engine": engine.file_sha256(Path(engine.__file__).resolve()),
        "sft": engine.file_sha256(SFT_SCRIPT),
        "objective": engine.file_sha256(OBJECTIVE_SCRIPT),
        "model_config": engine.file_sha256(engine.BASE_MODEL / "config.json"),
        "model_index": engine.file_sha256(
            engine.BASE_MODEL / "model.safetensors.index.json"
        ),
        "layer_plan": engine.file_sha256(
            Path(comparison["eggroll_es_layer_plan"])
        ),
        "split_manifest": engine.file_sha256(Path(comparison["split_manifest"])),
    }
    expected = {
        "launcher": implementation["launcher_sha256"],
        "engine": implementation["engine_sha256"],
        "sft": implementation["sft_sha256"],
        "objective": implementation["equal_unit_objective_sha256"],
        "model_config": model["config_sha256"],
        "model_index": model["index_sha256"],
        "layer_plan": comparison["eggroll_es_layer_plan_sha256"],
        "split_manifest": comparison["split_manifest_file_sha256"],
    }
    if observed != expected:
        raise RuntimeError("V42A preregistration implementation binding changed")
    initialization = sft.validate_initialization_artifact_v42a(
        Path(args.initial_adapter)
    )
    if value.get("initialization") != initialization:
        raise RuntimeError("V42A preregistered initialization identity changed")
    return {
        "path": str(path),
        "file_sha256": args.preregistration_sha256,
        "content_sha256": content_hash,
        "implementation_bindings": observed,
        "source_initialization": initialization,
    }


def validate_output_artifacts(output_dir: Path, ignored_steps: int) -> dict:
    del ignored_steps
    final = output_dir / "final"
    config_path = final / "adapter_config.json"
    adapter_path = final / "adapter_model.safetensors"
    if not config_path.is_file() or not adapter_path.is_file():
        raise RuntimeError("V42A final LoRA adapter is incomplete")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_modules = {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
        "down_proj", "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z",
        "out_proj",
    }
    expected_parameters = {
        f"model.layers.{layer}.mlp.gate.weight"
        for layer in sft.EXPECTED_LAYERS_V42A
    }
    if (
        config.get("r") != sft.EXPECTED_RANK_V42A
        or config.get("lora_alpha") != sft.EXPECTED_ALPHA_V42A
        or config.get("lora_dropout") != 0.0
        or config.get("layers_to_transform") != sft.EXPECTED_LAYERS_V42A
        or set(config.get("target_modules", [])) != expected_modules
        or set(config.get("target_parameters", [])) != expected_parameters
    ):
        raise RuntimeError("V42A saved adapter configuration changed")
    checkpoints = sorted(
        int(path.name.removeprefix("checkpoint-"))
        for path in output_dir.glob("checkpoint-*")
        if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
    )
    state_path = output_dir / "checkpoint-48/trainer_state.json"
    if checkpoints != EXPECTED_CHECKPOINTS_V42A or not state_path.is_file():
        raise RuntimeError("V42A checkpoint schedule changed")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        state.get("global_step") != EXPECTED_STEPS_V42A
        or float(state.get("epoch")) != 3.0
    ):
        raise RuntimeError("V42A final optimizer step count changed")
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
    log_text = Path(args.stdout_log).read_text(encoding="utf-8")
    weighting = engine.extract_json_event(log_text, "weighting_audit")
    initialization = engine.extract_json_event(
        log_text, "source_initialization_audit"
    )
    expected_initialization = expected_initialization_runtime_audit_v42a()
    if weighting["value"] != EXPECTED_WEIGHTING_AUDIT_V42A:
        raise RuntimeError("V42A observed weighting audit changed")
    if initialization["value"] != expected_initialization:
        raise RuntimeError("V42A observed source initialization audit changed")
    report.pop("content_sha256_before_self_field", None)
    report.update({
        "schema": "specialist-sft-matched-init-equal-unit-runtime-v42a",
        "status": "complete_train_only_state_sealed_shadow_unopened",
        "selection_surface": "fold_3_train_only",
        "validation_ood_or_holdout_opened": False,
        "observed_weighting_audit": weighting,
        "source_initialization": {
            "expected": expected_initialization,
            "observed": initialization,
            "all_ddp_rank_emissions_identical": True,
        },
        "expected_optimizer_steps": EXPECTED_STEPS_V42A,
        "implementation": {
            "launcher": str(Path(__file__).resolve()),
            "launcher_sha256": engine.file_sha256(Path(__file__).resolve()),
            "engine": str(Path(engine.__file__).resolve()),
            "engine_sha256": engine.file_sha256(Path(engine.__file__).resolve()),
            "sft_script": str(SFT_SCRIPT),
            "sft_script_sha256": engine.file_sha256(SFT_SCRIPT),
            "equal_unit_objective": str(OBJECTIVE_SCRIPT),
            "equal_unit_objective_sha256": engine.file_sha256(OBJECTIVE_SCRIPT),
        },
        "interpretation": (
            "matched-initialization equal-unit SFT state is sealed; no "
            "shadow-dev, external validation, OOD, holdout, quality, or "
            "promotion conclusion is authorized"
        ),
    })
    report["recipe"]["loss_mode"] = "equal_conflict_unit_answer_token_mean"
    report["recipe"]["initialization"] = expected_initialization["source"]
    engine.atomic_write_json(report_path, engine.self_hashed(report))
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt.pop("content_sha256_before_self_field", None)
    attempt.update({
        "schema": "specialist-sft-matched-init-equal-unit-attempt-v42a",
        "source_initialization": expected_initialization["source"],
        "final_report_sha256": engine.file_sha256(report_path),
    })
    engine.atomic_write_json(attempt_path, engine.self_hashed(attempt))


def main(argv: list[str] | None = None) -> int:
    engine.SFT_SCRIPT = SFT_SCRIPT
    engine.EXPECTED_ENCODING_AUDIT = EXPECTED_ENCODING_AUDIT_V42A
    engine.parser = parser
    engine.build_train_command = build_train_command
    engine.validate_preregistration = validate_preregistration
    engine.validate_output_artifacts = validate_output_artifacts
    result = engine.main(argv)
    if result == 0 and argv is not None and "--dry-run" not in argv:
        rewrite_evidence(parser().parse_args(argv))
    elif result == 0 and argv is None:
        import sys
        if "--dry-run" not in sys.argv[1:]:
            rewrite_evidence(parser().parse_args())
    return result


if __name__ == "__main__":
    raise SystemExit(main())
