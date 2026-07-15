#!/usr/bin/env python3
"""Fail-closed V42B retry launcher for direct canonical adapter loading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_sft_equal_unit_matched_init_v42a as v42a_runner
import run_sft_train_only_control_v36a as engine
import sft_lora_equal_unit_matched_init_v42a as source_contract
import sft_lora_equal_unit_matched_init_v42b as sft


ROOT = Path(__file__).resolve().parent
SFT_SCRIPT = (ROOT / "sft_lora_equal_unit_matched_init_v42b.py").resolve()
V42A_SFT_SCRIPT = (ROOT / "sft_lora_equal_unit_matched_init_v42a.py").resolve()
OBJECTIVE_SCRIPT = (ROOT / "sft_lora_equal_unit_v37a.py").resolve()
FAILED_V42A_ATTEMPT = (
    ROOT / "experiments/sft_controls/v42a_matched_init_equal_unit_fold3_v412/"
    "attempt_v42a.json"
).resolve()
FAILED_V42A_STDOUT = FAILED_V42A_ATTEMPT.with_name("stdout_v42a.log")
FAILED_V42A_ATTEMPT_SHA256 = (
    "3028be34a788ccf837e63955efee02c6be761742bd9cfaee7155d7bbf282ceea"
)
FAILED_V42A_ATTEMPT_CONTENT_SHA256 = (
    "08b63af9e7a157ab686a39c70abd9ff35651c697beb0ee5dce3c88381815324d"
)
FAILED_V42A_STDOUT_SHA256 = (
    "0db57168d1fd664b491fb0e024f99f1ce8d3a83d6f1c6b91358002b073b5f7ae"
)
CPU_MODEL_LOAD_SMOKE = (
    ROOT / "experiments/sft_controls/"
    "v42b_matched_init_equal_unit_fold3_v412_retry_direct_load/"
    "cpu_model_load_smoke_v42b.json"
).resolve()
CPU_MODEL_LOAD_SMOKE_SHA256 = (
    "c46fa20aa1864a41401da8b269e1515d407af642f99f6e2161cc503389f60999"
)
CPU_MODEL_LOAD_SMOKE_CONTENT_SHA256 = (
    "19cc6d795f22d668b57ce5323a563fecb341348e01a6d16f4e9d0f79e1013ac7"
)
EXPECTED_FAILURE_SIGNATURE = (
    "WeightConverter.__init__() got an unexpected keyword argument "
    "'distributed_operation'"
)
BASE_PARSER_V42B = v42a_runner.parser


def parser() -> argparse.ArgumentParser:
    return BASE_PARSER_V42B()


def build_train_command(args: argparse.Namespace) -> list[str]:
    command = v42a_runner.build_train_command(args)
    if command[3] != str(v42a_runner.SFT_SCRIPT):
        raise RuntimeError("V42B inherited child-command layout changed")
    command[3] = str(SFT_SCRIPT)
    return command


def validate_failed_predecessor_v42b() -> dict:
    if (
        engine.file_sha256(FAILED_V42A_ATTEMPT)
        != FAILED_V42A_ATTEMPT_SHA256
        or engine.file_sha256(FAILED_V42A_STDOUT) != FAILED_V42A_STDOUT_SHA256
    ):
        raise RuntimeError("V42B predecessor failure evidence changed")
    attempt = json.loads(FAILED_V42A_ATTEMPT.read_text(encoding="utf-8"))
    content = attempt.get("content_sha256_before_self_field")
    without_self = {
        key: value for key, value in attempt.items()
        if key != "content_sha256_before_self_field"
    }
    stdout = FAILED_V42A_STDOUT.read_text(encoding="utf-8")
    if (
        content != FAILED_V42A_ATTEMPT_CONTENT_SHA256
        or engine.canonical_sha256(without_self) != content
        or attempt.get("status") != "failed"
        or attempt.get("returncode") != 1
        or EXPECTED_FAILURE_SIGNATURE not in stdout
    ):
        raise RuntimeError("V42B predecessor failure contract changed")
    return {
        "schema": "specialist-v42a-peft-load-failure-binding-v42b",
        "attempt": str(FAILED_V42A_ATTEMPT),
        "attempt_file_sha256": FAILED_V42A_ATTEMPT_SHA256,
        "attempt_content_sha256": FAILED_V42A_ATTEMPT_CONTENT_SHA256,
        "stdout": str(FAILED_V42A_STDOUT),
        "stdout_file_sha256": FAILED_V42A_STDOUT_SHA256,
        "failure_signature": EXPECTED_FAILURE_SIGNATURE,
        "failed_before_optimizer_construction": True,
        "retry_changes_only_adapter_load_mechanism": True,
    }


def validate_cpu_model_load_smoke_v42b() -> dict:
    if engine.file_sha256(CPU_MODEL_LOAD_SMOKE) != CPU_MODEL_LOAD_SMOKE_SHA256:
        raise RuntimeError("V42B real CPU model-load smoke file changed")
    value = json.loads(CPU_MODEL_LOAD_SMOKE.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    without_self = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        content != CPU_MODEL_LOAD_SMOKE_CONTENT_SHA256
        or engine.canonical_sha256(without_self) != content
        or value.get("schema") != "specialist-real-cpu-model-load-smoke-v42b"
        or value.get("status") != "complete_exact_load_verified"
        or value.get("gpu_accessed") is not False
        or value.get("cuda_initialized") is not False
        or value.get("visible_cuda_device_count") != 0
        or value.get("device_types") != ["cpu"]
        or value.get("dataset_or_training_examples_accessed") is not False
        or value.get("shadow_ood_holdout_or_heldout_accessed") is not False
        or value.get("evaluation_performed") is not False
        or value.get("source_initialization")
        != source_contract.validate_initialization_artifact_v42a(
            source_contract.INITIAL_ADAPTER_V42A
        )
        or value.get("adapter_loader") != sft.expected_loader_audit_v42b()
        or value.get("trainable_inventory") != engine.EXPECTED_TRAINABLE_INVENTORY
        or value.get("loaded_readback", {}).get("matches_source_tensor_bytes")
        is not True
    ):
        raise RuntimeError("V42B real CPU model-load smoke contract changed")
    return {
        "schema": "specialist-real-cpu-model-load-smoke-binding-v42b",
        "report": str(CPU_MODEL_LOAD_SMOKE),
        "report_file_sha256": CPU_MODEL_LOAD_SMOKE_SHA256,
        "report_content_sha256": CPU_MODEL_LOAD_SMOKE_CONTENT_SHA256,
        "verified": True,
        "device_types": ["cpu"],
        "visible_cuda_device_count": 0,
        "trainable_tensors": 70,
        "trainable_elements": 4_528_128,
        "matches_source_tensor_bytes": True,
    }


def validate_preregistration(
    args: argparse.Namespace, command: list[str]
) -> dict:
    path = Path(args.preregistration).resolve()
    if engine.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V42B preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content_hash = value.get("content_sha256_before_self_field")
    if (
        content_hash != args.preregistration_content_sha256
        or content_hash != engine.canonical_sha256({
            key: item for key, item in value.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("V42B preregistration content identity changed")
    expected_paths = {
        "output_dir": str(Path(args.output_dir).resolve()),
        "stdout_log": str(Path(args.stdout_log).resolve()),
        "gpu_log": str(Path(args.gpu_log).resolve()),
        "report": str(Path(args.report).resolve()),
        "attempt_report": str(Path(args.attempt_report).resolve()),
    }
    if (
        value.get("schema")
        != "specialist-sft-matched-init-equal-unit-preregistration-v42b"
        or value.get("status") != "preregistered_retry_not_yet_run"
        or value.get("contains_external_validation_ood_or_holdout_content")
        is not False
        or value.get("recipe", {}).get("command") != command
        or value.get("dataset", {}).get("sha256") != args.dataset_sha256
        or value.get("dataset", {}).get("rows") != args.dataset_rows
        or value.get("artifacts") != expected_paths
    ):
        raise RuntimeError("V42B preregistration contract changed")

    implementation = value["implementation"]
    model = value["model"]
    comparison = value["comparison_binding"]
    observed = {
        "launcher": engine.file_sha256(Path(__file__).resolve()),
        "engine": engine.file_sha256(Path(engine.__file__).resolve()),
        "sft": engine.file_sha256(SFT_SCRIPT),
        "v42a_source_contract": engine.file_sha256(V42A_SFT_SCRIPT),
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
        "v42a_source_contract": implementation["v42a_source_contract_sha256"],
        "objective": implementation["equal_unit_objective_sha256"],
        "model_config": model["config_sha256"],
        "model_index": model["index_sha256"],
        "layer_plan": comparison["eggroll_es_layer_plan_sha256"],
        "split_manifest": comparison["split_manifest_file_sha256"],
    }
    if observed != expected:
        raise RuntimeError("V42B preregistration implementation binding changed")
    initialization = source_contract.validate_initialization_artifact_v42a(
        Path(args.initial_adapter)
    )
    predecessor = validate_failed_predecessor_v42b()
    loader = sft.expected_loader_audit_v42b()
    cpu_smoke = validate_cpu_model_load_smoke_v42b()
    if (
        value.get("initialization") != initialization
        or value.get("predecessor_failure") != predecessor
        or value.get("adapter_loader") != loader
        or value.get("cpu_model_load_smoke") != cpu_smoke
    ):
        raise RuntimeError("V42B preregistered retry identity changed")
    return {
        "path": str(path),
        "file_sha256": args.preregistration_sha256,
        "content_sha256": content_hash,
        "implementation_bindings": observed,
        "source_initialization": initialization,
        "adapter_loader": loader,
        "cpu_model_load_smoke": cpu_smoke,
        "predecessor_failure": predecessor,
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
    loader = engine.extract_json_event(
        log_text, "initialization_loader_audit_v42b"
    )
    expected_initialization = (
        v42a_runner.expected_initialization_runtime_audit_v42a()
    )
    expected_loader = sft.expected_loader_audit_v42b()
    if weighting["value"] != v42a_runner.EXPECTED_WEIGHTING_AUDIT_V42A:
        raise RuntimeError("V42B observed weighting audit changed")
    if initialization["value"] != expected_initialization:
        raise RuntimeError("V42B observed source initialization audit changed")
    if loader["value"] != expected_loader:
        raise RuntimeError("V42B observed direct loader audit changed")
    report.pop("content_sha256_before_self_field", None)
    report.update({
        "schema": "specialist-sft-matched-init-equal-unit-runtime-v42b",
        "status": "complete_train_only_retry_state_sealed_shadow_unopened",
        "selection_surface": "fold_3_train_only",
        "validation_ood_or_holdout_opened": False,
        "observed_weighting_audit": weighting,
        "source_initialization": {
            "expected": expected_initialization,
            "observed": initialization,
            "all_ddp_rank_emissions_identical": True,
        },
        "adapter_loader": {
            "expected": expected_loader,
            "observed": loader,
            "all_ddp_rank_emissions_identical": True,
        },
        "predecessor_failure": validate_failed_predecessor_v42b(),
        "cpu_model_load_smoke": validate_cpu_model_load_smoke_v42b(),
        "expected_optimizer_steps": v42a_runner.EXPECTED_STEPS_V42A,
        "implementation": {
            "launcher": str(Path(__file__).resolve()),
            "launcher_sha256": engine.file_sha256(Path(__file__).resolve()),
            "engine": str(Path(engine.__file__).resolve()),
            "engine_sha256": engine.file_sha256(Path(engine.__file__).resolve()),
            "sft_script": str(SFT_SCRIPT),
            "sft_script_sha256": engine.file_sha256(SFT_SCRIPT),
            "v42a_source_contract": str(V42A_SFT_SCRIPT),
            "v42a_source_contract_sha256": engine.file_sha256(V42A_SFT_SCRIPT),
            "equal_unit_objective": str(OBJECTIVE_SCRIPT),
            "equal_unit_objective_sha256": engine.file_sha256(OBJECTIVE_SCRIPT),
        },
        "interpretation": (
            "matched-initialization equal-unit SFT retry is sealed; no "
            "shadow-dev, external validation, OOD, holdout, quality, or "
            "promotion conclusion is authorized"
        ),
    })
    report["recipe"]["loss_mode"] = "equal_conflict_unit_answer_token_mean"
    report["recipe"]["initialization"] = expected_initialization["source"]
    report["recipe"]["adapter_loader"] = expected_loader
    engine.atomic_write_json(report_path, engine.self_hashed(report))
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt.pop("content_sha256_before_self_field", None)
    attempt.update({
        "schema": "specialist-sft-matched-init-equal-unit-attempt-v42b",
        "source_initialization": expected_initialization["source"],
        "adapter_loader": expected_loader,
        "predecessor_failure": validate_failed_predecessor_v42b(),
        "cpu_model_load_smoke": validate_cpu_model_load_smoke_v42b(),
        "final_report_sha256": engine.file_sha256(report_path),
    })
    engine.atomic_write_json(attempt_path, engine.self_hashed(attempt))


def main(argv: list[str] | None = None) -> int:
    engine.SFT_SCRIPT = SFT_SCRIPT
    engine.EXPECTED_ENCODING_AUDIT = (
        v42a_runner.EXPECTED_ENCODING_AUDIT_V42A
    )
    engine.parser = parser
    engine.build_train_command = build_train_command
    engine.validate_preregistration = validate_preregistration
    engine.validate_output_artifacts = v42a_runner.validate_output_artifacts
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
