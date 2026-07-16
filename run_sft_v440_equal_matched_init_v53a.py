#!/usr/bin/env python3
"""Fail-closed four-GPU matched-init equal-weight SFT on v440 fold 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_train_refresh_v440_v53a as refresh
import run_sft_equal_unit_matched_init_v42a as v42a_launcher
import run_sft_equal_unit_matched_init_v47c as parent
import run_sft_train_only_control_v36a as engine
import sft_lora_equal_unit_matched_init_v42a as source_contract
import sft_lora_equal_unit_matched_init_v42b as loader_contract
import sft_lora_equal_unit_matched_init_v47a as schedule


ROOT = Path(__file__).resolve().parent
SFT_SCRIPT = parent.SFT_SCRIPT
BUILDER = (ROOT / "build_sft_v440_equal_preregistration_v53a.py").resolve()
TESTS = (ROOT / "test_sft_v440_equal_v53a.py").resolve()
LEARNING_RATE = 5.5e-5
EXPECTED_STEPS = 48
EXPECTED_ENCODING_AUDIT = {
    "prompt_mode": "es_exact",
    "eos_appended": False,
    "train_prompt_tokens": 22_340,
    "train_answer_tokens": 11_872,
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
    "identity_sha256": refresh.EXPECTED["weight_identity_sha256"],
}
MANIFEST_FILE_SHA256 = (
    "d6400a50f81867de70a5a79632793bac9ba2abf43307addc5e69abbca5b6bd87"
)


def parser() -> argparse.ArgumentParser:
    return parent.parser()


def _with_parent_contract(function, *args):
    old_script = parent.SFT_SCRIPT
    old_encoding = parent.EXPECTED_ENCODING_AUDIT
    old_weighting = parent.EXPECTED_WEIGHTING_AUDIT
    parent.SFT_SCRIPT = SFT_SCRIPT
    parent.EXPECTED_ENCODING_AUDIT = EXPECTED_ENCODING_AUDIT
    parent.EXPECTED_WEIGHTING_AUDIT = EXPECTED_WEIGHTING_AUDIT
    try:
        return function(*args)
    finally:
        parent.EXPECTED_WEIGHTING_AUDIT = old_weighting
        parent.EXPECTED_ENCODING_AUDIT = old_encoding
        parent.SFT_SCRIPT = old_script


def validate_recipe(args: argparse.Namespace) -> None:
    _with_parent_contract(parent.validate_recipe, args)


def build_train_command(args: argparse.Namespace) -> list[str]:
    return _with_parent_contract(parent.build_train_command, args)


def validate_refresh_manifest_v53a() -> dict:
    if engine.file_sha256(refresh.MANIFEST) != MANIFEST_FILE_SHA256:
        raise RuntimeError("V53A refresh manifest file changed")
    value = json.loads(refresh.MANIFEST.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    train = value.get("fold_3_train", {})
    access = value.get("access_firewall", {})
    if (
        content != refresh.EXPECTED["manifest_content_sha256"]
        or content != engine.canonical_sha256({
            key: item for key, item in value.items()
            if key != "content_sha256_before_self_field"
        })
        or value.get("schema")
        != "specialist-v440-train-only-fold3-refresh-v53a"
        or value.get("status")
        != "sealed_v440_projection_fold3_train_unlaunched"
        or value.get("projection", {}).get("sha256")
        != refresh.EXPECTED["projection_sha256"]
        or train.get("sha256") != refresh.EXPECTED["train_sha256"]
        or train.get("rows") != 448
        or train.get("conflict_units") != 208
        or train.get("root_membership_sha256")
        != refresh.EXPECTED["root_membership_sha256"]
        or train.get("membership_exactly_frozen_v412_fold3_train") is not True
        or value.get("equal_conflict_unit_weighting")
        != EXPECTED_WEIGHTING_AUDIT
        or access.get("shadow_artifact_opened") is not False
        or access.get("eval_ood_holdout_or_benchmark_opened") is not False
    ):
        raise RuntimeError("V53A refresh manifest content changed")
    return value


def implementation_bindings_v53a() -> dict[str, str]:
    paths = {
        "launcher_v53a": Path(__file__).resolve(),
        "builder_v53a": BUILDER,
        "tests_v53a": TESTS,
        "refresh_builder_v53a": Path(refresh.__file__).resolve(),
        "refresh_manifest_v53a": refresh.MANIFEST,
        "train_projection_v440": refresh.PROJECTION,
        "fold3_train_v440": refresh.TRAIN,
        "engine_v36a": Path(engine.__file__).resolve(),
        "launcher_parent_v47c": Path(parent.__file__).resolve(),
        "sft_schedule_v47a": Path(schedule.__file__).resolve(),
        "sft_source_v42a": Path(source_contract.__file__).resolve(),
        "sft_loader_v42b": Path(loader_contract.__file__).resolve(),
        "sft_script": SFT_SCRIPT,
        "model_config": engine.BASE_MODEL / "config.json",
        "model_index": engine.BASE_MODEL / "model.safetensors.index.json",
    }
    return {label: engine.file_sha256(path) for label, path in paths.items()}


def validate_preregistration(args, command) -> dict:
    path = Path(args.preregistration).resolve()
    if engine.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V53A preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    recipe = value.get("recipe", {})
    firewall = value.get("access_firewall", {})
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
        != "specialist-sft-v440-equal-train-only-preregistration-v53a"
        or value.get("status") != "sealed_unlaunched_train_only"
        or value.get("training_launch_authorized") is not True
        or value.get("evaluation_launch_authorized") is not False
        or value.get("contains_external_validation_ood_or_holdout_content")
        is not False
        or value.get("dataset", {}).get("path")
        != str(Path(args.dataset).resolve())
        or value.get("dataset", {}).get("sha256") != args.dataset_sha256
        or value.get("dataset", {}).get("rows") != 448
        or recipe.get("command") != command
        or recipe.get("expected_encoding_audit") != EXPECTED_ENCODING_AUDIT
        or recipe.get("expected_weighting_audit") != EXPECTED_WEIGHTING_AUDIT
        or recipe.get("expected_schedule_audit")
        != schedule.schedule_audit_v47a(EXPECTED_STEPS)
        or recipe.get("explicit_max_steps_cap") != EXPECTED_STEPS
        or recipe.get("expected_optimizer_steps") != EXPECTED_STEPS
        or recipe.get("learning_rate") != LEARNING_RATE
        or value.get("artifacts") != expected_artifacts
        or value.get("implementation", {}).get("file_sha256_bindings")
        != implementation_bindings_v53a()
        or firewall.get("shadow_artifact_opened") is not False
        or firewall.get("eval_ood_holdout_opened") is not False
        or firewall.get("post_training_evaluation_authorized") is not False
    ):
        raise RuntimeError("V53A sealed preregistration changed")
    initialization = source_contract.validate_initialization_artifact_v42a(
        Path(args.initial_adapter)
    )
    loader = loader_contract.expected_loader_audit_v42b()
    if (
        value.get("initialization") != initialization
        or value.get("adapter_loader") != loader
    ):
        raise RuntimeError("V53A matched initialization or loader changed")
    manifest = validate_refresh_manifest_v53a()
    return {
        "path": str(path),
        "file_sha256": args.preregistration_sha256,
        "content_sha256": content,
        "implementation_bindings": implementation_bindings_v53a(),
        "source_initialization": initialization,
        "adapter_loader": loader,
        "refresh_manifest_content_sha256": manifest[
            "content_sha256_before_self_field"
        ],
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
    observed_schedule = engine.extract_json_event(log_text, "schedule_audit_v47a")
    expected_initialization = (
        v42a_launcher.expected_initialization_runtime_audit_v42a()
    )
    expected_loader = loader_contract.expected_loader_audit_v42b()
    expected_schedule = schedule.schedule_audit_v47a(EXPECTED_STEPS)
    if (
        weighting["value"] != EXPECTED_WEIGHTING_AUDIT
        or initialization["value"] != expected_initialization
        or loader["value"] != expected_loader
        or observed_schedule["value"] != expected_schedule
    ):
        raise RuntimeError("V53A runtime audit changed")
    report.pop("content_sha256_before_self_field", None)
    report.update({
        "schema": "specialist-sft-v440-equal-train-only-runtime-v53a",
        "status": "complete_train_only_state_sealed_non_train_unopened",
        "validation_ood_or_holdout_opened": False,
        "shadow_artifact_opened": False,
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
        "explicit_step_cap": observed_schedule,
        "expected_optimizer_steps": EXPECTED_STEPS,
        "refresh_manifest": str(refresh.MANIFEST),
    })
    report["recipe"].update({
        "learning_rate": LEARNING_RATE,
        "loss_mode": "equal_conflict_unit_answer_token_mean",
        "initialization": expected_initialization["source"],
        "adapter_loader": expected_loader,
        "max_steps": EXPECTED_STEPS,
        "recipe_parent": "V49D equal exact recipe and matched initialization",
    })
    engine.atomic_write_json(report_path, engine.self_hashed(report))
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt.pop("content_sha256_before_self_field", None)
    attempt.update({
        "schema": "specialist-sft-v440-equal-train-only-attempt-v53a",
        "source_initialization": expected_initialization["source"],
        "adapter_loader": expected_loader,
        "weighting_identity_sha256": EXPECTED_WEIGHTING_AUDIT[
            "identity_sha256"
        ],
        "final_report_sha256": engine.file_sha256(report_path),
    })
    engine.atomic_write_json(attempt_path, engine.self_hashed(attempt))


def main(argv: list[str] | None = None) -> int:
    originals = (
        engine.SFT_SCRIPT,
        engine.EXPECTED_ENCODING_AUDIT,
        engine.parser,
        engine.build_train_command,
        engine.validate_preregistration,
        engine.validate_output_artifacts,
    )
    engine.SFT_SCRIPT = SFT_SCRIPT
    engine.EXPECTED_ENCODING_AUDIT = EXPECTED_ENCODING_AUDIT
    engine.parser = parser
    engine.build_train_command = build_train_command
    engine.validate_preregistration = validate_preregistration
    engine.validate_output_artifacts = v42a_launcher.validate_output_artifacts
    try:
        result = engine.main(argv)
    finally:
        (
            engine.SFT_SCRIPT,
            engine.EXPECTED_ENCODING_AUDIT,
            engine.parser,
            engine.build_train_command,
            engine.validate_preregistration,
            engine.validate_output_artifacts,
        ) = originals
    effective = parser().parse_args(argv) if argv is not None else parser().parse_args()
    if result == 0 and not effective.dry_run:
        rewrite_evidence(effective)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
