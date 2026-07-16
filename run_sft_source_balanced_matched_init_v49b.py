#!/usr/bin/env python3
"""Fail-closed V49B matched SFT launcher with V49A row weights only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_sft_equal_unit_matched_init_v42a as v42a
import run_sft_equal_unit_matched_init_v47c as v47c
import run_sft_train_only_control_v36a as engine
import seal_sft_source_balanced_input_v49b as sealed_input
import sft_lora_equal_unit_matched_init_v42a as source_contract
import sft_lora_equal_unit_matched_init_v42b as sft_v42b
import sft_lora_equal_unit_matched_init_v47a as schedule
import sft_source_balanced_weighting_v49b as weighting


ROOT = Path(__file__).resolve().parent
SFT_SCRIPT = (
    ROOT / "sft_lora_source_balanced_matched_init_v49b.py"
).resolve()
BUILDER = (
    ROOT / "build_sft_source_balanced_matched_init_preregistration_v49b.py"
).resolve()
TESTS = (ROOT / "test_sft_source_balanced_matched_init_v49b.py").resolve()
LEARNING_RATE = 5.5e-5
EXPECTED_STEPS = 48
EXPECTED_ENCODING_AUDIT = {
    "prompt_mode": "es_exact",
    "eos_appended": False,
    "train_prompt_tokens": 22_340,
    "train_answer_tokens": 11_746,
    "train_rows": 448,
}
WEIGHT_AUDIT_FILE_SHA256 = (
    "0a45915bcd47f8144068b2d40a3fc1dbaf6d8b2c6d3a5946ed74fade61e8543f"
)
WEIGHT_AUDIT_CONTENT_SHA256 = (
    "273912c3bf6502c720781ae78e058fabe1834eceb6e4d5a0b84a442b85a972cb"
)
INPUT_MANIFEST_FILE_SHA256 = (
    "dedf18048ac6774b15a4cf3bbfe122958a60573c5b3bc2eb51ae5c6fc13a16e4"
)
INPUT_MANIFEST_CONTENT_SHA256 = (
    "a2aaa30e5b4df42f4d3efc6a94b16fcc6ec3b796c97e3889c16f08a7e3a2a418"
)


def load_weighting_audit_v49b() -> tuple[dict, dict]:
    path = sealed_input.WEIGHT_AUDIT
    if engine.file_sha256(path) != WEIGHT_AUDIT_FILE_SHA256:
        raise RuntimeError("V49B sealed weighting audit file changed")
    complete = json.loads(path.read_text(encoding="utf-8"))
    content = complete.get("content_sha256_before_self_field")
    if (
        content != WEIGHT_AUDIT_CONTENT_SHA256
        or content != engine.canonical_sha256({
            key: item for key, item in complete.items()
            if key != "content_sha256_before_self_field"
        })
        or len(complete.get("per_row", [])) != 448
        or complete.get("identity_sha256")
        != weighting.ALTERNATIVE_NORMALIZED_WEIGHT_SHA256
        or complete.get("per_source_identity_sha256")
        != weighting.SOURCE_MASS_TABLE_SHA256
        or complete.get("access_firewall", {}).get(
            "shadow_semantics_opened"
        ) is not False
        or complete.get("access_firewall", {}).get(
            "eval_ood_holdout_semantics_opened"
        ) is not False
    ):
        raise RuntimeError("V49B sealed weighting audit content changed")
    return complete, weighting.compact_weighting_audit_v49b(complete)


EXPECTED_COMPLETE_WEIGHTING_AUDIT, EXPECTED_WEIGHTING_AUDIT = (
    load_weighting_audit_v49b()
)


def parser() -> argparse.ArgumentParser:
    return v47c.parser()


def _with_parent_contract(function, *args):
    old_script = v47c.SFT_SCRIPT
    old_weighting = v47c.EXPECTED_WEIGHTING_AUDIT
    v47c.SFT_SCRIPT = SFT_SCRIPT
    v47c.EXPECTED_WEIGHTING_AUDIT = EXPECTED_WEIGHTING_AUDIT
    try:
        return function(*args)
    finally:
        v47c.SFT_SCRIPT = old_script
        v47c.EXPECTED_WEIGHTING_AUDIT = old_weighting


def validate_recipe(args: argparse.Namespace) -> None:
    _with_parent_contract(v47c.validate_recipe, args)


def build_train_command(args: argparse.Namespace) -> list[str]:
    return _with_parent_contract(v47c.build_train_command, args)


def validate_input_manifest_v49b() -> dict:
    path = sealed_input.INPUT_MANIFEST
    if engine.file_sha256(path) != INPUT_MANIFEST_FILE_SHA256:
        raise RuntimeError("V49B sealed input manifest file changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    dataset = value.get("dataset", {})
    audit = value.get("weighting_audit", {})
    disjoint = value.get("document_disjoint_membership", {})
    if (
        content != INPUT_MANIFEST_CONTENT_SHA256
        or content != engine.canonical_sha256({
            key: item for key, item in value.items()
            if key != "content_sha256_before_self_field"
        })
        or value.get("schema")
        != "specialist-source-balanced-train-input-manifest-v49b"
        or value.get("status") != "sealed_train_only_before_launch"
        or dataset.get("path") != str(sealed_input.TRAIN)
        or dataset.get("file_sha256") != weighting.v49a.V434_TRAIN_SHA256
        or dataset.get("rows") != 448
        or dataset.get("conflict_units") != 208
        or dataset.get("root_membership_sha256")
        != weighting.ROOT_MEMBERSHIP_SHA256
        or dataset.get("membership_exactly_frozen_v412_fold3_train")
        is not True
        or audit.get("path") != str(sealed_input.WEIGHT_AUDIT)
        or audit.get("file_sha256") != WEIGHT_AUDIT_FILE_SHA256
        or audit.get("content_sha256") != WEIGHT_AUDIT_CONTENT_SHA256
        or audit.get("alternative_normalized_weight_sha256")
        != weighting.ALTERNATIVE_NORMALIZED_WEIGHT_SHA256
        or disjoint.get("confirmatory_fold") != 3
        or disjoint.get("train_dev_conflict_unit_intersection") != 0
        or any(disjoint.get("train_dev_edge_identity_intersections", {}).values())
        or disjoint.get("non_train_rows_opened") is not False
        or value.get("access_firewall", {}).get("shadow_semantics_opened")
        is not False
        or value.get("access_firewall", {}).get(
            "eval_ood_holdout_semantics_opened"
        ) is not False
    ):
        raise RuntimeError("V49B sealed input manifest content changed")
    return value


def implementation_bindings_v49b() -> dict[str, str]:
    paths = {
        "launcher_v49b": Path(__file__).resolve(),
        "sft_v49b": SFT_SCRIPT,
        "weighting_v49b": Path(weighting.__file__).resolve(),
        "input_sealer_v49b": Path(sealed_input.__file__).resolve(),
        "builder_v49b": BUILDER,
        "tests_v49b": TESTS,
        "engine_v36a": Path(engine.__file__).resolve(),
        "launcher_parent_v47c": Path(v47c.__file__).resolve(),
        "sft_schedule_v47a": Path(schedule.__file__).resolve(),
        "sft_source_v42a": Path(source_contract.__file__).resolve(),
        "sft_loader_v42b": Path(sft_v42b.__file__).resolve(),
        "v49a_design": weighting.V49A_DESIGN,
        "v49a_runtime": Path(weighting.v49a.__file__).resolve(),
        "train_v434": sealed_input.TRAIN,
        "weighting_audit": sealed_input.WEIGHT_AUDIT,
        "input_manifest": sealed_input.INPUT_MANIFEST,
        "fold_manifest": sealed_input.FOLD_MANIFEST,
        "v42i_preregistration": v47c.V42I_PREREGISTRATION,
        "model_config": engine.BASE_MODEL / "config.json",
        "model_index": engine.BASE_MODEL / "model.safetensors.index.json",
    }
    return {label: engine.file_sha256(path) for label, path in paths.items()}


def validate_preregistration(args, command) -> dict:
    path = Path(args.preregistration).resolve()
    if engine.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V49B preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    expected_artifacts = {
        "output_dir": str(Path(args.output_dir).resolve()),
        "stdout_log": str(Path(args.stdout_log).resolve()),
        "gpu_log": str(Path(args.gpu_log).resolve()),
        "report": str(Path(args.report).resolve()),
        "attempt_report": str(Path(args.attempt_report).resolve()),
    }
    recipe = value.get("recipe", {})
    firewall = value.get("access_firewall", {})
    if (
        content != args.preregistration_content_sha256
        or content != engine.canonical_sha256({
            key: item for key, item in value.items()
            if key != "content_sha256_before_self_field"
        })
        or value.get("schema")
        != "specialist-sft-source-balanced-preregistration-v49b"
        or value.get("status") != "sealed_unlaunched_train_only"
        or value.get("contains_external_validation_ood_or_holdout_content")
        is not False
        or value.get("dataset", {}).get("path")
        != str(Path(args.dataset).resolve())
        or value.get("dataset", {}).get("sha256") != args.dataset_sha256
        or value.get("dataset", {}).get("rows") != 448
        or value.get("input_manifest", {}).get("file_sha256")
        != INPUT_MANIFEST_FILE_SHA256
        or value.get("weighting_audit", {}).get("file_sha256")
        != WEIGHT_AUDIT_FILE_SHA256
        or value.get("weighting_audit", {}).get("runtime_compact")
        != EXPECTED_WEIGHTING_AUDIT
        or recipe.get("command") != command
        or recipe.get("expected_encoding_audit") != EXPECTED_ENCODING_AUDIT
        or recipe.get("expected_weighting_audit")
        != EXPECTED_WEIGHTING_AUDIT
        or recipe.get("expected_schedule_audit")
        != schedule.schedule_audit_v47a(EXPECTED_STEPS)
        or recipe.get("explicit_max_steps_cap") != EXPECTED_STEPS
        or recipe.get("expected_optimizer_steps") != EXPECTED_STEPS
        or recipe.get("learning_rate") != LEARNING_RATE
        or recipe.get("only_change_from_matched_parent")
        != "per-row example weights use the exact V49A alternative identity"
        or value.get("artifacts") != expected_artifacts
        or value.get("implementation", {}).get("file_sha256_bindings")
        != implementation_bindings_v49b()
        or firewall.get("training_input")
        != "sealed v434 projection of frozen v412 fold-3 train roots only"
        or firewall.get("shadow_semantics_opened_during_preregistration")
        is not False
        or firewall.get("shadow_semantics_opened_during_training") is not False
        or firewall.get("eval_ood_holdout_opened") is not False
        or firewall.get("post_training_evaluation_authorized") is not False
    ):
        raise RuntimeError("V49B sealed preregistration contract changed")
    input_manifest = validate_input_manifest_v49b()
    initialization = source_contract.validate_initialization_artifact_v42a(
        Path(args.initial_adapter)
    )
    loader = sft_v42b.expected_loader_audit_v42b()
    if (
        value.get("initialization") != initialization
        or value.get("adapter_loader") != loader
    ):
        raise RuntimeError("V49B matched initialization or loader changed")
    return {
        "path": str(path),
        "file_sha256": args.preregistration_sha256,
        "content_sha256": content,
        "implementation_bindings": implementation_bindings_v49b(),
        "source_initialization": initialization,
        "adapter_loader": loader,
        "input_manifest_content_sha256": input_manifest[
            "content_sha256_before_self_field"
        ],
        "weighting_audit_content_sha256": WEIGHT_AUDIT_CONTENT_SHA256,
    }


def rewrite_evidence(args: argparse.Namespace) -> None:
    report_path = Path(args.report).resolve()
    attempt_path = Path(args.attempt_report).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    log_text = Path(args.stdout_log).read_text(encoding="utf-8")
    observed_weighting = engine.extract_json_event(log_text, "weighting_audit")
    initialization = engine.extract_json_event(
        log_text, "source_initialization_audit"
    )
    loader = engine.extract_json_event(
        log_text, "initialization_loader_audit_v42b"
    )
    observed_schedule = engine.extract_json_event(
        log_text, "schedule_audit_v47a"
    )
    expected_initialization = v42a.expected_initialization_runtime_audit_v42a()
    expected_loader = sft_v42b.expected_loader_audit_v42b()
    expected_schedule = schedule.schedule_audit_v47a(EXPECTED_STEPS)
    if (
        observed_weighting["value"] != EXPECTED_WEIGHTING_AUDIT
        or initialization["value"] != expected_initialization
        or loader["value"] != expected_loader
        or observed_schedule["value"] != expected_schedule
    ):
        raise RuntimeError("V49B runtime audit changed")
    report.pop("content_sha256_before_self_field", None)
    report.update({
        "schema": "specialist-sft-source-balanced-runtime-v49b",
        "status": "complete_train_only_state_sealed_non_train_unopened",
        "selection_surface": "source_balanced_v434_frozen_fold3_train_only",
        "validation_ood_or_holdout_opened": False,
        "shadow_semantics_opened": False,
        "observed_weighting_audit": observed_weighting,
        "sealed_weighting_audit": {
            "path": str(sealed_input.WEIGHT_AUDIT),
            "file_sha256": WEIGHT_AUDIT_FILE_SHA256,
            "content_sha256": WEIGHT_AUDIT_CONTENT_SHA256,
            "per_row_identity_sha256": EXPECTED_WEIGHTING_AUDIT[
                "per_row_identity_sha256"
            ],
            "per_source_identity_sha256": EXPECTED_WEIGHTING_AUDIT[
                "per_source_identity_sha256"
            ],
            "per_category_identity_sha256": EXPECTED_WEIGHTING_AUDIT[
                "per_category_identity_sha256"
            ],
        },
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
        "input_manifest": str(sealed_input.INPUT_MANIFEST),
    })
    report["recipe"].update({
        "learning_rate": LEARNING_RATE,
        "loss_mode": "V49A_source_balanced_answer_token_example_mean",
        "initialization": expected_initialization["source"],
        "adapter_loader": expected_loader,
        "max_steps": EXPECTED_STEPS,
        "recipe_parent": "V47C/V42I exact recipe except per-row weights",
    })
    engine.atomic_write_json(report_path, engine.self_hashed(report))
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt.pop("content_sha256_before_self_field", None)
    attempt.update({
        "schema": "specialist-sft-source-balanced-attempt-v49b",
        "source_initialization": expected_initialization["source"],
        "adapter_loader": expected_loader,
        "weighting_identity_sha256": weighting.ALTERNATIVE_NORMALIZED_WEIGHT_SHA256,
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
