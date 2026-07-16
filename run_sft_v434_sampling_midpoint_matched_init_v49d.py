#!/usr/bin/env python3
"""Fail-closed launcher for either matched V49D v434 weighting arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_sft_equal_unit_matched_init_v42a as v42a
import run_sft_equal_unit_matched_init_v47c as v47c
import run_sft_train_only_control_v36a as engine
import seal_sft_v434_sampling_midpoint_input_v49d as sealed_input
import sft_lora_equal_unit_matched_init_v42a as source_contract
import sft_lora_equal_unit_matched_init_v42b as sft_v42b
import sft_lora_equal_unit_matched_init_v47a as schedule
import sft_v434_sampling_midpoint_weighting_v49d as weighting


ROOT = Path(__file__).resolve().parent
ARMS = weighting.ARMS
SFT_SCRIPTS = {
    "v434_equal": (ROOT / "sft_lora_v434_equal_matched_init_v49d.py").resolve(),
    "v434_source50": (
        ROOT / "sft_lora_v434_source50_matched_init_v49d.py"
    ).resolve(),
}
BUILDER = (
    ROOT / "build_sft_v434_sampling_midpoint_preregistration_v49d.py"
).resolve()
FUTURE_EVAL_BUILDER = (
    ROOT / "build_sft_v434_sampling_midpoint_future_eval_v49d.py"
).resolve()
TESTS = (ROOT / "test_sft_v434_sampling_midpoint_v49d.py").resolve()
LEARNING_RATE = 5.5e-5
EXPECTED_STEPS = 48
EXPECTED_ENCODING_AUDIT = {
    "prompt_mode": "es_exact",
    "eos_appended": False,
    "train_prompt_tokens": 22_340,
    "train_answer_tokens": 11_746,
    "train_rows": 448,
}
INPUT_MANIFEST_FILE_SHA256 = (
    "fdcec5e5d5e888cd9dd5d1c3bf7886c4a2c9ee62dc0a19e7c61790ab00918be8"
)
INPUT_MANIFEST_CONTENT_SHA256 = (
    "fb0f3484e785822836bcda40843dd5c9fcf823501cb2d5ae792745eadb6a5d81"
)
WEIGHT_AUDIT_HASHES = {
    "v434_equal": {
        "file": "14806537002ea9caabd83b3df19aeecd56103934cfdd5a3b8c9c45610775a4d2",
        "content": "07582b3652c08209e5ced3e58d17ef125ac71bad2539b0e19a34a7062424907c",
    },
    "v434_source50": {
        "file": "e27f4c446c261fc7e1e2209a4d196a24ba4af740981226c0e55c228b9779e056",
        "content": "06981d1960cca0f0ade861c7952300ccb8b0db2f6fbf70d7a2408cc354515b41",
    },
}


def load_weighting_audit_v49d(arm: str) -> tuple[dict, dict]:
    if arm not in ARMS:
        raise ValueError(f"unsupported V49D arm: {arm}")
    path = sealed_input.WEIGHT_AUDITS[arm]
    hashes = WEIGHT_AUDIT_HASHES[arm]
    if engine.file_sha256(path) != hashes["file"]:
        raise RuntimeError(f"V49D {arm} weighting audit file changed")
    complete = json.loads(path.read_text(encoding="utf-8"))
    content = complete.get("content_sha256_before_self_field")
    expected = weighting.EXPECTED[arm]
    if (
        content != hashes["content"]
        or content != engine.canonical_sha256({
            key: value for key, value in complete.items()
            if key != "content_sha256_before_self_field"
        })
        or len(complete.get("per_row", [])) != 448
        or complete.get("identity_sha256")
        != expected["normalized_weight_sha256"]
        or complete.get("trainer_example_weight_identity_sha256")
        != expected["trainer_weight_sha256"]
        or complete.get("per_row_identity_sha256") != expected["per_row_sha256"]
        or complete.get("per_source_identity_sha256")
        != expected["per_source_sha256"]
        or complete.get("per_category_identity_sha256")
        != expected["per_category_sha256"]
        or complete.get("access_firewall", {}).get("shadow_semantics_opened")
        is not False
        or complete.get("access_firewall", {}).get(
            "eval_ood_holdout_semantics_opened"
        ) is not False
    ):
        raise RuntimeError(f"V49D {arm} weighting audit content changed")
    return complete, weighting.compact_weighting_audit_v49d(complete)


EXPECTED_COMPLETE_AUDITS = {}
EXPECTED_WEIGHTING_AUDITS = {}
for _arm in ARMS:
    _complete, _compact = load_weighting_audit_v49d(_arm)
    EXPECTED_COMPLETE_AUDITS[_arm] = _complete
    EXPECTED_WEIGHTING_AUDITS[_arm] = _compact


def parser() -> argparse.ArgumentParser:
    result = v47c.parser()
    result.add_argument("--arm", required=True, choices=ARMS)
    return result


def _with_parent_contract(arm: str, function, *args):
    old_script = v47c.SFT_SCRIPT
    old_weighting = v47c.EXPECTED_WEIGHTING_AUDIT
    v47c.SFT_SCRIPT = SFT_SCRIPTS[arm]
    v47c.EXPECTED_WEIGHTING_AUDIT = EXPECTED_WEIGHTING_AUDITS[arm]
    try:
        return function(*args)
    finally:
        v47c.SFT_SCRIPT = old_script
        v47c.EXPECTED_WEIGHTING_AUDIT = old_weighting


def validate_recipe(args: argparse.Namespace) -> None:
    _with_parent_contract(args.arm, v47c.validate_recipe, args)


def build_train_command(args: argparse.Namespace) -> list[str]:
    return _with_parent_contract(args.arm, v47c.build_train_command, args)


def validate_input_manifest_v49d() -> dict:
    path = sealed_input.INPUT_MANIFEST
    if engine.file_sha256(path) != INPUT_MANIFEST_FILE_SHA256:
        raise RuntimeError("V49D sealed input manifest file changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    dataset = value.get("dataset", {})
    disjoint = value.get("document_disjoint_membership", {})
    if (
        content != INPUT_MANIFEST_CONTENT_SHA256
        or content != engine.canonical_sha256({
            key: item for key, item in value.items()
            if key != "content_sha256_before_self_field"
        })
        or value.get("schema")
        != "specialist-v434-sampling-midpoint-input-manifest-v49d"
        or value.get("status") != "sealed_train_only_before_launch"
        or dataset.get("path") != str(sealed_input.TRAIN)
        or dataset.get("file_sha256") != weighting.v49b.v49a.V434_TRAIN_SHA256
        or dataset.get("rows") != 448
        or dataset.get("conflict_units") != 208
        or dataset.get("root_membership_sha256")
        != weighting.v49b.ROOT_MEMBERSHIP_SHA256
        or dataset.get("same_exact_bytes_for_both_arms") is not True
        or value.get("arm_order") != list(ARMS)
        or any(
            value.get("weighting_audits", {}).get(arm, {}).get("file_sha256")
            != WEIGHT_AUDIT_HASHES[arm]["file"]
            for arm in ARMS
        )
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
        raise RuntimeError("V49D sealed input manifest content changed")
    return value


def implementation_bindings_v49d() -> dict[str, str]:
    paths = {
        "launcher_v49d": Path(__file__).resolve(),
        "sft_equal_v49d": SFT_SCRIPTS["v434_equal"],
        "sft_source50_v49d": SFT_SCRIPTS["v434_source50"],
        "weighting_v49d": Path(weighting.__file__).resolve(),
        "input_sealer_v49d": Path(sealed_input.__file__).resolve(),
        "builder_v49d": BUILDER,
        "future_eval_builder_v49d": FUTURE_EVAL_BUILDER,
        "tests_v49d": TESTS,
        "engine_v36a": Path(engine.__file__).resolve(),
        "launcher_parent_v47c": Path(v47c.__file__).resolve(),
        "sft_schedule_v47a": Path(schedule.__file__).resolve(),
        "sft_source_v42a": Path(source_contract.__file__).resolve(),
        "sft_loader_v42b": Path(sft_v42b.__file__).resolve(),
        "v49b_weighting_parent": Path(weighting.v49b.__file__).resolve(),
        "v49a_design": weighting.v49b.V49A_DESIGN,
        "train_v434": sealed_input.TRAIN,
        "weighting_audit_equal": sealed_input.WEIGHT_AUDITS["v434_equal"],
        "weighting_audit_source50": sealed_input.WEIGHT_AUDITS["v434_source50"],
        "input_manifest": sealed_input.INPUT_MANIFEST,
        "fold_manifest": sealed_input.v49b_input.FOLD_MANIFEST,
        "v42i_preregistration": v47c.V42I_PREREGISTRATION,
        "model_config": engine.BASE_MODEL / "config.json",
        "model_index": engine.BASE_MODEL / "model.safetensors.index.json",
    }
    return {label: engine.file_sha256(path) for label, path in paths.items()}


def validate_preregistration(args, command) -> dict:
    path = Path(args.preregistration).resolve()
    if engine.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V49D preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    arm = value.get("training_arms", {}).get(args.arm, {})
    expected_artifacts = {
        "output_dir": str(Path(args.output_dir).resolve()),
        "stdout_log": str(Path(args.stdout_log).resolve()),
        "gpu_log": str(Path(args.gpu_log).resolve()),
        "report": str(Path(args.report).resolve()),
        "attempt_report": str(Path(args.attempt_report).resolve()),
    }
    recipe = arm.get("recipe", {})
    firewall = value.get("access_firewall", {})
    if (
        content != args.preregistration_content_sha256
        or content != engine.canonical_sha256({
            key: item for key, item in value.items()
            if key != "content_sha256_before_self_field"
        })
        or value.get("schema")
        != "specialist-v434-sampling-midpoint-preregistration-v49d"
        or value.get("status") != "sealed_unlaunched_train_only"
        or value.get("training_launch_authorized") is not True
        or value.get("evaluation_launch_authorized") is not False
        or value.get("contains_external_validation_ood_or_holdout_content")
        is not False
        or value.get("dataset", {}).get("path")
        != str(Path(args.dataset).resolve())
        or value.get("dataset", {}).get("sha256") != args.dataset_sha256
        or value.get("dataset", {}).get("rows") != 448
        or value.get("input_manifest", {}).get("file_sha256")
        != INPUT_MANIFEST_FILE_SHA256
        or arm.get("weighting_audit", {}).get("file_sha256")
        != WEIGHT_AUDIT_HASHES[args.arm]["file"]
        or arm.get("weighting_audit", {}).get("runtime_compact")
        != EXPECTED_WEIGHTING_AUDITS[args.arm]
        or recipe.get("command") != command
        or recipe.get("expected_encoding_audit") != EXPECTED_ENCODING_AUDIT
        or recipe.get("expected_weighting_audit")
        != EXPECTED_WEIGHTING_AUDITS[args.arm]
        or recipe.get("expected_schedule_audit")
        != schedule.schedule_audit_v47a(EXPECTED_STEPS)
        or recipe.get("explicit_max_steps_cap") != EXPECTED_STEPS
        or recipe.get("expected_optimizer_steps") != EXPECTED_STEPS
        or recipe.get("learning_rate") != LEARNING_RATE
        or arm.get("artifacts") != expected_artifacts
        or value.get("implementation", {}).get("file_sha256_bindings")
        != implementation_bindings_v49d()
        or firewall.get("shadow_semantics_opened_during_preregistration")
        is not False
        or firewall.get("eval_ood_holdout_opened") is not False
        or firewall.get("post_training_evaluation_authorized") is not False
    ):
        raise RuntimeError(f"V49D {args.arm} sealed preregistration changed")
    input_manifest = validate_input_manifest_v49d()
    initialization = source_contract.validate_initialization_artifact_v42a(
        Path(args.initial_adapter)
    )
    loader = sft_v42b.expected_loader_audit_v42b()
    if value.get("initialization") != initialization or value.get(
        "adapter_loader"
    ) != loader:
        raise RuntimeError("V49D matched initialization or loader changed")
    return {
        "path": str(path),
        "file_sha256": args.preregistration_sha256,
        "content_sha256": content,
        "arm": args.arm,
        "implementation_bindings": implementation_bindings_v49d(),
        "source_initialization": initialization,
        "adapter_loader": loader,
        "input_manifest_content_sha256": input_manifest[
            "content_sha256_before_self_field"
        ],
        "weighting_audit_content_sha256": WEIGHT_AUDIT_HASHES[args.arm][
            "content"
        ],
    }


def rewrite_evidence(args: argparse.Namespace) -> None:
    report_path = Path(args.report).resolve()
    attempt_path = Path(args.attempt_report).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    log_text = Path(args.stdout_log).read_text(encoding="utf-8")
    observed_weighting = engine.extract_json_event(log_text, "weighting_audit")
    initialization = engine.extract_json_event(log_text, "source_initialization_audit")
    loader = engine.extract_json_event(log_text, "initialization_loader_audit_v42b")
    observed_schedule = engine.extract_json_event(log_text, "schedule_audit_v47a")
    expected_initialization = v42a.expected_initialization_runtime_audit_v42a()
    expected_loader = sft_v42b.expected_loader_audit_v42b()
    expected_schedule = schedule.schedule_audit_v47a(EXPECTED_STEPS)
    if (
        observed_weighting["value"] != EXPECTED_WEIGHTING_AUDITS[args.arm]
        or initialization["value"] != expected_initialization
        or loader["value"] != expected_loader
        or observed_schedule["value"] != expected_schedule
    ):
        raise RuntimeError(f"V49D {args.arm} runtime audit changed")
    report.pop("content_sha256_before_self_field", None)
    report.update({
        "schema": "specialist-v434-sampling-midpoint-runtime-v49d",
        "arm": args.arm,
        "status": "complete_train_only_state_sealed_non_train_unopened",
        "validation_ood_or_holdout_opened": False,
        "shadow_semantics_opened": False,
        "observed_weighting_audit": observed_weighting,
        "sealed_weighting_audit": {
            "path": str(sealed_input.WEIGHT_AUDITS[args.arm]),
            "file_sha256": WEIGHT_AUDIT_HASHES[args.arm]["file"],
            "content_sha256": WEIGHT_AUDIT_HASHES[args.arm]["content"],
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
        "loss_mode": f"{args.arm}_answer_token_example_mean",
        "initialization": expected_initialization["source"],
        "adapter_loader": expected_loader,
        "max_steps": EXPECTED_STEPS,
        "recipe_parent": "V47C/V42I exact recipe except per-row weights",
    })
    engine.atomic_write_json(report_path, engine.self_hashed(report))
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt.pop("content_sha256_before_self_field", None)
    attempt.update({
        "schema": "specialist-v434-sampling-midpoint-attempt-v49d",
        "arm": args.arm,
        "source_initialization": expected_initialization["source"],
        "adapter_loader": expected_loader,
        "weighting_identity_sha256": weighting.EXPECTED[args.arm][
            "normalized_weight_sha256"
        ],
        "final_report_sha256": engine.file_sha256(report_path),
    })
    engine.atomic_write_json(attempt_path, engine.self_hashed(attempt))


def main(argv: list[str] | None = None) -> int:
    originals = (
        engine.SFT_SCRIPT, engine.EXPECTED_ENCODING_AUDIT, engine.parser,
        engine.build_train_command, engine.validate_preregistration,
        engine.validate_output_artifacts,
    )
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
