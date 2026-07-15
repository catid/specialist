#!/usr/bin/env python3
"""Fail-closed fresh-process driver for the V16 fused-MoE task A/B."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import eggroll_es_fused_moe_task_ab_preregistration_v16 as prereg_v16
import run_eggroll_es_train_panels_v13 as driver_v13
import train_eggroll_es_fused_moe_task_ab_v16 as trainer_v16
import train_eggroll_es_specialist as base
import train_eggroll_es_specialist_anchor_v11 as anchor_v11
import train_eggroll_es_specialist_anchor_v13 as anchor_v13


ROOT = Path(__file__).resolve().parent
FROZEN_MODEL_V16 = driver_v13.FROZEN_MODEL_V13
FROZEN_OUTPUT_DIRECTORY_V16 = driver_v13.FROZEN_OUTPUT_DIRECTORY_V13
EXPERIMENT_NAME_V16 = prereg_v16.EXPERIMENT_NAME_V16
REPORT_NAME_V16 = "fused_moe_task_ab_v16.json"
TEST_PATH_V16 = (ROOT / "test_eggroll_es_fused_moe_task_ab_runtime_v16.py").resolve()
RUNTIME_TRAINER_PATH_V16 = (
    ROOT / "train_eggroll_es_fused_moe_task_ab_v16.py"
).resolve()
PREREG_TEST_PATH_V16 = (
    ROOT / "test_eggroll_es_fused_moe_task_ab_preregistration_v16.py"
).resolve()
PREREGISTRATION_FILE_SHA256_V16 = (
    "53d796cee9c0ef67fdcb549f9bda55a978dd2840812d6a0ae0a8a4e363d24853"
)
PREREGISTRATION_CONTENT_SHA256_V16 = (
    "82569802ad89a0c3c92e4bb0a28a2db867a0bcb01ab7268ff9ab6048558a115c"
)
PREREG_MODULE_FILE_SHA256_V16 = (
    "7735fd46837d5b281badc785d70671879101f6520d0dc40307e1e9a0c4c1fe51"
)
PREREG_TEST_FILE_SHA256_V16 = (
    "b765e39fd9a6f415b72d4d6328b7ea5ce2781a0072eb2b2e53162ee8cd883cbd"
)
PROTOCOL_FILE_SHA256_V16 = (
    "c013b595857653830c54353f89a716193518d23f3c52dc355875f95dcf12df88"
)
PHASE1_PATHS_V16 = {
    "prereg_module_v16": Path(prereg_v16.__file__).resolve(),
    "prereg_tests_v16": PREREG_TEST_PATH_V16,
    "preregistration_v16": prereg_v16.PREREGISTRATION_PATH_V16,
    "protocol_v16": prereg_v16.PROTOCOL_PATH_V16,
}
PHASE1_HASHES_V16 = {
    "prereg_module_v16": PREREG_MODULE_FILE_SHA256_V16,
    "prereg_tests_v16": PREREG_TEST_FILE_SHA256_V16,
    "preregistration_v16": PREREGISTRATION_FILE_SHA256_V16,
    "protocol_v16": PROTOCOL_FILE_SHA256_V16,
}
IMPLEMENTATION_PATHS_V16 = {
    **driver_v13.IMPLEMENTATION_PATHS_V13,
    **PHASE1_PATHS_V16,
    "runtime_trainer_v16": RUNTIME_TRAINER_PATH_V16,
    "runtime_driver_v16": Path(__file__).resolve(),
    "runtime_tests_v16": TEST_PATH_V16,
}
FORBIDDEN_SURFACE_TOKENS_V16 = (
    "heldout", "validation", "ood", "benchmark-outcome", "eval-output",
)
MOE_CONFOUNDING_ENV_V16 = (
    "VLLM_ROCM_USE_AITER", "VLLM_ROCM_USE_AITER_MOE",
    "VLLM_USE_DEEP_GEMM", "VLLM_MOE_USE_DEEP_GEMM",
)
TUNED_ENV_NAME_V16 = "VLLM_TUNED_CONFIG_FOLDER"


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    return prereg_v16.canonical_sha256(value)


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _assert_train_only_argv_v16(argv):
    for token in argv:
        lowered = str(token).lower()
        if any(value in lowered for value in FORBIDDEN_SURFACE_TOKENS_V16):
            raise ValueError(f"v16 rejects non-train surface: {token}")


def implementation_identity_v16():
    inherited = driver_v13.implementation_identity_v13()
    files = {
        key: {
            "path": str(Path(path).resolve()),
            "file_sha256": _file_sha256(path),
        }
        for key, path in IMPLEMENTATION_PATHS_V16.items()
    }
    if {key: files[key] for key in inherited["files"]} != inherited["files"]:
        raise RuntimeError("v16 exact inherited V13 implementation changed")
    if {
        key: files[key]["file_sha256"] for key in PHASE1_HASHES_V16
    } != PHASE1_HASHES_V16:
        raise RuntimeError("v16 committed preregistration phase changed")
    if {
        key: files[key]["file_sha256"]
        for key in prereg_v16.V13_IMPLEMENTATION_HASHES_V16
    } != prereg_v16.V13_IMPLEMENTATION_HASHES_V16:
        raise RuntimeError("v16 frozen V13 runtime identity changed")
    return {"files": files, "bundle_sha256": _canonical(files)}


def _source_provenance_v16(implementation):
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    committed = {}
    for key, item in implementation["files"].items():
        path = Path(item["path"]).resolve()
        relative = path.relative_to(ROOT).as_posix()
        try:
            raw = subprocess.check_output(
                ["git", "show", f"{head}:{relative}"], cwd=ROOT,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"v16 real launch requires committed implementation: {relative}"
            ) from error
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item["file_sha256"]:
            raise RuntimeError(f"v16 source differs from committed HEAD: {relative}")
        committed[key] = {"relative_path": relative, "file_sha256": digest}
    result = {
        "schema": "eggroll-es-committed-source-bundle-v16",
        "git_head": head,
        "files": committed,
        "implementation_bundle_sha256": implementation["bundle_sha256"],
    }
    result["content_sha256_before_self_field"] = _canonical(result)
    return result


def _parser_v16():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v16-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-recipe-sha256")
    parser.add_argument(
        "--model-name", default=str(FROZEN_MODEL_V16),
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--sigma", type=float, default=0.0003)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=56)
    parser.add_argument("--mini-batch-size", type=int, default=56)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--n-vllm-engines", type=int, default=4)
    parser.add_argument("--n-gpu-per-vllm-engine", type=int, default=1)
    parser.add_argument("--use-gpus", default="0,1,2,3")
    parser.add_argument("--reward-function-timeout", type=int, default=10)
    parser.add_argument(
        "--output-directory", default=str(FROZEN_OUTPUT_DIRECTORY_V16),
    )
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME_V16)
    parser.add_argument("--logging", choices=("none",), default="none")
    parser.add_argument("--wandb-project", default="specialist-eggroll-es")
    parser.add_argument(
        "--internal-arm-worker", choices=prereg_v16.ARM_ORDER_V16,
    )
    parser.add_argument("--internal-arm-token")
    parser.add_argument("--internal-arm-output")
    return parser


def _declared_parent_moe_environment_v16():
    return {
        "moe_backend": "triton",
        "vllm_tuned_config_folder": None,
        "confounding_backend_environment": {
            name: None for name in MOE_CONFOUNDING_ENV_V16
        },
        "backend_bound_by_committed_engine_recipe": True,
    }


def _parent_environment_v16():
    conflicts = {
        name: os.environ.get(name)
        for name in (TUNED_ENV_NAME_V16, *MOE_CONFOUNDING_ENV_V16)
        if os.environ.get(name) not in (None, "")
    }
    if conflicts:
        raise ValueError("v16 parent requires all MoE override environment unset")
    return _declared_parent_moe_environment_v16()


def _arm_environment_v16(arm, base_environment=None):
    if arm not in prereg_v16.ARM_ORDER_V16:
        raise ValueError("v16 unknown arm")
    environment = dict(os.environ if base_environment is None else base_environment)
    for name in MOE_CONFOUNDING_ENV_V16:
        if environment.get(name) not in (None, ""):
            raise ValueError("v16 confounding MoE backend override is set")
        environment.pop(name, None)
    environment.pop(TUNED_ENV_NAME_V16, None)
    if arm == "tuned_triton":
        environment[TUNED_ENV_NAME_V16] = str(prereg_v16.TUNING_DIRECTORY_V16)
    return environment


def _arm_environment_difference_v16(default_environment, tuned_environment):
    keys = set(default_environment) | set(tuned_environment)
    difference = {
        key: (default_environment.get(key), tuned_environment.get(key))
        for key in keys
        if default_environment.get(key) != tuned_environment.get(key)
    }
    expected = {
        TUNED_ENV_NAME_V16: (
            None, str(prereg_v16.TUNING_DIRECTORY_V16),
        )
    }
    if difference != expected:
        raise RuntimeError("v16 arm environments differ by more than tuned folder")
    return {
        "only_difference": TUNED_ENV_NAME_V16,
        "default": None,
        "tuned": str(prereg_v16.TUNING_DIRECTORY_V16),
        "exact_difference_verified": True,
    }


def validate_runtime_v16(args, bundle, implementation):
    anchor_v13.validate_frozen_layer_plan_bundle_v13(bundle)
    plan = anchor_v11.FROZEN_STABILITY_PLANS_V11[
        driver_v13.driver_v10.MIDDLE_LATE_PLAN_SHA256_V10
    ]
    internal = args.internal_arm_worker is not None
    if not internal:
        _parent_environment_v16()
    elif any(
        os.environ.get(name) not in (None, "")
        for name in MOE_CONFOUNDING_ENV_V16
    ):
        raise ValueError("v16 internal arm has a confounding MoE override")
    internal_complete = (
        internal and args.internal_arm_token and args.internal_arm_output
        and args.expected_recipe_sha256
    )
    if (
        bundle["plan_sha256"]
        != driver_v13.driver_v10.MIDDLE_LATE_PLAN_SHA256_V10
        or Path(bundle["path"]).resolve() != Path(plan["path"]).resolve()
        or bundle["file_sha256"] != plan["file_sha256"]
        or Path(args.model_name).resolve() != FROZEN_MODEL_V16
        or args.checkpoint is not None
        or args.sigma != 0.0003
        or args.population_size != 32
        or args.batch_size != 56
        or args.mini_batch_size != 56
        or args.max_tokens != 1
        or args.seed != 43
        or args.alpha != 0.0
        or args.n_vllm_engines != 4
        or args.n_gpu_per_vllm_engine != 1
        or args.use_gpus != "0,1,2,3"
        or args.reward_function_timeout != 10
        or Path(args.output_directory).resolve() != FROZEN_OUTPUT_DIRECTORY_V16
        or args.experiment_name != EXPERIMENT_NAME_V16
        or args.logging != "none"
        or args.wandb_project != "specialist-eggroll-es"
        or (internal and not internal_complete)
        or (not internal and any((args.internal_arm_token, args.internal_arm_output)))
    ):
        raise ValueError("v16 frozen alpha-zero four-GPU task recipe changed")
    expected = args.expected_implementation_bundle_sha256
    if not args.v16_dry_run and expected is None:
        raise ValueError("v16 real launch requires implementation bundle hash")
    if expected is not None and expected != implementation["bundle_sha256"]:
        raise ValueError("v16 implementation bundle hash changed")


def recipe_v16(bundle, implementation):
    frozen = prereg_v16.build_preregistration_v16()
    recipe = {
        "schema": "eggroll-es-fused-moe-task-ab-recipe-v16",
        "experiment_name": EXPERIMENT_NAME_V16,
        "model": str(FROZEN_MODEL_V16),
        "layer_plan": {
            key: bundle[key]
            for key in ("path", "file_sha256", "plan_sha256", "model_config_sha256")
        },
        "task": copy.deepcopy(frozen["task"]),
        "hardware": copy.deepcopy(frozen["hardware"]),
        "arm_order": list(prereg_v16.ARM_ORDER_V16),
        "arms": copy.deepcopy(frozen["arms"]),
        "fresh_process_per_arm": True,
        "timing_protocol": copy.deepcopy(frozen["timing_protocol"]),
        "promotion_gate": copy.deepcopy(frozen["promotion_gate"]),
        "preregistration": {
            "path": str(prereg_v16.PREREGISTRATION_PATH_V16),
            "file_sha256": PREREGISTRATION_FILE_SHA256_V16,
            "content_sha256": PREREGISTRATION_CONTENT_SHA256_V16,
        },
        "parent_moe_environment": _declared_parent_moe_environment_v16(),
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "model_update_allowed": False,
        "evaluation_allowed": False,
        "raw_content_persisted": False,
    }
    recipe["content_sha256_before_self_field"] = _canonical(recipe)
    return recipe


def _seal(value):
    value.pop("content_sha256_before_self_field", None)
    value["content_sha256_before_self_field"] = _canonical(value)
    return value


def _exclusive_write_json(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    _seal(value)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError(f"v16 exclusive output already exists: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _rewrite_json(path, value):
    _seal(value)
    driver_v13.driver_v1.atomic_write_json(path, value)


def _attempt_path_v16():
    return FROZEN_OUTPUT_DIRECTORY_V16 / f".{EXPERIMENT_NAME_V16}.launch_attempt.json"


def _layer_cli_v16(bundle):
    return [
        "--layer-plan-json", bundle["path"],
        "--expected-layer-plan-file-sha256", bundle["file_sha256"],
        "--expected-layer-plan-sha256", bundle["plan_sha256"],
        "--expected-model-config-sha256", bundle["model_config_sha256"],
    ]


def _arm_token_v16(recipe_hash, implementation_hash, arm):
    return _canonical({
        "schema": "eggroll-es-internal-arm-token-v16",
        "recipe_sha256": recipe_hash,
        "implementation_bundle_sha256": implementation_hash,
        "arm": arm,
    })


def _make_trainer_v16(args, bundle):
    trainer_class = trainer_v16.load_trainer(bundle)
    return trainer_class(
        model_name=args.model_name, checkpoint=None, sigma=0.0003, alpha=0.0,
        population_size=32, reward_shaping="z-scores", num_iterations=1,
        max_tokens=1, batch_size=56, mini_batch_size=56,
        reward_function=base.specialist_reward,
        template_function=base.specialist_template,
        train_dataloader=[], eval_dataloader_dict={}, eval_freq=1,
        n_vllm_engines=4, n_gpu_per_vllm_engine=1, logging="none",
        global_seed=43, use_gpus="0,1,2,3", experiment_name=EXPERIMENT_NAME_V16,
        wandb_project="specialist-eggroll-es", save_best_models=False,
        reward_function_timeout=10, output_directory=args.output_directory,
    )


def _run_arm_worker_v16(args, bundle, implementation, recipe):
    arm = args.internal_arm_worker
    recipe_hash = recipe["content_sha256_before_self_field"]
    expected_token = _arm_token_v16(
        recipe_hash, implementation["bundle_sha256"], arm,
    )
    expected_folder = (
        None if arm == "default_triton"
        else str(prereg_v16.TUNING_DIRECTORY_V16)
    )
    actual_folder = os.environ.get(TUNED_ENV_NAME_V16)
    if (
        args.v16_dry_run
        or args.expected_recipe_sha256 != recipe_hash
        or args.internal_arm_token != expected_token
        or actual_folder != expected_folder
        or any(os.environ.get(name) not in (None, "") for name in MOE_CONFOUNDING_ENV_V16)
    ):
        raise ValueError("v16 internal arm authorization or environment changed")
    driver_v13.validate_arrow_train_v13(driver_v13.FROZEN_TRAIN_DATASET_V13)
    panels = anchor_v13.load_panel_bundle_v13()
    trainer = None
    failure = None
    try:
        base.set_seed(43)
        trainer = _make_trainer_v16(args, bundle)
        trainer.configure_task_ab_v16(panels, frozen_layer_plan=bundle)
        diagnostic, timing = trainer.estimate_task_ab_v16(
            anchor_v13.PERTURBATION_SEEDS_V13,
        )
        compact = trainer_v16.compact_arm_v16(diagnostic, timing)
    except BaseException as error:
        failure = error
        raise
    finally:
        if trainer is not None:
            try:
                base.close_trainer(trainer)
            except BaseException:
                if failure is None:
                    raise
    envelope = {
        "schema": "eggroll-es-fused-moe-task-arm-v16",
        "arm": arm,
        "recipe_sha256": recipe_hash,
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "fresh_process_arm_worker": True,
        "moe_backend": "triton",
        "vllm_tuned_config_folder": expected_folder,
        "compact_arm": compact,
        "persisted_raw_content": False,
    }
    _exclusive_write_json(args.internal_arm_output, envelope)
    return envelope


def _load_arm_envelope_v16(path, arm, recipe, implementation):
    value = json.loads(Path(path).read_text())
    expected_keys = {
        "schema", "arm", "recipe_sha256", "implementation_bundle_sha256",
        "fresh_process_arm_worker", "moe_backend",
        "vllm_tuned_config_folder", "compact_arm", "persisted_raw_content",
        "content_sha256_before_self_field",
    }
    expected_folder = (
        None if arm == "default_triton"
        else str(prereg_v16.TUNING_DIRECTORY_V16)
    )
    if (
        set(value) != expected_keys
        or value.get("schema") != "eggroll-es-fused-moe-task-arm-v16"
        or value.get("arm") != arm
        or value.get("recipe_sha256")
        != recipe["content_sha256_before_self_field"]
        or value.get("implementation_bundle_sha256")
        != implementation["bundle_sha256"]
        or value.get("fresh_process_arm_worker") is not True
        or value.get("moe_backend") != "triton"
        or value.get("vllm_tuned_config_folder") != expected_folder
        or value.get("persisted_raw_content") is not False
        or value.get("content_sha256_before_self_field")
        != _canonical(_without_self(value))
    ):
        raise RuntimeError("v16 compact arm envelope changed")
    return value["compact_arm"]


def _candidate_v16(arms):
    value = {
        "schema": "eggroll-es-fused-moe-task-ab-summary-v16",
        "experiment_name": EXPERIMENT_NAME_V16,
        "alpha": 0.0,
        "model_update_applied": False,
        "validation_ood_heldout_or_benchmark_used": False,
        "arm_order": list(prereg_v16.ARM_ORDER_V16),
        "arms": arms,
        "panel_bundle_content_sha256": (
            prereg_v16.V13_PANEL_BUNDLE_CONTENT_SHA256_V16
        ),
        "panel_identities": copy.deepcopy(anchor_v13.PANEL_ORDERED_ROW_SHA256_V13),
        "perturbation_basis_sha256": (
            prereg_v16.V13_PERTURBATION_BASIS_SHA256_V16
        ),
        "all_integrity_audits_passed": True,
        "persisted_response_vectors_or_row_content": False,
    }
    value["content_sha256_before_self_field"] = _canonical(value)
    return value


def _run_fresh_arm_process_v16(
    args, bundle, implementation, recipe, arm, output_path, environment,
):
    command = [
        sys.executable, str(Path(__file__).resolve()), *_layer_cli_v16(bundle),
        "--expected-implementation-bundle-sha256",
        implementation["bundle_sha256"],
        "--expected-recipe-sha256", recipe["content_sha256_before_self_field"],
        "--internal-arm-worker", arm,
        "--internal-arm-token", _arm_token_v16(
            recipe["content_sha256_before_self_field"],
            implementation["bundle_sha256"], arm,
        ),
        "--internal-arm-output", str(output_path),
    ]
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def run_exact_v16(args, bundle, implementation, recipe):
    attempt_path = _attempt_path_v16().resolve()
    run_dir = (FROZEN_OUTPUT_DIRECTORY_V16 / EXPERIMENT_NAME_V16).resolve()
    report_path = run_dir / REPORT_NAME_V16
    arm_paths = {
        arm: run_dir / f"{arm}.compact.json"
        for arm in prereg_v16.ARM_ORDER_V16
    }
    if attempt_path.exists() or run_dir.exists():
        raise ValueError("v16 requires fresh exclusive attempt and run paths")
    provenance = _source_provenance_v16(implementation)
    default_environment = _arm_environment_v16("default_triton")
    tuned_environment = _arm_environment_v16(
        "tuned_triton", base_environment=default_environment,
    )
    environment_binding = _arm_environment_difference_v16(
        default_environment, tuned_environment,
    )
    attempt = {
        "schema": "eggroll-es-fused-moe-task-ab-attempt-v16",
        "status": "launching",
        "phase": "before_fresh_arm_processes",
        "recipe": recipe,
        "source_provenance": provenance,
        "fresh_process_per_arm": True,
        "environment_binding": environment_binding,
        "model_update_applied": False,
        "evaluation_opened": False,
        "raw_content_persisted": False,
    }
    _exclusive_write_json(attempt_path, attempt)
    try:
        for arm, environment in zip(
            prereg_v16.ARM_ORDER_V16,
            (default_environment, tuned_environment),
        ):
            _run_fresh_arm_process_v16(
                args, bundle, implementation, recipe, arm, arm_paths[arm],
                environment,
            )
        arms = {
            arm: _load_arm_envelope_v16(
                arm_paths[arm], arm, recipe, implementation,
            )
            for arm in prereg_v16.ARM_ORDER_V16
        }
        candidate = _candidate_v16(arms)
        gate = prereg_v16.evaluate_candidate_v16(candidate)
        report = {
            "schema": "eggroll-es-fused-moe-task-ab-report-v16",
            "recipe": recipe,
            "source_provenance": provenance,
            "environment_binding": environment_binding,
            "candidate": candidate,
            "gate": gate,
            "fresh_process_per_arm": True,
            "model_update_applied": False,
            "evaluation_opened": False,
            "raw_content_persisted": False,
        }
        _exclusive_write_json(report_path, report)
    except BaseException as error:
        attempt.update({
            "status": "failed",
            "phase": "fresh_process_task_ab",
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "model_update_applied": False,
            "evaluation_opened": False,
            "raw_content_persisted": False,
        })
        _rewrite_json(attempt_path, attempt)
        raise
    attempt.update({
        "status": "complete",
        "phase": "after_gate_and_compact_report",
        "report_binding": {
            "path": str(report_path),
            "file_sha256": _file_sha256(report_path),
            "content_sha256": report["content_sha256_before_self_field"],
        },
        "gate_content_sha256": gate["content_sha256_before_self_field"],
        "model_update_applied": False,
        "evaluation_opened": False,
        "raw_content_persisted": False,
    })
    _rewrite_json(attempt_path, attempt)
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    _assert_train_only_argv_v16(argv)
    bundle, remaining = anchor_v13.parse_frozen_layer_plan_cli_v13(argv)
    args = _parser_v16().parse_args(remaining)
    implementation = implementation_identity_v16()
    validate_runtime_v16(args, bundle, implementation)
    recipe = recipe_v16(bundle, implementation)
    if args.expected_recipe_sha256 not in (
        None, recipe["content_sha256_before_self_field"],
    ):
        raise ValueError("v16 recipe hash changed")
    if args.v16_dry_run:
        if args.internal_arm_worker is not None:
            raise ValueError("v16 dry run cannot be an internal arm worker")
        default_environment = _arm_environment_v16("default_triton")
        tuned_environment = _arm_environment_v16(
            "tuned_triton", base_environment=default_environment,
        )
        payload = {
            "schema": "eggroll-es-fused-moe-task-ab-dry-run-v16",
            "recipe": recipe,
            "implementation": implementation,
            "environment_binding": _arm_environment_difference_v16(
                default_environment, tuned_environment,
            ),
            "fresh_process_per_arm": True,
            "real_launch_requires_committed_bundle_and_recipe_hash": True,
            "gpu_launched": False,
        }
        payload["content_sha256_before_self_field"] = _canonical(payload)
        print(json.dumps(payload, sort_keys=True))
        return payload
    if args.internal_arm_worker is not None:
        return _run_arm_worker_v16(args, bundle, implementation, recipe)
    return run_exact_v16(args, bundle, implementation, recipe)


if __name__ == "__main__":
    main()
