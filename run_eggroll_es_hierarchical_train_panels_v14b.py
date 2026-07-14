#!/usr/bin/env python3
"""Fail-closed launch driver for the V14b k=2 train-only diagnostic."""

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

import eggroll_es_hierarchical_preregistration_v14b as prereg_v14b
import run_eggroll_es_hierarchical_train_panels_v14a as driver_v14a
import train_eggroll_es_specialist as base
import train_eggroll_es_specialist_anchor_v14b as anchor_v14b


ROOT = Path(__file__).resolve().parent
FROZEN_MODEL_V14B = driver_v14a.FROZEN_MODEL_V14A
FROZEN_TRAIN_DATASET_V14B = driver_v14a.FROZEN_TRAIN_DATASET_V14A
FROZEN_OUTPUT_DIRECTORY_V14B = driver_v14a.FROZEN_OUTPUT_DIRECTORY_V14A
EXPERIMENT_NAME_V14B = prereg_v14b.EXPERIMENT_NAME_V14B
REPORT_NAME_V14B = "paired_distinct_row_diagnostic_v14b.json"
TEST_PATH_V14B = (ROOT / "test_eggroll_es_hierarchical_train_panels_v14b.py").resolve()
PROTOCOL_PATH_V14B = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_PAIRED_DISTINCT_ROW_FULL_FRAME_V14B_PROTOCOL.md"
).resolve()
V14A_NEGATIVE_BUILDER_PATH_V14B = (
    ROOT / "build_eggroll_es_v14a_evidence_v14b.py"
).resolve()
V14A_NEGATIVE_TEST_PATH_V14B = (
    ROOT / "test_eggroll_es_v14a_evidence_v14b.py"
).resolve()
COMMITTED_DEPENDENCY_HASHES_V14B = {
    "sampler_v14b": (
        "6a827fbd4f42bd9c5b785eaa4ed5ed8b78663cbb7d035fbf0bb84e57c9cebe92"
    ),
    "sampler_tests_v14b": (
        "1557e83a2b953cfd0c55d814c84d7a62a139c5c96592a76269ed7527fee891c0"
    ),
    "prereg_module_v14b": (
        "dec672a3d91c27f853eadcdd4f30640924ed33fb0932b78e9d2d9f7a0aedc168"
    ),
    "prereg_tests_v14b": (
        "f879e86b1ebeed5f029a1c6c77cdec36b18d3ce11158fbbd507956c2e4b4925c"
    ),
    "preregistration_v14b": (
        "dcab1a49befebc8b67bbb9a80b866e876438a1e960e37953dff8c742b4e2c8ec"
    ),
    "protocol_v14b": (
        "103585b69a4785d6802de47a7af40c7db8b53674bb00a00bb528a054c7bff6d2"
    ),
    "v14a_negative_builder": (
        "c2c17bb13197d9fe84e73b752b1e8a624efba17cc7776835560bad345f31c151"
    ),
    "v14a_negative_evidence": (
        "9329c09fec5d76e209cff36ac80ffbbec69f53fe7bae9edcc069421d626cc9e9"
    ),
    "v14a_negative_tests": (
        "0a3725f4d1f63e7c77e2311a69b23b1a333ba8a987faaad8131e473edd47a2c3"
    ),
}
IMPLEMENTATION_PATHS_V14B = {
    **driver_v14a.IMPLEMENTATION_PATHS_V14A,
    "trainer_v14b": ROOT / "train_eggroll_es_specialist_anchor_v14b.py",
    "driver_v14b": Path(__file__).resolve(),
    "tests_v14b": TEST_PATH_V14B,
    "sampler_v14b": ROOT / "eggroll_es_paired_distinct_row_sampler_v14b.py",
    "sampler_tests_v14b": ROOT / "test_eggroll_es_paired_distinct_row_sampler_v14b.py",
    "prereg_module_v14b": ROOT / "eggroll_es_hierarchical_preregistration_v14b.py",
    "prereg_tests_v14b": ROOT / "test_eggroll_es_hierarchical_preregistration_v14b.py",
    "preregistration_v14b": prereg_v14b.PREREGISTRATION_PATH_V14B,
    "protocol_v14b": PROTOCOL_PATH_V14B,
    "v14a_negative_builder": V14A_NEGATIVE_BUILDER_PATH_V14B,
    "v14a_negative_evidence": prereg_v14b.V14A_NEGATIVE_EVIDENCE_PATH_V14B,
    "v14a_negative_tests": V14A_NEGATIVE_TEST_PATH_V14B,
}
FORBIDDEN_SURFACE_TOKENS_V14B = (
    "heldout", "validation", "ood", "benchmark", "eval-output",
)
MOE_ENVIRONMENT_OVERRIDES_V14B = (
    "VLLM_TUNED_CONFIG_FOLDER", "VLLM_ROCM_USE_AITER",
    "VLLM_ROCM_USE_AITER_MOE", "VLLM_USE_DEEP_GEMM",
    "VLLM_MOE_USE_DEEP_GEMM",
)


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    return driver_v14a._canonical(value)


def _assert_train_only_argv_v14b(argv):
    for token in argv:
        lowered = str(token).lower()
        if any(value in lowered for value in FORBIDDEN_SURFACE_TOKENS_V14B):
            raise ValueError(f"v14b rejects non-train surface: {token}")


def _moe_environment_binding_v14b():
    values = {name: os.environ.get(name) for name in MOE_ENVIRONMENT_OVERRIDES_V14B}
    changed = {name: value for name, value in values.items() if value not in (None, "")}
    if changed:
        raise ValueError(
            "v14b requires all MoE tuning/backend environment overrides unset"
        )
    return {
        "moe_backend": "triton",
        "vllm_tuned_config_folder": None,
        "backend_selector_environment_overrides": {
            name: None for name in MOE_ENVIRONMENT_OVERRIDES_V14B[1:]
        },
        "explicit_backend_bound_by_committed_inherited_engine_recipe": True,
    }


def implementation_identity_v14b():
    base_identity = driver_v14a.implementation_identity_v14a()
    files = {
        key: {"path": str(Path(path).resolve()), "file_sha256": _file_sha256(path)}
        for key, path in IMPLEMENTATION_PATHS_V14B.items()
    }
    if {key: files[key] for key in base_identity["files"]} != base_identity["files"]:
        raise RuntimeError("v14b inherited V14a implementation changed")
    if {
        key: files[key]["file_sha256"] for key in COMMITTED_DEPENDENCY_HASHES_V14B
    } != COMMITTED_DEPENDENCY_HASHES_V14B:
        raise RuntimeError("v14b committed preregistration dependency changed")
    return {"files": files, "bundle_sha256": _canonical(files)}


def _source_provenance_v14b(implementation):
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    committed = {}
    for key, item in implementation["files"].items():
        path = Path(item["path"]).resolve()
        relative = path.relative_to(ROOT).as_posix()
        try:
            raw = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"v14b real launch requires committed implementation: {relative}"
            ) from error
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item["file_sha256"]:
            raise RuntimeError(f"v14b source differs from committed HEAD: {relative}")
        committed[key] = {"relative_path": relative, "file_sha256": digest}
    result = {
        "schema": "eggroll-es-committed-source-bundle-v14b",
        "git_head": head, "files": committed,
        "implementation_bundle_sha256": implementation["bundle_sha256"],
    }
    result["content_sha256_before_self_field"] = _canonical(result)
    return result


def _parser_v14b():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v14b-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--model-name", default=str(FROZEN_MODEL_V14B))
    parser.add_argument("--checkpoint")
    parser.add_argument("--train-dataset", default=str(FROZEN_TRAIN_DATASET_V14B))
    parser.add_argument("--train-source", default=str(prereg_v14b.sampler_v13.DEFAULT_SOURCE))
    parser.add_argument("--preregistration", default=str(prereg_v14b.PREREGISTRATION_PATH_V14B))
    parser.add_argument("--v14a-negative-evidence", default=str(prereg_v14b.V14A_NEGATIVE_EVIDENCE_PATH_V14B))
    parser.add_argument("--v13-baseline-evidence", default=str(prereg_v14b.V13_BASELINE_EVIDENCE_PATH_V14B))
    parser.add_argument("--sigma", type=float, default=0.0003)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=481)
    parser.add_argument("--mini-batch-size", type=int, default=481)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--n-vllm-engines", type=int, default=4)
    parser.add_argument("--n-gpu-per-vllm-engine", type=int, default=1)
    parser.add_argument("--use-gpus", default="0,1,2,3")
    parser.add_argument("--reward-function-timeout", type=int, default=10)
    parser.add_argument("--output-directory", default=str(FROZEN_OUTPUT_DIRECTORY_V14B))
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME_V14B)
    parser.add_argument("--logging", choices=("none",), default="none")
    parser.add_argument("--wandb-project", default="specialist-eggroll-es")
    return parser


def _bound_preregistration_v14b(args):
    if (
        Path(args.preregistration).resolve() != prereg_v14b.PREREGISTRATION_PATH_V14B
        or Path(args.v14a_negative_evidence).resolve()
        != prereg_v14b.V14A_NEGATIVE_EVIDENCE_PATH_V14B
        or Path(args.v13_baseline_evidence).resolve()
        != prereg_v14b.V13_BASELINE_EVIDENCE_PATH_V14B
    ):
        raise RuntimeError("v14b frozen evidence path changed")
    frozen = json.loads(Path(args.preregistration).read_text())
    negative = prereg_v14b.load_v14a_negative_evidence_v14b(
        args.v14a_negative_evidence
    )
    baseline = prereg_v14b.load_v13_baseline_evidence_v14b()
    if (
        frozen != prereg_v14b.build_preregistration_v14b()
        or _file_sha256(args.preregistration)
        != "dcab1a49befebc8b67bbb9a80b866e876438a1e960e37953dff8c742b4e2c8ec"
        or frozen.get("content_sha256_before_self_field")
        != "0963d1a8e18a97af949b94762292536c606279610c7c239e445d85e5be2c3216"
    ):
        raise RuntimeError("v14b preregistration identity changed")
    return frozen, negative, baseline


def validate_runtime_v14b(args, bundle, implementation):
    _moe_environment_binding_v14b()
    anchor_v14b.validate_frozen_layer_plan_bundle_v14b(bundle)
    plan = driver_v14a.driver_v13.anchor_v11.FROZEN_STABILITY_PLANS_V11[
        driver_v14a.driver_v13.driver_v10.MIDDLE_LATE_PLAN_SHA256_V10
    ]
    if (
        bundle["plan_sha256"] != prereg_v14b.LAYER_PLAN_SHA256_V14B
        or Path(bundle["path"]).resolve() != Path(plan["path"]).resolve()
        or bundle["file_sha256"] != prereg_v14b.LAYER_PLAN_FILE_SHA256_V14B
        or Path(args.model_name).resolve() != FROZEN_MODEL_V14B
        or args.checkpoint is not None
        or Path(args.train_dataset).resolve() != FROZEN_TRAIN_DATASET_V14B
        or Path(args.train_source).resolve() != prereg_v14b.sampler_v13.DEFAULT_SOURCE.resolve()
        or args.sigma != 0.0003 or args.population_size != 32
        or args.batch_size != 481 or args.mini_batch_size != 481
        or args.max_tokens != 1 or args.seed != 43 or args.alpha != 0.0
        or args.n_vllm_engines != 4 or args.n_gpu_per_vllm_engine != 1
        or args.use_gpus != "0,1,2,3" or args.reward_function_timeout != 10
        or Path(args.output_directory).resolve() != FROZEN_OUTPUT_DIRECTORY_V14B
        or args.experiment_name != EXPERIMENT_NAME_V14B
        or args.logging != "none" or args.wandb_project != "specialist-eggroll-es"
    ):
        raise ValueError("v14b frozen alpha-zero four-GPU recipe changed")
    expected = args.expected_implementation_bundle_sha256
    if not args.v14b_dry_run and expected is None:
        raise ValueError("v14b real launch requires implementation bundle hash")
    if expected is not None and expected != implementation["bundle_sha256"]:
        raise ValueError("v14b implementation bundle hash changed")


def recipe_v14b(args, bundle, arrow, panels, preregistration, negative, baseline, implementation):
    recipe = {
        "schema": "eggroll-es-k2-recipe-v14b",
        "model": str(FROZEN_MODEL_V14B), "train_arrow": arrow,
        "source": copy.deepcopy(panels["source"]),
        "preregistration": copy.deepcopy(panels["preregistration"]),
        "v14a_negative_evidence_content_sha256": negative[
            "content_sha256_before_self_field"
        ],
        "v13_baseline_evidence_content_sha256": baseline[
            "content_sha256_before_self_field"
        ],
        "full_frame": {
            "documents": 310, "prompts": 481,
            "content_sha256": panels["full_frame"]["content_sha256"],
            "ordered_prompt_identity_sha256": panels["full_frame"][
                "ordered_prompt_identity_sha256"
            ],
        },
        "generation": {
            "prompts_per_engine_per_sign": 481,
            "generation_calls_per_engine_per_sign": 1,
            "document_means_before_equal_document_aggregation": True,
            "matched_and_crossfit_derived_without_generation": True,
        },
        "perturbation_basis_sha256": prereg_v14b.PERTURBATION_BASIS_SHA256_V14B,
        "perturbation_seed_sha256": _canonical(anchor_v14b.PERTURBATION_SEEDS_V14B),
        "sigma": 0.0003, "population_size": 32, "alpha": 0.0,
        "model_update_allowed": False,
        "hardware": {
            "engine_count": 4, "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3], "complete_wave_required": True,
        },
        "moe_kernel_environment": _moe_environment_binding_v14b(),
        "layer_plan": {
            "path": bundle["path"], "file_sha256": bundle["file_sha256"],
            "plan_sha256": bundle["plan_sha256"],
            "model_config_sha256": bundle["model_config_sha256"],
        },
        "promotion_gate": copy.deepcopy(preregistration["promotion_gate"]),
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "output_directory": str(FROZEN_OUTPUT_DIRECTORY_V14B),
        "experiment_name": EXPERIMENT_NAME_V14B,
    }
    recipe["content_sha256_before_self_field"] = _canonical(recipe)
    return recipe


def _attempt_path_v14b():
    return FROZEN_OUTPUT_DIRECTORY_V14B / f".{EXPERIMENT_NAME_V14B}.launch_attempt.json"


def _make_trainer_v14b(args, bundle):
    trainer_class = anchor_v14b.load_trainer(bundle)
    return trainer_class(
        model_name=args.model_name, checkpoint=None, sigma=0.0003, alpha=0.0,
        population_size=32, reward_shaping="z-scores", num_iterations=1,
        max_tokens=1, batch_size=481, mini_batch_size=481,
        reward_function=base.specialist_reward,
        template_function=base.specialist_template,
        train_dataloader=[], eval_dataloader_dict={}, eval_freq=1,
        n_vllm_engines=4, n_gpu_per_vllm_engine=1, logging="none",
        global_seed=43, use_gpus="0,1,2,3", experiment_name=EXPERIMENT_NAME_V14B,
        wandb_project="specialist-eggroll-es", save_best_models=False,
        reward_function_timeout=10, output_directory=args.output_directory,
    )


def run_exact_v14b(args, bundle, panels, implementation, recipe):
    attempt_path = _attempt_path_v14b().resolve()
    run_dir = (FROZEN_OUTPUT_DIRECTORY_V14B / EXPERIMENT_NAME_V14B).resolve()
    if attempt_path.exists() or run_dir.exists():
        raise ValueError("v14b requires fresh exclusive attempt and run paths")
    provenance = _source_provenance_v14b(implementation)
    attempt = {
        "schema": "eggroll-es-durable-launch-attempt-v14b",
        "status": "launching", "phase": "before_trainer_creation",
        "experiment_name": EXPERIMENT_NAME_V14B,
        "run_directory": str(run_dir), "recipe": recipe,
        "source_provenance": provenance, "target_alpha_zero_only": True,
        "model_update_applied": False,
        "sealed_or_nontrain_surface_opened": False,
    }
    driver_v14a.driver_v13._exclusive_write_json(attempt_path, attempt)
    if run_dir.exists():
        attempt.update({
            "status": "failed", "phase": "exclusive_claim_detected_existing_run_directory",
            "failure": {"type": "FreshRunReservationError", "message": "v14b run directory appeared after attempt claim", "traceback": ""},
            "run_directory_exists_after_attempt": True,
        })
        driver_v14a.driver_v13._rewrite_json(attempt_path, attempt)
        raise ValueError("v14b run directory appeared after exclusive claim")
    trainer = None
    configuration = None
    diagnostic = None
    failure = None
    failure_traceback = None
    try:
        base.set_seed(43)
        trainer = _make_trainer_v14b(args, bundle)
        configuration = trainer.configure_full_frame_v14b(
            panels, frozen_layer_plan=bundle,
        )
        diagnostic = trainer.estimate_full_frame_v14b(
            anchor_v14b.PERTURBATION_SEEDS_V14B,
        )
        anchor_v14b.validate_diagnostic_v14b(diagnostic)
    except BaseException as error:
        failure = error
        failure_traceback = traceback.format_exc()
    finally:
        if trainer is not None:
            try:
                base.close_trainer(trainer)
            except BaseException as cleanup_error:
                if failure is None:
                    failure = cleanup_error
                    failure_traceback = traceback.format_exc()
                else:
                    failure_traceback += "\nCleanup failure:\n" + traceback.format_exc()
    if failure is not None:
        attempt.update({
            "status": "failed", "phase": "inside_k2_diagnostic",
            "failure": {"type": type(failure).__name__, "message": str(failure), "traceback": failure_traceback},
            "run_directory_exists_after_attempt": run_dir.exists(),
            "report_exists_after_attempt": (run_dir / REPORT_NAME_V14B).exists(),
            "model_update_applied": False,
        })
        driver_v14a.driver_v13._rewrite_json(attempt_path, attempt)
        raise failure
    report = {
        "schema": "eggroll-es-k2-alpha-zero-report-v14b",
        "recipe": recipe, "configuration": configuration,
        "diagnostic": diagnostic, "implementation": implementation,
        "model_update_applied": False,
        "sealed_or_nontrain_surface_opened": False,
        "decision": "train_only_k2_estimator_gate_no_model_update",
    }
    report["content_sha256_before_self_field"] = _canonical(report)
    report_path = run_dir / REPORT_NAME_V14B
    try:
        driver_v14a.driver_v13._exclusive_write_json(report_path, report)
    except BaseException as error:
        attempt.update({
            "status": "failed", "phase": "writing_final_report",
            "failure": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
            "run_directory_exists_after_attempt": run_dir.exists(),
            "report_exists_after_attempt": report_path.exists(),
            "model_update_applied": False,
        })
        driver_v14a.driver_v13._rewrite_json(attempt_path, attempt)
        raise
    attempt.update({
        "status": "complete", "phase": "after_trainer_cleanup_and_report",
        "run_directory_exists_after_attempt": True,
        "report_exists_after_attempt": True,
        "report_binding": {
            "path": str(report_path), "file_sha256": _file_sha256(report_path),
            "content_sha256": report["content_sha256_before_self_field"],
        },
        "model_update_applied": False,
    })
    driver_v14a.driver_v13._rewrite_json(attempt_path, attempt)
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    _assert_train_only_argv_v14b(argv)
    bundle, remaining = anchor_v14b.parse_frozen_layer_plan_cli_v14b(argv)
    args = _parser_v14b().parse_args(remaining)
    implementation = implementation_identity_v14b()
    validate_runtime_v14b(args, bundle, implementation)
    arrow = driver_v14a.driver_v13.validate_arrow_train_v13(args.train_dataset)
    preregistration, negative, baseline = _bound_preregistration_v14b(args)
    panels = anchor_v14b.load_panel_bundle_v14b()
    recipe = recipe_v14b(
        args, bundle, arrow, panels, preregistration, negative, baseline,
        implementation,
    )
    if args.v14b_dry_run:
        payload = {
            "schema": "eggroll-es-k2-launch-dry-run-v14b",
            "recipe": recipe, "implementation": implementation,
            "real_launch_requires_committed_bundle": True,
            "gpu_launched": False,
        }
        payload["content_sha256_before_self_field"] = _canonical(payload)
        print(json.dumps(payload, sort_keys=True))
        return payload
    return run_exact_v14b(args, bundle, panels, implementation, recipe)


if __name__ == "__main__":
    main()
