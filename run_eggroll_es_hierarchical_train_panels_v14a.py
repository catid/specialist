#!/usr/bin/env python3
"""Fail-closed launch driver for the V14a full-frame train diagnostic."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path

import eggroll_es_hierarchical_preregistration_v14a as prereg_v14a
import run_eggroll_es_train_panels_v13 as driver_v13
import train_eggroll_es_specialist as base
import train_eggroll_es_specialist_anchor_v14a as anchor_v14a


ROOT = Path(__file__).resolve().parent
FROZEN_MODEL_V14A = driver_v13.FROZEN_MODEL_V13
FROZEN_TRAIN_DATASET_V14A = driver_v13.FROZEN_TRAIN_DATASET_V13
FROZEN_OUTPUT_DIRECTORY_V14A = driver_v13.FROZEN_OUTPUT_DIRECTORY_V13
EXPERIMENT_NAME_V14A = prereg_v14a.EXPERIMENT_NAME_V14A
REPORT_NAME_V14A = "full_frame_diagnostic_v14a.json"
TEST_PATH_V14A = (ROOT / "test_eggroll_es_hierarchical_train_panels_v14a.py").resolve()
PROTOCOL_PATH_V14A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_HIERARCHICAL_ROTATING_TRAIN_PANELS_V14A_PROTOCOL.md"
).resolve()
BASE_V14_PROTOCOL_PATH = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_HIERARCHICAL_ROTATING_TRAIN_SAMPLING_V14_PROTOCOL.md"
).resolve()
COMMITTED_DEPENDENCY_HASHES_V14A = {
    "v14_sampler": (
        "6981a746d6e0fc0904603abaf584ab71b9cc8a777a9abc00f4d305a98ebd186a"
    ),
    "v14_protocol": (
        "cb0a0d76efa4a078e2e2472338d484c5c1fe2e1f40593bd5828cb427e3f1a75d"
    ),
    "v14a_prereg_module": (
        "21df48d4a5f14b6c704a206bcacac3129ef67b6fea2c47a4ba366a976dd48802"
    ),
    "v14a_prereg_tests": (
        "a16f1404fbca7f1c5fe53a16b1dac761c38e6c1d2c82e1f9c59f9944e9d26744"
    ),
    "v14a_protocol": (
        "5490b1d8ddbbb490c1e01669640c05f5c2824602595f98fb816bf57a77f9a11b"
    ),
    "v14a_preregistration": (
        "d27052ee26d9ba5dd4383491b3d093d0a2f9469ddb4a073909a2b6590e0cba3e"
    ),
    "v13b_evidence_builder": (
        "531c5e5d9fc73ef568b886f2ae63b3d6b4b149be6d67aec127a8a0c009cee5c6"
    ),
    "v13b_compact_evidence": (
        "d367c9c4de1e1f3526ddb3dfba2f5bf24efc77cbccf951f7359eb1969fcd7b54"
    ),
    "v12_evidence_builder": (
        "148538069cdcf1cc5e8dd1d7a9c606ef4a8b4ec8d39fa6e7fb5f30a056bb755c"
    ),
    "v12_negative_evidence": (
        "4fec87ad1f41e40ba2ebc97dd46b58f4f7bf345e78d364a0ff2e98b9969a6512"
    ),
}
IMPLEMENTATION_PATHS_V14A = {
    **driver_v13.IMPLEMENTATION_PATHS_V13,
    "trainer_v14a": ROOT / "train_eggroll_es_specialist_anchor_v14a.py",
    "driver_v14a": Path(__file__).resolve(),
    "tests_v14a": TEST_PATH_V14A,
    "v14_sampler": ROOT / "eggroll_es_hierarchical_train_sampler_v14.py",
    "v14_protocol": BASE_V14_PROTOCOL_PATH,
    "v14a_prereg_module": ROOT / "eggroll_es_hierarchical_preregistration_v14a.py",
    "v14a_prereg_tests": ROOT / "test_eggroll_es_hierarchical_preregistration_v14a.py",
    "v14a_protocol": PROTOCOL_PATH_V14A,
    "v14a_preregistration": prereg_v14a.PREREGISTRATION_PATH_V14A,
    "v13b_evidence_builder": ROOT / "build_eggroll_es_v13b_evidence_v14a.py",
    "v13b_compact_evidence": prereg_v14a.EVIDENCE_PATH_V14A,
    "v12_evidence_builder": ROOT / "build_eggroll_es_v12_preseal_evidence_v14a.py",
    "v12_negative_evidence": prereg_v14a.V12_NEGATIVE_EVIDENCE_PATH_V14A,
}
FORBIDDEN_SURFACE_TOKENS_V14A = (
    "heldout", "validation", "ood", "benchmark", "eval-output",
)


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    return driver_v13._canonical(value)


def _assert_train_only_argv_v14a(argv):
    for token in argv:
        lowered = str(token).lower()
        if any(value in lowered for value in FORBIDDEN_SURFACE_TOKENS_V14A):
            raise ValueError(f"v14a rejects non-train surface: {token}")


def implementation_identity_v14a():
    base_identity = driver_v13.implementation_identity_v13()
    files = {
        key: {
            "path": str(Path(path).resolve()),
            "file_sha256": _file_sha256(path),
        }
        for key, path in IMPLEMENTATION_PATHS_V14A.items()
    }
    if {
        key: files[key] for key in base_identity["files"]
    } != base_identity["files"]:
        raise RuntimeError("v14a inherited V13 implementation changed")
    if {
        key: files[key]["file_sha256"]
        for key in COMMITTED_DEPENDENCY_HASHES_V14A
    } != COMMITTED_DEPENDENCY_HASHES_V14A:
        raise RuntimeError("v14a committed preregistration dependency changed")
    return {"files": files, "bundle_sha256": _canonical(files)}


def _source_provenance_v14a(implementation):
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
                f"v14a real launch requires committed implementation: {relative}"
            ) from error
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item["file_sha256"]:
            raise RuntimeError(f"v14a source differs from committed HEAD: {relative}")
        committed[key] = {"relative_path": relative, "file_sha256": digest}
    result = {
        "schema": "eggroll-es-committed-source-bundle-v14a",
        "git_head": head,
        "files": committed,
        "implementation_bundle_sha256": implementation["bundle_sha256"],
    }
    result["content_sha256_before_self_field"] = _canonical(result)
    return result


def _parser_v14a():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v14a-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--model-name", default=str(FROZEN_MODEL_V14A))
    parser.add_argument("--checkpoint")
    parser.add_argument("--train-dataset", default=str(FROZEN_TRAIN_DATASET_V14A))
    parser.add_argument(
        "--train-source", default=str(prereg_v14a.sampler_v13.DEFAULT_SOURCE),
    )
    parser.add_argument(
        "--preregistration", default=str(prereg_v14a.PREREGISTRATION_PATH_V14A),
    )
    parser.add_argument(
        "--v13b-evidence", default=str(prereg_v14a.EVIDENCE_PATH_V14A),
    )
    parser.add_argument(
        "--v12-negative-evidence",
        default=str(prereg_v14a.V12_NEGATIVE_EVIDENCE_PATH_V14A),
    )
    parser.add_argument("--sigma", type=float, default=0.0003)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=310)
    parser.add_argument("--mini-batch-size", type=int, default=310)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--n-vllm-engines", type=int, default=4)
    parser.add_argument("--n-gpu-per-vllm-engine", type=int, default=1)
    parser.add_argument("--use-gpus", default="0,1,2,3")
    parser.add_argument("--reward-function-timeout", type=int, default=10)
    parser.add_argument("--output-directory", default=str(FROZEN_OUTPUT_DIRECTORY_V14A))
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME_V14A)
    parser.add_argument("--logging", choices=("none",), default="none")
    parser.add_argument("--wandb-project", default="specialist-eggroll-es")
    return parser


def _bound_preregistration_v14a(args):
    if (
        Path(args.preregistration).resolve()
        != prereg_v14a.PREREGISTRATION_PATH_V14A
        or Path(args.v13b_evidence).resolve() != prereg_v14a.EVIDENCE_PATH_V14A
        or Path(args.v12_negative_evidence).resolve()
        != prereg_v14a.V12_NEGATIVE_EVIDENCE_PATH_V14A
    ):
        raise RuntimeError("v14a frozen evidence path changed")
    v13b = prereg_v14a.load_evidence_v14a(args.v13b_evidence)
    v12 = prereg_v14a.load_v12_negative_evidence_v14a(
        args.v12_negative_evidence
    )
    frozen = json.loads(Path(args.preregistration).read_text())
    if (
        frozen != prereg_v14a.build_preregistration_v14a()
        or frozen.get("content_sha256_before_self_field")
        != "e610c4bd83449b6b9cb3a0055f8e099ebae32ff6827aa64c6521d74705bda59d"
        or _file_sha256(args.preregistration)
        != "d27052ee26d9ba5dd4383491b3d093d0a2f9469ddb4a073909a2b6590e0cba3e"
    ):
        raise RuntimeError("v14a machine preregistration identity changed")
    return frozen, v13b, v12


def validate_runtime_v14a(args, bundle, implementation):
    anchor_v14a.validate_frozen_layer_plan_bundle_v14a(bundle)
    plan = driver_v13.anchor_v11.FROZEN_STABILITY_PLANS_V11[
        driver_v13.driver_v10.MIDDLE_LATE_PLAN_SHA256_V10
    ]
    if (
        bundle["plan_sha256"] != driver_v13.driver_v10.MIDDLE_LATE_PLAN_SHA256_V10
        or Path(bundle["path"]).resolve() != Path(plan["path"]).resolve()
        or bundle["file_sha256"] != plan["file_sha256"]
        or Path(args.model_name).resolve() != FROZEN_MODEL_V14A
        or args.checkpoint is not None
        or Path(args.train_dataset).resolve() != FROZEN_TRAIN_DATASET_V14A
        or Path(args.train_source).resolve()
        != prereg_v14a.sampler_v13.DEFAULT_SOURCE.resolve()
        or args.sigma != 0.0003
        or args.population_size != 32
        or args.batch_size != 310
        or args.mini_batch_size != 310
        or args.max_tokens != 1
        or args.seed != 43
        or args.alpha != 0.0
        or args.n_vllm_engines != 4
        or args.n_gpu_per_vllm_engine != 1
        or args.use_gpus != "0,1,2,3"
        or args.reward_function_timeout != 10
        or Path(args.output_directory).resolve() != FROZEN_OUTPUT_DIRECTORY_V14A
        or args.experiment_name != EXPERIMENT_NAME_V14A
        or args.logging != "none"
        or args.wandb_project != "specialist-eggroll-es"
    ):
        raise ValueError("v14a frozen alpha-zero four-GPU recipe changed")
    expected = args.expected_implementation_bundle_sha256
    if not args.v14a_dry_run and expected is None:
        raise ValueError("v14a real launch requires implementation bundle hash")
    if expected is not None and expected != implementation["bundle_sha256"]:
        raise ValueError("v14a implementation bundle hash changed")


def recipe_v14a(args, bundle, arrow, panels, preregistration, v13b, v12, implementation):
    recipe = {
        "schema": "eggroll-es-full-frame-recipe-v14a",
        "model": str(FROZEN_MODEL_V14A),
        "train_arrow": arrow,
        "source": copy.deepcopy(panels["source"]),
        "preregistration": copy.deepcopy(panels["preregistration"]),
        "v13b_evidence_content_sha256": v13b["content_sha256_before_self_field"],
        "v12_negative_evidence_content_sha256": v12[
            "content_sha256_before_self_field"
        ],
        "full_frame": {
            "rows": 310,
            "content_sha256": panels["full_frame"]["content_sha256"],
            "ordered_row_identity_sha256": panels["full_frame"][
                "ordered_row_identity_sha256"
            ],
        },
        "matched56": {
            name: {
                "role": panels["matched56"][name]["role"],
                "content_sha256": panels["matched56"][name]["content_sha256"],
                "ordered_row_identity_sha256": panels["matched56"][name][
                    "ordered_row_identity_sha256"
                ],
            } for name in prereg_v14a.PANEL_NAMES_V14A
        },
        "generation": {
            "prompts_per_engine_per_sign": 310,
            "generation_calls_per_engine_per_sign": 1,
            "matched_and_crossfit_responses_derived_without_generation": True,
        },
        "perturbation_basis_sha256": prereg_v14a.PERTURBATION_BASIS_SHA256_V14A,
        "perturbation_seed_sha256": _canonical(
            anchor_v14a.PERTURBATION_SEEDS_V14A
        ),
        "sigma": 0.0003,
        "population_size": 32,
        "alpha": 0.0,
        "model_update_allowed": False,
        "hardware": {
            "engine_count": 4, "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3], "complete_wave_required": True,
        },
        "layer_plan": {
            "path": bundle["path"], "file_sha256": bundle["file_sha256"],
            "plan_sha256": bundle["plan_sha256"],
            "model_config_sha256": bundle["model_config_sha256"],
        },
        "promotion_gate": copy.deepcopy(preregistration["promotion_gate"]),
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "output_directory": str(FROZEN_OUTPUT_DIRECTORY_V14A),
        "experiment_name": EXPERIMENT_NAME_V14A,
    }
    recipe["content_sha256_before_self_field"] = _canonical(recipe)
    return recipe


def _attempt_path_v14a():
    return FROZEN_OUTPUT_DIRECTORY_V14A / f".{EXPERIMENT_NAME_V14A}.launch_attempt.json"


def _make_trainer_v14a(args, bundle):
    trainer_class = anchor_v14a.load_trainer(bundle)
    return trainer_class(
        model_name=args.model_name, checkpoint=None, sigma=0.0003, alpha=0.0,
        population_size=32, reward_shaping="z-scores", num_iterations=1,
        max_tokens=1, batch_size=310, mini_batch_size=310,
        reward_function=base.specialist_reward,
        template_function=base.specialist_template,
        train_dataloader=[], eval_dataloader_dict={}, eval_freq=1,
        n_vllm_engines=4, n_gpu_per_vllm_engine=1, logging="none",
        global_seed=43, use_gpus="0,1,2,3",
        experiment_name=EXPERIMENT_NAME_V14A,
        wandb_project="specialist-eggroll-es", save_best_models=False,
        reward_function_timeout=10, output_directory=args.output_directory,
    )


def run_exact_v14a(args, bundle, panels, implementation, recipe):
    attempt_path = _attempt_path_v14a().resolve()
    run_dir = (FROZEN_OUTPUT_DIRECTORY_V14A / EXPERIMENT_NAME_V14A).resolve()
    if attempt_path.exists() or run_dir.exists():
        raise ValueError("v14a requires fresh exclusive attempt and run paths")
    provenance = _source_provenance_v14a(implementation)
    attempt = {
        "schema": "eggroll-es-durable-launch-attempt-v14a",
        "status": "launching", "phase": "before_trainer_creation",
        "experiment_name": EXPERIMENT_NAME_V14A,
        "run_directory": str(run_dir), "recipe": recipe,
        "source_provenance": provenance, "target_alpha_zero_only": True,
        "model_update_applied": False,
        "sealed_or_nontrain_surface_opened": False,
    }
    driver_v13._exclusive_write_json(attempt_path, attempt)
    if run_dir.exists():
        attempt.update({
            "status": "failed",
            "phase": "exclusive_claim_detected_existing_run_directory",
            "failure": {
                "type": "FreshRunReservationError",
                "message": "v14a run directory appeared after attempt claim",
                "traceback": "",
            },
            "run_directory_exists_after_attempt": True,
        })
        driver_v13._rewrite_json(attempt_path, attempt)
        raise ValueError("v14a run directory appeared after exclusive claim")
    trainer = None
    configuration = None
    diagnostic = None
    failure = None
    failure_traceback = None
    try:
        base.set_seed(43)
        trainer = _make_trainer_v14a(args, bundle)
        configuration = trainer.configure_full_frame_v14a(
            panels, frozen_layer_plan=bundle,
        )
        diagnostic = trainer.estimate_full_frame_v14a(
            anchor_v14a.PERTURBATION_SEEDS_V14A,
        )
        anchor_v14a.validate_diagnostic_v14a(diagnostic, panels)
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
                    failure_traceback += (
                        "\nCleanup failure after primary failure:\n"
                        + traceback.format_exc()
                    )
    if failure is not None:
        attempt.update({
            "status": "failed", "phase": "inside_full_frame_diagnostic",
            "failure": {
                "type": type(failure).__name__, "message": str(failure),
                "traceback": failure_traceback,
            },
            "run_directory_exists_after_attempt": run_dir.exists(),
            "report_exists_after_attempt": (run_dir / REPORT_NAME_V14A).exists(),
            "model_update_applied": False,
        })
        driver_v13._rewrite_json(attempt_path, attempt)
        raise failure
    report = {
        "schema": "eggroll-es-full-frame-alpha-zero-report-v14a",
        "recipe": recipe, "configuration": configuration,
        "diagnostic": diagnostic, "implementation": implementation,
        "model_update_applied": False,
        "sealed_or_nontrain_surface_opened": False,
        "decision": "train_only_sampler_gate_no_model_update",
    }
    report["content_sha256_before_self_field"] = _canonical(report)
    report_path = run_dir / REPORT_NAME_V14A
    try:
        driver_v13._exclusive_write_json(report_path, report)
    except BaseException as error:
        attempt.update({
            "status": "failed", "phase": "writing_final_report",
            "failure": {
                "type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "run_directory_exists_after_attempt": run_dir.exists(),
            "report_exists_after_attempt": report_path.exists(),
            "model_update_applied": False,
        })
        driver_v13._rewrite_json(attempt_path, attempt)
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
    driver_v13._rewrite_json(attempt_path, attempt)
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    _assert_train_only_argv_v14a(argv)
    bundle, remaining = anchor_v14a.parse_frozen_layer_plan_cli_v14a(argv)
    args = _parser_v14a().parse_args(remaining)
    implementation = implementation_identity_v14a()
    validate_runtime_v14a(args, bundle, implementation)
    arrow = driver_v13.validate_arrow_train_v13(args.train_dataset)
    preregistration, v13b, v12 = _bound_preregistration_v14a(args)
    panels = anchor_v14a.load_panel_bundle_v14a()
    recipe = recipe_v14a(
        args, bundle, arrow, panels, preregistration, v13b, v12,
        implementation,
    )
    if args.v14a_dry_run:
        payload = {
            "schema": "eggroll-es-full-frame-launch-dry-run-v14a",
            "recipe": recipe, "implementation": implementation,
            "real_launch_requires_committed_bundle": True,
            "gpu_launched": False,
        }
        payload["content_sha256_before_self_field"] = _canonical(payload)
        print(json.dumps(payload, sort_keys=True))
        return payload
    return run_exact_v14a(args, bundle, panels, implementation, recipe)


if __name__ == "__main__":
    main()
