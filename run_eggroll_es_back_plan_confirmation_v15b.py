#!/usr/bin/env python3
"""Fail-closed V15B paired fresh-basis confirmation driver."""

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

import eggroll_es_back_plan_confirmation_preregistration_v15b as prereg_v15b
import run_eggroll_es_back_plan_stability_v15a as driver_v15a
import train_eggroll_es_specialist as base
import train_eggroll_es_specialist_anchor_v15b as anchor_v15b


ROOT = Path(__file__).resolve().parent
FROZEN_MODEL_V15B = driver_v15a.FROZEN_MODEL_V15A
FROZEN_TRAIN_DATASET_V15B = driver_v15a.FROZEN_TRAIN_DATASET_V15A
FROZEN_OUTPUT_DIRECTORY_V15B = driver_v15a.FROZEN_OUTPUT_DIRECTORY_V15A
EXPERIMENT_NAME_V15B = prereg_v15b.EXPERIMENT_NAME_V15B
REPORT_NAME_V15B = "paired_back_confirmation_v15b.json"
TEST_PATH_V15B = (
    ROOT / "test_eggroll_es_back_plan_confirmation_runtime_v15b.py"
).resolve()
COMMITTED_DEPENDENCY_HASHES_V15B = {
    "prereg_module_v15b": (
        "e3c502f913a83705bddbefaa5e0f163712e41b1b82b23b7ce7b3a31d65bc8bd1"
    ),
    "prereg_tests_v15b": (
        "d07756c6bed35cecb4610f887a8a8ff4a546398130a0f85a2593caf6963953ee"
    ),
    "preregistration_v15b": (
        "5b90f16961c94d3a04b72ae29860094f7f1e6e8bad793780967c27448e0ba57f"
    ),
    "protocol_v15b": (
        "1de04a9612dc85debed3a4d7c30c10bdb348fbd3440c5e22f24a326f8619668e"
    ),
    "v15a_positive_evidence": (
        "1e14abee9e1514915bc241c8f6caacbe1bb7103e1c69a9afdde1f9ce13661ae1"
    ),
}
IMPLEMENTATION_PATHS_V15B = {
    **driver_v15a.IMPLEMENTATION_PATHS_V15A,
    "trainer_v15b": ROOT / "train_eggroll_es_specialist_anchor_v15b.py",
    "driver_v15b": Path(__file__).resolve(),
    "tests_v15b": TEST_PATH_V15B,
    "prereg_module_v15b": (
        ROOT / "eggroll_es_back_plan_confirmation_preregistration_v15b.py"
    ),
    "prereg_tests_v15b": (
        ROOT / "test_eggroll_es_back_plan_confirmation_preregistration_v15b.py"
    ),
    "preregistration_v15b": prereg_v15b.PREREGISTRATION_PATH_V15B,
    "protocol_v15b": prereg_v15b.PROTOCOL_PATH_V15B,
    "v15a_positive_evidence": prereg_v15b.V15A_POSITIVE_PATH_V15B,
}
FORBIDDEN_SURFACE_TOKENS_V15B = (
    "heldout", "validation", "ood", "benchmark", "eval-output",
)


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    return prereg_v15b.canonical_sha256(value)


def _assert_train_only_argv_v15b(argv):
    for token in argv:
        lowered = str(token).lower()
        if any(value in lowered for value in FORBIDDEN_SURFACE_TOKENS_V15B):
            raise ValueError(f"v15b rejects non-train surface: {token}")


def _moe_environment_binding_v15b():
    return driver_v15a._moe_environment_binding_v15a()


def implementation_identity_v15b():
    base_identity = driver_v15a.implementation_identity_v15a()
    files = {
        key: {
            "path": str(Path(path).resolve()),
            "file_sha256": _file_sha256(path),
        }
        for key, path in IMPLEMENTATION_PATHS_V15B.items()
    }
    if {
        key: files[key] for key in base_identity["files"]
    } != base_identity["files"]:
        raise RuntimeError("v15b inherited V15A implementation changed")
    if {
        key: files[key]["file_sha256"]
        for key in COMMITTED_DEPENDENCY_HASHES_V15B
    } != COMMITTED_DEPENDENCY_HASHES_V15B:
        raise RuntimeError("v15b committed preregistration dependency changed")
    return {"files": files, "bundle_sha256": _canonical(files)}


def _source_provenance_v15b(implementation):
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
                f"v15b real launch requires committed implementation: {relative}"
            ) from error
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item["file_sha256"]:
            raise RuntimeError(f"v15b source differs from committed HEAD: {relative}")
        committed[key] = {"relative_path": relative, "file_sha256": digest}
    value = {
        "schema": "eggroll-es-committed-source-bundle-v15b",
        "git_head": head,
        "files": committed,
        "implementation_bundle_sha256": implementation["bundle_sha256"],
    }
    value["content_sha256_before_self_field"] = _canonical(value)
    return value


def _parser_v15b():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v15b-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-recipe-content-sha256")
    parser.add_argument("--model-name", default=str(FROZEN_MODEL_V15B))
    parser.add_argument("--checkpoint")
    parser.add_argument("--train-dataset", default=str(FROZEN_TRAIN_DATASET_V15B))
    parser.add_argument(
        "--train-source", default=str(anchor_v15b.anchor_v13.TRAIN_SOURCE_PATH_V13)
    )
    parser.add_argument(
        "--panel-manifest",
        default=str(anchor_v15b.anchor_v13.PANEL_MANIFEST_PATH_V13),
    )
    parser.add_argument(
        "--preregistration", default=str(prereg_v15b.PREREGISTRATION_PATH_V15B)
    )
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
    parser.add_argument("--output-directory", default=str(FROZEN_OUTPUT_DIRECTORY_V15B))
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME_V15B)
    parser.add_argument("--logging", choices=("none",), default="none")
    parser.add_argument("--wandb-project", default="specialist-eggroll-es")
    return parser


def _bound_preregistration_v15b(args):
    if Path(args.preregistration).resolve() != prereg_v15b.PREREGISTRATION_PATH_V15B:
        raise RuntimeError("v15b frozen preregistration path changed")
    frozen = json.loads(Path(args.preregistration).read_text())
    rebuilt = prereg_v15b.build_preregistration_v15b()
    if (
        frozen != rebuilt
        or _file_sha256(args.preregistration)
        != COMMITTED_DEPENDENCY_HASHES_V15B["preregistration_v15b"]
        or frozen.get("content_sha256_before_self_field")
        != "0a4efb1a8a07cd194876d0942e77b188e4463b86cd6076606bccfb92f054f720"
    ):
        raise RuntimeError("v15b preregistration identity changed")
    return frozen


def _load_plans_v15b():
    return driver_v15a.prereg_v15a.validate_layer_plans_v15a()


def validate_runtime_v15b(args, plans, implementation):
    _moe_environment_binding_v15b()
    if tuple(plans) != prereg_v15b.ARM_ORDER_V15B:
        raise ValueError("v15b paired arm order changed")
    for name in prereg_v15b.ARM_ORDER_V15B:
        if anchor_v15b.validate_frozen_layer_plan_bundle_v15b(
            plans[name]
        )["arm"] != name:
            raise ValueError("v15b paired layer-plan identity changed")
    if (
        Path(args.model_name).resolve() != FROZEN_MODEL_V15B
        or args.checkpoint is not None
        or Path(args.train_dataset).resolve() != FROZEN_TRAIN_DATASET_V15B
        or Path(args.train_source).resolve()
        != anchor_v15b.anchor_v13.TRAIN_SOURCE_PATH_V13
        or Path(args.panel_manifest).resolve()
        != anchor_v15b.anchor_v13.PANEL_MANIFEST_PATH_V13
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
        or Path(args.output_directory).resolve() != FROZEN_OUTPUT_DIRECTORY_V15B
        or args.experiment_name != EXPERIMENT_NAME_V15B
        or args.logging != "none"
        or args.wandb_project != "specialist-eggroll-es"
    ):
        raise ValueError("v15b frozen alpha-zero paired four-GPU recipe changed")
    expected = args.expected_implementation_bundle_sha256
    if not args.v15b_dry_run and expected is None:
        raise ValueError("v15b real launch requires implementation bundle hash")
    if expected is not None and expected != implementation["bundle_sha256"]:
        raise ValueError("v15b implementation bundle hash changed")


def recipe_v15b(args, plans, preregistration, model, implementation):
    estimator = preregistration["estimator"]
    panels = {
        name: {
            "role": "optimization" if index < 3 else "train_only_screen",
            "rows": 56,
            "ordered_row_identity_sha256": identity,
        }
        for index, (name, identity) in enumerate(
            estimator["ordered_panel_identities"].items()
        )
    }
    v15a_prereg = prereg_v15b.load_v15a_preregistration_v15b()
    recipe = {
        "schema": "eggroll-es-paired-confirmation-recipe-v15b",
        "model": model,
        "train_identity": {
            "dataset_path": str(FROZEN_TRAIN_DATASET_V15B),
            "source": copy.deepcopy(v15a_prereg["v13_estimator"]["source"]),
            "manifest": copy.deepcopy(v15a_prereg["v13_estimator"]["manifest"]),
        },
        "panels": panels,
        "panel_bundle_content_sha256": estimator[
            "panel_bundle_content_sha256"
        ],
        "preregistration": {
            "path": str(prereg_v15b.PREREGISTRATION_PATH_V15B),
            "file_sha256": COMMITTED_DEPENDENCY_HASHES_V15B[
                "preregistration_v15b"
            ],
            "content_sha256": preregistration[
                "content_sha256_before_self_field"
            ],
        },
        "paired_architecture": {
            "arm_order": list(prereg_v15b.ARM_ORDER_V15B),
            "candidate_arm": "back",
            "control_arm": "middle_late",
            "control_can_be_promoted": False,
            "same_fresh_basis_both_arms": True,
            "only_intended_difference": "selected_dense_layer_location",
            "arms": {
                name: {
                    "path": plans[name]["path"],
                    "file_sha256": plans[name]["file_sha256"],
                    "plan_sha256": plans[name]["plan_sha256"],
                    "model_config_sha256": plans[name]["model_config_sha256"],
                    "layers": copy.deepcopy(
                        preregistration["paired_architecture"]["arms"][name]
                        ["layers"]
                    ),
                    "capacity": copy.deepcopy(
                        preregistration["paired_architecture"]["arms"][name]
                        ["capacity"]
                    ),
                }
                for name in prereg_v15b.ARM_ORDER_V15B
            },
        },
        "perturbation_basis": prereg_v15b.perturbation_basis_v15b(),
        "perturbation_basis_sha256": prereg_v15b.PERTURBATION_BASIS_SHA256_V15B,
        "perturbation_seed_sha256": _canonical(
            prereg_v15b.PERTURBATION_SEEDS_V15B
        ),
        "sigma": 0.0003,
        "population_size": 32,
        "signs": ["plus", "minus"],
        "alpha": 0.0,
        "model_update_allowed": False,
        "generation": {
            "seed": 43,
            "temperature": 0.0,
            "prompts_per_direction_and_sign": 280,
            "same_panel_and_prompt_order_both_arms": True,
        },
        "hardware": {
            "engine_count": 4,
            "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "complete_four_direction_waves_required": True,
            "population_waves_per_arm": 8,
            "signed_waves_per_arm": 16,
        },
        "moe_kernel_environment": _moe_environment_binding_v15b(),
        "promotion_gate": copy.deepcopy(preregistration["promotion_gate"]),
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "output_directory": str(FROZEN_OUTPUT_DIRECTORY_V15B),
        "experiment_name": EXPERIMENT_NAME_V15B,
        "persist_response_vectors_or_row_content": False,
    }
    recipe["content_sha256_before_self_field"] = _canonical(recipe)
    return recipe


def audit_dry_run_payload_v15b(payload):
    if (
        set(payload) != {
            "schema", "recipe", "implementation",
            "real_launch_requires_committed_bundle_and_recipe", "gpu_launched",
            "content_sha256_before_self_field",
        }
        or payload.get("schema")
        != "eggroll-es-paired-confirmation-launch-dry-run-v15b"
        or payload.get("gpu_launched") is not False
        or payload.get("real_launch_requires_committed_bundle_and_recipe")
        is not True
        or payload.get("recipe", {}).get("content_sha256_before_self_field")
        != _canonical({
            key: item for key, item in payload["recipe"].items()
            if key != "content_sha256_before_self_field"
        })
        or payload.get("implementation", {}).get("bundle_sha256")
        != _canonical(payload.get("implementation", {}).get("files"))
        or payload.get("recipe", {}).get("implementation_bundle_sha256")
        != payload.get("implementation", {}).get("bundle_sha256")
        or payload.get("content_sha256_before_self_field")
        != _canonical({
            key: item for key, item in payload.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v15b dry-run payload audit failed")
    return {
        "implementation_bundle_sha256": payload["implementation"][
            "bundle_sha256"
        ],
        "recipe_content_sha256": payload["recipe"][
            "content_sha256_before_self_field"
        ],
        "payload_content_sha256": payload[
            "content_sha256_before_self_field"
        ],
    }


def _attempt_path_v15b():
    return FROZEN_OUTPUT_DIRECTORY_V15B / f".{EXPERIMENT_NAME_V15B}.launch_attempt.json"


def _make_trainer_v15b(args, bundle):
    trainer_class = anchor_v15b.load_trainer(bundle)
    return trainer_class(
        model_name=args.model_name,
        checkpoint=None,
        sigma=0.0003,
        alpha=0.0,
        population_size=32,
        reward_shaping="z-scores",
        num_iterations=1,
        max_tokens=1,
        batch_size=56,
        mini_batch_size=56,
        reward_function=base.specialist_reward,
        template_function=base.specialist_template,
        train_dataloader=[],
        eval_dataloader_dict={},
        eval_freq=1,
        n_vllm_engines=4,
        n_gpu_per_vllm_engine=1,
        logging="none",
        global_seed=43,
        use_gpus="0,1,2,3",
        experiment_name=EXPERIMENT_NAME_V15B,
        wandb_project="specialist-eggroll-es",
        save_best_models=False,
        reward_function_timeout=10,
        output_directory=args.output_directory,
    )


def _execute_arm_v15b(args, name, bundle, panels):
    trainer = None
    failure = None
    failure_traceback = None
    configuration = diagnostic = None
    try:
        base.set_seed(43)
        trainer = _make_trainer_v15b(args, bundle)
        configuration = trainer.configure_train_panels_v15b(
            panels, frozen_layer_plan=bundle,
        )
        diagnostic = trainer.estimate_train_panels_v15b(
            anchor_v15b.PERTURBATION_SEEDS_V15B,
        )
        anchor_v15b.validate_diagnostic_v15b(diagnostic)
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
        setattr(failure, "_v15b_traceback", failure_traceback)
        raise failure
    return (
        anchor_v15b.compact_configuration_v15b(name, configuration),
        anchor_v15b.compact_arm_summary_v15b(name, diagnostic),
    )


def _reserve_run_directory_v15b(run_dir, attempt_path, attempt):
    try:
        os.mkdir(run_dir, 0o700)
    except BaseException as error:
        attempt.update({
            "status": "failed",
            "phase": "exclusive_run_directory_reservation",
            "failure": {"type": type(error).__name__, "message": str(error)},
            "run_directory_exists_after_attempt": run_dir.exists(),
            "model_update_applied": False,
        })
        driver_v15a.driver_v13._rewrite_json(attempt_path, attempt)
        raise ValueError("v15b requires a fresh exclusive run path") from error


def run_exact_v15b(args, plans, panels, implementation, recipe):
    attempt_path = _attempt_path_v15b().resolve()
    run_dir = (FROZEN_OUTPUT_DIRECTORY_V15B / EXPERIMENT_NAME_V15B).resolve()
    if attempt_path.exists() or run_dir.exists():
        raise ValueError("v15b requires fresh exclusive attempt and run paths")
    provenance = _source_provenance_v15b(implementation)
    attempt = {
        "schema": "eggroll-es-durable-launch-attempt-v15b",
        "status": "launching",
        "phase": "before_exclusive_run_directory_reservation",
        "experiment_name": EXPERIMENT_NAME_V15B,
        "run_directory": str(run_dir),
        "recipe": recipe,
        "source_provenance": provenance,
        "target_alpha_zero_only": True,
        "arm_order": list(prereg_v15b.ARM_ORDER_V15B),
        "completed_arm_summary_bindings": {},
        "model_update_applied": False,
        "sealed_or_nontrain_surface_opened": False,
    }
    driver_v15a.driver_v13._exclusive_write_json(attempt_path, attempt)
    _reserve_run_directory_v15b(run_dir, attempt_path, attempt)
    configurations = {}
    summaries = {}
    active_arm = None
    try:
        for name in prereg_v15b.ARM_ORDER_V15B:
            active_arm = name
            configurations[name], summaries[name] = _execute_arm_v15b(
                args, name, plans[name], panels,
            )
            attempt["phase"] = f"after_{name}_cleanup_before_next_arm_or_report"
            attempt["completed_arm_summary_bindings"][name] = {
                "plan_sha256": summaries[name]["plan_sha256"],
                "summary_content_sha256": summaries[name][
                    "content_sha256_before_self_field"
                ],
                "diagnostic_content_sha256": summaries[name][
                    "diagnostic_content_sha256"
                ],
            }
            driver_v15a.driver_v13._rewrite_json(attempt_path, attempt)
    except BaseException as error:
        attempt.update({
            "status": "failed",
            "phase": "inside_paired_confirmation_diagnostic",
            "active_arm": active_arm,
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": getattr(error, "_v15b_traceback", traceback.format_exc()),
            },
            "run_directory_exists_after_attempt": run_dir.exists(),
            "report_exists_after_attempt": (run_dir / REPORT_NAME_V15B).exists(),
            "model_update_applied": False,
        })
        driver_v15a.driver_v13._rewrite_json(attempt_path, attempt)
        raise
    candidate = anchor_v15b.build_candidate_v15b(summaries)
    gate = prereg_v15b.evaluate_candidate_v15b(candidate)
    report = {
        "schema": "eggroll-es-paired-back-confirmation-alpha-zero-report-v15b",
        "recipe": recipe,
        "configurations": configurations,
        "arm_summaries": summaries,
        "candidate_summary": candidate,
        "promotion_gate": gate,
        "implementation": implementation,
        "model_update_applied": False,
        "sealed_or_nontrain_surface_opened": False,
        "persisted_response_vectors_or_row_content": False,
        "decision": (
            "authorize_only_separate_back_plan_train_update_preregistration"
            if gate[
                "eligible_for_separate_back_plan_train_update_preregistration"
            ]
            else "retain_v13_middle_late_and_keep_all_eval_surfaces_closed"
        ),
    }
    report["content_sha256_before_self_field"] = _canonical(report)
    report_path = run_dir / REPORT_NAME_V15B
    try:
        driver_v15a.driver_v13._exclusive_write_json(report_path, report)
    except BaseException as error:
        attempt.update({
            "status": "failed",
            "phase": "writing_final_report",
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "run_directory_exists_after_attempt": run_dir.exists(),
            "report_exists_after_attempt": report_path.exists(),
            "model_update_applied": False,
        })
        driver_v15a.driver_v13._rewrite_json(attempt_path, attempt)
        raise
    attempt.update({
        "status": "complete",
        "phase": "after_both_arm_cleanups_and_report",
        "run_directory_exists_after_attempt": True,
        "report_exists_after_attempt": True,
        "report_binding": {
            "path": str(report_path),
            "file_sha256": _file_sha256(report_path),
            "content_sha256": report["content_sha256_before_self_field"],
        },
        "model_update_applied": False,
    })
    driver_v15a.driver_v13._rewrite_json(attempt_path, attempt)
    return report


def _validate_real_train_inputs_v15b(args, preregistration):
    arrow = driver_v15a.driver_v13.validate_arrow_train_v13(args.train_dataset)
    panels = anchor_v15b.anchor_v13.load_panel_bundle_v13(
        args.panel_manifest, args.train_source,
    )
    source = prereg_v15b.load_v15a_preregistration_v15b()["v13_estimator"][
        "source"
    ]
    if (
        arrow.get("rows") != source["rows"]
        or arrow.get("arrow_sha256") != source["arrow_sha256"]
    ):
        raise RuntimeError("v15b frozen train Arrow identity changed")
    if (
        panels.get("content_sha256_before_self_field")
        != preregistration["estimator"]["panel_bundle_content_sha256"]
        or {
            name: panel["ordered_row_identity_sha256"]
            for name, panel in panels["panels"].items()
        } != preregistration["estimator"]["ordered_panel_identities"]
    ):
        raise RuntimeError("v15b frozen panel identity changed")
    return panels


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    _assert_train_only_argv_v15b(argv)
    args = _parser_v15b().parse_args(argv)
    implementation = implementation_identity_v15b()
    plans = _load_plans_v15b()
    validate_runtime_v15b(args, plans, implementation)
    preregistration = _bound_preregistration_v15b(args)
    model = driver_v15a.model_identity_v15a()
    recipe = recipe_v15b(
        args, plans, preregistration, model, implementation,
    )
    expected_recipe = args.expected_recipe_content_sha256
    if not args.v15b_dry_run and expected_recipe is None:
        raise ValueError("v15b real launch requires recipe content hash")
    if (
        expected_recipe is not None
        and expected_recipe != recipe["content_sha256_before_self_field"]
    ):
        raise ValueError("v15b recipe content hash changed")
    if args.v15b_dry_run:
        payload = {
            "schema": "eggroll-es-paired-confirmation-launch-dry-run-v15b",
            "recipe": recipe,
            "implementation": implementation,
            "real_launch_requires_committed_bundle_and_recipe": True,
            "gpu_launched": False,
        }
        payload["content_sha256_before_self_field"] = _canonical(payload)
        audit_dry_run_payload_v15b(payload)
        print(json.dumps(payload, sort_keys=True))
        return payload
    panels = _validate_real_train_inputs_v15b(args, preregistration)
    return run_exact_v15b(args, plans, panels, implementation, recipe)


if __name__ == "__main__":
    main()
