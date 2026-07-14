#!/usr/bin/env python3
"""Fail-closed paired V15A architecture-stability launch driver."""

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

import eggroll_es_back_plan_preregistration_v15a as prereg_v15a
import run_eggroll_es_train_panels_v13 as driver_v13
import train_eggroll_es_specialist as base
import train_eggroll_es_specialist_anchor_v15a as anchor_v15a


ROOT = Path(__file__).resolve().parent
FROZEN_MODEL_V15A = (ROOT / "models/Qwen3.6-35B-A3B").resolve()
FROZEN_TRAIN_DATASET_V15A = driver_v13.FROZEN_TRAIN_DATASET_V13
FROZEN_OUTPUT_DIRECTORY_V15A = driver_v13.FROZEN_OUTPUT_DIRECTORY_V13
EXPERIMENT_NAME_V15A = prereg_v15a.EXPERIMENT_NAME_V15A
REPORT_NAME_V15A = "paired_architecture_stability_v15a.json"
TEST_PATH_V15A = (ROOT / "test_eggroll_es_back_plan_stability_v15a.py").resolve()

MODEL_METADATA_SHA256_V15A = {
    "config.json": "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
    "generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
    "model.safetensors.index.json": "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
    "tokenizer_config.json": "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b",
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "chat_template.jinja": "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259",
}
MODEL_SHARD_MANIFEST_SHA256_V15A = (
    "e904bfbcf0608701d016618cb1857a0a93c114fbfa25c25d191a6709b0ff965c"
)
MODEL_SHARD_COUNT_V15A = 26
MODEL_SHARD_TOTAL_BYTES_V15A = 71_903_776_776

COMMITTED_DEPENDENCY_HASHES_V15A = {
    "trainer_v6": "3620fae91c34873d5657ec5491290c0d160f0b8cab451868b6e98401a4f6f0e5",
    "worker_v6": "5ffd843344931a02594f096ba9e1b7ac3291d345ab71ad76df2ec9c03d0c2b26",
    "prereg_module_v15a": "89e6ceb808758db8ca528500d08ad23d377874ac075ab0b84d0cc4e9a092ea10",
    "prereg_tests_v15a": "fbd30995c3c80a49149566f183e460349776521ec9f3dc24e31fa373e002f4d0",
    "preregistration_v15a": "ad86f388ff4effbc195a3fd60d6d32c430a83026a331a18d625d477d390f3b88",
    "protocol_v15a": "10902864bbf9f4d560d73127ab08402e6b41f3129413e96eb38392bba5894f0f",
    "middle_late_plan_v6": "d65d702969dcec7a56ca4fcf461d402c44642966191a57c2ef092ec339e3e3df",
    "back_plan_v6": "73bfc82ba057908c0071d3c5e190581fecf6147cc398f06a994231f31908187e",
    "v13_evidence": "d367c9c4de1e1f3526ddb3dfba2f5bf24efc77cbccf951f7359eb1969fcd7b54",
    "v14a_negative_evidence": "9329c09fec5d76e209cff36ac80ffbbec69f53fe7bae9edcc069421d626cc9e9",
    "v14b_negative_evidence": "735ad52b6395700feb4e8a3dccab165f9b79e620a53918d96e0a26979f58224c",
}

IMPLEMENTATION_PATHS_V15A = {
    **driver_v13.IMPLEMENTATION_PATHS_V13,
    "trainer_v6": ROOT / "train_eggroll_es_specialist_anchor_v6.py",
    "worker_v6": ROOT / "eggroll_es_worker_v6.py",
    "trainer_v15a": ROOT / "train_eggroll_es_specialist_anchor_v15a.py",
    "driver_v15a": Path(__file__).resolve(),
    "tests_v15a": TEST_PATH_V15A,
    "prereg_module_v15a": ROOT / "eggroll_es_back_plan_preregistration_v15a.py",
    "prereg_tests_v15a": ROOT / "test_eggroll_es_back_plan_preregistration_v15a.py",
    "preregistration_v15a": prereg_v15a.PREREGISTRATION_PATH_V15A,
    "protocol_v15a": prereg_v15a.PROTOCOL_PATH_V15A,
    "middle_late_plan_v6": Path(
        prereg_v15a.LAYER_PLANS_V15A["middle_late"]["path"]
    ),
    "back_plan_v6": Path(prereg_v15a.LAYER_PLANS_V15A["back"]["path"]),
    "v13_evidence": prereg_v15a.V13_EVIDENCE_PATH_V15A,
    "v14a_negative_evidence": prereg_v15a.V14A_NEGATIVE_PATH_V15A,
    "v14b_negative_evidence": prereg_v15a.V14B_NEGATIVE_PATH_V15A,
}

FORBIDDEN_SURFACE_TOKENS_V15A = (
    "heldout", "validation", "ood", "benchmark", "eval-output",
)
MOE_ENVIRONMENT_OVERRIDES_V15A = (
    "VLLM_TUNED_CONFIG_FOLDER",
    "VLLM_ROCM_USE_AITER",
    "VLLM_ROCM_USE_AITER_MOE",
    "VLLM_USE_DEEP_GEMM",
    "VLLM_MOE_USE_DEEP_GEMM",
    "VLLM_USE_FLASHINFER_MOE_INT4",
    "VLLM_HUMMING_MOE_GEMM_TYPE",
)
MOE_BACKEND_ENVIRONMENT_MARKERS_V15A = (
    "MOE", "GEMM", "AITER", "FLASHINFER", "MARLIN", "CUTLASS",
)


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    return driver_v13._canonical(value)


def _assert_train_only_argv_v15a(argv):
    for token in argv:
        lowered = str(token).lower()
        if any(value in lowered for value in FORBIDDEN_SURFACE_TOKENS_V15A):
            raise ValueError(f"v15a rejects non-train surface: {token}")


def _moe_environment_binding_v15a():
    names = sorted({
        *MOE_ENVIRONMENT_OVERRIDES_V15A,
        *(
            name for name in os.environ
            if name.startswith("VLLM_") and any(
                marker in name for marker in MOE_BACKEND_ENVIRONMENT_MARKERS_V15A
            )
        ),
    })
    changed = {
        name: os.environ.get(name)
        for name in names if os.environ.get(name) not in (None, "")
    }
    if changed:
        raise ValueError(
            "v15a requires tuned-config and backend environment overrides unset"
        )
    return {
        "moe_backend": "triton",
        "vllm_tuned_config_folder": None,
        "known_backend_selector_environment_overrides": {
            name: None for name in MOE_ENVIRONMENT_OVERRIDES_V15A[1:]
        },
        "dynamic_rejection_policy": {
            "prefix": "VLLM_",
            "name_contains_any": list(MOE_BACKEND_ENVIRONMENT_MARKERS_V15A),
            "nonempty_values_allowed": False,
        },
        "v026_tuned_config_commit_82cdf8e_used": False,
        "explicit_backend_bound_by_committed_inherited_engine_recipe": True,
    }


def model_identity_v15a():
    metadata = {
        name: {
            "path": str((FROZEN_MODEL_V15A / name).resolve()),
            "file_sha256": _file_sha256(FROZEN_MODEL_V15A / name),
        }
        for name in MODEL_METADATA_SHA256_V15A
    }
    if {
        name: item["file_sha256"] for name, item in metadata.items()
    } != MODEL_METADATA_SHA256_V15A:
        raise RuntimeError("v15a frozen model metadata changed")
    index = json.loads(
        (FROZEN_MODEL_V15A / "model.safetensors.index.json").read_text()
    )
    names = sorted(set(index.get("weight_map", {}).values()))
    shards = [
        {"name": name, "size": (FROZEN_MODEL_V15A / name).stat().st_size}
        for name in names
    ]
    if (
        len(shards) != MODEL_SHARD_COUNT_V15A
        or sum(item["size"] for item in shards) != MODEL_SHARD_TOTAL_BYTES_V15A
        or _canonical(shards) != MODEL_SHARD_MANIFEST_SHA256_V15A
        or any(not (FROZEN_MODEL_V15A / item["name"]).is_file() for item in shards)
    ):
        raise RuntimeError("v15a frozen model shard manifest changed")
    value = {
        "schema": "eggroll-es-local-model-identity-v15a",
        "path": str(FROZEN_MODEL_V15A),
        "metadata": metadata,
        "weight_shard_count": MODEL_SHARD_COUNT_V15A,
        "weight_shard_total_bytes": MODEL_SHARD_TOTAL_BYTES_V15A,
        "weight_shard_name_size_manifest_sha256": (
            MODEL_SHARD_MANIFEST_SHA256_V15A
        ),
    }
    value["content_sha256_before_self_field"] = _canonical(value)
    return value


def implementation_identity_v15a():
    base_identity = driver_v13.implementation_identity_v13()
    files = {
        key: {
            "path": str(Path(path).resolve()),
            "file_sha256": _file_sha256(path),
        }
        for key, path in IMPLEMENTATION_PATHS_V15A.items()
    }
    if {
        key: files[key] for key in base_identity["files"]
    } != base_identity["files"]:
        raise RuntimeError("v15a inherited V13 implementation changed")
    if {
        key: files[key]["file_sha256"]
        for key in COMMITTED_DEPENDENCY_HASHES_V15A
    } != COMMITTED_DEPENDENCY_HASHES_V15A:
        raise RuntimeError("v15a committed preregistration dependency changed")
    return {"files": files, "bundle_sha256": _canonical(files)}


def _source_provenance_v15a(implementation):
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
                f"v15a real launch requires committed implementation: {relative}"
            ) from error
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item["file_sha256"]:
            raise RuntimeError(f"v15a source differs from committed HEAD: {relative}")
        committed[key] = {
            "relative_path": relative,
            "file_sha256": digest,
        }
    result = {
        "schema": "eggroll-es-committed-source-bundle-v15a",
        "git_head": head,
        "files": committed,
        "implementation_bundle_sha256": implementation["bundle_sha256"],
    }
    result["content_sha256_before_self_field"] = _canonical(result)
    return result


def _parser_v15a():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v15a-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-recipe-content-sha256")
    parser.add_argument("--model-name", default=str(FROZEN_MODEL_V15A))
    parser.add_argument("--checkpoint")
    parser.add_argument("--train-dataset", default=str(FROZEN_TRAIN_DATASET_V15A))
    parser.add_argument("--train-source", default=str(anchor_v15a.anchor_v13.TRAIN_SOURCE_PATH_V13))
    parser.add_argument("--panel-manifest", default=str(anchor_v15a.anchor_v13.PANEL_MANIFEST_PATH_V13))
    parser.add_argument("--preregistration", default=str(prereg_v15a.PREREGISTRATION_PATH_V15A))
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
    parser.add_argument("--output-directory", default=str(FROZEN_OUTPUT_DIRECTORY_V15A))
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME_V15A)
    parser.add_argument("--logging", choices=("none",), default="none")
    parser.add_argument("--wandb-project", default="specialist-eggroll-es")
    return parser


def _bound_preregistration_v15a(args):
    if Path(args.preregistration).resolve() != prereg_v15a.PREREGISTRATION_PATH_V15A:
        raise RuntimeError("v15a frozen preregistration path changed")
    frozen = json.loads(Path(args.preregistration).read_text())
    rebuilt = prereg_v15a.build_preregistration_v15a()
    if (
        frozen != rebuilt
        or _file_sha256(args.preregistration)
        != COMMITTED_DEPENDENCY_HASHES_V15A["preregistration_v15a"]
        or frozen.get("content_sha256_before_self_field")
        != "dda0f49e470cf5bb550f80d27a2389d069c8d064975c18b581261054462bb7c7"
    ):
        raise RuntimeError("v15a preregistration identity changed")
    return frozen


def validate_runtime_v15a(args, plans, implementation):
    _moe_environment_binding_v15a()
    if tuple(plans) != prereg_v15a.ARM_ORDER_V15A:
        raise ValueError("v15a paired arm order changed")
    for name in prereg_v15a.ARM_ORDER_V15A:
        metadata = anchor_v15a.validate_frozen_layer_plan_bundle_v15a(plans[name])
        if metadata["arm"] != name:
            raise ValueError("v15a paired layer-plan identity changed")
    if (
        Path(args.model_name).resolve() != FROZEN_MODEL_V15A
        or args.checkpoint is not None
        or Path(args.train_dataset).resolve() != FROZEN_TRAIN_DATASET_V15A
        or Path(args.train_source).resolve()
        != anchor_v15a.anchor_v13.TRAIN_SOURCE_PATH_V13
        or Path(args.panel_manifest).resolve()
        != anchor_v15a.anchor_v13.PANEL_MANIFEST_PATH_V13
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
        or Path(args.output_directory).resolve() != FROZEN_OUTPUT_DIRECTORY_V15A
        or args.experiment_name != EXPERIMENT_NAME_V15A
        or args.logging != "none"
        or args.wandb_project != "specialist-eggroll-es"
    ):
        raise ValueError("v15a frozen alpha-zero paired four-GPU recipe changed")
    expected = args.expected_implementation_bundle_sha256
    if not args.v15a_dry_run and expected is None:
        raise ValueError("v15a real launch requires implementation bundle hash")
    if expected is not None and expected != implementation["bundle_sha256"]:
        raise ValueError("v15a implementation bundle hash changed")


def recipe_v15a(
    args, plans, arrow, panels, preregistration, model_identity, implementation,
):
    panel_contract = {
        name: {
            "role": panels["panels"][name]["role"],
            "rows": 56,
            "ordered_row_identity_sha256": panels["panels"][name][
                "ordered_row_identity_sha256"
            ],
        }
        for name in anchor_v15a.anchor_v13.PANEL_NAMES_V13
    }
    recipe = {
        "schema": "eggroll-es-paired-architecture-recipe-v15a",
        "model": model_identity,
        "train_arrow": arrow,
        "train_source": copy.deepcopy(panels["source"]),
        "panel_manifest": copy.deepcopy(panels["manifest"]),
        "panels": panel_contract,
        "panel_bundle_content_sha256": panels[
            "content_sha256_before_self_field"
        ],
        "preregistration": {
            "path": str(prereg_v15a.PREREGISTRATION_PATH_V15A),
            "file_sha256": COMMITTED_DEPENDENCY_HASHES_V15A[
                "preregistration_v15a"
            ],
            "content_sha256": preregistration[
                "content_sha256_before_self_field"
            ],
        },
        "paired_architecture": {
            "arm_order": list(prereg_v15a.ARM_ORDER_V15A),
            "only_intended_difference": "selected_dense_layer_location",
            "same_fresh_basis_both_arms": True,
            "arms": {
                name: {
                    "path": plans[name]["path"],
                    "file_sha256": plans[name]["file_sha256"],
                    "plan_sha256": plans[name]["plan_sha256"],
                    "model_config_sha256": plans[name]["model_config_sha256"],
                    "layers": list(prereg_v15a.LAYER_PLANS_V15A[name]["layers"]),
                    "capacity": copy.deepcopy(prereg_v15a.CAPACITY_V15A),
                }
                for name in prereg_v15a.ARM_ORDER_V15A
            },
        },
        "perturbation_basis": prereg_v15a.perturbation_basis_v15a(),
        "perturbation_basis_sha256": prereg_v15a.PERTURBATION_BASIS_SHA256_V15A,
        "perturbation_seed_sha256": _canonical(
            prereg_v15a.PERTURBATION_SEEDS_V15A
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
        "moe_kernel_environment": _moe_environment_binding_v15a(),
        "promotion_gate": copy.deepcopy(preregistration["promotion_gate"]),
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "output_directory": str(FROZEN_OUTPUT_DIRECTORY_V15A),
        "experiment_name": EXPERIMENT_NAME_V15A,
        "persist_response_vectors_or_row_content": False,
    }
    recipe["content_sha256_before_self_field"] = _canonical(recipe)
    return recipe


def audit_dry_run_payload_v15a(payload):
    if (
        not isinstance(payload, dict)
        or set(payload) != {
            "schema", "recipe", "implementation",
            "real_launch_requires_committed_bundle_and_recipe",
            "gpu_launched", "content_sha256_before_self_field",
        }
        or payload.get("schema")
        != "eggroll-es-paired-architecture-launch-dry-run-v15a"
        or payload.get("gpu_launched") is not False
        or payload.get("real_launch_requires_committed_bundle_and_recipe") is not True
        or payload.get("recipe", {}).get("content_sha256_before_self_field")
        != _canonical({
            key: value for key, value in payload.get("recipe", {}).items()
            if key != "content_sha256_before_self_field"
        })
        or payload.get("implementation", {}).get("bundle_sha256")
        != _canonical(payload.get("implementation", {}).get("files"))
        or payload.get("recipe", {}).get("implementation_bundle_sha256")
        != payload.get("implementation", {}).get("bundle_sha256")
        or payload.get("content_sha256_before_self_field")
        != _canonical({
            key: value for key, value in payload.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v15a dry-run payload audit failed")
    return {
        "recipe_content_sha256": payload["recipe"][
            "content_sha256_before_self_field"
        ],
        "payload_content_sha256": payload[
            "content_sha256_before_self_field"
        ],
        "implementation_bundle_sha256": payload["implementation"][
            "bundle_sha256"
        ],
    }


def _attempt_path_v15a():
    return FROZEN_OUTPUT_DIRECTORY_V15A / f".{EXPERIMENT_NAME_V15A}.launch_attempt.json"


def _make_trainer_v15a(args, bundle):
    trainer_class = anchor_v15a.load_trainer(bundle)
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
        experiment_name=EXPERIMENT_NAME_V15A,
        wandb_project="specialist-eggroll-es",
        save_best_models=False,
        reward_function_timeout=10,
        output_directory=args.output_directory,
    )


def _execute_arm_v15a(args, name, bundle, panels):
    trainer = None
    configuration = None
    diagnostic = None
    failure = None
    failure_traceback = None
    try:
        base.set_seed(43)
        trainer = _make_trainer_v15a(args, bundle)
        configuration = trainer.configure_train_panels_v15a(
            panels, frozen_layer_plan=bundle,
        )
        diagnostic = trainer.estimate_train_panels_v15a(
            anchor_v15a.PERTURBATION_SEEDS_V15A,
        )
        anchor_v15a.validate_diagnostic_v15a(diagnostic)
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
        setattr(failure, "_v15a_traceback", failure_traceback)
        raise failure
    configuration = anchor_v15a.compact_configuration_v15a(
        name, configuration,
    )
    summary = anchor_v15a.compact_arm_summary_v15a(name, diagnostic)
    diagnostic = None
    return configuration, summary


def _reserve_run_directory_v15a(run_dir, attempt_path, attempt):
    try:
        os.mkdir(run_dir, 0o700)
    except BaseException as error:
        attempt.update({
            "status": "failed",
            "phase": "exclusive_run_directory_reservation",
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "run_directory_exists_after_attempt": run_dir.exists(),
            "model_update_applied": False,
        })
        driver_v13._rewrite_json(attempt_path, attempt)
        raise ValueError("v15a requires a fresh exclusive run path") from error


def run_exact_v15a(
    args, plans, panels, implementation, recipe,
):
    attempt_path = _attempt_path_v15a().resolve()
    run_dir = (FROZEN_OUTPUT_DIRECTORY_V15A / EXPERIMENT_NAME_V15A).resolve()
    if attempt_path.exists() or run_dir.exists():
        raise ValueError("v15a requires fresh exclusive attempt and run paths")
    provenance = _source_provenance_v15a(implementation)
    attempt = {
        "schema": "eggroll-es-durable-launch-attempt-v15a",
        "status": "launching",
        "phase": "before_exclusive_run_directory_reservation",
        "experiment_name": EXPERIMENT_NAME_V15A,
        "run_directory": str(run_dir),
        "recipe": recipe,
        "source_provenance": provenance,
        "target_alpha_zero_only": True,
        "arm_order": list(prereg_v15a.ARM_ORDER_V15A),
        "completed_arm_summary_bindings": {},
        "model_update_applied": False,
        "sealed_or_nontrain_surface_opened": False,
    }
    driver_v13._exclusive_write_json(attempt_path, attempt)
    _reserve_run_directory_v15a(run_dir, attempt_path, attempt)
    configurations = {}
    summaries = {}
    active_arm = None
    try:
        for name in prereg_v15a.ARM_ORDER_V15A:
            active_arm = name
            configurations[name], summaries[name] = _execute_arm_v15a(
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
            driver_v13._rewrite_json(attempt_path, attempt)
    except BaseException as error:
        attempt.update({
            "status": "failed",
            "phase": "inside_paired_architecture_diagnostic",
            "active_arm": active_arm,
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": getattr(
                    error, "_v15a_traceback", traceback.format_exc()
                ),
            },
            "run_directory_exists_after_attempt": run_dir.exists(),
            "report_exists_after_attempt": (run_dir / REPORT_NAME_V15A).exists(),
            "model_update_applied": False,
        })
        driver_v13._rewrite_json(attempt_path, attempt)
        raise
    candidate = anchor_v15a.build_candidate_v15a(summaries)
    gate = prereg_v15a.evaluate_candidate_v15a(candidate)
    report = {
        "schema": "eggroll-es-paired-architecture-alpha-zero-report-v15a",
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
            "authorize_only_separate_fresh_basis_back_confirmation"
            if gate["eligible_for_fresh_basis_back_plan_confirmation"]
            else "retain_v13_middle_late_and_keep_all_eval_surfaces_closed"
        ),
    }
    report["content_sha256_before_self_field"] = _canonical(report)
    report_path = run_dir / REPORT_NAME_V15A
    try:
        driver_v13._exclusive_write_json(report_path, report)
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
        driver_v13._rewrite_json(attempt_path, attempt)
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
    driver_v13._rewrite_json(attempt_path, attempt)
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    _assert_train_only_argv_v15a(argv)
    args = _parser_v15a().parse_args(argv)
    implementation = implementation_identity_v15a()
    plans = prereg_v15a.validate_layer_plans_v15a()
    validate_runtime_v15a(args, plans, implementation)
    model = model_identity_v15a()
    arrow = driver_v13.validate_arrow_train_v13(args.train_dataset)
    panels = anchor_v15a.anchor_v13.load_panel_bundle_v13(
        args.panel_manifest, args.train_source,
    )
    preregistration = _bound_preregistration_v15a(args)
    recipe = recipe_v15a(
        args, plans, arrow, panels, preregistration, model, implementation,
    )
    expected_recipe = args.expected_recipe_content_sha256
    if not args.v15a_dry_run and expected_recipe is None:
        raise ValueError("v15a real launch requires recipe content hash")
    if (
        expected_recipe is not None
        and expected_recipe != recipe["content_sha256_before_self_field"]
    ):
        raise ValueError("v15a recipe content hash changed")
    if args.v15a_dry_run:
        payload = {
            "schema": "eggroll-es-paired-architecture-launch-dry-run-v15a",
            "recipe": recipe,
            "implementation": implementation,
            "real_launch_requires_committed_bundle_and_recipe": True,
            "gpu_launched": False,
        }
        payload["content_sha256_before_self_field"] = _canonical(payload)
        audit_dry_run_payload_v15a(payload)
        print(json.dumps(payload, sort_keys=True))
        return payload
    return run_exact_v15a(args, plans, panels, implementation, recipe)


if __name__ == "__main__":
    main()
