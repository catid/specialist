#!/usr/bin/env python3
"""Fail-closed launch driver for the V13 alpha-zero train-panel diagnostic."""

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

from datasets import load_from_disk

import eggroll_es_train_panel_sampler_v13 as panel_sampler
import run_eggroll_es_anchor_line_search as driver_v1
import run_eggroll_es_anchor_variance_v10 as driver_v10
import train_eggroll_es_specialist as base
import train_eggroll_es_specialist_anchor_v11 as anchor_v11
import train_eggroll_es_specialist_anchor_v13 as anchor_v13


ROOT = Path(__file__).resolve().parent
FROZEN_MODEL_V13 = (ROOT / "models/Qwen3.6-35B-A3B").resolve()
FROZEN_TRAIN_DATASET_V13 = driver_v10.FROZEN_TRAIN_DATASET_V10.resolve()
FROZEN_OUTPUT_DIRECTORY_V13 = driver_v10.FROZEN_OUTPUT_DIRECTORY_V10.resolve()
EXPERIMENT_NAME_V13 = (
    "snapshot794_layer_v13_document_balanced_five_panel_alpha_zero_"
    "resident_sign_basis20260714"
)
REPORT_NAME_V13 = "train_panel_diagnostic_v13.json"
PROTOCOL_PATH_V13 = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_DOCUMENT_BALANCED_TRAIN_PANELS_V13_PROTOCOL.md"
).resolve()
LAUNCH_PROTOCOL_PATH_V13 = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_DOCUMENT_BALANCED_TRAIN_PANELS_V13_LAUNCH_PROTOCOL.md"
).resolve()
COMMITTED_DEPENDENCY_HASHES_V13 = {
    "sampler_v13": (
        "81ca63c230995d44dfdb739a78b4a1a0f85d09724ba5915860ae36f78ccc3da9"
    ),
    "panel_builder_v13": (
        "5a1dee89bd57812ba64a345a95c43442a0b110a2fef489afaa5e1c3ff7f2bfc0"
    ),
    "protocol_v13": (
        "b8805fb87365eb8106345d8f80101f1a31201d822fd064c62db8a89483de9ba0"
    ),
    "panel_manifest_v13": (
        "e555d9d6746cde6297cd3ab523b16dd7d78d81e2674447ee46d754ebfac52da7"
    ),
    "lineage_driver_v11g": (
        "913a0a728ae9e087e1240d4b6e56af02938eeda522b155d54593bec6d9fc127f"
    ),
    "trainer_v11c": (
        "c663b62f9d7990a2c59d8b46ad6258209b590bb29aa48946755a7d263a3d0799"
    ),
    "worker_v11c": (
        "d75951483058de340185fc81f6ed050deeac1551107c0357349a1f311cdb2c22"
    ),
    "trainer_v11b": (
        "2db34d796f7a39c85187964bdbd333d153212af47b257c9dfd0dbe92965c6254"
    ),
    "worker_v11b": (
        "64a0af9c977d8e09282560e8f8e2979a50034d6d78e081387ee1383bee97baa7"
    ),
    "trainer_v11": (
        "c3c4cde40408eeb91dd51cd48c17294cdc4389ee51359a0c560fa1ff94f4fec9"
    ),
    "worker_v11": (
        "1bda6397fd5f1478d47de929babe509e6a3475c882ffa642d5252a3f283cd8d5"
    ),
    "trainer_v10": (
        "7dcc3bbf4d4640ae0d004a1c791af81460ea418939d90b01eaef48faa9392df5"
    ),
    "worker_v10": (
        "42fce05ea99e457df17b9cd11c53ad8340b1b1321d2d4ebe9191ded1a964e0a9"
    ),
    "trainer_v4": (
        "9b771f5f5578cc233bea9b92ee48dbad8bbf7f363fae44d4dadde5e926f6cd65"
    ),
    "worker_v4": (
        "876033b6b2ac8a869f0f82656c7e49434b7ba25789c1ddf776ac433659f72f59"
    ),
}
IMPLEMENTATION_PATHS_V13 = {
    "worker_v13": ROOT / "eggroll_es_worker_v13.py",
    "trainer_v13": ROOT / "train_eggroll_es_specialist_anchor_v13.py",
    "driver_v13": Path(__file__).resolve(),
    "tests_v13": ROOT / "test_eggroll_es_train_panels_runtime_v13.py",
    "launch_protocol_v13": LAUNCH_PROTOCOL_PATH_V13,
    "sampler_v13": ROOT / "eggroll_es_train_panel_sampler_v13.py",
    "panel_builder_v13": ROOT / "build_eggroll_es_train_panels_v13.py",
    "protocol_v13": PROTOCOL_PATH_V13,
    "panel_manifest_v13": anchor_v13.PANEL_MANIFEST_PATH_V13,
    "lineage_driver_v11g": ROOT / "run_eggroll_es_anchor_equivalence_v11g.py",
    "trainer_v11c": ROOT / "train_eggroll_es_specialist_anchor_v11c.py",
    "worker_v11c": ROOT / "eggroll_es_worker_v11c.py",
    "trainer_v11b": ROOT / "train_eggroll_es_specialist_anchor_v11b.py",
    "worker_v11b": ROOT / "eggroll_es_worker_v11b.py",
    "trainer_v11": ROOT / "train_eggroll_es_specialist_anchor_v11.py",
    "worker_v11": ROOT / "eggroll_es_worker_v11.py",
    "trainer_v10": ROOT / "train_eggroll_es_specialist_anchor_v10.py",
    "worker_v10": ROOT / "eggroll_es_worker_v10.py",
    "trainer_v4": ROOT / "train_eggroll_es_specialist_anchor_v4.py",
    "worker_v4": ROOT / "eggroll_es_worker_v4.py",
}
FORBIDDEN_SURFACE_TOKENS_V13 = (
    "heldout", "validation", "ood", "benchmark", "eval-output",
)


def _file_sha256(path):
    return driver_v1.file_sha256(path)


def _canonical(value):
    return driver_v1.canonical_sha256(value)


def _assert_train_only_argv_v13(argv):
    for token in argv:
        lowered = str(token).lower()
        if any(value in lowered for value in FORBIDDEN_SURFACE_TOKENS_V13):
            raise ValueError(f"v13 rejects non-train surface: {token}")


def implementation_identity_v13():
    files = {
        key: {"path": str(Path(path).resolve()), "file_sha256": _file_sha256(path)}
        for key, path in IMPLEMENTATION_PATHS_V13.items()
    }
    actual_dependencies = {
        key: files[key]["file_sha256"] for key in COMMITTED_DEPENDENCY_HASHES_V13
    }
    if actual_dependencies != COMMITTED_DEPENDENCY_HASHES_V13:
        raise RuntimeError("v13 committed protocol/sampler/manifest changed")
    return {"files": files, "bundle_sha256": _canonical(files)}


def _source_provenance_v13(implementation):
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
                f"v13 real launch requires committed implementation: {relative}"
            ) from error
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item["file_sha256"]:
            raise RuntimeError(f"v13 source differs from committed HEAD: {relative}")
        committed[key] = {
            "relative_path": relative, "file_sha256": digest,
        }
    result = {
        "schema": "eggroll-es-committed-source-bundle-v13",
        "git_head": head,
        "files": committed,
        "implementation_bundle_sha256": implementation["bundle_sha256"],
    }
    result["content_sha256_before_self_field"] = _canonical(result)
    return result


def _parser_v13():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v13-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--model-name", default=str(FROZEN_MODEL_V13))
    parser.add_argument("--checkpoint")
    parser.add_argument("--train-dataset", default=str(FROZEN_TRAIN_DATASET_V13))
    parser.add_argument("--train-source", default=str(anchor_v13.TRAIN_SOURCE_PATH_V13))
    parser.add_argument(
        "--panel-manifest", default=str(anchor_v13.PANEL_MANIFEST_PATH_V13),
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
    parser.add_argument("--output-directory", default=str(FROZEN_OUTPUT_DIRECTORY_V13))
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME_V13)
    parser.add_argument("--logging", choices=("none",), default="none")
    parser.add_argument("--wandb-project", default="specialist-eggroll-es")
    return parser


def validate_arrow_train_v13(path):
    path = Path(path).resolve()
    if path != FROZEN_TRAIN_DATASET_V13:
        raise ValueError("v13 frozen train dataset path changed")
    dataset_dict = load_from_disk(str(path))
    if list(dataset_dict) != ["train"]:
        raise RuntimeError("v13 training artifact must contain exactly train")
    dataset = dataset_dict["train"]
    if (
        len(dataset) != panel_sampler.SOURCE_ROWS
        or len(dataset.cache_files) != 1
        or _file_sha256(dataset.cache_files[0]["filename"])
        != anchor_v13.TRAIN_ARROW_FILE_SHA256_V13
    ):
        raise RuntimeError("v13 frozen train Arrow identity changed")
    return {
        "path": str(path), "rows": len(dataset),
        "arrow_path": str(Path(dataset.cache_files[0]["filename"]).resolve()),
        "arrow_sha256": anchor_v13.TRAIN_ARROW_FILE_SHA256_V13,
    }


def validate_runtime_v13(args, bundle, implementation):
    anchor_v13.validate_frozen_layer_plan_bundle_v13(bundle)
    frozen_plan = anchor_v11.FROZEN_STABILITY_PLANS_V11[
        driver_v10.MIDDLE_LATE_PLAN_SHA256_V10
    ]
    if (
        bundle["plan_sha256"] != driver_v10.MIDDLE_LATE_PLAN_SHA256_V10
        or Path(bundle["path"]).resolve() != Path(frozen_plan["path"]).resolve()
        or bundle["file_sha256"] != frozen_plan["file_sha256"]
        or Path(args.model_name).resolve() != FROZEN_MODEL_V13
        or args.checkpoint is not None
        or Path(args.train_dataset).resolve() != FROZEN_TRAIN_DATASET_V13
        or Path(args.train_source).resolve() != anchor_v13.TRAIN_SOURCE_PATH_V13
        or Path(args.panel_manifest).resolve() != anchor_v13.PANEL_MANIFEST_PATH_V13
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
        or Path(args.output_directory).resolve() != FROZEN_OUTPUT_DIRECTORY_V13
        or args.experiment_name != EXPERIMENT_NAME_V13
        or args.logging != "none"
        or args.wandb_project != "specialist-eggroll-es"
    ):
        raise ValueError("v13 frozen alpha-zero four-GPU recipe changed")
    expected = args.expected_implementation_bundle_sha256
    if not args.v13_dry_run and expected is None:
        raise ValueError("v13 real launch requires implementation bundle hash")
    if expected is not None and expected != implementation["bundle_sha256"]:
        raise ValueError("v13 implementation bundle hash changed")


def recipe_v13(args, bundle, arrow, panels, implementation):
    panel_contract = {
        name: {
            "role": panels["panels"][name]["role"],
            "rows": panel_sampler.PANEL_SIZE,
            "ordered_row_identity_sha256": panels["panels"][name][
                "ordered_row_identity_sha256"
            ],
        }
        for name in anchor_v13.PANEL_NAMES_V13
    }
    recipe = {
        "schema": "eggroll-es-five-panel-recipe-v13",
        "model": str(FROZEN_MODEL_V13),
        "train_arrow": arrow,
        "train_source": copy.deepcopy(panels["source"]),
        "panel_manifest": copy.deepcopy(panels["manifest"]),
        "panels": panel_contract,
        "optimization_panels": list(anchor_v13.OPTIMIZATION_PANELS_V13),
        "train_screens": list(anchor_v13.TRAIN_SCREENS_V13),
        "perturbation_basis_seed": anchor_v13.PERTURBATION_BASIS_SEED_V13,
        "perturbation_basis_sha256": anchor_v13.PERTURBATION_BASIS_SHA256_V13,
        "perturbation_seed_sha256": _canonical(anchor_v13.PERTURBATION_SEEDS_V13),
        "sigma": args.sigma,
        "population_size": args.population_size,
        "signs": ["plus", "minus"],
        "alpha": 0.0,
        "model_update_allowed": False,
        "hardware": {
            "engine_count": 4, "tp_per_engine": 1, "gpu_ids": [0, 1, 2, 3],
            "complete_wave_required": True,
        },
        "layer_plan": {
            "path": bundle["path"], "file_sha256": bundle["file_sha256"],
            "plan_sha256": bundle["plan_sha256"],
            "model_config_sha256": bundle["model_config_sha256"],
        },
        "aggregate": (
            "coordinatewise_median_of_three_independently_standardized_"
            "Horvitz_Thompson_central_response_vectors"
        ),
        "screen_use": "comparison_only_excluded_from_aggregate",
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "output_directory": str(FROZEN_OUTPUT_DIRECTORY_V13),
        "experiment_name": EXPERIMENT_NAME_V13,
    }
    recipe["content_sha256_before_self_field"] = _canonical(recipe)
    return recipe


def _seal(payload):
    payload.pop("content_sha256_before_self_field", None)
    payload["content_sha256_before_self_field"] = _canonical(payload)


def _exclusive_write_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    _seal(payload)
    raw = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError(f"v13 exclusive output already exists: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())


def _rewrite_json(path, payload):
    _seal(payload)
    driver_v1.atomic_write_json(path, payload)


def _attempt_path_v13():
    return FROZEN_OUTPUT_DIRECTORY_V13 / f".{EXPERIMENT_NAME_V13}.launch_attempt.json"


def _make_trainer_v13(args, bundle):
    trainer_class = anchor_v13.load_trainer(bundle)
    return trainer_class(
        model_name=args.model_name, checkpoint=None, sigma=args.sigma,
        alpha=0.0, population_size=32, reward_shaping="z-scores",
        num_iterations=1, max_tokens=1, batch_size=56, mini_batch_size=56,
        reward_function=base.specialist_reward,
        template_function=base.specialist_template,
        train_dataloader=[], eval_dataloader_dict={}, eval_freq=1,
        n_vllm_engines=4, n_gpu_per_vllm_engine=1, logging="none",
        global_seed=43, use_gpus="0,1,2,3",
        experiment_name=EXPERIMENT_NAME_V13,
        wandb_project="specialist-eggroll-es", save_best_models=False,
        reward_function_timeout=10, output_directory=args.output_directory,
    )


def run_exact_v13(args, bundle, arrow, panels, implementation, recipe):
    attempt_path = _attempt_path_v13().resolve()
    run_dir = (FROZEN_OUTPUT_DIRECTORY_V13 / EXPERIMENT_NAME_V13).resolve()
    if attempt_path.exists() or run_dir.exists():
        raise ValueError("v13 requires fresh exclusive attempt and run paths")
    provenance = _source_provenance_v13(implementation)
    attempt = {
        "schema": "eggroll-es-durable-launch-attempt-v13",
        "status": "launching",
        "phase": "before_trainer_creation",
        "experiment_name": EXPERIMENT_NAME_V13,
        "run_directory": str(run_dir),
        "recipe": recipe,
        "source_provenance": provenance,
        "target_alpha_zero_only": True,
        "model_update_applied": False,
        "sealed_or_nontrain_surface_opened": False,
    }
    _exclusive_write_json(attempt_path, attempt)
    if run_dir.exists():
        attempt.update({
            "status": "failed",
            "phase": "exclusive_claim_detected_existing_run_directory",
            "failure": {
                "type": "FreshRunReservationError",
                "message": "v13 run directory appeared after attempt claim",
                "traceback": "",
            },
            "run_directory_exists_after_attempt": True,
        })
        _rewrite_json(attempt_path, attempt)
        raise ValueError("v13 run directory appeared after exclusive claim")
    trainer = None
    diagnostic = None
    configured = None
    failure = None
    failure_traceback = None
    try:
        base.set_seed(43)
        trainer = _make_trainer_v13(args, bundle)
        configured = trainer.configure_train_panels_v13(
            panels, frozen_layer_plan=bundle,
        )
        diagnostic = trainer.estimate_train_panels_v13(
            anchor_v13.PERTURBATION_SEEDS_V13,
        )
        anchor_v13.validate_diagnostic_v13(diagnostic)
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
            "status": "failed", "phase": "inside_train_panel_diagnostic",
            "failure": {
                "type": type(failure).__name__, "message": str(failure),
                "traceback": failure_traceback,
            },
            "run_directory_exists_after_attempt": run_dir.exists(),
            "report_exists_after_attempt": (run_dir / REPORT_NAME_V13).exists(),
            "model_update_applied": False,
        })
        _rewrite_json(attempt_path, attempt)
        raise failure
    report = {
        "schema": "eggroll-es-five-panel-alpha-zero-report-v13",
        "recipe": recipe,
        "configuration": configured,
        "diagnostic": diagnostic,
        "implementation": implementation,
        "model_update_applied": False,
        "sealed_or_nontrain_surface_opened": False,
        "decision": "diagnostic_only_no_promotion_interpretation",
    }
    report["content_sha256_before_self_field"] = _canonical(report)
    report_path = run_dir / REPORT_NAME_V13
    try:
        _exclusive_write_json(report_path, report)
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
        _rewrite_json(attempt_path, attempt)
        raise
    report_binding = {
        "path": str(report_path), "file_sha256": _file_sha256(report_path),
        "content_sha256": report["content_sha256_before_self_field"],
    }
    attempt.update({
        "status": "complete", "phase": "after_trainer_cleanup_and_report",
        "run_directory_exists_after_attempt": True,
        "report_exists_after_attempt": True,
        "report_binding": report_binding,
        "model_update_applied": False,
    })
    _rewrite_json(attempt_path, attempt)
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    _assert_train_only_argv_v13(argv)
    bundle, remaining = anchor_v13.parse_frozen_layer_plan_cli_v13(argv)
    args = _parser_v13().parse_args(remaining)
    implementation = implementation_identity_v13()
    validate_runtime_v13(args, bundle, implementation)
    arrow = validate_arrow_train_v13(args.train_dataset)
    panels = anchor_v13.load_panel_bundle_v13(
        args.panel_manifest, args.train_source,
    )
    recipe = recipe_v13(args, bundle, arrow, panels, implementation)
    if args.v13_dry_run:
        payload = {
            "schema": "eggroll-es-five-panel-launch-dry-run-v13",
            "recipe": recipe,
            "implementation": implementation,
            "real_launch_requires_committed_bundle": True,
            "gpu_launched": False,
        }
        payload["content_sha256_before_self_field"] = _canonical(payload)
        print(json.dumps(payload, sort_keys=True))
        return payload
    return run_exact_v13(
        args, bundle, arrow, panels, implementation, recipe,
    )


if __name__ == "__main__":
    main()
