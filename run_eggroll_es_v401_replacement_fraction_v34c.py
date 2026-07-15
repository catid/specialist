#!/usr/bin/env python3
"""Fail-closed four-GPU execution adapter for frozen V34B."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import build_eggroll_es_v401_replacement_fraction_frame_v34b as frame_v34b
import eggroll_es_v401_replacement_fraction_preregistration_v34b as prereg_v34b
import eggroll_es_worker_v33a as worker_v33a
import run_eggroll_es_insertion_stability_v23a as runtime_v23a
import run_eggroll_es_insertion_stability_v23a_retry_r2 as runtime_r2
import run_eggroll_es_paired_data_compat_v33a as runtime_v33a
import run_eggroll_es_v401_replacement_fraction_v34b as cpu_runner_v34b
import train_eggroll_es_specialist_anchor_v4 as anchor_v4
import train_eggroll_es_v401_replacement_fraction_v34b as mechanics_v34b


ROOT = Path(__file__).resolve().parent
V34B_COMMIT = "b254d4bdae0bb3fcb98d015c155393df9cca2d5d"
EXPERIMENT_NAME = "s6_v34c_production_v401_replacement_fraction_hpo_basis20261015"
OUTPUT_DIRECTORY = (ROOT / "experiments/eggroll_es_hpo/runs").resolve()
ATTEMPT_PATH = OUTPUT_DIRECTORY / f".{EXPERIMENT_NAME}.launch_attempt.json"
RUN_DIRECTORY = OUTPUT_DIRECTORY / EXPERIMENT_NAME
REPORT_PATH = RUN_DIRECTORY / "production_v401_replacement_fraction_report_v34c.json"
WORKER_EXTENSION = "eggroll_es_worker_v33a.PairedDataCompatWorkerExtensionV33A"
TEST_PATH = (ROOT / "test_run_eggroll_es_v401_replacement_fraction_v34c.py").resolve()

V34B_BOUND_FILES = {
    "frame_builder_v34b": (
        ROOT / "build_eggroll_es_v401_replacement_fraction_frame_v34b.py",
        "936efe422f560fd49f4f6bfa775465c786e9b3d2299189c4343f38ae1ede1774",
    ),
    "frame_test_v34b": (
        ROOT / "test_build_eggroll_es_v401_replacement_fraction_frame_v34b.py",
        "b1a8c7098302c844966dc056e191f193a3cfd908f17558e2c591ebdfc6f17ff7",
    ),
    "frame_v34b": (
        frame_v34b.OUTPUT_PATH,
        "832bbea07d08c487621e2dc88dfb8ebffc4b05d888badbbe5eb0fd71124efde3",
    ),
    "preregistration_module_v34b": (
        ROOT / "eggroll_es_v401_replacement_fraction_preregistration_v34b.py",
        "5ebb49b951c61e144cda1f6dd06d23b4b1f746cd500a58059f77fc8aec48f41b",
    ),
    "preregistration_test_v34b": (
        ROOT / "test_eggroll_es_v401_replacement_fraction_preregistration_v34b.py",
        "1499e8d5aff293fcdafc14896c89cc3506c691a1468ac39a0b7d15a16296b13d",
    ),
    "preregistration_v34b": (
        prereg_v34b.OUTPUT_PATH,
        "b852730872621fe9259087dd681ebf8854f985e8caa9208f0e8257a1d07de91b",
    ),
    "mechanics_v34b": (
        ROOT / "train_eggroll_es_v401_replacement_fraction_v34b.py",
        "5a59d618ba690f354c0564a52a391602b6f7a207a076523eac5ae262ba50c183",
    ),
    "mechanics_test_v34b": (
        ROOT / "test_train_eggroll_es_v401_replacement_fraction_v34b.py",
        "d6d1b87d53b18f9c0e69087c9e02cc2445e1cfb81d9acaf83535b6ec261bb3f9",
    ),
    "cpu_runner_v34b": (
        ROOT / "run_eggroll_es_v401_replacement_fraction_v34b.py",
        "4cfa9e0da6038b2ba5f3998c8597827e715b130e0bda7a2ca4c7145bf58cd02e",
    ),
    "cpu_runner_test_v34b": (
        ROOT / "test_run_eggroll_es_v401_replacement_fraction_v34b.py",
        "fc3e789b9d1a69f94f8c02e70cbb422ceee503bfb7d740930277703928066e45",
    ),
}
REUSED_V33A_FILES = {
    "gpu_runtime_v33a": (
        Path(runtime_v33a.__file__).resolve(),
        "4197fa48d9d719b55548bbdccbff6c22eeb7b0e6dec359bbff007463ad36eb38",
    ),
    "gpu_runtime_test_v33a": (
        ROOT / "test_run_eggroll_es_paired_data_compat_v33a.py",
        "131007f99576e0439946e148678c30ee4098690f267ba86c01ce4b9fd97c2f6c",
    ),
    "worker_v33a": (
        Path(worker_v33a.__file__).resolve(),
        "a65c506b8c96db64cc332578384ea9465c845329288591c7abc96e3a6e24e38e",
    ),
    "worker_test_v33a": (
        ROOT / "test_eggroll_es_worker_v33a.py",
        "50e034b43f760dea8c4122a7dd4085ab3d7a3b447f2d1515af2f899547e620f2",
    ),
}
EXPECTED_GPU_NAME = runtime_v33a.EXPECTED_GPU_NAME_V33A
EXPECTED_DRIVER_VERSION = runtime_v33a.EXPECTED_DRIVER_VERSION_V33A
EXPECTED_GPU_IDENTITIES = runtime_v33a.EXPECTED_GPU_IDENTITIES_V33A
ALLOWED_UNTRACKED_PREFIXES = ("experiments/dataset_probes/",)
ALLOWED_UNTRACKED_PATHS = frozenset({
    "experiments/gpu_utilization_v31_base_qwen36_train_reward_probe.jsonl",
})
CURATOR_UNTRACKED_PATTERN = re.compile(
    r"data/manual_reviews/context_merit_audit_v([0-9]+)/.+\Z"
)
WORKTREE_ALLOWLIST_CONTRACT = {
    "curator_snapshot_directories": (
        "data/manual_reviews/context_merit_audit_vN/ where N >= 390"
    ),
    "path_prefixes": list(ALLOWED_UNTRACKED_PREFIXES),
    "exact_paths": sorted(ALLOWED_UNTRACKED_PATHS),
    "tracked_changes_allowed": False,
    "other_untracked_paths_allowed": False,
}
FORBIDDEN_ARGV_TOKENS = (
    "validation", "heldout", "holdout", "ood", "benchmark", "eval",
    "checkpoint", "update", "promotion", "save-model", "train-dataset",
)
FORBIDDEN_PERSISTED_KEYS = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "prompt_token_ids", "unit_scores", "responses", "coefficients",
    "bootstrap_replicates", "bootstrap_draws", "row_content", "row_index",
    "row_sha256", "document_sha256", "unit_id", "unit_ids", "pids",
    "timings", "memory_samples", "traceback",
}


canonical_sha256 = prereg_v34b.canonical_sha256
file_sha256 = prereg_v34b.file_sha256
_seal = runtime_v23a._seal_v23a


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _recursive_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).lower()
            yield from _recursive_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _recursive_keys(item)


def assert_compact(value):
    overlap = FORBIDDEN_PERSISTED_KEYS & set(_recursive_keys(value))
    if overlap:
        raise RuntimeError(f"V34C compact output contains forbidden keys: {sorted(overlap)}")
    return value


def assert_train_only_argv(argv):
    for token in argv:
        lowered = str(token).lower()
        if any(item in lowered for item in FORBIDDEN_ARGV_TOKENS):
            raise ValueError(f"V34C rejects forbidden runtime surface: {token}")


def _git_blob(commit, path):
    relative = Path(path).resolve().relative_to(ROOT).as_posix()
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)


def verify_frozen_v34b():
    result = {}
    for name, (path, expected) in V34B_BOUND_FILES.items():
        path = Path(path).resolve()
        if (
            file_sha256(path) != expected
            or hashlib.sha256(_git_blob(V34B_COMMIT, path)).hexdigest() != expected
        ):
            raise RuntimeError(f"V34C frozen V34B binding changed: {name}")
        result[name] = {
            "relative_path": path.relative_to(ROOT).as_posix(),
            "file_sha256": expected,
            "commit": V34B_COMMIT,
        }
    for name, (path, expected) in REUSED_V33A_FILES.items():
        path = Path(path).resolve()
        if file_sha256(path) != expected:
            raise RuntimeError(f"V34C hash-bound V33A reuse changed: {name}")
        result[name] = {
            "relative_path": path.relative_to(ROOT).as_posix(),
            "file_sha256": expected,
        }
    # These source identities are also revalidated by the V34B frame mechanics.
    if (
        frame_v34b.CANDIDATE_FREEZE_COMMIT
        != "59dfe718a914be8b37e05ff9daa822ab467d18a4"
        or frame_v34b.CANDIDATE_SHA256
        != "8e29826dd389171c69f5eb6f43781f900345974c3d4d11274268e86c6145693b"
        or frame_v34b.PRODUCTION_SHA256
        != "62e7ae28c86a458d4d33bf3f73f1b91b873c86e3f70ce87706a7394d1f391507"
    ):
        raise RuntimeError("V34C V401 or production source identity changed")
    return result


def load_preregistration():
    verify_frozen_v34b()
    return mechanics_v34b.load_hardened_preregistration()


def load_layer_bundle(preregistration):
    plan = preregistration["frozen_recipe"]["layer_plan"]
    return anchor_v4.load_frozen_layer_plan_v4(
        plan["path"],
        expected_file_sha256=plan["file_sha256"],
        expected_plan_sha256=plan["content_sha256"],
        expected_model_config_sha256=preregistration["frozen_recipe"][
            "model_config_sha256"
        ],
    )


def validate_live_model(preregistration):
    model = Path(preregistration["frozen_recipe"]["model"]).resolve()
    if (
        file_sha256(model / "config.json")
        != preregistration["frozen_recipe"]["model_config_sha256"]
        or file_sha256(model / "model.safetensors.index.json")
        != preregistration["frozen_recipe"]["model_index_sha256"]
    ):
        raise RuntimeError("V34C live BF16 model identity changed")
    return _seal({
        "schema": "eggroll-es-v34c-live-model-audit",
        "config_sha256": preregistration["frozen_recipe"]["model_config_sha256"],
        "index_sha256": preregistration["frozen_recipe"]["model_index_sha256"],
    })


def recheck_postcleanup_bindings(
    preregistration, panel_bundle, layer_bundle, implementation, recipe,
):
    """Re-hash every frozen input after GPU cleanup, before compact analysis."""
    bound = verify_frozen_v34b()
    source_hashes = {
        "production": file_sha256(frame_v34b.PRODUCTION_PATH),
        "candidate_v401": file_sha256(frame_v34b.CANDIDATE_PATH),
        "candidate_manifest_v401": file_sha256(
            frame_v34b.CANDIDATE_MANIFEST_PATH
        ),
        "preregistration_v34b": file_sha256(prereg_v34b.OUTPUT_PATH),
    }
    if source_hashes != {
        "production": frame_v34b.PRODUCTION_SHA256,
        "candidate_v401": frame_v34b.CANDIDATE_SHA256,
        "candidate_manifest_v401": frame_v34b.CANDIDATE_MANIFEST_SHA256,
        "preregistration_v34b": mechanics_v34b.PREREGISTRATION_FILE_SHA256,
    }:
        raise RuntimeError("V34C post-cleanup source binding changed")
    current_implementation = implementation_identity()
    if current_implementation != implementation:
        raise RuntimeError("V34C post-cleanup implementation bundle changed")
    current_recipe = recipe_v34c(
        preregistration, panel_bundle, layer_bundle, current_implementation
    )
    if current_recipe != recipe:
        raise RuntimeError("V34C post-cleanup runtime recipe changed")
    return _seal({
        "schema": "eggroll-es-v34c-postcleanup-binding-recheck",
        "frozen_v34b_binding_sha256": canonical_sha256(bound),
        "source_file_sha256": source_hashes,
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "recipe_sha256": recipe["content_sha256_before_self_field"],
        "performed_after_gpu_cleanup_before_fraction_analysis": True,
    })


def seed_projection_contract(preregistration):
    seeds = preregistration["frozen_recipe"]["perturbation_basis"]["direction_seeds"]
    projections = [{
        "full_seed": seed,
        "numpy_legacy_seed": worker_v33a.worker_r1.project_numpy_legacy_seed_r1(seed),
    } for seed in seeds]
    expected_list = preregistration["frozen_recipe"]["perturbation_basis"][
        "direction_seed_list_sha256"
    ]
    if (
        seeds != prereg_v34b.perturbation_seeds()
        or len(seeds) != len(set(seeds))
        or len(seeds) != 64
        or canonical_sha256(seeds) != expected_list
        or len({item["numpy_legacy_seed"] for item in projections}) != 64
        or any(item["numpy_legacy_seed"] == 0 for item in projections)
    ):
        raise RuntimeError("V34C fresh V34B seed projection changed")
    return {
        "schema": "eggroll-es-v34c-seed-projection-contract",
        "direction_count": 64,
        "direction_seed_list_sha256": expected_list,
        "full_to_numpy_projection_sha256": canonical_sha256(projections),
        "numpy_projection_unique_count": 64,
        "worker_extension": WORKER_EXTENSION,
    }


def implementation_identity():
    inherited = runtime_r2.implementation_identity_r2()
    files = dict(inherited["files"])
    bound = verify_frozen_v34b()
    for name, item in bound.items():
        files[name] = {
            "path": str((ROOT / item["relative_path"]).resolve()),
            "file_sha256": item["file_sha256"],
        }
    overlay = {
        "runtime_v34c": Path(__file__).resolve(),
        "runtime_test_v34c": TEST_PATH,
    }
    for name, path in overlay.items():
        files[name] = {"path": str(path), "file_sha256": file_sha256(path)}
    return {
        "files": files,
        "inherited_v23a_r2_bundle_sha256": inherited["bundle_sha256"],
        "frozen_v34b_commit": V34B_COMMIT,
        "frozen_v34b_and_v33a_binding_sha256": canonical_sha256(bound),
        "v34c_overlay_bundle_sha256": canonical_sha256({
            name: files[name] for name in overlay
        }),
        "bundle_sha256": canonical_sha256(files),
    }


def recipe_v34c(preregistration, panel_bundle, layer_bundle, implementation):
    mechanics_v34b.validate_panel_bundle(panel_bundle)
    schedule = mechanics_v34b.resident_signed_wave_schedule()
    budget = preregistration["hardware_and_budget"]
    if (
        len(schedule) != 32
        or budget.get("perturbed_requests") != 49_920
        or budget.get("full_context_requests") != 4_680
        or budget.get("fraction_specific_requests") != 0
        or budget.get("total_generation_requests") != 54_600
    ):
        raise RuntimeError("V34C frozen schedule or request budget changed")
    value = {
        "schema": "eggroll-es-v401-replacement-fraction-gpu-recipe-v34c",
        "experiment_name": EXPERIMENT_NAME,
        "frozen_v34b": {
            "commit": V34B_COMMIT,
            "preregistration_file_sha256": mechanics_v34b.PREREGISTRATION_FILE_SHA256,
            "preregistration_content_sha256": mechanics_v34b.PREREGISTRATION_CONTENT_SHA256,
            "frame_file_sha256": prereg_v34b.FRAME_FILE_SHA256,
            "frame_content_sha256": prereg_v34b.FRAME_CONTENT_SHA256,
            "mechanics_file_sha256": V34B_BOUND_FILES["mechanics_v34b"][1],
        },
        "sources": {
            "production_file_sha256": frame_v34b.PRODUCTION_SHA256,
            "candidate_v401_file_sha256": frame_v34b.CANDIDATE_SHA256,
            "candidate_v401_freeze_commit": frame_v34b.CANDIDATE_FREEZE_COMMIT,
            "transient_panel_bundle_content_sha256": panel_bundle[
                "content_sha256_before_self_field"
            ],
            "raw_arms_scored": ["production", "candidate_v401"],
            "fraction_specific_raw_arm_or_model_request_count": 0,
        },
        "model": preregistration["frozen_recipe"]["model"],
        "layer_plan": {
            "path": layer_bundle["path"],
            "file_sha256": layer_bundle["file_sha256"],
            "plan_sha256": layer_bundle["plan_sha256"],
            "model_config_sha256": layer_bundle["model_config_sha256"],
        },
        "perturbation": {
            "sigma": 0.0003,
            "alpha": 0.0,
            "population_size": 64,
            "direction_seed_list_sha256": preregistration["frozen_recipe"][
                "perturbation_basis"
            ]["direction_seed_list_sha256"],
            "signed_schedule_sha256": canonical_sha256(schedule),
            "synchronized_signed_wave_count": 32,
            "engine_signed_direction_evaluation_count": 128,
            "all_four_tp1_engines_every_wave": True,
            "both_raw_sources_before_each_exact_restore": True,
        },
        "fraction_analysis": {
            "fractions_in_fixed_test_order": list(prereg_v34b.REPLACEMENT_FRACTIONS),
            "derived_algebraically_by_v34b_mechanics": True,
            "stop_at_first_failure": True,
            "all_12_point_and_Bonferroni_LCB_gates_per_fraction": True,
        },
        "guards": {
            "full_context_A_B_before_population_and_C_after": True,
            "A_equals_B_and_A_equals_C_exact": True,
            "exact_selected_restore_after_each_signed_wave": True,
            "full_partition_population_boundary": True,
            "unselected_origin_required_by_full_partition_audit": True,
            "all_four_engine_activity_each_wave": True,
            "preclaim_and_final_all_four_gpu_idle": True,
            "fail_closed_cleanup": True,
        },
        "request_budget": copy.deepcopy(budget),
        "hardware": {
            "gpu_ids": [0, 1, 2, 3],
            "engine_count": 4,
            "tp_per_engine": 1,
            "expected_gpu_name": EXPECTED_GPU_NAME,
            "expected_driver_version": EXPECTED_DRIVER_VERSION,
            "expected_physical_gpu_identity_sha256": canonical_sha256(
                EXPECTED_GPU_IDENTITIES
            ),
        },
        "seed_projection": seed_projection_contract(preregistration),
        "worker_extension": WORKER_EXTENSION,
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "fresh_paths": {
            "attempt": str(ATTEMPT_PATH),
            "run_directory": str(RUN_DIRECTORY),
            "report": str(REPORT_PATH),
        },
        "authority": {
            "direct_dataset_adoption": False,
            "model_update": False,
            "checkpoint_write": False,
            "evaluation": False,
            "dataset_promotion": False,
            "fp8_or_nontrain_reuse": False,
            "pass_authorizes_only_separately_frozen_train_only_recipe": True,
        },
        "persistence": {
            "compact_aggregate_only": True,
            "semantic_or_per_example_payloads": False,
            "score_vectors_or_bootstrap_draws": False,
        },
    }
    return _seal(value)


def _is_allowed_untracked(relative):
    if relative in ALLOWED_UNTRACKED_PATHS or any(
        relative.startswith(prefix) for prefix in ALLOWED_UNTRACKED_PREFIXES
    ):
        return True
    match = CURATOR_UNTRACKED_PATTERN.fullmatch(relative)
    return match is not None and int(match.group(1)) >= 390


def validate_worktree_status(raw_status):
    if not isinstance(raw_status, str):
        raise TypeError("V34C worktree status must be text")
    allowed = []
    rejected = []
    for line in raw_status.splitlines():
        if len(line) < 4 or line[2] != " ":
            raise RuntimeError("V34C worktree status record changed")
        status, relative = line[:2], line[3:]
        if status == "??" and _is_allowed_untracked(relative):
            allowed.append(relative)
        else:
            rejected.append((status, relative))
    if rejected:
        raise RuntimeError(
            "V34C real launch requires committed-clean source outside the exact allowlist"
        )
    return {
        "all_tracked_files_clean": True,
        "only_explicitly_allowlisted_untracked_paths_present": True,
        "allowed_untracked_entry_count": len(allowed),
        "allowed_untracked_entries_sha256": canonical_sha256(sorted(allowed)),
        "allowlist_contract_sha256": canonical_sha256(WORKTREE_ALLOWLIST_CONTRACT),
    }


def certify_committed_source(implementation, expected_source_commit):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != expected_source_commit or len(head) != 40:
        raise RuntimeError("V34C exact expected source commit changed")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", V34B_COMMIT, head],
        cwd=ROOT, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode != 0:
        raise RuntimeError("V34C frozen V34B commit is not an ancestor")
    committed = {}
    for name, item in implementation["files"].items():
        path = Path(item["path"]).resolve()
        if not path.is_relative_to(ROOT):
            continue
        relative = path.relative_to(ROOT).as_posix()
        try:
            raw = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"V34C real launch requires committed source: {relative}") from error
        if hashlib.sha256(raw).hexdigest() != item["file_sha256"]:
            raise RuntimeError(f"V34C source differs from expected commit: {relative}")
        committed[name] = item["file_sha256"]
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
    )
    worktree = validate_worktree_status(status)
    return _seal({
        "schema": "eggroll-es-v34c-committed-clean-source-certificate",
        "git_head": head,
        "committed_implementation_file_count": len(committed),
        "committed_implementation_sha256": canonical_sha256(committed),
        **worktree,
    })


def _request_identity(prompt_items):
    return canonical_sha256([item["prompt_token_ids"] for item in prompt_items])


def _score_panel_outputs(dense_items, outputs):
    return runtime_v33a._score_panel_outputs(dense_items, outputs)


def _phase_equal(reference, candidate):
    ref_scores, ref_commitments = reference
    scores, commitments = candidate
    return {
        "all_source_engine_panel_score_arrays_exact": all(
            np.array_equal(ref_scores[source], scores[source])
            for source in mechanics_v34b.SOURCES
        ),
        "all_dense_result_commitments_exact": ref_commitments == commitments,
    }


def _phase_identity(phase):
    scores, commitments = phase
    identities = {}
    for source in mechanics_v34b.SOURCES:
        values = np.asarray(scores[source], dtype=np.float64)
        if values.shape != (4, 5, 39) or not np.isfinite(values).all():
            raise RuntimeError("V34C full-context score geometry changed")
        identities[source] = {
            "shape": [4, 5, 39],
            "dtype": values.dtype.str,
            "byte_sha256": hashlib.sha256(
                np.ascontiguousarray(values).tobytes(order="C")
            ).hexdigest(),
        }
    return {
        "score_arrays_sha256": canonical_sha256(identities),
        "dense_commitments_sha256": canonical_sha256(commitments),
    }


class ReplacementFractionRuntimeMixinV34C:
    def configure_replacement_fraction_v34c(
        self, preregistration, panel_bundle, layer_bundle,
    ):
        panel_bundle = mechanics_v34b.validate_panel_bundle(copy.deepcopy(panel_bundle))
        if (
            len(self.engines) != 4
            or self.n_vllm_engines != 4
            or self.n_gpu_per_vllm_engine != 1
            or self.population_size != 64
            or not math.isclose(float(self.sigma), 0.0003, rel_tol=0.0, abs_tol=0.0)
            or not math.isclose(float(self.alpha), 0.0, rel_tol=0.0, abs_tol=0.0)
        ):
            raise RuntimeError("V34C exact four-TP1 alpha-zero recipe changed")
        projection = seed_projection_contract(preregistration)
        seeds = preregistration["frozen_recipe"]["perturbation_basis"]["direction_seeds"]
        seed_reports = [
            runtime_v23a._unwrap_one_v23a(item, "seed_projection_certificate_v33a")
            for item in self._resolve([
                engine.collective_rpc.remote(
                    "seed_projection_certificate_v33a",
                    args=(
                        seeds,
                        projection["direction_seed_list_sha256"],
                        projection["full_to_numpy_projection_sha256"],
                    ),
                )
                for engine in self.engines
            ])
        ]
        if (
            len(seed_reports) != 4
            or len({canonical_sha256(item) for item in seed_reports}) != 1
            or any(
                item.get("schema")
                != "eggroll-es-v33a-seed-projection-worker-certificate"
                or item.get("direction_count") != 64
                or item.get("direction_seed_list_sha256")
                != projection["direction_seed_list_sha256"]
                or item.get("full_to_numpy_projection_sha256")
                != projection["full_to_numpy_projection_sha256"]
                or item.get("numpy_projection_unique_count") != 64
                or item.get("numpy_projection_contains_zero") is not False
                for item in seed_reports
            )
        ):
            raise RuntimeError("V34C worker seed certificates changed")
        devices = [
            runtime_v23a._unwrap_one_v23a(item, "runtime_device_identity_v23a")
            for item in self._resolve([
                engine.collective_rpc.remote(
                    "runtime_device_identity_v23a", args=(f"direction_slot_{rank}",)
                )
                for rank, engine in enumerate(self.engines)
            ])
        ]
        if any(
            item.get("rank") != rank
            or item.get("world_size") != 4
            or item.get("arm") != f"direction_slot_{rank}"
            or item.get("cuda_visible_devices") != str(rank)
            or item.get("runtime_cuda_device") != 0
            or item.get("update_surfaces_closed") is not True
            for rank, item in enumerate(devices)
        ):
            raise RuntimeError("V34C fixed rank-to-GPU identity changed")
        installs = [
            runtime_v23a._unwrap_one_v23a(item, "install_layer_plan_v4")
            for item in self._resolve([
                engine.collective_rpc.remote(
                    "install_layer_plan_v4",
                    args=(
                        Path(layer_bundle["path"]).read_bytes(),
                        layer_bundle["file_sha256"],
                        layer_bundle["plan_sha256"],
                        anchor_v4.DENSE_GOLD_REWARD_CONFIG_V4,
                        anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4,
                    ),
                )
                for engine in self.engines
            ])
        ]
        if (
            any(
                item.get("installed") is not True
                or item.get("rank") != rank
                or item.get("world_size") != 4
                or item.get("plan_sha256", item.get("layer_plan_sha256"))
                != layer_bundle["plan_sha256"]
                or item.get("runtime_selected_parameter_count") != 23
                or item.get("selected_element_count") != 142_999_552
                or not isinstance(item.get("unselected_origin_sha256"), str)
                for rank, item in enumerate(installs)
            )
            or len({canonical_sha256(item.get("initial_identity")) for item in installs}) != 1
        ):
            raise RuntimeError("V34C layer-plan installation changed")
        references = [
            runtime_v23a._unwrap_one_v23a(item, "save_self_exact_reference")
            for item in self._resolve([
                engine.collective_rpc.remote("save_self_exact_reference", args=())
                for engine in self.engines
            ])
        ]
        selected = [
            runtime_v23a._unwrap_one_v23a(item, "verify_self_exact_reference")
            for item in self._resolve([
                engine.collective_rpc.remote("verify_self_exact_reference", args=())
                for engine in self.engines
            ])
        ]
        reference_sha256 = runtime_v33a._validate_exact_references_v33a(
            references, selected
        )
        self._v34c_preregistration = copy.deepcopy(preregistration)
        self._v34c_panel_bundle = panel_bundle
        self._v34c_references = [item["identity"] for item in references]
        self._v34c_installs = installs
        return _seal({
            "schema": "eggroll-es-v401-replacement-fraction-configuration-v34c",
            "device_identity_sha256": canonical_sha256(devices),
            "installation_sha256": canonical_sha256(installs),
            "seed_certificate_sha256": canonical_sha256(seed_reports),
            "selected_reference_identity_sha256": reference_sha256,
            "panel_bundle_content_sha256": panel_bundle[
                "content_sha256_before_self_field"
            ],
            "all_four_tp1_engines": True,
            "alpha_zero": True,
            "update_checkpoint_evaluation_surfaces_closed": True,
        })

    def _prepared_fixed_batches_v34c(self):
        bundle = mechanics_v34b.validate_panel_bundle(self._v34c_panel_bundle)
        prepared = {}
        request_identity = {}
        token_audit = {}
        for source in mechanics_v34b.SOURCES:
            panels = {}
            prompt_items = []
            cursor = 0
            source_token_audit = {}
            for panel_name in mechanics_v34b.PANEL_NAMES:
                batch = bundle["panels"][panel_name]["sources"][source]
                prompts = [runtime_v23a.base.specialist_template(item) for item in batch["questions"]]
                dense_items = anchor_v4.prepare_gold_answer_items_v4(
                    self.tokenizer, prompts, batch["answers"]
                )
                requests = [{"prompt_token_ids": item["prompt_token_ids"]} for item in dense_items]
                panels[panel_name] = {
                    "dense_items": dense_items,
                    "slice": (cursor, cursor + len(requests)),
                }
                prompt_items.extend(requests)
                cursor += len(requests)
                source_token_audit[panel_name] = {
                    "request_count": len(dense_items),
                    "dense_token_contract_sha256": canonical_sha256(dense_items),
                    "all_within_1024_token_cap": all(
                        len(item["prompt_token_ids"]) <= 1024 for item in dense_items
                    ),
                }
            if cursor != 195 or any(
                item["request_count"] != 39
                or item["all_within_1024_token_cap"] is not True
                for item in source_token_audit.values()
            ):
                raise RuntimeError("V34C fixed source request/token coverage changed")
            prepared[source] = {"panels": panels, "prompt_items": prompt_items}
            request_identity[source] = _request_identity(prompt_items)
            token_audit[source] = source_token_audit
        self._v34c_fixed_request_identity = request_identity
        return prepared, request_identity, token_audit

    def _generate_source_all_engines_v34c(self, source, prepared):
        if (
            source not in mechanics_v34b.SOURCES
            or _request_identity(prepared[source]["prompt_items"])
            != self._v34c_fixed_request_identity[source]
        ):
            raise RuntimeError("V34C fixed source request identity changed")
        batches = self._resolve([
            engine.generate.remote(
                list(prepared[source]["prompt_items"]),
                self._dense_sampling_params_v4(0),
                use_tqdm=False,
            )
            for engine in self.engines
        ])
        if len(batches) != 4 or any(len(batch) != 195 for batch in batches):
            raise RuntimeError("V34C all-four-engine source generation incomplete")
        return batches

    def _score_source_batches_v34c(self, source, prepared, batches):
        scores = np.empty((4, 5, 39), dtype=np.float64)
        commitments = []
        for engine_index, batch in enumerate(batches):
            for panel_index, panel_name in enumerate(mechanics_v34b.PANEL_NAMES):
                panel = prepared[source]["panels"][panel_name]
                start, stop = panel["slice"]
                rewards, digest = _score_panel_outputs(
                    panel["dense_items"], batch[start:stop]
                )
                scores[engine_index, panel_index] = rewards
                commitments.append(digest)
        return scores, commitments

    def _full_context_phase_v34c(self, prepared):
        scores = {}
        commitments = {}
        for source in mechanics_v34b.SOURCES:
            batches = self._generate_source_all_engines_v34c(source, prepared)
            scores[source], commitments[source] = self._score_source_batches_v34c(
                source, prepared, batches
            )
        return scores, commitments

    def _perturb_signed_wave_v34c(self, seeds, negate):
        if len(seeds) != 4:
            raise RuntimeError("V34C partial perturbation wave forbidden")
        raw = self._resolve([
            engine.collective_rpc.remote(
                "perturb_self_weights", args=(int(seed), 0.0003, bool(negate))
            )
            for engine, seed in zip(self.engines, seeds)
        ])
        runtime_v23a._validate_perturbation_results_v23a(raw)

    def _score_resident_source_v34c(
        self, source, prepared, schedule_item, unit_scores, commitments,
    ):
        batches = self._generate_source_all_engines_v34c(source, prepared)
        scores, dense = self._score_source_batches_v34c(source, prepared, batches)
        source_index = mechanics_v34b.SOURCES.index(source)
        sign_index = mechanics_v34b.SIGNS.index(schedule_item["sign"])
        for engine_index, direction_index in enumerate(
            schedule_item["engine_direction_indices"]
        ):
            unit_scores[source_index, :, sign_index, direction_index] = scores[
                engine_index
            ]
        commitments.extend(dense)
        return {
            "source": source,
            "all_four_engine_batch_sizes": [len(batch) for batch in batches],
            "dense_commitments_sha256": canonical_sha256(dense),
        }

    def _restore_and_verify_v34c(self):
        restored = [
            runtime_v23a._unwrap_one_v23a(item, "restore_self_weights_exact")
            for item in self._resolve([
                engine.collective_rpc.remote("restore_self_weights_exact", args=())
                for engine in self.engines
            ])
        ]
        checks = [
            runtime_v23a._unwrap_one_v23a(item, "verify_self_exact_reference")
            for item in self._resolve([
                engine.collective_rpc.remote("verify_self_exact_reference", args=())
                for engine in self.engines
            ])
        ]
        if (
            restored != [True] * 4
            or any(
                item.get("schema") != "eggroll-es-selected-exact-reference-check-v4"
                or item.get("passed") is not True
                or item.get("reference_generation") != 1
                or item.get("reference") != self._v34c_references[rank]["selected"]
                or item.get("current") != item.get("reference")
                for rank, item in enumerate(checks)
            )
        ):
            raise RuntimeError("V34C exact selected restore changed")
        return canonical_sha256(checks)

    def _run_signed_wave_v34c(
        self, schedule_item, prepared, unit_scores, dense_commitments,
    ):
        restore_hashes = []
        captures = mechanics_v34b.execute_resident_signed_wave(
            schedule_item,
            perturb=lambda seeds, negate: self._perturb_signed_wave_v34c(seeds, negate),
            score_source=lambda source: self._score_resident_source_v34c(
                source, prepared, schedule_item, unit_scores, dense_commitments
            ),
            restore=lambda: restore_hashes.append(self._restore_and_verify_v34c()),
        )
        if (
            len(restore_hashes) != 1
            or tuple(captures) != tuple(schedule_item["resident_source_order"])
            or any(
                item["all_four_engine_batch_sizes"] != [195, 195, 195, 195]
                for item in captures.values()
            )
        ):
            raise RuntimeError("V34C four-engine signed-wave activity changed")
        return {
            "signed_wave_index": schedule_item["signed_wave_index"],
            "restore_sha256": restore_hashes[0],
            "source_activity_sha256": canonical_sha256(captures),
            "all_four_engines_active_for_both_sources": True,
        }

    def _population_boundary_v34c(self):
        reports = [
            runtime_v23a._unwrap_one_v23a(item, "audit_population_completion_v4")
            for item in self._resolve([
                self.engines[rank].collective_rpc.remote(
                    "audit_population_completion_v4",
                    args=(4, 1, self._v34c_references[rank]["sha256"]),
                )
                for rank in range(4)
            ])
        ]
        if any(
            item.get("schema") != "eggroll-es-post-population-audit-v4"
            or item.get("passed") is not True
            or item.get("rank") != rank
            or item.get("world_size") != 4
            or item.get("reference_generation") != 1
            or item.get("reference_sha256") != self._v34c_references[rank]["sha256"]
            or item.get("current_identity") != self._v34c_references[rank]
            or item.get("unselected_origin_sha256")
            != self._v34c_installs[rank]["unselected_origin_sha256"]
            for rank, item in enumerate(reports)
        ):
            raise RuntimeError("V34C full-partition/unselected-origin boundary changed")
        return canonical_sha256(reports)

    def capture_raw_arms_v34c(self):
        schedule = mechanics_v34b.resident_signed_wave_schedule()
        prepared, request_identity, token_audit = self._prepared_fixed_batches_v34c()
        phase_a = self._full_context_phase_v34c(prepared)
        phase_b = self._full_context_phase_v34c(prepared)
        a_b_equal = _phase_equal(phase_a, phase_b)
        if not all(a_b_equal.values()):
            raise RuntimeError("V34C full-context A/B guard is not exact")
        unit_scores = np.full((2, 5, 2, 64, 39), np.nan, dtype=np.float64)
        dense_commitments = []
        wave_activity = []
        for item in schedule:
            wave_activity.append(self._run_signed_wave_v34c(
                item, prepared, unit_scores, dense_commitments
            ))
        if (
            not np.isfinite(unit_scores).all()
            or len(dense_commitments) != 1_280
            or len(wave_activity) != 32
            or any(
                item["all_four_engines_active_for_both_sources"] is not True
                for item in wave_activity
            )
        ):
            raise RuntimeError("V34C paired raw-arm population capture incomplete")
        boundary_sha256 = self._population_boundary_v34c()
        phase_c = self._full_context_phase_v34c(prepared)
        a_c_equal = _phase_equal(phase_a, phase_c)
        if not all(a_c_equal.values()):
            raise RuntimeError("V34C full-context A/C guard is not exact")
        audit = _seal({
            "schema": "eggroll-es-v401-replacement-fraction-preanalysis-audit-v34c",
            "fixed_request_identity_sha256": canonical_sha256(request_identity),
            "token_audit_sha256": canonical_sha256(token_audit),
            "full_context_guard": {
                "phase_a": _phase_identity(phase_a),
                "phase_b": _phase_identity(phase_b),
                "phase_c": _phase_identity(phase_c),
                "a_b_exact": a_b_equal,
                "a_c_exact": a_c_equal,
                "excluded_from_fraction_analysis": True,
            },
            "signed_schedule_sha256": canonical_sha256(schedule),
            "wave_activity_and_restore_sha256": canonical_sha256(wave_activity),
            "dense_result_commitments_sha256": canonical_sha256(dense_commitments),
            "population_boundary_sha256": boundary_sha256,
            "synchronized_signed_wave_count": 32,
            "all_four_tp1_engines_both_sources_every_wave": True,
            "perturbed_request_count": 49_920,
            "full_context_request_count": 4_680,
            "total_generation_request_count": 54_600,
            "fraction_specific_request_count": 0,
            "raw_scores_or_semantic_payloads_persisted": False,
            "model_update_applied": False,
        })
        assert_compact(audit)
        return unit_scores, audit

    @staticmethod
    def _closed_surface_v34c(*_args, **_kwargs):
        raise RuntimeError("V34C closes update checkpoint evaluation and promotion surfaces")

    configure_anchor = _closed_surface_v34c
    configure_train_panels_v13 = _closed_surface_v34c
    estimate_train_panels_v13 = _closed_surface_v34c
    estimate_step_coefficients = _closed_surface_v34c
    apply_seed_coefficients = _closed_surface_v34c
    train_step = _closed_surface_v34c
    evaluate_handle = _closed_surface_v34c
    evaluate_population_on_batch = _closed_surface_v34c
    eval_step = _closed_surface_v34c
    fit = _closed_surface_v34c


def load_runtime_trainer(preregistration, layer_bundle):
    model = preregistration["frozen_recipe"]["model"]
    parent = anchor_v4.load_trainer(layer_bundle)

    class ReplacementFractionRuntimeTrainerV34C(
        ReplacementFractionRuntimeMixinV34C, parent,
    ):
        def launch_engines(
            self, num_engines=4, n_gpu_per_vllm_engine=1,
            model_name="unused", precision="bfloat16",
        ):
            if num_engines != 4 or n_gpu_per_vllm_engine != 1:
                raise ValueError("V34C requires exactly four TP1 engines")
            import ray
            from es_at_scale.trainer.es_trainer import ESNcclLLM
            from ray.util.placement_group import placement_group
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

            groups = [
                placement_group(
                    [{"GPU": 1, "CPU": 0}], strategy="PACK", lifetime="detached"
                )
                for _ in range(4)
            ]
            ray.get([group.ready() for group in groups])
            strategies = [
                PlacementGroupSchedulingStrategy(
                    placement_group=group,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=0,
                )
                for group in groups
            ]
            engines = []
            for rank in range(4):
                engines.append(ray.remote(
                    num_cpus=0,
                    num_gpus=1,
                    scheduling_strategy=strategies[rank],
                )(ESNcclLLM).remote(
                    model=model,
                    tensor_parallel_size=1,
                    worker_extension_cls=WORKER_EXTENSION,
                    dtype=precision,
                    enable_prefix_caching=False,
                    enforce_eager=True,
                    gpu_memory_utilization=0.82,
                    max_model_len=2048,
                    limit_mm_per_prompt={"image": 0, "video": 0},
                    mm_processor_cache_gb=0,
                    skip_mm_profiling=True,
                    moe_backend="triton",
                ))
            return engines, groups

    return ReplacementFractionRuntimeTrainerV34C


def make_trainer(preregistration, layer_bundle):
    cls = load_runtime_trainer(preregistration, layer_bundle)
    return cls(
        model_name=preregistration["frozen_recipe"]["model"],
        checkpoint=None,
        sigma=0.0003,
        alpha=0.0,
        population_size=64,
        reward_shaping="z-scores",
        num_iterations=1,
        max_tokens=1,
        batch_size=195,
        mini_batch_size=195,
        reward_function=runtime_v23a.base.specialist_reward,
        template_function=runtime_v23a.base.specialist_template,
        train_dataloader=[],
        eval_dataloader_dict={},
        eval_freq=1,
        n_vllm_engines=4,
        n_gpu_per_vllm_engine=1,
        logging="none",
        global_seed=43,
        use_gpus="0,1,2,3",
        experiment_name=EXPERIMENT_NAME,
        wandb_project="specialist-eggroll-es",
        save_best_models=False,
        reward_function_timeout=10,
        output_directory=str(OUTPUT_DIRECTORY),
    )


def runtime_integrity_after_cleanup():
    value = {
        key: True for key in mechanics_v34b.RUNTIME_INTEGRITY_KEYS
    }
    return mechanics_v34b.validate_runtime_integrity(value)


def _write_attempt(value):
    runtime_v23a._exclusive_write_json_v23a(ATTEMPT_PATH, value)


def _rewrite_attempt(value):
    runtime_v23a._rewrite_json_v23a(ATTEMPT_PATH, value)


def run_exact_v34c(
    preregistration, panel_bundle, layer_bundle, implementation, recipe,
    committed_source,
):
    environment = runtime_r2.certify_runtime_environment_r2()
    live_model = validate_live_model(preregistration)
    if ATTEMPT_PATH.exists() or RUN_DIRECTORY.exists():
        raise RuntimeError("V34C requires fresh exclusive attempt and run paths")
    prelaunch_idle = runtime_v33a.assert_all_four_gpus_idle_v33a()
    attempt = _seal({
        "schema": "eggroll-es-v34c-durable-launch-attempt",
        "status": "launching",
        "phase": "before_trainer_creation",
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "recipe_sha256": recipe["content_sha256_before_self_field"],
        "committed_source_certificate_sha256": committed_source[
            "content_sha256_before_self_field"
        ],
        "runtime_environment_certificate_sha256": environment[
            "content_sha256_before_self_field"
        ],
        "live_model_audit_sha256": live_model["content_sha256_before_self_field"],
        "prelaunch_idle_certificate_sha256": prelaunch_idle[
            "content_sha256_before_self_field"
        ],
        "final_idle_certificate_sha256": None,
        "report_binding": None,
        "model_update_applied": False,
        "checkpoint_written": False,
        "evaluation_opened": False,
        "dataset_promotion_applied": False,
        "nontrain_surface_opened": False,
    })
    _write_attempt(attempt)
    if RUN_DIRECTORY.exists():
        final_idle = runtime_v33a.wait_for_final_gpu_idle_v33a(prelaunch_idle)
        attempt.update({
            "status": "failed",
            "phase": "fresh_run_reservation_race_after_attempt_claim",
            "failure_type": "FreshRunReservationError",
            "failure_sha256": canonical_sha256(
                "V34C run directory appeared after attempt claim"
            ),
            "final_idle_certificate_sha256": final_idle[
                "content_sha256_before_self_field"
            ],
            "all_four_gpus_idle_after_cleanup": True,
        })
        _rewrite_attempt(_seal(_without_self(attempt)))
        raise RuntimeError("V34C run directory appeared after attempt claim")
    trainer = None
    unit_scores = None
    configuration = preanalysis_audit = None
    failure = None
    final_idle = None
    try:
        runtime_v23a.base.set_seed(43)
        trainer = make_trainer(preregistration, layer_bundle)
        configuration = trainer.configure_replacement_fraction_v34c(
            preregistration, panel_bundle, layer_bundle
        )
        unit_scores, preanalysis_audit = trainer.capture_raw_arms_v34c()
    except BaseException as error:
        failure = error
    finally:
        if trainer is not None:
            try:
                runtime_v23a.base.close_trainer(trainer)
            except BaseException as cleanup_error:
                if failure is None:
                    failure = cleanup_error
        try:
            final_idle = runtime_v33a.wait_for_final_gpu_idle_v33a(prelaunch_idle)
        except BaseException as idle_error:
            if failure is None:
                failure = idle_error
    if failure is not None:
        attempt.update({
            "status": "failed",
            "phase": "after_fail_closed_cleanup",
            "failure_type": type(failure).__name__,
            "failure_sha256": canonical_sha256({
                "type": type(failure).__name__,
                "phase": "gpu_capture_or_cleanup",
            }),
            "final_idle_certificate_sha256": (
                None if final_idle is None
                else final_idle["content_sha256_before_self_field"]
            ),
            "all_four_gpus_idle_after_cleanup": final_idle is not None,
        })
        _rewrite_attempt(_seal(_without_self(attempt)))
        raise failure
    try:
        postcleanup_bindings = recheck_postcleanup_bindings(
            preregistration,
            panel_bundle,
            layer_bundle,
            implementation,
            recipe,
        )
        integrity = runtime_integrity_after_cleanup()
        summary = mechanics_v34b.build_compact_summary(
            unit_scores, panel_bundle, integrity
        )
        unit_scores = None
        gate = mechanics_v34b.evaluate_gate(summary)
        report = _seal({
            "schema": "eggroll-es-v401-replacement-fraction-report-v34c",
            "status": "completed_train_only_alpha_zero_no_update",
            "implementation_bundle_sha256": implementation["bundle_sha256"],
            "recipe_sha256": recipe["content_sha256_before_self_field"],
            "committed_source_certificate_sha256": committed_source[
                "content_sha256_before_self_field"
            ],
            "runtime_environment_certificate_sha256": environment[
                "content_sha256_before_self_field"
            ],
            "live_model_audit_sha256": live_model[
                "content_sha256_before_self_field"
            ],
            "prelaunch_idle_certificate_sha256": prelaunch_idle[
                "content_sha256_before_self_field"
            ],
            "final_idle_certificate_sha256": final_idle[
                "content_sha256_before_self_field"
            ],
            "configuration": configuration,
            "preanalysis_runtime_audit": preanalysis_audit,
            "postcleanup_binding_recheck_sha256": postcleanup_bindings[
                "content_sha256_before_self_field"
            ],
            "summary": summary,
            "gate": gate,
            "all_four_gpus_idle_after_cleanup": True,
            "model_update_applied": False,
            "checkpoint_written": False,
            "evaluation_opened": False,
            "dataset_promotion_applied": False,
            "nontrain_surface_opened": False,
            "direct_action_taken": False,
        })
        assert_compact(report)
        runtime_v23a._exclusive_write_json_v23a(REPORT_PATH, report)
        attempt.update({
            "status": "complete",
            "phase": "after_compact_report_and_final_gpu_cleanup",
            "final_idle_certificate_sha256": final_idle[
                "content_sha256_before_self_field"
            ],
            "report_binding": {
                "path": str(REPORT_PATH),
                "file_sha256": file_sha256(REPORT_PATH),
                "content_sha256": report["content_sha256_before_self_field"],
            },
            "all_four_gpus_idle_after_cleanup": True,
        })
        _rewrite_attempt(_seal(_without_self(attempt)))
        return report
    except BaseException as error:
        unit_scores = None
        attempt.update({
            "status": "failed",
            "phase": "after_cleanup_during_compact_analysis_or_report",
            "failure_type": type(error).__name__,
            "failure_sha256": canonical_sha256({
                "type": type(error).__name__,
                "phase": "compact_analysis_or_report",
            }),
            "final_idle_certificate_sha256": final_idle[
                "content_sha256_before_self_field"
            ],
            "all_four_gpus_idle_after_cleanup": True,
        })
        _rewrite_attempt(_seal(_without_self(attempt)))
        raise


def validate_runtime_args(args, preregistration, implementation, recipe):
    load_preregistration()
    if any(os.environ.get(key) for key in runtime_v23a.MOE_OVERRIDE_ENVIRONMENT_V23A):
        raise ValueError("V34C rejects external MoE backend overrides")
    if os.environ.get("VLLM_BATCH_INVARIANT") not in (None, "0"):
        raise ValueError("V34C rejects the vLLM batch-invariant backend swap")
    for expected, actual, label in (
        (
            args.expected_implementation_bundle_sha256,
            implementation["bundle_sha256"],
            "implementation bundle",
        ),
        (
            args.expected_recipe_sha256,
            recipe["content_sha256_before_self_field"],
            "recipe",
        ),
    ):
        if not args.v34c_dry_run and expected is None:
            raise ValueError(f"V34C real launch requires exact expected {label} hash")
        if expected is not None and expected != actual:
            raise ValueError(f"V34C expected {label} hash changed")
    if not args.v34c_dry_run and args.expected_source_commit is None:
        raise ValueError("V34C real launch requires exact expected source commit")
    if args.v34c_dry_run:
        return None
    return certify_committed_source(implementation, args.expected_source_commit)


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v34c-dry-run", action="store_true")
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-recipe-sha256")
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    assert_train_only_argv(argv)
    args = _parser().parse_args(argv)
    preregistration = load_preregistration()
    panel_bundle = mechanics_v34b.materialize_paired_panel_bundle()
    layer_bundle = load_layer_bundle(preregistration)
    implementation = implementation_identity()
    recipe = recipe_v34c(
        preregistration, panel_bundle, layer_bundle, implementation
    )
    committed_source = validate_runtime_args(
        args, preregistration, implementation, recipe
    )
    if args.v34c_dry_run:
        payload = _seal({
            "schema": "eggroll-es-v401-replacement-fraction-gpu-dry-run-v34c",
            "frozen_v34b_commit": V34B_COMMIT,
            "implementation_bundle_sha256": implementation["bundle_sha256"],
            "recipe_sha256": recipe["content_sha256_before_self_field"],
            "direction_seed_list_sha256": recipe["perturbation"][
                "direction_seed_list_sha256"
            ],
            "signed_schedule_sha256": recipe["perturbation"][
                "signed_schedule_sha256"
            ],
            "request_budget": recipe["request_budget"],
            "fresh_exclusive_attempt_and_run_required": True,
            "committed_clean_source_and_exact_cli_hashes_required": True,
            "all_four_tp1_engines_every_signed_wave": True,
            "exact_restore_boundary_unselected_origin_A_B_C_activity_cleanup_required": True,
            "fractions_derived_algebraically_with_zero_requests": True,
            "compact_aggregate_persistence_only": True,
            "direct_adoption_update_checkpoint_evaluation_dataset_fp8_or_nontrain_authority": False,
            "gpu_launched": False,
            "runtime_launched": False,
            "semantic_or_nontrain_content_opened": False,
        })
        assert_compact(payload)
        print(json.dumps(payload, sort_keys=True))
        return payload
    return run_exact_v34c(
        preregistration,
        panel_bundle,
        layer_bundle,
        implementation,
        recipe,
        committed_source,
    )


if __name__ == "__main__":
    main()
