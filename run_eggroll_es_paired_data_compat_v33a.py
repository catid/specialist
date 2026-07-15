#!/usr/bin/env python3
"""Fail-closed four-GPU train-only runtime for immutable V33A."""

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
import time
from pathlib import Path

import numpy as np

import eggroll_es_paired_data_compat_preregistration_v33a as prereg_v33a
import eggroll_es_worker_v33a as worker_v33a
import run_eggroll_es_insertion_stability_v23a as runtime_v23a
import run_eggroll_es_insertion_stability_v23a_retry_r2 as runtime_r2
import run_eggroll_es_paired_data_compat_v17a as runtime_v17a
import train_eggroll_es_paired_data_compat_v33a as mechanics_v33a


ROOT = Path(__file__).resolve().parent
PREREG_PATH_V33A = prereg_v33a.PREREGISTRATION_PATH
PREREG_FILE_SHA256_V33A = mechanics_v33a.PREREGISTRATION_FILE_SHA256_V33A
PREREG_CONTENT_SHA256_V33A = mechanics_v33a.PREREGISTRATION_CONTENT_SHA256_V33A
EXPERIMENT_NAME_V33A = (
    "s6_v33a_paired_production_v364_train_only_runtime_basis20261008"
)
OUTPUT_DIRECTORY_V33A = (ROOT / "experiments/eggroll_es_hpo/runs").resolve()
ATTEMPT_NAME_V33A = f".{EXPERIMENT_NAME_V33A}.launch_attempt.json"
REPORT_NAME_V33A = "paired_production_v364_compatibility_v33a.json"
WORKER_EXTENSION_V33A = (
    "eggroll_es_worker_v33a.PairedDataCompatWorkerExtensionV33A"
)
TEST_PATH_V33A = (
    ROOT / "test_run_eggroll_es_paired_data_compat_v33a.py"
).resolve()
MECHANICS_TEST_PATH_V33A = (
    ROOT / "test_train_eggroll_es_paired_data_compat_v33a.py"
).resolve()
BF16_CONFIG_SHA256_V33A = (
    "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
)
BF16_INDEX_SHA256_V33A = (
    "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"
)
EXPECTED_GPU_NAME_V33A = (
    "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
)
EXPECTED_DRIVER_VERSION_V33A = "610.43.02"
EXPECTED_GPU_IDENTITIES_V33A = {
    0: {
        "nvml_uuid": "GPU-4c394fc5-b18f-6622-ca94-f7fbd7112927",
        "pci_bus_id": "00000000:01:00.0",
        "total_bytes": 102_641_958_912,
    },
    1: {
        "nvml_uuid": "GPU-f10c2baf-536b-1d40-cd4b-25b202ae0ded",
        "pci_bus_id": "00000000:21:00.0",
        "total_bytes": 102_641_958_912,
    },
    2: {
        "nvml_uuid": "GPU-04cde663-7c53-2f18-3ec4-1699820e2640",
        "pci_bus_id": "00000000:C1:00.0",
        "total_bytes": 102_641_958_912,
    },
    3: {
        "nvml_uuid": "GPU-972bf85d-1b32-2d1b-20f6-babc4c804999",
        "pci_bus_id": "00000000:F1:00.0",
        "total_bytes": 102_641_958_912,
    },
}
BOUND_FILES_V33A = {
    "candidate_v364": (
        prereg_v33a.frame_v33a.CANDIDATE_PATH,
        prereg_v33a.frame_v33a.CANDIDATE_SHA256,
    ),
    "candidate_manifest_v364": (
        prereg_v33a.frame_v33a.CANDIDATE_MANIFEST_PATH,
        prereg_v33a.frame_v33a.CANDIDATE_MANIFEST_SHA256,
    ),
    "candidate_projection_v364": (
        prereg_v33a.frame_v33a.CANDIDATE_PROJECTION_PATH,
        prereg_v33a.frame_v33a.CANDIDATE_PROJECTION_SHA256,
    ),
    "candidate_freeze_tests_v364": (
        prereg_v33a.frame_v33a.CANDIDATE_TEST_PATH,
        prereg_v33a.frame_v33a.CANDIDATE_TEST_SHA256,
    ),
    "production_v13_source": (
        prereg_v33a.frame_v33a.PRODUCTION_PATH,
        prereg_v33a.frame_v33a.PRODUCTION_SHA256,
    ),
    "v25a_promising_unconfirmed_aggregate": (
        prereg_v33a.V25A_AGGREGATE_EVIDENCE_PATH,
        prereg_v33a.V25A_AGGREGATE_EVIDENCE_FILE_SHA256,
    ),
    "frame_builder_v33a": (
        ROOT / "build_eggroll_es_joint_panels_v33a.py",
        prereg_v33a.FRAME_BUILDER_SHA256,
    ),
    "frame_tests_v33a": (
        ROOT / "test_build_eggroll_es_joint_panels_v33a.py",
        prereg_v33a.FRAME_TEST_SHA256,
    ),
    "frame_manifest_v33a": (
        prereg_v33a.FRAME_PATH,
        prereg_v33a.FRAME_FILE_SHA256,
    ),
    "prereg_module_v33a": (
        Path(prereg_v33a.__file__).resolve(),
        "becaa29e97f584d5787866a921eb9259ea6bad6aeb6de208f45b0d78d3390bf1",
    ),
    "prereg_tests_v33a": (
        ROOT / "test_eggroll_es_paired_data_compat_preregistration_v33a.py",
        "01a3c4452a095c0d325b452122ce99e65377650e2556e7f5351b76e8c54afc15",
    ),
    "preregistration_v33a": (
        PREREG_PATH_V33A,
        PREREG_FILE_SHA256_V33A,
    ),
}
IMPLEMENTATION_PATHS_V33A = {
    **{key: path for key, (path, _digest) in BOUND_FILES_V33A.items()},
    "reference_mechanics_v17a": Path(runtime_v17a.mechanics_v17a.__file__).resolve(),
    "reference_runtime_v17a": Path(runtime_v17a.__file__).resolve(),
    "worker_v33a": Path(worker_v33a.__file__).resolve(),
    "worker_tests_v33a": ROOT / "test_eggroll_es_worker_v33a.py",
    "mechanics_v33a": Path(mechanics_v33a.__file__).resolve(),
    "mechanics_tests_v33a": MECHANICS_TEST_PATH_V33A,
    "runtime_v33a": Path(__file__).resolve(),
    "runtime_tests_v33a": TEST_PATH_V33A,
}
FORBIDDEN_ARGV_TOKENS_V33A = (
    "checkpoint", "update", "heldout", "validation", "ood", "benchmark",
    "eval", "save-model", "train-dataset", "promotion", "union",
)
FORBIDDEN_PERSISTED_KEYS_V33A = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "prompt_token_ids", "unit_scores", "responses", "coefficients",
    "bootstrap_replicates", "bootstrap_draws", "row_content",
    "unit_id", "unit_ids", "row_index", "row_sha256", "document_sha256",
    "pairing_anchor", "pairing_anchors", "weights", "strata",
}
ALLOWED_UNTRACKED_PREFIXES_V33A = (
    "experiments/dataset_probes/",
)
ALLOWED_UNTRACKED_PATHS_V33A = frozenset({
    "experiments/gpu_utilization_v31_base_qwen36_train_reward_probe.jsonl",
})
CURATOR_UNTRACKED_PATTERN_V33A = re.compile(
    r"data/manual_reviews/context_merit_audit_v([0-9]+)/.+\Z"
)
WORKTREE_ALLOWLIST_CONTRACT_V33A = {
    "curator_snapshot_directories": (
        "data/manual_reviews/context_merit_audit_vN/ where N >= 390"
    ),
    "path_prefixes": list(ALLOWED_UNTRACKED_PREFIXES_V33A),
    "exact_paths": sorted(ALLOWED_UNTRACKED_PATHS_V33A),
    "tracked_changes_allowed": False,
    "other_untracked_paths_allowed": False,
}


canonical_sha256 = prereg_v33a.canonical_sha256
file_sha256 = prereg_v33a.file_sha256
anchor_v4 = runtime_v23a.anchor_v4
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


def _assert_compact(value):
    overlap = FORBIDDEN_PERSISTED_KEYS_V33A & set(_recursive_keys(value))
    if overlap:
        raise RuntimeError(f"v33a compact output contains forbidden keys: {sorted(overlap)}")


def _assert_train_only_argv(argv):
    for token in argv:
        lowered = str(token).lower()
        if any(value in lowered for value in FORBIDDEN_ARGV_TOKENS_V33A):
            raise ValueError(f"v33a rejects forbidden runtime surface: {token}")


def _verify_bound_files_v33a():
    identities = {}
    for key, (path, expected) in BOUND_FILES_V33A.items():
        path = Path(path).resolve()
        relative = path.relative_to(ROOT).as_posix()
        if file_sha256(path) != expected:
            raise RuntimeError("v33a immutable train-only input changed")
        identities[key] = {
            "relative_path": relative,
            "file_sha256": expected,
        }
    return identities


def _is_allowed_untracked_v33a(relative):
    if relative in ALLOWED_UNTRACKED_PATHS_V33A or any(
        relative.startswith(prefix) for prefix in ALLOWED_UNTRACKED_PREFIXES_V33A
    ):
        return True
    match = CURATOR_UNTRACKED_PATTERN_V33A.fullmatch(relative)
    return match is not None and int(match.group(1)) >= 390


def _validate_worktree_status_v33a(raw_status):
    if not isinstance(raw_status, str):
        raise TypeError("v33a worktree status must be text")
    allowed_untracked = []
    rejected = []
    for line in raw_status.splitlines():
        if len(line) < 4 or line[2] != " ":
            raise RuntimeError("v33a worktree status record changed")
        status, relative = line[:2], line[3:]
        if status == "??" and _is_allowed_untracked_v33a(relative):
            allowed_untracked.append(relative)
        else:
            rejected.append({"status": status, "relative_path": relative})
    if rejected:
        raise RuntimeError(
            "v33a real launch requires a clean worktree outside the explicit "
            "untracked allowlist"
        )
    return {
        "all_tracked_files_clean": True,
        "only_explicitly_allowlisted_untracked_paths_present": True,
        "allowed_untracked_entry_count": len(allowed_untracked),
        "allowed_untracked_entries_sha256": canonical_sha256(
            sorted(allowed_untracked)
        ),
        "allowlist_contract_sha256": canonical_sha256(
            WORKTREE_ALLOWLIST_CONTRACT_V33A
        ),
    }


def _certify_real_launch_committed_source_v33a():
    relative_paths = sorted({
        Path(path).resolve().relative_to(ROOT).as_posix()
        for path in IMPLEMENTATION_PATHS_V33A.values()
        if Path(path).resolve().is_relative_to(ROOT)
    })
    for relative in relative_paths:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for cached in (False, True):
            command = ["git", "diff", "--quiet"]
            if cached:
                command.append("--cached")
            command.extend(["--", relative])
            subprocess.run(command, cwd=ROOT, check=True)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    if len(head) != 40:
        raise RuntimeError("v33a committed source head changed")
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
    )
    worktree = _validate_worktree_status_v33a(status)
    return _seal({
        "schema": "eggroll-es-committed-clean-source-certificate-v33a",
        "git_head": head,
        **worktree,
    })


def load_preregistration_v33a():
    _verify_bound_files_v33a()
    value = json.loads(PREREG_PATH_V33A.read_text(encoding="utf-8"))
    if (
        file_sha256(PREREG_PATH_V33A) != PREREG_FILE_SHA256_V33A
        or value.get("content_sha256_before_self_field")
        != PREREG_CONTENT_SHA256_V33A
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(value))
        or value.get("status") != "preregistered_runtime_not_yet_authorized"
        or value.get("required_runtime_adapter", {}).get(
            "runtime_launch_authorized"
        ) is not False
    ):
        raise RuntimeError("v33a immutable preregistration changed")
    prereg_v33a.validate_sha_fields(value)
    return value


def implementation_identity_v33a():
    inherited = runtime_r2.implementation_identity_r2()
    bound_files = _verify_bound_files_v33a()
    files = dict(inherited["files"])
    for key, path in IMPLEMENTATION_PATHS_V33A.items():
        files[key] = {
            "path": str(Path(path).resolve()),
            "file_sha256": file_sha256(path),
        }
    return {
        "files": files,
        "inherited_v23a_r2_bundle_sha256": inherited["bundle_sha256"],
        "immutable_bound_files": bound_files,
        "v33a_overlay_bundle_sha256": canonical_sha256({
            key: files[key] for key in IMPLEMENTATION_PATHS_V33A
        }),
        "bundle_sha256": canonical_sha256(files),
    }


def seed_projection_contract_v33a(preregistration):
    seeds = preregistration["frozen_recipe"]["perturbation_basis"][
        "direction_seeds"
    ]
    projections = [
        {"full_seed": seed, "numpy_legacy_seed": seed % (2**32)}
        for seed in seeds
    ]
    if (
        seeds != prereg_v33a.perturbation_seeds()
        or len(seeds) != 64
        or len(set(seeds)) != 64
        or len({item["numpy_legacy_seed"] for item in projections}) != 64
        or any(item["numpy_legacy_seed"] == 0 for item in projections)
    ):
        raise RuntimeError("v33a perturbation seed projection changed")
    return {
        "schema": "eggroll-es-v33a-seed-projection-contract",
        "direction_seed_list_sha256": canonical_sha256(seeds),
        "full_to_numpy_projection_sha256": canonical_sha256(projections),
        "direction_count": 64,
        "numpy_projection_unique_count": 64,
        "only_numpy_legacy_seed_is_projected": True,
        "worker_extension": WORKER_EXTENSION_V33A,
    }


def load_layer_bundle_v33a(preregistration):
    plan = preregistration["frozen_recipe"]["layer_plan"]
    return anchor_v4.load_frozen_layer_plan_v4(
        plan["path"],
        expected_file_sha256=plan["file_sha256"],
        expected_plan_sha256=plan["plan_sha256"],
        expected_model_config_sha256=BF16_CONFIG_SHA256_V33A,
    )


def validate_live_model_v33a(preregistration):
    model = Path(preregistration["frozen_recipe"]["model"]).resolve()
    if (
        file_sha256(model / "config.json") != BF16_CONFIG_SHA256_V33A
        or file_sha256(model / "model.safetensors.index.json")
        != BF16_INDEX_SHA256_V33A
    ):
        raise RuntimeError("v33a live BF16 model identity changed")
    return _seal({
        "schema": "eggroll-es-v33a-live-model-audit",
        "model_path": str(model),
        "config_sha256": BF16_CONFIG_SHA256_V33A,
        "index_sha256": BF16_INDEX_SHA256_V33A,
    })


def _observe_all_four_gpus_v33a():
    import pynvml

    pynvml.nvmlInit()
    rows = []
    try:
        if pynvml.nvmlDeviceGetCount() != 4:
            raise RuntimeError("v33a requires exactly four physical GPUs")
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode("ascii")
        for gpu_id in range(4):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            processes = []
            for function_name in (
                "nvmlDeviceGetComputeRunningProcesses",
                "nvmlDeviceGetGraphicsRunningProcesses",
            ):
                function = getattr(pynvml, function_name, None)
                if function is not None:
                    try:
                        processes.extend(function(handle))
                    except pynvml.NVMLError_NotSupported:
                        pass
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            pci_bus_id = pynvml.nvmlDeviceGetPciInfo(handle).busId
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            if isinstance(uuid, bytes):
                uuid = uuid.decode("ascii")
            if isinstance(pci_bus_id, bytes):
                pci_bus_id = pci_bus_id.decode("ascii")
            rows.append({
                "physical_gpu_id": gpu_id,
                "name": str(name),
                "driver_version": str(driver),
                "nvml_uuid": str(uuid),
                "pci_bus_id": str(pci_bus_id),
                "total_bytes": int(memory.total),
                "running_process_count": len({
                    int(item.pid) for item in processes
                }),
            })
    finally:
        pynvml.nvmlShutdown()
    return {
        "gpus": rows,
        "all_four_idle": all(
            item["running_process_count"] == 0 for item in rows
        ),
    }


def _physical_gpu_identity_v33a(certificate):
    result = {
        item["physical_gpu_id"]: {
            "name": item.get("name"),
            "driver_version": item.get("driver_version"),
            "nvml_uuid": item.get("nvml_uuid"),
            "pci_bus_id": item.get("pci_bus_id"),
            "total_bytes": item.get("total_bytes"),
        }
        for item in certificate.get("gpus", [])
    }
    if (
        set(result) != set(range(4))
        or any(
            item["name"] != EXPECTED_GPU_NAME_V33A
            or item["driver_version"] != EXPECTED_DRIVER_VERSION_V33A
            or item["nvml_uuid"]
            != EXPECTED_GPU_IDENTITIES_V33A[gpu_id]["nvml_uuid"]
            or str(item["pci_bus_id"]).upper()
            != EXPECTED_GPU_IDENTITIES_V33A[gpu_id]["pci_bus_id"].upper()
            or item["total_bytes"]
            != EXPECTED_GPU_IDENTITIES_V33A[gpu_id]["total_bytes"]
            for gpu_id, item in result.items()
        )
    ):
        raise RuntimeError("v33a physical GPU identity changed")
    return result


def assert_all_four_gpus_idle_v33a():
    observation = _observe_all_four_gpus_v33a()
    _physical_gpu_identity_v33a(observation)
    if observation.get("all_four_idle") is not True:
        raise RuntimeError("v33a requires all four GPUs idle before claim")
    return _seal({
        "schema": "eggroll-es-v33a-prelaunch-idle-certificate",
        **observation,
    })


def wait_for_final_gpu_idle_v33a(
    expected_idle_certificate, *, timeout_seconds=30.0, interval_seconds=0.5,
):
    if timeout_seconds != 30.0 or interval_seconds != 0.5:
        raise RuntimeError("v33a final GPU cleanup polling contract changed")
    expected = _physical_gpu_identity_v33a(expected_idle_certificate)
    started = time.monotonic()
    polls = 0
    while True:
        observation = _observe_all_four_gpus_v33a()
        polls += 1
        if _physical_gpu_identity_v33a(observation) != expected:
            raise RuntimeError("v33a final cleanup physical GPU identity changed")
        elapsed = time.monotonic() - started
        if observation.get("all_four_idle") is True:
            return _seal({
                "schema": "eggroll-es-v33a-final-idle-certificate",
                **observation,
                "poll_count": polls,
                "elapsed_milliseconds": int(round(elapsed * 1000.0)),
                "bounded_async_cleanup_wait": True,
            })
        remaining = timeout_seconds - elapsed
        if remaining <= 0:
            raise RuntimeError("v33a final GPU cleanup exceeded 30 seconds")
        time.sleep(min(interval_seconds, remaining))


def recipe_v33a(
    preregistration, panel_bundle, layer_bundle, implementation,
):
    mechanics_v33a.validate_paired_panel_bundle_v33a(panel_bundle)
    schedule = mechanics_v33a.resident_signed_wave_schedule_v33a()
    panels = {}
    for name in mechanics_v33a.PANEL_NAMES_V33A:
        panel = panel_bundle["panels"][name]
        panels[name] = {
            "role": panel["role"],
            "paired_units": 39,
            "ordered_unit_identity_sha256": panel[
                "ordered_unit_identity_sha256"
            ],
            "stratum_counts": {
                stratum: panel["strata"].count(stratum)
                for stratum in mechanics_v33a.STRATA_V33A
            },
            "ordered_version_row_identity_sha256": {
                version: panel["versions"][version]["ordered_row_identity_sha256"]
                for version in mechanics_v33a.VERSIONS_V33A
            },
            "pairing_anchor_counts": {
                anchor: panel["pairing_anchors"].count(anchor)
                for anchor in (
                    "shared_document", "joint_component_cross_side_link",
                )
            },
        }
    value = {
        "schema": "eggroll-es-paired-data-compat-runtime-recipe-v33a",
        "experiment_name": EXPERIMENT_NAME_V33A,
        "preregistration": {
            "path": str(PREREG_PATH_V33A),
            "file_sha256": PREREG_FILE_SHA256_V33A,
            "content_sha256": PREREG_CONTENT_SHA256_V33A,
            "candidate_freeze_commit": preregistration["inputs"][
                "candidate_freeze_commit"
            ],
            "real_launch_requires_committed_clean_source_bundle": True,
        },
        "prior_aggregate_evidence": copy.deepcopy(
            preregistration["audited_prior_aggregate_evidence"]
        ),
        "frame": {
            **copy.deepcopy(panel_bundle["frame"]),
            "globally_disjoint_joint_units": True,
            "fixed_side_representatives": True,
        },
        "sources": copy.deepcopy(panel_bundle["sources"]),
        "materialized_panel_bundle_content_sha256": panel_bundle[
            "content_sha256_before_self_field"
        ],
        "panels": panels,
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
            "basis_seed": prereg_v33a.PERTURBATION_BASIS_SEED,
            "direction_seed_list_sha256": canonical_sha256(
                prereg_v33a.perturbation_seeds()
            ),
            "preregistered_signed_schedule_sha256": preregistration[
                "frozen_recipe"
            ]["perturbation_basis"]["signed_population_schedule_sha256"],
            "runtime_resident_schedule_sha256": canonical_sha256(schedule),
            "synchronized_four_engine_signed_waves": 32,
            "engine_signed_direction_evaluations": 128,
            "all_four_engines_score_both_versions_every_signed_wave": True,
            "same_resident_perturbation_scores_both_versions": True,
            "alternating_version_order": [
                item["resident_version_order"] for item in schedule
            ],
            "restore_once_after_both_versions_each_signed_wave": True,
        },
        "seed_projection": seed_projection_contract_v33a(preregistration),
        "scoring": {
            "objective": "per_example_mean_full_gold_answer_token_logprob",
            "dense_reward_config_sha256": anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4,
            "fixed_version_batches_materialized_once": True,
            "token_identity_audited_before_a_b_c_and_population": True,
            "max_tokens": 1,
            "prompt_logprobs": 1,
            "detokenize": False,
        },
        "matched_full_context_guard": {
            "phase_order": [
                "full_context_a", "repeat_full_context_b",
                "all_32_synchronized_four_engine_signed_waves",
                "full_population_boundary_audit",
                "post_population_full_context_c",
            ],
            "all_four_engines_score_both_versions_each_phase": True,
            "a_equals_b_before_first_perturbation": True,
            "a_equals_c_after_population_boundary": True,
            "guard_phases_excluded_from_estimator_and_bootstrap": True,
        },
        "request_accounting": {
            "paired_units_per_panel": 39,
            "panels": 5,
            "requests_per_version_per_engine_call": 195,
            "versions_per_signed_wave": 2,
            "requests_per_engine_per_signed_wave": 390,
            "requests_all_engines_per_signed_wave": 1_560,
            "synchronized_four_engine_signed_waves": 32,
            "engine_signed_direction_evaluations": 128,
            "perturbed_requests_all_engines": 49_920,
            "full_context_requests_all_engines_per_phase": 1_560,
            "full_context_phase_count": 3,
            "full_context_requests_all_engines": 4_680,
            "total_generation_requests": 54_600,
        },
        "bootstrap": copy.deepcopy(preregistration["analysis"]["bootstrap"]),
        "endpoints": list(prereg_v33a.ENDPOINTS),
        "promotion_gate": copy.deepcopy(preregistration["promotion_gate"]),
        "restoration": {
            "exact_selected_reference_restore_after_each_signed_wave": True,
            "selected_and_unselected_population_boundary_after_all_waves": True,
        },
        "hardware": {
            "gpu_ids": [0, 1, 2, 3],
            "engine_count": 4,
            "tp_per_engine": 1,
            "fixed_rank_to_physical_gpu": True,
            "expected_gpu_name": EXPECTED_GPU_NAME_V33A,
            "expected_driver_version": EXPECTED_DRIVER_VERSION_V33A,
            "expected_physical_gpu_identity_sha256": canonical_sha256(
                EXPECTED_GPU_IDENTITIES_V33A
            ),
            "all_four_gpus_idle_immediately_before_attempt_claim": True,
            "bounded_final_idle_certificate_required_after_cleanup": True,
            "final_idle_timeout_seconds": 30.0,
            "final_idle_poll_interval_seconds": 0.5,
        },
        "source_cleanliness": copy.deepcopy(
            WORKTREE_ALLOWLIST_CONTRACT_V33A
        ),
        "worker_extension": WORKER_EXTENSION_V33A,
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "output_directory": str(OUTPUT_DIRECTORY_V33A),
        "fresh_exclusive_paths": {
            "attempt_name": ATTEMPT_NAME_V33A,
            "run_directory_name": EXPERIMENT_NAME_V33A,
            "report_name": REPORT_NAME_V33A,
        },
        "authority": {
            "model_update_allowed": False,
            "checkpoint_write_allowed": False,
            "evaluation_allowed": False,
            "dataset_promotion_allowed": False,
            "pass_authorizes_only_separate_fresh_basis_v364_confirmation_preregistration": True,
        },
    }
    return _seal(value)


def validate_runtime_v33a(args, preregistration, implementation, recipe):
    load_preregistration_v33a()
    if preregistration.get("content_sha256_before_self_field") != PREREG_CONTENT_SHA256_V33A:
        raise ValueError("v33a preregistration changed")
    if any(
        os.environ.get(key) for key in runtime_v23a.MOE_OVERRIDE_ENVIRONMENT_V23A
    ):
        raise ValueError("v33a rejects external MoE backend overrides")
    if os.environ.get("VLLM_BATCH_INVARIANT") not in (None, "0"):
        raise ValueError("v33a rejects the vLLM batch-invariant backend swap")
    if recipe.get("content_sha256_before_self_field") != canonical_sha256(
        _without_self(recipe)
    ):
        raise ValueError("v33a runtime recipe changed")
    for expected, actual, label in (
        (
            args.expected_implementation_bundle_sha256,
            implementation["bundle_sha256"],
            "implementation",
        ),
        (
            args.expected_recipe_sha256,
            recipe["content_sha256_before_self_field"],
            "recipe",
        ),
    ):
        if not args.v33a_dry_run and expected is None:
            raise ValueError(f"v33a real launch requires expected {label} hash")
        if expected is not None and expected != actual:
            raise ValueError(f"v33a {label} hash changed")
    if not args.v33a_dry_run:
        return _certify_real_launch_committed_source_v33a()
    return None


def _request_identity(prompt_items):
    return canonical_sha256([item["prompt_token_ids"] for item in prompt_items])


def _score_panel_outputs(dense_items, outputs):
    dense = anchor_v4.score_gold_answer_outputs_v4(dense_items, outputs)
    rewards = np.asarray([
        item["mean_answer_token_logprob"] for item in dense["examples"]
    ], dtype=np.float64)
    if rewards.shape != (39,) or not np.isfinite(rewards).all():
        raise RuntimeError("v33a per-unit dense score coverage changed")
    return rewards, canonical_sha256(dense)


def _phase_equal(reference, candidate):
    ref_scores, ref_commitments = reference
    scores, commitments = candidate
    return {
        "all_version_engine_panel_score_arrays_exact": all(
            np.array_equal(ref_scores[version], scores[version])
            for version in mechanics_v33a.VERSIONS_V33A
        ),
        "all_dense_result_commitments_exact": ref_commitments == commitments,
    }


def _phase_identity(phase):
    scores, commitments = phase
    identities = {}
    for version in mechanics_v33a.VERSIONS_V33A:
        values = np.asarray(scores[version], dtype=np.float64)
        if values.shape != (4, 5, 39) or not np.isfinite(values).all():
            raise RuntimeError("v33a full-context score geometry changed")
        identities[version] = {
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


def _validate_exact_references_v33a(references, selected):
    """Validate the v4 reference schema; rank mapping is certified separately."""
    if len(references) != 4 or len(selected) != 4:
        raise RuntimeError("v33a exact reference coverage changed")
    if any(
        reference.get("schema")
        != "eggroll-es-selected-exact-reference-state-v4"
        or reference.get("reference_generation") != 1
        or reference.get("fresh_for_population") is not True
        or not isinstance(reference.get("identity"), dict)
        or selected[rank].get("schema")
        != "eggroll-es-selected-exact-reference-check-v4"
        or selected[rank].get("passed") is not True
        or selected[rank].get("reference_generation") != 1
        or selected[rank].get("reference")
        != reference["identity"].get("selected")
        or selected[rank].get("current") != selected[rank].get("reference")
        for rank, reference in enumerate(references)
    ):
        raise RuntimeError("v33a exact reference certificate changed")
    if (
        len({canonical_sha256(item["reference"]) for item in selected}) != 1
        or len({
            canonical_sha256(reference["identity"])
            for reference in references
        }) != 1
    ):
        raise RuntimeError("v33a selected reference identity differs across engines")
    return canonical_sha256(selected[0]["reference"])


class PairedDataCompatRuntimeMixinV33A:
    def configure_paired_data_compat_v33a(
        self, preregistration, panel_bundle, layer_bundle,
    ):
        panel_bundle = mechanics_v33a.validate_paired_panel_bundle_v33a(
            copy.deepcopy(panel_bundle)
        )
        if (
            len(self.engines) != 4
            or self.n_vllm_engines != 4
            or self.n_gpu_per_vllm_engine != 1
            or self.population_size != 64
            or not math.isclose(float(self.sigma), 0.0003, rel_tol=0.0, abs_tol=0.0)
            or not math.isclose(float(self.alpha), 0.0, rel_tol=0.0, abs_tol=0.0)
        ):
            raise RuntimeError("v33a exact four-engine alpha-zero recipe changed")
        projection = seed_projection_contract_v33a(preregistration)
        seeds = preregistration["frozen_recipe"]["perturbation_basis"][
            "direction_seeds"
        ]
        seed_reports = [
            runtime_v23a._unwrap_one_v23a(item, "seed_projection_certificate_v23a_r1")
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
            or len({canonical_sha256(x) for x in seed_reports}) != 1
            or any(
                report.get("schema")
                != "eggroll-es-v33a-seed-projection-worker-certificate"
                or report.get("direction_count") != projection["direction_count"]
                or report.get("direction_seed_list_sha256")
                != projection["direction_seed_list_sha256"]
                or report.get("full_to_numpy_projection_sha256")
                != projection["full_to_numpy_projection_sha256"]
                or report.get("numpy_projection_unique_count") != 64
                or report.get("numpy_projection_contains_zero") is not False
                or report.get("python_random_receives_full_seed") is not True
                or report.get("torch_global_receives_full_seed") is not True
                or report.get("torch_cuda_all_receives_full_seed") is not True
                or report.get("explicit_torch_generator_receives_full_seed")
                is not True
                or report.get("only_numpy_legacy_seed_is_projected") is not True
                for report in seed_reports
            )
        ):
            raise RuntimeError("v33a worker seed certificates changed")
        devices = [
            runtime_v23a._unwrap_one_v23a(item, "runtime_device_identity_v23a")
            for item in self._resolve([
                engine.collective_rpc.remote(
                    "runtime_device_identity_v23a", args=(f"direction_slot_{rank}",)
                )
                for rank, engine in enumerate(self.engines)
            ])
        ]
        for rank, report in enumerate(devices):
            if (
                report.get("rank") != rank
                or report.get("world_size") != 4
                or report.get("arm") != f"direction_slot_{rank}"
                or report.get("cuda_visible_devices") != str(rank)
                or report.get("runtime_cuda_device") != 0
                or report.get("update_surfaces_closed") is not True
            ):
                raise RuntimeError("v33a fixed rank-to-GPU mapping changed")
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
                for rank, item in enumerate(installs)
            )
            or len({
                canonical_sha256(item.get("initial_identity"))
                for item in installs
            }) != 1
        ):
            raise RuntimeError("v33a layer-plan installation changed")
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
        selected_reference_identity_sha256 = _validate_exact_references_v33a(
            references, selected,
        )
        self._v33a_preregistration = copy.deepcopy(preregistration)
        self._v33a_panel_bundle = panel_bundle
        self._v33a_references = [item["identity"] for item in references]
        return _seal({
            "schema": "eggroll-es-paired-runtime-configuration-v33a",
            "device_identity_sha256": canonical_sha256(devices),
            "installation_sha256": canonical_sha256(installs),
            "seed_certificate_sha256": canonical_sha256(seed_reports),
            "selected_reference_identity_sha256": (
                selected_reference_identity_sha256
            ),
            "panel_bundle_content_sha256": panel_bundle[
                "content_sha256_before_self_field"
            ],
            "worker_extension": WORKER_EXTENSION_V33A,
            "model_update_allowed": False,
            "checkpoint_write_allowed": False,
            "evaluation_allowed": False,
        })

    def _prepared_fixed_batches_v33a(self):
        bundle = mechanics_v33a.validate_paired_panel_bundle_v33a(
            self._v33a_panel_bundle
        )
        prepared = {}
        request_identity = {}
        token_audit = {}
        for version in mechanics_v33a.VERSIONS_V33A:
            panels = {}
            prompt_items = []
            cursor = 0
            token_panels = {}
            for panel_name in mechanics_v33a.PANEL_NAMES_V33A:
                batch = bundle["panels"][panel_name]["versions"][version]
                prompts = [runtime_v23a.base.specialist_template(item) for item in batch["questions"]]
                dense_items = anchor_v4.prepare_gold_answer_items_v4(
                    self.tokenizer, prompts, batch["answers"]
                )
                current = [
                    {"prompt_token_ids": item["prompt_token_ids"]}
                    for item in dense_items
                ]
                panels[panel_name] = {
                    "dense_items": dense_items,
                    "slice": (cursor, cursor + len(current)),
                }
                prompt_items.extend(current)
                cursor += len(current)
                token_panels[panel_name] = {
                    "request_count": len(dense_items),
                    "dense_token_contract_sha256": canonical_sha256(dense_items),
                    "total_combined_tokens": sum(
                        len(item["prompt_token_ids"]) for item in dense_items
                    ),
                    "total_answer_tokens": sum(
                        item["answer_token_count"] for item in dense_items
                    ),
                    "all_within_frozen_1024_token_cap": all(
                        len(item["prompt_token_ids"]) <= 1024 for item in dense_items
                    ),
                }
            if cursor != 195 or any(
                item["request_count"] != 39
                or item["all_within_frozen_1024_token_cap"] is not True
                for item in token_panels.values()
            ):
                raise RuntimeError("v33a fixed version request/token count changed")
            prepared[version] = {"panels": panels, "prompt_items": prompt_items}
            request_identity[version] = _request_identity(prompt_items)
            token_audit[version] = token_panels
        self._v33a_fixed_request_identity = request_identity
        self._v33a_token_audit_sha256 = canonical_sha256(token_audit)
        return prepared, request_identity, token_audit

    def _assert_fixed_request_v33a(self, version, prepared):
        if (
            version not in mechanics_v33a.VERSIONS_V33A
            or _request_identity(prepared[version]["prompt_items"])
            != self._v33a_fixed_request_identity[version]
        ):
            raise RuntimeError("v33a fixed version request identity changed")

    def _generate_version_all_engines_v33a(self, version, prepared):
        self._assert_fixed_request_v33a(version, prepared)
        batches = self._resolve([
            engine.generate.remote(
                list(prepared[version]["prompt_items"]),
                self._dense_sampling_params_v4(0),
                use_tqdm=False,
            )
            for engine in self.engines
        ])
        if len(batches) != 4 or any(len(batch) != 195 for batch in batches):
            raise RuntimeError("v33a all-engine version generation incomplete")
        return batches

    def _score_version_batches_v33a(self, version, prepared, batches):
        scores = np.empty((4, 5, 39), dtype=np.float64)
        commitments = []
        for engine_index, batch in enumerate(batches):
            for panel_index, panel_name in enumerate(mechanics_v33a.PANEL_NAMES_V33A):
                panel = prepared[version]["panels"][panel_name]
                start, stop = panel["slice"]
                rewards, digest = _score_panel_outputs(
                    panel["dense_items"], batch[start:stop]
                )
                scores[engine_index, panel_index] = rewards
                commitments.append(digest)
        return scores, commitments

    def _full_context_phase_v33a(self, prepared):
        scores = {}
        commitments = {}
        for version in mechanics_v33a.VERSIONS_V33A:
            batches = self._generate_version_all_engines_v33a(version, prepared)
            scores[version], commitments[version] = self._score_version_batches_v33a(
                version, prepared, batches
            )
        return scores, commitments

    def _perturb_signed_wave_v33a(self, seeds, negate):
        if len(seeds) != 4:
            raise RuntimeError("v33a partial perturbation wave forbidden")
        raw = self._resolve([
            engine.collective_rpc.remote(
                "perturb_self_weights",
                args=(int(seed), 0.0003, bool(negate)),
            )
            for engine, seed in zip(self.engines, seeds)
        ])
        runtime_v23a._validate_perturbation_results_v23a(raw)

    def _score_resident_version_v33a(
        self, version, prepared, schedule_item, unit_scores, commitments,
    ):
        batches = self._generate_version_all_engines_v33a(version, prepared)
        scores, dense = self._score_version_batches_v33a(
            version, prepared, batches
        )
        version_index = mechanics_v33a.VERSIONS_V33A.index(version)
        sign_index = mechanics_v33a.SIGNS_V33A.index(schedule_item["sign"])
        for engine_index, direction_index in enumerate(
            schedule_item["engine_direction_indices"]
        ):
            unit_scores[version_index, :, sign_index, direction_index] = scores[
                engine_index
            ]
        commitments.extend(dense)
        return canonical_sha256(dense)

    def _restore_and_verify_v33a(self):
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
                item.get("schema")
                != "eggroll-es-selected-exact-reference-check-v4"
                or item.get("passed") is not True
                or item.get("reference_generation") != 1
                or item.get("reference")
                != self._v33a_references[rank]["selected"]
                or item.get("current") != item.get("reference")
                for rank, item in enumerate(checks)
            )
        ):
            raise RuntimeError("v33a exact selected restore changed")
        return canonical_sha256(checks)

    def _run_signed_wave_v33a(
        self, schedule_item, prepared, unit_scores, commitments,
    ):
        restore_hashes = []
        captures = mechanics_v33a.execute_paired_resident_signed_wave_v33a(
            schedule_item,
            perturb=lambda seeds, negate: self._perturb_signed_wave_v33a(
                seeds, negate
            ),
            score_version=lambda version: self._score_resident_version_v33a(
                version, prepared, schedule_item, unit_scores, commitments
            ),
            restore=lambda: restore_hashes.append(self._restore_and_verify_v33a()),
        )
        if len(restore_hashes) != 1 or tuple(captures) != tuple(
            schedule_item["resident_version_order"]
        ):
            raise RuntimeError("v33a resident signed-wave transaction changed")
        return restore_hashes[0]

    def _population_boundary_v33a(self):
        reports = [
            runtime_v23a._unwrap_one_v23a(item, "audit_population_completion_v4")
            for item in self._resolve([
                self.engines[rank].collective_rpc.remote(
                    "audit_population_completion_v4",
                    args=(4, 1, self._v33a_references[rank]["sha256"]),
                )
                for rank in range(4)
            ])
        ]
        if any(
            item.get("schema") != "eggroll-es-post-population-audit-v4"
            or item.get("passed") is not True
            or item.get("rank") != index
            or item.get("world_size") != 4
            or item.get("reference_generation") != 1
            or item.get("reference_sha256")
            != self._v33a_references[index]["sha256"]
            or item.get("current_identity") != self._v33a_references[index]
            for index, item in enumerate(reports)
        ):
            raise RuntimeError("v33a selected/unselected population boundary changed")
        return canonical_sha256(reports)

    def estimate_paired_data_compat_v33a(self):
        schedule = mechanics_v33a.resident_signed_wave_schedule_v33a()
        prepared, request_identity, token_audit = self._prepared_fixed_batches_v33a()
        phase_a = self._full_context_phase_v33a(prepared)
        phase_b = self._full_context_phase_v33a(prepared)
        a_b_equal = _phase_equal(phase_a, phase_b)
        if not all(a_b_equal.values()):
            raise RuntimeError("v33a full-context A/B guard is not exact")
        unit_scores = np.full((2, 5, 2, 64, 39), np.nan, dtype=np.float64)
        dense_commitments = []
        restore_hashes = []
        for schedule_item in schedule:
            restore_hashes.append(self._run_signed_wave_v33a(
                schedule_item, prepared, unit_scores, dense_commitments
            ))
        if (
            not np.isfinite(unit_scores).all()
            or len(dense_commitments) != 1_280
            or len(restore_hashes) != 32
        ):
            raise RuntimeError("v33a paired population capture incomplete")
        boundary_sha256 = self._population_boundary_v33a()
        phase_c = self._full_context_phase_v33a(prepared)
        a_c_equal = _phase_equal(phase_a, phase_c)
        if not all(a_c_equal.values()):
            raise RuntimeError("v33a full-context A/C guard is not exact")
        compact = mechanics_v33a.build_compact_estimator_summary_v33a(
            unit_scores, self._v33a_panel_bundle
        )
        unit_scores = None
        runtime_integrity = {
            "all_four_engines_scored_both_versions_every_signed_wave": True,
            "fixed_side_representatives_every_direction_and_sign": True,
            "same_resident_perturbation_scored_both_versions": True,
            "alternating_version_order_complete": True,
            "exact_selected_restore_after_every_signed_wave": True,
            "selected_and_unselected_population_boundary_passed": True,
            "full_context_a_b_equal_before_first_perturbation": True,
            "full_context_a_c_equal_after_population_boundary": True,
            "token_and_request_identity_audits_passed": True,
            "all_integrity_audits_passed": True,
        }
        summary = _seal({
            "schema": "eggroll-es-paired-data-compat-summary-v33a",
            "experiment_name": EXPERIMENT_NAME_V33A,
            "alpha": 0.0,
            "sigma": 0.0003,
            "frame_content_sha256": prereg_v33a.FRAME_CONTENT_SHA256,
            "perturbation_basis_content_sha256": self._v33a_preregistration[
                "frozen_recipe"
            ]["perturbation_basis"]["basis_content_sha256"],
            "runtime_integrity": runtime_integrity,
            "versions": compact["versions"],
            "paired_bootstrap": compact["paired_bootstrap"],
            "cross_version_direction_similarity_diagnostic": compact[
                "cross_version_direction_similarity_diagnostic"
            ],
            "persisted_response_vectors_rows_draws_or_replicates": False,
            "model_update_applied": False,
            "evaluation_opened": False,
        })
        gate = mechanics_v33a.evaluate_candidate_v33a(summary)
        guard = {
            "schema": "eggroll-es-v33a-full-context-a-b-c-guard",
            "phase_a": _phase_identity(phase_a),
            "phase_b": _phase_identity(phase_b),
            "phase_c": _phase_identity(phase_c),
            "a_b_exact": a_b_equal,
            "a_c_exact": a_c_equal,
            "all_four_engines_both_versions_each_phase": True,
            "excluded_from_estimator_and_bootstrap": True,
            "raw_scores_or_outputs_persisted": False,
        }
        audit = _seal({
            "schema": "eggroll-es-paired-runtime-compact-audit-v33a",
            "fixed_request_identity_sha256": canonical_sha256(request_identity),
            "token_audit_sha256": canonical_sha256(token_audit),
            "full_context_guard": guard,
            "signed_wave_schedule_sha256": canonical_sha256(schedule),
            "dense_result_commitments_sha256": canonical_sha256(dense_commitments),
            "restore_checks_sha256": canonical_sha256(restore_hashes),
            "population_boundary_audit_sha256": boundary_sha256,
            "synchronized_four_engine_signed_wave_count": 32,
            "engine_signed_direction_evaluation_count": 128,
            "perturbed_requests_all_engines": 49_920,
            "full_context_requests_all_engines": 4_680,
            "total_generation_requests": 54_600,
            "per_unit_scores_or_bootstrap_replicates_persisted": False,
            "model_update_applied": False,
            "evaluation_opened": False,
        })
        _assert_compact({"summary": summary, "gate": gate, "audit": audit})
        return summary, gate, audit

    @staticmethod
    def _closed_surface_v33a(*_args, **_kwargs):
        raise RuntimeError("v33a closes update checkpoint evaluation and union surfaces")

    configure_anchor = _closed_surface_v33a
    configure_train_panels_v13 = _closed_surface_v33a
    estimate_train_panels_v13 = _closed_surface_v33a
    estimate_step_coefficients = _closed_surface_v33a
    apply_seed_coefficients = _closed_surface_v33a
    train_step = _closed_surface_v33a
    evaluate_handle = _closed_surface_v33a
    evaluate_population_on_batch = _closed_surface_v33a
    eval_step = _closed_surface_v33a
    fit = _closed_surface_v33a


def load_runtime_trainer_v33a(preregistration, layer_bundle):
    model = preregistration["frozen_recipe"]["model"]
    parent = anchor_v4.load_trainer(layer_bundle)

    class PairedDataCompatRuntimeTrainerV33A(
        PairedDataCompatRuntimeMixinV33A, parent,
    ):
        def launch_engines(
            self, num_engines=4, n_gpu_per_vllm_engine=1,
            model_name="unused", precision="bfloat16",
        ):
            if num_engines != 4 or n_gpu_per_vllm_engine != 1:
                raise ValueError("v33a requires exactly four TP1 engines")
            import ray
            from es_at_scale.trainer.es_trainer import ESNcclLLM
            from ray.util.placement_group import placement_group
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

            pgs = [
                placement_group(
                    [{"GPU": 1, "CPU": 0}], strategy="PACK", lifetime="detached"
                )
                for _ in range(4)
            ]
            ray.get([pg.ready() for pg in pgs])
            strategies = [
                PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=0,
                )
                for pg in pgs
            ]
            engines = []
            for rank in range(4):
                engines.append(ray.remote(
                    num_cpus=0, num_gpus=1, scheduling_strategy=strategies[rank]
                )(ESNcclLLM).remote(
                    model=model,
                    tensor_parallel_size=1,
                    worker_extension_cls=WORKER_EXTENSION_V33A,
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
            return engines, pgs

    return PairedDataCompatRuntimeTrainerV33A


def _make_trainer_v33a(preregistration, layer_bundle):
    cls = load_runtime_trainer_v33a(preregistration, layer_bundle)
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
        experiment_name=EXPERIMENT_NAME_V33A,
        wandb_project="specialist-eggroll-es",
        save_best_models=False,
        reward_function_timeout=10,
        output_directory=str(OUTPUT_DIRECTORY_V33A),
    )


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v33a-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-recipe-sha256")
    return parser


def run_exact_v33a(
    preregistration, panel_bundle, layer_bundle, implementation, recipe,
    committed_clean_source,
):
    environment = runtime_r2.certify_runtime_environment_r2()
    attempt_path = OUTPUT_DIRECTORY_V33A / ATTEMPT_NAME_V33A
    run_dir = OUTPUT_DIRECTORY_V33A / EXPERIMENT_NAME_V33A
    report_path = run_dir / REPORT_NAME_V33A
    if attempt_path.exists() or run_dir.exists():
        raise RuntimeError("v33a requires fresh exclusive attempt and run paths")
    provenance = runtime_v23a._source_provenance_v23a(implementation)
    live_model = validate_live_model_v33a(preregistration)
    if (
        committed_clean_source.get("schema")
        != "eggroll-es-committed-clean-source-certificate-v33a"
        or committed_clean_source.get("git_head") != provenance.get("git_head")
        or committed_clean_source.get("all_tracked_files_clean") is not True
        or committed_clean_source.get(
            "only_explicitly_allowlisted_untracked_paths_present"
        ) is not True
    ):
        raise RuntimeError("v33a committed-clean source certificate changed")
    prelaunch_idle = assert_all_four_gpus_idle_v33a()
    attempt = {
        "schema": "eggroll-es-durable-launch-attempt-v33a",
        "status": "launching",
        "phase": "before_trainer_creation",
        "recipe": recipe,
        "source_provenance": provenance,
        "committed_clean_source_certificate": committed_clean_source,
        "runtime_environment_certificate": environment,
        "live_model_audit": live_model,
        "prelaunch_idle_certificate_sha256": prelaunch_idle[
            "content_sha256_before_self_field"
        ],
        "final_idle_certificate_sha256": None,
        "model_update_applied": False,
        "checkpoint_written": False,
        "evaluation_opened": False,
        "dataset_promotion_applied": False,
    }
    runtime_v23a._exclusive_write_json_v23a(attempt_path, attempt)
    if run_dir.exists():
        final_idle = None
        final_idle_failure = None
        try:
            final_idle = wait_for_final_gpu_idle_v33a(prelaunch_idle)
        except BaseException as error:
            final_idle_failure = error
        attempt.update({
            "status": "failed",
            "phase": "fresh_run_reservation_race_after_finalization",
            "failure_type": "FreshRunReservationError",
            "failure_sha256": canonical_sha256(
                "v33a run directory appeared after exclusive attempt claim"
            ),
            "final_idle_certificate_sha256": (
                None
                if final_idle is None
                else final_idle["content_sha256_before_self_field"]
            ),
            "all_four_gpus_idle_after_cleanup": final_idle is not None,
            "final_idle_failure_sha256": (
                None
                if final_idle_failure is None
                else canonical_sha256({
                    "type": type(final_idle_failure).__name__,
                    "repr": repr(final_idle_failure),
                })
            ),
        })
        runtime_v23a._rewrite_json_v23a(attempt_path, attempt)
        if final_idle_failure is not None:
            raise final_idle_failure
        raise RuntimeError("v33a run directory appeared after exclusive attempt claim")
    trainer = None
    failure = None
    final_idle = None
    final_idle_failure = None
    configuration = summary = gate = audit = None
    try:
        runtime_v23a.base.set_seed(43)
        trainer = _make_trainer_v33a(preregistration, layer_bundle)
        configuration = trainer.configure_paired_data_compat_v33a(
            preregistration, panel_bundle, layer_bundle
        )
        summary, gate, audit = trainer.estimate_paired_data_compat_v33a()
        if gate != mechanics_v33a.evaluate_candidate_v33a(summary):
            raise RuntimeError("v33a compact gate recomputation changed")
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
            final_idle = wait_for_final_gpu_idle_v33a(prelaunch_idle)
        except BaseException as idle_error:
            final_idle_failure = idle_error
            if failure is None:
                failure = idle_error
    if failure is not None:
        attempt.update({
            "status": "failed",
            "phase": "after_cleanup_or_final_idle_failure",
            "failure_type": type(failure).__name__,
            "failure_sha256": canonical_sha256({
                "type": type(failure).__name__, "repr": repr(failure),
            }),
            "report_exists_after_attempt": report_path.exists(),
            "final_idle_certificate_sha256": (
                None
                if final_idle is None
                else final_idle["content_sha256_before_self_field"]
            ),
            "all_four_gpus_idle_after_cleanup": final_idle is not None,
            "final_idle_failure_sha256": (
                None
                if final_idle_failure is None
                else canonical_sha256({
                    "type": type(final_idle_failure).__name__,
                    "repr": repr(final_idle_failure),
                })
            ),
            "model_update_applied": False,
            "checkpoint_written": False,
            "evaluation_opened": False,
            "dataset_promotion_applied": False,
        })
        runtime_v23a._rewrite_json_v23a(attempt_path, attempt)
        raise failure
    try:
        report = _seal({
            "schema": "eggroll-es-paired-data-compat-report-v33a",
            "recipe": recipe,
            "configuration": configuration,
            "summary": summary,
            "gate": gate,
            "runtime_audit": audit,
            "implementation": implementation,
            "committed_clean_source_certificate_sha256": committed_clean_source[
                "content_sha256_before_self_field"
            ],
            "prelaunch_idle_certificate_sha256": prelaunch_idle[
                "content_sha256_before_self_field"
            ],
            "final_idle_certificate_sha256": final_idle[
                "content_sha256_before_self_field"
            ],
            "all_four_gpus_idle_after_cleanup": True,
            "model_update_applied": False,
            "checkpoint_written": False,
            "evaluation_opened": False,
            "dataset_promotion_applied": False,
            "direct_action_taken": False,
        })
        _assert_compact(report)
        runtime_v23a._exclusive_write_json_v23a(report_path, report)
        attempt.update({
            "status": "complete",
            "phase": "after_compact_report_and_final_gpu_cleanup",
            "report_binding": {
                "path": str(report_path),
                "file_sha256": file_sha256(report_path),
                "content_sha256": report["content_sha256_before_self_field"],
            },
            "final_idle_certificate_sha256": final_idle[
                "content_sha256_before_self_field"
            ],
            "all_four_gpus_idle_after_cleanup": True,
        })
        runtime_v23a._rewrite_json_v23a(attempt_path, attempt)
        return report
    except BaseException as finalization_error:
        attempt.update({
            "status": "failed",
            "phase": "after_final_idle_during_compact_report_finalization",
            "failure_type": type(finalization_error).__name__,
            "failure_sha256": canonical_sha256({
                "type": type(finalization_error).__name__,
                "repr": repr(finalization_error),
            }),
            "report_exists_after_attempt": report_path.exists(),
            "final_idle_certificate_sha256": final_idle[
                "content_sha256_before_self_field"
            ],
            "all_four_gpus_idle_after_cleanup": True,
            "model_update_applied": False,
            "checkpoint_written": False,
            "evaluation_opened": False,
            "dataset_promotion_applied": False,
        })
        runtime_v23a._rewrite_json_v23a(attempt_path, attempt)
        raise


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    _assert_train_only_argv(argv)
    args = _parser().parse_args(argv)
    preregistration = load_preregistration_v33a()
    panel_bundle = mechanics_v33a.load_paired_panel_bundle_v33a()
    layer_bundle = load_layer_bundle_v33a(preregistration)
    implementation = implementation_identity_v33a()
    recipe = recipe_v33a(
        preregistration, panel_bundle, layer_bundle, implementation
    )
    committed_clean_source = validate_runtime_v33a(
        args, preregistration, implementation, recipe
    )
    if args.v33a_dry_run:
        payload = _seal({
            "schema": "eggroll-es-paired-data-compat-dry-run-v33a",
            "implementation": implementation,
            "recipe": recipe,
            "fresh_exclusive_attempt_required": True,
            "real_launch_requires_exact_post_cherry_pick_implementation_and_recipe_hashes": True,
            "pass_authority_limited_to_separate_fresh_basis_v364_confirmation_preregistration": True,
            "dataset_promotion_model_update_checkpoint_and_evaluation_authorized": False,
            "row_response_score_or_bootstrap_content_persisted": False,
            "gpu_launched": False,
            "train_only_runtime_launched": False,
            "validation_heldout_ood_benchmark_or_evaluation_opened": False,
        })
        _assert_compact(payload)
        print(json.dumps(payload, sort_keys=True))
        return payload
    return run_exact_v33a(
        preregistration, panel_bundle, layer_bundle, implementation, recipe,
        committed_clean_source,
    )


if __name__ == "__main__":
    main()
