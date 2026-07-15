#!/usr/bin/env python3
"""CPU-only fail-closed runner frame for the V34B four-GPU experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import eggroll_es_v401_replacement_fraction_preregistration_v34b as prereg_v34b
import eggroll_es_worker_v33a as worker_v33a
import train_eggroll_es_v401_replacement_fraction_v34b as mechanics_v34b


ROOT = Path(__file__).resolve().parent
EXPERIMENT_NAME = "s6_v34b_production_v401_replacement_fraction_hpo_basis20261015"
OUTPUT_DIRECTORY = (ROOT / "experiments/eggroll_es_hpo/runs").resolve()
ATTEMPT_PATH = OUTPUT_DIRECTORY / f".{EXPERIMENT_NAME}.launch_attempt.json"
RUN_DIRECTORY = OUTPUT_DIRECTORY / EXPERIMENT_NAME
REPORT_PATH = RUN_DIRECTORY / "production_v401_replacement_fraction_v34b.json"
WORKER_EXTENSION = "eggroll_es_worker_v33a.PairedDataCompatWorkerExtensionV33A"

BOUND_FILES = {
    "candidate_v401": (
        mechanics_v34b.frame_v34b.CANDIDATE_PATH,
        mechanics_v34b.frame_v34b.CANDIDATE_SHA256,
    ),
    "candidate_manifest_v401": (
        mechanics_v34b.frame_v34b.CANDIDATE_MANIFEST_PATH,
        mechanics_v34b.frame_v34b.CANDIDATE_MANIFEST_SHA256,
    ),
    "production": (
        mechanics_v34b.frame_v34b.PRODUCTION_PATH,
        mechanics_v34b.frame_v34b.PRODUCTION_SHA256,
    ),
    "frame_builder_v34b": (
        Path(mechanics_v34b.frame_v34b.__file__).resolve(),
        prereg_v34b.FRAME_BUILDER_SHA256,
    ),
    "frame_test_v34b": (
        ROOT / "test_build_eggroll_es_v401_replacement_fraction_frame_v34b.py",
        prereg_v34b.FRAME_TEST_SHA256,
    ),
    "frame_v34b": (prereg_v34b.FRAME_PATH, prereg_v34b.FRAME_FILE_SHA256),
    "preregistration_module_v34b": (
        Path(prereg_v34b.__file__).resolve(),
        mechanics_v34b.PREREGISTRATION_MODULE_SHA256,
    ),
    "preregistration_test_v34b": (
        ROOT / "test_eggroll_es_v401_replacement_fraction_preregistration_v34b.py",
        mechanics_v34b.PREREGISTRATION_TEST_SHA256,
    ),
    "preregistration_v34b": (
        prereg_v34b.OUTPUT_PATH,
        mechanics_v34b.PREREGISTRATION_FILE_SHA256,
    ),
    "mechanics_v34b": (
        Path(mechanics_v34b.__file__).resolve(),
        "5a59d618ba690f354c0564a52a391602b6f7a207a076523eac5ae262ba50c183",
    ),
    "mechanics_test_v34b": (
        ROOT / "test_train_eggroll_es_v401_replacement_fraction_v34b.py",
        "d6d1b87d53b18f9c0e69087c9e02cc2445e1cfb81d9acaf83535b6ec261bb3f9",
    ),
    "runner_test_v34b": (
        ROOT / "test_run_eggroll_es_v401_replacement_fraction_v34b.py",
        "fc3e789b9d1a69f94f8c02e70cbb422ceee503bfb7d740930277703928066e45",
    ),
    "worker_v33a_hash_bound_reuse": (
        Path(worker_v33a.__file__).resolve(),
        "a65c506b8c96db64cc332578384ea9465c845329288591c7abc96e3a6e24e38e",
    ),
    "worker_test_v33a": (
        ROOT / "test_eggroll_es_worker_v33a.py",
        "50e034b43f760dea8c4122a7dd4085ab3d7a3b447f2d1515af2f899547e620f2",
    ),
}
FORBIDDEN_ARGV_TOKENS = (
    "validation", "heldout", "ood", "benchmark", "eval", "checkpoint",
    "update", "promotion", "train-dataset", "save-model",
)
FORBIDDEN_PERSISTED_KEYS = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "prompt_token_ids", "unit_scores", "responses", "coefficients",
    "bootstrap_replicates", "bootstrap_draws", "row_content", "row_index",
    "row_sha256", "document_sha256", "unit_id", "unit_ids", "weights",
    "strata", "pids", "timings", "memory",
}


canonical_sha256 = prereg_v34b.canonical_sha256
file_sha256 = prereg_v34b.file_sha256


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _seal(value):
    value = copy.deepcopy(value)
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


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
        raise RuntimeError(f"v34b compact artifact contains forbidden keys: {sorted(overlap)}")
    return value


def assert_train_only_argv(argv):
    for token in argv:
        lowered = str(token).lower()
        if any(forbidden in lowered for forbidden in FORBIDDEN_ARGV_TOKENS):
            raise ValueError(f"v34b rejects forbidden runtime surface: {token}")


def verify_bound_files():
    identities = {}
    for name, (path, expected) in BOUND_FILES.items():
        path = Path(path).resolve()
        if file_sha256(path) != expected:
            raise RuntimeError(f"v34b immutable input changed: {name}")
        identities[name] = {
            "relative_path": path.relative_to(ROOT).as_posix(),
            "file_sha256": expected,
        }
    return identities


def implementation_identity():
    identities = verify_bound_files()
    runner_path = Path(__file__).resolve()
    value = {
        "schema": "eggroll-es-v401-replacement-fraction-implementation-v34b",
        "bound_files": identities,
        "runner": {
            "relative_path": runner_path.relative_to(ROOT).as_posix(),
            "file_sha256": file_sha256(runner_path),
        },
        "worker_extension": WORKER_EXTENSION,
    }
    value["bundle_sha256"] = canonical_sha256(value)
    return value


def seed_projection_contract(preregistration):
    seeds = preregistration["frozen_recipe"]["perturbation_basis"]["direction_seeds"]
    projections = [{
        "full_seed": seed,
        "numpy_legacy_seed": worker_v33a.worker_r1.project_numpy_legacy_seed_r1(seed),
    } for seed in seeds]
    if (
        len(seeds) != len(set(seeds)) != 0
        or len(seeds) != 64
        or len({item["numpy_legacy_seed"] for item in projections}) != 64
        or any(item["numpy_legacy_seed"] == 0 for item in projections)
    ):
        raise RuntimeError("v34b projected seed coverage changed")
    return {
        "direction_count": 64,
        "direction_seed_list_sha256": canonical_sha256(seeds),
        "full_to_numpy_projection_sha256": canonical_sha256(projections),
        "numpy_projection_unique_count": 64,
    }


def execution_contract(preregistration, panel_bundle, implementation):
    schedule = mechanics_v34b.resident_signed_wave_schedule()
    if (
        len(schedule) != 32
        or any(item["engine_direction_indices"] != list(range(
            item["population_wave_index"] * 4,
            (item["population_wave_index"] + 1) * 4,
        )) for item in schedule)
        or any(len(set(item["engine_direction_seeds"])) != 4 for item in schedule)
        or any(item["all_four_tp1_engines_required"] is not True for item in schedule)
        or any(item["restore_after_both_sources"] is not True for item in schedule)
    ):
        raise RuntimeError("v34b synchronized activity schedule changed")
    value = {
        "schema": "eggroll-es-v401-replacement-fraction-execution-contract-v34b",
        "experiment_name": EXPERIMENT_NAME,
        "preregistration": {
            "file_sha256": mechanics_v34b.PREREGISTRATION_FILE_SHA256,
            "content_sha256": preregistration["content_sha256_before_self_field"],
        },
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "transient_panel_bundle_content_sha256": panel_bundle[
            "content_sha256_before_self_field"
        ],
        "model": preregistration["frozen_recipe"]["model"],
        "model_config_sha256": preregistration["frozen_recipe"]["model_config_sha256"],
        "model_index_sha256": preregistration["frozen_recipe"]["model_index_sha256"],
        "layer_plan": preregistration["frozen_recipe"]["layer_plan"],
        "sigma": 0.0003,
        "alpha": 0.0,
        "population_size": 64,
        "engine_contract": {
            "gpu_ids": [0, 1, 2, 3],
            "engine_count": 4,
            "tensor_parallel_per_engine": 1,
            "one_distinct_gpu_per_engine": True,
            "all_four_engines_active_every_signed_wave": True,
            "partial_wave_allowed": False,
            "worker_extension": WORKER_EXTENSION,
        },
        "seed_projection": seed_projection_contract(preregistration),
        "signed_schedule_sha256": canonical_sha256(schedule),
        "phase_order": [
            "prelaunch_bound_hash_clean_source_exclusive_path_and_all_gpu_idle_gates",
            "launch_exactly_four_tp1_engines_and_install_hash_bound_layer_plan",
            "save_and_verify_exact_reference_on_all_four_engines",
            "full_context_phase_A_then_exact_repeat_B_for_both_sources",
            "thirty_two_synchronized_signed_waves_score_both_sources_then_exact_restore",
            "selected_and_unselected_population_boundary_and_origin_audits",
            "full_context_phase_C_and_exact_A_equals_C_gate",
            "sequential_fraction_bootstrap_stop_at_first_failure",
            "compact_gate_and_report_only",
            "engine_cleanup_and_final_all_gpu_idle_gate",
        ],
        "request_budget": copy.deepcopy(preregistration["hardware_and_budget"]),
        "fraction_contract": copy.deepcopy(preregistration["replacement_fraction_hpo"]),
        "fixed_sequence_gate": copy.deepcopy(preregistration["fixed_sequence_gate"]),
        "persistence": {
            "compact_aggregate_only": True,
            "response_vectors_rows_scores_coefficients_bootstrap_draws_timings_memory_or_pids": False,
            "attempt_path": str(ATTEMPT_PATH),
            "run_directory": str(RUN_DIRECTORY),
            "report_path": str(REPORT_PATH),
            "fresh_O_EXCL_required": True,
        },
        "closed_surfaces": {
            "model_update": True,
            "checkpoint_write": True,
            "dataset_promotion": True,
            "validation_heldout_ood_benchmark_evaluation": True,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def build_cpu_dry_run():
    preregistration = mechanics_v34b.load_hardened_preregistration()
    panel_bundle = mechanics_v34b.materialize_paired_panel_bundle()
    implementation = implementation_identity()
    contract = execution_contract(preregistration, panel_bundle, implementation)
    payload = _seal({
        "schema": "eggroll-es-v401-replacement-fraction-cpu-dry-run-v34b",
        "implementation": implementation,
        "execution_contract": contract,
        "real_launch_requires_committed_clean_hash_identical_bundle": True,
        "real_launch_requires_exact_expected_implementation_and_recipe_hashes": True,
        "this_cpu_bundle_launches_or_reserves_no_gpu": True,
        "GPU_launched": False,
        "train_only_runtime_launched": False,
        "validation_heldout_ood_benchmark_or_evaluation_opened": False,
        "dataset_promotion_model_update_or_checkpoint_authorized": False,
    })
    assert_compact(payload)
    return payload


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v34b-dry-run", action="store_true")
    parser.add_argument("--expected-implementation-bundle-sha256")
    parser.add_argument("--expected-execution-contract-sha256")
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    assert_train_only_argv(argv)
    args = _parser().parse_args(argv)
    if not args.v34b_dry_run:
        raise ValueError(
            "v34b is a CPU-only preregistration runner; GPU execution is intentionally unavailable"
        )
    payload = build_cpu_dry_run()
    for expected, actual, label in (
        (
            args.expected_implementation_bundle_sha256,
            payload["implementation"]["bundle_sha256"],
            "implementation",
        ),
        (
            args.expected_execution_contract_sha256,
            payload["execution_contract"]["content_sha256_before_self_field"],
            "execution contract",
        ),
    ):
        if expected is not None and expected != actual:
            raise ValueError(f"v34b expected {label} hash changed")
    print(json.dumps(payload, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
