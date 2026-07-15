#!/usr/bin/env python3
"""Preregister the frozen train-only v401 replacement-fraction HPO V34B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import build_eggroll_es_v401_replacement_fraction_frame_v34b as frame_v34b


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V401_REPLACEMENT_FRACTION_HPO_V34B_PREREGISTRATION.json"
).resolve()
FRAME_PATH = frame_v34b.OUTPUT_PATH
LAYER_PLAN_PATH = (ROOT / "experiments/layer_plans/v23a_base_middle_late_dense.json").resolve()
MODEL_PATH = (ROOT / "models/Qwen3.6-35B-A3B").resolve()

FRAME_BUILDER_SHA256 = "936efe422f560fd49f4f6bfa775465c786e9b3d2299189c4343f38ae1ede1774"
FRAME_TEST_SHA256 = "b1a8c7098302c844966dc056e191f193a3cfd908f17558e2c591ebdfc6f17ff7"
FRAME_FILE_SHA256 = "832bbea07d08c487621e2dc88dfb8ebffc4b05d888badbbe5eb0fd71124efde3"
FRAME_CONTENT_SHA256 = frame_v34b.EXPECTED_AGGREGATE_CONTENT_SHA256
LAYER_PLAN_FILE_SHA256 = "4dca69212f2eee1c5882a7f3de944e3dadd8de94b42e58e7d7495547e8b1c747"
LAYER_PLAN_CONTENT_SHA256 = "07a155d1217b27ba1bf30e057024247236a812841c52bab401d465c9fdb5273f"
MODEL_CONFIG_SHA256 = "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
MODEL_INDEX_SHA256 = "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"

EXPERIMENT_NAME = "production_v401_replacement_fraction_hpo_v34b_basis20261015"
PERTURBATION_BASIS_SEED = 20261015
BOOTSTRAP_SEED = 20261016
POPULATION_SIZE = 64
BOOTSTRAP_REPETITIONS = 50_000
FAMILYWISE_ALPHA = 0.05
REPLACEMENT_FRACTIONS = (0.05, 0.10, 0.20, 0.40, 1.00)
METRIC_FAMILIES = (
    "optimization_pairwise_cosine",
    "optimization_pairwise_sign_agreement",
    "aggregate_to_optimization_cosine",
    "aggregate_to_optimization_sign_agreement",
    "train_screen_cosine",
    "train_screen_sign_agreement",
)
ENDPOINTS = tuple(
    f"{family}_{summary}"
    for family in METRIC_FAMILIES
    for summary in ("median", "worst")
)
PRIOR_BASIS_CONTENT_SHA256 = {
    "v22a": "f68624388ac0549ac82ba3d1e64a317233c42f900502a6f5c6d6f07071b4c60e",
    "v30a": "fb9d939f9fc3444694c74cd6805f24a54a936a532b02cf73e4711abc885145b5",
    "v33a": "75a9350f002f462dfb2e7413d30eef4bdc322f558535cf8a871258081467a3e6",
}
PRIOR_DIRECTION_LIST_SHA256 = {
    "v22a": "9faecdc81492052a6c466b0e986df9e31be0c0fccf24687a96ed604f2ef0f553",
    "v30a": "29d165336769bbd89ae3eebf56c8d74f5b4fe603b226506eed1c632dd630b7af",
    "v33a": "4227e7c741175eb29f10c73b70f40e4442ebc6f2ca3d9f798dc7639cfe5a8e5f",
}


canonical_sha256 = frame_v34b.canonical_sha256
file_sha256 = frame_v34b.file_sha256


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def perturbation_seeds() -> list[int]:
    return [
        int(value)
        for value in np.random.default_rng(PERTURBATION_BASIS_SEED).integers(
            0, 2**30, size=POPULATION_SIZE, dtype=np.int64
        )
    ]


def signed_population_schedule() -> list[dict]:
    seeds = perturbation_seeds()
    result = []
    for wave in range(POPULATION_SIZE // 4):
        wave_seeds = seeds[wave * 4:(wave + 1) * 4]
        for sign, negate in (("plus", False), ("minus", True)):
            signed_index = len(result)
            result.append({
                "signed_wave_index": signed_index,
                "population_wave_index": wave,
                "sign": sign,
                "negate": negate,
                "engine_direction_indices": list(range(wave * 4, (wave + 1) * 4)),
                "engine_direction_seeds": wave_seeds,
                "resident_source_order": (
                    ["production", "candidate_v401"]
                    if signed_index % 2 == 0
                    else ["candidate_v401", "production"]
                ),
                "all_four_tp1_engines_required": True,
                "restore_after_both_sources": True,
            })
    return result


def bootstrap_draw_plan_sha256() -> str:
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    digest = hashlib.sha256()
    for panel in frame_v34b.PANEL_NAMES:
        for stratum in frame_v34b.sampler_v13.STRATA:
            count = frame_v34b.STRATUM_QUOTAS[stratum]
            draw = generator.integers(
                0, count, size=(BOOTSTRAP_REPETITIONS, count), dtype=np.int64
            )
            header = json.dumps({
                "panel": panel,
                "stratum": stratum,
                "shape": list(draw.shape),
                "dtype": "int64",
            }, sort_keys=True, separators=(",", ":")).encode()
            digest.update(len(header).to_bytes(8, "little"))
            digest.update(header)
            digest.update(draw.tobytes(order="C"))
    return digest.hexdigest()


def load_bound_inputs() -> tuple[dict, dict]:
    expected = {
        Path(frame_v34b.__file__).resolve(): FRAME_BUILDER_SHA256,
        ROOT / "test_build_eggroll_es_v401_replacement_fraction_frame_v34b.py": FRAME_TEST_SHA256,
        FRAME_PATH: FRAME_FILE_SHA256,
        LAYER_PLAN_PATH: LAYER_PLAN_FILE_SHA256,
        MODEL_PATH / "config.json": MODEL_CONFIG_SHA256,
        MODEL_PATH / "model.safetensors.index.json": MODEL_INDEX_SHA256,
    }
    if any(file_sha256(path) != digest for path, digest in expected.items()):
        raise RuntimeError("v34b bound frame, recipe, or model identity changed")
    frame = json.loads(FRAME_PATH.read_text(encoding="utf-8"))
    frame_v34b.validate_manifest(frame)
    layer = json.loads(LAYER_PLAN_PATH.read_text(encoding="utf-8"))
    if (
        frame["content_sha256_before_self_field"] != FRAME_CONTENT_SHA256
        or layer.get("plan_sha256") != LAYER_PLAN_CONTENT_SHA256
        or layer.get("layers") != [20, 21, 22, 23]
    ):
        raise RuntimeError("v34b bound frame or layer-plan semantics changed")
    return frame, layer


def build_preregistration() -> dict:
    frame, layer = load_bound_inputs()
    seeds = perturbation_seeds()
    schedule = signed_population_schedule()
    basis = {
        "schema": "eggroll-es-v34b-fresh-64-direction-basis",
        "basis_seed": PERTURBATION_BASIS_SEED,
        "generator": "numpy.default_rng(seed).integers(0,2**30,size=64,dtype=int64)",
        "population_size": POPULATION_SIZE,
        "direction_seeds": seeds,
        "direction_seed_list_sha256": canonical_sha256(seeds),
        "signed_population_schedule": schedule,
        "signed_population_schedule_sha256": canonical_sha256(schedule),
    }
    basis["basis_content_sha256"] = canonical_sha256(basis)
    if (
        len(set(seeds)) != POPULATION_SIZE
        or basis["basis_content_sha256"] in set(PRIOR_BASIS_CONTENT_SHA256.values())
        or basis["direction_seed_list_sha256"] in set(PRIOR_DIRECTION_LIST_SHA256.values())
    ):
        raise RuntimeError("v34b perturbation basis is not fresh")
    value = {
        "schema": "eggroll-es-v401-replacement-fraction-preregistration-v34b",
        "status": "preregistered_cpu_only_gpu_runtime_not_launched",
        "experiment_name": EXPERIMENT_NAME,
        "scientific_objective": (
            "select_the_largest_predeclared_v401_replacement_fraction_whose_"
            "train_only_ES_estimator_stability_is_noninferior_to_production"
        ),
        "strict_train_only_firewall": {
            "files_programmatically_opened": "only_the_two_hash_bound_train_sources",
            "validation_heldout_ood_eval_or_benchmark_opened": False,
            "dataset_rows_questions_answers_or_document_text_persisted": False,
            "model_update_or_checkpoint_write": False,
            "dataset_promotion": False,
        },
        "inputs": {
            "production": frame["inputs"]["production"],
            "candidate_v401": frame["inputs"]["candidate_v401"],
            "joint_frame": {
                "path": str(FRAME_PATH),
                "builder_file_sha256": FRAME_BUILDER_SHA256,
                "builder_test_file_sha256": FRAME_TEST_SHA256,
                "file_sha256": FRAME_FILE_SHA256,
                "content_sha256": FRAME_CONTENT_SHA256,
                "runtime_content_sha256": frame["runtime_frame_content_sha256"],
                "paired_units": frame["joint_frame"]["paired_unit_count"],
                "selected_units": frame["joint_frame"]["selected_paired_unit_count"],
                "reserve_units": frame["joint_frame"]["reserve_paired_unit_count"],
            },
        },
        "frozen_recipe": {
            "model": str(MODEL_PATH),
            "model_config_sha256": MODEL_CONFIG_SHA256,
            "model_index_sha256": MODEL_INDEX_SHA256,
            "layer_plan": {
                "path": str(LAYER_PLAN_PATH),
                "file_sha256": LAYER_PLAN_FILE_SHA256,
                "content_sha256": LAYER_PLAN_CONTENT_SHA256,
                "layers": layer["layers"],
            },
            "sigma": 0.0003,
            "alpha": 0.0,
            "model_update_allowed": False,
            "same_resident_perturbation_scores_both_sources": True,
            "perturbation_basis": basis,
        },
        "replacement_fraction_hpo": {
            "fractions_in_fixed_test_order": list(REPLACEMENT_FRACTIONS),
            "control_fraction": 0.0,
            "source_requests": ["production", "candidate_v401"],
            "fraction_model_requests": 0,
            "algebra": (
                "central_panel_response_fraction=(1-f)*central_panel_response_"
                "production+f*central_panel_response_v401"
            ),
            "algebra_applied_before_standardization_and_endpoint_computation": True,
            "same_paired_bootstrap_draws_for_control_sources_and_all_fractions": True,
            "no_fraction_specific_sampling_perturbations_or_model_requests": True,
        },
        "panels": {
            "names": list(frame_v34b.PANEL_NAMES),
            "optimization": list(frame_v34b.OPTIMIZATION_PANELS),
            "train_only_screens": list(frame_v34b.TRAIN_SCREENS),
            "panel_size_per_source_per_direction": frame_v34b.PANEL_SIZE,
            "stratum_quotas": frame_v34b.STRATUM_QUOTAS,
            "globally_disjoint_joint_units": True,
            "horvitz_thompson_weights": True,
        },
        "hardware_and_budget": {
            "gpu_ids": [0, 1, 2, 3],
            "engine_count": 4,
            "tp_per_engine": 1,
            "all_four_engines_required_every_signed_wave": True,
            "population_waves": 16,
            "synchronized_signed_waves": 32,
            "requests_per_source_per_engine_signed_wave": 195,
            "requests_per_engine_signed_wave": 390,
            "requests_all_engines_signed_wave": 1560,
            "perturbed_requests": 49_920,
            "full_context_probe_phases": 3,
            "full_context_requests": 4_680,
            "fraction_specific_requests": 0,
            "total_generation_requests": 54_600,
            "partial_or_unsynchronized_waves_allowed": False,
        },
        "analysis": {
            "metric_families": list(METRIC_FAMILIES),
            "endpoints": list(ENDPOINTS),
            "endpoint_count_per_fraction": len(ENDPOINTS),
            "noninferiority_margin": 0.0,
            "bootstrap": {
                "seed": BOOTSTRAP_SEED,
                "repetitions": BOOTSTRAP_REPETITIONS,
                "resampling_unit": "paired_joint_unit_within_panel_stratum",
                "same_draw_indices_for_both_sources_and_every_fraction": True,
                "draw_plan_sha256": bootstrap_draw_plan_sha256(),
                "quantile_method": "linear",
                "one_sided_family_alpha": FAMILYWISE_ALPHA,
                "within_fraction_bonferroni_quantile": FAMILYWISE_ALPHA / len(ENDPOINTS),
                "raw_draws_or_replicates_persisted": False,
            },
            "multiplicity_control": {
                "within_fraction": "Bonferroni_over_all_12_conjunctive_endpoints",
                "across_fractions": "predeclared_fixed_sequence_gatekeeping_smallest_to_largest",
                "alpha_recycling": (
                    "the_same_0.05_family_alpha_is_used_at_the_next_fraction_only_"
                    "after_all_12_nulls_at_every_prior_fraction_are_rejected"
                ),
                "strong_familywise_rationale": (
                    "the_first_family_containing_any_true_null_is_reached_only_if_"
                    "all_prior_families_pass_and_Bonferroni_bounds_the_probability_"
                    "of_erroneously_passing_that_family_by_0.05"
                ),
                "posthoc_fraction_endpoint_margin_or_alpha_change_allowed": False,
            },
        },
        "fixed_sequence_gate": {
            "test_order": list(REPLACEMENT_FRACTIONS),
            "per_fraction_pass": (
                "all_12_observed_fraction_minus_production_point_deltas_gte_0_"
                "AND_all_12_one_sided_Bonferroni_familywise_LCBs_gte_0"
            ),
            "stop_at_first_failure": True,
            "fractions_after_first_failure_are_not_authorized_or_interpreted": True,
            "selection": "largest_consecutively_passing_fraction_or_0.0_if_0.05_fails",
            "pass_authority": (
                "authorize_only_a_separately_frozen_train_only_HPO_or_training_recipe_"
                "using_the_selected_fraction"
            ),
            "direct_dataset_promotion_model_update_checkpoint_write_or_eval_authorized": False,
        },
        "runtime_integrity": {
            "fresh_O_EXCL_attempt_and_run_paths_required": True,
            "committed_clean_source_bundle_required": True,
            "prelaunch_and_final_all_four_GPU_idle_certificates_required": True,
            "all_four_engines_active_every_signed_wave_required": True,
            "exact_reference_restore_after_every_signed_wave_required": True,
            "selected_and_unselected_population_boundary_audits_required": True,
            "matched_full_context_A_B_before_and_A_C_after_required": True,
            "failure_cleanup_and_final_idle_required": True,
            "bound_file_and_preregistration_hash_recheck_before_output_required": True,
            "compact_aggregate_persistence_only": True,
            "persist_rows_prompts_answers_scores_vectors_coefficients_draws_timings_memory_or_pids": False,
        },
        "required_runtime_adapter": {
            "module": "run_eggroll_es_v401_replacement_fraction_v34b.py",
            "must_bind_this_file_and_content_hash": True,
            "runtime_launch_authorized_by_preregistration_alone": False,
            "gpu_launched_during_preregistration": False,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def exclusive_write(path: Path, value: dict) -> None:
    path = Path(path).resolve()
    if path != OUTPUT_PATH:
        raise ValueError("v34b preregistration output path changed")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise ValueError("v34b preregistration already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    value = build_preregistration()
    if not args.dry_run:
        exclusive_write(args.output, value)
    print(json.dumps({
        "schema": "eggroll-es-v401-replacement-fraction-preregistration-build-v34b",
        "content_sha256": value["content_sha256_before_self_field"],
        "direction_seed_list_sha256": value["frozen_recipe"]["perturbation_basis"][
            "direction_seed_list_sha256"
        ],
        "bootstrap_draw_plan_sha256": value["analysis"]["bootstrap"]["draw_plan_sha256"],
        "total_generation_requests": value["hardware_and_budget"]["total_generation_requests"],
        "gpu_launched": False,
    }, indent=2, sort_keys=True))
    return value


if __name__ == "__main__":
    main()
