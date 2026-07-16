#!/usr/bin/env python3
"""CPU-only fifth-halfspace reprojection of frozen V48B signed scores."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import eggroll_es_multi_anchor_v43h as multi_anchor
import eggroll_es_worker_v3 as worker_v3
import lora_es_generation_boundary_backtracking_v48d as v48d
import run_lora_es_generation_boundary_v48b as v48b


TARGET_NORM_RATIOS_V48E = (
    0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625,
)
EXPECTED_FILES_V48E = v48d.EXPECTED_FILES_V48D
EXPECTED_SUBSET_FILE_SHA256_V48E = v48d.EXPECTED_SUBSET_FILE_SHA256_V48D
EXPECTED_SUBSET_CONTENT_SHA256_V48E = (
    v48d.EXPECTED_SUBSET_CONTENT_SHA256_V48D
)
REQUEST_ORDER_SHA256_V48E = v48d.REQUEST_ORDER_SHA256_V48D
RESTORED_MASTER_SHA256_V48E = v48d.RESTORED_MASTER_SHA256_V48D
RESTORED_RUNTIME_SHA256_V48E = v48d.RESTORED_RUNTIME_SHA256_V48D
SOURCE_TRUST_CAP_V48E = 0.5
EXPECTED_FULL_SIGNED_SCORE_SHA256_V48E = (
    "38b8088cf6048675be10989a9be620c891bf7b63868c252c76965ef23d553dbb"
)
EXPECTED_EXTRACTED_FIVE_OBJECTIVE_SHA256_V48E = (
    "a8012eac29a85f260aaa33d19e07d0e9bef00d73bbb4e807059c628b011fe730"
)
EXPECTED_QA_GENERATION_SIGN_SCORES_SHA256_V48E = (
    "b0484437a3f867924f4706091bd2aa952f39db6b576e24fda99f82d192f0b564"
)
EXPECTED_STORED_FOUR_OBJECTIVE_SHA256_V48E = (
    "885b8fc2d2abeb20d0b2b51a158105ff536e202dfdf76d963e7ee9f28e3798fd"
)
EXPECTED_REPROJECTION_CONTENT_SHA256_V48E = (
    "093c0b3c88588783717ae651898bd327443d6b1258b2860ca9191c0e4c627617"
)
OBJECTIVE_PATHS_V48E = {
    "domain": ("domain", "aggregate", "equal_unit_mean"),
    "fragile_generation_f1": (
        "fragile_generation", "equal_conflict_unit_mean_f1",
    ),
    "prose_lm": ("prose_lm", "mean_token_logprob"),
    "qa_answer_logprob": (
        "qa_answer_logprob", "mean_example_logprob",
    ),
    "qa_generation_f1": ("qa_generation", "mean_f1"),
}


def _nested(value: dict, path: tuple[str, ...]) -> float:
    for key in path:
        value = value[key]
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("v48e signed score is non-finite")
    return result


def extract_five_objective_sign_scores_v48e(population: dict) -> dict:
    signed = population.get("signed_scores", {})
    if v48b.v43i.v40a.canonical_sha256(signed) != (
        EXPECTED_FULL_SIGNED_SCORE_SHA256_V48E
    ):
        raise RuntimeError("v48e full V48B signed-score inventory changed")
    result = {
        objective: {
            sign: [[
                _nested(signed[sign][direction][actor], path)
                for actor in range(4)
            ] for direction in range(8)]
            for sign in ("plus", "minus")
        }
        for objective, path in OBJECTIVE_PATHS_V48E.items()
    }
    if (
        v48b.v43i.v40a.canonical_sha256(result)
        != EXPECTED_EXTRACTED_FIVE_OBJECTIVE_SHA256_V48E
        or v48b.v43i.v40a.canonical_sha256(result["qa_generation_f1"])
        != EXPECTED_QA_GENERATION_SIGN_SCORES_SHA256_V48E
    ):
        raise RuntimeError("v48e extracted signed-score inventory changed")
    stored = population.get("objective_sign_scores", {})
    if (
        v48b.v43i.v40a.canonical_sha256(stored)
        != EXPECTED_STORED_FOUR_OBJECTIVE_SHA256_V48E
        or {key: result[key] for key in stored} != stored
    ):
        raise RuntimeError("v48e extraction differs from V48B four objectives")
    return result


def reproject_five_halfspaces_v48e(population: dict) -> dict:
    sign_scores = extract_five_objective_sign_scores_v48e(population)
    objectives = {
        name: multi_anchor.objective_coefficients_v43h(scores)
        for name, scores in sign_scores.items()
    }
    stored = population.get("objective_fitness", {})
    if (
        {key: objectives[key] for key in stored} != stored
        or objectives["qa_generation_f1"].get("zero_spread") is not False
    ):
        raise RuntimeError("v48e centered-rank objective derivation changed")
    domain = np.asarray(
        objectives["domain"]["coefficients"], dtype=np.float64,
    )
    fragile = np.asarray(
        objectives["fragile_generation_f1"]["coefficients"],
        dtype=np.float64,
    )
    primary = 0.5 * (
        domain / np.linalg.norm(domain)
        + fragile / np.linalg.norm(fragile)
    )
    anchors = {
        name: objective["coefficients"]
        for name, objective in objectives.items()
    }
    projection = multi_anchor.project_multi_anchor_trust_region_v43h(
        primary.tolist(), anchors,
        max_norm_ratio=multi_anchor.TRUST_REGION_NORM_RATIO_V43H,
    )
    diagnostics = projection["diagnostics"]
    if (
        projection.get("content_sha256")
        != EXPECTED_REPROJECTION_CONTENT_SHA256_V48E
        or diagnostics.get("decision") != "project_and_trust_region"
        or diagnostics.get("anchor_order") != list(OBJECTIVE_PATHS_V48E)
        or diagnostics.get("all_anchor_halfspaces_satisfied") is not True
        or diagnostics.get("anchor_directional_derivatives_after", {}).get(
            "qa_generation_f1", -1.0
        ) < -multi_anchor.FEASIBILITY_TOLERANCE_V43H
    ):
        raise RuntimeError("v48e fifth-halfspace reprojection changed")
    old = np.asarray(population["coefficients"], dtype=np.float64)
    new = np.asarray(projection["coefficients"], dtype=np.float64)
    qa = np.asarray(
        objectives["qa_generation_f1"]["coefficients"], dtype=np.float64,
    )
    cosine = float(np.dot(old, new) / (
        np.linalg.norm(old) * np.linalg.norm(new)
    ))
    geometry = {
        "v48b_projection_content_sha256": population[
            "projection"
        ]["content_sha256"],
        "v48e_projection_content_sha256": projection["content_sha256"],
        "coefficient_cosine_v48b_to_v48e": cosine,
        "coefficient_l2_delta": float(np.linalg.norm(new - old)),
        "v48b_qa_generation_f1_directional_derivative": float(np.dot(old, qa)),
        "v48e_qa_generation_f1_directional_derivative": float(np.dot(new, qa)),
        "v48b_active_anchor_names": population[
            "projection"
        ]["diagnostics"]["active_anchor_names"],
        "v48e_active_anchor_names": diagnostics["active_anchor_names"],
        "same_primary_domain_plus_fragile_construction": True,
        "only_added_halfspace": "qa_generation_f1",
    }
    if (
        geometry["v48b_qa_generation_f1_directional_derivative"] >= 0.0
        or geometry["v48e_qa_generation_f1_directional_derivative"]
        < -multi_anchor.FEASIBILITY_TOLERANCE_V43H
        or not 0.0 < cosine < 1.0
    ):
        raise RuntimeError("v48e geometry comparison changed")
    result = {
        "schema": "frozen-v48b-five-halfspace-reprojection-v48e",
        "source_signed_score_inventory_sha256": (
            EXPECTED_FULL_SIGNED_SCORE_SHA256_V48E
        ),
        "extracted_five_objective_sha256": (
            EXPECTED_EXTRACTED_FIVE_OBJECTIVE_SHA256_V48E
        ),
        "qa_generation_sign_scores_sha256": (
            EXPECTED_QA_GENERATION_SIGN_SCORES_SHA256_V48E
        ),
        "objective_fitness": objectives,
        "primary_coefficients": primary.tolist(),
        "projection": projection,
        "coefficients": projection["coefficients"],
        "geometry_vs_v48b": geometry,
        "population_resampled": False,
        "population_scores_recomputed": False,
        "projection_recomputed_from_frozen_signed_scores": True,
        "new_generation_performed": False,
        "protected_semantics_opened": False,
    }
    result["content_sha256"] = v48b.v43i.v40a.canonical_sha256(result)
    return result


def scale_plans_v48e(reprojection: dict) -> list[dict]:
    diagnostics = reprojection["projection"]["diagnostics"]
    source = [float(value) for value in reprojection["coefficients"]]
    unconstrained = float(diagnostics["unconstrained_domain_norm"])
    source_ratio = math.sqrt(math.fsum(x * x for x in source)) / unconstrained
    if not math.isclose(
        source_ratio, SOURCE_TRUST_CAP_V48E, rel_tol=0.0, abs_tol=1e-12,
    ):
        raise RuntimeError("v48e source reprojection is not at trust cap")
    plans = []
    for target in TARGET_NORM_RATIOS_V48E:
        multiplier = target / source_ratio
        coefficients = [value * multiplier for value in source]
        actual = math.sqrt(math.fsum(x * x for x in coefficients)) / unconstrained
        if not math.isclose(actual, target, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("v48e scaled norm changed")
        plans.append({
            "target_norm_ratio": target,
            "v48e_coefficient_multiplier": multiplier,
            "coefficients": coefficients,
            "coefficient_sha256": worker_v3.coefficient_sha256_v3(
                v48b.v43i.SEEDS, coefficients,
            ),
            "actual_norm_ratio": actual,
            "five_halfspaces_preserved_by_positive_scaling": True,
        })
    return plans


def largest_passing_scale_v48e(results: list[dict]) -> float | None:
    if not isinstance(results, list) or len(results) > len(TARGET_NORM_RATIOS_V48E):
        raise ValueError("v48e scale result coverage changed")
    for index, result in enumerate(results):
        target = TARGET_NORM_RATIOS_V48E[index]
        if result.get("target_norm_ratio") != target:
            raise ValueError("v48e scale evaluation order changed")
        if result.get("gate_passed") is True:
            if index != len(results) - 1:
                raise ValueError("v48e evaluated a smaller scale after a pass")
            return target
        if result.get("exact_abort_readback_passed") is not True:
            raise ValueError("v48e rejected scale was not exactly aborted")
    return None


def load_v48e_design(
    evidence_paths: dict[str, Path], subset: Path, preregistration: Path,
) -> dict:
    evidence = v48d.load_v48b_evidence_v48d(
        evidence_paths, subset, preregistration,
    )
    population_artifact = v48d._sealed(
        evidence_paths["population_reliability"],
        v48d.EXPECTED_FILES_V48D["population_reliability"],
    )
    reprojection = reproject_five_halfspaces_v48e(
        population_artifact["population"],
    )
    result = {
        "schema": "sealed-v48e-cpu-reprojection-design",
        "source_v48b_evidence": evidence,
        "reprojection": reprojection,
        "scale_plans": scale_plans_v48e(reprojection),
        "restored_master_identity": evidence["restored_master_identity"],
        "restored_runtime_values_sha256": evidence[
            "restored_runtime_values_sha256"
        ],
        "anchor_calibrated_margins": evidence["anchor_calibrated_margins"],
        "numeric_calibration_bootstrap_bounds": evidence[
            "numeric_calibration_bootstrap_bounds"
        ],
        "sealed_subset": evidence["sealed_subset"],
        "v48b_population_content_sha256": evidence[
            "v48b_population_content_sha256"
        ],
        "v48b_projection_content_sha256": evidence[
            "v48b_projection_content_sha256"
        ],
        "v48e_projection_content_sha256": reprojection[
            "projection"
        ]["content_sha256"],
        "population_resampled": False,
        "population_scores_recomputed": False,
        "projection_recomputed_from_frozen_signed_scores": True,
        "new_generation_performed": False,
        "protected_semantics_opened": False,
        "shadow_ood_holdout_or_benchmark_opened": False,
    }
    result["content_sha256"] = v48b.v43i.v40a.canonical_sha256(result)
    return result
