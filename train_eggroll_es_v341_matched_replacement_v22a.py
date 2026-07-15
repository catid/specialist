#!/usr/bin/env python3
"""Pure train-only mechanics for V22A exact-v341 matched replacement."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

import build_eggroll_es_v341_matched_replacement_draw_plan_v22a as draw_v22a
import build_eggroll_es_v341_matched_replacement_frame_v22a as frame_v22a
import eggroll_es_v341_matched_replacement_preregistration_v22a as prereg_v22a


ROOT = Path(__file__).resolve().parent
ARMS_V22A = frame_v22a.ARM_ORDER_V22A
PANEL_NAMES_V22A = frame_v22a.PANEL_NAMES_V22A
OPTIMIZATION_PANELS_V22A = frame_v22a.OPTIMIZATION_PANELS_V22A
TRAIN_SCREEN_PANELS_V22A = frame_v22a.TRAIN_SCREEN_PANELS_V22A
BASE_CATEGORIES_V22A = frame_v22a.BASE_CATEGORIES_V22A
ENDPOINT_NAMES_V22A = prereg_v22a.ENDPOINT_NAMES_V22A
SIGNS_V22A = ("plus", "minus")

DRAW_PLAN_COMMIT_V22A = "0614595174cf8ea73b06d259a7b87e83c183c62b"
DRAW_BUILDER_SHA256_V22A = (
    "c4f3fd7871d908f9a75a1b33589aa0f41e6e810c7e161dc0a2db3e285503d0aa"
)
DRAW_TEST_SHA256_V22A = (
    "cb1e6607fded05b6df4a68c3c1e23875c453108843b48bae76ef58ade618aaee"
)
DRAW_FILE_SHA256_V22A = (
    "45d7f525b3ed1307c9b18f25c7a7d08a79da054b31e28be6913fab27be98bca2"
)
DRAW_CONTENT_SHA256_V22A = (
    "bb7fb2d5ca147142c0a8406fbe929944d964452deaeb2378f7a7286988fb7b2e"
)
PANEL_BUNDLE_CONTENT_SHA256_V22A = (
    "bda020933ff7b6abf3c9dd21e79e743d66bdf72f1ca23dbd31b5a96f9f571b0e"
)
BOOTSTRAP_REPETITIONS_V22A = 50_000
BOOTSTRAP_DEFAULT_CHUNK_SIZE_V22A = 128
BOOTSTRAP_QUANTILE_METHOD_V22A = "linear"
STANDARDIZATION_EPSILON_V22A = 1e-8
PANEL_ORDER_SEED_V22A = 20260825
RUNTIME_INTEGRITY_KEYS_V22A = {
    "all_four_tp1_engines_every_signed_wave",
    "all_ten_panels_every_direction_sign_and_arm",
    "all_thirty_two_signed_waves_complete",
    "counterbalanced_arm_order_complete",
    "same_resident_perturbation_both_arms",
    "exact_reference_restored_once_per_signed_wave",
    "pre_post_raw_reference_probes_equal",
    "population_boundary_audit_passed",
    "unselected_origin_audit_passed",
    "union_planner_called",
    "all_integrity_audits_passed",
}

canonical_sha256 = prereg_v22a.canonical_sha256
file_sha256 = prereg_v22a.file_sha256


def _without_self(value: dict) -> dict:
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _verify_commit_file(path: Path, digest: str) -> None:
    path = Path(path).resolve()
    relative = path.relative_to(ROOT).as_posix()
    raw = subprocess.check_output(
        ["git", "show", f"{DRAW_PLAN_COMMIT_V22A}:{relative}"], cwd=ROOT
    )
    if hashlib.sha256(raw).hexdigest() != digest or file_sha256(path) != digest:
        raise RuntimeError(f"v22a mechanics draw artifact changed: {relative}")


def load_hardened_foundation_v22a() -> tuple[dict, dict, dict]:
    for path, digest in (
        (Path(draw_v22a.__file__), DRAW_BUILDER_SHA256_V22A),
        (
            ROOT / "test_build_eggroll_es_v341_matched_replacement_draw_plan_v22a.py",
            DRAW_TEST_SHA256_V22A,
        ),
        (draw_v22a.OUTPUT_PATH_V22A, DRAW_FILE_SHA256_V22A),
    ):
        _verify_commit_file(path, digest)
    frame, _evidence, _layer = prereg_v22a.load_bound_inputs_v22a()
    preregistration = json.loads(prereg_v22a.OUTPUT_PATH_V22A.read_text())
    prereg_v22a.validate_preregistration_v22a(preregistration)
    draw = json.loads(draw_v22a.OUTPUT_PATH_V22A.read_text())
    draw_v22a.validate_draw_plan_certificate_v22a(draw)
    if (
        frame["content_sha256_before_self_field"]
        != prereg_v22a.FRAME_CERTIFICATE_CONTENT_SHA256_V22A
        or preregistration["content_sha256_before_self_field"]
        != draw_v22a.PREREG_CONTENT_SHA256_V22A
        or draw["content_sha256_before_self_field"]
        != DRAW_CONTENT_SHA256_V22A
    ):
        raise RuntimeError("v22a hardened mechanics foundation changed")
    return frame, preregistration, draw


def _panel_order_v22a(panel: str, component_indices: list[int], components):
    return sorted(
        component_indices,
        key=lambda index: canonical_sha256({
            "schema": "eggroll-es-runtime-panel-order-v22a",
            "seed": PANEL_ORDER_SEED_V22A,
            "panel": panel,
            "joint_id": components[index]["joint_id"],
        }),
    )


def _materialize_panel_bundle_v22a() -> dict:
    """Automated train-only row path; returned bundle must never be persisted."""
    persisted_frame, preregistration, _draw = load_hardened_foundation_v22a()
    candidate = frame_v22a._read_jsonl_internal_only(
        frame_v22a.candidate_v341.OUTPUT_PATH_V341
    )
    production = frame_v22a._read_jsonl_internal_only(
        frame_v22a.frame_v21a.frame_v18a.PRODUCTION_PATH_V18A
    )
    frame = frame_v22a._build_joint_frame_internal_only(candidate, production)
    frozen_frame, frozen_tiers, base_flow = frame_v22a.frame_v21a._base_flow_v21a()

    components = frame["joint_components"]
    candidate_units = frame["side_units"]["candidate"]
    production_units = frame["side_units"]["production"]
    frozen_components = frozen_frame["joint_components"]
    frozen_production_units = frozen_frame["side_units"]["production"]
    by_production_unit_id = {
        production_units[item["production_unit"]]["unit_id"]: index
        for index, item in enumerate(components)
        if item["production_unit"] is not None
    }
    side_rows = {"candidate": candidate, "production": production}
    certificate_panels = {
        panel["name"]: panel for panel in persisted_frame["panels"]
    }
    panels = {}
    total_replacements = 0
    total_unchanged = 0
    for panel in PANEL_NAMES_V22A:
        frozen_indices = base_flow["selected_base"][panel]
        category_by_new_index = {}
        selected_indices = []
        for frozen_index in frozen_indices:
            production_index = frozen_components[frozen_index]["production_unit"]
            unit_id = frozen_production_units[production_index]["unit_id"]
            new_index = by_production_unit_id[unit_id]
            selected_indices.append(new_index)
            category_by_new_index[new_index] = frozen_tiers["base_category"][
                frozen_index
            ]
        selected_indices = _panel_order_v22a(
            panel, selected_indices, components
        )
        arms = {}
        for arm in ARMS_V22A:
            rows = []
            row_sha256 = []
            joint_ids = []
            ht_strata = []
            weights = []
            representative_sides = []
            for component_index in selected_indices:
                component = components[component_index]
                side = "production"
                unit = production_units[component["production_unit"]]
                if (
                    arm == "v341_matched_replacement"
                    and component["relation"] == "paired"
                ):
                    side = "candidate"
                    unit = candidate_units[component["candidate_unit"]]
                row = side_rows[side][unit["representative_index"]]
                digest = frame_v22a.frame_v21a.frame_v18a.sampler_v13.row_sha256(
                    row
                )
                if (
                    digest != unit["representative_row_sha256"]
                    or not isinstance(row.get("question"), str)
                    or not row["question"]
                    or not isinstance(row.get("answer"), str)
                    or not row["answer"]
                ):
                    raise RuntimeError("v22a fixed representative content changed")
                stratum = category_by_new_index[component_index]
                weight = (
                    frame_v22a.BASE_CATEGORY_POPULATIONS_V22A[stratum]
                    / frame_v22a.BASE_CATEGORY_QUOTA_V22A
                )
                rows.append(row)
                row_sha256.append(digest)
                joint_ids.append(component["joint_id"])
                ht_strata.append(stratum)
                weights.append(float(weight))
                representative_sides.append(side)
            certificate_arm = certificate_panels[panel]["arms"][arm]
            representative_root = frame_v22a.identity_root_sha256(
                f"{joint_id}:{side}:{row_digest}"
                for joint_id, side, row_digest in zip(
                    joint_ids, representative_sides, row_sha256
                )
            )
            if (
                len(rows) != 24
                or len(joint_ids) != len(set(joint_ids))
                or not math.isclose(
                    math.fsum(weights), 272.0, rel_tol=0.0, abs_tol=1e-12
                )
                or frame_v22a.identity_root_sha256(joint_ids)
                != certificate_arm["active_component_identity_root_sha256"]
                or representative_root
                != certificate_arm["representative_assignment_root_sha256"]
                or representative_sides.count("candidate")
                != certificate_arm["candidate_representatives"]
                or representative_sides.count("production")
                != certificate_arm["production_representatives"]
            ):
                raise RuntimeError("v22a materialized arm estimand changed")
            arms[arm] = {
                "ordered_joint_identity_sha256": canonical_sha256(joint_ids),
                "ordered_row_identity_sha256": canonical_sha256(row_sha256),
                "joint_ids": joint_ids,
                "row_sha256": row_sha256,
                "ht_strata": ht_strata,
                "weights": weights,
                "representative_sides": representative_sides,
                "questions": [row["question"] for row in rows],
                "answers": [row["answer"] for row in rows],
                "raw_prompt_answer_sha256": canonical_sha256({
                    "questions": [row["question"] for row in rows],
                    "answers": [row["answer"] for row in rows],
                }),
            }
        control = arms["production_control"]
        treatment = arms["v341_matched_replacement"]
        expected_replacements = frame_v22a.EXPECTED_REPLACEMENTS_BY_PANEL_V22A[
            panel
        ]
        if (
            treatment["joint_ids"] != control["joint_ids"]
            or treatment["ht_strata"] != control["ht_strata"]
            or treatment["weights"] != control["weights"]
            or control["representative_sides"] != ["production"] * 24
            or treatment["representative_sides"].count("candidate")
            != expected_replacements
            or treatment["representative_sides"].count("production")
            != 24 - expected_replacements
        ):
            raise RuntimeError("v22a matched replacement pairing changed")
        total_replacements += expected_replacements
        total_unchanged += 24 - expected_replacements
        panels[panel] = {
            "name": panel,
            "role": (
                "optimization" if panel in OPTIMIZATION_PANELS_V22A
                else "train_only_screen"
            ),
            "arms": arms,
        }
    if total_replacements != 184 or total_unchanged != 56:
        raise RuntimeError("v22a global materialized replacement count changed")
    result = {
        "schema": "eggroll-es-materialized-v341-matched-replacement-batches-v22a",
        "preregistration": {
            "commit": draw_v22a.PREREG_COMMIT_V22A,
            "file_sha256": draw_v22a.PREREG_FILE_SHA256_V22A,
            "content_sha256": draw_v22a.PREREG_CONTENT_SHA256_V22A,
        },
        "frame": {
            "commit": prereg_v22a.FRAME_COMMIT_V22A,
            "file_sha256": prereg_v22a.FRAME_CERTIFICATE_SHA256_V22A,
            "content_sha256": prereg_v22a.FRAME_CERTIFICATE_CONTENT_SHA256_V22A,
        },
        "draw_plan": {
            "commit": DRAW_PLAN_COMMIT_V22A,
            "file_sha256": DRAW_FILE_SHA256_V22A,
            "content_sha256": DRAW_CONTENT_SHA256_V22A,
        },
        "sources": {
            "candidate_v341": {
                "source_commit": frame_v22a.candidate_v341.V341_SOURCE_COMMIT,
                "rows": 528,
                "file_sha256": frame_v22a.candidate_v341.V341_SHA256,
            },
            "production": {
                "rows": 784,
                "file_sha256": frame_v22a.PRODUCTION_SHA256_V22A,
            },
        },
        "replacement_counts": {
            "sampled_candidate_representatives": 184,
            "sampled_production_representatives_unchanged": 56,
            "candidate_only_components_included": 0,
        },
        "panels": panels,
        "token_boundary_audit_required_before_any_scoring": preregistration[
            "scoring"
        ]["v341_token_boundary_audit_required_before_any_future_runtime"],
        "contains_train_question_answer_content_in_memory": True,
        "contains_evaluation_content": False,
        "persisted_to_disk": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def validate_panel_bundle_v22a(bundle: dict) -> dict:
    if (
        not isinstance(bundle, dict)
        or bundle.get("schema")
        != "eggroll-es-materialized-v341-matched-replacement-batches-v22a"
        or bundle.get("replacement_counts") != {
            "sampled_candidate_representatives": 184,
            "sampled_production_representatives_unchanged": 56,
            "candidate_only_components_included": 0,
        }
        or bundle.get("contains_train_question_answer_content_in_memory") is not True
        or bundle.get("contains_evaluation_content") is not False
        or bundle.get("persisted_to_disk") is not False
        or bundle.get("token_boundary_audit_required_before_any_scoring") is not True
        or tuple(bundle.get("panels", {})) != PANEL_NAMES_V22A
    ):
        raise RuntimeError("v22a materialized panel bundle changed")
    for panel_name in PANEL_NAMES_V22A:
        panel = bundle["panels"][panel_name]
        if set(panel.get("arms", {})) != set(ARMS_V22A):
            raise RuntimeError("v22a materialized arm coverage changed")
        control = panel["arms"]["production_control"]
        treatment = panel["arms"]["v341_matched_replacement"]
        if (
            treatment.get("joint_ids") != control.get("joint_ids")
            or treatment.get("ht_strata") != control.get("ht_strata")
            or treatment.get("weights") != control.get("weights")
            or control.get("representative_sides") != ["production"] * 24
            or treatment.get("representative_sides", []).count("candidate")
            != frame_v22a.EXPECTED_REPLACEMENTS_BY_PANEL_V22A[panel_name]
        ):
            raise RuntimeError("v22a paired panel mapping changed")
        for arm in ARMS_V22A:
            batch = panel["arms"][arm]
            sequence_fields = (
                "joint_ids", "row_sha256", "ht_strata", "weights",
                "representative_sides", "questions", "answers",
            )
            if (
                any(len(batch.get(field, [])) != 24 for field in sequence_fields)
                or len(set(batch["joint_ids"])) != 24
                or canonical_sha256(batch["joint_ids"])
                != batch["ordered_joint_identity_sha256"]
                or canonical_sha256(batch["row_sha256"])
                != batch["ordered_row_identity_sha256"]
                or not math.isclose(
                    math.fsum(batch["weights"]), 272.0,
                    rel_tol=0.0, abs_tol=1e-12,
                )
                or {
                    category: batch["ht_strata"].count(category)
                    for category in BASE_CATEGORIES_V22A
                } != {category: 6 for category in BASE_CATEGORIES_V22A}
                or canonical_sha256({
                    "questions": batch["questions"],
                    "answers": batch["answers"],
                }) != batch["raw_prompt_answer_sha256"]
            ):
                raise RuntimeError("v22a materialized panel batch changed")
    if (
        bundle.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(bundle))
        or bundle.get("content_sha256_before_self_field")
        != PANEL_BUNDLE_CONTENT_SHA256_V22A
    ):
        raise RuntimeError("v22a materialized panel bundle changed")
    return bundle


def load_panel_bundle_v22a() -> dict:
    return validate_panel_bundle_v22a(_materialize_panel_bundle_v22a())


def _unit_score_arrays_v22a(unit_scores) -> dict[str, np.ndarray]:
    if not isinstance(unit_scores, dict) or set(unit_scores) != set(ARMS_V22A):
        raise RuntimeError("v22a per-unit score arm mapping changed")
    result = {}
    for arm in ARMS_V22A:
        values = np.asarray(unit_scores[arm], dtype=np.float64)
        if values.shape != (10, 2, 64, 24) or not np.isfinite(values).all():
            raise RuntimeError("v22a per-unit score tensor changed")
        result[arm] = values
    return result


def observed_panel_scores_v22a(unit_scores, panel_bundle) -> np.ndarray:
    values = _unit_score_arrays_v22a(unit_scores)
    panel_bundle = validate_panel_bundle_v22a(panel_bundle)
    panel_scores = np.empty((2, 10, 2, 64), dtype=np.float64)
    for arm_index, arm in enumerate(ARMS_V22A):
        weights = np.asarray([
            panel_bundle["panels"][panel]["arms"][arm]["weights"]
            for panel in PANEL_NAMES_V22A
        ], dtype=np.float64)
        panel_scores[arm_index] = np.einsum(
            "psdu,pu->psd", values[arm], weights
        ) / 272.0
    if not np.isfinite(panel_scores).all():
        raise RuntimeError("v22a HT panel-score recomputation changed")
    return panel_scores


def _cosine_last_axis(left, right):
    numerator = np.sum(left * right, axis=-1)
    denominator = np.sqrt(
        np.sum(left * left, axis=-1) * np.sum(right * right, axis=-1)
    )
    if np.any(denominator == 0.0):
        raise RuntimeError("v22a endpoint cosine received a zero direction")
    return numerator / denominator


def _sign_agreement_last_axis(left, right):
    return np.mean(np.sign(left) == np.sign(right), axis=-1)


def _endpoint_arrays_v22a(panel_scores):
    values = np.asarray(panel_scores, dtype=np.float64)
    if (
        values.ndim != 5
        or values.shape[0] != 2
        or values.shape[2:] != (10, 2, 64)
        or not np.isfinite(values).all()
    ):
        raise RuntimeError("v22a panel-score endpoint tensor changed")
    central = 0.5 * (values[:, :, :, 0, :] - values[:, :, :, 1, :])
    means = np.mean(central, axis=-1, keepdims=True)
    spreads = np.sqrt(np.mean((central - means) ** 2, axis=-1))
    if np.any(spreads == 0.0):
        raise RuntimeError("v22a panel response spread is zero")
    coefficients = (central - means) / (
        spreads[..., np.newaxis] + STANDARDIZATION_EPSILON_V22A
    )
    aggregate = np.median(coefficients[:, :, :6, :], axis=2)
    pairs = tuple(
        (left, right) for left in range(6) for right in range(left + 1, 6)
    )
    families = {
        "optimization_pairwise_cosine": np.stack([
            _cosine_last_axis(
                coefficients[:, :, left, :], coefficients[:, :, right, :]
            ) for left, right in pairs
        ], axis=-1),
        "optimization_pairwise_sign_agreement": np.stack([
            _sign_agreement_last_axis(
                coefficients[:, :, left, :], coefficients[:, :, right, :]
            ) for left, right in pairs
        ], axis=-1),
        "aggregate_to_optimization_cosine": np.stack([
            _cosine_last_axis(aggregate, coefficients[:, :, index, :])
            for index in range(6)
        ], axis=-1),
        "aggregate_to_optimization_sign_agreement": np.stack([
            _sign_agreement_last_axis(aggregate, coefficients[:, :, index, :])
            for index in range(6)
        ], axis=-1),
        "train_screen_cosine": np.stack([
            _cosine_last_axis(aggregate, coefficients[:, :, index, :])
            for index in range(6, 10)
        ], axis=-1),
        "train_screen_sign_agreement": np.stack([
            _sign_agreement_last_axis(aggregate, coefficients[:, :, index, :])
            for index in range(6, 10)
        ], axis=-1),
    }
    widths = {
        "optimization_pairwise_cosine": 15,
        "optimization_pairwise_sign_agreement": 15,
        "aggregate_to_optimization_cosine": 6,
        "aggregate_to_optimization_sign_agreement": 6,
        "train_screen_cosine": 4,
        "train_screen_sign_agreement": 4,
    }
    if any(
        family.shape != values.shape[:2] + (widths[name],)
        for name, family in families.items()
    ):
        raise RuntimeError("v22a endpoint family geometry changed")
    endpoints = {}
    for family in widths:
        endpoints[f"{family}_median"] = np.median(families[family], axis=-1)
        endpoints[f"{family}_worst"] = np.min(families[family], axis=-1)
    if tuple(endpoints) != ENDPOINT_NAMES_V22A or any(
        item.shape != values.shape[:2] or not np.isfinite(item).all()
        for item in endpoints.values()
    ):
        raise RuntimeError("v22a endpoint family recomputation changed")
    return {
        "endpoints": endpoints,
        "coefficients": coefficients,
        "aggregate": aggregate,
        "spreads": spreads,
    }


def recompute_observed_endpoints_v22a(unit_scores, panel_bundle) -> dict:
    panel_scores = observed_panel_scores_v22a(unit_scores, panel_bundle)
    analyzed = _endpoint_arrays_v22a(panel_scores[:, np.newaxis, ...])
    result = {}
    for arm_index, arm in enumerate(ARMS_V22A):
        endpoint_values = {
            name: float(values[arm_index, 0])
            for name, values in analyzed["endpoints"].items()
        }
        compact_payload = {
            "schema": "eggroll-es-compact-v341-matched-estimator-v22a",
            "arm": arm,
            "panel_scores": panel_scores[arm_index].tolist(),
            "coefficients": analyzed["coefficients"][arm_index, 0].tolist(),
            "aggregate": analyzed["aggregate"][arm_index, 0].tolist(),
            "endpoint_values": endpoint_values,
        }
        result[arm] = {
            "all_panel_spreads_nonzero": bool(
                np.all(analyzed["spreads"][arm_index, 0] > 0.0)
            ),
            "endpoint_values": endpoint_values,
            "compact_estimator_sha256": canonical_sha256(compact_payload),
        }
    return {"arms": result}


def _bootstrap_draw_plan_v22a() -> dict:
    certificate = json.loads(draw_v22a.OUTPUT_PATH_V22A.read_text())
    draw_v22a.validate_draw_plan_certificate_v22a(certificate)
    array = draw_v22a.materialize_draw_array_v22a()
    if (
        hashlib.sha256(array.tobytes()).hexdigest()
        != certificate["base_draws"]["bytes_sha256"]
    ):
        raise RuntimeError("v22a exact bootstrap draw array changed")
    return {
        "base": array,
        "content_sha256": certificate["content_sha256_before_self_field"],
    }


def _stratum_positions_v22a(panel_bundle):
    positions = {
        arm: {
            panel_index: {
                category: np.asarray([
                    index for index, value in enumerate(
                        panel_bundle["panels"][panel]["arms"][arm]["ht_strata"]
                    ) if value == category
                ], dtype=np.int64)
                for category in BASE_CATEGORIES_V22A
            }
            for panel_index, panel in enumerate(PANEL_NAMES_V22A)
        }
        for arm in ARMS_V22A
    }
    if any(
        item.shape != (6,)
        for arm in ARMS_V22A
        for panel in positions[arm].values()
        for item in panel.values()
    ) or any(
        not np.array_equal(
            positions["production_control"][panel_index][category],
            positions["v341_matched_replacement"][panel_index][category],
        )
        for panel_index in range(10) for category in BASE_CATEGORIES_V22A
    ):
        raise RuntimeError("v22a shared bootstrap stratum positions changed")
    return positions


def _bootstrap_panel_scores_v22a(values, panel_bundle, draw_plan, start, stop):
    batch_size = stop - start
    if not 0 <= start < stop <= BOOTSTRAP_REPETITIONS_V22A:
        raise ValueError("v22a bootstrap chunk bounds changed")
    result = np.zeros((2, batch_size, 10, 2, 64), dtype=np.float64)
    positions = _stratum_positions_v22a(panel_bundle)
    for arm_index, arm in enumerate(ARMS_V22A):
        for target in range(10):
            target_scores = np.broadcast_to(
                values[arm][target],
                (batch_size,) + values[arm][target].shape,
            )
            for category_index, category in enumerate(BASE_CATEGORIES_V22A):
                source_positions = positions[arm][target][category]
                sampled_positions = source_positions[
                    draw_plan["base"][target, category_index, start:stop]
                ]
                sampled = np.take_along_axis(
                    target_scores,
                    sampled_positions[:, np.newaxis, np.newaxis, :],
                    axis=-1,
                )
                population = frame_v22a.BASE_CATEGORY_POPULATIONS_V22A[category]
                result[arm_index, :, target] += (
                    float(population) / 272.0
                ) * np.mean(sampled, axis=-1)
    return result


def paired_stratified_bootstrap_v22a(
    unit_scores,
    panel_bundle,
    *,
    chunk_size=BOOTSTRAP_DEFAULT_CHUNK_SIZE_V22A,
):
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("v22a bootstrap chunk size must be positive")
    values = _unit_score_arrays_v22a(unit_scores)
    panel_bundle = validate_panel_bundle_v22a(panel_bundle)
    observed = recompute_observed_endpoints_v22a(values, panel_bundle)["arms"]
    draw_plan = _bootstrap_draw_plan_v22a()
    comparisons = {name: [] for name in ENDPOINT_NAMES_V22A}
    treatment = ARMS_V22A.index("v341_matched_replacement")
    control = ARMS_V22A.index("production_control")
    completed = 0
    while completed < BOOTSTRAP_REPETITIONS_V22A:
        stop = min(BOOTSTRAP_REPETITIONS_V22A, completed + chunk_size)
        panel_scores = _bootstrap_panel_scores_v22a(
            values, panel_bundle, draw_plan, completed, stop
        )
        analyzed = _endpoint_arrays_v22a(panel_scores)
        for name, endpoint_values in analyzed["endpoints"].items():
            comparisons[name].append(
                endpoint_values[treatment] - endpoint_values[control]
            )
        completed = stop
    quantile = 0.05 / 12
    endpoints = {}
    for name, chunks in comparisons.items():
        deltas = np.concatenate(chunks)
        if deltas.shape != (50_000,) or not np.isfinite(deltas).all():
            raise RuntimeError("v22a bootstrap replicate coverage changed")
        observed_delta = (
            observed["v341_matched_replacement"]["endpoint_values"][name]
            - observed["production_control"]["endpoint_values"][name]
        )
        endpoints[name] = {
            "treatment_minus_control": float(observed_delta),
            "familywise_lcb": float(np.quantile(
                deltas, quantile, method=BOOTSTRAP_QUANTILE_METHOD_V22A
            )),
            "noninferiority_margin": 0.0,
        }
    return {
        "seed": prereg_v22a.BOOTSTRAP_SEED_V22A,
        "repetitions": 50_000,
        "one_sided_quantile": quantile,
        "quantile_method": BOOTSTRAP_QUANTILE_METHOD_V22A,
        "draw_plan_content_sha256": draw_plan["content_sha256"],
        "paired_same_draws_both_arms": True,
        "same_ht_coefficients_and_denominator_both_arms": True,
        "candidate_only_resampling_present": False,
        "whole_panel_block_resampling_used": False,
        "comparison": {
            "name": "v341_matched_replacement_vs_production",
            "treatment": "v341_matched_replacement",
            "control": "production_control",
            "endpoints": endpoints,
        },
    }


def evaluate_compatibility_gate_v22a(summary):
    arms = summary.get("arms", {})
    bootstrap = summary.get("paired_bootstrap", {})
    comparison = bootstrap.get("comparison", {})
    endpoints = comparison.get("endpoints", {})
    integrity = summary.get("runtime_integrity", {})
    expected_bootstrap_keys = {
        "seed", "repetitions", "one_sided_quantile", "quantile_method",
        "draw_plan_content_sha256", "paired_same_draws_both_arms",
        "same_ht_coefficients_and_denominator_both_arms",
        "candidate_only_resampling_present", "whole_panel_block_resampling_used",
        "comparison",
    }
    expected_comparison_keys = {"name", "treatment", "control", "endpoints"}
    expected_endpoint_keys = {
        "treatment_minus_control", "familywise_lcb", "noninferiority_margin",
    }

    def finite_number(value):
        return (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and math.isfinite(float(value))
        )

    arm_endpoints = {
        arm: arms.get(arm, {}).get("endpoint_values", {}) for arm in ARMS_V22A
    }
    component_integrity = all(
        integrity.get(key) is True
        for key in RUNTIME_INTEGRITY_KEYS_V22A
        if key not in {"union_planner_called", "all_integrity_audits_passed"}
    ) and integrity.get("union_planner_called") is False
    if (
        set(arms) != set(ARMS_V22A)
        or any(set(arm_endpoints[arm]) != set(ENDPOINT_NAMES_V22A) for arm in ARMS_V22A)
        or any(arms[arm].get("all_panel_spreads_nonzero") not in (True, False) for arm in ARMS_V22A)
        or set(bootstrap) != expected_bootstrap_keys
        or set(comparison) != expected_comparison_keys
        or set(endpoints) != set(ENDPOINT_NAMES_V22A)
        or any(set(item) != expected_endpoint_keys for item in endpoints.values())
        or bootstrap.get("seed") != prereg_v22a.BOOTSTRAP_SEED_V22A
        or bootstrap.get("repetitions") != 50_000
        or bootstrap.get("one_sided_quantile") != 0.05 / 12
        or bootstrap.get("quantile_method") != BOOTSTRAP_QUANTILE_METHOD_V22A
        or bootstrap.get("draw_plan_content_sha256") != DRAW_CONTENT_SHA256_V22A
        or bootstrap.get("paired_same_draws_both_arms") is not True
        or bootstrap.get("same_ht_coefficients_and_denominator_both_arms") is not True
        or bootstrap.get("candidate_only_resampling_present") is not False
        or bootstrap.get("whole_panel_block_resampling_used") is not False
        or comparison.get("name") != "v341_matched_replacement_vs_production"
        or comparison.get("treatment") != "v341_matched_replacement"
        or comparison.get("control") != "production_control"
        or any(
            not all(finite_number(item[key]) for key in expected_endpoint_keys)
            or item["noninferiority_margin"] != 0.0
            for item in endpoints.values()
        )
        or any(
            not finite_number(value)
            for values in arm_endpoints.values() for value in values.values()
        )
        or any(
            endpoints[name]["treatment_minus_control"]
            != arm_endpoints["v341_matched_replacement"][name]
            - arm_endpoints["production_control"][name]
            for name in ENDPOINT_NAMES_V22A
        )
        or set(integrity) != RUNTIME_INTEGRITY_KEYS_V22A
        or integrity.get("union_planner_called") is not False
        or any(
            integrity.get(key) not in (True, False)
            for key in RUNTIME_INTEGRITY_KEYS_V22A
            if key != "union_planner_called"
        )
        or integrity.get("all_integrity_audits_passed") is not component_integrity
    ):
        raise RuntimeError("v22a gate input coverage changed")
    observed = {
        name: item["treatment_minus_control"] >= 0.0
        for name, item in endpoints.items()
    }
    lower = {
        name: item["familywise_lcb"] >= 0.0
        for name, item in endpoints.items()
    }
    spreads = all(
        arms[arm]["all_panel_spreads_nonzero"] is True for arm in ARMS_V22A
    )
    passed = (
        all(observed.values())
        and all(lower.values())
        and spreads
        and integrity["all_integrity_audits_passed"]
    )
    return {
        "schema": "eggroll-es-v341-matched-replacement-compatibility-gate-v22a",
        "observed_pass_count": sum(observed.values()),
        "bootstrap_pass_count": sum(lower.values()),
        "all_twelve_observed_passed": all(observed.values()),
        "all_twelve_bootstrap_passed": all(lower.values()),
        "all_panel_spreads_nonzero": spreads,
        "all_runtime_integrity_audits_passed": integrity[
            "all_integrity_audits_passed"
        ],
        "compatibility_gate_passed": bool(passed),
        "decision": (
            "authorize_only_separate_fresh_basis_train_only_confirmation_preregistration"
            if passed else "retain_production_dataset_and_v13_recipe"
        ),
        "dataset_promotion_authorized": False,
        "model_update_authorized": False,
        "evaluation_authorized": False,
    }


def build_compact_estimator_summary_v22a(unit_scores, panel_bundle):
    observed = recompute_observed_endpoints_v22a(unit_scores, panel_bundle)
    value = {
        "schema": "eggroll-es-v341-matched-replacement-compact-estimator-v22a",
        "arms": copy.deepcopy(observed["arms"]),
        "paired_bootstrap": paired_stratified_bootstrap_v22a(
            unit_scores, panel_bundle
        ),
        "runtime_integrity_required_before_gate": True,
        "persisted_response_vectors_or_row_content": False,
        "bootstrap_replicates_persisted": False,
        "bootstrap_draws_persisted": False,
        "unit_scores_persisted": False,
        "dataset_promotion_authorized": False,
        "model_update_authorized": False,
        "evaluation_authorized": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value
