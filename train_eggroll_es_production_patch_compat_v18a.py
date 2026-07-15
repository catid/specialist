#!/usr/bin/env python3
"""Pure trainer mechanics for the V18A production-preserving patch scan."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

import build_eggroll_es_overlay_frame_v18a as frame_v18a
import eggroll_es_production_overlay_scan_preregistration_v18a as prereg_v18a
import train_eggroll_es_specialist_anchor_v13 as anchor_v13


ROOT = Path(__file__).resolve().parent
ARMS_V18A = prereg_v18a.ARM_ORDER_V18A
PANEL_NAMES_V18A = frame_v18a.PANEL_NAMES_V18A
OPTIMIZATION_PANELS_V18A = PANEL_NAMES_V18A[:3]
TRAIN_SCREENS_V18A = PANEL_NAMES_V18A[3:]
SIGNS_V18A = ("plus", "minus")
BASE_CATEGORIES_V18A = frame_v18a.BASE_CATEGORIES_V18A
BASE_FRAME_COMMIT_V18A = "7055a62d67d030cfddf594c9ecf1d9e290f9d0d5"
CORRECTED_PREREG_COMMIT_V18A = "3b7762215280be2d2bec2c63ec29a58ff7aadc6d"
PREREGISTRATION_FILE_SHA256_V18A = (
    "deb3522b5bcb32639702390450344b0e71f0820d418b843c86bfcaea21431b85"
)
PREREGISTRATION_CONTENT_SHA256_V18A = (
    "72509385d9df9fe2ce36a81c065dae995268cfcb3708eb10cd76ce39f9e77151"
)
PANEL_BUNDLE_CONTENT_SHA256_V18A = (
    "6c8e7a7bb923e047f76fc4b46db8fe138b9222dc14ad43156918518a777053e1"
)
BOOTSTRAP_CHUNK_SIZE_V18A = 128
BOOTSTRAP_QUANTILE_METHOD_V18A = "linear"

canonical_sha256 = prereg_v18a.canonical_sha256
file_sha256 = prereg_v18a.file_sha256


def _without_self_v18a(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _verify_commit_file_v18a(path: Path, commit: str, digest: str) -> None:
    path = Path(path).resolve()
    relative = path.relative_to(ROOT).as_posix()
    raw = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
    )
    if hashlib.sha256(raw).hexdigest() != digest or file_sha256(path) != digest:
        raise RuntimeError(f"v18a committed artifact changed: {relative}")


def load_hardened_preregistration_v18a() -> dict:
    base_bindings = {
        frame_v18a.CANDIDATE_PATH_V18A: frame_v18a.CANDIDATE_SHA256_V18A,
        frame_v18a.CANDIDATE_MANIFEST_PATH_V18A: (
            frame_v18a.CANDIDATE_MANIFEST_SHA256_V18A
        ),
        frame_v18a.OUTPUT_PATH_V18A: (
            prereg_v18a.FLOW_CERTIFICATE_FILE_SHA256_V18A
        ),
        Path(frame_v18a.__file__).resolve(): (
            prereg_v18a.FRAME_BUILDER_FILE_SHA256_V18A
        ),
    }
    for path, digest in base_bindings.items():
        _verify_commit_file_v18a(path, BASE_FRAME_COMMIT_V18A, digest)
    corrected_bindings = {
        prereg_v18a.OUTPUT_PATH_V18A: PREREGISTRATION_FILE_SHA256_V18A,
        Path(prereg_v18a.__file__).resolve(): (
            "683dace9ae2328dcf0d8c79e5a5b69ecdcc62dc55586f64376e4dc8fdcb4819f"
        ),
        prereg_v18a.token_audit_v18a.OUTPUT_PATH_V18A: (
            prereg_v18a.TOKEN_AUDIT_FILE_SHA256_V18A
        ),
        Path(prereg_v18a.token_audit_v18a.__file__).resolve(): (
            prereg_v18a.TOKEN_AUDIT_BUILDER_FILE_SHA256_V18A
        ),
    }
    for path, digest in corrected_bindings.items():
        _verify_commit_file_v18a(path, CORRECTED_PREREG_COMMIT_V18A, digest)
    persisted = json.loads(
        prereg_v18a.OUTPUT_PATH_V18A.read_text(encoding="utf-8")
    )
    if (
        persisted != prereg_v18a.build_preregistration_v18a()
        or persisted.get("content_sha256_before_self_field")
        != PREREGISTRATION_CONTENT_SHA256_V18A
        or persisted.get("patch_semantics", {}).get(
            "paired_candidate_and_production_both_present_same_arm"
        ) is not False
        or persisted.get("scoring", {}).get(
            "objective_change_allowed_in_v18a"
        ) is not False
    ):
        raise RuntimeError("v18a hardened preregistration content changed")
    return persisted


def _panel_order_v18a(panel: str, component_indices: list[int], components):
    return sorted(
        component_indices,
        key=lambda index: canonical_sha256({
            "schema": "eggroll-es-runtime-panel-order-v18a",
            "seed": frame_v18a.FLOW_SEED_V18A,
            "panel": panel,
            "joint_id": components[index]["joint_id"],
        }),
    )


def _materialize_patch_panel_bundle_v18a() -> dict:
    preregistration = load_hardened_preregistration_v18a()
    candidate, production = frame_v18a.load_bound_rows_v18a()
    frame = frame_v18a.build_joint_frame_v18a(candidate, production)
    layers = frame_v18a.assign_patch_layers_v18a(frame)
    flow = frame_v18a.solve_patch_flow_v18a(frame, layers)
    persisted_frame = json.loads(
        frame_v18a.OUTPUT_PATH_V18A.read_text(encoding="utf-8")
    )
    if frame_v18a.build_certificate_v18a() != persisted_frame:
        raise RuntimeError("v18a reconstructed patch frame differs from certificate")
    components = frame["joint_components"]
    side_rows = {"candidate": candidate, "production": production}
    panels = {}
    arm_tiers = dict(zip(ARMS_V18A, range(4)))
    for panel in PANEL_NAMES_V18A:
        selected = flow["selected"][panel]
        base_indices = _panel_order_v18a(
            panel,
            [index for index in selected if index in layers["base_category"]],
            components,
        )
        candidate_only_by_layer = {
            layer: next(
                index for index in selected
                if layers["candidate_only_layer"].get(index) == layer
            )
            for layer in (1, 2, 3)
        }
        arms = {}
        for arm, maximum_tier in arm_tiers.items():
            active_indices = base_indices + [
                candidate_only_by_layer[layer]
                for layer in range(1, maximum_tier + 1)
            ]
            rows = []
            row_sha256 = []
            joint_ids = []
            ht_strata = []
            weights = []
            representative_sides = []
            for component_index in active_indices:
                component = components[component_index]
                representative = frame_v18a._representative_for_arm_v18a(
                    component,
                    maximum_tier,
                    layers["paired_tier"].get(component_index),
                    layers["candidate_only_layer"].get(component_index),
                    frame,
                )
                if representative is None:
                    raise RuntimeError("v18a active component lacks representative")
                side, expected_row_sha256 = representative
                unit = frame["side_units"][side][component[f"{side}_unit"]]
                row = side_rows[side][unit["representative_index"]]
                if (
                    frame_v18a.sampler_v13.row_sha256(row)
                    != expected_row_sha256
                    or not isinstance(row.get("question"), str)
                    or not row["question"]
                    or not isinstance(row.get("answer"), str)
                    or not row["answer"]
                ):
                    raise RuntimeError("v18a fixed representative content changed")
                if component_index in layers["base_category"]:
                    stratum = layers["base_category"][component_index]
                    population = frame_v18a.BASE_CATEGORY_POPULATIONS_V18A[stratum]
                    weight = population / frame_v18a.BASE_CATEGORY_QUOTA_V18A
                else:
                    layer = layers["candidate_only_layer"][component_index]
                    stratum = f"candidate_only_layer_{layer}"
                    weight = float(
                        frame_v18a.CANDIDATE_ONLY_LAYER_POPULATIONS_V18A[layer]
                    )
                rows.append(row)
                row_sha256.append(expected_row_sha256)
                joint_ids.append(component["joint_id"])
                ht_strata.append(stratum)
                weights.append(float(weight))
                representative_sides.append(side)
            if (
                len(rows) != frame_v18a.ARM_REQUESTS_PER_PANEL_V18A[arm]
                or len(joint_ids) != len(set(joint_ids))
                or not math.isclose(
                    math.fsum(weights),
                    float(frame_v18a.ARM_POPULATIONS_V18A[arm]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise RuntimeError("v18a materialized arm estimand changed")
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
        panels[panel] = {
            "name": panel,
            "role": "optimization" if panel in OPTIMIZATION_PANELS_V18A else "train_only_screen",
            "arms": arms,
        }
    result = {
        "schema": "eggroll-es-materialized-production-patch-batches-v18a",
        "preregistration": {
            "file_sha256": PREREGISTRATION_FILE_SHA256_V18A,
            "content_sha256": preregistration["content_sha256_before_self_field"],
            "corrected_commit": CORRECTED_PREREG_COMMIT_V18A,
        },
        "frame": {
            "file_sha256": prereg_v18a.FLOW_CERTIFICATE_FILE_SHA256_V18A,
            "content_sha256": prereg_v18a.FLOW_CERTIFICATE_CONTENT_SHA256_V18A,
            "base_commit": BASE_FRAME_COMMIT_V18A,
        },
        "sources": {
            "candidate_v298": {
                "rows": frame_v18a.CANDIDATE_ROWS_V18A,
                "file_sha256": frame_v18a.CANDIDATE_SHA256_V18A,
            },
            "production": {
                "rows": frame_v18a.PRODUCTION_ROWS_V18A,
                "file_sha256": frame_v18a.PRODUCTION_SHA256_V18A,
            },
        },
        "panels": panels,
        "contains_evaluation_content": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def validate_patch_panel_bundle_v18a(bundle: dict) -> dict:
    if (
        not isinstance(bundle, dict)
        or bundle.get("schema")
        != "eggroll-es-materialized-production-patch-batches-v18a"
        or bundle.get("contains_evaluation_content") is not False
        or bundle.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self_v18a(bundle))
        or bundle.get("content_sha256_before_self_field")
        != PANEL_BUNDLE_CONTENT_SHA256_V18A
        or tuple(bundle.get("panels", {})) != PANEL_NAMES_V18A
    ):
        raise RuntimeError("v18a materialized patch panel bundle changed")
    for panel_name in PANEL_NAMES_V18A:
        panel = bundle["panels"][panel_name]
        if set(panel.get("arms", {})) != set(ARMS_V18A):
            raise RuntimeError("v18a materialized arm coverage changed")
        for arm in ARMS_V18A:
            batch = panel["arms"][arm]
            expected_rows = frame_v18a.ARM_REQUESTS_PER_PANEL_V18A[arm]
            sequence_fields = (
                "joint_ids", "row_sha256", "ht_strata", "weights",
                "representative_sides", "questions", "answers",
            )
            if (
                any(len(batch.get(field, [])) != expected_rows for field in sequence_fields)
                or len(set(batch["joint_ids"])) != expected_rows
                or canonical_sha256(batch["joint_ids"])
                != batch["ordered_joint_identity_sha256"]
                or canonical_sha256(batch["row_sha256"])
                != batch["ordered_row_identity_sha256"]
                or not math.isclose(
                    math.fsum(batch["weights"]),
                    float(frame_v18a.ARM_POPULATIONS_V18A[arm]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or canonical_sha256({
                    "questions": batch["questions"],
                    "answers": batch["answers"],
                }) != batch["raw_prompt_answer_sha256"]
            ):
                raise RuntimeError("v18a materialized patch batch changed")
    return bundle


def load_patch_panel_bundle_v18a() -> dict:
    return validate_patch_panel_bundle_v18a(
        _materialize_patch_panel_bundle_v18a()
    )


def resident_signed_wave_schedule_v18a(seeds=None) -> list[dict]:
    seeds = list(anchor_v13.PERTURBATION_SEEDS_V13 if seeds is None else seeds)
    if seeds != anchor_v13.PERTURBATION_SEEDS_V13:
        raise RuntimeError("v18a fixed perturbation basis changed")
    prereg_orders = prereg_v18a.signed_wave_arm_orders_v18a()
    schedule = []
    for wave_index, start in enumerate(range(0, len(seeds), 4)):
        wave = [int(seed) for seed in seeds[start:start + 4]]
        if len(wave) != 4:
            raise RuntimeError("v18a partial four-engine wave is forbidden")
        for sign_index, sign in enumerate(SIGNS_V18A):
            signed_wave_index = 2 * wave_index + sign_index
            schedule.append({
                "signed_wave_index": signed_wave_index,
                "population_wave_index": wave_index,
                "sign": sign,
                "negate": sign == "minus",
                "engine_seeds": wave,
                "resident_arm_order": prereg_orders[signed_wave_index]["arm_order"],
                "restore_after_all_four_arms": True,
            })
    if (
        len(schedule) != 16
        or [item["signed_wave_index"] for item in schedule] != list(range(16))
        or any(set(item["resident_arm_order"]) != set(ARMS_V18A) for item in schedule)
    ):
        raise RuntimeError("v18a resident four-arm schedule changed")
    return schedule


def execute_patch_resident_signed_wave_v18a(
    schedule_item, *, perturb, score_arm, restore,
):
    signed_wave_index = (
        schedule_item.get("signed_wave_index")
        if isinstance(schedule_item, dict)
        else None
    )
    if (
        not isinstance(schedule_item, dict)
        or not isinstance(signed_wave_index, int)
        or signed_wave_index not in range(16)
        or schedule_item
        != resident_signed_wave_schedule_v18a()[signed_wave_index]
    ):
        raise RuntimeError("v18a resident signed-wave schedule item changed")
    if not all(callable(item) for item in (perturb, score_arm, restore)):
        raise TypeError("v18a resident signed-wave callbacks must be callable")
    captures = {}
    try:
        perturb(list(schedule_item["engine_seeds"]), bool(schedule_item["negate"]))
        for arm in schedule_item["resident_arm_order"]:
            captures[arm] = score_arm(arm)
    finally:
        restore()
    if tuple(captures) != tuple(schedule_item["resident_arm_order"]):
        raise RuntimeError("v18a resident four-arm capture is incomplete")
    return captures


def _unit_score_arrays_v18a(unit_scores) -> dict[str, np.ndarray]:
    if not isinstance(unit_scores, dict) or set(unit_scores) != set(ARMS_V18A):
        raise RuntimeError("v18a per-unit score arm mapping changed")
    result = {}
    for arm in ARMS_V18A:
        values = np.asarray(unit_scores[arm], dtype=np.float64)
        expected = (
            5, 2, 32, frame_v18a.ARM_REQUESTS_PER_PANEL_V18A[arm]
        )
        if values.shape != expected or not np.isfinite(values).all():
            raise RuntimeError("v18a per-unit score tensor changed")
        result[arm] = values
    return result


def observed_panel_scores_v18a(unit_scores, panel_bundle) -> np.ndarray:
    values = _unit_score_arrays_v18a(unit_scores)
    panel_bundle = validate_patch_panel_bundle_v18a(panel_bundle)
    panel_scores = np.empty((4, 5, 2, 32), dtype=np.float64)
    for arm_index, arm in enumerate(ARMS_V18A):
        weights = np.asarray([
            panel_bundle["panels"][panel]["arms"][arm]["weights"]
            for panel in PANEL_NAMES_V18A
        ], dtype=np.float64)
        panel_scores[arm_index] = np.einsum(
            "psdu,pu->psd", values[arm], weights
        ) / float(frame_v18a.ARM_POPULATIONS_V18A[arm])
    if not np.isfinite(panel_scores).all():
        raise RuntimeError("v18a HT panel-score recomputation changed")
    return panel_scores


def _cosine_last_axis_v18a(left, right):
    numerator = np.sum(left * right, axis=-1)
    denominator = np.sqrt(
        np.sum(left * left, axis=-1) * np.sum(right * right, axis=-1)
    )
    if np.any(denominator == 0.0):
        raise RuntimeError("v18a endpoint cosine received a zero direction")
    return numerator / denominator


def _sign_agreement_last_axis_v18a(left, right):
    return np.mean(np.sign(left) == np.sign(right), axis=-1)


def _endpoint_arrays_v18a(panel_scores):
    values = np.asarray(panel_scores, dtype=np.float64)
    if (
        values.ndim != 5
        or values.shape[0] != 4
        or values.shape[2:] != (5, 2, 32)
        or not np.isfinite(values).all()
    ):
        raise RuntimeError("v18a panel-score endpoint tensor changed")
    central = 0.5 * (values[:, :, :, 0, :] - values[:, :, :, 1, :])
    means = np.mean(central, axis=-1, keepdims=True)
    spreads = np.sqrt(np.mean((central - means) ** 2, axis=-1))
    if np.any(spreads == 0.0):
        raise RuntimeError("v18a panel response spread is zero")
    coefficients = (central - means) / (
        spreads[..., np.newaxis] + anchor_v13.STANDARDIZATION_EPSILON_V13
    )
    aggregate = np.median(coefficients[:, :, :3, :], axis=2)
    pairs = ((0, 1), (0, 2), (1, 2))
    families = {
        "optimization_pairwise_cosine": np.stack([
            _cosine_last_axis_v18a(
                coefficients[:, :, left, :], coefficients[:, :, right, :]
            )
            for left, right in pairs
        ], axis=-1),
        "optimization_pairwise_sign_agreement": np.stack([
            _sign_agreement_last_axis_v18a(
                coefficients[:, :, left, :], coefficients[:, :, right, :]
            )
            for left, right in pairs
        ], axis=-1),
        "aggregate_to_optimization_cosine": np.stack([
            _cosine_last_axis_v18a(aggregate, coefficients[:, :, index, :])
            for index in range(3)
        ], axis=-1),
        "aggregate_to_optimization_sign_agreement": np.stack([
            _sign_agreement_last_axis_v18a(aggregate, coefficients[:, :, index, :])
            for index in range(3)
        ], axis=-1),
        "train_screen_cosine": np.stack([
            _cosine_last_axis_v18a(aggregate, coefficients[:, :, index, :])
            for index in (3, 4)
        ], axis=-1),
        "train_screen_sign_agreement": np.stack([
            _sign_agreement_last_axis_v18a(aggregate, coefficients[:, :, index, :])
            for index in (3, 4)
        ], axis=-1),
    }
    endpoints = {}
    for family in prereg_v18a.prereg_v17a.METRIC_FAMILIES_V17A:
        endpoints[f"{family}_median"] = np.median(families[family], axis=-1)
        endpoints[f"{family}_worst"] = np.min(families[family], axis=-1)
    if set(endpoints) != set(prereg_v18a.ENDPOINT_NAMES_V18A) or any(
        item.shape != values.shape[:2] or not np.isfinite(item).all()
        for item in endpoints.values()
    ):
        raise RuntimeError("v18a endpoint family recomputation changed")
    return {
        "endpoints": endpoints,
        "coefficients": coefficients,
        "aggregate": aggregate,
        "spreads": spreads,
    }


def recompute_observed_endpoints_v18a(unit_scores, panel_bundle) -> dict:
    panel_scores = observed_panel_scores_v18a(unit_scores, panel_bundle)
    analyzed = _endpoint_arrays_v18a(panel_scores[:, np.newaxis, ...])
    result = {}
    for arm_index, arm in enumerate(ARMS_V18A):
        endpoint_values = {
            name: float(values[arm_index, 0])
            for name, values in analyzed["endpoints"].items()
        }
        compact_payload = {
            "schema": "eggroll-es-compact-patch-estimator-commitment-v18a",
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


def _bootstrap_panel_scores_v18a(
    values, panel_bundle, rng, batch_size,
) -> np.ndarray:
    result = np.zeros((4, batch_size, 5, 2, 32), dtype=np.float64)
    base_positions = {
        arm: {
            panel_index: {
                category: np.asarray([
                    index for index, value in enumerate(
                        panel_bundle["panels"][panel]["arms"][arm]["ht_strata"]
                    ) if value == category
                ], dtype=np.int64)
                for category in BASE_CATEGORIES_V18A
            }
            for panel_index, panel in enumerate(PANEL_NAMES_V18A)
        }
        for arm in ARMS_V18A
    }
    layer_positions = {
        arm: {
            panel_index: {
                layer: next((
                    index for index, value in enumerate(
                        panel_bundle["panels"][panel]["arms"][arm]["ht_strata"]
                    ) if value == f"candidate_only_layer_{layer}"
                ), None)
                for layer in (1, 2, 3)
            }
            for panel_index, panel in enumerate(PANEL_NAMES_V18A)
        }
        for arm in ARMS_V18A
    }
    shared_draws = {
        (target, category): rng.integers(
            0, 13, size=(batch_size, 13), dtype=np.int64,
        )
        for target in range(5) for category in BASE_CATEGORIES_V18A
    }
    candidate_sources = {}
    for role_indices in (np.asarray([0, 1, 2]), np.asarray([3, 4])):
        for target in role_indices:
            for layer in (1, 2, 3):
                source_offsets = rng.integers(
                    0, len(role_indices), size=batch_size, dtype=np.int64,
                )
                candidate_sources[(int(target), layer)] = role_indices[
                    source_offsets
                ]
    for arm_index, arm in enumerate(ARMS_V18A):
        denominator = float(frame_v18a.ARM_POPULATIONS_V18A[arm])
        maximum_tier = ARMS_V18A.index(arm)
        for target in range(5):
            target_scores = np.broadcast_to(
                values[arm][target],
                (batch_size,) + values[arm][target].shape,
            )
            for category in BASE_CATEGORIES_V18A:
                positions = base_positions[arm][target][category]
                if positions.shape != (13,):
                    raise RuntimeError("v18a bootstrap base quota changed")
                sampled_positions = positions[
                    shared_draws[(target, category)]
                ]
                sampled = np.take_along_axis(
                    target_scores,
                    sampled_positions[:, np.newaxis, np.newaxis, :],
                    axis=-1,
                )
                population = frame_v18a.BASE_CATEGORY_POPULATIONS_V18A[category]
                result[arm_index, :, target] += (
                    float(population) / denominator
                ) * np.mean(sampled, axis=-1)
            for layer in range(1, maximum_tier + 1):
                sources = candidate_sources[(target, layer)]
                source_scores = values[arm][sources]
                positions = np.asarray([
                    layer_positions[arm][int(source)][layer]
                    for source in sources
                ], dtype=np.int64)
                if np.any(positions < 0):
                    raise RuntimeError("v18a bootstrap candidate-only layer changed")
                sampled = np.take_along_axis(
                    source_scores,
                    positions[:, np.newaxis, np.newaxis, np.newaxis],
                    axis=-1,
                )[..., 0]
                population = frame_v18a.CANDIDATE_ONLY_LAYER_POPULATIONS_V18A[layer]
                result[arm_index, :, target] += (
                    float(population) / denominator
                ) * sampled
    return result


def paired_stratified_bootstrap_v18a(unit_scores, panel_bundle) -> dict:
    values = _unit_score_arrays_v18a(unit_scores)
    panel_bundle = validate_patch_panel_bundle_v18a(panel_bundle)
    observed = recompute_observed_endpoints_v18a(values, panel_bundle)["arms"]
    rng = np.random.default_rng(prereg_v18a.BOOTSTRAP_SEED_V18A)
    comparisons = {
        arm: {name: [] for name in prereg_v18a.ENDPOINT_NAMES_V18A}
        for arm in ARMS_V18A[1:]
    }
    completed = 0
    while completed < prereg_v18a.BOOTSTRAP_REPETITIONS_V18A:
        batch_size = min(
            BOOTSTRAP_CHUNK_SIZE_V18A,
            prereg_v18a.BOOTSTRAP_REPETITIONS_V18A - completed,
        )
        panel_scores = _bootstrap_panel_scores_v18a(
            values, panel_bundle, rng, batch_size,
        )
        analyzed = _endpoint_arrays_v18a(panel_scores)
        for arm_index, arm in enumerate(ARMS_V18A[1:], 1):
            for name, endpoint_values in analyzed["endpoints"].items():
                comparisons[arm][name].append(
                    endpoint_values[arm_index] - endpoint_values[0]
                )
        completed += batch_size
    quantile = prereg_v18a.FAMILYWISE_ALPHA_V18A / 36
    output = {}
    for arm in ARMS_V18A[1:]:
        endpoints = {}
        for name, chunks in comparisons[arm].items():
            deltas = np.concatenate(chunks)
            if (
                deltas.shape != (prereg_v18a.BOOTSTRAP_REPETITIONS_V18A,)
                or not np.isfinite(deltas).all()
            ):
                raise RuntimeError("v18a bootstrap replicate coverage changed")
            observed_delta = (
                observed[arm]["endpoint_values"][name]
                - observed["production_only"]["endpoint_values"][name]
            )
            endpoints[name] = {
                "patch_minus_production": float(observed_delta),
                "familywise_lcb": float(np.quantile(
                    deltas,
                    quantile,
                    method=BOOTSTRAP_QUANTILE_METHOD_V18A,
                )),
                "noninferiority_margin": 0.0,
            }
        output[arm] = endpoints
    return {
        "seed": prereg_v18a.BOOTSTRAP_SEED_V18A,
        "repetitions": prereg_v18a.BOOTSTRAP_REPETITIONS_V18A,
        "one_sided_quantile": quantile,
        "comparisons": output,
    }


def evaluate_patch_gate_v18a(summary: dict) -> dict:
    arms = summary.get("arms", {})
    bootstrap = summary.get("paired_bootstrap", {})
    integrity = summary.get("runtime_integrity", {})
    if (
        set(arms) != set(ARMS_V18A)
        or set(bootstrap.get("comparisons", {})) != set(ARMS_V18A[1:])
        or bootstrap.get("repetitions") != 50_000
    ):
        raise RuntimeError("v18a gate input coverage changed")
    arm_results = {}
    for arm in ARMS_V18A[1:]:
        endpoints = bootstrap["comparisons"][arm]
        if set(endpoints) != set(prereg_v18a.ENDPOINT_NAMES_V18A):
            raise RuntimeError("v18a gate endpoint coverage changed")
        observed_pass = {
            name: item["patch_minus_production"] >= 0.0
            for name, item in endpoints.items()
        }
        bootstrap_pass = {
            name: item["familywise_lcb"] >= 0.0
            for name, item in endpoints.items()
        }
        passed = (
            all(observed_pass.values())
            and all(bootstrap_pass.values())
            and arms[arm].get("all_panel_spreads_nonzero") is True
            and arms["production_only"].get("all_panel_spreads_nonzero") is True
            and integrity.get("all_integrity_audits_passed") is True
        )
        arm_results[arm] = {
            "observed_pass_count": sum(observed_pass.values()),
            "bootstrap_pass_count": sum(bootstrap_pass.values()),
            "all_twelve_observed_passed": all(observed_pass.values()),
            "all_twelve_bootstrap_passed": all(bootstrap_pass.values()),
            "preregistered_gate_passed": bool(passed),
        }
    passing = [
        arm for arm in ARMS_V18A[1:]
        if arm_results[arm]["preregistered_gate_passed"]
    ]
    selected = passing[-1] if passing else None
    return {
        "schema": "eggroll-es-production-patch-gate-v18a",
        "arms": arm_results,
        "selected_largest_passing_patch": selected,
        "any_patch_passed": bool(passing),
        "decision": (
            "authorize_separate_train_only_recipe_preregistration"
            if selected else "retain_production_dataset_and_v13_recipe"
        ),
        "dataset_promotion_authorized": False,
        "model_update_authorized": False,
        "evaluation_authorized": False,
    }


def build_compact_estimator_summary_v18a(unit_scores, panel_bundle) -> dict:
    observed = recompute_observed_endpoints_v18a(unit_scores, panel_bundle)
    return {
        "arms": copy.deepcopy(observed["arms"]),
        "paired_bootstrap": paired_stratified_bootstrap_v18a(
            unit_scores, panel_bundle,
        ),
        "persisted_response_vectors_or_row_content": False,
    }
