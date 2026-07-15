#!/usr/bin/env python3
"""Pure mechanics for the train-only V19A disjoint-tier attribution scan."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

import build_eggroll_es_disjoint_tier_frame_v19a as frame_v19a
import eggroll_es_disjoint_tier_attribution_preregistration_v19a as prereg_v19a


ROOT = Path(__file__).resolve().parent
ARMS_V19A = prereg_v19a.ARM_ORDER_V19A
PANEL_NAMES_V19A = frame_v19a.PANEL_NAMES_V19A
OPTIMIZATION_PANELS_V19A = frame_v19a.OPTIMIZATION_PANELS_V19A
TRAIN_SCREEN_PANELS_V19A = frame_v19a.TRAIN_SCREEN_PANELS_V19A
BASE_CATEGORIES_V19A = frame_v19a.BASE_CATEGORIES_V19A
SIGNS_V19A = ("plus", "minus")

SEALED_PREREG_COMMIT_V19A = "97cddf65df07ce1c87e3838272f2558bcb53d911"
FRAME_BUILDER_FILE_SHA256_V19A = (
    "4de43c263f580143d1b1f09a67240b8a05df7d45b419b19a211b549bf85b2052"
)
FRAME_CERTIFICATE_FILE_SHA256_V19A = (
    "50820c67844a7b11c92bc0bbaa9c594c683440a65cb882de244216a77dca5fed"
)
FRAME_CERTIFICATE_CONTENT_SHA256_V19A = (
    "7ad195a55b1e51268dfba1cddb43f869b014bdb1ef329f5d73c8246ac6cbff58"
)
PREREGISTRATION_BUILDER_FILE_SHA256_V19A = (
    "e684da1e53a27c535eb441d0ce8109b8aadff9bad2f49117ea27b97c24d41365"
)
PREREGISTRATION_FILE_SHA256_V19A = (
    "cd65f6127ce8d4b401d837bb23c424f0860259ab6920e25a68a4098f1798a987"
)
PREREGISTRATION_CONTENT_SHA256_V19A = (
    "db2036da7c9b382a542d7c002f5af8a76bbe1101d342fad8bfdc8e3a67b6c997"
)
PANEL_BUNDLE_CONTENT_SHA256_V19A = (
    "b9bfc1868f5e2a6f54cd9531e0b759872020d2bc8fb9e8a8a287b548293d4f06"
)

BOOTSTRAP_REPETITIONS_V19A = 50_000
BOOTSTRAP_DEFAULT_CHUNK_SIZE_V19A = 128
BOOTSTRAP_QUANTILE_METHOD_V19A = "linear"
BOOTSTRAP_DRAW_PLAN_CONTENT_SHA256_V19A = (
    "e1842969819e6d26836aafe7d4640d2cb0f6540530007b60a9ce950103b0865a"
)
STANDARDIZATION_EPSILON_V19A = 1e-8

canonical_sha256 = prereg_v19a.canonical_sha256
file_sha256 = prereg_v19a.file_sha256


def _without_self(value: dict) -> dict:
    return {
        key: item
        for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _verify_commit_file_v19a(path: Path, digest: str) -> None:
    path = Path(path).resolve()
    relative = path.relative_to(ROOT).as_posix()
    raw = subprocess.check_output(
        ["git", "show", f"{SEALED_PREREG_COMMIT_V19A}:{relative}"], cwd=ROOT
    )
    if hashlib.sha256(raw).hexdigest() != digest or file_sha256(path) != digest:
        raise RuntimeError(f"v19a committed artifact changed: {relative}")


def load_hardened_preregistration_v19a() -> dict:
    bindings = {
        Path(frame_v19a.__file__).resolve(): FRAME_BUILDER_FILE_SHA256_V19A,
        frame_v19a.OUTPUT_PATH_V19A: FRAME_CERTIFICATE_FILE_SHA256_V19A,
        Path(prereg_v19a.__file__).resolve(): (
            PREREGISTRATION_BUILDER_FILE_SHA256_V19A
        ),
        prereg_v19a.OUTPUT_PATH_V19A: PREREGISTRATION_FILE_SHA256_V19A,
    }
    for path, digest in bindings.items():
        _verify_commit_file_v19a(path, digest)
    flow = json.loads(frame_v19a.OUTPUT_PATH_V19A.read_text(encoding="utf-8"))
    frame_v19a.validate_certificate_v19a(flow)
    persisted = json.loads(
        prereg_v19a.OUTPUT_PATH_V19A.read_text(encoding="utf-8")
    )
    if (
        flow.get("content_sha256_before_self_field")
        != FRAME_CERTIFICATE_CONTENT_SHA256_V19A
        or persisted.get("content_sha256_before_self_field")
        != PREREGISTRATION_CONTENT_SHA256_V19A
        or persisted != prereg_v19a.build_preregistration_v19a()
        or persisted.get("frozen_recipe", {}).get("perturbation_basis_sha256")
        != prereg_v19a.PERTURBATION_BASIS_SHA256_V19A
        or persisted.get("frozen_recipe", {}).get("perturbation_basis", {}).get(
            "seeds"
        )
        != prereg_v19a.PERTURBATION_SEEDS_V19A
        or persisted.get("firewall", {}).get("gpu_launch_allowed") is not False
    ):
        raise RuntimeError("v19a hardened preregistration content changed")
    return persisted


def _panel_order_v19a(panel: str, component_indices: list[int], components):
    return sorted(
        component_indices,
        key=lambda index: canonical_sha256({
            "schema": "eggroll-es-runtime-panel-order-v19a",
            "seed": frame_v19a.FLOW_SEED_V19A,
            "panel": panel,
            "joint_id": components[index]["joint_id"],
        }),
    )


def _materialize_panel_bundle_v19a() -> dict:
    preregistration = load_hardened_preregistration_v19a()
    candidate, production = frame_v19a.frame_v18a.load_bound_rows_v18a()
    frame = frame_v19a.frame_v18a.build_joint_frame_v18a(candidate, production)
    tiers = frame_v19a.frame_v18a.assign_patch_layers_v18a(frame)
    base_flow = frame_v19a.solve_base_flow_v19a(frame, tiers)
    candidate_assignment = frame_v19a.assign_candidate_only_v19a(frame, tiers)
    persisted_frame = json.loads(
        frame_v19a.OUTPUT_PATH_V19A.read_text(encoding="utf-8")
    )
    if frame_v19a.build_certificate_v19a() != persisted_frame:
        raise RuntimeError("v19a reconstructed frame differs from certificate")
    token_audit = preregistration["immutable_inputs"]["token_length_audit"]
    if (
        token_audit.get("over_frozen_1024_total_token_cap_count") != 0
        or token_audit.get("observed_combined_token_max", 1025) > 1024
        or token_audit.get("content_sha256")
        != prereg_v19a.TOKEN_AUDIT_CONTENT_SHA256_V19A
    ):
        raise RuntimeError("v19a inherited V18 token audit changed")

    components = frame["joint_components"]
    side_rows = {"candidate": candidate, "production": production}
    certificate_panels = {
        panel["name"]: panel for panel in persisted_frame["panels"]
    }
    panels = {}
    for panel in PANEL_NAMES_V19A:
        base_indices = _panel_order_v19a(
            panel, base_flow["selected_base"][panel], components
        )
        arms = {}
        for arm in ARMS_V19A:
            active_tier = frame_v19a.ARM_ACTIVE_TIER_V19A[arm]
            active_indices = list(base_indices)
            if active_tier is not None:
                active_indices.append(
                    candidate_assignment["assignments"][panel][active_tier]
                )
            rows = []
            row_sha256 = []
            joint_ids = []
            ht_strata = []
            weights = []
            representative_sides = []
            for component_index in active_indices:
                component = components[component_index]
                side, expected_row_sha256 = frame_v19a._representative_for_arm_v19a(
                    component_index, arm, frame, tiers
                )
                unit = frame["side_units"][side][component[f"{side}_unit"]]
                row = side_rows[side][unit["representative_index"]]
                if (
                    frame_v19a.frame_v18a.sampler_v13.row_sha256(row)
                    != expected_row_sha256
                    or not isinstance(row.get("question"), str)
                    or not row["question"]
                    or not isinstance(row.get("answer"), str)
                    or not row["answer"]
                ):
                    raise RuntimeError("v19a fixed representative content changed")
                if component_index in tiers["base_category"]:
                    stratum = tiers["base_category"][component_index]
                    weight = (
                        frame_v19a.BASE_CATEGORY_POPULATIONS_V19A[stratum]
                        / frame_v19a.BASE_CATEGORY_QUOTA_V19A
                    )
                else:
                    tier = tiers["candidate_only_layer"][component_index]
                    if tier != active_tier:
                        raise RuntimeError("v19a candidate-only tier leaked across arms")
                    stratum = f"candidate_only_tier_{tier}"
                    weight = float(
                        frame_v19a.CANDIDATE_ONLY_TIER_POPULATIONS_V19A[tier]
                    )
                rows.append(row)
                row_sha256.append(expected_row_sha256)
                joint_ids.append(component["joint_id"])
                ht_strata.append(stratum)
                weights.append(float(weight))
                representative_sides.append(side)
            certificate_arm = certificate_panels[panel]["arms"][arm]
            representative_root = frame_v19a.identity_root_sha256(
                f"{joint_id}:{side}:{row_digest}"
                for joint_id, side, row_digest in zip(
                    joint_ids, representative_sides, row_sha256
                )
            )
            if (
                len(rows) != frame_v19a.ARM_REQUESTS_PER_PANEL_V19A[arm]
                or len(joint_ids) != len(set(joint_ids))
                or joint_ids[:24]
                != [components[index]["joint_id"] for index in base_indices]
                or not math.isclose(
                    math.fsum(weights),
                    float(frame_v19a.ARM_POPULATIONS_V19A[arm]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or frame_v19a.identity_root_sha256(joint_ids)
                != certificate_arm["active_joint_component_identity_root_sha256"]
                or representative_root
                != certificate_arm["representative_assignment_root_sha256"]
                or certificate_arm["same_arm_paired_duplicate_count"] != 0
            ):
                raise RuntimeError("v19a materialized arm estimand changed")
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
        base_joint_ids = arms["production_only"]["joint_ids"]
        if any(arms[arm]["joint_ids"][:24] != base_joint_ids for arm in ARMS_V19A):
            raise RuntimeError("v19a paired base order differs across arms")
        panels[panel] = {
            "name": panel,
            "role": (
                "optimization"
                if panel in OPTIMIZATION_PANELS_V19A
                else "train_only_screen"
            ),
            "arms": arms,
        }
    result = {
        "schema": "eggroll-es-materialized-disjoint-tier-batches-v19a",
        "preregistration": {
            "file_sha256": PREREGISTRATION_FILE_SHA256_V19A,
            "content_sha256": PREREGISTRATION_CONTENT_SHA256_V19A,
            "sealed_commit": SEALED_PREREG_COMMIT_V19A,
        },
        "frame": {
            "file_sha256": FRAME_CERTIFICATE_FILE_SHA256_V19A,
            "content_sha256": FRAME_CERTIFICATE_CONTENT_SHA256_V19A,
            "sealed_commit": SEALED_PREREG_COMMIT_V19A,
        },
        "token_audit": {
            "content_sha256": prereg_v19a.TOKEN_AUDIT_CONTENT_SHA256_V19A,
            "frozen_total_token_cap": 1024,
            "over_cap_count": 0,
            "observed_combined_token_max": token_audit[
                "observed_combined_token_max"
            ],
        },
        "sources": {
            "candidate_v298": {
                "rows": frame_v19a.frame_v18a.CANDIDATE_ROWS_V18A,
                "file_sha256": frame_v19a.frame_v18a.CANDIDATE_SHA256_V18A,
            },
            "production": {
                "rows": frame_v19a.frame_v18a.PRODUCTION_ROWS_V18A,
                "file_sha256": frame_v19a.frame_v18a.PRODUCTION_SHA256_V18A,
            },
        },
        "panels": panels,
        "contains_train_question_answer_content_in_memory": True,
        "contains_evaluation_content": False,
        "persisted_to_disk": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def validate_panel_bundle_v19a(bundle: dict) -> dict:
    if (
        not isinstance(bundle, dict)
        or bundle.get("schema")
        != "eggroll-es-materialized-disjoint-tier-batches-v19a"
        or bundle.get("contains_train_question_answer_content_in_memory")
        is not True
        or bundle.get("contains_evaluation_content") is not False
        or bundle.get("persisted_to_disk") is not False
        or bundle.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(bundle))
        or bundle.get("content_sha256_before_self_field")
        != PANEL_BUNDLE_CONTENT_SHA256_V19A
        or tuple(bundle.get("panels", {})) != PANEL_NAMES_V19A
        or bundle.get("token_audit", {}).get("over_cap_count") != 0
        or bundle.get("token_audit", {}).get("observed_combined_token_max", 1025)
        > 1024
    ):
        raise RuntimeError("v19a materialized panel bundle changed")
    for panel_name in PANEL_NAMES_V19A:
        panel = bundle["panels"][panel_name]
        if set(panel.get("arms", {})) != set(ARMS_V19A):
            raise RuntimeError("v19a materialized arm coverage changed")
        production_base = panel["arms"]["production_only"]["joint_ids"]
        for arm in ARMS_V19A:
            batch = panel["arms"][arm]
            expected_rows = frame_v19a.ARM_REQUESTS_PER_PANEL_V19A[arm]
            sequence_fields = (
                "joint_ids",
                "row_sha256",
                "ht_strata",
                "weights",
                "representative_sides",
                "questions",
                "answers",
            )
            active_tier = frame_v19a.ARM_ACTIVE_TIER_V19A[arm]
            candidate_strata = [
                item
                for item in batch.get("ht_strata", [])
                if item.startswith("candidate_only_tier_")
            ]
            if (
                any(
                    len(batch.get(field, [])) != expected_rows
                    for field in sequence_fields
                )
                or len(set(batch["joint_ids"])) != expected_rows
                or batch["joint_ids"][:24] != production_base
                or canonical_sha256(batch["joint_ids"])
                != batch["ordered_joint_identity_sha256"]
                or canonical_sha256(batch["row_sha256"])
                != batch["ordered_row_identity_sha256"]
                or not math.isclose(
                    math.fsum(batch["weights"]),
                    float(frame_v19a.ARM_POPULATIONS_V19A[arm]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or candidate_strata
                != ([] if active_tier is None else [f"candidate_only_tier_{active_tier}"])
                or canonical_sha256({
                    "questions": batch["questions"],
                    "answers": batch["answers"],
                })
                != batch["raw_prompt_answer_sha256"]
            ):
                raise RuntimeError("v19a materialized panel batch changed")
    return bundle


def load_panel_bundle_v19a() -> dict:
    return validate_panel_bundle_v19a(_materialize_panel_bundle_v19a())


def _unit_score_arrays_v19a(unit_scores) -> dict[str, np.ndarray]:
    if not isinstance(unit_scores, dict) or set(unit_scores) != set(ARMS_V19A):
        raise RuntimeError("v19a per-unit score arm mapping changed")
    result = {}
    for arm in ARMS_V19A:
        values = np.asarray(unit_scores[arm], dtype=np.float64)
        expected = (
            10,
            2,
            32,
            frame_v19a.ARM_REQUESTS_PER_PANEL_V19A[arm],
        )
        if values.shape != expected or not np.isfinite(values).all():
            raise RuntimeError("v19a per-unit score tensor changed")
        result[arm] = values
    return result


def observed_panel_scores_v19a(unit_scores, panel_bundle) -> np.ndarray:
    values = _unit_score_arrays_v19a(unit_scores)
    panel_bundle = validate_panel_bundle_v19a(panel_bundle)
    panel_scores = np.empty((4, 10, 2, 32), dtype=np.float64)
    for arm_index, arm in enumerate(ARMS_V19A):
        weights = np.asarray([
            panel_bundle["panels"][panel]["arms"][arm]["weights"]
            for panel in PANEL_NAMES_V19A
        ], dtype=np.float64)
        panel_scores[arm_index] = np.einsum(
            "psdu,pu->psd", values[arm], weights
        ) / float(frame_v19a.ARM_POPULATIONS_V19A[arm])
    if not np.isfinite(panel_scores).all():
        raise RuntimeError("v19a HT panel-score recomputation changed")
    return panel_scores


def _cosine_last_axis_v19a(left, right):
    numerator = np.sum(left * right, axis=-1)
    denominator = np.sqrt(
        np.sum(left * left, axis=-1) * np.sum(right * right, axis=-1)
    )
    if np.any(denominator == 0.0):
        raise RuntimeError("v19a endpoint cosine received a zero direction")
    return numerator / denominator


def _sign_agreement_last_axis_v19a(left, right):
    return np.mean(np.sign(left) == np.sign(right), axis=-1)


def _endpoint_arrays_v19a(panel_scores):
    values = np.asarray(panel_scores, dtype=np.float64)
    if (
        values.ndim != 5
        or values.shape[0] != 4
        or values.shape[2:] != (10, 2, 32)
        or not np.isfinite(values).all()
    ):
        raise RuntimeError("v19a panel-score endpoint tensor changed")
    central = 0.5 * (values[:, :, :, 0, :] - values[:, :, :, 1, :])
    means = np.mean(central, axis=-1, keepdims=True)
    spreads = np.sqrt(np.mean((central - means) ** 2, axis=-1))
    if np.any(spreads == 0.0):
        raise RuntimeError("v19a panel response spread is zero")
    coefficients = (central - means) / (
        spreads[..., np.newaxis] + STANDARDIZATION_EPSILON_V19A
    )
    aggregate = np.median(coefficients[:, :, :6, :], axis=2)
    pairs = tuple(
        (left, right)
        for left in range(6)
        for right in range(left + 1, 6)
    )
    families = {
        "optimization_pairwise_cosine": np.stack([
            _cosine_last_axis_v19a(
                coefficients[:, :, left, :], coefficients[:, :, right, :]
            )
            for left, right in pairs
        ], axis=-1),
        "optimization_pairwise_sign_agreement": np.stack([
            _sign_agreement_last_axis_v19a(
                coefficients[:, :, left, :], coefficients[:, :, right, :]
            )
            for left, right in pairs
        ], axis=-1),
        "aggregate_to_optimization_cosine": np.stack([
            _cosine_last_axis_v19a(aggregate, coefficients[:, :, index, :])
            for index in range(6)
        ], axis=-1),
        "aggregate_to_optimization_sign_agreement": np.stack([
            _sign_agreement_last_axis_v19a(
                aggregate, coefficients[:, :, index, :]
            )
            for index in range(6)
        ], axis=-1),
        "train_screen_cosine": np.stack([
            _cosine_last_axis_v19a(aggregate, coefficients[:, :, index, :])
            for index in range(6, 10)
        ], axis=-1),
        "train_screen_sign_agreement": np.stack([
            _sign_agreement_last_axis_v19a(
                aggregate, coefficients[:, :, index, :]
            )
            for index in range(6, 10)
        ], axis=-1),
    }
    expected_family_widths = {
        "optimization_pairwise_cosine": 15,
        "optimization_pairwise_sign_agreement": 15,
        "aggregate_to_optimization_cosine": 6,
        "aggregate_to_optimization_sign_agreement": 6,
        "train_screen_cosine": 4,
        "train_screen_sign_agreement": 4,
    }
    if any(
        family.shape != values.shape[:2] + (expected_family_widths[name],)
        for name, family in families.items()
    ):
        raise RuntimeError("v19a endpoint family geometry changed")
    endpoints = {}
    for family in prereg_v19a.prereg_v18a.prereg_v17a.METRIC_FAMILIES_V17A:
        endpoints[f"{family}_median"] = np.median(families[family], axis=-1)
        endpoints[f"{family}_worst"] = np.min(families[family], axis=-1)
    if set(endpoints) != set(prereg_v19a.ENDPOINT_NAMES_V19A) or any(
        item.shape != values.shape[:2] or not np.isfinite(item).all()
        for item in endpoints.values()
    ):
        raise RuntimeError("v19a endpoint family recomputation changed")
    return {
        "endpoints": endpoints,
        "families": families,
        "coefficients": coefficients,
        "aggregate": aggregate,
        "spreads": spreads,
    }


def recompute_observed_endpoints_v19a(unit_scores, panel_bundle) -> dict:
    panel_scores = observed_panel_scores_v19a(unit_scores, panel_bundle)
    analyzed = _endpoint_arrays_v19a(panel_scores[:, np.newaxis, ...])
    result = {}
    for arm_index, arm in enumerate(ARMS_V19A):
        endpoint_values = {
            name: float(values[arm_index, 0])
            for name, values in analyzed["endpoints"].items()
        }
        compact_payload = {
            "schema": "eggroll-es-compact-disjoint-tier-estimator-v19a",
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


def _bootstrap_draw_plan_v19a() -> dict:
    rng = np.random.default_rng(prereg_v19a.BOOTSTRAP_SEED_V19A)
    base = np.empty(
        (10, len(BASE_CATEGORIES_V19A), BOOTSTRAP_REPETITIONS_V19A, 6),
        dtype=np.uint8,
    )
    for target in range(10):
        for category_index in range(len(BASE_CATEGORIES_V19A)):
            base[target, category_index] = rng.integers(
                0,
                6,
                size=(BOOTSTRAP_REPETITIONS_V19A, 6),
                dtype=np.uint8,
            )
    candidate_source_offsets = np.empty(
        (10, BOOTSTRAP_REPETITIONS_V19A), dtype=np.uint8
    )
    for role_indices in (np.arange(6), np.arange(6, 10)):
        for target in role_indices:
            candidate_source_offsets[target] = rng.integers(
                0,
                len(role_indices),
                size=BOOTSTRAP_REPETITIONS_V19A,
                dtype=np.uint8,
            )
    result = {
        "base": base,
        "candidate_source_offsets": candidate_source_offsets,
        "content_sha256": canonical_sha256({
            "schema": "eggroll-es-bootstrap-draw-plan-v19a",
            "seed": prereg_v19a.BOOTSTRAP_SEED_V19A,
            "base_sha256": hashlib.sha256(base.tobytes()).hexdigest(),
            "candidate_source_offsets_sha256": hashlib.sha256(
                candidate_source_offsets.tobytes()
            ).hexdigest(),
        }),
    }
    if result["content_sha256"] != BOOTSTRAP_DRAW_PLAN_CONTENT_SHA256_V19A:
        raise RuntimeError("v19a exact bootstrap draw plan changed")
    return result


def _bootstrap_positions_v19a(panel_bundle):
    base_positions = {
        arm: {
            panel_index: {
                category: np.asarray([
                    index
                    for index, value in enumerate(
                        panel_bundle["panels"][panel]["arms"][arm]["ht_strata"]
                    )
                    if value == category
                ], dtype=np.int64)
                for category in BASE_CATEGORIES_V19A
            }
            for panel_index, panel in enumerate(PANEL_NAMES_V19A)
        }
        for arm in ARMS_V19A
    }
    candidate_positions = {
        arm: {
            panel_index: next((
                index
                for index, value in enumerate(
                    panel_bundle["panels"][panel]["arms"][arm]["ht_strata"]
                )
                if value.startswith("candidate_only_tier_")
            ), None)
            for panel_index, panel in enumerate(PANEL_NAMES_V19A)
        }
        for arm in ARMS_V19A[1:]
    }
    if any(
        positions.shape != (6,)
        for arm in ARMS_V19A
        for panel in base_positions[arm].values()
        for positions in panel.values()
    ) or any(
        position is None
        for arm in ARMS_V19A[1:]
        for position in candidate_positions[arm].values()
    ):
        raise RuntimeError("v19a bootstrap stratum positions changed")
    return base_positions, candidate_positions


def _bootstrap_panel_scores_v19a(
    values,
    panel_bundle,
    draw_plan,
    start: int,
    stop: int,
) -> np.ndarray:
    batch_size = stop - start
    if not 0 <= start < stop <= BOOTSTRAP_REPETITIONS_V19A:
        raise ValueError("v19a bootstrap chunk bounds changed")
    result = np.zeros((4, batch_size, 10, 2, 32), dtype=np.float64)
    base_positions, candidate_positions = _bootstrap_positions_v19a(panel_bundle)
    for arm_index, arm in enumerate(ARMS_V19A):
        denominator = float(frame_v19a.ARM_POPULATIONS_V19A[arm])
        for target in range(10):
            target_scores = np.broadcast_to(
                values[arm][target],
                (batch_size,) + values[arm][target].shape,
            )
            for category_index, category in enumerate(BASE_CATEGORIES_V19A):
                positions = base_positions[arm][target][category]
                sampled_positions = positions[
                    draw_plan["base"][target, category_index, start:stop]
                ]
                sampled = np.take_along_axis(
                    target_scores,
                    sampled_positions[:, np.newaxis, np.newaxis, :],
                    axis=-1,
                )
                population = frame_v19a.BASE_CATEGORY_POPULATIONS_V19A[category]
                result[arm_index, :, target] += (
                    float(population) / denominator
                ) * np.mean(sampled, axis=-1)
            active_tier = frame_v19a.ARM_ACTIVE_TIER_V19A[arm]
            if active_tier is not None:
                role_indices = (
                    np.arange(6, dtype=np.int64)
                    if target < 6
                    else np.arange(6, 10, dtype=np.int64)
                )
                offsets = draw_plan["candidate_source_offsets"][target, start:stop]
                sources = role_indices[offsets]
                source_scores = values[arm][sources]
                positions = np.asarray([
                    candidate_positions[arm][int(source)] for source in sources
                ], dtype=np.int64)
                sampled = np.take_along_axis(
                    source_scores,
                    positions[:, np.newaxis, np.newaxis, np.newaxis],
                    axis=-1,
                )[..., 0]
                population = frame_v19a.CANDIDATE_ONLY_TIER_POPULATIONS_V19A[
                    active_tier
                ]
                result[arm_index, :, target] += (
                    float(population) / denominator
                ) * sampled
    return result


def paired_stratified_bootstrap_v19a(
    unit_scores,
    panel_bundle,
    *,
    chunk_size: int = BOOTSTRAP_DEFAULT_CHUNK_SIZE_V19A,
) -> dict:
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("v19a bootstrap chunk size must be positive")
    values = _unit_score_arrays_v19a(unit_scores)
    panel_bundle = validate_panel_bundle_v19a(panel_bundle)
    observed = recompute_observed_endpoints_v19a(values, panel_bundle)["arms"]
    draw_plan = _bootstrap_draw_plan_v19a()
    comparisons = {
        arm: {name: [] for name in prereg_v19a.ENDPOINT_NAMES_V19A}
        for arm in ARMS_V19A[1:]
    }
    completed = 0
    while completed < BOOTSTRAP_REPETITIONS_V19A:
        stop = min(BOOTSTRAP_REPETITIONS_V19A, completed + chunk_size)
        panel_scores = _bootstrap_panel_scores_v19a(
            values, panel_bundle, draw_plan, completed, stop
        )
        analyzed = _endpoint_arrays_v19a(panel_scores)
        for arm_index, arm in enumerate(ARMS_V19A[1:], 1):
            for name, endpoint_values in analyzed["endpoints"].items():
                comparisons[arm][name].append(
                    endpoint_values[arm_index] - endpoint_values[0]
                )
        completed = stop
    quantile = prereg_v19a.FAMILYWISE_ALPHA_V19A / 36
    output = {}
    for arm in ARMS_V19A[1:]:
        endpoints = {}
        for name, chunks in comparisons[arm].items():
            deltas = np.concatenate(chunks)
            if (
                deltas.shape != (BOOTSTRAP_REPETITIONS_V19A,)
                or not np.isfinite(deltas).all()
            ):
                raise RuntimeError("v19a bootstrap replicate coverage changed")
            observed_delta = (
                observed[arm]["endpoint_values"][name]
                - observed["production_only"]["endpoint_values"][name]
            )
            endpoints[name] = {
                "patch_minus_production": float(observed_delta),
                "familywise_lcb": float(np.quantile(
                    deltas,
                    quantile,
                    method=BOOTSTRAP_QUANTILE_METHOD_V19A,
                )),
                "noninferiority_margin": 0.0,
            }
        output[arm] = endpoints
    return {
        "seed": prereg_v19a.BOOTSTRAP_SEED_V19A,
        "repetitions": BOOTSTRAP_REPETITIONS_V19A,
        "one_sided_quantile": quantile,
        "draw_plan_content_sha256": draw_plan["content_sha256"],
        "whole_panel_block_resampling_used": False,
        "comparisons": output,
    }


def evaluate_attribution_gate_v19a(summary: dict) -> dict:
    arms = summary.get("arms", {})
    bootstrap = summary.get("paired_bootstrap", {})
    integrity = summary.get("runtime_integrity", {})
    if (
        set(arms) != set(ARMS_V19A)
        or set(bootstrap.get("comparisons", {})) != set(ARMS_V19A[1:])
        or bootstrap.get("repetitions") != BOOTSTRAP_REPETITIONS_V19A
        or bootstrap.get("one_sided_quantile")
        != prereg_v19a.FAMILYWISE_ALPHA_V19A / 36
        or bootstrap.get("whole_panel_block_resampling_used") is not False
    ):
        raise RuntimeError("v19a gate input coverage changed")
    arm_results = {}
    for arm in ARMS_V19A[1:]:
        endpoints = bootstrap["comparisons"][arm]
        if set(endpoints) != set(prereg_v19a.ENDPOINT_NAMES_V19A):
            raise RuntimeError("v19a gate endpoint coverage changed")
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
            "preregistered_attribution_gate_passed": bool(passed),
        }
    passing = [
        arm
        for arm in ARMS_V19A[1:]
        if arm_results[arm]["preregistered_attribution_gate_passed"]
    ]
    return {
        "schema": "eggroll-es-disjoint-tier-attribution-gate-v19a",
        "arms": arm_results,
        "passing_tiers": passing,
        "any_tier_passed": bool(passing),
        "decision": (
            "authorize_only_separate_fresh_basis_train_only_confirmation_preregistration"
            if passing
            else "retain_production_dataset_and_v13_recipe"
        ),
        "dataset_promotion_authorized": False,
        "model_update_authorized": False,
        "evaluation_authorized": False,
    }


def build_compact_estimator_summary_v19a(unit_scores, panel_bundle) -> dict:
    observed = recompute_observed_endpoints_v19a(unit_scores, panel_bundle)
    value = {
        "schema": "eggroll-es-disjoint-tier-compact-estimator-summary-v19a",
        "arms": copy.deepcopy(observed["arms"]),
        "paired_bootstrap": paired_stratified_bootstrap_v19a(
            unit_scores, panel_bundle
        ),
        "runtime_integrity_required_before_gate": True,
        "persisted_response_vectors_or_row_content": False,
        "bootstrap_draws_persisted": False,
        "unit_scores_persisted": False,
        "dataset_promotion_authorized": False,
        "model_update_authorized": False,
        "evaluation_authorized": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value
