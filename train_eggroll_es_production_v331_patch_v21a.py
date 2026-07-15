#!/usr/bin/env python3
"""Pure train-only estimator mechanics for V21A production-v331 compatibility."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

import build_eggroll_es_production_v331_patch_draw_plan_v21a as draw_v21a
import build_eggroll_es_production_v331_patch_frame_v21a as frame_v21a
import eggroll_es_production_v331_patch_preregistration_v21a as prereg_v21a


ROOT = Path(__file__).resolve().parent
ARMS_V21A = frame_v21a.ARM_ORDER_V21A
PANEL_NAMES_V21A = frame_v21a.PANEL_NAMES_V21A
OPTIMIZATION_PANELS_V21A = frame_v21a.OPTIMIZATION_PANELS_V21A
TRAIN_SCREEN_PANELS_V21A = frame_v21a.TRAIN_SCREEN_PANELS_V21A
BASE_CATEGORIES_V21A = frame_v21a.BASE_CATEGORIES_V21A
ENDPOINT_NAMES_V21A = prereg_v21a.ENDPOINT_NAMES_V21A
SIGNS_V21A = ("plus", "minus")

DRAW_PLAN_COMMIT_V21A = "5b7a5ff0f4180a7c756345352550464de5e08f2b"
DRAW_BUILDER_SHA256_V21A = (
    "8316d6c10eb9df375838c51ae1e138f665a4b2a1c2b172f1df9a9ef2b5002819"
)
DRAW_TEST_SHA256_V21A = (
    "298baa65bb0c1b9e5797c63247f6e0f057d45e2dfccef34aefcee9d6e5b23c01"
)
DRAW_FILE_SHA256_V21A = (
    "8400cc0b53e1ae0fbbd903eced511fa643441e4c0fa393ab0677743ff778779b"
)
DRAW_CONTENT_SHA256_V21A = (
    "b7964861ccd092f2c8c1177de5ff041f561e9a8f3a4d8b65bb2e4deb9aa7e820"
)
PANEL_BUNDLE_CONTENT_SHA256_V21A = (
    "b565a3fee777d5a42414dd2f6c2a100e89f38c83c699e221fda7e6728875ad7e"
)
BOOTSTRAP_REPETITIONS_V21A = 50_000
BOOTSTRAP_DEFAULT_CHUNK_SIZE_V21A = 128
BOOTSTRAP_QUANTILE_METHOD_V21A = "linear"
STANDARDIZATION_EPSILON_V21A = 1e-8
PANEL_ORDER_SEED_V21A = 20260823

canonical_sha256 = prereg_v21a.canonical_sha256
file_sha256 = prereg_v21a.file_sha256


def _without_self(value: dict) -> dict:
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _verify_commit_file(path: Path, digest: str) -> None:
    path = Path(path).resolve()
    relative = path.relative_to(ROOT).as_posix()
    raw = subprocess.check_output(
        ["git", "show", f"{DRAW_PLAN_COMMIT_V21A}:{relative}"], cwd=ROOT
    )
    if hashlib.sha256(raw).hexdigest() != digest or file_sha256(path) != digest:
        raise RuntimeError(f"v21a mechanics draw artifact changed: {relative}")


def load_hardened_foundation_v21a() -> tuple[dict, dict, dict]:
    for path, digest in (
        (Path(draw_v21a.__file__), DRAW_BUILDER_SHA256_V21A),
        (
            ROOT / "test_build_eggroll_es_production_v331_patch_draw_plan_v21a.py",
            DRAW_TEST_SHA256_V21A,
        ),
        (draw_v21a.OUTPUT_PATH_V21A, DRAW_FILE_SHA256_V21A),
    ):
        _verify_commit_file(path, digest)
    frame, _evidence, _layer = prereg_v21a.load_bound_inputs_v21a()
    preregistration = json.loads(prereg_v21a.OUTPUT_PATH_V21A.read_text())
    prereg_v21a.validate_preregistration_v21a(preregistration)
    draw = json.loads(draw_v21a.OUTPUT_PATH_V21A.read_text())
    draw_v21a.validate_draw_plan_certificate_v21a(draw)
    if (
        frame["content_sha256_before_self_field"]
        != prereg_v21a.FRAME_CERTIFICATE_CONTENT_SHA256_V21A
        or preregistration["content_sha256_before_self_field"]
        != draw_v21a.PREREG_CONTENT_SHA256_V21A
        or draw["content_sha256_before_self_field"]
        != DRAW_CONTENT_SHA256_V21A
    ):
        raise RuntimeError("v21a hardened mechanics foundation changed")
    return frame, preregistration, draw


def _panel_order_v21a(panel: str, component_indices: list[int], components):
    return sorted(
        component_indices,
        key=lambda index: canonical_sha256({
            "schema": "eggroll-es-runtime-panel-order-v21a",
            "seed": PANEL_ORDER_SEED_V21A,
            "panel": panel,
            "joint_id": components[index]["joint_id"],
        }),
    )


def _materialize_panel_bundle_v21a() -> dict:
    persisted_frame, preregistration, _draw = load_hardened_foundation_v21a()
    candidate, production = frame_v21a.load_bound_rows_v21a()
    frame = frame_v21a.build_joint_frame_v21a(candidate, production)
    frozen_frame, frozen_tiers, base_flow = frame_v21a._base_flow_v21a()
    candidate_assignment = frame_v21a.assign_candidate_only_v21a(frame)
    if frame_v21a.build_certificate_v21a() != persisted_frame:
        raise RuntimeError("v21a reconstructed frame differs from certificate")

    components = frame["joint_components"]
    candidate_units = frame["side_units"]["candidate"]
    production_units = frame["side_units"]["production"]
    frozen_components = frozen_frame["joint_components"]
    frozen_production_units = frozen_frame["side_units"]["production"]
    new_by_production_unit_id = {
        production_units[item["production_unit"]]["unit_id"]: index
        for index, item in enumerate(components)
        if item["production_unit"] is not None
    }
    side_rows = {"candidate": candidate, "production": production}
    certificate_panels = {
        panel["name"]: panel for panel in persisted_frame["panels"]
    }
    panels = {}
    for panel in PANEL_NAMES_V21A:
        frozen_indices = base_flow["selected_base"][panel]
        category_by_new_index = {}
        base_indices = []
        for frozen_index in frozen_indices:
            production_index = frozen_components[frozen_index]["production_unit"]
            unit_id = frozen_production_units[production_index]["unit_id"]
            new_index = new_by_production_unit_id[unit_id]
            base_indices.append(new_index)
            category_by_new_index[new_index] = frozen_tiers["base_category"][
                frozen_index
            ]
        base_indices = _panel_order_v21a(panel, base_indices, components)
        candidate_indices = _panel_order_v21a(
            panel, candidate_assignment["assignments"][panel], components
        )
        arms = {}
        for arm in ARMS_V21A:
            active_indices = list(base_indices)
            if arm == "production_plus_v331_patch":
                active_indices.extend(candidate_indices)
            rows = []
            row_sha256 = []
            joint_ids = []
            ht_strata = []
            weights = []
            representative_sides = []
            for component_index in active_indices:
                component = components[component_index]
                if component["relation"] == "candidate_only":
                    side = "candidate"
                    unit = candidate_units[component["candidate_unit"]]
                    topic = unit["stratum"]
                    stratum = f"candidate_only_topic:{topic}"
                    weight = (
                        frame_v21a.CANDIDATE_ONLY_TOPIC_POPULATIONS_V21A[topic]
                        / frame_v21a.CANDIDATE_ONLY_TOPIC_QUOTAS_V21A[topic]
                    )
                else:
                    side = "production"
                    unit = production_units[component["production_unit"]]
                    stratum = category_by_new_index[component_index]
                    weight = (
                        frame_v21a.BASE_CATEGORY_POPULATIONS_V21A[stratum]
                        / frame_v21a.BASE_CATEGORY_QUOTA_V21A
                    )
                row = side_rows[side][unit["representative_index"]]
                digest = frame_v21a.frame_v18a.sampler_v13.row_sha256(row)
                if (
                    digest != unit["representative_row_sha256"]
                    or not isinstance(row.get("question"), str)
                    or not row["question"]
                    or not isinstance(row.get("answer"), str)
                    or not row["answer"]
                ):
                    raise RuntimeError("v21a fixed representative content changed")
                rows.append(row)
                row_sha256.append(digest)
                joint_ids.append(component["joint_id"])
                ht_strata.append(stratum)
                weights.append(float(weight))
                representative_sides.append(side)
            certificate_arm = certificate_panels[panel]["arms"][arm]
            representative_root = frame_v21a.identity_root_sha256(
                f"{joint_id}:{side}:{row_digest}"
                for joint_id, side, row_digest in zip(
                    joint_ids, representative_sides, row_sha256
                )
            )
            if (
                len(rows) != frame_v21a.ARM_REQUESTS_PER_PANEL_V21A[arm]
                or len(joint_ids) != len(set(joint_ids))
                or not math.isclose(
                    math.fsum(weights),
                    float(frame_v21a.ARM_POPULATIONS_V21A[arm]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or frame_v21a.identity_root_sha256(joint_ids)
                != certificate_arm["active_joint_component_identity_root_sha256"]
                or representative_root
                != certificate_arm["representative_assignment_root_sha256"]
                or certificate_arm["paired_candidate_replacements"] != 0
                or certificate_arm["same_arm_component_duplicate_count"] != 0
            ):
                raise RuntimeError("v21a materialized arm estimand changed")
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
        if (
            arms["production_plus_v331_patch"]["joint_ids"][:24]
            != arms["production_only"]["joint_ids"]
            or arms["production_plus_v331_patch"]["row_sha256"][:24]
            != arms["production_only"]["row_sha256"]
            or arms["production_plus_v331_patch"]["representative_sides"]
            != ["production"] * 24 + ["candidate"] * 6
        ):
            raise RuntimeError("v21a full merge changed or replaced production")
        panels[panel] = {
            "name": panel,
            "role": (
                "optimization"
                if panel in OPTIMIZATION_PANELS_V21A
                else "train_only_screen"
            ),
            "arms": arms,
        }
    result = {
        "schema": "eggroll-es-materialized-production-v331-patch-batches-v21a",
        "preregistration": {
            "commit": draw_v21a.PREREG_COMMIT_V21A,
            "file_sha256": draw_v21a.PREREG_FILE_SHA256_V21A,
            "content_sha256": draw_v21a.PREREG_CONTENT_SHA256_V21A,
        },
        "frame": {
            "commit": prereg_v21a.FRAME_COMMIT_V21A,
            "file_sha256": prereg_v21a.FRAME_CERTIFICATE_SHA256_V21A,
            "content_sha256": prereg_v21a.FRAME_CERTIFICATE_CONTENT_SHA256_V21A,
        },
        "draw_plan": {
            "commit": DRAW_PLAN_COMMIT_V21A,
            "file_sha256": DRAW_FILE_SHA256_V21A,
            "content_sha256": DRAW_CONTENT_SHA256_V21A,
        },
        "sources": {
            "candidate_v331": {
                "source_commit": frame_v21a.candidate_v331.V331_SOURCE_COMMIT,
                "rows": 527,
                "file_sha256": frame_v21a.candidate_v331.V331_SHA256,
            },
            "production": {
                "rows": 784,
                "file_sha256": frame_v21a.PRODUCTION_SHA256_V21A,
            },
        },
        "panels": panels,
        "token_boundary_audit_required_before_any_scoring": preregistration[
            "scoring"
        ]["v331_token_boundary_audit_required_before_any_future_runtime"],
        "contains_train_question_answer_content_in_memory": True,
        "contains_evaluation_content": False,
        "persisted_to_disk": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def validate_panel_bundle_v21a(bundle: dict) -> dict:
    if (
        not isinstance(bundle, dict)
        or bundle.get("schema")
        != "eggroll-es-materialized-production-v331-patch-batches-v21a"
        or bundle.get("contains_train_question_answer_content_in_memory") is not True
        or bundle.get("contains_evaluation_content") is not False
        or bundle.get("persisted_to_disk") is not False
        or bundle.get("token_boundary_audit_required_before_any_scoring") is not True
        or tuple(bundle.get("panels", {})) != PANEL_NAMES_V21A
    ):
        raise RuntimeError("v21a materialized panel bundle changed")
    for panel_name in PANEL_NAMES_V21A:
        panel = bundle["panels"][panel_name]
        if set(panel.get("arms", {})) != set(ARMS_V21A):
            raise RuntimeError("v21a materialized arm coverage changed")
        production = panel["arms"]["production_only"]
        merged = panel["arms"]["production_plus_v331_patch"]
        if (
            merged.get("joint_ids", [])[:24] != production.get("joint_ids")
            or merged.get("row_sha256", [])[:24] != production.get("row_sha256")
        ):
            raise RuntimeError("v21a paired production base changed")
        for arm in ARMS_V21A:
            batch = panel["arms"][arm]
            expected = frame_v21a.ARM_REQUESTS_PER_PANEL_V21A[arm]
            sequence_fields = (
                "joint_ids", "row_sha256", "ht_strata", "weights",
                "representative_sides", "questions", "answers",
            )
            candidate_strata = [
                item for item in batch.get("ht_strata", [])
                if item.startswith("candidate_only_topic:")
            ]
            if (
                any(len(batch.get(field, [])) != expected for field in sequence_fields)
                or len(set(batch["joint_ids"])) != expected
                or canonical_sha256(batch["joint_ids"])
                != batch["ordered_joint_identity_sha256"]
                or canonical_sha256(batch["row_sha256"])
                != batch["ordered_row_identity_sha256"]
                or not math.isclose(
                    math.fsum(batch["weights"]),
                    float(frame_v21a.ARM_POPULATIONS_V21A[arm]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or len(candidate_strata) != (0 if arm == "production_only" else 6)
                or canonical_sha256({
                    "questions": batch["questions"],
                    "answers": batch["answers"],
                }) != batch["raw_prompt_answer_sha256"]
            ):
                raise RuntimeError("v21a materialized panel batch changed")
    if (
        bundle.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(bundle))
        or bundle.get("content_sha256_before_self_field")
        != PANEL_BUNDLE_CONTENT_SHA256_V21A
    ):
        raise RuntimeError("v21a materialized panel bundle changed")
    return bundle


def load_panel_bundle_v21a() -> dict:
    return validate_panel_bundle_v21a(_materialize_panel_bundle_v21a())


def _unit_score_arrays_v21a(unit_scores) -> dict[str, np.ndarray]:
    if not isinstance(unit_scores, dict) or set(unit_scores) != set(ARMS_V21A):
        raise RuntimeError("v21a per-unit score arm mapping changed")
    result = {}
    for arm in ARMS_V21A:
        values = np.asarray(unit_scores[arm], dtype=np.float64)
        expected = (
            10, 2, 64, frame_v21a.ARM_REQUESTS_PER_PANEL_V21A[arm]
        )
        if values.shape != expected or not np.isfinite(values).all():
            raise RuntimeError("v21a per-unit score tensor changed")
        result[arm] = values
    return result


def observed_panel_scores_v21a(unit_scores, panel_bundle) -> np.ndarray:
    values = _unit_score_arrays_v21a(unit_scores)
    panel_bundle = validate_panel_bundle_v21a(panel_bundle)
    panel_scores = np.empty((2, 10, 2, 64), dtype=np.float64)
    for arm_index, arm in enumerate(ARMS_V21A):
        weights = np.asarray([
            panel_bundle["panels"][panel]["arms"][arm]["weights"]
            for panel in PANEL_NAMES_V21A
        ], dtype=np.float64)
        panel_scores[arm_index] = np.einsum(
            "psdu,pu->psd", values[arm], weights
        ) / float(frame_v21a.ARM_POPULATIONS_V21A[arm])
    if not np.isfinite(panel_scores).all():
        raise RuntimeError("v21a HT panel-score recomputation changed")
    return panel_scores


def _cosine_last_axis(left, right):
    numerator = np.sum(left * right, axis=-1)
    denominator = np.sqrt(
        np.sum(left * left, axis=-1) * np.sum(right * right, axis=-1)
    )
    if np.any(denominator == 0.0):
        raise RuntimeError("v21a endpoint cosine received a zero direction")
    return numerator / denominator


def _sign_agreement_last_axis(left, right):
    return np.mean(np.sign(left) == np.sign(right), axis=-1)


def _endpoint_arrays_v21a(panel_scores):
    values = np.asarray(panel_scores, dtype=np.float64)
    if (
        values.ndim != 5
        or values.shape[0] != 2
        or values.shape[2:] != (10, 2, 64)
        or not np.isfinite(values).all()
    ):
        raise RuntimeError("v21a panel-score endpoint tensor changed")
    central = 0.5 * (values[:, :, :, 0, :] - values[:, :, :, 1, :])
    means = np.mean(central, axis=-1, keepdims=True)
    spreads = np.sqrt(np.mean((central - means) ** 2, axis=-1))
    if np.any(spreads == 0.0):
        raise RuntimeError("v21a panel response spread is zero")
    coefficients = (central - means) / (
        spreads[..., np.newaxis] + STANDARDIZATION_EPSILON_V21A
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
        raise RuntimeError("v21a endpoint family geometry changed")
    endpoints = {}
    for family in widths:
        endpoints[f"{family}_median"] = np.median(families[family], axis=-1)
        endpoints[f"{family}_worst"] = np.min(families[family], axis=-1)
    if tuple(endpoints) != ENDPOINT_NAMES_V21A or any(
        item.shape != values.shape[:2] or not np.isfinite(item).all()
        for item in endpoints.values()
    ):
        raise RuntimeError("v21a endpoint family recomputation changed")
    return {
        "endpoints": endpoints,
        "families": families,
        "coefficients": coefficients,
        "aggregate": aggregate,
        "spreads": spreads,
    }


def recompute_observed_endpoints_v21a(unit_scores, panel_bundle) -> dict:
    panel_scores = observed_panel_scores_v21a(unit_scores, panel_bundle)
    analyzed = _endpoint_arrays_v21a(panel_scores[:, np.newaxis, ...])
    result = {}
    for arm_index, arm in enumerate(ARMS_V21A):
        endpoint_values = {
            name: float(values[arm_index, 0])
            for name, values in analyzed["endpoints"].items()
        }
        compact_payload = {
            "schema": "eggroll-es-compact-production-v331-estimator-v21a",
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


def _bootstrap_draw_plan_v21a() -> dict:
    certificate = json.loads(draw_v21a.OUTPUT_PATH_V21A.read_text())
    draw_v21a.validate_draw_plan_certificate_v21a(certificate)
    arrays = draw_v21a.materialize_draw_arrays_v21a()
    if (
        hashlib.sha256(arrays["base"].tobytes()).hexdigest()
        != certificate["base_draws"]["bytes_sha256"]
        or any(
            hashlib.sha256(arrays["candidate"][role][topic].tobytes()).hexdigest()
            != certificate["candidate_only_draws"][role][topic]["bytes_sha256"]
            for role in draw_v21a.ROLE_PANEL_COUNTS_V21A
            for topic in frame_v21a.CANDIDATE_ONLY_TOPIC_QUOTAS_V21A
        )
    ):
        raise RuntimeError("v21a exact bootstrap draw arrays changed")
    return {**arrays, "content_sha256": certificate["content_sha256_before_self_field"]}


def _stratum_positions_v21a(panel_bundle):
    base = {
        arm: {
            panel_index: {
                category: np.asarray([
                    index for index, value in enumerate(
                        panel_bundle["panels"][panel]["arms"][arm]["ht_strata"]
                    ) if value == category
                ], dtype=np.int64)
                for category in BASE_CATEGORIES_V21A
            }
            for panel_index, panel in enumerate(PANEL_NAMES_V21A)
        }
        for arm in ARMS_V21A
    }
    candidate_pools = {}
    merged = "production_plus_v331_patch"
    for role, panels in (
        ("optimization", OPTIMIZATION_PANELS_V21A),
        ("train_only_screen", TRAIN_SCREEN_PANELS_V21A),
    ):
        candidate_pools[role] = {}
        for topic in frame_v21a.CANDIDATE_ONLY_TOPIC_QUOTAS_V21A:
            marker = f"candidate_only_topic:{topic}"
            candidate_pools[role][topic] = [
                (PANEL_NAMES_V21A.index(panel), position)
                for panel in panels
                for position, value in enumerate(
                    panel_bundle["panels"][panel]["arms"][merged]["ht_strata"]
                )
                if value == marker
            ]
    if any(
        positions.shape != (6,)
        for arm in ARMS_V21A
        for panel in base[arm].values()
        for positions in panel.values()
    ) or any(
        len(candidate_pools[role][topic])
        != draw_v21a.CANDIDATE_SOURCE_SLOT_COUNTS_V21A[role][topic]
        for role in candidate_pools for topic in candidate_pools[role]
    ):
        raise RuntimeError("v21a bootstrap stratum positions changed")
    return base, candidate_pools


def _bootstrap_panel_scores_v21a(
    values, panel_bundle, draw_plan, start, stop,
):
    batch_size = stop - start
    if not 0 <= start < stop <= BOOTSTRAP_REPETITIONS_V21A:
        raise ValueError("v21a bootstrap chunk bounds changed")
    result = np.zeros((2, batch_size, 10, 2, 64), dtype=np.float64)
    base_positions, candidate_pools = _stratum_positions_v21a(panel_bundle)
    for arm_index, arm in enumerate(ARMS_V21A):
        denominator = float(frame_v21a.ARM_POPULATIONS_V21A[arm])
        for target in range(10):
            target_scores = np.broadcast_to(
                values[arm][target],
                (batch_size,) + values[arm][target].shape,
            )
            for category_index, category in enumerate(BASE_CATEGORIES_V21A):
                positions = base_positions[arm][target][category]
                sampled_positions = positions[
                    draw_plan["base"][target, category_index, start:stop]
                ]
                sampled = np.take_along_axis(
                    target_scores,
                    sampled_positions[:, np.newaxis, np.newaxis, :],
                    axis=-1,
                )
                population = frame_v21a.BASE_CATEGORY_POPULATIONS_V21A[category]
                result[arm_index, :, target] += (
                    float(population) / denominator
                ) * np.mean(sampled, axis=-1)
    merged_index = ARMS_V21A.index("production_plus_v331_patch")
    merged_values = values["production_plus_v331_patch"]
    for target, panel in enumerate(PANEL_NAMES_V21A):
        role = "optimization" if panel in OPTIMIZATION_PANELS_V21A else "train_only_screen"
        local_target = (
            OPTIMIZATION_PANELS_V21A.index(panel)
            if role == "optimization" else TRAIN_SCREEN_PANELS_V21A.index(panel)
        )
        denominator = float(frame_v21a.ARM_POPULATIONS_V21A[
            "production_plus_v331_patch"
        ])
        for topic in frame_v21a.CANDIDATE_ONLY_TOPIC_QUOTAS_V21A:
            pool = candidate_pools[role][topic]
            source = np.stack([
                merged_values[panel_index, :, :, position]
                for panel_index, position in pool
            ], axis=0)
            offsets = draw_plan["candidate"][role][topic][
                local_target, start:stop
            ]
            sampled = source[offsets]
            population = frame_v21a.CANDIDATE_ONLY_TOPIC_POPULATIONS_V21A[topic]
            result[merged_index, :, target] += (
                float(population) / denominator
            ) * np.mean(sampled, axis=1)
    return result


def paired_stratified_bootstrap_v21a(
    unit_scores,
    panel_bundle,
    *,
    chunk_size=BOOTSTRAP_DEFAULT_CHUNK_SIZE_V21A,
):
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("v21a bootstrap chunk size must be positive")
    values = _unit_score_arrays_v21a(unit_scores)
    panel_bundle = validate_panel_bundle_v21a(panel_bundle)
    observed = recompute_observed_endpoints_v21a(values, panel_bundle)["arms"]
    draw_plan = _bootstrap_draw_plan_v21a()
    comparisons = {name: [] for name in ENDPOINT_NAMES_V21A}
    treatment = ARMS_V21A.index("production_plus_v331_patch")
    control = ARMS_V21A.index("production_only")
    completed = 0
    while completed < BOOTSTRAP_REPETITIONS_V21A:
        stop = min(BOOTSTRAP_REPETITIONS_V21A, completed + chunk_size)
        panel_scores = _bootstrap_panel_scores_v21a(
            values, panel_bundle, draw_plan, completed, stop
        )
        analyzed = _endpoint_arrays_v21a(panel_scores)
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
            raise RuntimeError("v21a bootstrap replicate coverage changed")
        observed_delta = (
            observed["production_plus_v331_patch"]["endpoint_values"][name]
            - observed["production_only"]["endpoint_values"][name]
        )
        endpoints[name] = {
            "treatment_minus_control": float(observed_delta),
            "familywise_lcb": float(np.quantile(
                deltas, quantile, method=BOOTSTRAP_QUANTILE_METHOD_V21A
            )),
            "noninferiority_margin": 0.0,
        }
    return {
        "seed": prereg_v21a.BOOTSTRAP_SEED_V21A,
        "repetitions": 50_000,
        "one_sided_quantile": quantile,
        "quantile_method": BOOTSTRAP_QUANTILE_METHOD_V21A,
        "draw_plan_content_sha256": draw_plan["content_sha256"],
        "paired_same_draws_both_arms": True,
        "whole_panel_block_resampling_used": False,
        "comparison": {
            "name": "production_plus_v331_patch_vs_production",
            "treatment": "production_plus_v331_patch",
            "control": "production_only",
            "endpoints": endpoints,
        },
    }


def evaluate_compatibility_gate_v21a(summary):
    arms = summary.get("arms", {})
    bootstrap = summary.get("paired_bootstrap", {})
    comparison = bootstrap.get("comparison", {})
    endpoints = comparison.get("endpoints", {})
    integrity = summary.get("runtime_integrity", {})
    if (
        set(arms) != set(ARMS_V21A)
        or set(endpoints) != set(ENDPOINT_NAMES_V21A)
        or bootstrap.get("repetitions") != 50_000
        or bootstrap.get("one_sided_quantile") != 0.05 / 12
        or bootstrap.get("paired_same_draws_both_arms") is not True
        or bootstrap.get("whole_panel_block_resampling_used") is not False
        or comparison.get("treatment") != "production_plus_v331_patch"
        or comparison.get("control") != "production_only"
    ):
        raise RuntimeError("v21a gate input coverage changed")
    observed = {
        name: item["treatment_minus_control"] >= 0.0
        for name, item in endpoints.items()
    }
    lower = {
        name: item["familywise_lcb"] >= 0.0
        for name, item in endpoints.items()
    }
    spreads = all(
        arms[arm].get("all_panel_spreads_nonzero") is True for arm in ARMS_V21A
    )
    passed = (
        all(observed.values())
        and all(lower.values())
        and spreads
        and integrity.get("all_integrity_audits_passed") is True
    )
    return {
        "schema": "eggroll-es-production-v331-patch-compatibility-gate-v21a",
        "observed_pass_count": sum(observed.values()),
        "bootstrap_pass_count": sum(lower.values()),
        "all_twelve_observed_passed": all(observed.values()),
        "all_twelve_bootstrap_passed": all(lower.values()),
        "all_panel_spreads_nonzero": spreads,
        "compatibility_gate_passed": bool(passed),
        "decision": (
            "authorize_only_separate_fresh_basis_train_only_confirmation_preregistration"
            if passed else "retain_production_dataset_and_v13_recipe"
        ),
        "dataset_promotion_authorized": False,
        "model_update_authorized": False,
        "evaluation_authorized": False,
    }


def build_compact_estimator_summary_v21a(unit_scores, panel_bundle):
    observed = recompute_observed_endpoints_v21a(unit_scores, panel_bundle)
    value = {
        "schema": "eggroll-es-production-v331-patch-compact-estimator-v21a",
        "arms": copy.deepcopy(observed["arms"]),
        "paired_bootstrap": paired_stratified_bootstrap_v21a(
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
