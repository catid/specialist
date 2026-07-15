#!/usr/bin/env python3
"""Pure train-only mechanics for V20A nested tier interaction attribution."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

import build_eggroll_es_nested_tier_draw_plan_v20a as draw_v20a
import build_eggroll_es_nested_tier_interaction_frame_v20a as frame_v20a
import eggroll_es_nested_tier_interaction_preregistration_v20a as prereg_v20a
import train_eggroll_es_disjoint_tier_attribution_v19a as mechanics_v19a


ROOT = Path(__file__).resolve().parent
ARMS_V20A = prereg_v20a.ARM_ORDER_V20A
CONTRASTS_V20A = prereg_v20a.CONTRASTS_V20A
PANEL_NAMES_V20A = frame_v20a.PANEL_NAMES_V20A
OPTIMIZATION_PANELS_V20A = frame_v20a.OPTIMIZATION_PANELS_V20A
TRAIN_SCREEN_PANELS_V20A = frame_v20a.TRAIN_SCREEN_PANELS_V20A
BASE_CATEGORIES_V20A = frame_v20a.BASE_CATEGORIES_V20A
SIGNS_V20A = ("plus", "minus")

SEALED_FOUNDATION_COMMIT_V20A = "f8860e14c693020badf25985cb2ba6b4d4339e30"
FRAME_BUILDER_FILE_SHA256_V20A = (
    "f1fc7dcea9780c27bd490caab4821f0ec8339ad748bdedb3cb2609e404743a76"
)
FRAME_CERTIFICATE_FILE_SHA256_V20A = (
    "cd2cb7134716602f1d51fe4049c95c49c33ddee429dcc6e752e2cb0b4196f444"
)
FRAME_CERTIFICATE_CONTENT_SHA256_V20A = (
    "1199f2bcb4cf3c6f394c2faf2edf14cb9ab74323c47614118511d768197c8078"
)
PREREGISTRATION_BUILDER_FILE_SHA256_V20A = (
    "53d8d1186c5164ad1f787d4eb8966aa0b759d2ce3d9f6f371108419475dadde5"
)
PREREGISTRATION_FILE_SHA256_V20A = (
    "1a7f372bf6f2af6606acc4e6d4adbf9815b96931a532aa07348335a1416d6963"
)
PREREGISTRATION_CONTENT_SHA256_V20A = (
    "9ce64a2d9cc91da2dd83aeb0d1e5adf3c0a3216e69613e6a61683c001909345e"
)
DRAW_PLAN_BUILDER_FILE_SHA256_V20A = (
    "ce6fd6aa16fb2477dd27ad402b4c72af2e7f01589c5bc4637e833a2dabb50c70"
)
DRAW_PLAN_FILE_SHA256_V20A = (
    "9032aa332f9b143d5f00dde7a522b5085163c6727a0c13bf8670664c24a99fff"
)
DRAW_PLAN_CONTENT_SHA256_V20A = (
    "2eb5de70d60be3178ea8f27ffcf7a54293fdfa3c4ed412cfb0f0093e7c5fae28"
)
PANEL_BUNDLE_CONTENT_SHA256_V20A = (
    "bbf1d592799ba30e4506c4e5abe9851bc3673e619ed29f65d6054c8b150681bd"
)

BOOTSTRAP_REPETITIONS_V20A = 50_000
BOOTSTRAP_DEFAULT_CHUNK_SIZE_V20A = 128
BOOTSTRAP_QUANTILE_METHOD_V20A = "linear"
STANDARDIZATION_EPSILON_V20A = mechanics_v19a.STANDARDIZATION_EPSILON_V19A

canonical_sha256 = prereg_v20a.canonical_sha256
file_sha256 = prereg_v20a.file_sha256


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _verify_commit_file_v20a(path: Path, digest: str) -> None:
    path = Path(path).resolve()
    relative = path.relative_to(ROOT).as_posix()
    raw = subprocess.check_output(
        ["git", "show", f"{SEALED_FOUNDATION_COMMIT_V20A}:{relative}"],
        cwd=ROOT,
    )
    if hashlib.sha256(raw).hexdigest() != digest or file_sha256(path) != digest:
        raise RuntimeError(f"v20a committed foundation changed: {relative}")


def load_hardened_preregistration_v20a() -> dict:
    bindings = {
        Path(frame_v20a.__file__).resolve(): FRAME_BUILDER_FILE_SHA256_V20A,
        frame_v20a.OUTPUT_PATH_V20A: FRAME_CERTIFICATE_FILE_SHA256_V20A,
        Path(prereg_v20a.__file__).resolve(): PREREGISTRATION_BUILDER_FILE_SHA256_V20A,
        prereg_v20a.OUTPUT_PATH_V20A: PREREGISTRATION_FILE_SHA256_V20A,
    }
    for path, digest in bindings.items():
        _verify_commit_file_v20a(path, digest)
    if (
        file_sha256(Path(draw_v20a.__file__).resolve())
        != DRAW_PLAN_BUILDER_FILE_SHA256_V20A
        or file_sha256(draw_v20a.OUTPUT_PATH_V20A) != DRAW_PLAN_FILE_SHA256_V20A
    ):
        raise RuntimeError("v20a draw-plan source or artifact changed")
    frame = json.loads(frame_v20a.OUTPUT_PATH_V20A.read_text(encoding="utf-8"))
    frame_v20a.validate_certificate_v20a(frame)
    preregistration = json.loads(
        prereg_v20a.OUTPUT_PATH_V20A.read_text(encoding="utf-8")
    )
    prereg_v20a.validate_preregistration_v20a(preregistration)
    draw = json.loads(draw_v20a.OUTPUT_PATH_V20A.read_text(encoding="utf-8"))
    draw_v20a.validate_draw_plan_certificate_v20a(draw)
    if (
        frame.get("content_sha256_before_self_field")
        != FRAME_CERTIFICATE_CONTENT_SHA256_V20A
        or preregistration.get("content_sha256_before_self_field")
        != PREREGISTRATION_CONTENT_SHA256_V20A
        or preregistration != prereg_v20a.build_preregistration_v20a()
        or draw.get("content_sha256_before_self_field")
        != DRAW_PLAN_CONTENT_SHA256_V20A
        or draw != draw_v20a.build_draw_plan_certificate_v20a()
        or preregistration.get("frozen_recipe", {}).get(
            "perturbation_basis_sha256"
        ) != prereg_v20a.PERTURBATION_BASIS_SHA256_V20A
        or preregistration.get("firewall", {}).get("gpu_launch_allowed") is not False
    ):
        raise RuntimeError("v20a hardened preregistration content changed")
    return preregistration


def _panel_order_v20a(panel, component_indices, components):
    return sorted(component_indices, key=lambda index: canonical_sha256({
        "schema": "eggroll-es-runtime-panel-order-v20a",
        "seed": frame_v20a.FLOW_SEED_V20A,
        "panel": panel,
        "joint_id": components[index]["joint_id"],
    }))


def _materialize_panel_bundle_v20a() -> dict:
    preregistration = load_hardened_preregistration_v20a()
    candidate, production = frame_v20a.frame_v19a.frame_v18a.load_bound_rows_v18a()
    frame = frame_v20a.frame_v19a.frame_v18a.build_joint_frame_v18a(
        candidate, production
    )
    tiers = frame_v20a.frame_v19a.frame_v18a.assign_patch_layers_v18a(frame)
    base_flow = frame_v20a.solve_base_flow_v20a(frame, tiers)
    candidate_assignment = frame_v20a.assign_candidate_only_v20a(frame, tiers)
    persisted_frame = json.loads(
        frame_v20a.OUTPUT_PATH_V20A.read_text(encoding="utf-8")
    )
    if frame_v20a.build_certificate_v20a() != persisted_frame:
        raise RuntimeError("v20a reconstructed frame differs from certificate")
    token_audit = preregistration["immutable_inputs"]["token_length_audit"]
    if (
        token_audit.get("over_frozen_1024_total_token_cap_count") != 0
        or token_audit.get("observed_combined_token_max", 1025) > 1024
    ):
        raise RuntimeError("v20a inherited token audit changed")

    components = frame["joint_components"]
    side_rows = {"candidate": candidate, "production": production}
    certificate_panels = {
        panel["name"]: panel for panel in persisted_frame["panels"]
    }
    panels = {}
    for panel in PANEL_NAMES_V20A:
        base_indices = _panel_order_v20a(
            panel, base_flow["selected_base"][panel], components
        )
        arms = {}
        for arm in ARMS_V20A:
            active_tiers = frame_v20a.ARM_ACTIVE_TIERS_V20A[arm]
            active_indices = list(base_indices) + [
                candidate_assignment["assignments"][panel][tier]
                for tier in active_tiers
            ]
            rows = []
            row_digests = []
            joint_ids = []
            ht_strata = []
            weights = []
            representative_sides = []
            for component_index in active_indices:
                component = components[component_index]
                side, expected_digest = frame_v20a._representative_for_arm_v20a(
                    component_index, active_tiers, frame, tiers
                )
                unit = frame["side_units"][side][component[f"{side}_unit"]]
                row = side_rows[side][unit["representative_index"]]
                if (
                    frame_v20a.frame_v19a.frame_v18a.sampler_v13.row_sha256(row)
                    != expected_digest
                    or not isinstance(row.get("question"), str)
                    or not row["question"]
                    or not isinstance(row.get("answer"), str)
                    or not row["answer"]
                ):
                    raise RuntimeError("v20a fixed train representative changed")
                if component_index in tiers["base_category"]:
                    stratum = tiers["base_category"][component_index]
                    weight = (
                        frame_v20a.BASE_CATEGORY_POPULATIONS_V20A[stratum]
                        / frame_v20a.BASE_CATEGORY_QUOTA_V20A
                    )
                else:
                    tier = tiers["candidate_only_layer"][component_index]
                    if tier not in active_tiers:
                        raise RuntimeError("v20a candidate-only tier leaked across arms")
                    stratum = f"candidate_only_tier_{tier}"
                    weight = float(
                        frame_v20a.CANDIDATE_ONLY_TIER_POPULATIONS_V20A[tier]
                    )
                rows.append(row)
                row_digests.append(expected_digest)
                joint_ids.append(component["joint_id"])
                ht_strata.append(stratum)
                weights.append(float(weight))
                representative_sides.append(side)
            certificate_arm = certificate_panels[panel]["arms"][arm]
            representative_root = frame_v20a.identity_root_sha256(
                f"{joint_id}:{side}:{row_digest}"
                for joint_id, side, row_digest in zip(
                    joint_ids, representative_sides, row_digests
                )
            )
            if (
                len(rows) != frame_v20a.ARM_REQUESTS_PER_PANEL_V20A[arm]
                or len(joint_ids) != len(set(joint_ids))
                or joint_ids[:24]
                != [components[index]["joint_id"] for index in base_indices]
                or not math.isclose(
                    math.fsum(weights),
                    float(frame_v20a.ARM_POPULATIONS_V20A[arm]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or frame_v20a.identity_root_sha256(joint_ids)
                != certificate_arm["active_joint_component_identity_root_sha256"]
                or representative_root
                != certificate_arm["representative_assignment_root_sha256"]
                or certificate_arm["same_arm_paired_duplicate_count"] != 0
            ):
                raise RuntimeError("v20a materialized arm estimand changed")
            arms[arm] = {
                "ordered_joint_identity_sha256": canonical_sha256(joint_ids),
                "ordered_row_identity_sha256": canonical_sha256(row_digests),
                "joint_ids": joint_ids,
                "row_sha256": row_digests,
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
        production_base = arms["production_only"]["joint_ids"]
        if any(arms[arm]["joint_ids"][:24] != production_base for arm in ARMS_V20A):
            raise RuntimeError("v20a paired base order differs across arms")
        panels[panel] = {
            "name": panel,
            "role": (
                "optimization"
                if panel in OPTIMIZATION_PANELS_V20A
                else "train_only_screen"
            ),
            "arms": arms,
        }
    result = {
        "schema": "eggroll-es-materialized-nested-tier-batches-v20a",
        "preregistration": {
            "file_sha256": PREREGISTRATION_FILE_SHA256_V20A,
            "content_sha256": PREREGISTRATION_CONTENT_SHA256_V20A,
            "sealed_commit": SEALED_FOUNDATION_COMMIT_V20A,
        },
        "frame": {
            "file_sha256": FRAME_CERTIFICATE_FILE_SHA256_V20A,
            "content_sha256": FRAME_CERTIFICATE_CONTENT_SHA256_V20A,
            "sealed_commit": SEALED_FOUNDATION_COMMIT_V20A,
        },
        "draw_plan": {
            "file_sha256": DRAW_PLAN_FILE_SHA256_V20A,
            "content_sha256": DRAW_PLAN_CONTENT_SHA256_V20A,
        },
        "token_audit": {
            "content_sha256": token_audit["content_sha256"],
            "frozen_total_token_cap": 1024,
            "over_cap_count": 0,
            "observed_combined_token_max": token_audit[
                "observed_combined_token_max"
            ],
        },
        "sources": {
            "candidate_v298": {
                "rows": frame_v20a.frame_v19a.frame_v18a.CANDIDATE_ROWS_V18A,
                "file_sha256": frame_v20a.frame_v19a.frame_v18a.CANDIDATE_SHA256_V18A,
            },
            "production": {
                "rows": frame_v20a.frame_v19a.frame_v18a.PRODUCTION_ROWS_V18A,
                "file_sha256": frame_v20a.frame_v19a.frame_v18a.PRODUCTION_SHA256_V18A,
            },
        },
        "panels": panels,
        "contains_train_question_answer_content_in_memory": True,
        "contains_evaluation_content": False,
        "persisted_to_disk": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def validate_panel_bundle_v20a(bundle):
    if (
        not isinstance(bundle, dict)
        or bundle.get("schema") != "eggroll-es-materialized-nested-tier-batches-v20a"
        or bundle.get("contains_train_question_answer_content_in_memory") is not True
        or bundle.get("contains_evaluation_content") is not False
        or bundle.get("persisted_to_disk") is not False
        or bundle.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(bundle))
        or bundle.get("content_sha256_before_self_field")
        != PANEL_BUNDLE_CONTENT_SHA256_V20A
        or tuple(bundle.get("panels", {})) != PANEL_NAMES_V20A
        or bundle.get("token_audit", {}).get("over_cap_count") != 0
        or bundle.get("token_audit", {}).get("observed_combined_token_max", 1025)
        > 1024
    ):
        raise RuntimeError("v20a materialized panel bundle changed")
    for panel_name in PANEL_NAMES_V20A:
        panel = bundle["panels"][panel_name]
        if set(panel.get("arms", {})) != set(ARMS_V20A):
            raise RuntimeError("v20a materialized arm coverage changed")
        production_base = panel["arms"]["production_only"]["joint_ids"]
        for arm in ARMS_V20A:
            batch = panel["arms"][arm]
            expected_rows = frame_v20a.ARM_REQUESTS_PER_PANEL_V20A[arm]
            active_tiers = frame_v20a.ARM_ACTIVE_TIERS_V20A[arm]
            candidate_strata = [
                item for item in batch.get("ht_strata", [])
                if item.startswith("candidate_only_tier_")
            ]
            if (
                any(len(batch.get(field, [])) != expected_rows for field in (
                    "joint_ids", "row_sha256", "ht_strata", "weights",
                    "representative_sides", "questions", "answers",
                ))
                or len(set(batch["joint_ids"])) != expected_rows
                or batch["joint_ids"][:24] != production_base
                or canonical_sha256(batch["joint_ids"])
                != batch["ordered_joint_identity_sha256"]
                or canonical_sha256(batch["row_sha256"])
                != batch["ordered_row_identity_sha256"]
                or not math.isclose(
                    math.fsum(batch["weights"]),
                    float(frame_v20a.ARM_POPULATIONS_V20A[arm]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or candidate_strata
                != [f"candidate_only_tier_{tier}" for tier in active_tiers]
                or canonical_sha256({
                    "questions": batch["questions"],
                    "answers": batch["answers"],
                }) != batch["raw_prompt_answer_sha256"]
            ):
                raise RuntimeError("v20a materialized panel batch changed")
    return bundle


def load_panel_bundle_v20a():
    return validate_panel_bundle_v20a(_materialize_panel_bundle_v20a())


def _unit_score_arrays_v20a(unit_scores):
    if not isinstance(unit_scores, dict) or set(unit_scores) != set(ARMS_V20A):
        raise RuntimeError("v20a per-unit score arm mapping changed")
    result = {}
    for arm in ARMS_V20A:
        values = np.asarray(unit_scores[arm], dtype=np.float64)
        expected = (10, 2, 32, frame_v20a.ARM_REQUESTS_PER_PANEL_V20A[arm])
        if values.shape != expected or not np.isfinite(values).all():
            raise RuntimeError("v20a per-unit score tensor changed")
        result[arm] = values
    return result


def observed_panel_scores_v20a(unit_scores, panel_bundle):
    values = _unit_score_arrays_v20a(unit_scores)
    panel_bundle = validate_panel_bundle_v20a(panel_bundle)
    result = np.empty((4, 10, 2, 32), dtype=np.float64)
    for arm_index, arm in enumerate(ARMS_V20A):
        weights = np.asarray([
            panel_bundle["panels"][panel]["arms"][arm]["weights"]
            for panel in PANEL_NAMES_V20A
        ], dtype=np.float64)
        result[arm_index] = np.einsum(
            "psdu,pu->psd", values[arm], weights
        ) / float(frame_v20a.ARM_POPULATIONS_V20A[arm])
    if not np.isfinite(result).all():
        raise RuntimeError("v20a HT panel-score recomputation changed")
    return result


def _endpoint_arrays_v20a(panel_scores):
    return mechanics_v19a._endpoint_arrays_v19a(panel_scores)


def recompute_observed_endpoints_v20a(unit_scores, panel_bundle):
    panel_scores = observed_panel_scores_v20a(unit_scores, panel_bundle)
    analyzed = _endpoint_arrays_v20a(panel_scores[:, np.newaxis, ...])
    result = {}
    for arm_index, arm in enumerate(ARMS_V20A):
        endpoint_values = {
            name: float(values[arm_index, 0])
            for name, values in analyzed["endpoints"].items()
        }
        compact_payload = {
            "schema": "eggroll-es-compact-nested-tier-estimator-v20a",
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


def _bootstrap_draw_plan_v20a():
    certificate = json.loads(draw_v20a.OUTPUT_PATH_V20A.read_text(encoding="utf-8"))
    draw_v20a.validate_draw_plan_certificate_v20a(certificate)
    arrays = draw_v20a.materialize_draw_arrays_v20a()
    if (
        hashlib.sha256(arrays["base"].tobytes()).hexdigest()
        != certificate["base_draws"]["bytes_sha256"]
        or hashlib.sha256(arrays["candidate_source_offsets"].tobytes()).hexdigest()
        != certificate["candidate_source_offsets"]["bytes_sha256"]
    ):
        raise RuntimeError("v20a exact bootstrap draw arrays changed")
    return {
        **arrays,
        "content_sha256": certificate["content_sha256_before_self_field"],
    }


def _bootstrap_positions_v20a(panel_bundle):
    base_positions = {
        arm: {
            panel_index: {
                category: np.asarray([
                    index for index, value in enumerate(
                        panel_bundle["panels"][panel]["arms"][arm]["ht_strata"]
                    ) if value == category
                ], dtype=np.int64)
                for category in BASE_CATEGORIES_V20A
            }
            for panel_index, panel in enumerate(PANEL_NAMES_V20A)
        }
        for arm in ARMS_V20A
    }
    candidate_positions = {
        arm: {
            panel_index: {
                tier: next((
                    index for index, value in enumerate(
                        panel_bundle["panels"][panel]["arms"][arm]["ht_strata"]
                    ) if value == f"candidate_only_tier_{tier}"
                ), None)
                for tier in frame_v20a.ARM_ACTIVE_TIERS_V20A[arm]
            }
            for panel_index, panel in enumerate(PANEL_NAMES_V20A)
        }
        for arm in ARMS_V20A
    }
    if any(
        positions.shape != (6,)
        for arm in ARMS_V20A
        for panel in base_positions[arm].values()
        for positions in panel.values()
    ) or any(
        position is None
        for arm in ARMS_V20A
        for panel in candidate_positions[arm].values()
        for position in panel.values()
    ):
        raise RuntimeError("v20a bootstrap stratum positions changed")
    return base_positions, candidate_positions


def _bootstrap_panel_scores_v20a(values, panel_bundle, draw_plan, start, stop):
    batch_size = stop - start
    if not 0 <= start < stop <= BOOTSTRAP_REPETITIONS_V20A:
        raise ValueError("v20a bootstrap chunk bounds changed")
    result = np.zeros((4, batch_size, 10, 2, 32), dtype=np.float64)
    base_positions, candidate_positions = _bootstrap_positions_v20a(panel_bundle)
    for arm_index, arm in enumerate(ARMS_V20A):
        denominator = float(frame_v20a.ARM_POPULATIONS_V20A[arm])
        for target in range(10):
            target_scores = np.broadcast_to(
                values[arm][target],
                (batch_size,) + values[arm][target].shape,
            )
            for category_index, category in enumerate(BASE_CATEGORIES_V20A):
                positions = base_positions[arm][target][category]
                sampled_positions = positions[
                    draw_plan["base"][target, category_index, start:stop]
                ]
                sampled = np.take_along_axis(
                    target_scores,
                    sampled_positions[:, np.newaxis, np.newaxis, :],
                    axis=-1,
                )
                population = frame_v20a.BASE_CATEGORY_POPULATIONS_V20A[category]
                result[arm_index, :, target] += (
                    float(population) / denominator
                ) * np.mean(sampled, axis=-1)
            for tier in frame_v20a.ARM_ACTIVE_TIERS_V20A[arm]:
                role_indices = (
                    np.arange(6, dtype=np.int64)
                    if target < 6 else np.arange(6, 10, dtype=np.int64)
                )
                offsets = draw_plan["candidate_source_offsets"][
                    tier - 1, target, start:stop
                ]
                sources = role_indices[offsets]
                source_scores = values[arm][sources]
                positions = np.asarray([
                    candidate_positions[arm][int(source)][tier]
                    for source in sources
                ], dtype=np.int64)
                sampled = np.take_along_axis(
                    source_scores,
                    positions[:, np.newaxis, np.newaxis, np.newaxis],
                    axis=-1,
                )[..., 0]
                population = frame_v20a.CANDIDATE_ONLY_TIER_POPULATIONS_V20A[tier]
                result[arm_index, :, target] += (
                    float(population) / denominator
                ) * sampled
    return result


def paired_stratified_bootstrap_v20a(
    unit_scores,
    panel_bundle,
    *,
    chunk_size=BOOTSTRAP_DEFAULT_CHUNK_SIZE_V20A,
):
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("v20a bootstrap chunk size must be positive")
    values = _unit_score_arrays_v20a(unit_scores)
    panel_bundle = validate_panel_bundle_v20a(panel_bundle)
    observed = recompute_observed_endpoints_v20a(values, panel_bundle)["arms"]
    draw_plan = _bootstrap_draw_plan_v20a()
    comparisons = {
        contrast: {name: [] for name in prereg_v20a.ENDPOINT_NAMES_V20A}
        for contrast in CONTRASTS_V20A
    }
    arm_index = {arm: index for index, arm in enumerate(ARMS_V20A)}
    completed = 0
    while completed < BOOTSTRAP_REPETITIONS_V20A:
        stop = min(BOOTSTRAP_REPETITIONS_V20A, completed + chunk_size)
        panel_scores = _bootstrap_panel_scores_v20a(
            values, panel_bundle, draw_plan, completed, stop
        )
        analyzed = _endpoint_arrays_v20a(panel_scores)
        for contrast, contract in CONTRASTS_V20A.items():
            treatment = arm_index[contract["treatment"]]
            control = arm_index[contract["control"]]
            for name, endpoint_values in analyzed["endpoints"].items():
                comparisons[contrast][name].append(
                    endpoint_values[treatment] - endpoint_values[control]
                )
        completed = stop
    quantile = prereg_v20a.FAMILYWISE_ALPHA_V20A / 60
    output = {}
    for contrast, contract in CONTRASTS_V20A.items():
        endpoints = {}
        for name, chunks in comparisons[contrast].items():
            deltas = np.concatenate(chunks)
            if (
                deltas.shape != (BOOTSTRAP_REPETITIONS_V20A,)
                or not np.isfinite(deltas).all()
            ):
                raise RuntimeError("v20a bootstrap replicate coverage changed")
            observed_delta = (
                observed[contract["treatment"]]["endpoint_values"][name]
                - observed[contract["control"]]["endpoint_values"][name]
            )
            endpoints[name] = {
                "treatment_minus_control": float(observed_delta),
                "familywise_lcb": float(np.quantile(
                    deltas, quantile, method=BOOTSTRAP_QUANTILE_METHOD_V20A
                )),
                "noninferiority_margin": 0.0,
            }
        output[contrast] = endpoints
    return {
        "seed": prereg_v20a.BOOTSTRAP_SEED_V20A,
        "repetitions": BOOTSTRAP_REPETITIONS_V20A,
        "one_sided_quantile": quantile,
        "draw_plan_content_sha256": draw_plan["content_sha256"],
        "whole_panel_block_resampling_used": False,
        "comparisons": output,
    }


def evaluate_attribution_gate_v20a(summary):
    arms = summary.get("arms", {})
    bootstrap = summary.get("paired_bootstrap", {})
    integrity = summary.get("runtime_integrity", {})
    if (
        set(arms) != set(ARMS_V20A)
        or set(bootstrap.get("comparisons", {})) != set(CONTRASTS_V20A)
        or bootstrap.get("repetitions") != BOOTSTRAP_REPETITIONS_V20A
        or bootstrap.get("one_sided_quantile") != 0.05 / 60
        or bootstrap.get("whole_panel_block_resampling_used") is not False
    ):
        raise RuntimeError("v20a gate input coverage changed")
    results = {}
    for contrast, contract in CONTRASTS_V20A.items():
        endpoints = bootstrap["comparisons"][contrast]
        if set(endpoints) != set(prereg_v20a.ENDPOINT_NAMES_V20A):
            raise RuntimeError("v20a gate endpoint coverage changed")
        observed_pass = {
            name: item["treatment_minus_control"] >= 0.0
            for name, item in endpoints.items()
        }
        bootstrap_pass = {
            name: item["familywise_lcb"] >= 0.0
            for name, item in endpoints.items()
        }
        spreads = (
            arms[contract["treatment"]].get("all_panel_spreads_nonzero") is True
            and arms[contract["control"]].get("all_panel_spreads_nonzero") is True
        )
        passed = (
            all(observed_pass.values())
            and all(bootstrap_pass.values())
            and spreads
            and integrity.get("all_integrity_audits_passed") is True
        )
        results[contrast] = {
            "observed_pass_count": sum(observed_pass.values()),
            "bootstrap_pass_count": sum(bootstrap_pass.values()),
            "all_twelve_observed_passed": all(observed_pass.values()),
            "all_twelve_bootstrap_passed": all(bootstrap_pass.values()),
            "preregistered_contrast_gate_passed": bool(passed),
        }
    all_passed = all(
        item["preregistered_contrast_gate_passed"] for item in results.values()
    )
    return {
        "schema": "eggroll-es-nested-tier-attribution-gate-v20a",
        "contrasts": results,
        "all_five_contrasts_passed": bool(all_passed),
        "decision": (
            "authorize_only_separate_fresh_basis_train_only_confirmation_preregistration"
            if all_passed else "retain_production_dataset_and_v13_recipe"
        ),
        "dataset_promotion_authorized": False,
        "model_update_authorized": False,
        "evaluation_authorized": False,
    }


def build_compact_estimator_summary_v20a(unit_scores, panel_bundle):
    observed = recompute_observed_endpoints_v20a(unit_scores, panel_bundle)
    value = {
        "schema": "eggroll-es-nested-tier-compact-estimator-summary-v20a",
        "arms": copy.deepcopy(observed["arms"]),
        "paired_bootstrap": paired_stratified_bootstrap_v20a(
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
