#!/usr/bin/env python3
"""Pure train-only mechanics for the immutable V25A paired data scan."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import build_eggroll_es_joint_panels_v25a as frame_v25a
import eggroll_es_paired_data_compat_preregistration_v25a as prereg_v25a
import eggroll_es_train_panel_sampler_v13 as sampler_v13
import train_eggroll_es_specialist_anchor_v13 as anchor_v13


ROOT = Path(__file__).resolve().parent
CANONICAL_ROOT = Path("/home/catid/specialist")
PREREGISTRATION_FILE_SHA256_V25A = (
    "6ace4b6d8f1fb9948c1f1c698b1e201ff782018c335b1ffdc68ed56dee49f64a"
)
PREREGISTRATION_CONTENT_SHA256_V25A = (
    "0b5dfc076304bb8eb8bddd4f0f0d9d7754a0220c01839188b7be82484525a748"
)
PANEL_BUNDLE_CONTENT_SHA256_V25A = (
    "969a37771bca7da2aa53248a76e4434c923ffa99717cb6e79932653f84d6b6a0"
)
CANDIDATE_PATH_V25A = (
    CANONICAL_ROOT / "experiments/eggroll_es_hpo/dataset_candidates/v364/"
    "train_qa_context_merit_v364.jsonl"
)
PRODUCTION_PATH_V25A = CANONICAL_ROOT / "data/train_qa_curated_v1.jsonl"
VERSIONS_V25A = ("production", "candidate_v364")
VERSION_TO_FRAME_SIDE_V25A = {
    "production": "production",
    "candidate_v364": "candidate",
}
PANEL_NAMES_V25A = frame_v25a.PANEL_NAMES
OPTIMIZATION_PANELS_V25A = frame_v25a.OPTIMIZATION_PANELS
TRAIN_SCREENS_V25A = frame_v25a.TRAIN_SCREENS
STRATA_V25A = tuple(sampler_v13.STRATA)
SIGNS_V25A = ("plus", "minus")
PAIRED_FRAME_POPULATION_V25A = frame_v25a.EXPECTED_PAIRED_UNITS
BOOTSTRAP_CHUNK_SIZE_V25A = 128
BOOTSTRAP_QUANTILE_METHOD_V25A = "linear"
EXPECTED_DRAW_PLAN_SHA256_V25A = (
    "44569a4a813d0b736b6c093b7c2b5e1ffd4b1a353398b98cc14dabe4a718f7c2"
)


canonical_sha256 = prereg_v25a.canonical_sha256
file_sha256 = prereg_v25a.file_sha256


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _read_bound_rows(path, expected_sha256, expected_rows):
    path = Path(path).resolve()
    if file_sha256(path) != expected_sha256:
        raise RuntimeError("v25a paired source file identity changed")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected_rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("v25a paired source row count or schema changed")
    return rows


def load_hardened_preregistration_v25a():
    if (
        file_sha256(prereg_v25a.PREREGISTRATION_PATH)
        != PREREGISTRATION_FILE_SHA256_V25A
    ):
        raise RuntimeError("v25a preregistration file changed")
    persisted = json.loads(
        prereg_v25a.PREREGISTRATION_PATH.read_text(encoding="utf-8")
    )
    if (
        persisted.get("content_sha256_before_self_field")
        != PREREGISTRATION_CONTENT_SHA256_V25A
        or persisted.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(persisted))
        or persisted.get("required_runtime_adapter", {}).get("status")
        != "not_yet_implemented"
        or persisted.get("required_runtime_adapter", {}).get(
            "runtime_launch_authorized"
        ) is not False
    ):
        raise RuntimeError("v25a preregistration content changed")
    prereg_v25a.validate_sha_fields(persisted)
    return persisted


def _materialize_paired_panel_bundle_v25a():
    preregistration = load_hardened_preregistration_v25a()
    frame, _negative, _layer_plan = prereg_v25a.load_bound_evidence()
    sources = {
        "production": _read_bound_rows(
            PRODUCTION_PATH_V25A,
            frame_v25a.PRODUCTION_SHA256,
            frame_v25a.PRODUCTION_ROWS,
        ),
        "candidate_v364": _read_bound_rows(
            CANDIDATE_PATH_V25A,
            frame_v25a.CANDIDATE_SHA256,
            frame_v25a.CANDIDATE_ROWS,
        ),
    }
    panels = {}
    anchor_counts = {"shared_document": 0, "joint_component_cross_side_link": 0}
    for manifest_panel in frame["panels"]:
        panel_name = manifest_panel["name"]
        items = manifest_panel["items"]
        version_batches = {}
        for version in VERSIONS_V25A:
            side = VERSION_TO_FRAME_SIDE_V25A[version]
            source = sources[version]
            rows = []
            row_hashes = []
            for item in items:
                representative = item["sides"][side]
                row_index = representative["row_index"]
                if (
                    isinstance(row_index, bool)
                    or not isinstance(row_index, int)
                    or not 0 <= row_index < len(source)
                ):
                    raise RuntimeError("v25a fixed representative index changed")
                row = source[row_index]
                row_hash = sampler_v13.row_sha256(row)
                if (
                    row_hash != representative["row_sha256"]
                    or row.get("document_sha256")
                    != representative["document_sha256"]
                    or not isinstance(row.get("question"), str)
                    or not row["question"]
                    or not isinstance(row.get("answer"), str)
                    or not row["answer"]
                ):
                    raise RuntimeError("v25a fixed representative content changed")
                rows.append(row)
                row_hashes.append(row_hash)
            version_batches[version] = {
                "ordered_row_identity_sha256": canonical_sha256(row_hashes),
                "row_sha256": row_hashes,
                "questions": [row["question"] for row in rows],
                "answers": [row["answer"] for row in rows],
                "raw_prompt_answer_sha256": canonical_sha256({
                    "questions": [row["question"] for row in rows],
                    "answers": [row["answer"] for row in rows],
                }),
            }
        for item in items:
            anchor_counts[item["pairing_anchor"]] += 1
        panels[panel_name] = {
            "name": panel_name,
            "role": manifest_panel["role"],
            "ordered_unit_identity_sha256": manifest_panel[
                "ordered_unit_identity_sha256"
            ],
            "unit_ids": [item["unit_id"] for item in items],
            "strata": [item["stratum"] for item in items],
            "weights": [
                float(item["horvitz_thompson_unit_weight"]) for item in items
            ],
            "pairing_anchors": [item["pairing_anchor"] for item in items],
            "versions": version_batches,
        }
    if anchor_counts != {
        "shared_document": 193,
        "joint_component_cross_side_link": 2,
    }:
        raise RuntimeError("v25a selected pairing-anchor coverage changed")
    result = {
        "schema": "eggroll-es-materialized-paired-data-batches-v25a",
        "preregistration": {
            "file_sha256": PREREGISTRATION_FILE_SHA256_V25A,
            "content_sha256": preregistration["content_sha256_before_self_field"],
        },
        "frame": {
            "file_sha256": prereg_v25a.FRAME_FILE_SHA256,
            "content_sha256": prereg_v25a.FRAME_CONTENT_SHA256,
            "selected_paired_units": 195,
            "reserve_paired_units": 10,
            "shared_document_anchors": 193,
            "joint_component_cross_side_anchors": 2,
        },
        "sources": {
            "production": {
                "rows": frame_v25a.PRODUCTION_ROWS,
                "file_sha256": frame_v25a.PRODUCTION_SHA256,
            },
            "candidate_v364": {
                "rows": frame_v25a.CANDIDATE_ROWS,
                "file_sha256": frame_v25a.CANDIDATE_SHA256,
            },
        },
        "panels": panels,
        "contains_validation_ood_heldout_or_benchmark_content": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def validate_paired_panel_bundle_v25a(bundle):
    if (
        not isinstance(bundle, dict)
        or set(bundle) != {
            "schema", "preregistration", "frame", "sources", "panels",
            "contains_validation_ood_heldout_or_benchmark_content",
            "content_sha256_before_self_field",
        }
        or bundle.get("schema")
        != "eggroll-es-materialized-paired-data-batches-v25a"
        or bundle.get("contains_validation_ood_heldout_or_benchmark_content")
        is not False
        or bundle.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(bundle))
        or bundle.get("content_sha256_before_self_field")
        != PANEL_BUNDLE_CONTENT_SHA256_V25A
        or bundle.get("preregistration") != {
            "file_sha256": PREREGISTRATION_FILE_SHA256_V25A,
            "content_sha256": PREREGISTRATION_CONTENT_SHA256_V25A,
        }
        or bundle.get("frame") != {
            "file_sha256": prereg_v25a.FRAME_FILE_SHA256,
            "content_sha256": prereg_v25a.FRAME_CONTENT_SHA256,
            "selected_paired_units": 195,
            "reserve_paired_units": 10,
            "shared_document_anchors": 193,
            "joint_component_cross_side_anchors": 2,
        }
        or bundle.get("sources") != {
            "production": {
                "rows": frame_v25a.PRODUCTION_ROWS,
                "file_sha256": frame_v25a.PRODUCTION_SHA256,
            },
            "candidate_v364": {
                "rows": frame_v25a.CANDIDATE_ROWS,
                "file_sha256": frame_v25a.CANDIDATE_SHA256,
            },
        }
        or tuple(bundle.get("panels", {})) != PANEL_NAMES_V25A
    ):
        raise RuntimeError("v25a materialized paired panel bundle changed")
    frame = json.loads(frame_v25a.OUTPUT_PATH.read_text(encoding="utf-8"))
    frame_v25a.validate_manifest(frame)
    manifest_panels = {item["name"]: item for item in frame["panels"]}
    anchor_counts = {"shared_document": 0, "joint_component_cross_side_link": 0}
    selected_ids = []
    for panel_name in PANEL_NAMES_V25A:
        panel = bundle["panels"][panel_name]
        manifest_panel = manifest_panels[panel_name]
        items = manifest_panel["items"]
        selected_ids.extend(panel.get("unit_ids", []))
        if (
            set(panel) != {
                "name", "role", "ordered_unit_identity_sha256", "unit_ids",
                "strata", "weights", "pairing_anchors", "versions",
            }
            or panel.get("name") != panel_name
            or panel.get("role") != manifest_panel["role"]
            or panel.get("unit_ids") != [item["unit_id"] for item in items]
            or panel.get("strata") != [item["stratum"] for item in items]
            or panel.get("pairing_anchors")
            != [item["pairing_anchor"] for item in items]
            or panel.get("weights") != [
                float(item["horvitz_thompson_unit_weight"]) for item in items
            ]
            or canonical_sha256(panel["unit_ids"])
            != panel.get("ordered_unit_identity_sha256")
            or panel.get("ordered_unit_identity_sha256")
            != manifest_panel["ordered_unit_identity_sha256"]
            or set(panel.get("versions", {})) != set(VERSIONS_V25A)
            or not math.isclose(
                math.fsum(panel["weights"]),
                float(PAIRED_FRAME_POPULATION_V25A),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError(f"v25a materialized {panel_name} contract changed")
        for anchor in panel["pairing_anchors"]:
            anchor_counts[anchor] += 1
        for version in VERSIONS_V25A:
            side = VERSION_TO_FRAME_SIDE_V25A[version]
            version_batch = panel["versions"][version]
            expected_hashes = [item["sides"][side]["row_sha256"] for item in items]
            if (
                set(version_batch) != {
                    "ordered_row_identity_sha256", "row_sha256", "questions",
                    "answers", "raw_prompt_answer_sha256",
                }
                or [
                    len(version_batch.get(key, []))
                    for key in ("row_sha256", "questions", "answers")
                ] != [frame_v25a.PANEL_SIZE] * 3
                or version_batch["row_sha256"] != expected_hashes
                or canonical_sha256(expected_hashes)
                != version_batch["ordered_row_identity_sha256"]
                or version_batch["ordered_row_identity_sha256"]
                != manifest_panel["ordered_side_row_identity_sha256"][side]
                or canonical_sha256({
                    "questions": version_batch["questions"],
                    "answers": version_batch["answers"],
                }) != version_batch["raw_prompt_answer_sha256"]
            ):
                raise RuntimeError(
                    f"v25a materialized {panel_name} {version} batch changed"
                )
    if (
        len(selected_ids) != 195
        or len(set(selected_ids)) != 195
        or anchor_counts != {
            "shared_document": 193,
            "joint_component_cross_side_link": 2,
        }
    ):
        raise RuntimeError("v25a selected frame identity changed")
    return bundle


def load_paired_panel_bundle_v25a():
    return validate_paired_panel_bundle_v25a(
        _materialize_paired_panel_bundle_v25a()
    )


def resident_signed_wave_schedule_v25a():
    frozen = load_hardened_preregistration_v25a()
    base_schedule = frozen["frozen_recipe"]["perturbation_basis"][
        "signed_population_schedule"
    ]
    if base_schedule != prereg_v25a.signed_population_schedule():
        raise RuntimeError("v25a signed population schedule changed")
    result = []
    for item in base_schedule:
        signed_index = item["signed_wave_index"]
        order = (
            list(VERSIONS_V25A)
            if signed_index % 2 == 0
            else list(reversed(VERSIONS_V25A))
        )
        result.append({
            **copy.deepcopy(item),
            "resident_version_order": order,
            "restore_after_both_versions": True,
        })
    if (
        len(result) != 16
        or [item["signed_wave_index"] for item in result] != list(range(16))
        or any(set(item["resident_version_order"]) != set(VERSIONS_V25A) for item in result)
        or sum(item["resident_version_order"][0] == "production" for item in result)
        != 8
    ):
        raise RuntimeError("v25a resident version-order balance changed")
    return result


def execute_paired_resident_signed_wave_v25a(
    schedule_item, *, perturb, score_version, restore,
):
    schedule = resident_signed_wave_schedule_v25a()
    index = schedule_item.get("signed_wave_index") if isinstance(schedule_item, dict) else None
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < 16
        or schedule_item != schedule[index]
    ):
        raise RuntimeError("v25a resident signed-wave schedule item changed")
    if not all(callable(value) for value in (perturb, score_version, restore)):
        raise TypeError("v25a resident signed-wave callbacks must be callable")
    captures = {}
    try:
        perturb(
            list(schedule_item["engine_direction_seeds"]),
            bool(schedule_item["negate"]),
        )
        for version in schedule_item["resident_version_order"]:
            captures[version] = score_version(version)
    finally:
        restore()
    if tuple(captures) != tuple(schedule_item["resident_version_order"]):
        raise RuntimeError("v25a resident paired-version capture is incomplete")
    return captures


def _unit_score_array(unit_scores):
    values = np.asarray(unit_scores, dtype=np.float64)
    expected = (2, 5, 2, 32, 39)
    if values.shape != expected or not np.isfinite(values).all():
        raise RuntimeError("v25a paired per-unit score tensor changed")
    return values


def _panel_design(panel_bundle):
    panel_bundle = validate_paired_panel_bundle_v25a(panel_bundle)
    weights = np.asarray([
        panel_bundle["panels"][name]["weights"] for name in PANEL_NAMES_V25A
    ], dtype=np.float64)
    strata = [
        list(panel_bundle["panels"][name]["strata"])
        for name in PANEL_NAMES_V25A
    ]
    if weights.shape != (5, 39):
        raise RuntimeError("v25a paired panel design changed")
    return weights, strata


def observed_panel_scores_v25a(unit_scores, panel_bundle):
    values = _unit_score_array(unit_scores)
    weights, _strata = _panel_design(panel_bundle)
    panel_scores = np.einsum("vpsdu,pu->vpsd", values, weights)
    panel_scores /= float(PAIRED_FRAME_POPULATION_V25A)
    if panel_scores.shape != (2, 5, 2, 32) or not np.isfinite(panel_scores).all():
        raise RuntimeError("v25a observed panel-score recomputation changed")
    return panel_scores


def _cosine_last_axis(left, right):
    numerator = np.sum(left * right, axis=-1)
    denominator = np.sqrt(
        np.sum(left * left, axis=-1) * np.sum(right * right, axis=-1)
    )
    if np.any(denominator == 0.0):
        raise RuntimeError("v25a endpoint cosine received a zero direction")
    return numerator / denominator


def _sign_agreement_last_axis(left, right):
    return np.mean(np.sign(left) == np.sign(right), axis=-1)


def _endpoint_arrays(panel_scores):
    values = np.asarray(panel_scores, dtype=np.float64)
    if (
        values.ndim != 5
        or values.shape[0] != 2
        or values.shape[2:] != (5, 2, 32)
        or not np.isfinite(values).all()
    ):
        raise RuntimeError("v25a panel-score endpoint tensor changed")
    central = 0.5 * (values[:, :, :, 0, :] - values[:, :, :, 1, :])
    means = np.mean(central, axis=-1, keepdims=True)
    spreads = np.sqrt(np.mean((central - means) ** 2, axis=-1))
    if np.any(spreads == 0.0):
        raise RuntimeError("v25a panel response spread is zero")
    coefficients = (central - means) / (
        spreads[..., np.newaxis] + anchor_v13.STANDARDIZATION_EPSILON_V13
    )
    aggregate = np.median(coefficients[:, :, :3, :], axis=2)
    optimization_pairs = ((0, 1), (0, 2), (1, 2))
    families = {
        "optimization_pairwise_cosine": np.stack([
            _cosine_last_axis(
                coefficients[:, :, left, :], coefficients[:, :, right, :]
            )
            for left, right in optimization_pairs
        ], axis=-1),
        "optimization_pairwise_sign_agreement": np.stack([
            _sign_agreement_last_axis(
                coefficients[:, :, left, :], coefficients[:, :, right, :]
            )
            for left, right in optimization_pairs
        ], axis=-1),
        "aggregate_to_optimization_cosine": np.stack([
            _cosine_last_axis(aggregate, coefficients[:, :, index, :])
            for index in range(3)
        ], axis=-1),
        "aggregate_to_optimization_sign_agreement": np.stack([
            _sign_agreement_last_axis(aggregate, coefficients[:, :, index, :])
            for index in range(3)
        ], axis=-1),
        "train_screen_cosine": np.stack([
            _cosine_last_axis(aggregate, coefficients[:, :, index, :])
            for index in (3, 4)
        ], axis=-1),
        "train_screen_sign_agreement": np.stack([
            _sign_agreement_last_axis(aggregate, coefficients[:, :, index, :])
            for index in (3, 4)
        ], axis=-1),
    }
    endpoints = {}
    for family in prereg_v25a.METRIC_FAMILIES:
        components = families[family]
        endpoints[f"{family}_median"] = np.median(components, axis=-1)
        endpoints[f"{family}_worst"] = np.min(components, axis=-1)
    if set(endpoints) != set(prereg_v25a.ENDPOINTS) or any(
        value.shape != values.shape[:2]
        or not np.isfinite(value).all()
        or np.any(value < -1.0 - 1e-12)
        or np.any(value > 1.0 + 1e-12)
        for value in endpoints.values()
    ):
        raise RuntimeError("v25a endpoint family recomputation changed")
    return {
        "endpoints": endpoints,
        "coefficients": coefficients,
        "aggregate": aggregate,
        "spreads": spreads,
    }


def recompute_observed_endpoints_v25a(unit_scores, panel_bundle):
    panel_scores = observed_panel_scores_v25a(unit_scores, panel_bundle)
    analyzed = _endpoint_arrays(panel_scores[:, np.newaxis, ...])
    versions = {}
    for version_index, version in enumerate(VERSIONS_V25A):
        endpoint_values = {
            name: float(values[version_index, 0])
            for name, values in analyzed["endpoints"].items()
        }
        versions[version] = {
            "all_panel_spreads_nonzero": bool(
                np.all(analyzed["spreads"][version_index, 0] > 0.0)
            ),
            "endpoint_values": endpoint_values,
            "compact_estimator_sha256": canonical_sha256({
                "schema": "eggroll-es-compact-estimator-commitment-v25a",
                "version": version,
                "panel_scores": panel_scores[version_index].tolist(),
                "coefficients": analyzed["coefficients"][version_index, 0].tolist(),
                "aggregate": analyzed["aggregate"][version_index, 0].tolist(),
                "endpoint_values": endpoint_values,
            }),
        }
    cross_cosine = float(_cosine_last_axis(
        analyzed["aggregate"][0, 0], analyzed["aggregate"][1, 0]
    ))
    return {
        "versions": versions,
        "cross_version_direction_similarity_diagnostic": {
            "used_for_gate": False,
            "content_sha256": canonical_sha256({
                "schema": "eggroll-es-cross-version-direction-diagnostic-v25a",
                "cosine": cross_cosine,
            }),
        },
    }


def _bootstrap_draw_plan_v25a(panel_bundle, repetitions, expected_sha256):
    _weights, strata_by_panel = _panel_design(panel_bundle)
    generator = np.random.default_rng(prereg_v25a.BOOTSTRAP_SEED)
    digest = hashlib.sha256()
    draws = {}
    for panel_index, panel in enumerate(PANEL_NAMES_V25A):
        draws[panel] = {}
        strata = strata_by_panel[panel_index]
        for stratum in STRATA_V25A:
            positions = np.asarray([
                index for index, value in enumerate(strata) if value == stratum
            ], dtype=np.int64)
            quota = frame_v25a.STRATUM_QUOTAS[stratum]
            if len(positions) != quota:
                raise RuntimeError("v25a bootstrap stratum quota changed")
            draw = generator.integers(
                0, quota, size=(repetitions, quota), dtype=np.int64,
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
            draws[panel][stratum] = {"positions": positions, "draw": draw}
    actual = digest.hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise RuntimeError("v25a bootstrap draw-plan identity changed")
    return draws, actual


def _paired_stratified_bootstrap_impl(
    unit_scores, panel_bundle, repetitions, expected_draw_plan_sha256,
):
    values = _unit_score_array(unit_scores)
    observed = recompute_observed_endpoints_v25a(values, panel_bundle)["versions"]
    draw_plan, draw_plan_sha256 = _bootstrap_draw_plan_v25a(
        panel_bundle, repetitions, expected_draw_plan_sha256,
    )
    deltas = {name: np.empty(repetitions, dtype=np.float64) for name in prereg_v25a.ENDPOINTS}
    for start in range(0, repetitions, BOOTSTRAP_CHUNK_SIZE_V25A):
        stop = min(start + BOOTSTRAP_CHUNK_SIZE_V25A, repetitions)
        batch_size = stop - start
        panel_scores = np.zeros((2, batch_size, 5, 2, 32), dtype=np.float64)
        for panel_index, panel in enumerate(PANEL_NAMES_V25A):
            for stratum in STRATA_V25A:
                plan = draw_plan[panel][stratum]
                stratum_scores = np.take(
                    values[:, panel_index], plan["positions"], axis=-1,
                )
                resampled = np.take(
                    stratum_scores, plan["draw"][start:stop], axis=-1,
                )
                stratum_means = np.mean(resampled, axis=-1).transpose(0, 3, 1, 2)
                population = frame_v25a.EXPECTED_PAIRED_STRATA[stratum]
                panel_scores[:, :, panel_index] += (
                    float(population) / float(PAIRED_FRAME_POPULATION_V25A)
                ) * stratum_means
        analyzed = _endpoint_arrays(panel_scores)
        for name, endpoint_values in analyzed["endpoints"].items():
            deltas[name][start:stop] = endpoint_values[1] - endpoint_values[0]
    quantile = prereg_v25a.FAMILYWISE_ALPHA / len(prereg_v25a.ENDPOINTS)
    endpoints = {}
    for name, replicate_values in deltas.items():
        if replicate_values.shape != (repetitions,) or not np.isfinite(replicate_values).all():
            raise RuntimeError("v25a bootstrap replicate coverage changed")
        observed_delta = (
            observed["candidate_v364"]["endpoint_values"][name]
            - observed["production"]["endpoint_values"][name]
        )
        endpoints[name] = {
            "candidate_v364_minus_production": float(observed_delta),
            "familywise_lcb": float(np.quantile(
                replicate_values,
                quantile,
                method=BOOTSTRAP_QUANTILE_METHOD_V25A,
            )),
            "noninferiority_margin": 0.0,
        }
    return {
        "seed": prereg_v25a.BOOTSTRAP_SEED,
        "repetitions": repetitions,
        "one_sided_quantile": quantile,
        "draw_plan_sha256": draw_plan_sha256,
        "endpoints": endpoints,
        "raw_draws_or_replicates_persisted": False,
    }


def paired_stratified_bootstrap_v25a(unit_scores, panel_bundle):
    result = _paired_stratified_bootstrap_impl(
        unit_scores,
        panel_bundle,
        prereg_v25a.BOOTSTRAP_REPETITIONS,
        EXPECTED_DRAW_PLAN_SHA256_V25A,
    )
    if (
        result["repetitions"] != 50_000
        or result["draw_plan_sha256"] != EXPECTED_DRAW_PLAN_SHA256_V25A
        or set(result["endpoints"]) != set(prereg_v25a.ENDPOINTS)
    ):
        raise RuntimeError("v25a compact bootstrap output changed")
    return result


def build_compact_estimator_summary_v25a(unit_scores, panel_bundle):
    observed = recompute_observed_endpoints_v25a(unit_scores, panel_bundle)
    return {
        "versions": copy.deepcopy(observed["versions"]),
        "paired_bootstrap": paired_stratified_bootstrap_v25a(
            unit_scores, panel_bundle,
        ),
        "cross_version_direction_similarity_diagnostic": copy.deepcopy(
            observed["cross_version_direction_similarity_diagnostic"]
        ),
        "persisted_response_vectors_rows_draws_or_replicates": False,
    }


def evaluate_candidate_v25a(summary):
    if (
        not isinstance(summary, dict)
        or summary.get("runtime_integrity", {}).get("all_integrity_audits_passed")
        is not True
        or set(summary.get("paired_bootstrap", {}).get("endpoints", {}))
        != set(prereg_v25a.ENDPOINTS)
    ):
        raise RuntimeError("v25a compact candidate summary changed")
    endpoints = summary["paired_bootstrap"]["endpoints"]
    passed = all(
        item.get("noninferiority_margin") == 0.0
        and isinstance(item.get("familywise_lcb"), float)
        and math.isfinite(item["familywise_lcb"])
        and item["familywise_lcb"] >= 0.0
        for item in endpoints.values()
    )
    result = {
        "schema": "eggroll-es-paired-data-compat-gate-v25a",
        "all_12_familywise_lcbs_nonnegative": passed,
        "all_runtime_integrity_audits_passed": True,
        "pass": passed,
        "decision": (
            "authorize_only_separate_full-v364_train-only_HPO_preregistration"
            if passed
            else "retain_production_dataset_and_v13_recipe"
        ),
        "dataset_promotion_authorized": False,
        "model_update_authorized": False,
        "checkpoint_write_authorized": False,
        "evaluation_authorized": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result
