#!/usr/bin/env python3
"""Pure train-only mechanics for the immutable V33A paired data scan."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import build_eggroll_es_joint_panels_v33a as frame_v33a
import eggroll_es_paired_data_compat_preregistration_v33a as prereg_v33a
import eggroll_es_train_panel_sampler_v13 as sampler_v13
import train_eggroll_es_specialist_anchor_v13 as anchor_v13


ROOT = Path(__file__).resolve().parent
CANONICAL_ROOT = Path("/home/catid/specialist")
PREREGISTRATION_FILE_SHA256_V33A = (
    "c83e11376922ac273b5b6496ef60126a2cdc6ae044f6a9ab05d9481dc539bcda"
)
PREREGISTRATION_CONTENT_SHA256_V33A = (
    "f053c686721401d08b31f2619ba23511159e0a85cfbda2df606e0b00fa98bc61"
)
PANEL_BUNDLE_CONTENT_SHA256_V33A = (
    "26546dd89dfe1967f0a5482b234cb3dd39a3e0f2bf23754fca18e6d6512185f1"
)
CANDIDATE_PATH_V33A = (
    CANONICAL_ROOT / "experiments/eggroll_es_hpo/dataset_candidates/v364/"
    "train_qa_context_merit_v364.jsonl"
)
PRODUCTION_PATH_V33A = CANONICAL_ROOT / "data/train_qa_curated_v1.jsonl"
VERSIONS_V33A = ("production", "candidate_v364")
VERSION_TO_FRAME_SIDE_V33A = {
    "production": "production",
    "candidate_v364": "candidate",
}
PANEL_NAMES_V33A = frame_v33a.PANEL_NAMES
OPTIMIZATION_PANELS_V33A = frame_v33a.OPTIMIZATION_PANELS
TRAIN_SCREENS_V33A = frame_v33a.TRAIN_SCREENS
STRATA_V33A = tuple(sampler_v13.STRATA)
SIGNS_V33A = ("plus", "minus")
PAIRED_FRAME_POPULATION_V33A = frame_v33a.EXPECTED_PAIRED_UNITS
BOOTSTRAP_CHUNK_SIZE_V33A = 128
BOOTSTRAP_QUANTILE_METHOD_V33A = "linear"
EXPECTED_DRAW_PLAN_SHA256_V33A = (
    "ef44acfe80d9afab5e17621eb62acc09572a6c3486f1f0a61dd6848ab6398b37"
)


canonical_sha256 = prereg_v33a.canonical_sha256
file_sha256 = prereg_v33a.file_sha256


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _read_bound_rows(path, expected_sha256, expected_rows):
    path = Path(path).resolve()
    if file_sha256(path) != expected_sha256:
        raise RuntimeError("v33a paired source file identity changed")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected_rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("v33a paired source row count or schema changed")
    return rows


def load_hardened_preregistration_v33a():
    if (
        file_sha256(prereg_v33a.PREREGISTRATION_PATH)
        != PREREGISTRATION_FILE_SHA256_V33A
    ):
        raise RuntimeError("v33a preregistration file changed")
    persisted = json.loads(
        prereg_v33a.PREREGISTRATION_PATH.read_text(encoding="utf-8")
    )
    if (
        persisted.get("content_sha256_before_self_field")
        != PREREGISTRATION_CONTENT_SHA256_V33A
        or persisted.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(persisted))
        or persisted.get("required_runtime_adapter", {}).get("status")
        != "not_yet_implemented"
        or persisted.get("required_runtime_adapter", {}).get(
            "runtime_launch_authorized"
        ) is not False
    ):
        raise RuntimeError("v33a preregistration content changed")
    prereg_v33a.validate_sha_fields(persisted)
    return persisted


def _materialize_paired_panel_bundle_v33a():
    preregistration = load_hardened_preregistration_v33a()
    aggregate, _prior_evidence, _layer_plan = prereg_v33a.load_bound_evidence()
    frame = frame_v33a.build_runtime_manifest_v33a()
    if frame_v33a._aggregate_from_runtime_v33a(frame) != aggregate:
        raise RuntimeError("v33a transient frame does not match sealed aggregate")
    sources = {
        "production": _read_bound_rows(
            PRODUCTION_PATH_V33A,
            frame_v33a.PRODUCTION_SHA256,
            frame_v33a.PRODUCTION_ROWS,
        ),
        "candidate_v364": _read_bound_rows(
            CANDIDATE_PATH_V33A,
            frame_v33a.CANDIDATE_SHA256,
            frame_v33a.CANDIDATE_ROWS,
        ),
    }
    panels = {}
    anchor_counts = {"shared_document": 0, "joint_component_cross_side_link": 0}
    for manifest_panel in frame["panels"]:
        panel_name = manifest_panel["name"]
        items = manifest_panel["items"]
        version_batches = {}
        for version in VERSIONS_V33A:
            side = VERSION_TO_FRAME_SIDE_V33A[version]
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
                    raise RuntimeError("v33a fixed representative index changed")
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
                    raise RuntimeError("v33a fixed representative content changed")
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
        raise RuntimeError("v33a selected pairing-anchor coverage changed")
    result = {
        "schema": "eggroll-es-materialized-paired-data-batches-v33a",
        "preregistration": {
            "file_sha256": PREREGISTRATION_FILE_SHA256_V33A,
            "content_sha256": preregistration["content_sha256_before_self_field"],
        },
        "frame": {
            "file_sha256": prereg_v33a.FRAME_FILE_SHA256,
            "content_sha256": prereg_v33a.FRAME_CONTENT_SHA256,
            "transient_runtime_frame_content_sha256": frame[
                "content_sha256_before_self_field"
            ],
            "selected_paired_units": 195,
            "reserve_paired_units": 10,
            "shared_document_anchors": 193,
            "joint_component_cross_side_anchors": 2,
        },
        "sources": {
            "production": {
                "rows": frame_v33a.PRODUCTION_ROWS,
                "file_sha256": frame_v33a.PRODUCTION_SHA256,
                "source_commit": frame_v33a.PRODUCTION_SOURCE_COMMIT,
            },
            "candidate_v364": {
                "rows": frame_v33a.CANDIDATE_ROWS,
                "file_sha256": frame_v33a.CANDIDATE_SHA256,
                "freeze_commit": frame_v33a.CANDIDATE_FREEZE_COMMIT,
                "manifest_file_sha256": frame_v33a.CANDIDATE_MANIFEST_SHA256,
                "projection_report_file_sha256": (
                    frame_v33a.CANDIDATE_PROJECTION_SHA256
                ),
                "freeze_test_file_sha256": frame_v33a.CANDIDATE_TEST_SHA256,
            },
        },
        "panels": panels,
        "contains_validation_ood_heldout_or_benchmark_content": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def validate_paired_panel_bundle_v33a(bundle):
    if (
        not isinstance(bundle, dict)
        or set(bundle) != {
            "schema", "preregistration", "frame", "sources", "panels",
            "contains_validation_ood_heldout_or_benchmark_content",
            "content_sha256_before_self_field",
        }
        or bundle.get("schema")
        != "eggroll-es-materialized-paired-data-batches-v33a"
        or bundle.get("contains_validation_ood_heldout_or_benchmark_content")
        is not False
        or bundle.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(bundle))
        or bundle.get("content_sha256_before_self_field")
        != PANEL_BUNDLE_CONTENT_SHA256_V33A
        or bundle.get("preregistration") != {
            "file_sha256": PREREGISTRATION_FILE_SHA256_V33A,
            "content_sha256": PREREGISTRATION_CONTENT_SHA256_V33A,
        }
        or bundle.get("frame") != {
            "file_sha256": prereg_v33a.FRAME_FILE_SHA256,
            "content_sha256": prereg_v33a.FRAME_CONTENT_SHA256,
            "transient_runtime_frame_content_sha256": (
                frame_v33a.EXPECTED_RUNTIME_FRAME_CONTENT_SHA256
            ),
            "selected_paired_units": 195,
            "reserve_paired_units": 10,
            "shared_document_anchors": 193,
            "joint_component_cross_side_anchors": 2,
        }
        or bundle.get("sources") != {
            "production": {
                "rows": frame_v33a.PRODUCTION_ROWS,
                "file_sha256": frame_v33a.PRODUCTION_SHA256,
                "source_commit": frame_v33a.PRODUCTION_SOURCE_COMMIT,
            },
            "candidate_v364": {
                "rows": frame_v33a.CANDIDATE_ROWS,
                "file_sha256": frame_v33a.CANDIDATE_SHA256,
                "freeze_commit": frame_v33a.CANDIDATE_FREEZE_COMMIT,
                "manifest_file_sha256": frame_v33a.CANDIDATE_MANIFEST_SHA256,
                "projection_report_file_sha256": (
                    frame_v33a.CANDIDATE_PROJECTION_SHA256
                ),
                "freeze_test_file_sha256": frame_v33a.CANDIDATE_TEST_SHA256,
            },
        }
        or tuple(bundle.get("panels", {})) != PANEL_NAMES_V33A
    ):
        raise RuntimeError("v33a materialized paired panel bundle changed")
    aggregate = json.loads(frame_v33a.OUTPUT_PATH.read_text(encoding="utf-8"))
    frame_v33a.validate_manifest(aggregate)
    frame = frame_v33a.build_runtime_manifest_v33a()
    if frame_v33a._aggregate_from_runtime_v33a(frame) != aggregate:
        raise RuntimeError("v33a transient validation frame changed")
    manifest_panels = {item["name"]: item for item in frame["panels"]}
    anchor_counts = {"shared_document": 0, "joint_component_cross_side_link": 0}
    selected_ids = []
    for panel_name in PANEL_NAMES_V33A:
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
            or set(panel.get("versions", {})) != set(VERSIONS_V33A)
            or not math.isclose(
                math.fsum(panel["weights"]),
                float(PAIRED_FRAME_POPULATION_V33A),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError(f"v33a materialized {panel_name} contract changed")
        for anchor in panel["pairing_anchors"]:
            anchor_counts[anchor] += 1
        for version in VERSIONS_V33A:
            side = VERSION_TO_FRAME_SIDE_V33A[version]
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
                ] != [frame_v33a.PANEL_SIZE] * 3
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
                    f"v33a materialized {panel_name} {version} batch changed"
                )
    if (
        len(selected_ids) != 195
        or len(set(selected_ids)) != 195
        or anchor_counts != {
            "shared_document": 193,
            "joint_component_cross_side_link": 2,
        }
    ):
        raise RuntimeError("v33a selected frame identity changed")
    return bundle


def load_paired_panel_bundle_v33a():
    return validate_paired_panel_bundle_v33a(
        _materialize_paired_panel_bundle_v33a()
    )


def resident_signed_wave_schedule_v33a():
    frozen = load_hardened_preregistration_v33a()
    base_schedule = frozen["frozen_recipe"]["perturbation_basis"][
        "signed_population_schedule"
    ]
    if base_schedule != prereg_v33a.signed_population_schedule():
        raise RuntimeError("v33a signed population schedule changed")
    result = []
    for item in base_schedule:
        signed_index = item["signed_wave_index"]
        order = (
            list(VERSIONS_V33A)
            if signed_index % 2 == 0
            else list(reversed(VERSIONS_V33A))
        )
        result.append({
            **copy.deepcopy(item),
            "resident_version_order": order,
            "restore_after_both_versions": True,
        })
    if (
        len(result) != 32
        or [item["signed_wave_index"] for item in result] != list(range(32))
        or any(set(item["resident_version_order"]) != set(VERSIONS_V33A) for item in result)
        or sum(item["resident_version_order"][0] == "production" for item in result)
        != 16
    ):
        raise RuntimeError("v33a resident version-order balance changed")
    return result


def execute_paired_resident_signed_wave_v33a(
    schedule_item, *, perturb, score_version, restore,
):
    schedule = resident_signed_wave_schedule_v33a()
    index = schedule_item.get("signed_wave_index") if isinstance(schedule_item, dict) else None
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < 32
        or schedule_item != schedule[index]
    ):
        raise RuntimeError("v33a resident signed-wave schedule item changed")
    if not all(callable(value) for value in (perturb, score_version, restore)):
        raise TypeError("v33a resident signed-wave callbacks must be callable")
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
        raise RuntimeError("v33a resident paired-version capture is incomplete")
    return captures


def _unit_score_array(unit_scores):
    values = np.asarray(unit_scores, dtype=np.float64)
    expected = (2, 5, 2, 64, 39)
    if values.shape != expected or not np.isfinite(values).all():
        raise RuntimeError("v33a paired per-unit score tensor changed")
    return values


def _panel_design(panel_bundle):
    panel_bundle = validate_paired_panel_bundle_v33a(panel_bundle)
    weights = np.asarray([
        panel_bundle["panels"][name]["weights"] for name in PANEL_NAMES_V33A
    ], dtype=np.float64)
    strata = [
        list(panel_bundle["panels"][name]["strata"])
        for name in PANEL_NAMES_V33A
    ]
    if weights.shape != (5, 39):
        raise RuntimeError("v33a paired panel design changed")
    return weights, strata


def observed_panel_scores_v33a(unit_scores, panel_bundle):
    values = _unit_score_array(unit_scores)
    weights, _strata = _panel_design(panel_bundle)
    panel_scores = np.einsum("vpsdu,pu->vpsd", values, weights)
    panel_scores /= float(PAIRED_FRAME_POPULATION_V33A)
    if panel_scores.shape != (2, 5, 2, 64) or not np.isfinite(panel_scores).all():
        raise RuntimeError("v33a observed panel-score recomputation changed")
    return panel_scores


def _cosine_last_axis(left, right):
    numerator = np.sum(left * right, axis=-1)
    denominator = np.sqrt(
        np.sum(left * left, axis=-1) * np.sum(right * right, axis=-1)
    )
    if np.any(denominator == 0.0):
        raise RuntimeError("v33a endpoint cosine received a zero direction")
    return numerator / denominator


def _sign_agreement_last_axis(left, right):
    return np.mean(np.sign(left) == np.sign(right), axis=-1)


def _endpoint_arrays(panel_scores):
    values = np.asarray(panel_scores, dtype=np.float64)
    if (
        values.ndim != 5
        or values.shape[0] != 2
        or values.shape[2:] != (5, 2, 64)
        or not np.isfinite(values).all()
    ):
        raise RuntimeError("v33a panel-score endpoint tensor changed")
    central = 0.5 * (values[:, :, :, 0, :] - values[:, :, :, 1, :])
    means = np.mean(central, axis=-1, keepdims=True)
    spreads = np.sqrt(np.mean((central - means) ** 2, axis=-1))
    if np.any(spreads == 0.0):
        raise RuntimeError("v33a panel response spread is zero")
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
    for family in prereg_v33a.METRIC_FAMILIES:
        components = families[family]
        endpoints[f"{family}_median"] = np.median(components, axis=-1)
        endpoints[f"{family}_worst"] = np.min(components, axis=-1)
    if set(endpoints) != set(prereg_v33a.ENDPOINTS) or any(
        value.shape != values.shape[:2]
        or not np.isfinite(value).all()
        or np.any(value < -1.0 - 1e-12)
        or np.any(value > 1.0 + 1e-12)
        for value in endpoints.values()
    ):
        raise RuntimeError("v33a endpoint family recomputation changed")
    return {
        "endpoints": endpoints,
        "coefficients": coefficients,
        "aggregate": aggregate,
        "spreads": spreads,
    }


def recompute_observed_endpoints_v33a(unit_scores, panel_bundle):
    panel_scores = observed_panel_scores_v33a(unit_scores, panel_bundle)
    analyzed = _endpoint_arrays(panel_scores[:, np.newaxis, ...])
    versions = {}
    for version_index, version in enumerate(VERSIONS_V33A):
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
                "schema": "eggroll-es-compact-estimator-commitment-v33a",
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
                "schema": "eggroll-es-cross-version-direction-diagnostic-v33a",
                "cosine": cross_cosine,
            }),
        },
    }


def _bootstrap_draw_plan_v33a(panel_bundle, repetitions, expected_sha256):
    _weights, strata_by_panel = _panel_design(panel_bundle)
    generator = np.random.default_rng(prereg_v33a.BOOTSTRAP_SEED)
    digest = hashlib.sha256()
    draws = {}
    for panel_index, panel in enumerate(PANEL_NAMES_V33A):
        draws[panel] = {}
        strata = strata_by_panel[panel_index]
        for stratum in STRATA_V33A:
            positions = np.asarray([
                index for index, value in enumerate(strata) if value == stratum
            ], dtype=np.int64)
            quota = frame_v33a.STRATUM_QUOTAS[stratum]
            if len(positions) != quota:
                raise RuntimeError("v33a bootstrap stratum quota changed")
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
        raise RuntimeError("v33a bootstrap draw-plan identity changed")
    return draws, actual


def _paired_stratified_bootstrap_impl(
    unit_scores, panel_bundle, repetitions, expected_draw_plan_sha256,
):
    values = _unit_score_array(unit_scores)
    observed = recompute_observed_endpoints_v33a(values, panel_bundle)["versions"]
    draw_plan, draw_plan_sha256 = _bootstrap_draw_plan_v33a(
        panel_bundle, repetitions, expected_draw_plan_sha256,
    )
    deltas = {name: np.empty(repetitions, dtype=np.float64) for name in prereg_v33a.ENDPOINTS}
    for start in range(0, repetitions, BOOTSTRAP_CHUNK_SIZE_V33A):
        stop = min(start + BOOTSTRAP_CHUNK_SIZE_V33A, repetitions)
        batch_size = stop - start
        panel_scores = np.zeros((2, batch_size, 5, 2, 64), dtype=np.float64)
        for panel_index, panel in enumerate(PANEL_NAMES_V33A):
            for stratum in STRATA_V33A:
                plan = draw_plan[panel][stratum]
                stratum_scores = np.take(
                    values[:, panel_index], plan["positions"], axis=-1,
                )
                resampled = np.take(
                    stratum_scores, plan["draw"][start:stop], axis=-1,
                )
                stratum_means = np.mean(resampled, axis=-1).transpose(0, 3, 1, 2)
                population = frame_v33a.EXPECTED_PAIRED_STRATA[stratum]
                panel_scores[:, :, panel_index] += (
                    float(population) / float(PAIRED_FRAME_POPULATION_V33A)
                ) * stratum_means
        analyzed = _endpoint_arrays(panel_scores)
        for name, endpoint_values in analyzed["endpoints"].items():
            deltas[name][start:stop] = endpoint_values[1] - endpoint_values[0]
    quantile = prereg_v33a.FAMILYWISE_ALPHA / len(prereg_v33a.ENDPOINTS)
    endpoints = {}
    for name, replicate_values in deltas.items():
        if replicate_values.shape != (repetitions,) or not np.isfinite(replicate_values).all():
            raise RuntimeError("v33a bootstrap replicate coverage changed")
        observed_delta = (
            observed["candidate_v364"]["endpoint_values"][name]
            - observed["production"]["endpoint_values"][name]
        )
        endpoints[name] = {
            "candidate_v364_minus_production": float(observed_delta),
            "familywise_lcb": float(np.quantile(
                replicate_values,
                quantile,
                method=BOOTSTRAP_QUANTILE_METHOD_V33A,
            )),
            "noninferiority_margin": 0.0,
        }
    return {
        "seed": prereg_v33a.BOOTSTRAP_SEED,
        "repetitions": repetitions,
        "one_sided_quantile": quantile,
        "draw_plan_sha256": draw_plan_sha256,
        "endpoints": endpoints,
        "raw_draws_or_replicates_persisted": False,
    }


def paired_stratified_bootstrap_v33a(unit_scores, panel_bundle):
    result = _paired_stratified_bootstrap_impl(
        unit_scores,
        panel_bundle,
        prereg_v33a.BOOTSTRAP_REPETITIONS,
        EXPECTED_DRAW_PLAN_SHA256_V33A,
    )
    if (
        result["repetitions"] != 50_000
        or result["draw_plan_sha256"] != EXPECTED_DRAW_PLAN_SHA256_V33A
        or set(result["endpoints"]) != set(prereg_v33a.ENDPOINTS)
    ):
        raise RuntimeError("v33a compact bootstrap output changed")
    return result


def build_compact_estimator_summary_v33a(unit_scores, panel_bundle):
    observed = recompute_observed_endpoints_v33a(unit_scores, panel_bundle)
    return {
        "versions": copy.deepcopy(observed["versions"]),
        "paired_bootstrap": paired_stratified_bootstrap_v33a(
            unit_scores, panel_bundle,
        ),
        "cross_version_direction_similarity_diagnostic": copy.deepcopy(
            observed["cross_version_direction_similarity_diagnostic"]
        ),
        "persisted_response_vectors_rows_draws_or_replicates": False,
    }


def evaluate_candidate_v33a(summary):
    if (
        not isinstance(summary, dict)
        or summary.get("runtime_integrity", {}).get("all_integrity_audits_passed")
        is not True
        or set(summary.get("paired_bootstrap", {}).get("endpoints", {}))
        != set(prereg_v33a.ENDPOINTS)
    ):
        raise RuntimeError("v33a compact candidate summary changed")
    endpoints = summary["paired_bootstrap"]["endpoints"]
    all_lcbs_nonnegative = all(
        item.get("noninferiority_margin") == 0.0
        and isinstance(item.get("familywise_lcb"), float)
        and math.isfinite(item["familywise_lcb"])
        and item["familywise_lcb"] >= 0.0
        for item in endpoints.values()
    )
    all_point_deltas_nonnegative = all(
        isinstance(item.get("candidate_v364_minus_production"), float)
        and math.isfinite(item["candidate_v364_minus_production"])
        and item["candidate_v364_minus_production"] >= 0.0
        for item in endpoints.values()
    )
    passed = all_lcbs_nonnegative and all_point_deltas_nonnegative
    result = {
        "schema": "eggroll-es-paired-data-compat-gate-v33a",
        "all_12_familywise_lcbs_nonnegative": all_lcbs_nonnegative,
        "all_12_observed_point_deltas_nonnegative": all_point_deltas_nonnegative,
        "all_runtime_integrity_audits_passed": True,
        "pass": passed,
        "decision": (
            "authorize_only_separate_fresh-basis-v364_confirmation_preregistration"
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
