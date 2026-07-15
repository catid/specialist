#!/usr/bin/env python3
"""Pure trainer-side mechanics for the V17A paired data compatibility run.

This module materializes the frozen train-only paired batches and implements
the preregistered estimator analysis.  It deliberately has no engine launch,
model update, checkpoint, evaluation, or persistence entrypoint.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

import build_eggroll_es_joint_panels_v17a as frame_v17a
import eggroll_es_paired_data_compat_preregistration_v17a as prereg_v17a
import eggroll_es_train_panel_sampler_v13 as sampler_v13
import train_eggroll_es_specialist_anchor_v13 as anchor_v13


ROOT = Path(__file__).resolve().parent
ARMS_V17A = prereg_v17a.ARM_ORDER_V17A
PANEL_NAMES_V17A = frame_v17a.PANEL_NAMES_V17A
OPTIMIZATION_PANELS_V17A = frame_v17a.OPTIMIZATION_PANELS_V17A
TRAIN_SCREENS_V17A = frame_v17a.TRAIN_SCREENS_V17A
SIGNS_V17A = ("plus", "minus")
STRATA_V17A = tuple(sampler_v13.STRATA)
ARM_TO_FRAME_SIDE_V17A = {
    "production": "production",
    "candidate_v283": "candidate",
}
PREREGISTRATION_FILE_SHA256_V17A = (
    "85a30be591f72376e220447ce9f1be0d04919b2855a987b757d0d71bd90fba1f"
)
PREREGISTRATION_CONTENT_SHA256_V17A = (
    "c324282a0e7151103fa90f6e724119a4eaf90d800ded2ff8ad57a008d3a63440"
)
PAIRED_PANEL_BUNDLE_CONTENT_SHA256_V17A = (
    "9820a7887c576e65fcbd1d2fb0ead0d52944b5ccfa4beca34b7fb4d7e39f4990"
)
BOOTSTRAP_CHUNK_SIZE_V17A = 256
BOOTSTRAP_QUANTILE_METHOD_V17A = "linear"
PAIRED_FRAME_POPULATION_V17A = frame_v17a.EXPECTED_PAIRED_UNITS_V17A

canonical_sha256 = prereg_v17a.canonical_sha256
file_sha256 = prereg_v17a.file_sha256


def _without_self_v17a(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _read_bound_jsonl_v17a(path, expected_sha256, expected_rows):
    path = Path(path).resolve()
    if file_sha256(path) != expected_sha256:
        raise RuntimeError("v17a paired source file identity changed")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected_rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("v17a paired source row count or schema changed")
    return rows


def _load_hardened_preregistration_v17a():
    if (
        file_sha256(prereg_v17a.PREREGISTRATION_PATH_V17A)
        != PREREGISTRATION_FILE_SHA256_V17A
    ):
        raise RuntimeError("v17a hardened preregistration file changed")
    persisted = json.loads(
        prereg_v17a.PREREGISTRATION_PATH_V17A.read_text(encoding="utf-8")
    )
    if (
        persisted != prereg_v17a.build_preregistration_v17a()
        or persisted.get("content_sha256_before_self_field")
        != PREREGISTRATION_CONTENT_SHA256_V17A
        or persisted.get("required_runtime_adapter", {}).get(
            "summary_diagnostic_exact_keys"
        ) != ["used_for_gate", "content_sha256"]
        or persisted.get("required_runtime_adapter", {}).get(
            "all_persisted_hash_fields_lowercase_hex"
        ) is not True
    ):
        raise RuntimeError("v17a hardened preregistration content changed")
    return persisted


def _materialize_paired_panel_bundle_v17a():
    """Build the raw in-memory batch; callers must validate before use."""
    preregistration = _load_hardened_preregistration_v17a()
    frame, _candidate_manifest, _v13 = (
        prereg_v17a.load_bound_aggregate_evidence_v17a()
    )
    sources = {
        "production": _read_bound_jsonl_v17a(
            frame_v17a.PRODUCTION_PATH_V17A,
            frame_v17a.PRODUCTION_SHA256_V17A,
            frame_v17a.PRODUCTION_ROWS_V17A,
        ),
        "candidate_v283": _read_bound_jsonl_v17a(
            frame_v17a.CANDIDATE_PATH_V17A,
            frame_v17a.CANDIDATE_SHA256_V17A,
            frame_v17a.CANDIDATE_ROWS_V17A,
        ),
    }
    panels = {}
    for manifest_panel in frame["panels"]:
        panel_name = manifest_panel["name"]
        items = manifest_panel["items"]
        arm_batches = {}
        for arm in ARMS_V17A:
            side = ARM_TO_FRAME_SIDE_V17A[arm]
            source = sources[arm]
            rows = []
            row_hashes = []
            for item in items:
                representative = item["sides"][side]
                row_index = representative["row_index"]
                if isinstance(row_index, bool) or not isinstance(row_index, int):
                    raise RuntimeError("v17a paired representative index changed")
                if not 0 <= row_index < len(source):
                    raise RuntimeError("v17a paired representative index is invalid")
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
                    raise RuntimeError("v17a fixed representative content changed")
                rows.append(row)
                row_hashes.append(row_hash)
            arm_batches[arm] = {
                "ordered_row_identity_sha256": canonical_sha256(row_hashes),
                "row_sha256": row_hashes,
                "questions": [row["question"] for row in rows],
                "answers": [row["answer"] for row in rows],
                "raw_prompt_answer_sha256": canonical_sha256({
                    "questions": [row["question"] for row in rows],
                    "answers": [row["answer"] for row in rows],
                }),
            }
        panels[panel_name] = {
            "name": panel_name,
            "role": manifest_panel["role"],
            "ordered_unit_identity_sha256": manifest_panel[
                "ordered_unit_identity_sha256"
            ],
            "unit_ids": [item["unit_id"] for item in items],
            "strata": [item["stratum"] for item in items],
            "weights": [
                float(item["horvitz_thompson_unit_weight"])
                for item in items
            ],
            "arms": arm_batches,
        }
    result = {
        "schema": "eggroll-es-materialized-paired-data-batches-v17a",
        "preregistration": {
            "file_sha256": PREREGISTRATION_FILE_SHA256_V17A,
            "content_sha256": preregistration[
                "content_sha256_before_self_field"
            ],
        },
        "frame": {
            "file_sha256": prereg_v17a.FRAME_FILE_SHA256_V17A,
            "content_sha256": prereg_v17a.FRAME_CONTENT_SHA256_V17A,
        },
        "sources": {
            "production": {
                "rows": frame_v17a.PRODUCTION_ROWS_V17A,
                "file_sha256": frame_v17a.PRODUCTION_SHA256_V17A,
            },
            "candidate_v283": {
                "rows": frame_v17a.CANDIDATE_ROWS_V17A,
                "file_sha256": frame_v17a.CANDIDATE_SHA256_V17A,
            },
        },
        "panels": panels,
        "contains_evaluation_content": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def validate_paired_panel_bundle_v17a(bundle):
    if (
        not isinstance(bundle, dict)
        or set(bundle) != {
            "schema", "preregistration", "frame", "sources", "panels",
            "contains_evaluation_content", "content_sha256_before_self_field",
        }
        or bundle.get("schema")
        != "eggroll-es-materialized-paired-data-batches-v17a"
        or bundle.get("contains_evaluation_content") is not False
        or bundle.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self_v17a(bundle))
        or bundle.get("content_sha256_before_self_field")
        != PAIRED_PANEL_BUNDLE_CONTENT_SHA256_V17A
        or bundle.get("preregistration") != {
            "file_sha256": PREREGISTRATION_FILE_SHA256_V17A,
            "content_sha256": PREREGISTRATION_CONTENT_SHA256_V17A,
        }
        or bundle.get("frame") != {
            "file_sha256": prereg_v17a.FRAME_FILE_SHA256_V17A,
            "content_sha256": prereg_v17a.FRAME_CONTENT_SHA256_V17A,
        }
        or bundle.get("sources") != {
            "production": {
                "rows": frame_v17a.PRODUCTION_ROWS_V17A,
                "file_sha256": frame_v17a.PRODUCTION_SHA256_V17A,
            },
            "candidate_v283": {
                "rows": frame_v17a.CANDIDATE_ROWS_V17A,
                "file_sha256": frame_v17a.CANDIDATE_SHA256_V17A,
            },
        }
        or tuple(bundle.get("panels", {})) != PANEL_NAMES_V17A
    ):
        raise RuntimeError("v17a materialized paired panel bundle changed")
    frame = json.loads(frame_v17a.OUTPUT_PATH_V17A.read_text(encoding="utf-8"))
    frame_v17a.validate_manifest_v17a(frame)
    manifest_panels = {item["name"]: item for item in frame["panels"]}
    for panel_name in PANEL_NAMES_V17A:
        panel = bundle["panels"][panel_name]
        manifest_panel = manifest_panels[panel_name]
        items = manifest_panel["items"]
        if (
            set(panel) != {
                "name", "role", "ordered_unit_identity_sha256", "unit_ids",
                "strata", "weights", "arms",
            }
            or panel.get("name") != panel_name
            or panel.get("role") != manifest_panel["role"]
            or panel.get("unit_ids") != [item["unit_id"] for item in items]
            or panel.get("strata") != [item["stratum"] for item in items]
            or panel.get("weights") != [
                float(item["horvitz_thompson_unit_weight"])
                for item in items
            ]
            or canonical_sha256(panel["unit_ids"])
            != panel.get("ordered_unit_identity_sha256")
            or panel.get("ordered_unit_identity_sha256")
            != manifest_panel["ordered_unit_identity_sha256"]
            or set(panel.get("arms", {})) != set(ARMS_V17A)
            or not math.isclose(
                math.fsum(panel["weights"]),
                float(PAIRED_FRAME_POPULATION_V17A),
                rel_tol=0.0, abs_tol=1e-12,
            )
        ):
            raise RuntimeError(f"v17a materialized {panel_name} contract changed")
        for arm in ARMS_V17A:
            side = ARM_TO_FRAME_SIDE_V17A[arm]
            arm_batch = panel["arms"][arm]
            expected_hashes = [item["sides"][side]["row_sha256"] for item in items]
            lengths = [
                len(arm_batch.get(key, []))
                for key in ("row_sha256", "questions", "answers")
            ]
            if (
                set(arm_batch) != {
                    "ordered_row_identity_sha256", "row_sha256", "questions",
                    "answers", "raw_prompt_answer_sha256",
                }
                or lengths != [frame_v17a.PANEL_SIZE_V17A] * 3
                or arm_batch["row_sha256"] != expected_hashes
                or canonical_sha256(expected_hashes)
                != arm_batch["ordered_row_identity_sha256"]
                or arm_batch["ordered_row_identity_sha256"]
                != manifest_panel["ordered_side_row_identity_sha256"][side]
                or canonical_sha256({
                    "questions": arm_batch["questions"],
                    "answers": arm_batch["answers"],
                }) != arm_batch["raw_prompt_answer_sha256"]
            ):
                raise RuntimeError(
                    f"v17a materialized {panel_name} {arm} batch changed"
                )
    return bundle


def load_paired_panel_bundle_v17a():
    return validate_paired_panel_bundle_v17a(
        _materialize_paired_panel_bundle_v17a()
    )


def resident_signed_wave_schedule_v17a(seeds=None):
    """Return the only allowed perturbation/sign/paired-arm execution order."""
    seeds = list(anchor_v13.PERTURBATION_SEEDS_V13 if seeds is None else seeds)
    if seeds != anchor_v13.PERTURBATION_SEEDS_V13:
        raise RuntimeError("v17a fixed perturbation basis changed")
    schedule = []
    for wave_index, start in enumerate(range(0, len(seeds), 4)):
        wave = [int(seed) for seed in seeds[start:start + 4]]
        if len(wave) != 4:
            raise RuntimeError("v17a partial four-engine wave is forbidden")
        for sign_index, sign in enumerate(SIGNS_V17A):
            signed_wave_index = 2 * wave_index + sign_index
            order = (
                list(ARMS_V17A)
                if signed_wave_index % 2 == 0
                else list(reversed(ARMS_V17A))
            )
            schedule.append({
                "signed_wave_index": signed_wave_index,
                "population_wave_index": wave_index,
                "sign": sign,
                "negate": sign == "minus",
                "engine_seeds": wave,
                "resident_arm_order": order,
                "restore_after_both_arms": True,
            })
    if (
        len(schedule) != 16
        or [item["signed_wave_index"] for item in schedule] != list(range(16))
        or any(set(item["resident_arm_order"]) != set(ARMS_V17A) for item in schedule)
        or sum(item["resident_arm_order"][0] == "production" for item in schedule)
        != 8
    ):
        raise RuntimeError("v17a resident paired-arm balance changed")
    return schedule


def execute_paired_resident_signed_wave_v17a(
    schedule_item, *, perturb, score_arm, restore,
):
    """Execute one perturb-once, score-both-arms, restore-once transaction.

    The callbacks keep this unit GPU-free while making the eventual engine
    adapter reuse one fail-closed ordering primitive.
    """
    if not isinstance(schedule_item, dict):
        raise RuntimeError("v17a resident signed-wave schedule item changed")
    signed_wave_index = schedule_item.get("signed_wave_index")
    if (
        isinstance(signed_wave_index, bool)
        or not isinstance(signed_wave_index, int)
        or not 0 <= signed_wave_index < 16
        or schedule_item
        != resident_signed_wave_schedule_v17a()[signed_wave_index]
    ):
        raise RuntimeError("v17a resident signed-wave schedule item changed")
    if not all(callable(item) for item in (perturb, score_arm, restore)):
        raise TypeError("v17a resident signed-wave callbacks must be callable")
    captures = {}
    try:
        perturb(
            list(schedule_item["engine_seeds"]),
            bool(schedule_item["negate"]),
        )
        for arm in schedule_item["resident_arm_order"]:
            captures[arm] = score_arm(arm)
    finally:
        restore()
    if tuple(captures) != tuple(schedule_item["resident_arm_order"]):
        raise RuntimeError("v17a resident paired-arm capture is incomplete")
    return captures


def _unit_score_array_v17a(unit_scores):
    values = np.asarray(unit_scores, dtype=np.float64)
    expected = (
        len(ARMS_V17A), len(PANEL_NAMES_V17A), len(SIGNS_V17A),
        prereg_v17a.POPULATION_SIZE_V17A, frame_v17a.PANEL_SIZE_V17A,
    )
    if values.shape != expected or not np.isfinite(values).all():
        raise RuntimeError("v17a paired per-unit score tensor changed")
    return values


def _panel_design_v17a(panel_bundle):
    panel_bundle = validate_paired_panel_bundle_v17a(panel_bundle)
    weights = np.asarray([
        panel_bundle["panels"][name]["weights"] for name in PANEL_NAMES_V17A
    ], dtype=np.float64)
    strata = [
        list(panel_bundle["panels"][name]["strata"])
        for name in PANEL_NAMES_V17A
    ]
    return weights, strata


def observed_panel_scores_v17a(unit_scores, panel_bundle):
    """Recompute each arm/panel/sign/direction Horvitz-Thompson mean."""
    values = _unit_score_array_v17a(unit_scores)
    weights, _strata = _panel_design_v17a(panel_bundle)
    panel_scores = np.einsum("apsdu,pu->apsd", values, weights)
    panel_scores /= float(PAIRED_FRAME_POPULATION_V17A)
    if panel_scores.shape != (2, 5, 2, 32) or not np.isfinite(panel_scores).all():
        raise RuntimeError("v17a observed panel-score recomputation changed")
    return panel_scores


def _cosine_last_axis_v17a(left, right):
    numerator = np.sum(left * right, axis=-1)
    denominator = np.sqrt(
        np.sum(left * left, axis=-1) * np.sum(right * right, axis=-1)
    )
    if np.any(denominator == 0.0):
        raise RuntimeError("v17a endpoint cosine received a zero direction")
    return numerator / denominator


def _sign_agreement_last_axis_v17a(left, right):
    return np.mean(np.sign(left) == np.sign(right), axis=-1)


def _endpoint_arrays_v17a(panel_scores):
    """Compute all 12 endpoints for [arm, replicate, panel, sign, direction]."""
    values = np.asarray(panel_scores, dtype=np.float64)
    if (
        values.ndim != 5
        or values.shape[0] != 2
        or values.shape[2:] != (5, 2, 32)
        or not np.isfinite(values).all()
    ):
        raise RuntimeError("v17a panel-score endpoint tensor changed")
    central = 0.5 * (values[:, :, :, 0, :] - values[:, :, :, 1, :])
    means = np.mean(central, axis=-1, keepdims=True)
    spreads = np.sqrt(np.mean((central - means) ** 2, axis=-1))
    if np.any(spreads == 0.0):
        raise RuntimeError("v17a panel response spread is zero")
    coefficients = (central - means) / (
        spreads[..., np.newaxis] + anchor_v13.STANDARDIZATION_EPSILON_V13
    )
    aggregate = np.median(coefficients[:, :, :3, :], axis=2)
    optimization_pairs = ((0, 1), (0, 2), (1, 2))
    families = {
        "optimization_pairwise_cosine": np.stack([
            _cosine_last_axis_v17a(
                coefficients[:, :, left, :], coefficients[:, :, right, :]
            )
            for left, right in optimization_pairs
        ], axis=-1),
        "optimization_pairwise_sign_agreement": np.stack([
            _sign_agreement_last_axis_v17a(
                coefficients[:, :, left, :], coefficients[:, :, right, :]
            )
            for left, right in optimization_pairs
        ], axis=-1),
        "aggregate_to_optimization_cosine": np.stack([
            _cosine_last_axis_v17a(aggregate, coefficients[:, :, index, :])
            for index in range(3)
        ], axis=-1),
        "aggregate_to_optimization_sign_agreement": np.stack([
            _sign_agreement_last_axis_v17a(
                aggregate, coefficients[:, :, index, :]
            )
            for index in range(3)
        ], axis=-1),
        "train_screen_cosine": np.stack([
            _cosine_last_axis_v17a(aggregate, coefficients[:, :, index, :])
            for index in (3, 4)
        ], axis=-1),
        "train_screen_sign_agreement": np.stack([
            _sign_agreement_last_axis_v17a(
                aggregate, coefficients[:, :, index, :]
            )
            for index in (3, 4)
        ], axis=-1),
    }
    endpoints = {}
    for family in prereg_v17a.METRIC_FAMILIES_V17A:
        components = families[family]
        endpoints[f"{family}_median"] = np.median(components, axis=-1)
        endpoints[f"{family}_worst"] = np.min(components, axis=-1)
    if set(endpoints) != set(prereg_v17a.ENDPOINT_CONTRACT_V17A) or any(
        value.shape != values.shape[:2]
        or not np.isfinite(value).all()
        or np.any(value < -1.0 - 1e-12)
        or np.any(value > 1.0 + 1e-12)
        for value in endpoints.values()
    ):
        raise RuntimeError("v17a endpoint family recomputation changed")
    return {
        "endpoints": endpoints,
        "coefficients": coefficients,
        "aggregate": aggregate,
        "spreads": spreads,
    }


def recompute_observed_endpoints_v17a(unit_scores, panel_bundle):
    panel_scores = observed_panel_scores_v17a(unit_scores, panel_bundle)
    analyzed = _endpoint_arrays_v17a(panel_scores[:, np.newaxis, ...])
    result = {}
    for arm_index, arm in enumerate(ARMS_V17A):
        endpoint_values = {
            name: float(values[arm_index, 0])
            for name, values in analyzed["endpoints"].items()
        }
        compact_payload = {
            "schema": "eggroll-es-compact-estimator-commitment-v17a",
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
    cross_cosine = float(_cosine_last_axis_v17a(
        analyzed["aggregate"][0, 0], analyzed["aggregate"][1, 0]
    ))
    return {
        "versions": result,
        "cross_dataset_direction_similarity_diagnostic": {
            "used_for_gate": False,
            "content_sha256": canonical_sha256({
                "schema": "eggroll-es-cross-dataset-direction-diagnostic-v17a",
                "cosine": cross_cosine,
            }),
        },
    }


def paired_stratified_bootstrap_v17a(unit_scores, panel_bundle):
    """Run the exact preregistered 20k paired within-panel/stratum bootstrap."""
    values = _unit_score_array_v17a(unit_scores)
    _weights, strata_by_panel = _panel_design_v17a(panel_bundle)
    observed = recompute_observed_endpoints_v17a(values, panel_bundle)["versions"]
    rng = np.random.default_rng(prereg_v17a.BOOTSTRAP_SEED_V17A)
    deltas = {
        name: [] for name in prereg_v17a.ENDPOINT_CONTRACT_V17A
    }
    completed = 0
    while completed < prereg_v17a.BOOTSTRAP_REPETITIONS_V17A:
        batch_size = min(
            BOOTSTRAP_CHUNK_SIZE_V17A,
            prereg_v17a.BOOTSTRAP_REPETITIONS_V17A - completed,
        )
        panel_scores = np.zeros((2, batch_size, 5, 2, 32), dtype=np.float64)
        for panel_index, strata in enumerate(strata_by_panel):
            for stratum in STRATA_V17A:
                positions = np.asarray([
                    index for index, value in enumerate(strata)
                    if value == stratum
                ], dtype=np.int64)
                quota = frame_v17a.STRATUM_QUOTAS_V17A[stratum]
                if len(positions) != quota:
                    raise RuntimeError("v17a bootstrap stratum quota changed")
                draws = rng.integers(
                    0, quota, size=(batch_size, quota), dtype=np.int64,
                )
                stratum_scores = np.take(
                    values[:, panel_index], positions, axis=-1,
                )
                resampled = np.take(stratum_scores, draws, axis=-1)
                stratum_means = np.mean(resampled, axis=-1).transpose(0, 3, 1, 2)
                population = frame_v17a.EXPECTED_PAIRED_STRATA_V17A[stratum]
                panel_scores[:, :, panel_index] += (
                    float(population) / float(PAIRED_FRAME_POPULATION_V17A)
                ) * stratum_means
        analyzed = _endpoint_arrays_v17a(panel_scores)
        for name, endpoint_values in analyzed["endpoints"].items():
            deltas[name].append(endpoint_values[1] - endpoint_values[0])
        completed += batch_size
    quantile = (
        prereg_v17a.FAMILYWISE_ALPHA_V17A
        / len(prereg_v17a.ENDPOINT_CONTRACT_V17A)
    )
    endpoints = {}
    for name, chunks in deltas.items():
        values_delta = np.concatenate(chunks)
        if (
            values_delta.shape != (prereg_v17a.BOOTSTRAP_REPETITIONS_V17A,)
            or not np.isfinite(values_delta).all()
        ):
            raise RuntimeError("v17a bootstrap replicate coverage changed")
        observed_delta = (
            observed["candidate_v283"]["endpoint_values"][name]
            - observed["production"]["endpoint_values"][name]
        )
        endpoints[name] = {
            "candidate_minus_production": float(observed_delta),
            "familywise_lcb": float(np.quantile(
                values_delta, quantile, method=BOOTSTRAP_QUANTILE_METHOD_V17A,
            )),
            "noninferiority_margin": 0.0,
        }
    result = {
        "seed": prereg_v17a.BOOTSTRAP_SEED_V17A,
        "repetitions": prereg_v17a.BOOTSTRAP_REPETITIONS_V17A,
        "one_sided_quantile": quantile,
        "endpoints": endpoints,
    }
    if (
        set(result["endpoints"]) != set(prereg_v17a.ENDPOINT_CONTRACT_V17A)
        or result["repetitions"] != 20_000
    ):
        raise RuntimeError("v17a compact bootstrap output changed")
    return result


def build_compact_estimator_summary_v17a(unit_scores, panel_bundle):
    """Return compact estimator evidence; no unit scores or draws survive."""
    observed = recompute_observed_endpoints_v17a(unit_scores, panel_bundle)
    return {
        "versions": copy.deepcopy(observed["versions"]),
        "paired_bootstrap": paired_stratified_bootstrap_v17a(
            unit_scores, panel_bundle,
        ),
        "cross_dataset_direction_similarity_diagnostic": copy.deepcopy(
            observed["cross_dataset_direction_similarity_diagnostic"]
        ),
        "persisted_response_vectors_or_row_content": False,
    }
