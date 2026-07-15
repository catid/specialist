#!/usr/bin/env python3
"""Pure train-only mechanics for the frozen V34B replacement-fraction HPO."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import build_eggroll_es_v401_replacement_fraction_frame_v34b as frame_v34b
import eggroll_es_train_panel_sampler_v13 as sampler_v13
import eggroll_es_v401_replacement_fraction_preregistration_v34b as prereg_v34b
import train_eggroll_es_specialist_anchor_v13 as anchor_v13


ROOT = Path(__file__).resolve().parent
PREREGISTRATION_FILE_SHA256 = "b852730872621fe9259087dd681ebf8854f985e8caa9208f0e8257a1d07de91b"
PREREGISTRATION_CONTENT_SHA256 = "8ebf4b34b693e459c2b59f331e0768eb116842dbd61c24202302794b3e3b439b"
PREREGISTRATION_MODULE_SHA256 = "5ebb49b951c61e144cda1f6dd06d23b4b1f746cd500a58059f77fc8aec48f41b"
PREREGISTRATION_TEST_SHA256 = "1499e8d5aff293fcdafc14896c89cc3506c691a1468ac39a0b7d15a16296b13d"
PANEL_BUNDLE_CONTENT_SHA256 = (
    "9d9824dfb0051ab8ed39d2a1d01ad22baa77c94305a9c76d434b38c41aca2f6c"
)

SOURCES = ("production", "candidate_v401")
SOURCE_TO_FRAME_SIDE = {"production": "production", "candidate_v401": "candidate"}
PANEL_NAMES = frame_v34b.PANEL_NAMES
STRATA = tuple(sampler_v13.STRATA)
SIGNS = ("plus", "minus")
FRACTIONS = prereg_v34b.REPLACEMENT_FRACTIONS
PAIRED_FRAME_POPULATION = frame_v34b.EXPECTED_PAIRED_UNITS
BOOTSTRAP_CHUNK_SIZE = 128
BOOTSTRAP_QUANTILE_METHOD = "linear"
EXPECTED_DRAW_PLAN_SHA256 = "458d4bdaf9e8f990258712e561699c630ab8e1091f9919979492081c283d5dec"
RUNTIME_INTEGRITY_KEYS = {
    "all_four_tp1_engines_every_signed_wave",
    "all_thirty_two_signed_waves_complete",
    "both_sources_every_direction_and_sign",
    "counterbalanced_source_order_complete",
    "same_resident_perturbation_both_sources",
    "exact_reference_restored_after_each_signed_wave",
    "pre_post_full_context_reference_probes_equal",
    "population_boundary_selected_and_unselected_audits_passed",
    "base_layer_and_unselected_origin_audits_passed",
    "source_and_preregistration_hashes_rechecked",
    "fresh_exclusive_paths_and_committed_clean_source_passed",
    "failure_cleanup_and_final_all_gpu_idle_passed",
    "all_integrity_audits_passed",
}


canonical_sha256 = prereg_v34b.canonical_sha256
file_sha256 = prereg_v34b.file_sha256


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def load_hardened_preregistration() -> dict:
    expected = {
        prereg_v34b.OUTPUT_PATH: PREREGISTRATION_FILE_SHA256,
        Path(prereg_v34b.__file__).resolve(): PREREGISTRATION_MODULE_SHA256,
        ROOT / "test_eggroll_es_v401_replacement_fraction_preregistration_v34b.py": (
            PREREGISTRATION_TEST_SHA256
        ),
    }
    if any(file_sha256(path) != digest for path, digest in expected.items()):
        raise RuntimeError("v34b preregistration bundle changed")
    persisted = json.loads(prereg_v34b.OUTPUT_PATH.read_text(encoding="utf-8"))
    if (
        persisted.get("schema")
        != "eggroll-es-v401-replacement-fraction-preregistration-v34b"
        or persisted.get("content_sha256_before_self_field")
        != PREREGISTRATION_CONTENT_SHA256
        or persisted.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(persisted))
        or persisted.get("frozen_recipe", {}).get("alpha") != 0.0
        or persisted.get("fixed_sequence_gate", {}).get("stop_at_first_failure")
        is not True
    ):
        raise RuntimeError("v34b preregistration semantics changed")
    return persisted


def _read_bound_rows(path: Path, expected_sha256: str, expected_rows: int) -> list[dict]:
    if file_sha256(path) != expected_sha256:
        raise RuntimeError("v34b train source identity changed")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected_rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("v34b train source row count changed")
    return rows


def materialize_paired_panel_bundle() -> dict:
    preregistration = load_hardened_preregistration()
    persisted = json.loads(frame_v34b.OUTPUT_PATH.read_text(encoding="utf-8"))
    frame_v34b.validate_manifest(persisted)
    runtime = frame_v34b.build_runtime_manifest_v34b()
    if frame_v34b.aggregate_from_runtime_v34b(runtime) != persisted:
        raise RuntimeError("v34b transient frame does not match sealed frame")
    rows = {
        "production": _read_bound_rows(
            frame_v34b.PRODUCTION_PATH,
            frame_v34b.PRODUCTION_SHA256,
            frame_v34b.PRODUCTION_ROWS,
        ),
        "candidate_v401": _read_bound_rows(
            frame_v34b.CANDIDATE_PATH,
            frame_v34b.CANDIDATE_SHA256,
            frame_v34b.CANDIDATE_ROWS,
        ),
    }
    panels = {}
    anchor_counts = {"shared_document": 0, "joint_component_cross_side_link": 0}
    for manifest_panel in runtime["panels"]:
        items = manifest_panel["items"]
        versions = {}
        for source_name in SOURCES:
            side = SOURCE_TO_FRAME_SIDE[source_name]
            source_rows = rows[source_name]
            selected = []
            row_hashes = []
            for item in items:
                representative = item["sides"][side]
                index = representative["row_index"]
                if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(source_rows):
                    raise RuntimeError("v34b representative index changed")
                row = source_rows[index]
                row_hash = sampler_v13.row_sha256(row)
                if (
                    row_hash != representative["row_sha256"]
                    or row.get("document_sha256") != representative["document_sha256"]
                    or not isinstance(row.get("question"), str)
                    or not row["question"]
                    or not isinstance(row.get("answer"), str)
                    or not row["answer"]
                ):
                    raise RuntimeError("v34b representative train row changed")
                selected.append(row)
                row_hashes.append(row_hash)
            versions[source_name] = {
                "ordered_row_identity_sha256": canonical_sha256(row_hashes),
                "row_sha256": row_hashes,
                "questions": [row["question"] for row in selected],
                "answers": [row["answer"] for row in selected],
                "raw_prompt_answer_sha256": canonical_sha256({
                    "questions": [row["question"] for row in selected],
                    "answers": [row["answer"] for row in selected],
                }),
            }
        for item in items:
            anchor_counts[item["pairing_anchor"]] += 1
        panels[manifest_panel["name"]] = {
            "name": manifest_panel["name"],
            "role": manifest_panel["role"],
            "ordered_unit_identity_sha256": manifest_panel["ordered_unit_identity_sha256"],
            "unit_ids": [item["unit_id"] for item in items],
            "strata": [item["stratum"] for item in items],
            "weights": [float(item["horvitz_thompson_unit_weight"]) for item in items],
            "pairing_anchors": [item["pairing_anchor"] for item in items],
            "sources": versions,
        }
    if anchor_counts != {"shared_document": 193, "joint_component_cross_side_link": 2}:
        raise RuntimeError("v34b selected pairing-anchor coverage changed")
    value = {
        "schema": "eggroll-es-v401-replacement-fraction-transient-batches-v34b",
        "preregistration": {
            "file_sha256": PREREGISTRATION_FILE_SHA256,
            "content_sha256": preregistration["content_sha256_before_self_field"],
        },
        "frame": {
            "file_sha256": prereg_v34b.FRAME_FILE_SHA256,
            "content_sha256": prereg_v34b.FRAME_CONTENT_SHA256,
            "runtime_content_sha256": frame_v34b.EXPECTED_RUNTIME_FRAME_CONTENT_SHA256,
            "selected_units": 195,
            "reserve_units": 10,
            "shared_document_anchors": 193,
            "cross_component_anchors": 2,
        },
        "sources": {
            "production": {
                "rows": frame_v34b.PRODUCTION_ROWS,
                "file_sha256": frame_v34b.PRODUCTION_SHA256,
                "commit": frame_v34b.PRODUCTION_SOURCE_COMMIT,
            },
            "candidate_v401": {
                "rows": frame_v34b.CANDIDATE_ROWS,
                "file_sha256": frame_v34b.CANDIDATE_SHA256,
                "manifest_file_sha256": frame_v34b.CANDIDATE_MANIFEST_SHA256,
                "commit": frame_v34b.CANDIDATE_FREEZE_COMMIT,
            },
        },
        "panels": panels,
        "contains_validation_ood_heldout_or_benchmark_content": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return validate_panel_bundle(value, require_frozen_hash=bool(PANEL_BUNDLE_CONTENT_SHA256))


def validate_panel_bundle(value: dict, *, require_frozen_hash=True) -> dict:
    if (
        not isinstance(value, dict)
        or value.get("schema")
        != "eggroll-es-v401-replacement-fraction-transient-batches-v34b"
        or value.get("content_sha256_before_self_field") != canonical_sha256(_without_self(value))
        or value.get("contains_validation_ood_heldout_or_benchmark_content") is not False
        or tuple(value.get("panels", {})) != PANEL_NAMES
    ):
        raise RuntimeError("v34b transient batch bundle changed")
    if require_frozen_hash and value["content_sha256_before_self_field"] != PANEL_BUNDLE_CONTENT_SHA256:
        raise RuntimeError("v34b transient batch identity changed")
    selected_ids = []
    anchors = []
    for name in PANEL_NAMES:
        panel = value["panels"][name]
        selected_ids.extend(panel.get("unit_ids", []))
        anchors.extend(panel.get("pairing_anchors", []))
        if (
            set(panel) != {
                "name", "role", "ordered_unit_identity_sha256", "unit_ids",
                "strata", "weights", "pairing_anchors", "sources",
            }
            or len(panel["unit_ids"]) != frame_v34b.PANEL_SIZE
            or len(panel["strata"]) != frame_v34b.PANEL_SIZE
            or len(panel["weights"]) != frame_v34b.PANEL_SIZE
            or not math.isclose(math.fsum(panel["weights"]), float(PAIRED_FRAME_POPULATION), abs_tol=1e-12)
            or canonical_sha256(panel["unit_ids"]) != panel["ordered_unit_identity_sha256"]
            or set(panel["sources"]) != set(SOURCES)
        ):
            raise RuntimeError(f"v34b transient {name} design changed")
        for source in SOURCES:
            batch = panel["sources"][source]
            if (
                set(batch) != {
                    "ordered_row_identity_sha256", "row_sha256", "questions",
                    "answers", "raw_prompt_answer_sha256",
                }
                or [len(batch[key]) for key in ("row_sha256", "questions", "answers")]
                != [frame_v34b.PANEL_SIZE] * 3
                or canonical_sha256(batch["row_sha256"])
                != batch["ordered_row_identity_sha256"]
                or canonical_sha256({
                    "questions": batch["questions"], "answers": batch["answers"]
                }) != batch["raw_prompt_answer_sha256"]
            ):
                raise RuntimeError(f"v34b transient {name}/{source} batch changed")
    if (
        len(selected_ids) != len(set(selected_ids))
        or len(selected_ids) != 195
        or anchors.count("shared_document") != 193
        or anchors.count("joint_component_cross_side_link") != 2
    ):
        raise RuntimeError("v34b selected transient coverage changed")
    return value


def resident_signed_wave_schedule() -> list[dict]:
    frozen = load_hardened_preregistration()
    schedule = frozen["frozen_recipe"]["perturbation_basis"]["signed_population_schedule"]
    if schedule != prereg_v34b.signed_population_schedule():
        raise RuntimeError("v34b signed schedule changed")
    if (
        len(schedule) != 32
        or sum(item["resident_source_order"][0] == "production" for item in schedule) != 16
        or any(set(item["resident_source_order"]) != set(SOURCES) for item in schedule)
    ):
        raise RuntimeError("v34b signed schedule balance changed")
    return copy.deepcopy(schedule)


def execute_resident_signed_wave(schedule_item, *, perturb, score_source, restore):
    schedule = resident_signed_wave_schedule()
    index = schedule_item.get("signed_wave_index") if isinstance(schedule_item, dict) else None
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 32 or schedule_item != schedule[index]:
        raise RuntimeError("v34b signed schedule item changed")
    if not all(callable(item) for item in (perturb, score_source, restore)):
        raise TypeError("v34b signed-wave callbacks must be callable")
    captures = {}
    try:
        perturb(list(schedule_item["engine_direction_seeds"]), bool(schedule_item["negate"]))
        for source in schedule_item["resident_source_order"]:
            captures[source] = score_source(source)
    finally:
        restore()
    if tuple(captures) != tuple(schedule_item["resident_source_order"]):
        raise RuntimeError("v34b resident source capture incomplete")
    return captures


def _unit_score_array(unit_scores) -> np.ndarray:
    values = np.asarray(unit_scores, dtype=np.float64)
    expected = (2, 5, 2, 64, 39)
    if values.shape != expected or not np.isfinite(values).all():
        raise RuntimeError("v34b per-unit score tensor changed")
    return values


def _panel_design(panel_bundle):
    panel_bundle = validate_panel_bundle(panel_bundle)
    weights = np.asarray([
        panel_bundle["panels"][name]["weights"] for name in PANEL_NAMES
    ], dtype=np.float64)
    strata = [list(panel_bundle["panels"][name]["strata"]) for name in PANEL_NAMES]
    if weights.shape != (5, 39):
        raise RuntimeError("v34b panel design changed")
    return weights, strata


def observed_source_panel_scores(unit_scores, panel_bundle) -> np.ndarray:
    values = _unit_score_array(unit_scores)
    weights, _strata = _panel_design(panel_bundle)
    panel_scores = np.einsum("vpsdu,pu->vpsd", values, weights)
    panel_scores /= float(PAIRED_FRAME_POPULATION)
    if panel_scores.shape != (2, 5, 2, 64) or not np.isfinite(panel_scores).all():
        raise RuntimeError("v34b observed source panel scores changed")
    return panel_scores


def convex_fraction_panel_scores(source_panel_scores, fractions=FRACTIONS) -> np.ndarray:
    values = np.asarray(source_panel_scores, dtype=np.float64)
    if values.ndim < 1 or values.shape[0] != 2 or not np.isfinite(values).all():
        raise RuntimeError("v34b convex source tensor changed")
    production, candidate = values
    result = np.stack(
        [production] + [
            (1.0 - fraction) * production + fraction * candidate
            for fraction in fractions
        ],
        axis=0,
    )
    if (
        not np.array_equal(result[0], production)
        or (fractions and fractions[-1] == 1.0 and not np.array_equal(result[-1], candidate))
    ):
        raise RuntimeError("v34b convex fraction endpoints changed")
    return result


def _cosine_last_axis(left, right):
    numerator = np.sum(left * right, axis=-1)
    denominator = np.sqrt(np.sum(left * left, axis=-1) * np.sum(right * right, axis=-1))
    if np.any(denominator == 0.0):
        raise RuntimeError("v34b endpoint cosine received zero direction")
    return numerator / denominator


def _sign_agreement_last_axis(left, right):
    return np.mean(np.sign(left) == np.sign(right), axis=-1)


def endpoint_arrays(panel_scores):
    values = np.asarray(panel_scores, dtype=np.float64)
    if (
        values.ndim != 5
        or values.shape[2:] != (5, 2, 64)
        or not np.isfinite(values).all()
    ):
        raise RuntimeError("v34b endpoint panel tensor changed")
    central = 0.5 * (values[:, :, :, 0, :] - values[:, :, :, 1, :])
    means = np.mean(central, axis=-1, keepdims=True)
    spreads = np.sqrt(np.mean((central - means) ** 2, axis=-1))
    if np.any(spreads == 0.0):
        raise RuntimeError("v34b panel response spread is zero")
    coefficients = (central - means) / (
        spreads[..., np.newaxis] + anchor_v13.STANDARDIZATION_EPSILON_V13
    )
    aggregate = np.median(coefficients[:, :, :3, :], axis=2)
    pairs = ((0, 1), (0, 2), (1, 2))
    families = {
        "optimization_pairwise_cosine": np.stack([
            _cosine_last_axis(coefficients[:, :, left], coefficients[:, :, right])
            for left, right in pairs
        ], axis=-1),
        "optimization_pairwise_sign_agreement": np.stack([
            _sign_agreement_last_axis(coefficients[:, :, left], coefficients[:, :, right])
            for left, right in pairs
        ], axis=-1),
        "aggregate_to_optimization_cosine": np.stack([
            _cosine_last_axis(aggregate, coefficients[:, :, index]) for index in range(3)
        ], axis=-1),
        "aggregate_to_optimization_sign_agreement": np.stack([
            _sign_agreement_last_axis(aggregate, coefficients[:, :, index]) for index in range(3)
        ], axis=-1),
        "train_screen_cosine": np.stack([
            _cosine_last_axis(aggregate, coefficients[:, :, index]) for index in (3, 4)
        ], axis=-1),
        "train_screen_sign_agreement": np.stack([
            _sign_agreement_last_axis(aggregate, coefficients[:, :, index]) for index in (3, 4)
        ], axis=-1),
    }
    endpoints = {}
    for family in prereg_v34b.METRIC_FAMILIES:
        endpoints[f"{family}_median"] = np.median(families[family], axis=-1)
        endpoints[f"{family}_worst"] = np.min(families[family], axis=-1)
    if set(endpoints) != set(prereg_v34b.ENDPOINTS) or any(
        item.shape != values.shape[:2]
        or not np.isfinite(item).all()
        or np.any(item < -1.0 - 1e-12)
        or np.any(item > 1.0 + 1e-12)
        for item in endpoints.values()
    ):
        raise RuntimeError("v34b endpoint family changed")
    return {
        "endpoints": endpoints,
        "central": central,
        "coefficients": coefficients,
        "aggregate": aggregate,
        "spreads": spreads,
    }


def observed_fraction_endpoints(unit_scores, panel_bundle):
    sources = observed_source_panel_scores(unit_scores, panel_bundle)
    arms = convex_fraction_panel_scores(sources)[:, np.newaxis]
    analyzed = endpoint_arrays(arms)
    names = ("production",) + tuple(f"fraction_{fraction:.2f}" for fraction in FRACTIONS)
    result = {}
    for index, name in enumerate(names):
        endpoint_values = {
            endpoint: float(values[index, 0])
            for endpoint, values in analyzed["endpoints"].items()
        }
        result[name] = {
            "all_panel_spreads_nonzero": bool(np.all(analyzed["spreads"][index, 0] > 0.0)),
            "endpoint_values": endpoint_values,
            "compact_estimator_sha256": canonical_sha256({
                "schema": "eggroll-es-v34b-compact-estimator-commitment",
                "arm": name,
                "panel_scores": arms[index, 0].tolist(),
                "central": analyzed["central"][index, 0].tolist(),
                "coefficients": analyzed["coefficients"][index, 0].tolist(),
                "aggregate": analyzed["aggregate"][index, 0].tolist(),
                "endpoint_values": endpoint_values,
            }),
        }
    return result


def _bootstrap_draw_plan(panel_bundle, repetitions, expected_sha256):
    _weights, strata_by_panel = _panel_design(panel_bundle)
    generator = np.random.default_rng(prereg_v34b.BOOTSTRAP_SEED)
    digest = hashlib.sha256()
    result = {}
    for panel_index, panel in enumerate(PANEL_NAMES):
        result[panel] = {}
        for stratum in STRATA:
            positions = np.asarray([
                index for index, value in enumerate(strata_by_panel[panel_index])
                if value == stratum
            ], dtype=np.int64)
            count = frame_v34b.STRATUM_QUOTAS[stratum]
            if positions.shape != (count,):
                raise RuntimeError("v34b bootstrap stratum quota changed")
            draw = generator.integers(0, count, size=(repetitions, count), dtype=np.int64)
            header = json.dumps({
                "panel": panel,
                "stratum": stratum,
                "shape": list(draw.shape),
                "dtype": "int64",
            }, sort_keys=True, separators=(",", ":")).encode()
            digest.update(len(header).to_bytes(8, "little"))
            digest.update(header)
            digest.update(draw.tobytes(order="C"))
            result[panel][stratum] = {"positions": positions, "draw": draw}
    actual = digest.hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise RuntimeError("v34b bootstrap draw-plan identity changed")
    return result, actual


def _bootstrap_source_panel_scores(values, draw_plan, start, stop):
    batch_size = stop - start
    result = np.zeros((2, batch_size, 5, 2, 64), dtype=np.float64)
    for panel_index, panel in enumerate(PANEL_NAMES):
        for stratum in STRATA:
            plan = draw_plan[panel][stratum]
            stratum_scores = np.take(values[:, panel_index], plan["positions"], axis=-1)
            resampled = np.take(stratum_scores, plan["draw"][start:stop], axis=-1)
            means = np.mean(resampled, axis=-1).transpose(0, 3, 1, 2)
            population = frame_v34b.EXPECTED_PAIRED_STRATA[stratum]
            result[:, :, panel_index] += (
                float(population) / float(PAIRED_FRAME_POPULATION)
            ) * means
    return result


def analyze_fixed_sequence_impl(
    unit_scores,
    panel_bundle,
    *,
    repetitions,
    expected_draw_plan_sha256,
):
    values = _unit_score_array(unit_scores)
    observed = observed_fraction_endpoints(values, panel_bundle)
    draw_plan, draw_sha256 = _bootstrap_draw_plan(
        panel_bundle, repetitions, expected_draw_plan_sha256
    )
    quantile = prereg_v34b.FAMILYWISE_ALPHA / len(prereg_v34b.ENDPOINTS)
    tested = []
    selected = 0.0
    stopped = False
    for fraction in FRACTIONS:
        fraction_name = f"fraction_{fraction:.2f}"
        replicates = {
            endpoint: np.empty(repetitions, dtype=np.float64)
            for endpoint in prereg_v34b.ENDPOINTS
        }
        for start in range(0, repetitions, BOOTSTRAP_CHUNK_SIZE):
            stop = min(repetitions, start + BOOTSTRAP_CHUNK_SIZE)
            sources = _bootstrap_source_panel_scores(values, draw_plan, start, stop)
            arms = convex_fraction_panel_scores(sources, fractions=(fraction,))
            analyzed = endpoint_arrays(arms)
            for endpoint, endpoint_values in analyzed["endpoints"].items():
                replicates[endpoint][start:stop] = endpoint_values[1] - endpoint_values[0]
        endpoints = {}
        for endpoint, draws in replicates.items():
            point_delta = (
                observed[fraction_name]["endpoint_values"][endpoint]
                - observed["production"]["endpoint_values"][endpoint]
            )
            endpoints[endpoint] = {
                "fraction_minus_production": float(point_delta),
                "familywise_lcb": float(np.quantile(
                    draws, quantile, method=BOOTSTRAP_QUANTILE_METHOD
                )),
                "noninferiority_margin": 0.0,
            }
        point_pass = all(item["fraction_minus_production"] >= 0.0 for item in endpoints.values())
        lcb_pass = all(item["familywise_lcb"] >= 0.0 for item in endpoints.values())
        passed = point_pass and lcb_pass
        tested.append({
            "fraction": fraction,
            "all_12_point_deltas_nonnegative": point_pass,
            "all_12_familywise_lcbs_nonnegative": lcb_pass,
            "pass": passed,
            "endpoints": endpoints,
            "fraction_compact_estimator_sha256": observed[fraction_name][
                "compact_estimator_sha256"
            ],
        })
        if not passed:
            stopped = True
            break
        selected = fraction
    untested = list(FRACTIONS[len(tested):])
    result = {
        "schema": "eggroll-es-v401-replacement-fraction-fixed-sequence-analysis-v34b",
        "bootstrap": {
            "seed": prereg_v34b.BOOTSTRAP_SEED,
            "repetitions": repetitions,
            "draw_plan_sha256": draw_sha256,
            "one_sided_bonferroni_quantile": quantile,
            "quantile_method": BOOTSTRAP_QUANTILE_METHOD,
            "raw_draws_or_replicates_persisted": False,
        },
        "production_compact_estimator_sha256": observed["production"][
            "compact_estimator_sha256"
        ],
        "tested_fractions": tested,
        "untested_fractions_after_first_failure": untested,
        "stopped_at_first_failure": stopped,
        "largest_consecutively_passing_fraction": selected,
        "fraction_specific_model_requests": 0,
        "persisted_response_vectors_unit_scores_coefficients_or_draws": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def analyze_fixed_sequence(unit_scores, panel_bundle):
    result = analyze_fixed_sequence_impl(
        unit_scores,
        panel_bundle,
        repetitions=prereg_v34b.BOOTSTRAP_REPETITIONS,
        expected_draw_plan_sha256=EXPECTED_DRAW_PLAN_SHA256,
    )
    if (
        result["bootstrap"]["repetitions"] != 50_000
        or result["bootstrap"]["draw_plan_sha256"] != EXPECTED_DRAW_PLAN_SHA256
    ):
        raise RuntimeError("v34b hardened fixed-sequence analysis changed")
    return result


def validate_runtime_integrity(integrity: dict) -> dict:
    component_keys = RUNTIME_INTEGRITY_KEYS - {"all_integrity_audits_passed"}
    expected_all = all(integrity.get(key) is True for key in component_keys)
    if (
        not isinstance(integrity, dict)
        or set(integrity) != RUNTIME_INTEGRITY_KEYS
        or integrity.get("all_integrity_audits_passed") is not expected_all
        or any(integrity.get(key) not in (True, False) for key in RUNTIME_INTEGRITY_KEYS)
    ):
        raise RuntimeError("v34b runtime integrity contract changed")
    return integrity


def build_compact_summary(unit_scores, panel_bundle, runtime_integrity):
    integrity = validate_runtime_integrity(runtime_integrity)
    if integrity["all_integrity_audits_passed"] is not True:
        raise RuntimeError("v34b refuses analysis after runtime-integrity failure")
    analysis = analyze_fixed_sequence(unit_scores, panel_bundle)
    value = {
        "schema": "eggroll-es-v401-replacement-fraction-compact-summary-v34b",
        "preregistration_content_sha256": PREREGISTRATION_CONTENT_SHA256,
        "frame_content_sha256": prereg_v34b.FRAME_CONTENT_SHA256,
        "runtime_integrity": copy.deepcopy(integrity),
        "fixed_sequence_analysis": analysis,
        "contains_dataset_rows_questions_answers_document_or_eval_content": False,
        "contains_unit_scores_response_vectors_coefficients_bootstrap_draws_or_replicates": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def evaluate_gate(summary: dict) -> dict:
    if (
        not isinstance(summary, dict)
        or summary.get("schema")
        != "eggroll-es-v401-replacement-fraction-compact-summary-v34b"
        or summary.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(summary))
        or summary.get("runtime_integrity", {}).get("all_integrity_audits_passed")
        is not True
    ):
        raise RuntimeError("v34b compact gate input changed")
    analysis = summary.get("fixed_sequence_analysis", {})
    tested = analysis.get("tested_fractions", [])
    expected_order = list(FRACTIONS[:len(tested)])
    if (
        [item.get("fraction") for item in tested] != expected_order
        or any(set(item.get("endpoints", {})) != set(prereg_v34b.ENDPOINTS) for item in tested)
        or any(item.get("pass") is not (
            item.get("all_12_point_deltas_nonnegative") is True
            and item.get("all_12_familywise_lcbs_nonnegative") is True
        ) for item in tested)
        or any(item["pass"] is False for item in tested[:-1])
        or analysis.get("largest_consecutively_passing_fraction")
        != (tested[-1]["fraction"] if tested and tested[-1]["pass"] else (
            tested[-2]["fraction"] if len(tested) > 1 else 0.0
        ))
        or analysis.get("untested_fractions_after_first_failure")
        != list(FRACTIONS[len(tested):])
    ):
        raise RuntimeError("v34b fixed-sequence result changed")
    selected = analysis["largest_consecutively_passing_fraction"]
    result = {
        "schema": "eggroll-es-v401-replacement-fraction-gate-v34b",
        "largest_consecutively_passing_fraction": selected,
        "tested_fraction_count": len(tested),
        "stopped_at_first_failure": analysis["stopped_at_first_failure"],
        "decision": (
            "retain_production_no_fraction_authorized"
            if selected == 0.0
            else "authorize_only_separately_frozen_train_only_HPO_or_training_recipe"
        ),
        "direct_dataset_promotion_authorized": False,
        "model_update_authorized": False,
        "checkpoint_write_authorized": False,
        "validation_heldout_ood_or_benchmark_evaluation_authorized": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result
