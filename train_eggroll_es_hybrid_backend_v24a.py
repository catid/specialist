#!/usr/bin/env python3
"""Pure memory-only estimator for the preregistered V24A backend scan."""

from __future__ import annotations

import hashlib
import math

import numpy as np

import eggroll_es_hybrid_backend_preregistration_v24a as prereg
import train_eggroll_es_specialist_anchor_v13 as anchor_v13


ARMS = prereg.ARM_ORDER_V24A
PAIRS = prereg.PAIR_ORDER_V24A
PANELS = prereg.PANEL_NAMES_V24A
QUALITY_ENDPOINTS = tuple(prereg.QUALITY_ENDPOINT_THRESHOLDS_V24A)
REPETITIONS = prereg.BOOTSTRAP_REPETITIONS_V24A
CHUNK = 128
HT_DENOMINATOR = 310.0
EPSILON = 1e-8
RUNTIME_INTEGRITY_KEYS = {
    "all_64_signed_waves_complete",
    "all_five_panels_every_arm_and_signed_wave",
    "same_direction_sign_rows_and_sampling_all_four_arms",
    "exact_selected_restore_every_arm_every_signed_wave",
    "unselected_partition_identity_unchanged",
    "full_context_a_b_equal_before_first_perturbation",
    "full_context_a_c_equal_after_population_audit",
    "distinct_gpu_assignment_verified",
    "timing_boundary_verified",
    "nvml_memory_boundary_verified",
    "update_and_nontrain_surfaces_closed",
}


canonical_sha256 = prereg.canonical_sha256


def _arrays(unit_scores, reference_scores, wave_seconds):
    if tuple(unit_scores) != ARMS or tuple(reference_scores) != ARMS:
        raise RuntimeError("v24a score arm order changed")
    unit = {arm: np.asarray(unit_scores[arm], dtype=np.float64) for arm in ARMS}
    reference = {
        arm: np.asarray(reference_scores[arm], dtype=np.float64) for arm in ARMS
    }
    timing = np.asarray(wave_seconds, dtype=np.float64)
    if (
        any(value.shape != (5, 2, 32, 56) for value in unit.values())
        or any(value.shape != (5, 56) for value in reference.values())
        or any(not np.isfinite(value).all() for value in (*unit.values(), *reference.values()))
        or timing.shape != (64, 4)
        or not np.isfinite(timing).all()
        or np.any(timing <= 0.0)
    ):
        raise RuntimeError("v24a score or timing geometry changed")
    return unit, reference, timing


def _weights(panel_bundle):
    anchor_v13.validate_panel_bundle_v13(panel_bundle)
    values = np.asarray([
        panel_bundle["panels"][panel]["weights"] for panel in PANELS
    ], dtype=np.float64)
    if values.shape != (5, 56) or not np.allclose(
        values.sum(axis=1), HT_DENOMINATOR, rtol=0.0, atol=1e-12,
    ):
        raise RuntimeError("v24a panel weights changed")
    return values


def _standardized_panel_directions(unit, weights):
    values = np.stack([unit[arm] for arm in ARMS], axis=0)
    panel_sign_direction = np.einsum("pu,apsdu->apsd", weights, values) / HT_DENOMINATOR
    central = 0.5 * (panel_sign_direction[:, :, 0] - panel_sign_direction[:, :, 1])
    means = central.mean(axis=-1, keepdims=True)
    spreads = np.sqrt(np.mean((central - means) ** 2, axis=-1, keepdims=True))
    if np.any(spreads == 0.0):
        raise RuntimeError("v24a panel direction spread is zero")
    return central, (central - means) / (spreads + EPSILON)


def _cosine(left, right):
    numerator = np.sum(left * right, axis=-1)
    denominator = np.sqrt(np.sum(left * left, axis=-1) * np.sum(right * right, axis=-1))
    if np.any(denominator == 0.0):
        raise RuntimeError("v24a cosine received a zero vector")
    return numerator / denominator


def _metric_endpoints(bf16, hybrid):
    cosine = _cosine(bf16, hybrid)
    sign = np.mean(np.sign(bf16) == np.sign(hybrid), axis=-1)
    # Panel is the last dimension after the direction reduction.  The leading
    # dimensions, if any, are bootstrap replicates.
    groups = {
        "optimization_panel": slice(0, 3),
        "train_screen": slice(3, 5),
        "all_panel": slice(0, 5),
    }
    result = {}
    for name, selection in groups.items():
        result[f"{name}_cosine_median"] = np.median(cosine[..., selection], axis=-1)
        result[f"{name}_cosine_worst"] = np.min(cosine[..., selection], axis=-1)
        result[f"{name}_sign_agreement_median"] = np.median(sign[..., selection], axis=-1)
        result[f"{name}_sign_agreement_worst"] = np.min(sign[..., selection], axis=-1)
    return result


def _reference_endpoints(bf16, hybrid, weights):
    delta = hybrid - bf16
    panel_delta = np.einsum("pu,pu->p", weights, delta) / HT_DENOMINATOR
    left = bf16.reshape(-1)
    right = hybrid.reshape(-1)
    correlation = float(np.corrcoef(left, right)[0, 1])
    mae = float(np.mean(np.abs(right - left)))
    if not math.isfinite(correlation):
        raise RuntimeError("v24a reference correlation is not finite")
    return {
        "unperturbed_reward_delta_median": float(np.median(panel_delta)),
        "unperturbed_reward_delta_worst": float(np.min(panel_delta)),
        "unperturbed_row_reward_correlation": correlation,
        "unperturbed_row_reward_mae_headroom": 0.02 - mae,
    }


def _threshold_transform(name, value):
    threshold = prereg.QUALITY_ENDPOINT_THRESHOLDS_V24A[name]
    return np.asarray(value, dtype=np.float64) - threshold


def _draw_plan(panel_bundle):
    generator = np.random.Generator(np.random.PCG64(prereg.BOOTSTRAP_SEED_V24A))
    directions = generator.integers(0, 32, size=(REPETITIONS, 32), dtype=np.uint8)
    rows = {}
    row_hashes = {}
    for panel in PANELS:
        strata = panel_bundle["panels"][panel]["strata"]
        selected = []
        for stratum in anchor_v13.panel_sampler.STRATA:
            indices = np.asarray([
                index for index, value in enumerate(strata) if value == stratum
            ], dtype=np.uint8)
            quota = anchor_v13.panel_sampler.STRATUM_QUOTAS[stratum]
            if len(indices) != quota:
                raise RuntimeError("v24a stratum quota changed")
            local = generator.integers(0, quota, size=(REPETITIONS, quota), dtype=np.uint8)
            selected.append(indices[local])
        rows[panel] = np.concatenate(selected, axis=1)
        row_hashes[panel] = hashlib.sha256(rows[panel].tobytes()).hexdigest()
    speed = generator.integers(0, 64, size=(REPETITIONS, 64), dtype=np.uint8)
    certificate = {
        "schema": "eggroll-es-v24a-bootstrap-draw-certificate",
        "seed": prereg.BOOTSTRAP_SEED_V24A,
        "repetitions": REPETITIONS,
        "direction_draw_sha256": hashlib.sha256(directions.tobytes()).hexdigest(),
        "row_draw_sha256": row_hashes,
        "speed_draw_sha256": hashlib.sha256(speed.tobytes()).hexdigest(),
        "draw_arrays_persisted": False,
    }
    certificate["content_sha256_before_self_field"] = canonical_sha256(certificate)
    return certificate, directions, rows, speed


def _reference_bootstrap_pair(bf16, hybrid, weights, rows):
    result = {name: np.empty(REPETITIONS) for name in QUALITY_ENDPOINTS[-4:]}
    for start in range(0, REPETITIONS, CHUNK):
        stop = min(start + CHUNK, REPETITIONS)
        count = stop - start
        left_parts = []
        right_parts = []
        panel_deltas = np.empty((count, 5), dtype=np.float64)
        for panel_index, panel in enumerate(PANELS):
            draw = rows[panel][start:stop]
            left = bf16[panel_index][draw]
            right = hybrid[panel_index][draw]
            sampled_weights = weights[panel_index][draw]
            panel_deltas[:, panel_index] = np.sum(
                sampled_weights * (right - left), axis=1
            ) / HT_DENOMINATOR
            left_parts.append(left)
            right_parts.append(right)
        left = np.concatenate(left_parts, axis=1)
        right = np.concatenate(right_parts, axis=1)
        left_centered = left - left.mean(axis=1, keepdims=True)
        right_centered = right - right.mean(axis=1, keepdims=True)
        denominator = np.sqrt(
            np.sum(left_centered**2, axis=1) * np.sum(right_centered**2, axis=1)
        )
        if np.any(denominator == 0.0):
            raise RuntimeError("v24a bootstrap correlation denominator is zero")
        result["unperturbed_reward_delta_median"][start:stop] = np.median(panel_deltas, axis=1)
        result["unperturbed_reward_delta_worst"][start:stop] = np.min(panel_deltas, axis=1)
        result["unperturbed_row_reward_correlation"][start:stop] = np.sum(
            left_centered * right_centered, axis=1
        ) / denominator
        result["unperturbed_row_reward_mae_headroom"][start:stop] = 0.02 - np.mean(
            np.abs(right - left), axis=1
        )
    return result


def build_compact_summary_v24a(
    unit_scores, reference_scores, wave_seconds, resident_bytes,
    panel_bundle, runtime_integrity,
):
    unit, reference, timing = _arrays(unit_scores, reference_scores, wave_seconds)
    if (
        set(resident_bytes) != set(ARMS)
        or any(not isinstance(value, int) or value <= 0 for value in resident_bytes.values())
        or set(runtime_integrity) != set(ARMS)
        or any(set(runtime_integrity[arm]) != RUNTIME_INTEGRITY_KEYS for arm in ARMS)
        or any(not all(runtime_integrity[arm].values()) for arm in ARMS)
    ):
        raise RuntimeError("v24a runtime integrity or memory coverage changed")
    weights = _weights(panel_bundle)
    central, standardized = _standardized_panel_directions(unit, weights)
    certificate, direction_draws, row_draws, speed_draws = _draw_plan(panel_bundle)
    pair_specs = prereg.build_preregistration_v24a()["pairing"]["pairs"]
    quantile = 0.05 / prereg.FAMILY_HYPOTHESIS_COUNT_V24A
    pairs = {}
    for pair in PAIRS:
        bf16_arm = pair_specs[pair]["bf16"]
        hybrid_arm = pair_specs[pair]["hybrid"]
        left_index, right_index = ARMS.index(bf16_arm), ARMS.index(hybrid_arm)
        observed = _metric_endpoints(standardized[left_index], standardized[right_index])
        observed.update(_reference_endpoints(reference[bf16_arm], reference[hybrid_arm], weights))
        bootstrap = {name: np.empty(REPETITIONS) for name in QUALITY_ENDPOINTS[:12]}
        for start in range(0, REPETITIONS, CHUNK):
            stop = min(start + CHUNK, REPETITIONS)
            draws = direction_draws[start:stop]
            sampled = np.take(central, draws, axis=2).transpose(0, 2, 1, 3)
            means = sampled.mean(axis=-1, keepdims=True)
            spreads = np.sqrt(np.mean((sampled - means) ** 2, axis=-1, keepdims=True))
            if np.any(spreads == 0.0):
                raise RuntimeError("v24a bootstrap direction spread is zero")
            normalized = (sampled - means) / (spreads + EPSILON)
            values = _metric_endpoints(normalized[left_index], normalized[right_index])
            for name, value in values.items():
                bootstrap[name][start:stop] = value
        bootstrap.update(_reference_bootstrap_pair(
            reference[bf16_arm], reference[hybrid_arm], weights, row_draws
        ))
        endpoints = {}
        for name in QUALITY_ENDPOINTS:
            transformed = _threshold_transform(name, bootstrap[name])
            endpoints[name] = {
                "observed": float(np.asarray(observed[name])),
                "threshold": prereg.QUALITY_ENDPOINT_THRESHOLDS_V24A[name],
                "familywise_lcb_headroom": float(np.quantile(
                    transformed, quantile, method="linear"
                )),
            }
        ratios = timing[:, left_index] / timing[:, right_index]
        speed_bootstrap = np.median(ratios[speed_draws], axis=1) - 1.05
        speed = {
            "observed_bf16_over_hybrid_median": float(np.median(ratios)),
            "threshold": 1.05,
            "familywise_lcb_headroom": float(np.quantile(
                speed_bootstrap, quantile, method="linear"
            )),
        }
        memory_reduction = 1.0 - resident_bytes[hybrid_arm] / resident_bytes[bf16_arm]
        quality_pass = all(item["familywise_lcb_headroom"] >= 0.0 for item in endpoints.values())
        pairs[pair] = {
            "bf16_arm": bf16_arm,
            "hybrid_arm": hybrid_arm,
            "quality_endpoints": endpoints,
            "speed_endpoint": speed,
            "memory": {
                "bf16_resident_bytes": resident_bytes[bf16_arm],
                "hybrid_resident_bytes": resident_bytes[hybrid_arm],
                "reduction": memory_reduction,
                "threshold": 0.40,
            },
            "quality_pass": quality_pass,
            "speed_pass": speed["familywise_lcb_headroom"] >= 0.0,
            "memory_pass": memory_reduction >= 0.40,
            "runtime_integrity_pass": all(runtime_integrity[bf16_arm].values())
            and all(runtime_integrity[hybrid_arm].values()),
        }
        pairs[pair]["pair_pass"] = all(pairs[pair][key] for key in (
            "quality_pass", "speed_pass", "memory_pass", "runtime_integrity_pass"
        ))
    result = {
        "schema": "eggroll-es-hybrid-backend-compact-estimator-v24a",
        "pairs": pairs,
        "bootstrap": {
            "seed": prereg.BOOTSTRAP_SEED_V24A,
            "repetitions": REPETITIONS,
            "family_hypothesis_count": 34,
            "one_sided_familywise_quantile": quantile,
            "draw_plan_content_sha256": certificate["content_sha256_before_self_field"],
            "draw_arrays_persisted": False,
        },
        "global_pass": all(pairs[pair]["pair_pass"] for pair in PAIRS),
        "decision": (
            "authorize_only_separate_fresh_basis_train_only_mapping_reversed_confirmation"
            if all(pairs[pair]["pair_pass"] for pair in PAIRS)
            else "retain_full_bf16_backend"
        ),
        "direct_backend_substitution_authorized": False,
        "model_update_authorized": False,
        "checkpoint_write_authorized": False,
        "evaluation_authorized": False,
        "dataset_promotion_authorized": False,
        "raw_scores_timing_vectors_bootstrap_draws_or_replicates_persisted": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result
