#!/usr/bin/env python3
"""Pure aggregate-only estimator mechanics for V23A insertion stability."""

from __future__ import annotations

import hashlib
import math

import numpy as np

import eggroll_es_insertion_stability_preregistration_v23a as prereg_v23a
import train_eggroll_es_specialist_anchor_v13 as anchor_v13


ARMS_V23A = prereg_v23a.ARM_ORDER_V23A
CANDIDATE_ARMS_V23A = prereg_v23a.CANDIDATE_ARMS_V23A
PANEL_NAMES_V23A = anchor_v13.PANEL_NAMES_V13
OPTIMIZATION_PANELS_V23A = anchor_v13.OPTIMIZATION_PANELS_V13
TRAIN_SCREENS_V23A = anchor_v13.TRAIN_SCREENS_V13
GRADIENT_ENDPOINTS_V23A = prereg_v23a.GRADIENT_ENDPOINTS_V23A
REFERENCE_ENDPOINTS_V23A = prereg_v23a.REFERENCE_ENDPOINTS_V23A
ALL_ENDPOINTS_V23A = prereg_v23a.ALL_ENDPOINTS_V23A
BOOTSTRAP_REPETITIONS_V23A = prereg_v23a.BOOTSTRAP_REPETITIONS_V23A
BOOTSTRAP_SEED_V23A = prereg_v23a.BOOTSTRAP_SEED_V23A
BOOTSTRAP_CHUNK_SIZE_V23A = 128
HT_DENOMINATOR_V23A = 310.0
STANDARDIZATION_EPSILON_V23A = 1e-8
RUNTIME_INTEGRITY_KEYS_V23A = {
    "all_sixty_four_signed_waves_complete",
    "all_five_panels_every_arm_and_signed_wave",
    "same_direction_seed_all_four_arms",
    "same_fixed_requests_all_four_arms",
    "exact_selected_reference_restored_every_signed_wave",
    "unselected_origin_unchanged",
    "pre_post_unperturbed_reference_probe_equal",
    "distinct_gpu_assignment_verified",
    "union_planner_called",
    "all_integrity_audits_passed",
}

canonical_sha256 = prereg_v23a.canonical_sha256


def load_panel_bundle_v23a():
    bundle = anchor_v13.load_panel_bundle_v13()
    if (
        bundle["content_sha256_before_self_field"]
        != anchor_v13.PANEL_BUNDLE_CONTENT_SHA256_V13
        or tuple(bundle["panels"]) != PANEL_NAMES_V23A
    ):
        raise RuntimeError("v23a frozen train-only panel bundle changed")
    return bundle


def _score_arrays_v23a(unit_scores, reference_scores):
    if not isinstance(unit_scores, dict) or tuple(unit_scores) != ARMS_V23A:
        raise RuntimeError("v23a unit-score arm order changed")
    if not isinstance(reference_scores, dict) or tuple(reference_scores) != ARMS_V23A:
        raise RuntimeError("v23a reference-score arm order changed")
    unit = {}
    reference = {}
    for arm in ARMS_V23A:
        unit[arm] = np.asarray(unit_scores[arm], dtype=np.float64)
        reference[arm] = np.asarray(reference_scores[arm], dtype=np.float64)
        if (
            unit[arm].shape != (5, 2, 32, 56)
            or reference[arm].shape != (5, 56)
            or not np.isfinite(unit[arm]).all()
            or not np.isfinite(reference[arm]).all()
        ):
            raise RuntimeError("v23a raw score tensor geometry changed")
    return unit, reference


def _panel_weights_v23a(panel_bundle):
    anchor_v13.validate_panel_bundle_v13(panel_bundle)
    weights = np.asarray([
        panel_bundle["panels"][panel]["weights"] for panel in PANEL_NAMES_V23A
    ], dtype=np.float64)
    if (
        weights.shape != (5, 56)
        or not np.allclose(
            np.sum(weights, axis=1), HT_DENOMINATOR_V23A,
            rtol=0.0, atol=1e-12,
        )
    ):
        raise RuntimeError("v23a Horvitz-Thompson weights changed")
    return weights


def _cosine(left, right):
    numerator = np.sum(left * right, axis=-1)
    denominator = np.sqrt(
        np.sum(left * left, axis=-1) * np.sum(right * right, axis=-1)
    )
    if np.any(denominator == 0.0):
        raise RuntimeError("v23a endpoint cosine received a zero direction")
    return numerator / denominator


def _sign_agreement(left, right):
    return np.mean(np.sign(left) == np.sign(right), axis=-1)


def _gradient_endpoint_arrays(coefficients):
    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim != 4 or values.shape[0] != 4 or values.shape[2:] != (5, 32):
        raise RuntimeError("v23a coefficient tensor geometry changed")
    pairs = ((0, 1), (0, 2), (1, 2))
    aggregate = np.median(values[:, :, :3, :], axis=2)
    families = {
        "optimization_pairwise_cosine": np.stack([
            _cosine(values[:, :, left, :], values[:, :, right, :])
            for left, right in pairs
        ], axis=-1),
        "optimization_pairwise_sign_agreement": np.stack([
            _sign_agreement(values[:, :, left, :], values[:, :, right, :])
            for left, right in pairs
        ], axis=-1),
        "aggregate_to_optimization_cosine": np.stack([
            _cosine(aggregate, values[:, :, index, :]) for index in range(3)
        ], axis=-1),
        "aggregate_to_optimization_sign_agreement": np.stack([
            _sign_agreement(aggregate, values[:, :, index, :])
            for index in range(3)
        ], axis=-1),
        "train_screen_cosine": np.stack([
            _cosine(aggregate, values[:, :, index, :]) for index in range(3, 5)
        ], axis=-1),
        "train_screen_sign_agreement": np.stack([
            _sign_agreement(aggregate, values[:, :, index, :])
            for index in range(3, 5)
        ], axis=-1),
    }
    endpoints = {}
    for name, family in families.items():
        endpoints[f"{name}_median"] = np.median(family, axis=-1)
        endpoints[f"{name}_worst"] = np.min(family, axis=-1)
    if tuple(endpoints) != GRADIENT_ENDPOINTS_V23A:
        raise RuntimeError("v23a gradient endpoint order changed")
    return endpoints


def _observed_gradient_v23a(unit, weights):
    values = np.stack([unit[arm] for arm in ARMS_V23A], axis=0)
    panel_sign_direction = np.einsum(
        "pu,apsdu->apsd", weights, values
    ) / HT_DENOMINATOR_V23A
    central = 0.5 * (
        panel_sign_direction[:, :, 0, :] - panel_sign_direction[:, :, 1, :]
    )
    means = np.mean(central, axis=-1, keepdims=True)
    spreads = np.sqrt(np.mean((central - means) ** 2, axis=-1))
    if np.any(spreads == 0.0):
        raise RuntimeError("v23a panel response spread is zero")
    coefficients = (central - means) / (
        spreads[..., np.newaxis] + STANDARDIZATION_EPSILON_V23A
    )
    arrays = _gradient_endpoint_arrays(coefficients[:, np.newaxis, :, :])
    arms = {
        arm: {
            name: float(arrays[name][arm_index, 0])
            for name in GRADIENT_ENDPOINTS_V23A
        } for arm_index, arm in enumerate(ARMS_V23A)
    }
    return arms, central


def _observed_reference_v23a(reference, weights):
    values = np.stack([reference[arm] for arm in ARMS_V23A], axis=0)
    panel_rewards = np.einsum("pu,apu->ap", weights, values) / HT_DENOMINATOR_V23A
    base = panel_rewards[0]
    result = {}
    for candidate_index, arm in enumerate(CANDIDATE_ARMS_V23A, start=1):
        reward_delta = panel_rewards[candidate_index] - base
        loss_compatibility = (-base) - (-panel_rewards[candidate_index])
        if not np.array_equal(reward_delta, loss_compatibility):
            raise RuntimeError("v23a reward/loss compatibility dual changed")
        result[arm] = {
            "unperturbed_reward_delta_median": float(np.median(reward_delta)),
            "unperturbed_reward_delta_worst": float(np.min(reward_delta)),
            "unperturbed_loss_compatibility_median": float(
                np.median(loss_compatibility)
            ),
            "unperturbed_loss_compatibility_worst": float(
                np.min(loss_compatibility)
            ),
        }
    return result


def bootstrap_draw_plan_v23a(panel_bundle):
    anchor_v13.validate_panel_bundle_v13(panel_bundle)
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED_V23A))
    direction = generator.integers(
        0, 32, size=(BOOTSTRAP_REPETITIONS_V23A, 32), dtype=np.uint8
    )
    row_draws = {}
    row_hashes = {}
    for panel in PANEL_NAMES_V23A:
        strata = panel_bundle["panels"][panel]["strata"]
        row_draws[panel] = {}
        row_hashes[panel] = {}
        for stratum in anchor_v13.panel_sampler.STRATA:
            indices = np.asarray([
                index for index, value in enumerate(strata) if value == stratum
            ], dtype=np.uint8)
            quota = anchor_v13.panel_sampler.STRATUM_QUOTAS[stratum]
            if len(indices) != quota:
                raise RuntimeError("v23a panel stratum quota changed")
            local = generator.integers(
                0, quota, size=(BOOTSTRAP_REPETITIONS_V23A, quota),
                dtype=np.uint8,
            )
            draws = indices[local]
            row_draws[panel][stratum] = draws
            row_hashes[panel][stratum] = hashlib.sha256(draws.tobytes()).hexdigest()
    certificate = {
        "schema": "eggroll-es-insertion-bootstrap-draw-plan-v23a",
        "seed": BOOTSTRAP_SEED_V23A,
        "generator": "numpy.PCG64",
        "repetitions": BOOTSTRAP_REPETITIONS_V23A,
        "direction_draw_shape": list(direction.shape),
        "direction_draw_sha256": hashlib.sha256(direction.tobytes()).hexdigest(),
        "row_draw_sha256": row_hashes,
        "paired_same_direction_draws_all_arms": True,
        "paired_same_stratified_row_draws_all_arms": True,
        "draw_arrays_persisted": False,
    }
    certificate["content_sha256_before_self_field"] = canonical_sha256(certificate)
    return certificate, direction, row_draws


def _bootstrap_gradient_deltas_v23a(central, direction_draws):
    result = {
        arm: {
            name: np.empty(BOOTSTRAP_REPETITIONS_V23A, dtype=np.float64)
            for name in GRADIENT_ENDPOINTS_V23A
        } for arm in CANDIDATE_ARMS_V23A
    }
    for start in range(0, BOOTSTRAP_REPETITIONS_V23A, BOOTSTRAP_CHUNK_SIZE_V23A):
        stop = min(start + BOOTSTRAP_CHUNK_SIZE_V23A, BOOTSTRAP_REPETITIONS_V23A)
        draws = direction_draws[start:stop]
        sampled = np.take(central, draws, axis=-1).transpose(0, 2, 1, 3)
        means = np.mean(sampled, axis=-1, keepdims=True)
        spreads = np.sqrt(np.mean((sampled - means) ** 2, axis=-1))
        if np.any(spreads == 0.0):
            raise RuntimeError("v23a bootstrap sampled a zero-spread panel")
        coefficients = (sampled - means) / (
            spreads[..., np.newaxis] + STANDARDIZATION_EPSILON_V23A
        )
        endpoints = _gradient_endpoint_arrays(coefficients)
        for candidate_index, arm in enumerate(CANDIDATE_ARMS_V23A, start=1):
            for name in GRADIENT_ENDPOINTS_V23A:
                result[arm][name][start:stop] = (
                    endpoints[name][candidate_index] - endpoints[name][0]
                )
    return result


def _bootstrap_reference_deltas_v23a(reference, weights, row_draws, panel_bundle):
    values = np.stack([reference[arm] for arm in ARMS_V23A], axis=0)
    panel_deltas = np.empty(
        (3, BOOTSTRAP_REPETITIONS_V23A, 5), dtype=np.float64
    )
    for panel_index, panel in enumerate(PANEL_NAMES_V23A):
        strata = panel_bundle["panels"][panel]["strata"]
        totals = np.zeros((3, BOOTSTRAP_REPETITIONS_V23A), dtype=np.float64)
        for stratum in anchor_v13.panel_sampler.STRATA:
            draws = row_draws[panel][stratum]
            for candidate_index in range(1, 4):
                delta = values[candidate_index, panel_index] - values[0, panel_index]
                totals[candidate_index - 1] += np.sum(
                    weights[panel_index, draws] * delta[draws], axis=1
                )
        panel_deltas[:, :, panel_index] = totals / HT_DENOMINATOR_V23A
    result = {}
    for candidate_index, arm in enumerate(CANDIDATE_ARMS_V23A):
        median = np.median(panel_deltas[candidate_index], axis=-1)
        worst = np.min(panel_deltas[candidate_index], axis=-1)
        result[arm] = {
            "unperturbed_reward_delta_median": median,
            "unperturbed_reward_delta_worst": worst,
            "unperturbed_loss_compatibility_median": median.copy(),
            "unperturbed_loss_compatibility_worst": worst.copy(),
        }
    return result


def build_compact_estimator_summary_v23a(unit_scores, reference_scores, panel_bundle):
    unit, reference = _score_arrays_v23a(unit_scores, reference_scores)
    weights = _panel_weights_v23a(panel_bundle)
    gradient_arms, central = _observed_gradient_v23a(unit, weights)
    reference_candidates = _observed_reference_v23a(reference, weights)
    certificate, direction_draws, row_draws = bootstrap_draw_plan_v23a(panel_bundle)
    gradient_bootstrap = _bootstrap_gradient_deltas_v23a(central, direction_draws)
    reference_bootstrap = _bootstrap_reference_deltas_v23a(
        reference, weights, row_draws, panel_bundle
    )
    quantile = 0.05 / prereg_v23a.FAMILY_HYPOTHESIS_COUNT_V23A
    comparisons = {}
    for arm in CANDIDATE_ARMS_V23A:
        endpoints = {}
        for name in GRADIENT_ENDPOINTS_V23A:
            delta = gradient_arms[arm][name] - gradient_arms["base_middle_late"][name]
            endpoints[name] = {
                "kind": "gradient_stability",
                "candidate_minus_base": delta,
                "familywise_lcb": float(np.quantile(
                    gradient_bootstrap[arm][name], quantile, method="linear"
                )),
                "noninferiority_margin": 0.0,
            }
        for name in REFERENCE_ENDPOINTS_V23A:
            endpoints[name] = {
                "kind": "unperturbed_train_reference_compatibility",
                "candidate_minus_base": reference_candidates[arm][name],
                "familywise_lcb": float(np.quantile(
                    reference_bootstrap[arm][name], quantile, method="linear"
                )),
                "noninferiority_margin": 0.0,
            }
        comparisons[arm] = {
            "candidate": arm, "control": "base_middle_late",
            "endpoint_count": 16, "endpoints": endpoints,
        }
    direction_draws = None
    row_draws = None
    result = {
        "schema": "eggroll-es-insertion-location-compact-estimator-v23a",
        "arms": {
            arm: {
                "gradient_endpoint_values": gradient_arms[arm],
                "compact_estimator_sha256": canonical_sha256({
                    "gradient": gradient_arms[arm],
                    "reference": reference_candidates.get(arm),
                }),
            } for arm in ARMS_V23A
        },
        "comparisons": comparisons,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED_V23A,
            "repetitions": BOOTSTRAP_REPETITIONS_V23A,
            "one_sided_familywise_quantile": quantile,
            "family_hypothesis_count": 48,
            "quantile_method": "linear",
            "draw_plan_content_sha256": certificate[
                "content_sha256_before_self_field"
            ],
            "paired_same_direction_and_row_draws_all_arms": True,
            "draw_arrays_persisted": False,
        },
        "persisted_response_vectors_or_row_content": False,
        "unit_scores_persisted": False,
        "bootstrap_draws_or_replicates_persisted": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def evaluate_gate_v23a(summary):
    comparisons = summary.get("comparisons", {})
    integrity = summary.get("runtime_integrity", {})
    if (
        set(comparisons) != set(CANDIDATE_ARMS_V23A)
        or set(integrity) != set(ARMS_V23A)
        or any(set(integrity[arm]) != RUNTIME_INTEGRITY_KEYS_V23A for arm in ARMS_V23A)
        or any(integrity[arm]["union_planner_called"] is not False for arm in ARMS_V23A)
        or any(
            integrity[arm][key] not in (True, False)
            for arm in ARMS_V23A
            for key in RUNTIME_INTEGRITY_KEYS_V23A - {"union_planner_called"}
        )
    ):
        raise RuntimeError("v23a gate runtime integrity changed")
    location_results = {}
    passing = []
    for arm in CANDIDATE_ARMS_V23A:
        endpoints = comparisons[arm].get("endpoints", {})
        if (
            set(endpoints) != set(ALL_ENDPOINTS_V23A)
            or comparisons[arm].get("endpoint_count") != 16
            or any(
                set(item) != {
                    "kind", "candidate_minus_base", "familywise_lcb",
                    "noninferiority_margin",
                }
                or item["noninferiority_margin"] != 0.0
                or not math.isfinite(float(item["candidate_minus_base"]))
                or not math.isfinite(float(item["familywise_lcb"]))
                for item in endpoints.values()
            )
        ):
            raise RuntimeError("v23a gate endpoint coverage changed")
        endpoint_passes = {
            name: item["familywise_lcb"] >= 0.0
            for name, item in endpoints.items()
        }
        gradient_count = sum(endpoint_passes[name] for name in GRADIENT_ENDPOINTS_V23A)
        reference_count = sum(endpoint_passes[name] for name in REFERENCE_ENDPOINTS_V23A)
        base_integrity = all(
            integrity["base_middle_late"][key] is True
            for key in RUNTIME_INTEGRITY_KEYS_V23A - {"union_planner_called"}
        )
        candidate_integrity = all(
            integrity[arm][key] is True
            for key in RUNTIME_INTEGRITY_KEYS_V23A - {"union_planner_called"}
        )
        endpoint_gate_passed = gradient_count == 12 and reference_count == 4
        passed = endpoint_gate_passed and base_integrity and candidate_integrity
        if passed:
            passing.append(arm)
        lcbs = [item["familywise_lcb"] for item in endpoints.values()]
        location_results[arm] = {
            "gradient_pass_count": gradient_count,
            "reference_pass_count": reference_count,
            "all_sixteen_familywise_lcbs_passed": endpoint_gate_passed,
            "base_runtime_integrity_passed": base_integrity,
            "candidate_runtime_integrity_passed": candidate_integrity,
            "location_passed": passed,
            "minimum_familywise_lcb": min(lcbs),
            "mean_familywise_lcb": math.fsum(lcbs) / 16,
        }
    fixed_order = {arm: index for index, arm in enumerate(CANDIDATE_ARMS_V23A)}
    selected = (
        max(
            passing,
            key=lambda arm: (
                location_results[arm]["minimum_familywise_lcb"],
                location_results[arm]["mean_familywise_lcb"],
                -fixed_order[arm],
            ),
        ) if passing else None
    )
    return {
        "schema": "eggroll-es-insertion-location-gate-v23a",
        "family_hypothesis_count": 48,
        "location_results": location_results,
        "passing_location_count": len(passing),
        "selected_location_for_confirmation": selected,
        "compatibility_gate_passed": selected is not None,
        "decision": (
            "authorize_only_separate_fresh_basis_train_only_confirmation"
            if selected is not None else "retain_v13_base_middle_late_recipe"
        ),
        "direct_model_update_authorized": False,
        "checkpoint_write_authorized": False,
        "evaluation_authorized": False,
        "dataset_promotion_authorized": False,
    }
