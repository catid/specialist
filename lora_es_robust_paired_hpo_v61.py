#!/usr/bin/env python3
"""CPU-only panel and paired-bootstrap contracts for the V61 HPO preview."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np


RANKING_QUOTAS_V61 = {
    "stable_partial": 24,
    "difficult": 16,
    "actor_unstable": 24,
}
RANKING_UNITS_V61 = 64
BOOTSTRAP_REPLICATES_V61 = 4096
BOOTSTRAP_ALPHA_V61 = 0.05
BOOTSTRAP_SEED_V61 = 2_026_071_611
ACTORS_V61 = 4
PASSES_PER_STATE_V61 = 2
REPLICAS_PER_UNIT_V61 = ACTORS_V61 * PASSES_PER_STATE_V61
GENERATION_COMPOSITE_WEIGHTS_V61 = {
    "f1_delta_lcb": 0.80,
    "nonzero_delta_lcb": 0.20,
    "stability_improvement_lcb": 0.25,
}
STABILITY_WEIGHTS_V61 = {
    "mean_pairwise_absolute_f1_delta": 1.0,
    "exact_label_disagreement": 0.05,
    "nonzero_label_disagreement": 0.02,
}
V61A_STRATA_FILE_SHA256 = (
    "23c8393555c3d7f09c95ecc7e23a04637f86df8fd20f55f67b67000ae78257f5"
)
V61A_STRATA_CONTENT_SHA256 = (
    "d6a34b36fea22a8bdc97698a377ffb4df596bade8cf1506c41721f5db9c4185a"
)
V61B_EVIDENCE_FILE_SHA256 = (
    "ea8ec108938ef2b17cf2572e4debca17d8986afa1f52fb2494d1fa87f54545b9"
)
V61B_EVIDENCE_CONTENT_SHA256 = (
    "a2a0f8cf07510e5f1e61635d2199ccfe5de3562ef9c859c2e23a1e8a960a29c8"
)
V61B_ANALYSIS_FILE_SHA256 = (
    "30a569fd89d38e66e95de9a55cc83c84e760007d1527bd6fa076f7ffc8c89961"
)
V61B_ANALYSIS_CONTENT_SHA256 = (
    "354317a4bcca08bac4d0bc0f2d269f019b401b84c137acd202b131c616818b74"
)
V61B_REPORT_FILE_SHA256 = (
    "ad970297015200d5e01e116b9afffd1125d165c715aa6ebd1490b683470327cd"
)
V61B_REPORT_CONTENT_SHA256 = (
    "b50a5aefbd716a5ac47a427ea172a18a248734c6fd0df77d1f19957d22f25acf"
)


def canonical_sha256_v61(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _compact_self_hash_v61(value: dict) -> str:
    return canonical_sha256_v61({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def build_panels_v61(strata: dict) -> dict:
    """Build content-free panels without overriding V61A's failed gate."""
    units = list(strata.get("units", []))
    if (
        strata.get("schema") != "v61a-v434-train-baseline-census-strata"
        or strata.get("status") != "fail_closed_insufficient_stable_exact_support"
        or strata.get("later_v61_hpo_authorized") is not False
        or strata.get("content_sha256_before_self_field")
        != V61A_STRATA_CONTENT_SHA256
        or _compact_self_hash_v61(strata) != V61A_STRATA_CONTENT_SHA256
        or strata.get("stratum_counts") != {
            "stable_exact": {"total": 3, "selection_pool": 3, "holdback": 0},
            "stable_partial": {"total": 35, "selection_pool": 27, "holdback": 8},
            "difficult": {"total": 30, "selection_pool": 23, "holdback": 7},
            "actor_unstable": {"total": 140, "selection_pool": 105, "holdback": 35},
        }
        or len(units) != 208
        or len({item.get("unit_identity_sha256") for item in units}) != 208
    ):
        raise ValueError("v61 sealed V61A strata contract changed")

    def compact(item: dict, role: str, index: int) -> dict:
        return {
            "role": role,
            "role_index": index,
            "row_sha256": item["row_sha256"],
            "unit_identity_sha256": item["unit_identity_sha256"],
            "stratum": item["stratum"],
            "base_mean_f1": item["mean_f1"],
            "base_exact_actor_count": item["exact_actor_count"],
            "base_nonzero_actor_count": item["nonzero_actor_count"],
            "selection_priority_sha256": item["selection_priority_sha256"],
        }

    sentinel_source = sorted(
        [item for item in units if item["exact_actor_count"] > 0],
        key=lambda item: (item["selection_priority_sha256"], item["unit_identity_sha256"]),
    )
    if (
        len(sentinel_source) != 4
        or sorted(item["exact_actor_count"] for item in sentinel_source) != [2, 4, 4, 4]
    ):
        raise ValueError("v61 sparse exact sentinel support changed")
    sentinel_ids = {item["unit_identity_sha256"] for item in sentinel_source}
    holdback_source = sorted(
        [item for item in units if item["panel_partition"] == "holdback"],
        key=lambda item: (item["stratum"], item["selection_priority_sha256"]),
    )
    if len(holdback_source) != 50 or any(
        item["unit_identity_sha256"] in sentinel_ids for item in holdback_source
    ):
        raise ValueError("v61 untouched holdback coverage changed")
    ranking_source = []
    for stratum, quota in RANKING_QUOTAS_V61.items():
        candidates = sorted([
            item for item in units
            if item["panel_partition"] == "selection_pool"
            and item["stratum"] == stratum
            and item["unit_identity_sha256"] not in sentinel_ids
        ], key=lambda item: (
            item["selection_priority_sha256"], item["unit_identity_sha256"],
        ))
        if len(candidates) < quota:
            raise ValueError(f"v61 insufficient {stratum} ranking candidates")
        ranking_source.extend(candidates[:quota])
    if len(ranking_source) != RANKING_UNITS_V61:
        raise RuntimeError("v61 ranking quota sum changed")
    ranking_ids = {item["unit_identity_sha256"] for item in ranking_source}
    holdback_ids = {item["unit_identity_sha256"] for item in holdback_source}
    if (
        len(ranking_ids) != 64 or len(holdback_ids) != 50
        or ranking_ids & holdback_ids or ranking_ids & sentinel_ids
        or holdback_ids & sentinel_ids
    ):
        raise RuntimeError("v61 ranking/holdback/sentinel panels overlap")
    reserve_source = sorted([
        item for item in units
        if item["unit_identity_sha256"] not in ranking_ids | holdback_ids | sentinel_ids
    ], key=lambda item: item["unit_identity_sha256"])
    if len(reserve_source) != 90:
        raise RuntimeError("v61 reserve partition changed")
    result = {
        "schema": "v61-content-free-ranking-holdback-sentinel-panels",
        "status": "cpu_only_preview_launch_ineligible",
        "source_v61a_gate_passed": False,
        "gpu_launch_authorized": False,
        "ranking_quotas": dict(RANKING_QUOTAS_V61),
        "ranking": [compact(item, "ranking", index)
                    for index, item in enumerate(ranking_source)],
        "untouched_holdback": [compact(item, "untouched_holdback", index)
                               for index, item in enumerate(holdback_source)],
        "exact_sentinel": [compact(item, "exact_sentinel", index)
                           for index, item in enumerate(sentinel_source)],
        "unused_reserve": [compact(item, "unused_reserve", index)
                           for index, item in enumerate(reserve_source)],
        "conflict_units_are_connected_components_over_document_url_lineage_and_semantic_identity": True,
        "v61a_baseline_model_outcomes_used_for_train_only_stratification": True,
        "future_candidate_outcomes_used_for_panel_selection": False,
        "protected_or_holdback_outcomes_used": False,
        "train_only_adaptive_design": True,
        "protected_semantics_opened": False,
        "panel_manifest_sha256": canonical_sha256_v61({
            "ranking": sorted(ranking_ids),
            "holdback": sorted(holdback_ids),
            "sentinel": sorted(sentinel_ids),
            "reserve": sorted(item["unit_identity_sha256"] for item in reserve_source),
        }),
    }
    result["content_sha256_before_self_field"] = canonical_sha256_v61(result)
    return result


def _validate_metric_array_v61(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != 4 or array.shape[1:] != (ACTORS_V61, PASSES_PER_STATE_V61, 3)
        or array.shape[0] < 4 or not np.isfinite(array).all()
        or (array < 0.0).any() or (array > 1.0).any()
        or not np.isin(array[..., 1:], [0.0, 1.0]).all()
        or not np.array_equal(array[..., 2], (array[..., 0] > 0.0).astype(float))
        or (array[..., 1] > array[..., 2]).any()
    ):
        raise ValueError(f"v61 malformed {label} unit×actor×pass metrics")
    return array


def _unit_instability_v61(array: np.ndarray) -> np.ndarray:
    # array: units × replicas × metrics
    replicas = array.shape[1]
    pairwise = []
    for left in range(replicas):
        for right in range(left + 1, replicas):
            pairwise.append(np.abs(array[:, left, 0] - array[:, right, 0]))
    mean_pairwise = np.mean(np.stack(pairwise, axis=1), axis=1)
    exact_disagreement = (
        np.max(array[..., 1], axis=1) != np.min(array[..., 1], axis=1)
    ).astype(np.float64)
    nonzero_disagreement = (
        np.max(array[..., 2], axis=1) != np.min(array[..., 2], axis=1)
    ).astype(np.float64)
    return (
        STABILITY_WEIGHTS_V61["mean_pairwise_absolute_f1_delta"] * mean_pairwise
        + STABILITY_WEIGHTS_V61["exact_label_disagreement"] * exact_disagreement
        + STABILITY_WEIGHTS_V61["nonzero_label_disagreement"] * nonzero_disagreement
    )


def _ordered_quantile_v61(values: np.ndarray, probability: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    index = int(math.floor(probability * (len(ordered) - 1)))
    return float(ordered[index])


def paired_unit_actor_bootstrap_v61(
    reference, candidate, *, seed: int = BOOTSTRAP_SEED_V61,
    replicates: int = BOOTSTRAP_REPLICATES_V61,
) -> dict:
    """Paired block bootstrap over conservative units and actor/pass replicas."""
    reference = _validate_metric_array_v61(reference, "reference")
    candidate = _validate_metric_array_v61(candidate, "candidate")
    if reference.shape != candidate.shape or replicates < 100:
        raise ValueError("v61 paired bootstrap coverage changed")
    units = reference.shape[0]
    reference_flat = reference.reshape(units, REPLICAS_PER_UNIT_V61, 3)
    candidate_flat = candidate.reshape(units, REPLICAS_PER_UNIT_V61, 3)
    deltas = candidate_flat - reference_flat
    stability_improvement = (
        _unit_instability_v61(reference_flat) - _unit_instability_v61(candidate_flat)
    )
    point = {
        "f1_delta": float(np.mean(deltas[..., 0])),
        "exact_delta": float(np.mean(deltas[..., 1])),
        "nonzero_delta": float(np.mean(deltas[..., 2])),
        "stability_improvement": float(np.mean(stability_improvement)),
    }
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    samples = {key: np.empty(replicates, dtype=np.float64) for key in point}
    for sample in range(replicates):
        unit_indices = rng.integers(0, units, size=units)
        replica_indices = rng.integers(0, REPLICAS_PER_UNIT_V61, size=units)
        selected = deltas[unit_indices, replica_indices]
        samples["f1_delta"][sample] = np.mean(selected[:, 0])
        samples["exact_delta"][sample] = np.mean(selected[:, 1])
        samples["nonzero_delta"][sample] = np.mean(selected[:, 2])
        samples["stability_improvement"][sample] = np.mean(
            stability_improvement[unit_indices]
        )
    lcb = {key: _ordered_quantile_v61(values, BOOTSTRAP_ALPHA_V61)
           for key, values in samples.items()}
    composite = (
        GENERATION_COMPOSITE_WEIGHTS_V61["f1_delta_lcb"] * lcb["f1_delta"]
        + GENERATION_COMPOSITE_WEIGHTS_V61["nonzero_delta_lcb"]
        * lcb["nonzero_delta"]
        + GENERATION_COMPOSITE_WEIGHTS_V61["stability_improvement_lcb"]
        * lcb["stability_improvement"]
    )
    result = {
        "schema": "v61-paired-conflict-unit-actor-pass-bootstrap",
        "units": units,
        "actors": ACTORS_V61,
        "passes": PASSES_PER_STATE_V61,
        "paired_replicas_per_unit": REPLICAS_PER_UNIT_V61,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": int(seed),
        "one_sided_alpha": BOOTSTRAP_ALPHA_V61,
        "point": point,
        "lower_confidence_bounds": lcb,
        "generation_composite_weights": dict(GENERATION_COMPOSITE_WEIGHTS_V61),
        "robust_generation_fitness": float(composite),
        "content_free_numeric_only": True,
    }
    result["content_sha256"] = canonical_sha256_v61(result)
    return result


def exact_sentinel_gate_v61(reference, candidate) -> dict:
    reference = _validate_metric_array_v61(reference, "sentinel reference")
    candidate = _validate_metric_array_v61(candidate, "sentinel candidate")
    if reference.shape != candidate.shape or reference.shape[0] != 4:
        raise ValueError("v61 exact sentinel must contain four sparse-support units")
    reference_counts = np.sum(reference[..., 1], axis=(1, 2)).astype(int)
    candidate_counts = np.sum(candidate[..., 1], axis=(1, 2)).astype(int)
    checks = {
        "total_exact_noninferiority": int(np.sum(candidate_counts)) >= int(np.sum(reference_counts)),
        "per_unit_exact_noninferiority": bool(np.all(candidate_counts >= reference_counts)),
        "three_all_exact_reference_units_present": int(np.sum(reference_counts == 8)) >= 3,
        "any_exact_reference_units": int(np.sum(reference_counts > 0)) == 4,
    }
    return {
        "schema": "v61-sparse-exact-sentinel-gate",
        "reference_exact_counts_by_unit": reference_counts.tolist(),
        "candidate_exact_counts_by_unit": candidate_counts.tolist(),
        "checks": checks,
        "passed": all(checks.values()),
    }
