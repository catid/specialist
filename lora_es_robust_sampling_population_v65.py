#!/usr/bin/env python3
"""Pure contracts for the V65 robust-sampling LoRA-ES population.

V65 is a measurement-only population pass.  It repeats the exact reliable
V53 sigma=0.0048 antithetic states on V61's 64 conflict-unit ranking panel,
with four actors and two passes per signed state.  No candidate update, train
holdback, exact sentinel, OOD, or terminal holdout is opened here.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

import lora_es_nested_population_v52 as design52
import lora_es_robust_paired_hpo_v61 as design61


ROOT = Path(__file__).resolve().parent
PREVIEW_V61 = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_paired_block_bootstrap_v61_preview.json"
).resolve()
PREVIEW_V61_FILE_SHA256 = (
    "a9ce060ce81df5b1fbddcc40db572fe56974ea6dfb6ef2e6ebf3e81925a400e2"
)
PREVIEW_V61_CONTENT_SHA256 = (
    "1b25f3c667fc0e9eeddc19f1d20aebc70c2a0127db0c3eafe11c2f19fb35a0f0"
)
V53_SIGMA_ARM = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v53_lora_es_sigma_discrimination/sigma_0p0048_arm_v53.json"
).resolve()
V53_SIGMA_ARM_FILE_SHA256 = (
    "f73adf79eb1cc10315a088568cd68816a5233fdb46217bb5ed55e3c0be778659"
)
V53_SIGMA_ARM_CONTENT_SHA256 = (
    "f60eb4f5c66a3f26571c07111ec1ccf9c26ecd8121e3c160007c5ad7220ae486"
)
V53_CANDIDATE_IDENTITY_INVENTORY_SHA256 = (
    "da18f263be3db7a0febe79cc90eb8b2be2677d80b892760544fe1023d0096364"
)
V53_RUNTIME_IDENTITY_INVENTORY_SHA256 = (
    "edd251c4209bc1e7f1aee4d16aacccef29292da51a889f881d7e4cf3efb015d5"
)
V61C_PANEL = (
    ROOT / "experiments/eggroll_es_hpo/datasets/"
    "v61c_paired_null_calibration_panel.json"
).resolve()
V61C_PANEL_FILE_SHA256 = (
    "92e0c6160bfc7884a00be4c34c427685dcb2bf5a6aa8c3820f5c53e225f8091c"
)
V61C_PANEL_CONTENT_SHA256 = (
    "ca0a947e6437c0d84360176087b0a9dab12b79cf6ba1be8f965b24e9f4ec7ba4"
)
V61C_ROWS = (
    ROOT / "experiments/eggroll_es_hpo/datasets/"
    "v61c_paired_null_calibration_rows.jsonl"
).resolve()
V61C_ROWS_FILE_SHA256 = (
    "9c1b7f69595cf70ef045259e2097c39546e9f1d84a6b0870fcb14e987655079a"
)
V61C_FINALIZED = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v61c_v434_identical_state_paired_evaluator_calibration/"
    "paired_null_finalized_v61c.json"
).resolve()
V61C_FINALIZED_FILE_SHA256 = (
    "d3d5eabf1e5d9b0bed2dfd2a355ed5eb839a22cb4bcdea58af0ab84231042d46"
)
V61C_FINALIZED_CONTENT_SHA256 = (
    "7bc9735dea87ae8bf2374bcefb7c290b7bb273f2394b44d54dc1fa69e8e851c0"
)

POPULATION_SIZE_V65 = 16
ACTORS_V65 = 4
PASSES_PER_SIGNED_STATE_V65 = 2
RANKING_UNITS_V65 = 64
SIGMA_V65 = 0.0048
SEEDS_V65 = tuple(design52.P16_SEEDS_V52)
METRIC_ORDER_V65 = ("f1", "exact", "nonzero")
STATE_COUNT_V65 = (
    POPULATION_SIZE_V65 * 2 * PASSES_PER_SIGNED_STATE_V65
)
GENERATION_COMPLETIONS_V65 = (
    STATE_COUNT_V65 * ACTORS_V65 * RANKING_UNITS_V65
)
BOOTSTRAP_REPLICATES_V65 = 4096
BOOTSTRAP_SEED_V65 = 2_026_071_612
BOOTSTRAP_ALPHA_V65 = 0.05
V61C_RANKING_F1_NULL_HALFWIDTH = 0.00039520537364525746
MINIMUM_SPLIT_PASS_SPEARMAN_V65 = 0.50
MINIMUM_SPLIT_PASS_CENTERED_COSINE_V65 = 0.50
MINIMUM_DIRECTION_SPREAD_V65 = 2.0 * V61C_RANKING_F1_NULL_HALFWIDTH
MINIMUM_STABILITY_DIRECTION_SPREAD_V65 = 1e-12


def canonical_sha256_v65(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def file_sha256_v65(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def self_content_sha256_v65(value: dict) -> str:
    return canonical_sha256_v65({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def read_exact_self_hashed_v65(
    path: Path, file_sha256: str, content_sha256: str,
) -> dict:
    if file_sha256_v65(path) != file_sha256:
        raise RuntimeError(f"v65 sealed file changed: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("content_sha256_before_self_field") != content_sha256
        or self_content_sha256_v65(value) != content_sha256
    ):
        raise RuntimeError(f"v65 sealed content changed: {path}")
    return value


def build_ranking_panel_v65(preview: dict) -> dict:
    """Project only V61's frozen ranking hashes into a launch panel."""
    panels = preview.get("panels", {})
    ranking = panels.get("ranking", [])
    holdback = panels.get("untouched_holdback", [])
    sentinel = panels.get("exact_sentinel", [])
    reserve = panels.get("unused_reserve", [])
    role_sets = [
        {row.get("unit_identity_sha256") for row in role}
        for role in (ranking, holdback, sentinel, reserve)
    ]
    if (
        preview.get("schema")
        != "matched-lora-es-paired-block-bootstrap-hpo-preview-v61"
        or preview.get("status") != "cpu_only_preview_frozen_launch_ineligible"
        or preview.get("gpu_launch_authorized") is not False
        or preview.get("protected_semantics_opened") is not False
        or len(ranking) != RANKING_UNITS_V65
        or [len(role) for role in (holdback, sentinel, reserve)] != [50, 4, 90]
        or any(
            role_sets[left] & role_sets[right]
            for left in range(4) for right in range(left + 1, 4)
        )
        or len(set.union(*role_sets)) != 208
        or any(row.get("role_index") != index
               for index, row in enumerate(ranking))
    ):
        raise RuntimeError("v65 frozen V61 panel contract changed")
    items = [{
        "request_index": index,
        "row_sha256": row["row_sha256"],
        "unit_identity_sha256": row["unit_identity_sha256"],
        "stratum": row["stratum"],
        "selection_priority_sha256": row["selection_priority_sha256"],
    } for index, row in enumerate(ranking)]
    result = {
        "schema": "v65-robust-sampling-ranking-panel",
        "status": "content_free_hash_only_before_population",
        "source_preview_file_sha256": PREVIEW_V61_FILE_SHA256,
        "source_preview_content_sha256": PREVIEW_V61_CONTENT_SHA256,
        "source_panel_manifest_sha256": panels["panel_manifest_sha256"],
        "ranking_units": RANKING_UNITS_V65,
        "ranking_quotas": dict(panels["ranking_quotas"]),
        "items": items,
        "request_order_sha256": canonical_sha256_v65([
            row["row_sha256"] for row in items
        ]),
        "unit_order_sha256": canonical_sha256_v65([
            row["unit_identity_sha256"] for row in items
        ]),
        "untouched_partition_counts": {
            "train_holdback": len(holdback),
            "exact_sentinel": len(sentinel),
            "unused_reserve": len(reserve),
        },
        "ranking_disjoint_from_every_untouched_partition": True,
        "future_candidate_outcomes_used_for_selection": False,
        "question_answer_or_generation_text_persisted": False,
        "protected_semantics_opened": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256_v65(result)
    return result


def state_derivations_v65() -> list[dict]:
    """Counterbalance the two temporal passes of each exact V53 state."""
    result = []
    for direction, seed in enumerate(SEEDS_V65):
        # Pass 0 observes plus before minus; pass 1 observes minus before plus.
        # This preserves exact V53 values while counterbalancing state order.
        for label, sign, pass_index in (
            ("plus", 1, 0), ("minus", -1, 0),
            ("minus", -1, 1), ("plus", 1, 1),
        ):
            result.append({
                "state_index": len(result),
                "direction": direction,
                "label": label,
                "sign": sign,
                "pass_index": pass_index,
                "candidate_after_antithetic_peer": (
                    (label == "minus" and pass_index == 0)
                    or (label == "plus" and pass_index == 1)
                ),
                "seed": seed,
                "sigma": SIGMA_V65,
                "master_sha256": design52.MASTER_SHA256_V52,
                "derivation": (
                    "V41A antithetic candidate from pinned FP32 master"
                ),
            })
    if len(result) != STATE_COUNT_V65:
        raise RuntimeError("v65 state grid changed")
    return result


def expected_v53_state_identities_v65(arm: dict) -> list[dict]:
    """Extract the exact ordered 32-state identities from sealed V53."""
    states = arm.get("timing", {}).get("states", [])
    result = []
    for direction in range(POPULATION_SIZE_V65):
        for label, sign in (("plus", 1), ("minus", -1)):
            index = 2 * direction + int(label == "minus")
            if index >= len(states):
                raise RuntimeError("v65 V53 state identity coverage changed")
            state = states[index].get("state", {})
            if (
                state.get("state_index") != index
                or state.get("direction") != direction
                or state.get("label") != label
                or state.get("sign") != sign
                or state.get("seed") != SEEDS_V65[direction]
                or state.get("sigma") != SIGMA_V65
            ):
                raise RuntimeError("v65 V53 exact state order changed")
            result.append({
                "direction": direction,
                "label": label,
                "sign": sign,
                "seed": SEEDS_V65[direction],
                "candidate_identity_sha256": state[
                    "candidate_identity_sha256"
                ],
                "runtime_values_sha256": state["runtime_values_sha256"],
            })
    candidates = [row["candidate_identity_sha256"] for row in result]
    runtimes = [row["runtime_values_sha256"] for row in result]
    if (
        len(states) != 32 or len(set(candidates)) != 32
        or len(set(runtimes)) != 32
        or canonical_sha256_v65(candidates)
        != V53_CANDIDATE_IDENTITY_INVENTORY_SHA256
        or canonical_sha256_v65(runtimes)
        != V53_RUNTIME_IDENTITY_INVENTORY_SHA256
    ):
        raise RuntimeError("v65 V53 ordered identity inventory changed")
    return result


def _validate_metric_matrix_v65(value, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    expected = (
        RANKING_UNITS_V65, ACTORS_V65,
        PASSES_PER_SIGNED_STATE_V65, len(METRIC_ORDER_V65),
    )
    if (
        array.shape != expected or not np.isfinite(array).all()
        or (array < 0.0).any() or (array > 1.0).any()
        or not np.isin(array[..., 1:], [0.0, 1.0]).all()
        or (
            (array[..., 1] == 1.0)
            & (np.abs(array[..., 0] - 1.0) > 1e-12)
        ).any()
        or not np.array_equal(
            array[..., 2], (array[..., 0] > 0.0).astype(np.float64),
        )
        or (array[..., 1] > array[..., 2]).any()
    ):
        raise ValueError(f"v65 malformed {label} metric matrix")
    return array


def _unit_norm_center_v65(values: list[float], label: str) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (POPULATION_SIZE_V65,) or not np.isfinite(array).all():
        raise ValueError(f"v65 malformed {label} direction vector")
    centered = array - np.mean(array)
    norm = float(np.linalg.norm(centered))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError(f"v65 {label} direction has zero population spread")
    result = centered / norm
    if abs(float(np.sum(result))) > 1e-10:
        raise RuntimeError(f"v65 {label} centered direction drifted")
    return result.tolist()


def _unit_instability_v65(array: np.ndarray) -> np.ndarray:
    """Return one continuous disagreement score per conflict unit."""
    replicas = array.shape[1]
    pairwise = [
        np.abs(array[:, left, 0] - array[:, right, 0])
        for left in range(replicas) for right in range(left + 1, replicas)
    ]
    mean_pairwise = np.mean(np.stack(pairwise, axis=1), axis=1)
    exact_disagreement = (
        np.max(array[..., 1], axis=1) != np.min(array[..., 1], axis=1)
    ).astype(np.float64)
    nonzero_disagreement = (
        np.max(array[..., 2], axis=1) != np.min(array[..., 2], axis=1)
    ).astype(np.float64)
    weights = design61.STABILITY_WEIGHTS_V61
    return (
        weights["mean_pairwise_absolute_f1_delta"] * mean_pairwise
        + weights["exact_label_disagreement"] * exact_disagreement
        + weights["nonzero_label_disagreement"] * nonzero_disagreement
    )


def frozen_bootstrap_indices_v65() -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED_V65))
    return rng.integers(
        0, RANKING_UNITS_V65,
        size=(BOOTSTRAP_REPLICATES_V65, RANKING_UNITS_V65),
        dtype=np.int64,
    )


def _ordered_quantile_v65(values: np.ndarray, probability: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    index = int(math.floor(probability * (len(ordered) - 1)))
    return float(ordered[index])


def paired_cluster_bootstrap_v65(
    reference, candidate, *, bootstrap_indices: np.ndarray,
) -> dict:
    """Bootstrap units while preserving/averaging all eight paired replicas."""
    reference = _validate_metric_matrix_v65(reference, "reference")
    candidate = _validate_metric_matrix_v65(candidate, "candidate")
    indices = np.asarray(bootstrap_indices, dtype=np.int64)
    if (
        indices.shape != (BOOTSTRAP_REPLICATES_V65, RANKING_UNITS_V65)
        or (indices < 0).any() or (indices >= RANKING_UNITS_V65).any()
    ):
        raise ValueError("v65 frozen bootstrap index matrix changed")
    reference_flat = reference.reshape(RANKING_UNITS_V65, -1, 3)
    candidate_flat = candidate.reshape(RANKING_UNITS_V65, -1, 3)
    delta_unit_means = np.mean(candidate_flat - reference_flat, axis=1)
    stability_improvement = (
        _unit_instability_v65(reference_flat)
        - _unit_instability_v65(candidate_flat)
    )
    point = {
        "f1_delta": float(np.mean(delta_unit_means[:, 0])),
        "exact_delta": float(np.mean(delta_unit_means[:, 1])),
        "nonzero_delta": float(np.mean(delta_unit_means[:, 2])),
        "stability_improvement": float(np.mean(stability_improvement)),
    }
    samples = {
        "f1_delta": np.mean(delta_unit_means[indices, 0], axis=1),
        "exact_delta": np.mean(delta_unit_means[indices, 1], axis=1),
        "nonzero_delta": np.mean(delta_unit_means[indices, 2], axis=1),
        "stability_improvement": np.mean(stability_improvement[indices], axis=1),
    }
    lower = {
        key: _ordered_quantile_v65(value, BOOTSTRAP_ALPHA_V65)
        for key, value in samples.items()
    }
    upper = {
        key: _ordered_quantile_v65(value, 1.0 - BOOTSTRAP_ALPHA_V65)
        for key, value in samples.items()
    }
    weights = design61.GENERATION_COMPOSITE_WEIGHTS_V61
    composite = (
        weights["f1_delta_lcb"] * lower["f1_delta"]
        + weights["nonzero_delta_lcb"] * lower["nonzero_delta"]
        + weights["stability_improvement_lcb"]
        * lower["stability_improvement"]
    )
    return {
        "schema": "v65-paired-conflict-unit-cluster-bootstrap",
        "units": RANKING_UNITS_V65,
        "actors": ACTORS_V65,
        "passes": PASSES_PER_SIGNED_STATE_V65,
        "paired_replicas_per_unit_preserved_and_averaged": 8,
        "resampled_axis": "conflict_unit_only",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES_V65,
        "bootstrap_seed": BOOTSTRAP_SEED_V65,
        "bootstrap_index_matrix_sha256": hashlib.sha256(
            indices.astype("<i8", copy=False).tobytes(order="C")
        ).hexdigest(),
        "one_sided_alpha": BOOTSTRAP_ALPHA_V65,
        "point": point,
        "lower_confidence_bounds": lower,
        "upper_confidence_bounds": upper,
        "generation_composite_weights": dict(weights),
        "robust_generation_fitness": float(composite),
    }


def _rankdata_average_v65(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _split_pass_reliability_v65(
    signed_metrics: dict, robust_raw: list[float],
    stability_raw: list[float],
) -> dict:
    pass_scores = []
    weights = design61.GENERATION_COMPOSITE_WEIGHTS_V61
    for pass_index in range(PASSES_PER_SIGNED_STATE_V65):
        direction_scores = []
        for direction in range(POPULATION_SIZE_V65):
            plus = _validate_metric_matrix_v65(
                signed_metrics["plus"][direction], "split-pass plus",
            )[:, :, pass_index, :]
            minus = _validate_metric_matrix_v65(
                signed_metrics["minus"][direction], "split-pass minus",
            )[:, :, pass_index, :]
            delta = plus - minus
            plus_flat = plus.reshape(RANKING_UNITS_V65, ACTORS_V65, 3)
            minus_flat = minus.reshape(RANKING_UNITS_V65, ACTORS_V65, 3)
            stability = float(np.mean(
                _unit_instability_v65(minus_flat)
                - _unit_instability_v65(plus_flat)
            ))
            direction_scores.append(float(
                weights["f1_delta_lcb"] * np.mean(delta[..., 0])
                + weights["nonzero_delta_lcb"] * np.mean(delta[..., 2])
                + weights["stability_improvement_lcb"] * stability
            ))
        pass_scores.append(np.asarray(direction_scores, dtype=np.float64))
    left, right = pass_scores
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    cosine = float(np.dot(left_centered, right_centered) / denominator) \
        if denominator > 1e-15 else 0.0
    left_rank = _rankdata_average_v65(left)
    right_rank = _rankdata_average_v65(right)
    rank_denominator = float(
        np.linalg.norm(left_rank - np.mean(left_rank))
        * np.linalg.norm(right_rank - np.mean(right_rank))
    )
    spearman = float(np.dot(
        left_rank - np.mean(left_rank), right_rank - np.mean(right_rank),
    ) / rank_denominator) if rank_denominator > 1e-15 else 0.0
    spread = float(np.std(np.asarray(robust_raw, dtype=np.float64), ddof=0))
    stability_spread = float(np.std(
        np.asarray(stability_raw, dtype=np.float64), ddof=0,
    ))
    checks = {
        "split_pass_spearman_at_least_0p50": (
            spearman >= MINIMUM_SPLIT_PASS_SPEARMAN_V65
        ),
        "split_pass_centered_cosine_at_least_0p50": (
            cosine >= MINIMUM_SPLIT_PASS_CENTERED_COSINE_V65
        ),
        "direction_spread_strictly_above_twice_v61c_null_halfwidth": (
            spread > MINIMUM_DIRECTION_SPREAD_V65
        ),
        "stability_direction_spread_strictly_positive": (
            stability_spread > MINIMUM_STABILITY_DIRECTION_SPREAD_V65
        ),
    }
    return {
        "schema": "v65-split-pass-direction-discriminability",
        "pass_scores": [left.tolist(), right.tolist()],
        "split_pass_spearman": spearman,
        "split_pass_centered_cosine": cosine,
        "robust_direction_population_standard_deviation": spread,
        "stability_lcb_direction_population_standard_deviation": (
            stability_spread
        ),
        "v61c_ranking_f1_null_halfwidth": V61C_RANKING_F1_NULL_HALFWIDTH,
        "minimum_direction_spread": MINIMUM_DIRECTION_SPREAD_V65,
        "minimum_stability_direction_spread": (
            MINIMUM_STABILITY_DIRECTION_SPREAD_V65
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }


def analyze_signed_metrics_v65(signed_metrics: dict) -> dict:
    """Compute one paired conflict-unit bootstrap per antithetic direction."""
    if set(signed_metrics) != {"plus", "minus"}:
        raise ValueError("v65 signed metric labels changed")
    plus = signed_metrics["plus"]
    minus = signed_metrics["minus"]
    if len(plus) != POPULATION_SIZE_V65 or len(minus) != POPULATION_SIZE_V65:
        raise ValueError("v65 signed metric direction coverage changed")
    directions = []
    robust = []
    stability = []
    bootstrap_indices = frozen_bootstrap_indices_v65()
    for direction in range(POPULATION_SIZE_V65):
        reference = _validate_metric_matrix_v65(
            minus[direction], f"direction {direction} minus",
        )
        candidate = _validate_metric_matrix_v65(
            plus[direction], f"direction {direction} plus",
        )
        paired = paired_cluster_bootstrap_v65(
            reference, candidate, bootstrap_indices=bootstrap_indices,
        )
        robust.append(paired["robust_generation_fitness"])
        stability.append(
            paired["lower_confidence_bounds"]["stability_improvement"]
        )
        directions.append({
            "direction": direction,
            "seed": SEEDS_V65[direction],
            "paired_bootstrap": paired,
        })
    discriminability = _split_pass_reliability_v65(
        signed_metrics, robust, stability,
    )
    result = {
        "schema": "v65-robust-sampling-population-analysis",
        "population_size": POPULATION_SIZE_V65,
        "ranking_units": RANKING_UNITS_V65,
        "actors": ACTORS_V65,
        "passes_per_signed_state": PASSES_PER_SIGNED_STATE_V65,
        "generation_completions": GENERATION_COMPLETIONS_V65,
        "directions": directions,
        "robust_generation_raw": robust,
        "discriminability_gate": discriminability,
        "coefficients_actionable_for_later_preregistered_projection": (
            discriminability["passed"]
        ),
        "robust_generation_unit_norm_centered_coefficients": (
            _unit_norm_center_v65(robust, "robust generation")
            if discriminability["passed"] else None
        ),
        "stability_lcb_raw": stability,
        "stability_lcb_unit_norm_centered_coefficients": (
            _unit_norm_center_v65(stability, "stability LCB")
            if discriminability["passed"] else None
        ),
        "exact_used_for_population_ranking": False,
        "exact_reason": (
            "sparse exact support remains in the untouched sentinel panel"
        ),
        "candidate_update_or_projection_performed": False,
        "train_holdback_or_exact_sentinel_opened": False,
        "protected_semantics_opened": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256_v65(result)
    return result
