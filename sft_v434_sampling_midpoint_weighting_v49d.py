#!/usr/bin/env python3
"""Deterministic equal-control and lambda=0.5 V49A weights for V49D."""

from __future__ import annotations

import math
from collections import defaultdict

import run_sft_train_only_control_v36a as engine
import sft_source_balanced_weighting_v49b as v49b


ARMS = ("v434_equal", "v434_source50")
ROW_COUNT = 448
INTERPOLATION_LAMBDA = 0.5
MIN_SOURCE50_MULTIPLIER = 1.0 + 0.5 * ((2.0 / 3.0) - 1.0)
MAX_SOURCE50_MULTIPLIER = 5.0 / 4.0
EXPECTED = {
    "v434_equal": {
        "normalized_weight_sha256": "a8cbc597f865123100de870fdbc22a2529ed5ec3534531f897d27763baa4492f",
        "trainer_weight_sha256": "7db37e1a41c9e3e422ce1dd2cab037f7bffb9dc7fbf21013ce816a892602d6a8",
        "per_row_sha256": "215668010e05c5892e0c8cf1db6a95a308dd20c1c4aa3bfbab0952a223087b80",
        "per_source_sha256": "5280d146db573b09949e3abd630863cc781def88a7799071dfe323406531eeb1",
        "per_category_sha256": "0fb1a79c61d9366957011f7db76b8de4dbb0ad108c33cb504eee477daad08f7b",
    },
    "v434_source50": {
        "normalized_weight_sha256": "725e115ac0e4c4f20653e6c95f163edda0602621ff9d83f0c7d94b5f81b29890",
        "trainer_weight_sha256": "49e160524a46bd8a8dd4407d51dd23415df9591110b5a34c0f9de1aa80ac9f16",
        "per_row_sha256": "4733367b8fb29277f9e8119dbc6df4c019436eb7b1280ce9b18528094f9ed5f6",
        "per_source_sha256": "84a51e3237d2e35dfcc4a94b655bb742bead9f77ecd1c4d86cc2aa322bd9a4c9",
        "per_category_sha256": "450c811da6ff0a6d63a673f09f0ab079105f97c1efdd4662bc995ba54900c510",
    },
}


def _weight_identity(rows: list[dict], weights: list[float]) -> str:
    return engine.canonical_sha256([
        {"fact_id": row["fact_id"], "normalized_weight_hex": weight.hex()}
        for row, weight in zip(rows, weights, strict=True)
    ])


def _trainer_identity(rows: list[dict], weights: list[float]) -> str:
    return engine.canonical_sha256([
        {"fact_id": row["fact_id"], "example_weight_hex": weight.hex()}
        for row, weight in zip(rows, weights, strict=True)
    ])


def _compact(complete: dict) -> dict:
    excluded = {
        "per_row", "per_source", "per_category", "v49b_parent_compact",
        "equal_unit_parent_audit", "access_firewall",
        "content_sha256_before_self_field",
    }
    return {key: value for key, value in complete.items() if key not in excluded}


def _build_arm_audit(
    rows: list[dict],
    arm: str,
    equal_normalized: list[float],
    full_normalized: list[float],
    full_multipliers: list[float],
    arm_normalized: list[float],
    arm_multipliers: list[float],
    full_audit: dict,
    equal_audit: dict,
) -> tuple[list[float], dict]:
    row_count = len(rows)
    trainer_weights = [weight * row_count for weight in arm_normalized]
    full_rows = full_audit["per_row"]
    categories = [record["category"] for record in full_rows]
    unit_ids = [record["unit_identity_sha256"] for record in full_rows]
    unit_sets = defaultdict(set)
    source_rows = defaultdict(int)
    source_units = defaultdict(set)
    category_rows = defaultdict(int)
    equal_source = defaultdict(float)
    full_source = defaultdict(float)
    arm_source = defaultdict(float)
    equal_category = defaultdict(float)
    arm_category = defaultdict(float)
    for index, (row, category, unit_id, before, full, after) in enumerate(zip(
        rows, categories, unit_ids, equal_normalized, full_normalized,
        arm_normalized, strict=True,
    )):
        source = row["source"]
        unit_sets[category].add(unit_id)
        source_rows[source] += 1
        source_units[source].add(unit_id)
        category_rows[category] += 1
        equal_source[source] += before
        full_source[source] += full
        arm_source[source] += after
        equal_category[category] += before
        arm_category[category] += after

    source_table = [{
        "source": source,
        "rows": source_rows[source],
        "conflict_units_touched": len(source_units[source]),
        "equal_mass": equal_source[source],
        "v49a_full_mass": full_source[source],
        "arm_mass": arm_source[source],
        "arm_minus_equal": arm_source[source] - equal_source[source],
        "effective_mass_multiplier": arm_source[source] / equal_source[source],
    } for source in sorted(equal_source)]
    category_table = [{
        "category": category,
        "rows": category_rows[category],
        "conflict_units": len(unit_sets[category]),
        "equal_mass": equal_category[category],
        "arm_mass": arm_category[category],
        "mass_delta": arm_category[category] - equal_category[category],
    } for category in sorted(equal_category)]
    if max(abs(record["mass_delta"]) for record in category_table) > 1e-15:
        raise RuntimeError(f"V49D {arm} category objective mass changed")

    per_row = [{
        "row_index": index,
        "fact_id": row["fact_id"],
        "document_sha256": row["document_sha256"],
        "source": row["source"],
        "category": categories[index],
        "unit_identity_sha256": unit_ids[index],
        "equal_normalized_weight_hex": equal_normalized[index].hex(),
        "v49a_full_normalized_weight_hex": full_normalized[index].hex(),
        "arm_normalized_weight_hex": arm_normalized[index].hex(),
        "trainer_example_weight_hex": trainer_weights[index].hex(),
        "v49a_full_multiplier_hex": full_multipliers[index].hex(),
        "arm_multiplier_hex": arm_multipliers[index].hex(),
    } for index, row in enumerate(rows)]
    normalized_identity = _weight_identity(rows, arm_normalized)
    compact = {
        "schema": "specialist-v434-sampling-interpolation-weighting-v49d",
        "arm": arm,
        "rows": row_count,
        "conflict_units": 208,
        "interpolation_lambda": 0.0 if arm == "v434_equal" else 0.5,
        "interpolation_formula": (
            "w_arm=(1-lambda)*w_equal+lambda*w_v49a_full"
        ),
        "ordinary_row_mean_weight": math.fsum(trainer_weights) / row_count,
        "minimum_row_weight": min(trainer_weights),
        "maximum_row_weight": max(trainer_weights),
        "identity_sha256": normalized_identity,
        "equal_normalized_weight_sha256": _weight_identity(
            rows, equal_normalized
        ),
        "v49a_full_normalized_weight_sha256": _weight_identity(
            rows, full_normalized
        ),
        "trainer_example_weight_identity_sha256": _trainer_identity(
            rows, trainer_weights
        ),
        "per_row_identity_sha256": engine.canonical_sha256(per_row),
        "per_source_identity_sha256": engine.canonical_sha256(source_table),
        "per_category_identity_sha256": engine.canonical_sha256(category_table),
        "ordered_fact_id_sha256": engine.canonical_sha256([
            row["fact_id"] for row in rows
        ]),
        "ordered_document_sha256": engine.canonical_sha256([
            row["document_sha256"] for row in rows
        ]),
        "unique_document_membership_sha256": engine.canonical_sha256(
            sorted({row["document_sha256"] for row in rows})
        ),
        "root_membership_sha256": v49b.ROOT_MEMBERSHIP_SHA256,
        "v434_train_jsonl_sha256": v49b.v49a.V434_TRAIN_SHA256,
        "minimum_applied_multiplier": min(arm_multipliers),
        "maximum_applied_multiplier": max(arm_multipliers),
        "preregistered_multiplier_range_exact_rationals": (
            ["1", "1"] if arm == "v434_equal" else ["5/6", "5/4"]
        ),
        "full_parent_minimum_multiplier": min(full_multipliers),
        "full_parent_maximum_multiplier": max(full_multipliers),
        "category_masses_preserved_exactly": True,
        "only_per_row_example_weights_differ_between_v49d_arms": True,
    }
    expected = EXPECTED[arm]
    observed = {
        "normalized_weight_sha256": normalized_identity,
        "trainer_weight_sha256": compact[
            "trainer_example_weight_identity_sha256"
        ],
        "per_row_sha256": compact["per_row_identity_sha256"],
        "per_source_sha256": compact["per_source_identity_sha256"],
        "per_category_sha256": compact["per_category_identity_sha256"],
    }
    if any(value != "PENDING" for value in expected.values()) and observed != expected:
        raise RuntimeError(f"V49D {arm} sealed weight identity changed")
    complete = {
        **compact,
        "equal_unit_parent_audit": equal_audit,
        "v49b_parent_compact": v49b.compact_weighting_audit_v49b(full_audit),
        "per_row": per_row,
        "per_source": source_table,
        "per_category": category_table,
        "access_firewall": {
            "train_rows_and_train_metadata_opened": True,
            "shadow_semantics_opened": False,
            "eval_ood_holdout_semantics_opened": False,
            "gpu_accessed": False,
        },
    }
    complete["content_sha256_before_self_field"] = engine.canonical_sha256(
        complete
    )
    return trainer_weights, complete


def compute_v49d(rows: list[dict]) -> dict[str, tuple[list[float], dict]]:
    """Construct both V49D arms from one exact V49B computation."""
    payload = v49b.v49a.frozen.jsonl_bytes(rows)
    if (
        len(rows) != ROW_COUNT
        or v49b.v49a.hashlib.sha256(payload).hexdigest()
        != v49b.v49a.V434_TRAIN_SHA256
    ):
        raise RuntimeError("V49D exact v434 train projection changed")
    equal_trainer, equal_audit = v49b.BASE_EQUAL_UNIT_ASSIGNER_V49B(rows)
    equal_normalized = [weight / ROW_COUNT for weight in equal_trainer]
    _full_trainer, full_audit = v49b.compute_source_balanced_weights_v49b(rows)
    full_normalized = [
        float.fromhex(record["alternative_normalized_weight_hex"])
        for record in full_audit["per_row"]
    ]
    full_multipliers = [
        float.fromhex(record["applied_multiplier_hex"])
        for record in full_audit["per_row"]
    ]
    if _weight_identity(rows, equal_normalized) != v49b.CURRENT_NORMALIZED_WEIGHT_SHA256:
        raise RuntimeError("V49D equal-control identity changed")
    source50 = [
        0.5 * before + 0.5 * after
        for before, after in zip(equal_normalized, full_normalized, strict=True)
    ]
    source50_multipliers = [
        1.0 + 0.5 * (multiplier - 1.0) for multiplier in full_multipliers
    ]
    if (
        min(source50_multipliers) != MIN_SOURCE50_MULTIPLIER
        or max(source50_multipliers) != MAX_SOURCE50_MULTIPLIER
        or not math.isclose(math.fsum(source50), 1.0, abs_tol=1e-15, rel_tol=0)
    ):
        raise RuntimeError("V49D lambda=0.5 exact interpolation range changed")
    return {
        "v434_equal": _build_arm_audit(
            rows, "v434_equal", equal_normalized, full_normalized,
            full_multipliers, equal_normalized, [1.0] * ROW_COUNT,
            full_audit, equal_audit,
        ),
        "v434_source50": _build_arm_audit(
            rows, "v434_source50", equal_normalized, full_normalized,
            full_multipliers, source50, source50_multipliers,
            full_audit, equal_audit,
        ),
    }


def compact_weighting_audit_v49d(complete: dict) -> dict:
    return _compact(complete)


def assign_equal_weights_v49d(rows: list[dict]) -> tuple[list[float], dict]:
    weights, complete = compute_v49d(rows)["v434_equal"]
    return weights, _compact(complete)


def assign_source50_weights_v49d(rows: list[dict]) -> tuple[list[float], dict]:
    weights, complete = compute_v49d(rows)["v434_source50"]
    return weights, _compact(complete)
