#!/usr/bin/env python3
"""Exact V49A alternative weights for the sealed V49B SFT arm."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import run_sft_train_only_control_v36a as engine
import v49a_train_only_source_temperature_weighting as v49a


ROOT = Path(__file__).resolve().parent
# V49A imports the same V42A module object that the runtime wrapper patches.
# Capture the committed equal-unit parent before that patch so constructing the
# alternative weights cannot recurse back into this function under torchrun.
BASE_EQUAL_UNIT_ASSIGNER_V49B = v49a.equal_unit.assign_equal_unit_weights
V49A_DESIGN = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "train_only_capped_source_temperature_weighting_v49a.json"
).resolve()
V49A_DESIGN_FILE_SHA256 = (
    "01fc746881e7c7ace9f7bddf14b85fecf3d40cf9bd5e15c613115a826af537c4"
)
V49A_DESIGN_CONTENT_SHA256 = (
    "cf376bb772c39f564050290a3958bba753434c4c8c2e4902f1347e0d7ee21826"
)
V49A_RUNTIME_FILE_SHA256 = (
    "76074a778950f69d3e84c49eb145c1546d65b17b3cdfd3d95b78987521fa38df"
)
ALTERNATIVE_NORMALIZED_WEIGHT_SHA256 = (
    "76dd9224cde643b2dd22123c2bd7952a830809a3ed84c977d61208da874de612"
)
CURRENT_NORMALIZED_WEIGHT_SHA256 = (
    "a8cbc597f865123100de870fdbc22a2529ed5ec3534531f897d27763baa4492f"
)
SOURCE_MASS_TABLE_SHA256 = (
    "3ab5d46ab944944137fabe1bd20db95000292c89804340f5d061b56bb1da77e0"
)
ROOT_MEMBERSHIP_SHA256 = (
    "e0e5eca72438aebc40d915eb824e03c42441c180d6ab585e1b375f83ddec3275"
)


def validate_v49a_design_v49b() -> dict:
    if (
        engine.file_sha256(V49A_DESIGN) != V49A_DESIGN_FILE_SHA256
        or engine.file_sha256(Path(v49a.__file__).resolve())
        != V49A_RUNTIME_FILE_SHA256
    ):
        raise RuntimeError("V49B committed V49A implementation changed")
    value = json.loads(V49A_DESIGN.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    identities = value.get("membership_and_weight_identities", {})
    if (
        content != V49A_DESIGN_CONTENT_SHA256
        or content != engine.canonical_sha256({
            key: item for key, item in value.items()
            if key != "content_sha256_before_self_field"
        })
        or value.get("schema")
        != "specialist-train-only-source-temperature-weighting-v49a"
        or value.get("status") != "complete_bounded_design_unlaunched"
        or value.get("artifact_role")
        != "preregisterable_design_only_no_launch_authority"
        or value.get("gpu_launch_authorized") is not False
        or value.get("training_launch_authorized") is not False
        or identities.get("train_jsonl_sha256") != v49a.V434_TRAIN_SHA256
        or identities.get("train_rows") != v49a.V434_TRAIN_ROWS
        or identities.get("train_conflict_units")
        != v49a.V434_TRAIN_CONFLICT_UNITS
        or identities.get("root_membership_sha256")
        != ROOT_MEMBERSHIP_SHA256
        or identities.get("current_normalized_weight_sha256")
        != CURRENT_NORMALIZED_WEIGHT_SHA256
        or identities.get("alternative_normalized_weight_sha256")
        != ALTERNATIVE_NORMALIZED_WEIGHT_SHA256
        or identities.get("source_mass_table_sha256")
        != SOURCE_MASS_TABLE_SHA256
        or value.get("recommendation", {}).get(
            "merits_one_later_preregistered_hpo_arm"
        ) is not True
        or value.get("access_firewall", {}).get(
            "non_train_semantics_opened_received_or_inferred"
        ) is not False
    ):
        raise RuntimeError("V49B committed V49A design contract changed")
    return value


def _identity(rows: list[dict], key: str) -> str:
    return engine.canonical_sha256([
        {"fact_id": row["fact_id"], key: row[key]}
        for row in rows
    ])


def compute_source_balanced_weights_v49b(
    rows: list[dict],
) -> tuple[list[float], dict]:
    """Return mean-one Trainer weights and a complete deterministic audit."""
    design = validate_v49a_design_v49b()
    payload = v49a.frozen.jsonl_bytes(rows)
    if (
        len(rows) != v49a.V434_TRAIN_ROWS
        or v49a.hashlib.sha256(payload).hexdigest() != v49a.V434_TRAIN_SHA256
    ):
        raise RuntimeError("V49B exact v434 train projection changed")
    units = v49a.frozen.build_conflict_units(rows)
    ordinary, equal_audit = BASE_EQUAL_UNIT_ASSIGNER_V49B(rows)
    row_count = len(rows)
    current = [weight / row_count for weight in ordinary]
    categories = [None] * row_count
    unit_by_row = [None] * row_count
    unit_identity_by_row = [None] * row_count
    for unit_index, unit in enumerate(units):
        for row_index in unit["indices"]:
            categories[row_index] = unit["stratum"]
            unit_by_row[row_index] = unit_index
            unit_identity_by_row[row_index] = unit["identity_sha256"]
    if any(item is None for item in categories + unit_by_row + unit_identity_by_row):
        raise RuntimeError("V49B conflict-unit coverage incomplete")

    current_source = defaultdict(float)
    current_category = defaultdict(float)
    source_rows = defaultdict(int)
    source_units = defaultdict(set)
    category_rows = defaultdict(int)
    category_units = defaultdict(set)
    for index, (row, weight, category) in enumerate(
        zip(rows, current, categories, strict=True)
    ):
        source = row["source"]
        current_source[source] += weight
        current_category[category] += weight
        source_rows[source] += 1
        source_units[source].add(unit_by_row[index])
        category_rows[category] += 1
        category_units[category].add(unit_by_row[index])
    positive = sorted(current_source.values())
    median_mass = positive[len(positive) // 2]
    raw_by_source = {
        source: (mass / median_mass) ** (-v49a.SOURCE_TEMPERATURE)
        for source, mass in current_source.items()
    }
    raw = [raw_by_source[row["source"]] for row in rows]
    multipliers = [None] * row_count
    for category, target in current_category.items():
        indices = [
            index for index, observed in enumerate(categories)
            if observed == category
        ]
        solved = v49a._solve_category_multiplier(
            indices, current, raw, target,
        )
        for index, multiplier in zip(indices, solved, strict=True):
            multipliers[index] = multiplier
    if any(item is None for item in multipliers):
        raise RuntimeError("V49B category normalization omitted a row")
    alternative = [
        weight * multiplier
        for weight, multiplier in zip(current, multipliers, strict=True)
    ]
    example_weights = [weight * row_count for weight in alternative]
    current_identity = v49a._weight_identity(rows, current)
    alternative_identity = v49a._weight_identity(rows, alternative)
    if (
        current_identity != CURRENT_NORMALIZED_WEIGHT_SHA256
        or alternative_identity != ALTERNATIVE_NORMALIZED_WEIGHT_SHA256
        or not math.isclose(math.fsum(alternative), 1.0, abs_tol=1e-15)
        or not math.isclose(
            math.fsum(example_weights) / row_count, 1.0, abs_tol=1e-15,
        )
    ):
        raise RuntimeError("V49B V49A alternative weight identity changed")

    alternative_source = defaultdict(float)
    alternative_category = defaultdict(float)
    for row, after, category in zip(
        rows, alternative, categories, strict=True,
    ):
        alternative_source[row["source"]] += after
        alternative_category[category] += after
    source_table = [{
        "source": source,
        "rows": source_rows[source],
        "conflict_units_touched": len(source_units[source]),
        "current_mass": current_source[source],
        "alternative_mass": alternative_source[source],
        "mass_delta": alternative_source[source] - current_source[source],
        "effective_mass_multiplier": (
            alternative_source[source] / current_source[source]
        ),
    } for source in sorted(current_source)]
    source_identity = engine.canonical_sha256(source_table)
    if source_identity != SOURCE_MASS_TABLE_SHA256:
        raise RuntimeError("V49B V49A per-source mass identity changed")
    category_table = [{
        "category": category,
        "rows": category_rows[category],
        "conflict_units": len(category_units[category]),
        "current_mass": current_category[category],
        "alternative_mass": alternative_category[category],
        "mass_delta": alternative_category[category] - current_category[category],
    } for category in sorted(current_category)]
    if max(abs(item["mass_delta"]) for item in category_table) > 1e-15:
        raise RuntimeError("V49B category objective mass changed")

    per_row = [{
        "row_index": index,
        "fact_id": row["fact_id"],
        "document_sha256": row["document_sha256"],
        "source": row["source"],
        "category": categories[index],
        "unit_identity_sha256": unit_identity_by_row[index],
        "current_normalized_weight_hex": current[index].hex(),
        "alternative_normalized_weight_hex": alternative[index].hex(),
        "trainer_example_weight_hex": example_weights[index].hex(),
        "applied_multiplier_hex": multipliers[index].hex(),
    } for index, row in enumerate(rows)]
    trainer_identity = engine.canonical_sha256([{
        "fact_id": row["fact_id"],
        "example_weight_hex": weight.hex(),
    } for row, weight in zip(rows, example_weights, strict=True)])
    compact = {
        "schema": "specialist-source-balanced-example-weighting-v49b",
        "rows": row_count,
        "conflict_units": len(units),
        "ordinary_row_mean_weight": math.fsum(example_weights) / row_count,
        "minimum_row_weight": min(example_weights),
        "maximum_row_weight": max(example_weights),
        "identity_sha256": alternative_identity,
        "current_normalized_weight_sha256": current_identity,
        "alternative_normalized_weight_sha256": alternative_identity,
        "trainer_example_weight_identity_sha256": trainer_identity,
        "per_row_identity_sha256": engine.canonical_sha256(per_row),
        "per_source_identity_sha256": source_identity,
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
        "root_membership_sha256": ROOT_MEMBERSHIP_SHA256,
        "v434_train_jsonl_sha256": v49a.V434_TRAIN_SHA256,
        "v49a_design_file_sha256": V49A_DESIGN_FILE_SHA256,
        "v49a_design_content_sha256": V49A_DESIGN_CONTENT_SHA256,
        "source_temperature": v49a.SOURCE_TEMPERATURE,
        "minimum_applied_multiplier": min(multipliers),
        "maximum_applied_multiplier": max(multipliers),
        "category_masses_preserved_exactly": True,
        "only_per_row_example_weights_changed": True,
    }
    complete = {
        **compact,
        "equal_unit_parent_audit": equal_audit,
        "per_row": per_row,
        "per_source": source_table,
        "per_category": category_table,
        "v49a_recommendation": design["recommendation"],
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
    return example_weights, complete


def compact_weighting_audit_v49b(complete: dict) -> dict:
    excluded = {
        "per_row", "per_source", "per_category", "equal_unit_parent_audit",
        "v49a_recommendation", "access_firewall",
        "content_sha256_before_self_field",
    }
    return {
        key: item for key, item in complete.items()
        if key not in excluded
    }


def assign_source_balanced_weights_v49b(
    rows: list[dict],
) -> tuple[list[float], dict]:
    weights, complete = compute_source_balanced_weights_v49b(rows)
    return weights, compact_weighting_audit_v49b(complete)
