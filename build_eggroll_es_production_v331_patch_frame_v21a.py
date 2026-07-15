#!/usr/bin/env python3
"""Build the aggregate-only V21A production-plus-v331 patch frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import build_eggroll_es_disjoint_tier_frame_v19a as frame_v19a
import build_eggroll_es_overlay_frame_v18a as frame_v18a
import seal_eggroll_es_candidate_v331 as candidate_v331


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH_V21A = (
    ROOT / "experiments/eggroll_es_hpo/train_panel_sampling_v21a/"
    "production_v331_patch_panels_v21a.json"
).resolve()

CANDIDATE_SEAL_COMMIT_V21A = "0913c3e5b3aacc7234a11cd8a53471511736602c"
V19A_FRAME_COMMIT_V21A = "97cddf65df07ce1c87e3838272f2558bcb53d911"
V18A_FRAME_COMMIT_V21A = "7055a62d67d030cfddf594c9ecf1d9e290f9d0d5"
SAMPLER_COMMIT_V21A = "9ee0af18f3e6ac59bdcf8a2f1e0850ab357c05f9"
PRODUCTION_COMMIT_V21A = "a21de35748054c3ae8737a767606234952f9561e"

CANDIDATE_MANIFEST_SHA256_V21A = (
    "179c35e154c47ce057d6edec3f1e9efa25e113803cee9d176da11bb8e112c847"
)
CANDIDATE_MANIFEST_CONTENT_SHA256_V21A = (
    "64900f822d619aa3c4c9fb9652a14dd9b4abf964a8946a2655214790a26c31b5"
)
V19A_FRAME_BUILDER_SHA256_V21A = (
    "4de43c263f580143d1b1f09a67240b8a05df7d45b419b19a211b549bf85b2052"
)
V19A_FRAME_TEST_SHA256_V21A = (
    "2034c1d1e214589a057197ffe1cfb0d19d803cb14b6d8b1418bee4cb034466d0"
)
V19A_FRAME_CERTIFICATE_SHA256_V21A = (
    "50820c67844a7b11c92bc0bbaa9c594c683440a65cb882de244216a77dca5fed"
)
V19A_FRAME_CERTIFICATE_CONTENT_SHA256_V21A = (
    "7ad195a55b1e51268dfba1cddb43f869b014bdb1ef329f5d73c8246ac6cbff58"
)
V18A_FRAME_BUILDER_SHA256_V21A = (
    "73d7c71233946996d64d776fb68c31609ad512379a15a75fae46dd0d8c395ba0"
)
SAMPLER_SHA256_V21A = (
    "81ca63c230995d44dfdb739a78b4a1a0f85d09724ba5915860ae36f78ccc3da9"
)
PRODUCTION_SHA256_V21A = (
    "62e7ae28c86a458d4d33bf3f73f1b91b873c86e3f70ce87706a7394d1f391507"
)

PANEL_NAMES_V21A = frame_v19a.PANEL_NAMES_V19A
OPTIMIZATION_PANELS_V21A = PANEL_NAMES_V21A[:6]
TRAIN_SCREEN_PANELS_V21A = PANEL_NAMES_V21A[6:]
ARM_ORDER_V21A = ("production_only", "production_plus_v331_patch")
BASE_CATEGORIES_V21A = frame_v19a.BASE_CATEGORIES_V19A
BASE_CATEGORY_POPULATIONS_V21A = frame_v19a.BASE_CATEGORY_POPULATIONS_V19A
BASE_CATEGORY_QUOTA_V21A = frame_v19a.BASE_CATEGORY_QUOTA_V19A
PRODUCTION_TOPIC_QUOTAS_V21A = frame_v19a.PRODUCTION_TOPIC_QUOTAS_V19A
CANDIDATE_ONLY_TOPIC_POPULATIONS_V21A = {
    "safety_consent": 18,
    "technique": 18,
    "equipment_material": 10,
    "resources_general": 8,
}
CANDIDATE_ONLY_TOPIC_QUOTAS_V21A = {
    "safety_consent": 2,
    "technique": 2,
    "equipment_material": 1,
    "resources_general": 1,
}
ARM_REQUESTS_PER_PANEL_V21A = {
    "production_only": 24,
    "production_plus_v331_patch": 30,
}
ARM_POPULATIONS_V21A = {
    "production_only": 272,
    "production_plus_v331_patch": 326,
}
CANDIDATE_ASSIGNMENT_SEED_V21A = 20260820

canonical_sha256 = frame_v18a.canonical_sha256
file_sha256 = frame_v18a.file_sha256
identity_root_sha256 = frame_v18a.identity_root_sha256


def _without_self(value: dict) -> dict:
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _verify_commit_file(path: Path, commit: str, digest: str) -> None:
    path = Path(path).resolve()
    relative = path.relative_to(ROOT).as_posix()
    raw = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
    if hashlib.sha256(raw).hexdigest() != digest or file_sha256(path) != digest:
        raise RuntimeError(f"v21a committed train artifact changed: {relative}")


def verify_bound_inputs_v21a() -> dict:
    bindings = (
        (
            candidate_v331.OUTPUT_PATH_V331,
            CANDIDATE_SEAL_COMMIT_V21A,
            candidate_v331.V331_SHA256,
        ),
        (
            candidate_v331.MANIFEST_PATH_V331,
            CANDIDATE_SEAL_COMMIT_V21A,
            CANDIDATE_MANIFEST_SHA256_V21A,
        ),
        (
            Path(frame_v19a.__file__),
            V19A_FRAME_COMMIT_V21A,
            V19A_FRAME_BUILDER_SHA256_V21A,
        ),
        (
            ROOT / "test_build_eggroll_es_disjoint_tier_frame_v19a.py",
            V19A_FRAME_COMMIT_V21A,
            V19A_FRAME_TEST_SHA256_V21A,
        ),
        (
            frame_v19a.OUTPUT_PATH_V19A,
            V19A_FRAME_COMMIT_V21A,
            V19A_FRAME_CERTIFICATE_SHA256_V21A,
        ),
        (
            Path(frame_v18a.__file__),
            V18A_FRAME_COMMIT_V21A,
            V18A_FRAME_BUILDER_SHA256_V21A,
        ),
        (
            Path(frame_v18a.sampler_v13.__file__),
            SAMPLER_COMMIT_V21A,
            SAMPLER_SHA256_V21A,
        ),
        (
            frame_v18a.PRODUCTION_PATH_V18A,
            PRODUCTION_COMMIT_V21A,
            PRODUCTION_SHA256_V21A,
        ),
    )
    for path, commit, digest in bindings:
        _verify_commit_file(path, commit, digest)
    candidate_v331.validate_candidate_snapshot_v331()
    manifest = json.loads(candidate_v331.MANIFEST_PATH_V331.read_text())
    candidate_v331.validate_manifest_v331(manifest)
    frozen_v19a = json.loads(frame_v19a.OUTPUT_PATH_V19A.read_text())
    frame_v19a.validate_certificate_v19a(frozen_v19a)
    if (
        manifest["content_sha256_before_self_field"]
        != CANDIDATE_MANIFEST_CONTENT_SHA256_V21A
        or frozen_v19a["content_sha256_before_self_field"]
        != V19A_FRAME_CERTIFICATE_CONTENT_SHA256_V21A
    ):
        raise RuntimeError("v21a bound train certificate content changed")
    return {"candidate_manifest": manifest, "v19a_frame": frozen_v19a}


def _read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_bound_rows_v21a() -> tuple[list[dict], list[dict]]:
    verify_bound_inputs_v21a()
    candidate = _read_jsonl(candidate_v331.OUTPUT_PATH_V331)
    production = _read_jsonl(frame_v18a.PRODUCTION_PATH_V18A)
    if len(candidate) != 527 or len(production) != 784:
        raise RuntimeError("v21a frozen train row count changed")
    return candidate, production


def build_joint_frame_v21a(candidate: list[dict], production: list[dict]) -> dict:
    """Build the train-only v331/production joint conflict-unit frame."""
    side_rows = {"candidate": candidate, "production": production}
    side_units = {}
    row_to_unit = {}
    for side, rows in side_rows.items():
        side_units[side], row_to_unit[side] = frame_v18a.build_side_units_v18a(
            rows, side
        )
    capacity = {
        side: dict(Counter(unit["stratum"] for unit in units))
        for side, units in side_units.items()
    }
    if capacity != {
        "candidate": {
            "safety_consent": 80,
            "technique": 73,
            "equipment_material": 23,
            "resources_general": 84,
        },
        "production": frame_v18a.PRODUCTION_COMPONENT_CAPACITY_V18A,
    }:
        raise RuntimeError("v21a standalone train component capacity changed")

    tagged = [
        {"side": "candidate", "side_index": index, "row": row}
        for index, row in enumerate(candidate)
    ] + [
        {"side": "production", "side_index": index, "row": row}
        for index, row in enumerate(production)
    ]
    combined_rows = [item["row"] for item in tagged]
    semantic_ids = frame_v18a.sampler_v13.build_semantic_clusters(combined_rows)
    disjoint = frame_v18a.DisjointSetV18A(len(combined_rows))
    frame_v18a._connect_rows_v18a(combined_rows, semantic_ids, disjoint)
    first_side_unit = {}
    for combined_index, item in enumerate(tagged):
        key = (
            item["side"],
            row_to_unit[item["side"]][item["side_index"]],
        )
        if key in first_side_unit:
            disjoint.union(combined_index, first_side_unit[key])
        else:
            first_side_unit[key] = combined_index
    components = defaultdict(list)
    for index in range(len(tagged)):
        components[disjoint.find(index)].append(index)

    joint_components = []
    for members in components.values():
        unit_indices = {"candidate": set(), "production": set()}
        for combined_index in members:
            item = tagged[combined_index]
            unit_indices[item["side"]].add(
                row_to_unit[item["side"]][item["side_index"]]
            )
        if any(len(unit_indices[side]) > 1 for side in unit_indices):
            raise RuntimeError("v21a joint frame merged same-side units")
        candidate_unit = next(iter(unit_indices["candidate"]), None)
        production_unit = next(iter(unit_indices["production"]), None)
        identities = sorted(
            f"{side}:{side_units[side][unit_index]['unit_id']}"
            for side, unit_index in (
                ("candidate", candidate_unit),
                ("production", production_unit),
            )
            if unit_index is not None
        )
        relation = "candidate_only" if production_unit is None else (
            "production_only" if candidate_unit is None else "paired"
        )
        match_class = relation
        if relation == "paired":
            candidate_contract = side_units["candidate"][candidate_unit]
            production_contract = side_units["production"][production_unit]
            if (
                candidate_contract["document_sha256s"]
                & production_contract["document_sha256s"]
            ):
                match_class = "shared_document"
            elif candidate_contract["urls"] & production_contract["urls"]:
                match_class = "shared_url_without_shared_document"
            elif (
                candidate_contract["lineages"]
                & production_contract["lineages"]
            ):
                match_class = "shared_lineage_without_shared_document"
            else:
                match_class = "semantic_only_without_shared_document"
        joint_components.append({
            "joint_id": canonical_sha256({
                "schema": "eggroll-es-joint-component-v21a",
                "members": identities,
            }),
            "candidate_unit": candidate_unit,
            "production_unit": production_unit,
            "relation": relation,
            "match_class": match_class,
        })
    joint_components.sort(key=lambda item: item["joint_id"])
    relation_counts = Counter(item["relation"] for item in joint_components)
    match_counts = Counter(item["match_class"] for item in joint_components)
    candidate_only_topics = Counter(
        side_units["candidate"][item["candidate_unit"]]["stratum"]
        for item in joint_components if item["relation"] == "candidate_only"
    )
    if (
        len(joint_components) != 326
        or relation_counts
        != {"paired": 206, "candidate_only": 54, "production_only": 66}
        or match_counts != {
            "shared_document": 204,
            "shared_url_without_shared_document": 2,
            "candidate_only": 54,
            "production_only": 66,
        }
        or candidate_only_topics != CANDIDATE_ONLY_TOPIC_POPULATIONS_V21A
    ):
        raise RuntimeError("v21a joint train relation frame changed")
    return {
        "side_units": side_units,
        "joint_components": joint_components,
        "capacity": capacity,
        "relation_counts": dict(relation_counts),
        "match_counts": dict(match_counts),
        "candidate_only_topic_counts": dict(candidate_only_topics),
    }


def _assignment_cost_v21a(component: dict, topic: str, reuse: bool) -> str:
    return canonical_sha256({
        "schema": (
            "eggroll-es-v331-candidate-reuse-order-v21a"
            if reuse else "eggroll-es-v331-candidate-exhaustion-order-v21a"
        ),
        "seed": CANDIDATE_ASSIGNMENT_SEED_V21A,
        "topic": topic,
        "joint_id": component["joint_id"],
    })


def assign_candidate_only_v21a(frame: dict) -> dict:
    components = frame["joint_components"]
    candidate_units = frame["side_units"]["candidate"]
    by_topic = {
        topic: [
            index for index, component in enumerate(components)
            if component["relation"] == "candidate_only"
            and candidate_units[component["candidate_unit"]]["stratum"] == topic
        ]
        for topic in CANDIDATE_ONLY_TOPIC_QUOTAS_V21A
    }
    assignments = {panel: [] for panel in PANEL_NAMES_V21A}
    topic_summary = {}
    reuse_membership = set()
    for topic, quota in CANDIDATE_ONLY_TOPIC_QUOTAS_V21A.items():
        indices = by_topic[topic]
        unique_order = sorted(
            indices,
            key=lambda index: _assignment_cost_v21a(
                components[index], topic, False
            ),
        )
        reuse_order = sorted(
            indices,
            key=lambda index: _assignment_cost_v21a(
                components[index], topic, True
            ),
        )
        slots = len(PANEL_NAMES_V21A) * quota
        reuse_count = slots - len(unique_order)
        sequence = unique_order + reuse_order[:reuse_count]
        if (
            len(indices) != CANDIDATE_ONLY_TOPIC_POPULATIONS_V21A[topic]
            or reuse_count < 0
            or len(sequence) != slots
            or set(sequence[:len(unique_order)]) != set(indices)
        ):
            raise RuntimeError("v21a candidate breadth exhaustion changed")
        cursor = 0
        for panel in PANEL_NAMES_V21A:
            chosen = sequence[cursor:cursor + quota]
            if len(chosen) != len(set(chosen)):
                raise RuntimeError("v21a panel repeats one candidate component")
            assignments[panel].extend(chosen)
            for position, index in enumerate(chosen, cursor):
                if position >= len(unique_order):
                    reuse_membership.add((panel, index))
            cursor += quota
        topic_summary[topic] = {
            "population": len(indices),
            "quota_per_panel": quota,
            "total_assignments": slots,
            "unique_components_used": len(set(sequence)),
            "reuse_assignments_after_exhaustion": reuse_count,
            "all_unique_assigned_before_first_reuse": True,
            "ordered_assignment_identity_sha256": canonical_sha256([
                components[index]["joint_id"] for index in sequence
            ]),
        }
    if any(len(values) != 6 or len(values) != len(set(values)) for values in assignments.values()):
        raise RuntimeError("v21a per-panel candidate-only quota changed")
    unique = {index for values in assignments.values() for index in values}
    if len(unique) != 54 or len(reuse_membership) != 6:
        raise RuntimeError("v21a candidate-only breadth coverage changed")
    role_summary = {}
    for role, panels in (
        ("optimization", OPTIMIZATION_PANELS_V21A),
        ("train_only_screen", TRAIN_SCREEN_PANELS_V21A),
    ):
        role_values = [(panel, index) for panel in panels for index in assignments[panel]]
        role_summary[role] = {
            "panel_count": len(panels),
            "assignment_count": len(role_values),
            "within_role_unique_component_count": len({
                index for _, index in role_values
            }),
            "reuse_assignment_count": sum(item in reuse_membership for item in role_values),
            "new_global_unique_component_count": (
                len(role_values)
                - sum(item in reuse_membership for item in role_values)
            ),
            "topic_assignment_counts": dict(Counter(
                candidate_units[components[index]["candidate_unit"]]["stratum"]
                for _, index in role_values
            )),
        }
    if role_summary != {
        "optimization": {
            "panel_count": 6,
            "assignment_count": 36,
            "within_role_unique_component_count": 36,
            "reuse_assignment_count": 0,
            "new_global_unique_component_count": 36,
            "topic_assignment_counts": {
                "safety_consent": 12,
                "technique": 12,
                "equipment_material": 6,
                "resources_general": 6,
            },
        },
        "train_only_screen": {
            "panel_count": 4,
            "assignment_count": 24,
            "within_role_unique_component_count": 23,
            "reuse_assignment_count": 6,
            "new_global_unique_component_count": 18,
            "topic_assignment_counts": {
                "safety_consent": 8,
                "technique": 8,
                "equipment_material": 4,
                "resources_general": 4,
            },
        },
    }:
        raise RuntimeError("v21a candidate topic-by-role allocation changed")
    return {
        "assignments": assignments,
        "topic_summary": topic_summary,
        "role_summary": role_summary,
        "reuse_membership": reuse_membership,
    }


def _base_flow_v21a() -> tuple[dict, dict, dict]:
    frozen_frame, tiers = frame_v19a.load_frozen_frame_v19a()
    flow = frame_v19a.solve_base_flow_v19a(frozen_frame, tiers)
    return frozen_frame, tiers, flow


def materialize_panel_contracts_v21a(
    frame: dict,
    frozen_frame: dict,
    frozen_tiers: dict,
    base_flow: dict,
    candidate_assignment: dict,
) -> list[dict]:
    components = frame["joint_components"]
    candidate_units = frame["side_units"]["candidate"]
    production_units = frame["side_units"]["production"]
    new_by_production_unit_id = {
        production_units[item["production_unit"]]["unit_id"]: index
        for index, item in enumerate(components)
        if item["production_unit"] is not None
    }
    frozen_components = frozen_frame["joint_components"]
    frozen_production_units = frozen_frame["side_units"]["production"]
    selected_base_global = []
    summaries = []
    for panel in PANEL_NAMES_V21A:
        frozen_indices = base_flow["selected_base"][panel]
        base = []
        for frozen_index in frozen_indices:
            production_index = frozen_components[frozen_index]["production_unit"]
            unit_id = frozen_production_units[production_index]["unit_id"]
            base.append(new_by_production_unit_id[unit_id])
        candidate_only = candidate_assignment["assignments"][panel]
        base_categories = Counter(
            frozen_tiers["base_category"][index] for index in frozen_indices
        )
        topics = Counter(
            frozen_production_units[
                frozen_components[index]["production_unit"]
            ]["stratum"]
            for index in frozen_indices
        )
        candidate_topics = Counter(
            candidate_units[components[index]["candidate_unit"]]["stratum"]
            for index in candidate_only
        )
        if (
            base_categories != {
                category: BASE_CATEGORY_QUOTA_V21A
                for category in BASE_CATEGORIES_V21A
            }
            or topics != PRODUCTION_TOPIC_QUOTAS_V21A
            or candidate_topics != CANDIDATE_ONLY_TOPIC_QUOTAS_V21A
            or any(components[index]["relation"] == "candidate_only" for index in base)
        ):
            raise RuntimeError("v21a solved panel violates an exact quota")
        selected_base_global.extend(
            production_units[components[index]["production_unit"]]["unit_id"]
            for index in base
        )
        arms = {}
        for arm in ARM_ORDER_V21A:
            active = list(base)
            if arm == "production_plus_v331_patch":
                active.extend(candidate_only)
            representatives = []
            representative_sides = Counter()
            for index in active:
                component = components[index]
                if component["relation"] == "candidate_only":
                    side = "candidate"
                    unit = candidate_units[component["candidate_unit"]]
                else:
                    side = "production"
                    unit = production_units[component["production_unit"]]
                representative_sides[side] += 1
                representatives.append(
                    f"{component['joint_id']}:{side}:{unit['representative_row_sha256']}"
                )
            if (
                len(active) != ARM_REQUESTS_PER_PANEL_V21A[arm]
                or len(active) != len(set(active))
                or representative_sides["production"] != 24
                or representative_sides["candidate"] != (0 if arm == "production_only" else 6)
            ):
                raise RuntimeError("v21a full-merge arm contract changed")
            arms[arm] = {
                "requests": len(active),
                "population_denominator": ARM_POPULATIONS_V21A[arm],
                "production_representatives": representative_sides["production"],
                "candidate_only_representatives": representative_sides["candidate"],
                "paired_candidate_replacements": 0,
                "active_joint_component_identity_root_sha256": (
                    identity_root_sha256(components[index]["joint_id"] for index in active)
                ),
                "representative_assignment_root_sha256": (
                    identity_root_sha256(representatives)
                ),
                "same_arm_component_duplicate_count": 0,
            }
        summaries.append({
            "name": panel,
            "role": (
                "optimization" if panel in OPTIMIZATION_PANELS_V21A
                else "train_only_screen"
            ),
            "base_components": 24,
            "base_category_counts": dict(base_categories),
            "production_topic_counts": dict(topics),
            "candidate_only_components": 6,
            "candidate_only_topic_counts": dict(candidate_topics),
            "selected_base_joint_component_identity_root_sha256": (
                identity_root_sha256(components[index]["joint_id"] for index in base)
            ),
            "selected_candidate_only_joint_component_identity_root_sha256": (
                identity_root_sha256(
                    components[index]["joint_id"] for index in candidate_only
                )
            ),
            "arms": arms,
        })
    if len(selected_base_global) != 240 or len(set(selected_base_global)) != 240:
        raise RuntimeError("v21a production base panels are not globally disjoint")
    return summaries


def build_certificate_v21a() -> dict:
    bound = verify_bound_inputs_v21a()
    candidate, production = load_bound_rows_v21a()
    frame = build_joint_frame_v21a(candidate, production)
    frozen_frame, frozen_tiers, base_flow = _base_flow_v21a()
    candidate_assignment = assign_candidate_only_v21a(frame)
    panels = materialize_panel_contracts_v21a(
        frame, frozen_frame, frozen_tiers, base_flow, candidate_assignment
    )
    components = frame["joint_components"]
    assignment_sequence = [
        {
            "panel": panel,
            "joint_ids": [
                components[index]["joint_id"]
                for index in candidate_assignment["assignments"][panel]
            ],
        }
        for panel in PANEL_NAMES_V21A
    ]
    certificate = {
        "schema": "eggroll-es-production-v331-patch-flow-certificate-v21a",
        "status": "feasible_offline_frame_only_no_runtime_or_update_authority",
        "arm_order": list(ARM_ORDER_V21A),
        "inputs": {
            "candidate_v331": {
                "source_commit": candidate_v331.V331_SOURCE_COMMIT,
                "sealed_snapshot_commit": CANDIDATE_SEAL_COMMIT_V21A,
                "path": str(candidate_v331.OUTPUT_PATH_V331),
                "rows": candidate_v331.V331_ROWS,
                "file_sha256": candidate_v331.V331_SHA256,
                "manifest_path": str(candidate_v331.MANIFEST_PATH_V331),
                "manifest_file_sha256": CANDIDATE_MANIFEST_SHA256_V21A,
                "manifest_content_sha256": (
                    CANDIDATE_MANIFEST_CONTENT_SHA256_V21A
                ),
                "ongoing_curation_used": False,
            },
            "production": {
                "commit": PRODUCTION_COMMIT_V21A,
                "path": str(frame_v18a.PRODUCTION_PATH_V18A),
                "rows": 784,
                "file_sha256": PRODUCTION_SHA256_V21A,
            },
            "pinned_v13_grouping": {
                "commit": SAMPLER_COMMIT_V21A,
                "path": str(Path(frame_v18a.sampler_v13.__file__).resolve()),
                "file_sha256": SAMPLER_SHA256_V21A,
            },
            "frozen_v19a_base_frame": {
                "commit": V19A_FRAME_COMMIT_V21A,
                "builder_file_sha256": V19A_FRAME_BUILDER_SHA256_V21A,
                "certificate_file_sha256": V19A_FRAME_CERTIFICATE_SHA256_V21A,
                "certificate_content_sha256": (
                    V19A_FRAME_CERTIFICATE_CONTENT_SHA256_V21A
                ),
            },
        },
        "joint_frame": {
            "grouping": (
                "combined_document_normalized_url_raw_lineage_and_pinned_v13_"
                "lexical_semantic_connected_components_with_side_units_intact"
            ),
            "joint_component_count": 326,
            "relation_counts": frame["relation_counts"],
            "match_class_counts": frame["match_counts"],
            "production_component_count": 272,
            "candidate_component_count": 260,
            "candidate_only_component_count": 54,
            "candidate_only_topic_populations": (
                CANDIDATE_ONLY_TOPIC_POPULATIONS_V21A
            ),
            "joint_component_identity_root_sha256": identity_root_sha256(
                item["joint_id"] for item in components
            ),
        },
        "constrained_production_base": {
            "source": "exact_frozen_v19a_ten_panel_base_flow",
            "solver": base_flow["solver"],
            "panel_count": 10,
            "optimization_panel_count": 6,
            "train_only_screen_panel_count": 4,
            "base_category_quota_per_panel": {
                category: BASE_CATEGORY_QUOTA_V21A
                for category in BASE_CATEGORIES_V21A
            },
            "production_topic_quota_per_panel": PRODUCTION_TOPIC_QUOTAS_V21A,
            "selected_base_components": 240,
            "globally_panel_disjoint_base_components": True,
            "quota_seed_or_solver_relaxation_used": False,
        },
        "candidate_only_assignment": {
            "seed": CANDIDATE_ASSIGNMENT_SEED_V21A,
            "quota_per_panel_by_topic": CANDIDATE_ONLY_TOPIC_QUOTAS_V21A,
            "requests_per_panel": 6,
            "topic_summary": candidate_assignment["topic_summary"],
            "role_summary": candidate_assignment["role_summary"],
            "every_unique_item_before_deterministic_within_topic_reuse": True,
            "total_panel_assignments": 60,
            "unique_candidate_only_components": 54,
            "reuse_assignments_after_exhaustion": 6,
            "ordered_assignment_sha256": canonical_sha256(assignment_sequence),
            "minimal_uniform_per_panel_quota_for_complete_topic_breadth": True,
            "inherited_q3_rejected_as_insufficient": True,
        },
        "estimand": {
            "semantics": (
                "production_only_versus_production_plus_all_candidate_only_v331_"
                "conflict_units_with_no_paired_candidate_replacement"
            ),
            "arm_population_denominators": ARM_POPULATIONS_V21A,
            "arm_requests_per_panel": ARM_REQUESTS_PER_PANEL_V21A,
            "base_population_ht_strata": {
                category: {
                    "population": population,
                    "per_panel_quota": BASE_CATEGORY_QUOTA_V21A,
                    "horvitz_thompson_weight": population / BASE_CATEGORY_QUOTA_V21A,
                }
                for category, population in BASE_CATEGORY_POPULATIONS_V21A.items()
            },
            "candidate_only_ht_strata": {
                topic: {
                    "population": CANDIDATE_ONLY_TOPIC_POPULATIONS_V21A[topic],
                    "per_panel_quota": quota,
                    "horvitz_thompson_weight": (
                        CANDIDATE_ONLY_TOPIC_POPULATIONS_V21A[topic] / quota
                    ),
                }
                for topic, quota in CANDIDATE_ONLY_TOPIC_QUOTAS_V21A.items()
            },
            "full_merge_keeps_every_sampled_production_representative": True,
            "paired_candidate_replacement_count": 0,
            "plain_request_mean_used": False,
        },
        "panels": panels,
        "bound_input_content_sha256": {
            "candidate_manifest": bound["candidate_manifest"][
                "content_sha256_before_self_field"
            ],
            "v19a_frame": bound["v19a_frame"][
                "content_sha256_before_self_field"
            ],
        },
        "firewall": {
            "contains_question_answer_prompt_response_token_or_row_content": False,
            "contains_evaluation_content": False,
            "heldout_validation_ood_eval_or_benchmark_content_opened": False,
            "depends_on_v20_runtime_result": False,
            "runtime_launch_authorized": False,
            "gpu_launch_authorized": False,
            "model_update_authorized": False,
            "checkpoint_write_authorized": False,
            "evaluation_authorized": False,
            "dataset_promotion_authorized": False,
        },
    }
    certificate["content_sha256_before_self_field"] = canonical_sha256(certificate)
    return validate_certificate_v21a(certificate)


def validate_certificate_v21a(value: dict) -> dict:
    flow = value.get("constrained_production_base", {})
    candidate = value.get("candidate_only_assignment", {})
    panels = value.get("panels", [])
    firewall = value.get("firewall", {})
    if (
        value.get("schema")
        != "eggroll-es-production-v331-patch-flow-certificate-v21a"
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(value))
        or value.get("arm_order") != list(ARM_ORDER_V21A)
        or value.get("inputs", {}).get("candidate_v331", {}).get("source_commit")
        != candidate_v331.V331_SOURCE_COMMIT
        or value.get("inputs", {}).get("candidate_v331", {}).get("file_sha256")
        != candidate_v331.V331_SHA256
        or value.get("inputs", {}).get("candidate_v331", {}).get(
            "ongoing_curation_used"
        ) is not False
        or value.get("joint_frame", {}).get("candidate_only_component_count") != 54
        or value.get("joint_frame", {}).get("candidate_only_topic_populations")
        != CANDIDATE_ONLY_TOPIC_POPULATIONS_V21A
        or flow.get("globally_panel_disjoint_base_components") is not True
        or flow.get("selected_base_components") != 240
        or flow.get("quota_seed_or_solver_relaxation_used") is not False
        or not math.isclose(
            flow.get("solver", {}).get("mip_gap", math.inf), 0.0, abs_tol=1e-15
        )
        or candidate.get("quota_per_panel_by_topic")
        != CANDIDATE_ONLY_TOPIC_QUOTAS_V21A
        or candidate.get("unique_candidate_only_components") != 54
        or candidate.get("reuse_assignments_after_exhaustion") != 6
        or candidate.get("inherited_q3_rejected_as_insufficient") is not True
        or len(panels) != 10
        or tuple(panel.get("name") for panel in panels) != PANEL_NAMES_V21A
        or any(
            panel.get("base_category_counts")
            != {category: 6 for category in BASE_CATEGORIES_V21A}
            or panel.get("production_topic_counts")
            != PRODUCTION_TOPIC_QUOTAS_V21A
            or panel.get("candidate_only_topic_counts")
            != CANDIDATE_ONLY_TOPIC_QUOTAS_V21A
            or set(panel.get("arms", {})) != set(ARM_ORDER_V21A)
            or panel["arms"]["production_only"].get("requests") != 24
            or panel["arms"]["production_plus_v331_patch"].get("requests") != 30
            or any(
                arm.get("paired_candidate_replacements") != 0
                or arm.get("same_arm_component_duplicate_count") != 0
                for arm in panel.get("arms", {}).values()
            )
            for panel in panels
        )
        or firewall.get("depends_on_v20_runtime_result") is not False
        or firewall.get(
            "heldout_validation_ood_eval_or_benchmark_content_opened"
        ) is not False
        or any(firewall.get(key) is not False for key in (
            "runtime_launch_authorized", "gpu_launch_authorized",
            "model_update_authorized", "checkpoint_write_authorized",
            "evaluation_authorized", "dataset_promotion_authorized",
        ))
    ):
        raise RuntimeError("v21a production-v331 patch frame certificate changed")
    return value


def _exclusive_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("immutable v21a frame certificate already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH_V21A))
    args = parser.parse_args(argv)
    if Path(args.output).resolve() != OUTPUT_PATH_V21A:
        raise ValueError("v21a frame certificate output path changed")
    certificate = build_certificate_v21a()
    _exclusive_write(OUTPUT_PATH_V21A, certificate)
    result = {
        "schema": "eggroll-es-production-v331-patch-flow-build-v21a",
        "output": str(OUTPUT_PATH_V21A),
        "file_sha256": file_sha256(OUTPUT_PATH_V21A),
        "content_sha256": certificate["content_sha256_before_self_field"],
        "candidate_only_components": 54,
        "candidate_only_requests_per_panel": 6,
        "runtime_launch_authorized": False,
        "gpu_launched": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
