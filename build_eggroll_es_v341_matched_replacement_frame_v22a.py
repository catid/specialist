#!/usr/bin/env python3
"""Build the aggregate-only V22A exact-v341 matched-replacement frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import build_eggroll_es_production_v331_patch_frame_v21a as frame_v21a
import seal_eggroll_es_candidate_v341 as candidate_v341


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH_V22A = (
    ROOT / "experiments/eggroll_es_hpo/train_panel_sampling_v22a/"
    "production_v341_matched_replacement_panels_v22a.json"
).resolve()

CANDIDATE_SEAL_COMMIT_V22A = "01ca5809321d00a904f5bb8fff1b3d8dd402db25"
V21A_FRAME_COMMIT_V22A = "32bd189372a14b0c7db155d8986dcc6d4928b93b"
PRODUCTION_COMMIT_V22A = frame_v21a.PRODUCTION_COMMIT_V21A
SAMPLER_COMMIT_V22A = frame_v21a.SAMPLER_COMMIT_V21A

CANDIDATE_SEAL_BUILDER_SHA256_V22A = (
    "ec5bcd53f66897ac92a9a15d5dace5f6e0173bb213d577329eedf04c5c6424d0"
)
CANDIDATE_SEAL_TEST_SHA256_V22A = (
    "70d7e911edd50fec95fa3249ebcbadb01f329775bee07a5e87786bbc2ef217ae"
)
CANDIDATE_MANIFEST_SHA256_V22A = (
    "2e766332f2686ad312f92f37fd7d457bd88c9d9c309498fc965cb6301483a9ac"
)
CANDIDATE_MANIFEST_CONTENT_SHA256_V22A = (
    "a93278d011495d3d1f95894a505cd38e5408fed6510f60f12df6079fb1cab77a"
)
V21A_FRAME_BUILDER_SHA256_V22A = (
    "31a39322e325ff29007824d7b4165de67a34b5ac5fd88935e5991dfb2e3cb0b6"
)
V21A_FRAME_TEST_SHA256_V22A = (
    "6f212f5996fa21603a006b34a261318a1e24e0347204d1a1b3357a23afe8fc81"
)
V21A_FRAME_CERTIFICATE_SHA256_V22A = (
    "9dea3ff4eb970087a17daf2dbbeb1a0f49985330b515160fcb0c0db87216fee9"
)
V21A_FRAME_CERTIFICATE_CONTENT_SHA256_V22A = (
    "59e3352aabe851e31cf9e5ee47559051373f397d7354d3125ace821c65b118c6"
)
PRODUCTION_SHA256_V22A = frame_v21a.PRODUCTION_SHA256_V21A
SAMPLER_SHA256_V22A = frame_v21a.SAMPLER_SHA256_V21A

PANEL_NAMES_V22A = frame_v21a.PANEL_NAMES_V21A
OPTIMIZATION_PANELS_V22A = frame_v21a.OPTIMIZATION_PANELS_V21A
TRAIN_SCREEN_PANELS_V22A = frame_v21a.TRAIN_SCREEN_PANELS_V21A
BASE_CATEGORIES_V22A = frame_v21a.BASE_CATEGORIES_V21A
BASE_CATEGORY_POPULATIONS_V22A = frame_v21a.BASE_CATEGORY_POPULATIONS_V21A
BASE_CATEGORY_QUOTA_V22A = frame_v21a.BASE_CATEGORY_QUOTA_V21A
PRODUCTION_TOPIC_QUOTAS_V22A = frame_v21a.PRODUCTION_TOPIC_QUOTAS_V21A
ARM_ORDER_V22A = ("production_control", "v341_matched_replacement")
ARM_REQUESTS_PER_PANEL_V22A = {
    "production_control": 24,
    "v341_matched_replacement": 24,
}
ARM_POPULATION_DENOMINATORS_V22A = {
    "production_control": 272,
    "v341_matched_replacement": 272,
}
EXPECTED_REPLACEMENTS_BY_PANEL_V22A = {
    "optimization_0": 18,
    "optimization_1": 18,
    "optimization_2": 18,
    "optimization_3": 17,
    "optimization_4": 18,
    "optimization_5": 18,
    "train_screen_0": 18,
    "train_screen_1": 20,
    "train_screen_2": 20,
    "train_screen_3": 19,
}
EXPECTED_UNCHANGED_BY_PANEL_V22A = {
    panel: 24 - replacements
    for panel, replacements in EXPECTED_REPLACEMENTS_BY_PANEL_V22A.items()
}

canonical_sha256 = frame_v21a.canonical_sha256
file_sha256 = frame_v21a.file_sha256
identity_root_sha256 = frame_v21a.identity_root_sha256


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
        raise RuntimeError(f"v22a committed train artifact changed: {relative}")


def verify_bound_inputs_v22a() -> dict:
    for path, commit, digest in (
        (
            Path(candidate_v341.__file__),
            CANDIDATE_SEAL_COMMIT_V22A,
            CANDIDATE_SEAL_BUILDER_SHA256_V22A,
        ),
        (
            ROOT / "test_seal_eggroll_es_candidate_v341.py",
            CANDIDATE_SEAL_COMMIT_V22A,
            CANDIDATE_SEAL_TEST_SHA256_V22A,
        ),
        (
            candidate_v341.OUTPUT_PATH_V341,
            CANDIDATE_SEAL_COMMIT_V22A,
            candidate_v341.V341_SHA256,
        ),
        (
            candidate_v341.MANIFEST_PATH_V341,
            CANDIDATE_SEAL_COMMIT_V22A,
            CANDIDATE_MANIFEST_SHA256_V22A,
        ),
        (
            Path(frame_v21a.__file__),
            V21A_FRAME_COMMIT_V22A,
            V21A_FRAME_BUILDER_SHA256_V22A,
        ),
        (
            ROOT / "test_build_eggroll_es_production_v331_patch_frame_v21a.py",
            V21A_FRAME_COMMIT_V22A,
            V21A_FRAME_TEST_SHA256_V22A,
        ),
        (
            frame_v21a.OUTPUT_PATH_V21A,
            V21A_FRAME_COMMIT_V22A,
            V21A_FRAME_CERTIFICATE_SHA256_V22A,
        ),
        (
            frame_v21a.frame_v18a.PRODUCTION_PATH_V18A,
            PRODUCTION_COMMIT_V22A,
            PRODUCTION_SHA256_V22A,
        ),
        (
            Path(frame_v21a.frame_v18a.sampler_v13.__file__),
            SAMPLER_COMMIT_V22A,
            SAMPLER_SHA256_V22A,
        ),
    ):
        _verify_commit_file(path, commit, digest)
    candidate_v341.validate_candidate_snapshot_v341()
    manifest = json.loads(candidate_v341.MANIFEST_PATH_V341.read_text())
    candidate_v341.validate_manifest_v341(manifest)
    prior_frame = json.loads(frame_v21a.OUTPUT_PATH_V21A.read_text())
    frame_v21a.validate_certificate_v21a(prior_frame)
    if (
        manifest["content_sha256_before_self_field"]
        != CANDIDATE_MANIFEST_CONTENT_SHA256_V22A
        or prior_frame["content_sha256_before_self_field"]
        != V21A_FRAME_CERTIFICATE_CONTENT_SHA256_V22A
    ):
        raise RuntimeError("v22a bound train certificate content changed")
    return {"candidate_manifest": manifest, "prior_frame": prior_frame}


def _read_jsonl_internal_only(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _build_joint_frame_internal_only(
    candidate: list[dict], production: list[dict]
) -> dict:
    side_rows = {"candidate": candidate, "production": production}
    side_units = {}
    row_to_unit = {}
    for side, rows in side_rows.items():
        side_units[side], row_to_unit[side] = (
            frame_v21a.frame_v18a.build_side_units_v18a(rows, side)
        )
    tagged = [
        {"side": "candidate", "side_index": index, "row": row}
        for index, row in enumerate(candidate)
    ] + [
        {"side": "production", "side_index": index, "row": row}
        for index, row in enumerate(production)
    ]
    combined_rows = [item["row"] for item in tagged]
    semantic_ids = (
        frame_v21a.frame_v18a.sampler_v13.build_semantic_clusters(combined_rows)
    )
    disjoint = frame_v21a.frame_v18a.DisjointSetV18A(len(combined_rows))
    frame_v21a.frame_v18a._connect_rows_v18a(
        combined_rows, semantic_ids, disjoint
    )
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
    connected = defaultdict(list)
    for index in range(len(tagged)):
        connected[disjoint.find(index)].append(index)

    joint_components = []
    for members in connected.values():
        unit_indices = {"candidate": set(), "production": set()}
        for combined_index in members:
            item = tagged[combined_index]
            unit_indices[item["side"]].add(
                row_to_unit[item["side"]][item["side_index"]]
            )
        if any(len(unit_indices[side]) > 1 for side in unit_indices):
            raise RuntimeError("v22a isolated projection merged same-side units")
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
                "schema": "eggroll-es-joint-component-v22a",
                "members": identities,
            }),
            "candidate_unit": candidate_unit,
            "production_unit": production_unit,
            "relation": relation,
            "match_class": match_class,
        })
    joint_components.sort(key=lambda item: item["joint_id"])
    return {"side_units": side_units, "joint_components": joint_components}


def _internal_projection_v22a() -> dict:
    """Run only in the isolated subprocess; emit hashes/counts, never rows."""
    verify_bound_inputs_v22a()
    candidate = _read_jsonl_internal_only(candidate_v341.OUTPUT_PATH_V341)
    production = _read_jsonl_internal_only(
        frame_v21a.frame_v18a.PRODUCTION_PATH_V18A
    )
    if len(candidate) != 528 or len(production) != 784:
        raise RuntimeError("v22a isolated train row count changed")
    frame = _build_joint_frame_internal_only(candidate, production)
    components = frame["joint_components"]
    candidate_units = frame["side_units"]["candidate"]
    production_units = frame["side_units"]["production"]
    relations = Counter(item["relation"] for item in components)
    matches = Counter(item["match_class"] for item in components)
    candidate_capacity = Counter(unit["stratum"] for unit in candidate_units)
    candidate_only_topics = Counter(
        candidate_units[item["candidate_unit"]]["stratum"]
        for item in components if item["relation"] == "candidate_only"
    )
    if (
        len(candidate_units) != 259
        or len(production_units) != 272
        or len(components) != 326
        or relations != {
            "paired": 205, "candidate_only": 54, "production_only": 67,
        }
        or matches != {
            "shared_document": 203,
            "shared_url_without_shared_document": 2,
            "candidate_only": 54,
            "production_only": 67,
        }
        or candidate_capacity != {
            "safety_consent": 81,
            "technique": 71,
            "equipment_material": 23,
            "resources_general": 84,
        }
        or candidate_only_topics != {
            "safety_consent": 19,
            "technique": 17,
            "equipment_material": 10,
            "resources_general": 8,
        }
    ):
        raise RuntimeError("v22a exact-v341 joint frame changed")

    frozen_frame, frozen_tiers, base_flow = frame_v21a._base_flow_v21a()
    frozen_components = frozen_frame["joint_components"]
    frozen_production_units = frozen_frame["side_units"]["production"]
    by_production_unit_id = {
        production_units[item["production_unit"]]["unit_id"]: index
        for index, item in enumerate(components)
        if item["production_unit"] is not None
    }
    panels = []
    selected_global = []
    replacement_global = []
    unchanged_global = []
    for panel in PANEL_NAMES_V22A:
        frozen_indices = base_flow["selected_base"][panel]
        selected = []
        for frozen_index in frozen_indices:
            production_index = frozen_components[frozen_index]["production_unit"]
            unit_id = frozen_production_units[production_index]["unit_id"]
            selected.append(by_production_unit_id[unit_id])
        base_categories = Counter(
            frozen_tiers["base_category"][index] for index in frozen_indices
        )
        production_topics = Counter(
            frozen_production_units[
                frozen_components[index]["production_unit"]
            ]["stratum"]
            for index in frozen_indices
        )
        paired = [
            index for index in selected
            if components[index]["relation"] == "paired"
        ]
        production_only = [
            index for index in selected
            if components[index]["relation"] == "production_only"
        ]
        if (
            len(selected) != len(set(selected)) != 0
            or len(selected) != 24
            or base_categories != {
                category: BASE_CATEGORY_QUOTA_V22A
                for category in BASE_CATEGORIES_V22A
            }
            or production_topics != PRODUCTION_TOPIC_QUOTAS_V22A
            or len(paired) != EXPECTED_REPLACEMENTS_BY_PANEL_V22A[panel]
            or len(production_only) != EXPECTED_UNCHANGED_BY_PANEL_V22A[panel]
        ):
            raise RuntimeError("v22a exact frozen panel contract changed")
        selected_ids = [components[index]["joint_id"] for index in selected]
        control_representatives = []
        treatment_representatives = []
        for index in selected:
            component = components[index]
            production_unit = production_units[component["production_unit"]]
            control_representatives.append(
                f"{component['joint_id']}:production:"
                f"{production_unit['representative_row_sha256']}"
            )
            if component["relation"] == "paired":
                candidate_unit = candidate_units[component["candidate_unit"]]
                treatment_representatives.append(
                    f"{component['joint_id']}:candidate:"
                    f"{candidate_unit['representative_row_sha256']}"
                )
            else:
                treatment_representatives.append(
                    f"{component['joint_id']}:production:"
                    f"{production_unit['representative_row_sha256']}"
                )
        selected_global.extend(selected_ids)
        replacement_global.extend(components[index]["joint_id"] for index in paired)
        unchanged_global.extend(
            components[index]["joint_id"] for index in production_only
        )
        panels.append({
            "name": panel,
            "role": (
                "optimization" if panel in OPTIMIZATION_PANELS_V22A
                else "train_only_screen"
            ),
            "sampled_component_count": 24,
            "base_category_counts": dict(base_categories),
            "production_topic_counts": dict(production_topics),
            "matched_candidate_replacement_count": len(paired),
            "unchanged_production_only_count": len(production_only),
            "excluded_candidate_only_count": 0,
            "sampled_component_identity_root_sha256": identity_root_sha256(
                selected_ids
            ),
            "matched_replacement_component_identity_root_sha256": (
                identity_root_sha256(
                    components[index]["joint_id"] for index in paired
                )
            ),
            "unchanged_component_identity_root_sha256": identity_root_sha256(
                components[index]["joint_id"] for index in production_only
            ),
            "control_representative_assignment_root_sha256": (
                identity_root_sha256(control_representatives)
            ),
            "treatment_representative_assignment_root_sha256": (
                identity_root_sha256(treatment_representatives)
            ),
        })
    if (
        len(selected_global) != len(set(selected_global)) != 0
        or len(selected_global) != 240
        or len(replacement_global) != len(set(replacement_global)) != 0
        or len(replacement_global) != 184
        or len(unchanged_global) != len(set(unchanged_global)) != 0
        or len(unchanged_global) != 56
        or set(replacement_global) & set(unchanged_global)
    ):
        raise RuntimeError("v22a global exact-v341 replacement contract changed")
    projection = {
        "schema": "eggroll-es-v341-matched-replacement-projection-v22a",
        "joint_frame": {
            "candidate_component_count": len(candidate_units),
            "production_component_count": len(production_units),
            "joint_component_count": len(components),
            "relation_counts": dict(relations),
            "match_class_counts": dict(matches),
            "candidate_topic_component_counts": dict(candidate_capacity),
            "candidate_only_topic_counts": dict(candidate_only_topics),
            "joint_component_identity_root_sha256": identity_root_sha256(
                item["joint_id"] for item in components
            ),
        },
        "frozen_production_base": {
            "population_component_count": 272,
            "panel_count": 10,
            "sampled_component_count": 240,
            "globally_panel_disjoint": True,
            "sampled_component_identity_root_sha256": identity_root_sha256(
                selected_global
            ),
            "matched_candidate_replacement_count": 184,
            "unchanged_production_only_count": 56,
            "excluded_candidate_only_count": 54,
            "matched_replacement_component_identity_root_sha256": (
                identity_root_sha256(replacement_global)
            ),
            "unchanged_component_identity_root_sha256": identity_root_sha256(
                unchanged_global
            ),
        },
        "panels": panels,
        "firewall": {
            "isolated_worker_read_train_rows": True,
            "projection_contains_row_or_qa_content": False,
            "projection_contains_only_hashes_counts_and_safe_identifiers": True,
            "heldout_validation_ood_eval_or_benchmark_content_opened": False,
        },
    }
    projection["content_sha256_before_self_field"] = canonical_sha256(projection)
    return validate_projection_v22a(projection)


def validate_projection_v22a(value: dict) -> dict:
    joint = value.get("joint_frame", {})
    base = value.get("frozen_production_base", {})
    panels = value.get("panels", [])
    firewall = value.get("firewall", {})
    if (
        value.get("schema")
        != "eggroll-es-v341-matched-replacement-projection-v22a"
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(value))
        or joint.get("candidate_component_count") != 259
        or joint.get("production_component_count") != 272
        or joint.get("joint_component_count") != 326
        or joint.get("relation_counts") != {
            "paired": 205, "candidate_only": 54, "production_only": 67,
        }
        or joint.get("match_class_counts") != {
            "shared_document": 203,
            "shared_url_without_shared_document": 2,
            "candidate_only": 54,
            "production_only": 67,
        }
        or base.get("population_component_count") != 272
        or base.get("sampled_component_count") != 240
        or base.get("globally_panel_disjoint") is not True
        or base.get("matched_candidate_replacement_count") != 184
        or base.get("unchanged_production_only_count") != 56
        or base.get("excluded_candidate_only_count") != 54
        or len(panels) != 10
        or tuple(panel.get("name") for panel in panels) != PANEL_NAMES_V22A
        or any(
            panel.get("sampled_component_count") != 24
            or panel.get("matched_candidate_replacement_count")
            != EXPECTED_REPLACEMENTS_BY_PANEL_V22A[panel["name"]]
            or panel.get("unchanged_production_only_count")
            != EXPECTED_UNCHANGED_BY_PANEL_V22A[panel["name"]]
            or panel.get("excluded_candidate_only_count") != 0
            or panel.get("base_category_counts")
            != {
                category: BASE_CATEGORY_QUOTA_V22A
                for category in BASE_CATEGORIES_V22A
            }
            or panel.get("production_topic_counts")
            != PRODUCTION_TOPIC_QUOTAS_V22A
            for panel in panels
        )
        or firewall != {
            "isolated_worker_read_train_rows": True,
            "projection_contains_row_or_qa_content": False,
            "projection_contains_only_hashes_counts_and_safe_identifiers": True,
            "heldout_validation_ood_eval_or_benchmark_content_opened": False,
        }
    ):
        raise RuntimeError("v22a firewalled train projection changed")
    return value


def run_firewalled_projection_v22a() -> dict:
    process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--internal-projection"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError("v22a isolated train projection failed closed")
    try:
        projection = json.loads(process.stdout)
    except (json.JSONDecodeError, TypeError) as error:
        raise RuntimeError("v22a isolated projection emitted non-contract output") from error
    return validate_projection_v22a(projection)


def build_certificate_v22a() -> dict:
    bound = verify_bound_inputs_v22a()
    projection = run_firewalled_projection_v22a()
    panels = []
    for panel in projection["panels"]:
        arms = {
            "production_control": {
                "requests": 24,
                "population_denominator": 272,
                "candidate_representatives": 0,
                "production_representatives": 24,
                "active_component_identity_root_sha256": panel[
                    "sampled_component_identity_root_sha256"
                ],
                "representative_assignment_root_sha256": panel[
                    "control_representative_assignment_root_sha256"
                ],
            },
            "v341_matched_replacement": {
                "requests": 24,
                "population_denominator": 272,
                "candidate_representatives": panel[
                    "matched_candidate_replacement_count"
                ],
                "production_representatives": panel[
                    "unchanged_production_only_count"
                ],
                "active_component_identity_root_sha256": panel[
                    "sampled_component_identity_root_sha256"
                ],
                "representative_assignment_root_sha256": panel[
                    "treatment_representative_assignment_root_sha256"
                ],
            },
        }
        panels.append({**panel, "arms": arms})
    value = {
        "schema": "eggroll-es-v341-matched-replacement-flow-certificate-v22a",
        "status": "feasible_offline_frame_only_no_runtime_or_update_authority",
        "arm_order": list(ARM_ORDER_V22A),
        "inputs": {
            "candidate_v341": {
                "source_commit": candidate_v341.V341_SOURCE_COMMIT,
                "sealed_snapshot_commit": CANDIDATE_SEAL_COMMIT_V22A,
                "path": str(candidate_v341.OUTPUT_PATH_V341),
                "rows": candidate_v341.V341_ROWS,
                "file_sha256": candidate_v341.V341_SHA256,
                "manifest_path": str(candidate_v341.MANIFEST_PATH_V341),
                "manifest_file_sha256": CANDIDATE_MANIFEST_SHA256_V22A,
                "manifest_content_sha256": (
                    CANDIDATE_MANIFEST_CONTENT_SHA256_V22A
                ),
                "ongoing_curation_used": False,
            },
            "production": {
                "commit": PRODUCTION_COMMIT_V22A,
                "path": str(frame_v21a.frame_v18a.PRODUCTION_PATH_V18A),
                "rows": 784,
                "file_sha256": PRODUCTION_SHA256_V22A,
            },
            "frozen_v19a_panel_source": {
                "via_v21a_frame_commit": V21A_FRAME_COMMIT_V22A,
                "via_v21a_frame_file_sha256": (
                    V21A_FRAME_CERTIFICATE_SHA256_V22A
                ),
                "via_v21a_frame_content_sha256": (
                    V21A_FRAME_CERTIFICATE_CONTENT_SHA256_V22A
                ),
            },
        },
        "preregistration_time_correction": {
            "cause": (
                "exact_v341_correctness_removal_eliminated_one_v331_pair"
            ),
            "prior_assumed_relation_counts": {
                "paired": 206, "candidate_only": 54, "production_only": 66,
            },
            "exact_v341_relation_counts": {
                "paired": 205, "candidate_only": 54, "production_only": 67,
            },
            "affected_frozen_sample_panel": "train_screen_0",
            "prior_expected_sampled_replacements": 185,
            "exact_v341_sampled_replacements": 184,
            "fabricated_or_substituted_candidate_representative": False,
            "correction_made_before_runtime_or_result_observation": True,
            "post_result_adaptation": False,
        },
        "joint_frame": projection["joint_frame"],
        "frozen_production_base": {
            **projection["frozen_production_base"],
            "source": "exact_frozen_v19a_ten_panel_base_flow",
            "base_category_quota_per_panel": {
                category: BASE_CATEGORY_QUOTA_V22A
                for category in BASE_CATEGORIES_V22A
            },
            "production_topic_quota_per_panel": PRODUCTION_TOPIC_QUOTAS_V22A,
            "quota_seed_or_solver_relaxation_used": False,
        },
        "estimand": {
            "semantics": (
                "same_frozen_production_components_with_v341_candidate_"
                "representative_only_when_exactly_paired"
            ),
            "arm_population_denominators": ARM_POPULATION_DENOMINATORS_V22A,
            "arm_requests_per_panel": ARM_REQUESTS_PER_PANEL_V22A,
            "base_population_ht_strata": {
                category: {
                    "population": population,
                    "per_panel_quota": BASE_CATEGORY_QUOTA_V22A,
                    "horvitz_thompson_weight": (
                        population / BASE_CATEGORY_QUOTA_V22A
                    ),
                }
                for category, population in BASE_CATEGORY_POPULATIONS_V22A.items()
            },
            "same_components_panels_ht_weights_and_denominator_both_arms": True,
            "candidate_only_additions_excluded": 54,
            "sampled_matched_replacements": 184,
            "sampled_production_only_unchanged": 56,
            "plain_request_mean_used": False,
        },
        "panels": panels,
        "isolated_projection_content_sha256": projection[
            "content_sha256_before_self_field"
        ],
        "bound_input_content_sha256": {
            "candidate_manifest": bound["candidate_manifest"][
                "content_sha256_before_self_field"
            ],
            "prior_frame": bound["prior_frame"][
                "content_sha256_before_self_field"
            ],
        },
        "firewall": {
            "foreground_builder_opened_or_parsed_jsonl_rows": False,
            "isolated_projection_worker_read_train_rows": True,
            "projection_contains_question_answer_prompt_response_or_row_content": False,
            "contains_evaluation_content": False,
            "heldout_validation_ood_eval_or_benchmark_content_opened": False,
            "runtime_launch_authorized": False,
            "gpu_launch_authorized": False,
            "model_update_authorized": False,
            "checkpoint_write_authorized": False,
            "evaluation_authorized": False,
            "dataset_promotion_authorized": False,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return validate_certificate_v22a(value)


def validate_certificate_v22a(value: dict) -> dict:
    correction = value.get("preregistration_time_correction", {})
    joint = value.get("joint_frame", {})
    base = value.get("frozen_production_base", {})
    estimand = value.get("estimand", {})
    panels = value.get("panels", [])
    firewall = value.get("firewall", {})
    projected_base_keys = {
        "population_component_count", "panel_count", "sampled_component_count",
        "globally_panel_disjoint", "sampled_component_identity_root_sha256",
        "matched_candidate_replacement_count",
        "unchanged_production_only_count", "excluded_candidate_only_count",
        "matched_replacement_component_identity_root_sha256",
        "unchanged_component_identity_root_sha256",
    }
    projected_panel_keys = {
        "name", "role", "sampled_component_count", "base_category_counts",
        "production_topic_counts", "matched_candidate_replacement_count",
        "unchanged_production_only_count", "excluded_candidate_only_count",
        "sampled_component_identity_root_sha256",
        "matched_replacement_component_identity_root_sha256",
        "unchanged_component_identity_root_sha256",
        "control_representative_assignment_root_sha256",
        "treatment_representative_assignment_root_sha256",
    }
    reconstructed_projection = {
        "schema": "eggroll-es-v341-matched-replacement-projection-v22a",
        "joint_frame": joint,
        "frozen_production_base": {
            key: base.get(key) for key in projected_base_keys
        },
        "panels": [
            {key: panel.get(key) for key in projected_panel_keys}
            for panel in panels
        ],
        "firewall": {
            "isolated_worker_read_train_rows": True,
            "projection_contains_row_or_qa_content": False,
            "projection_contains_only_hashes_counts_and_safe_identifiers": True,
            "heldout_validation_ood_eval_or_benchmark_content_opened": False,
        },
    }
    reconstructed_projection["content_sha256_before_self_field"] = (
        canonical_sha256(reconstructed_projection)
    )
    if (
        value.get("schema")
        != "eggroll-es-v341-matched-replacement-flow-certificate-v22a"
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(value))
        or value.get("arm_order") != list(ARM_ORDER_V22A)
        or value.get("isolated_projection_content_sha256")
        != reconstructed_projection["content_sha256_before_self_field"]
        or value.get("inputs", {}).get("candidate_v341", {}).get("source_commit")
        != candidate_v341.V341_SOURCE_COMMIT
        or value.get("inputs", {}).get("candidate_v341", {}).get("file_sha256")
        != candidate_v341.V341_SHA256
        or value.get("inputs", {}).get("candidate_v341", {}).get(
            "ongoing_curation_used"
        ) is not False
        or correction.get("exact_v341_relation_counts") != {
            "paired": 205, "candidate_only": 54, "production_only": 67,
        }
        or correction.get("exact_v341_sampled_replacements") != 184
        or correction.get("fabricated_or_substituted_candidate_representative")
        is not False
        or correction.get("correction_made_before_runtime_or_result_observation")
        is not True
        or correction.get("post_result_adaptation") is not False
        or joint.get("relation_counts") != {
            "paired": 205, "candidate_only": 54, "production_only": 67,
        }
        or base.get("population_component_count") != 272
        or base.get("sampled_component_count") != 240
        or base.get("globally_panel_disjoint") is not True
        or base.get("matched_candidate_replacement_count") != 184
        or base.get("unchanged_production_only_count") != 56
        or base.get("excluded_candidate_only_count") != 54
        or base.get("quota_seed_or_solver_relaxation_used") is not False
        or estimand.get("arm_population_denominators")
        != ARM_POPULATION_DENOMINATORS_V22A
        or estimand.get("arm_requests_per_panel")
        != ARM_REQUESTS_PER_PANEL_V22A
        or estimand.get("base_population_ht_strata") != {
            category: {
                "population": population,
                "per_panel_quota": BASE_CATEGORY_QUOTA_V22A,
                "horvitz_thompson_weight": (
                    population / BASE_CATEGORY_QUOTA_V22A
                ),
            }
            for category, population in BASE_CATEGORY_POPULATIONS_V22A.items()
        }
        or estimand.get(
            "same_components_panels_ht_weights_and_denominator_both_arms"
        ) is not True
        or estimand.get("candidate_only_additions_excluded") != 54
        or len(panels) != 10
        or tuple(panel.get("name") for panel in panels) != PANEL_NAMES_V22A
        or any(
            set(panel.get("arms", {})) != set(ARM_ORDER_V22A)
            or panel["arms"]["production_control"].get("requests") != 24
            or panel["arms"]["v341_matched_replacement"].get("requests") != 24
            or panel["arms"]["production_control"].get(
                "active_component_identity_root_sha256"
            ) != panel["arms"]["v341_matched_replacement"].get(
                "active_component_identity_root_sha256"
            )
            or panel["arms"]["production_control"].get(
                "active_component_identity_root_sha256"
            ) != panel.get("sampled_component_identity_root_sha256")
            or panel["arms"]["production_control"].get(
                "representative_assignment_root_sha256"
            ) != panel.get("control_representative_assignment_root_sha256")
            or panel["arms"]["v341_matched_replacement"].get(
                "representative_assignment_root_sha256"
            ) != panel.get("treatment_representative_assignment_root_sha256")
            or panel["arms"]["v341_matched_replacement"].get(
                "candidate_representatives"
            ) != EXPECTED_REPLACEMENTS_BY_PANEL_V22A[panel["name"]]
            or panel["arms"]["v341_matched_replacement"].get(
                "production_representatives"
            ) != EXPECTED_UNCHANGED_BY_PANEL_V22A[panel["name"]]
            for panel in panels
        )
        or firewall.get("foreground_builder_opened_or_parsed_jsonl_rows")
        is not False
        or firewall.get("isolated_projection_worker_read_train_rows") is not True
        or firewall.get(
            "heldout_validation_ood_eval_or_benchmark_content_opened"
        ) is not False
        or any(firewall.get(key) is not False for key in (
            "runtime_launch_authorized", "gpu_launch_authorized",
            "model_update_authorized", "checkpoint_write_authorized",
            "evaluation_authorized", "dataset_promotion_authorized",
        ))
    ):
        raise RuntimeError("v22a exact-v341 matched-replacement frame changed")
    return value


def _exclusive_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("immutable v22a frame certificate already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH_V22A))
    parser.add_argument("--internal-projection", action="store_true")
    args = parser.parse_args(argv)
    if args.internal_projection:
        projection = _internal_projection_v22a()
        print(json.dumps(projection, sort_keys=True, separators=(",", ":")))
        return projection
    if Path(args.output).resolve() != OUTPUT_PATH_V22A:
        raise ValueError("v22a frame certificate output path changed")
    certificate = build_certificate_v22a()
    _exclusive_write(OUTPUT_PATH_V22A, certificate)
    result = {
        "schema": "eggroll-es-v341-matched-replacement-flow-build-v22a",
        "output": str(OUTPUT_PATH_V22A),
        "file_sha256": file_sha256(OUTPUT_PATH_V22A),
        "content_sha256": certificate["content_sha256_before_self_field"],
        "joint_relation_counts": certificate["joint_frame"]["relation_counts"],
        "sampled_matched_replacements": 184,
        "sampled_production_only_unchanged": 56,
        "runtime_launch_authorized": False,
        "gpu_launched": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
