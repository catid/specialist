#!/usr/bin/env python3
"""Build the content-free paired production/v283 panel frame for V17A."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import eggroll_es_train_panel_sampler_v13 as sampler_v13


ROOT = Path(__file__).resolve().parent
CANDIDATE_PATH_V17A = (
    ROOT / "experiments/eggroll_es_hpo/dataset_candidates/v283/"
    "train_qa_context_merit_v283.jsonl"
).resolve()
CANDIDATE_MANIFEST_PATH_V17A = (
    ROOT / "experiments/eggroll_es_hpo/dataset_candidates/v283/manifest.json"
).resolve()
PRODUCTION_PATH_V17A = (ROOT / "data/train_qa_curated_v1.jsonl").resolve()
OUTPUT_PATH_V17A = (
    ROOT / "experiments/eggroll_es_hpo/train_panel_sampling_v17a/"
    "paired_data_version_panels_v17a.json"
).resolve()

CANDIDATE_SHA256_V17A = (
    "83d14d9d42740c836b49a8ec9e4237766e9d751c827c21d4d2c79500ee4bc3b9"
)
CANDIDATE_MANIFEST_SHA256_V17A = (
    "014f37177073d5a433b2da2b01298463cc87856f0278a60d66e53a0dce55bbfb"
)
PRODUCTION_SHA256_V17A = (
    "62e7ae28c86a458d4d33bf3f73f1b91b873c86e3f70ce87706a7394d1f391507"
)
SAMPLER_SHA256_V17A = (
    "81ca63c230995d44dfdb739a78b4a1a0f85d09724ba5915860ae36f78ccc3da9"
)
CANDIDATE_ROWS_V17A = 492
PRODUCTION_ROWS_V17A = 784
PANEL_SEED_V17A = 20260717
ROTATION_SEED_V17A = 20260718
PANEL_NAMES_V17A = (
    "optimization_0", "optimization_1", "optimization_2",
    "train_screen_0", "train_screen_1",
)
OPTIMIZATION_PANELS_V17A = PANEL_NAMES_V17A[:3]
TRAIN_SCREENS_V17A = PANEL_NAMES_V17A[3:]
PANEL_SIZE_V17A = 38
STRATUM_QUOTAS_V17A = {
    "safety_consent": 14,
    "technique": 8,
    "equipment_material": 2,
    "resources_general": 14,
}
REQUIRED_PAIRED_STRATA_V17A = {
    name: len(PANEL_NAMES_V17A) * quota
    for name, quota in STRATUM_QUOTAS_V17A.items()
}
EXPECTED_JOINT_COMPONENTS_V17A = 276
EXPECTED_PAIRED_UNITS_V17A = 195
EXPECTED_CANDIDATE_ONLY_UNITS_V17A = 4
EXPECTED_PRODUCTION_ONLY_UNITS_V17A = 77
EXPECTED_PAIRED_STRATA_V17A = {
    "safety_consent": 70,
    "technique": 41,
    "equipment_material": 13,
    "resources_general": 71,
}


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def normalize_url_v17a(value):
    parsed = urlsplit(str(value).strip())
    if not parsed.scheme or not parsed.netloc:
        return str(value).strip()
    query = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
        and key.lower() not in {"si", "fbclid", "gclid"}
    ]
    path = re.sub(r"/+", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((
        parsed.scheme.lower(), parsed.netloc.lower(), path,
        urlencode(sorted(query)), "",
    ))


def row_urls_v17a(row):
    scalar = (
        "url", "evidence_url", "canonical_url", "supplied_url",
        "original_ropetopia_url", "title_evidence_url", "url_evidence_url",
        "canonical_resource_url",
    )
    arrays = ("urls", "canonical_urls", "supplied_urls")
    values = {row[key] for key in scalar if row.get(key)}
    values.update(
        url for key in arrays for url in row.get(key, ()) if url
    )
    return {normalize_url_v17a(value) for value in values}


class DisjointSetV17A:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def _load_bound_rows_v17a():
    expected = {
        CANDIDATE_PATH_V17A: CANDIDATE_SHA256_V17A,
        CANDIDATE_MANIFEST_PATH_V17A: CANDIDATE_MANIFEST_SHA256_V17A,
        PRODUCTION_PATH_V17A: PRODUCTION_SHA256_V17A,
        Path(sampler_v13.__file__).resolve(): SAMPLER_SHA256_V17A,
    }
    if any(file_sha256(path) != digest for path, digest in expected.items()):
        raise RuntimeError("v17a frozen data or conflict implementation changed")
    candidate = _read_jsonl(CANDIDATE_PATH_V17A)
    production = _read_jsonl(PRODUCTION_PATH_V17A)
    if len(candidate) != CANDIDATE_ROWS_V17A or len(production) != PRODUCTION_ROWS_V17A:
        raise RuntimeError("v17a frozen row counts changed")
    return candidate, production


def _member_order_v17a(unit_id, side, members):
    ordered = sorted(
        members,
        key=lambda item: canonical_sha256({
            "schema": "eggroll-es-v17a-side-row-order",
            "seed": ROTATION_SEED_V17A,
            "unit_id": unit_id,
            "side": side,
            "row_sha256": item["row_sha256"],
        }),
    )
    return {
        "row_count": len(ordered),
        "row_indices": [item["row_index"] for item in ordered],
        "row_sha256": [item["row_sha256"] for item in ordered],
        "ordered_row_sha256": canonical_sha256(
            [item["row_sha256"] for item in ordered]
        ),
        "rotation": (
            "direction_index_mod_side_row_count; plus_and_minus_share_row"
        ),
    }


def build_joint_units_v17a(candidate, production):
    tagged = [
        {"side": "candidate", "row_index": index, "row": row}
        for index, row in enumerate(candidate)
    ] + [
        {"side": "production", "row_index": index, "row": row}
        for index, row in enumerate(production)
    ]
    rows = [item["row"] for item in tagged]
    semantic_ids = sampler_v13.build_semantic_clusters(rows)
    disjoint = DisjointSetV17A(len(tagged))
    first = {}

    def connect(identifier, index):
        if identifier in first:
            disjoint.union(index, first[identifier])
        else:
            first[identifier] = index

    for index, (item, semantic_id) in enumerate(zip(tagged, semantic_ids)):
        row = item["row"]
        connect(f"document:{row['document_sha256']}", index)
        for url in row_urls_v17a(row):
            connect(f"url:{url}", index)
        lineage = row.get("source_lineage") or {}
        for key in ("raw", "raw_document", "raw_successor_document"):
            if lineage.get(key):
                connect(
                    f"lineage:{key}:{json.dumps(lineage[key], sort_keys=True)}",
                    index,
                )
        connect(f"semantic:{semantic_id}", index)

    components = defaultdict(list)
    for index in range(len(tagged)):
        components[disjoint.find(index)].append(tagged[index])

    paired = []
    candidate_only = 0
    production_only = 0
    assignments = []
    for members in components.values():
        by_side = {
            side: [item for item in members if item["side"] == side]
            for side in ("candidate", "production")
        }
        if not by_side["candidate"]:
            production_only += 1
            continue
        if not by_side["production"]:
            candidate_only += 1
            continue
        member_ids = sorted(
            f"{item['side']}:{sampler_v13.row_sha256(item['row'])}"
            for item in members
        )
        unit_id = canonical_sha256({
            "schema": "eggroll-es-joint-conflict-unit-v17a",
            "members": member_ids,
        })
        candidate_counts = Counter(
            sampler_v13.classify_stratum(item["row"])
            for item in by_side["candidate"]
        )
        stratum = max(
            sampler_v13.STRATA,
            key=lambda name: (
                candidate_counts[name], sampler_v13._TIE_PRIORITY[name],
            ),
        )
        side_contract = {}
        for side in ("candidate", "production"):
            side_members = [{
                "row_index": item["row_index"],
                "row_sha256": sampler_v13.row_sha256(item["row"]),
            } for item in by_side[side]]
            side_contract[side] = _member_order_v17a(
                unit_id, side, side_members,
            )
            assignments.extend(
                f"{side}:{row_sha256}\t{unit_id}"
                for row_sha256 in side_contract[side]["row_sha256"]
            )
        paired.append({
            "unit_id": unit_id,
            "stratum": stratum,
            "candidate_stratum_row_counts": dict(sorted(candidate_counts.items())),
            "sides": side_contract,
        })
    return {
        "joint_component_count": len(components),
        "paired_units": paired,
        "candidate_only_unit_count": candidate_only,
        "production_only_unit_count": production_only,
        "paired_assignment_root_sha256": hashlib.sha256(
            ("\n".join(sorted(assignments)) + "\n").encode()
        ).hexdigest(),
    }


def _panel_order_v17a(name, items):
    return sorted(
        items,
        key=lambda item: canonical_sha256({
            "schema": "eggroll-es-v17a-panel-order",
            "seed": PANEL_SEED_V17A,
            "panel": name,
            "unit_id": item["unit_id"],
        }),
    )


def build_manifest_v17a():
    candidate, production = _load_bound_rows_v17a()
    joint = build_joint_units_v17a(candidate, production)
    paired = joint.pop("paired_units")
    strata = Counter(unit["stratum"] for unit in paired)
    if (
        joint["joint_component_count"] != EXPECTED_JOINT_COMPONENTS_V17A
        or len(paired) != EXPECTED_PAIRED_UNITS_V17A
        or joint["candidate_only_unit_count"]
        != EXPECTED_CANDIDATE_ONLY_UNITS_V17A
        or joint["production_only_unit_count"]
        != EXPECTED_PRODUCTION_ONLY_UNITS_V17A
        or dict(strata) != EXPECTED_PAIRED_STRATA_V17A
        or any(
            strata[name] < required
            for name, required in REQUIRED_PAIRED_STRATA_V17A.items()
        )
    ):
        raise RuntimeError("v17a joint paired capacity changed or is insufficient")

    by_stratum = {
        name: sorted(
            (unit for unit in paired if unit["stratum"] == name),
            key=lambda unit: canonical_sha256({
                "schema": "eggroll-es-v17a-stratum-permutation",
                "seed": PANEL_SEED_V17A,
                "stratum": name,
                "unit_id": unit["unit_id"],
            }),
        )
        for name in sampler_v13.STRATA
    }
    panels = []
    selected_ids = []
    for panel_index, panel_name in enumerate(PANEL_NAMES_V17A):
        selected = []
        for stratum in sampler_v13.STRATA:
            quota = STRATUM_QUOTAS_V17A[stratum]
            start = panel_index * quota
            population = len(by_stratum[stratum])
            for unit in by_stratum[stratum][start:start + quota]:
                selected.append({
                    **unit,
                    "inclusion_probability_per_panel": quota / population,
                    "horvitz_thompson_unit_weight": population / quota,
                })
        selected = _panel_order_v17a(panel_name, selected)
        if len(selected) != PANEL_SIZE_V17A:
            raise RuntimeError("v17a paired panel size changed")
        selected_ids.extend(item["unit_id"] for item in selected)
        panels.append({
            "name": panel_name,
            "role": (
                "optimization" if panel_name in OPTIMIZATION_PANELS_V17A
                else "train_only_screen"
            ),
            "rows_per_side_per_direction": PANEL_SIZE_V17A,
            "stratum_counts": dict(STRATUM_QUOTAS_V17A),
            "ordered_unit_identity_sha256": canonical_sha256(
                [item["unit_id"] for item in selected]
            ),
            "items": selected,
        })
    if len(selected_ids) != len(set(selected_ids)) != 0:
        raise RuntimeError("v17a panels are not globally conflict-unit disjoint")
    if len(selected_ids) != len(PANEL_NAMES_V17A) * PANEL_SIZE_V17A:
        raise RuntimeError("v17a selected paired-unit coverage changed")

    manifest = {
        "schema": "eggroll-es-paired-data-version-panel-manifest-v17a",
        "status": "paired_alpha_zero_frame_only_no_gpu_or_update_authorization",
        "inputs": {
            "candidate": {
                "path": str(CANDIDATE_PATH_V17A),
                "rows": CANDIDATE_ROWS_V17A,
                "file_sha256": CANDIDATE_SHA256_V17A,
                "freeze_manifest_path": str(CANDIDATE_MANIFEST_PATH_V17A),
                "freeze_manifest_file_sha256": CANDIDATE_MANIFEST_SHA256_V17A,
            },
            "production": {
                "path": str(PRODUCTION_PATH_V17A),
                "rows": PRODUCTION_ROWS_V17A,
                "file_sha256": PRODUCTION_SHA256_V17A,
            },
            "conflict_implementation": {
                "path": str(Path(sampler_v13.__file__).resolve()),
                "file_sha256": SAMPLER_SHA256_V17A,
            },
        },
        "joint_frame": {
            **joint,
            "paired_unit_count": len(paired),
            "paired_stratum_counts": dict(strata),
            "selected_paired_unit_count": len(selected_ids),
            "reserve_paired_unit_count": len(paired) - len(selected_ids),
            "selected_unit_id_root_sha256": hashlib.sha256(
                ("\n".join(sorted(selected_ids)) + "\n").encode()
            ).hexdigest(),
            "cross_side_grouping": (
                "connected_components_over_combined_candidate_and_production_"
                "rows_using_document_normalized_url_lineage_family_and_pinned_"
                "v13_lexical_semantic_links"
            ),
            "stratum_assignment": "candidate_side_dominant_explicit_stratum",
        },
        "panel_contract": {
            "panel_seed": PANEL_SEED_V17A,
            "rotation_seed": ROTATION_SEED_V17A,
            "panel_names": list(PANEL_NAMES_V17A),
            "optimization_panels": list(OPTIMIZATION_PANELS_V17A),
            "train_only_screens": list(TRAIN_SCREENS_V17A),
            "panel_size": PANEL_SIZE_V17A,
            "stratum_quotas": dict(STRATUM_QUOTAS_V17A),
            "required_paired_strata": dict(REQUIRED_PAIRED_STRATA_V17A),
            "globally_disjoint_joint_conflict_units": True,
            "same_unit_membership_order_and_direction_rotation_both_versions": True,
            "plus_minus_share_exact_rotated_row_per_side": True,
            "same_resident_perturbation_scores_both_versions": True,
            "version_generation_order": (
                "alternate_candidate_first_and_production_first_by_signed_wave"
            ),
            "estimand": (
                "equal_weight_joint_conflict_unit_mean_with_exact_stratum_"
                "Horvitz_Thompson_weights"
            ),
        },
        "panels": panels,
        "separation": {
            "v17a_role": "matched_data_version_compatibility_only",
            "v17b_role": "separate_full_candidate_5x39_hpo_if_v17a_passes",
            "v17a_common_subset_is_not_v17b_training_estimand": True,
            "contains_row_prompt_or_answer_content": False,
            "contains_evaluation_content": False,
        },
    }
    manifest["content_sha256_before_self_field"] = canonical_sha256(manifest)
    validate_manifest_v17a(manifest)
    return manifest


def validate_manifest_v17a(manifest):
    if (
        manifest.get("schema")
        != "eggroll-es-paired-data-version-panel-manifest-v17a"
        or manifest.get("content_sha256_before_self_field")
        != canonical_sha256({
            key: value for key, value in manifest.items()
            if key != "content_sha256_before_self_field"
        })
        or manifest.get("joint_frame", {}).get("joint_component_count")
        != EXPECTED_JOINT_COMPONENTS_V17A
        or manifest.get("joint_frame", {}).get("paired_unit_count")
        != EXPECTED_PAIRED_UNITS_V17A
        or manifest.get("joint_frame", {}).get("paired_stratum_counts")
        != EXPECTED_PAIRED_STRATA_V17A
        or manifest.get("joint_frame", {}).get("reserve_paired_unit_count") != 5
        or [panel.get("name") for panel in manifest.get("panels", [])]
        != list(PANEL_NAMES_V17A)
        or any(
            len(panel.get("items", [])) != PANEL_SIZE_V17A
            or panel.get("stratum_counts") != STRATUM_QUOTAS_V17A
            for panel in manifest.get("panels", [])
        )
    ):
        raise RuntimeError("v17a paired manifest contract changed")
    items = [item for panel in manifest["panels"] for item in panel["items"]]
    if (
        len(items) != 190
        or len({item["unit_id"] for item in items}) != 190
        or any(set(item["sides"]) != {"candidate", "production"} for item in items)
        or any(
            side["row_count"] <= 0
            or side["row_count"] != len(side["row_indices"])
            or side["row_count"] != len(side["row_sha256"])
            or side["ordered_row_sha256"]
            != canonical_sha256(side["row_sha256"])
            for item in items for side in item["sides"].values()
        )
        or any(
            not math.isclose(
                item["inclusion_probability_per_panel"]
                * item["horvitz_thompson_unit_weight"],
                1.0, rel_tol=1e-15, abs_tol=1e-15,
            )
            for item in items
        )
    ):
        raise RuntimeError("v17a paired item identity or weighting changed")
    return manifest


def _exclusive_write(path, value):
    path = Path(path).resolve()
    if path != OUTPUT_PATH_V17A:
        raise ValueError("v17a output path changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("v17a paired manifest already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH_V17A))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if Path(args.output).resolve() != OUTPUT_PATH_V17A:
        raise ValueError("v17a output path changed")
    manifest = build_manifest_v17a()
    if not args.dry_run:
        _exclusive_write(args.output, manifest)
    result = {
        "schema": "eggroll-es-paired-panel-build-v17a",
        "output": str(OUTPUT_PATH_V17A),
        "content_sha256": manifest["content_sha256_before_self_field"],
        "joint_component_count": manifest["joint_frame"]["joint_component_count"],
        "paired_unit_count": manifest["joint_frame"]["paired_unit_count"],
        "paired_stratum_counts": manifest["joint_frame"]["paired_stratum_counts"],
        "selected_paired_unit_count": manifest["joint_frame"]["selected_paired_unit_count"],
        "reserve_paired_unit_count": manifest["joint_frame"]["reserve_paired_unit_count"],
        "gpu_launched": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
