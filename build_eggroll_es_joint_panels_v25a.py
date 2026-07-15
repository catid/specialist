#!/usr/bin/env python3
"""Build a content-free paired production/v364 train-only panel frame."""

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
CANDIDATE_PATH = (
    ROOT / "experiments/eggroll_es_hpo/dataset_candidates/v364/"
    "train_qa_context_merit_v364.jsonl"
).resolve()
CANDIDATE_MANIFEST_PATH = (
    ROOT / "experiments/eggroll_es_hpo/dataset_candidates/v364/manifest.json"
).resolve()
PRODUCTION_PATH = (ROOT / "data/train_qa_curated_v1.jsonl").resolve()
OUTPUT_PATH = (
    ROOT / "experiments/eggroll_es_hpo/train_panel_sampling_v25a/"
    "paired_data_version_panels_v25a.json"
).resolve()

CANDIDATE_SHA256 = "874b77dd8ef988bb24d4b13999ddebc2068b053eab208def9bb1e23e7138c36a"
CANDIDATE_MANIFEST_SHA256 = "42ca9d7186f70b94681b88c70eca341d19fff50ac422e2539b05912fe5cf6bfd"
PRODUCTION_SHA256 = "62e7ae28c86a458d4d33bf3f73f1b91b873c86e3f70ce87706a7394d1f391507"
SAMPLER_SHA256 = "81ca63c230995d44dfdb739a78b4a1a0f85d09724ba5915860ae36f78ccc3da9"
CANDIDATE_ROWS = 531
PRODUCTION_ROWS = 784
PANEL_SEED = 20260905
REPRESENTATIVE_SEED = 20260906
PANEL_NAMES = (
    "optimization_0", "optimization_1", "optimization_2",
    "train_screen_0", "train_screen_1",
)
OPTIMIZATION_PANELS = PANEL_NAMES[:3]
TRAIN_SCREENS = PANEL_NAMES[3:]
STRATUM_QUOTAS = {
    "safety_consent": 12,
    "technique": 10,
    "equipment_material": 2,
    "resources_general": 15,
}
PANEL_SIZE = sum(STRATUM_QUOTAS.values())
REQUIRED_PAIRED_STRATA = {
    name: len(PANEL_NAMES) * quota for name, quota in STRATUM_QUOTAS.items()
}
EXPECTED_JOINT_COMPONENTS = 326
EXPECTED_PAIRED_UNITS = 205
EXPECTED_CANDIDATE_ONLY_UNITS = 54
EXPECTED_PRODUCTION_ONLY_UNITS = 67
EXPECTED_PAIRED_STRATA = {
    "safety_consent": 62,
    "technique": 54,
    "equipment_material": 13,
    "resources_general": 76,
}


def canonical_sha256(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def normalize_url(value):
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


def row_urls(row):
    scalar = (
        "url", "evidence_url", "canonical_url", "supplied_url",
        "original_ropetopia_url", "title_evidence_url", "url_evidence_url",
        "canonical_resource_url",
    )
    arrays = ("urls", "canonical_urls", "supplied_urls")
    values = {row[key] for key in scalar if row.get(key)}
    values.update(url for key in arrays for url in row.get(key, ()) if url)
    return {normalize_url(value) for value in values}


class DisjointSet:
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


def load_bound_rows():
    expected = {
        CANDIDATE_PATH: CANDIDATE_SHA256,
        CANDIDATE_MANIFEST_PATH: CANDIDATE_MANIFEST_SHA256,
        PRODUCTION_PATH: PRODUCTION_SHA256,
        Path(sampler_v13.__file__).resolve(): SAMPLER_SHA256,
    }
    if any(file_sha256(path) != digest for path, digest in expected.items()):
        raise RuntimeError("v25a frozen data or conflict implementation changed")
    candidate = read_jsonl(CANDIDATE_PATH)
    production = read_jsonl(PRODUCTION_PATH)
    if len(candidate) != CANDIDATE_ROWS or len(production) != PRODUCTION_ROWS:
        raise RuntimeError("v25a frozen row counts changed")
    return candidate, production


def representative(unit_id, side, members, stratum):
    preferred = [item for item in members if item["classified_stratum"] == stratum]
    selected = min(
        preferred or members,
        key=lambda item: canonical_sha256({
            "schema": "eggroll-es-v25a-fixed-side-representative",
            "seed": REPRESENTATIVE_SEED,
            "unit_id": unit_id,
            "side": side,
            "row_sha256": item["row_sha256"],
        }),
    )
    return {
        "row_index": selected["row_index"],
        "row_sha256": selected["row_sha256"],
        "document_sha256": selected["document_sha256"],
        "classified_stratum": selected["classified_stratum"],
        "preferred_dominant_stratum": bool(preferred),
        "reuse": "fixed_for_every_direction_and_both_signs",
    }


def build_joint_units(candidate, production):
    tagged = [
        {"side": "candidate", "row_index": index, "row": row}
        for index, row in enumerate(candidate)
    ] + [
        {"side": "production", "row_index": index, "row": row}
        for index, row in enumerate(production)
    ]
    semantic_ids = sampler_v13.build_semantic_clusters([
        item["row"] for item in tagged
    ])
    disjoint = DisjointSet(len(tagged))
    first = {}

    def connect(identifier, index):
        if identifier in first:
            disjoint.union(index, first[identifier])
        else:
            first[identifier] = index

    for index, (item, semantic_id) in enumerate(zip(tagged, semantic_ids)):
        row = item["row"]
        connect(f"document:{row['document_sha256']}", index)
        for url in row_urls(row):
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
            "schema": "eggroll-es-joint-conflict-unit-v25a",
            "members": member_ids,
        })
        documents = {}
        for side in ("candidate", "production"):
            grouped = defaultdict(list)
            for item in by_side[side]:
                grouped[item["row"]["document_sha256"]].append(item)
            documents[side] = grouped
        common_documents = set(documents["candidate"]) & set(
            documents["production"]
        )
        common_candidate_items = [
            item for document in common_documents
            for item in documents["candidate"][document]
        ]
        stratum_pool = common_candidate_items or by_side["candidate"]
        candidate_counts = Counter(
            sampler_v13.classify_stratum(item["row"])
            for item in stratum_pool
        )
        stratum = max(
            sampler_v13.STRATA,
            key=lambda name: (
                candidate_counts[name], sampler_v13._TIE_PRIORITY[name],
            ),
        )
        eligible_documents = [
            document for document in common_documents
            if any(
                sampler_v13.classify_stratum(item["row"]) == stratum
                for item in documents["candidate"][document]
            )
        ]
        if eligible_documents:
            shared_document = min(
                eligible_documents,
                key=lambda document: canonical_sha256({
                    "schema": "eggroll-es-v25a-shared-document-selection",
                    "seed": REPRESENTATIVE_SEED,
                    "unit_id": unit_id,
                    "document_sha256": document,
                }),
            )
            side_pools = {
                side: documents[side][shared_document]
                for side in ("candidate", "production")
            }
            pairing_anchor = "shared_document"
        else:
            shared_document = None
            side_pools = by_side
            pairing_anchor = "joint_component_cross_side_link"
        side_contract = {}
        for side in ("candidate", "production"):
            side_members = [{
                "row_index": item["row_index"],
                "row_sha256": sampler_v13.row_sha256(item["row"]),
                "document_sha256": item["row"]["document_sha256"],
                "classified_stratum": sampler_v13.classify_stratum(item["row"]),
            } for item in side_pools[side]]
            side_contract[side] = representative(
                unit_id, side, side_members, stratum,
            )
            assignments.extend(
                f"{side}:{row_sha256}\t{unit_id}"
                for row_sha256 in sorted(
                    sampler_v13.row_sha256(item["row"])
                    for item in by_side[side]
                )
            )
        if side_contract["candidate"]["classified_stratum"] != stratum:
            raise RuntimeError("v25a candidate representative missed its stratum")
        paired.append({
            "unit_id": unit_id,
            "stratum": stratum,
            "pairing_anchor": pairing_anchor,
            "shared_document_sha256": shared_document,
            "common_document_count": len(common_documents),
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


def panel_order(name, items):
    return sorted(
        items,
        key=lambda item: canonical_sha256({
            "schema": "eggroll-es-v25a-panel-order",
            "seed": PANEL_SEED,
            "panel": name,
            "unit_id": item["unit_id"],
        }),
    )


def build_manifest():
    candidate, production = load_bound_rows()
    joint = build_joint_units(candidate, production)
    paired = joint.pop("paired_units")
    strata = Counter(unit["stratum"] for unit in paired)
    if (
        joint["joint_component_count"] != EXPECTED_JOINT_COMPONENTS
        or len(paired) != EXPECTED_PAIRED_UNITS
        or joint["candidate_only_unit_count"] != EXPECTED_CANDIDATE_ONLY_UNITS
        or joint["production_only_unit_count"] != EXPECTED_PRODUCTION_ONLY_UNITS
        or dict(strata) != EXPECTED_PAIRED_STRATA
        or any(
            strata[name] < required
            for name, required in REQUIRED_PAIRED_STRATA.items()
        )
    ):
        raise RuntimeError("v25a joint paired capacity changed or is insufficient")
    by_stratum = {
        name: sorted(
            (unit for unit in paired if unit["stratum"] == name),
            key=lambda unit: canonical_sha256({
                "schema": "eggroll-es-v25a-stratum-permutation",
                "seed": PANEL_SEED,
                "stratum": name,
                "unit_id": unit["unit_id"],
            }),
        )
        for name in sampler_v13.STRATA
    }
    panels = []
    selected_ids = []
    for panel_index, panel_name in enumerate(PANEL_NAMES):
        selected = []
        for stratum in sampler_v13.STRATA:
            quota = STRATUM_QUOTAS[stratum]
            start = panel_index * quota
            population = len(by_stratum[stratum])
            for unit in by_stratum[stratum][start:start + quota]:
                selected.append({
                    **unit,
                    "inclusion_probability_per_panel": quota / population,
                    "horvitz_thompson_unit_weight": population / quota,
                })
        selected = panel_order(panel_name, selected)
        if len(selected) != PANEL_SIZE:
            raise RuntimeError("v25a paired panel size changed")
        selected_ids.extend(item["unit_id"] for item in selected)
        panels.append({
            "name": panel_name,
            "role": (
                "optimization" if panel_name in OPTIMIZATION_PANELS
                else "train_only_screen"
            ),
            "rows_per_side_per_direction": PANEL_SIZE,
            "stratum_counts": dict(STRATUM_QUOTAS),
            "ordered_unit_identity_sha256": canonical_sha256(
                [item["unit_id"] for item in selected]
            ),
            "ordered_side_row_identity_sha256": {
                side: canonical_sha256([
                    item["sides"][side]["row_sha256"] for item in selected
                ])
                for side in ("candidate", "production")
            },
            "items": selected,
        })
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("v25a panels are not globally conflict-unit disjoint")
    if len(selected_ids) != len(PANEL_NAMES) * PANEL_SIZE:
        raise RuntimeError("v25a selected paired-unit coverage changed")

    manifest = {
        "schema": "eggroll-es-paired-data-version-panel-manifest-v25a",
        "status": "paired_alpha_zero_frame_only_no_gpu_or_update_authorization",
        "inputs": {
            "candidate": {
                "path": str(CANDIDATE_PATH),
                "rows": CANDIDATE_ROWS,
                "file_sha256": CANDIDATE_SHA256,
                "freeze_manifest_path": str(CANDIDATE_MANIFEST_PATH),
                "freeze_manifest_file_sha256": CANDIDATE_MANIFEST_SHA256,
            },
            "production": {
                "path": str(PRODUCTION_PATH),
                "rows": PRODUCTION_ROWS,
                "file_sha256": PRODUCTION_SHA256,
            },
            "conflict_implementation": {
                "path": str(Path(sampler_v13.__file__).resolve()),
                "file_sha256": SAMPLER_SHA256,
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
            "stratum_assignment": (
                "candidate_side_dominant_stratum_on_common_documents_when_"
                "available_otherwise_on_the_full_paired_component"
            ),
            "pairing_anchor_preference": (
                "same_document_when_available_otherwise_same_joint_conflict_unit"
            ),
        },
        "panel_contract": {
            "panel_seed": PANEL_SEED,
            "representative_seed": REPRESENTATIVE_SEED,
            "panel_names": list(PANEL_NAMES),
            "optimization_panels": list(OPTIMIZATION_PANELS),
            "train_only_screens": list(TRAIN_SCREENS),
            "panel_size": PANEL_SIZE,
            "stratum_quotas": dict(STRATUM_QUOTAS),
            "required_paired_strata": dict(REQUIRED_PAIRED_STRATA),
            "globally_disjoint_joint_conflict_units": True,
            "fixed_side_representative_every_direction_and_sign": True,
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
            "v25a_role": "matched_data_version_compatibility_only",
            "future_role": "separate_full_candidate_hpo_only_if_v25a_passes",
            "common_subset_is_not_full_candidate_training_estimand": True,
            "contains_row_prompt_or_answer_content": False,
            "contains_evaluation_content": False,
        },
    }
    manifest["content_sha256_before_self_field"] = canonical_sha256(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    joint = manifest.get("joint_frame", {})
    if (
        manifest.get("schema")
        != "eggroll-es-paired-data-version-panel-manifest-v25a"
        or manifest.get("content_sha256_before_self_field")
        != canonical_sha256({
            key: value for key, value in manifest.items()
            if key != "content_sha256_before_self_field"
        })
        or joint.get("joint_component_count") != EXPECTED_JOINT_COMPONENTS
        or joint.get("paired_unit_count") != EXPECTED_PAIRED_UNITS
        or joint.get("paired_stratum_counts") != EXPECTED_PAIRED_STRATA
        or joint.get("selected_paired_unit_count") != 195
        or joint.get("reserve_paired_unit_count") != 10
        or [panel.get("name") for panel in manifest.get("panels", [])]
        != list(PANEL_NAMES)
    ):
        raise RuntimeError("v25a paired manifest contract changed")
    items = [item for panel in manifest["panels"] for item in panel["items"]]
    if (
        len(items) != 195
        or len({item["unit_id"] for item in items}) != 195
        or any(set(item["sides"]) != {"candidate", "production"} for item in items)
        or any(
            len(panel["items"]) != PANEL_SIZE
            or panel["stratum_counts"] != STRATUM_QUOTAS
            or panel["ordered_side_row_identity_sha256"] != {
                side: canonical_sha256([
                    item["sides"][side]["row_sha256"] for item in panel["items"]
                ])
                for side in ("candidate", "production")
            }
            for panel in manifest["panels"]
        )
        or any(
            item["sides"]["candidate"]["classified_stratum"] != item["stratum"]
            or not item["sides"]["candidate"]["preferred_dominant_stratum"]
            for item in items
        )
        or any(
            item["pairing_anchor"] == "shared_document"
            and any(
                side["document_sha256"] != item["shared_document_sha256"]
                for side in item["sides"].values()
            )
            for item in items
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
        raise RuntimeError("v25a paired item identity or weighting changed")
    return manifest


def exclusive_write(path, value):
    path = Path(path).resolve()
    if path != OUTPUT_PATH:
        raise ValueError("v25a output path changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("v25a paired manifest already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if Path(args.output).resolve() != OUTPUT_PATH:
        raise ValueError("v25a output path changed")
    manifest = build_manifest()
    if not args.dry_run:
        exclusive_write(args.output, manifest)
    result = {
        "schema": "eggroll-es-paired-panel-build-v25a",
        "output": str(OUTPUT_PATH),
        "content_sha256": manifest["content_sha256_before_self_field"],
        "joint_component_count": manifest["joint_frame"]["joint_component_count"],
        "paired_unit_count": manifest["joint_frame"]["paired_unit_count"],
        "paired_stratum_counts": manifest["joint_frame"]["paired_stratum_counts"],
        "selected_paired_unit_count": manifest["joint_frame"][
            "selected_paired_unit_count"
        ],
        "reserve_paired_unit_count": manifest["joint_frame"][
            "reserve_paired_unit_count"
        ],
        "gpu_launched": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
