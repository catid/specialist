#!/usr/bin/env python3
"""Frozen k=2 document-mean sampler for the V14b train-only diagnostic."""

from __future__ import annotations

import hashlib
import math
from collections import Counter

import eggroll_es_hierarchical_train_sampler_v14 as sampler_v14
import eggroll_es_train_panel_sampler_v13 as sampler_v13


SCHEMA = "eggroll-es-paired-distinct-row-full-frame-v14b"
MATCHED_SCHEMA = "eggroll-es-paired-distinct-row-matched56-v14b"
COMPLEMENT_SCHEMA = "eggroll-es-paired-distinct-row-complement-v14b"
MASTER_SEED = "specialist-s6-paired-distinct-row-v14b-20260714"
DOCUMENTS = 310
SINGLE_ROW_DOCUMENTS = 139
MULTIROW_DOCUMENTS = 171
PROMPTS = 481
ROWS_PER_MULTIROW_DOCUMENT = 2
PANEL_NAMES = (
    "optimization_0", "optimization_1", "optimization_2",
    "train_screen_0", "train_screen_1",
)
PANEL_ROLES = {
    "optimization_0": "optimization",
    "optimization_1": "optimization",
    "optimization_2": "optimization",
    "train_screen_0": "train_only_screen",
    "train_screen_1": "train_only_screen",
}


def _canonical(value):
    return sampler_v13.canonical_sha256(value)


def _key(*parts):
    payload = "\0".join((MASTER_SEED, *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _select_distinct_rows(document, rows):
    """Choose up to two rows by a frozen hash permutation without replacement."""
    ordered = sorted(
        document["row_indices"],
        key=lambda index: _key(
            "within-document-without-replacement",
            document["document_sha256"], sampler_v13.row_sha256(rows[index]),
        ),
    )
    return ordered[:min(ROWS_PER_MULTIROW_DOCUMENT, len(ordered))]


def build_full_frame(rows):
    documents, frame = sampler_v14.build_document_frame(rows)
    documents = sorted(
        documents,
        key=lambda item: _key("full-frame-document-order", item["document_sha256"]),
    )
    items = []
    prompt_cursor = 0
    for document_position, document in enumerate(documents):
        selected = _select_distinct_rows(document, rows)
        selected_rows = []
        for selection_rank, row_index in enumerate(selected):
            row = rows[row_index]
            selected_rows.append({
                "selection_rank": selection_rank,
                "prompt_position": prompt_cursor,
                "row_index": row_index,
                "row_sha256": sampler_v13.row_sha256(row),
                "fact_id": row["fact_id"],
                "source": row.get("source"),
            })
            prompt_cursor += 1
        items.append({
            "document_position": document_position,
            "document_sha256": document["document_sha256"],
            "document_row_count": document["row_count"],
            "selected_row_count": len(selected_rows),
            "stratum": document["stratum"],
            "full_frame_equal_document_weight": 1.0,
            "selected_rows": selected_rows,
        })
    singleton = sum(item["document_row_count"] == 1 for item in items)
    multirow = sum(item["document_row_count"] > 1 for item in items)
    full = {
        "schema": SCHEMA,
        "train_only": True,
        "master_seed": MASTER_SEED,
        "rows_per_multirow_document": ROWS_PER_MULTIROW_DOCUMENT,
        "documents": len(items),
        "single_row_documents": singleton,
        "multirow_documents": multirow,
        "prompts": prompt_cursor,
        "frame_sha256": frame["frame_sha256"],
        "ordered_document_identity_sha256": _canonical([
            item["document_sha256"] for item in items
        ]),
        "ordered_prompt_identity_sha256": _canonical([
            selected["row_sha256"]
            for item in items for selected in item["selected_rows"]
        ]),
        "items": items,
        "reduction": (
            "arithmetic mean of distinct selected row rewards within each "
            "document, followed by an equal-document arithmetic mean"
        ),
        "common_random_numbers": (
            "reuse this exact 481-prompt order for every direction and both signs"
        ),
        "hard_replay_fraction": 0.0,
        "validation_ood_or_heldout_used": False,
    }
    full["content_sha256_before_self_field"] = _canonical(full)
    validate_full_frame(full, rows, recompute=False)
    return full


def validate_full_frame(full, rows, *, recompute=True):
    if not isinstance(full, dict) or set(full) != {
        "schema", "train_only", "master_seed", "rows_per_multirow_document",
        "documents", "single_row_documents", "multirow_documents", "prompts",
        "frame_sha256", "ordered_document_identity_sha256",
        "ordered_prompt_identity_sha256", "items", "reduction",
        "common_random_numbers", "hard_replay_fraction",
        "validation_ood_or_heldout_used", "content_sha256_before_self_field",
    }:
        raise ValueError("v14b full-frame shape changed")
    if full["content_sha256_before_self_field"] != _canonical({
        key: value for key, value in full.items()
        if key != "content_sha256_before_self_field"
    }):
        raise ValueError("v14b full-frame content hash changed")
    if (
        full["schema"] != SCHEMA
        or full["train_only"] is not True
        or full["master_seed"] != MASTER_SEED
        or full["rows_per_multirow_document"] != 2
        or full["documents"] != DOCUMENTS
        or full["single_row_documents"] != SINGLE_ROW_DOCUMENTS
        or full["multirow_documents"] != MULTIROW_DOCUMENTS
        or full["prompts"] != PROMPTS
        or full["hard_replay_fraction"] != 0.0
        or full["validation_ood_or_heldout_used"] is not False
    ):
        raise ValueError("v14b frozen full-frame policy changed")
    source_documents, frame = sampler_v14.build_document_frame(rows)
    source_by_document = {
        item["document_sha256"]: item for item in source_documents
    }
    expected_order = sorted(
        source_documents,
        key=lambda item: _key("full-frame-document-order", item["document_sha256"]),
    )
    items = full["items"]
    if (
        full["frame_sha256"] != frame["frame_sha256"]
        or len(items) != DOCUMENTS
        or len({item["document_sha256"] for item in items}) != DOCUMENTS
        or [item["document_position"] for item in items] != list(range(DOCUMENTS))
        or [item["document_sha256"] for item in items]
        != [item["document_sha256"] for item in expected_order]
        or Counter(item["selected_row_count"] for item in items)
        != Counter({1: SINGLE_ROW_DOCUMENTS, 2: MULTIROW_DOCUMENTS})
    ):
        raise ValueError("v14b document allocation changed")
    prompt_positions = []
    prompt_hashes = []
    for item in items:
        source_document = source_by_document.get(item["document_sha256"])
        selected = item["selected_rows"]
        if (
            source_document is None
            or item["document_row_count"] != source_document["row_count"]
            or item["stratum"] != source_document["stratum"]
            or item["full_frame_equal_document_weight"] != 1.0
            or len(selected) != item["selected_row_count"]
            or len({entry["row_index"] for entry in selected}) != len(selected)
            or len({entry["row_sha256"] for entry in selected}) != len(selected)
            or [entry["selection_rank"] for entry in selected]
            != list(range(len(selected)))
            or [entry["row_index"] for entry in selected]
            != _select_distinct_rows(source_document, rows)
        ):
            raise ValueError("v14b without-replacement selection changed")
        for entry in selected:
            row = rows[entry["row_index"]]
            if (
                entry["row_sha256"] != sampler_v13.row_sha256(row)
                or entry["fact_id"] != row["fact_id"]
                or entry["source"] != row.get("source")
                or item["document_sha256"] != row["document_sha256"]
            ):
                raise ValueError("v14b selected row binding changed")
            prompt_positions.append(entry["prompt_position"])
            prompt_hashes.append(entry["row_sha256"])
    if (
        prompt_positions != list(range(PROMPTS))
        or full["ordered_document_identity_sha256"]
        != _canonical([item["document_sha256"] for item in items])
        or full["ordered_prompt_identity_sha256"] != _canonical(prompt_hashes)
    ):
        raise ValueError("v14b prompt ordering changed")
    if recompute and full != build_full_frame(rows):
        raise ValueError("v14b full frame differs from deterministic rebuild")
    return True


def build_matched_panels(rows, full=None):
    full = build_full_frame(rows) if full is None else full
    validate_full_frame(full, rows)
    by_document = {
        item["document_sha256"]: item for item in full["items"]
    }
    panels = {}
    allocated = set()
    for iteration, name in enumerate(PANEL_NAMES):
        base = sampler_v14.build_iteration_panel(rows, iteration)
        items = []
        for base_item in base["items"]:
            document = by_document[base_item["document_sha256"]]
            items.append({
                "position": len(items),
                "full_frame_document_position": document["document_position"],
                "document_sha256": document["document_sha256"],
                "document_row_count": document["document_row_count"],
                "selected_row_count": document["selected_row_count"],
                "stratum": document["stratum"],
                "equal_document_ht_weight": base_item[
                    "equal_document_ht_weight"
                ],
                "prompt_positions": [
                    selected["prompt_position"]
                    for selected in document["selected_rows"]
                ],
                "row_sha256s": [
                    selected["row_sha256"]
                    for selected in document["selected_rows"]
                ],
            })
        documents = {item["document_sha256"] for item in items}
        if allocated.intersection(documents):
            raise RuntimeError("v14b matched56 document allocations overlap")
        allocated.update(documents)
        panel = {
            "schema": MATCHED_SCHEMA,
            "train_only": True,
            "name": name,
            "role": PANEL_ROLES[name],
            "document_allocation_iteration": iteration,
            "documents": len(items),
            "prompts": sum(item["selected_row_count"] for item in items),
            "base_v14_panel_content_sha256": base[
                "content_sha256_before_self_field"
            ],
            "ordered_document_identity_sha256": _canonical([
                item["document_sha256"] for item in items
            ]),
            "ordered_prompt_identity_sha256": _canonical([
                row_sha256 for item in items for row_sha256 in item["row_sha256s"]
            ]),
            "items": items,
            "validation_ood_or_heldout_used": False,
        }
        panel["content_sha256_before_self_field"] = _canonical(panel)
        panels[name] = panel
    if len(allocated) != 280:
        raise RuntimeError("v14b matched56 document coverage changed")
    validate_matched_panels(panels, rows, full, recompute=False)
    return panels


def validate_matched_panels(panels, rows, full, *, recompute=True):
    validate_full_frame(full, rows)
    if not isinstance(panels, dict) or tuple(panels) != PANEL_NAMES:
        raise ValueError("v14b matched56 panel surface changed")
    by_document = {
        item["document_sha256"]: item for item in full["items"]
    }
    allocated = set()
    for iteration, name in enumerate(PANEL_NAMES):
        panel = panels[name]
        base = sampler_v14.build_iteration_panel(rows, iteration)
        if (
            not isinstance(panel, dict)
            or set(panel) != {
                "schema", "train_only", "name", "role",
                "document_allocation_iteration", "documents", "prompts",
                "base_v14_panel_content_sha256",
                "ordered_document_identity_sha256",
                "ordered_prompt_identity_sha256", "items",
                "validation_ood_or_heldout_used",
                "content_sha256_before_self_field",
            }
            or panel["content_sha256_before_self_field"] != _canonical({
                key: value for key, value in panel.items()
                if key != "content_sha256_before_self_field"
            })
            or panel["schema"] != MATCHED_SCHEMA
            or panel["train_only"] is not True
            or panel["name"] != name
            or panel["role"] != PANEL_ROLES[name]
            or panel["document_allocation_iteration"] != iteration
            or panel["documents"] != 56
            or panel["base_v14_panel_content_sha256"]
            != base["content_sha256_before_self_field"]
            or panel["validation_ood_or_heldout_used"] is not False
        ):
            raise ValueError(f"v14b matched56 {name} contract changed")
        items = panel["items"]
        if (
            len(items) != 56
            or [item["position"] for item in items] != list(range(56))
            or [item["document_sha256"] for item in items]
            != [item["document_sha256"] for item in base["items"]]
            or panel["prompts"]
            != sum(item["selected_row_count"] for item in items)
            or not math.isclose(
                math.fsum(item["equal_document_ht_weight"] for item in items),
                DOCUMENTS, rel_tol=0.0, abs_tol=1e-12,
            )
        ):
            raise ValueError(f"v14b matched56 {name} allocation changed")
        for item, base_item in zip(items, base["items"]):
            document = by_document.get(item["document_sha256"])
            if (
                document is None
                or item["full_frame_document_position"]
                != document["document_position"]
                or item["document_row_count"] != document["document_row_count"]
                or item["selected_row_count"] != document["selected_row_count"]
                or item["stratum"] != document["stratum"]
                or item["equal_document_ht_weight"]
                != base_item["equal_document_ht_weight"]
                or item["prompt_positions"] != [
                    selected["prompt_position"]
                    for selected in document["selected_rows"]
                ]
                or item["row_sha256s"] != [
                    selected["row_sha256"]
                    for selected in document["selected_rows"]
                ]
            ):
                raise ValueError(f"v14b matched56 {name} item changed")
        if (
            panel["ordered_document_identity_sha256"] != _canonical([
                item["document_sha256"] for item in items
            ])
            or panel["ordered_prompt_identity_sha256"] != _canonical([
                row_sha256
                for item in items for row_sha256 in item["row_sha256s"]
            ])
        ):
            raise ValueError(f"v14b matched56 {name} identity changed")
        documents = {item["document_sha256"] for item in items}
        if allocated.intersection(documents):
            raise ValueError("v14b matched56 allocations are not disjoint")
        allocated.update(documents)
    if len(allocated) != 280:
        raise ValueError("v14b matched56 coverage changed")
    if recompute and panels != build_matched_panels(rows, full):
        raise ValueError("v14b matched56 panels differ from deterministic rebuild")
    return True


def build_screen_complements(full, panels):
    prompt_sha256_by_position = {
        selected["prompt_position"]: selected["row_sha256"]
        for document in full["items"] for selected in document["selected_rows"]
    }
    by_name = {}
    for name in PANEL_NAMES[3:]:
        excluded = {
            item["document_sha256"] for item in panels[name]["items"]
        }
        items = [
            {
                "position": position,
                "full_frame_document_position": item["document_position"],
                "document_sha256": item["document_sha256"],
                "selected_row_count": item["selected_row_count"],
                "prompt_positions": [
                    selected["prompt_position"] for selected in item["selected_rows"]
                ],
            }
            for position, item in enumerate(
                item for item in full["items"]
                if item["document_sha256"] not in excluded
            )
        ]
        complement = {
            "schema": COMPLEMENT_SCHEMA,
            "train_only": True,
            "screen": name,
            "documents": len(items),
            "prompts": sum(item["selected_row_count"] for item in items),
            "ordered_document_identity_sha256": _canonical([
                item["document_sha256"] for item in items
            ]),
            "ordered_prompt_identity_sha256": _canonical([
                prompt_sha256_by_position[prompt_position]
                for item in items for prompt_position in item["prompt_positions"]
            ]),
            "items": items,
            "screen_documents_excluded": True,
            "validation_ood_or_heldout_used": False,
        }
        complement["content_sha256_before_self_field"] = _canonical(complement)
        by_name[name] = complement
    validate_screen_complements(by_name, full, panels, recompute=False)
    return by_name


def validate_screen_complements(complements, full, panels, *, recompute=True):
    if not isinstance(complements, dict) or tuple(complements) != PANEL_NAMES[3:]:
        raise ValueError("v14b complement surface changed")
    full_by_document = {
        item["document_sha256"]: item for item in full["items"]
    }
    prompt_sha256_by_position = {
        selected["prompt_position"]: selected["row_sha256"]
        for document in full["items"] for selected in document["selected_rows"]
    }
    full_documents = set(full_by_document)
    for name in PANEL_NAMES[3:]:
        complement = complements[name]
        screen = {
            item["document_sha256"] for item in panels[name]["items"]
        }
        kept = {
            item["document_sha256"] for item in complement.get("items", [])
        }
        if (
            not isinstance(complement, dict)
            or set(complement) != {
                "schema", "train_only", "screen", "documents", "prompts",
                "ordered_document_identity_sha256",
                "ordered_prompt_identity_sha256", "items",
                "screen_documents_excluded", "validation_ood_or_heldout_used",
                "content_sha256_before_self_field",
            }
            or complement["content_sha256_before_self_field"] != _canonical({
                key: value for key, value in complement.items()
                if key != "content_sha256_before_self_field"
            })
            or complement["schema"] != COMPLEMENT_SCHEMA
            or complement["train_only"] is not True
            or complement["screen"] != name
            or complement["documents"] != 254
            or len(kept) != 254
            or not screen.isdisjoint(kept)
            or screen | kept != full_documents
            or complement["screen_documents_excluded"] is not True
            or complement["validation_ood_or_heldout_used"] is not False
        ):
            raise ValueError(f"v14b {name} complement changed")
        for position, item in enumerate(complement["items"]):
            document = full_by_document.get(item["document_sha256"])
            if (
                document is None
                or item["position"] != position
                or item["full_frame_document_position"]
                != document["document_position"]
                or item["selected_row_count"] != document["selected_row_count"]
                or item["prompt_positions"] != [
                    selected["prompt_position"]
                    for selected in document["selected_rows"]
                ]
            ):
                raise ValueError(f"v14b {name} complement item changed")
        if (
            complement["prompts"] != sum(
                item["selected_row_count"] for item in complement["items"]
            )
            or complement["ordered_document_identity_sha256"] != _canonical([
                item["document_sha256"] for item in complement["items"]
            ])
            or complement["ordered_prompt_identity_sha256"] != _canonical([
                prompt_sha256_by_position[prompt_position]
                for item in complement["items"]
                for prompt_position in item["prompt_positions"]
            ])
        ):
            raise ValueError(f"v14b {name} complement identity changed")
    if recompute and complements != build_screen_complements(full, panels):
        raise ValueError("v14b complements differ from deterministic rebuild")
    return True


def document_means(prompt_values, full):
    if len(prompt_values) != PROMPTS:
        raise ValueError("v14b prompt value count changed")
    values = [float(value) for value in prompt_values]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("v14b prompt reward is non-finite")
    return {
        item["document_sha256"]: math.fsum(
            values[selected["prompt_position"]] for selected in item["selected_rows"]
        ) / item["selected_row_count"]
        for item in full["items"]
    }


def full_frame_mean(document_values, full):
    if set(document_values) != {
        item["document_sha256"] for item in full["items"]
    }:
        raise ValueError("v14b document mean coverage changed")
    return math.fsum(document_values.values()) / DOCUMENTS


def matched_panel_mean(document_values, panel):
    weights = [item["equal_document_ht_weight"] for item in panel["items"]]
    if not math.isclose(math.fsum(weights), DOCUMENTS, abs_tol=1e-12):
        raise RuntimeError("v14b matched56 weights changed")
    return math.fsum(
        item["equal_document_ht_weight"]
        * document_values[item["document_sha256"]]
        for item in panel["items"]
    ) / DOCUMENTS


def complement_mean(document_values, complement):
    if complement["documents"] != 254:
        raise RuntimeError("v14b complement size changed")
    return math.fsum(
        document_values[item["document_sha256"]]
        for item in complement["items"]
    ) / 254.0


def common_random_number_schedule(full, direction_ids):
    direction_ids = [str(direction_id) for direction_id in direction_ids]
    if not direction_ids or len(direction_ids) % 4 != 0:
        raise ValueError("v14b schedule requires complete four-direction waves")
    validate = full["ordered_prompt_identity_sha256"]
    result = []
    for start in range(0, len(direction_ids), 4):
        wave = direction_ids[start:start + 4]
        for sign in ("plus", "minus"):
            for engine_index, direction_id in enumerate(wave):
                result.append({
                    "execution_index": len(result),
                    "wave_index": start // 4,
                    "engine_index": engine_index,
                    "direction_id": direction_id,
                    "sign": sign,
                    "ordered_prompt_identity_sha256": validate,
                    "prompt_count": PROMPTS,
                })
    return result
