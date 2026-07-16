#!/usr/bin/env python3
"""Stage only V61C ranking/sentinel rows and seal document-block identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import lora_es_nested_population_v52 as design_v52
import lora_es_robust_paired_hpo_v61 as preview_math


ROOT = Path(__file__).resolve().parent
PREVIEW = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_paired_block_bootstrap_v61_preview.json"
).resolve()
PREVIEW_FILE_SHA256 = (
    "a9ce060ce81df5b1fbddcc40db572fe56974ea6dfb6ef2e6ebf3e81925a400e2"
)
PREVIEW_CONTENT_SHA256 = (
    "1b25f3c667fc0e9eeddc19f1d20aebc70c2a0127db0c3eafe11c2f19fb35a0f0"
)
OUTPUT_DATASET = (
    ROOT / "experiments/eggroll_es_hpo/datasets/"
    "v61c_paired_null_calibration_rows.jsonl"
).resolve()
OUTPUT_PANEL = (
    ROOT / "experiments/eggroll_es_hpo/datasets/"
    "v61c_paired_null_calibration_panel.json"
).resolve()
EXPECTED_DOCUMENTS = 234
EXPECTED_SELECTION_DOCUMENTS = 166
EXPECTED_HOLDBACK_DOCUMENTS = 68
EXPECTED_UNIT_DOCUMENT_COUNT_HISTOGRAM = {"1": 203, "2": 2, "4": 2, "19": 1}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_sha256(row: dict) -> str:
    return hashlib.sha256(json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _read_self_hashed(path: Path, file_sha: str, content_sha: str) -> dict:
    if file_sha256(path) != file_sha:
        raise RuntimeError(f"v61c input file changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {key: item for key, item in value.items()
               if key != "content_sha256_before_self_field"}
    if (
        value.get("content_sha256_before_self_field") != content_sha
        or preview_math.canonical_sha256_v61(compact) != content_sha
    ):
        raise RuntimeError(f"v61c input content changed: {path}")
    return value


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_inputs_v61c() -> tuple[bytes, dict]:
    preview = _read_self_hashed(
        PREVIEW, PREVIEW_FILE_SHA256, PREVIEW_CONTENT_SHA256,
    )
    membership = _read_self_hashed(
        design_v52.TRAIN_MEMBERSHIP_V52,
        design_v52.MEMBERSHIP_SHA256_V52,
        design_v52.MEMBERSHIP_CONTENT_SHA256_V52,
    )
    if (
        preview.get("status") != "cpu_only_preview_frozen_launch_ineligible"
        or preview.get("gpu_launch_authorized") is not False
        or preview.get("panels", {}).get("status")
        != "cpu_only_preview_launch_ineligible"
        or preview.get("adaptive_design_provenance") != {
            "v61a_baseline_model_outcomes_used_for_train_only_stratification": True,
            "future_candidate_outcomes_used_for_panel_selection": False,
            "protected_or_holdback_outcomes_used": False,
            "train_only_adaptive_design": True,
        }
        or preview.get("panels", {}).get(
            "v61a_baseline_model_outcomes_used_for_train_only_stratification"
        ) is not True
        or preview.get("panels", {}).get(
            "future_candidate_outcomes_used_for_panel_selection"
        ) is not False
        or preview.get("panels", {}).get(
            "protected_or_holdback_outcomes_used"
        ) is not False
        or preview.get("panels", {}).get("train_only_adaptive_design") is not True
        or membership.get("rows") != 448
        or membership.get("conflict_units") != 208
        or file_sha256(design_v52.TRAIN_DATASET_V52)
        != design_v52.DATASET_SHA256_V52
    ):
        raise RuntimeError("v61c preview/train membership contract changed")
    rows = [json.loads(line) for line in design_v52.TRAIN_DATASET_V52.read_text(
        encoding="utf-8"
    ).splitlines() if line]
    members = membership["items"]
    if len(rows) != 448 or len(members) != 448:
        raise RuntimeError("v61c source train coverage changed")
    by_sha = {}
    unit_docs = defaultdict(set)
    doc_units = defaultdict(set)
    for index, (row, member) in enumerate(zip(rows, members, strict=True)):
        sha = row_sha256(row)
        document = row.get("document_sha256")
        unit = member.get("unit_identity_sha256")
        if (
            member.get("row_index") != index or member.get("row_sha256") != sha
            or not isinstance(document, str) or len(document) != 64
            or not isinstance(unit, str) or len(unit) != 64 or sha in by_sha
        ):
            raise RuntimeError("v61c row/document/unit identity changed")
        by_sha[sha] = (row, member)
        unit_docs[unit].add(document); doc_units[document].add(unit)
    if len(doc_units) != EXPECTED_DOCUMENTS or any(len(value) != 1 for value in doc_units.values()):
        raise RuntimeError("v61c document crossed conflict-unit boundary")
    histogram = Counter(len(value) for value in unit_docs.values())
    histogram_json = {str(key): value for key, value in sorted(histogram.items())}
    if histogram_json != EXPECTED_UNIT_DOCUMENT_COUNT_HISTOGRAM:
        raise RuntimeError("v61c unit document-count distribution changed")

    strata_units = {
        item["unit_identity_sha256"]: item
        for item in preview["panels"]["ranking"]
        + preview["panels"]["untouched_holdback"]
        + preview["panels"]["exact_sentinel"]
        + preview["panels"]["unused_reserve"]
    }
    # Recover the original selection-vs-holdback split from the preview roles:
    # ranking, sentinel, and reserve rows whose source role was selection make
    # the 158-unit selection pool; the explicitly untouched 50 are holdback.
    holdback_units = {
        item["unit_identity_sha256"]
        for item in preview["panels"]["untouched_holdback"]
    }
    selection_units = set(unit_docs) - holdback_units
    selection_docs = set().union(*(unit_docs[unit] for unit in selection_units))
    holdback_docs = set().union(*(unit_docs[unit] for unit in holdback_units))
    if (
        len(selection_docs) != EXPECTED_SELECTION_DOCUMENTS
        or len(holdback_docs) != EXPECTED_HOLDBACK_DOCUMENTS
        or selection_docs & holdback_docs
        or selection_docs | holdback_docs != set(doc_units)
        or set(strata_units) != set(unit_docs)
    ):
        raise RuntimeError("v61c selection/holdback document partition changed")

    selected = preview["panels"]["ranking"] + preview["panels"]["exact_sentinel"]
    if len(selected) != 68:
        raise RuntimeError("v61c calibration selection count changed")
    staged_rows = []
    items = []
    selected_docs = set()
    for request_index, item in enumerate(selected):
        row_sha = item["row_sha256"]; unit = item["unit_identity_sha256"]
        if row_sha not in by_sha or unit in holdback_units:
            raise RuntimeError("v61c selected a holdback or unknown row")
        row, member = by_sha[row_sha]
        if member["unit_identity_sha256"] != unit:
            raise RuntimeError("v61c selected row/unit mismatch")
        staged_rows.append(row)
        docs = sorted(unit_docs[unit]); selected_docs.update(docs)
        items.append({
            "request_index": request_index,
            "role": item["role"],
            "role_index": item["role_index"],
            "row_sha256": row_sha,
            "unit_identity_sha256": unit,
            "representative_document_sha256": row["document_sha256"],
            "unit_document_sha256": docs,
            "unit_document_count": len(docs),
            "base_mean_f1": item["base_mean_f1"],
            "base_exact_actor_count": item["base_exact_actor_count"],
            "base_nonzero_actor_count": item["base_nonzero_actor_count"],
        })
    if selected_docs & holdback_docs:
        raise RuntimeError("v61c runtime panel intersects holdback documents")
    dataset_payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in staged_rows
    ).encode("utf-8")
    mapping = sorted({
        (document, next(iter(units)))
        for document, units in doc_units.items()
    })
    panel = {
        "schema": "v61c-paired-null-calibration-panel",
        "status": "sealed_cpu_only_before_v61c_preregistration",
        "explicit_characterization_authorization": (
            "V61A outcome strata may select V61C null-calibration ranking and "
            "exact-sentinel rows only; HPO/selection/promotion remains unauthorized"
        ),
        "adaptive_design_provenance": {
            "v61a_baseline_model_outcomes_used_for_train_only_stratification": True,
            "future_candidate_outcomes_used_for_panel_selection": False,
            "protected_or_holdback_outcomes_used": False,
            "train_only_adaptive_design": True,
        },
        "source": {
            "preview_path": str(PREVIEW),
            "preview_file_sha256": PREVIEW_FILE_SHA256,
            "preview_content_sha256": PREVIEW_CONTENT_SHA256,
            "train_dataset_file_sha256": design_v52.DATASET_SHA256_V52,
            "membership_file_sha256": design_v52.MEMBERSHIP_SHA256_V52,
            "membership_content_sha256": design_v52.MEMBERSHIP_CONTENT_SHA256_V52,
        },
        "staged_dataset": {
            "path": str(OUTPUT_DATASET),
            "file_sha256": hashlib.sha256(dataset_payload).hexdigest(),
            "rows": 68,
        },
        "ranking_units": 64,
        "exact_sentinel_units": 4,
        "items": items,
        "request_order_row_sha256": [item["row_sha256"] for item in items],
        "request_order_sha256": preview_math.canonical_sha256_v61([
            item["row_sha256"] for item in items
        ]),
        "document_block_audit": {
            "source_rows": 448,
            "unique_documents": len(doc_units),
            "conflict_units": len(unit_docs),
            "each_document_maps_to_exactly_one_conflict_unit": True,
            "unit_document_count_histogram": histogram_json,
            "selection_pool_documents": len(selection_docs),
            "holdback_documents": len(holdback_docs),
            "selection_holdback_document_intersection": 0,
            "runtime_selected_holdback_document_intersection": 0,
            "document_unit_mapping_sha256": preview_math.canonical_sha256_v61(mapping),
            "selection_document_set_sha256": preview_math.canonical_sha256_v61(
                sorted(selection_docs)
            ),
            "holdback_document_set_sha256": preview_math.canonical_sha256_v61(
                sorted(holdback_docs)
            ),
            "runtime_selected_document_set_sha256": preview_math.canonical_sha256_v61(
                sorted(selected_docs)
            ),
        },
        "holdback_units_in_runtime_dataset": 0,
        "holdback_documents_in_runtime_dataset": 0,
        "question_answer_or_generation_text_persisted_in_panel": False,
        "train_semantics_opened_only_for_cpu_staging": True,
        "model_or_gpu_accessed": False,
        "protected_semantics_opened": False,
    }
    panel["content_sha256_before_self_field"] = preview_math.canonical_sha256_v61(panel)
    return dataset_payload, panel


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-output", default=str(OUTPUT_DATASET))
    parser.add_argument("--panel-output", default=str(OUTPUT_PANEL))
    args = parser.parse_args(argv)
    dataset = Path(args.dataset_output).resolve(); panel_path = Path(args.panel_output).resolve()
    if dataset.exists() or panel_path.exists():
        raise FileExistsError("v61c staged outputs must both be fresh")
    dataset_payload, panel = build_inputs_v61c()
    _exclusive_write(dataset, dataset_payload)
    _exclusive_write(panel_path, (
        json.dumps(panel, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8"))
    print(json.dumps({
        "dataset": str(dataset), "dataset_file_sha256": file_sha256(dataset),
        "panel": str(panel_path), "panel_file_sha256": file_sha256(panel_path),
        "panel_content_sha256": panel["content_sha256_before_self_field"],
        "runtime_rows": 68, "holdback_units": 0, "holdback_documents": 0,
        "model_or_gpu_accessed": False, "protected_semantics_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
