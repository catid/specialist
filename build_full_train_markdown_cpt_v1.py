#!/usr/bin/env python3
"""Materialize every sealed train-partition Markdown token for CPT ablation.

Unlike the protocol-core 100k high-information subset, this arm performs no
content filtering or deduplication: it preserves all 970,455 Qwen tokens in the
sealed train site projection.  It never opens mixed-source lineage paths and
never reads development/final/protected data.  Each row remains within one
source document and one source group.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import build_high_information_domain_corpus_v1 as corpus


OUTPUT_DIR = (
    corpus.ROOT / "data/training_inventory/full_train_markdown_cpt_v1"
).resolve()
DATASET = (OUTPUT_DIR / "train.jsonl").resolve()
MANIFEST = (OUTPUT_DIR / "manifest.json").resolve()
ABLATION = (OUTPUT_DIR / "core100k_vs_full970455_ablation.json").resolve()
EXPECTED_ROWS = 1_554
EXPECTED_TOKENS = 970_455
RESERVED_MODEL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")


def transform_row(source: dict, tokenizer: Any) -> dict:
    text = source.get("text")
    if (
        source.get("schema") != corpus.SITE_SCHEMA
        or source.get("split") != "train"
        or source.get("training_role") != "cpt_raw_markdown_source_span"
        or source.get("training_format") != "causal_next_token_markdown"
        or source.get("assistant_supervision") is not False
        or source.get("cross_document_packing_performed") is not False
        or not isinstance(text, str)
        or not text
        or any(token in text for token in RESERVED_MODEL_TOKENS)
        or corpus.sha256_bytes(text.encode("utf-8")) != source.get("text_sha256")
        or corpus.token_count(tokenizer, text) != source.get("qwen36_token_count")
    ):
        raise RuntimeError("sealed train Markdown row contract changed")
    identity = {
        "sealed_projection_record_id": source["projection_record_id"],
        "source_group_id": source["source_group_id"],
        "text_sha256": source["text_sha256"],
    }
    return {
        "schema": "full-train-markdown-cpt-row-v1",
        "record_id": corpus.content_id("full-train-markdown-cpt-v1", identity),
        "split": "train",
        "training_role": "cpt_raw_markdown",
        "training_format": "causal_next_token_markdown",
        "assistant_supervision": False,
        "text": text,
        "text_sha256": source["text_sha256"],
        "qwen36_token_count": source["qwen36_token_count"],
        "source_group_id": source["source_group_id"],
        "duplicate_component_id": source["duplicate_component_id"],
        "source_document_identity_sha256": source[
            "source_document_identity_sha256"
        ],
        "markdown_sha256": source["markdown_sha256"],
        "resource_id": source["resource_id"],
        "artifact_id": source["artifact_id"],
        "span_id": source["span_id"],
        "parent_span_id": source["parent_span_id"],
        "role": source["role"],
        "byte_start": source["byte_start"],
        "byte_end": source["byte_end"],
        "rights_basis": source["rights_basis"],
        "safety_transfer_flags": source["safety_transfer_flags"],
        "lineage": {
            "sealed_projection_record_id": source["projection_record_id"],
            "source_identity_sha256s": source["source_identity_sha256s"],
            "source_lineage_commitment_sha256": corpus.canonical_sha256(
                source.get("lineage")
            ),
            "provenance_mapping_commitment_sha256": corpus.canonical_sha256(
                source.get("provenance_mapping")
            ),
            "mixed_source_paths_dereferenced": False,
        },
        "cross_document_packing_performed": False,
        "content_filtering_performed": False,
        "content_deduplication_performed": False,
        "url_memorization_qa_created": False,
    }


def build_rows(tokenizer: Any) -> list[dict]:
    source_rows = corpus.load_site_rows(tokenizer)
    rows = [transform_row(source, tokenizer) for source in source_rows]
    rows.sort(
        key=lambda row: (
            row["source_document_identity_sha256"],
            row["byte_start"],
            row["span_id"],
        )
    )
    if (
        len(rows) != EXPECTED_ROWS
        or len({row["record_id"] for row in rows}) != len(rows)
        or sum(row["qwen36_token_count"] for row in rows) != EXPECTED_TOKENS
        or {row["lineage"]["sealed_projection_record_id"] for row in rows}
        != {source["projection_record_id"] for source in source_rows}
    ):
        raise RuntimeError("full train Markdown coverage changed")
    return rows


def _json_payload(value: dict) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def build_sampling_schedules(
    *,
    full_raw_tokens: int,
    qa_tokens: int = 750_000,
    replay_unique_tokens: int = 150_000,
    equal_update_raw_tokens: int = 100_000,
) -> dict:
    """Return the three sealed token-exposure schedules for the raw-data ablation.

    Token exposure, rather than unique-token count, is explicit throughout.  In
    particular, preserving a 15% replay share while traversing the full raw pool
    requires sampling the fixed replay pool with replacement.
    """
    values = {
        "full_raw_tokens": full_raw_tokens,
        "qa_tokens": qa_tokens,
        "replay_unique_tokens": replay_unique_tokens,
        "equal_update_raw_tokens": equal_update_raw_tokens,
    }
    if any(not isinstance(value, int) or value <= 0 for value in values.values()):
        raise ValueError("sampling-schedule token counts must be positive integers")

    equal_update_total = (
        equal_update_raw_tokens + qa_tokens + replay_unique_tokens
    )
    one_pass_nonreplay = full_raw_tokens + qa_tokens
    one_pass_fixed_replay_total = one_pass_nonreplay + replay_unique_tokens

    # Solve r / (one_pass_nonreplay + r) ~= 15 / 100, then round r to
    # the nearest integer token exposure (half-up; the denominator is odd).
    replay_balanced_exposure = (one_pass_nonreplay * 15 + 42) // 85
    replay_balanced_total = one_pass_nonreplay + replay_balanced_exposure

    return {
        "equal_update_1m": {
            "purpose": "isolate 100k raw-selection quality at fixed update tokens",
            "total_token_exposures": equal_update_total,
            "qa_assistant_token_exposures": qa_tokens,
            "replay_token_exposures": replay_unique_tokens,
            "replay_fraction": {
                "numerator": replay_unique_tokens,
                "denominator": equal_update_total,
                "decimal": replay_unique_tokens / equal_update_total,
            },
            "raw_arms": {
                "protocol_core_100k": {
                    "token_exposures": equal_update_raw_tokens,
                    "sampling": "all materialized curated core tokens once",
                },
                "full_pool_sample_100k": {
                    "token_exposures": equal_update_raw_tokens,
                    "sampling": (
                        "deterministic without-replacement sample from the sealed "
                        f"{full_raw_tokens}-token full pool; sample artifact must be "
                        "content-addressed before launch"
                    ),
                    "sample_materialized": False,
                },
            },
        },
        "one_pass_full_fixed_replay": {
            "purpose": "measure full raw coverage with fixed unique QA/replay exposure",
            "raw_token_exposures": full_raw_tokens,
            "qa_assistant_token_exposures": qa_tokens,
            "replay_token_exposures": replay_unique_tokens,
            "total_token_exposures": one_pass_fixed_replay_total,
            "replay_fraction": {
                "numerator": replay_unique_tokens,
                "denominator": one_pass_fixed_replay_total,
                "decimal": replay_unique_tokens / one_pass_fixed_replay_total,
            },
            "full_raw_passes": 1.0,
            "replay_pool_mean_exposure": 1.0,
        },
        "one_pass_full_replay_balanced": {
            "purpose": "hold replay near 15% while retaining one full raw pass",
            "raw_token_exposures": full_raw_tokens,
            "qa_assistant_token_exposures": qa_tokens,
            "replay_unique_pool_tokens": replay_unique_tokens,
            "replay_token_exposures": replay_balanced_exposure,
            "repeated_replay_token_exposures": (
                replay_balanced_exposure - replay_unique_tokens
            ),
            "replay_pool_mean_exposure": (
                replay_balanced_exposure / replay_unique_tokens
            ),
            "total_token_exposures": replay_balanced_total,
            "target_replay_fraction": 0.15,
            "realized_replay_fraction": {
                "numerator": replay_balanced_exposure,
                "denominator": replay_balanced_total,
                "decimal": replay_balanced_exposure / replay_balanced_total,
            },
            "integer_rounding_policy": (
                "nearest replay-token exposure; exact 15% is arithmetically "
                f"impossible with fixed {one_pass_nonreplay} non-replay tokens"
            ),
            "full_raw_passes": 1.0,
            "replay_sampling_with_replacement": True,
        },
    }


def construct() -> tuple[dict, dict, dict[Path, bytes]]:
    tokenizer = corpus.load_tokenizer()
    rows = build_rows(tokenizer)
    dataset_payload = corpus.jsonl_payload(rows)
    source_groups = {row["source_group_id"] for row in rows}
    documents = {row["source_document_identity_sha256"] for row in rows}
    resources = {row["resource_id"] for row in rows}
    core_manifest = json.loads(corpus.MANIFEST.read_text(encoding="utf-8"))
    core_raw = core_manifest["raw_continuation"]
    ablation = {
        "schema": "raw-markdown-coverage-ablation-v1",
        "status": "materialized_not_launched",
        "fixed_variables": [
            "base_checkpoint",
            "source_disjoint_train_authority",
            "MLP-only expert-aware LoRA topology",
            "optimizer",
            "learning-rate schedule",
            "replay source pool identity",
            "seed",
            "assistant-supervision data",
        ],
        "arms": {
            "protocol_core_100k": {
                "dataset_path": core_raw["receipt"]["path"],
                "dataset_sha256": core_raw["receipt"]["file_sha256"],
                "qwen36_tokens": core_raw["qwen36_tokens"],
                "source_groups": core_raw["source_groups"],
                "selection": "high-information source-balanced exact subset",
            },
            "full_train_markdown_cpt": {
                "dataset_path": corpus.relative(DATASET),
                "dataset_sha256": corpus.sha256_bytes(dataset_payload),
                "qwen36_tokens": EXPECTED_TOKENS,
                "source_groups": len(source_groups),
                "selection": "all sealed train projection rows without filtering",
            },
        },
        "sampling_schedules": build_sampling_schedules(
            full_raw_tokens=EXPECTED_TOKENS
        ),
        "evaluation": {
            "pilot_then_main": True,
            "knowledge_and_regression_gates_unchanged": True,
            "development_or_final_data_used_for_training": False,
            "winner_selected_only_after_source_disjoint_evaluation": True,
        },
        "training_launched": False,
    }
    ablation["content_sha256_before_self_field"] = corpus.canonical_sha256(ablation)
    ablation_payload = _json_payload(ablation)
    manifest = {
        "schema": "full-train-markdown-cpt-manifest-v1",
        "status": "materialized_not_training_authorization",
        "sealed_train_input": {
            "path": corpus.relative(corpus.SITE_INPUT),
            "file_sha256": corpus.file_sha256(corpus.SITE_INPUT),
            "rows": EXPECTED_ROWS,
            "qwen36_tokens": EXPECTED_TOKENS,
        },
        "dataset": {
            "path": corpus.relative(DATASET),
            "file_sha256": corpus.sha256_bytes(dataset_payload),
            "file_bytes": len(dataset_payload),
            "rows": len(rows),
            "qwen36_tokens": sum(row["qwen36_token_count"] for row in rows),
            "source_groups": len(source_groups),
            "source_documents": len(documents),
            "resources": len(resources),
        },
        "coverage": {
            "all_sealed_train_projection_rows_represented": True,
            "all_sealed_train_tokens_represented": True,
            "all_source_groups_represented": True,
            "complete_document_and_group_lineage_preserved": True,
            "rights_and_safety_metadata_preserved": True,
            "content_filtering_performed": False,
            "URL_provenance_bytes_preserved_as_raw_markdown": True,
            "URL_memorization_QA_created": False,
        },
        "boundary": {
            "only_exact_sealed_train_projection_opened": True,
            "development_final_protected_holdout_ood_terminal_incident_or_manual_review_opened": False,
            "mixed_source_lineage_paths_dereferenced": False,
            "cross_document_packing_performed": False,
        },
        "tokenizer": core_manifest["tokenizer"],
        "ablation": {
            "path": corpus.relative(ABLATION),
            "file_sha256": corpus.sha256_bytes(ablation_payload),
        },
        "builder": {
            "path": corpus.relative(Path(__file__).resolve()),
            "file_sha256": corpus.file_sha256(Path(__file__).resolve()),
        },
        "training_launch_authorized": False,
    }
    manifest["content_sha256_before_self_field"] = corpus.canonical_sha256(manifest)
    return manifest, ablation, {
        DATASET: dataset_payload,
        ABLATION: ablation_payload,
    }


def build(*, check: bool = False) -> dict:
    manifest, _, payloads = construct()
    manifest_payload = _json_payload(manifest)
    expected = {**payloads, MANIFEST: manifest_payload}
    if check:
        stale = [
            path.as_posix()
            for path, payload in expected.items()
            if not path.exists() or path.read_bytes() != payload
        ]
        if stale:
            raise RuntimeError("full train Markdown artifacts are stale: " + ", ".join(stale))
        return manifest
    for path, payload in payloads.items():
        corpus.atomic_write(path, payload)
    corpus.atomic_write(MANIFEST, manifest_payload)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    manifest = build(check=arguments.check)
    print(
        json.dumps(
            {
                "manifest": corpus.relative(MANIFEST),
                "content_sha256": manifest["content_sha256_before_self_field"],
                "rows": manifest["dataset"]["rows"],
                "qwen36_tokens": manifest["dataset"]["qwen36_tokens"],
                "source_groups": manifest["dataset"]["source_groups"],
                "training_launch_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
