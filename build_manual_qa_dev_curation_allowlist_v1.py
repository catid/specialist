#!/usr/bin/env python3
"""Build the exact four-source manual QA-dev curation allowlist."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "manual_qa_dev_curation_allowlist_v1.json"
)
V2_CONTRACT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "recipe_evaluation_compute_contract_v2.json"
)
QUARANTINE_BOUNDARY = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "quarantine_boundary_registry_v3.json"
)
SCHEMA = "specialist-manual-qa-dev-curation-allowlist-v1"
V2_FILE_SHA256 = (
    "fdb018a73f81ff491246fdf6162910ed5ba27f49159a64dc2a1102bbe4cf3047"
)
V2_CONTENT_SHA256 = (
    "ffe0bd6fdb51f8bf96f4736091156a2c146bde3a66e8ed44ab5e3b21b1016e51"
)
QUARANTINE_BOUNDARY_FILE_SHA256 = (
    "3d8ef097a1419e03f4b735e6f8d30e5a876b0a8e86c4b0f1ac100114cb7daf5d"
)
QUARANTINE_BOUNDARY_CONTENT_SHA256 = (
    "5a8c6dbd3b0c225992bb2cc4ae5c02a5ec7774a175fe1a138c4e78260eed7fc7"
)
DENIED_RELATIVE_PREFIXES = (
    "data/manual_reviews",
    "experiments/sft_controls",
)
DEV_SOURCE_BINDINGS = (
    (
        "data/site_corpora/nps_textile_fiber_aging_appendix_k_2002/CORPUS.md",
        "04949a4d3685ad91b052ba81224a770b39328c2715ebc5fc8284bebf34d33453",
        "47ec28655e0be3f432f8b721e90baf8772625f9740cffd8d65bbb1a305c623f2",
        "503b658f0290b4f040fe9b4202fb77096ee87ca2c603ea19fc23447198ca853a",
    ),
    (
        "data/site_corpora/usfs_rigging_for_trail_work/CORPUS.md",
        "4ba06a9c38b1270ce09de54f9e9d58eff016996ac14fe79d94a4519e1b28210a",
        "5b4344a0bc2ef38896ee460e447249fe0ef731370235a1c730db0e4758526454",
        "a436664ba7a3460e89d058e06dda1ada9d91b9dc9b0fb6d2690984b10528f1ed",
    ),
    (
        "data/site_corpora/usfs_region5_hazard_tree_2022/CORPUS.md",
        "17c8eca67538c64c8f624972ec1965aa7f764180412a3598108a1d00d34eae9e",
        "690060d665705f9d68e796a46a8034e787582ba3c3fa2445bd615ba8cf17d2d7",
        "8b844cc70ae2c5d486dde1e940d350da7e71b740f3b1a539967157a6f848f09e",
    ),
    (
        "data/site_corpora/phm_synthetic_fiber_rope_condition_monitoring_review_2017/CORPUS.md",
        "8b709d07150e9f4d5e9c6bf0bb8f8e8af08c81f316581023f3f468371d5ae27e",
        "98394761ee7672fc871336b1da008483c269a7f7dc222cfa66a1acc852412e65",
        "965278a4a53b4d90703a5955fa238c1f0aea077040f7e3b67d55f187c2e86859",
    ),
)
ALLOWED_OUTPUT_PATHS = (
    "experiments/eggroll_es_hpo/datasets/manual_qa_dev_v2/manual_qa_dev.jsonl",
    "experiments/eggroll_es_hpo/datasets/manual_qa_dev_v2/manual_qa_dev.report.json",
    "experiments/eggroll_es_hpo/datasets/manual_qa_dev_v2/manual_qa_dev.curation.jsonl",
)


def canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(schema: str, value: str) -> str:
    return canonical_sha256({"schema": schema, "value": value})


def _under_denied_prefix(relative: str) -> bool:
    return any(
        relative == prefix or relative.startswith(prefix + "/")
        for prefix in DENIED_RELATIVE_PREFIXES
    )


def _read_bound_json(path: Path, expected_file: str, expected_content: str) -> dict:
    if file_sha256(path) != expected_file:
        raise RuntimeError("bound allowlist input bytes changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != expected_content
        or canonical_sha256(compact) != expected_content
    ):
        raise RuntimeError("bound allowlist input content changed")
    return value


def _validated_dev_sources(contract: dict) -> list[dict]:
    contract_bindings = contract.get("roles", {}).get("dev", {}).get(
        "source_bindings", ()
    )
    expected_by_item = {
        item_id: {
            "corpus_file_sha256": file_hash,
            "opaque_item_identity": item_id,
            "source_path_identity_sha256": path_identity,
        }
        for _relative, file_hash, item_id, path_identity in DEV_SOURCE_BINDINGS
    }
    observed_by_item = {
        item.get("opaque_item_identity"): item
        for item in contract_bindings
        if isinstance(item, dict)
    }
    if set(observed_by_item) != set(expected_by_item):
        raise RuntimeError("V2 dev allocation changed")
    allowed = []
    for relative, file_hash, item_id, path_identity in DEV_SOURCE_BINDINGS:
        if _under_denied_prefix(relative):
            raise RuntimeError("DEV allowlist source falls under denied prefix")
        binding = observed_by_item[item_id]
        if any(binding.get(key) != value for key, value in expected_by_item[item_id].items()):
            raise RuntimeError("V2 dev opaque binding changed")
        path = ROOT / relative
        if path.is_symlink() or file_sha256(path) != file_hash:
            raise RuntimeError("DEV allowlist source bytes changed or are aliased")
        allowed.append({
            "repository_relative_path": relative,
            "file_sha256": file_hash,
            "opaque_item_identity": item_id,
            "source_path_identity_sha256": path_identity,
            "permission": "read_for_manual_multi_item_qa_dev_curation_only",
        })
    return allowed


def build_allowlist() -> dict:
    contract = _read_bound_json(V2_CONTRACT, V2_FILE_SHA256, V2_CONTENT_SHA256)
    boundary = _read_bound_json(
        QUARANTINE_BOUNDARY,
        QUARANTINE_BOUNDARY_FILE_SHA256,
        QUARANTINE_BOUNDARY_CONTENT_SHA256,
    )
    allowed_sources = _validated_dev_sources(contract)
    if any(_under_denied_prefix(path) for path in ALLOWED_OUTPUT_PATHS):
        raise RuntimeError("manual QA-dev output falls under denied prefix")
    terminal = contract["roles"]["protected_terminal"]
    allowlist = {
        "schema": SCHEMA,
        "status": "exact_dev_read_and_exact_safe_output_write_allowlist",
        "created_at_utc": "2026-07-17T00:00:00+00:00",
        "authority": {
            "manual_read_of_exact_dev_sources_authorized": True,
            "manual_multi_item_qa_dev_curation_authorized": True,
            "model_inference_training_or_embedding_authorized": False,
            "qa_hpo_or_quality_promotion_authorized": False,
            "recursive_search_glob_or_directory_scan_authorized": False,
        },
        "allowed_dev_sources": allowed_sources,
        "allowed_dev_source_count": len(allowed_sources),
        "allowed_output_paths": list(ALLOWED_OUTPUT_PATHS),
        "allowed_output_path_count": len(ALLOWED_OUTPUT_PATHS),
        "output_policy": {
            "role": "dev_only_never_model_adaptation",
            "minimum_qa_items_per_source": 4,
            "minimum_total_qa_items": 16,
            "source_document_identity_required_per_item": True,
            "manual_evidence_and_reviewer_fields_required": True,
            "new_successor_contract_validation_required_before_use": True,
        },
        "terminal_boundary": {
            "terminal_source_count": terminal["documents"],
            "terminal_opaque_identity_set_sha256": terminal[
                "selected_identity_set_sha256"
            ],
            "terminal_paths_persisted": False,
            "terminal_file_hashes_persisted": False,
            "terminal_read_or_resolution_authorized": False,
        },
        "denied_boundary": {
            "quarantine_registry_content_sha256": boundary[
                "content_sha256_before_self_field"
            ],
            "denied_prefix_identity_sha256": boundary[
                "prefix_identity_sha256"
            ],
            "denied_exact_path_identity_sha256": boundary[
                "exact_path_identity_sha256"
            ],
            "deny_before_resolution_stat_hash_or_open": True,
        },
        "v2_binding": {
            "contract_file_sha256": V2_FILE_SHA256,
            "contract_content_sha256": V2_CONTENT_SHA256,
            "dev_source_identity_set_sha256": contract["roles"]["dev"][
                "source_identity_set_sha256"
            ],
            "qa_hpo_or_general_quality_promotion_authorized": False,
        },
        "implementation": {
            "builder_path": Path(__file__).name,
            "builder_file_sha256": file_sha256(Path(__file__)),
        },
    }
    allowlist["content_sha256_before_self_field"] = canonical_sha256(allowlist)
    return allowlist


def validate_allowlist(allowlist: dict) -> None:
    compact = {
        key: value for key, value in allowlist.items()
        if key != "content_sha256_before_self_field"
    }
    sources = allowlist.get("allowed_dev_sources", ())
    terminal = allowlist.get("terminal_boundary", {})
    authority = allowlist.get("authority", {})
    outputs = allowlist.get("allowed_output_paths", ())
    if (
        allowlist.get("schema") != SCHEMA
        or allowlist.get("status")
        != "exact_dev_read_and_exact_safe_output_write_allowlist"
        or allowlist.get("content_sha256_before_self_field")
        != canonical_sha256(compact)
        or sources != _validated_dev_sources(
            _read_bound_json(V2_CONTRACT, V2_FILE_SHA256, V2_CONTENT_SHA256)
        )
        or allowlist.get("allowed_dev_source_count") != 4
        or list(outputs) != list(ALLOWED_OUTPUT_PATHS)
        or any(_under_denied_prefix(path) for path in outputs)
        or authority.get("manual_read_of_exact_dev_sources_authorized")
        is not True
        or authority.get("model_inference_training_or_embedding_authorized")
        is not False
        or authority.get("qa_hpo_or_quality_promotion_authorized") is not False
        or authority.get("recursive_search_glob_or_directory_scan_authorized")
        is not False
        or terminal.get("terminal_source_count") != 9
        or terminal.get("terminal_paths_persisted") is not False
        or terminal.get("terminal_file_hashes_persisted") is not False
        or terminal.get("terminal_read_or_resolution_authorized") is not False
        or allowlist.get("denied_boundary", {}).get(
            "denied_prefix_identity_sha256"
        ) != _read_bound_json(
            QUARANTINE_BOUNDARY,
            QUARANTINE_BOUNDARY_FILE_SHA256,
            QUARANTINE_BOUNDARY_CONTENT_SHA256,
        )["prefix_identity_sha256"]
        or allowlist.get("v2_binding", {}).get(
            "qa_hpo_or_general_quality_promotion_authorized"
        ) is not False
        or allowlist.get("implementation", {}).get("builder_file_sha256")
        != file_sha256(Path(__file__))
    ):
        raise RuntimeError("invalid manual QA-dev curation allowlist")


def _write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    value = build_allowlist()
    validate_allowlist(value)
    if args.check:
        persisted = json.loads(args.output.read_text(encoding="utf-8"))
        if persisted != value:
            raise RuntimeError("persisted manual QA-dev allowlist differs")
    else:
        _write_exclusive(args.output, value)
    print(json.dumps({
        "allowed_dev_sources": value["allowed_dev_source_count"],
        "allowed_outputs": value["allowed_output_path_count"],
        "qa_hpo_or_quality_promotion_authorized": False,
        "terminal_paths_persisted": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
