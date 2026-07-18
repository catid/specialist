#!/usr/bin/env python3
"""Build the complete rights-gated Markdown CPT training snapshot.

The source registry is the inventory authority.  Every registered artifact is
accounted for exactly once as either a full-content training document or an
explicit rights/policy exclusion.  Eligible documents are chunked only between
Markdown paragraphs, never packed across source documents, and can be
reconstructed byte-for-byte from their ordered chunk text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parent
REGISTRY = (
    ROOT / "data/site_corpora/registry/site_corpus_registry_v1.json"
).resolve()
TOKENIZER = (ROOT / "models/Qwen3.6-35B-A3B/tokenizer.json").resolve()
DEFAULT_OUTPUT = (
    ROOT / "data/site_corpora/training/site_markdown_cpt_v1"
).resolve()
TRAIN_FILENAME = "train.jsonl"
MANIFEST_FILENAME = "manifest.json"
SCHEMA = "site-markdown-cpt-training-snapshot-v1"
ROW_SCHEMA = "site-markdown-cpt-chunk-v1"
MAX_TOKENS = 1_024
ALLOWED_RIGHTS = frozenset({
    "explicit_open_license",
    "federal_text_public_domain_presumption",
    "public_domain_in_usa_source_with_trademark_and_jurisdiction_limits",
})
SPECIAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
FORBIDDEN_PATH_TOKENS = frozenset({
    "benchmark",
    "benchmarks",
    "dev",
    "development",
    "developments",
    "eval",
    "evaluation",
    "evaluations",
    "final",
    "finals",
    "heldout",
    "holdout",
    "holdouts",
    "incident",
    "incidents",
    "manualreview",
    "manualreviews",
    "ood",
    "protected",
    "terminal",
    "terminals",
})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return bytes_sha256(payload)


def token_ids_sha256(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise RuntimeError("tokenizer emitted a non-integer token ID")
        if token_id < 0 or token_id >= 2**32:
            raise RuntimeError("tokenizer emitted an out-of-range token ID")
        digest.update(token_id.to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def read_object(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _path_tokens(path: Path) -> set[str]:
    values = set()
    for component in path.parts:
        collapsed = re.sub(r"[^a-z0-9]", "", component.casefold())
        if collapsed:
            values.add(collapsed)
        values.update(
            part
            for part in re.split(r"[^a-z0-9]+", component.casefold())
            if part
        )
    return values


def secure_repo_regular(path: Path, role: str, *, root: Path = ROOT) -> Path:
    """Reject escapes and aliases before opening any registered input."""

    root_lexical = Path(os.path.abspath(os.fspath(root)))
    lexical = Path(os.path.abspath(os.fspath(path)))
    if not lexical.is_relative_to(root_lexical):
        raise RuntimeError(f"{role} escapes the repository")
    if _path_tokens(lexical) & FORBIDDEN_PATH_TOKENS:
        raise RuntimeError(f"{role} uses a forbidden source path class")
    current = Path(lexical.anchor)
    metadata = None
    for component in lexical.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise RuntimeError(f"{role} is missing") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"{role} uses a symlink alias")
    if metadata is None or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{role} is not a regular file")
    if metadata.st_nlink != 1:
        raise RuntimeError(f"{role} uses a hard-link alias")
    return lexical


def registered_path(value: Any, role: str, *, root: Path = ROOT) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{role} path is missing")
    declared = Path(value)
    if declared.is_absolute():
        raise RuntimeError(f"{role} path must be repository relative")
    lexical = Path(os.path.abspath(os.fspath(root / declared)))
    return secure_repo_regular(lexical, role, root=root)


def validate_registered_artifact_inputs(
    artifact: dict,
    *,
    root: Path = ROOT,
) -> tuple[Path, Path]:
    resource = artifact.get("resource_id", "unknown")
    markdown_path = registered_path(
        artifact.get("markdown_path"),
        f"registered Markdown for {resource}",
        root=root,
    )
    manifest_path = registered_path(
        artifact.get("manifest_path"),
        f"registered corpus manifest for {resource}",
        root=root,
    )
    if (
        not isinstance(artifact.get("markdown_sha256"), str)
        or file_sha256(markdown_path) != artifact["markdown_sha256"]
        or not isinstance(artifact.get("manifest_sha256"), str)
        or file_sha256(manifest_path) != artifact["manifest_sha256"]
    ):
        raise RuntimeError(f"registered source receipts changed: {resource}")
    return markdown_path, manifest_path


def is_training_eligible(artifact: dict) -> bool:
    rights = artifact.get("rights_basis", {})
    return (
        artifact.get("declared_direct_training_ready") is True
        and rights.get("status") in ALLOWED_RIGHTS
        and str(rights.get("promotion_gate", "")).startswith("ready_")
    )


def paragraph_atoms(text: str) -> list[str]:
    """Split at blank-line boundaries while preserving every character."""
    parts = re.split(r"(\n[ \t]*\n+)", text)
    atoms = []
    for index in range(0, len(parts), 2):
        body = parts[index]
        separator = parts[index + 1] if index + 1 < len(parts) else ""
        atom = body + separator
        if atom:
            atoms.append(atom)
    if "".join(atoms) != text:
        raise RuntimeError("paragraph splitter did not preserve source text")
    return atoms


def chunk_document(text: str, tokenizer: Tokenizer, max_tokens: int) -> list[dict]:
    if not text:
        raise RuntimeError("cannot train on an empty Markdown artifact")
    if max_tokens <= 1:
        raise ValueError("max_tokens must exceed one")
    if any(marker in text for marker in SPECIAL_TOKENS):
        raise RuntimeError("Markdown artifact contains a reserved model token")
    chunks: list[str] = []
    current = ""
    for atom in paragraph_atoms(text):
        atom_ids = tokenizer.encode(atom, add_special_tokens=False).ids
        if len(atom_ids) > max_tokens:
            raise RuntimeError(
                "a Markdown paragraph exceeds the token cap; manual paragraph "
                "editing is required instead of mid-paragraph truncation"
            )
        candidate = current + atom
        candidate_ids = tokenizer.encode(
            candidate, add_special_tokens=False
        ).ids
        if current and len(candidate_ids) > max_tokens:
            chunks.append(current)
            current = atom
        else:
            current = candidate
    if current:
        chunks.append(current)
    if not chunks or "".join(chunks) != text:
        raise RuntimeError("Markdown chunks do not exactly reconstruct the source")
    result = []
    char_start = 0
    for chunk_index, chunk in enumerate(chunks):
        ids = tokenizer.encode(chunk, add_special_tokens=False).ids
        if not ids or len(ids) > max_tokens:
            raise RuntimeError("Markdown chunk has invalid token geometry")
        char_stop = char_start + len(chunk)
        result.append({
            "chunk_index": chunk_index,
            "document_char_start": char_start,
            "document_char_stop": char_stop,
            "text": chunk,
            "text_sha256": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
            "token_count": len(ids),
            "token_ids_sha256": token_ids_sha256(ids),
        })
        char_start = char_stop
    if char_start != len(text):
        raise RuntimeError("Markdown character coverage is incomplete")
    return result


def _row(artifact: dict, chunk: dict, chunk_count: int) -> dict:
    rights = artifact["rights_basis"]
    group = artifact["required_single_document_split_group"]
    value = {
        "schema": ROW_SCHEMA,
        "training_role": "cpt_raw_markdown",
        "training_format": "causal_next_token_markdown",
        "assistant_supervision": False,
        "split": "train",
        "resource_id": artifact["resource_id"],
        "artifact_id": artifact["artifact_id"],
        "source_label": artifact["source_label"],
        "source_document_identity_sha256": artifact[
            "source_document_identity_sha256"
        ],
        "source_document_group_id": group["group_id"],
        "markdown_path": artifact["markdown_path"],
        "markdown_sha256": artifact["markdown_sha256"],
        "manifest_path": artifact["manifest_path"],
        "manifest_sha256": artifact["manifest_sha256"],
        "rights_status": rights["status"],
        "license": rights.get("license"),
        "promotion_gate": rights["promotion_gate"],
        "attribution_required": rights.get("attribution_required", False),
        "safety_transfer_flags": artifact.get("safety_transfer_flags", []),
        "chunk_index": chunk["chunk_index"],
        "chunk_count": chunk_count,
        "document_char_start": chunk["document_char_start"],
        "document_char_stop": chunk["document_char_stop"],
        "text": chunk["text"],
        "text_sha256": chunk["text_sha256"],
        "token_count": chunk["token_count"],
        "token_ids_sha256": chunk["token_ids_sha256"],
    }
    value["chunk_id"] = canonical_sha256({
        "schema": ROW_SCHEMA,
        "source_document_identity_sha256": value[
            "source_document_identity_sha256"
        ],
        "markdown_sha256": value["markdown_sha256"],
        "chunk_index": value["chunk_index"],
        "text_sha256": value["text_sha256"],
    })
    return value


def render_snapshot(
    registry_path: Path = REGISTRY,
    tokenizer_path: Path = TOKENIZER,
    max_tokens: int = MAX_TOKENS,
) -> tuple[bytes, bytes, dict]:
    registry_path = secure_repo_regular(
        Path(os.path.abspath(os.fspath(registry_path))),
        "site corpus registry",
    )
    tokenizer_path = secure_repo_regular(
        Path(os.path.abspath(os.fspath(tokenizer_path))),
        "training tokenizer",
    )
    registry = read_object(registry_path)
    if registry.get("schema") != "site-corpus-registry-v1":
        raise RuntimeError("unsupported site corpus registry schema")
    tokenizer_sha = file_sha256(tokenizer_path)
    if (
        registry.get("tokenizer", {}).get("tokenizer_json_sha256")
        != tokenizer_sha
    ):
        raise RuntimeError("registry and training tokenizer identities differ")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    artifacts = registry.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError("site corpus registry has no artifacts")

    rows = []
    included = []
    blocked = []
    seen_resources = set()
    seen_artifacts = set()
    seen_documents = set()
    seen_groups = set()
    for artifact in artifacts:
        resource = artifact.get("resource_id")
        if not isinstance(resource, str) or not resource or resource in seen_resources:
            raise RuntimeError("registry resource IDs must be unique nonempty strings")
        artifact_id = artifact.get("artifact_id")
        document_id = artifact.get("source_document_identity_sha256")
        group = artifact.get("required_single_document_split_group")
        group_id = group.get("group_id") if isinstance(group, dict) else None
        if (
            not isinstance(artifact_id, str)
            or not artifact_id
            or artifact_id in seen_artifacts
            or not isinstance(document_id, str)
            or not document_id
            or document_id in seen_documents
            or not isinstance(group_id, str)
            or not group_id
            or group_id in seen_groups
        ):
            raise RuntimeError(
                "registry artifact, document, and split-group identities must be unique"
            )
        seen_resources.add(resource)
        seen_artifacts.add(artifact_id)
        seen_documents.add(document_id)
        seen_groups.add(group_id)
        if not is_training_eligible(artifact):
            blocked.append({
                "resource_id": resource,
                "artifact_id": artifact_id,
                "source_document_identity_sha256": document_id,
                "source_document_group_id": group_id,
                "manifest_path": artifact.get("manifest_path"),
                "markdown_path": artifact.get("markdown_path"),
                "available_tokens": artifact.get("qwen36_token_count"),
                "rights_status": artifact.get("rights_basis", {}).get("status"),
                "rights_basis": artifact.get("rights_basis"),
                "promotion_gate": artifact.get("rights_basis", {}).get(
                    "promotion_gate"
                ),
                "safety_transfer_flags": artifact.get(
                    "safety_transfer_flags", []
                ),
                "reason": "rights_promotion_gate_not_ready",
            })
            continue
        markdown_path, _ = validate_registered_artifact_inputs(artifact)
        text = markdown_path.read_text(encoding="utf-8")
        full_ids = tokenizer.encode(text, add_special_tokens=False).ids
        if len(full_ids) != artifact["qwen36_token_count"]:
            raise RuntimeError(f"registered Markdown token count changed: {resource}")
        chunks = chunk_document(text, tokenizer, max_tokens)
        document_rows = [_row(artifact, chunk, len(chunks)) for chunk in chunks]
        if "".join(row["text"] for row in document_rows) != text:
            raise RuntimeError(f"training chunks omitted source text: {resource}")
        rows.extend(document_rows)
        included.append({
            "resource_id": resource,
            "artifact_id": artifact["artifact_id"],
            "markdown_path": artifact["markdown_path"],
            "markdown_sha256": artifact["markdown_sha256"],
            "source_document_identity_sha256": artifact[
                "source_document_identity_sha256"
            ],
            "source_document_group_id": artifact[
                "required_single_document_split_group"
            ]["group_id"],
            "source_tokens": len(full_ids),
            "emitted_chunk_tokens": sum(row["token_count"] for row in document_rows),
            "chunks": len(document_rows),
            "characters": len(text),
            "complete_character_coverage": True,
            "exact_ordered_reconstruction": True,
            "rights_status": artifact["rights_basis"]["status"],
            "promotion_gate": artifact["rights_basis"]["promotion_gate"],
        })

    policy_blocked = []
    for item in registry.get("excluded_nontraining_manifests", []):
        policy_blocked.append({
            "resource_id": item["resource_id"],
            "manifest_path": item["manifest_path"],
            "reason": item["reason"],
        })
    accounted = (
        {item["resource_id"] for item in included}
        | {item["resource_id"] for item in blocked}
    )
    if accounted != seen_resources or len(accounted) != len(artifacts):
        raise RuntimeError("not every registered Markdown artifact was accounted for")
    if len(rows) != len({row["chunk_id"] for row in rows}):
        raise RuntimeError("training chunk identities are not unique")
    source_counts = Counter(row["resource_id"] for row in rows)
    if set(source_counts) != {item["resource_id"] for item in included}:
        raise RuntimeError("an eligible source has no emitted training chunk")

    train_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        .encode("utf-8")
        for row in rows
    )
    manifest = {
        "schema": SCHEMA,
        "status": "complete_training_input_pending_source_disjoint_launch_gate",
        "training_role": "causal_language_modeling",
        "training_format": "causal_next_token_markdown",
        "assistant_supervision": False,
        "split_assignment": "all_included_source_document_groups_train",
        "cross_document_packing": False,
        "paragraph_boundary_chunking": True,
        "full_eligible_document_content_retained": True,
        "launch_authorized_by_snapshot": False,
        "launch_gate": (
            "fresh source-disjoint evaluation-contract extension is required"
        ),
        "registry_path": os.path.relpath(registry_path, ROOT),
        "registry_file_sha256": file_sha256(registry_path),
        "tokenizer_path": os.path.relpath(tokenizer_path, ROOT),
        "tokenizer_file_sha256": tokenizer_sha,
        "max_tokens_per_chunk": max_tokens,
        "train_jsonl": TRAIN_FILENAME,
        "train_jsonl_sha256": bytes_sha256(train_bytes),
        "train_rows": len(rows),
        "source_documents_included": len(included),
        "source_documents_rights_blocked": len(blocked),
        "policy_excluded_manifests": len(policy_blocked),
        "source_tokens_included": sum(item["source_tokens"] for item in included),
        "emitted_chunk_tokens": sum(item["emitted_chunk_tokens"] for item in included),
        "included_documents": included,
        "rights_blocked_documents": blocked,
        "policy_excluded_documents": policy_blocked,
        "accounting": {
            "registry_artifact_count": len(artifacts),
            "registry_artifacts_accounted_for": len(accounted),
            "all_registry_artifacts_accounted_for": True,
            "all_eligible_artifacts_have_training_rows": True,
            "all_included_documents_exactly_reconstruct_from_chunks": True,
            "omission_is_a_build_error": True,
        },
        "builder": {
            "path": os.path.relpath(Path(__file__).resolve(), ROOT),
            "file_sha256": file_sha256(Path(__file__).resolve()),
        },
    }
    manifest["content_sha256_before_self_field"] = canonical_sha256(manifest)
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return train_bytes, manifest_bytes, manifest


def atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def build(output: Path = DEFAULT_OUTPUT, check: bool = False) -> dict:
    output = Path(output).resolve()
    train_bytes, manifest_bytes, manifest = render_snapshot()
    paths = {
        output / TRAIN_FILENAME: train_bytes,
        output / MANIFEST_FILENAME: manifest_bytes,
    }
    if check:
        for path, expected in paths.items():
            if not path.is_file() or path.read_bytes() != expected:
                raise RuntimeError(f"Markdown training snapshot drift: {path}")
    else:
        for path, value in paths.items():
            atomic_write(path, value)
    return {
        "output": str(output),
        "checked": check,
        "train_rows": manifest["train_rows"],
        "source_documents_included": manifest["source_documents_included"],
        "source_documents_rights_blocked": manifest[
            "source_documents_rights_blocked"
        ],
        "policy_excluded_manifests": manifest["policy_excluded_manifests"],
        "source_tokens_included": manifest["source_tokens_included"],
        "train_jsonl_sha256": manifest["train_jsonl_sha256"],
        "manifest_content_sha256": manifest[
            "content_sha256_before_self_field"
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)
    print(json.dumps(build(arguments.output, arguments.check), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
