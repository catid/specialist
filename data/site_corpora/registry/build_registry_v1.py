#!/usr/bin/env python3
"""Build the deterministic first-class registry for direct-training Markdown.

The registry is an inventory and split contract. It does not merge, chunk,
sample, or assign any corpus to a train/evaluation split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[3]
REGISTRY_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = REGISTRY_DIR / "source_registry_config_v1.json"
DEFAULT_REGISTRY = REGISTRY_DIR / "site_corpus_registry_v1.json"
DEFAULT_TOKENIZER = ROOT / "models/Qwen3.6-35B-A3B/tokenizer.json"
DEFAULT_TOKENIZER_CONFIG = ROOT / "models/Qwen3.6-35B-A3B/tokenizer_config.json"
WORD_RE = re.compile(r"[\w\u2019'-]+", flags=re.UNICODE)
AUXILIARY_MARKDOWN_BASENAMES = {"README.md", "REPORT.md", "guide_source.md"}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def git_tracked_paths() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "data/site_corpora"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return {
        item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    }


def classify_manifest(manifest: dict[str, Any]) -> str:
    if manifest.get("direct_training_ready") is True:
        return "direct_training"
    if manifest.get("direct_training_ready") is False:
        return "excluded_nontraining"
    if manifest.get("artifact_role") == "canonical_trainable_source_corpus":
        return "direct_training"
    if manifest.get("training_use") == "direct source-corpus training input":
        return "direct_training"
    if manifest.get("schema") == "rope-topia-dense-corpus-manifest-v1":
        return "direct_training"
    corpus = manifest.get("corpus")
    if isinstance(corpus, dict) and str(corpus.get("path", "")).endswith(".md"):
        return "direct_training"
    raise ValueError(
        "manifest has no deterministic direct-training or exclusion declaration"
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def validate_config(config: dict[str, Any], tracked: set[str]) -> list[dict[str, Any]]:
    if config.get("schema") != "site-corpus-registry-config-v1":
        raise ValueError("unexpected registry config schema")
    records = config.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("registry config needs a non-empty records list")

    configured_manifests: list[str] = []
    configured_markdown: list[str] = []
    resource_ids: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("every registry config record must be an object")
        required = {
            "resource_id",
            "source_label",
            "manifest_path",
            "markdown_path",
            "direct_training_declaration",
            "source_document_identity",
            "rights_basis",
            "safety_transfer_flags",
        }
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"config record missing {missing}: {record!r}")
        resource_ids.append(record["resource_id"])
        configured_manifests.append(record["manifest_path"])
        configured_markdown.append(record["markdown_path"])
        for field in ("manifest_path", "markdown_path"):
            path = record[field]
            if path not in tracked:
                raise ValueError(f"configured {field} is not Git tracked: {path}")
        if not record["markdown_path"].endswith(".md"):
            raise ValueError("direct-training artifact must be Markdown")
        if not record["safety_transfer_flags"]:
            raise ValueError("every artifact needs explicit safety/transfer flags")
        rights = record["rights_basis"]
        for field in {
            "status",
            "basis",
            "license",
            "attribution_required",
            "promotion_gate",
            "limitations",
        }:
            if field not in rights:
                raise ValueError(
                    f"rights basis lacks {field}: {record['resource_id']}"
                )
        if not rights["limitations"]:
            raise ValueError("rights limitations must be explicit")

    for label, values in {
        "resource IDs": resource_ids,
        "manifest paths": configured_manifests,
        "Markdown paths": configured_markdown,
    }.items():
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate configured {label}")

    tracked_manifests = sorted(
        path
        for path in tracked
        if path.startswith("data/site_corpora/")
        and path.endswith("/manifest.json")
    )
    direct_manifests: list[str] = []
    for manifest_path in tracked_manifests:
        manifest = load_json(ROOT / manifest_path)
        if classify_manifest(manifest) == "direct_training":
            direct_manifests.append(manifest_path)
    if set(configured_manifests) != set(direct_manifests):
        missing = sorted(set(direct_manifests) - set(configured_manifests))
        extra = sorted(set(configured_manifests) - set(direct_manifests))
        raise ValueError(
            "registry config/direct-manifest mismatch; "
            f"missing={missing}, extra={extra}"
        )

    direct_dirs = {str(Path(path).parent) for path in direct_manifests}
    unclassified_markdown = sorted(
        path
        for path in tracked
        if str(Path(path).parent) in direct_dirs
        and path.endswith(".md")
        and Path(path).name not in AUXILIARY_MARKDOWN_BASENAMES
        and path not in configured_markdown
    )
    if unclassified_markdown:
        raise ValueError(
            "unregistered direct-corpus Markdown candidates: "
            f"{unclassified_markdown}"
        )
    return records


def build_registry(
    config_path: Path = DEFAULT_CONFIG,
    tokenizer_path: Path = DEFAULT_TOKENIZER,
    tokenizer_config_path: Path = DEFAULT_TOKENIZER_CONFIG,
) -> dict[str, Any]:
    tracked = git_tracked_paths()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    records = validate_config(config, tracked)

    if relative(tokenizer_path) not in tracked:
        # Model assets are intentionally not Git-tracked, but their exact hashes
        # are sealed below. The explicit branch documents that distinction.
        pass
    tokenizer_bytes = tokenizer_path.read_bytes()
    tokenizer_config_bytes = tokenizer_config_path.read_bytes()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    artifacts: list[dict[str, Any]] = []
    fingerprint_rows: list[dict[str, str]] = []
    excluded_manifests: list[dict[str, Any]] = []

    for manifest_path in sorted(
        path
        for path in tracked
        if path.startswith("data/site_corpora/")
        and path.endswith("/manifest.json")
    ):
        manifest = load_json(ROOT / manifest_path)
        if classify_manifest(manifest) != "excluded_nontraining":
            continue
        excluded_manifests.append(
            {
                "manifest_path": manifest_path,
                "resource_id": manifest.get("resource_id"),
                "reason": manifest.get("policy_reason", "direct_training_ready=false"),
            }
        )

    for spec in sorted(records, key=lambda item: item["resource_id"]):
        manifest_path = ROOT / spec["manifest_path"]
        markdown_path = ROOT / spec["markdown_path"]
        manifest_bytes = manifest_path.read_bytes()
        markdown_bytes = markdown_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if classify_manifest(manifest) != "direct_training":
            raise ValueError(f"configured nontraining manifest: {manifest_path}")
        manifest_resource_id = manifest.get("resource_id")
        if manifest_resource_id not in {None, spec["resource_id"]}:
            raise ValueError(
                f"resource mismatch: {manifest_resource_id} != {spec['resource_id']}"
            )

        text = markdown_bytes.decode("utf-8")
        identity = spec["source_document_identity"]
        identity_sha = sha256_bytes(canonical_json_bytes(identity))
        markdown_sha = sha256_bytes(markdown_bytes)
        manifest_sha = sha256_bytes(manifest_bytes)
        split_group_id = f"source-document-v1:{identity_sha}"
        qwen_tokens = len(tokenizer.encode(text, add_special_tokens=False).ids)

        manifest_schema = {
            "name": manifest.get("schema"),
            "version": manifest.get("schema_version"),
        }
        artifact = {
            "artifact_id": f"canonical-markdown:{spec['resource_id']}",
            "dataset_layer": "canonical_markdown_source_corpus",
            "resource_id": spec["resource_id"],
            "source_label": spec["source_label"],
            "markdown_path": spec["markdown_path"],
            "markdown_sha256": markdown_sha,
            "byte_length": len(markdown_bytes),
            "unicode_word_count": len(WORD_RE.findall(text)),
            "whitespace_delimited_word_count": len(text.split()),
            "qwen36_token_count": qwen_tokens,
            "manifest_path": spec["manifest_path"],
            "manifest_sha256": manifest_sha,
            "manifest_schema": manifest_schema,
            "direct_training_declaration": spec["direct_training_declaration"],
            "declared_direct_training_ready": True,
            "non_qa": True,
            "source_document_identity": identity,
            "source_document_identity_sha256": identity_sha,
            "rights_basis": spec["rights_basis"],
            "safety_transfer_flags": sorted(spec["safety_transfer_flags"]),
            "required_single_document_split_group": {
                "group_id": split_group_id,
                "required": True,
                "assignment_unit": "source_document",
                "assign_before": "markdown_chunking_or_qa_derivation",
                "members": "the complete Markdown artifact, every chunk, and every derived QA or other descendant",
                "cross_split_reuse_forbidden": True,
            },
        }
        artifacts.append(artifact)
        fingerprint_rows.extend(
            [
                {"path": spec["manifest_path"], "sha256": manifest_sha},
                {"path": spec["markdown_path"], "sha256": markdown_sha},
            ]
        )

    split_groups = {
        artifact["required_single_document_split_group"]["group_id"]
        for artifact in artifacts
    }
    if len(split_groups) != len(artifacts):
        raise ValueError(
            "two direct-training artifacts resolve to one source-document group; "
            "represent them explicitly before building the registry"
        )

    registry = {
        "schema": "site-corpus-registry-v1",
        "schema_version": 1,
        "dataset_layer": {
            "layer_id": "canonical_markdown_source_corpus",
            "first_class_training_layer": True,
            "non_qa": True,
            "merge_performed": False,
            "chunking_performed": False,
            "split_assignment_performed": False,
            "omission_of_registered_artifact_is_fatal": True,
            "split_assignment_unit": "required_single_document_split_group.group_id",
            "all_descendants_inherit_source_document_split": True,
            "source_caps_required_before_snapshot": True,
            "token_volume_alone_must_not_determine_source_weight": True,
        },
        "count_methods": {
            "unicode_word_count": "Python Unicode regex [\\w\\u2019'-]+ over the complete UTF-8 Markdown artifact",
            "whitespace_delimited_word_count": "Python str.split() over the complete UTF-8 Markdown artifact",
            "qwen36_token_count": "tokenizers.Tokenizer from the sealed Qwen3.6-35B-A3B tokenizer.json; encode(add_special_tokens=False) over the complete Markdown artifact",
        },
        "tokenizer": {
            "model_family": "Qwen3.6-35B-A3B",
            "tokenizer_json_path": relative(tokenizer_path),
            "tokenizer_json_sha256": sha256_bytes(tokenizer_bytes),
            "tokenizer_config_path": relative(tokenizer_config_path),
            "tokenizer_config_sha256": sha256_bytes(tokenizer_config_bytes),
            "add_special_tokens": False,
        },
        "config": {
            "path": relative(config_path),
            "sha256": sha256_bytes(config_bytes),
        },
        "source_tree_fingerprint_sha256": sha256_bytes(
            canonical_json_bytes(sorted(fingerprint_rows, key=lambda row: row["path"]))
        ),
        "totals": {
            "artifact_count": len(artifacts),
            "source_document_split_group_count": len(split_groups),
            "byte_length": sum(item["byte_length"] for item in artifacts),
            "unicode_word_count": sum(
                item["unicode_word_count"] for item in artifacts
            ),
            "whitespace_delimited_word_count": sum(
                item["whitespace_delimited_word_count"] for item in artifacts
            ),
            "qwen36_token_count": sum(
                item["qwen36_token_count"] for item in artifacts
            ),
        },
        "excluded_nontraining_manifests": excluded_manifests,
        "artifacts": artifacts,
    }
    return registry


def render_registry(registry: dict[str, Any]) -> str:
    return json.dumps(
        registry,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--tokenizer-config", type=Path, default=DEFAULT_TOKENIZER_CONFIG
    )
    args = parser.parse_args()

    rendered = render_registry(
        build_registry(args.config, args.tokenizer, args.tokenizer_config)
    )
    if args.check:
        if not args.registry.exists():
            print(f"registry missing: {args.registry}", file=sys.stderr)
            return 1
        current = args.registry.read_text(encoding="utf-8")
        if current != rendered:
            print(
                "registry is stale or a direct-training corpus is missing; "
                "regenerate and review site_corpus_registry_v1.json",
                file=sys.stderr,
            )
            return 1
        return 0
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
