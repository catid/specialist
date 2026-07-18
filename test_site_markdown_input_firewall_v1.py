from __future__ import annotations

import os
from pathlib import Path

import pytest

import build_site_markdown_training_dataset_v1 as builder


def _artifact(root: Path) -> tuple[dict, Path, Path]:
    directory = root / "data/site_corpora/synthetic_resource"
    directory.mkdir(parents=True)
    markdown = directory / "CORPUS.md"
    manifest = directory / "manifest.json"
    markdown.write_text("# Synthetic\n\nTraining prose.\n", encoding="utf-8")
    manifest.write_text('{"schema":"synthetic-corpus-manifest-v1"}\n', encoding="utf-8")
    artifact = {
        "resource_id": "synthetic_resource",
        "markdown_path": markdown.relative_to(root).as_posix(),
        "markdown_sha256": builder.file_sha256(markdown),
        "manifest_path": manifest.relative_to(root).as_posix(),
        "manifest_sha256": builder.file_sha256(manifest),
    }
    return artifact, markdown, manifest


def test_registered_artifact_binds_markdown_and_manifest_receipts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "synthetic_repo"
    artifact, markdown, manifest = _artifact(root)
    assert builder.validate_registered_artifact_inputs(
        artifact, root=root
    ) == (markdown, manifest)
    manifest.write_text('{"schema":"changed"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="source receipts changed"):
        builder.validate_registered_artifact_inputs(artifact, root=root)


def test_registered_artifact_rejects_repo_escape(tmp_path: Path) -> None:
    root = tmp_path / "synthetic_repo"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("synthetic", encoding="utf-8")
    artifact = {
        "resource_id": "synthetic_resource",
        "markdown_path": "../outside.md",
        "markdown_sha256": builder.file_sha256(outside),
        "manifest_path": "../outside.md",
        "manifest_sha256": builder.file_sha256(outside),
    }
    with pytest.raises(RuntimeError, match="escapes the repository"):
        builder.validate_registered_artifact_inputs(artifact, root=root)


def test_registered_artifact_rejects_symlink_and_hardlink_aliases(
    tmp_path: Path,
) -> None:
    root = tmp_path / "synthetic_repo"
    artifact, markdown, _ = _artifact(root)
    alias = markdown.with_name("alias.md")
    alias.symlink_to(markdown)
    changed = dict(artifact, markdown_path=alias.relative_to(root).as_posix())
    with pytest.raises(RuntimeError, match="symlink alias"):
        builder.validate_registered_artifact_inputs(changed, root=root)

    alias.unlink()
    os.link(markdown, alias)
    with pytest.raises(RuntimeError, match="hard-link alias"):
        builder.validate_registered_artifact_inputs(changed, root=root)


def test_registered_artifact_rejects_forbidden_path_class(
    tmp_path: Path,
) -> None:
    root = tmp_path / "synthetic_repo"
    artifact, markdown, _ = _artifact(root)
    forbidden = markdown.parent / "manual-review" / "CORPUS.md"
    forbidden.parent.mkdir()
    forbidden.write_bytes(markdown.read_bytes())
    changed = dict(
        artifact,
        markdown_path=forbidden.relative_to(root).as_posix(),
    )
    with pytest.raises(RuntimeError, match="forbidden source path class"):
        builder.validate_registered_artifact_inputs(changed, root=root)
