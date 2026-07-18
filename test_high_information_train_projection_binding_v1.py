from __future__ import annotations

import json
from pathlib import Path

import pytest

import build_high_information_domain_corpus_v1 as corpus


def _write_authority(
    root: Path,
    site: Path,
    qa: Path,
    *,
    final_records: bool = False,
) -> Path:
    site_payload = site.read_bytes()
    qa_payload = qa.read_bytes()
    authority = {
        "schema": "specialist-source-group-split-authority-v1",
        "status": "sealed_source_disjoint_assignment_launch_still_gated",
        "assignments": {
            "final": {
                "records_redacted": not final_records,
                **({"records": [{"synthetic": True}]} if final_records else {}),
            }
        },
        "invariants": {"final_records_emitted": final_records},
        "materialized_train_development_projections": {
            "train": {
                "site_spans": {
                    "schema": "site-source-span-projection-v1",
                    "path": site.relative_to(root).as_posix(),
                    "file_sha256": corpus.sha256_bytes(site_payload),
                    "rows": 1,
                    "qwen36_tokens": 3,
                    "source_groups": 1,
                    "contains_only_partition": "train",
                },
                "v440_qa": {
                    "schema": "v440-qa-source-split-projection-v1",
                    "path": qa.relative_to(root).as_posix(),
                    "file_sha256": corpus.sha256_bytes(qa_payload),
                    "rows": 1,
                    "qwen36_tokens": 4,
                    "source_document_groups": 1,
                    "contains_only_partition": "train",
                },
            },
            "development": {
                "site_spans": {
                    "path": "must-not-be-opened/development-site.jsonl",
                    "file_sha256": "a" * 64,
                },
                "v440_qa": {
                    "path": "must-not-be-opened/development-qa.jsonl",
                    "file_sha256": "b" * 64,
                },
            },
        },
    }
    authority["content_sha256_before_self_field"] = corpus.canonical_sha256(
        authority
    )
    path = root / "data/training_inventory/source_group_split_authority_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(authority, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    final_records: bool = False,
) -> tuple[Path, Path, Path]:
    root = tmp_path / "synthetic_repo"
    projection = root / "data/training_inventory/source_group_split_v1"
    projection.mkdir(parents=True)
    site = projection / "train_site_spans.jsonl"
    qa = projection / "train_v440_qa.jsonl"
    site.write_bytes(b'{"synthetic":"site"}\n')
    qa.write_bytes(b'{"synthetic":"qa"}\n')
    authority = _write_authority(
        root,
        site,
        qa,
        final_records=final_records,
    )
    monkeypatch.setattr(corpus, "ROOT", root)
    monkeypatch.setattr(corpus, "SITE_INPUT", site)
    monkeypatch.setattr(corpus, "QA_INPUT", qa)
    monkeypatch.setattr(corpus, "SOURCE_SPLIT_AUTHORITY", authority)
    return site, qa, authority


def test_train_inputs_bind_to_sealed_authority_without_opening_development(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site, qa, _ = _fixture(tmp_path, monkeypatch)
    site_receipt = corpus.sealed_train_projection_receipt(site)
    qa_receipt = corpus.sealed_train_projection_receipt(qa)
    assert site_receipt["contains_only_partition"] == "train"
    assert qa_receipt["contains_only_partition"] == "train"
    assert corpus._sealed_train_projection_bytes(site)[0] == site.read_bytes()
    assert corpus._sealed_train_projection_bytes(qa)[0] == qa.read_bytes()


def test_train_input_byte_drift_fails_against_authority_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site, _, _ = _fixture(tmp_path, monkeypatch)
    site.write_bytes(b'{"synthetic":"changed"}\n')
    with pytest.raises(RuntimeError, match="bytes differ"):
        corpus._sealed_train_projection_bytes(site)


def test_train_input_symlink_alias_is_rejected_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site, _, authority = _fixture(tmp_path, monkeypatch)
    target = site.with_name("synthetic_site_target.jsonl")
    target.write_bytes(site.read_bytes())
    site.unlink()
    site.symlink_to(target)
    monkeypatch.setattr(corpus, "SITE_INPUT", site)
    monkeypatch.setattr(corpus, "SOURCE_SPLIT_AUTHORITY", authority)
    with pytest.raises(RuntimeError, match="symlink alias"):
        corpus.sealed_train_projection_receipt(site)


def test_authority_that_emits_final_records_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site, _, _ = _fixture(tmp_path, monkeypatch, final_records=True)
    with pytest.raises(RuntimeError, match="unsealed or unsafe"):
        corpus.sealed_train_projection_receipt(site)


def test_forbidden_lexical_train_alias_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site, _, authority = _fixture(tmp_path, monkeypatch)
    forbidden = site.parent / "manual-review" / site.name
    forbidden.parent.mkdir()
    forbidden.write_bytes(site.read_bytes())
    monkeypatch.setattr(corpus, "SITE_INPUT", forbidden)
    monkeypatch.setattr(corpus, "SOURCE_SPLIT_AUTHORITY", authority)
    with pytest.raises(RuntimeError, match="forbidden boundary"):
        corpus.sealed_train_projection_receipt(forbidden)


@pytest.mark.parametrize(
    ("projection", "count_key"),
    [("site_spans", "source_groups"), ("v440_qa", "source_document_groups")],
)
def test_boolean_group_count_cannot_masquerade_as_integer_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    projection: str,
    count_key: str,
) -> None:
    site, qa, authority_path = _fixture(tmp_path, monkeypatch)
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["materialized_train_development_projections"]["train"][
        projection
    ][count_key] = True
    authority.pop("content_sha256_before_self_field")
    authority["content_sha256_before_self_field"] = corpus.canonical_sha256(
        authority
    )
    authority_path.write_text(
        json.dumps(authority, sort_keys=True) + "\n", encoding="utf-8"
    )
    selected = site if projection == "site_spans" else qa
    with pytest.raises(RuntimeError, match="receipt changed"):
        corpus.sealed_train_projection_receipt(selected)
