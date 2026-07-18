from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

import build_source_group_split_authority_v1 as split


def _synthetic_group(
    *,
    resource_id: str,
    text: str,
    source_identities: set[str] | None = None,
    descendant_fact_ids: list[str] | None = None,
) -> split.SourceGroup:
    path = f"synthetic/{resource_id}.md"
    data = text.encode("utf-8")
    span = split._make_span(
        resource_id=resource_id,
        markdown_path=path,
        data=data,
        start=0,
        end=len(data),
        role="synthetic_source",
        heading_level=None,
        source_identity_sha256s=source_identities or set(),
    )
    return split._make_group(
        origin_kind="synthetic",
        resource_id=resource_id,
        spans=[span],
        data_by_path={path: data},
        source_identity_sha256s=source_identities or set(),
        provenance_mapping="synthetic_fixture",
        descendant_fact_ids=descendant_fact_ids or [],
    )


def test_url_normalization_collapses_transport_spelling_not_page_identity():
    left = (
        "http://ExAmPle.com:80/a/../b/%7euser/?utm_source=x&z=2&a=1#part"
    )
    right = "https://example.com/b/~user?a=1&z=2"
    assert split.normalize_url(left) == right
    assert split.source_url_identity(left) == split.source_url_identity(right)
    assert split.normalize_url("https://example.com/path/") == (
        "https://example.com/path"
    )
    assert split.normalize_url("https://example.com/") == "https://example.com/"


@pytest.mark.parametrize(
    "value",
    ["", "ftp://example.com/file", "https://user:secret@example.com/path"],
)
def test_url_normalization_fails_closed_on_invalid_source_identity(value: str):
    with pytest.raises(RuntimeError):
        split.normalize_url(value)


def test_hierarchical_spans_cover_every_byte_and_descendants_share_parent_group():
    data = (
        b"# Synthetic corpus\n\nmetadata\n\n"
        b"## Parent A\nintro\n"
        b"### Child A1\nalpha\n"
        b"### Child A2\nbeta\n"
        b"## Parent B\ngamma\n"
    )
    path = "synthetic/hierarchy.md"
    segments, gaps = split._segments(data, {2, 3})
    assert Counter(level for _, _, level in segments) == Counter({2: 2, 3: 2})
    spans: list[split.Span] = []
    groups: list[split.SourceGroup] = []
    current_parent: split.Span | None = None
    children: list[split.Span] = []

    def finish() -> None:
        nonlocal current_parent, children
        if current_parent is None:
            return
        groups.append(
            split._make_group(
                origin_kind="synthetic_parent",
                resource_id="hierarchy",
                spans=[current_parent, *children],
                data_by_path={path: data},
                source_identity_sha256s=set(),
                provenance_mapping="synthetic_fixture",
            )
        )
        current_parent = None
        children = []

    for start, end, level in segments:
        if level == 2:
            finish()
            current_parent = split._make_span(
                resource_id="hierarchy",
                markdown_path=path,
                data=data,
                start=start,
                end=end,
                role="parent",
                heading_level=2,
            )
            spans.append(current_parent)
        else:
            assert current_parent is not None
            child = split._make_span(
                resource_id="hierarchy",
                markdown_path=path,
                data=data,
                start=start,
                end=end,
                role="child",
                heading_level=3,
                parent_span_id=current_parent.span_id,
            )
            spans.append(child)
            children.append(child)
    finish()
    metadata = [
        split._make_span(
            resource_id="hierarchy",
            markdown_path=path,
            data=data,
            start=start,
            end=end,
            role="metadata",
            heading_level=None,
        )
        for start, end in gaps
    ]
    spans.extend(metadata)
    groups.append(
        split._make_group(
            origin_kind="metadata",
            resource_id="hierarchy",
            spans=metadata,
            data_by_path={path: data},
            source_identity_sha256s={"synthetic-metadata"},
            provenance_mapping="synthetic_fixture",
        )
    )
    plan = split.DocumentPlan(
        resource_id="hierarchy",
        artifact_id="synthetic-artifact",
        markdown_path=path,
        markdown_sha256=split.sha256_bytes(data),
        byte_length=len(data),
        spans=spans,
        groups=groups,
        excluded_spans=[],
        source_identity_count=0,
        construction="synthetic",
    )
    split._assert_complete_coverage(plan)
    for group in groups:
        owned = {span.span_id for span in group.spans}
        assert all(
            span.parent_span_id is None or span.parent_span_id in owned
            for span in group.spans
        )


def test_coverage_validation_rejects_gap_and_duplicate_ownership():
    data = b"abcdefghij"
    path = "synthetic/gap.md"
    left = split._make_span(
        resource_id="gap",
        markdown_path=path,
        data=data,
        start=0,
        end=4,
        role="left",
        heading_level=None,
    )
    right = split._make_span(
        resource_id="gap",
        markdown_path=path,
        data=data,
        start=5,
        end=10,
        role="right",
        heading_level=None,
    )
    group = split._make_group(
        origin_kind="synthetic",
        resource_id="gap",
        spans=[left, right],
        data_by_path={path: data},
        source_identity_sha256s=set(),
        provenance_mapping="synthetic_fixture",
    )
    plan = split.DocumentPlan(
        resource_id="gap",
        artifact_id="synthetic",
        markdown_path=path,
        markdown_sha256=split.sha256_bytes(data),
        byte_length=len(data),
        spans=[left, right],
        groups=[group],
        excluded_spans=[],
        source_identity_count=0,
        construction="synthetic",
    )
    with pytest.raises(RuntimeError, match="coverage gap or overlap"):
        split._assert_complete_coverage(plan)
    plan.spans = [left]
    plan.groups = [group]
    plan.excluded_spans = [
        split._make_span(
            resource_id="gap",
            markdown_path=path,
            data=data,
            start=4,
            end=10,
            role="excluded",
            heading_level=None,
        )
    ]
    with pytest.raises(RuntimeError, match="span ownership"):
        split._assert_complete_coverage(plan)


def test_exact_and_near_duplicates_form_one_transitive_component_deterministically():
    words = [f"token{index}" for index in range(240)]
    base = " ".join(words)
    near_words = words.copy()
    near_words[120] = "replacement"
    near = " ".join(near_words)
    url_a = split.source_url_identity("http://example.test/topic/?utm_source=x")
    url_b = split.source_url_identity("https://EXAMPLE.test/topic")
    exact_left = _synthetic_group(
        resource_id="exact-left", text=base, source_identities={url_a}
    )
    exact_right = _synthetic_group(
        resource_id="exact-right", text=base, source_identities={url_b}
    )
    near_right = _synthetic_group(resource_id="near-right", text=near)
    groups = [exact_left, exact_right, near_right]
    mapping, components, near_edges = split.build_duplicate_components(groups)
    reversed_mapping, reversed_components, reversed_edges = (
        split.build_duplicate_components(list(reversed(groups)))
    )
    assert mapping == reversed_mapping
    assert components == reversed_components
    assert near_edges == reversed_edges
    assert len(set(mapping.values())) == 1
    assignments = split.assign_groups(groups, mapping)
    assert len(set(assignments.values())) == 1
    split.validate_disjointness(groups, mapping, assignments, near_edges)


def test_leakage_validation_rejects_url_content_and_near_duplicate_cross_split():
    identity = split.source_url_identity("https://example.test/same")
    left = _synthetic_group(
        resource_id="left",
        text="alpha beta gamma delta epsilon " * 20,
        source_identities={identity},
    )
    right = _synthetic_group(
        resource_id="right",
        text="different words still share the source identity " * 20,
        source_identities={identity},
    )
    groups = [left, right]
    fake_components = {left.group_id: "component-left", right.group_id: "component-right"}
    fake_assignments = {left.group_id: "train", right.group_id: "development"}
    with pytest.raises(RuntimeError, match="URL/source identity"):
        split.validate_disjointness(
            groups, fake_components, fake_assignments, near_edges=[]
        )

    exact_right = _synthetic_group(resource_id="exact-right", text="same content")
    exact_left = _synthetic_group(resource_id="exact-left", text="same content")
    groups = [exact_left, exact_right]
    fake_components = {exact_left.group_id: "a", exact_right.group_id: "b"}
    fake_assignments = {exact_left.group_id: "train", exact_right.group_id: "final"}
    with pytest.raises(RuntimeError, match="exact content"):
        split.validate_disjointness(
            groups, fake_components, fake_assignments, near_edges=[]
        )


def test_descendant_inheritance_validation_rejects_orphaned_child():
    data = b"### child\nbody\n"
    path = "synthetic/orphan.md"
    child = split._make_span(
        resource_id="orphan",
        markdown_path=path,
        data=data,
        start=0,
        end=len(data),
        role="child",
        heading_level=3,
        parent_span_id="source-span-v1:" + "0" * 64,
    )
    group = split._make_group(
        origin_kind="synthetic",
        resource_id="orphan",
        spans=[child],
        data_by_path={path: data},
        source_identity_sha256s=set(),
        provenance_mapping="synthetic_fixture",
    )
    mapping = {group.group_id: "component"}
    assignments = {group.group_id: "train"}
    with pytest.raises(RuntimeError, match="descendant span escaped"):
        split.validate_disjointness([group], mapping, assignments, near_edges=[])


def test_manifest_url_inventory_rejects_normalized_duplicate_drift():
    manifest = {
        "coverage": {
            "included_urls": [
                "http://example.test/page/",
                "https://EXAMPLE.test/page?utm_source=duplicate",
            ]
        }
    }
    with pytest.raises(RuntimeError, match="not unique"):
        split._manifest_included_urls("rope365", manifest)


def test_content_addressed_input_receipt_rejects_concurrent_drift(tmp_path):
    source = tmp_path / "safe-public-manifest.json"
    source.write_text('{"version": 1}\n', encoding="utf-8")
    receipts = [
        {
            "path": source.name,
            "file_sha256": split.file_sha256(source),
        }
    ]
    split._assert_receipts_current(receipts, root=tmp_path)
    source.write_text('{"version": 2}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed during build"):
        split._assert_receipts_current(receipts, root=tmp_path)


def test_final_partition_emits_only_counts_and_commitments():
    groups = [
        _synthetic_group(
            resource_id=f"resource-{index}",
            text=f"synthetic source {index} with enough distinct words for identity",
            descendant_fact_ids=[f"fact-{index}"],
        )
        for index in range(3)
    ]
    components = {
        f"component-{index}": [group.group_id]
        for index, group in enumerate(groups)
    }
    mapping = {
        group.group_id: f"component-{index}" for index, group in enumerate(groups)
    }
    assignments = {
        groups[0].group_id: "train",
        groups[1].group_id: "development",
        groups[2].group_id: "final",
    }
    final = split._split_summary(
        "final",
        groups,
        mapping,
        components,
        assignments,
        include_records=False,
    )
    assert final["source_group_count"] == 1
    assert final["v440_descendant_fact_count"] == 1
    assert final["records_redacted"] is True
    assert "records" not in final
    serialized = split.canonical_json_bytes(final)
    assert groups[2].group_id.encode() not in serialized
    assert b"fact-2" not in serialized
    assert b"synthetic/resource-2.md" not in serialized


def test_split_threshold_is_deterministic_and_uses_only_component_identity():
    component = "source-component-v1:" + "a" * 64
    observed = split.split_for_component(component)
    assert observed in {"train", "development", "final"}
    assert {split.split_for_component(component) for _ in range(20)} == {observed}


def test_sealed_check_never_touches_original_mixed_source_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    corpus_markdown = tmp_path / "corpus" / "public.md"
    corpus_manifest = tmp_path / "corpus" / "manifest.json"
    mixed_v440 = tmp_path / "mixed" / "v440.jsonl"
    registry_path = tmp_path / "registry.json"
    v440_manifest_path = tmp_path / "v440_manifest.json"
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_config_path = tmp_path / "tokenizer_config.json"
    builder_path = tmp_path / "builder.py"
    train_site = tmp_path / "derived" / "train_site.jsonl"
    dev_site = tmp_path / "derived" / "dev_site.jsonl"
    train_qa = tmp_path / "derived" / "train_qa.jsonl"
    dev_qa = tmp_path / "derived" / "dev_qa.jsonl"
    output = tmp_path / "authority.json"
    for path, payload in (
        (tokenizer_path, b"synthetic tokenizer\n"),
        (tokenizer_config_path, b"synthetic config\n"),
        (builder_path, b"# synthetic builder\n"),
        (train_site, b'{"split":"train"}\n'),
        (dev_site, b'{"split":"development"}\n'),
        (train_qa, b'{"split":"train"}\n'),
        (dev_qa, b'{"split":"development"}\n'),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    markdown_sha = "1" * 64
    manifest_sha = "2" * 64
    mixed_sha = "3" * 64
    registry = {
        "schema": "site-corpus-registry-v1",
        "source_tree_fingerprint_sha256": "4" * 64,
        "artifacts": [
            {
                "resource_id": "synthetic",
                "artifact_id": "artifact:synthetic",
                "markdown_path": "corpus/public.md",
                "markdown_sha256": markdown_sha,
                "manifest_path": "corpus/manifest.json",
                "manifest_sha256": manifest_sha,
                "byte_length": 10,
            }
        ],
    }
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    registry_sha = split.file_sha256(registry_path)
    v440_manifest = {
        "projection": {
            "path": str(mixed_v440),
            "sha256": mixed_sha,
            "rows": 2,
        }
    }
    v440_manifest_path.write_text(json.dumps(v440_manifest), encoding="utf-8")
    v440_manifest_sha = split.file_sha256(v440_manifest_path)

    def relative(path: Path) -> str:
        return path.relative_to(tmp_path).as_posix()

    projections = {}
    for partition, site_path, qa_path in (
        ("train", train_site, train_qa),
        ("development", dev_site, dev_qa),
    ):
        projections[partition] = {
            "site_spans": {
                "path": relative(site_path),
                "file_sha256": split.file_sha256(site_path),
                "contains_only_partition": partition,
            },
            "v440_qa": {
                "path": relative(qa_path),
                "file_sha256": split.file_sha256(qa_path),
                "contains_only_partition": partition,
            },
        }
    receipts = [
        {"path": relative(registry_path), "file_sha256": registry_sha},
        {"path": relative(corpus_markdown), "file_sha256": markdown_sha},
        {"path": relative(corpus_manifest), "file_sha256": manifest_sha},
        {
            "path": relative(tokenizer_path),
            "file_sha256": split.file_sha256(tokenizer_path),
        },
        {
            "path": relative(tokenizer_config_path),
            "file_sha256": split.file_sha256(tokenizer_config_path),
        },
        {
            "path": relative(v440_manifest_path),
            "file_sha256": v440_manifest_sha,
        },
        {"path": relative(mixed_v440), "file_sha256": mixed_sha},
    ]
    authority = {
        "schema": split.SCHEMA,
        "status": "sealed_source_disjoint_assignment_launch_still_gated",
        "source_registry_binding": {
            "path": relative(registry_path),
            "file_sha256": registry_sha,
            "source_tree_fingerprint_sha256": "4" * 64,
            "artifact_count": 1,
        },
        "tokenizer_binding": {
            "tokenizer_json_path": relative(tokenizer_path),
            "tokenizer_json_sha256": split.file_sha256(tokenizer_path),
            "tokenizer_config_path": relative(tokenizer_config_path),
            "tokenizer_config_sha256": split.file_sha256(tokenizer_config_path),
        },
        "builder_receipt": {
            "path": relative(builder_path),
            "file_sha256": split.file_sha256(builder_path),
        },
        "totals": {"registered_site_artifacts": 1},
        "site_byte_coverage": [
            {
                "resource_id": "synthetic",
                "artifact_id": "artifact:synthetic",
                "markdown_path": "corpus/public.md",
                "markdown_sha256": markdown_sha,
                "registered_byte_length": 10,
                "covered_byte_length": 10,
            }
        ],
        "assignments": {
            "final": {"records_redacted": True, "source_group_count": 1}
        },
        "materialized_train_development_projections": projections,
        "v440_source_projection_summary": {
            "semantic_fields_read_for_tokenization_and_train_development_projection": True,
            "semantic_fields_used_for_split_assignment": False,
            "final_semantic_records_emitted": False,
        },
        "construction_contract": {
            "post_seal_check_contract": {
                "original_site_markdown_opened_hashed_or_statted": False,
                "original_site_manifests_opened_hashed_or_statted": False,
                "mixed_v440_projection_opened_hashed_or_statted": False,
                "reconstruction_after_seal_forbidden": True,
            }
        },
        "invariants": {
            "final_records_emitted": False,
            "protected_holdout_ood_terminal_incident_or_manual_review_sources_opened": False,
            "v440_semantic_fields_read_for_tokenization_and_train_development_projection": True,
            "v440_semantic_fields_used_for_split_assignment": False,
            "v440_final_semantic_records_emitted": False,
        },
        "safe_input_receipts": receipts,
    }
    authority["content_sha256_before_self_field"] = split.canonical_sha256(authority)
    output.write_text(json.dumps(authority), encoding="utf-8")

    forbidden = {corpus_markdown, corpus_manifest, mixed_v440}
    original_hash = split.file_sha256

    def guarded_hash(path: Path) -> str:
        if Path(path) in forbidden:
            raise AssertionError(f"sealed check hashed mixed source: {path}")
        return original_hash(Path(path))

    original_read_bytes = Path.read_bytes
    original_stat = Path.stat

    def guarded_read_bytes(path: Path) -> bytes:
        if path in forbidden:
            raise AssertionError(f"sealed check opened mixed source: {path}")
        return original_read_bytes(path)

    def guarded_stat(path: Path, *args, **kwargs):
        if path in forbidden:
            raise AssertionError(f"sealed check statted mixed source: {path}")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(split, "file_sha256", guarded_hash)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    checked = split.sealed_check(
        output=output,
        root=tmp_path,
        registry_path=registry_path,
        v440_manifest_path=v440_manifest_path,
        builder_path=builder_path,
    )
    assert checked["content_sha256_before_self_field"] == authority[
        "content_sha256_before_self_field"
    ]


def test_reviewed_disclosure_correction_distinguishes_read_assignment_and_emission(
    monkeypatch: pytest.MonkeyPatch,
):
    old = {
        "schema": split.SCHEMA,
        "v440_nonsemantic_identity_summary": {
            "selected_rows": 516,
            "semantic_fields_accessed_or_emitted": False,
        },
        "invariants": {
            "v440_semantic_fields_accessed_or_emitted": False,
        },
        "builder_receipt": {"path": "old.py", "file_sha256": "0" * 64},
    }
    old["content_sha256_before_self_field"] = split.canonical_sha256(old)
    monkeypatch.setattr(
        split,
        "PRE_DISCLOSURE_CORRECTION_AUTHORITY_SHA256",
        old["content_sha256_before_self_field"],
    )
    new_receipt = {"path": "new.py", "file_sha256": "1" * 64}
    corrected = split._correct_v440_semantic_access_disclosure(
        old, builder_receipt=new_receipt
    )
    summary = corrected["v440_source_projection_summary"]
    assert summary[
        "semantic_fields_read_for_tokenization_and_train_development_projection"
    ] is True
    assert summary["semantic_fields_used_for_split_assignment"] is False
    assert summary["final_semantic_records_emitted"] is False
    assert "v440_nonsemantic_identity_summary" not in corrected
    assert corrected["builder_receipt"] == new_receipt
    unsigned = dict(corrected)
    declared = unsigned.pop("content_sha256_before_self_field")
    assert split.canonical_sha256(unsigned) == declared


def test_disclosure_reseal_hashes_only_derived_projections_not_mixed_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "authority.json"
    builder = tmp_path / "builder.py"
    builder.write_text("# corrected builder\n", encoding="utf-8")
    mixed = tmp_path / "mixed-source.jsonl"
    projections = {}
    for partition in ("train", "development"):
        projections[partition] = {}
        for kind in ("site_spans", "v440_qa"):
            path = tmp_path / "derived" / f"{partition}-{kind}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f'{{"split":"{partition}"}}\n', encoding="utf-8")
            projections[partition][kind] = {
                "path": path.relative_to(tmp_path).as_posix(),
                "file_sha256": split.file_sha256(path),
            }
    old = {
        "schema": split.SCHEMA,
        "status": "sealed_source_disjoint_assignment_launch_still_gated",
        "v440_nonsemantic_identity_summary": {
            "semantic_fields_accessed_or_emitted": False,
        },
        "invariants": {"v440_semantic_fields_accessed_or_emitted": False},
        "builder_receipt": {"path": "old.py", "file_sha256": "0" * 64},
        "materialized_train_development_projections": projections,
        "safe_input_receipts": [
            {"path": mixed.name, "file_sha256": "9" * 64}
        ],
    }
    old["content_sha256_before_self_field"] = split.canonical_sha256(old)
    output.write_text(json.dumps(old), encoding="utf-8")
    monkeypatch.setattr(
        split,
        "PRE_DISCLOSURE_CORRECTION_AUTHORITY_SHA256",
        old["content_sha256_before_self_field"],
    )
    original_hash = split.file_sha256

    def guarded_hash(path: Path) -> str:
        if Path(path) == mixed:
            raise AssertionError("metadata reseal touched mixed source")
        return original_hash(Path(path))

    monkeypatch.setattr(split, "file_sha256", guarded_hash)
    corrected = split.reseal_semantic_access_disclosure(
        output=output, root=tmp_path, builder_path=builder
    )
    assert corrected["v440_source_projection_summary"][
        "semantic_fields_used_for_split_assignment"
    ] is False
    assert not mixed.exists()


def test_reconstruction_fails_before_any_source_read_when_authority_is_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sealed = tmp_path / "authority.json"
    sealed.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(split, "OUTPUT", sealed)
    with pytest.raises(RuntimeError, match="reconstruction would reopen"):
        split.construct()
