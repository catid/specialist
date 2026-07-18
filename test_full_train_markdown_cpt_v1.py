from __future__ import annotations

import re
from pathlib import Path

import pytest

import build_full_train_markdown_cpt_v1 as full
import build_high_information_domain_corpus_v1 as corpus


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        assert add_special_tokens is False
        return list(range(len(re.findall(r"\S+", text))))


def source_row(index: int = 1) -> dict:
    text = f"# Synthetic {index}\n\nComplete source text {index}.\n"
    return {
        "schema": corpus.SITE_SCHEMA,
        "split": "train",
        "training_role": "cpt_raw_markdown_source_span",
        "training_format": "causal_next_token_markdown",
        "assistant_supervision": False,
        "cross_document_packing_performed": False,
        "text": text,
        "text_sha256": corpus.sha256_bytes(text.encode()),
        "qwen36_token_count": len(text.split()),
        "projection_record_id": f"projection-{index}",
        "source_group_id": f"group-{index}",
        "duplicate_component_id": f"component-{index}",
        "source_document_identity_sha256": f"{index:064x}",
        "markdown_sha256": f"{index + 10:064x}",
        "resource_id": f"resource-{index}",
        "artifact_id": f"artifact-{index}",
        "span_id": f"span-{index}",
        "parent_span_id": None,
        "role": "included_page_h3",
        "byte_start": index * 100,
        "byte_end": index * 100 + len(text.encode()),
        "rights_basis": {"status": "synthetic"},
        "safety_transfer_flags": ["synthetic-scope"],
        "source_identity_sha256s": [f"{index + 20:064x}"],
        "lineage": {"markdown_path": "/must/not/open.md"},
        "provenance_mapping": {"source_url": "https://not-opened.invalid"},
        "markdown_path": "/must/not/open.md",
    }


def test_full_row_preserves_every_token_and_opaque_lineage_without_paths():
    source = source_row()
    row = full.transform_row(source, FakeTokenizer())
    assert row["text"] == source["text"]
    assert row["qwen36_token_count"] == source["qwen36_token_count"]
    assert row["source_group_id"] == source["source_group_id"]
    assert row["rights_basis"] == source["rights_basis"]
    assert row["safety_transfer_flags"] == source["safety_transfer_flags"]
    assert row["cross_document_packing_performed"] is False
    assert row["content_filtering_performed"] is False
    assert "markdown_path" not in row
    assert "/must/not/open.md" not in repr(row)
    assert "not-opened.invalid" not in repr(row)


@pytest.mark.parametrize(
    "mutation",
    [
        {"split": "development"},
        {"training_role": "assistant_answer"},
        {"cross_document_packing_performed": True},
        {"text": "<|im_start|> forbidden"},
    ],
)
def test_full_row_fails_closed_on_boundary_or_reserved_token(mutation: dict):
    source = source_row()
    source.update(mutation)
    if "text" in mutation:
        source["text_sha256"] = corpus.sha256_bytes(source["text"].encode())
        source["qwen36_token_count"] = len(source["text"].split())
    with pytest.raises(RuntimeError, match="contract changed"):
        full.transform_row(source, FakeTokenizer())


def test_full_build_rows_keeps_duplicates_and_all_source_groups(
    monkeypatch: pytest.MonkeyPatch,
):
    sources = [source_row(1), source_row(2)]
    sources[1]["text"] = sources[0]["text"]
    sources[1]["text_sha256"] = sources[0]["text_sha256"]
    sources[1]["qwen36_token_count"] = sources[0]["qwen36_token_count"]
    monkeypatch.setattr(full.corpus, "load_site_rows", lambda tokenizer: sources)
    monkeypatch.setattr(full, "EXPECTED_ROWS", 2)
    monkeypatch.setattr(
        full,
        "EXPECTED_TOKENS",
        sum(item["qwen36_token_count"] for item in sources),
    )
    rows = full.build_rows(FakeTokenizer())
    assert len(rows) == 2
    assert len({row["source_group_id"] for row in rows}) == 2
    assert rows[0]["text"] == rows[1]["text"]


def test_full_check_mode_is_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    artifact = tmp_path / "train.jsonl"
    manifest_path = tmp_path / "manifest.json"
    manifest = {"schema": "synthetic"}
    artifact.write_bytes(b"synthetic\n")
    manifest_path.write_bytes(full._json_payload(manifest))
    monkeypatch.setattr(full, "MANIFEST", manifest_path)
    monkeypatch.setattr(
        full,
        "construct",
        lambda: (manifest, {}, {artifact: b"synthetic\n"}),
    )
    monkeypatch.setattr(
        full.corpus,
        "atomic_write",
        lambda *_: pytest.fail("check mode attempted a write"),
    )
    assert full.build(check=True) == manifest


def test_sampling_schedules_seal_equal_update_and_full_coverage_math():
    schedules = full.build_sampling_schedules(full_raw_tokens=970_455)

    equal = schedules["equal_update_1m"]
    assert equal["total_token_exposures"] == 1_000_000
    assert equal["replay_fraction"] == {
        "numerator": 150_000,
        "denominator": 1_000_000,
        "decimal": 0.15,
    }
    assert equal["raw_arms"]["protocol_core_100k"]["token_exposures"] == 100_000
    assert equal["raw_arms"]["full_pool_sample_100k"] == {
        "token_exposures": 100_000,
        "sampling": (
            "deterministic without-replacement sample from the sealed "
            "970455-token full pool; sample artifact must be "
            "content-addressed before launch"
        ),
        "sample_materialized": False,
    }

    fixed = schedules["one_pass_full_fixed_replay"]
    assert fixed["total_token_exposures"] == 1_870_455
    assert fixed["replay_fraction"]["numerator"] == 150_000
    assert fixed["replay_fraction"]["denominator"] == 1_870_455
    assert fixed["replay_fraction"]["decimal"] == pytest.approx(
        0.08019439120427917
    )

    balanced = schedules["one_pass_full_replay_balanced"]
    assert balanced["replay_token_exposures"] == 303_610
    assert balanced["repeated_replay_token_exposures"] == 153_610
    assert balanced["total_token_exposures"] == 2_024_065
    assert balanced["replay_pool_mean_exposure"] == pytest.approx(
        2.0240666666666667
    )
    assert balanced["realized_replay_fraction"] == {
        "numerator": 303_610,
        "denominator": 2_024_065,
        "decimal": pytest.approx(0.15000012351381997),
    }
    assert balanced["replay_sampling_with_replacement"] is True


def test_sampling_schedules_reject_nonpositive_or_noninteger_counts():
    for value in (0, -1, 1.5):
        with pytest.raises(ValueError, match="positive integers"):
            full.build_sampling_schedules(full_raw_tokens=value)
