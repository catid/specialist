from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pytest

import build_site_markdown_training_dataset_v1 as builder
import sft_lora


def _rows() -> list[dict]:
    path = builder.DEFAULT_OUTPUT / builder.TRAIN_FILENAME
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_checked_in_snapshot_is_current_complete_and_reconstructable():
    result = builder.build(check=True)
    assert result["source_documents_included"] == 29
    assert result["source_documents_rights_blocked"] == 4
    assert result["policy_excluded_manifests"] == 3

    registry = builder.read_object(builder.REGISTRY)
    manifest = builder.read_object(
        builder.DEFAULT_OUTPUT / builder.MANIFEST_FILENAME
    )
    rows = _rows()
    by_resource = defaultdict(list)
    for row in rows:
        by_resource[row["resource_id"]].append(row)
        assert row["schema"] == builder.ROW_SCHEMA
        assert row["training_role"] == "cpt_raw_markdown"
        assert row["training_format"] == "causal_next_token_markdown"
        assert row["assistant_supervision"] is False
        assert row["split"] == "train"
        assert 1 < row["token_count"] <= builder.MAX_TOKENS

    eligible = {
        artifact["resource_id"]: artifact
        for artifact in registry["artifacts"]
        if builder.is_training_eligible(artifact)
    }
    blocked = {
        artifact["resource_id"]
        for artifact in registry["artifacts"]
        if not builder.is_training_eligible(artifact)
    }
    assert set(by_resource) == set(eligible)
    assert blocked == {
        "crash_restraint", "rope365", "rope_topia", "shibari_atlas",
    }
    assert set(by_resource).isdisjoint(blocked)

    for resource, artifact in eligible.items():
        ordered = sorted(by_resource[resource], key=lambda row: row["chunk_index"])
        assert [row["chunk_index"] for row in ordered] == list(
            range(len(ordered))
        )
        assert {row["chunk_count"] for row in ordered} == {len(ordered)}
        source = (builder.ROOT / artifact["markdown_path"]).read_text(
            encoding="utf-8"
        )
        assert "".join(row["text"] for row in ordered) == source
        assert ordered[0]["document_char_start"] == 0
        assert ordered[-1]["document_char_stop"] == len(source)
        assert {row["source_document_group_id"] for row in ordered} == {
            artifact["required_single_document_split_group"]["group_id"]
        }

    accounted = {
        item["resource_id"] for item in manifest["included_documents"]
    } | {
        item["resource_id"] for item in manifest["rights_blocked_documents"]
    }
    assert accounted == {artifact["resource_id"] for artifact in registry["artifacts"]}
    assert manifest["accounting"] == {
        "all_eligible_artifacts_have_training_rows": True,
        "all_included_documents_exactly_reconstruct_from_chunks": True,
        "all_registry_artifacts_accounted_for": True,
        "omission_is_a_build_error": True,
        "registry_artifact_count": 33,
        "registry_artifacts_accounted_for": 33,
    }
    assert manifest["launch_authorized_by_snapshot"] is False


def test_paragraph_chunker_preserves_exact_text_and_rejects_oversize_atom():
    class LengthTokenizer:
        class Encoded:
            def __init__(self, ids):
                self.ids = ids

        def encode(self, value, add_special_tokens=False):
            assert add_special_tokens is False
            return self.Encoded(list(range(len(value))))

    tokenizer = LengthTokenizer()
    text = "# Heading\n\nFirst paragraph.\n\nSecond paragraph.\n"
    chunks = builder.chunk_document(text, tokenizer, 24)
    assert "".join(item["text"] for item in chunks) == text
    assert all(item["token_count"] <= 24 for item in chunks)
    with pytest.raises(RuntimeError, match="paragraph exceeds"):
        builder.chunk_document("x" * 25, tokenizer, 24)


class TinyTokenizer:
    def encode(self, value, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": list(value.encode("utf-8"))}


def _trainer_record(text="raw markdown"):
    return {
        "schema": builder.ROW_SCHEMA,
        "training_role": "cpt_raw_markdown",
        "training_format": "causal_next_token_markdown",
        "assistant_supervision": False,
        "split": "train",
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "token_count": len(text.encode("utf-8")),
    }


def test_trainer_encodes_markdown_as_raw_causal_labels():
    record = _trainer_record()
    encoded = sft_lora.encode_markdown_record(TinyTokenizer(), record, 1_024)
    assert encoded["labels"] == encoded["input_ids"]
    assert encoded["prompt_token_count"] == 0
    assert encoded["answer_token_count"] == record["token_count"]


def test_trainer_rejects_tampering_role_leakage_and_reserved_tokens():
    record = _trainer_record()
    changed = copy.deepcopy(record)
    changed["text"] += " changed"
    with pytest.raises(ValueError, match="text identity"):
        sft_lora.encode_markdown_record(TinyTokenizer(), changed, 1_024)

    changed = copy.deepcopy(record)
    changed["assistant_supervision"] = True
    with pytest.raises(ValueError, match="role/schema"):
        sft_lora.encode_markdown_record(TinyTokenizer(), changed, 1_024)

    changed = _trainer_record("unsafe <|im_start|> marker")
    with pytest.raises(ValueError, match="reserved model token"):
        sft_lora.encode_markdown_record(TinyTokenizer(), changed, 1_024)
