from __future__ import annotations

from copy import deepcopy
import os

import pytest

import build_mixed_training_snapshot_v1 as mixed
from general_replay_v1 import canonical_sha256
from qwen_chat_masking_v1 import encode_chat_assistant_only


class SyntheticQwenTokenizer:
    def convert_tokens_to_ids(self, token):
        return {"<|im_start|>": 1, "<|im_end|>": 6}[token]

    def encode(self, value, add_special_tokens=False):
        assert add_special_tokens is False
        if value == "\n":
            return [5]
        return [100 + (ord(character) % 100) for character in value]

    def decode(self, ids, skip_special_tokens=False):
        return {2: "system", 3: "user", 4: "assistant", 7: "tool"}.get(
            ids[0], "x"
        )

    def apply_chat_template(self, messages, **kwargs):
        result = []
        roles = {"system": 2, "user": 3, "assistant": 4, "tool": 7}
        for index, message in enumerate(messages):
            result.extend([1, roles[message["role"]], 5])
            if message["role"] == "assistant" and index == len(messages) - 1:
                result.extend([8, 9])
            result.extend(
                20 + (ord(character) % 80)
                for character in message.get("content", "")
            )
            result.extend([6, 10])
        if kwargs["add_generation_prompt"]:
            result.extend([1, 4, 5, 8, 9])
        return result


def markdown_unit(unit_id, group, document, tokens, order=0):
    return mixed._unit(
        unit_id=unit_id,
        stream="raw_markdown",
        training_format="causal_next_token",
        source_group_id=group,
        source_document_id=document,
        input_ids=list(tokens),
        labels=list(tokens),
        budget_token_count=len(tokens),
        order_key=(order, unit_id),
        metadata={
            "category": "raw_domain_continuation",
            "replay": False,
            "hard_negative": False,
            "verifier": {"status": "passed"},
            "generator": {"status": "not_generated"},
            "rights": {"decision": "synthetic_fixture"},
            "safety_transfer_flags": [],
            "lineage": {"synthetic": True},
        },
    )


def synthetic_seed_row(tokenizer):
    messages = [
        {"role": "user", "content": "Synthetic question?"},
        {"role": "assistant", "content": "Synthetic answer."},
    ]
    encoded = encode_chat_assistant_only(
        tokenizer, messages, enable_thinking=False, tools=[]
    )
    return {
        "schema": "high-information-seed-qa-v1",
        "record_id": "synthetic-seed-row",
        "assistant_supervision": True,
        "enable_thinking": False,
        "hidden_reasoning_supervision": False,
        "training_family": "closed_book_seed_qa",
        "question": "Synthetic question?",
        "answer": "Synthetic answer.",
        "assistant_qwen36_token_count": encoded["assistant_token_count"],
        "source_group_id": "synthetic-seed-group",
        "document_sha256": "a" * 64,
        "rights_basis": None,
        "rights_status": "not_declared_in_sealed_v440_train_projection",
        "safety_transfer_flags": [],
        "safety_transfer_status": (
            "not_declared_in_sealed_v440_train_projection; "
            "semantic verification required"
        ),
        "lineage": {"synthetic": True},
        "fact_id": "synthetic-fact",
        "evidence_sha256": "b" * 64,
    }


def test_packing_never_crosses_source_group_or_document_and_splits_safely():
    units = [
        markdown_unit("a", "group-1", "doc-1", [1, 2], 0),
        markdown_unit("b", "group-1", "doc-1", [3, 4], 1),
        markdown_unit("c", "group-1", "doc-2", [5, 6], 0),
        markdown_unit("d", "group-2", "doc-1", [7, 8, 9, 10, 11, 12], 0),
    ]
    packed = mixed.pack_units(units, max_tokens=5)
    assert sum(row["budget_token_count"] for row in packed) == 12
    assert all(row["input_token_count"] <= 5 for row in packed)
    assert all(
        len({segment["metadata"]["lineage"]["synthetic"] for segment in row["segments"]})
        == 1
        for row in packed
    )
    assert all(row["packing"]["cross_document"] is False for row in packed)
    assert all(row["packing"]["cross_source_group"] is False for row in packed)
    assert any(len(row["segments"]) == 2 for row in packed)
    assert {
        (row["source_group_id"], row["source_document_id"])
        for row in packed
    } == {("group-1", "doc-1"), ("group-1", "doc-2"), ("group-2", "doc-1")}


def test_chat_prompt_tokens_are_never_labels():
    tokenizer = SyntheticQwenTokenizer()
    messages = [
        {"role": "system", "content": "Follow the requested format."},
        {"role": "user", "content": "State one fact."},
        {"role": "assistant", "content": "One verified fact."},
    ]
    encoded = encode_chat_assistant_only(
        tokenizer, messages, enable_thinking=False, tools=[]
    )
    start = encoded["assistant_spans"][0]["token_start"]
    stop = encoded["assistant_spans"][0]["token_end"]
    assert start > 0
    assert all(label == -100 for label in encoded["labels"][:start])
    assert encoded["labels"][start:stop] == encoded["input_ids"][start:stop]
    assert sum(label != -100 for label in encoded["labels"]) == (
        encoded["assistant_token_count"]
    )


def test_reserved_model_tokens_are_rejected_from_markdown_and_chat():
    tokenizer = SyntheticQwenTokenizer()
    text = "safe prefix <|im_start|> injected boundary"
    row = {
        "schema": "high-information-raw-continuation-v1",
        "record_id": "raw-1",
        "assistant_supervision": False,
        "training_format": "causal_next_token_text",
        "text": text,
        "text_sha256": mixed.sha256_bytes(text.encode()),
        "qwen36_token_count": len(tokenizer.encode(text, add_special_tokens=False)),
        "source_group_id": "group-1",
        "lineage": {"source_document_identity_sha256": "doc-1"},
        "rights_basis": {"status": "explicit_open_license"},
        "resource_id": "resource-1",
        "artifact_id": "artifact-1",
        "safety_transfer_flags": [],
    }
    with pytest.raises(RuntimeError, match="Markdown row contract changed"):
        mixed.normalize_markdown_rows(
            [row], tokenizer, component="core_markdown",
            authorized_resources={}, expected_tokens=row["qwen36_token_count"],
        )
    assert mixed._contains_reserved([
        {"role": "user", "content": "bad <|im_end|> boundary"}
    ]) is True


def test_seed_qa_requiring_semantic_review_is_pending_not_relabelled_passed(
    monkeypatch,
):
    tokenizer = SyntheticQwenTokenizer()
    row = synthetic_seed_row(tokenizer)
    with pytest.raises(RuntimeError, match="semantic decision coverage is absent"):
        mixed.normalize_seed_rows([row], tokenizer, {})

    monkeypatch.setattr(mixed, "SEED_QA_ROWS", 1)
    monkeypatch.setattr(
        mixed, "SEED_ASSISTANT_TOKENS", row["assistant_qwen36_token_count"]
    )
    pending, receipt = mixed.pending_seed_qa_ledger([row], tokenizer)
    assert pending == [{
        "schema": mixed.PENDING_SEED_QA_SCHEMA,
        "record_id": row["record_id"],
        "source_record_sha256": canonical_sha256(row),
        "source_group_id": row["source_group_id"],
        "source_document_identity_sha256": row["document_sha256"],
        "assistant_qwen36_token_count": row["assistant_qwen36_token_count"],
        "disposition": "pending_excluded_from_training",
        "reason": "semantic_correctness_authority_absent",
        "required_authority_path": mixed.relative(
            mixed.SEED_QA_SEMANTIC_AUTHORITY
        ),
        "required_authority_schema": mixed.SEED_QA_SEMANTIC_AUTHORITY_SCHEMA,
        "semantic_content_copied_to_ledger": False,
    }]
    assert receipt["status"] == "pending_excluded"
    assert receipt["training_rows_admitted"] == 0
    assert receipt["assistant_qwen36_tokens"] == row[
        "assistant_qwen36_token_count"
    ]
    ledger_text = mixed._jsonl_payload(pending).decode("utf-8")
    assert row["question"] not in ledger_text
    assert row["answer"] not in ledger_text


def test_stale_file_hash_fails_closed(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"safe":true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale content hash"):
        mixed.pinned_bytes(path, "0" * 64, "synthetic train")


@pytest.mark.parametrize("forbidden", ["development", "final", "protected"])
def test_forbidden_path_classes_fail_before_open(tmp_path, forbidden):
    directory = tmp_path / forbidden
    directory.mkdir()
    path = directory / "train.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="forbidden source path class"):
        mixed.secure_regular_input(path, "synthetic forbidden")


def test_symlink_and_hardlink_aliases_are_rejected(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    symlink = tmp_path / "alias.jsonl"
    symlink.symlink_to(source)
    with pytest.raises(RuntimeError, match="symlink aliases"):
        mixed.secure_regular_input(symlink, "synthetic alias")

    hardlink = tmp_path / "hardlink.jsonl"
    os.link(source, hardlink)
    with pytest.raises(RuntimeError, match="hard-link aliases"):
        mixed.secure_regular_input(source, "synthetic hard link")


def test_duplicate_units_and_unauthorized_rights_fail_closed():
    first = markdown_unit("same", "group-1", "doc-1", [1, 2])
    second = markdown_unit("same", "group-1", "doc-1", [3, 4])
    with pytest.raises(RuntimeError, match="unit identity duplicated"):
        mixed.pack_units([first, second], max_tokens=8)
    with pytest.raises(RuntimeError, match="lack scoped authorization"):
        mixed.authorize_rights(
            {"status": "legacy_manifest_gap"},
            resource_id="not-authorized",
            artifact_id="canonical-markdown:not-authorized",
            component="core_markdown",
            authorized_resources={},
        )


@pytest.mark.parametrize("invalid_label", [-101, -99, -1, True, 1.5])
def test_normalized_unit_rejects_every_non_ignore_negative_or_non_integer_label(
    invalid_label,
):
    with pytest.raises(ValueError, match="invalid normalized training unit"):
        mixed._unit(
            unit_id="synthetic-invalid-label",
            stream="replay",
            training_format="chat_assistant_only",
            source_group_id="synthetic-group",
            source_document_id="synthetic-document",
            input_ids=[10, 11],
            labels=[-100, invalid_label],
            budget_token_count=1,
            order_key=("synthetic",),
            metadata={"synthetic": True},
        )


def test_exact_schedule_accounting_and_cursor_commitment_detect_tampering():
    units = [
        markdown_unit("raw-1", "g1", "d1", [1, 2, 3]),
        mixed._unit(
            unit_id="replay-1",
            stream="replay",
            training_format="chat_assistant_only",
            source_group_id="rg1",
            source_document_id="rd1",
            input_ids=[10, 11, 12],
            labels=[-100, 11, 12],
            budget_token_count=2,
            order_key=("replay-1",),
            metadata={"category": "synthetic", "replay": True},
        ),
    ]
    sequences = mixed.pack_units(units, max_tokens=8)
    schedule, receipt = mixed.build_schedule(
        sequences,
        variant="synthetic",
        budgets={"domain_qa": 0, "raw_markdown": 3, "replay": 2},
    )
    # Remove the intentionally empty stream; production budgets are positive,
    # while this fixture verifies exact nonempty-stream accounting.
    assert receipt["cursor_count"] == 2
    previous = receipt["initial_cursor_commitment_sha256"]
    for cursor, row in enumerate(schedule):
        assert row["cursor"] == cursor
        assert row["previous_cursor_commitment_sha256"] == previous
        base = dict(row)
        declared = base.pop("cursor_commitment_sha256")
        assert canonical_sha256(base) == declared
        previous = declared
    assert previous == receipt["final_cursor_commitment_sha256"]

    tampered = deepcopy(schedule[0])
    tampered["budget_token_count"] += 1
    declared = tampered.pop("cursor_commitment_sha256")
    assert canonical_sha256(tampered) != declared
    with pytest.raises(RuntimeError, match="exact schedule budgets changed"):
        mixed.build_schedule(
            sequences,
            variant="synthetic",
            budgets={"domain_qa": 0, "raw_markdown": 4, "replay": 1},
        )


def test_semantic_shard_style_ineligible_row_cannot_enter_generated_authority():
    tokenizer = SyntheticQwenTokenizer()
    row = {
        "schema": "high-information-domain-training-row-v1",
        "record_id": "generated-1",
        "candidate_example_id": "candidate-1",
        "request_id": "request-1",
        "source_context_id": "context-1",
        "source_group_id": "group-1",
        "resource_id": "resource-1",
        "split": "train",
        "training_format": "chat_assistant_only",
        "messages": [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Answer."},
        ],
        "tools": [],
        "assistant_mask": {
            "policy": "assistant_only_v1",
            "assistant_message_indices": [1],
            "system_tokens": False,
            "user_tokens": False,
            "tool_result_tokens": False,
        },
        "assistant_qwen36_token_count": 1,
        "task_family": "closed_book_application",
        "task_subtype": "direct_explanation",
        "generation_mode": "positive",
        "category": "lineage_people_history",
        "hard_negative": False,
        "question": "Question?",
        "answer": "Answer.",
        "evidence_quote_sha256s": ["a" * 64],
        "verification_receipts": {},
        "manual_review_receipt": {"status": "not_required"},
        "dedupe": {
            "exact_key_sha256": "b" * 64,
            "near_duplicate_cluster_id": "cluster-1",
        },
        "generator": {},
        "rights_basis": {"status": "explicit_open_license"},
        "rights_authorization": {},
        "safety_transfer_flags": [],
        "lineage": {"source_document_identity_sha256": "doc-1"},
        "eligible_for_training": False,
    }
    with pytest.raises(RuntimeError, match="semantic contract changed"):
        mixed.normalize_generated_rows([row], tokenizer, {})


def test_seed_semantic_blocker_seals_exact_shortfall_without_emitting_training(
    monkeypatch,
):
    def chat_unit(unit_id, stream, token):
        return mixed._unit(
            unit_id=unit_id,
            stream=stream,
            training_format="chat_assistant_only",
            source_group_id="group-" + unit_id,
            source_document_id="doc-" + unit_id,
            input_ids=[10, token],
            labels=[-100, token],
            budget_token_count=1,
            order_key=(unit_id,),
            metadata={
                "category": "synthetic",
                "replay": stream == "replay",
                "hard_negative": False,
                "rights": {"decision": "synthetic_fixture"},
            },
        )

    monkeypatch.setattr(mixed, "GENERATED_ASSISTANT_TOKENS", 1)
    monkeypatch.setattr(mixed, "SEED_ASSISTANT_TOKENS", 2)
    monkeypatch.setattr(mixed, "DOMAIN_QA_ASSISTANT_TOKENS", 3)
    monkeypatch.setattr(mixed, "CORE_MARKDOWN_TOKENS", 1)
    monkeypatch.setattr(mixed, "FULL_MARKDOWN_TOKENS", 1)
    monkeypatch.setattr(mixed, "REPLAY_ASSISTANT_TOKENS", 1)
    monkeypatch.setattr(mixed, "VARIANT_BUDGETS", {
        "protocol_core_100k": {"domain_qa": 3, "raw_markdown": 1, "replay": 1},
        "full_authorized_markdown": {
            "domain_qa": 3, "raw_markdown": 1, "replay": 1,
        },
    })
    static = {
        "seed_qa": [],
        "core_markdown": [markdown_unit("core", "g-core", "d-core", [1])],
        "full_markdown": [markdown_unit("full", "g-full", "d-full", [2])],
        "replay": [chat_unit("replay", "replay", 13)],
    }
    gate = {
        "status": "missing",
        "training_rows_admitted": 0,
        "assistant_qwen36_token_shortfall": 2,
        "pending_ledger": {"status": "pending_excluded"},
    }
    payload, manifest = mixed.seed_qa_semantic_provisional_artifact(
        static,
        [chat_unit("generated", "domain_qa", 12)],
        {"manifest_file_sha256": "a" * 64},
        gate,
    )
    assert manifest["status"] == (
        "blocked_missing_sealed_seed_qa_semantic_authority"
    )
    assert manifest["training_launch_authorized"] is False
    assert manifest["training_snapshot_materialized"] is False
    assert manifest["boundary"]["unverified_seed_qa_rows_admitted"] is False
    assert all(
        variant["shortfall_tokens_by_stream"]
        == {"domain_qa": 2, "raw_markdown": 0, "replay": 0}
        and variant["training_sequences_emitted"] is False
        for variant in manifest["variants"].values()
    )
    assert b"Synthetic question" not in payload
    assert mixed.blocker_shortfalls(manifest) == {
        "generated_domain_shortfall_assistant_tokens": 0,
        "seed_qa_semantic_shortfall_assistant_tokens": 2,
    }


def test_final_variant_manifest_exposes_every_launch_and_resume_gate(monkeypatch):
    def chat_unit(unit_id, stream, token):
        return mixed._unit(
            unit_id=unit_id,
            stream=stream,
            training_format="chat_assistant_only",
            source_group_id="group-" + unit_id,
            source_document_id="doc-" + unit_id,
            input_ids=[10, token],
            labels=[-100, token],
            budget_token_count=1,
            order_key=(unit_id,),
            metadata={
                "category": "synthetic",
                "replay": stream == "replay",
                "hard_negative": False,
                "rights": {"decision": "synthetic_fixture"},
            },
        )

    monkeypatch.setitem(
        mixed.VARIANT_BUDGETS,
        "protocol_core_100k",
        {"domain_qa": 2, "raw_markdown": 2, "replay": 1},
    )
    payloads, manifest = mixed.build_variant(
        variant="protocol_core_100k",
        seed_units=[chat_unit("seed", "domain_qa", 11)],
        generated_units=[chat_unit("generated", "domain_qa", 12)],
        markdown_units=[markdown_unit("raw", "raw-group", "raw-doc", [1, 2])],
        replay_units=[chat_unit("replay", "replay", 13)],
        output_directory=mixed.ROOT / "synthetic_mixed_snapshot_fixture",
        source_disjoint_receipt={"accepted": True, "file_sha256": "a" * 64},
        generated_receipt={"manifest_file_sha256": "b" * 64},
        seed_qa_semantic_receipt={
            "schema": mixed.SEED_QA_SEMANTIC_AUTHORITY_SCHEMA,
            "status": "sealed_passed",
            "semantic_correctness_verified": True,
            "eligible_for_training": True,
            "rows": mixed.SEED_QA_ROWS,
            "assistant_qwen36_tokens": mixed.SEED_ASSISTANT_TOKENS,
            "training_rows_admitted": mixed.SEED_QA_ROWS,
            "file_sha256": "c" * 64,
            "content_sha256": "d" * 64,
            "source_dataset_file_sha256": mixed.STATIC_SHA256[mixed.SEED_QA],
        },
    )
    assert len(payloads) == 3
    assert manifest["schema"] == "mixed-training-snapshot-variant-manifest-v1"
    assert manifest["status"] == "complete_launchable"
    assert manifest["training_launch_authorized"] is True
    assert manifest["source_disjoint_extension_accepted"] is True
    assert manifest["seed_qa_semantic_authority_passed"] is True
    assert manifest["gates"]["seed_qa_semantic_authority_passed"] is True
    assert manifest["rights_exclusion_gate_passed"] is True
    assert manifest["max_sequence_length"] == 2048
    assert manifest["budget_tokens_by_stream"] == {
        "domain_qa": 2, "raw_markdown": 2, "replay": 1,
    }
    assert manifest["resume"]["initial_cursor_commitment_sha256"] == "0" * 64
    assert manifest["resume"]["cursor_commitment_algorithm"] == (
        "sha256-canonical-schedule-row-without-current-commitment-v1"
    )
    assert manifest["sequence_set_identity_sha256"] == (
        manifest["resume"]["sequence_set_identity_sha256"]
    )
    assert manifest["schedule_final_cursor_commitment_sha256"] == (
        manifest["resume"]["final_cursor_commitment_sha256"]
    )
