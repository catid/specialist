from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import pytest

import qwen36_mixed_training_runtime_v1 as runtime


CORE = "protocol_core_100k"
TINY_BUDGETS = {
    "domain_qa": 2,
    "raw_markdown": 3,
    "replay": 2,
}


def _seal(value: dict) -> dict:
    sealed = copy.deepcopy(value)
    sealed["content_sha256_before_self_field"] = runtime.canonical_sha256(sealed)
    return sealed


def _write_json(path: Path, value: dict) -> bytes:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _write_jsonl(path: Path, rows: list[dict]) -> bytes:
    payload = b"".join(runtime.canonical_bytes(row) for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _metadata(*, replay: bool) -> dict:
    return {
        "category": "synthétique",
        "replay": replay,
        "hard_negative": False,
        "verifier": {"status": "passed", "type": "synthetic"},
        "generator": {"status": "synthetic"},
        "rights": {"decision": "authorized_synthetic_or_train_only_replay"},
        "safety_transfer_flags": [],
        "lineage": {"fixture": True},
    }


def _sequence(
    *,
    unit_id: str,
    stream: str,
    input_ids: list[int],
    labels: list[int],
) -> dict:
    training_format = (
        "causal_next_token" if stream == "raw_markdown" else "chat_assistant_only"
    )
    budget = (
        len(input_ids)
        if training_format == "causal_next_token"
        else sum(label != -100 for label in labels)
    )
    source_group = f"group-{unit_id}"
    source_document = f"document-{unit_id}"
    segment = {
        "unit_id": unit_id,
        "token_start": 0,
        "token_stop": len(input_ids),
        "source_token_start": 0,
        "source_token_stop": len(input_ids),
        "budget_token_count": budget,
        "metadata": _metadata(replay=stream == "replay"),
    }
    identity = {
        "stream": stream,
        "training_format": training_format,
        "source_group_id": source_group,
        "source_document_id": source_document,
        "input_ids_sha256": runtime.canonical_sha256(input_ids),
        "labels_sha256": runtime.canonical_sha256(labels),
        "segment_spans": [
            {
                key: segment[key]
                for key in (
                    "unit_id",
                    "token_start",
                    "token_stop",
                    "source_token_start",
                    "source_token_stop",
                )
            }
        ],
    }
    return {
        "schema": runtime.SEQUENCE_SCHEMA,
        "sequence_id": "mixed-sequence-v1:" + runtime.canonical_sha256(identity),
        "stream": stream,
        "training_format": training_format,
        "label_semantics": (
            "causal_next_token_all_tokens_v1"
            if training_format == "causal_next_token"
            else "official_qwen_chat_assistant_only_v1"
        ),
        "source_group_id": source_group,
        "source_document_id": source_document,
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "input_token_count": len(input_ids),
        "budget_token_count": budget,
        "shifted_supervised_token_count": sum(
            label != -100 for label in labels[1:]
        ),
        "segments": [segment],
        "packing": {
            "policy": "same_stream_format_source_group_and_document_greedy_v1",
            "max_tokens": 2048,
            "cross_source_group": False,
            "cross_document": False,
        },
    }


def _valid_sequences() -> list[dict]:
    return [
        _sequence(
            unit_id="domain",
            stream="domain_qa",
            input_ids=[10, 11, 12],
            labels=[-100, 11, 12],
        ),
        _sequence(
            unit_id="markdown",
            stream="raw_markdown",
            input_ids=[20, 21, 22],
            labels=[20, 21, 22],
        ),
        _sequence(
            unit_id="replay",
            stream="replay",
            input_ids=[30, 31, 32],
            labels=[-100, 31, 32],
        ),
    ]


def _schedule(sequences: list[dict], *, variant: str = CORE) -> tuple[list[dict], dict]:
    by_id = {row["sequence_id"]: row for row in sequences}
    queues = {
        stream: sorted(
            [sequence_id for sequence_id, row in by_id.items() if row["stream"] == stream]
        )
        for stream in runtime.STREAM_ORDER
    }
    positions = {stream: 0 for stream in runtime.STREAM_ORDER}
    delivered = {stream: 0 for stream in runtime.STREAM_ORDER}
    sequence_receipts = {
        sequence_id: runtime.canonical_sha256(row)
        for sequence_id, row in by_id.items()
    }
    rows = []
    previous = runtime.ZERO_COMMITMENT
    while any(positions[stream] < len(queues[stream]) for stream in queues):
        available = [
            stream
            for stream in runtime.STREAM_ORDER
            if positions[stream] < len(queues[stream])
        ]
        stream = min(
            available,
            key=lambda item: (
                Fraction(delivered[item], TINY_BUDGETS[item]),
                runtime.STREAM_ORDER.index(item),
            ),
        )
        sequence_id = queues[stream][positions[stream]]
        positions[stream] += 1
        count = by_id[sequence_id]["budget_token_count"]
        delivered[stream] += count
        base = {
            "schema": runtime.SCHEDULE_SCHEMA,
            "cursor": len(rows),
            "variant": variant,
            "sequence_id": sequence_id,
            "sequence_sha256": sequence_receipts[sequence_id],
            "stream": stream,
            "budget_token_count": count,
            "cumulative_budget_tokens": sum(delivered.values()),
            "cumulative_stream_budget_tokens": dict(delivered),
            "previous_cursor_commitment_sha256": previous,
        }
        commitment = runtime.canonical_sha256(base)
        rows.append({**base, "cursor_commitment_sha256": commitment})
        previous = commitment
    sequence_set_identity = runtime.canonical_sha256(
        {
            "variant": variant,
            "budgets": TINY_BUDGETS,
            "sequence_receipts": dict(sorted(sequence_receipts.items())),
        }
    )
    return rows, {
        "sequence_set_identity_sha256": sequence_set_identity,
        "initial_cursor_commitment_sha256": runtime.ZERO_COMMITMENT,
        "final_cursor_commitment_sha256": previous,
        "cursor_count": len(rows),
        "resume_identity": runtime.RESUME_IDENTITY,
        "cursor_commitment_algorithm": runtime.CURSOR_ALGORITHM,
        "cursor_commitment_formula": runtime.CURSOR_FORMULA,
    }


def _generated_receipt() -> dict:
    return {
        "manifest_file_sha256": "1" * 64,
        "manifest_content_sha256": "2" * 64,
        "report_file_sha256": "3" * 64,
        "report_content_sha256": "4" * 64,
        "dataset_file_sha256": "5" * 64,
        "rows": 3,
        "accounting": {"assistant_tokens": 740_847},
    }


def _seed_qa_semantic_receipt() -> dict:
    return {
        "schema": "seed-qa-semantic-authority-v1",
        "status": "sealed_passed",
        "semantic_correctness_verified": True,
        "eligible_for_training": True,
        "rows": runtime.SEED_QA_ROWS,
        "assistant_qwen36_tokens": runtime.SEED_QA_ASSISTANT_TOKENS,
        "training_rows_admitted": runtime.SEED_QA_ROWS,
        "file_sha256": "9" * 64,
        "content_sha256": "a" * 64,
        "source_dataset_file_sha256": "b" * 64,
    }


def _build_fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    variant_mutator=None,
) -> Path:
    monkeypatch.setattr(runtime, "ROOT", root)
    monkeypatch.setattr(
        runtime,
        "EXPECTED_BUDGETS",
        {
            CORE: copy.deepcopy(TINY_BUDGETS),
            "full_authorized_markdown": copy.deepcopy(TINY_BUDGETS),
        },
    )
    snapshot = root / "snapshot"
    variant_dir = snapshot / CORE
    sequences = _valid_sequences()
    schedule, resume = _schedule(sequences)
    sequence_payload = _write_jsonl(variant_dir / "sequences.jsonl", sequences)
    schedule_payload = _write_jsonl(variant_dir / "schedule.jsonl", schedule)

    extension = _seal(
        {
            "schema": "mixed-training-source-disjoint-extension-v1",
            "status": "accepted",
            "accepted_for_training": True,
            "generated_domain_manifest_file_sha256": "1" * 64,
            "static_inputs_sha256": {},
            "opaque_collision_contract": {
                "status": "passed",
                "train_collision_count": 0,
                "protected_source_content_opened": False,
                "protected_identifiers_disclosed": False,
                "opaque_receipt_sha256": "6" * 64,
            },
        }
    )
    extension_payload = _write_json(
        snapshot / "source_disjoint_extension_v1.json", extension
    )
    disjoint_receipt = {
        "path": "snapshot/source_disjoint_extension_v1.json",
        "file_sha256": _sha256(extension_payload),
        "content_sha256": extension["content_sha256_before_self_field"],
        "accepted": True,
        "opaque_receipt_sha256": "6" * 64,
    }
    generated = _generated_receipt()
    seed_semantic = _seed_qa_semantic_receipt()
    tokenizer = {
        "path": "/synthetic/tokenizer",
        "revision": "synthetic-revision",
        "chat_template_sha256": "7" * 64,
    }
    rights = {
        "schema": "mixed-training-rights-exclusion-receipt-v1",
        "status": "passed",
        "rows": 3,
        "rights_decisions": {"authorized_synthetic_or_train_only_replay": 3},
        "unresolved_or_unauthorized_rows": 0,
        "forbidden_source_classes_opened": False,
        "rejected_unreviewed_or_ineligible_rows_included": False,
        "source_rights_status_rewritten": False,
    }
    manifest = {
        "schema": runtime.VARIANT_SCHEMA,
        "status": "complete_launchable",
        "variant": CORE,
        "max_sequence_length": 2048,
        "budget_tokens_by_stream": copy.deepcopy(TINY_BUDGETS),
        "budget_tokens": sum(TINY_BUDGETS.values()),
        "sequences": {
            "path": f"snapshot/{CORE}/sequences.jsonl",
            "sha256": _sha256(sequence_payload),
            "rows": len(sequences),
        },
        "schedule": {
            "path": f"snapshot/{CORE}/schedule.jsonl",
            "sha256": _sha256(schedule_payload),
            "rows": len(schedule),
        },
        "resume": resume,
        "sequence_set_identity_sha256": resume["sequence_set_identity_sha256"],
        "schedule_final_cursor_commitment_sha256": resume[
            "final_cursor_commitment_sha256"
        ],
        "label_semantics": {
            "chat": "official_qwen_chat_assistant_only_v1",
            "markdown": "causal_next_token_all_tokens_v1",
            "prompt_tokens_supervised": False,
        },
        "packing": {
            "policy": "same_stream_format_source_group_and_document_greedy_v1",
            "same_source_group_and_document_only": True,
            "cross_document": False,
            "cross_source_group": False,
        },
        "tokenizer": tokenizer,
        "gates": {
            "generated_domain_semantic_authority_passed": True,
            "generated_domain_receipt": generated,
            "seed_qa_semantic_authority_passed": True,
            "seed_qa_semantic_authority_receipt": seed_semantic,
            "source_disjoint_extension_accepted": True,
            "source_disjoint_extension_receipt": disjoint_receipt,
            "rights_exclusion_gate_passed": True,
            "rights_exclusion_receipt": rights,
            "tokenizer_identity_passed": True,
            "exact_token_accounting_passed": True,
            "packing_invariants_passed": True,
        },
        "source_disjoint_extension_accepted": True,
        "seed_qa_semantic_authority_passed": True,
        "rights_exclusion_gate_passed": True,
        "training_launch_authorized": True,
    }
    if variant_mutator is not None:
        variant_mutator(manifest)
    manifest = _seal(manifest)
    variant_payload = _write_json(variant_dir / "manifest.json", manifest)

    reference = {
        "manifest_path": f"snapshot/{CORE}/manifest.json",
        "manifest_file_sha256": _sha256(variant_payload),
        "manifest_content_sha256": manifest["content_sha256_before_self_field"],
        "budget_tokens": manifest["budget_tokens"],
        "sequence_file_sha256": manifest["sequences"]["sha256"],
        "sequence_rows": manifest["sequences"]["rows"],
        "schedule_file_sha256": manifest["schedule"]["sha256"],
        "schedule_rows": manifest["schedule"]["rows"],
        "sequence_set_identity_sha256": manifest["sequence_set_identity_sha256"],
        "schedule_final_cursor_commitment_sha256": manifest[
            "schedule_final_cursor_commitment_sha256"
        ],
        "exact_budget_tokens_by_stream": copy.deepcopy(
            manifest["budget_tokens_by_stream"]
        ),
    }
    unused_reference = copy.deepcopy(reference)
    unused_reference["manifest_path"] = (
        "snapshot/full_authorized_markdown/manifest.json"
    )
    top = _seal(
        {
            "schema": runtime.TOP_SCHEMA,
            "status": "complete_launchable",
            "generated_domain_authority": generated,
            "seed_qa_semantic_authority": seed_semantic,
            "variants": {
                CORE: reference,
                "full_authorized_markdown": unused_reference,
            },
            "tokenizer": tokenizer,
            "assembler": {"path": "builder.py", "sha256": "8" * 64},
            "gates": {
                "generated_domain_semantic_authority_passed": True,
                "seed_qa_semantic_authority_passed": True,
                "source_disjoint_extension_accepted": True,
                "source_disjoint_extension_receipt": disjoint_receipt,
                "rights_exclusion_gate_passed": True,
                "tokenizer_identity_passed": True,
                "all_variant_manifests_launch_authorized": True,
            },
            "source_disjoint_extension_accepted": True,
            "seed_qa_semantic_authority_passed": True,
            "rights_exclusion_gate_passed": True,
            "max_sequence_length": 2048,
            "packing_invariants": {
                "same_stream_format_source_group_and_document_only": True,
                "cross_source_group": False,
                "cross_document": False,
            },
            "training_launch_authorized": True,
        }
    )
    top_path = snapshot / "manifest.json"
    _write_json(top_path, top)
    return top_path


def _reseal_schedule(rows: list[dict]) -> None:
    delivered = {stream: 0 for stream in runtime.STREAM_ORDER}
    previous = runtime.ZERO_COMMITMENT
    for cursor, row in enumerate(rows):
        row["cursor"] = cursor
        delivered[row["stream"]] += row["budget_token_count"]
        row["cumulative_budget_tokens"] = sum(delivered.values())
        row["cumulative_stream_budget_tokens"] = dict(delivered)
        row["previous_cursor_commitment_sha256"] = previous
        unsigned = copy.deepcopy(row)
        unsigned.pop("cursor_commitment_sha256", None)
        row["cursor_commitment_sha256"] = runtime.canonical_sha256(unsigned)
        previous = row["cursor_commitment_sha256"]


def test_provisional_manifest_fails_before_any_content_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    provisional = _seal(
        {
            "schema": "mixed-training-snapshot-provisional-gate-v1",
            "status": "blocked_missing_sealed_generated_domain_authority",
            "training_launch_authorized": False,
            "training_snapshot_materialized": False,
        }
    )
    path = tmp_path / "snapshot" / "manifest.json"
    _write_json(path, provisional)

    def forbidden_hash(_path: Path) -> str:
        raise AssertionError("training content receipt was opened")

    monkeypatch.setattr(runtime, "file_sha256", forbidden_hash)
    with pytest.raises(runtime.SnapshotContractError, match="manifest schema changed"):
        runtime.load_training_authority(path, variant=CORE)


def test_variant_launch_gate_fails_before_sequence_or_schedule_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _build_fixture(
        tmp_path,
        monkeypatch,
        variant_mutator=lambda manifest: manifest.update(
            training_launch_authorized=False
        ),
    )
    original = runtime.file_sha256
    content_paths = []

    def guarded_hash(candidate: Path) -> str:
        if candidate.name in {"sequences.jsonl", "schedule.jsonl"}:
            content_paths.append(candidate)
            raise AssertionError("training content was opened before launch gates")
        return original(candidate)

    monkeypatch.setattr(runtime, "file_sha256", guarded_hash)
    with pytest.raises(runtime.SnapshotContractError, match="does not authorize training"):
        runtime.load_training_authority(path, variant=CORE)
    assert content_paths == []


def test_missing_seed_semantic_authority_fails_before_training_content_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _build_fixture(tmp_path, monkeypatch)
    top = json.loads(path.read_text(encoding="utf-8"))
    top.pop("content_sha256_before_self_field")
    top.pop("seed_qa_semantic_authority")
    top["seed_qa_semantic_authority_passed"] = False
    top["gates"]["seed_qa_semantic_authority_passed"] = False
    _write_json(path, _seal(top))
    original = runtime.file_sha256
    opened_content = []

    def guarded_hash(candidate: Path) -> str:
        if candidate.name in {"sequences.jsonl", "schedule.jsonl"}:
            opened_content.append(candidate)
            raise AssertionError("training content opened before seed semantic gate")
        return original(candidate)

    monkeypatch.setattr(runtime, "file_sha256", guarded_hash)
    with pytest.raises(runtime.SnapshotContractError, match="seed QA semantic gate"):
        runtime.load_training_authority(path, variant=CORE)
    assert opened_content == []


def test_valid_synthetic_authority_loads_and_preserves_unicode_canonical_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _build_fixture(tmp_path, monkeypatch)
    authority = runtime.load_training_authority(path, variant=CORE)
    assert len(authority.sequences) == 3
    assert len(authority.schedule) == 3
    assert authority.sequence_for_cursor(0)["stream"] == "domain_qa"


def test_sequence_rejects_prompt_supervision() -> None:
    row = _sequence(
        unit_id="bad-chat",
        stream="domain_qa",
        input_ids=[10, 11, 12],
        labels=[10, 11, 12],
    )
    with pytest.raises(runtime.SnapshotContractError, match="prompt is unmasked"):
        runtime.validate_sequences([row])


def test_sequence_rejects_markdown_label_masking() -> None:
    row = _sequence(
        unit_id="bad-markdown",
        stream="raw_markdown",
        input_ids=[20, 21, 22],
        labels=[20, -100, 22],
    )
    with pytest.raises(runtime.SnapshotContractError, match="Markdown is not all-token"):
        runtime.validate_sequences([row])


def test_sequence_rejects_token_outside_qwen_vocabulary() -> None:
    row = _sequence(
        unit_id="bad-token-id",
        stream="raw_markdown",
        input_ids=[20, runtime.QWEN36_VOCAB_SIZE],
        labels=[20, runtime.QWEN36_VOCAB_SIZE],
    )
    with pytest.raises(runtime.SnapshotContractError, match="outside.*vocabulary"):
        runtime.validate_sequences([row])


def test_sequence_rejects_cross_document_packing() -> None:
    row = _valid_sequences()[0]
    row["packing"]["cross_document"] = True
    with pytest.raises(runtime.SnapshotContractError, match="cross-document packing"):
        runtime.validate_sequences([row])


def test_sequence_rejects_declared_budget_not_backed_by_labels() -> None:
    row = _valid_sequences()[0]
    row["segments"][0]["budget_token_count"] = 1
    row["budget_token_count"] = 1
    with pytest.raises(runtime.SnapshotContractError, match="assistant budget changed"):
        runtime.validate_sequences([row])


def test_sequence_rejects_failed_or_empty_verifier_receipt() -> None:
    for verifier in ({"status": "failed"}, {}):
        row = _valid_sequences()[0]
        row["segments"][0]["metadata"]["verifier"] = verifier
        with pytest.raises(
            runtime.SnapshotContractError, match="verifier receipt did not pass"
        ):
            runtime.validate_sequences([row])


def test_schedule_rejects_commitment_tampering(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime,
        "EXPECTED_BUDGETS",
        {CORE: copy.deepcopy(TINY_BUDGETS)},
    )
    sequences = runtime.validate_sequences(_valid_sequences())
    schedule, _ = _schedule(list(sequences.values()))
    schedule[1]["cursor_commitment_sha256"] = "f" * 64
    with pytest.raises(runtime.SnapshotContractError, match="commitment is invalid"):
        runtime.validate_schedule(
            schedule,
            variant=CORE,
            sequences=sequences,
            budgets=TINY_BUDGETS,
        )


def test_schedule_rejects_reordered_but_fully_resealed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime,
        "EXPECTED_BUDGETS",
        {CORE: copy.deepcopy(TINY_BUDGETS)},
    )
    sequences = runtime.validate_sequences(_valid_sequences())
    schedule, _ = _schedule(list(sequences.values()))
    schedule[0], schedule[1] = schedule[1], schedule[0]
    _reseal_schedule(schedule)
    with pytest.raises(runtime.SnapshotContractError, match="deterministic schedule order"):
        runtime.validate_schedule(
            schedule,
            variant=CORE,
            sequences=sequences,
            budgets=TINY_BUDGETS,
        )


def test_manifest_rejects_inexact_budget_before_training_content_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate(manifest: dict) -> None:
        manifest["budget_tokens_by_stream"]["raw_markdown"] += 1
        manifest["budget_tokens"] += 1

    path = _build_fixture(tmp_path, monkeypatch, variant_mutator=mutate)
    original = runtime.file_sha256

    def guarded_hash(candidate: Path) -> str:
        if candidate.name in {"sequences.jsonl", "schedule.jsonl"}:
            raise AssertionError("content opened before exact-budget gate")
        return original(candidate)

    monkeypatch.setattr(runtime, "file_sha256", guarded_hash)
    with pytest.raises(runtime.SnapshotContractError, match="stream budgets changed"):
        runtime.load_training_authority(path, variant=CORE)


def test_resume_identity_accepts_only_exact_cursor_chain_tuple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _build_fixture(tmp_path, monkeypatch)
    authority = runtime.load_training_authority(path, variant=CORE)
    initial = {
        "variant": CORE,
        "sequence_set_identity_sha256": authority.sequence_set_identity_sha256,
        "cursor": 0,
        "cursor_commitment_sha256": runtime.ZERO_COMMITMENT,
    }
    assert runtime.validate_resume_identity(authority, initial) == 0
    after_one = {
        **initial,
        "cursor": 1,
        "cursor_commitment_sha256": authority.schedule[0][
            "cursor_commitment_sha256"
        ],
    }
    assert runtime.validate_resume_identity(authority, after_one) == 1

    for bad_state in (
        {**after_one, "sequence_set_identity_sha256": "0" * 64},
        {**after_one, "cursor_commitment_sha256": "0" * 64},
        {key: value for key, value in after_one.items() if key != "cursor"}
        | {"next_cursor": 1},
    ):
        with pytest.raises(runtime.SnapshotContractError):
            runtime.validate_resume_identity(authority, bad_state)
