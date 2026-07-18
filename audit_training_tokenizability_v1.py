#!/usr/bin/env python3
"""Audit every non-protected training surface with the pinned Qwen tokenizer.

The inventory is deliberately authority-scoped.  Evaluation, holdout, OOD,
terminal, quarantine, incident, manual-review, and active semantic-judge paths
are never discovered or opened.  The report contains lineage and code points,
not copied training text, so findings can be repaired manually in place.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, Iterable

from qwen_chat_masking_v1 import encode_chat_assistant_only


ROOT = Path(__file__).resolve().parent
TOKENIZER_ROOT = ROOT / "models/Qwen3.6-35B-A3B"
REGISTRY = ROOT / "data/site_corpora/registry/site_corpus_registry_v1.json"
STAGING_MANIFEST = (
    ROOT
    / "data/site_corpora/staging/additional_resource_acquisition_v2_batch1/manifest.json"
)
OUTPUT = (
    ROOT / "data/training_inventory/tokenizability_audit_v1/report.json"
)
SCHEMA = "qwen36-training-tokenizability-audit-v1"
MAX_CHAT_TOKENS = 2_048
MODEL_VOCAB_SIZE = 248_320
RESERVED_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
FORBIDDEN_PATH_PARTS = re.compile(
    r"(?:^|[_-])(eval|evaluation|holdout|heldout|ood|terminal|protected|"
    r"quarantine|incident)(?:$|[_-])",
    re.IGNORECASE,
)

SEED_QA = (
    ROOT / "data/training_inventory/high_information_domain_corpus_v1/seed_qa_train.jsonl"
)
CORE_MARKDOWN = (
    ROOT
    / "data/training_inventory/high_information_domain_corpus_v1/raw_continuation_train.jsonl"
)
FULL_MARKDOWN = ROOT / "data/training_inventory/full_train_markdown_cpt_v1/train.jsonl"
REPLAY = (
    ROOT
    / "data/general_replay_v1/replay_authority_v1_150k/general_replay_authority_v1_150k.jsonl"
)
GENERATED_DOMAIN = (
    ROOT
    / "data/training_inventory/high_information_domain_corpus_v1/"
    "generated_domain_authority_v1/train.jsonl"
)
LEGACY_SERIALIZED = (
    ROOT / "data/train_qa_v3_clean.jsonl",
    ROOT / "data/train_qa_verified.jsonl",
)
LEGACY_QA = (
    ROOT / "data/rope_topia_manual_v1.jsonl",
    ROOT / "data/rope_resource_factual_qa_v1.jsonl",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _self_addressed(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _safe_training_file(path: Path) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    _require(lexical.is_relative_to(ROOT), f"training path escapes repository: {path}")
    relative = lexical.relative_to(ROOT)
    _require(
        not any(FORBIDDEN_PATH_PARTS.search(part) for part in relative.parts),
        f"protected/evaluation-like path is outside audit authority: {relative}",
    )
    _require("manual_reviews" not in relative.parts, "manual-review path is excluded")
    current = Path(lexical.anchor)
    metadata = None
    for component in lexical.parts[1:]:
        current /= component
        metadata = current.lstat()
        _require(not stat.S_ISLNK(metadata.st_mode), f"symlink path is forbidden: {relative}")
    _require(metadata is not None and stat.S_ISREG(metadata.st_mode), f"not a regular file: {relative}")
    _require(metadata.st_nlink == 1, f"hard-linked training input is forbidden: {relative}")
    return lexical


def _codepoint(text: str, offset: int) -> dict[str, Any] | None:
    if not (0 <= offset < len(text)):
        return None
    value = ord(text[offset])
    return {"character": text[offset], "codepoint": f"U+{value:04X}"}


def _first_difference(left: str, right: str) -> int:
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return index
    return min(len(left), len(right))


def _encode_plain(tokenizer: Any, text: str) -> list[int]:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        return list(backend.encode(text, add_special_tokens=False).ids)
    return list(tokenizer.encode(text, add_special_tokens=False))


def audit_plain_text(
    tokenizer: Any,
    text: Any,
    *,
    allow_reserved: bool = False,
    maximum_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Return tokenization findings and emitted-token count for one field."""

    findings: list[dict[str, Any]] = []
    if not isinstance(text, str) or not text.strip():
        return [{"failure": "text_is_not_a_nonempty_string"}], 0

    for offset, character in enumerate(text):
        value = ord(character)
        failure = None
        if character == "\ufffd":
            failure = "unicode_replacement_character"
        elif character == "\x00" or (value < 32 and character not in "\n\r\t"):
            failure = "unsupported_control_character"
        elif 0xFDD0 <= value <= 0xFDEF or value & 0xFFFF in (0xFFFE, 0xFFFF):
            failure = "unicode_noncharacter"
        if failure is not None:
            findings.append({"failure": failure, "offset": offset, **_codepoint(text, offset)})

    if not allow_reserved:
        for token in RESERVED_TOKENS:
            start = 0
            while True:
                offset = text.find(token, start)
                if offset < 0:
                    break
                findings.append(
                    {
                        "failure": "reserved_chat_control_token_in_content",
                        "offset": offset,
                        "reserved_token": token,
                    }
                )
                start = offset + len(token)

    try:
        token_ids = _encode_plain(tokenizer, text)
    except Exception as exc:  # exact tokenizer exception is part of the receipt
        findings.append(
            {"failure": "tokenizer_exception", "exception_type": type(exc).__name__}
        )
        return findings, 0
    if not token_ids:
        findings.append({"failure": "tokenizer_emitted_no_tokens"})
    if not all(
        isinstance(item, int)
        and not isinstance(item, bool)
        and 0 <= item < MODEL_VOCAB_SIZE
        for item in token_ids
    ):
        findings.append({"failure": "tokenizer_emitted_out_of_model_vocabulary_id"})
    if maximum_tokens is not None and len(token_ids) > maximum_tokens:
        findings.append(
            {
                "failure": "unsplittable_training_unit_exceeds_token_limit",
                "token_count": len(token_ids),
                "maximum_tokens": maximum_tokens,
            }
        )
    try:
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    if decoded != text:
        offset = _first_difference(text, decoded)
        findings.append(
            {
                "failure": "encode_decode_roundtrip_changed_text",
                "offset": offset,
                "original": _codepoint(text, offset),
                "decoded": _codepoint(decoded, offset),
                "original_length": len(text),
                "decoded_length": len(decoded),
            }
        )
    return findings, len(token_ids)


def _string_leaves(value: Any, prefix: str = "tools") -> Iterable[tuple[str, Any]]:
    if isinstance(value, str):
        yield prefix, value
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _string_leaves(item, f"{prefix}.{index}")
    elif isinstance(value, dict):
        for key, item in sorted(value.items()):
            yield from _string_leaves(item, f"{prefix}.{key}")


def audit_chat_row(tokenizer: Any, row: dict[str, Any]) -> tuple[list[dict[str, Any]], int, int]:
    findings: list[dict[str, Any]] = []
    messages = row.get("messages")
    tools = row.get("tools") or []
    if not isinstance(messages, list) or not messages:
        return [{"failure": "chat_messages_are_missing", "field": "messages"}], 0, 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            findings.append({"failure": "chat_message_is_not_an_object", "field": f"messages.{index}"})
            continue
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        # Qwen's official template represents an assistant function call with
        # an empty textual content field and a non-empty structured tool_calls
        # list.  That is a tokenizable structured message, not missing data.
        structured_assistant_call = (
            message.get("role") == "assistant"
            and content == ""
            and isinstance(tool_calls, list)
            and bool(tool_calls)
        )
        if not structured_assistant_call:
            field_findings, _ = audit_plain_text(
                tokenizer, content, allow_reserved=False
            )
            findings.extend(
                {**item, "field": f"messages.{index}.content"}
                for item in field_findings
            )
        if tool_calls is not None:
            if not isinstance(tool_calls, list) or not tool_calls:
                findings.append(
                    {
                        "failure": "tool_calls_is_not_a_nonempty_list",
                        "field": f"messages.{index}.tool_calls",
                    }
                )
            else:
                for field, text in _string_leaves(
                    tool_calls, f"messages.{index}.tool_calls"
                ):
                    field_findings, _ = audit_plain_text(
                        tokenizer, text, allow_reserved=False
                    )
                    findings.extend({**item, "field": field} for item in field_findings)
    for field, text in _string_leaves(tools):
        field_findings, _ = audit_plain_text(tokenizer, text, allow_reserved=False)
        findings.extend({**item, "field": field} for item in field_findings)
    try:
        encoded = encode_chat_assistant_only(
            tokenizer, messages, enable_thinking=False, tools=tools
        )
    except Exception as exc:
        findings.append(
            {
                "failure": "official_chat_template_or_mask_exception",
                "field": "messages",
                "exception_type": type(exc).__name__,
            }
        )
        return findings, 0, 0
    total = encoded["total_token_count"]
    assistant = encoded["assistant_token_count"]
    if total > MAX_CHAT_TOKENS:
        findings.append(
            {
                "failure": "unsplittable_chat_exceeds_token_limit",
                "field": "messages",
                "token_count": total,
                "maximum_tokens": MAX_CHAT_TOKENS,
            }
        )
    declared = row.get("assistant_token_count", row.get("assistant_qwen36_token_count"))
    if declared is not None and declared != assistant:
        findings.append(
            {
                "failure": "declared_assistant_token_count_changed",
                "field": "assistant_token_count",
                "declared": declared,
                "observed": assistant,
            }
        )
    if not all(0 <= item < MODEL_VOCAB_SIZE for item in encoded["input_ids"]):
        findings.append({"failure": "chat_emitted_out_of_model_vocabulary_id", "field": "messages"})
    return findings, total, assistant


def _finding_rows(
    path: Path,
    locator: str | int,
    default_field: str,
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for item in findings:
        value = dict(item)
        value.setdefault("field", default_field)
        result.append({"path": _relative(path), "record": locator, **value})
    return result


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, 1):
            _require(line.endswith("\n") and line.strip(), f"{_relative(path)}:{line_number}: invalid line")
            value = json.loads(line)
            _require(isinstance(value, dict), f"{_relative(path)}:{line_number}: row is not an object")
            yield line_number, value


def _source_summary(path: Path, role: str) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "records": 0,
        "text_fields": 0,
        "characters": 0,
        "tokens": 0,
        "assistant_tokens": 0,
        "findings": 0,
    }


def _audit_markdown(tokenizer: Any, path: Path, role: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = _safe_training_file(path)
    summary = _source_summary(path, role)
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        finding = {
            "path": _relative(path),
            "record": "document",
            "field": "markdown",
            "failure": "invalid_utf8",
            "byte_offset": exc.start,
        }
        summary["findings"] = 1
        return summary, [finding]
    findings, tokens = audit_plain_text(tokenizer, text)
    rows = _finding_rows(path, "document", "markdown", findings)
    summary.update(
        records=1,
        text_fields=1,
        characters=len(text),
        tokens=tokens,
        findings=len(rows),
    )
    return summary, rows


def _audit_jsonl(tokenizer: Any, path: Path, role: str, mode: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = _safe_training_file(path)
    summary = _source_summary(path, role)
    findings: list[dict[str, Any]] = []
    try:
        rows = _read_jsonl(path)
        for line_number, row in rows:
            summary["records"] += 1
            if mode == "replay_chat" or mode == "generated_chat":
                found, total, assistant = audit_chat_row(tokenizer, row)
                summary["text_fields"] += len(row.get("messages", []))
                summary["characters"] += sum(
                    len(message.get("content", ""))
                    for message in row.get("messages", [])
                    if isinstance(message, dict) and isinstance(message.get("content"), str)
                )
                summary["tokens"] += total
                summary["assistant_tokens"] += assistant
                findings.extend(_finding_rows(path, line_number, "messages", found))
            elif mode == "seed_qa":
                messages = [
                    {"role": "user", "content": row.get("question")},
                    {"role": "assistant", "content": row.get("answer")},
                ]
                synthetic = {
                    "messages": messages,
                    "tools": [],
                    "assistant_qwen36_token_count": row.get("assistant_qwen36_token_count"),
                }
                found, total, assistant = audit_chat_row(tokenizer, synthetic)
                summary["text_fields"] += 2
                summary["characters"] += sum(
                    len(message["content"]) for message in messages if isinstance(message["content"], str)
                )
                summary["tokens"] += total
                summary["assistant_tokens"] += assistant
                findings.extend(_finding_rows(path, line_number, "messages", found))
            else:
                text = row.get("text")
                allow_reserved = mode == "legacy_serialized"
                maximum = MAX_CHAT_TOKENS if mode == "legacy_serialized" else None
                found, tokens = audit_plain_text(
                    tokenizer,
                    text,
                    allow_reserved=allow_reserved,
                    maximum_tokens=maximum,
                )
                summary["text_fields"] += 1
                summary["characters"] += len(text) if isinstance(text, str) else 0
                summary["tokens"] += tokens
                findings.extend(_finding_rows(path, line_number, "text", found))
    except UnicodeDecodeError as exc:
        findings.append(
            {
                "path": _relative(path),
                "record": "file",
                "field": "jsonl",
                "failure": "invalid_utf8",
                "byte_offset": exc.start,
            }
        )
    except (json.JSONDecodeError, RuntimeError) as exc:
        findings.append(
            {
                "path": _relative(path),
                "record": "file",
                "field": "jsonl",
                "failure": "invalid_jsonl_contract",
                "exception_type": type(exc).__name__,
            }
        )
    summary["findings"] = len(findings)
    return summary, findings


def _load_tokenizer() -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ROOT,
        local_files_only=True,
        trust_remote_code=False,
    )
    _require(len(tokenizer) <= MODEL_VOCAB_SIZE, "tokenizer vocabulary exceeds model embeddings")
    _require(isinstance(tokenizer.chat_template, str) and tokenizer.chat_template, "official chat template missing")
    return tokenizer


def _markdown_inventory() -> tuple[list[tuple[Path, str, str | None]], list[dict[str, Any]]]:
    targets: list[tuple[Path, str, str | None]] = []
    registry_path = _safe_training_file(REGISTRY)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    _require(registry.get("schema") == "site-corpus-registry-v1", "site registry schema changed")
    registered = set()
    for artifact in registry.get("artifacts", []):
        path = ROOT / artifact["markdown_path"]
        registered.add(path.resolve())
        targets.append((path, "registered_markdown_training_pool", artifact.get("markdown_sha256")))

    staging_path = _safe_training_file(STAGING_MANIFEST)
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    for receipt in staging.get("sealed_files", []):
        if str(receipt.get("path", "")).endswith("/CORPUS.md"):
            targets.append(
                (
                    staging_path.parent / receipt["path"],
                    "staged_markdown_candidate_not_training_eligible",
                    receipt.get("sha256"),
                )
            )

    known = {path.resolve() for path, _, _ in targets}
    for path in sorted((ROOT / "data/site_corpora").glob("*/CORPUS.md")):
        if path.resolve() not in known and path.resolve() not in registered:
            targets.append((path, "unregistered_markdown_excluded_from_training", None))
    receipts = [
        {"path": _relative(path), "role": role, "expected_sha256": expected}
        for path, role, expected in targets
    ]
    return sorted(targets, key=lambda item: _relative(item[0])), receipts


def build_report(tokenizer: Any | None = None) -> dict[str, Any]:
    tokenizer = _load_tokenizer() if tokenizer is None else tokenizer
    sources: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    markdown_targets, declared_markdown = _markdown_inventory()
    for path, role, expected in markdown_targets:
        path = _safe_training_file(path)
        if expected is not None and file_sha256(path) != expected:
            findings.append(
                {
                    "path": _relative(path),
                    "record": "file",
                    "field": "receipt",
                    "failure": "declared_source_sha256_changed",
                }
            )
        summary, found = _audit_markdown(tokenizer, path, role)
        sources.append(summary)
        findings.extend(found)

    jsonl_targets = [
        (SEED_QA, "sealed_seed_qa_training_input", "seed_qa"),
        (CORE_MARKDOWN, "protocol_core_markdown_training_input", "plain_text"),
        (FULL_MARKDOWN, "full_markdown_training_input", "plain_text"),
        (REPLAY, "sealed_general_replay_training_input", "replay_chat"),
        *((path, "legacy_training_candidate_not_direct_authority", "legacy_serialized") for path in LEGACY_SERIALIZED),
        *((path, "legacy_qa_candidate_not_direct_authority", "legacy_qa") for path in LEGACY_QA),
    ]
    for path in sorted((ROOT / "data/public_training_shards").glob("*/qa.jsonl")):
        jsonl_targets.append((path, "public_qa_candidate_not_direct_authority", "legacy_qa"))
    generated_status = "absent_pending_sealed_generated_domain_authority"
    if GENERATED_DOMAIN.exists():
        jsonl_targets.append((GENERATED_DOMAIN, "sealed_generated_domain_training_input", "generated_chat"))
        generated_status = "present_and_scanned"
    for path, role, mode in jsonl_targets:
        # Legacy QA rows store explicit question/answer rather than serialized
        # text.  Convert them to chat in-memory without mutating source data.
        if mode == "legacy_qa":
            path = _safe_training_file(path)
            summary = _source_summary(path, role)
            found_rows: list[dict[str, Any]] = []
            for line_number, row in _read_jsonl(path):
                summary["records"] += 1
                synthetic = {
                    "messages": [
                        {"role": "user", "content": row.get("question")},
                        {"role": "assistant", "content": row.get("answer")},
                    ],
                    "tools": [],
                }
                found, total, assistant = audit_chat_row(tokenizer, synthetic)
                summary["text_fields"] += 2
                summary["characters"] += sum(
                    len(message["content"])
                    for message in synthetic["messages"]
                    if isinstance(message["content"], str)
                )
                summary["tokens"] += total
                summary["assistant_tokens"] += assistant
                found_rows.extend(_finding_rows(path, line_number, "messages", found))
            summary["findings"] = len(found_rows)
            sources.append(summary)
            findings.extend(found_rows)
        else:
            summary, found = _audit_jsonl(tokenizer, path, role, mode)
            sources.append(summary)
            findings.extend(found)

    sources.sort(key=lambda item: (item["path"], item["role"]))
    findings.sort(
        key=lambda item: (
            item["path"], str(item["record"]), item.get("field", ""),
            item["failure"], item.get("offset", -1),
        )
    )
    value = {
        "schema": SCHEMA,
        "status": "clean" if not findings else "manual_repair_required",
        "scope": {
            "registered_markdown_inventory": _relative(REGISTRY),
            "declared_markdown_targets": declared_markdown,
            "generated_domain": generated_status,
            "active_semantic_judge_partials_opened": False,
            "protected_or_evaluation_sources_opened": False,
            "explicitly_never_opened_path_classes": [
                "evaluation", "eval", "holdout", "heldout", "ood",
                "terminal", "protected", "quarantine", "incident",
                "manual_reviews", "active semantic-judge partial outputs",
            ],
        },
        "tokenizer": {
            "path": _relative(TOKENIZER_ROOT),
            "tokenizer_json_sha256": file_sha256(TOKENIZER_ROOT / "tokenizer.json"),
            "tokenizer_config_sha256": file_sha256(TOKENIZER_ROOT / "tokenizer_config.json"),
            "chat_template_sha256": hashlib.sha256(
                tokenizer.chat_template.encode("utf-8")
            ).hexdigest(),
            "tokenizer_length": len(tokenizer),
            "model_embedding_vocabulary_size": MODEL_VOCAB_SIZE,
            "maximum_unsplittable_chat_tokens": MAX_CHAT_TOKENS,
        },
        "checks": [
            "strict_utf8", "nonempty_text", "no_replacement_control_or_noncharacter_codepoints",
            "no_reserved_chat_control_tokens_in_content", "qwen_encode_succeeds",
            "emitted_ids_within_model_vocabulary", "encode_decode_roundtrip_exact",
            "official_chat_template_and_assistant_mask", "declared_assistant_token_count",
            "unsplittable_chat_at_most_2048_tokens",
        ],
        "accounting": {
            "files": len(sources),
            "records": sum(item["records"] for item in sources),
            "text_fields": sum(item["text_fields"] for item in sources),
            "characters": sum(item["characters"] for item in sources),
            "tokens": sum(item["tokens"] for item in sources),
            "assistant_tokens": sum(item["assistant_tokens"] for item in sources),
            "findings": len(findings),
        },
        "sources": sources,
        "findings": findings,
        "scanner": {
            "path": _relative(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
    }
    return _self_addressed(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--allow-findings", action="store_true")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)
    report = build_report()
    output = arguments.output.resolve()
    _require(output.is_relative_to(ROOT), "audit output must remain inside repository")
    if arguments.check:
        _require(output.is_file() and not output.is_symlink(), "checked audit report is missing")
        observed = json.loads(output.read_text(encoding="utf-8"))
        _require(observed == report, "tokenizability audit report changed")
    else:
        _atomic_json(output, report)
    _require(
        arguments.allow_findings or report["status"] == "clean",
        "training tokenizability findings require manual repair",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "files": report["accounting"]["files"],
                "records": report["accounting"]["records"],
                "findings": report["accounting"]["findings"],
                "content_sha256": report["content_sha256_before_self_field"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
