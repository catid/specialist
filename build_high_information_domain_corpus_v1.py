#!/usr/bin/env python3
"""Build the train-only high-information domain-corpus generation plan.

The only semantic inputs are the two sealed TRAIN projections produced by the
source-group authority.  Lineage paths inside rows are inert strings: this
builder never resolves, stats, hashes, or opens them.  It emits:

* an exact, URL-scrubbed, source-balanced 100k-token raw continuation subset;
* URL-free seed QA rows already present in the sealed V440 train projection;
* one sanitized source context per train site source group;
* content-addressed candidate-generation requests balanced over four shards;
* a prompt/verifier contract and a self-addressed manifest.

Candidate answers are not fabricated by this builder.  The 550k/200k assistant
token allocations are generation targets and remain unachieved until candidates
pass both deterministic checks and a source-entailment/calibration verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import re
import stat
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from qwen_chat_masking_v1 import encode_chat_assistant_only, token_ids


ROOT = Path(__file__).resolve().parent
INPUT_DIR = (ROOT / "data/training_inventory/source_group_split_v1").resolve()
SOURCE_SPLIT_AUTHORITY = (
    ROOT / "data/training_inventory/source_group_split_authority_v1.json"
).resolve()
SITE_INPUT = (INPUT_DIR / "train_site_spans.jsonl").resolve()
QA_INPUT = (INPUT_DIR / "train_v440_qa.jsonl").resolve()
OUTPUT_DIR = (
    ROOT / "data/training_inventory/high_information_domain_corpus_v1"
).resolve()
MANIFEST = (OUTPUT_DIR / "manifest.json").resolve()
PROMPT_SPEC = (OUTPUT_DIR / "prompt_spec.json").resolve()
SEMANTIC_VERIFIER_CONTRACT = (
    OUTPUT_DIR / "semantic_verifier_contract.json"
).resolve()
SOURCE_CONTEXTS = (OUTPUT_DIR / "source_contexts.jsonl").resolve()
RAW_CONTINUATION = (OUTPUT_DIR / "raw_continuation_train.jsonl").resolve()
SEED_QA = (OUTPUT_DIR / "seed_qa_train.jsonl").resolve()
REQUEST_SHARDS = tuple(
    (OUTPUT_DIR / f"generation_requests_gpu{index}.jsonl").resolve()
    for index in range(4)
)

TOKENIZER_JSON = (ROOT / "models/Qwen3.6-35B-A3B/tokenizer.json").resolve()
TOKENIZER_CONFIG = (
    ROOT / "models/Qwen3.6-35B-A3B/tokenizer_config.json"
).resolve()
TOKENIZER_JSON_SHA256 = (
    "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
)
TOKENIZER_CONFIG_SHA256 = (
    "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b"
)

SCHEMA = "high-information-domain-corpus-plan-v1"
SITE_SCHEMA = "site-source-span-projection-v1"
QA_LINEAGE_SCHEMA = "v440-qa-source-split-lineage-v1"
RAW_TARGET_TOKENS = 100_000
CLOSED_BOOK_TARGET_ASSISTANT_TOKENS = 550_000
GROUNDED_TARGET_ASSISTANT_TOKENS = 200_000
HARD_NEGATIVE_NUMERATOR = 75
HARD_NEGATIVE_DENOMINATOR = 1_000
RAW_SOURCE_BALANCED_PREFIX_TOKENS = 90_000
RAW_MIN_PARAGRAPH_TOKENS = 8
RAW_MAX_PARAGRAPH_TOKENS = 512
GPU_SHARDS = 4
GENERATOR_MODEL_IDENTITY_SHA256 = (
    "4a4960ba80c4e6532f5225984310af35a2a79cd50ffb642ef1c5a54bbe5fba3c"
)
ASSISTANT_MASK_METHOD = (
    "official_template_role_blocks_and_final_prefix_alignment_v1"
)

FORBIDDEN_SOURCE_TOKENS = (
    "benchmark",
    "dev",
    "development",
    "eval",
    "evaluation",
    "final",
    "protected",
    "holdout",
    "heldout",
    "ood",
    "terminal",
    "incident",
    "manual_review",
    "manual-review",
    "manualreview",
)
URL_LINE_RE = re.compile(
    r"(?im)^\s*(?:source(?: url)?|original url|archived capture|"
    r"verified successor|primary source)\s*:\s*https?://\S+\s*$"
)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(https?://[^)]+\)", re.I)
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
DOMAIN_RE = re.compile(
    r"(?i)(?<![\w@])(?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|io|co|uk|jp)\b"
)
URL_MEMORIZATION_RE = re.compile(
    r"(?i)\b(?:canonical\s+url|url|web\s+address|website|hyperlink|"
    r"which\s+site|where\s+online|domain\s+name)\b"
)
HIDDEN_REASONING_RE = re.compile(
    r"(?is)<\|im_(?:start|end)\|>|<think>|</think>|"
    r"chain[- ]of[- ]thought|hidden reasoning|scratchpad"
)
HEADING_ONLY_RE = re.compile(r"(?m)^\s*#{1,6}\s*$")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def content_id(prefix: str, value: Any) -> str:
    return f"{prefix}:{canonical_sha256(value)}"


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def atomic_write(path: Path, payload: bytes) -> None:
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


def jsonl_payload(rows: Sequence[dict]) -> bytes:
    return b"".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _secure_regular_exact(path: Path, expected: Path, role: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    exact = Path(os.path.abspath(os.fspath(expected)))
    if lexical != exact:
        raise RuntimeError(f"{role} is not the canonical allowlisted path")
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


def _load_source_split_authority() -> tuple[dict, bytes]:
    authority_path = _secure_regular_exact(
        SOURCE_SPLIT_AUTHORITY,
        SOURCE_SPLIT_AUTHORITY,
        "source-disjoint split authority",
    )
    raw = authority_path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError("source-disjoint split authority is not an object")
    declared = value.get("content_sha256_before_self_field")
    unsigned = dict(value)
    unsigned.pop("content_sha256_before_self_field", None)
    final = value.get("assignments", {}).get("final", {})
    if (
        value.get("schema") != "specialist-source-group-split-authority-v1"
        or value.get("status")
        != "sealed_source_disjoint_assignment_launch_still_gated"
        or not isinstance(declared, str)
        or declared != canonical_sha256(unsigned)
        or not isinstance(final, dict)
        or final.get("records_redacted") is not True
        or "records" in final
        or value.get("invariants", {}).get("final_records_emitted") is not False
    ):
        raise RuntimeError("source-disjoint split authority is unsealed or unsafe")
    return value, raw


def sealed_train_projection_receipt(path: Path) -> dict:
    lexical = Path(os.path.abspath(os.fspath(path)))
    allowed = {
        Path(os.path.abspath(os.fspath(SITE_INPUT))): (
            "site_spans",
            "site-source-span-projection-v1",
        ),
        Path(os.path.abspath(os.fspath(QA_INPUT))): (
            "v440_qa",
            "v440-qa-source-split-projection-v1",
        ),
    }
    contract = allowed.get(lexical)
    if contract is None:
        raise RuntimeError(f"input is not an allowlisted sealed TRAIN projection: {path}")
    if _path_tokens(lexical) & set(FORBIDDEN_SOURCE_TOKENS):
        raise RuntimeError(f"sealed TRAIN input crosses a forbidden boundary: {lexical}")
    _secure_regular_exact(lexical, lexical, "sealed TRAIN projection")
    authority, _ = _load_source_split_authority()
    projections = authority.get("materialized_train_development_projections")
    if not isinstance(projections, dict) or set(projections) != {
        "train",
        "development",
    }:
        raise RuntimeError("source-disjoint projection receipts are missing")
    train = projections.get("train")
    if not isinstance(train, dict):
        raise RuntimeError("source-disjoint TRAIN projection receipts are missing")
    projection_kind, expected_schema = contract
    receipt = train.get(projection_kind)
    group_count_key = (
        "source_groups"
        if projection_kind == "site_spans"
        else "source_document_groups"
    )
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != expected_schema
        or receipt.get("contains_only_partition") != "train"
        or receipt.get("path") != relative(lexical)
        or not isinstance(receipt.get("file_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", receipt["file_sha256"])
        or not isinstance(receipt.get("rows"), int)
        or isinstance(receipt.get("rows"), bool)
        or receipt["rows"] <= 0
        or not isinstance(receipt.get("qwen36_tokens"), int)
        or isinstance(receipt.get("qwen36_tokens"), bool)
        or receipt["qwen36_tokens"] <= 0
        or not isinstance(receipt.get(group_count_key), int)
        or isinstance(receipt.get(group_count_key), bool)
        or receipt[group_count_key] <= 0
    ):
        raise RuntimeError("sealed TRAIN projection receipt changed")
    return receipt


def _sealed_train_projection_bytes(path: Path) -> tuple[bytes, dict]:
    receipt = sealed_train_projection_receipt(path)
    raw = Path(path).read_bytes()
    if sha256_bytes(raw) != receipt["file_sha256"]:
        raise RuntimeError("sealed TRAIN projection bytes differ from authority receipt")
    return raw, receipt


def _assert_exact_train_projection(path: Path) -> None:
    """Compatibility wrapper retained for callers that only need the gate."""

    sealed_train_projection_receipt(path)


def load_tokenizer() -> Any:
    if (
        file_sha256(TOKENIZER_JSON) != TOKENIZER_JSON_SHA256
        or file_sha256(TOKENIZER_CONFIG) != TOKENIZER_CONFIG_SHA256
    ):
        raise RuntimeError("pinned Qwen3.6 tokenizer changed")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("run with .venv/bin/python; transformers is required") from error
    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_JSON.parent),
        local_files_only=True,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("pinned Qwen3.6 tokenizer has no official chat template")
    return tokenizer


def token_count(tokenizer: Any, text: str) -> int:
    return len(token_ids(tokenizer.encode(text, add_special_tokens=False)))


def seed_qa_messages(question: str, answer: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Answer the domain question briefly and factually. State uncertainty "
                "when the supplied knowledge does not support a definite answer."
            ),
        },
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def official_assistant_token_count(
    tokenizer: Any, messages: Sequence[dict[str, str]]
) -> int:
    encoded = encode_chat_assistant_only(
        tokenizer,
        messages,
        enable_thinking=False,
    )
    count = encoded["assistant_token_count"]
    if (
        encoded.get("mask_method") != ASSISTANT_MASK_METHOD
        or count <= 0
        or not any(encoded["assistant_mask"])
    ):
        raise RuntimeError("official-template assistant target mask is empty")
    return count


def sanitize_source_text(text: str) -> str:
    """Remove URL-learning surfaces while preserving substantive anchor text."""

    if not isinstance(text, str) or not text:
        raise RuntimeError("sealed site span has no text")
    cleaned = URL_LINE_RE.sub("", text)
    cleaned = MARKDOWN_LINK_RE.sub(lambda match: match.group(1), cleaned)
    cleaned = URL_RE.sub("", cleaned)
    cleaned = DOMAIN_RE.sub("", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    cleaned = HEADING_ONLY_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if not cleaned:
        raise RuntimeError("URL scrubbing removed an entire source span")
    cleaned += "\n"
    if URL_RE.search(cleaned) or DOMAIN_RE.search(cleaned):
        raise RuntimeError("URL-like content survived source sanitization")
    return cleaned


def has_url_memorization_surface(*values: str) -> bool:
    text = "\n".join(value for value in values if isinstance(value, str))
    return bool(URL_RE.search(text) or DOMAIN_RE.search(text) or URL_MEMORIZATION_RE.search(text))


def load_site_rows(tokenizer: Any) -> list[dict]:
    raw, receipt = _sealed_train_projection_bytes(SITE_INPUT)
    rows = []
    seen_records: set[str] = set()
    try:
        text_input = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("sealed site TRAIN projection is not UTF-8") from exc
    for line_number, line in enumerate(text_input.splitlines(), 1):
        if not line.strip():
            raise RuntimeError(
                f"sealed site projection has a blank row at line {line_number}"
            )
        row = json.loads(line)
        if (
            row.get("schema") != SITE_SCHEMA
            or row.get("split") != "train"
            or row.get("assistant_supervision") is not False
            or row.get("cross_document_packing_performed") is not False
        ):
            raise RuntimeError(f"sealed site row contract changed at line {line_number}")
        record_id = row.get("projection_record_id")
        text = row.get("text")
        if (
            not isinstance(record_id, str)
            or record_id in seen_records
            or not isinstance(text, str)
            or sha256_bytes(text.encode("utf-8")) != row.get("text_sha256")
            or token_count(tokenizer, text) != row.get("qwen36_token_count")
        ):
            raise RuntimeError(f"sealed site identity/token drift at line {line_number}")
        if not isinstance(row.get("rights_basis"), dict) or not isinstance(
            row.get("safety_transfer_flags"), list
        ):
            raise RuntimeError(f"site rights/safety lineage missing at line {line_number}")
        seen_records.add(record_id)
        derived = dict(row)
        derived["sanitized_text"] = sanitize_source_text(text)
        derived["sanitized_text_sha256"] = sha256_bytes(
            derived["sanitized_text"].encode("utf-8")
        )
        derived["sanitized_qwen36_token_count"] = token_count(
            tokenizer, derived["sanitized_text"]
        )
        rows.append(derived)
    if (
        not rows
        or len(rows) != receipt["rows"]
        or sum(row["qwen36_token_count"] for row in rows)
        != receipt["qwen36_tokens"]
        or len({row["source_group_id"] for row in rows})
        != receipt.get("source_groups")
    ):
        raise RuntimeError("sealed site TRAIN projection accounting changed")
    return rows


def _seed_qa_exclusion_reason(row: dict) -> str | None:
    if row.get("kind") in {"qa_resource_direct", "qa_resource_category"}:
        return "resource_lookup_kind"
    question = row.get("question")
    answer = row.get("answer")
    if not isinstance(question, str) or not isinstance(answer, str):
        return "missing_question_or_answer"
    if has_url_memorization_surface(question, answer):
        return "url_or_web_lookup_surface"
    if HIDDEN_REASONING_RE.search(question) or HIDDEN_REASONING_RE.search(answer):
        return "hidden_reasoning_or_protocol_surface"
    return None


def load_seed_qa(tokenizer: Any) -> tuple[list[dict], dict]:
    raw, receipt = _sealed_train_projection_bytes(QA_INPUT)
    rows = []
    excluded: Counter[str] = Counter()
    excluded_ids: list[str] = []
    seen_facts: set[str] = set()
    input_source_groups: set[str] = set()
    included_source_groups: set[str] = set()
    source_projection_tokens = 0
    try:
        text_input = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("sealed QA TRAIN projection is not UTF-8") from exc
    for line_number, line in enumerate(text_input.splitlines(), 1):
        if not line.strip():
            raise RuntimeError(
                f"sealed QA projection has a blank row at line {line_number}"
            )
        source = json.loads(line)
        lineage = source.get("source_split_lineage_v1")
        fact_id = source.get("fact_id")
        if (
            not isinstance(lineage, dict)
            or lineage.get("schema") != QA_LINEAGE_SCHEMA
            or lineage.get("split") != "train"
            or not isinstance(fact_id, str)
            or not fact_id
            or fact_id in seen_facts
        ):
            raise RuntimeError(f"sealed QA lineage changed at line {line_number}")
        seen_facts.add(fact_id)
        source_group_id = lineage.get("source_group_id")
        if not isinstance(source_group_id, str) or not source_group_id:
            raise RuntimeError(f"sealed QA source group missing at line {line_number}")
        projection_tokens = lineage.get("qwen36_token_count")
        if (
            not isinstance(projection_tokens, int)
            or isinstance(projection_tokens, bool)
            or projection_tokens <= 0
        ):
            raise RuntimeError(f"sealed QA token lineage missing at line {line_number}")
        source_projection_tokens += projection_tokens
        input_source_groups.add(source_group_id)
        reason = _seed_qa_exclusion_reason(source)
        if reason is not None:
            excluded[reason] += 1
            excluded_ids.append(fact_id)
            continue
        answer = source["answer"]
        messages = seed_qa_messages(source["question"], answer)
        assistant_tokens = official_assistant_token_count(tokenizer, messages)
        evidence = source.get("evidence")
        sanitized_evidence = (
            sanitize_source_text(evidence)
            if isinstance(evidence, str) and evidence.strip()
            else None
        )
        row_identity = {
            "fact_id": fact_id,
            "document_sha256": source.get("document_sha256"),
            "source_group_id": lineage.get("source_group_id"),
            "question_sha256": sha256_bytes(source["question"].encode("utf-8")),
            "answer_sha256": sha256_bytes(answer.encode("utf-8")),
        }
        rows.append(
            {
                "schema": "high-information-seed-qa-v1",
                "record_id": content_id("seed-qa-v1", row_identity),
                "training_family": "closed_book_seed_qa",
                "question": source["question"],
                "answer": answer,
                "answer_text_qwen36_token_count": token_count(tokenizer, answer),
                "assistant_qwen36_token_count": assistant_tokens,
                "assistant_token_mask_method": ASSISTANT_MASK_METHOD,
                "enable_thinking": False,
                "kind": source.get("kind"),
                "fact_id": fact_id,
                "document_sha256": source.get("document_sha256"),
                "source_group_id": lineage.get("source_group_id"),
                "duplicate_component_id": lineage.get("duplicate_component_id"),
                "evidence": sanitized_evidence,
                "evidence_sha256": (
                    sha256_bytes(sanitized_evidence.encode("utf-8"))
                    if sanitized_evidence is not None
                    else None
                ),
                "rights_basis": None,
                "rights_status": "not_declared_in_sealed_v440_train_projection",
                "safety_transfer_flags": [],
                "safety_transfer_status": (
                    "not_declared_in_sealed_v440_train_projection; "
                    "semantic verification required"
                ),
                "lineage": {
                    "sealed_parent_projection_sha256": lineage.get(
                        "parent_projection_sha256"
                    ),
                    "source_lineage_commitment_sha256": canonical_sha256(
                        source.get("source_lineage")
                    ),
                    "source_record_commitment_sha256": canonical_sha256(source),
                    "original_lineage_paths_dereferenced": False,
                },
                "assistant_supervision": True,
                "hidden_reasoning_supervision": False,
            }
        )
        included_source_groups.add(source_group_id)
    if (
        len(seen_facts) != receipt["rows"]
        or source_projection_tokens != receipt["qwen36_tokens"]
        or len(input_source_groups) != receipt.get("source_document_groups")
    ):
        raise RuntimeError("sealed QA TRAIN projection accounting changed")
    rows.sort(key=lambda item: item["record_id"])
    return rows, {
        "input_rows": len(seen_facts),
        "included_rows": len(rows),
        "included_assistant_qwen36_tokens": sum(
            row["assistant_qwen36_token_count"] for row in rows
        ),
        "excluded_rows": sum(excluded.values()),
        "exclusion_counts": dict(sorted(excluded.items())),
        "excluded_fact_id_commitment_sha256": canonical_sha256(sorted(excluded_ids)),
        "input_source_groups": len(input_source_groups),
        "included_source_groups": len(included_source_groups),
        "excluded_only_source_groups": len(
            input_source_groups - included_source_groups
        ),
        "excluded_only_source_group_commitment_sha256": canonical_sha256(
            sorted(input_source_groups - included_source_groups)
        ),
        "all_input_source_groups_accounted_for": True,
    }


def build_source_contexts(site_rows: Sequence[dict], tokenizer: Any) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in site_rows:
        grouped[row["source_group_id"]].append(row)
    contexts = []
    for source_group_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda item: (item["byte_start"], item["span_id"]))
        rights_digests = {canonical_sha256(row["rights_basis"]) for row in rows}
        if len(rights_digests) != 1:
            raise RuntimeError(f"rights basis varies within source group {source_group_id}")
        text = "\n".join(row["sanitized_text"].rstrip() for row in rows).strip() + "\n"
        if URL_RE.search(text) or DOMAIN_RE.search(text):
            raise RuntimeError("URL-like content survived source-context build")
        safety = sorted(
            {flag for row in rows for flag in row["safety_transfer_flags"]}
        )
        context_identity = {
            "source_group_id": source_group_id,
            "span_ids": [row["span_id"] for row in rows],
            "text_sha256": sha256_bytes(text.encode("utf-8")),
        }
        contexts.append(
            {
                "schema": "high-information-source-context-v1",
                "context_id": content_id("source-context-v1", context_identity),
                "source_group_id": source_group_id,
                "duplicate_component_id": rows[0]["duplicate_component_id"],
                "resource_id": rows[0]["resource_id"],
                "artifact_id": rows[0]["artifact_id"],
                "span_ids": [row["span_id"] for row in rows],
                "parent_span_ids": sorted(
                    {
                        row["parent_span_id"]
                        for row in rows
                        if row["parent_span_id"] is not None
                    }
                ),
                "roles": sorted({row["role"] for row in rows}),
                "text": text,
                "text_sha256": sha256_bytes(text.encode("utf-8")),
                "qwen36_token_count": token_count(tokenizer, text),
                "rights_basis": rows[0]["rights_basis"],
                "safety_transfer_flags": safety,
                "lineage": {
                    "sealed_projection_record_ids": [
                        row["projection_record_id"] for row in rows
                    ],
                    "source_document_identity_sha256": rows[0][
                        "source_document_identity_sha256"
                    ],
                    "markdown_sha256": rows[0]["markdown_sha256"],
                    "mixed_source_paths_dereferenced": False,
                },
                "url_memorization_surface_removed": True,
            }
        )
    if len(contexts) != len(grouped):
        raise RuntimeError("source-context build lost a source group")
    return contexts


def _paragraph_candidates(site_rows: Sequence[dict], tokenizer: Any) -> list[dict]:
    candidates = []
    seen_text: set[str] = set()
    for row in site_rows:
        paragraphs = re.split(r"\n\s*\n", row["sanitized_text"].strip())
        for paragraph_index, paragraph in enumerate(paragraphs):
            text = paragraph.strip() + "\n"
            count = token_count(tokenizer, text)
            text_sha = sha256_bytes(text.encode("utf-8"))
            if (
                count < RAW_MIN_PARAGRAPH_TOKENS
                or count > RAW_MAX_PARAGRAPH_TOKENS
                or text_sha in seen_text
            ):
                continue
            seen_text.add(text_sha)
            identity = {
                "source_group_id": row["source_group_id"],
                "span_id": row["span_id"],
                "paragraph_index": paragraph_index,
                "text_sha256": text_sha,
            }
            candidates.append(
                {
                    "candidate_id": content_id("raw-candidate-v1", identity),
                    "resource_id": row["resource_id"],
                    "source_group_id": row["source_group_id"],
                    "span_id": row["span_id"],
                    "parent_span_id": row["parent_span_id"],
                    "artifact_id": row["artifact_id"],
                    "role": row["role"],
                    "paragraph_index": paragraph_index,
                    "text": text,
                    "text_sha256": text_sha,
                    "qwen36_token_count": count,
                    "rights_basis": row["rights_basis"],
                    "safety_transfer_flags": row["safety_transfer_flags"],
                    "lineage": {
                        "sealed_projection_record_id": row["projection_record_id"],
                        "source_document_identity_sha256": row[
                            "source_document_identity_sha256"
                        ],
                        "mixed_source_path_dereferenced": False,
                    },
                }
            )
    return candidates


def _source_balanced_prefix(
    candidates: Sequence[dict], target: int
) -> tuple[list[dict], list[dict]]:
    """Cover every source group with useful prose, then equalize token loads."""

    def seed_quality(item: dict) -> tuple[int, int, int, str]:
        text = item.get("text", "").strip()
        heading_only = bool(re.fullmatch(r"#{1,6}\s+[^\n]+", text))
        count = item["qwen36_token_count"]
        preferred_band_penalty = 0 if 24 <= count <= 96 else 1
        return (
            int(heading_only),
            preferred_band_penalty,
            abs(count - 56),
            item["candidate_id"],
        )

    by_group: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_group[candidate["source_group_id"]].append(candidate)
    for rows in by_group.values():
        rows.sort(key=seed_quality)
    if not by_group:
        raise RuntimeError("raw continuation has no source groups")

    selected = [rows[0] for _, rows in sorted(by_group.items())]
    selected_tokens = sum(item["qwen36_token_count"] for item in selected)
    if selected_tokens > target:
        raise RuntimeError(
            "raw source-group coverage exceeds the balanced-prefix budget: "
            f"{selected_tokens}/{target}"
        )
    selected_ids = {item["candidate_id"] for item in selected}
    if len(selected_ids) != len(selected):
        raise RuntimeError("raw source-group seed candidates are duplicated")

    heap: list[tuple[int, str, str, int]] = []
    per_group: Counter[str] = Counter(
        {
            group_id: rows[0]["qwen36_token_count"]
            for group_id, rows in by_group.items()
        }
    )
    for group_id, rows in sorted(by_group.items()):
        if len(rows) > 1:
            heapq.heappush(
                heap,
                (per_group[group_id], canonical_sha256(group_id), group_id, 1),
            )
    while heap and selected_tokens < target:
        _, group_hash, group_id, index = heapq.heappop(heap)
        rows = by_group[group_id]
        if index >= len(rows):
            continue
        candidate = rows[index]
        count = candidate["qwen36_token_count"]
        if selected_tokens + count <= target:
            selected.append(candidate)
            selected_ids.add(candidate["candidate_id"])
            selected_tokens += count
            per_group[group_id] += count
        next_index = index + 1
        if next_index < len(rows):
            heapq.heappush(
                heap,
                (
                    per_group[group_id],
                    group_hash,
                    group_id,
                    next_index,
                ),
            )
    remaining = [
        candidate
        for candidate in candidates
        if candidate["candidate_id"] not in selected_ids
    ]
    return selected, remaining


def _subset_fill(candidates: Sequence[dict], target: int) -> tuple[list[dict], int]:
    if target <= 0:
        return [], 0
    eligible = sorted(
        (
            item
            for item in candidates
            if item["qwen36_token_count"] <= target
        ),
        key=lambda item: item["candidate_id"],
    )
    mask = (1 << (target + 1)) - 1
    reachable = 1
    states = [reachable]
    for item in eligible:
        reachable |= (
            reachable << item["qwen36_token_count"]
        ) & mask
        states.append(reachable)
    achieved = target if (reachable >> target) & 1 else reachable.bit_length() - 1
    selected = []
    cursor = achieved
    for index in range(len(eligible) - 1, -1, -1):
        previous = states[index]
        if not ((previous >> cursor) & 1):
            item = eligible[index]
            selected.append(item)
            cursor -= item["qwen36_token_count"]
    if cursor != 0:
        raise RuntimeError("raw continuation subset reconstruction failed")
    return selected, achieved


def select_raw_continuation(site_rows: Sequence[dict], tokenizer: Any) -> list[dict]:
    candidates = _paragraph_candidates(site_rows, tokenizer)
    candidate_groups = {item["source_group_id"] for item in candidates}
    site_groups = {item["source_group_id"] for item in site_rows}
    if candidate_groups != site_groups:
        raise RuntimeError("raw paragraph extraction cannot represent every source group")
    prefix, remaining = _source_balanced_prefix(
        candidates, RAW_SOURCE_BALANCED_PREFIX_TOKENS
    )
    prefix_tokens = sum(item["qwen36_token_count"] for item in prefix)
    tail, achieved = _subset_fill(remaining, RAW_TARGET_TOKENS - prefix_tokens)
    selected = prefix + tail
    total = prefix_tokens + achieved
    if total != RAW_TARGET_TOKENS:
        raise RuntimeError(
            f"raw continuation target cannot be met without padding: {total}/{RAW_TARGET_TOKENS}"
        )
    rows = []
    for item in selected:
        row = dict(item)
        row.pop("candidate_id")
        row["schema"] = "high-information-raw-continuation-v1"
        row["record_id"] = content_id(
            "raw-continuation-v1",
            {
                "source_group_id": row["source_group_id"],
                "span_id": row["span_id"],
                "paragraph_index": row["paragraph_index"],
                "text_sha256": row["text_sha256"],
            },
        )
        row["training_family"] = "raw_domain_continuation"
        row["training_format"] = "causal_next_token_text"
        row["assistant_supervision"] = False
        row["url_memorization_surface_removed"] = True
        rows.append(row)
    rows.sort(key=lambda item: item["record_id"])
    if len({row["record_id"] for row in rows}) != len(rows):
        raise RuntimeError("raw continuation records are duplicated")
    if {row["source_group_id"] for row in rows} != site_groups:
        raise RuntimeError("raw continuation selection lost a train source group")
    return rows


def _allocate_budget(
    contexts: Sequence[dict], *, target: int, family: str
) -> dict[str, int]:
    if family == "closed_book_application":
        minimum = 64
        capacities = {
            item["context_id"]: max(
                minimum, min(1024, item["qwen36_token_count"] * 2)
            )
            for item in contexts
        }
    elif family == "grounded_synthesis":
        minimum = 48
        capacities = {
            item["context_id"]: max(
                minimum, min(512, item["qwen36_token_count"])
            )
            for item in contexts
        }
    else:
        raise RuntimeError(f"unsupported generation family: {family}")
    if minimum * len(contexts) > target or sum(capacities.values()) < target:
        raise RuntimeError(
            f"{family} target exceeds non-padding source capacity: "
            f"target={target} capacity={sum(capacities.values())}"
        )
    allocation = {item["context_id"]: minimum for item in contexts}
    weights = {
        item["context_id"]: max(1, math.isqrt(item["qwen36_token_count"]))
        for item in contexts
    }
    remaining = target - sum(allocation.values())
    while remaining:
        eligible = [
            context_id
            for context_id in sorted(allocation)
            if allocation[context_id] < capacities[context_id]
        ]
        if not eligible:
            raise RuntimeError(f"{family} budget allocator exhausted capacity")
        total_weight = sum(weights[context_id] for context_id in eligible)
        progress = 0
        fractional = []
        for context_id in eligible:
            numerator = remaining * weights[context_id]
            share = numerator // total_weight
            room = capacities[context_id] - allocation[context_id]
            granted = min(room, share)
            if granted:
                allocation[context_id] += granted
                progress += granted
            fractional.append((-(numerator % total_weight), context_id))
        remaining -= progress
        if remaining and progress == 0:
            for _, context_id in sorted(fractional):
                if allocation[context_id] >= capacities[context_id]:
                    continue
                allocation[context_id] += 1
                remaining -= 1
                if not remaining:
                    break
    if sum(allocation.values()) != target:
        raise RuntimeError(f"{family} allocation did not reach its exact target")
    return allocation


def build_prompt_spec() -> dict:
    value = {
        "schema": "high-information-generation-prompt-spec-v1",
        "status": "generation_and_semantic_verification_pending",
        "objective": (
            "derive dense factual, application, comparison, mechanism, grounded-"
            "synthesis, and calibrated-negative supervision without URL memorization"
        ),
        "input_contract": {
            "context_schema": "high-information-source-context-v1",
            "contexts_are_train_only": True,
            "mixed_source_lineage_paths_must_not_be_dereferenced": True,
            "source_group_and_safety_lineage_must_be_copied": True,
            "chat_template": "official_qwen_checkpoint_template",
            "assistant_mask_method": (
                ASSISTANT_MASK_METHOD
            ),
            "enable_thinking": False,
            "zero_assistant_mask_must_fail": True,
        },
        "families": {
            "closed_book_application": {
                "eventual_training_prompt_contains_source_context": False,
                "subtype_mix": {
                    "direct_explanation": 0.30,
                    "application_scenario": 0.35,
                    "comparison_or_mechanism": 0.20,
                    "misconception_correction": 0.15,
                },
                "requirements": [
                    "answer only facts supported by the source context",
                    "make questions useful without mentioning pages, URLs, or source location",
                    "include exact evidence excerpts for verification only",
                    "do not emit chain-of-thought, scratchpads, or hidden reasoning",
                    "preserve uncertainty, scope, attribution, and safety-transfer limits",
                ],
            },
            "grounded_synthesis": {
                "eventual_training_prompt_contains_source_context": True,
                "subtype_mix": {
                    "evidence_grounded_answer": 0.45,
                    "multi_fact_synthesis": 0.30,
                    "conflict_or_scope_resolution": 0.15,
                    "calibrated_unanswerable": 0.10,
                },
                "requirements": [
                    "cite evidence by quoted text or opaque evidence ID, never by URL",
                    "distinguish source claim from established consensus",
                    "do not infer missing visual procedures or hidden steps",
                    "do not emit chain-of-thought, scratchpads, or hidden reasoning",
                    "preserve every applicable safety-transfer flag",
                ],
            },
            "calibrated_hard_negative": {
                "allowed_types": [
                    "answer_absent_from_context",
                    "false_premise",
                    "unsupported_precision_or_threshold",
                    "source_scope_or_authority_mismatch",
                    "conflicting_or_insufficient_evidence",
                ],
                "requirements": [
                    "state the uncertainty or false premise directly",
                    "do not invent a correction not supported by context",
                    "do not turn historical, industrial, medical, or engineering claims into universal rules",
                    "include evidence excerpts demonstrating the limitation",
                ],
            },
        },
        "candidate_output_schema": {
            "request_id": "string",
            "examples_count": 4,
            "examples": [
                {
                    "example_type": "string",
                    "question": "string",
                    "answer": "string",
                    "evidence_quotes": ["exact context substring"],
                    "negative_type": "string-or-null",
                }
            ],
        },
        "deterministic_rejection_rules": [
            "URL, domain, website, canonical-link, or location memorization surface",
            "ChatML/thinking tokens, chain-of-thought, scratchpad, or hidden reasoning",
            "empty or duplicated question/answer",
            "evidence quote absent from the sealed train-derived context",
            "source-group or request identity mismatch",
            "missing calibrated-negative type when calibration mode is requested",
        ],
        "semantic_verification_required": [
            "answer_entailment",
            "application_correctness",
            "hard_negative_unanswerability_or_false_premise",
            "safety_scope_and_attribution_preservation",
        ],
        "candidate_count_per_request": 4,
        "candidate_count_unit": "examples_in_one_structured_generation_record",
        "hidden_reasoning_training_rows_allowed": False,
        "url_memorization_training_rows_allowed": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def build_semantic_verifier_contract() -> dict:
    semantic_validator = ROOT / "verify_high_information_semantic_decisions_v1.py"
    value = {
        "schema": "high-information-semantic-verifier-contract-v1",
        "status": "pinned_independent_verifier_authority_pending",
        "generator_model_identity_sha256": GENERATOR_MODEL_IDENTITY_SHA256,
        "pinned_verifier_authority_sha256": None,
        "decision_validator": {
            "path": relative(semantic_validator),
            "file_sha256": file_sha256(semantic_validator),
        },
        "input_contract": {
            "structural_review_schema": (
                "high-information-candidate-structural-review-v1"
            ),
            "candidate_status_required": (
                "structurally_valid_semantic_verification_pending"
            ),
            "candidate_must_be_training_ineligible": True,
            "context_must_match_manifest_receipt": True,
            "mixed_source_lineage_paths_must_not_be_opened": True,
        },
        "verifier_authority": {
            "must_be_content_addressed_and_pinned_before_use": True,
            "must_be_independent_of_generator": True,
            "generator_identity_may_not_verify_itself": True,
            "allowed_authority_kinds": [
                "dual_human_consensus",
                "pinned_independent_entailment_pipeline",
                "deterministic_task_verifier_plus_independent_entailment",
            ],
            "opaque_single_llm_judge_is_sufficient": False,
            "package_model_prompt_and_rule_versions_must_be_pinned": True,
            "required_consensus_components": {
                "guided_json_judge": {
                    "model": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
                    "revision": "95a6d26c4bfb886c58daf9d3f7332c857cb27b43",
                    "independent_prompt_passes": 2,
                    "model_authored_final_verdict_allowed": False,
                    "requested_facet_set_fixed_by_deterministic_code": True,
                },
                "nli_prefilter": {
                    "model": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                    "revision": "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7",
                    "sole_acceptance_authority": False,
                    "nonpass_requires_rejection_or_manual_review": True,
                },
                "manual_review": {
                    "all_uncertainty_or_component_disagreement": True,
                    "deterministic_stratified_high_risk_sample": True,
                    "unresolved_review_row_training_eligible": False,
                },
            },
        },
        "required_gates": {
            "exact_evidence_quote_match": {
                "method": "deterministic_exact_substring_and_sha256",
                "required_verdict": "pass",
                "not_applicable_allowed": False,
            },
            "source_entailment": {
                "method": "two_pass_guided_evidence_consensus_plus_independent_nli",
                "required_verdict": "pass",
                "not_applicable_allowed": False,
                "unsupported_claims_fail": True,
            },
            "citation_support_coverage": {
                "method": "claim_to_exact_quote_coverage",
                "required_verdict": "pass",
                "not_applicable_allowed": False,
            },
            "question_answer_completeness": {
                "method": (
                    "deterministic_requested_facet_set_plus_two_pass_exact_answer_span_"
                    "and_cited_evidence_mapping"
                ),
                "required_verdict": "pass",
                "not_applicable_allowed": False,
                "partial_answers_fail": True,
                "coordination_examples": [
                    "when and where requires both time and place",
                    "compare A and B requires both sides and the requested contrast",
                    "how and why requires both procedure and rationale",
                ],
            },
            "application_correctness": {
                "method": "scenario_constraints_and_domain_prerequisites",
                "required_for_subtypes": ["application_scenario"],
                "otherwise_required_verdict": "not_applicable",
            },
            "hard_negative_calibration": {
                "method": "absence_false_premise_or_scope_check",
                "required_for_generation_mode": "calibrated_hard_negative",
                "otherwise_required_verdict": "not_applicable",
            },
            "safety_transfer_preservation": {
                "method": "all_context_safety_flags_and_omitted_steps_checked",
                "required_verdict": "pass",
                "not_applicable_allowed": False,
            },
            "attribution_and_scope_preservation": {
                "method": "source_claim_consensus_and_scope_distinguished",
                "required_verdict": "pass",
                "not_applicable_allowed": False,
            },
            "unsupported_claim_absence": {
                "method": "independent_claim_inventory_against_context",
                "required_verdict": "pass",
                "not_applicable_allowed": False,
            },
            "training_value_and_nontriviality": {
                "method": "domain_knowledge_value_relative_to_same_source_group",
                "required_verdict": "pass",
                "not_applicable_allowed": False,
                "isolated_trivia_fails_when_higher_information_candidate_exists": True,
                "low_value_examples": [
                    "URL or canonical-link recall",
                    "isolated date or venue recall without lineage significance",
                    "public-identity trivia without technique, history, or domain context",
                ],
            },
        },
        "post_verification_selection_gates": {
            "exact_and_normalized_duplicate_control": {
                "method": (
                    "deterministic Unicode-normalized question-answer hash over the "
                    "entire candidate corpus"
                ),
                "per_row_judge_may_not_self_certify": True,
                "duplicate_training_rows_allowed": False,
            },
            "near_duplicate_and_information_preference": {
                "method": (
                    "source-group-aware lexical or embedding clustering followed by "
                    "deterministic preference for complete higher-information examples"
                ),
                "per_row_judge_may_not_self_certify": True,
                "isolated_trivia_deprioritized": True,
            },
            "manual_review": {
                "all_uncertain_rows_required": True,
                "deterministic_stratified_high_risk_sample_required": True,
                "stratified_sample_of_other_accepts_required": True,
                "unresolved_rows_training_eligible": False,
            },
        },
        "decision_schema": {
            "exact_fields": [
                "schema",
                "packet_id",
                "candidate_example_id",
                "verifier_authority_sha256",
                "gate_results",
                "evidence_quote_sha256s",
                "content_sha256_before_self_field",
            ],
            "gate_result_exact_fields": ["verdict", "method", "evidence_sha256s"],
            "allowed_verdicts": ["pass", "fail", "not_applicable"],
            "free_form_chain_of_thought_allowed": False,
        },
        "receipt_schema": {
            "schema": "high-information-semantic-verification-receipt-v1",
            "must_commit": [
                "plan_manifest_sha256",
                "structural_review_sha256",
                "semantic_packet_sha256",
                "decision_file_sha256",
                "verifier_authority_sha256",
                "accepted_candidate_id_commitment_sha256",
                "rejected_candidate_id_commitment_sha256",
                "per_gate_counts",
            ],
            "semantic_verification_completed_requires_all_packets_decided": True,
            "structural_pass_alone_may_be_accepted": False,
            "training_eligibility_requires_every_applicable_gate_pass": True,
            "exact_target_selection_occurs_after_semantic_verification": True,
        },
        "current_gate": (
            "no semantic decision may be accepted until an independent verifier "
            "authority is content-addressed and its SHA-256 is pinned in this contract"
        ),
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def _request_subtype(context_id: str, family: str, spec: dict) -> str:
    mix = spec["families"][family]["subtype_mix"]
    value = int(hashlib.sha256(f"{context_id}\0{family}".encode()).hexdigest()[:16], 16)
    fraction = value / (1 << 64)
    cumulative = 0.0
    for subtype, weight in mix.items():
        cumulative += weight
        if fraction < cumulative:
            return subtype
    return next(reversed(mix))


def _carve_hard_negative_budgets(
    base: Sequence[dict], target: int
) -> dict[tuple[str, str], int]:
    # Each request must support four useful calibrated answers.  Partition the
    # exact target into 48-64-token request budgets instead of leaving tiny,
    # impossible residual requests merely to hit an arithmetic total.
    request_count = math.ceil(target / 64)
    if request_count <= 0 or target < request_count * 48:
        raise RuntimeError("hard-negative target cannot form four-answer requests")
    base_tokens, extra = divmod(target, request_count)
    budgets = [base_tokens + (index < extra) for index in range(request_count)]
    if min(budgets) < 48 or max(budgets) > 64 or sum(budgets) != target:
        raise RuntimeError("hard-negative exact budget partition failed")
    eligible = sorted(
        (
            item
            for item in base
            if item["target_tokens"] - 32 >= 64
        ),
        key=lambda item: canonical_sha256(
            [item["context_id"], item["family"], "hard-negative-carve"]
        ),
    )
    if len(eligible) < request_count:
        raise RuntimeError("hard-negative target exceeds safe request carve capacity")
    allocation = {
        (item["context_id"], item["family"]): budget
        for item, budget in zip(eligible[:request_count], budgets, strict=True)
    }
    return allocation


def build_generation_requests(
    contexts: Sequence[dict],
    seed_qa_assistant_tokens: int,
    prompt_spec: dict,
) -> list[dict]:
    closed_generation_target = (
        CLOSED_BOOK_TARGET_ASSISTANT_TOKENS - seed_qa_assistant_tokens
    )
    if closed_generation_target <= 0:
        raise RuntimeError("seed QA unexpectedly exceeds closed-book target")
    closed = _allocate_budget(
        contexts, target=closed_generation_target, family="closed_book_application"
    )
    grounded = _allocate_budget(
        contexts, target=GROUNDED_TARGET_ASSISTANT_TOKENS, family="grounded_synthesis"
    )
    base = []
    for context in contexts:
        context_id = context["context_id"]
        for family, allocation in (
            ("closed_book_application", closed),
            ("grounded_synthesis", grounded),
        ):
            base.append(
                {
                    "context_id": context_id,
                    "source_group_id": context["source_group_id"],
                    "family": family,
                    "subtype": _request_subtype(context_id, family, prompt_spec),
                    "target_tokens": allocation[context_id],
                    "context_qwen36_tokens": context["qwen36_token_count"],
                }
            )
    hard_negative_target = (
        (CLOSED_BOOK_TARGET_ASSISTANT_TOKENS + GROUNDED_TARGET_ASSISTANT_TOKENS)
        * HARD_NEGATIVE_NUMERATOR
        // HARD_NEGATIVE_DENOMINATOR
    )
    carved = _carve_hard_negative_budgets(base, hard_negative_target)
    spec_sha = prompt_spec["content_sha256_before_self_field"]
    requests = []
    for item in base:
        key = (item["context_id"], item["family"])
        negative_tokens = carved.get(key, 0)
        positive_tokens = item["target_tokens"] - negative_tokens
        modes = [("positive", positive_tokens)]
        if negative_tokens:
            modes.append(("calibrated_hard_negative", negative_tokens))
        for mode, target_tokens in modes:
            identity = {
                "prompt_spec_sha256": spec_sha,
                "context_id": item["context_id"],
                "source_group_id": item["source_group_id"],
                "family": item["family"],
                "subtype": item["subtype"],
                "mode": mode,
                "target_verified_assistant_tokens": target_tokens,
            }
            requests.append(
                {
                    "schema": "high-information-generation-request-v1",
                    "request_id": content_id("generation-request-v1", identity),
                    "prompt_spec_sha256": spec_sha,
                    "source_context_id": item["context_id"],
                    "source_group_id": item["source_group_id"],
                    "task_family": item["family"],
                    "task_subtype": item["subtype"],
                    "generation_mode": mode,
                    "target_verified_assistant_tokens": target_tokens,
                    "source_context_qwen36_tokens": item[
                        "context_qwen36_tokens"
                    ],
                    "candidate_count": prompt_spec["candidate_count_per_request"],
                    "source_entailment_verification_required": True,
                    "safety_transfer_verification_required": True,
                    "hidden_reasoning_allowed": False,
                    "url_memorization_allowed": False,
                    "status": "generation_pending",
                }
            )
    if sum(row["target_verified_assistant_tokens"] for row in requests) != (
        closed_generation_target + GROUNDED_TARGET_ASSISTANT_TOKENS
    ):
        raise RuntimeError("generation request budgets do not sum to target")
    if sum(
        row["target_verified_assistant_tokens"]
        for row in requests
        if row["generation_mode"] == "calibrated_hard_negative"
    ) != hard_negative_target:
        raise RuntimeError("hard-negative request budget drifted")
    return sorted(requests, key=lambda item: item["request_id"])


def shard_requests(requests: Sequence[dict]) -> tuple[list[list[dict]], list[int]]:
    shards: list[list[dict]] = [[] for _ in range(GPU_SHARDS)]
    loads = [0] * GPU_SHARDS
    ordered = sorted(
        requests,
        key=lambda item: (-item["target_verified_assistant_tokens"], item["request_id"]),
    )
    for request in ordered:
        shard = min(range(GPU_SHARDS), key=lambda index: (loads[index], index))
        row = dict(request)
        row["gpu_shard"] = shard
        shards[shard].append(row)
        loads[shard] += row["target_verified_assistant_tokens"]
    for rows in shards:
        rows.sort(key=lambda item: item["request_id"])
    if max(loads) - min(loads) > max(
        request["target_verified_assistant_tokens"] for request in requests
    ):
        raise RuntimeError("four-GPU request sharding is unnecessarily imbalanced")
    return shards, loads


def _json_payload(value: dict) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _artifact_receipt(path: Path, payload: bytes, rows: int | None = None) -> dict:
    value = {
        "path": relative(path),
        "file_sha256": sha256_bytes(payload),
        "file_bytes": len(payload),
    }
    if rows is not None:
        value["rows"] = rows
    return value


def construct() -> tuple[dict, dict[Path, bytes]]:
    tokenizer = load_tokenizer()
    site_rows = load_site_rows(tokenizer)
    seed_qa, seed_summary = load_seed_qa(tokenizer)
    contexts = build_source_contexts(site_rows, tokenizer)
    raw_rows = select_raw_continuation(site_rows, tokenizer)
    prompt_spec = build_prompt_spec()
    semantic_contract = build_semantic_verifier_contract()
    requests = build_generation_requests(
        contexts,
        seed_summary["included_assistant_qwen36_tokens"],
        prompt_spec,
    )
    shards, shard_loads = shard_requests(requests)

    prompt_payload = _json_payload(prompt_spec)
    semantic_contract_payload = _json_payload(semantic_contract)
    context_payload = jsonl_payload(contexts)
    raw_payload = jsonl_payload(raw_rows)
    seed_payload = jsonl_payload(seed_qa)
    artifact_payloads = {
        PROMPT_SPEC: prompt_payload,
        SEMANTIC_VERIFIER_CONTRACT: semantic_contract_payload,
        SOURCE_CONTEXTS: context_payload,
        RAW_CONTINUATION: raw_payload,
        SEED_QA: seed_payload,
    }
    shard_receipts = []
    for path, rows in zip(REQUEST_SHARDS, shards, strict=True):
        payload = jsonl_payload(rows)
        artifact_payloads[path] = payload
        shard_receipts.append(_artifact_receipt(path, payload, len(rows)))

    site_group_ids = {row["source_group_id"] for row in site_rows}
    context_group_ids = {row["source_group_id"] for row in contexts}
    requested_group_ids = {row["source_group_id"] for row in requests}
    qa_group_ids = {row["source_group_id"] for row in seed_qa}
    if context_group_ids != site_group_ids or requested_group_ids != site_group_ids:
        raise RuntimeError("generation plan does not represent every train site source group")
    raw_tokens = sum(row["qwen36_token_count"] for row in raw_rows)
    hard_negative_tokens = sum(
        row["target_verified_assistant_tokens"]
        for row in requests
        if row["generation_mode"] == "calibrated_hard_negative"
    )
    total_assistant_target = (
        CLOSED_BOOK_TARGET_ASSISTANT_TOKENS + GROUNDED_TARGET_ASSISTANT_TOKENS
    )
    generation_target = sum(
        row["target_verified_assistant_tokens"] for row in requests
    )
    value = {
        "schema": SCHEMA,
        "status": "raw_selection_and_generation_plan_complete_candidates_pending",
        "purpose": (
            "deterministic train-only high-information corpus plan; not accepted "
            "candidate data and not training authorization"
        ),
        "sealed_train_inputs": [
            {
                "path": relative(SITE_INPUT),
                "file_sha256": file_sha256(SITE_INPUT),
                "rows": len(site_rows),
                "source_groups": len(site_group_ids),
            },
            {
                "path": relative(QA_INPUT),
                "file_sha256": file_sha256(QA_INPUT),
                "rows": seed_summary["input_rows"],
                "source_groups": seed_summary["input_source_groups"],
                "source_groups_included_after_url_filter": len(qa_group_ids),
            },
        ],
        "input_boundary": {
            "only_sealed_train_projection_semantics_opened": True,
            "development_projection_opened": False,
            "final_or_protected_source_opened": False,
            "mixed_source_lineage_paths_dereferenced": False,
            "pending_49_qa_status": (
                "gated_no_explicit_safe_training_projection_in_sealed_input_directory"
            ),
        },
        "tokenizer": {
            "model_family": "Qwen3.6-35B-A3B",
            "tokenizer_json_path": relative(TOKENIZER_JSON),
            "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
            "tokenizer_config_path": relative(TOKENIZER_CONFIG),
            "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
            "add_special_tokens": False,
            "chat_template": "official_checkpoint_template",
            "assistant_mask_method": (
                ASSISTANT_MASK_METHOD
            ),
            "enable_thinking": False,
            "zero_assistant_mask_must_fail": True,
            "mask_implementation": {
                "path": relative(ROOT / "qwen_chat_masking_v1.py"),
                "file_sha256": file_sha256(ROOT / "qwen_chat_masking_v1.py"),
            },
        },
        "targets": {
            "closed_book_application_assistant_tokens": (
                CLOSED_BOOK_TARGET_ASSISTANT_TOKENS
            ),
            "grounded_synthesis_assistant_tokens": GROUNDED_TARGET_ASSISTANT_TOKENS,
            "raw_continuation_tokens": RAW_TARGET_TOKENS,
            "hard_negative_target_fraction": (
                HARD_NEGATIVE_NUMERATOR / HARD_NEGATIVE_DENOMINATOR
            ),
        },
        "materialized_seed_qa": {
            **seed_summary,
            "receipt": _artifact_receipt(SEED_QA, seed_payload, len(seed_qa)),
            "counts_toward_closed_book_target": True,
            "url_memorization_rows_excluded": True,
            "hidden_reasoning_rows_excluded": True,
        },
        "raw_continuation": {
            "receipt": _artifact_receipt(
                RAW_CONTINUATION, raw_payload, len(raw_rows)
            ),
            "qwen36_tokens": raw_tokens,
            "source_groups": len({row["source_group_id"] for row in raw_rows}),
            "source_resources": len({row["resource_id"] for row in raw_rows}),
            "selection": (
                "paragraph units; every source group seeded with a non-heading, "
                "information-bearing 24-96-token paragraph when available, then a "
                "90k fair token-load prefix across source groups plus deterministic "
                "exact subset fill; no padding or cross-document packing"
            ),
            "all_train_site_source_groups_represented": (
                {row["source_group_id"] for row in raw_rows} == site_group_ids
            ),
            "url_memorization_surface_removed": True,
        },
        "source_contexts": {
            "receipt": _artifact_receipt(
                SOURCE_CONTEXTS, context_payload, len(contexts)
            ),
            "source_groups": len(contexts),
            "all_train_site_source_groups_represented": True,
            "rights_safety_and_lineage_preserved": True,
            "mixed_source_paths_included": False,
        },
        "prompt_spec": _artifact_receipt(PROMPT_SPEC, prompt_payload),
        "semantic_verifier_contract": _artifact_receipt(
            SEMANTIC_VERIFIER_CONTRACT, semantic_contract_payload
        ),
        "generation_requests": {
            "rows": len(requests),
            "source_groups": len(requested_group_ids),
            "all_train_site_source_groups_represented": True,
            "planned_closed_book_generation_assistant_tokens": (
                CLOSED_BOOK_TARGET_ASSISTANT_TOKENS
                - seed_summary["included_assistant_qwen36_tokens"]
            ),
            "planned_grounded_generation_assistant_tokens": (
                GROUNDED_TARGET_ASSISTANT_TOKENS
            ),
            "planned_generation_assistant_tokens": generation_target,
            "planned_hard_negative_assistant_tokens": hard_negative_tokens,
            "planned_hard_negative_fraction_of_all_assistant_targets": (
                hard_negative_tokens / total_assistant_target
            ),
            "candidate_count_per_request": prompt_spec[
                "candidate_count_per_request"
            ],
            "gpu_shards": shard_receipts,
            "gpu_shard_target_token_loads": shard_loads,
            "generation_completed": False,
            "semantic_verification_completed": False,
        },
        "candidate_worker": {
            "path": relative(ROOT / "run_high_information_candidate_shard_v1.py"),
            "file_sha256": file_sha256(
                ROOT / "run_high_information_candidate_shard_v1.py"
            ),
            "runtime": "one vLLM 0.25.0 BF16 engine per request/GPU shard",
            "generation_launched": False,
            "outputs_are_training_eligible": False,
        },
        "candidate_verifier": {
            "path": relative(ROOT / "verify_high_information_candidates_v1.py"),
            "file_sha256": file_sha256(
                ROOT / "verify_high_information_candidates_v1.py"
            ),
            "structural_pass_is_training_eligible": False,
            "semantic_verification_required": True,
        },
        "semantic_decision_validator": {
            "path": relative(
                ROOT / "verify_high_information_semantic_decisions_v1.py"
            ),
            "file_sha256": file_sha256(
                ROOT / "verify_high_information_semantic_decisions_v1.py"
            ),
            "pinned_independent_authority_present": False,
            "can_accept_structural_only_passes": False,
        },
        "target_accounting": {
            "closed_book_seed_assistant_tokens": seed_summary[
                "included_assistant_qwen36_tokens"
            ],
            "closed_book_generation_target_assistant_tokens": (
                CLOSED_BOOK_TARGET_ASSISTANT_TOKENS
                - seed_summary["included_assistant_qwen36_tokens"]
            ),
            "closed_book_total_target_assistant_tokens": (
                CLOSED_BOOK_TARGET_ASSISTANT_TOKENS
            ),
            "grounded_total_target_assistant_tokens": (
                GROUNDED_TARGET_ASSISTANT_TOKENS
            ),
            "raw_continuation_materialized_tokens": raw_tokens,
            "targets_padded_or_fabricated": False,
            "target_status": (
                "raw target materialized exactly; assistant targets are request budgets "
                "pending candidate generation and verification"
            ),
        },
        "generation_blockers": [
            "no candidate generator has run over the four request shards",
            "no independent source-entailment and calibrated-negative verifier receipts exist",
            "accepted assistant-token totals cannot be claimed before candidate verification",
            "the 49 pending QA rows remain gated because no explicit sealed safe training projection is present",
        ],
        "invariants": {
            "url_memorization_training_rows_emitted": False,
            "synthetic_hidden_reasoning_training_rows_emitted": False,
            "cross_document_raw_packing_performed": False,
            "all_train_assigned_site_sources_represented": True,
            "all_v440_train_source_groups_accounted_for": (
                seed_summary["all_input_source_groups_accounted_for"]
            ),
            "v440_url_only_source_groups_excluded_from_training": (
                seed_summary["excluded_only_source_groups"]
            ),
            "development_final_protected_holdout_ood_terminal_incident_or_manual_review_data_opened": False,
            "mixed_source_lineage_paths_dereferenced": False,
            "training_launch_authorized": False,
        },
        "builder_receipt": {
            "path": relative(Path(__file__)),
            "file_sha256": file_sha256(Path(__file__)),
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value, artifact_payloads


def render_manifest(value: dict) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def build(*, check: bool = False) -> dict:
    value, artifact_payloads = construct()
    payload = render_manifest(value)
    if check:
        expected = {**artifact_payloads, MANIFEST: payload}
        stale = [
            path.as_posix()
            for path, expected_payload in expected.items()
            if not path.exists() or path.read_bytes() != expected_payload
        ]
        if stale:
            raise RuntimeError(
                "checked high-information corpus artifacts are stale: "
                + ", ".join(stale)
            )
        return value
    for path, artifact_payload in artifact_payloads.items():
        atomic_write(path, artifact_payload)
    atomic_write(MANIFEST, payload)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    value = build(check=arguments.check)
    print(
        json.dumps(
            {
                "manifest": relative(MANIFEST),
                "content_sha256": value["content_sha256_before_self_field"],
                "raw_tokens": value["raw_continuation"]["qwen36_tokens"],
                "seed_qa_assistant_tokens": value["materialized_seed_qa"][
                    "included_assistant_qwen36_tokens"
                ],
                "generation_requests": value["generation_requests"]["rows"],
                "generation_completed": False,
                "training_launch_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
