#!/usr/bin/env python3
"""Build the immutable source-group split authority for domain training data.

The builder has a deliberately narrow read boundary:

* the 33 Markdown artifacts and manifests named by the public site registry;
* the four public multi-page manifests needed to recover page/section identity;
* the selected V440 projection once, before sealing, so its nonsemantic source
  identities can drive assignment while full train/development rows can be
  tokenized and projected.

No pre-existing protected, holdout, OOD, terminal, incident, or manual-review
source is opened.  V440 semantic fields are read only during the one-time
pre-seal tokenization/projection pass; split assignment uses only nonsemantic
identity.  Source groups are reconstructed before any chunk or QA derivation,
exact and near duplicates are joined into connected components, and a stable
content-addressed 80/10/10 assignment is made at component level.  Train and
development membership is materialized.  Final membership is represented only
by aggregate counts and cryptographic commitments in the public artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import unicodedata
import urllib.parse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


ROOT = Path(__file__).resolve().parent
REGISTRY = (
    ROOT / "data/site_corpora/registry/site_corpus_registry_v1.json"
).resolve()
V440_MANIFEST = (
    ROOT
    / "experiments/sft_controls/v53a_train_refresh_v440_fold3/manifest_v53a.json"
).resolve()
V440_PROJECTION = (
    ROOT
    / "experiments/sft_controls/v53a_train_refresh_v440_fold3/train_projection_v440.jsonl"
).resolve()
OUTPUT = (
    ROOT / "data/training_inventory/source_group_split_authority_v1.json"
).resolve()
PROJECTION_DIR = (
    ROOT / "data/training_inventory/source_group_split_v1"
).resolve()
PROJECTION_PATHS = {
    "train": {
        "site_spans": (PROJECTION_DIR / "train_site_spans.jsonl").resolve(),
        "v440_qa": (PROJECTION_DIR / "train_v440_qa.jsonl").resolve(),
    },
    "development": {
        "site_spans": (PROJECTION_DIR / "development_site_spans.jsonl").resolve(),
        "v440_qa": (PROJECTION_DIR / "development_v440_qa.jsonl").resolve(),
    },
}

SCHEMA = "specialist-source-group-split-authority-v1"
SPLIT_NAMESPACE = "specialist-low-regression-source-split-v1"
PRE_DISCLOSURE_CORRECTION_AUTHORITY_SHA256 = (
    "d9753b00b41bf68ff1c6039ce3200ca389d54730d14b0af5ea195d3db60cf25d"
)
SPECIAL_RESOURCES = frozenset(
    {"crash_restraint", "rope365", "rope_topia", "shibari_atlas"}
)
EXPECTED_ARTIFACT_COUNT = 33
EXPECTED_SPECIAL_COUNTS = {
    "crash_restraint": {"included_pages": 116, "h2": 116},
    "shibari_atlas": {"included_pages": 1714, "h3": 1714},
    "rope_topia": {"included_pages": 8, "h2_included": 8, "h2_excluded": 1},
    "rope365": {"included_pages": 111, "h2": 16, "h3": 58},
}
FORBIDDEN_PATH_TOKENS = (
    "protected",
    "holdout",
    "heldout",
    "ood",
    "terminal",
    "incident",
    "manual_review",
    "manual-review",
)
TRACKING_QUERY_KEYS = frozenset(
    {
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
        "ref_src",
    }
)
URL_LABELS = {
    "crash_restraint": re.compile(
        rb"(?m)^Source URL:[ \t]*(https?://\S+)[ \t]*$"
    ),
    "shibari_atlas": re.compile(
        rb"(?m)^Source URL:[ \t]*(https?://\S+)[ \t]*$"
    ),
    "rope_topia": re.compile(
        rb"(?m)^Original URL:[ \t]*(https?://\S+)[ \t]*$"
    ),
    "rope365": re.compile(rb"(?m)^Source:[ \t]*(https?://\S+)[ \t]*$"),
}
WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)?", re.UNICODE)
URL_IN_TEXT_RE = re.compile(r"https?://\S+", re.IGNORECASE)
MARKDOWN_PUNCTUATION_RE = re.compile(r"[#*_`~>|\[\](){}]+")
MIN_NEAR_DUPLICATE_TOKENS = 20
SHINGLE_WIDTH = 5
MINHASH_SIZE = 32
MINHASH_BAND_SIZE = 4
NEAR_DUPLICATE_JACCARD = 0.90
_MINHASH_PRIME = (1 << 61) - 1
_MINHASH_A = tuple(
    1 + int.from_bytes(hashlib.sha256(f"a:{index}".encode()).digest()[:8], "big")
    % (_MINHASH_PRIME - 1)
    for index in range(MINHASH_SIZE)
)
_MINHASH_B = tuple(
    int.from_bytes(hashlib.sha256(f"b:{index}".encode()).digest()[:8], "big")
    % _MINHASH_PRIME
    for index in range(MINHASH_SIZE)
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _content_address(prefix: str, value: Any) -> str:
    return f"{prefix}:{canonical_sha256(value)}"


def _relative(path: Path, root: Path = ROOT) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _assert_receipts_current(
    receipts: Sequence[dict], *, root: Path = ROOT
) -> None:
    for receipt in receipts:
        path = _assert_safe_registered_path(receipt.get("path"), root=root)
        if file_sha256(path) != receipt.get("file_sha256"):
            raise RuntimeError(
                f"content-addressed input changed during build: {_relative(path, root)}"
            )


def _resolve_safe_registered_path(path: str, *, root: Path = ROOT) -> Path:
    if not isinstance(path, str) or not path:
        raise RuntimeError("registered path is missing")
    candidate = (root / path).resolve()
    try:
        relative = candidate.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError(f"registered path escapes repository root: {path}") from error
    folded = relative.casefold()
    if any(token in folded for token in FORBIDDEN_PATH_TOKENS):
        raise RuntimeError(f"registered path crosses forbidden boundary: {relative}")
    return candidate


def _assert_safe_registered_path(path: str, *, root: Path = ROOT) -> Path:
    candidate = _resolve_safe_registered_path(path, root=root)
    relative = candidate.relative_to(root.resolve()).as_posix()
    if not candidate.is_file():
        raise RuntimeError(f"registered input is not a file: {relative}")
    return candidate


def _load_pinned_tokenizer(
    registry: dict, *, root: Path = ROOT
) -> tuple[Any, dict, list[dict]]:
    tokenizer_contract = registry.get("tokenizer")
    if not isinstance(tokenizer_contract, dict):
        raise RuntimeError("registry tokenizer contract is missing")
    expected = {
        "add_special_tokens",
        "model_family",
        "tokenizer_config_path",
        "tokenizer_config_sha256",
        "tokenizer_json_path",
        "tokenizer_json_sha256",
    }
    if set(tokenizer_contract) != expected:
        raise RuntimeError("registry tokenizer contract changed schema")
    if (
        tokenizer_contract["model_family"] != "Qwen3.6-35B-A3B"
        or tokenizer_contract["add_special_tokens"] is not False
    ):
        raise RuntimeError("source split requires the pinned Qwen3.6 tokenizer")
    tokenizer_path = _assert_safe_registered_path(
        tokenizer_contract["tokenizer_json_path"], root=root
    )
    config_path = _assert_safe_registered_path(
        tokenizer_contract["tokenizer_config_path"], root=root
    )
    tokenizer_sha = file_sha256(tokenizer_path)
    config_sha = file_sha256(config_path)
    if (
        tokenizer_sha != tokenizer_contract["tokenizer_json_sha256"]
        or config_sha != tokenizer_contract["tokenizer_config_sha256"]
    ):
        raise RuntimeError("pinned Qwen tokenizer content address changed")
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError(
            "tokenizers is required; run the one-time constructor with .venv/bin/python"
        ) from error
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    binding = {
        "model_family": tokenizer_contract["model_family"],
        "add_special_tokens": False,
        "tokenizer_json_path": _relative(tokenizer_path, root),
        "tokenizer_json_sha256": tokenizer_sha,
        "tokenizer_config_path": _relative(config_path, root),
        "tokenizer_config_sha256": config_sha,
        "site_span_count_method": (
            "tokenizers.Tokenizer.encode(exact_span_text, add_special_tokens=False)"
        ),
        "v440_qa_count_method": (
            "es_exact_Qwen_ChatML_prompt_plus_raw_answer_boundary_without_EOS"
        ),
    }
    receipts = [
        {"path": _relative(tokenizer_path, root), "file_sha256": tokenizer_sha},
        {"path": _relative(config_path, root), "file_sha256": config_sha},
    ]
    return tokenizer, binding, receipts


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def _specialist_prompt(question: str) -> str:
    return (
        "<|im_start|>system\n"
        "Answer the question briefly and factually. Return only the answer."
        "<|im_end|>\n<|im_start|>user\n"
        f"{question}<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


def _encode_v440_pair(tokenizer: Any, row: dict) -> tuple[int, int]:
    from qa_quality import qa_pair_from_record

    pair = qa_pair_from_record(row)
    if pair is None:
        raise RuntimeError("selected V440 row is not a supported QA pair")
    question, answer = pair
    prompt = _specialist_prompt(question)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    full_ids = tokenizer.encode(prompt + answer, add_special_tokens=False).ids
    if not prompt_ids or full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError("V440 Qwen prompt/answer token boundary changed")
    answer_count = len(full_ids) - len(prompt_ids)
    if answer_count <= 0:
        raise RuntimeError("V440 row has no answer tokens")
    return len(prompt_ids), answer_count


def _remove_dot_segments(path: str) -> str:
    output: list[str] = []
    for part in path.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if output:
                output.pop()
            continue
        output.append(part)
    return "/" + "/".join(output)


def _normalize_percent_encoding(path: str) -> str:
    # Decode only RFC 3986 unreserved bytes and uppercase all other escapes.
    unreserved = frozenset(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
    )

    def replace(match: re.Match[str]) -> str:
        byte = int(match.group(1), 16)
        character = chr(byte)
        return character if character in unreserved else f"%{byte:02X}"

    return re.sub(r"%([0-9A-Fa-f]{2})", replace, path)


def normalize_url(value: str) -> str:
    """Return the deterministic source-identity form of an HTTP(S) URL."""

    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("source URL is empty")
    parsed = urllib.parse.urlsplit(value.strip())
    if parsed.scheme.casefold() not in {"http", "https"}:
        raise RuntimeError(f"unsupported source URL scheme: {value!r}")
    if parsed.username is not None or parsed.password is not None:
        raise RuntimeError("credential-bearing source URL is forbidden")
    if not parsed.hostname:
        raise RuntimeError(f"source URL has no host: {value!r}")
    try:
        host = parsed.hostname.rstrip(".").encode("idna").decode("ascii").casefold()
        port = parsed.port
    except (UnicodeError, ValueError) as error:
        raise RuntimeError(f"source URL authority is invalid: {value!r}") from error
    # Public source identity treats HTTP and HTTPS spellings as one page identity.
    scheme = "https"
    netloc = host
    if port not in (None, 80, 443):
        netloc = f"{host}:{port}"
    path = _remove_dot_segments(_normalize_percent_encoding(parsed.path or "/"))
    path = re.sub(r"/{2,}", "/", path)
    if path != "/":
        path = path.rstrip("/")
    query = []
    for key, item in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True):
        folded = key.casefold()
        if folded.startswith("utm_") or folded in TRACKING_QUERY_KEYS:
            continue
        query.append((key, item))
    query.sort()
    return urllib.parse.urlunsplit(
        (scheme, netloc, path, urllib.parse.urlencode(query, doseq=True), "")
    )


def source_url_identity(value: str) -> str:
    return f"url-v1:{sha256_bytes(normalize_url(value).encode('utf-8'))}"


def _heading_matches(data: bytes, levels: set[int]) -> list[re.Match[bytes]]:
    matches = list(re.finditer(rb"(?m)^(#{1,6})[ \t]+[^\r\n]*(?:\r?\n|$)", data))
    return [match for match in matches if len(match.group(1)) in levels]


@dataclass
class Span:
    resource_id: str
    markdown_path: str
    start: int
    end: int
    role: str
    heading_level: int | None
    source_identity_sha256s: tuple[str, ...] = ()
    provenance_mapping: str = "not_applicable"
    parent_span_id: str | None = None
    content_sha256: str = ""
    span_id: str = ""
    qwen36_token_count: int | None = None

    @property
    def byte_length(self) -> int:
        return self.end - self.start


@dataclass
class SourceGroup:
    origin_kind: str
    resource_id: str | None
    spans: list[Span]
    source_identity_sha256s: set[str]
    content_sha256: str
    normalized_text_sha256: str | None
    shingles: frozenset[int]
    provenance_mapping: str
    descendant_fact_ids: list[str] = field(default_factory=list)
    descendant_prompt_token_count: int = 0
    descendant_answer_token_count: int = 0
    group_id: str = ""

    @property
    def byte_length(self) -> int:
        return sum(span.byte_length for span in self.spans)


@dataclass
class DocumentPlan:
    resource_id: str
    artifact_id: str
    markdown_path: str
    markdown_sha256: str
    byte_length: int
    spans: list[Span]
    groups: list[SourceGroup]
    excluded_spans: list[Span]
    source_identity_count: int
    construction: str


class UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in self.parent}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _make_span(
    *,
    resource_id: str,
    markdown_path: str,
    data: bytes,
    start: int,
    end: int,
    role: str,
    heading_level: int | None,
    source_identity_sha256s: Iterable[str] = (),
    provenance_mapping: str = "not_applicable",
    parent_span_id: str | None = None,
) -> Span:
    if not (0 <= start < end <= len(data)):
        raise RuntimeError(
            f"invalid byte span for {resource_id}: {start}:{end}/{len(data)}"
        )
    identities = tuple(sorted(set(source_identity_sha256s)))
    content_sha = sha256_bytes(data[start:end])
    identity = {
        "resource_id": resource_id,
        "markdown_path": markdown_path,
        "start": start,
        "end": end,
        "content_sha256": content_sha,
        "role": role,
        "source_identity_sha256s": identities,
    }
    return Span(
        resource_id=resource_id,
        markdown_path=markdown_path,
        start=start,
        end=end,
        role=role,
        heading_level=heading_level,
        source_identity_sha256s=identities,
        provenance_mapping=provenance_mapping,
        parent_span_id=parent_span_id,
        content_sha256=content_sha,
        span_id=_content_address("source-span-v1", identity),
    )


def _normalized_tokens(parts: Sequence[bytes]) -> tuple[str, ...]:
    text = b"\n".join(parts).decode("utf-8", errors="strict")
    text = unicodedata.normalize("NFKC", text).casefold()
    text = URL_IN_TEXT_RE.sub(" ", text)
    text = MARKDOWN_PUNCTUATION_RE.sub(" ", text)
    return tuple(match.group(0) for match in WORD_RE.finditer(text))


def _token_shingles(tokens: Sequence[str]) -> frozenset[int]:
    if len(tokens) < MIN_NEAR_DUPLICATE_TOKENS:
        return frozenset()
    values: set[int] = set()
    for index in range(len(tokens) - SHINGLE_WIDTH + 1):
        payload = "\x1f".join(tokens[index : index + SHINGLE_WIDTH]).encode("utf-8")
        values.add(int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big"))
    return frozenset(values)


def _framed_content_sha(parts: Sequence[bytes]) -> str:
    if len(parts) == 1:
        return sha256_bytes(parts[0])
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _make_group(
    *,
    origin_kind: str,
    resource_id: str | None,
    spans: Sequence[Span],
    data_by_path: dict[str, bytes],
    source_identity_sha256s: Iterable[str],
    provenance_mapping: str,
    descendant_fact_ids: Sequence[str] = (),
    descendant_prompt_token_count: int = 0,
    descendant_answer_token_count: int = 0,
    external_content_sha256: str | None = None,
) -> SourceGroup:
    parts = [data_by_path[span.markdown_path][span.start : span.end] for span in spans]
    tokens = _normalized_tokens(parts) if parts else ()
    normalized_text_sha = (
        sha256_bytes("\x1f".join(tokens).encode("utf-8")) if tokens else None
    )
    content_sha = external_content_sha256 or _framed_content_sha(parts)
    identities = set(source_identity_sha256s)
    group_payload = {
        "origin_kind": origin_kind,
        "resource_id": resource_id,
        "span_ids": [span.span_id for span in spans],
        "source_identity_sha256s": sorted(identities),
        "content_sha256": content_sha,
        "descendant_fact_ids_sha256": canonical_sha256(sorted(descendant_fact_ids)),
    }
    return SourceGroup(
        origin_kind=origin_kind,
        resource_id=resource_id,
        spans=list(spans),
        source_identity_sha256s=identities,
        content_sha256=content_sha,
        normalized_text_sha256=normalized_text_sha,
        shingles=_token_shingles(tokens),
        provenance_mapping=provenance_mapping,
        descendant_fact_ids=sorted(descendant_fact_ids),
        descendant_prompt_token_count=descendant_prompt_token_count,
        descendant_answer_token_count=descendant_answer_token_count,
        group_id=_content_address("source-group-v1", group_payload),
    )


def _segments(data: bytes, levels: set[int]) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]]]:
    matches = _heading_matches(data, levels)
    heading_segments: list[tuple[int, int, int]] = []
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for index, match in enumerate(matches):
        if cursor < match.start():
            gaps.append((cursor, match.start()))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(data)
        level = len(match.group(1))
        heading_segments.append((match.start(), end, level))
        cursor = end
    if not matches and data:
        gaps.append((0, len(data)))
    elif cursor < len(data):
        gaps.append((cursor, len(data)))
    return heading_segments, gaps


def _extract_source_urls(pattern: re.Pattern[bytes], payload: bytes) -> list[str]:
    result = []
    for match in pattern.findall(payload):
        result.append(normalize_url(match.decode("utf-8", errors="strict")))
    return result


def _manifest_included_urls(resource_id: str, manifest: dict) -> list[str]:
    if resource_id == "crash_restraint":
        values = [
            item["url"]
            for item in manifest.get("entries", [])
            if item.get("disposition") == "included"
        ]
    elif resource_id == "shibari_atlas":
        values = [
            item["url"]
            for item in manifest.get("canonical_inventory", {}).get("urls", [])
            if item.get("decision") == "include"
        ]
    elif resource_id == "rope_topia":
        values = [
            item["canonical_url"]
            for item in manifest.get("inventory", [])
            if item.get("corpus_status") == "included"
        ]
    elif resource_id == "rope365":
        values = manifest.get("coverage", {}).get("included_urls", [])
    else:
        raise RuntimeError(f"unsupported multi-page resource: {resource_id}")
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise RuntimeError(f"included URL inventory changed schema: {resource_id}")
    normalized = [normalize_url(value) for value in values]
    if len(set(normalized)) != len(normalized):
        raise RuntimeError(f"included URL inventory is not unique: {resource_id}")
    return normalized


def _metadata_group(
    *,
    artifact: dict,
    spans: Sequence[Span],
    data_by_path: dict[str, bytes],
) -> SourceGroup:
    identity = f"corpus-metadata-v1:{artifact['source_document_identity_sha256']}"
    return _make_group(
        origin_kind="corpus_metadata",
        resource_id=artifact["resource_id"],
        spans=spans,
        data_by_path=data_by_path,
        source_identity_sha256s={identity},
        provenance_mapping="document_level_metadata_not_a_source_page",
    )


def _assert_complete_coverage(plan: DocumentPlan) -> None:
    ordered = sorted(plan.spans + plan.excluded_spans, key=lambda item: item.start)
    cursor = 0
    for span in ordered:
        if span.start != cursor:
            raise RuntimeError(
                f"byte coverage gap or overlap in {plan.resource_id} at {cursor}:{span.start}"
            )
        if span.end <= span.start:
            raise RuntimeError(f"empty coverage span in {plan.resource_id}")
        cursor = span.end
    if cursor != plan.byte_length:
        raise RuntimeError(
            f"byte coverage incomplete in {plan.resource_id}: {cursor}/{plan.byte_length}"
        )
    grouped_ids = [span.span_id for group in plan.groups for span in group.spans]
    included_ids = [span.span_id for span in plan.spans]
    if Counter(grouped_ids) != Counter(included_ids):
        raise RuntimeError(f"included span ownership changed for {plan.resource_id}")


def _build_crash_restraint(
    artifact: dict, data: bytes, manifest: dict, data_by_path: dict[str, bytes]
) -> DocumentPlan:
    resource_id = artifact["resource_id"]
    markdown_path = artifact["markdown_path"]
    target = EXPECTED_SPECIAL_COUNTS[resource_id]
    included_urls = _manifest_included_urls(resource_id, manifest)
    segments, gaps = _segments(data, {2})
    if len(segments) != target["h2"] or len(included_urls) != target["included_pages"]:
        raise RuntimeError("Crash Restraint H2/page contract drifted")
    spans: list[Span] = []
    groups: list[SourceGroup] = []
    seen_urls: list[str] = []
    for start, end, level in segments:
        urls = _extract_source_urls(URL_LABELS[resource_id], data[start:end])
        if len(urls) != 1:
            raise RuntimeError("Crash Restraint H2 must contain exactly one Source URL")
        seen_urls.extend(urls)
        identities = {source_url_identity(urls[0])}
        span = _make_span(
            resource_id=resource_id,
            markdown_path=markdown_path,
            data=data,
            start=start,
            end=end,
            role="included_page_h2",
            heading_level=level,
            source_identity_sha256s=identities,
            provenance_mapping="one_manifest_page_to_one_h2_section",
        )
        spans.append(span)
        groups.append(
            _make_group(
                origin_kind="site_page",
                resource_id=resource_id,
                spans=[span],
                data_by_path=data_by_path,
                source_identity_sha256s=identities,
                provenance_mapping="one_manifest_page_to_one_h2_section",
            )
        )
    if set(seen_urls) != set(included_urls) or len(seen_urls) != len(set(seen_urls)):
        raise RuntimeError("Crash Restraint H2 URL coverage drifted")
    metadata_spans = [
        _make_span(
            resource_id=resource_id,
            markdown_path=markdown_path,
            data=data,
            start=start,
            end=end,
            role="corpus_metadata",
            heading_level=None,
        )
        for start, end in gaps
    ]
    if metadata_spans:
        spans.extend(metadata_spans)
        groups.append(
            _metadata_group(artifact=artifact, spans=metadata_spans, data_by_path=data_by_path)
        )
    plan = DocumentPlan(
        resource_id=resource_id,
        artifact_id=artifact["artifact_id"],
        markdown_path=markdown_path,
        markdown_sha256=artifact["markdown_sha256"],
        byte_length=len(data),
        spans=spans,
        groups=groups,
        excluded_spans=[],
        source_identity_count=len(included_urls),
        construction="116_manifest_pages_mapped_one_to_one_to_116_h2_sections",
    )
    _assert_complete_coverage(plan)
    return plan


def _build_shibari_atlas(
    artifact: dict, data: bytes, manifest: dict, data_by_path: dict[str, bytes]
) -> DocumentPlan:
    resource_id = artifact["resource_id"]
    markdown_path = artifact["markdown_path"]
    target = EXPECTED_SPECIAL_COUNTS[resource_id]
    included_urls = _manifest_included_urls(resource_id, manifest)
    segments, gaps = _segments(data, {2, 3})
    h3 = [segment for segment in segments if segment[2] == 3]
    structural = [segment for segment in segments if segment[2] == 2]
    if len(h3) != target["h3"] or len(included_urls) != target["included_pages"]:
        raise RuntimeError("Shibari Atlas H3/page contract drifted")
    spans: list[Span] = []
    groups: list[SourceGroup] = []
    seen_urls: list[str] = []
    for start, end, level in h3:
        urls = _extract_source_urls(URL_LABELS[resource_id], data[start:end])
        if len(urls) != 1:
            raise RuntimeError("Shibari Atlas H3 must contain exactly one Source URL")
        seen_urls.extend(urls)
        identities = {source_url_identity(urls[0])}
        span = _make_span(
            resource_id=resource_id,
            markdown_path=markdown_path,
            data=data,
            start=start,
            end=end,
            role="included_page_h3",
            heading_level=level,
            source_identity_sha256s=identities,
            provenance_mapping="one_inventory_page_to_one_h3_section",
        )
        spans.append(span)
        groups.append(
            _make_group(
                origin_kind="site_page",
                resource_id=resource_id,
                spans=[span],
                data_by_path=data_by_path,
                source_identity_sha256s=identities,
                provenance_mapping="one_inventory_page_to_one_h3_section",
            )
        )
    if set(seen_urls) != set(included_urls) or len(seen_urls) != len(set(seen_urls)):
        raise RuntimeError("Shibari Atlas H3 URL coverage drifted")
    metadata_ranges = gaps + [(start, end) for start, end, _ in structural]
    metadata_ranges.sort()
    metadata_spans = [
        _make_span(
            resource_id=resource_id,
            markdown_path=markdown_path,
            data=data,
            start=start,
            end=end,
            role="corpus_metadata_or_h2_group_header",
            heading_level=(2 if any(start == item[0] for item in structural) else None),
        )
        for start, end in metadata_ranges
    ]
    if metadata_spans:
        spans.extend(metadata_spans)
        groups.append(
            _metadata_group(artifact=artifact, spans=metadata_spans, data_by_path=data_by_path)
        )
    plan = DocumentPlan(
        resource_id=resource_id,
        artifact_id=artifact["artifact_id"],
        markdown_path=markdown_path,
        markdown_sha256=artifact["markdown_sha256"],
        byte_length=len(data),
        spans=spans,
        groups=groups,
        excluded_spans=[],
        source_identity_count=len(included_urls),
        construction="1714_inventory_pages_mapped_one_to_one_to_1714_h3_sections",
    )
    _assert_complete_coverage(plan)
    return plan


def _build_rope_topia(
    artifact: dict, data: bytes, manifest: dict, data_by_path: dict[str, bytes]
) -> DocumentPlan:
    resource_id = artifact["resource_id"]
    markdown_path = artifact["markdown_path"]
    target = EXPECTED_SPECIAL_COUNTS[resource_id]
    included_urls = _manifest_included_urls(resource_id, manifest)
    segments, gaps = _segments(data, {2})
    included: list[tuple[int, int, int, str]] = []
    excluded: list[tuple[int, int, int]] = []
    for start, end, level in segments:
        urls = _extract_source_urls(URL_LABELS[resource_id], data[start:end])
        if len(urls) == 1 and urls[0] in set(included_urls):
            included.append((start, end, level, urls[0]))
        elif not urls:
            excluded.append((start, end, level))
        else:
            raise RuntimeError("Rope-topia H2 URL is not an included manifest page")
    if (
        len(included) != target["h2_included"]
        or len(excluded) != target["h2_excluded"]
        or len(included_urls) != target["included_pages"]
        or not excluded
        or excluded[0][1] != len(data)
    ):
        raise RuntimeError("Rope-topia included/appendix H2 contract drifted")
    spans: list[Span] = []
    groups: list[SourceGroup] = []
    seen_urls: list[str] = []
    for start, end, level, url in included:
        seen_urls.append(url)
        identities = {source_url_identity(url)}
        span = _make_span(
            resource_id=resource_id,
            markdown_path=markdown_path,
            data=data,
            start=start,
            end=end,
            role="included_page_h2",
            heading_level=level,
            source_identity_sha256s=identities,
            provenance_mapping="one_manifest_page_to_one_h2_section",
        )
        spans.append(span)
        groups.append(
            _make_group(
                origin_kind="site_page",
                resource_id=resource_id,
                spans=[span],
                data_by_path=data_by_path,
                source_identity_sha256s=identities,
                provenance_mapping="one_manifest_page_to_one_h2_section",
            )
        )
    if set(seen_urls) != set(included_urls) or len(seen_urls) != len(set(seen_urls)):
        raise RuntimeError("Rope-topia included H2 URL coverage drifted")
    metadata_spans = [
        _make_span(
            resource_id=resource_id,
            markdown_path=markdown_path,
            data=data,
            start=start,
            end=end,
            role="corpus_metadata",
            heading_level=None,
        )
        for start, end in gaps
    ]
    if metadata_spans:
        spans.extend(metadata_spans)
        groups.append(
            _metadata_group(artifact=artifact, spans=metadata_spans, data_by_path=data_by_path)
        )
    excluded_spans = [
        _make_span(
            resource_id=resource_id,
            markdown_path=markdown_path,
            data=data,
            start=start,
            end=end,
            role="excluded_recovered_pages_appendix",
            heading_level=level,
            provenance_mapping="manifest_excluded_pages_documented_not_trained",
        )
        for start, end, level in excluded
    ]
    plan = DocumentPlan(
        resource_id=resource_id,
        artifact_id=artifact["artifact_id"],
        markdown_path=markdown_path,
        markdown_sha256=artifact["markdown_sha256"],
        byte_length=len(data),
        spans=spans,
        groups=groups,
        excluded_spans=excluded_spans,
        source_identity_count=len(included_urls),
        construction="8_included_h2_pages_plus_one_explicit_excluded_appendix",
    )
    _assert_complete_coverage(plan)
    return plan


def _build_rope365(
    artifact: dict, data: bytes, manifest: dict, data_by_path: dict[str, bytes]
) -> DocumentPlan:
    resource_id = artifact["resource_id"]
    markdown_path = artifact["markdown_path"]
    target = EXPECTED_SPECIAL_COUNTS[resource_id]
    included_urls = _manifest_included_urls(resource_id, manifest)
    segments, gaps = _segments(data, {2, 3})
    if Counter(level for _, _, level in segments) != Counter(
        {2: target["h2"], 3: target["h3"]}
    ) or len(included_urls) != target["included_pages"]:
        raise RuntimeError("Rope365 H2/H3/page contract drifted")

    spans: list[Span] = []
    groups: list[SourceGroup] = []
    current_parent: Span | None = None
    current_children: list[Span] = []
    seen_urls: list[str] = []

    def finish_parent() -> None:
        nonlocal current_parent, current_children
        if current_parent is None:
            return
        owned = [current_parent, *current_children]
        identities = {
            identity for span in owned for identity in span.source_identity_sha256s
        }
        groups.append(
            _make_group(
                origin_kind="rope365_synthesized_h2_parent",
                resource_id=resource_id,
                spans=owned,
                data_by_path=data_by_path,
                source_identity_sha256s=identities,
                provenance_mapping=(
                    "h2_parent_with_h3_descendants; listed_page_sets_are_exact_but_"
                    "synthesis_prevents_inference_of_unlisted_page_mappings"
                ),
            )
        )
        current_parent = None
        current_children = []

    for start, end, level in segments:
        urls = _extract_source_urls(URL_LABELS[resource_id], data[start:end])
        seen_urls.extend(urls)
        identities = {source_url_identity(url) for url in urls}
        if len(urls) == 0:
            provenance = "zero_listed_pages_structural_or_synthesis_span"
        elif len(urls) == 1:
            provenance = "one_listed_page_in_synthesized_section"
        else:
            provenance = "multiple_listed_pages_synthesized_into_one_section"
        if level == 2:
            finish_parent()
            current_parent = _make_span(
                resource_id=resource_id,
                markdown_path=markdown_path,
                data=data,
                start=start,
                end=end,
                role="synthesized_h2_split_parent",
                heading_level=2,
                source_identity_sha256s=identities,
                provenance_mapping=provenance,
            )
            spans.append(current_parent)
        else:
            if current_parent is None:
                raise RuntimeError("Rope365 H3 appears before an H2 split parent")
            child = _make_span(
                resource_id=resource_id,
                markdown_path=markdown_path,
                data=data,
                start=start,
                end=end,
                role="synthesized_h3_descendant",
                heading_level=3,
                source_identity_sha256s=identities,
                provenance_mapping=provenance,
                parent_span_id=current_parent.span_id,
            )
            current_children.append(child)
            spans.append(child)
    finish_parent()
    if set(seen_urls) != set(included_urls) or len(seen_urls) != len(set(seen_urls)):
        raise RuntimeError("Rope365 listed page coverage drifted")
    if len(groups) != target["h2"]:
        raise RuntimeError("Rope365 split-parent count drifted")
    metadata_spans = [
        _make_span(
            resource_id=resource_id,
            markdown_path=markdown_path,
            data=data,
            start=start,
            end=end,
            role="corpus_metadata",
            heading_level=None,
        )
        for start, end in gaps
    ]
    if metadata_spans:
        spans.extend(metadata_spans)
        groups.append(
            _metadata_group(artifact=artifact, spans=metadata_spans, data_by_path=data_by_path)
        )
    plan = DocumentPlan(
        resource_id=resource_id,
        artifact_id=artifact["artifact_id"],
        markdown_path=markdown_path,
        markdown_sha256=artifact["markdown_sha256"],
        byte_length=len(data),
        spans=spans,
        groups=groups,
        excluded_spans=[],
        source_identity_count=len(included_urls),
        construction=(
            "16_h2_split_parents_with_58_h3_descendants_and_exact_111_page_"
            "label_coverage_without_invented_page_mappings"
        ),
    )
    _assert_complete_coverage(plan)
    return plan


def _build_single_document(
    artifact: dict, data: bytes, data_by_path: dict[str, bytes]
) -> DocumentPlan:
    resource_id = artifact["resource_id"]
    markdown_path = artifact["markdown_path"]
    source_identity = artifact.get("source_document_identity")
    if not isinstance(source_identity, dict):
        raise RuntimeError(f"source identity schema changed for {resource_id}")
    identities = {f"document-identity-v1:{artifact['source_document_identity_sha256']}"}
    canonical_url = source_identity.get("canonical_url")
    if canonical_url is not None:
        identities.add(source_url_identity(canonical_url))
    span = _make_span(
        resource_id=resource_id,
        markdown_path=markdown_path,
        data=data,
        start=0,
        end=len(data),
        role="registered_single_document",
        heading_level=None,
        source_identity_sha256s=identities,
        provenance_mapping="one_registered_document_to_one_source_group",
    )
    group = _make_group(
        origin_kind="site_single_document",
        resource_id=resource_id,
        spans=[span],
        data_by_path=data_by_path,
        source_identity_sha256s=identities,
        provenance_mapping="one_registered_document_to_one_source_group",
    )
    plan = DocumentPlan(
        resource_id=resource_id,
        artifact_id=artifact["artifact_id"],
        markdown_path=markdown_path,
        markdown_sha256=artifact["markdown_sha256"],
        byte_length=len(data),
        spans=[span],
        groups=[group],
        excluded_spans=[],
        source_identity_count=1,
        construction="registered_document_remains_one_source_group",
    )
    _assert_complete_coverage(plan)
    return plan


def _read_registry_and_plans(
    *, root: Path = ROOT, registry_path: Path = REGISTRY
) -> tuple[
    dict,
    str,
    list[DocumentPlan],
    dict[str, bytes],
    list[dict],
    Any,
    dict,
]:
    initial_registry_bytes = registry_path.read_bytes()
    registry_sha = sha256_bytes(initial_registry_bytes)
    registry = json.loads(initial_registry_bytes)
    if registry.get("schema") != "site-corpus-registry-v1":
        raise RuntimeError("unsupported site corpus registry schema")
    tokenizer, tokenizer_binding, tokenizer_receipts = _load_pinned_tokenizer(
        registry, root=root
    )
    artifacts = registry.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != EXPECTED_ARTIFACT_COUNT:
        raise RuntimeError("source split requires exactly 33 registered artifacts")
    by_resource = {item.get("resource_id"): item for item in artifacts}
    if len(by_resource) != len(artifacts) or set(SPECIAL_RESOURCES) - set(by_resource):
        raise RuntimeError("registered resource IDs are incomplete or duplicated")
    data_by_path: dict[str, bytes] = {}
    plans: list[DocumentPlan] = []
    receipts: list[dict] = [
        {"path": _relative(registry_path, root), "file_sha256": registry_sha}
    ] + tokenizer_receipts
    builders = {
        "crash_restraint": _build_crash_restraint,
        "shibari_atlas": _build_shibari_atlas,
        "rope_topia": _build_rope_topia,
        "rope365": _build_rope365,
    }
    for resource_id in sorted(by_resource):
        artifact = by_resource[resource_id]
        if artifact.get("declared_direct_training_ready") is not True:
            raise RuntimeError(f"registered artifact is not training-ready: {resource_id}")
        markdown_path = _assert_safe_registered_path(artifact.get("markdown_path"), root=root)
        manifest_path = _assert_safe_registered_path(artifact.get("manifest_path"), root=root)
        markdown = markdown_path.read_bytes()
        manifest_bytes = manifest_path.read_bytes()
        if sha256_bytes(markdown) != artifact.get("markdown_sha256"):
            raise RuntimeError(f"registered Markdown content address drifted: {resource_id}")
        if sha256_bytes(manifest_bytes) != artifact.get("manifest_sha256"):
            raise RuntimeError(f"registered manifest content address drifted: {resource_id}")
        if len(markdown) != artifact.get("byte_length"):
            raise RuntimeError(f"registered Markdown byte count drifted: {resource_id}")
        relative_markdown = _relative(markdown_path, root)
        data_by_path[relative_markdown] = markdown
        receipts.extend(
            [
                {"path": relative_markdown, "file_sha256": sha256_bytes(markdown)},
                {
                    "path": _relative(manifest_path, root),
                    "file_sha256": sha256_bytes(manifest_bytes),
                },
            ]
        )
        if resource_id in SPECIAL_RESOURCES:
            manifest = json.loads(manifest_bytes)
            plan = builders[resource_id](artifact, markdown, manifest, data_by_path)
        else:
            plan = _build_single_document(artifact, markdown, data_by_path)
        for span in plan.spans:
            span_text = markdown[span.start : span.end].decode(
                "utf-8", errors="strict"
            )
            span.qwen36_token_count = _token_count(tokenizer, span_text)
            if span.qwen36_token_count <= 0:
                raise RuntimeError(f"included span has no Qwen tokens: {span.span_id}")
        plans.append(plan)
    # A concurrent registry rewrite invalidates the entire build even if every
    # artifact happened to remain readable during the race.
    if registry_path.read_bytes() != initial_registry_bytes:
        raise RuntimeError("site corpus registry changed during source split build")
    return (
        registry,
        registry_sha,
        plans,
        data_by_path,
        receipts,
        tokenizer,
        tokenizer_binding,
    )


def _read_v440_preseal_groups_and_records(
    *, root: Path = ROOT,
    manifest_path: Path = V440_MANIFEST,
    projection_path: Path = V440_PROJECTION,
    tokenizer: Any,
) -> tuple[list[SourceGroup], dict, list[dict], list[dict]]:
    for path in (manifest_path, projection_path):
        relative = _relative(path, root).casefold()
        if any(token in relative for token in FORBIDDEN_PATH_TOKENS):
            raise RuntimeError(f"V440 identity input crosses forbidden boundary: {relative}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    projection = manifest.get("projection")
    if not isinstance(projection, dict):
        raise RuntimeError("V440 projection metadata is missing")
    expected_path = Path(projection.get("path", "")).resolve()
    if expected_path != projection_path.resolve():
        raise RuntimeError("V440 projection path changed")
    if file_sha256(projection_path) != projection.get("sha256"):
        raise RuntimeError("V440 projection content address changed")
    expected_rows = projection.get("rows")
    if not isinstance(expected_rows, int) or expected_rows <= 0:
        raise RuntimeError("V440 projection row count changed")
    by_document: dict[str, list[str]] = defaultdict(list)
    token_counts_by_document: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    excluded_index_fact_ids: list[str] = []
    selected_records: list[dict] = []
    all_fact_ids: set[str] = set()
    rows = 0
    with projection_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            # Deliberately select only the approved nonsemantic identity fields.
            kind = row.get("kind")
            fact_id = row.get("fact_id")
            document_sha = row.get("document_sha256")
            if (
                not isinstance(kind, str)
                or not isinstance(fact_id, str)
                or not isinstance(document_sha, str)
                or not re.fullmatch(r"[0-9a-f]{64}", document_sha)
                or not fact_id
            ):
                raise RuntimeError("V440 nonsemantic identity metadata changed schema")
            if fact_id in all_fact_ids:
                raise RuntimeError("V440 fact identity is duplicated")
            all_fact_ids.add(fact_id)
            if kind == "qa_resource_index":
                excluded_index_fact_ids.append(fact_id)
            else:
                prompt_tokens, answer_tokens = _encode_v440_pair(tokenizer, row)
                by_document[document_sha].append(fact_id)
                token_counts_by_document[document_sha][0] += prompt_tokens
                token_counts_by_document[document_sha][1] += answer_tokens
                selected_records.append(
                    {
                        "record": row,
                        "fact_id": fact_id,
                        "document_sha256": document_sha,
                        "prompt_token_count": prompt_tokens,
                        "answer_token_count": answer_tokens,
                    }
                )
            rows += 1
    if rows != expected_rows:
        raise RuntimeError("V440 projection row count does not match manifest")
    groups = []
    for document_sha, fact_ids in sorted(by_document.items()):
        identity = f"document-content-v1:{document_sha}"
        prompt_tokens, answer_tokens = token_counts_by_document[document_sha]
        groups.append(
            _make_group(
                origin_kind="v440_qa_source_document",
                resource_id=None,
                spans=[],
                data_by_path={},
                source_identity_sha256s={identity},
                provenance_mapping="all_qa_rows_inherit_document_sha256_split",
                descendant_fact_ids=fact_ids,
                descendant_prompt_token_count=prompt_tokens,
                descendant_answer_token_count=answer_tokens,
                external_content_sha256=document_sha,
            )
        )
    summary = {
        "projection_rows": rows,
        "selected_rows": sum(len(value) for value in by_document.values()),
        "selected_source_document_groups": len(by_document),
        "selected_prompt_tokens": sum(
            item["prompt_token_count"] for item in selected_records
        ),
        "selected_answer_tokens": sum(
            item["answer_token_count"] for item in selected_records
        ),
        "selected_qwen36_tokens": sum(
            item["prompt_token_count"] + item["answer_token_count"]
            for item in selected_records
        ),
        "excluded_qa_resource_index_rows": len(excluded_index_fact_ids),
        "excluded_qa_resource_index_commitment_sha256": canonical_sha256(
            sorted(excluded_index_fact_ids)
        ),
        "semantic_fields_read_for_tokenization_and_train_development_projection": True,
        "semantic_fields_used_for_split_assignment": False,
        "final_semantic_records_emitted": False,
    }
    receipts = [
        {"path": _relative(manifest_path, root), "file_sha256": sha256_bytes(manifest_bytes)},
        {"path": _relative(projection_path, root), "file_sha256": file_sha256(projection_path)},
    ]
    return groups, summary, receipts, selected_records


def _minhash(shingles: frozenset[int]) -> tuple[int, ...]:
    if not shingles:
        return ()
    return tuple(
        min((a * (value % _MINHASH_PRIME) + b) % _MINHASH_PRIME for value in shingles)
        for a, b in zip(_MINHASH_A, _MINHASH_B)
    )


def _near_duplicate_edges(groups: Sequence[SourceGroup]) -> list[tuple[str, str]]:
    signatures: dict[str, tuple[int, ...]] = {}
    buckets: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
    by_id = {group.group_id: group for group in groups}
    for group in groups:
        signature = _minhash(group.shingles)
        if not signature:
            continue
        signatures[group.group_id] = signature
        for band in range(0, MINHASH_SIZE, MINHASH_BAND_SIZE):
            buckets[(band, signature[band : band + MINHASH_BAND_SIZE])].append(
                group.group_id
            )
    candidates: set[tuple[str, str]] = set()
    for members in buckets.values():
        unique = sorted(set(members))
        for left_index, left in enumerate(unique):
            for right in unique[left_index + 1 :]:
                candidates.add((left, right))
    result = []
    for left, right in sorted(candidates):
        left_values = by_id[left].shingles
        right_values = by_id[right].shingles
        union = len(left_values | right_values)
        if union and len(left_values & right_values) / union >= NEAR_DUPLICATE_JACCARD:
            result.append((left, right))
    return result


def build_duplicate_components(
    groups: Sequence[SourceGroup],
) -> tuple[dict[str, str], dict[str, list[str]], list[tuple[str, str]]]:
    by_id = {group.group_id: group for group in groups}
    if len(by_id) != len(groups):
        raise RuntimeError("source group identities are not unique")
    union = UnionFind(by_id)
    exact_keys: dict[tuple[str, str], str] = {}
    for group in groups:
        keys = [("content", group.content_sha256)]
        if group.normalized_text_sha256:
            keys.append(("normalized_text", group.normalized_text_sha256))
        keys.extend(("source_identity", value) for value in group.source_identity_sha256s)
        for key in keys:
            previous = exact_keys.get(key)
            if previous is None:
                exact_keys[key] = group.group_id
            else:
                union.union(previous, group.group_id)
    near_edges = _near_duplicate_edges(groups)
    for left, right in near_edges:
        union.union(left, right)
    members_by_root: dict[str, list[str]] = defaultdict(list)
    for group_id in sorted(by_id):
        members_by_root[union.find(group_id)].append(group_id)
    components: dict[str, list[str]] = {}
    group_to_component: dict[str, str] = {}
    for members in members_by_root.values():
        component_id = _content_address(
            "source-component-v1", {"source_group_ids": sorted(members)}
        )
        components[component_id] = sorted(members)
        for group_id in members:
            group_to_component[group_id] = component_id
    return group_to_component, dict(sorted(components.items())), near_edges


def split_for_component(component_id: str) -> str:
    digest = hashlib.sha256(
        f"{SPLIT_NAMESPACE}\x00{component_id}".encode("ascii")
    ).digest()
    bucket = int.from_bytes(digest[:8], "big")
    cardinality = 1 << 64
    if bucket < cardinality * 8 // 10:
        return "train"
    if bucket < cardinality * 9 // 10:
        return "development"
    return "final"


def assign_groups(
    groups: Sequence[SourceGroup], group_to_component: dict[str, str]
) -> dict[str, str]:
    return {
        group.group_id: split_for_component(group_to_component[group.group_id])
        for group in groups
    }


def validate_disjointness(
    groups: Sequence[SourceGroup],
    group_to_component: dict[str, str],
    assignments: dict[str, str],
    near_edges: Sequence[tuple[str, str]],
) -> None:
    by_id = {group.group_id: group for group in groups}
    if set(by_id) != set(assignments) or set(by_id) != set(group_to_component):
        raise RuntimeError("split assignment does not cover every source group exactly once")
    allowed = {"train", "development", "final"}
    if set(assignments.values()) - allowed:
        raise RuntimeError("unsupported split assignment")
    component_splits: dict[str, set[str]] = defaultdict(set)
    for group_id, split in assignments.items():
        component_splits[group_to_component[group_id]].add(split)
    if any(len(values) != 1 for values in component_splits.values()):
        raise RuntimeError("duplicate component leaked across splits")
    identity_splits: dict[str, set[str]] = defaultdict(set)
    exact_splits: dict[str, set[str]] = defaultdict(set)
    normalized_splits: dict[str, set[str]] = defaultdict(set)
    for group in groups:
        split = assignments[group.group_id]
        owned_span_ids = {span.span_id for span in group.spans}
        for identity in group.source_identity_sha256s:
            identity_splits[identity].add(split)
        exact_splits[group.content_sha256].add(split)
        if group.normalized_text_sha256:
            normalized_splits[group.normalized_text_sha256].add(split)
        for span in group.spans:
            if (
                span.parent_span_id is not None
                and span.parent_span_id not in owned_span_ids
            ):
                raise RuntimeError("descendant span escaped its split parent")
    if any(len(values) != 1 for values in identity_splits.values()):
        raise RuntimeError("normalized URL/source identity leaked across splits")
    if any(len(values) != 1 for values in exact_splits.values()):
        raise RuntimeError("exact content duplicate leaked across splits")
    if any(len(values) != 1 for values in normalized_splits.values()):
        raise RuntimeError("normalized content duplicate leaked across splits")
    for left, right in near_edges:
        if assignments[left] != assignments[right]:
            raise RuntimeError("near duplicate leaked across splits")


def _span_record(span: Span) -> dict:
    return {
        "span_id": span.span_id,
        "markdown_path": span.markdown_path,
        "byte_start": span.start,
        "byte_end": span.end,
        "byte_length": span.byte_length,
        "qwen36_token_count": span.qwen36_token_count,
        "content_sha256": span.content_sha256,
        "role": span.role,
        "heading_level": span.heading_level,
        "parent_span_id": span.parent_span_id,
        "source_identity_sha256s": list(span.source_identity_sha256s),
        "provenance_mapping": span.provenance_mapping,
    }


def _group_record(
    group: SourceGroup,
    *,
    component_id: str,
    component_size: int,
) -> dict:
    return {
        "source_group_id": group.group_id,
        "duplicate_component_id": component_id,
        "duplicate_component_source_group_count": component_size,
        "origin_kind": group.origin_kind,
        "resource_id": group.resource_id,
        "byte_length": group.byte_length,
        "site_qwen36_token_count": sum(
            span.qwen36_token_count or 0 for span in group.spans
        ),
        "content_sha256": group.content_sha256,
        "normalized_text_sha256": group.normalized_text_sha256,
        "source_identity_sha256s": sorted(group.source_identity_sha256s),
        "provenance_mapping": group.provenance_mapping,
        "spans": [_span_record(span) for span in group.spans],
        "descendant_fact_ids": group.descendant_fact_ids,
        "descendant_prompt_token_count": group.descendant_prompt_token_count,
        "descendant_answer_token_count": group.descendant_answer_token_count,
        "descendant_qwen36_token_count": (
            group.descendant_prompt_token_count
            + group.descendant_answer_token_count
        ),
        "descendant_fact_ids_commitment_sha256": canonical_sha256(
            group.descendant_fact_ids
        ),
    }


def _split_summary(
    split: str,
    groups: Sequence[SourceGroup],
    group_to_component: dict[str, str],
    components: dict[str, list[str]],
    assignments: dict[str, str],
    *,
    include_records: bool,
) -> dict:
    selected = sorted(
        (group for group in groups if assignments[group.group_id] == split),
        key=lambda item: item.group_id,
    )
    component_ids = sorted({group_to_component[group.group_id] for group in selected})
    full_records = [
        _group_record(
            group,
            component_id=group_to_component[group.group_id],
            component_size=len(components[group_to_component[group.group_id]]),
        )
        for group in selected
    ]
    result = {
        "source_group_count": len(selected),
        "duplicate_component_count": len(component_ids),
        "site_source_group_count": sum(group.resource_id is not None for group in selected),
        "v440_source_document_group_count": sum(
            group.origin_kind == "v440_qa_source_document" for group in selected
        ),
        "span_count": sum(len(group.spans) for group in selected),
        "site_byte_count": sum(group.byte_length for group in selected),
        "site_qwen36_token_count": sum(
            sum(span.qwen36_token_count or 0 for span in group.spans)
            for group in selected
        ),
        "v440_descendant_fact_count": sum(
            len(group.descendant_fact_ids) for group in selected
        ),
        "v440_prompt_token_count": sum(
            group.descendant_prompt_token_count for group in selected
        ),
        "v440_answer_token_count": sum(
            group.descendant_answer_token_count for group in selected
        ),
        "v440_qwen36_token_count": sum(
            group.descendant_prompt_token_count
            + group.descendant_answer_token_count
            for group in selected
        ),
        "source_group_membership_commitment_sha256": canonical_sha256(
            [group.group_id for group in selected]
        ),
        "duplicate_component_membership_commitment_sha256": canonical_sha256(
            component_ids
        ),
        "full_records_commitment_sha256": canonical_sha256(full_records),
    }
    if include_records:
        result["records"] = full_records
        result["records_redacted"] = False
    else:
        result["records_redacted"] = True
        result["redaction_reason"] = (
            "final membership is sealed by aggregate counts and commitments; "
            "no final source identity, span, URL, or fact membership is emitted"
        )
    return result


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
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


def _jsonl_payload(rows: Sequence[dict]) -> bytes:
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


def _materialize_train_dev_projections(
    *,
    registry: dict,
    registry_sha: str,
    plans: Sequence[DocumentPlan],
    data_by_path: dict[str, bytes],
    v440_records: Sequence[dict],
    groups: Sequence[SourceGroup],
    group_to_component: dict[str, str],
    assignments: dict[str, str],
    v440_projection_sha256: str,
) -> dict:
    artifacts = {item["resource_id"]: item for item in registry["artifacts"]}
    group_by_span: dict[str, SourceGroup] = {}
    for group in groups:
        for span in group.spans:
            if span.span_id in group_by_span:
                raise RuntimeError("site span belongs to more than one source group")
            group_by_span[span.span_id] = group
    group_by_document = {
        group.content_sha256: group
        for group in groups
        if group.origin_kind == "v440_qa_source_document"
    }
    plan_spans = [span for plan in plans for span in plan.spans]
    if set(group_by_span) != {span.span_id for span in plan_spans}:
        raise RuntimeError("site projection span ownership is incomplete")

    result: dict[str, dict] = {}
    for partition in ("train", "development"):
        site_rows = []
        for span in sorted(
            plan_spans,
            key=lambda item: (item.resource_id, item.start, item.span_id),
        ):
            group = group_by_span[span.span_id]
            if assignments[group.group_id] != partition:
                continue
            artifact = artifacts[span.resource_id]
            text = data_by_path[span.markdown_path][span.start : span.end].decode(
                "utf-8", errors="strict"
            )
            if sha256_bytes(text.encode("utf-8")) != span.content_sha256:
                raise RuntimeError("site projection text no longer matches span identity")
            row_identity = {
                "split": partition,
                "source_group_id": group.group_id,
                "span_id": span.span_id,
                "text_sha256": span.content_sha256,
            }
            site_rows.append(
                {
                    "schema": "site-source-span-projection-v1",
                    "projection_record_id": _content_address(
                        "site-source-span-record-v1", row_identity
                    ),
                    "split": partition,
                    "training_role": "cpt_raw_markdown_source_span",
                    "training_format": "causal_next_token_markdown",
                    "assistant_supervision": False,
                    "cross_document_packing_performed": False,
                    "resource_id": span.resource_id,
                    "artifact_id": artifact["artifact_id"],
                    "source_document_identity_sha256": artifact[
                        "source_document_identity_sha256"
                    ],
                    "source_group_id": group.group_id,
                    "duplicate_component_id": group_to_component[group.group_id],
                    "span_id": span.span_id,
                    "parent_span_id": span.parent_span_id,
                    "role": span.role,
                    "provenance_mapping": span.provenance_mapping,
                    "source_identity_sha256s": list(span.source_identity_sha256s),
                    "markdown_path": span.markdown_path,
                    "markdown_sha256": artifact["markdown_sha256"],
                    "byte_start": span.start,
                    "byte_end": span.end,
                    "byte_length": span.byte_length,
                    "text_sha256": span.content_sha256,
                    "qwen36_token_count": span.qwen36_token_count,
                    "rights_basis": artifact["rights_basis"],
                    "safety_transfer_flags": artifact.get(
                        "safety_transfer_flags", []
                    ),
                    "lineage": {
                        "registry_path": _relative(REGISTRY),
                        "registry_file_sha256": registry_sha,
                        "registry_artifact_id": artifact["artifact_id"],
                        "assignment_preceded_projection": True,
                    },
                    "text": text,
                }
            )

        qa_rows = []
        qa_documents: set[str] = set()
        prompt_tokens = 0
        answer_tokens = 0
        for item in sorted(v440_records, key=lambda value: value["fact_id"]):
            group = group_by_document[item["document_sha256"]]
            if assignments[group.group_id] != partition:
                continue
            original = item["record"]
            if "source_split_lineage_v1" in original:
                raise RuntimeError("V440 row collides with split lineage field")
            projected = dict(original)
            projected["source_split_lineage_v1"] = {
                "schema": "v440-qa-source-split-lineage-v1",
                "split": partition,
                "source_group_id": group.group_id,
                "duplicate_component_id": group_to_component[group.group_id],
                "parent_projection_sha256": v440_projection_sha256,
                "prompt_token_count": item["prompt_token_count"],
                "answer_token_count": item["answer_token_count"],
                "qwen36_token_count": (
                    item["prompt_token_count"] + item["answer_token_count"]
                ),
                "prompt_mode": "es_exact",
                "eos_appended": False,
                "assignment_preceded_projection": True,
            }
            qa_rows.append(projected)
            qa_documents.add(item["document_sha256"])
            prompt_tokens += item["prompt_token_count"]
            answer_tokens += item["answer_token_count"]

        site_payload = _jsonl_payload(site_rows)
        qa_payload = _jsonl_payload(qa_rows)
        site_path = PROJECTION_PATHS[partition]["site_spans"]
        qa_path = PROJECTION_PATHS[partition]["v440_qa"]
        _atomic_write_bytes(site_path, site_payload)
        _atomic_write_bytes(qa_path, qa_payload)
        result[partition] = {
            "site_spans": {
                "schema": "site-source-span-projection-v1",
                "path": _relative(site_path),
                "file_sha256": sha256_bytes(site_payload),
                "file_bytes": len(site_payload),
                "rows": len(site_rows),
                "source_groups": len({row["source_group_id"] for row in site_rows}),
                "source_documents": len({row["resource_id"] for row in site_rows}),
                "source_text_bytes": sum(row["byte_length"] for row in site_rows),
                "qwen36_tokens": sum(
                    row["qwen36_token_count"] for row in site_rows
                ),
                "cross_document_packing_performed": False,
                "contains_only_partition": partition,
            },
            "v440_qa": {
                "schema": "v440-qa-source-split-projection-v1",
                "path": _relative(qa_path),
                "file_sha256": sha256_bytes(qa_payload),
                "file_bytes": len(qa_payload),
                "rows": len(qa_rows),
                "source_document_groups": len(qa_documents),
                "prompt_tokens": prompt_tokens,
                "answer_tokens": answer_tokens,
                "qwen36_tokens": prompt_tokens + answer_tokens,
                "contains_only_partition": partition,
            },
        }
    emitted_facts = sum(
        result[partition]["v440_qa"]["rows"]
        for partition in ("train", "development")
    )
    final_facts = sum(
        len(group.descendant_fact_ids)
        for group in groups
        if assignments[group.group_id] == "final"
    )
    if emitted_facts + final_facts != len(v440_records):
        raise RuntimeError("V440 train/development projections lost selected rows")
    emitted_spans = sum(
        result[partition]["site_spans"]["rows"]
        for partition in ("train", "development")
    )
    final_spans = sum(
        len(group.spans)
        for group in groups
        if assignments[group.group_id] == "final"
    )
    if emitted_spans + final_spans != len(plan_spans):
        raise RuntimeError("site train/development projections lost included spans")
    return result


def construct() -> dict:
    if OUTPUT.exists():
        raise RuntimeError(
            "sealed source split authority already exists; reconstruction would "
            "reopen mixed source files containing final data"
        )
    (
        registry,
        registry_sha,
        plans,
        data_by_path,
        receipts,
        tokenizer,
        tokenizer_binding,
    ) = _read_registry_and_plans()
    site_groups = [group for plan in plans for group in plan.groups]
    (
        v440_groups,
        v440_summary,
        v440_receipts,
        v440_records,
    ) = _read_v440_preseal_groups_and_records(tokenizer=tokenizer)
    groups = site_groups + v440_groups
    group_to_component, components, near_edges = build_duplicate_components(groups)
    assignments = assign_groups(groups, group_to_component)
    validate_disjointness(groups, group_to_component, assignments, near_edges)
    v440_projection_sha = next(
        item["file_sha256"]
        for item in v440_receipts
        if item["path"] == _relative(V440_PROJECTION)
    )
    projections = _materialize_train_dev_projections(
        registry=registry,
        registry_sha=registry_sha,
        plans=plans,
        data_by_path=data_by_path,
        v440_records=v440_records,
        groups=groups,
        group_to_component=group_to_component,
        assignments=assignments,
        v440_projection_sha256=v440_projection_sha,
    )

    plan_by_resource = {plan.resource_id: plan for plan in plans}
    coverage = []
    for artifact in sorted(registry["artifacts"], key=lambda item: item["resource_id"]):
        plan = plan_by_resource[artifact["resource_id"]]
        all_spans = sorted(plan.spans + plan.excluded_spans, key=lambda item: item.start)
        coverage.append(
            {
                "resource_id": plan.resource_id,
                "artifact_id": plan.artifact_id,
                "markdown_path": plan.markdown_path,
                "markdown_sha256": plan.markdown_sha256,
                "registered_byte_length": plan.byte_length,
                "covered_byte_length": sum(span.byte_length for span in all_spans),
                "included_qwen36_token_count": sum(
                    span.qwen36_token_count or 0 for span in plan.spans
                ),
                "included_span_count": len(plan.spans),
                "excluded_span_count": len(plan.excluded_spans),
                "independent_source_group_count": len(plan.groups),
                "source_identity_count": plan.source_identity_count,
                "construction": plan.construction,
                "ordered_span_commitment_sha256": canonical_sha256(
                    [_span_record(span) for span in all_spans]
                ),
            }
        )

    split_summaries = {
        "train": _split_summary(
            "train", groups, group_to_component, components, assignments, include_records=True
        ),
        "development": _split_summary(
            "development",
            groups,
            group_to_component,
            components,
            assignments,
            include_records=True,
        ),
        "final": _split_summary(
            "final", groups, group_to_component, components, assignments, include_records=False
        ),
    }
    for summary in split_summaries.values():
        summary["realized_source_group_fraction"] = (
            summary["source_group_count"] / len(groups)
        )
        summary["realized_duplicate_component_fraction"] = (
            summary["duplicate_component_count"] / len(components)
        )
    all_group_count = sum(item["source_group_count"] for item in split_summaries.values())
    if all_group_count != len(groups):
        raise RuntimeError("split summaries lost source groups")
    site_byte_total = sum(item["site_byte_count"] for item in split_summaries.values())
    included_site_bytes = sum(
        span.byte_length for plan in plans for span in plan.spans
    )
    if site_byte_total != included_site_bytes:
        raise RuntimeError("split summaries lost included site bytes")
    included_site_tokens = sum(
        span.qwen36_token_count or 0 for plan in plans for span in plan.spans
    )
    if sum(
        item["site_qwen36_token_count"] for item in split_summaries.values()
    ) != included_site_tokens:
        raise RuntimeError("split summaries lost included site Qwen tokens")
    for partition in ("train", "development"):
        if (
            projections[partition]["site_spans"]["rows"]
            != split_summaries[partition]["span_count"]
            or projections[partition]["site_spans"]["source_text_bytes"]
            != split_summaries[partition]["site_byte_count"]
            or projections[partition]["site_spans"]["qwen36_tokens"]
            != split_summaries[partition]["site_qwen36_token_count"]
            or projections[partition]["v440_qa"]["rows"]
            != split_summaries[partition]["v440_descendant_fact_count"]
            or projections[partition]["v440_qa"]["prompt_tokens"]
            != split_summaries[partition]["v440_prompt_token_count"]
            or projections[partition]["v440_qa"]["answer_tokens"]
            != split_summaries[partition]["v440_answer_token_count"]
        ):
            raise RuntimeError(f"{partition} projection accounting drifted")

    excluded_spans = [span for plan in plans for span in plan.excluded_spans]
    all_receipts = sorted(receipts + v440_receipts, key=lambda item: item["path"])
    _assert_receipts_current(all_receipts)
    result = {
        "schema": SCHEMA,
        "status": "sealed_source_disjoint_assignment_launch_still_gated",
        "canonical_training_protocol": "plan.md",
        "purpose": (
            "source-group split authority created before chunking or QA derivation; "
            "not a training launch authorization"
        ),
        "source_registry_binding": {
            "path": _relative(REGISTRY),
            "file_sha256": registry_sha,
            "source_tree_fingerprint_sha256": registry[
                "source_tree_fingerprint_sha256"
            ],
            "artifact_count": len(registry["artifacts"]),
            "binding_mode": (
                "runtime content address; --check reconstructs against the current "
                "registry and fails closed on concurrent or unregistered drift"
            ),
        },
        "tokenizer_binding": tokenizer_binding,
        "builder_receipt": {
            "path": _relative(Path(__file__)),
            "file_sha256": file_sha256(Path(__file__)),
        },
        "construction_contract": {
            "assignment_precedes_chunking_and_qa_derivation": True,
            "all_descendants_inherit_parent_source_group_split": True,
            "exact_duplicates_grouped_before_assignment": True,
            "near_duplicates_grouped_before_assignment": True,
            "duplicate_clustering": "transitive_connected_components",
            "near_duplicate_algorithm": {
                "normalization": "NFKC_casefold_strip_urls_and_markdown_punctuation",
                "tokenization": "unicode_words",
                "shingle_width": SHINGLE_WIDTH,
                "minimum_tokens": MIN_NEAR_DUPLICATE_TOKENS,
                "candidate_generation": (
                    f"minhash_{MINHASH_SIZE}_permutations_"
                    f"{MINHASH_BAND_SIZE}_rows_per_band"
                ),
                "confirmation_jaccard_minimum": NEAR_DUPLICATE_JACCARD,
            },
            "url_normalization": (
                "IDNA_lower_host_https_identity_default_port_and_fragment_removed_"
                "dot_segments_and_trailing_slash_normalized_tracking_query_removed_"
                "remaining_query_sorted"
            ),
            "split_algorithm": (
                "sha256(namespace_NUL_duplicate_component_id)_first_u64_threshold"
            ),
            "split_namespace": SPLIT_NAMESPACE,
            "thresholds": {
                "train": "[0.0,0.8)",
                "development": "[0.8,0.9)",
                "final": "[0.9,1.0)",
            },
            "final_public_representation": "aggregate_counts_and_commitments_only",
            "post_seal_check_contract": {
                "original_site_markdown_opened_hashed_or_statted": False,
                "original_site_manifests_opened_hashed_or_statted": False,
                "mixed_v440_projection_opened_hashed_or_statted": False,
                "safe_metadata_reopened": [
                    "sealed_authority",
                    "site_registry",
                    "v440_metadata_manifest",
                    "pinned_tokenizer_files",
                    "builder_source",
                ],
                "derived_projections_rehashed": [
                    "train_site_spans",
                    "development_site_spans",
                    "train_v440_qa",
                    "development_v440_qa",
                ],
                "reconstruction_after_seal_forbidden": True,
            },
        },
        "totals": {
            "registered_site_artifacts": len(plans),
            "site_independent_source_groups": len(site_groups),
            "v440_independent_source_document_groups": len(v440_groups),
            "independent_source_groups": len(groups),
            "duplicate_components": len(components),
            "near_duplicate_edges": len(near_edges),
            "site_included_spans": sum(len(plan.spans) for plan in plans),
            "site_excluded_spans": len(excluded_spans),
            "site_registered_bytes": sum(plan.byte_length for plan in plans),
            "site_included_bytes": included_site_bytes,
            "site_included_qwen36_tokens": included_site_tokens,
            "site_excluded_bytes": sum(span.byte_length for span in excluded_spans),
            "all_source_group_membership_commitment_sha256": canonical_sha256(
                sorted(group.group_id for group in groups)
            ),
            "all_component_membership_commitment_sha256": canonical_sha256(
                sorted(components)
            ),
        },
        "v440_source_projection_summary": v440_summary,
        "site_byte_coverage": coverage,
        "assignments": split_summaries,
        "materialized_train_development_projections": projections,
        "post_split_training_budget": {
            "train_site_qwen36_tokens": split_summaries["train"][
                "site_qwen36_token_count"
            ],
            "train_v440_qwen36_tokens": split_summaries["train"][
                "v440_qwen36_token_count"
            ],
            "train_total_qwen36_tokens": (
                split_summaries["train"]["site_qwen36_token_count"]
                + split_summaries["train"]["v440_qwen36_token_count"]
            ),
            "development_site_qwen36_tokens": split_summaries["development"][
                "site_qwen36_token_count"
            ],
            "development_v440_qwen36_tokens": split_summaries["development"][
                "v440_qwen36_token_count"
            ],
            "development_total_qwen36_tokens": (
                split_summaries["development"]["site_qwen36_token_count"]
                + split_summaries["development"]["v440_qwen36_token_count"]
            ),
            "final_site_qwen36_tokens_aggregate_only": split_summaries["final"][
                "site_qwen36_token_count"
            ],
            "final_v440_qwen36_tokens_aggregate_only": split_summaries["final"][
                "v440_qwen36_token_count"
            ],
            "final_records_or_identities_emitted": False,
        },
        "explicit_nontraining_exclusions": {
            "rope_topia_recovered_pages_appendix": {
                "span_count": len(excluded_spans),
                "byte_count": sum(span.byte_length for span in excluded_spans),
                "ordered_span_commitment_sha256": canonical_sha256(
                    [_span_record(span) for span in excluded_spans]
                ),
                "reason": "manifest-excluded pages are documented but not factual training prose",
            },
            "v440_qa_resource_index": {
                "row_count": v440_summary["excluded_qa_resource_index_rows"],
                "membership_commitment_sha256": v440_summary[
                    "excluded_qa_resource_index_commitment_sha256"
                ],
                "reason": "URL-index memorization rows are not knowledge training examples",
            },
        },
        "invariants": {
            "all_33_registered_markdown_artifacts_accounted_for": len(plans) == 33,
            "every_registered_markdown_byte_covered_exactly_once": True,
            "rope_topia_appendix_explicitly_excluded": len(excluded_spans) == 1,
            "source_url_identity_cross_split_overlap": False,
            "exact_content_cross_split_overlap": False,
            "near_duplicate_cross_split_overlap": False,
            "descendant_cross_split_overlap": False,
            "final_records_emitted": False,
            "train_and_development_projections_materialized_pre_seal": True,
            "downstream_must_not_reopen_mixed_source_inputs": True,
            "post_seal_check_reopens_original_mixed_source_files": False,
            "protected_holdout_ood_terminal_incident_or_manual_review_sources_opened": False,
            "v440_semantic_fields_read_for_tokenization_and_train_development_projection": True,
            "v440_semantic_fields_used_for_split_assignment": False,
            "v440_final_semantic_records_emitted": False,
            "training_launch_authorized": False,
        },
        "safe_input_receipts": all_receipts,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def render() -> bytes:
    return (
        json.dumps(construct(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _lexical_repo_relative(path: str | Path, *, root: Path = ROOT) -> str:
    root_text = os.path.abspath(os.fspath(root))
    value = os.fspath(path)
    absolute = os.path.normpath(
        value if os.path.isabs(value) else os.path.join(root_text, value)
    )
    try:
        common = os.path.commonpath((root_text, absolute))
    except ValueError as error:
        raise RuntimeError(f"path is outside sealed-check root: {value}") from error
    if common != root_text:
        raise RuntimeError(f"path is outside sealed-check root: {value}")
    relative = os.path.relpath(absolute, root_text).replace(os.sep, "/")
    folded = relative.casefold()
    if any(token in folded for token in FORBIDDEN_PATH_TOKENS):
        raise RuntimeError(f"path crosses forbidden boundary: {relative}")
    return relative


def _validate_self_addressed_artifact(value: dict) -> None:
    declared = value.get("content_sha256_before_self_field")
    if not isinstance(declared, str) or not re.fullmatch(r"[0-9a-f]{64}", declared):
        raise RuntimeError("sealed source split authority has no valid self address")
    unsigned = dict(value)
    del unsigned["content_sha256_before_self_field"]
    if canonical_sha256(unsigned) != declared:
        raise RuntimeError("sealed source split authority self address changed")


def _correct_v440_semantic_access_disclosure(
    value: dict, *, builder_receipt: dict
) -> dict:
    value = json.loads(json.dumps(value))
    if value.get("content_sha256_before_self_field") != (
        PRE_DISCLOSURE_CORRECTION_AUTHORITY_SHA256
    ):
        raise RuntimeError("semantic-access correction applies only to the reviewed seal")
    _validate_self_addressed_artifact(value)
    old_summary = value.pop("v440_nonsemantic_identity_summary", None)
    if (
        not isinstance(old_summary, dict)
        or old_summary.pop("semantic_fields_accessed_or_emitted", None) is not False
        or "v440_source_projection_summary" in value
    ):
        raise RuntimeError("reviewed V440 summary disclosure state changed")
    old_summary.update(
        {
            "semantic_fields_read_for_tokenization_and_train_development_projection": True,
            "semantic_fields_used_for_split_assignment": False,
            "final_semantic_records_emitted": False,
        }
    )
    value["v440_source_projection_summary"] = old_summary
    invariants = value.get("invariants")
    if (
        not isinstance(invariants, dict)
        or invariants.pop("v440_semantic_fields_accessed_or_emitted", None) is not False
    ):
        raise RuntimeError("reviewed V440 invariant disclosure state changed")
    invariants.update(
        {
            "v440_semantic_fields_read_for_tokenization_and_train_development_projection": True,
            "v440_semantic_fields_used_for_split_assignment": False,
            "v440_final_semantic_records_emitted": False,
        }
    )
    value["builder_receipt"] = builder_receipt
    del value["content_sha256_before_self_field"]
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def reseal_semantic_access_disclosure(
    *,
    output: Path = OUTPUT,
    root: Path = ROOT,
    builder_path: Path | None = None,
) -> dict:
    """Correct reviewed metadata using only the seal and derived projections."""

    builder_path = Path(__file__) if builder_path is None else builder_path
    value = json.loads(output.read_bytes())
    if value.get("schema") != SCHEMA or value.get("status") != (
        "sealed_source_disjoint_assignment_launch_still_gated"
    ):
        raise RuntimeError("semantic-access correction requires the sealed authority")
    _validate_self_addressed_artifact(value)
    projections = value.get("materialized_train_development_projections")
    if not isinstance(projections, dict) or set(projections) != {
        "train",
        "development",
    }:
        raise RuntimeError("reviewed derived projection ledger changed")
    for partition in ("train", "development"):
        for projection_kind in ("site_spans", "v440_qa"):
            item = projections.get(partition, {}).get(projection_kind)
            if not isinstance(item, dict):
                raise RuntimeError("reviewed derived projection receipt changed")
            relative = _lexical_repo_relative(item.get("path", ""), root=root)
            if file_sha256(root / relative) != item.get("file_sha256"):
                raise RuntimeError("derived projection changed before disclosure correction")
    builder_receipt = {
        "path": _lexical_repo_relative(builder_path, root=root),
        "file_sha256": file_sha256(builder_path),
    }
    corrected = _correct_v440_semantic_access_disclosure(
        value, builder_receipt=builder_receipt
    )
    _atomic_write_bytes(
        output,
        (
            json.dumps(corrected, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8"),
    )
    return corrected


def sealed_check(
    *,
    output: Path = OUTPUT,
    root: Path = ROOT,
    registry_path: Path = REGISTRY,
    v440_manifest_path: Path = V440_MANIFEST,
    builder_path: Path | None = None,
) -> dict:
    """Validate the sealed authority without touching mixed source files.

    Corpus Markdown, corpus manifests, and the mixed V440 projection may now
    contain final-partition data.  Their paths are compared only against safe
    registry/manifest metadata; this function never opens, hashes, or stats
    them.  Only the authority, registry, V440 metadata manifest, pinned
    tokenizer files, builder, and emitted train/development projections are
    opened.
    """

    builder_path = Path(__file__) if builder_path is None else builder_path
    value = json.loads(output.read_bytes())
    if value.get("schema") != SCHEMA:
        raise RuntimeError("unsupported sealed source split authority schema")
    if value.get("status") != "sealed_source_disjoint_assignment_launch_still_gated":
        raise RuntimeError("source split authority is not sealed")
    _validate_self_addressed_artifact(value)
    final = value.get("assignments", {}).get("final")
    if (
        not isinstance(final, dict)
        or final.get("records_redacted") is not True
        or "records" in final
        or value.get("invariants", {}).get("final_records_emitted") is not False
    ):
        raise RuntimeError("sealed final membership was emitted")

    receipts = value.get("safe_input_receipts")
    if not isinstance(receipts, list):
        raise RuntimeError("sealed safe input receipt ledger is missing")
    receipt_by_path = {
        item.get("path"): item.get("file_sha256")
        for item in receipts
        if isinstance(item, dict)
    }
    if len(receipt_by_path) != len(receipts):
        raise RuntimeError("sealed safe input receipt ledger is duplicated")

    registry_bytes = registry_path.read_bytes()
    registry_sha = sha256_bytes(registry_bytes)
    registry = json.loads(registry_bytes)
    binding = value.get("source_registry_binding", {})
    registry_relative = _lexical_repo_relative(registry_path, root=root)
    if (
        registry.get("schema") != "site-corpus-registry-v1"
        or binding.get("path") != registry_relative
        or binding.get("file_sha256") != registry_sha
        or receipt_by_path.get(registry_relative) != registry_sha
        or binding.get("source_tree_fingerprint_sha256")
        != registry.get("source_tree_fingerprint_sha256")
    ):
        raise RuntimeError("sealed source registry binding changed")
    artifacts = registry.get("artifacts")
    if (
        not isinstance(artifacts, list)
        or len(artifacts) != binding.get("artifact_count")
        or len(artifacts) != value.get("totals", {}).get("registered_site_artifacts")
    ):
        raise RuntimeError("sealed registered artifact membership changed")
    coverage_by_resource = {
        item.get("resource_id"): item
        for item in value.get("site_byte_coverage", [])
        if isinstance(item, dict)
    }
    if len(coverage_by_resource) != len(artifacts):
        raise RuntimeError("sealed site coverage ledger changed")
    for artifact in artifacts:
        resource_id = artifact.get("resource_id")
        markdown_relative = _lexical_repo_relative(
            artifact.get("markdown_path", ""), root=root
        )
        manifest_relative = _lexical_repo_relative(
            artifact.get("manifest_path", ""), root=root
        )
        # Metadata comparison only: do not resolve, stat, hash, or open either
        # mixed source path after sealing.
        if (
            receipt_by_path.get(markdown_relative) != artifact.get("markdown_sha256")
            or receipt_by_path.get(manifest_relative) != artifact.get("manifest_sha256")
        ):
            raise RuntimeError(f"sealed source receipt drifted: {resource_id}")
        coverage = coverage_by_resource.get(resource_id)
        if (
            coverage is None
            or coverage.get("artifact_id") != artifact.get("artifact_id")
            or coverage.get("markdown_path") != artifact.get("markdown_path")
            or coverage.get("markdown_sha256") != artifact.get("markdown_sha256")
            or coverage.get("registered_byte_length") != artifact.get("byte_length")
            or coverage.get("covered_byte_length") != artifact.get("byte_length")
        ):
            raise RuntimeError(f"sealed coverage metadata drifted: {resource_id}")

    tokenizer_binding = value.get("tokenizer_binding")
    if not isinstance(tokenizer_binding, dict):
        raise RuntimeError("sealed tokenizer binding is missing")
    for path_key, hash_key in (
        ("tokenizer_json_path", "tokenizer_json_sha256"),
        ("tokenizer_config_path", "tokenizer_config_sha256"),
    ):
        relative = _lexical_repo_relative(tokenizer_binding.get(path_key, ""), root=root)
        path = root / relative
        observed = file_sha256(path)
        if (
            observed != tokenizer_binding.get(hash_key)
            or receipt_by_path.get(relative) != observed
        ):
            raise RuntimeError("sealed tokenizer binding changed")

    v440_manifest_bytes = v440_manifest_path.read_bytes()
    v440_manifest_sha = sha256_bytes(v440_manifest_bytes)
    v440_manifest_relative = _lexical_repo_relative(v440_manifest_path, root=root)
    if receipt_by_path.get(v440_manifest_relative) != v440_manifest_sha:
        raise RuntimeError("sealed V440 metadata manifest changed")
    v440_manifest = json.loads(v440_manifest_bytes)
    projection = v440_manifest.get("projection")
    if not isinstance(projection, dict):
        raise RuntimeError("sealed V440 projection metadata is missing")
    projection_relative = _lexical_repo_relative(projection.get("path", ""), root=root)
    # Deliberately compare the declared digest without opening the mixed V440
    # projection now that some of its rows are terminal.
    if receipt_by_path.get(projection_relative) != projection.get("sha256"):
        raise RuntimeError("sealed V440 source projection receipt changed")

    builder_receipt = value.get("builder_receipt", {})
    builder_relative = _lexical_repo_relative(builder_path, root=root)
    if (
        builder_receipt.get("path") != builder_relative
        or builder_receipt.get("file_sha256") != file_sha256(builder_path)
    ):
        raise RuntimeError("sealed source split builder changed")

    projections = value.get("materialized_train_development_projections")
    if not isinstance(projections, dict) or set(projections) != {
        "train",
        "development",
    }:
        raise RuntimeError("sealed train/development projections are missing")
    for partition in ("train", "development"):
        for projection_kind in ("site_spans", "v440_qa"):
            item = projections.get(partition, {}).get(projection_kind)
            if not isinstance(item, dict):
                raise RuntimeError("sealed projection receipt changed schema")
            relative = _lexical_repo_relative(item.get("path", ""), root=root)
            observed = file_sha256(root / relative)
            if observed != item.get("file_sha256"):
                raise RuntimeError(
                    f"sealed {partition} {projection_kind} projection changed"
                )
            if item.get("contains_only_partition") != partition:
                raise RuntimeError("sealed projection partition label changed")

    if value.get("invariants", {}).get(
        "protected_holdout_ood_terminal_incident_or_manual_review_sources_opened"
    ) is not False:
        raise RuntimeError("sealed check access invariant changed")
    v440_summary = value.get("v440_source_projection_summary", {})
    invariants = value.get("invariants", {})
    if (
        v440_summary.get(
            "semantic_fields_read_for_tokenization_and_train_development_projection"
        )
        is not True
        or v440_summary.get("semantic_fields_used_for_split_assignment") is not False
        or v440_summary.get("final_semantic_records_emitted") is not False
        or invariants.get(
            "v440_semantic_fields_read_for_tokenization_and_train_development_projection"
        )
        is not True
        or invariants.get("v440_semantic_fields_used_for_split_assignment") is not False
        or invariants.get("v440_final_semantic_records_emitted") is not False
    ):
        raise RuntimeError("sealed V440 semantic-access disclosure changed")
    post_seal = value.get("construction_contract", {}).get(
        "post_seal_check_contract", {}
    )
    if (
        post_seal.get("original_site_markdown_opened_hashed_or_statted") is not False
        or post_seal.get("original_site_manifests_opened_hashed_or_statted") is not False
        or post_seal.get("mixed_v440_projection_opened_hashed_or_statted") is not False
        or post_seal.get("reconstruction_after_seal_forbidden") is not True
    ):
        raise RuntimeError("sealed non-reopening contract changed")
    return value


def build(*, check: bool = False) -> dict:
    if check:
        return sealed_check()
    if OUTPUT.exists():
        raise RuntimeError(
            "sealed source split authority already exists; use --check and consume "
            "only the materialized train/development projections"
        )
    payload = render()
    value = json.loads(payload)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{OUTPUT.name}.", dir=OUTPUT.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, OUTPUT)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--construct-unsealed", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--reseal-semantic-access-disclosure", action="store_true")
    arguments = parser.parse_args()
    if arguments.reseal_semantic_access_disclosure:
        value = reseal_semantic_access_disclosure()
    else:
        value = build(check=arguments.check)
    # Membership is deliberately never printed.  Only aggregate counts leave
    # the builder, including for the sealed final partition.
    print(
        json.dumps(
            {
                "output": _relative(OUTPUT),
                "content_sha256": value["content_sha256_before_self_field"],
                "train_source_groups": value["assignments"]["train"][
                    "source_group_count"
                ],
                "development_source_groups": value["assignments"]["development"][
                    "source_group_count"
                ],
                "final_source_groups": value["assignments"]["final"][
                    "source_group_count"
                ],
                "final_records_emitted": False,
                "training_launch_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
