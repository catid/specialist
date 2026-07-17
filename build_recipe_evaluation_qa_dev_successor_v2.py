#!/usr/bin/env python3
"""Seal the manually curated QA DEV bundle as an additive V2 successor.

The builder is deliberately narrow.  It may open only the four exact inputs
named below (the allowlist plus its three curated outputs) and the already
sealed V2 contract.  It never resolves, stats, hashes, or opens a caller-
supplied evaluation path.  The terminal side of the successor is copied only
as opaque identities from V2; no terminal source or claim-state path is
probed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "recipe_evaluation_qa_dev_successor_v2.json"
)
V2_CONTRACT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "recipe_evaluation_compute_contract_v2.json"
)
ALLOWLIST = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "manual_qa_dev_curation_allowlist_v1.json"
)
QA_DEV = (
    ROOT / "experiments/eggroll_es_hpo/datasets/manual_qa_dev_v2/"
    "manual_qa_dev.jsonl"
)
CURATION = (
    ROOT / "experiments/eggroll_es_hpo/datasets/manual_qa_dev_v2/"
    "manual_qa_dev.curation.jsonl"
)
REPORT = (
    ROOT / "experiments/eggroll_es_hpo/datasets/manual_qa_dev_v2/"
    "manual_qa_dev.report.json"
)
BUILDER = ROOT / "build_recipe_evaluation_qa_dev_successor_v2.py"

SCHEMA = "specialist-recipe-evaluation-qa-dev-successor-v2"
STATUS = "qa_dev_scoring_authorized_terminal_unclaimed_no_promotion"
CREATED_AT_UTC = "2026-07-17T00:00:00+00:00"
QA_SCHEMA = "specialist-manual-qa-dev-v2"
CURATION_SCHEMA = "specialist-manual-qa-dev-curation-v2"
REPORT_SCHEMA = "specialist-manual-qa-dev-report-v2"
ALLOWLIST_SCHEMA = "specialist-manual-qa-dev-curation-allowlist-v1"
V2_SCHEMA = "specialist-recipe-evaluation-compute-contract-v2"

EXPECTED_FILE_SHA256 = {
    V2_CONTRACT: "fdb018a73f81ff491246fdf6162910ed5ba27f49159a64dc2a1102bbe4cf3047",
    ALLOWLIST: "9012e251106895b1b0de6a8947099bbcec24ba3a6ed7ce171020e3d6a8638d05",
    QA_DEV: "df5567961a8fc6c475f633fdd89637825dbc00713c7a6ed1fca88020ce77dd6a",
    CURATION: "626eba5e7851a7097107530c55140e51a0faa5e92e7ef4e4210066cb47e1b7e2",
    REPORT: "0541f2e3db19fe846b71cc194287ab60e3202331af745cdd2131945b73d26265",
}
EXPECTED_V2_CONTENT_SHA256 = (
    "ffe0bd6fdb51f8bf96f4736091156a2c146bde3a66e8ed44ab5e3b21b1016e51"
)
EXPECTED_ALLOWLIST_CONTENT_SHA256 = (
    "c31b1c21fb22e74ac35691f5f20cce796b7037688784e052045f68f788bb18b3"
)
EXPECTED_DEV_SOURCE_IDENTITY_SET_SHA256 = (
    "3b40b564411ee883bef776d0ad9f0936bb159b95867e05d323604e67dfca05e0"
)
EXPECTED_TERMINAL_IDENTITY_SET_SHA256 = (
    "2ecc4d7bf7684b463f0957f5e1ccf5ce759a2e2a789bc9a25d369d2d88b07810"
)
EXPECTED_QA_ROWS = 20
EXPECTED_CURATION_ROWS = 20
EXPECTED_SOURCE_COUNT = 4
EXPECTED_QA_PER_SOURCE = 5
EXPECTED_TERMINAL_SOURCE_COUNT = 9

ALLOWED_SCORING_PURPOSES = frozenset(
    ("qa_dev_scoring", "general_quality_dev_scoring")
)
PROHIBITED_PURPOSES = frozenset(
    (
        "cpt",
        "sft",
        "training",
        "model_adaptation",
        "gradient_update",
        "es_reward_training",
        "hpo_promotion",
        "recipe_selection",
        "checkpoint_selection",
        "terminal_evaluation",
        "ood_evaluation",
        "final_benchmark",
    )
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
URL_RE = re.compile(r"(?:https?://|www\.|\b[a-z0-9-]+\.(?:com|org|net)\b)", re.I)
SOURCE_TRIVIA_RE = re.compile(
    r"\b(?:canonical\s+)?(?:url|website address|web address|page number|"
    r"source title|document title|file name)\b",
    re.I,
)
CHAT_CONTROL_RE = re.compile(
    r"(?:<\|im_(?:start|end)\|>|<\|endoftext\|>|</?think>)", re.I
)
TERMINAL_SEMANTIC_KEYS = frozenset(
    (
        "question",
        "answer",
        "excerpt",
        "prompt",
        "completion",
        "response",
        "text",
        "title",
        "url",
        "urls",
        "path",
        "relative_path",
        "source_path",
    )
)
QA_KEYS = frozenset(
    (
        "schema",
        "id",
        "split",
        "role",
        "question",
        "answer",
        "source_document",
        "grounding",
        "reviewer",
        "curation",
    )
)
SOURCE_KEYS = frozenset(
    (
        "repository_relative_path",
        "opaque_item_identity",
        "source_path_identity_sha256",
        "file_sha256",
    )
)
CURATION_KEYS = frozenset(
    (
        "schema",
        "qa_id",
        "source_document_identity",
        "grounding",
        "reviewer",
        "curation",
    )
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _lexical_absolute(path: Path | str) -> Path:
    """Return an absolute lexical path without filesystem resolution."""

    return Path(os.path.abspath(os.fspath(path)))


ALLOWED_SEMANTIC_INPUTS = frozenset(
    _lexical_absolute(path) for path in (ALLOWLIST, QA_DEV, CURATION, REPORT)
)
ALLOWED_METADATA_INPUTS = frozenset((_lexical_absolute(V2_CONTRACT),))


def _guard_exact_path(
    path: Path | str, *, allowed: frozenset[Path], label: str
) -> Path:
    lexical = _lexical_absolute(path)
    if lexical not in allowed:
        raise RuntimeError(
            f"{label} denied before resolution, stat, hash, or open"
        )
    return lexical


def _read_nofollow(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("sealed input is not a regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def read_allowed_semantic_bytes(path: Path | str) -> bytes:
    return _read_nofollow(
        _guard_exact_path(
            path, allowed=ALLOWED_SEMANTIC_INPUTS, label="semantic input"
        )
    )


def read_allowed_metadata_bytes(path: Path | str) -> bytes:
    return _read_nofollow(
        _guard_exact_path(
            path, allowed=ALLOWED_METADATA_INPUTS, label="metadata input"
        )
    )


def _decode_json(payload: bytes, label: str) -> dict:
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _decode_jsonl(payload: bytes, label: str) -> list[dict]:
    rows = [
        json.loads(line)
        for line in payload.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"{label} must be nonempty object JSONL")
    return rows


def _without_self_hash(value: dict) -> dict:
    return {
        key: item
        for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _require_self_hash(value: dict, expected: str, label: str) -> None:
    observed = value.get("content_sha256_before_self_field")
    calculated = canonical_sha256(_without_self_hash(value))
    if observed != expected or calculated != expected:
        raise RuntimeError(f"{label} content binding changed")


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"{label} must be an opaque SHA-256 identity")
    return value


def _require_commit(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise RuntimeError(f"{label} must be an opaque Git commit identity")
    return value


def _assert_no_terminal_semantics(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).casefold() in TERMINAL_SEMANTIC_KEYS:
                raise RuntimeError("terminal boundary contains semantic leakage")
            _assert_no_terminal_semantics(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_terminal_semantics(item)


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _source_tuple(source: dict) -> tuple[str, str, str, str]:
    if not SOURCE_KEYS.issubset(source):
        raise RuntimeError("source binding lacks required identity fields")
    path = source.get("repository_relative_path")
    if not isinstance(path, str) or not path.endswith("/CORPUS.md"):
        raise RuntimeError("QA source path is not an exact corpus document")
    return (
        path,
        _require_sha(source.get("opaque_item_identity"), "opaque item"),
        _require_sha(
            source.get("source_path_identity_sha256"), "source path identity"
        ),
        _require_sha(source.get("file_sha256"), "source file identity"),
    )


def validate_allowlist(allowlist: dict, *, production: bool = True) -> list[dict]:
    if allowlist.get("schema") != ALLOWLIST_SCHEMA:
        raise RuntimeError("unexpected manual QA allowlist schema")
    if production:
        _require_self_hash(
            allowlist, EXPECTED_ALLOWLIST_CONTENT_SHA256, "manual QA allowlist"
        )
    sources = allowlist.get("allowed_dev_sources")
    if not isinstance(sources, list) or len(sources) != EXPECTED_SOURCE_COUNT:
        raise RuntimeError("manual QA allowlist must bind four DEV sources")
    tuples = []
    for source in sources:
        if not isinstance(source, dict):
            raise RuntimeError("invalid DEV source allowlist entry")
        if source.get("permission") != (
            "read_for_manual_multi_item_qa_dev_curation_only"
        ):
            raise RuntimeError("DEV source has unexpected curation permission")
        tuples.append(
            (
                source.get("repository_relative_path"),
                _require_sha(source.get("opaque_item_identity"), "opaque item"),
                _require_sha(
                    source.get("source_path_identity_sha256"),
                    "source path identity",
                ),
                _require_sha(source.get("file_sha256"), "source file identity"),
            )
        )
    if len(set(tuples)) != EXPECTED_SOURCE_COUNT:
        raise RuntimeError("manual QA allowlist source identities are not unique")
    if allowlist.get("authority") != {
        "manual_multi_item_qa_dev_curation_authorized": True,
        "manual_read_of_exact_dev_sources_authorized": True,
        "model_inference_training_or_embedding_authorized": False,
        "qa_hpo_or_quality_promotion_authorized": False,
        "recursive_search_glob_or_directory_scan_authorized": False,
    }:
        raise RuntimeError("manual QA allowlist authority changed")
    terminal = allowlist.get("terminal_boundary")
    if not isinstance(terminal, dict):
        raise RuntimeError("manual QA allowlist lacks terminal boundary")
    _assert_no_terminal_semantics(terminal)
    if terminal != {
        "terminal_file_hashes_persisted": False,
        "terminal_opaque_identity_set_sha256": EXPECTED_TERMINAL_IDENTITY_SET_SHA256,
        "terminal_paths_persisted": False,
        "terminal_read_or_resolution_authorized": False,
        "terminal_source_count": EXPECTED_TERMINAL_SOURCE_COUNT,
    }:
        raise RuntimeError("manual QA allowlist terminal boundary changed")
    return sources


def validate_qa_rows(
    rows: list[dict], allowed_sources: Iterable[dict]
) -> dict:
    source_by_tuple = {_source_tuple(source): source for source in allowed_sources}
    if len(rows) != EXPECTED_QA_ROWS:
        raise RuntimeError("manual QA DEV must contain exactly 20 rows")
    ids = []
    normalized_questions = []
    normalized_answers = []
    source_counts: Counter[tuple[str, str, str, str]] = Counter()
    type_counts: Counter[str] = Counter()
    for row in rows:
        if frozenset(row) != QA_KEYS or row.get("schema") != QA_SCHEMA:
            raise RuntimeError("manual QA row schema or fields changed")
        qa_id = row.get("id")
        question = row.get("question")
        answer = row.get("answer")
        if not all(isinstance(value, str) and value.strip() for value in (
            qa_id, question, answer
        )):
            raise RuntimeError("manual QA row has an empty identity, question, or answer")
        if row.get("split") != "dev" or row.get("role") != (
            "dev_only_never_model_adaptation"
        ):
            raise RuntimeError("manual QA row escaped the DEV-only role")
        combined = "\n".join((question, answer))
        if URL_RE.search(combined) or SOURCE_TRIVIA_RE.search(question):
            raise RuntimeError("manual QA row contains URL or source trivia")
        if CHAT_CONTROL_RE.search(combined):
            raise RuntimeError("manual QA row contains chat-control semantic leakage")
        source = row.get("source_document")
        if not isinstance(source, dict) or frozenset(source) != SOURCE_KEYS:
            raise RuntimeError("manual QA row lacks source binding")
        source_identity = _source_tuple(source)
        if source_identity not in source_by_tuple:
            raise RuntimeError("manual QA row is outside the DEV source allowlist")
        grounding = row.get("grounding")
        if not isinstance(grounding, dict) or frozenset(grounding) != {
            "evidence", "locator", "support"
        }:
            raise RuntimeError("manual QA row lacks exact grounding fields")
        if grounding.get("support") != "direct" or not all(
            isinstance(grounding.get(key), str) and grounding[key].strip()
            for key in ("evidence", "locator")
        ):
            raise RuntimeError("manual QA grounding is not direct and nonempty")
        if URL_RE.search(grounding["evidence"]) or CHAT_CONTROL_RE.search(
            grounding["evidence"]
        ):
            raise RuntimeError("manual QA evidence contains semantic leakage")
        reviewer = row.get("reviewer")
        if not isinstance(reviewer, dict) or not all(
            isinstance(reviewer.get(key), str) and reviewer[key].strip()
            for key in ("reviewer_id", "reviewer_type", "reviewed_at_utc")
        ):
            raise RuntimeError("manual QA row lacks manual reviewer evidence")
        curation = row.get("curation")
        if not isinstance(curation, dict) or curation != {
            "method": "manual_exact_source_read",
            "decision": "include",
            "grounding_check": "passed",
            "scope_check": "passed",
            "safety_check": "passed",
            "url_or_source_trivia": False,
        }:
            raise RuntimeError("manual QA row lacks passing curation evidence")
        ids.append(qa_id)
        normalized_questions.append(_normalized_text(question))
        normalized_answers.append(_normalized_text(answer))
        source_counts[source_identity] += 1
        type_counts[qa_id.split("_")[1] if "_" in qa_id else "unknown"] += 1
    if len(ids) != len(set(ids)):
        raise RuntimeError("manual QA identities are not unique")
    if len(normalized_questions) != len(set(normalized_questions)):
        raise RuntimeError("manual QA questions are not unique")
    if len(normalized_answers) != len(set(normalized_answers)):
        raise RuntimeError("manual QA answers are not unique")
    if set(source_counts) != set(source_by_tuple) or set(source_counts.values()) != {
        EXPECTED_QA_PER_SOURCE
    }:
        raise RuntimeError("manual QA rows must contribute five items per source")
    return {
        "qa_row_count": len(rows),
        "unique_qa_id_count": len(set(ids)),
        "unique_normalized_question_count": len(set(normalized_questions)),
        "unique_normalized_answer_count": len(set(normalized_answers)),
        "source_count": len(source_counts),
        "per_source_counts": sorted(source_counts.values()),
        "qa_id_set_sha256": canonical_sha256(sorted(ids)),
        "normalized_question_set_sha256": canonical_sha256(
            sorted(normalized_questions)
        ),
        "all_rows_dev_only_never_model_adaptation": True,
        "all_rows_directly_grounded": True,
        "all_rows_manually_reviewed": True,
        "url_or_source_trivia_rows": 0,
        "chat_control_leakage_rows": 0,
    }


def validate_curation_rows(
    rows: list[dict], qa_rows: list[dict]
) -> dict:
    if len(rows) != EXPECTED_CURATION_ROWS:
        raise RuntimeError("manual QA curation must contain exactly 20 rows")
    qa_by_id = {row["id"]: row for row in qa_rows}
    seen = set()
    question_types: Counter[str] = Counter()
    for row in rows:
        if frozenset(row) != CURATION_KEYS or row.get("schema") != CURATION_SCHEMA:
            raise RuntimeError("manual QA curation row schema or fields changed")
        qa_id = row.get("qa_id")
        if qa_id not in qa_by_id or qa_id in seen:
            raise RuntimeError("manual QA curation identity mismatch")
        seen.add(qa_id)
        expected_source = {
            key: qa_by_id[qa_id]["source_document"][key]
            for key in (
                "opaque_item_identity",
                "source_path_identity_sha256",
                "file_sha256",
            )
        }
        if row.get("source_document_identity") != expected_source:
            raise RuntimeError("manual QA curation source identity mismatch")
        grounding = row.get("grounding")
        if not isinstance(grounding, dict) or grounding.get("alignment") != (
            "direct"
        ) or grounding.get("answer_faithfulness") != "passed" or not isinstance(
            grounding.get("evidence_locator"), str
        ) or not grounding["evidence_locator"].strip():
            raise RuntimeError("manual QA curation grounding did not pass")
        curation = row.get("curation")
        if not isinstance(curation, dict) or any(
            curation.get(key) != expected
            for key, expected in {
                "decision": "include",
                "usefulness": "passed",
                "clarity": "passed",
                "safety": "passed",
                "source_or_url_trivia": False,
                "weird_list_recall": False,
                "unsupported_advice": False,
            }.items()
        ):
            raise RuntimeError("manual QA curation quality checks did not pass")
        question_type = curation.get("question_type")
        if not isinstance(question_type, str) or not question_type:
            raise RuntimeError("manual QA curation lacks a useful question type")
        reviewer = row.get("reviewer")
        if reviewer != qa_by_id[qa_id]["reviewer"]:
            raise RuntimeError("manual QA curation reviewer mismatch")
        question_types[question_type] += 1
    if seen != set(qa_by_id):
        raise RuntimeError("manual QA rows are missing curation records")
    return {
        "curation_row_count": len(rows),
        "qa_and_curation_id_sets_match": True,
        "all_decisions_include": True,
        "all_usefulness_clarity_safety_checks_passed": True,
        "all_source_or_url_trivia_checks_false": True,
        "all_weird_list_recall_checks_false": True,
        "all_unsupported_advice_checks_false": True,
        "question_type_counts": dict(sorted(question_types.items())),
    }


def validate_report(
    report: dict,
    *,
    qa_audit: dict,
    curation_audit: dict,
    allowed_sources: list[dict],
) -> None:
    if report.get("schema") != REPORT_SCHEMA or report.get("status") != (
        "curation_complete_validation_passed"
    ):
        raise RuntimeError("manual QA report schema or status changed")
    contract = report.get("contract", {})
    if contract.get("allowlist_content_sha256_before_self_field") != (
        EXPECTED_ALLOWLIST_CONTENT_SHA256
    ) or contract.get("v2_contract_content_sha256") != (
        EXPECTED_V2_CONTENT_SHA256
    ) or contract.get("v2_contract_file_sha256") != EXPECTED_FILE_SHA256[
        V2_CONTRACT
    ] or contract.get("dev_source_identity_set_sha256") != (
        EXPECTED_DEV_SOURCE_IDENTITY_SET_SHA256
    ):
        raise RuntimeError("manual QA report predecessor bindings changed")
    scope = report.get("scope", {})
    if scope.get("role") != "dev_only_never_model_adaptation" or any(
        scope.get(key) is not False
        for key in (
            "authorized_for_model_adaptation",
            "authorized_for_qa_hpo_or_general_quality_promotion",
            "terminal_or_holdout_data_accessed",
            "model_inference_training_or_embedding_performed",
            "gpu_work_performed",
        )
    ) or scope.get("successor_contract_validation_required_before_use") is not True:
        raise RuntimeError("manual QA report authority changed")
    validation = report.get("validation", {})
    expected_validation = {
        "qa_record_count": qa_audit["qa_row_count"],
        "curation_record_count": curation_audit["curation_row_count"],
        "unique_qa_id_count": qa_audit["unique_qa_id_count"],
        "unique_curation_qa_id_count": curation_audit["curation_row_count"],
        "qa_and_curation_id_sets_match": True,
        "source_count": EXPECTED_SOURCE_COUNT,
        "per_source_counts": [EXPECTED_QA_PER_SOURCE] * EXPECTED_SOURCE_COUNT,
        "qa_required_fields_valid": True,
        "curation_required_fields_valid": True,
        "all_grounding_marked_direct": True,
        "all_curation_decisions_include": True,
        "questions_containing_urls": 0,
    }
    if any(validation.get(key) != value for key, value in expected_validation.items()):
        raise RuntimeError("manual QA report validation claims changed")
    report_sources = report.get("sources")
    if not isinstance(report_sources, list) or len(report_sources) != (
        EXPECTED_SOURCE_COUNT
    ):
        raise RuntimeError("manual QA report source inventory changed")
    allowlist_by_tuple = {_source_tuple(source) for source in allowed_sources}
    report_tuples = set()
    for source in report_sources:
        source_copy = dict(source)
        if source_copy.pop("qa_item_count", None) != EXPECTED_QA_PER_SOURCE:
            raise RuntimeError("manual QA report per-source count changed")
        if frozenset(source_copy) != SOURCE_KEYS:
            raise RuntimeError("manual QA report source fields changed")
        report_tuples.add(_source_tuple(source_copy))
    if report_tuples != allowlist_by_tuple:
        raise RuntimeError("manual QA report source inventory mismatch")


def audit_source_disjointness(
    dev_sources: Iterable[dict], terminal_sources: Iterable[dict]
) -> dict:
    """Compare only opaque source identities; semantic content is forbidden."""

    dev_tuples = {_source_tuple(source) for source in dev_sources}
    terminal_list = list(terminal_sources)
    _assert_no_terminal_semantics(terminal_list)
    terminal_opaque = {
        _require_sha(source.get("opaque_item_identity"), "terminal opaque item")
        for source in terminal_list
    }
    terminal_paths = {
        _require_sha(
            source.get("source_path_identity_sha256"),
            "terminal source path identity",
        )
        for source in terminal_list
    }
    terminal_files = {
        _require_sha(source.get("corpus_file_sha256"), "terminal corpus file")
        for source in terminal_list
    }
    intersections = {
        "opaque_item_identity": len({item[1] for item in dev_tuples} & terminal_opaque),
        "source_path_identity_sha256": len(
            {item[2] for item in dev_tuples} & terminal_paths
        ),
        "corpus_file_sha256": len({item[3] for item in dev_tuples} & terminal_files),
    }
    if any(intersections.values()):
        raise RuntimeError("manual QA DEV overlaps terminal via opaque identity")
    return intersections


def validate_v2_predecessor(v2: dict, allowed_sources: list[dict]) -> dict:
    if v2.get("schema") != V2_SCHEMA or v2.get("content_sha256_before_self_field") != (
        EXPECTED_V2_CONTENT_SHA256
    ):
        raise RuntimeError("unexpected V2 predecessor")
    _require_self_hash(v2, EXPECTED_V2_CONTENT_SHA256, "V2 predecessor")
    if v2.get("authority") != {
        "qa_hpo_or_general_quality_promotion_authorized": False,
        "separate_source_disjoint_multi_item_qa_dev_required": True,
        "systems_and_prose_logprob_tuning_only": True,
        "terminal_access_authorized": False,
    }:
        raise RuntimeError("V2 predecessor authority changed")
    roles = v2.get("roles")
    if not isinstance(roles, dict):
        raise RuntimeError("V2 predecessor roles missing")
    dev = roles.get("dev")
    terminal = roles.get("protected_terminal")
    if not isinstance(dev, dict) or not isinstance(terminal, dict):
        raise RuntimeError("V2 predecessor DEV or terminal role missing")
    expected_dev = {_source_tuple(source) for source in allowed_sources}
    observed_dev = set()
    for source in dev.get("source_bindings", ()):
        observed_dev.add(
            (
                next(
                    item[0]
                    for item in expected_dev
                    if item[1] == source.get("opaque_item_identity")
                ),
                _require_sha(source.get("opaque_item_identity"), "V2 DEV opaque item"),
                _require_sha(
                    source.get("source_path_identity_sha256"),
                    "V2 DEV source path",
                ),
                _require_sha(source.get("corpus_file_sha256"), "V2 DEV file"),
            )
        )
    if observed_dev != expected_dev or dev.get("source_identity_set_sha256") != (
        EXPECTED_DEV_SOURCE_IDENTITY_SET_SHA256
    ) or dev.get("terminal_identity_intersection") != 0 or dev.get(
        "removed_from_terminal_eligibility"
    ) is not True:
        raise RuntimeError("manual QA sources do not match reserved V2 DEV sources")
    _assert_no_terminal_semantics(terminal)
    selected_ids = terminal.get("selected_opaque_item_identities")
    selected_sources = terminal.get("selected_sources")
    if not isinstance(selected_ids, list) or not isinstance(selected_sources, list):
        raise RuntimeError("V2 terminal opaque identity inventory missing")
    if len(selected_ids) != EXPECTED_TERMINAL_SOURCE_COUNT or len(
        selected_sources
    ) != EXPECTED_TERMINAL_SOURCE_COUNT or len(set(selected_ids)) != len(selected_ids):
        raise RuntimeError("V2 terminal opaque identity inventory changed")
    if terminal.get("rows") != EXPECTED_TERMINAL_SOURCE_COUNT or terminal.get(
        "documents"
    ) != EXPECTED_TERMINAL_SOURCE_COUNT or terminal.get(
        "selected_identity_set_sha256"
    ) != EXPECTED_TERMINAL_IDENTITY_SET_SHA256 or terminal.get(
        "access_authorized_by_this_contract"
    ) is not False or terminal.get("selection_or_tuning_use") != "prohibited":
        raise RuntimeError("V2 terminal authority or identity binding changed")
    terminal_by_id = {}
    for source in selected_sources:
        if not isinstance(source, dict):
            raise RuntimeError("V2 terminal source identity is invalid")
        opaque_id = _require_sha(
            source.get("opaque_item_identity"), "terminal opaque item"
        )
        terminal_by_id[opaque_id] = {
            "added_commit": _require_commit(
                source.get("added_commit"), "added commit"
            ),
            "corpus_file_sha256": _require_sha(
                source.get("corpus_file_sha256"), "terminal corpus file"
            ),
            "metadata_bundle_sha256": _require_sha(
                source.get("metadata_bundle_sha256"), "terminal metadata bundle"
            ),
            "opaque_item_identity": opaque_id,
            "source_path_identity_sha256": _require_sha(
                source.get("source_path_identity_sha256"),
                "terminal source path identity",
            ),
            "url_identity_set_sha256": _require_sha(
                source.get("url_identity_set_sha256"), "terminal URL identity set"
            ),
        }
    if set(terminal_by_id) != set(selected_ids):
        raise RuntimeError("V2 terminal selected identity lists disagree")
    intersections = audit_source_disjointness(
        allowed_sources, terminal_by_id.values()
    )
    return {
        "terminal_sources": [terminal_by_id[key] for key in sorted(terminal_by_id)],
        "terminal_selected_identity_set_sha256": terminal[
            "selected_identity_set_sha256"
        ],
        "terminal_role_content_sha256": canonical_sha256(terminal),
        "intersections": intersections,
    }


def assert_authorized_use(purpose: str) -> None:
    if purpose in ALLOWED_SCORING_PURPOSES:
        return
    if purpose in PROHIBITED_PURPOSES:
        raise RuntimeError(f"{purpose} is not authorized by the QA DEV successor")
    raise RuntimeError("unknown purpose fails closed")


def _artifact_binding(path: Path, payload: bytes, record_count: int) -> dict:
    return {
        "path": _relative(path),
        "file_sha256": bytes_sha256(payload),
        "record_count": record_count,
    }


def build_contract() -> dict:
    payloads = {
        ALLOWLIST: read_allowed_semantic_bytes(ALLOWLIST),
        QA_DEV: read_allowed_semantic_bytes(QA_DEV),
        CURATION: read_allowed_semantic_bytes(CURATION),
        REPORT: read_allowed_semantic_bytes(REPORT),
        V2_CONTRACT: read_allowed_metadata_bytes(V2_CONTRACT),
    }
    for path, expected in EXPECTED_FILE_SHA256.items():
        observed = bytes_sha256(payloads[path])
        if observed != expected:
            raise RuntimeError(f"sealed input bytes changed: {_relative(path)}")
    allowlist = _decode_json(payloads[ALLOWLIST], "manual QA allowlist")
    qa_rows = _decode_jsonl(payloads[QA_DEV], "manual QA DEV")
    curation_rows = _decode_jsonl(payloads[CURATION], "manual QA curation")
    report = _decode_json(payloads[REPORT], "manual QA report")
    v2 = _decode_json(payloads[V2_CONTRACT], "V2 predecessor")
    sources = validate_allowlist(allowlist)
    qa_audit = validate_qa_rows(qa_rows, sources)
    curation_audit = validate_curation_rows(curation_rows, qa_rows)
    validate_report(
        report,
        qa_audit=qa_audit,
        curation_audit=curation_audit,
        allowed_sources=sources,
    )
    predecessor_audit = validate_v2_predecessor(v2, sources)
    source_bindings = [
        {
            "repository_relative_path": source["repository_relative_path"],
            "opaque_item_identity": source["opaque_item_identity"],
            "source_path_identity_sha256": source[
                "source_path_identity_sha256"
            ],
            "corpus_file_sha256": source["file_sha256"],
            "qa_item_count": EXPECTED_QA_PER_SOURCE,
        }
        for source in sorted(sources, key=lambda item: item["opaque_item_identity"])
    ]
    contract = {
        "schema": SCHEMA,
        "status": STATUS,
        "created_at_utc": CREATED_AT_UTC,
        "purpose": (
            "Add source-disjoint, manually grounded multi-item QA DEV scoring "
            "authority to the sealed V2 systems/prose boundary without model "
            "adaptation, promotion, OOD, or terminal authority."
        ),
        "predecessor": {
            "path": _relative(V2_CONTRACT),
            "schema": V2_SCHEMA,
            "file_sha256": EXPECTED_FILE_SHA256[V2_CONTRACT],
            "content_sha256_before_self_field": EXPECTED_V2_CONTENT_SHA256,
            "status": v2["status"],
            "history_rewritten": False,
        },
        "bound_inputs": {
            "allowlist": _artifact_binding(ALLOWLIST, payloads[ALLOWLIST], 1),
            "qa_dev": _artifact_binding(QA_DEV, payloads[QA_DEV], len(qa_rows)),
            "curation": _artifact_binding(
                CURATION, payloads[CURATION], len(curation_rows)
            ),
            "report": _artifact_binding(REPORT, payloads[REPORT], 1),
        },
        "qa_dev": {
            "role": "dev_only_never_model_adaptation",
            "source_identity_set_sha256": EXPECTED_DEV_SOURCE_IDENTITY_SET_SHA256,
            "source_bindings": source_bindings,
            "quality_audit": qa_audit | curation_audit,
            "raw_semantic_rows_persisted_in_successor": False,
            "scoring_input_loader": "load_authorized_qa_dev_rows",
        },
        "source_disjointness": {
            "passed": True,
            "comparison": "four exact V2 DEV identities vs nine opaque terminal identities",
            "intersection_counts": predecessor_audit["intersections"],
            "v2_full_role_audit_passed": v2.get("disjointness", {}).get(
                "audit", {}
            ).get("passed") is True,
            "future_adaptation_reservation_inherited": True,
            "missing_or_unverifiable_provenance": "reject_fail_closed",
        },
        "terminal_boundary": {
            "source_count": EXPECTED_TERMINAL_SOURCE_COUNT,
            "selected_identity_set_sha256": predecessor_audit[
                "terminal_selected_identity_set_sha256"
            ],
            "v2_terminal_role_content_sha256": predecessor_audit[
                "terminal_role_content_sha256"
            ],
            "selected_sources_opaque_only": predecessor_audit["terminal_sources"],
            "semantic_source_fields_persisted": False,
            "terminal_source_opened_or_resolved_by_builder": False,
            "terminal_claim_state_probed_by_builder": False,
            "terminal_claim_created_by_successor": False,
            "terminal_access_authorized": False,
            "one_shot_status": (
                "inherited_unclaimed_from_bound_v2_and_untouched_by_successor"
            ),
        },
        "authority": {
            "qa_dev_scoring_authorized": True,
            "general_quality_dev_scoring_authorized": True,
            "model_adaptation_or_training_authorized": False,
            "gradient_or_optimizer_update_authorized": False,
            "qa_rows_may_enter_training_sampler": False,
            "qa_hpo_or_recipe_promotion_authorized": False,
            "checkpoint_selection_authorized": False,
            "ood_evaluation_authorized": False,
            "terminal_evaluation_authorized": False,
            "final_benchmark_authorized": False,
        },
        "downstream_blockers": {
            "hpo_or_recipe_promotion": (
                "requires an additive downstream execution contract binding "
                "the scorer, candidate set, budgets, stopping rule, and OOD guard"
            ),
            "terminal_evaluation": (
                "requires a frozen recipe plus the explicit one-use aggregate runner"
            ),
            "final_benchmark": (
                "requires successful DEV/OOD gates and a separate terminal claim"
            ),
        },
        "implementation": {
            "builder_path": _relative(BUILDER),
            "builder_file_sha256": bytes_sha256(_read_nofollow(BUILDER)),
            "path_policy": (
                "exact lexical allowlist then O_NOFOLLOW regular-file read"
            ),
            "protected_or_unlisted_path_resolution_stat_hash_open": False,
        },
    }
    _assert_no_terminal_semantics(contract["terminal_boundary"])
    contract["content_sha256_before_self_field"] = canonical_sha256(contract)
    validate_contract(contract)
    return contract


def validate_contract(contract: dict) -> None:
    expected_authority = {
        "qa_dev_scoring_authorized": True,
        "general_quality_dev_scoring_authorized": True,
        "model_adaptation_or_training_authorized": False,
        "gradient_or_optimizer_update_authorized": False,
        "qa_rows_may_enter_training_sampler": False,
        "qa_hpo_or_recipe_promotion_authorized": False,
        "checkpoint_selection_authorized": False,
        "ood_evaluation_authorized": False,
        "terminal_evaluation_authorized": False,
        "final_benchmark_authorized": False,
    }
    terminal = contract.get("terminal_boundary")
    if not isinstance(terminal, dict):
        raise RuntimeError("QA DEV successor lacks terminal boundary")
    _assert_no_terminal_semantics(terminal)
    value_without_hash = _without_self_hash(contract)
    if (
        contract.get("schema") != SCHEMA
        or contract.get("status") != STATUS
        or contract.get("content_sha256_before_self_field")
        != canonical_sha256(value_without_hash)
        or contract.get("predecessor", {}).get("file_sha256")
        != EXPECTED_FILE_SHA256[V2_CONTRACT]
        or contract.get("predecessor", {}).get(
            "content_sha256_before_self_field"
        )
        != EXPECTED_V2_CONTENT_SHA256
        or contract.get("bound_inputs", {}).get("allowlist", {}).get(
            "file_sha256"
        )
        != EXPECTED_FILE_SHA256[ALLOWLIST]
        or contract.get("bound_inputs", {}).get("qa_dev", {}).get(
            "file_sha256"
        )
        != EXPECTED_FILE_SHA256[QA_DEV]
        or contract.get("bound_inputs", {}).get("curation", {}).get(
            "file_sha256"
        )
        != EXPECTED_FILE_SHA256[CURATION]
        or contract.get("bound_inputs", {}).get("report", {}).get(
            "file_sha256"
        )
        != EXPECTED_FILE_SHA256[REPORT]
        or contract.get("qa_dev", {}).get("role")
        != "dev_only_never_model_adaptation"
        or contract.get("qa_dev", {}).get("source_identity_set_sha256")
        != EXPECTED_DEV_SOURCE_IDENTITY_SET_SHA256
        or len(contract.get("qa_dev", {}).get("source_bindings", ()))
        != EXPECTED_SOURCE_COUNT
        or contract.get("qa_dev", {}).get("quality_audit", {}).get(
            "qa_row_count"
        )
        != EXPECTED_QA_ROWS
        or contract.get("qa_dev", {}).get("quality_audit", {}).get(
            "curation_row_count"
        )
        != EXPECTED_CURATION_ROWS
        or contract.get("source_disjointness", {}).get("passed") is not True
        or any(
            contract.get("source_disjointness", {})
            .get("intersection_counts", {})
            .values()
        )
        or terminal.get("source_count") != EXPECTED_TERMINAL_SOURCE_COUNT
        or terminal.get("selected_identity_set_sha256")
        != EXPECTED_TERMINAL_IDENTITY_SET_SHA256
        or len(terminal.get("selected_sources_opaque_only", ()))
        != EXPECTED_TERMINAL_SOURCE_COUNT
        or terminal.get("semantic_source_fields_persisted") is not False
        or terminal.get("terminal_source_opened_or_resolved_by_builder") is not False
        or terminal.get("terminal_claim_state_probed_by_builder") is not False
        or terminal.get("terminal_claim_created_by_successor") is not False
        or terminal.get("terminal_access_authorized") is not False
        or contract.get("authority") != expected_authority
    ):
        raise RuntimeError("invalid recipe evaluation QA DEV successor V2")


def load_authorized_qa_dev_rows(purpose: str) -> list[dict]:
    """Load only the sealed QA rows for an explicitly authorized DEV scorer."""

    assert_authorized_use(purpose)
    contract = _decode_json(_read_nofollow(OUTPUT), "QA DEV successor")
    validate_contract(contract)
    payload = read_allowed_semantic_bytes(QA_DEV)
    if bytes_sha256(payload) != contract["bound_inputs"]["qa_dev"]["file_sha256"]:
        raise RuntimeError("sealed QA DEV bytes changed")
    rows = _decode_jsonl(payload, "manual QA DEV")
    allowlist = _decode_json(
        read_allowed_semantic_bytes(ALLOWLIST), "manual QA allowlist"
    )
    validate_qa_rows(rows, validate_allowlist(allowlist))
    return rows


def _serialized(contract: dict) -> bytes:
    return (json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = _serialized(build_contract())
    if args.write:
        _atomic_write(OUTPUT, payload)
        print(
            json.dumps(
                {
                    "output": _relative(OUTPUT),
                    "file_sha256": bytes_sha256(payload),
                    "content_sha256_before_self_field": json.loads(payload)[
                        "content_sha256_before_self_field"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    if _read_nofollow(OUTPUT) != payload:
        raise RuntimeError("QA DEV successor bytes are not current")
    print(
        json.dumps(
            {
                "status": "ok",
                "output": _relative(OUTPUT),
                "file_sha256": bytes_sha256(payload),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
