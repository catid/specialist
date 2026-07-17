#!/usr/bin/env python3
"""Merge validated QA tranches into one deterministic future-training set."""
from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from build_leakfree_qa import eval_facts
from qa_quality import (LOW_VALUE, has_protocol_tokens, leakage_reason,
                        normalize_text, parse_qa, qa_pair_from_record,
                        stable_fact_id)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEFAULT_INPUTS = [
    DATA / "train_qa_verified_leakfree_v2.jsonl",
    DATA / "train_qa_manual_v1.jsonl",
    DATA / "rope_resource_qa_v1.jsonl",
    DATA / "rope_resource_factual_qa_v1.jsonl",
    DATA / "rope_resource_manual_v1.jsonl",
    DATA / "rope_topia_manual_v1.jsonl",
]
DEFAULT_OUTPUT = DATA / "train_qa_curated_v1.jsonl"
DEFAULT_REPORT = DATA / "train_qa_curated_v1.report.json"
DEFAULT_CURATIONS = [
    DATA / "train_qa_curated_v1.curation.jsonl",
    DATA / "train_qa_kinbakutoday.curation.jsonl",
]
QUARANTINED_LEGACY_EVAL = frozenset((
    Path(os.path.abspath(DATA / "eval_qa.jsonl")),
    Path(os.path.abspath(DATA / "eval_qa_v2.jsonl")),
))
CURATION_ACTIONS = {"drop", "edit"}
EDIT_SUPPORT_TYPES = {"extractive", "manual_paraphrase"}
OPAQUE_COLLISION_SCHEMA = "curated-qa-opaque-collision-authorization-v1"
OPAQUE_IDENTITY_SCHEME = "sha256-canonical-question-answer-v1"
SHA256_HEX_LENGTH = 64
FORBIDDEN_PATH_TOKENS = frozenset({
    "eval",
    "evaluation",
    "evaluations",
    "holdout",
    "holdouts",
    "incident",
    "incidents",
    "ood",
    "protected",
    "terminal",
    "terminals",
})


@dataclass(frozen=True)
class OpaqueCollisionAuthorization:
    """A hash-pinned, content-free collision authorization."""

    payload: dict
    manifest_sha256: str
    manifest_path: Path | None = None


def _absolute_lexical_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        return os.path.commonpath(
            (os.fspath(directory), os.fspath(path))
        ) == os.fspath(directory)
    except ValueError:
        return False


def _path_has_forbidden_components(path: Path) -> bool:
    for component in path.parts:
        tokens = [
            token for token in re.split(
                r"[^a-z0-9]+", component.casefold()
            ) if token
        ]
        if set(tokens) & FORBIDDEN_PATH_TOKENS:
            return True
        for index, token in enumerate(tokens[:-1]):
            if token == "manual" and tokens[index + 1] in {
                    "review", "reviews"}:
                return True
    return False


def _path_is_repo_eval_artifact(path: Path) -> bool:
    if not _path_is_within(path, ROOT):
        return False
    try:
        relative = path.relative_to(ROOT)
    except ValueError:
        return False
    return any(
        component.casefold().startswith("eval")
        for component in relative.parts
    )


def _reject_path_before_target_access(
        path: Path, role: str, *, reject_repository: bool,
        reject_protected_components: bool) -> None:
    if path in QUARANTINED_LEGACY_EVAL:
        raise RuntimeError(f"{role} resolves to a quarantined evaluation path")
    if reject_repository and _path_is_within(path, ROOT):
        raise RuntimeError(f"{role} must remain outside the repository")
    if (reject_protected_components and
            (_path_has_forbidden_components(path) or
             _path_is_repo_eval_artifact(path))):
        raise RuntimeError(f"{role} uses a forbidden protected-data path")


def _resolve_without_target_access(
        path: Path, role: str, *, reject_repository: bool = False,
        reject_protected_components: bool = False,
) -> tuple[Path, bool, bool, os.stat_result | None]:
    """Resolve symlinks while checking each target before target metadata."""
    lexical_path = _absolute_lexical_path(path)
    _reject_path_before_target_access(
        lexical_path,
        role,
        reject_repository=reject_repository,
        reject_protected_components=reject_protected_components,
    )
    pending = list(lexical_path.parts[1:])
    resolved = Path(lexical_path.anchor)
    followed_symlink = False
    expansions = 0
    seen_targets = set()
    metadata = None

    while pending:
        component = pending.pop(0)
        candidate = resolved / component
        try:
            metadata = os.lstat(candidate)
        except FileNotFoundError:
            missing_path = _absolute_lexical_path(
                candidate.joinpath(*pending)
            )
            _reject_path_before_target_access(
                missing_path,
                role,
                reject_repository=reject_repository,
                reject_protected_components=reject_protected_components,
            )
            return missing_path, followed_symlink, False, None
        except OSError as exc:
            raise ValueError(f"{role} path cannot be inspected safely") from exc

        if not stat.S_ISLNK(metadata.st_mode):
            resolved = candidate
            continue

        followed_symlink = True
        expansions += 1
        if expansions > 40:
            raise RuntimeError(f"{role} contains too many symlink aliases")
        try:
            target_text = os.readlink(candidate)
        except OSError as exc:
            raise ValueError(f"{role} symlink cannot be read safely") from exc
        target = Path(target_text)
        if not target.is_absolute():
            target = resolved / target
        expanded = _absolute_lexical_path(target.joinpath(*pending))
        _reject_path_before_target_access(
            expanded,
            role,
            reject_repository=reject_repository,
            reject_protected_components=reject_protected_components,
        )
        state = os.fspath(expanded)
        if state in seen_targets:
            raise RuntimeError(f"{role} contains a symlink cycle")
        seen_targets.add(state)
        pending = list(expanded.parts[1:])
        resolved = Path(expanded.anchor)

    return resolved, followed_symlink, True, metadata


def _resolved_path_identity(
        path: Path, role: str, *, must_exist: bool,
) -> tuple[Path, tuple[int, int] | None]:
    """Resolve aliases and return a stable identity without reading content."""
    resolved, _, exists, metadata = _resolve_without_target_access(path, role)
    if must_exist and not exists:
        raise ValueError(f"{role} path does not exist")
    inode = None if metadata is None else (metadata.st_dev, metadata.st_ino)
    return resolved, inode


def _path_identities_alias(
        first: tuple[Path, tuple[int, int] | None],
        second: tuple[Path, tuple[int, int] | None]) -> bool:
    first_path, first_inode = first
    second_path, second_inode = second
    return (
        first_path == second_path or
        first_inode is not None and first_inode == second_inode
    )


def require_training_input_path_firewall(
        input_paths: list[Path], curation_paths_: list[Path]) -> None:
    """Reject protected/evaluation paths before any dataset loader runs."""
    roles = [
        *((f"training input {index}", path)
          for index, path in enumerate(input_paths)),
        *((f"curation input {index}", path)
          for index, path in enumerate(curation_paths_)),
    ]
    for role, path in roles:
        _, followed_symlink, exists, metadata = _resolve_without_target_access(
            path,
            role,
            reject_protected_components=True,
        )
        if followed_symlink:
            raise RuntimeError(f"{role} symlink aliases are forbidden")
        if not exists or metadata is None:
            raise ValueError(f"{role} path does not exist")
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{role} must be a regular file")
        if metadata.st_nlink != 1:
            raise RuntimeError(f"{role} hard-link aliases are forbidden")


def preflight_path_role_disjointness(
        input_paths: list[Path], curation_paths_: list[Path],
        output_path: Path, report_path: Path,
        authorization_path: Path | None = None) -> None:
    """Ensure neither write target aliases any read target or each other."""
    output_identity = _resolved_path_identity(
        output_path, "output", must_exist=False
    )
    report_identity = _resolved_path_identity(
        report_path, "report", must_exist=False
    )
    if _path_identities_alias(output_identity, report_identity):
        raise ValueError("output and report paths must be disjoint")

    read_identities = []
    for index, path in enumerate(input_paths):
        read_identities.append((
            f"training input {index}",
            _resolved_path_identity(
                path, f"training input {index}", must_exist=True
            ),
        ))
    for index, path in enumerate(curation_paths_):
        read_identities.append((
            f"curation input {index}",
            _resolved_path_identity(
                path, f"curation input {index}", must_exist=True
            ),
        ))
    if authorization_path is not None:
        read_identities.append((
            "collision authorization",
            _resolved_path_identity(
                authorization_path, "collision authorization",
                must_exist=True,
            ),
        ))

    for write_role, write_identity in (
            ("output", output_identity), ("report", report_identity)):
        for read_role, read_identity in read_identities:
            if _path_identities_alias(write_identity, read_identity):
                raise ValueError(
                    f"{write_role} path aliases {read_role} path"
                )


def _read_regular_file_without_symlinks(path: Path) -> bytes:
    """Open a resolved regular file atomically without following its leaf."""
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(
            "collision authorization cannot be opened safely"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("collision authorization must be a regular file")
        if metadata.st_nlink != 1:
            raise RuntimeError(
                "collision authorization hard-link aliases are forbidden"
            )
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            return source.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(item: object) -> bytes:
    """Return the one accepted content-addressed JSON representation."""
    return (
        json.dumps(item, ensure_ascii=True, separators=(",", ":"),
                   sort_keys=True) + "\n"
    ).encode("ascii")


def _require_exact_keys(item: object, keys: set[str], location: str) -> dict:
    if not isinstance(item, dict):
        raise ValueError(f"{location}: expected an object")
    actual = set(item)
    if actual != keys:
        raise ValueError(
            f"{location}: expected fields {sorted(keys)!r}, "
            f"found {sorted(actual)!r}"
        )
    return item


def _require_count(value: object, location: str, *, positive: bool) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{location}: expected a {qualifier} integer")
    return value


def _require_sha256(value: object, location: str) -> str:
    if (not isinstance(value, str) or len(value) != SHA256_HEX_LENGTH or
            any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{location}: expected a lowercase SHA-256 digest")
    return value


def opaque_file_bindings(paths: list[Path]) -> list[dict]:
    """Bind ordered, nonsemantic input bytes without recording path names."""
    return [
        {"index": index, "sha256": file_sha256(path)}
        for index, path in enumerate(paths)
    ]


def opaque_candidate_binding(rows: list[dict]) -> dict:
    """Summarize candidate Q/A identities without exposing their contents."""
    identities = []
    for item in rows:
        pair = qa_pair_from_record(item)
        if pair is None:
            raise ValueError("candidate row has unsupported QA serialization")
        question, answer = pair
        identity = hashlib.sha256(canonical_json_bytes({
            "answer": answer,
            "question": question,
        })).hexdigest()
        identities.append(identity)
    identities.sort()
    return {
        "count": len(identities),
        "identity_sha256": hashlib.sha256(
            canonical_json_bytes(identities)
        ).hexdigest(),
    }


def _validate_file_bindings(value: object, location: str) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError(f"{location}: expected an array")
    result = []
    for index, raw_binding in enumerate(value):
        binding = _require_exact_keys(
            raw_binding, {"index", "sha256"}, f"{location}[{index}]"
        )
        if binding["index"] != index:
            raise ValueError(
                f"{location}[{index}].index: expected ordered index {index}"
            )
        result.append({
            "index": index,
            "sha256": _require_sha256(
                binding["sha256"], f"{location}[{index}].sha256"
            ),
        })
    return result


def _validate_identity_binding(
        value: object, location: str, *, positive: bool) -> dict:
    binding = _require_exact_keys(
        value, {"count", "identity_sha256"}, location
    )
    return {
        "count": _require_count(
            binding["count"], f"{location}.count", positive=positive
        ),
        "identity_sha256": _require_sha256(
            binding["identity_sha256"], f"{location}.identity_sha256"
        ),
    }


def validate_opaque_collision_payload(payload: object) -> dict:
    """Validate that a manifest contains only opaque hashes and counts."""
    payload = _require_exact_keys(payload, {
        "candidate",
        "collision",
        "curation_inputs",
        "evaluation",
        "identity_scheme",
        "schema",
        "training_inputs",
    }, "collision authorization")
    if payload["schema"] != OPAQUE_COLLISION_SCHEMA:
        raise ValueError("collision authorization: unsupported schema")
    if payload["identity_scheme"] != OPAQUE_IDENTITY_SCHEME:
        raise ValueError("collision authorization: unsupported identity scheme")
    collision = _require_exact_keys(
        payload["collision"], {"count"}, "collision authorization.collision"
    )
    collision_count = _require_count(
        collision["count"], "collision authorization.collision.count",
        positive=False,
    )
    if collision_count:
        raise ValueError(
            "collision authorization reports one or more collisions"
        )
    return {
        "schema": OPAQUE_COLLISION_SCHEMA,
        "identity_scheme": OPAQUE_IDENTITY_SCHEME,
        "training_inputs": _validate_file_bindings(
            payload["training_inputs"],
            "collision authorization.training_inputs",
        ),
        "curation_inputs": _validate_file_bindings(
            payload["curation_inputs"],
            "collision authorization.curation_inputs",
        ),
        "candidate": _validate_identity_binding(
            payload["candidate"], "collision authorization.candidate",
            positive=True,
        ),
        "evaluation": _validate_identity_binding(
            payload["evaluation"], "collision authorization.evaluation",
            positive=True,
        ),
        "collision": {"count": 0},
    }


def load_opaque_collision_authorization(
        path: Path, expected_sha256: str) -> OpaqueCollisionAuthorization:
    """Load exactly one externally hash-pinned opaque authorization file."""
    expected_sha256 = _require_sha256(
        expected_sha256, "--collision-authorization-sha256"
    )
    lexical_path = Path(os.path.abspath(os.fspath(path)))
    if lexical_path in QUARANTINED_LEGACY_EVAL:
        raise RuntimeError(
            "a quarantined evaluation source cannot be an authorization"
        )
    try:
        lexical_metadata = path.lstat()
    except OSError as exc:
        raise ValueError(
            "collision authorization path cannot be inspected safely"
        ) from exc
    if stat.S_ISLNK(lexical_metadata.st_mode):
        raise RuntimeError("collision authorization symlinks are forbidden")
    resolved_path, _ = _resolved_path_identity(
        path, "collision authorization", must_exist=True
    )
    if resolved_path in QUARANTINED_LEGACY_EVAL:
        raise RuntimeError(
            "a quarantined evaluation alias cannot be an authorization"
        )
    if resolved_path != lexical_path:
        raise RuntimeError("collision authorization path aliases another path")
    raw = _read_regular_file_without_symlinks(resolved_path)
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "collision authorization SHA-256 does not match the pinned digest"
        )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("collision authorization is not valid JSON") from exc
    validated = validate_opaque_collision_payload(payload)
    if raw != canonical_json_bytes(validated):
        raise ValueError(
            "collision authorization is not canonical content-addressed JSON"
        )
    return OpaqueCollisionAuthorization(
        validated, actual_sha256, resolved_path
    )


def validate_opaque_collision_authorization(
        authorization: OpaqueCollisionAuthorization,
        input_paths: list[Path], curation_paths_: list[Path],
        rows: list[dict]) -> dict:
    """Fail closed unless the authorization binds this exact candidate."""
    if not isinstance(authorization, OpaqueCollisionAuthorization):
        raise ValueError("opaque collision authorization object is required")
    _require_sha256(
        authorization.manifest_sha256,
        "collision authorization manifest SHA-256",
    )
    payload = validate_opaque_collision_payload(authorization.payload)
    canonical_payload_sha256 = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    if canonical_payload_sha256 != authorization.manifest_sha256:
        raise ValueError(
            "collision authorization payload does not match its manifest "
            "SHA-256"
        )
    expected_training = opaque_file_bindings(input_paths)
    if payload["training_inputs"] != expected_training:
        raise ValueError("collision authorization has stale training input hashes")
    expected_curation = opaque_file_bindings(curation_paths_)
    if payload["curation_inputs"] != expected_curation:
        raise ValueError("collision authorization has stale curation input hashes")
    expected_candidate = opaque_candidate_binding(rows)
    if payload["candidate"] != expected_candidate:
        raise ValueError("collision authorization has a stale candidate identity")
    return {
        "authorization_sha256": authorization.manifest_sha256,
        "candidate": expected_candidate,
        "collision_count": 0,
        "evaluation": payload["evaluation"],
        "identity_scheme": OPAQUE_IDENTITY_SCHEME,
        "mode": "opaque_collision_authorization",
        "schema": OPAQUE_COLLISION_SCHEMA,
    }


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def require_text(item: dict, field: str, location: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: missing non-empty {field}")
    return value.strip()


def curation_paths(value: Path | list[Path] | None) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, Path):
        return [value]
    return list(value)


def load_curation(
        paths: Path | list[Path] | None,
) -> tuple[dict[str, dict], dict[str, Path]]:
    decisions = {}
    sources = {}
    for path in curation_paths(paths):
        with path.open() as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                location = "{}:{}".format(path, line_number)
                item = json.loads(line)
                action = require_text(item, "action", location)
                fact_id = require_text(item, "fact_id", location)
                require_text(item, "expected_question", location)
                require_text(item, "expected_answer", location)
                require_text(item, "reason", location)
                require_text(item, "reason_code", location)
                require_text(item, "reviewer", location)
                require_text(item, "reviewed_at", location)
                if action not in CURATION_ACTIONS:
                    raise ValueError(
                        f"{location}: unsupported action {action!r}")
                if fact_id in decisions:
                    raise ValueError(
                        f"{location}: duplicate curation fact_id {fact_id}")
                if action == "edit":
                    question = require_text(item, "question", location)
                    answer = require_text(item, "answer", location)
                    evidence = require_text(item, "evidence", location)
                    require_text(item, "evidence_url", location)
                    support_type = item.get("support_type", "extractive")
                    if support_type not in EDIT_SUPPORT_TYPES:
                        raise ValueError(
                            f"{location}: unsupported edit support_type "
                            f"{support_type!r}"
                        )
                    if not question.endswith("?"):
                        raise ValueError(
                            f"{location}: edited question must end with '?'")
                    if "\n" in question or "\n" in answer:
                        raise ValueError(
                            f"{location}: edited QA must be one line")
                    if has_protocol_tokens(
                            question) or has_protocol_tokens(answer):
                        raise ValueError(
                            f"{location}: protocol token in edited QA")
                    if (support_type == "extractive" and
                            normalize_text(answer) not in
                            normalize_text(evidence)):
                        raise ValueError(
                            f"{location}: edited answer must be extractive "
                            "from evidence"
                        )
                    if support_type == "manual_paraphrase":
                        require_text(item, "paraphrase_rationale", location)
                    rendered = f"Question: {question}\nAnswer: {answer}"
                    if parse_qa(rendered) != (question, answer):
                        raise ValueError(
                            f"{location}: edited QA does not round-trip")
                decisions[fact_id] = item
                sources[fact_id] = path
    return decisions, sources


def merge(
        input_paths: list[Path], output_path: Path, report_path: Path,
        facts, curation_path: Path | list[Path] | None = None,
        collision_authorization: OpaqueCollisionAuthorization | None = None,
) -> dict:
    if (facts is None) == (collision_authorization is None):
        raise ValueError(
            "exactly one synthetic fact set or opaque collision "
            "authorization is required"
        )
    paths = curation_paths(curation_path)
    if collision_authorization is None:
        require_synthetic_fixture_paths(
            input_paths, output_path, report_path, paths
        )
        authorization_path = None
    else:
        if not isinstance(collision_authorization,
                          OpaqueCollisionAuthorization):
            raise ValueError("opaque collision authorization object is required")
        if not isinstance(collision_authorization.manifest_path, Path):
            raise ValueError("collision authorization manifest path is required")
        authorization_path = collision_authorization.manifest_path
    require_training_input_path_firewall(input_paths, paths)
    preflight_path_role_disjointness(
        input_paths, paths, output_path, report_path, authorization_path
    )
    decisions, decision_sources = load_curation(paths)
    applied_decisions = set()
    rows = []
    questions = {}
    pairs = {}
    fact_ids = {}
    counts_by_input = collections.Counter()
    counts_by_kind = collections.Counter()
    counts_by_source = collections.Counter()
    exclusions = []
    exclusion_reasons = collections.Counter()
    for path in input_paths:
        with path.open() as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                location = "{}:{}".format(path, line_number)
                item = json.loads(line)
                try:
                    pair = qa_pair_from_record(item)
                except ValueError as exc:
                    raise ValueError(f"{location}: {exc}") from exc
                if pair is None:
                    raise ValueError(
                        f"{location}: unsupported QA serialization")
                question, answer = pair
                original_fact_id = item.get("fact_id")
                decision = decisions.get(original_fact_id)
                if decision is not None:
                    if (question != decision["expected_question"] or
                            answer != decision["expected_answer"]):
                        raise ValueError(
                            f"{location}: stale curation decision for "
                            f"{original_fact_id}")
                    source_evidence_urls = {
                        item.get("url"),
                        item.get("evidence_url"),
                        item.get("url_evidence_url"),
                    }
                    if decision.get(
                            "evidence_url", item.get("url")
                    ) not in source_evidence_urls:
                        raise ValueError(
                            f"{location}: curation evidence URL disagrees "
                            "with source"
                        )
                    applied_decisions.add(original_fact_id)
                    action = decision["action"]
                    reason_code = decision["reason_code"]
                    if action == "drop":
                        reason = "manual_curation:{}".format(reason_code)
                        exclusion_reasons[reason] += 1
                        exclusions.append({
                            "fact_id": original_fact_id,
                            "input": portable_path(path),
                            "line": line_number,
                            "reason": reason,
                        })
                        continue
                    question = decision["question"].strip()
                    answer = decision["answer"].strip()
                    item = dict(item)
                    curation_metadata = {
                        "action": "edit",
                        "decision_file": portable_path(
                            decision_sources[original_fact_id]),
                        "original_fact_id": original_fact_id,
                        "reason": decision["reason"],
                        "reason_code": reason_code,
                        "reviewed_at": decision["reviewed_at"],
                        "reviewer": decision["reviewer"],
                    }
                    if "support_type" in decision:
                        curation_metadata["support_type"] = decision[
                            "support_type"]
                    if "paraphrase_rationale" in decision:
                        curation_metadata["paraphrase_rationale"] = decision[
                            "paraphrase_rationale"]
                    item.update({
                        "answer": answer,
                        "curation": curation_metadata,
                        "evidence": decision["evidence"].strip(),
                        "evidence_url": decision["evidence_url"].strip(),
                        "fact_id": stable_fact_id(question, answer),
                        "quality_schema": "curated-qa-v1",
                        "question": question,
                        "text": f"Question: {question}\nAnswer: {answer}",
                    })
                # The source tranches contain several legacy prompt renderings,
                # including generator instructions.  Structured Q/A is the
                # validated contract, so render every accepted row canonically.
                item = dict(item)
                item.update({
                    "answer": answer,
                    "question": question,
                    "text": f"Question: {question}\nAnswer: {answer}",
                })
                if has_protocol_tokens(
                        question) or has_protocol_tokens(answer):
                    raise ValueError(f"{location}: protocol token in QA")
                if LOW_VALUE.search(question):
                    reason = "low_value_or_time_sensitive"
                    exclusion_reasons[reason] += 1
                    exclusions.append({
                        "fact_id": item.get("fact_id"),
                        "input": portable_path(path),
                        "line": line_number,
                        "reason": reason,
                    })
                    continue
                if facts is not None:
                    leak = leakage_reason(question, answer, facts)
                    if leak:
                        exclusion_reasons[leak] += 1
                        exclusions.append({
                            "fact_id": item.get("fact_id"),
                            "input": portable_path(path),
                            "line": line_number,
                            "reason": leak,
                        })
                        continue
                question_key = normalize_text(question)
                pair_key = (question_key, normalize_text(answer))
                if question_key in questions:
                    raise ValueError(
                        f"{location}: duplicate question also at "
                        f"{questions[question_key]}"
                    )
                if pair_key in pairs:
                    raise ValueError(
                        f"{location}: duplicate QA pair also at "
                        f"{pairs[pair_key]}"
                    )
                fact_id = item.get("fact_id")
                if not isinstance(fact_id, str) or not fact_id:
                    raise ValueError(f"{location}: missing fact_id")
                if fact_id in fact_ids:
                    raise ValueError(
                        f"{location}: duplicate fact_id also at "
                        f"{fact_ids[fact_id]}"
                    )
                questions[question_key] = location
                pairs[pair_key] = location
                fact_ids[fact_id] = location
                rows.append(item)
                counts_by_input[portable_path(path)] += 1
                counts_by_kind[item.get("kind", "unknown")] += 1
                counts_by_source[item.get("source", "unknown")] += 1

    unused_decisions = sorted(set(decisions) - applied_decisions)
    if unused_decisions:
        raise ValueError(
            "curation decisions did not match an input fact: " +
            ", ".join(unused_decisions))

    rows.sort(key=lambda item: (
        normalize_text(qa_pair_from_record(item)[0]), item["fact_id"]))
    if collision_authorization is None:
        evaluation_boundary = {
            "evaluation_fact_count": len(facts),
            "mode": "synthetic_fact_fixture",
        }
    else:
        evaluation_boundary = validate_opaque_collision_authorization(
            collision_authorization, input_paths, paths, rows
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output:
        for item in rows:
            output.write(json.dumps(item, ensure_ascii=False,
                                    sort_keys=True) + "\n")
    report = {
        "schema": "curated-training-qa-report-v1",
        "inputs": [portable_path(path) for path in input_paths],
        "input_sha256": {
            portable_path(path): file_sha256(path) for path in input_paths
        },
        "eval_fact_count": (
            len(facts) if facts is not None
            else evaluation_boundary["evaluation"]["count"]
        ),
        "evaluation_boundary": evaluation_boundary,
        "curation": {
            "artifacts": [
                {
                    "path": portable_path(path),
                    "sha256": file_sha256(path),
                }
                for path in paths
            ],
            "by_action": dict(sorted(collections.Counter(
                decision["action"]
                for decision in decisions.values()).items())),
            "by_reason": dict(sorted(collections.Counter(
                decision["reason_code"]
                for decision in decisions.values()).items())),
            "edit_support_types": dict(sorted(collections.Counter(
                decision.get("support_type", "extractive")
                for decision in decisions.values()
                if decision["action"] == "edit").items())),
            "decisions": len(decisions),
        },
        "exclusions": exclusions,
        "output": portable_path(output_path),
        "output_sha256": file_sha256(output_path),
        "counts": {
            "by_input": dict(sorted(counts_by_input.items())),
            "by_kind": dict(sorted(counts_by_kind.items())),
            "by_source": dict(sorted(counts_by_source.items())),
            "excluded": len(exclusions),
            "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
            "output": len(rows),
            "unique_fact_ids": len(fact_ids),
            "unique_questions": len(questions),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def require_synthetic_evaluation_fixture_paths(paths: list[Path]) -> None:
    """Allow legacy fact loading only for explicit external fixtures."""
    for index, path in enumerate(paths):
        role = f"synthetic evaluation fixture {index}"
        lexical_path = _absolute_lexical_path(path)
        filename_tokens = {
            token for token in re.split(
                r"[^a-z0-9]+", lexical_path.name.casefold()
            ) if token
        }
        if "synthetic" not in filename_tokens:
            raise RuntimeError(
                f"{role} filename must explicitly identify a synthetic fixture"
            )
        _, followed_symlink, exists, metadata = _resolve_without_target_access(
            path, role, reject_repository=True
        )
        if followed_symlink:
            raise RuntimeError(f"{role} symlink aliases are forbidden")
        if not exists or metadata is None:
            raise ValueError(f"{role} path does not exist")
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{role} must be a regular file")
        if metadata.st_nlink != 1:
            raise RuntimeError(f"{role} hard-link aliases are forbidden")


def evaluation_facts(
        paths: list[Path] | None, *, synthetic_empty: bool) -> list:
    """Legacy synthetic-test helper; production CLI never loads this path."""
    if synthetic_empty:
        if paths is not None:
            raise RuntimeError(
                "synthetic empty facts cannot be combined with fixture paths"
            )
        return []
    if not paths:
        raise RuntimeError("explicit synthetic evaluation fixture paths required")
    lexical_paths = {
        Path(os.path.abspath(os.fspath(path))) for path in paths
    }
    if lexical_paths & QUARANTINED_LEGACY_EVAL:
        raise RuntimeError(
            "legacy evaluation collision sources are quarantined; refusing access"
        )
    require_synthetic_evaluation_fixture_paths(paths)
    return eval_facts(paths)


def require_synthetic_fixture_paths(
        input_paths: list[Path], output_path: Path, report_path: Path,
        curation_paths_: list[Path]) -> None:
    """Keep the no-evaluation mode from writing repository artifacts."""
    paths = [
        *((f"training input {index}", path, True)
          for index, path in enumerate(input_paths)),
        *((f"curation input {index}", path, True)
          for index, path in enumerate(curation_paths_)),
        ("output", output_path, False),
        ("report", report_path, False),
    ]
    for role, path, must_exist in paths:
        _, followed_symlink, exists, metadata = _resolve_without_target_access(
            path, role, reject_repository=True
        )
        if followed_symlink:
            raise RuntimeError(f"synthetic {role} symlink aliases are forbidden")
        if must_exist and (not exists or metadata is None):
            raise ValueError(f"synthetic {role} path does not exist")
        if must_exist and not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"synthetic {role} must be a regular file")
        if must_exist and metadata.st_nlink != 1:
            raise RuntimeError(
                f"synthetic {role} hard-link aliases are forbidden"
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path,
                        default=DEFAULT_INPUTS)
    boundary = parser.add_mutually_exclusive_group(required=True)
    boundary.add_argument("--collision-authorization", type=Path)
    boundary.add_argument("--synthetic-empty-eval", action="store_true")
    parser.add_argument("--collision-authorization-sha256")
    parser.add_argument("--curation", nargs="*", type=Path,
                        default=DEFAULT_CURATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    authorization = None
    if args.collision_authorization is not None:
        if args.collision_authorization_sha256 is None:
            parser.error(
                "--collision-authorization requires "
                "--collision-authorization-sha256"
            )
        authorization = load_opaque_collision_authorization(
            args.collision_authorization,
            args.collision_authorization_sha256,
        )
        facts = None
    else:
        if args.collision_authorization_sha256 is not None:
            parser.error(
                "--collision-authorization-sha256 requires "
                "--collision-authorization"
            )
        require_synthetic_fixture_paths(
            args.inputs, args.output, args.report,
            curation_paths(args.curation),
        )
        facts = []
    report = merge(
        args.inputs, args.output, args.report, facts,
        args.curation,
        collision_authorization=authorization,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
