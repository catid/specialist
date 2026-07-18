#!/usr/bin/env python3
"""Run a conservative two-pass independent semantic audit for one train shard.

Mistral emits only guided evidence maps.  Final gate states are derived by this
program from two independent prompt passes, deterministic requested-facet
checks, and the pinned DeBERTa NLI prefilter.  No model-authored overall verdict
is accepted, and every output remains training-ineligible pending global
deduplication, required manual review, and exact token-budget selection.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Sequence

import build_high_information_domain_corpus_v1 as corpus
import high_information_semantic_facets_v1 as facets
import run_high_information_nli_prefilter_v1 as nli
import verify_high_information_candidates_v1 as structural


MODEL_ID = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
MODEL_REVISION = "95a6d26c4bfb886c58daf9d3f7332c857cb27b43"
MODEL_DIRECTORY = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--mistralai--Mistral-Small-3.2-24B-Instruct-2506"
    / "snapshots"
    / MODEL_REVISION
).resolve()
MODEL_BLOB_RECEIPTS = {
    "config.json": "786d10ba172eb033ce576f199e13cd65c3a8d905",
    "model.safetensors.index.json": "c8d12958a1c6d6de8f6539883802e6e1a7bcac6a",
    "params.json": "ba80d5d1ebf33f2b8e7ad064f7a0c05057af7ab0",
    "tekken.json": "6e2501687ccd0e1f30f36319eaf2b46958b897811e246cd8eb5d385b9e3de7d1",
    "consolidated.safetensors": "2cc4cc283a4bf3820f77ddb73bef3128b7159057c240b94a72b663d97af86f89",
    "model-00001-of-00010.safetensors": "91831c2ce219df0ce63bc33c6249e5cb01db8d93816bcebf975f1c406286520e",
    "model-00002-of-00010.safetensors": "8ffe80706a66b2f5ef1fb058806ccf09f124ec4ad38af7a377e44ab1ee2fd664",
    "model-00003-of-00010.safetensors": "99ec66e891f9563f568734eadfc5b7701e04620e8e163d4d5755277a3b50cf2f",
    "model-00004-of-00010.safetensors": "e1df1527b12b1eb5cbd9a50914f9e6eb24e885ec830a3c16b5eed6ad0b53a396",
    "model-00005-of-00010.safetensors": "3556ac03f47c24eb8ad27c237e25baad639c651d9596fd72cb1523137bf56163",
    "model-00006-of-00010.safetensors": "2c41e6f80f2b5ca384ce703eac048a13daf2aff689c3acca66a8943f45338aae",
    "model-00007-of-00010.safetensors": "62a725f154f6ba942a36b5cc450db2b2df32f434e3224558c789bc04fa05fd36",
    "model-00008-of-00010.safetensors": "3a1a6ac77e6434418bb7273b68a7b3534fed5217c990061c92a8f990dd6ab20e",
    "model-00009-of-00010.safetensors": "e1fffc9bb2b77d4d2382c1bd9053e9d017741d67ca00cc6f77034a294f2f5cfd",
    "model-00010-of-00010.safetensors": "116ef7ae6fa0fd46b478324e4aa6a49f448afed900ca9f71d4fbd3d02289bbd4",
}
RUNTIME_MODEL_FILE_SHA256 = {
    "config.json": "01ab910a5dda7995709cc355d094eabb8094b78d49240cd167188606c3ff5edb",
    "params.json": "23febf6d98a78f896149ab08733356d5b99e71b2e7e4c3caddfa4f48cda87885",
    "tekken.json": "6e2501687ccd0e1f30f36319eaf2b46958b897811e246cd8eb5d385b9e3de7d1",
    "consolidated.safetensors": "2cc4cc283a4bf3820f77ddb73bef3128b7159057c240b94a72b663d97af86f89",
}
VLLM_VERSION = "0.25.0"
RESULT_SCHEMA = "high-information-two-pass-semantic-judge-result-v1"
RECORD_SCHEMA = "high-information-two-pass-semantic-judge-request-v1"
PASS_NAMES = ("facet_evidence_audit", "adversarial_omission_audit")
MODEL_GATES = (
    "source_entailment",
    "citation_support_coverage",
    "application_correctness",
    "hard_negative_calibration",
    "safety_transfer_preservation",
    "attribution_and_scope_preservation",
    "unsupported_claim_absence",
    "training_value_and_nontriviality",
)
EVIDENCE_REQUIRED_PASS_GATES = frozenset(
    {
        "source_entailment",
        "citation_support_coverage",
        "application_correctness",
        "hard_negative_calibration",
    }
)
ALL_GATES = (
    "exact_evidence_quote_match",
    "source_entailment",
    "citation_support_coverage",
    "question_answer_completeness",
    "application_correctness",
    "hard_negative_calibration",
    "safety_transfer_preservation",
    "attribution_and_scope_preservation",
    "unsupported_claim_absence",
    "training_value_and_nontriviality",
)
FAILURE_CODES = (
    "unsupported_claim",
    "missing_requested_facet",
    "facet_not_supported_by_citation",
    "citation_coverage_gap",
    "application_not_justified",
    "hard_negative_is_answerable",
    "hard_negative_invents_correction",
    "safety_scope_lost",
    "attribution_or_scope_lost",
    "low_value_trivia",
    "ambiguous_or_uncertain",
)
FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.I | re.S)
COMMON_FLAGS = {
    "not_a_technical_medical_or_suspension_certification",
    "historical_lineage_and_identity_claims_require_source_attribution",
    "external_reference_pages_must_not_be_imputed_or_copied",
    "contested_history_and_aliases_require_provenance",
}


GUIDED_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["results"],
    "properties": {
        "results": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "candidate_example_id",
                    "facet_mappings",
                    "gate_evidence",
                    "confidence",
                    "failure_codes",
                ],
                "properties": {
                    "candidate_example_id": {"type": "string"},
                    "facet_mappings": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "facet_id",
                                "status",
                                "answer_span",
                                "evidence_quote_indices",
                            ],
                            "properties": {
                                "facet_id": {"type": "string"},
                                "status": {
                                    "enum": [
                                        "supported",
                                        "missing",
                                        "unsupported",
                                        "uncertain",
                                    ]
                                },
                                "answer_span": {"type": ["string", "null"]},
                                "evidence_quote_indices": {
                                    "type": "array",
                                    "items": {"type": "integer", "minimum": 0},
                                },
                            },
                        },
                    },
                    "gate_evidence": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": list(MODEL_GATES),
                        "properties": {
                            gate: {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["verdict", "evidence_quote_indices"],
                                "properties": {
                                    "verdict": {
                                        "enum": [
                                            "pass",
                                            "fail",
                                            "uncertain",
                                            "not_applicable",
                                        ]
                                    },
                                    "evidence_quote_indices": {
                                        "type": "array",
                                        "items": {"type": "integer", "minimum": 0},
                                    },
                                },
                            }
                            for gate in MODEL_GATES
                        },
                    },
                    "confidence": {"enum": ["high", "medium", "low"]},
                    "failure_codes": {
                        "type": "array",
                        "items": {"enum": list(FAILURE_CODES)},
                    },
                },
            },
        }
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write(path: Path, payload: bytes) -> None:
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


def _self_address(value: dict) -> str:
    unsigned = dict(value)
    unsigned.pop("content_sha256_before_self_field", None)
    return corpus.canonical_sha256(unsigned)


def _require_self_address(value: dict, label: str) -> None:
    if value.get("content_sha256_before_self_field") != _self_address(value):
        raise RuntimeError(f"{label} content address changed")


def validate_model_snapshot(model_directory: Path) -> dict[str, dict[str, Any]]:
    model_directory = model_directory.expanduser().resolve()
    if model_directory != MODEL_DIRECTORY:
        raise RuntimeError("semantic judge requires the pinned local Mistral revision")
    observed: dict[str, dict[str, Any]] = {}
    for name, expected_blob in MODEL_BLOB_RECEIPTS.items():
        path = model_directory / name
        if not path.is_symlink() or not path.is_file():
            raise RuntimeError(f"pinned judge snapshot file is missing: {name}")
        target = Path(os.readlink(path)).name
        if target != expected_blob:
            raise RuntimeError(f"pinned judge snapshot file changed: {name}")
        receipt: dict[str, Any] = {
            "snapshot_blob_id": target,
            "file_bytes": path.stat().st_size,
            "runtime_loaded": name in RUNTIME_MODEL_FILE_SHA256,
        }
        if name in RUNTIME_MODEL_FILE_SHA256:
            digest = corpus.file_sha256(path)
            if digest != RUNTIME_MODEL_FILE_SHA256[name]:
                raise RuntimeError(
                    f"pinned judge runtime file content changed: {name}"
                )
            receipt["file_sha256"] = digest
        observed[name] = receipt
    if set(RUNTIME_MODEL_FILE_SHA256) - set(observed):
        raise RuntimeError("judge runtime file receipt coverage changed")
    return observed


def output_paths(shard_index: int, *, smoke: bool) -> dict[str, Path]:
    suffix = ".smoke" if smoke else ""
    stem = f"semantic_judge_gpu{shard_index}{suffix}"
    return {
        "partial": corpus.OUTPUT_DIR / f"{stem}.partial.jsonl",
        "output": corpus.OUTPUT_DIR / f"{stem}.jsonl",
        "report": corpus.OUTPUT_DIR / f"{stem}.report.json",
        "telemetry": corpus.OUTPUT_DIR / f"{stem}.telemetry.json",
    }


def load_nli_results(shard_index: int, packets: Sequence[dict]) -> tuple[dict[str, dict], dict]:
    paths = nli.shard_paths(shard_index)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    _require_self_address(report, "NLI report")
    if (
        report.get("schema") != "high-information-nli-prefilter-report-v1"
        or report.get("status") != "complete_semantic_pending"
        or report.get("gpu_shard") != shard_index
        or report.get("packets") != len(packets)
        or report.get("output") != corpus.relative(paths["output"])
        or report.get("output_sha256") != corpus.file_sha256(paths["output"])
        or report.get("semantic_verification_completed") is not False
        or report.get("training_rows_emitted") is not False
    ):
        raise RuntimeError("NLI report contract changed")
    nli_contract = report.get("run_contract")
    if not isinstance(nli_contract, dict):
        raise RuntimeError("NLI report lacks a sealed run contract")
    _require_self_address(nli_contract, "NLI run contract")
    nli_contract_sha256 = nli_contract["content_sha256_before_self_field"]
    if (
        nli_contract.get("worker_file_sha256")
        != corpus.file_sha256(Path(nli.__file__).resolve())
        or report.get("worker_file_sha256") != nli_contract["worker_file_sha256"]
    ):
        raise RuntimeError("NLI implementation receipt changed")
    rows = [
        json.loads(line)
        for line in paths["output"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != len(packets):
        raise RuntimeError("NLI result count changed")
    index = {}
    for row, packet in zip(rows, packets, strict=True):
        nli.validate_result(row, packet, nli_contract_sha256)
        index[row["candidate_example_id"]] = row
    if len(index) != len(rows):
        raise RuntimeError("NLI output duplicates candidate identity")
    return index, report


def groups_by_request(packets: Sequence[dict]) -> list[list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for packet in packets:
        grouped[packet["request_id"]].append(packet)
    groups = []
    for request_id in sorted(grouped):
        request_candidates = sorted(
            grouped[request_id], key=lambda value: value["candidate_example_id"]
        )
        if not 1 <= len(request_candidates) <= 4:
            raise RuntimeError("semantic judge request group size changed")
        # Keep the source/request ordering stable, but audit one candidate per
        # conversation.  Live probes showed that asking the 24B judge to track
        # up to four independent facet/gate inventories in one JSON response
        # caused pervasive cross-candidate omissions and conditional-schema
        # errors.  vLLM still batches these singleton conversations together.
        groups.extend([[packet] for packet in request_candidates])
    return groups


def prompt_payload(group: Sequence[dict]) -> str:
    context_hashes = {packet["context_text_sha256"] for packet in group}
    context_texts = {packet["context_text"] for packet in group}
    if len(context_hashes) != 1 or len(context_texts) != 1:
        raise RuntimeError("semantic judge group crossed source context")
    candidates = []
    for packet in group:
        requested = facets.deterministic_facet_signals(
            packet["question"], packet["answer"]
        )
        candidates.append(
            {
                "candidate_example_id": packet["candidate_example_id"],
                "task_subtype": packet["task_subtype"],
                "generation_mode": packet["generation_mode"],
                "question": packet["question"],
                "answer": packet["answer"],
                "requested_facets_fixed_by_code": requested,
                "evidence_quotes_numbered_from_zero": [
                    {"index": index, "quote": quote}
                    for index, quote in enumerate(packet["evidence_quotes"])
                ],
                "safety_transfer_flags": packet["safety_transfer_flags"],
            }
        )
    return json.dumps(
        {
            "train_only_source_context": group[0]["context_text"],
            "candidates": candidates,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def judge_messages(group: Sequence[dict], pass_name: str) -> list[dict[str, str]]:
    if pass_name not in PASS_NAMES:
        raise ValueError("unknown semantic judge pass")
    shared = (
        "Return one JSON object matching the supplied schema and no prose or code "
        "fence. You produce evidence mappings only; you do not set an overall "
        "accept/reject verdict. Preserve each candidate_example_id exactly. The "
        "requested facet set is fixed by deterministic code: do not add, remove, "
        "merge, rename, or reorder facets. For every supported facet, answer_span "
        "must be a nonempty exact substring of the candidate answer and at least "
        "one numbered evidence quote must support it. Mark a facet missing if the "
        "answer omits it even when the context contains it. Mark unsupported when "
        "the answer states it but the cited evidence does not support it. Use only "
        "quote indices supplied for that candidate. A pass for source_entailment, "
        "citation_support_coverage, application_correctness, or hard_negative_calibration "
        "needs supporting quote indices. Negative/meta gates (unsupported-claim absence, "
        "safety and attribution preservation, and training value) may have no quote index "
        "when their verdict is based on absence, preserved answer scope, or task quality; "
        "do not invent an index. A not_applicable gate needs none. Application is applicable "
        "only to application_scenario. Hard-negative calibration is applicable only "
        "to calibrated_hard_negative. Do not reveal reasoning. Use failure codes, "
        "not free-form explanations. Isolated dates, venues, or public-identity "
        "trivia fail training value when the same context supports more useful "
        "technique, mechanism, safety, lineage-significance, or application knowledge."
    )
    if pass_name == "facet_evidence_audit":
        role = (
            "Audit every requested question facet against the answer and exact "
            "quotes. Check entailment, complete citation coverage, application "
            "prerequisites, calibrated unanswerability, safety, attribution, and "
            "knowledge value. Be conservative: uncertainty is not a pass."
        )
    else:
        role = (
            "Act as an adversarial omission and unsupported-claim auditor. Try to "
            "find any requested facet the answer skipped, any answer claim not "
            "covered by an exact quote, any false-premise answer that is actually "
            "answerable, and any lost safety or attribution limitation. Be "
            "conservative: uncertainty is not a pass."
        )
    return [
        {"role": "system", "content": shared},
        {
            "role": "user",
            "content": f"AUDIT MODE: {pass_name}\n{role}\n\nINPUT JSON\n{prompt_payload(group)}",
        },
    ]


def parse_judge_output(text: str) -> dict:
    candidate = text.strip()
    match = FENCE_RE.fullmatch(candidate)
    if match:
        candidate = match.group(1).strip()
    raw = text.encode("utf-8")

    def parse_failure(code: str, detail: str) -> dict:
        # Structured decoding is a constraint, not an integrity guarantee.  A
        # malformed completion must reject/review every candidate in this pass
        # without aborting the shard or persisting the potentially very large
        # raw completion.  Its digest preserves exact incident identity.
        return {
            "results": None,
            "__parse_failure__": {
                "code": code,
                "detail": detail,
                "raw_text_sha256": corpus.sha256_bytes(raw),
                "raw_text_bytes": len(raw),
            },
        }

    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return parse_failure("invalid_json", "judge output is invalid JSON")
    if not isinstance(value, dict) or set(value) != {"results"}:
        return parse_failure(
            "top_level_schema_invalid", "judge top-level schema changed"
        )
    return value


def _expected_gate_verdict(packet: dict, gate: str) -> str:
    if gate == "application_correctness" and packet["task_subtype"] != "application_scenario":
        return "not_applicable"
    if gate == "hard_negative_calibration" and packet["generation_mode"] != "calibrated_hard_negative":
        return "not_applicable"
    return "pass"


def validate_pass_result(result: Any, packet: dict) -> dict:
    expected_fields = {
        "candidate_example_id",
        "facet_mappings",
        "gate_evidence",
        "confidence",
        "failure_codes",
    }
    if not isinstance(result, dict) or set(result) != expected_fields:
        raise RuntimeError("judge candidate schema changed")
    if result["candidate_example_id"] != packet["candidate_example_id"]:
        raise RuntimeError("judge candidate identity changed")
    if result["confidence"] not in {"high", "medium", "low"}:
        raise RuntimeError("judge confidence changed")
    if (
        not isinstance(result["failure_codes"], list)
        or len(set(result["failure_codes"])) != len(result["failure_codes"])
        or any(code not in FAILURE_CODES for code in result["failure_codes"])
    ):
        raise RuntimeError("judge failure-code schema changed")

    deterministic = facets.deterministic_facet_signals(
        packet["question"], packet["answer"]
    )
    expected_facets = {item["facet_id"]: item for item in deterministic}
    mappings = result["facet_mappings"]
    if not isinstance(mappings, list) or len(mappings) != len(expected_facets):
        raise RuntimeError("judge added or dropped a requested facet")
    mapping_index = {item.get("facet_id"): item for item in mappings if isinstance(item, dict)}
    if set(mapping_index) != set(expected_facets) or len(mapping_index) != len(mappings):
        raise RuntimeError("judge changed requested facet identity")
    quote_count = len(packet["evidence_quotes"])
    normalized_mappings = {}
    for facet_id in expected_facets:
        mapping = mapping_index[facet_id]
        if set(mapping) != {
            "facet_id", "status", "answer_span", "evidence_quote_indices"
        }:
            raise RuntimeError("judge facet mapping schema changed")
        status = mapping["status"]
        span = mapping["answer_span"]
        indices = mapping["evidence_quote_indices"]
        if (
            status not in {"supported", "missing", "unsupported", "uncertain"}
            or not isinstance(indices, list)
            or len(set(indices)) != len(indices)
            or any(not isinstance(index, int) or not 0 <= index < quote_count for index in indices)
        ):
            raise RuntimeError("judge facet mapping value changed")
        if status == "supported":
            if not isinstance(span, str) or not span or span not in packet["answer"] or not indices:
                raise RuntimeError("supported facet lacks exact answer span or evidence")
        elif status == "missing":
            if span is not None or indices:
                raise RuntimeError("missing facet improperly claims answer or evidence")
        else:
            if span is not None and (not isinstance(span, str) or span not in packet["answer"]):
                raise RuntimeError("facet answer span is not exact candidate text")
        normalized_mappings[facet_id] = mapping

    gates = result["gate_evidence"]
    if not isinstance(gates, dict) or set(gates) != set(MODEL_GATES):
        raise RuntimeError("judge gate set changed")
    normalized_gates = {}
    for gate in MODEL_GATES:
        value = gates[gate]
        if not isinstance(value, dict) or set(value) != {"verdict", "evidence_quote_indices"}:
            raise RuntimeError("judge gate schema changed")
        verdict = value["verdict"]
        indices = value["evidence_quote_indices"]
        if (
            verdict not in {"pass", "fail", "uncertain", "not_applicable"}
            or not isinstance(indices, list)
            or len(set(indices)) != len(indices)
            or any(not isinstance(index, int) or not 0 <= index < quote_count for index in indices)
            or (
                verdict == "pass"
                and gate in EVIDENCE_REQUIRED_PASS_GATES
                and not indices
            )
            or (verdict == "not_applicable" and indices)
            or (_expected_gate_verdict(packet, gate) == "not_applicable" and verdict != "not_applicable")
            or (_expected_gate_verdict(packet, gate) == "pass" and verdict == "not_applicable")
        ):
            raise RuntimeError("judge gate value is inconsistent")
        normalized_gates[gate] = value
    return {
        "facet_mappings": normalized_mappings,
        "gate_evidence": normalized_gates,
        "confidence": result["confidence"],
        "failure_codes": sorted(result["failure_codes"]),
    }


def validate_pass_output(value: dict, group: Sequence[dict]) -> dict[str, dict]:
    results = value.get("results")
    if not isinstance(results, list) or len(results) != len(group):
        raise RuntimeError("judge output candidate count changed")
    index = {item.get("candidate_example_id"): item for item in results if isinstance(item, dict)}
    expected = {packet["candidate_example_id"]: packet for packet in group}
    if set(index) != set(expected) or len(index) != len(results):
        raise RuntimeError("judge output omitted or duplicated candidate identity")
    return {
        candidate_id: validate_pass_result(index[candidate_id], packet)
        for candidate_id, packet in expected.items()
    }


def invalid_pass_result(packet: dict) -> dict:
    """Return a fail-closed normalized result for an inconsistent judge row."""

    deterministic = facets.deterministic_facet_signals(
        packet["question"], packet["answer"]
    )
    return {
        "facet_mappings": {
            item["facet_id"]: {
                "facet_id": item["facet_id"],
                "status": "uncertain",
                "answer_span": None,
                "evidence_quote_indices": [],
            }
            for item in deterministic
        },
        "gate_evidence": {
            gate: {
                "verdict": (
                    "not_applicable"
                    if _expected_gate_verdict(packet, gate) == "not_applicable"
                    else "uncertain"
                ),
                "evidence_quote_indices": [],
            }
            for gate in MODEL_GATES
        },
        "confidence": "low",
        "failure_codes": ["ambiguous_or_uncertain"],
    }


def normalize_pass_output(
    value: dict, group: Sequence[dict]
) -> tuple[dict[str, dict], dict[str, dict[str, str]], dict[str, Any]]:
    """Normalize valid rows and quarantine model-inconsistent rows.

    Guided decoding guarantees JSON shape, but cannot express every conditional
    relationship enforced by ``validate_pass_result``.  A model inconsistency
    must reject/review the affected candidate without aborting the whole shard.
    """

    expected = {packet["candidate_example_id"]: packet for packet in group}
    results = value.get("results") if isinstance(value, dict) else None
    if not isinstance(results, list) or len(results) != len(group):
        parse_failure = (
            value.get("__parse_failure__") if isinstance(value, dict) else None
        )
        if (
            isinstance(parse_failure, dict)
            and parse_failure.get("code")
            in {"invalid_json", "top_level_schema_invalid"}
            and isinstance(parse_failure.get("detail"), str)
        ):
            error = {
                "code": parse_failure["code"],
                "detail": parse_failure["detail"],
            }
        else:
            error = {
                "code": "candidate_inventory_invalid",
                "detail": "judge output candidate count changed",
            }
        return (
            {
                candidate_id: invalid_pass_result(packet)
                for candidate_id, packet in expected.items()
            },
            {candidate_id: dict(error) for candidate_id in expected},
            {"__entire_pass_output__": value},
        )
    index = {
        item.get("candidate_example_id"): item
        for item in results
        if isinstance(item, dict)
    }
    if set(index) != set(expected) or len(index) != len(results):
        error = {
            "code": "candidate_inventory_invalid",
            "detail": "judge output omitted or duplicated candidate identity",
        }
        return (
            {
                candidate_id: invalid_pass_result(packet)
                for candidate_id, packet in expected.items()
            },
            {candidate_id: dict(error) for candidate_id in expected},
            {"__entire_pass_output__": value},
        )

    normalized: dict[str, dict] = {}
    errors: dict[str, dict[str, str]] = {}
    invalid_raw: dict[str, Any] = {}
    for candidate_id, packet in expected.items():
        try:
            normalized[candidate_id] = validate_pass_result(
                index[candidate_id], packet
            )
        except RuntimeError as error:
            normalized[candidate_id] = invalid_pass_result(packet)
            errors[candidate_id] = {
                "code": "candidate_consistency_invalid",
                "detail": str(error),
            }
            invalid_raw[candidate_id] = index[candidate_id]
    return normalized, errors, invalid_raw


def is_high_risk_sample(packet: dict) -> bool:
    specific_flags = set(packet["safety_transfer_flags"]) - COMMON_FLAGS
    high_risk = bool(specific_flags) or packet["task_subtype"] == "application_scenario"
    if not high_risk:
        return False
    bucket = int(hashlib.sha256(packet["candidate_example_id"].encode()).hexdigest()[:8], 16)
    return bucket % 10 == 0


def aggregate_candidate(
    packet: dict,
    first: dict,
    second: dict,
    nli_result: dict,
    pass_validation_errors: dict[str, dict[str, str]] | None = None,
) -> dict:
    review_reasons = []
    for pass_name, error in sorted((pass_validation_errors or {}).items()):
        review_reasons.append(
            f"judge_pass_validation_error:{pass_name}:{error['code']}"
        )
    deterministic = facets.deterministic_facet_signals(packet["question"], packet["answer"])
    missing = facets.missing_high_confidence_facets(
        packet["question"], packet["answer"]
    )
    heuristic_missing = [
        item["facet_id"]
        for item in deterministic
        if item["deterministic_status"] == "missing"
        and item["facet_id"] not in missing
    ]
    facet_consensus = {}
    for item in deterministic:
        facet_id = item["facet_id"]
        a = first["facet_mappings"][facet_id]
        b = second["facet_mappings"][facet_id]
        if a["status"] != b["status"]:
            review_reasons.append(f"facet_disagreement:{facet_id}")
        if (
            a["answer_span"] != b["answer_span"]
            or a["evidence_quote_indices"] != b["evidence_quote_indices"]
        ):
            review_reasons.append(f"facet_evidence_disagreement:{facet_id}")
        facet_consensus[facet_id] = {
            "deterministic_status": item["deterministic_status"],
            "pass_a_status": a["status"],
            "pass_b_status": b["status"],
            "consensus_supported": (
                facet_id not in missing
                and a["status"] == b["status"] == "supported"
            ),
        }
    completeness_pass = not missing and all(
        item["consensus_supported"] for item in facet_consensus.values()
    )
    if missing:
        review_reasons.extend(f"deterministic_missing_facet:{item}" for item in missing)
    review_reasons.extend(
        f"heuristic_missing_facet_review:{item}" for item in heuristic_missing
    )

    gate_results = {
        "exact_evidence_quote_match": {
            "verdict": "pass",
            "basis": "structural_exact_substring_and_sha256",
        },
        "question_answer_completeness": {
            "verdict": "pass" if completeness_pass else "fail",
            "basis": "deterministic_facets_and_two_pass_exact_span_evidence_consensus",
        },
    }
    for gate in MODEL_GATES:
        a = first["gate_evidence"][gate]["verdict"]
        b = second["gate_evidence"][gate]["verdict"]
        expected = _expected_gate_verdict(packet, gate)
        if a != b:
            review_reasons.append(f"judge_gate_disagreement:{gate}")
        if (
            first["gate_evidence"][gate]["evidence_quote_indices"]
            != second["gate_evidence"][gate]["evidence_quote_indices"]
        ):
            review_reasons.append(f"judge_gate_evidence_disagreement:{gate}")
        if expected == "not_applicable":
            verdict = "not_applicable"
        else:
            verdict = "pass" if a == b == "pass" else "fail"
        gate_results[gate] = {"verdict": verdict, "pass_a": a, "pass_b": b}

    if packet["generation_mode"] != "calibrated_hard_negative":
        nli_verdict = nli_result["verdict"]
        if nli_verdict != "pass":
            gate_results["source_entailment"]["verdict"] = "fail"
            review_reasons.append(f"nli_nonpass:{nli_verdict}")
        if (
            nli_verdict != "pass"
            and first["gate_evidence"]["source_entailment"]["verdict"] == "pass"
            and second["gate_evidence"]["source_entailment"]["verdict"] == "pass"
        ):
            review_reasons.append("judge_nli_disagreement")
    if first["confidence"] != "high" or second["confidence"] != "high":
        review_reasons.append("judge_not_high_confidence")
    failure_codes = {
        PASS_NAMES[0]: first["failure_codes"],
        PASS_NAMES[1]: second["failure_codes"],
    }
    if first["failure_codes"] != second["failure_codes"]:
        review_reasons.append("judge_failure_code_disagreement")
    if first["failure_codes"] or second["failure_codes"]:
        review_reasons.append("judge_reported_failure_code")
    if is_high_risk_sample(packet):
        review_reasons.append("deterministic_high_risk_review_sample")

    applicable_pass = not any(failure_codes.values()) and all(
        value["verdict"] in {"pass", "not_applicable"}
        for value in gate_results.values()
    )
    manual_review_required = bool(review_reasons)
    return {
        "schema": RESULT_SCHEMA,
        "packet_id": packet["packet_id"],
        "candidate_example_id": packet["candidate_example_id"],
        "request_id": packet["request_id"],
        "source_group_id": packet["source_group_id"],
        "assistant_qwen36_token_count": None,
        "deterministic_facets": deterministic,
        "facet_consensus": facet_consensus,
        "gate_results": gate_results,
        "nli_verdict": nli_result["verdict"],
        "failure_codes_by_pass": failure_codes,
        "judge_consensus_passed": applicable_pass,
        "manual_review_required": manual_review_required,
        "manual_review_reasons": sorted(set(review_reasons)),
        "semantic_verification_status": (
            "passed_manual_or_global_selection_pending"
            if applicable_pass else "rejected_or_manual_review"
        ),
        "eligible_for_training": False,
    }


def make_record(
    group: Sequence[dict],
    pass_outputs: dict[str, dict],
    nli_index: dict[str, dict],
    *,
    run_contract_sha256: str,
) -> dict:
    validated = {}
    validation_errors = {}
    invalid_raw_outputs = {}
    for name in PASS_NAMES:
        normalized, errors, invalid_raw = normalize_pass_output(
            pass_outputs[name], group
        )
        validated[name] = normalized
        validation_errors[name] = errors
        invalid_raw_outputs[name] = invalid_raw
    results = []
    for packet in group:
        candidate_id = packet["candidate_example_id"]
        candidate_validation_errors = {
            name: validation_errors[name][candidate_id]
            for name in PASS_NAMES
            if candidate_id in validation_errors[name]
        }
        result = aggregate_candidate(
            packet,
            validated[PASS_NAMES[0]][candidate_id],
            validated[PASS_NAMES[1]][candidate_id],
            nli_index[candidate_id],
            pass_validation_errors=candidate_validation_errors,
        )
        result["assistant_qwen36_token_count"] = packet[
            "assistant_qwen36_token_count"
        ]
        results.append(result)
    record = {
        "schema": RECORD_SCHEMA,
        "request_id": group[0]["request_id"],
        "source_group_id": group[0]["source_group_id"],
        "candidate_example_ids": [item["candidate_example_id"] for item in group],
        "pass_output_sha256s": {
            name: corpus.canonical_sha256(pass_outputs[name]) for name in PASS_NAMES
        },
        "normalized_pass_outputs": validated,
        "pass_validation_errors": validation_errors,
        "invalid_raw_pass_outputs": invalid_raw_outputs,
        "results": results,
        "run_contract_sha256": run_contract_sha256,
        "semantic_verification_completed": False,
        "training_rows_emitted": False,
    }
    record["content_sha256_before_self_field"] = _self_address(record)
    return record


def validate_record(record: dict, group: Sequence[dict], run_contract_sha256: str) -> None:
    _require_self_address(record, "semantic judge record")
    if (
        record.get("schema") != RECORD_SCHEMA
        or record.get("request_id") != group[0]["request_id"]
        or record.get("source_group_id") != group[0]["source_group_id"]
        or record.get("candidate_example_ids")
        != [item["candidate_example_id"] for item in group]
        or record.get("run_contract_sha256") != run_contract_sha256
        or record.get("semantic_verification_completed") is not False
        or record.get("training_rows_emitted") is not False
        or len(record.get("results", [])) != len(group)
        or set(record.get("normalized_pass_outputs", {})) != set(PASS_NAMES)
        or set(record.get("pass_validation_errors", {})) != set(PASS_NAMES)
        or set(record.get("invalid_raw_pass_outputs", {})) != set(PASS_NAMES)
        or any(item.get("eligible_for_training") is not False for item in record["results"])
    ):
        raise RuntimeError("semantic judge record contract changed")


def build_run_contract(
    *,
    shard_index: int,
    structural_summary: dict,
    nli_report: dict,
    model_receipts: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    prompt_statistics: dict,
) -> dict:
    worker = Path(__file__).resolve()
    contract = {
        "schema": "high-information-semantic-judge-run-contract-v1",
        "gpu_shard": shard_index,
        "structural_review_sha256": structural_summary["report_file_sha256"],
        "nli_output_sha256": nli_report["output_sha256"],
        "nli_worker_file_sha256": nli_report["worker_file_sha256"],
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "model_blob_receipts": model_receipts,
        "vllm_version": VLLM_VERSION,
        "dtype": "bfloat16",
        "passes": list(PASS_NAMES),
        "candidate_grouping": "singleton_sorted_within_request_v1",
        "guided_schema_sha256": corpus.canonical_sha256(GUIDED_SCHEMA),
        "worker_file_sha256": corpus.file_sha256(worker),
        "facet_implementation_sha256": corpus.file_sha256(Path(facets.__file__).resolve()),
        "nli_validation_implementation_sha256": corpus.file_sha256(Path(nli.__file__).resolve()),
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "request_batch_size": args.request_batch_size,
        "temperature": 0.0,
        "prompt_statistics": prompt_statistics,
        "training_rows_emitted": False,
    }
    contract["content_sha256_before_self_field"] = _self_address(contract)
    return contract


def mistral_prompt_statistics(groups: Sequence[Sequence[dict]], max_tokens: int) -> dict:
    from mistral_common.protocol.instruct.messages import SystemMessage, UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tokenizer = MistralTokenizer.from_file(MODEL_DIRECTORY / "tekken.json")
    counts = []
    for group in groups:
        for pass_name in PASS_NAMES:
            messages = judge_messages(group, pass_name)
            request = ChatCompletionRequest(
                messages=[
                    SystemMessage(content=messages[0]["content"]),
                    UserMessage(content=messages[1]["content"]),
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            counts.append(len(tokenizer.encode_chat_completion(request).tokens))
    if not counts:
        raise RuntimeError("semantic judge has no rendered prompts")
    return {
        "official_tokenizer": "mistral_common.MistralTokenizer.from_file(tekken.json)",
        "prompts": len(counts),
        "minimum_prompt_tokens": min(counts),
        "maximum_prompt_tokens": max(counts),
        "maximum_prompt_plus_output_budget": max(counts) + max_tokens,
    }


def _load_partial(path: Path, groups: Sequence[Sequence[dict]], contract_sha256: str) -> list[dict]:
    if not path.exists():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) > len(groups):
        raise RuntimeError("semantic judge partial exceeds request groups")
    for row, group in zip(rows, groups, strict=False):
        validate_record(row, group, contract_sha256)
    return rows


def _payload(rows: Sequence[dict]) -> bytes:
    return corpus.jsonl_payload(rows)


def preflight(args: argparse.Namespace) -> dict:
    model_receipts = validate_model_snapshot(args.model_directory)
    review_path, summary_path = nli.structural_paths(args.shard_index)
    packets, structural_summary = nli.load_structural_packets(
        args.shard_index, review_path, summary_path
    )
    nli_index, nli_report = load_nli_results(args.shard_index, packets)
    groups = groups_by_request(packets)
    prompt_statistics = mistral_prompt_statistics(groups, args.max_tokens)
    if prompt_statistics["maximum_prompt_plus_output_budget"] > args.max_model_len:
        raise RuntimeError("semantic judge prompt plus output exceeds max model length")
    contract = build_run_contract(
        shard_index=args.shard_index,
        structural_summary=structural_summary,
        nli_report=nli_report,
        model_receipts=model_receipts,
        args=args,
        prompt_statistics=prompt_statistics,
    )
    return {
        "status": "semantic_judge_preflight_complete_no_gpu_launch",
        "gpu_shard": args.shard_index,
        "request_groups": len(groups),
        "candidates": len(packets),
        "nli_results": len(nli_index),
        "run_contract_sha256": contract["content_sha256_before_self_field"],
        "model_revision": MODEL_REVISION,
        "prompt_statistics": prompt_statistics,
        "training_rows_emitted": False,
    }


def run(args: argparse.Namespace) -> dict:
    started_at = utc_now()
    model_receipts = validate_model_snapshot(args.model_directory)
    review_path, summary_path = nli.structural_paths(args.shard_index)
    packets, structural_summary = nli.load_structural_packets(
        args.shard_index, review_path, summary_path
    )
    nli_index, nli_report = load_nli_results(args.shard_index, packets)
    groups = groups_by_request(packets)
    if args.smoke:
        groups = groups[:1]
    prompt_statistics = mistral_prompt_statistics(groups, args.max_tokens)
    if prompt_statistics["maximum_prompt_plus_output_budget"] > args.max_model_len:
        raise RuntimeError("semantic judge prompt plus output exceeds max model length")
    contract = build_run_contract(
        shard_index=args.shard_index,
        structural_summary=structural_summary,
        nli_report=nli_report,
        model_receipts=model_receipts,
        args=args,
        prompt_statistics=prompt_statistics,
    )
    contract_sha256 = contract["content_sha256_before_self_field"]
    paths = output_paths(args.shard_index, smoke=args.smoke)
    rows = _load_partial(paths["partial"], groups, contract_sha256)
    completed = len(rows)

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu_index):
        raise RuntimeError("semantic judge requires exactly its assigned physical GPU")
    from vllm import LLM, SamplingParams, __version__ as vllm_version
    from vllm.sampling_params import StructuredOutputsParams

    if vllm_version != VLLM_VERSION:
        raise RuntimeError("semantic judge requires vLLM 0.25.0 exactly")
    engine = LLM(
        model=str(args.model_directory),
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=32,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        enforce_eager=args.enforce_eager,
        seed=17,
    )
    structured = StructuredOutputsParams(
        json=GUIDED_SCHEMA,
        disable_additional_properties=True,
    )
    for batch_start in range(completed, len(groups), args.request_batch_size):
        batch_started = time.monotonic()
        batch_groups = groups[batch_start : batch_start + args.request_batch_size]
        conversations = []
        parameters = []
        mapping = []
        for group_index, group in enumerate(batch_groups):
            for pass_index, pass_name in enumerate(PASS_NAMES):
                conversations.append(judge_messages(group, pass_name))
                parameters.append(
                    SamplingParams(
                        temperature=0.0,
                        seed=1009 + pass_index,
                        max_tokens=args.max_tokens,
                        structured_outputs=structured,
                    )
                )
                mapping.append((group_index, pass_name))
        outputs = engine.chat(
            conversations,
            sampling_params=parameters,
            use_tqdm=False,
        )
        if len(outputs) != len(mapping):
            raise RuntimeError("semantic judge output count changed")
        parsed_by_group: list[dict[str, dict]] = [dict() for _ in batch_groups]
        for output, (group_index, pass_name) in zip(outputs, mapping, strict=True):
            if len(output.outputs) != 1:
                raise RuntimeError("semantic judge output multiplicity changed")
            parsed_by_group[group_index][pass_name] = parse_judge_output(
                output.outputs[0].text
            )
        for group, pass_outputs in zip(batch_groups, parsed_by_group, strict=True):
            record = make_record(
                group,
                pass_outputs,
                nli_index,
                run_contract_sha256=contract_sha256,
            )
            validate_record(record, group, contract_sha256)
            rows.append(record)
        _atomic_write(paths["partial"], _payload(rows))
        telemetry = {
            "schema": "high-information-semantic-judge-telemetry-v1",
            "updated_at": utc_now(),
            "status": "running",
            "pid": os.getpid(),
            "gpu_shard": args.shard_index,
            "physical_gpu_index": args.gpu_index,
            "request_groups": len(groups),
            "completed_request_groups": len(rows),
            "candidates_completed": sum(len(row["results"]) for row in rows),
            "last_batch_seconds": time.monotonic() - batch_started,
            "run_contract_sha256": contract_sha256,
        }
        _atomic_write(paths["telemetry"], corpus.canonical_bytes(telemetry))
        print(json.dumps(telemetry, sort_keys=True), flush=True)

    output_payload = _payload(rows)
    _atomic_write(paths["output"], output_payload)
    results = [result for row in rows for result in row["results"]]
    report = {
        "schema": "high-information-semantic-judge-report-v1",
        "status": "complete_manual_and_global_selection_pending",
        "started_at": started_at,
        "completed_at": utc_now(),
        "gpu_shard": args.shard_index,
        "physical_gpu_index": args.gpu_index,
        "request_groups": len(groups),
        "candidates": len(results),
        "judge_consensus_passed": sum(item["judge_consensus_passed"] for item in results),
        "manual_review_required": sum(item["manual_review_required"] for item in results),
        "output": corpus.relative(paths["output"]),
        "output_sha256": corpus.sha256_bytes(output_payload),
        "run_contract": contract,
        "semantic_verification_completed": False,
        "training_rows_emitted": False,
    }
    report["content_sha256_before_self_field"] = _self_address(report)
    _atomic_write(paths["report"], (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8"))
    _atomic_write(paths["telemetry"], corpus.canonical_bytes({
        "schema": "high-information-semantic-judge-telemetry-v1",
        "updated_at": utc_now(),
        "status": "complete_manual_and_global_selection_pending",
        "pid": os.getpid(),
        "gpu_shard": args.shard_index,
        "physical_gpu_index": args.gpu_index,
        "request_groups": len(groups),
        "completed_request_groups": len(rows),
        "candidates_completed": len(results),
        "run_contract_sha256": contract_sha256,
    }))
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--shard-index", type=int, required=True)
    result.add_argument("--gpu-index", type=int, required=True)
    result.add_argument("--model-directory", type=Path, default=MODEL_DIRECTORY)
    result.add_argument("--request-batch-size", type=int, default=8)
    result.add_argument("--max-model-len", type=int, default=16_384)
    result.add_argument("--max-tokens", type=int, default=3_072)
    result.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    result.add_argument("--enforce-eager", action="store_true")
    result.add_argument("--smoke", action="store_true")
    result.add_argument("--check-plan", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.shard_index not in range(4) or args.gpu_index != args.shard_index:
        raise ValueError("semantic judge shard 0-3 must use its matching GPU")
    if not 1 <= args.request_batch_size <= 16:
        raise ValueError("semantic judge request batch must be 1-16")
    if not 8_192 <= args.max_model_len <= 32_768:
        raise ValueError("semantic judge max model length is out of bounds")
    if not 1_024 <= args.max_tokens <= 4_096:
        raise ValueError("semantic judge output budget is out of bounds")
    if not 0.6 <= args.gpu_memory_utilization <= 0.95:
        raise ValueError("semantic judge GPU memory utilization is out of bounds")
    value = preflight(args) if args.check_plan else run(args)
    print(json.dumps(value, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
