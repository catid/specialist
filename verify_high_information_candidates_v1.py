#!/usr/bin/env python3
"""Deterministically screen high-information generation candidates.

This verifier consumes only train-derived source contexts, generation requests,
and candidate output.  It never opens a mixed-source lineage path.  Passing the
checks below is necessary but not sufficient: answer entailment, application
correctness, calibrated unanswerability, and safety-scope preservation remain
explicit semantic-verifier gates.  Consequently this script emits review
records, never accepted training rows.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Sequence

import build_high_information_domain_corpus_v1 as corpus


ROOT = Path(__file__).resolve().parent
PLAN_DIR = (
    ROOT / "data/training_inventory/high_information_domain_corpus_v1"
).resolve()
DEFAULT_REPORT = (PLAN_DIR / "candidate_structural_review.jsonl").resolve()
DEFAULT_SUMMARY = (PLAN_DIR / "candidate_structural_review.summary.json").resolve()

CALIBRATION_RE = re.compile(
    r"(?i)\b(?:cannot (?:determine|establish|confirm|verify|conclude)|"
    r"cannot be (?:determined|established|confirmed|verified|concluded)|"
    r"not enough (?:information|evidence)|no (?:information|evidence|support)|"
    r"(?:information|evidence) (?:is|was) not (?:provided|available)|"
    r"lacks? (?:information|evidence|support)|not supported|"
    r"does not (?:say|state|establish|support)|unsupported|false premise|"
    r"insufficient evidence|uncertain|unknown|not provided)\b"
)
GENERATED_CANDIDATE_SCHEMA = "high-information-generated-candidate-v1"
GENERATED_CANDIDATE_FIELDS = {
    "schema",
    "request_id",
    "source_group_id",
    "gpu_shard",
    "examples",
    "parse_status",
    "parse_error",
    "raw_completion_sha256",
    "finish_reason",
    "generated_token_count",
    "generator",
    "semantic_verification_completed",
    "eligible_for_training",
    "content_sha256_before_self_field",
}


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


def _canonical_question(value: str) -> str:
    return " ".join(value.casefold().split())


def _address_review_row(value: dict) -> dict:
    row = dict(value)
    row["schema"] = "high-information-candidate-structural-review-row-v1"
    row["content_sha256_before_self_field"] = corpus.canonical_sha256(row)
    return row


def _assert_plan_local_path(path: Path, *, role: str, must_exist: bool) -> Path:
    resolved = path.expanduser().resolve()
    try:
        relative_path = resolved.relative_to(PLAN_DIR)
    except ValueError as error:
        raise RuntimeError(f"{role} must remain inside the train-derived plan directory") from error
    lowered = relative_path.as_posix().casefold()
    if any(token in lowered for token in corpus.FORBIDDEN_SOURCE_TOKENS):
        raise RuntimeError(f"{role} path crosses a forbidden data boundary")
    if must_exist and (not resolved.is_file() or resolved.suffix != ".jsonl"):
        raise RuntimeError(f"{role} must be an existing JSONL file")
    return resolved


def _validate_receipt(path: Path, receipt: dict, *, rows: int | None = None) -> None:
    if (
        not isinstance(receipt, dict)
        or receipt.get("path") != corpus.relative(path)
        or receipt.get("file_sha256") != corpus.file_sha256(path)
        or receipt.get("file_bytes") != path.stat().st_size
        or (rows is not None and receipt.get("rows") != rows)
    ):
        raise RuntimeError(f"plan artifact no longer matches its manifest receipt: {path.name}")


def load_plan(plan_dir: Path = PLAN_DIR) -> tuple[dict, dict[str, dict], dict[str, dict]]:
    if plan_dir.resolve() != PLAN_DIR:
        raise RuntimeError("candidate verifier may open only the sealed train-derived plan")
    manifest_path = plan_dir / "manifest.json"
    spec_path = plan_dir / "prompt_spec.json"
    semantic_contract_path = plan_dir / "semantic_verifier_contract.json"
    context_path = plan_dir / "source_contexts.jsonl"
    shard_paths = [plan_dir / f"generation_requests_gpu{index}.jsonl" for index in range(4)]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_declared = manifest.get("content_sha256_before_self_field")
    manifest_unsigned = dict(manifest)
    manifest_unsigned.pop("content_sha256_before_self_field", None)
    if (
        manifest.get("schema") != corpus.SCHEMA
        or corpus.canonical_sha256(manifest_unsigned) != manifest_declared
        or manifest.get("input_boundary", {}).get(
            "only_sealed_train_projection_semantics_opened"
        )
        is not True
        or manifest.get("input_boundary", {}).get("final_or_protected_source_opened")
        is not False
        or manifest.get("invariants", {}).get("training_launch_authorized") is not False
    ):
        raise RuntimeError("high-information plan manifest contract changed")
    code_receipts = {
        "candidate_worker": ROOT / "run_high_information_candidate_shard_v1.py",
        "candidate_verifier": Path(__file__).resolve(),
    }
    for key, path in code_receipts.items():
        receipt = manifest.get(key)
        if (
            not isinstance(receipt, dict)
            or receipt.get("path") != corpus.relative(path)
            or receipt.get("file_sha256") != corpus.file_sha256(path)
        ):
            raise RuntimeError(f"{key} implementation no longer matches the plan")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    declared = spec.get("content_sha256_before_self_field")
    unsigned = dict(spec)
    unsigned.pop("content_sha256_before_self_field", None)
    if corpus.canonical_sha256(unsigned) != declared:
        raise RuntimeError("generation prompt spec self address changed")
    _validate_receipt(spec_path, manifest.get("prompt_spec"))
    semantic_contract = json.loads(
        semantic_contract_path.read_text(encoding="utf-8")
    )
    semantic_declared = semantic_contract.get("content_sha256_before_self_field")
    semantic_unsigned = dict(semantic_contract)
    semantic_unsigned.pop("content_sha256_before_self_field", None)
    if (
        semantic_contract.get("schema")
        != "high-information-semantic-verifier-contract-v1"
        or corpus.canonical_sha256(semantic_unsigned) != semantic_declared
        or semantic_contract.get("receipt_schema", {}).get(
            "structural_pass_alone_may_be_accepted"
        )
        is not False
    ):
        raise RuntimeError("semantic verifier contract changed")
    _validate_receipt(
        semantic_contract_path,
        manifest.get("semantic_verifier_contract"),
    )
    contexts = {}
    with context_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            context_id = row.get("context_id")
            text = row.get("text")
            if (
                row.get("schema") != "high-information-source-context-v1"
                or not isinstance(context_id, str)
                or context_id in contexts
                or not isinstance(text, str)
                or corpus.sha256_bytes(text.encode("utf-8")) != row.get("text_sha256")
                or corpus.URL_RE.search(text)
                or corpus.DOMAIN_RE.search(text)
                or row.get("context_id")
                != corpus.content_id(
                    "source-context-v1",
                    {
                        "source_group_id": row.get("source_group_id"),
                        "span_ids": row.get("span_ids"),
                        "text_sha256": row.get("text_sha256"),
                    },
                )
            ):
                raise RuntimeError("source context identity or URL scrub changed")
            contexts[context_id] = row
    _validate_receipt(
        context_path,
        manifest.get("source_contexts", {}).get("receipt"),
        rows=len(contexts),
    )
    requests = {}
    shard_receipts = manifest.get("generation_requests", {}).get("gpu_shards")
    if not isinstance(shard_receipts, list) or len(shard_receipts) != 4:
        raise RuntimeError("generation request shard receipt contract changed")
    for shard, path in enumerate(shard_paths):
        shard_rows = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                request_id = row.get("request_id")
                identity = {
                    "prompt_spec_sha256": row.get("prompt_spec_sha256"),
                    "context_id": row.get("source_context_id"),
                    "source_group_id": row.get("source_group_id"),
                    "family": row.get("task_family"),
                    "subtype": row.get("task_subtype"),
                    "mode": row.get("generation_mode"),
                    "target_verified_assistant_tokens": row.get(
                        "target_verified_assistant_tokens"
                    ),
                }
                if (
                    row.get("schema") != "high-information-generation-request-v1"
                    or row.get("gpu_shard") != shard
                    or row.get("prompt_spec_sha256") != declared
                    or not isinstance(request_id, str)
                    or request_id in requests
                    or row.get("source_context_id") not in contexts
                    or row.get("source_group_id")
                    != contexts[row["source_context_id"]]["source_group_id"]
                    or request_id
                    != corpus.content_id("generation-request-v1", identity)
                ):
                    raise RuntimeError("generation request identity changed")
                requests[request_id] = row
                shard_rows += 1
        _validate_receipt(path, shard_receipts[shard], rows=shard_rows)
    if {row["source_group_id"] for row in requests.values()} != {
        row["source_group_id"] for row in contexts.values()
    }:
        raise RuntimeError("request shards no longer cover every source context")
    return spec, contexts, requests


def candidate_training_messages(
    *, request: dict, context: dict, question: str, answer: str
) -> list[dict[str, str]]:
    if request["task_family"] == "grounded_synthesis":
        system = (
            "Answer using only the supplied domain context. Preserve uncertainty, "
            "scope, attribution, and safety limits."
        )
        user = f"Domain context:\n\n{context['text']}\n\nQuestion:\n{question}"
    elif request["task_family"] == "closed_book_application":
        system = (
            "Answer the domain question factually and directly. State uncertainty "
            "rather than inventing unsupported details."
        )
        user = question
    else:
        raise RuntimeError("candidate request has an unknown task family")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": answer},
    ]


def validate_generated_candidate_envelope(candidate: Any, request: dict) -> list[str]:
    if not isinstance(candidate, dict) or set(candidate) != GENERATED_CANDIDATE_FIELDS:
        return ["generated_candidate_envelope_schema_changed"]
    unsigned = dict(candidate)
    declared = unsigned.pop("content_sha256_before_self_field", None)
    errors = []
    if candidate.get("schema") != GENERATED_CANDIDATE_SCHEMA:
        errors.append("generated_candidate_schema_changed")
    if candidate.get("request_id") != request["request_id"]:
        errors.append("generated_candidate_request_identity_changed")
    if candidate.get("source_group_id") != request["source_group_id"]:
        errors.append("generated_candidate_source_group_changed")
    if candidate.get("gpu_shard") != request["gpu_shard"]:
        errors.append("generated_candidate_gpu_shard_changed")
    if candidate.get("semantic_verification_completed") is not False:
        errors.append("generated_candidate_claims_semantic_verification")
    if candidate.get("eligible_for_training") is not False:
        errors.append("generated_candidate_claims_training_eligibility")
    if corpus.canonical_sha256(unsigned) != declared:
        errors.append("generated_candidate_content_address_changed")
    parse_error = candidate.get("parse_error")
    if (
        (parse_error is None and candidate.get("parse_status") != "parsed_unverified")
        or (parse_error is not None and candidate.get("parse_status") != "rejected_parse")
    ):
        errors.append("generated_candidate_parse_status_changed")
    if parse_error is not None and candidate.get("examples") != []:
        errors.append("parse_rejection_contains_examples")
    if (
        not isinstance(candidate.get("generated_token_count"), int)
        or candidate["generated_token_count"] <= 0
    ):
        errors.append("generated_candidate_token_count_invalid")
    return sorted(set(errors))


def verify_example(
    example: dict,
    *,
    request: dict,
    context: dict,
    tokenizer: Any,
) -> tuple[dict | None, list[str]]:
    errors = []
    required = {"example_type", "question", "answer", "evidence_quotes", "negative_type"}
    if not isinstance(example, dict) or set(example) != required:
        return None, ["candidate_example_schema_changed"]
    question = example["question"]
    answer = example["answer"]
    evidence = example["evidence_quotes"]
    if not isinstance(question, str) or not question.strip():
        errors.append("empty_question")
    if not isinstance(answer, str) or not answer.strip():
        errors.append("empty_answer")
    if not isinstance(evidence, list) or not evidence or any(
        not isinstance(value, str) or not value.strip() for value in evidence
    ):
        errors.append("missing_exact_evidence_quote")
    combined = "\n".join(
        value
        for value in [question, answer, *(evidence if isinstance(evidence, list) else [])]
        if isinstance(value, str)
    )
    if corpus.URL_RE.search(combined) or corpus.DOMAIN_RE.search(combined):
        errors.append("url_or_domain_surface")
    if corpus.URL_MEMORIZATION_RE.search(question if isinstance(question, str) else ""):
        errors.append("url_memorization_question")
    if corpus.HIDDEN_REASONING_RE.search(combined):
        errors.append("hidden_reasoning_or_protocol_surface")
    if isinstance(evidence, list) and any(
        quote not in context["text"] for quote in evidence if isinstance(quote, str)
    ):
        errors.append("evidence_quote_not_in_source_context")
    negative = request["generation_mode"] == "calibrated_hard_negative"
    expected_example_type = (
        "calibrated_hard_negative" if negative else request["task_subtype"]
    )
    if example.get("example_type") != expected_example_type:
        errors.append("example_type_does_not_match_request")
    if negative:
        if example.get("negative_type") not in {
            "answer_absent_from_context",
            "false_premise",
            "unsupported_precision_or_threshold",
            "source_scope_or_authority_mismatch",
            "conflicting_or_insufficient_evidence",
        }:
            errors.append("missing_or_invalid_negative_type")
        if isinstance(answer, str) and not CALIBRATION_RE.search(answer):
            errors.append("hard_negative_answer_is_not_calibrated")
    elif example.get("negative_type") is not None:
        errors.append("positive_example_has_negative_type")
    if errors:
        return None, sorted(set(errors))
    messages = candidate_training_messages(
        request=request,
        context=context,
        question=question,
        answer=answer,
    )
    answer_tokens = corpus.official_assistant_token_count(tokenizer, messages)
    if answer_tokens <= 0:
        return None, ["answer_has_no_tokens"]
    identity = {
        "request_id": request["request_id"],
        "question_sha256": corpus.sha256_bytes(question.encode("utf-8")),
        "answer_sha256": corpus.sha256_bytes(answer.encode("utf-8")),
        "evidence_sha256s": [
            corpus.sha256_bytes(value.encode("utf-8")) for value in evidence
        ],
    }
    return {
        "candidate_example_id": corpus.content_id("candidate-example-v1", identity),
        "request_id": request["request_id"],
        "source_context_id": context["context_id"],
        "source_group_id": context["source_group_id"],
        "task_family": request["task_family"],
        "task_subtype": request["task_subtype"],
        "generation_mode": request["generation_mode"],
        "question": question,
        "answer": answer,
        "assistant_qwen36_token_count": answer_tokens,
        "assistant_token_mask_method": corpus.ASSISTANT_MASK_METHOD,
        "enable_thinking": False,
        "evidence_quotes": evidence,
        "negative_type": example["negative_type"],
        "rights_basis": context["rights_basis"],
        "safety_transfer_flags": context["safety_transfer_flags"],
        "lineage": context["lineage"],
        "deterministic_structure_status": "passed",
        "semantic_verification_status": "pending",
        "semantic_verification_required": [
            "answer_entailment",
            "application_correctness",
            "hard_negative_unanswerability_or_false_premise",
            "safety_scope_and_attribution_preservation",
        ],
        "eligible_for_training": False,
    }, []


def verify_candidate_record(
    candidate: dict,
    *,
    requests: dict[str, dict],
    contexts: dict[str, dict],
    tokenizer: Any,
) -> dict:
    request_id = candidate.get("request_id") if isinstance(candidate, dict) else None
    if request_id not in requests:
        return {
            "request_id": request_id,
            "status": "rejected",
            "errors": ["unknown_request_id"],
            "structurally_valid_examples": [],
        }
    examples = candidate.get("examples")
    if not isinstance(examples, list) or not examples:
        return {
            "request_id": request_id,
            "status": "rejected",
            "errors": ["missing_candidate_examples"],
            "structurally_valid_examples": [],
        }
    request = requests[request_id]
    if len(examples) != request.get("candidate_count"):
        return {
            "request_id": request_id,
            "source_group_id": request["source_group_id"],
            "status": "rejected",
            "errors": ["candidate_example_count_does_not_match_request"],
            "structurally_valid_examples": [],
            "semantic_verification_completed": False,
            "training_rows_emitted": False,
        }
    context = contexts[request["source_context_id"]]
    accepted = []
    errors = []
    questions: set[str] = set()
    for index, example in enumerate(examples):
        verified, item_errors = verify_example(
            example, request=request, context=context, tokenizer=tokenizer
        )
        if verified is not None:
            key = _canonical_question(verified["question"])
            if key in questions:
                item_errors = ["duplicate_question_within_candidate"]
                verified = None
            else:
                questions.add(key)
        if verified is not None:
            accepted.append(verified)
        if item_errors:
            errors.append({"example_index": index, "errors": item_errors})
    generated_tokens = sum(
        item["assistant_qwen36_token_count"] for item in accepted
    )
    return {
        "request_id": request_id,
        "source_group_id": request["source_group_id"],
        "status": (
            "structurally_valid_semantic_verification_pending"
            if accepted and not errors
            else "partially_or_fully_rejected"
        ),
        "target_verified_assistant_tokens": request[
            "target_verified_assistant_tokens"
        ],
        "structurally_valid_assistant_tokens": generated_tokens,
        "structurally_valid_examples": accepted,
        "errors": errors,
        "semantic_verification_completed": False,
        "training_rows_emitted": False,
    }


def verify_candidates(
    candidate_path: Path,
    *,
    report_path: Path = DEFAULT_REPORT,
    summary_path: Path = DEFAULT_SUMMARY,
) -> dict:
    candidate_path = _assert_plan_local_path(
        candidate_path, role="candidate input", must_exist=True
    )
    report_path = _assert_plan_local_path(
        report_path, role="candidate review output", must_exist=False
    )
    summary_path = _assert_plan_local_path(
        summary_path, role="candidate summary output", must_exist=False
    )
    reserved_inputs = {
        (PLAN_DIR / "manifest.json").resolve(),
        (PLAN_DIR / "prompt_spec.json").resolve(),
        (PLAN_DIR / "source_contexts.jsonl").resolve(),
        *(
            (PLAN_DIR / f"generation_requests_gpu{index}.jsonl").resolve()
            for index in range(4)
        ),
    }
    if candidate_path in reserved_inputs or report_path in reserved_inputs or summary_path in reserved_inputs:
        raise RuntimeError("candidate verification path overlaps an immutable plan input")
    if len({report_path, summary_path}) != 2:
        raise RuntimeError("candidate review and summary outputs must be distinct")
    _, contexts, requests = load_plan()
    tokenizer = corpus.load_tokenizer()
    seen_requests: set[str] = set()
    reports = []
    with candidate_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            candidate = json.loads(line)
            request_id = candidate.get("request_id") if isinstance(candidate, dict) else None
            if request_id in requests:
                envelope_errors = validate_generated_candidate_envelope(
                    candidate, requests[request_id]
                )
                if envelope_errors:
                    reports.append(
                        _address_review_row({
                            "request_id": request_id,
                            "status": "rejected",
                            "errors": envelope_errors,
                            "structurally_valid_examples": [],
                        })
                    )
                    continue
            if isinstance(request_id, str) and request_id in seen_requests:
                reports.append(
                    _address_review_row({
                        "request_id": request_id,
                        "status": "rejected",
                        "errors": ["duplicate_candidate_request_id"],
                        "structurally_valid_examples": [],
                    })
                )
                continue
            if isinstance(request_id, str):
                seen_requests.add(request_id)
            reports.append(
                _address_review_row(verify_candidate_record(
                    candidate,
                    requests=requests,
                    contexts=contexts,
                    tokenizer=tokenizer,
                ))
            )
    report_payload = corpus.jsonl_payload(reports)
    _atomic_write(report_path, report_payload)
    summary = {
        "schema": "high-information-candidate-structural-review-v1",
        "candidate_path": corpus.relative(candidate_path),
        "candidate_file_sha256": corpus.file_sha256(candidate_path),
        "requests_with_candidate_records": len(reports),
        "fully_structurally_valid_requests": sum(
            item["status"] == "structurally_valid_semantic_verification_pending"
            for item in reports
        ),
        "structurally_valid_examples": sum(
            len(item.get("structurally_valid_examples", [])) for item in reports
        ),
        "structurally_valid_assistant_tokens": sum(
            item.get("structurally_valid_assistant_tokens", 0) for item in reports
        ),
        "semantic_verification_completed": False,
        "accepted_training_rows_emitted": False,
        "report_path": corpus.relative(report_path),
        "report_file_sha256": corpus.sha256_bytes(report_payload),
    }
    summary["content_sha256_before_self_field"] = corpus.canonical_sha256(summary)
    summary_payload = (
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write(summary_path, summary_payload)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-plan", action="store_true")
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    arguments = parser.parse_args()
    if arguments.check_plan:
        _, contexts, requests = load_plan()
        print(
            json.dumps(
                {
                    "contexts": len(contexts),
                    "requests": len(requests),
                    "semantic_verification_completed": False,
                    "training_rows_emitted": False,
                },
                sort_keys=True,
            )
        )
        return
    if arguments.candidates is None:
        parser.error("--candidates is required unless --check-plan is used")
    summary = verify_candidates(
        arguments.candidates,
        report_path=arguments.report,
        summary_path=arguments.summary,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
