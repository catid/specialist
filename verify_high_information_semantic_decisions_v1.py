#!/usr/bin/env python3
"""Fail-closed contract for independent domain-candidate semantic decisions.

This module deliberately cannot authorize current candidates: the generated
contract has no pinned independent verifier authority yet.  It defines and
tests the packet, authority, per-gate decision, and aggregate receipt semantics
needed to make that future authorization content-addressed.  Structural passes
alone are never accepted.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Sequence

import build_high_information_domain_corpus_v1 as corpus
import verify_high_information_candidates_v1 as structural


AUTHORITY_SCHEMA = "high-information-semantic-verifier-authority-v1"
PACKET_SCHEMA = "high-information-semantic-verification-packet-v1"
DECISION_SCHEMA = "high-information-semantic-verification-decision-v1"
RECEIPT_SCHEMA = "high-information-semantic-verification-receipt-v1"


def _self_address(value: dict) -> str:
    unsigned = dict(value)
    unsigned.pop("content_sha256_before_self_field", None)
    return corpus.canonical_sha256(unsigned)


def _require_self_address(value: dict, label: str) -> None:
    if value.get("content_sha256_before_self_field") != _self_address(value):
        raise RuntimeError(f"{label} content address changed")


def load_contract() -> dict:
    _, _, _ = structural.load_plan()
    value = json.loads(
        corpus.SEMANTIC_VERIFIER_CONTRACT.read_text(encoding="utf-8")
    )
    _require_self_address(value, "semantic verifier contract")
    if (
        value.get("schema") != "high-information-semantic-verifier-contract-v1"
        or value.get("receipt_schema", {}).get(
            "structural_pass_alone_may_be_accepted"
        )
        is not False
    ):
        raise RuntimeError("semantic verifier contract changed")
    return value


def authority_sha256(authority: dict) -> str:
    _require_self_address(authority, "semantic verifier authority")
    return authority["content_sha256_before_self_field"]


def validate_authority(authority: Any, contract: dict) -> str:
    expected_fields = {
        "schema",
        "authority_kind",
        "verifier_identity_sha256",
        "generator_model_identity_sha256",
        "independent_of_generator",
        "implementation_receipts",
        "prompt_or_protocol_sha256",
        "content_sha256_before_self_field",
    }
    if not isinstance(authority, dict) or set(authority) != expected_fields:
        raise RuntimeError("semantic verifier authority schema changed")
    _require_self_address(authority, "semantic verifier authority")
    digest = authority["content_sha256_before_self_field"]
    allowed = contract["verifier_authority"]["allowed_authority_kinds"]
    receipts = authority["implementation_receipts"]
    if (
        authority["schema"] != AUTHORITY_SCHEMA
        or authority["authority_kind"] not in allowed
        or authority["generator_model_identity_sha256"]
        != contract["generator_model_identity_sha256"]
        or authority["independent_of_generator"] is not True
        or authority["verifier_identity_sha256"]
        == authority["generator_model_identity_sha256"]
        or not isinstance(authority["verifier_identity_sha256"], str)
        or len(authority["verifier_identity_sha256"]) != 64
        or not isinstance(receipts, list)
        or not receipts
        or any(
            not isinstance(item, dict)
            or set(item) != {"path", "file_sha256"}
            or not isinstance(item["path"], str)
            or not isinstance(item["file_sha256"], str)
            or len(item["file_sha256"]) != 64
            for item in receipts
        )
        or not isinstance(authority["prompt_or_protocol_sha256"], str)
        or len(authority["prompt_or_protocol_sha256"]) != 64
    ):
        raise RuntimeError("semantic verifier authority is not independent and pinned")
    pinned = contract.get("pinned_verifier_authority_sha256")
    if not isinstance(pinned, str) or len(pinned) != 64:
        raise RuntimeError("no independent semantic verifier authority is pinned")
    if digest != pinned:
        raise RuntimeError("semantic verifier authority differs from the pinned receipt")
    return digest


def build_packet(candidate: dict, request: dict, context: dict) -> dict:
    if (
        candidate.get("deterministic_structure_status") != "passed"
        or candidate.get("semantic_verification_status") != "pending"
        or candidate.get("eligible_for_training") is not False
        or candidate.get("request_id") != request["request_id"]
        or candidate.get("source_context_id") != context["context_id"]
        or candidate.get("source_group_id") != context["source_group_id"]
    ):
        raise RuntimeError("semantic packet input is not a structural-only pass")
    quotes = candidate.get("evidence_quotes")
    if (
        not isinstance(quotes, list)
        or not quotes
        or any(not isinstance(quote, str) or quote not in context["text"] for quote in quotes)
    ):
        raise RuntimeError("semantic packet evidence is not exact source text")
    quote_hashes = [corpus.sha256_bytes(quote.encode("utf-8")) for quote in quotes]
    identity = {
        "candidate_example_id": candidate["candidate_example_id"],
        "request_id": request["request_id"],
        "context_text_sha256": context["text_sha256"],
        "evidence_quote_sha256s": quote_hashes,
    }
    packet = {
        "schema": PACKET_SCHEMA,
        "packet_id": corpus.content_id("semantic-packet-v1", identity),
        "candidate_example_id": candidate["candidate_example_id"],
        "request_id": request["request_id"],
        "source_context_id": context["context_id"],
        "source_group_id": context["source_group_id"],
        "task_family": request["task_family"],
        "task_subtype": request["task_subtype"],
        "generation_mode": request["generation_mode"],
        "question": candidate["question"],
        "answer": candidate["answer"],
        "assistant_qwen36_token_count": candidate["assistant_qwen36_token_count"],
        "negative_type": candidate["negative_type"],
        "evidence_quotes": quotes,
        "evidence_quote_sha256s": quote_hashes,
        "context_text": context["text"],
        "context_text_sha256": context["text_sha256"],
        "rights_basis": context["rights_basis"],
        "safety_transfer_flags": context["safety_transfer_flags"],
        "structural_status": "passed_semantic_pending",
        "eligible_for_training": False,
    }
    packet["content_sha256_before_self_field"] = _self_address(packet)
    return packet


def expected_gate_verdicts(packet: dict, contract: dict) -> dict[str, str]:
    expected = {
        "exact_evidence_quote_match": "pass",
        "source_entailment": "pass",
        "citation_support_coverage": "pass",
        "question_answer_completeness": "pass",
        "application_correctness": (
            "pass" if packet["task_subtype"] == "application_scenario"
            else "not_applicable"
        ),
        "hard_negative_calibration": (
            "pass" if packet["generation_mode"] == "calibrated_hard_negative"
            else "not_applicable"
        ),
        "safety_transfer_preservation": "pass",
        "attribution_and_scope_preservation": "pass",
        "unsupported_claim_absence": "pass",
        "training_value_and_nontriviality": "pass",
    }
    if set(expected) != set(contract["required_gates"]):
        raise RuntimeError("semantic verifier gate set changed")
    return expected


def validate_decision(
    packet: dict,
    decision: Any,
    authority: dict,
    contract: dict,
) -> dict:
    _require_self_address(packet, "semantic packet")
    authority_digest = validate_authority(authority, contract)
    expected_fields = set(contract["decision_schema"]["exact_fields"])
    if not isinstance(decision, dict) or set(decision) != expected_fields:
        raise RuntimeError("semantic decision schema changed")
    _require_self_address(decision, "semantic decision")
    if (
        decision["schema"] != DECISION_SCHEMA
        or decision["packet_id"] != packet["packet_id"]
        or decision["candidate_example_id"] != packet["candidate_example_id"]
        or decision["verifier_authority_sha256"] != authority_digest
        or decision["evidence_quote_sha256s"] != packet["evidence_quote_sha256s"]
    ):
        raise RuntimeError("semantic decision lineage changed")
    expected = expected_gate_verdicts(packet, contract)
    results = decision["gate_results"]
    if not isinstance(results, dict) or set(results) != set(expected):
        raise RuntimeError("semantic decision gate coverage changed")
    failed = []
    gate_fields = set(contract["decision_schema"]["gate_result_exact_fields"])
    quote_hashes = set(packet["evidence_quote_sha256s"])
    for gate, required in expected.items():
        result = results[gate]
        if not isinstance(result, dict) or set(result) != gate_fields:
            raise RuntimeError(f"semantic gate result schema changed: {gate}")
        if result["method"] != contract["required_gates"][gate]["method"]:
            raise RuntimeError(f"semantic gate method changed: {gate}")
        evidence = result["evidence_sha256s"]
        if (
            not isinstance(evidence, list)
            or any(value not in quote_hashes for value in evidence)
            or (required == "pass" and not evidence)
            or (required == "not_applicable" and evidence)
        ):
            raise RuntimeError(f"semantic gate evidence changed: {gate}")
        verdict = result["verdict"]
        if verdict not in contract["decision_schema"]["allowed_verdicts"]:
            raise RuntimeError(f"semantic gate verdict changed: {gate}")
        if verdict != required:
            failed.append(gate)
    return {
        "schema": "high-information-semantic-example-result-v1",
        "packet_id": packet["packet_id"],
        "candidate_example_id": packet["candidate_example_id"],
        "verifier_authority_sha256": authority_digest,
        "semantic_verification_status": "passed" if not failed else "rejected",
        "failed_gates": failed,
        "eligible_for_training": not failed,
        "decision_content_sha256": decision["content_sha256_before_self_field"],
    }


def build_receipt(
    packets: Sequence[dict],
    decisions: Sequence[dict],
    authority: dict,
    contract: dict,
    *,
    plan_manifest_sha256: str,
    structural_review_sha256: str,
    semantic_packet_sha256: str,
    decision_file_sha256: str,
) -> tuple[dict, list[dict]]:
    if not packets or len(packets) != len(decisions):
        raise RuntimeError("semantic verification requires one decision per packet")
    decision_index = {item.get("packet_id"): item for item in decisions}
    if len(decision_index) != len(decisions):
        raise RuntimeError("semantic decisions duplicate packet identity")
    results = []
    gate_counts: Counter[str] = Counter()
    for packet in packets:
        decision = decision_index.get(packet.get("packet_id"))
        if decision is None:
            raise RuntimeError("semantic verification omitted a structural pass")
        result = validate_decision(packet, decision, authority, contract)
        results.append(result)
        gate_counts.update(
            f"{gate}:{gate_result['verdict']}"
            for gate, gate_result in decision["gate_results"].items()
        )
        if result["eligible_for_training"]:
            gate_counts["accepted"] += 1
        else:
            gate_counts["rejected"] += 1
            gate_counts.update(f"failed:{gate}" for gate in result["failed_gates"])
    accepted = sorted(
        item["candidate_example_id"]
        for item in results if item["eligible_for_training"]
    )
    rejected = sorted(
        item["candidate_example_id"]
        for item in results if not item["eligible_for_training"]
    )
    commitments = {
        "plan_manifest_sha256": plan_manifest_sha256,
        "structural_review_sha256": structural_review_sha256,
        "semantic_packet_sha256": semantic_packet_sha256,
        "decision_file_sha256": decision_file_sha256,
    }
    if any(
        not isinstance(value, str) or len(value) != 64
        for value in commitments.values()
    ):
        raise RuntimeError("semantic receipt input commitment is malformed")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        **commitments,
        "verifier_authority_sha256": authority_sha256(authority),
        "packets": len(packets),
        "decisions": len(decisions),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "accepted_candidate_id_commitment_sha256": corpus.canonical_sha256(accepted),
        "rejected_candidate_id_commitment_sha256": corpus.canonical_sha256(rejected),
        "per_gate_counts": dict(sorted(gate_counts.items())),
        "semantic_verification_completed": True,
        "structural_only_pass_accepted": False,
        "exact_target_selection_completed": False,
        "training_launch_authorized": False,
    }
    receipt["content_sha256_before_self_field"] = _self_address(receipt)
    return receipt, results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-contract", action="store_true")
    arguments = parser.parse_args()
    if not arguments.check_contract:
        parser.error("only --check-contract is available until verifier authority is pinned")
    contract = load_contract()
    print(
        json.dumps(
            {
                "schema": contract["schema"],
                "pinned_verifier_authority": contract[
                    "pinned_verifier_authority_sha256"
                ]
                is not None,
                "structural_only_acceptance": False,
                "semantic_verification_completed": False,
                "training_launch_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
