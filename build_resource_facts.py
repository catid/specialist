#!/usr/bin/env python3
"""Validate manually reviewed web-resource facts and build training QA."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

from build_leakfree_qa import eval_facts
from build_manual_qa import CONTEXT_DEPENDENT
from qa_quality import (LOW_VALUE, has_protocol_tokens, leakage_reason,
                        normalize_text, parse_qa, stable_fact_id)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEFAULT_MANIFEST = ROOT / "sources" / "rope_resources_v1.json"
DEFAULT_PACKETS = ROOT / "sources" / "manual_facts"
DEFAULT_OUTPUT = DATA / "rope_resource_factual_qa_v1.jsonl"
DEFAULT_REPORT = DATA / "rope_resource_factual_qa_v1.report.json"
DEFAULT_EVAL = [DATA / "eval_qa.jsonl", DATA / "eval_qa_v2.jsonl"]
CLAIM_TYPES = {
    "durable_org", "instructional", "manufacturer_attributed",
    "vendor_attributed",
}
VOLATILE = re.compile(
    r"\b(?:today|currently|upcoming|next event|when is|what date|schedule|"
    r"ticket(?:s|ing)?|price|cost|in stock|sold out|available now|weekly|"
    r"every (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b|"
    r"[$€£]\s*\d",
    re.IGNORECASE,
)
TOKENS = re.compile(r"\d+|[^\W\d_]+", re.UNICODE)
STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "by", "does", "for", "from",
    "how", "in", "is", "it", "its", "of", "on", "or", "say", "says",
    "that", "the", "their", "them", "they", "this", "to", "we", "what",
    "which", "who", "with",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def normalized_host(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username:
        raise ValueError(f"expected a public HTTPS URL, got {url!r}")
    host = parsed.hostname.casefold()
    return host[4:] if host.startswith("www.") else host


def evidence_support(answer: str, evidence: str) -> float:
    answer_tokens = [token for token in TOKENS.findall(normalize_text(answer))
                     if token not in STOPWORDS]
    evidence_tokens = set(TOKENS.findall(normalize_text(evidence)))
    if not answer_tokens:
        return 0.0
    return sum(token in evidence_tokens for token in answer_tokens) / len(answer_tokens)


def load_resources(manifest_path: Path) -> dict[str, dict]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "rope-resource-manifest-v1":
        raise ValueError(f"{manifest_path}: unsupported manifest schema")
    resources = manifest.get("resources")
    if not isinstance(resources, list):
        raise ValueError(f"{manifest_path}: resources must be a list")
    by_id = {}
    for resource in resources:
        resource_id = resource.get("id")
        if not isinstance(resource_id, str) or resource_id in by_id:
            raise ValueError(f"{manifest_path}: missing or duplicate resource id")
        by_id[resource_id] = resource
    return by_id


def build(manifest_path: Path, packet_paths: list[Path], output_path: Path,
          report_path: Path, facts) -> dict:
    resources = load_resources(manifest_path)
    output = []
    questions = {}
    fact_ids = {}
    counts_by_claim = collections.Counter()
    counts_by_packet = collections.Counter()
    counts_by_resource = collections.Counter()
    for packet_path in sorted(packet_paths):
        with packet_path.open() as packet:
            for line_number, line in enumerate(packet, 1):
                if not line.strip():
                    continue
                location = f"{packet_path}:{line_number}"
                item = json.loads(line)
                question = require_text(item, "question", location)
                answer = require_text(item, "answer", location)
                evidence = require_text(item, "evidence", location)
                evidence_url = require_text(item, "evidence_url", location)
                resource_id = require_text(item, "resource_id", location)
                source = require_text(item, "source", location)
                reviewer = require_text(item, "reviewer", location)
                verified_at = require_text(item, "verified_at", location)
                claim_type = require_text(item, "claim_type", location)
                if resource_id not in resources:
                    raise ValueError(f"{location}: unknown resource_id {resource_id!r}")
                if claim_type not in CLAIM_TYPES:
                    raise ValueError(f"{location}: unsupported claim_type {claim_type!r}")
                try:
                    observed = date.fromisoformat(verified_at)
                except ValueError as exc:
                    raise ValueError(f"{location}: verified_at must be YYYY-MM-DD") from exc
                if observed > date.today():
                    raise ValueError(f"{location}: verified_at is in the future")
                if normalized_host(evidence_url) != normalized_host(
                        resources[resource_id]["canonical_url"]):
                    raise ValueError(f"{location}: evidence_url is off the resource site")
                if not question.endswith("?"):
                    raise ValueError(f"{location}: question must end with '?'")
                if not (10 <= len(question) <= 300 and 1 <= len(answer) <= 400):
                    raise ValueError(f"{location}: invalid question or answer length")
                if not (20 <= len(evidence) <= 1000):
                    raise ValueError(f"{location}: invalid evidence length")
                if "\n" in question or "\n" in answer:
                    raise ValueError(f"{location}: question and answer must be one line")
                if (has_protocol_tokens(question) or has_protocol_tokens(answer)
                        or has_protocol_tokens(evidence)):
                    raise ValueError(f"{location}: protocol token")
                if CONTEXT_DEPENDENT.search(question):
                    raise ValueError(f"{location}: context-dependent question")
                if LOW_VALUE.search(question) or VOLATILE.search(
                        f"{question}\n{answer}"):
                    raise ValueError(f"{location}: volatile or low-value QA")
                support = evidence_support(answer, evidence)
                if support < 0.50:
                    raise ValueError(
                        f"{location}: answer is weakly supported by evidence ({support:.2f})")
                leak = leakage_reason(question, answer, facts)
                if leak:
                    raise ValueError(f"{location}: evaluation leakage ({leak})")
                rendered = f"Question: {question}\nAnswer: {answer}"
                if parse_qa(rendered) != (question, answer):
                    raise ValueError(f"{location}: QA does not round-trip")
                question_key = normalize_text(question)
                if question_key in questions:
                    raise ValueError(
                        f"{location}: duplicate question also at {questions[question_key]}")
                fact_id = stable_fact_id(question, answer)
                if fact_id in fact_ids:
                    raise ValueError(
                        f"{location}: duplicate fact also at {fact_ids[fact_id]}")
                questions[question_key] = location
                fact_ids[fact_id] = location
                evidence_sha256 = hashlib.sha256(evidence.encode()).hexdigest()
                output.append({
                    "answer": answer,
                    "canonical_resource_url": resources[resource_id]["canonical_url"],
                    "claim_type": claim_type,
                    "document_sha256": evidence_sha256,
                    "evidence": evidence,
                    "evidence_sha256": evidence_sha256,
                    "fact_id": fact_id,
                    "kind": "qa_resource_manual_fact",
                    "quality_schema": "manual-resource-fact-v1",
                    "question": question,
                    "resource_id": resource_id,
                    "review": {"reviewer": reviewer, "verified_at": verified_at},
                    "source": source,
                    "source_lineage": {
                        "manifest": portable_path(manifest_path),
                        "review_packet": portable_path(packet_path),
                        "review_packet_line": line_number,
                    },
                    "supplied_url": resources[resource_id]["supplied_url"],
                    "text": rendered,
                    "url": evidence_url,
                })
                counts_by_claim[claim_type] += 1
                counts_by_packet[portable_path(packet_path)] += 1
                counts_by_resource[resource_id] += 1

    output.sort(key=lambda item: (item["resource_id"],
                                  normalize_text(item["question"])))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as destination:
        for item in output:
            destination.write(json.dumps(item, ensure_ascii=False,
                                         sort_keys=True) + "\n")
    report = {
        "schema": "manual-resource-fact-report-v1",
        "manifest": portable_path(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "packets": [portable_path(path) for path in sorted(packet_paths)],
        "packet_sha256": {portable_path(path): file_sha256(path)
                          for path in sorted(packet_paths)},
        "output": portable_path(output_path),
        "output_sha256": file_sha256(output_path),
        "counts": {
            "by_claim_type": dict(sorted(counts_by_claim.items())),
            "by_packet": dict(sorted(counts_by_packet.items())),
            "by_resource": dict(sorted(counts_by_resource.items())),
            "output": len(output),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--packets", nargs="+", type=Path,
                        default=sorted(DEFAULT_PACKETS.glob("*.jsonl")))
    parser.add_argument("--eval", nargs="+", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = build(args.manifest, args.packets, args.output, args.report,
                   eval_facts(args.eval))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
