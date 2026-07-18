#!/usr/bin/env python3
"""Deterministic requested-facet extraction and conservative answer checks.

These checks can reject or route an answer to review; they never establish
semantic correctness.  Their main purpose is to prevent a judge from silently
dropping a coordinated question facet such as the location in "when and where".
"""

from __future__ import annotations

import re
import unicodedata
from typing import Sequence


WH_PATTERNS: Sequence[tuple[str, re.Pattern[str]]] = (
    ("temporal", re.compile(r"(?i)\bwhen\b")),
    ("location", re.compile(r"(?i)\bwhere\b")),
    ("person", re.compile(r"(?i)\bwho(?:se|m)?\b")),
    ("reason", re.compile(r"(?i)\bwhy\b")),
    ("quantity", re.compile(r"(?i)\bhow\s+(?:many|much)\b")),
    ("duration", re.compile(r"(?i)\bhow\s+long\b")),
    ("frequency", re.compile(r"(?i)\bhow\s+often\b")),
    (
        "method",
        re.compile(r"(?i)\bhow\b(?!\s+(?:many|much|long|often)\b)"),
    ),
)
GENERIC_PATTERNS: Sequence[tuple[str, re.Pattern[str]]] = (
    (
        "comparison",
        re.compile(r"(?i)\b(?:compare|contrast|difference|differ|versus|vs\.?|both)\b"),
    ),
    (
        "requested_content",
        re.compile(r"(?i)\b(?:what|which|list|name|identify|describe|explain)\b"),
    ),
)

TEMPORAL_SIGNAL = re.compile(
    r"(?i)(?:\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})(?:s)?\b|"
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december|spring|summer|autumn|fall|winter|"
    r"before|after|during|since|formerly|later|earlier|century|decade|era)\b)"
)
LOCATION_SIGNAL = re.compile(
    r"(?:\b(?:in|at|near|from|within|outside|inside|across|around)\s+"
    r"(?:the\s+)?(?:[A-Z][\w'’-]+|studio\b|school\b|city\b|country\b|"
    r"region\b|venue\b|building\b|room\b|ceiling\b|tree\b))|"
    r"\b(?:located|location|venue|site|indoors|outdoors)\b"
)
PERSON_SIGNAL = re.compile(
    r"(?:\b[A-Z][\w'’-]+\s+[A-Z][\w'’-]+\b|"
    r"\b(?:by|from|named|called)\s+[A-Z][\w'’-]+\b|"
    r"(?i:\b(?:person|practitioner|artist|teacher|founder|author|creator|rigger)\b))"
)
REASON_SIGNAL = re.compile(
    r"(?i)\b(?:because|since|therefore|reason|purpose|so that|in order to|"
    r"due to|owing to|to prevent|to avoid|to ensure)\b"
)
METHOD_SIGNAL = re.compile(
    r"(?i)\b(?:by\s+\w+ing|using|first|next|then|finally|step|method|"
    r"tie|tied|wrap|wrapped|pass|passed|route|routed|attach|attached|"
    r"secure|secured|form|formed|position|placed|apply|applied)\b"
)
QUANTITY_SIGNAL = re.compile(
    r"(?i)(?:\b\d+(?:\.\d+)?\b|\b(?:one|two|three|four|five|six|seven|"
    r"eight|nine|ten|several|multiple|few|many)\b)"
)
DURATION_SIGNAL = re.compile(
    r"(?i)(?:\b\d+(?:\.\d+)?\s*(?:seconds?|minutes?|hours?|days?|weeks?|"
    r"months?|years?)\b|\b(?:briefly|temporarily|permanently)\b)"
)
FREQUENCY_SIGNAL = re.compile(
    r"(?i)(?:\b(?:always|never|often|rarely|daily|weekly|monthly|annually|"
    r"each time|every time)\b|\b\d+\s+times?\b)"
)
COMPARISON_SIGNAL = re.compile(
    r"(?i)\b(?:whereas|while|unlike|both|difference|differ|compared|contrast|"
    r"respectively|on the other hand)\b"
)


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def extract_requested_facets(question: str) -> list[dict]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonempty text")
    normalized = normalize_text(question)
    matches: list[tuple[int, str, str]] = []
    occupied_how_spans: list[tuple[int, int]] = []
    for kind, pattern in WH_PATTERNS:
        for match in pattern.finditer(normalized):
            if kind in {"quantity", "duration", "frequency"}:
                occupied_how_spans.append(match.span())
            if kind == "method" and any(
                start <= match.start() < end for start, end in occupied_how_spans
            ):
                continue
            matches.append((match.start(), kind, match.group(0)))
    for kind, pattern in GENERIC_PATTERNS:
        match = pattern.search(normalized)
        if match:
            matches.append((match.start(), kind, match.group(0)))

    # A generic "what/which/explain" facet adds no information when a more
    # specific wh-facet is already present (for example, "what and why" is kept,
    # but "explain how" should remain a method facet only).
    specific = {kind for _, kind, _ in matches if kind != "requested_content"}
    if specific:
        matches = [
            item
            for item in matches
            if item[1] != "requested_content"
            or any(token in normalized.casefold() for token in ("what and", "and what"))
        ]
    if not matches:
        matches = [(0, "primary_answer", normalized[:80])]

    deduplicated = []
    seen = set()
    for position, kind, span in sorted(matches):
        if kind in seen:
            continue
        seen.add(kind)
        deduplicated.append(
            {
                "facet_id": f"facet_{len(deduplicated):02d}_{kind}",
                "kind": kind,
                "question_span": span,
            }
        )
    return deduplicated


def deterministic_facet_signals(question: str, answer: str) -> list[dict]:
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be nonempty text")
    patterns = {
        "temporal": TEMPORAL_SIGNAL,
        "location": LOCATION_SIGNAL,
        "person": PERSON_SIGNAL,
        "reason": REASON_SIGNAL,
        "quantity": QUANTITY_SIGNAL,
        "duration": DURATION_SIGNAL,
        "frequency": FREQUENCY_SIGNAL,
        "method": METHOD_SIGNAL,
        "comparison": COMPARISON_SIGNAL,
    }
    results = []
    for facet in extract_requested_facets(question):
        pattern = patterns.get(facet["kind"])
        if pattern is None:
            status = "unresolved"
            signal = None
        else:
            match = pattern.search(answer)
            if (
                match is None
                and facet["kind"] == "location"
                and re.fullmatch(r"\s*[A-Z][\w'’-]+[.!]?\s*", answer)
            ):
                match = re.search(r"[A-Z][\w'’-]+", answer)
            status = "present" if match else "missing"
            signal = match.group(0) if match else None
        results.append(
            {
                **facet,
                "deterministic_status": status,
                "answer_signal": signal,
            }
        )
    return results


def missing_high_confidence_facets(question: str, answer: str) -> list[str]:
    signals = deterministic_facet_signals(question, answer)
    # Absence of a surface cue is decisive only for coordinated temporal/place
    # questions.  Other wh-facets still fix the judge's mapping schema, but a
    # heuristic miss routes to review rather than becoming an automatic reject.
    decisive_kinds = {"temporal", "location"} if len(signals) > 1 else set()
    return [
        item["facet_id"]
        for item in signals
        if item["kind"] in decisive_kinds
        and item["deterministic_status"] == "missing"
    ]
