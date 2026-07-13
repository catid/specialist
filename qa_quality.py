"""Deterministic QA parsing, normalization, and split-leakage checks.

The checks intentionally combine question similarity with answer identity.
Question-only lexical thresholds miss paraphrases such as "When did X die?"
versus "In what year did X pass away?"; answer-only matching over-filters common
answers.  Requiring an entity/context overlap and the same answer catches the
former without treating every shared "yes" or year as the same fact.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, Optional


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "into", "is", "it", "of", "on", "or",
    "that", "the", "their", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "why", "with", "would", "year",
}

LOW_VALUE = re.compile(
    r"\b(?:coupon|discount|promo(?:tional)? code|price|cost|shipping|add to cart|"
    r"sold out|upcoming|currently available|this (?:year|month|week)|next "
    r"(?:year|month|week)|event schedule|registration deadline)\b",
    re.IGNORECASE,
)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[^\W_]+", value, flags=re.UNICODE))


def tokens(value: str) -> set[str]:
    normalized = normalize_text(value)
    return set(normalized.split()) if normalized else set()


def content_tokens(value: str) -> set[str]:
    return {token for token in tokens(value) if token not in STOPWORDS}


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(len(left | right), 1)


def containment(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(min(len(left), len(right)), 1)


def answers_equivalent(left: str, right: str) -> bool:
    nl, nr = normalize_text(left), normalize_text(right)
    if not nl or not nr:
        return False
    if nl == nr:
        return True
    lt, rt = tokens(left), tokens(right)
    # Multi-token aliases often add a title or parenthetical clarification.
    return (min(len(nl), len(nr)) >= 5 and (nl in nr or nr in nl)
            and containment(lt, rt) >= 0.75)


PROTOCOL_TOKENS = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")

_CHAT_QA = re.compile(
    r"\A<\|im_start\|>user\n(?P<user>.*?)<\|im_end\|>\n"
    r"<\|im_start\|>assistant\n(?P<assistant>.*?)<\|im_end\|>\s*\Z",
    re.DOTALL,
)
_LABELED_QA = (
    re.compile(
        r"\AQuestion:[ \t]*(?P<question>.+?)\n"
        r"Answer:[ \t]*(?P<answer>.+?)\s*\Z",
        re.DOTALL,
    ),
    re.compile(
        r"\AQ:[ \t]*(?P<question>.+?)\n"
        r"A:[ \t]*(?P<answer>.+?)\s*\Z",
        re.DOTALL,
    ),
)
_INSTRUCTION_QA = re.compile(
    r"\AAnswer this question[^\n]*:\n\n"
    r"(?P<question>.+?)\n\n(?P<answer>.+?)\s*\Z",
    re.DOTALL,
)
_LEADING_THINK = re.compile(
    r"\A<think>(?P<reasoning>.*?)</think>\s*(?P<answer>.+)\Z",
    re.DOTALL,
)
_SECOND_QA = re.compile(r"(?m)^\s*(?:Question|Q):")


def has_protocol_tokens(value: str) -> bool:
    return any(token in value for token in PROTOCOL_TOKENS)


def _clean_pair(question: str, answer: str) -> Optional[tuple[str, str]]:
    question, answer = question.strip(), answer.strip()
    if not question or not answer:
        return None
    # Parsed metadata must never contain serialization/control syntax. Fail
    # closed so a future template change cannot silently poison supervision.
    if has_protocol_tokens(question) or has_protocol_tokens(answer):
        return None
    if _SECOND_QA.search(answer):
        return None
    return question, answer


def parse_qa(text: str) -> Optional[tuple[str, str]]:
    """Parse supported QA serializations without crossing role boundaries."""
    text = text.strip().replace("\r\n", "\n").replace("\r", "\n")
    match = _CHAT_QA.fullmatch(text)
    if match:
        user = match.group("user").strip()
        instruction, separator, question = user.partition("\n\n")
        if (not separator or
                not re.fullmatch(r"Answer this question[^\n]*:", instruction)):
            return None
        assistant = match.group("assistant").strip()
        if assistant.startswith("<think>"):
            thought = _LEADING_THINK.fullmatch(assistant)
            if thought is None:
                return None
            answer = thought.group("answer")
        else:
            if "<think>" in assistant or "</think>" in assistant:
                return None
            answer = assistant
        return _clean_pair(question, answer)

    # Any ChatML sentinel commits the record to the strict grammar above; a
    # malformed chat record must never fall through to a broad plain parser.
    if "<|im_" in text or "<think>" in text or "</think>" in text:
        return None

    for pattern in (*_LABELED_QA, _INSTRUCTION_QA):
        match = pattern.fullmatch(text)
        if match:
            return _clean_pair(match.group("question"), match.group("answer"))
    return None


def qa_pair_from_record(item: dict) -> Optional[tuple[str, str]]:
    """Return a clean pair and reject disagreement between metadata and text."""
    parsed = parse_qa(item.get("text", ""))
    has_explicit = "question" in item or "answer" in item
    if not has_explicit:
        return parsed
    explicit = _clean_pair(item.get("question", ""), item.get("answer", ""))
    if explicit is None:
        raise ValueError("question/answer metadata contains protocol tokens or is empty")
    if parsed is None:
        raise ValueError("text field is not a supported QA serialization")
    if tuple(map(normalize_text, explicit)) != tuple(map(normalize_text, parsed)):
        raise ValueError("question/answer metadata disagrees with parsed text")
    return explicit


@dataclass(frozen=True)
class EvalFact:
    question: str
    answer: str
    item_id: str = ""
    split: str = ""

    # Leakage checks compare every training candidate with the same evaluation
    # facts. Cache normalized forms once instead of repeating Unicode/regex
    # work tens of millions of times for large dataset builds.
    @cached_property
    def normalized_question(self) -> str:
        return normalize_text(self.question)

    @cached_property
    def question_tokens(self) -> frozenset[str]:
        return frozenset(tokens(self.question))

    @cached_property
    def question_content_tokens(self) -> frozenset[str]:
        return frozenset(content_tokens(self.question))

    @cached_property
    def normalized_answer(self) -> str:
        return normalize_text(self.answer)

    @cached_property
    def answer_tokens(self) -> frozenset[str]:
        return frozenset(tokens(self.answer))


def _prepared_answers_equivalent(
    left_normalized: str,
    left_tokens: set[str],
    right_normalized: str,
    right_tokens: set[str],
) -> bool:
    if not left_normalized or not right_normalized:
        return False
    if left_normalized == right_normalized:
        return True
    return (
        min(len(left_normalized), len(right_normalized)) >= 5
        and (left_normalized in right_normalized
             or right_normalized in left_normalized)
        and containment(left_tokens, right_tokens) >= 0.75
    )


def leakage_reason(question: str, answer: str,
                   eval_facts: Iterable[EvalFact]) -> Optional[str]:
    q_norm = normalize_text(question)
    q_tokens = tokens(question)
    q_content = content_tokens(question)
    answer_norm = normalize_text(answer)
    answer_token_set = tokens(answer)
    for fact in eval_facts:
        other_norm = fact.normalized_question
        other_tokens = fact.question_tokens
        if q_norm == other_norm:
            return "exact_question"
        if jaccard(q_tokens, other_tokens) >= 0.60:
            return "near_duplicate_question"
        if not _prepared_answers_equivalent(
            answer_norm,
            answer_token_set,
            fact.normalized_answer,
            fact.answer_tokens,
        ):
            continue
        other_content = fact.question_content_tokens
        shared = q_content & other_content
        # Same answer + overlapping named entity/relation context is a fact
        # collision even when surface wording is quite different.
        if (jaccard(q_content, other_content) >= 0.25 or
                (len(shared) >= 2 and containment(q_content, other_content) >= 0.40)):
            return "entity_answer_fact"
    return None


def stable_fact_id(question: str, answer: str, source: str = "") -> str:
    material = "\0".join((normalize_text(question), normalize_text(answer), source))
    return "fact-" + hashlib.sha256(material.encode()).hexdigest()[:20]


def document_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
