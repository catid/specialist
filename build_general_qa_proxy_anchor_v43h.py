#!/usr/bin/env python3
"""Build a train-only instruction/QA proxy from the approved prose anchor.

This builder never opens the underlying HotpotQA benchmark file or any
evaluation, OOD, heldout, or sealed artifact.  Its sole semantic input is the
already-approved, collision-screened general prose anchor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PARENT_PROSE = (ROOT / "data/general_prose_anchor_v1.jsonl").resolve()
PARENT_REPORT = (ROOT / "data/general_prose_anchor_v1.report.json").resolve()
DEFAULT_OUTPUT = (ROOT / "data/general_qa_proxy_anchor_v43h.jsonl").resolve()
DEFAULT_REPORT = (ROOT / "data/general_qa_proxy_anchor_v43h.report.json").resolve()
PARENT_PROSE_SHA256 = "a693e23c48e558e9b72c30b0ae31f0b3e580a665371846978ad4d3eca7ef5f7d"
PARENT_REPORT_SHA256 = "913ff2cb786ac50ffe86770291b6173a14220afce3682dfea67359c45cf6e9f5"
DIRECT_BENCHMARK_SOURCE = (
    ROOT / "sglang/benchmark/react/hotpotqa_100.jsonl"
).resolve()
SPACE = re.compile(r"\s+")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"\w+", re.UNICODE)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ) + "\n"
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _brief_target(text: str) -> str:
    # Repair only the source extractor's repeated-period join artifact.  Keep
    # complete leading sentences until the target contains useful context.
    text = SPACE.sub(" ", re.sub(r"\.{2,}", ".", text)).strip()
    segments = SENTENCE_BOUNDARY.split(text)
    selected = []
    words = 0
    for segment in segments:
        selected.append(segment)
        words += len(WORD.findall(segment))
        if words >= 12:
            break
    target = " ".join(selected).strip()
    if not target or len(WORD.findall(target)) < 8:
        raise ValueError("v43h prose row cannot form a factual QA proxy target")
    return target


def build_v43h() -> tuple[bytes, dict]:
    if (
        file_sha256(PARENT_PROSE) != PARENT_PROSE_SHA256
        or file_sha256(PARENT_REPORT) != PARENT_REPORT_SHA256
    ):
        raise RuntimeError("v43h approved prose parent identity changed")
    parent_report = json.loads(PARENT_REPORT.read_text(encoding="utf-8"))
    if (
        parent_report.get("schema") != "general-prose-anchor-build-v1"
        or parent_report.get("output_sha256") != PARENT_PROSE_SHA256
        or parent_report.get("output_rows") != 128
        or parent_report.get("unique_documents") != 128
        or parent_report.get("unique_texts") != 128
        or parent_report.get("policy", {}).get("excluded_sensitive_fields")
        != ["answer", "target", "thought", "question", "model_output"]
        or len(parent_report.get("protected_artifacts", [])) != 4
    ):
        raise RuntimeError("v43h parent collision-screening receipt changed")

    rows = []
    seen_items = set()
    seen_documents = set()
    for line_number, line in enumerate(
        PARENT_PROSE.read_text(encoding="utf-8").splitlines(), 1,
    ):
        if not line.strip():
            continue
        parent = json.loads(line)
        if (
            parent.get("split") != "anchor_prose"
            or parent.get("quality_bucket") != "train_only_general_prose"
            or not isinstance(parent.get("title"), str)
            or not isinstance(parent.get("text"), str)
            or not isinstance(parent.get("text_sha256"), str)
        ):
            raise RuntimeError(f"v43h invalid approved prose row {line_number}")
        document_id = str(parent["document_id"])
        if document_id in seen_documents:
            raise RuntimeError("v43h approved prose document repeated")
        seen_documents.add(document_id)
        instruction = (
            "Give a brief factual description of "
            + json.dumps(parent["title"], ensure_ascii=False)
            + "."
        )
        answer = _brief_target(parent["text"])
        identity = {
            "parent_item_id": parent["item_id"],
            "parent_text_sha256": parent["text_sha256"],
            "instruction": instruction,
            "answer": answer,
        }
        item_hash = canonical_sha256(identity)
        item_id = "general-qa-proxy-v43h-" + item_hash[:20]
        if item_id in seen_items:
            raise RuntimeError("v43h QA proxy identity repeated")
        seen_items.add(item_id)
        rows.append({
            "answer": answer,
            "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
            "document_id": document_id,
            "instruction": instruction,
            "instruction_sha256": hashlib.sha256(
                instruction.encode("utf-8")
            ).hexdigest(),
            "item_id": item_id,
            "parent_item_id": parent["item_id"],
            "parent_text_sha256": parent["text_sha256"],
            "quality_bucket": "train_only_general_knowledge_instruction_proxy",
            "split": "anchor_general_qa_proxy",
        })
    if len(rows) != 128:
        raise RuntimeError("v43h QA proxy must cover all 128 approved prose documents")
    output = b"".join(canonical_bytes(row) for row in rows)
    report = {
        "schema": "general-qa-proxy-anchor-build-v43h",
        "status": "train_only_derived_from_approved_anchor",
        "parent": {
            "prose_path": str(PARENT_PROSE),
            "prose_sha256": PARENT_PROSE_SHA256,
            "report_path": str(PARENT_REPORT),
            "report_sha256": PARENT_REPORT_SHA256,
            "protected_collision_receipt_inherited": True,
        },
        "direct_benchmark_source": {
            "path": str(DIRECT_BENCHMARK_SOURCE),
            "authorized_for_qa_semantics": False,
            "opened": False,
            "reason": (
                "repository evidence approves only the collision-screened prose "
                "derivative, not benchmark questions or answers"
            ),
        },
        "policy": {
            "semantic_inputs": [str(PARENT_PROSE)],
            "protected_eval_ood_heldout_or_benchmark_semantics_opened": False,
            "construction": (
                "fixed instruction from approved title; complete leading factual "
                "sentence target from approved prose"
            ),
            "manual_content_selection": False,
        },
        "rows": len(rows),
        "unique_documents": len({row["document_id"] for row in rows}),
        "unique_items": len({row["item_id"] for row in rows}),
        "output_sha256": hashlib.sha256(output).hexdigest(),
    }
    report["content_sha256_before_self_field"] = canonical_sha256(report)
    return output, report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output", default=str(DEFAULT_OUTPUT))
    result.add_argument("--report", default=str(DEFAULT_REPORT))
    result.add_argument("--check", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    output, report = build_v43h()
    report_bytes = json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True,
    ).encode("utf-8") + b"\n"
    if args.check:
        if output_path.read_bytes() != output or report_path.read_bytes() != report_bytes:
            raise RuntimeError("v43h QA proxy artifact changed")
        return 0
    if output_path.exists() or report_path.exists():
        raise FileExistsError("v43h QA proxy requires fresh artifact paths")
    output_path.write_bytes(output)
    report_path.write_bytes(report_bytes)
    print(json.dumps({
        "output": str(output_path),
        "output_sha256": report["output_sha256"],
        "report": str(report_path),
        "report_content_sha256": report["content_sha256_before_self_field"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
