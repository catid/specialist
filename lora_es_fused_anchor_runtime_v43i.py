#!/usr/bin/env python3
"""CPU-side fused anchor preparation, scoring, calibration, and gating for V43I."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path


PROSE_SHA256_V43I = "a693e23c48e558e9b72c30b0ae31f0b3e580a665371846978ad4d3eca7ef5f7d"
PROSE_REPORT_SHA256_V43I = "913ff2cb786ac50ffe86770291b6173a14220afce3682dfea67359c45cf6e9f5"
QA_PROXY_SHA256_V43I = "d250980c5b88308452aba4ee8d3e43090b1444c250547c2515c838593f2f391f"
QA_PROXY_REPORT_SHA256_V43I = "b176f4ea0d1d4b466c83b68c243bfb4a9e315df8574d9ba2c606333fbfc4ee8f"
PANEL_SEED_V43I = "v43h-fixed-multi-anchor-panel-20260715"
PANEL_SIZE_V43I = 32
FULL_SIZE_V43I = 128
MAX_PROSE_TOKENS_V43I = 1024
MAX_QA_TOKENS_V43I = 1024
MAX_GENERATION_TOKENS_V43I = 64
LOGPROB_MARGIN_CEILING_V43I = 0.001802103667415178
GENERATION_F1_MARGIN_CEILING_V43I = 0.01
GENERATION_COUNT_MARGIN_CEILING_V43I = 1
CALIBRATION_WARMUPS_V43I = 2
CALIBRATION_REPEATS_V43I = 8
_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
_TOKEN = re.compile(r"[a-z0-9]+")


def file_sha256_v43i(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256_v43i(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"v43i {path} line {line_number} is not an object")
        rows.append(row)
    return rows


def load_anchor_bundle_v43i(
    prose_path: Path,
    prose_report_path: Path,
    qa_path: Path,
    qa_report_path: Path,
) -> dict:
    """Load only sealed train-anchor artifacts; direct benchmark paths are absent."""
    paths = [Path(item).resolve() for item in (
        prose_path, prose_report_path, qa_path, qa_report_path,
    )]
    forbidden = ("eval_qa", "ood_qa", "ood_prose", "heldout", "holdout")
    if any(any(token in str(path).lower() for token in forbidden) for path in paths):
        raise ValueError("v43i anchor loader rejects protected paths")
    expected = (
        PROSE_SHA256_V43I, PROSE_REPORT_SHA256_V43I,
        QA_PROXY_SHA256_V43I, QA_PROXY_REPORT_SHA256_V43I,
    )
    if tuple(file_sha256_v43i(path) for path in paths) != expected:
        raise RuntimeError("v43i train-anchor artifact identity changed")
    prose_report = json.loads(paths[1].read_text(encoding="utf-8"))
    qa_report = json.loads(paths[3].read_text(encoding="utf-8"))
    if (
        prose_report.get("schema") != "general-prose-anchor-build-v1"
        or prose_report.get("output_sha256") != PROSE_SHA256_V43I
        or qa_report.get("schema") != "general-qa-proxy-anchor-build-v43h"
        or qa_report.get("output_sha256") != QA_PROXY_SHA256_V43I
        or qa_report.get("direct_benchmark_source", {}).get("opened") is not False
        or qa_report.get("direct_benchmark_source", {}).get(
            "authorized_for_qa_semantics"
        ) is not False
    ):
        raise RuntimeError("v43i train-anchor firewall receipt changed")
    prose_rows = _load_jsonl(paths[0])
    qa_rows = _load_jsonl(paths[2])
    prose = {row.get("document_id"): row for row in prose_rows}
    qa = {row.get("document_id"): row for row in qa_rows}
    if (
        len(prose_rows) != FULL_SIZE_V43I
        or len(qa_rows) != FULL_SIZE_V43I
        or len(prose) != FULL_SIZE_V43I
        or set(prose) != set(qa)
        or any(row.get("split") != "anchor_prose" for row in prose_rows)
        or any(row.get("split") != "anchor_general_qa_proxy" for row in qa_rows)
    ):
        raise RuntimeError("v43i aligned train-anchor inventory changed")
    ordered = sorted(qa_rows, key=lambda row: hashlib.sha256(
        (PANEL_SEED_V43I + "\0" + row["item_id"]).encode("utf-8")
    ).digest())
    panel_documents = [row["document_id"] for row in ordered[:PANEL_SIZE_V43I]]
    full_documents = [row["document_id"] for row in qa_rows]

    def materialize(documents: list[str]) -> list[dict]:
        return [{"prose": prose[document], "qa": qa[document]} for document in documents]

    return {
        "schema": "aligned-train-only-multi-anchor-bundle-v43i",
        "paths": {
            "prose": str(paths[0]), "prose_report": str(paths[1]),
            "qa": str(paths[2]), "qa_report": str(paths[3]),
        },
        "panel": materialize(panel_documents),
        "full": materialize(full_documents),
        "panel_document_identity_sha256": canonical_sha256_v43i(panel_documents),
        "full_document_identity_sha256": canonical_sha256_v43i(full_documents),
        "direct_benchmark_source_opened": False,
        "protected_semantics_opened": False,
    }


def _encode(tokenizer, text: str, maximum: int, label: str) -> list[int]:
    values = list(tokenizer.encode(text, add_special_tokens=False))
    if len(values) < 2 or len(values) > maximum or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    ):
        raise ValueError(f"v43i {label} tokenization violates its sealed bounds")
    return values


def prepare_anchor_items_v43i(
    rows: list[dict], tokenizer, specialist_template, dense_anchor_module,
) -> dict:
    prose_items = []
    qa_prompts, qa_answers = [], []
    qa_generation = []
    for index, item in enumerate(rows):
        prose = item["prose"]
        qa = item["qa"]
        if prose["document_id"] != qa["document_id"]:
            raise RuntimeError("v43i aligned anchor document changed")
        prose_ids = _encode(
            tokenizer, prose["text"], MAX_PROSE_TOKENS_V43I, "prose",
        )
        prose_items.append({
            "index": index,
            "document_id": prose["document_id"],
            "prompt_token_ids": prose_ids,
            "token_ids_sha256": canonical_sha256_v43i(prose_ids),
        })
        prompt = specialist_template(qa["instruction"])
        qa_prompts.append(prompt)
        qa_answers.append(qa["answer"])
        prompt_ids = _encode(
            tokenizer, prompt, MAX_QA_TOKENS_V43I, "QA generation prompt",
        )
        qa_generation.append({
            "index": index,
            "document_id": qa["document_id"],
            "answer": qa["answer"],
            "answer_sha256": qa["answer_sha256"],
            "prompt_token_ids": prompt_ids,
            "prompt_token_ids_sha256": canonical_sha256_v43i(prompt_ids),
        })
    qa_teacher = dense_anchor_module.prepare_gold_answer_items_v4(
        tokenizer, qa_prompts, qa_answers,
    )
    if len(qa_teacher) != len(rows):
        raise RuntimeError("v43i QA teacher preparation changed coverage")
    return {
        "schema": "prepared-fused-train-anchor-items-v43i",
        "prose": prose_items,
        "qa_teacher": qa_teacher,
        "qa_generation": qa_generation,
        "documents": len(rows),
    }


def fused_requests_v43i(domain_requests: list[dict], anchors: dict) -> dict:
    requests = list(domain_requests)
    slices = {}

    def append(label: str, items: list[dict]) -> None:
        start = len(requests)
        requests.extend({"prompt_token_ids": item["prompt_token_ids"]} for item in items)
        slices[label] = [start, len(requests)]

    slices["domain"] = [0, len(requests)]
    append("prose", anchors["prose"])
    append("qa_teacher", anchors["qa_teacher"])
    append("qa_generation", anchors["qa_generation"])
    if len(requests) != len(domain_requests) + 3 * anchors["documents"]:
        raise RuntimeError("v43i fused request coverage changed")
    return {"requests": requests, "slices": slices}


def anchor_only_requests_v43i(anchors: dict) -> dict:
    return fused_requests_v43i([], anchors)


def _selected_prompt_logprobs(output, expected_ids: list[int]) -> list[float]:
    returned = list(getattr(output, "prompt_token_ids", None) or [])
    prompt_logprobs = getattr(output, "prompt_logprobs", None)
    if returned != expected_ids or prompt_logprobs is None or len(prompt_logprobs) != len(expected_ids):
        raise ValueError("v43i prose prompt-logprob alignment changed")
    result = []
    for position, token_id in enumerate(expected_ids[1:], 1):
        candidates = prompt_logprobs[position]
        if candidates is None or token_id not in candidates:
            raise ValueError("v43i prose selected token logprob is absent")
        selected = candidates[token_id]
        value = float(selected.logprob if hasattr(selected, "logprob") else selected["logprob"])
        if not math.isfinite(value):
            raise ValueError("v43i prose logprob is non-finite")
        result.append(value)
    return result


def score_prose_v43i(items: list[dict], outputs: list) -> dict:
    if len(items) != len(outputs) or not items:
        raise ValueError("v43i prose output coverage changed")
    documents, all_values = [], []
    for item, output in zip(items, outputs, strict=True):
        values = _selected_prompt_logprobs(output, item["prompt_token_ids"])
        all_values.extend(values)
        documents.append({
            "document_id": item["document_id"],
            "scored_token_count": len(values),
            "sum_token_logprob": math.fsum(values),
        })
    return {
        "mean_token_logprob": math.fsum(all_values) / len(all_values),
        "scored_token_count": len(all_values),
        "document_numeric_sha256": canonical_sha256_v43i(documents),
    }


def _extract_answer(text: str) -> str:
    text = _THINK.sub(" ", text).replace("<|im_end|>", " ").strip()
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


def _f1(prediction: str, reference: str) -> float:
    from collections import Counter
    predicted, expected = _tokens(prediction), _tokens(reference)
    if not predicted or not expected:
        return 0.0
    common = sum((Counter(predicted) & Counter(expected)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(predicted), common / len(expected)
    return 2.0 * precision * recall / (precision + recall)


def score_qa_generation_v43i(items: list[dict], outputs: list) -> dict:
    if len(items) != len(outputs) or not items:
        raise ValueError("v43i QA generation output coverage changed")
    rows = []
    for item, output in zip(items, outputs, strict=True):
        generated = getattr(output, "outputs", None)
        if not isinstance(generated, list) or len(generated) != 1:
            raise ValueError("v43i QA generation completion count changed")
        prediction = _extract_answer(str(generated[0].text))
        f1 = _f1(prediction, item["answer"])
        exact = int(_tokens(prediction) == _tokens(item["answer"]) and bool(_tokens(item["answer"])))
        rows.append({
            "document_id": item["document_id"],
            "prediction_sha256": hashlib.sha256(prediction.encode("utf-8")).hexdigest(),
            "f1": f1,
            "exact": exact,
            "nonzero": int(f1 > 0.0),
        })
    return {
        "mean_f1": math.fsum(item["f1"] for item in rows) / len(rows),
        "exact_count": sum(item["exact"] for item in rows),
        "nonzero_count": sum(item["nonzero"] for item in rows),
        "item_numeric_sha256": canonical_sha256_v43i(rows),
    }


def score_fused_outputs_v43i(
    plan: dict,
    outputs: list,
    anchors: dict,
    dense_anchor_module,
    *,
    domain_scorer=None,
) -> dict:
    if len(outputs) != len(plan["requests"]):
        raise RuntimeError("v43i fused output count changed")
    slices = plan["slices"]

    def take(label: str) -> list:
        start, stop = slices[label]
        return outputs[start:stop]

    domain = take("domain")
    if domain_scorer is None and domain:
        raise ValueError("v43i domain outputs require a scorer")
    qa_dense = dense_anchor_module.score_gold_answer_outputs_v4(
        anchors["qa_teacher"], take("qa_teacher"),
    )
    result = {
        "schema": "fused-train-anchor-score-v43i",
        "prose_lm": score_prose_v43i(anchors["prose"], take("prose")),
        "qa_answer_logprob": {
            "mean_example_logprob": qa_dense["mean_example_mean_logprob"],
            "scored_answer_tokens": qa_dense["answer_token_count"],
            "dense_result_sha256": canonical_sha256_v43i(qa_dense),
        },
        "qa_generation": score_qa_generation_v43i(
            anchors["qa_generation"], take("qa_generation"),
        ),
    }
    if domain:
        result["domain"] = domain_scorer(domain)
    return result


def calibration_margins_v43i(records: list[dict]) -> dict:
    """Fit fixed-state inference margins before any population is opened."""
    if len(records) != CALIBRATION_REPEATS_V43I or any(len(row) != 4 for row in records):
        raise ValueError("v43i anchor calibration requires eight four-actor repeats")
    paths = {
        "prose_lm": ("prose_lm", "mean_token_logprob"),
        "qa_answer_logprob": ("qa_answer_logprob", "mean_example_logprob"),
        "qa_generation_f1": ("qa_generation", "mean_f1"),
        "qa_generation_exact": ("qa_generation", "exact_count"),
        "qa_generation_nonzero": ("qa_generation", "nonzero_count"),
    }
    margins = {}
    for name, path in paths.items():
        values = [[float(actor[path[0]][path[1]]) for actor in repeat] for repeat in records]
        flattened = [value for repeat in values for value in repeat]
        margin = max(flattened) - min(flattened)
        margins[name] = {
            "margin": margin,
            "minimum": min(flattened),
            "maximum": max(flattened),
            "repeat_actor_ranges": [max(row) - min(row) for row in values],
        }
    passed = bool(
        margins["prose_lm"]["margin"] <= LOGPROB_MARGIN_CEILING_V43I
        and margins["qa_answer_logprob"]["margin"] <= LOGPROB_MARGIN_CEILING_V43I
        and margins["qa_generation_f1"]["margin"] <= GENERATION_F1_MARGIN_CEILING_V43I
        and margins["qa_generation_exact"]["margin"] <= GENERATION_COUNT_MARGIN_CEILING_V43I
        and margins["qa_generation_nonzero"]["margin"] <= GENERATION_COUNT_MARGIN_CEILING_V43I
    )
    result = {
        "schema": "fused-anchor-fixed-state-calibration-v43i",
        "records": CALIBRATION_REPEATS_V43I,
        "actors": 4,
        "margins": margins,
        "ceilings": {
            "logprob": LOGPROB_MARGIN_CEILING_V43I,
            "generation_f1": GENERATION_F1_MARGIN_CEILING_V43I,
            "generation_count": GENERATION_COUNT_MARGIN_CEILING_V43I,
        },
        "passed": passed,
    }
    result["content_sha256"] = canonical_sha256_v43i(result)
    return result

def _actor_metric(actor: dict, metric: str) -> float:
    paths = {
        "domain": ("domain", "aggregate", "equal_unit_mean"),
        "prose_lm": ("prose_lm", "mean_token_logprob"),
        "qa_answer_logprob": ("qa_answer_logprob", "mean_example_logprob"),
        "qa_generation_f1": ("qa_generation", "mean_f1"),
        "qa_generation_exact": ("qa_generation", "exact_count"),
        "qa_generation_nonzero": ("qa_generation", "nonzero_count"),
    }
    value = actor
    for key in paths[metric]:
        value = value[key]
    return float(value)


def candidate_gate_v43i(
    reference_actors: list[dict],
    candidate_actors: list[dict],
    calibration: dict,
) -> dict:
    if len(reference_actors) != 4 or len(candidate_actors) != 4 or calibration.get("passed") is not True:
        raise ValueError("v43i candidate gate inputs are incomplete")
    metrics = {}
    for metric in (
        "domain", "prose_lm", "qa_answer_logprob", "qa_generation_f1",
        "qa_generation_exact", "qa_generation_nonzero",
    ):
        before = [_actor_metric(actor, metric) for actor in reference_actors]
        after = [_actor_metric(actor, metric) for actor in candidate_actors]
        deltas = [right - left for left, right in zip(before, after, strict=True)]
        metrics[metric] = {
            "reference_actor_values": before,
            "candidate_actor_values": after,
            "paired_actor_deltas": deltas,
            "median_paired_delta": statistics.median(deltas),
        }
    margins = calibration["margins"]
    checks = {
        "domain_point_improvement": metrics["domain"]["median_paired_delta"] > 0.0,
        "prose_lm_noninferiority": metrics["prose_lm"]["median_paired_delta"]
        >= -margins["prose_lm"]["margin"],
        "qa_logprob_noninferiority": metrics["qa_answer_logprob"]["median_paired_delta"]
        >= -margins["qa_answer_logprob"]["margin"],
        "qa_generation_f1_noninferiority": metrics["qa_generation_f1"]["median_paired_delta"]
        >= -margins["qa_generation_f1"]["margin"],
        "qa_generation_exact_noninferiority": metrics["qa_generation_exact"]["median_paired_delta"]
        >= -margins["qa_generation_exact"]["margin"],
        "qa_generation_nonzero_noninferiority": metrics["qa_generation_nonzero"]["median_paired_delta"]
        >= -margins["qa_generation_nonzero"]["margin"],
    }
    result = {
        "schema": "uncommitted-fused-anchor-candidate-gate-v43i",
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }
    result["content_sha256"] = canonical_sha256_v43i(result)
    return result
