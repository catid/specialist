#!/usr/bin/env python3
"""Merge four base-model response shards into an aggregate-only baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import build_development_baseline_evaluation_v1 as protocol
import run_development_response_shard_v1 as runner


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[\w]+", value, flags=re.UNICODE))


def token_f1(observed: str, reference: str) -> float:
    observed_tokens = normalize_text(observed).split()
    reference_tokens = normalize_text(reference).split()
    if not observed_tokens and not reference_tokens:
        return 1.0
    if not observed_tokens or not reference_tokens:
        return 0.0
    overlap = sum((Counter(observed_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(observed_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _strip_code_fence(value: str) -> str:
    stripped = value.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else stripped


def parse_json_object(value: str) -> dict | None:
    try:
        parsed = json.loads(_strip_code_fence(value))
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _claim_units(value: str) -> set[str]:
    units = set(re.findall(r"\b\d+(?:\.\d+)?\b", value))
    units.update(re.findall(r"\b[A-Z][\w-]{2,}\b", value))
    units.update(re.findall(r"\b[\w]+-[\w-]+\b", value))
    return {normalize_text(unit) for unit in units if normalize_text(unit)}


def unsupported_claim_unit_rate(answer: str, evidence: str, reference: str) -> float:
    units = _claim_units(answer)
    if not units:
        return 0.0
    support = normalize_text(evidence + " " + reference)
    unsupported = sum(unit not in support for unit in units)
    return unsupported / len(units)


def _valid_confidence(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) and 0.0 <= numeric <= 1.0 else None


def score_domain_response(item: dict, response: str, source_row: dict) -> dict:
    parsed = parse_json_object(response)
    json_valid = parsed is not None
    answer = parsed.get("answer") if parsed is not None else None
    answer = answer if isinstance(answer, str) else ""
    exact = normalize_text(answer) == normalize_text(source_row["answer"])
    f1 = token_f1(answer, source_row["answer"])
    correct = f1 >= 0.65
    confidence = _valid_confidence(parsed.get("confidence")) if parsed else None
    brier = (
        (confidence - float(correct)) ** 2
        if confidence is not None
        else 1.0
    )
    result = {
        "normalized_exact": float(exact),
        "token_f1": f1,
        "correct_at_token_f1_0_65": float(correct),
        "json_valid": float(json_valid),
        "confidence_valid": float(confidence is not None),
        "confidence_brier": brier,
        "grounded_exact_support": None,
        "unsupported_claim_unit_rate": None,
    }
    if item["form"] == "grounded":
        support = parsed.get("support") if parsed else None
        support_valid = isinstance(support, str) and bool(support) and support in source_row[
            "evidence"
        ]
        result["grounded_exact_support"] = float(support_valid)
        result["unsupported_claim_unit_rate"] = unsupported_claim_unit_rate(
            answer,
            source_row["evidence"],
            source_row["answer"],
        )
    return result


def score_domain_extension_response(
    item: dict,
    response: str,
    source_rows: list[dict],
) -> dict:
    parsed = parse_json_object(response)
    confidence = _valid_confidence(parsed.get("confidence")) if parsed else None
    extension_type = item["extension_type"]
    component_scores = []
    unanswerable_false_answer = None
    if extension_type in {"paraphrase_recall", "application"}:
        answer = parsed.get("answer") if parsed else None
        f1 = token_f1(answer if isinstance(answer, str) else "", source_rows[0]["answer"])
        component_scores.append(float(f1 >= 0.65))
    elif extension_type == "synthesis":
        if len(source_rows) != 2:
            raise RuntimeError("sealed synthesis source count changed")
        answer_a = parsed.get("answer_a") if parsed else None
        answer_b = parsed.get("answer_b") if parsed else None
        f1_a = token_f1(
            answer_a if isinstance(answer_a, str) else "",
            source_rows[0]["answer"],
        )
        f1_b = token_f1(
            answer_b if isinstance(answer_b, str) else "",
            source_rows[1]["answer"],
        )
        component_scores.extend([float(f1_a >= 0.65), float(f1_b >= 0.65)])
    elif extension_type == "contradiction":
        verdict = parsed.get("verdict") if parsed else None
        corrected = parsed.get("corrected_answer") if parsed else None
        f1 = token_f1(
            corrected if isinstance(corrected, str) else "",
            source_rows[0]["answer"],
        )
        component_scores.extend([
            float(verdict == "not_supported"),
            float(f1 >= 0.65),
        ])
    elif extension_type == "unanswerable":
        answer = parsed.get("answer") if parsed else None
        passed = normalize_text(answer if isinstance(answer, str) else "") == (
            normalize_text("INSUFFICIENT_INFORMATION")
        )
        component_scores.append(float(passed))
        unanswerable_false_answer = float(not passed)
    elif extension_type == "false_premise":
        verdict = parsed.get("verdict") if parsed else None
        answer = parsed.get("answer") if parsed else None
        f1 = token_f1(
            answer if isinstance(answer, str) else "",
            source_rows[0]["answer"],
        )
        component_scores.extend([
            float(verdict == "false_premise"),
            float(f1 >= 0.65),
        ])
    else:
        raise RuntimeError(f"unsupported sealed domain extension: {extension_type}")
    extension_score = _mean(component_scores)
    brier = (
        (confidence - extension_score) ** 2
        if confidence is not None
        else 1.0
    )
    return {
        "extension_score": extension_score,
        "json_valid": float(parsed is not None),
        "confidence_valid": float(confidence is not None),
        "confidence_brier": brier,
        "unanswerable_false_answer": unanswerable_false_answer,
    }


def score_general_response(fixture: dict, response: str) -> dict:
    if fixture["scoring_status"] != "deterministic":
        return {
            "deterministic_pass": None,
            "format_valid": float(bool(response.strip())),
            "tool_call_valid": None,
            "judge_status": "gated_not_scored",
        }
    reference = fixture["reference"]
    verifier = reference["verifier"]
    expected = reference["expected"]
    observed = response.strip()
    format_valid = True
    if verifier == "exact":
        passed = observed == str(expected)
    elif verifier == "one_of":
        passed = observed in expected
    elif verifier == "json_exact":
        parsed = parse_json_object(response)
        format_valid = parsed is not None
        passed = parsed == expected
    else:
        raise RuntimeError(f"unsupported deterministic verifier: {verifier}")
    is_tool = fixture["category"] == "tool_function_calling"
    return {
        "deterministic_pass": float(passed),
        "format_valid": float(format_valid),
        "tool_call_valid": float(passed) if is_tool else None,
        "judge_status": None,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile of empty values")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def grouped_bootstrap_mean_ci(
    records: list[dict],
    *,
    value_key: str,
    group_key: str,
    seed: int = protocol.BOOTSTRAP_SEED,
    replicates: int = protocol.BOOTSTRAP_REPLICATES,
) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = record[value_key]
        if value is not None:
            groups[record[group_key]].append(float(value))
    names = sorted(groups)
    if not names:
        raise RuntimeError("bootstrap has no scored groups")
    point_values = [value for name in names for value in groups[name]]
    generator = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sampled = [names[generator.randrange(len(names))] for _ in names]
        values = [value for name in sampled for value in groups[name]]
        draws.append(_mean(values))
    draws.sort()
    return {
        "mean": _mean(point_values),
        "lower_95": _percentile(draws, 0.025),
        "upper_95": _percentile(draws, 0.975),
        "groups": len(names),
        "observations": len(point_values),
        "replicates": replicates,
        "seed": seed,
    }


def paired_grouped_bootstrap_difference_ci(
    base_records: list[dict],
    candidate_records: list[dict],
    *,
    item_key: str,
    value_key: str,
    group_key: str,
    seed: int = protocol.BOOTSTRAP_SEED,
    replicates: int = protocol.BOOTSTRAP_REPLICATES,
) -> dict:
    base = {record[item_key]: record for record in base_records}
    candidate = {record[item_key]: record for record in candidate_records}
    if len(base) != len(base_records) or len(candidate) != len(candidate_records):
        raise RuntimeError("paired bootstrap item identifiers are duplicated")
    if base.keys() != candidate.keys():
        raise RuntimeError("paired bootstrap base and candidate item sets differ")
    differences = []
    for item_id in sorted(base):
        base_record = base[item_id]
        candidate_record = candidate[item_id]
        if base_record[group_key] != candidate_record[group_key]:
            raise RuntimeError("paired bootstrap source group changed")
        differences.append({
            item_key: item_id,
            group_key: base_record[group_key],
            "paired_difference": float(candidate_record[value_key])
            - float(base_record[value_key]),
        })
    result = grouped_bootstrap_mean_ci(
        differences,
        value_key="paired_difference",
        group_key=group_key,
        seed=seed,
        replicates=replicates,
    )
    result["estimand"] = "candidate_minus_base"
    result["paired_item_count"] = len(differences)
    return result


def jensen_shannon_divergence(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise RuntimeError("JSD distributions have incompatible support")
    if any(not math.isfinite(value) or value < 0 for value in left + right):
        raise RuntimeError("JSD distributions contain invalid mass")
    left_sum = sum(left)
    right_sum = sum(right)
    if left_sum <= 0 or right_sum <= 0:
        raise RuntimeError("JSD distributions have zero total mass")
    p = [value / left_sum for value in left]
    q = [value / right_sum for value in right]
    midpoint = [(a + b) / 2 for a, b in zip(p, q, strict=True)]

    def kl(values: list[float], reference: list[float]) -> float:
        return sum(
            value * math.log(value / target)
            for value, target in zip(values, reference, strict=True)
            if value > 0
        )

    return (kl(p, midpoint) + kl(q, midpoint)) / 2


def topk_plus_residual_kl(
    base_log_probabilities: list[float],
    base_residual_probability: float,
    candidate_log_probabilities_at_base_ids: list[float],
    candidate_residual_probability: float,
) -> float:
    if len(base_log_probabilities) != len(candidate_log_probabilities_at_base_ids):
        raise RuntimeError("top-k KL supports differ")
    if not 0.0 <= base_residual_probability <= 1.0 or not 0.0 <= (
        candidate_residual_probability
    ) <= 1.0:
        raise RuntimeError("top-k KL residual mass is invalid")
    base_probabilities = [math.exp(value) for value in base_log_probabilities]
    candidate_probabilities = [
        math.exp(value) for value in candidate_log_probabilities_at_base_ids
    ]
    if abs(sum(base_probabilities) + base_residual_probability - 1.0) > 1e-5:
        raise RuntimeError("base top-k plus residual mass does not sum to one")
    if abs(sum(candidate_probabilities) + candidate_residual_probability - 1.0) > 1e-5:
        raise RuntimeError("candidate top-k plus residual mass does not sum to one")
    if any(value <= 0 for value in candidate_probabilities) or (
        base_residual_probability > 0 and candidate_residual_probability <= 0
    ):
        return math.inf
    result = sum(
        base_probability
        * (base_log_probability - candidate_log_probability)
        for base_probability, base_log_probability, candidate_log_probability in zip(
            base_probabilities,
            base_log_probabilities,
            candidate_log_probabilities_at_base_ids,
            strict=True,
        )
        if base_probability > 0
    )
    if base_residual_probability > 0:
        result += base_residual_probability * math.log(
            base_residual_probability / candidate_residual_probability
        )
    return max(0.0, result)


def _load_manifests() -> tuple[
    dict,
    dict,
    dict[str, dict],
    dict[str, dict],
    dict[str, dict],
    list[tuple[bytes, dict]],
]:
    contract, shard_manifest, domain, extensions, general = runner._load_static_inputs()
    qa_rows = runner._load_qa_rows()
    return contract, shard_manifest, domain, extensions, general, qa_rows


def _load_response_shards(contract: dict, shard_manifest: dict) -> tuple[list[dict], list[dict]]:
    shards = []
    rows = []
    all_seen = set()
    for shard_index in range(protocol.SHARD_COUNT):
        path = (
            protocol.OUTPUT_ROOT
            / "base_model/responses"
            / f"response_shard_{shard_index}.json"
        )
        if not path.is_file():
            raise RuntimeError(f"base response shard is missing: {path.name}")
        value = json.loads(path.read_text())
        protocol.validate_self_address(value, schema=runner.RESPONSE_SHARD_SCHEMA)
        if (
            value["status"] != "complete"
            or value["shard_index"] != shard_index
            or value["preregistration_content_sha256"]
            != contract["content_sha256_before_self_field"]
            or value["response_shard_manifest_content_sha256"]
            != shard_manifest["content_sha256_before_self_field"]
        ):
            raise RuntimeError(f"base response shard contract mismatch: {path.name}")
        expected = shard_manifest["shards"][str(shard_index)]
        observed = [row["item_id"] for row in value["items"]]
        if observed != expected:
            raise RuntimeError(f"base response shard assignment mismatch: {path.name}")
        if any(item_id in all_seen for item_id in observed):
            raise RuntimeError("response item appears in multiple shards")
        all_seen.update(observed)
        for row in value["items"]:
            protocol.validate_self_address(row)
        rows.extend(value["items"])
        shards.append({
            "shard_index": shard_index,
            "path": protocol.display_path(path),
            "file_sha256": protocol.file_sha256(path),
            "content_sha256": value["content_sha256_before_self_field"],
            "item_count": value["item_count"],
            "physical_gpu_index": value["physical_gpu_index"],
            "peak_allocated_bytes": value["peak_allocated_bytes"],
            "peak_reserved_bytes": value["peak_reserved_bytes"],
            "routing_accumulator": value["routing_accumulator"],
            "runtime": value["runtime"],
        })
    expected_all = {
        item_id
        for assigned in shard_manifest["shards"].values()
        for item_id in assigned
    }
    if all_seen != expected_all:
        raise RuntimeError("four response shards do not cover the sealed suite")
    return shards, rows


def _merge_routing(shards: list[dict]) -> dict:
    layer_names = sorted(shards[0]["routing_accumulator"])
    if len(layer_names) != 40:
        raise RuntimeError("baseline routing evidence does not contain 40 layers")
    result = {}
    for name in layer_names:
        states = [shard["routing_accumulator"][name] for shard in shards]
        num_experts = states[0]["num_experts"]
        if num_experts != 256 or any(state["num_experts"] != num_experts for state in states):
            raise RuntimeError("baseline routing expert geometry changed")
        token_count = sum(state["token_count"] for state in states)
        entropy_sum = sum(state["router_entropy_sum"] for state in states)
        high_count = sum(
            state["top1_probability_ge_0_90_count"] for state in states
        )
        probability_load = [
            sum(state["probability_load_sum"][index] for state in states)
            / token_count
            for index in range(num_experts)
        ]
        selected_counts = [
            sum(state["selected_assignment_count"][index] for state in states)
            for index in range(num_experts)
        ]
        load_mean = _mean(probability_load)
        result[name] = {
            "num_experts": num_experts,
            "token_count": token_count,
            "active_expert_count": sum(count > 0 for count in selected_counts),
            "active_expert_fraction": sum(count > 0 for count in selected_counts)
            / num_experts,
            "normalized_router_entropy": entropy_sum
            / token_count
            / math.log(num_experts),
            "probability_load": probability_load,
            "selected_assignment_count": selected_counts,
            "load_imbalance_coefficient_of_variation": statistics.pstdev(
                probability_load
            )
            / load_mean,
            "dominant_expert_probability_fraction": max(probability_load),
            "top1_probability_ge_0_90_frequency": high_count / token_count,
            "expert_load_jsd_against_base": jensen_shannon_divergence(
                probability_load,
                probability_load,
            ),
        }
    return {
        "scope": "official_template_teacher_forced_references",
        "layers": result,
        "layer_count": len(result),
        "base_self_jsd_is_zero": True,
    }


def _refusal_marker(response: str) -> bool:
    normalized = response.casefold()
    markers = (
        "insufficient_information",
        "cannot determine",
        "can't determine",
        "not enough information",
        "cannot assist",
        "can't assist",
        "i don't know",
    )
    return any(marker in normalized for marker in markers)


def score() -> dict:
    (
        contract,
        shard_manifest,
        domain_manifest,
        extension_manifest,
        general_manifest,
        qa_rows,
    ) = _load_manifests()
    extension_seal = contract["domain_extension_seal"]
    if (
        extension_seal["current_state"]
        != "sealed_before_base_baseline_and_pilots"
        or not extension_seal["application_score_composite_eligible"]
        or protocol.file_sha256(protocol.DOMAIN_EXTENSIONS)
        != extension_seal["artifact"]["sha256"]
    ):
        raise RuntimeError(
            "application/synthesis score is not eligible before exact extension seal"
        )
    shards, rows = _load_response_shards(contract, shard_manifest)
    domain_records = []
    extension_records = []
    general_records = []
    total_entropy = 0.0
    total_generated_tokens = 0
    total_thinking_tokens = 0
    max_token_stops = 0
    refusal_non_safety = 0
    refusal_non_safety_denominator = 0
    replay_nll_sum = 0.0
    replay_target_tokens = 0
    topk_reference_positions = 0
    for response_row in rows:
        item_id = response_row["item_id"]
        response = response_row["generation"]["response_text"]
        generation = response_row["generation"]
        total_entropy += generation["output_entropy_sum"]
        total_generated_tokens += generation["generated_token_count"]
        total_thinking_tokens += generation["thinking_token_count"]
        max_token_stops += int(generation["generation_stopped_at_max_new_tokens"])
        if response_row["category"] not in {
            "safety_refusal",
            "uncertainty",
            "domain_extension_unanswerable",
        }:
            refusal_non_safety_denominator += 1
            refusal_non_safety += int(_refusal_marker(response))
        if item_id in domain_manifest:
            item = domain_manifest[item_id]
            selector = item["source_selector"]
            raw, source = qa_rows[selector["line_number_1_based"] - 1]
            if hashlib.sha256(raw).hexdigest() != selector["raw_row_sha256"]:
                raise RuntimeError("domain scoring selector drift")
            metrics = score_domain_response(item, response, source)
            domain_records.append({
                "item_id": item_id,
                "source_group_id": selector["source_group_id"],
                "form": item["form"],
                **metrics,
            })
        elif item_id in extension_manifest:
            item = extension_manifest[item_id]
            source_rows = []
            for selector in item["source_selectors"]:
                raw, source = qa_rows[selector["line_number_1_based"] - 1]
                if (
                    hashlib.sha256(raw).hexdigest() != selector["raw_row_sha256"]
                    or source["source_split_lineage_v1"]["source_group_id"]
                    != item["source_group_id"]
                ):
                    raise RuntimeError("domain extension scoring selector drift")
                source_rows.append(source)
            metrics = score_domain_extension_response(item, response, source_rows)
            extension_records.append({
                "item_id": item_id,
                "source_group_id": item["source_group_id"],
                "extension_type": item["extension_type"],
                **metrics,
            })
        elif item_id in general_manifest:
            fixture = general_manifest[item_id]
            metrics = score_general_response(fixture, response)
            general_records.append({
                "item_id": item_id,
                "bootstrap_unit": fixture["bootstrap_unit"],
                "category": fixture["category"],
                "scoring_status": fixture["scoring_status"],
                **metrics,
            })
            if fixture["scoring_status"] == "deterministic":
                teacher = response_row["teacher_forced"]
                if (
                    not teacher
                    or not teacher["official_template_prefix_alignment_passed"]
                    or teacher["assistant_target_token_count"] <= 0
                ):
                    raise RuntimeError("synthetic replay target alignment is absent")
                replay_nll_sum += teacher["nll_sum"]
                replay_target_tokens += teacher["assistant_target_token_count"]
                reference = teacher.get("base_topk_reference")
                if not reference or reference["k"] != 64:
                    raise RuntimeError("base top-k KL reference is absent")
                topk_reference_positions += len(reference["positions"])
        else:
            raise RuntimeError(f"response has unknown item: {item_id}")

    if (
        len(domain_records) != 148
        or len(extension_records) != 88
        or len(general_records) != 400
    ):
        raise RuntimeError("baseline response family counts changed")
    deterministic_general = [
        row for row in general_records if row["scoring_status"] == "deterministic"
    ]
    if len(deterministic_general) != 340:
        raise RuntimeError("deterministic synthetic general count changed")
    category_scores = {}
    for category in contract["composite_weights"][
        "general_deterministic_renormalized"
    ]:
        category_rows = [
            row for row in deterministic_general if row["category"] == category
        ]
        category_scores[category] = _mean(
            [row["deterministic_pass"] for row in category_rows]
        )
    weights = contract["composite_weights"]["general_deterministic_renormalized"]
    general_composite = sum(
        weights[category] * category_scores[category]
        for category in weights
    )
    closed = [row for row in domain_records if row["form"] == "closed_book"]
    grounded = [row for row in domain_records if row["form"] == "grounded"]
    closed_correctness = _mean(
        [row["correct_at_token_f1_0_65"] for row in closed]
    )
    grounded_correctness = _mean(
        [row["correct_at_token_f1_0_65"] for row in grounded]
    )
    grounding = _mean([
        (
            row["grounded_exact_support"]
            + (1.0 - row["unsupported_claim_unit_rate"])
        )
        / 2
        for row in grounded
    ])
    calibration = 1.0 - _mean([
        row["confidence_brier"] for row in domain_records + extension_records
    ])
    extension_scores = {
        extension_type: _mean([
            row["extension_score"]
            for row in extension_records
            if row["extension_type"] == extension_type
        ])
        for extension_type in (
            "paraphrase_recall",
            "application",
            "synthesis",
            "contradiction",
            "unanswerable",
            "false_premise",
        )
    }
    application_and_synthesis = (
        0.60 * extension_scores["application"]
        + 0.40 * extension_scores["synthesis"]
    )
    uncertainty_and_false_premise = _mean([
        category_scores["uncertainty"],
        extension_scores["contradiction"],
        extension_scores["unanswerable"],
        extension_scores["false_premise"],
    ])
    available_domain_weight = 1.0
    domain_current = (
        0.30 * closed_correctness
        + 0.25 * grounded_correctness
        + 0.10 * grounding
        + 0.10 * calibration
        + 0.10 * extension_scores["paraphrase_recall"]
        + 0.10 * application_and_synthesis
        + 0.05 * uncertainty_and_false_premise
    )
    routing = _merge_routing(shards)
    selection_composite = (
        0.55 * domain_current
        + 0.25 * general_composite
        + 0.10
        + 0.10
    )
    domain_bootstrap = grouped_bootstrap_mean_ci(
        [
            {
                "source_group_id": row["source_group_id"],
                "score": row["correct_at_token_f1_0_65"],
            }
            for row in domain_records
        ]
        + [
            {
                "source_group_id": row["source_group_id"],
                "score": row["extension_score"],
            }
            for row in extension_records
        ],
        value_key="score",
        group_key="source_group_id",
    )
    general_bootstrap = grouped_bootstrap_mean_ci(
        deterministic_general,
        value_key="deterministic_pass",
        group_key="bootstrap_unit",
        seed=protocol.BOOTSTRAP_SEED + 1,
    )
    artifact = protocol.self_address({
        "schema": protocol.BASELINE_SCHEMA,
        "status": "base_baseline_complete",
        "preregistration_content_sha256": contract[
            "content_sha256_before_self_field"
        ],
        "response_shard_manifest_content_sha256": shard_manifest[
            "content_sha256_before_self_field"
        ],
        "expected_shards": [0, 1, 2, 3],
        "completed_shards": [0, 1, 2, 3],
        "response_shard_receipts": [
            {key: value for key, value in shard.items() if key not in (
                "routing_accumulator",
                "runtime",
            )}
            for shard in shards
        ],
        "runtime_receipts": [shard["runtime"] for shard in shards],
        "aggregate_metrics": {
            "domain": {
                "closed_book_correctness": closed_correctness,
                "grounded_correctness": grounded_correctness,
                "grounded_support_and_claim_score": grounding,
                "calibration_score_one_minus_brier": calibration,
                "sealed_extension_scores": extension_scores,
                "application_and_synthesis_score": application_and_synthesis,
                "uncertainty_contradiction_false_premise_score": (
                    uncertainty_and_false_premise
                ),
                "normalized_exact_rate": _mean(
                    [row["normalized_exact"] for row in domain_records]
                ),
                "mean_token_f1": _mean([row["token_f1"] for row in domain_records]),
                "malformed_json_rate": 1.0
                - _mean([row["json_valid"] for row in domain_records]),
                "unsupported_claim_unit_rate": _mean([
                    row["unsupported_claim_unit_rate"] for row in grounded
                ]),
                "current_available_weight": available_domain_weight,
                "sealed_domain_composite": domain_current,
                "correctness_bootstrap_95": domain_bootstrap,
            },
            "general": {
                "category_pass_rates": category_scores,
                "weighted_deterministic_composite": general_composite,
                "deterministic_pass_bootstrap_95": general_bootstrap,
                "format_validity_rate": _mean([
                    row["format_valid"] for row in deterministic_general
                ]),
                "tool_call_validity_rate": _mean([
                    row["tool_call_valid"]
                    for row in deterministic_general
                    if row["tool_call_valid"] is not None
                ]),
                "unanswerable_false_answer_rate": _mean(
                    [
                        1.0 - row["deterministic_pass"]
                        for row in deterministic_general
                        if row["category"] == "uncertainty"
                    ]
                    + [
                        row["unanswerable_false_answer"]
                        for row in extension_records
                        if row["extension_type"] == "unanswerable"
                    ]
                ),
                "subjective_conversation_and_safety_quality": None,
                "subjective_judge_status": "gated_not_scored",
            },
            "distributional": {
                "synthetic_replay_nll": replay_nll_sum / replay_target_tokens,
                "synthetic_replay_nll_sum": replay_nll_sum,
                "synthetic_replay_target_tokens": replay_target_tokens,
                "base_to_base_kl_nats_per_token": 0.0,
                "base_top64_plus_residual_reference_positions": topk_reference_positions,
                "mean_output_entropy": total_entropy / total_generated_tokens,
                "mean_response_token_length": total_generated_tokens / len(rows),
                "mean_thinking_token_length": total_thinking_tokens / len(rows),
                "generation_max_token_stop_rate": max_token_stops / len(rows),
                "non_safety_refusal_marker_rate": refusal_non_safety
                / refusal_non_safety_denominator,
            },
            "selection_composite": selection_composite,
            "hard_regression_gates_evaluated_against_self": False,
        },
        "routing_metrics": routing,
        "replay_nll_and_kl_reference": {
            "nll_surface": "340 deterministic synthetic general fixtures",
            "base_top_k": 64,
            "residual_bucket": True,
            "kl_direction_for_candidate": "KL(base || candidate)",
            "base_self_kl": 0.0,
        },
        "adapter_scale": 0.0,
        "pilot_ready": True,
        "pilot_blockers": [],
        "final_or_terminal_claim": None,
        "authority": {
            "development_only": True,
            "synthetic_general_only": True,
            "training_launched": False,
            "final_or_terminal_data_accessed": False,
            "subjective_judge_used": False,
        },
        "source_receipts": {
            "scorer_sha256": protocol.file_sha256(Path(__file__)),
            "runner_sha256": protocol.file_sha256(Path(runner.__file__)),
        },
    })
    protocol._atomic_write(
        protocol.BASELINE_ARTIFACT,
        (json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    artifact = score()
    print(json.dumps({
        "status": artifact["status"],
        "output": protocol.display_path(protocol.BASELINE_ARTIFACT),
        "content_sha256": artifact["content_sha256_before_self_field"],
        "completed_shards": artifact["completed_shards"],
        "pilot_ready": artifact["pilot_ready"],
        "raw_prompt_or_response_printed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
