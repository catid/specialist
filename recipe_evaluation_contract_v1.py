#!/usr/bin/env python3
"""Immutable evaluation and compute contract for specialist recipe HPO.

The contract reuses the frozen V434 train identity registry, the V37A
train-derived shadow fold, and Eval V3/OOD identities.  Protected rows are
opened only while this CPU-only builder computes collision-free identities;
their questions, answers, excerpts, URLs, and per-row metrics are never
persisted in the contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import build_eval_v3 as eval_v3
import build_train_shadow_folds_v37a as folds_v37a
import eggroll_es_train_panel_sampler_v13 as semantic_v13


ROOT = Path(__file__).resolve().parent
CONTRACT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "recipe_evaluation_compute_contract_v1.json"
).resolve()
TRAIN_REGISTRY = (
    ROOT / "experiments/eggroll_es_hpo/datasets/"
    "v56_v434_train_disjoint_identity_registry.json"
).resolve()
TRAIN = (
    ROOT / "experiments/sft_controls/"
    "v49d_v434_sampling_midpoint_lr5p5e5/train_v434_fold3_v49d.jsonl"
).resolve()
DEV_MANIFEST = (
    ROOT / "experiments/sft_controls/v37a_shadow_folds_v412/manifest_v37a.json"
).resolve()
DEV = (
    ROOT / "experiments/sft_controls/"
    "v37a_shadow_folds_v412/fold_3_shadow_dev.jsonl"
).resolve()
EVAL_REPORT = (ROOT / "data/eval_v3.report.json").resolve()
DOMAIN_EVAL = (ROOT / "data/eval_qa_v3.jsonl").resolve()
OOD_QA = (ROOT / "data/ood_qa_v3.jsonl").resolve()
OOD_PROSE = (ROOT / "data/ood_prose_v3.jsonl").resolve()

EXPECTED = {
    "train_registry_file": (
        "907886ccf689618cd58e68eff05e8212a29826a6c7655c7698632164f9ec5bc8"
    ),
    "train_registry_content": (
        "aea5b80183b2d98cf0dff37fd5f68cd6a8573901cf71c13a4558417c851cae8a"
    ),
    "train": "ae949c37de6abcd57fd8e2b9da8148b80ee072cfc16a7cf023c4ca89021b840a",
    "dev_manifest_file": (
        "7d2a8f2b86f9007aa2bfe8ae043be15647451cc4bbea53a18d5915085879ee9d"
    ),
    "dev_manifest_content": (
        "3fcc2820e8dffe6a21198d0520365aace049735ac84bda179ea44bc8ad0881eb"
    ),
    "dev": "6d5b72f7506a752fd5275425739ec785e25f0ff486f5c03b68e91c8e99d7ebeb",
    "eval_report": (
        "245097c9fab935558b246d577c55c5fe3d64df534de8690e75256f10a8d05d9f"
    ),
    "domain_eval": (
        "ab9a391e249910e876826dfab9c8e2f8e17a7b8695e6f018a3e515e5aa69603b"
    ),
    "ood_qa": "25a48b9494134731e51043047afadb340291a9ae3e9cfec9d9cfd8c73ddb255d",
    "ood_prose": (
        "3299457c7a23dfb0eb10408b2226b6231e291b519a52325feed607d901605e57"
    ),
}

ROLE_ORDER = ("train", "dev", "protected_holdout", "ood_qa", "ood_prose")
QA_ROLES = frozenset(("train", "dev", "protected_holdout", "ood_qa"))
PROTECTED_TEXT_FIELDS = frozenset(
    ("question", "answer", "excerpt", "text", "title", "url")
)
FORBIDDEN_TERMINAL_KEYS = re.compile(
    r"(?:question|answer|excerpt|prompt|completion|response|text|title|"
    r"per[_-]?item|token|logit|raw)",
    re.IGNORECASE,
)


def canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"{path}: empty or invalid JSONL")
    return rows


def _require_file(path: Path, expected: str, label: str) -> None:
    observed = file_sha256(path)
    if observed != expected:
        raise RuntimeError(f"{label} bytes changed: {observed}")


def _item_identity(item_id: str) -> str:
    return canonical_sha256({
        "schema": "protected-item-opaque-identity-v1",
        "item_id": item_id,
    })


def _lineage_identity(value: str) -> str:
    return canonical_sha256({
        "schema": "raw-lineage-edge-identity-v56",
        "value": value,
    })


def _row_urls(row: dict) -> frozenset[str]:
    return frozenset(
        eval_v3.normalize_source_url(value)
        for _field, value in eval_v3.source_urls(row)
    )


def _document_ids(row: dict, role: str) -> frozenset[str]:
    if isinstance(row.get("document_sha256"), str):
        return frozenset((row["document_sha256"],))
    if isinstance(row.get("source_document_sha256"), str):
        return frozenset((row["source_document_sha256"],))
    if role == "ood_prose" and isinstance(row.get("text"), str):
        return frozenset((hashlib.sha256(row["text"].encode()).hexdigest(),))
    if role == "ood_qa":
        return frozenset((EXPECTED["ood_qa"],))
    return frozenset()


def _lineage_ids(row: dict) -> frozenset[str]:
    return frozenset(
        _lineage_identity(value)
        for value in folds_v37a.row_lineage_identities(row)
    )


def _body(row: dict) -> str:
    if isinstance(row.get("question"), str):
        return row["question"] + "\n" + str(row.get("answer", ""))
    return str(row.get("title", "")) + "\n" + str(row.get("text", ""))


def _tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(re.findall(r"[a-z0-9]+", normalized))


def _token_ngrams(tokens: tuple[str, ...], width: int = 3) -> frozenset[tuple]:
    return frozenset(
        tuple(tokens[index:index + width])
        for index in range(max(0, len(tokens) - width + 1))
    )


def _jaccard(left: frozenset, right: frozenset) -> float:
    return len(left & right) / len(left | right) if left and right else 0.0


def _record(row: dict, role: str) -> dict:
    body_tokens = _tokens(_body(row))
    qa_features = None
    if role in QA_ROLES:
        qa_features = (
            semantic_v13._content_tokens(str(row.get("question", ""))),
            semantic_v13._content_tokens(str(row.get("answer", ""))),
        )
    return {
        "documents": _document_ids(row, role),
        "urls": _row_urls(row),
        "lineages": _lineage_ids(row),
        "qa_features": qa_features,
        "tokens": body_tokens,
        "ngrams": _token_ngrams(body_tokens),
    }


def _near_duplicate(left: dict, right: dict) -> bool:
    if left["qa_features"] is not None and right["qa_features"] is not None:
        if semantic_v13._semantic_match(
            left["qa_features"], right["qa_features"]
        ):
            return True
    if left["tokens"] == right["tokens"] and left["tokens"]:
        return True
    lgrams, rgrams = left["ngrams"], right["ngrams"]
    if not lgrams or not rgrams:
        return False
    intersection = len(lgrams & rgrams)
    union = len(lgrams | rgrams)
    containment = intersection / min(len(lgrams), len(rgrams))
    return intersection / union >= 0.80 or (
        min(len(left["tokens"]), len(right["tokens"])) >= 12
        and containment >= 0.90
    )


def _collision_reasons(left: dict, right: dict) -> frozenset[str]:
    reasons = set()
    if left["documents"] & right["documents"]:
        reasons.add("document_sha256")
    if left["urls"] & right["urls"]:
        reasons.add("normalized_url")
    if left["lineages"] & right["lineages"]:
        reasons.add("raw_lineage")
    if _near_duplicate(left, right):
        reasons.add("near_duplicate")
    return frozenset(reasons)


def audit_role_records(role_records: dict[str, list[dict]]) -> dict:
    """Return content-free cross-role collision counts."""
    pairs = {}
    passed = True
    for left_index, left_role in enumerate(ROLE_ORDER):
        for right_role in ROLE_ORDER[left_index + 1:]:
            reasons = defaultdict(int)
            colliding_pairs = 0
            for left in role_records[left_role]:
                for right in role_records[right_role]:
                    found = _collision_reasons(left, right)
                    if found:
                        colliding_pairs += 1
                        for reason in found:
                            reasons[reason] += 1
            key = f"{left_role}__{right_role}"
            pairs[key] = {
                "colliding_row_pairs": colliding_pairs,
                "by_identity_domain": {
                    name: reasons[name]
                    for name in (
                        "document_sha256", "normalized_url",
                        "raw_lineage", "near_duplicate",
                    )
                },
            }
            passed = passed and colliding_pairs == 0
    return {"passed": passed, "pairs": pairs}


def _validate_existing_registries() -> tuple[dict, dict, dict]:
    for label, path in (
        ("train_registry_file", TRAIN_REGISTRY),
        ("train", TRAIN),
        ("dev_manifest_file", DEV_MANIFEST),
        ("dev", DEV),
        ("eval_report", EVAL_REPORT),
        ("domain_eval", DOMAIN_EVAL),
        ("ood_qa", OOD_QA),
        ("ood_prose", OOD_PROSE),
    ):
        _require_file(path, EXPECTED[label], label)

    registry = _read_json(TRAIN_REGISTRY)
    compact = {
        key: value for key, value in registry.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        registry.get("schema") != "v434-train-disjoint-identity-registry-v56"
        or registry.get("content_sha256_before_self_field")
        != EXPECTED["train_registry_content"]
        or canonical_sha256(compact) != EXPECTED["train_registry_content"]
        or registry.get("aggregate", {}).get("rows") != 448
        or registry.get("source", {}).get("file_sha256") != EXPECTED["train"]
    ):
        raise RuntimeError("V434 train identity registry contract changed")

    manifest = _read_json(DEV_MANIFEST)
    compact = {
        key: value for key, value in manifest.items()
        if key != "content_sha256_before_self_field"
    }
    fold = next(item for item in manifest.get("folds", ()) if item["fold"] == 3)
    if (
        manifest.get("content_sha256_before_self_field")
        != EXPECTED["dev_manifest_content"]
        or canonical_sha256(compact) != EXPECTED["dev_manifest_content"]
        or fold.get("shadow_dev", {}).get("sha256") != EXPECTED["dev"]
        or fold.get("train_dev_conflict_unit_intersection") != 0
        or any(fold.get("train_dev_edge_identity_intersections", {}).values())
    ):
        raise RuntimeError("V37A fold-3 train/dev disjointness contract changed")

    report = _read_json(EVAL_REPORT)
    if (
        report.get("schema") != "specialist-eval-v3-build-report-v1"
        or not report.get("disjointness", {}).get("passed")
        or report.get("outputs", {}).get("domain_eval", {}).get("sha256")
        != EXPECTED["domain_eval"]
        or report.get("outputs", {}).get("ood_qa", {}).get("sha256")
        != EXPECTED["ood_qa"]
        or report.get("outputs", {}).get("ood_prose", {}).get("sha256")
        != EXPECTED["ood_prose"]
    ):
        raise RuntimeError("Eval V3 registry contract changed")
    return registry, manifest, report


def _protected_subset(
    candidates: list[dict], public_records: list[dict]
) -> tuple[list[dict], dict]:
    selected, excluded = [], defaultdict(int)
    for row in candidates:
        record = _record(row, "protected_holdout")
        reasons = set()
        for public in public_records:
            reasons.update(_collision_reasons(record, public))
        if reasons:
            excluded["rows"] += 1
            for reason in reasons:
                excluded[reason] += 1
        else:
            selected.append(row)
    return selected, {
        "candidate_rows": len(candidates),
        "selected_rows": len(selected),
        "excluded_rows": excluded["rows"],
        "excluded_row_reason_counts": {
            name: excluded[name]
            for name in (
                "document_sha256", "normalized_url",
                "raw_lineage", "near_duplicate",
            )
        },
    }


def build_contract() -> dict:
    registry, manifest, eval_report = _validate_existing_registries()
    train_rows = _read_jsonl(TRAIN)
    dev_rows = _read_jsonl(DEV)
    domain_rows = _read_jsonl(DOMAIN_EVAL)
    ood_qa_rows = _read_jsonl(OOD_QA)
    ood_prose_rows = _read_jsonl(OOD_PROSE)
    if len(train_rows) != 448 or len(dev_rows) != 83:
        raise RuntimeError("frozen train/dev row count changed")
    if len(domain_rows) != 59 or len(ood_qa_rows) != 24 or len(ood_prose_rows) != 16:
        raise RuntimeError("frozen evaluation row count changed")

    train_records = [_record(row, "train") for row in train_rows]
    dev_records = [_record(row, "dev") for row in dev_rows]
    ood_qa_records = [_record(row, "ood_qa") for row in ood_qa_rows]
    ood_prose_records = [_record(row, "ood_prose") for row in ood_prose_rows]
    protected_candidates = [
        row for row in domain_rows if row.get("split") == "heldout"
    ]
    protected_rows, exclusion = _protected_subset(
        protected_candidates,
        train_records + dev_records + ood_qa_records + ood_prose_records,
    )
    role_records = {
        "train": train_records,
        "dev": dev_records,
        "protected_holdout": [
            _record(row, "protected_holdout") for row in protected_rows
        ],
        "ood_qa": ood_qa_records,
        "ood_prose": ood_prose_records,
    }
    audit = audit_role_records(role_records)
    if not audit["passed"]:
        raise RuntimeError("recipe role disjointness audit failed")
    protected_identities = sorted(
        _item_identity(row["item_id"]) for row in protected_rows
    )
    protected_documents = {
        item for record in role_records["protected_holdout"]
        for item in record["documents"]
    }

    contract = {
        "schema": "specialist-recipe-evaluation-compute-contract-v1",
        "status": "sealed_before_recipe_hpo_protected_holdout_unopened_by_hpo",
        "created_at_utc": "2026-07-17T00:00:00+00:00",
        "purpose": (
            "One immutable data boundary, score rule, compute ledger, seed "
            "schedule, stopping rule, and terminal-access firewall for all "
            "specialist-0j5 recipe ablations."
        ),
        "roles": {
            "train": {
                "use": "model updates and train-only sampling statistics",
                "path": str(TRAIN),
                "file_sha256": EXPECTED["train"],
                "rows": 448,
                "identity_registry_path": str(TRAIN_REGISTRY),
                "identity_registry_file_sha256": EXPECTED["train_registry_file"],
                "identity_registry_content_sha256": (
                    EXPECTED["train_registry_content"]
                ),
                "source_documents": registry["aggregate"]["documents"],
                "semantic_clusters": registry["aggregate"]["semantic_clusters"],
            },
            "dev": {
                "use": "HPO, early stopping only at fixed rung boundaries",
                "path": str(DEV),
                "file_sha256": EXPECTED["dev"],
                "rows": 83,
                "conflict_units": 51,
                "train_derived_not_external_test": True,
                "manifest_path": str(DEV_MANIFEST),
                "manifest_file_sha256": EXPECTED["dev_manifest_file"],
                "manifest_content_sha256": EXPECTED["dev_manifest_content"],
            },
            "protected_holdout": {
                "use": "one terminal aggregate report after recipe freeze only",
                "source_path": str(DOMAIN_EVAL),
                "source_file_sha256": EXPECTED["domain_eval"],
                "source_rows": 59,
                "legacy_heldout_candidate_rows": exclusion["candidate_rows"],
                "rows": len(protected_rows),
                "documents": len(protected_documents),
                "selected_opaque_item_identities": protected_identities,
                "selected_identity_set_sha256": canonical_sha256(
                    protected_identities
                ),
                "excluded_for_current_train_or_dev_collision": exclusion,
                "access_authorized_by_this_contract": False,
                "accesses_per_frozen_recipe_program": 1,
                "selection_or_tuning_use": "prohibited",
            },
            "ood": {
                "use": (
                    "noninferiority/trust-region gate; never optimize its "
                    "point score directly"
                ),
                "qa": {
                    "path": str(OOD_QA),
                    "file_sha256": EXPECTED["ood_qa"],
                    "rows": 24,
                },
                "prose": {
                    "path": str(OOD_PROSE),
                    "file_sha256": EXPECTED["ood_prose"],
                    "rows": 16,
                },
            },
        },
        "disjointness": {
            "passed": True,
            "audit": audit,
            "identity_domains": [
                "document SHA-256", "normalized provenance URL",
                "raw-lineage identity", "lexical near-duplicate",
            ],
            "source_normalization": (
                "Eval V3 normalize_source_url: scheme/default-port/path/"
                "tracking normalization with YouTube alias folding"
            ),
            "near_duplicate_rule": {
                "qa_semantic": {
                    "question_jaccard_direct": 0.82,
                    "question_jaccard_joint": 0.66,
                    "answer_jaccard_joint": 0.86,
                    "implementation": "frozen V13 lexical-semantic rule",
                },
                "copy_detection": {
                    "unicode": "NFKC plus casefold",
                    "token_ngram_width": 3,
                    "jaccard_threshold": 0.80,
                    "containment_threshold": 0.90,
                    "containment_minimum_tokens": 12,
                },
            },
            "future_train_refresh_rule": (
                "A refresh must rebuild this four-domain audit against the "
                "same opaque protected selection before any model update."
            ),
        },
        "score_aggregation": {
            "dev_primary": (
                "mean reward within each of 51 immutable conflict units, then "
                "uniform mean over units"
            ),
            "dev_secondary_order": [
                "generated exact count", "generated nonzero count",
                "teacher-forced mean answer-token logprob",
            ],
            "ood_noninferiority": {
                "paired_bootstrap_samples": 20000,
                "bootstrap_seed": 2026071701,
                "qa_mean_reward_delta_95_lcb_minimum": -0.02,
                "qa_exact_count_delta_minimum": -1,
                "prose_mean_token_logprob_delta_95_lcb_minimum": -0.02,
                "all_conditions_required": True,
            },
            "protected_terminal": (
                "uniform mean within source document then uniform document "
                "mean; report reward, exact, nonzero, and paired intervals"
            ),
            "protected_result_can_change_recipe": False,
        },
        "compute_accounting": {
            "charged_gpu_second": (
                "sum over physical GPUs of all model-resident monotonic-time "
                "intervals, including load, candidate materialization, forward, "
                "backward/update, evaluation, checkpoint, and failed attempts "
                "after any model output is observed"
            ),
            "excluded_time": [
                "scheduler queue", "CPU-only preregistration",
                "CPU-only dataset build before model allocation",
            ],
            "generated_rollout": (
                "one generated completion; count candidates, both mirrored "
                "signs, repeats, dev/OOD checks, and observed failed attempts"
            ),
            "also_required": [
                "prompt count", "generated tokens", "teacher-forced tokens",
                "SFT non-padding train tokens", "checkpoint count",
            ],
            "budget_modes": {
                "estimator_control": {
                    "screen_target_generated_rollouts_per_arm": 2048,
                    "screen_gpu_second_ceiling_per_arm": 14400,
                    "confirmation_target_generated_rollouts_per_seed": 2048,
                    "confirmation_gpu_second_ceiling_per_seed": 14400,
                    "equality": (
                        "exact optimization/evaluation rollout counts and CRN "
                        "identities; GPU seconds are reported and capped"
                    ),
                },
                "compute_matched_quality": {
                    "screen_target_gpu_seconds_per_arm": 14400,
                    "confirmation_target_gpu_seconds_per_seed": 14400,
                    "terminal_target_gpu_seconds_per_seed": 28800,
                    "relative_match_tolerance": 0.02,
                    "equality": (
                        "charged GPU seconds within tolerance and exact final "
                        "evaluation requests; family-native work is reported"
                    ),
                },
                "systems_throughput": {
                    "quality_promotion_authorized": False,
                    "equality": (
                        "exact model state, prompts, candidates, token limits, "
                        "and outputs; GPU seconds and bandwidth are outcomes"
                    ),
                },
            },
            "four_gpu_requirement": {
                "physical_gpu_ids": [0, 1, 2, 3],
                "all_model_resident_intervals_attributed": True,
                "useful_positive_activity_each_gpu_per_training_phase": True,
                "idle_capacity_does_not_count_as_useful_activity": True,
            },
        },
        "seeds": {
            "split_seed": eval_report["parameters"]["split_seed"],
            "screen_training_seeds": [1701],
            "confirmation_training_seeds": [1701, 1702, 1703],
            "bootstrap_seed": 2026071701,
            "crn_derivation": (
                "SHA-256(contract content, phase, training seed, conflict-unit "
                "identity, direction identity, repeat); identical across arms"
            ),
            "unregistered_seed_retry": "prohibited",
        },
        "stopping_and_promotion": {
            "screening": {
                "checks_only_at_preregistered_rung_boundaries": True,
                "protected_holdout_visible": False,
                "early_hard_failures": [
                    "nonfinite state or score", "integrity/replica mismatch",
                    "compute ceiling exceeded", "OOD hard gate failure",
                    "missing four-GPU attribution",
                ],
                "failed_arm_budget_reallocated": False,
            },
            "promotion_to_confirmation": {
                "dev_tuple_must_strictly_exceed_frozen_baseline": True,
                "every_ood_noninferiority_condition_must_pass": True,
                "compute_contract_must_pass": True,
                "protocol_or_leak_counter_may_not_increase": True,
            },
            "recipe_freeze": {
                "three_registered_seeds_required": True,
                "positive_dev_primary_seeds_minimum": 2,
                "pooled_dev_primary_paired_95_lcb_minimum": 0.0,
                "all_seed_ood_gates_required": True,
                "tie_breaker": (
                    "dev secondary tuple, then lower charged GPU seconds, "
                    "then lexicographic preregistered recipe id"
                ),
                "selection_receipt_required_before_terminal_access": True,
            },
            "terminal": {
                "one_shot_after_recipe_selection": True,
                "claim_persisted_before_protected_file_read": True,
                "crash_after_claim_consumes_access": True,
                "aggregate_only": True,
                "retry_or_reopen": "prohibited",
                "post_result_hpo_or_recipe_change": "prohibited",
            },
        },
        "content_minimization": {
            "protected_question_persisted": False,
            "protected_answer_persisted": False,
            "protected_excerpt_persisted": False,
            "protected_url_persisted": False,
            "protected_per_item_metric_persisted": False,
            "protected_selection_uses_opaque_hashes_only": True,
            "adaptation_artifacts_may_not_bind_domain_eval_source": True,
        },
        "implementation_bindings": {
            "builder": str(Path(__file__).resolve()),
            "builder_file_sha256": file_sha256(Path(__file__).resolve()),
            "eval_normalizer_file_sha256": file_sha256(eval_v3.__file__),
            "semantic_rule_file_sha256": file_sha256(semantic_v13.__file__),
        },
    }
    contract["content_sha256_before_self_field"] = canonical_sha256(contract)
    return contract


def validate_contract(contract: dict) -> None:
    content_sha = contract.get("content_sha256_before_self_field")
    compact = {
        key: value for key, value in contract.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        contract.get("schema")
        != "specialist-recipe-evaluation-compute-contract-v1"
        or content_sha != canonical_sha256(compact)
        or not contract.get("disjointness", {}).get("passed")
        or contract.get("roles", {}).get("protected_holdout", {}).get(
            "access_authorized_by_this_contract"
        ) is not False
    ):
        raise RuntimeError("invalid recipe evaluation contract")


def assert_adaptation_inputs(paths: Iterable[Path | str], contract: dict) -> None:
    """Allow only the exact registered train bytes in a model-update process.

    A future curriculum dataset must first produce a replacement/extension
    contract with a fresh four-domain audit; arbitrary paths are deliberately
    not trusted merely because their names omit a holdout marker.
    """
    validate_contract(contract)
    protected = contract["roles"]["protected_holdout"]
    protected_path = Path(protected["source_path"]).resolve()
    protected_sha = protected["source_file_sha256"]
    allowed_sha = contract["roles"]["train"]["file_sha256"]
    for value in paths:
        path = Path(value).resolve()
        if not path.is_file():
            raise RuntimeError("adaptation input does not exist")
        observed = file_sha256(path)
        if path == protected_path or observed == protected_sha:
            raise RuntimeError("protected holdout cannot enter adaptation inputs")
        if observed != allowed_sha:
            raise RuntimeError(
                "adaptation input is absent from the audited train registry"
            )


def charge_compute_attempt(attempt: dict) -> dict:
    """Charge disjoint per-GPU residency intervals and all observed rollouts."""
    by_gpu = defaultdict(list)
    for interval in attempt.get("gpu_residency_intervals", ()):
        gpu = int(interval["physical_gpu_id"])
        start, end = float(interval["start_s"]), float(interval["end_s"])
        if gpu not in (0, 1, 2, 3) or not math.isfinite(start + end) or end <= start:
            raise ValueError("invalid GPU residency interval")
        by_gpu[gpu].append((start, end))
    if set(by_gpu) != {0, 1, 2, 3}:
        raise ValueError("compute attempt must attribute all four GPUs")
    gpu_seconds = 0.0
    for intervals in by_gpu.values():
        intervals.sort()
        if any(right[0] < left[1] for left, right in zip(intervals, intervals[1:])):
            raise ValueError("overlapping residency intervals on one GPU")
        gpu_seconds += math.fsum(end - start for start, end in intervals)
    counts = {}
    for key in (
        "optimization_generated_rollouts", "evaluation_generated_rollouts",
        "generated_tokens", "teacher_forced_tokens", "sft_nonpadding_tokens",
    ):
        value = attempt.get(key, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"invalid compute count {key}")
        counts[key] = value
    return {"charged_gpu_seconds": gpu_seconds, **counts}


def aggregate_compute_ledger(attempts: Iterable[dict]) -> dict[str, dict]:
    totals = defaultdict(lambda: defaultdict(float))
    for attempt in attempts:
        arm = attempt.get("arm")
        if not isinstance(arm, str) or not arm:
            raise ValueError("compute attempt has no arm")
        charge = charge_compute_attempt(attempt)
        for key, value in charge.items():
            totals[arm][key] += value
    return {
        arm: {
            key: (int(value) if key != "charged_gpu_seconds" else value)
            for key, value in values.items()
        }
        for arm, values in sorted(totals.items())
    }


def validate_compute_match(
    totals: dict[str, dict], *, mode: str, contract: dict
) -> dict:
    validate_contract(contract)
    if len(totals) < 2:
        raise ValueError("compute matching requires at least two arms")
    policy = contract["compute_accounting"]["budget_modes"].get(mode)
    if policy is None:
        raise ValueError(f"unsupported compute mode {mode}")
    values = list(totals.values())
    gpu_seconds = [float(item["charged_gpu_seconds"]) for item in values]
    if mode == "estimator_control":
        target = policy["screen_target_generated_rollouts_per_arm"]
        if any(
            item["optimization_generated_rollouts"] != target
            for item in values
        ):
            raise RuntimeError("estimator-control rollout budgets differ")
        if len({item["evaluation_generated_rollouts"] for item in values}) != 1:
            raise RuntimeError("estimator-control evaluation rollouts differ")
        if any(value > policy["screen_gpu_second_ceiling_per_arm"] for value in gpu_seconds):
            raise RuntimeError("estimator-control GPU-second ceiling exceeded")
    elif mode == "compute_matched_quality":
        tolerance = policy["relative_match_tolerance"]
        if max(gpu_seconds) / min(gpu_seconds) - 1.0 > tolerance:
            raise RuntimeError("quality arms are not GPU-second matched")
        if len({item["evaluation_generated_rollouts"] for item in values}) != 1:
            raise RuntimeError("quality evaluation rollout counts differ")
    else:
        work_keys = (
            "optimization_generated_rollouts", "evaluation_generated_rollouts",
            "generated_tokens", "teacher_forced_tokens", "sft_nonpadding_tokens",
        )
        if any(len({item[key] for item in values}) != 1 for key in work_keys):
            raise RuntimeError("systems-throughput workloads differ")
    return {
        "passed": True,
        "mode": mode,
        "arms": sorted(totals),
        "gpu_second_min": min(gpu_seconds),
        "gpu_second_max": max(gpu_seconds),
    }


def _validate_selection_receipt(selection: dict, contract: dict) -> str:
    compact = {
        key: value for key, value in selection.items()
        if key != "content_sha256_before_self_field"
    }
    selection_sha = canonical_sha256(compact)
    if (
        selection.get("schema") != "specialist-recipe-selection-receipt-v1"
        or selection.get("status") != "recipe_selected_frozen_hpo_closed"
        or selection.get("content_sha256_before_self_field") != selection_sha
        or selection.get("contract_content_sha256")
        != contract["content_sha256_before_self_field"]
        or selection.get("hpo_closed") is not True
        or selection.get("protected_access_count_before_selection") != 0
        or not isinstance(selection.get("selected_recipe_id"), str)
        or not isinstance(selection.get("selected_checkpoint_sha256"), str)
    ):
        raise RuntimeError("terminal access requires a valid frozen selection")
    return selection_sha


def claim_protected_access_once(
    state_path: Path | str, contract: dict, selection: dict
) -> dict:
    """Persist the irreversible claim before any protected source is opened."""
    validate_contract(contract)
    selection_sha = _validate_selection_receipt(selection, contract)
    state = {
        "schema": "specialist-protected-access-state-v1",
        "status": "claimed_before_source_read",
        "access_count": 1,
        "contract_content_sha256": contract["content_sha256_before_self_field"],
        "selection_content_sha256": selection_sha,
        "protected_source_opened": False,
    }
    state["content_sha256_before_self_field"] = canonical_sha256(state)
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(state, sort_keys=True, indent=2) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return state


def load_claimed_protected_rows(
    state_path: Path | str,
    contract: dict,
    *,
    loader: Callable[[Path], list[dict]] = _read_jsonl,
) -> list[dict]:
    """Load the opaque selected subset after (and only after) a durable claim."""
    validate_contract(contract)
    state_path = Path(state_path)
    state = _read_json(state_path)
    compact = {
        key: value for key, value in state.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        state.get("status") != "claimed_before_source_read"
        or state.get("access_count") != 1
        or state.get("protected_source_opened") is not False
        or state.get("contract_content_sha256")
        != contract["content_sha256_before_self_field"]
        or state.get("content_sha256_before_self_field")
        != canonical_sha256(compact)
    ):
        raise RuntimeError("protected access was already consumed or invalid")
    protected = contract["roles"]["protected_holdout"]
    source = Path(protected["source_path"])
    _require_file(source, protected["source_file_sha256"], "protected source")
    rows = loader(source)
    selected = set(protected["selected_opaque_item_identities"])
    result = [
        row for row in rows
        if row.get("split") == "heldout"
        and isinstance(row.get("item_id"), str)
        and _item_identity(row["item_id"]) in selected
    ]
    if (
        len(result) != protected["rows"]
        or {_item_identity(row["item_id"]) for row in result} != selected
    ):
        raise RuntimeError("protected opaque selection no longer resolves")
    consumed = dict(state)
    consumed.pop("content_sha256_before_self_field", None)
    consumed.update({
        "status": "consumed_no_retry",
        "protected_source_opened": True,
    })
    consumed["content_sha256_before_self_field"] = canonical_sha256(consumed)
    raw = (json.dumps(consumed, sort_keys=True, indent=2) + "\n").encode()
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=state_path.parent, prefix=f".{state_path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(state_path)
    return result


def validate_terminal_aggregate_receipt(receipt: dict) -> None:
    """Permit only aggregate numeric terminal evidence, never row content."""
    def visit(value: object, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if FORBIDDEN_TERMINAL_KEYS.search(str(key)):
                    raise RuntimeError(
                        "protected terminal receipt contains a forbidden field: "
                        + ".".join(path + (str(key),))
                    )
                visit(item, path + (str(key),))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, path + (str(index),))
        elif not isinstance(value, (str, int, float, bool, type(None))):
            raise RuntimeError("terminal receipt has a non-JSON value")

    visit(receipt)
    if receipt.get("aggregate_only") is not True:
        raise RuntimeError("terminal receipt must be aggregate-only")


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CONTRACT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    value = build_contract()
    if args.check:
        if _read_json(args.output) != value:
            raise RuntimeError("persisted recipe contract differs from rebuild")
    else:
        _atomic_json(args.output.resolve(), value)
    print(json.dumps({
        "path": str(args.output.resolve()),
        "content_sha256": value["content_sha256_before_self_field"],
        "protected_rows": value["roles"]["protected_holdout"]["rows"],
        "protected_text_persisted": False,
        "disjointness_passed": value["disjointness"]["passed"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
