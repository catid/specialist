#!/usr/bin/env python3
"""Deterministic category/source-balanced train-only sampling ablation.

The three variants share the exact V434 train bytes and 64-request budget:
source/category-capped uniform conflict units, fixed category stratification,
and stratification with a 50% lagged weakness-replay component.  Weakness is
derived only from completed training rollouts and is never read from dev, OOD,
protected, judge, or benchmark artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import build_train_shadow_folds_v37a as conflict_v37a
import qa_quality
import recipe_evaluation_contract_v1 as evaluation_contract


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/datasets/"
    "recipe_sampling_ablation_v1.json"
).resolve()
CONTRACT = evaluation_contract.CONTRACT
TRAIN = evaluation_contract.TRAIN
CENSUS = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v61a_v434_train_only_baseline_census/baseline_census_strata_v61a.json"
).resolve()
CENSUS_EVIDENCE = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v61a_v434_train_only_baseline_census/baseline_census_evidence_v61a.json"
).resolve()

EXPECTED = {
    "contract_file": (
        "04af81499067e2feb0186c0a61e4c1af10f838a8eb7deec6dd41cd192748cacf"
    ),
    "contract_content": (
        "2442c0c2be3ac4c883612f400f8f213ce3bc82ef96e03fad1ef10ec3b7d11fad"
    ),
    "train": "ae949c37de6abcd57fd8e2b9da8148b80ee072cfc16a7cf023c4ca89021b840a",
    "census_file": (
        "23c8393555c3d7f09c95ecc7e23a04637f86df8fd20f55f67b67000ae78257f5"
    ),
    "census_content": (
        "d6a34b36fea22a8bdc97698a377ffb4df596bade8cf1506c41721f5db9c4185a"
    ),
    "census_evidence_file": (
        "bb95aa2b99d292f0c5cff27afbd255d4ca0697097e8c38e1dbda4dfa63280640"
    ),
    "census_evidence_content": (
        "92df95db709e05c2c81c94d98d755a81d15069c94e7bbffa37c9566e4a33b3b5"
    ),
}

MASTER_SEED = "specialist-0j5.4-category-source-replay-20260717"
CATEGORIES = (
    "techniques", "rigging", "safety", "lineage", "equipment",
    "troubleshooting", "resources",
)
CATEGORY_QUOTAS = {
    "techniques": 12,
    "rigging": 8,
    "safety": 10,
    "lineage": 9,
    "equipment": 8,
    "troubleshooting": 9,
    "resources": 8,
}
PRIORITY_QUOTAS = {
    "techniques": 6,
    "rigging": 4,
    "safety": 5,
    "lineage": 4,
    "equipment": 4,
    "troubleshooting": 5,
    "resources": 4,
}
PANEL_SIZE = 64
SOURCE_CAP = 15
UNIFORM_CATEGORY_CAP = 20
PRIORITY_FRACTION = 0.5
PRIORITY_MULTIPLIER_CAP = 2.0
EMA_NEW_OBSERVATION_WEIGHT = 0.25
SCREEN_POPULATION = 16
SCREEN_GENERATIONS = 2

_PATTERNS = {
    "safety": (
        r"\b(?:safe(?:ty|ly)?|risk|nerve|circulat\w*|numb\w*|tingl\w*|"
        r"pain\w*|injur\w*|emergenc\w*|consent\w*|negotiat\w*|safeword\w*|"
        r"aftercare|breath\w*|neck|fall\w*|medical|danger\w*|body check|"
        r"monitor\w*|red flag|cut(?:ting)? (?:the )?rope|emt shears?)\b"
    ),
    "rigging": (
        r"\b(?:suspension|upline|hard ?point|anchor\w*|carabiner\w*|rigg\w*|"
        r"bamboo|frame|tripod|ceiling|beam|load[- ]?bear\w*|load limit|hoist|"
        r"pulley|redundan\w*|hanger|ring placement)\b"
    ),
    "troubleshooting": (
        r"\b(?:troubleshoot\w*|mistake\w*|slip\w*|jam\w*|collapse\w*|"
        r"uneven|too tight|too loose|loosen\w*|adjust\w*|problem\w*|"
        r"prevent\w*|avoid\w*|correct\w*|fix\w*|tension issue|bunch\w*|"
        r"twist\w*|distort\w*)\b"
    ),
    "equipment": (
        r"\b(?:jute|hemp|nylon|synthetic|natural fiber|rope material|diameter|"
        r"rope length|condition(?:ing)? rope|wash\w*|clean\w*|maintenance|"
        r"stor(?:e|age|ing)|coil\w*|wax\w*|oil\w*|shears?|hardware|equipment|"
        r"rope end|whip\w*|singed)\b"
    ),
    "lineage": (
        r"\b(?:history|historical|origin|lineage|pioneer\w*|founded|founder|"
        r"influenc\w*|mentor|student|developed|introduced|popularized|"
        r"photograph\w*|magazine|publication|era|school|style|tradition|"
        r"etymolog\w*|term|translat\w*|meaning of|japanese bondage|"
        r"who (?:was|is|created|developed)|when (?:was|did))\b"
    ),
    "resources": (
        r"\b(?:resource\w*|where can|where should|where .{0,24} find|"
        r"learn online|tutorial|class|course|workshop|community|event|"
        r"convention|calendar|educator|supplier|purchase|buy|website|video|"
        r"channel|study platform|meet(?:s|ing)? every)\b"
    ),
}
_COMPILED_PATTERNS = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in _PATTERNS.items()
}
_CATEGORY_PRIORITY = (
    "resources", "safety", "rigging", "troubleshooting", "equipment",
    "lineage",
)
_URL_TRIVIA = re.compile(
    r"\b(?:canonical|exact)\s+(?:rope[- ]?topia\s+)?(?:url|link)\b|"
    r"\b(?:which|what)\s+(?:canonical\s+)?(?:url|link)\b|"
    r"\bweb\s+address\b",
    re.IGNORECASE,
)
_FORBIDDEN_LEDGER_KEYS = re.compile(
    r"(?:dev|eval|validation|ood|holdout|protected|benchmark|judge|question|"
    r"answer|prompt|completion|response|excerpt|text|url)",
    re.IGNORECASE,
)


def canonical_sha256(value: object) -> str:
    return evaluation_contract.canonical_sha256(value)


def file_sha256(path: Path | str) -> str:
    return evaluation_contract.file_sha256(path)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: expected object")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"{path}: invalid JSONL")
    return rows


def _key(*values: object) -> str:
    return hashlib.sha256(
        "\0".join(map(str, values)).encode("utf-8")
    ).hexdigest()


def _unit_uniform(unit_id: str, generation: int, purpose: str) -> float:
    raw = int(_key(MASTER_SEED, purpose, generation, unit_id)[:16], 16)
    return (raw + 1) / (2**64 + 1)


def classify_category(row: dict) -> str:
    material = " ".join(
        str(row.get(key, "")) for key in ("question", "answer")
    )
    for name in _CATEGORY_PRIORITY:
        if _COMPILED_PATTERNS[name].search(material):
            return name
    return "techniques"


def exclusion_reason(row: dict) -> str | None:
    try:
        pair = qa_quality.qa_pair_from_record(row)
    except (TypeError, ValueError):
        return "defective_qa"
    if pair is None:
        return "defective_qa"
    question, answer = pair
    if row.get("kind") == "qa_resource_index" or _URL_TRIVIA.search(question):
        return "canonical_url_trivia"
    if qa_quality.LOW_VALUE.search(question + " " + answer):
        return "volatile_or_low_value"
    if qa_quality.normalize_text(question) == qa_quality.normalize_text(answer):
        return "question_answer_identity"
    return None


def _validate_inputs() -> tuple[dict, list[dict], dict, dict]:
    if file_sha256(CONTRACT) != EXPECTED["contract_file"]:
        raise RuntimeError("evaluation contract file changed")
    contract = _read_json(CONTRACT)
    evaluation_contract.validate_contract(contract)
    if contract["content_sha256_before_self_field"] != EXPECTED["contract_content"]:
        raise RuntimeError("evaluation contract content changed")
    evaluation_contract.assert_adaptation_inputs([TRAIN], contract)
    if file_sha256(TRAIN) != EXPECTED["train"]:
        raise RuntimeError("V434 train bytes changed")
    rows = _read_jsonl(TRAIN)
    if len(rows) != 448:
        raise RuntimeError("V434 train row count changed")

    if file_sha256(CENSUS) != EXPECTED["census_file"]:
        raise RuntimeError("V61A census bytes changed")
    if file_sha256(CENSUS_EVIDENCE) != EXPECTED["census_evidence_file"]:
        raise RuntimeError("V61A census evidence bytes changed")
    census, evidence = _read_json(CENSUS), _read_json(CENSUS_EVIDENCE)
    for value, schema, content in (
        (
            census, "v61a-v434-train-baseline-census-strata",
            EXPECTED["census_content"],
        ),
        (
            evidence, "v61a-v434-train-baseline-census-evidence",
            EXPECTED["census_evidence_content"],
        ),
    ):
        compact = {
            key: item for key, item in value.items()
            if key != "content_sha256_before_self_field"
        }
        if (
            value.get("schema") != schema
            or value.get("content_sha256_before_self_field") != content
            or canonical_sha256(compact) != content
            or value.get("eval_ood_shadow_or_holdout_opened") is not False
            or value.get("raw_question_answer_or_generation_text_persisted")
            is not False
            or value.get("candidate_selection_or_promotion_performed")
            is not False
        ):
            raise RuntimeError("V61A train-only census contract changed")
    if (
        evidence.get("train_dataset_file_sha256") != EXPECTED["train"]
        or census.get("conflict_unit_count") != 208
        or evidence.get("conflict_unit_count") != 208
        or census.get("actor_count") != 4
    ):
        raise RuntimeError("V61A census/train identity changed")
    return contract, rows, census, evidence


def build_frame(rows: list[dict]) -> dict:
    units = conflict_v37a.build_conflict_units(rows)
    exclusions = Counter()
    excluded_row_sha256s = []
    candidates_by_category = {name: [] for name in CATEGORIES}
    uniform_candidates = []
    unit_ids = set()
    eligible_rows = 0
    for unit in units:
        unit_id = unit["identity_sha256"]
        unit_ids.add(unit_id)
        by_category = defaultdict(list)
        all_eligible = []
        for index in unit["indices"]:
            row = rows[index]
            reason = exclusion_reason(row)
            if reason is not None:
                exclusions[reason] += 1
                excluded_row_sha256s.append(conflict_v37a.row_sha256(row))
                continue
            category = classify_category(row)
            by_category[category].append(index)
            all_eligible.append(index)
            eligible_rows += 1
        if not all_eligible:
            continue

        def representative(indices: list[int], purpose: str) -> int:
            return min(
                indices,
                key=lambda index: _key(
                    MASTER_SEED, purpose, unit_id,
                    conflict_v37a.row_sha256(rows[index]),
                ),
            )

        uniform_index = representative(all_eligible, "uniform-representative")
        uniform_row = rows[uniform_index]
        uniform_candidates.append({
            "unit_identity_sha256": unit_id,
            "row_index": uniform_index,
            "row_sha256": conflict_v37a.row_sha256(uniform_row),
            "source": uniform_row["source"],
            "category": classify_category(uniform_row),
        })
        for category, indices in by_category.items():
            index = representative(indices, f"{category}-representative")
            row = rows[index]
            candidates_by_category[category].append({
                "unit_identity_sha256": unit_id,
                "row_index": index,
                "row_sha256": conflict_v37a.row_sha256(row),
                "source": row["source"],
                "category": category,
            })
    if len(units) != 208 or len(unit_ids) != 208:
        raise RuntimeError("sampling frame conflict-unit identity changed")
    if eligible_rows + sum(exclusions.values()) != len(rows):
        raise RuntimeError("sampling frame row accounting changed")
    return {
        "units": units,
        "unit_ids": unit_ids,
        "uniform_candidates": uniform_candidates,
        "candidates_by_category": candidates_by_category,
        "eligible_rows": eligible_rows,
        "excluded_row_sha256s": sorted(excluded_row_sha256s),
        "exclusions": dict(sorted(exclusions.items())),
    }


def build_initial_weakness_ledger(census: dict, frame: dict) -> dict:
    actors = int(census["actor_count"])
    entries = []
    for item in census["units"]:
        unit_id = item["unit_identity_sha256"]
        mean_f1 = float(item["mean_f1"])
        exact_rate = int(item["exact_actor_count"]) / actors
        nonzero_rate = int(item["nonzero_actor_count"]) / actors
        weakness = (
            0.50 * (1.0 - mean_f1)
            + 0.30 * (1.0 - exact_rate)
            + 0.20 * (1.0 - nonzero_rate)
        )
        weakness = min(1.0, max(0.0, weakness))
        entries.append({
            "unit_identity_sha256": unit_id,
            "observations": actors,
            "ema_weakness_hex": weakness.hex(),
            "priority_multiplier_hex": (
                min(PRIORITY_MULTIPLIER_CAP, 1.0 + weakness).hex()
            ),
        })
    entries.sort(key=lambda item: item["unit_identity_sha256"])
    if (
        len(entries) != 208
        or {item["unit_identity_sha256"] for item in entries}
        != frame["unit_ids"]
    ):
        raise RuntimeError("V61A weakness ledger does not cover the train frame")
    ledger = {
        "schema": "specialist-train-only-weakness-ledger-v1",
        "status": "frozen_historical_train_only_cold_start",
        "train_dataset_sha256": EXPECTED["train"],
        "as_of_completed_generation": -1,
        "source": {
            "kind": "four-actor V61A unadapted train-only census",
            "file_sha256": EXPECTED["census_file"],
            "content_sha256": EXPECTED["census_content"],
            "model_updates_or_candidate_selection_used": False,
            "nontrain_surfaces_opened": False,
        },
        "formula": {
            "mean_f1_error_weight": 0.50,
            "exact_rate_error_weight": 0.30,
            "nonzero_rate_error_weight": 0.20,
            "new_observation_ema_weight": EMA_NEW_OBSERVATION_WEIGHT,
            "priority_multiplier": "min(2.0, 1.0 + ema_weakness)",
        },
        "entries": entries,
    }
    ledger["content_sha256_before_self_field"] = canonical_sha256(ledger)
    return ledger


def validate_weakness_ledger(ledger: dict, frame: dict) -> None:
    def keys_are_safe(value: object) -> bool:
        if isinstance(value, dict):
            return all(
                not _FORBIDDEN_LEDGER_KEYS.search(str(key))
                and keys_are_safe(item)
                for key, item in value.items()
            )
        if isinstance(value, list):
            return all(keys_are_safe(item) for item in value)
        return True

    compact = {
        key: value for key, value in ledger.items()
        if key != "content_sha256_before_self_field"
    }
    entries = ledger.get("entries", ())
    if (
        ledger.get("schema") != "specialist-train-only-weakness-ledger-v1"
        or ledger.get("train_dataset_sha256") != EXPECTED["train"]
        or ledger.get("content_sha256_before_self_field")
        != canonical_sha256(compact)
        or not keys_are_safe(ledger)
        or len(entries) != 208
        or {item.get("unit_identity_sha256") for item in entries}
        != frame["unit_ids"]
    ):
        raise RuntimeError("invalid or non-train weakness ledger")
    for item in entries:
        weakness = float.fromhex(item["ema_weakness_hex"])
        multiplier = float.fromhex(item["priority_multiplier_hex"])
        if (
            not 0.0 <= weakness <= 1.0
            or not 1.0 <= multiplier <= PRIORITY_MULTIPLIER_CAP
            or int(item["observations"]) < 1
        ):
            raise RuntimeError("weakness ledger value exceeded its cap")


def update_weakness_ledger(
    ledger: dict,
    frame: dict,
    observations: list[dict],
    *,
    completed_generation: int,
) -> dict:
    """Apply aggregate train-only observations for use next generation."""
    validate_weakness_ledger(ledger, frame)
    if completed_generation <= ledger["as_of_completed_generation"]:
        raise RuntimeError("weakness observations must advance generation")
    allowed = {
        "unit_identity_sha256", "mean_reward", "exact_rate",
        "nonzero_rate", "observations",
    }
    by_unit = {}
    for item in observations:
        if set(item) != allowed or item["unit_identity_sha256"] in by_unit:
            raise RuntimeError("weakness observation schema is not train-only")
        if item["unit_identity_sha256"] not in frame["unit_ids"]:
            raise RuntimeError("weakness observation references a nontrain unit")
        numeric = [
            float(item["mean_reward"]), float(item["exact_rate"]),
            float(item["nonzero_rate"]),
        ]
        if (
            any(not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in numeric)
            or not isinstance(item["observations"], int)
            or item["observations"] <= 0
        ):
            raise RuntimeError("weakness observation value is invalid")
        by_unit[item["unit_identity_sha256"]] = item

    entries = []
    for old in ledger["entries"]:
        item = dict(old)
        observation = by_unit.get(item["unit_identity_sha256"])
        if observation is not None:
            fresh = (
                0.50 * (1.0 - float(observation["mean_reward"]))
                + 0.30 * (1.0 - float(observation["exact_rate"]))
                + 0.20 * (1.0 - float(observation["nonzero_rate"]))
            )
            prior = float.fromhex(item["ema_weakness_hex"])
            weakness = (
                (1.0 - EMA_NEW_OBSERVATION_WEIGHT) * prior
                + EMA_NEW_OBSERVATION_WEIGHT * fresh
            )
            item["ema_weakness_hex"] = weakness.hex()
            item["priority_multiplier_hex"] = min(
                PRIORITY_MULTIPLIER_CAP, 1.0 + weakness
            ).hex()
            item["observations"] += observation["observations"]
        entries.append(item)
    updated = {
        "schema": ledger["schema"],
        "status": "lagged_completed_train_generation_only",
        "train_dataset_sha256": ledger["train_dataset_sha256"],
        "as_of_completed_generation": completed_generation,
        "source": {
            "kind": "prior ledger plus aggregate completed training rollouts",
            "previous_content_sha256": ledger[
                "content_sha256_before_self_field"
            ],
            "completed_generation": completed_generation,
            "updated_units": len(by_unit),
            "model_updates_or_candidate_selection_used": False,
            "nontrain_surfaces_opened": False,
        },
        "formula": dict(ledger["formula"]),
        "entries": entries,
    }
    updated["content_sha256_before_self_field"] = canonical_sha256(updated)
    validate_weakness_ledger(updated, frame)
    return updated


def _select_uniform(frame: dict, generation: int) -> list[dict]:
    source_counts, category_counts = Counter(), Counter()
    selected = []
    ordered = sorted(
        frame["uniform_candidates"],
        key=lambda item: _key(
            MASTER_SEED, "uniform", generation,
            item["unit_identity_sha256"],
        ),
    )
    for candidate in ordered:
        if (
            source_counts[candidate["source"]] >= SOURCE_CAP
            or category_counts[candidate["category"]] >= UNIFORM_CATEGORY_CAP
        ):
            continue
        selected.append({**candidate, "component": "uniform"})
        source_counts[candidate["source"]] += 1
        category_counts[candidate["category"]] += 1
        if len(selected) == PANEL_SIZE:
            break
    if len(selected) != PANEL_SIZE:
        raise RuntimeError("uniform capped panel cannot fill")
    return selected


def _candidate_order(
    candidates: list[dict], *, category: str, generation: int,
    component: str, multipliers: dict[str, float],
) -> list[dict]:
    if component == "category_uniform":
        return sorted(
            candidates,
            key=lambda item: _key(
                MASTER_SEED, component, generation, category,
                item["unit_identity_sha256"],
            ),
        )

    def weighted_key(item: dict) -> tuple[float, str]:
        unit_id = item["unit_identity_sha256"]
        uniform = _unit_uniform(unit_id, generation, f"priority-{category}")
        key = -math.log(uniform) / multipliers[unit_id]
        return key, unit_id

    return sorted(candidates, key=weighted_key)


def _select_stratified(
    frame: dict,
    generation: int,
    *,
    priority: bool,
    multipliers: dict[str, float],
) -> list[dict]:
    used_units, source_counts = set(), Counter()
    selected = []
    categories = sorted(
        CATEGORIES,
        key=lambda name: (
            len(frame["candidates_by_category"][name])
            / CATEGORY_QUOTAS[name],
            name,
        ),
    )
    for category in categories:
        priority_slots = PRIORITY_QUOTAS[category] if priority else 0
        components = (
            ["category_uniform"] * (CATEGORY_QUOTAS[category] - priority_slots)
            + ["weakness_replay"] * priority_slots
        )
        for component in components:
            candidates = _candidate_order(
                frame["candidates_by_category"][category],
                category=category,
                generation=generation,
                component=component,
                multipliers=multipliers,
            )
            chosen = next((
                item for item in candidates
                if item["unit_identity_sha256"] not in used_units
                and source_counts[item["source"]] < SOURCE_CAP
            ), None)
            if chosen is None:
                raise RuntimeError(
                    f"cannot fill {category}/{component} within source cap"
                )
            selected.append({**chosen, "component": component})
            used_units.add(chosen["unit_identity_sha256"])
            source_counts[chosen["source"]] += 1
    if len(selected) != PANEL_SIZE:
        raise RuntimeError("stratified panel size changed")
    return selected


def _seal_panel(
    name: str, selected: list[dict], generation: int,
    multipliers: dict[str, float],
) -> dict:
    selected = sorted(
        selected,
        key=lambda item: _key(
            MASTER_SEED, "request-order", generation, name,
            item["unit_identity_sha256"],
        ),
    )
    items = []
    for position, item in enumerate(selected):
        unit_id = item["unit_identity_sha256"]
        items.append({
            "position": position,
            "row_sha256": item["row_sha256"],
            "unit_identity_sha256": unit_id,
            "category": item["category"],
            "source": item["source"],
            "component": item["component"],
            "priority_multiplier_hex": multipliers[unit_id].hex(),
        })
    categories = Counter(item["category"] for item in items)
    sources = Counter(item["source"] for item in items)
    components = Counter(item["component"] for item in items)
    if (
        len(items) != PANEL_SIZE
        or len({item["unit_identity_sha256"] for item in items}) != PANEL_SIZE
        or max(sources.values()) > SOURCE_CAP
        or max(categories.values()) > UNIFORM_CATEGORY_CAP
        or any(
            not 1.0 <= float.fromhex(item["priority_multiplier_hex"])
            <= PRIORITY_MULTIPLIER_CAP
            for item in items
        )
    ):
        raise RuntimeError(f"{name} sampling caps failed")
    if name != "uniform_capped" and categories != Counter(CATEGORY_QUOTAS):
        raise RuntimeError(f"{name} category quotas changed")
    if name == "prioritized_capped" and components["weakness_replay"] != 32:
        raise RuntimeError("priority replay fraction changed")
    return {
        "name": name,
        "generation": generation,
        "rows": len(items),
        "items": items,
        "category_counts": dict(sorted(categories.items())),
        "source_counts": dict(sorted(sources.items())),
        "component_counts": dict(sorted(components.items())),
        "ordered_row_identity_sha256": canonical_sha256([
            item["row_sha256"] for item in items
        ]),
        "ordered_unit_identity_sha256": canonical_sha256([
            item["unit_identity_sha256"] for item in items
        ]),
    }


def build_manifest(
    *, generation: int = 0, weakness_ledger: dict | None = None
) -> dict:
    contract, rows, census, _evidence = _validate_inputs()
    frame = build_frame(rows)
    ledger = weakness_ledger or build_initial_weakness_ledger(census, frame)
    validate_weakness_ledger(ledger, frame)
    if ledger["as_of_completed_generation"] >= generation:
        raise RuntimeError(
            "priority ledger must predate the generation being sampled"
        )
    multipliers = {
        item["unit_identity_sha256"]: float.fromhex(
            item["priority_multiplier_hex"]
        )
        for item in ledger["entries"]
    }
    variants = [
        _seal_panel(
            "uniform_capped", _select_uniform(frame, generation),
            generation, multipliers,
        ),
        _seal_panel(
            "category_stratified",
            _select_stratified(
                frame, generation, priority=False, multipliers=multipliers,
            ),
            generation, multipliers,
        ),
        _seal_panel(
            "prioritized_capped",
            _select_stratified(
                frame, generation, priority=True, multipliers=multipliers,
            ),
            generation, multipliers,
        ),
    ]
    manifest = {
        "schema": "specialist-category-prioritized-sampling-ablation-v1",
        "status": "sealed_cpu_only_before_sampling_ablation_launch",
        "generation": generation,
        "evaluation_contract": {
            "path": str(CONTRACT),
            "file_sha256": EXPECTED["contract_file"],
            "content_sha256": EXPECTED["contract_content"],
            "protected_access_authorized": False,
        },
        "source": {
            "path": str(TRAIN),
            "file_sha256": EXPECTED["train"],
            "rows": len(rows),
            "conflict_units": len(frame["units"]),
            "same_exact_train_bytes_for_every_variant": True,
        },
        "eligibility": {
            "eligible_rows": frame["eligible_rows"],
            "excluded_rows": sum(frame["exclusions"].values()),
            "exclusion_counts": frame["exclusions"],
            "excluded_row_identity_set_sha256": canonical_sha256(
                frame["excluded_row_sha256s"]
            ),
            "url_trivia_or_defective_qa_can_be_sampled": False,
        },
        "category_rule": {
            "categories": list(CATEGORIES),
            "priority_order": list(_CATEGORY_PRIORITY),
            "patterns_sha256": canonical_sha256(_PATTERNS),
            "default": "techniques",
            "stratified_quotas": dict(CATEGORY_QUOTAS),
            "uniform_category_cap": UNIFORM_CATEGORY_CAP,
            "maximum_uniform_category_fraction": (
                UNIFORM_CATEGORY_CAP / PANEL_SIZE
            ),
        },
        "source_balance": {
            "per_panel_source_cap": SOURCE_CAP,
            "maximum_source_fraction": SOURCE_CAP / PANEL_SIZE,
            "source_cap_applies_to_every_variant": True,
        },
        "priority_replay": {
            "maximum_panel_fraction": PRIORITY_FRACTION,
            "priority_rows_per_panel": sum(PRIORITY_QUOTAS.values()),
            "priority_quotas": dict(PRIORITY_QUOTAS),
            "multiplier_cap": PRIORITY_MULTIPLIER_CAP,
            "sampling": "deterministic weighted without replacement",
            "ledger_content_sha256": ledger[
                "content_sha256_before_self_field"
            ],
            "ledger_as_of_completed_generation": ledger[
                "as_of_completed_generation"
            ],
            "same_generation_feedback_allowed": False,
            "train_statistics_only": True,
        },
        "compute_match": {
            "mode": "estimator_control",
            "panel_rows_per_variant_per_generation": PANEL_SIZE,
            "mirrored_population_per_generation": SCREEN_POPULATION,
            "screen_generations": SCREEN_GENERATIONS,
            "optimization_generated_rollouts_per_variant": (
                PANEL_SIZE * SCREEN_POPULATION * SCREEN_GENERATIONS
            ),
            "same_generation_count_prompt_count_population_decoding_and_seed": True,
            "gpu_second_ceiling_per_arm": contract["compute_accounting"][
                "budget_modes"
            ]["estimator_control"]["screen_gpu_second_ceiling_per_arm"],
            "contract_rollout_target": contract["compute_accounting"][
                "budget_modes"
            ]["estimator_control"][
                "screen_target_generated_rollouts_per_arm"
            ],
        },
        "variants": variants,
        "weakness_ledger": ledger,
        "access_receipt": {
            "train_semantics_opened": True,
            "dev_semantics_opened": False,
            "ood_semantics_opened": False,
            "protected_semantics_opened": False,
            "model_or_gpu_accessed": False,
            "model_outcomes_used": "frozen V61A train-only census only",
        },
        "content_minimization": {
            "question_persisted": False,
            "answer_persisted": False,
            "evidence_persisted": False,
            "url_persisted": False,
            "row_content_persisted": False,
            "only_row_unit_hashes_category_source_and_capped_train_stats": True,
        },
        "implementation_bindings": {
            "builder": str(Path(__file__).resolve()),
            "builder_file_sha256": file_sha256(Path(__file__).resolve()),
            "category_patterns_sha256": canonical_sha256(_PATTERNS),
            "quality_gate_file_sha256": file_sha256(qa_quality.__file__),
            "conflict_unit_file_sha256": file_sha256(conflict_v37a.__file__),
        },
    }
    if (
        manifest["compute_match"]["optimization_generated_rollouts_per_variant"]
        != manifest["compute_match"]["contract_rollout_target"]
    ):
        raise RuntimeError("sampling screen does not meet contract rollout budget")
    manifest["content_sha256_before_self_field"] = canonical_sha256(manifest)
    return manifest


def validate_manifest(manifest: dict) -> None:
    compact = {
        key: value for key, value in manifest.items()
        if key != "content_sha256_before_self_field"
    }
    variants = manifest.get("variants", ())
    if (
        manifest.get("schema")
        != "specialist-category-prioritized-sampling-ablation-v1"
        or manifest.get("content_sha256_before_self_field")
        != canonical_sha256(compact)
        or manifest.get("source", {}).get("file_sha256") != EXPECTED["train"]
        or manifest.get("evaluation_contract", {}).get(
            "protected_access_authorized"
        ) is not False
        or len(variants) != 3
        or any(panel.get("rows") != PANEL_SIZE for panel in variants)
    ):
        raise RuntimeError("invalid sampling ablation manifest")


def materialize_variant_rows(manifest: dict, variant: str) -> list[dict]:
    """Resolve a sealed panel from the exact registered train bytes."""
    validate_manifest(manifest)
    if file_sha256(TRAIN) != EXPECTED["train"]:
        raise RuntimeError("sampling train bytes changed before materialization")
    panels = [item for item in manifest["variants"] if item["name"] == variant]
    if len(panels) != 1:
        raise ValueError(f"unknown or repeated sampling variant {variant}")
    rows = _read_jsonl(TRAIN)
    by_identity = {
        conflict_v37a.row_sha256(row): row for row in rows
    }
    if len(by_identity) != len(rows):
        raise RuntimeError("sampling train row identities repeated")
    result = []
    for item in panels[0]["items"]:
        try:
            result.append(by_identity[item["row_sha256"]])
        except KeyError as exc:
            raise RuntimeError("sampling panel row no longer resolves") from exc
    if len(result) != PANEL_SIZE:
        raise RuntimeError("materialized sampling panel size changed")
    return result


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
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    value = build_manifest()
    if args.check:
        if _read_json(args.output) != value:
            raise RuntimeError("persisted sampling manifest differs from rebuild")
    else:
        _atomic_json(args.output.resolve(), value)
    print(json.dumps({
        "path": str(args.output.resolve()),
        "content_sha256": value["content_sha256_before_self_field"],
        "variants": [item["name"] for item in value["variants"]],
        "rows_per_variant": PANEL_SIZE,
        "rollouts_per_variant": value["compute_match"][
            "optimization_generated_rollouts_per_variant"
        ],
        "protected_semantics_opened": False,
        "model_or_gpu_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
