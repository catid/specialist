#!/usr/bin/env python3
"""CPU-only greedy versus seeded-stochastic robustness contract.

The module plans content-free evaluation cells and consumes only numeric/hash
receipts.  It has no dataset-path or generated-text API.  Candidate identity is
excluded from decode-seed and payload derivation so every candidate receives
the exact same prompts, parameters, and random draw in each paired cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import eggroll_es_multiobjective_trust_region_v67 as trust_v67


ROOT = Path(__file__).resolve().parent
PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "greedy_seeded_stochastic_robustness_v68.json"
).resolve()
TRUST_PREREGISTRATION = trust_v67.PREREGISTRATION
EXPECTED_TRUST_PREREGISTRATION_FILE_SHA256 = (
    "5001148bc27fb7550dc6b40336ed2d32d9296f2b17f6372a075a02beeee6bf7d"
)
EXPECTED_TRUST_PREREGISTRATION_CONTENT_SHA256 = (
    "feb96e50782f99d736d15e92cddb3ed32a0defad4af07281b5044f5c33e79e11"
)

TRAINING_SEEDS = (1701, 1702, 1703)
DECODE_MODES = ("greedy", "seeded_stochastic")
PROMPT_ROLE_ORDER = ("train", "dev", "ood_qa")
REPEATS_PER_MODE = 4
TOKENS_PER_COMPLETION = 64
MINIMUM_ADJACENT_RANK_MARGIN = 0.0001
STOCHASTIC_MODAL_RANKING_REPEATS_MINIMUM = 3
STOCHASTIC_MEDIAN_PAIRWISE_SPEARMAN_MINIMUM = 0.0

_HEX64 = re.compile(r"[0-9a-f]{64}")

_COMMON_DECODE = {
    "n": 1,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "max_tokens": TOKENS_PER_COMPLETION,
    "min_tokens": TOKENS_PER_COMPLETION,
    "ignore_eos": True,
    "stop": [],
    "stop_token_ids": [],
    "include_stop_str_in_output": False,
    "detokenize": False,
    "skip_special_tokens": False,
    "spaces_between_special_tokens": True,
    "truncate_prompt_tokens": None,
    "logprobs": None,
    "prompt_logprobs": None,
}
_MODE_VALUES = {
    "greedy": {"temperature": 0.0, "top_p": 1.0, "top_k": -1},
    "seeded_stochastic": {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
    },
}


def canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(value: object, keys: Iterable[str], label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    expected = set(keys)
    if set(value) != expected:
        raise ValueError(
            f"{label} keys changed; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _load_trust_preregistration() -> dict:
    if file_sha256(TRUST_PREREGISTRATION) != (
        EXPECTED_TRUST_PREREGISTRATION_FILE_SHA256
    ):
        raise RuntimeError("V67 trust preregistration bytes changed")
    value = json.loads(TRUST_PREREGISTRATION.read_text(encoding="utf-8"))
    trust_v67.validate_preregistration(value)
    if value.get("content_sha256_before_self_field") != (
        EXPECTED_TRUST_PREREGISTRATION_CONTENT_SHA256
    ):
        raise RuntimeError("V67 trust preregistration content changed")
    return value


def decode_contract(mode: str, seed: int) -> dict:
    if mode not in DECODE_MODES:
        raise ValueError("unregistered decode mode")
    seed = _integer(seed, "decode seed")
    result = dict(_COMMON_DECODE)
    result.update(_MODE_VALUES[mode])
    result["seed"] = seed
    return result


def derive_decode_seed(
    training_seed: int,
    mode: str,
    repeat_index: int,
    role: str,
    prompt_block_sha256: str,
) -> int:
    """Derive a draw without candidate identity, preventing seed leakage."""
    if training_seed not in TRAINING_SEEDS:
        raise ValueError("unsealed training seed")
    if mode not in DECODE_MODES:
        raise ValueError("unregistered decode mode")
    if repeat_index not in range(REPEATS_PER_MODE):
        raise ValueError("unregistered decode repeat")
    if role not in PROMPT_ROLE_ORDER:
        raise ValueError("unregistered prompt role")
    block = _sha(prompt_block_sha256, "prompt block")
    digest = hashlib.sha256(json.dumps({
        "schema": "specialist-decode-seed-v68",
        "evaluation_contract_content_sha256": (
            trust_v67.EXPECTED_EVALUATION_CONTRACT_CONTENT_SHA256
        ),
        "training_seed": training_seed,
        "decode_mode": mode,
        "repeat_index": repeat_index,
        "role": role,
        "prompt_block_sha256": block,
    }, sort_keys=True, separators=(",", ":")).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _prompt_blocks(value: list[dict]) -> list[dict]:
    if not isinstance(value, list) or len(value) != len(PROMPT_ROLE_ORDER):
        raise ValueError("exactly one train, dev, and OOD-QA block is required")
    result = []
    for index, block in enumerate(value):
        _exact_keys(
            block, ("role", "prompt_block_sha256", "prompt_count"),
            f"prompt block {index}",
        )
        if block["role"] != PROMPT_ROLE_ORDER[index]:
            raise ValueError("prompt blocks must use the frozen role order")
        result.append({
            "role": block["role"],
            "prompt_block_sha256": _sha(
                block["prompt_block_sha256"], f"prompt block {index}"
            ),
            "prompt_count": _integer(
                block["prompt_count"], f"prompt block {index} count", 1
            ),
        })
    return result


def _judge_contract() -> dict:
    return {
        "schema": "specialist-multiobjective-judge-binding-v68",
        "trust_preregistration_content_sha256": (
            EXPECTED_TRUST_PREREGISTRATION_CONTENT_SHA256
        ),
        "policy_semantics_sha256": trust_v67.POLICY_SEMANTICS_SHA256,
        "component_order": list(trust_v67.COMPONENT_ORDER),
        "hard_event_order": list(trust_v67.HARD_EVENT_ORDER),
        "identical_judge_payload_and_version_all_candidates": True,
    }


def build_ablation_plan(
    candidate_checkpoint_sha256s: list[str],
    prompt_blocks: list[dict],
    *,
    training_seeds: Iterable[int] = TRAINING_SEEDS,
) -> dict:
    _load_trust_preregistration()
    candidates = sorted(
        _sha(item, "candidate checkpoint")
        for item in candidate_checkpoint_sha256s
    )
    if len(candidates) < 2 or len(set(candidates)) != len(candidates):
        raise ValueError("at least two unique candidate checkpoints are required")
    seeds = tuple(training_seeds)
    if (
        any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or seeds != TRAINING_SEEDS
    ):
        raise ValueError("ablation requires all and only the sealed training seeds")
    blocks = _prompt_blocks(prompt_blocks)
    judge = _judge_contract()
    judge_sha = canonical_sha256(judge)
    assignments = []
    for candidate in candidates:
        for training_seed in seeds:
            for mode in DECODE_MODES:
                for repeat_index in range(REPEATS_PER_MODE):
                    for block in blocks:
                        seed = derive_decode_seed(
                            training_seed, mode, repeat_index, block["role"],
                            block["prompt_block_sha256"],
                        )
                        decode = decode_contract(mode, seed)
                        decode_sha = canonical_sha256(decode)
                        payload = {
                            "schema": "specialist-hash-only-evaluation-payload-v68",
                            "prompt_block_sha256": block["prompt_block_sha256"],
                            "prompt_count": block["prompt_count"],
                            "decode_contract_sha256": decode_sha,
                            "judge_contract_sha256": judge_sha,
                            "training_seed": training_seed,
                            "decode_mode": mode,
                            "repeat_index": repeat_index,
                            "role": block["role"],
                        }
                        payload_sha = canonical_sha256(payload)
                        identity = {
                            "candidate_checkpoint_sha256": candidate,
                            "evaluation_payload_sha256": payload_sha,
                        }
                        assignments.append({
                            "assignment_id": canonical_sha256(identity),
                            "candidate_checkpoint_sha256": candidate,
                            "training_seed": training_seed,
                            "decode_mode": mode,
                            "repeat_index": repeat_index,
                            "role": block["role"],
                            "prompt_block_sha256": block["prompt_block_sha256"],
                            "prompt_count": block["prompt_count"],
                            "decode_seed": seed,
                            "decode_contract_sha256": decode_sha,
                            "judge_contract_sha256": judge_sha,
                            "evaluation_payload_sha256": payload_sha,
                            "scheduled_completions": block["prompt_count"],
                            "scheduled_generated_tokens": (
                                block["prompt_count"] * TOKENS_PER_COMPLETION
                            ),
                        })

    budgets = {}
    for candidate in candidates:
        budgets[candidate] = {}
        for mode in DECODE_MODES:
            cells = [
                item for item in assignments
                if item["candidate_checkpoint_sha256"] == candidate
                and item["decode_mode"] == mode
            ]
            budgets[candidate][mode] = {
                "assignments": len(cells),
                "scheduled_completions": sum(
                    item["scheduled_completions"] for item in cells
                ),
                "scheduled_generated_tokens": sum(
                    item["scheduled_generated_tokens"] for item in cells
                ),
            }
    budget_identities = {
        canonical_sha256(budgets[candidate][mode])
        for candidate in candidates for mode in DECODE_MODES
    }
    if len(budget_identities) != 1:
        raise AssertionError("candidate/mode token budgets are not exactly equal")

    plan = {
        "schema": "specialist-greedy-stochastic-ablation-plan-v68",
        "status": "sealed_before_any_candidate_output",
        "evaluation_contract_content_sha256": (
            trust_v67.EXPECTED_EVALUATION_CONTRACT_CONTENT_SHA256
        ),
        "trust_preregistration_content_sha256": (
            EXPECTED_TRUST_PREREGISTRATION_CONTENT_SHA256
        ),
        "candidate_checkpoint_sha256s": candidates,
        "training_seeds": list(seeds),
        "decode_modes": list(DECODE_MODES),
        "repeats_per_mode": REPEATS_PER_MODE,
        "tokens_per_completion": TOKENS_PER_COMPLETION,
        "prompt_blocks": blocks,
        "judge_contract": judge,
        "judge_contract_sha256": judge_sha,
        "assignments": assignments,
        "candidate_mode_budgets": budgets,
        "candidate_identity_excluded_from_seed_and_payload": True,
        "raw_prompt_or_output_access_count": 0,
        "protected_access_count": 0,
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def validate_ablation_plan(plan: dict) -> None:
    _exact_keys(plan, (
        "schema", "status", "evaluation_contract_content_sha256",
        "trust_preregistration_content_sha256",
        "candidate_checkpoint_sha256s", "training_seeds", "decode_modes",
        "repeats_per_mode", "tokens_per_completion", "prompt_blocks",
        "judge_contract", "judge_contract_sha256", "assignments",
        "candidate_mode_budgets",
        "candidate_identity_excluded_from_seed_and_payload",
        "raw_prompt_or_output_access_count", "protected_access_count",
        "plan_sha256",
    ), "ablation plan")
    expected = build_ablation_plan(
        plan["candidate_checkpoint_sha256s"],
        plan["prompt_blocks"],
        training_seeds=plan["training_seeds"],
    )
    if plan != expected:
        raise RuntimeError("ablation plan identity or common payload changed")


_CELL_KEYS = (
    "schema", "assignment_id", "candidate_checkpoint_sha256",
    "training_seed", "decode_mode", "repeat_index", "role",
    "prompt_block_sha256", "evaluation_payload_sha256", "decode_seed",
    "scheduled_completions", "scheduled_generated_tokens",
    "realized_completions", "realized_generated_tokens",
    "length_finish_count", "early_stop_count", "output_inventory_sha256",
    "judge_metrics_sha256", "unit_balanced_reward",
    "all_item_hard_gates_passed", "wall_seconds", "raw_output_access_count",
    "raw_output_persisted", "protected_access_count",
    "protected_source_opened", "content_sha256_before_self_field",
)


def _validate_cell(receipt: dict, assignment: dict) -> dict[str, bool]:
    _exact_keys(receipt, _CELL_KEYS, "cell receipt")
    compact = {
        key: value for key, value in receipt.items()
        if key != "content_sha256_before_self_field"
    }
    if receipt["content_sha256_before_self_field"] != canonical_sha256(compact):
        raise RuntimeError("cell receipt content hash changed")
    if receipt["schema"] != "specialist-decode-cell-receipt-v68":
        raise ValueError("cell receipt schema changed")
    for key in (
        "assignment_id", "candidate_checkpoint_sha256", "training_seed",
        "decode_mode", "repeat_index", "role", "prompt_block_sha256",
        "evaluation_payload_sha256", "decode_seed", "scheduled_completions",
        "scheduled_generated_tokens",
    ):
        if receipt[key] != assignment[key]:
            raise RuntimeError(f"cell receipt leaked or changed sealed {key}")
    for key in (
        "realized_completions", "realized_generated_tokens",
        "length_finish_count", "early_stop_count", "raw_output_access_count",
        "protected_access_count",
    ):
        _integer(receipt[key], f"cell {key}")
    if (
        receipt["raw_output_access_count"] != 0
        or receipt["raw_output_persisted"] is not False
    ):
        raise RuntimeError("raw candidate output access is prohibited")
    if (
        receipt["protected_access_count"] != 0
        or receipt["protected_source_opened"] is not False
    ):
        raise RuntimeError("protected output access is prohibited")
    _sha(receipt["output_inventory_sha256"], "output inventory")
    _sha(receipt["judge_metrics_sha256"], "judge metrics")
    reward = _finite(receipt["unit_balanced_reward"], "unit-balanced reward")
    if reward < -1.0 or reward > 1.0:
        raise ValueError("unit-balanced reward must be in [-1, 1]")
    if not isinstance(receipt["all_item_hard_gates_passed"], bool):
        raise ValueError("cell hard-gate status must be boolean")
    if _finite(receipt["wall_seconds"], "cell wall seconds") <= 0.0:
        raise ValueError("cell wall seconds must be positive")
    return {
        "completion_count_exact": (
            receipt["realized_completions"]
            == assignment["scheduled_completions"]
        ),
        "generated_token_count_exact": (
            receipt["realized_generated_tokens"]
            == assignment["scheduled_generated_tokens"]
        ),
        "length_finish_count_exact": (
            receipt["length_finish_count"]
            == assignment["scheduled_completions"]
        ),
        "no_early_stop": receipt["early_stop_count"] == 0,
        "multiobjective_item_gates": receipt["all_item_hard_gates_passed"],
    }


def bind_trust_receipt(
    candidate_checkpoint_sha256: str,
    training_seed: int,
    receipt: dict,
) -> dict:
    candidate = _sha(candidate_checkpoint_sha256, "candidate checkpoint")
    if training_seed not in TRAINING_SEEDS:
        raise ValueError("unsealed trust-receipt seed")
    value = {
        "candidate_checkpoint_sha256": candidate,
        "training_seed": training_seed,
        "trust_receipt": receipt,
        "trust_receipt_content_sha256": receipt.get(
            "content_sha256_before_self_field"
        ),
    }
    value["binding_sha256"] = canonical_sha256(value)
    return value


def _validate_trust_bindings(
    bindings: list[dict], candidates: list[str]
) -> dict[tuple[str, int], dict]:
    if not isinstance(bindings, list):
        raise ValueError("trust receipt bindings must be a list")
    expected_pairs = {
        (candidate, seed) for candidate in candidates for seed in TRAINING_SEEDS
    }
    result = {}
    for binding in bindings:
        _exact_keys(binding, (
            "candidate_checkpoint_sha256", "training_seed", "trust_receipt",
            "trust_receipt_content_sha256", "binding_sha256",
        ), "trust receipt binding")
        compact = {
            key: value for key, value in binding.items()
            if key != "binding_sha256"
        }
        if binding["binding_sha256"] != canonical_sha256(compact):
            raise RuntimeError("trust receipt binding changed")
        candidate = _sha(
            binding["candidate_checkpoint_sha256"], "trust candidate"
        )
        seed = binding["training_seed"]
        if seed not in TRAINING_SEEDS:
            raise ValueError("unsealed trust-receipt seed")
        if binding["trust_receipt_content_sha256"] != binding[
            "trust_receipt"
        ].get("content_sha256_before_self_field"):
            raise RuntimeError("trust receipt content binding changed")
        key = (candidate, seed)
        if key in result:
            raise ValueError("duplicate candidate/seed trust receipt")
        try:
            trust_v67.require_promotion(binding["trust_receipt"])
            passed = True
        except RuntimeError as error:
            if "non-compensable hard gate" not in str(error):
                raise
            passed = False
        result[key] = {
            "passed": passed,
            "content_sha256": binding["trust_receipt_content_sha256"],
        }
    if set(result) != expected_pairs:
        raise ValueError("trust receipts must cover every candidate and sealed seed")
    return result


def _ranking(scores: dict[str, float]) -> tuple[tuple[str, ...], float]:
    ordered = tuple(sorted(scores, key=lambda item: (-scores[item], item)))
    margins = [
        scores[left] - scores[right]
        for left, right in zip(ordered, ordered[1:])
    ]
    return ordered, min(margins)


def _spearman(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if set(left) != set(right) or len(left) < 2:
        raise ValueError("rankings must cover the same candidates")
    positions = {candidate: index for index, candidate in enumerate(right)}
    squared = math.fsum(
        (index - positions[candidate]) ** 2
        for index, candidate in enumerate(left)
    )
    count = len(left)
    return 1.0 - 6.0 * squared / (count * (count * count - 1))


def _summary(values: list[float]) -> dict:
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "population_stddev": math.sqrt(variance),
        "minimum": min(values),
        "maximum": max(values),
    }


def analyze_ablation(
    plan: dict,
    cell_receipts: list[dict],
    trust_receipt_bindings: list[dict],
) -> dict:
    validate_ablation_plan(plan)
    assignments = {item["assignment_id"]: item for item in plan["assignments"]}
    if not isinstance(cell_receipts, list):
        raise ValueError("cell receipts must be a list")
    receipts = {}
    protocol_failures = []
    for receipt in cell_receipts:
        assignment_id = receipt.get("assignment_id") if isinstance(receipt, dict) else None
        if assignment_id not in assignments or assignment_id in receipts:
            raise ValueError("missing, duplicate, or foreign cell assignment")
        checks = _validate_cell(receipt, assignments[assignment_id])
        receipts[assignment_id] = receipt
        protocol_failures.extend(
            {"assignment_id": assignment_id, "failure": name}
            for name, passed in checks.items() if not passed
        )
    if set(receipts) != set(assignments):
        raise ValueError("cell receipts do not exactly cover the sealed plan")

    candidates = plan["candidate_checkpoint_sha256s"]
    trust_status = _validate_trust_bindings(
        trust_receipt_bindings, candidates
    )
    checks = {
        "all_cell_protocol_gates": not protocol_failures,
    }

    observed_budgets = {}
    for candidate in candidates:
        observed_budgets[candidate] = {}
        for mode in DECODE_MODES:
            cells = [
                receipt for receipt in receipts.values()
                if receipt["candidate_checkpoint_sha256"] == candidate
                and receipt["decode_mode"] == mode
            ]
            observed_budgets[candidate][mode] = {
                "completions": sum(item["realized_completions"] for item in cells),
                "generated_tokens": sum(
                    item["realized_generated_tokens"] for item in cells
                ),
            }
    budget_hashes = {
        canonical_sha256(observed_budgets[candidate][mode])
        for candidate in candidates for mode in DECODE_MODES
    }
    checks["observed_candidate_mode_token_budgets_equal"] = len(budget_hashes) == 1

    greedy_identity_failures = []
    for candidate in candidates:
        for training_seed in TRAINING_SEEDS:
            for role in PROMPT_ROLE_ORDER:
                cells = sorted((
                    receipt for receipt in receipts.values()
                    if receipt["candidate_checkpoint_sha256"] == candidate
                    and receipt["training_seed"] == training_seed
                    and receipt["decode_mode"] == "greedy"
                    and receipt["role"] == role
                ), key=lambda item: item["repeat_index"])
                if (
                    len(cells) != REPEATS_PER_MODE
                    or len({item["output_inventory_sha256"] for item in cells}) != 1
                    or len({item["judge_metrics_sha256"] for item in cells}) != 1
                ):
                    greedy_identity_failures.append({
                        "candidate_checkpoint_sha256": candidate,
                        "training_seed": training_seed,
                        "role": role,
                    })
    checks["greedy_output_and_judge_identity_replicated"] = (
        not greedy_identity_failures
    )

    dev_cells = {
        (
            receipt["candidate_checkpoint_sha256"], receipt["training_seed"],
            receipt["decode_mode"], receipt["repeat_index"],
        ): receipt
        for receipt in receipts.values() if receipt["role"] == "dev"
    }
    rank_summaries = []
    stable_winners = []
    for training_seed in TRAINING_SEEDS:
        for mode in DECODE_MODES:
            rankings = []
            margins = []
            for repeat_index in range(REPEATS_PER_MODE):
                scores = {
                    candidate: dev_cells[
                        (candidate, training_seed, mode, repeat_index)
                    ]["unit_balanced_reward"]
                    for candidate in candidates
                }
                ranking, margin = _ranking(scores)
                rankings.append(ranking)
                margins.append(margin)
            counts = Counter(rankings)
            modal_ranking, modal_count = sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )[0]
            correlations = [
                _spearman(left, right)
                for index, left in enumerate(rankings)
                for right in rankings[index + 1:]
            ]
            median_spearman = statistics.median(correlations)
            margins_passed = min(margins) >= MINIMUM_ADJACENT_RANK_MARGIN
            if mode == "greedy":
                stability_passed = modal_count == REPEATS_PER_MODE
            else:
                stability_passed = (
                    modal_count >= STOCHASTIC_MODAL_RANKING_REPEATS_MINIMUM
                    and median_spearman
                    >= STOCHASTIC_MEDIAN_PAIRWISE_SPEARMAN_MINIMUM
                )
            stratum_passed = margins_passed and stability_passed
            checks[f"rank:{training_seed}:{mode}:tie_free"] = margins_passed
            checks[f"rank:{training_seed}:{mode}:replicated_stability"] = (
                stability_passed
            )
            if stratum_passed:
                stable_winners.append(modal_ranking[0])
            rank_summaries.append({
                "training_seed": training_seed,
                "decode_mode": mode,
                "rankings": [list(item) for item in rankings],
                "minimum_adjacent_margin": min(margins),
                "modal_ranking": list(modal_ranking),
                "modal_ranking_repeats": modal_count,
                "median_pairwise_spearman": median_spearman,
                "stable_winner": modal_ranking[0] if stratum_passed else None,
                "passed": stratum_passed,
            })
    selected = (
        stable_winners[0]
        if len(stable_winners) == len(TRAINING_SEEDS) * len(DECODE_MODES)
        and len(set(stable_winners)) == 1
        else None
    )
    checks["same_stable_winner_every_seed_and_decode_mode"] = selected is not None
    if selected is None:
        checks["selected_candidate_passes_train_dev_ood_all_seeds"] = False
    else:
        checks["selected_candidate_passes_train_dev_ood_all_seeds"] = all(
            trust_status[(selected, seed)]["passed"] for seed in TRAINING_SEEDS
        )

    reward_statistics = {}
    throughput = {}
    for candidate in candidates:
        reward_statistics[candidate] = {}
        throughput[candidate] = {}
        for mode in DECODE_MODES:
            mode_cells = [
                receipt for receipt in receipts.values()
                if receipt["candidate_checkpoint_sha256"] == candidate
                and receipt["decode_mode"] == mode
            ]
            dev_rewards = [
                receipt["unit_balanced_reward"]
                for receipt in mode_cells if receipt["role"] == "dev"
            ]
            reward_statistics[candidate][mode] = _summary(dev_rewards)
            seconds = math.fsum(receipt["wall_seconds"] for receipt in mode_cells)
            tokens = sum(
                receipt["realized_generated_tokens"] for receipt in mode_cells
            )
            throughput[candidate][mode] = {
                "actor_generated_tokens": tokens,
                "actor_wall_seconds_sum": seconds,
                "actor_tokens_per_second": tokens / seconds,
            }

    passed = all(checks.values())
    result = {
        "schema": "specialist-greedy-stochastic-robustness-analysis-v68",
        "plan_sha256": plan["plan_sha256"],
        "policy_semantics_sha256": trust_v67.POLICY_SEMANTICS_SHA256,
        "cell_receipts": len(receipts),
        "protocol_failures": protocol_failures,
        "greedy_identity_failures": greedy_identity_failures,
        "observed_candidate_mode_budgets": observed_budgets,
        "rank_stability": rank_summaries,
        "reward_statistics": reward_statistics,
        "throughput": throughput,
        "trust_region_status": {
            candidate: {
                str(seed): trust_status[(candidate, seed)]
                for seed in TRAINING_SEEDS
            }
            for candidate in candidates
        },
        "hard_gate_checks": dict(sorted(checks.items())),
        "failed_hard_gates": sorted(
            name for name, check in checks.items() if not check
        ),
        "selected_candidate_checkpoint_sha256": selected,
        "all_hard_gates_passed": passed,
        "promotion_eligible": passed,
        "selection_uses_modal_replicated_ranking_not_largest_outlier_mean": True,
        "raw_prompt_output_or_token_ids_persisted": False,
        "protected_access_count": 0,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def require_selection(result: dict) -> str:
    raise RuntimeError(
        "V68 selection is transitively bound to quarantined evaluation V1 and "
        "is disabled"
    )
    compact = {
        key: value for key, value in result.items()
        if key != "content_sha256_before_self_field"
    }
    checks = result.get("hard_gate_checks")
    passed = (
        isinstance(checks, dict) and bool(checks)
        and all(isinstance(value, bool) for value in checks.values())
        and all(checks.values())
    )
    selected = result.get("selected_candidate_checkpoint_sha256")
    if (
        result.get("schema")
        != "specialist-greedy-stochastic-robustness-analysis-v68"
        or result.get("policy_semantics_sha256")
        != trust_v67.POLICY_SEMANTICS_SHA256
        or result.get("content_sha256_before_self_field")
        != canonical_sha256(compact)
        or result.get("all_hard_gates_passed") is not passed
        or result.get("promotion_eligible") is not passed
        or result.get("raw_prompt_output_or_token_ids_persisted") is not False
        or result.get("protected_access_count") != 0
        or (passed and (not isinstance(selected, str) or _HEX64.fullmatch(selected) is None))
    ):
        raise RuntimeError("invalid or mutated decode-robustness analysis")
    if not passed:
        raise RuntimeError("decode-robustness candidate failed a hard gate")
    return selected


def build_preregistration() -> dict:
    raise RuntimeError(
        "V68 preregistration is historical/nonpromotable; create a V2 successor"
    )
    trust_prereg = _load_trust_preregistration()
    contract = trust_v67.load_evaluation_contract()
    value = {
        "schema": "specialist-greedy-seeded-stochastic-preregistration-v68",
        "status": "sealed_before_any_greedy_or_stochastic_candidate_output",
        "created_at_utc": "2026-07-17T00:00:00+00:00",
        "purpose": (
            "Determine whether a greedy-trained or seeded-stochastic-trained "
            "candidate remains selected under replicated paired decoding."
        ),
        "evaluation_contract": {
            "content_sha256": contract["content_sha256_before_self_field"],
            "confirmation_training_seeds": list(TRAINING_SEEDS),
            "unregistered_seed_retry": "prohibited",
        },
        "multiobjective_trust_contract": {
            "path": str(TRUST_PREREGISTRATION),
            "file_sha256": EXPECTED_TRUST_PREREGISTRATION_FILE_SHA256,
            "content_sha256": trust_prereg["content_sha256_before_self_field"],
            "policy_semantics_sha256": trust_v67.POLICY_SEMANTICS_SHA256,
        },
        "decode_modes": {
            mode: {
                **dict(_COMMON_DECODE),
                **_MODE_VALUES[mode],
                "seed": "derived by the frozen candidate-independent rule",
            }
            for mode in DECODE_MODES
        },
        "pairing": {
            "training_seeds": list(TRAINING_SEEDS),
            "repeats_per_mode": REPEATS_PER_MODE,
            "prompt_role_order": list(PROMPT_ROLE_ORDER),
            "candidate_identity_in_seed_or_payload": False,
            "same_prompt_block_decode_seed_and_judge_each_candidate_cell": True,
            "seed_derivation": (
                "SHA-256(evaluation contract, training seed, decode mode, "
                "repeat, role, prompt-block hash), truncated to nonnegative 63-bit"
            ),
        },
        "generated_token_budget": {
            "tokens_per_completion": TOKENS_PER_COMPLETION,
            "min_tokens_equals_max_tokens": True,
            "ignore_eos": True,
            "early_stop_count_required": 0,
            "finish_reason_length_count_must_equal_completions": True,
            "realized_tokens_must_equal_scheduled_tokens": True,
            "candidate_and_mode_totals_must_be_exactly_equal": True,
        },
        "rank_and_selection_stability": {
            "ranking_role": "dev",
            "minimum_adjacent_rank_margin": MINIMUM_ADJACENT_RANK_MARGIN,
            "greedy_repeats_must_have_identical_outputs_judges_and_ranking": True,
            "stochastic_modal_full_ranking_repeats_minimum": (
                STOCHASTIC_MODAL_RANKING_REPEATS_MINIMUM
            ),
            "stochastic_median_pairwise_spearman_minimum": (
                STOCHASTIC_MEDIAN_PAIRWISE_SPEARMAN_MINIMUM
            ),
            "same_winner_required_every_seed_and_mode": True,
            "selection_statistic": "modal replicated ranking; never largest mean outlier",
        },
        "promotion": {
            "selected_candidate_train_dev_ood_trust_must_pass_each_seed": True,
            "weighted_aggregate_cannot_mask_any_protocol_rank_or_trust_gate": True,
            "protected_terminal_access_before_selection": "prohibited",
            "throughput_is_reported_but_never_a_quality_override": True,
        },
        "receipt_firewall": {
            "accepted_candidate_evidence": (
                "hashes, counts, timings, numeric unit-balanced rewards, and "
                "V67 aggregate trust receipts only"
            ),
            "raw_prompt_generated_text_token_ids_or_per_item_metrics": "prohibited",
            "raw_output_access_count_required": 0,
            "protected_access_count_required": 0,
            "unsealed_seed_or_payload": "hard error",
        },
        "adversarial_cases": [
            "greedy reward tie blocks arbitrary lexicographic selection",
            "greedy output or judge nondeterminism blocks promotion",
            "candidate-specific or unsealed seed is rejected",
            "early stop or unequal realized token budget blocks promotion",
            "one stochastic reward outlier cannot override the modal ranking",
            "aggregate reward cannot mask one seed, rank, item, or OOD hard failure",
        ],
        "implementation": {
            "module": str(Path(__file__).resolve()),
            "module_file_sha256": file_sha256(Path(__file__).resolve()),
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def validate_preregistration(value: dict) -> None:
    raise RuntimeError(
        "V68 historical preregistration is nonpromotable after V1 quarantine"
    )
    if value != build_preregistration():
        raise RuntimeError("decode-robustness preregistration changed")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--check", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    value = build_preregistration()
    if args.check:
        validate_preregistration(json.loads(
            PREREGISTRATION.read_text(encoding="utf-8")
        ))
    else:
        print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
