#!/usr/bin/env python3
"""Deterministic mirrored-ES planning and fail-closed wave execution.

The production LoRA worker owns numerical state.  This module owns the
estimator and scheduling contract: both signs of a direction see exactly the
same prompt/decode/judge payload, every four-actor wave is complete, and every
actor is unconditionally restored after all submitted work has drained.

The module deliberately has no torch, Ray, vLLM, dataset, or GPU imports so its
state-machine and estimator algebra can be tested before a Qwen calibration.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


SCHEMA_V66 = "mirrored-es-common-random-numbers-v66"
REQUIRED_ENGINES_V66 = 4
SIGNS_V66 = (1, -1)


class MirroredExecutionErrorV66(RuntimeError):
    """A wave failed, but every actor was restored successfully."""


class MirroredRestoreErrorV66(RuntimeError):
    """At least one actor could not prove an exact restore; stop the run."""


def canonical_sha256_v66(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _exact_nonnegative_int_v66(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"v66 {label} must be an exact integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"v66 {label} must be nonnegative")
    return result


def _json_roundtrip_v66(value: Any, label: str) -> Any:
    """Reject non-JSON, nonfinite, or lossy evaluation-contract values."""
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        reopened = json.loads(payload)
    except (TypeError, ValueError) as error:
        raise ValueError(f"v66 {label} must be finite canonical JSON") from error
    if reopened != value:
        raise ValueError(f"v66 {label} changes under a JSON round trip")
    return reopened


def common_evaluation_payload_v66(
    prompt_block: Sequence[Any],
    decode_contract: Mapping[str, Any],
    judge_contract: Mapping[str, Any],
    evaluation_seed: int,
) -> dict:
    """Seal one payload that is shared byte-for-byte by both signs.

    ``prompt_block`` may contain token-id requests or compact row identities.
    The executor passes this same immutable value to every signed candidate;
    only model state is allowed to differ within a mirrored pair.
    """
    prompts = _json_roundtrip_v66(list(prompt_block), "prompt block")
    decode = _json_roundtrip_v66(dict(decode_contract), "decode contract")
    judge = _json_roundtrip_v66(dict(judge_contract), "judge contract")
    seed = _exact_nonnegative_int_v66(evaluation_seed, "evaluation seed")
    if not prompts:
        raise ValueError("v66 prompt block must not be empty")
    if not decode or not judge:
        raise ValueError("v66 decode and judge contracts must not be empty")
    prompt_sha = canonical_sha256_v66(prompts)
    decode_sha = canonical_sha256_v66(decode)
    judge_sha = canonical_sha256_v66(judge)
    contract = {
        "schema": SCHEMA_V66,
        "prompt_block_sha256": prompt_sha,
        "prompt_count": len(prompts),
        "decode_contract_sha256": decode_sha,
        "judge_contract_sha256": judge_sha,
        "evaluation_seed": seed,
        "same_prompt_block_for_both_signs": True,
        "same_decode_seed_and_parameters_for_both_signs": True,
        "same_judge_for_both_signs": True,
    }
    contract["evaluation_contract_sha256"] = canonical_sha256_v66(contract)
    return {
        "schema": "mirrored-es-evaluation-payload-v66",
        "prompts": prompts,
        "decode_contract": decode,
        "judge_contract": judge,
        "contract": contract,
    }


def validate_evaluation_payload_v66(payload: Mapping[str, Any]) -> dict:
    if not isinstance(payload, Mapping):
        raise ValueError("v66 evaluation payload must be a mapping")
    expected = common_evaluation_payload_v66(
        payload.get("prompts", []),
        payload.get("decode_contract", {}),
        payload.get("judge_contract", {}),
        payload.get("contract", {}).get("evaluation_seed", -1),
    )
    if dict(payload) != expected:
        raise RuntimeError("v66 evaluation payload identity changed")
    return expected


def mirrored_population_plan_v66(
    direction_seeds: Sequence[int],
    sigma: float,
    evaluation_payload: Mapping[str, Any],
    engine_count: int = REQUIRED_ENGINES_V66,
) -> dict:
    """Pair directions into full four-engine waves with rotating signs.

    A production plan uses a direction count divisible by four.  Each wave
    evaluates two complete +/- pairs.  Rotating signs between ranks prevents a
    persistent GPU/rank effect from being confounded with perturbation sign.
    """
    if isinstance(engine_count, bool) or int(engine_count) != REQUIRED_ENGINES_V66:
        raise ValueError("v66 mirrored execution requires exactly four engines")
    payload = validate_evaluation_payload_v66(evaluation_payload)
    seeds = [
        _exact_nonnegative_int_v66(seed, "direction seed")
        for seed in direction_seeds
    ]
    if len(seeds) < REQUIRED_ENGINES_V66 or len(seeds) % REQUIRED_ENGINES_V66:
        raise ValueError("v66 direction count must be a positive multiple of four")
    if len(set(seeds)) != len(seeds):
        raise ValueError("v66 direction seeds must be unique")
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("v66 sigma must be finite and positive")

    contract = payload["contract"]
    waves = []
    assignments = []
    for wave_index, first_direction in enumerate(range(0, len(seeds), 2)):
        wave = []
        sign_order = SIGNS_V66 if wave_index % 2 == 0 else tuple(reversed(SIGNS_V66))
        for pair_offset, direction_index in enumerate(
            (first_direction, first_direction + 1)
        ):
            pair_id = canonical_sha256_v66({
                "schema": "mirrored-es-pair-id-v66",
                "direction_index": direction_index,
                "seed": seeds[direction_index],
                "sigma": sigma,
                "evaluation_contract_sha256": contract[
                    "evaluation_contract_sha256"
                ],
            })
            for sign_offset, sign in enumerate(sign_order):
                engine_rank = pair_offset * 2 + sign_offset
                assignment = {
                    "schema": "mirrored-es-signed-assignment-v66",
                    "wave_index": wave_index,
                    "engine_rank": engine_rank,
                    "direction_index": direction_index,
                    "direction_seed": seeds[direction_index],
                    "sigma": sigma,
                    "sign": sign,
                    "sign_label": "plus" if sign == 1 else "minus",
                    "pair_id": pair_id,
                    "evaluation_contract_sha256": contract[
                        "evaluation_contract_sha256"
                    ],
                    "prompt_block_sha256": contract["prompt_block_sha256"],
                    "decode_contract_sha256": contract[
                        "decode_contract_sha256"
                    ],
                    "judge_contract_sha256": contract[
                        "judge_contract_sha256"
                    ],
                    "evaluation_seed": contract["evaluation_seed"],
                }
                wave.append(assignment)
                assignments.append(assignment)
        if {item["engine_rank"] for item in wave} != set(range(4)):
            raise RuntimeError("v66 constructed an incomplete four-engine wave")
        waves.append(wave)

    per_rank = {
        rank: [item for item in assignments if item["engine_rank"] == rank]
        for rank in range(4)
    }
    expected_per_rank = len(assignments) // 4
    if any(len(items) != expected_per_rank for items in per_rank.values()):
        raise RuntimeError("v66 signed candidate dispatch is not rank-balanced")
    if any(
        sum(item["sign"] == 1 for item in items)
        != sum(item["sign"] == -1 for item in items)
        for items in per_rank.values()
    ):
        raise RuntimeError("v66 perturbation signs are not balanced per rank")
    plan = {
        "schema": "mirrored-es-four-engine-plan-v66",
        "direction_seeds": seeds,
        "direction_count": len(seeds),
        "signed_population_size": 2 * len(seeds),
        "sigma": sigma,
        "engine_count": 4,
        "wave_count": len(waves),
        "candidates_per_engine": expected_per_rank,
        "evaluation_contract": contract,
        "waves": waves,
        "paired_signs_are_concurrent_within_wave": True,
        "rank_sign_balance": True,
    }
    plan["plan_sha256"] = canonical_sha256_v66(plan)
    return plan


def validate_mirrored_plan_v66(
    plan: Mapping[str, Any], evaluation_payload: Mapping[str, Any]
) -> dict:
    if not isinstance(plan, Mapping):
        raise ValueError("v66 mirrored plan must be a mapping")
    expected = mirrored_population_plan_v66(
        plan.get("direction_seeds", []),
        plan.get("sigma", float("nan")),
        evaluation_payload,
        plan.get("engine_count", -1),
    )
    if dict(plan) != expected:
        raise RuntimeError("v66 mirrored population plan identity changed")
    return expected


def pair_difference_update_v66(
    plan: Mapping[str, Any],
    signed_rewards: Sequence[Mapping[str, Any]],
    learning_rate: float,
) -> dict:
    """Compile rewards into the canonical central-difference ES update.

    The V41A worker applies ``alpha / population_size`` to the supplied noise
    coefficients.  Setting ``alpha = learning_rate / (2*sigma)`` and using one
    coefficient ``reward_plus - reward_minus`` per direction therefore yields
    ``learning_rate/(2*N*sigma) * sum(delta_reward * epsilon)``.
    """
    if plan.get("schema") != "mirrored-es-four-engine-plan-v66":
        raise ValueError("v66 pair-difference update requires a mirrored plan")
    sealed_plan = dict(plan)
    observed_plan_sha = sealed_plan.pop("plan_sha256", None)
    if (
        not isinstance(observed_plan_sha, str)
        or canonical_sha256_v66(sealed_plan) != observed_plan_sha
    ):
        raise RuntimeError("v66 pair-difference plan identity changed")
    learning_rate = float(learning_rate)
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("v66 learning rate must be finite and positive")
    assignments = [item for wave in plan["waves"] for item in wave]
    expected_keys = {
        (item["pair_id"], item["sign"]): item for item in assignments
    }
    observed = {}
    for record in signed_rewards:
        if not isinstance(record, Mapping):
            raise ValueError("v66 signed reward must be a mapping")
        key = (record.get("pair_id"), record.get("sign"))
        reward = float(record.get("reward", float("nan")))
        if key not in expected_keys or key in observed or not math.isfinite(reward):
            raise ValueError("v66 signed reward coverage or value changed")
        assignment = expected_keys[key]
        for field in (
            "direction_index",
            "direction_seed",
            "evaluation_contract_sha256",
        ):
            if record.get(field) != assignment[field]:
                raise RuntimeError(f"v66 signed reward {field} changed")
        observed[key] = reward
    if set(observed) != set(expected_keys):
        raise RuntimeError("v66 signed reward matrix is incomplete")

    pairs = []
    coefficients = []
    for direction_index, seed in enumerate(plan["direction_seeds"]):
        matching = [
            item for item in assignments
            if item["direction_index"] == direction_index
        ]
        pair_ids = {item["pair_id"] for item in matching}
        if len(matching) != 2 or len(pair_ids) != 1:
            raise RuntimeError("v66 direction does not have exactly one signed pair")
        pair_id = next(iter(pair_ids))
        plus = observed[(pair_id, 1)]
        minus = observed[(pair_id, -1)]
        difference = plus - minus
        coefficients.append(difference)
        pairs.append({
            "direction_index": direction_index,
            "direction_seed": seed,
            "pair_id": pair_id,
            "reward_plus": plus,
            "reward_minus": minus,
            "pair_difference": difference,
            "evaluation_contract_sha256": plan["evaluation_contract"][
                "evaluation_contract_sha256"
            ],
        })
    worker_alpha = learning_rate / (2.0 * float(plan["sigma"]))
    population_size = len(coefficients)
    result = {
        "schema": "mirrored-es-pair-difference-update-v66",
        "plan_sha256": plan["plan_sha256"],
        "direction_seeds": list(plan["direction_seeds"]),
        "coefficients": coefficients,
        "pairs": pairs,
        "worker_alpha": worker_alpha,
        "worker_population_size": population_size,
        "effective_noise_scale": worker_alpha / population_size,
        "estimator": "learning_rate/(2*N*sigma)*sum((Rplus-Rminus)*epsilon)",
        "unpaired_reward_centering_used": False,
    }
    result["coefficient_sha256"] = canonical_sha256_v66({
        "seeds": result["direction_seeds"],
        "coefficients": coefficients,
    })
    return result


def _drain_submitted_v66(
    submitted: Iterable[tuple[dict, Any]],
    resolve_one: Callable[[Any], Any],
) -> tuple[list[tuple[dict, Any]], list[tuple[dict, BaseException]]]:
    values = []
    errors = []
    for assignment, handle in submitted:
        try:
            values.append((assignment, resolve_one(handle)))
        except BaseException as error:  # cleanup must also follow interrupts
            errors.append((assignment, error))
    return values, errors


def _submit_complete_wave_v66(
    wave: Sequence[dict], submit: Callable[[dict], Any]
) -> tuple[list[tuple[dict, Any]], list[tuple[dict, BaseException]]]:
    submitted = []
    errors = []
    for assignment in wave:
        try:
            submitted.append((assignment, submit(assignment)))
        except BaseException as error:
            errors.append((assignment, error))
    return submitted, errors


def execute_mirrored_plan_v66(
    plan: Mapping[str, Any],
    evaluation_payload: Mapping[str, Any],
    submit_materialize: Callable[[dict], Any],
    submit_evaluate: Callable[[dict, dict], Any],
    submit_restore: Callable[[int, str], Any],
    resolve_one: Callable[[Any], Any],
) -> dict:
    """Execute full waves while making exact restoration non-optional.

    All four materializations are submitted before any is resolved, and all
    four evaluations are likewise submitted before resolution.  Resolution
    drains every handle even after an error.  Finally, restore RPCs are sent to
    every actor (including actors whose materialization status is uncertain)
    and fully drained before an error is propagated.
    """
    plan = validate_mirrored_plan_v66(plan, evaluation_payload)
    payload = validate_evaluation_payload_v66(evaluation_payload)
    contract_sha = payload["contract"]["evaluation_contract_sha256"]
    signed_rewards = []
    materialization_receipts = []
    restore_receipts = []
    for wave in plan["waves"]:
        primary_errors: list[tuple[str, dict, BaseException]] = []
        restore_errors: list[tuple[int, BaseException]] = []
        try:
            submitted, submit_errors = _submit_complete_wave_v66(
                wave, submit_materialize
            )
            primary_errors.extend(
                ("materialize_submit", item, error)
                for item, error in submit_errors
            )
            values, resolve_errors = _drain_submitted_v66(submitted, resolve_one)
            primary_errors.extend(
                ("materialize_resolve", item, error)
                for item, error in resolve_errors
            )
            for assignment, receipt in values:
                if not isinstance(receipt, Mapping) or any(
                    receipt.get(field) != assignment[field]
                    for field in (
                        "pair_id", "direction_seed", "sign",
                        "evaluation_contract_sha256",
                    )
                ):
                    primary_errors.append((
                        "materialize_receipt",
                        assignment,
                        RuntimeError("v66 materialization receipt changed"),
                    ))
                else:
                    materialization_receipts.append(dict(receipt))

            if not primary_errors:
                submitted_eval, eval_submit_errors = _submit_complete_wave_v66(
                    wave,
                    lambda assignment: submit_evaluate(assignment, payload),
                )
                primary_errors.extend(
                    ("evaluate_submit", item, error)
                    for item, error in eval_submit_errors
                )
                evaluated, eval_resolve_errors = _drain_submitted_v66(
                    submitted_eval, resolve_one
                )
                primary_errors.extend(
                    ("evaluate_resolve", item, error)
                    for item, error in eval_resolve_errors
                )
                for assignment, reward_value in evaluated:
                    try:
                        reward = float(reward_value)
                    except (TypeError, ValueError) as error:
                        primary_errors.append(("reward", assignment, error))
                        continue
                    if not math.isfinite(reward):
                        primary_errors.append((
                            "reward",
                            assignment,
                            ValueError("v66 reward must be finite"),
                        ))
                        continue
                    signed_rewards.append({
                        "pair_id": assignment["pair_id"],
                        "sign": assignment["sign"],
                        "direction_index": assignment["direction_index"],
                        "direction_seed": assignment["direction_seed"],
                        "evaluation_contract_sha256": contract_sha,
                        "reward": reward,
                    })
        except BaseException as error:
            primary_errors.append(("executor", wave[0], error))
        finally:
            restore_submitted = []
            for rank in range(REQUIRED_ENGINES_V66):
                try:
                    restore_submitted.append((
                        rank,
                        submit_restore(rank, f"wave_{wave[0]['wave_index']}_finalize"),
                    ))
                except BaseException as error:
                    restore_errors.append((rank, error))
            for rank, handle in restore_submitted:
                try:
                    receipt = resolve_one(handle)
                    if (
                        not isinstance(receipt, Mapping)
                        or receipt.get("restored") is not True
                        or receipt.get("terminal_poisoned") is not False
                    ):
                        raise RuntimeError("v66 exact restore receipt changed")
                    restore_receipts.append(dict(receipt))
                except BaseException as error:
                    restore_errors.append((rank, error))

        if restore_errors:
            detail = ", ".join(
                f"rank {rank}: {type(error).__name__}: {error}"
                for rank, error in restore_errors
            )
            raise MirroredRestoreErrorV66(
                f"v66 wave {wave[0]['wave_index']} failed closed during restore: "
                f"{detail}"
            ) from restore_errors[0][1]
        if primary_errors:
            phase, assignment, error = primary_errors[0]
            raise MirroredExecutionErrorV66(
                f"v66 wave {assignment['wave_index']} failed at {phase}; "
                "all four actors restored"
            ) from error

    if (
        len(signed_rewards) != plan["signed_population_size"]
        or len(restore_receipts) != plan["wave_count"] * 4
    ):
        raise RuntimeError("v66 completed execution coverage changed")
    return {
        "schema": "mirrored-es-execution-complete-v66",
        "plan_sha256": plan["plan_sha256"],
        "evaluation_contract_sha256": contract_sha,
        "signed_rewards": signed_rewards,
        "materialization_receipts": materialization_receipts,
        "restore_receipts": restore_receipts,
        "all_submitted_work_drained": True,
        "all_four_actors_restored_after_every_wave": True,
        "prompt_decode_judge_payload_shared_across_signs": True,
    }
