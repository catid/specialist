#!/usr/bin/env python3
"""Additive V73B runner using exact same-live reward update equivalence.

V73 correctly failed closed when a new live execution did not reproduce the
accepted V66d floating answer-logprob values bit for bit.  Cross-run reward
floats are measurements, not update-implementation inputs suitable for an
identity gate.  This adapter preserves V73's candidate/audit/accept/abort and
telemetry lifecycle while comparing two independent update compilers against
the *same* newly observed reward vector and requiring four-actor exact output
identity consensus.  It never commits, checkpoints, promotes, or opens
protected data.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import build_lora_es_v71_v72_same_live_preregistration_v73b as builder
import run_lora_es_mirrored_calibration_v66 as v66
import run_lora_es_v71_v72_live_calibration_v73 as v73


PREREGISTRATION = builder.OUTPUT
RUN_DIR = builder.RUN
_ARTIFACTS = builder.artifacts_v73b()
ATTEMPT = Path(_ARTIFACTS["attempt"]).resolve()
GPU_LOG = Path(_ARTIFACTS["gpu_log"]).resolve()
GPU_WORK_LOG = Path(_ARTIFACTS["actor_cuda_work_log"]).resolve()
HOST_SAMPLES = Path(_ARTIFACTS["host_process_samples"]).resolve()
HOST_SUMMARY = Path(_ARTIFACTS["host_process_summary"]).resolve()
POPULATION = Path(_ARTIFACTS["population"]).resolve()
UPDATE = Path(_ARTIFACTS["update"]).resolve()
AUDIT_TRAFFIC = Path(_ARTIFACTS["audit_traffic"]).resolve()
EQUIVALENCE = Path(_ARTIFACTS["equivalence"]).resolve()
REPORT = Path(_ARTIFACTS["report"]).resolve()
FAILURE = Path(_ARTIFACTS["failure"]).resolve()

_BASE_RPC_ALL_V73 = v73._rpc_all_v73


def artifacts_v73b() -> dict[str, str]:
    return dict(_ARTIFACTS)


def load_preregistration_v73b(args) -> dict:
    path = Path(args.preregistration).resolve()
    if v66.file_sha256_v66(path) != args.preregistration_sha256:
        raise RuntimeError("v73b preregistration file identity changed")
    value = json.loads(path.read_text(encoding="ascii"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or v66.mirrored.canonical_sha256_v66(compact)
        != args.preregistration_content_sha256
        or value != builder.build_preregistration_v73b()
        or value.get("schema")
        != "lora-es-v71-v72-qwen36-same-live-calibration-"
        "preregistration-v73b"
        or value.get("status")
        != "sealed_cpu_only_after_failed_v73_before_v73b_train_model_ray_gpu_"
        "or_protected_access"
        or value.get("artifacts") != artifacts_v73b()
    ):
        raise RuntimeError("v73b preregistration content changed")
    return value


def independent_pair_difference_update_v73b(
    plan: Mapping,
    signed_rewards: Sequence[Mapping],
    learning_rate: float,
) -> dict:
    """Independently compile the V66 estimator with a one-pass pair table."""
    if not isinstance(plan, Mapping) or plan.get("schema") != (
        "mirrored-es-four-engine-plan-v66"
    ):
        raise ValueError("v73b pair compiler requires a mirrored V66 plan")
    sealed = dict(plan)
    observed_plan_sha = sealed.pop("plan_sha256", None)
    if (
        not isinstance(observed_plan_sha, str)
        or v66.mirrored.canonical_sha256_v66(sealed) != observed_plan_sha
    ):
        raise RuntimeError("v73b mirrored plan identity changed")
    learning_rate = float(learning_rate)
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("v73b learning rate must be finite and positive")

    assignments = [item for wave in plan["waves"] for item in wave]
    by_key = {}
    direction_pair_ids: dict[int, set[str]] = {}
    for assignment in assignments:
        key = (assignment["pair_id"], assignment["sign"])
        if key in by_key or assignment["sign"] not in {-1, 1}:
            raise RuntimeError("v73b plan signed assignment matrix changed")
        by_key[key] = assignment
        direction_pair_ids.setdefault(
            int(assignment["direction_index"]), set()
        ).add(str(assignment["pair_id"]))

    observed = {}
    for record in signed_rewards:
        if not isinstance(record, Mapping):
            raise ValueError("v73b signed reward must be a mapping")
        key = (record.get("pair_id"), record.get("sign"))
        raw_reward = record.get("reward")
        if isinstance(raw_reward, bool):
            raise ValueError("v73b boolean reward is invalid")
        try:
            reward = float(raw_reward)
        except (TypeError, ValueError) as error:
            raise ValueError("v73b reward is not numeric") from error
        if key not in by_key or key in observed or not math.isfinite(reward):
            raise ValueError("v73b signed reward coverage or value changed")
        assignment = by_key[key]
        for field in (
            "direction_index", "direction_seed",
            "evaluation_contract_sha256",
        ):
            if record.get(field) != assignment[field]:
                raise RuntimeError(
                    f"v73b signed reward {field} changed"
                )
        observed[key] = reward
    if set(observed) != set(by_key):
        raise RuntimeError("v73b signed reward matrix is incomplete")

    pairs = []
    coefficients = []
    seeds = list(plan["direction_seeds"])
    if set(direction_pair_ids) != set(range(len(seeds))):
        raise RuntimeError("v73b direction coverage changed")
    for direction_index, seed in enumerate(seeds):
        pair_ids = direction_pair_ids[direction_index]
        if len(pair_ids) != 1:
            raise RuntimeError("v73b direction pair identity changed")
        pair_id = next(iter(pair_ids))
        if (pair_id, 1) not in observed or (pair_id, -1) not in observed:
            raise RuntimeError("v73b direction does not have both signs")
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
        "direction_seeds": seeds,
        "coefficients": coefficients,
        "pairs": pairs,
        "worker_alpha": worker_alpha,
        "worker_population_size": population_size,
        "effective_noise_scale": worker_alpha / population_size,
        "estimator": (
            "learning_rate/(2*N*sigma)*sum((Rplus-Rminus)*epsilon)"
        ),
        "unpaired_reward_centering_used": False,
    }
    result["coefficient_sha256"] = v66.mirrored.canonical_sha256_v66({
        "seeds": seeds,
        "coefficients": coefficients,
    })
    return result


_POPULATION_NON_REWARD_FIELDS_V73B = tuple(
    field for field in v73._POPULATION_EQUIVALENCE_FIELDS_V73
    if field not in {"signed_rewards", "signed_reward_sha256"}
)


def _reward_metadata_v73b(row: Mapping) -> dict:
    return {key: value for key, value in row.items() if key != "reward"}


def _pair_sign_v73b(rewards: Sequence[Mapping]) -> dict[str, int]:
    values: dict[str, dict[int, float]] = {}
    for row in rewards:
        values.setdefault(str(row["pair_id"]), {})[int(row["sign"])] = float(
            row["reward"]
        )
    result = {}
    for pair_id, signs in values.items():
        if set(signs) != {-1, 1}:
            raise RuntimeError("v73b paired reward sign coverage changed")
        difference = signs[1] - signs[-1]
        result[pair_id] = (difference > 0.0) - (difference < 0.0)
    return result


def population_equivalence_v73b(candidate: Mapping, control: Mapping) -> dict:
    changed = [
        field for field in _POPULATION_NON_REWARD_FIELDS_V73B
        if candidate.get(field) != control.get(field)
    ]
    if changed:
        raise RuntimeError(
            f"v73b non-reward population differs from accepted V66d: {changed}"
        )
    candidate_rewards = candidate.get("signed_rewards")
    control_rewards = control.get("signed_rewards")
    if (
        not isinstance(candidate_rewards, list)
        or not isinstance(control_rewards, list)
        or len(candidate_rewards) != v73.SIGNED_CANDIDATES_V73
        or len(control_rewards) != v73.SIGNED_CANDIDATES_V73
        or [_reward_metadata_v73b(row) for row in candidate_rewards]
        != [_reward_metadata_v73b(row) for row in control_rewards]
    ):
        raise RuntimeError(
            "v73b live reward assignment metadata differs from accepted V66d"
        )
    live_hash = v66.mirrored.canonical_sha256_v66(candidate_rewards)
    if candidate.get("signed_reward_sha256") != live_hash:
        raise RuntimeError("v73b live signed reward digest changed")
    # The independent compiler provides the complete/finite/unique matrix gate
    # before any worker is allowed to accept these rewards.
    context = v73._ACTIVE_CONTEXT_V73
    if context is None:
        raise RuntimeError("v73b active integration context is absent")
    independent_pair_difference_update_v73b(
        candidate["plan"],
        candidate_rewards,
        context.preregistration["fixed_recipe"]["learning_rate"],
    )
    deltas = [
        float(live["reward"]) - float(old["reward"])
        for live, old in zip(candidate_rewards, control_rewards, strict=True)
    ]
    live_signs = _pair_sign_v73b(candidate_rewards)
    control_signs = _pair_sign_v73b(control_rewards)
    result = {
        "schema": "eggroll-es-same-live-population-equivalence-v73b",
        "exact_non_reward_fields": list(_POPULATION_NON_REWARD_FIELDS_V73B),
        "candidate_count": len(candidate_rewards),
        "candidate_identity_count": len(candidate["materializations"]),
        "candidate_and_runtime_identity_exact_to_accepted_v66d": True,
        "restore_identity_exact_to_accepted_v66d": True,
        "reward_assignment_metadata_exact_to_accepted_v66d": True,
        "live_reward_matrix_complete_finite_and_unique": True,
        "live_signed_reward_sha256": live_hash,
        "accepted_signed_reward_sha256": control["signed_reward_sha256"],
        "historical_reward_bit_exact": candidate_rewards == control_rewards,
        "historical_reward_values_are_diagnostic_only": True,
        "historical_reward_delta": {
            "maximum_absolute": max(abs(value) for value in deltas),
            "mean_absolute": math.fsum(abs(value) for value in deltas) / len(deltas),
            "l2": math.sqrt(math.fsum(value * value for value in deltas)),
            "exact_value_match_count": sum(value == 0.0 for value in deltas),
            "pair_preference_sign_match_count": sum(
                live_signs[key] == control_signs[key] for key in live_signs
            ),
            "pair_count": len(live_signs),
            "acceptance_threshold_applied": False,
        },
        "accepted_population_file_sha256": (
            builder.v73.CONTROL_FILES_V73["population"]["file_sha256"]
        ),
    }
    result["equivalence_sha256"] = v66.mirrored.canonical_sha256_v66(result)
    return result


def _population_accept_then_update_v73b(
    canonical_compiler,
    plan,
    signed_rewards,
    learning_rate,
) -> dict:
    context = v73._ACTIVE_CONTEXT_V73
    if (
        context is None
        or context.trainer is None
        or context.population_equivalence is None
        or context.population_acceptance is not None
    ):
        raise RuntimeError("v73b population acceptance dependency changed")
    context.phase.value = "candidate_audit_matrix_all_actors"
    started = time.monotonic_ns()
    matrices = v73._rpc_all_v73(
        context.trainer, "candidate_audit_matrix_v71"
    )
    audit_consensus = v73.candidate_audit_consensus_v73(matrices)
    ended = time.monotonic_ns()
    context.record_operation(
        "candidate_audit_matrix", started, ended,
        candidate_count=v73.SIGNED_CANDIDATES_V73,
    )
    context.candidate_audit_consensus = audit_consensus

    context.phase.value = "population_reward_acceptance_all_actors"
    started = time.monotonic_ns()
    acceptances = v73._rpc_ranked_v73(
        context.trainer,
        "accept_population_rewards_v71",
        [
            (item["candidate_audit_sha256"],)
            for item in audit_consensus["by_rank"]
        ],
    )
    population_acceptance = v73.population_acceptance_consensus_v73(
        acceptances, audit_consensus
    )
    ended = time.monotonic_ns()
    context.record_operation(
        "population_reward_acceptance", started, ended,
        actor_count=v73.WORLD_SIZE_V73,
    )
    context.population_acceptance = population_acceptance
    context.capture_host_boundary("population_rewards_accepted")

    started = time.monotonic_ns()
    canonical = canonical_compiler(plan, signed_rewards, learning_rate)
    independent = independent_pair_difference_update_v73b(
        plan, signed_rewards, learning_rate
    )
    if canonical != independent:
        changed = sorted(
            set(canonical) | set(independent),
            key=str,
        )
        changed = [
            key for key in changed if canonical.get(key) != independent.get(key)
        ]
        raise RuntimeError(
            f"v73b same-live update compilers differ: {changed}"
        )
    context.same_live_update = independent
    context.same_live_compiler_equivalence = {
        "schema": "eggroll-es-same-live-compiler-equivalence-v73b",
        "live_signed_reward_sha256": v66.mirrored.canonical_sha256_v66(
            signed_rewards
        ),
        "canonical_compiler": (
            "eggroll_es_mirrored_v66.pair_difference_update_v66"
        ),
        "independent_compiler": "one_pass_pair_table_v73b",
        "same_live_reward_object_used_for_both": True,
        "whole_result_mapping_exact": True,
        "coefficient_sha256": independent["coefficient_sha256"],
        "direction_count": len(independent["coefficients"]),
    }
    context.same_live_compiler_equivalence["equivalence_sha256"] = (
        v66.mirrored.canonical_sha256_v66(
            context.same_live_compiler_equivalence
        )
    )
    ended = time.monotonic_ns()
    context.record_operation(
        "same_live_pair_difference_update_math", started, ended,
        coefficient_sha256=independent["coefficient_sha256"],
    )
    return independent


def _rpc_all_v73b(trainer, method: str, args=()):
    values = _BASE_RPC_ALL_V73(trainer, method, args)
    if method != "execute_sharded_adapter_update_v41a":
        return values
    context = v73._ACTIVE_CONTEXT_V73
    if context is None or getattr(context, "same_live_update", None) is None:
        raise RuntimeError("v73b update execution preceded same-live compilation")
    candidate_hashes = {
        item.get("candidate_identity", {}).get("sha256") for item in values
    }
    runtime_hashes = {
        item.get("materialization", {}).get("runtime_values_sha256")
        for item in values
    }
    if (
        len(candidate_hashes) != 1
        or len(runtime_hashes) != 1
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in candidate_hashes | runtime_hashes
        )
        or candidate_hashes == {v66.MASTER_SHA256_V66}
        or runtime_hashes == {v66.MASTER_RUNTIME_SHA256_V66}
    ):
        raise RuntimeError(
            "v73b four-actor live update identity consensus changed"
        )
    context.live_update_execution = {
        "schema": "eggroll-es-four-actor-live-update-consensus-v73b",
        "actor_count": len(values),
        "candidate_master_sha256": next(iter(candidate_hashes)),
        "candidate_runtime_values_sha256": next(iter(runtime_hashes)),
        "candidate_differs_from_master": True,
        "runtime_differs_from_master": True,
        "all_actor_identities_exact": True,
    }
    context.live_update_execution["consensus_sha256"] = (
        v66.mirrored.canonical_sha256_v66(context.live_update_execution)
    )
    # The immutable V73 executor compares against two fields in its accepted
    # control.  Rebind only this private in-memory copy to the already-proved
    # four-actor consensus; the sealed V66d artifact and historical diagnostic
    # copy remain untouched.
    context.control["update"]["candidate_master_sha256"] = next(
        iter(candidate_hashes)
    )
    context.control["update"]["candidate_runtime_values_sha256"] = next(
        iter(runtime_hashes)
    )
    return values


_STATIC_UPDATE_FIELDS_V73B = (
    "prepared_rank_shards", "candidate_differs_from_master",
    "candidate_runtime_differs_from_master", "master_committed",
    "all_four_abort_receipts_exact",
    "checkpoint_snapshot_or_promotion_performed", "plan_sha256",
    "protected_dev_ood_or_holdout_opened",
)


def update_equivalence_v73b(candidate: Mapping, _control: Mapping) -> dict:
    context = v73._ACTIVE_CONTEXT_V73
    if (
        context is None
        or getattr(context, "same_live_update", None) is None
        or getattr(context, "same_live_compiler_equivalence", None) is None
        or getattr(context, "live_update_execution", None) is None
        or context.update_invariants is None
    ):
        raise RuntimeError("v73b same-live update evidence is incomplete")
    historical = context.historical_update_control
    changed = [
        field for field in _STATIC_UPDATE_FIELDS_V73B
        if candidate.get(field) != historical.get(field)
    ]
    if changed:
        raise RuntimeError(
            f"v73b static update semantics differ from accepted V66d: {changed}"
        )
    expected = context.same_live_update
    expected_fields = {
        "coefficient_l2": math.sqrt(math.fsum(
            value * value for value in expected["coefficients"]
        )),
        "nonzero_pair_differences": sum(
            value != 0.0 for value in expected["coefficients"]
        ),
        "direction_count": len(expected["coefficients"]),
        "worker_alpha": expected["worker_alpha"],
        "worker_population_size": expected["worker_population_size"],
        "effective_noise_scale": expected["effective_noise_scale"],
        "coefficient_sha256": expected["coefficient_sha256"],
    }
    numerical_changed = [
        field for field, value in expected_fields.items()
        if candidate.get(field) != value
    ]
    if numerical_changed:
        raise RuntimeError(
            f"v73b executed update differs from same-live compiler: "
            f"{numerical_changed}"
        )
    execution = context.live_update_execution
    if (
        candidate.get("candidate_master_sha256")
        != execution["candidate_master_sha256"]
        or candidate.get("candidate_runtime_values_sha256")
        != execution["candidate_runtime_values_sha256"]
        or candidate.get("manifest_sha256")
        != context.update_invariants["manifest_sha256"]
        or expected_fields["nonzero_pair_differences"] == 0
    ):
        raise RuntimeError("v73b live update execution evidence changed")
    result = {
        "schema": "eggroll-es-same-live-update-equivalence-v73b",
        "static_fields_exact_to_accepted_v66d": list(
            _STATIC_UPDATE_FIELDS_V73B
        ),
        "canonical_and_independent_compilers_exact": True,
        "compiler_equivalence_sha256": context.same_live_compiler_equivalence[
            "equivalence_sha256"
        ],
        "live_signed_reward_sha256": context.same_live_compiler_equivalence[
            "live_signed_reward_sha256"
        ],
        "coefficient_fields_exact_to_same_live_compiler": sorted(
            expected_fields
        ),
        "coefficient_sha256": expected["coefficient_sha256"],
        "prepared_shards_exact": True,
        "four_actor_candidate_and_runtime_identity_exact": True,
        "live_update_execution": execution,
        "abort_semantics_exact": True,
        "historical_reward_derived_update_identity_required": False,
        "historical_candidate_master_sha256": historical[
            "candidate_master_sha256"
        ],
        "historical_candidate_runtime_values_sha256": historical[
            "candidate_runtime_values_sha256"
        ],
        "historical_candidate_identity_match": (
            candidate["candidate_master_sha256"]
            == historical["candidate_master_sha256"]
        ),
        "historical_runtime_identity_match": (
            candidate["candidate_runtime_values_sha256"]
            == historical["candidate_runtime_values_sha256"]
        ),
        "accepted_update_file_sha256": (
            builder.v73.CONTROL_FILES_V73["update"]["file_sha256"]
        ),
    }
    result["equivalence_sha256"] = v66.mirrored.canonical_sha256_v66(result)
    return result


def _write_rehashed_v73b(path: Path, value: Mapping) -> dict:
    result = dict(value)
    result.pop("content_sha256_before_self_field", None)
    result["content_sha256_before_self_field"] = (
        v66.mirrored.canonical_sha256_v66(result)
    )
    payload = (
        json.dumps(result, ensure_ascii=True, allow_nan=False, indent=2,
                   sort_keys=True) + "\n"
    ).encode("ascii")
    temporary = path.with_name(f".{path.name}.tmp-v73b-{os.getpid()}")
    with temporary.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    return result


def _artifact_reference_v73b(path: Path, value: Mapping) -> dict:
    return {
        "path": str(path),
        "file_sha256": v66.file_sha256_v66(path),
        "content_sha256": value["content_sha256_before_self_field"],
    }


def finalize_success_artifacts_v73b(context) -> dict:
    population = json.loads(POPULATION.read_text(encoding="ascii"))
    population["schema"] = "v73b-v71-v72-qwen36-population-evidence"
    population["same_live_population_equivalence"] = population.pop(
        "v66d_population_equivalence"
    )
    population["v71_candidate_rewards_provisional"] = False
    population["v71_live_rewards_accepted_before_update_math"] = True
    population = _write_rehashed_v73b(POPULATION, population)

    update = json.loads(UPDATE.read_text(encoding="ascii"))
    update["schema"] = "v73b-v71-v72-qwen36-update-evidence"
    update["population_content_sha256"] = population[
        "content_sha256_before_self_field"
    ]
    update["same_live_update_equivalence"] = update.pop(
        "v66d_update_equivalence"
    )
    update["same_live_compiler_equivalence"] = (
        context.same_live_compiler_equivalence
    )
    update = _write_rehashed_v73b(UPDATE, update)

    equivalence = json.loads(EQUIVALENCE.read_text(encoding="ascii"))
    equivalence["schema"] = "eggroll-es-same-live-equivalence-v73b"
    equivalence["population"] = context.population_equivalence
    equivalence["update"] = context.update_equivalence
    equivalence.pop("candidate_reward_update_exact", None)
    equivalence.update({
        "candidate_runtime_restore_and_work_metadata_exact_to_v66d": True,
        "historical_reward_float_identity_required": False,
        "same_live_canonical_and_independent_compilers_exact": True,
        "four_actor_live_update_identity_consensus": True,
        "master_committed": False,
    })
    equivalence = _write_rehashed_v73b(EQUIVALENCE, equivalence)

    report = json.loads(REPORT.read_text(encoding="ascii"))
    report["schema"] = "v73b-v71-v72-qwen36-calibration-report"
    report["status"] = (
        "complete_same_live_update_equivalence_no_commit_profiled"
    )
    report["beads"] = [
        "specialist-0j5.29", "specialist-0j5.19",
        "specialist-0j5.21", "specialist-0j5.20",
    ]
    report["population"] = _artifact_reference_v73b(
        POPULATION, population
    )
    report["nonzero_update"] = _artifact_reference_v73b(UPDATE, update)
    report["same_live_equivalence"] = _artifact_reference_v73b(
        EQUIVALENCE, equivalence
    )
    report.pop("accepted_v66d_equivalence", None)
    report.update({
        "historical_reward_floats_used_as_acceptance_gate": False,
        "same_live_reward_vector_used_by_both_compilers": True,
        "same_live_compiler_output_whole_mapping_exact": True,
        "all_v71_rewards_accepted_before_update_math": True,
        "checkpoint_snapshot_or_promotion_performed": False,
        "protected_dev_ood_or_holdout_opened": False,
    })
    return _write_rehashed_v73b(REPORT, report)


def finalize_failure_artifact_v73b() -> None:
    if not FAILURE.is_file():
        return
    failure = json.loads(FAILURE.read_text(encoding="ascii"))
    failure["schema"] = "v73b-v71-v72-qwen36-calibration-failure"
    failure["historical_reward_floats_used_as_acceptance_gate"] = False
    failure["same_live_equivalence_required"] = True
    _write_rehashed_v73b(FAILURE, failure)


@contextmanager
def patched_live_v73b(preregistration: Mapping, control: Mapping):
    replacements = {
        "PREREGISTRATION": PREREGISTRATION,
        "RUN_DIR": RUN_DIR,
        "_ARTIFACTS": _ARTIFACTS,
        "ATTEMPT": ATTEMPT,
        "GPU_LOG": GPU_LOG,
        "GPU_WORK_LOG": GPU_WORK_LOG,
        "HOST_SAMPLES": HOST_SAMPLES,
        "HOST_SUMMARY": HOST_SUMMARY,
        "POPULATION": POPULATION,
        "UPDATE": UPDATE,
        "AUDIT_TRAFFIC": AUDIT_TRAFFIC,
        "EQUIVALENCE": EQUIVALENCE,
        "REPORT": REPORT,
        "FAILURE": FAILURE,
        "population_equivalence_v73": population_equivalence_v73b,
        "update_equivalence_v73": update_equivalence_v73b,
        "_population_accept_then_update_v73": (
            _population_accept_then_update_v73b
        ),
        "_rpc_all_v73": _rpc_all_v73b,
    }
    saved = {name: getattr(v73, name) for name in replacements}
    for name, value in replacements.items():
        setattr(v73, name, value)
    try:
        with v73.patched_live_v73(preregistration, control) as context:
            context.historical_update_control = copy.deepcopy(
                control["update"]
            )
            context.same_live_update = None
            context.same_live_compiler_equivalence = None
            context.live_update_execution = None
            yield context
    finally:
        for name, value in saved.items():
            setattr(v73, name, value)


def parser_v73b() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", default=str(PREREGISTRATION))
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--preregistration-content-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv=None) -> int:
    args = parser_v73b().parse_args(argv)
    if args.dry_run == args.execute:
        raise ValueError("v73b requires exactly one of --dry-run or --execute")
    preregistration = load_preregistration_v73b(args)
    worker_contract = v73.validate_lora_worker_contract_v73()
    if args.dry_run:
        recipe = preregistration["fixed_recipe"]
        print(json.dumps({
            "schema": preregistration["schema"],
            "model": "Qwen3.6-35B-A3B",
            "four_tp1_engines": True,
            "direction_count": len(recipe["direction_seeds"]),
            "signed_population_size": 2 * len(recipe["direction_seeds"]),
            "train_only_rows_per_candidate": 64,
            "expected_artifacts": artifacts_v73b(),
            "worker_contract": worker_contract,
            "accepted_v66d_control": preregistration[
                "accepted_v66d_control"
            ],
            "failed_v73_observation": preregistration[
                "failed_v73_observation"
            ],
            "dependency_order": recipe["integration_v73b"][
                "dependency_order"
            ],
            "same_live_reward_dual_compiler_equivalence": "whole_mapping_exact",
            "historical_reward_floats_are_diagnostic_only": True,
            "rank_local_population_and_update_tokens": True,
            "host_rss_numa_fault_and_phase_telemetry": True,
            "train_semantics_model_ray_or_gpu_loaded": False,
            "filesystem_writes": False,
            "protected_dev_ood_or_holdout_opened": False,
            "checkpoint_snapshot_commit_or_promotion_authorized": False,
        }, sort_keys=True))
        return 0
    if Path(sys.executable).absolute() != v66.REQUIRED_PYTHON_V66:
        raise RuntimeError(
            f"v73b requires {v66.REQUIRED_PYTHON_V66}; observed {sys.executable}"
        )
    control = v73.load_accepted_control_values_v73()
    try:
        with patched_live_v73b(preregistration, copy.deepcopy(control)) as context:
            result = v66.execute_v66(preregistration, args)
            finalize_success_artifacts_v73b(context)
            return result
    except BaseException:
        finalize_failure_artifact_v73b()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
