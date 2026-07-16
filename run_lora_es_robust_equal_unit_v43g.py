#!/usr/bin/env python3
"""Robust complete-actor-block, train-only matched LoRA EGGROLL-ES V43G.

V43G doubles exploration sigma while retaining update alpha, evaluates every
antithetic direction on all four actors, and applies EGGROLL centered-rank
fitness shaping to median signed scores.  The exact FP32 state protocol is
unchanged.  Only the frozen fold-3 training panel may be scored.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import eggroll_es_worker_v3 as worker_v3
import lora_es_robust_consensus_v43g as numeric
import run_lora_es_equal_unit_v43a as prior


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "v43g_matched_lora_es_fold3_pop8_robust"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
REPORT = (RUN_DIR / "matched_lora_es_report_v43g.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v43g.jsonl").resolve()
SNAPSHOT = (RUN_DIR / "adapter_step1_v43g").resolve()
CALIBRATION_ARTIFACT = (RUN_DIR / "numeric_calibration_v43g.json").resolve()
RELIABILITY_ARTIFACT = (RUN_DIR / "population_reliability_v43g.json").resolve()
POST_UPDATE_ARTIFACT = (RUN_DIR / "post_update_consensus_v43g.json").resolve()

SOURCE = prior.SOURCE
SOURCE_WEIGHTS = prior.SOURCE_WEIGHTS
SOURCE_CONFIG = prior.SOURCE_CONFIG
SOURCE_MANIFEST = prior.SOURCE_MANIFEST
STAGED = prior.STAGED
STAGED_WEIGHTS = prior.STAGED_WEIGHTS
STAGED_CONFIG = prior.STAGED_CONFIG
STAGED_MANIFEST = prior.STAGED_MANIFEST
DATASET = prior.DATASET
SPLIT_MANIFEST = prior.SPLIT_MANIFEST
DATASET_SHA256 = prior.DATASET_SHA256
SPLIT_MANIFEST_SHA256 = prior.SPLIT_MANIFEST_SHA256
TRAIN_BUNDLE_SHA256 = prior.TRAIN_BUNDLE_SHA256
WORKER_EXTENSION = prior.WORKER_EXTENSION
POPULATION_SIZE = prior.POPULATION_SIZE
SEEDS = list(prior.SEEDS)
SIGMA = 0.0006
ALPHA = prior.ALPHA
GLOBAL_SEED = prior.GLOBAL_SEED
STANDARDIZATION_EPSILON = prior.STANDARDIZATION_EPSILON
CALIBRATION_SIGMA = numeric.calibration_sigma_v43g(ALPHA, POPULATION_SIZE)

v40a = prior.v40a
v40c = prior.v40c
equal_v38 = prior.equal_v38
anchor_v4 = prior.anchor_v4


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--preregistration", required=True)
    result.add_argument("--preregistration-sha256", required=True)
    result.add_argument("--preregistration-content-sha256", required=True)
    result.add_argument("--dry-run", action="store_true")
    return result


def _paths() -> dict[str, Path]:
    return {
        "runtime": Path(__file__).resolve(),
        "builder": ROOT / "build_lora_es_robust_equal_unit_preregistration_v43g.py",
        "numeric_consensus": Path(numeric.__file__).resolve(),
        "numeric_consensus_v43f": Path(numeric.v43f.__file__).resolve(),
        "initialization_builder": ROOT / "build_matched_lora_initialization_v41a.py",
        "staging_runtime": ROOT / "stage_matched_lora_initialization_vllm_v41b.py",
        "worker": ROOT / "eggroll_es_worker_lora_v41a.py",
        "worker_v3": ROOT / "eggroll_es_worker_v3.py",
        "runtime_v40a": Path(v40a.__file__).resolve(),
        "resolver_runtime_v40c": Path(v40c.__file__).resolve(),
        "equal_unit_runtime_v38": Path(equal_v38.__file__).resolve(),
        "dense_reward_runtime_v4": Path(anchor_v4.__file__).resolve(),
        "source_weights": SOURCE_WEIGHTS,
        "source_config": SOURCE_CONFIG,
        "source_manifest": SOURCE_MANIFEST,
        "staged_weights": STAGED_WEIGHTS,
        "staged_config": STAGED_CONFIG,
        "staged_manifest": STAGED_MANIFEST,
        "dataset": DATASET,
        "split_manifest": SPLIT_MANIFEST,
        "model_config": v40a.MODEL / "config.json",
        "model_index": v40a.MODEL / "model.safetensors.index.json",
        "tuned_table": v40a.TUNED_FILE,
        "base_runtime": ROOT / "train_eggroll_es_specialist.py",
        "cleanup_runtime": ROOT / "run_eggroll_es_equal_unit_v38a.py",
    }


def implementation_bindings() -> dict[str, str]:
    result = {key: v40a.file_sha256(path) for key, path in _paths().items()}
    result["model_shards_content_sha256"] = v40a.MODEL_SHARDS_CONTENT_SHA256
    return result


def load_preregistration(args: argparse.Namespace) -> dict:
    path = Path(args.preregistration).resolve()
    if v40a.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("v43g preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    recipe = value.get("recipe", {})
    calibration = value.get("numeric_calibration", {})
    consensus = value.get("post_update_consensus", {})
    if (
        content != args.preregistration_content_sha256
        or content != v40a.canonical_sha256(compact)
        or value.get("schema") != "matched-lora-es-preregistration-v43g"
        or value.get("status") != "preregistered_before_train_only_launch"
        or value.get("shadow_dev_external_eval_ood_or_holdout_authorized") is not False
        or value.get("sealed_holdout_opened") is not False
        or value.get("quality_selection_or_promotion_authorized") is not False
        or value.get("access_contract", {}).get("protected_semantic_access")
        is not False
        or value.get("access_contract", {}).get("only_runtime_train_path_may_open")
        != str(DATASET)
        or value.get("implementation_bindings") != implementation_bindings()
        or recipe.get("population_size") != POPULATION_SIZE
        or recipe.get("seeds") != SEEDS
        or recipe.get("sigma") != SIGMA
        or recipe.get("alpha") != ALPHA
        or recipe.get("dataset") != str(DATASET)
        or recipe.get("dataset_sha256") != DATASET_SHA256
        or recipe.get("train_bundle_content_sha256") != TRAIN_BUNDLE_SHA256
        or recipe.get("matched_initialization") != str(SOURCE)
        or recipe.get("staged_initialization") != str(STAGED)
        or recipe.get("worker_extension") != WORKER_EXTENSION
        or calibration.get("synthetic_noise_seed")
        != numeric.CALIBRATION_NOISE_SEED_V43G
        or calibration.get("synthetic_sigma") != CALIBRATION_SIGMA
        or calibration.get("warmup_repeats") != numeric.CALIBRATION_WARMUPS_V43G
        or calibration.get("retained_repeats_per_actor")
        != numeric.CALIBRATION_REPEATS_V43G
        or calibration.get("bootstrap_resamples")
        != numeric.BOOTSTRAP_RESAMPLES_V43G
        or calibration.get("bootstrap_seed")
        != numeric.CALIBRATION_BOOTSTRAP_SEED_V43G
        or calibration.get("familywise_confidence")
        != numeric.BOOTSTRAP_CONFIDENCE_V43G
        or recipe.get("signed_replicates_per_direction")
        != numeric.SIGNED_REPLICATES_V43G
        or recipe.get("signed_replication_assignment")
        != "complete four-actor paired antithetic block per direction"
        or recipe.get("fitness_shaping")
        != "centered ranks over 16 median signed scores; coefficient=u_plus-u_minus"
        or recipe.get("minimum_response_reliability")
        != numeric.RELIABILITY_MINIMUM_V43G
        or recipe.get("minimum_split_half_spearman")
        != numeric.SPLIT_HALF_SPEARMAN_MINIMUM_V43G
        or calibration.get("historical_catastrophic_divergence_ceiling")
        != numeric.V43F_HISTORICAL_EQUAL_UNIT_BOUND
        or consensus.get("retained_repeats_per_actor")
        != numeric.POST_UPDATE_REPEATS_V43G
        or consensus.get("bootstrap_resamples") != numeric.BOOTSTRAP_RESAMPLES_V43G
        or consensus.get("bootstrap_seed")
        != numeric.POST_UPDATE_BOOTSTRAP_SEED_V43G
        or consensus.get("familywise_confidence")
        != numeric.BOOTSTRAP_CONFIDENCE_V43G
    ):
        raise RuntimeError("v43g preregistration contract changed")
    forbidden = ("shadow_dev", "eval_qa", "ood_qa", "holdout", "heldout")
    for label in ("dataset", "matched_initialization", "staged_initialization"):
        if any(token in str(recipe[label]).lower() for token in forbidden):
            raise RuntimeError(f"v43g forbidden selection path in {label}")
    return value


def _make_trainer(prereg: dict):
    saved = (v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION)
    v40a.EXPERIMENT = EXPERIMENT
    v40a.RUN_DIR = RUN_DIR
    v40a.WORKER_EXTENSION = WORKER_EXTENSION
    try:
        trainer = v40c.make_trainer_v40c(prereg)
    except Exception:
        v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION = saved
        raise
    return trainer, saved


def _lora_request():
    from vllm.lora.request import LoRARequest
    return LoRARequest(
        "matched_lora_initialization_v41b", 1, str(STAGED),
        base_model_name=str(v40a.MODEL),
    )


def _sampling_params():
    from vllm import SamplingParams
    return SamplingParams(
        n=1,
        seed=GLOBAL_SEED,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=1,
        detokenize=False,
    )


def _prepare(trainer, bundle: dict):
    prompts = [
        v40a.base.specialist_template(question) for question in bundle["questions"]
    ]
    dense_items = anchor_v4.prepare_gold_answer_items_v4(
        trainer.tokenizer, prompts, bundle["answers"],
    )
    requests = [
        {"prompt_token_ids": item["prompt_token_ids"]} for item in dense_items
    ]
    return dense_items, requests


def augment_unit_membership_v43g(bundle: dict) -> dict:
    """Attach content-free conflict-unit IDs to the frozen train bundle."""
    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    row_to_unit = {}
    for unit in manifest["content_free_unit_commitments"]:
        if unit["fold"] == 3:
            continue
        for row_sha in unit["row_sha256"]:
            if row_sha in row_to_unit:
                raise RuntimeError("v43g train row belongs to multiple conflict units")
            row_to_unit[row_sha] = {
                "unit_identity_sha256": unit["unit_identity_sha256"],
                "row_count": int(unit["row_count"]),
            }
    memberships = [row_to_unit.get(row_sha) for row_sha in bundle["row_sha256"]]
    if (
        any(item is None for item in memberships)
        or len({item["unit_identity_sha256"] for item in memberships})
        != numeric.EXPECTED_CONFLICT_UNITS_V43G
        or len(memberships) != 448
    ):
        raise RuntimeError("v43g train conflict-unit membership changed")
    bundle = dict(bundle)
    bundle["unit_membership_v43g"] = memberships
    bundle["unit_membership_sha256_v43g"] = v40a.canonical_sha256([
        {"row_sha256": row_sha, **membership}
        for row_sha, membership in zip(
            bundle["row_sha256"], memberships, strict=True,
        )
    ])
    return bundle


def score_batch_detailed_v43g(bundle: dict, dense_items: list, outputs: list) -> dict:
    dense = anchor_v4.score_gold_answer_outputs_v4(dense_items, outputs)
    rewards = [item["mean_answer_token_logprob"] for item in dense["examples"]]
    if len(rewards) != 448 or len(bundle["unit_membership_v43g"]) != len(rewards):
        raise RuntimeError("v43g score coverage changed")
    grouped: dict[str, dict] = {}
    for reward, membership in zip(
        rewards, bundle["unit_membership_v43g"], strict=True,
    ):
        unit_id = membership["unit_identity_sha256"]
        group = grouped.setdefault(unit_id, {
            "row_count": membership["row_count"], "values": [],
        })
        if group["row_count"] != membership["row_count"]:
            raise RuntimeError("v43g unit row count changed")
        group["values"].append(float(reward))
    units = []
    for unit_id in sorted(grouped):
        group = grouped[unit_id]
        if len(group["values"]) != group["row_count"]:
            raise RuntimeError("v43g unit score coverage changed")
        units.append({
            "unit_identity_sha256": unit_id,
            "row_count": group["row_count"],
            "mean_answer_token_logprob": (
                math.fsum(group["values"]) / group["row_count"]
            ),
        })
    if len(units) != numeric.EXPECTED_CONFLICT_UNITS_V43G:
        raise RuntimeError("v43g scored conflict-unit count changed")
    equal_mean = math.fsum(
        weight * reward
        for weight, reward in zip(bundle["weights"], rewards, strict=True)
    )
    unit_mean = math.fsum(
        item["mean_answer_token_logprob"] for item in units
    ) / len(units)
    if not math.isclose(equal_mean, unit_mean, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("v43g equal-unit aggregation changed")
    aggregate = {
        "equal_unit_mean": equal_mean,
        "unweighted_row_mean": math.fsum(rewards) / len(rewards),
        "dense_result_sha256": v40a.canonical_sha256(dense),
        "scored_answer_tokens": dense["answer_token_count"],
        "unit_aggregate_sha256": v40a.canonical_sha256(units),
    }
    return {"aggregate": aggregate, "units": units}


def _generate_actor_scores(trainer, bundle, dense_items, requests) -> list[dict]:
    batches = trainer._resolve([
        engine.generate.remote(
            requests,
            _sampling_params(),
            use_tqdm=False,
            lora_request=_lora_request(),
        )
        for engine in trainer.engines
    ])
    if len(batches) != 4 or any(len(batch) != 448 for batch in batches):
        raise RuntimeError("v43g replicated train-score coverage changed")
    return [
        {
            "actor_rank": actor_rank,
            **score_batch_detailed_v43g(bundle, dense_items, batch),
        }
        for actor_rank, batch in enumerate(batches)
    ]


def _exact_base_score(trainer, bundle, dense_items, requests) -> dict:
    actors = _generate_actor_scores(trainer, bundle, dense_items, requests)
    hashes = [v40a.canonical_sha256(item) for item in actors]
    # actor_rank is the only intentional record difference.
    score_hashes = [
        v40a.canonical_sha256({
            key: value for key, value in item.items() if key != "actor_rank"
        })
        for item in actors
    ]
    if len(set(score_hashes)) != 1:
        raise RuntimeError("v43g zero-effect base score differs across actors")
    return {
        "consensus": actors[0],
        "actors": actors,
        "exact_actor_record_sha256": hashes,
        "exact_score_sha256": score_hashes[0],
        "exact_bitwise_score_consensus": True,
    }


def _score_repeats(
    trainer,
    bundle,
    dense_items,
    requests,
    *,
    warmups: int,
    retained: int,
) -> list[dict]:
    for _index in range(int(warmups)):
        _generate_actor_scores(trainer, bundle, dense_items, requests)
    records = []
    for repeat_index in range(int(retained)):
        records.append({
            "repeat_index": repeat_index,
            "actors": _generate_actor_scores(
                trainer, bundle, dense_items, requests,
            ),
        })
    return records


def _persist_phase(path: Path, value: dict) -> dict:
    if path.exists():
        raise FileExistsError(path)
    sealed = v40a.self_hashed(value)
    v40a.atomic_json(path, sealed)
    reopened = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in reopened.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        reopened != sealed
        or reopened["content_sha256_before_self_field"]
        != v40a.canonical_sha256(compact)
    ):
        raise RuntimeError(f"v43g phase artifact changed on readback: {path}")
    return sealed


def _calibrate_numeric_path(
    trainer, bundle, dense_items, requests, master_sha: str,
) -> dict:
    materialized = v40a._rpc_all(
        trainer,
        "materialize_antithetic_adapter_v41a",
        (
            numeric.CALIBRATION_NOISE_SEED_V43G,
            CALIBRATION_SIGMA,
            1,
            master_sha,
        ),
    )
    candidate_hashes = {
        v40a.canonical_sha256(item["candidate_identity"])
        for item in materialized
    }
    runtime_hashes = {
        item["materialization"]["runtime_values_sha256"]
        for item in materialized
    }
    if len(candidate_hashes) != 1 or len(runtime_hashes) != 1:
        raise RuntimeError("v43g calibration state/materialization differs across ranks")
    records = None
    try:
        records = _score_repeats(
            trainer,
            bundle,
            dense_items,
            requests,
            warmups=numeric.CALIBRATION_WARMUPS_V43G,
            retained=numeric.CALIBRATION_REPEATS_V43G,
        )
    finally:
        restored = v40a._rpc_all(trainer, "restore_adapter_master_v41a")
    if any(item["restored_identity"]["sha256"] != master_sha for item in restored):
        raise RuntimeError("v43g calibration exact restore changed master identity")
    bounds = numeric.calibration_bootstrap_bounds_v43g(records)
    artifact = _persist_phase(CALIBRATION_ARTIFACT, {
        "schema": "matched-lora-es-numeric-calibration-v43g",
        "status": "complete_before_population",
        "synthetic_state": {
            "seed": numeric.CALIBRATION_NOISE_SEED_V43G,
            "sigma": CALIBRATION_SIGMA,
            "derivation": "alpha/sqrt(population_size)",
            "candidate_identity_sha256": next(iter(candidate_hashes)),
            "runtime_values_sha256": next(iter(runtime_hashes)),
            "all_four_ranks_exact": True,
        },
        "warmup_repeats_discarded": numeric.CALIBRATION_WARMUPS_V43G,
        "retained_repeats_per_actor": numeric.CALIBRATION_REPEATS_V43G,
        "records": records,
        "bootstrap": bounds,
        "restore_certificates": restored,
        "train_bundle_content_sha256": TRAIN_BUNDLE_SHA256,
        "shadow_dev_external_eval_ood_or_holdout_opened": False,
    })
    return artifact


def _compact_signed_score(score: dict) -> dict:
    return {
        "aggregate": score["aggregate"],
        "unit_aggregate_sha256": score["aggregate"]["unit_aggregate_sha256"],
    }


def _replicated_population(
    trainer, bundle, dense_items, requests, master_sha: str,
) -> dict:
    scores: dict[str, list[list[dict | None]]] = {
        "plus": [
            [None] * numeric.SIGNED_REPLICATES_V43G
            for _ in range(POPULATION_SIZE)
        ],
        "minus": [
            [None] * numeric.SIGNED_REPLICATES_V43G
            for _ in range(POPULATION_SIZE)
        ],
    }
    perturbations: dict[str, list[list[dict | None]]] = {
        "plus": [
            [None] * numeric.SIGNED_REPLICATES_V43G
            for _ in range(POPULATION_SIZE)
        ],
        "minus": [
            [None] * numeric.SIGNED_REPLICATES_V43G
            for _ in range(POPULATION_SIZE)
        ],
    }
    restorations = []
    assignments = []
    for direction in range(POPULATION_SIZE):
        assignment = numeric.complete_actor_assignments_v43g(direction)
        assignments.append(assignment)
        for label, sign in (("plus", 1), ("minus", -1)):
            values = trainer._resolve([
                trainer.engines[actor].collective_rpc.remote(
                    "materialize_antithetic_adapter_v41a",
                    args=(SEEDS[direction], SIGMA, sign, master_sha),
                )
                for actor in range(4)
            ])
            if len(values) != 4 or any(len(value) != 1 for value in values):
                raise RuntimeError("v43g complete-block perturbation coverage changed")
            certificates = [value[0] for value in values]
            batches = None
            try:
                batches = trainer._resolve([
                    trainer.engines[actor].generate.remote(
                        requests,
                        _sampling_params(),
                        use_tqdm=False,
                        lora_request=_lora_request(),
                    )
                    for actor in range(4)
                ])
            finally:
                restored = v40a._rpc_all(trainer, "restore_adapter_master_v41a")
                restorations.extend(restored)
            if len(batches) != 4 or any(len(batch) != 448 for batch in batches):
                raise RuntimeError("v43g complete signed actor block is incomplete")
            for actor_rank, batch in enumerate(batches):
                scored = score_batch_detailed_v43g(bundle, dense_items, batch)
                scores[label][direction][actor_rank] = {
                    "actor_rank": actor_rank,
                    **_compact_signed_score(scored),
                }
                perturbations[label][direction][actor_rank] = certificates[actor_rank]
    if any(
        item is None
        for label in scores.values() for direction in label for item in direction
    ):
        raise RuntimeError("v43g signed score matrix is incomplete")
    if any(
        item["restored_identity"]["sha256"] != master_sha for item in restorations
    ):
        raise RuntimeError("v43g signed exact restore changed master identity")
    for label in ("plus", "minus"):
        for direction in range(POPULATION_SIZE):
            certs = perturbations[label][direction]
            if (
                len({
                    v40a.canonical_sha256(item["candidate_identity"])
                    for item in certs
                }) != 1
                or len({
                    item["materialization"]["runtime_values_sha256"]
                    for item in certs
                }) != 1
            ):
                raise RuntimeError(
                    f"v43g replicated state differs for {label} direction {direction}"
                )
    equal_sign = {
        label: [
            [
                float(scores[label][direction][rep]["aggregate"]["equal_unit_mean"])
                for rep in range(numeric.SIGNED_REPLICATES_V43G)
            ]
            for direction in range(POPULATION_SIZE)
        ]
        for label in ("plus", "minus")
    }
    row_sign = {
        label: [
            [
                float(scores[label][direction][rep]["aggregate"]["unweighted_row_mean"])
                for rep in range(numeric.SIGNED_REPLICATES_V43G)
            ]
            for direction in range(POPULATION_SIZE)
        ]
        for label in ("plus", "minus")
    }
    robust = numeric.robust_population_v43g(equal_sign)
    if robust["zero_utility_update"]:
        raise RuntimeError("v43g centered-rank population produced a zero update")
    robust_details = {key: value for key, value in robust.items() if key != "schema"}
    return {
        "schema": "complete-actor-block-robust-lora-es-population-v43g",
        "assignments": assignments,
        "signed_scores": scores,
        "equal_unit_sign_scores": equal_sign,
        "unweighted_row_sign_scores": row_sign,
        "fitness_shaping_schema": robust["schema"],
        **robust_details,
        "perturbation_certificates": perturbations,
        "restoration_certificate_count": len(restorations),
        "all_exact_restores_passed": True,
    }


def _materialization_consensus(items: list[dict], phase: str) -> str:
    hashes = {
        item["materialization"]["runtime_values_sha256"] for item in items
    }
    bases = {item["base_identity"]["inventory_sha256"] for item in items}
    if len(hashes) != 1 or len(bases) != 1:
        raise RuntimeError(f"v43g runtime/base materialization differs at {phase}")
    return next(iter(hashes))


def _apply_update(
    trainer,
    master: dict,
    reference_generation: int,
    coefficients: list[float],
    plan_id: str,
) -> dict:
    coefficient_sha = worker_v3.coefficient_sha256_v3(SEEDS, coefficients)
    prepared = v40a._rpc_all(trainer, "prepare_sharded_adapter_update_v41a", (
        SEEDS,
        coefficients,
        coefficient_sha,
        POPULATION_SIZE,
        4,
        ALPHA,
        plan_id,
        master["sha256"],
        reference_generation,
    ))
    manifests = {item["manifest_sha256"] for item in prepared}
    if (
        len(manifests) != 1
        or {item["rank"] for item in prepared} != {0, 1, 2, 3}
        or len({
            v40a.canonical_sha256(item["master_identity"]) for item in prepared
        }) != 1
    ):
        raise RuntimeError("v43g prepared update consensus changed")
    manifest_sha = next(iter(manifests))
    executed = v40a._rpc_all(
        trainer, "execute_sharded_adapter_update_v41a", (manifest_sha,),
    )
    identities = [item["candidate_identity"] for item in executed]
    if len({v40a.canonical_sha256(item) for item in identities}) != 1:
        raise RuntimeError("v43g exact update candidate differs across ranks")
    final_identity = identities[0]
    execute_runtime_sha = _materialization_consensus(executed, "execute")
    committed = v40a._rpc_all(trainer, "commit_sharded_adapter_update_v41a", (
        manifest_sha, final_identity["sha256"],
    ))
    if (
        any(item["final_identity"] != final_identity for item in committed)
        or any(item.get("requires_cross_rank_finalize") is not True for item in committed)
    ):
        raise RuntimeError("v43g exact committed identity differs across ranks")
    commit_runtime_sha = _materialization_consensus(committed, "commit")
    finalized = v40a._rpc_all(trainer, "finalize_sharded_adapter_update_v41a", (
        manifest_sha, final_identity["sha256"],
    ))
    if (
        any(item.get("finalized") is not True for item in finalized)
        or any(item["final_identity"] != final_identity for item in finalized)
    ):
        raise RuntimeError("v43g exact finalized identity differs across ranks")
    finalize_runtime_sha = _materialization_consensus(finalized, "finalize")
    if len({execute_runtime_sha, commit_runtime_sha, finalize_runtime_sha}) != 1:
        raise RuntimeError("v43g exact runtime materialization changed across update phases")
    references = v40a._rpc_all(trainer, "capture_adapter_reference_v41a")
    if len({
        v40a.canonical_sha256(item["reference_identity"]) for item in references
    }) != 1:
        raise RuntimeError("v43g post-update exact references differ across ranks")
    snapshots = v40a._rpc_all(
        trainer,
        "save_adapter_snapshot_v41a",
        (str(SNAPSHOT), final_identity["sha256"]),
    )
    written = [item for item in snapshots if item.get("written")]
    if (
        len(written) != 1
        or not written[0].get("readback_verified")
        or written[0].get("readback_identity") != final_identity
    ):
        raise RuntimeError("v43g exact snapshot readback coverage changed")
    return {
        "coefficient_sha256": coefficient_sha,
        "manifest_sha256": manifest_sha,
        "prepared": prepared,
        "executed": executed,
        "committed": committed,
        "finalized": finalized,
        "final_identity": final_identity,
        "runtime_values_sha256": execute_runtime_sha,
        "new_references": references,
        "snapshots": snapshots,
        "exact_fp32_and_runtime_consensus": True,
    }


def _post_update_consensus(
    trainer, bundle, dense_items, requests, calibration: dict,
) -> dict:
    records = _score_repeats(
        trainer,
        bundle,
        dense_items,
        requests,
        warmups=0,
        retained=numeric.POST_UPDATE_REPEATS_V43G,
    )
    result = numeric.post_update_consensus_v43g(
        records, calibration["bootstrap"]["bounds"],
    )
    return _persist_phase(POST_UPDATE_ARTIFACT, {
        "schema": "matched-lora-es-post-update-consensus-v43g",
        "status": "complete_before_acceptance_decision",
        "records": records,
        "equivalence": result,
        "calibration_content_sha256": calibration[
            "content_sha256_before_self_field"
        ],
        "train_bundle_content_sha256": TRAIN_BUNDLE_SHA256,
        "shadow_dev_external_eval_ood_or_holdout_opened": False,
    })


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    prereg = load_preregistration(args)
    if args.dry_run:
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "train_only": True,
            "sealed_holdout_opened": False,
            "protected_semantic_access": False,
            "protected_paths_opened": [],
            "train_dataset_content_hashed_for_binding": True,
            "train_dataset_semantics_loaded": False,
            "model_metadata_hashed_for_binding": True,
            "model_runtime_loaded": False,
            "gpu_launched": False,
            "filesystem_writes": False,
        }, sort_keys=True))
        return 0
    if ATTEMPT.exists() or RUN_DIR.exists():
        raise RuntimeError("v43g requires fresh artifact paths")
    preflight = v40a.gpu_preflight()
    attempt = v40a.self_hashed({
        "schema": "matched-lora-es-attempt-v43g",
        "status": "launching",
        "phase": "before_model_or_train_data_load",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_file_sha256": args.preregistration_sha256,
        "preregistration_content_sha256": args.preregistration_content_sha256,
        "preflight": preflight,
        "shadow_dev_external_eval_ood_or_holdout_opened": False,
    })
    v40a.atomic_json(ATTEMPT, attempt)
    RUN_DIR.mkdir(parents=True)
    trainer = monitor = saved = None
    stop = threading.Event()
    failures: queue.Queue = queue.Queue()
    phase = v40a.Phase()
    started = time.monotonic()
    try:
        bundle = equal_v38.load_equal_unit_train_bundle(
            DATASET, DATASET_SHA256, SPLIT_MANIFEST, SPLIT_MANIFEST_SHA256,
        )
        if bundle["content_sha256_before_self_field"] != TRAIN_BUNDLE_SHA256:
            raise RuntimeError("v43g frozen train bundle identity changed")
        bundle = augment_unit_membership_v43g(bundle)
        v40a.base.set_seed(GLOBAL_SEED)
        trainer, saved = _make_trainer(prereg)
        actor_ids = trainer._resolve([
            engine.runtime_identity_v40a.remote() for engine in trainer.engines
        ])
        pid_map = prior._actor_pid_map(actor_ids)
        monitor = threading.Thread(
            target=v40a.monitor_gpus,
            args=(stop, phase, pid_map, GPU_LOG, failures),
            daemon=True,
        )
        monitor.start()
        dense_items, requests = _prepare(trainer, bundle)

        phase.value = "activate_matched_initialization"
        preinstall = _exact_base_score(trainer, bundle, dense_items, requests)
        phase.value = "install_canonical_master"
        installs = v40a._rpc_all(trainer, "install_adapter_state_v41a", (
            str(SOURCE_WEIGHTS),
            str(SOURCE_CONFIG),
            v40a.file_sha256(SOURCE_WEIGHTS),
            v40a.file_sha256(SOURCE_CONFIG),
        ))
        certificates = v40a._rpc_all(trainer, "adapter_state_certificate_v41a")
        masters = [item["current_identity"] for item in certificates]
        if len({v40a.canonical_sha256(item) for item in masters}) != 1:
            raise RuntimeError("v43g canonical FP32 master differs across ranks")
        if len({
            item["materialization"]["runtime_values_sha256"]
            for item in certificates
        }) != 1:
            raise RuntimeError("v43g initial runtime materialization differs across ranks")
        master = masters[0]
        references = v40a._rpc_all(trainer, "capture_adapter_reference_v41a")
        reference_generations = {item["reference_generation"] for item in references}
        if len(reference_generations) != 1:
            raise RuntimeError("v43g reference generation differs across ranks")
        postinstall = _exact_base_score(trainer, bundle, dense_items, requests)
        if preinstall["consensus"] != postinstall["consensus"]:
            raise RuntimeError("v43g canonical install changed matched-init train score")

        phase.value = "nonzero_numeric_calibration"
        calibration = _calibrate_numeric_path(
            trainer, bundle, dense_items, requests, master["sha256"],
        )
        post_calibration = _exact_base_score(
            trainer, bundle, dense_items, requests,
        )
        if post_calibration["consensus"] != postinstall["consensus"]:
            raise RuntimeError("v43g calibration restore changed matched-init score")

        phase.value = "complete_actor_block_population_pop8"
        population = _replicated_population(
            trainer, bundle, dense_items, requests, master["sha256"],
        )
        post_population = _exact_base_score(
            trainer, bundle, dense_items, requests,
        )
        if post_population["consensus"] != postinstall["consensus"]:
            raise RuntimeError("v43g replicated population restore changed base score")
        reliability = numeric.reliability_gate_v43g(
            population["central_replicates"],
            calibration["bootstrap"]["bounds"]["equal_unit_mean"][
                "observed_maximum_repeat_actor_spread"
            ],
        )
        reliability_artifact = _persist_phase(RELIABILITY_ARTIFACT, {
            "schema": "matched-lora-es-population-reliability-v43g",
            "status": "complete_before_update_decision",
            "population": population,
            "reliability_gate": reliability,
            "calibration_content_sha256": calibration[
                "content_sha256_before_self_field"
            ],
            "train_bundle_content_sha256": TRAIN_BUNDLE_SHA256,
            "shadow_dev_external_eval_ood_or_holdout_opened": False,
        })
        if not reliability["passed"]:
            raise RuntimeError(
                "v43g replicated population failed preregistered reliability gate: "
                f"reliability={reliability['reliability']} "
                f"split_half_spearman={reliability['split_half_spearman']} "
                "fresh_calibration_inside_historical_ceiling="
                f"{reliability['fresh_calibration_inside_historical_ceiling']}"
            )

        plan_id = v40a.canonical_sha256({
            "schema": "matched-lora-es-update-plan-v43g",
            "master_sha256": master["sha256"],
            "seeds": SEEDS,
            "coefficients": population["coefficients"],
            "sigma": SIGMA,
            "alpha": ALPHA,
            "train_bundle_content_sha256": TRAIN_BUNDLE_SHA256,
            "reference_generation": next(iter(reference_generations)),
            "calibration_content_sha256": calibration[
                "content_sha256_before_self_field"
            ],
            "reliability_content_sha256": reliability_artifact[
                "content_sha256_before_self_field"
            ],
        })
        phase.value = "distributed_exact_fp32_update"
        update = _apply_update(
            trainer,
            master,
            next(iter(reference_generations)),
            population["coefficients"],
            plan_id,
        )
        phase.value = "post_update_replicated_consensus"
        post_update = _post_update_consensus(
            trainer, bundle, dense_items, requests, calibration,
        )
        if not post_update["equivalence"]["passed"]:
            raise RuntimeError(
                "v43g post-update scores failed preregistered calibrated equivalence"
            )

        stop.set()
        monitor.join(timeout=10)
        if monitor.is_alive() or not failures.empty():
            raise RuntimeError("v43g GPU monitor failed") from (
                failures.get() if not failures.empty() else None
            )
        gpu = v40a.summarize_gpu(GPU_LOG, pid_map)
        cleanup = v40a.cleanup_v38a.strict_close_trainer_v38a(trainer)
        trainer = None
        import ray
        ray.shutdown()
        idle = v40a.cleanup_v38a.wait_for_gpu_idle()
        report = v40a.self_hashed({
            "schema": "matched-lora-es-train-only-report-v43g",
            "status": "complete_robust_centered_rank_one_update_state_sealed",
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_runtime_seconds": time.monotonic() - started,
            "preregistration": {
                "path": str(Path(args.preregistration).resolve()),
                "file_sha256": args.preregistration_sha256,
                "content_sha256": args.preregistration_content_sha256,
            },
            "train_bundle": {
                "path": str(DATASET),
                "sha256": DATASET_SHA256,
                "content_sha256": TRAIN_BUNDLE_SHA256,
                "rows": 448,
                "conflict_units": numeric.EXPECTED_CONFLICT_UNITS_V43G,
                "weight_identity_sha256": bundle["weight_identity_sha256"],
                "unit_membership_sha256": bundle["unit_membership_sha256_v43g"],
            },
            "recipe": {
                "population_size": POPULATION_SIZE,
                "seeds": SEEDS,
                "sigma": SIGMA,
                "alpha": ALPHA,
                "signed_replicates_per_direction": numeric.SIGNED_REPLICATES_V43G,
                "signed_sequence_presentations": (
                    2 * numeric.SIGNED_REPLICATES_V43G * POPULATION_SIZE * 448
                ),
                "fitness_shaping": (
                    "centered ranks over median signed actor scores; "
                    "coefficient=u_plus-u_minus"
                ),
                "calibration_sequence_presentations": (
                    numeric.CALIBRATION_WARMUPS_V43G
                    + numeric.CALIBRATION_REPEATS_V43G
                ) * 4 * 448,
                "post_update_sequence_presentations": (
                    numeric.POST_UPDATE_REPEATS_V43G * 4 * 448
                ),
                "objective": anchor_v4.DENSE_GOLD_REWARD_CONFIG_V4,
                "matched_initialization_weights_sha256": v40a.file_sha256(
                    SOURCE_WEIGHTS
                ),
            },
            "actor_identities": actor_ids,
            "physical_gpu_pid_map": pid_map,
            "installations": installs,
            "initial_master_identity": master,
            "initial_references": references,
            "score_audits": {
                "preinstall": preinstall,
                "postinstall": postinstall,
                "post_calibration": post_calibration,
                "post_population": post_population,
                "canonical_install_identity_passed": True,
                "all_exact_restores_passed": True,
            },
            "calibration": {
                "path": str(CALIBRATION_ARTIFACT),
                "file_sha256": v40a.file_sha256(CALIBRATION_ARTIFACT),
                "content_sha256": calibration["content_sha256_before_self_field"],
                "bounds": calibration["bootstrap"]["bounds"],
            },
            "population_reliability": {
                "path": str(RELIABILITY_ARTIFACT),
                "file_sha256": v40a.file_sha256(RELIABILITY_ARTIFACT),
                "content_sha256": reliability_artifact[
                    "content_sha256_before_self_field"
                ],
                "gate": reliability,
            },
            "plan_id": plan_id,
            "update": update,
            "post_update_consensus": {
                "path": str(POST_UPDATE_ARTIFACT),
                "file_sha256": v40a.file_sha256(POST_UPDATE_ARTIFACT),
                "content_sha256": post_update[
                    "content_sha256_before_self_field"
                ],
                "equivalence": post_update["equivalence"],
            },
            "gpu_activity": gpu,
            "placement_group_cleanup": cleanup,
            "final_gpu_idle": idle,
            "preflight": preflight,
            "gpu_log": {
                "path": str(GPU_LOG),
                "file_sha256": v40a.file_sha256(GPU_LOG),
            },
            "shadow_dev_external_eval_ood_or_holdout_opened": False,
            "quality_or_promotion_conclusion_authorized": False,
        })
        v40a.atomic_json(REPORT, report)
        print(json.dumps({
            "report": str(REPORT),
            "report_sha256": v40a.file_sha256(REPORT),
            "reliability": reliability["reliability"],
            "post_update_consensus_passed": post_update["equivalence"]["passed"],
            "snapshot": str(SNAPSHOT),
        }, sort_keys=True))
        return 0
    except BaseException as error:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=10)
        v40a.atomic_json(RUN_DIR / "failure_v43g.json", v40a.self_hashed({
            "schema": "matched-lora-es-failure-v43g",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "persisted_phase_artifacts": {
                "calibration": str(CALIBRATION_ARTIFACT)
                if CALIBRATION_ARTIFACT.exists() else None,
                "reliability": str(RELIABILITY_ARTIFACT)
                if RELIABILITY_ARTIFACT.exists() else None,
                "post_update": str(POST_UPDATE_ARTIFACT)
                if POST_UPDATE_ARTIFACT.exists() else None,
                "snapshot": str(SNAPSHOT) if SNAPSHOT.exists() else None,
            },
            "shadow_dev_external_eval_ood_or_holdout_opened": False,
        }))
        raise
    finally:
        if trainer is not None:
            try:
                v40a.base.close_trainer(trainer)
            except Exception:
                pass
        try:
            import ray
            ray.shutdown()
        except Exception:
            pass
        if saved is not None:
            v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION = saved


if __name__ == "__main__":
    raise SystemExit(main())
