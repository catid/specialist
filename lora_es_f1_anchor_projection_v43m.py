#!/usr/bin/env python3
"""CPU-only three-anchor projection from V43I's sealed signed population."""

from __future__ import annotations

import json
from pathlib import Path

import eggroll_es_multi_anchor_v43h as projection
import run_lora_es_multi_anchor_v43i as v43i


ROOT = Path(__file__).resolve().parent
POPULATION = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v43i_matched_lora_es_fold3_pop8_multi_anchor/"
    "population_reliability_v43i.json"
).resolve()
POPULATION_FILE_SHA256 = (
    "7a1f04dc57aa42cab36ccb60d3382c798614bef9f29d2c84708dfc9178a330f6"
)
POPULATION_CONTENT_SHA256 = (
    "998cc1567ca96cd5a91c236e86347b72931ecc0f3cb525426cd83e2bc861797b"
)
PROJECTION_ARTIFACT = (
    ROOT / "experiments/eggroll_es_hpo/projections/"
    "lora_es_three_anchor_projection_v43m.json"
).resolve()
PROJECTION_ARTIFACT_FILE_SHA256 = (
    "182221606b470f0af23267f829a607ca881eae40ca1c9ac81e815554a7345e35"
)
PROJECTION_ARTIFACT_CONTENT_SHA256 = (
    "ef4b9ed5ddd1393dde53e4dfc056f5cd178bb1f0d583452a7ec4cde40c9b57c4"
)
V43I_CANDIDATE_GATE = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v43i_matched_lora_es_fold3_pop8_multi_anchor/candidate_gate_v43i.json"
).resolve()
V43I_CANDIDATE_GATE_FILE_SHA256 = (
    "fc6ed9fc2908cd33975a21877e22269dfc1cdd4c5f89a7d14b51d5b0823bdc6a"
)
V43I_CANDIDATE_GATE_CONTENT_SHA256 = (
    "070d225fd279de5cb969c9c74d2200bf1978fbbf7e0ea2967be24688234f70b1"
)
V43I_EXACT_ABORT = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v43i_matched_lora_es_fold3_pop8_multi_anchor/exact_abort_v43i.json"
).resolve()
V43I_EXACT_ABORT_FILE_SHA256 = (
    "83cc533e72100018a0498c4555c7f4f543bf78cf2c6bc2bdc248a6918ceea9b5"
)
V43I_EXACT_ABORT_CONTENT_SHA256 = (
    "039f6c8d122c1d06aa06d42671631a604d8599af9873cabb11c60b2c97371de8"
)
V43I_CANDIDATE_STATE_SHA256 = (
    "291027e22afc08a106e44675c73954ca5d4b0c2f1a2f5f946a82cc738748dfb3"
)
V43I_RESTORED_MASTER_SHA256 = (
    "dfb8ef8981cd4a21bd8d342353fc3d9c84c5d4759c38973e1528245f2baff192"
)
IDENTICAL_COEFFICIENT_SHA256 = (
    "b090190a86f3219973255a7f1e8f5d652e661b380a819c6256a562edaa57bb8f"
)
REQUIRED_ANCHORS_V43M = (
    "prose_lm", "qa_answer_logprob", "qa_generation_mean_f1",
)
OBJECTIVE_PATHS_V43M = {
    "domain": ("domain", "aggregate", "equal_unit_mean"),
    "prose_lm": ("prose_lm", "mean_token_logprob"),
    "qa_answer_logprob": ("qa_answer_logprob", "mean_example_logprob"),
    "qa_generation_mean_f1": ("qa_generation", "mean_f1"),
}


def _nested(value: dict, path: tuple[str, ...]) -> float:
    for key in path:
        value = value[key]
    return float(value)


def _read_population_v43m() -> dict:
    if v43i.v40a.file_sha256(POPULATION) != POPULATION_FILE_SHA256:
        raise RuntimeError("v43m V43I population file identity changed")
    value = json.loads(POPULATION.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field")
        != POPULATION_CONTENT_SHA256
        or v43i.v40a.canonical_sha256(compact) != POPULATION_CONTENT_SHA256
        or value.get("schema")
        != "matched-lora-es-population-reliability-v43i"
        or value.get("status") != "complete_before_update_decision"
        or value.get("reliability_gate", {}).get("passed") is not True
        or value.get("shadow_dev_external_eval_ood_or_holdout_opened") is not False
    ):
        raise RuntimeError("v43m V43I population seal changed")
    return value


def _sign_scores_v43m(population: dict, objective: str) -> dict:
    path = OBJECTIVE_PATHS_V43M[objective]
    signed = population["signed_scores"]
    result = {
        sign: [
            [_nested(actor, path) for actor in direction]
            for direction in signed[sign]
        ]
        for sign in ("plus", "minus")
    }
    if (
        any(len(result[sign]) != 8 for sign in result)
        or any(len(direction) != 4 for sign in result for direction in result[sign])
    ):
        raise RuntimeError("v43m signed four-actor population coverage changed")
    return result


def build_projection_v43m() -> dict:
    sealed = _read_population_v43m()
    population = sealed["population"]
    if (
        population.get("schema")
        != "fused-complete-actor-block-multi-anchor-population-v43i"
        or population.get("all_exact_restores_passed") is not True
        or population.get("restoration_certificate_count") != 64
        or len(population.get("assignments", [])) != 8
        or population.get("fused_requests_per_actor_state") != 544
    ):
        raise RuntimeError("v43m V43I signed population inventory changed")
    sign_scores = {
        objective: _sign_scores_v43m(population, objective)
        for objective in OBJECTIVE_PATHS_V43M
    }
    objectives = {
        objective: projection.objective_coefficients_v43h(scores)
        for objective, scores in sign_scores.items()
    }
    for objective in ("domain", "prose_lm", "qa_answer_logprob"):
        if objectives[objective] != population["objective_fitness"][objective]:
            raise RuntimeError(
                f"v43m reconstructed V43I centered ranks changed: {objective}"
            )
    if any(objectives[name]["zero_spread"] for name in objectives):
        raise RuntimeError("v43m required objective has zero centered-rank spread")
    projected = projection.project_multi_anchor_trust_region_v43h(
        objectives["domain"]["coefficients"],
        {name: objectives[name]["coefficients"] for name in REQUIRED_ANCHORS_V43M},
        max_norm_ratio=projection.TRUST_REGION_NORM_RATIO_V43H,
    )
    diagnostics = projected["diagnostics"]
    if (
        diagnostics.get("decision") != "project_and_trust_region"
        or diagnostics.get("anchor_order") != list(REQUIRED_ANCHORS_V43M)
        or diagnostics.get("all_anchor_halfspaces_satisfied") is not True
        or diagnostics.get("trust_region_norm_ratio") != 0.5
        or diagnostics.get("update_norm_ratio") > 0.5
        or not any(value != 0.0 for value in projected["coefficients"])
    ):
        raise RuntimeError("v43m three-anchor projection failed closed")
    return v43i.v40a.self_hashed({
        "schema": "lora-es-three-anchor-cpu-projection-v43m",
        "status": "complete_from_sealed_v43i_signed_population",
        "source": {
            "path": str(POPULATION),
            "file_sha256": POPULATION_FILE_SHA256,
            "content_sha256": POPULATION_CONTENT_SHA256,
            "population_resampled": False,
            "population_rescored": False,
            "projection_reused_without_new_model_inference": True,
        },
        "seeds": list(v43i.SEEDS),
        "population_size": v43i.POPULATION_SIZE,
        "signed_actor_replicates_per_direction": 4,
        "objective_paths": {
            key: list(path) for key, path in OBJECTIVE_PATHS_V43M.items()
        },
        "objective_sign_scores": sign_scores,
        "objective_fitness": objectives,
        "required_centered_rank_anchors": list(REQUIRED_ANCHORS_V43M),
        "projection": projected,
        "qa_generation_mean_f1_is_projection_constraint": True,
        "shadow_ood_holdout_or_heldout_opened": False,
        "protected_semantics_opened": False,
        "current_fixed_holdout_cycle_eligible": False,
    })


def _read_self_hashed_v43m(
    path: Path, expected_file: str, expected_content: str,
) -> dict:
    if v43i.v40a.file_sha256(path) != expected_file:
        raise RuntimeError(f"v43m sealed file changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != expected_content
        or v43i.v40a.canonical_sha256(compact) != expected_content
    ):
        raise RuntimeError(f"v43m sealed content changed: {path}")
    return value


def build_identity_noop_v43m() -> dict:
    """Fail closed when the proposed V43M transaction is exactly V43I again."""
    projection_artifact = _read_self_hashed_v43m(
        PROJECTION_ARTIFACT,
        PROJECTION_ARTIFACT_FILE_SHA256,
        PROJECTION_ARTIFACT_CONTENT_SHA256,
    )
    population = _read_population_v43m()["population"]
    gate = _read_self_hashed_v43m(
        V43I_CANDIDATE_GATE,
        V43I_CANDIDATE_GATE_FILE_SHA256,
        V43I_CANDIDATE_GATE_CONTENT_SHA256,
    )
    abort = _read_self_hashed_v43m(
        V43I_EXACT_ABORT,
        V43I_EXACT_ABORT_FILE_SHA256,
        V43I_EXACT_ABORT_CONTENT_SHA256,
    )
    proposed = projection_artifact["projection"]["coefficients"]
    prior_coefficients = population["coefficients"]
    coefficient_sha = v43i.worker_v3.coefficient_sha256_v3(
        v43i.SEEDS, proposed,
    )
    if (
        proposed != prior_coefficients
        or coefficient_sha != IDENTICAL_COEFFICIENT_SHA256
        or gate.get("schema")
        != "matched-lora-es-uncommitted-candidate-gate-v43i"
        or gate.get("status") != "complete_before_commit"
        or gate.get("candidate_identity", {}).get("sha256")
        != V43I_CANDIDATE_STATE_SHA256
        or gate.get("gate", {}).get("passed") is not False
        or abort.get("status") != "candidate_rejected_and_exactly_restored"
        or abort.get("all_four_ranks_exact") is not True
        or abort.get("candidate_gate_content_sha256")
        != V43I_CANDIDATE_GATE_CONTENT_SHA256
        or abort.get("restored_master_identity", {}).get("sha256")
        != V43I_RESTORED_MASTER_SHA256
        or abort.get("protected_semantics_opened") is not False
    ):
        raise RuntimeError("v43m identity/no-op proof failed closed")
    return v43i.v40a.self_hashed({
        "schema": "lora-es-three-anchor-identity-noop-v43m",
        "status": "cpu_only_noop_identical_to_rejected_v43i_candidate",
        "decision": "do_not_launch_duplicate_gpu_evaluation",
        "identity_basis": {
            "same_initial_master_sha256": V43I_RESTORED_MASTER_SHA256,
            "same_seeds": list(v43i.SEEDS),
            "same_population_size": v43i.POPULATION_SIZE,
            "same_alpha": v43i.ALPHA,
            "same_fp32_coefficients": proposed,
            "same_coefficient_sha256": coefficient_sha,
            "coefficient_lists_equal_exactly": True,
            "deterministic_scaled_tensor_update_equal_exactly": True,
        },
        "already_evaluated_state": {
            "revision": "v43i",
            "candidate_state_sha256": V43I_CANDIDATE_STATE_SHA256,
            "candidate_gate_passed": False,
            "candidate_gate_file_sha256": V43I_CANDIDATE_GATE_FILE_SHA256,
            "candidate_gate_content_sha256": V43I_CANDIDATE_GATE_CONTENT_SHA256,
            "exact_abort_file_sha256": V43I_EXACT_ABORT_FILE_SHA256,
            "exact_abort_content_sha256": V43I_EXACT_ABORT_CONTENT_SHA256,
            "all_four_ranks_exactly_restored": True,
        },
        "three_anchor_result": {
            "f1_anchor_added": True,
            "f1_halfspace_nonbinding": True,
            "projection_coefficients_changed_from_v43i": False,
        },
        "gpu_model_or_dataset_accessed": False,
        "shadow_ood_holdout_or_heldout_opened": False,
        "protected_semantics_opened": False,
        "current_fixed_holdout_cycle_eligible": False,
    })
