#!/usr/bin/env python3
"""Preregister the train-only lagged hard-replay calibration V35A.

This stage does not choose a replay fraction and cannot update a model.  It
freezes an unperturbed, four-engine calibration pass over the three V13
optimization panels.  The resulting content-free candidate pool must pass a
blinded manual correctness audit before a separately preregistered replay HPO
may use it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import eggroll_es_train_panel_sampler_v13 as sampler_v13


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_LAGGED_REPLAY_CALIBRATION_V35A_PREREGISTRATION.json"
).resolve()
PANEL_MANIFEST_PATH = (
    ROOT / "experiments/eggroll_es_hpo/train_panel_sampling_v13/"
    "document_balanced_train_panels_v13.json"
).resolve()
V13_EVIDENCE_PATH = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V13B_TRAIN_PANEL_AGGREGATE_EVIDENCE_V14A.json"
).resolve()
V14A_NEGATIVE_PATH = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V14A_FULL_FRAME_NEGATIVE_AGGREGATE_EVIDENCE_V14B.json"
).resolve()
V14B_NEGATIVE_PATH = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V14B_K2_DISTINCT_ROW_NEGATIVE_AGGREGATE_EVIDENCE_V15.json"
).resolve()
LAYER_PLAN_PATH = (
    ROOT / "experiments/layer_plans/v23a_base_middle_late_dense.json"
).resolve()
MODEL_PATH = (ROOT / "models/Qwen3.6-35B-A3B").resolve()
PROTOCOL_PATH = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_LAGGED_REPLAY_CALIBRATION_V35A_PROTOCOL.md"
).resolve()

BOUND_FILES = {
    "protocol_v35a": (
        PROTOCOL_PATH,
        "cf6f3896d92a54fa4ab297c1818cd421481e860d563030d883ee2c85491b541b",
    ),
    "sampler_v13": (
        Path(sampler_v13.__file__).resolve(),
        "81ca63c230995d44dfdb739a78b4a1a0f85d09724ba5915860ae36f78ccc3da9",
    ),
    "panel_manifest_v13": (
        PANEL_MANIFEST_PATH,
        "e555d9d6746cde6297cd3ab523b16dd7d78d81e2674447ee46d754ebfac52da7",
    ),
    "v13_train_aggregate_evidence": (
        V13_EVIDENCE_PATH,
        "d367c9c4de1e1f3526ddb3dfba2f5bf24efc77cbccf951f7359eb1969fcd7b54",
    ),
    "v14a_negative_evidence": (
        V14A_NEGATIVE_PATH,
        "9329c09fec5d76e209cff36ac80ffbbec69f53fe7bae9edcc069421d626cc9e9",
    ),
    "v14b_negative_evidence": (
        V14B_NEGATIVE_PATH,
        "735ad52b6395700feb4e8a3dccab165f9b79e620a53918d96e0a26979f58224c",
    ),
    "layer_plan": (
        LAYER_PLAN_PATH,
        "4dca69212f2eee1c5882a7f3de944e3dadd8de94b42e58e7d7495547e8b1c747",
    ),
    "model_config": (
        MODEL_PATH / "config.json",
        "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
    ),
    "model_index": (
        MODEL_PATH / "model.safetensors.index.json",
        "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
    ),
}

PANEL_MANIFEST_CONTENT_SHA256 = (
    "46cc98b694c98c1ee1c5456b855fb3b1db4534b3df2dcda69fc690a2d8a61bf5"
)
V13_EVIDENCE_CONTENT_SHA256 = (
    "06f662574013345a6c777af8688a38f3941286d9e11a427ed3342de53451b1e3"
)
V14A_NEGATIVE_CONTENT_SHA256 = (
    "ee4ded3d974dfd0becaedb1007f96888e133db51e62130d2844ab9c25e2ccf2b"
)
V14B_NEGATIVE_CONTENT_SHA256 = (
    "440504e6c81673ea8de89f336d587a0c57408ea21d3a925ed73b73ecfbeaa7b8"
)
LAYER_PLAN_CONTENT_SHA256 = (
    "07a155d1217b27ba1bf30e057024247236a812841c52bab401d465c9fdb5273f"
)
OPTIMIZATION_PANELS = tuple(sampler_v13.PANEL_NAMES[:3])
EXCLUDED_SCREENS = tuple(sampler_v13.PANEL_NAMES[3:])
PROVISIONAL_POOL_FRACTION = 0.50
FINAL_HARD_TIER_CAP = 0.25
REPLAY_FRACTIONS_RECOMMENDED = (0.10, 0.20)
GENERATION_SEED = 43
MAX_TOKENS = 1


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("ascii")).hexdigest()


def without_self(value: dict) -> dict:
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _load_bound_json(path: Path, expected_content: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("content_sha256_before_self_field") != expected_content
        or canonical_sha256(without_self(value)) != expected_content
    ):
        raise RuntimeError(f"v35a bound JSON content changed: {path.name}")
    return value


def load_bound_inputs() -> tuple[dict, dict, dict, dict, dict]:
    if any(file_sha256(path) != digest for path, digest in BOUND_FILES.values()):
        raise RuntimeError("v35a bound train-only source, evidence, model, or plan changed")
    manifest = _load_bound_json(
        PANEL_MANIFEST_PATH, PANEL_MANIFEST_CONTENT_SHA256,
    )
    v13 = _load_bound_json(V13_EVIDENCE_PATH, V13_EVIDENCE_CONTENT_SHA256)
    v14a = _load_bound_json(
        V14A_NEGATIVE_PATH, V14A_NEGATIVE_CONTENT_SHA256,
    )
    v14b = _load_bound_json(
        V14B_NEGATIVE_PATH, V14B_NEGATIVE_CONTENT_SHA256,
    )
    layer = json.loads(LAYER_PLAN_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != sampler_v13.SCHEMA
        or manifest.get("train_only") is not True
        or [panel.get("name") for panel in manifest.get("panels", [])]
        != list(sampler_v13.PANEL_NAMES)
        or manifest.get("hard_example_mixture", {}).get("enabled") is not False
        or v13.get("selection_surface") != "frozen_train_panels_only"
        or v14a.get("decision", {}).get("sampler") != "retain_v13"
        or v14b.get("decision", {}).get("sampler") != "retain_v13"
        or layer.get("plan_sha256") != LAYER_PLAN_CONTENT_SHA256
        or layer.get("layers") != [20, 21, 22, 23]
    ):
        raise RuntimeError("v35a train-only lineage semantics changed")
    return manifest, v13, v14a, v14b, layer


def _panel_contract(manifest: dict) -> dict:
    result = {}
    for panel in manifest["panels"]:
        result[panel["name"]] = {
            "role": panel["role"],
            "rows": panel["rows"],
            "stratum_counts": panel["stratum_counts"],
            "ordered_row_identity_sha256": panel[
                "ordered_row_identity_sha256"
            ],
        }
    return result


def _tier_counts() -> dict:
    return {
        stratum: {
            "panel_rows": count,
            "provisional_candidates": math.ceil(
                PROVISIONAL_POOL_FRACTION * count
            ),
            "required_final_hard_rows": math.ceil(FINAL_HARD_TIER_CAP * count),
        }
        for stratum, count in sampler_v13.STRATUM_QUOTAS.items()
    }


def build_preregistration() -> dict:
    manifest, v13, v14a, v14b, layer = load_bound_inputs()
    files = {
        key: {"path": str(path), "file_sha256": digest}
        for key, (path, digest) in BOUND_FILES.items()
    }
    panel_contract = _panel_contract(manifest)
    tier_counts = _tier_counts()
    candidate_rows_per_panel = sum(
        item["provisional_candidates"] for item in tier_counts.values()
    )
    hard_rows_per_panel = sum(
        item["required_final_hard_rows"] for item in tier_counts.values()
    )
    value = {
        "schema": "eggroll-es-lagged-replay-calibration-preregistration-v35a",
        "status": "preregistered_calibration_only_no_replay_HPO_or_update_authority",
        "scientific_objective": (
            "freeze_a_train_only_independent_base_probe_difficulty_tier_then_"
            "manually_exclude_incorrect_or_malformed_examples_before_testing_"
            "lagged_replay_on_a_fresh_perturbation_basis"
        ),
        "nonredundancy": {
            "v13_authoritative_sampler_retained": True,
            "v14a_document_first_failed": v14a["decision"]["sampler"] == "retain_v13",
            "v14b_k2_document_mean_failed": v14b["decision"]["sampler"] == "retain_v13",
            "difference_from_closed_v14_family": (
                "no_document_estimand_or_row_multiplicity_change; calibrate_a_"
                "lagged_hardness_mixture_with_untouched_train_screens"
            ),
            "v13_enablement_requirements_addressed": [
                "difficulty_artifact_bound_to_exact_v13_frame_and_panels",
                "difficulty_from_unperturbed_prior_probe_before_candidate_responses",
                "hard_tier_frozen_before_fresh_basis_HPO",
                "future_mixture_has_exact_stratum_preserving_weights",
                "hard_tier_and_future_replay_fraction_each_capped_at_0.25",
            ],
        },
        "strict_train_only_firewall": {
            "allowed_inputs": [
                "hash_bound_v13_train_optimization_panels",
                "hash_bound_base_Qwen3.6-35B-A3B",
                "manual_review_of_only_provisional_training_examples",
            ],
            "excluded_panels_not_generated_or_opened": list(EXCLUDED_SCREENS),
            "nontrain_evaluation_surfaces_opened": False,
            "dataset_rows_questions_answers_or_document_text_persisted_in_runtime_artifact": False,
            "raw_rewards_logprobs_tokens_outputs_or_manual_review_text_persisted": False,
            "model_update_checkpoint_or_dataset_promotion": False,
        },
        "bound_inputs": {
            "files": files,
            "file_bundle_sha256": canonical_sha256(files),
            "panel_manifest_content_sha256": PANEL_MANIFEST_CONTENT_SHA256,
            "v13_evidence_content_sha256": V13_EVIDENCE_CONTENT_SHA256,
            "v14a_negative_content_sha256": V14A_NEGATIVE_CONTENT_SHA256,
            "v14b_negative_content_sha256": V14B_NEGATIVE_CONTENT_SHA256,
            "v13_stability_block_sha256": canonical_sha256(v13["stability"]),
        },
        "frozen_recipe": {
            "model": str(MODEL_PATH),
            "model_config_sha256": BOUND_FILES["model_config"][1],
            "model_index_sha256": BOUND_FILES["model_index"][1],
            "checkpoint": None,
            "layer_plan": {
                "path": str(LAYER_PLAN_PATH),
                "file_sha256": BOUND_FILES["layer_plan"][1],
                "content_sha256": layer["plan_sha256"],
                "layers": layer["layers"],
            },
            "sigma": 0.0,
            "alpha": 0.0,
            "generation_seed": GENERATION_SEED,
            "temperature": 0.0,
            "max_tokens": MAX_TOKENS,
            "reward": "mean_gold_answer_token_logprob_from_teacher_forced_dense_scoring",
            "model_update_allowed": False,
        },
        "panels": {
            "contract": panel_contract,
            "generated": list(OPTIMIZATION_PANELS),
            "excluded_untouched_train_screens": list(EXCLUDED_SCREENS),
            "rows_per_optimization_panel": sampler_v13.PANEL_SIZE,
            "optimization_request_union": len(OPTIMIZATION_PANELS) * sampler_v13.PANEL_SIZE,
            "same_order_on_all_four_engines": True,
        },
        "hardware_and_generation": {
            "gpu_ids": [0, 1, 2, 3],
            "engine_count": 4,
            "tp_per_engine": 1,
            "all_four_engines_generate_the_complete_optimization_union": True,
            "requests_per_engine": len(OPTIMIZATION_PANELS) * sampler_v13.PANEL_SIZE,
            "total_requests": 4 * len(OPTIMIZATION_PANELS) * sampler_v13.PANEL_SIZE,
            "exact_tokens_and_logprobs_required_across_all_four_engines": True,
            "all_four_gpu_activity_and_cleanup_certificates_required": True,
        },
        "difficulty_calibration": {
            "score_direction": "lower_mean_gold_answer_token_logprob_is_harder",
            "ranking_unit": "within_each_optimization_panel_and_stratum",
            "tie_break": "ascending_row_sha256",
            "provisional_candidate_fraction": PROVISIONAL_POOL_FRACTION,
            "final_hard_tier_cap": FINAL_HARD_TIER_CAP,
            "tier_counts_per_panel": tier_counts,
            "provisional_candidates_per_panel": candidate_rows_per_panel,
            "required_final_hard_rows_per_panel": hard_rows_per_panel,
            "provisional_candidates_total": (
                candidate_rows_per_panel * len(OPTIMIZATION_PANELS)
            ),
            "required_final_hard_rows_total": (
                hard_rows_per_panel * len(OPTIMIZATION_PANELS)
            ),
            "no_candidate_response_or_fresh_HPO_basis_observed_before_tier_freeze": True,
        },
        "blinded_manual_quality_gate": {
            "reviewer_receives": (
                "deterministically_shuffled_training_rows_without_scores_ranks_"
                "or_future_HPO_results"
            ),
            "one_decision_per_provisional_candidate": ["eligible", "ineligible"],
            "eligible_requires_all": [
                "question_is_clear_and_useful",
                "answer_is_factually_supported_by_its_training_source",
                "answer_directly_answers_the_question",
                "no_template_protocol_or_control_token_leakage",
                "no_unsafe_instruction_or_missing_material_safety_context",
            ],
            "ineligible_rows_are_not_rewritten_or_replaced_inside_this_experiment": True,
            "selection_after_audit": (
                "for_each_panel_and_stratum_take_the_lowest_calibration_ranked_"
                "eligible_rows_until_the_frozen_required_count"
            ),
            "insufficient_eligible_candidates": "fail_calibration_and_authorize_nothing",
            "audit_artifact_persists_only_row_sha256_and_eligibility_enum": True,
            "review_text_or_dataset_content_persisted_in_experiment_artifact": False,
        },
        "required_content_free_calibration_artifact": {
            "provisional_pool": (
                "row_sha256_panel_stratum_and_calibration_rank_only;_no_scores"
            ),
            "manual_audit": "row_sha256_and_eligibility_enum_only",
            "final_hard_tier": "row_sha256_panel_stratum_and_rank_only",
            "score_vectors": "sha256_commitments_only",
            "cross_engine_outputs": "sha256_commitments_and_exact_equality_boolean_only",
            "raw_content_scores_tokens_logprobs_outputs_or_pids": False,
            "self_hash_and_every_bound_file_hash_required": True,
        },
        "future_HPO_boundary": {
            "recommended_fixed_sequence_replay_fractions": list(
                REPLAY_FRACTIONS_RECOMMENDED
            ),
            "interpretation": "90_10_then_80_20_lagged_replay_mixtures",
            "mixture_algebra": (
                "within_each_panel_stratum:(1-f)*uniform_all_rows+f*uniform_"
                "frozen_hard_tier;preserve_v13_stratum_mass"
            ),
            "candidate_responses_must_use_a_fresh_direction_basis": True,
            "same_requests_and_perturbations_for_control_and_all_fractions": True,
            "untouched_train_screens_used_only_after_tier_freeze": True,
            "fixed_sequence_familywise_noninferiority_gate_required": True,
            "separate_preregistration_and_runtime_adapter_required": True,
        },
        "authority": {
            "run_this_calibration_after_committed_runtime_adapter": True,
            "run_replay_HPO": False,
            "choose_or_adopt_replay_fraction": False,
            "apply_model_update_or_write_checkpoint": False,
            "open_nontrain_evaluation": False,
            "promote_or_mutate_dataset": False,
            "reuse_for_nontrain_semantic_claims": False,
        },
        "runtime_integrity": {
            "fresh_exclusive_attempt_and_output_paths": True,
            "committed_clean_source_bundle": True,
            "prelaunch_and_final_all_gpu_idle_certificates": True,
            "all_four_engine_activity_required": True,
            "exact_cross_engine_output_equivalence_required": True,
            "full_context_A_B_before_and_A_C_after_required": True,
            "failure_cleanup_and_final_idle_required": True,
            "post_cleanup_bound_hash_revalidation_required": True,
            "compact_content_free_persistence_only": True,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def exclusive_write(path: Path, value: dict) -> None:
    path = Path(path).resolve()
    if path != OUTPUT_PATH:
        raise ValueError("v35a preregistration output path changed")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise ValueError("v35a preregistration already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    value = build_preregistration()
    if not args.dry_run:
        exclusive_write(args.output, value)
    print(json.dumps({
        "schema": "eggroll-es-lagged-replay-calibration-preregistration-build-v35a",
        "content_sha256": value["content_sha256_before_self_field"],
        "optimization_requests": value["hardware_and_generation"]["total_requests"],
        "provisional_candidates": value["difficulty_calibration"][
            "provisional_candidates_total"
        ],
        "required_hard_rows": value["difficulty_calibration"][
            "required_final_hard_rows_total"
        ],
        "gpu_launched": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
