#!/usr/bin/env python3
"""Preregister the train-only paired production/v283 compatibility A/B."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path

import build_eggroll_es_joint_panels_v17a as frame_v17a
import train_eggroll_es_specialist_anchor_v13 as anchor_v13


ROOT = Path(__file__).resolve().parent
PREREGISTRATION_PATH_V17A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_PAIRED_DATA_COMPAT_V17A_PREREGISTRATION.json"
).resolve()
PROTOCOL_PATH_V17A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_PAIRED_DATA_COMPAT_V17A_PROTOCOL.md"
).resolve()
FRAME_PATH_V17A = frame_v17a.OUTPUT_PATH_V17A
FRAME_FILE_SHA256_V17A = (
    "bfd53bb2c2148381e0b5b9b24102a67e20ef65f7dabe96314e097ad800ea7ff1"
)
FRAME_CONTENT_SHA256_V17A = (
    "eaad58e01a429cae00f85af1d057dbde9e72a3a0c196cd90089ad0f1366ca194"
)
FRAME_BUILDER_FILE_SHA256_V17A = (
    "57da9ef815b169a4492169b07f794fdcc2bd44d6721c2e708be2e5e6582cd1a0"
)
FRAME_TEST_FILE_SHA256_V17A = (
    "cc780da59f54f4166e73bb8a62e1dd614c661cc30a109d353aeca6116fbe1d09"
)
CANDIDATE_FREEZE_COMMIT_V17A = (
    "2bf505e"
)
CANDIDATE_TEST_PATH_V17A = (
    ROOT / "experiments/eggroll_es_hpo/dataset_candidates/v283/"
    "test_frozen_candidate.py"
).resolve()
CANDIDATE_TEST_FILE_SHA256_V17A = (
    "168b711ecd50e43e7698d311b89e2ca51b1d7a242189eede8846f74396b197af"
)
V13_EVIDENCE_PATH_V17A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V13B_TRAIN_PANEL_AGGREGATE_EVIDENCE_V14A.json"
).resolve()
V13_EVIDENCE_FILE_SHA256_V17A = (
    "d367c9c4de1e1f3526ddb3dfba2f5bf24efc77cbccf951f7359eb1969fcd7b54"
)
V13_EVIDENCE_CONTENT_SHA256_V17A = (
    "06f662574013345a6c777af8688a38f3941286d9e11a427ed3342de53451b1e3"
)
LAYER_PLAN_PATH_V17A = (
    ROOT / "experiments/layer_plans/middle_late_dense_v6.json"
).resolve()
LAYER_PLAN_FILE_SHA256_V17A = (
    "d65d702969dcec7a56ca4fcf461d402c44642966191a57c2ef092ec339e3e3df"
)
LAYER_PLAN_SHA256_V17A = (
    "03745c603a6b48898b41afbd4d9121aef276d7e45ca1a3ae14607ec5d1042cb9"
)
MODEL_CONFIG_SHA256_V17A = (
    "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
)
DENSE_REWARD_CONFIG_SHA256_V17A = (
    "4941f2e94091b1f8e7ab7b5294ebc6520b80aba1326b7dc6ccea5140a3da5da2"
)
V13_IMPLEMENTATION_PATHS_V17A = {
    "trainer_v13": ROOT / "train_eggroll_es_specialist_anchor_v13.py",
    "worker_v13": ROOT / "eggroll_es_worker_v13.py",
    "driver_v13": ROOT / "run_eggroll_es_train_panels_v13.py",
    "tests_v13": ROOT / "test_eggroll_es_train_panels_runtime_v13.py",
}
V13_IMPLEMENTATION_HASHES_V17A = {
    "trainer_v13": "1a8a4145a85c183bb6121914357b7e6bce916b4f76a0693887ac41fa3a8c4c6e",
    "worker_v13": "5596bff9174e5e94e812181a51f8cc9f9b2a73f3a4cb58c45d5346147c8d6367",
    "driver_v13": "1fcd287c62084588d4264376eea01f216bef390561cbd5078ee2f77bac552ce0",
    "tests_v13": "16346a09e6d4e274919cece443c80c221a1f40d89d570b38c657217d58ebfa10",
}

EXPERIMENT_NAME_V17A = (
    "snapshot_v283_data_v17a_paired_production_candidate_5x38_"
    "alpha_zero_middle_late_basis20260714"
)
ARM_ORDER_V17A = ("production", "candidate_v283")
SIGMA_V17A = 0.0003
ALPHA_V17A = 0.0
POPULATION_SIZE_V17A = 32
BOOTSTRAP_SEED_V17A = 20260719
BOOTSTRAP_REPETITIONS_V17A = 20_000
FAMILYWISE_ALPHA_V17A = 0.05

METRIC_FAMILIES_V17A = (
    "optimization_pairwise_cosine",
    "optimization_pairwise_sign_agreement",
    "aggregate_to_optimization_cosine",
    "aggregate_to_optimization_sign_agreement",
    "train_screen_cosine",
    "train_screen_sign_agreement",
)
ENDPOINT_CONTRACT_V17A = {
    f"{family}_{summary}": {
        "metric_family": family,
        "summary": summary,
        "noninferiority_margin": 0.0,
    }
    for family in METRIC_FAMILIES_V17A
    for summary in ("median", "worst")
}
TOKEN_LENGTH_AUDIT_V17A = {
    "tokenizer_boundary_mismatch_count": {
        "production": 0, "candidate_v283": 0,
    },
    "over_frozen_1024_total_token_cap_count": {
        "production": 0, "candidate_v283": 0,
    },
    "combined_prompt_answer_token_quantiles_p50_p90_p95_p99_max": {
        "production": [54, 63, 67, 86, 142],
        "candidate_v283": [67, 91, 106, 131, 144],
    },
    "candidate_answer_token_quantiles_p50_p90_p95_p99_max": [18, 42, 53, 81, 86],
}
LOWERCASE_SHA256_PATTERN_V17A = re.compile(r"[0-9a-f]{64}\Z")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_persisted_sha256_fields_v17a(value):
    """Reject every persisted SHA-labelled field that is not lowercase hex."""
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            for key, item in current.items():
                if (
                    isinstance(key, str)
                    and "sha256" in key
                    and (
                        not isinstance(item, str)
                        or LOWERCASE_SHA256_PATTERN_V17A.fullmatch(item) is None
                    )
                ):
                    raise RuntimeError(
                        "v17a persisted SHA-256 field is not lowercase hex"
                    )
                pending.append(item)
        elif isinstance(current, list):
            pending.extend(current)
    return True


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def load_bound_aggregate_evidence_v17a():
    expected = {
        frame_v17a.CANDIDATE_PATH_V17A: frame_v17a.CANDIDATE_SHA256_V17A,
        frame_v17a.CANDIDATE_MANIFEST_PATH_V17A:
            frame_v17a.CANDIDATE_MANIFEST_SHA256_V17A,
        frame_v17a.PRODUCTION_PATH_V17A: frame_v17a.PRODUCTION_SHA256_V17A,
        CANDIDATE_TEST_PATH_V17A: CANDIDATE_TEST_FILE_SHA256_V17A,
        FRAME_PATH_V17A: FRAME_FILE_SHA256_V17A,
        Path(frame_v17a.__file__).resolve(): FRAME_BUILDER_FILE_SHA256_V17A,
        ROOT / "test_eggroll_es_joint_panels_v17a.py": FRAME_TEST_FILE_SHA256_V17A,
        V13_EVIDENCE_PATH_V17A: V13_EVIDENCE_FILE_SHA256_V17A,
        LAYER_PLAN_PATH_V17A: LAYER_PLAN_FILE_SHA256_V17A,
        **{
            V13_IMPLEMENTATION_PATHS_V17A[name]: digest
            for name, digest in V13_IMPLEMENTATION_HASHES_V17A.items()
        },
    }
    if any(file_sha256(path) != digest for path, digest in expected.items()):
        raise RuntimeError("v17a frozen train-only evidence or code changed")
    frame = json.loads(FRAME_PATH_V17A.read_text())
    frame_v17a.validate_manifest_v17a(frame)
    candidate_manifest = json.loads(
        frame_v17a.CANDIDATE_MANIFEST_PATH_V17A.read_text()
    )
    v13 = json.loads(V13_EVIDENCE_PATH_V17A.read_text())
    if (
        frame.get("content_sha256_before_self_field")
        != FRAME_CONTENT_SHA256_V17A
        or candidate_manifest.get("schema")
        != "specialist-train-only-candidate-freeze-v1"
        or candidate_manifest.get("artifact", {}).get("sha256")
        != frame_v17a.CANDIDATE_SHA256_V17A
        or candidate_manifest.get("artifact", {}).get("is_overlay") is not False
        or v13.get("schema")
        != "eggroll-es-v13b-train-panel-aggregate-evidence-v14a"
        or v13.get("content_sha256_before_self_field")
        != V13_EVIDENCE_CONTENT_SHA256_V17A
        or canonical_sha256(_without_self(v13))
        != V13_EVIDENCE_CONTENT_SHA256_V17A
        or v13.get("contains_response_vectors_or_row_content") is not False
        or v13.get("contains_validation_ood_or_heldout_content") is not False
        or anchor_v13.PERTURBATION_BASIS_SHA256_V13
        != "29e7ceb1753c39b310a176d827e222b9a5b2c85edf9f2fef5c68b630b8fabc11"
        or anchor_v13.PERTURBATION_BASIS_SEED_V13 != 20260714
    ):
        raise RuntimeError("v17a aggregate train-only evidence contract changed")
    return frame, candidate_manifest, v13


def build_preregistration_v17a():
    frame, candidate_manifest, v13 = load_bound_aggregate_evidence_v17a()
    value = {
        "schema": "eggroll-es-paired-data-compat-preregistration-v17a",
        "status": "preregistered_runtime_not_yet_authorized",
        "experiment_name": EXPERIMENT_NAME_V17A,
        "hypothesis_count": 1,
        "hypothesis": (
            "on the exact common joint-conflict frame, v283 produces a train-"
            "only ES estimator that is noninferior in within-version stability "
            "to production under otherwise paired alpha-zero mechanics"
        ),
        "stage_separation": {
            "v17a": "paired_data_version_compatibility_only",
            "v17b": (
                "later_separate_full_candidate_5x39_sigma_alpha_hpo_only_if_"
                "v17a_passes"
            ),
            "v17a_common_only_frame_is_not_v17b_estimand": True,
            "v17a_pass_does_not_promote_dataset_or_authorize_update": True,
        },
        "inputs": {
            "candidate_freeze_commit": CANDIDATE_FREEZE_COMMIT_V17A,
            "candidate": copy.deepcopy(frame["inputs"]["candidate"]),
            "production": copy.deepcopy(frame["inputs"]["production"]),
            "candidate_role": candidate_manifest["artifact"]["candidate_role"],
            "joint_frame": {
                "path": str(FRAME_PATH_V17A),
                "file_sha256": FRAME_FILE_SHA256_V17A,
                "content_sha256": FRAME_CONTENT_SHA256_V17A,
                "joint_components": 276,
                "paired_units": 195,
                "selected_units": 190,
                "reserve_units": 5,
                "paired_strata": copy.deepcopy(
                    frame["joint_frame"]["paired_stratum_counts"]
                ),
            },
            "retained_v13_aggregate": {
                "path": str(V13_EVIDENCE_PATH_V17A),
                "file_sha256": V13_EVIDENCE_FILE_SHA256_V17A,
                "content_sha256": V13_EVIDENCE_CONTENT_SHA256_V17A,
                "stability": copy.deepcopy(v13["stability"]),
            },
        },
        "frozen_recipe": {
            "model": str((ROOT / "models/Qwen3.6-35B-A3B").resolve()),
            "layer_plan": {
                "description": "middle-late dense layers 20-23",
                "path": str(LAYER_PLAN_PATH_V17A),
                "file_sha256": LAYER_PLAN_FILE_SHA256_V17A,
                "plan_sha256": LAYER_PLAN_SHA256_V17A,
                "model_config_sha256": MODEL_CONFIG_SHA256_V17A,
            },
            "sigma": SIGMA_V17A,
            "alpha": ALPHA_V17A,
            "population_size": POPULATION_SIZE_V17A,
            "perturbation_basis_seed": anchor_v13.PERTURBATION_BASIS_SEED_V13,
            "perturbation_basis_sha256": anchor_v13.PERTURBATION_BASIS_SHA256_V13,
            "perturbation_seed_sha256": canonical_sha256(
                anchor_v13.PERTURBATION_SEEDS_V13
            ),
            "sign_order": ["plus", "minus"],
            "arm_order": list(ARM_ORDER_V17A),
            "arm_order_by_signed_wave": (
                "alternate_production_first_and_candidate_first"
            ),
            "same_resident_perturbation_scores_both_versions": True,
            "same_fixed_ordered_side_batch_every_direction_and_sign": True,
            "model_update_allowed": False,
        },
        "panels": {
            "names": list(frame_v17a.PANEL_NAMES_V17A),
            "optimization": list(frame_v17a.OPTIMIZATION_PANELS_V17A),
            "train_only_screens": list(frame_v17a.TRAIN_SCREENS_V17A),
            "panel_size": frame_v17a.PANEL_SIZE_V17A,
            "stratum_quotas": copy.deepcopy(frame_v17a.STRATUM_QUOTAS_V17A),
            "globally_disjoint_joint_units": True,
            "exact_horvitz_thompson_weights": True,
            "fixed_same_document_representatives": True,
            "screen_use": (
                "v17a_compatibility_only_not_reused_as_v17b_full_candidate_screens"
            ),
        },
        "scoring": {
            "objective": "per_example_mean_full_gold_answer_token_logprob",
            "dense_reward_config_sha256": DENSE_REWARD_CONFIG_SHA256_V17A,
            "prompt_contains_full_gold_answer": True,
            "prompt_logprobs": 1,
            "scored_positions": "all_aligned_answer_tokens_only",
            "max_tokens": 1,
            "max_tokens_role": "dummy_generation_trigger_not_answer_cap",
            "detokenize": False,
            "max_total_prompt_answer_tokens": 1024,
            "eos_scored": False,
            "token_length_audit": copy.deepcopy(TOKEN_LENGTH_AUDIT_V17A),
            "autoregressive_accuracy_or_token_f1_used": False,
            "objective_change_allowed_in_v17a": False,
        },
        "analysis": {
            "within_version_stability_only_for_gate": True,
            "cross_dataset_direction_cosine": "diagnostic_only",
            "robust_direction": (
                "coordinatewise_median_of_three_independently_standardized_"
                "Horvitz_Thompson_central_response_vectors"
            ),
            "bootstrap": {
                "seed": BOOTSTRAP_SEED_V17A,
                "repetitions": BOOTSTRAP_REPETITIONS_V17A,
                "resampling_unit": "paired_joint_conflict_unit_within_panel_stratum",
                "same_resample_indices_both_versions": True,
                "preserve_per_panel_stratum_counts": [14, 8, 2, 14],
                "recompute_each_replicate": (
                    "Horvitz_Thompson_panel_scores_then_32_coefficients_then_"
                    "all_nonlinear_median_and_worst_endpoints"
                ),
                "reserve_units_used": False,
                "persist_per_unit_scores_or_bootstrap_replicates": False,
                "familywise_alpha": FAMILYWISE_ALPHA_V17A,
                "endpoint_count": len(ENDPOINT_CONTRACT_V17A),
                "one_sided_quantile": (
                    FAMILYWISE_ALPHA_V17A / len(ENDPOINT_CONTRACT_V17A)
                ),
                "multiplicity": "Bonferroni_over_six_families_times_two_summaries",
            },
            "endpoint_contract": copy.deepcopy(ENDPOINT_CONTRACT_V17A),
        },
        "promotion_gate": {
            "all_rules_conjunctive": True,
            "all_file_content_source_and_runtime_integrity_audits": True,
            "both_versions_all_panel_spreads_nonzero": True,
            "candidate_median_and_worst_not_below_production_all_six_families": True,
            "every_zero_margin_paired_familywise_lcb_nonnegative": True,
            "cross_dataset_direction_similarity_used_for_gate": False,
            "pass_decision": (
                "authorize_only_drafting_and_separate_preregistration_of_v17b_"
                "full_candidate_5x39_train_only_hpo"
            ),
            "failure_decision": "retain_production_dataset_and_v13_recipe",
            "pass_does_not_authorize_dataset_promotion_update_or_evaluation": True,
        },
        "power": {
            "selected_fraction_of_paired_frame": 190 / 195,
            "per_panel_units": 38,
            "v13_per_panel_units": 56,
            "standard_error_inflation_vs_v13_if_iid": math.sqrt(56 / 38),
            "paired_common_random_numbers_reduce_version_delta_variance": True,
            "equipment_units_per_panel": 2,
            "stratum_specific_improvement_claim_authorized": False,
            "interpretation": (
                "compatibility_noninferiority_not_superiority_or_dataset_quality_"
                "promotion"
            ),
        },
        "hardware": {
            "gpu_ids": [0, 1, 2, 3],
            "engine_count": 4,
            "tp_per_engine": 1,
            "all_four_engines_required_every_signed_wave": True,
            "partial_waves_allowed": False,
            "estimated_wall_minutes": [15, 30],
            "estimated_four_gpu_hours": [1.0, 2.0],
        },
        "integrity": {
            "fresh_o_excl_attempt_and_run_required": True,
            "committed_source_bundle_required": True,
            "exact_reference_restore_after_both_version_calls_each_sign": True,
            "pre_post_base_probe_both_versions_must_match": True,
            "population_boundary_audit_required": True,
            "unselected_parameter_origin_must_not_change": True,
            "persist_response_vectors_or_row_content": False,
        },
        "required_runtime_adapter": {
            "status": "not_yet_implemented",
            "runtime_not_authorized_by_this_preregistration_commit": True,
            "must_bind_preregistration_file_and_content_hashes": True,
            "must_prove_no_model_update_entrypoint_reachable": True,
            "summary_diagnostic_exact_keys": [
                "used_for_gate", "content_sha256",
            ],
            "all_persisted_hash_fields_lowercase_hex": True,
        },
        "firewall": {
            "train_only": True,
            "evaluation_surfaces_opened": False,
            "heldout_opened": False,
            "benchmark_outcomes_used": False,
            "sigma_or_alpha_hpo_in_v17a": False,
            "model_update_or_checkpoint_write_in_v17a": False,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    validate_persisted_sha256_fields_v17a(value)
    return value


def evaluate_candidate_v17a(candidate):
    preregistration = build_preregistration_v17a()
    expected_keys = {
        "schema", "experiment_name", "alpha", "sigma",
        "model_update_applied", "evaluation_surfaces_opened",
        "frame_content_sha256", "perturbation_basis_sha256",
        "runtime_integrity", "versions", "paired_bootstrap",
        "cross_dataset_direction_similarity_diagnostic",
        "persisted_response_vectors_or_row_content",
        "content_sha256_before_self_field",
    }
    if (
        not isinstance(candidate, dict)
        or set(candidate) != expected_keys
        or candidate.get("schema")
        != "eggroll-es-paired-data-compat-summary-v17a"
        or candidate.get("experiment_name") != EXPERIMENT_NAME_V17A
        or candidate.get("alpha") != 0.0
        or candidate.get("sigma") != SIGMA_V17A
        or candidate.get("model_update_applied") is not False
        or candidate.get("evaluation_surfaces_opened") is not False
        or candidate.get("frame_content_sha256") != FRAME_CONTENT_SHA256_V17A
        or candidate.get("perturbation_basis_sha256")
        != anchor_v13.PERTURBATION_BASIS_SHA256_V13
        or candidate.get("persisted_response_vectors_or_row_content") is not False
        or candidate.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(candidate))
    ):
        raise RuntimeError("v17a candidate summary contract changed")
    validate_persisted_sha256_fields_v17a(candidate)
    integrity = candidate["runtime_integrity"]
    required_integrity = {
        "all_four_engines_every_signed_wave": True,
        "fixed_side_batch_identity_every_direction_and_sign": True,
        "same_resident_perturbation_both_versions": True,
        "alternating_version_order_complete": True,
        "exact_reference_restoration_passed": True,
        "pre_post_base_probes_equal_both_versions": True,
        "population_boundary_audit_passed": True,
        "tokenizer_and_prompt_logprob_contract_passed": True,
        "all_integrity_audits_passed": True,
    }
    if integrity != required_integrity:
        raise RuntimeError("v17a runtime integrity contract failed")
    if set(candidate["versions"]) != set(ARM_ORDER_V17A):
        raise RuntimeError("v17a version coverage changed")
    endpoint_names = set(ENDPOINT_CONTRACT_V17A)
    observed_results = {}
    for version in ARM_ORDER_V17A:
        result = candidate["versions"][version]
        if (
            set(result) != {
                "all_panel_spreads_nonzero", "endpoint_values",
                "compact_estimator_sha256",
            }
            or result.get("all_panel_spreads_nonzero") is not True
            or set(result.get("endpoint_values", {})) != endpoint_names
            or not isinstance(result.get("compact_estimator_sha256"), str)
            or len(result["compact_estimator_sha256"]) != 64
        ):
            raise RuntimeError("v17a within-version stability contract changed")
        if any(
            not isinstance(result["endpoint_values"][name], (int, float))
            or isinstance(result["endpoint_values"][name], bool)
            or not math.isfinite(float(result["endpoint_values"][name]))
            or not -1.0 - 1e-12 <= float(result["endpoint_values"][name]) <= 1.0 + 1e-12
            for name in ENDPOINT_CONTRACT_V17A
        ):
            raise RuntimeError("v17a within-version endpoint values changed")
    observed_results = {
        name: (
            float(candidate["versions"]["candidate_v283"]["endpoint_values"][name])
            >= float(candidate["versions"]["production"]["endpoint_values"][name])
        )
        for name in ENDPOINT_CONTRACT_V17A
    }
    bootstrap = candidate["paired_bootstrap"]
    if (
        set(bootstrap) != {
            "seed", "repetitions", "one_sided_quantile", "endpoints",
        }
        or bootstrap.get("seed") != BOOTSTRAP_SEED_V17A
        or bootstrap.get("repetitions") != BOOTSTRAP_REPETITIONS_V17A
        or bootstrap.get("one_sided_quantile")
        != FAMILYWISE_ALPHA_V17A / len(ENDPOINT_CONTRACT_V17A)
        or set(bootstrap.get("endpoints", {})) != endpoint_names
    ):
        raise RuntimeError("v17a paired bootstrap contract changed")
    bootstrap_results = {}
    for name, contract in ENDPOINT_CONTRACT_V17A.items():
        endpoint = bootstrap["endpoints"][name]
        if (
            set(endpoint) != {
                "candidate_minus_production", "familywise_lcb",
                "noninferiority_margin",
            }
            or endpoint.get("noninferiority_margin")
            != contract["noninferiority_margin"]
            or any(
                not isinstance(endpoint.get(key), (int, float))
                or isinstance(endpoint.get(key), bool)
                or not math.isfinite(float(endpoint[key]))
                for key in ("candidate_minus_production", "familywise_lcb")
            )
        ):
            raise RuntimeError("v17a bootstrap endpoint contract changed")
        expected_delta = (
            float(candidate["versions"]["candidate_v283"]["endpoint_values"][name])
            - float(candidate["versions"]["production"]["endpoint_values"][name])
        )
        if not math.isclose(
            float(endpoint["candidate_minus_production"]), expected_delta,
            rel_tol=1e-12, abs_tol=1e-12,
        ):
            raise RuntimeError("v17a observed paired endpoint delta changed")
        bootstrap_results[name] = (
            float(endpoint["familywise_lcb"])
            >= 0.0
        )
    diagnostic = candidate["cross_dataset_direction_similarity_diagnostic"]
    if (
        not isinstance(diagnostic, dict)
        or set(diagnostic) != {"used_for_gate", "content_sha256"}
        or diagnostic.get("used_for_gate") is not False
    ):
        raise RuntimeError("v17a cross-dataset diagnostic contract changed")
    passed = (
        all(observed_results.values())
        and all(bootstrap_results.values())
    )
    gate = {
        "schema": "eggroll-es-paired-data-compat-gate-v17a",
        "eligible_for_separate_v17b_preregistration": passed,
        "eligible_for_dataset_promotion": False,
        "eligible_for_model_update": False,
        "eligible_to_open_evaluation": False,
        "observed_candidate_not_below_production": observed_results,
        "paired_noninferiority_results": bootstrap_results,
        "cross_dataset_direction_similarity_used_for_gate": False,
        "candidate_content_sha256": candidate[
            "content_sha256_before_self_field"
        ],
        "pass_decision": (
            "authorize_only_separate_v17b_train_only_preregistration"
            if passed else None
        ),
        "failure_decision": (
            None if passed else "retain_production_dataset_and_v13_recipe"
        ),
    }
    gate["content_sha256_before_self_field"] = canonical_sha256(gate)
    return gate


def _exclusive_write(path, value):
    path = Path(path).resolve()
    if path != PREREGISTRATION_PATH_V17A:
        raise ValueError("v17a preregistration output path changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("v17a preregistration already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(PREREGISTRATION_PATH_V17A))
    args = parser.parse_args(argv)
    if Path(args.output).resolve() != PREREGISTRATION_PATH_V17A:
        raise ValueError("v17a preregistration output path changed")
    value = build_preregistration_v17a()
    _exclusive_write(args.output, value)
    result = {
        "schema": "eggroll-es-paired-data-compat-prereg-write-v17a",
        "path": str(PREREGISTRATION_PATH_V17A),
        "file_sha256": file_sha256(PREREGISTRATION_PATH_V17A),
        "content_sha256": value["content_sha256_before_self_field"],
        "gpu_launched": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
