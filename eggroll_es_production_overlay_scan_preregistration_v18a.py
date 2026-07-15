#!/usr/bin/env python3
"""Preregister the train-only V18A production-preserving patch scan."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import audit_eggroll_es_token_lengths_v18a as token_audit_v18a
import build_eggroll_es_overlay_frame_v18a as frame_v18a
import build_eggroll_es_v17a_compact_evidence as evidence_v17a
import eggroll_es_paired_data_compat_preregistration_v17a as prereg_v17a
import run_eggroll_es_paired_data_compat_v17a as driver_v17a


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH_V18A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_PRODUCTION_PATCH_SCAN_V18A_PREREGISTRATION.json"
).resolve()
FLOW_CERTIFICATE_PATH_V18A = frame_v18a.OUTPUT_PATH_V18A
V17A_EVIDENCE_PATH = evidence_v17a.OUTPUT_PATH_V17A

EXPERIMENT_NAME_V18A = (
    "snapshot_v298_production_patch_v18a_nested_thirds_"
    "5x52_plus3_alpha_zero_middle_late_basis20260714"
)
MATERIALIZER_FILE_SHA256_V18A = (
    "3734d87c80fdba6d5ca85d3b75fd7f43f83399e673312589aa62818f376c344a"
)
FRAME_BUILDER_FILE_SHA256_V18A = (
    "73d7c71233946996d64d776fb68c31609ad512379a15a75fae46dd0d8c395ba0"
)
FLOW_CERTIFICATE_FILE_SHA256_V18A = (
    "0887f936fd00d205fab46d490810732d16ddc2b34d96fdc507d43c20dec60f8e"
)
FLOW_CERTIFICATE_CONTENT_SHA256_V18A = (
    "6a844851cc4e5a07c08b338c5e48f3b5ab58dbeb10c765570f3fa02f20c77b3b"
)
V17A_EVIDENCE_FILE_SHA256 = (
    "1fb9a855eccc62d3faa61cfb7756f957728f46099eb1c93ebfacb480bddb3c99"
)
V17A_EVIDENCE_CONTENT_SHA256 = (
    "3566fb66bed9ca694877efad281b451b73c28cc6c0b8b50b7472d00e92f3a3d7"
)
TOKEN_AUDIT_BUILDER_FILE_SHA256_V18A = (
    "9ff344ce001673f21a3782c813f2545f7503c19f8f0cc6be7d57ae64bfc8e7f3"
)
TOKEN_AUDIT_FILE_SHA256_V18A = (
    "df1d3810e988c3ece4ef921643ffe226fa7bb7f2f91edf2895865afe78c7ee6f"
)
TOKEN_AUDIT_CONTENT_SHA256_V18A = (
    "8157f64794cfe34f50dc795bed338f4109aa84addb38f114932b95f3d7598329"
)

ARM_ORDER_V18A = frame_v18a.ARM_ORDER_V18A
ENDPOINT_NAMES_V18A = tuple(prereg_v17a.ENDPOINT_CONTRACT_V17A)
COMPARISON_COUNT_V18A = 3
HYPOTHESIS_COUNT_V18A = len(ENDPOINT_NAMES_V18A) * COMPARISON_COUNT_V18A
BOOTSTRAP_SEED_V18A = 20260724
BOOTSTRAP_REPETITIONS_V18A = 50_000
FAMILYWISE_ALPHA_V18A = 0.05

canonical_sha256 = prereg_v17a.canonical_sha256
file_sha256 = prereg_v17a.file_sha256


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def load_bound_inputs_v18a() -> tuple[dict, dict, dict, dict]:
    expected = {
        frame_v18a.CANDIDATE_PATH_V18A: frame_v18a.CANDIDATE_SHA256_V18A,
        frame_v18a.CANDIDATE_MANIFEST_PATH_V18A: (
            frame_v18a.CANDIDATE_MANIFEST_SHA256_V18A
        ),
        Path(frame_v18a.candidate_v298.__file__).resolve(): (
            MATERIALIZER_FILE_SHA256_V18A
        ),
        Path(frame_v18a.__file__).resolve(): FRAME_BUILDER_FILE_SHA256_V18A,
        FLOW_CERTIFICATE_PATH_V18A: FLOW_CERTIFICATE_FILE_SHA256_V18A,
        V17A_EVIDENCE_PATH: V17A_EVIDENCE_FILE_SHA256,
        Path(token_audit_v18a.__file__).resolve(): (
            TOKEN_AUDIT_BUILDER_FILE_SHA256_V18A
        ),
        token_audit_v18a.OUTPUT_PATH_V18A: TOKEN_AUDIT_FILE_SHA256_V18A,
    }
    if any(file_sha256(path) != digest for path, digest in expected.items()):
        raise RuntimeError("v18a immutable input binding changed")
    candidate_manifest = json.loads(
        frame_v18a.CANDIDATE_MANIFEST_PATH_V18A.read_text(encoding="utf-8")
    )
    frame_v18a.candidate_v298.validate_manifest_v298(candidate_manifest)
    flow = json.loads(FLOW_CERTIFICATE_PATH_V18A.read_text(encoding="utf-8"))
    frame_v18a.validate_certificate_v18a(flow)
    evidence = json.loads(V17A_EVIDENCE_PATH.read_text(encoding="utf-8"))
    evidence_v17a.validate_evidence_v17a(evidence)
    token_audit = json.loads(
        token_audit_v18a.OUTPUT_PATH_V18A.read_text(encoding="utf-8")
    )
    token_audit_v18a.validate_audit_v18a(token_audit)
    if (
        flow.get("content_sha256_before_self_field")
        != FLOW_CERTIFICATE_CONTENT_SHA256_V18A
        or evidence.get("content_sha256_before_self_field")
        != V17A_EVIDENCE_CONTENT_SHA256
        or evidence.get("status")
        != "valid_completed_run_preregistered_gate_failed"
        or evidence.get("decision", {}).get("retain_dataset") != "production"
        or evidence.get("decision", {}).get("retain_recipe") != "v13"
        or token_audit.get("content_sha256_before_self_field")
        != TOKEN_AUDIT_CONTENT_SHA256_V18A
    ):
        raise RuntimeError("v18a flow or V17A evidence content changed")
    return candidate_manifest, flow, evidence, token_audit


def scoring_contract_v18a(token_audit: dict) -> dict:
    scoring = copy.deepcopy(
        prereg_v17a.build_preregistration_v17a()["scoring"]
    )
    if scoring.pop("objective_change_allowed_in_v17a", None) is not False:
        raise RuntimeError("v18a inherited scoring objective contract changed")
    scoring["objective_change_allowed_in_v18a"] = False
    sources = token_audit["sources"]
    scoring["token_length_audit"] = {
        "quantile_method": token_audit["quantile_method"],
        "tokenizer_boundary_mismatch_count": {
            name: source["tokenizer_boundary_mismatch_count"]
            for name, source in sources.items()
        },
        "over_frozen_1024_total_token_cap_count": {
            name: source["over_frozen_1024_total_token_cap_count"]
            for name, source in sources.items()
        },
        "combined_prompt_answer_token_quantiles_p50_p90_p95_p99_max": {
            name: source[
                "combined_prompt_answer_token_quantiles_p50_p90_p95_p99_max"
            ]
            for name, source in sources.items()
        },
        "answer_token_quantiles_p50_p90_p95_p99_max": {
            name: source["answer_token_quantiles_p50_p90_p95_p99_max"]
            for name, source in sources.items()
        },
    }
    return scoring


def signed_wave_arm_orders_v18a() -> list[dict]:
    arms = list(ARM_ORDER_V18A)
    orders = []
    for population_wave in range(8):
        for sign_index, sign in enumerate(("plus", "minus")):
            rotation = (
                population_wave if sign == "plus" else population_wave + 2
            ) % len(arms)
            order = arms[rotation:] + arms[:rotation]
            orders.append({
                "signed_wave_index": 2 * population_wave + sign_index,
                "population_wave_index": population_wave,
                "sign": sign,
                "arm_order": order,
            })
    for sign in ("plus", "minus"):
        signed = [item for item in orders if item["sign"] == sign]
        for arm in arms:
            positions = [item["arm_order"].index(arm) for item in signed]
            if sorted(positions) != [0, 0, 1, 1, 2, 2, 3, 3]:
                raise RuntimeError("v18a signed arm-position balance changed")
    return orders


def build_preregistration_v18a() -> dict:
    candidate_manifest, flow, evidence, token_audit = load_bound_inputs_v18a()
    endpoint_contract = {
        f"{arm}__{endpoint}": {
            "arm": arm,
            "control": "production_only",
            "endpoint": endpoint,
            "noninferiority_margin": 0.0,
        }
        for arm in ARM_ORDER_V18A[1:]
        for endpoint in ENDPOINT_NAMES_V18A
    }
    arm_contract = {}
    maximum_tier = dict(zip(ARM_ORDER_V18A, range(4)))
    for arm in ARM_ORDER_V18A:
        tier = maximum_tier[arm]
        arm_contract[arm] = {
            "maximum_active_patch_tier": tier,
            "eligible_patch_population_components": (
                frame_v18a.ELIGIBLE_PATCH_POPULATIONS_V18A[arm]
            ),
            "eligible_patch_fraction_numerator_denominator": {
                "production_only": [0, 1],
                "patch_one_third": [1, 3],
                "patch_two_thirds": [2, 3],
                "patch_full": [1, 1],
            }[arm],
            "active_population_components": frame_v18a.ARM_POPULATIONS_V18A[arm],
            "requests_per_panel": frame_v18a.ARM_REQUESTS_PER_PANEL_V18A[arm],
            "active_base_categories": list(frame_v18a.BASE_CATEGORIES_V18A),
            "active_candidate_only_layers": list(range(1, tier + 1)),
            "paired_tiers_using_candidate_representative": list(
                range(1, tier + 1)
            ),
            "paired_tiers_using_production_representative": list(
                range(tier + 1, 4)
            ),
            "fallback_always_uses_production_representative": True,
            "same_arm_paired_duplicate_count": 0,
        }
    value = {
        "schema": "eggroll-es-production-patch-scan-preregistration-v18a",
        "status": "preregistered_frozen_frame_runtime_adapter_not_authorized",
        "experiment_name": EXPERIMENT_NAME_V18A,
        "motivation": {
            "v17a_result": "valid_negative_full_replacement_compatibility",
            "v17a_observed_pass_count": evidence["gate_summary"][
                "observed_pass_count"
            ],
            "v17a_bootstrap_pass_count": evidence["gate_summary"][
                "bootstrap_pass_count"
            ],
            "design_response": (
                "scan_nested_coverage_preserving_candidate_patches_without_"
                "retaining_superseded_production_QA_beside_candidate_QA"
            ),
            "quality_or_promotion_claim": False,
        },
        "immutable_inputs": {
            "candidate": {
                "version": 298,
                "path": str(frame_v18a.CANDIDATE_PATH_V18A),
                "rows": frame_v18a.CANDIDATE_ROWS_V18A,
                "file_sha256": frame_v18a.CANDIDATE_SHA256_V18A,
                "manifest_path": str(frame_v18a.CANDIDATE_MANIFEST_PATH_V18A),
                "manifest_file_sha256": frame_v18a.CANDIDATE_MANIFEST_SHA256_V18A,
                "manifest_content_sha256": candidate_manifest[
                    "content_sha256_before_self_field"
                ],
                "materializer_path": str(
                    Path(frame_v18a.candidate_v298.__file__).resolve()
                ),
                "materializer_file_sha256": MATERIALIZER_FILE_SHA256_V18A,
                "newer_candidate_discovery_allowed": False,
            },
            "production": {
                "path": str(frame_v18a.PRODUCTION_PATH_V18A),
                "rows": frame_v18a.PRODUCTION_ROWS_V18A,
                "file_sha256": frame_v18a.PRODUCTION_SHA256_V18A,
            },
            "flow_certificate": {
                "path": str(FLOW_CERTIFICATE_PATH_V18A),
                "file_sha256": FLOW_CERTIFICATE_FILE_SHA256_V18A,
                "content_sha256": FLOW_CERTIFICATE_CONTENT_SHA256_V18A,
                "builder_path": str(Path(frame_v18a.__file__).resolve()),
                "builder_file_sha256": FRAME_BUILDER_FILE_SHA256_V18A,
                "solver_status": flow["constrained_flow"]["solver"]["status"],
                "solver_mip_gap": flow["constrained_flow"]["solver"]["mip_gap"],
                "quota_relaxation_used": False,
            },
            "v17a_negative_evidence": {
                "path": str(V17A_EVIDENCE_PATH),
                "file_sha256": V17A_EVIDENCE_FILE_SHA256,
                "content_sha256": V17A_EVIDENCE_CONTENT_SHA256,
                "preregistered_gate_passed": False,
            },
            "token_length_audit": {
                "path": str(token_audit_v18a.OUTPUT_PATH_V18A),
                "file_sha256": TOKEN_AUDIT_FILE_SHA256_V18A,
                "content_sha256": TOKEN_AUDIT_CONTENT_SHA256_V18A,
                "builder_path": str(
                    Path(token_audit_v18a.__file__).resolve()
                ),
                "builder_file_sha256": TOKEN_AUDIT_BUILDER_FILE_SHA256_V18A,
                "candidate_label": "candidate_v298",
                "boundary_mismatch_count": 0,
                "over_frozen_1024_total_token_cap_count": 0,
            },
        },
        "patch_semantics": {
            "one_representative_per_joint_component_per_arm": True,
            "production_only_components_use": "production_representative",
            "eligible_paired_component_before_activation_uses": (
                "production_representative"
            ),
            "eligible_paired_component_after_activation_uses": (
                "candidate_representative_instead_of_production"
            ),
            "candidate_only_component_after_activation": "added_once",
            "fallback_components_use": "production_representative_every_arm",
            "ambiguous_shared_url_candidate_topic": "technique",
            "ambiguous_pair_action": "always_production_fallback",
            "paired_candidate_and_production_both_present_same_arm": False,
            "same_arm_paired_duplicate_count": 0,
        },
        "population": {
            "joint_components": 295,
            "production_base_components": 272,
            "eligible_shared_document_paired_components": 202,
            "candidate_only_components": 23,
            "ambiguous_paired_fallback_components": 1,
            "eligible_patch_components": 225,
            "paired_tier_populations": {"1": 67, "2": 68, "3": 67},
            "candidate_only_layer_populations": {"1": 8, "2": 7, "3": 8},
            "cumulative_patch_population_by_arm": copy.deepcopy(
                frame_v18a.ELIGIBLE_PATCH_POPULATIONS_V18A
            ),
            "arm_population_denominators": copy.deepcopy(
                frame_v18a.ARM_POPULATIONS_V18A
            ),
            "exact_thirds_of_eligible_225": True,
        },
        "panels": {
            "names": list(frame_v18a.PANEL_NAMES_V18A),
            "optimization": list(frame_v18a.PANEL_NAMES_V18A[:3]),
            "train_only_screens": list(frame_v18a.PANEL_NAMES_V18A[3:]),
            "base_category_quota_per_panel": {
                category: 13 for category in frame_v18a.BASE_CATEGORIES_V18A
            },
            "production_topic_quota_per_panel": copy.deepcopy(
                frame_v18a.PRODUCTION_TOPIC_QUOTAS_V18A
            ),
            "candidate_only_layer_quota_per_panel": {"1": 1, "2": 1, "3": 1},
            "arm_requests_per_panel": copy.deepcopy(
                frame_v18a.ARM_REQUESTS_PER_PANEL_V18A
            ),
            "globally_joint_component_disjoint": True,
            "fixed_side_representatives_every_direction_sign_and_arm": True,
            "screens_excluded_from_every_direction": True,
            "infeasible_action": (
                "abort_v18a_without_quota_seed_grouping_or_solver_fallback"
            ),
        },
        "arms": arm_contract,
        "estimator": {
            "base_ht_strata": copy.deepcopy(flow["estimand"]["base_population_ht_strata"]),
            "candidate_only_ht_strata": copy.deepcopy(
                flow["estimand"]["candidate_only_ht_strata"]
            ),
            "base_category_populations": copy.deepcopy(
                frame_v18a.BASE_CATEGORY_POPULATIONS_V18A
            ),
            "candidate_only_layer_populations": {
                str(key): item
                for key, item in frame_v18a.CANDIDATE_ONLY_LAYER_POPULATIONS_V18A.items()
            },
            "arm_total": flow["estimand"]["arm_total"],
            "arm_mean": flow["estimand"]["arm_mean"],
            "arm_population_denominators": copy.deepcopy(
                frame_v18a.ARM_POPULATIONS_V18A
            ),
            "candidate_ht_targets_only_active_sealed_subset": True,
            "candidate_ht_never_upweights_to_all_226": True,
            "plain_request_mean_used": False,
            "shared_component_counted_once_per_arm": True,
        },
        "frozen_recipe": {
            "model": str(driver_v17a.FROZEN_MODEL_V17A),
            "layers": [20, 21, 22, 23],
            "layer_plan": {
                "path": str(prereg_v17a.LAYER_PLAN_PATH_V17A),
                "file_sha256": prereg_v17a.LAYER_PLAN_FILE_SHA256_V17A,
                "plan_sha256": prereg_v17a.LAYER_PLAN_SHA256_V17A,
                "model_config_sha256": prereg_v17a.MODEL_CONFIG_SHA256_V17A,
            },
            "sigma": 0.0003,
            "alpha": 0.0,
            "population_size": 32,
            "perturbation_basis_seed": (
                prereg_v17a.anchor_v13.PERTURBATION_BASIS_SEED_V13
            ),
            "perturbation_basis_sha256": (
                prereg_v17a.anchor_v13.PERTURBATION_BASIS_SHA256_V13
            ),
            "perturbation_seed_sha256": canonical_sha256(
                prereg_v17a.anchor_v13.PERTURBATION_SEEDS_V13
            ),
            "moe_backend": driver_v17a._declared_moe_backend_v17a(),
            "model_update_allowed": False,
            "checkpoint_write_allowed": False,
            "evaluation_surfaces_opened": False,
        },
        "common_random_numbers": {
            "same_resident_perturbation_scores_all_four_arms_before_restore": True,
            "signed_wave_arm_orders": signed_wave_arm_orders_v18a(),
            "each_arm_appears_each_position_twice_per_sign": True,
            "exact_restore_and_verify_once_after_all_four_arms_each_sign": True,
            "pre_post_base_probe_all_four_arms_must_match": True,
            "all_four_tp1_engines_every_signed_wave": True,
            "partial_waves_allowed": False,
        },
        "scoring": scoring_contract_v18a(token_audit),
        "analysis": {
            "metric_families": list(prereg_v17a.METRIC_FAMILIES_V17A),
            "endpoint_names": list(ENDPOINT_NAMES_V18A),
            "patch_to_production_comparisons": list(ARM_ORDER_V18A[1:]),
            "endpoint_contract": endpoint_contract,
            "hypothesis_count": HYPOTHESIS_COUNT_V18A,
            "bootstrap": {
                "seed": BOOTSTRAP_SEED_V18A,
                "repetitions": BOOTSTRAP_REPETITIONS_V18A,
                "familywise_alpha": FAMILYWISE_ALPHA_V18A,
                "one_sided_quantile": (
                    FAMILYWISE_ALPHA_V18A / HYPOTHESIS_COUNT_V18A
                ),
                "multiplicity": (
                    "Bonferroni_over_three_patch_arms_times_twelve_stability_"
                    "endpoints"
                ),
                "panel_block_resampling": (
                    "resample_optimization_three_and_train_screen_two_panels_"
                    "within_role_with_shared_indices_all_arms"
                ),
                "within_panel_base_resampling": (
                    "resample_thirteen_components_within_each_of_four_base_"
                    "HT_strata_using_shared_uniforms_all_arms"
                ),
                "candidate_only_resampling": (
                    "one_component_per_active_layer_is_inherited_from_the_"
                    "resampled_panel_block"
                ),
                "recompute_HT_arm_totals_exact_denominator_coefficients_"
                "aggregate_and_all_nonlinear_endpoints_each_replicate": True,
                "persist_per_unit_scores_or_bootstrap_draws": False,
            },
        },
        "promotion_gate": {
            "all_rules_conjunctive_within_each_patch_arm": True,
            "observed_patch_not_below_production_all_twelve": True,
            "every_zero_margin_familywise_lcb_nonnegative_all_twelve": True,
            "all_panel_spreads_nonzero": True,
            "all_runtime_identity_restoration_boundary_and_backend_audits": True,
            "selection_rule": (
                "largest_preregistered_patch_fraction_passing_all_twelve_"
                "observed_and_all_twelve_familywise_bootstrap_rules"
            ),
            "no_passing_patch_decision": "retain_production_dataset_and_v13_recipe",
            "passing_patch_decision": (
                "authorize_only_a_separate_train_only_recipe_preregistration_"
                "for_the_selected_patch_fraction"
            ),
            "dataset_promotion_authorized": False,
            "model_update_authorized": False,
            "evaluation_authorized": False,
        },
        "runtime_estimate": {
            "requests_per_engine_per_signed_wave_all_arms": 214,
            "v17a_requests_per_engine_per_signed_wave": 380,
            "relative_prompt_request_count_vs_v17a": 214 / 380,
            "estimated_wall_minutes_four_gpus": [14, 28],
            "estimated_aggregate_gpu_hours": [0.9, 1.9],
            "bootstrap_cpu_repetitions": BOOTSTRAP_REPETITIONS_V18A,
        },
        "required_next_artifacts": {
            "separate_committed_runtime_adapter": True,
            "runtime_adapter_must_implement_patch_substitution_not_union": True,
            "runtime_adapter_must_recompute_subset_specific_HT": True,
            "runtime_launch_authorized_by_this_preregistration": False,
        },
        "firewall": {
            "train_only": True,
            "manual_candidate_seal_exactly_v298": True,
            "newer_candidate_discovery_allowed": False,
            "heldout_validation_ood_eval_or_benchmark_content_opened": False,
            "per_example_content_persisted": False,
            "model_update_checkpoint_or_evaluation_allowed": False,
            "gpu_launch_allowed": False,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    validate_preregistration_v18a(value)
    return value


def validate_preregistration_v18a(value: dict) -> dict:
    scoring = value.get("scoring", {})
    token_lengths = scoring.get("token_length_audit", {})
    if (
        not isinstance(value, dict)
        or value.get("schema")
        != "eggroll-es-production-patch-scan-preregistration-v18a"
        or value.get("status")
        != "preregistered_frozen_frame_runtime_adapter_not_authorized"
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(value))
        or tuple(value.get("arms", {})) != ARM_ORDER_V18A
        or value.get("population", {}).get("eligible_patch_components") != 225
        or value.get("population", {}).get("arm_population_denominators")
        != frame_v18a.ARM_POPULATIONS_V18A
        or value.get("panels", {}).get("arm_requests_per_panel")
        != frame_v18a.ARM_REQUESTS_PER_PANEL_V18A
        or value.get("analysis", {}).get("hypothesis_count") != 36
        or len(value.get("analysis", {}).get("endpoint_contract", {})) != 36
        or scoring.get("objective_change_allowed_in_v18a") is not False
        or "objective_change_allowed_in_v17a" in scoring
        or set(token_lengths.get("tokenizer_boundary_mismatch_count", {}))
        != {"candidate_v298", "production"}
        or token_lengths.get(
            "combined_prompt_answer_token_quantiles_p50_p90_p95_p99_max", {}
        ).get("candidate_v298") != [67, 91, 104, 129, 144]
        or token_lengths.get(
            "answer_token_quantiles_p50_p90_p95_p99_max", {}
        ).get("candidate_v298") != [19, 42, 52, 74, 86]
        or value.get("patch_semantics", {}).get(
            "paired_candidate_and_production_both_present_same_arm"
        ) is not False
        or value.get("promotion_gate", {}).get("dataset_promotion_authorized")
        is not False
        or value.get("promotion_gate", {}).get("model_update_authorized")
        is not False
        or value.get("promotion_gate", {}).get("evaluation_authorized")
        is not False
        or value.get("required_next_artifacts", {}).get(
            "runtime_launch_authorized_by_this_preregistration"
        ) is not False
        or value.get("firewall", {}).get("gpu_launch_allowed") is not False
        or value.get("firewall", {}).get(
            "heldout_validation_ood_eval_or_benchmark_content_opened"
        ) is not False
    ):
        raise RuntimeError("v18a production patch preregistration changed")
    prereg_v17a.validate_persisted_sha256_fields_v17a(value)
    return value


def _exclusive_write(path: Path, value: dict) -> None:
    if path.resolve() != OUTPUT_PATH_V18A:
        raise ValueError("v18a preregistration output path changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("immutable v18a preregistration already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH_V18A))
    args = parser.parse_args(argv)
    if Path(args.output).resolve() != OUTPUT_PATH_V18A:
        raise ValueError("v18a preregistration output path changed")
    value = build_preregistration_v18a()
    _exclusive_write(OUTPUT_PATH_V18A, value)
    result = {
        "schema": "eggroll-es-production-patch-prereg-write-v18a",
        "path": str(OUTPUT_PATH_V18A),
        "file_sha256": file_sha256(OUTPUT_PATH_V18A),
        "content_sha256": value["content_sha256_before_self_field"],
        "runtime_launch_authorized": False,
        "model_update_authorized": False,
        "evaluation_authorized": False,
        "gpu_launched": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
