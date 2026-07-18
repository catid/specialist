#!/usr/bin/env python3
"""Seal category/source balance constraints for verified domain candidates.

This builder reads only the train-derived high-information plan metadata.  It
does not inspect development, final, protected, holdout, OOD, terminal,
incident, or manual-review sources.  The output is a selection contract, not a
selected dataset and not training authorization.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import build_high_information_domain_corpus_v1 as corpus
import verify_high_information_candidates_v1 as structural


OUTPUT = corpus.OUTPUT_DIR / "category_balanced_candidate_selection_contract.json"
TARGET_GENERATED_ASSISTANT_TOKENS = 740_847
SEED_QA_ASSISTANT_TOKENS = 9_153
TOTAL_DOMAIN_ASSISTANT_TOKENS = 750_000
MINIMUM_ACCEPTED_TOKENS_PER_SOURCE = 2_000

CATEGORY_SOURCE_TARGETS: dict[str, dict[str, int]] = {
    "lineage_people_history": {
        "shibari_atlas": 148_169,
    },
    "tying_knots_frictions_technique": {
        "rope365": 30_000,
        "rope_topia": 8_000,
        "mdpi_tree_climbing_friction_hitch_2021": 30_000,
        "mdpi_knot_efficiency_statistics_2022": 26_000,
        "acta_loop_knot_efficiency_experiments_2020": 26_000,
    },
    "rigging_hardpoints_uplines_mechanics": {
        "crash_restraint": 70_000,
        "usfs_rigging_for_trail_work": 25_000,
        "usbr_testing_verifying_rope_access_anchors": 20_000,
        "hse_treework_lifting_and_climbing": 20_000,
        "hse_temporary_works_faqs": 15_000,
        "nistir_6096_post_installed_anchors_review": 20_000,
        "gutenberg_brady_kedge_anchor_77729": 10_000,
        "noaa_eight_strand_rope_structural_model_1989": 20_000,
    },
    "safety_anatomy_consent_risk": {
        "kink_education_code_of_conduct": 30_000,
        "europepmc_bdsm_fatality_review": 32_000,
        "europepmc_entrapment_neuropathy_review": 30_000,
        "europepmc_icar_suspension_syndrome": 30_000,
        "europepmc_rope_neuropathy_study": 28_000,
    },
    "materials_inspection_care_equipment": {
        "nist_manila_rope_color_serviceability_1933": 8_000,
        "nist_manila_rope_statistics_1947": 10_000,
        "nist_manila_rope_tests_t198_1921": 13_000,
        "noaa_synthetic_rope_deterioration_1990": 14_000,
        "nps_textile_fiber_aging_appendix_k_2002": 12_000,
        "phm_synthetic_fiber_rope_condition_monitoring_review_2017": 14_000,
        "maib_zarga_hmpe_inspection_failure_2017": 14_000,
        "maib_throwbag_hidden_fused_joints_2019": 10_000,
        "innotrac_camera_visual_rope_inspection_2020": 12_000,
        "sage_aramid_three_strand_contact_forces_2025": 15_678,
    },
}

TASK_SUBTYPE_TARGETS = {
    "application_scenario": 202_540,
    "calibrated_unanswerable": 18_032,
    "comparison_or_mechanism": 103_854,
    "conflict_or_scope_resolution": 29_085,
    "direct_explanation": 155_487,
    "evidence_grounded_answer": 90_129,
    "misconception_correction": 78_966,
    "multi_fact_synthesis": 62_754,
}
TASK_FAMILY_TARGETS = {
    "closed_book_application": 540_847,
    "grounded_synthesis": 200_000,
}
GENERATION_MODE_TARGETS = {
    "positive": 684_597,
    "calibrated_hard_negative": 56_250,
}


def source_targets() -> dict[str, int]:
    flattened = {
        source: target
        for category in CATEGORY_SOURCE_TARGETS.values()
        for source, target in category.items()
    }
    if sum(len(value) for value in CATEGORY_SOURCE_TARGETS.values()) != len(flattened):
        raise RuntimeError("a source is assigned to multiple primary categories")
    return flattened


def category_targets() -> dict[str, int]:
    return {
        category: sum(values.values())
        for category, values in CATEGORY_SOURCE_TARGETS.items()
    }


def validate_static_targets() -> None:
    sources = source_targets()
    categories = category_targets()
    expected = TARGET_GENERATED_ASSISTANT_TOKENS
    if (
        sum(sources.values()) != expected
        or sum(categories.values()) != expected
        or sum(TASK_SUBTYPE_TARGETS.values()) != expected
        or sum(TASK_FAMILY_TARGETS.values()) != expected
        or sum(GENERATION_MODE_TARGETS.values()) != expected
        or SEED_QA_ASSISTANT_TOKENS + expected != TOTAL_DOMAIN_ASSISTANT_TOKENS
        or len(sources) != 29
        or min(sources.values()) < MINIMUM_ACCEPTED_TOKENS_PER_SOURCE
        or sources.get("shibari_atlas") != categories["lineage_people_history"]
        or sources["shibari_atlas"] / expected > 0.20
    ):
        raise RuntimeError("category-balanced target arithmetic changed")


def audit_current_plan(
    contexts: Mapping[str, dict], requests: Sequence[dict]
) -> dict:
    source_by_context = {
        context_id: context["resource_id"] for context_id, context in contexts.items()
    }
    planned_by_source: Counter[str] = Counter()
    subtype: Counter[str] = Counter()
    family: Counter[str] = Counter()
    mode: Counter[str] = Counter()
    for request in requests:
        context_id = request.get("source_context_id")
        if context_id not in source_by_context:
            raise RuntimeError("generation request lacks sealed source metadata")
        tokens = request.get("target_verified_assistant_tokens")
        if not isinstance(tokens, int) or tokens <= 0:
            raise RuntimeError("generation request token target changed")
        planned_by_source[source_by_context[context_id]] += tokens
        subtype[request["task_subtype"]] += tokens
        family[request["task_family"]] += tokens
        mode[request["generation_mode"]] += tokens

    targets = source_targets()
    if set(planned_by_source) != set(targets):
        raise RuntimeError("current plan resource inventory changed")
    source_category = {
        source: category
        for category, sources in CATEGORY_SOURCE_TARGETS.items()
        for source in sources
    }
    planned_by_category: Counter[str] = Counter()
    for source, tokens in planned_by_source.items():
        planned_by_category[source_category[source]] += tokens
    total = sum(planned_by_source.values())
    if (
        total != TARGET_GENERATED_ASSISTANT_TOKENS
        or dict(sorted(subtype.items())) != dict(sorted(TASK_SUBTYPE_TARGETS.items()))
        or dict(sorted(family.items())) != dict(sorted(TASK_FAMILY_TARGETS.items()))
        or dict(sorted(mode.items())) != dict(sorted(GENERATION_MODE_TARGETS.items()))
    ):
        raise RuntimeError("current generation plan accounting changed")
    largest_source, largest_tokens = max(
        planned_by_source.items(), key=lambda item: (item[1], item[0])
    )
    return {
        "target_verified_assistant_tokens": total,
        "planned_tokens_by_source": dict(sorted(planned_by_source.items())),
        "planned_tokens_by_category": dict(sorted(planned_by_category.items())),
        "largest_source": largest_source,
        "largest_source_tokens": largest_tokens,
        "largest_source_fraction": largest_tokens / total,
        "task_subtype_tokens": dict(sorted(subtype.items())),
        "task_family_tokens": dict(sorted(family.items())),
        "generation_mode_tokens": dict(sorted(mode.items())),
    }


def construct() -> dict:
    validate_static_targets()
    _, contexts, request_index = structural.load_plan()
    plan_manifest = json.loads(corpus.MANIFEST.read_text(encoding="utf-8"))
    requests = sorted(request_index.values(), key=lambda value: value["request_id"])
    audit = audit_current_plan(contexts, requests)
    sources = source_targets()
    categories = category_targets()
    if set(sources) != {context["resource_id"] for context in contexts.values()}:
        raise RuntimeError("selection source targets do not cover every train resource")

    seed_receipt = plan_manifest["materialized_seed_qa"]
    contract = {
        "schema": "category-balanced-candidate-selection-contract-v1",
        "status": "contract_only_candidates_and_manual_review_pending",
        "purpose": (
            "prevent source-group count and Shibari Atlas volume from dominating "
            "the accepted knowledge QA corpus"
        ),
        "input_boundary": {
            "only_train_derived_high_information_plan_metadata_opened": True,
            "development_final_protected_holdout_ood_terminal_incident_or_manual_review_opened": False,
            "mixed_source_lineage_paths_dereferenced": False,
        },
        "input_receipts": {
            "plan_manifest_path": corpus.relative(corpus.MANIFEST),
            "plan_manifest_file_sha256": corpus.file_sha256(corpus.MANIFEST),
            "plan_manifest_self_sha256": plan_manifest[
                "content_sha256_before_self_field"
            ],
            "source_contexts": plan_manifest["source_contexts"],
            "generation_request_shards": plan_manifest["generation_requests"][
                "gpu_shards"
            ],
            "seed_qa": seed_receipt,
        },
        "token_accounting": {
            "opaque_precurated_seed_qa_assistant_tokens": SEED_QA_ASSISTANT_TOKENS,
            "category_balanced_generated_assistant_tokens": TARGET_GENERATED_ASSISTANT_TOKENS,
            "total_domain_assistant_tokens": TOTAL_DOMAIN_ASSISTANT_TOKENS,
            "seed_qa_category_is_opaque_and_not_used_to_relax_generated_targets": True,
        },
        "current_source_group_weighted_plan_audit": audit,
        "accepted_token_targets": {
            "category_tokens": categories,
            "source_tokens": sources,
            "task_subtype_tokens": TASK_SUBTYPE_TARGETS,
            "task_family_tokens": TASK_FAMILY_TARGETS,
            "generation_mode_tokens": GENERATION_MODE_TARGETS,
            "all_axes_must_hold_simultaneously": True,
            "source_targets_are_exact_not_sampling_weights": True,
            "source_size_is_not_final_selection_weight": True,
        },
        "anti_dominance_gates": {
            "shibari_atlas_max_tokens": sources["shibari_atlas"],
            "shibari_atlas_max_fraction_of_generated": (
                sources["shibari_atlas"] / TARGET_GENERATED_ASSISTANT_TOKENS
            ),
            "minimum_accepted_tokens_per_source": MINIMUM_ACCEPTED_TOKENS_PER_SOURCE,
            "every_train_resource_has_an_explicit_target": True,
            "lineage_people_history_may_not_borrow_technical_deficits": True,
            "technical_deficits_may_not_be_filled_with_lineage_or_identity_trivia": True,
        },
        "candidate_pool_contract": {
            "generation_passes_remain_separate_until_candidate_content_addressing": True,
            "duplicate_request_ids_across_generation_passes_are_expected": True,
            "candidate_example_id_is_the_merge_identity": True,
            "primary_generation_path_pattern": (
                "data/training_inventory/high_information_domain_corpus_v1/"
                "generation_candidates_gpu{0..3}.jsonl"
            ),
            "quality_fill_path_pattern": (
                "data/training_inventory/high_information_domain_corpus_v1/"
                "generation_fill_candidates_gpu{0..3}.jsonl"
            ),
            "category_balanced_technical_deficit_pass_required": True,
            "candidate_structural_pass_is_not_semantic_acceptance": True,
        },
        "eligibility_gates": {
            "structural_verification": "pass",
            "two_pass_guided_semantic_consensus": "pass",
            "independent_nli": "pass_or_hard_negative_not_applicable",
            "manual_review_if_selected": "resolved_pass",
            "all_requested_facets_complete": True,
            "unsupported_claims": "none",
            "safety_and_attribution_scope": "preserved",
            "training_value_and_nontriviality": "pass",
            "eligible_before_all_gates": False,
        },
        "deduplication_and_selection": {
            "exact_key": "NFKC_casefold_whitespace_question_plus_answer",
            "exact_duplicate_rows_allowed": False,
            "near_duplicate_clustering_required": True,
            "near_duplicate_preference_order": [
                "complete over incomplete",
                "technical/application/mechanistic over isolated trivia",
                "higher independent entailment confidence",
                "stronger two-pass evidence agreement",
                "rarer source/category deficit",
                "candidate_example_id lexical tie-break",
            ],
            "same_fact_multiple_views_allowed_only_when_task_and_information_differ": True,
            "URL_or_page_location_memorization_allowed": False,
        },
        "exact_budget_solver": {
            "candidate_rows_are_atomic_and_may_not_be_truncated": True,
            "solve_source_category_and_task_constraints_jointly": True,
            "selection_tie_break_is_deterministic": True,
            "unfilled_exact_tokens_create_content_addressed_deficit_requests": True,
            "silent_cross_source_or_cross_category_reallocation_allowed": False,
            "overshoot_or_tolerance_may_not_be_reported_as_exact": True,
        },
        "selection_materialized": False,
        "semantic_verification_completed": False,
        "manual_review_completed": False,
        "training_launch_authorized": False,
        "builder": {
            "path": corpus.relative(Path(__file__).resolve()),
            "file_sha256": corpus.file_sha256(Path(__file__).resolve()),
        },
    }
    contract["content_sha256_before_self_field"] = corpus.canonical_sha256(contract)
    return contract


def build(*, check: bool = False) -> dict:
    contract = construct()
    payload = (
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if check:
        if not OUTPUT.is_file() or OUTPUT.read_bytes() != payload:
            raise RuntimeError("category-balanced selection contract is stale")
        return contract
    corpus.atomic_write(OUTPUT, payload)
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    value = build(check=args.check)
    print(
        json.dumps(
            {
                "path": corpus.relative(OUTPUT),
                "content_sha256": value["content_sha256_before_self_field"],
                "current_largest_source": value[
                    "current_source_group_weighted_plan_audit"
                ]["largest_source"],
                "current_largest_source_fraction": value[
                    "current_source_group_weighted_plan_audit"
                ]["largest_source_fraction"],
                "atlas_target_fraction": value["anti_dominance_gates"][
                    "shibari_atlas_max_fraction_of_generated"
                ],
                "training_launch_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
