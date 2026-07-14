#!/usr/bin/env python3
"""Immutable preregistration for the V14b k=2 train-only estimator."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import eggroll_es_hierarchical_preregistration_v14a as prereg_v14a
import eggroll_es_hierarchical_train_sampler_v14 as sampler_v14
import eggroll_es_paired_distinct_row_sampler_v14b as sampler_v14b
import eggroll_es_train_panel_sampler_v13 as sampler_v13


ROOT = Path(__file__).resolve().parent
PREREGISTRATION_PATH_V14B = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_PAIRED_DISTINCT_ROW_FULL_FRAME_V14B_PREREGISTRATION.json"
).resolve()
V14A_NEGATIVE_EVIDENCE_PATH_V14B = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V14A_FULL_FRAME_NEGATIVE_AGGREGATE_EVIDENCE_V14B.json"
).resolve()
V14A_NEGATIVE_EVIDENCE_FILE_SHA256_V14B = (
    "9329c09fec5d76e209cff36ac80ffbbec69f53fe7bae9edcc069421d626cc9e9"
)
V14A_NEGATIVE_EVIDENCE_CONTENT_SHA256_V14B = (
    "ee4ded3d974dfd0becaedb1007f96888e133db51e62130d2844ab9c25e2ccf2b"
)
V13_BASELINE_EVIDENCE_PATH_V14B = prereg_v14a.EVIDENCE_PATH_V14A
V13_BASELINE_EVIDENCE_FILE_SHA256_V14B = (
    "d367c9c4de1e1f3526ddb3dfba2f5bf24efc77cbccf951f7359eb1969fcd7b54"
)
V13_BASELINE_EVIDENCE_CONTENT_SHA256_V14B = (
    "06f662574013345a6c777af8688a38f3941286d9e11a427ed3342de53451b1e3"
)
SAMPLER_FILE_SHA256_V14B = (
    "6a827fbd4f42bd9c5b785eaa4ed5ed8b78663cbb7d035fbf0bb84e57c9cebe92"
)
BASE_SAMPLER_FILE_SHA256_V14B = (
    "6981a746d6e0fc0904603abaf584ab71b9cc8a777a9abc00f4d305a98ebd186a"
)
SOURCE_SHA256_V14B = sampler_v13.SOURCE_SHA256
SOURCE_ARROW_SHA256_V14B = sampler_v13.SOURCE_ARROW_SHA256
FRAME_SHA256_V14B = (
    "ce50531881f4b7044bf82fc3e8fd52d603ba53041fccc4b934aa307840862d6c"
)
PERTURBATION_BASIS_SHA256_V14B = prereg_v14a.PERTURBATION_BASIS_SHA256_V14A
LAYER_PLAN_PATH_V14B = (
    ROOT / "experiments/layer_plans/middle_late_dense_v6.json"
).resolve()
LAYER_PLAN_FILE_SHA256_V14B = (
    "d65d702969dcec7a56ca4fcf461d402c44642966191a57c2ef092ec339e3e3df"
)
LAYER_PLAN_SHA256_V14B = (
    "03745c603a6b48898b41afbd4d9121aef276d7e45ca1a3ae14607ec5d1042cb9"
)
MODEL_CONFIG_SHA256_V14B = (
    "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
)
EXPERIMENT_NAME_V14B = (
    "snapshot794_layer_v14b_k2_distinct_row_document_mean_matched56_"
    "crossfit_alpha_zero_resident_sign_basis20260714"
)
FULL_FRAME_IDENTITY_V14B = {
    "documents": 310,
    "single_row_documents": 139,
    "multirow_documents": 171,
    "prompts": 481,
    "ordered_document_identity_sha256": (
        "3574c320bee17d7df2f31b2cc35d0ee018e903702c1dc1c3dda2778c77e05c0f"
    ),
    "ordered_prompt_identity_sha256": (
        "b3e7c0eb24a04377fc7727cb1972fefdca016dd112c0076068f2527677866ba9"
    ),
    "content_sha256": (
        "9d9fc31e928948cae12d7dc4b5ffedfd9def8482a4e6af8e82dc7dcfce7cb3d4"
    ),
}
PANEL_NAMES_V14B = sampler_v14b.PANEL_NAMES
PANEL_ROLES_V14B = dict(sampler_v14b.PANEL_ROLES)
PANEL_IDENTITIES_V14B = {
    "optimization_0": {
        "document_allocation_iteration": 0, "documents": 56, "prompts": 92,
        "ordered_document_identity_sha256": (
            "92eb9f78251f8344f4ccc141b43bc51337f7bad45340c95e4528c9be0fc4be1a"
        ),
        "ordered_prompt_identity_sha256": (
            "5a075fbad2e2ecd9a2168d8a992def93d0e346a086b5b06858230aa84b5c7454"
        ),
        "content_sha256": (
            "0e348d25fe298cd38f9b266a96860fa540e74864d809f0a917f1679ad011b159"
        ),
        "base_v14_panel_content_sha256": (
            "03d3a6b1484e794bd6a8994597e33918810249ed8fb193f405b8f01656cc9a3c"
        ),
    },
    "optimization_1": {
        "document_allocation_iteration": 1, "documents": 56, "prompts": 81,
        "ordered_document_identity_sha256": (
            "24d5d6f79863a32ff0e266c26dae9ad510fb75d132b397f5e17687e095f69474"
        ),
        "ordered_prompt_identity_sha256": (
            "08a6314cbb6b0265369d8a152cc182cdb34cd3f7ed7c83612173288ccf93419d"
        ),
        "content_sha256": (
            "77dd5dabda5853b1d97c16b173ffa2ff0bd46906a1fd792520e4e2b23853c096"
        ),
        "base_v14_panel_content_sha256": (
            "dfe4b140e1fce5cdce2877dea67e0f4bb5c3a98bc8dca2ca447d712cd8293450"
        ),
    },
    "optimization_2": {
        "document_allocation_iteration": 2, "documents": 56, "prompts": 87,
        "ordered_document_identity_sha256": (
            "e4387b99c2942b373e997d005d28dfe31c8a0da0496b82de064e9afe912068ca"
        ),
        "ordered_prompt_identity_sha256": (
            "9ca75904c6fa375c188a6f57be27f4119efca1d2fcd58121c38c6c2df3634306"
        ),
        "content_sha256": (
            "fa18e8b504a2ffb7ddc321bb485f465ce83fe54791f0349feb81b26faa38f3b8"
        ),
        "base_v14_panel_content_sha256": (
            "9e315f00f6c775bcff7ccf2b88272e3969ffd7fd6f66fb41ebec0f71a1edc1c6"
        ),
    },
    "train_screen_0": {
        "document_allocation_iteration": 3, "documents": 56, "prompts": 88,
        "ordered_document_identity_sha256": (
            "808b3cbe4d800368fa00d413d32e6ef125795ccc7b61b644074f1c2666323317"
        ),
        "ordered_prompt_identity_sha256": (
            "7900da162c6b5da2e3678b2402c3cc0c440302686b1e1582fd3ee3e53a0ba66f"
        ),
        "content_sha256": (
            "f36b0a66654073cca01d7747c71ea1cdba88394c3a2a353781e86f3d43cf4afc"
        ),
        "base_v14_panel_content_sha256": (
            "0ffb9f2aedf1a2f42c38a27ccfbd86d9cce0c61137f1f79e5728af62a63413ab"
        ),
    },
    "train_screen_1": {
        "document_allocation_iteration": 4, "documents": 56, "prompts": 86,
        "ordered_document_identity_sha256": (
            "5d09f1ea579f10e0ff8424f8bcf6aad768b8be61844af9481dd43b7513e2a12d"
        ),
        "ordered_prompt_identity_sha256": (
            "1912a4b7631fa52f57103d5eb07d4bb084185beab58d33a44a93b9d6662ee118"
        ),
        "content_sha256": (
            "eb064f15db70528aff7754a6bd5d149d9c960a0ffee45519b112db620fb2fcb3"
        ),
        "base_v14_panel_content_sha256": (
            "e574097798875a025a595e7b97a7e0cb03655b1754361ba4134bd571fb8cb088"
        ),
    },
}
COMPLEMENT_IDENTITIES_V14B = {
    "train_screen_0": {
        "documents": 254, "prompts": 393,
        "ordered_document_identity_sha256": (
            "e99aa858a1938a924ccfc90a3dd5e05b163dbde5f19fe1b2c642a1456d177fba"
        ),
        "ordered_prompt_identity_sha256": (
            "7e101286eed1bd63c060ceb10fa64c5d0c99b4d3efcaa262991b602d3b84c005"
        ),
        "content_sha256": (
            "5c55889cbdcee4a925ecb150dacdd9e6801c44aad95ae70a467ac16fd5f7d0b7"
        ),
    },
    "train_screen_1": {
        "documents": 254, "prompts": 395,
        "ordered_document_identity_sha256": (
            "1af45c94b265d66ea9c0c19b3cbb67fd869af5a220aa72cc409fb67e059f6cf7"
        ),
        "ordered_prompt_identity_sha256": (
            "36a26428a1da718c9fdd984152ddd698fff0da7dedee5d8dbde37e9358d7ec95"
        ),
        "content_sha256": (
            "1f4efd6f4158fd512fbeb009f615ca3bf96364651b7da209a89ff4d2ad940d03"
        ),
    },
}
BASELINE_STABILITY_V14B = copy.deepcopy(prereg_v14a.BASELINE_STABILITY_V14A)


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    return sampler_v13.canonical_sha256(value)


def load_v14a_negative_evidence_v14b(path=V14A_NEGATIVE_EVIDENCE_PATH_V14B):
    path = Path(path).resolve()
    if (
        path != V14A_NEGATIVE_EVIDENCE_PATH_V14B
        or _file_sha256(path) != V14A_NEGATIVE_EVIDENCE_FILE_SHA256_V14B
    ):
        raise RuntimeError("v14b V14a negative evidence file identity changed")
    evidence = json.loads(path.read_text())
    if (
        evidence.get("schema")
        != "eggroll-es-v14a-full-frame-negative-aggregate-evidence-v14b"
        or evidence.get("passed") is not True
        or evidence.get("contains_response_vectors_or_dense_result_hashes")
        is not False
        or evidence.get(
            "contains_source_rows_questions_answers_or_document_content"
        ) is not False
        or evidence.get("contains_validation_ood_heldout_or_benchmark_content")
        is not False
        or evidence.get("aggregate_gate", {}).get(
            "eligible_for_train_only_sampler_adoption"
        ) is not False
        or evidence.get("aggregate_gate", {}).get("eligible_for_model_update")
        is not False
        or evidence.get("aggregate_gate", {}).get("baseline")
        != BASELINE_STABILITY_V14B
        or evidence.get("decision") != {
            "sampler": "retain_v13",
            "row_draw_iteration_1_confirmation_authorized": False,
            "evaluation_surface_opened": False,
            "model_update_authorized": False,
            "reason": "v14a_failed_its_preregistered_conjunctive_gate",
        }
        or evidence.get("content_sha256_before_self_field")
        != V14A_NEGATIVE_EVIDENCE_CONTENT_SHA256_V14B
        or evidence.get("content_sha256_before_self_field")
        != _canonical({
            key: value for key, value in evidence.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v14b V14a negative evidence content changed")
    return evidence


def load_v13_baseline_evidence_v14b():
    evidence = prereg_v14a.load_evidence_v14a(
        V13_BASELINE_EVIDENCE_PATH_V14B
    )
    if (
        _file_sha256(V13_BASELINE_EVIDENCE_PATH_V14B)
        != V13_BASELINE_EVIDENCE_FILE_SHA256_V14B
        or evidence.get("content_sha256_before_self_field")
        != V13_BASELINE_EVIDENCE_CONTENT_SHA256_V14B
    ):
        raise RuntimeError("v14b V13 baseline evidence changed")
    return evidence


def materialize_sampler_v14b():
    if (
        _file_sha256(sampler_v14b.__file__) != SAMPLER_FILE_SHA256_V14B
        or _file_sha256(sampler_v14.__file__) != BASE_SAMPLER_FILE_SHA256_V14B
    ):
        raise RuntimeError("v14b sampler implementation changed")
    rows, source_sha256 = sampler_v13.load_frozen_train()
    full = sampler_v14b.build_full_frame(rows)
    panels = sampler_v14b.build_matched_panels(rows, full)
    complements = sampler_v14b.build_screen_complements(full, panels)
    sampler_v14b.validate_full_frame(full, rows)
    sampler_v14b.validate_matched_panels(panels, rows, full)
    sampler_v14b.validate_screen_complements(complements, full, panels)
    if (
        source_sha256 != SOURCE_SHA256_V14B
        or full["frame_sha256"] != FRAME_SHA256_V14B
        or {
            **{key: full[key] for key in (
                "documents", "single_row_documents", "multirow_documents",
                "prompts", "ordered_document_identity_sha256",
                "ordered_prompt_identity_sha256",
            )},
            "content_sha256": full["content_sha256_before_self_field"],
        } != FULL_FRAME_IDENTITY_V14B
    ):
        raise RuntimeError("v14b frozen source or full frame changed")
    for name in PANEL_NAMES_V14B:
        panel = panels[name]
        expected = PANEL_IDENTITIES_V14B[name]
        actual = {
            key: panel[key] for key in (
                "document_allocation_iteration", "documents", "prompts",
                "ordered_document_identity_sha256",
                "ordered_prompt_identity_sha256",
                "base_v14_panel_content_sha256",
            )
        }
        actual["content_sha256"] = panel["content_sha256_before_self_field"]
        if actual != expected:
            raise RuntimeError(f"v14b frozen {name} matched56 identity changed")
    for name, expected in COMPLEMENT_IDENTITIES_V14B.items():
        complement = complements[name]
        actual = {
            key: complement[key] for key in (
                "documents", "prompts", "ordered_document_identity_sha256",
                "ordered_prompt_identity_sha256",
            )
        }
        actual["content_sha256"] = complement[
            "content_sha256_before_self_field"
        ]
        if actual != expected:
            raise RuntimeError(f"v14b frozen {name} complement identity changed")
    return rows, full, panels, complements


def build_preregistration_v14b():
    negative = load_v14a_negative_evidence_v14b()
    baseline = load_v13_baseline_evidence_v14b()
    _rows, full, panels, complements = materialize_sampler_v14b()
    if _file_sha256(LAYER_PLAN_PATH_V14B) != LAYER_PLAN_FILE_SHA256_V14B:
        raise RuntimeError("v14b middle-late layer plan file changed")
    value = {
        "schema": "eggroll-es-paired-distinct-row-preregistration-v14b",
        "status": "preregistered_not_launch_authorized",
        "experiment_name": EXPERIMENT_NAME_V14B,
        "selection_surface": "frozen_train_source_and_aggregate_evidence_only",
        "contains_source_row_or_response_content": False,
        "contains_validation_ood_heldout_or_benchmark_content": False,
        "v14a_negative_evidence": {
            "path": str(V14A_NEGATIVE_EVIDENCE_PATH_V14B),
            "file_sha256": V14A_NEGATIVE_EVIDENCE_FILE_SHA256_V14B,
            "content_sha256": V14A_NEGATIVE_EVIDENCE_CONTENT_SHA256_V14B,
            "decision": copy.deepcopy(negative["decision"]),
            "failed_rules": copy.deepcopy(
                negative["aggregate_gate"]["failed_rules"]
            ),
        },
        "v13_baseline_evidence": {
            "path": str(V13_BASELINE_EVIDENCE_PATH_V14B),
            "file_sha256": V13_BASELINE_EVIDENCE_FILE_SHA256_V14B,
            "content_sha256": V13_BASELINE_EVIDENCE_CONTENT_SHA256_V14B,
            "selection_surface": baseline["selection_surface"],
        },
        "sampling": {
            "sampler_path": str(Path(sampler_v14b.__file__).resolve()),
            "sampler_file_sha256": SAMPLER_FILE_SHA256_V14B,
            "base_v14_sampler_file_sha256": BASE_SAMPLER_FILE_SHA256_V14B,
            "source_sha256": SOURCE_SHA256_V14B,
            "source_arrow_sha256": SOURCE_ARROW_SHA256_V14B,
            "frame_sha256": FRAME_SHA256_V14B,
            "master_seed": sampler_v14b.MASTER_SEED,
            "without_replacement_algorithm": (
                "for each document sort every row by SHA256(master_seed, "
                "within-document-without-replacement, document_sha256, "
                "row_sha256), then take min(2,row_count)"
            ),
            "document_order_algorithm": (
                "sort all documents by SHA256(master_seed, "
                "full-frame-document-order, document_sha256), then emit each "
                "document's selected rows in selection-rank order"
            ),
            "hypothesis_count": 1,
            "documents": 310,
            "single_row_documents": 139,
            "multirow_documents": 171,
            "distinct_rows_per_multirow_document": 2,
            "generation_prompt_count_per_direction_and_sign": 481,
            "relative_prompt_increase_over_v14a": 481.0 / 310.0 - 1.0,
            "hard_replay_fraction": 0.0,
        },
        "full_frame": {
            **copy.deepcopy(FULL_FRAME_IDENTITY_V14B),
            "estimand": (
                "equal weight mean of 310 document means after averaging the "
                "one or two frozen distinct row rewards within each document"
            ),
            "row_rewards_pooled_before_document_aggregation": False,
            "document_means_computed_before_equal_document_mean": True,
        },
        "matched56_subpanels": {
            name: {
                **copy.deepcopy(PANEL_IDENTITIES_V14B[name]),
                "role": PANEL_ROLES_V14B[name],
                "same_document_allocation_as_v14a": True,
                "derived_from_full_frame_generation": True,
                "document_mean_reduction_precedes_ht_weighting": True,
            }
            for name in PANEL_NAMES_V14B
        },
        "crossfit_complements": {
            name: {
                **copy.deepcopy(COMPLEMENT_IDENTITIES_V14B[name]),
                "screen_documents_excluded": True,
                "derived_from_full_frame_generation": True,
                "document_mean_reduction_precedes_complement_mean": True,
            }
            for name in PANEL_NAMES_V14B[3:]
        },
        "runtime": {
            "model": str((ROOT / "models/Qwen3.6-35B-A3B").resolve()),
            "layer_plan": {
                "name": "middle_late",
                "path": str(LAYER_PLAN_PATH_V14B),
                "file_sha256": LAYER_PLAN_FILE_SHA256_V14B,
                "plan_sha256": LAYER_PLAN_SHA256_V14B,
                "model_config_sha256": MODEL_CONFIG_SHA256_V14B,
                "layers": [20, 21, 22, 23],
            },
            "alpha": 0.0,
            "model_update_allowed": False,
            "population_size": 32,
            "perturbation_basis_sha256": PERTURBATION_BASIS_SHA256_V14B,
            "sign_order": ["plus", "minus"],
            "engine_count": 4,
            "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "complete_four_direction_waves_required": True,
            "execution_order": (
                "eight waves; for each wave perturb/generate all four plus "
                "directions, restore all engines exactly, then perturb/generate "
                "all four minus directions and restore all engines exactly"
            ),
            "one_481_prompt_order_reused_across_every_direction_and_sign": True,
            "matched56_and_crossfit_derived_without_extra_generation": True,
            "update_rpc_reachable": False,
        },
        "integrity": {
            "pre_and_post_population_base_probe_must_match_exactly": True,
            "exact_reference_check_after_population_required": True,
            "population_boundary_audit_required": True,
            "restore_in_finally_after_every_sign_required": True,
            "all_four_engines_must_participate_in_every_signed_wave": True,
            "partial_wave_allowed": False,
            "fresh_o_excl_attempt_and_run_required": True,
            "committed_source_bundle_required": True,
            "dense_result_hash_per_direction_sign_required": True,
            "persist_per_document_reward_vectors": False,
            "persist_only_aggregate_signed_vectors_and_dense_hashes": True,
            "report_and_attempt_self_hashes_required": True,
        },
        "analysis": {
            "full_frame_response": (
                "arithmetic mean of 310 within-document row means"
            ),
            "matched56_response": (
                "stratum-weighted equal-document HT mean of within-document "
                "row means"
            ),
            "crossfit_response": (
                "arithmetic mean of within-document row means for each exact "
                "254-document complement"
            ),
            "central_response": "(plus-minus)/2",
            "per_estimate_standardization_epsilon": 1e-8,
            "standardized_estimates": [
                "full_frame", *PANEL_NAMES_V14B,
                "complement_train_screen_0", "complement_train_screen_1",
            ],
            "full_frame_is_primary_estimator": True,
        },
        "promotion_gate": {
            "scope": "train_only_estimator_adoption_not_model_update",
            "baseline_source": "completed_v13b_aggregate_evidence",
            "baseline": copy.deepcopy(BASELINE_STABILITY_V14B),
            "rules": {
                "matched56_pairwise_cosine": "median_and_worst_strictly_greater",
                "matched56_pairwise_sign_agreement": "median_and_worst_not_lower",
                "full_to_matched56_optimization_cosine": (
                    "median_and_worst_not_lower"
                ),
                "full_to_matched56_optimization_sign_agreement": (
                    "median_and_worst_not_lower"
                ),
                "crossfit_complement_to_screen_cosine": (
                    "median_and_worst_strictly_greater"
                ),
                "crossfit_complement_to_screen_sign_agreement": (
                    "median_and_worst_not_lower"
                ),
                "all_eight_spreads_nonzero": True,
                "full_frame_coefficient_has_32_finite_nonzero_coordinates": True,
                "all_integrity_rules_required": True,
                "all_rules_conjunctive": True,
            },
            "failure_decision": (
                "retain_v13_sampler_and_keep_all_eval_surfaces_closed"
            ),
            "pass_decision": (
                "authorize_only_a_separately_preregistered_k2_alpha_zero_"
                "confirmation_on_a_fresh_32_direction_basis"
            ),
            "pass_does_not_authorize_model_update_or_evaluation": True,
        },
        "firewall": {
            "allowed": [
                "frozen snapshot794 train source",
                "compact V13 baseline aggregate evidence",
                "compact V14a negative aggregate evidence",
            ],
            "forbidden": [
                "validation", "OOD", "heldout", "benchmark outcomes",
                "adaptive hard replay", "model update", "architecture HPO",
            ],
            "reject_forbidden_cli_tokens_before_parser": True,
            "v14a_status": "closed_failed_gate_no_row_draw1_confirmation",
            "current_sampler": "retain_v13",
        },
        "required_runtime_adapter": {
            "status": "not_yet_implemented",
            "required_files": [
                "train_eggroll_es_specialist_anchor_v14b.py",
                "run_eggroll_es_hierarchical_train_panels_v14b.py",
                "test_eggroll_es_hierarchical_train_panels_v14b.py",
            ],
            "inherit_update_disabled_worker": "eggroll_es_worker_v13.py",
            "real_launch_requires_fresh_committed_implementation_bundle": True,
        },
    }
    value["content_sha256_before_self_field"] = _canonical(value)
    return value


def _metric(value, count):
    if (
        not isinstance(value, dict)
        or set(value) != {"count", "median", "worst"}
        or value["count"] != count
        or any(
            not isinstance(value[key], (int, float))
            or isinstance(value[key], bool)
            or not math.isfinite(float(value[key]))
            for key in ("median", "worst")
        )
        or not -1.0 <= float(value["worst"]) <= float(value["median"]) <= 1.0
    ):
        raise RuntimeError("v14b candidate metric summary changed")
    return value


def candidate_panel_identities_v14b():
    return {
        "full_frame": FULL_FRAME_IDENTITY_V14B[
            "ordered_prompt_identity_sha256"
        ],
        **{
            name: PANEL_IDENTITIES_V14B[name][
                "ordered_prompt_identity_sha256"
            ] for name in PANEL_NAMES_V14B
        },
        **{
            f"complement_{name}": COMPLEMENT_IDENTITIES_V14B[name][
                "ordered_prompt_identity_sha256"
            ] for name in PANEL_NAMES_V14B[3:]
        },
    }


def evaluate_candidate_v14b(candidate):
    if (
        not isinstance(candidate, dict)
        or set(candidate) != {
            "schema", "experiment_name", "alpha", "model_update_applied",
            "validation_ood_or_heldout_used", "perturbation_basis_sha256",
            "panel_identities", "stability", "all_panel_spreads_nonzero",
            "robust_aggregate", "all_integrity_audits_passed",
            "content_sha256_before_self_field",
        }
        or candidate.get("schema")
        != "eggroll-es-paired-distinct-row-summary-v14b"
        or candidate.get("experiment_name") != EXPERIMENT_NAME_V14B
        or candidate.get("alpha") != 0.0
        or candidate.get("model_update_applied") is not False
        or candidate.get("validation_ood_or_heldout_used") is not False
        or candidate.get("perturbation_basis_sha256")
        != PERTURBATION_BASIS_SHA256_V14B
        or candidate.get("panel_identities") != candidate_panel_identities_v14b()
        or candidate.get("all_integrity_audits_passed") is not True
        or candidate.get("content_sha256_before_self_field")
        != _canonical({
            key: value for key, value in candidate.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v14b candidate summary contract changed")
    stability = candidate.get("stability", {})
    if set(stability) != set(BASELINE_STABILITY_V14B):
        raise RuntimeError("v14b candidate stability surface changed")
    for name, metric in stability.items():
        _metric(metric, BASELINE_STABILITY_V14B[name]["count"])
    aggregate = candidate.get("robust_aggregate", {})
    if (
        set(aggregate) != {
            "coefficient_sha256", "l2_norm", "nonzero_coordinate_count",
        }
        or not isinstance(aggregate.get("coefficient_sha256"), str)
        or len(aggregate["coefficient_sha256"]) != 64
        or not isinstance(aggregate.get("l2_norm"), (int, float))
        or isinstance(aggregate["l2_norm"], bool)
        or not math.isfinite(float(aggregate["l2_norm"]))
        or aggregate["l2_norm"] <= 0.0
        or aggregate.get("nonzero_coordinate_count") != 32
    ):
        raise RuntimeError("v14b robust aggregate changed")
    conditions = {}
    for name, baseline in BASELINE_STABILITY_V14B.items():
        value = stability[name]
        strict = name in {
            "matched56_pairwise_cosine",
            "crossfit_complement_to_screen_cosine",
        }
        conditions[name] = {
            "median_passed": (
                value["median"] > baseline["median"] if strict
                else value["median"] >= baseline["median"]
            ),
            "worst_passed": (
                value["worst"] > baseline["worst"] if strict
                else value["worst"] >= baseline["worst"]
            ),
            "comparison": "strictly_greater" if strict else "not_lower",
        }
    conditions["all_panel_spreads_nonzero"] = {
        "passed": candidate["all_panel_spreads_nonzero"] is True,
    }
    conditions["all_integrity_audits_passed"] = {"passed": True}
    passed = all(
        all(
            flag for key, flag in condition.items() if key.endswith("passed")
        )
        for condition in conditions.values()
    )
    gate = {
        "schema": "eggroll-es-paired-distinct-row-promotion-gate-v14b",
        "eligible_for_train_only_estimator_confirmation": passed,
        "eligible_for_model_update": False,
        "eligible_to_open_evaluation": False,
        "conditions": conditions,
        "candidate_content_sha256": candidate[
            "content_sha256_before_self_field"
        ],
        "baseline_evidence_content_sha256": (
            V13_BASELINE_EVIDENCE_CONTENT_SHA256_V14B
        ),
        "failure_decision": (
            None if passed
            else "retain_v13_sampler_and_keep_all_eval_surfaces_closed"
        ),
        "pass_decision": (
            "preregister_fresh_basis_k2_alpha_zero_confirmation"
            if passed else None
        ),
    }
    gate["content_sha256_before_self_field"] = _canonical(gate)
    return gate


def _exclusive_write(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("v14b preregistration already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default=str(PREREGISTRATION_PATH_V14B))
    args = parser.parse_args(argv)
    output = Path(args.output_json).resolve()
    if output != PREREGISTRATION_PATH_V14B:
        raise ValueError("v14b preregistration requires its canonical path")
    preregistration = build_preregistration_v14b()
    _exclusive_write(output, preregistration)
    print(json.dumps({
        "output": str(output),
        "file_sha256": _file_sha256(output),
        "content_sha256": preregistration[
            "content_sha256_before_self_field"
        ],
    }, sort_keys=True))
    return preregistration


if __name__ == "__main__":
    main()
