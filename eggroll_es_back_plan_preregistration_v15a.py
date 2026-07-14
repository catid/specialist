#!/usr/bin/env python3
"""Immutable paired back-versus-middle-late V15A preregistration.

V15A is train-only and alpha-zero.  It changes the selected dense-layer
location while holding capacity, the V13 five-panel estimator, generation,
and a newly frozen perturbation basis common to both architecture arms fixed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

import train_eggroll_es_specialist_anchor_v6 as anchor_v6
import train_eggroll_es_specialist_anchor_v13 as anchor_v13


ROOT = Path(__file__).resolve().parent
PREREGISTRATION_PATH_V15A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_BACK_ONLY_ARCHITECTURE_STABILITY_V15A_PREREGISTRATION.json"
).resolve()
PROTOCOL_PATH_V15A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_BACK_ONLY_ARCHITECTURE_STABILITY_V15A_PROTOCOL.md"
).resolve()

V13_EVIDENCE_PATH_V15A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V13B_TRAIN_PANEL_AGGREGATE_EVIDENCE_V14A.json"
).resolve()
V13_EVIDENCE_FILE_SHA256_V15A = (
    "d367c9c4de1e1f3526ddb3dfba2f5bf24efc77cbccf951f7359eb1969fcd7b54"
)
V13_EVIDENCE_CONTENT_SHA256_V15A = (
    "06f662574013345a6c777af8688a38f3941286d9e11a427ed3342de53451b1e3"
)
V14A_NEGATIVE_PATH_V15A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V14A_FULL_FRAME_NEGATIVE_AGGREGATE_EVIDENCE_V14B.json"
).resolve()
V14A_NEGATIVE_FILE_SHA256_V15A = (
    "9329c09fec5d76e209cff36ac80ffbbec69f53fe7bae9edcc069421d626cc9e9"
)
V14A_NEGATIVE_CONTENT_SHA256_V15A = (
    "ee4ded3d974dfd0becaedb1007f96888e133db51e62130d2844ab9c25e2ccf2b"
)
V14B_NEGATIVE_PATH_V15A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V14B_K2_DISTINCT_ROW_NEGATIVE_AGGREGATE_EVIDENCE_V15.json"
).resolve()
V14B_NEGATIVE_FILE_SHA256_V15A = (
    "735ad52b6395700feb4e8a3dccab165f9b79e620a53918d96e0a26979f58224c"
)
V14B_NEGATIVE_CONTENT_SHA256_V15A = (
    "440504e6c81673ea8de89f336d587a0c57408ea21d3a925ed73b73ecfbeaa7b8"
)

V13_ESTIMATOR_IMPLEMENTATION_V15A = {
    "trainer": {
        "path": str(Path(anchor_v13.__file__).resolve()),
        "file_sha256": (
            "1a8a4145a85c183bb6121914357b7e6bce916b4f76a0693887ac41fa3a8c4c6e"
        ),
    },
    "worker": {
        "path": str((ROOT / "eggroll_es_worker_v13.py").resolve()),
        "file_sha256": (
            "5596bff9174e5e94e812181a51f8cc9f9b2a73f3a4cb58c45d5346147c8d6367"
        ),
    },
    "sampler": {
        "path": str((ROOT / "eggroll_es_train_panel_sampler_v13.py").resolve()),
        "file_sha256": (
            "81ca63c230995d44dfdb739a78b4a1a0f85d09724ba5915860ae36f78ccc3da9"
        ),
    },
}

MODEL_CONFIG_SHA256_V15A = (
    "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
)
LAYER_PLANS_V15A = {
    "middle_late": {
        "path": str((ROOT / "experiments/layer_plans/middle_late_dense_v6.json").resolve()),
        "file_sha256": (
            "d65d702969dcec7a56ca4fcf461d402c44642966191a57c2ef092ec339e3e3df"
        ),
        "plan_sha256": (
            "03745c603a6b48898b41afbd4d9121aef276d7e45ca1a3ae14607ec5d1042cb9"
        ),
        "layers": [20, 21, 22, 23],
        "role": "retained_v13_contemporaneous_control",
    },
    "back": {
        "path": str((ROOT / "experiments/layer_plans/back_dense.json").resolve()),
        "file_sha256": (
            "73bfc82ba057908c0071d3c5e190581fecf6147cc398f06a994231f31908187e"
        ),
        "plan_sha256": (
            "6da92a4db760676acda1bcbcaec4a925a6dd7b641c250a58a3fe4837d97ac93a"
        ),
        "layers": [36, 37, 38, 39],
        "role": "single_candidate",
    },
}
CAPACITY_V15A = {
    "source_unit_count": 35,
    "runtime_selected_parameter_count": 23,
    "selected_element_count": 142_999_552,
    "selected_byte_count": 285_999_104,
}

PERTURBATION_BASIS_SEED_V15A = 20260715
POPULATION_SIZE_V15A = 32
PERTURBATION_SEEDS_V15A = [
    781155634, 166511697, 553997177, 29327742,
    528373473, 598596232, 971738046, 413904000,
    254562666, 669724709, 740139907, 159170652,
    403976347, 671566993, 933764233, 368638977,
    467611975, 969075605, 711629688, 635576630,
    929490240, 1064578919, 442972541, 115403656,
    881242302, 173904944, 30426680, 530098843,
    1006569494, 388812159, 329373152, 772045351,
]
PERTURBATION_SEED_LIST_SHA256_V15A = (
    "0d8bc84fd1190a7a4d7c4d6cf4b81836ae57bd77a2a05088353e82e0cce6538b"
)
PERTURBATION_BASIS_SHA256_V15A = (
    "6c358060c5f9a0a7b00e953bd230b18f915950f0233f38321e0e048a67ea05e7"
)
PREVIOUS_PERTURBATION_BASIS_SHA256_V15A = (
    "29e7ceb1753c39b310a176d827e222b9a5b2c85edf9f2fef5c68b630b8fabc11"
)

EXPERIMENT_NAME_V15A = (
    "snapshot794_layer_v15a_back36_39_vs_middle_late20_23_"
    "paired_v13_panels_alpha_zero_basis20260715"
)
ARM_ORDER_V15A = ("middle_late", "back")
METRIC_COUNTS_V15A = {
    "optimization_pairwise_cosine": 3,
    "optimization_pairwise_sign_agreement": 3,
    "aggregate_to_optimization_cosine": 3,
    "aggregate_to_optimization_sign_agreement": 3,
    "train_screen_cosine": 2,
    "train_screen_sign_agreement": 2,
}
V13_BASELINE_STABILITY_V15A = {
    "optimization_pairwise_cosine": {
        "count": 3, "median": 0.47411088498906484,
        "worst": 0.3900621868364503,
    },
    "optimization_pairwise_sign_agreement": {
        "count": 3, "median": 0.59375, "worst": 0.5625,
    },
    "aggregate_to_optimization_cosine": {
        "count": 3, "median": 0.7608236805612648,
        "worst": 0.7082628389768383,
    },
    "aggregate_to_optimization_sign_agreement": {
        "count": 3, "median": 0.8125, "worst": 0.75,
    },
    "train_screen_cosine": {
        "count": 2, "median": 0.3936314430866483,
        "worst": 0.314941371734614,
    },
    "train_screen_sign_agreement": {
        "count": 2, "median": 0.65625, "worst": 0.53125,
    },
}
COSINE_MINIMUM_IMPROVEMENT_V15A = 0.05


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    return anchor_v13.canonical_sha256(value)


def perturbation_basis_v15a():
    return {
        "schema": "eggroll-es-paired-architecture-perturbation-basis-v15a",
        "basis_seed": PERTURBATION_BASIS_SEED_V15A,
        "population_size": POPULATION_SIZE_V15A,
        "seeds": list(PERTURBATION_SEEDS_V15A),
    }


def validate_perturbation_basis_v15a():
    regenerated = np.random.default_rng(
        seed=PERTURBATION_BASIS_SEED_V15A
    ).integers(
        0, 2**30, size=POPULATION_SIZE_V15A, dtype=np.int64,
    ).tolist()
    basis = perturbation_basis_v15a()
    if (
        regenerated != PERTURBATION_SEEDS_V15A
        or len(set(regenerated)) != POPULATION_SIZE_V15A
        or _canonical(regenerated) != PERTURBATION_SEED_LIST_SHA256_V15A
        or _canonical(basis) != PERTURBATION_BASIS_SHA256_V15A
        or PERTURBATION_BASIS_SHA256_V15A
        == PREVIOUS_PERTURBATION_BASIS_SHA256_V15A
    ):
        raise RuntimeError("v15a fresh perturbation basis changed")
    return basis


def _load_compact(path, file_sha256, content_sha256, schema):
    path = Path(path).resolve()
    if _file_sha256(path) != file_sha256:
        raise RuntimeError("v15a compact evidence file identity changed")
    evidence = json.loads(path.read_text())
    if (
        evidence.get("schema") != schema
        or evidence.get("passed") is not True
        or evidence.get("content_sha256_before_self_field") != content_sha256
        or evidence.get("content_sha256_before_self_field") != _canonical({
            key: value for key, value in evidence.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v15a compact evidence content changed")
    return evidence


def load_v13_evidence_v15a():
    evidence = _load_compact(
        V13_EVIDENCE_PATH_V15A,
        V13_EVIDENCE_FILE_SHA256_V15A,
        V13_EVIDENCE_CONTENT_SHA256_V15A,
        "eggroll-es-v13b-train-panel-aggregate-evidence-v14a",
    )
    if (
        evidence.get("selection_surface") != "frozen_train_panels_only"
        or evidence.get("contains_response_vectors_or_row_content") is not False
        or evidence.get("contains_validation_ood_or_heldout_content") is not False
        or {
            name: evidence.get("stability", {}).get(name)
            for name in V13_BASELINE_STABILITY_V15A
        } != V13_BASELINE_STABILITY_V15A
    ):
        raise RuntimeError("v15a V13 native baseline changed")
    return evidence


def load_v14a_negative_v15a():
    evidence = _load_compact(
        V14A_NEGATIVE_PATH_V15A,
        V14A_NEGATIVE_FILE_SHA256_V15A,
        V14A_NEGATIVE_CONTENT_SHA256_V15A,
        "eggroll-es-v14a-full-frame-negative-aggregate-evidence-v14b",
    )
    if (
        evidence.get("decision") != {
            "sampler": "retain_v13",
            "row_draw_iteration_1_confirmation_authorized": False,
            "evaluation_surface_opened": False,
            "model_update_authorized": False,
            "reason": "v14a_failed_its_preregistered_conjunctive_gate",
        }
        or evidence.get("contains_response_vectors_or_dense_result_hashes")
        is not False
        or evidence.get(
            "contains_validation_ood_heldout_or_benchmark_content"
        ) is not False
    ):
        raise RuntimeError("v15a V14a negative evidence changed")
    return evidence


def load_v14b_negative_v15a():
    evidence = _load_compact(
        V14B_NEGATIVE_PATH_V15A,
        V14B_NEGATIVE_FILE_SHA256_V15A,
        V14B_NEGATIVE_CONTENT_SHA256_V15A,
        "eggroll-es-v14b-k2-negative-aggregate-evidence-v15",
    )
    if (
        evidence.get("decision") != {
            "evaluation_surface_opened_or_authorized": False,
            "fresh_basis_k2_confirmation_authorized": False,
            "model_update_applied_or_authorized": False,
            "reason": "v14b_failed_its_preregistered_conjunctive_gate",
            "sampler": "retain_v13",
        }
        or evidence.get("contains_response_vectors_or_dense_result_hashes")
        is not False
        or evidence.get(
            "contains_validation_ood_heldout_or_benchmark_content"
        ) is not False
    ):
        raise RuntimeError("v15a V14b negative evidence changed")
    return evidence


def validate_v13_estimator_v15a():
    for identity in V13_ESTIMATOR_IMPLEMENTATION_V15A.values():
        if _file_sha256(identity["path"]) != identity["file_sha256"]:
            raise RuntimeError("v15a V13 estimator implementation changed")
    bundle = anchor_v13.load_panel_bundle_v13()
    if (
        bundle.get("content_sha256_before_self_field")
        != anchor_v13.PANEL_BUNDLE_CONTENT_SHA256_V13
        or bundle.get("source", {}).get("rows") != 794
        or len(bundle.get("panels", {})) != 5
        or any(len(panel["questions"]) != 56 for panel in bundle["panels"].values())
    ):
        raise RuntimeError("v15a exact V13 five-panel estimator changed")
    return bundle


def validate_layer_plans_v15a():
    result = {}
    for name in ARM_ORDER_V15A:
        spec = LAYER_PLANS_V15A[name]
        if _file_sha256(spec["path"]) != spec["file_sha256"]:
            raise RuntimeError("v15a layer plan file changed")
        bundle = anchor_v6.load_frozen_layer_plan_v6(
            spec["path"],
            expected_file_sha256=spec["file_sha256"],
            expected_plan_sha256=spec["plan_sha256"],
            expected_model_config_sha256=MODEL_CONFIG_SHA256_V15A,
        )
        capacity = anchor_v6.FROZEN_RUNTIME_EXPECTATIONS_V6[
            spec["plan_sha256"]
        ]
        if capacity != CAPACITY_V15A:
            raise RuntimeError("v15a paired architecture capacity changed")
        result[name] = bundle
    return result


def _panel_identities(bundle):
    return {
        name: panel["ordered_row_identity_sha256"]
        for name, panel in bundle["panels"].items()
    }


def build_preregistration_v15a():
    v13 = load_v13_evidence_v15a()
    v14a = load_v14a_negative_v15a()
    v14b = load_v14b_negative_v15a()
    panel_bundle = validate_v13_estimator_v15a()
    validate_layer_plans_v15a()
    basis = validate_perturbation_basis_v15a()
    value = {
        "schema": "eggroll-es-back-plan-stability-preregistration-v15a",
        "status": "preregistered_not_launch_authorized",
        "experiment_name": EXPERIMENT_NAME_V15A,
        "hypothesis_count": 1,
        "hypothesis": (
            "at matched selected capacity, moving the dense ES partition from "
            "middle-late layers 20-23 to back layers 36-39 improves the exact "
            "V13 train-panel direction-stability endpoints"
        ),
        "selection_surface": "exact_frozen_v13_train_panels_only",
        "contains_validation_ood_heldout_or_benchmark_content": False,
        "evidence": {
            "retained_v13_baseline": {
                "path": str(V13_EVIDENCE_PATH_V15A),
                "file_sha256": V13_EVIDENCE_FILE_SHA256_V15A,
                "content_sha256": V13_EVIDENCE_CONTENT_SHA256_V15A,
                "decision": "retain_as_absolute_stability_floor",
                "report_content_sha256": v13["v13b_report"]["content_sha256"],
            },
            "v14a_negative": {
                "path": str(V14A_NEGATIVE_PATH_V15A),
                "file_sha256": V14A_NEGATIVE_FILE_SHA256_V15A,
                "content_sha256": V14A_NEGATIVE_CONTENT_SHA256_V15A,
                "decision": copy.deepcopy(v14a["decision"]),
            },
            "v14b_negative": {
                "path": str(V14B_NEGATIVE_PATH_V15A),
                "file_sha256": V14B_NEGATIVE_FILE_SHA256_V15A,
                "content_sha256": V14B_NEGATIVE_CONTENT_SHA256_V15A,
                "decision": copy.deepcopy(v14b["decision"]),
            },
        },
        "paired_architecture": {
            "arm_order": list(ARM_ORDER_V15A),
            "same_fresh_basis_both_arms": True,
            "same_panels_generation_and_objective_both_arms": True,
            "only_intended_difference": "selected_dense_layer_location",
            "arms": {
                name: {
                    **copy.deepcopy(LAYER_PLANS_V15A[name]),
                    "capacity": copy.deepcopy(CAPACITY_V15A),
                    "model_config_sha256": MODEL_CONFIG_SHA256_V15A,
                }
                for name in ARM_ORDER_V15A
            },
        },
        "v13_estimator": {
            "implementation": copy.deepcopy(V13_ESTIMATOR_IMPLEMENTATION_V15A),
            "source": copy.deepcopy(panel_bundle["source"]),
            "manifest": copy.deepcopy(panel_bundle["manifest"]),
            "panel_bundle_content_sha256": panel_bundle[
                "content_sha256_before_self_field"
            ],
            "panel_names": list(anchor_v13.PANEL_NAMES_V13),
            "optimization_panels": list(anchor_v13.OPTIMIZATION_PANELS_V13),
            "train_screens": list(anchor_v13.TRAIN_SCREENS_V13),
            "panel_size": 56,
            "generation_prompts_per_direction_and_sign_per_arm": 280,
            "ordered_panel_identities": _panel_identities(panel_bundle),
            "estimand": "equal_weight_document_semantic_conflict_unit_mean",
            "optimization_response": (
                "stratum-weighted Horvitz-Thompson mean"
            ),
            "central_response": "(plus-minus)/2",
            "per_panel_standardization_epsilon": 1e-8,
            "robust_aggregate": (
                "coordinatewise median of the three independently standardized "
                "optimization-panel coefficient vectors"
            ),
            "native_endpoints_only": list(METRIC_COUNTS_V15A),
            "no_full_frame_or_disjoint_crossfit_claim": True,
        },
        "runtime": {
            "model": str((ROOT / "models/Qwen3.6-35B-A3B").resolve()),
            "alpha": 0.0,
            "model_update_allowed": False,
            "perturbation_basis": basis,
            "perturbation_basis_sha256": PERTURBATION_BASIS_SHA256_V15A,
            "previous_basis_sha256": PREVIOUS_PERTURBATION_BASIS_SHA256_V15A,
            "same_seed_algorithm_as_v8_v13": (
                "numpy.default_rng(basis_seed).integers(0,2**30,size=32,dtype=int64)"
            ),
            "sign_order": ["plus", "minus"],
            "engine_count": 4,
            "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "all_four_engines_required_every_signed_wave": True,
            "complete_four_direction_waves_required": True,
            "population_waves_per_arm": 8,
            "signed_waves_per_arm": 16,
            "arm_execution_order": list(ARM_ORDER_V15A),
            "greedy_generation_seed": 43,
            "same_order_every_direction_sign_and_arm": True,
            "objective_and_generation_identical_to_v13": True,
        },
        "integrity": {
            "exact_reference_restoration_after_every_sign": True,
            "restore_in_finally_after_every_sign_required": True,
            "pre_and_post_population_base_probe_must_match_exactly": True,
            "population_boundary_audit_required_per_arm": True,
            "all_four_engines_must_participate_in_every_signed_wave": True,
            "partial_wave_allowed": False,
            "fresh_o_excl_attempt_and_run_required": True,
            "committed_source_bundle_required": True,
            "dense_result_hash_per_direction_sign_required": True,
            "persist_response_vectors_or_row_content": False,
            "report_and_attempt_self_hashes_required": True,
        },
        "promotion_gate": {
            "scope": "train_only_back_plan_stability_confirmation_only",
            "historical_v13_baseline": copy.deepcopy(
                V13_BASELINE_STABILITY_V15A
            ),
            "metric_counts": copy.deepcopy(METRIC_COUNTS_V15A),
            "absolute_v13_rules": {
                "cosine_median_and_worst": (
                    "at_least_v13_plus_0.05_for_every_cosine_family"
                ),
                "sign_median_and_worst": (
                    "not_lower_than_v13_for_every_sign_family"
                ),
            },
            "contemporaneous_control_rules": {
                "cosine_median_and_worst": (
                    "back_at_least_middle_late_plus_0.05_for_every_cosine_family"
                ),
                "sign_median_and_worst": (
                    "back_not_lower_than_middle_late_for_every_sign_family"
                ),
            },
            "cosine_minimum_improvement": COSINE_MINIMUM_IMPROVEMENT_V15A,
            "multiplicity_control": (
                "one hypothesis; all six cosine median/worst margins and all "
                "six sign nonregression checks are conjunctive against both "
                "the historical floor and contemporaneous control"
            ),
            "all_twelve_endpoint_summaries_must_not_regress": True,
            "all_panel_spreads_nonzero_both_arms": True,
            "all_integrity_audits_required": True,
            "all_rules_conjunctive": True,
            "failure_decision": (
                "retain_v13_middle_late_and_keep_all_eval_surfaces_closed"
            ),
            "pass_decision": (
                "authorize_only_a_separately_preregistered_back_plan_alpha_zero_"
                "confirmation_on_another_fresh_32_direction_basis"
            ),
            "pass_does_not_authorize_model_update_or_evaluation": True,
        },
        "firewall": {
            "allowed": [
                "exact frozen snapshot794 V13 train source and panels",
                "compact V13 aggregate baseline evidence",
                "compact V14a and V14b negative aggregate evidence",
            ],
            "forbidden": [
                "validation", "OOD", "heldout", "benchmark outcomes",
                "model update", "layer insertion", "adaptive architecture fallback",
                "front plus back result selection",
            ],
            "current_retained_recipe": "V13 middle-late layers 20-23",
            "v14a_status": "closed_failed_gate_no_confirmation",
            "v14b_status": "closed_failed_gate_no_confirmation",
        },
        "separate_future_hypothesis": {
            "front_plus_back_is_part_of_v15a": False,
            "front_plus_back_authorized_on_v15a_failure": False,
            "front_plus_back_authorized_on_v15a_pass": False,
            "requirement": (
                "a separate preregistration, basis, implementation, and gate "
                "before any front-plus-back experiment"
            ),
        },
        "required_runtime_adapter": {
            "status": "not_yet_implemented",
            "required_files": [
                "train_eggroll_es_specialist_anchor_v15a.py",
                "run_eggroll_es_back_plan_stability_v15a.py",
                "test_eggroll_es_back_plan_stability_v15a.py",
            ],
            "real_launch_requires_fresh_committed_implementation_bundle": True,
            "runtime_not_authorized_by_this_preregistration_commit": True,
        },
    }
    value["content_sha256_before_self_field"] = _canonical(value)
    return value


def _metric(value, count):
    if (
        not isinstance(value, dict)
        or set(value) != {"count", "median", "worst"}
        or value.get("count") != count
        or any(
            not isinstance(value.get(key), (int, float))
            or isinstance(value.get(key), bool)
            or not math.isfinite(float(value[key]))
            for key in ("median", "worst")
        )
        or not -1.0 <= float(value["worst"]) <= float(value["median"]) <= 1.0
    ):
        raise RuntimeError("v15a candidate metric summary changed")
    return value


def _validate_arm_summary(name, arm):
    spec = LAYER_PLANS_V15A[name]
    if (
        not isinstance(arm, dict)
        or set(arm) != {"plan_sha256", "stability", "robust_aggregate"}
        or arm.get("plan_sha256") != spec["plan_sha256"]
        or set(arm.get("stability", {})) != set(METRIC_COUNTS_V15A)
    ):
        raise RuntimeError("v15a candidate arm contract changed")
    for metric, count in METRIC_COUNTS_V15A.items():
        _metric(arm["stability"][metric], count)
    aggregate = arm.get("robust_aggregate", {})
    if (
        set(aggregate) != {
            "coefficient_sha256", "l2_norm", "nonzero_coordinate_count",
        }
        or not isinstance(aggregate.get("coefficient_sha256"), str)
        or len(aggregate["coefficient_sha256"]) != 64
        or not isinstance(aggregate.get("l2_norm"), (int, float))
        or isinstance(aggregate["l2_norm"], bool)
        or not math.isfinite(float(aggregate["l2_norm"]))
        or float(aggregate["l2_norm"]) <= 0.0
        or aggregate.get("nonzero_coordinate_count") != 32
    ):
        raise RuntimeError("v15a candidate robust aggregate changed")


def evaluate_candidate_v15a(candidate):
    panel_bundle = validate_v13_estimator_v15a()
    expected_panel_identities = _panel_identities(panel_bundle)
    if (
        not isinstance(candidate, dict)
        or set(candidate) != {
            "schema", "experiment_name", "alpha", "model_update_applied",
            "validation_ood_heldout_or_benchmark_used",
            "perturbation_basis_sha256", "panel_bundle_content_sha256",
            "panel_identities", "arm_order", "arms",
            "all_panel_spreads_nonzero", "all_integrity_audits_passed",
            "content_sha256_before_self_field",
        }
        or candidate.get("schema")
        != "eggroll-es-back-plan-stability-summary-v15a"
        or candidate.get("experiment_name") != EXPERIMENT_NAME_V15A
        or candidate.get("alpha") != 0.0
        or candidate.get("model_update_applied") is not False
        or candidate.get("validation_ood_heldout_or_benchmark_used") is not False
        or candidate.get("perturbation_basis_sha256")
        != PERTURBATION_BASIS_SHA256_V15A
        or candidate.get("panel_bundle_content_sha256")
        != anchor_v13.PANEL_BUNDLE_CONTENT_SHA256_V13
        or candidate.get("panel_identities") != expected_panel_identities
        or candidate.get("arm_order") != list(ARM_ORDER_V15A)
        or tuple(candidate.get("arms", {})) != ARM_ORDER_V15A
        or candidate.get("all_panel_spreads_nonzero") != {
            "middle_late": True, "back": True,
        }
        or candidate.get("all_integrity_audits_passed") is not True
        or candidate.get("content_sha256_before_self_field") != _canonical({
            key: value for key, value in candidate.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v15a candidate summary contract changed")
    for name in ARM_ORDER_V15A:
        _validate_arm_summary(name, candidate["arms"][name])

    back = candidate["arms"]["back"]["stability"]
    control = candidate["arms"]["middle_late"]["stability"]
    conditions = {"absolute_v13": {}, "paired_middle_late_control": {}}
    for metric, baseline in V13_BASELINE_STABILITY_V15A.items():
        is_cosine = metric.endswith("_cosine")
        margin = COSINE_MINIMUM_IMPROVEMENT_V15A if is_cosine else 0.0
        conditions["absolute_v13"][metric] = {
            "required_margin": margin,
            "median_passed": back[metric]["median"] >= baseline["median"] + margin,
            "worst_passed": back[metric]["worst"] >= baseline["worst"] + margin,
            "comparison": (
                "minimum_strict_improvement" if is_cosine else "not_lower"
            ),
        }
        conditions["paired_middle_late_control"][metric] = {
            "required_margin": margin,
            "median_passed": back[metric]["median"] >= control[metric]["median"] + margin,
            "worst_passed": back[metric]["worst"] >= control[metric]["worst"] + margin,
            "comparison": (
                "minimum_strict_improvement" if is_cosine else "not_lower"
            ),
        }
    conditions["integrity"] = {
        "all_panel_spreads_nonzero_both_arms_passed": True,
        "all_integrity_audits_passed": True,
    }
    passed = (
        all(
            flag
            for family_name in ("absolute_v13", "paired_middle_late_control")
            for condition in conditions[family_name].values()
            for label, flag in condition.items()
            if label.endswith("passed")
        )
        and all(conditions["integrity"].values())
    )
    gate = {
        "schema": "eggroll-es-back-plan-stability-gate-v15a",
        "eligible_for_fresh_basis_back_plan_confirmation": passed,
        "eligible_for_model_update": False,
        "eligible_to_open_evaluation": False,
        "conditions": conditions,
        "candidate_content_sha256": candidate[
            "content_sha256_before_self_field"
        ],
        "v13_baseline_evidence_content_sha256": (
            V13_EVIDENCE_CONTENT_SHA256_V15A
        ),
        "failure_decision": (
            None if passed
            else "retain_v13_middle_late_and_keep_all_eval_surfaces_closed"
        ),
        "pass_decision": (
            "preregister_back_plan_alpha_zero_confirmation_on_another_fresh_basis"
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
        raise ValueError("v15a preregistration already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default=str(PREREGISTRATION_PATH_V15A))
    args = parser.parse_args(argv)
    output = Path(args.output_json).resolve()
    if output != PREREGISTRATION_PATH_V15A:
        raise ValueError("v15a preregistration requires its canonical path")
    preregistration = build_preregistration_v15a()
    _exclusive_write(output, preregistration)
    print(json.dumps({
        "output": str(output),
        "file_sha256": _file_sha256(output),
        "content_sha256": preregistration["content_sha256_before_self_field"],
    }, sort_keys=True))
    return preregistration


if __name__ == "__main__":
    main()
