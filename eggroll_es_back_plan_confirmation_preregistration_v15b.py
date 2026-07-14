#!/usr/bin/env python3
"""Train-only V15B fresh-basis confirmation preregistration.

V15B repeats the V15A paired architecture diagnostic on a new deterministic
32-direction basis.  Middle-late is a nuisance contemporaneous control; only
the back arm can be confirmed.  The experiment is alpha-zero and cannot open
an evaluation surface or apply a model update.
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


ROOT = Path(__file__).resolve().parent
PREREGISTRATION_PATH_V15B = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_BACK_PLAN_FRESH_BASIS_CONFIRMATION_V15B_PREREGISTRATION.json"
).resolve()
PROTOCOL_PATH_V15B = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_BACK_PLAN_FRESH_BASIS_CONFIRMATION_V15B_PROTOCOL.md"
).resolve()
V15A_POSITIVE_PATH_V15B = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V15A_BACK_PLAN_POSITIVE_AGGREGATE_EVIDENCE_V15B.json"
).resolve()
V15A_POSITIVE_FILE_SHA256_V15B = (
    "1e14abee9e1514915bc241c8f6caacbe1bb7103e1c69a9afdde1f9ce13661ae1"
)
V15A_POSITIVE_CONTENT_SHA256_V15B = (
    "c9ab854c8417c4f6c74e5fe54de29e2f6a8b222b4c7fc454b5f22b183c3b08b2"
)
V15A_PREREGISTRATION_PATH_V15B = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_BACK_ONLY_ARCHITECTURE_STABILITY_V15A_PREREGISTRATION.json"
).resolve()
V15A_PREREGISTRATION_FILE_SHA256_V15B = (
    "ad86f388ff4effbc195a3fd60d6d32c430a83026a331a18d625d477d390f3b88"
)
V15A_PREREGISTRATION_CONTENT_SHA256_V15B = (
    "dda0f49e470cf5bb550f80d27a2389d069c8d064975c18b581261054462bb7c7"
)
V13_EVIDENCE_PATH_V15B = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V13B_TRAIN_PANEL_AGGREGATE_EVIDENCE_V14A.json"
).resolve()
V13_EVIDENCE_FILE_SHA256_V15B = (
    "d367c9c4de1e1f3526ddb3dfba2f5bf24efc77cbccf951f7359eb1969fcd7b54"
)
V13_EVIDENCE_CONTENT_SHA256_V15B = (
    "06f662574013345a6c777af8688a38f3941286d9e11a427ed3342de53451b1e3"
)
V15A_PROTOCOL_PATH_V15B = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_BACK_ONLY_ARCHITECTURE_STABILITY_V15A_PROTOCOL.md"
).resolve()
V15A_PROTOCOL_FILE_SHA256_V15B = (
    "10902864bbf9f4d560d73127ab08402e6b41f3129413e96eb38392bba5894f0f"
)

EXPERIMENT_NAME_V15B = (
    "snapshot794_layer_v15b_back36_39_vs_middle_late20_23_"
    "paired_confirmation_v13_panels_alpha_zero_basis20260716"
)
ARM_ORDER_V15B = ("middle_late", "back")
POPULATION_SIZE_V15B = 32
PERTURBATION_BASIS_SEED_V15B = 20260716
PERTURBATION_SEEDS_V15B = [
    589835896, 74723119, 95809569, 881146472,
    600975249, 443961320, 572096434, 747617196,
    646036023, 481017673, 143654908, 861807725,
    101874884, 957267096, 277878054, 888258292,
    900315680, 843855, 54803855, 303665889,
    882956525, 531498790, 351640649, 794527719,
    939235373, 135972612, 345623308, 910628968,
    740478064, 423613537, 449466860, 7545937,
]
PERTURBATION_SEED_LIST_SHA256_V15B = (
    "2d5321430357820773e313a01663918d943af19b01902344a916527a323dc388"
)
PERTURBATION_BASIS_SHA256_V15B = (
    "97e9c5687677bd02365f77671141031ba2739018ed07ccd1bbb3eaabbc0a94f8"
)
V15A_PERTURBATION_BASIS_SHA256_V15B = (
    "6c358060c5f9a0a7b00e953bd230b18f915950f0233f38321e0e048a67ea05e7"
)
COSINE_MINIMUM_IMPROVEMENT_V15B = 0.05
COSINE_V15A_REPLICATION_TOLERANCE_V15B = 0.05
SIGN_V15A_REPLICATION_TOLERANCE_V15B = 1.0 / POPULATION_SIZE_V15B
METRIC_COUNTS_V15B = {
    "optimization_pairwise_cosine": 3,
    "optimization_pairwise_sign_agreement": 3,
    "aggregate_to_optimization_cosine": 3,
    "aggregate_to_optimization_sign_agreement": 3,
    "train_screen_cosine": 2,
    "train_screen_sign_agreement": 2,
}


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _load_sealed(path, file_sha256, content_sha256, schema):
    path = Path(path).resolve()
    if _file_sha256(path) != file_sha256:
        raise RuntimeError("v15b sealed evidence file identity changed")
    value = json.loads(path.read_text())
    if (
        value.get("schema") != schema
        or value.get("content_sha256_before_self_field") != content_sha256
        or canonical_sha256(_without_self(value)) != content_sha256
    ):
        raise RuntimeError("v15b sealed evidence content changed")
    return value


def load_v15a_positive_v15b():
    value = _load_sealed(
        V15A_POSITIVE_PATH_V15B,
        V15A_POSITIVE_FILE_SHA256_V15B,
        V15A_POSITIVE_CONTENT_SHA256_V15B,
        "eggroll-es-v15a-back-plan-positive-aggregate-evidence-v15b",
    )
    if value.get("decision") != {
        "back_alpha_zero_confirmation_on_another_fresh_basis_authorized": True,
        "evaluation_surface_opened_or_authorized": False,
        "front_plus_back_authorized": False,
        "model_update_applied_or_authorized": False,
        "retained_recipe_pending_confirmation": "V13 middle-late layers 20-23",
    }:
        raise RuntimeError("v15b authorization changed")
    if (
        value.get("contains_response_vectors_or_row_content") is not False
        or value.get(
            "contains_validation_ood_heldout_or_benchmark_content"
        ) is not False
        or value.get("aggregate_gate", {}).get(
            "eligible_for_fresh_basis_back_plan_confirmation"
        ) is not True
    ):
        raise RuntimeError("v15b authorization firewall changed")
    return value


def load_v15a_preregistration_v15b():
    value = _load_sealed(
        V15A_PREREGISTRATION_PATH_V15B,
        V15A_PREREGISTRATION_FILE_SHA256_V15B,
        V15A_PREREGISTRATION_CONTENT_SHA256_V15B,
        "eggroll-es-back-plan-stability-preregistration-v15a",
    )
    if (
        value.get("contains_validation_ood_heldout_or_benchmark_content")
        is not False
        or value.get("runtime", {}).get("model_update_allowed") is not False
        or value.get("paired_architecture", {}).get("arm_order")
        != list(ARM_ORDER_V15B)
    ):
        raise RuntimeError("v15b V15A preregistration contract changed")
    return value


def load_v13_evidence_v15b():
    value = _load_sealed(
        V13_EVIDENCE_PATH_V15B,
        V13_EVIDENCE_FILE_SHA256_V15B,
        V13_EVIDENCE_CONTENT_SHA256_V15B,
        "eggroll-es-v13b-train-panel-aggregate-evidence-v14a",
    )
    if (
        value.get("passed") is not True
        or value.get("selection_surface") != "frozen_train_panels_only"
        or value.get("contains_response_vectors_or_row_content") is not False
        or value.get("contains_validation_ood_or_heldout_content") is not False
    ):
        raise RuntimeError("v15b V13 baseline contract changed")
    return value


def perturbation_basis_v15b():
    return {
        "schema": "eggroll-es-paired-architecture-perturbation-basis-v15b",
        "basis_seed": PERTURBATION_BASIS_SEED_V15B,
        "population_size": POPULATION_SIZE_V15B,
        "seeds": list(PERTURBATION_SEEDS_V15B),
    }


def validate_perturbation_basis_v15b():
    regenerated = np.random.default_rng(
        seed=PERTURBATION_BASIS_SEED_V15B
    ).integers(
        0, 2**30, size=POPULATION_SIZE_V15B, dtype=np.int64,
    ).tolist()
    basis = perturbation_basis_v15b()
    if (
        regenerated != PERTURBATION_SEEDS_V15B
        or len(set(regenerated)) != POPULATION_SIZE_V15B
        or canonical_sha256(regenerated)
        != PERTURBATION_SEED_LIST_SHA256_V15B
        or canonical_sha256(basis) != PERTURBATION_BASIS_SHA256_V15B
        or PERTURBATION_BASIS_SHA256_V15B
        == V15A_PERTURBATION_BASIS_SHA256_V15B
    ):
        raise RuntimeError("v15b fresh perturbation basis changed")
    return basis


def _native_stability(value):
    return {
        metric: copy.deepcopy(value[metric])
        for metric in METRIC_COUNTS_V15B
    }


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
        raise RuntimeError("v15b candidate metric summary changed")


def _validate_arm(name, arm, plans):
    if (
        not isinstance(arm, dict)
        or set(arm) != {"plan_sha256", "stability", "robust_aggregate"}
        or arm.get("plan_sha256") != plans[name]["plan_sha256"]
        or set(arm.get("stability", {})) != set(METRIC_COUNTS_V15B)
    ):
        raise RuntimeError("v15b candidate arm contract changed")
    for metric, count in METRIC_COUNTS_V15B.items():
        _metric(arm["stability"][metric], count)
    aggregate = arm.get("robust_aggregate", {})
    if (
        set(aggregate)
        != {"coefficient_sha256", "l2_norm", "nonzero_coordinate_count"}
        or not isinstance(aggregate.get("coefficient_sha256"), str)
        or len(aggregate["coefficient_sha256"]) != 64
        or not isinstance(aggregate.get("l2_norm"), (int, float))
        or isinstance(aggregate["l2_norm"], bool)
        or not math.isfinite(float(aggregate["l2_norm"]))
        or float(aggregate["l2_norm"]) <= 0.0
        or aggregate.get("nonzero_coordinate_count") != POPULATION_SIZE_V15B
    ):
        raise RuntimeError("v15b candidate robust aggregate changed")


def build_preregistration_v15b():
    positive = load_v15a_positive_v15b()
    v15a_prereg = load_v15a_preregistration_v15b()
    v13 = load_v13_evidence_v15b()
    if _file_sha256(V15A_PROTOCOL_PATH_V15B) != V15A_PROTOCOL_FILE_SHA256_V15B:
        raise RuntimeError("v15b V15A protocol identity changed")
    basis = validate_perturbation_basis_v15b()
    plans = copy.deepcopy(v15a_prereg["paired_architecture"]["arms"])
    v13_baseline = _native_stability(v13["stability"])
    v15a_back = copy.deepcopy(positive["stability"]["back"])
    panel_identities = copy.deepcopy(
        v15a_prereg["v13_estimator"]["ordered_panel_identities"]
    )
    value = {
        "schema": "eggroll-es-back-plan-confirmation-preregistration-v15b",
        "status": "preregistered_runtime_not_yet_authorized",
        "experiment_name": EXPERIMENT_NAME_V15B,
        "hypothesis_count": 1,
        "hypothesis": (
            "back layers 36-39 reproduce their V15A train-panel stability "
            "advantage on another fresh basis while retaining the V13 "
            "absolute floors and fresh-basis middle-late control advantage"
        ),
        "authorization": {
            "source": {
                "path": str(V15A_POSITIVE_PATH_V15B),
                "file_sha256": V15A_POSITIVE_FILE_SHA256_V15B,
                "content_sha256": V15A_POSITIVE_CONTENT_SHA256_V15B,
            },
            "authorized_action": (
                "one separately preregistered alpha-zero back-plan "
                "confirmation on another fresh 32-direction basis"
            ),
            "paired_control_interpretation": (
                "middle_late is a non-promotable nuisance control sharing "
                "the new basis; only the back arm can be confirmed"
            ),
            "paired_rerun_within_authorization": True,
            "second_architecture_hypothesis_added": False,
        },
        "selection_surface": "exact_frozen_v13_train_panels_only",
        "contains_validation_ood_heldout_or_benchmark_content": False,
        "paired_architecture": {
            "arm_order": list(ARM_ORDER_V15B),
            "arms": plans,
            "candidate_arm": "back",
            "control_arm": "middle_late",
            "control_can_be_promoted": False,
            "same_fresh_basis_both_arms": True,
            "same_panels_generation_and_objective_both_arms": True,
            "only_intended_difference": "selected_dense_layer_location",
        },
        "estimator": {
            "panel_bundle_content_sha256": v15a_prereg[
                "v13_estimator"
            ]["panel_bundle_content_sha256"],
            "ordered_panel_identities": panel_identities,
            "panel_names": list(panel_identities),
            "panel_size": v15a_prereg["v13_estimator"]["panel_size"],
            "native_endpoints_only": list(METRIC_COUNTS_V15B),
            "v13_estimator_unchanged": True,
            "no_row_prompt_response_or_answer_content_persisted": True,
        },
        "runtime": {
            "alpha": 0.0,
            "model_update_allowed": False,
            "model": v15a_prereg["runtime"]["model"],
            "greedy_generation_seed": v15a_prereg["runtime"][
                "greedy_generation_seed"
            ],
            "objective_and_generation_identical_to_v13_v15a": True,
            "arm_execution_order": list(ARM_ORDER_V15B),
            "perturbation_basis": basis,
            "perturbation_basis_sha256": PERTURBATION_BASIS_SHA256_V15B,
            "previous_basis_sha256": V15A_PERTURBATION_BASIS_SHA256_V15B,
            "engine_count": 4,
            "gpu_ids": [0, 1, 2, 3],
            "tp_per_engine": 1,
            "population_waves_per_arm": 8,
            "signed_waves_per_arm": 16,
            "complete_four_direction_waves_required": True,
            "all_four_engines_required_every_signed_wave": True,
            "sign_order": ["plus", "minus"],
        },
        "promotion_gate": {
            "all_rules_conjunctive": True,
            "metric_counts": copy.deepcopy(METRIC_COUNTS_V15B),
            "historical_v13_baseline": v13_baseline,
            "contemporaneous_control_rules": {
                "cosine_median_and_worst": (
                    "back_at_least_middle_late_plus_0.05_every_family"
                ),
                "sign_median_and_worst": (
                    "back_not_lower_than_middle_late_every_family"
                ),
            },
            "absolute_v13_rules": {
                "cosine_median_and_worst": (
                    "back_at_least_v13_plus_0.05_every_family"
                ),
                "sign_median_and_worst": (
                    "back_not_lower_than_v13_every_family"
                ),
            },
            "v15a_back_reference": v15a_back,
            "predeclared_replication_stability_rules": {
                "cosine_median_and_worst": (
                    "back_not_lower_than_v15a_back_minus_0.05_every_family"
                ),
                "cosine_tolerance": COSINE_V15A_REPLICATION_TOLERANCE_V15B,
                "cosine_tolerance_justification": (
                    "reuse_the_preregistered_material_effect_unit"
                ),
                "sign_median_and_worst": (
                    "back_not_lower_than_v15a_back_minus_one_over_32_"
                    "every_family"
                ),
                "sign_tolerance": SIGN_V15A_REPLICATION_TOLERANCE_V15B,
                "sign_tolerance_justification": (
                    "one_direction_coordinate_of_the_frozen_population"
                ),
                "post_hoc_tolerance_selection_allowed": False,
            },
            "all_panel_spreads_nonzero_both_arms": True,
            "all_integrity_audits_required": True,
            "pass_decision": (
                "authorize_only_separate_preregistration_of_a_back_plan_"
                "nonzero_alpha_train_update_experiment"
            ),
            "failure_decision": (
                "retain_v13_middle_late_and_keep_all_eval_surfaces_closed"
            ),
            "pass_does_not_authorize_model_update_or_evaluation": True,
        },
        "integrity": {
            "fresh_o_excl_attempt_and_run_required": True,
            "committed_source_bundle_required": True,
            "report_and_attempt_self_hashes_required": True,
            "exact_reference_restoration_after_every_sign": True,
            "restore_in_finally_after_every_sign_required": True,
            "pre_and_post_population_base_probe_must_match_exactly": True,
            "population_boundary_audit_required_per_arm": True,
            "dense_result_hash_per_direction_sign_required": True,
            "persist_response_vectors_or_row_content": False,
            "partial_wave_allowed": False,
            "all_four_engines_must_participate_in_every_signed_wave": True,
            "persisted_candidate_replay_uses_explicit_arm_order": True,
        },
        "firewall": {
            "forbidden": [
                "validation", "OOD", "heldout", "benchmark outcomes",
                "model update", "layer insertion", "front plus back",
                "adaptive tolerance", "adaptive endpoint selection",
            ],
            "current_retained_recipe": "V13 middle-late layers 20-23",
            "all_surfaces_remain_closed_before_and_after_v15b": True,
        },
        "required_runtime_adapter": {
            "status": "not_yet_implemented",
            "runtime_not_authorized_by_this_preregistration_commit": True,
            "required_files": [
                "train_eggroll_es_specialist_anchor_v15b.py",
                "run_eggroll_es_back_plan_confirmation_v15b.py",
                "test_eggroll_es_back_plan_confirmation_runtime_v15b.py",
            ],
            "real_launch_requires_fresh_committed_implementation_bundle": True,
        },
        "evidence_bindings": {
            "v13": {
                "path": str(V13_EVIDENCE_PATH_V15B),
                "file_sha256": V13_EVIDENCE_FILE_SHA256_V15B,
                "content_sha256": V13_EVIDENCE_CONTENT_SHA256_V15B,
            },
            "v15a_preregistration": {
                "path": str(V15A_PREREGISTRATION_PATH_V15B),
                "file_sha256": V15A_PREREGISTRATION_FILE_SHA256_V15B,
                "content_sha256": V15A_PREREGISTRATION_CONTENT_SHA256_V15B,
            },
            "v15a_protocol": {
                "path": str(V15A_PROTOCOL_PATH_V15B),
                "file_sha256": V15A_PROTOCOL_FILE_SHA256_V15B,
            },
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def evaluate_candidate_v15b(candidate):
    preregistration = build_preregistration_v15b()
    expected_panels = preregistration["estimator"]["ordered_panel_identities"]
    plans = preregistration["paired_architecture"]["arms"]
    expected_keys = {
        "schema", "experiment_name", "alpha", "model_update_applied",
        "validation_ood_heldout_or_benchmark_used",
        "perturbation_basis_sha256", "panel_bundle_content_sha256",
        "panel_identities", "arm_order", "arms",
        "all_panel_spreads_nonzero", "all_integrity_audits_passed",
        "content_sha256_before_self_field",
    }
    if (
        not isinstance(candidate, dict)
        or set(candidate) != expected_keys
        or candidate.get("schema")
        != "eggroll-es-back-plan-confirmation-summary-v15b"
        or candidate.get("experiment_name") != EXPERIMENT_NAME_V15B
        or candidate.get("alpha") != 0.0
        or candidate.get("model_update_applied") is not False
        or candidate.get("validation_ood_heldout_or_benchmark_used") is not False
        or candidate.get("perturbation_basis_sha256")
        != PERTURBATION_BASIS_SHA256_V15B
        or candidate.get("panel_bundle_content_sha256")
        != preregistration["estimator"]["panel_bundle_content_sha256"]
        or candidate.get("panel_identities") != expected_panels
        or candidate.get("arm_order") != list(ARM_ORDER_V15B)
        or set(candidate.get("arms", {})) != set(ARM_ORDER_V15B)
        or candidate.get("all_panel_spreads_nonzero")
        != {"middle_late": True, "back": True}
        or candidate.get("all_integrity_audits_passed") is not True
        or candidate.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(candidate))
    ):
        raise RuntimeError("v15b candidate summary contract changed")
    for name in ARM_ORDER_V15B:
        _validate_arm(name, candidate["arms"][name], plans)

    back = candidate["arms"]["back"]["stability"]
    control = candidate["arms"]["middle_late"]["stability"]
    baseline = preregistration["promotion_gate"]["historical_v13_baseline"]
    v15a = preregistration["promotion_gate"]["v15a_back_reference"]
    conditions = {
        "absolute_v13": {},
        "paired_middle_late_control": {},
        "v15a_replication_stability": {},
    }
    for metric in METRIC_COUNTS_V15B:
        is_cosine = metric.endswith("_cosine")
        margin = COSINE_MINIMUM_IMPROVEMENT_V15B if is_cosine else 0.0
        tolerance = (
            COSINE_V15A_REPLICATION_TOLERANCE_V15B
            if is_cosine else SIGN_V15A_REPLICATION_TOLERANCE_V15B
        )
        conditions["absolute_v13"][metric] = {
            "required_margin": margin,
            "median_passed": (
                back[metric]["median"] >= baseline[metric]["median"] + margin
            ),
            "worst_passed": (
                back[metric]["worst"] >= baseline[metric]["worst"] + margin
            ),
        }
        conditions["paired_middle_late_control"][metric] = {
            "required_margin": margin,
            "median_passed": (
                back[metric]["median"] >= control[metric]["median"] + margin
            ),
            "worst_passed": (
                back[metric]["worst"] >= control[metric]["worst"] + margin
            ),
        }
        conditions["v15a_replication_stability"][metric] = {
            "maximum_decline": tolerance,
            "median_passed": (
                back[metric]["median"] >= v15a[metric]["median"] - tolerance
            ),
            "worst_passed": (
                back[metric]["worst"] >= v15a[metric]["worst"] - tolerance
            ),
        }
    conditions["integrity"] = {
        "all_panel_spreads_nonzero_both_arms_passed": True,
        "all_integrity_audits_passed": True,
    }
    passed = (
        all(
            flag
            for family in (
                "absolute_v13", "paired_middle_late_control",
                "v15a_replication_stability",
            )
            for condition in conditions[family].values()
            for label, flag in condition.items()
            if label.endswith("passed")
        )
        and all(conditions["integrity"].values())
    )
    gate = {
        "schema": "eggroll-es-back-plan-confirmation-gate-v15b",
        "eligible_for_separate_back_plan_train_update_preregistration": passed,
        "eligible_for_model_update": False,
        "eligible_to_open_evaluation": False,
        "conditions": conditions,
        "candidate_content_sha256": candidate[
            "content_sha256_before_self_field"
        ],
        "v13_baseline_evidence_content_sha256": (
            V13_EVIDENCE_CONTENT_SHA256_V15B
        ),
        "v15a_positive_evidence_content_sha256": (
            V15A_POSITIVE_CONTENT_SHA256_V15B
        ),
        "failure_decision": (
            None if passed
            else "retain_v13_middle_late_and_keep_all_eval_surfaces_closed"
        ),
        "pass_decision": (
            "preregister_back_plan_nonzero_alpha_train_update_experiment"
            if passed else None
        ),
    }
    gate["content_sha256_before_self_field"] = canonical_sha256(gate)
    return gate


def _exclusive_write(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("v15b preregistration already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default=str(PREREGISTRATION_PATH_V15B))
    args = parser.parse_args(argv)
    output = Path(args.output_json).resolve()
    if output != PREREGISTRATION_PATH_V15B:
        raise ValueError("v15b preregistration requires its canonical path")
    value = build_preregistration_v15b()
    _exclusive_write(output, value)
    result = {
        "schema": "eggroll-es-back-plan-confirmation-preregistration-write-v15b",
        "path": str(output),
        "file_sha256": _file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
