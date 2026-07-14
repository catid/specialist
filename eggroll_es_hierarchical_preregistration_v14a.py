#!/usr/bin/env python3
"""Immutable V14a train-only sampler preregistration and numeric gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import eggroll_es_hierarchical_train_sampler_v14 as sampler_v14
import eggroll_es_train_panel_sampler_v13 as sampler_v13


ROOT = Path(__file__).resolve().parent
EVIDENCE_PATH_V14A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V13B_TRAIN_PANEL_AGGREGATE_EVIDENCE_V14A.json"
).resolve()
PREREGISTRATION_PATH_V14A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_HIERARCHICAL_ROTATING_TRAIN_PANELS_V14A_PREREGISTRATION.json"
).resolve()
EVIDENCE_FILE_SHA256_V14A = (
    "d367c9c4de1e1f3526ddb3dfba2f5bf24efc77cbccf951f7359eb1969fcd7b54"
)
EVIDENCE_CONTENT_SHA256_V14A = (
    "06f662574013345a6c777af8688a38f3941286d9e11a427ed3342de53451b1e3"
)
V12_NEGATIVE_EVIDENCE_PATH_V14A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V12_PRESEAL_NEGATIVE_AGGREGATE_EVIDENCE_V14A.json"
).resolve()
V12_NEGATIVE_EVIDENCE_FILE_SHA256_V14A = (
    "4fec87ad1f41e40ba2ebc97dd46b58f4f7bf345e78d364a0ff2e98b9969a6512"
)
V12_NEGATIVE_EVIDENCE_CONTENT_SHA256_V14A = (
    "ce259d30481d8de85089116a215ac710db791e27ffd1592f846c5c9a1e56bb59"
)
SAMPLER_FILE_SHA256_V14A = (
    "6981a746d6e0fc0904603abaf584ab71b9cc8a777a9abc00f4d305a98ebd186a"
)
SOURCE_SHA256_V14A = sampler_v13.SOURCE_SHA256
SOURCE_ARROW_SHA256_V14A = sampler_v13.SOURCE_ARROW_SHA256
FRAME_SHA256_V14A = (
    "ce50531881f4b7044bf82fc3e8fd52d603ba53041fccc4b934aa307840862d6c"
)
POLICY_CONTENT_SHA256_V14A = (
    "b4c8d038da0c670f2cc8602b822e249de427b124e80fd2d65d69b6c48d980ebc"
)
PERTURBATION_BASIS_SHA256_V14A = (
    "29e7ceb1753c39b310a176d827e222b9a5b2c85edf9f2fef5c68b630b8fabc11"
)
EXPERIMENT_NAME_V14A = (
    "snapshot794_layer_v14a_full_frame_matched56_crossfit_alpha_zero_"
    "resident_sign_basis20260714"
)
FULL_FRAME_IDENTITY_V14A = {
    "row_draw_iteration": 0,
    "rows": 310,
    "content_sha256": (
        "f91b3388226e2d6cfec60e4f62c2cc5e2b28161b8fd5071d89a5700540a587b2"
    ),
    "ordered_row_identity_sha256": (
        "2f2de46c2e4c35ba03aedba93a7ce58426aa17e5028263f1035d7199e650c798"
    ),
}
PANEL_NAMES_V14A = (
    "optimization_0", "optimization_1", "optimization_2",
    "train_screen_0", "train_screen_1",
)
PANEL_ROLES_V14A = {
    "optimization_0": "optimization",
    "optimization_1": "optimization",
    "optimization_2": "optimization",
    "train_screen_0": "train_only_screen",
    "train_screen_1": "train_only_screen",
}
PANEL_IDENTITIES_V14A = {
    "optimization_0": {
        "iteration": 0,
        "content_sha256": (
            "5e0c16a68a2c64a5b1ee9ce95786bb2a1bcd229d4bc5ed0878c723337256898b"
        ),
        "ordered_row_identity_sha256": (
            "97444de398399cf4f5875258b9c853629bb9d3afdaec858c664f3dab611ad1d1"
        ),
    },
    "optimization_1": {
        "iteration": 1,
        "content_sha256": (
            "8109e6a9c5d1de68e41a260bcb863316a415ef5623dff93e1d73502586991b94"
        ),
        "ordered_row_identity_sha256": (
            "c7c700cca6202a35381df596b994f018da0dfab0cfb45832ede2f1486a8cbad8"
        ),
    },
    "optimization_2": {
        "iteration": 2,
        "content_sha256": (
            "8ef3b4f774dc64720fa6dc05104b86c1bea78e4b6a30c991365e16baf26ec685"
        ),
        "ordered_row_identity_sha256": (
            "0d67a589670532dcd9ee5b447036e38da222101ce196b381ff5c44b8c23a5403"
        ),
    },
    "train_screen_0": {
        "iteration": 3,
        "content_sha256": (
            "7c7e01ec15bf4e438341bed25763c5f00e4f7eef58d1550da944a1c5952e1a4b"
        ),
        "ordered_row_identity_sha256": (
            "9855bd37e99eea106a3c53d72f5e8d2153d3f17c7bb1f973006e0d0e0d75f3c9"
        ),
    },
    "train_screen_1": {
        "iteration": 4,
        "content_sha256": (
            "47cd836f4a0acf995ff07ba5aa67e1f2bb84ec8f97051c39db399518532e3688"
        ),
        "ordered_row_identity_sha256": (
            "0b3c066f684186fd9ce7025a114088c973009034fb5ff78f6c3766baffad81f6"
        ),
    },
}
BASELINE_STABILITY_V14A = {
    "matched56_pairwise_cosine": {
        "count": 3, "median": 0.47411088498906484,
        "worst": 0.3900621868364503,
    },
    "matched56_pairwise_sign_agreement": {
        "count": 3, "median": 0.59375, "worst": 0.5625,
    },
    "full_to_matched56_optimization_cosine": {
        "count": 3, "median": 0.7608236805612648,
        "worst": 0.7082628389768383,
    },
    "full_to_matched56_optimization_sign_agreement": {
        "count": 3, "median": 0.8125, "worst": 0.75,
    },
    "crossfit_complement_to_screen_cosine": {
        "count": 2, "median": 0.3936314430866483,
        "worst": 0.314941371734614,
    },
    "crossfit_complement_to_screen_sign_agreement": {
        "count": 2, "median": 0.65625, "worst": 0.53125,
    },
}
BASELINE_EVIDENCE_KEY_V14A = {
    "matched56_pairwise_cosine": "optimization_pairwise_cosine",
    "matched56_pairwise_sign_agreement": (
        "optimization_pairwise_sign_agreement"
    ),
    "full_to_matched56_optimization_cosine": (
        "aggregate_to_optimization_cosine"
    ),
    "full_to_matched56_optimization_sign_agreement": (
        "aggregate_to_optimization_sign_agreement"
    ),
    "crossfit_complement_to_screen_cosine": "train_screen_cosine",
    "crossfit_complement_to_screen_sign_agreement": (
        "train_screen_sign_agreement"
    ),
}


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    return sampler_v13.canonical_sha256(value)


def load_evidence_v14a(path=EVIDENCE_PATH_V14A):
    path = Path(path).resolve()
    if (
        path != EVIDENCE_PATH_V14A
        or _file_sha256(path) != EVIDENCE_FILE_SHA256_V14A
    ):
        raise RuntimeError("v14a compact V13b evidence file identity changed")
    evidence = json.loads(path.read_text())
    if (
        evidence.get("schema")
        != "eggroll-es-v13b-train-panel-aggregate-evidence-v14a"
        or evidence.get("passed") is not True
        or evidence.get("selection_surface") != "frozen_train_panels_only"
        or evidence.get("contains_response_vectors_or_row_content") is not False
        or evidence.get("contains_validation_ood_or_heldout_content") is not False
        or evidence.get("content_sha256_before_self_field")
        != EVIDENCE_CONTENT_SHA256_V14A
        or evidence.get("content_sha256_before_self_field")
        != _canonical({
            key: value for key, value in evidence.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v14a compact V13b evidence content changed")
    for key, value in BASELINE_STABILITY_V14A.items():
        if evidence["stability"].get(BASELINE_EVIDENCE_KEY_V14A[key]) != value:
            raise RuntimeError("v14a frozen stability baseline changed")
    return evidence


def load_v12_negative_evidence_v14a(path=V12_NEGATIVE_EVIDENCE_PATH_V14A):
    path = Path(path).resolve()
    if (
        path != V12_NEGATIVE_EVIDENCE_PATH_V14A
        or _file_sha256(path) != V12_NEGATIVE_EVIDENCE_FILE_SHA256_V14A
    ):
        raise RuntimeError("v14a V12 negative evidence file identity changed")
    evidence = json.loads(path.read_text())
    if (
        evidence.get("schema")
        != "eggroll-es-v12-preseal-negative-aggregate-evidence-v14a"
        or evidence.get("passed") is not True
        or evidence.get("decision")
        != "close_v12_candidate_without_confirmation_or_release"
        or evidence.get("contains_response_documents_or_row_content") is not False
        or evidence.get("contains_validation_ood_or_heldout_content") is not False
        or evidence.get("aggregate_failure", {}).get(
            "eligible_positive_alpha_count"
        ) != 0
        or evidence.get("aggregate_failure", {}).get(
            "all_positive_screen_lcbs_negative"
        ) is not True
        or evidence.get("aggregate_failure", {}).get(
            "all_positive_anchor_lcbs_negative"
        ) is not True
        or evidence.get("aggregate_failure", {}).get("candidate_seal_written")
        is not False
        or evidence.get("content_sha256_before_self_field")
        != V12_NEGATIVE_EVIDENCE_CONTENT_SHA256_V14A
        or evidence.get("content_sha256_before_self_field")
        != _canonical({
            key: value for key, value in evidence.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v14a V12 negative evidence content changed")
    return evidence


def materialize_panels_v14a():
    if _file_sha256(sampler_v14.__file__) != SAMPLER_FILE_SHA256_V14A:
        raise RuntimeError("v14a sampler implementation changed")
    rows, source_sha256 = sampler_v13.load_frozen_train()
    documents, frame = sampler_v14.build_document_frame(rows)
    if (
        source_sha256 != SOURCE_SHA256_V14A
        or frame["frame_sha256"] != FRAME_SHA256_V14A
        or frame["documents"] != 310
        or len(documents) != 310
        or sampler_v14.build_policy(rows)["content_sha256_before_self_field"]
        != POLICY_CONTENT_SHA256_V14A
    ):
        raise RuntimeError("v14a frozen document frame or policy changed")
    full_items = []
    for document in documents:
        row_index = sampler_v14._choose_row(document, rows, 0)
        row = rows[row_index]
        full_items.append({
            "document_sha256": document["document_sha256"],
            "document_row_count": document["row_count"],
            "fact_id": row["fact_id"],
            "row_index": row_index,
            "row_sha256": sampler_v13.row_sha256(row),
            "source": row.get("source"),
            "stratum": document["stratum"],
            "full_frame_equal_document_weight": 1.0,
        })
    full_items.sort(key=lambda item: sampler_v14._key(
        "full-frame-panel-order-v14a",
        item["document_sha256"], item["row_sha256"],
    ))
    for position, item in enumerate(full_items):
        item["position"] = position
    full_frame = {
        "schema": "eggroll-es-full-frame-document-panel-v14a",
        "train_only": True,
        "row_draw_iteration": 0,
        "rows": 310,
        "frame_sha256": frame["frame_sha256"],
        "ordered_row_identity_sha256": _canonical([
            item["row_sha256"] for item in full_items
        ]),
        "items": full_items,
        "validation_ood_or_heldout_used": False,
    }
    full_frame["content_sha256_before_self_field"] = _canonical(full_frame)
    if (
        full_frame["content_sha256_before_self_field"]
        != FULL_FRAME_IDENTITY_V14A["content_sha256"]
        or full_frame["ordered_row_identity_sha256"]
        != FULL_FRAME_IDENTITY_V14A["ordered_row_identity_sha256"]
    ):
        raise RuntimeError("v14a frozen full-frame panel changed")

    by_document = {
        item["document_sha256"]: item for item in full_frame["items"]
    }
    result = {}
    all_documents = set()
    for name in PANEL_NAMES_V14A:
        expected = PANEL_IDENTITIES_V14A[name]
        base_panel = sampler_v14.build_iteration_panel(
            rows, expected["iteration"],
        )
        items = []
        for base_item in base_panel["items"]:
            item = {
                key: value for key, value in by_document[
                    base_item["document_sha256"]
                ].items() if key != "position"
            }
            item["equal_document_ht_weight"] = base_item[
                "equal_document_ht_weight"
            ]
            item["position"] = len(items)
            items.append(item)
        panel = {
            "schema": "eggroll-es-full-frame-matched-subpanel-v14a",
            "train_only": True,
            "name": name,
            "role": PANEL_ROLES_V14A[name],
            "document_allocation_iteration": expected["iteration"],
            "row_draw_iteration": 0,
            "rows": 56,
            "base_v14_panel_content_sha256": base_panel[
                "content_sha256_before_self_field"
            ],
            "ordered_row_identity_sha256": _canonical([
                item["row_sha256"] for item in items
            ]),
            "items": items,
            "validation_ood_or_heldout_used": False,
        }
        panel["content_sha256_before_self_field"] = _canonical(panel)
        if (
            panel["content_sha256_before_self_field"]
            != expected["content_sha256"]
            or panel["ordered_row_identity_sha256"]
            != expected["ordered_row_identity_sha256"]
            or panel["validation_ood_or_heldout_used"] is not False
        ):
            raise RuntimeError(f"v14a frozen {name} panel changed")
        documents_here = {item["document_sha256"] for item in panel["items"]}
        if all_documents.intersection(documents_here):
            raise RuntimeError("v14a five-panel document disjointness changed")
        all_documents.update(documents_here)
        result[name] = panel
    if len(all_documents) != 280:
        raise RuntimeError("v14a five-panel document coverage changed")
    if any(
        item["document_sha256"] not in by_document
        for panel in result.values() for item in panel["items"]
    ):
        raise RuntimeError("v14a matched subpanel left the full frame")
    return rows, full_frame, result


def build_preregistration_v14a():
    evidence = load_evidence_v14a()
    v12_negative = load_v12_negative_evidence_v14a()
    _rows, full_frame, panels = materialize_panels_v14a()
    preregistration = {
        "schema": "eggroll-es-full-frame-matched56-preregistration-v14a",
        "status": "preregistered_not_launch_authorized",
        "experiment_name": EXPERIMENT_NAME_V14A,
        "selection_surface": "frozen_train_panels_only",
        "contains_validation_ood_or_heldout_content": False,
        "v13b_aggregate_evidence": {
            "path": str(EVIDENCE_PATH_V14A),
            "file_sha256": EVIDENCE_FILE_SHA256_V14A,
            "content_sha256": EVIDENCE_CONTENT_SHA256_V14A,
            "v13b_report_content_sha256": evidence["v13b_report"][
                "content_sha256"
            ],
        },
        "v12_preseal_negative_evidence": {
            "path": str(V12_NEGATIVE_EVIDENCE_PATH_V14A),
            "file_sha256": V12_NEGATIVE_EVIDENCE_FILE_SHA256_V14A,
            "content_sha256": V12_NEGATIVE_EVIDENCE_CONTENT_SHA256_V14A,
            "report_content_sha256": v12_negative["report"]["content_sha256"],
            "decision": v12_negative["decision"],
        },
        "sampling": {
            "sampler_path": str(Path(sampler_v14.__file__).resolve()),
            "sampler_file_sha256": SAMPLER_FILE_SHA256_V14A,
            "source_sha256": SOURCE_SHA256_V14A,
            "source_arrow_sha256": SOURCE_ARROW_SHA256_V14A,
            "frame_sha256": FRAME_SHA256_V14A,
            "policy_content_sha256": POLICY_CONTENT_SHA256_V14A,
            "generation_prompt_count_per_direction_and_sign": 310,
            "v13b_generation_prompt_count_per_direction_and_sign": 280,
            "relative_prompt_increase_over_v13b": 310.0 / 280.0 - 1.0,
            "full_frame_documents": 310,
            "full_frame_document_selection_variance": 0.0,
            "matched_subpanel_size": 56,
            "unique_documents_across_five_matched_subpanels": 280,
            "row_draw_iteration": 0,
            "hard_replay_fraction": 0.0,
        },
        "full_frame": {
            **copy.deepcopy(FULL_FRAME_IDENTITY_V14A),
            "rows": full_frame["rows"],
            "estimand": "equal_weight_mean_of_310_document_row_draws",
        },
        "matched56_subpanels": {
            name: {
                **copy.deepcopy(PANEL_IDENTITIES_V14A[name]),
                "role": PANEL_ROLES_V14A[name],
                "rows": panels[name]["rows"],
                "is_subset_of_full_frame_generation": True,
            }
            for name in PANEL_NAMES_V14A
        },
        "runtime": {
            "model": str((ROOT / "models/Qwen3.6-35B-A3B").resolve()),
            "alpha": 0.0,
            "model_update_allowed": False,
            "population_size": 32,
            "perturbation_basis_sha256": PERTURBATION_BASIS_SHA256_V14A,
            "sign_order": ["plus", "minus"],
            "engine_count": 4,
            "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "complete_four_direction_waves_required": True,
            "exact_restore_after_every_sign": True,
            "one_full_frame_order_reused_across_every_direction_and_sign": True,
            "matched56_responses_derived_without_extra_generation": True,
        },
        "analysis": {
            "full_frame_response": "unweighted mean of all 310 documents",
            "matched56_optimization_panels": list(PANEL_NAMES_V14A[:3]),
            "matched56_train_screens": list(PANEL_NAMES_V14A[3:]),
            "central_response": "(plus-minus)/2",
            "per_panel_standardization_epsilon": 1e-8,
            "matched56_response": (
                "stratum-weighted equal-document Horvitz-Thompson mean"
            ),
            "crossfit_screen_response": (
                "each 56-document screen compared with a separately "
                "standardized response from its 254-document complement"
            ),
            "screens_excluded_from_their_complement": True,
            "full_frame_is_primary_estimator": True,
            "alternate_row_draw_confirmation": (
                "separate later alpha-zero run using row_draw_iteration=1"
            ),
        },
        "promotion_gate": {
            "scope": "sampler_adoption_only_not_model_update",
            "baseline": copy.deepcopy(BASELINE_STABILITY_V14A),
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
                "all_panel_spreads_nonzero": True,
                "all_rules_required": True,
            },
            "failure_decision": (
                "retain_v13_sampler_and_do_not_open_eval_surfaces"
            ),
            "pass_decision": (
                "run_a_separate_alpha_zero_row_draw_iteration_1_confirmation; "
                "still do not open evaluation surfaces or apply an update"
            ),
        },
        "firewall": {
            "allowed": [
                "frozen train source", "aggregate V13b train evidence",
                "aggregate V12 closed-preseal evidence",
            ],
            "forbidden": [
                "validation", "OOD", "heldout", "benchmark outcomes",
                "V11f relaunch", "adaptive hard replay",
            ],
            "v11f_status": "immutable_failed_superseded_by_completed_v11g",
            "v12_candidate_status": (
                "closed_no_eligible_alpha_no_confirmation_no_release"
            ),
        },
        "architecture_roadmap_after_estimator_confirmation": {
            "not_part_of_v14a": True,
            "prior_result": "front_only_failed_slot_cosine_0.0687",
            "alpha_zero_same_basis_order": [
                "back_layers_36_39", "front_0_3_plus_back_36_39",
                "middle_late_control",
            ],
            "layer_insertion_or_nonzero_update_before_comparison": False,
        },
        "required_runtime_adapter": {
            "status": "not_yet_implemented",
            "required_files": [
                "train_eggroll_es_specialist_anchor_v14a.py",
                "run_eggroll_es_hierarchical_train_panels_v14a.py",
                "test_eggroll_es_hierarchical_train_panels_v14a.py",
            ],
            "inherit_update_disabled_worker": "eggroll_es_worker_v13.py",
            "real_launch_requires_fresh_committed_implementation_bundle": True,
        },
    }
    preregistration["content_sha256_before_self_field"] = _canonical(
        preregistration
    )
    return preregistration


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
        raise RuntimeError("v14a candidate metric summary changed")
    return value


def evaluate_candidate_v14a(candidate):
    expected_keys = {
        "schema", "experiment_name", "alpha", "model_update_applied",
        "validation_ood_or_heldout_used", "perturbation_basis_sha256",
        "panel_identities", "stability", "all_panel_spreads_nonzero",
        "robust_aggregate", "content_sha256_before_self_field",
    }
    if (
        not isinstance(candidate, dict)
        or set(candidate) != expected_keys
        or candidate.get("schema")
        != "eggroll-es-full-frame-matched56-summary-v14a"
        or candidate.get("experiment_name") != EXPERIMENT_NAME_V14A
        or candidate.get("alpha") != 0.0
        or candidate.get("model_update_applied") is not False
        or candidate.get("validation_ood_or_heldout_used") is not False
        or candidate.get("perturbation_basis_sha256")
        != PERTURBATION_BASIS_SHA256_V14A
        or candidate.get("panel_identities") != {
            "full_frame": FULL_FRAME_IDENTITY_V14A[
                "ordered_row_identity_sha256"
            ],
            **{
                name: PANEL_IDENTITIES_V14A[name][
                    "ordered_row_identity_sha256"
                ] for name in PANEL_NAMES_V14A
            },
        }
        or candidate.get("content_sha256_before_self_field")
        != _canonical({
            key: value for key, value in candidate.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v14a candidate summary contract changed")
    stability = candidate.get("stability", {})
    if set(stability) != set(BASELINE_STABILITY_V14A):
        raise RuntimeError("v14a candidate stability surface changed")
    for key in stability:
        _metric(stability[key], BASELINE_STABILITY_V14A[key]["count"])
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
        raise RuntimeError("v14a candidate robust aggregate changed")

    conditions = {}
    for key, baseline in BASELINE_STABILITY_V14A.items():
        value = stability[key]
        strict = key in {
            "matched56_pairwise_cosine",
            "crossfit_complement_to_screen_cosine",
        }
        conditions[key] = {
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
    passed = all(
        all(
            flag for name, flag in condition.items()
            if name.endswith("passed")
        )
        for condition in conditions.values()
    )
    gate = {
        "schema": "eggroll-es-hierarchical-sampler-promotion-gate-v14a",
        "eligible_for_train_only_sampler_adoption": passed,
        "eligible_for_model_update": False,
        "conditions": conditions,
        "candidate_content_sha256": candidate[
            "content_sha256_before_self_field"
        ],
        "baseline_evidence_content_sha256": EVIDENCE_CONTENT_SHA256_V14A,
        "failure_decision": (
            None if passed else "retain_v13_sampler_and_keep_eval_surfaces_closed"
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
        raise ValueError("v14a preregistration already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default=str(PREREGISTRATION_PATH_V14A))
    args = parser.parse_args(argv)
    output = Path(args.output_json).resolve()
    if output != PREREGISTRATION_PATH_V14A:
        raise ValueError("v14a preregistration requires its canonical path")
    preregistration = build_preregistration_v14a()
    _exclusive_write(output, preregistration)
    print(json.dumps({
        "output": str(output),
        "file_sha256": _file_sha256(output),
        "content_sha256": preregistration["content_sha256_before_self_field"],
    }, sort_keys=True))
    return preregistration


if __name__ == "__main__":
    main()
