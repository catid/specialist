#!/usr/bin/env python3

import hashlib
import json
import subprocess
from pathlib import Path

import eggroll_es_back_plan_confirmation_preregistration_v15b as prereg


ROOT = Path(__file__).resolve().parent
RUN_NAME = prereg.EXPERIMENT_NAME_V15B
REPORT = ROOT / (
    "experiments/eggroll_es_hpo/runs/" + RUN_NAME
    + "/paired_back_confirmation_v15b.json"
)
ATTEMPT = ROOT / (
    "experiments/eggroll_es_hpo/runs/." + RUN_NAME + ".launch_attempt.json"
)
EVIDENCE = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V15B_BACK_PLAN_CONFIRMATION_NEGATIVE_AGGREGATE_EVIDENCE_V16.json"
)


def _canonical(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_self_hashes(value):
    if isinstance(value, dict):
        if "content_sha256_before_self_field" in value:
            assert value["content_sha256_before_self_field"] == _canonical({
                key: item for key, item in value.items()
                if key != "content_sha256_before_self_field"
            })
        for item in value.values():
            _assert_self_hashes(item)
    elif isinstance(value, list):
        for item in value:
            _assert_self_hashes(item)


def _all_keys(value):
    result = set()
    if isinstance(value, dict):
        result.update(value)
        for item in value.values():
            result.update(_all_keys(item))
    elif isinstance(value, list):
        for item in value:
            result.update(_all_keys(item))
    return result


def test_v15b_negative_evidence_exactly_replays_sealed_compact_gate():
    report = json.loads(REPORT.read_text())
    attempt = json.loads(ATTEMPT.read_text())
    frozen = json.loads(prereg.PREREGISTRATION_PATH_V15B.read_text())
    v15a = json.loads(prereg.V15A_POSITIVE_PATH_V15B.read_text())
    v13 = json.loads(prereg.V13_EVIDENCE_PATH_V15B.read_text())
    evidence = json.loads(EVIDENCE.read_text())

    assert _file_sha256(REPORT) == (
        "a48af8ae7435afe658305399984699a8ed639df4db1bc025ec643ef5b221082a"
    )
    assert _file_sha256(ATTEMPT) == (
        "63af96edeb3b73cd7b9b61e4ae9d8300b250ee5ed2ff4a005e61176dd3dd9c2f"
    )
    assert _file_sha256(prereg.PREREGISTRATION_PATH_V15B) == (
        "5b90f16961c94d3a04b72ae29860094f7f1e6e8bad793780967c27448e0ba57f"
    )
    assert _file_sha256(prereg.V15A_POSITIVE_PATH_V15B) == (
        "1e14abee9e1514915bc241c8f6caacbe1bb7103e1c69a9afdde1f9ce13661ae1"
    )
    assert _file_sha256(prereg.V13_EVIDENCE_PATH_V15B) == (
        "d367c9c4de1e1f3526ddb3dfba2f5bf24efc77cbccf951f7359eb1969fcd7b54"
    )
    for value in (report, attempt, frozen, v15a, v13, evidence):
        _assert_self_hashes(value)

    assert attempt["status"] == "complete"
    assert attempt["phase"] == "after_both_arm_cleanups_and_report"
    assert attempt["report_binding"] == {
        "path": str(REPORT),
        "file_sha256": _file_sha256(REPORT),
        "content_sha256": report["content_sha256_before_self_field"],
    }
    assert attempt["recipe"] == report["recipe"]
    assert attempt["source_provenance"]["implementation_bundle_sha256"] == (
        report["implementation"]["bundle_sha256"]
    )
    assert _canonical(report["implementation"]["files"]) == (
        report["implementation"]["bundle_sha256"]
    )
    subprocess.run(
        [
            "git", "cat-file", "-e",
            attempt["source_provenance"]["git_head"] + "^{commit}",
        ],
        cwd=ROOT,
        check=True,
    )

    recipe = report["recipe"]
    candidate = report["candidate_summary"]
    assert recipe["content_sha256_before_self_field"] == (
        "da2dfe0ab624120a5729fd89c31f119b953f57d5a863b0ea190e7e892005d34c"
    )
    assert recipe["alpha"] == 0.0
    assert recipe["model_update_allowed"] is False
    assert recipe["perturbation_basis"] == frozen["runtime"][
        "perturbation_basis"
    ]
    assert recipe["paired_architecture"]["arm_order"] == [
        "middle_late", "back",
    ]
    assert recipe["panel_bundle_content_sha256"] == frozen["estimator"][
        "panel_bundle_content_sha256"
    ]
    assert candidate["panel_identities"] == frozen["estimator"][
        "ordered_panel_identities"
    ]
    for arm in ("middle_late", "back"):
        summary = report["arm_summaries"][arm]
        assert summary["all_panel_spreads_nonzero"] is True
        assert summary["persisted_response_vectors"] is False
        assert summary["persisted_row_content"] is False
        assert all(
            item is True for key, item in summary["integrity_audits"].items()
            if key != "content_sha256_before_self_field"
        )
        assert candidate["arms"][arm] == {
            key: summary[key]
            for key in ("plan_sha256", "stability", "robust_aggregate")
        }
        assert report["configurations"][arm][
            "persisted_configuration_payload"
        ] is False

    replayed = prereg.evaluate_candidate_v15b(candidate)
    assert replayed == report["promotion_gate"]
    assert replayed["content_sha256_before_self_field"] == (
        "e1dbfed2988ff1295d1dad6ba6eaf71ad5bfa83549b6cd770d80aa8f90b46a8e"
    )
    assert replayed[
        "eligible_for_separate_back_plan_train_update_preregistration"
    ] is False
    assert replayed["eligible_for_model_update"] is False
    assert replayed["eligible_to_open_evaluation"] is False

    failed = []
    for family in (
        "absolute_v13", "paired_middle_late_control",
        "v15a_replication_stability",
    ):
        for metric, condition in replayed["conditions"][family].items():
            for endpoint in ("median", "worst"):
                if condition[f"{endpoint}_passed"] is False:
                    failed.append(f"{family}.{metric}.{endpoint}")
    assert set(failed) == set(evidence["aggregate_gate"]["failed_checks"])
    assert evidence["aggregate_gate"]["content_sha256"] == replayed[
        "content_sha256_before_self_field"
    ]
    assert evidence["stability"]["back"] == candidate["arms"]["back"][
        "stability"
    ]
    assert evidence["stability"]["middle_late"] == (
        candidate["arms"]["middle_late"]["stability"]
    )
    assert evidence["stability"]["v13_baseline"] == frozen[
        "promotion_gate"
    ]["historical_v13_baseline"]
    assert evidence["stability"]["v15a_back"] == frozen[
        "promotion_gate"
    ]["v15a_back_reference"]
    assert evidence["decision"] == {
        "back_plan_nonzero_alpha_train_update_preregistration_authorized": False,
        "evaluation_surface_opened_or_authorized": False,
        "model_update_applied_or_authorized": False,
        "reason": "v15b_failed_its_preregistered_conjunctive_confirmation_gate",
        "sampler": "retain_v13_middle_late_layers_20_23",
    }
    assert report["decision"] == (
        "retain_v13_middle_late_and_keep_all_eval_surfaces_closed"
    )
    assert report["model_update_applied"] is False
    assert report["sealed_or_nontrain_surface_opened"] is False
    assert report["persisted_response_vectors_or_row_content"] is False
    assert not {
        "responses", "coefficients", "questions", "answers", "documents",
    }.intersection(_all_keys(evidence))
