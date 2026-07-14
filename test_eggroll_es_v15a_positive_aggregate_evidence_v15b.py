#!/usr/bin/env python3

import hashlib
import json
import subprocess
from pathlib import Path

import eggroll_es_back_plan_preregistration_v15a as prereg


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / (
    "experiments/eggroll_es_hpo/runs/"
    "snapshot794_layer_v15a_back36_39_vs_middle_late20_23_"
    "paired_v13_panels_alpha_zero_basis20260715/"
    "paired_architecture_stability_v15a.json"
)
ATTEMPT = ROOT / (
    "experiments/eggroll_es_hpo/runs/"
    ".snapshot794_layer_v15a_back36_39_vs_middle_late20_23_"
    "paired_v13_panels_alpha_zero_basis20260715.launch_attempt.json"
)
PREREG = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_BACK_ONLY_ARCHITECTURE_STABILITY_V15A_PREREGISTRATION.json"
)
EVIDENCE = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V15A_BACK_PLAN_POSITIVE_AGGREGATE_EVIDENCE_V15B.json"
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


def test_v15a_positive_evidence_recomputes_exact_gate_without_content_access(
    monkeypatch,
):
    report = json.loads(REPORT.read_text())
    attempt = json.loads(ATTEMPT.read_text())
    frozen = json.loads(PREREG.read_text())
    evidence = json.loads(EVIDENCE.read_text())

    assert _file_sha256(REPORT) == (
        "6ea50294be2c0f498c55e0ee99ab11650419621ee6f2355d9a336e91e5f2ee86"
    )
    assert _file_sha256(ATTEMPT) == (
        "1856914876d075dac19a43af6a2a490e5c1050cc6705c8b8a177c3380e77aab8"
    )
    assert _file_sha256(PREREG) == (
        "ad86f388ff4effbc195a3fd60d6d32c430a83026a331a18d625d477d390f3b88"
    )
    for value in (report, attempt, frozen, evidence):
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
    assert recipe["alpha"] == 0.0
    assert recipe["model_update_allowed"] is False
    assert recipe["perturbation_basis"] == frozen["runtime"][
        "perturbation_basis"
    ]
    assert recipe["perturbation_basis_sha256"] == frozen["runtime"][
        "perturbation_basis_sha256"
    ]
    assert recipe["paired_architecture"]["arm_order"] == [
        "middle_late", "back",
    ]
    for arm in ("middle_late", "back"):
        runtime_arm = recipe["paired_architecture"]["arms"][arm]
        frozen_arm = frozen["paired_architecture"]["arms"][arm]
        for key in (
            "capacity", "file_sha256", "layers", "model_config_sha256",
            "path", "plan_sha256",
        ):
            assert runtime_arm[key] == frozen_arm[key]
        summary = report["arm_summaries"][arm]
        assert summary["all_panel_spreads_nonzero"] is True
        assert summary["persisted_response_vectors"] is False
        assert summary["persisted_row_content"] is False
        assert all(
            item is True for key, item in summary["integrity_audits"].items()
            if key != "content_sha256_before_self_field"
        )
        assert attempt["completed_arm_summary_bindings"][arm] == {
            "diagnostic_content_sha256": summary[
                "diagnostic_content_sha256"
            ],
            "plan_sha256": summary["plan_sha256"],
            "summary_content_sha256": summary[
                "content_sha256_before_self_field"
            ],
        }

    panel_identities = frozen["v13_estimator"]["ordered_panel_identities"]
    monkeypatch.setattr(
        prereg,
        "validate_v13_estimator_v15a",
        lambda: {
            "panels": {
                name: {"ordered_row_identity_sha256": identity}
                for name, identity in panel_identities.items()
            }
        },
    )
    candidate = dict(report["candidate_summary"])
    candidate["arms"] = {
        name: candidate["arms"][name] for name in candidate["arm_order"]
    }
    recomputed = prereg.evaluate_candidate_v15a(candidate)
    assert recomputed == report["promotion_gate"]
    assert recomputed["content_sha256_before_self_field"] == (
        "540cb4759e44e1c989414b50cd8c8bf577ddf77b1aa7b5153a34f1ffbab9fc7a"
    )
    assert recomputed["eligible_for_fresh_basis_back_plan_confirmation"]
    assert recomputed["eligible_for_model_update"] is False
    assert recomputed["eligible_to_open_evaluation"] is False
    assert all(
        flag
        for family in ("absolute_v13", "paired_middle_late_control")
        for condition in recomputed["conditions"][family].values()
        for label, flag in condition.items()
        if label.endswith("passed")
    )

    assert evidence["aggregate_gate"]["content_sha256"] == recomputed[
        "content_sha256_before_self_field"
    ]
    assert evidence["stability"]["back"] == report["candidate_summary"][
        "arms"
    ]["back"]["stability"]
    assert evidence["stability"]["middle_late"] == report[
        "candidate_summary"
    ]["arms"]["middle_late"]["stability"]
    assert evidence["stability"]["v13_baseline"] == frozen[
        "promotion_gate"
    ]["historical_v13_baseline"]
    assert evidence["decision"] == {
        "back_alpha_zero_confirmation_on_another_fresh_basis_authorized": True,
        "evaluation_surface_opened_or_authorized": False,
        "front_plus_back_authorized": False,
        "model_update_applied_or_authorized": False,
        "retained_recipe_pending_confirmation": "V13 middle-late layers 20-23",
    }
    assert report["model_update_applied"] is False
    assert report["persisted_response_vectors_or_row_content"] is False
    assert report["sealed_or_nontrain_surface_opened"] is False
    assert not {
        "responses", "coefficients", "questions", "answers", "documents",
    }.intersection(_all_keys(evidence))
