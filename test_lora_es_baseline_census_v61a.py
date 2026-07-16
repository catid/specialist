#!/usr/bin/env python3

import copy
import json
import types

import pytest

import lora_es_baseline_census_strata_v61a as subject
import build_lora_es_baseline_census_preregistration_v61a as builder
import run_lora_es_baseline_census_v61a as runtime


def _sha(number: int) -> str:
    return f"{number:064x}"


def _actors(f1s, exacts=None):
    if exacts is None:
        exacts = [int(value == 1.0) for value in f1s]
    return [{
        "actor_rank": rank,
        "generation_seed": subject.ACTOR_GENERATION_SEEDS_V61A[rank],
        "f1": float(f1),
        "exact": int(exact),
        "nonzero": int(f1 > 0.0),
    } for rank, (f1, exact) in enumerate(zip(f1s, exacts, strict=True))]


def _row(index: int, unit: int, f1s, exacts=None):
    return {
        "row_index": index,
        "row_sha256": _sha(index + 1),
        "unit_identity_sha256": _sha(10_000 + unit),
        "row_count": 1,
        "actors": _actors(f1s, exacts),
    }


def test_v61a_row_strata_are_exhaustive_and_content_free():
    cases = [
        ([1.0] * 4, "stable_exact"),
        ([0.5] * 4, "stable_partial"),
        ([0.2] * 4, "difficult"),
        ([0.0] * 4, "difficult"),
        ([0.5, 0.5, 0.0, 0.5], "actor_unstable"),
        ([0.5, 0.5, 0.5, 0.5000001], "actor_unstable"),
    ]
    for index, (f1s, expected) in enumerate(cases):
        classified = subject.classify_row_v61a(_row(index, index, f1s))
        assert classified["stratum"] == expected
        encoded = json.dumps(classified, sort_keys=True)
        assert '"question"' not in encoded
        assert '"answer"' not in encoded
        assert '"generation"' not in encoded


def _synthetic_evidence(stable_exact_units: int = 8):
    rows = []
    for unit in range(subject.TRAIN_CONFLICT_UNITS_V61A):
        if unit < stable_exact_units:
            f1s = [1.0] * 4
        elif unit % 4 == 0:
            f1s = [0.5] * 4
        elif unit % 4 == 1:
            f1s = [0.1] * 4
        elif unit % 4 == 2:
            f1s = [0.0] * 4
        else:
            f1s = [0.5, 0.5, 0.0, 0.5]
        rows.append(_row(unit, unit, f1s))
    # Add 240 rows to existing units, preserving declared multiplicities.
    for extra in range(subject.TRAIN_ROWS_V61A - len(rows)):
        unit = extra % subject.TRAIN_CONFLICT_UNITS_V61A
        original = rows[unit]
        original["row_count"] += 1
        clone = copy.deepcopy(original)
        clone["row_index"] = len(rows)
        clone["row_sha256"] = _sha(len(rows) + 1)
        rows.append(clone)
    for row in rows:
        unit = int(row["unit_identity_sha256"], 16) - 10_000
        row["row_count"] = sum(
            item["unit_identity_sha256"] == row["unit_identity_sha256"]
            for item in rows
        )
        assert unit >= 0
    evidence = {
        "schema": "v61a-v434-train-baseline-census-evidence",
        "status": "complete_characterization_only",
        "row_count": subject.TRAIN_ROWS_V61A,
        "conflict_unit_count": subject.TRAIN_CONFLICT_UNITS_V61A,
        "actor_count": subject.ACTORS_V61A,
        "actor_generation_seeds": list(subject.ACTOR_GENERATION_SEEDS_V61A),
        "generation_params_without_seed": dict(
            subject.GENERATION_PARAMS_WITHOUT_SEED_V61A
        ),
        "rows": rows,
        "raw_question_answer_or_generation_text_persisted": False,
        "eval_ood_shadow_or_holdout_opened": False,
        "candidate_selection_or_promotion_performed": False,
        "content_sha256_before_self_field": _sha(999_999),
    }
    return evidence


def test_v61a_partition_is_deterministic_disjoint_and_exact_supported():
    evidence = _synthetic_evidence(8)
    first = subject.build_stratified_census_v61a(evidence)
    second = subject.build_stratified_census_v61a(copy.deepcopy(evidence))
    assert first == second
    assert first["later_v61_hpo_authorized"] is True
    assert first["conflict_unit_count"] == 208
    assert len(first["units"]) == 208
    assert len({item["unit_identity_sha256"] for item in first["units"]}) == 208
    exact = first["stratum_counts"]["stable_exact"]
    assert exact == {"total": 8, "selection_pool": 6, "holdback": 2}
    encoded = json.dumps(first, sort_keys=True)
    assert '"question"' not in encoded
    assert '"answer"' not in encoded


def test_v61a_stable_exact_shortfall_fails_closed_without_quota_relaxation():
    result = subject.build_stratified_census_v61a(_synthetic_evidence(7))
    assert result["status"] == "fail_closed_insufficient_stable_exact_support"
    assert result["later_v61_hpo_authorized"] is False
    assert result["stratum_counts"]["stable_exact"]["total"] == 7
    assert result["stable_exact_fail_closed_minima"] == {
        "total": 8, "selection_pool": 4, "holdback": 2,
    }


def test_v61a_rejects_actor_seed_or_metric_schema_drift():
    row = _row(0, 0, [0.5] * 4)
    row["actors"][2]["generation_seed"] += 1
    with pytest.raises(ValueError, match="actor metric"):
        subject.classify_row_v61a(row)


def test_v61a_preregistration_freezes_census_before_live_access(tmp_path):
    value = builder.build_v61a()
    assert value["status"] == (
        "preregistered_before_v61a_model_gpu_or_train_row_access"
    )
    assert value["candidate_selection_update_or_promotion_authorized"] is False
    assert value["access_contract"]["preregistration_builder_reads_train_rows"] is False
    assert value["fixed_census_recipe"]["greedy_completions"] == 1792
    assert value["frozen_stratification_and_partition"][
        "quota_or_threshold_relaxation_after_outcomes"
    ] is False
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    args = types.SimpleNamespace(
        preregistration=str(path),
        preregistration_sha256=runtime.file_sha256_v61a(path),
        preregistration_content_sha256=value[
            "content_sha256_before_self_field"
        ],
    )
    assert runtime.load_preregistration_v61a(args) == value


def test_v61a_dry_run_opens_no_train_model_or_gpu(
    tmp_path, monkeypatch, capsys,
):
    value = builder.build_v61a()
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(
        runtime, "load_train_inputs_v61a",
        lambda: (_ for _ in ()).throw(AssertionError("dry-run opened train")),
    )
    monkeypatch.setattr(
        runtime, "_make_trainer_v61a",
        lambda *_args: (_ for _ in ()).throw(AssertionError("dry-run made trainer")),
    )
    assert runtime.main([
        "--preregistration", str(path),
        "--preregistration-sha256", runtime.file_sha256_v61a(path),
        "--preregistration-content-sha256",
        value["content_sha256_before_self_field"],
        "--dry-run",
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["train_rows_opened"] == 0
    assert output["model_or_gpu_loaded"] is False
    assert output["filesystem_writes"] is False


def test_v61a_evidence_builder_persists_numeric_hash_schema_only():
    rows = [{
        "row_index": index,
        "row_sha256": _sha(index + 1),
        "unit_identity_sha256": _sha(20_000 + (index % 208)),
        "row_count": 2 if index < 416 else 1,
        "question": f"synthetic q {index}",
        "answer": f"synthetic a {index}",
    } for index in range(subject.TRAIN_ROWS_V61A)]
    actor_metrics = [[{
        "actor_rank": actor,
        "generation_seed": subject.ACTOR_GENERATION_SEEDS_V61A[actor],
        "f1": 1.0,
        "exact": 1,
        "nonzero": 1,
    } for _row_value in rows] for actor in range(4)]
    evidence = runtime.build_evidence_v61a(
        rows, actor_metrics, {"sha256": runtime.design_v52.MASTER_SHA256_V52},
    )
    encoded = json.dumps(evidence, sort_keys=True)
    assert '"question"' not in encoded
    assert '"answer"' not in encoded
    assert "synthetic q" not in encoded
    assert "synthetic a" not in encoded
    assert evidence["candidate_selection_or_promotion_performed"] is False


def test_v61a_failure_artifact_digests_raw_error_message():
    failure = runtime._sanitize_failure_v61a(
        RuntimeError("forbidden synthetic prompt and completion")
    )
    encoded = json.dumps(failure, sort_keys=True)
    assert "forbidden synthetic prompt" not in encoded
    assert failure["raw_error_message_or_traceback_persisted"] is False
    row = _row(0, 0, [0.5] * 4)
    row["actors"][0]["output_text"] = "forbidden"
    with pytest.raises(ValueError, match="actor metric"):
        subject.classify_row_v61a(row)
