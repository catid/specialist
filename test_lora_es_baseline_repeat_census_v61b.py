#!/usr/bin/env python3

import copy
import json
import types

import pytest

import build_lora_es_baseline_repeat_census_preregistration_v61b as builder
import lora_es_baseline_repeat_census_v61b as subject
import run_lora_es_baseline_repeat_census_v61b as runtime


def _sha(value: int) -> str:
    return f"{value:064x}"


def _metric(actor: int, pass_index: int, f1: float) -> dict:
    return {
        "actor_rank": actor,
        "pass_index": pass_index,
        "generation_seed": subject.COMMON_GENERATION_SEED_V61B,
        "f1": f1,
        "exact": int(f1 == 1.0),
        "nonzero": int(f1 > 0.0),
    }


def _evidence() -> dict:
    rows = []
    counts = {unit: 0 for unit in range(208)}
    assignments = list(range(208)) + [index % 208 for index in range(240)]
    for unit in assignments: counts[unit] += 1
    for index, unit in enumerate(assignments):
        rows.append({
            "row_index": index,
            "row_sha256": _sha(index + 1),
            "unit_identity_sha256": _sha(10_000 + unit),
            "row_count": counts[unit],
            "passes": [{
                "pass_index": pass_index,
                "actors": [_metric(actor, pass_index, 0.5) for actor in range(4)],
            } for pass_index in range(2)],
        })
    value = {
        "schema": "v61b-v434-common-seed-repeat-census-evidence",
        "status": "complete_characterization_only",
        "row_count": 448, "conflict_unit_count": 208,
        "actor_count": 4, "pass_count": 2,
        "common_generation_seed": subject.COMMON_GENERATION_SEED_V61B,
        "generation_params_without_seed": dict(
            subject.GENERATION_PARAMS_WITHOUT_SEED_V61B
        ),
        "rows": rows,
        "raw_question_answer_or_generation_text_persisted": False,
        "selection_update_or_promotion_performed": False,
        "eval_ood_shadow_or_holdout_opened": False,
        "content_sha256_before_self_field": _sha(999_999),
    }
    return value


def test_v61b_identical_common_seed_repeats_have_zero_disagreement():
    result = subject.build_repeat_analysis_v61b(_evidence())
    within = result["within_actor_pass_repeat"]
    assert within["all_actor_row_comparisons"]["mean_absolute_f1_delta"] == 0.0
    assert all(item["exact_label_disagreement_rows"] == 0 for item in within["actors"])
    for pass_value in result["cross_actor_same_seed_by_pass"]:
        assert pass_value["maximum_absolute_f1_delta"] == 0.0
        assert pass_value["rows_with_exact_label_disagreement"] == 0
    assert result["v61a_distinct_seed_bound_aggregate"] == subject.V61A_BOUND_AGGREGATES


def test_v61b_freezes_continuous_thresholds_and_separates_repeat_from_actor():
    evidence = _evidence()
    # One within-actor repeat delta of .25; a separate cross-actor delta of .20.
    evidence["rows"][0]["passes"][1]["actors"][0]["f1"] = 0.75
    evidence["rows"][1]["passes"][0]["actors"][1]["f1"] = 0.70
    result = subject.build_repeat_analysis_v61b(evidence)
    actor0 = result["within_actor_pass_repeat"]["actors"][0]
    assert actor0["f1_absolute_delta_gt_counts"] == {
        "1e-12": 1, "0.01": 1, "0.05": 1, "0.1": 1, "0.25": 0,
    }
    pass0 = result["cross_actor_same_seed_by_pass"][0]
    assert pass0["f1_absolute_delta_gt_counts"]["0.1"] == 1
    assert result["f1_absolute_delta_thresholds"] == [1e-12, .01, .05, .10, .25]


def test_v61b_rejects_seed_or_text_schema_drift():
    evidence = _evidence()
    evidence["rows"][0]["passes"][0]["actors"][0]["generation_seed"] += 1
    with pytest.raises(ValueError, match="actor/pass metric"):
        subject.build_repeat_analysis_v61b(evidence)
    evidence = _evidence()
    evidence["rows"][0]["passes"][0]["actors"][0]["output_text"] = "forbidden"
    with pytest.raises(ValueError, match="actor/pass metric"):
        subject.build_repeat_analysis_v61b(evidence)


def test_v61b_preregistration_and_dry_run_open_nothing_live(
    tmp_path, monkeypatch, capsys,
):
    value = builder.build_v61b()
    assert value["fixed_census_recipe"]["total_completions"] == 3584
    assert value["fixed_census_recipe"]["strict_pass_order"] == [0, 1]
    assert value["access_contract"]["builder_or_runtime_opens_v61a_row_level_evidence"] is False
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    args = types.SimpleNamespace(
        preregistration=str(path),
        preregistration_sha256=runtime.runtime_v61a.file_sha256_v61a(path),
        preregistration_content_sha256=value["content_sha256_before_self_field"],
    )
    assert runtime.load_preregistration_v61b(args) == value
    monkeypatch.setattr(
        runtime.runtime_v61a, "load_train_inputs_v61a",
        lambda: (_ for _ in ()).throw(AssertionError("dry-run opened train")),
    )
    assert runtime.main([
        "--preregistration", str(path),
        "--preregistration-sha256", args.preregistration_sha256,
        "--preregistration-content-sha256", args.preregistration_content_sha256,
        "--dry-run",
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["train_rows_opened"] == 0
    assert output["v61a_row_evidence_opened"] is False
    assert output["model_or_gpu_loaded"] is False
    assert output["filesystem_writes"] is False


def test_v61b_evidence_builder_persists_only_hashes_and_numeric_outcomes():
    base = _evidence()
    rows = [{
        "row_index": row["row_index"],
        "row_sha256": row["row_sha256"],
        "unit_identity_sha256": row["unit_identity_sha256"],
        "row_count": row["row_count"],
        "question": "synthetic forbidden question",
        "answer": "synthetic forbidden answer",
    } for row in base["rows"]]
    metrics = [[
        [copy.deepcopy(base["rows"][index]["passes"][pass_index]["actors"][actor])
         for index in range(448)]
        for actor in range(4)
    ] for pass_index in range(2)]
    evidence = runtime.build_evidence_v61b(
        rows, metrics, {"sha256": runtime.design_v52.MASTER_SHA256_V52},
    )
    encoded = json.dumps(evidence, sort_keys=True)
    assert "synthetic forbidden" not in encoded
    assert '"question"' not in encoded and '"answer"' not in encoded
    assert evidence["numeric_actor_pass_manifest_sha256"]
