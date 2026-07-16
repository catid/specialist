import json

import pytest

import build_lora_es_selected_score_backtracking_preregistration_v54 as builder
import lora_es_nested_population_v52 as design
import run_lora_es_nested_population_v52 as v52
import run_lora_es_selected_score_backtracking_v54 as runtime


def test_v54_reuses_exact_selected_v53_scores_without_population_rerun():
    projection, plans = builder.selected_projection_v54()
    assert projection["population_size"] == 16
    assert list(projection["objective_fitness"]) == list(
        design.OBJECTIVE_PATHS_V52
    )
    assert [plan["target_norm_ratio"] for plan in plans] == list(
        design.SCALE_ORDER_V52
    )
    population = runtime.selected_score_projection_v54()
    assert population["population_generation_or_scoring_performed"] is False
    assert population["arms"]["p8"]["evaluation_authorized"] is False
    assert population["arms"]["p16"]["projection"] == projection
    assert population["arms"]["p16"]["scale_plans"] == plans


def test_v54_p8_compatibility_receipt_never_calls_candidate_evaluator(
    monkeypatch,
):
    monkeypatch.setattr(
        runtime, "ORIGINAL_EVALUATE",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("V54 p8 no-op called candidate evaluator")
        ),
    )
    result = runtime.evaluate_only_selected_p16_v54("p8")
    assert result["evaluation_performed"] is False
    assert result["candidate_transactions_opened"] == 0


def test_v54_exact_telemetry_path_phase_and_artifact_contract():
    value = builder.build_v54()
    assert value["artifacts"] == builder.artifacts_v54()
    assert value["artifacts"]["gpu_log"] == str(runtime.GPU_LOG)
    assert value["telemetry"] == {
        "gpu_log": str(runtime.GPU_LOG),
        "projection_phase": runtime.PHASE,
        "inherited_v52_filename_or_population_phase_allowed": False,
    }
    with runtime.patched_executor_v54():
        assert v52.runtime_telemetry_contract_v52() == {
            "gpu_log": runtime.GPU_LOG,
            "population_phase": runtime.PHASE,
        }


def test_v54_dry_run_loads_bound_preregistration_without_writes(
    tmp_path, monkeypatch, capsys,
):
    value = builder.build_v54()
    path = tmp_path / "v54.json"
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    before = {item.name: item.stat().st_mtime_ns for item in tmp_path.iterdir()}
    monkeypatch.setattr(
        runtime, "execute_v54",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("V54 dry-run entered live executor")
        ),
    )
    assert runtime.main([
        "--preregistration", str(path),
        "--preregistration-sha256", builder.file_sha256(path),
        "--preregistration-content-sha256",
        value["content_sha256_before_self_field"],
        "--dry-run",
    ]) == 0
    assert before == {
        item.name: item.stat().st_mtime_ns for item in tmp_path.iterdir()
    }
    output = json.loads(capsys.readouterr().out)
    assert output["population_rerun"] is False
    assert output["p8_evaluation"] is False
    assert output["filesystem_writes"] is False
    assert output["protected_semantics_loaded"] is False


def test_v54_rejects_population_or_master_commit_authorization():
    value = builder.build_v54()
    assert value["authorization"]["population_generation_or_scoring"] is False
    assert value["authorization"]["optimizer_master_commit"] is False
    assert value["authorization"]["p8_evaluation"] is False
    assert value["selected_v53_input"]["p16_seeds"] == list(
        design.P16_SEEDS_V52
    )
    assert value["selected_v53_input"]["optimizer_update_alpha"] == (
        design.ALPHA_V52
    )
    assert value["train_only_gate"]["every_in_memory_candidate_exactly_aborted_to_master"] is True
