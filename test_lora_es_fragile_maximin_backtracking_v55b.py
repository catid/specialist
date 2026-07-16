import json
import math

import numpy as np

import build_lora_es_fragile_maximin_backtracking_preregistration_v55b as builder
import lora_es_fragile_maximin_projection_v55b as maximin
import lora_es_nested_population_v52 as design
import run_lora_es_fragile_maximin_backtracking_v55b as runtime
import run_lora_es_nested_population_v52 as v52


def test_v55b_closed_form_positive_dual_maximin_kkt_certificate():
    value = maximin.maximin_projection_v55b()
    assert value["objective_order"] == sorted(maximin.OBJECTIVE_PATHS_V55B)
    assert value["gram_rank"] == 6
    assert value["gram_condition"] < 10.0
    assert min(value["dual_lambda"]) > 0.1
    assert value["maximin_margin"] > 0.3
    assert value["maximum_equal_margin_residual"] < 1e-12
    assert all(
        margin > 0.3 for margin in value["objective_margins"].values()
    )
    assert max(value["objective_margins"].values()) - min(
        value["objective_margins"].values()
    ) < 1e-12
    assert math.isclose(
        np.linalg.norm(value["direction"]), 1.0,
        rel_tol=0.0, abs_tol=1e-12,
    )
    assert value["kkt_certificate"]["strong_duality_verified"] is True


def test_v55b_uses_only_supported_spread_and_keeps_three_endpoint_gates():
    value = maximin.maximin_projection_v55b()
    assert all(
        item["zero_spread"] is False
        for item in value["objective_fitness"].values()
    )
    assert set(value["zero_spread_endpoint_only_metrics"]) == {
        "fragile_generation_exact",
        "qa_generation_exact",
        "qa_generation_nonzero",
    }
    assert all(
        item["zero_spread"] is True
        for item in value["zero_spread_endpoint_only_metrics"].values()
    )


def test_v55b_changes_only_direction_and_matches_every_v54_scale_norm():
    projection = maximin.maximin_projection_v55b()
    plans = maximin.scale_plans_v55b(projection)
    assert projection["genuinely_different_from_v54"] is True
    assert projection["direction_cosine_vs_v54"] < 0.99
    assert [item["target_norm_ratio"] for item in plans] == list(
        design.SCALE_ORDER_V52
    )
    for item in plans:
        assert item[
            "coefficient_l2_norm_exactly_matches_v54_same_ratio"
        ] is True
        assert math.isclose(
            item["coefficient_l2_norm"],
            item["v54_reference_coefficient_l2_norm"],
            rel_tol=0.0, abs_tol=1e-12,
        )


def test_v55b_projection_has_barrier_no_population_and_no_p8(monkeypatch):
    sleeps = []
    monkeypatch.setattr(runtime.time, "sleep", sleeps.append)
    population = runtime.selected_maximin_projection_v55b()
    assert sleeps == [1.25]
    assert population["population_generation_or_scoring_performed"] is False
    assert population["arms"]["p8"]["evaluation_authorized"] is False
    assert population["arms"]["p16"]["projection"] == (
        maximin.maximin_projection_v55b()
    )
    original = runtime.ORIGINAL_EVALUATE
    monkeypatch.setattr(
        runtime, "ORIGINAL_EVALUATE",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("v55b p8 no-op entered candidate evaluator")
        ),
    )
    receipt = runtime.evaluate_only_p16_v55b("p8")
    assert receipt["evaluation_performed"] is False
    assert receipt["candidate_transactions_opened"] == 0
    monkeypatch.setattr(runtime, "ORIGINAL_EVALUATE", original)


def test_v55b_preregistration_binds_exact_telemetry_artifacts_and_gates():
    value = builder.build_v55b()
    assert value["artifacts"] == builder.artifacts_v55b()
    assert value["telemetry"] == {
        "gpu_log": str(runtime.GPU_LOG),
        "projection_phase": runtime.PHASE,
        "minimum_projection_phase_barrier_seconds": 1.25,
        "inherited_v52_filename_or_population_phase_allowed": False,
    }
    assert value["single_scientific_change"]["variable"] == (
        "projection_direction_only"
    )
    assert value["train_only_gate"][
        "all_nine_checks_required_without_weakening"
    ] is True
    assert value["authorization"]["population_generation_or_scoring"] is False
    assert value["authorization"]["optimizer_master_commit"] is False
    assert value["authorization"]["p8_evaluation"] is False
    with runtime.patched_executor_v55b():
        assert v52.runtime_telemetry_contract_v52() == {
            "gpu_log": runtime.GPU_LOG,
            "population_phase": runtime.PHASE,
        }


def test_v55b_dry_run_loads_bound_preregistration_without_writes(
    tmp_path, monkeypatch, capsys,
):
    value = builder.build_v55b()
    path = tmp_path / "v55b.json"
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    before = {item.name: item.stat().st_mtime_ns for item in tmp_path.iterdir()}
    monkeypatch.setattr(
        runtime, "execute_v55b",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("v55b dry-run entered live executor")
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
