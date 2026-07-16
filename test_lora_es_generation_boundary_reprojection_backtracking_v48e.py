#!/usr/bin/env python3

import inspect
import json

import pytest

import build_lora_es_generation_boundary_reprojection_backtracking_v48e as builder
import lora_es_generation_boundary_reprojection_v48e as planning
import run_lora_es_generation_boundary_reprojection_backtracking_v48e as runtime


def test_v48e_loads_exact_v48b_population_restore_and_boundary_subset():
    evidence = runtime.load_evidence_v48e()
    assert evidence["restored_master_identity"]["sha256"] == (
        planning.RESTORED_MASTER_SHA256_V48E
    )
    assert evidence["restored_runtime_values_sha256"] == (
        planning.RESTORED_RUNTIME_SHA256_V48E
    )
    assert evidence["sealed_subset"]["request_order_sha256"] == (
        planning.REQUEST_ORDER_SHA256_V48E
    )
    assert evidence["population_resampled"] is False
    assert evidence["projection_recomputed_from_frozen_signed_scores"] is True
    assert [item["target_norm_ratio"] for item in evidence["scale_plans"]] == [
        0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625,
    ]


def test_v48e_scale_grid_is_exact_positive_scaling_of_reprojected_direction():
    evidence = runtime.load_evidence_v48e()
    source = evidence["reprojection"]["coefficients"]
    hashes = []
    for plan, target, multiplier in zip(
        evidence["scale_plans"], planning.TARGET_NORM_RATIOS_V48E,
        (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125), strict=True,
    ):
        assert plan["target_norm_ratio"] == target
        assert plan["actual_norm_ratio"] == pytest.approx(target, abs=1e-12)
        assert plan["v48e_coefficient_multiplier"] == multiplier
        assert plan["coefficients"] == pytest.approx([
            value * multiplier for value in source
        ])
        assert len(plan["coefficient_sha256"]) == 64
        hashes.append(plan["coefficient_sha256"])
    assert len(set(hashes)) == 6


def test_v48e_signed_inventory_centered_ranks_and_geometry_are_exact():
    evidence = runtime.load_evidence_v48e()
    reprojection = evidence["reprojection"]
    assert reprojection["source_signed_score_inventory_sha256"] == (
        planning.EXPECTED_FULL_SIGNED_SCORE_SHA256_V48E
    )
    assert reprojection["extracted_five_objective_sha256"] == (
        planning.EXPECTED_EXTRACTED_FIVE_OBJECTIVE_SHA256_V48E
    )
    qa = reprojection["objective_fitness"]["qa_generation_f1"]
    assert qa["zero_spread"] is False
    assert qa["coefficients"] == pytest.approx([
        0.5333333333333334, -0.3333333333333333,
        0.19999999999999996, 0.4666666666666667,
        -0.06666666666666665, 0.8,
        0.06666666666666665, -0.4666666666666667,
    ])
    assert reprojection["projection"]["content_sha256"] == (
        planning.EXPECTED_REPROJECTION_CONTENT_SHA256_V48E
    )
    geometry = reprojection["geometry_vs_v48b"]
    assert geometry["coefficient_cosine_v48b_to_v48e"] == pytest.approx(
        0.7729919963550018
    )
    assert geometry["v48b_qa_generation_f1_directional_derivative"] < 0.0
    assert geometry["v48e_qa_generation_f1_directional_derivative"] == (
        pytest.approx(0.0, abs=1e-12)
    )
    assert geometry["v48e_active_anchor_names"] == [
        "prose_lm", "qa_answer_logprob", "qa_generation_f1",
    ]


def test_v48e_largest_pass_policy_requires_exact_abort_and_stops():
    results = [{
        "target_norm_ratio": 0.5,
        "gate_passed": False,
        "exact_abort_readback_passed": True,
    }, {
        "target_norm_ratio": 0.25,
        "gate_passed": True,
        "exact_abort_readback_passed": False,
    }]
    assert planning.largest_passing_scale_v48e(results) == 0.25
    with pytest.raises(ValueError, match="not exactly aborted"):
        planning.largest_passing_scale_v48e([{
            "target_norm_ratio": 0.5,
            "gate_passed": False,
            "exact_abort_readback_passed": False,
        }])
    with pytest.raises(ValueError, match="smaller scale after"):
        planning.largest_passing_scale_v48e(results + [{
            "target_norm_ratio": 0.125,
            "gate_passed": False,
            "exact_abort_readback_passed": True,
        }])


def test_v48e_all_rejected_requires_six_exact_abort_readbacks():
    results = [{
        "target_norm_ratio": ratio,
        "gate_passed": False,
        "exact_abort_readback_passed": True,
    } for ratio in planning.TARGET_NORM_RATIOS_V48E]
    assert planning.largest_passing_scale_v48e(results) is None


def test_v48e_uses_v48b_fragile_path_and_all_nine_gates():
    required = {
        "domain_point_improvement", "prose_lm_noninferiority",
        "qa_logprob_noninferiority", "qa_generation_f1_noninferiority",
        "qa_generation_exact_noninferiority",
        "qa_generation_nonzero_noninferiority",
        "fragile_generation_f1_noninferiority",
        "fragile_generation_exact_noninferiority",
        "fragile_generation_nonzero_noninferiority",
    }
    value = builder.build_v48e()
    assert set(value["candidate_gate"]["required_checks"]) == required
    assert value["candidate_gate"][
        "fragile_subset_request_order_sha256"
    ] == planning.REQUEST_ORDER_SHA256_V48E
    original_gate = runtime.prior.fused.candidate_gate_v43i
    original_fused = runtime.prior.fused.fused_requests_v43i
    with runtime.v48b.patched_v43i_v48b():
        assert runtime.prior.fused.candidate_gate_v43i is (
            runtime.v48b.candidate_gate_v48b
        )
        assert runtime.prior.fused.fused_requests_v43i is (
            runtime.v48b.fused_requests_v48b
        )
    assert runtime.prior.fused.candidate_gate_v43i is original_gate
    assert runtime.prior.fused.fused_requests_v43i is original_fused


def test_v48e_population_is_608_but_full_candidate_is_896_with_same_fragile_order(
    monkeypatch,
):
    fragile = [{"prompt_token_ids": [9000 + index]}
               for index in range(64)]
    monkeypatch.setattr(runtime.v48b, "_PREPARED_FRAGILE", fragile)
    domain = [{"prompt_token_ids": [index]} for index in range(448)]

    def anchors(documents):
        return {
            "documents": documents,
            "prose": [{"prompt_token_ids": [1000 + index]}
                      for index in range(documents)],
            "qa_teacher": [{"prompt_token_ids": [2000 + index]}
                           for index in range(documents)],
            "qa_generation": [{"prompt_token_ids": [3000 + index]}
                              for index in range(documents)],
        }

    panel = runtime.v48b.fused_requests_v48b(domain, anchors(32))
    full = runtime.v48b.fused_requests_v48b(domain, anchors(128))
    assert len(panel["requests"]) == 608
    assert panel["slices"] == {
        "domain": [0, 448], "prose": [448, 480],
        "qa_teacher": [480, 512], "qa_generation": [512, 544],
        "fragile_generation": [544, 608],
    }
    assert len(full["requests"]) == 896
    assert full["slices"] == {
        "domain": [0, 448], "prose": [448, 576],
        "qa_teacher": [576, 704], "qa_generation": [704, 832],
        "fragile_generation": [832, 896],
    }
    expected_fragile = [item["prompt_token_ids"] for item in fragile]
    assert [item["prompt_token_ids"] for item in panel["requests"][544:]] == (
        expected_fragile
    )
    assert [item["prompt_token_ids"] for item in full["requests"][832:]] == (
        expected_fragile
    )


def test_v48e_builder_loader_and_dry_run_are_zero_write_and_protected_free(
    tmp_path, monkeypatch, capsys,
):
    forbidden = ("shadow_dev", "eval_qa", "ood_qa", "ood_prose",
                 "holdout", "heldout", "benchmark")
    original_hash = runtime.v40a.file_sha256

    def guarded(path):
        assert not any(token in str(path).casefold() for token in forbidden)
        return original_hash(path)

    monkeypatch.setattr(runtime.v40a, "file_sha256", guarded)
    value = builder.build_v48e()
    assert value["gpu_launch_authorized"] is True
    assert value["protected_semantic_access_authorized"] is False
    assert value["recipe"]["resample_population"] is False
    assert value["recipe"]["recompute_population_scores"] is False
    assert value["recipe"][
        "recompute_projection_from_frozen_signed_scores"
    ] is True
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    file_sha = original_hash(path)
    args = type("Args", (), {
        "preregistration": str(path),
        "preregistration_sha256": file_sha,
        "preregistration_content_sha256": value[
            "content_sha256_before_self_field"
        ],
    })()
    loaded, evidence = runtime.load_preregistration_v48e(args)
    assert loaded["v48b_evidence"] == evidence
    assert runtime.main([
        "--preregistration", str(path),
        "--preregistration-sha256", file_sha,
        "--preregistration-content-sha256",
        value["content_sha256_before_self_field"],
        "--dry-run",
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["population_resampled"] is False
    assert output["population_scores_recomputed"] is False
    assert output["projection_recomputed_from_frozen_signed_scores"] is True
    assert output["new_population_generation"] is False
    assert output["model_or_gpu_loaded"] is False
    assert output["train_semantics_loaded"] is False
    assert output["protected_semantic_access_count"] == 0
    assert output["filesystem_writes"] is False


def test_v48e_runtime_reuses_population_and_is_transactional():
    source = inspect.getsource(runtime.main)
    assert "_replicated_population" not in source
    assert "_calibrate_numeric_path" not in source
    assert "_calibrate_anchor_path" not in source
    assert "patched_v43i_v48b" in source
    assert "candidate_gate_v43i" in source
    assert source.count('"fragile_request_order_sha256"') >= 2
    assert source.count('"sealed_subset_content_sha256"') >= 2
    assert "_exact_abort_transaction" in source
    assert source.index("_exact_abort_transaction") < source.index("continue")
    assert "_commit_accept_update" in source
    assert "_seal_accepted_update" in source
