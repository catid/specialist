import json
import inspect

import pytest

import build_lora_es_backtracking_preregistration_v43l as builder
import lora_es_backtracking_v43l as planning
import run_lora_es_backtracking_v43l as runtime


def test_v43i_negative_evidence_and_exact_restore_are_sealed_v43l():
    evidence = planning.load_v43i_evidence_v43l(runtime.V43I_EVIDENCE_PATHS)
    assert evidence["restored_master_identity"]["sha256"] == (
        planning.RESTORED_MASTER_SHA256_V43L
    )
    assert evidence["restored_runtime_values_sha256"] == (
        planning.RESTORED_RUNTIME_SHA256_V43L
    )
    assert [item["target_norm_ratio"] for item in evidence["scale_plans"]] == [
        0.03125, 0.015625,
    ]
    assert evidence["protected_semantics_opened"] is False
    predecessor = runtime.v43k_train_only_evidence_v43l()
    assert predecessor["ratios_evaluated_by_v43k"] == [0.125, 0.0625]
    assert predecessor["ratios_untried_before_v43l"] == [0.03125, 0.015625]
    assert set(predecessor["per_scale_gate_and_abort_seals"]) == {
        "0p125", "0p0625",
    }
    assert predecessor["holdout_or_heldout_opened"] is False


def test_scale_grid_is_exact_positive_homogeneous_backtracking_v43l():
    evidence = planning.load_v43i_evidence_v43l(runtime.V43I_EVIDENCE_PATHS)
    source = evidence["v43i_projected_coefficients"]
    expected_hashes = [
        "d9b17c56445f8d15ad8834ec2fca946ccea83dbd8e5acde06c9c6d47f08eacda",
        "a6bc4aecae443cfb204c9c7fa654398bb85b6bef60efaf61687a7fd108155218",
    ]
    for plan, target, multiplier, expected_hash in zip(
        evidence["scale_plans"], (0.03125, 0.015625),
        (0.0625, 0.03125), expected_hashes, strict=True,
    ):
        assert plan["target_norm_ratio"] == target
        assert plan["actual_norm_ratio"] == target
        assert plan["v43i_coefficient_multiplier"] == multiplier
        assert plan["coefficients"] == pytest.approx([
            value * multiplier for value in source
        ])
        assert plan["coefficient_sha256"] == expected_hash
        assert plan[
            "anchor_halfspaces_preserved_by_positive_homogeneous_scaling"
        ] is True


def test_six_gate_policy_opens_smaller_scale_only_after_gate_failure_v43l():
    results = [{
        "target_norm_ratio": 0.03125,
        "six_train_only_gates_passed": False,
        "candidate_consensus_passed": None,
        "exact_abort_readback_passed": True,
    }, {
        "target_norm_ratio": 0.015625,
        "six_train_only_gates_passed": True,
        "candidate_consensus_passed": True,
        "exact_abort_readback_passed": False,
    }]
    assert planning.selected_diagnostic_scale_v43l(results) == 0.015625
    with pytest.raises(ValueError, match="not exactly aborted"):
        planning.selected_diagnostic_scale_v43l([{
            "target_norm_ratio": 0.03125,
            "six_train_only_gates_passed": False,
            "candidate_consensus_passed": None,
            "exact_abort_readback_passed": False,
        }])
    consensus_failure = [{
        "target_norm_ratio": 0.03125,
        "six_train_only_gates_passed": True,
        "candidate_consensus_passed": False,
        "exact_abort_readback_passed": True,
    }]
    assert planning.selected_diagnostic_scale_v43l(consensus_failure) is None
    with pytest.raises(ValueError, match="smaller scale after six-gate pass"):
        planning.selected_diagnostic_scale_v43l(consensus_failure + [{
            "target_norm_ratio": 0.015625,
            "six_train_only_gates_passed": False,
            "candidate_consensus_passed": None,
            "exact_abort_readback_passed": True,
        }])


def test_no_scale_passed_requires_two_exact_abort_readbacks_v43l():
    results = [{
        "target_norm_ratio": ratio,
        "six_train_only_gates_passed": False,
        "candidate_consensus_passed": None,
        "exact_abort_readback_passed": True,
    } for ratio in planning.TARGET_NORM_RATIOS_V43L]
    assert planning.selected_diagnostic_scale_v43l(results) is None


def test_builder_and_loader_bind_train_only_v43i_evidence_v43l(
    tmp_path, monkeypatch,
):
    protected_tokens = ("shadow_dev", "eval_qa", "ood_qa", "ood_prose",
                        "holdout", "heldout")
    original = runtime.v40a.file_sha256

    def guarded(path):
        assert not any(token in str(path).casefold() for token in protected_tokens)
        return original(path)

    monkeypatch.setattr(runtime.v40a, "file_sha256", guarded)
    value = builder.build_v43l()
    assert value["gpu_launch_authorized"] is True
    assert value["sealed_holdout_opened"] is False
    assert value["protected_semantic_access"] is False
    assert value["current_v42i_holdout_cycle_eligible"] is False
    assert value[
        "current_fixed_holdout_cycle_result_may_be_used_for_tuning"
    ] is False
    assert value["access_contract"][
        "current_fixed_holdout_cycle_result_influenced_design"
    ] is False
    assert value["recipe"]["resample_population"] is False
    assert value["recipe"]["recompute_projection"] is False
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    args = type("Args", (), {
        "preregistration": str(path),
        "preregistration_sha256": original(path),
        "preregistration_content_sha256": value[
            "content_sha256_before_self_field"
        ],
    })()
    loaded, evidence = runtime.load_preregistration_v43l(args)
    assert loaded["v43i_evidence"] == evidence


def test_dry_run_loads_only_sealed_numeric_evidence_v43l(tmp_path, capsys):
    value = builder.build_v43l()
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    result = runtime.main([
        "--preregistration", str(path),
        "--preregistration-sha256", runtime.v40a.file_sha256(path),
        "--preregistration-content-sha256",
        value["content_sha256_before_self_field"],
        "--dry-run",
    ])
    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output["v43i_evidence_loaded"] is True
    assert output["population_resampled"] is False
    assert output["projection_recomputed"] is False
    assert output["gpu_launched"] is False
    assert output["model_runtime_loaded"] is False
    assert output["protected_paths_opened"] == []
    assert output["current_v42i_holdout_cycle_eligible"] is False


def test_tampered_scale_order_fails_preregistration_closed_v43l(tmp_path):
    value = builder.build_v43l()
    value["recipe"]["scale_order"] = [0.015625, 0.03125]
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    value["content_sha256_before_self_field"] = runtime.v40a.canonical_sha256(
        compact
    )
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    args = type("Args", (), {
        "preregistration": str(path),
        "preregistration_sha256": runtime.v40a.file_sha256(path),
        "preregistration_content_sha256": value[
            "content_sha256_before_self_field"
        ],
    })()
    with pytest.raises(RuntimeError, match="preregistration contract changed"):
        runtime.load_preregistration_v43l(args)


def test_runtime_reuses_population_and_keeps_every_scale_transactional_v43l():
    source = inspect.getsource(runtime.main)
    assert "_replicated_population" not in source
    assert "_calibrate_numeric_path" not in source
    assert "_calibrate_anchor_path" not in source
    assert "_generate_fused_actor_scores" in source
    assert "candidate_gate_v43i" in source
    assert "_exact_abort_transaction" in source
    assert source.index("_exact_abort_transaction") < source.index("continue")
    assert "_commit_accept_update" in source
    assert "_seal_accepted_update" in source
    assert "if six_gate_passed" in source
    assert source.index("if six_gate_passed") < source.index("continue")
