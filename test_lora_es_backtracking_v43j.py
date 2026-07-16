import json
import inspect

import pytest

import build_lora_es_backtracking_preregistration_v43j as builder
import lora_es_backtracking_v43j as planning
import run_lora_es_backtracking_v43j as runtime


def test_v43i_negative_evidence_and_exact_restore_are_sealed_v43j():
    evidence = planning.load_v43i_evidence_v43j(runtime.V43I_EVIDENCE_PATHS)
    assert evidence["restored_master_identity"]["sha256"] == (
        planning.RESTORED_MASTER_SHA256_V43J
    )
    assert evidence["restored_runtime_values_sha256"] == (
        planning.RESTORED_RUNTIME_SHA256_V43J
    )
    assert [item["target_norm_ratio"] for item in evidence["scale_plans"]] == [
        0.25, 0.125, 0.0625,
    ]
    assert evidence["protected_semantics_opened"] is False


def test_scale_grid_is_exact_positive_homogeneous_backtracking_v43j():
    evidence = planning.load_v43i_evidence_v43j(runtime.V43I_EVIDENCE_PATHS)
    source = evidence["v43i_projected_coefficients"]
    expected_hashes = [
        "b02021bc840bb988be52ded8c6fbe971f117590c748edb9edc5d2085b88c3040",
        "e02fad377a7220538a6ee63589221c66f3fff6314ba7edfa8b67a90d893ed533",
        "664816234bc7f30eebc40e25ece17764648d2c247788395c27a8865bed69106f",
    ]
    for plan, target, multiplier, expected_hash in zip(
        evidence["scale_plans"], (0.25, 0.125, 0.0625),
        (0.5, 0.25, 0.125), expected_hashes, strict=True,
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


def test_largest_passing_policy_requires_abort_and_stops_at_first_pass_v43j():
    results = [{
        "target_norm_ratio": 0.25,
        "gate_passed": False,
        "exact_abort_readback_passed": True,
    }, {
        "target_norm_ratio": 0.125,
        "gate_passed": True,
        "exact_abort_readback_passed": False,
    }]
    assert planning.largest_passing_scale_v43j(results) == 0.125
    with pytest.raises(ValueError, match="not exactly aborted"):
        planning.largest_passing_scale_v43j([{
            "target_norm_ratio": 0.25,
            "gate_passed": False,
            "exact_abort_readback_passed": False,
        }])
    with pytest.raises(ValueError, match="smaller scale after"):
        planning.largest_passing_scale_v43j(results + [{
            "target_norm_ratio": 0.0625,
            "gate_passed": False,
            "exact_abort_readback_passed": True,
        }])


def test_no_scale_passed_requires_three_exact_abort_readbacks_v43j():
    results = [{
        "target_norm_ratio": ratio,
        "gate_passed": False,
        "exact_abort_readback_passed": True,
    } for ratio in planning.TARGET_NORM_RATIOS_V43J]
    assert planning.largest_passing_scale_v43j(results) is None


def test_builder_and_loader_bind_train_only_v43i_evidence_v43j(
    tmp_path, monkeypatch,
):
    protected_tokens = ("shadow_dev", "eval_qa", "ood_qa", "ood_prose",
                        "holdout", "heldout")
    original = runtime.v40a.file_sha256

    def guarded(path):
        assert not any(token in str(path).casefold() for token in protected_tokens)
        return original(path)

    monkeypatch.setattr(runtime.v40a, "file_sha256", guarded)
    value = builder.build_v43j()
    assert value["gpu_launch_authorized"] is True
    assert value["sealed_holdout_opened"] is False
    assert value["protected_semantic_access"] is False
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
    loaded, evidence = runtime.load_preregistration_v43j(args)
    assert loaded["v43i_evidence"] == evidence


def test_dry_run_loads_only_sealed_numeric_evidence_v43j(tmp_path, capsys):
    value = builder.build_v43j()
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


def test_tampered_scale_order_fails_preregistration_closed_v43j(tmp_path):
    value = builder.build_v43j()
    value["recipe"]["scale_order"] = [0.125, 0.25, 0.0625]
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
        runtime.load_preregistration_v43j(args)


def test_runtime_reuses_population_and_keeps_every_scale_transactional_v43j():
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
