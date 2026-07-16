import json
import math

import pytest

import build_lora_es_robust_equal_unit_preregistration_v43g as builder
import lora_es_robust_consensus_v43g as numeric
import run_lora_es_robust_equal_unit_v43g as runtime


def _parent(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_complete_actor_assignment_covers_every_actor_once_v43g():
    for direction in range(8):
        assignment = numeric.complete_actor_assignments_v43g(direction)
        assert [item["actor_rank"] for item in assignment] == [0, 1, 2, 3]
        assert [item["replicate"] for item in assignment] == [0, 1, 2, 3]
        assert {item["direction_index"] for item in assignment} == {direction}


def test_robust_population_uses_median_signed_centered_ranks_v43g():
    plus = []
    minus = []
    for direction in range(8):
        center = (direction - 3.5) * 0.001
        plus.append([center, center, center, center + 100.0])
        minus.append([-center, -center, -center, -center - 100.0])
    result = numeric.robust_population_v43g({"plus": plus, "minus": minus})
    assert result["robust_signed_scores"]["plus"] == pytest.approx(
        [(direction - 3.5) * 0.001 for direction in range(8)]
    )
    assert result["robust_signed_scores"]["minus"] == pytest.approx(
        [-(direction - 3.5) * 0.001 for direction in range(8)]
    )
    assert result["coefficients"] == sorted(result["coefficients"])
    assert result["zero_utility_update"] is False


def test_complete_actor_reliability_gate_passes_high_snr_and_fails_ceiling_v43g():
    signal = [-0.004, -0.003, -0.002, -0.001, 0.001, 0.002, 0.003, 0.004]
    noise = [-0.00003, 0.00002, -0.00001, 0.000025]
    central = [[value + item for item in noise] for value in signal]
    passed = numeric.reliability_gate_v43g(central, 0.0005)
    assert passed["passed"] is True
    assert passed["reliability"] > 0.99
    assert passed["split_half_spearman"] == pytest.approx(1.0)
    failed = numeric.reliability_gate_v43g(central, 0.002)
    assert failed["passed"] is False
    assert failed["fresh_calibration_inside_historical_ceiling"] is False


def test_v43f_train_only_diagnostic_is_exact_and_motivates_v43g():
    diagnostic = numeric.diagnose_v43f_artifacts_v43g(
        _parent(builder.PARENT_V43F_CALIBRATION),
        _parent(builder.PARENT_V43F_RELIABILITY),
    )
    assert diagnostic["v43f_reliability"] == 0.6581937522502567
    assert diagnostic["v43f_calibration_bound_fraction"] == 4.5349602966089995
    assert diagnostic["largest_replicate_disagreement_direction_indices"] == [3, 4, 7]
    assert diagnostic["leave_direction_3_out_reliability"] == pytest.approx(
        0.8054091497119735
    )
    assert diagnostic["mean_calibration_repeat_actor_pattern_correlation"] == pytest.approx(
        0.02031403774515431
    )
    assert diagnostic["linear_response_projected_v43g_reliability"] == pytest.approx(
        0.9390432439905053
    )
    assert diagnostic["interpretation"]["exact_state_failure_observed"] is False


def test_builder_and_dry_run_bind_train_only_parents_without_semantic_access_v43g(
    tmp_path, capsys,
):
    output = tmp_path / "v43g.json"
    assert builder.main(["--output", str(output)]) == 0
    built_stdout = capsys.readouterr().out
    assert json.loads(built_stdout)["path"] == str(output)
    prereg = json.loads(output.read_text(encoding="utf-8"))
    assert prereg["sealed_holdout_opened"] is False
    assert prereg["access_contract"]["protected_semantic_access"] is False
    assert prereg["recipe"]["sigma"] == 0.0006
    assert prereg["recipe"]["alpha"] == 0.00015
    assert prereg["recipe"]["signed_replicates_per_direction"] == 4
    assert set(prereg["parents"]) >= {
        "v43f_failure", "v43f_numeric_calibration", "v43f_population_reliability",
    }
    assert runtime.main([
        "--preregistration", str(output),
        "--preregistration-sha256", runtime.v40a.file_sha256(output),
        "--preregistration-content-sha256",
        prereg["content_sha256_before_self_field"],
        "--dry-run",
    ]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["protected_semantic_access"] is False
    assert dry["protected_paths_opened"] == []
    assert dry["train_dataset_content_hashed_for_binding"] is True
    assert dry["train_dataset_semantics_loaded"] is False
    assert dry["model_metadata_hashed_for_binding"] is True
    assert dry["model_runtime_loaded"] is False
    assert dry["gpu_launched"] is False
    assert dry["filesystem_writes"] is False


def test_planning_projection_matches_v43f_variance_components_v43g():
    value = numeric.predicted_reliability_v43g(
        1.0393607640691642e-07,
        1.0794997722485564e-07,
    )
    assert math.isclose(value, 0.9390432439905053, rel_tol=0.0, abs_tol=1e-15)
