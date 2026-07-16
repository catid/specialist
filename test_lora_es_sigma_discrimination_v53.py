import copy

import pytest

import lora_es_sigma_discrimination_v53 as design
import run_lora_es_sigma_discrimination_v53 as runtime


def test_v53_freezes_fixed_p16_seeds_and_three_ascending_sigmas():
    assert design.SIGMAS_V53 == (0.0012, 0.0024, 0.0048)
    assert len(design.SEEDS_V53) == len(set(design.SEEDS_V53)) == 16
    for sigma in design.SIGMAS_V53:
        states = design.state_derivations_v53(sigma)
        assert len(states) == 32
        assert [row["state_index"] for row in states] == list(range(32))
        assert [(row["direction"], row["sign"]) for row in states] == [
            (direction, sign) for direction in range(16) for sign in (1, -1)
        ]
        assert {row["sigma"] for row in states} == {sigma}


def test_v53_adaptive_selection_stops_at_first_pass():
    assert design.select_smallest_passing_sigma_v53([]) is None
    failed = [{"sigma": 0.0012, "population_size": 16, "passed": False}]
    assert design.select_smallest_passing_sigma_v53(failed) is None
    passed = failed + [{"sigma": 0.0024, "population_size": 16, "passed": True}]
    assert design.select_smallest_passing_sigma_v53(passed) == 0.0024
    with pytest.raises(ValueError):
        design.select_smallest_passing_sigma_v53(
            passed + [{"sigma": 0.0048, "population_size": 16, "passed": False}]
        )


def test_v53_inherits_unchanged_reliability_gates():
    stable = [[float(index) * 0.01 + delta for delta in (0.0, 0.00001, -0.00001, 0.0)] for index in range(16)]
    result = design.reliability_gate_v53(stable, 0.001)
    assert result["passed"] is True
    assert result["minimum_reliability"] == 0.8
    assert result["minimum_split_half_spearman"] == 0.7
    assert result["estimated_signal_standard_deviation_clears_fresh_calibration_maximum"] is True


def test_v53_mandatory_stop_forbids_projection_update_and_train_gate():
    population = {
        "schema": "adaptive-sigma-discrimination-population-v53",
        "all_completed_arms_persisted_before_stop": True,
        "projection_performed": False,
        "optimizer_update_or_train_gate_opened": False,
    }
    with pytest.raises(runtime.V53MeasurementComplete):
        runtime.require_measurement_stop(population)
    changed = copy.deepcopy(population)
    changed["projection_performed"] = True
    with pytest.raises(RuntimeError) as caught:
        runtime.require_measurement_stop(changed)
    assert not isinstance(caught.value, runtime.V53MeasurementComplete)
