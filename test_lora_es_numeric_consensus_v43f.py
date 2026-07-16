import argparse
import json
import math

import pytest

import build_lora_es_equal_unit_preregistration_v43f as builder
import lora_es_numeric_consensus_v43f as numeric
import run_lora_es_equal_unit_v43f as runtime


def _records_v43f(offsets, *, repeats=8, units=16, jitter=1e-7):
    records = []
    for repeat in range(repeats):
        actors = []
        for actor, offset in enumerate(offsets):
            unit_records = []
            for unit in range(units):
                unit_records.append({
                    "unit_identity_sha256": f"unit-{unit:03d}",
                    "row_count": 1 + unit % 3,
                    "mean_answer_token_logprob": (
                        -2.0 + unit * 0.01 + offset + (repeat % 2) * jitter
                    ),
                })
            actors.append({
                "actor_rank": actor,
                "aggregate": {"dense_result_sha256": f"dense-{repeat}-{actor}"},
                "units": unit_records,
            })
        records.append({"repeat_index": repeat, "actors": actors})
    return records


def test_calibration_sigma_and_balanced_assignment_v43f():
    assert runtime.CALIBRATION_SIGMA == runtime.ALPHA / math.sqrt(8)
    assert numeric.calibration_sigma_v43f(0.00015, 8) == runtime.CALIBRATION_SIGMA
    assignments = numeric.balanced_cyclic_assignments_v43f(4)
    assert len(assignments) == 2
    assert all({item["actor_rank"] for item in group} == {0, 1, 2, 3}
               for group in assignments)
    for direction in range(4, 8):
        selected = [
            item for group in assignments for item in group
            if item["direction_index"] == direction
        ]
        assert len(selected) == 2
        assert len({item["actor_rank"] for item in selected}) == 2
    with pytest.raises(ValueError):
        numeric.balanced_cyclic_assignments_v43f(2)


def test_calibration_bootstrap_is_deterministic_and_simultaneous_v43f():
    records = _records_v43f([0.0, 2e-5, -1e-5, 1e-5])
    first = numeric.calibration_bootstrap_bounds_v43f(
        records, resamples=128, seed=17, batch_size=16,
    )
    second = numeric.calibration_bootstrap_bounds_v43f(
        records, resamples=128, seed=17, batch_size=31,
    )
    assert first == second
    assert first["familywise_confidence"] == 0.99
    assert first["per_metric_quantile"] == 0.995
    for metric in ("equal_unit_mean", "unweighted_row_mean"):
        bound = first["bounds"][metric]
        assert bound["upper_actor_spread"] >= (
            bound["observed_maximum_repeat_actor_spread"] - 1e-14
        )


def test_reliability_gate_passes_signal_and_rejects_noise_v43f():
    reliable = [
        [direction * 0.01, direction * 0.01 + 1e-5]
        for direction in range(8)
    ]
    passed = numeric.reliability_gate_v43f(reliable, 1e-5)
    assert passed["passed"]
    assert passed["reliability"] >= 0.8
    noisy = [
        [direction * 0.001 + 0.01, direction * 0.001 - 0.01]
        for direction in range(8)
    ]
    failed = numeric.reliability_gate_v43f(noisy, 1e-5)
    assert not failed["passed"]
    assert failed["reliability"] < 0.8


def test_post_update_pairwise_ci_and_spread_gates_v43f():
    records = _records_v43f([0.0, 1e-5, -1e-5, 2e-5])
    generous = {
        "equal_unit_mean": {"upper_actor_spread": 1e-3},
        "unweighted_row_mean": {"upper_actor_spread": 1e-3},
    }
    passed = numeric.post_update_consensus_v43f(
        records, generous, resamples=256, seed=23, batch_size=32,
    )
    assert passed["passed"]
    assert passed["comparison_family_count"] == 12
    assert passed["all_pairwise_intervals_inside_calibrated_margins"]
    tight = {
        "equal_unit_mean": {"upper_actor_spread": 1e-7},
        "unweighted_row_mean": {"upper_actor_spread": 1e-7},
    }
    failed = numeric.post_update_consensus_v43f(
        records, tight, resamples=256, seed=23, batch_size=32,
    )
    assert not failed["passed"]
    assert not failed["all_actor_mean_spreads_inside_calibrated_margins"]


def test_score_record_validation_rejects_missing_actor_v43f():
    records = _records_v43f([0.0, 0.0, 0.0, 0.0], repeats=1)
    records[0]["actors"].pop()
    with pytest.raises(ValueError, match="actor ranks"):
        numeric.score_records_to_arrays_v43f(records)


def test_preregistration_and_runtime_dry_run_are_cpu_only_v43f(tmp_path, capsys):
    output = tmp_path / "preregistration-v43f.json"
    assert builder.main(["--output", str(output)]) == 0
    value = json.loads(output.read_text())
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    assert value["content_sha256_before_self_field"] == runtime.v40a.canonical_sha256(
        compact
    )
    assert value["schema"] == "matched-lora-es-preregistration-v43f"
    assert value["numeric_calibration"]["warmup_repeats"] == 2
    assert value["numeric_calibration"]["retained_repeats_per_actor"] == 8
    assert value["numeric_calibration"]["bootstrap_resamples"] == 10_000
    assert value["recipe"]["signed_replicates_per_direction"] == 2
    assert value["post_update_consensus"]["retained_repeats_per_actor"] == 8
    args = [
        "--preregistration", str(output),
        "--preregistration-sha256", runtime.v40a.file_sha256(output),
        "--preregistration-content-sha256",
        value["content_sha256_before_self_field"],
        "--dry-run",
    ]
    assert runtime.main(args) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    dry = json.loads(lines[-1])
    assert dry == {
        "content_sha256": value["content_sha256_before_self_field"],
        "gpu_launched": False,
        "schema": "matched-lora-es-preregistration-v43f",
        "sealed_holdout_opened": False,
        "train_only": True,
    }
