#!/usr/bin/env python3

import audit_lora_es_transition_microbenchmark_v51 as subject


def test_v51_audit_accepts_exact_direct_master_speed_recipe():
    value = subject.build_audit()
    assert value["decision"] == {
        "accepted": True,
        "scope": "population transition implementation only",
        "quality_or_checkpoint_promotion_authorized": False,
        "reason": (
            "The direct-master single-slot transition passed every exact "
            "state/restore/safety gate and both preregistered speed gates."
        ),
    }
    assert all(value["correctness"].values())
    speed = value["speed"]
    assert speed["v50"]["wall_seconds"] == 254.277634
    assert speed["v51"]["wall_seconds"] == 208.916206
    assert speed["population_wall_reduction_fraction"] > 0.178
    assert speed["median_idle_gap_reduction_fraction"] > 0.402
    assert speed["population_wall_gate"]["passed"] is True
    assert speed["median_idle_gap_gate"]["passed"] is True
