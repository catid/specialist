import json
from pathlib import Path

import numpy as np
import pytest

import build_general_qa_proxy_anchor_v43h as qa_builder
import build_lora_es_multi_anchor_preregistration_v43h as prereg_builder
import eggroll_es_multi_anchor_v43h as subject


def test_simultaneous_projection_satisfies_every_anchor_and_trust_region_v43h():
    domain = [-2.0, -1.0, 3.0, 0.0]
    anchors = {
        "prose_lm": [1.0, 0.0, 0.0, -1.0],
        "qa_answer_logprob": [0.0, 1.0, -1.0, 0.0],
    }
    result = subject.project_multi_anchor_trust_region_v43h(domain, anchors)
    coefficients = np.asarray(result["coefficients"])
    assert np.linalg.norm(coefficients) <= 0.5 * np.linalg.norm(domain) + 1e-12
    assert all(
        float(np.dot(coefficients, values)) >= -1e-12
        for values in anchors.values()
    )
    assert result["diagnostics"]["all_anchor_halfspaces_satisfied"] is True
    assert result["diagnostics"]["update_norm_ratio"] <= 0.5 + 1e-12


def test_projection_handles_opposed_anchors_by_safely_removing_coordinate_v43h():
    result = subject.project_multi_anchor_trust_region_v43h(
        [2.0, 1.0, -1.0],
        {
            "negative_axis": [-1.0, 0.0, 0.0],
            "positive_axis": [1.0, 0.0, 0.0],
        },
        max_norm_ratio=1.0,
    )
    assert result["coefficients"][0] == pytest.approx(0.0, abs=1e-12)
    assert result["coefficients"][1:] == pytest.approx([1.0, -1.0])


def test_zero_spread_required_anchor_fails_closed_v43h():
    result = subject.project_multi_anchor_trust_region_v43h(
        [1.0, -1.0, 0.5, -0.5],
        {
            "prose_lm": [0.0, 0.0, 0.0, 0.0],
            "qa_answer_logprob": [1.0, -1.0, 1.0, -1.0],
        },
    )
    assert result["coefficients"] == [0.0] * 4
    assert result["diagnostics"]["decision"] == "skip_required_anchor_no_spread"
    assert result["diagnostics"]["zero_spread_anchors"] == ["prose_lm"]


def test_anchor_names_and_inputs_fail_closed_v43h():
    with pytest.raises(ValueError, match="canonical sorted order"):
        subject.project_multi_anchor_trust_region_v43h(
            [1.0, -1.0], {"z": [1.0, 0.0], "a": [0.0, 1.0]},
        )
    with pytest.raises(ValueError, match="population size"):
        subject.project_multi_anchor_trust_region_v43h(
            [1.0, -1.0], {"anchor": [1.0, 0.0, -1.0]},
        )


def test_qa_proxy_builder_never_opens_direct_benchmark_source_v43h(monkeypatch):
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.resolve() == qa_builder.DIRECT_BENCHMARK_SOURCE:
            raise AssertionError("direct benchmark source was opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    output, report = qa_builder.build_v43h()
    rows = [json.loads(line) for line in output.decode("utf-8").splitlines()]
    assert len(rows) == 128
    assert len({row["document_id"] for row in rows}) == 128
    assert len({row["item_id"] for row in rows}) == 128
    assert all(row["split"] == "anchor_general_qa_proxy" for row in rows)
    assert all(".." not in row["answer"] for row in rows)
    assert report["direct_benchmark_source"]["opened"] is False
    assert report["direct_benchmark_source"]["authorized_for_qa_semantics"] is False
    assert report["policy"][
        "protected_eval_ood_heldout_or_benchmark_semantics_opened"
    ] is False


def test_objective_coefficients_reuse_robust_antithetic_centered_ranks_v43h():
    plus, minus = [], []
    for direction in range(8):
        value = (direction - 3.5) * 0.001
        plus.append([value, value, value, value + 50.0])
        minus.append([-value, -value, -value, -value - 50.0])
    result = subject.objective_coefficients_v43h({"plus": plus, "minus": minus})
    assert result["zero_spread"] is False
    assert result["coefficients"] == sorted(result["coefficients"])


def test_static_preregistration_targets_v44c_collapse_but_is_not_launchable_v43h():
    value = prereg_builder.build_v43h()
    assert value["gpu_launch_authorized"] is False
    assert value["sealed_holdout_opened"] is False
    assert value["source_firewall"]["direct_hotpotqa_benchmark_opened"] is False
    v44c = value["parents"]["v44c_aggregate_only_evidence"]["numeric_evidence"]
    assert v44c["ood_qa_mean_reward_delta"] == -0.4081396422205246
    assert v44c["ood_qa_exact_count_delta"] == -11
    v45a = value["parents"]["v45a_aggregate_only_counterevidence"][
        "numeric_evidence"
    ]
    assert v45a["selected_arm"] == "sft_v42g"
    assert v45a["final_gate_passed"] is True
    assert value["recipe"]["projection"]["trust_region_max_norm_ratio"] == 0.5
    assert value["recipe"]["required_gradient_anchors"] == [
        "prose_lm", "qa_answer_logprob",
    ]
    assert "not yet built" in value["launch_blocker"]
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    assert value["content_sha256_before_self_field"] == (
        prereg_builder.canonical_sha256(compact)
    )
