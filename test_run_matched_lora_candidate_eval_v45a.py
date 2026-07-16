#!/usr/bin/env python3

import json
import types

import pytest

import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v45a as subject


class FakeRemote:
    def __init__(self, index, calls):
        self.index = index
        self.calls = calls

    def remote(self, prompts, params, **kwargs):
        self.calls.append((self.index, prompts, params, kwargs))
        return len(self.calls)


def test_three_full_waves_use_every_gpu_with_same_batch_v45a(monkeypatch):
    waves = subject.arm_wave_plan_v45a()
    assert len(waves) == 3
    assert all(tuple(engine for _, engine in wave) == (0, 1, 2, 3) for wave in waves)
    assert tuple(arm for wave in waves for arm, _ in wave) == subject.ARMS_V45A
    calls = []
    engines = [types.SimpleNamespace(generate=FakeRemote(i, calls)) for i in range(4)]
    monkeypatch.setattr(core, "lora_request_v44a", lambda arm: f"request:{arm}")
    with subject.patched_candidate_globals_v45a():
        handles = core.arm_requests_v44a(engines, ["same"], "same-params")
    assert handles == list(range(1, 13))
    assert [call[0] for call in calls] == [0, 1, 2, 3] * 3
    assert all(call[1] == ["same"] and call[2] == "same-params" for call in calls)
    assert [call[3].get("lora_request") for call in calls[:3]] == [None] * 3
    assert all(call[3].get("lora_request") for call in calls[3:])


def metric(value):
    return {
        "generated_equal_unit_mean_reward": value,
        "generated_exact_count": int(value * 1000),
        "generated_nonzero_count": int(value * 1000),
        "teacher_forced_equal_unit_mean_answer_logprob": value,
        "protocol_leak_counters": {"x": 0},
    }


def gate_table(eligible=()):
    return {
        arm: {
            "eligible": arm in eligible,
            "no_protocol_or_leak_counter_increase": True,
        }
        for arm in subject.CANDIDATE_ARMS_V45A
    }


def test_unsafe_higher_shadow_cannot_mask_safe_lower_shadow_v45a():
    shadow = {arm: metric(0.10) for arm in subject.ARMS_V45A}
    shadow["sft_v42b"] = metric(0.90)  # unsafe but highest shadow
    shadow["sft_v42c"] = metric(0.20)  # safe and lower shadow
    result = subject.choose_eligible_candidate_v45a(
        shadow, gate_table(("sft_v42c",))
    )
    assert result["selected_arm"] == "sft_v42c"
    assert "sft_v42b" in result["ineligible_arms"]
    assert result["ood_eligible_set_constructed_before_shadow_ranking"] is True
    assert result["shadow_improvement_gate_passed"] is True


def test_no_eligible_candidate_uses_failing_base_sentinel_v45a():
    shadow = {arm: metric(0.10) for arm in subject.ARMS_V45A}
    result = subject.choose_eligible_candidate_v45a(shadow, gate_table())
    assert result["selected_arm"] == "base_a"
    assert result["selected_candidate_arm"] is None
    assert result["eligible_arms"] == []
    assert result["shadow_improvement_gate_passed"] is False


def test_paired_qa_bootstrap_is_aligned_and_deterministic_v45a():
    base = [
        {"item_sha256": str(i), "reward": 0.0, "format": "wrong"}
        for i in range(4)
    ]
    candidate = [
        {"item_sha256": str(i), "reward": 1.0, "format": "exact"}
        for i in range(4)
    ]
    one = subject.paired_qa_bootstrap_v45a(base, candidate, samples=100)
    two = subject.paired_qa_bootstrap_v45a(base, candidate, samples=100)
    assert one == two
    assert one["reward_mean_delta_paired_item_bootstrap_95_ci"] == [1.0, 1.0]
    assert one["exact_rate_delta_paired_item_bootstrap_95_ci"] == [1.0, 1.0]
    candidate[0]["item_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="identities changed"):
        subject.paired_qa_bootstrap_v45a(base, candidate, samples=2)


def test_all_nine_candidate_stages_and_three_bases_are_bound_v45a():
    stages = subject.staged_adapter_bindings_v45a()
    assert tuple(stages) == subject.CANDIDATE_ARMS_V45A
    assert len(stages) == 9
    assert subject.BASE_ARMS_V45A == ("base_a", "base_b", "base_c")
    assert all(value["tensor_count"] == 70 for value in stages.values())
    assert all(value["tensor_bytes_preserved_exactly"] for value in stages.values())


def test_builder_loader_and_holdout_firewall_v45a(tmp_path, monkeypatch):
    import build_matched_lora_candidate_eval_preregistration_v45a as builder

    value = builder.build()
    assert value["selection_protocol_v45a"][
        "unsafe_higher_shadow_arm_cannot_mask_safe_lower_shadow_arm"
    ] is True
    assert value["selection_protocol_v45a"][
        "candidate_order_affects_eligibility"
    ] is False
    assert value["heldout_or_holdout_access_authorized"] is False
    assert value["runtime"]["every_gpu_receives_one_request_per_wave"] is True
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    protected = {item["path"] for item in core.PROTECTED_INPUTS_V44A.values()}
    original = core.file_sha256

    def guarded(candidate):
        assert str(candidate) not in protected
        return original(candidate)

    monkeypatch.setattr(core, "file_sha256", guarded)
    args = type("Args", (), {
        "preregistration": str(path),
        "preregistration_sha256": original(path),
        "preregistration_content_sha256": value[
            "content_sha256_before_self_field"
        ],
    })()
    assert subject.load_preregistration_v45a(args)["arms"] == list(
        subject.ARMS_V45A
    )
