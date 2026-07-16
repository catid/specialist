#!/usr/bin/env python3

import types

import run_matched_lora_candidate_eval_v44a as subject


class FakeRemote:
    def __init__(self, index, calls):
        self.index = index
        self.calls = calls

    def remote(self, prompts, params, **kwargs):
        self.calls.append((self.index, prompts, params, kwargs))
        return len(self.calls)


def test_frozen_six_arm_four_engine_schedule_v44a(monkeypatch):
    calls = []
    engines = [types.SimpleNamespace(generate=FakeRemote(i, calls)) for i in range(4)]
    monkeypatch.setattr(subject, "lora_request_v44a", lambda arm: f"request:{arm}")
    handles = subject.arm_requests_v44a(engines, ["p"], "params")
    assert handles == [1, 2, 3, 4, 5, 6]
    assert [row[0] for row in calls] == [0, 1, 2, 3, 2, 3]
    assert [row[3].get("lora_request") for row in calls] == [
        None, None, "request:sft_v42b", "request:sft_v42c",
        "request:sft_v42d", "request:lora_es_v43d",
    ]
    assert set(index for index, *_ in calls) == {0, 1, 2, 3}


def _metric(value, counter=0):
    return {
        "generated_equal_unit_mean_reward": value,
        "generated_exact_count": int(value * 100),
        "generated_nonzero_count": int(value * 100),
        "teacher_forced_equal_unit_mean_answer_logprob": value,
        "protocol_leak_counters": {"x": counter},
    }


def test_selection_is_candidates_only_and_frozen_tie_v44a():
    metrics = {arm: _metric(0.1) for arm in subject.ARMS}
    for arm in subject.CANDIDATE_ARMS:
        metrics[arm] = _metric(0.2)
    result = subject.select_candidate_v44a(metrics)
    assert result["selected_arm"] == "lora_es_v43d"
    assert result["shadow_improvement_gate_passed"] is True


def test_nonprotected_inventory_excludes_protected_inputs_v44a():
    protected = {item["path"] for item in subject.PROTECTED_INPUTS_V44A.values()}
    assert protected.isdisjoint(str(path) for path in subject.nonprotected_paths_v44a().values())
    assert subject.CANDIDATE_ARMS == (
        "sft_v42b", "sft_v42c", "sft_v42d", "lora_es_v43d"
    )
