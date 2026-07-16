#!/usr/bin/env python3

import json
import types

import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v45b as subject


class FakeRemote:
    def __init__(self, index, calls):
        self.index = index
        self.calls = calls

    def remote(self, prompts, params, **kwargs):
        self.calls.append((self.index, prompts, params, kwargs))
        return len(self.calls)


def test_four_full_waves_and_identical_work_v45b(monkeypatch):
    waves = subject.arm_wave_plan_v45b()
    assert len(waves) == 4
    assert all(tuple(engine for _, engine in wave) == (0, 1, 2, 3) for wave in waves)
    assert tuple(arm for wave in waves for arm, _ in wave) == subject.ARMS_V45B
    calls = []
    engines = [types.SimpleNamespace(generate=FakeRemote(i, calls)) for i in range(4)]
    monkeypatch.setattr(core, "lora_request_v44a", lambda arm: f"request:{arm}")
    with subject.patched_prior_constants_v45b():
        with subject.prior.patched_candidate_globals_v45a():
            handles = core.arm_requests_v44a(engines, ["same"], "same-params")
    assert handles == list(range(1, 17))
    assert [call[0] for call in calls] == [0, 1, 2, 3] * 4
    assert all(call[1] == ["same"] and call[2] == "same-params" for call in calls)
    assert all(call[3].get("lora_request") is None for call in calls[:6])
    assert all(call[3].get("lora_request") for call in calls[6:])


def metric(value):
    return {
        "generated_equal_unit_mean_reward": value,
        "generated_exact_count": int(value * 100),
        "generated_nonzero_count": int(value * 100),
        "teacher_forced_equal_unit_mean_answer_logprob": value,
        "protocol_leak_counters": {"x": 0},
    }


def test_padding_bases_are_excluded_and_both_es_arms_are_candidates_v45b():
    shadow = {arm: metric(0.1) for arm in subject.ARMS_V45B}
    shadow["base_f"] = metric(9.0)  # padding cannot enter candidate ranking
    shadow["lora_es_v43d"] = metric(0.2)
    shadow["lora_es_v43g"] = metric(0.3)
    gates = {
        arm: {
            "eligible": arm == "lora_es_v43g",
            "no_protocol_or_leak_counter_increase": True,
        }
        for arm in subject.CANDIDATE_ARMS_V45B
    }
    with subject.patched_prior_constants_v45b():
        result = subject.prior.choose_eligible_candidate_v45a(shadow, gates)
    assert result["selected_arm"] == "lora_es_v43g"
    assert "lora_es_v43d" in subject.CANDIDATE_ARMS_V45B
    assert "lora_es_v43g" in subject.CANDIDATE_ARMS_V45B
    assert all(base not in gates for base in subject.BASE_ARMS_V45B)


def test_ten_candidate_stages_include_v43d_and_v43g_v45b():
    stages = subject.staged_adapter_bindings_v45b()
    assert tuple(stages) == subject.CANDIDATE_ARMS_V45B
    assert len(stages) == 10
    assert stages["lora_es_v43g"]["tensor_count"] == 70
    assert stages["lora_es_v43g"]["tensor_bytes_preserved_exactly"] is True


def test_v45a_result_and_v43g_success_are_nonraw_sealed_provenance_v45b():
    result = subject.prior_result_v45b()
    source = subject.v43g_stage.source_seal_v45b()
    assert result["selected_arm"] == "sft_v42g"
    assert result["strict_gate_passed"] is True
    assert result["heldout_or_holdout_opened"] is False
    assert source["state_complete"] is True
    assert source["selection_data_opened"] is False
    assert source["snapshot_master_sha256"] == subject.v43g_stage.EXPECTED[
        "master_sha256"
    ]


def test_builder_and_loader_do_not_touch_protected_semantics_v45b(
    tmp_path, monkeypatch
):
    import build_matched_lora_candidate_eval_preregistration_v45b as builder

    protected = {item["path"] for item in core.PROTECTED_INPUTS_V44A.values()}
    original = core.file_sha256

    def guarded(candidate):
        assert str(candidate) not in protected
        return original(candidate)

    monkeypatch.setattr(core, "file_sha256", guarded)
    value = builder.build()
    assert value["protected_semantics_inspected_during_v45b_revision"] is False
    assert value["heldout_or_holdout_access_authorized"] is False
    assert value["padding_base_arms"] == ["base_d", "base_e", "base_f"]
    assert value["selection_protocol_v45a"][
        "padding_base_arms_affect_candidate_eligibility_or_ranking"
    ] is False
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    args = type("Args", (), {
        "preregistration": str(path),
        "preregistration_sha256": original(path),
        "preregistration_content_sha256": value[
            "content_sha256_before_self_field"
        ],
    })()
    assert subject.load_preregistration_v45b(args)["candidate_arms"] == list(
        subject.CANDIDATE_ARMS_V45B
    )
