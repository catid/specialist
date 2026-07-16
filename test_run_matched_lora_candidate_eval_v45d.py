#!/usr/bin/env python3

import json
import types

import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v45d as subject


class FakeRemote:
    def __init__(self, index, calls):
        self.index, self.calls = index, calls

    def remote(self, prompts, params, **kwargs):
        self.calls.append((self.index, prompts, params, kwargs))
        return len(self.calls)


def test_two_full_waves_cover_exact_compact_boundary_v45d(monkeypatch):
    waves = subject.arm_wave_plan_v45d()
    assert len(waves) == 2
    assert all(tuple(engine for _, engine in wave) == (0, 1, 2, 3)
               for wave in waves)
    assert tuple(arm for wave in waves for arm, _ in wave) == subject.ARMS_V45D
    calls = []
    engines = [
        types.SimpleNamespace(generate=FakeRemote(i, calls)) for i in range(4)
    ]
    monkeypatch.setattr(core, "lora_request_v44a", lambda arm: f"request:{arm}")
    with subject.patched_prior_constants_v45d():
        with subject.prior.patched_candidate_globals_v45a():
            handles = core.arm_requests_v44a(engines, ["same"], "params")
    assert handles == list(range(1, 9))
    assert [call[0] for call in calls] == [0, 1, 2, 3] * 2
    assert all(call[3].get("lora_request") is None for call in calls[:4])
    assert all(call[3].get("lora_request") for call in calls[4:])


def test_all_four_candidate_stages_are_exact_v45d():
    stages = subject.staged_adapter_bindings_v45d()
    assert tuple(stages) == subject.CANDIDATE_ARMS_V45D
    assert len(stages) == 4
    assert all(stage["tensor_count"] == 70 for stage in stages.values())
    assert all(stage["tensor_bytes_preserved_exactly"] is True
               for stage in stages.values())


def test_builder_loader_access_zero_protected_semantics_v45d(tmp_path,
                                                              monkeypatch):
    import build_matched_lora_candidate_eval_preregistration_v45d as builder
    protected = {item["path"] for item in core.PROTECTED_INPUTS_V44A.values()}
    original = core.file_sha256

    def guarded(path):
        assert str(path) not in protected
        return original(path)

    monkeypatch.setattr(core, "file_sha256", guarded)
    value = builder.build()
    assert value["protected_semantics_inspected_during_v45d_revision"] is False
    assert value["heldout_or_holdout_access_authorized"] is False
    assert value["runtime"]["all_four_gpus_busy_in_every_evaluation_wave"]
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    args = type("Args", (), {
        "preregistration": str(path),
        "preregistration_sha256": original(path),
        "preregistration_content_sha256": value[
            "content_sha256_before_self_field"
        ],
    })()
    assert subject.load_preregistration_v45d(args)["candidate_arms"] == list(
        subject.CANDIDATE_ARMS_V45D
    )
