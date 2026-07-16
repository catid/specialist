#!/usr/bin/env python3

import json
import types

import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v45c as subject


class FakeRemote:
    def __init__(self, index, calls): self.index, self.calls = index, calls
    def remote(self, prompts, params, **kwargs):
        self.calls.append((self.index, prompts, params, kwargs)); return len(self.calls)


def test_five_full_waves_cover_twenty_arms_v45c(monkeypatch):
    waves = subject.arm_wave_plan_v45c()
    assert len(waves) == 5
    assert all(tuple(engine for _, engine in wave) == (0, 1, 2, 3) for wave in waves)
    assert tuple(arm for wave in waves for arm, _ in wave) == subject.ARMS_V45C
    calls = []; engines = [
        types.SimpleNamespace(generate=FakeRemote(i, calls)) for i in range(4)
    ]
    monkeypatch.setattr(core, "lora_request_v44a", lambda arm: f"request:{arm}")
    with subject.patched_prior_constants_v45c():
        with subject.prior.patched_prior_constants_v45b():
            with subject.prior.prior.patched_candidate_globals_v45a():
                handles = core.arm_requests_v44a(engines, ["same"], "params")
    assert handles == list(range(1, 21))
    assert [call[0] for call in calls] == [0, 1, 2, 3] * 5
    assert all(call[3].get("lora_request") is None for call in calls[:9])
    assert all(call[3].get("lora_request") for call in calls[9:])


def test_v42h_is_eleventh_candidate_and_padding_never_ranks_v45c():
    assert len(subject.CANDIDATE_ARMS_V45C) == 11
    assert subject.CANDIDATE_ARMS_V45C[-1] == "sft_v42h"
    assert set(subject.PADDING_BASE_ARMS_V45C).isdisjoint(
        subject.CANDIDATE_ARMS_V45C
    )
    assert "lora_es_v43g" in subject.CANDIDATE_ARMS_V45C


def test_all_eleven_stages_are_exact_v45c():
    stages = subject.staged_adapter_bindings_v45c()
    assert tuple(stages) == subject.CANDIDATE_ARMS_V45C
    assert len(stages) == 11
    assert stages["sft_v42h"]["tensor_count"] == 70
    assert stages["sft_v42h"]["tensor_bytes_preserved_exactly"] is True


def test_builder_loader_never_hash_protected_semantics_v45c(tmp_path, monkeypatch):
    import build_matched_lora_candidate_eval_preregistration_v45c as builder
    protected = {item["path"] for item in core.PROTECTED_INPUTS_V44A.values()}
    original = core.file_sha256
    def guarded(path): assert str(path) not in protected; return original(path)
    monkeypatch.setattr(core, "file_sha256", guarded)
    value = builder.build()
    assert value["protected_semantics_inspected_during_v45c_revision"] is False
    assert value["heldout_or_holdout_access_authorized"] is False
    assert value["runtime"]["every_gpu_receives_one_request_per_wave"] is True
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    args = type("Args", (), {
        "preregistration": str(path), "preregistration_sha256": original(path),
        "preregistration_content_sha256": value["content_sha256_before_self_field"],
    })()
    assert subject.load_preregistration_v45c(args)["candidate_arms"] == list(
        subject.CANDIDATE_ARMS_V45C
    )
