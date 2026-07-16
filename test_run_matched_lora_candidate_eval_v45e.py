#!/usr/bin/env python3

import json
import types

import pytest

import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v45e as subject


class FakeRemote:
    def __init__(self, index, calls):
        self.index, self.calls = index, calls

    def remote(self, prompts, params, **kwargs):
        self.calls.append((self.index, prompts, params, kwargs))
        return len(self.calls)


def test_three_full_waves_and_padding_positions_v45e(monkeypatch):
    waves = subject.arm_wave_plan_v45e()
    assert len(waves) == 3
    assert all(tuple(engine for _, engine in wave) == (0, 1, 2, 3)
               for wave in waves)
    assert tuple(arm for wave in waves for arm, _ in wave) == subject.ARMS_V45E
    calls = []
    engines = [
        types.SimpleNamespace(generate=FakeRemote(i, calls)) for i in range(4)
    ]
    monkeypatch.setattr(core, "lora_request_v44a", lambda arm: f"request:{arm}")
    with subject.patched_candidate_globals_v45e():
        handles = core.arm_requests_v44a(engines, ["same"], "params")
    assert handles == list(range(1, 13))
    assert [call[0] for call in calls] == [0, 1, 2, 3] * 3
    for index, arm in enumerate(subject.ARMS_V45E):
        request = calls[index][3].get("lora_request")
        assert (request is None) == (arm in subject.BASE_ARMS_V45E)


def test_exact_replica_stage_bindings_share_bytes_v45e():
    stages = subject.replica_stage_bindings_v45e()
    assert tuple(stages) == subject.CANDIDATE_ARMS_V45E
    for logical, (left, right) in subject.LOGICAL_REPLICAS_V45E.items():
        assert stages[left]["logical_candidate"] == logical
        assert stages[right]["logical_candidate"] == logical
        for key in (
            "directory", "weights_file_sha256", "adapter_config_file_sha256",
            "manifest_file_sha256", "manifest_content_sha256",
            "transformed_identity_sha256",
        ):
            assert stages[left][key] == stages[right][key]
        assert stages[left]["adapter_id"] != stages[right]["adapter_id"]
        assert stages[left]["tensor_bytes_preserved_exactly"] is True


def test_replica_equivalence_is_fail_closed_v45e():
    metrics = {arm: {"value": 1.0} for arm in subject.ARMS_V45E}
    raw = {arm: [{"value": 1.0}] for arm in subject.ARMS_V45E}
    result = subject.assert_replica_equivalence_v45e(metrics, raw, "synthetic")
    assert result["all_logical_replicas_exact"] is True
    metrics["sft_v42k_replica_b"] = {"value": 1.00001}
    with pytest.raises(RuntimeError, match="sft_v42k replica equivalence"):
        subject.assert_replica_equivalence_v45e(metrics, raw, "synthetic")


def test_builder_loader_use_zero_protected_semantics_v45e(tmp_path,
                                                            monkeypatch):
    import build_matched_lora_candidate_eval_preregistration_v45e as builder
    protected = {item["path"] for item in core.PROTECTED_INPUTS_V44A.values()}
    original = core.file_sha256

    def guarded(path):
        assert str(path) not in protected
        return original(path)

    monkeypatch.setattr(core, "file_sha256", guarded)
    value = builder.build()
    assert value["protected_semantics_inspected_during_v45e_revision"] is False
    assert value["heldout_or_holdout_access_authorized"] is False
    assert value["runtime"]["all_four_gpus_busy_in_every_evaluation_wave"]
    assert value["selection_protocol_v45e"][
        "ood_eligible_set_constructed_before_shadow_ranking"
    ] is True
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    args = type("Args", (), {
        "preregistration": str(path),
        "preregistration_sha256": original(path),
        "preregistration_content_sha256": value[
            "content_sha256_before_self_field"
        ],
    })()
    loaded = subject.load_preregistration_v45e(args)
    assert loaded["logical_candidates"] == list(subject.LOGICAL_CANDIDATES_V45E)
