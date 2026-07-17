from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import build_lora_es_mirrored_calibration_preregistration_v66d as builder
import eggroll_es_gpu_telemetry_v66 as telemetry
import run_lora_es_mirrored_calibration_v66 as v66
import run_lora_es_mirrored_calibration_v66d as runtime


def write_preregistration(path: Path):
    value = builder.build_preregistration_v66d()
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return value


def test_builder_is_deterministic_and_binds_substantive_v66c_evidence():
    first = builder.build_preregistration_v66d()
    second = builder.build_preregistration_v66d()
    assert first == second
    compact = {
        key: item for key, item in first.items()
        if key != "content_sha256_before_self_field"
    }
    assert first["content_sha256_before_self_field"] == (
        v66.mirrored.canonical_sha256_v66(compact)
    )
    v66c = first["supersedes_failed_attempts"]["v66c"]
    assert v66c["signed_candidates_completed"] == 16
    assert v66c["mirrored_waves_completed"] == 4
    assert v66c["nonzero_pair_differences"] == 8
    assert v66c["state_or_es_protocol_failure"] is False
    assert v66c["final_gpu_idle"] is True
    assert first["beads"] == [
        "specialist-0j5.2", "specialist-0j5.12", "specialist-nen.25",
    ]
    for binding in first["implementation_bindings"].values():
        path = Path(binding["path"])
        assert path.is_file()
        assert v66.file_sha256_v66(path) == binding["file_sha256"]


def test_dry_run_seals_handshake_and_actor_receipts_without_writes(
    tmp_path, capsys,
):
    preregistration = tmp_path / "v66d.json"
    value = write_preregistration(preregistration)
    before = (runtime.ATTEMPT.exists(), runtime.RUN_DIR.exists())
    result = runtime.main([
        "--preregistration", str(preregistration),
        "--preregistration-sha256", v66.file_sha256_v66(preregistration),
        "--preregistration-content-sha256",
        value["content_sha256_before_self_field"],
        "--dry-run",
    ])
    assert result == 0
    assert (runtime.ATTEMPT.exists(), runtime.RUN_DIR.exists()) == before
    output = json.loads(capsys.readouterr().out)
    assert output["phase_transition_requires_four_gpu_sample_ack"] is True
    assert output["actor_cuda_event_and_output_receipt_per_candidate"] is True
    assert output["nvml_positive_sample_required_for_short_phase"] is False
    assert output["worker_contract"]["required_worker_endpoints"] == list(
        runtime.REQUIRED_WORKER_ENDPOINTS_V66D
    )
    assert output["train_semantics_model_ray_or_gpu_loaded"] is False
    assert output["filesystem_writes"] is False


def test_v66d_paths_are_fresh_and_do_not_reuse_v66c_artifacts():
    assert "v66d_" in runtime.RUN_DIR.name
    assert runtime.RUN_DIR != builder.FAILED_V66C_RUN
    assert runtime.GPU_WORK_LOG.parent == runtime.RUN_DIR
    assert set(runtime.artifacts_v66d()) == {
        "attempt", "run_directory", "gpu_log", "actor_cuda_work_log",
        "population", "update", "report", "failure",
    }


def test_worker_contract_resolves_cuda_event_endpoints_before_gpu_access():
    contract = runtime.validate_lora_worker_contract_v66d()
    assert contract["state_mode"] == "external_worker"
    assert contract["resolved_class"] == runtime.WORKER_EXTENSION_V66D
    assert contract["dense_full_weight_master_install_authorized"] is False
    assert "begin_actor_gpu_work_v66d" in contract[
        "required_worker_endpoints"
    ]
    assert "end_actor_gpu_work_v66d" in contract[
        "required_worker_endpoints"
    ]


class PhaseRecorder:
    def __init__(self, calls):
        self.calls = calls
        self._value = "materialize"

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self.calls.append(("phase_acknowledged", value))
        self._value = value


class CollectiveRemote:
    def __init__(self, calls, begin_receipt):
        self.calls = calls
        self.begin_receipt = begin_receipt

    def remote(self, method, args=()):
        self.calls.append(("collective_submit", method, args))
        return ("collective", self.begin_receipt)


class GenerateRemote:
    def __init__(self, calls):
        self.calls = calls

    def remote(self, *args, **kwargs):
        self.calls.append(("generation_submit", args, kwargs))
        return "generation-ref"


def test_generation_submission_waits_for_phase_ack_then_cuda_begin():
    calls = []
    assignment = {
        "wave_index": 0,
        "engine_rank": 0,
        "direction_seed": 123,
        "sign": 1,
        "pair_id": "a" * 64,
        "evaluation_contract_sha256": "b" * 64,
    }
    binding = {
        "actor_rank": 0,
        "worker_pid": 9000,
        "physical_gpu_id": 2,
    }
    begin = {
        "schema": "eggroll-es-actor-cuda-work-begin-v66d",
        "work_id": telemetry.work_id_v66d(assignment),
        **assignment,
        "worker_pid": 9000,
        "physical_gpu_id": 2,
        "cuda_event_start_recorded": True,
        "output_or_token_cardinality_observed": False,
    }
    context = runtime.LiveContextV66D()
    context.bindings = [
        binding,
        {"actor_rank": 1, "worker_pid": 9001, "physical_gpu_id": 0},
        {"actor_rank": 2, "worker_pid": 9002, "physical_gpu_id": 3},
        {"actor_rank": 3, "worker_pid": 9003, "physical_gpu_id": 1},
    ]
    engine = SimpleNamespace(
        collective_rpc=CollectiveRemote(calls, begin),
        generate=GenerateRemote(calls),
    )

    class Trainer:
        engines = [engine]

        @staticmethod
        def _resolve(handles):
            calls.append(("collective_resolved", handles))
            return [[handles[0][1]]]

    callback = object.__new__(runtime.RayMirroredCallbacksV66D)
    callback.context = context
    callback.trainer = Trainer()
    callback.phase = PhaseRecorder(calls)
    callback.sampling = object()
    callback.prior = SimpleNamespace(_lora_request=lambda: object())
    callback.prepared = {}
    payload = {
        "contract": {
            "evaluation_contract_sha256": assignment[
                "evaluation_contract_sha256"
            ],
        },
        "prompts": [{"prompt_token_ids": [1]}],
    }
    handle = callback.submit_evaluate(assignment, payload)
    assert isinstance(handle, runtime.RuntimeGenerationHandleV66D)
    assert [item[0] for item in calls] == [
        "phase_acknowledged",
        "collective_submit",
        "collective_resolved",
        "generation_submit",
    ]


def test_live_patch_relabels_attempt_and_restores_legacy_hooks(
    tmp_path, monkeypatch,
):
    import run_lora_topology_probe_v40a as v40a

    attempt = tmp_path / "attempt.json"
    monkeypatch.setattr(runtime, "ATTEMPT", attempt)
    original_atomic = v40a.atomic_json
    with runtime.patched_live_v66d():
        assert v40a.atomic_json is not original_atomic
        v40a.atomic_json(attempt, v40a.self_hashed({
            "schema": "v66-mirrored-qwen36-calibration-attempt",
            "status": "launching",
        }))
    assert v40a.atomic_json is original_atomic
    value = json.loads(attempt.read_text(encoding="utf-8"))
    assert value["schema"] == "v66d-mirrored-qwen36-calibration-attempt"
    assert value["gpu_work_attribution"].startswith("four-row")
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    assert value["content_sha256_before_self_field"] == v40a.canonical_sha256(
        compact
    )
