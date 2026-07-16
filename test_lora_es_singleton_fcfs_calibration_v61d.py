#!/usr/bin/env python3

import copy
import inspect
import json
import types

import build_lora_es_singleton_fcfs_preregistration_v61d as builder
import lora_es_singleton_fcfs_calibration_v61d as subject
import run_lora_es_singleton_fcfs_calibration_v61d as runtime
from test_lora_es_paired_null_calibration_v61c import _evidence as v61c_evidence


def _evidence_v61d() -> dict:
    value = v61c_evidence()
    value["schema"] = (
        "v61d-singleton-fcfs-identical-state-paired-evaluator-evidence"
    )
    value["status"] = "complete_matched_alpha_zero_no_update_characterization"
    value["runtime_determinism_controls"] = copy.deepcopy(
        subject.RUNTIME_CONTROLS_V61D
    )
    value["matched_v61c_panel_labels_metrics_bootstrap_thresholds"] = True
    value["v61c_thresholds_relaxed_or_changed"] = False
    return value


def test_v61d_support_audit_and_engine_kwargs_bind_feasible_controls():
    audit = runtime._read_support_audit_v61d()
    assert audit["singleton_fcfs_controls_supported"] is True
    assert audit["requested_runtime_controls"] == subject.RUNTIME_CONTROLS_V61D
    assert audit["batch_invariant_environment_resolved_false"] is True
    assert audit["global_batch_invariance_claimed"] is False
    fake = types.SimpleNamespace(MODEL="model")
    kwargs = runtime.engine_kwargs_v61d(fake)
    assert {key: kwargs[key] for key in (
        "async_scheduling", "max_num_seqs", "scheduling_policy", "enforce_eager",
    )} == {
        "async_scheduling": False,
        "max_num_seqs": 1,
        "scheduling_policy": "fcfs",
        "enforce_eager": True,
    }
    source = inspect.getsource(runtime._make_trainer_v61d)
    for check in (
        'batch_invariant_env != "0"',
        "vllm_envs.VLLM_BATCH_INVARIANT is not False",
        "scheduler.async_scheduling is not False",
        "scheduler.max_num_seqs != 1",
        'scheduler.policy != "fcfs"',
        "config.model_config.enforce_eager is not True",
        "tuned_table_content_sha256",
    ):
        assert check in source


def test_v61d_numeric_analysis_is_exact_v61c_contract_under_new_schema():
    evidence = _evidence_v61d()
    result = subject.build_analysis_v61d(evidence)
    converted = subject._as_v61c_evidence(evidence)
    baseline = subject.v61c.build_analysis_v61c(converted)
    for key in (
        "ranking_bootstrap", "within_actor_same_label_repeat",
        "counterbalance_pair_receipts", "noise_scale_comparison",
        "exact_sentinel", "alpha",
    ):
        assert result[key] == baseline[key]
    assert result["runtime_determinism_controls"] == subject.RUNTIME_CONTROLS_V61D
    assert result["matched_v61c_panel_labels_metrics_bootstrap_thresholds"] is True
    assert result["v61c_thresholds_relaxed_or_changed"] is False


def test_v61d_preregistration_is_matched_no_relaxation_and_dry_run_is_closed(
    tmp_path, monkeypatch, capsys,
):
    value = builder.build_v61d()
    assert value["gpu_launch_authorized"] is True
    assert value["fixed_calibration_recipe"][
        "runtime_determinism_controls"
    ] == subject.RUNTIME_CONTROLS_V61D
    assert value["matched_v61c_contract"]["threshold_relaxation"] is False
    unchanged = value["unchanged_analysis_contract"]
    assert unchanged["bootstrap_replicates"] == 4096
    assert unchanged["bootstrap_seed"] == 2026071612
    assert unchanged["generated_f1_practical_effect_scale"] == 0.01
    assert unchanged["teacher_logprob_practical_effect_scale"] == 0.001
    assert unchanged["teacher_max_absolute_point_null"] == 0.00025
    assert unchanged["teacher_max_ci_halfwidth"] == 0.001
    assert unchanged["teacher_max_repeat_mean_absolute_delta"] == 0.001
    assert unchanged["any_individual_exact_sentinel_flip_fails"] is True
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    file_sha = runtime.runtime_v61a.file_sha256_v61a(path)
    content_sha = value["content_sha256_before_self_field"]
    args = types.SimpleNamespace(
        preregistration=str(path), preregistration_sha256=file_sha,
        preregistration_content_sha256=content_sha,
    )
    assert runtime.load_preregistration_v61d(args) == value
    monkeypatch.setattr(
        runtime.runtime_v61c, "load_staged_inputs_v61c",
        lambda: (_ for _ in ()).throw(AssertionError("dry-run opened staged data")),
    )
    assert runtime.main([
        "--preregistration", str(path),
        "--preregistration-sha256", file_sha,
        "--preregistration-content-sha256", content_sha,
        "--dry-run",
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["singleton_fcfs_controls_supported"] is True
    assert output["staged_train_rows_opened"] == 0
    assert output["model_or_gpu_loaded"] is False
    assert output["filesystem_writes"] is False


def test_v61d_evidence_schema_requires_exact_runtime_controls():
    value = _evidence_v61d()
    assert len(subject.validate_evidence_v61d(value)) == 68
    changed = copy.deepcopy(value)
    changed["runtime_determinism_controls"]["max_num_seqs"] = 2
    try:
        subject.validate_evidence_v61d(changed)
    except ValueError as error:
        assert "scheduling provenance" in str(error)
    else:
        raise AssertionError("v61d accepted changed scheduler controls")
