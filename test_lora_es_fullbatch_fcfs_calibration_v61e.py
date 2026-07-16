#!/usr/bin/env python3

import copy
import inspect
import json
import types

import build_lora_es_fullbatch_fcfs_preregistration_v61e as builder
import lora_es_fullbatch_fcfs_calibration_v61e as subject
import run_lora_es_fullbatch_fcfs_calibration_v61e as runtime
from test_lora_es_paired_null_calibration_v61c import _evidence as v61c_evidence


def _evidence_v61e() -> dict:
    value = v61c_evidence()
    value["schema"] = (
        "v61e-fullbatch-fcfs-identical-state-paired-evaluator-evidence"
    )
    value["status"] = "complete_matched_alpha_zero_no_update_characterization"
    value["runtime_determinism_controls"] = copy.deepcopy(
        subject.RUNTIME_CONTROLS_V61E
    )
    value["matched_v61c_panel_labels_metrics_bootstrap_thresholds"] = True
    value["v61c_effective_request_batch_size"] = 68
    value["v61c_thresholds_relaxed_or_changed"] = False
    return value


def test_v61e_support_audit_and_engine_kwargs_bind_fullbatch_sync_controls():
    audit = runtime._read_support_audit_v61e()
    assert audit["fullbatch_fcfs_controls_supported"] is True
    assert audit["requested_runtime_controls"] == subject.RUNTIME_CONTROLS_V61E
    assert audit["scheduler_config_projection"] == {
        "async_scheduling": False,
        "max_num_seqs": 68,
        "policy": "fcfs",
        "scheduler_class": "Scheduler",
    }
    fake = types.SimpleNamespace(MODEL="model")
    kwargs = runtime.engine_kwargs_v61e(fake)
    assert {key: kwargs[key] for key in (
        "async_scheduling",
        "max_num_seqs",
        "scheduling_policy",
        "enforce_eager",
    )} == {
        "async_scheduling": False,
        "max_num_seqs": 68,
        "scheduling_policy": "fcfs",
        "enforce_eager": True,
    }
    source = inspect.getsource(runtime._make_trainer_v61e)
    for check in (
        'batch_invariant_env != "0"',
        "vllm_envs.VLLM_BATCH_INVARIANT is not False",
        "scheduler.async_scheduling is not False",
        "scheduler.max_num_seqs != 68",
        'scheduler.policy != "fcfs"',
        "config.model_config.enforce_eager is not True",
        "tuned_table_content_sha256",
    ):
        assert check in source


def test_v61e_numeric_analysis_is_exact_v61c_contract_under_new_schema():
    evidence = _evidence_v61e()
    result = subject.build_analysis_v61e(evidence)
    baseline = subject.v61c.build_analysis_v61c(
        subject._as_v61c_evidence(evidence)
    )
    for key in (
        "ranking_bootstrap",
        "within_actor_same_label_repeat",
        "counterbalance_pair_receipts",
        "noise_scale_comparison",
        "exact_sentinel",
        "alpha",
    ):
        assert result[key] == baseline[key]
    assert result["runtime_determinism_controls"] == (
        subject.RUNTIME_CONTROLS_V61E
    )
    assert result["v61c_effective_request_batch_size"] == 68
    assert result["v61c_thresholds_relaxed_or_changed"] is False


def test_v61e_preregistration_is_matched_no_relaxation_and_dry_run_closed(
    tmp_path,
    monkeypatch,
    capsys,
):
    value = builder.build_v61e()
    assert value["gpu_launch_authorized"] is True
    assert value["fixed_calibration_recipe"][
        "runtime_determinism_controls"
    ] == subject.RUNTIME_CONTROLS_V61E
    assert value["fixed_calibration_recipe"][
        "submitted_requests_per_actor_call"
    ] == 68
    assert value["matched_v61c_v61d_contract"]["threshold_relaxation"] is False
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
        preregistration=str(path),
        preregistration_sha256=file_sha,
        preregistration_content_sha256=content_sha,
    )
    assert runtime.load_preregistration_v61e(args) == value
    monkeypatch.setattr(
        runtime.runtime_v61c,
        "load_staged_inputs_v61c",
        lambda: (_ for _ in ()).throw(
            AssertionError("dry-run opened staged data")
        ),
    )
    assert runtime.main([
        "--preregistration",
        str(path),
        "--preregistration-sha256",
        file_sha,
        "--preregistration-content-sha256",
        content_sha,
        "--dry-run",
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["fullbatch_fcfs_controls_supported"] is True
    assert output["staged_train_rows_opened"] == 0
    assert output["prior_numeric_evidence_opened"] is False
    assert output["model_or_gpu_loaded"] is False
    assert output["filesystem_writes"] is False


def test_v61e_binds_v61c_and_v61d_evidence_and_finalizers_without_opening():
    value = builder.build_v61e()
    assert value["matched_numeric_sources"] == {
        "v61c_evidence": {
            "file_sha256": (
                "5be0a46ef0051c760b89d535cc252eeb1c9a6b2c700c209799049191615fa3dc"
            ),
            "content_sha256": (
                "15b7d74ea9b003d03ad4ba7667936ac80fac121cbbc28e4ced2c1cd9f57c7fa8"
            ),
        },
        "v61c_finalizer": {
            "file_sha256": (
                "d3d5eabf1e5d9b0bed2dfd2a355ed5eb839a22cb4bcdea58af0ab84231042d46"
            ),
            "content_sha256": (
                "7bc9735dea87ae8bf2374bcefb7c290b7bb273f2394b44d54dc1fa69e8e851c0"
            ),
        },
        "v61d_evidence": {
            "file_sha256": (
                "49be43e8a2e02093952bec7a0186f900fd64e3ec00057ece31e290a540c7044e"
            ),
            "content_sha256": (
                "f07a24fcd5ae0cedf1703f1bf25a7e9b6ca3db900d4bd58cc7351a68ec795048"
            ),
        },
        "v61d_finalizer": {
            "file_sha256": (
                "98da3f65e5d6a3801d1b56a143b7d9a44d95b971d290523372013525fad814fd"
            ),
            "content_sha256": (
                "58f15e71f7bdf2b7e3804479627bd14e782303004728079d18b2e8fbe09c657a"
            ),
        },
    }
    access = value["access_contract"]
    assert access["matched_prior_numeric_files_opened_to_build_preregistration"] is False
    assert access["v61c_or_v61d_row_level_evidence_may_open"] is False


def test_v61e_evidence_schema_requires_exact_fullbatch_controls():
    value = _evidence_v61e()
    assert len(subject.validate_evidence_v61e(value)) == 68
    changed = copy.deepcopy(value)
    changed["runtime_determinism_controls"]["max_num_seqs"] = 1
    try:
        subject.validate_evidence_v61e(changed)
    except ValueError as error:
        assert "scheduling provenance" in str(error)
    else:
        raise AssertionError("v61e accepted singleton scheduler controls")
