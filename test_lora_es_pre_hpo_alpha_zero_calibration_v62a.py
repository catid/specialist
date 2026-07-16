#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import types

import numpy as np

import audit_vllm_pre_hpo_alpha_zero_support_v62a as support
import build_lora_es_pre_hpo_alpha_zero_preregistration_v62a as builder
import lora_es_pre_hpo_alpha_zero_calibration_v62a as subject
import run_lora_es_pre_hpo_alpha_zero_calibration_v62a as runtime


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _metric(f1: float, exact: int = 0) -> dict:
    return {
        "f1": float(f1),
        "exact": int(exact),
        "nonzero": int(f1 > 0.0),
    }


def _evidence_v62a() -> dict:
    state_receipts = []
    for period in range(4):
        state = {
            "canonical_fp32_master_sha256": (
                subject.CANONICAL_FP32_MASTER_SHA256_V62A
            ),
            "canonical_master_identity_sha256": _sha("v434-identity"),
            "four_actor_certificate_sha256": _sha("four-certificates"),
            "bf16_runtime_values_sha256": (
                subject.BF16_RUNTIME_VALUES_SHA256_V62A
            ),
        }
        state_receipts.append({
            "period_index": period,
            "before": copy.deepcopy(state),
            "after": copy.deepcopy(state),
            "identical_v434_state": True,
        })
    rows = []
    for request_index in range(68):
        sentinel = request_index >= 64
        unit_sha = (
            subject.SENTINEL_ORDER_V62A[request_index - 64]
            if sentinel else _sha(f"ranking-unit-{request_index}")
        )
        periods = []
        for period in range(4):
            actors = []
            for actor in range(4):
                exact = int(sentinel)
                actors.append({
                    "actor_rank": actor,
                    "label": subject.LABEL_PLAN_V62A[str(actor)][period],
                    "generation": _metric(float(exact), exact),
                })
            periods.append({
                "period_index": period,
                "request_type": "generation",
                "actors": actors,
            })
        rows.append({
            "request_index": request_index,
            "row_sha256": _sha(f"row-{request_index}"),
            "unit_identity_sha256": unit_sha,
            "role": "exact_sentinel" if sentinel else "ranking",
            "periods": periods,
        })
    value = {
        "schema": "v62a-pre-hpo-alpha-zero-generation-only-evidence",
        "status": "complete_alpha_zero_generation_only_characterization",
        "v62_methodology_commit": subject.V62_METHOD_COMMIT,
        "v62_numeric_audit_identities": dict(
            subject.V62_NUMERIC_AUDIT_IDENTITIES
        ),
        "v62_preregistration_identities": dict(
            subject.V62_PREREGISTRATION_IDENTITIES
        ),
        "staged_dataset_file_sha256": (
            runtime.runtime_v61c.STAGED_DATASET_FILE_SHA256
        ),
        "staged_panel_file_sha256": (
            runtime.runtime_v61c.STAGED_PANEL_FILE_SHA256
        ),
        "staged_panel_content_sha256": (
            runtime.runtime_v61c.STAGED_PANEL_CONTENT_SHA256
        ),
        "canonical_fp32_master_sha256": runtime.design_v52.MASTER_SHA256_V52,
        "bf16_runtime_values_sha256": (
            runtime.design_v52.MASTER_RUNTIME_SHA256_V52
        ),
        "row_count": 68,
        "ranking_units": 64,
        "exact_sentinel_units": 4,
        "actor_count": 4,
        "period_count": 4,
        "pairs_per_actor": 2,
        "replicas_per_unit": 8,
        "label_plan": copy.deepcopy(subject.LABEL_PLAN_V62A),
        "pair_periods": [list(pair) for pair in subject.PAIR_PERIODS_V62A],
        "common_generation_seed": subject.COMMON_GENERATION_SEED_V62A,
        "generation_params_without_seed": copy.deepcopy(
            subject.GENERATION_PARAMS_WITHOUT_SEED_V62A
        ),
        "runtime_determinism_controls": copy.deepcopy(
            subject.RUNTIME_CONTROLS_V62A
        ),
        "state_receipts": state_receipts,
        "numeric_state_receipts_sha256": subject.canonical_sha256_v62a(
            state_receipts
        ),
        "rows": rows,
        "numeric_actor_period_manifest_sha256": (
            subject.canonical_sha256_v62a(rows)
        ),
        "generation_only": True,
        "generation_completions": 1088,
        "teacher_forced_requests": 0,
        "alpha": 0.0,
        "adapter_update_candidate_or_hpo_performed": False,
        "holdback_ood_shadow_or_protected_opened": False,
        "raw_question_answer_or_generation_text_persisted": False,
    }
    value["content_sha256_before_self_field"] = (
        subject.canonical_sha256_v62a(value)
    )
    return value


def _reself(value: dict) -> dict:
    value = copy.deepcopy(value)
    value.pop("content_sha256_before_self_field", None)
    value["content_sha256_before_self_field"] = (
        subject.canonical_sha256_v62a(value)
    )
    return value


def test_v62a_support_audit_binds_actual_sync_fullbatch_generation_runtime():
    value = runtime._read_support_audit_v62a()
    assert value["pre_hpo_alpha_zero_runtime_supported"] is True
    assert value["support_audit_authorizes_gpu_launch"] is False
    assert value["requested_runtime_controls"] == subject.RUNTIME_CONTROLS_V62A
    intended = value["intended_evaluator_projection"]
    assert intended["actors"] == 4
    assert intended["tensor_parallel_size_per_actor"] == 1
    assert intended["rows_per_actor_call"] == 68
    assert intended["generation_only"] is True
    assert intended["teacher_forced_requests"] == 0
    assert support.build_audit_v62a()["status"] == "supported"


def test_v62a_primary_averages_all_eight_then_bootstraps_units_and_uses_loo():
    zeros = np.zeros((64, 4, 2), dtype=np.float64)
    primary = subject.primary_f1_bootstrap_v62a(zeros)
    assert primary["within_unit_actor_pair_replicas_preserved_and_averaged"] == 8
    assert primary["resampled_axis"] == "conflict_unit"
    assert primary["bootstrap_replicates"] == 4096
    assert primary["bootstrap_seed"] == 2026071612
    assert primary["point"] == primary["lcb"] == primary["ucb"] == 0.0
    assert primary["contains_zero"] is True

    actor_values = zeros.copy()
    actor_values[:, 0, :] = 1.0
    influence = subject.actor_influence_v62a(actor_values)
    assert influence["full_four_actor_point"] == 0.25
    assert influence["leave_one_actor_out_points"][0] == 0.0
    assert influence["maximum_absolute_leave_one_actor_out_shift"] == 0.25


def test_v62a_gate_is_exactly_frozen_and_inclusive():
    primary = {
        "contains_zero": True,
        "halfwidth": subject.MAX_PRIMARY_CI_HALFWIDTH_V62A,
    }
    actor = {
        "maximum_absolute_leave_one_actor_out_shift": (
            subject.MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V62A
        ),
    }
    gate = subject.gate_v62a(primary, actor)
    assert gate["passed"] is True
    assert gate["maximum_primary_ci_halfwidth_inclusive"] == (
        0.000773822590292528
    )
    assert gate["maximum_actor_leave_one_out_shift_inclusive"] == (
        0.0012119648781783704
    )
    changed = copy.deepcopy(primary)
    changed["halfwidth"] = np.nextafter(
        subject.MAX_PRIMARY_CI_HALFWIDTH_V62A,
        np.inf,
    )
    assert subject.gate_v62a(changed, actor)["passed"] is False


def test_v62a_generation_only_evidence_and_exact_roles_are_diagnostic_only():
    evidence = _evidence_v62a()
    assert len(subject.validate_evidence_v62a(evidence)) == 68
    result = subject.build_analysis_v62a(evidence)
    assert result["required_pre_hpo_gate"]["passed"] is True
    assert result["teacher_forced_metric_computed"] is False
    exact = result["exact_sentinel_diagnostics"]
    assert exact["stable_panel"]["unit_count"] == 3
    assert exact["stable_panel"]["candidate_exact_pass_total_of_24"] == 24
    assert exact["actor_unstable_stress_unit"]["unit_identity_sha256"] == (
        subject.ACTOR_UNSTABLE_EXACT_UNIT_SHA256_V62A
    )
    assert exact["actor_unstable_stress_unit"]["role"] == (
        "actor_unstable_stress_diagnostic_only"
    )
    assert exact["used_in_alpha_zero_gate"] is False
    assert exact["any_single_flip_aborts"] is False
    assert exact["any_per_unit_eight_of_eight_failure_aborts"] is False

    changed = copy.deepcopy(evidence)
    changed["rows"][0]["periods"][0]["actors"][0]["teacher_forced"] = {}
    changed = _reself(changed)
    try:
        subject.validate_evidence_v62a(changed)
    except ValueError as error:
        assert "counterbalance" in str(error)
    else:
        raise AssertionError("v62a accepted a teacher-forced evidence field")


def test_v62a_runtime_is_four_tp1_same_call_generation_only():
    fake = types.SimpleNamespace(MODEL="model")
    kwargs = runtime.engine_kwargs_v62a(fake)
    assert kwargs["tensor_parallel_size"] == 1
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
    trainer_source = inspect.getsource(runtime._make_trainer_v62a)
    live_source = inspect.getsource(runtime.main)
    for check in (
        'batch_invariant_env != "0"',
        "vllm_envs.VLLM_BATCH_INVARIANT is not False",
        "scheduler.async_scheduling is not False",
        "scheduler.max_num_seqs != 68",
        'scheduler.policy != "fcfs"',
        "config.model_config.enforce_eager is not True",
        "tuned_table_content_sha256",
    ):
        assert check in trainer_source
    assert "same-call 64+4 generation coverage changed" in live_source
    assert "score_generation_batch_v61c" in live_source
    assert "score_teacher_batch" not in live_source
    assert "prepare_gold_answer" not in live_source


def test_v62a_preregistration_binds_v62_and_exact_dry_run_is_closed(
    tmp_path,
    monkeypatch,
    capsys,
):
    value = builder.build_v62a()
    assert value["v62_methodology_commit"] == (
        "3878fb4d85ddbb1fb96d382d1af21446bc8764c0"
    )
    assert value["v62_numeric_audit_identities"] == (
        subject.V62_NUMERIC_AUDIT_IDENTITIES
    )
    assert value["v62_preregistration_identities"] == (
        subject.V62_PREREGISTRATION_IDENTITIES
    )
    assert value["specific_alpha_zero_calibration_gpu_launch_authorized"]
    assert value["hpo_population_update_or_candidate_authorized"] is False
    recipe = value["fixed_calibration_recipe"]
    assert recipe["actors"] == 4
    assert recipe["tensor_parallel_size_per_actor"] == 1
    assert recipe["submitted_requests_per_actor_call"] == 68
    assert recipe["generation_only"] is True
    assert recipe["teacher_forced_requests"] == 0
    gates = value["required_alpha_zero_gates"]
    assert gates["generated_f1_primary_interval_must_contain_zero"] is True
    assert gates["maximum_primary_ci_halfwidth_inclusive"] == (
        subject.MAX_PRIMARY_CI_HALFWIDTH_V62A
    )
    assert gates["maximum_actor_leave_one_out_shift_inclusive"] == (
        subject.MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V62A
    )

    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    file_sha = runtime.runtime_v61a.file_sha256_v61a(path)
    content_sha = value["content_sha256_before_self_field"]
    args = types.SimpleNamespace(
        preregistration=str(path),
        preregistration_sha256=file_sha,
        preregistration_content_sha256=content_sha,
    )
    assert runtime.load_preregistration_v62a(args) == value
    monkeypatch.setattr(
        runtime.runtime_v61c,
        "load_staged_inputs_v61c",
        lambda: (_ for _ in ()).throw(
            AssertionError("v62a dry-run opened staged train data")
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
    assert output["v62_methodology_commit"] == subject.V62_METHOD_COMMIT
    assert output["staged_train_rows_opened"] == 0
    assert output["generation_only"] is True
    assert output["teacher_forced_requests"] == 0
    assert output["model_or_gpu_loaded"] is False
    assert output["filesystem_writes"] is False
    assert output["hpo_update_candidate_or_protected_access"] is False
