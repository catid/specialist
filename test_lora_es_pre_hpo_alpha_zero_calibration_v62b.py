#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import types

import numpy as np
import pytest

import audit_vllm_pre_hpo_alpha_zero_support_v62b as support
import build_lora_es_pre_hpo_alpha_zero_preregistration_v62b as builder
import lora_es_pre_hpo_alpha_zero_calibration_v62b as subject
import run_lora_es_pre_hpo_alpha_zero_calibration_v62b as runtime


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _metric(f1: float, exact: int = 0) -> dict:
    return {
        "f1": float(f1),
        "exact": int(exact),
        "nonzero": int(f1 > 0.0),
    }


def _state() -> dict:
    return {
        "canonical_fp32_master_sha256": (
            subject.CANONICAL_FP32_MASTER_SHA256_V62B
        ),
        "canonical_master_identity_sha256": _sha("v434-identity"),
        "four_actor_certificate_sha256": _sha("four-certificates"),
        "bf16_runtime_values_sha256": (
            subject.BF16_RUNTIME_VALUES_SHA256_V62B
        ),
    }


def _receipts(kind: str, count: int) -> list[dict]:
    state = _state()
    return [{
        "period_kind": kind,
        "period_index": period,
        "before": copy.deepcopy(state),
        "after": copy.deepcopy(state),
        "identical_v434_state": True,
    } for period in range(count)]


def _rows() -> list[dict]:
    rows = []
    for request_index in range(subject.ROWS_V62B):
        sentinel = request_index >= subject.RANKING_UNITS_V62B
        rows.append({
            "request_index": request_index,
            "row_sha256": _sha(f"row-{request_index}"),
            "unit_identity_sha256": (
                subject.SENTINEL_ORDER_V62B[
                    request_index - subject.RANKING_UNITS_V62B
                ] if sentinel else _sha(f"ranking-unit-{request_index}")
            ),
            "role": "exact_sentinel" if sentinel else "ranking",
        })
    return rows


def _scored_periods() -> list[list[list[dict]]]:
    return [[[
        _metric(float(request_index >= subject.RANKING_UNITS_V62B),
                int(request_index >= subject.RANKING_UNITS_V62B))
        for request_index in range(subject.ROWS_V62B)
    ] for _actor in range(subject.ACTORS_V62B)]
        for _period in range(subject.SCORED_PERIODS_V62B)]


def _evidence_v62b() -> dict:
    return runtime.build_evidence_v62b(
        _rows(),
        _scored_periods(),
        _receipts("unscored_warmup", subject.WARMUP_PERIODS_V62B),
        _receipts("scored", subject.SCORED_PERIODS_V62B),
    )


def _reself(value: dict) -> dict:
    value = copy.deepcopy(value)
    value.pop("content_sha256_before_self_field", None)
    value["content_sha256_before_self_field"] = (
        subject.canonical_sha256_v62b(value)
    )
    return value


def _actor_identities() -> list[dict]:
    return [{
        "schema": "pre-hpo-alpha-zero-actor-identity-v62a",
        "pid": 1000 + gpu,
        "physical_gpu_id": gpu,
        "cuda_visible_devices": str(gpu),
        "cuda_current_device": 0,
        "VLLM_BATCH_INVARIANT": False,
        "async_scheduling": False,
        "max_num_seqs": 68,
        "scheduling_policy": "fcfs",
        "scheduler_class": "Scheduler",
        "enforce_eager": True,
        "tuned_folder": str(runtime.design_v52.RUNTIME_V52["tuned_folder"]),
        "tuned_table_content_sha256": (
            runtime.design_v52.RUNTIME_V52["tuned_table_content_sha256"]
        ),
        "submitted_request_batch_size": 68,
        "generation_only": True,
        "global_batch_invariance_claimed": False,
    } for gpu in range(4)]


def test_v62b_fixed_schedule_is_exactly_balanced_and_sized():
    assert subject.WARMUP_PERIODS_V62B == 4
    assert subject.SCORED_BLOCKS_V62B == 6
    assert subject.SCORED_PERIODS_V62B == 24
    assert subject.TOTAL_PERIODS_V62B == 28
    assert subject.PAIRS_PER_ACTOR_V62B == 12
    assert subject.REPLICAS_PER_UNIT_V62B == 48
    assert subject.WARMUP_GENERATION_COMPLETIONS_V62B == 1088
    assert subject.SCORED_GENERATION_COMPLETIONS_V62B == 6528
    assert subject.TOTAL_GENERATION_COMPLETIONS_V62B == 7616
    assert subject.WARMUP_LABEL_PLAN_V62B == subject.BLOCK_LABEL_PLAN_V62B
    assert len(subject.PAIR_PERIODS_V62B) == 12
    for actor in range(4):
        labels = subject.LABEL_PLAN_V62B[str(actor)]
        after = 0
        before = 0
        for left, right in subject.PAIR_PERIODS_V62B:
            assert {labels[left], labels[right]} == {"reference", "candidate"}
            candidate = left if labels[left] == "candidate" else right
            reference = left if labels[left] == "reference" else right
            after += candidate > reference
            before += candidate < reference
        assert (after, before) == (6, 6)


def test_v62b_support_audit_binds_exact_runtime_and_full_schedule():
    value = runtime._read_support_audit_v62b()
    assert value["status"] == "supported"
    assert value["support_audit_authorizes_gpu_launch"] is False
    assert value["requested_runtime_controls"] == subject.RUNTIME_CONTROLS_V62B
    intended = value["intended_evaluator_projection"]
    assert intended["actors"] == 4
    assert intended["rows_per_actor_call"] == 68
    assert intended["unscored_warmup_periods"] == 4
    assert intended["scored_periods"] == 24
    assert intended["scored_replicas_per_conflict_unit"] == 48
    assert intended["warmup_outputs_scored_or_persisted"] is False
    assert support.build_audit_v62b()["status"] == "supported"


def test_v62b_primary_averages_48_then_resamples_only_conflict_units():
    zeros = np.zeros((64, 4, 12), dtype=np.float64)
    primary = subject.primary_f1_bootstrap_v62b(zeros)
    assert primary["within_unit_actor_pair_replicas_preserved_and_averaged"] == 48
    assert primary["resampled_axis"] == "conflict_unit"
    assert primary["bootstrap_replicates"] == 4096
    assert primary["bootstrap_seed"] == 2026071612
    assert primary["point"] == primary["lcb"] == primary["ucb"] == 0.0
    assert primary["contains_zero"] is True

    actor_values = zeros.copy()
    actor_values[:, 0, :] = 1.0
    influence = subject.actor_influence_v62b(actor_values)
    assert influence["full_four_actor_point"] == 0.25
    assert influence["leave_one_actor_out_points"][0] == 0.0
    assert influence["maximum_absolute_leave_one_actor_out_shift"] == 0.25


def test_v62b_three_gates_and_bootstrap_thresholds_are_unchanged_and_inclusive():
    assert subject.MAX_PRIMARY_CI_HALFWIDTH_V62B == 0.000773822590292528
    assert subject.MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V62B == 0.0012119648781783704
    primary = {
        "contains_zero": True,
        "halfwidth": subject.MAX_PRIMARY_CI_HALFWIDTH_V62B,
    }
    actor = {
        "maximum_absolute_leave_one_actor_out_shift": (
            subject.MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V62B
        ),
    }
    assert subject.gate_v62b(primary, actor)["passed"] is True
    changed = copy.deepcopy(primary)
    changed["halfwidth"] = np.nextafter(
        subject.MAX_PRIMARY_CI_HALFWIDTH_V62B,
        np.inf,
    )
    assert subject.gate_v62b(changed, actor)["passed"] is False
    changed = copy.deepcopy(primary)
    changed["contains_zero"] = False
    assert subject.gate_v62b(changed, actor)["passed"] is False


def test_v62b_evidence_excludes_warmup_and_sentinels_remain_diagnostic_only():
    evidence = _evidence_v62b()
    assert len(subject.validate_evidence_v62b(evidence)) == 68
    assert "warmup_periods" not in evidence["rows"][0]
    assert evidence["warmup_raw_outputs_persisted"] is False
    assert evidence["warmup_generation_metrics_computed_or_persisted"] is False
    result = subject.build_analysis_v62b(evidence)
    assert result["unscored_warmup_excluded_from_every_metric"] is True
    assert result["required_pre_hpo_gate"]["passed"] is True
    exact = result["exact_sentinel_diagnostics"]
    assert exact["stable_panel"]["unit_count"] == 3
    assert exact["stable_panel"]["candidate_exact_pass_total_of_144"] == 144
    assert exact["actor_unstable_stress_unit"]["role"] == (
        "actor_unstable_stress_diagnostic_only"
    )
    assert exact["used_in_alpha_zero_gate"] is False
    assert exact["any_single_flip_aborts"] is False
    assert exact["any_per_unit_all_replicas_failure_aborts"] is False


@pytest.mark.parametrize("tamper", [
    "missing_warmup_receipt",
    "warmup_state_drift",
    "missing_scored_receipt",
    "scored_state_drift",
    "counterbalance_label",
    "warmup_metric_field",
    "scored_early_stop_flag",
])
def test_v62b_rehashed_evidence_tampering_fails_closed(tamper):
    value = _evidence_v62b()
    if tamper == "missing_warmup_receipt":
        value["warmup_state_receipts"].pop()
        value["numeric_warmup_state_receipts_sha256"] = (
            subject.canonical_sha256_v62b(value["warmup_state_receipts"])
        )
    elif tamper == "warmup_state_drift":
        value["warmup_state_receipts"][0]["after"][
            "canonical_master_identity_sha256"
        ] = _sha("changed")
        value["numeric_warmup_state_receipts_sha256"] = (
            subject.canonical_sha256_v62b(value["warmup_state_receipts"])
        )
    elif tamper == "missing_scored_receipt":
        value["scored_state_receipts"].pop()
        value["numeric_scored_state_receipts_sha256"] = (
            subject.canonical_sha256_v62b(value["scored_state_receipts"])
        )
    elif tamper == "scored_state_drift":
        value["scored_state_receipts"][1]["after"][
            "canonical_master_identity_sha256"
        ] = _sha("changed")
        value["numeric_scored_state_receipts_sha256"] = (
            subject.canonical_sha256_v62b(value["scored_state_receipts"])
        )
    elif tamper == "counterbalance_label":
        value["rows"][0]["scored_periods"][0]["actors"][0]["label"] = (
            "candidate"
        )
        value["numeric_actor_period_manifest_sha256"] = (
            subject.canonical_sha256_v62b(value["rows"])
        )
    elif tamper == "warmup_metric_field":
        value["warmup_generation_metrics"] = []
    elif tamper == "scored_early_stop_flag":
        value[
            "scored_period_adaptive_retry_drop_reorder_or_early_stop_performed"
        ] = True
    with pytest.raises(ValueError):
        subject.validate_evidence_v62b(_reself(value))


def test_v62b_runtime_warmup_is_discard_only_and_scored_loop_is_fixed():
    source = inspect.getsource(runtime.main)
    warmup = source.split("warmup_state_receipts = []", 1)[1].split(
        "scored_periods = []", 1
    )[0]
    scored = source.split("scored_periods = []", 1)[1].split(
        "evidence = build_evidence_v62b", 1
    )[0]
    assert "range(analysis.WARMUP_PERIODS_V62B)" in warmup
    assert "score_generation_batch_v61c" not in warmup
    assert "del warmup_batches" in warmup
    assert "range(analysis.SCORED_PERIODS_V62B)" in scored
    assert "score_generation_batch_v61c" in scored
    assert "break" not in scored
    assert "score_teacher_batch" not in source
    assert "prepare_gold_answer" not in source


def test_v62b_actor_batch_utilization_and_cleanup_tampering_fails_closed():
    actor_ids = _actor_identities()
    tuned = runtime.design_v52.RUNTIME_V52["tuned_table_content_sha256"]
    pid_map = runtime._validate_actor_identities_v62b(actor_ids, tuned)
    assert pid_map == {gpu: 1000 + gpu for gpu in range(4)}

    changed = copy.deepcopy(actor_ids)
    changed[1]["pid"] = changed[0]["pid"]
    with pytest.raises(RuntimeError):
        runtime._validate_actor_identities_v62b(changed, tuned)
    changed = copy.deepcopy(actor_ids)
    changed[2]["enforce_eager"] = False
    with pytest.raises(RuntimeError):
        runtime._validate_actor_identities_v62b(changed, tuned)

    batches = [[object()] * 68 for _ in range(4)]
    assert runtime._validate_batches_v62b(batches, "scored") is batches
    with pytest.raises(RuntimeError):
        runtime._validate_batches_v62b(batches[:3], "scored")
    changed_batches = copy.deepcopy(batches)
    changed_batches[0].pop()
    with pytest.raises(RuntimeError):
        runtime._validate_batches_v62b(changed_batches, "unscored_warmup")

    gpu = {
        "all_four_attributed_positive": True,
        "by_gpu": {
            str(index): {
                "expected_pid": pid_map[index],
                "positive_samples": 1,
            } for index in range(4)
        },
    }
    cleanup = {
        "schema": "eggroll-es-placement-group-cleanup-v38a",
        "engine_kill_count": 4,
        "placement_group_remove_count": 4,
        "all_four_gcs_states_removed": True,
    }
    idle = {"all_four_compute_process_lists_empty": True}
    runtime._validate_postrun_integrity_v62b(gpu, cleanup, idle, pid_map)
    changed_gpu = copy.deepcopy(gpu)
    changed_gpu["by_gpu"]["3"]["positive_samples"] = 0
    with pytest.raises(RuntimeError):
        runtime._validate_postrun_integrity_v62b(
            changed_gpu, cleanup, idle, pid_map,
        )
    changed_cleanup = copy.deepcopy(cleanup)
    changed_cleanup["engine_kill_count"] = 3
    with pytest.raises(RuntimeError):
        runtime._validate_postrun_integrity_v62b(
            gpu, changed_cleanup, idle, pid_map,
        )
    with pytest.raises(RuntimeError):
        runtime._validate_postrun_integrity_v62b(
            gpu, cleanup, {"all_four_compute_process_lists_empty": False}, pid_map,
        )


def test_v62b_preregistration_and_dry_run_are_closed_and_tamper_resistant(
    tmp_path,
    monkeypatch,
    capsys,
):
    value = builder.build_v62b()
    assert value["v62_methodology_commit"] == (
        "3878fb4d85ddbb1fb96d382d1af21446bc8764c0"
    )
    recipe = value["fixed_calibration_recipe"]
    assert recipe["unscored_warmup_periods"] == 4
    assert recipe["scored_counterbalanced_blocks"] == 6
    assert recipe["scored_sequential_periods"] == 24
    assert recipe["counterbalanced_pairs_per_actor"] == 12
    assert recipe["replicas_per_conflict_unit"] == 48
    assert recipe["candidate_after_reference_pairs_per_actor"] == 6
    assert recipe["candidate_before_reference_pairs_per_actor"] == 6
    assert recipe["warmup_outputs_scored_or_persisted"] is False
    assert recipe["all_scored_periods_included"] is True
    assert value["required_alpha_zero_gates"] == runtime.required_gates_v62b()
    assert value["hpo_population_update_or_candidate_authorized"] is False

    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    file_sha = runtime.runtime_v61a.file_sha256_v61a(path)
    content_sha = value["content_sha256_before_self_field"]
    args = types.SimpleNamespace(
        preregistration=str(path),
        preregistration_sha256=file_sha,
        preregistration_content_sha256=content_sha,
    )
    assert runtime.load_preregistration_v62b(args) == value

    tampered = copy.deepcopy(value)
    tampered["fixed_calibration_recipe"]["scored_sequential_periods"] = 23
    tampered = _reself(tampered)
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n")
    tampered_args = types.SimpleNamespace(
        preregistration=str(tampered_path),
        preregistration_sha256=runtime.runtime_v61a.file_sha256_v61a(
            tampered_path
        ),
        preregistration_content_sha256=tampered[
            "content_sha256_before_self_field"
        ],
    )
    with pytest.raises(RuntimeError):
        runtime.load_preregistration_v62b(tampered_args)

    monkeypatch.setattr(
        runtime.runtime_v61c,
        "load_staged_inputs_v61c",
        lambda: (_ for _ in ()).throw(
            AssertionError("v62b dry-run opened staged train data")
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
    assert output["staged_train_rows_opened"] == 0
    assert output["unscored_warmup_periods"] == 4
    assert output["scored_periods"] == 24
    assert output["scored_replicas_per_conflict_unit"] == 48
    assert output["warmup_outputs_scored_or_persisted"] is False
    assert output["model_or_gpu_loaded"] is False
    assert output["filesystem_writes"] is False
    assert output["hpo_update_candidate_or_protected_access"] is False
