#!/usr/bin/env python3
"""CPU-only contract, numeric, finalizer, and tamper tests for V63."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import build_lora_es_v59_vs_v434_robust_confirmation_preregistration_v63 as builder
import finalize_lora_es_v59_vs_v434_robust_confirmation_v63 as finalizer
import lora_es_v59_vs_v434_robust_confirmation_v63 as analysis
import run_lora_es_v59_vs_v434_robust_confirmation_v63 as runtime


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _metric(f1: float, exact: int = 0) -> dict:
    return {
        "f1": float(f1),
        "exact": int(exact),
        "nonzero": int(float(f1) > 0.0),
    }


def _receipt(period_kind: str, period_index: int) -> dict:
    identities = analysis.expected_adapter_identities_v63()
    return {
        "period_kind": period_kind,
        "period_index": period_index,
        "before": copy.deepcopy(identities),
        "after": copy.deepcopy(identities),
        "actor_request_assignments": runtime._assignments_v63(
            period_kind, period_index
        ),
        "both_adapter_files_exact_and_unchanged": True,
    }


def make_evidence(
    delta: float | list[float] = 0.01,
    *,
    stable_reference_counts: tuple[int, int, int] = (48, 48, 48),
    stable_candidate_counts: tuple[int, int, int] = (48, 48, 48),
    stable_reference_nonzero_counts: tuple[int, int, int] = (48, 48, 48),
    stable_candidate_nonzero_counts: tuple[int, int, int] = (48, 48, 48),
    stress_reference_count: int = 0,
    stress_candidate_count: int = 0,
) -> dict:
    actor_delta = [float(delta)] * 4 if isinstance(delta, (int, float)) else delta
    rows = []
    for index in range(analysis.RANKING_UNITS_V63):
        rows.append({
            "row_sha256": _sha(f"row-{index}"),
            "unit_identity_sha256": _sha(f"unit-{index}"),
            "role": "ranking",
        })
    for sentinel_index, unit_sha in enumerate(analysis.SENTINEL_ORDER_V63):
        rows.append({
            "row_sha256": _sha(f"sentinel-row-{sentinel_index}"),
            "unit_identity_sha256": unit_sha,
            "role": "exact_sentinel",
        })

    label_seen = {
        (sentinel_index, label): 0
        for sentinel_index in range(analysis.EXACT_SENTINEL_UNITS_V63)
        for label in ("reference", "candidate")
    }
    scored = []
    for period_index in range(analysis.SCORED_PERIODS_V63):
        actor_batches = []
        for actor_rank in range(analysis.ACTORS_V63):
            label = analysis.LABEL_PLAN_V63[str(actor_rank)][period_index]
            batch = []
            for row_index in range(analysis.RANKING_UNITS_V63):
                base = 0.4
                value = base + (actor_delta[actor_rank]
                                if label == "candidate" else 0.0)
                batch.append(_metric(value))
            for sentinel_index in range(analysis.EXACT_SENTINEL_UNITS_V63):
                seen = label_seen[(sentinel_index, label)]
                label_seen[(sentinel_index, label)] += 1
                sentinel_sha = analysis.SENTINEL_ORDER_V63[sentinel_index]
                if sentinel_sha in analysis.STABLE_EXACT_UNIT_SHA256_V63:
                    stable_index = analysis.STABLE_EXACT_UNIT_SHA256_V63.index(
                        sentinel_sha
                    )
                    count = (
                        stable_reference_counts[stable_index]
                        if label == "reference"
                        else stable_candidate_counts[stable_index]
                    )
                    nonzero_count = (
                        stable_reference_nonzero_counts[stable_index]
                        if label == "reference"
                        else stable_candidate_nonzero_counts[stable_index]
                    )
                else:
                    count = (
                        stress_reference_count
                        if label == "reference" else stress_candidate_count
                    )
                    nonzero_count = count
                exact = int(seen < count)
                nonzero = int(seen < nonzero_count)
                if exact > nonzero:
                    raise ValueError("exact sentinel count cannot exceed nonzero count")
                batch.append(_metric(1.0 if exact else (0.5 if nonzero else 0.0), exact))
            actor_batches.append(batch)
        scored.append(actor_batches)
    assert all(value == 48 for value in label_seen.values())
    warmup_receipts = [
        _receipt("unscored_warmup", index)
        for index in range(analysis.WARMUP_PERIODS_V63)
    ]
    scored_receipts = [
        _receipt("scored", index)
        for index in range(analysis.SCORED_PERIODS_V63)
    ]
    return runtime.build_evidence_v63(
        rows, scored, warmup_receipts, scored_receipts
    )


@pytest.fixture(scope="session")
def passing_evidence() -> dict:
    return make_evidence(0.01)


@pytest.fixture(scope="session")
def passing_analysis(passing_evidence) -> dict:
    return analysis.build_analysis_v63(passing_evidence)


def _write_self_hashed(path: Path, value: dict) -> finalizer.SelfHashedSourceV63:
    value = copy.deepcopy(value)
    value.pop("content_sha256_before_self_field", None)
    content = analysis.canonical_sha256_v63(value)
    value["content_sha256_before_self_field"] = content
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return finalizer.SelfHashedSourceV63(
        path, runtime.file_sha256_v63(path), content
    )


def _actor_identities() -> list[dict]:
    return [{
        "physical_gpu_id": gpu,
        "pid": 10_000 + gpu,
    } for gpu in range(4)]


def make_finalizer_sources(
    tmp_path: Path,
    evidence: dict,
) -> finalizer.FinalizerSourcesV63:
    prereg = builder.build_v63()
    stored_analysis = analysis.build_analysis_v63(evidence)
    prereg_source = _write_self_hashed(tmp_path / "prereg.json", prereg)
    evidence_source = _write_self_hashed(tmp_path / "evidence.json", evidence)
    analysis_source = _write_self_hashed(
        tmp_path / "analysis.json", stored_analysis
    )
    gate = stored_analysis["required_confirmation_gate"]
    report = {
        "schema": "v63-v59-vs-v434-train-only-confirmation-report",
        "status": (
            "complete_gate_passed_without_promotion_authority"
            if gate["passed"] else "complete_gate_failed_closed"
        ),
        "v62b_finalized": runtime.verify_v62b_eligibility_v63(),
        "adapter_artifact_identities": analysis.expected_adapter_identities_v63(),
        "evidence": {
            "file_sha256": evidence_source.file_sha256,
            "content_sha256": evidence_source.content_sha256,
        },
        "analysis": {
            "file_sha256": analysis_source.file_sha256,
            "content_sha256": analysis_source.content_sha256,
            "required_confirmation_gate": gate,
        },
        "actor_identities": _actor_identities(),
        "generation_only": True,
        "teacher_forced_requests": 0,
        "adaptive_retry_drop_reorder_or_early_stop_performed": False,
        "median_consensus_or_best_of_selection_performed": False,
        "adapter_update_hpo_master_checkpoint_or_promotion_performed": False,
        "holdback_ood_shadow_or_protected_opened": False,
        "raw_question_answer_or_generation_text_persisted": False,
        "result_authorizes_update_hpo_promotion_or_protected_access": False,
    }
    report_source = _write_self_hashed(tmp_path / "report.json", report)
    return finalizer.FinalizerSourcesV63(
        prereg_source, evidence_source, analysis_source, report_source
    )


def test_schedule_is_exact_v62b_eager_design():
    assert analysis.WARMUP_PERIODS_V63 == 4
    assert analysis.SCORED_PERIODS_V63 == 24
    assert analysis.PAIRS_PER_ACTOR_V63 == 12
    assert analysis.REPLICAS_PER_UNIT_V63 == 48
    assert analysis.RUNTIME_CONTROLS_V63 == {
        "VLLM_BATCH_INVARIANT": False,
        "async_scheduling": False,
        "max_num_seqs": 68,
        "scheduling_policy": "fcfs",
        "enforce_eager": True,
        "max_loras": 1,
        "max_cpu_loras": 2,
    }


def test_exact_six_six_order_balance_per_actor():
    generation = np.zeros((64, 4, 24, 3), dtype=np.float64)
    _, receipts = analysis.paired_deltas_v63(generation)
    for actor in range(4):
        actor_receipts = [item for item in receipts if item["actor_rank"] == actor]
        assert sum(item["candidate_after_reference"] for item in actor_receipts) == 6
        assert sum(not item["candidate_after_reference"] for item in actor_receipts) == 6


def test_passing_primary_and_stable_exact_gate(passing_analysis):
    gate = passing_analysis["required_confirmation_gate"]
    assert gate["passed"] is True
    assert gate["robust_fitness"] == pytest.approx(0.01)
    assert all(gate["checks"].values())


def test_point_threshold_is_exactly_twice_frozen_null_width():
    assert analysis.MINIMUM_POINT_IMPROVEMENT_V63 == (
        2 * analysis.FROZEN_NULL_WIDTH_V63
    )


def test_point_threshold_is_inclusive():
    primary = {
        "point": analysis.MINIMUM_POINT_IMPROVEMENT_V63,
        "lcb": 0.01,
    }
    actor = {"maximum_absolute_leave_one_actor_out_shift": 0.0}
    exact = analysis.exact_diagnostics_v63(
        np.ones((4, 4, 24, 3), dtype=np.float64),
        [{"unit_identity_sha256": value} for value in analysis.SENTINEL_ORDER_V63],
    )
    assert analysis.robust_gate_v63(primary, actor, exact)["checks"][
        "point_improvement_at_least_twice_frozen_null_width_inclusive"
    ] is True


def test_robust_fitness_is_strictly_positive():
    exact = analysis.exact_diagnostics_v63(
        np.ones((4, 4, 24, 3), dtype=np.float64),
        [{"unit_identity_sha256": value} for value in analysis.SENTINEL_ORDER_V63],
    )
    gate = analysis.robust_gate_v63(
        {"point": 0.02, "lcb": 0.01},
        {"maximum_absolute_leave_one_actor_out_shift": 0.01},
        exact,
    )
    assert gate["robust_fitness"] == 0.0
    assert gate["checks"][
        "robust_fitness_lcb_minus_max_actor_loo_shift_strictly_positive"
    ] is False


def test_stable_per_unit_noninferiority_is_required():
    evidence = make_evidence(
        0.01,
        stable_reference_counts=(48, 48, 48),
        stable_candidate_counts=(48, 47, 48),
    )
    gate = analysis.build_analysis_v63(evidence)["required_confirmation_gate"]
    assert gate["passed"] is False
    assert gate["checks"][
        "stable_exact_every_unit_candidate_noninferior_inclusive"
    ] is False


def test_stable_nonzero_aggregate_noninferiority_is_required():
    evidence = make_evidence(
        0.01,
        stable_reference_counts=(0, 0, 0),
        stable_candidate_counts=(0, 0, 0),
        stable_reference_nonzero_counts=(48, 48, 48),
        stable_candidate_nonzero_counts=(48, 47, 48),
    )
    gate = analysis.build_analysis_v63(evidence)["required_confirmation_gate"]
    assert gate["checks"][
        "stable_nonzero_aggregate_candidate_noninferior_inclusive"
    ] is False
    assert gate["passed"] is False


def test_stress_sentinel_is_diagnostic_only():
    evidence = make_evidence(
        0.01, stress_reference_count=48, stress_candidate_count=0
    )
    built = analysis.build_analysis_v63(evidence)
    assert built["required_confirmation_gate"]["passed"] is True
    assert built["exact_sentinel_diagnostics"][
        "actor_unstable_stress_unit_used_in_required_gate"
    ] is False


def test_evidence_label_tamper_fails(passing_evidence):
    changed = copy.deepcopy(passing_evidence)
    changed["rows"][0]["scored_periods"][0]["actors"][0]["label"] = "candidate"
    changed["numeric_actor_period_manifest_sha256"] = analysis.canonical_sha256_v63(
        changed["rows"]
    )
    changed.pop("content_sha256_before_self_field")
    changed["content_sha256_before_self_field"] = analysis.canonical_sha256_v63(changed)
    with pytest.raises(ValueError, match="counterbalance"):
        analysis.validate_evidence_v63(changed)


def test_evidence_adapter_receipt_tamper_fails(passing_evidence):
    changed = copy.deepcopy(passing_evidence)
    changed["scored_state_receipts"][0]["before"]["candidate"][
        "canonical_fp32_state_sha256"
    ] = "0" * 64
    changed["numeric_scored_state_receipts_sha256"] = analysis.canonical_sha256_v63(
        changed["scored_state_receipts"]
    )
    changed.pop("content_sha256_before_self_field")
    changed["content_sha256_before_self_field"] = analysis.canonical_sha256_v63(changed)
    with pytest.raises(ValueError, match="adapter identity"):
        analysis.validate_evidence_v63(changed)


def test_evidence_self_hash_tamper_fails(passing_evidence):
    changed = copy.deepcopy(passing_evidence)
    changed["row_count"] = 67
    with pytest.raises(ValueError, match="contract"):
        analysis.validate_evidence_v63(changed)


def test_v62b_eligibility_hashes_and_non_authority_are_exact():
    value = runtime.verify_v62b_eligibility_v63()
    assert value["file_sha256"] == (
        "92b7e847ef42b06735d29d2a3f345a8c1cc8233c8408395de8d7016e9838ae72"
    )
    assert value["content_sha256"] == (
        "f05506bfcca63bf2723b10518708897ab871b1073f654ed8a221608aaacd2149"
    )
    assert value["launch_or_update_authority"] is False


def test_v59_and_v434_adapter_identities_are_exact():
    assert runtime.verify_adapter_artifacts_v63() == (
        analysis.expected_adapter_identities_v63()
    )


def test_static_two_request_support_binds_ids_one_and_two():
    support = runtime.installed_two_adapter_support_v63()
    assert support["status"] == "static_api_supported_live_switch_not_preclaimed"
    assert [item["lora_int_id"] for item in support["request_projection"]] == [1, 2]
    assert support["live_gpu_switch_probe_performed"] is False


def test_engine_kwargs_are_eager_bi_false_and_cpu_cache_two():
    holder = type("V40", (), {"MODEL": runtime.BASE_MODEL})()
    kwargs = runtime.engine_kwargs_v63(holder)
    assert kwargs["enforce_eager"] is True
    assert kwargs["async_scheduling"] is False
    assert kwargs["max_num_seqs"] == 68
    assert kwargs["max_loras"] == 1
    assert kwargs["max_cpu_loras"] == 2


def test_builder_is_cpu_only_and_does_not_claim_support_as_authority():
    value = builder.build_v63()
    assert value["builder_or_dry_run_performed_gpu_launch"] is False
    assert value["eligibility_or_static_support_alone_authorizes_launch"] is False
    assert value["access_contract"][
        "builder_or_dry_run_reads_staged_rows_or_panel"
    ] is False
    assert value[
        "update_hpo_candidate_promotion_or_protected_access_authorized"
    ] is False


def test_dry_run_writes_nothing_and_opens_no_dataset_model_or_gpu(
    tmp_path, capsys
):
    prereg = builder.build_v63()
    prereg_source = _write_self_hashed(tmp_path / "prereg.json", prereg)
    before = set(tmp_path.iterdir())
    assert runtime.main([
        "--preregistration", str(prereg_source.path),
        "--preregistration-sha256", prereg_source.file_sha256,
        "--preregistration-content-sha256", prereg_source.content_sha256,
        "--dry-run",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["staged_train_rows_or_panel_opened"] == 0
    assert payload["base_model_or_gpu_loaded"] is False
    assert payload["filesystem_writes"] is False
    assert set(tmp_path.iterdir()) == before


def test_live_path_requires_explicit_execute(tmp_path):
    prereg_source = _write_self_hashed(
        tmp_path / "prereg.json", builder.build_v63()
    )
    with pytest.raises(RuntimeError, match="requires --execute"):
        runtime.main([
            "--preregistration", str(prereg_source.path),
            "--preregistration-sha256", prereg_source.file_sha256,
            "--preregistration-content-sha256", prereg_source.content_sha256,
        ])


def test_finalizer_production_path_fails_until_outcome_hashes_are_sealed():
    with pytest.raises(RuntimeError, match="production outcome hash is not sealed"):
        finalizer.production_sources_v63("0" * 64, "1" * 64)


def test_finalizer_accepts_passing_outcome_without_authority(
    tmp_path, passing_evidence
):
    result = finalizer.finalize_v63(
        make_finalizer_sources(tmp_path, passing_evidence)
    )
    assert result["status"] == "complete_gate_passed_without_authority"
    assert result["observed_numeric_outcome_without_authorization"][
        "required_confirmation_gate"
    ]["passed"] is True
    assert result["frozen_non_authorization"][
        "adapter_update_hpo_candidate_promotion_authorized"
    ] is False


def test_finalizer_accepts_failed_outcome_without_relaxation(tmp_path):
    evidence = make_evidence(0.001)
    result = finalizer.finalize_v63(make_finalizer_sources(tmp_path, evidence))
    assert result["status"] == "complete_gate_failed_closed"
    assert result["observed_numeric_outcome_without_authorization"][
        "required_confirmation_gate"
    ]["passed"] is False
    assert result["frozen_non_authorization"][
        "failed_gate_reinterpreted_or_relaxed"
    ] is False


def test_finalizer_rejects_file_tamper(tmp_path, passing_evidence):
    sources = make_finalizer_sources(tmp_path, passing_evidence)
    sources.evidence.path.write_text(
        sources.evidence.path.read_text() + "\n"
    )
    with pytest.raises(RuntimeError, match="input file changed"):
        finalizer.finalize_v63(sources)


def test_finalizer_rejects_rehashed_analysis_drift(tmp_path, passing_evidence):
    sources = make_finalizer_sources(tmp_path, passing_evidence)
    changed = json.loads(sources.analysis.path.read_text())
    changed["primary_generated_f1"]["point"] += 0.1
    changed_source = _write_self_hashed(sources.analysis.path, changed)
    report = json.loads(sources.report.path.read_text())
    report["analysis"]["file_sha256"] = changed_source.file_sha256
    report["analysis"]["content_sha256"] = changed_source.content_sha256
    report_source = _write_self_hashed(sources.report.path, report)
    changed_sources = finalizer.FinalizerSourcesV63(
        sources.preregistration,
        sources.evidence,
        changed_source,
        report_source,
    )
    with pytest.raises(RuntimeError, match="differs from exact numeric rebuild"):
        finalizer.finalize_v63(changed_sources)


def test_finalizer_rejects_rehashed_authority_tamper(tmp_path, passing_evidence):
    sources = make_finalizer_sources(tmp_path, passing_evidence)
    report = json.loads(sources.report.path.read_text())
    report["result_authorizes_update_hpo_promotion_or_protected_access"] = True
    report_source = _write_self_hashed(sources.report.path, report)
    changed_sources = finalizer.FinalizerSourcesV63(
        sources.preregistration,
        sources.evidence,
        sources.analysis,
        report_source,
    )
    with pytest.raises(RuntimeError, match="report or integrity"):
        finalizer.finalize_v63(changed_sources)


def test_finalizer_rejects_rehashed_text_leakage(tmp_path, passing_evidence):
    sources = make_finalizer_sources(tmp_path, passing_evidence)
    report = json.loads(sources.report.path.read_text())
    report["prompt"] = "forbidden"
    report_source = _write_self_hashed(sources.report.path, report)
    changed_sources = finalizer.FinalizerSourcesV63(
        sources.preregistration,
        sources.evidence,
        sources.analysis,
        report_source,
    )
    with pytest.raises(RuntimeError, match="forbidden text-bearing key"):
        finalizer.finalize_v63(changed_sources)


def test_finalizer_rejects_rehashed_prereg_gate_tamper(tmp_path, passing_evidence):
    sources = make_finalizer_sources(tmp_path, passing_evidence)
    prereg = json.loads(sources.preregistration.path.read_text())
    prereg["required_confirmation_gates"][
        "stable_nonzero_aggregate_candidate_at_least_reference_inclusive"
    ] = False
    prereg_source = _write_self_hashed(sources.preregistration.path, prereg)
    changed_sources = finalizer.FinalizerSourcesV63(
        prereg_source, sources.evidence, sources.analysis, sources.report
    )
    with pytest.raises(RuntimeError, match="preregistration changed"):
        finalizer.finalize_v63(changed_sources)


def test_analysis_never_authorizes_or_selects(passing_analysis):
    assert passing_analysis[
        "update_hpo_candidate_promotion_or_protected_access_authorized"
    ] is False
    assert passing_analysis[
        "median_consensus_or_best_of_selection_performed"
    ] is False
    assert passing_analysis["required_confirmation_gate"][
        "authorizes_update_hpo_promotion_or_protected_access"
    ] is False
