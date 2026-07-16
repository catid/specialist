#!/usr/bin/env python3

import copy
import hashlib
import json
import math
import types

import numpy as np

import build_lora_es_paired_null_inputs_v61c as input_builder
import build_lora_es_paired_null_preregistration_v61c as prereg_builder
import lora_es_paired_null_calibration_v61c as subject
import run_lora_es_paired_null_calibration_v61c as runtime


def _sha(value: int) -> str:
    return f"{value:064x}"


def _generation(f1: float = 0.5) -> dict:
    return {
        "f1": f1,
        "exact": int(f1 == 1.0),
        "nonzero": int(f1 > 0.0),
    }


def _evidence() -> dict:
    rows = []
    for request_index in range(68):
        periods = []
        for period_index in range(4):
            actors = []
            for actor_rank in range(4):
                actors.append({
                    "actor_rank": actor_rank,
                    "label": subject.LABEL_PLAN_V61C[str(actor_rank)][period_index],
                    "generation": _generation(),
                    "teacher_forced": {
                        "mean_answer_token_logprob": -1.0,
                        "answer_token_count": 2,
                        "numeric_example_sha256": _sha(
                            1_000_000 + request_index * 16
                            + period_index * 4 + actor_rank
                        ),
                    },
                })
            periods.append({
                "period_index": period_index,
                "request_type_order": subject.REQUEST_TYPE_ORDER_V61C[
                    str(period_index)
                ],
                "actors": actors,
            })
        rows.append({
            "request_index": request_index,
            "row_sha256": _sha(request_index + 1),
            "unit_identity_sha256": _sha(10_000 + request_index),
            "role": "ranking" if request_index < 64 else "exact_sentinel",
            "periods": periods,
        })
    return {
        "schema": "v61c-identical-state-paired-evaluator-evidence",
        "status": "complete_alpha_zero_no_update_characterization",
        "row_count": 68,
        "ranking_units": 64,
        "exact_sentinel_units": 4,
        "actor_count": 4,
        "period_count": 4,
        "label_plan": copy.deepcopy(subject.LABEL_PLAN_V61C),
        "request_type_order": copy.deepcopy(subject.REQUEST_TYPE_ORDER_V61C),
        "common_generation_seed": subject.COMMON_GENERATION_SEED_V61C,
        "generation_params_without_seed": copy.deepcopy(
            subject.GENERATION_PARAMS_WITHOUT_SEED_V61C
        ),
        "teacher_forced_params_without_seed": copy.deepcopy(
            subject.TEACHER_FORCED_PARAMS_WITHOUT_SEED_V61C
        ),
        "alpha": 0.0,
        "adapter_update_or_candidate_materialization_performed": False,
        "holdback_semantics_opened": False,
        "raw_question_answer_or_generation_text_persisted": False,
        "protected_semantics_opened": False,
        "rows": rows,
        "content_sha256_before_self_field": _sha(999_999),
    }


def test_v61c_primary_cluster_bootstrap_preserves_all_replicas_and_zero():
    zero_gen = np.zeros((64, 4, 2, 3), dtype=np.float64)
    zero_teacher = np.zeros((64, 4, 2), dtype=np.float64)
    zero = subject.paired_block_bootstrap_v61c(
        zero_gen, zero_teacher, replicates=256,
    )
    primary = zero["primary_conflict_unit_cluster_bootstrap"]
    assert primary["within_unit_actor_pair_replicas_preserved_and_averaged"] == 8
    assert set(zero["point"].values()) == {0.0}
    assert all(
        set(interval.values()) == {0.0, True}
        for interval in primary["intervals"].values()
    )

    # A shift common to every actor/pair replica is the actual eight-replica
    # unit estimand and must remain exactly .125 after unit resampling.
    shifted_gen = zero_gen.copy(); shifted_teacher = zero_teacher.copy()
    shifted_gen[..., 0] = 0.125; shifted_teacher[...] = 0.002
    shifted = subject.paired_block_bootstrap_v61c(
        shifted_gen, shifted_teacher, replicates=256,
    )
    assert shifted["point"]["generated_f1_delta"] == 0.125
    assert math.isclose(
        shifted["point"]["teacher_forced_logprob_delta"], 0.002,
        rel_tol=0.0, abs_tol=1e-15,
    )
    assert shifted["primary_conflict_unit_cluster_bootstrap"]["intervals"][
        "generated_f1_delta"
    ] == {
        "lcb": 0.125, "ucb": 0.125, "halfwidth": 0.0,
        "contains_zero": False,
    }


def test_v61c_identical_evidence_is_zero_and_logprob_primary_eligible():
    result = subject.build_analysis_v61c(_evidence())
    assert set(result["ranking_bootstrap"]["point"].values()) == {0.0}
    assert result["noise_scale_comparison"][
        "teacher_forced_logprob_primary_eligible"
    ] is True
    assert result["exact_sentinel"]["passed"] is True
    assert result["exact_sentinel"][
        "nonzero_individual_paired_exact_delta_count"
    ] == 0


def test_v61c_exact_sentinel_rejects_cancelling_individual_label_changes():
    evidence = _evidence()
    sentinel = evidence["rows"][64]
    # actor 0 pair 0: +1 exact; actor 1 pair 0: -1 exact.  Aggregate and
    # per-unit sums cancel to zero, but the strict individual check must fail.
    sentinel["periods"][1]["actors"][0]["generation"] = _generation(1.0)
    sentinel["periods"][1]["actors"][1]["generation"] = _generation(1.0)
    result = subject.build_analysis_v61c(evidence)
    checks = result["exact_sentinel"]["checks"]
    assert checks["zero_total_paired_exact_delta"] is True
    assert checks["zero_per_unit_paired_exact_delta"] is True
    assert checks["zero_every_individual_paired_exact_delta"] is False
    assert result["exact_sentinel"][
        "nonzero_individual_paired_exact_delta_count"
    ] == 2
    assert result["exact_sentinel"][
        "maximum_absolute_individual_paired_exact_delta"
    ] == 1.0
    assert result["exact_sentinel"]["passed"] is False


def test_v61c_staged_inputs_bind_corrected_adaptive_preview_without_holdback():
    dataset_payload, panel = input_builder.build_inputs_v61c()
    assert hashlib.sha256(dataset_payload).hexdigest() == (
        "9c1b7f69595cf70ef045259e2097c39546e9f1d84a6b0870fcb14e987655079a"
    )
    assert panel["source"]["preview_file_sha256"] == (
        "a9ce060ce81df5b1fbddcc40db572fe56974ea6dfb6ef2e6ebf3e81925a400e2"
    )
    assert panel["adaptive_design_provenance"] == {
        "v61a_baseline_model_outcomes_used_for_train_only_stratification": True,
        "future_candidate_outcomes_used_for_panel_selection": False,
        "protected_or_holdback_outcomes_used": False,
        "train_only_adaptive_design": True,
    }
    assert panel["holdback_units_in_runtime_dataset"] == 0
    assert panel["holdback_documents_in_runtime_dataset"] == 0
    assert panel["document_block_audit"][
        "runtime_selected_holdback_document_intersection"
    ] == 0


def test_v61c_preregistration_freezes_primary_estimand_and_dry_run_opens_nothing(
    tmp_path, monkeypatch, capsys,
):
    value = prereg_builder.build_v61c()
    primary = value["paired_analysis_contract"][
        "primary_conflict_unit_cluster_bootstrap"
    ]
    assert primary["within_unit_actor_pair_replicas_preserved_and_averaged"] == 8
    assert primary["used_for_logprob_primary_eligibility"] is True
    diagnostic = value["paired_analysis_contract"][
        "future_single_replica_noise_diagnostic"
    ]
    assert diagnostic["not_the_intended_eight_replica_hpo_estimator"] is True
    assert diagnostic["used_for_logprob_primary_eligibility"] is False
    assert value["paired_analysis_contract"]["sparse_exact_sentinel"][
        "any_individual_paired_exact_label_delta_fails"
    ] is True
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    args = types.SimpleNamespace(
        preregistration=str(path),
        preregistration_sha256=runtime.runtime_v61a.file_sha256_v61a(path),
        preregistration_content_sha256=value["content_sha256_before_self_field"],
    )
    assert runtime.load_preregistration_v61c(args) == value
    monkeypatch.setattr(
        runtime, "load_staged_inputs_v61c",
        lambda: (_ for _ in ()).throw(AssertionError("dry-run opened staged rows")),
    )
    assert runtime.main([
        "--preregistration", str(path),
        "--preregistration-sha256", args.preregistration_sha256,
        "--preregistration-content-sha256",
        args.preregistration_content_sha256,
        "--dry-run",
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["staged_train_rows_opened"] == 0
    assert output["full_v52_train_or_membership_opened"] is False
    assert output["model_or_gpu_loaded"] is False
    assert output["filesystem_writes"] is False


def test_v61c_evidence_builder_persists_only_hashes_and_numeric_metrics():
    source = _evidence()
    rows = [{
        "request_index": row["request_index"],
        "row_sha256": row["row_sha256"],
        "unit_identity_sha256": row["unit_identity_sha256"],
        "role": row["role"],
        "question": "synthetic forbidden question",
        "answer": "synthetic forbidden answer",
    } for row in source["rows"]]
    periods = []
    for period_index in range(4):
        periods.append({
            "generation": [[
                copy.deepcopy(row["periods"][period_index]["actors"][actor][
                    "generation"
                ]) for row in source["rows"]
            ] for actor in range(4)],
            "teacher_forced": [[
                copy.deepcopy(row["periods"][period_index]["actors"][actor][
                    "teacher_forced"
                ]) for row in source["rows"]
            ] for actor in range(4)],
        })
    receipts = [{
        "period_index": period,
        "before": {"four_actor_certificate_sha256": _sha(20_000 + period)},
        "after": {"four_actor_certificate_sha256": _sha(20_000 + period)},
        "identical_v434_state": True,
    } for period in range(4)]
    evidence = runtime.build_evidence_v61c(rows, periods, receipts)
    encoded = json.dumps(evidence, sort_keys=True)
    assert "synthetic forbidden" not in encoded
    assert '"question"' not in encoded and '"answer"' not in encoded
    assert evidence["alpha"] == 0.0
    assert evidence[
        "adapter_update_or_candidate_materialization_performed"
    ] is False
