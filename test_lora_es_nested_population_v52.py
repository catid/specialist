#!/usr/bin/env python3

import builtins
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import build_lora_es_nested_population_preregistration_v52 as builder
import eggroll_es_worker_lora_v52 as worker
import lora_es_nested_population_v52 as design
import run_lora_es_nested_population_v52 as runtime


def test_v52_has_exactly_one_nested_scientific_variable():
    arms = design.scientific_arms_v52()
    proof = design.assert_one_scientific_variable_v52(arms)
    assert proof == {
        "sole_scientific_variable": "antithetic_direction_population_size",
        "control": 8,
        "treatment": 16,
        "p8_is_exact_nested_prefix_of_p16": True,
        "all_other_scientific_fields_equal": True,
    }
    assert arms["p8"]["seeds"] == arms["p16"]["seeds"][:8]
    assert arms["p16"]["seeds"][8:] == [
        863156398, 658045682, 947615772, 615729462,
        958574585, 1048698881, 573870406, 938961107,
    ]


def test_v52_freezes_all_32_direct_master_state_derivations():
    states = design.state_derivations_v52()
    assert len(states) == 32
    assert [item["state_index"] for item in states] == list(range(32))
    assert [item["direction"] for item in states] == [
        index // 2 for index in range(32)
    ]
    assert all(item["master_sha256"] == design.MASTER_SHA256_V52 for item in states)
    assert all("four-actor" in item["candidate_identity_policy"] for item in states)


def test_v52_centered_rank_objective_supports_p8_and_p16():
    for population_size in design.POPULATION_SIZES_V52:
        values = {
            "plus": [[float(index + actor / 100)] * 1
                     for index in range(population_size)
                     for actor in []],
            "minus": [],
        }
        values = {
            sign: [[
                float((index + 1) * (1 if sign == "plus" else -1) + actor / 100)
                for actor in range(4)
            ] for index in range(population_size)]
            for sign in ("plus", "minus")
        }
        result = design.objective_coefficients_v52(values)
        assert len(result["coefficients"]) == population_size
        assert result["zero_spread"] is False
        assert result["actor_reducer"] == (
            "fixed mean in ascending actor-rank order"
        )
        assert result["fixed_four_actor_mean_signed_scores"]["plus"][0] == (
            pytest.approx(sum(values["plus"][0]) / 4)
        )


def test_v52_reliability_gate_accepts_stable_p16_and_rejects_shape():
    stable = [[float(index), float(index), float(index), float(index)]
              for index in range(16)]
    result = design.reliability_gate_v52(stable, 0.0007)
    assert result["passed"] is True
    assert result["reliability"] == pytest.approx(1.0)
    assert result["split_half_spearman"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="8x4 or 16x4"):
        design.reliability_gate_v52([[1.0] * 4] * 12, 0.0007)


def _checks(value):
    return {name: value for name in design.TRAIN_GATE_NAMES_V52}


def test_v52_scale_search_stops_at_largest_pass_and_requires_exact_abort():
    results = [{
        "target_norm_ratio": 0.5,
        "checks": _checks(False),
        "candidate_consensus_passed": True,
        "exact_abort_readback_passed": True,
    }, {
        "target_norm_ratio": 0.25,
        "checks": _checks(True),
        "candidate_consensus_passed": True,
        "exact_abort_readback_passed": False,
    }]
    assert design.selected_train_ratio_v52(results) == 0.25
    broken = [dict(results[0], exact_abort_readback_passed=False)]
    with pytest.raises(ValueError, match="not exactly aborted"):
        design.selected_train_ratio_v52(broken)


def _ood_gate():
    return {
        "ood_qa": {"exact_count_delta": 0, "mean_reward_delta": 0.0},
        "ood_prose": {
            "point_delta": 0.001,
            "paired_document_bootstrap_95_ci": [0.0001, 0.002],
        },
        "protocol": {"counter_increase": 0},
        "raw_questions_answers_or_generations_persisted": False,
    }


def _shadow():
    return {
        "rows": 83,
        "conflict_units": 51,
        "split_manifest_file_sha256": design.SPLIT_MANIFEST_FILE_SHA256_V52,
        "edge_identity_intersections": {
            key: 0 for key in design.EDGE_IDENTITY_KEYS_V52
        },
        "document_disjoint_from_fold3_train": True,
        "better_than_master": True,
        "better_than_p8": True,
    }


def test_v52_ood_and_document_disjoint_gates_are_strict():
    assert design.ood_eligible_v52(_ood_gate()) is True
    failed = _ood_gate()
    failed["ood_prose"]["paired_document_bootstrap_95_ci"][0] = -1e-9
    assert design.ood_eligible_v52(failed) is False
    assert design.document_disjoint_shadow_eligible_v52(_shadow()) is True
    overlap = _shadow()
    overlap["edge_identity_intersections"]["normalized_url"] = 1
    assert design.document_disjoint_shadow_eligible_v52(overlap) is False


def test_v52_runtime_seals_four_actor_identity_before_scoring():
    state = design.state_derivations_v52()[0]
    candidate = "a" * 64
    runtime_sha = "b" * 64
    receipts = [{
        "actor_rank": rank,
        "state_index": state["state_index"],
        "seed": state["seed"],
        "sigma": state["sigma"],
        "sign": state["sign"],
        "master_identity": {"sha256": design.MASTER_SHA256_V52},
        "candidate_identity": {"sha256": candidate},
        "materialization": {"runtime_values_sha256": runtime_sha},
        "direct_from_pinned_fp32_master": True,
        "cumulative_candidate_delta_used": False,
    } for rank in range(4)]
    sealed = runtime.seal_runtime_state_consensus_v52(state, receipts)
    assert sealed["scoring_authorized"] is True
    receipts[3]["candidate_identity"]["sha256"] = "c" * 64
    with pytest.raises(RuntimeError, match="identity consensus"):
        runtime.seal_runtime_state_consensus_v52(state, receipts)


def test_v52_timing_requires_640_actor_phase_receipts():
    states = []
    for index in range(32):
        row = {"state_index": index}
        for phase in design.PHASES_V52:
            row[phase] = {"actors": [
                {"actor_rank": rank, "elapsed_ns": 0} for rank in range(4)
            ]}
        states.append(row)
    result = runtime.validate_timing_coverage_v52(states)
    assert result["total_actor_phase_receipts"] == 640
    assert result["phase_actor_receipts"] == {
        phase: 128 for phase in design.PHASES_V52
    }


def _fake_worker(monkeypatch):
    value = worker.LoRAAdapterStateWorkerExtensionV52.__new__(
        worker.LoRAAdapterStateWorkerExtensionV52
    )
    master_sha = "d" * 64
    value.device = torch.device("cpu")
    value._v41_master = {"kind": torch.tensor(0)}
    value._v41_current_identity = {"sha256": master_sha}
    value._v41_reference_identity = value._v41_current_identity
    value._v41_reference_fresh = True
    value._v41_active_perturbation = None
    value._v41_pending_update = None
    value._v41_committed_rollback = None
    value._v43i_accepted_rollback = None
    value._require_installed_v41a = lambda: None
    value._base_check_v41a = lambda phase: {"phase": phase, "unchanged": True}

    def identity(tensors):
        kind = int(tensors["kind"].item())
        return {"sha256": master_sha if kind == 0 else f"{kind:064x}"}

    def candidate(master, seed, _sigma, sign, _device):
        assert master is value._v41_master
        return {"kind": torch.tensor(seed + (1 if sign > 0 else 100))}

    def materialize(tensors, phase):
        identity_sha = identity(tensors)["sha256"]
        return {"phase": phase, "runtime_values_sha256": identity_sha[::-1]}

    monkeypatch.setattr(worker.state_v41a, "adapter_identity_v41a", identity)
    monkeypatch.setattr(worker.state_v41a, "antithetic_candidate_v41a", candidate)
    value._materialize_v41a = materialize
    return value, master_sha


def test_v52_worker_derives_every_state_directly_from_pinned_master(monkeypatch):
    value, master_sha = _fake_worker(monkeypatch)
    first = value.transition_derived_antithetic_from_pinned_master_v52(
        0, 7, 0.0006, 1, master_sha, master_sha,
    )
    second = value.transition_derived_antithetic_from_pinned_master_v52(
        1, 7, 0.0006, -1, master_sha,
        first["candidate_identity"]["sha256"],
    )
    assert first["intermediate_master_restore_elided"] is False
    assert second["intermediate_master_restore_elided"] is True
    assert second["direct_from_pinned_fp32_master"] is True
    assert second["cumulative_candidate_delta_used"] is False


def test_v52_worker_snapshots_pending_candidate_without_committing(monkeypatch):
    value, master_sha = _fake_worker(monkeypatch)
    candidate = {"kind": torch.tensor(8)}
    candidate_identity = {"sha256": f"{8:064x}"}
    runtime_sha = candidate_identity["sha256"][::-1]
    value.inter_pg = SimpleNamespace(rank=1)
    value._v41_pending_update = {
        "phase": "executed",
        "manifest_sha256": "e" * 64,
        "candidate_master": candidate,
        "candidate_identity": candidate_identity,
        "rollback_identity": {"sha256": master_sha},
    }
    monkeypatch.setattr(
        worker.state_v41a, "_clone_master_v41a",
        lambda tensors: dict(tensors),
    )
    receipt = value.save_pending_candidate_snapshot_v52(
        "/must/not/be/written/by/rank1", "e" * 64,
        candidate_identity["sha256"], runtime_sha,
    )
    assert receipt["written"] is False
    assert receipt["master_committed"] is False
    assert receipt["exact_abort_required_after_snapshot"] is True
    assert value._v41_current_identity == {"sha256": master_sha}
    assert value._v41_pending_update["phase"] == "executed"


class _MockOperationsV52:
    def __init__(self, fail=None):
        self.fail = fail
        self.events = []
        self.pending = 0
        self.maximum_pending = 0
        self.restore_calls = 0

    @staticmethod
    def _actors(schema, state, elapsed=1):
        return [{
            "schema": schema,
            "state_index": state["state_index"],
            "actor_rank": rank,
            "elapsed_ns": elapsed,
        } for rank in range(4)]

    def transition(self, derivation, previous):
        index = derivation["state_index"]
        self.events.append(("transition", index))
        if self.fail == ("transition", index):
            raise ValueError("mock v52 transition failure")
        candidate = f"{index + 1:064x}"
        runtime_sha = f"{index + 101:064x}"
        previous_sha = (
            design.MASTER_SHA256_V52 if previous is None
            else previous["candidate_identity_sha256"]
        )
        actors = self._actors("transition", derivation)
        for row in actors:
            row.update({
                "previous_candidate_sha256": previous_sha,
                "candidate_identity": {"sha256": candidate},
                "intermediate_master_restore_elided": previous is not None,
            })
        return {
            "schema": "transition",
            "actors": actors,
            "consensus": {
                "candidate_identity_sha256": candidate,
                "runtime_values_sha256": runtime_sha,
            },
        }

    def launch_generation(self, state):
        self.events.append(("launch", state["state_index"]))
        return [state["state_index"]] * 4

    def wait_generation(self, state, handles):
        index = state["state_index"]
        self.events.append(("generate", index))
        if self.fail == ("generate", index):
            raise ValueError("mock v52 generation failure")
        assert handles == [index] * 4
        return {
            "schema": "generate",
            "actors": self._actors("generate", state),
        }

    def submit_scoring(self, state, handles):
        self.events.append(("submit", state["state_index"]))
        assert self.pending == 0
        self.pending += 1
        self.maximum_pending = max(self.maximum_pending, self.pending)
        return handles

    def resolve_scoring(self, state, handles):
        index = state["state_index"]
        self.events.append(("drain", index))
        if self.fail == ("drain", index):
            raise ValueError("mock v52 drain failure")
        assert self.pending == 1
        self.pending -= 1
        return (
            [{"actor_rank": rank} for rank in range(4)],
            {"schema": "score", "actors": self._actors("score", state)},
            {"schema": "drain", "actors": self._actors("drain", state)},
        )

    def final_restore(self, reason):
        self.events.append(("restore", reason))
        self.restore_calls += 1
        state = {"state_index": 31}
        actors = self._actors("restore", state)
        for row in actors:
            row["restored"] = True
        return {
            "schema": "restore",
            "mode": "actual_exact_final_restore",
            "actors": actors,
        }

    def cancel_scoring(self, state, _handles):
        self.events.append(("cancel", state["state_index"]))
        self.pending = 0


def test_v52_live_schedule_overlaps_scoring_and_has_640_receipts():
    mock = _MockOperationsV52()
    completed, restored = runtime.run_direct_master_pipeline_v52(
        design.state_derivations_v52(), mock,
    )
    states = [{
        "state_index": item["state"]["state_index"],
        "materialize": item["transition"],
        "generate": item["generation"],
        "score": item["score_timing"],
        "restore": item["restore_timing"],
        "drain": item["drain_timing"],
    } for item in completed]
    coverage = runtime.validate_timing_coverage_v52(states)
    assert coverage["total_actor_phase_receipts"] == 640
    assert mock.maximum_pending == 1
    assert mock.pending == 0
    assert mock.restore_calls == 1
    assert restored == completed[-1]["restore_timing"]
    for left in range(31):
        assert mock.events.index(("submit", left)) < mock.events.index(
            ("transition", left + 1)
        )
        assert mock.events.index(("transition", left + 1)) < (
            mock.events.index(("drain", left))
        )


@pytest.mark.parametrize("phase,index", [
    ("transition", 1), ("generate", 1), ("drain", 0),
])
def test_v52_live_schedule_restores_and_cancels_on_failure(phase, index):
    mock = _MockOperationsV52(fail=(phase, index))
    with pytest.raises(ValueError, match="mock v52"):
        runtime.run_direct_master_pipeline_v52(
            design.state_derivations_v52(), mock,
        )
    assert mock.restore_calls == 1
    assert mock.pending == 0


def test_v52_preregistration_freezes_gates_and_compute_plan():
    value = builder.build_v52()
    assert value["retry_revision"] == design.RETRY_REVISION_V52
    assert value["launcher_fix"]["required_python"] == str(
        design.REQUIRED_PYTHON_V52
    )
    assert value["retry_science_equivalence"]["byte_equivalent"] is True
    assert value["retry_science_equivalence"][
        "original_science_content_sha256"
    ] == value["retry_science_equivalence"][
        "retry_science_content_sha256"
    ]
    assert value["measurement_contract"][
        "cross_actor_score_bit_equality_required"
    ] is False
    assert value["measurement_contract"][
        "postinstall_each_actor_must_equal_own_preinstall"
    ] is True
    assert value["measurement_contract"]["population_actor_reducer"] == (
        "fixed arithmetic mean in ascending actor-rank order"
    )
    assert runtime.require_live_interpreter_v52(
        str(design.REQUIRED_PYTHON_V52)
    )["matched"] is True
    with pytest.raises(RuntimeError, match="live launch requires"):
        runtime.require_live_interpreter_v52(str(design.ROOT / ".venv/bin/python"))
    assert value["single_scientific_variable"]["control"] == 8
    assert value["single_scientific_variable"]["treatment"] == 16
    assert value["transition_schedule"]["intermediate_exact_master_restores_eliminated"] == 31
    assert value["train_only_selection"]["all_nine_checks_required"] is True
    assert value["ood_first_gate"]["prose_paired_document_bootstrap_95_lcb_minimum"] == 0.0
    assert value["stop_go_gates"]["never_open_sealed_holdout"] is True
    assert value["compute_plan"]["phase_actor_receipts"] == 640
    assert value["runtime"] == design.RUNTIME_V52
    recipe = value["fixed_recipe"]
    assert recipe["master_sha256"] == (
        "eea2d60e19530ba99e9ac4bc50f2806b20aa13ed30e159bad63a0144d0cb81b6"
    )
    assert recipe["master_runtime_values_sha256"] == (
        "a1353c47bc11f02a9b67d7859d6670b07d6754c285ac4f357255878c09384f5b"
    )
    assert recipe["dataset_sha256"] == (
        "ae949c37de6abcd57fd8e2b9da8148b80ee072cfc16a7cf023c4ca89021b840a"
    )
    assert recipe["train_bundle_content_sha256"] == (
        design.TRAIN_BUNDLE_CONTENT_SHA256_V52
    )
    assert recipe["membership_file_sha256"] == design.MEMBERSHIP_SHA256_V52
    assert recipe["generation_panel_file_sha256"] == design.SUBSET_FILE_SHA256_V52
    assert value["train_only_selection"][
        "shared_fresh_calibration_before_population"
    ] is True
    assert value["train_only_selection"][
        "historical_v48e_calibration_values_reused"
    ] is False
    assert value["artifacts"]["p8_candidate_snapshot"] == str(
        runtime.P8_SNAPSHOT
    )
    assert value["artifacts"]["p16_candidate_snapshot"] == str(
        runtime.P16_SNAPSHOT
    )
    assert value["artifacts"]["numeric_calibration"] == str(
        runtime.NUMERIC_CALIBRATION
    )
    assert value["artifacts"]["anchor_calibration"] == str(
        runtime.ANCHOR_CALIBRATION
    )
    assert value["artifacts"]["preinstall_actor_baseline"] == str(
        runtime.PREINSTALL_BASELINE
    )
    assert value["artifacts"]["master_identity_audit"] == str(
        runtime.MASTER_IDENTITY_AUDIT
    )
    assert value["sealed_holdout_opened"] is False


def test_v52_actor_self_identity_allows_cross_actor_difference_but_not_drift():
    def record(rank, value):
        score = {"aggregate": {"equal_unit_mean": value}}
        return {
            "actor_rank": rank,
            "score": score,
            "exact_score_sha256": design.canonical_sha256_v52(score),
            "output_manifests": {
                "dense_result_sha256": f"{rank + 1:064x}",
                "unit_aggregate_sha256": f"{rank + 11:064x}",
                "scored_answer_tokens": 10,
                "unit_count": 208,
            },
        }

    baseline = {"actors": [record(rank, float(rank)) for rank in range(4)]}
    same = json.loads(json.dumps(baseline))
    receipt = runtime.require_actor_self_score_identity_v52(
        baseline, same, phase="postinstall",
    )
    assert receipt["all_four_actor_self_identities_exact"] is True
    assert len({item["exact_score_sha256"] for item in baseline["actors"]}) == 4
    same["actors"][2]["score"]["aggregate"]["equal_unit_mean"] += 1e-12
    with pytest.raises(RuntimeError, match="within an actor"):
        runtime.require_actor_self_score_identity_v52(
            baseline, same, phase="postpopulation",
        )


def test_v52_v434_content_free_train_contract_is_exact_and_source_faithful():
    membership = runtime._read_self_hashed_v52(
        design.TRAIN_MEMBERSHIP_V52,
        design.MEMBERSHIP_SHA256_V52,
        design.MEMBERSHIP_CONTENT_SHA256_V52,
    )
    panel = runtime.load_generation_panel_v52()
    assert membership["rows"] == 448
    assert membership["conflict_units"] == 208
    assert membership["source"]["train_dataset_file_sha256"] == (
        design.DATASET_SHA256_V52
    )
    assert membership["question_answer_evidence_or_text_persisted"] is False
    assert membership["protected_semantics_opened"] is False
    assert panel["selected_conflict_units"] == 64
    assert panel["request_order_sha256"] == design.REQUEST_ORDER_SHA256_V52
    assert panel["model_outcomes_used_for_selection"] is False
    assert panel["protected_semantics_opened"] is False


def test_v52_equal_v434_parent_passed_replicated_ood_and_shadow_without_holdout():
    report = design.sealed_json_v52("v49e_replicated_shadow_report")
    assert report["ood_eligibility_proof"][
        "both_equal_replicas_independently_ood_eligible"
    ] is True
    assert report["replicated_equal_vs_base_decision"][
        "replicated_equal_vs_base_decision_passed"
    ] is True
    assert report["heldout_or_holdout_opened"] is False
    contract = runtime.verify_adapter_contract_v52()
    assert contract["canonical_fp32_master_sha256"] == design.MASTER_SHA256_V52
    assert contract["bf16_runtime_values_sha256"] == (
        design.MASTER_RUNTIME_SHA256_V52
    )


def test_v52_dry_run_has_zero_writes_model_gpu_train_or_protected_access(
    tmp_path, monkeypatch, capsys,
):
    value = builder.build_v52()
    path = tmp_path / "prereg_v52.json"
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    file_sha = design.file_sha256_v52(path)
    before = {item.name: item.stat().st_mtime_ns for item in tmp_path.iterdir()}
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"ray", "vllm", "transformers", "datasets"}:
            raise AssertionError(f"v52 dry-run imported forbidden runtime {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        runtime, "_execute_v52",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("v52 dry-run entered live execution")
        ),
    )
    assert runtime.main([
        "--preregistration", str(path),
        "--preregistration-sha256", file_sha,
        "--preregistration-content-sha256",
        value["content_sha256_before_self_field"],
        "--dry-run",
    ]) == 0
    after = {item.name: item.stat().st_mtime_ns for item in tmp_path.iterdir()}
    assert before == after
    output = json.loads(capsys.readouterr().out)
    assert output["train_semantics_loaded"] is False
    assert output["protected_semantics_loaded"] is False
    assert output["model_imported_or_loaded"] is False
    assert output["gpu_queried_or_loaded"] is False
    assert output["filesystem_writes"] is False
    assert output["sealed_holdout_opened"] is False


def test_v52_sources_allow_only_required_v49d_artifacts_and_no_protected_paths():
    sources = "\n".join(Path(module.__file__).read_text(encoding="utf-8") for module in (
        design, builder, runtime, worker,
    ))
    assert "v49d_v434_sampling_midpoint_lr5p5e5" in sources
    assert "train_v434_fold3_v49d.jsonl" in sources
    assert "v434_equal_r32_seed17_init20260715041/final" in sources
    assert "dfb8ef8981cd4a21bd8d342353fc3d9c84c5d4759c38973e1528245f2baff192" not in sources
    assert "97fc920ac39f67536df26977de951e8c34bf8486eb8f42fbb0a67687f025a92a" not in sources
    assert "v48b_fold3_train_v412.jsonl" not in sources
    assert "import sft_v434" not in sources
    assert "data/eval_qa_v3.jsonl" not in sources
    assert "data/ood_qa_v3.jsonl" not in sources
    assert "data/ood_prose_v3.jsonl" not in sources
    assert "fold_3_shadow_dev.jsonl" not in sources
    assert "live GPU launch is sealed but not invoked" not in sources
    assert "run_direct_master_pipeline_v52" in sources
    assert "save_pending_candidate_snapshot_v52" in sources
    assert "_calibrate_numeric_path" in sources
    assert "_calibrate_anchor_path" in sources
    assert '["v48b_evidence"]' not in sources
    assert "strict_close_trainer_v38a" in sources
    assert "wait_for_gpu_idle" in sources
