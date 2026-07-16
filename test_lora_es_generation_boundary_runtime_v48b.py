#!/usr/bin/env python3

import json
import math
import types

import build_lora_es_generation_boundary_preregistration_v48b as builder
import run_lora_es_generation_boundary_v48b as subject


class _Tokenizer:
    def encode(self, _text, add_special_tokens=False):
        assert add_special_tokens is False
        return [11, 12, 13]


class _Trainer:
    tokenizer = _Tokenizer()


class _Completion:
    def __init__(self, text):
        self.text = text


class _Output:
    def __init__(self, text):
        self.outputs = [_Completion(text)]


def _selected_items(bundle):
    seen = set()
    result = []
    for row_sha, member in zip(
        bundle["row_sha256"], bundle["unit_membership_v48b"], strict=True,
    ):
        unit = member["unit_identity_sha256"]
        if unit in seen:
            continue
        seen.add(unit)
        result.append({
            "row_sha256": row_sha,
            "unit_identity_sha256": unit,
            "request_index": len(result),
        })
        if len(result) == 64:
            return result
    raise AssertionError("insufficient distinct train units")


def _sealed_subset(tmp_path):
    bundle = subject.load_train_bundle_v48b()
    selected = _selected_items(bundle)
    request_rows = [item["row_sha256"] for item in selected]
    nested = {
        "schema": "train-generation-boundary-subset-v48a",
        "status": "selected_once_from_prepopulation_base_evidence",
        "selected_rows": 64,
        "selected_conflict_units": 64,
        "items": selected,
        "request_order_row_sha256": request_rows,
        "request_order_sha256": (
            subject.boundary.canonical_sha256_v48a(request_rows)
        ),
        "common_random_generation_params": dict(
            subject.boundary.GENERATION_PARAMS_V48A
        ),
        "teacher_forced_domain_sampling_changed": False,
        "rows_duplicated_or_oversampled_in_domain_objective": False,
    }
    nested["content_sha256_before_self_field"] = (
        subject.boundary.canonical_sha256_v48a(nested)
    )
    value = {
        "schema": "sealed-train-generation-boundary-subset-v48b",
        "status": "complete_before_population_launch",
        "source": {
            "evidence_file_sha256": "1" * 64,
            "evidence_content_sha256": "2" * 64,
            "evidence_report_file_sha256": "3" * 64,
            "evidence_report_content_sha256": "4" * 64,
        },
        "subset": nested,
        "selected_rows": 64,
        "selected_conflict_units": 64,
        "request_order_sha256": nested["request_order_sha256"],
        "common_random_generation_params": dict(
            subject.boundary.GENERATION_PARAMS_V48A
        ),
        "question_answer_or_generation_text_persisted": False,
        "protected_semantics_opened": False,
        "shadow_ood_holdout_or_benchmark_opened": False,
    }
    value["content_sha256_before_self_field"] = (
        subject.v43i.v40a.canonical_sha256(value)
    )
    path = tmp_path / "subset.json"
    subject.v43i.v40a.atomic_json(path, value)
    return path, value


def test_v48b_train_bundle_is_exact_equal_unit_input():
    bundle = subject.load_train_bundle_v48b()
    assert len(bundle["questions"]) == len(bundle["answers"]) == 448
    assert bundle["conflict_units"] == 208
    assert len({
        item["unit_identity_sha256"]
        for item in bundle["unit_membership_v48b"]
    }) == 208
    assert math.isclose(sum(bundle["weights"]), 1.0, abs_tol=1e-15)
    assert bundle["content_sha256_before_self_field"] == (
        subject.EXPECTED_TRAIN_BUNDLE_CONTENT_SHA256_V48B
    )


def test_v48b_prepares_exact_sealed_fragile_order(monkeypatch):
    bundle = subject.load_train_bundle_v48b()
    selected = _selected_items(bundle)
    subset = {"subset": {"items": selected}}
    marker = (object(), object(), object(), object())
    monkeypatch.setattr(subject, "_SEALED_SUBSET", subset)
    monkeypatch.setattr(
        subject, "_ORIGINAL_PREPARE", lambda trainer, train, anchors: marker,
    )
    assert subject.prepare_v48b(_Trainer(), bundle, {}) == marker
    assert [item["row_sha256"] for item in subject._PREPARED_FRAGILE] == [
        item["row_sha256"] for item in selected
    ]
    assert len(subject._PREPARED_FRAGILE) == 64
    assert all(item["prompt_token_ids"] == [11, 12, 13]
               for item in subject._PREPARED_FRAGILE)


def test_v48b_fused_plan_generates_only_qa_and_fragile_slices(monkeypatch):
    monkeypatch.setattr(subject, "_PREPARED_FRAGILE", [{
        "prompt_token_ids": [index + 1],
    } for index in range(64)])
    anchors = {
        "documents": 32,
        "prose": [{"prompt_token_ids": [1]} for _ in range(32)],
        "qa_teacher": [{"prompt_token_ids": [2]} for _ in range(32)],
        "qa_generation": [{"prompt_token_ids": [3]} for _ in range(32)],
    }
    plan = subject.fused_requests_v48b(
        [{"prompt_token_ids": [4]} for _ in range(448)], anchors,
    )
    assert len(plan["requests"]) == 608
    assert plan["slices"]["fragile_generation"] == [544, 608]
    teacher, generation = object(), object()
    monkeypatch.setattr(subject.v43i, "_teacher_sampling_params", lambda: teacher)
    monkeypatch.setattr(
        subject.v43i, "_generation_sampling_params", lambda: generation,
    )
    params = subject.sampling_params_for_plan_v48b(plan)
    assert sum(item is generation for item in params) == 96
    assert all(item is generation for item in params[512:544])
    assert all(item is generation for item in params[544:608])
    assert all(item is teacher for item in params[:512])


def test_v48b_fragile_scoring_persists_numeric_hashes_not_text(monkeypatch):
    items = [{
        "row_sha256": f"{index:064x}",
        "unit_identity_sha256": f"{index + 1000:064x}",
        "request_index": index,
    } for index in range(64)]
    monkeypatch.setattr(subject, "_SEALED_SUBSET", {"subset": {"items": items}})
    monkeypatch.setattr(subject, "_PREPARED_FRAGILE", [{
        **item, "answer": "correct answer", "prompt_token_ids": [index + 1],
    } for index, item in enumerate(items)])
    monkeypatch.setattr(
        subject, "_ORIGINAL_SCORE_FUSED",
        lambda plan, outputs, anchors, dense, domain_scorer=None: {
            "domain": {"aggregate": {"equal_unit_mean": 0.0}},
        },
    )
    plan = {
        "requests": [{"prompt_token_ids": [1]} for _ in range(66)],
        "slices": {"domain": [0, 2], "fragile_generation": [2, 66]},
    }
    result = subject.score_fused_outputs_v48b(
        plan, [_Output("")] * 2 + [_Output("correct answer")] * 64,
        {}, object(), domain_scorer=lambda outputs: {},
    )
    fragile = result["fragile_generation"]
    assert fragile["equal_conflict_unit_mean_f1"] == 1.0
    assert fragile["exact_count"] == fragile["nonzero_count"] == 64
    serialized = json.dumps(fragile, sort_keys=True)
    assert "correct answer" not in serialized


def test_v48b_candidate_gate_fails_closed_on_fragile_regression(monkeypatch):
    monkeypatch.setattr(
        subject, "_ORIGINAL_CANDIDATE_GATE",
        lambda reference, candidate, calibration: {
            "schema": "base", "checks": {"base_gate": True}, "passed": True,
        },
    )
    reference = [{"fragile_generation": {
        "equal_conflict_unit_mean_f1": 0.5,
        "exact_count": 20, "nonzero_count": 30,
    }} for _ in range(4)]
    candidate = [{"fragile_generation": {
        "equal_conflict_unit_mean_f1": 0.49,
        "exact_count": 20, "nonzero_count": 30,
    }} for _ in range(4)]
    gate = subject.candidate_gate_v48b(reference, candidate, {})
    assert gate["checks"]["base_gate"] is True
    assert gate["checks"]["fragile_generation_f1_noninferiority"] is False
    assert gate["passed"] is False


def test_v48b_patch_context_restores_parent_runtime():
    original = {
        "prepare": subject.v43i._prepare,
        "population": subject.v43i._replicated_population,
        "fused": subject.v43i.fused.fused_requests_v43i,
        "loader": subject.v43i.equal_v38.load_equal_unit_train_bundle,
    }
    with subject.patched_v43i_v48b():
        assert subject.v43i._prepare is subject.prepare_v48b
        assert subject.v43i._replicated_population is subject.replicated_population_v48b
        assert subject.v43i.fused.fused_requests_v43i is subject.fused_requests_v48b
        assert subject.v43i.equal_v38.load_equal_unit_train_bundle is (
            subject.load_train_bundle_v48b
        )
    assert subject.v43i._prepare is original["prepare"]
    assert subject.v43i._replicated_population is original["population"]
    assert subject.v43i.fused.fused_requests_v43i is original["fused"]
    assert subject.v43i.equal_v38.load_equal_unit_train_bundle is original["loader"]


def test_v48b_implementation_binding_covers_integrated_runtime():
    bindings = subject.implementation_bindings_v48b("a" * 64)
    assert bindings["subset"] == "a" * 64
    assert len(bindings["runtime"]) == len(bindings["tests"]) == 64
    assert bindings["train_dataset"] == subject.evidence_runtime.EXPECTED_TRAIN_SHA256


def test_v48b_preregistration_and_dry_run_are_subset_bound_and_gpu_free(
    tmp_path, monkeypatch, capsys,
):
    subset_path, subset = _sealed_subset(tmp_path)
    subset_file_sha = subject.v43i.v40a.file_sha256(subset_path)
    subset_content_sha = subset["content_sha256_before_self_field"]
    monkeypatch.setattr(subject, "SUBSET", subset_path.resolve())
    prereg = builder.build_v48b(
        subset_path, subset_file_sha, subset_content_sha,
    )
    assert prereg["gpu_launch_authorized"] is True
    assert prereg["recipe"]["fused_requests_per_population_actor_state"] == 608
    assert prereg["generation_boundary_objective"][
        "fail_if_fragile_generation_f1_has_zero_population_spread"
    ] is True
    prereg_path = tmp_path / "prereg.json"
    subject.v43i.v40a.atomic_json(prereg_path, prereg)
    prereg_file_sha = subject.v43i.v40a.file_sha256(prereg_path)
    args = types.SimpleNamespace(
        preregistration=str(prereg_path),
        preregistration_sha256=prereg_file_sha,
        preregistration_content_sha256=prereg[
            "content_sha256_before_self_field"
        ],
    )
    subject.load_preregistration_v48b(args)
    monkeypatch.setattr(
        subject, "load_train_bundle_v48b",
        lambda: (_ for _ in ()).throw(AssertionError("dry-run loaded train")),
    )
    assert subject.main([
        "--preregistration", str(prereg_path),
        "--preregistration-sha256", prereg_file_sha,
        "--preregistration-content-sha256",
        prereg["content_sha256_before_self_field"],
        "--dry-run",
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["train_semantics_loaded"] is False
    assert output["model_or_gpu_loaded"] is False
    assert output["filesystem_writes"] is False
