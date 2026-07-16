#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import build_v55b_vs_v434_ood_preregistration_v56 as builder
import run_v55b_vs_v434_replicated_ood_only_v56 as subject


def _synthetic_ood_rows():
    qa_url = "https://example.com/v56-safe-qa"
    prose_url = "https://example.net/v56-safe-prose"
    qa = [{
        "item_id": "synthetic-qa",
        "question": "Which cobalt marker identifies the synthetic fixture?",
        "answer": "The marker is amber kestrel.",
        "url": qa_url,
        "normalized_source_url": subject.eval_v3.normalize_source_url(qa_url),
    }]
    prose = [{
        "item_id": "synthetic-prose",
        "title": "Synthetic violet almanac",
        "text": "Quartz finches catalogue a deliberately unrelated fixture.",
        "url": prose_url,
        "normalized_source_url": subject.eval_v3.normalize_source_url(prose_url),
    }]
    return qa, prose


def _safe_train_item():
    return {
        "row_sha256": "1" * 64,
        "document_sha256": "2" * 64,
        "normalized_urls": ["https://train.example.org/unrelated"],
        "raw_lineage_identity_sha256s": ["3" * 64],
        "semantic_cluster_sha256": "4" * 64,
        "semantic_question_tokens": ["completely", "different", "train"],
        "semantic_answer_tokens": ["separate", "vocabulary"],
    }


def _registry(item=None):
    return {"items": [_safe_train_item() if item is None else item]}


def _assert_collision(domain: str, item: dict):
    qa, prose = _synthetic_ood_rows()
    with pytest.raises(RuntimeError) as error:
        subject.prove_train_ood_disjoint_v56(qa, prose, _registry(item))
    assert f'"{domain}": 1' in str(error.value)


def test_v56_sources_and_stages_are_exact_and_train_only():
    v434 = subject._source_seal("v434_equal")
    v55b = subject._source_seal("v55b_candidate")
    assert v434["completed_steps"] == 48
    assert v434["shadow_ood_holdout_or_heldout_opened"] is False
    assert v55b["all_nine_train_endpoint_gates_passed"] is True
    assert v55b["selected_target_norm_ratio"] == 0.25
    assert v55b["shadow_ood_holdout_or_heldout_opened"] is False
    for logical in subject.LOGICAL_CANDIDATES:
        binding = subject.canonical_stage_binding_v56(logical)
        assert binding["weights_file_sha256"] == subject.STAGE_EXPECTED[
            logical
        ]["weights"]
        assert binding["tensor_bytes_preserved_exactly"] is True


def test_v56_two_full_waves_are_exact_and_candidate_pairs_interleaved():
    value = builder.build()
    waves = value["runtime"]["two_full_fixed_waves"]
    assert [[item["arm"] for item in wave] for wave in waves] == [
        ["base_a", "base_b", "base_c", "base_d"],
        [
            "v434_equal_a", "v55b_candidate_a",
            "v434_equal_b", "v55b_candidate_b",
        ],
    ]
    assert all(
        [item["engine_index"] for item in wave] == [0, 1, 2, 3]
        for wave in waves
    )
    assert value["runtime"]["all_four_gpus_busy_in_every_wave"] is True
    assert value["runtime"]["no_partial_or_third_wave"] is True


def test_v56_resolver_capability_is_audited_without_model_or_gpu_creation():
    receipt = subject.resolver_surface_preflight_v56()
    assert receipt["checked_without_model_or_gpu_creation"] is True
    assert receipt["v44a_make_trainer_dispatches_to_v40c"] is True
    assert receipt["v40c_make_trainer_attaches_resolve"] is True
    assert receipt["resolver_runtime_file_sha256"] == subject.core.file_sha256(
        Path(subject.core.v40c.__file__).resolve()
    )


def test_v56_tuned_table_uses_exact_v40c_runtime_projection_not_raw_json():
    receipt = subject.tuned_table_projection_preflight_v56()
    assert receipt["checked_without_model_or_gpu_creation"] is True
    assert receipt["file_content_sha256"] == (
        subject.TUNED_TABLE_FILE_CONTENT_SHA256
    )
    assert receipt["runtime_projection_sha256"] == (
        subject.TUNED_TABLE_RUNTIME_PROJECTION_SHA256
    )
    assert receipt["file_content_sha256"] != receipt[
        "runtime_projection_sha256"
    ]
    value = builder.build()
    assert value["runtime"]["tuned_table_content_sha256"] == (
        "4c4a0d4bbb400ea1d881bea3aae144d6865c34199fbb67889eda9e92d3a2543d"
    )


def test_v56_builder_and_dry_run_open_zero_ood_semantics_or_gpu(
    tmp_path, monkeypatch, capsys,
):
    protected = {
        str(Path(item["path"]).resolve())
        for item in subject.OOD_INPUTS.values()
    }
    original_open = Path.open
    original_hash = subject.core.file_sha256

    def guarded_open(path, *args, **kwargs):
        assert str(Path(path).resolve()) not in protected
        return original_open(path, *args, **kwargs)

    def guarded_hash(path):
        assert str(Path(path).resolve()) not in protected
        return original_hash(path)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(subject.core, "file_sha256", guarded_hash)
    value = builder.build()
    path = tmp_path / "prereg.json"
    subject.core.atomic_json(path, value)
    monkeypatch.setattr(
        subject.topology,
        "gpu_preflight",
        lambda: (_ for _ in ()).throw(AssertionError("dry-run reached GPU")),
    )
    monkeypatch.setattr(
        subject.core,
        "SingleSemanticAccessV44A",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run opened protected input")
        ),
    )
    argv = [
        "--preregistration", str(path),
        "--preregistration-sha256", original_hash(path),
        "--preregistration-content-sha256",
        value["content_sha256_before_self_field"],
        "--dry-run",
    ]
    assert subject.main(argv) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["single_access_labels"] == ["ood_prose", "ood_qa"]
    assert output["protected_semantic_access_count"] == 0
    assert output["train_rows_opened"] == 0
    assert output["gpu_accessed"] is False


def test_v56_safe_synthetic_identity_inventory_passes_all_four_domains():
    qa, prose = _synthetic_ood_rows()
    proof = subject.prove_train_ood_disjoint_v56(qa, prose, _registry())
    assert proof["all_four_identity_domains_disjoint"] is True
    assert proof["intersection_counts"] == {
        "document_sha256": 0,
        "normalized_url": 0,
        "raw_lineage": 0,
        "semantic_cluster": 0,
    }
    assert proof["checked_before_model_creation"] is True


def test_v56_document_identity_collision_fails_closed():
    item = _safe_train_item()
    item["document_sha256"] = subject.OOD_INPUTS["ood_qa"]["file_sha256"]
    _assert_collision("document_sha256", item)


def test_v56_normalized_url_collision_fails_closed():
    qa, _ = _synthetic_ood_rows()
    item = _safe_train_item()
    item["normalized_urls"] = [qa[0]["normalized_source_url"]]
    _assert_collision("normalized_url", item)


def test_v56_raw_lineage_collision_fails_closed():
    item = _safe_train_item()
    item["raw_lineage_identity_sha256s"] = [
        subject._raw_lineage_v56("ood_qa", 1)
    ]
    _assert_collision("raw_lineage", item)


def test_v56_semantic_cluster_collision_fails_closed():
    qa, _ = _synthetic_ood_rows()
    item = _safe_train_item()
    item["semantic_question_tokens"] = sorted(
        subject.semantic._content_tokens(qa[0]["question"])
    )
    item["semantic_answer_tokens"] = sorted(
        subject.semantic._content_tokens(qa[0]["answer"])
    )
    _assert_collision("semantic_cluster", item)


def test_v56_frozen_v13_direct_question_threshold_positive_and_negative():
    common_positive = frozenset(f"p{index}" for index in range(10))
    positive_left = common_positive | {"left"}
    positive_right = common_positive | {"right"}
    assert subject.semantic._jaccard(positive_left, positive_right) == pytest.approx(
        10 / 12
    )
    assert subject.semantic._semantic_match(
        (positive_left, frozenset()), (positive_right, frozenset())
    ) is True

    common_negative = frozenset(f"n{index}" for index in range(9))
    negative_left = common_negative | {"left"}
    negative_right = common_negative | {"right"}
    assert subject.semantic._jaccard(negative_left, negative_right) == pytest.approx(
        9 / 11
    )
    assert subject.semantic._semantic_match(
        (negative_left, frozenset()), (negative_right, frozenset())
    ) is False


def test_v56_frozen_v13_joint_threshold_positive_and_negative():
    question_left = frozenset({"alpha", "beta"})
    question_right = frozenset({"alpha", "beta", "gamma"})
    answer_common_positive = frozenset(f"a{index}" for index in range(7))
    answer_positive_left = answer_common_positive | {"left"}
    answer_positive_right = answer_common_positive
    assert subject.semantic._jaccard(
        question_left, question_right
    ) == pytest.approx(2 / 3)
    assert subject.semantic._jaccard(
        answer_positive_left, answer_positive_right
    ) == pytest.approx(7 / 8)
    assert subject.semantic._semantic_match(
        (question_left, answer_positive_left),
        (question_right, answer_positive_right),
    ) is True

    answer_common_negative = frozenset(f"b{index}" for index in range(6))
    answer_negative_left = answer_common_negative | {"left"}
    answer_negative_right = answer_common_negative
    assert subject.semantic._jaccard(
        answer_negative_left, answer_negative_right
    ) == pytest.approx(6 / 7)
    assert subject.semantic._semantic_match(
        (question_left, answer_negative_left),
        (question_right, answer_negative_right),
    ) is False


def _qa_summary(reward=1.0, logprob=-0.5, exact=1, nonzero=1):
    return {
        "generated_row_mean_reward": reward,
        "generated_equal_unit_mean_reward": reward,
        "generated_exact_count": exact,
        "generated_nonzero_count": nonzero,
        "teacher_forced_equal_unit_mean_answer_logprob": logprob,
        "protocol_leak_counters": {
            "protocol_token_emission": 0,
            "prompt_echo": 0,
            "empty_extracted_answer": 0,
        },
    }


def _qa_raw(reward=1.0, logprob=-0.5, fmt="exact"):
    return [{
        "item_sha256": "9" * 64,
        "reward": reward,
        "format": fmt,
        "teacher": {"mean_answer_token_logprob": logprob},
    }]


def _prose_detail(logprob=-1.0):
    return {
        "item_count": 1,
        "scored_token_count": 1,
        "sum_token_logprob": logprob,
        "mean_token_logprob": logprob,
        "items": [{
            "item_id": "synthetic-prose",
            "normalized_source_url": "https://example.net/synthetic-prose",
            "text_sha256": "7" * 64,
            "token_ids_sha256": "8" * 64,
            "prompt_token_count": 1,
            "scored_token_count": 1,
            "sum_token_logprob": logprob,
        }],
    }


def _passing_direct_fixture():
    arms = {
        arm for _pair, reference, candidate in subject.DIRECT_PAIRS
        for arm in (reference, candidate)
    }
    qa = {arm: _qa_summary() for arm in arms}
    prose = {arm: _prose_detail() for arm in arms}
    raw = {"ood_qa": {arm: _qa_raw() for arm in arms}}
    return qa, prose, raw


def test_v56_direct_gate_requires_both_replicas_and_all_paired_lcbs():
    qa, prose, raw = _passing_direct_fixture()
    result = subject.direct_gate_table_v56(qa, prose, raw)
    assert result[
        "both_candidate_replicas_independently_directly_noninferior"
    ] is True
    for replica in result["replicas"]:
        assert replica["independently_directly_noninferior"] is True
        assert all(replica["qa_point_checks"].values())
        assert all(replica["qa_paired_bootstrap_checks"].values())
        assert all(replica["prose_checks"].values())
        assert all(replica["protocol_no_increase_checks"].values())

    failing_qa = copy.deepcopy(qa)
    failing_raw = copy.deepcopy(raw)
    failing_qa["v55b_candidate_b"] = _qa_summary(
        reward=0.0, exact=0, nonzero=0
    )
    failing_raw["ood_qa"]["v55b_candidate_b"] = _qa_raw(
        reward=0.0, fmt="wrong"
    )
    failed = subject.direct_gate_table_v56(failing_qa, prose, failing_raw)
    assert failed[
        "both_candidate_replicas_independently_directly_noninferior"
    ] is False
    assert failed["replicas"][1]["independently_directly_noninferior"] is False


def test_v56_four_raw_base_controls_require_aggregate_and_raw_exactness():
    table = {arm: {"metric": 1.0} for arm in subject.BASE_ARMS}
    raw = {arm: [{"metric": 1.0}] for arm in subject.BASE_ARMS}
    receipt = subject.assert_four_raw_base_controls_v56(table, raw, "fixture")
    assert receipt["all_four_aggregate_outputs_exact"] is True
    assert receipt["all_four_raw_outputs_exact"] is True
    raw["base_d"] = [{"metric": 0.0}]
    with pytest.raises(RuntimeError, match="four raw-base determinism"):
        subject.assert_four_raw_base_controls_v56(table, raw, "fixture")


def test_v56_gpu_summary_requires_both_ood_phases_not_shadow(tmp_path):
    path = tmp_path / "gpu.jsonl"
    pids = {gpu: 10_000 + gpu for gpu in range(4)}
    rows = []
    for phase in subject.GPU_PHASES:
        for gpu, pid in pids.items():
            rows.append({
                "phase": phase,
                "gpu": gpu,
                "expected_pid": pid,
                "compute_pids": [pid],
                "foreign_compute_pids": [],
                "utilization_percent": 75,
                "memory_used_mib": 80_000,
            })
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    receipt = subject.summarize_ood_gpu_v56(path, pids)
    assert receipt["required_phase_labels_exact"] == [
        "v56_ood_qa", "v56_ood_prose"
    ]
    assert receipt["all_four_attributed_positive_each_ood_phase"] is True
    assert receipt["shadow_phase_required_or_opened"] is False
    assert set(receipt["by_gpu"]) == {"0", "1", "2", "3"}

    broken = copy.deepcopy(rows)
    broken[0]["foreign_compute_pids"] = [99_999]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in broken),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="coverage or exclusivity"):
        subject.summarize_ood_gpu_v56(path, pids)


def test_v56_prereg_documents_exact_document_identity_definitions_and_gates():
    value = builder.build()
    definitions = value["input_scope"]["document_identity_definitions"]
    assert "JSONL file SHA256" in definitions["ood_qa"]
    assert definitions["ood_prose"] == "SHA256 of the exact prose text field"
    registry = value["train_identity_registry"]
    assert registry["all_four_domains_must_be_disjoint_before_model_creation"]
    assert registry["semantic_comparison"]["equality_only_is_sufficient"] is False
    gates = value["direct_v55b_vs_v434_gates"]
    assert gates["both_v55b_replicas_must_independently_pass"] is True
    assert gates["generated_qa_paired_item_bootstrap_gates"][
        "samples"
    ] == 20_000


def test_v56_parser_requires_exactly_one_runtime_mode():
    required = [
        "--preregistration", "x",
        "--preregistration-sha256", "y",
        "--preregistration-content-sha256", "z",
    ]
    with pytest.raises(SystemExit):
        subject.parser().parse_args(required)
    with pytest.raises(SystemExit):
        subject.parser().parse_args(required + ["--dry-run", "--execute"])
