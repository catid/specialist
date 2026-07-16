import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import eggroll_es_worker_lora_v43i as worker
import build_lora_es_multi_anchor_preregistration_v43i as prereg_builder
import lora_es_fused_anchor_runtime_v43i as subject
import run_lora_es_multi_anchor_v43i as runtime


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [1] + [2 + (ord(char) % 17) for char in text[:40]]


class FakeDense:
    @staticmethod
    def prepare_gold_answer_items_v4(tokenizer, prompts, answers):
        return [{"prompt_token_ids": [11, 12, 13], "answer": answer}
                for answer in answers]

    @staticmethod
    def score_gold_answer_outputs_v4(items, outputs):
        values = [float(output.dense_logprob) for output in outputs]
        return {
            "mean_example_mean_logprob": sum(values) / len(values),
            "answer_token_count": len(values),
            "examples": [{"mean_answer_token_logprob": value} for value in values],
        }


def _rows(count=2):
    return [{
        "prose": {"document_id": f"d{index}", "text": f"prose {index}"},
        "qa": {
            "document_id": f"d{index}", "instruction": f"question {index}",
            "answer": f"answer {index}", "answer_sha256": f"a{index}",
        },
    } for index in range(count)]


def _prose_output(ids, value):
    logprobs = [None] + [
        {token: SimpleNamespace(logprob=value)} for token in ids[1:]
    ]
    return SimpleNamespace(prompt_token_ids=ids, prompt_logprobs=logprobs)


def test_real_anchor_loader_is_train_only_and_deterministic_v43i():
    value = subject.load_anchor_bundle_v43i(
        runtime.PROSE_ANCHOR, runtime.PROSE_REPORT,
        runtime.QA_ANCHOR, runtime.QA_REPORT,
    )
    assert len(value["panel"]) == 32
    assert len(value["full"]) == 128
    assert value["direct_benchmark_source_opened"] is False
    assert value["protected_semantics_opened"] is False
    assert len({row["qa"]["document_id"] for row in value["panel"]}) == 32


def test_loader_rejects_protected_path_before_open_v43i(tmp_path):
    forbidden = tmp_path / "ood_qa.jsonl"
    with pytest.raises(ValueError, match="protected paths"):
        subject.load_anchor_bundle_v43i(
            forbidden, runtime.PROSE_REPORT, runtime.QA_ANCHOR, runtime.QA_REPORT,
        )


def test_fused_plan_and_scores_all_three_anchors_v43i():
    anchors = subject.prepare_anchor_items_v43i(
        _rows(), FakeTokenizer(), lambda value: f"chat:{value}", FakeDense,
    )
    domain = [{"prompt_token_ids": [90, 91]}]
    plan = subject.fused_requests_v43i(domain, anchors)
    assert plan["slices"] == {
        "domain": [0, 1], "prose": [1, 3], "qa_teacher": [3, 5],
        "qa_generation": [5, 7],
    }
    outputs = [SimpleNamespace(domain=True)]
    outputs += [_prose_output(item["prompt_token_ids"], -0.25)
                for item in anchors["prose"]]
    outputs += [SimpleNamespace(dense_logprob=-0.5),
                SimpleNamespace(dense_logprob=-0.3)]
    outputs += [
        SimpleNamespace(outputs=[SimpleNamespace(text="answer 0")]),
        SimpleNamespace(outputs=[SimpleNamespace(text="wrong")]),
    ]
    scored = subject.score_fused_outputs_v43i(
        plan, outputs, anchors, FakeDense,
        domain_scorer=lambda values: {"aggregate": {"equal_unit_mean": 1.0}},
    )
    assert scored["prose_lm"]["mean_token_logprob"] == pytest.approx(-0.25)
    assert scored["qa_answer_logprob"]["mean_example_logprob"] == pytest.approx(-0.4)
    assert scored["qa_generation"]["mean_f1"] == pytest.approx(0.5)
    assert scored["qa_generation"]["exact_count"] == 1
    assert scored["domain"]["aggregate"]["equal_unit_mean"] == 1.0


def _actor(domain=1.0, prose=2.0, qa=3.0, f1=0.5, exact=10, nonzero=20):
    return {
        "domain": {"aggregate": {"equal_unit_mean": domain}},
        "prose_lm": {"mean_token_logprob": prose},
        "qa_answer_logprob": {"mean_example_logprob": qa},
        "qa_generation": {
            "mean_f1": f1, "exact_count": exact, "nonzero_count": nonzero,
        },
    }


def test_calibration_and_candidate_gate_pass_and_fail_closed_v43i():
    records = [[_actor(
        prose=2.0 + repeat * 1e-5 + rank * 1e-6,
        qa=3.0 + repeat * 1e-5 + rank * 1e-6,
    ) for rank in range(4)] for repeat in range(8)]
    calibration = subject.calibration_margins_v43i(records)
    assert calibration["passed"] is True
    before = [_actor() for _ in range(4)]
    good = [_actor(domain=1.01, prose=2.0, qa=3.0) for _ in range(4)]
    assert subject.candidate_gate_v43i(before, good, calibration)["passed"] is True
    bad = [_actor(domain=1.01, qa=2.9) for _ in range(4)]
    gate = subject.candidate_gate_v43i(before, bad, calibration)
    assert gate["passed"] is False
    assert gate["checks"]["qa_logprob_noninferiority"] is False


def test_worker_abort_readback_is_idempotent_at_exact_master_v43i():
    instance = object.__new__(worker.LoRAAdapterStateWorkerExtensionV43I)
    instance._v41_pending_update = None
    instance._v41_committed_rollback = None
    instance._v41_current_identity = {"sha256": "master"}
    instance._verify_master_materialized_v41a = lambda phase: {
        "master_identity": {"sha256": "master"},
        "materialization": {"runtime_values_sha256": "runtime"},
    }
    instance._base_check_v41a = lambda phase: {"inventory_sha256": "base"}
    result = instance.abort_or_readback_sharded_adapter_update_v43i(
        "manifest", "master", "runtime",
    )
    assert result["aborted_or_verified"] is True
    assert result["disposition"] == "already_quiescent_at_expected_master"
    assert result["transaction_state_quiescent"] is True


def test_worker_accept_is_idempotent_after_partial_controller_response_v43i():
    instance = object.__new__(worker.LoRAAdapterStateWorkerExtensionV43I)
    instance._v41_pending_update = None
    instance._v41_committed_rollback = None
    instance._v41_current_identity = {"sha256": "candidate"}
    instance._verify_master_materialized_v41a = lambda phase: {
        "master_identity": {"sha256": "candidate"},
        "materialization": {"runtime_values_sha256": "runtime"},
    }
    instance._base_check_v41a = lambda phase: {"inventory_sha256": "base"}
    result = instance.accept_committed_adapter_update_v43i(
        "manifest", "candidate",
    )
    assert result["accepted"] is True
    assert result["disposition"] == "already_accepted"


def test_unsealed_accept_retains_exact_abort_path_v43i():
    instance = object.__new__(worker.LoRAAdapterStateWorkerExtensionV43I)
    rollback = {"manifest_sha256": "manifest"}
    instance._v41_pending_update = None
    instance._v41_committed_rollback = None
    instance._v43i_accepted_rollback = rollback
    instance._v41_current_identity = {"sha256": "candidate"}

    def abort(manifest):
        assert instance._v41_committed_rollback is rollback
        instance._v41_committed_rollback = None
        instance._v41_current_identity = {"sha256": "master"}
        return {"rolled_back": True}

    instance.abort_sharded_adapter_update_v41a = abort
    instance._verify_master_materialized_v41a = lambda phase: {
        "master_identity": {"sha256": "master"},
        "materialization": {"runtime_values_sha256": "runtime"},
    }
    instance._base_check_v41a = lambda phase: {"inventory_sha256": "base"}
    result = instance.abort_or_readback_sharded_adapter_update_v43i(
        "manifest", "master", "runtime",
    )
    assert result["disposition"] == "accepted_but_unsealed_update_rolled_back"
    assert instance._v43i_accepted_rollback is None


def test_launch_preregistration_and_dry_run_are_fail_closed_v43i(tmp_path, capsys):
    value = prereg_builder.build_v43i()
    assert value["gpu_launch_authorized"] is True
    assert value["sealed_holdout_opened"] is False
    assert value["access_contract"]["direct_benchmark_source_opened"] is False
    assert value["uncommitted_candidate_gate"]["score_before_commit"] is True
    path = tmp_path / "v43i.json"
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    result = runtime.main([
        "--preregistration", str(path),
        "--preregistration-sha256", runtime.v40a.file_sha256(path),
        "--preregistration-content-sha256",
        value["content_sha256_before_self_field"],
        "--dry-run",
    ])
    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output["gpu_launched"] is False
    assert output["protected_paths_opened"] == []
    assert output["direct_benchmark_source_opened"] is False
