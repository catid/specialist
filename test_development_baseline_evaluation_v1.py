from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
import re
from collections import Counter

import pytest

import build_development_baseline_evaluation_v1 as build
import run_development_response_shard_v1 as runner
import score_development_baseline_v1 as score


def _synthetic_qa_rows() -> list[tuple[bytes, dict]]:
    group_ids = []
    for group_index, size in enumerate([1] * 33 + [2] * 5 + [3] * 5 + [6, 10]):
        group_ids.extend([f"synthetic-group-{group_index:03d}"] * size)
    assert len(group_ids) == 74
    rows = []
    for index in range(74):
        row = {
            "fact_id": f"synthetic-fact-{index:03d}",
            "question": f"What is synthetic value {index}?",
            "answer": f"value-{index}",
            "evidence": f"Synthetic evidence states that item {index} has value-{index}.",
            "source_split_lineage_v1": {
                "split": "development",
                "source_group_id": group_ids[index],
                "duplicate_component_id": group_ids[index].replace(
                    "group", "component"
                ),
                "parent_projection_sha256": "0" * 64,
                "assignment_preceded_projection": True,
            },
        }
        raw = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        rows.append((raw, row))
    return rows


def test_synthetic_general_fixture_proportions_and_judge_gate_are_exact():
    fixtures = build._synthetic_fixture_specs()
    assert len(fixtures) == 400
    assert Counter(item["category"] for item in fixtures) == {
        "coding": 80,
        "mathematics": 60,
        "structured_json": 40,
        "tool_function_calling": 40,
        "instruction_following": 40,
        "conversation": 40,
        "multilingual": 40,
        "safety_refusal": 20,
        "uncertainty": 20,
        "long_context": 20,
    }
    gated = [item for item in fixtures if item["scoring_status"] != "deterministic"]
    assert len(gated) == 60
    assert all(
        item["judge_gate"]["state"] == "disabled_until_sealed_judge_contract"
        and item["judge_gate"]["cannot_affect_selection_before_seal"] is True
        for item in gated
    )
    assert all(item["synthetic"] is True for item in fixtures)
    assert all(
        build.canonical_sha256({
            key: value
            for key, value in item.items()
            if key != "content_sha256_before_self_field"
        })
        == item["content_sha256_before_self_field"]
        for item in fixtures
    )
    for item in fixtures:
        if item["scoring_status"] != "deterministic" or item["category"] == "long_context":
            continue
        reference = item["reference"]
        values = reference["expected"] if reference["verifier"] == "one_of" else [reference["expected"]]
        for value in values:
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
            if rendered.isalnum():
                assert re.search(
                    rf"(?<![\w]){re.escape(rendered)}(?![\w])",
                    item["prompt"],
                ) is None
            else:
                assert rendered not in item["prompt"]


def test_synthetic_domain_projection_builds_74_paired_group_preserving_forms():
    source = _synthetic_qa_rows()
    items = build._domain_items(source)
    assert len(items) == 148
    assert Counter(item["form"] for item in items) == {
        "closed_book": 74,
        "grounded": 74,
    }
    by_fact = {}
    for item in items:
        by_fact.setdefault(item["source_selector"]["fact_id"], []).append(item)
    assert len(by_fact) == 74
    assert all(
        {item["form"] for item in pair} == {"closed_book", "grounded"}
        and len({item["source_selector"]["source_group_id"] for item in pair}) == 1
        for pair in by_fact.values()
    )


def test_domain_extensions_are_sealed_source_group_preserving_and_complete():
    rows = _synthetic_qa_rows()
    extensions = build._domain_extension_items(rows)
    assert len(extensions) == 88
    assert Counter(item["extension_type"] for item in extensions) == {
        "paraphrase_recall": 20,
        "application": 20,
        "synthesis": 12,
        "contradiction": 12,
        "unanswerable": 12,
        "false_premise": 12,
    }
    assert all(item["seal_state"] == "sealed_before_base_baseline_and_pilots" for item in extensions)
    assert all(item["judge_gate"] is None for item in extensions)
    assert all(item["composite_eligible_after_seal"] is True for item in extensions)
    assert all(
        len({selector["source_group_id"] for selector in item["source_selectors"]})
        == 1
        for item in extensions
    )
    assert all(
        len(item["source_selectors"]) == 2
        for item in extensions
        if item["extension_type"] == "synthesis"
    )


def test_runtime_domain_selector_and_grounded_support_use_only_supplied_synthetic_rows():
    rows = _synthetic_qa_rows()
    item = next(
        item for item in build._domain_items(rows) if item["form"] == "grounded"
    )
    runtime = runner._domain_runtime_item(item, rows)
    assert runtime["teacher_forcing"] is True
    assert runtime["family"] == "domain_knowledge"
    assert "Synthetic evidence" in runtime["prompt"]
    reference = json.loads(runtime["reference_text"])
    assert reference["support"] in rows[0][1]["evidence"]
    assert reference["answer"] == rows[0][1]["answer"]


def test_runtime_extension_prompts_keep_hidden_verifier_targets_separate():
    rows = _synthetic_qa_rows()
    extensions = build._domain_extension_items(rows)
    for extension_type in (
        "paraphrase_recall",
        "application",
        "contradiction",
        "unanswerable",
        "false_premise",
    ):
        item = next(
            item for item in extensions if item["extension_type"] == extension_type
        )
        runtime = runner._domain_extension_runtime_item(item, rows)
        reference = json.loads(runtime["reference_text"])
        assert runtime["reference_text"] not in runtime["prompt"]
        if extension_type in {"paraphrase_recall", "application"}:
            assert reference["answer"] not in runtime["prompt"]
        elif extension_type == "contradiction":
            assert reference["verdict"] not in runtime["prompt"]
        elif extension_type == "unanswerable":
            assert reference["answer"] not in runtime["prompt"]
        elif extension_type == "false_premise":
            assert reference["verdict"] not in runtime["prompt"]


def test_unallowlisted_data_path_fails_before_json_loading(tmp_path):
    path = tmp_path / "synthetic.jsonl"
    path.write_text("{}\n")
    with pytest.raises(RuntimeError, match="not allowlisted"):
        build._require_exact_data_input(path)


def test_general_deterministic_verifiers_are_strict_and_subjective_is_unscored():
    fixtures = build._synthetic_fixture_specs()
    exact = next(item for item in fixtures if item["fixture_name"] == "math_linear_00")
    assert score.score_general_response(exact, "41")["deterministic_pass"] == 1.0
    assert score.score_general_response(exact, "The answer is 41")["deterministic_pass"] == 0.0
    structured = next(item for item in fixtures if item["fixture_name"] == "json_boolean_00")
    assert score.score_general_response(
        structured, '{"note":null,"count":14,"ok":true}'
    )["deterministic_pass"] == 1.0
    assert score.score_general_response(structured, "not json")["format_valid"] == 0.0
    subjective = next(
        item for item in fixtures if item["scoring_status"] != "deterministic"
    )
    result = score.score_general_response(subjective, "A synthetic response")
    assert result["deterministic_pass"] is None
    assert result["judge_status"] == "gated_not_scored"


def test_domain_verifier_scores_json_answer_support_calibration_and_claims():
    item = {"form": "grounded"}
    source = {
        "answer": "The synthetic value is 42.",
        "evidence": "A sealed synthetic sentence says the synthetic value is 42.",
    }
    response = json.dumps({
        "answer": "The synthetic value is 42.",
        "support": source["evidence"],
        "confidence": 0.9,
    })
    result = score.score_domain_response(item, response, source)
    assert result["normalized_exact"] == 1.0
    assert result["correct_at_token_f1_0_65"] == 1.0
    assert result["json_valid"] == 1.0
    assert result["grounded_exact_support"] == 1.0
    assert result["unsupported_claim_unit_rate"] == 0.0
    assert result["confidence_brier"] == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("extension_type", "response", "source_rows"),
    [
        (
            "application",
            {"answer": "value-1", "confidence": 1.0},
            [{"answer": "value-1"}],
        ),
        (
            "synthesis",
            {"answer_a": "value-a", "answer_b": "value-b", "confidence": 1.0},
            [{"answer": "value-a"}, {"answer": "value-b"}],
        ),
        (
            "contradiction",
            {"verdict": "not_supported", "corrected_answer": "value-1", "confidence": 1.0},
            [{"answer": "value-1"}],
        ),
        (
            "unanswerable",
            {"answer": "INSUFFICIENT_INFORMATION", "confidence": 1.0},
            [{"answer": "unused"}],
        ),
        (
            "false_premise",
            {"verdict": "false_premise", "answer": "value-1", "confidence": 1.0},
            [{"answer": "value-1"}],
        ),
    ],
)
def test_domain_extension_deterministic_verifiers(
    extension_type,
    response,
    source_rows,
):
    result = score.score_domain_extension_response(
        {"extension_type": extension_type},
        json.dumps(response),
        source_rows,
    )
    assert result["extension_score"] == 1.0
    assert result["json_valid"] == 1.0
    assert result["confidence_brier"] == 0.0


def test_grouped_bootstrap_is_deterministic_and_keeps_groups_whole():
    records = [
        {"group": "a", "value": 1.0},
        {"group": "a", "value": 0.0},
        {"group": "b", "value": 1.0},
        {"group": "c", "value": 0.0},
    ]
    first = score.grouped_bootstrap_mean_ci(
        records,
        value_key="value",
        group_key="group",
        seed=17,
        replicates=500,
    )
    second = score.grouped_bootstrap_mean_ci(
        records,
        value_key="value",
        group_key="group",
        seed=17,
        replicates=500,
    )
    assert first == second
    assert first["groups"] == 3
    assert first["observations"] == 4
    assert first["lower_95"] <= first["mean"] <= first["upper_95"]


def test_paired_bootstrap_fails_closed_on_item_or_group_mismatch():
    base = [
        {"item": "a", "group": "g1", "value": 0.0},
        {"item": "b", "group": "g2", "value": 1.0},
    ]
    candidate = [
        {"item": "a", "group": "g1", "value": 1.0},
        {"item": "b", "group": "g2", "value": 1.0},
    ]
    result = score.paired_grouped_bootstrap_difference_ci(
        base,
        candidate,
        item_key="item",
        value_key="value",
        group_key="group",
        seed=3,
        replicates=200,
    )
    assert result["estimand"] == "candidate_minus_base"
    assert result["paired_item_count"] == 2
    assert result["mean"] == 0.5
    with pytest.raises(RuntimeError, match="item sets differ"):
        score.paired_grouped_bootstrap_difference_ci(
            base,
            candidate[:-1],
            item_key="item",
            value_key="value",
            group_key="group",
            replicates=10,
        )
    mismatched = [dict(candidate[0]), dict(candidate[1])]
    mismatched[1]["group"] = "changed"
    with pytest.raises(RuntimeError, match="source group changed"):
        score.paired_grouped_bootstrap_difference_ci(
            base,
            mismatched,
            item_key="item",
            value_key="value",
            group_key="group",
            replicates=10,
        )


def test_jsd_and_topk_residual_kl_are_exact_on_synthetic_probabilities():
    assert score.jensen_shannon_divergence([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert score.jensen_shannon_divergence([1.0, 0.0], [0.0, 1.0]) == pytest.approx(
        math.log(2)
    )
    logs = [math.log(0.6), math.log(0.3)]
    assert score.topk_plus_residual_kl(logs, 0.1, logs, 0.1) == pytest.approx(0.0)
    assert score.topk_plus_residual_kl(
        logs,
        0.1,
        [math.log(0.5), math.log(0.3)],
        0.2,
    ) > 0


def test_synthetic_routing_merge_reports_all_required_baseline_metrics():
    def state() -> dict:
        layers = {}
        for layer in range(40):
            layers[f"model.layers.{layer}.mlp.gate"] = {
                "num_experts": 256,
                "token_count": 8,
                "router_entropy_sum": 8 * 5.0,
                "top1_probability_ge_0_90_count": 0,
                "probability_load_sum": [8 / 256] * 256,
                "selected_assignment_count": [1] * 256,
            }
        return layers

    merged = score._merge_routing([
        {"routing_accumulator": state()} for _ in range(4)
    ])
    assert merged["layer_count"] == 40
    assert merged["base_self_jsd_is_zero"] is True
    for layer in merged["layers"].values():
        assert layer["active_expert_fraction"] == 1.0
        assert layer["expert_load_jsd_against_base"] == 0.0
        assert layer["load_imbalance_coefficient_of_variation"] == 0.0


def test_content_address_and_response_sharding_are_stable_on_synthetic_items():
    domain = build._domain_items(_synthetic_qa_rows())
    extensions = build._domain_extension_items(_synthetic_qa_rows())
    general = build._synthetic_fixture_specs()
    manifest = build._shard_manifest(domain, extensions, general)
    build.validate_self_address(manifest, schema=build.SHARD_SCHEMA)
    assert manifest["expected_total_items"] == 636
    assert [len(manifest["shards"][str(index)]) for index in range(4)] == [159] * 4
    assigned = [item for shard in manifest["shards"].values() for item in shard]
    assert len(assigned) == len(set(assigned)) == 636
    assert manifest["all_item_ids_sha256"] == build.canonical_sha256(sorted(assigned))


def test_existing_complete_shard_resumes_without_model_and_stale_shard_fails(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(runner.protocol, "display_path", lambda path: Path(path).name)
    row = build.self_address({"item_id": "synthetic-item"})
    contract = {"content_sha256_before_self_field": "a" * 64}
    shard_manifest = {
        "content_sha256_before_self_field": "b" * 64,
        "shards": {"0": ["synthetic-item"]},
    }
    value = build.self_address({
        "schema": runner.RESPONSE_SHARD_SCHEMA,
        "status": "complete",
        "shard_index": 0,
        "physical_gpu_index": 0,
        "preregistration_content_sha256": "a" * 64,
        "response_shard_manifest_content_sha256": "b" * 64,
        "items": [row],
        "item_count": 1,
    })
    path = tmp_path / "response.json"
    path.write_text(json.dumps(value))
    summary = runner._existing_complete_summary(
        path,
        contract=contract,
        shard_manifest=shard_manifest,
        shard_index=0,
        physical_gpu_index=0,
    )
    assert summary["status"] == "already_complete"
    assert summary["model_loaded"] is False
    value["preregistration_content_sha256"] = "c" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="quarantine"):
        runner._existing_complete_summary(
            path,
            contract=contract,
            shard_manifest=shard_manifest,
            shard_index=0,
            physical_gpu_index=0,
        )


def test_runner_and_scorer_have_no_dataset_path_interface_or_repository_discovery():
    assert list(inspect.signature(runner.run).parameters) == [
        "shard_index",
        "physical_gpu_index",
        "dry_run",
    ]
    assert set(build.EXACT_DATA_INPUT_ALLOWLIST) == {
        build.DEVELOPMENT_V440,
        build.DEVELOPMENT_SITE,
    }
    source_text = inspect.getsource(runner) + inspect.getsource(score)
    for forbidden_discovery in (".glob(", ".rglob(", "os.walk(", "Path.cwd("):
        assert forbidden_discovery not in source_text


def test_model_preflight_is_static_and_cannot_invoke_cross_gpu_probes():
    preflight_source = inspect.getsource(runner._validate_static_fast_engine_pin)
    load_source = inspect.getsource(runner._load_model)
    for forbidden in (
        "subprocess",
        "_gpu_inventory",
        "_run_worker_process",
        "construct(",
        "check_fast_linear_attention_contract",
    ):
        assert forbidden not in preflight_source
        assert forbidden not in load_source
    assert "_distribution_receipt" in preflight_source
    assert "gpu_inventory_or_kernel_probe_invoked" in preflight_source
