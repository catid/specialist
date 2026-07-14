import copy
import json
from types import SimpleNamespace

import pytest

import eggroll_es_robust_anchor as robust_anchor
import run_eggroll_es_anchor_line_search_v5 as line_search_v5
import train_eggroll_es_specialist_anchor_v5 as anchor_v5


def document(document_id, token_count, token_sum):
    return {
        "document_id": document_id,
        "scored_token_count": token_count,
        "sum_token_logprob": token_sum,
    }


def prompt_output(token_ids, value):
    logprobs = [None]
    logprobs.extend({
        token_id: SimpleNamespace(logprob=value)
    } for token_id in token_ids[1:])
    return SimpleNamespace(
        prompt_token_ids=list(token_ids),
        prompt_logprobs=logprobs,
    )


def robust_plan():
    seeds = [11, 22, 33, 44]
    reference = [
        document("doc-a", 2, -2.0),
        document("doc-b", 3, -3.0),
        document("doc-c", 1, -1.0),
    ]
    populations = []
    for offset, seed in enumerate(seeds):
        populations.append({
            "seed": seed,
            "documents": [
                document("doc-a", 2, -2.0 + 0.1 * offset),
                document("doc-b", 3, -3.0 - 0.03 * offset),
                document("doc-c", 1, -1.0 + 0.04 * offset),
            ],
        })
    result = robust_anchor.score_population_document_lcbs(
        reference, populations,
    )
    score_by_seed = {
        row["seed"]: row["score"] for row in result["robust_scores"]
    }
    anchor_scores = [score_by_seed[seed] for seed in seeds]
    domain_scores = [-4.0, -3.5, -3.7, -3.0]
    projection = (
        anchor_v5.anchor_v4.anchor_v3.anchor_v2.anchor_v1
        .project_anchor_safe_coefficients(
            domain_scores, anchor_scores, min_anchor_cosine=0.8,
        )
    )
    plan = {
        "seeds": seeds,
        "domain_scores": domain_scores,
        "anchor_scores": anchor_scores,
        "coefficients": projection["coefficients"],
        "projection": projection["diagnostics"],
        "frozen_layer_plan_v4": {"plan_sha256": "a" * 64},
        "document_lcb_anchor_v5": result,
        "identity_audit": {
            "passed": True,
            "pre_probe": {
                "document_lcb_anchor_v5": {
                    "config_sha256": (
                        robust_anchor.DOCUMENT_LCB_CONFIG_SHA256
                    ),
                    "document_count": result["reference"]["document_count"],
                    "scored_token_count": result["reference"][
                        "scored_token_count"
                    ],
                    "document_manifest_sha256": result[
                        "document_manifest_sha256"
                    ],
                    "reference_numeric_summary_sha256": result[
                        "reference"
                    ]["numeric_summary_sha256"],
                    "raw_document_content_persisted": False,
                },
            },
        },
    }
    plan["identity_audit"]["post_probe"] = copy.deepcopy(
        plan["identity_audit"]["pre_probe"]
    )
    plan["coefficient_sha256"] = anchor_v5.coefficient_sha256(
        seeds, plan["coefficients"],
    )
    plan["robust_plan_binding_v5"] = anchor_v5._robust_binding_v5(
        plan, result,
    )
    return plan


def contains_key(value, target):
    if isinstance(value, dict):
        return target in value or any(
            contains_key(item, target) for item in value.values()
        )
    if isinstance(value, list):
        return any(contains_key(item, target) for item in value)
    return False


def v5_snapshot_provenance(seed=42):
    implementation = {
        "distributed_driver_v5": "1" * 64,
        "distributed_trainer_v5": "2" * 64,
        "robust_anchor_v5": "3" * 64,
    }
    return {
        "schema": "eggroll-es-anchor-line-search-snapshot-v5",
        "recipe": {"seed": seed},
        "implementation": implementation,
        "document_lcb_anchor_v5": {
            "config": robust_anchor.document_lcb_config(),
            "config_sha256": robust_anchor.DOCUMENT_LCB_CONFIG_SHA256,
            "objective_source": "frozen_train_only_anchor_prose",
            "ood_validation_heldout_as_objective": False,
            "implementation_bundle_sha256": (
                robust_anchor.canonical_sha256(implementation)
            ),
        },
    }


def test_document_summaries_are_per_document_selected_token_statistics():
    items = [
        {"document_id": "a", "prompt_token_ids": [1, 2]},
        {"document_id": "b", "prompt_token_ids": [3, 4, 5, 6]},
    ]
    summaries = anchor_v5.summarize_anchor_documents_v5(items, [
        prompt_output([1, 2], -1.0),
        prompt_output([3, 4, 5, 6], -3.0),
    ])
    assert summaries == [
        {"document_id": "a", "scored_token_count": 1,
         "sum_token_logprob": -1.0},
        {"document_id": "b", "scored_token_count": 3,
         "sum_token_logprob": -9.0},
    ]


def test_v5_identity_probe_reuses_one_anchor_dispatch(monkeypatch):
    dense_items = [
        {"prompt_token_ids": [index, index + 1]} for index in range(4)
    ]
    anchor_items = [
        {"document_id": "a", "prompt_token_ids": [1, 2]},
        {"document_id": "b", "prompt_token_ids": [3, 4]},
    ]
    monkeypatch.setattr(
        anchor_v5.anchor_v4, "prepare_gold_answer_items_v4",
        lambda tokenizer, inputs, answers: dense_items,
    )
    dense = {"mean_example_mean_logprob": -1.0}
    monkeypatch.setattr(
        anchor_v5.anchor_v4, "score_gold_answer_outputs_v4",
        lambda items, outputs: dense,
    )
    calls = []

    def dispatch(engines, prompts, sampling, resolve):
        calls.append((prompts, sampling))
        return [object() for _ in prompts]

    monkeypatch.setattr(
        anchor_v5.anchor_v4.anchor_v3.anchor_v2.anchor_v1,
        "dispatch_eval_batch", dispatch,
    )
    monkeypatch.setattr(
        anchor_v5.anchor_v4.anchor_v3.anchor_v2,
        "anchor_output_sha256", lambda items, outputs: "b" * 64,
    )
    summaries = [
        document("a", 1, -1.0), document("b", 1, -2.0),
    ]
    monkeypatch.setattr(
        anchor_v5, "summarize_anchor_documents_v5",
        lambda items, outputs: summaries,
    )
    trainer = SimpleNamespace(
        _v4_identity_probe_count=0,
        _v4_identity_target_answers=["answer"] * 4,
        tokenizer=object(),
        engines=[object()] * 4,
        _dense_sampling_params_v4=lambda iteration: "dense",
        _anchor_sampling_v5=lambda iteration: "anchor",
        _resolve=lambda values: values,
        _v4_reward_config_sha256="c" * 64,
        _v4_layer_plan={"plan_sha256": "d" * 64},
        _v5_reference_document_summaries=None,
    )
    probe = anchor_v5.DocumentLCBAnchoredMixinV5._identity_probe(
        trainer, ["input"] * 4, None, anchor_items, 0,
    )
    assert [sampling for _, sampling in calls] == ["dense", "anchor"]
    assert probe["anchor_requests"] == 2
    assert probe["document_lcb_anchor_v5"]["document_count"] == 2


def test_robust_plan_binds_scores_projection_and_all_numeric_provenance():
    plan = robust_plan()
    binding = anchor_v5.validate_robust_plan_v5(
        plan, recompute_numeric=True,
    )
    result = plan["document_lcb_anchor_v5"]
    assert binding["result_sha256"] == result[
        "content_sha256_before_self_field"
    ]
    assert binding["bootstrap_plan_sha256"] == result[
        "bootstrap_plan"
    ]["plan_sha256"]
    assert contains_key(result, "document_id") is False


@pytest.mark.parametrize(
    "mutation,message",
    [
        (
            lambda plan: plan["document_lcb_anchor_v5"]["population"][0]
            .__setitem__("mean_delta", 9.0),
            "provenance",
        ),
        (
            lambda plan: plan["anchor_scores"].__setitem__(0, 9.0),
            "LCB fitness",
        ),
        (
            lambda plan: plan["coefficients"].__setitem__(0, 9.0),
            "projection",
        ),
        (
            lambda plan: plan["robust_plan_binding_v5"].__setitem__(
                "config_sha256", "0" * 64,
            ),
            "binding",
        ),
    ],
)
def test_robust_plan_fails_closed_on_any_tampering(mutation, message):
    plan = robust_plan()
    mutation(plan)
    with pytest.raises(RuntimeError, match=message):
        anchor_v5.validate_robust_plan_v5(plan)


def test_robust_plan_rejects_self_consistent_numeric_forgery():
    plan = robust_plan()
    result = plan["document_lcb_anchor_v5"]
    result["population"][0]["mean_delta"] = 9.0
    result["content_sha256_before_self_field"] = robust_anchor.canonical_sha256({
        key: value for key, value in result.items()
        if key != "content_sha256_before_self_field"
    })
    plan["robust_plan_binding_v5"] = anchor_v5._robust_binding_v5(
        plan, result,
    )
    with pytest.raises(RuntimeError, match="numeric recomputation"):
        anchor_v5.validate_robust_plan_v5(
            plan, recompute_numeric=True,
        )


def test_v5_cli_forbids_using_ood_as_the_training_anchor():
    with pytest.raises(ValueError, match="frozen training anchor"):
        line_search_v5.validate_frozen_execution_cli_v5([
            "--anchor-prose-jsonl", "data/ood_prose_v3.jsonl",
        ])
    validated = line_search_v5.validate_frozen_execution_cli_v5([])
    assert validated["anchor_items_per_step"] == 128
    assert validated["min_anchor_cosine"] == 0.8
    assert validated["document_lcb_config_sha256"] == (
        robust_anchor.DOCUMENT_LCB_CONFIG_SHA256
    )


def test_v5_snapshot_extends_the_complete_v4_identity_chain(monkeypatch):
    monkeypatch.setattr(
        line_search_v5.driver_v4,
        "build_snapshot",
        lambda *args, **kwargs: {
            "schema": "eggroll-es-anchor-line-search-snapshot-v4",
            "implementation": {"distributed_driver_v4": "v4-hash"},
        },
    )
    snapshot = line_search_v5.build_snapshot()
    assert snapshot["schema"] == "eggroll-es-anchor-line-search-snapshot-v5"
    assert snapshot["implementation"]["distributed_driver_v4"] == "v4-hash"
    assert set(snapshot["implementation"]) >= {
        "distributed_driver_v5", "distributed_trainer_v5",
        "robust_anchor_v5",
    }
    assert snapshot["document_lcb_anchor_v5"][
        "ood_validation_heldout_as_objective"
    ] is False
    implementation_v5 = {
        key: snapshot["implementation"][key]
        for key in line_search_v5.V5_IMPLEMENTATION_KEYS
    }
    assert snapshot["document_lcb_anchor_v5"][
        "implementation_bundle_sha256"
    ] == robust_anchor.canonical_sha256(implementation_v5)


def test_execute_upgrades_v4_journal_and_self_validates(tmp_path, monkeypatch):
    plan = robust_plan()
    journal = {
        "schema": "eggroll-es-anchor-alpha-line-search-v4",
        "status": "complete",
        "in_progress": None,
        "snapshot": v5_snapshot_provenance(),
        "policy": {},
        "seeds": list(plan["seeds"]),
        "states": [],
        "coefficient_plan": {
            "coefficients": list(plan["coefficients"]),
            "coefficient_sha256": plan["coefficient_sha256"],
            "projection": dict(plan["projection"]),
            "frozen_layer_plan_v4": dict(plan["frozen_layer_plan_v4"]),
            "identity_audit": copy.deepcopy(plan["identity_audit"]),
        },
    }
    monkeypatch.setattr(
        line_search_v5.driver_v4, "execute_line_search",
        lambda *args, **kwargs: journal,
    )
    audited = {}

    def validate_v4_compatibility(value):
        audited["journal"] = value
        assert value["schema"] == "eggroll-es-anchor-alpha-line-search-v4"
        assert value["snapshot"]["schema"] == (
            "eggroll-es-anchor-line-search-snapshot-v4"
        )
        assert "document_lcb_anchor_v5" not in value["snapshot"]
        assert value["coefficient_plan"]["identity_audit"][
            "pre_probe"
        ]["schema"] == "eggroll-es-train-only-identity-probe-v4"
        return {
            "seed": value["snapshot"]["recipe"]["seed"],
            "content_sha256": value["content_sha256_before_self_field"],
        }

    monkeypatch.setattr(
        line_search_v5.offline_audit, "validate_journal",
        validate_v4_compatibility,
    )
    trainer = SimpleNamespace(_latest_anchor_plan=plan)
    path = tmp_path / "journal.json"
    upgraded = line_search_v5.execute_line_search(
        trainer, journal_path=path,
    )
    validated = line_search_v5.validate_completed_journal_v5(upgraded)
    assert upgraded["schema"] == "eggroll-es-anchor-alpha-line-search-v5"
    assert validated["seed"] == 42
    assert validated["robust_plan_sha256"] == plan[
        "robust_plan_binding_v5"
    ]["robust_plan_sha256"]
    assert "journal" in audited

    monkeypatch.setattr(
        line_search_v5, "validate_completed_journal_v5",
        lambda value: (_ for _ in ()).throw(RuntimeError("offline failure")),
    )
    failed_path = tmp_path / "failed.json"
    with pytest.raises(RuntimeError, match="offline failure"):
        line_search_v5.execute_line_search(
            trainer, journal_path=failed_path,
        )
    failed = json.loads(failed_path.read_text())
    assert failed["status"] == "failed"
    assert failed["failure"]["phase"] == (
        "validating_complete_v5_release_audit"
    )
    assert "content_sha256_before_self_field" not in failed


def test_completed_v5_requires_exact_optimization_policy(monkeypatch):
    plan = robust_plan()
    journal = {
        "schema": "eggroll-es-anchor-alpha-line-search-v5",
        "status": "complete",
        "in_progress": None,
        "policy": {
            "document_lcb_anchor_required": True,
            "optimization_data": "train_and_anchor_only",
            "ood_validation_heldout_as_objective": False,
        },
        "snapshot": v5_snapshot_provenance(),
        "seeds": plan["seeds"],
        "states": [],
        "coefficient_plan": {
            "coefficients": plan["coefficients"],
            "coefficient_sha256": plan["coefficient_sha256"],
            "projection": plan["projection"],
            "frozen_layer_plan_v4": plan["frozen_layer_plan_v4"],
            "identity_audit": plan["identity_audit"],
            "domain_scores_v5": plan["domain_scores"],
            "anchor_scores_v5": plan["anchor_scores"],
            "document_lcb_anchor_v5": plan["document_lcb_anchor_v5"],
            "robust_plan_binding_v5": plan["robust_plan_binding_v5"],
        },
    }
    journal["content_sha256_before_self_field"] = (
        robust_anchor.canonical_sha256(journal)
    )
    monkeypatch.setattr(
        line_search_v5.offline_audit, "validate_journal",
        lambda value: {
            "seed": 42,
            "content_sha256": value["content_sha256_before_self_field"],
        },
    )
    assert line_search_v5.validate_completed_journal_v5(journal)["seed"] == 42
    for key, unsafe in (
        ("document_lcb_anchor_required", False),
        ("optimization_data", "evaluation"),
        ("ood_validation_heldout_as_objective", True),
    ):
        changed = copy.deepcopy(journal)
        changed["policy"][key] = unsafe
        changed["content_sha256_before_self_field"] = (
            robust_anchor.canonical_sha256({
                name: value for name, value in changed.items()
                if name != "content_sha256_before_self_field"
            })
        )
        with pytest.raises(RuntimeError, match="optimization policy"):
            line_search_v5.validate_completed_journal_v5(changed)
    changed = copy.deepcopy(journal)
    changed["snapshot"]["implementation"]["robust_anchor_v5"] = "invalid"
    changed["content_sha256_before_self_field"] = (
        robust_anchor.canonical_sha256({
            name: value for name, value in changed.items()
            if name != "content_sha256_before_self_field"
        })
    )
    with pytest.raises(RuntimeError, match="snapshot robust-objective"):
        line_search_v5.validate_completed_journal_v5(changed)
    changed = copy.deepcopy(journal)
    changed["snapshot"]["document_lcb_anchor_v5"]["unexpected"] = True
    changed["content_sha256_before_self_field"] = (
        robust_anchor.canonical_sha256({
            name: value for name, value in changed.items()
            if name != "content_sha256_before_self_field"
        })
    )
    with pytest.raises(RuntimeError, match="snapshot robust-objective"):
        line_search_v5.validate_completed_journal_v5(changed)
