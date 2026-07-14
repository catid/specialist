import json
from pathlib import Path

import pytest

from run_eggroll_es_anchor_line_search import (
    ALLOWED_EVAL_SPLITS,
    execute_line_search,
    load_allowlisted_eval_datasets,
    parse_target_alphas,
    strict_qa_gate,
    summarize_eval_file,
    validate_eval_splits,
)
from train_eggroll_es_specialist_anchor import coefficient_sha256


def eval_rows(rewards):
    return [
        {
            "prompt": f"prompt-{index}",
            "answer": f"answer-{index}",
            "reward": reward,
            "format": "exact" if reward == 1.0 else (
                "partial" if reward > 0.0 else "incorrect"
            ),
        }
        for index, reward in enumerate(rewards)
    ]


class FakeTrainer:
    def __init__(self, directory, fail_iteration=None, mutate_plan=False):
        self.logging_dir = str(directory)
        (directory / "eval-output").mkdir(parents=True)
        self.eval_dataloader_dict = {
            "validation": object(), "ood_qa": object(),
        }
        self.fail_iteration = fail_iteration
        self.mutate_plan = mutate_plan
        self.events = []
        self.plan = None

    def eval_step(self, iteration):
        self.events.append(("eval", iteration))
        if iteration == self.fail_iteration:
            raise RuntimeError("simulated evaluation failure")
        alpha = 0.0 if self.plan is None else self.plan["applied_alpha"]
        validation = [0.0, min(1.0, alpha * 10)]
        ood = [1.0, 0.25]
        for split, rows in (("validation", validation), ("ood_qa", ood)):
            path = (
                Path(self.logging_dir) / "eval-output"
                / f"model_eval_task{split}_iteration{iteration + 1}.json"
            )
            path.write_text(json.dumps(eval_rows(rows)))

    def estimate_step_coefficients(self, iteration, seeds, prompts, answers):
        self.events.append(("estimate", iteration, list(seeds)))
        coefficients = [0.5, -0.5]
        self.plan = {
            "seeds": list(seeds),
            "coefficients": coefficients,
            "coefficient_sha256": coefficient_sha256(seeds, coefficients),
            "projection": {"decision": "project_to_anchor_cone"},
            "applied_alpha": 0.0,
            "journal_path": "seed-plan.json",
        }
        return self.plan

    def apply_seed_coefficients(self, plan, target_alpha):
        self.events.append(("apply", target_alpha))
        plan["applied_alpha"] = target_alpha
        if self.mutate_plan:
            plan["coefficients"][0] += 0.1


@pytest.mark.parametrize(
    "value,message",
    [
        ("", "empty"),
        ("0.1,0.2", "begin"),
        ("0,0", "increasing"),
        ("0,0.2,0.1", "increasing"),
        ("0,nan", "finite"),
        ("0,nope", "numeric"),
    ],
)
def test_target_alpha_parser_fails_closed(value, message):
    with pytest.raises(ValueError, match=message):
        parse_target_alphas(value)


def test_target_alpha_parser_requires_explicit_zero():
    assert parse_target_alphas("0,0.000025,0.00005") == [
        0.0, 0.000025, 0.00005,
    ]


def test_eval_allowlist_rejects_heldout_and_reordering():
    assert validate_eval_splits("validation,ood_qa") == ALLOWED_EVAL_SPLITS
    with pytest.raises(ValueError, match="exactly"):
        validate_eval_splits("validation,ood_qa,heldout")
    with pytest.raises(ValueError, match="exactly"):
        validate_eval_splits("ood_qa,validation")


def test_eval_loader_never_opens_heldout_directory(tmp_path):
    root = tmp_path / "eval"
    root.mkdir()
    (root / "dataset_dict.json").write_text(json.dumps({
        "splits": ["validation", "ood_qa", "heldout"],
    }))
    opened = []

    def loader(path):
        opened.append(Path(path).name)
        return Path(path).name

    datasets = load_allowlisted_eval_datasets(
        root, ALLOWED_EVAL_SPLITS, loader=loader,
    )
    assert datasets == {"validation": "validation", "ood_qa": "ood_qa"}
    assert opened == ["validation", "ood_qa"]
    assert "heldout" not in opened


def test_eval_summary_hashes_rows_and_strict_gate_checks_all_metrics(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(eval_rows([1.0, 0.25, 0.0])))
    summary = summarize_eval_file(path)
    assert summary["rows"] == 3
    assert summary["exact"] == 1
    assert summary["nonzero"] == 2
    assert len(summary["sha256"]) == 64
    worse = dict(summary, mean_reward=summary["mean_reward"] - 0.01)
    assert strict_qa_gate(summary, worse)["passed"] is False
    fewer_exact = dict(summary, exact=0)
    assert strict_qa_gate(summary, fewer_exact)["passed"] is False
    fewer_nonzero = dict(summary, nonzero=1)
    assert strict_qa_gate(summary, fewer_nonzero)["passed"] is False


def test_search_estimates_once_and_applies_only_monotonic_targets(tmp_path):
    trainer = FakeTrainer(tmp_path / "run")
    journal_path = tmp_path / "run" / "journal.json"
    journal = execute_line_search(
        trainer,
        targets=[0.0, 0.1, 0.25],
        seeds=[11, 22],
        input_prompts=["prompt"],
        target_answers=["answer"],
        snapshot={"sha256": "snapshot"},
        journal_path=journal_path,
    )
    assert trainer.events == [
        ("eval", 0),
        ("estimate", 0, [11, 22]),
        ("apply", 0.1),
        ("eval", 1),
        ("apply", 0.25),
        ("eval", 2),
    ]
    assert journal["status"] == "complete"
    assert [state["target_alpha"] for state in journal["states"]] == [
        0.0, 0.1, 0.25,
    ]
    assert [state["alpha_increment"] for state in journal["states"]] == [
        0.0, 0.1, 0.15,
    ]
    assert len({
        state["coefficient_sha256"] for state in journal["states"]
    }) == 1
    assert json.loads(journal_path.read_text())["status"] == "complete"


def test_strict_prose_gate_is_recorded_for_every_alpha(tmp_path):
    trainer = FakeTrainer(tmp_path / "run")
    labels = []

    def scorer(current_trainer, label):
        labels.append(label)
        alpha = (
            0.0 if current_trainer.plan is None
            else current_trainer.plan["applied_alpha"]
        )
        path = Path(current_trainer.logging_dir) / f"prose-{label}.json"
        path.write_text("[]")
        return {
            "results_path": str(path),
            "item_count": 1,
            "scored_token_count": 10,
            "mean_token_logprob": -1.0 - alpha,
        }

    def comparator(baseline, candidate):
        delta = (
            candidate["mean_token_logprob"]
            - baseline["mean_token_logprob"]
        )
        return {"delta": delta, "max_degradation": 0.0, "passed": delta >= 0}

    journal = execute_line_search(
        trainer,
        targets=[0.0, 0.1],
        seeds=[11, 22],
        input_prompts=["prompt"],
        target_answers=["answer"],
        snapshot={},
        journal_path=tmp_path / "run" / "journal.json",
        prose_scorer=scorer,
        prose_comparator=comparator,
    )
    assert labels == ["alpha_0000", "alpha_0001"]
    assert journal["states"][0]["ood_prose_gate"]["passed"] is True
    assert journal["states"][1]["ood_prose_gate"]["passed"] is False
    assert journal["states"][1]["strict_guards_passed"] is False
    assert len(journal["states"][1]["ood_prose"]["results_sha256"]) == 64


def test_prose_scorer_without_comparator_fails_closed(tmp_path):
    trainer = FakeTrainer(tmp_path / "run")
    with pytest.raises(ValueError, match="enabled together"):
        execute_line_search(
            trainer,
            targets=[0.0, 0.1],
            seeds=[11, 22],
            input_prompts=["prompt"],
            target_answers=["answer"],
            snapshot={},
            journal_path=tmp_path / "run" / "journal.json",
            prose_scorer=lambda trainer, label: {},
        )


def test_failure_leaves_atomic_failed_journal_without_rollback(tmp_path):
    trainer = FakeTrainer(tmp_path / "run", fail_iteration=1)
    journal_path = tmp_path / "run" / "journal.json"
    with pytest.raises(RuntimeError, match="simulated"):
        execute_line_search(
            trainer,
            targets=[0.0, 0.1, 0.2],
            seeds=[11, 22],
            input_prompts=["prompt"],
            target_answers=["answer"],
            snapshot={},
            journal_path=journal_path,
        )
    journal = json.loads(journal_path.read_text())
    assert journal["status"] == "failed"
    assert len(journal["states"]) == 1
    assert journal["failure"]["completed_state_count"] == 1
    assert journal["in_progress"]["phase"] == "weights_incremented"
    assert trainer.events[-2:] == [("apply", 0.1), ("eval", 1)]
    assert not any(event[0] == "rollback" for event in trainer.events)
    assert not journal_path.with_name(journal_path.name + ".tmp").exists()


def test_changed_fixed_coefficients_fail_before_next_state(tmp_path):
    trainer = FakeTrainer(tmp_path / "run", mutate_plan=True)
    journal_path = tmp_path / "run" / "journal.json"
    with pytest.raises(ValueError, match="changed"):
        execute_line_search(
            trainer,
            targets=[0.0, 0.1],
            seeds=[11, 22],
            input_prompts=["prompt"],
            target_answers=["answer"],
            snapshot={},
            journal_path=journal_path,
        )
    journal = json.loads(journal_path.read_text())
    assert journal["status"] == "failed"
    assert trainer.events[-1] == ("apply", 0.1)


def test_existing_journal_forbids_resume(tmp_path):
    trainer = FakeTrainer(tmp_path / "run")
    journal_path = tmp_path / "run" / "journal.json"
    journal_path.write_text("{}")
    with pytest.raises(ValueError, match="resume"):
        execute_line_search(
            trainer,
            targets=[0.0, 0.1],
            seeds=[11, 22],
            input_prompts=["prompt"],
            target_answers=["answer"],
            snapshot={},
            journal_path=journal_path,
        )
