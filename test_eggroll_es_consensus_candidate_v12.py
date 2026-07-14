import copy
import json
from pathlib import Path

import pytest
from datasets import load_from_disk

import build_eggroll_es_v11_evidence_v12 as evidence_v12
import run_eggroll_es_anchor_line_search as driver_v1
import run_eggroll_es_anchor_variance_v10 as driver_v10
import run_eggroll_es_consensus_candidate_v12 as driver_v12
import train_eggroll_es_specialist as base
import train_eggroll_es_specialist_anchor_v11 as anchor_v11
import train_eggroll_es_specialist_anchor_v11b as anchor_v11b
import train_eggroll_es_specialist_anchor_v12 as anchor_v12


PLAN_SHA = driver_v12.MIDDLE_LATE_PLAN_SHA256_V12


def load_bundle():
    spec = anchor_v11.FROZEN_STABILITY_PLANS_V11[PLAN_SHA]
    return anchor_v12.load_frozen_layer_plan_v12(
        spec["path"], expected_file_sha256=spec["file_sha256"],
        expected_plan_sha256=PLAN_SHA,
        expected_model_config_sha256=anchor_v11.MODEL_CONFIG_SHA256_V11,
    )


def cli(stage="preseal", extra=None):
    spec = anchor_v11.FROZEN_STABILITY_PLANS_V11[PLAN_SHA]
    values = [
        "--layer-plan-json", str(spec["path"]),
        "--expected-layer-plan-file-sha256", spec["file_sha256"],
        "--expected-layer-plan-sha256", PLAN_SHA,
        "--expected-model-config-sha256", anchor_v11.MODEL_CONFIG_SHA256_V11,
        "--v12-stage", stage, "--v12-dry-run", "--v12-fixture-evidence",
    ]
    return values + list(extra or [])


def resident_cross_from_v10():
    journal = json.loads((
        Path(__file__).resolve().parent
        / "experiments/eggroll_es_hpo/runs/"
        "snapshot794_layer_v10_middle_late_antithetic_cross_"
        "d43d44_a43a44_basis20260714/alpha_line_search.json"
    ).read_text())
    return anchor_v11._resident_artifact_v11(
        journal["coefficient_plan"]["antithetic_cross_v10"]
    )


def test_v12_consensus_is_exact_and_collapses_duplicate_anchor_cells():
    consensus = anchor_v12.consensus_from_resident_cross_v12(
        resident_cross_from_v10()
    )
    anchor_v12.validate_consensus_v12(consensus)
    assert consensus["coefficient_sha256"] == (
        anchor_v12.EXPECTED_CONSENSUS_COEFFICIENT_SHA256_V12
    )
    assert consensus["cell_cosines"] == {
        "D43": pytest.approx(anchor_v12.EXPECTED_CELL_COSINE_V12, abs=3e-16),
        "D44": pytest.approx(anchor_v12.EXPECTED_CELL_COSINE_V12, abs=3e-16),
    }
    assert consensus["anchor_cosine"] == pytest.approx(
        anchor_v12.EXPECTED_ANCHOR_COSINE_V12, abs=3e-16,
    )
    tampered = resident_cross_from_v10()
    tampered["cells"]["D43xA44"]["coefficients"][0] += 1e-12
    with pytest.raises(RuntimeError, match="duplicates"):
        anchor_v12.consensus_from_resident_cross_v12(tampered)


def test_v12_c45_c46_are_exact_and_disjoint_from_d43_d44_and_each_other():
    dataset = load_from_disk(str(driver_v12.FROZEN_TRAIN_DATASET_V12))["train"]
    screens = anchor_v12.build_disjoint_screens_v12(
        dataset, base.build_train_loader,
    )
    assert set(screens) == {"C45", "C46"}
    assert all(len(rows[0]) == len(rows[1]) == 64 for rows in screens.values())
    d43_d44 = []
    for seed in (43, 44):
        questions, answers = next(iter(base.build_train_loader(dataset, 64, seed)))
        d43_d44.extend(zip(questions, answers))
    assert not set(zip(*screens["C45"])).intersection(d43_d44)
    assert not set(zip(*screens["C46"])).intersection(d43_d44)
    assert not set(zip(*screens["C45"])).intersection(zip(*screens["C46"]))


def _documents(delta):
    return [
        {
            "document_id": f"doc-{index}", "scored_token_count": 1,
            "sum_token_logprob": float(index) + delta,
        }
        for index in range(8)
    ]


def test_v12_paired_lcb_and_smallest_eligible_policy_fail_closed():
    positive = anchor_v12._paired_lcb_v12(_documents(0.0), _documents(0.5))
    zero = anchor_v12._paired_lcb_v12(_documents(0.0), _documents(0.0))
    assert positive["lower_confidence_bound"] > 0.0
    assert zero["lower_confidence_bound"] == 0.0
    states = [
        {"alpha": 0.0, "gate": {"eligible": False}},
        {"alpha": anchor_v12.ALPHA_GRID_V12[1], "gate": {"eligible": True}},
        {"alpha": anchor_v12.ALPHA_GRID_V12[2], "gate": {"eligible": True}},
    ]
    assert anchor_v12.select_smallest_eligible_v12(states)["alpha"] == (
        anchor_v12.ALPHA_GRID_V12[1]
    )
    states[1]["gate"]["eligible"] = False
    states[2]["gate"]["eligible"] = False
    assert anchor_v12.select_smallest_eligible_v12(states) is None


def test_v12_dry_run_freezes_four_gpu_and_preseal_surface(capsys):
    payload = driver_v12.main(cli())
    assert payload["schema"] == "eggroll-es-consensus-candidate-dry-run-v12"
    assert payload["stage"] == "preseal"
    assert payload["benchmark_surface_available"] is False
    assert payload["four_gpu_contract"] == {
        "engines": 4, "tp": 1, "gpu_ids": [0, 1, 2, 3],
    }
    assert "consensus-candidate-dry-run-v12" in capsys.readouterr().out
    with pytest.raises(ValueError, match="rejects benchmark/OOD"):
        driver_v12.main(cli(extra=["--eval-dataset", "/tmp/eval"]))
    with pytest.raises(ValueError, match="rejects benchmark/OOD"):
        driver_v12.main(cli(extra=["--ood-prose-jsonl", "/tmp/ood.jsonl"]))
    with pytest.raises(ValueError, match="rejects benchmark/OOD"):
        driver_v12.main(cli(extra=["--candidate-seal", "/tmp/heldout.json"]))


def test_v12_real_stage_cannot_use_fixture_and_requires_bound_evidence(
    monkeypatch,
):
    with pytest.raises(RuntimeError, match="fixture evidence is forbidden"):
        driver_v12.load_v11_evidence_v12("unused", dry_run=False, fixture=True)
    evidence = driver_v12.load_v11_evidence_v12(
        driver_v12.V11_EVIDENCE_PATH_V12, dry_run=False, fixture=False,
    )
    assert evidence["content_sha256_before_self_field"] == (
        driver_v12.EXPECTED_V11_EVIDENCE_CONTENT_SHA256_V12
    )
    monkeypatch.setattr(
        driver_v12, "EXPECTED_V11_EVIDENCE_FILE_SHA256_V12", "0" * 64,
    )
    with pytest.raises(RuntimeError, match="file identity"):
        driver_v12.load_v11_evidence_v12(
            driver_v12.V11_EVIDENCE_PATH_V12, dry_run=False, fixture=False,
        )


def test_v12_candidate_seal_and_confirmation_are_exact_hash_bindings(tmp_path):
    seal = {
        "schema": "eggroll-es-immutable-candidate-seal-v12",
        "alpha": anchor_v12.ALPHA_GRID_V12[1],
        "coefficient_sha256": anchor_v12.EXPECTED_CONSENSUS_COEFFICIENT_SHA256_V12,
        "selection": {
            "eligible": True,
            "gate_content_sha256": "1" * 64,
            "policy": "smallest_positive_eligible_alpha",
        },
        "preseal_report": {
            "path": str(tmp_path / "preseal_screen.json"),
            "file_sha256": "2" * 64,
            "content_sha256": "3" * 64,
        },
        "v11_evidence_content_sha256": "4" * 64,
        "implementation_bundle_sha256": "5" * 64,
        "benchmark_content_opened_before_seal": False,
        "fresh_direct_confirmation_required": True,
        "release_fallback_allowed": False,
    }
    seal["content_sha256_before_self_field"] = driver_v1.canonical_sha256(seal)
    seal_path = tmp_path / "candidate.json"
    driver_v1.atomic_write_json(seal_path, seal)
    args = type("Args", (), {
        "candidate_seal": str(seal_path),
        "expected_candidate_seal_file_sha256": driver_v1.file_sha256(seal_path),
        "expected_candidate_seal_content_sha256": seal[
            "content_sha256_before_self_field"
        ],
    })()
    loaded = driver_v12.load_candidate_seal_v12(args)
    assert loaded == seal
    changed = copy.deepcopy(args)
    changed.expected_candidate_seal_file_sha256 = "0" * 64
    with pytest.raises(RuntimeError, match="file identity"):
        driver_v12.load_candidate_seal_v12(changed)
    extra = copy.deepcopy(seal)
    extra["unreviewed_surface"] = True
    extra["content_sha256_before_self_field"] = driver_v1.canonical_sha256({
        key: value for key, value in extra.items()
        if key != "content_sha256_before_self_field"
    })
    extra_path = tmp_path / "candidate-extra.json"
    driver_v1.atomic_write_json(extra_path, extra)
    extra_args = copy.deepcopy(args)
    extra_args.candidate_seal = str(extra_path)
    extra_args.expected_candidate_seal_file_sha256 = driver_v1.file_sha256(
        extra_path
    )
    extra_args.expected_candidate_seal_content_sha256 = extra[
        "content_sha256_before_self_field"
    ]
    with pytest.raises(RuntimeError, match="release-eligible"):
        driver_v12.load_candidate_seal_v12(extra_args)


def test_v12_worker_and_trainer_keep_exact_v11_four_engine_contract():
    trainer = anchor_v12.load_trainer(load_bundle())
    assert trainer.estimate_step_coefficients.__module__ == anchor_v12.__name__
    assert trainer.apply_seed_coefficients.__module__ == anchor_v12.__name__
    assert anchor_v12.REQUIRED_ENGINE_COUNT == 4
    assert anchor_v12.ALPHA_GRID_V12 == [0.0, 7.8125e-7, 1.5625e-6]
    assert anchor_v11b.DualManifestResidentSignContractMixinV11B in trainer.__mro__


def _mock_v11g_validation(monkeypatch, tmp_path):
    run_dir = tmp_path / "v11g-run"
    run_dir.mkdir()
    journal_path = run_dir / "alpha_line_search.json"
    cross = resident_cross_from_v10()
    journal = {
        "schema": "eggroll-es-anchor-alpha-line-search-v11c",
        "seeds": [1, 2, 3],
        "policy": {"frozen": True},
        "coefficient_plan": {"resident_sign_cross_v11": cross},
        "content_sha256_before_self_field": "6" * 64,
    }
    driver_v1.atomic_write_json(journal_path, journal)
    attempt_path = tmp_path / ".v11g.launch_attempt.json"
    journal_binding = {
        "schema": "eggroll-es-v11g-journal-binding",
        "path": str(journal_path.resolve()),
        "file_sha256": driver_v1.file_sha256(journal_path),
        "content_sha256": journal["content_sha256_before_self_field"],
        "journal_schema": journal["schema"],
        "seed_sha256": driver_v1.canonical_sha256(journal["seeds"]),
        "policy_sha256": driver_v1.canonical_sha256(journal["policy"]),
    }
    failure = {"binding_sha256": "9" * 64}
    implementation = {"implementation": True}
    effective_cli = {"effective": True}
    seed_audit = {"seed": True}
    policy_audit = {"policy": True}
    source = {
        "schema": "eggroll-es-v11g-source",
        "git_head": "a" * 40,
        "relative_path": "run_eggroll_es_anchor_equivalence_v11g.py",
        "file_sha256": "b" * 64,
    }
    attempt = {
        "schema": "eggroll-es-durable-launch-attempt-v11g",
        "status": "complete",
        "phase": "after_v11c_driver_main",
        "experiment_name": evidence_v12.equivalence_v11g.EXPERIMENT_NAME_V11G,
        "run_directory": str(run_dir.resolve()),
        "source_provenance": source,
        "v11c_implementation": implementation,
        "v11f_failure_evidence": failure,
        "effective_cli": effective_cli,
        "seed_forwarding_audit": seed_audit,
        "policy_forwarding_audit": policy_audit,
        "intended_recipe_or_data_changed_from_v11f": False,
        "effective_completion_policy_corrected": True,
        "target_alpha_zero_only": True,
        "model_update_applied": False,
        "prior_v11f_baseline_validation_and_ood_scored": True,
        "v11g_baseline_validation_and_ood_scored": True,
        "sealed_evaluation_data_opened_or_scored": False,
        "run_directory_exists_after_attempt": True,
        "v11c_journal_exists_after_attempt": True,
        "journal_binding": journal_binding,
    }
    attempt["content_sha256_before_self_field"] = driver_v1.canonical_sha256(
        attempt
    )
    driver_v1.atomic_write_json(attempt_path, attempt)

    monkeypatch.setattr(evidence_v12, "V11G_RUN_V12", run_dir.resolve())
    monkeypatch.setattr(evidence_v12, "V11G_ATTEMPT_V12", attempt_path.resolve())
    monkeypatch.setattr(evidence_v12, "V11G_JOURNAL_V12", journal_path.resolve())
    monkeypatch.setattr(evidence_v12, "_validate_source_v12", lambda value: value)
    monkeypatch.setattr(
        evidence_v12.equivalence_v11g, "bind_v11f_failure_v11g",
        lambda: failure,
    )
    monkeypatch.setattr(
        evidence_v12.equivalence_v11g.driver_v11e,
        "audit_v11c_implementation_v11e", lambda: implementation,
    )
    monkeypatch.setattr(
        evidence_v12.equivalence_v11g, "audit_effective_cli_v11g",
        lambda _argv: effective_cli,
    )
    monkeypatch.setattr(
        evidence_v12.equivalence_v11g.driver_v11f,
        "seed_forwarding_audit_v11f", lambda _seeds: seed_audit,
    )
    monkeypatch.setattr(
        evidence_v12.equivalence_v11g, "policy_forwarding_audit_v11g",
        lambda _policy: policy_audit,
    )
    monkeypatch.setattr(
        evidence_v12.equivalence_v11g, "_patch_v11c_globals", lambda: None,
    )
    monkeypatch.setattr(
        evidence_v12.equivalence_v11g, "_restore_v11c_globals", lambda _prior: None,
    )
    audit = {
        "content_sha256": journal["content_sha256_before_self_field"],
        "equivalence": {"all_exact": True, "binding_sha256": "c" * 64},
        "resident": {"content_sha256": cross["content_sha256_before_self_field"]},
    }
    monkeypatch.setattr(
        evidence_v12.equivalence_v11g.driver_v11c,
        "validate_completed_journal_v11c", lambda _journal: audit,
    )
    monkeypatch.setattr(
        evidence_v12.equivalence_v11g, "_completed_journal_binding",
        lambda _run_dir: journal_binding,
    )
    return attempt_path, attempt


def test_v12_evidence_builder_requires_exact_completed_v11g_and_mints_consensus(
    monkeypatch, tmp_path,
):
    attempt_path, attempt = _mock_v11g_validation(monkeypatch, tmp_path)
    evidence = evidence_v12.build_evidence(attempt_path)
    assert evidence["schema"] == (
        "eggroll-es-v11g-compact-equivalence-evidence-v12"
    )
    assert evidence["passed"] is True
    assert evidence["consensus"]["coefficient_sha256"] == (
        anchor_v12.EXPECTED_CONSENSUS_COEFFICIENT_SHA256_V12
    )
    assert evidence["content_sha256_before_self_field"] == (
        driver_v1.canonical_sha256({
            key: value for key, value in evidence.items()
            if key != "content_sha256_before_self_field"
        })
    )

    attempt["unreviewed_surface"] = True
    attempt["content_sha256_before_self_field"] = driver_v1.canonical_sha256({
        key: value for key, value in attempt.items()
        if key != "content_sha256_before_self_field"
    })
    driver_v1.atomic_write_json(attempt_path, attempt)
    with pytest.raises(RuntimeError, match="completed V11g launch attempt"):
        evidence_v12.build_evidence(attempt_path)


def test_v12_evidence_output_uses_exclusive_creation(tmp_path):
    output = tmp_path / "evidence.json"
    evidence_v12._exclusive_write_json(output, {"passed": True})
    assert json.loads(output.read_text()) == {"passed": True}
    with pytest.raises(ValueError, match="already exists"):
        evidence_v12._exclusive_write_json(output, {"passed": True})
