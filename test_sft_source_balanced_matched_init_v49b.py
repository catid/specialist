#!/usr/bin/env python3

import json
from pathlib import Path

import pytest

import build_sft_source_balanced_matched_init_preregistration_v49b as builder
import run_sft_source_balanced_matched_init_v49b as runner
import seal_sft_source_balanced_input_v49b as sealer
import sft_lora_source_balanced_matched_init_v49b as sft_runtime
import sft_source_balanced_weighting_v49b as weighting


def _rows():
    return [
        json.loads(line)
        for line in sealer.TRAIN.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_v49b_sealed_v434_input_and_all_weight_identities_are_exact():
    assert runner.engine.file_sha256(sealer.TRAIN) == (
        weighting.v49a.V434_TRAIN_SHA256
    )
    assert runner.engine.file_sha256(sealer.WEIGHT_AUDIT) == (
        runner.WEIGHT_AUDIT_FILE_SHA256
    )
    assert runner.engine.file_sha256(sealer.INPUT_MANIFEST) == (
        runner.INPUT_MANIFEST_FILE_SHA256
    )
    weights, audit = weighting.compute_source_balanced_weights_v49b(_rows())
    compact = weighting.compact_weighting_audit_v49b(audit)
    assert len(weights) == compact["rows"] == 448
    assert compact == runner.EXPECTED_WEIGHTING_AUDIT
    assert compact["identity_sha256"] == (
        weighting.ALTERNATIVE_NORMALIZED_WEIGHT_SHA256
    )
    assert compact["per_row_identity_sha256"] == (
        "12175f8a48150f2ee04942334a12d2255da73d1d20edfe5fe391f2f37313f90d"
    )
    assert compact["per_source_identity_sha256"] == (
        weighting.SOURCE_MASS_TABLE_SHA256
    )
    assert compact["per_category_identity_sha256"] == (
        "e248f3d1eea9de0445248189bc4b9264447978d4736cb1b759b5a60c083f08d9"
    )
    assert compact["trainer_example_weight_identity_sha256"] == (
        "6d382148030767d62e5dfa3e887a4b0630c1e76a1757d14270616f0c5b2eb51e"
    )
    assert compact["minimum_applied_multiplier"] >= 2 / 3 - 1e-15
    assert compact["maximum_applied_multiplier"] <= 1.5
    assert all(abs(row["mass_delta"]) <= 1e-15
               for row in audit["per_category"])


def test_v49b_only_training_code_change_is_weight_assignment(monkeypatch):
    observed = {}
    original = sft_runtime.v42a.assign_equal_unit_weights

    def fake_parent(argv):
        observed["argv"] = argv
        observed["assignment"] = sft_runtime.v42a.assign_equal_unit_weights

    monkeypatch.setattr(sft_runtime.v47a, "main", fake_parent)
    sft_runtime.main(["marker"])
    assert observed == {
        "argv": ["marker"],
        "assignment": weighting.assign_source_balanced_weights_v49b,
    }
    assert sft_runtime.v42a.assign_equal_unit_weights is original


def test_v49b_command_preserves_v47c_v42i_runtime_recipe():
    value = builder.build()
    recipe = value["recipe"]
    command = recipe["command"]
    parent_args = builder.launcher.v47c.parser().parse_args(
        builder.argument_vector()
    )
    parent_command = builder.launcher.build_train_command(parent_args)
    assert command == parent_command
    assert command[:4] == [
        str((builder.ROOT / ".venv/bin/torchrun").resolve()),
        "--standalone", "--nproc-per-node=4", str(runner.SFT_SCRIPT),
    ]
    assert command[command.index("--data") + 1] == str(sealer.TRAIN)
    assert command[command.index("--expected-weight-identity-sha256") + 1] == (
        weighting.ALTERNATIVE_NORMALIZED_WEIGHT_SHA256
    )
    assert command[command.index("--max-steps") + 1] == "48"
    assert command[command.index("--learning-rate") + 1] == "5.5e-05"
    assert recipe["expected_optimizer_steps"] == 48
    assert recipe["per_device_batch_size"] == 7
    assert recipe["effective_global_batch_size"] == 28
    assert recipe["target_layers"] == [20, 21, 22, 23]
    assert recipe["rank"] == 32 and recipe["lora_alpha"] == 64
    assert recipe["prompt_mode"] == "es_exact"
    assert recipe["expected_encoding_audit"] == runner.EXPECTED_ENCODING_AUDIT
    assert value["initialization"]["tensor_identity_sha256"] == (
        builder.source_contract.INITIAL_TENSOR_IDENTITY_SHA256_V42A
    )


def test_v49b_membership_is_document_disjoint_without_nontrain_access():
    manifest = runner.validate_input_manifest_v49b()
    disjoint = manifest["document_disjoint_membership"]
    assert manifest["dataset"]["root_membership_sha256"] == (
        weighting.ROOT_MEMBERSHIP_SHA256
    )
    assert disjoint["train_dev_conflict_unit_intersection"] == 0
    assert not any(disjoint["train_dev_edge_identity_intersections"].values())
    assert disjoint["membership_replayed_by_content_free_root_identity_only"] is True
    assert disjoint["non_train_rows_opened"] is False
    assert manifest["access_firewall"]["shadow_semantics_opened"] is False
    assert manifest["access_firewall"]["eval_ood_holdout_semantics_opened"] is False


def test_v49b_dry_run_validates_everything_without_gpu_launch(
    tmp_path, monkeypatch, capsys,
):
    value = builder.build()
    path = tmp_path / "preregistration_v49b.json"
    runner.engine.atomic_write_json(path, value)
    file_sha = runner.engine.file_sha256(path)
    args = builder.argument_vector(
        file_sha, value["content_sha256_before_self_field"]
    )
    index = args.index("--preregistration") + 1
    args[index] = str(path)
    monkeypatch.setattr(
        runner.engine, "assert_gpu_exclusive",
        lambda: (_ for _ in ()).throw(
            AssertionError("V49B dry-run reached GPU preflight")
        ),
    )
    monkeypatch.setattr(
        runner.engine.subprocess, "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("V49B dry-run launched torchrun")
        ),
    )
    assert runner.main([*args, "--dry-run"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["dataset"]["sha256"] == weighting.v49a.V434_TRAIN_SHA256
    assert output["preregistration"]["weighting_audit_content_sha256"] == (
        runner.WEIGHT_AUDIT_CONTENT_SHA256
    )


def test_v49b_sources_bind_no_protected_dataset_path():
    source = "\n".join(
        Path(path).read_text(encoding="utf-8").lower()
        for path in (
            weighting.__file__, sealer.__file__, sft_runtime.__file__,
            runner.__file__, builder.__file__,
        )
    )
    for forbidden_path in (
        "fold_3_shadow_dev.jsonl", "ood_qa_v3", "ood_prose_v3",
        "eval_qa", "holdout_eval",
    ):
        assert forbidden_path not in source
