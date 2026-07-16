#!/usr/bin/env python3

import json
import math
from pathlib import Path

import pytest

import build_sft_v434_sampling_midpoint_future_eval_v49d as future_builder
import build_sft_v434_sampling_midpoint_preregistration_v49d as builder
import run_sft_v434_sampling_midpoint_matched_init_v49d as runner
import seal_sft_v434_sampling_midpoint_input_v49d as sealer
import sft_lora_v434_equal_matched_init_v49d as equal_runtime
import sft_lora_v434_source50_matched_init_v49d as source50_runtime
import sft_v434_sampling_midpoint_weighting_v49d as weighting


def _rows():
    return [
        json.loads(line)
        for line in sealer.TRAIN.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_v49d_seals_one_exact_dataset_and_exact_interpolated_weights():
    assert runner.engine.file_sha256(sealer.TRAIN) == (
        weighting.v49b.v49a.V434_TRAIN_SHA256
    )
    computed = weighting.compute_v49d(_rows())
    equal_weights, equal = computed["v434_equal"]
    source_weights, source = computed["v434_source50"]
    assert equal["identity_sha256"] == weighting.EXPECTED["v434_equal"][
        "normalized_weight_sha256"
    ]
    assert source["identity_sha256"] == weighting.EXPECTED["v434_source50"][
        "normalized_weight_sha256"
    ]
    assert source["preregistered_multiplier_range_exact_rationals"] == [
        "5/6", "5/4"
    ]
    assert source["minimum_applied_multiplier"] == (
        weighting.MIN_SOURCE50_MULTIPLIER
    )
    assert source["maximum_applied_multiplier"] == 5 / 4
    assert math.isclose(math.fsum(equal_weights) / 448, 1.0, abs_tol=1e-15)
    assert math.isclose(math.fsum(source_weights) / 448, 1.0, abs_tol=1e-15)
    for row in source["per_row"]:
        before = float.fromhex(row["equal_normalized_weight_hex"])
        full = float.fromhex(row["v49a_full_normalized_weight_hex"])
        observed = float.fromhex(row["arm_normalized_weight_hex"])
        assert observed == 0.5 * before + 0.5 * full
    assert all(abs(row["mass_delta"]) <= 1e-15 for row in source["per_category"])


def test_v49d_input_manifest_binds_both_audits_and_never_opens_nontrain_rows():
    manifest = runner.validate_input_manifest_v49d()
    assert manifest["dataset"]["same_exact_bytes_for_both_arms"] is True
    assert manifest["controlled_contrast"]["source50_exact_multiplier_range"] == [
        "5/6", "5/4"
    ]
    assert manifest["controlled_contrast"]["no_other_lambda_or_hpo_arm_authorized"]
    assert manifest["document_disjoint_membership"][
        "train_dev_conflict_unit_intersection"
    ] == 0
    assert manifest["document_disjoint_membership"]["non_train_rows_opened"] is False
    assert manifest["access_firewall"]["shadow_semantics_opened"] is False
    assert manifest["access_firewall"]["eval_ood_holdout_semantics_opened"] is False


@pytest.mark.parametrize("module,assigner", [
    (equal_runtime, weighting.assign_equal_weights_v49d),
    (source50_runtime, weighting.assign_source50_weights_v49d),
])
def test_v49d_sft_wrappers_change_only_weight_assignment(monkeypatch, module, assigner):
    observed = {}
    original = module.v42a.assign_equal_unit_weights

    def fake_parent(argv):
        observed["argv"] = argv
        observed["assignment"] = module.v42a.assign_equal_unit_weights

    monkeypatch.setattr(module.v47a, "main", fake_parent)
    module.main(["marker"])
    assert observed == {"argv": ["marker"], "assignment": assigner}
    assert module.v42a.assign_equal_unit_weights is original


def test_v49d_joint_prereg_is_a_strict_matched_48_step_control():
    value = builder.build()
    assert value["training_arm_order"] == ["v434_equal", "v434_source50"]
    assert value["training_launch_authorized"] is True
    assert value["evaluation_launch_authorized"] is False
    contract = value["matched_control_contract"]
    assert contract["same_exact_v434_training_bytes"] is True
    assert contract["same_seed_prompt_encoding_batches_and_dataloader_order"] is True
    assert contract["same_48_steps_and_three_complete_row_passes"] is True
    assert contract["only_permitted_difference"] == "per-row Trainer example weights"
    assert contract["no_other_lambda_or_hpo_arm_authorized"] is True
    recipes = [value["training_arms"][arm]["recipe"] for arm in runner.ARMS]
    assert {recipe["common_recipe_identity_sha256"] for recipe in recipes} == {
        contract["recipe_identity_sha256"]
    }
    for recipe in recipes:
        assert recipe["expected_optimizer_steps"] == 48
        assert recipe["complete_all_row_passes"] == 3.0
        assert recipe["physical_gpu_ids"] == [0, 1, 2, 3]
        assert recipe["all_four_gpu_activity_and_residency_required"] is True
        assert recipe["learning_rate"] == 5.5e-5
        assert recipe["target_layers"] == [20, 21, 22, 23]


@pytest.mark.parametrize("arm", weighting.ARMS)
def test_v49d_each_arm_dry_run_is_gpu_free_and_fully_validated(
    arm, tmp_path, monkeypatch, capsys,
):
    value = builder.build()
    path = tmp_path / f"preregistration_{arm}.json"
    runner.engine.atomic_write_json(path, value)
    file_sha = runner.engine.file_sha256(path)
    args = builder.argument_vector(
        arm, file_sha, value["content_sha256_before_self_field"]
    )
    args[args.index("--preregistration") + 1] = str(path)
    monkeypatch.setattr(
        runner.engine, "assert_gpu_exclusive",
        lambda: (_ for _ in ()).throw(AssertionError("dry-run reached GPU")),
    )
    monkeypatch.setattr(
        runner.engine.subprocess, "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run launched torchrun")
        ),
    )
    assert runner.main([*args, "--dry-run"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["dataset"]["sha256"] == weighting.v49b.v49a.V434_TRAIN_SHA256
    assert output["preregistration"]["arm"] == arm
    assert output["command"][0:3] == [
        str((builder.ROOT / ".venv/bin/torchrun").resolve()),
        "--standalone", "--nproc-per-node=4",
    ]


def test_v49d_future_eval_is_two_wave_ood_first_and_nonlaunchable():
    value = future_builder.build(builder.PREREGISTRATION)
    assert value["evaluation_launch_authorized"] is False
    assert value["heldout_or_holdout_access_authorized"] is False
    waves = value["runtime_shape"]["two_full_fixed_waves"]
    assert [[record["arm"] for record in wave] for wave in waves] == [
        ["base_a", "base_b", "base_c", "base_d"],
        ["v434_equal_a", "v434_equal_b", "v434_source50_a", "v434_source50_b"],
    ]
    assert all([record["engine_index"] for record in wave] == [0, 1, 2, 3]
               for wave in waves)
    ood = value["ood_first_eligibility_gates"]
    assert ood["applied_independently_to_each_of_four_candidate_replicas"]
    assert ood["both_replicas_of_each_logical_candidate_must_pass"]
    assert ood["base_relative_ood_qa_mean_reward_delta_minimum"] == 0.0
    assert ood["base_relative_ood_qa_exact_count_delta_minimum"] == 0
    assert ood["base_relative_ood_prose_paired_document_bootstrap_lcb_minimum"] == 0.0
    direct = value["direct_hypothesis_gates"]
    assert direct["mean_replicated_ood_qa_reward_delta_minimum"] == 0.0
    assert direct["mean_replicated_ood_qa_exact_count_delta_minimum"] == 0
    assert direct["mean_replicated_shadow_reward_delta_minimum"] == 0.0008257591
    assert direct["paired_ood_qa_bootstrap_ci_role"] == "informational_not_a_gate"
    assert value["access_firewall"]["shadow_semantics_opened"] is False
    assert value["access_firewall"]["ood_qa_semantics_opened"] is False
    assert value["access_firewall"]["ood_prose_semantics_opened"] is False


def test_v49d_source_files_do_not_open_protected_inputs():
    source = Path(future_builder.__file__).read_text(encoding="utf-8")
    assert "PARENT_PROTOCOL.read_text" not in source
    assert "ood_qa_v3.jsonl" not in source
    assert "ood_prose_v3.jsonl" not in source
    assert "fold_3_shadow_dev.jsonl" not in source
