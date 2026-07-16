#!/usr/bin/env python3

import build_sft_v430_refresh_preregistration_v47a as builder
import run_sft_equal_unit_matched_init_v47a as runner


def test_v47a_is_exact_v42i_recipe_on_refreshed_train_only_fold():
    value = builder.build()
    recipe = value["recipe"]
    assert value["dataset"]["rows"] == 459
    assert value["dataset"]["sha256"] == builder.refresh.EXPECTED["train_sha256"]
    assert recipe["learning_rate"] == 5.5e-5
    assert recipe["epochs_argument"] == 3.0
    assert recipe["expected_optimizer_steps"] == 48
    assert recipe["explicit_max_steps_cap"] == 48
    assert recipe["complete_three_all_row_passes_claimed"] is False
    assert recipe["expected_schedule_audit"]["max_steps"] == 48
    assert recipe["target_layers"] == [20, 21, 22, 23]
    assert recipe["world_size"] == 4
    assert recipe["all_four_gpu_activity_and_residency_required"] is True
    assert value["initialization"]["tensor_identity_sha256"] == (
        builder.source_contract.INITIAL_TENSOR_IDENTITY_SHA256_V42A
    )
    assert value["fold_binding"]["unexplained_unit_identity_changes"] == 0


def test_v47a_command_and_access_firewall():
    value = builder.build()
    command = value["recipe"]["command"]
    assert command[:4] == [
        str((builder.ROOT / ".venv/bin/torchrun").resolve()),
        "--standalone", "--nproc-per-node=4", str(runner.SFT_SCRIPT),
    ]
    assert command[command.index("--learning-rate") + 1] == "5.5e-05"
    assert command[command.index("--max-steps") + 1] == "48"
    assert command[command.index("--data") + 1] == str(builder.refresh.TRAIN)
    assert value["access_firewall"]["shadow_dev_opened_during_training"] is False
    assert value["access_firewall"]["eval_ood_holdout_opened"] is False
    assert value["access_firewall"]["post_training_evaluation_authorized"] is False
    assert "v46d" not in str(value).lower()
