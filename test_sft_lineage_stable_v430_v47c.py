#!/usr/bin/env python3

from pathlib import Path

import build_sft_lineage_stable_v430_preregistration_v47c as builder
import run_sft_equal_unit_matched_init_v47c as runner


def test_v47c_is_exact_v42i_recipe_on_lineage_stable_v430_train():
    value = builder.build()
    recipe = value["recipe"]
    fold = value["fold_binding"]
    assert value["dataset"]["rows"] == 448
    assert value["dataset"]["sha256"] == builder.refresh.EXPECTED["train_sha256"]
    assert fold["train_conflict_units"] == 208
    assert fold["shadow_rows_aggregate_only"] == 83
    assert fold["shadow_conflict_units_aggregate_only"] == 51
    assert fold["original_root_membership_assignment_retained"] is True
    assert fold["fold_assignment_changes"] == 0
    assert recipe["learning_rate"] == 5.5e-5
    assert recipe["explicit_max_steps_cap"] == 48
    assert recipe["expected_optimizer_steps"] == 48
    assert recipe["complete_all_row_passes"] == 3.0
    assert recipe["target_layers"] == [20, 21, 22, 23]
    assert recipe["world_size"] == 4
    assert value["initialization"]["tensor_identity_sha256"] == (
        builder.source_contract.INITIAL_TENSOR_IDENTITY_SHA256_V42A
    )


def test_v47c_command_and_access_firewall_have_no_protected_binding():
    value = builder.build()
    command = value["recipe"]["command"]
    assert command[:4] == [
        str((builder.ROOT / ".venv/bin/torchrun").resolve()),
        "--standalone", "--nproc-per-node=4", str(runner.SFT_SCRIPT),
    ]
    assert command[command.index("--max-steps") + 1] == "48"
    assert command[command.index("--learning-rate") + 1] == "5.5e-05"
    assert command[command.index("--data") + 1] == str(builder.refresh.TRAIN)
    assert value["access_firewall"]["shadow_dev_opened_during_training"] is False
    assert value["access_firewall"]["eval_ood_holdout_opened"] is False
    assert value["access_firewall"]["post_training_evaluation_authorized"] is False
    source = "\n".join(
        Path(path).read_text(encoding="utf-8").lower()
        for path in (builder.__file__, runner.__file__)
    )
    assert "v47b" not in source
    assert "v46d" not in source
