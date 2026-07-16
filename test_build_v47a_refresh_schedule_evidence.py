#!/usr/bin/env python3

import build_v47a_refresh_schedule_evidence as subject


def test_refresh_delta_has_no_unexplained_identity_or_membership_drift():
    value = subject.build()
    delta = value["refresh_delta"]
    assert delta["component_membership_sets_identical"] is True
    assert delta["unit_row_multiplicities_identical"] is True
    assert delta["units_with_row_multiplicity_change"] == 0
    assert delta["unexplained_unit_identity_changes"] == 0
    assert delta["unedited_units_with_exact_identity_preserved"] == 222
    assert delta["fold_3"]["rows_before"] == 83
    assert delta["fold_3"]["rows_after"] == 72


def test_installed_drop_last_schedule_has_explicit_48_step_cap():
    value = subject.build()
    schedule = value["step_schedule"]
    assert schedule["epochs_argument"] == 3
    assert schedule["expected_completed_dataloader_epochs"] == 3
    assert schedule["optimizer_steps_per_dataloader_epoch"] == 16
    assert schedule["expected_optimizer_steps"] == 48
    assert schedule["explicit_max_steps_cap"] == 48
    assert schedule["max_steps_is_terminal_authority"] is True
    assert schedule["complete_three_all_row_passes_claimed"] is False
    assert schedule["source_rows_not_emitted_per_dataloader_epoch"] == 11
    assert value["access_firewall"]["shadow_dev_opened"] is False
    assert value["access_firewall"]["eval_ood_holdout_or_benchmark_opened"] is False
