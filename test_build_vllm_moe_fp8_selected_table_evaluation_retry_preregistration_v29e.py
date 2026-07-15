#!/usr/bin/env python3

import copy
import json

import pytest

import build_vllm_moe_fp8_selected_table_evaluation_preregistration_v29d as v29d
import build_vllm_moe_fp8_selected_table_evaluation_retry_preregistration_v29e as prereg


def test_v29e_retry_binds_exact_failure_evidence_and_v29d_commit():
    value = prereg.build_preregistration_v29e()
    retry = value["retry_of"]
    assert retry["v29d_preregistration_commit"] == (
        "23d6bd0f7d90a7438488e55eb796928b9a0bfd31"
    )
    assert retry["v29d_failure_evidence_file_sha256"] == (
        "9320d617a93527f6005a91156cbef58c174619df496f44a1dc552c8673ac34e0"
    )
    assert retry["v29d_failure_evidence_commit"] == (
        "756b026d6694ff5a379e3212e7a592ccf5c9981b"
    )
    assert retry["v29d_failure_was_activity_observability_only"] is True
    assert retry["v29d_report_absent"] is True
    assert retry["v29d_final_idle_certificate_present"] is True


def test_v29e_changes_only_observability_and_preserves_all_semantic_gates():
    base = v29d.build_preregistration_v29d()
    value = prereg.build_preregistration_v29e()
    assert value["selection_evidence"] == base["selection_evidence"]
    assert value["selected_table"] == base["selected_table"]
    assert value["statistical_contract"] == base["statistical_contract"]
    assert value["authority"] == base["authority"]
    assert value["schedule"]["fixed_seeds"] == base["schedule"]["fixed_seeds"]
    assert value["schedule"]["paired_counterbalanced_schedule"] == (
        base["schedule"]["paired_counterbalanced_schedule"]
    )
    assert value["kernel_contract"]["output_equivalence"] == (
        base["kernel_contract"]["output_equivalence"]
    )


def test_v29e_freezes_1000_iterations_50ms_poll_and_common_start_witness():
    value = prereg.build_preregistration_v29e()
    correction = value["sole_infrastructure_correction"]
    assert value["kernel_contract"]["official_num_iters"] == 1000
    assert correction["nvml_poll_interval_seconds_after"] == 0.05
    witness = correction["common_start_activity_witness_recipe"]
    assert witness["minimum_cuda_activity_seconds"] == 0.75
    assert witness["tensor_shape"] == [4096, 4096]
    assert witness["synchronize_after_each_measured_iteration"] is True
    assert witness["witness_excluded_from_latency_and_peak_vram_measurement"] is True
    assert witness["reset_peak_memory_stats_immediately_before_official_benchmark"] is True
    assert correction["common_start_activity_witness_recipe_sha256"] == (
        prereg.canonical_sha256(witness)
    )


def test_v29e_1000_iteration_largest_batch_memory_audit_is_below_5_gib():
    audit = prereg.build_preregistration_v29e()[
        "sole_infrastructure_correction"
    ]["max_batch_memory_audit"]
    assert audit["batch_size"] == 2048
    assert audit["gating_output_bytes"] == 2_097_152_000
    assert audit["audited_persistent_tensor_upper_bound_bytes"] == 2_997_026_824
    assert audit["audited_persistent_tensor_upper_bound_gib"] < 5.0
    assert audit["passes_required_upper_bound"] is True


def test_v29e_mutating_authority_or_statistics_fails_retry_validation():
    base = v29d.build_preregistration_v29d()
    value = prereg.build_preregistration_v29e()
    changed = copy.deepcopy(value)
    changed["authority"]["direct_table_adoption_authorized"] = True
    changed = prereg._seal(changed)
    with pytest.raises(RuntimeError, match="non-observability"):
        prereg.validate_preregistration_v29e(changed, base)
    changed = copy.deepcopy(value)
    changed["statistical_contract"]["bootstrap_resamples"] = 10
    changed = prereg._seal(changed)
    with pytest.raises(RuntimeError, match="non-observability"):
        prereg.validate_preregistration_v29e(changed, base)
    changed = copy.deepcopy(value)
    changed["sole_infrastructure_correction"][
        "common_start_activity_witness_recipe"
    ]["synchronize_after_each_measured_iteration"] = False
    changed = prereg._seal(changed)
    with pytest.raises(RuntimeError, match="non-observability"):
        prereg.validate_preregistration_v29e(changed, base)


def test_v29e_preregistration_build_is_deterministic_and_dry(capsys):
    first = prereg.build_preregistration_v29e()
    second = prereg.main(["--dry-run"])
    output = json.loads(capsys.readouterr().out)
    assert first == second
    assert output["gpu_launched"] is False
    assert output["official_num_iters"] == 1000
