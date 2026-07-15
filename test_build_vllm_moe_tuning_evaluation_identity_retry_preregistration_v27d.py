#!/usr/bin/env python3

import json

import build_vllm_moe_tuning_evaluation_identity_retry_preregistration_v27d as v27d


def test_v27d_is_deterministic_and_self_sealed():
    frozen = json.loads(v27d.OUTPUT_PATH.read_text())
    built = v27d.build_preregistration()
    assert built == frozen
    assert frozen["content_sha256_before_self_field"] == v27d.canonical_sha256({
        key: value for key, value in frozen.items()
        if key != "content_sha256_before_self_field"
    })


def test_v27d_is_only_a_strict_gpu_id_representation_retry():
    value = v27d.build_preregistration()
    probe = value["identity_probe"]
    overlay = value["retry_overlay"]
    assert probe["raw_ray_gpu_ids_in_physical_order"] == [
        ["0"], ["1"], ["2"], ["3"],
    ]
    assert probe["cuda_visible_devices_in_physical_order"] == ["0", "1", "2", "3"]
    assert overlay["accepted_canonical_ids"] == [0, 1, 2, 3]
    assert overlay["booleans_floats_padded_strings_uuids_and_other_values_rejected"] is True
    assert overlay["benchmark_configs_iterations_seeds_arm_order_and_gate_unchanged"] is True


def test_v27d_binds_zero_measurement_failure_and_keeps_authority_closed():
    value = v27d.build_preregistration()
    failed = value["failed_evaluation_attempt"]
    assert failed["kernel_timing_count"] == 0
    assert failed["report_written"] is False
    assert value["evaluation"]["retry_not_launched_by_this_artifact"] is True
    assert value["authority"][
        "direct_recipe_adoption_model_update_checkpoint_dataset_promotion"
    ] is False
    assert value["authority"]["validation_heldout_ood_or_benchmark_open"] is False
