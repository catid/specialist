import json

import build_qwen36_lora_fp32_collective_coalescing_preregistration_v83a as builder
import eggroll_es_fp32_collective_coalescing_v83a as contract


def test_preregistration_and_evidence_are_current_and_self_hashed():
    value = builder.build_preregistration_v83a()
    stored = json.loads(builder.OUTPUT.read_text(encoding="ascii"))
    assert builder.validate_preregistration_v83a(stored) == value
    assert builder.OUTPUT.read_text(encoding="ascii") \
        == builder.render_json_v83a(value)
    assert builder.REPORT.read_text(encoding="ascii") \
        == builder.render_report_v83a(value)
    body = dict(value)
    claimed = body.pop("content_sha256_before_self_field")
    assert contract.canonical_sha256_v83a(body) == claimed


def test_preregistration_binds_exact_surface_choices_and_actual_interface():
    value = builder.build_preregistration_v83a()
    surface = value["canonical_lora_surface"]
    assert surface["tensor_count"] == 70
    assert surface["elements"] == 4_528_128
    assert surface["bytes"] == 18_112_512
    assert len(surface["ordered_records"]) == 70
    assert surface["ordered_shape_manifest_sha256"] \
        == contract.EXPECTED_SOURCE_MANIFEST_SHA256_V83A
    plans = value["deterministic_bucket_plans"]
    assert [
        plans[name]["coalesced_collective_calls"]
        for name, _capacity in contract.BUCKET_CHOICES_V83A
    ] == [1, 3, 5, 10]
    runtime = value["runtime_semantics"]
    assert runtime["actual_future_communicator"] == "self.inter_pg"
    assert runtime["actual_future_method"] == "all_reduce"
    assert runtime["exact_future_call_expression"] == (
        "self.inter_pg.all_reduce(bucket, out_tensor=bucket, stream=stream)"
    )
    assert runtime["event_synchronized_before_host_candidate_consumption"] \
        is True
    assert runtime["failure_preserves_exact_original_or_terminally_poisons"] \
        is True


def test_live_launch_selection_speed_quality_and_promotion_stay_blocked():
    value = builder.build_preregistration_v83a()
    authority = value["authority"]
    assert authority == {
        "source_and_synthetic_cpu_only": True,
        "dataset_training_examples_or_site_corpus_opened": False,
        "evaluation_dev_ood_holdout_shadow_or_probe_opened": False,
        "model_ray_or_gpu_launched": False,
        "live_pynccl_executed": False,
        "adapter_update_or_checkpoint_written": False,
        "bucket_choice_selected": False,
        "live_arm_authorized": False,
        "training_hpo_quality_or_promotion_authorized": False,
    }
    assert value["selection"]["selected_choice"] is None
    assert value["future_gate"]["v73d_receipt_read_or_bound_by_v83a"] is False
    assert value["future_gate"]["gpu_launch_authorized_now"] is False
    assert value["future_gate"]["speed_vram_or_bandwidth_improvement_claimed_now"] \
        is False
    assert value["future_gate"]["bead_remains_open"] is True


def test_memory_formulas_separate_staging_from_hbm_and_network_payload():
    value = builder.build_preregistration_v83a()
    memory = value["memory_and_bandwidth"]
    assert memory["network_payload_bytes_change"] == 0
    assert memory["nominal_ring_bytes_change"] == 0
    assert memory["flat_sequential_staging_bytes"] == 18_112_512
    assert memory["native_maximum_accumulator_bytes"] == 1_048_576
    assert memory["flat_incremental_staging_bytes_versus_native_maximum"] \
        == 17_063_936
    assert memory["materialized_pack_read_plus_write_hbm_bytes"] == 36_225_024
    assert memory["materialized_gpu_unpack_read_plus_write_hbm_bytes"] \
        == 36_225_024
    assert memory["conservative_pack_plus_gpu_unpack_hbm_bytes"] == 72_450_048
    assert memory["unchanged_d2h_source_hbm_read_bytes"] == 18_112_512
    assert memory["direct_fill_hbm_saving_measured"] is False
    assert memory["nccl_internal_hbm_measured"] is False
