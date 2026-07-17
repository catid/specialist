import copy

import build_qwen36_fp8_kv_cleanup_preregistration_v79b as subject


def test_v79b_is_additive_model_invariant_and_self_hashed():
    value = subject.build_v79b()
    assert value["schema"] == subject.SCHEMA
    assert value["implementation_correction"]["model_or_workload_change"] is False
    assert value["parent_v79"]["selected_runtime_retained_exactly"][
        "gpu_memory_utilization"
    ] == 0.485
    assert value["cleanup_acceptance"]["minimum_consecutive_batches"] == 3
    assert value["cleanup_acceptance"]["memory_used_mib_max"] == 4
    assert value["authority"][
        "scored_training_checkpoint_or_promotion_authorized"
    ] is False
    body = copy.deepcopy(value)
    claimed = body.pop("content_sha256_before_self_field")
    assert subject.canonical_sha256(body) == claimed


def test_v79b_binds_current_sources_and_exact_fresh_command():
    value = subject.build_v79b()
    assert subject.source_inventory() == value["sealed_sources"]["files"]
    assert "v79b_fp8_kv_capacity_0485_r5_sealed_cleanup" in value["launch"][
        "exact_command"
    ]
    assert value["launch"]["launch_performed_by_builder"] is False
