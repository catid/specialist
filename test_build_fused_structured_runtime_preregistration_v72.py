import copy
import json

import pytest

import build_fused_structured_runtime_preregistration_v72 as builder
import eggroll_es_fused_structured_runtime_v72 as fused


def test_builder_matches_machine_readable_preregistration_exactly():
    built = builder.build_preregistration_v72()
    observed = json.loads(builder.OUTPUT.read_text(encoding="utf-8"))
    assert observed == built
    assert built["schema"] == builder.SCHEMA
    assert built["content_sha256_before_self_field"] == (
        "d3144d7f22570951974b8eb366d10e5e4ab9a3d3cc149d44afcb4b2be5c2ee58"
    )
    compact = dict(built)
    compact.pop("content_sha256_before_self_field")
    assert fused.canonical_sha256_v72(compact) == (
        built["content_sha256_before_self_field"]
    )


def test_production_projection_binds_all_four_identical_topologies():
    built = builder.build_preregistration_v72()
    projection = built["production_projection"]
    manifest = projection["manifest"]
    fused.validate_runtime_projection_manifest_v72(
        manifest, require_production_shape=True
    )
    assert manifest["content_sha256_before_self_field"] == (
        "7ad7c2ec6f55d38915744a6287e1d0bd56b4393f319053c62f3f4c9e36c9dcf5"
    )
    assert projection["source_tensor_count"] == 70
    assert projection["source_elements"] == 4_528_128
    assert projection["runtime_view_count"] == 82
    assert projection["runtime_elements"] == 4_921_344
    assert projection["runtime_bf16_bytes"] == 9_842_688
    source = built["implementation_bindings"]["production_topology"]
    assert source["four_physical_gpu_installations"] is True
    assert source["assignments_identical_across_gpus"] is True
    assert source["source_or_runtime_tensor_payload_opened"] is False


def test_production_byte_ledgers_seal_scratch_and_traffic():
    ledgers = builder.build_preregistration_v72()["production_byte_ledgers"]
    expected = {
        "iid_absolute_index": (4_528_128, 229_376, 196_608, 0),
        "structured_rank_1": (143_744, 262_272, 229_504, 32_896),
        "structured_rank_4": (574_976, 360_960, 328_192, 131_584),
        "structured_rank_8": (1_149_952, 492_544, 459_776, 263_168),
        "structured_rank_16": (2_299_904, 755_712, 722_944, 526_336),
    }
    assert set(ledgers) == set(expected)
    for key, values in expected.items():
        ledger = ledgers[key]
        assert (
            ledger["unique_random_values_per_direction"],
            ledger["candidate_scratch_ceiling_bytes"],
            ledger["weighted_update_scratch_ceiling_bytes"],
            ledger["maximum_factor_cache_bytes"],
        ) == values
        assert ledger["source_fp32_master_bytes"] == 18_112_512
        assert ledger["runtime_bf16_bytes"] == 9_842_688
        assert ledger["candidate_direct_runtime_write_bytes"] == 9_842_688
        assert ledger["candidate_post_generation_exact_readback_bytes"] == 9_842_688
        assert ledger["restore_direct_runtime_write_bytes"] == 9_842_688
        assert ledger["restore_exact_readback_bytes"] == 9_842_688
        assert ledger["eliminated_candidate_fp32_device_to_host_bytes"] == 18_112_512
        assert ledger["eliminated_pre_generation_runtime_equality_readback_bytes"] == (
            9_842_688
        )
        assert ledger["per_16_candidate_direct_runtime_write_bytes"] == 157_483_008
        assert ledger["per_16_candidate_restore_runtime_write_bytes"] == 157_483_008
        assert ledger["whole_surface_noise_elements_allocated"] == 0
        assert ledger["whole_surface_candidate_elements_allocated"] == 0
        assert ledger["whole_surface_update_elements_allocated"] == 0


def test_v71_restore_poison_and_commit_boundaries_are_retained():
    built = builder.build_preregistration_v72()
    candidate = built["direct_runtime_candidate_lifecycle"]
    update = built["streamed_update_lifecycle"]
    assert candidate["unknown_or_partial_rpc"] == "exact full restore or terminal poison"
    assert "one exact BF16 readback" in candidate["post_generation"]
    assert update["precondition"] == "exact V71 update_acceptance_sha256"
    assert update["commit"] == "provisional until exact V71 commit boundary"
    assert update["final"] == "rollback retained until exact V71 final boundary"
    assert update["final_update_maximum_ulp_vs_cpu_oracle"] == 2


def test_authority_and_promotion_fail_closed():
    built = builder.build_preregistration_v72()
    assert built["authority"] == {
        "cpu_oracle_and_fault_tests": True,
        "gpu_launch": False,
        "dataset_or_protected_content_access": False,
        "training_or_scored_evaluation": False,
        "candidate_or_update_promotion": False,
    }
    assert len(built["promotion_blockers"]) == 6
    assert built["required_live_experiment"]["physical_gpus"] == [0, 1, 2, 3]
    assert built["required_live_experiment"]["paired_replicates_minimum"] == 3
    assert built["required_live_experiment"][
        "promotion_authorized_by_preregistration"
    ] is False
    with pytest.raises(RuntimeError, match="launch forbidden"):
        builder.validate_preregistration_v72(built, launch=True)


def test_tampered_preregistration_is_rejected():
    built = builder.build_preregistration_v72()
    changed = copy.deepcopy(built)
    changed["production_projection"]["runtime_view_count"] = 81
    with pytest.raises(RuntimeError, match="preregistration changed"):
        builder.validate_preregistration_v72(changed)
