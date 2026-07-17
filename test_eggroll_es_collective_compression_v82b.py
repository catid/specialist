import ast
import copy
import json

import pytest

import build_qwen36_lora_collective_compression_preregistration_v82b as builder
import eggroll_es_collective_compression_v82b as oracle


def source_records():
    value = json.loads(builder.MANIFEST.read_text(encoding="utf-8"))
    return value["surface"]["records"]


def materiality_profile(fractions=(0.02, 0.03, 0.04), ranks=(4, 5, 6)):
    runs = []
    for seed, fraction, rank in zip(
        oracle.PROFILE_SEEDS_V82B, fractions, ranks, strict=True
    ):
        runs.append(
            {
                "seed": seed,
                "world_size": 4,
                "actors": 4,
                "collective_calls_per_actor": 70,
                "elements_per_actor": 4_528_128,
                "update_execute_seconds": 4.0,
                "unoverlapped_collective_seconds": 4.0 * fraction,
                "collective_bottleneck_rank": rank,
                "link_bytes_measured": True,
                "hbm_bytes_measured": True,
                "all_four_gpus_attributed": True,
                "cleanup_idle": True,
            }
        )
    body = {
        "schema": oracle.PROFILE_SCHEMA_V82B,
        "scope_content_sha256": "a" * 64,
        "worker_v72_file_sha256": builder.WORKER_V72_SHA256,
        "runs": runs,
        "authority": {
            "dataset_or_training_examples_opened": False,
            "protected_dev_ood_or_holdout_opened": False,
            "adapter_update_committed": False,
            "compression_arm_executed": False,
            "promotion_authorized": False,
        },
    }
    return {**body, "content_sha256": oracle.canonical_sha256_v82b(body)}


def test_exact_sealed_manifest_is_70_tensor_lora_surface():
    records = oracle.validate_ordered_shape_manifest_v82b(source_records())
    assert len(records) == 70
    assert len({row["module"] for row in records}) == 35
    assert sum(row["elements"] for row in records) == 4_528_128
    assert max(row["elements"] for row in records) == 262_144
    assert [row["key"] for row in records] == sorted(row["key"] for row in records)
    assert [row["ordinal"] for row in records] == list(range(70))


@pytest.mark.parametrize("mutation", ["reorder", "shape", "dtype", "drop"])
def test_manifest_mutations_fail_closed(mutation):
    records = copy.deepcopy(source_records())
    if mutation == "reorder":
        records[0], records[1] = records[1], records[0]
    elif mutation == "shape":
        records[0]["shape"][0] += 1
    elif mutation == "dtype":
        records[0]["dtype"] = "torch.bfloat16"
    else:
        records.pop()
    with pytest.raises((RuntimeError, ValueError)):
        oracle.validate_ordered_shape_manifest_v82b(records)


def test_corrected_ring_residual_staging_and_hbm_formulas_are_exact():
    value = oracle.collective_byte_accounting_v82b(source_records())
    fp32 = value["fp32_control"]
    bf16 = value["bf16_error_feedback_hypothetical"]
    ring = value["nominal_projection"]
    assert value["scope"]["collective_calls_per_actor_per_update"] == 70
    assert value["scope"]["element_count_distribution"] == {
        "1024": 6,
        "8192": 4,
        "16384": 14,
        "65536": 35,
        "131072": 7,
        "262144": 4,
    }
    assert fp32["payload_bytes_per_actor_per_update"] == 4 * 4_528_128
    assert bf16["payload_bytes_per_actor_per_update"] == 2 * 4_528_128
    assert fp32["nominal_ring_bus_bytes_per_actor_per_update"] == 6 * 4_528_128
    assert bf16["nominal_ring_bus_bytes_per_actor_per_update"] == 3 * 4_528_128
    assert ring["ring_bus_bytes_saved_per_actor_per_update"] == 13_584_384
    assert ring["ring_bus_bytes_saved_all_actors_per_update"] == 54_337_536
    assert bf16["steady_residual_bank_bytes_per_actor"] == 18_112_512
    assert bf16["transaction_two_residual_banks_bytes_per_actor"] == 36_225_024
    assert bf16["maximum_bf16_gpu_staging_bytes"] == 524_288
    assert bf16["incremental_transaction_peak_gpu_bytes_per_actor"] == 36_749_312
    assert bf16["fused_prepare_hbm_bytes_per_actor_per_update_lower_bound"] == 63_393_792
    assert (
        bf16[
            "incremental_local_hbm_bytes_per_actor_per_update_lower_bound_"
            "versus_fp32_excluding_nccl"
        ]
        == 54_337_536
    )


def test_byte_accounting_is_self_hashed_and_marks_ring_as_projection():
    value = oracle.collective_byte_accounting_v82b(source_records())
    body = copy.deepcopy(value)
    claimed = body.pop("content_sha256")
    assert oracle.canonical_sha256_v82b(body) == claimed
    assert value["nominal_projection"][
        "algorithm_is_projection_not_measured_pynccl_behavior"
    ] is True
    assert value["nominal_projection"]["nccl_internal_hbm_bytes_excluded"] is True


def test_low_materiality_profile_does_not_register_compression():
    profile = materiality_profile()
    result = oracle.evaluate_materiality_profile_v82b(profile, "a" * 64)
    assert result["profile_valid"] is True
    assert result["materiality_thresholds_passed"] is False
    assert result["compressed_live_arm_implementation_authorized"] is False
    assert result["training_or_promotion_authorized"] is False


def test_only_prospectively_material_profile_can_authorize_implementation_not_training():
    profile = materiality_profile(
        fractions=(0.06, 0.07, 0.08), ranks=(1, 2, 4)
    )
    result = oracle.evaluate_materiality_profile_v82b(profile, "a" * 64)
    assert result["median_unoverlapped_collective_fraction_of_update"] == 0.07
    assert result["top_three_replicates"] == 2
    assert result["materiality_thresholds_passed"] is True
    assert result["compressed_live_arm_implementation_authorized"] is True
    assert result["training_or_promotion_authorized"] is False


@pytest.mark.parametrize(
    "mutation",
    ["hash", "scope", "compression_executed", "wrong_count", "missing_hbm"],
)
def test_materiality_profile_identity_authority_and_scope_fail_closed(mutation):
    profile = materiality_profile()
    if mutation == "hash":
        profile["content_sha256"] = "0" * 64
    elif mutation == "scope":
        profile["scope_content_sha256"] = "b" * 64
    elif mutation == "compression_executed":
        profile["authority"]["compression_arm_executed"] = True
    elif mutation == "wrong_count":
        profile["runs"][0]["collective_calls_per_actor"] = 23
    else:
        profile["runs"][0]["hbm_bytes_measured"] = False
    if mutation != "hash":
        body = {key: value for key, value in profile.items() if key != "content_sha256"}
        profile["content_sha256"] = oracle.canonical_sha256_v82b(body)
    with pytest.raises(RuntimeError):
        oracle.evaluate_materiality_profile_v82b(profile, "a" * 64)


def test_preregistration_binds_immutable_incident_and_supersedes_only_layout():
    value = builder.build_preregistration_v82b()
    incident = value["immutable_incident_lineage"]
    assert incident["v82"]["commit"] == builder.V82_COMMIT
    assert incident["v82"]["immutable_wrong_scope_evidence"] is True
    assert incident["v82"]["historical_result_rewritten"] is False
    assert incident["v82"]["wrong_surface"]["elements"] == 142_999_552
    assert incident["v82"]["additional_backend_mismatch"][
        "torch_dtype_capability_evidence_proves_canonical_communicator"
    ] is False
    assert incident["v75"]["immutable_artifact_mutated"] is False
    assert incident["v75"]["collective_layout_claim_superseded"] is True
    assert incident["v75"][
        "collective_layout_promotable_before_rebuilt_decision"
    ] is False
    assert value["v75_additive_correction"][
        "v75_noncollective_fields_not_rejudged_here"
    ] is True


def test_preregistration_embeds_full_ordered_shape_manifest_and_correct_scope():
    value = builder.build_preregistration_v82b()
    master = value["canonical_lora_update_scope"]["canonical_master"]
    assert len(master["ordered_shape_manifest"]) == 70
    assert master["elements"] == 4_528_128
    assert master["bytes"] == 18_112_512
    assert value["canonical_lora_update_scope"]["update_execution"][
        "collective_calls_per_actor_per_update"
    ] == 70
    assert value["legacy_to_corrected_comparison"][
        "legacy_over_corrected_element_ratio"
    ] == 142_999_552 / 4_528_128


def test_current_decision_is_no_live_arm_and_no_promotion():
    value = builder.build_preregistration_v82b()
    assert value["authority"]["live_compression_arm_registered"] is False
    assert value["materiality_reassessment"]["materiality_established"] is False
    assert value["materiality_reassessment"]["current_decision"] == (
        "retain_exact_fp32_control_no_live_compression_arm"
    )
    assert value["correctness_and_safety"]["promotion_authorized"] is False
    assert value["correctness_and_safety"][
        "BF16_support_through_canonical_PyNccl_communicator_proven"
    ] is False


def test_preregistration_and_reports_are_current_and_self_hashed():
    value = builder.build_preregistration_v82b()
    body = copy.deepcopy(value)
    claimed = body.pop("content_sha256_before_self_field")
    assert oracle.canonical_sha256_v82b(body) == claimed
    assert builder.OUTPUT.read_text(encoding="ascii") == builder.render_json_v82b(value)
    assert builder.REPORT.read_text(encoding="ascii") == builder.render_report_v82b(value)
    assert builder.INCIDENT.read_text(encoding="ascii") == builder.render_incident_v82b(value)
    assert "Do not launch or implement" in builder.REPORT.read_text(encoding="ascii")
    assert "specialist-nen.31" in builder.INCIDENT.read_text(encoding="ascii")


def test_cpu_contract_sources_do_not_import_torch_or_cuda():
    for path in (builder.ORACLE, builder.ROOT / builder.__file__.split("/")[-1]):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert "torch" not in imported
        assert "cuda" not in imported

