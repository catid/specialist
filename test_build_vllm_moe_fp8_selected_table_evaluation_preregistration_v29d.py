#!/usr/bin/env python3

import copy
import json

import pytest

import build_vllm_moe_fp8_selected_table_evaluation_preregistration_v29d as prereg


def test_v29d_binds_exact_v29c_commit_evidence_and_original_table():
    value = prereg.build_preregistration_v29d()
    assert value["selection_evidence"] == {
        "commit": "a203f4821c4a737310df75543353d21ce6cea978",
        "path": str(prereg.EVIDENCE_PATH_V29D),
        "file_sha256": "47d1b09fb188dd1f8ff16314f1c20fe614f02b1cff067a1615a0d6f0f5ce2a7b",
        "content_sha256": "dc4d3b6d2b090e4e740f63de573875f331a456d6951b62cf49a003b1114ee02e",
        "authorizes_only_this_separate_evaluation_preregistration": True,
    }
    assert value["selected_table"]["file_sha256"] == (
        "1a4ed0f44c6d7cc788baecd073107b4634db4c769f0820d10174f61117b25618"
    )
    assert value["selected_table"]["content_sha256"] == (
        "d4a49735ccfd094d6e5a3ee763eca99ed355a51fd11a7c835bfecf9fafeaa50d"
    )
    assert value["selected_table"]["exact_configs"] == prereg.EXPECTED_CONFIGS_V29D


def test_v29d_schedule_is_exact_paired_deterministic_and_counterbalanced():
    value = prereg.build_preregistration_v29d()
    schedule = value["schedule"]
    rows = schedule["paired_counterbalanced_schedule"]
    assert schedule["repetitions"] == 8
    assert [row["seed"] for row in rows] == list(prereg.SEEDS_V29D)
    assert sum(row["arm_order"] == ["default", "tuned"] for row in rows) == 4
    assert sum(row["arm_order"] == ["tuned", "default"] for row in rows) == 4
    assert schedule["fresh_four_worker_ray_wave_per_arm"] is True
    assert value["kernel_contract"]["same_seed_and_official_tensor_constructor_for_both_paired_arms"] is True
    assert value["kernel_contract"]["output_equivalence"] == (
        "exact_dtype_shape_and_byte_sha256"
    )


def test_v29d_statistics_freeze_bootstrap_multiplicity_and_all_zero_margin_gates():
    stats = prereg.build_preregistration_v29d()["statistical_contract"]
    assert stats["bootstrap_seed"] == 20_261_005
    assert stats["bootstrap_resamples"] == 50_000
    assert stats["familywise_alpha"] == 0.05
    assert stats["per_endpoint_one_sided_alpha"] == 0.005
    assert len(stats["endpoints"]) == 10
    assert stats["latency_noninferiority_margin"] == 0.0
    assert stats["peak_vram_regression_margin"] == 0.0
    assert stats["all_exact_outputs_must_match"] is True
    assert stats["all_gates_are_conjunctive"] is True


def test_v29d_binds_exact_fp8_geometry_software_model_and_hardware():
    value = prereg.build_preregistration_v29d()
    kernel = value["kernel_contract"]
    assert (kernel["experts"], kernel["official_shard_intermediate_size"]) == (256, 1024)
    assert (kernel["hidden_size"], kernel["topk"]) == (2048, 8)
    assert kernel["dtype"] == "fp8_w8a8"
    assert kernel["block_shape"] == [128, 128]
    assert kernel["batches"] == [256, 512, 1024, 2048]
    assert value["software_identity"]["versions"] == {
        "vllm": "0.25.0", "torch": "2.11.0+cu130",
        "triton": "3.6.0", "ray": "2.56.0",
    }
    assert value["model_identity"]["config_sha256"] == (
        "570ef7ea45a7e1d3de2b1d3c70c4ac3562d0e768acdc195778cb4f4d95025845"
    )
    assert len(value["hardware_contract"]["identities"]) == 4


def test_v29d_authority_is_closed_and_mutation_fails_validation():
    value = prereg.build_preregistration_v29d()
    authority = value["authority"]
    assert authority["direct_table_adoption_authorized"] is False
    assert authority["model_update_training_checkpoint_write_dataset_promotion_authorized"] is False
    assert authority["dataset_evaluation_validation_heldout_ood_or_benchmark_access_authorized"] is False
    changed = copy.deepcopy(value)
    changed["authority"]["direct_table_adoption_authorized"] = True
    changed = prereg._seal(changed)
    with pytest.raises(RuntimeError, match="schedule statistics or authority"):
        prereg.validate_preregistration_v29d(changed)


def test_v29d_changed_evidence_bytes_fail_closed(monkeypatch, tmp_path):
    changed = tmp_path / prereg.EVIDENCE_PATH_V29D.name
    changed.write_bytes(prereg.EVIDENCE_PATH_V29D.read_bytes() + b"\n")
    monkeypatch.setattr(prereg, "EVIDENCE_PATH_V29D", changed)
    with pytest.raises(RuntimeError, match="static input identity"):
        prereg.validate_static_inputs_v29d()


def test_v29d_dry_run_does_not_write_or_launch_gpu(capsys):
    value = prereg.main(["--dry-run"])
    output = json.loads(capsys.readouterr().out)
    assert output["content_sha256"] == value["content_sha256_before_self_field"]
    assert output["gpu_launched"] is False
