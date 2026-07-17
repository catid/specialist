import json

import pytest

pytest.skip(
    "historical V1-bound FP32 preregistration suite is nonpromotable",
    allow_module_level=True,
)
import torch

import build_fp32_es_optimizer_sigma_preregistration_v1 as builder
import fp32_es_optimizer_ablation_v1 as contract


@pytest.fixture(scope="module")
def built():
    return builder.build_preregistration_v1()


def test_builder_is_deterministic_and_valid(built):
    assert builder.build_preregistration_v1() == built
    result = contract.validate_preregistration_v1(built)
    assert result["arm_count"] == 8


def test_builder_does_not_authorize_gpu_live_or_protected_access(built):
    assert built["authorization"] == {
        "cpu_preview": True,
        "gpu_launch": False,
        "train_semantics": True,
        "dev_after_training": True,
        "ood_after_training": True,
        "protected_holdout": False,
        "live_run_access": False,
    }
    assert built["evaluation_gates"]["protected_paths_or_semantics_persisted"] is False


def test_builder_seals_actual_fp32_surface_and_module_scales(built):
    surface = built["parameter_surface"]
    assert surface["tensor_count"] == 70
    assert surface["logical_module_count"] == 35
    assert surface["elements"] == 4_528_128
    assert surface["bytes"] == 18_112_512
    assert surface["tensor_inventory_sha256"] == (
        "eea2d60e19530ba99e9ac4bc50f2806b20aa13ed30e159bad63a0144d0cb81b6"
    )
    assert built["sigma_contract"]["max_module_expected_energy_share"] < 0.03


def test_checked_in_artifact_is_exact_builder_output(built):
    checked = json.loads(builder.OUTPUT.read_text(encoding="utf-8"))
    assert checked == built
    assert checked["content_sha256_before_self_field"] == (
        "e8c646b5929de49805421035bb56f2eca2ed2010f7d1fce6893f5b095303dbc9"
    )
    assert builder.file_sha256_v1(builder.OUTPUT) == (
        "428d1de245a5cd5ad3cb976aa5312f6eda0874efb895d298bb5731a05f326924"
    )


def test_sealed_production_surface_can_hit_fp32_update_norm_budget():
    master = builder._load_master_v1()
    direction = {key: torch.ones_like(value) for key, value in master.items()}
    result = contract.apply_update_norm_budget_v1(master, direction)
    assert result["relative_error"] <= contract.UPDATE_NORM_RELATIVE_TOLERANCE_V1
