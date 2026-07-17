from __future__ import annotations

import ast
import json
from pathlib import Path

import build_reward_shaping_ablation_preregistration_v1 as builder
import reward_shaping_ablation_v1 as shaping


def test_preregistration_binds_contracts_compute_and_three_unique_gradients():
    result = builder.build_preregistration_v1()
    compact = dict(result)
    observed = compact.pop("content_sha256_before_self_field")
    assert shaping.canonical_sha256_v1(compact) == observed
    assert result["parents"]["evaluation_contract"][
        "protected_access_authorized"
    ] is False
    assert result["compute_match"]["optimization_rollouts_per_method_per_seed"] == 2048
    assert result["compute_match"]["three_unique_parameter_updates_not_four"] is True
    assert sum(method["unique_gradient_arm"] for method in result["methods"]) == 3
    assert result["registered_training_seeds"]["confirmation"] == [1701, 1702, 1703]
    assert result["selection_rule"]["protected_holdout_visible"] is False
    assert result["selection_rule"]["protected_result_can_change_method_or_recipe"] is False


def test_builder_reads_only_sealed_metadata_and_implementation_files(monkeypatch):
    opened = []
    original_open = Path.open

    def recording_open(path, *args, **kwargs):
        opened.append(Path(path).resolve())
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)
    result = builder.build_preregistration_v1()
    allowed = {
        builder.EVALUATION_CONTRACT,
        builder.SAMPLING_CONTRACT,
        Path(builder.__file__).resolve(),
        Path(shaping.__file__).resolve(),
        builder.MIRRORED_V66,
        builder.CENTERED_RANK_V43G,
        builder.LEGACY_ZSCORE,
    }
    assert set(opened) <= allowed
    role_paths = {
        Path(item["path"]).resolve()
        for item in json.loads(
            builder.EVALUATION_CONTRACT.read_text(encoding="utf-8")
        )["roles"].values()
        if isinstance(item, dict) and "path" in item
    }
    assert not set(opened) & role_paths
    assert result["access_receipt"]["protected_holdout_semantics_opened"] is False
    assert result["access_receipt"]["train_semantics_opened"] is False


def test_synthetic_diagnostic_is_not_used_as_model_evidence():
    diagnostic = builder.build_preregistration_v1()["cpu_synthetic_diagnostic"]
    assert diagnostic["model_reward_improvement_claimed"] is False
    assert diagnostic["method_selected_from_synthetic_evidence"] is False
    assert diagnostic["raw_and_direct_pair_coefficients_exactly_equal"] is True
    assert diagnostic["every_noncontaminated_prompt_group_bitwise_unchanged"] is True
    raw = diagnostic["single_extreme_outlier"]["raw_rewards"]
    rank = diagnostic["single_extreme_outlier"]["within_prompt_centered_rank"]
    zscore = diagnostic["single_extreme_outlier"][
        "within_prompt_centered_zscore"
    ]
    assert raw["coefficient_l2_delta"] > 1e11
    assert rank["coefficient_l2_delta"] < 1.0
    assert zscore["coefficient_l2_delta"] < 2.0


def test_cpu_transform_has_no_model_dataset_or_gpu_imports():
    tree = ast.parse(Path(shaping.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(item.name.split(".")[0] for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"torch", "ray", "vllm", "numpy", "datasets"}


def test_main_writes_reopenable_canonical_artifact(tmp_path, capsys):
    output = tmp_path / "reward-shaping.json"
    assert builder.main(["--output", str(output)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["path"] == str(output.resolve())
    assert receipt["file_sha256"] == builder.file_sha256_v1(output)
    value = json.loads(output.read_text(encoding="utf-8"))
    compact = dict(value)
    content_sha = compact.pop("content_sha256_before_self_field")
    assert content_sha == receipt["content_sha256"]
    assert shaping.canonical_sha256_v1(compact) == content_sha


def test_checked_in_preregistration_is_exact_builder_output():
    checked_in = json.loads(builder.OUTPUT.read_text(encoding="utf-8"))
    assert checked_in == builder.build_preregistration_v1()
    assert builder.file_sha256_v1(builder.OUTPUT) == (
        "06a24c50c44f684534d1e189dd145c8705c5f3867053982590f7f262d86a2615"
    )
