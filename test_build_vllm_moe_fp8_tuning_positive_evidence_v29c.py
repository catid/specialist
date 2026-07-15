import copy
import json

import pytest

import build_vllm_moe_fp8_tuning_positive_evidence_v29c as evidence


def _artifacts():
    return (
        json.loads(evidence.ATTEMPT_PATH_V29C.read_text(encoding="utf-8")),
        json.loads(evidence.REPORT_PATH_V29C.read_text(encoding="utf-8")),
        json.loads(evidence.PREREG_PATH_V29C.read_text(encoding="utf-8")),
        json.loads(evidence.LIVE_TABLE_PATH_V29C.read_text(encoding="utf-8")),
    )


def test_v29c_evidence_rebuilds_exactly_and_stops_before_evaluation():
    value = evidence.build_positive_evidence_v29c()
    persisted = json.loads(evidence.OUTPUT_PATH_V29C.read_text(encoding="utf-8"))
    assert value == persisted
    assert value["status"] == "valid_completed_selection_not_evaluation"
    result = value["aggregate_result"]
    assert result["official_worker_count"] == 4
    assert result["configurations_per_worker"] == 1920
    assert result["total_configurations"] == 7680
    assert result["simultaneous_all_four_positive_observation_count"] == 4842
    assert result["all_four_gpus_finally_idle"] is True
    assert result["official_source_unmodified"] is True
    assert result["only_official_OutOfResources_skip_active"] is True
    assert result["bf16_v27c_table_loaded_or_reused"] is False
    assert set(result["per_gpu"]) == set(evidence.EXPECTED_UUIDS_V29C)
    for gpu_id, item in result["per_gpu"].items():
        assert item["nvml_uuid"] == evidence.EXPECTED_UUIDS_V29C[gpu_id]
        assert item["assigned_official_actor_pid_observed"] is True
        assert item["maximum_gpu_utilization_percent"] == 100
        assert item["sample_count"] == 10321
    decision = value["decision"]
    assert decision[
        "authorize_only_separate_fp8_table_evaluation_preregistration"
    ] is True
    assert decision["evaluation_authorized_by_this_evidence"] is False
    for key in (
        "selected_table_direct_adoption_authorized",
        "training_or_model_update_authorized",
        "checkpoint_write_authorized",
        "dataset_promotion_authorized",
        "validation_heldout_ood_or_benchmark_open_authorized",
    ):
        assert decision[key] is False


def test_v29c_live_attempt_report_prereg_and_original_table_are_exactly_bound():
    attempt, report, preregistration, configs = (
        evidence.validate_bound_artifacts_v29c()
    )
    assert attempt["report_binding"]["file_sha256"] == (
        evidence.REPORT_FILE_SHA256_V29C
    )
    assert report["content_sha256_before_self_field"] == (
        evidence.REPORT_CONTENT_SHA256_V29C
    )
    assert preregistration["content_sha256_before_self_field"] == (
        evidence.PREREG_CONTENT_SHA256_V29C
    )
    assert configs == evidence.EXPECTED_CONFIGS_V29C
    artifact = evidence.build_positive_evidence_v29c()["artifacts"][
        "original_official_selected_table"
    ]
    assert artifact["relative_path"] == evidence.LIVE_TABLE_RELATIVE_PATH_V29C
    assert artifact[
        "must_be_force_added_at_exact_original_path_in_v29c_commit"
    ] is True
    assert artifact["file_sha256"] == evidence.SELECTED_TABLE_FILE_SHA256_V29C


def test_v29c_report_self_hash_reconstructs_integer_batch_keys_exactly():
    _attempt, report, _preregistration, _table = _artifacts()
    evidence._verify_report_self_hash_v29c(report)
    changed = copy.deepcopy(report)
    changed["runtime_integrity"]["selected_configs"]["256"][
        "num_stages"
    ] += 1
    with pytest.raises(RuntimeError, match="report self hash changed"):
        evidence._verify_report_self_hash_v29c(changed)
    missing = copy.deepcopy(report)
    del missing["runtime_integrity"]["selected_configs"]["256"]
    with pytest.raises(RuntimeError, match="integer batch-key reconstruction"):
        evidence._verify_report_self_hash_v29c(missing)


@pytest.mark.parametrize(
    ("target", "mutate", "message"),
    (
        (
            "report",
            lambda value: value.__setitem__("implementation_bundle_sha256", "0" * 64),
            "implementation or recipe binding changed",
        ),
        (
            "report",
            lambda value: value.__setitem__("recipe_content_sha256", "0" * 64),
            "implementation or recipe binding changed",
        ),
        (
            "attempt",
            lambda value: value["report_binding"].__setitem__(
                "content_sha256", "0" * 64
            ),
            "attempt report binding changed",
        ),
        (
            "report",
            lambda value: value["runtime_integrity"][
                "inflight_physical_gpu_utilization"
            ]["per_gpu"]["0"].__setitem__("sample_count", 10320),
            "GPU 0 utilization certificate changed",
        ),
        (
            "report",
            lambda value: value["runtime_integrity"][
                "inflight_physical_gpu_utilization"
            ].__setitem__(
                "simultaneous_all_four_assigned_actor_pids_and_positive_utilization_observation_count",
                4841,
            ),
            "aggregate official tuning integrity changed",
        ),
        (
            "report",
            lambda value: value["runtime_integrity"].__setitem__(
                "official_source_unmodified", False
            ),
            "aggregate official tuning integrity changed",
        ),
        (
            "report",
            lambda value: value["runtime_integrity"].__setitem__(
                "only_official_OutOfResources_skip_active", False
            ),
            "aggregate official tuning integrity changed",
        ),
        (
            "report",
            lambda value: value["runtime_integrity"].__setitem__(
                "bf16_v27c_table_loaded_or_reused", True
            ),
            "aggregate official tuning integrity changed",
        ),
        (
            "report",
            lambda value: value["runtime_integrity"].__setitem__(
                "all_four_gpus_idle_after_cleanup", False
            ),
            "aggregate official tuning integrity changed",
        ),
        (
            "report",
            lambda value: value.__setitem__(
                "direct_adoption_training_model_update_checkpoint_evaluation_or_dataset_promotion_authorized",
                True,
            ),
            "closed authority or decision changed",
        ),
        (
            "attempt",
            lambda value: value.__setitem__(
                "dataset_train_evaluation_validation_heldout_ood_or_benchmark_surface_opened",
                True,
            ),
            "closed authority or decision changed",
        ),
        (
            "table",
            lambda value: value["256"].__setitem__("num_warps", 8),
            "selected table content changed",
        ),
    ),
)
def test_v29c_rejects_semantic_mutations(target, mutate, message):
    attempt, report, preregistration, table = _artifacts()
    values = {
        "attempt": attempt,
        "report": report,
        "preregistration": preregistration,
        "table": table,
    }
    mutate(values[target])
    with pytest.raises(RuntimeError, match=message):
        evidence._validate_semantics_v29c(
            attempt, report, preregistration, table,
        )


@pytest.mark.parametrize(
    ("path_name", "message"),
    (
        ("ATTEMPT_PATH_V29C", "file hash changed"),
        ("REPORT_PATH_V29C", "file hash changed"),
        ("PREREG_PATH_V29C", "file hash changed"),
        ("LIVE_TABLE_PATH_V29C", "file hash changed"),
    ),
)
def test_v29c_rejects_changed_live_artifact_bytes(
    monkeypatch, tmp_path, path_name, message,
):
    source = getattr(evidence, path_name)
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b"\n")
    monkeypatch.setattr(evidence, path_name, changed)
    with pytest.raises(RuntimeError, match=message):
        evidence.validate_bound_artifacts_v29c()


def test_v29c_compact_evidence_rejects_raw_or_data_payload_keys():
    for forbidden in (
        "question", "response", "progress_log", "compiler_log",
        "search_results", "timing_vectors", "raw_pids",
    ):
        with pytest.raises(RuntimeError, match="forbidden keys"):
            evidence._assert_compact_v29c({forbidden: "not allowed"})
