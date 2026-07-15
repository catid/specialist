import copy
import json

import pytest

import build_vllm_moe_fp8_full_model_runtime_ab_negative_evidence_v29i as evidence


def _artifacts():
    return (
        json.loads(evidence.ATTEMPT_PATH_V29I.read_text(encoding="utf-8")),
        json.loads(evidence.REPORT_PATH_V29I.read_text(encoding="utf-8")),
    )


def test_v29i_binds_exact_completed_compact_artifacts():
    value = evidence.build_negative_evidence_v29i()
    assert value["artifacts"] == {
        "durable_attempt": {
            "relative_path": evidence.ATTEMPT_RELATIVE_PATH_V29I,
            "file_sha256": evidence.ATTEMPT_FILE_SHA256_V29I,
            "content_sha256": evidence.ATTEMPT_CONTENT_SHA256_V29I,
        },
        "compact_report": {
            "relative_path": evidence.REPORT_RELATIVE_PATH_V29I,
            "file_sha256": evidence.REPORT_FILE_SHA256_V29I,
            "content_sha256": evidence.REPORT_CONTENT_SHA256_V29I,
        },
    }


def test_v29i_binds_exact_source_implementation_recipe_and_lifecycle():
    bindings = evidence.build_negative_evidence_v29i()["frozen_bindings"]
    assert bindings == {
        "committed_source_sha256": evidence.COMMITTED_SOURCE_SHA256_V29I,
        "implementation_bundle_sha256": (
            evidence.IMPLEMENTATION_BUNDLE_SHA256_V29I
        ),
        "recipe_content_sha256": evidence.RECIPE_CONTENT_SHA256_V29I,
        "runtime_environment_sha256": evidence.RUNTIME_ENVIRONMENT_SHA256_V29I,
        "cpu_disk_audit_sha256": evidence.CPU_DISK_AUDIT_SHA256_V29I,
        "prelaunch_idle_sha256": evidence.PRELAUNCH_IDLE_SHA256_V29I,
        "final_idle_sha256": evidence.FINAL_IDLE_SHA256_V29I,
        "authorization_gate_content_sha256": evidence.GATE_CONTENT_SHA256_V29I,
    }


def test_v29i_binds_preregistration_table_and_checkpoint_identities():
    identities = evidence.build_negative_evidence_v29i()[
        "transitively_frozen_contract_identities"
    ]
    assert identities["binding_basis"] == {
        "committed_source_sha256": evidence.COMMITTED_SOURCE_SHA256_V29I,
        "implementation_bundle_sha256": evidence.IMPLEMENTATION_BUNDLE_SHA256_V29I,
        "recipe_content_sha256": evidence.RECIPE_CONTENT_SHA256_V29I,
        "cpu_disk_audit_sha256": evidence.CPU_DISK_AUDIT_SHA256_V29I,
    }
    assert identities["v29h_preregistration"] == (
        evidence.PREREGISTRATION_IDENTITY_V29I
    )
    assert identities["selected_table"] == evidence.SELECTED_TABLE_IDENTITY_V29I
    assert identities["serialized_fp8_checkpoint"] == (
        evidence.SERIALIZED_FP8_CHECKPOINT_IDENTITY_V29I
    )
    assert identities["serialized_fp8_checkpoint"]["weight_shards"][
        "file_count"
    ] == 42
    assert identities["serialized_fp8_checkpoint"]["all_files"][
        "file_count"
    ] == 56


def test_v29i_all_eight_pairs_and_every_component_are_exact():
    execution = evidence.build_negative_evidence_v29i()["aggregate_execution"]
    assert execution["equivalence_pair_count"] == 8
    assert execution["exact_equivalence_pair_count"] == 8
    assert execution["equivalence_component_count"] == 4
    assert execution["all_equivalence_components_exact_for_all_pairs"] is True
    assert execution["paired_output_commitment_sha256"] == (
        evidence.PAIRED_OUTPUT_COMMITMENT_SHA256_V29I
    )


def test_v29i_all_16_activity_config_identity_cleanup_groups_and_64_loads_passed():
    execution = evidence.build_negative_evidence_v29i()["aggregate_execution"]
    assert execution["fresh_four_engine_group_count"] == 16
    assert execution["all_four_activity_group_count"] == 16
    assert execution[
        "all_16_groups_activity_config_identity_and_cleanup_passed"
    ] is True
    assert execution["serialized_fp8_tp1_model_load_count"] == 64
    assert execution["all_four_finally_idle"] is True
    assert execution["minimum_activity_sample_count"] == 116
    assert execution["minimum_simultaneous_positive_sample_count"] == 7


def test_v29i_global_latency_gate_fails_with_exact_values():
    performance = evidence.build_negative_evidence_v29i()["aggregate_performance"]
    assert performance["global_full_model_latency"] == {
        "familywise_lower_confidence_bound": 0.9633065683884765,
        "geometric_mean_speedup": 0.9813847810295329,
        "lower_bound_threshold": 0.99,
        "pass": False,
        "point_threshold": 1.002,
    }


def test_v29i_all_four_per_gpu_latency_gates_fail_with_exact_values():
    performance = evidence.build_negative_evidence_v29i()["aggregate_performance"]
    assert performance["latency_by_physical_gpu"] == (
        evidence._expected_latency_by_gpu_v29i()
    )
    assert all(
        endpoint["pass"] is False
        for endpoint in performance["latency_by_physical_gpu"].values()
    )
    assert performance["all_five_latency_endpoints_failed"] is True


def test_v29i_all_memory_ratios_ucbs_and_absolute_nvml_gate_pass():
    performance = evidence.build_negative_evidence_v29i()["aggregate_performance"]
    assert performance["global_peak_vram"] == {
        "familywise_upper_confidence_bound": 1.0,
        "max_per_gpu_median_ratio": 1.0,
        "pass": True,
        "point_threshold": 1.01,
        "upper_bound_threshold": 1.02,
    }
    assert performance["peak_vram_by_physical_gpu"] == (
        evidence._expected_peak_vram_by_gpu_v29i()
    )
    assert performance["all_five_vram_endpoints_passed"] is True
    assert performance["maximum_absolute_nvml_fraction"] == 0.7951821998835392
    assert performance["absolute_nvml_fraction_limit"] == 0.95
    assert performance["absolute_nvml_gate_passed"] is True
    assert performance["passing_endpoint_count"] == 5
    assert performance["failing_endpoint_count"] == 5


def test_v29i_retains_empty_default_and_closes_every_direct_authority():
    value = evidence.build_negative_evidence_v29i()
    assert value["decision"] == {
        "retain_empty_default_serialized_fp8_runtime": True,
        "direct_table_or_recipe_adoption_authority": False,
        "model_update_or_training_authority": False,
        "checkpoint_write_authority": False,
        "evaluation_validation_heldout_ood_or_benchmark_access_authority": False,
        "dataset_promotion_authority": False,
        "nontrain_runtime_reuse_authority": False,
    }
    assert all(item is False for item in value["side_effects"].values())


def test_v29i_input_scope_is_compact_only_and_no_gpu_or_semantic_access():
    scope = evidence.build_negative_evidence_v29i()["input_scope"]
    assert scope == {
        "compact_attempt_and_report_aggregates_only": True,
        "preregistration_table_or_checkpoint_content_opened": False,
        "raw_stdout_or_runtime_logs_opened": False,
        "dataset_or_semantic_content_opened": False,
        "gpu_launched": False,
    }


@pytest.mark.parametrize(
    "name,message",
    (
        ("ATTEMPT_PATH_V29I", "attempt file hash changed"),
        ("REPORT_PATH_V29I", "report file hash changed"),
    ),
)
def test_v29i_changed_compact_artifact_bytes_fail_closed(
    monkeypatch, tmp_path, name, message,
):
    source = getattr(evidence, name)
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b"\n")
    monkeypatch.setattr(evidence, name, changed)
    with pytest.raises(RuntimeError, match=message):
        evidence.validate_bound_artifacts_v29i()


def test_v29i_tampered_attempt_self_hash_or_report_binding_fails_closed(
    monkeypatch,
):
    attempt, report = _artifacts()
    changed = copy.deepcopy(attempt)
    changed["recipe_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="attempt self hash changed"):
        evidence._validate_attempt_v29i(changed, report)
    changed = copy.deepcopy(attempt)
    changed["report_binding"]["file_sha256"] = "0" * 64
    changed = evidence._seal(changed)
    monkeypatch.setattr(
        evidence,
        "ATTEMPT_CONTENT_SHA256_V29I",
        changed["content_sha256_before_self_field"],
    )
    with pytest.raises(RuntimeError, match="report binding changed"):
        evidence._validate_attempt_v29i(changed, report)


def test_v29i_tampered_equivalence_fails_closed():
    _attempt, report = _artifacts()
    changed = copy.deepcopy(report)
    changed["equivalence"]["exact_pair_count"] = 7
    with pytest.raises(RuntimeError, match="exact 8-pair equivalence"):
        evidence._validate_equivalence_v29i(changed)
    changed = copy.deepcopy(report)
    changed["equivalence"]["component_pass_counts"][
        "timed_commitments_exact"
    ] = 7
    with pytest.raises(RuntimeError, match="exact 8-pair equivalence"):
        evidence._validate_equivalence_v29i(changed)


@pytest.mark.parametrize(
    "key,value",
    (
        ("fresh_four_engine_group_count", 15),
        ("all_16_groups_activity_config_identity_and_cleanup_passed", False),
        ("serialized_fp8_tp1_model_load_count", 63),
        ("all_four_finally_idle", False),
    ),
)
def test_v29i_tampered_activity_config_cleanup_or_load_fails_closed(key, value):
    _attempt, report = _artifacts()
    changed = copy.deepcopy(report)
    changed["runtime_integrity"][key] = value
    with pytest.raises(RuntimeError, match="16-group activity"):
        evidence._validate_runtime_integrity_v29i(changed)


def test_v29i_tampered_global_or_per_gpu_latency_fails_closed():
    _attempt, report = _artifacts()
    changed = copy.deepcopy(report)
    changed["performance"]["global"]["full_model_latency"][
        "geometric_mean_speedup"
    ] = 1.01
    with pytest.raises(RuntimeError, match="global full-model latency"):
        evidence._validate_performance_v29i(changed)
    changed = copy.deepcopy(report)
    changed["performance"]["latency_by_physical_gpu"]["3"]["pass"] = True
    with pytest.raises(RuntimeError, match="four per-GPU latency"):
        evidence._validate_performance_v29i(changed)


def test_v29i_tampered_memory_or_nvml_fails_closed():
    _attempt, report = _artifacts()
    changed = copy.deepcopy(report)
    changed["performance"]["peak_vram_by_physical_gpu"]["0"][
        "familywise_upper_confidence_bound"
    ] = 1.03
    with pytest.raises(RuntimeError, match="VRAM endpoints"):
        evidence._validate_performance_v29i(changed)
    changed = copy.deepcopy(report)
    changed["performance"]["absolute_nvml_gate_passed"] = False
    with pytest.raises(RuntimeError, match="absolute NVML"):
        evidence._validate_performance_v29i(changed)


def test_v29i_tampered_authority_or_side_effect_fails_closed():
    attempt, report = _artifacts()
    changed = copy.deepcopy(report)
    changed["gate"]["direct_table_or_recipe_adoption_authorized"] = True
    changed["gate"] = evidence._seal(changed["gate"])
    with pytest.raises(RuntimeError, match="authorization gate self hash changed"):
        evidence._validate_closed_authority_v29i(attempt, changed)
    changed = copy.deepcopy(report)
    changed[
        "model_update_training_checkpoint_evaluation_or_dataset_action_applied"
    ] = True
    with pytest.raises(RuntimeError, match="forbidden side effect"):
        evidence._validate_closed_authority_v29i(attempt, changed)


@pytest.mark.parametrize(
    "forbidden",
    (
        "questions", "answers", "prompts", "token_ids", "outputs",
        "logprobs", "stdout", "stderr", "logs", "pids", "timings",
        "memory_samples", "pair_vectors", "bootstrap_replicates",
        "bootstrap_draws",
    ),
)
def test_v29i_compact_evidence_rejects_detailed_payload_keys(forbidden):
    with pytest.raises(RuntimeError, match="forbidden detailed payload keys"):
        evidence._assert_compact_v29i({forbidden: []})


def test_v29i_build_is_deterministic_and_dry_run_does_not_write(capsys):
    first = evidence.build_negative_evidence_v29i()
    assert first == evidence.build_negative_evidence_v29i()
    assert first["content_sha256_before_self_field"] == evidence.canonical_sha256(
        evidence._without_self(first)
    )
    result = evidence.main(["--dry-run"])
    output = json.loads(capsys.readouterr().out)
    assert result == first
    assert output["content_sha256"] == first[
        "content_sha256_before_self_field"
    ]
    assert output["gate_pass"] is False
    assert output["gpu_launched"] is False


def test_v29i_materialized_evidence_is_exact_build_output():
    materialized = json.loads(evidence.OUTPUT_PATH_V29I.read_text(encoding="utf-8"))
    assert materialized == evidence.build_negative_evidence_v29i()


def test_v29i_exclusive_write_is_immutable(monkeypatch, tmp_path):
    output = tmp_path / "negative.json"
    monkeypatch.setattr(evidence, "OUTPUT_PATH_V29I", output)
    value = evidence.build_negative_evidence_v29i()
    evidence._exclusive_write_json_v29i(output, value)
    with pytest.raises(RuntimeError, match="already exists"):
        evidence._exclusive_write_json_v29i(output, value)
