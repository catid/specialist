import copy
import json

import pytest

import build_eggroll_es_v28c_negative_evidence_v28d as evidence


def _artifacts():
    return (
        json.loads(evidence.ATTEMPT_PATH_V28D.read_text(encoding="utf-8")),
        json.loads(evidence.REPORT_PATH_V28D.read_text(encoding="utf-8")),
    )


def test_v28d_binds_exact_completed_compact_artifacts():
    value = evidence.build_negative_evidence_v28d()
    assert value["artifacts"] == {
        "durable_attempt": {
            "relative_path": evidence.ATTEMPT_RELATIVE_PATH_V28D,
            "file_sha256": evidence.ATTEMPT_FILE_SHA256_V28D,
            "content_sha256": evidence.ATTEMPT_CONTENT_SHA256_V28D,
        },
        "compact_report": {
            "relative_path": evidence.REPORT_RELATIVE_PATH_V28D,
            "file_sha256": evidence.REPORT_FILE_SHA256_V28D,
            "content_sha256": evidence.REPORT_CONTENT_SHA256_V28D,
        },
    }


def test_v28d_binds_exact_source_recipe_environment_and_idle_aggregates():
    bindings = evidence.build_negative_evidence_v28d()["frozen_bindings"]
    assert bindings == {
        "implementation_bundle_sha256": evidence.IMPLEMENTATION_BUNDLE_SHA256_V28D,
        "recipe_content_sha256": evidence.RECIPE_CONTENT_SHA256_V28D,
        "committed_clean_source_content_sha256": (
            evidence.COMMITTED_CLEAN_SOURCE_CONTENT_SHA256_V28D
        ),
        "runtime_environment_content_sha256": (
            evidence.RUNTIME_ENVIRONMENT_CONTENT_SHA256_V28D
        ),
        "live_cpu_disk_audit_content_sha256": (
            evidence.LIVE_CPU_DISK_AUDIT_CONTENT_SHA256_V28D
        ),
        "prelaunch_idle_content_sha256": evidence.PRELAUNCH_IDLE_CONTENT_SHA256_V28D,
        "final_idle_content_sha256": evidence.FINAL_IDLE_CONTENT_SHA256_V28D,
        "authorization_gate_content_sha256": evidence.GATE_CONTENT_SHA256_V28D,
    }


def test_v28d_all_twelve_pairs_and_every_component_are_exact():
    execution = evidence.build_negative_evidence_v28d()["aggregate_execution"]
    assert execution["equivalence_pair_count"] == 12
    assert execution["exact_equivalence_pair_count"] == 12
    assert execution["equivalence_component_count"] == 8
    assert execution["all_equivalence_components_exact_for_all_pairs"] is True


def test_v28d_all_24_activity_cleanup_and_idle_gates_passed():
    execution = evidence.build_negative_evidence_v28d()["aggregate_execution"]
    assert execution["fresh_engine_group_count"] == 24
    assert execution["all_four_activity_group_count"] == 24
    assert execution["all_24_fresh_group_cleanup_gates_passed"] is True
    assert execution["all_24_groups_observed_all_four_active"] is True
    assert execution["final_all_four_idle"] is True
    assert execution["physical_gpu_identity_preserved"] is True
    assert execution["minimum_activity_sample_count"] == 3269
    assert execution["minimum_qualifying_activity_sample_count"] == 2947


def test_v28d_speed_is_the_sole_performance_failure():
    performance = evidence.build_negative_evidence_v28d()["aggregate_performance"]
    assert performance["inferential_endpoint_count"] == 3
    assert performance["passing_endpoint_count"] == 2
    assert performance["failing_endpoint_count"] == 1
    assert performance["sole_failure"] == "complete_train_step_speed"
    speed = performance["complete_train_step"]
    assert speed == {
        "median_default_over_tuned_ratio": 1.003176690951185,
        "point_threshold": 1.01,
        "point_gate_passed": False,
        "familywise_lower_confidence_bound": 0.9945921239188809,
        "lower_bound_strict_threshold": 1.0,
        "lower_bound_gate_passed": False,
        "endpoint_passed": False,
    }


def test_v28d_torch_memory_and_absolute_vram_gates_passed():
    performance = evidence.build_negative_evidence_v28d()["aggregate_performance"]
    expected = {
        "familywise_upper_confidence_bound": 1.0,
        "median_tuned_over_default_ratio": 1.0,
        "pass": True,
        "point_threshold": 1.01,
        "upper_bound_threshold": 1.02,
    }
    assert performance["peak_torch_allocated"] == expected
    assert performance["peak_torch_reserved"] == expected
    assert performance["memory_ratio_and_familywise_upper_bound_gates_passed"] is True
    assert performance["absolute_peak_nvml_fraction_observed"] == 0.8575398163188166
    assert performance["absolute_peak_nvml_fraction_limit"] == 0.95
    assert performance["absolute_peak_nvml_gate_passed"] is True


def test_v28d_retains_empty_default_and_closes_every_direct_authority():
    value = evidence.build_negative_evidence_v28d()
    assert value["decision"] == {
        "retain_empty_default_bf16_training_recipe": True,
        "v27c_tuned_recipe_direct_adoption_authority": False,
        "model_update_authority": False,
        "checkpoint_write_authority": False,
        "evaluation_authority": False,
        "dataset_authority": False,
        "fp8_authority": False,
        "nontrain_authority": False,
    }
    assert all(item is False for item in value["side_effects"].values())
    assert value["input_scope"] == {
        "compact_attempt_and_report_aggregates_only": True,
        "raw_stdout_or_runtime_logs_opened": False,
        "dataset_or_semantic_content_opened": False,
        "detailed_runtime_payloads_persisted": False,
    }


@pytest.mark.parametrize(
    "name,message",
    (
        ("ATTEMPT_PATH_V28D", "attempt file hash changed"),
        ("REPORT_PATH_V28D", "report file hash changed"),
    ),
)
def test_v28d_changed_compact_artifact_bytes_fail_closed(
    monkeypatch, tmp_path, name, message,
):
    source = getattr(evidence, name)
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b"\n")
    monkeypatch.setattr(evidence, name, changed)
    with pytest.raises(RuntimeError, match=message):
        evidence.validate_bound_artifacts_v28d()


def test_v28d_tampered_attempt_self_hash_or_report_binding_fails_closed(monkeypatch):
    attempt, report = _artifacts()
    changed = copy.deepcopy(attempt)
    changed["recipe_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="attempt self hash changed"):
        evidence._validate_attempt_v28d(changed, report)
    changed = copy.deepcopy(attempt)
    changed["report_binding"]["file_sha256"] = "0" * 64
    changed = evidence._seal(changed)
    monkeypatch.setattr(
        evidence,
        "ATTEMPT_CONTENT_SHA256_V28D",
        changed["content_sha256_before_self_field"],
    )
    with pytest.raises(RuntimeError, match="report binding changed"):
        evidence._validate_attempt_v28d(changed, report)


def test_v28d_tampered_equivalence_pair_or_component_fails_closed():
    _attempt, report = _artifacts()
    changed = copy.deepcopy(report)
    changed["equivalence"]["exact_pair_count"] = 11
    with pytest.raises(RuntimeError, match="exact 12-pair equivalence"):
        evidence._validate_equivalence_v28d(changed)
    changed = copy.deepcopy(report)
    changed["equivalence"]["component_pass_counts"][
        "identity_guard_exact"
    ] = 11
    with pytest.raises(RuntimeError, match="exact 12-pair equivalence"):
        evidence._validate_equivalence_v28d(changed)


def test_v28d_tampered_activity_cleanup_or_idle_fails_closed():
    _attempt, report = _artifacts()
    for key, value in (
        ("fresh_engine_group_count", 23),
        ("all_24_groups_observed_all_four_active", False),
        ("final_all_four_idle", False),
    ):
        changed = copy.deepcopy(report)
        changed["runtime_integrity"][key] = value
        with pytest.raises(RuntimeError, match="24-group activity or cleanup"):
            evidence._validate_runtime_integrity_v28d(changed)


def test_v28d_tampered_speed_memory_or_vram_fails_closed():
    _attempt, report = _artifacts()
    changed = copy.deepcopy(report)
    changed["performance"]["endpoints"]["complete_train_step_speed"][
        "median_default_over_tuned_ratio"
    ] = 1.02
    with pytest.raises(RuntimeError, match="sole performance failure"):
        evidence._validate_performance_v28d(changed)
    changed = copy.deepcopy(report)
    changed["performance"]["endpoints"]["peak_torch_allocated"][
        "familywise_upper_confidence_bound"
    ] = 1.03
    with pytest.raises(RuntimeError, match="memory endpoints"):
        evidence._validate_performance_v28d(changed)
    changed = copy.deepcopy(report)
    changed["performance"]["absolute_peak_nvml_gate_passed"] = False
    with pytest.raises(RuntimeError, match="absolute VRAM"):
        evidence._validate_performance_v28d(changed)


def test_v28d_tampered_authority_or_side_effect_fails_closed():
    attempt, report = _artifacts()
    changed = copy.deepcopy(report)
    changed["gate"]["direct_recipe_adoption_authorized"] = True
    changed["gate"] = evidence._seal(changed["gate"])
    with pytest.raises(RuntimeError, match="authorization gate self hash changed"):
        evidence._validate_closed_authority_v28d(attempt, changed)
    changed = copy.deepcopy(report)
    changed["model_update_applied"] = True
    with pytest.raises(RuntimeError, match="forbidden side effect"):
        evidence._validate_closed_authority_v28d(attempt, changed)


@pytest.mark.parametrize(
    "forbidden",
    (
        "questions", "answers", "prompts", "tokens", "stdout", "stderr",
        "logs", "pids", "timings", "memory_samples", "pair_vectors",
        "bootstrap_replicates", "bootstrap_draws",
    ),
)
def test_v28d_compact_evidence_rejects_detailed_payload_keys(forbidden):
    with pytest.raises(RuntimeError, match="forbidden detailed payload keys"):
        evidence._assert_compact_v28d({forbidden: []})


def test_v28d_build_is_deterministic_and_dry_run_does_not_write(capsys):
    first = evidence.build_negative_evidence_v28d()
    assert first == evidence.build_negative_evidence_v28d()
    assert first["content_sha256_before_self_field"] == evidence.canonical_sha256(
        evidence._without_self(first)
    )
    result = evidence.main(["--dry-run"])
    output = json.loads(capsys.readouterr().out)
    assert result == first
    assert output["content_sha256"] == first["content_sha256_before_self_field"]
    assert output["gate_pass"] is False
    assert output["gpu_launched"] is False


def test_v28d_materialized_evidence_is_exact_build_output():
    materialized = json.loads(evidence.OUTPUT_PATH_V28D.read_text(encoding="utf-8"))
    assert materialized == evidence.build_negative_evidence_v28d()


def test_v28d_exclusive_write_is_immutable(monkeypatch, tmp_path):
    output = tmp_path / "negative.json"
    monkeypatch.setattr(evidence, "OUTPUT_PATH_V28D", output)
    value = evidence.build_negative_evidence_v28d()
    evidence._exclusive_write_json_v28d(output, value)
    with pytest.raises(RuntimeError, match="already exists"):
        evidence._exclusive_write_json_v28d(output, value)
