#!/usr/bin/env python3

import copy
import json

import pytest

import build_vllm_moe_fp8_selected_table_evaluation_positive_evidence_v29f as evidence


def _artifacts():
    return tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (
            evidence.PREREG_PATH_V29F,
            evidence.SELECTION_PATH_V29F,
            evidence.FAILURE_PATH_V29F,
            evidence.TABLE_PATH_V29F,
            evidence.ATTEMPT_PATH_V29F,
            evidence.REPORT_PATH_V29F,
        )
    )


def test_v29f_evidence_rebuilds_exactly_and_is_self_sealed():
    value = evidence.build_positive_evidence_v29f()
    persisted = json.loads(evidence.OUTPUT_PATH_V29F.read_text(encoding="utf-8"))
    assert value == persisted
    assert value["content_sha256_before_self_field"] == evidence.canonical_sha256(
        evidence._without_self(value)
    )
    assert value["status"] == "valid_completed_synthetic_kernel_evaluation_passed"


def test_v29f_binds_exact_commits_sources_and_selected_table():
    artifacts = evidence.validate_bound_artifacts_v29f()
    assert len(artifacts) == 6
    value = evidence.build_positive_evidence_v29f()
    assert value["contracts"] == {
        "v29e_preregistration_commit": evidence.PREREG_COMMIT_V29F,
        "v29e_preregistration_file_sha256": evidence.PREREG_FILE_SHA256_V29F,
        "v29e_preregistration_content_sha256": evidence.PREREG_CONTENT_SHA256_V29F,
        "v29e_implementation_bundle_sha256": evidence.IMPLEMENTATION_BUNDLE_SHA256_V29F,
        "v29e_recipe_content_sha256": evidence.RECIPE_CONTENT_SHA256_V29F,
        "runtime_environment_certificate_sha256": evidence.RUNTIME_ENVIRONMENT_SHA256_V29F,
        "live_cpu_disk_audit_content_sha256": evidence.LIVE_CPU_DISK_AUDIT_SHA256_V29F,
        "v29c_selection_commit": evidence.SELECTION_COMMIT_V29F,
        "v29d_failure_evidence_commit": evidence.FAILURE_COMMIT_V29F,
    }
    table = value["artifacts"]["committed_selected_table"]
    assert table["relative_path"] == evidence.TABLE_RELATIVE_PATH_V29F
    assert table["file_sha256"] == evidence.TABLE_FILE_SHA256_V29F
    assert table["content_sha256"] == evidence.TABLE_CONTENT_SHA256_V29F


def test_v29f_exact_aggregate_gates_and_integrity_are_preserved():
    result = evidence.build_positive_evidence_v29f()["aggregate_result"]
    assert result["strict_synthetic_kernel_only"] is True
    assert result["physical_gpu_count"] == 4
    assert result["repetitions"] == 8
    assert result["counterbalanced_arm_count"] == 16
    assert result["official_iterations_per_arm"] == 1000
    assert result["endpoint_count"] == 10
    assert result["per_gpu"] == evidence.EXPECTED_PER_GPU_V29F
    assert result["global"] == evidence.EXPECTED_GLOBAL_V29F
    assert result["exact_output_equivalence"] == {
        "matched_pairs": 32,
        "paired_output_digest_commitment_sha256": (
            evidence.OUTPUT_COMMITMENT_SHA256_V29F
        ),
        "pass": True,
        "required_pairs": 32,
    }
    assert result["all_16_fresh_four_worker_arms_passed"] is True
    assert result["all_four_exact_gpus_finally_idle"] is True
    assert result[
        "minimum_one_simultaneous_all_four_positive_observation_per_arm"
    ] is True
    assert result["pass"] is True


def test_v29f_scope_is_synthetic_only_and_all_direct_authority_stays_closed():
    value = evidence.build_positive_evidence_v29f()
    decision = value["decision"]
    assert decision["synthetic_kernel_positive_evidence_valid"] is True
    assert decision[
        "authorize_only_separate_fp8_runtime_or_training_ab_preregistration"
    ] is True
    assert decision[
        "full_model_or_bf16_training_path_integration_demonstrated"
    ] is False
    for key in (
        "direct_selected_table_adoption_authorized",
        "training_or_model_update_authorized",
        "checkpoint_write_authorized",
        "dataset_promotion_authorized",
        "dataset_evaluation_validation_heldout_ood_or_benchmark_open_authorized",
    ):
        assert decision[key] is False
    assert value["contains_dataset_rows_questions_answers_or_document_content"] is False
    assert value["contains_validation_heldout_ood_or_benchmark_content"] is False
    assert value["raw_timing_memory_input_output_vectors_or_pids_persisted"] is False


@pytest.mark.parametrize(
    ("target_index", "mutate", "message"),
    (
        (
            0,
            lambda value: value["authority"].__setitem__(
                "direct_table_adoption_authorized", True
            ),
            "frozen preregistration contract changed",
        ),
        (
            1,
            lambda value: value["decision"].__setitem__(
                "selected_table_direct_adoption_authorized", True
            ),
            "selection evidence changed",
        ),
        (
            2,
            lambda value: value["failure_boundary"].__setitem__(
                "kernel_statistical_evaluation_completed", True
            ),
            "failure evidence changed",
        ),
        (
            3,
            lambda value: value["256"].__setitem__("num_warps", 8),
            "selected table identity or content changed",
        ),
        (
            4,
            lambda value: value["report_binding"].__setitem__(
                "content_sha256", "0" * 64
            ),
            "attempt report binding changed",
        ),
        (
            5,
            lambda value: value["summary"]["global"].__setitem__(
                "latency_gate_pass", False
            ),
            "aggregate result or authority changed",
        ),
        (
            5,
            lambda value: value["runtime_integrity"].__setitem__(
                "all_four_exact_gpus_finally_idle", False
            ),
            "runtime integrity changed",
        ),
    ),
)
def test_v29f_rejects_semantic_mutations(target_index, mutate, message):
    artifacts = list(_artifacts())
    artifacts[target_index] = copy.deepcopy(artifacts[target_index])
    mutate(artifacts[target_index])
    with pytest.raises(RuntimeError, match=message):
        evidence._validate_semantics_v29f(*artifacts)


@pytest.mark.parametrize(
    "path_name",
    (
        "PREREG_PATH_V29F",
        "SELECTION_PATH_V29F",
        "FAILURE_PATH_V29F",
        "TABLE_PATH_V29F",
        "ATTEMPT_PATH_V29F",
        "REPORT_PATH_V29F",
    ),
)
def test_v29f_rejects_changed_live_artifact_bytes(
    monkeypatch, tmp_path, path_name,
):
    source = getattr(evidence, path_name)
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b"\n")
    monkeypatch.setattr(evidence, path_name, changed)
    with pytest.raises(RuntimeError, match="file hash changed"):
        evidence.validate_bound_artifacts_v29f()


def test_v29f_compact_evidence_rejects_raw_or_data_payload_keys():
    for forbidden in (
        "question", "responses", "token_ids", "timing_vectors",
        "raw_pids", "raw_tensors", "traceback", "search_results",
    ):
        with pytest.raises(RuntimeError, match="forbidden keys"):
            evidence._assert_compact_v29f({forbidden: []})


def test_v29f_build_is_deterministic_and_dry(capsys):
    first = evidence.build_positive_evidence_v29f()
    second = evidence.main(["--dry-run"])
    output = json.loads(capsys.readouterr().out)
    assert first == second
    assert output["content_sha256"] == first["content_sha256_before_self_field"]
    assert output["synthetic_kernel_pass"] is True
    assert output["full_model_integration_demonstrated"] is False
    assert output["evaluation_authorized"] is False
    assert output["gpu_launched"] is False


def test_v29f_exclusive_writer_rejects_wrong_or_existing_path(tmp_path):
    value = evidence.build_positive_evidence_v29f()
    with pytest.raises(ValueError, match="output path changed"):
        evidence._exclusive_write_json(tmp_path / "wrong.json", value)
    assert evidence.OUTPUT_PATH_V29F.exists()
    with pytest.raises(RuntimeError, match="already exists"):
        evidence._exclusive_write_json(evidence.OUTPUT_PATH_V29F, value)
