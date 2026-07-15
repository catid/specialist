import copy
import json
from pathlib import Path

import pytest

import build_eggroll_es_v23a_seed_failure_evidence_r1 as evidence_r1


SOURCE_ATTEMPT = Path(
    "/home/catid/specialist/experiments/eggroll_es_hpo/runs/"
    ".insertion_location_stability_v23a_authoritative_raw.launch_attempt.json"
)
SOURCE_REPORT = Path(
    "/home/catid/specialist/experiments/eggroll_es_hpo/runs/"
    "insertion_location_stability_v23a_authoritative_raw/"
    "insertion_location_stability_v23a.json"
)


def test_v23a_r1_failure_evidence_is_compact_and_exact():
    value = evidence_r1.build_failure_evidence_r1(SOURCE_ATTEMPT, SOURCE_REPORT)
    assert value["failed_attempt"]["file_sha256"] == evidence_r1.EXPECTED_ATTEMPT_FILE_SHA256_R1
    assert value["failed_attempt"]["content_sha256"] == evidence_r1.EXPECTED_ATTEMPT_CONTENT_SHA256_R1
    assert value["failed_attempt"]["compact_report_absent"] is True
    assert value["seed_domain"]["direction_count"] == 32
    assert value["seed_domain"]["all_direction_seeds_exceed_numpy_legacy_max"] is True
    assert value["seed_domain"]["numpy_projection_unique_count"] == 32
    assert value["seed_domain"]["numpy_projection_minimum"] == 276_724_920
    assert value["seed_domain"]["numpy_projection_maximum"] == 4_263_717_227
    assert value["seed_domain"]["numpy_projection_contains_zero"] is False
    basis = evidence_r1.prereg_v23a.perturbation_basis_v23a()["direction_seeds"]
    mapping = [{"full_seed": seed, "numpy_legacy_seed": seed % (2**32)} for seed in basis]
    assert value["seed_domain"]["full_to_numpy_projection_sha256"] == (
        evidence_r1.prereg_v23a.canonical_sha256(mapping)
    )
    assert value["failure_boundary"]["selected_parameter_add_reached"] is False
    assert value["failure_boundary"]["perturbed_generation_reached"] is False
    assert value["traceback_or_model_repr_persisted"] is False
    assert value["row_or_response_content_persisted"] is False
    assert value["content_sha256_before_self_field"] == evidence_r1.prereg_v23a.canonical_sha256(
        evidence_r1._without_self(value)
    )
    assert not (evidence_r1.FORBIDDEN_KEYS_R1 & set(evidence_r1._recursive_keys(value)))


def test_v23a_r1_failure_evidence_rejects_mutation_and_report(tmp_path):
    attempt = json.loads(SOURCE_ATTEMPT.read_text())
    attempt["model_update_applied"] = True
    bad_attempt = tmp_path / "attempt.json"
    bad_attempt.write_text(json.dumps(attempt))
    with pytest.raises(RuntimeError, match="evidence changed"):
        evidence_r1.build_failure_evidence_r1(bad_attempt, tmp_path / "missing.json")

    report = tmp_path / "report.json"
    report.write_text("{}")
    with pytest.raises(RuntimeError, match="evidence changed"):
        evidence_r1.build_failure_evidence_r1(SOURCE_ATTEMPT, report)


def test_v23a_r1_failure_evidence_forbids_verbose_or_dataset_payloads():
    for key in ("traceback", "message", "question", "responses"):
        with pytest.raises(RuntimeError, match="forbidden keys"):
            evidence_r1._assert_compact({key: "not allowed"})
