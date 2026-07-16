#!/usr/bin/env python3

import hashlib

import pytest

import run_lora_es_base_generation_evidence_v48b as evidence_runtime
import seal_lora_es_generation_boundary_subset_v48b as subject
import run_lora_es_multi_anchor_v43i as v43i


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _inputs(tmp_path):
    membership = subject._read_sealed(
        evidence_runtime.MEMBERSHIP,
        evidence_runtime.EXPECTED_MEMBERSHIP_SHA256,
        evidence_runtime.EXPECTED_MEMBERSHIP_CONTENT_SHA256,
    )
    rows = []
    for item in membership["items"]:
        kind = int(item["unit_identity_sha256"][:2], 16) % 4
        actors = []
        for actor in range(4):
            if kind == 0:
                f1, exact, nonzero = (0.4 if actor < 2 else 0.6), 0, 1
                prediction = _sha(f"unstable-{actor}")
            elif kind == 1:
                f1, exact, nonzero, prediction = 0.5, 0, 1, _sha("partial")
            elif kind == 2:
                f1, exact, nonzero, prediction = 1.0, 1, 1, _sha("exact")
            else:
                f1, exact, nonzero, prediction = 0.0, 0, 0, _sha("zero")
            actors.append({
                "actor_rank": actor, "prediction_sha256": prediction,
                "f1": f1, "exact": exact, "nonzero": nonzero,
            })
        rows.append({
            "row_index": item["row_index"],
            "row_sha256": item["row_sha256"],
            "unit_identity_sha256": item["unit_identity_sha256"],
            "row_count": item["row_count"], "actors": actors,
        })
    evidence = v43i.v40a.self_hashed({
        "schema": "train-only-four-actor-base-generation-evidence-v48a",
        "revision": "v48b", "status": "complete_before_population",
        "train_bundle_content_sha256": v43i.TRAIN_BUNDLE_SHA256,
        "train_dataset_file_sha256": evidence_runtime.EXPECTED_TRAIN_SHA256,
        "membership_file_sha256": evidence_runtime.EXPECTED_MEMBERSHIP_SHA256,
        "membership_content_sha256": evidence_runtime.EXPECTED_MEMBERSHIP_CONTENT_SHA256,
        "membership_ordered_sha256": membership["ordered_membership_sha256"],
        "matched_master_sha256": evidence_runtime.v43m.V43I_RESTORED_MASTER_SHA256,
        "generation_params": dict(subject.boundary.GENERATION_PARAMS_V48A),
        "rows": rows, "row_count": 448, "actor_count": 4,
        "raw_question_answer_or_generation_text_persisted": False,
        "selection_or_population_opened": False,
        "protected_semantics_opened": False,
        "shadow_ood_holdout_or_benchmark_opened": False,
    })
    evidence_path = tmp_path / "evidence.json"
    v43i.v40a.atomic_json(evidence_path, evidence)
    evidence_file = v43i.v40a.file_sha256(evidence_path)
    report = v43i.v40a.self_hashed({
        "schema": "matched-lora-es-base-generation-evidence-report-v48b",
        "status": "complete_train_only_evidence_sealed",
        "evidence": {
            "file_sha256": evidence_file,
            "content_sha256": evidence["content_sha256_before_self_field"],
        },
        "gpu_activity": {
            "all_four_attributed_positive": True,
            "by_gpu": {str(index): {"positive_samples": 1} for index in range(4)},
        },
        "raw_question_answer_or_generation_text_persisted": False,
        "protected_semantics_opened": False,
        "shadow_ood_holdout_or_benchmark_opened": False,
    })
    report_path = tmp_path / "report.json"
    v43i.v40a.atomic_json(report_path, report)
    return (
        evidence_path, evidence_file, evidence["content_sha256_before_self_field"],
        report_path, v43i.v40a.file_sha256(report_path),
        report["content_sha256_before_self_field"],
    )


def test_v48b_subset_seals_64_distinct_units_without_semantics(tmp_path):
    value = subject.seal_subset_v48b(*_inputs(tmp_path))
    assert value["selected_rows"] == 64
    assert value["selected_conflict_units"] == 64
    assert len({
        item["unit_identity_sha256"] for item in value["subset"]["items"]
    }) == 64
    assert value["question_answer_or_generation_text_persisted"] is False
    assert value["gpu_or_model_accessed"] is False
    assert value["protected_semantics_opened"] is False


def test_v48b_subset_rejects_wrong_evidence_hash(tmp_path):
    args = list(_inputs(tmp_path))
    args[1] = _sha("wrong")
    with pytest.raises(RuntimeError, match="input file changed"):
        subject.seal_subset_v48b(*args)
