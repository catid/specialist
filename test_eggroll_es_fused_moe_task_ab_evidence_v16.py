#!/usr/bin/env python3
"""Focused compact-only replay test for the sealed V16 systems A/B."""

import hashlib
import json
import math
from pathlib import Path

import eggroll_es_fused_moe_task_ab_preregistration_v16 as prereg_v16


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_FUSED_MOE_TASK_AB_V16_AGGREGATE_EVIDENCE.json"
)


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def test_v16_compact_evidence_replays_exact_negative_material_speed_gate():
    evidence = json.loads(EVIDENCE.read_text())
    assert evidence["content_sha256_before_self_field"] == canonical_sha256(
        without_self(evidence)
    )
    documents = {}
    for name, binding in evidence["input_bindings"].items():
        path = Path(binding["path"])
        assert file_sha256(path) == binding["file_sha256"]
        documents[name] = json.loads(path.read_text())
        assert documents[name]["content_sha256_before_self_field"] == (
            binding["content_sha256"]
        )
        assert binding["content_sha256"] == canonical_sha256(
            without_self(documents[name])
        )

    report = documents["report"]
    attempt = documents["attempt"]
    assert attempt["status"] == "complete"
    assert attempt["report_binding"]["file_sha256"] == (
        evidence["input_bindings"]["report"]["file_sha256"]
    )
    assert attempt["report_binding"]["content_sha256"] == (
        report["content_sha256_before_self_field"]
    )
    assert attempt["gate_content_sha256"] == report["gate"][
        "content_sha256_before_self_field"
    ]
    assert attempt["source_provenance"] == report["source_provenance"]
    assert attempt["recipe"] == report["recipe"]

    candidate = report["candidate"]
    for name in prereg_v16.ARM_ORDER_V16:
        envelope = documents[name]
        assert envelope["compact_arm"] == candidate["arms"][name]
        assert envelope["fresh_process_arm_worker"] is True
        assert envelope["moe_backend"] == "triton"
        assert envelope["persisted_raw_content"] is False
        compact = candidate["arms"][name]
        assert compact["all_integrity_audits_passed"] is True
        assert compact["persisted_raw_content"] is False
        assert len(compact["generation_timing"]["wave_seconds"]) == 16
        assert math.isclose(
            math.fsum(compact["generation_timing"]["wave_seconds"]),
            compact["generation_timing"]["total_seconds"],
            rel_tol=1e-12, abs_tol=1e-12,
        )
        bound = evidence["arm_results"][name]
        assert bound["diagnostic_content_sha256"] == compact[
            "diagnostic_content_sha256"
        ]
        assert bound["dense_result_manifest_sha256"] == compact[
            "dense_result_manifest_sha256"
        ]
        assert bound["task_output_sha256"] == compact["task_output_sha256"]
        assert bound["compact_estimator_sha256"] == canonical_sha256(
            compact["compact_estimator"]
        )

    replayed = prereg_v16.evaluate_candidate_v16(candidate)
    assert replayed == report["gate"]
    assert replayed["exact_equivalence"] == {
        "diagnostic_content_sha256_equal": True,
        "dense_result_manifest_sha256_equal": True,
        "task_output_sha256_equal": True,
        "compact_estimator_equal": True,
    }
    timing = replayed["timing"]
    assert timing["total_generation_time_speedup"] == 1.0151885293261422
    assert timing["median_paired_wave_speedup"] == 1.0155962902048992
    assert timing["nonregressive_wave_count"] == 16
    assert timing["total_speedup_passed"] is False
    assert timing["median_speedup_passed"] is False
    assert replayed["eligible_for_later_opt_in_training_preregistration"] is False
    assert replayed["eligible_for_model_update"] is False
    assert replayed["eligible_to_open_evaluation"] is False
    assert evidence["decision"] == {
        "retain_default_triton_v13": True,
        "enable_tuned_config": False,
        "authorize_later_opt_in_training_preregistration": False,
        "authorize_model_update": False,
        "authorize_evaluation": False,
    }
