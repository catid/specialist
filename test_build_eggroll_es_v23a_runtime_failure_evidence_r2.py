import json
from pathlib import Path

import build_eggroll_es_v23a_runtime_failure_evidence_r2 as evidence_r2


MAIN_RUNS = Path("/home/catid/specialist/experiments/eggroll_es_hpo/runs")
MAIN_ATTEMPT = MAIN_RUNS / evidence_r2.ATTEMPT_RELATIVE_PATH_R2.split("runs/", 1)[1]
MAIN_REPORT = MAIN_RUNS / evidence_r2.REPORT_RELATIVE_PATH_R2.split("runs/", 1)[1]


def test_v23a_r2_rebuilds_compact_r1_environment_failure_exactly():
    rebuilt = evidence_r2.build_runtime_failure_evidence_r2(MAIN_ATTEMPT, MAIN_REPORT)
    persisted = json.loads(evidence_r2.OUTPUT_PATH_R2.read_text(encoding="utf-8"))
    assert rebuilt == persisted
    assert rebuilt["failed_attempt"]["failure_type"] == "ModuleNotFoundError"
    assert rebuilt["failed_attempt"]["missing_dependency"] == "vllm"
    assert rebuilt["failure_boundary"]["engine_actor_creation_reached"] is False
    assert rebuilt["failure_boundary"]["reference_scoring_reached"] is False
    assert rebuilt["failure_boundary"]["perturbation_reached"] is False
    assert rebuilt["failed_attempt"]["model_update_applied"] is False
    assert rebuilt["failed_attempt"]["nontrain_surface_opened"] is False
    assert rebuilt["traceback_or_model_repr_persisted"] is False
    assert rebuilt["row_or_response_content_persisted"] is False
