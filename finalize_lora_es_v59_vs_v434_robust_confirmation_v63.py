#!/usr/bin/env python3
"""Outcome-agnostic numeric finalizer for a future sealed V63 live run.

Production outcome hashes intentionally remain unset in this package.  The
finalizer therefore fails closed until a complete fixed run exists and its
file/content hashes are explicitly sealed.  Tests exercise both scientific
outcomes and tampering without launching a model or GPU.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import lora_es_v59_vs_v434_robust_confirmation_v63 as analysis
import run_lora_es_v59_vs_v434_robust_confirmation_v63 as runtime


ROOT = Path(__file__).resolve().parent
OUTPUT = (runtime.RUN_DIR / "confirmation_finalized_v63.json").resolve()
EVIDENCE_FILE_SHA256: str | None = None
EVIDENCE_CONTENT_SHA256: str | None = None
ANALYSIS_FILE_SHA256: str | None = None
ANALYSIS_CONTENT_SHA256: str | None = None
REPORT_FILE_SHA256: str | None = None
REPORT_CONTENT_SHA256: str | None = None

FORBIDDEN_TEXT_KEYS_V63 = {
    "answer", "completion", "completion_text", "generated_text",
    "output_text", "outputs", "prediction", "prompt", "prompt_token_ids",
    "question", "raw_text", "response", "text", "token_ids",
}


@dataclass(frozen=True)
class SelfHashedSourceV63:
    path: Path
    file_sha256: str
    content_sha256: str


@dataclass(frozen=True)
class FinalizerSourcesV63:
    preregistration: SelfHashedSourceV63
    evidence: SelfHashedSourceV63
    analysis: SelfHashedSourceV63
    report: SelfHashedSourceV63


def _require_sha256_v63(value: str | None, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"v63 production outcome hash is not sealed: {name}")
    return value


def production_sources_v63(
    preregistration_file_sha256: str,
    preregistration_content_sha256: str,
) -> FinalizerSourcesV63:
    return FinalizerSourcesV63(
        preregistration=SelfHashedSourceV63(
            runtime.PREREGISTRATION,
            _require_sha256_v63(
                preregistration_file_sha256, "preregistration file"
            ),
            _require_sha256_v63(
                preregistration_content_sha256, "preregistration content"
            ),
        ),
        evidence=SelfHashedSourceV63(
            runtime.EVIDENCE,
            _require_sha256_v63(EVIDENCE_FILE_SHA256, "evidence file"),
            _require_sha256_v63(EVIDENCE_CONTENT_SHA256, "evidence content"),
        ),
        analysis=SelfHashedSourceV63(
            runtime.ANALYSIS,
            _require_sha256_v63(ANALYSIS_FILE_SHA256, "analysis file"),
            _require_sha256_v63(ANALYSIS_CONTENT_SHA256, "analysis content"),
        ),
        report=SelfHashedSourceV63(
            runtime.REPORT,
            _require_sha256_v63(REPORT_FILE_SHA256, "report file"),
            _require_sha256_v63(REPORT_CONTENT_SHA256, "report content"),
        ),
    )


def _read_self_hashed_v63(source: SelfHashedSourceV63) -> dict:
    if runtime.file_sha256_v63(source.path) != source.file_sha256:
        raise RuntimeError(f"v63 finalizer input file changed: {source.path}")
    value = json.loads(source.path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != source.content_sha256
        or analysis.canonical_sha256_v63(compact) != source.content_sha256
    ):
        raise RuntimeError(
            f"v63 finalizer input content changed: {source.path}"
        )
    return value


def _verify_no_text_keys_v63(name: str, value: object) -> dict:
    found = []

    def visit(item: object, path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if str(key).lower() in FORBIDDEN_TEXT_KEYS_V63:
                    found.append(f"{path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, name)
    if found:
        raise RuntimeError(f"v63 forbidden text-bearing key: {found[0]}")
    return {"source": name, "forbidden_text_key_count": 0}


def _verify_preregistration_v63(prereg: dict) -> dict:
    if (
        prereg.get("schema")
        != "v63-v59-vs-v434-train-only-robust-confirmation-preregistration"
        or prereg.get("status")
        != "preregistered_before_train_semantics_model_or_gpu_access"
        or prereg.get("specific_v63_confirmation_gpu_launch_authorized")
        is not True
        or prereg.get(
            "update_hpo_candidate_promotion_or_protected_access_authorized"
        ) is not False
        or prereg.get("v62b_finalized_eligibility")
        != runtime.verify_v62b_eligibility_v63()
        or prereg.get("required_confirmation_gates")
        != runtime.required_gates_v63()
        or prereg.get("fixed_confirmation_recipe")
        != runtime.fixed_recipe_v63()
        or prereg.get("protected_semantics_opened") is not False
        or prereg.get("ood_shadow_holdout_or_terminal_opened") is not False
    ):
        raise RuntimeError("v63 finalizer preregistration changed")
    return {
        "fixed_recipe_exact": True,
        "fixed_gates_exact": True,
        "v62b_finalized_eligibility_exact": True,
    }


def _verify_report_v63(
    report: dict,
    evidence: dict,
    stored_analysis: dict,
    sources: FinalizerSourcesV63,
) -> dict:
    gate = stored_analysis["required_confirmation_gate"]
    expected_status = (
        "complete_gate_passed_without_promotion_authority"
        if gate["passed"] else "complete_gate_failed_closed"
    )
    actors = report.get("actor_identities", [])
    gpu_ids = {item.get("physical_gpu_id") for item in actors}
    pids = {item.get("pid") for item in actors}
    if (
        report.get("schema")
        != "v63-v59-vs-v434-train-only-confirmation-report"
        or report.get("status") != expected_status
        or report.get("evidence", {}).get("file_sha256")
        != sources.evidence.file_sha256
        or report.get("evidence", {}).get("content_sha256")
        != sources.evidence.content_sha256
        or report.get("analysis", {}).get("file_sha256")
        != sources.analysis.file_sha256
        or report.get("analysis", {}).get("content_sha256")
        != sources.analysis.content_sha256
        or report.get("analysis", {}).get("required_confirmation_gate") != gate
        or report.get("v62b_finalized")
        != runtime.verify_v62b_eligibility_v63()
        or report.get("adapter_artifact_identities")
        != analysis.expected_adapter_identities_v63()
        or report.get("generation_only") is not True
        or report.get("teacher_forced_requests") != 0
        or report.get(
            "adaptive_retry_drop_reorder_or_early_stop_performed"
        ) is not False
        or report.get("median_consensus_or_best_of_selection_performed")
        is not False
        or report.get(
            "adapter_update_hpo_master_checkpoint_or_promotion_performed"
        ) is not False
        or report.get("holdback_ood_shadow_or_protected_opened") is not False
        or report.get("raw_question_answer_or_generation_text_persisted")
        is not False
        or report.get(
            "result_authorizes_update_hpo_promotion_or_protected_access"
        ) is not False
        or len(actors) != analysis.ACTORS_V63
        or gpu_ids != {0, 1, 2, 3}
        or len(pids) != analysis.ACTORS_V63
        or any(not isinstance(pid, int) or pid <= 0 for pid in pids)
        or evidence.get("content_sha256_before_self_field")
        != sources.evidence.content_sha256
    ):
        raise RuntimeError("v63 finalizer report or integrity changed")
    return {
        "actor_count": analysis.ACTORS_V63,
        "physical_gpu_ids": [0, 1, 2, 3],
        "unique_processes": analysis.ACTORS_V63,
        "fixed_complete_schedule": True,
        "result_authority": False,
    }


def finalize_v63(sources: FinalizerSourcesV63) -> dict:
    prereg = _read_self_hashed_v63(sources.preregistration)
    evidence = _read_self_hashed_v63(sources.evidence)
    stored_analysis = _read_self_hashed_v63(sources.analysis)
    report = _read_self_hashed_v63(sources.report)
    prereg_verification = _verify_preregistration_v63(prereg)
    analysis.validate_evidence_v63(evidence)
    rebuilt = analysis.build_analysis_v63(evidence)
    if rebuilt != stored_analysis:
        raise RuntimeError("v63 stored analysis differs from exact numeric rebuild")
    report_verification = _verify_report_v63(
        report, evidence, stored_analysis, sources
    )
    no_text = {
        name: _verify_no_text_keys_v63(name, value)
        for name, value in (
            ("evidence", evidence),
            ("analysis", stored_analysis),
            ("report", report),
        )
    }
    gate = copy.deepcopy(stored_analysis["required_confirmation_gate"])
    failed = [name for name, passed in gate["checks"].items() if not passed]
    value = {
        "schema": "v63-v59-vs-v434-independent-numeric-finalizer",
        "status": (
            "complete_gate_passed_without_authority"
            if gate["passed"] else "complete_gate_failed_closed"
        ),
        "source_hashes": {
            name: {
                "file_sha256": source.file_sha256,
                "content_sha256": source.content_sha256,
            }
            for name, source in (
                ("preregistration", sources.preregistration),
                ("evidence", sources.evidence),
                ("analysis", sources.analysis),
                ("report", sources.report),
            )
        },
        "observed_numeric_outcome_without_authorization": {
            "primary_generated_f1": copy.deepcopy(
                stored_analysis["primary_generated_f1"]
            ),
            "actor_influence": copy.deepcopy(
                stored_analysis["actor_influence"]
            ),
            "required_confirmation_gate": gate,
            "exact_sentinel_diagnostics": copy.deepcopy(
                stored_analysis["exact_sentinel_diagnostics"]
            ),
            "failed_gates": failed,
            "passed_gate_count": len(gate["checks"]) - len(failed),
            "failed_gate_count": len(failed),
        },
        "verification": {
            "preregistration": prereg_verification,
            "report": report_verification,
            "analysis_exactly_rebuilt": True,
            "no_text_leakage": no_text,
            "all_file_and_self_hashes_verified": True,
        },
        "frozen_non_authorization": {
            "finalizer_accepts_and_records_either_gate_outcome": True,
            "outcome_assumed_before_read": False,
            "thresholds_changed_after_outcome": False,
            "failed_gate_reinterpreted_or_relaxed": False,
            "gpu_or_model_launch_authorized": False,
            "adapter_update_hpo_candidate_promotion_authorized": False,
            "holdback_ood_shadow_terminal_or_protected_access_authorized": False,
        },
        "raw_question_answer_prediction_or_generation_text_persisted": False,
        "protected_semantics_opened": False,
    }
    value["content_sha256_before_self_field"] = (
        analysis.canonical_sha256_v63(value)
    )
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration-file-sha256", required=True)
    parser.add_argument("--preregistration-content-sha256", required=True)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    result = finalize_v63(production_sources_v63(
        args.preregistration_file_sha256,
        args.preregistration_content_sha256,
    ))
    runtime.runtime_v61a.atomic_json_v61a(output, result)
    print(json.dumps({
        "path": str(output),
        "file_sha256": runtime.file_sha256_v63(output),
        "content_sha256": result["content_sha256_before_self_field"],
        "required_confirmation_gate_passed": result[
            "observed_numeric_outcome_without_authorization"
        ]["required_confirmation_gate"]["passed"],
        "update_hpo_promotion_or_protected_access_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
