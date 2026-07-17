#!/usr/bin/env python3
"""Seal the additive V73G controller-bootstrap repair and fresh profile paths."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
from pathlib import Path
from typing import Any, Mapping

import qwen36_v73g_exact_phase_profiler_contract as contract
import build_qwen36_v73e_exact_phase_profiler_preregistration as _base


ROOT = Path(__file__).resolve().parent
OUTPUT = contract.OUTPUT
EVIDENCE = (
    ROOT
    / "experiments/eggroll_es_hpo/"
    "qwen36_v73g_exact_phase_profiler_cpu_evidence_20260717.md"
).resolve()
SYSTEMS_ONLY_CLOSURE = (
    ROOT
    / "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_v73g_systems_only_closure.json"
).resolve()
SYSTEMS_ONLY_CLOSURE_AUDITOR = (
    ROOT / "audit_v73g_systems_only_import_graph_v1.py"
).resolve()

V73E_PREREGISTRATION = (
    ROOT
    / "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_v73e_exact_phase_profiler.json"
).resolve()
V73E_PROFILE_ATTEMPT = (
    ROOT
    / "experiments/eggroll_es_hpo/profiles/"
    ".v73e_timeline_lora_es_content_free_qwen36_exact_phase.attempt.json"
).resolve()
V73E_PROFILE_DIR = (
    ROOT
    / "experiments/eggroll_es_hpo/profiles/"
    "v73e_timeline_lora_es_content_free_qwen36_exact_phase"
).resolve()
V73E_RUN_ATTEMPT = (
    ROOT
    / "experiments/eggroll_es_hpo/runs/"
    ".v73e_timeline_lora_es_content_free_qwen36_exact_phase.attempt.json"
).resolve()
V73E_RUN_DIR = (
    ROOT
    / "experiments/eggroll_es_hpo/runs/"
    "v73e_timeline_lora_es_content_free_qwen36_exact_phase"
).resolve()

V73E_ATTEMPT_1_HASHES = {
    V73E_PREREGISTRATION: (
        "cc31616b59c53da930e6d672f0f6913332b3db5c395e2568317b51366bc269c2"
    ),
    V73E_PROFILE_ATTEMPT: (
        "9f14cb7bd419d527a0ba4a2201cb63665b6de4ffa5607f20458d5a1ec67a522f"
    ),
    V73E_PROFILE_DIR / "profile_failure_v73e.json": (
        "43541b858d6cb60088f46bb065e8057db2768bc4979fd02e7064a7f7000d5d58"
    ),
    V73E_PROFILE_DIR
    / "v73e_timeline_lora_es_content_free_qwen36_exact_phase.nsys-rep": (
        "51f1c511e9dc7d308fdc973cbbb84ee9c5484201c46765a89dd18c4e2bd0b743"
    ),
    V73E_PROFILE_DIR
    / "v73e_timeline_lora_es_content_free_qwen36_exact_phase.sqlite": (
        "6dd132da295c24a2a6ab0ac994dc5aa07e791e03810aabe1df02ca38be0ca478"
    ),
    V73E_RUN_ATTEMPT: (
        "dbcb55bfa29b3ced1b8d404a61a14904610253770b5dbf6382dccf7fc9a8182f"
    ),
    V73E_RUN_DIR / "failure_v73e.json": (
        "9381ce465efa44c3d0d1686311efd91efb8ca612dde8e9ed54f0da11e5f00783"
    ),
    V73E_RUN_DIR / "exact_phase_ranges_v73e.json": (
        "73bbdba0f98c80474e697bab0eb1cdfb22f5df154624f7876a5dc5673ece0586"
    ),
}


# Reuse the sealed V73E builder implementation with only prospective globals
# rebound. Its generated-contract validator remains the runtime compatibility
# gate; V73G adds an independently self-hashed amendment below.
_base.OUTPUT = OUTPUT
_base.EVIDENCE = EVIDENCE
_base.SYSTEMS_ONLY_CLOSURE = SYSTEMS_ONLY_CLOSURE
_base.SYSTEMS_ONLY_CLOSURE_AUDITOR = SYSTEMS_ONLY_CLOSURE_AUDITOR
_base.TARGET = contract.TARGET
_base.LAUNCHER = contract.LAUNCHER
_base.PHASE_DOMAIN = contract.PHASE_DOMAIN
_base.PHASES = contract.PHASES
_base.runtime_contract = contract.BASE_RUNTIME_CONTRACT
_base.arm_artifacts_v73e = contract.arm_artifacts_v73e
_base.command_template_v73e = contract.command_template_v73e
_base.expand_command_v73e = contract.expand_command_v73e


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _source(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    _require(path.is_file() and not path.is_symlink(), f"V73G source absent: {path}")
    return {"path": str(path), "file_sha256": contract.file_sha256(path)}


def immutable_v73e_attempt_1_predecessor() -> dict[str, Any]:
    for path, expected in V73E_ATTEMPT_1_HASHES.items():
        _require(
            path.is_file()
            and not path.is_symlink()
            and contract.file_sha256(path) == expected,
            f"V73G immutable V73E predecessor changed: {path}",
        )
    return {
        "status": "immutable_failed_closed_controller_preimport_classification",
        "failure_cause": (
            "controller_side_worker_class_validation_rejected_parent_modules_"
            "already_imported_after_controller_sitecustomize_guard"
        ),
        "artifacts_must_not_be_deleted_modified_or_reused": True,
        "v73e_sealed_source_commit": (
            "be91b189117f35c934b85ca277884be12cf93d9f"
        ),
        "v73e_failure_artifact_commit": (
            "4f05f1f94aa6a6af212c65a275f1070d509e0d4b"
        ),
        "preregistration": {
            **_source(V73E_PREREGISTRATION),
            "content_sha256": (
                "e898c1984cb87b3ed83425b851cf972d3969f6d070df551fd1a0cd97b6ab94d2"
            ),
        },
        "profile_attempt": {
            **_source(V73E_PROFILE_ATTEMPT),
            "content_sha256": (
                "7d16b606066033137df10ca1182977bfe03e8bd92238b97e944d17d296227582"
            ),
        },
        "profile_failure": {
            **_source(V73E_PROFILE_DIR / "profile_failure_v73e.json"),
            "content_sha256": (
                "55ecc36187aa231d5f464a368f51d1bae677e7f38c95aeef159405c474da9bf1"
            ),
        },
        "nsys_report": _source(
            V73E_PROFILE_DIR
            / "v73e_timeline_lora_es_content_free_qwen36_exact_phase.nsys-rep"
        ),
        "sqlite_export": _source(
            V73E_PROFILE_DIR
            / "v73e_timeline_lora_es_content_free_qwen36_exact_phase.sqlite"
        ),
        "run_attempt": {
            **_source(V73E_RUN_ATTEMPT),
            "content_sha256": (
                "3f84748e1ef61449cab3268d0eaa1544d1df031600b26d7954a40c416eb0911f"
            ),
        },
        "run_failure": {
            **_source(V73E_RUN_DIR / "failure_v73e.json"),
            "content_sha256": (
                "10c216780078ad8f82c9455608920a9dd94a6c0b79f9c8d6571a186977753b37"
            ),
        },
        "partial_phase_receipt": {
            **_source(V73E_RUN_DIR / "exact_phase_ranges_v73e.json"),
            "content_sha256": (
                "db0a04f64c8d95e227d8d2027f8e1b587a59ca240e8183a8a8a31ec8b711fc5c"
            ),
        },
        "model_load_reached": False,
        "ray_actor_guard_bootstrap_reached": False,
        "partial_actor_cuda_receipt_count": 0,
        "protected_or_semantic_data_opened": False,
        "quality_or_semantic_authority": False,
        "promotion_charged_gpu_seconds": 0,
        "final_all_four_gpus_idle": True,
    }


def build_preregistration_v73g() -> dict[str, Any]:
    value = copy.deepcopy(_base.build_preregistration_v73e())
    value.pop("content_sha256_before_self_field", None)
    predecessor = immutable_v73e_attempt_1_predecessor()
    value["purpose"] = (
        "Additively supersede immutable V73E attempt 1 by distinguishing "
        "controller-side class validation under the process-start "
        "sitecustomize guard from actor-side pre-parent guard installation. "
        "Retain the exact V73E content-free workload, staged adapter, phases, "
        "failure semantics, and postrun acceptance on fresh V73G paths."
    )
    value["immutable_v73e_attempt_1_predecessor"] = predecessor
    value["v73g_successor_amendment"] = {
        "schema": "qwen36-v73g-controller-bootstrap-repair-v1",
        "status": "prospectively_sealed_additive_successor",
        "only_runtime_logic_change": (
            "controller_branch_allows_parent_modules_imported_after_the_"
            "controller_sitecustomize_guard"
        ),
        "actor_branch_still_requires_parent_and_historical_modules_absent": True,
        "controller_guard_mechanism_pid_and_non_actor_environment_required": True,
        "fresh_artifact_paths_required": True,
        "v73e_attempt_1_reuse_or_overwrite_authorized": False,
        "quality_hpo_or_promotion_authorized": False,
        "protected_dev_ood_or_holdout_opened": False,
        "bindings": {
            "successor_builder": _source(Path(__file__).resolve()),
            "successor_contract": _source(Path(contract.__file__).resolve()),
            "successor_target": _source(contract.TARGET),
            "successor_launcher": _source(contract.LAUNCHER),
            "successor_worker": _source(contract.WORKER),
            "successor_closure_auditor": _source(
                SYSTEMS_ONLY_CLOSURE_AUDITOR
            ),
            "successor_regression_tests": _source(
                ROOT / "test_qwen36_v73g_controller_bootstrap.py"
            ),
        },
    }
    value["content_sha256_before_self_field"] = contract.canonical_sha256(value)
    contract.validate_generated_preregistration_v73e(value)
    return value


def render_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def render_evidence_v73g(
    value: Mapping[str, Any], output: Path, payload: str
) -> str:
    base = _base.render_evidence_v73e(value, output, payload).rstrip()
    file_sha = hashlib.sha256(payload.encode("ascii")).hexdigest()
    launch = [
        str(contract.REQUIRED_PYTHON),
        str(contract.LAUNCHER),
        "--preregistration",
        str(output),
        "--preregistration-sha256",
        file_sha,
        "--preregistration-content-sha256",
        value["content_sha256_before_self_field"],
        "--arm",
        "timeline",
        "--execute",
    ]
    predecessor = value["immutable_v73e_attempt_1_predecessor"]
    return base + "\n\n" + "\n".join([
        "## V73G additive successor amendment",
        "",
        "- V73E attempt 1 remains immutable; run/profile failure SHA-256: "
        f"`{predecessor['run_failure']['file_sha256']}` / "
        f"`{predecessor['profile_failure']['file_sha256']}`.",
        "- The controller may observe parent workers imported after its "
        "process-start sitecustomize guard; actors still reject any parent "
        "import before their actor-specific guard.",
        "- V73G uses fresh run/profile paths and retains zero quality, semantic, "
        "checkpoint, or promotion authority.",
        f"- Preregistration file SHA-256: `{file_sha}`.",
        "",
        "```bash",
        shlex.join(launch),
        "```",
        "",
    ])


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload.encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--evidence", default=str(EVIDENCE))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    evidence = Path(args.evidence).resolve()
    _require(output == OUTPUT, "V73G preregistration path must remain canonical")
    _require(evidence == EVIDENCE, "V73G evidence path must remain canonical")
    value = build_preregistration_v73g()
    payload = render_json(value)
    evidence_payload = render_evidence_v73g(value, output, payload)
    if args.check:
        _require(output.is_file(), f"V73G preregistration absent: {output}")
        _require(output.read_text(encoding="ascii") == payload, "V73G preregistration stale")
        _require(evidence.is_file(), f"V73G evidence absent: {evidence}")
        _require(evidence.read_text(encoding="ascii") == evidence_payload, "V73G evidence stale")
    else:
        if output.exists() or evidence.exists():
            raise FileExistsError(output if output.exists() else evidence)
        _atomic_write(output, payload)
        _atomic_write(evidence, evidence_payload)
    print(json.dumps({
        "path": str(output),
        "file_sha256": contract.file_sha256(output) if output.is_file() else None,
        "content_sha256": value["content_sha256_before_self_field"],
        "evidence": str(evidence),
        "evidence_file_sha256": (
            contract.file_sha256(evidence) if evidence.is_file() else None
        ),
        "timeline_launch_sealed": True,
        "hbm_arm_blocked": True,
        "checked": bool(args.check),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
