#!/usr/bin/env python3
"""Seal V73H's process-start Ray actor guard and fresh profile paths."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
from pathlib import Path
from typing import Any, Mapping

import qwen36_v73h_exact_phase_profiler_contract as contract
import build_qwen36_v73e_exact_phase_profiler_preregistration as _base


ROOT = Path(__file__).resolve().parent
OUTPUT = contract.OUTPUT
EVIDENCE = (
    ROOT
    / "experiments/eggroll_es_hpo/"
    "qwen36_v73h_exact_phase_profiler_cpu_evidence_20260717.md"
).resolve()
SYSTEMS_ONLY_CLOSURE = (
    ROOT
    / "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_v73h_systems_only_closure.json"
).resolve()
SYSTEMS_ONLY_CLOSURE_AUDITOR = (
    ROOT / "audit_v73h_systems_only_import_graph_v1.py"
).resolve()

V73G_PREREGISTRATION = (
    ROOT
    / "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_v73g_exact_phase_profiler.json"
).resolve()
V73G_PROFILE_ATTEMPT = (
    ROOT
    / "experiments/eggroll_es_hpo/profiles/"
    ".v73g_timeline_lora_es_content_free_qwen36_exact_phase.attempt.json"
).resolve()
V73G_PROFILE_DIR = (
    ROOT
    / "experiments/eggroll_es_hpo/profiles/"
    "v73g_timeline_lora_es_content_free_qwen36_exact_phase"
).resolve()
V73G_RUN_ATTEMPT = (
    ROOT
    / "experiments/eggroll_es_hpo/runs/"
    ".v73g_timeline_lora_es_content_free_qwen36_exact_phase.attempt.json"
).resolve()
V73G_RUN_DIR = (
    ROOT
    / "experiments/eggroll_es_hpo/runs/"
    "v73g_timeline_lora_es_content_free_qwen36_exact_phase"
).resolve()

V73G_ATTEMPT_1_HASHES = {
    V73G_PREREGISTRATION: (
        "7cf5f02dfe63383d555e9a2a6b4a69c64a2b5c9dd316f5182e4a1c0d39a7bc86"
    ),
    V73G_PROFILE_ATTEMPT: (
        "463332fe19264c2d302588fa590003d72aa66511d8fc10391bcd5e5bb942a580"
    ),
    V73G_PROFILE_DIR / "profile_failure_v73g.json": (
        "95e16d339848bc9dcc8bed6435c968c09d5709d3656bc6face77f23535319997"
    ),
    V73G_PROFILE_DIR
    / "v73g_timeline_lora_es_content_free_qwen36_exact_phase.nsys-rep": (
        "fdc747f4f0a41990abb25fb8caabe1418b481ca06462e79f5600cc733b7a8dbb"
    ),
    V73G_PROFILE_DIR
    / "v73g_timeline_lora_es_content_free_qwen36_exact_phase.sqlite": (
        "76a9999f809c5b9e55a9b8461fae0c02fd1a1f4a6c823babb6bba567cdaa1deb"
    ),
    V73G_RUN_ATTEMPT: (
        "d94194d377e858e3a74bbc1fc425b436a07920d199fdfa6dea68b5b1acc9b9c5"
    ),
    V73G_RUN_DIR / "failure_v73g.json": (
        "6904e092142e85dc3f8d9f090fb08dbc06a1d057d1b45c86cebc97ff6f0ff860"
    ),
    V73G_RUN_DIR / "exact_phase_ranges_v73g.json": (
        "17cfb121e92027a8de0028782a32c818043b31524d156d0308115069d6704d44"
    ),
}


# Reuse the sealed V73E builder implementation with only prospective globals
# rebound. Its generated-contract validator remains the runtime compatibility
# gate; V73H adds an independently self-hashed amendment below.
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
    _require(path.is_file() and not path.is_symlink(), f"V73H source absent: {path}")
    return {"path": str(path), "file_sha256": contract.file_sha256(path)}


def immutable_v73g_attempt_1_predecessor() -> dict[str, Any]:
    for path, expected in V73G_ATTEMPT_1_HASHES.items():
        _require(
            path.is_file()
            and not path.is_symlink()
            and contract.file_sha256(path) == expected,
            f"V73H immutable V73G predecessor changed: {path}",
        )
    return {
        "status": "immutable_failed_closed_real_ray_actor_preload_before_worker_guard",
        "failure_cause": (
            "real_ray_vllm_actor_preloaded_application_and_vllm_modules_before_"
            "resolving_the_worker_extension_guard"
        ),
        "artifacts_must_not_be_deleted_modified_or_reused": True,
        "v73g_sealed_source_commit": (
            "80f329a9547f4ba86c5c4b5d9f79fafef72a96f6"
        ),
        "v73g_failure_artifact_commit": (
            "d8a37b1b7a31d5ed014e09cc8073a0a478b7eb47"
        ),
        "preregistration": {
            **_source(V73G_PREREGISTRATION),
            "content_sha256": (
                "94050a7d96d537dcf1cf49f9531125009f977d885b3ec884c5cd0940bed15d0d"
            ),
        },
        "profile_attempt": {
            **_source(V73G_PROFILE_ATTEMPT),
            "content_sha256": (
                "8b06f92a3069824099164aafec8b7e9076e3d2e3c8a99cbd6d1423f83d1bfc89"
            ),
        },
        "profile_failure": {
            **_source(V73G_PROFILE_DIR / "profile_failure_v73g.json"),
            "content_sha256": (
                "614994ae13a02dba1fd8423c4372b43f051598290c89d1ed3ed306475f06a885"
            ),
        },
        "nsys_report": _source(
            V73G_PROFILE_DIR
            / "v73g_timeline_lora_es_content_free_qwen36_exact_phase.nsys-rep"
        ),
        "sqlite_export": _source(
            V73G_PROFILE_DIR
            / "v73g_timeline_lora_es_content_free_qwen36_exact_phase.sqlite"
        ),
        "run_attempt": {
            **_source(V73G_RUN_ATTEMPT),
            "content_sha256": (
                "baafa0d03c10da9c8c2959365b51ee0d59ac9bba1dc5e2af5dd2abfda7610dae"
            ),
        },
        "run_failure": {
            **_source(V73G_RUN_DIR / "failure_v73g.json"),
            "content_sha256": (
                "988892befc4925ec6a1dd5c9640e39aba0f1e79e4579ce5627be470ea8ba4e18"
            ),
        },
        "partial_phase_receipt": {
            **_source(V73G_RUN_DIR / "exact_phase_ranges_v73g.json"),
            "content_sha256": (
                "3def879e6cf3d251f65d425707e69bf5b630b7a60c17ede7e882bc983c3f7f3a"
            ),
        },
        "model_load_reached": False,
        "ray_actors_launched": 4,
        "ray_actor_guard_bootstrap_reached": True,
        "partial_actor_cuda_receipt_count": 0,
        "protected_or_semantic_data_opened": False,
        "quality_or_semantic_authority": False,
        "promotion_charged_gpu_seconds": 0,
        "final_all_four_gpus_idle": True,
    }


def build_preregistration_v73h() -> dict[str, Any]:
    value = copy.deepcopy(_base.build_preregistration_v73e())
    value.pop("content_sha256_before_self_field", None)
    predecessor = immutable_v73g_attempt_1_predecessor()
    value["purpose"] = (
        "Additively supersede immutable V73G attempt 1 by installing the Ray "
        "actor path guard from actor sitecustomize before application and vLLM "
        "runtime imports. Retain the exact content-free workload, staged "
        "adapter, phases, failure semantics, and postrun acceptance on fresh "
        "V73H paths."
    )
    value["immutable_v73g_attempt_1_predecessor"] = predecessor
    bootstrap = value["ray_actor_guard_bootstrap"]
    bootstrap["actor_mechanism"] = contract.ACTOR_GUARD_MECHANISM
    bootstrap[
        "actor_guard_installed_by_sitecustomize_before_runtime_imports"
    ] = True
    bootstrap["worker_extension_resolved_after_process_start_guard"] = True
    bootstrap["actor_guard_install_deferred_until_worker_extension"] = False
    value["v73h_successor_amendment"] = {
        "schema": "qwen36-v73h-process-start-actor-guard-amendment-v1",
        "status": "prospectively_sealed_additive_successor",
        "only_runtime_logic_change": (
            "actor_sitecustomize_installs_the_exact_actor_guard_before_"
            "application_and_vllm_runtime_imports"
        ),
        "actor_sitecustomize_requires_exact_flag_hash_and_inherited_controller_pid": True,
        "parent_and_historical_modules_imported_after_actor_guard_are_covered": True,
        "worker_extension_validates_preinstalled_actor_guard_receipt": True,
        "controller_and_actor_guard_mechanisms_remain_distinct": True,
        "fresh_artifact_paths_required": True,
        "v73g_attempt_1_reuse_or_overwrite_authorized": False,
        "quality_hpo_or_promotion_authorized": False,
        "protected_dev_ood_or_holdout_opened": False,
        "bindings": {
            "successor_builder": _source(Path(__file__).resolve()),
            "successor_contract": _source(Path(contract.__file__).resolve()),
            "successor_target": _source(contract.TARGET),
            "successor_launcher": _source(contract.LAUNCHER),
            "successor_worker": _source(contract.WORKER),
            "successor_guard": _source(contract.GUARD),
            "successor_sitecustomize": _source(contract.SITECUSTOMIZE),
            "successor_closure_auditor": _source(
                SYSTEMS_ONLY_CLOSURE_AUDITOR
            ),
            "successor_regression_tests": _source(
                ROOT / "test_qwen36_v73h_process_start_actor_guard.py"
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


def render_evidence_v73h(
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
    predecessor = value["immutable_v73g_attempt_1_predecessor"]
    return base + "\n\n" + "\n".join([
        "## V73H additive successor amendment",
        "",
        "- V73G attempt 1 remains immutable; run/profile failure SHA-256: "
        f"`{predecessor['run_failure']['file_sha256']}` / "
        f"`{predecessor['profile_failure']['file_sha256']}`.",
        "- Ray actors install their distinct path guard from sitecustomize "
        "before application or vLLM runtime imports; the later worker extension "
        "validates that exact process-start receipt.",
        "- V73H uses fresh run/profile paths and retains zero quality, semantic, "
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
    _require(output == OUTPUT, "V73H preregistration path must remain canonical")
    _require(evidence == EVIDENCE, "V73H evidence path must remain canonical")
    value = build_preregistration_v73h()
    payload = render_json(value)
    evidence_payload = render_evidence_v73h(value, output, payload)
    if args.check:
        _require(output.is_file(), f"V73H preregistration absent: {output}")
        _require(output.read_text(encoding="ascii") == payload, "V73H preregistration stale")
        _require(evidence.is_file(), f"V73H evidence absent: {evidence}")
        _require(evidence.read_text(encoding="ascii") == evidence_payload, "V73H evidence stale")
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
